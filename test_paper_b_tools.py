"""Paper B examples and generalized calculus/statistics cases."""

import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np
import sympy as sp

from DataPreparation import descriptive_statistics
from FunctionUtils import check_minimum
from MatrixCalculus import differentiate_matrix
from ProbabilityUtils import compare_grouped_rates, sampling_without_replacement


class MinimumTests(unittest.TestCase):
    def test_paper_b_minimum_and_value(self):
        result = check_minimum('(x-3)^2+2', [3])
        self.assertTrue(result['is_minimum'])
        self.assertTrue(result['global_minimum'])
        self.assertEqual(result['value'], 2)
        self.assertEqual(result['classification'], 'Strict global minimum')

    def test_zero_gradient_does_not_prove_minimum(self):
        result = check_minimum('x^3', [0])
        self.assertFalse(result['is_minimum'])
        self.assertEqual(result['classification'], 'Neither minimum nor maximum')
        self.assertEqual(result['higher_derivative_order'], 3)

    def test_higher_order_minimum_and_maximum(self):
        self.assertTrue(check_minimum('x^4', [0])['is_minimum'])
        self.assertEqual(check_minimum('-x^6', [0])['classification'], 'Strict local maximum')

    def test_nonstationary_and_tiny_nonzero_derivative(self):
        self.assertFalse(check_minimum('(x-3)^2', [2])['is_minimum'])
        self.assertFalse(check_minimum('x^2 + x/10^20', [0])['is_minimum'])

    def test_multivariable_minimum_maximum_saddle(self):
        self.assertTrue(check_minimum('x^2+2*y^2', [0, 0])['global_minimum'])
        self.assertEqual(check_minimum('-x^2-y^2', [0, 0])['classification'], 'Strict local maximum')
        self.assertEqual(check_minimum('x^2-y^2', [0, 0])['classification'], 'Saddle point')

    def test_semidefinite_quadratic_and_higher_order_inconclusive(self):
        result = check_minimum('x^2', [0, 4], ['x', 'y'])
        self.assertTrue(result['global_minimum'])
        self.assertEqual(result['classification'], 'Global minimum (not strict)')
        self.assertIsNone(check_minimum('x^2-y^4', [0, 0])['is_minimum'])

    def test_fraction_and_local_minimum(self):
        result = check_minimum('(x-1/3)^4', ['1/3'])
        self.assertTrue(result['is_minimum'])
        self.assertFalse(result['global_minimum'])
        self.assertEqual(check_minimum('-cos(x)', [0])['classification'], 'Strict local minimum')
        self.assertEqual(check_minimum('sin(x)', ['-pi/2'])['classification'], 'Strict local minimum')

    def test_nondifferentiability_domain_and_invalid_inputs(self):
        self.assertIsNone(check_minimum('Abs(x)', [0])['is_minimum'])
        self.assertIsNone(check_minimum('sqrt(x)', [0])['is_minimum'])
        for raw, point in [('x;y', [0, 0]), ('x+y', [0]), ('log(x)', [-1])]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                check_minimum(raw, point)


class MatrixCalculusTests(unittest.TestCase):
    def test_paper_b_nonsymmetric_quadratic(self):
        result = differentiate_matrix("x'*A*x", 'A = [1 2;0 3]; syms x [2 1] real', 'x', 'x = [1;1]')
        np.testing.assert_allclose(result['derivative_at_point'], [[4], [8]])
        np.testing.assert_allclose(result['hessian_at_point'], [[2, 2], [2, 6]])
        np.testing.assert_allclose(result['value_at_point'], [[6]])

    def test_symbolic_quadratic_formula(self):
        result = differentiate_matrix("x.'*A*x", 'syms A [2 2] real; syms x [2 1] real', 'x')
        x1, x2, a11, a12, a21, a22 = sp.symbols('x1 x2 A_1_1 A_1_2 A_2_1 A_2_2', real=True)
        expected = sp.Matrix([2*a11*x1+(a12+a21)*x2, (a12+a21)*x1+2*a22*x2])
        self.assertEqual((result['gradient']-expected).applyfunc(sp.simplify), sp.zeros(2, 1))
        self.assertIsNone(result['derivative_at_point'])

    def test_least_squares_ridge_norm_and_trace(self):
        definitions = 'X = [1 0;1 1;1 2]; y=[1;2;2]; lambda=2; syms w [2 1] real'
        norm = differentiate_matrix("norm(X*w-y)^2 + lambda*(w'*w)", definitions, 'w', 'w=[1;1]')
        summed = differentiate_matrix("sum((X*w-y).^2) + lambda*sum(w.^2)", definitions, 'w', 'w=[1;1]')
        self.assertEqual(norm['gradient'], summed['gradient'])
        np.testing.assert_allclose(norm['derivative_at_point'], [[6], [8]])
        result = differentiate_matrix("trace(W'*W)", 'syms W [2 2] real', 'W', 'W=[1 -2;3 4]')
        np.testing.assert_allclose(result['derivative_at_point'], [[2, -4], [6, 8]])
        np.testing.assert_allclose(result['hessian_at_point'], 2*np.eye(4))

    def test_vector_jacobian_column_major(self):
        result = differentiate_matrix('A*x', 'A=[1 2;3 4]; syms x [2 1] real', 'x')
        self.assertEqual(result['jacobian'], sp.Matrix([[1, 2], [3, 4]]))
        result = differentiate_matrix("W'", 'syms W [2 2] real', 'W')
        self.assertEqual(result['variables'], ['W_1_1', 'W_2_1', 'W_1_2', 'W_2_2'])
        self.assertEqual(result['jacobian'], sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))

    def test_matlab_literals_functions_indexing_and_precedence(self):
        result = differentiate_matrix('f = -x^2 + x(1)*2^3;', 'syms x [1 1] real', 'x', 'x=[2]')
        np.testing.assert_allclose(result['derivative_at_point'], [[4]])
        result = differentiate_matrix("norm(W,'fro')^2", 'syms W [2 2] real', 'W', 'W=[1 -2;3 4]')
        np.testing.assert_allclose(result['value_at_point'], [[30]])
        result = differentiate_matrix("sum(W.^2,'all')", 'syms W [2 2] real', 'W')
        self.assertEqual(result['gradient'], 2*sp.Matrix(2, 2, lambda i, j: sp.Symbol(f'W_{i+1}_{j+1}', real=True)))

    def test_matrix_operations_and_multiple_numeric_assignments(self):
        result = differentiate_matrix('A\\x', 'A=[2 0;0 4]; syms x [2 1] real', 'x', 'x=[2;4]')
        np.testing.assert_allclose(result['value_at_point'], [[1], [1]])
        np.testing.assert_allclose(result['derivative_at_point'], [[.5, 0], [0, .25]])
        result = differentiate_matrix('a*x^2', 'syms a x real', 'x', 'a=3;x=2')
        np.testing.assert_allclose(result['derivative_at_point'], [[12]])

    def test_matlab_power_associativity_and_transpose_precedence(self):
        result = differentiate_matrix('x^2^3', 'syms x real', 'x', 'x=2')
        np.testing.assert_allclose(result['derivative_at_point'], [[192]])
        result = differentiate_matrix("A.^B'", 'A=[1 2;3 4]; B=[1 2;3 4];syms x real', 'x')
        np.testing.assert_allclose(result['value_at_point'], [[1, 27], [4, 256]])

    def test_matrix_dimension_syntax_and_target_validation(self):
        examples = [('A*x', 'A=[1 2 3];syms x [2 1] real', 'x'),
                    ('x*x', 'syms x [2 1] real', 'x'),
                    ("norm(A)", 'syms A [2 2] real', 'A'),
                    ('x^2', 'x=[1;2]', 'x'),
                    ('system(1)', 'syms x real', 'x'),
                    ('x.__class__', 'syms x real', 'x')]
        for raw, definitions, target in examples:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                differentiate_matrix(raw, definitions, target)
        with self.assertRaises(ValueError):
            differentiate_matrix('x^2', 'syms x [2 1] real', 'x', 'x=[1 2]')

    def test_multioutput_ridge_gradient_matches_matrix_identity(self):
        X = np.array([[1., 2.], [0., 1.], [2., -1.]])
        Y = np.array([[2., 0.], [1., 3.], [0., 1.]])
        W = np.array([[1., -1.], [2., 0.]])
        result = differentiate_matrix("trace((X*W-Y)'*(X*W-Y))+lambda*trace(W'*W)",
                                      'X=[1 2;0 1;2 -1];Y=[2 0;1 3;0 1];lambda=2;syms W [2 2] real',
                                      'W', 'W=[1 -1;2 0]')
        expected = 2*X.T@(X@W-Y)+4*W
        np.testing.assert_allclose(result['derivative_at_point'], expected)
        norm_result = differentiate_matrix("norm(X*W-Y,'fro')^2+lambda*sum(W.^2,'all')",
                                           'X=[1 2;0 1;2 -1];Y=[2 0;1 3;0 1];lambda=2;syms W [2 2] real',
                                           'W', 'W=[1 -1;2 0]')
        np.testing.assert_allclose(norm_result['derivative_at_point'], expected)


class PaperBTerminalTests(unittest.TestCase):
    def test_new_shortcuts(self):
        import main
        scenarios = [(['minimum', '(x-3)^2+2', '', '3'], [[2]]),
                     (['matdiff', "x'*A*x", 'A=[1 2;0 3];syms x [2 1] real', 'x', 'x=[1;1]'], [[4], [8]]),
                     (['stats', '1;2;3;4;8;9;10;12', ''], [[1, 2.5, 6, 9.5, 12, 7]]),
                     (['probability', 'sampling', '6', '2', '2', '>=', '1'], [[.6]]),
                     (['probability', 'rates', '9 10 80 100;20 100 1 10'], [[29/110, 81/110]])]
        for inputs, expected in scenarios:
            with self.subTest(shortcut=inputs[0]), patch('builtins.input', side_effect=inputs+['']), patch.object(main, 'clear'), contextlib.redirect_stdout(io.StringIO()):
                main.process_input()
                np.testing.assert_allclose(main.LAST_RESULT, expected)


class PaperBStatisticsTests(unittest.TestCase):
    def test_paper_b_quartiles(self):
        result = descriptive_statistics([1, 2, 3, 4, 8, 9, 10, 12])
        self.assertEqual(result['q3'][0], 9.5)
        self.assertEqual(result['q1'][0], 2.5)
        self.assertEqual(result['median'][0], 6)
        self.assertEqual(result['iqr'][0], 7)

    def test_quartile_conventions_columns_and_small_samples(self):
        self.assertEqual(descriptive_statistics([1, 2, 3, 4, 5])['q3'][0], 4.5)
        self.assertEqual(descriptive_statistics([1, 2, 3, 4, 5], 'Linear interpolation')['q3'][0], 4)
        self.assertEqual(descriptive_statistics([5])['q1'][0], 5)
        np.testing.assert_allclose(descriptive_statistics([[1, 10], [3, 20]])['q3'], [3, 20])

    def test_paper_b_drawing_without_replacement(self):
        result = sampling_without_replacement(6, 2, 2, '>=', 1)
        self.assertEqual(result['exact_probability'], sp.Rational(3, 5))
        self.assertAlmostEqual(result['event_probability'], .6)

    def test_sampling_generalized_and_edge_cases(self):
        result = sampling_without_replacement(10, 4, 3, 'between', 1, 2)
        self.assertEqual(result['exact_probability'], sp.Rational(4, 5))
        self.assertEqual(sampling_without_replacement(6, 0, 0, '=', 0)['event_probability'], 1)
        self.assertEqual(sampling_without_replacement(6, 6, 6, '>=', 1)['event_probability'], 1)
        for values in [(5, 6, 1), (5, 2, 6), (0, 0, 0), (6, 2.5, 2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                sampling_without_replacement(*values)

    def test_paper_b_grouped_rates_reversal(self):
        result = compare_grouped_rates([[9, 10, 80, 100], [20, 100, 1, 10]])
        np.testing.assert_allclose(result['rates'], [[.9, .8], [.2, .1]])
        np.testing.assert_allclose(result['pooled'], [29/110, 81/110])
        self.assertTrue(result['reversal'])

    def test_rates_weighted_by_counts_and_invalid_counts(self):
        result = compare_grouped_rates([[1, 2, 1, 2], [9, 10, 1, 10]])
        self.assertFalse(result['reversal'])
        np.testing.assert_allclose(result['pooled'], np.sum(result['weights']*result['rates'], axis=0))
        for table in [[[2, 1, 1, 2]], [[0, 0, 1, 2]], [[.5, 2, 1, 2]], [[1, 2, 3]]]:
            with self.subTest(table=table), self.assertRaises(ValueError):
                compare_grouped_rates(table)


if __name__ == '__main__':
    unittest.main(verbosity=2)
