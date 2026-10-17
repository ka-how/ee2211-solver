"""Generalized calculators and regression checks using Paper A examples."""

import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np

from CommonUtils import add_bias_column, parse_number, parse_numeric_array
from DataPreparation import matrix_calculation, prepare_data
from FunctionUtils import analyze_functions
from ProbabilityUtils import bayes_update, density_probability, discrete_distribution, event_probabilities
from RegressionUtils import polynomial_features_details, regression_diagnostics, regression_shapes, solve_normal_equations
from webui import fit_regression_model


class GeneralToolsTests(unittest.TestCase):
    def test_terminal_probability_and_summary_shortcuts(self):
        import main

        stdout = io.StringIO()
        with patch("builtins.input", side_effect=["probability", "discrete", "-2 1/4;0 ?;3 1/2", "", "", ""]), patch.object(main, "clear"), contextlib.redirect_stdout(stdout):
            self.assertTrue(main.process_input())
        self.assertIn("event_probability: 0.75", stdout.getvalue())
        stdout = io.StringIO()
        with patch("builtins.input", side_effect=["normal", "4 0;0 9", "8;18", "1", "", ""]), patch.object(main, "clear"), contextlib.redirect_stdout(stdout):
            self.assertTrue(main.process_input())
        np.testing.assert_allclose(main.LAST_RESULT, [[1.6], [1.8]])

    def test_terminal_function_and_imputation_shortcuts(self):
        import main

        with patch("builtins.input", side_effect=["functions", "1-2*x1+x1*x2", "", "2,1", "sign", ""]), patch.object(main, "clear"), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(main.process_input())
        np.testing.assert_allclose(main.LAST_RESULT, [[-1]])
        with patch("builtins.input", side_effect=["prepare", "mean imputation", "2;4;12;?", "", ""]), patch.object(main, "clear"), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(main.process_input())
        np.testing.assert_allclose(main.LAST_RESULT, [[2], [4], [12], [6]])

    def test_fraction_input_all_matrix_formats(self):
        for raw in ("1/3 2/3;-1/2 3/4", "1/3,2/3\n-1/2,3/4", "[[1/3,2/3],[-1/2,3/4]]"):
            np.testing.assert_allclose(parse_numeric_array(raw), [[1/3, 2/3], [-1/2, 3/4]])
        self.assertEqual(parse_number("1e-3"), .001)
        for raw in ("1/0", "nan", "inf", "1+2", "__import__('os')"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_number(raw)

    def test_missing_inputs_are_allowed_only_when_requested(self):
        for raw in ("2;4;12;?", "[[2],[4],[12],[NA]]", "[2,4,12,None]"):
            values = parse_numeric_array(raw, allow_missing=True)
            self.assertTrue(np.isnan(values[-1, 0]))
            with self.assertRaises(ValueError):
                parse_numeric_array(raw)
        with self.assertRaises(ValueError):
            parse_numeric_array("1;1e309", allow_missing=True)

    def test_paper_a_minmax_and_reverse(self):
        result = prepare_data([[18]], "Min-max scaling", first=10, second=30)
        np.testing.assert_allclose(result["result"], [[.4]])
        inverse = prepare_data([[.4]], "Min-max scaling", first=10, second=30, inverse=True)
        np.testing.assert_allclose(inverse["result"], [[18]])

    def test_reference_fits_columns_and_does_not_clip_new_values(self):
        result = prepare_data([[15, 250]], "Min-max scaling", reference=[[10, 100], [20, 200]], output_range=(-1, 1))
        np.testing.assert_allclose(result["result"], [[0, 2]])
        inverse = prepare_data([[0, 2]], "Min-max scaling", reference=[[10, 100], [20, 200]], output_range=(-1, 1), inverse=True)
        np.testing.assert_allclose(inverse["result"], [[15, 250]])

    def test_paper_a_mean_imputation_observed_only(self):
        result = prepare_data([[2], [4], [12], [np.nan]], "Mean imputation")
        np.testing.assert_allclose(result["result"], [[2], [4], [12], [6]])
        self.assertEqual(result["observed"][0], 3)

    def test_sentinel_and_reference_imputation(self):
        result = prepare_data([[-1, 0], [8, -1]], "Mean imputation", reference=[[2, 10], [4, 20], [-1, 30]], missing_marker=-1)
        np.testing.assert_allclose(result["result"], [[3, 0], [8, 20]])

    def test_zscores_population_and_known_parameters(self):
        result = prepare_data([[17]], "Z-score standardization", first=11, second=3)
        np.testing.assert_allclose(result["result"], [[2]])
        inverse = prepare_data([[2]], "Z-score standardization", first=11, second=3, inverse=True)
        np.testing.assert_allclose(inverse["result"], [[17]])
        result = prepare_data([[1], [2], [3]], "Z-score standardization")
        self.assertAlmostEqual(np.var(result["result"]), 1)

    def test_bad_preparation_inputs(self):
        cases = [([[1], [1]], "Min-max scaling", {}), ([[np.nan]], "Mean imputation", {}),
                 ([[np.nan], [1]], "Z-score standardization", {}), ([[1]], "Min-max scaling", {"first": 0}),
                 ([[1]], "Min-max scaling", {"first": 0, "second": 1, "output_range": (1, 0)}),
                 ([[1]], "Z-score standardization", {"first": 0, "second": 0}),
                 ([[1]], "Min-max scaling", {"inverse": True}),
                 ([[1]], "Mean imputation", {"inverse": True}),
                 ([[1]], "Mean imputation", {"reference": [[1, 2]]})]
        for data, method, options in cases:
            with self.subTest(method=method, options=options), self.assertRaises(ValueError):
                prepare_data(data, method, **options)

    def test_paper_a_dot_product_both_orientations(self):
        np.testing.assert_allclose(matrix_calculation([[2], [-1], [3]], "Dot product", [[0, 4, 1]]), [[-1]])

    def test_matrix_operations_and_dimensions(self):
        A, B = [[1, 2], [3, 4]], [[0, 1], [1, 0]]
        np.testing.assert_allclose(matrix_calculation(A, "Matrix product", B), [[2, 1], [4, 3]])
        np.testing.assert_allclose(matrix_calculation(A, "Trace"), [[5]])
        np.testing.assert_allclose(matrix_calculation(A, "Add", B), [[1, 3], [4, 4]])
        np.testing.assert_allclose(matrix_calculation(A, "Subtract", B), [[1, 1], [2, 4]])
        np.testing.assert_allclose(matrix_calculation(A, "Elementwise product", B), [[0, 2], [3, 0]])
        for operation in ("Matrix product", "Dot product", "Add"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                matrix_calculation([[1, 2]], operation, [[1, 2, 3]])

    def test_paper_a_unknown_probability_and_query(self):
        details = discrete_distribution([-2, 0, 3], [.25, np.nan, .5])
        np.testing.assert_allclose(details["probabilities"], [.25, .25, .5])
        self.assertEqual(details["event_probability"], .75)
        self.assertEqual(details["mean"], 1)
        self.assertEqual(details["variance"], 4.5)

    def test_distribution_queries_and_duplicate_outcomes(self):
        for comparison, expected in ((">", .5), (">=", 1), ("<", 0), ("<=", .5), ("=", .5), ("between", 1)):
            result = discrete_distribution([0, 0, 1], [.2, .3, .5], comparison, 0, 1)
            self.assertEqual(result["event_probability"], expected)
            np.testing.assert_allclose(result["cdf"], [.5, 1])

    def test_invalid_probability_tables(self):
        for probabilities in ([.2, .3], [-.2, 1.2], [np.nan, np.nan], [.7, .6, np.nan], [np.inf, 0]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                discrete_distribution(range(len(probabilities)), probabilities)

    def test_paper_a_bayes_supplier_example(self):
        details = bayes_update([.3, .7], [.1, .04])
        self.assertAlmostEqual(details["evidence"], .058)
        np.testing.assert_allclose(details["posterior"], [15/29, 14/29])

    def test_bayes_complement_unknown_prior_and_zero_event(self):
        details = bayes_update([.3, np.nan], [.1, .04], False)
        np.testing.assert_allclose(details["posterior"], np.array([.27, .672]) / .942)
        with self.assertRaises(ValueError):
            bayes_update([.3, .7], [0, 0])
        with self.assertRaises(ValueError):
            bayes_update([.3, .7], [.1, 1.1])
        details = bayes_update([.1, .2, np.nan], [.2, .5, .9])
        np.testing.assert_allclose(details["posterior"], np.array([.02, .1, .63]) / .75)

    def test_event_independence_exclusivity_and_union(self):
        details = event_probabilities(.3, .7, 0)
        self.assertFalse(details["independent"])
        self.assertTrue(details["mutually_exclusive"])
        details = event_probabilities(.5, .4, .2)
        self.assertTrue(details["independent"])
        self.assertFalse(details["mutually_exclusive"])
        details = event_probabilities(.4, .6, .8, "union")
        self.assertAlmostEqual(details["p_a_given_b"], 1/3)
        self.assertAlmostEqual(np.sum(details["joint_table"]), 1)
        self.assertIsNone(event_probabilities(0, 0, 0)["p_a_given_b"])
        with self.assertRaises(ValueError):
            event_probabilities(.8, .8, .1)

    def test_continuous_density_and_single_point_probability(self):
        self.assertAlmostEqual(density_probability("2*x", 0, 1, .5, 1)["probability"], .75)
        self.assertAlmostEqual(density_probability("2", 0, .5, 0, .25)["probability"], .5)
        self.assertEqual(density_probability("1/4", 2, 6, 3, 3)["probability"], 0)
        self.assertEqual(density_probability("1/4", 2, 6, 0, 8)["probability"], 1)

    def test_invalid_density_does_not_produce_probabilities(self):
        for density in ("4*x-1", "3*x", "1/x", "x+y"):
            with self.subTest(density=density), self.assertRaises(ValueError):
                density_probability(density, 0, 1, 0, .5)

    def test_paper_a_gradient_and_hessian(self):
        details = analyze_functions("x1**2+3*x1*x2+2*x2**2", [1, -1])
        np.testing.assert_allclose(details["jacobian_at_point"].T, [[-1], [-1]])
        np.testing.assert_allclose(details["hessian_at_point"], [[2, 3], [3, 4]])

    def test_vector_jacobian_preserves_unused_input(self):
        details = analyze_functions("x1**2+x2; x1*x2", [2, -1, 7], variables=["x1", "x2", "x3"])
        self.assertEqual(details["jacobian_at_point"].shape, (2, 3))
        np.testing.assert_allclose(details["jacobian_at_point"], [[4, 1, 0], [-1, 2, 0]])

    def test_function_linear_affine_nonlinear_and_constant(self):
        for raw, expected in (("4*x1-2*x2", "Strictly linear"), ("4*x1-2*x2+1", "Affine with offset"), ("x1*x2", "Nonlinear")):
            self.assertEqual(analyze_functions(raw, [1, 2])["dependence"], [expected])
        self.assertEqual(analyze_functions("5", [])["value"].item(), 5)

    def test_paper_a_supplied_score_evaluation(self):
        details = analyze_functions("1-2*x1+x1*x2", [2, 1])
        self.assertEqual(details["value"].item(), -1)

    def test_function_domain_and_coordinate_errors(self):
        for raw, point, variables in (("log(x)", [-1], None), ("1/x", [0], None), ("x+y", [1], None), ("x", [1], ["y"]), ("x", [1, 2], ["x", "x"])):
            with self.subTest(raw=raw, variables=variables), self.assertRaises(ValueError):
                analyze_functions(raw, point, variables)

    def test_paper_a_supplied_ridge_normal_equations(self):
        details = solve_normal_equations([[4, 0], [0, 9]], [[8], [18]], 1)
        np.testing.assert_allclose(details["w"], [[1.6], [1.8]])
        np.testing.assert_allclose(details["normal_residual"], [[0], [0]], atol=1e-12)

    def test_singular_normal_equations_and_multi_output(self):
        details = solve_normal_equations([[1, 1], [1, 1]], [[2, 4], [2, 4]])
        np.testing.assert_allclose(details["w"], [[1, 2], [1, 2]])
        details = solve_normal_equations([[1, 1], [1, 1]], [[2], [2]], 1e-20)
        np.testing.assert_allclose(details["w"], [[1], [1]], atol=1e-10)

    def test_impossible_normal_equation_summaries(self):
        for gram, rhs in (([[1, 2], [0, 1]], [[1], [1]]), ([[1, 0], [0, -1]], [[1], [1]]), ([[0]], [[1]]), ([[1, 2]], [[1]])):
            for penalty in (0, 1):
                with self.subTest(gram=gram, penalty=penalty), self.assertRaises(ValueError):
                    solve_normal_equations(gram, rhs, penalty)

    def test_paper_a_polynomial_count_and_feature_order(self):
        details = polynomial_features_details(3, 2, [[2, -1, 3]])
        self.assertEqual(details["count"], 10)
        self.assertEqual(details["names"], ["1", "x1", "x2", "x3", "x1^2", "x1 x2", "x1 x3", "x2^2", "x2 x3", "x3^2"])
        np.testing.assert_allclose(details["transformed"], [[1, 2, -1, 3, 4, -2, 6, 1, -3, 9]])
        self.assertEqual(polynomial_features_details(3, 2, include_bias=False)["count"], 9)

    def test_feature_preview_limits_and_dimensions(self):
        with self.assertRaises(ValueError):
            polynomial_features_details(100, 12)
        with self.assertRaises(ValueError):
            polynomial_features_details(3, 2, [[1, 2]])
        shapes = regression_shapes(9, 40)
        self.assertEqual(shapes["dual"], (9, 9))
        self.assertEqual(shapes["primal"], (40, 40))
        self.assertEqual(shapes["recommended_form"], "dual form")

    def test_paper_a_two_output_regression_and_normal_checks(self):
        X = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
        Y = [[-1, 4], [4, 2], [0, 1], [2, 6], [2, 2]]
        result = fit_regression_model(X, Y, [[2, -1]], None, "Linear", 1, "Auto", "None", 1, "auto", True)
        np.testing.assert_allclose(result["test_pred"], [[5.4, -1.5]])
        np.testing.assert_allclose(result["diagnostics"]["normal_residual"], np.zeros((3, 2)), atol=1e-10)
        self.assertEqual(result["target_mode"], "Numeric regression")

    def test_ridge_stationarity_and_total_error(self):
        X, Y = [[1, 0], [1, 1]], [[1], [3]]
        details = regression_diagnostics(X, Y, [[1], [1]], 1)
        self.assertEqual(details["total_sse"], 1)
        self.assertEqual(details["sse_per_sample"], .5)
        self.assertEqual(details["objective"], 3)
        np.testing.assert_allclose(details["objective_gradient"], [[0], [0]])
        np.testing.assert_allclose(details["normal_residual"], [[-1], [-1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
