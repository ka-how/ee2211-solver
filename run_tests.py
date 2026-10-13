import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from CommonUtils import (
    add_bias_column,
    calculate_closed_form_weights,
    calculate_ridge_weights,
    ensure_2d,
    format_number,
    parse_mixed_array,
    parse_numeric_array,
    pretty_print_array,
    print_error_summary,
    python_array_literal,
    squared_error_summary,
)
from DecisionTreeUtils import classification_impurity_summary, classification_split_details, find_best_regression_split, regression_threshold_summary
from GradientDescent import gradient_descent
from KMeansUtils import k_means
from LinearRegression import linear_regression
from OneHotLinearClassification import (
    fit_onehot_linearclassification,
    fit_onehot_polynomialclassification,
    predict_onehot_linearclassification,
    predict_onehot_polynomialclassification,
    prepare_class_targets,
)
from PolynomialRegression import fit_polynomial_regression, predict_polynomial_regression
from RidgePolynomialRegression import fit_ridge_poly_regression, predict_ridge_poly_regression
from RidgeRegression import fit_ridge_regression, predict_ridge_regression
from pearson_correlation import pearson_correlation, pearson_correlation_details
from webui import fit_regression_model, matrix_editor_text, parse_matrix, prepare_test_target_matrix, resolve_target_mode, solve_linear_system, inspect_matrix, deserialize_cache_entry


@contextlib.contextmanager
def quiet_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class SolverComputationTests(unittest.TestCase):
    def test_parent_is_excluded_from_user_tree_example(self):
        counts = [[8, 5, 5], [6, 4, 0], [2, 1, 5]]
        expected_impurities = [0.48, 0.53125]
        expected_gini = (10 * 0.48 + 8 * 0.53125) / 18
        details = classification_split_details(counts, "gini", includes_parent=True)
        np.testing.assert_allclose(details["weights"], [10 / 18, 8 / 18])
        np.testing.assert_allclose(details["impurities"], expected_impurities)
        np.testing.assert_allclose(details["weighted_impurity"], expected_gini)
        np.testing.assert_allclose(details["parent_impurity"], 35 / 54)
        np.testing.assert_allclose(details["gain"], 35 / 54 - expected_gini)
        for measure in ("gini", "entropy", "misclassification"):
            with self.subTest(measure=measure):
                parent = classification_split_details(counts, measure, True)
                children = classification_split_details(counts[1:], measure)
                self.assertAlmostEqual(parent["weighted_impurity"], children["weighted_impurity"])
                self.assertAlmostEqual(sum(parent["weighted_contributions"]), parent["weighted_impurity"])

    def test_terminal_tree_defaults_to_parent_followed_by_children(self):
        from main import run_classification_tree_impurity

        stdout = io.StringIO()
        with patch("builtins.input", side_effect=["8 5 5;6 4 0;2 1 5", "", "gini"]), contextlib.redirect_stdout(stdout):
            run_classification_tree_impurity()
        self.assertIn("weighted_gini: 0.5028", stdout.getvalue())
        self.assertIn("gini_gain: 0.1454", stdout.getvalue())

    def test_book_golf_entropy_and_gain(self):
        details = classification_split_details([[10, 5], [2, 3], [4, 0], [4, 2]], "entropy", True)
        self.assertAlmostEqual(details["weighted_impurity"], 0.6910, places=4)
        self.assertAlmostEqual(details["gain"], 0.2273, places=4)

    def test_tree_count_validation_and_unused_branches(self):
        details = classification_split_details([[2, 0], [0, 0], [0, 3]], "entropy")
        np.testing.assert_allclose(details["weights"], [0.4, 0, 0.6])
        self.assertEqual(details["weighted_impurity"], 0)
        for counts in ([[0, 0]], [[1, -1]], [[np.nan, 2]], [[np.inf, 1]]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                classification_split_details(counts)
        with self.assertRaisesRegex(ValueError, "Parent class counts"):
            classification_split_details([[8, 5, 4], [6, 4, 0], [2, 1, 5]], includes_parent=True)

    def test_tree_threshold_boundary_matches_book(self):
        summary = regression_threshold_summary([1, 2, 3], [1, 5, 9], 2)
        self.assertEqual(summary["left_size"], 1)
        self.assertEqual(summary["right_size"], 2)
        self.assertEqual(summary["right_mean"], 7)
        self.assertAlmostEqual(summary["overall_mse"], 8 / 3)

    def test_tree_rejects_multiple_features_and_nonfinite_thresholds(self):
        with self.assertRaisesRegex(ValueError, "single row or column"):
            find_best_regression_split([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        for threshold in (np.nan, np.inf, -np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                regression_threshold_summary([1, 2], [1, 2], threshold)

    def test_tree_book_house_price_split(self):
        best, _ = find_best_regression_split([0.5, 0.6, 1, 2, 3, 3.2, 3.8], [0.19, 0.23, 0.28, 0.42, 0.53, 0.75, 0.8])
        self.assertEqual(best["threshold"], 2.5)
        self.assertAlmostEqual(best["overall_mse"], 0.0102, places=4)

    def test_tree_splits_adjacent_and_large_feature_values(self):
        for low in (1.0, 1e308):
            high = np.nextafter(low, np.inf)
            best, summaries = find_best_regression_split([high, low, high], [3, 1, 3])
            self.assertEqual(len(summaries), 1)
            self.assertEqual(best["left_size"], 1)
            self.assertEqual(best["overall_mse"], 0)

    def test_singular_regression_has_minimum_norm_weights(self):
        for X, y, expected in (
            ([[1, 1], [2, 2]], [[2], [4]], [[1], [1]]),
            ([[1, 1], [2, 2], [3, 3]], [[2], [4], [6]], [[1], [1]]),
            ([[1, 1, 0]], [[2]], [[1], [1], [0]]),
        ):
            with self.subTest(X=X):
                _, weights = calculate_closed_form_weights(X, y)
                np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_ridge_auto_regularizes_square_matrices(self):
        for form in ("auto", "primal form", "dual form"):
            with self.subTest(form=form):
                system, resolved, weights = calculate_ridge_weights(np.eye(2), [[2], [4]], 1, form)
                self.assertEqual(system, "square")
                self.assertIn(resolved, ("primal form", "dual form"))
                np.testing.assert_allclose(weights, [[1], [2]])

    def test_zero_ridge_handles_rank_deficiency_and_invalid_settings(self):
        for form in ("auto", "primal form", "dual form"):
            _, _, weights = calculate_ridge_weights([[1, 1], [2, 2]], [[2], [4]], 0, form)
            np.testing.assert_allclose(weights, [[1], [1]])
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate_ridge_weights(np.eye(2), [[1], [2]], value)
        with self.assertRaisesRegex(ValueError, "Unknown ridge form"):
            calculate_ridge_weights(np.eye(2), [[1], [2]], 1, "invalid")

    def test_copy_and_transpose_preserve_numeric_precision(self):
        values = np.array([[1 / 3, 1e-12], [9999.01, -np.pi]])
        np.testing.assert_array_equal(parse_numeric_array(python_array_literal(values)), values)
        np.testing.assert_array_equal(parse_matrix(matrix_editor_text(values.T)), values.T)
        self.assertEqual(format_number(9999.01), "9999.01")
        self.assertEqual(format_number(1e-12), "1e-12")
        old_cache = deserialize_cache_entry({"value": values.tolist(), "text": "[[0.3333,0],[9999,-3.1416]]"})
        np.testing.assert_array_equal(parse_numeric_array(old_cache["text"]), values)
        self.assertIsNone(deserialize_cache_entry({"value": []}))

    def test_array_validation_and_multiline_input(self):
        np.testing.assert_array_equal(parse_numeric_array("1 −2\n3 4"), [[1, -2], [3, 4]])
        np.testing.assert_array_equal(parse_mixed_array("low\nhigh"), [["low"], ["high"]])
        for value in ([], [[[1]]], [[np.nan]], [[np.inf]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ensure_2d(value)
        with self.assertRaises(ValueError):
            squared_error_summary([[1], [2]], [[1, 2], [3, 4]])

    def test_onehot_test_indices_keep_training_column_order(self):
        encoded = prepare_test_target_matrix([[2], [0]], "One-hot classification", None, 3)
        np.testing.assert_array_equal(encoded, [[0, 0, 1], [1, 0, 0]])
        with self.assertRaises(ValueError):
            prepare_test_target_matrix([[3]], "One-hot classification", None, 3)

    def test_classifier_validates_targets_and_preserves_numeric_labels(self):
        for value in ([["cat", "dog"]], [[1, 1]], [[0.1, 0.9]], [[np.nan]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_class_targets(value)
        _, labels = prepare_class_targets([[100000], [100000.1]])
        self.assertEqual(labels, [100000, 100000.1])
        self.assertEqual(resolve_target_mode([[2, 3], [4, 5]], "Auto"), "Numeric regression")
        self.assertEqual(resolve_target_mode([[0, 1], [1, 0]], "Auto"), "One-hot classification")

    def test_gradient_tolerance_does_not_treat_oscillation_as_convergence(self):
        _, _, history = gradient_descent("x**2", initial_values=[1], learning_rate=1, iterations=4, tolerance=1e-6)
        self.assertEqual(len(history), 5)
        np.testing.assert_allclose([item["values"][0] for item in history], [1, -1, 1, -1, 1])

    def test_gradient_validates_variables_and_settings(self):
        for options in ({"learning_rate": 0}, {"learning_rate": np.nan}, {"iterations": -1}, {"iterations": 1.5}, {"tolerance": -1}, {"variables": ["y"]}, {"variables": ["x", "x"]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                gradient_descent("x**2", initial_values=[1], **options)
        with self.assertRaises(ValueError):
            gradient_descent("log(x)", initial_values=[-1])

    def test_pearson_accepts_small_nonconstant_values(self):
        details = pearson_correlation_details([1e-12, 2e-12, 3e-12], [2e-12, 4e-12, 6e-12])
        np.testing.assert_allclose(details["pearson"], [[1]])
        with self.assertRaises(ValueError):
            pearson_correlation_details([1, 1, 1], [1, 2, 3])

    def test_kmeans_small_centroid_changes_are_not_skipped(self):
        history = k_means([[0], [2e-12]], [[0]])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["centroids"][0, 0], 1e-12)

    def test_matrix_inspection_and_left_solve(self):
        _, inverses = inspect_matrix([[1, 1], [1, 1.00001], [1, 0]])
        np.testing.assert_allclose(inverses[0][1] @ [[1, 1], [1, 1.00001], [1, 0]], np.eye(2), atol=1e-10)
        solution = solve_linear_system([[1, 2, 3], [0, 1, 0]], [[2, 7, 6]], "wX = y")
        np.testing.assert_allclose(solution["w"], [[2, 3]])
        self.assertEqual(solution["status"], "unique exact")

    def test_common_array_parsing_accepts_bracket_variants(self):
        np.testing.assert_allclose(parse_numeric_array("(1 2; 3 4)"), [[1, 2], [3, 4]])
        np.testing.assert_allclose(parse_numeric_array("{1, 2; 3, 4}"), [[1, 2], [3, 4]])
        np.testing.assert_allclose(parse_numeric_array("{[1, 2], [3, 4]}"), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(parse_mixed_array("{cat; dog}"), np.asarray([["cat"], ["dog"]], dtype=object))

    def test_single_value_arrays_print_inline(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_error_summary(np.array([[1]], dtype=float), np.array([[1]], dtype=float))
            pretty_print_array("mse", np.array([[0]], dtype=float), show_python=False)

        output = stdout.getvalue()
        self.assertIn("square error: 0", output)
        self.assertIn("MEAN square error: 0", output)
        self.assertIn("mse: 0", output)
        self.assertNotIn("[0]", output)

    def test_closed_form_and_linear_regression(self):
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
        y = np.array([[1], [3], [5]], dtype=float)

        system, expected_w = calculate_closed_form_weights(X, y)
        self.assertEqual(system, "overdetermined")
        np.testing.assert_allclose(expected_w, [[1], [2]], atol=1e-10)

        with quiet_stdout():
            _, w, _, _, y_predicted = linear_regression(X, y, np.array([[1, 3]], dtype=float))
        assert w.shape == (2, 1), f"Expected weight shape (2, 1), got {w.shape}"
        assert y_predicted is not None, "y_predicted should not be None"
        assert y_predicted.shape == (1, 1), f"Expected predicted shape (1, 1), got {y_predicted.shape}"
        np.testing.assert_allclose(w, expected_w, atol=1e-10)
        np.testing.assert_allclose(y_predicted, [[7]], atol=1e-10)

    def test_polynomial_regression_prediction(self):
        X = np.array([[0], [1], [2]], dtype=float)
        y = 1 + 2 * X + X**2
        with quiet_stdout():
            poly, _, w, _, _ = fit_polynomial_regression(X, y, order=2)
            y_predicted = predict_polynomial_regression(poly, w, np.array([[3]], dtype=float))
        assert w.shape == (3, 1), f"Expected weight shape (3, 1), got {w.shape}"
        assert y_predicted is not None, "y_predicted should not be None"
        assert y_predicted.shape == (1, 1), f"Expected predicted shape (1, 1), got {y_predicted.shape}"
        np.testing.assert_allclose(w.reshape(-1), [1, 2, 1], atol=1e-10)
        np.testing.assert_allclose(y_predicted, [[16]], atol=1e-10)

    def test_ridge_regression(self):
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
        y = np.array([[1], [3], [5]], dtype=float)
        ridge_lambda = 0.5
        _, _, expected_w = calculate_ridge_weights(X, y, ridge_lambda, form="primal form")

        with quiet_stdout():
            _, _, w, _, _ = fit_ridge_regression(X, y, LAMBDA=ridge_lambda, form="primal form")
            y_predicted = predict_ridge_regression(w, np.array([[1, 3]], dtype=float))
        assert w.shape == (2, 1), f"Expected weight shape (2, 1), got {w.shape}"
        assert y_predicted is not None, "y_predicted should not be None"
        assert y_predicted.shape == (1, 1), f"Expected predicted shape (1, 1), got {y_predicted.shape}"
        np.testing.assert_allclose(w, expected_w, atol=1e-10)
        np.testing.assert_allclose(y_predicted, np.array([[1, 3]], dtype=float) @ expected_w, atol=1e-10)

    def test_ridge_polynomial_regression_prediction(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = 1 + X
        with quiet_stdout():
            poly, _, _, w, _, _ = fit_ridge_poly_regression(X, y, LAMBDA=0.1, order=2, form="primal form")
            y_predicted = predict_ridge_poly_regression(poly, w, np.array([[4]], dtype=float))
        expected = poly.transform(np.array([[4]], dtype=float)) @ w
        np.testing.assert_allclose(y_predicted, expected, atol=1e-10)

    def test_gui_regression_optional_test_mse(self):
        result = fit_regression_model(
            X=np.array([[0], [1], [2], [3]], dtype=float),
            Y=np.array([[1], [3], [5], [7]], dtype=float),
            X_test=np.array([[4], [5]], dtype=float),
            Y_test=np.array([[9], [12]], dtype=float),
            model_type="Linear",
            order=1,
            target_mode="Numeric regression",
            regularization="None",
            ridge_lambda=1.0,
            ridge_form="auto",
            include_bias=True,
        )

        np.testing.assert_allclose(result["test_pred"], [[9], [11]], atol=1e-10)
        np.testing.assert_allclose(result["test_sse"], [[1]], atol=1e-10)
        np.testing.assert_allclose(result["test_mse"], [[0.5]], atol=1e-10)

    def test_onehot_classification(self):
        X = add_bias_column(np.array([[0], [1], [2], [3]], dtype=float))
        y = np.array([["low"], ["low"], ["high"], ["high"]], dtype=object)

        with quiet_stdout():
            _, w, _, labels, _, _ = fit_onehot_linearclassification(X, y)
            y_predicted = predict_onehot_linearclassification(w, np.array([[1, 2.5]], dtype=float), labels)
        assert labels is not None, "Labels should not be None"
        self.assertEqual(labels, ["low", "high"])
        self.assertEqual(labels[int(np.argmax(y_predicted[0]))], "high")

    def test_onehot_polynomial_classification_prediction(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([["left"], ["left"], ["right"], ["right"]], dtype=object)
        with quiet_stdout():
            poly, _, w, _, labels, _, _ = fit_onehot_polynomialclassification(X, y, order=2)
            y_predicted = predict_onehot_polynomialclassification(poly, w, np.array([[2.5]], dtype=float), labels)
        expected = poly.transform(np.array([[2.5]], dtype=float)) @ w
        np.testing.assert_allclose(y_predicted, expected, atol=1e-10)

    def test_pearson_correlation_for_non_six_observations(self):
        X = np.array([[1, 2, 3, 4], [1, 1, 2, 2]], dtype=float)
        Y = np.array([[2, 4, 6, 8]], dtype=float)
        with quiet_stdout():
            pearson = pearson_correlation(X, Y)
        np.testing.assert_allclose(pearson[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(pearson[1, 0], 1 / np.sqrt(1.25), atol=1e-10)

    def test_pearson_correlation_for_multiple_feature_and_output_columns(self):
        X = np.array([[1, 1], [2, 1], [3, 2], [4, 2]], dtype=float)
        Y = np.array([[2, 1, 4], [4, 1, 3], [6, 2, 2], [8, 2, 1]], dtype=float)
        expected = np.empty((X.shape[1], Y.shape[1]))
        for x_index in range(X.shape[1]):
            for y_index in range(Y.shape[1]):
                expected[x_index, y_index] = np.corrcoef(X[:, x_index], Y[:, y_index])[0, 1]

        with quiet_stdout():
            pearson = pearson_correlation(X, Y)

        self.assertEqual(pearson.shape, (2, 3))
        np.testing.assert_allclose(pearson, expected, atol=1e-10)

    def test_gradient_descent_lecture_and_past_paper_examples(self):
        _, _, history = gradient_descent("x**2", variables=["x"], initial_values=[1], learning_rate=0.4, iterations=4)
        np.testing.assert_allclose(history[-1]["values"], [0.0016], atol=1e-12)

        _, _, history = gradient_descent("sin(w)**2", variables=["w"], initial_values=[3], learning_rate=0.1, iterations=1)
        np.testing.assert_allclose(history[0]["gradient"], [2 * np.sin(3) * np.cos(3)], atol=1e-12)
        np.testing.assert_allclose(history[1]["values"], [3.0279415498198925], atol=1e-12)
        np.testing.assert_allclose(
            history[1]["gradient"],
            [2 * np.sin(history[1]["values"][0]) * np.cos(history[1]["values"][0])],
            atol=1e-12,
        )

        _, _, history = gradient_descent("x**2 + x*y**2", variables=["x", "y"], initial_values=[3, 2], learning_rate=0.2, iterations=1)
        np.testing.assert_allclose(history[0]["gradient"], [10, 12], atol=1e-12)
        np.testing.assert_allclose(history[1]["values"], [1, -0.4], atol=1e-12)
        np.testing.assert_allclose(history[1]["gradient"], [2.16, -0.8], atol=1e-12)

    def test_classification_tree_impurity(self):
        _, impurities, overall = classification_impurity_summary([[2, 5], [6, 0]], measure="gini")
        np.testing.assert_allclose(impurities, [20 / 49, 0], atol=1e-10)
        np.testing.assert_allclose(overall, 7 / 13 * 20 / 49, atol=1e-10)

        _, impurities, overall = classification_impurity_summary([[2, 5], [6, 0]], measure="entropy")
        expected_entropy = -(2 / 7) * np.log2(2 / 7) - (5 / 7) * np.log2(5 / 7)
        np.testing.assert_allclose(impurities, [expected_entropy, 0], atol=1e-10)
        np.testing.assert_allclose(overall, 7 / 13 * expected_entropy, atol=1e-10)

    def test_regression_tree_mse_and_best_split(self):
        x = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
        y = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])
        summary = regression_threshold_summary(x, y, 3)
        np.testing.assert_allclose(summary["root_mse"], 20.6381, atol=1e-4)
        np.testing.assert_allclose(summary["overall_mse"], 5.5648, atol=1e-4)

        best, summaries = find_best_regression_split(x, y)
        self.assertEqual(len(summaries), len(np.unique(x)) - 1)
        self.assertLessEqual(best["overall_mse"], summary["overall_mse"])

    def test_k_means_runs_until_convergence(self):
        X = np.array([[1, 1], [1.5, 2], [5, 7], [6, 8]], dtype=float)
        initial_centroids = np.array([[1, 1], [5, 7]], dtype=float)

        history = k_means(X, initial_centroids)

        self.assertTrue(history[-1]["converged"])
        self.assertEqual(len(history), 2)
        np.testing.assert_array_equal(history[0]["assignments"], [0, 0, 1, 1])
        np.testing.assert_allclose(history[0]["centroids"], initial_centroids, atol=1e-10)
        np.testing.assert_allclose(history[0]["cost"], 3.25, atol=1e-10)
        np.testing.assert_array_equal(history[-1]["assignments"], [0, 0, 1, 1])
        np.testing.assert_allclose(history[-1]["centroids"], [[1.25, 1.5], [5.5, 7.5]], atol=1e-10)
        np.testing.assert_allclose(history[-1]["cost"], 1.625, atol=1e-10)

    def test_k_means_max_iterations_stops_early(self):
        X = np.array([[1, 1], [1.5, 2], [5, 7], [6, 8]], dtype=float)
        initial_centroids = np.array([[1, 1], [5, 7]], dtype=float)

        history = k_means(X, initial_centroids, max_iterations=1)

        self.assertEqual(len(history), 2)
        self.assertFalse(history[-1]["converged"])
        np.testing.assert_allclose(history[-1]["centroids"], [[1.25, 1.5], [5.5, 7.5]], atol=1e-10)

    def test_k_means_validates_dimensions(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        centroids = np.array([[1, 2, 3]], dtype=float)

        with self.assertRaisesRegex(ValueError, "Feature dimension mismatch"):
            k_means(X, centroids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
