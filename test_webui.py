"""Run with python -m unittest test_webui; each test uses isolated saved state."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

import webui


class WebUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        state_patch = patch.object(webui, "STATE_FILE", Path(self.temp.name) / "state.json")
        state_patch.start()
        self.addCleanup(state_patch.stop)
        self.app = AppTest.from_string("import webui\nwebui.main()", default_timeout=15).run()
        self.assertEqual(len(self.app.exception), 0)

    def cached(self, name):
        value = next(entry["value"] for entry in self.app.session_state["matrix_cache"] if entry["name"] == name)
        return np.asarray(value, dtype=float)

    def assert_success(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual([error.value for error in self.app.error], [])

    def test_parent_tree_example_and_children_only(self):
        self.app.button("FormSubmitter:classification_tree_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("weighted_gini"), [[0.5027777777777778]])
        np.testing.assert_allclose(self.cached("weighted_gini_per_node"), [[10 / 18 * 0.48], [8 / 18 * 0.53125]])
        self.app.checkbox("tree_includes_parent").uncheck()
        self.app.text_area("tree_counts_raw").set_value("6 4 0\n2 1 5")
        self.app.button("FormSubmitter:classification_tree_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("weighted_gini"), [[0.5027777777777778]])

    def test_invalid_parent_is_reported(self):
        self.app.text_area("tree_counts_raw").set_value("8 5 4\n6 4 0\n2 1 5")
        self.app.button("FormSubmitter:classification_tree_form-Calculate").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("Parent class counts", self.app.error[0].value)

    def test_array_solve_pearson_gradient_and_kmeans(self):
        self.app.button("FormSubmitter:array_form-Inspect").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("inverse"), [[-2, 1], [1.5, -0.5]])
        self.app.button("FormSubmitter:solve_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("w"), [[1], [1]])
        self.app.text_area("pearson_x_raw").set_value("1;2;3")
        self.app.text_area("pearson_y_raw").set_value("2;4;6")
        self.app.button("FormSubmitter:pearson_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("pearson"), [[1]])
        self.app.number_input("gradient_iterations").set_value(1)
        self.app.button("FormSubmitter:gradient_form-Run").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("final_parameters"), [[1], [-0.4]])
        self.app.text_area("kmeans_x_raw").set_value("1 1;1.5 2;5 7;6 8")
        self.app.text_area("kmeans_centroids_raw").set_value("1 1;5 7")
        self.app.button("FormSubmitter:kmeans_form-Run").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("final_centroids"), [[1.25, 1.5], [5.5, 7.5]])

    def test_regression_square_ridge_and_onehot_predictions(self):
        self.app.selectbox("reg_target_mode").select("Numeric regression")
        self.app.selectbox("reg_regularization").select("Ridge").run()
        self.app.text_area("reg_x_raw").set_value("0;1")
        self.app.text_area("reg_y_raw").set_value("1;3")
        self.app.button("FormSubmitter:regression_form-Fit").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("w"), [[1], [1]])
        self.app.selectbox("reg_target_mode").select("One-hot classification")
        self.app.selectbox("reg_regularization").select("None")
        self.app.checkbox("reg_predict_test").check().run()
        self.app.text_area("reg_x_raw").set_value("0;1;2;3")
        self.app.text_area("reg_y_raw").set_value("low;low;high;high")
        self.app.text_area("reg_x_test_raw").set_value("2.5")
        self.app.text_area("reg_y_test_raw").set_value("high")
        self.app.button("FormSubmitter:regression_form-Fit").click().run()
        self.assert_success()
        self.assertTrue(any("high" in latex.value for latex in self.app.latex))
        np.testing.assert_allclose(self.cached("y_predicted"), [[0.1, 0.9]])

    def test_regression_tree_boundary_and_best_split(self):
        self.app.selectbox("tree_mode").select("Regression MSE / split").run()
        self.app.text_area("tree_x_raw").set_value("1;2;3")
        self.app.text_area("tree_y_raw").set_value("1;5;9")
        self.app.text_input("tree_threshold").set_value("2")
        self.app.button("FormSubmitter:regression_tree_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("threshold_summary").astype(float).reshape(-1)[-1], 8 / 3)
        np.testing.assert_allclose(self.cached("best_split").astype(float).reshape(-1), [1.5, 8 / 3])

    def test_invalid_optional_test_targets_are_not_ignored(self):
        self.app.selectbox("reg_target_mode").select("Numeric regression")
        self.app.checkbox("reg_predict_test").check().run()
        self.app.text_area("reg_y_raw").set_value("1;3;5;7")
        self.app.text_area("reg_y_test_raw").set_value("1 2;3")
        self.app.button("FormSubmitter:regression_form-Fit").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("Correct Y_test", self.app.error[0].value)

    def test_polynomial_and_sign_examples(self):
        self.app.selectbox("reg_model_type").select("Polynomial")
        self.app.selectbox("reg_target_mode").select("Numeric regression")
        self.app.checkbox("reg_predict_test").check().run()
        self.app.text_area("reg_x_raw").set_value("0;1;2")
        self.app.text_area("reg_y_raw").set_value("1;4;9")
        self.app.text_area("reg_x_test_raw").set_value("3")
        self.app.button("FormSubmitter:regression_form-Fit").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("y_predicted"), [[16]])
        self.app.selectbox("reg_model_type").select("Linear")
        self.app.selectbox("reg_target_mode").select("Sign classification").run()
        self.app.checkbox("reg_include_bias").uncheck()
        self.app.text_area("reg_x_raw").set_value("-1;1")
        self.app.text_area("reg_y_raw").set_value("-1;1")
        self.app.text_area("reg_x_test_raw").set_value("-2;0;2")
        self.app.button("FormSubmitter:regression_form-Fit").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("y_predicted"), [[-2], [0], [2]])
        self.assertTrue(any("-1" in latex.value and "0" in latex.value for latex in self.app.latex))

    def test_transpose_precision_and_cache_clear(self):
        values = [[1 / 3, 1e-12], [9999.01, -np.pi]]
        self.app.text_area("array_inspect_raw").set_value(str(values))
        self.app.button("array_inspect_transpose").click().run()
        np.testing.assert_array_equal(webui.parse_matrix(self.app.text_area("array_inspect_raw").value), np.array(values).T)
        self.app.button("FormSubmitter:array_form-Inspect").click().run()
        self.assert_success()
        clear = next(button for button in self.app.button if button.label == "Clear cache")
        clear.click().run()
        self.assertEqual(self.app.session_state["matrix_cache"], [])

    def test_new_modes_keep_default_forms_compact(self):
        self.assertEqual(len(self.app.tabs), 9)
        self.assertEqual([tab.label for tab in self.app.tabs][-2:], ["Probability", "Cache"])
        self.assertFalse(any(field.key in {"prep_data", "normal_gram_raw", "function_expressions", "bayes_table"} for field in self.app.text_area))
        self.app.selectbox("reg_workflow").select("Explore polynomial features").run()
        self.assertFalse(any(field.key == "reg_y_raw" for field in self.app.text_area))

    def test_data_preparation_and_matrix_calculation(self):
        self.app.selectbox("array_mode").select("Prepare data").run()
        self.app.checkbox("prep_supplied").check().run()
        self.app.text_area("prep_data").set_value("18")
        self.app.button("FormSubmitter:preparation_form-Transform").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("prepared_data"), [[.4]])
        self.app.selectbox("prep_method").select("Mean imputation").run()
        self.app.text_area("prep_data").set_value("2;4;12;?")
        self.app.button("FormSubmitter:preparation_form-Transform").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("prepared_data"), [[2], [4], [12], [6]])
        self.app.selectbox("array_mode").select("Calculate").run()
        self.app.selectbox("matrix_operation").select("Dot product").run()
        self.app.button("FormSubmitter:matrix_calculation_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("matrix_result"), [[-1]])

    def test_probability_modes(self):
        self.app.button("FormSubmitter:discrete_probability_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("event_probability"), [[.75]])
        self.app.selectbox("probability_mode").select("Bayes update").run()
        self.app.button("FormSubmitter:bayes_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("posterior"), [[15/29], [14/29]])
        self.app.selectbox("probability_mode").select("Two events").run()
        self.app.button("FormSubmitter:events_form-Calculate").click().run()
        self.assert_success()
        self.assertTrue(any("Independent: yes" in item.value for item in self.app.markdown))
        self.app.selectbox("probability_mode").select("Continuous density").run()
        self.app.button("FormSubmitter:density_form-Integrate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("interval_probability"), [[.75]])

    def test_function_gradients_jacobians_and_given_scores(self):
        self.app.selectbox("gradient_mode").select("Function and derivatives at a point").run()
        self.app.button("FormSubmitter:function_analysis_form-Evaluate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("gradient"), [[-1], [-1]])
        self.app.selectbox("function_interpretation").select("Sign classification").run()
        self.app.text_area("function_expressions").set_value("1-2*x1+x1*x2")
        self.app.text_input("function_point").set_value("2,1")
        self.app.button("FormSubmitter:function_analysis_form-Evaluate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("function_class"), [[-1]])
        self.app.selectbox("function_interpretation").select("Numeric values").run()
        self.app.text_area("function_expressions").set_value("x1**2+x2;x1*x2")
        self.app.text_input("function_variables").set_value("x1,x2,x3")
        self.app.text_input("function_point").set_value("2,-1,7")
        self.app.button("FormSubmitter:function_analysis_form-Evaluate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("jacobian"), [[4, 1, 0], [-1, 2, 0]])
        self.app.selectbox("function_interpretation").select("Class scores").run()
        self.app.text_area("function_expressions").set_value("-0.2;-0.8;-0.1")
        self.app.text_input("function_variables").set_value("")
        self.app.text_input("function_point").set_value("")
        self.app.button("FormSubmitter:function_analysis_form-Evaluate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("function_class"), [[2]])

    def test_normal_summaries_and_feature_preview(self):
        self.app.selectbox("reg_workflow").select("Use normal equation summaries").run()
        self.app.selectbox("normal_regularization").select("Ridge").run()
        self.app.text_area("normal_predict_raw").set_value("1 1")
        self.app.button("FormSubmitter:normal_equations_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("summary_weights"), [[1.6], [1.8]])
        np.testing.assert_allclose(self.cached("summary_prediction"), [[3.4]])
        self.app.selectbox("reg_workflow").select("Explore polynomial features").run()
        self.app.text_area("features_points_raw").set_value("2 -1 3")
        self.app.button("FormSubmitter:polynomial_features_form-Preview").click().run()
        self.assert_success()
        self.assertTrue(any(metric.label == "Monomial coefficients" and metric.value == "10" for metric in self.app.metric))
        np.testing.assert_allclose(self.cached("polynomial_features"), [[1, 2, -1, 3, 4, -2, 6, 1, -3, 9]])

    def test_invalid_probability_is_reported_without_crashing(self):
        self.app.text_area("pmf_table").set_value("0 .7;1 .6")
        self.app.button("FormSubmitter:discrete_probability_form-Calculate").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("sum to 1", self.app.error[0].value)

    def test_minimum_point_and_stationary_nonminimum(self):
        self.app.selectbox("gradient_mode").select("Check a minimum point").run()
        self.app.button("FormSubmitter:minimum_form-Check point").click().run()
        self.assert_success()
        self.assertEqual(self.app.success[0].value, "Strict global minimum")
        np.testing.assert_allclose(self.cached("minimum_value"), [[2]])
        self.app.text_input("minimum_expression").set_value("x^3")
        self.app.text_input("minimum_point").set_value("0")
        self.app.button("FormSubmitter:minimum_form-Check point").click().run()
        self.assert_success()
        self.assertEqual(self.app.info[0].value, "Neither minimum nor maximum")
        self.app.text_input("minimum_expression").set_value("x^2-y^4")
        self.app.text_input("minimum_point").set_value("0,0")
        self.app.button("FormSubmitter:minimum_form-Check point").click().run()
        self.assert_success()
        self.assertEqual(self.app.warning[0].value, "Inconclusive")

    def test_matlab_matrix_differentiation_symbolic_and_numeric(self):
        self.app.selectbox("gradient_mode").select("Differentiate a matrix expression").run()
        self.app.button("FormSubmitter:matrix_derivative_form-Differentiate").click().run()
        self.assert_success()
        self.assertTrue(any("2 x_{1}" in item.value for item in self.app.latex))
        self.app.text_area("matdiff_evaluation").set_value("x=[1;1]")
        self.app.button("FormSubmitter:matrix_derivative_form-Differentiate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("matrix_derivative"), [[4], [8]])
        self.app.text_area("matdiff_expression").set_value("A*x")
        self.app.button("FormSubmitter:matrix_derivative_form-Differentiate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("matrix_derivative"), [[1, 2], [0, 3]])

    def test_invalid_matlab_shape_is_reported_without_crashing(self):
        self.app.selectbox("gradient_mode").select("Differentiate a matrix expression").run()
        self.app.text_area("matdiff_expression").set_value("A*x")
        self.app.text_area("matdiff_definitions").set_value("A=[1 2 3];syms x [2 1] real")
        self.app.button("FormSubmitter:matrix_derivative_form-Differentiate").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("dimensions", self.app.error[0].value)

    def test_quartiles_and_conventions(self):
        self.app.selectbox("array_mode").select("Summarize data").run()
        self.app.button("FormSubmitter:statistics_form-Summarize").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("quartile_summary"), [[1, 2.5, 6, 9.5, 12, 7]])
        self.app.text_area("stats_data").set_value("1;2;3;4;5")
        self.app.selectbox("stats_convention").select("Linear interpolation")
        self.app.button("FormSubmitter:statistics_form-Summarize").click().run()
        self.assert_success()
        self.assertEqual(self.cached("quartile_summary")[0, 3], 4)

    def test_sampling_without_replacement_and_grouped_rates(self):
        self.app.selectbox("probability_mode").select("Sampling without replacement").run()
        self.app.button("FormSubmitter:sampling_form-Calculate").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("sampling_probability"), [[.6]])
        self.app.selectbox("probability_mode").select("Compare grouped rates").run()
        self.app.button("FormSubmitter:rates_form-Compare").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("pooled_rates"), [[29/110, 81/110]])
        self.assertIn("reversal", self.app.info[0].value)

    def test_dynamic_probability_solver_and_related_answers(self):
        self.app.selectbox("probability_mode").select("Solve from known probabilities").run()
        self.app.checkbox("probability_equations_related").check()
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("P(B)"), [[.4]])
        self.assertTrue(any("2/5" in str(table.value) for table in self.app.dataframe))
        self.app.text_area("probability_equations").set_value("a*b=.1; a|b=.25; b=???")
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("P(B)"), [[.4]])

    def test_dynamic_probability_solver_reports_ranges_and_bad_facts(self):
        self.app.selectbox("probability_mode").select("Solve from known probabilities").run()
        self.app.text_area("probability_equations").set_value("a=.4; b=???")
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assert_success()
        self.assertTrue(any("Not uniquely determined" in str(table.value) for table in self.app.dataframe))
        self.app.text_area("probability_equations").set_value("a=.9; b=.9; a*b=.1; a|b=???")
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("inconsistent", self.app.error[0].value)

    def test_dynamic_probability_solver_accepts_c(self):
        self.app.selectbox("probability_mode").select("Solve from known probabilities").run()
        self.app.text_area("probability_equations").set_value("a*c=.12; a|c=.3; c=???")
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("P(C)"), [[.4]])
        self.app.text_area("probability_equations").set_value("a*b*c=.05; b*c=.15; a|(b*c)=???")
        self.app.button("FormSubmitter:probability_equations_form-Solve").click().run()
        self.assert_success()
        np.testing.assert_allclose(self.cached("P(A|(B*C))"), [[1/3]])

    def test_decision_boundary_and_compact_paper_b_modes(self):
        self.assertEqual(len(self.app.tabs), 9)
        self.assertFalse(any(field.key in {"matdiff_expression", "rates_table", "stats_data"} for field in self.app.text_area))
        self.assertFalse(any(field.key == "minimum_expression" for field in self.app.text_input))
        self.app.selectbox("gradient_mode").select("Function and derivatives at a point").run()
        self.app.selectbox("function_interpretation").select("Sign classification").run()
        self.app.text_area("function_expressions").set_value("2*x1-x2-1")
        self.app.text_input("function_point").set_value("1,1")
        self.app.button("FormSubmitter:function_analysis_form-Evaluate").click().run()
        self.assert_success()
        self.assertTrue(any("x_{2} = 2 x_{1} - 1" in item.value for item in self.app.latex))


if __name__ == "__main__":
    unittest.main(verbosity=2)
