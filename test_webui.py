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


if __name__ == "__main__":
    unittest.main(verbosity=2)
