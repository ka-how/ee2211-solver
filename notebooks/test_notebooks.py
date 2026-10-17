"""Known-answer checks for notebook calculators, input cells, and shared cache."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FOLDER))

from notebook_tools import (NotebookCache, fit_model, fit_regression_model, inspect_matrix,
    matrix, optional_matrix, number, integer, boolean, coordinates, variable_names, solve_linear_system)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        cache_path = Path(self.temporary.name) / "cache.sqlite3"
        environment_patch = patch.dict(os.environ, EE2211_NOTEBOOK_CACHE=str(cache_path))
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def workflow(self, filename, parameter_index, calculation_index, **overrides):
        document = json.loads((FOLDER / filename).read_text(encoding="utf-8"))
        cells = {int(cell["id"].split("-")[1]): cell for cell in document["cells"]}
        namespace = {}
        with contextlib.chdir(FOLDER), patch("IPython.display.display"), patch("notebook_tools.display"), contextlib.redirect_stdout(io.StringIO()):
            exec("".join(cells[1]["source"]), namespace)
            exec("".join(cells[parameter_index]["source"]), namespace)
            namespace.update(overrides)
            exec("".join(cells[calculation_index]["source"]), namespace)
        return namespace

    def test_all_notebooks_execute_and_have_known_answers(self):
        paths = sorted(FOLDER.glob("*.ipynb"))
        self.assertEqual(len(paths), 9)
        namespaces = {}
        with contextlib.chdir(FOLDER), patch("IPython.display.display"), patch("notebook_tools.display"), contextlib.redirect_stdout(io.StringIO()):
            for path in paths:
                document = json.loads(path.read_text(encoding="utf-8"))
                namespace = {"__name__": "__main__"}
                for index, cell in enumerate(document["cells"]):
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        ast.parse(source, feature_version=(3, 11))
                        with self.subTest(notebook=path.name, cell=index):
                            exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
                            if cell.get("id") == "03-003":
                                # Use fixed test data without replacing the user's saved inputs.
                                namespace.update(X="0;1;2;3", Y="1;3;5;7", X_test="4;5", Y_test="9;11",
                                                 model="linear", target="numeric", bias=True,
                                                 ridge=0, degree=2, form="auto")
                namespaces[path.name[:2]] = namespace
        np.testing.assert_allclose(dict(namespaces["01"]["inverses"])["inverse"], [[-2, 1], [1.5, -.5]])
        np.testing.assert_allclose(namespaces["01"]["calculation_result"], [[-1]])
        np.testing.assert_allclose(namespaces["01"]["preparation"]["result"], [[0], [.4], [1]])
        np.testing.assert_allclose(namespaces["01"]["imputation"]["result"], [[2], [4], [12], [6]])
        np.testing.assert_allclose(namespaces["02"]["solution"]["w"], [[1], [1]])
        fitted = namespaces["03"]
        np.testing.assert_allclose(fitted["regression"]["test_pred"], [[9], [11]])
        self.assertTrue(fitted["bias"])
        np.testing.assert_allclose(fitted["variants"]["polynomial"]["test_pred"], [[16]])
        np.testing.assert_allclose(fitted["variants"]["ridge"]["w"], [[1], [1]])
        self.assertEqual(fitted["variants"]["one_hot"]["test_class"].item(), "high")
        np.testing.assert_array_equal(fitted["variants"]["sign"]["test_class"], [[-1], [0], [1]])
        np.testing.assert_allclose(fitted["summary_prediction"], [[3.4]])
        self.assertEqual(fitted["features"]["count"], 10)
        np.testing.assert_allclose(fitted["features"]["transformed"], [[1, 2, -1, 3, 4, -2, 6, 1, -3, 9]])
        np.testing.assert_allclose(np.diag(namespaces["04"]["pearson"]["pearson"]), [1, 1])
        gradient = namespaces["05"]
        np.testing.assert_allclose(gradient["gradient_history"][1]["values"], [1, -.4])
        np.testing.assert_allclose(gradient["function"]["jacobian_at_point"].T, [[-1], [-1]])
        np.testing.assert_allclose(gradient["vector_function"]["jacobian_at_point"], [[4, 1, 0], [-1, 2, 0]])
        self.assertEqual(gradient["minimum"]["classification"], "Strict global minimum")
        np.testing.assert_allclose(gradient["matrix_derivative"]["derivative_at_point"], [[4], [8]])
        self.assertAlmostEqual(namespaces["06"]["classification_summaries"]["gini"]["weighted_impurity"], .5027777777777778)
        np.testing.assert_allclose(namespaces["07"]["final"]["centroids"], [[1.25, 1.5], [5.5, 7.5]])
        probability = namespaces["08"]
        self.assertEqual(probability["distribution"]["event_probability"], .75)
        np.testing.assert_allclose(probability["bayes"]["posterior"], [15/29, 14/29])
        self.assertEqual(probability["density"]["probability"], .75)
        self.assertAlmostEqual(probability["sampling"]["event_probability"], .6)
        self.assertTrue(probability["rates"]["reversal"])
        self.assertGreater(len(namespaces["09"]["entries"]), 0)

    def test_solve_right_orientation_and_inconsistent_system(self):
        right = solve_linear_system(matrix("1 1 1; 1 2 3"), matrix("2 3 4"), "wX = y")
        np.testing.assert_allclose(right["w"], [[1, 1]])
        self.assertEqual(right["predicted"].shape, (1, 3))
        inconsistent = solve_linear_system([[1], [1]], [[1], [3]])
        self.assertEqual(inconsistent["status"], "least squares")
        np.testing.assert_allclose(inconsistent["w"], [[2]])
        np.testing.assert_allclose(inconsistent["mse"], [[1]])
        under = solve_linear_system([[1, 1]], [[2]])
        self.assertEqual(under["status"], "many exact")
        np.testing.assert_allclose(under["w"], [[1], [1]])

    def test_ridge_forms_agree_for_polynomial_multioutput(self):
        arguments = dict(X=[[0], [1], [2]], Y=[[1, 2], [4, 3], [9, 5]], X_test=[[3]],
                         model_type="Polynomial", order=3, regularization="Ridge", ridge_lambda=.5,
                         target_mode="Numeric regression")
        primal = fit_regression_model(**arguments, ridge_form="primal form")
        dual = fit_regression_model(**arguments, ridge_form="dual form")
        np.testing.assert_allclose(primal["w"], dual["w"], atol=1e-12)
        np.testing.assert_allclose(primal["diagnostics"]["objective_gradient"], 0, atol=1e-10)

    def test_encoded_class_targets_and_unseen_label(self):
        result = fit_regression_model(X=[[0], [1]], Y=[[1, 0], [0, 1]],
            X_test=[[1]], Y_test=[[1]], target_mode="Auto")
        self.assertEqual(result["target_mode"], "One-hot classification")
        np.testing.assert_allclose(result["test_target_matrix"], [[0, 1]])
        self.assertEqual(result["test_class"].item(), 1)
        with self.assertRaises(ValueError):
            fit_regression_model(X=[[0], [1]], Y=[["low"], ["high"]],
                X_test=[[2]], Y_test=[["unknown"]], target_mode="One-hot classification")

    def test_cache_persists_between_instances_and_retains_precision_and_labels(self):
        first, second = NotebookCache(), NotebookCache()
        entry_id = first.save("fractions", [[1/3, 1e-12]])
        np.testing.assert_array_equal(second.load(entry_id), [[1/3, 1e-12]])
        first.save("fractions", [[2/3]])
        np.testing.assert_array_equal(second.load("fractions"), [[2/3]])
        first.save("labels", [["low"], ["high"]])
        np.testing.assert_array_equal(second.load("labels"), [["low"], ["high"]])
        self.assertEqual(second.text("labels"), "[['low'], ['high']]")
        second.set_limit(2)
        self.assertEqual(first.limit, 2)
        self.assertEqual(len(first.table()), 2)
        with self.assertRaises(KeyError):
            first.load(entry_id)
        latest_id = second.table().iloc[0]["id"]
        self.assertEqual(first.remove(latest_id), 1)
        second.clear()
        self.assertTrue(first.table().empty)

    def test_empty_cache_notebook_is_safe_and_bootstrap_works_from_root(self):
        path = FOLDER / "09_cache.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        namespace = {}
        with contextlib.chdir(ROOT), patch("IPython.display.display"), patch("notebook_tools.display"), contextlib.redirect_stdout(io.StringIO()):
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), namespace)
        self.assertTrue(namespace["cache"].table().empty)
        self.assertIsNone(namespace["selected"])

    def test_bad_inputs_and_invalid_limits_raise(self):
        with self.assertRaises(ValueError):
            matrix("1 2; 3")
        with self.assertRaises(ValueError):
            solve_linear_system([[1]], [[2]], "unknown")
        details, inverses = inspect_matrix([[1, 2], [2, 4]])
        self.assertEqual(details["inverse"], "no")
        self.assertEqual(inverses, [])
        cache = NotebookCache()
        for limit in (0, 1001, 2.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                cache.set_limit(limit)

    def test_reported_onehot_case_with_all_optional_target_forms(self):
        arguments = dict(X="1 3 -2;-4 0 -1;3 1 8;2 1 6;8 4 6", Y="1;1;2;3;3",
                         X_test="1 -2 4", target="onehot", bias=False)
        for absent in (None, "None", " none ", "", " ", "null"):
            with self.subTest(Y_test=absent):
                result = fit_model(**arguments, Y_test=absent)
                self.assertEqual(result["class_labels"], [1, 2, 3])
                np.testing.assert_array_equal(result["target_matrix"],
                    [[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
                self.assertEqual(result["test_class"].item(), 2)
                self.assertEqual(result["test_pred"].shape, (1, 3))
                self.assertIsNone(result["test_mse"])
        scored = fit_model(**arguments, Y_test="2")
        np.testing.assert_array_equal(scored["test_target_matrix"], [[0,1,0]])
        self.assertIsNotNone(scored["test_mse"])
        legacy = fit_regression_model(arguments["X"], arguments["Y"], arguments["X_test"], "None",
            model_type="Linear", order=2, include_bias=False, target_mode="One-hot classification",
            regularization="None", ridge_lambda=1, ridge_form="auto")
        self.assertEqual(legacy["test_class"].item(), 2)

    def test_invalid_test_inputs_remain_errors_and_required_labels_are_preserved(self):
        for invalid in ("1 2;3", "nan", "1/0"):
            with self.subTest(X_test=invalid), self.assertRaises(ValueError):
                fit_model("0;1", "1;3", invalid)
        with self.assertRaises(ValueError):
            fit_model("0;1", "low;high", "2", "unknown", target="onehot")
        with self.assertRaises(ValueError):
            fit_model("0;1", "1;3", None, "5")
        # A required target column can still contain a real class named None.
        classified = fit_model("0;1", "None;high", "0", [["None"]], target="onehot")
        self.assertEqual(classified["class_labels"], ["None", "high"])

    def test_short_options_and_polynomial_bias_are_honored(self):
        no_constant = fit_model("1;2;3", "2;4;6", "4", model="poly", target="numeric", bias=False, degree=1)
        self.assertEqual(no_constant["model_matrix"].shape, (3,1))
        np.testing.assert_allclose(no_constant["test_pred"], [[8]])
        constant = fit_model("1;2;3", "3;5;7", "4", model="POLYNOMIAL", target="Numeric regression", bias=True, degree=1)
        self.assertEqual(constant["model_matrix"].shape, (3,2))
        np.testing.assert_allclose(constant["test_pred"], [[9]])
        no_bias_string = fit_model("1;2", "2;4", bias="False")
        self.assertEqual(no_bias_string["model_matrix"].shape[1], 1)
        for degree in (0, -1, 1.5, True, 100):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                fit_model("0 1;1 2", "1;2", model="poly", degree=degree)
        for ridge in (-1, float("nan"), True):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                fit_model("0;1", "1;3", ridge=ridge)
        primal = fit_model("0;1", "1;3", ridge="1/2", form="primal")
        dual = fit_model("0;1", "1;3", ridge="1/2", form="dual")
        np.testing.assert_allclose(primal["w"], dual["w"])

    def test_single_multifeature_prediction_list_is_a_row(self):
        result = fit_model([[1,0],[0,1],[1,1]], [2,3,5], [2,1], target="numeric", bias=False)
        np.testing.assert_allclose(result["test_pred"], [[7]])
        self.assertEqual(result["test_pred"].shape, (1,1))

    def test_optional_inputs_coordinates_and_scalar_validation(self):
        for absent in (None, "None", "", "null"):
            self.assertIsNone(optional_matrix(absent))
            self.assertIsNone(number(absent, optional=True))
            self.assertIsNone(integer(absent, optional=True))
            self.assertIsNone(variable_names(absent))
        self.assertEqual(number("1/3"), 1/3)
        self.assertEqual(coordinates("1/3,2"), [1/3, 2])
        self.assertEqual(coordinates("pi/2,0", exact=True), ["pi/2", "0"])
        self.assertEqual(coordinates(3, exact=True), [3])
        self.assertEqual(coordinates([]), [])
        self.assertEqual(variable_names("x1, x2"), ["x1", "x2"])
        self.assertIsNone(variable_names([]))
        self.assertFalse(boolean("False"))
        for invalid in (True, float("inf"), "1/0"):
            with self.subTest(number=invalid), self.assertRaises(ValueError):
                number(invalid)
        with self.assertRaises(ValueError):
            integer(1.5)

    def test_optional_fields_and_other_calculator_branches_in_notebook_cells(self):
        scaled = self.workflow("01_array.ipynb", 9, 10, data="18", first="10", second="30",
                               reference="None", missing_marker="None", inverse="False")
        np.testing.assert_allclose(scaled["preparation"]["result"], [[.4]])
        zscore = self.workflow("01_array.ipynb", 9, 10, method="Z-score standardization",
                              data="8;11;14", first="11", second="3", reference="")
        np.testing.assert_allclose(zscore["preparation"]["result"], [[-1],[0],[1]])
        inverse = self.workflow("01_array.ipynb", 9, 10, data="0.4", first="10", second="30", inverse=True)
        np.testing.assert_allclose(inverse["preparation"]["result"], [[18]])
        imputed = self.workflow("01_array.ipynb", 9, 10, method="Mean imputation", data="2;4;12;-1", missing_marker="-1")
        np.testing.assert_allclose(imputed["preparation"]["result"], [[2],[4],[12],[6]])
        summaries = self.workflow("03_regression.ipynb", 10, 11, rows="None", penalty="1/2")
        self.assertNotIn("summary_prediction", summaries)
        preview = self.workflow("03_regression.ipynb", 13, 14, points="None", d="3", power="2", constant=False)
        self.assertEqual(preview["features"]["count"], 9)
        self.assertIsNone(preview["features"]["transformed"])
        constant = self.workflow("05_gradient.ipynb", 6, 7, outputs="-0.2;-0.8;-0.1", wrt="None", point=[], interpretation="Class scores")
        self.assertEqual(np.argmax(constant["function"]["value"]), 2)
        vector = self.workflow("05_gradient.ipynb", 6, 7, outputs="x1**2+x2;x1*x2", wrt="x1,x2,x3", point="2,-1,7")
        np.testing.assert_allclose(vector["function"]["jacobian_at_point"], [[4,1,0],[-1,2,0]])
        symbolic = self.workflow("05_gradient.ipynb", 14, 15, at=None)
        self.assertIsNone(symbolic["matrix_derivative"]["derivative_at_point"])
        stationary = self.workflow("05_gradient.ipynb", 11, 12, f="x^3", candidate="0", min_wrt="x")
        self.assertFalse(stationary["minimum"]["is_minimum"])
        children = self.workflow("06_decision_tree.ipynb", 3, 4, class_counts="6 4 0;2 1 5", parent_first="False")
        self.assertAlmostEqual(children["classification_summaries"]["gini"]["weighted_impurity"], .5027777777777778)
        split = self.workflow("06_decision_tree.ipynb", 6, 7, threshold="None", find_best="True")
        self.assertIn("best_split", split)
        clusters = self.workflow("07_kmeans.ipynb", 3, 4, max_iter="None")
        self.assertTrue(clusters["final"]["converged"])
        probability = self.workflow("08_probability.ipynb", 3, 4, comparison="between", boundary="0", upper="3")
        self.assertAlmostEqual(probability["distribution"]["event_probability"], .75)
        complement = self.workflow("08_probability.ipynb", 6, 7, event_occurred="False")
        np.testing.assert_allclose(complement["bayes"]["posterior"], [45/157,112/157])
        events = self.workflow("08_probability.ipynb", 9, 10, p_a="1/2", p_b="2/5", known="7/10", known_kind="union")
        self.assertAlmostEqual(events["events"]["intersection"], .2)
        density = self.workflow("08_probability.ipynb", 12, 13, pdf="1/2", support=(0,2), interval=(1,3))
        self.assertAlmostEqual(density["density"]["probability"], .5)
        bag = self.workflow("08_probability.ipynb", 15, 16, event="between", count="0", max_count="1")
        self.assertAlmostEqual(bag["sampling"]["event_probability"], 14/15)
        existing_count = len(NotebookCache().table())
        untouched = self.workflow("09_cache.ipynb", 11, 12, new_limit="None", remove_id="", clear_cache="False")
        self.assertEqual(len(untouched["cache"].table()), existing_count)

    def test_every_matrix_operation_in_notebook_cells(self):
        examples = [
            ("Dot product", "2;-1;3", "0;4;1", [[-1]]),
            ("Matrix product", "1 2;3 4", "2;1", [[4],[10]]),
            ("Add", "1 2;3 4", "4 3;2 1", [[5,5],[5,5]]),
            ("Subtract", "1 2;3 4", "4 3;2 1", [[-3,-1],[1,3]]),
            ("Elementwise product", "1 2;3 4", "4 3;2 1", [[4,6],[6,4]]),
            ("Trace", "1 2;3 4", None, [[5]]),
        ]
        for operation, a, b, answer in examples:
            with self.subTest(operation=operation):
                result = self.workflow("01_array.ipynb", 6, 7, operation=operation, A=a, B=b)
                np.testing.assert_allclose(result["calculation_result"], answer)

    def test_other_workflows_do_not_overwrite_regression_fit_settings(self):
        book = json.loads((FOLDER / "03_regression.ipynb").read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in book["cells"]}
        namespace = {}
        with contextlib.chdir(FOLDER), patch("IPython.display.display"), contextlib.redirect_stdout(io.StringIO()):
            for cell_id in ("03-001", "03-003"):
                exec("".join(cells[cell_id]["source"]), namespace)
            namespace.update(bias=False, degree=3)
            for cell_id in ("03-013", "03-014"):
                exec("".join(cells[cell_id]["source"]), namespace)
        self.assertFalse(namespace["bias"])
        self.assertEqual(namespace["degree"], 3)

    def test_helper_imports_project_utilities_without_running_notebook_setup(self):
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import notebook_tools, CommonUtils, OneHotLinearClassification, RegressionUtils; "
            "from pathlib import Path; "
            "assert Path(CommonUtils.__file__).resolve().parent == Path(sys.argv[2]).resolve(); "
            "result = notebook_tools.fit_model('0;1', '1;3', '2'); "
            "assert abs(result['test_pred'].item() - 5) < 1e-10; "
            "print('Project utilities found')"
        )
        for working_directory in (FOLDER, Path(self.temporary.name)):
            with self.subTest(working_directory=working_directory):
                result = subprocess.run([sys.executable, "-c", program, str(FOLDER), str(ROOT)],
                    cwd=working_directory, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Project utilities found", result.stdout)

    def test_dynamic_probability_solver_notebook_inputs(self):
        solved = self.workflow("08_probability.ipynb", 21, 22,
                               statements="a=.4; a*b=.1; a+b=.7; b=???", show_related=True)
        self.assertAlmostEqual(float(solved["event_solution"]["results"][0]["value"]), .4)
        np.testing.assert_allclose(solved["cache"].load("P(B)"), [[.4]])
        conditional = self.workflow("08_probability.ipynb", 21, 22,
                                    statements="P(A*B)=1/10; P(A|B)=1/4; P(B)=???")
        self.assertAlmostEqual(float(conditional["event_solution"]["results"][0]["value"]), .4)
        bounds = self.workflow("08_probability.ipynb", 21, 22,
                               statements="a=.4; b=???")
        answer = bounds["event_solution"]["results"][0]
        self.assertEqual(answer["status"], "underdetermined")
        self.assertEqual((answer["lower"], answer["upper"]), (0,1))
        third = self.workflow("08_probability.ipynb", 21, 22,
                              statements="a*c=.12; a|c=.3; c=???", show_related=True)
        self.assertAlmostEqual(float(third["event_solution"]["results"][0]["value"]), .4)
        np.testing.assert_allclose(third["cache"].load("P(C)"), [[.4]])
        triple = self.workflow("08_probability.ipynb", 21, 22,
                               statements="a*b*c=.05; b*c=.15; a|(b*c)=???")
        self.assertAlmostEqual(float(triple["event_solution"]["results"][0]["value"]), 1/3)


if __name__ == "__main__":
    unittest.main()
