"""Notebook display, WebUI-equivalent orchestration, and a shared local cache.

The mathematical implementations live in the repository's calculator modules.
The inspect/solve/fit adapters follow webui.py without importing Streamlit.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import uuid

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.preprocessing import PolynomialFeatures

# Resolve sibling project modules even when only notebooks/ is on Python's path.
# Unlike the working directory, this location is stable across notebook kernels.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_MODULES = ("CommonUtils.py", "OneHotLinearClassification.py", "RegressionUtils.py")
_MISSING_MODULES = [name for name in _PROJECT_MODULES if not (_PROJECT_ROOT / name).is_file()]
if _MISSING_MODULES:
    raise ModuleNotFoundError(
        "Project utility files are missing: " + ", ".join(_MISSING_MODULES)
        + ". Keep notebooks/ inside the complete ee2211-solver repository. "
        "These files are included in the project; they are not Conda packages."
    )
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CommonUtils import (
    add_bias_column, calculate_closed_form_weights, calculate_ridge_weights,
    ensure_2d, format_number, parse_mixed_array, parse_numeric_array, parse_number,
    squared_error_summary, system_type, validate_same_feature_count,
    validate_same_sample_count,
)
from OneHotLinearClassification import prepare_class_targets, validate_onehot_targets
from RegressionUtils import regression_diagnostics


def as_2d_any(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Input must be a nonempty two-dimensional matrix.")
    return arr


def contains_text(value):
    return any(isinstance(item, str) for item in as_2d_any(value).flat)


def matrix(raw, allow_mixed=False, allow_missing=False):
    """Accept the website's matrix strings, nested lists, or NumPy arrays."""
    if isinstance(raw, str):
        if allow_mixed:
            return parse_mixed_array(raw)
        return parse_numeric_array(raw, allow_missing=allow_missing)
    return as_2d_any(raw) if allow_mixed else ensure_2d(raw, allow_missing=allow_missing)


def is_blank(value):
    """Only optional fields reserve empty text, None and null as absent values."""
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def optional_matrix(value, *, allow_mixed=False, allow_missing=False):
    return None if is_blank(value) else matrix(value, allow_mixed, allow_missing)


def number(value, *, optional=False):
    """Read a finite scalar or fraction; optional fields also accept blank/None."""
    if optional and is_blank(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Enter a number, not True or False.")
    return parse_number(value)


def integer(value, *, optional=False):
    result = number(value, optional=optional)
    if result is None:
        return None
    if result != int(result):
        raise ValueError("Enter a whole number.")
    return int(result)


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError("Enter True or False.")


def variable_names(value):
    if is_blank(value):
        return None
    names = re.split(r"[,;\s]+", value.strip()) if isinstance(value, str) else list(value)
    return names or None


def coordinates(value, *, exact=False):
    if is_blank(value) or (not isinstance(value, str) and np.asarray(value).size == 0):
        return []
    if exact:
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
        return [value] if np.ndim(value) == 0 else list(value)
    return matrix(value).reshape(-1).tolist()


def choice(value, aliases, field):
    key = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if key not in aliases:
        raise ValueError(f"Unknown {field}: {value!r}. Choose from {', '.join(sorted(set(aliases.values())))}.")
    return aliases[key]


def model_choice(value):
    return choice(value, {"linear": "Linear", "poly": "Polynomial", "polynomial": "Polynomial"}, "model")


def target_choice(value):
    return choice(value, {"auto": "Auto", "numeric": "Numeric regression", "regression": "Numeric regression",
        "numeric regression": "Numeric regression", "sign": "Sign classification", "sign classification": "Sign classification",
        "onehot": "One-hot classification", "one hot": "One-hot classification",
        "one hot classification": "One-hot classification"}, "target mode")


def form_choice(value):
    return choice(value, {"auto": "auto", "primal": "primal form", "primal form": "primal form",
                          "dual": "dual form", "dual form": "dual form"}, "ridge form")


def show(name, value, *, cache=None, source="", kind="output"):
    """Display a matrix at full stored precision, optionally saving it for reuse."""
    if value is None:
        return
    display(Markdown(f"**{name}**"))
    display(pd.DataFrame(as_2d_any(value)))
    if cache is not None:
        cache.save(name, value, source=source, kind=kind)


def show_metrics(values):
    display(pd.DataFrame({"Value": pd.Series(values, dtype=object)}))


def inspect_matrix(array):
    array = ensure_2d(array)
    rows, cols = array.shape
    rank = int(np.linalg.matrix_rank(array))
    square = rows == cols
    details = {
        "shape": f"{rows} x {cols}", "rank": rank,
        "determinant": format_number(np.linalg.det(array)) if square else "n/a",
        "inverse": "yes" if square and rank == rows else "no",
        "left inverse": "yes" if rank == cols else "no",
        "right inverse": "yes" if rank == rows else "no",
    }
    matrices = []
    if square and rank == rows:
        matrices.append(("inverse", np.linalg.inv(array)))
    if rank == cols:
        matrices.append(("left_inverse", np.linalg.pinv(array)))
    if rank == rows:
        matrices.append(("right_inverse", np.linalg.pinv(array)))
    return details, matrices


def solve_linear_system(X, y, equation="Xw = y"):
    X, y = ensure_2d(X), ensure_2d(y)
    if equation == "Xw = y":
        validate_same_sample_count(X, y)
        coefficients, target = X, y
    elif equation == "wX = y":
        if X.shape[1] != y.shape[1]:
            raise ValueError("X and y must have the same column count for wX = y.")
        coefficients, target = X.T, y.T
    else:
        raise ValueError("Choose Xw = y or wX = y.")
    rows, unknowns = coefficients.shape
    rank = int(np.linalg.matrix_rank(coefficients))
    augmented_rank = int(np.linalg.matrix_rank(np.hstack((coefficients, target))))
    solution = np.linalg.lstsq(coefficients, target, rcond=None)[0]
    predicted = coefficients @ solution
    residual = predicted - target
    mse = np.mean(residual ** 2, axis=0, keepdims=True)
    status = "least squares" if rank != augmented_rank else (
        "unique exact" if rank == unknowns else "many exact"
    )
    def orient(value):
        return value if equation == "Xw = y" else value.T
    return {
        "system": system_type(coefficients), "equations": rows, "unknowns": unknowns,
        "rank_coefficients": rank, "rank_augmented": augmented_rank, "status": status,
        "w": orient(solution), "predicted": orient(predicted),
        "residual": orient(residual), "mse": orient(mse),
        "standard_error": orient(np.sqrt(mse)),
    }


def resolve_target_mode(Y, requested):
    choices = {"Auto", "Numeric regression", "Sign classification", "One-hot classification"}
    if requested not in choices:
        raise ValueError(f"Unknown target mode: {requested}.")
    if requested != "Auto":
        return requested
    if contains_text(Y):
        return "One-hot classification"
    numeric = ensure_2d(Y)
    if numeric.shape[1] == 1 and set(np.unique(numeric)).issubset({-1.0, 1.0}):
        return "Sign classification"
    if numeric.shape[1] > 1 and np.all((numeric == 0) | (numeric == 1)) and np.all(numeric.sum(axis=1) == 1):
        return "One-hot classification"
    return "Numeric regression"


def prepare_test_target_matrix(Y_test, target_mode, class_labels, class_count):
    if target_mode != "One-hot classification":
        return ensure_2d(Y_test)
    arr = as_2d_any(Y_test)
    if arr.shape[1] > 1:
        return validate_onehot_targets(arr)
    if class_labels is None:
        indices = ensure_2d(arr).reshape(-1)
        if np.any(indices != np.floor(indices)) or np.any(indices < 0) or np.any(indices >= class_count):
            raise ValueError("Test labels must be zero-based training class column indices.")
        return np.eye(class_count)[indices.astype(int)]
    labels = arr.reshape(-1).tolist()
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    if any(label not in class_to_index for label in labels):
        raise ValueError("Y_test contains a class label not present in training.")
    return np.eye(len(class_labels))[[class_to_index[label] for label in labels]]


def classify_predictions(prediction, target_mode, class_labels):
    if target_mode == "Sign classification":
        return np.sign(prediction)
    if target_mode == "One-hot classification":
        indices = np.argmax(prediction, axis=1)
        if class_labels is not None:
            return np.asarray(class_labels, dtype=object)[indices].reshape(-1, 1)
        return indices.reshape(-1, 1)
    return None


def fit_regression_model(X, Y, X_test=None, Y_test=None, model_type="Linear", order=2,
                         target_mode="Auto", regularization="None", ridge_lambda=1.0,
                         ridge_form="auto", include_bias=True):
    model_type, target_mode = model_choice(model_type), target_choice(target_mode)
    regularization = choice(regularization, {"none": "None", "ols": "None", "ridge": "Ridge"}, "regularization")
    include_bias = boolean(include_bias)
    X = matrix(X)
    Y = matrix(Y, allow_mixed=target_mode in {"Auto", "One-hot classification"})
    if not is_blank(X_test) and not isinstance(X_test, str) and np.ndim(X_test) == 1 and X.shape[1] > 1:
        X_test = np.asarray(X_test).reshape(1, -1)
    X_test = optional_matrix(X_test)
    Y_test = optional_matrix(Y_test, allow_mixed=target_mode in {"Auto", "One-hot classification"})
    validate_same_sample_count(X, Y)
    if Y_test is not None and X_test is None:
        raise ValueError("Supply X_test when supplying Y_test.")
    poly = None
    if model_type == "Polynomial":
        from RegressionUtils import polynomial_features_details
        order = integer(order)
        polynomial_features_details(X.shape[1], order, include_bias=include_bias)
        poly = PolynomialFeatures(order, include_bias=include_bias)
        design = poly.fit_transform(X)
        model_name, model_label = "P", f"Polynomial order {order}"
    else:
        design = add_bias_column(X) if include_bias else X
        model_name, model_label = "X_model", "Linear + bias" if include_bias else "Linear"
    resolved = resolve_target_mode(Y, target_mode)
    targets, labels = prepare_class_targets(Y) if resolved == "One-hot classification" else (ensure_2d(Y), None)
    # The existing closed-form helper prints its selected form; keep notebook output tidy.
    with contextlib.redirect_stdout(io.StringIO()):
        if regularization == "Ridge":
            ridge_lambda = number(ridge_lambda)
            system, form, weights = calculate_ridge_weights(design, targets, ridge_lambda, form_choice(ridge_form))
            form = f"{form}, lambda={format_number(ridge_lambda)}"
        else:
            system, weights = calculate_closed_form_weights(design, targets)
            form = "closed form"
    train_pred = design @ weights
    sse, mse = squared_error_summary(targets, train_pred)
    test_pred = test_class = test_targets = test_sse = test_mse = None
    if X_test is not None:
        X_test = ensure_2d(X_test)
        validate_same_feature_count(X, X_test)
        test_design = poly.transform(X_test) if poly is not None else (add_bias_column(X_test) if include_bias else X_test)
        test_pred = test_design @ weights
        test_class = classify_predictions(test_pred, resolved, labels)
        if Y_test is not None:
            test_targets = prepare_test_target_matrix(Y_test, resolved, labels, targets.shape[1])
            validate_same_sample_count(X_test, test_targets)
            test_sse, test_mse = squared_error_summary(test_targets, test_pred)
    return {
        "model_label": model_label, "model_matrix_name": model_name, "model_matrix": design,
        "target_mode": resolved, "target_matrix": targets, "class_labels": labels,
        "system": system, "form": form, "w": weights, "train_pred": train_pred,
        "train_class": classify_predictions(train_pred, resolved, labels),
        "sse": np.asarray(sse).reshape(1, -1), "mse": np.asarray(mse).reshape(1, -1),
        "test_pred": test_pred, "test_class": test_class, "test_target_matrix": test_targets,
        "test_sse": None if test_sse is None else np.asarray(test_sse).reshape(1, -1),
        "test_mse": None if test_mse is None else np.asarray(test_mse).reshape(1, -1),
        "diagnostics": regression_diagnostics(design, targets, weights, ridge_lambda if regularization == "Ridge" else 0),
    }


def fit_model(X, Y, X_test=None, Y_test=None, *, model="linear", target="auto",
              bias=True, ridge=0, degree=2, form="auto"):
    """Compact notebook interface: ridge=0 means OLS; positive ridge means a penalty."""
    penalty = number(ridge)
    if penalty < 0:
        raise ValueError("Ridge must be zero (OLS) or a positive penalty.")
    return fit_regression_model(X, Y, X_test, Y_test, model_type=model,
        target_mode=target, include_bias=bias, regularization="None" if penalty == 0 else "Ridge",
        ridge_lambda=penalty, order=degree, ridge_form=form)


class NotebookCache:
    """Persist recent matrices across notebook kernels using standard-library SQLite.

    Each operation opens a short connection so other notebooks see fresh results.
    The notebook cache is separate from the website's saved state.
    """

    def __init__(self, path=None):
        default = Path(__file__).resolve().parent / ".cache" / "matrices.sqlite3"
        self.path = Path(path or os.environ.get("EE2211_NOTEBOOK_CACHE", default)).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value INTEGER)")
            connection.execute("INSERT OR IGNORE INTO settings VALUES ('limit', 50)")
            connection.execute("""CREATE TABLE IF NOT EXISTS matrices (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, name TEXT,
                source TEXT, kind TEXT, shape TEXT, value TEXT, is_text INTEGER,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )""")

    @contextlib.contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _trim(connection):
        connection.execute("""DELETE FROM matrices WHERE seq NOT IN
            (SELECT seq FROM matrices ORDER BY seq DESC
             LIMIT (SELECT value FROM settings WHERE key = 'limit'))""")

    @property
    def limit(self):
        with self._connect() as connection:
            return connection.execute("SELECT value FROM settings WHERE key = 'limit'").fetchone()[0]

    def set_limit(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or not 1 <= limit <= 1000:
            raise ValueError("Cache limit must be an integer from 1 to 1000.")
        with self._connect() as connection:
            connection.execute("UPDATE settings SET value = ? WHERE key = 'limit'", (int(limit),))
            self._trim(connection)

    def save(self, name, value, source="", kind="output"):
        arr = as_2d_any(value)
        is_text = contains_text(arr)
        if not is_text:
            arr = ensure_2d(arr)
        payload = json.dumps(arr.tolist(), ensure_ascii=False, allow_nan=False)
        entry_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO matrices (id, name, source, kind, shape, value, is_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry_id, str(name), str(source), str(kind), f"{arr.shape[0]} x {arr.shape[1]}", payload, int(is_text)),
            )
            self._trim(connection)
        return entry_id

    def table(self):
        with self._connect() as connection:
            return pd.read_sql_query("SELECT id, name, source, kind, shape, created_at FROM matrices ORDER BY seq DESC", connection)

    def load(self, name_or_id):
        with self._connect() as connection:
            row = connection.execute("SELECT value, is_text FROM matrices WHERE id = ? OR name = ? ORDER BY seq DESC LIMIT 1", (name_or_id, name_or_id)).fetchone()
        if row is None:
            raise KeyError(f"No cached matrix named {name_or_id!r}. Run a calculation first.")
        return np.asarray(json.loads(row["value"]), dtype=object if row["is_text"] else float)

    def text(self, name_or_id):
        return repr(self.load(name_or_id).tolist())

    def remove(self, entry_id):
        with self._connect() as connection:
            return connection.execute("DELETE FROM matrices WHERE id = ?", (entry_id,)).rowcount

    def clear(self):
        with self._connect() as connection:
            connection.execute("DELETE FROM matrices")
