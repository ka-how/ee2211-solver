import numpy as np

from CommonUtils import ensure_2d, format_number, pretty_print_array, print_value


CLASSIFICATION_MEASURES = ("gini", "entropy", "misclassification")


def class_probabilities(class_counts):
    counts = np.asarray(class_counts, dtype=float).reshape(-1)
    if not np.all(np.isfinite(counts)):
        raise ValueError("Class counts must be finite.")
    if np.any(counts < 0):
        raise ValueError("Class counts cannot be negative.")
    total = np.sum(counts)
    if total <= 0:
        raise ValueError("A node must contain at least one sample.")
    return counts / total


def classification_impurity(class_counts, measure="gini"):
    probabilities = class_probabilities(class_counts)
    measure = measure.lower()
    if measure == "gini":
        return 1.0 - np.sum(np.square(probabilities))
    if measure == "entropy":
        nonzero = probabilities[probabilities > 0]
        return -np.sum(nonzero * np.log2(nonzero))
    if measure == "misclassification":
        return 1.0 - np.max(probabilities)
    raise ValueError(f"Unknown impurity measure: {measure}.")


def classification_split_details(class_counts_by_node, measure="gini", includes_parent=False):
    """Weight disjoint child nodes by their share of the parent samples.

    With includes_parent=True the first row is the parent and must equal the
    column-wise sum of the remaining rows. It is never counted as a child.
    """
    counts = ensure_2d(class_counts_by_node)
    if np.any(counts < 0):
        raise ValueError("Class counts cannot be negative.")
    if includes_parent:
        if counts.shape[0] < 2:
            raise ValueError("Enter a parent row followed by at least one child row.")
        parent_counts, counts = counts[0], counts[1:]
        if not np.allclose(parent_counts, np.sum(counts, axis=0), atol=1e-10, rtol=0):
            raise ValueError("Parent class counts must equal the sum of the child rows in each class.")
    else:
        parent_counts = np.sum(counts, axis=0)
    node_sizes = np.sum(counts, axis=1)
    parent_impurity = float(classification_impurity(parent_counts, measure))
    # An unused branch has weight zero and contributes no impurity.
    impurities = np.array([
        classification_impurity(row, measure) if size else 0.0
        for row, size in zip(counts, node_sizes)
    ], dtype=float)
    weights = node_sizes / np.sum(node_sizes)
    contributions = weights * impurities
    weighted_impurity = float(np.sum(contributions))
    return {
        "child_counts": counts,
        "node_sizes": node_sizes,
        "impurities": impurities,
        "weights": weights,
        "weighted_contributions": contributions,
        "weighted_impurity": weighted_impurity,
        "parent_impurity": parent_impurity,
        "gain": parent_impurity - weighted_impurity,
    }


def classification_impurity_summary(class_counts_by_node, measure="gini", includes_parent=False):
    details = classification_split_details(class_counts_by_node, measure, includes_parent)
    return details["node_sizes"], details["impurities"], details["weighted_impurity"]


def regression_node_mse(y):
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("A regression-tree node must contain at least one sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Regression targets must be finite.")
    mean = float(np.mean(values))
    mse = float(np.mean(np.square(values - mean)))
    return mean, mse


def regression_split_summary(y_left, y_right):
    left = np.asarray(y_left, dtype=float).reshape(-1)
    right = np.asarray(y_right, dtype=float).reshape(-1)
    left_mean, left_mse = regression_node_mse(left)
    right_mean, right_mse = regression_node_mse(right)
    total = left.size + right.size
    overall_mse = float((left.size * left_mse + right.size * right_mse) / total)
    return {
        "left_size": left.size,
        "right_size": right.size,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_mse": left_mse,
        "right_mse": right_mse,
        "overall_mse": overall_mse,
    }


def regression_threshold_summary(x, y, threshold):
    x_values, y_values = _prepare_xy(x, y)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    left_mask = x_values < threshold
    right_mask = x_values >= threshold
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("Threshold must leave at least one sample on each side.")

    root_mean, root_mse = regression_node_mse(y_values)
    split = regression_split_summary(y_values[left_mask], y_values[right_mask])
    split.update(
        {
            "threshold": float(threshold),
            "root_mean": root_mean,
            "root_mse": root_mse,
            "x_sorted": x_values,
            "y_sorted": y_values,
        }
    )
    return split


def find_best_regression_split(x, y):
    x_values, y_values = _prepare_xy(x, y)
    unique_x = np.unique(x_values)
    if unique_x.size < 2:
        raise ValueError("At least two unique x values are needed to split.")

    # Avoid overflowing the sum or rounding the midpoint down to the left value.
    thresholds = unique_x[:-1] / 2.0 + unique_x[1:] / 2.0
    thresholds = np.where(thresholds <= unique_x[:-1], unique_x[1:], thresholds)
    summaries = [regression_threshold_summary(x_values, y_values, threshold) for threshold in thresholds]
    best = min(summaries, key=lambda item: item["overall_mse"])
    return best, summaries


def print_classification_impurity_summary(class_counts_by_node, measure="gini", includes_parent=False):
    measures = CLASSIFICATION_MEASURES if measure == "all" else (measure,)
    pretty_print_array("class_counts_by_node", class_counts_by_node)
    for item in measures:
        details = classification_split_details(class_counts_by_node, item, includes_parent)
        pretty_print_array(f"{item}_per_node", details["impurities"].reshape(-1, 1), show_python=False)
        pretty_print_array("node_sizes", details["node_sizes"].reshape(-1, 1), show_python=False)
        pretty_print_array("node_weights", details["weights"].reshape(-1, 1), show_python=False)
        pretty_print_array(f"weighted_{item}_per_node", details["weighted_contributions"].reshape(-1, 1), show_python=False)
        print_value(f"weighted_{item}", details["weighted_impurity"])
        print_value(f"parent_{item}", details["parent_impurity"])
        print_value(f"{item}_gain", details["gain"])
        print("")


def print_regression_threshold_summary(summary):
    print_value("threshold", summary["threshold"])
    print_value("root_mean", summary["root_mean"])
    print_value("root_mse", summary["root_mse"])
    print_value("left_mean", summary["left_mean"])
    print_value("left_mse", summary["left_mse"])
    print_value("right_mean", summary["right_mean"])
    print_value("right_mse", summary["right_mse"])
    print_value("split_overall_mse", summary["overall_mse"])
    print("")


def print_regression_split_candidates(summaries):
    print("threshold | overall_mse")
    print("-----------------------")
    for summary in summaries:
        print(f"{format_number(summary['threshold'])} | {format_number(summary['overall_mse'])}")
    print("")


def _prepare_xy(x, y):
    x_array, y_array = ensure_2d(x), ensure_2d(y)
    if min(x_array.shape) != 1 or min(y_array.shape) != 1:
        raise ValueError("Regression-tree x and y must each be a single row or column of values.")
    x_values = x_array.reshape(-1)
    y_values = y_array.reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError(f"x and y must have the same length, got {x_values.size} and {y_values.size}.")
    if x_values.size < 2:
        raise ValueError("At least two samples are needed.")
    sort_index = np.argsort(x_values, kind="mergesort")
    return x_values[sort_index], y_values[sort_index]
