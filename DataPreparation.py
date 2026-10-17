"""Column-wise transforms fitted on observed reference data or supplied statistics."""

import numpy as np

from CommonUtils import ensure_2d


PREPARATION_METHODS = ("Min-max scaling", "Z-score standardization", "Mean imputation")


def descriptive_statistics(data, convention="Median of halves"):
    """Per-column quartiles; an odd middle observation is excluded from halves."""
    values = ensure_2d(data)
    if convention not in {"Median of halves", "Linear interpolation"}:
        raise ValueError("Choose Median of halves or Linear interpolation.")
    ordered = np.sort(values, axis=0)
    n = len(ordered)
    if convention == "Median of halves":
        q1 = np.median(ordered[:n//2], axis=0) if n > 1 else ordered[0]
        q3 = np.median(ordered[(n+1)//2:], axis=0) if n > 1 else ordered[0]
    else:
        q1, q3 = np.quantile(ordered, [.25, .75], axis=0, method="linear")
    iqr = q3-q1
    return {"count": n, "minimum": ordered[0], "q1": q1, "median": np.median(ordered, axis=0),
            "q3": q3, "maximum": ordered[-1], "iqr": iqr, "mean": np.mean(ordered, axis=0),
            "population_std": np.std(ordered, axis=0), "lower_fence": q1-1.5*iqr,
            "upper_fence": q3+1.5*iqr, "sorted": ordered}


def _column_parameters(value, columns, name):
    values = ensure_2d(value).reshape(-1)
    if values.size == 1:
        return np.full(columns, values[0])
    if values.size != columns:
        raise ValueError(f"{name} must contain one value or one value per feature column.")
    return values


def prepare_data(data, method, reference=None, first=None, second=None,
                 output_range=(0, 1), inverse=False, missing_marker=None):
    if method not in PREPARATION_METHODS:
        raise ValueError(f"Unknown preparation method: {method}.")
    values = ensure_2d(data, allow_missing=True).copy()
    source = values.copy() if reference is None else ensure_2d(reference, allow_missing=True).copy()
    if source.shape[1] != values.shape[1]:
        raise ValueError("Reference data must have the same feature columns as the input.")
    if missing_marker is not None:
        if not np.isfinite(missing_marker):
            raise ValueError("The missing-value marker must be finite.")
        values[values == missing_marker] = np.nan
        source[source == missing_marker] = np.nan
    columns = values.shape[1]
    missing = np.sum(np.isnan(values), axis=0)
    observed = np.sum(~np.isnan(source), axis=0)
    supplied = first is not None
    if method != "Mean imputation" and supplied != (second is not None):
        raise ValueError("Supply both transform parameters, or leave both empty to fit them.")
    if not supplied and np.any(observed == 0):
        raise ValueError("Every reference column needs at least one observed value.")
    if inverse and method == "Mean imputation":
        raise ValueError("Mean imputation cannot be reversed.")
    if inverse and reference is None and not supplied:
        raise ValueError("Inverse scaling needs original reference data or supplied parameters.")

    if method == "Mean imputation":
        means = _column_parameters(first, columns, "Mean") if supplied else np.nanmean(source, axis=0)
        result = np.where(np.isnan(values), means, values)
        return {"result": ensure_2d(result), "first": means, "second": None,
                "missing": missing, "observed": observed}

    if np.any(np.isnan(values)):
        raise ValueError("Impute missing input values before scaling them.")
    if method == "Min-max scaling":
        minimum = _column_parameters(first, columns, "Minimum") if supplied else np.nanmin(source, axis=0)
        maximum = _column_parameters(second, columns, "Maximum") if supplied else np.nanmax(source, axis=0)
        span = maximum - minimum
        if np.any(span <= 0):
            raise ValueError("Each feature maximum must exceed its minimum; constant features cannot be min-max scaled.")
        low, high = np.asarray(output_range, dtype=float).reshape(-1)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("The output range needs finite, increasing endpoints.")
        result = minimum + (values - low) * span / (high - low) if inverse else low + (values - minimum) * (high - low) / span
        params = minimum, maximum
    else:
        mean = _column_parameters(first, columns, "Mean") if supplied else np.nanmean(source, axis=0)
        std = _column_parameters(second, columns, "Standard deviation") if supplied else np.nanstd(source, axis=0)
        if np.any(std <= 0):
            raise ValueError("Each standard deviation must be positive; constant features cannot be standardized.")
        result = values * std + mean if inverse else (values - mean) / std
        params = mean, std
    return {"result": ensure_2d(result), "first": params[0], "second": params[1],
            "missing": missing, "observed": observed}


def matrix_calculation(A, operation, B=None):
    A = ensure_2d(A)
    if operation == "Trace":
        if A.shape[0] != A.shape[1]:
            raise ValueError("Trace requires a square matrix.")
        return ensure_2d([[np.trace(A)]])
    B = ensure_2d(B)
    if operation == "Dot product":
        if min(A.shape) != 1 or min(B.shape) != 1 or A.size != B.size:
            raise ValueError("Dot product requires two row or column vectors of equal length.")
        return ensure_2d([[A.reshape(-1) @ B.reshape(-1)]])
    if operation == "Matrix product":
        if A.shape[1] != B.shape[0]:
            raise ValueError("For A @ B, A's columns must equal B's rows.")
        return ensure_2d(A @ B)
    if operation in {"Add", "Subtract", "Elementwise product"}:
        if A.shape != B.shape:
            raise ValueError("These operations require matrices with identical shapes.")
        result = {"Add": np.add, "Subtract": np.subtract, "Elementwise product": np.multiply}[operation](A, B)
        return ensure_2d(result)
    raise ValueError(f"Unknown matrix operation: {operation}.")
