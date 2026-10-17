"""Model feature descriptions, supplied normal equations, and optimality checks."""

from math import comb

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import ensure_2d, validate_same_sample_count


def polynomial_features_details(n_features, degree, points=None, include_bias=True):
    if int(n_features) != n_features or n_features < 1 or int(degree) != degree or degree < 1:
        raise ValueError("Feature count and polynomial order must be positive integers.")
    count = comb(int(n_features) + int(degree), int(degree)) - (not include_bias)
    if count > 5000:
        raise ValueError(f"This expansion has {count} columns. Reduce the feature count or order to preview at most 5000.")
    transformer = PolynomialFeatures(int(degree), include_bias=include_bias)
    transformer.fit(np.zeros((1, int(n_features))))
    names = transformer.get_feature_names_out([f"x{i + 1}" for i in range(int(n_features))]).tolist()
    transformed = None
    if points is not None:
        points = ensure_2d(points)
        if points.shape[1] != n_features:
            raise ValueError("Points must have the selected number of input feature columns.")
        transformed = transformer.transform(points)
    return {"count": count, "names": names, "powers": transformer.powers_, "transformed": transformed}


def regression_shapes(samples, parameters, outputs=1):
    for value in (samples, parameters, outputs):
        if int(value) != value or value < 1:
            raise ValueError("Sample, parameter, and output counts must be positive integers.")
    return {"design": (samples, parameters), "weights": (parameters, outputs),
            "primal": (parameters, parameters), "dual": (samples, samples),
            "recommended_form": "dual form" if samples < parameters else "primal form"}


def solve_normal_equations(gram, rhs, ridge_lambda=0):
    gram, rhs = ensure_2d(gram), ensure_2d(rhs)
    if gram.shape[0] != gram.shape[1] or rhs.shape[0] != gram.shape[0]:
        raise ValueError("X.T @ X must be square, and X.T @ Y must have the same number of rows.")
    tolerance = max(1, np.linalg.norm(gram, ord=2)) * np.finfo(float).eps * gram.shape[0] * 10
    if not np.allclose(gram, gram.T, rtol=0, atol=tolerance):
        raise ValueError("X.T @ X must be symmetric.")
    gram = (gram + gram.T) / 2
    if np.min(np.linalg.eigvalsh(gram)) < -tolerance:
        raise ValueError("X.T @ X must be positive semidefinite.")
    ridge_lambda = float(ridge_lambda)
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("Lambda must be finite and nonnegative.")
    ordinary_weights = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    rhs_tolerance = max(1, np.linalg.norm(rhs)) * np.finfo(float).eps * gram.shape[0] * 10
    if not np.allclose(gram @ ordinary_weights, rhs, rtol=1e-9, atol=rhs_tolerance):
        raise ValueError("These summaries are inconsistent: X.T @ Y is outside the range of X.T @ X.")
    if ridge_lambda == 0:
        weights = ordinary_weights
    else:
        # Eigenvectors avoid losing tiny penalties when a supplied Gram matrix is singular.
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0)
        coordinates = eigenvectors.T @ rhs
        null = eigenvalues <= np.max(eigenvalues) * np.finfo(float).eps * gram.shape[0]
        coordinates[null] = 0
        weights = eigenvectors @ (coordinates / (eigenvalues + ridge_lambda)[:, None])
    return {"w": weights, "regularized_gram": gram + ridge_lambda * np.eye(gram.shape[0]),
            "normal_residual": (gram + ridge_lambda * np.eye(gram.shape[0])) @ weights - rhs,
            "gram_rank": int(np.linalg.matrix_rank(gram))}


def regression_diagnostics(X, Y, weights, ridge_lambda=0):
    X, Y, weights = ensure_2d(X), ensure_2d(Y), ensure_2d(weights)
    validate_same_sample_count(X, Y)
    if weights.shape != (X.shape[1], Y.shape[1]):
        raise ValueError("Weights must match the model feature columns and target columns.")
    residual = X @ weights - Y
    normal_residual = X.T @ residual
    stationarity = normal_residual + ridge_lambda * weights
    sse = float(np.sum(np.square(residual)))
    return {"gram": X.T @ X, "rhs": X.T @ Y, "residual": residual,
            "normal_residual": normal_residual, "stationarity": stationarity,
            "objective_gradient": 2 * stationarity, "total_sse": sse,
            "sse_per_sample": sse / X.shape[0], "objective": sse + ridge_lambda * float(np.sum(weights ** 2)),
            "shapes": regression_shapes(X.shape[0], X.shape[1], Y.shape[1])}
