"""Evaluate supplied scalar/vector functions and their symbolic derivatives."""

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain
from MatrixCalculus import exact_coordinate


def parse_functions(raw_text, variables=None):
    pieces = [part.strip() for line in raw_text.splitlines() for part in line.split(";") if part.strip()]
    if not pieces:
        raise ValueError("Enter at least one scalar expression; separate vector outputs with semicolons or new lines.")
    supplied = [name.strip() for name in variables] if variables is not None else None
    if supplied is not None and (not supplied or any(not name.isidentifier() for name in supplied) or len(set(supplied)) != len(supplied)):
        raise ValueError("Variable names must be distinct identifiers.")
    locals_dict = {name: sp.Symbol(name) for name in supplied} if supplied is not None else {}
    expressions = [sp.sympify(part.replace("^", "**"), locals=locals_dict) for part in pieces]
    if any(not isinstance(expression, sp.Expr) for expression in expressions):
        raise ValueError("Each output must be a scalar expression.")
    free = set().union(*(expression.free_symbols for expression in expressions))
    symbols = [locals_dict[name] for name in supplied] if supplied is not None else sorted(free, key=lambda item: item.name)
    if free - set(symbols):
        raise ValueError("Variable order must include every symbol used in the expressions.")
    return expressions, symbols


def _finite_matrix(expression):
    try:
        result = np.asarray(expression.evalf().tolist(), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("The function or derivatives are undefined or not real at this point.") from None
    if not np.all(np.isfinite(result)):
        raise ValueError("The function or derivatives are not finite at this point.")
    return result


def analyze_functions(raw_text, point, variables=None):
    expressions, symbols = parse_functions(raw_text, variables)
    values = np.asarray(point, dtype=float).reshape(-1)
    if values.size != len(symbols) or not np.all(np.isfinite(values)):
        raise ValueError(f"Enter {len(symbols)} finite coordinate(s) in the displayed variable order.")
    function = sp.Matrix(expressions)
    substitutions = dict(zip(symbols, values))
    jacobian = function.jacobian(symbols) if symbols else sp.zeros(len(expressions), 0)
    result = {"expressions": expressions, "variables": [str(symbol) for symbol in symbols],
              "value": _finite_matrix(function.subs(substitutions)), "jacobian": jacobian,
              "jacobian_at_point": _finite_matrix(jacobian.subs(substitutions)),
              "gradient": None, "hessian": None, "hessian_at_point": None}
    if len(expressions) == 1:
        hessian = sp.hessian(expressions[0], symbols) if symbols else sp.zeros(0, 0)
        result.update({"gradient": jacobian.T, "hessian": hessian,
                       "hessian_at_point": _finite_matrix(hessian.subs(substitutions))})
    dependence = []
    for expression in expressions:
        if not symbols:
            dependence.append("Strictly linear" if expression == 0 else "Affine constant")
        elif expression.is_polynomial(*symbols):
            polynomial = sp.Poly(expression, *symbols)
            if polynomial.total_degree() <= 1:
                offset = polynomial.coeff_monomial(1)
                dependence.append("Strictly linear" if offset == 0 else "Affine with offset")
            else:
                dependence.append("Nonlinear")
        else:
            dependence.append("Not a polynomial")
    result["dependence"] = dependence
    return result


def check_minimum(raw_text, point, variables=None):
    """Classify a supplied unconstrained point without equating stationarity to a minimum.

    Exact signs decide the tests. A singular multivariable Hessian generally
    remains inconclusive; stationary convex quadratics have a global certificate.
    """
    expressions, old_symbols = parse_functions(raw_text, variables)
    if len(expressions) != 1 or not old_symbols:
        raise ValueError("Enter one scalar function with at least one variable (specify variables for a constant function).")
    if len(old_symbols) > 12:
        raise ValueError("Check at most 12 variables at once.")
    symbols = [sp.Symbol(symbol.name, real=True) for symbol in old_symbols]
    expression = expressions[0].xreplace(dict(zip(old_symbols, symbols)))
    coordinates = np.asarray(point, dtype=object).reshape(-1)
    if len(coordinates) != len(symbols):
        raise ValueError(f"Enter {len(symbols)} coordinate(s) in the displayed order.")
    try:
        exact_point = [exact_coordinate(value) for value in coordinates]
        if any(not value.is_finite for value in exact_point):
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("Coordinates must be finite numbers, fractions, or numeric expressions such as pi/2.") from None
    substitutions = dict(zip(symbols, exact_point))
    function_value = _finite_matrix(sp.Matrix([expression.subs(substitutions)])).item()
    gradient = sp.Matrix([sp.diff(expression, symbol) for symbol in symbols])
    hessian = sp.hessian(expression, symbols)
    grad_at = gradient.subs(substitutions).applyfunc(sp.simplify)
    hess_at = hessian.subs(substitutions).applyfunc(sp.simplify)
    result = {"expression": expression, "variables": [s.name for s in symbols],
              "value": function_value, "gradient": gradient, "hessian": hessian,
              "gradient_at_point": None, "hessian_at_point": None, "eigenvalues": None,
              "is_minimum": None, "classification": "Inconclusive", "global_minimum": False,
              "stationary": None, "reason": "", "higher_derivative_order": None}
    try:
        if expression.has(sp.Piecewise, sp.Max, sp.Min, sp.floor, sp.ceiling, sp.sign, sp.Heaviside):
            raise ValueError("A smooth domain needs a separate check for this function.")
        result["gradient_at_point"] = _finite_matrix(grad_at)
        result["hessian_at_point"] = _finite_matrix(hess_at)
        if len(symbols) == 1:
            # Derivatives at a boundary or discontinuity cannot use the interior test.
            for term in (expression, gradient[0], hessian[0, 0]):
                domain = continuous_domain(term, symbols[0], sp.S.Reals)
                if domain.interior.contains(exact_point[0]) is not sp.S.true:
                    raise ValueError("The point is not inside a verified smooth domain.")
    except (ValueError, NotImplementedError):
        result["reason"] = "The smooth interior-point derivative test could not be verified here. Nondifferentiable or domain-boundary minima need a separate test."
        return result
    result["eigenvalues"] = np.linalg.eigvalsh(result["hessian_at_point"])
    if any(entry.is_zero is False for entry in grad_at):
        result.update(is_minimum=False, stationary=False, classification="Not a minimum",
                      reason="The gradient is nonzero, so this smooth interior point is not a local minimum.")
        return result
    if any(entry.is_zero is None for entry in grad_at):
        result["reason"] = "The gradient could not be verified as exactly zero."
        return result
    result["stationary"] = True
    constant_hessian = not hessian.free_symbols
    if constant_hessian and hess_at.is_positive_semidefinite is True:
        strict = hess_at.is_positive_definite is True
        result.update(is_minimum=True, global_minimum=True,
                      classification="Strict global minimum" if strict else "Global minimum (not strict)",
                      reason="The gradient is zero and the constant Hessian is positive semidefinite, so this quadratic or affine function is globally convex.")
    elif hess_at.is_positive_definite is True:
        result.update(is_minimum=True, classification="Strict local minimum",
                      reason="The gradient is zero and the Hessian is positive definite.")
    elif hess_at.is_negative_definite is True:
        result.update(is_minimum=False, classification="Strict local maximum",
                      reason="The gradient is zero and the Hessian is negative definite.")
    elif hess_at.is_indefinite is True:
        result.update(is_minimum=False, classification="Saddle point",
                      reason="The gradient is zero and the Hessian has both positive and negative directions.")
    elif len(symbols) == 1:
        if sp.simplify(expression - expression.subs(substitutions)) == 0:
            result.update(is_minimum=True, global_minimum=True, classification="Global minimum (not strict)",
                          reason="The function is constant, so every point is both a minimum and a maximum.")
        else:
            for order in range(3, 13):
                derivative = sp.diff(expression, symbols[0], order)
                sign = sp.simplify(derivative.subs(substitutions))
                if sign == 0:
                    continue
                if sign.is_positive is not True and sign.is_negative is not True:
                    break
                try:
                    if continuous_domain(derivative, symbols[0], sp.S.Reals).interior.contains(exact_point[0]) is not sp.S.true:
                        break
                except NotImplementedError:
                    break
                result["higher_derivative_order"] = order
                if order % 2:
                    result.update(is_minimum=False, classification="Neither minimum nor maximum",
                                  reason=f"The first nonzero derivative has odd order {order}; the function crosses its value at the point.")
                else:
                    positive = sign.is_positive is True
                    result.update(is_minimum=positive, classification="Strict local minimum" if positive else "Strict local maximum",
                                  reason=f"The first nonzero derivative has even order {order} and is {'positive' if positive else 'negative'}.")
                break
    elif hess_at.is_negative_semidefinite is True and hess_at != sp.zeros(*hess_at.shape):
        result.update(is_minimum=False, classification="Not a minimum",
                      reason="The Hessian has a negative direction. A singular Hessian does not by itself prove a maximum.")
    if not result["reason"]:
        result["reason"] = "The gradient is zero, but the Hessian test is inconclusive. Higher-order or other analysis is needed; zero derivatives alone do not prove a minimum."
    return result
