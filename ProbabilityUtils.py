"""Probability calculations without silently normalizing invalid distributions."""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb
import re
from sympy.calculus.util import continuous_domain, minimum

from CommonUtils import ensure_2d


@dataclass(frozen=True)
class _ProbabilityExpression:
    name: str
    numerator: tuple
    denominator: tuple | None = None


class _EventParser:
    """Small event grammar; never evaluate the supplied text as Python code."""

    def __init__(self, text, event_names=("a", "b", "c")):
        text = text.strip()
        wrapped = re.fullmatch(r"P\s*\((.*)\)", text, flags=re.IGNORECASE)
        if wrapped:
            text = wrapped.group(1)
        self.text = re.sub(r"\s+", "", text.lower()).replace("∩", "*").replace("∪", "+").replace("¬", "~")
        self.position = 0
        atoms = tuple(product((1, 0), repeat=len(event_names)))
        self.events = {name: tuple(atom[index] for atom in atoms) for index, name in enumerate(event_names)}
        self.events.update({"0": (0,) * len(atoms), "1": (1,) * len(atoms)})
        if not self.text or len(self.text) > 256:
            raise ValueError("Enter an event expression of at most 256 characters using A, B, and C.")

    def take(self, token):
        if self.text.startswith(token, self.position):
            self.position += len(token)
            return True
        return False

    def union(self):
        result = self.intersection()
        while self.take("+"):
            right = self.intersection()
            result = tuple(int(a or b) for a, b in zip(result, right))
        return result

    def intersection(self):
        result = self.unary()
        while self.take("*"):
            right = self.unary()
            result = tuple(a * b for a, b in zip(result, right))
        return result

    def unary(self):
        if self.take("~") or self.take("!"):
            result = tuple(1 - value for value in self.unary())
        elif self.take("("):
            result = self.union()
            if not self.take(")"):
                raise ValueError("Close each event-expression parenthesis; put | between the two events.")
        else:
            if self.position >= len(self.text) or self.text[self.position] not in self.events:
                raise ValueError("Use A, B, or C, * for intersection, + for union, ~ or ' for complement, and | for conditional probability.")
            result = self.events[self.text[self.position]]
            self.position += 1
        while self.take("'"):
            result = tuple(1 - value for value in result)
        return result

    def parse(self):
        left = self.union()
        denominator = None
        numerator = left
        if self.take("|"):
            denominator = self.union()
            numerator = tuple(a * b for a, b in zip(left, denominator))
        if self.position != len(self.text):
            raise ValueError("Unexpected event syntax. Use, for example, P(C), A*B*C, A+B+C, A|(B+C), or ~C.")
        return _ProbabilityExpression("P(" + self.text.upper() + ")", numerator, denominator)


def _probability_value(value):
    try:
        text = str(value).strip()
        percentage = text.endswith("%")
        fraction = Fraction(text[:-1].strip() if percentage else text)
        if percentage:
            fraction /= 100
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError("Enter a probability such as 0.4, 2/5, or 40%; use ??? for an unknown.") from None
    if not 0 <= fraction <= 1:
        raise ValueError("Each known probability must be between 0 and 1.")
    return sp.Rational(fraction.numerator, fraction.denominator)


def _parse_probability_statements(statements):
    if isinstance(statements, dict):
        pairs = [(str(name), value) for name, value in statements.items()]
    elif isinstance(statements, str):
        pairs = []
        for line in statements.replace(";", "\n").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.count("=") != 1:
                raise ValueError("Use one assignment per line, for example a = 0.4 or b = ???.")
            name, value = line.split("=", 1)
            pairs.append((name, value))
    else:
        raise ValueError("Supply probability assignments as text or a dictionary.")
    if not pairs or len(pairs) > 64:
        raise ValueError("Enter between 1 and 64 known probabilities or unknown queries.")
    # Keep the smaller two-event model for existing A/B questions. Mentioning C
    # anywhere, including an unknown query, selects the eight-cell joint model.
    event_names = ("a", "b", "c") if any("c" in name.lower() for name, _ in pairs) else ("a", "b")
    known, queries = [], []
    for name, value in pairs:
        event = _EventParser(name, event_names).parse()
        unknown = value is None or str(value).strip().lower() in {"", "none", "null"} or re.fullmatch(r"\?+", str(value).strip())
        if unknown:
            if event not in queries:
                queries.append(event)
        else:
            known.append((event, _probability_value(value)))
    return known, queries, event_names


def _dot(coefficients, point):
    return sum((coefficient * value for coefficient, value in zip(coefficients, point)), sp.S.Zero)


def _joint_probability_vertices(known, cell_count):
    """Enumerate the vertices of the joint distribution exactly.

    With normalization, equality constraints, and nonnegative cells, each vertex
    is a basic feasible solution. Two events use four cells; three events use
    eight. At most 70 column bases are needed, avoiding numeric tolerances and
    additional packages.
    """
    rows, rhs = [[1] * cell_count], [sp.S.One]
    required_positive = []
    for event, value in known:
        if event.denominator is None:
            rows.append(list(event.numerator))
            rhs.append(value)
        else:
            rows.append([a - value * b for a, b in zip(event.numerator, event.denominator)])
            rhs.append(sp.S.Zero)
            required_positive.append(event.denominator)
    augmented, pivots = sp.Matrix(rows).row_join(sp.Matrix(rhs)).rref()
    if cell_count in pivots:
        raise ValueError("The supplied probabilities are inconsistent; no joint distribution satisfies them.")
    rank = len(pivots)
    coefficients, targets = augmented[:rank, :cell_count], augmented[:rank, cell_count]
    vertices = set()
    for basis in combinations(range(cell_count), rank):
        square = coefficients[:, list(basis)]
        if square.det() == 0:
            continue
        solution = square.inv() * targets
        if any(value < 0 for value in solution):
            continue
        point = [sp.S.Zero] * cell_count
        for index, value in zip(basis, solution):
            point[index] = value
        vertices.add(tuple(point))
    vertices = sorted(vertices)
    if not vertices:
        raise ValueError("The supplied probabilities are inconsistent; no nonnegative joint distribution satisfies them.")
    for denominator in required_positive:
        if not any(_dot(denominator, point) > 0 for point in vertices):
            raise ValueError("A supplied conditional probability is undefined: its conditioning event has probability zero.")
    return vertices, required_positive


def _face_is_allowed(vertices, required_positive):
    # Averaging feasible vertices gives every required event positive mass iff
    # each event has positive mass at some vertex of this face.
    return bool(vertices) and all(any(_dot(event, point) > 0 for point in vertices) for event in required_positive)


def _solve_probability_query(event, vertices, required_positive):
    denominator = event.denominator or (1,) * len(event.numerator)
    positive = [point for point in vertices if _dot(denominator, point) > 0]
    result = {"expression": event.name, "status": "undefined", "value": None,
              "lower": None, "upper": None, "lower_inclusive": False,
              "upper_inclusive": False, "undefined_possible": False}
    if not positive:
        result["undefined_possible"] = True
        return result
    ratios = [_dot(event.numerator, point) / _dot(denominator, point) for point in positive]
    lower, upper = min(ratios), max(ratios)
    lower_face = [point for point in vertices if _dot(event.numerator, point) == lower * _dot(denominator, point)]
    upper_face = [point for point in vertices if _dot(event.numerator, point) == upper * _dot(denominator, point)]
    needed = [*required_positive, denominator]
    undefined_face = [point for point in vertices if _dot(denominator, point) == 0]
    result.update(status="solved" if lower == upper else "underdetermined",
                  value=lower if lower == upper else None, lower=lower, upper=upper,
                  lower_inclusive=_face_is_allowed(lower_face, needed),
                  upper_inclusive=_face_is_allowed(upper_face, needed),
                  undefined_possible=_face_is_allowed(undefined_face, required_positive))
    return result


def solve_probabilities(statements, include_related=False):
    """Solve known probabilities and ??? queries for events A, B, and C.

    Examples: a=.4; a*b=.1; a+b=.7; b=??? and a*b=.1; a|b=.25; b=???.
    * means intersection, + means union, | means conditional probability, and
    ~ or a trailing apostrophe means complement. P(...) wrappers are optional.
    Probabilities are exact decimals, fractions, or percentages. No independence
    is assumed. Underdetermined answers include exact feasible bounds, including
    open endpoints and possible undefined conditionals.
    """
    known, queries, event_names = _parse_probability_statements(statements)
    vertices, required_positive = _joint_probability_vertices(known, 2 ** len(event_names))
    common_text = list(event_names)
    for size in range(2, len(event_names) + 1):
        for group in combinations(event_names, size):
            common_text.extend(("*".join(group), "+".join(group)))
    common_text.extend("~" + name for name in event_names)
    for left, right in permutations(event_names, 2):
        common_text.extend((left + "|" + right, left + "|~" + right))
    common = [_EventParser(text, event_names).parse() for text in common_text]
    requested = queries or common
    results = [_solve_probability_query(event, vertices, required_positive) for event in requested]
    related = [_solve_probability_query(event, vertices, required_positive) for event in common if event not in requested] if include_related else []
    identities = ["P(A+B) = P(A) + P(B) - P(A*B)",
                  "P(A|B) = P(A*B) / P(B), when P(B) > 0",
                  "P(~A) = 1 - P(A)"]
    if "c" in event_names:
        identities.extend(("P(A+B+C) = P(A) + P(B) + P(C) - P(A*B) - P(A*C) - P(B*C) + P(A*B*C)",
                           "P(A|B*C) = P(A*B*C) / P(B*C), when P(B*C) > 0"))
    return {"results": results, "related": related, "events": list(event_names),
            "known": [{"expression": event.name, "value": value} for event, value in known],
            "joint_vertices": vertices, "identities": identities}


def format_probability_result(result):
    """A shared, copyable answer for notebook and website result tables."""
    if result["status"] == "undefined":
        return "Undefined: the conditioning event has probability zero."
    if result["value"] is not None:
        value = result["value"]
        text = f"{value} = {float(value):.10g}" if value.q != 1 else str(value)
    else:
        left = "[" if result["lower_inclusive"] else "("
        right = "]" if result["upper_inclusive"] else ")"
        text = f"Not uniquely determined; range {left}{result['lower']}, {result['upper']}{right}"
    if result["undefined_possible"]:
        text += " (when defined; a zero-probability conditioning event is also possible)"
    return text


def sampling_without_replacement(total, successes, draws, comparison=">=", boundary=1, upper=None):
    """Exact hypergeometric counts for a uniformly selected sample from a finite bag."""
    for name, value in (("Total items", total), ("Marked items", successes), ("Draws", draws), ("Boundary", boundary)):
        if not np.isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    total, successes, draws, boundary = map(int, (total, successes, draws, boundary))
    if total < 1 or successes > total or draws > total:
        raise ValueError("Use a positive total, with marked items and draws no larger than the total.")
    if total > 10000 or draws > 1000:
        raise ValueError("Use at most 10000 items and 1000 draws.")
    low, high = max(0, draws-(total-successes)), min(draws, successes)
    outcomes = np.arange(low, high+1)
    denominator = comb(total, draws)
    counts = [comb(successes, int(k))*comb(total-successes, draws-int(k)) for k in outcomes]
    probabilities = [count/denominator for count in counts]
    details = discrete_distribution(outcomes, probabilities, comparison, boundary, upper)
    if comparison == "between":
        if upper is None or int(upper) != upper:
            raise ValueError("The upper count must be an integer.")
        mask = (outcomes >= boundary) & (outcomes <= upper)
    else:
        mask = {"<": outcomes < boundary, "<=": outcomes <= boundary, ">": outcomes > boundary, ">=": outcomes >= boundary, "=": outcomes == boundary}[comparison]
    exact = sp.Rational(sum(count for count, included in zip(counts, mask) if included), denominator)
    details.update(exact_probability=exact, event_probability=float(exact))
    return details


def compare_grouped_rates(table):
    """Each row contains A successes/total and B successes/total in one stratum."""
    values = ensure_2d(table)
    if values.shape[1] != 4 or np.any(values != np.floor(values)) or np.any(values < 0):
        raise ValueError("Use four nonnegative integer counts per row: A successes, A total, B successes, B total.")
    totals = values[:, [1, 3]]
    counts = values[:, [0, 2]]
    if np.any(totals == 0) or np.any(counts > totals):
        raise ValueError("Each total must be positive and at least as large as its success count.")
    rates = counts/totals
    pooled = np.sum(counts, axis=0)/np.sum(totals, axis=0)
    signs = np.sign(rates[:, 0]-rates[:, 1])
    pooled_sign = np.sign(pooled[0]-pooled[1])
    reversal = bool((np.all(signs > 0) and pooled_sign < 0) or (np.all(signs < 0) and pooled_sign > 0))
    return {"counts": counts, "totals": totals, "rates": rates, "pooled": pooled,
            "weights": totals/np.sum(totals, axis=0), "reversal": reversal}


def probability_vector(values, allow_unknown=False):
    probabilities = ensure_2d(values, allow_missing=allow_unknown).reshape(-1).copy()
    unknown = np.isnan(probabilities)
    if np.sum(unknown) > 1:
        raise ValueError("At most one probability may be unknown.")
    if np.any(probabilities[~unknown] < 0) or np.any(probabilities[~unknown] > 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if np.any(unknown):
        remaining = 1 - np.sum(probabilities[~unknown])
        if remaining < -1e-12:
            raise ValueError("Known probabilities already sum to more than 1.")
        probabilities[unknown] = np.clip(remaining, 0, 1)
    if not np.isclose(np.sum(probabilities), 1, rtol=0, atol=1e-10):
        raise ValueError("Probabilities must sum to 1.")
    return probabilities, unknown


def discrete_distribution(outcomes, probabilities, comparison=">=", boundary=0, upper=None):
    outcomes = ensure_2d(outcomes).reshape(-1)
    probabilities, unknown = probability_vector(probabilities, allow_unknown=True)
    if outcomes.size != probabilities.size:
        raise ValueError("Enter one probability for each outcome.")
    if not np.isfinite(boundary):
        raise ValueError("The query boundary must be finite.")
    if comparison == "between":
        if upper is None or not np.isfinite(upper) or upper < boundary:
            raise ValueError("An inclusive interval needs finite, ordered endpoints.")
        mask = (outcomes >= boundary) & (outcomes <= upper)
    elif comparison in {"<", "<=", ">", ">=", "="}:
        functions = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "=": np.equal}
        mask = functions[comparison](outcomes, boundary)
    else:
        raise ValueError("Unknown probability comparison.")
    mean = float(outcomes @ probabilities)
    variance = float(np.sum(probabilities * np.square(outcomes - mean)))
    unique_outcomes = np.unique(outcomes)
    masses = np.array([np.sum(probabilities[outcomes == value]) for value in unique_outcomes])
    return {"outcomes": outcomes, "probabilities": probabilities, "unknown": unknown,
            "event_probability": float(np.sum(probabilities[mask])), "mean": mean,
            "variance": variance, "std": np.sqrt(variance),
            "sorted_outcomes": unique_outcomes, "cdf": np.cumsum(masses)}


def bayes_update(priors, likelihoods, event_occurred=True):
    priors, unknown = probability_vector(priors, allow_unknown=True)
    likelihoods = ensure_2d(likelihoods).reshape(-1)
    if likelihoods.size != priors.size:
        raise ValueError("Enter one likelihood per prior group.")
    if np.any(likelihoods < 0) or np.any(likelihoods > 1):
        raise ValueError("Likelihoods must be between 0 and 1.")
    used_likelihoods = likelihoods if event_occurred else 1 - likelihoods
    joint = priors * used_likelihoods
    evidence = float(np.sum(joint))
    if evidence == 0:
        raise ValueError("The observed event has probability zero; its posterior is undefined.")
    return {"priors": priors, "likelihoods": used_likelihoods, "joint": joint,
            "evidence": evidence, "posterior": joint / evidence, "unknown": unknown}


def event_probabilities(p_a, p_b, known, known_kind="intersection"):
    values = np.array([p_a, p_b, known], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Event probabilities must be finite numbers between 0 and 1.")
    if known_kind not in {"intersection", "union"}:
        raise ValueError("Supply an intersection or union probability.")
    joint = known if known_kind == "intersection" else p_a + p_b - known
    if joint < max(0, p_a + p_b - 1) - 1e-12 or joint > min(p_a, p_b) + 1e-12:
        raise ValueError("These probabilities are inconsistent with any joint distribution.")
    joint = float(np.clip(joint, max(0, p_a + p_b - 1), min(p_a, p_b)))
    table = np.array([[joint, p_a - joint], [p_b - joint, 1 - p_a - p_b + joint]])
    return {"intersection": joint, "union": p_a + p_b - joint,
            "p_a_given_b": joint / p_b if p_b > 0 else None,
            "p_b_given_a": joint / p_a if p_a > 0 else None,
            "independent": bool(np.isclose(joint, p_a * p_b, atol=1e-12, rtol=1e-10)),
            "mutually_exclusive": joint == 0, "joint_table": table}


def density_probability(density, support_low, support_high, query_low, query_high):
    bounds = np.array([support_low, support_high, query_low, query_high], dtype=float)
    if not np.all(np.isfinite(bounds)) or support_low >= support_high or query_low > query_high:
        raise ValueError("Use finite support and query endpoints in increasing order.")
    x = sp.Symbol("x", real=True)
    expression = sp.sympify(density.replace("^", "**"), locals={"x": x})
    if not isinstance(expression, sp.Expr) or expression.free_symbols - {x}:
        raise ValueError("The density must be a scalar expression using x only.")
    support = sp.Interval(support_low, support_high)
    try:
        if not support.is_subset(continuous_domain(expression, x, sp.S.Reals)):
            raise ValueError("The density must be continuous on its support.")
        lower_value = minimum(expression, x, support)
        if lower_value.is_nonnegative is not True:
            raise ValueError("The density must be nonnegative throughout its support.")
        total_expression = sp.integrate(expression, (x, support_low, support_high))
        total = float(total_expression)
    except (NotImplementedError, TypeError):
        raise ValueError("This density could not be verified symbolically. Use a continuous finite-support expression.") from None
    if not np.isfinite(total) or not np.isclose(total, 1, atol=1e-10, rtol=1e-10):
        raise ValueError(f"The density must integrate to 1 on its support; its area is {total:g}.")
    low, high = max(query_low, support_low), min(query_high, support_high)
    probability = float(sp.integrate(expression, (x, low, high))) if low < high else 0.0
    return {"expression": expression, "total_area": total, "probability": probability}
