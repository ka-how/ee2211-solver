"""Differentiate real matrix expressions using a bounded MATLAB-style grammar.

This parser evaluates only arithmetic and the documented matrix functions. It
does not execute MATLAB/Python code. Jacobians use column-major vectorization.
"""

import re

import numpy as np
import sympy as sp


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\.\*|\./|\.\^|\.'|'(?:fro|all)'|'|[+*/\\^(),;\[\]-]))")


def _matrix(value):
    return isinstance(value, sp.MatrixBase)


def _scalar(value):
    return isinstance(value, sp.Expr)


def _flatten(value):
    return [value[row, col] for col in range(value.cols) for row in range(value.rows)] if _matrix(value) else [value]


def _binary(left, op, right):
    lm, rm = _matrix(left), _matrix(right)
    if op in {"+", "-"}:
        if lm and not rm:
            return left.applyfunc(lambda item: item + right if op == "+" else item - right)
        if rm and not lm:
            return right.applyfunc(lambda item: left + item if op == "+" else left - item)
        return left + right if op == "+" else left - right
    if op == "*":
        return left * right
    if op in {".*", "./", ".^"}:
        if lm and rm:
            if left.shape != right.shape:
                raise ValueError("Elementwise operations require matching matrix shapes.")
            action = {".*": lambda a, b: a*b, "./": lambda a, b: a/b, ".^": lambda a, b: a**b}[op]
            return sp.Matrix(left.rows, left.cols, lambda i, j: action(left[i, j], right[i, j]))
        if lm:
            return left.applyfunc(lambda item: _binary(item, op, right))
        if rm:
            return right.applyfunc(lambda item: _binary(left, op, item))
        return {".*": lambda: left*right, "./": lambda: left/right, ".^": lambda: left**right}[op]()
    if op == "/":
        if rm:
            if not lm or right.rows != right.cols:
                raise ValueError("Matrix / needs a matrix numerator and square invertible denominator.")
            return left * right.inv()
        return left / right
    if op == "\\":
        if lm:
            if left.rows != left.cols or not rm:
                raise ValueError("Matrix \\ supports square invertible left operands; use the Solve tab for rectangular systems.")
            return left.inv() * right
        return right / left
    if op == "^":
        if rm or (lm and (not right.is_Integer or abs(int(right)) > 12)):
            raise ValueError("Matrix powers require an integer between -12 and 12. Use .^ for elementwise powers.")
        return left ** right
    raise ValueError("Unsupported operator.")


class _Parser:
    def __init__(self, text, environment):
        self.environment = environment
        self.tokens = []
        offset = 0
        while offset < len(text):
            if not text[offset:].strip():
                break
            match = _TOKEN.match(text, offset)
            if not match:
                raise ValueError(f"Unsupported MATLAB syntax near {text[offset:offset+20]!r}.")
            group = next(i for i in range(1, 4) if match.group(i) is not None)
            self.tokens.append((match.group(group), match.start(group), match.end(group)))
            offset = match.end()
        self.index = 0

    def peek(self):
        return self.tokens[self.index][0] if self.index < len(self.tokens) else None

    def pop(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"Expected {expected or 'an expression'}, got {token or 'end of input'}.")
        self.index += 1
        return token

    def expression(self, minimum=0, matrix_entry=False):
        token = self.pop()
        if token in {"+", "-"}:
            result = self.expression(25, matrix_entry)
            result = result if token == "+" else -result
        elif token == "(":
            result = self.expression()
            self.pop(")")
        elif token == "[":
            rows, row = [], []
            while True:
                if self.peek() in {None, "]", ";"}:
                    raise ValueError("Matrix literals cannot contain empty rows.")
                item = self.expression(matrix_entry=True)
                if not _scalar(item):
                    raise ValueError("Matrix literal entries must be scalars; concatenation is not supported.")
                row.append(item)
                if self.peek() == ",":
                    self.pop()
                elif self.peek() in {";", "]"}:
                    end = self.pop()
                    rows.append(row)
                    row = []
                    if end == "]":
                        break
                elif self.peek() is None:
                    raise ValueError("Close the matrix literal with ].")
                elif self.tokens[self.index][1] == self.tokens[self.index-1][2]:
                    raise ValueError("Separate matrix entries with spaces or commas.")
            if len({len(row) for row in rows}) != 1:
                raise ValueError("All matrix rows must have the same width.")
            result = sp.Matrix(rows)
        elif token in {"'fro'", "'all'"}:
            result = token[1:-1]
        elif re.fullmatch(r"[A-Za-z_]\w*", token):
            if self.peek() == "(":
                self.pop()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.expression())
                        if self.peek() != ",":
                            break
                        self.pop()
                self.pop(")")
                result = self.call(token, args)
            else:
                result = self.environment.get(token, {"pi": sp.pi}.get(token, sp.Symbol(token, real=True)))
        else:
            result = sp.Rational(token)
        # MATLAB puts powers and transposes at the same precedence level, with
        # equal-precedence operators evaluated left to right.
        priorities = {"+": 10, "-": 10, "*": 20, "/": 20, "\\": 20, ".*": 20, "./": 20, "^": 30, ".^": 30, "'": 30, ".'": 30}
        while self.peek() in priorities and priorities[self.peek()] >= minimum:
            op = self.peek()
            # MATLAB [1 -2] has two entries, whereas [1 - 2] has one.
            if matrix_entry and op in {"+", "-"} and self.index + 1 < len(self.tokens):
                if self.tokens[self.index][1] > self.tokens[self.index-1][2] and self.tokens[self.index+1][1] == self.tokens[self.index][2]:
                    break
            self.pop()
            if op in {"'", ".'"}:
                result = result.T if _matrix(result) else result
            else:
                right = self.expression(priorities[op] + 1, matrix_entry)
                result = _binary(result, op, right)
        return result

    def call(self, name, args):
        if name in self.environment and _matrix(self.environment[name]):
            array = self.environment[name]
            if len(args) not in {1, 2} or any(not _scalar(a) or not a.is_Integer or a < 1 for a in args):
                raise ValueError("Matrix indexing uses one or two positive, one-based integer indices.")
            indices = [int(a)-1 for a in args]
            if len(indices) == 1:
                if indices[0] >= array.rows*array.cols:
                    raise ValueError("Matrix index exceeds its dimensions.")
                return _flatten(array)[indices[0]]
            return array[indices[0], indices[1]]
        if name in {"eye", "zeros", "ones"}:
            if len(args) not in {1, 2} or any(not _scalar(a) or not a.is_Integer or a < 1 for a in args):
                raise ValueError("Matrix constructors need one or two positive integer sizes.")
            rows, cols = (int(args[0]), int(args[-1]))
            if rows*cols > 100:
                raise ValueError("Limit each matrix to 100 entries.")
            return {"eye": sp.eye, "zeros": sp.zeros, "ones": sp.ones}[name](rows, cols)
        if name == "norm" and len(args) in {1, 2}:
            a = args[0]
            if len(args) == 2 and args[1] not in ("fro", sp.Integer(2)):
                raise ValueError("Use norm(v), norm(v,2), or norm(A,'fro').")
            if _matrix(a) and min(a.shape) > 1 and (len(args) < 2 or args[1] != "fro"):
                raise ValueError("For a matrix use norm(A,'fro'); spectral matrix norms are not supported.")
            return sp.sqrt(sum(item**2 for item in _flatten(a)))
        if name == "sum" and len(args) in {1, 2}:
            a = args[0]
            if len(args) == 2 and args[1] != "all":
                raise ValueError("Use sum(A) or sum(A,'all').")
            if not _matrix(a):
                return a
            if len(args) == 2 or min(a.shape) == 1:
                return sum(_flatten(a))
            return sp.Matrix(1, a.cols, lambda i, j: sum(a[:, j]))
        if len(args) != 1:
            raise ValueError(f"Unsupported function or arguments: {name}.")
        a = args[0]
        if name in {"trace", "det", "inv", "transpose", "diag"}:
            if not _matrix(a):
                return a if name in {"trace", "det", "transpose", "diag"} else 1/a
            if name == "transpose":
                return a.T
            if name == "diag":
                return sp.diag(*_flatten(a)) if min(a.shape) == 1 else sp.Matrix(a.diagonal()).T
            if a.rows != a.cols:
                raise ValueError(f"{name} needs a square matrix.")
            return {"trace": a.trace, "det": a.det, "inv": a.inv}[name]()
        if name in {"sin", "cos", "tan", "exp", "log", "sqrt", "abs"}:
            function = {"abs": sp.Abs}.get(name, getattr(sp, name, None))
            return a.applyfunc(function) if _matrix(a) else function(a)
        raise ValueError(f"Unsupported function: {name}.")

    def parse(self):
        result = self.expression()
        if self.peek() is not None or not (_scalar(result) or _matrix(result)):
            raise ValueError("Enter one matrix expression using the documented MATLAB syntax.")
        return result


def _statements(text):
    text = re.sub(r"%[^\n]*", "", text).strip()
    if len(text) > 12000:
        raise ValueError("Limit input to 12000 characters.")
    depth, start, parts = 0, 0, []
    for i, char in enumerate(text):
        if char in "[(":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char in ";\n" and depth == 0:
            if text[start:i].strip():
                parts.append(text[start:i].strip())
            start = i+1
    if depth != 0:
        raise ValueError("Check the brackets and parentheses.")
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def parse_definitions(text, environment=None, declarations=True):
    environment = dict(environment or {})
    assigned = set()
    for statement in _statements(text):
        if declarations and statement.startswith("syms "):
            match = re.fullmatch(r"syms\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*[, ]\s*(\d+)\s*\]\s*(?:real)?", statement)
            if match:
                name, rows, cols = match.group(1), int(match.group(2)), int(match.group(3))
                if rows < 1 or cols < 1 or rows*cols > 100:
                    raise ValueError("Symbolic matrices need positive dimensions and at most 100 entries.")
                value = sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f"{name}{i+1}" if cols == 1 else f"{name}_{i+1}_{j+1}", real=True))
                names = [(name, value)]
            else:
                scalar_names = statement[5:].split()
                if scalar_names and scalar_names[-1] == "real":
                    scalar_names.pop()
                if not scalar_names or any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in scalar_names):
                    raise ValueError("Use syms x [2 1] real, or syms a b real.")
                names = [(name, sp.Symbol(name, real=True)) for name in scalar_names]
        else:
            match = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", statement, re.S)
            if not match:
                raise ValueError("Definitions use name = value or syms name [rows columns] real.")
            name = match.group(1)
            # Newlines inside matrix literals act as row separators.
            value = _Parser(match.group(2).replace("\n", ";"), environment).parse()
            names = [(name, value)]
        for name, value in names:
            if name in assigned:
                raise ValueError(f"Define {name} only once.")
            if _matrix(value) and value.rows*value.cols > 100:
                raise ValueError("Limit each matrix to 100 entries.")
            assigned.add(name)
            environment[name] = value
    return environment, assigned


def _numeric(value):
    entries = _flatten(value)
    if any(entry.free_symbols for entry in entries):
        return None
    try:
        array = np.asarray(sp.Matrix(value) if _matrix(value) else [[value]], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("The expression or derivatives are undefined or not real at the supplied values.") from None
    if not np.all(np.isfinite(array)):
        raise ValueError("The expression or derivatives are not finite at the supplied values.")
    return array


def exact_coordinate(text):
    """A numeric scalar coordinate, including exact pi, fractions, and sqrt."""
    value = _Parser(str(text).strip().replace("**", "^"), {}).parse()
    if not _scalar(value) or value.free_symbols or value.is_real is not True or value.is_finite is not True:
        raise ValueError("Coordinates must be finite real numbers, fractions, or numeric expressions such as pi/2.")
    return value


def differentiate_matrix(expression, definitions, variable, evaluation=""):
    """Return exact entry derivatives plus optional evaluation at supplied values."""
    environment, _ = parse_definitions(definitions)
    variable = variable.strip()
    target = environment.get(variable)
    if target is None:
        if not re.fullmatch(r"[A-Za-z_]\w*", variable):
            raise ValueError("Enter one differentiation variable name.")
        target = sp.Symbol(variable, real=True)
        environment[variable] = target
    symbols = _flatten(target)
    if not symbols or any(not isinstance(item, sp.Symbol) for item in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("The differentiation variable must contain distinct symbolic entries. Use syms, then supply its numeric point under Evaluate.")
    statements = _statements(expression)
    if len(statements) != 1:
        raise ValueError("Enter one expression; put assignments in Matrices and constants.")
    raw = re.sub(r"^[A-Za-z_]\w*\s*=\s*", "", statements[0])
    try:
        value = _Parser(raw.replace("\n", ";"), environment).parse()
        if len(_flatten(value))*len(symbols) > 5000:
            raise ValueError("Limit the Jacobian to 5000 entries.")
        jacobian = sp.Matrix(_flatten(value)).jacobian(symbols).applyfunc(sp.simplify)
        scalar_output = len(_flatten(value)) == 1
        gradient = (sp.Matrix(target.rows, target.cols, lambda i, j: jacobian[0, j*target.rows+i]) if _matrix(target) else jacobian) if scalar_output else None
        hessian = sp.hessian(_flatten(value)[0], symbols).applyfunc(sp.simplify) if scalar_output else None
    except (sp.ShapeError, sp.NonSquareMatrixError, IndexError, TypeError) as exc:
        raise ValueError(f"Check matrix dimensions and operators: {exc}") from None
    substitutions = {}
    if evaluation.strip():
        evaluated, assigned = parse_definitions(evaluation, environment, declarations=False)
        free_symbols = set().union(*(entry.free_symbols for entry in _flatten(value))) | set(symbols)
        for name in assigned:
            before = environment.get(name, sp.Symbol(name, real=True))
            after = evaluated[name]
            if _matrix(before) != _matrix(after) or (_matrix(before) and before.shape != after.shape):
                raise ValueError(f"Evaluation shape for {name} must match its definition.")
            if any(entry.free_symbols for entry in _flatten(after)):
                raise ValueError("Evaluation assignments must give numeric values.")
            for old, new in zip(_flatten(before), _flatten(after)):
                if not isinstance(old, sp.Symbol) or old not in free_symbols:
                    raise ValueError(f"{name} is fixed or unused; define symbolic constants with syms before evaluating them.")
                substitutions[old] = new
    return {"value": value, "gradient": gradient, "jacobian": jacobian, "hessian": hessian,
            "variables": [str(item) for item in symbols], "target_shape": target.shape if _matrix(target) else (1, 1),
            "value_at_point": _numeric(value.subs(substitutions)),
            "derivative_at_point": _numeric((gradient if scalar_output else jacobian).subs(substitutions)),
            "hessian_at_point": _numeric(hessian.subs(substitutions)) if hessian is not None else None}
