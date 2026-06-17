# MathAI — Production-grade Mathematical Expression Solver

A robust symbolic mathematics solver built on SymPy, supporting simplification, equation solving, differentiation, integration, matrix analysis, and Taylor series expansion — with automatic operation detection, input validation, and structured result objects.

---

## What Is This?

`mathai.py` is a production-quality symbolic math engine that provides:

- **Auto-detection** — `process()` and `MathAI()` infer the operation type from the query automatically
- **Symbolic computation** — exact symbolic results powered by SymPy (no floating-point approximation unless requested)
- **Numeric evaluation** — automatically computes a decimal approximation when the result is a pure number
- **Equation solving** — single equations and systems of equations (via `x + y = 5 and x - y = 1` syntax)
- **Calculus** — differentiation (any order) and definite/indefinite integration
- **Matrix analysis** — determinant, trace, rank, inverse, and eigenvalues in one call
- **Taylor/Laurent series** — configurable expansion point and order
- **Input validation** — rejects empty input, oversized expressions, and dangerous code patterns
- **Structured results** — every operation returns a `MathResult` dataclass with `success`, `result`, `simplified`, `numeric`, and `metadata` fields
- **Structured logging** — all internal events go through Python's `logging` module
- **Convenience API** — single `MathAI(query, operation)` function for quick one-liner use

---

## Installation

```bash
pip install sympy
```

---

## How to Use

### 1. One-liner convenience function (simplest)

```python
from mathai import MathAI

print(MathAI("x^2 + 2*x + 1"))
print(MathAI("x^2 - 4 = 0", operation="solve"))
print(MathAI("sin(x)", operation="differentiate"))
```

### 2. Auto-detect operation

The `MathAI()` function (and `MathSolver.process()`) detect the operation from the query:

- Contains `=` → solve
- Starts with `Matrix` → matrix operations
- Starts with `diff` or `derivative(...)` → differentiate
- Starts with `int` or `integrate(...)` → integrate
- Anything else → simplify

```python
from mathai import MathAI

print(MathAI("x^2 - 9 = 0"))                         # → solve
print(MathAI("Matrix([[2, 1], [5, 3]])"))             # → matrix
print(MathAI("sin(x)^2 + cos(x)^2"))                 # → simplify
```

### 3. Using `MathSolver` directly

```python
from mathai import MathSolver

solver = MathSolver()

result = solver.simplify("sin(x)^2 + cos(x)^2")
print(result)
```

### 4. Simplification

```python
result = solver.simplify("x^3 + 3*x^2 + 3*x + 1")
print(result.simplified)   # (x + 1)**3
print(result.numeric)      # set if result is a pure number
```

### 5. Solving equations

```python
# Single equation
result = solver.solve_equation("x^2 - 4 = 0")
print(result.result)   # [{x: -2}, {x: 2}]

# System of equations (separate with "and")
result = solver.solve_equation("x + y = 10 and x - y = 2")
print(result.result)   # [{x: 6, y: 4}]
```

### 6. Differentiation

```python
# First derivative (default)
result = solver.differentiate("x^3 + sin(x)")
print(result.result)      # 3*x**2 + cos(x)
print(result.simplified)  # simplified form

# Higher-order derivative
result = solver.differentiate("x^5", var="x", order=3)
print(result.result)   # 60*x**2
```

### 7. Integration

```python
# Indefinite integral
result = solver.integrate("x^2 + 3*x")
print(result.result)   # x**3/3 + 3*x**2/2

# Definite integral
result = solver.integrate("x^2", var="x", limits=(0, 1))
print(result.result)   # 1/3
print(result.numeric)  # 0.333333333333333
```

### 8. Matrix operations

```python
result = solver.matrix_operations("Matrix([[1, 2], [3, 4]])")
print(result.metadata["determinant"])   # -2
print(result.metadata["inverse"])       # Matrix([[-2, 1], [3/2, -1/2]])
print(result.metadata["eigenvalues"])   # {-sqrt(33)/2 + 5/2: 1, sqrt(33)/2 + 5/2: 1}
print(result.metadata["rank"])          # 2
print(result.metadata["trace"])         # 5
```

### 9. Taylor series expansion

```python
# Default: expand around x=0, 6 terms
result = solver.series_expansion("sin(x)")
print(result.result)   # x - x**3/6 + x**5/120 + O(x**6)

# Custom point and order
result = solver.series_expansion("exp(x)", var="x", point=0, order=4)
print(result.result)   # 1 + x + x**2/2 + x**3/6 + O(x**4)
```

---

## API Reference

### `MathAI(query, operation)` → `str`

Top-level convenience function. Returns a formatted string of the result.

| Parameter   | Type  | Default  | Description                                                                 |
|-------------|-------|----------|-----------------------------------------------------------------------------|
| `query`     | `str` | `'1*x + 2*x - 3*x'` | Mathematical expression or equation                          |
| `operation` | `str` | `'auto'` | One of: `auto`, `simplify`, `solve`, `differentiate`, `integrate`, `matrix`, `series` |

---

### `MathSolver` methods

| Method                                              | Description                                          |
|-----------------------------------------------------|------------------------------------------------------|
| `process(query)`                                    | Auto-detect and dispatch to the right operation      |
| `simplify(expr)`                                    | Simplify a symbolic expression                       |
| `solve_equation(expr)`                              | Solve one or more equations                          |
| `differentiate(expr, var, order)`                   | Differentiate to any order                           |
| `integrate(expr, var, limits)`                      | Indefinite or definite integral                      |
| `matrix_operations(expr)`                           | Det, trace, rank, inverse, eigenvalues               |
| `series_expansion(expr, var, point, order)`         | Taylor/Laurent series around a point                 |

---

### `MathResult` fields

Every method returns a `MathResult` dataclass:

| Field        | Type            | Description                                              |
|--------------|-----------------|----------------------------------------------------------|
| `success`    | `bool`          | `True` if the operation completed without error          |
| `operation`  | `str`           | Name of the operation performed                          |
| `input_expr` | `str`           | The original input string                                |
| `result`     | `str` or `None` | Primary result of the operation                          |
| `simplified` | `str` or `None` | Simplified form (where applicable)                       |
| `numeric`    | `str` or `None` | Decimal evaluation (when result is a pure number)        |
| `error`      | `str` or `None` | Error message if `success` is `False`                    |
| `metadata`   | `dict` or `None`| Extra details (variable, limits, matrix properties, etc.)|

Calling `str(result)` produces a human-readable summary of all populated fields.

---

## Supported Symbols and Functions

The solver pre-loads a wide set of SymPy functions accessible directly in expressions:

| Category          | Available                                                      |
|-------------------|----------------------------------------------------------------|
| Trigonometric     | `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` |
| Exponential / Log | `exp`, `log`, `ln`                                             |
| Constants         | `e`, `pi`, `I` (imaginary unit), `oo` (infinity)              |
| Algebra           | `sqrt`, `abs`, `floor`, `ceil`, `factorial`, `binomial`        |
| Calculus          | `diff`, `integrate`, `limit`, `Sum`, `Product`                 |
| Linear Algebra    | `Matrix`, `det`, `transpose`                                   |
| Simplification    | `factor`, `expand`, `simplify`, `cancel`, `apart`, `together`  |
| Special Functions | `gamma`, `erf`                                                 |
| Variables         | `x y z a b c … theta phi alpha beta gamma delta epsilon`       |

---

## Input Validation

Before any operation, expressions are checked for:

- Empty or whitespace-only input
- Expressions exceeding 10,000 characters
- Forbidden patterns: `__`, `import`, `eval`, `exec`, `compile`, `open`, `file`

Rejected inputs return a `MathResult` with `success=False` and a descriptive `error` message — no exceptions are raised to the caller.

---

## Examples Summary

```python
from mathai import MathAI, MathSolver

# One-liner API
print(MathAI("x^2 + 2*x + 1"))
print(MathAI("x^2 - 4 = 0", operation="solve"))
print(MathAI("sin(x)", operation="differentiate"))
print(MathAI("x^2", operation="integrate"))
print(MathAI("Matrix([[1, 2], [3, 4]])", operation="matrix"))
print(MathAI("cos(x)", operation="series"))

# Using MathSolver directly
solver = MathSolver()
r = solver.solve_equation("x + y = 10 and x - y = 2")
print(r.result)

r = solver.differentiate("x^5", order=3)
print(r.simplified)

r = solver.integrate("x^2", limits=(0, 1))
print(r.numeric)

r = solver.matrix_operations("Matrix([[1, 2], [3, 4]])")
print(r.metadata)
```
