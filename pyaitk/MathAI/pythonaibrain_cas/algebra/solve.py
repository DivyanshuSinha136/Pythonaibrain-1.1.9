"""
algebra/solve.py
================
Symbolic and numerical equation solver.
"""

from __future__ import annotations
import cmath
import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from pythonaibrain_cas.core.expression import Expr, Symbol, Number, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def solve(
    expr: Expr,
    var: Optional[Symbol] = None,
    domain: str = 'complex',
) -> List[Expr]:
    """
    Solve expr = 0 for var.

    Parameters
    ----------
    expr   : symbolic expression equal to zero
    var    : variable to solve for (auto-detected if None)
    domain : 'real' or 'complex'

    Returns
    -------
    List of solutions as Expr objects.
    """
    from pythonaibrain_cas.algebra.simplify import expand
    from pythonaibrain_cas.algebra.polynomial import Polynomial

    expr = _coerce(expr)
    if var is None:
        syms = list(expr.free_symbols())
        if not syms:
            # No variable - check if true/false
            return [Number(0)] if expr == Number(0) else []
        var = syms[0]

    # Try polynomial solve
    try:
        expanded = expand(expr)
        poly = Polynomial.from_expr(expanded, var)
        roots = _solve_polynomial(poly, domain)
        return roots
    except (ValueError, NotImplementedError):
        pass

    # Fallback: numerical solve (Newton's method)
    try:
        return _numerical_solve(expr, var, domain)
    except Exception:
        return []


def _solve_polynomial(poly, domain: str) -> List[Expr]:
    """Solve polynomial = 0 exactly for low degrees, numerically otherwise."""
    from pythonaibrain_cas.algebra.polynomial import Polynomial

    d = poly.degree
    if d < 0:
        raise ValueError("Zero polynomial has infinite solutions")
    if d == 0:
        return []

    coeffs = [float(c) for c in poly.coeffs]

    if d == 1:
        # ax + b = 0 -> x = -b/a
        b, a = poly.coeffs[0], poly.coeffs[1]
        val = Fraction(-b, a) if a != 0 else None
        return [Number(val)] if val is not None else []

    if d == 2:
        c, b, a = [Fraction(poly.coeffs[i]) for i in range(3)]
        disc = b * b - 4 * a * c
        if disc >= 0 or domain == 'complex':
            if disc >= 0:
                sq = Fraction(math.isqrt(disc.numerator * disc.denominator))
                if sq * sq == disc.numerator * disc.denominator:
                    # Perfect square
                    sq_frac = Fraction(math.isqrt(disc.numerator), math.isqrt(disc.denominator))
                    r1 = Fraction(-b + sq_frac, 2 * a)
                    r2 = Fraction(-b - sq_frac, 2 * a)
                    return [Number(r1), Number(r2)]
            # Return numerical approximations
            roots = poly.roots_numerical()
            result = []
            for r in roots:
                if domain == 'real' and abs(r.imag) > 1e-10:
                    continue
                result.append(_complex_to_expr(r))
            return result
        return []

    # Higher degree: numerical
    roots_num = poly.roots_numerical()
    result = []
    for r in roots_num:
        if domain == 'real' and abs(r.imag) > 1e-10:
            continue
        result.append(_complex_to_expr(r))
    return result


def _complex_to_expr(z: complex) -> Expr:
    """Convert a complex number to a CAS expression."""
    from pythonaibrain_cas.core.expression import ImaginaryUnit
    tol = 1e-10
    if abs(z.imag) < tol:
        return Number(Fraction(z.real).limit_denominator(10**6))
    if abs(z.real) < tol:
        return Mul(Number(Fraction(z.imag).limit_denominator(10**6)), ImaginaryUnit)
    return Add(
        Number(Fraction(z.real).limit_denominator(10**6)),
        Mul(Number(Fraction(z.imag).limit_denominator(10**6)), ImaginaryUnit)
    )


def _numerical_solve(
    expr: Expr,
    var: Symbol,
    domain: str,
    x0: complex = complex(1.0),
    tol: float = 1e-12,
    max_iter: int = 100,
) -> List[Expr]:
    """Newton–Raphson root finding."""
    from pythonaibrain_cas.calculus.differentiate import diff

    df = diff(expr, var)

    def f(val):
        return expr.evalf({var: Number(Fraction(val.real).limit_denominator(10**8))})

    def df_val(val):
        return df.evalf({var: Number(Fraction(val.real).limit_denominator(10**8))})

    solutions = []
    for start in [complex(0.5), complex(-0.5), complex(0, 1), complex(1)]:
        x = start
        for _ in range(max_iter):
            fx  = f(x)
            dfx = df_val(x)
            if abs(dfx) < 1e-30:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        if abs(f(x)) < 1e-8:
            sol = _complex_to_expr(x)
            # Deduplicate
            if not any(abs((sol.evalf() if not isinstance(sol, Number) else complex(float(sol.value))) -
                           (s.evalf() if not isinstance(s, Number) else complex(float(s.value)))) < tol
                       for s in solutions):
                solutions.append(sol)

    return solutions


def solve_system(
    equations: Sequence[Expr],
    variables: Sequence[Symbol],
) -> List[Dict[Symbol, Expr]]:
    """
    Solve a system of equations using Gaussian elimination.

    Parameters
    ----------
    equations : list of expressions, each set equal to zero
    variables : list of variables to solve for

    Returns
    -------
    List of solution dicts {var: value, …}
    """
    n = len(variables)
    m = len(equations)

    # Build augmented matrix [A|b] from linear equations
    try:
        matrix_rows = []
        rhs = []
        for eq in equations:
            row = []
            for v in variables:
                coeff = _linear_coeff(eq, v, variables)
                row.append(float(coeff))
            const = _constant_term(eq, variables)
            rhs.append(-float(const))
            matrix_rows.append(row)

        # Gaussian elimination
        aug = [matrix_rows[i] + [rhs[i]] for i in range(m)]
        solutions = _gaussian_elimination(aug, n)
        if solutions is None:
            return []
        return [{variables[i]: Number(Fraction(solutions[i]).limit_denominator(10**8))
                 for i in range(n)}]
    except (ValueError, NotImplementedError):
        # Non-linear system: try substitution / numerical
        return _numerical_system_solve(equations, variables)


def _linear_coeff(expr: Expr, var: Symbol, all_vars) -> Fraction:
    """Extract the coefficient of var in a linear expression."""
    from pythonaibrain_cas.algebra.simplify import expand
    from pythonaibrain_cas.algebra.polynomial import Polynomial

    expanded = expand(expr)
    try:
        poly = Polynomial.from_expr(expanded, var)
        if poly.degree == 1:
            return poly.coeffs[1]
        return Fraction(0)
    except Exception:
        return Fraction(0)


def _constant_term(expr: Expr, all_vars) -> Fraction:
    """Extract the constant term (no variables)."""
    subs = {v: Number(0) for v in all_vars}
    try:
        return Fraction(expr.subs(subs).evalf().real).limit_denominator(10**8)
    except Exception:
        return Fraction(0)


def _gaussian_elimination(aug: List[List[float]], n: int):
    """Solve Ax=b via Gaussian elimination with partial pivoting."""
    m = len(aug)
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, m):
            if abs(aug[row][col]) > 1e-12:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for row in range(m):
            if row != col and abs(aug[row][col]) > 1e-12:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(n + 1)]

    if m < n:
        return None
    return [aug[i][n] for i in range(n)]


def _numerical_system_solve(equations, variables):
    """Numerical Newton iteration for nonlinear systems."""
    import random
    n = len(variables)
    x = [complex(random.uniform(-2, 2)) for _ in range(n)]

    def eval_all(vals):
        subs = {variables[i]: Number(Fraction(vals[i].real).limit_denominator(10**6))
                for i in range(n)}
        return [eq.evalf(subs) for eq in equations]

    for _ in range(200):
        fx = eval_all(x)
        if all(abs(v) < 1e-10 for v in fx):
            break
        # Finite-difference Jacobian
        J = []
        for eq in equations:
            row = []
            for i, var in enumerate(variables):
                h = 1e-7
                xp = x[:]
                xp[i] += h
                subs_plus  = {variables[j]: Number(Fraction(xp[j].real).limit_denominator(10**6)) for j in range(n)}
                subs_minus = {variables[j]: Number(Fraction(x[j].real).limit_denominator(10**6)) for j in range(n)}
                fp = eq.evalf(subs_plus)
                fm = eq.evalf(subs_minus)
                row.append((fp - fm) / h)
            J.append(row)
        # Solve J*dx = -fx using simple elimination
        try:
            aug = [J[i] + [-fx[i]] for i in range(n)]
            dx = _gaussian_elimination(aug, n)
            if dx is None:
                break
            x = [x[i] + complex(dx[i]) * 0.5 for i in range(n)]
        except Exception:
            break

    fx = eval_all(x)
    if all(abs(v) < 1e-6 for v in fx):
        return [{variables[i]: _complex_to_expr(x[i]) for i in range(n)}]
    return []
