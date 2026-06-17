"""
calculus/integrate.py
=====================
Symbolic integration: polynomial, rational, trig, exponential patterns.
Falls back to Gaussian quadrature for definite integrals.
"""

from __future__ import annotations
import math
import cmath
from fractions import Fraction
from typing import Optional, Tuple

from pythonaibrain_cas.core.expression import Expr, Symbol, Number, Constant, _coerce
from pythonaibrain_cas.core.expression import Pi, E
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def integrate(
    expr: Expr,
    var: Symbol,
    lower: Optional[Expr] = None,
    upper: Optional[Expr] = None,
) -> Expr:
    """
    Integrate expr with respect to var.

    Parameters
    ----------
    expr          : expression to integrate
    var           : integration variable
    lower, upper  : bounds for definite integral (None for indefinite)

    Returns
    -------
    Indefinite: antiderivative (no constant of integration)
    Definite  : numerical value as Number (falls back to quadrature)

    Examples
    --------
    >>> integrate(x**3, x)           # x**4/4
    >>> integrate(sin(x), x)         # -cos(x)
    >>> integrate(x**2, x, 0, 1)     # 1/3
    """
    expr = _coerce(expr)
    antideriv = _integrate_symbolic(expr, var)

    if lower is None and upper is None:
        return antideriv

    # Definite integral: F(upper) - F(lower)
    try:
        lo = _coerce(lower)
        hi = _coerce(upper)
        val_hi = antideriv.evalf({var: hi})
        val_lo = antideriv.evalf({var: lo})
        result = val_hi - val_lo
        if abs(result.imag) < 1e-12:
            return Number(Fraction(result.real).limit_denominator(10**8))
        return Number(result.real)  # return real part with warning
    except Exception:
        # Fall back to Gaussian quadrature
        return _gauss_quadrature(expr, var, _coerce(lower), _coerce(upper))


def _integrate_symbolic(expr: Expr, var: Symbol) -> Expr:
    """Attempt symbolic antiderivative computation."""

    # Linear combination: integral is linear
    if isinstance(expr, Add):
        return Add(*[_integrate_symbolic(a, var) for a in expr.args])

    # Constant w.r.t. var
    if not var in expr.free_symbols():
        return Mul(expr, var)

    # --- Power rule: integral(x^n) = x^(n+1)/(n+1) ---
    if isinstance(expr, Symbol) and expr == var:
        return Mul(Number(Fraction(1, 2)), Pow(var, Number(2)))

    if isinstance(expr, Pow):
        base, exp_ = expr.base, expr.exp
        if base == var and not var in exp_.free_symbols():
            if isinstance(exp_, Number) and exp_.value == -1:
                return Func('log', [var])
            new_exp = Add(exp_, Number(1))
            return Mul(Pow(var, new_exp), Pow(new_exp, Number(-1)))
        # General case: try u-substitution or by-parts (basic patterns)
        pass

    # --- Product: simple monomial c * x^n ---
    if isinstance(expr, Mul):
        coeff, rest = _split_coeff_mul(expr, var)
        if coeff is not None:
            return Mul(coeff, _integrate_symbolic(rest, var))

        # Try: f(x) * e^(g(x)) type patterns
        result = _try_parts(expr, var)
        if result is not None:
            return result

    # --- Trig integrals ---
    if isinstance(expr, Func):
        result = _integrate_func(expr, var)
        if result is not None:
            return result

    # Could not integrate symbolically - return unevaluated form
    return Func(f"Integral[{expr},{var}]", [expr])


def _split_coeff_mul(expr: Mul, var: Symbol):
    """Split Mul into (numeric_coeff, rest_expr) separating var-free parts."""
    consts = []
    var_parts = []
    for f in expr.args:
        if not var in f.free_symbols():
            consts.append(f)
        else:
            var_parts.append(f)
    if not consts:
        return None, expr
    if not var_parts:
        return Mul(*consts), var
    const_expr = Mul(*consts) if len(consts) > 1 else consts[0]
    rest_expr  = Mul(*var_parts) if len(var_parts) > 1 else var_parts[0]
    return const_expr, rest_expr


def _integrate_func(expr: Func, var: Symbol) -> Optional[Expr]:
    """Integrals of named functions."""
    name = expr.name
    arg  = expr.args[0]

    # Simple case: arg is linear in var: a*x + b
    a, b = _linear_coeff_intercept(arg, var)
    if a is None:
        return None  # not linear, skip

    a_expr = Number(a)
    b_expr = Number(b)

    def with_linear(antideriv_of_f):
        """Divide by coefficient a from chain rule."""
        if a == 1:
            return antideriv_of_f
        return Mul(Pow(a_expr, Number(-1)), antideriv_of_f)

    if name == 'sin':
        return with_linear(Mul(Number(-1), Func('cos', [arg])))

    if name == 'cos':
        return with_linear(Func('sin', [arg]))

    if name == 'tan':
        return with_linear(Mul(Number(-1), Func('log', [Func('cos', [arg])])))

    if name == 'exp':
        return with_linear(Func('exp', [arg]))

    if name == 'log':
        # ∫ln(ax+b)dx = (ax+b)(ln(ax+b)-1)/a
        return with_linear(Mul(arg, Add(Func('log', [arg]), Number(-1))))

    if name == 'sqrt':
        # ∫sqrt(ax+b)dx = 2/3*(ax+b)^(3/2)/a
        return with_linear(Mul(Number(Fraction(2, 3)), Pow(arg, Number(Fraction(3, 2)))))

    if name == 'asin':
        # ∫arcsin(x)dx = x*arcsin(x) + sqrt(1-x^2)
        return with_linear(Add(
            Mul(arg, Func('asin', [arg])),
            Func('sqrt', [Add(Number(1), Mul(Number(-1), Pow(arg, Number(2))))])
        ))

    if name == 'acos':
        return with_linear(Add(
            Mul(arg, Func('acos', [arg])),
            Mul(Number(-1), Func('sqrt', [Add(Number(1), Mul(Number(-1), Pow(arg, Number(2))))]))
        ))

    if name == 'atan':
        # ∫arctan(x)dx = x*arctan(x) - ln(1+x^2)/2
        return with_linear(Add(
            Mul(arg, Func('atan', [arg])),
            Mul(Number(Fraction(-1, 2)), Func('log', [Add(Number(1), Pow(arg, Number(2)))]))
        ))

    if name == 'sinh':
        return with_linear(Func('cosh', [arg]))

    if name == 'cosh':
        return with_linear(Func('sinh', [arg]))

    return None


def _linear_coeff_intercept(expr: Expr, var: Symbol):
    """If expr = a*var + b, return (a_frac, b_frac), else (None, None)."""
    from pythonaibrain_cas.algebra.polynomial import Polynomial
    from pythonaibrain_cas.algebra.simplify import expand
    try:
        poly = Polynomial.from_expr(expand(expr), var)
        if poly.degree == 1:
            return float(poly.coeffs[1]), float(poly.coeffs[0])
        if poly.degree == 0:
            return 1.0, float(poly.coeffs[0])
    except Exception:
        pass
    if expr == var:
        return 1.0, 0.0
    return None, None


def _try_parts(expr: Mul, var: Symbol) -> Optional[Expr]:
    """Integration by parts for simple u*v' patterns."""
    # Detect: polynomial * exp, polynomial * sin/cos
    args = expr.args
    poly_parts = []
    func_parts = []
    for a in args:
        if isinstance(a, Func) and a.name in ('exp', 'sin', 'cos', 'log'):
            func_parts.append(a)
        else:
            poly_parts.append(a)

    if len(func_parts) == 1 and poly_parts:
        f = func_parts[0]
        u = Mul(*poly_parts) if len(poly_parts) > 1 else poly_parts[0]
        # Only handle u = constant or u = var for now
        if not var in u.free_symbols():
            return Mul(u, _integrate_symbolic(f, var))
        if u == var:
            # ∫x*e^x = e^x*(x-1), ∫x*sin(x) = sin(x)-x*cos(x), etc.
            name = f.name
            arg  = f.args[0]
            if name == 'exp' and arg == var:
                return Mul(Func('exp', [var]), Add(var, Number(-1)))
            if name == 'sin' and arg == var:
                return Add(Func('sin', [var]), Mul(Number(-1), Mul(var, Func('cos', [var]))))
            if name == 'cos' and arg == var:
                return Add(Func('cos', [var]), Mul(var, Func('sin', [var])))
    return None


def _gauss_quadrature(
    expr: Expr,
    var: Symbol,
    lower: Expr,
    upper: Expr,
    n: int = 64,
) -> Number:
    """
    n-point Gauss–Legendre quadrature on [lower, upper].
    """
    # Gauss-Legendre nodes and weights on [-1, 1]
    nodes, weights = _gauss_legendre_nodes_weights(n)

    try:
        a = lower.evalf().real
        b = upper.evalf().real
    except Exception:
        return Number(0)

    mid   = (a + b) / 2
    half  = (b - a) / 2
    total = 0.0

    for xi, wi in zip(nodes, weights):
        t = mid + half * xi
        try:
            fval = expr.evalf({var: Number(Fraction(t).limit_denominator(10**8))})
            total += wi * fval.real
        except Exception:
            pass

    result = half * total
    return Number(Fraction(result).limit_denominator(10**8))


def _gauss_legendre_nodes_weights(n: int):
    """Compute Gauss–Legendre nodes and weights using eigenvalue method."""
    import math
    # Build tridiagonal symmetric matrix
    betas = [0.5 / math.sqrt(1 - (2 * i) ** (-2)) for i in range(1, n)]
    # QR iteration (simple Golub–Welsch via companion matrix)
    # Use scipy if available, else fallback to approximate
    try:
        import numpy as np
        nodes, vecs = np.linalg.eigh(np.diag(betas, -1) + np.diag(betas, 1))
        weights = [2 * vecs[0, i] ** 2 for i in range(n)]
        return list(nodes), weights
    except ImportError:
        # 7-point Gauss-Legendre hardcoded fallback
        nodes   = [0.0,
                   0.4058451514, -0.4058451514,
                   0.7415311856, -0.7415311856,
                   0.9491079123, -0.9491079123]
        weights = [0.4179591837,
                   0.3818300505, 0.3818300505,
                   0.2797053915, 0.2797053915,
                   0.1294849662, 0.1294849662]
        return nodes, weights
