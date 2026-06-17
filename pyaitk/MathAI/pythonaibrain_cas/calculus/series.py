"""
calculus/series.py
==================
Taylor and Maclaurin series expansion.
"""

from __future__ import annotations
from fractions import Fraction
from typing import List, Optional
from pythonaibrain_cas.core.expression import Expr, Symbol, Number, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def series_expand(
    expr: Expr,
    var: Symbol,
    point: Expr = None,
    n: int = 6,
) -> Expr:
    """
    Compute the Taylor series of expr around var=point up to order n.

    Parameters
    ----------
    expr  : expression to expand
    var   : expansion variable
    point : expansion point (default 0 → Maclaurin)
    n     : number of terms

    Returns
    -------
    Polynomial approximation as Add of terms.

    Examples
    --------
    >>> series_expand(sin(x), x, n=5)
    # x - x**3/6 + x**5/120
    """
    from pythonaibrain_cas.calculus.differentiate import diff
    from pythonaibrain_cas.algebra.simplify import simplify

    expr  = _coerce(expr)
    point = _coerce(point) if point is not None else Number(0)

    # Compute successive derivatives at the point
    terms = []
    factorial = 1
    current = expr

    for k in range(n + 1):
        # Evaluate k-th derivative at point
        try:
            val = current.evalf({var: point})
            coeff = Fraction(val.real).limit_denominator(10**8)
        except Exception:
            coeff = Fraction(0)

        if coeff != 0:
            if k == 0:
                terms.append(Number(coeff))
            elif k == 1:
                base_term = Add(var, Mul(Number(-1), point)) if point != Number(0) else var
                terms.append(Mul(Number(Fraction(coeff, factorial)), base_term))
            else:
                base_term = Pow(
                    Add(var, Mul(Number(-1), point)) if point != Number(0) else var,
                    Number(k)
                )
                terms.append(Mul(Number(Fraction(coeff, factorial)), base_term))

        # Next derivative
        try:
            current = simplify(diff(current, var))
        except Exception:
            break
        factorial *= (k + 1)

    if not terms:
        return Number(0)
    return Add(*terms)
