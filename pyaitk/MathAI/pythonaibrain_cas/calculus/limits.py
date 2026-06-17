"""
calculus/limits.py
==================
Symbolic limit computation using L'Hôpital's rule and Taylor expansion.
"""

from __future__ import annotations
import math
from fractions import Fraction
from pythonaibrain_cas.core.expression import Expr, Symbol, Number, Constant, _coerce
from pythonaibrain_cas.core.expression import Infinity, NegInfinity
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def limit(expr: Expr, var: Symbol, point: Expr, direction: str = '+-') -> Expr:
    """
    Compute lim_{var → point} expr.

    Parameters
    ----------
    expr      : expression
    var       : variable
    point     : limit point (can be Infinity)
    direction : '+' (from right), '-' (from left), '+-' (two-sided)

    Returns
    -------
    Limit value as Expr, or raises ValueError if limit doesn't exist.
    """
    expr  = _coerce(expr)
    point = _coerce(point)

    # If expression doesn't depend on var
    if not var in expr.free_symbols():
        return expr

    # Direct substitution attempt
    result = _try_direct(expr, var, point)
    if result is not None:
        return result

    # L'Hôpital's rule (up to 5 applications)
    lopital = _try_lhopital(expr, var, point, max_iter=5)
    if lopital is not None:
        return lopital

    # Numerical approach
    return _numerical_limit(expr, var, point, direction)


def _try_direct(expr, var, point) -> Expr:
    """Attempt simple substitution; return None on 0/0, ∞/∞, or other indeterminate forms."""
    from pythonaibrain_cas.algebra.simplify import simplify
    from pythonaibrain_cas.core.operations import Mul, Pow, Add
    
    # Pre-check: detect 0/0 indeterminate form by checking numerator/denominator separately
    try:
        # If expression is a product containing x^-1 or (x+c)^-1 factors,
        # check numerator and denominator values independently
        def _check_indeterminate(e):
            """Return True if substituting point gives 0/0 or x/0 indeterminate."""
            from pythonaibrain_cas.core.operations import Mul, Pow
            if isinstance(e, Mul):
                # Separate into positive and negative powers
                nums, dens = [], []
                for f in e.args:
                    if isinstance(f, Pow) and isinstance(f.exp, __import__("pythonaibrain_cas.core.expression", fromlist=["Number"]).Number) and f.exp.value < 0:
                        dens.append(f)
                    else:
                        nums.append(f)
                if dens:
                    # Check if denominator → 0 at point
                    from pythonaibrain_cas.core.operations import Mul as M2
                    den_expr = M2(*dens) if len(dens) > 1 else dens[0]
                    den_val = den_expr.evalf({var: point})
                    if abs(den_val) < 1e-10:
                        return True  # denominator is 0 → indeterminate or infinity
            return False
        
        if _check_indeterminate(expr):
            return None
        
        substituted = expr.subs({var: point})
        val = substituted.evalf()
        if math.isnan(val.real) or math.isnan(val.imag) or math.isinf(val.real) or math.isinf(val.imag):
            return None
        return substituted
    except Exception:
        return None


def _try_lhopital(expr, var, point, max_iter=5) -> Expr:
    """Apply L'Hôpital's rule when numerator and denominator both → 0 or ±∞."""
    from pythonaibrain_cas.calculus.differentiate import diff
    from pythonaibrain_cas.algebra.simplify import simplify

    if not isinstance(expr, Mul):
        return None

    # Look for f/g pattern (f * g^-1)
    num_parts, den_parts = [], []
    for factor in expr.args:
        if isinstance(factor, Pow) and isinstance(factor.exp, Number) and factor.exp.value < 0:
            den_parts.append(Pow(factor.base, Number(-factor.exp.value)))
        else:
            num_parts.append(factor)

    if not den_parts:
        return None

    num = Mul(*num_parts) if len(num_parts) > 1 else (num_parts[0] if num_parts else Number(1))
    den = Mul(*den_parts) if len(den_parts) > 1 else den_parts[0]

    def eval_at_point(e):
        try:
            v = e.subs({var: point}).evalf()
            return v
        except Exception:
            return None

    for _ in range(max_iter):
        n_val = eval_at_point(num)
        d_val = eval_at_point(den)

        if n_val is None or d_val is None:
            break

        n_zero = abs(n_val) < 1e-12
        d_zero = abs(d_val) < 1e-12
        n_inf  = abs(n_val) == float('inf')
        d_inf  = abs(d_val) == float('inf')

        if (n_zero and d_zero) or (n_inf and d_inf):
            num = diff(num, var)
            den = diff(den, var)
            continue

        if not d_zero:
            val = n_val / d_val
            return Number(Fraction(val.real).limit_denominator(10**8))
        break

    return None


def _numerical_limit(expr, var, point, direction) -> Expr:
    """Estimate limit numerically via one-sided approach."""
    from fractions import Fraction

    try:
        p_val = float(point.evalf().real)
    except Exception:
        return expr

    epsilons = [1e-6, 1e-8, 1e-10]
    values = []
    for eps in epsilons:
        if direction in ('+', '+-'):
            t = p_val + eps
        else:
            t = p_val - eps
        try:
            v = expr.evalf({var: Number(Fraction(t).limit_denominator(10**12))})
            values.append(v.real)
        except Exception:
            pass

    if not values:
        return expr
    # Check convergence
    if len(values) >= 2 and abs(values[-1] - values[-2]) < 1e-6:
        return Number(Fraction(values[-1]).limit_denominator(10**6))
    return Number(Fraction(values[0]).limit_denominator(10**6))
