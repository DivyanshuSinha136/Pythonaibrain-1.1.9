"""
calculus/differentiate.py
=========================
Full symbolic differentiation with chain rule, product rule, quotient rule,
and all standard function derivatives.
"""

from __future__ import annotations
from fractions import Fraction
from pythonaibrain_cas.core.expression import Expr, Symbol, Number, Constant, _coerce
from pythonaibrain_cas.core.expression import Pi, E, ImaginaryUnit, Infinity
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def diff(expr: Expr, var: Symbol, n: int = 1) -> Expr:
    """
    Compute the n-th order derivative of expr with respect to var.

    Parameters
    ----------
    expr : symbolic expression
    var  : variable to differentiate with respect to
    n    : order of derivative (default 1)

    Returns
    -------
    Symbolic derivative as Expr.

    Examples
    --------
    >>> x = Symbol('x')
    >>> diff(x**3, x)         # 3*x**2
    >>> diff(sin(x)*cos(x), x) # cos(x)**2 - sin(x)**2
    >>> diff(x**5, x, 3)       # 60*x**2  (3rd derivative)
    """
    expr = _coerce(expr)
    for _ in range(n):
        expr = _diff_once(expr, var)
    from pythonaibrain_cas.algebra.simplify import simplify
    return simplify(expr)


def _diff_once(expr: Expr, var: Symbol) -> Expr:
    # Constants and unrelated symbols
    if isinstance(expr, Number) or isinstance(expr, Constant):
        return Number(0)
    if isinstance(expr, Symbol):
        return Number(1) if expr == var else Number(0)

    # --- Add: linearity d/dx(f+g) = f' + g' ---
    if isinstance(expr, Add):
        return Add(*[_diff_once(a, var) for a in expr.args])

    # --- Mul: product rule d/dx(f*g) = f'*g + f*g' ---
    if isinstance(expr, Mul):
        args = expr.args
        terms = []
        for i, factor in enumerate(args):
            d_factor = _diff_once(factor, var)
            if d_factor == Number(0):
                continue
            others = [args[j] for j in range(len(args)) if j != i]
            if others:
                term = Mul(d_factor, *others)
            else:
                term = d_factor
            terms.append(term)
        if not terms:
            return Number(0)
        return Add(*terms)

    # --- Pow: generalised power rule ---
    if isinstance(expr, Pow):
        base, exp_ = expr.base, expr.exp
        base_contains = var in base.free_symbols()
        exp_contains  = var in exp_.free_symbols()

        if not base_contains and not exp_contains:
            return Number(0)

        if not exp_contains:
            # d/dx(f(x)^n) = n * f(x)^(n-1) * f'(x)
            return Mul(
                exp_,
                Pow(base, Add(exp_, Number(-1))),
                _diff_once(base, var)
            )

        if not base_contains:
            # d/dx(a^g(x)) = a^g(x) * ln(a) * g'(x)
            from pythonaibrain_cas.core.operations import log as log_func
            return Mul(expr, log_func(base), _diff_once(exp_, var))

        # General: d/dx(f^g) = f^g * (g'*ln(f) + g*f'/f)
        from pythonaibrain_cas.core.operations import log as log_func
        return Mul(
            expr,
            Add(
                Mul(_diff_once(exp_, var), log_func(base)),
                Mul(exp_, _diff_once(base, var), Pow(base, Number(-1)))
            )
        )

    # --- Func: chain rule d/dx(f(g(x))) = f'(g(x)) * g'(x) ---
    if isinstance(expr, Func):
        return _diff_func(expr, var)

    return Number(0)


def _diff_func(expr: Func, var: Symbol) -> Expr:
    """Differentiate a named function application."""
    name = expr.name
    args = expr.args
    arg  = args[0] if args else None

    # Chain rule factor
    def chain(outer_deriv: Expr) -> Expr:
        da = _diff_once(arg, var)
        if da == Number(0):
            return Number(0)
        if da == Number(1):
            return outer_deriv
        return Mul(outer_deriv, da)

    # Derivatives of elementary functions
    if name == 'sin':
        return chain(Func('cos', [arg]))

    if name == 'cos':
        return chain(Mul(Number(-1), Func('sin', [arg])))

    if name == 'tan':
        # d/dx tan(x) = sec^2(x) = 1 + tan^2(x)
        return chain(Add(Number(1), Pow(Func('tan', [arg]), Number(2))))

    if name == 'asin':
        # 1/sqrt(1-x^2)
        return chain(Pow(Add(Number(1), Mul(Number(-1), Pow(arg, Number(2)))), Number(Fraction(-1, 2))))

    if name == 'acos':
        return chain(Mul(Number(-1), Pow(Add(Number(1), Mul(Number(-1), Pow(arg, Number(2)))), Number(Fraction(-1, 2)))))

    if name == 'atan':
        return chain(Pow(Add(Number(1), Pow(arg, Number(2))), Number(-1)))

    if name == 'sinh':
        return chain(Func('cosh', [arg]))

    if name == 'cosh':
        return chain(Func('sinh', [arg]))

    if name == 'tanh':
        # sech^2(x) = 1 - tanh^2(x)
        return chain(Add(Number(1), Mul(Number(-1), Pow(Func('tanh', [arg]), Number(2)))))

    if name == 'exp':
        return chain(Func('exp', [arg]))

    if name == 'log':
        return chain(Pow(arg, Number(-1)))

    if name == 'log10':
        from pythonaibrain_cas.core.operations import log
        import math
        return chain(Mul(Pow(arg, Number(-1)), Pow(log(Number(10)), Number(-1))))

    if name == 'sqrt':
        # 1/(2*sqrt(x))
        return chain(Mul(Number(Fraction(1, 2)), Pow(Func('sqrt', [arg]), Number(-1))))

    if name == 'Abs':
        # d/dx |x| = sign(x) for real x
        return chain(Func('sign', [arg]))

    # Unknown function: symbolic derivative
    return Mul(Func(f"D[{name}]", [arg]), _diff_once(arg, var))
