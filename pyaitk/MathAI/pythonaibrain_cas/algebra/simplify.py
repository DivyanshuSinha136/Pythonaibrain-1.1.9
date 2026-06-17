"""
algebra/simplify.py
===================
Algebraic simplification: simplify, expand, factor, collect.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Dict, Optional
from pythonaibrain_cas.core.expression import Expr, Number, Symbol, Constant, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


def expand(expr: Expr) -> Expr:
    """
    Expand products and powers into sum of monomials.

    Examples:
        expand((x+1)**2)     -> x**2 + 2*x + 1
        expand((x+y)*(x-y))  -> x**2 - y**2
    """
    expr = _coerce(expr)
    result = _expand(expr)
    return _simplify_add(result) if isinstance(result, Add) else result


def _expand(expr: Expr) -> Expr:
    if isinstance(expr, Number) or isinstance(expr, Symbol) or isinstance(expr, Constant):
        return expr

    if isinstance(expr, Pow):
        base = _expand(expr.base)
        exp  = expr.exp
        if isinstance(exp, Number) and exp.is_integer and int(exp.value) > 0:
            n = int(exp.value)
            return _expand_pow(base, n)
        return Pow(base, _expand(exp))

    if isinstance(expr, Mul):
        # Distribute: (a+b)*c -> a*c + b*c
        expanded_args = [_expand(a) for a in expr.args]
        result = expanded_args[0]
        for arg in expanded_args[1:]:
            result = _distribute(result, arg)
        return result

    if isinstance(expr, Add):
        return Add(*[_expand(a) for a in expr.args])

    if isinstance(expr, Func):
        return Func(expr.name, [_expand(a) for a in expr.args])

    return expr


def _expand_pow(base: Expr, n: int) -> Expr:
    """Expand base^n using repeated multiplication."""
    if n == 0:
        return Number(1)
    if n == 1:
        return base
    if isinstance(base, Add):
        # Binomial or multinomial expansion
        result = base
        for _ in range(n - 1):
            result = _distribute(result, base)
        return result
    return Pow(base, Number(n))


def _distribute(a: Expr, b: Expr) -> Expr:
    """Distribute multiplication over addition: (a1+a2)*b -> a1*b + a2*b."""
    if isinstance(a, Add) and isinstance(b, Add):
        terms = []
        for ai in a.args:
            for bi in b.args:
                terms.append(_expand(Mul(ai, bi)))
        return Add(*terms) if terms else Number(0)
    if isinstance(a, Add):
        return Add(*[_expand(Mul(ai, b)) for ai in a.args])
    if isinstance(b, Add):
        return Add(*[_expand(Mul(a, bi)) for bi in b.args])
    return Mul(a, b)


def collect(expr: Expr, var: Symbol) -> Expr:
    """
    Collect terms by powers of var.

    Example: collect(a*x**2 + b*x**2 + c*x, x)  ->  (a+b)*x**2 + c*x
    """
    expanded = expand(expr)
    terms = _flatten_add(expanded)

    buckets: Dict[int, list] = {}
    for term in terms:
        deg, coeff_expr = _factor_out(term, var)
        buckets.setdefault(deg, []).append(coeff_expr)

    result_terms = []
    for deg in sorted(buckets.keys(), reverse=True):
        coeff = Add(*buckets[deg]) if len(buckets[deg]) > 1 else buckets[deg][0]
        if deg == 0:
            result_terms.append(coeff)
        elif deg == 1:
            result_terms.append(Mul(coeff, var))
        else:
            result_terms.append(Mul(coeff, Pow(var, Number(deg))))

    return Add(*result_terms) if result_terms else Number(0)


def _flatten_add(expr: Expr) -> list:
    if isinstance(expr, Add):
        result = []
        for a in expr.args:
            result.extend(_flatten_add(a))
        return result
    return [expr]


def _factor_out(term: Expr, var: Symbol) -> tuple:
    """Return (degree, coefficient_expr) for a monomial term in var."""
    if isinstance(term, Symbol):
        if term == var:
            return (1, Number(1))
        return (0, term)
    if isinstance(term, Number) or isinstance(term, Constant):
        return (0, term)
    if isinstance(term, Pow):
        if term.base == var and isinstance(term.exp, Number) and term.exp.is_integer:
            return (int(term.exp.value), Number(1))
    if isinstance(term, Mul):
        deg = 0
        coeff_parts = []
        for f in term.args:
            d, _ = _factor_out(f, var)
            if d > 0:
                deg += d
            else:
                coeff_parts.append(f)
        coeff = Mul(*coeff_parts) if coeff_parts else Number(1)
        return (deg, coeff)
    return (0, term)


def simplify(expr: Expr) -> Expr:
    """
    Heuristic simplification: combines algebraic identities,
    cancels common factors, and applies trig identities.
    """
    expr = _coerce(expr)
    expr = _simplify_pass(expr)
    return expr


def _simplify_pass(expr: Expr) -> Expr:
    if isinstance(expr, (Number, Symbol, Constant)):
        return expr

    if isinstance(expr, Add):
        args = [_simplify_pass(a) for a in expr.args]
        return _simplify_add(Add(*args))

    if isinstance(expr, Mul):
        args = [_simplify_pass(a) for a in expr.args]
        return _simplify_mul(Mul(*args))

    if isinstance(expr, Pow):
        base = _simplify_pass(expr.base)
        exp  = _simplify_pass(expr.exp)
        return _simplify_pow(Pow(base, exp))

    if isinstance(expr, Func):
        args = [_simplify_pass(a) for a in expr.args]
        return _simplify_func(Func(expr.name, args))

    return expr


def _simplify_add(expr: Expr) -> Expr:
    if not isinstance(expr, Add):
        return expr

    # Combine like terms using structural hash as key: 2x + 3x -> 5x
    # We use a list of (key_expr, coeff) pairs and match via __eq__
    like_keys: list = []   # list of Expr keys
    like_coeffs: list = [] # list of Fraction
    const = Fraction(0)

    for term in expr.args:
        key, coeff = _split_coeff(term)
        if key is None:
            const += coeff
        else:
            # Find existing key using structural equality
            found = False
            for i, existing_key in enumerate(like_keys):
                if existing_key == key:
                    like_coeffs[i] += coeff
                    found = True
                    break
            if not found:
                like_keys.append(key)
                like_coeffs.append(coeff)

    terms = []
    if const != 0:
        terms.append(Number(const))
    for key, coeff in zip(like_keys, like_coeffs):
        if coeff == 0:
            continue
        if coeff == 1:
            terms.append(key)
        else:
            terms.append(Mul(Number(coeff), key))

    if not terms:
        return Number(0)
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _split_coeff(term: Expr):
    """Return (base_expr, coefficient) by separating numeric factor."""
    if isinstance(term, Number):
        return None, term.value
    if isinstance(term, Mul):
        if isinstance(term.args[0], Number):
            rest = Mul(*term.args[1:]) if len(term.args) > 2 else term.args[1]
            return rest, term.args[0].value
    return term, Fraction(1)


def _simplify_mul(expr: Expr) -> Expr:
    if not isinstance(expr, Mul):
        return expr
    # Already canonicalised in Mul.__new__
    return expr


def _simplify_pow(expr: Expr) -> Expr:
    if not isinstance(expr, Pow):
        return expr

    base, exp = expr.base, expr.exp

    # (x^a)^b -> x^(a*b)
    if isinstance(base, Pow):
        new_exp = Mul(base.exp, exp)
        return Pow(base.base, new_exp)

    return expr


def _simplify_func(expr: Func) -> Expr:
    name = expr.name
    arg  = expr.args[0] if expr.args else None

    # Trig identities on numbers
    if arg is not None and isinstance(arg, Number):
        try:
            return Number(expr.evalf())
        except Exception:
            pass

    # sin(0) = 0, cos(0) = 1
    if name == 'sin' and isinstance(arg, Number) and arg.is_zero:
        return Number(0)
    if name == 'cos' and isinstance(arg, Number) and arg.is_zero:
        return Number(1)

    # exp(log(x)) = x, log(exp(x)) = x
    if name == 'exp' and isinstance(arg, Func) and arg.name == 'log':
        return arg.args[0]
    if name == 'log' and isinstance(arg, Func) and arg.name == 'exp':
        return arg.args[0]

    return expr


def factor(expr: Expr, var: Optional[Symbol] = None) -> Expr:
    """
    Factor a polynomial expression.
    Uses rational root theorem + synthetic division.
    """
    from pythonaibrain_cas.algebra.polynomial import Polynomial

    if var is None:
        syms = list(expr.free_symbols())
        if not syms:
            return expr
        var = syms[0]

    try:
        poly = Polynomial.from_expr(expand(expr), var)
    except ValueError:
        return expr

    if poly.degree <= 1:
        return expr

    # Try rational roots: ±p/q where p|const, q|leading
    from fractions import Fraction
    lc = poly.leading_coeff
    const = poly.coeffs[0]

    if const == 0:
        # x is a factor
        deflated = Polynomial(poly.coeffs[1:], var)
        return Mul(var, factor(deflated.to_expr(), var))

    # Candidate rational roots
    def divisors(n):
        n = abs(int(n))
        if n == 0:
            return [1]
        return [i for i in range(1, n + 1) if n % i == 0]

    p_divs = divisors(const.numerator) if hasattr(const, 'numerator') else divisors(int(abs(const)))
    q_divs = divisors(lc.numerator)    if hasattr(lc,    'numerator') else divisors(int(abs(lc)))

    candidates = {Fraction(p, q) * sign
                  for p in p_divs for q in q_divs
                  for sign in (1, -1)}

    factors = []
    remaining = poly
    for r in sorted(candidates, key=lambda x: abs(float(x))):
        if remaining.degree <= 0:
            break
        if remaining.evaluate(complex(float(r))).real < 1e-10 and remaining.evaluate(complex(float(r))).imag < 1e-10:
            # r is a root; (x - r) is a factor
            root_poly = Polynomial([-r, Fraction(1)], var)
            q, rem = remaining.divmod(root_poly)
            if rem.is_zero or all(abs(float(c)) < 1e-12 for c in rem.coeffs):
                factors.append(Add(var, Number(-r)))
                remaining = q

    if not factors:
        return expr

    result = remaining.to_expr()
    for f in factors:
        result = Mul(result, f)
    return result
