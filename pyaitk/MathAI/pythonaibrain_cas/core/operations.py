"""
core/operations.py
==================
Compound expression nodes: Add, Mul, Pow, Func.
Each constructor performs lightweight canonicalisation so the tree
stays in a predictable normal form without full simplification.
"""

from __future__ import annotations
import math
import cmath
from fractions import Fraction
from functools import reduce
from typing import Dict, Optional, Sequence, Tuple

from pythonaibrain_cas.core.expression import Expr, Number, Symbol, Constant, _coerce
from pythonaibrain_cas.core.expression import Pi, E, ImaginaryUnit, Infinity


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class Add(Expr):
    """
    Flat, sorted sum: a₁ + a₂ + … + aₙ

    Canonicalisation at construction:
    - Flatten nested Adds
    - Merge Number summands into one constant term
    - Drop zero terms
    - Single-term Add collapses to that term
    """

    _precedence = 10

    def __new__(cls, *args: Expr) -> Expr:
        # Flatten
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, Add):
                flat.extend(a.args)
            else:
                flat.append(_coerce(a))

        # Collect numeric constant
        const = Fraction(0)
        rest: list[Expr] = []
        for term in flat:
            if isinstance(term, Number):
                const += term.value
            else:
                rest.append(term)

        if const != 0:
            rest.append(Number(const))

        if not rest:
            return Number(0)
        if len(rest) == 1:
            return rest[0]

        obj = object.__new__(cls)
        obj._args = tuple(sorted(rest, key=_sort_key))
        return obj

    @property
    def args(self):
        return self._args

    def _structural_eq(self, other):
        return isinstance(other, Add) and self._args == other._args

    def _hash(self):
        return hash(("Add", self._args))

    def subs(self, substitutions):
        new_args = tuple(a.subs(substitutions) for a in self._args)
        return Add(*new_args)

    def evalf(self, subs=None, prec=15):
        return sum(a.evalf(subs, prec) for a in self._args)

    def __str__(self):
        parts = []
        for term in self._args:
            s = str(term)
            if parts and not s.startswith('-'):
                parts.append('+')
            parts.append(s)
        return ' '.join(parts)

    def _latex(self):
        parts = []
        for term in self._args:
            s = term._latex()
            if parts and not s.startswith('-'):
                parts.append('+')
            parts.append(s)
        return ' '.join(parts)

    def _mathml(self):
        result = self._args[0]._mathml()
        for term in self._args[1:]:
            result += f"<mo>+</mo>{term._mathml()}"
        return f"<mrow>{result}</mrow>"


# ---------------------------------------------------------------------------
# Mul
# ---------------------------------------------------------------------------

class Mul(Expr):
    """
    Flat, sorted product: a₁ · a₂ · … · aₙ

    Canonicalisation:
    - Flatten nested Muls
    - Merge Number factors
    - Drop 1-factors
    - Zero check
    - Combine same-base Pow terms: x² · x³ → x⁵
    """

    _precedence = 20

    def __new__(cls, *args: Expr) -> Expr:
        # Flatten
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, Mul):
                flat.extend(a.args)
            else:
                flat.append(_coerce(a))

        # Collect numeric coefficient
        coeff = Fraction(1)
        rest: list[Expr] = []
        for f in flat:
            if isinstance(f, Number):
                coeff *= f.value
            else:
                rest.append(f)

        if coeff == 0:
            return Number(0)

        # Combine powers with same base: x^a * x^b -> x^(a+b)
        base_exp: dict = {}
        others: list[Expr] = []
        for f in rest:
            if isinstance(f, Pow):
                base, exp = f.base, f.exp
                key = _sort_key(base)
                if key in base_exp:
                    base_exp[key] = (base, Add(base_exp[key][1], exp))
                else:
                    base_exp[key] = (base, exp)
            elif isinstance(f, Symbol):
                key = _sort_key(f)
                if key in base_exp:
                    base_exp[key] = (f, Add(base_exp[key][1], Number(1)))
                else:
                    base_exp[key] = (f, Number(1))
            else:
                others.append(f)

        combined = [Pow(b, e) for (b, e) in base_exp.values()]
        combined.extend(others)

        if coeff != 1:
            combined.insert(0, Number(coeff))
        elif not combined:
            return Number(1)

        if not combined:
            return Number(coeff)
        if len(combined) == 1:
            return combined[0]

        obj = object.__new__(cls)
        # Put Number first, rest sorted
        nums = [x for x in combined if isinstance(x, Number)]
        non_nums = sorted([x for x in combined if not isinstance(x, Number)], key=_sort_key)
        obj._args = tuple(nums + non_nums)
        return obj

    @property
    def args(self):
        return self._args

    @property
    def coefficient(self) -> Number:
        if self._args and isinstance(self._args[0], Number):
            return self._args[0]
        return Number(1)

    @property
    def factors(self) -> Tuple[Expr, ...]:
        if self._args and isinstance(self._args[0], Number):
            return self._args[1:]
        return self._args

    def _structural_eq(self, other):
        return isinstance(other, Mul) and self._args == other._args

    def _hash(self):
        return hash(("Mul", self._args))

    def subs(self, substitutions):
        new_args = tuple(a.subs(substitutions) for a in self._args)
        return Mul(*new_args)

    def evalf(self, subs=None, prec=15):
        result = complex(1)
        for a in self._args:
            result *= a.evalf(subs, prec)
        return result

    def __str__(self):
        parts = []
        for i, f in enumerate(self._args):
            if isinstance(f, Number) and f.value == -1 and i == 0 and len(self._args) > 1:
                parts.append('-')
                continue
            s = str(f)
            if isinstance(f, Add):
                s = f'({s})'
            if parts and parts[-1] != '-':
                parts.append('*')
            parts.append(s)
        return ''.join(parts)

    def _latex(self):
        parts = []
        neg = False
        for i, f in enumerate(self._args):
            if isinstance(f, Number) and f.value == -1 and i == 0:
                neg = True
                continue
            s = f._latex()
            if isinstance(f, Add):
                s = rf'\left({s}\right)'
            parts.append(s)
        result = ' '.join(parts)
        return f"-{result}" if neg else result

    def _mathml(self):
        result = self._args[0]._mathml()
        for f in self._args[1:]:
            result += f"<mo>&middot;</mo>{f._mathml()}"
        return f"<mrow>{result}</mrow>"


# ---------------------------------------------------------------------------
# Pow
# ---------------------------------------------------------------------------

class Pow(Expr):
    """
    Power: base ** exp

    Canonicalisation:
    - x^0  → 1
    - x^1  → x
    - 0^x  → 0  (x ≠ 0)
    - 1^x  → 1
    - Number^Number evaluated exactly when safe
    """

    _precedence = 30

    def __new__(cls, base: Expr, exp: Expr) -> Expr:
        base = _coerce(base)
        exp  = _coerce(exp)

        # Numeric shortcuts
        if isinstance(exp, Number):
            if exp.is_zero:
                return Number(1)
            if exp.is_one:
                return base
        if isinstance(base, Number):
            if base.is_zero:
                return Number(0)
            if base.is_one:
                return Number(1)
            if isinstance(exp, Number) and exp.is_integer:
                e = int(exp.value)
                if abs(e) < 100:
                    return Number(Fraction(base.value ** e))

        obj = object.__new__(cls)
        obj._base = base
        obj._exp  = exp
        return obj

    @property
    def base(self):
        return self._base

    @property
    def exp(self):
        return self._exp

    @property
    def args(self):
        return (self._base, self._exp)

    def _structural_eq(self, other):
        return isinstance(other, Pow) and self._base == other._base and self._exp == other._exp

    def _hash(self):
        return hash(("Pow", self._base, self._exp))

    def subs(self, substitutions):
        return Pow(self._base.subs(substitutions), self._exp.subs(substitutions))

    def evalf(self, subs=None, prec=15):
        b = self._base.evalf(subs, prec)
        e = self._exp.evalf(subs, prec)
        return b ** e

    def __str__(self):
        bs = str(self._base)
        es = str(self._exp)
        if isinstance(self._base, (Add, Mul)):
            bs = f'({bs})'
        if isinstance(self._exp, (Add, Mul)):
            es = f'({es})'
        return f'{bs}**{es}'

    def _latex(self):
        bs = self._base._latex()
        es = self._exp._latex()
        if isinstance(self._base, (Add, Mul, Pow)):
            bs = rf'\left({bs}\right)'
        # Special case: square root
        if isinstance(self._exp, Number) and self._exp.value == Fraction(1, 2):
            return rf'\sqrt{{{bs[8:-7] if bs.startswith(r"\left(") else bs}}}'
        return rf'{bs}^{{{es}}}'

    def _mathml(self):
        return f"<msup>{self._base._mathml()}{self._exp._mathml()}</msup>"


# ---------------------------------------------------------------------------
# Func
# ---------------------------------------------------------------------------

# Known function metadata: name -> (latex_name, n_args, evaluator)
_FUNC_TABLE: Dict[str, Dict] = {
    'sin':   {'latex': r'\sin',   'nargs': 1, 'eval': cmath.sin},
    'cos':   {'latex': r'\cos',   'nargs': 1, 'eval': cmath.cos},
    'tan':   {'latex': r'\tan',   'nargs': 1, 'eval': cmath.tan},
    'asin':  {'latex': r'\arcsin','nargs': 1, 'eval': cmath.asin},
    'acos':  {'latex': r'\arccos','nargs': 1, 'eval': cmath.acos},
    'atan':  {'latex': r'\arctan','nargs': 1, 'eval': cmath.atan},
    'atan2': {'latex': r'\text{atan2}','nargs': 2, 'eval': lambda y,x: complex(math.atan2(y.real,x.real))},
    'sinh':  {'latex': r'\sinh',  'nargs': 1, 'eval': cmath.sinh},
    'cosh':  {'latex': r'\cosh',  'nargs': 1, 'eval': cmath.cosh},
    'tanh':  {'latex': r'\tanh',  'nargs': 1, 'eval': cmath.tanh},
    'exp':   {'latex': r'\exp',   'nargs': 1, 'eval': cmath.exp},
    'log':   {'latex': r'\ln',    'nargs': 1, 'eval': cmath.log},
    'log10': {'latex': r'\log_{10}', 'nargs': 1, 'eval': lambda z: cmath.log(z, 10)},
    'sqrt':  {'latex': r'\sqrt',  'nargs': 1, 'eval': cmath.sqrt},
    'abs':   {'latex': r'\left|', 'nargs': 1, 'eval': abs},
    'Abs':   {'latex': r'\left|', 'nargs': 1, 'eval': abs},
    'ceiling': {'latex': r'\lceil', 'nargs': 1, 'eval': lambda z: complex(math.ceil(z.real))},
    'floor': {'latex': r'\lfloor','nargs': 1, 'eval': lambda z: complex(math.floor(z.real))},
    'sign':  {'latex': r'\text{sign}', 'nargs': 1, 'eval': lambda z: complex(0 if z==0 else (1 if z.real>0 else -1))},
    'conjugate': {'latex': r'\overline', 'nargs': 1, 'eval': lambda z: z.conjugate()},
    'gcd':   {'latex': r'\gcd',   'nargs': 2, 'eval': lambda a,b: complex(math.gcd(int(a.real),int(b.real)))},
    'lcm':   {'latex': r'\text{lcm}','nargs': 2, 'eval': lambda a,b: complex(math.lcm(int(a.real),int(b.real)))},
}


class Func(Expr):
    """
    Named function application: f(arg₁, arg₂, …)

    Supports built-in functions with exact symbolic representation
    and numerical fallback via evalf.
    """

    _precedence = 50

    def __init__(self, name: str, args: Sequence[Expr]):
        self.name = name
        self._fargs = tuple(_coerce(a) for a in args)
        meta = _FUNC_TABLE.get(name, {})
        nargs = meta.get('nargs')
        if nargs is not None and len(self._fargs) != nargs:
            raise ValueError(f"{name} expects {nargs} argument(s), got {len(self._fargs)}")

    @property
    def args(self):
        return self._fargs

    def _structural_eq(self, other):
        return isinstance(other, Func) and self.name == other.name and self._fargs == other._fargs

    def _hash(self):
        return hash(("Func", self.name, self._fargs))

    def subs(self, substitutions):
        new_args = tuple(a.subs(substitutions) for a in self._fargs)
        return Func(self.name, new_args)

    def evalf(self, subs=None, prec=15):
        evaluated = [a.evalf(subs, prec) for a in self._fargs]
        meta = _FUNC_TABLE.get(self.name)
        if meta and 'eval' in meta:
            return meta['eval'](*evaluated)
        raise NotImplementedError(f"Cannot numerically evaluate {self.name}")

    def __str__(self):
        args_str = ', '.join(str(a) for a in self._fargs)
        return f"{self.name}({args_str})"

    def _latex(self):
        meta = _FUNC_TABLE.get(self.name, {})
        lname = meta.get('latex', rf'\text{{{self.name}}}')
        if self.name in ('abs', 'Abs'):
            return rf'\left|{self._fargs[0]._latex()}\right|'
        args_str = ', '.join(a._latex() for a in self._fargs)
        if self.name == 'sqrt':
            return rf'\sqrt{{{args_str}}}'
        return rf'{lname}\left({args_str}\right)'

    def _mathml(self):
        args_str = ''.join(a._mathml() for a in self._fargs)
        return f"<mrow><mi>{self.name}</mi><mo>(</mo>{args_str}<mo>)</mo></mrow>"


# ---------------------------------------------------------------------------
# Convenience function constructors (sin(x), cos(x), …)
# ---------------------------------------------------------------------------

def _make_func_constructor(name):
    def f(*args):
        return Func(name, args)
    f.__name__ = name
    f.__qualname__ = name
    return f

sin   = _make_func_constructor('sin')
cos   = _make_func_constructor('cos')
tan   = _make_func_constructor('tan')
asin  = _make_func_constructor('asin')
acos  = _make_func_constructor('acos')
atan  = _make_func_constructor('atan')
atan2 = _make_func_constructor('atan2')
sinh  = _make_func_constructor('sinh')
cosh  = _make_func_constructor('cosh')
tanh  = _make_func_constructor('tanh')
exp   = _make_func_constructor('exp')
log   = _make_func_constructor('log')
log10 = _make_func_constructor('log10')
sqrt  = _make_func_constructor('sqrt')


# ---------------------------------------------------------------------------
# Sort key for canonical ordering
# ---------------------------------------------------------------------------

def _sort_key(expr: Expr) -> tuple:
    """Deterministic sort key for expressions."""
    if isinstance(expr, Number):
        return (0, float(expr.value), '')
    if isinstance(expr, Constant):
        return (1, 0, expr.name)
    if isinstance(expr, Symbol):
        return (2, 0, expr.name)
    if isinstance(expr, Pow):
        return (3,) + _sort_key(expr.base) + (str(expr.exp),)
    if isinstance(expr, Func):
        return (4, 0, expr.name)
    if isinstance(expr, Mul):
        return (5, 0, str(expr))
    if isinstance(expr, Add):
        return (6, 0, str(expr))
    return (99, 0, str(expr))
