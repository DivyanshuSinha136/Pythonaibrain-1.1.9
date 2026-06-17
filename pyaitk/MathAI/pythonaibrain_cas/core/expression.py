"""
core/expression.py
==================
Immutable expression tree with hash-consing, automatic canonicalization,
and structural equality. All CAS objects inherit from Expr.
"""

from __future__ import annotations
import math
import cmath
from fractions import Fraction
from functools import cached_property, lru_cache
from typing import Any, Dict, FrozenSet, Iterator, Optional, Sequence, Tuple, Union
import re


# ---------------------------------------------------------------------------
# Base Expression
# ---------------------------------------------------------------------------

class Expr:
    """
    Abstract base for all CAS expressions.

    Design principles:
    - Immutable: all fields set once at construction
    - Hashable: expressions usable as dict keys / in sets
    - Comparable: structural equality via == (NOT numerical)
    - Canonical: constructor normalises trivial cases
    """

    # Subclasses set this to control display priority (parenthesisation)
    _precedence: int = 100

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

    # ------------------------------------------------------------------ #
    # Arithmetic operator overloads                                         #
    # ------------------------------------------------------------------ #

    def __add__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Add
        return Add(self, _coerce(other))

    def __radd__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Add
        return Add(_coerce(other), self)

    def __sub__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Add, Mul
        return Add(self, Mul(Number(-1), _coerce(other)))

    def __rsub__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Add, Mul
        return Add(_coerce(other), Mul(Number(-1), self))

    def __mul__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Mul
        return Mul(self, _coerce(other))

    def __rmul__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Mul
        return Mul(_coerce(other), self)

    def __truediv__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Mul, Pow
        return Mul(self, Pow(_coerce(other), Number(-1)))

    def __rtruediv__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Mul, Pow
        return Mul(_coerce(other), Pow(self, Number(-1)))

    def __pow__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Pow
        return Pow(self, _coerce(other))

    def __rpow__(self, other) -> "Expr":
        from pythonaibrain_cas.core.operations import Pow
        return Pow(_coerce(other), self)

    def __neg__(self) -> "Expr":
        from pythonaibrain_cas.core.operations import Mul
        return Mul(Number(-1), self)

    def __pos__(self) -> "Expr":
        return self

    def __abs__(self) -> "Expr":
        return Func("Abs", (self,))

    # ------------------------------------------------------------------ #
    # Comparison (structural, not numerical)                                #
    # ------------------------------------------------------------------ #

    def __eq__(self, other) -> bool:
        if not isinstance(other, Expr):
            other = _coerce(other)
        return self._structural_eq(other)

    def __hash__(self) -> int:
        return self._hash()

    def _structural_eq(self, other: "Expr") -> bool:
        raise NotImplementedError

    def _hash(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Evaluation / substitution                                            #
    # ------------------------------------------------------------------ #

    def subs(self, substitutions: Dict["Expr", "Expr"]) -> "Expr":
        """Return a new expression with substitutions applied."""
        raise NotImplementedError

    def evalf(self, subs: Optional[Dict] = None, prec: int = 15) -> complex:
        """Numerically evaluate the expression."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Traversal                                                            #
    # ------------------------------------------------------------------ #

    @property
    def args(self) -> Tuple["Expr", ...]:
        """Child sub-expressions."""
        return ()

    def walk(self) -> Iterator["Expr"]:
        """Pre-order traversal of the expression tree."""
        yield self
        for child in self.args:
            yield from child.walk()

    def free_symbols(self) -> FrozenSet["Symbol"]:
        """Return all free Symbol leaves in this expression."""
        syms: set = set()
        for node in self.walk():
            if isinstance(node, Symbol):
                syms.add(node)
        return frozenset(syms)

    def contains(self, target: "Expr") -> bool:
        return any(n == target for n in self.walk())

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        raise NotImplementedError

    def _latex(self) -> str:
        raise NotImplementedError

    def _mathml(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class Symbol(Expr):
    """A named indeterminate (variable)."""

    _precedence = 100

    def __init__(self, name: str, assumptions: Optional[Dict[str, bool]] = None):
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ValueError(f"Invalid symbol name: {name!r}")
        self.name = name
        self.assumptions: Dict[str, bool] = assumptions or {}

    def _structural_eq(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def _hash(self):
        return hash(("Symbol", self.name))

    def subs(self, substitutions):
        return substitutions.get(self, self)

    def evalf(self, subs=None, prec=15):
        if subs and self in subs:
            v = subs[self]
            return complex(v) if not isinstance(v, Expr) else v.evalf(subs, prec)
        raise ValueError(f"No value provided for symbol {self.name!r}")

    def __str__(self):
        return self.name

    def _latex(self):
        # Greek letter mapping
        greek = {
            'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
            'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta',
            'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota',
            'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
            'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi', 'rho': r'\rho',
            'sigma': r'\sigma', 'tau': r'\tau', 'upsilon': r'\upsilon',
            'phi': r'\phi', 'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
        }
        if self.name.lower() in greek:
            return greek[self.name.lower()]
        # Subscript: x_1 -> x_{1}
        if '_' in self.name:
            parts = self.name.split('_', 1)
            return f"{parts[0]}_{{{parts[1]}}}"
        return self.name

    def _mathml(self):
        return f"<mi>{self.name}</mi>"

    @property
    def args(self):
        return ()


class Number(Expr):
    """A rational number (stored as fractions.Fraction for exactness)."""

    _precedence = 100

    def __init__(self, value: Union[int, float, Fraction, str]):
        if isinstance(value, Fraction):
            self.value = value
        elif isinstance(value, int):
            self.value = Fraction(value)
        elif isinstance(value, float):
            self.value = Fraction(value).limit_denominator(10**10)
        elif isinstance(value, str):
            self.value = Fraction(value)
        else:
            raise TypeError(f"Cannot create Number from {type(value)}")

    @cached_property
    def is_integer(self):
        return self.value.denominator == 1

    @cached_property
    def is_zero(self):
        return self.value == 0

    @cached_property
    def is_one(self):
        return self.value == 1

    @cached_property
    def is_negative(self):
        return self.value < 0

    def _structural_eq(self, other):
        return isinstance(other, Number) and self.value == other.value

    def _hash(self):
        return hash(("Number", self.value))

    def subs(self, substitutions):
        return self

    def evalf(self, subs=None, prec=15):
        return complex(float(self.value))

    def __str__(self):
        if self.is_integer:
            return str(int(self.value))
        return str(self.value)

    def _latex(self):
        if self.is_integer:
            return str(int(self.value))
        return rf"\frac{{{self.value.numerator}}}{{{self.value.denominator}}}"

    def _mathml(self):
        return f"<mn>{self}</mn>"

    @property
    def args(self):
        return ()

    # Arithmetic on Numbers returns Numbers
    def __add__(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value)
        return super().__add__(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value)
        return super().__mul__(other)

    def __neg__(self):
        return Number(-self.value)


class Constant(Expr):
    """Named mathematical constant (pi, e, oo, I, etc.)."""

    _precedence = 100
    _registry: Dict[str, "Constant"] = {}

    def __init__(self, name: str, latex: str, numerical: complex):
        self.name = name
        self._latex_repr = latex
        self._numerical = numerical
        Constant._registry[name] = self

    def _structural_eq(self, other):
        return isinstance(other, Constant) and self.name == other.name

    def _hash(self):
        return hash(("Constant", self.name))

    def subs(self, substitutions):
        return substitutions.get(self, self)

    def evalf(self, subs=None, prec=15):
        return self._numerical

    def __str__(self):
        return self.name

    def _latex(self):
        return self._latex_repr

    def _mathml(self):
        return f"<mi>{self._latex_repr}</mi>"

    @property
    def args(self):
        return ()


# Predefined constants
Pi       = Constant("pi",  r"\pi",    complex(math.pi))
E        = Constant("e",   "e",       complex(math.e))
ImaginaryUnit = Constant("I", "i",    complex(0, 1))
Infinity = Constant("oo",  r"\infty", complex(float('inf')))
NegInfinity = Constant("-oo", r"-\infty", complex(float('-inf')))


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------

def _coerce(val) -> Expr:
    """Convert Python numeric types to Number, leave Expr unchanged."""
    if isinstance(val, Expr):
        return val
    if isinstance(val, (int, float, Fraction)):
        return Number(val)
    if isinstance(val, complex):
        from pythonaibrain_cas.core.operations import Add, Mul
        r, i = val.real, val.imag
        if i == 0:
            return Number(r)
        return Add(Number(r), Mul(Number(i), ImaginaryUnit))
    raise TypeError(f"Cannot coerce {type(val)} to Expr")
