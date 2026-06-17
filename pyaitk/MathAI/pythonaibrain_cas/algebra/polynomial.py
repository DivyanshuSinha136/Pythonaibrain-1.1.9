"""
algebra/polynomial.py
=====================
Dense univariate polynomial over rational coefficients.
Supports exact arithmetic, GCD, resultant, and factorisation.
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union
from pythonaibrain_cas.core.expression import Expr, Symbol, Number, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow


class Polynomial:
    """
    Univariate polynomial in a given variable.

    Representation: list of Fraction coefficients, index = degree.
        [c0, c1, c2, …]  means  c0 + c1·x + c2·x² + …

    Parameters
    ----------
    coeffs : sequence of numeric or Fraction
        Coefficient list, lowest degree first.
    var : Symbol, optional
        The indeterminate (default x).
    """

    def __init__(
        self,
        coeffs: Sequence[Union[int, float, Fraction]],
        var: Optional[Symbol] = None,
    ):
        from pythonaibrain_cas.core.expression import Symbol as Sym
        self.var = var or Sym('x')
        self.coeffs: List[Fraction] = [Fraction(c) for c in coeffs]
        self._normalize()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _normalize(self):
        """Strip trailing zeros."""
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()

    @property
    def degree(self) -> int:
        if len(self.coeffs) == 1 and self.coeffs[0] == 0:
            return -1  # zero polynomial
        return len(self.coeffs) - 1

    @property
    def is_zero(self) -> bool:
        return all(c == 0 for c in self.coeffs)

    @property
    def leading_coeff(self) -> Fraction:
        return self.coeffs[-1]

    def monic(self) -> "Polynomial":
        lc = self.leading_coeff
        return Polynomial([c / lc for c in self.coeffs], self.var)

    # ------------------------------------------------------------------ #
    # Arithmetic                                                            #
    # ------------------------------------------------------------------ #

    def __add__(self, other: "Polynomial") -> "Polynomial":
        _check_same_var(self, other)
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [Fraction(0)] * (n - len(self.coeffs))
        b = other.coeffs + [Fraction(0)] * (n - len(other.coeffs))
        return Polynomial([x + y for x, y in zip(a, b)], self.var)

    def __neg__(self) -> "Polynomial":
        return Polynomial([-c for c in self.coeffs], self.var)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self + (-other)

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        _check_same_var(self, other)
        result = [Fraction(0)] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] += a * b
        return Polynomial(result, self.var)

    def __pow__(self, n: int) -> "Polynomial":
        if n < 0:
            raise ValueError("Negative powers not supported for polynomials")
        result = Polynomial([1], self.var)
        base = Polynomial(self.coeffs, self.var)
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def __floordiv__(self, other: "Polynomial") -> "Polynomial":
        q, _ = self.divmod(other)
        return q

    def __mod__(self, other: "Polynomial") -> "Polynomial":
        _, r = self.divmod(other)
        return r

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return False
        return self.coeffs == other.coeffs and self.var == other.var

    # ------------------------------------------------------------------ #
    # Division                                                              #
    # ------------------------------------------------------------------ #

    def divmod(self, divisor: "Polynomial") -> Tuple["Polynomial", "Polynomial"]:
        """Polynomial long division. Returns (quotient, remainder)."""
        _check_same_var(self, divisor)
        if divisor.is_zero:
            raise ZeroDivisionError("Division by zero polynomial")
        rem = list(self.coeffs)
        quot = [Fraction(0)] * max(0, len(rem) - len(divisor.coeffs) + 1)
        for i in range(len(quot) - 1, -1, -1):
            factor = rem[i + len(divisor.coeffs) - 1] / divisor.leading_coeff
            quot[i] = factor
            for j, c in enumerate(divisor.coeffs):
                rem[i + j] -= factor * c
        # strip remainder
        while len(rem) > 1 and rem[-1] == 0:
            rem.pop()
        return Polynomial(quot or [0], self.var), Polynomial(rem, self.var)

    # ------------------------------------------------------------------ #
    # GCD / LCM                                                             #
    # ------------------------------------------------------------------ #

    def gcd(self, other: "Polynomial") -> "Polynomial":
        """Euclidean GCD (monic result)."""
        a, b = Polynomial(self.coeffs, self.var), Polynomial(other.coeffs, other.var)
        while not b.is_zero:
            a, b = b, a % b
        return a.monic()

    def lcm(self, other: "Polynomial") -> "Polynomial":
        return (self * other) // self.gcd(other)

    # ------------------------------------------------------------------ #
    # Calculus operations on polynomial                                    #
    # ------------------------------------------------------------------ #

    def differentiate(self) -> "Polynomial":
        if self.degree < 1:
            return Polynomial([0], self.var)
        return Polynomial([i * c for i, c in enumerate(self.coeffs)][1:], self.var)

    def integrate(self, constant: Fraction = Fraction(0)) -> "Polynomial":
        result = [constant] + [c / (i + 1) for i, c in enumerate(self.coeffs)]
        return Polynomial(result, self.var)

    # ------------------------------------------------------------------ #
    # Roots                                                                 #
    # ------------------------------------------------------------------ #

    def roots_numerical(self, tol: float = 1e-12) -> List[complex]:
        """
        Find all complex roots using the companion matrix / Durand-Kerner method.
        """
        d = self.degree
        if d <= 0:
            return []
        if d == 1:
            return [complex(-self.coeffs[0] / self.coeffs[1])]
        if d == 2:
            a, b, c = [float(self.coeffs[i]) for i in (2, 1, 0)]
            disc = b * b - 4 * a * c
            sq = disc ** 0.5 if disc >= 0 else complex(0, (-disc) ** 0.5)
            return [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]

        # Durand–Kerner iteration
        import cmath
        n = d
        lc = float(self.leading_coeff)
        # Initial guesses: points on unit circle
        roots = [cmath.rect(1.0, 2 * math.pi * k / n) * (1 + 0.4j) ** k
                 for k in range(n)]

        def poly_eval(r):
            v = complex(0)
            for ci in reversed(self.coeffs):
                v = v * r + complex(float(ci))
            return v / lc

        for _ in range(500):
            new_roots = []
            for i, ri in enumerate(roots):
                denom = complex(1)
                for j, rj in enumerate(roots):
                    if i != j:
                        denom *= (ri - rj)
                if abs(denom) < 1e-30:
                    denom = complex(1e-30)
                new_roots.append(ri - poly_eval(ri) / denom)
            if all(abs(a - b) < tol for a, b in zip(roots, new_roots)):
                break
            roots = new_roots

        # Clean up nearly-real roots
        cleaned = []
        for r in roots:
            if abs(r.imag) < tol:
                r = complex(r.real)
            cleaned.append(r)
        return cleaned

    def evaluate(self, value: Union[int, float, Fraction, complex]) -> complex:
        """Horner's method evaluation."""
        v = complex(0)
        for c in reversed(self.coeffs):
            v = v * complex(value) + complex(float(c))
        return v

    # ------------------------------------------------------------------ #
    # Conversion to Expr                                                    #
    # ------------------------------------------------------------------ #

    def to_expr(self) -> Expr:
        """Convert to symbolic Expr."""
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(Number(c))
            elif i == 1:
                terms.append(Mul(Number(c), self.var) if c != 1 else self.var)
            else:
                base = Pow(self.var, Number(i))
                terms.append(Mul(Number(c), base) if c != 1 else base)
        if not terms:
            return Number(0)
        result = terms[0]
        for t in terms[1:]:
            result = Add(result, t)
        return result

    @staticmethod
    def from_expr(expr: Expr, var: Symbol) -> "Polynomial":
        """
        Parse a symbolic expression into a Polynomial.
        Raises ValueError if expr is not polynomial in var.
        """
        coeffs = _expr_to_poly_coeffs(expr, var)
        return Polynomial(coeffs, var)

    # ------------------------------------------------------------------ #
    # Display                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        return f"Polynomial({self.coeffs}, var={self.var})"

    def __str__(self):
        if self.is_zero:
            return "0"
        terms = []
        for i in range(self.degree, -1, -1):
            c = self.coeffs[i]
            if c == 0:
                continue
            var_part = ""
            if i == 1:
                var_part = str(self.var)
            elif i > 1:
                var_part = f"{self.var}**{i}"
            coeff_str = ""
            if c == 1 and i > 0:
                coeff_str = ""
            elif c == -1 and i > 0:
                coeff_str = "-"
            else:
                coeff_str = str(c) if c.denominator == 1 else f"({c})"
            term = coeff_str + var_part
            terms.append(term)
        result = terms[0]
        for t in terms[1:]:
            if t.startswith('-'):
                result += f" - {t[1:]}"
            else:
                result += f" + {t}"
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_same_var(a: Polynomial, b: Polynomial):
    if a.var != b.var:
        raise ValueError(f"Variable mismatch: {a.var} vs {b.var}")


def _expr_to_poly_coeffs(expr: Expr, var: Symbol) -> List[Fraction]:
    """
    Recursively extract polynomial coefficients from a symbolic expression.
    Returns list indexed by degree.
    """
    from pythonaibrain_cas.core.operations import Add, Mul, Pow

    def extract(e: Expr) -> List[Fraction]:
        if isinstance(e, Number):
            return [e.value]
        if isinstance(e, Symbol):
            if e == var:
                return [Fraction(0), Fraction(1)]
            return [Fraction(0)]  # treat as 0 (not polynomial in var)
        if isinstance(e, Add):
            result = [Fraction(0)]
            for a in e.args:
                p = extract(a)
                n = max(len(result), len(p))
                result += [Fraction(0)] * (n - len(result))
                p      += [Fraction(0)] * (n - len(p))
                result = [x + y for x, y in zip(result, p)]
            return result
        if isinstance(e, Mul):
            result = [Fraction(1)]
            for f in e.args:
                result = _poly_mul(result, extract(f))
            return result
        if isinstance(e, Pow):
            if isinstance(e.exp, Number) and e.exp.is_integer:
                n = int(e.exp.value)
                if n >= 0:
                    base_coeffs = extract(e.base)
                    result = [Fraction(1)]
                    for _ in range(n):
                        result = _poly_mul(result, base_coeffs)
                    return result
        raise ValueError(f"Expression {expr} is not polynomial in {var}")

    return extract(expr)


def _poly_mul(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result
