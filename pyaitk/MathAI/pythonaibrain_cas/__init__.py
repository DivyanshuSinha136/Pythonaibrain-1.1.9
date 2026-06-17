"""
Pythonaibrain-CAS: Production-Grade Computer Algebra System
============================================================
A modular, advanced Computer Algebra System built in pure Python.

Modules:
    core        - Expression tree, symbols, constants
    algebra     - Polynomial, rational, simplification
    calculus    - Differentiation, integration, limits, series
    geometry    - 2D/3D geometric objects and computations
    number_theory - Primes, factorization, modular arithmetic
    matrix      - Matrix algebra, decompositions, eigenvalues
    parser      - LaTeX and string expression parser
    utils       - Helpers, caching, pretty printing
    io          - Export to LaTeX, MathML, Python code
"""

__version__ = "1.0.0"
__author__ = "Divyanshu Sinha"
__license__ = "MIT"

from pythonaibrain_cas.core.expression import Expr, Symbol, Number, Constant
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func
from pythonaibrain_cas.core.context import CASContext

from pythonaibrain_cas.algebra.polynomial import Polynomial
from pythonaibrain_cas.algebra.simplify import simplify, expand, factor, collect
from pythonaibrain_cas.algebra.solve import solve, solve_system

from pythonaibrain_cas.calculus.differentiate import diff
from pythonaibrain_cas.calculus.integrate import integrate
from pythonaibrain_cas.calculus.limits import limit
from pythonaibrain_cas.calculus.series import series_expand

from pythonaibrain_cas.matrix.matrix import Matrix
from pythonaibrain_cas.matrix.linalg import det, inv, eigenvalues, eigenvectors, rank, lu_decomp, null_space

from pythonaibrain_cas.number_theory.primes import is_prime, primes_up_to, prime_factors, nth_prime, euler_totient, mobius, divisors, next_prime
from pythonaibrain_cas.number_theory.modular import mod_pow, mod_inv, chinese_remainder, discrete_log, legendre_symbol, jacobi_symbol, primitive_root, solve_linear_congruence

from pythonaibrain_cas.geometry.objects import Point, Line, Circle, Polygon
from pythonaibrain_cas.geometry.compute import distance, area, intersection

from pythonaibrain_cas.parser.parser import parse
from pythonaibrain_cas.io.latex import to_latex
from pythonaibrain_cas.io.mathml import to_mathml
from pythonaibrain_cas.utils.pretty import pprint

# Elementary function constructors
from pythonaibrain_cas.core.operations import (
    sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, exp, log, log10, sqrt,
)

# Predefined symbols and constants
x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
n, m, k = Symbol('n'), Symbol('m'), Symbol('k')

from pythonaibrain_cas.core.expression import (
    E as e_const, Pi as pi, ImaginaryUnit as I, Infinity as oo
)

__all__ = [
    # Core
    "Expr", "Symbol", "Number", "Constant",
    "Add", "Mul", "Pow", "Func", "CASContext",
    # Algebra
    "Polynomial", "simplify", "expand", "factor", "collect",
    "solve", "solve_system",
    # Calculus
    "diff", "integrate", "limit", "series_expand",
    # Matrix
    "Matrix", "det", "inv", "eigenvalues", "eigenvectors", "rank", "lu_decomp", "null_space",
    # Number Theory
    "is_prime", "primes_up_to", "prime_factors", "nth_prime", "euler_totient", "mobius", "divisors", "next_prime",
    "mod_pow", "mod_inv", "chinese_remainder", "discrete_log", "legendre_symbol", "jacobi_symbol", "primitive_root", "solve_linear_congruence",
    # Geometry
    "Point", "Line", "Circle", "Polygon",
    "distance", "area", "intersection",
    # Parser & IO
    "parse", "to_latex", "to_mathml", "pprint",
    # Elementary functions
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt",
    # Prebuilt symbols
    "x", "y", "z", "a", "b", "c", "n", "m", "k",
    "e_const", "pi", "I", "oo",
]
