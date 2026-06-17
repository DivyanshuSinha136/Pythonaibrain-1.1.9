"""
io/latex.py
===========
Export CAS expressions to LaTeX strings.
"""

from __future__ import annotations
from pythonaibrain_cas.core.expression import Expr, Number, Symbol, Constant
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func
from pythonaibrain_cas.matrix.matrix import Matrix


def to_latex(expr) -> str:
    """
    Convert a CAS expression or Matrix to a LaTeX string.

    Parameters
    ----------
    expr : Expr or Matrix

    Returns
    -------
    LaTeX string (without surrounding $…$)

    Examples
    --------
    >>> to_latex(x**2 + 2*x + 1)
    'x^{2} + 2 x + 1'
    >>> to_latex(Matrix([[1,2],[3,4]]))
    '\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}'
    """
    if isinstance(expr, Matrix):
        return expr._latex()
    if isinstance(expr, Expr):
        return _to_latex(expr)
    raise TypeError(f"Cannot convert {type(expr)} to LaTeX")


def _to_latex(expr: Expr) -> str:
    return expr._latex()
