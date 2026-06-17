"""
io/mathml.py
============
Export CAS expressions to MathML strings.
"""

from __future__ import annotations
from pythonaibrain_cas.core.expression import Expr
from pythonaibrain_cas.matrix.matrix import Matrix


def to_mathml(expr, display: bool = True) -> str:
    """
    Convert expression to MathML string.

    Parameters
    ----------
    expr    : Expr or Matrix
    display : True for display math, False for inline

    Returns
    -------
    Complete MathML XML string
    """
    mode = 'display' if display else 'inline'
    if isinstance(expr, Matrix):
        inner = expr._mathml() if hasattr(expr, '_mathml') else str(expr)
    elif isinstance(expr, Expr):
        inner = expr._mathml()
    else:
        inner = f"<mtext>{expr}</mtext>"

    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{mode}">{inner}</math>'
