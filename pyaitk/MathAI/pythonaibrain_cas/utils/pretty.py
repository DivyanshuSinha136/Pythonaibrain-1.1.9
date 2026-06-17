"""
utils/pretty.py
===============
Pretty-printing for CAS expressions, matrices, and results.
"""

from __future__ import annotations
from typing import Any


def pprint(obj: Any, mode: str = 'str') -> None:
    """
    Pretty-print a CAS object to stdout.

    Parameters
    ----------
    obj  : Expr, Matrix, list, or any
    mode : 'str' | 'latex' | 'repr'
    """
    if mode == 'latex':
        from pythonaibrain_cas.io.latex import to_latex
        try:
            print(f"LaTeX: {to_latex(obj)}")
            return
        except Exception:
            pass

    from pythonaibrain_cas.matrix.matrix import Matrix
    from pythonaibrain_cas.core.expression import Expr

    if isinstance(obj, Matrix):
        print(str(obj))
    elif isinstance(obj, list):
        if not obj:
            print("[]")
        else:
            print(f"[{', '.join(str(v) for v in obj)}]")
    elif isinstance(obj, dict):
        parts = [f"  {str(k)} = {str(v)}" for k, v in obj.items()]
        print("{\n" + ",\n".join(parts) + "\n}")
    elif isinstance(obj, Expr):
        print(str(obj))
    else:
        print(repr(obj))


def expr_tree(expr, indent: int = 0) -> str:
    """
    Return a string representation of the expression tree.

    Example output:
        Add
          Mul
            Number(3)
            Symbol(x)
          Number(1)
    """
    from pythonaibrain_cas.core.expression import Expr, Number, Symbol, Constant
    from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func

    prefix = "  " * indent
    if isinstance(expr, Number):
        return f"{prefix}Number({expr})"
    if isinstance(expr, Symbol):
        return f"{prefix}Symbol({expr.name})"
    if isinstance(expr, Constant):
        return f"{prefix}Constant({expr.name})"
    if isinstance(expr, Add):
        children = "\n".join(expr_tree(a, indent + 1) for a in expr.args)
        return f"{prefix}Add\n{children}"
    if isinstance(expr, Mul):
        children = "\n".join(expr_tree(a, indent + 1) for a in expr.args)
        return f"{prefix}Mul\n{children}"
    if isinstance(expr, Pow):
        base = expr_tree(expr.base, indent + 1)
        exp  = expr_tree(expr.exp, indent + 1)
        return f"{prefix}Pow\n{base}\n{exp}"
    if isinstance(expr, Func):
        children = "\n".join(expr_tree(a, indent + 1) for a in expr.args)
        return f"{prefix}Func({expr.name})\n{children}"
    return f"{prefix}{type(expr).__name__}({expr})"


def print_tree(expr) -> None:
    """Print the expression tree to stdout."""
    print(expr_tree(expr))
