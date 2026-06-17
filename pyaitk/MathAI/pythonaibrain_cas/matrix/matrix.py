"""
matrix/matrix.py
================
Symbolic matrix class with exact rational arithmetic.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple, Union

from pythonaibrain_cas.core.expression import Expr, Number, Symbol, _coerce
from pythonaibrain_cas.core.operations import Add, Mul


class Matrix:
    """
    M×N matrix of symbolic expressions.

    Stored in row-major order: self._data[i][j] = element (i,j).

    Parameters
    ----------
    data : nested list / sequence of sequences of numbers or Expr

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> B = Matrix([[x, 0], [0, x]])
    >>> (A + B).det()
    """

    def __init__(self, data: Sequence[Sequence]):
        if not data:
            raise ValueError("Matrix cannot be empty")
        rows = len(data)
        cols = len(data[0])
        if any(len(row) != cols for row in data):
            raise ValueError("All rows must have equal length")
        self._data: List[List[Expr]] = [[_coerce(v) for v in row] for row in data]
        self.rows = rows
        self.cols = cols

    # ------------------------------------------------------------------ #
    # Factory methods                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def zeros(m: int, n: int) -> "Matrix":
        return Matrix([[Number(0)] * n for _ in range(m)])

    @staticmethod
    def eye(n: int) -> "Matrix":
        data = [[Number(1) if i == j else Number(0) for j in range(n)] for i in range(n)]
        return Matrix(data)

    @staticmethod
    def diag(*values) -> "Matrix":
        n = len(values)
        data = [[_coerce(values[i]) if i == j else Number(0) for j in range(n)] for i in range(n)]
        return Matrix(data)

    @staticmethod
    def from_func(m: int, n: int, func: Callable[[int, int], Expr]) -> "Matrix":
        return Matrix([[func(i, j) for j in range(n)] for i in range(m)])

    # ------------------------------------------------------------------ #
    # Element access                                                        #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx) -> Union[Expr, "Matrix"]:
        if isinstance(idx, tuple):
            r, c = idx
            if isinstance(r, int) and isinstance(c, int):
                return self._data[r][c]
            # Slicing
            rows = self._data[r] if isinstance(r, int) else self._data[r.start:r.stop:r.step]
            if isinstance(r, int):
                rows = [rows]
            sub = []
            for row in rows:
                if isinstance(c, int):
                    sub.append([row[c]])
                else:
                    sub.append(row[c])
            return Matrix(sub)
        if isinstance(idx, int):
            return Matrix([self._data[idx]])
        raise IndexError(f"Invalid index {idx}")

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(idx) == 2:
            r, c = idx
            self._data[r][c] = _coerce(value)
        else:
            raise IndexError("Use [row, col] indexing for assignment")

    # ------------------------------------------------------------------ #
    # Arithmetic                                                            #
    # ------------------------------------------------------------------ #

    def __add__(self, other: "Matrix") -> "Matrix":
        _check_shape(self, other)
        return Matrix([[Add(self._data[i][j], other._data[i][j])
                        for j in range(self.cols)]
                       for i in range(self.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        _check_shape(self, other)
        return Matrix([[Add(self._data[i][j], Mul(Number(-1), other._data[i][j]))
                        for j in range(self.cols)]
                       for i in range(self.rows)])

    def __mul__(self, other) -> "Matrix":
        if isinstance(other, (int, float, Fraction, Expr)):
            s = _coerce(other)
            return Matrix([[Mul(s, self._data[i][j]) for j in range(self.cols)]
                           for i in range(self.rows)])
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Shape mismatch: ({self.rows},{self.cols}) @ ({other.rows},{other.cols})")
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    entry = sum(
                        (Mul(self._data[i][k], other._data[k][j]) for k in range(self.cols)),
                        Number(0)
                    )
                    row.append(entry)
                result.append(row)
            return Matrix(result)
        raise TypeError(f"Cannot multiply Matrix by {type(other)}")

    def __rmul__(self, other) -> "Matrix":
        s = _coerce(other)
        return Matrix([[Mul(s, self._data[i][j]) for j in range(self.cols)]
                       for i in range(self.rows)])

    def __neg__(self) -> "Matrix":
        return self * Number(-1)

    def __pow__(self, n: int) -> "Matrix":
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be raised to a power")
        if n == 0:
            return Matrix.eye(self.rows)
        if n < 0:
            return self.inv() ** (-n)
        result = Matrix.eye(self.rows)
        base = Matrix([row[:] for row in self._data])
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def T(self) -> "Matrix":
        """Transpose."""
        return Matrix([[self._data[i][j] for i in range(self.rows)]
                       for j in range(self.cols)])

    def trace(self) -> Expr:
        if self.rows != self.cols:
            raise ValueError("Trace only defined for square matrices")
        total = self._data[0][0]
        for i in range(1, self.rows):
            total = Add(total, self._data[i][i])
        return total

    def to_numpy(self):
        """Convert to numpy array (requires numpy)."""
        import numpy as np
        def to_float(e):
            try:
                return float(e.evalf().real)
            except Exception:
                return float(complex(e.evalf()).real)
        return np.array([[to_float(self._data[i][j])
                          for j in range(self.cols)]
                         for i in range(self.rows)])

    def to_list(self) -> List[List[Expr]]:
        return [row[:] for row in self._data]

    # ------------------------------------------------------------------ #
    # Display                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        return f"Matrix({self.rows}×{self.cols})"

    def __str__(self):
        col_widths = []
        str_data = [[str(self._data[i][j]) for j in range(self.cols)]
                    for i in range(self.rows)]
        for j in range(self.cols):
            w = max(len(str_data[i][j]) for i in range(self.rows))
            col_widths.append(w)
        lines = []
        for row in str_data:
            cells = [cell.rjust(col_widths[j]) for j, cell in enumerate(row)]
            lines.append('[ ' + '  '.join(cells) + ' ]')
        return '\n'.join(lines)

    def _latex(self) -> str:
        rows_latex = []
        for row in self._data:
            rows_latex.append(' & '.join(e._latex() for e in row))
        body = r' \\ '.join(rows_latex)
        return rf'\begin{{pmatrix}} {body} \end{{pmatrix}}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_shape(a: Matrix, b: Matrix):
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"Shape mismatch: ({a.rows},{a.cols}) vs ({b.rows},{b.cols})")
