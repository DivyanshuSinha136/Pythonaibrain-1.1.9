"""
matrix/linalg.py
================
Linear algebra operations: determinant, inverse, rank, eigenvalues,
eigenvectors, LU/QR decomposition.
"""

from __future__ import annotations
import math
import cmath
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from pythonaibrain_cas.core.expression import Expr, Number, Symbol, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow
from pythonaibrain_cas.matrix.matrix import Matrix


def det(A: Matrix) -> Expr:
    """
    Compute determinant using cofactor expansion (exact for symbolic matrices).
    For large numeric matrices, uses LU decomposition.
    """
    if A.rows != A.cols:
        raise ValueError("Determinant only defined for square matrices")
    n = A.rows
    if n == 1:
        return A[0, 0]
    if n == 2:
        return Add(
            Mul(A[0, 0], A[1, 1]),
            Mul(Number(-1), Mul(A[0, 1], A[1, 0]))
        )
    if n == 3:
        return _det3(A)
    # General: Gaussian elimination on a copy
    return _det_gauss(A)


def _det3(A: Matrix) -> Expr:
    """3×3 Sarrus rule."""
    a, b, c = A[0,0], A[0,1], A[0,2]
    d, e, f = A[1,0], A[1,1], A[1,2]
    g, h, i = A[2,0], A[2,1], A[2,2]
    return Add(
        Mul(a, Add(Mul(e, i), Mul(Number(-1), Mul(f, h)))),
        Mul(Number(-1), Mul(b, Add(Mul(d, i), Mul(Number(-1), Mul(f, g))))),
        Mul(c, Add(Mul(d, h), Mul(Number(-1), Mul(e, g)))),
    )


def _det_gauss(A: Matrix) -> Expr:
    """Gaussian elimination determinant with floating-point fallback."""
    try:
        import numpy as np
        arr = A.to_numpy()
        return Number(Fraction(float(np.linalg.det(arr))).limit_denominator(10**6))
    except ImportError:
        # Manual float elimination
        n = A.rows
        data = [[float(A[i, j].evalf().real) for j in range(n)] for i in range(n)]
        sign = 1
        for col in range(n):
            pivot = None
            for row in range(col, n):
                if abs(data[row][col]) > 1e-12:
                    pivot = row
                    break
            if pivot is None:
                return Number(0)
            if pivot != col:
                data[col], data[pivot] = data[pivot], data[col]
                sign *= -1
            pv = data[col][col]
            for row in range(col + 1, n):
                factor = data[row][col] / pv
                for k in range(col, n):
                    data[row][k] -= factor * data[col][k]
        d = sign
        for i in range(n):
            d *= data[i][i]
        return Number(Fraction(d).limit_denominator(10**6))


def inv(A: Matrix) -> Matrix:
    """
    Compute matrix inverse via Gauss–Jordan elimination.
    Raises ValueError if A is singular.
    """
    if A.rows != A.cols:
        raise ValueError("Only square matrices are invertible")
    n = A.rows

    try:
        import numpy as np
        arr = A.to_numpy()
        arr_inv = np.linalg.inv(arr)
        return Matrix([[Number(Fraction(float(arr_inv[i, j])).limit_denominator(10**6))
                        for j in range(n)] for i in range(n)])
    except ImportError:
        pass

    # Pure Python Gauss–Jordan
    aug = [[float(A[i, j].evalf().real) for j in range(n)] +
           [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if abs(aug[row][col]) > 1e-12:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for row in range(n):
            if row != col and abs(aug[row][col]) > 1e-12:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]

    return Matrix([[Number(Fraction(aug[i][n + j]).limit_denominator(10**6))
                    for j in range(n)] for i in range(n)])


def rank(A: Matrix) -> int:
    """Compute the rank of A via row reduction."""
    try:
        import numpy as np
        return int(np.linalg.matrix_rank(A.to_numpy()))
    except ImportError:
        pass
    # Manual row echelon
    n, m = A.rows, A.cols
    data = [[float(A[i, j].evalf().real) for j in range(m)] for i in range(n)]
    r = 0
    for col in range(m):
        pivot = None
        for row in range(r, n):
            if abs(data[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        data[r], data[pivot] = data[pivot], data[r]
        pv = data[r][col]
        data[r] = [x / pv for x in data[r]]
        for row in range(n):
            if row != r and abs(data[row][col]) > 1e-10:
                factor = data[row][col]
                data[row] = [data[row][k] - factor * data[r][k] for k in range(m)]
        r += 1
    return r


def eigenvalues(A: Matrix) -> List[Expr]:
    """
    Compute eigenvalues numerically.
    Returns list of Expr (Number for real, complex Add for imaginary).
    """
    if A.rows != A.cols:
        raise ValueError("Eigenvalues only for square matrices")
    try:
        import numpy as np
        vals = np.linalg.eigvals(A.to_numpy())
        from pythonaibrain_cas.algebra.solve import _complex_to_expr
        return [_complex_to_expr(complex(v)) for v in vals]
    except ImportError:
        raise NotImplementedError("numpy required for eigenvalues; install with: pip install numpy")


def eigenvectors(A: Matrix) -> List[Tuple[Expr, Matrix]]:
    """
    Compute eigenvalue–eigenvector pairs.
    Returns list of (eigenvalue, column Matrix).
    """
    if A.rows != A.cols:
        raise ValueError("Eigenvectors only for square matrices")
    try:
        import numpy as np
        vals, vecs = np.linalg.eig(A.to_numpy())
        from pythonaibrain_cas.algebra.solve import _complex_to_expr
        result = []
        for i in range(len(vals)):
            ev = _complex_to_expr(complex(vals[i]))
            vec = Matrix([[Number(Fraction(float(vecs[j, i].real)).limit_denominator(10**6))]
                          for j in range(A.rows)])
            result.append((ev, vec))
        return result
    except ImportError:
        raise NotImplementedError("numpy required for eigenvectors")


def lu_decomp(A: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
    """
    LU decomposition with partial pivoting: PA = LU.
    Returns (P, L, U).
    """
    try:
        import numpy as np
        import scipy.linalg as la
        arr = A.to_numpy()
        P_arr, L_arr, U_arr = la.lu(arr)

        def arr_to_mat(x):
            n, m = x.shape
            return Matrix([[Number(Fraction(float(x[i, j])).limit_denominator(10**6))
                            for j in range(m)] for i in range(n)])
        return arr_to_mat(P_arr), arr_to_mat(L_arr), arr_to_mat(U_arr)
    except ImportError:
        raise NotImplementedError("scipy required for LU decomposition")


def null_space(A: Matrix) -> List[Matrix]:
    """Compute null space basis vectors."""
    try:
        import numpy as np
        arr = A.to_numpy()
        _, s, Vt = np.linalg.svd(arr)
        tol = max(arr.shape) * np.finfo(float).eps * s.max() if s.any() else 1e-10
        null_mask = s < tol
        null_vecs = Vt[null_mask]
        result = []
        for vec in null_vecs:
            result.append(Matrix([[Number(Fraction(float(v)).limit_denominator(10**6))]
                                  for v in vec]))
        return result
    except ImportError:
        raise NotImplementedError("numpy required for null space")
