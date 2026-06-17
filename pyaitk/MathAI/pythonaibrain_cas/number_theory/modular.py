"""
number_theory/modular.py
========================
Modular arithmetic: power, inverse, CRT, discrete log, Legendre symbol.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple


def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    Fast modular exponentiation: base^exp mod mod.
    Uses Python's built-in pow for efficiency.
    """
    if mod == 1:
        return 0
    return pow(base, exp, mod)


def mod_inv(a: int, m: int) -> int:
    """
    Modular multiplicative inverse: a^(-1) mod m.
    Requires gcd(a, m) = 1.
    Uses extended Euclidean algorithm.

    Raises
    ------
    ValueError if inverse doesn't exist.
    """
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse of {a} mod {m} does not exist (gcd={g})")
    return x % m


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) s.t. a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def chinese_remainder(remainders: List[int], moduli: List[int]) -> int:
    """
    Chinese Remainder Theorem solver.

    Find x such that:
        x ≡ r_i (mod m_i)  for all i

    Parameters
    ----------
    remainders : list of r_i
    moduli     : list of m_i (must be pairwise coprime)

    Returns
    -------
    Smallest non-negative x satisfying all congruences.

    Examples
    --------
    >>> chinese_remainder([2, 3, 2], [3, 5, 7])
    23
    """
    if len(remainders) != len(moduli):
        raise ValueError("Lists must have equal length")

    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        yi = mod_inv(Mi, m)
        x += r * Mi * yi

    return x % M


def discrete_log(g: int, h: int, p: int) -> Optional[int]:
    """
    Baby-step giant-step algorithm for discrete logarithm.
    Finds x such that g^x ≡ h (mod p).

    Parameters
    ----------
    g : generator
    h : target
    p : prime modulus

    Returns
    -------
    x if found, None otherwise.

    Time/Space: O(sqrt(p))
    """
    import math
    m = math.isqrt(p) + 1

    # Baby steps: compute g^j for j = 0..m
    baby = {}
    gj = 1
    for j in range(m):
        baby[gj] = j
        gj = (gj * g) % p

    # Giant step factor: g^(-m)
    g_inv_m = mod_pow(mod_inv(g, p), m, p)

    # Giant steps: h * g^(-im)
    gamma = h
    for i in range(m):
        if gamma in baby:
            return (i * m + baby[gamma]) % (p - 1)
        gamma = (gamma * g_inv_m) % p

    return None


def legendre_symbol(a: int, p: int) -> int:
    """
    Legendre symbol (a/p): 0 if p|a, 1 if a is QR mod p, -1 otherwise.
    p must be an odd prime.
    """
    if p == 2:
        raise ValueError("p must be an odd prime")
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def jacobi_symbol(a: int, n: int) -> int:
    """
    Jacobi symbol (a/n) generalizing Legendre symbol to odd n > 0.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def primitive_root(p: int) -> int:
    """Find the smallest primitive root modulo prime p."""
    from pythonaibrain_cas.number_theory.primes import prime_factors, euler_totient
    if p == 2:
        return 1
    phi = p - 1  # p is prime
    factors = prime_factors(phi)

    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"No primitive root found for {p}")


def solve_linear_congruence(a: int, b: int, m: int) -> List[int]:
    """
    Solve a*x ≡ b (mod m).
    Returns list of solutions in [0, m).
    """
    g = math.gcd(a, m)
    if b % g != 0:
        return []  # No solution
    a //= g
    b //= g
    m //= g
    x0 = (b * mod_inv(a, m)) % m
    return [x0 + i * m for i in range(g)]


def isqrt_exact(n: int) -> Optional[int]:
    """Return integer square root if n is a perfect square, else None."""
    r = math.isqrt(n)
    return r if r * r == n else None
