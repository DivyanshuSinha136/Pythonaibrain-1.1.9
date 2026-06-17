"""
number_theory/primes.py
=======================
Prime numbers, factorisation, Euler's totient, Möbius function.
"""

from __future__ import annotations
import math
from functools import lru_cache
from typing import Dict, List, Optional


def is_prime(n: int) -> bool:
    """
    Miller–Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981).
    """
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses sufficient for deterministic result up to 3.3×10^24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_up_to(n: int) -> List[int]:
    """Sieve of Eratosthenes up to n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]


def next_prime(n: int) -> int:
    """Return the smallest prime > n."""
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(candidate):
        candidate += 2 if candidate > 2 else 1
    return candidate


def prime_factors(n: int) -> Dict[int, int]:
    """
    Return prime factorisation as {prime: exponent}.
    Uses Pollard's rho for large factors.

    Examples
    --------
    >>> prime_factors(360)
    {2: 3, 3: 2, 5: 1}
    """
    if n <= 1:
        return {}
    factors: Dict[int, int] = {}

    def add_factor(p):
        factors[p] = factors.get(p, 0) + 1

    # Remove small primes
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            add_factor(p)
            n //= p

    if n == 1:
        return factors

    # Trial division up to sqrt
    i = 17
    while i * i <= n and i < 10**6:
        while n % i == 0:
            add_factor(i)
            n //= i
        i += 2

    if n == 1:
        return factors

    # Pollard's rho for remaining large factors
    stack = [n]
    while stack:
        num = stack.pop()
        if num == 1:
            continue
        if is_prime(num):
            add_factor(num)
        else:
            d = _pollard_rho(num)
            if d == num:
                # Couldn't factor, treat as prime
                add_factor(num)
            else:
                stack.extend([d, num // d])

    return factors


def _pollard_rho(n: int) -> int:
    """Pollard's rho algorithm for integer factorisation."""
    if n % 2 == 0:
        return 2
    from random import randint

    def f(x, c):
        return (x * x + c) % n

    for _ in range(50):
        x = randint(2, n - 1)
        y = x
        c = randint(1, n - 1)
        d = 1
        while d == 1:
            x = f(x, c)
            y = f(f(y, c), c)
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
    return n


def euler_totient(n: int) -> int:
    """Euler's totient function φ(n)."""
    result = n
    factors = set(prime_factors(n).keys())
    for p in factors:
        result -= result // p
    return result


def mobius(n: int) -> int:
    """Möbius function μ(n)."""
    if n == 1:
        return 1
    pf = prime_factors(n)
    # If any prime appears more than once, μ(n) = 0
    if any(e > 1 for e in pf.values()):
        return 0
    return (-1) ** len(pf)


def divisors(n: int) -> List[int]:
    """Return sorted list of all positive divisors of n."""
    pf = prime_factors(n)
    result = [1]
    for p, e in pf.items():
        result = [d * p**k for d in result for k in range(e + 1)]
    return sorted(result)


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def is_perfect(n: int) -> bool:
    """True if n equals the sum of its proper divisors."""
    return n > 1 and sum(divisors(n)[:-1]) == n


def nth_prime(n: int) -> int:
    """Return the n-th prime (1-indexed)."""
    if n < 1:
        raise ValueError("n must be positive")
    # Estimate upper bound via prime number theorem
    if n < 6:
        return [2, 3, 5, 7, 11][n - 1]
    limit = int(n * (math.log(n) + math.log(math.log(n))) * 1.2) + 100
    ps = primes_up_to(limit)
    while len(ps) < n:
        limit *= 2
        ps = primes_up_to(limit)
    return ps[n - 1]
