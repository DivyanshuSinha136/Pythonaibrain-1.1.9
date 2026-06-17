"""
parser/parser.py
================
Recursive-descent parser for algebraic expression strings.

Supports:
- Basic arithmetic: +, -, *, /, ^, **
- Named functions: sin, cos, tan, log, exp, sqrt, …
- Named constants: pi, e, I, oo
- Parentheses and implicit multiplication (2x, x(x+1))
- Rational numbers: 1/3, fractions
- Subscripted symbols: x_1, alpha_2

Grammar (EBNF):
    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := base ('^' | '**') factor | unary
    unary  := '-' factor | atom
    atom   := number | name | '(' expr ')'
    name   := identifier ('(' arglist ')')?
    arglist:= expr (',' expr)*
"""

from __future__ import annotations
import re
from fractions import Fraction
from typing import List, Optional

from pythonaibrain_cas.core.expression import Expr, Symbol, Number, Constant, _coerce
from pythonaibrain_cas.core.expression import Pi, E, ImaginaryUnit, Infinity
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


# Tokeniser
_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'   # 1: number (int or float)
    r'|([a-zA-Z_][a-zA-Z0-9_]*)'           # 2: identifier
    r'|(\*\*|[+\-*/^(),])'                 # 3: operator / punctuation
    r')'
)


def _tokenize(text: str) -> List[tuple]:
    """Return list of (type, value) tokens."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(1):
            tokens.append(('NUM', m.group(1)))
        elif m.group(2):
            tokens.append(('ID', m.group(2)))
        elif m.group(3):
            tokens.append(('OP', m.group(3)))
    tokens.append(('EOF', ''))
    return tokens


class _Parser:
    """Recursive-descent parser."""

    _CONSTANTS = {
        'pi': Pi, 'PI': Pi,
        'e': E,
        'I': ImaginaryUnit, 'i': ImaginaryUnit,
        'oo': Infinity, 'inf': Infinity, 'Inf': Infinity,
    }

    _FUNCTIONS = {
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'ln',
        'sqrt', 'abs', 'Abs', 'ceiling', 'floor', 'sign',
        'conjugate', 'gcd', 'lcm',
    }

    def __init__(self, tokens: List[tuple]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> tuple:
        return self._tokens[self._pos]

    def _consume(self, expected_type=None, expected_val=None) -> tuple:
        tok = self._tokens[self._pos]
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        if expected_val and tok[1] != expected_val:
            raise SyntaxError(f"Expected '{expected_val}', got '{tok[1]}'")
        self._pos += 1
        return tok

    def parse(self) -> Expr:
        result = self._parse_expr()
        if self._peek()[0] != 'EOF':
            raise SyntaxError(f"Unexpected token: {self._peek()}")
        return result

    def _parse_expr(self) -> Expr:
        """expr := term (('+' | '-') term)*"""
        left = self._parse_term()
        while self._peek() == ('OP', '+') or self._peek() == ('OP', '-'):
            op = self._consume()[1]
            right = self._parse_term()
            if op == '+':
                left = Add(left, right)
            else:
                left = Add(left, Mul(Number(-1), right))
        return left

    def _parse_term(self) -> Expr:
        """term := factor (('*' | '/') factor)*"""
        left = self._parse_factor()
        while self._peek() in (('OP', '*'), ('OP', '/'), ('OP', '**')):
            if self._peek() == ('OP', '**'):
                break  # ** handled in factor
            op = self._consume()[1]
            right = self._parse_factor()
            if op == '*':
                left = Mul(left, right)
            else:
                left = Mul(left, Pow(right, Number(-1)))
        return left

    def _parse_factor(self) -> Expr:
        """factor := unary (('^' | '**') factor)?"""
        base = self._parse_unary()
        if self._peek() in (('OP', '^'), ('OP', '**')):
            self._consume()
            exp = self._parse_factor()  # right-associative
            return Pow(base, exp)
        return base

    def _parse_unary(self) -> Expr:
        """unary := '-' factor | '+' factor | atom"""
        if self._peek() == ('OP', '-'):
            self._consume()
            return Mul(Number(-1), self._parse_factor())
        if self._peek() == ('OP', '+'):
            self._consume()
            return self._parse_factor()
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        """atom := number | name_or_func | '(' expr ')'"""
        tok = self._peek()

        if tok[0] == 'NUM':
            self._consume()
            s = tok[1]
            if '.' in s or 'e' in s or 'E' in s:
                return Number(Fraction(float(s)).limit_denominator(10**10))
            return Number(int(s))

        if tok[0] == 'ID':
            return self._parse_name()

        if tok == ('OP', '('):
            self._consume('OP', '(')
            expr = self._parse_expr()
            self._consume('OP', ')')
            # Check for implicit multiplication: (expr)(expr)
            if self._peek() == ('OP', '('):
                right = self._parse_atom()
                return Mul(expr, right)
            return expr

        raise SyntaxError(f"Unexpected token: {tok}")

    def _parse_name(self) -> Expr:
        """name := identifier ('(' arglist ')')?"""
        _, name = self._consume('ID')

        # Check if it's a function call
        if self._peek() == ('OP', '('):
            # Normalise 'ln' -> 'log'
            fname = 'log' if name == 'ln' else name
            self._consume('OP', '(')
            args = [self._parse_expr()]
            while self._peek() == ('OP', ','):
                self._consume('OP', ',')
                args.append(self._parse_expr())
            self._consume('OP', ')')
            return Func(fname, args)

        # Check constants
        if name in self._CONSTANTS:
            const = self._CONSTANTS[name]
            # Implicit multiplication check: e.g. pi2 → pi*2
            if self._peek()[0] == 'NUM':
                right = self._parse_atom()
                return Mul(const, right)
            return const

        # Symbol - check for implicit multiplication: 2x, x(x+1)
        sym = Symbol(name)
        if self._peek() == ('OP', '('):
            right = self._parse_atom()
            return Mul(sym, right)
        return sym


def parse(text: str) -> Expr:
    """
    Parse a string expression into a CAS Expr.

    Parameters
    ----------
    text : str
        Mathematical expression string.

    Returns
    -------
    Expr

    Examples
    --------
    >>> parse("x^2 + 2*x + 1")
    >>> parse("sin(pi/6)")
    >>> parse("(x+y)*(x-y)")
    >>> parse("3*x**2 - 7*x + 2")
    """
    tokens = _tokenize(text.strip())
    return _Parser(tokens).parse()
