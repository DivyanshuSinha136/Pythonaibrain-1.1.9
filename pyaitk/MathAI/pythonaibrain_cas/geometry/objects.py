"""
geometry/objects.py
===================
2D and 3D geometric objects: Point, Line, Circle, Polygon, Plane, Sphere.
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union

from pythonaibrain_cas.core.expression import Expr, Number, Symbol, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------

class Point:
    """
    N-dimensional point with symbolic or numeric coordinates.

    Examples
    --------
    >>> P = Point(1, 2)
    >>> Q = Point(4, 6)
    >>> P.distance_to(Q)    # 5
    """

    def __init__(self, *coords):
        self.coords: Tuple[Expr, ...] = tuple(_coerce(c) for c in coords)
        self.dim = len(self.coords)

    def __getitem__(self, i) -> Expr:
        return self.coords[i]

    @property
    def x(self) -> Expr:
        return self.coords[0]

    @property
    def y(self) -> Expr:
        return self.coords[1] if self.dim > 1 else Number(0)

    @property
    def z(self) -> Expr:
        return self.coords[2] if self.dim > 2 else Number(0)

    def distance_to(self, other: "Point") -> Expr:
        if self.dim != other.dim:
            raise ValueError("Points must have same dimension")
        sq_sum = sum(
            Pow(Add(a, Mul(Number(-1), b)), Number(2))
            for a, b in zip(self.coords, other.coords)
        )
        return Func('sqrt', [sq_sum])

    def midpoint(self, other: "Point") -> "Point":
        mid = [Mul(Number(Fraction(1, 2)), Add(a, b))
               for a, b in zip(self.coords, other.coords)]
        return Point(*mid)

    def translate(self, *delta) -> "Point":
        new = [Add(c, _coerce(d)) for c, d in zip(self.coords, delta)]
        return Point(*new)

    def reflect_over(self, other: "Point") -> "Point":
        new = [Add(Mul(Number(2), o), Mul(Number(-1), c))
               for c, o in zip(self.coords, other.coords)]
        return Point(*new)

    def __eq__(self, other) -> bool:
        return isinstance(other, Point) and self.coords == other.coords

    def __repr__(self):
        return f"Point({', '.join(str(c) for c in self.coords)})"

    def _latex(self) -> str:
        inner = ', '.join(c._latex() for c in self.coords)
        return rf'\left({inner}\right)'

    def evalf(self) -> Tuple[complex, ...]:
        return tuple(c.evalf() for c in self.coords)


# ---------------------------------------------------------------------------
# Line (2D)
# ---------------------------------------------------------------------------

class Line:
    """
    2D line defined by two points or by coefficients ax + by + c = 0.

    Attributes
    ----------
    a, b, c : Expr
        Coefficients in ax + by + c = 0
    """

    def __init__(self, p1: Optional[Point] = None, p2: Optional[Point] = None,
                 a=None, b=None, c=None):
        if p1 is not None and p2 is not None:
            # Line through two points: (y2-y1)(x-x1) - (x2-x1)(y-y1) = 0
            dx = Add(p2.x, Mul(Number(-1), p1.x))
            dy = Add(p2.y, Mul(Number(-1), p1.y))
            # dy*x - dx*y + (dx*y1 - dy*x1) = 0
            self.a = dy
            self.b = Mul(Number(-1), dx)
            self.c = Add(Mul(dx, p1.y), Mul(Mul(Number(-1), dy), p1.x))
            self._p1, self._p2 = p1, p2
        elif a is not None and b is not None and c is not None:
            self.a = _coerce(a)
            self.b = _coerce(b)
            self.c = _coerce(c)
            self._p1 = self._p2 = None
        else:
            raise ValueError("Provide either two points or (a, b, c) coefficients")

    @property
    def slope(self) -> Optional[Expr]:
        """Slope as Expr, None if vertical."""
        try:
            a_val = self.a.evalf().real
            b_val = self.b.evalf().real
            if abs(b_val) < 1e-12:
                return None
            return Mul(Mul(Number(-1), self.a), Pow(self.b, Number(-1)))
        except Exception:
            return Mul(Mul(Number(-1), self.a), Pow(self.b, Number(-1)))

    @property
    def y_intercept(self) -> Optional[Expr]:
        try:
            b_val = self.b.evalf().real
            if abs(b_val) < 1e-12:
                return None
        except Exception:
            pass
        return Mul(Mul(Number(-1), self.c), Pow(self.b, Number(-1)))

    def point_on_line(self, point: Point) -> bool:
        val = Add(Add(Mul(self.a, point.x), Mul(self.b, point.y)), self.c)
        try:
            return abs(val.evalf().real) < 1e-10
        except Exception:
            return False

    def distance_to_point(self, point: Point) -> Expr:
        """Distance from point to line = |ax+by+c| / sqrt(a²+b²)."""
        numer = Func('Abs', [Add(Add(Mul(self.a, point.x), Mul(self.b, point.y)), self.c)])
        denom = Func('sqrt', [Add(Pow(self.a, Number(2)), Pow(self.b, Number(2)))])
        return Mul(numer, Pow(denom, Number(-1)))

    def is_parallel(self, other: "Line") -> bool:
        try:
            a1, b1 = self.a.evalf().real, self.b.evalf().real
            a2, b2 = other.a.evalf().real, other.b.evalf().real
            return abs(a1 * b2 - a2 * b1) < 1e-10
        except Exception:
            return False

    def is_perpendicular(self, other: "Line") -> bool:
        try:
            a1, b1 = self.a.evalf().real, self.b.evalf().real
            a2, b2 = other.a.evalf().real, other.b.evalf().real
            return abs(a1 * a2 + b1 * b2) < 1e-10
        except Exception:
            return False

    def angle_with(self, other: "Line") -> float:
        """Angle in radians between this line and other."""
        try:
            a1, b1 = self.a.evalf().real, self.b.evalf().real
            a2, b2 = other.a.evalf().real, other.b.evalf().real
            cos_theta = abs(a1*a2 + b1*b2) / (
                math.sqrt(a1**2 + b1**2) * math.sqrt(a2**2 + b2**2)
            )
            cos_theta = min(1.0, max(-1.0, cos_theta))
            return math.acos(cos_theta)
        except Exception:
            return 0.0

    def __repr__(self):
        return f"Line({self.a}*x + {self.b}*y + {self.c} = 0)"

    def _latex(self) -> str:
        return rf"{self.a._latex()} x + {self.b._latex()} y + {self.c._latex()} = 0"


# ---------------------------------------------------------------------------
# Circle (2D)
# ---------------------------------------------------------------------------

class Circle:
    """
    Circle defined by center and radius: (x-h)² + (y-k)² = r²

    Parameters
    ----------
    center : Point (2D)
    radius : Expr or number
    """

    def __init__(self, center: Point, radius):
        if center.dim != 2:
            raise ValueError("Circle requires a 2D center")
        self.center = center
        self.radius = _coerce(radius)

    @property
    def area(self) -> Expr:
        from pythonaibrain_cas.core.expression import Pi
        return Mul(Pi, Pow(self.radius, Number(2)))

    @property
    def circumference(self) -> Expr:
        from pythonaibrain_cas.core.expression import Pi
        return Mul(Mul(Number(2), Pi), self.radius)

    def contains_point(self, p: Point) -> bool:
        try:
            d = self.center.distance_to(p).evalf().real
            r = abs(self.radius.evalf().real)
            return d <= r + 1e-10
        except Exception:
            return False

    def tangent_at(self, p: Point) -> Line:
        """Tangent line to circle at point p (must be on circle)."""
        dx = Add(p.x, Mul(Number(-1), self.center.x))
        dy = Add(p.y, Mul(Number(-1), self.center.y))
        # Tangent: dx*(x-cx) + dy*(y-cy) = 0
        # => dx*x + dy*y - (dx*cx + dy*cy) = 0
        c_val = Mul(Number(-1), Add(Mul(dx, self.center.x), Mul(dy, self.center.y)))
        return Line(a=dx, b=dy, c=c_val)

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

    def _latex(self) -> str:
        h = self.center.x._latex()
        k = self.center.y._latex()
        r = self.radius._latex()
        return rf"(x - {h})^2 + (y - {k})^2 = {r}^2"


# ---------------------------------------------------------------------------
# Polygon (2D)
# ---------------------------------------------------------------------------

class Polygon:
    """
    2D polygon defined by ordered list of vertices.

    Parameters
    ----------
    vertices : list of Point
    """

    def __init__(self, vertices: List[Point]):
        if len(vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        if any(v.dim != 2 for v in vertices):
            raise ValueError("All vertices must be 2D")
        self.vertices = vertices

    @property
    def num_sides(self) -> int:
        return len(self.vertices)

    @property
    def perimeter(self) -> Expr:
        total = Number(0)
        n = len(self.vertices)
        for i in range(n):
            d = self.vertices[i].distance_to(self.vertices[(i + 1) % n])
            total = Add(total, d)
        return total

    @property
    def area(self) -> Expr:
        """Shoelace formula."""
        n = len(self.vertices)
        terms = []
        for i in range(n):
            xi, yi = self.vertices[i].x, self.vertices[i].y
            xj, yj = self.vertices[(i + 1) % n].x, self.vertices[(i + 1) % n].y
            terms.append(Add(Mul(xi, yj), Mul(Number(-1), Mul(xj, yi))))
        total = terms[0]
        for t in terms[1:]:
            total = Add(total, t)
        return Mul(Number(Fraction(1, 2)), Func('Abs', [total]))

    def centroid(self) -> Point:
        n = len(self.vertices)
        cx = sum((v.x for v in self.vertices), Number(0))
        cy = sum((v.y for v in self.vertices), Number(0))
        return Point(Mul(Number(Fraction(1, n)), cx),
                     Mul(Number(Fraction(1, n)), cy))

    def is_convex(self) -> bool:
        """Check convexity via cross product sign consistency."""
        n = len(self.vertices)
        sign = None
        for i in range(n):
            A = self.vertices[i]
            B = self.vertices[(i + 1) % n]
            C = self.vertices[(i + 2) % n]
            try:
                ax, ay = A.x.evalf().real, A.y.evalf().real
                bx, by = B.x.evalf().real, B.y.evalf().real
                cx, cy = C.x.evalf().real, C.y.evalf().real
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                s = 1 if cross > 0 else -1
                if sign is None:
                    sign = s
                elif sign != s:
                    return False
            except Exception:
                pass
        return True

    def __repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"
