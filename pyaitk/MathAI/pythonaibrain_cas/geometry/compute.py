"""
geometry/compute.py
===================
Geometric computations: distances, areas, intersections.
"""

from __future__ import annotations
import math
from fractions import Fraction
from typing import List, Optional, Tuple, Union

from pythonaibrain_cas.core.expression import Expr, Number, _coerce
from pythonaibrain_cas.core.operations import Add, Mul, Pow, Func
from pythonaibrain_cas.geometry.objects import Point, Line, Circle, Polygon


def distance(obj1, obj2) -> Expr:
    """
    Compute distance between two geometric objects.

    Supported:
    - Point to Point
    - Point to Line
    - Line to Line (parallel lines only; returns 0 for intersecting)
    """
    if isinstance(obj1, Point) and isinstance(obj2, Point):
        return obj1.distance_to(obj2)

    if isinstance(obj1, Point) and isinstance(obj2, Line):
        return obj2.distance_to_point(obj1)

    if isinstance(obj1, Line) and isinstance(obj2, Point):
        return obj1.distance_to_point(obj2)

    if isinstance(obj1, Line) and isinstance(obj2, Line):
        if not obj1.is_parallel(obj2):
            return Number(0)
        # Parallel: pick point on line1, find distance to line2
        try:
            a, b, c1 = (obj1.a.evalf().real, obj1.b.evalf().real, obj1.c.evalf().real)
            c2 = obj2.c.evalf().real
            denom = math.sqrt(a**2 + b**2)
            d = abs(c1 - c2) / denom if denom > 1e-12 else 0
            return Number(Fraction(d).limit_denominator(10**6))
        except Exception:
            return Number(0)

    raise TypeError(f"Cannot compute distance between {type(obj1)} and {type(obj2)}")


def area(obj) -> Expr:
    """
    Compute area of a geometric object.

    Supported: Circle, Polygon, Triangle (three Points).
    """
    if isinstance(obj, Circle):
        return obj.area

    if isinstance(obj, Polygon):
        return obj.area

    if isinstance(obj, (list, tuple)) and len(obj) == 3 and all(isinstance(p, Point) for p in obj):
        # Triangle area via cross product
        A, B, C = obj
        ab = (Add(B.x, Mul(Number(-1), A.x)), Add(B.y, Mul(Number(-1), A.y)))
        ac = (Add(C.x, Mul(Number(-1), A.x)), Add(C.y, Mul(Number(-1), A.y)))
        cross = Add(Mul(ab[0], ac[1]), Mul(Number(-1), Mul(ab[1], ac[0])))
        return Mul(Number(Fraction(1, 2)), Func('Abs', [cross]))

    raise TypeError(f"Cannot compute area of {type(obj)}")


def intersection(obj1, obj2) -> Optional[Union[Point, List[Point], "Line"]]:
    """
    Compute intersection of two geometric objects.

    Supported:
    - Line ∩ Line   → Point or None
    - Line ∩ Circle → List[Point] (0, 1, or 2 points)
    - Circle ∩ Circle → List[Point]
    """
    if isinstance(obj1, Line) and isinstance(obj2, Line):
        return _line_line_intersection(obj1, obj2)

    if isinstance(obj1, Line) and isinstance(obj2, Circle):
        return _line_circle_intersection(obj1, obj2)

    if isinstance(obj1, Circle) and isinstance(obj2, Line):
        return _line_circle_intersection(obj2, obj1)

    if isinstance(obj1, Circle) and isinstance(obj2, Circle):
        return _circle_circle_intersection(obj1, obj2)

    raise TypeError(f"Cannot compute intersection of {type(obj1)} and {type(obj2)}")


def _line_line_intersection(l1: Line, l2: Line) -> Optional[Point]:
    """Solve a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0."""
    try:
        a1, b1, c1 = l1.a.evalf().real, l1.b.evalf().real, l1.c.evalf().real
        a2, b2, c2 = l2.a.evalf().real, l2.b.evalf().real, l2.c.evalf().real
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return None  # Parallel or coincident
        x = (-c1 * b2 + c2 * b1) / det
        y = (-a1 * c2 + a2 * c1) / det
        return Point(
            Number(Fraction(x).limit_denominator(10**6)),
            Number(Fraction(y).limit_denominator(10**6))
        )
    except Exception:
        return None


def _line_circle_intersection(line: Line, circle: Circle) -> List[Point]:
    """Find intersection points of line ax+by+c=0 with circle."""
    try:
        a, b, c = line.a.evalf().real, line.b.evalf().real, line.c.evalf().real
        h, k = circle.center.x.evalf().real, circle.center.y.evalf().real
        r = abs(circle.radius.evalf().real)

        points = []

        if abs(b) > 1e-12:
            # Express y from line: y = (-a*x - c) / b
            # Substitute into circle equation: (x-h)^2 + ((-a*x-c)/b - k)^2 = r^2
            # Expand and solve quadratic in x
            # Coefficients of Ax^2 + Bx + C = 0
            A_coef = 1 + (a / b) ** 2
            B_coef = -2 * h + 2 * (a / b) * (c / b + k)
            C_coef = h**2 + (c / b + k)**2 - r**2
            disc = B_coef**2 - 4 * A_coef * C_coef
            if disc < -1e-10:
                return []
            disc = max(0.0, disc)
            sq = math.sqrt(disc)
            for sign in (1, -1):
                x = (-B_coef + sign * sq) / (2 * A_coef)
                y = (-a * x - c) / b
                points.append(Point(
                    Number(Fraction(x).limit_denominator(10**6)),
                    Number(Fraction(y).limit_denominator(10**6))
                ))
                if disc < 1e-10:
                    break
        else:
            # Vertical line: x = -c/a
            x = -c / a
            A_coef = 1
            B_coef = -2 * k
            C_coef = k**2 + (x - h)**2 - r**2
            disc = B_coef**2 - 4 * C_coef
            if disc < -1e-10:
                return []
            disc = max(0.0, disc)
            sq = math.sqrt(disc)
            for sign in (1, -1):
                y = (-B_coef + sign * sq) / 2
                points.append(Point(
                    Number(Fraction(x).limit_denominator(10**6)),
                    Number(Fraction(y).limit_denominator(10**6))
                ))
                if disc < 1e-10:
                    break

        return points
    except Exception:
        return []


def _circle_circle_intersection(c1: Circle, c2: Circle) -> List[Point]:
    """Find intersection points of two circles."""
    try:
        h1, k1 = c1.center.x.evalf().real, c1.center.y.evalf().real
        h2, k2 = c2.center.x.evalf().real, c2.center.y.evalf().real
        r1 = abs(c1.radius.evalf().real)
        r2 = abs(c2.radius.evalf().real)

        d = math.sqrt((h2 - h1)**2 + (k2 - k1)**2)

        if d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10 or d < 1e-12:
            return []

        # Radical axis (common chord line)
        # 2*(x2-x1)*x + 2*(y2-y1)*y + (r1^2 - r2^2 - x1^2 + x2^2 - y1^2 + y2^2) = 0
        a = 2 * (h2 - h1)
        b = 2 * (k2 - k1)
        c = r1**2 - r2**2 - h1**2 + h2**2 - k1**2 + k2**2

        radical_line = Line(a=Number(Fraction(a).limit_denominator(10**6)),
                            b=Number(Fraction(b).limit_denominator(10**6)),
                            c=Number(Fraction(c).limit_denominator(10**6)))
        return _line_circle_intersection(radical_line, c1)
    except Exception:
        return []


def triangle_area(A: Point, B: Point, C: Point) -> Expr:
    """Area of triangle ABC via shoelace."""
    return area([A, B, C])


def angle_between_lines(l1: Line, l2: Line) -> float:
    """Angle in degrees between two lines."""
    return math.degrees(l1.angle_with(l2))
