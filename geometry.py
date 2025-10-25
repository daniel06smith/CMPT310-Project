"""
Geometry helpers for ray casting and vector math.
"""
from typing import Tuple, Optional

Vec2 = Tuple[float, float]


def cross2(a: Vec2, b: Vec2) -> float:
    return a[0]*b[1] - a[1]*b[0]


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0]-b[0], a[1]-b[1])


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0]+b[0], a[1]+b[1])


def mul(a: Vec2, s: float) -> Vec2:
    return (a[0]*s, a[1]*s)


def ray_segment_hit(p: Vec2, r: Vec2, q: Vec2, s: Vec2) -> Optional[float]:
    rxs = cross2(r, s)
    qp = sub(q, p)
    qpxr = cross2(qp, r)

    if abs(rxs) < 1e-9:
        return None

    t = cross2(qp, s) / rxs
    u = qpxr / rxs
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None
