"""
LiDAR sensor for distance measurement using ray casting.
"""
import math
from typing import List, Tuple, Optional
import numpy as np

from geometry import ray_segment_hit, sub

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]


# LiDAR parameters
ROOT2 = math.sqrt(2.0)
DIRS_8 = np.array([
    ( 0.0, -1.0),              # N  (up)
    ( 1.0/ROOT2, -1.0/ROOT2),  # NE
    ( 1.0,  0.0),              # E
    ( 1.0/ROOT2,  1.0/ROOT2),  # SE
    ( 0.0,  1.0),              # S
    (-1.0/ROOT2,  1.0/ROOT2),  # SW
    (-1.0,  0.0),              # W
    (-1.0/ROOT2, -1.0/ROOT2),  # NW
], dtype=np.float32)


def cast_ray(p: Vec2, dir_unit: Vec2, walls: List[Segment], r_max: float) -> float:
    """
    Cast a ray from point p in direction dir_unit and find the nearest wall intersection.
    
    Args:
        p: Starting point of the ray
        dir_unit: Unit direction vector
        walls: List of wall segments
        r_max: Maximum ray distance
    
    Returns:
        Distance to nearest wall (or r_max if no hit)
    """
    t_min: Optional[float] = None
    for (a, b) in walls:
        s = sub(b, a)
        t = ray_segment_hit(p, dir_unit, a, s)
        if t is not None and (t_min is None or t < t_min):
            t_min = t
    dist = t_min if t_min is not None else r_max
    return min(dist, r_max)


def lidar8(p: Vec2, walls: List[Segment], r_max: float = 100.0) -> np.ndarray:
    """
    Perform 8-direction LiDAR scan from position p.
    
    Args:
        p: Position to scan from
        walls: List of wall segments
        r_max: Maximum sensing range
    
    Returns:
        Numpy array of 8 normalized distances [0, 1]
    """
    dists = [cast_ray(p, (float(d[0]), float(d[1])), walls, r_max) for d in DIRS_8]
    return (np.array(dists, dtype=np.float32) / r_max).clip(0.0, 1.0)
