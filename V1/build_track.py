"""
2D Track Builder

Data model used by the main app:
    Segment = Tuple[Tuple[float, float], Tuple[float, float]]
    A wall is ((x1, y1), (x2, y2)) in *pygame screen coordinates*
    (origin top-left, y increases downward).
"""
from typing import List, Tuple

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]

# ------------------------------------------------------------
# Basic rectangular arena + a few obstacles (matches the demo)
# ------------------------------------------------------------

def square_track(width: int, height: int, margin: int = 40, track_width: int = 100) -> List[Segment]:
    """Return a list of wall segments that define a simple arena.

    - Outer rectangle inset by `margin`
    - A horizontal barrier, a short slanted wall, and a vertical post
    """
    walls: List[Segment] = []

    # outer walls
    tl: Vec2 = (margin, margin)
    tr: Vec2 = (width - margin, margin)
    br: Vec2 = (width - margin, height - margin)
    bl: Vec2 = (margin, height - margin)
    
    walls.extend([
        (tl, tr), 
        (tr, br), 
        (br, bl), 
        (bl, tl)
    ])
    
    # inner walls
    walls.extend([
        # (x1, y2), (x2, y2)
        ((tl[0]+track_width, tl[1]+track_width),   # top left -> top right
         (tr[0]-track_width, tr[1]+track_width)),
        ((bl[0]+track_width, bl[1]-track_width),   # bottom left -> bottom right
         (br[0]-track_width, br[1]-track_width)),
        ((tl[0]+track_width, tl[1]+track_width),   # top left -> bottom left
         (bl[0]+track_width, bl[1]-track_width)),
        ((tr[0]-track_width, tr[1]+track_width),   # top right -> bottom right
         (br[0]-track_width, br[1]-track_width))        
        ])

    return walls

# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

def build_polyline(points: List[Vec2]) -> List[Segment]:
    """Turn a list of points into segments (p0->p1, p1->p2, ...).
    Does not close the loop; for loops, append points[0] to the end.
    """
    if len(points) < 2:
        return []
    segs: List[Segment] = []
    for i in range(len(points) - 1):
        segs.append((points[i], points[i + 1]))
    return segs


def build_loop(points: List[Vec2]) -> List[Segment]:
    """Closed loop from points (adds last->first segment)."""
    if not points:
        return []
    segs = build_polyline(points)
    segs.append((points[-1], points[0]))
    return segs

# Example: porting coordinates from another repo that listed many lines
# in (x1, y1, x2, y2) format. Keep in pygame coordinates (no Y flip).

def build_from_xyxy(lines: List[Tuple[float, float, float, float]]) -> List[Segment]:
    """Convert [(x1,y1,x2,y2), ...] into our Segment list."""
    return [((x1, y1), (x2, y2)) for (x1, y1, x2, y2) in lines]


if __name__ == "__main__":
    # Tiny self-test
    demo = square_track(900, 600, 40)
    print(f"Built {len(demo)} wall segments")
