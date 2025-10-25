"""
- Arrow keys move the car (no rotation yet)
- 8 compass rays (N, NE, E, SE, S, SW, W, NW)
- Distances normalized to [0, 1] and displayed on screen
"""

import math
import sys
from typing import List, Tuple, Optional

import numpy as np
import pygame

from build_track import square_track

# ----------------------------
# Window & world parameters
# ----------------------------
WIDTH, HEIGHT = 900, 600
MARGIN = 40
BG_COLOR = (18, 18, 18)
WALL_COLOR = (230, 230, 230)
RAY_COLOR = (120, 200, 255)
CAR_COLOR = (255, 90, 90)
TEXT_COLOR = (230, 230, 230)

# LiDAR parameters
R_MAX = 100.0  # max sensing range in pixels
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

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]

# ----------------------------
# Geometry helpers
# ----------------------------

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


def cast_ray(p: Vec2, dir_unit: Vec2, walls: List[Segment], r_max: float) -> float:
    t_min: Optional[float] = None
    for (a, b) in walls:
        s = sub(b, a)
        t = ray_segment_hit(p, dir_unit, a, s)
        if t is not None and (t_min is None or t < t_min):
            t_min = t
    dist = t_min if t_min is not None else r_max
    return min(dist, r_max)


def lidar8(p: Vec2, walls: List[Segment], r_max: float = R_MAX) -> np.ndarray:
    dists = [cast_ray(p, (float(d[0]), float(d[1])), walls, r_max) for d in DIRS_8]
    return (np.array(dists, dtype=np.float32) / r_max).clip(0.0, 1.0)

# ----------------------------
# Drawing helpers
# ----------------------------

def draw_walls(screen: pygame.Surface, walls: List[Segment]):
    for (a, b) in walls:
        pygame.draw.line(screen, WALL_COLOR, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 2)


def draw_car(screen: pygame.Surface, pos: Vec2):
    pygame.draw.circle(screen, CAR_COLOR, (int(pos[0]), int(pos[1])), 6)


def draw_rays(screen: pygame.Surface, p: Vec2, norm_dists: np.ndarray):
    px, py = int(p[0]), int(p[1])
    for i, nd in enumerate(norm_dists):
        if not np.isfinite(nd):
            continue
        d = DIRS_8[i]
        dist = float(nd) * R_MAX
        end_x = int(round(p[0] + d[0] * dist))
        end_y = int(round(p[1] + d[1] * dist))
        pygame.draw.line(screen, RAY_COLOR, (px, py), (end_x, end_y), 1)
        pygame.draw.circle(screen, RAY_COLOR, (end_x, end_y), 2)


def draw_readout(screen: pygame.Surface, font: pygame.font.Font, norm_dists: np.ndarray):
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    lines = [f"{labels[i]}: {norm_dists[i]:.3f}" for i in range(8)]
    text = "  |  ".join(lines)
    surf = font.render(text, True, TEXT_COLOR)
    screen.blit(surf, (MARGIN, HEIGHT - MARGIN + 8 - 24))

# ----------------------------
# Main App
# ----------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LiDAR-8 Demo — Milestone 1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    walls = square_track(WIDTH, HEIGHT, MARGIN)

    pos = [WIDTH * 0.25, HEIGHT * 0.35]
    speed = 180.0

    running = True
    accum_dt = 0.0
    FIXED_DT = 1.0 / 60.0

    while running:
        dt = clock.tick(60) / 1000.0
        accum_dt += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        while accum_dt >= FIXED_DT:
            keys = pygame.key.get_pressed()
            dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP])
            mag = math.hypot(dx, dy)
            if mag > 0:
                dx /= mag
                dy /= mag
                pos[0] += dx * speed * FIXED_DT
                pos[1] += dy * speed * FIXED_DT

            pos[0] = float(np.clip(pos[0], MARGIN + 5, WIDTH - MARGIN - 5))
            pos[1] = float(np.clip(pos[1], MARGIN + 5, HEIGHT - MARGIN - 5))
            accum_dt -= FIXED_DT

        screen.fill(BG_COLOR)
        draw_walls(screen, walls)

        norm_dists = lidar8((pos[0], pos[1]), walls, R_MAX)
        draw_rays(screen, (pos[0], pos[1]), norm_dists)
        draw_car(screen, (pos[0], pos[1]))
        draw_readout(screen, font, norm_dists)

        hud = font.render("Arrow keys to move • Distances normalized to [0,1] • ESC to quit", True, TEXT_COLOR)
        screen.blit(hud, (MARGIN, MARGIN - 28))

        pygame.display.flip()

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()