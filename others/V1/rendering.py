"""
Rendering functions for the game (walls, car, LiDAR rays, UI).
"""
from typing import List, Tuple, Optional
import numpy as np
import pygame

from sensors import DIRS_8

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]

# Colors
BG_COLOR = (18, 18, 18)
WALL_COLOR = (230, 230, 230)
RAY_COLOR = (120, 200, 255)
CAR_COLOR = (255, 90, 90)
TEXT_COLOR = (230, 230, 230)


def draw_walls(screen: pygame.Surface, walls: List[Segment]):
    """Draw all wall segments."""
    for (a, b) in walls:
        pygame.draw.line(screen, WALL_COLOR, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 2)


def draw_car(screen: pygame.Surface, pos: Vec2, width: int, height: int, 
             car_image: Optional[pygame.Surface] = None, angle: float = 0, 
             is_colliding: bool = False):
    """
    Draw the car using an image if provided, otherwise draw a rectangle.
    
    Args:
        screen: Pygame surface to draw on
        pos: Center position of the car (x, y)
        width: Car width
        height: Car height
        car_image: Optional car sprite image
        angle: Rotation angle in degrees (0 = facing right)
        is_colliding: If True, inverts the colors
    """
    if car_image is not None:
        rotated_image = pygame.transform.rotate(car_image, -angle)
        
        # Invert colors if colliding
        if is_colliding:
            # Create inverted version of the image
            inverted = rotated_image.copy()
            pixels = pygame.surfarray.array3d(inverted)
            # Invert RGB values
            pixels[:] = 255 - pixels
            inverted = pygame.surfarray.make_surface(pixels)
            # Preserve alpha channel
            inverted.set_colorkey(rotated_image.get_colorkey())
            if rotated_image.get_alpha() is not None:
                inverted.set_alpha(rotated_image.get_alpha())
            rotated_image = inverted
        
        rect = rotated_image.get_rect(center=(int(pos[0]), int(pos[1])))
        screen.blit(rotated_image, rect)
        
    else:
        rect = pygame.Rect(int(pos[0] - width/2), int(pos[1] - height/2), width, height)
        # Invert color if colliding
        color = CAR_COLOR
        if is_colliding:
            color = (255 - CAR_COLOR[0], 255 - CAR_COLOR[1], 255 - CAR_COLOR[2])
        pygame.draw.rect(screen, color, rect, border_radius=4)


def draw_rays(screen: pygame.Surface, p: Vec2, norm_dists: np.ndarray, r_max: float):
    """Draw LiDAR rays from position p."""
    px, py = int(p[0]), int(p[1])
    for i, nd in enumerate(norm_dists):
        if not np.isfinite(nd):
            continue
        d = DIRS_8[i]
        dist = float(nd) * r_max
        end_x = int(round(p[0] + d[0] * dist))
        end_y = int(round(p[1] + d[1] * dist))
        pygame.draw.line(screen, RAY_COLOR, (px, py), (end_x, end_y), 1)
        pygame.draw.circle(screen, RAY_COLOR, (end_x, end_y), 2)


def draw_readout(screen: pygame.Surface, font: pygame.font.Font, norm_dists: np.ndarray, 
                 width: int, height: int, margin: int):
    """Draw LiDAR distance readout at bottom of screen."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    lines = [f"{labels[i]}: {norm_dists[i]:.2f}" for i in range(8)]
    text = " | ".join(lines)
    surf = font.render(text, True, TEXT_COLOR)
    # Center the text horizontally
    text_x = (width - surf.get_width()) // 2
    screen.blit(surf, (text_x, height - margin + 8 - 24))


def draw_hud(screen: pygame.Surface, font: pygame.font.Font, margin: int):
    """Draw HUD text at top of screen."""
    hud = font.render("Arrow keys to move • Distances normalized to [0,1] • ESC to quit", True, TEXT_COLOR)
    screen.blit(hud, (margin, margin - 28))
