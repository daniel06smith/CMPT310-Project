"""
LiDAR-8 Demo — Milestone 1
- Arrow keys move the car
- 8 compass rays (N, NE, E, SE, S, SW, W, NW)
- Distances normalized to [0, 1] and displayed on screen
"""

import sys
import pygame

from build_track import square_track
from car import Car
from sensors import lidar8
from rendering import (
    BG_COLOR, 
    draw_walls, 
    draw_car, 
    draw_rays, 
    draw_readout, 
    draw_hud
)

# ----------------------------
# Window & world parameters
# ----------------------------
WIDTH, HEIGHT = 900, 600
MARGIN = 40

# Car dimensions (for hitbox)
CAR_WIDTH = 40
CAR_HEIGHT = 24

# LiDAR parameters
R_MAX = 100.0  # max sensing range in pixels


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LiDAR-8 Demo — Milestone 1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    walls = square_track(WIDTH, HEIGHT, MARGIN)

    # Load car image (optional - set to None to use rectangle fallback)
    car_image = None
    try:
        # Try to load car.png from the same directory
        # Replace 'car.png' with your actual image filename
        original_car_image = pygame.image.load('car.png').convert_alpha()
        # Scale the image to match hitbox dimensions
        car_image = pygame.transform.scale(original_car_image, (CAR_WIDTH, CAR_HEIGHT))
    except FileNotFoundError:
        print("car.png not found - using rectangle fallback")
    except Exception as e:
        print(f"Error loading car image: {e} - using rectangle fallback")

    # Create car
    car = Car(WIDTH * 0.25, HEIGHT * 0.35, CAR_WIDTH, CAR_HEIGHT)

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
            
            # Update car
            car.update(dx, dy, FIXED_DT)
            car.constrain_to_bounds(MARGIN + 5, WIDTH - MARGIN - 5, 
                                   MARGIN + 5, HEIGHT - MARGIN - 5)
            
            accum_dt -= FIXED_DT

        screen.fill(BG_COLOR)
        draw_walls(screen, walls)

        # Check for collision
        is_colliding = car.check_collision(walls)

        # Get LiDAR data
        norm_dists = lidar8((car.pos[0], car.pos[1]), walls, R_MAX)
        
        # Draw everything
        draw_rays(screen, (car.pos[0], car.pos[1]), norm_dists, R_MAX)
        draw_car(screen, (car.pos[0], car.pos[1]), car.width, car.height, 
                car_image, car.angle, is_colliding)
        draw_readout(screen, font, norm_dists, WIDTH, HEIGHT, MARGIN)
        draw_hud(screen, font, MARGIN)

        pygame.display.flip()

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
