"""
Car class with physics, collision detection, and state management.
"""
import math
from typing import List, Tuple
import pygame

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]


class Car:
    """Represents a car with position, rotation, and collision detection."""
    
    def __init__(self, x: float, y: float, width: int = 40, height: int = 24):
        self.pos = [x, y]
        self.width = width
        self.height = height
        self.angle = 0.0  # Rotation angle in degrees
        self.velocity = [0.0, 0.0]
        self.speed = 180.0
        self.max_speed = 180.0
        self.acceleration = 300.0
        self.brake = 400.0
        self.friction = 2.0
        self.heading_r = 0.0 # radians

    def update(self, steer: float, throttle: float, brake: float, dt: float):
        # steer in [-1, 1], throttle in [-1, 1], brake in [0, 1]

        self.heading_r += steer * dt * 2.0  # Adjust heading based on steering input

        fx = math.cos(self.heading_r)
        fy = math.sin(self.heading_r)

        vx, vy = self.velocity
        v_long = fx * vx + fy * vy

        a = self.acceleration * throttle - self.brake * brake * math.copysign(1.0, v_long)

        v_long = v_long + a * dt
        v_long *= max(0.0, 1.0 - self.friction * dt)

        v_long = max(-0.25 * self.max_speed, min(self.max_speed, v_long))

        self.velocity = [fx * v_long, fy * v_long]

        self.pos[0] += self.velocity[0] * dt
        self.pos[1] += self.velocity[1] * dt
    
    def constrain_to_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float):
        """Keep car within bounds."""
        self.pos[0] = max(min_x, min(max_x, self.pos[0]))
        self.pos[1] = max(min_y, min(max_y, self.pos[1]))
    
    def get_hitbox(self) -> pygame.Rect:
        """Get the rectangular hitbox for collision detection."""
        return pygame.Rect(
            int(self.pos[0] - self.width/2), 
            int(self.pos[1] - self.height/2), 
            self.width, 
            self.height
        )
    
    def check_collision(self, walls: List[Segment]) -> bool:
        """Check if the car's hitbox collides with any wall."""
        car_rect = self.get_hitbox()
        
        # Check each wall segment for collision with the car rectangle
        for (a, b) in walls:
            # Check if line segment intersects with rectangle
            if car_rect.clipline(a, b):
                return True
        return False
