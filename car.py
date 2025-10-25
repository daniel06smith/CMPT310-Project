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
        
    def update(self, dx: float, dy: float, dt: float):
        """Update car position and rotation based on input."""
        mag = math.hypot(dx, dy)
        if mag > 0:
            dx /= mag
            dy /= mag
            self.pos[0] += dx * self.speed * dt
            self.pos[1] += dy * self.speed * dt
            
            # Update rotation angle based on movement direction
            self.angle = math.degrees(math.atan2(dy, dx))
            
            # Store velocity
            self.velocity = [dx * self.speed, dy * self.speed]
    
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
