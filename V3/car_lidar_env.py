import os
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class CarLidarEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None, track_num = 1):
        super().__init__()
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.render_mode = render_mode
        self.track_num = track_num

        self.checkpoint_colors = [
        # colors used with eye drop tool
        # (255, 0, 255),     # CP0: magenta
        # (0, 255, 255),     # CP1: cyan
        # (255, 255, 0),     # CP2: yellow
        # (255, 128, 0),     # CP3: orange
        # (0, 255, 0),       # CP4: green

        # real colors
        (234, 51, 247),     # CP0: magenta
        (117, 251, 253),     # CP1: cyan
        (255, 255, 84),     # CP2: yellow
        (240, 156, 73),     # CP3: orange
        (117, 251, 76),       # CP4: green
        ]
        self.current_checkpoint = 0

        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()

        # Load track and car
        self.track = pygame.image.load(f"./tracks/track{self.track_num}.png").convert()
        self.track = pygame.transform.scale(self.track, (self.WIDTH, self.HEIGHT))
        self.car_image = pygame.image.load("car.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (35, 30))
        self.car_w, self.car_h = self.car_image.get_size()

        # Define action and observation spaces
        # Actions: [steer_left, steer_right, accelerate]
        self.action_space = spaces.Discrete(3)

        # Observation: 5 LIDAR distances (normalized 0–1)
        self.num_lidars = 5
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_lidars,), dtype=np.float32)

        # Car parameters
        self.acceleration = 0.25
        self.friction = 0.05
        self.turn_speed = 4
        self.max_speed = 8
        self.max_lidar = 250

        self.reset()

    # -----------------------------------
    # Utility functions
    # -----------------------------------

    def get_rotated_hitbox(self, cx, cy, w, h, angle_deg):
        rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = w / 2, h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]

    def check_collision(self, corners):
        for (cx, cy) in corners:
            if 0 <= cx < self.WIDTH and 0 <= cy < self.HEIGHT:
                color = self.track.get_at((int(cx), int(cy)))[:3]
                if color <= (100, 100, 100):
                    return True
        return False

    def cast_lidar(self, cx, cy, angle_deg):
        rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for dist in range(0, self.max_lidar, 2):
            lx = int(cx + cos_a * dist)
            ly = int(cy + sin_a * dist)
            if not (0 <= lx < self.WIDTH and 0 <= ly < self.HEIGHT):
                return self.max_lidar
            color = self.track.get_at((lx, ly))[:3]
            if color <= (100, 100, 100):
                return dist
        return self.max_lidar

    def get_lidar_readings(self):
        angles = [-60, -30, 0, 30, 60]
        readings = [self.cast_lidar(self.x, self.y, self.angle + a) / self.max_lidar for a in angles]
        return np.array(readings, dtype=np.float32)
    
    def check_checkpoint_pixel(self):
        # Get pixel under the car
        cx, cy = int(self.x), int(self.y)
        color = self.track.get_at((cx, cy))[:3]

        expected_color = self.checkpoint_colors[self.current_checkpoint]
        # print("Pixel color at car:", color, "expected:", expected_color)

        # If car touches correct checkpoint color → progress!
        if self.color_close(color, expected_color):
            self.current_checkpoint += 1

            print(f"Hit checkpoint {self.current_checkpoint} at ({cx}, {cy})")

            # Completed all checkpoints? → Lap!
            if self.current_checkpoint >= len(self.checkpoint_colors):
                print("Completed a LAP!") 
                self.current_checkpoint = 0
                return "lap"

            return "checkpoint"

        return None
    
    def color_close(self, c1, c2, tol=40):
        return (
            abs(c1[0] - c2[0]) <= tol and
            abs(c1[1] - c2[1]) <= tol and
            abs(c1[2] - c2[2]) <= tol
        )



    # -----------------------------------
    # Core Gym methods
    # -----------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = 400, 500
        self.angle = 0
        self.velocity_x, self.velocity_y = 0, 0
        self.crashed = False
        obs = self.get_lidar_readings()
        return obs, {}

    def step(self, action):
        # Interpret action
        if action == 0:   # steer left
            self.angle += self.turn_speed
        elif action == 1:  # steer right
            self.angle -= self.turn_speed
        elif action == 2:  # accelerate
            self.velocity_x += math.cos(math.radians(self.angle)) * self.acceleration
            self.velocity_y -= math.sin(math.radians(self.angle)) * self.acceleration
        
        # Speed limiting + friction
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
        self.velocity_x *= (1 - self.friction)
        self.velocity_y *= (1 - self.friction)

        # Predict next position
        next_x = self.x + self.velocity_x
        next_y = self.y + self.velocity_y
        corners = self.get_rotated_hitbox(next_x, next_y, self.car_w, self.car_h, self.angle)

        reward = 0.02  # small positive reward for surviving
        reward += 0.03 * speed # for speed

        # Collision check
        if self.check_collision(corners):
            reward = -10.0
            terminated = True
            self.velocity_x = self.velocity_y = 0
        else:
            self.x, self.y = next_x, next_y
            terminated = False

        obs = self.get_lidar_readings()
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()
        
        # Checkpoints rewards
        result = self.check_checkpoint_pixel()

        if result == "checkpoint":
            reward += 5.0      # strong reward for progress
        elif result == "lap":
            reward += 20.0     # huge reward for completing the track


        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.blit(self.track, (0, 0))
        # Draw lidar
        angles = [-60, -30, 0, 30, 60]
        for a, dist in zip(angles, self.get_lidar_readings() * self.max_lidar):
            rad = math.radians(-self.angle - a)
            end_x = self.x + math.cos(rad) * dist
            end_y = self.y + math.sin(rad) * dist
            pygame.draw.line(self.screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 2)
        # Draw car
        rotated_car = pygame.transform.rotate(self.car_image, self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        self.screen.blit(rotated_car, rect.topleft)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


    def close(self):
        pygame.quit()
