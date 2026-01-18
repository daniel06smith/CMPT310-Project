import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from build_track import square_track
from car import Car
from sensors import lidar8
from rendering import BG_COLOR, draw_walls, draw_car, draw_rays, draw_hud

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
WIDTH, HEIGHT = 900, 600
MARGIN = 40
CAR_WIDTH, CAR_HEIGHT = 40, 24
R_MAX = 100.0  # LiDAR max range (pixels)

# Circular checkpoints (12 around track)
def generate_checkpoints(margin=60, width=WIDTH, height=HEIGHT, num_per_side=3):
    """Generate checkpoints along a rectangular loop (like the track walls)."""
    pts = []
    left = margin
    right = width - margin
    top = margin
    bottom = height - margin

    # top edge (left → right)
    for i in range(num_per_side):
        t = i / num_per_side
        pts.append((left + t * (right - left), top))

    # right edge (top → bottom)
    for i in range(1, num_per_side + 1):
        t = i / num_per_side
        pts.append((right, top + t * (bottom - top)))

    # bottom edge (right → left)
    for i in range(1, num_per_side + 1):
        t = i / num_per_side
        pts.append((right - t * (right - left), bottom))

    # left edge (bottom → top)
    for i in range(1, num_per_side + 1):
        t = i / num_per_side
        pts.append((left, bottom - t * (bottom - top)))

    return pts


class LidarLapEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ---------------- RL API ----------------
        # [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation = 8 LiDAR + speed + heading_sin + heading_cos + progress_dir(2) = 12 floats
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # ---------------- Simulation ----------------
        self.walls = square_track(WIDTH, HEIGHT, MARGIN)
        self.car = Car(WIDTH * 0.25, HEIGHT * 0.35, CAR_WIDTH, CAR_HEIGHT)
        self.checkpoints = generate_checkpoints(margin=MARGIN+30, num_per_side=3)
        self.num_checkpoints = len(self.checkpoints)

        self.current_cp = 0
        self.laps_completed = 0

        self.font = None
        self.clock = None
        self.screen = None
        self.car_image = None

        self.steps = 0
        self.max_steps = 4000

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("LiDAR-Lap RL Environment")
            self.font = pygame.font.SysFont("consolas", 14)
            self.clock = pygame.time.Clock()

    # ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start_cp = np.array(self.checkpoints[0])
        next_cp = np.array(self.checkpoints[1])

        self.car = Car(start_cp[0], start_cp[1], CAR_WIDTH, CAR_HEIGHT)
        self.car.heading_r = math.atan2(next_cp[1] - start_cp[1], next_cp[0] - start_cp[0])
        self.current_cp = 1
        self.laps_completed = 0
        self.steps = 0
        self.prev_dist = np.linalg.norm(next_cp - start_cp)
        obs = self._get_obs()
        return obs, {}


    # ---------------------------------------------------------
    def step(self, action):
        steer, throttle = np.clip(action, [-1, 0], [1, 1])
        self.car.update(steer=steer, throttle=throttle, brake=0.0, dt=1/60)
        self.car.constrain_to_bounds(
            MARGIN + 5, WIDTH - MARGIN - 5, MARGIN + 5, HEIGHT - MARGIN - 5
        )

        lidar = lidar8((self.car.pos[0], self.car.pos[1]), self.walls, R_MAX)
        lidar = np.clip(np.nan_to_num(lidar, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        reward, done = self._compute_reward(lidar)
        obs = self._get_obs(lidar)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        if self.render_mode == "human":
            self.render(lidar)

        return obs, reward, done, False, {}

    # ---------------------------------------------------------
    def _compute_reward(self, lidar):
        # Distance to next checkpoint
        next_cp = np.array(self.checkpoints[self.current_cp])
        dist_to_cp = np.linalg.norm(next_cp - self.car.pos)
        progress = getattr(self, "prev_dist", dist_to_cp) - dist_to_cp
        self.prev_dist = dist_to_cp

        # --- Core reward ---
        reward = 0.2 * self.car.speed
        reward += 0.1 * progress                     # reward getting closer to checkpoint
        reward += 0.05 * np.mean(lidar)              # wall distance
        reward -= 0.01                               # time penalty

        # --- Checkpoint reached ---
        if dist_to_cp < 30:
            reward += 10
            self.current_cp = (self.current_cp + 1) % self.num_checkpoints
            self.prev_dist = np.linalg.norm(
                np.array(self.checkpoints[self.current_cp]) - self.car.pos
            )
            if self.current_cp == 0:
                self.laps_completed += 1
                reward += 50  # bonus for completing lap

        # --- Collision penalty ---
        if self.car.check_collision(self.walls):
            reward -= 25
            done = True
        else:
            done = False

        return float(np.clip(reward, -25, 25)), done

    # ---------------------------------------------------------
    def _get_obs(self, lidar=None):
        if lidar is None:
            lidar = lidar8((self.car.pos[0], self.car.pos[1]), self.walls, R_MAX)
        lidar = np.clip(np.nan_to_num(lidar, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        v_norm = self.car.speed / (getattr(self.car, "max_speed", 5.0) + 1e-6)
        heading_sin, heading_cos = np.sin(self.car.heading_r), np.cos(self.car.heading_r)

        next_cp = np.array(self.checkpoints[self.current_cp])
        vec_to_cp = next_cp - self.car.pos
        dist_norm = np.clip(vec_to_cp / (R_MAX * 2), -1, 1)
        dx, dy = (dist_norm[0] + 1)/2, (dist_norm[1] + 1)/2  # normalize to [0,1]

        obs = np.array(list(lidar) + [v_norm, heading_sin, heading_cos, dx, dy], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(obs, 0.0, 1.0)

    # ---------------------------------------------------------
    def render(self, lidar=None):
        if self.screen is None:
            return
        self.screen.fill(BG_COLOR)
        draw_walls(self.screen, self.walls)

        # Draw checkpoints
        for i, cp in enumerate(self.checkpoints):
            color = (0, 255, 0) if i == self.current_cp else (60, 60, 60)
            pygame.draw.circle(self.screen, color, (int(cp[0]), int(cp[1])), 6)

        if lidar is None:
            lidar = lidar8((self.car.pos[0], self.car.pos[1]), self.walls, R_MAX)

        draw_rays(self.screen, (self.car.pos[0], self.car.pos[1]), lidar, R_MAX)
        draw_car(
            self.screen,
            (self.car.pos[0], self.car.pos[1]),
            self.car.width,
            self.car.height,
            self.car_image,
            math.degrees(self.car.heading_r),
            self.car.check_collision(self.walls),
        )
        draw_hud(self.screen, self.font, 20)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    # ---------------------------------------------------------
    def close(self):
        if self.render_mode == "human":
            pygame.quit()
