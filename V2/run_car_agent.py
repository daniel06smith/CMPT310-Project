import csv
import os
import time
import pygame
from stable_baselines3 import PPO
from car_lidar_env import CarLidarEnv

MODEL_PATH = "ppo_car_lidar"
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "run_log.csv")

# Load trained model
model = PPO.load(MODEL_PATH)

# Use human render to see it
env = CarLidarEnv(render_mode="human", track_num=5)

file_exists = os.path.isfile(LOG_PATH)

with open(LOG_PATH, mode="a", newline="") as log_file:
    log_writer = csv.writer(log_file)
    if not file_exists:
        log_writer.writerow(['episode', 'reward'])

    num_episodes = 5
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        env.hud_episode = ep

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    num_episodes = ep

            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            env.hud_reward = ep_reward

            env.render()
            done = terminated or truncated

        log_writer.writerow([ep, ep_reward])
        log_file.flush()
        print(f"Episode {ep} reward: {ep_reward}")

    # log at end of each episode
    log_writer.writerow([ep, ep_reward])
    log_file.flush()
    print(f"Episode {ep} reward: {ep_reward}")

log_file.close()
env.close()
