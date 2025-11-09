from stable_baselines3 import PPO
from car_lidar_env import CarLidarEnv
import time

# Load trained model
model = PPO.load("ppo_car_lidar")

# Use human render to see it
env = CarLidarEnv(render_mode="human", track_num=4)

num_episodes = 10  # Number of episodes to run
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done:
        env.set_hud(episode + 1, episode_reward)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        steps += 1
        env.render()
        done = terminated or truncated
        time.sleep(0.02)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

env.close()
