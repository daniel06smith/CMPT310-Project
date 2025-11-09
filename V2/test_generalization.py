from stable_baselines3 import PPO
from car_lidar_env import CarLidarEnv
import time

# Train on track 1
print("Training on track 1...")
env_train = CarLidarEnv(render_mode=None, track_num=1)
model = PPO("MlpPolicy", env_train, verbose=0)
model.learn(total_timesteps=50_000)  # shorter for demo
model.save("ppo_car_track1")
env_train.close()

# Test on track 2 with trained model
print("Testing trained model on track 2...")
env_test = CarLidarEnv(render_mode="human", track_num=2)
model_trained = PPO.load("ppo_car_track1")

num_episodes = 5
for episode in range(num_episodes):
    obs, _ = env_test.reset()
    done = False
    episode_reward = 0
    while not done:
        env_test.set_hud(episode + 1, episode_reward)
        action, _ = model_trained.predict(obs)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        episode_reward += reward
        env_test.render()
        done = terminated or truncated
        time.sleep(0.02)
    print(f"Trained on Track2 - Episode {episode + 1}: Reward = {episode_reward:.2f}")

env_test.close()

# Test untrained on track 2
print("Testing untrained model on track 2...")
env_untrained = CarLidarEnv(render_mode="human", track_num=2)
model_untrained = PPO("MlpPolicy", env_untrained, verbose=0)  # random actions

for episode in range(num_episodes):
    obs, _ = env_untrained.reset()
    done = False
    episode_reward = 0
    while not done:
        env_untrained.set_hud(episode + 1, episode_reward)
        action, _ = model_untrained.predict(obs)
        obs, reward, terminated, truncated, _ = env_untrained.step(action)
        episode_reward += reward
        env_untrained.render()
        done = terminated or truncated
        time.sleep(0.02)
    print(f"Untrained on Track2 - Episode {episode + 1}: Reward = {episode_reward:.2f}")

env_untrained.close()