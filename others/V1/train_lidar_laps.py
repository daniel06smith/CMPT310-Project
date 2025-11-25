from stable_baselines3 import PPO
from lidar_env_laps import LidarLapEnv

env = LidarLapEnv()  # headless training

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    clip_range=0.2,
    max_grad_norm=0.5,
)

print("🚀 Training PPO for lap navigation...")
model.learn(total_timesteps=1_000_000)
model.save("ppo_lidar8_laps")
env.close()
print("✅ Done! Run watch_lidar_laps.py to visualize.")
