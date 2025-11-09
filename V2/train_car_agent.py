from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from car_lidar_env import CarLidarEnv
from stable_baselines3.common.callbacks import BaseCallback
import os

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log rewards and other metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_rew = self.model.ep_info_buffer[-1]['r']
            ep_len = self.model.ep_info_buffer[-1]['l']
            self.logger.record('episode/reward', ep_rew)
            self.logger.record('episode/length', ep_len)
        return True

# Create env (no render for faster training)
env = CarLidarEnv(render_mode=None, track_num=4)

# Check compatibility
check_env(env, warn=True)

# Define PPO model with tensorboard
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=1024,
    gamma=0.99,
    tensorboard_log="./tb_logs/"
)

# Train agent
model.learn(total_timesteps=200_000, callback=TensorboardCallback())

# Save model
model.save("ppo_car_lidar")

env.close()
