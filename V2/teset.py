from car_lidar_env import CarLidarEnv

# Headless mode for training
env = CarLidarEnv(render_mode=None, track_num=1)
obs, _ = env.reset()
print("Observation shape:", obs.shape)
env.close()

# Visual mode for debugging
env = CarLidarEnv(render_mode="human")
obs, _ = env.reset()
for _ in range(100):
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    env.render()
env.close()
