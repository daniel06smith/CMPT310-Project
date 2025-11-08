from lidar_env_laps import LidarLapEnv
env = LidarLapEnv()
obs, _ = env.reset()
print("Observation shape:", obs.shape)
print("Declared space:", env.observation_space.shape)
