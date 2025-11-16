from car_lidar_env import CarLidarEnv
import time

# Create environment
env = CarLidarEnv(render_mode="human", track_num=2) 

print("\n--- Gymnasium Environment Debug Demo ---\n")

# Reset the environment
obs, info = env.reset()

print("Initial Observation (LIDAR distances):")
print(obs)
print("Observation shape:", len(obs))

print("\nAction Space:")
print(env.action_space)
print("Number of actions:", env.action_space.n)

print("\nTaking 10 random actions...\n")

for step in range(100):
    action = env.action_space.sample()   # random action
    next_obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}")
    print("  Action:", action)
    print("  Observation:", next_obs)
    print("  Reward:", reward)
    print("  Terminated:", terminated)
    print("  Truncated:", truncated)
    print("----------------------------------")

    env.render()

env.close()

print("\nDemo complete.\n")
