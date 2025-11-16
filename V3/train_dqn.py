import torch
from car_lidar_env import CarLidarEnv
from dqn_agent import DQNAgent, ReplayBuffer

env = CarLidarEnv(render_mode="human", track_num=3)

obs, _ = env.reset()
obs_dim = len(obs)
action_dim = env.action_space.n

agent = DQNAgent(obs_dim, action_dim)
buffer = ReplayBuffer()

episodes = 1000
target_update_freq = 500
global_step = 0

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done)

        loss = agent.train_step(buffer)

        if global_step % target_update_freq == 0:
            agent.update_target()

        agent.update_epsilon()

        obs = next_obs
        ep_reward += reward
        global_step += 1

    print(f"Episode {ep} | Reward: {ep_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

# Save model
torch.save(agent.q_net.state_dict(), "dqn_car.pth")
print("Saved model.")
