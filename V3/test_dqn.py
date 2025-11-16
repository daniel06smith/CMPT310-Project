import torch
from car_lidar_env import CarLidarEnv
from V3.dqn_agent import DQNAgent

env = CarLidarEnv(render_mode="human")

# Get observation/action size
obs, _ = env.reset()
obs_dim = len(obs)
action_dim = env.action_space.n

# Load agent
agent = DQNAgent(obs_dim, action_dim)
agent.q_net.load_state_dict(torch.load("dqn_car.pth"))
agent.q_net.eval()

done = False

while not done:
    with torch.no_grad():
        s = torch.FloatTensor(obs).unsqueeze(0)
        action = agent.q_net(s).argmax().item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    env.render()
