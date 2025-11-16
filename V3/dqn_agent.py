import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Q-Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, size=50_000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim

        self.q_net = DQN(obs_dim, action_dim)
        self.target_net = DQN(obs_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    # Action selection
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q = self.q_net(s)
            return q.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # One training step
    def train_step(self, buffer, batch_size=64):
        if len(buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        s = torch.FloatTensor(states)
        a = torch.LongTensor(actions)
        r = torch.FloatTensor(rewards)
        s2 = torch.FloatTensor(next_states)
        d = torch.FloatTensor(dones)

        q_vals = self.q_net(s)[range(batch_size), a]

        with torch.no_grad():
            next_q = self.target_net(s2).max(1)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = ((q_vals - target)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
