import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
#from tqdm import trange

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# Replay buffer seen in class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


class DQN(nn.Module):
    def __init__(self, state_dim, n_action):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_action)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = ReplayBuffer(buffer_size)
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.target_net.eval()
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                observation = torch.Tensor(observation).unsqueeze(0).to(device)
                q = self.model(observation)
                return torch.argmax(q).item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self):
        self.model = DQN(self.state_dim, self.action_dim)
        self.path = 'src/hiv_agent.pt'
        state_dict = torch.load(self.path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        actions = actions.to(torch.int64)  
        q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = ProjectAgent()

    n_episodes = 200
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10
    best_score = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        cum_reward = 0
        done = False
        trunc = False

        while not done and not trunc:
            action = agent.act(state, use_random=(random.random() < epsilon))
            next_state, reward, done, trunc, _ = env.step(action)
            agent.memory.append(state, action, reward, next_state, done)
            state = next_state

            cum_reward += reward
            agent.train_step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            agent.update_target()
        
        score = evaluate_HIV(agent, nb_episode=1)
        if score > best_score:
            best_score = score
            agent.save("hiv_agent.pt")

        print(f"Episode {episode + 1}, Reward: {cum_reward:.2e}, Score: {score:.2e}")