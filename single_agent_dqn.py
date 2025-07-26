import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn_model import DQN
from utils import plot_rewards

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 200
rewards = []

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    with torch.no_grad():
        return torch.argmax(model(torch.FloatTensor(state))).item()

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        target = reward + gamma * torch.max(model(torch.FloatTensor(next_state))).item() * (1 - done)
        output = model(torch.FloatTensor(state))[action]
        loss = criterion(output, torch.tensor(target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
    rewards.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1}, Reward: {total_reward}")

plot_rewards(rewards, "Single Agent DQN", "results/reward_plot_single.png")
env.close()
