import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn_model import DQN
from utils import plot_rewards

agents = 2
envs = [gym.make('CartPole-v1') for _ in range(agents)]
state_dim = envs[0].observation_space.shape[0]
action_dim = envs[0].action_space.n

models = [DQN(state_dim, action_dim) for _ in range(agents)]
optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in models]
criterion = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 200
rewards = [[] for _ in range(agents)]

def choose_action(model, state, eps):
    if np.random.rand() < eps:
        return np.random.randint(action_dim)
    with torch.no_grad():
        return torch.argmax(model(torch.FloatTensor(state))).item()

for episode in range(episodes):
    states = [env.reset() for env in envs]
    total_rewards = [0 for _ in range(agents)]
    done_flags = [False for _ in range(agents)]

    while not all(done_flags):
        for i in range(agents):
            if done_flags[i]:
                continue
            action = choose_action(models[i], states[i], epsilon)
            next_state, reward, done, _ = envs[i].step(action)
            total_rewards[i] += reward

            target = reward + gamma * torch.max(models[i](torch.FloatTensor(next_state))).item() * (1 - done)
            output = models[i](torch.FloatTensor(states[i]))[action]
            loss = criterion(output, torch.tensor(target))

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

            states[i] = next_state
            done_flags[i] = done

    for i in range(agents):
        rewards[i].append(total_rewards[i])
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1}, Rewards: {total_rewards}")

for i in range(agents):
    plot_rewards(rewards[i], f"Agent {i+1} Multi-Agent DQN", f"results/reward_plot_multi_agent_{i+1}.png")
for env in envs:
    env.close()
