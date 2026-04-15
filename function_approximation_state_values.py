import pandas as pd
import numpy as np
from copy import deepcopy

class LinearAgent:
    def __init__(self, states, feature_dim):
        self.alpha = 0.5
        self.gamma = 1
        self.state_feature_mapping = states
        self.n = feature_dim

        self.W = np.zeros((self.n,))

    def get_feature_state(self, state):
        return self.state_feature_mapping[state]
    
    def V_value(self, state, W):
        return np.dot(W, self.get_feature_state(state))
    
    def update(self, delta, state):
        self.W += self.alpha * delta * self.get_feature_state(state)

    def train_agent_TD(self, episodes, step = 1):
        print("Initial_weights : ", self.W) 
        for episode in episodes:
            for t in range(len(episode)+ step):
                # state, reward = episode[t]
                
                tau = t - step
                if tau >= 0:
                    G = 0
                    for t_ in range(tau, min(tau + step, len(episode))):
                        G += self.gamma ** (t_ - tau) * episode[t_][1]
                    if tau + step < len(episode):
                        sn = episode[tau + step][0]
                        G += self.gamma ** step * self.V_value(sn, self.W)
                    s = episode[tau][0]
                    delta = G - self.V_value(s, self.W)
                    self.update(delta, s)

        print("Final_weights : ", self.W) 

    def train_agent_MC(self, episodes):
        print("Initial_weights : ", self.W) 
        for episode in episodes:
            n = len(episode)
            returns = [0] * n
            returns[-1] = episode[-1][1]
            W_old = deepcopy(self.W)
            for t in range(n-2,-1, -1):
                returns[t] = self.gamma * returns[t+1] + episode[t][1]

            for t in range(n):
                s = episode[t][0]
                delta = returns[t] - self.V_value(s, W_old)
                self.update(delta, s)

        print("Final_weights : ", self.W) 

import torch
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class NonLinearAgent:
    def __init__(self, states, feature_dim):
        self.alpha = 0.01
        self.gamma = 1
        self.state_feature_mapping = states

        self.model = ValueNetwork(feature_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def get_feature_state(self, state):
        return torch.tensor(self.state_feature_mapping[state], dtype=torch.float32)

    def V_value(self, state):
        x = self.get_feature_state(state)
        return self.model(x).squeeze()
    
    def print_weights(self):
        print("\n--- Model Weights ---")
        for name, param in self.model.named_parameters():
            print(f"{name} | shape={param.shape}")
            print(param.data)
            print("-" * 30)

    def train_agent_TD(self, episodes, step=1):
        print("Training (NN TD)...")

        print("\nInitial Weights:")
        self.print_weights()

        for episode in episodes:
            T = len(episode)

            for t in range(T + step):

                tau = t - step

                if tau >= 0:
                    G = 0

                    # reward accumulation
                    for t_ in range(tau, min(tau + step, T)):
                        G += (self.gamma ** (t_ - tau)) * episode[t_][1]

                    # bootstrapping
                    if tau + step < T:
                        sn = episode[tau + step][0]
                        G += (self.gamma ** step) * self.V_value(sn).item()

                    s = episode[tau][0]

                    # prediction
                    pred = self.V_value(s)

                    # target
                    target = torch.tensor(G, dtype=torch.float32)

                    # loss
                    loss = self.loss_fn(pred, target)

                    # backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        print("\nFinal Weights:")
        self.print_weights()


episodes = [
    [("s1", 1), ("s3", 2), ("s7", 1)],
    [("s2", 0), ("s5", 3), ("s7", 9)]
]

features = {
            "s1": np.array([2,0,0,0,0,0,0,1]),
            "s2": np.array([1,1,0,0,0,0,0,1]),
            "s3": np.array([0,2,0,0,0,0,0,1]),
            "s4": np.array([0,0,2,0,0,0,0,1]),
            "s5": np.array([0,0,0,2,0,0,0,1]),
            "s6": np.array([0,0,0,0,2,0,0,1]),
            "s7": np.array([0,0,0,0,0,0,1,2]),
        }
agent = LinearAgent(states=features, feature_dim=8)
# agent.train_agent_TD(episodes=episodes, step = 3)
# agent.train_agent_MC(episodes=episodes)
            
agent = NonLinearAgent(states=features, feature_dim=8)
agent.train_agent_TD(episodes, step=3)

            

            

