import numpy as np
import pandas as pd
from Dynamic_programming_MDP import Agent, Maze
from copy import deepcopy

class Agent_Onpolicy(Agent):
    def __init__(self, env, gamma = 1, alpha = 0.5, num_episodes = 100, epsilon = 0.1):
        super().__init__(env, gamma, alpha)
        self.q_matrix = {(state, action): 0 for state in self.states for action in self.env.actions}
        self.q_returns = {(state, action): [] for state in self.states for action in self.env.actions}
        self.num_episodes = num_episodes
        self.epsilon = epsilon

    def generate_episode(self):
        episode = []
        state = self.states[np.random.randint(0, len(self.states))]
        while True:
            if self.env.is_terminal(state):
                break
            actions = list(self.env.actions.keys())
            probs = [self.env.action_prob[state][a] for a in actions]
            action = np.random.choice(actions, p=probs)
            
            transitions = self.env.transition_prob[state][action]
            probs = [p for p, _ in transitions]
            next_states = [s for _, s in transitions]

            idx = np.random.choice(len(next_states), p=probs)
            next_state = next_states[idx]
            reward, done = self.env.reward_transition(next_state)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        return episode if episode != [] else self.generate_episode()
    
    def on_policy_control_epsilon_greedy(self):
        
        for _ in range(self.num_episodes):
            episode = self.generate_episode()
            visited = []
            returns = [0] * len(episode)
            returns[-1] = episode[- 1][2]
            for t in range(len(episode)-2,-1,-1):
                returns[t] = self.gamma * returns[t+1] + episode[t][2]

            for t in range(len(episode)):
                state, action, reward = episode[t]
                if (state, action) not in visited:
                    visited.append((state, action))
                    self.q_returns[(state, action)].append(returns[t])
                    self.q_matrix[(state, action)] = np.mean(self.q_returns[(state, action)])

                    best_action = max(self.env.actions,key=lambda a: self.q_matrix[(state, a)])
                    num_actions = len(self.env.actions)
                    for a in self.env.actions:
                        self.env.action_prob[state][a] = self.epsilon / num_actions

                    self.env.action_prob[state][best_action] += 1 - self.epsilon
                    
    def print_policy(self):

        arrows = {
            "left": "L",
            "right": "R",
            "up": "U",
            "down": "D"
        }

        for i in range(self.env.n):
            for j in range(self.env.m):
                state = (i, j)
                if self.env.maze[i, j] == -10:
                    print("X", end=" ")
                elif self.env.maze[i, j] == 10:
                    print("G", end=" ")
                else:
                    best_action = max(self.env.actions, key=lambda a: self.q_matrix[(state, a)])
                    print(arrows[best_action], end=" ")

            print()

class Agent_Offpolicy(Agent):
    def __init__(self, env, gamma = 1, alpha = 0.5, num_episodes = 100, epsilon = 0.1):
        super().__init__(env, gamma, alpha)
        self.num_episodes = num_episodes
        self.q_matrix = {(state, action): 0 for state in self.states for action in self.env.actions}
        self.C = {(state, action): 0 for state in self.states for action in self.env.actions}

    def generate_episode(self):
        episode = []
        state = self.states[np.random.randint(0, len(self.states))]
        while True:
            if self.env.is_terminal(state):
                break
            # behavior policy: uniform random
            actions = list(self.env.actions.keys())
            action = np.random.choice(actions)
            transitions = self.env.transition_prob[state][action]
            probs = [p for p, _ in transitions]
            next_states = [s for _, s in transitions]

            idx = np.random.choice(len(next_states), p=probs)
            next_state = next_states[idx]
            reward, done = self.env.reward_transition(next_state)
            episode.append((state, action, reward))
            state = next_state

            if done:
                break

        return episode if episode != [] else self.generate_episode()

    def off_policy_control(self):

        for _ in range(self.num_episodes):
            episode = self.generate_episode()
            G = 0
            W = 1
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                self.C[(state, action)] += W
                self.q_matrix[(state, action)] += (W / self.C[(state, action)]) * (G - self.q_matrix[(state, action)])

                # target policy: greedy
                best_action = max(self.env.actions, key=lambda a: self.q_matrix[(state, a)])
                if action != best_action:
                    break
                # behavior policy probability (uniform)
                W *= len(self.env.actions)

    def print_policy(self):

        arrows = {
            "left": "L",
            "right": "R",
            "up": "U",
            "down": "D"
        }

        for i in range(self.env.n):
            for j in range(self.env.m):
                state = (i, j)
                if self.env.maze[i, j] == -10:
                    print("X", end=" ")

                elif self.env.maze[i, j] == 10:
                    print("G", end=" ")

                else:
                    best_action = max(self.env.actions, key=lambda a: self.q_matrix[(state, a)])
                    print(arrows[best_action], end=" ")
            print()

def main():
    env = np.array([[-1, -1, -1], [-10, -10, -1], [-1, -10, 10]])
    env = Maze(3, 3, env)
    env2 = deepcopy(env)
    agent = Agent_Onpolicy(env, 1, 0.5, 100, 0.1)
    agent.on_policy_control_epsilon_greedy()

    agent.print_policy()

    agent = Agent_Offpolicy(env, 1, 0.5, 100, 0.1)
    agent.off_policy_control()

    agent.print_policy()

if __name__ == "__main__":
    main()
            