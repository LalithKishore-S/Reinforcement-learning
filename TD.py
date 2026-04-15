import numpy as np
import random
from copy import deepcopy

class Maze:
    def __init__(self, n, m, maze=None):
        self.n = n
        self.m = m
        
        if maze is not None:
            self.maze = maze
        else:
            self.maze = self.generate_maze_random()

        self.actions = {"left" : (0, -1), "right" : (0, 1), "up" : (-1, 0), "down" : (1, 0)}
        self.action_prob = {(i, j):{key : 1/(len(self.actions.keys())) for key in self.actions.keys()} for i in range(n) for j in range(m)}
        self.transition_prob = self.generate_transition_prob_matrix()

    def generate_maze_random(self):
        maze = np.zeros((self.n, self.m))
        num_snakes = (self.n * self.m) // 6
        for _ in range(num_snakes):
            r = random.randint(0, self.n-1)
            c = random.randint(0, self.m-1)
            maze[r][c] = -10   
        goal_r = random.randint(0, self.n-1)
        goal_c = random.randint(0, self.m-1)

        maze[goal_r][goal_c] = 10

        for i in range(self.n):
            for j in range(self.m):
                if maze[i,j] not in [-10, 10]:
                    maze[i, j] = -1

        return maze
    
    def generate_transition_prob_matrix(self):
        t_matrix = {}
        for i in range(self.n):
            for j in range(self.m):
                state = (i, j)
                t_matrix[state] = {}
                for action, (dx, dy) in self.actions.items():
                    new_x = min(max(i + dx, 0), self.n - 1)
                    new_y = min(max(j + dy, 0), self.m - 1)
                    next_state = (new_x, new_y)
                    t_matrix[state][action] = [(1.0, next_state)]

        return t_matrix
    
    def reward_transition(self, next_state):
        r, c = next_state
        done = self.is_terminal(next_state)
        return self.maze[r, c], done
    
    def is_terminal(self, next_state):
        r, c = next_state
        reward = self.maze[r, c]
        return reward == np.max(self.maze)
    

class TD_Agent_SARSA:
    def __init__(self, env, alpha=0.5, gamma=1, step=1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.step = step
        self.states = [(i, j) for i in range(self.env.n) for j in range(self.env.m)]
        self.q_matrix = {(state, action): 0 for state in self.states for action in self.env.actions}

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return random.choice(list(self.env.actions.keys()))
        return max(self.env.actions, key=lambda a: self.q_matrix[(state, a)])

    def step_env(self, state, action):
        transitions = self.env.transition_prob[state][action]
        probs = [p for p, _ in transitions]
        next_states = [s for _, s in transitions]

        idx = np.random.choice(len(next_states), p=probs)
        next_state = next_states[idx]
        reward, done = self.env.reward_transition(next_state)

        return next_state, reward, done

    def SARSA_control(self, n_episodes=10, epsilon=0.1):

        for _ in range(n_episodes):

            episode = []
            state = self.states[np.random.randint(0, len(self.states))]
            action = self.choose_action(state, epsilon)

            t = 0
            T = float('inf')

            while True:

                if t < T:
                    next_state, reward, done = self.step_env(state, action)

                    episode.append((state, action, reward))  # no dummy

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)

                tau = t - self.step

                if tau >= 0:
                    G = 0

                    for i in range(tau, min(tau + self.step, T)):
                        G += (self.gamma ** (i - tau)) * episode[i][2]

                    if tau + self.step < T:
                        s_n, a_n, _ = episode[tau + self.step]
                        G += (self.gamma ** self.step) * self.q_matrix[(s_n, a_n)]

                    s_tau, a_tau, _ = episode[tau]
                    self.q_matrix[(s_tau, a_tau)] += self.alpha * (G - self.q_matrix[(s_tau, a_tau)])

                if tau == T - 1:
                    break

                t += 1
                state = next_state
                action = next_action if not done else None


    def print_policy(self):
        arrows = {"left": "L", "right": "R", "up": "U", "down": "D"}

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


class TD_Agent_QLearning:
    def __init__(self, env, alpha=0.5, gamma=1, step=1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.step = step
        self.states = [(i, j) for i in range(self.env.n) for j in range(self.env.m)]
        self.q_matrix = {(state, action): 0 for state in self.states for action in self.env.actions}

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return random.choice(list(self.env.actions.keys()))
        return max(self.env.actions, key=lambda a: self.q_matrix[(state, a)])

    def step_env(self, state, action):
        transitions = self.env.transition_prob[state][action]
        probs = [p for p, _ in transitions]
        next_states = [s for _, s in transitions]

        idx = np.random.choice(len(next_states), p=probs)
        next_state = next_states[idx]
        reward, done = self.env.reward_transition(next_state)

        return next_state, reward, done

    def Q_Learning_control(self, n_episodes=10, epsilon=0.1):

        for _ in range(n_episodes):

            episode = []
            state = self.states[np.random.randint(0, len(self.states))]
            action = self.choose_action(state, epsilon)

            t = 0
            T = float('inf')

            while True:

                if t < T:
                    next_state, reward, done = self.step_env(state, action)

                    episode.append((state, action, reward))  # no dummy

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)

                tau = t - self.step

                if tau >= 0:
                    G = 0

                    for i in range(tau, min(tau + self.step, T)):
                        G += (self.gamma ** (i - tau)) * episode[i][2]

                    if tau + self.step < T:
                        s_n = episode[tau + self.step][0]   # only need the state
                        G += (self.gamma ** self.step) * max(
                            self.q_matrix[(s_n, a)] for a in self.env.actions
                        )

                    s_tau, a_tau, _ = episode[tau]
                    self.q_matrix[(s_tau, a_tau)] += self.alpha * (G - self.q_matrix[(s_tau, a_tau)])

                if tau == T - 1:
                    break

                t += 1
                state = next_state
                action = next_action if not done else None


    def print_policy(self):
        arrows = {"left": "L", "right": "R", "up": "U", "down": "D"}

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
    agent = TD_Agent_SARSA(env, step = 2)
    agent.SARSA_control()

    agent.print_policy()

    agent = TD_Agent_QLearning(env, step = 2)
    agent.Q_Learning_control()

    agent.print_policy()

if __name__ == "__main__":
    main()
            