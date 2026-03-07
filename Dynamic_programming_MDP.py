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


class Agent:
    def __init__(self, environment, gamma = 1, alpha = 0.5):
        self.gamma = gamma
        self.alpha = alpha
        self.env = environment
        self.states = [(i, j) for i in range(self.env.n) for j in range(self.env.m)]
        
class Agent_MDP(Agent):
    def __init__(self, env, gamma = 1, alpha = 0.5):
        super().__init__(env, gamma, alpha)
        self.value_matrix = np.zeros_like(self.env.maze, dtype = np.float32)
        self.policy = {state: list(self.env.actions.keys()) for state in self.states}

    def value_iteration(self):

        while True:
            delta = 0
            prev_val_matrix = deepcopy(self.value_matrix)
            for state in self.states:
                if self.env.is_terminal(state):
                    continue
                prev_val = prev_val_matrix[state[0], state[1]]
                max_val = -float('inf')
                for action in self.env.actions.keys():
                    val = 0
                    for prob, next_state in self.env.transition_prob[state][action]:
                        val += prob * (self.env.reward_transition(next_state)[0] + self.gamma * prev_val_matrix[next_state[0], next_state[1]])

                    max_val = max(max_val, val)
                delta = max(delta, abs(max_val - prev_val))
                self.value_matrix[state[0], state[1]] = max_val

            if delta < 1e-4:
                break

        for state in self.states:
            if self.env.is_terminal(state):
                continue
            max_val = -float('inf')
            max_action = None
            for action in self.env.actions.keys():
                val = 0
                for prob, next_state in self.env.transition_prob[state][action]:
                        val += prob * (self.env.reward_transition(next_state)[0] + self.gamma * self.value_matrix[next_state[0], next_state[1]])
                if val > max_val:
                    max_val = val
                    max_action = action
        
            self.policy[state] = max_action

    def policy_iteration(self):
        while True:
            while True:
                delta = 0
                prev_val_matrix = deepcopy(self.value_matrix)
                for state in self.states:
                    if self.env.is_terminal(state):
                        continue
                    actions = self.policy[state] 
                    actions = actions if len(actions) == 1 else [actions]
                    val = 0
                    for action in actions:
                        action_val = 0
                        for prob, next_state in self.env.transition_prob[state][action]:
                            reward, _ = self.env.reward_transition(next_state)
                            action_val += prob * (reward + self.gamma * prev_val_matrix[next_state[0], next_state[1]])
                        action_val *= self.env.action_prob[state[0], state[1]][action]
                        val += action_val
                    delta = max(delta, abs(val - prev_val_matrix[state[0], state[1]]))
                    self.value_matrix[state[0], state[1]] = val
                if delta < 1e-4:
                    break

            policy_stable = True
            for state in self.states:
                if self.env.is_terminal(state):
                    continue
                old_action = self.policy[state]
                max_val = -float("inf")
                best_action = None
                for action in self.env.actions:
                    val = 0
                    for prob, next_state in self.env.transition_prob[state][action]:
                        reward, _ = self.env.reward_transition(next_state)
                        val += prob * (reward + self.gamma * self.value_matrix[next_state[0], next_state[1]])

                    if val > max_val:
                        max_val = val
                        best_action = action

                self.policy[state] = best_action

                if best_action != old_action:
                    policy_stable = False

            if policy_stable:
                break

    def print_policy(self):

        arrows = {
            "left":"L",
            "right":"R",
            "up":"U",
            "down":"D"
        }

        for i in range(self.env.n):
            for j in range(self.env.m):

                state = (i,j)

                if self.env.maze[i,j] == -10:
                    print("X", end=" ")
                elif self.env.maze[i,j] == 10:
                    print("G", end=" ")
                else:
                    print(arrows[self.policy[state]], end=" ")

            print()

def main():
    env = np.array([[-1, -1, -1], [-10, -10, -1], [-1, -10, 10]])
    env = Maze(3, 3, env)
    agent = Agent_MDP(env, 1, 0.5)
    agent.value_iteration()

    agent.print_policy()

main()
