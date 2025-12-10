import numpy as np

class Graph:
    def __init__(self, edge_list, nodes, goal_state):
        self.vertices = nodes
        self.num_vertices = len(self.vertices)
        self.R_matrix = self.get_adj_matrix(edge_list, self.num_vertices, goal_state)

    def get_adj_matrix(self, edge_list, n, goal):
        adj_matrix = np.zeros((n , n))
        for i in range(n):
            for j in range(n):
                adj_matrix[i, j] = -1
        for i in range(edge_list.shape[0]):
            node1, node2, weight = edge_list[i, 0] - 1, edge_list[i, 1] - 1, edge_list[i, 2]
            adj_matrix[node1, node2] = weight

        # adj_matrix[goal, goal] = np.max(edge_list[:,2])

        return adj_matrix
    

class Agent:
    def __init__(self, edge_list, nodes, goal_state):
        self.goal_state = goal_state - 1 
        self.R_matrix = Graph(edge_list, nodes, self.goal_state).R_matrix
        print(self.R_matrix)
        self.gamma = 0.8
        self.Q_matrix = self.get_Q_matrix()

    def get_Q_matrix(self):
        q_matrix = np.zeros_like(self.R_matrix)
        n = q_matrix.shape[0]
        n_iter = 100

        for i in range(n_iter):
            # print(f'Iteration {i+1}')
            curr_state = np.random.randint(low = 0, high = q_matrix.shape[0] , size = (1,))[0]
            # print(curr_state)
            
            while (curr_state != self.goal_state):
                possible_actions = [j for j in range(n) if self.R_matrix[curr_state, j] != -1]
                # print(possible_actions)
                action = np.random.choice(possible_actions, size=(1,))[0]
                possible_actions_next = [j for j in range(n) if self.R_matrix[action, j] != -1]
                
                q_matrix[curr_state, action] = self.R_matrix[curr_state, action] + self.gamma * max(q_matrix[action, possible_actions_next])
                curr_state = action
                # print(q_matrix)

        return q_matrix

    def get_max_reward_path(self, initial_state):
        path = [initial_state]
        curr_state = initial_state - 1

        while(curr_state != self.goal_state):
            path.append(int(np.argmax(self.Q_matrix[curr_state])) + 1)
            curr_state = path[-1] - 1

        return path
    

def main():
    n = 6
    nodes = np.array([i for i in range(1,n + 1)])
    edge_list = np.array([(1, 5, 0), (2, 6, 100), (2, 4, 0), (3, 4, 0), (4, 3, 0), (4, 2, 0), (4, 5, 0), (5, 6, 100), (5, 1, 0), (5, 4, 0), (6, 2, 0), (6, 6, 100), (6, 5, 0)])
    #print(edge_list)
    goal_state = 6
    initial_state = 3

    agent = Agent(edge_list, nodes, goal_state)

    print('Agent brain :')
    print(agent.Q_matrix)
    optimal_path = agent.get_max_reward_path(initial_state)

    print('Optimal path :')
    print(optimal_path)

main()
        


        





	
