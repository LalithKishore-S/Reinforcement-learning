import pandas as pd
import numpy as np

class Seed:
    def __init__(self, k, seed):
        self.k = k
        self.seed = seed
        self.average = 0
        self.num_trials = 0

    def choose(self, beta, prob):
        prob = prob[self.seed]
        prob = np.sum(prob, axis = 0)
        self.average = (self.average * self.num_trials + np.sum(prob >= beta))/(self.num_trials + 1)
        self.num_trials += 1
        return 
    
def greedy(trial, seeds, num_seeds_considered, prob):
    seed_not_chosen = np.array([True] * num_seeds_considered)
    maxi_exp_reward = 0
    maxi_arm = -1

    for t in range(trial):
        beta = np.random.uniform(low = 0, high = 1, size=(1,))
        
        if np.sum(seed_not_chosen) != 0:
            choices = np.where(seed_not_chosen)[0]
            choice = np.random.choice(choices, size=(1,))[0]
        else:
            choice = maxi_arm
        seeds[choice].choose(beta, prob)
        if seeds[choice].average > maxi_exp_reward:
                maxi_exp_reward = seeds[choice].average
                maxi_arm = choice

    return maxi_arm, maxi_exp_reward


def UCB(trial, seeds, num_seeds_considered, prob, c = 1):

    def compute_UCBMax(seeds, trial, c = 1):
        n = len(seeds)
        maxi_arm = -1
        maxi_exp_reward = 0
        for i in range(n):
            if maxi_exp_reward < seeds[i].average + c * np.sqrt(np.log(trial)/seeds[i].num_trials):
                maxi_exp_reward = seeds[i].average
                maxi_arm = i
        return maxi_arm

    seeds_un_explored = np.array([True] * num_seeds_considered)
    maxi_exp_reward = 0
    maxi_arm = -1


    for trial in range(1, trial+1):
        beta = np.random.uniform(low = 0, high = 1, size=(1,))
        if np.sum(seeds_un_explored) > 0:
            choices = np.where(seeds_un_explored)[0]
            choice = np.random.choice(choices, size=(1,))[0]
            seeds_un_explored[choice] = False

        else:
            choice = compute_UCBMax(seeds, trial, c)

        seeds[choice].choose(beta, prob)

    for i in range(num_seeds_considered):
        if maxi_exp_reward < seeds[i].average:
            maxi_exp_reward = seeds[i].average
            maxi_arm = i

    return maxi_arm, maxi_exp_reward

    

def main():
    trials = 100
    num_Users = 5

    k = np.random.randint(low = 1, high =  num_Users)
    num_seeds_considered = 10

    print(f" Number of Users = {num_Users}")
    print(f" Trials = {trials}")
    print(f" Number of users in each seed (k) = {k}")
    print(f" Number of seeds considered = {num_seeds_considered}")

    set_seeds = []
    seeds = []
    i = 0
    while len(set_seeds) < num_seeds_considered:
        s = np.random.choice(num_Users, size=k, replace=False)
        s_sorted = tuple(sorted(s))          
        if s_sorted not in set_seeds:
            set_seeds.append(s_sorted)
            seeds.append(Seed(k, list(s)))

    print(" Seeds under consideration :")
    for i in range(num_seeds_considered):
        print(seeds[i].seed)

    prob_matrix = np.random.uniform(low = 0, high = 1, size = (num_Users, num_Users))
    col_sums = prob_matrix.sum(axis=0)          
    prob_matrix = prob_matrix / col_sums

    optimal_seed_greedy, reward_g = greedy(trials, seeds, num_seeds_considered, prob_matrix)
    optimal_seed_UCB, reward_u = UCB(trials, seeds, num_seeds_considered, prob_matrix)

    print("Optimal seed using greedy => ", set_seeds[optimal_seed_greedy], "Reward => ", reward_g)
    print("Optimal seed using UCB => ", set_seeds[optimal_seed_UCB], "Reward => ", reward_u)

main()