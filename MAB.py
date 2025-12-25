import numpy as np

class Arm():

    def __init__(self):
        self.mean = int(np.random.uniform(1, 11))
        self.std = int(np.random.uniform(1,3))
        self.sample_avg = 0
        self.num_occur = 0

    def reward(self):
        reward = int(np.random.normal(loc = self.mean, scale = self.std, size=(1,)))
        self.sample_avg = (self.sample_avg * self.num_occur + reward) / (self.num_occur + 1)
        self.num_occur += 1 
        return reward
    
def decayed_epsilon_greedy(num_arms, epsilon, num_trial_decision):
    Arms = []
    arms_un_explored = np.array([True] * num_arms)
    maxi_exp_reward = 0
    maxi_arm = -1

    for i in range(num_arms):
        Arms.append(Arm())

    for trial in range(num_trial_decision - 1):
        prob = np.random.uniform(low = 0, high = 1, size=(1,))
        choice = -1
        if prob < epsilon and sum(arms_un_explored) != 0:
            #exploration
            choices = np.where(arms_un_explored)[0]
            choice = np.random.choice(choices, size=(1,))[0]
            print(f"Trial{trial+1} Exploration => Arm {choice+1} => Reward = {Arms[choice].reward()}")
            arms_un_explored[choice] = False

        else:
            #exploitation
            choice = maxi_arm
            print(f" Trial{trial+1} Exploitation => Arm {choice+1} => Reward = {Arms[choice].reward()}")

        if Arms[choice].sample_avg > maxi_exp_reward:
                maxi_exp_reward = Arms[choice].sample_avg
                maxi_arm = choice

        epsilon = (1 / (trial + 1) ** 0.01) * epsilon
        # print(f"Epsilon = {epsilon}")

    return maxi_arm, maxi_exp_reward

def incremental_uniform(num_arms, num_trial_decision):
    Arms = []
    arms_un_explored = np.array([True] * num_arms)
    maxi_exp_reward = 0
    maxi_arm = -1

    for i in range(num_arms):
        Arms.append(Arm())

    for trial in range(num_trial_decision - 1):
        choices = np.where(arms_un_explored)[0]
        choice = np.random.choice(choices, size=(1,))[0]

        print(f" Trial{trial+1} Exploitation => Arm {choice+1} => Reward = {Arms[choice].reward()}")
        arms_un_explored[choice] = False

        if Arms[choice].sample_avg > maxi_exp_reward:
                maxi_exp_reward = Arms[choice].sample_avg
                maxi_arm = choice

        if np.sum(arms_un_explored) == 0:
            arms_un_explored = np.array([True] * num_arms)

    return maxi_arm, maxi_exp_reward

def UCB(num_arms, num_trial_decision, c):

    def compute_UCBMax(Arms, trial, c):
        n = len(Arms)
        maxi_arm = -1
        maxi_exp_reward = 0
        for i in range(n):
            if maxi_exp_reward < Arms[i].sample_avg + c * np.sqrt(np.log(trial)/Arms[i].num_occur):
                maxi_exp_reward = Arms[i].sample_avg
                maxi_arm = i
        return maxi_arm

    Arms = []
    arms_un_explored = np.array([True] * num_arms)
    maxi_exp_reward = 0
    maxi_arm = -1

    for i in range(num_arms):
        Arms.append(Arm())

    for trial in range(num_trial_decision - 1):
        if np.sum(arms_un_explored) > 0:
            choices = np.where(arms_un_explored)[0]
            choice = np.random.choice(choices, size=(1,))[0]

            print(f"Trial{trial+1} Exploration => Arm {choice+1} => Reward = {Arms[choice].reward()}")
            arms_un_explored[choice] = False

        else:
            choice = compute_UCBMax(Arms, trial, c)
            print(f"Trial{trial+1} Exploitation => Arm {choice+1} => Reward = {Arms[choice].reward()}")

    for i in range(num_arms):
        if maxi_exp_reward < Arms[i].sample_avg:
            maxi_exp_reward = Arms[i].sample_avg
            maxi_arm = i

    return maxi_arm, maxi_exp_reward

def main():
    num_arms = 10
    epsilon = 1
    num_trial_decision = 20
    c = 1
    # maxi_arm, maxi_exp_reward = decayed_epsilon_greedy(num_arms, epsilon, num_trial_decision)
    # maxi_arm, maxi_exp_reward = incremental_uniform(num_arms, num_trial_decision)
    maxi_arm, maxi_exp_reward = UCB(num_arms, num_trial_decision, c)

    print(f'Arm chosen for trial {num_trial_decision} is {maxi_arm + 1} with a maximum expected value of {maxi_exp_reward}')


main()