import numpy as np
import matplotlib.pyplot as plt

from envs.bandit import Bandit
from policies.base_discrete import DiscretePolicy

class Experiment:
    def __init__(self, episodes:int, bandit:Bandit, policy:DiscretePolicy):
        self.optimal_reward = max(bandit.means)
        self.optimal_arm = np.argmax(bandit.means)
        self.episode = 0 
        self.max_episode = episodes

        self.policy = policy
        self.bandit = bandit

        self.regret = 0
        self.immediate_regret = list()

    def play_episode(self):
        action = self.policy.predict()
        reward, immediate_regret = self.bandit.pull(action)

        self.policy.update(reward)

        self.immediate_regret.append(immediate_regret)
        self.regret += self.immediate_regret[-1]

        self.episode += 1
        return action, reward, self.immediate_regret[-1]
    
    def reset(self):
        self.episode = 0
        self.regret = 0
        self.immediate_regret = list()      
        self.policy.reset() 

    def run(self, trials = 1, plot=True, logging=True):
        trial_regret = list()
        trial_sample_means = list()
        for i in range(trials):
            while self.episode < self.max_episode:
                self.play_episode()

            trial_regret.append(self.immediate_regret)
            trial_sample_means.append(self.policy.sample_means)

            if logging:
                print(f"Trial {i} sample means: {self.policy.sample_means}")
            self.reset()

        if plot:
            for i in range(len(trial_regret)):
                plt.plot(trial_regret[i])
            plt.xticks(np.arange(0, self.max_episode+1, (self.max_episode + 1) // 10 ))
            plt.ylabel("Immediate Regret")
            plt.show()

        #print(self.policy.sample_means)
        
