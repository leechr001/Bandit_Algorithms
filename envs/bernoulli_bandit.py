import numpy as np
from envs.bandit import Bandit

class BernoulliBandit(Bandit):
    """
    Implementation of the bernoulli bandit environment (Exercise 4.8)
    """

    def sample_reward(self, a: int):
        return np.random.binomial(p=self.means[a],n=1)