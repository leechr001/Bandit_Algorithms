import numpy as np
from envs.bandit import Bandit

class BernoulliBandit(Bandit):
    """
    Implementation of the bernoulli bandit environment (Exercise 4.8)
    """

    def sample_reward(self, a: int):
        if max(self.means) > 1 or min(self.means) < 0:
            raise ValueError("Mean is not in [0,1]")
        return np.random.binomial(p=self.means[a],n=1)