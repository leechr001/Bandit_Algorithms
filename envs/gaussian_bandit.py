import numpy as np
from envs.bandit import Bandit

class GaussianBandit(Bandit):
    """
    Implementation of the gaussian bandit environment
    """
    def __init__(self, means, variances=None):
        super().__init__(means)
        if variances:
            self.variances = variances
        else:
            self.variances = [1 for _ in range(len(means))]

    def sample_reward(self, a: int):
        return np.random.normal(loc=self.means[a],scale=self.variances[a],size=1)