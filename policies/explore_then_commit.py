from .helpers import random_argmax
from .base_discrete import DiscretePolicy

class ETC(DiscretePolicy):
    """
    explore then commit algorithm
    """
    def __init__(self, k:int, m:int):
        """
        k: number of arms in the bandit
        m: number of times to pull each arm in the explore phase
        """
        super().__init__(k)
        self.number_pulls_in_explore = m 
        self.commit_options = []

    def m(self):
        return self.number_pulls_in_explore

    def _predict(self):
        if self.t() < self.m() * self.k():
            return self.t() // self.m()
        
        return random_argmax(self.sample_means)
    
    def update(self, reward):
        """
        Override the update function to stop reward tracking at time mk
        """
        if self.t() < self.m() * self.k():
            super().update(reward)
