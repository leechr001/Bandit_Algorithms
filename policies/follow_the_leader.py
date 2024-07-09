from .helpers import random_argmax
from .base_discrete import DiscretePolicy

class FollowTheLeader(DiscretePolicy):
    def _predict(self):
        if self.t() < self.k():
            return self.t()
        
        return random_argmax(self.sample_means)

