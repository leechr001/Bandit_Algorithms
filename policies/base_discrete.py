import numpy as np

class DiscretePolicy:
    def __init__(self, n_arms):
        self.actions = list()
        self.rewards = list()
        self._rewards = [list() for _ in range(n_arms)]
        self.sample_means = np.empty((n_arms,), float)
        self.n_arms = n_arms

    def predict(self):
        action = self._predict()
        self.actions.append(action)
        return action

    def _predict(self):
        """
        Override this method to implement logic
        """
        pass

    def update(self, reward):
        self.rewards.append(reward)
        last_action = self.actions[-1]

        self._rewards[last_action].append(self.rewards[-1])
        self.sample_means[last_action] = sum(self._rewards[last_action]) / len(self._rewards[last_action])

    def reset(self):
        self.actions = list()
        self.rewards = list()
        self._rewards = [list() for _ in range(self.n_arms)]
        self.sample_means = np.empty((self.n_arms,), float)

    def t(self):
        """
        returns the turn (time) number, denoted t in bandit algorithms

        THIS IS 0-indexed!
        """
        return len(self.actions)
    
    def k(self):
        """
        returns the number of arms, denoted k in bandit algorithms
        """

        return self.n_arms
