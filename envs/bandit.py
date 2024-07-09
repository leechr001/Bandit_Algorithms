class Bandit:
    """
    Base class for bandits
    """

    def __init__(self, means):
        self.means = means
        self.n_arms = len(means)
        self.optimal_mean = max(means)

    def k(self):
        return self.n_arms
    
    def sample_reward(self, a:int):
        """
        Function to return a reward from the chosen arm
        """
        pass
    
    def pull(self, a:int):
        """
        Accepts a parameter 0 <= a <= K-1 and returns the
        realisation of random variable X with P(X = 1) being
        the mean of the (a+1)th arm.
        """

        if a < 0 or a >= self.k():
            raise ValueError(f"Attempted to pull arm at index {a}. Number of arms is {self.k()}")

        return self.sample_reward(a), self.optimal_mean - self.means[a]