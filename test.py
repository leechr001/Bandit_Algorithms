from envs.gaussian_bandit import GaussianBandit
from policies.explore_then_commit import ETC

from experiment import Experiment

bandit = GaussianBandit(means=[0, 0.1, 0.5, 0.9, 1.0])
policy = ETC(k=bandit.k(), m=10)

experiment = Experiment(episodes=100, bandit=bandit, policy=policy)

experiment.run(trials=3, plot=True)