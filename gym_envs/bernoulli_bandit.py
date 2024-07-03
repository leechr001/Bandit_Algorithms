"""
I will be follwing the tutorial: 
    https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
With help as needed from the implementation in gym of a bernoulli bandit found here: 
    https://github.com/openai/mlsh/blob/master/gym/gym/envs/rl2/bernoulli_bandit.py
"""
from gym.utils import colorize
from gym import Env, spaces

from io import StringIO
import sys

class BernoulliBanditEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n_arms, episode_cnt):
        self.n_arms = n_arms
        self.episode_cnt = episode_cnt

        # not sure I understand this part, there isn't really observation for bandit?
        # maybe observations space should be past actions but I would think this would be
        # stored by the policy controller

        # Copied from implementation of bernoulli bandit
        self.observation_space = spaces.Tuple([
            # State: just a placeholder for bandits
            spaces.Discrete(1),
            # Previous action
            spaces.Discrete(n_arms),
            # Previous reward
            spaces.Box(low=-1, high=1, shape=(1,)),
            # Previous termination flag
            # Since a bandit task always terminates after one time step, the termination flag is always on
            spaces.Box(low=0, high=1, shape=(1,)),
        ])

        self.action_space = spaces.Discrete(n_arms)

        #hidden states
        self._arm_means = None
        self._episode_cnt = None
        self._last_action = None
        self._last_reward = None
        self._last_terminal = None

        self.reset()

    def _get_obs(self):
        return 0, self._last_action, self._last_reward, self._last_terminal
    
    def _get_info(self):
        return {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._arm_means = self.np_random.uniform(low=0, high=1, size=(self.n_arms,))
        self._episode_cnt = 0
        self._last_action = 0
        self._last_reward = 0
        self._last_terminal = 1

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        observation = self._get_obs()
        reward = self.np_random.binomial(p=self._arm_means[action],n=1) 
        terminated = True #bandit terminates after each step
        truncated = False #we do not truncate
        info = self._get_info()

        self._episode_cnt += 1
        self._last_action = action
        self._last_reward = reward
        self._last_terminal = terminated

        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human', close=False):
        """
        Renders the bandit environment in ASCII style. Closely resembles the rendering implementation of algorithmic
        tasks.
        """
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('#Episode: {}\n'.format(self._episode_cnt))
        outfile.write('Total reward so far: {}\n'.format(self._total_reward))
        outfile.write('Action:')
        for idx, mean in enumerate(self._arm_means):
            if idx == self._last_action:
                if self._last_reward == 1:
                    outfile.write(' ' + colorize('%.2f' % mean, 'green', highlight=True))
                else:
                    outfile.write(' ' + colorize('%.2f' % mean, 'red', highlight=True))
            else:
                outfile.write(' ' + '%.2f' % mean)
        outfile.write('\n')

        return outfile


    
