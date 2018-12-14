import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class DeepSeaEnv(gym.Env):
    """
    Deep Sea Environment from RPF Paper
    """
    def __init__(self, n=20, random=True):
        self._n = n
        self._move_cost = 0.01 / n
        self._goal_reward = 1.

        self._x = 0
        self._y = 0
        self.seed()

        self._act_map = self.np_random.binomial(1, 0.5, n)
        self._reset_next_step = False
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(n**2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action_right = action == self._act_map[self._x]
        done = False

        # State dynamics
        if action_right:  # right
            self._y += 1
            reward = -self._move_cost
        else:  # left
            self._y = np.max(self._y - 1, 0)
            reward = 0
        self._x += 1

        if self._y == (self._n - 1):
            reward += 1

        # Check Termination
        if self._x == (self._n - 1):
            done = True
        return self._get_observation(self._x, self._y), reward, done, {}

    def _get_observation(self, x, y):
        state_map = np.zeros((self._n, self._n), dtype=np.float32)
        state_map[x][y] = 1.
        return state_map.flatten()

    def reset(self):
        self._x = 0
        self._y = 0
        return self._get_observation(self._x, self._y)

    @property
    def optimal_return(self):
        return self._goal_reward - ((self._n - 1) * self._move_cost)
    
