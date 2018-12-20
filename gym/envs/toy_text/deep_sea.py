import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class DeepSeaEnv(gym.Env):
    def __init__(self, n=20, random=True):
        self._size = n
        self._move_cost = 0.01 / n
        self._goal_reward = 1.

        self._column = 0
        self._row = 0

        if random:
          rng = np.random.RandomState(None)
          self._action_mapping = rng.binomial(1, 0.5, n)
        else:
          self._action_mapping = np.ones(n)

        self._reset_next_step = False

    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]

        # Compute the reward
        reward = 0.
        if self._column == self._size-1 and action_right:
            reward += self._goal_reward

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size-1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size-1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row-1, self._column)
            self._reset_next_step = True
            return observation, reward, True, {}
        else:
            observation = self._get_observation(self._row, self._column)
            return observation, reward, False, {}

    def reset(self):
        self._reset_next_step = False
        self._column = 0
        self._row = 0
        return self._get_observation(self._row, self._column)

    def _get_observation(self, row, column):
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1
        return observation

    @property
    def obs_shape(self):
        return self.reset().shape

    @property
    def num_actions(self):
        return 2

    @property
    def optimal_return(self):
        return self._goal_reward - (self._size) * self._move_cost
    
