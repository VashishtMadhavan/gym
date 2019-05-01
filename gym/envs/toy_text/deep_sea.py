import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class DeepSeaEnv(gym.Env):
    def __init__(self, n=20, random=True):
        self._size = n
        self._move_cost = 0.01 / (n - 1)
        self._goal_reward = 1.
        self._column = 0
        self._row = 0
        self.seed()

        if random:
          self._action_mapping = self.np_random.binomial(1, 0.5, n)
        else:
          self._action_mapping = np.ones(n)

        self.action_space = spaces.Discrete(2)
        self.flat_size = self._size ** 2
        obs_shape = self.flat_size * 2
        self.observation_space = spaces.Discrete(obs_shape)
        self.observation_space.shape = (obs_shape, )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # make sure this aligns with hindsight
    def compute_reward(self, action_right, achieved_goal, goal, info):
        success = list(achieved_goal) == list(goal)
        if success:
            reward = self._goal_reward - self._move_cost
        else:
            reward = -self._move_cost if action_right else 0
        info['is_success'] = float(success)
        info['achieved_goal'] = np.array(achieved_goal)
        info['desired_goal'] = np.array(goal)
        return float(success) - 1.0

    def step(self, action):
        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]
        # State dynamics
        self._row += 1
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)
        info = {}
        done = self._row == self._size - 1
        obs = self._get_observation(self._row, self._column)
        reward = self.compute_reward(action_right, obs[:self.flat_size], obs[self.flat_size:], info)
        return obs, reward, done, info

    def reset(self):
        self._column = 0; self._row = 0
        return self._get_observation(self._row, self._column)

    def _get_observation(self, row, column):
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1
        goal = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        goal[-1, -1] = 1
        return np.concatenate((observation.flatten(), goal.flatten()))

    @property
    def optimal_return(self):
        return self._goal_reward - (self._size - 1) * self._move_cost
    
