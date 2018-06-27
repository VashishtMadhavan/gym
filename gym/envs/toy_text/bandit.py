import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class BanditEnv(gym.Env):
    """
    Multi-Arm Bandits with random return
    """
    def __init__(self, k=5, n=10):
        self.k = k
        self.n = n
        self.probs = np.random.uniform(low=0.0, high=1.0, size=self.k)
        obs_dim = 3 + self.k
        self.observation_space = spaces.Box(-np.inf*np.ones(obs_dim), np.inf*np.ones(obs_dim))
        self.action_space = spaces.Discrete(self.k)
        self.t = 0
        self.prev_reward = 0.
        self.prev_done = 0.
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.
        self.t += 1
        if np.random.uniform() < self.probs[action]:
            reward = 1.
        done = False
        if self.t >= self.n:
            done = True
        action_one_hot = np.zeros(self.k)
        action_one_hot[action] = 1.
        obs = np.array([0., reward, done])
        obs = np.concatenate((obs, action_one_hot))
        self.prev_reward = reward
        self.prev_done = done
        return obs, reward, done, {}

    def reset(self):
        self.probs = np.random.uniform(low=0.0, high=1.0, size=self.k)
        self.t = 0
        self.prev_reward = 0.
        self.prev_done = 0.
        action_one_hot = np.zeros(self.k)
        action_one_hot[0] = 1.
        obs = np.array([0., 0., 0.])
        obs = np.concatenate((obs, action_one_hot))
        return obs
