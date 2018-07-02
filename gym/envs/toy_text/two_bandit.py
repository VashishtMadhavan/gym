import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class TwoBanditEnv(gym.Env):
    """
    Two-Arm Bandits with random return
    """
    def __init__(self, n=10):
        self.n = n
        self.probs = np.random.uniform(low=0.0, high=1.0, size=2)
        obs_dim = 6
        self.observation_space = spaces.Box(-np.inf*np.ones(obs_dim), np.inf*np.ones(obs_dim))
        self.action_space = spaces.Discrete(2)
        self.t = 0
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
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1.
        obs = np.array([0., reward, done, float(self.t)])
        obs = np.concatenate((obs, action_one_hot))
        return obs, reward, done, {}

    def reset(self):
        self.probs = np.random.uniform(low=0.0, high=1.0, size=2)
        self.t = 0
        action_one_hot = np.zeros(2)
        #action_one_hot[0] = 1.
        obs = np.array([0., 0., 0., 0.])
        obs = np.concatenate((obs, action_one_hot))
        return obs
