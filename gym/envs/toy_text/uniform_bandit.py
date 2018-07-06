import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class UniformBanditEnv(gym.Env):
    """
    Uniform Two-Armed Bandits with Dependent Arms
    """
    def __init__(self):
        self.n = 100
        p1 = np.random.uniform(low=0.0, high=1.0)
        p2 = 1. - p1
        self.probs = np.array([p1, p2])
        obs_dim = 4 + 2
        self.observation_space = spaces.Box(-np.inf*np.ones(obs_dim), np.inf*np.ones(obs_dim))
        self.action_space = spaces.Discrete(2)
        self.t = 0
        self.prev_reward = 0.
        self.prev_done = 0.
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)
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
        self.prev_reward = reward
        self.prev_done = done
        return obs, reward, done, {}

    def reset(self):
        p1 = np.random.uniform(low=0.0, high=1.0)
        p2 = 1. - p1
        self.probs = np.array([p1, p2])
        self.t = 0
        self.prev_reward = 0.
        self.prev_done = 0.
        action_one_hot = np.zeros(2)
        action_one_hot[0] = 1.
        obs = np.array([0., 0., 0., 0.])
        obs = np.concatenate((obs, action_one_hot))
        return obs
