import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class BitFlipEnv(gym.Env):
	""" bit flip environment

	Start of with bitstring of length n and try to achieve a random goal string of length n
	Run for n timesteps flipping random bits by taking actions i, corresponding to each bit
	Successful if goal bitsring is matched within n timesteps

	"""
	def __init__(self, n=40):
		self.n = n
		self.flip_tsteps = n
		self.action_space = spaces.Discrete(n)
		self.observation_space = spaces.Discrete(n*2)
		self.observation_space.shape = (n*2,)
		self.seed()
		self.state = self.np_random.binomial(1, 0.5, size=(2 * self.n)).astype(np.float32)

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def substitute_goal(self, observation, goal):
		k = int(self.n)
		return self.np_random.concatenate([observation[:k], goal], axis=-1)

	def compute_reward(self, achieved_goal, goal, info):
		success = list(achieved_goal) == list(goal)
		info['goal_dist'] = 0.
		info['is_success'] = float(success)
		info['achieved_goal'] = np.array(achieved_goal)
		info['desired_goal'] = np.array(goal)
		return float(success) - 1.

	def step(self, action):
		assert self.action_space.contains(action)
		self.state[action] = 1.0 - self.state[action]
		self.flip_tsteps -= 1
		
		info = {}
		reward = self.compute_reward(self.state[:self.n], self.state[self.n:], info)
		done = self.flip_tsteps <= 0
		return self.state, reward, done, info

	def reset(self):
		self.state = self.np_random.binomial(1, 0.5, size=(2 * self.n)).astype(np.float32)
		self.flip_tsteps = int(self.n)
		return self.state