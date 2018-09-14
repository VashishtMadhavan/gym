import gym
from gym import spaces
from ple import PLE
import numpy as np
import os
from gym.envs.ple.ple_env import PLEEnv

class MetaPLEEnv(gym.Env):
    def __init__(self, display_screen=True):
        self.env_names = ['originalGame','nosemantics','noobject','nosimilarity','noaffordance']
        self.disp = display_screen
        self.curr_env = PLEEnv(game_name=np.random.choice(self.env_names), display_screen=self.disp)
        self.action_space = self.curr_env.action_space
        self.screen_width, self.screen_height = self.curr_env.game_state.getScreenDims()
        self.observation_space = self.curr_env.observation_space

    def step(self, a):
        return self.curr_env.step(a)

    def _get_image(self):
       return self.curr_env._get_image()

    @property
    def _n_actions(self):
        return len(self.curr_env._action_set)

    def reset(self):
        self.curr_env = PLEEnv(game_name=np.random.choice(self.env_names), display_screen=self.disp)
        self.action_space = self.curr_env.action_space
        self.screen_width, self.screen_height = self.curr_env.game_state.getScreenDims()
        self.observation_space = self.curr_env.observation_space
        return self.curr_env.reset()

    def render(self, mode='human', close=False):
        self.curr_env.render(mode=mode, close=close)


    def seed(self, seed=None):
        self.curr_env.seed(seed=seed)