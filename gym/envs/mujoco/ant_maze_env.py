import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py.generated import const

class AntMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Observation Space:
        - x torso COM velocity
        - y torso COM velocity
        - 15 joint positions
        - 14 joint velocities
        - (optionally, commented for now) 84 contact forces
    """
    def __init__(self):
        self.goal_pos = np.array([4.0, 0.0, 0.75])
        self.goal_dist_radius = 0.1
        mujoco_env.MujocoEnv.__init__(self, 'ant_maze.xml', 5)
        utils.EzPickle.__init__(self)

    # TODO: could potentially add other sources of reward
    def step(self, action):
        self.prev_torso_pos = np.copy(self.get_body_com("torso")[:2])
        self.do_simulation(action, self.frame_skip)

        curr_pos = self.get_current_pos()
        reward = -np.linalg.norm(curr_pos - self.goal_pos)
        obs = self._get_obs()
        return obs, reward, False, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()
        contact_force = np.clip(self.sim.data.cfrc_ext, -1, 1).flat.copy()

        torso_pos = np.copy(self.get_body_com("torso")[:2])
        torso_vel = (torso_pos - self.prev_torso_pos) / self.dt

        #return np.concatenate((x_velocity, y_velocity, position, velocities))
        return np.concatenate((torso_vel, position, velocities, contact_force))

    def get_current_pos(self):
        return np.copy(self.get_body_com("torso"))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_torso_pos = np.copy(self.get_body_com("torso")[0:2])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent
        self.viewer.cam.lookat[0] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -85
        self.viewer.cam.azimuth = 235