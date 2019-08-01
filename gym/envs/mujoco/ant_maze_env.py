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
        self.goal_dist_radius = 0.15
        mujoco_env.MujocoEnv.__init__(self, 'ant_maze.xml', 5)
        utils.EzPickle.__init__(self)

    @property
    def contact_forces(self):
        return np.clip(self.sim.data.cfrc_ext, -1, 1)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action, self.frame_skip)

        curr_pos = self.get_current_pos()
        reward = -1.0
        if np.linalg.norm(curr_pos - self.goal_pos) < self.goal_dist_radius:
            reward = 0.0
        obs = self._get_obs()
        return obs, reward, False, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt
        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))
        #return np.concatenate((x_velocity, y_velocity, position, velocities))
        return np.concatenate((x_velocity, y_velocity, position, velocities, self.goal_pos.copy()))

    def get_current_pos(self):
        return np.copy(self.get_body_com("torso"))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
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

    @property
    def tasks(self):
        t = dict()
        return t
