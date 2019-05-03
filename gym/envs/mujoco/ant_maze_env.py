import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np

from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.maze_env_utils import construct_maze, ray_segment_intersect, point_distance

class AntMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            maze_id=0,
            length=1,
            maze_height=0.5,
            maze_size_scaling=3,
            coef_inner_rew=0.,  # a coef of 0 gives no reward to the maze from the wrapped env.
            goal_rew=1.,  # reward obtained when reaching the goal
            *args,
            **kwargs):
        # Vars specific to AntMaze
        self.MODEL_XML = osp.join(osp.dirname(__file__), "assets", "ant.xml")
        self.ORI_IND = 6

        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_id = maze_id
        self.length = length
        self.coef_inner_rew = coef_inner_rew
        self.goal_rew = goal_rew

        tree = ET.parse(self.MODEL_XML)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self.MAZE_STRUCTURE = structure = construct_maze(maze_id=self._maze_id, length=self.length)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name defined")

        _, model_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(model_path)  # here we write a temporal file with the robot specifications. Why not the original one??
        self._goal_range = self._find_goal_range()
        minx, maxx, miny, maxy = self._goal_range
        self.goal = np.array([(minx + maxx)/2., (miny + maxy)/2.])
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def compute_reward(self, achieved_goal, goal, info):
        if self._is_success(achieved_goal, goal):
            return self.goal_rew
        return -0.1*self.goal_rew

    def _is_success(self, achieved_goal, goal):
        gx, gy = goal
        x, y = achieved_goal
        thresh = self.MAZE_SIZE_SCALING * 0.5
        if np.abs(gx - x) <= thresh and np.abs(gy - y) <= thresh:
            return float(True)
        return float(False)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        state = self.state_vector()
        done = False
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'achieved_goal': obs['achieved_goal'].copy(),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _get_obs(self):
        ant_obs = np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        achieved_goal = self.get_body_com("torso")[:2]
        return {
            'observation': np.concatenate([ant_obs, self._get_maze_obs()]).copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

    # Helper functions for Maze Env 
    def _get_maze_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        robot_x, robot_y = self.get_body_com("torso")[:2]
        ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self._sensor_range:
                        wall_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self._sensor_range:
                        goal_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                else:
                    assert False

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])
        return obs

    def get_ori(self):
        """
        The default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        return self.sim.data.qpos[self.ORI_IND]

    def _find_robot(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):  # this only finds one goal!
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    
