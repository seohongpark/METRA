import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerBlockPickingEnv(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.5, 0.40, 0.07)
        obj_high = (0.5, 1, 0.5)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )
        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.7, 0.045]),
            'hand_init_pos': np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.liftThresh = liftThresh

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.goal_and_obj_space = Box(
            np.hstack((goal_low[:2], obj_low[:2])),
            np.hstack((goal_high[:2], obj_high[:2])),
        )

        self.goal_space = Box(goal_low, goal_high)
        self._random_reset_space = Box(low=np.array([-0.22, -0.02]),
                                       high=np.array([0.6, 0.8]))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_block_picking.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, placingDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': placingDist,
            'success': float(placingDist <= 0.07)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def _set_goal_xyz(self, goal):
        del goal  # rjulian: ??? What?
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.set_state(qpos, qvel)

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        #self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((self.obj_init_pos, [self.objHeight]))

        self._set_goal_xyz(self._target_pos)
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1]]) - np.array(self._target_pos)[:-1]) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def compute_reward(self, actions, obs):
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - placingGoal)
        reward = -reachDist - placingDist

        return [reward, reachDist, placingDist]
