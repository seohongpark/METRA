from gym.spaces import Box
import numpy as np
from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachPushPickPlaceEnv(SawyerXYZEnv):

    def __init__(self, task_type, full_state_reward=False):
        liftThresh = 0.04
        goal_low=(-0.1, 0.8, 0.05)
        goal_high=(0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)


        self.full_state_reward = full_state_reward
        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.task_type = task_type
        if self.task_type == 'pick_place':
            self.goal = np.array([0.1, 0.8, 0.2])
        elif self.task_type == 'reach':
            self.goal = np.array([-0.1, 0.8, 0.2])
        elif self.task_type == 'push':
            self.goal = np.array([0.1, 0.8, 0.02])


        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh


        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.goal_idx = 0

    def _set_task_inner(self, *, task_type, **kwargs):
        super()._set_task_inner(**kwargs)
        self.task_type = task_type

        # we only do one task from [pick_place, reach, push]
        # per instance of SawyerReachPushPickPlaceEnv.
        # Please only set task_type from constructor.
        if self.task_type == 'pick_place':
            self.goal = np.array([0.1, 0.8, 0.2])
        elif self.task_type == 'reach':
            self.goal = np.array([-0.1, 0.8, 0.2])
        elif self.task_type == 'push':
            self.goal = np.array([0.1, 0.8, 0.02])
        else:
            raise NotImplementedError

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_reach_push_pick_and_place.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, success, hand_distance, goal_distance = self.compute_reward(ob, self.goal_idx)

        self.curr_path_length +=1
        info = {
            'metric_reward': reward,
            'metric_success': success,
            'metric_hand_distance': hand_distance,
            'metric_goal_distance': goal_distance,
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        far_away = np.array([10., 10., 10.])
        return [
            ('goal_' + self.task_type, self._target_pos )
        ]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

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
        self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            if self.task_type == 'push':
                self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
                self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            else:
                self._target_pos = goal_pos[-3:]
                self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def add_pertask_success(self, obs, goal_idx = None):
        goal_idxs = [goal_idx] if goal_idx is not None else range(len(self.goals))
        for goal_idx in goal_idxs:
            reward, success, hand_distance, goal_distance = self.compute_reward(obs['state'], goal_idx)

            obs['metric_reward/goal_'+str(goal_idx)]= reward
            obs['metric_success/goal_'+str(goal_idx)]= success
            obs['metric_hand_distance/goal_'+str(goal_idx)]= hand_distance
            obs['metric_goal_distance/goal_'+str(goal_idx)]= goal_distance

        return obs

    def compute_reward(self, obs, goal_idx):

        if goal_idx is None:
          goal_idx = self.goal_idx
        goal = self.goals[goal_idx]
        if self.full_state_reward:
            hand_distance = np.linalg.norm(obs[:3] - self.hand_init_pos)
        else:
            hand_distance = np.linalg.norm(obs[:3] -  goal[:3])
        obj_distance =  np.linalg.norm(obs[3:6] - goal[:3])

        if self.task_type == 'reach':
            #reaching
            reward = -hand_distance
            success = float(hand_distance < 0.05)
            goal_distance = hand_distance
        else:
            #pushing, pickplace, stacking
            reward = -hand_distance -obj_distance
            success = float(obj_distance < 0.07)
            goal_distance = obj_distance

        success = float(goal_distance <= 0.05) if self.task_type == 'reach' \
                    else float(goal_distance <= 0.07)

        return [reward, success, hand_distance, goal_distance]
