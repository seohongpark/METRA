import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerTwoBlockBinEnv(SawyerXYZEnv):
    def __init__(self, front_facing_gripper=True, full_state_reward=False):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.5, 0.40, 0.07)
        obj_high = (0.5, 1, 0.5)

        self.front_facing_gripper = front_facing_gripper
        self.full_state_reward = full_state_reward
        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            front_facing_gripper=front_facing_gripper
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([-0.1, 0.7, 0.04, 0.1, 0.7, 0.04]),
            'hand_init_pos': np.array((0, 0.6, 0.2)),
        }
        self.goals = [np.array([-0.1, 0.7, 0.04, 0.1, 0.7, 0.04])]
        self.goal_idx = 0

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
        if self.front_facing_gripper:
            return full_v1_path_for('sawyer_xyz/sawyer_two_block_bin_ffg.xml')
        else:
            return full_v1_path_for('sawyer_xyz/sawyer_two_block_bin.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, success, hand_distance, obj1_distance, obj2_distance = self.compute_reward(ob, self.goal_idx)
        self.curr_path_length += 1

        info = {
            'metric_reward': reward,
            'metric_success': success,
            'metric_hand_distance': hand_distance,
            'metric_obj1_distance': obj1_distance,
            'metric_obj2_distance': obj2_distance
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        if self.use_dm_backend:
          obj1 = self.sim.named.data.geom_xpos['objGeom']
          obj2 = self.sim.named.data.geom_xpos['obj2Geom']
        else:
          obj1 = self.data.get_geom_xpos('objGeom')
          obj2 = self.data.get_geom_xpos('obj2Geom')

        return np.concatenate([obj1, obj2])

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos[:3].copy()
        qvel[9:21] = 0
        qpos[16:19] = pos[3:].copy()

        self.set_state(qpos, qvel)

    def reset_model(self):
        super()._reset_hand(10)

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = 0.02
        #self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((self.obj_init_pos, [self.objHeight]))

        self._set_obj_xyz(self.obj_init_pos)
        #self._target_pos = self.get_body_com("bin_goal")
        #self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1]]) - np.array(self._target_pos)[:-1]) + self.heightTarget

        return self._get_obs()

    def add_pertask_success(self, obs, goal_idx = None):
        goal_idxs = [goal_idx] if goal_idx is not None else range(len(self.goals))
        for goal_idx in goal_idxs:
            reward, success, hand_distance, obj1_distance, obj2_distance = self.compute_reward(obs['state'], goal_idx)

            obs['metric_reward/goal_'+str(goal_idx)]= reward
            obs['metric_success/goal_'+str(goal_idx)]= success
            obs['metric_hand_distance/goal_'+str(goal_idx)]= hand_distance
            obs['metric_obj1_distance/goal_'+str(goal_idx)]= obj1_distance
            obs['metric_obj2_distance/goal_'+str(goal_idx)]= obj2_distance
        return obs

    def compute_reward(self, obs, goal_idx):

        if goal_idx is None: 
          goal_idx = self.goal_idx
        goal = self.goals[goal_idx]
        hand_distance = np.linalg.norm(obs[:3] -  goal[:3])
        obj1_distance = np.linalg.norm(obs[3:6] - goal[3:6])
        obj2_distance = np.linalg.norm(obs[6:9] - goal[6:9])

        if self.goal_idx in [0,1]:
            #reaching
            reward = -hand_distance
            success = float(hand_distance < 0.1)
        else:
            #pushing, pickplace, stacking
            reward = -obj1_distance -obj2_distance
            success = float((obj1_distance < 0.1) and (obj2_distance < 0.1))

        if self.full_state_reward:
            reward = - obj1_distance - obj2_distance - hand_distance

        return [reward, success, hand_distance, obj1_distance, obj2_distance]
