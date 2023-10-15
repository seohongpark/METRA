import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerTwoDrawersOpenEnv(SawyerXYZEnv):
    def __init__(self):

        super().__init__(
            self.model_name,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
        )

        self.drawer1_base_pos = np.array([-0.17, 0.9, 0.04])
        self.drawer2_base_pos = np.array([ 0.17, 0.9, 0.04])

        #obj_init_pos indicates the extent to which each of the drawers are open
        self.init_config = {
            'obj_init_pos': np.array([0, 0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }

        self.obj_init_pos  = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.goal = np.array([0.1,0.1])

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_two_drawers.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        pullDist1 =  np.abs(ob[3:6] - self._target_pos1)[1]
        pullDist2 =  np.abs(ob[6:9] - self._target_pos2)[1]
        reward = -pullDist1 - pullDist2

        self.curr_path_length += 1
        info = {
            'pullDist1': pullDist1,
            'pullDist2': pullDist2,
            'goalDist' : pullDist1 + pullDist2,
            'success': float(pullDist1 <= 0.08 and pullDist2 <= 0.08)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        drawer1 = self.data.get_body_xpos('drawer1').copy()
        drawer2 = self.data.get_body_xpos('drawer2').copy()
        return np.concatenate([drawer1, drawer2])

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = (self._get_site_pos('handleStart').copy() + self.data.get_geom_xpos('drawer_wall2').copy()) / 2
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self._target_pos1 = self.drawer1_base_pos - np.array([0, self.goal[0], 0])
        self._target_pos2 = self.drawer2_base_pos - np.array([0, self.goal[1], 0])

        self.sim.model.body_pos[self.model.body_name2id('drawer1')] = self.drawer1_base_pos - np.array([0,self.obj_init_pos[0], 0])
        self.sim.model.body_pos[self.model.body_name2id('drawer2')] = self.drawer2_base_pos - np.array([0,self.obj_init_pos[1], 0])

        self.maxPullDist = 0.4
        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False
