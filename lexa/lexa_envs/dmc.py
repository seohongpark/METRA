import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import gym
import random
import re
from lexa_envs.base_envs import DmBenchEnv

class DmcEnv(DmBenchEnv):
  def __init__(self, name, size=(64, 64), action_repeat=2, use_goal_idx=False, log_per_goal=False):
    super().__init__(name, action_repeat, size)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.task_type = name.split('_')[0]
   
    self.goal_idx = 0
    self.goals = get_dmc_benchmark_goals(self.task_type)
    self.rendered_goal = False

  def reset(self):
    self._env.reset()
    if not self.use_goal_idx:
      self.goal_idx = np.random.randint(len(self.goals))
    self.goal = self.goals[self.goal_idx]
    self.rendered_goal = False
    self.rendered_goal_obj = self.render_goal()
    return super().reset()

  def _update_obs(self, obs):
    obs = super()._update_obs(obs)
    obs['image_goal'] = self.rendered_goal_obj
    obs['goal'] = self.goal
    obs['state'] = self._env.physics.data.qpos

    if self.log_per_goal:
      for i, goal in enumerate(self.goals):
        obs.update(self.compute_reward(i)[1])
    elif self.use_goal_idx:
      obs.update(self.compute_reward()[1])

    return obs

  def render_offscreen(self):
    return self.render()

  def step(self, action):
    obs, reward, done, info = super().step(action)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    reward = self.compute_reward()[0]
    return obs, reward, done, info

  def get_excluded_qpos(self):
    # Returns the indices of qpos elements that correspond to global coordinates
    task_type = self.task_type
    if task_type == 'walker':
      return [1, 5, 8] # global position and ankles
    if task_type == 'quadruped':
      return [0, 1]

  def _compute_reward(self, goal_idx, pose):
    task_type = self.task_type
    ex = self.get_excluded_qpos()
    distance = self.goals[goal_idx] - self._env.physics.data.qpos
    distance = np.linalg.norm(distance) - np.linalg.norm(distance[ex])
    reward = -distance

    if task_type == 'walker' :
      def get_su(_goal):
        dist = np.abs(pose - _goal)
        dist = dist[..., [0, 2, 3, 4, 6, 7]]
        dist[...,1] = shortest_angle(dist[...,1])
        return dist.max(-1)

      goal = self.goals[goal_idx]
      distance = min(get_su(goal), get_su(goal[..., [0, 1, 2, 6, 7, 8, 3, 4, 5]]))
      return -distance, (distance < 0.7).astype(np.float32)

    if task_type == 'quadruped':
      def get_su(state, goal):
        dist = np.abs(state - goal)
        dist[..., [1, 2, 3]] = shortest_angle(dist[..., [1, 2, 3]])
        if goal_idx in [0, 1, 2, 5, 6, 7, 8, 11]:
          dist = dist[..., [0,1,2,3,4,8,12,16]]
        if goal_idx in [12, 13]:
          dist = dist[..., [0,1,2,3]]
        return dist.max(-1)

      def rotate(s, times=1):
        # Invariance goes as follows: add 1.57 to azimuth, circle legs 0,1,2,3 -> 1,2,3,0
        s = s.copy()
        for i in range(times):
          s[..., 1] = s[..., 1] + 1.57
          s[..., -16:] = np.roll(s[..., -16:], 12)
        return s

      def normalize(s):
        return np.concatenate((s[..., 2:3], quat2euler(s[..., 3:7]), s[..., 7:]), -1)

      state = normalize(pose)
      goal = normalize(self.goals[goal_idx])
      distance = min(get_su(state, goal), get_su(rotate(state, 1), goal), get_su(rotate(state, 2), goal), get_su(
        rotate(state, 3), goal))
      return -distance, (distance < 0.7).astype(np.float32)

  def compute_reward(self, goal_idx=None):
    if goal_idx is None:
      goal_idx = self.goal_idx

    reward, success = self._compute_reward(goal_idx, self._env.physics.data.qpos)

    info = {
      f'metric_success/goal_{goal_idx}': success,
      f'metric_reward/goal_{goal_idx}': reward
    }
    return reward, info

  def get_goal_idx(self):
    return self.goal_idx

  def set_goal_idx(self, idx):
    self.goal_idx = idx

  def get_goals(self):
    return self.goals

  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj

    size = self._env.physics.get_state().shape[0] - self.goal.shape[0]
    self._env.physics.set_state(np.concatenate((self.goal, np.zeros([size]))))
    self._env.step(np.zeros_like(self.action_space.sample()))
    goal_img = self.render()
    self.rendered_goal = True
    return goal_img


def shortest_angle(angle):
  if not angle.shape:
    return shortest_angle(angle[None])[0]
  angle = angle % (2 * np.pi)
  angle[angle > np.pi] = 2 * np.pi - angle[angle > np.pi]
  return angle

def quat2euler(quat):
  rot = Rotation.from_quat(quat)
  return rot.as_euler('XYZ')

def get_dmc_benchmark_goals(task_type):

  if task_type == 'walker':
    # pose[0] is height
    # pose[1] is x
    # pose[2] is global rotation
    # pose[3:6] - first leg hip, knee, ankle
    # pose[6:9] - second leg hip, knee, ankle
    # Note: seems like walker can't bend legs backwards
    
    lie_back = [ -1.2 ,  0. ,  -1.57,  0, 0. , 0.0, 0, -0.,  0.0]
    lie_front = [-1.2, -0, 1.57, 0, 0, 0, 0, 0., 0.]
    legs_up = [ -1.24 ,  0. ,  -1.57,  1.57, 0. , 0.0,  1.57, -0.,  0.0]

    kneel = [ -0.5 ,  0. ,  0,  0, -1.57, -0.8,  1.57, -1.57,  0.0]
    side_angle = [ -0.3 ,  0. ,  0.9,  0, 0, -0.7,  1.87, -1.07,  0.0]
    stand_up = [-0.15, 0., 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1]
    
    lean_back = [-0.27, 0., -0.45, 0.22, -1.5, 0.86, 0.6, -0.8, -0.4]
    boat = [ -1.04 ,  0. ,  -0.8,  1.6, 0. , 0.0, 1.6, -0.,  0.0]
    bridge = [-1.1, 0., -2.2, -0.3, -1.5, 0., -0.3, -0.8, -0.4]

    head_stand = [-1, 0., -3, 0.6, -1, -0.3, 0.9, -0.5, 0.3]
    one_feet = [-0.2, 0., 0, 0.7, -1.34, 0.5, 1.5, -0.6, 0.1]
    arabesque = [-0.34, 0., 1.57, 1.57, 0, 0., 0, -0., 0.]
    # Other ideas: flamingo (hard), warrior (med), upside down boat (med), three legged dog

    goals = np.stack([lie_back, lie_front, legs_up, 
                      kneel, side_angle, stand_up, lean_back, boat,
                      bridge, one_feet, head_stand, arabesque])

  if task_type == 'quadruped':
    # pose[0,1] is x,y
    # pose[2] is height
    # pose[3:7] are vertical rotations in the form of a quaternion (i think?)
    # pose[7:11] are yaw pitch knee ankle for the front left leg
    # pose[11:15] same for the front right leg
    # pose[15:19] same for the back right leg
    # pose[19:23] same for the back left leg

    
    lie_legs_together = get_quadruped_pose([0, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]), [-0.7, 0.7, -0.7, 0.7])
    lie_rotated = get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[0, 1, 2, 3]))
    lie_two_legs_up = get_quadruped_pose([0.8, 3.14, 0], 0.2, dict(out_up=[1, 3], down=[0, 2]))

    lie_side = get_quadruped_pose([0., 0, -1.57], 0.3, dict(out=[0,1,2, 3]), [-0.7, 0.7, -0.7, 0.7])
    lie_side_back = get_quadruped_pose([0., 0, 1.57], 0.3, dict(out=[0,1,2, 3]), [-0.7, 0.7, -0.7, 0.7])
    stand = get_quadruped_pose([1.57, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))
    stand_rotated = get_quadruped_pose([0.8, 0, 0], 0.2, dict(up=[0, 1, 2, 3]))
 
    stand_leg_up = get_quadruped_pose([1.57, 0, 0.0], 0.7, dict(down=[0, 2, 3], out_up=[1]))
    attack = get_quadruped_pose([1.57, 0., -0.4], 0.7, dict(out=[0, 1, 2, 3]))
    balance_front = get_quadruped_pose([1.57, 0.0, 1.57], 0.7, dict(up=[0, 1, 2, 3]))
    balance_back = get_quadruped_pose([1.57, 0.0, -1.57], 0.7, dict(up=[0, 1, 2, 3]))
    balance_diag = get_quadruped_pose([1.57, 0, 0.0], 0.7, dict(down=[0, 2], out_up=[1,3]))

    goals = np.stack([ lie_legs_together, lie_rotated, lie_two_legs_up,
                      lie_side, lie_side_back, stand, stand_rotated,
                      stand_leg_up, attack, balance_front, balance_back,  balance_diag])

  return goals

def get_quadruped_pose(global_rot, global_pos=0.5, legs={}, legs_rot=[0, 0, 0, 0]):
  """

  :param angles: along height, along depth, along left-right
  :param height:
  :param legs:
  :return:
  """
  if not isinstance(global_pos, list):
    global_pos = [0, 0, global_pos]
  pose = np.zeros([23])
  pose[0:3] = global_pos
  pose[3:7] = (Rotation.from_euler('XYZ', global_rot).as_quat())

  pose[[7, 11, 15, 19]] = legs_rot
  for k, v in legs.items():
    for leg in v:
      if k == 'out':
        pose[[8 + leg * 4]] = 0.5  # pitch
        pose[[9 + leg * 4]] = -1.0  # knee
        pose[[10 + leg * 4]] = 0.5  # ankle
      if k == 'inward':
        pose[[8 + leg * 4]] = -0.35  # pitch
        pose[[9 + leg * 4]] = 0.9  # knee
        pose[[10 + leg * 4]] = -0.5  # ankle
      elif k == 'down':
        pose[[8 + leg * 4]] = 1.0  # pitch
        pose[[9 + leg * 4]] = -0.75  # knee
        pose[[10 + leg * 4]] = -0.3  # ankle
      elif k == 'out_up':
        pose[[8 + leg * 4]] = -0.2  # pitch
        pose[[9 + leg * 4]] = -0.8  # knee
        pose[[10 + leg * 4]] = 1.  # ankle
      elif k == 'up':
        pose[[8 + leg * 4]] = -0.35  # pitch
        pose[[9 + leg * 4]] = -0.2  # knee
        pose[[10 + leg * 4]] = 0.6  # ankle

  return pose
