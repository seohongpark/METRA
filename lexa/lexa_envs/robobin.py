import threading
import os
import re

import gym
import numpy as np

import metaworld.envs.mujoco.sawyer_xyz.v1 as sawyer
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer
from lexa_envs.base_envs import BenchEnv

class RoboBinEnv(BenchEnv):
  def __init__(self, action_repeat, use_goal_idx=False, log_per_goal=False, 
                image_width=64, metric_rew_cap=100000):
    super().__init__(action_repeat)

    self._env = sawyer.SawyerTwoBlockBinEnv()
    self._env.random_init = False

    #workspace limits
    self._env.mocap_low = (-0.5, 0.40, 0.07)
    self._env.mocap_high = (0.5, 0.8, 0.5)
    self._env.goals = get_robobin_benchmark_goals()

    self._action_repeat = action_repeat
    self._width = image_width
    self.metric_rew_cap = metric_rew_cap
    self._size = (self._width, self._width)

    #camera parameters
    self.renderer = DMRenderer(self._env.sim, camera_settings=dict(
          distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180))

    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.rendered_goal = False

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += min(reward, self.metric_rew_cap)
      if done:
        break
    obs = self._get_obs(state)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return obs, total_reward, done, info

  def reset(self):
    self.rendered_goal = False
    if self.use_goal_idx:
      self._env.goal = self.get_goals()[self.get_goal_idx()]
    return super().reset()

  def _get_obs(self, state):
    obs = super()._get_obs(state)
    # obs['image_goal'] = self.render_goal()
    # obs['goal'] = self._env.goals[self._env.goal_idx]
    if self.log_per_goal:
      obs = self._env.add_pertask_success(obs)
    elif self.use_goal_idx:
      obs = self._env.add_pertask_success(obs, self._env.goal_idx)

    return obs

  def set_goal_idx(self, idx):
    self._env.goal_idx = idx

  def get_goal_idx(self):
    return self._env.goal_idx

  def get_goals(self):
    return self._env.goals


  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj
    # TODO use self.render_state

    obj_init_pos_temp = self._env.init_config['obj_init_pos'].copy()
    goal = self._env.goals[self._env.goal_idx]

    self._env.init_config['obj_init_pos'] = goal[3:]
    self._env.obj_init_pos = goal[3:]
    self._env.hand_init_pos = goal[:3]
    self._env.reset_model()
    action = np.zeros(self._env.action_space.low.shape)
    state, reward, done, info = self._env.step(action)

    goal_obs = self.render_offscreen()
    self._env.hand_init_pos = self._env.init_config['hand_init_pos']
    self._env.init_config['obj_init_pos'] = obj_init_pos_temp
    self._env.obj_init_pos = self._env.init_config['obj_init_pos']
    self._env.reset()

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  def render_state(self, state):
    assert (len(state.shape) == 1)
    # Save init configs
    hand_init_pos = self._env.hand_init_pos
    obj_init_pos = self._env.init_config['obj_init_pos']
    # Render state
    hand_pos, obj_pos, hand_to_goal = np.split(state, 3)
    self._env.hand_init_pos = hand_pos
    self._env.init_config['obj_init_pos'] = obj_pos
    self._env.reset_model()
    obs = self._get_obs(state)
    # Revert environment
    self._env.hand_init_pos = hand_init_pos
    self._env.init_config['obj_init_pos'] = obj_init_pos
    self._env.reset()
    return obs['image']

  def render_states(self, states):
    assert (len(states.shape) == 2)
    imgs = []
    for s in states:
      img = self.render_state(s)
      imgs.append(img)
    return np.array(imgs)

def get_robobin_benchmark_goals():
  pos1 = np.array([-0.1, 0.7, 0.04])
  pos2 = np.array([ 0.1, 0.7, 0.04])
  delta = np.array([0, 0.15, 0])
  v_delta = np.array([0,0,0.06])
  hand = np.array([0, 0.65, 0.2])

  goaldictlist = [

    #reaching
    {'obj1': pos1, 'obj2': pos2, 'hand': hand + np.array([0.12,0.1, -0.1])},
    {'obj1': pos1, 'obj2': pos2, 'hand': hand + np.array([-0.1,0.2, -0.1])},

    #pushing
    {'obj1': pos1, 'obj2': pos2 + delta, 'hand': hand},
    {'obj1': pos1 - delta, 'obj2': pos2, 'hand': hand},

  #push both
    {'obj1': pos1+delta, 'obj2': pos2 + delta, 'hand': hand},
    {'obj1': pos1-delta, 'obj2': pos2 - delta, 'hand': hand},

    #pickplace
    {'obj1': pos2 + delta, 'obj2': pos2, 'hand': hand},

    #pickplace both
    {'obj1': pos2+delta, 'obj2': pos1+delta, 'hand': hand}]

  return [np.concatenate([_dict['hand'], _dict['obj1'], _dict['obj2']])
                          for _dict in goaldictlist]
