import threading
import os
import re

import gym
import numpy as np
import pickle
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer, MjPyRenderer
from d4rl.kitchen.adept_envs.simulation.sim_robot import  RenderMode


def get_device_id():
  return int(os.environ.get('GL_DEVICE_ID', 0))


class DmBenchEnv():

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  def _update_obs(self, obs):
    return obs

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
       break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}

    return self._update_obs(obs), reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return self._update_obs(obs)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class BenchEnv():
  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    return self._get_obs(state)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode, self._width, self._width)

  def render_offscreen(self):
    img = self.renderer.render_offscreen(
                self._width, self._width, mode=RenderMode.RGB, camera_id=-1)
    return np.flipud(np.fliplr(img))

  def _get_obs(self, state):
    return {'image': self.render_offscreen(), 'state': state}
