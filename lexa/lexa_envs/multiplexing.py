import random
import numpy as np
import gym
import re
from lexa_envs.utils import subdict


class MultiplexedEnv:
  def __init__(self, envs, action_repeat, size=(64, 64), use_goal_idx=False, log_per_goal=False):
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.envs = envs
    
    self.goals = sum(list(list(range(len(_env.get_goals()))) for _env in self.envs), [])
    self.active_env_idx = 0
    self.goal_idx = 0
  
  def reset(self):
    if not self.use_goal_idx:
      self.active_env_idx = random.randint(1, len(self.envs)) - 1
    return self.clean_obs(self.active.reset())
  
  def step(self, action):
    o, r, d, i = self.active.step(action)
    return self.clean_obs(o), r, d, i
  
  def clean_obs(self, obs):
    # This is needed for episode logging to work correctly
    #obs = subdict(obs, ['image', 'image_goal'] + list(k for k in obs.keys() if 'metric_' in k))
    keys = list(k for k in obs.keys() if 'metric_' in k)
    orig_obs = obs
    obs = subdict(obs, ['image', 'image_goal'])
    
    if self.use_goal_idx:
      # rename goals to global indexing
      # TODO implement this during training time as well - need to fill in other envs with zeros
      lens = np.cumsum(np.array(list(len(_env.get_goals()) for _env in self.envs)))
      alens = [0] + list(lens)
      offset = alens[self.active_env_idx]
      new_keys = list([re.sub('(metric_.*/goal_)(\d+)', lambda e: f'{e.group(1)}{int(e.group(2)) + offset}', k)
                       for k in keys])
      for i, k in enumerate(keys):
        obs[new_keys[i]] = orig_obs.pop(k)
    
    # This is a hack needed because the code expects goal and state to be there
    obs['goal'] = np.zeros(10)
    obs['state'] = np.zeros(10)
    obs['env_idx'] = self.active_env_idx
    
    return obs
  
  def get_goal_idx(self):
    return self.goal_idx
  
  def set_goal_idx(self, idx):
    # The goals are numbered throughout, i.e. 0-4 first env, 5-9 second env, etc
    self.goal_idx = idx
    lens = np.cumsum(np.array(list(len(_env.get_goals()) for _env in self.envs)))
    self.active_env_idx = np.argmax(idx < lens)
    self.active.reset()
    alens = [0] + list(lens)
    idx - alens[self.active_env_idx]
    self.active.set_goal_idx(idx - alens[self.active_env_idx])
  
  def get_goals(self):
    return self.goals
  
  @property
  def active(self):
    return self.envs[self.active_env_idx]
  
  def __getattr__(self, name):
    return getattr(self.active, name)


class PadActions(object):
  """Pad action space to the largest action space."""
  
  def __init__(self, env, spaces):
    self._env = env
    self._action_space = self._pad_box_space(spaces)
  
  @property
  def action_space(self):
    return self._action_space
  
  def __getattr__(self, name):
    return getattr(self._env, name)
  
  def step(self, action, *args, **kwargs):
    action = action[:len(self._env.action_space.low)]
    return self._env.step(action, *args, **kwargs)
  
  def _pad_box_space(self, spaces):
    assert all(len(space.low.shape) == 1 for space in spaces)
    length = max(len(space.low) for space in spaces)
    low, high = np.inf * np.ones(length), -np.inf * np.ones(length)
    for space in spaces:
      low[:len(space.low)] = np.minimum(space.low, low[:len(space.low)])
      high[:len(space.high)] = np.maximum(space.high, high[:len(space.high)])
    return gym.spaces.Box(low, high, dtype=np.float32)
