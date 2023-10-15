from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import envs.custom_dmc_tasks as cdmc
from envs.custom_dmc_tasks.wrappers import DMCGymWrapper


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats
        self.dt = num_repeats * self.physics.model.opt.timestep * self._n_sub_steps

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
    """Wraps a control environment to rescale actions to a specific range."""
    __slots__ = ("_action_spec", "_env")

    def __init__(self, env, minimum, maximum):
        """Initializes a new action scale Wrapper.

        Args:
          env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
            consist of a single `BoundedArray` with all-finite bounds.
          minimum: Scalar or array-like specifying element-wise lower bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.
          maximum: Scalar or array-like specifying element-wise upper bounds
            (inclusive) for the `action_spec` of the wrapped environment. Must be
            finite and broadcastable to the shape of the `action_spec`.

        Raises:
          ValueError: If `env.action_spec()` is not a single `BoundedArray`.
          ValueError: If `env.action_spec()` has non-finite bounds.
          ValueError: If `minimum` or `maximum` contain non-finite values.
          ValueError: If `minimum` or `maximum` are not broadcastable to
            `env.action_spec().shape`.
        """
        action_spec = env.action_spec()
        if not isinstance(action_spec, specs.BoundedArray):
            raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

        minimum = np.array(minimum)
        maximum = np.array(maximum)
        shape = action_spec.shape
        orig_minimum = action_spec.minimum
        orig_maximum = action_spec.maximum
        orig_dtype = action_spec.dtype


        self.validate(minimum, "minimum", shape)
        self.validate(maximum, "maximum", shape)
        self.validate(orig_minimum, "env.action_spec().minimum", shape)
        self.validate(orig_maximum, "env.action_spec().maximum", shape)

        scale = (orig_maximum - orig_minimum) / (maximum - minimum)

        self.orig_minimum = orig_minimum
        self.orig_maximum = orig_maximum
        self.orig_dtype = orig_dtype
        self.scale = scale
        self.minimum = minimum
        self.maximum = maximum

        dtype = np.result_type(minimum, maximum, orig_dtype)
        self._action_spec = action_spec.replace(
            minimum=minimum, maximum=maximum, dtype=dtype)
        self._env = env

    def validate(self, bounds, name, shape):
        if not np.all(np.isfinite(bounds)):
            raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
        try:
            np.broadcast_to(bounds, shape)
        except ValueError:
            raise ValueError(_MUST_BROADCAST.format(
                name=name, bounds=bounds, shape=shape))

    def transform(self, action):
        new_action = self.orig_minimum + self.scale * (action - self.minimum)
        return new_action.astype(self.orig_dtype, copy=False)

    def step(self, action):
        return self._env.step(self.transform(action))

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, task_kwargs=None):
    visualize_reward = False
    if task_kwargs is None:
        task_kwargs = {}
    task_kwargs['random'] = seed
    env = cdmc.make(domain,
                    task,
                    task_kwargs=task_kwargs,
                    environment_kwargs=dict(flat_observation=True),
                    visualize_reward=visualize_reward)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    return env


def make(name, obs_type, frame_stack, action_repeat, seed, task_kwargs=None):
    assert obs_type in ['states', 'pixels']
    domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, task_kwargs=task_kwargs)

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = ActionScaleWrapper(env, minimum=-1.0, maximum=1.0)
    env = ExtendedTimeStepWrapper(env)

    env = DMCGymWrapper(env, domain=domain)

    return env
