from collections import deque

import akro
import gym
import numpy as np
import matplotlib.pyplot as plt

from garagei.envs.akro_wrapper import AkroWrapperTrait


class RenderWrapper(AkroWrapperTrait, gym.Wrapper):
    def __init__(
            self,
            env,
    ):
        super().__init__(env)

        if env._domain == 'cheetah':
            l = len(env.physics.model.tex_type)
            for i in range(l):
                if env.physics.model.tex_type[i] == 0:
                    height = env.physics.model.tex_height[i]
                    width = env.physics.model.tex_width[i]
                    s = env.physics.model.tex_adr[i]
                    colors = []
                    for y in range(width):
                        scaled_y = np.clip((y / width - 0.5) * 4 + 0.5, 0, 1)
                        colors.append((np.array(plt.cm.rainbow(scaled_y))[:3] * 255).astype(np.uint8))
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            env.physics.model.tex_rgb[cur_s:cur_s + 3] = colors[y]
            env.physics.model.mat_texrepeat[:, :] = 1
        else:
            l = len(env.physics.model.tex_type)
            for i in range(l):
                if env.physics.model.tex_type[i] == 0:
                    height = env.physics.model.tex_height[i]
                    width = env.physics.model.tex_width[i]
                    s = env.physics.model.tex_adr[i]
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            env.physics.model.tex_rgb[cur_s:cur_s + 3] = [int(x / height * 255), int(y / width * 255), 128]
            env.physics.model.mat_texrepeat[:, :] = 1

        self.action_space = self.env.action_space
        self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(64, 64, 3))

        self.ob_info = dict(
            type='pixel',
            pixel_shape=(64, 64, 3),
        )

    def _transform(self, obs):
        pixels = self.env.render(mode='rgb_array', width=64, height=64).copy()
        pixels = pixels.flatten()
        return pixels

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._transform(obs)

    def step(self, action, **kwargs):
        next_obs, reward, done, info = self.env.step(action, **kwargs)
        return self._transform(next_obs), reward, done, info


class FrameStackWrapper(AkroWrapperTrait, gym.Wrapper):
    def __init__(
            self,
            env,
            num_frames
    ):
        super().__init__(env)

        self.num_frames = num_frames
        self.frames = deque([], maxlen=self.num_frames)

        self.ori_pixel_shape = self.env.ob_info['pixel_shape']
        self.ori_flat_pixel_shape = np.prod(self.ori_pixel_shape)
        self.new_pixel_shape = (self.ori_pixel_shape[0], self.ori_pixel_shape[1], self.ori_pixel_shape[2] * self.num_frames)

        self.action_space = self.env.action_space

        if env.ob_info['type'] == 'pixel':
            self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=self.new_pixel_shape)
            self.ob_info = dict(
                type='pixel',
                pixel_shape=self.new_pixel_shape,
            )
        elif env.ob_info['type'] == 'hybrid':
            self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(np.prod(self.new_pixel_shape) + np.prod(env.ob_info['state_shape']),))
            self.ob_info = dict(
                type='hybrid',
                pixel_shape=self.new_pixel_shape,
                state_shape=env.ob_info['state_shape'],
            )
        else:
            raise NotImplementedError

    def _transform_observation(self, cur_obs):
        assert len(self.frames) == self.num_frames
        obs = np.concatenate(list(self.frames), axis=2)
        return np.concatenate([obs.flatten(), cur_obs[self.ori_flat_pixel_shape:]], axis=-1)

    def _extract_pixels(self, obs):
        pixels = obs[:self.ori_flat_pixel_shape].reshape(self.ori_pixel_shape)
        return pixels

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        pixels = self._extract_pixels(obs)
        for _ in range(self.num_frames):
            self.frames.append(pixels)
        return self._transform_observation(obs)

    def step(self, action, **kwargs):
        next_obs, reward, done, info = self.env.step(action, **kwargs)
        pixels = self._extract_pixels(next_obs)
        self.frames.append(pixels)
        return self._transform_observation(next_obs), reward, done, info
