import akro
import gym
import numpy as np


class MazeEnv(gym.Env):
    def __init__(self, max_path_length, action_range=0.2):
        self.max_path_length = max_path_length
        self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = akro.Box(low=-action_range, high=action_range, shape=(2,))

    def reset(self):
        self._cur_step = 0
        self._state = np.zeros(2)
        return self._state

    def step(self, action):
        obsbefore = self._state
        self._cur_step += 1
        self._state = self._state + action
        obsafter = self._state
        done = self._cur_step >= self.max_path_length
        reward = obsafter[0] - obsbefore[0]
        return self._state, reward, done, {
            'coordinates': obsbefore,
            'next_coordinates': obsafter,
            'ori_obs': obsbefore,
            'next_ori_obs': obsafter,
        }

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        rmin, rmax = None, None
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

            if rmin is None or rmin > np.min(trajectory[:, :2]):
                rmin = np.min(trajectory[:, :2])
            if rmax is None or rmax < np.max(trajectory[:, :2]):
                rmax = np.max(trajectory[:, :2])

        if plot_axis == 'nowalls':
            rcenter = (rmax + rmin) / 2.0
            rmax = rcenter + (rmax - rcenter) * 1.2
            rmin = rcenter + (rmin - rcenter) * 1.2
            plot_axis = [rmin, rmax, rmin, rmax]

        if plot_axis is not None:
            ax.axis(plot_axis)
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))

        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return {}
