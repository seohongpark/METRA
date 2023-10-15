import numpy as np
from dm_env import specs
from gym import core, spaces


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCGymWrapper(core.Env):
    def __init__(
            self,
            env,
            from_pixels=False,
            height=84,
            width=84,
            channels_first=True,
            domain='',
    ):
        self._env = env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        if domain == 'quadruped':
            self._camera_id = 2
        else:
            self._camera_id = 0
        self._channels_first = channels_first
        self._frame_skip = 1
        self._domain = domain

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                [self._env.observation_spec()],
                np.float64
            )

        self._state_space = _spec_to_box(
            [self._env.observation_spec()],
            np.float64
        )

        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = time_step.observation
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action, render=False):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}
        xyz_before = self.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        obsbefore = self.physics.get_state()

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        xyz_after = self.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()

        obs = self._get_obs(time_step)
        self.current_state = time_step.observation
        obsafter = self.physics.get_state()
        extra['discount'] = time_step.discount

        if render:
            extra['render'] = self.render(mode='rgb_array', width=64, height=64).transpose(2, 0, 1)

        if self._domain in ['cheetah']:
            extra['coordinates'] = np.array([xyz_before[0], 0.])
            extra['next_coordinates'] = np.array([xyz_after[0], 0.])
        elif self._domain in ['quadruped', 'humanoid']:
            extra['coordinates'] = np.array([xyz_before[0], xyz_before[1]])
            extra['next_coordinates'] = np.array([xyz_after[0], xyz_after[1]])
        extra['ori_obs'] = obsbefore
        extra['next_ori_obs'] = obsafter

        return obs, reward, done, extra

    def calc_eval_metrics(self, trajectories, is_option_trajectories=False):
        return dict()

    def compute_reward(self, ob, next_ob, action=None):
        xposbefore = ob[:, 0]
        xposafter = next_ob[:, 0]

        reward = (xposafter - xposbefore) / self.dt
        done = np.zeros_like(reward)

        return reward, done

    def reset(self):
        time_step = self._env.reset()
        self.current_state = time_step.observation
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def plot_trajectory(self, trajectory, color, ax):
        if self._domain in ['cheetah']:
            trajectory = trajectory.copy()
            # https://stackoverflow.com/a/20474765/2182622
            from matplotlib.collections import LineCollection
            linewidths = np.linspace(0.2, 1.2, len(trajectory))
            points = np.reshape(trajectory, (-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=linewidths, color=color)
            ax.add_collection(lc)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            coordinates_trajectories.append(np.concatenate([
                trajectory['env_infos']['coordinates'],
                [trajectory['env_infos']['next_coordinates'][-1]]
            ]))
        if self._domain in ['cheetah']:
            for i, traj in enumerate(coordinates_trajectories):
                traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = {}

        coord_dim = 2 if self._domain in ['quadruped', 'humanoid'] else 1

        coords = []
        for traj in trajectories:
            traj1 = traj['env_infos']['coordinates'][:, :coord_dim]
            traj2 = traj['env_infos']['next_coordinates'][-1:, :coord_dim]
            coords.append(traj1)
            coords.append(traj2)
        coords = np.concatenate(coords, axis=0)
        uniq_coords = np.unique(np.floor(coords), axis=0)
        eval_metrics.update({
            'MjNumTrajs': len(trajectories),
            'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
            'MjNumCoords': len(coords),
            'MjNumUniqueCoords': len(uniq_coords),
        })

        return eval_metrics
