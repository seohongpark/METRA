import copy
import pathlib
import time
import os

import dowel_wrapper
import numpy as np
import torch
import platform
if 'macOS' in platform.platform():
    os.environ["IMAGEIO_FFMPEG_EXE"] = '/opt/homebrew/bin/ffmpeg'
from moviepy import editor as mpy
from garage.misc.tensor_utils import discount_cumsum
from matplotlib import figure
from matplotlib.patches import Ellipse
from sklearn import decomposition


def to_np_object_arr(x):
    arr = np.empty(len(x), dtype=object)
    for i, t in enumerate(x):
        arr[i] = t
    return arr


def get_torch_concat_obs(obs, option, dim=1):
    concat_obs = torch.cat([obs] + [option], dim=dim)
    return concat_obs


def get_np_concat_obs(obs, option):
    concat_obs = np.concatenate([obs] + [option])
    return concat_obs


def get_normalizer_preset(normalizer_type):
    # Precomputed mean and std of the state dimensions from 10000 length-50 random rollouts (without early termination)
    if normalizer_type == 'off':
        normalizer_mean = np.array([0.])
        normalizer_std = np.array([1.])
    elif normalizer_type == 'half_cheetah_preset':
        normalizer_mean = np.array(
            [-0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337, 0.01661406, -0.05476654,
             -0.04932635, -0.08061652, -0.05205841, 0.04500197, 0.02638421, -0.04570961, 0.03183838, 0.01736591,
             0.0091929, -0.0115027])
        normalizer_std = np.array(
            [0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397, 0.2896734, 0.2774097,
             0.73060024, 0.77360505, 1.5871304, 5.5405455, 6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197])
    elif normalizer_type == 'ant_preset':
        normalizer_mean = np.array(
            [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
             0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
             0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
             -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
        normalizer_std = np.array(
            [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
             0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
             0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
             4.3704205, 4.5175962, 4.3704395])
    else:
        raise NotImplementedError

    return normalizer_mean, normalizer_std


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def get_option_colors(options, color_range=4):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
        option_colors = get_2d_colors(options, (-color_range, -color_range), (color_range, color_range))
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors


def draw_2d_gaussians(means, stddevs, colors, ax, fill=False, alpha=0.8, use_adaptive_axis=False, draw_unit_gaussian=True, plot_axis=None):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2,
                          edgecolor='r', lw=1, facecolor='none', alpha=0.5)
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(xy=mean, width=stddev[0] * 2, height=stddev[1] * 2,
                          edgecolor=color, lw=1, facecolor='none' if not fill else color, alpha=alpha)
        ax.add_patch(ellipse)
        square_axis_limit = max(
                square_axis_limit,
                np.abs(mean[0] + stddev[0]),
                np.abs(mean[0] - stddev[0]),
                np.abs(mean[1] + stddev[1]),
                np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis('scaled')
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(runner, label, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = (pathlib.Path(runner._snapshotter.snapshot_dir)
                 / 'plots'
                 # / f'{label}_{runner.step_itr}.gif')
                 / f'{label}_{runner.step_itr}.mp4')
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)
    if 'WANDB_API_KEY' in os.environ:
        import wandb
        wandb.log({label: wandb.Video(str(plot_path))}, step=runner.step_itr)


def record_video(runner, label, trajectories, n_cols=None, skip_frames=1):
    renders = []
    for trajectory in trajectories:
        render = trajectory['env_infos']['render']
        if render.ndim >= 5:
            render = render.reshape(-1, *render.shape[-3:])
        elif render.ndim == 1:
            render = np.concatenate(render, axis=0)
        renders.append(render)
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    save_video(runner, label, renders, n_cols=n_cols)


class FigManager:
    def __init__(self, runner, label, extensions=None, subplot_spec=None):
        self.runner = runner
        self.label = label
        self.fig = figure.Figure()
        if subplot_spec is not None:
            self.ax = self.fig.subplots(*subplot_spec).flatten()
        else:
            self.ax = self.fig.add_subplot()

        if extensions is None:
            self.extensions = ['png']
        else:
            self.extensions = extensions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_paths = [(pathlib.Path(self.runner._snapshotter.snapshot_dir)
                       / 'plots'
                       / f'{self.label}_{self.runner.step_itr}.{extension}') for extension in self.extensions]
        plot_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            self.fig.savefig(plot_path, dpi=300)
        dowel_wrapper.get_tabular('plot').record(self.label, self.fig)


class MeasureAndAccTime:
    def __init__(self, target):
        assert isinstance(target, list)
        assert len(target) == 1
        self._target = target

    def __enter__(self):
        self._time_enter = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += (time.time() - self._time_enter)


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self, msg='', *args, **kwargs):
        print(f'{msg}: {time.time() - self.t:.20f}')
        self.t = time.time()


def valuewise_sequencify_dicts(dicts):
    result = dict((k, []) for k in dicts[0].keys())
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return result


def zip_dict(d):
    keys = list(d.keys())
    values = [d[k] for k in keys]
    for z in zip(*values):
        yield dict((k, v) for k, v in zip(keys, z))


def split_paths(paths, chunking_points):
    assert 0 in chunking_points
    assert len(chunking_points) >= 2
    if len(chunking_points) == 2:
        return

    orig_paths = copy.copy(paths)
    paths.clear()
    for path in orig_paths:
        ei = path
        for s, e in zip(chunking_points[:-1], chunking_points[1:]):
            assert len(set(
                len(v)
                for k, v in path.items()
                if k not in ['env_infos', 'agent_infos']
            )) == 1
            new_path = {
                k: v[s:e]
                for k, v in path.items()
                if k not in ['env_infos', 'agent_infos']
            }
            new_path['dones'][-1] = True

            assert len(set(
                len(v)
                for k, v in path['env_infos'].items()
            )) == 1
            new_path['env_infos'] = {
                k: v[s:e]
                for k, v in path['env_infos'].items()
            }

            assert len(set(
                len(v)
                for k, v in path['agent_infos'].items()
            )) == 1
            new_path['agent_infos'] = {
                k: v[s:e]
                for k, v in path['agent_infos'].items()
            }

            paths.append(new_path)


def compute_traj_batch_performance(batch, discount):
    returns = []
    undiscounted_returns = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))

    return dict(
        undiscounted_returns=undiscounted_returns,
        discounted_returns=[rtn[0] for rtn in returns],
    )
