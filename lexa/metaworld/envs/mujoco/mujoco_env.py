import abc

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym
from . import module

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        # env = args[0]
        # if not env._set_task_called:
        #     raise RuntimeError(
        #         'You must call env.set_task before using env.'
        #         + func.__name__
        #     )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """
    
    max_path_length = 150

    def __init__(self, model_path, frame_skip, rgb_array_res=(640, 480), use_dm_backend = True):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.use_dm_backend = use_dm_backend
        if self.use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_path.endswith(".mjb"):
                self.sim = dm_mujoco.Physics.from_binary_path(model_path)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_path)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)
        else:  
            self.model = mujoco_py.load_model_from_path(model_path)
            self.sim = mujoco_py.MjSim(self.model)
        
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self._rgb_array_res = rgb_array_res

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        if self.use_dm_backend:
            state = np.concatenate([qpos, qvel])
            self.sim.set_state(state)
            self.sim.forward()
        else:
            old_state = self.sim.get_state()
            new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                            old_state.act, old_state.udd_state)
            self.sim.set_state(new_state)
            self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human'):
        if mode == 'human':
            self._get_viewer(mode).render()
        elif mode == 'rgb_array':
            return self.sim.render(
                *self._rgb_array_res,
                mode='offscreen',
                camera_name='topview'
            )[:, :, ::-1]
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
    
    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self.use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()
    
    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
        assert self.use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), name.encode()
            )
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
            return obj_id

        def id2name(type_name, id):
            obj_name = mjlib.mj_id2name(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), id
            )
            return obj_name

        if not hasattr(model, "body_name2id"):
            model.body_name2id = lambda name: name2id("body", name)

        if not hasattr(model, "geom_name2id"):
            model.geom_name2id = lambda name: name2id("geom", name)

        if not hasattr(model, "geom_id2name"):
            model.geom_id2name = lambda id: id2name("geom", id)

        if not hasattr(model, "site_name2id"):
            model.site_name2id = lambda name: name2id("site", name)

        if not hasattr(model, "joint_name2id"):
            model.joint_name2id = lambda name: name2id("joint", name)

        if not hasattr(model, "actuator_name2id"):
            model.actuator_name2id = lambda name: name2id("actuator", name)

        if not hasattr(model, "camera_name2id"):
            model.camera_name2id = lambda name: name2id("camera", name)

        if not hasattr(model, "sensor_name2id"):
            model.sensor_name2id = lambda name: name2id("sensor", name)

        if not hasattr(data, "body_xpos"):
            data.body_xpos = data.xpos

        if not hasattr(data, "body_xquat"):
            data.body_xquat = data.xquat

        if not hasattr(data, "get_body_xpos"):
            data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

        if not hasattr(data, "get_body_xquat"):
            data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

        if not hasattr(data, "get_body_xmat"):
            data.get_body_xmat = lambda name: data.xmat[
                model.body_name2id(name)
            ].reshape(3, 3)

        if not hasattr(data, "get_geom_xpos"):
            data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

        if not hasattr(data, "get_geom_xquat"):
            data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

        if not hasattr(data, "get_joint_qpos"):
            data.get_joint_qpos = lambda name: data.qpos[model.joint_name2id(name)]

        if not hasattr(data, "set_joint_qpos"):

            def set_joint_qpos(name, value):
                data.qpos[
                    model.joint_name2id(name) : model.joint_name2id(name)
                    + value.shape[0]
                ] = value

            data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

        if not hasattr(data, "get_site_xmat"):
            data.get_site_xmat = lambda name: data.site_xmat[
                model.site_name2id(name)
            ].reshape(3, 3)

        if not hasattr(model, "get_joint_qpos_addr"):
            model.get_joint_qpos_addr = lambda name: model.joint_name2id(name)

        if not hasattr(model, "get_joint_qvel_addr"):
            model.get_joint_qvel_addr = lambda name: model.joint_name2id(name)

        if not hasattr(data, "get_geom_xmat"):
            data.get_geom_xmat = lambda name: data.geom_xmat[
                model.geom_name2id(name)
            ].reshape(3, 3)

        if not hasattr(data, "get_mocap_pos"):
            data.get_mocap_pos = lambda name: data.mocap_pos[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "get_mocap_quat"):
            data.get_mocap_quat = lambda name: data.mocap_quat[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "set_mocap_pos"):

            def set_mocap_pos(name, value):
                data.mocap_pos[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_pos = lambda name, value: set_mocap_pos(name, value)

        if not hasattr(data, "set_mocap_quat"):

            def set_mocap_quat(name, value):
                data.mocap_quat[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_quat = lambda name, value: set_mocap_quat(name, value)

        def site_jacp():
            jacps = np.zeros((model.nsite, 3 * model.nv))
            for i, jacp in enumerate(jacps):
                jacp_view = jacp.reshape(3, -1)
                mjlib.mj_jacSite(model.ptr, data.ptr, jacp_view, None, i)
            return jacps

        def site_xvelp():
            jacp = site_jacp().reshape((model.nsite, 3, model.nv))
            xvelp = np.dot(jacp, data.qvel)
            return xvelp

        def site_jacr():
            jacrs = np.zeros((model.nsite, 3 * model.nv))
            for i, jacr in enumerate(jacrs):
                jacr_view = jacr.reshape(3, -1)
                mjlib.mj_jacSite(model.ptr, data.ptr, None, jacr_view, i)
            return jacrs

        def site_xvelr():
            jacr = site_jacr().reshape((model.nsite, 3, model.nv))
            xvelr = np.dot(jacr, data.qvel)
            return xvelr

        if not hasattr(data, "site_xvelp"):
            data.site_xvelp = site_xvelp()

        if not hasattr(data, "site_xvelr"):
            data.site_xvelr = site_xvelr()

        if not hasattr(data, "get_site_jacp"):
            data.get_site_jacp = lambda name: site_jacp()[
                model.site_name2id(name)
            ].reshape(3, model.nv)

        if not hasattr(data, "get_site_jacr"):
            data.get_site_jacr = lambda name: site_jacr()[
                model.site_name2id(name)
            ].reshape(3, model.nv)

