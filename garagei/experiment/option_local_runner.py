import atexit
import copy
import os
import pickle
import signal
import time

import torch

from dowel import logger, tabular, TextOutput, CsvOutput, StdOutput
import numpy as np
import psutil
from garage.experiment import LocalRunner
from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.local_runner import SetupArgs, NotSetupError
from garage.sampler import WorkerFactory
from garage.sampler.sampler_deprecated import BaseSampler

import global_context
import dowel_wrapper
from garagei.sampler.option_local_sampler import OptionLocalSampler
from garagei.sampler.option_worker import OptionWorker


class OptionLocalRunner(LocalRunner):
    def setup(self,
              algo,
              env,
              make_env,
              sampler_cls=None,
              sampler_args=None,
              n_workers=psutil.cpu_count(logical=False),
              worker_class=None,
              worker_args=None,
              ):
        self._algo = algo
        self._env = env
        self._make_env = make_env
        self._n_workers = {}
        self._worker_class = worker_class
        if sampler_args is None:
            sampler_args = {}
        if sampler_cls is None:
            sampler_cls = getattr(algo, 'sampler_cls', None)
        if worker_class is None:
            worker_class = getattr(algo, 'worker_cls', OptionWorker)
        if worker_args is None:
            worker_args = {}

        self._worker_args = worker_args
        if sampler_cls is None:
            self._sampler = None
        else:
            self._sampler = {}
            for key, policy in self._algo.policy.items():
                sampler_key = key
                cur_worker_args = dict(worker_args, sampler_key=sampler_key)
                self._sampler[sampler_key] = self.make_sampler(
                    sampler_cls,
                    sampler_args=sampler_args,
                    n_workers=n_workers,
                    worker_class=worker_class,
                    worker_args=cur_worker_args,
                    policy=policy
                )

                sampler_key = f'local_{key}'
                cur_worker_args = dict(worker_args, sampler_key=sampler_key)
                self._n_workers[key] = n_workers
                self._sampler[sampler_key] = self.make_local_sampler(
                    policy=policy,
                    worker_args=cur_worker_args,
                )
                self._n_workers[sampler_key] = 1
            self.sampler_keys = list(self._sampler.keys())
        self._has_setup = True

        self._setup_args = SetupArgs(sampler_cls=sampler_cls,
                                     sampler_args=sampler_args,
                                     seed=get_seed())

        self._hanging_env_update = {}
        self._hanging_worker_update = {}
        for key in self.sampler_keys:
            self._hanging_env_update[key] = None
            self._hanging_worker_update[key] = None

    def save(self, epoch, new_save=False, pt_save=False, pkl_update=False):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the runner is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup runner before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['setup_args'] = self._setup_args
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        replay_buffer = self._algo.replay_buffer
        self._algo.replay_buffer = None  # Don't save replay buffer
        # params['env'] = self._env
        params['algo'] = self._algo
        params['n_workers'] = self._n_workers
        params['worker_class'] = self._worker_class
        params['worker_args'] = self._worker_args

        if new_save and epoch != 0:
            prev_snapshot_mode = self._snapshotter._snapshot_mode
            self._snapshotter._snapshot_mode = 'all'
            self._snapshotter.save_itr_params(epoch, params)
            self._snapshotter._snapshot_mode = prev_snapshot_mode
            file_name = os.path.join(self._snapshotter._snapshot_dir, f'option_policy{epoch}.pt')
            torch.save({
                'discrete': self._algo.discrete,
                'dim_option': self._algo.dim_option,
                'policy': self._algo.option_policy,
            }, file_name)
            file_name = os.path.join(self._snapshotter._snapshot_dir, f'traj_encoder{epoch}.pt')
            torch.save({
                'discrete': self._algo.discrete,
                'dim_option': self._algo.dim_option,
                'traj_encoder': self._algo.traj_encoder,
            }, file_name)

        if pt_save and epoch != 0:
            file_name = os.path.join(self._snapshotter._snapshot_dir, f'option_policy{epoch}.pt')
            torch.save({
                'discrete': self._algo.discrete,
                'dim_option': self._algo.dim_option,
                'policy': self._algo.option_policy,
            }, file_name)

        if pkl_update:
            self._snapshotter.save_itr_params(epoch, params)

        self._algo.replay_buffer = replay_buffer

        logger.log('Saved')

    def restore(self, from_dir, make_env, from_epoch='last', post_restore_handler=None):
        """Restore experiment from snapshot.

        Args:
            from_dir (str): Directory of the pickle file
                to resume experiment from.
            from_epoch (str or int): The epoch to restore from.
                Can be 'first', 'last' or a number.
                Not applicable when snapshot_mode='last'.

        Returns:
            TrainArgs: Arguments for train().

        """
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._setup_args = saved['setup_args']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._setup_args.seed)

        if post_restore_handler is not None:
            post_restore_handler(saved)

        self.setup(env=make_env(),  # Not use saved['env']
                   algo=saved['algo'],
                   make_env=make_env,
                   sampler_cls=self._setup_args.sampler_cls,
                   sampler_args=self._setup_args.sampler_args,
                   n_workers=saved['n_workers']['option_policy'],
                   )

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        batch_size = self._train_args.batch_size
        store_paths = self._train_args.store_paths
        pause_for_plot = self._train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_paths', store_paths))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))
        logger.log(fmt.format('-- Stats --', '-- Value --'))
        logger.log(fmt.format('last_itr', last_itr))
        logger.log(fmt.format('total_env_steps', total_env_steps))

        self._train_args.start_epoch = last_epoch
        return copy.copy(self._train_args)

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        for sampler in self._sampler.values():
            if isinstance(sampler, BaseSampler):
                sampler.start_worker()
        if self._plot:
            raise NotImplementedError()
        self._shutdown_worker_called = False
        atexit.register(self._shutdown_worker)
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._shutdown_worker_on_signal)

    def _shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        if self._shutdown_worker_called:
            return
        for sampler in self._sampler.values():
            if sampler is not None:
                sampler.shutdown_worker()
        if self._plot:
            raise NotImplementedError()
        self._shutdown_worker_called = True

    def _shutdown_worker_on_signal(self, signum, frame):
        self._shutdown_worker()

    def make_sampler(self,
                     sampler_cls,
                     *,
                     seed=None,
                     n_workers=psutil.cpu_count(logical=False),
                     max_path_length=None,
                     worker_class=OptionWorker,
                     sampler_args=None,
                     worker_args=None,
                     policy=None):
        if max_path_length is None:
            if hasattr(self._algo, 'max_path_length'):
                max_path_length = self._algo.max_path_length
            else:
                raise ValueError('If `sampler_cls` is specified in '
                                 'runner.setup, the algorithm must have '
                                 'a `max_path_length` field.')
        if seed is None:
            seed = get_seed()
        if sampler_args is None:
            sampler_args = {}
        if worker_args is None:
            worker_args = {}

        agents = policy

        if issubclass(sampler_cls, BaseSampler):
            raise NotImplementedError('BaseSampler does not support obtain_exact_trajectories()')
        else:
            return sampler_cls.from_worker_factory(WorkerFactory(
                seed=seed,
                max_path_length=max_path_length,
                n_workers=n_workers,
                worker_class=worker_class,
                worker_args=worker_args),
                agents=agents,
                make_env=self._make_env,
                **sampler_args)

    def make_local_sampler(self, policy, worker_args):
        max_path_length = self._algo.max_path_length
        seed = get_seed()

        agents = copy.deepcopy(policy)

        return OptionLocalSampler.from_worker_factory(WorkerFactory(
            seed=seed,
            max_path_length=max_path_length,
            n_workers=1,
            worker_class=OptionWorker,
            worker_args=worker_args),
            agents=agents,
            make_env=self._make_env)

    def set_hanging_env_update(self, env_update, sampler_keys):
        for k, v in env_update.items():
            setattr(self._env, k, v)
        for key in sampler_keys:
            self._hanging_env_update[key] = dict(env_update)

    def set_hanging_worker_update(self, worker_update, sampler_keys):
        for key in sampler_keys:
            self._hanging_worker_update[key] = dict(worker_update)

    def obtain_exact_trajectories(self,
                                  itr,
                                  sampler_key,
                                  batch_size=None,
                                  agent_update=None,
                                  env_update=None,
                                  worker_update=None,
                                  extras=None,
                                  max_path_length_override=None,
                                  get_attrs=None,
                                  update_normalized_env_ex=None,
                                  update_stats=True):
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError('Runner was not initialized with `batch_size`. '
                             'Either provide `batch_size` to runner.train, '
                             ' or pass `batch_size` to runner.obtain_samples.')
        sampler = self._sampler[sampler_key]
        if isinstance(sampler, BaseSampler):
            raise NotImplementedError('BaseSampler does not support obtain_exact_trajectories()')
        else:
            if agent_update is None:
                agent_update = self._algo.policy[sampler_key].get_param_values()

            if self._hanging_env_update[sampler_key] is not None and env_update is not None:
                if isinstance(self._hanging_env_update[sampler_key], dict) and isinstance(env_update, dict):
                    self._hanging_env_update[sampler_key].update(env_update)
                    env_update = None
                else:
                    raise NotImplementedError()
            if self._hanging_worker_update[sampler_key] is not None and worker_update is not None:
                if isinstance(self._hanging_worker_update[sampler_key], dict) and isinstance(worker_update, dict):
                    self._hanging_worker_update[sampler_key].update(worker_update)
                    worker_update = None
                else:
                    raise NotImplementedError()

            if self._hanging_env_update[sampler_key] is not None:
                env_update = self._hanging_env_update[sampler_key]
                self._hanging_env_update[sampler_key] = None
            if self._hanging_worker_update[sampler_key] is not None:
                worker_update = self._hanging_worker_update[sampler_key]
                self._hanging_worker_update[sampler_key] = None

            batch_size = (batch_size or self._train_args.batch_size)
            n_traj_per_workers = [
                batch_size // self._n_workers[sampler_key] + int(i < (batch_size % self._n_workers[sampler_key]))
                for i in range(self._n_workers[sampler_key])
            ]
            assert batch_size == sum(n_traj_per_workers)

            if env_update is None:
                env_update = {}
            if worker_update is None:
                worker_update = {}

            worker_update.update(dict(
                _max_path_length_override=max_path_length_override,
                _cur_extras=None,
                _cur_extra_idx=None,
            ))
            if extras is not None:
                assert batch_size == len(extras)

                worker_extras_list = np.array_split(extras, self._n_workers[sampler_key])
                worker_update = [
                    dict(
                        worker_update,
                        _cur_extras=worker_extras,
                        _cur_extra_idx=-1,
                    )
                    for worker_extras
                    in worker_extras_list
                ]

            if update_normalized_env_ex is not None:
                assert isinstance(env_update, dict)
                env_update.update(dict(
                    do_update=update_normalized_env_ex,
                ))

            paths, infos = sampler.obtain_exact_trajectories(
                n_traj_per_workers,
                agent_update=agent_update,
                env_update=env_update,
                worker_update=worker_update,
                get_attrs=get_attrs,
            )
            paths = paths.to_trajectory_list()

        if update_stats:
            # XXX: Assume that env_infos always contains 2D coordinates.
            self._stats.total_env_steps += sum([
                (len(p['env_infos']['coordinates'].reshape(-1, 2))
                 if p['env_infos']['coordinates'].dtype != object
                 else sum(len(l) for l in p['env_infos']['coordinates']))
                for p in paths
            ])

        return paths, infos

    def step_epochs(self, log_period=1, full_tb_epochs=None, tb_period=None, pt_save_period=None, pkl_update_period=None, new_save_period=None):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_path`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in runner.step_epochs():
                runner.step_path = runner.obtain_samples(...)
                self.train_once(...)
                runner.step_itr += 1

        """
        self._start_worker()
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_path = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                save_path = (self.step_path
                             if self._train_args.store_paths else None)

                self._stats.last_path = save_path
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                new_save = (new_save_period != 0 and self.step_itr % new_save_period == 0)
                pt_save = (pt_save_period != 0 and self.step_itr % pt_save_period == 0)
                pkl_update = (pkl_update_period != 0 and self.step_itr % pkl_update_period == 0)
                if new_save or pt_save or pkl_update:
                    self.save(epoch, new_save=new_save, pt_save=pt_save, pkl_update=pkl_update)

                yield epoch

                if self.enable_logging:
                    if self.step_itr % log_period == 0:
                        self.log_diagnostics(self._train_args.pause_for_plot)
                        if full_tb_epochs is None or tb_period is None:
                            logger.dump_all(self.step_itr)
                        else:
                            if self.step_itr <= full_tb_epochs or (tb_period != 0 and self.step_itr % tb_period == 0):
                                logger.dump_all(self.step_itr)
                            else:
                                logger.dump_output_type((TextOutput, CsvOutput, StdOutput), self.step_itr)

                    tabular.clear()


    def log_diagnostics(self, pause_for_plot=False):
        total_time = (time.time() - self._start_time)
        logger.log('Time %.2f s' % total_time)
        epoch_time = (time.time() - self._itr_start_time)
        logger.log('EpochTime %.2f s' % epoch_time)
        tabular.record('TotalEnvSteps', self._stats.total_env_steps)
        tabular.record('TotalEpoch', self._stats.total_epoch)
        tabular.record('TimeEpoch', epoch_time)
        tabular.record('TimeTotal', total_time)
        logger.log(tabular)

    def eval_log_diagnostics(self):
        if self.enable_logging:
            total_time = (time.time() - self._start_time)
            dowel_wrapper.get_tabular('eval').record('TotalEnvSteps', self._stats.total_env_steps)
            dowel_wrapper.get_tabular('eval').record('TotalEpoch', self._stats.total_epoch)
            dowel_wrapper.get_tabular('eval').record('TimeTotal', total_time)
            dowel_wrapper.get_logger('eval').log(dowel_wrapper.get_tabular('eval'))
            dowel_wrapper.get_logger('eval').dump_all(self.step_itr)
            dowel_wrapper.get_tabular('eval').clear()

    def plot_log_diagnostics(self):
        if self.enable_logging:
            dowel_wrapper.get_tabular('plot').record('TotalEnvSteps', self._stats.total_env_steps)
            dowel_wrapper.get_tabular('plot').record('TotalEpoch', self._stats.total_epoch)
            dowel_wrapper.get_logger('plot').log(dowel_wrapper.get_tabular('plot'))
            dowel_wrapper.get_logger('plot').dump_all(self.step_itr)
            dowel_wrapper.get_tabular('plot').clear()
