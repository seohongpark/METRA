from envs.custom_dmc_tasks import cheetah
from envs.custom_dmc_tasks import quadruped
from envs.custom_dmc_tasks import humanoid


def make(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    if domain == 'cheetah':
        return cheetah.make(task,
                            task_kwargs=task_kwargs,
                            environment_kwargs=environment_kwargs,
                            visualize_reward=visualize_reward)
    elif domain == 'quadruped':
        return quadruped.make(task,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)
    elif domain == 'humanoid':
        return humanoid.make(task,
                             task_kwargs=task_kwargs,
                             environment_kwargs=environment_kwargs,
                             visualize_reward=visualize_reward)
    else:
        raise NotImplementedError
