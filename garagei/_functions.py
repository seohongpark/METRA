import numpy as np

import global_context
import dowel_wrapper
from garage.misc.tensor_utils import discount_cumsum
from dowel import Histogram

def log_performance_ex(itr, batch, discount, additional_records=None, additional_prefix=''):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    if additional_records is None:
        additional_records = {}
    returns = []
    undiscounted_returns = []
    completion = []
    success = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))
        completion.append(float(trajectory.terminals.any()))
        if 'success' in trajectory.env_infos:
            success.append(float(trajectory.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    prefix_tabular = global_context.get_metric_prefix()
    with dowel_wrapper.get_tabular().prefix(prefix_tabular):
        def _record(key, val, pre=''):
            dowel_wrapper.get_tabular().record(
                    (pre + '/' if len(pre) > 0 else '') + key,
                    val)

        def _record_histogram(key, val):
            dowel_wrapper.get_tabular('plot').record(key, Histogram(val))

        _record('Iteration', itr)
        dowel_wrapper.get_tabular().record('Iteration', itr)
        _record('NumTrajs', len(returns))

        max_undiscounted_returns = np.max(undiscounted_returns)
        min_undiscounted_returns = np.min(undiscounted_returns)
        _record('AverageDiscountedReturn', average_discounted_return)
        _record('AverageReturn', np.mean(undiscounted_returns))
        _record('StdReturn', np.std(undiscounted_returns))
        _record('MaxReturn', max_undiscounted_returns)
        _record('MinReturn', min_undiscounted_returns)
        _record('DiffMaxMinReturn', max_undiscounted_returns - min_undiscounted_returns)
        _record('CompletionRate', np.mean(completion))
        if success:
            _record('SuccessRate', np.mean(success))

        for key, val in additional_records.items():
            is_scalar = True
            try:
                if len(val) > 1:
                    is_scalar = False
            except TypeError:
                pass
            if is_scalar:
                _record(key, val, pre=additional_prefix)
            else:
                _record_histogram(key, val)

    return dict(
        undiscounted_returns=undiscounted_returns,
        discounted_returns=[rtn[0] for rtn in returns],
    )

