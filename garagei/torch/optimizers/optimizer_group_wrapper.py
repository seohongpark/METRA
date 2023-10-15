"""A PyTorch optimizer wrapper that compute loss and optimize module."""
from garagei.np.optimizers.dict_minibatch_dataset import DictBatchDataset


class OptimizerGroupWrapper:
    """A wrapper class to handle torch.optim.optimizer.
    """

    def __init__(self,
                 optimizers,
                 max_optimization_epochs=1,
                 minibatch_size=None):
        self._optimizers = optimizers
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def get_minibatch(self, data, max_optimization_epochs=None):
        batch_dataset = DictBatchDataset(data, self._minibatch_size)

        if max_optimization_epochs is None:
            max_optimization_epochs = self._max_optimization_epochs

        for _ in range(max_optimization_epochs):
            for dataset in batch_dataset.iterate():
                yield dataset

    def zero_grad(self, keys=None):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""

        # TODO: optimize to param = None style.
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].zero_grad()

    def step(self, keys=None, **closure):
        """Performs a single optimization step.

        Arguments:
            **closure (callable, optional): A closure that reevaluates the
                model and returns the loss.

        """
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            self._optimizers[key].step(**closure)

    def target_parameters(self, keys=None):
        if keys is None:
            keys = self._optimizers.keys()
        for key in keys:
            for pg in self._optimizers[key].param_groups:
                for p in pg['params']:
                    yield p
