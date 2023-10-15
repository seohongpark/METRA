import numpy as np


class DictBatchDataset:
    """Use when the input is the dict type."""
    def __init__(self, inputs, batch_size):
        self._inputs = inputs
        self._batch_size = batch_size
        self._size = list(self._inputs.values())[0].shape[0]
        if batch_size is not None:
            self._ids = np.arange(self._size)
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._size * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield self._inputs
        else:
            if update:
                self.update()
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = {
                    k: v[batch_ids]
                    for k, v in self._inputs.items()
                }
                yield batch

    def update(self):
        np.random.shuffle(self._ids)
