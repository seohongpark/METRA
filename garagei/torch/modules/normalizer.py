import numpy as np
import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(
            self,
            shape,
            alpha=0.001,
            do_normalize=True,
    ):
        super().__init__()

        self.shape = shape
        self.alpha = alpha
        self.do_normalize = do_normalize

        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def update(self, data, override=False):
        if not self.do_normalize:
            return

        # Compute in numpy for performance.
        data = data.detach().cpu().numpy()
        if not override:
            running_mean = self.running_mean.detach().cpu().numpy()
            running_var = self.running_var.detach().cpu().numpy()
            for single_data in np.random.permutation(data):
                # https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
                delta = single_data - running_mean
                running_mean = running_mean + self.alpha * delta
                running_var = (1 - self.alpha) * (running_var + self.alpha * delta ** 2)
        else:
            running_mean = np.mean(data, axis=0)
            running_var = np.var(data, axis=0)
        self.running_mean = torch.from_numpy(running_mean)
        self.running_var = torch.from_numpy(running_var)

    @property
    def mean(self):
        return self.running_mean.detach().cpu().numpy()

    @property
    def var(self):
        return self.running_var.detach().cpu().numpy()

    @property
    def std(self):
        return self.var ** 0.5

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def do_scale(self, x):
        return x / self.std

