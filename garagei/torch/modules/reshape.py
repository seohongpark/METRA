import numpy as np
import torch
from torch import nn

class ReshapeModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        assert np.prod(x.shape[1:]) == np.prod(self.shape)
        return x.reshape(-1, *self.shape)


class ViewModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        assert np.prod(x.shape[1:]) == np.prod(self.shape)
        return x.view(-1, *self.shape)


