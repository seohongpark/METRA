import numpy as np
import torch
from torch import nn


class Multiplier(nn.Module):
    def __init__(self,
                 multiplicand,
                 requires_grad=False,
                 ):
        super().__init__()

        self._multiplicand = nn.Parameter(multiplicand, requires_grad=requires_grad)

    def forward(self, x):
        return x * self._multiplicand

