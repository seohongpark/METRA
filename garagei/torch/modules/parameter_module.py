import torch
from torch import nn


class ParameterModule(nn.Module):
    def __init__(
            self,
            init_value
    ):
        super().__init__()

        self.param = torch.nn.Parameter(init_value)
