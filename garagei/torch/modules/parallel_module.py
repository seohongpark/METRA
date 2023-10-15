import numpy as np
import torch
from torch import nn

class ParallelModule(nn.Module):
    def __init__(self,
                 input_dims,
                 parallel_modules,
                 post_parallel_module,
                 **kwargs):
        super().__init__()

        self._input_dims = input_dims
        self._parallel_modules = nn.ModuleList(parallel_modules)
        print(parallel_modules)
        self._post_parallel_module = post_parallel_module
        self._split_dim = -1
        assert len(self._input_dims) == len(self._parallel_modules)

    def _get_input_dim_cumsum(self):
        return np.cumsum([0] + self._input_dims[:-1])

    def _forward_parallel(self, *inputs):
        split_inputs = list(zip(*[
            torch.split(i, self._input_dims, dim=self._split_dim)
            for i in inputs
        ]))
        split_outputs = [
            m(*si)
            for si, m in zip(split_inputs, self._parallel_modules)
        ]
        return torch.cat(split_outputs, dim=-1)

    def forward(self, *inputs):
        out = self._forward_parallel(*inputs)
        if self._post_parallel_module is not None:
            out = self._post_parallel_module(out)
        return out

    def forward_mode(self, *inputs):
        out = self._forward_parallel(*inputs)
        if self._post_parallel_module is not None:
            out = self._post_parallel_module.forward_mode(out)
        return out

    def forward_with_transform(self, *inputs, transform):
        out = self._forward_parallel(*inputs)
        if self._post_parallel_module is not None:
            out = self._post_parallel_module.forward_with_transform(out, transform=transform)
        return out

    def forward_with_chunks(self, *inputs, merge):
        out = []
        for chunk_inputs in zip(*inputs):
            out.append(self._forward_parallel(*chunk_inputs))
        if self._post_parallel_module is not None:
            out = self._post_parallel_module.forward_with_chunks(out, merge=merge)
        return out

