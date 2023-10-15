import numpy as np

import torch
from torch import nn

from garagei.torch.modules.spectral_norm import spectral_norm


class NormLayer(nn.Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)


class CNN(nn.Module):
    def __init__(self, num_inputs, act=nn.ELU, norm='none', cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=(400, 400, 400, 400), spectral_normalization=False):
        super().__init__()

        self._num_inputs = num_inputs
        self._act = act()
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        self._conv_model = []
        for i, kernel in enumerate(self._cnn_kernels):
            if i == 0:
                prev_depth = num_inputs
            else:
                prev_depth = 2 ** (i - 1) * self._cnn_depth
            depth = 2 ** i * self._cnn_depth
            if spectral_normalization:
                self._conv_model.append(spectral_norm(nn.Conv2d(prev_depth, depth, kernel, stride=2)))
            else:
                self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
            self._conv_model.append(NormLayer(norm, depth))
            self._conv_model.append(self._act)
        self._conv_model = nn.Sequential(*self._conv_model)

    def forward(self, data):
        output = self._conv_model(data)
        output = output.reshape(output.shape[0], -1)
        return output


class Encoder(nn.Module):
    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
    ):
        super().__init__()

        self.pixel_shape = pixel_shape
        self.pixel_dim = np.prod(pixel_shape)

        self.pixel_depth = self.pixel_shape[-1]

        self.encoder = CNN(self.pixel_depth, spectral_normalization=spectral_normalization)

    def forward(self, input):
        assert len(input.shape) == 2

        pixel = input[..., :self.pixel_dim].reshape(-1, *self.pixel_shape).permute(0, 3, 1, 2)
        state = input[..., self.pixel_dim:]

        pixel = pixel / 255.

        rep = self.encoder(pixel)
        rep = rep.reshape(rep.shape[0], -1)
        output = torch.cat([rep, state], dim=-1)

        return output


class WithEncoder(nn.Module):
    def __init__(
            self,
            encoder,
            module,
    ):
        super().__init__()

        self.encoder = encoder
        self.module = module

    def get_rep(self, input):
        return self.encoder(input)

    def forward(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module(rep, *inputs[1:])

    def forward_mode(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module.forward_mode(rep, *inputs[1:])
