import abc

import torch
from torch import nn
from torch.distributions import Categorical, OneHotCategorical
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

from garagei.torch.distributions.transformed_distribution_ex import TransformedDistributionEx


class CategoricalMLPModuleEx(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 categorical_distribution_cls=Categorical,
                 distribution_transformations=None):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init

        self._layer_normalization = layer_normalization
        self._categorical_dist_class = categorical_distribution_cls
        self._distribution_transformations = distribution_transformations

        self._logits_module = MLPModule(
                input_dim=self._input_dim,
                output_dim=self._output_dim,
                hidden_sizes=self._hidden_sizes,
                hidden_nonlinearity=self._hidden_nonlinearity,
                hidden_w_init=self._hidden_w_init,
                hidden_b_init=self._hidden_b_init,
                output_nonlinearity=self._output_nonlinearity,
                output_w_init=self._output_w_init,
                output_b_init=self._output_b_init,
                layer_normalization=self._layer_normalization)


    def _maybe_move_distribution_transformations(self):
        device = next(self.parameters()).device
        if self._distribution_transformations is not None:
            self._distribution_transformations = [
                t.maybe_clone_to_device(device)
                for t in self._distribution_transformations
            ]
    # Parent module's .to(), .cpu(), and .cuda() call children's ._apply().
    def _apply(self, *args, **kwargs):
        ret = super()._apply(*args, **kwargs)
        self._maybe_move_distribution_transformations()
        return ret

    @abc.abstractmethod
    def _get_logits(self, *inputs):
        return self._logits_module(*inputs)

    def forward(self, *inputs):
        logits = self._get_logits(*inputs)

        dist = self._categorical_dist_class(logits=logits)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, (TanhNormal, OneHotCategorical)):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist

    def forward_mode(self, *inputs):
        logits = self._get_logits(*inputs)

        dist = self._categorical_dist_class(logits=logits)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                dist,
                self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, (TanhNormal, OneHotCategorical)):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist.mode

    def forward_with_transform(self, *inputs, transform):
        logits = self._get_logits(*inputs)

        dist = self._categorical_dist_class(logits=logits)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, (TanhNormal, OneHotCategorical)):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        logits = transform(logits)

        dist_transformed = self._categorical_dist_class(logits=logits)
        if self._distribution_transformations is not None:
            dist_transformed = TransformedDistributionEx(
                    dist_transformed,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist_transformed, (TanhNormal, OneHotCategorical)):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist_transformed.batch_shape samples.
            dist_transformed = Independent(dist_transformed, 1)

        return dist, dist_transformed

    def forward_with_chunks(self, *inputs, merge):
        logits = []
        for chunk_inputs in zip(*inputs):
            chunk_logits = self._get_logits(*chunk_inputs)
            logits.append(chunk_logits)
        logits = merge(logits, batch_dim=0)

        dist = self._categorical_dist_class(logits=logits)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, (TanhNormal, OneHotCategorical)):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


