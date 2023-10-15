import torch
from garage.torch.distributions import TanhNormal
from garage.torch.modules import MultiHeadedMLPModule
from garage.torch.modules.gaussian_mlp_module import GaussianMLPModule, GaussianMLPIndependentStdModule, \
    GaussianMLPTwoHeadedModule, GaussianMLPBaseModule
from garage.torch.modules.mlp_module import MLPModule
from torch import nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.distributions.independent import Independent


class ForwardWithTransformTrait(object):
    def forward_with_transform(self, *inputs, transform):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        mean = transform(mean)
        std = transform(std)

        dist_transformed = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist_transformed, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist_transformed.batch_shape samples.
            dist_transformed = Independent(dist_transformed, 1)

        return dist, dist_transformed

class ForwardWithChunksTrait(object):
    def forward_with_chunks(self, *inputs, merge):
        mean = []
        log_std_uncentered = []
        for chunk_inputs in zip(*inputs):
            chunk_mean, chunk_log_std_uncentered = self._get_mean_and_log_std(*chunk_inputs)
            mean.append(chunk_mean)
            log_std_uncentered.append(chunk_log_std_uncentered)
        mean = merge(mean, batch_dim=0)
        log_std_uncentered = merge(log_std_uncentered, batch_dim=0)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist

class ForwardModeTrait(object):
    def forward_mode(self, *inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist.mean



class GaussianMLPModuleEx(GaussianMLPModule, ForwardWithTransformTrait, ForwardWithChunksTrait, ForwardModeTrait):
    pass
class GaussianMLPIndependentStdModuleEx(GaussianMLPIndependentStdModule, ForwardWithTransformTrait, ForwardWithChunksTrait, ForwardModeTrait):
    pass
class GaussianMLPTwoHeadedModuleEx(GaussianMLPTwoHeadedModule, ForwardWithTransformTrait, ForwardWithChunksTrait, ForwardModeTrait):
    pass


class GaussianMixtureMLPModule(GaussianMLPBaseModule):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_components,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal,
                 **kwargs):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         learn_std=learn_std,
                         init_std=init_std,
                         min_std=min_std,
                         max_std=max_std,
                         std_parameterization=std_parameterization,
                         layer_normalization=layer_normalization,
                         normal_distribution_cls=normal_distribution_cls)

        self._mean_module = MultiHeadedMLPModule(
            n_heads=num_components + 1,
            input_dim=self._input_dim,
            output_dims=[self._action_dim] * num_components + [num_components],
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=self._output_b_init,
            layer_normalization=self._layer_normalization,
            **kwargs,
        )

    def forward(self, *inputs):
        assert len(inputs) == 1
        *means, logits = self._mean_module(*inputs)

        broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
        log_std_uncentered = torch.zeros(*broadcast_shape, device=self._init_std.device) + self._init_std

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        categorical_dist = Categorical(logits=logits)
        mean = torch.stack(means, dim=1)
        std = torch.unsqueeze(std, dim=1)
        std = std.expand(std.size(0), mean.size(1), std.size(2))
        assert self._norm_dist_class == Normal
        norm_dist = Independent(self._norm_dist_class(mean, std), 1)

        dist = MixtureSameFamily(categorical_dist, norm_dist)

        return dist
