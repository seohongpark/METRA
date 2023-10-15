"""This modules creates a continuous Q-function network."""

import torch

from garage.torch.modules import MLPModule


class ContinuousMLPQFunctionEx(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, obs_dim, action_dim, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        MLPModule.__init__(self,
                           input_dim=self.obs_dim + self.action_dim,
                           output_dim=1,
                           **kwargs)

    def forward(self, observations, actions):
        """Return Q-value(s)."""
        return super().forward(torch.cat([observations, actions], 1))
