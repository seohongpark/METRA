from math import inf
import math
import numpy as np
import torch
from torch.distributions.transforms import AffineTransform
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

from garagei.torch.distributions.transformed_distribution_ex import TransformedDistributionEx
from garagei.torch.distributions.transforms_ex import AffineTransformEx

def unsqueeze_expand_flat_dim0(x, num):
    return x.unsqueeze(dim=0).expand(num, *((-1,) * x.ndim)).reshape(
            num * x.size(0), *x.size()[1:])

def _get_transform_summary(transform):
    if isinstance(transform, AffineTransform):
        return f'{type(transform).__name__}({transform.loc}, {transform.scale})'
    raise NotImplementedError

def wrap_dist_with_transforms(base_dist_cls, transforms):
    def _create(*args, **kwargs):
        return TransformedDistributionEx(base_dist_cls(*args, **kwargs),
                                         transforms)
    _create.__name__ = (f'{base_dist_cls.__name__}['
                        + ', '.join(_get_transform_summary(t) for t in transforms) + ']')
    return _create

def unwrap_dist(dist):
    while hasattr(dist, 'base_dist'):
        dist = dist.base_dist
    return dist

def get_outermost_dist_attr(dist, attr):
    while (not hasattr(dist, attr)) and hasattr(dist, 'base_dist'):
        dist = dist.base_dist
    return getattr(dist, attr, None)

def get_affine_transform_for_beta_dist(target_min, target_max):
    # https://stackoverflow.com/a/12569453/2182622
    if isinstance(target_min, (np.ndarray, np.generic)):
        assert np.all(target_min <= target_max)
    else:
        assert target_min <= target_max
    #return AffineTransform(loc=torch.Tensor(target_min),
    #                       scale=torch.Tensor(target_max - target_min))
    return AffineTransformEx(loc=torch.tensor(target_min),
                             scale=torch.tensor(target_max - target_min))

def compute_total_norm(parameters, norm_type=2):
    # Code adopted from clip_grad_norm_().
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class TrainContext:
    def __init__(self, modules):
        self.modules = modules

    def __enter__(self):
        for m in self.modules:
            m.train()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.modules:
            m.eval()

def xavier_normal_ex(tensor, gain=1., multiplier=0.1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std * multiplier)

def kaiming_uniform_ex_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', gain=None):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = gain or torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

