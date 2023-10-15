import torch
from torch.distributions import Beta, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform

class TransformedDistributionEx(TransformedDistribution):
    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        ent = self.base_dist.entropy()
        for t in self.transforms:
            if isinstance(t, AffineTransform):
                affine_ent = torch.log(torch.abs(t.scale))
                if t.event_dim > 0:
                    sum_dims = list(range(-t.event_dim, 0))
                    affine_ent = affine_ent.sum(dim=sum_dims)
                ent = ent + affine_ent
            else:
                raise NotImplementedError
        return ent

