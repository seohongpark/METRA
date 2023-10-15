from torch.distributions.transforms import AffineTransform, _InverseTransform

class NoWeakrefTrait(object):
    def _inv_no_weakref(self):
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = None
        if self._inv is not None:
            #inv = self._inv()
            inv = self._inv
        if inv is None:
            inv = _InverseTransform(self)
            #inv = _InverseTransformNoWeakref(self)
            #self._inv = weakref.ref(inv)
            self._inv = inv
        return inv

class AffineTransformEx(AffineTransform, NoWeakrefTrait):
    @property
    def inv(self):
        return NoWeakrefTrait._inv_no_weakref(self)

    def maybe_clone_to_device(self, device):
        if device == self.loc.device:
            return self
        return AffineTransformEx(loc=self.loc.to(device, copy=True),
                                 scale=self.scale.to(device, copy=True))

