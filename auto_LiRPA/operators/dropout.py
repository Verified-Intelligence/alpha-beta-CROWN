from .base import *

class BoundDropout(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        if 'ratio' in attr:
            self.ratio = attr['ratio']
            self.dynamic = False
        else:
            self.ratio = None
            self.dynamic = True
        self.clear()

    def clear(self):
        self.mask = None

    def forward(self, *inputs):
        x = inputs[0]
        if not self.training:
            return x
        if self.dynamic:
            # Inputs: data, ratio (optional), training_mode (optional)
            # We assume ratio must exist in the inputs.
            # We ignore training_mode, but will use self.training which can be 
            # changed after BoundedModule is built.
            assert inputs[1].dtype == torch.float32
            self.ratio = inputs[1]
        if self.ratio >= 1:
            raise ValueError('Ratio in dropout should be less than 1')
        self.mask = torch.rand(x.shape) > self.ratio
        return x * self.mask / (1 - self.ratio)

    def _check_forward(self):
        """ If in the training mode, a forward pass should have been called."""
        if self.training and self.mask is None:
            raise RuntimeError('For a model with dropout in the training mode, '\
                'a clean forward pass must be called before bound computation')

    def bound_backward(self, last_lA, last_uA, *args):
        empty_A = [(None, None)] * (len(args) -1)
        if not self.training:
            return [(last_lA, last_uA), *empty_A], 0, 0
        self._check_forward()
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return last_A * self.mask / (1 - self.ratio)
        lA = _bound_oneside(last_lA)
        uA = _bound_oneside(last_uA)
        return [(lA, uA), *empty_A], 0, 0

    def bound_forward(self, dim_in, x, *args):
        if not self.training:
            return x
        self._check_forward()
        lw = x.lw * self.mask.unsqueeze(1) / (1 - self.ratio)
        lb = x.lb * self.mask / (1 - self.ratio)
        uw = x.uw * self.mask.unsqueeze(1) / (1 - self.ratio)
        ub = x.ub * self.mask / (1 - self.ratio)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        if not self.training:
            return v[0]
        self._check_forward()
        h_L, h_U = v[0] 
        lower = h_L * self.mask / (1 - self.ratio)
        upper = h_U * self.mask / (1 - self.ratio)
        return lower, upper