""" Logical operators"""
from .base import *


class BoundWhere(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(0)
        condition = v[0][0]
        return tuple([torch.where(condition, v[1][j], v[2][j]) for j in range(2)])

    def bound_backward(self, last_lA, last_uA, condition, x, y):
        assert torch.allclose(condition.lower.float(), condition.upper.float())
        assert self.from_input
        mask = condition.lower.float()

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            assert last_A.ndim > 1
            A_x = self.broadcast_backward(mask.unsqueeze(0) * last_A, x)
            A_y = self.broadcast_backward((1 - mask).unsqueeze(0) * last_A, y)
            return A_x, A_y

        lA_x, lA_y = _bound_oneside(last_lA)
        uA_x, uA_y = _bound_oneside(last_uA)

        return [(None, None), (lA_x, uA_x), (lA_y, uA_y)], 0, 0

class BoundNot(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return x.logical_not()