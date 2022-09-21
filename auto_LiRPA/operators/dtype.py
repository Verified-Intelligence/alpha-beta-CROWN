from .base import * 
from ..utils import Patches

class BoundCast(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.to = attr['to']
        self.data_types = [
            None,  torch.float, torch.uint8, torch.int8,
            None,  torch.int16, torch.int32, torch.int64,
            None,  torch.bool, torch.float16, torch.float32,
            None,  None
        ]
        self.type = self.data_types[self.to]
        assert self.type is not None
        self.use_default_ibp = True

    def forward(self, x):
        self.type_in = x.dtype
        return x.to(self.type)

    def bound_backward(self, last_lA, last_uA, x):
        if type(last_lA) == Tensor or type(last_lA) == Tensor:
            lA = last_lA.to(self.type_in) if last_lA is not None else None
            uA = last_uA.to(self.type_in) if last_uA is not None else None
        else:
            if last_lA is not None:
                lA = Patches(last_lA.patches.to(self.type_in), last_lA.stride, last_lA.padding, last_lA.shape, last_lA.identity, last_lA.unstable_idx, last_lA.output_shape)
            if last_uA is not None:
                uA = Patches(last_uA.patches.to(self.type_in), last_uA.stride, last_uA.padding, last_uA.shape, last_uA.identity, last_uA.unstable_idx, last_uA.output_shape)
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(
            x.lw.to(self.type), x.lb.to(self.type),
            x.uw.to(self.type), x.ub.to(self.type))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])
