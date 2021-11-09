from .base import * 

class BoundCast(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
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

    @Bound.save_io_shape
    def forward(self, x):
        self.type_in = x.dtype
        return x.to(self.type)

    def bound_backward(self, last_lA, last_uA, x):
        lA = last_lA.to(self.type_in) if last_lA is not None else None
        uA = last_uA.to(self.type_in) if last_uA is not None else None
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(
            x.lw.to(self.type), x.lb.to(self.type),
            x.uw.to(self.type), x.ub.to(self.type))
