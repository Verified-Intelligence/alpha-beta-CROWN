from .base import *

class BoundDropout(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.dropout = nn.Dropout(p=attr['ratio'])
        self.scale = 1 / (1 - attr['ratio'])

    @Bound.save_io_shape
    def forward(self, x):
        res = self.dropout(x)
        self.mask = res == 0
        return res

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.where(self.mask.unsqueeze(0), torch.tensor(0).to(last_A), last_A * self.scale)
        lA = _bound_oneside(last_lA)
        uA = _bound_oneside(last_uA)
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        assert (torch.min(self.mask) >= 0)
        lw = x.lw * self.mask.unsqueeze(1)
        lb = x.lb * self.mask
        uw = x.uw * self.mask.unsqueeze(1)
        ub = x.ub * self.mask
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        if not self.training:
            return h_L, h_U        
        else:
            lower = torch.where(self.mask, torch.tensor(0).to(h_L), h_L * self.scale)
            upper = torch.where(self.mask, torch.tensor(0).to(h_U), h_U * self.scale)
            return lower, upper