""" Softmax """
from .base import *

class BoundSoftmaxImpl(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        assert self.axis == int(self.axis)

    @Bound.save_io_shape
    def forward(self, x):
        max_x = torch.max(x, dim=self.axis).values
        x = torch.exp(x - max_x.unsqueeze(self.axis))
        s = torch.sum(x, dim=self.axis, keepdim=True)
        return x / s

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class BoundSoftmax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.option = options.get('softmax', 'complex')
        if self.option == 'complex':
            self.complex = True
        else:
            self.max_input = 30

    @Bound.save_io_shape
    def forward(self, x):
        assert self.axis == int(self.axis)
        if self.option == 'complex':
            self.input = (x,)
            self.model = BoundSoftmaxImpl(self.axis)
            self.model.device = self.device
            return self.model(x)
        else:
            return F.softmax(x, dim=self.axis)

    def interval_propagate(self, *v):
        assert self.option != 'complex'
        assert self.perturbed
        h_L, h_U = v[0]
        shift = h_U.max(dim=self.axis, keepdim=True).values
        exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
        lower = exp_L / (torch.sum(exp_U, dim=self.axis, keepdim=True) - exp_U + exp_L + epsilon)
        upper = exp_U / (torch.sum(exp_L, dim=self.axis, keepdim=True) - exp_L + exp_U + epsilon)
        return lower, upper  

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]
