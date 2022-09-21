""" Constant operators, including operators that are usually fixed nodes and not perturbed """
from .base import *

class BoundConstant(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.value = attr['value'].to(self.device)
        self.use_default_ibp = True

    def forward(self):
        return self.value.to(self.device)

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def bound_backward(self, last_lA, last_uA):
        def _bound_oneside(A):
            if A is None:
                return 0.0

            if type(A) == Tensor:
                if A.ndim > 2:
                    A = torch.sum(A, dim=list(range(2, A.ndim)))
            elif type(A) == Patches:
                assert A.padding == 0 or A.padding == (0, 0, 0, 0) or self.value == 0  # FIXME (09/19): adding padding here.
                patches_reshape = torch.sum(A.patches, dim=(-1, -2, -3)) * self.value.to(self.device)
                # Expected shape for bias is (spec, batch, out_h, out_w) or (unstable_size, batch)
                return patches_reshape

            return A * self.value.to(self.device)

        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(dim_in, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.value
        

class BoundPrimConstant(Bound):
    def __init__(self, attr, input, output_index, options):
        super().__init__(attr, input, output_index, options)

    def forward(self):
        return torch.tensor([], device=self.device)

class BoundConstantOfShape(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.value = attr['value'].to(self.device)

    def forward(self, x):
        self.x = x
        self.from_input = True
        return self.value.expand(*list(x))

    def bound_backward(self, last_lA, last_uA, x):
        if last_lA is not None:
            lower_sum_b = last_lA * self.value
            while lower_sum_b.ndim > 2:
                lower_sum_b = torch.sum(lower_sum_b, dim=-1)
        else:
            lower_sum_b = 0

        if last_uA is not None:
            upper_sum_b = last_uA * self.value
            while upper_sum_b.ndim > 2:
                upper_sum_b = torch.sum(upper_sum_b, dim=-1)
        else:
            upper_sum_b = 0

        return [(None, None)], lower_sum_b, upper_sum_b

    def bound_forward(self, dim_in, x):
        assert (len(self.x) >= 1)
        lb = ub = torch.ones(self.output_shape, device=self.device) * self.value
        lw = uw = torch.zeros(self.x[0], dim_in, *self.x[1:], device=self.device)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        self.x = v[0][0]
        size = int(v[0][0].item()) if isinstance(v[0][0], Tensor) else v[0][0]
        value = torch.ones(size, device=self.device) * self.value
        return value, value

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v)

    def infer_batch_dim(self, batch_size, *x):
        # FIXME Should avoid referring to batch_size; Treat `torch.Size` results differently
        if self.x[0] == batch_size:
            return 0
        else:
            return -1

class BoundRange(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, start, end, step):
        if start.dtype == end.dtype == step.dtype == torch.int64:
            return torch.arange(start, end, step, dtype=torch.int64, device=self.device)
        else:
            return torch.arange(start, end, step, device=self.device)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == x[1] == x[2] == -1
        return -1

class BoundATenDiag(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, x, diagonal=0):
        return torch.diag(x, diagonal=diagonal)

    def interval_propagate(self, *v):
        return Interval.make_interval(torch.diag(v[0][0], v[1][0]), torch.diag(v[0][1], v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        return 1  # This is not a batch operation.

class BoundATenDiagonal(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, x, offset=0, dim1=0, dim2=1):
        return torch.diagonal(x, offset=offset, dim1=dim1, dim2=dim2)

    def interval_propagate(self, *v):
        params = (v[1][0], v[2][0], v[3][0])
        return Interval.make_interval(torch.diagonal(v[0][0], *params), torch.diagonal(v[0][1], *params), v[0])
