""" Bivariate operators"""
from .base import *
from .activation import BoundSqrt, BoundReciprocal


class BoundMul(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    @staticmethod
    def get_bound_mul(x_l, x_u, y_l, y_u):
        alpha_l = y_l
        beta_l = x_l
        gamma_l = -alpha_l * beta_l

        alpha_u = y_u
        beta_u = x_l
        gamma_u = -alpha_u * beta_u

        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    # Special case when input is x * x.
    @staticmethod
    def get_bound_square(x_l, x_u):
        # Lower bound is a z=0 line if x_l and x_u have different signs.
        # Otherwise, the lower bound is a tangent line at x_l.
        # The lower bound should always be better than IBP.

        # If both x_l and x_u < 0, select x_u. If both > 0, select x_l.
        # If x_l < 0 and x_u > 0, we use the z=0 line as the lower bound.
        x_m = F.relu(x_l) - F.relu(-x_u)
        alpha_l = 2 * x_m
        gamma_l = - x_m * x_m

        # Upper bound: connect the two points (x_l, x_l^2) and (x_u, x_u^2).
        # The upper bound should always be better than IBP.
        alpha_u = x_l + x_u
        gamma_u = - x_l * x_u

        # Parameters before the second variable are all zeros, not used.
        beta_l = torch.zeros_like(x_l)
        beta_u = beta_l
        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    @staticmethod
    def _relax(x, y):
        if x is y:
            # A shortcut for x * x.
            return BoundMul.get_bound_square(x.lower, x.upper)

        x_l, x_u = x.lower, x.upper
        y_l, y_u = y.lower, y.upper

        # broadcast
        for k in [1, -1]:
            x_l = x_l + k * y_l
            x_u = x_u + k * y_u
        for k in [1, -1]:
            y_l = y_l + k * x_l
            y_u = y_u + k * x_u

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    def bound_backward(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)

        def _bound_oneside(last_A,
                           alpha_pos, beta_pos, gamma_pos,
                           alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            last_A_pos, last_A_neg = last_A.clamp(min=0), last_A.clamp(max=0)
            A_x = last_A_pos * alpha_pos + last_A_neg * alpha_neg
            A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
            last_A = last_A.reshape(last_A.shape[0], last_A.shape[1], -1)
            A_x = self.broadcast_backward(A_x, x)
            A_y = self.broadcast_backward(A_y, y)
            bias = self.get_bias(last_A_pos, gamma_pos) + \
                   self.get_bias(last_A_neg, gamma_neg)
            return A_x, A_y, bias

        lA_x, lA_y, lbias = _bound_oneside(
            last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(
            last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def bound_forward(dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = x.lw, x.lb, x.uw, x.ub
        y_lw, y_lb, y_uw, y_ub = y.lw, y.lb, y.uw, y.ub

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        if x_lw is None: x_lw = 0
        if y_lw is None: y_lw = 0
        if x_uw is None: x_uw = 0
        if y_uw is None: y_uw = 0

        lw = alpha_l.unsqueeze(1).clamp(min=0) * x_lw + alpha_l.unsqueeze(1).clamp(max=0) * x_uw
        lw = lw + beta_l.unsqueeze(1).clamp(min=0) * y_lw + beta_l.unsqueeze(1).clamp(max=0) * y_uw
        lb = alpha_l.clamp(min=0) * x_lb + alpha_l.clamp(max=0) * x_ub + \
             beta_l.clamp(min=0) * y_lb + beta_l.clamp(max=0) * y_ub + gamma_l
        uw = alpha_u.unsqueeze(1).clamp(max=0) * x_lw + alpha_u.unsqueeze(1).clamp(min=0) * x_uw
        uw = uw + beta_u.unsqueeze(1).clamp(max=0) * y_lw + beta_u.unsqueeze(1).clamp(min=0) * y_uw
        ub = alpha_u.clamp(max=0) * x_lb + alpha_u.clamp(min=0) * x_ub + \
             beta_u.clamp(max=0) * y_lb + beta_u.clamp(min=0) * y_ub + gamma_u

        return LinearBound(lw, lb, uw, ub)

    @staticmethod
    def interval_propagate(*v):
        x, y = v[0], v[1]
        if x is y:
            # A shortcut for x * x.
            h_L, h_U = v[0]
            r0 = h_L * h_L
            r1 = h_U * h_U
            # When h_L < 0, h_U > 0, lower bound is 0.
            # When h_L < 0, h_U < 0, lower bound is h_U * h_U.
            # When h_L > 0, h_U > 0, lower bound is h_L * h_L.
            l = F.relu(h_L) - F.relu(-h_U)
            return l * l, torch.max(r0, r1)

        if Interval.use_relative_bounds(x) and Interval.use_relative_bounds(y):
            nominal = x.nominal * y.nominal
            lower_offset = (
                x.nominal.clamp(min=0) * (y.lower_offset) + 
                x.nominal.clamp(max=0) * (y.upper_offset) + 
                y.nominal.clamp(min=0) * (x.lower_offset) + 
                y.nominal.clamp(max=0) * (x.upper_offset) + 
                torch.min(x.lower_offset * y.upper_offset, x.upper_offset * y.lower_offset))
            upper_offset = (
                x.nominal.clamp(min=0) * (y.upper_offset) + 
                x.nominal.clamp(max=0) * (y.lower_offset) + 
                y.nominal.clamp(min=0) * (x.upper_offset) + 
                y.nominal.clamp(max=0) * (x.lower_offset) + 
                torch.max(x.lower_offset * y.lower_offset, x.upper_offset * y.upper_offset))
            return Interval(None, None, nominal=nominal, lower_offset=lower_offset, upper_offset=upper_offset)

        r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
        lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        return lower, upper

    @staticmethod
    def infer_batch_dim(batch_size, *x):
        if x[0] == -1:
            return x[1]
        elif x[1] == -1:
            return x[0]
        else:
            assert x[0] == x[1]
            return x[0]


class BoundDiv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x, y):
        # ad-hoc implementation for layer normalization
        if isinstance(self.inputs[1], BoundSqrt):
            input = self.inputs[0].inputs[0]
            x = input.forward_value
            n = input.forward_value.shape[-1]

            dev = x * (1. - 1. / n) - (x.sum(dim=-1, keepdim=True) - x) / n
            dev_sqr = dev ** 2
            s = (dev_sqr.sum(dim=-1, keepdim=True) - dev_sqr) / dev_sqr.clamp(min=epsilon)
            sqrt = torch.sqrt(1. / n * (s + 1))
            return torch.sign(dev) * (1. / sqrt)

        self.x, self.y = x, y
        return x / y

    def bound_backward(self, last_lA, last_uA, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        A, lower_b, upper_b = mul.bound_backward(last_lA, last_uA, x, y_r)

        A_y, lower_b_y, upper_b_y = reciprocal.bound_backward(A[1][0], A[1][1], y)
        upper_b = upper_b + upper_b_y
        lower_b = lower_b + lower_b_y

        return [A[0], A_y[0]], lower_b, upper_b

    def bound_forward(self, dim_in, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        y_r_linear = reciprocal.bound_forward(dim_in, y)
        y_r_linear = y_r_linear._replace(lower=y_r.lower, upper=y_r.upper)
        return mul.bound_forward(dim_in, x, y_r_linear)

    def interval_propagate(self, *v):
        # ad-hoc implementation for layer normalization
        """
        Compute bounds for layer normalization

        Lower bound
            1) (x_i - mu) can be negative
                - 1 / ( sqrt (1/n * sum_j Lower{(x_j-mu)^2/(x_i-mu)^2} ))
            2) (x_i - mu) cannot be negative
                1 / ( sqrt (1/n * sum_j Upper{(x_j-mu)^2/(x_i-mu)^2} ))

        Lower{(x_j-mu)^2/(x_i-mu)^2}
            Lower{sum_j (x_j-mu)^2} / Upper{(x_i-mu)^2} 

        Upper{(x_j-mu)^2/(x_i-mu)^2}
            Upper{sum_j (x_j-mu)^2} / Lower{(x_i-mu)^2}     
        """        
        if isinstance(self.inputs[1], BoundSqrt):
            input = self.inputs[0].inputs[0]
            n = input.forward_value.shape[-1]
            
            h_L, h_U = input.lower, input.upper

            dev_lower = (
                h_L * (1 - 1. / n) - 
                (h_U.sum(dim=-1, keepdim=True) - h_U) / n
            )
            dev_upper = (
                h_U * (1 - 1. / n) - 
                (h_L.sum(dim=-1, keepdim=True) - h_L) / n
            )

            dev_sqr_lower = (1 - (dev_lower < 0).float() * (dev_upper > 0).float()) * \
                torch.min(dev_lower.abs(), dev_upper.abs())**2 
            dev_sqr_upper = torch.max(dev_lower.abs(), dev_upper.abs())**2

            sum_lower = (dev_sqr_lower.sum(dim=-1, keepdim=True) - dev_sqr_lower) / dev_sqr_upper.clamp(min=epsilon)
            sqrt_lower = torch.sqrt(1. / n * (sum_lower + 1))
            sum_upper = (dev_sqr_upper.sum(dim=-1, keepdim=True) - dev_sqr_upper) / \
                dev_sqr_lower.clamp(min=epsilon)
            sqrt_upper = torch.sqrt(1. / n * (sum_upper + 1))

            lower = (dev_lower < 0).float() * (-1. / sqrt_lower) + (dev_lower > 0).float() * (1. / sqrt_upper)
            upper = (dev_upper > 0).float() * (1. / sqrt_lower) + (dev_upper < 0).float() * (-1. / sqrt_upper)

            return lower, upper

        x, y = v[0], v[1]
        assert (y[0] > 0).all()
        return x[0] / y[1], x[1] / y[0]

    def _convert_to_mul(self, x, y):
        try:
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', self.ori_name, {}, [], 0, None,
                                         self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', self.ori_name, {}, [], 0, None, self.device)
        except:
            # to make it compatible with previous code
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', None, {}, [], 0, None, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', None, {}, [], 0, None, self.device)
        reciprocal.output_shape = mul.output_shape = self.output_shape
        reciprocal.batch_dim = mul.batch_dim = self.batch_dim

        y_r = copy.copy(y)
        if isinstance(y_r, LinearBound):
            y_r = y_r._replace(lower=1. / y.upper, upper=1. / y.lower)
        else:
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
        return reciprocal, mul, y_r

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)

class BoundAdd(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.mode = options.get("conv_mode", "matrix")

    @Bound.save_io_shape
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self.broadcast_backward(last_A, w)

        uA_x = _bound_oneside(last_uA, x)
        uA_y = _bound_oneside(last_uA, y)
        lA_x = _bound_oneside(last_lA, x)
        lA_y = _bound_oneside(last_lA, y)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = Bound.broadcast_forward(dim_in, x, self.output_shape)
        y_lw, y_lb, y_uw, y_ub = Bound.broadcast_forward(dim_in, y, self.output_shape)
        lw, lb = x_lw + y_lw, x_lb + y_lb
        uw, ub = x_uw + y_uw, x_ub + y_ub
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, torch.Tensor))

        if Interval.use_relative_bounds(x) and Interval.use_relative_bounds(y):
            return Interval(
                None, None, 
                x.nominal + y.nominal,
                x.lower_offset + y.lower_offset,
                x.upper_offset + y.upper_offset)

        return x[0] + y[0], x[1] + y[1]

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundSub(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            return self.broadcast_backward(sign * last_A, w)

        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = Bound.broadcast_forward(dim_in, x, self.output_shape)
        y_lw, y_lb, y_uw, y_ub = Bound.broadcast_forward(dim_in, y, self.output_shape)
        lw, lb = x_lw - y_uw, x_lb - y_ub
        uw, ub = x_uw - y_lw, x_ub - y_lb
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        if Interval.use_relative_bounds(x) and Interval.use_relative_bounds(y):
            return Interval(
                None, None, 
                x.nominal - y.nominal,
                x.lower_offset - y.upper_offset,
                x.upper_offset - y.lower_offset)

        return x[0] - y[1], x[1] - y[0]

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)

class BoundEqual(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, y):
        return x == y

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)        