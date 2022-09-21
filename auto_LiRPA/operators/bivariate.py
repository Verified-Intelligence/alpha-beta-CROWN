""" Bivariate operators"""
from .base import *
from .nonlinear import BoundSqrt, BoundReciprocal
from .clampmult import multiply_by_A_signs
from ..utils import *
from .solver_utils import grb
from .constant import BoundConstant
from .leaf import BoundParams, BoundBuffers


class BoundMul(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.is_constant_op = False
        for inp in inputs:
            if BoundMul._check_const_input(inp):
                # If any of the two inputs are constant, we do not need input bounds.
                # FIXME (05/11/2022): this is just a temporary workaround. We need better way to determine whether we need input bounds, not just for BoundConstant.
                self.is_constant_op = True
        if self.is_constant_op:
            # One input is constant; no bounds required.
            self.requires_input_bounds = []
        else:
            # Both inputs are perturbed. Need relaxation.
            self.requires_input_bounds = [0, 1]

    @staticmethod
    def _check_const_input(inp):
        return isinstance(inp, (BoundConstant, BoundBuffers)) or (isinstance(inp, BoundParams) and inp.perturbation is None)

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

        # Broadcast
        x_l = x_l + torch.zeros_like(y_l)
        x_u = x_u + torch.zeros_like(y_u)
        y_l = y_l + torch.zeros_like(x_l)
        y_u = y_u + torch.zeros_like(x_u)

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    @staticmethod
    def _multiply_by_const(x, const):
        if isinstance(x, torch.Tensor):
            return x * const
        elif isinstance(x, Patches):
            # Multiplies patches by a const. Assuming const is a tensor, and it must be in nchw format.
            assert isinstance(const, torch.Tensor) and const.ndim == 4
            if const.size(0) == x.patches.size(1) and const.size(1) == x.patches.size(-3) and const.size(2) == const.size(3) == 1:
                # The case that we can do channel-wise broadcasting multiplication
                # Shape of const: (batch, in_c, 1, 1)
                # Shape of patches when unstable_idx is None: (spec, batch, in_c, patch_h, patch_w)
                # Shape of patches when unstable_idx is not None: (out_c, batch, out_h, out_w, in_c, patch_h, patch_w)
                const_reshaped = const
            else:
                assert x.unstable_idx is None and (x.padding == 0 or x.padding == [0,0,0,0]) and x.stride == 1 and x.patches.size(-1) == x.patches.size(-2) == 1
                # The assumed dimension is (out_c, N, out_h, out_w, in_c, 1, 1) with padding =1 and stride = 0.
                # In this special case we can directly multiply.
                # After reshape it is (1, N, H, W, C, 1, 1)
                const_reshaped = const.permute(0, 2, 3, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return x.create_similar(x.patches * const_reshaped)
        else:
            raise ValueError(f'Unsupported x type {type(x)}')

    @staticmethod
    def bound_backward_constant(last_lA, last_uA, x, y, op=None):
        op = BoundMul._multiply_by_const if op is None else op
        # Handle the case of multiplication by a constant.
        factor = None
        if not BoundMul._check_const_input(x):
            factor = y.value
        if not BoundMul._check_const_input(y):
            factor = x.value
        # No need to compute A matrix if it is Constant.
        lAx = None if BoundMul._check_const_input(x) or last_lA is None else op(last_lA, factor)
        lAy = None if BoundMul._check_const_input(y) or last_lA is None else op(last_lA, factor)
        uAx = None if BoundMul._check_const_input(x) or last_uA is None else op(last_uA, factor)
        uAy = None if BoundMul._check_const_input(y) or last_uA is None else op(last_uA, factor)

        return [(lAx, uAx), (lAy, uAy)], 0., 0.


    def bound_backward(self, last_lA, last_uA, x, y):
        if self.is_constant_op:
            return self.bound_backward_constant(last_lA, last_uA, x, y)
        else:
            return self.bound_backward_both_perturbed(last_lA, last_uA, x, y)

    def bound_backward_both_perturbed(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)

        def _bound_oneside(last_A,
                           alpha_pos, beta_pos, gamma_pos,
                           alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0

            if type(last_A) == Patches:
                # In patches mode, we need to unfold lower and upper slopes. In matrix mode we simply return.
                def _maybe_unfold(d_tensor, last_A):
                    if d_tensor is None:
                        return None

                    d_shape = d_tensor.size()
                    # Reshape to 4-D tensor to unfold.
                    d_tensor = d_tensor.view(-1, *d_shape[-3:])
                    # unfold the slope matrix as patches. Patch shape is [spec * batch, out_h, out_w, in_c, H, W).
                    d_unfolded = inplace_unfold(d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding, inserted_zeros=last_A.inserted_zeros, output_padding=last_A.output_padding)
                    # Reshape to (spec, batch, out_h, out_w, in_c, H, W); here spec_size is out_c.
                    d_unfolded_r = d_unfolded.view(*last_A.shape[:3], *d_unfolded.shape[1:])
                    if last_A.unstable_idx is not None:
                        if d_unfolded_r.size(0) == 1:
                            # Broadcast the spec shape, so only need to select the reset dimensions.
                            # Change shape to (out_h, out_w, batch, in_c, H, W) or (out_h, out_w, in_c, H, W).
                            d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                            d_unfolded_r = d_unfolded_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
                            # output shape: (unstable_size, batch, in_c, H, W).
                        else:
                            d_unfolded_r = d_unfolded_r[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                        # For sparse patches, the shape after unfold is (unstable_size, batch_size, in_c, H, W).
                    # For regular patches, the shape after unfold is (spec, batch, out_h, out_w, in_c, H, W).
                    return d_unfolded_r
                # if last_A is not an identity matrix
                assert last_A.identity == 0
                if last_A.identity == 0:
                    # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W]. Here out_c is the spec dimension.
                    # for patches mode, we need to unfold the alpha_pos/neg and beta_pos/neg

                    alpha_pos = _maybe_unfold(alpha_pos, last_A)
                    alpha_neg = _maybe_unfold(alpha_neg, last_A)
                    beta_pos = _maybe_unfold(beta_pos, last_A)
                    beta_neg = _maybe_unfold(beta_neg, last_A)

                    gamma_pos = _maybe_unfold(gamma_pos, last_A)
                    gamma_neg = _maybe_unfold(gamma_neg, last_A)

                    patches = last_A.patches
                    patches_shape = patches.shape
                    A_x, bias = multiply_by_A_signs(patches.view(*patches_shape[:5], -1, *patches_shape[-2:]), alpha_pos, alpha_neg, gamma_pos, gamma_neg, patches_mode=True)
                    A_y, _ = multiply_by_A_signs(patches.view(*patches_shape[:5], -1, *patches_shape[-2:]), beta_pos, beta_neg, None, None, patches_mode=True)
                    A_x = A_x.view(patches_shape)
                    A_y = A_y.view(patches_shape)

                    # broadcast_backward
                    x_dims = []
                    y_dims = []

                    if A_x.shape[A_x.ndim-4] != x.output_shape[len(x.output_shape)-4]:
                        x_dims.append(A_x.ndim-4)

                    if A_y.shape[A_y.ndim-4] != y.output_shape[len(y.output_shape)-4]:
                        y_dims.append(A_y.ndim-4)

                    if len(x_dims) > 0:
                        A_x = A_x.sum(tuple(x_dims), keepdim=True)
                    if len(y_dims) > 0:
                        A_y = A_y.sum(tuple(y_dims), keepdim=True)

                    A_x = Patches(A_x, last_A.stride, last_A.padding, A_x.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape)
                    A_y = Patches(A_y, last_A.stride, last_A.padding, A_y.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape)
            if type(last_A) == Tensor:
                last_A_pos, last_A_neg = last_A.clamp(min=0), last_A.clamp(max=0)
                A_x = last_A_pos * alpha_pos + last_A_neg * alpha_neg
                A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
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

    def bound_forward(self, dim_in, x, y):
        if self.is_constant_op:
            raise NotImplementedError
        return self.bound_forward_both_perturbed(dim_in, x, y)

    @staticmethod
    def bound_forward_both_perturbed(dim_in, x, y):
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
    def interval_propagate_constant(*v, op=lambda x, const: x * const):
        x, y = v[0], v[1]
        x_is_const = x[0] is x[1]  # FIXME: using better way to represent constant perturbation.
        y_is_const = y[0] is y[1]  # We should not check the distance between x[0] and x[1]. It's slow!
        assert x_is_const or y_is_const
        const = x[0] if x_is_const else y[0]
        inp_lb = x[0] if y_is_const else y[0]
        inp_ub = x[1] if y_is_const else y[1]
        pos_mask = (const > 0).to(dtype=inp_lb.dtype)
        neg_mask = 1. - pos_mask
        lb = op(inp_lb, const * pos_mask) + op(inp_ub, const * neg_mask)
        ub = op(inp_ub, const * pos_mask) + op(inp_lb, const * neg_mask)
        return lb, ub

    def interval_propagate(self, *v):
        if self.is_constant_op:
            return self.interval_propagate_constant(*v)
        else:
            return self.interval_propagate_both_perturbed(*v)

    @staticmethod
    def interval_propagate_both_perturbed(*v):
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

        r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
        lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        return lower, upper

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        for vi in v:
            assert isinstance(vi, Tensor), "build solver for BoundMul only with tensors for now"
        self.solver_vars = v[0] * v[1]

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
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.is_constant_op = False
        for inp in inputs:
            if isinstance(inp, (BoundConstant, BoundBuffers)):
                # If any of the two inputs are constant, we do not need input bounds.
                # FIXME (05/11/2022): this is just a temporary workaround. We need better way to determine whether we need input bounds, not just for BoundConstant.
                # FIXME: unify this handling with BoundMul.
                self.is_constant_op = True
        if self.is_constant_op:
            # One input is constant; no bounds required.
            self.requires_input_bounds = []
        else:
            # Both inputs are perturbed. Need relaxation.
            self.requires_input_bounds = [0, 1]

    def forward(self, x, y):
        # FIXME (05/11/2022): ad-hoc implementation for layer normalization
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
        if self.is_constant_op:
            return BoundMul.bound_backward_constant(last_lA, last_uA, x, y, op=lambda x, const: BoundMul._multiply_by_const(x, 1/const))
        else:
            return self.bound_backward_both_perturbed(last_lA, last_uA, x, y)

    def bound_backward_both_perturbed(self, last_lA, last_uA, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        A, lower_b, upper_b = mul.bound_backward(last_lA, last_uA, x, y_r)
        A_y, lower_b_y, upper_b_y = reciprocal.bound_backward(A[1][0], A[1][1], y)
        if isinstance(upper_b_y, Tensor) and upper_b_y.ndim == 1:
            upper_b_y = upper_b_y.unsqueeze(-1)
        if isinstance(lower_b_y, Tensor) and lower_b_y.ndim == 1:
            lower_b_y = lower_b_y.unsqueeze(-1)
        upper_b = upper_b + upper_b_y
        lower_b = lower_b + lower_b_y
        return [A[0], A_y[0]], lower_b, upper_b

    def bound_forward(self, dim_in, x, y):
        assert not self.is_constant_op
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        y_r_linear = reciprocal.bound_forward(dim_in, y)
        y_r_linear.lower = y_r.lower
        y_r_linear.upper = y_r.upper
        return mul.bound_forward(dim_in, x, y_r_linear)

    def interval_propagate(self, *v):
        if self.is_constant_op:
            return BoundMul.interval_propagate_constant(*v, op=lambda x, const: x / const)
        else:
            return self.interval_propagate_both_perturbed(*v)

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

            dev_sqr_lower = (1 - (dev_lower < 0).to(dev_lower.dtype) * (dev_upper > 0).to(dev_lower.dtype)) * \
                torch.min(dev_lower.abs(), dev_upper.abs())**2
            dev_sqr_upper = torch.max(dev_lower.abs(), dev_upper.abs())**2

            sum_lower = (dev_sqr_lower.sum(dim=-1, keepdim=True) - dev_sqr_lower) / dev_sqr_upper.clamp(min=epsilon)
            sqrt_lower = torch.sqrt(1. / n * (sum_lower + 1))
            sum_upper = (dev_sqr_upper.sum(dim=-1, keepdim=True) - dev_sqr_upper) / \
                dev_sqr_lower.clamp(min=epsilon)
            sqrt_upper = torch.sqrt(1. / n * (sum_upper + 1))

            lower = (dev_lower < 0).to(dev_lower.dtype) * (-1. / sqrt_lower) + (dev_lower > 0).to(dev_lower.dtype) * (1. / sqrt_upper)
            upper = (dev_upper > 0).to(dev_upper.dtype) * (1. / sqrt_lower) + (dev_upper < 0).to(dev_upper.dtype) * (-1. / sqrt_upper)

            return lower, upper

        x, y = v[0], v[1]
        assert (y[0] > 0).all()
        return x[0] / y[1], x[1] / y[0]

    def _convert_to_mul(self, x, y):
        try:
            reciprocal = BoundReciprocal({}, [], 0, None)
            mul = BoundMul({}, [], 0, None)
        except:
            # to make it compatible with previous code
            reciprocal = BoundReciprocal(None, {}, [], 0, None)
            mul = BoundMul(None, {}, [], 0, None)
        reciprocal.output_shape = mul.output_shape = self.output_shape
        reciprocal.batch_dim = mul.batch_dim = self.batch_dim

        y_r = copy.copy(y)
        if isinstance(y_r, LinearBound):
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
        else:
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
        return reciprocal, mul, y_r

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        for vi in v:
            assert isinstance(vi, Tensor), "build solver for BoundDiv only with tensors for now"
        self.solver_vars = v[0] / v[1]

class BoundAdd(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # FIXME: This is not the right way to enable patches mode. Instead we must traverse the graph and determine when patches mode needs to be used.
        self.mode = options.get("conv_mode", "matrix")

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
        lb, ub = x.lb + y.lb, x.ub + y.ub

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        lw = add_w(x.lw, y.lw, x.lb, y.lb)
        uw = add_w(x.uw, y.uw, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, Tensor))
        return x[0] + y[0], x[1] + y[1]

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        gvar_array2 = np.array(v[1])
        assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

        # flatten to create vars and constrs first
        gvar_array1 = gvar_array1.reshape(-1)
        gvar_array2 = gvar_array2.reshape(-1)
        new_layer_gurobi_vars = []
        for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
            var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f'lay{self.name}_{neuron_idx}')
            model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

class BoundSub(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # FIXME: This is not the right way to enable patches mode. Instead we must traverse the graph and determine when patches mode needs to be used.
        self.mode = options.get("conv_mode", "matrix")

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                return self.broadcast_backward(sign * last_A, w)
            elif isinstance(last_A, Patches):
                if sign == 1:
                    # Patches shape requires no broadcast.
                    return last_A
                else:
                    # Multiply by the sign.
                    return last_A.create_similar(sign * last_A.patches)
            else:
                raise ValueError(f'Unknown last_A type {type(last_A)}')

        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        lb, ub = x.lb - y.ub, x.ub - y.lb

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        lw = add_w(x.lw, -y.uw, x.lb, y.lb)
        uw = add_w(x.uw, -y.lw, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        gvar_array2 = np.array(v[1])
        assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

        # flatten to create vars and constrs first
        gvar_array1 = gvar_array1.reshape(-1)
        gvar_array2 = gvar_array2.reshape(-1)
        new_layer_gurobi_vars = []
        for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
            var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f'lay{self.name}_{neuron_idx}')
            model.addConstr(var == (var1 - var2), name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

class BoundEqual(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x, y):
        return x == y
