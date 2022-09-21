"""Unary nonlinearities other than activation functions."""
import math
import torch
from .activation import BoundActivation, BoundTanh
from .base import epsilon, LinearBound

class BoundSin(BoundActivation):
    # Lookup tables shared by all BoundSin classes.
    xl_lower_tb = None
    xl_upper_tb = None
    xu_lower_tb = None
    xu_upper_tb = None
    func, d_func = torch.sin, torch.cos
    n_table_entries = 1001

    @staticmethod
    def n_crossing(start, end, s):
        """Check how many times we will encounter value s + k*2*pi within start and end for any integer k."""
        dtype = start.dtype
        cycles = torch.floor((end - start) / (2 * math.pi))  # Number of 2pi cycles.
        # Move s and end to the same 2 * pi cycle as start.
        dist = torch.floor((s - start) / (2 * math.pi))
        real_s = s - dist * 2 * math.pi
        real_end = end - cycles * 2 * math.pi
        # assert (real_end >= start - 2 ** (-20)).all()
        return (real_s >= start).to(dtype) * (real_s <= real_end).to(dtype) + cycles

    @staticmethod
    def get_intersection(start, end, c, theta=0.):
        """Get the number of intersections between y = sin(x + theta) and y = c between start and end."""
        # Use arcsine to find the first 2 intersections.
        crossing1 = torch.arcsin(c) - theta
        crossing2 = math.pi - crossing1 - 2 * theta  # Problematic at exact 1/2 pi, but ok in our case (happens only when lb=ub).
        return BoundSin.n_crossing(start, end, crossing1) + BoundSin.n_crossing(start, end, crossing2)

    @staticmethod
    def get_bounding_slope(xl, xu, c, theta=0.):
        """Find the point between xl and xu such that the tangent line at that point is a lower/upper bound."""
        dtype = xl.dtype
        crossing1 = torch.arcsin(c) - theta  # output is [-0.5 pi, 0.5 pi] - theta. For cosine, theta=0.5 pi and crossing point is between -pi to 0.
        crossing2 = math.pi - crossing1 - 2 * theta  # output is [0.5pi, 1.5pi] - theta. For cosine, it is between 0 and pi.
        # Find the crossing point between xl and xu.
        # First see how xl is away from the [-0.5pi, 1.5pi] range for sine or [-pi, pi] range for cosine.
        cycles1 = torch.floor((crossing1 - xl) / (2 * math.pi)) * 2 * math.pi
        # Move the two crossing points to the same cycle as xl.
        crossing1_moved = crossing1 - cycles1
        cycles2 = torch.floor((crossing2 - xl) / (2 * math.pi)) * 2 * math.pi
        crossing2_moved = crossing2 - cycles2
        # Then check which crossing point is the actual tangent point between xl and xu.
        crossing1_used = (crossing1_moved >= xl).to(dtype) * (crossing1_moved <= xu).to(dtype)
        crossing2_used = (crossing2_moved >= xl).to(dtype) * (crossing2_moved <= xu).to(dtype)
        crossing_point = crossing1_used * crossing1_moved + crossing2_used * crossing2_moved
        # print(f'c1={crossing1.item():.05f}, c2={crossing2.item():.05f}, cy1={cycles1.item():.05f}, cy2={cycles2.item():.05f}, c1m={crossing1_moved.item():.05f}, c2m={crossing2_moved.item():.05f}, u1={crossing1_used.item()}, u2={crossing2_used.item()}, xl={xl.item():.05f}, xu={xu.item():.05f}')
        return crossing_point

    @staticmethod
    def check_bound(tangent_point, x):
        """Check whether the tangent line at tangent_point is a valid lower/upper bound for x."""
        # evaluate the value of the tangent line at x and see it is >= 0 or <=0.
        d = BoundSin.d_func(tangent_point)
        val = d * (x - tangent_point) + BoundSin.func(tangent_point)
        # We want a positive margin when finding a lower line, but as close to 0 as possible.
        # We want a negative margin when finding a upper line, but as close to 0 as possible.
        margin = BoundSin.func(x) - val
        return margin

    @staticmethod
    @torch.no_grad()
    def get_lower_left_bound(xl, steps=20):
        """Get a global lower bound given lower bound on x. Return slope and intercept."""
        dtype = xl.dtype
        # Constrain xl into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xl + 0.5 * math.pi) / (2 * math.pi)) * (2 * math.pi)
        xl = xl - cycles
        use_tangent_line = (xl >= math.pi).to(dtype)
        # Case 1: xl > pi, Lower tangent line is the only possible lower bound.
        case1_d = BoundSin.d_func(xl)
        case1_b = BoundSin.func(xl) - case1_d * (xl + cycles)
        # Case 2: Binary search needed. Testing from another tangent endpoint in [pi, 1.5*pi]. It must be in this region.
        left = math.pi * torch.ones_like(xl)
        # The right end guarantees the margin > 0 because it is basically a IBP lower bound (-1).
        right = (1.5 * math.pi) * torch.ones_like(xl)
        last_right = right.clone()
        for i in range(steps):
            mid = (left + right) / 2.
            margin = BoundSin.check_bound(mid, xl)
            pos_mask = (margin > 0).to(dtype)  # We want to margin > 0 but at small as possible.
            neg_mask = 1.0 - pos_mask
            right = mid * pos_mask + right * neg_mask  # We have positive margin, reduce right hand side.
            last_right = mid * pos_mask + last_right * neg_mask  # Always sound, since the margin is positive.
            left = mid * neg_mask + left * pos_mask
        case2_d = BoundSin.d_func(last_right)
        case2_b = BoundSin.func(last_right) - case2_d * (last_right + cycles)
        d = case1_d * use_tangent_line + case2_d * (1. - use_tangent_line)
        b = case1_b * use_tangent_line + case2_b * (1. - use_tangent_line)
        # Return slope and bias.
        return [d, b]

    @staticmethod
    @torch.no_grad()
    def get_upper_left_bound(xl, steps=20):
        dtype = xl.dtype
        """Get a global upper bound given lower bound on x. Return slope and intercept."""
        # Constrain xl into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xl - 0.5 * math.pi) / (2 * math.pi)) * (2 * math.pi)
        xl = xl - cycles
        use_tangent_line = (xl >= 2.0 * math.pi).to(dtype)
        # Case 1: xl > pi, Lower tangent line is the only possible lower bound.
        case1_d = BoundSin.d_func(xl)
        case1_b = BoundSin.func(xl) - case1_d * (xl + cycles)
        # Case 2: Binary search needed. Testing from another tangent endpoint in [pi, 1.5*pi]. It must be in this region.
        left = (2.0 * math.pi) * torch.ones_like(xl)
        # The right end guarantees the margin > 0 because it is basically a IBP lower bound (-1).
        right = (2.5 * math.pi) * torch.ones_like(xl)
        last_right = right.clone()
        for i in range(steps):
            mid = (left + right) / 2.
            margin = BoundSin.check_bound(mid, xl)
            pos_mask = (margin > 0).to(dtype)  # We want to margin < 0 but at small as possible.
            neg_mask = 1.0 - pos_mask
            right = mid * neg_mask + right * pos_mask  # We have positive margin, reduce right hand side.
            last_right = mid * neg_mask + last_right * pos_mask  # Always sound, since the margin is positive.
            left = mid * pos_mask + left * neg_mask
        case2_d = BoundSin.d_func(last_right)
        case2_b = BoundSin.func(last_right) - case2_d * (last_right + cycles)
        d = case1_d * use_tangent_line + case2_d * (1. - use_tangent_line)
        b = case1_b * use_tangent_line + case2_b * (1. - use_tangent_line)
        # Return slope and bias.
        return [d, b]

    @staticmethod
    @torch.no_grad()
    def get_lower_right_bound(xu, steps=20):
        """Get a global lower bound given upper bound on x. Return slope and intercept."""
        # Constrain xu into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xu + 0.5 * math.pi) / (2 * math.pi)) * (2 * math.pi)
        xu = xu - cycles
        d, _ = BoundSin.get_lower_left_bound(math.pi - xu, steps)
        return [-d, BoundSin.func(xu) + d * (xu + cycles)]

    @staticmethod
    @torch.no_grad()
    def get_upper_right_bound(xu, steps=20):
        """Get a global upper bound given upper bound on x. Return slope and intercept."""
        # Constrain xu into the 0.5 pi to 2.5 pi region.
        cycles = torch.floor((xu - 0.5 * math.pi) / (2 * math.pi)) * (2 * math.pi)
        xu = xu - cycles
        d, _ = BoundSin.get_upper_left_bound(3 * math.pi - xu, steps)
        return [-d, BoundSin.func(xu) + d * (xu + cycles)]

    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # Bound limits used by IBP.
        self.max_point = math.pi / 2
        self.min_point = math.pi * 3 / 2

        self.all_table_x = torch.linspace(0, 2 * math.pi, BoundSin.n_table_entries, device=self.device)
        if BoundSin.xl_lower_tb is None:
            # Generate look-up tables.
            BoundSin.xl_lower_tb = BoundSin.get_lower_left_bound(self.all_table_x)
            BoundSin.xl_upper_tb = BoundSin.get_upper_left_bound(self.all_table_x)
            BoundSin.xu_lower_tb = BoundSin.get_lower_right_bound(self.all_table_x)
            BoundSin.xu_upper_tb = BoundSin.get_upper_right_bound(self.all_table_x)
            BoundSin.xl_lower_tb[0], BoundSin.xl_lower_tb[1] = BoundSin.xl_lower_tb[0].to(self.device), BoundSin.xl_lower_tb[1].to(self.device)
            BoundSin.xl_upper_tb[0], BoundSin.xl_upper_tb[1] = BoundSin.xl_upper_tb[0].to(self.device), BoundSin.xl_upper_tb[1].to(self.device)
            BoundSin.xu_lower_tb[0], BoundSin.xu_lower_tb[1] = BoundSin.xu_lower_tb[0].to(self.device), BoundSin.xu_lower_tb[1].to(self.device)
            BoundSin.xu_upper_tb[0], BoundSin.xu_upper_tb[1] = BoundSin.xu_upper_tb[0].to(self.device), BoundSin.xu_upper_tb[1].to(self.device)

    @staticmethod
    def interpoloate(x, lower_x, upper_x, lower_y, upper_y):
        # x = torch.clamp(x, min=lower_x, max=upper_x)  # For pytorch >= 1.11
        x = torch.max(torch.min(x, upper_x), lower_x)
        ratio = (x - lower_x) / (upper_x - lower_x + 1e-10)
        return lower_y * (1. - ratio) + upper_y * ratio

    def get_bound_tb(self, tb, x):
        """Find lower or upper bounds from lookup table."""
        step = 2 * math.pi / (BoundSin.n_table_entries - 1)
        # Move to 0 to 2 pi region.
        cycles = torch.floor(x / (2 * math.pi)) * (2 * math.pi)
        x = torch.clamp(x - cycles, min=0, max=2 * math.pi)
        # Find the indice within the lookup table from 0 - 2pi.
        indices = x.div(step).long()
        # Intepolate the nearest d and b. This has better differentiability.
        # Another option is to always take the lower/upper side (better soundness).
        upper_indices = torch.clamp(indices + 1, max=BoundSin.n_table_entries-1)
        lower_x = self.all_table_x[indices]
        upper_x = self.all_table_x[upper_indices]
        # print(indices.item(), lower_x.item(), upper_x.item(), tb[0][indices].item(), tb[0][upper_indices].item())
        d = self.interpoloate(x, lower_x, upper_x, tb[0][indices], tb[0][upper_indices])
        b = self.interpoloate(x, lower_x, upper_x, tb[1][indices], tb[1][upper_indices])
        return d, b - d * cycles

    def forward(self, x):
        return torch.sin(x)

    def interval_propagate(self, *v):
        # Check if a point is in [l, u], considering the 2pi period
        def check_crossing(ll, uu, point):
            return ((((uu - point) / (2 * math.pi)).floor() - ((ll - point) / (2 * math.pi)).floor()) > 0).to(h_Ls.dtype)
        h_L, h_U = v[0][0], v[0][1]
        h_Ls, h_Us = self.forward(h_L), self.forward(h_U)
        # If crossing pi/2, then max is fixed 1.0
        max_mask = check_crossing(h_L, h_U, self.max_point)
        # If crossing pi*3/2, then min is fixed -1.0
        min_mask = check_crossing(h_L, h_U, self.min_point)
        ub = torch.max(h_Ls, h_Us)
        ub = max_mask + (1 - max_mask) * ub
        lb = torch.min(h_Ls, h_Us)
        lb = - min_mask + (1 - min_mask) * lb
        return lb, ub

    def bound_relax_impl(self, lb, ub):
        dtype = lb.dtype
        # Case 1: Connect the two points as a line
        sub = self.func(ub)
        slb = self.func(lb)
        mid = (sub + slb) / 2.
        smid = self.func((ub + lb) / 2)
        case1_line_slope = (sub - slb) / (ub - lb + 1e-10)
        case1_line_bias = slb - case1_line_slope * lb
        gap = smid - mid
        # Check if there are crossings between the line and the sin function.
        grad_crossings = self.get_intersection(lb, ub, case1_line_slope, theta=0.5 * math.pi)
        # If there is no crossing, then we can connect the two points together as a lower/upper bound.
        use_line = grad_crossings == 1
        # Connected line is the upper bound.
        upper_use_line = torch.logical_and(gap <  0, use_line)
        # Connected line is the lower bound.
        lower_use_line = torch.logical_and(gap >= 0, use_line)
        # For the other bound, use the tangent line.
        case1_tangent_point = self.get_bounding_slope(lb, ub, case1_line_slope, theta=0.5 * math.pi)
        case1_tangent_slope = case1_line_slope  # Use the same slope so far.
        stangent = self.func(case1_tangent_point)
        case1_tangent_bias = stangent - case1_tangent_slope * case1_tangent_point
        # Choose the lower/upper based on gap.
        case1_lower_slope = lower_use_line * case1_line_slope + upper_use_line * case1_tangent_slope
        case1_lower_bias = lower_use_line * case1_line_bias + upper_use_line * case1_tangent_bias
        case1_upper_slope = upper_use_line * case1_line_slope + lower_use_line * case1_tangent_slope
        case1_upper_bias = upper_use_line * case1_line_bias + lower_use_line * case1_tangent_bias

        # Case 2: we will try the global lower/upper bounds at lb and ub.
        # For the points and lb and ub, we can construct both lower and upper bounds.
        left_lower = self.get_bound_tb(BoundSin.xl_lower_tb, lb)  # slope, bias.
        left_upper = self.get_bound_tb(BoundSin.xl_upper_tb, lb)
        right_lower = self.get_bound_tb(BoundSin.xu_lower_tb, ub)
        right_upper = self.get_bound_tb(BoundSin.xu_upper_tb, ub)
        # Determine which lower bound is tighter.
        left_lower_error = sub - (left_lower[0] * ub + left_lower[1])
        right_lower_error = slb - (right_lower[0] * lb + right_lower[1])
        left_upper_error = (left_upper[0] * ub + left_upper[1]) - sub
        right_upper_error = (right_upper[0] * lb + right_upper[1]) - slb
        use_left_lower = (left_lower_error < right_lower_error).to(dtype)
        use_right_lower = 1. - use_left_lower
        use_left_upper = (left_upper_error < right_upper_error).to(dtype)
        use_right_upper = 1. - use_left_upper
        # Choose the slope and bias in this case.
        case_2_lower_slope = use_left_lower * left_lower[0] + use_right_lower * right_lower[0]
        case_2_lower_bias = use_left_lower * left_lower[1] + use_right_lower * right_lower[1]
        case_2_upper_slope = use_left_upper * left_upper[0] + use_right_upper * right_upper[0]
        case_2_upper_bias = use_left_upper * left_upper[1] + use_right_upper * right_upper[1]

        # Finally, choose between case 1 and case 2.
        use_line = use_line.to(dtype)
        not_use_line = 1. - use_line
        lower_slope = use_line * case1_lower_slope + not_use_line * case_2_lower_slope
        lower_bias = use_line * case1_lower_bias + not_use_line * case_2_lower_bias
        upper_slope = use_line * case1_upper_slope + not_use_line * case_2_upper_slope
        upper_bias = use_line * case1_upper_bias + not_use_line * case_2_upper_bias
        # print(gap, lower_slope, lower_bias, upper_slope, upper_bias)
        return lower_slope, lower_bias, upper_slope, upper_bias

    def bound_relax(self, x):
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_impl(x.lower, x.upper)
        self.lw = lower_slope
        self.lb = lower_bias
        self.uw = upper_slope
        self.ub = upper_bias


class BoundCos(BoundSin):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.max_point = 0.0
        self.min_point = math.pi

    def forward(self, x):
        return torch.cos(x)

    def bound_relax(self, x):
        # Shift the input by 0.5*pi, and shifting the linear bounds back.
        lb = x.lower + 0.5 * math.pi
        ub = x.upper + 0.5 * math.pi
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_impl(lb, ub)
        self.lw = lower_slope
        self.lb = lower_slope * (0.5 * math.pi) + lower_bias
        self.uw = upper_slope
        self.ub = upper_slope * (0.5 * math.pi) + upper_bias


class BoundAtan(BoundTanh):
    def __init__(self, attr, inputs, output_index, options):
        super(BoundTanh, self).__init__(attr, inputs, output_index, options)
        self.precompute_relaxation('arctan', torch.arctan, self.darctan)
        # Alpha dimension is  (4, 2, output_shape, batch, *shape) for S-shaped functions.
        self.alpha_batch_dim = 3

    def forward(self, x):
        return torch.arctan(x)

    def darctan(self, x):
        return (x.square() + 1.).reciprocal()

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.arctan, self.darctan)


class BoundTan(BoundAtan):
    """
    The implementation of BoundTan is based on the S-shaped BoundAtan. We use the bounds from its
    inverse function and directly convert the bounds of the inverse function to bounds of the original
    function. This trick allows us to quickly implement bounds on inverse functions.
    """
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return torch.tan(x)

    def _check_bounds(self, lower, upper):
        # Lower and upper bounds must be within the same [-½π, ½π] region.
        lower_periods = torch.floor((lower + 0.5 * torch.pi) / torch.pi)
        upper_periods = torch.floor((upper + 0.5 * torch.pi) / torch.pi)
        if not torch.allclose(lower_periods, upper_periods):
            print('Tan preactivation lower bounds:\n', lower)
            print('Tan preactivation upper bounds:\n', upper)
            raise ValueError("BoundTan received pre-activation bounds that produce infinity. "
                    "The preactivation bounds are too loose. Try to reduce perturbation region.")
        # Return the period number for each neuron.
        # Period is 0 => bounds are within [-½π, ½π],
        # Period is 1 => bounds are within [-½π + π, ½π + π]
        # Period is -1 => bounds are within [-½π - π, ½π - π]
        return lower_periods

    def _init_masks(self, x):
        # The masks now must consider the periodicity.
        lower = torch.remainder(x.lower + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        upper = torch.remainder(x.upper + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        self.mask_pos = lower >= 0
        self.mask_neg = upper <= 0
        self.mask_both = torch.logical_not(torch.logical_or(self.mask_pos, self.mask_neg))

    def interval_propagate(self, *v):
        # We need to check if the input lower and upper bounds are within the same period.
        # Otherwise the bounds become infinity.
        concrete_lower, concrete_upper = v[0][0], v[0][1]
        self._check_bounds(concrete_lower, concrete_upper)
        return super().interval_propagate(*v)

    def bound_relax(self, x):
        periods = self._check_bounds(x.lower, x.upper)
        periods = torch.pi * periods
        # Create a fake x with inversed lower and upper.
        inverse_x = lambda: None
        inverse_x.lower = torch.tan(x.lower)
        inverse_x.upper = torch.tan(x.upper)
        super().bound_relax(inverse_x)
        # Lower slope, lower bias, upper slope and upper bias are saved to
        # self.lw, self.lb, self.uw, self.ub. We need to reverse them.
        # E.g., y = self.lw * x + self.lb, now becomes x = 1./self.lw * y - self.lb / self.lw
        # Additionally, we need to add the missing ½π periods.
        new_upper_slope = 1. / self.lw
        new_upper_bias = - self.lb / self.lw - periods / self.lw
        new_lower_slope = 1. / self.uw
        new_lower_bias = - self.ub / self.uw - periods / self.uw
        self.lw = new_lower_slope
        self.lb = new_lower_bias
        self.uw = new_upper_slope
        self.ub = new_upper_bias


class BoundExp(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.options = options.get('exp')
        self.max_input = 0

    def forward(self, x):
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return torch.exp(x - self.max_input)
        return torch.exp(x)

    def interval_propagate(self, *v):
        assert len(v) == 1
        # unary monotonous functions only
        h_L, h_U = v[0]
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(h_U, dim=-1, keepdim=True)[0]
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return torch.exp(h_L), torch.exp(h_U)

    def bound_forward(self, dim_in, x):
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)

        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)

        kl = exp_m
        lw = x.lw * kl.unsqueeze(1)
        lb = kl * (x.lb - m + 1)

        ku = (exp_u - exp_l) / (x.upper - x.lower + epsilon)
        uw = x.uw * ku.unsqueeze(1)
        ub = x.ub * ku - ku * x.lower + exp_l

        return LinearBound(lw, lb, uw, ub)

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        # Special case when computing log_softmax (FIXME: find a better solution, this trigger condition is not reliable).
        if self.loss_fusion and last_lA is None and last_uA is not None and torch.min(
                last_uA) >= 0 and x.from_input:
            # Adding an extra bias term to the input. This is equivalent to adding a constant and subtract layer before exp.
            # Note that we also need to adjust the bias term at the end.
            if self.options == 'no-detach':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0]
            elif self.options != 'no-max-input':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0].detach()
            else:
                self.max_input = 0
            adjusted_lower = x.lower - self.max_input
            adjusted_upper = x.upper - self.max_input
            # relaxation for upper bound only (used in loss fusion)
            exp_l, exp_u = torch.exp(adjusted_lower), torch.exp(adjusted_upper)
            k = (exp_u - exp_l) / (adjusted_upper - adjusted_lower + epsilon)
            if k.requires_grad:
                k = k.clamp(min=1e-6)
            uA = last_uA * k.unsqueeze(0)
            ubias = last_uA * (-adjusted_lower * k + exp_l).unsqueeze(0)

            if ubias.ndim > 2:
                ubias = torch.sum(ubias, dim=tuple(range(2, ubias.ndim)))
            # Also adjust the missing ubias term.
            if uA.ndim > self.max_input.ndim:
                A = torch.sum(uA, dim=tuple(range(self.max_input.ndim, uA.ndim)))
            else:
                A = uA

            # These should hold true in loss fusion
            assert self.batch_dim == 0
            assert A.shape[0] == 1

            batch_size = A.shape[1]
            ubias -= (A.reshape(batch_size, -1) * self.max_input.reshape(batch_size, -1)).sum(dim=-1).unsqueeze(0)
            return [(None, uA)], 0, ubias
        else:
            return super().bound_backward(last_lA, last_uA, x)

    def bound_relax(self, x):
        min_val = -1e9
        l, u = x.lower.clamp(min=min_val), x.upper.clamp(min=min_val)
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)
        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)
        k = exp_m
        self.add_linear_relaxation(mask=None, type='lower', k=k, x0=m, y0=exp_m)
        min_val = -1e9  # to avoid (-inf)-(-inf) when both input.lower and input.upper are -inf
        epsilon = 1e-20
        close = (u - l < epsilon).int()
        k = close * exp_u + (1 - close) * (exp_u - exp_l) / (u - l + epsilon)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=l, y0=exp_l)


class BoundLog(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].forward_value, dim=-1)
        return torch.log(x.clamp(min=epsilon))

    def bound_relax(self, x):
        rl, ru = self.forward(x.lower), self.forward(x.upper)
        ku = (ru - rl) / (x.upper - x.lower + epsilon)
        self.add_linear_relaxation(mask=None, type='lower', k=ku, x0=x.lower, y0=rl)
        m = (x.lower + x.upper) / 2
        k = torch.reciprocal(m)
        rm = self.forward(m)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=m, y0=rm)

    def interval_propagate(self, *v):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            par = self.inputs[0].inputs[0].inputs[0]
            lower = torch.logsumexp(par.lower, dim=-1)
            upper = torch.logsumexp(par.upper, dim=-1)
            return lower, upper
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        A, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            assert A[0][0] is None
            exp_module = self.inputs[0].inputs[0]
            ubias = ubias + self.get_bias(A[0][1], exp_module.max_input.squeeze(-1))
        return A, lbias, ubias


class BoundPow(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x, y):
        return torch.pow(x, y)

    def bound_backward(self, last_lA, last_uA, x, y):
        assert not self.is_input_perturbed(1)
        y = y.lower.item()
        if y == int(y) and y == 2:
            x_l = x.lower
            x_u = torch.max(x.upper, x.lower + 1e-8)

            pow_l = self.forward(x_l, y)
            pow_u = self.forward(x_u, y)
            k_u = (pow_u - pow_l) / (x_u - x_l).clamp(min=1e-8)
            b_u = pow_l - k_u * x_l

            k_l = torch.zeros_like(k_u)
            b_l = torch.zeros_like(b_u)
            x_m = (x_l + x_u) / 2

            # TODO this only holds for y=2
            x_m = (x_u < 0) * torch.max(x_m, x_u * 2) + (x_l > 0) * torch.min(x_m, x_l * 2)
            k_l = y * self.forward(x_m, y - 1)
            b_l = self.forward(x_m, y) - k_l * x_m

            if last_lA is not None:
                last_lA_pos, last_lA_neg = last_lA.clamp(min=0), last_lA.clamp(max=0)
                lA = last_lA_pos * k_l + last_lA_neg * k_u
                lb = self.get_bias(last_lA_pos, b_l) + self.get_bias(last_lA_neg, b_u)
            else:
                lA, lb = None, 0

            if last_uA is not None:
                last_uA_pos, last_uA_neg = last_uA.clamp(min=0), last_uA.clamp(max=0)
                uA = last_uA_pos * k_u + last_uA_neg * k_l
                ub = self.get_bias(last_uA_pos, b_u) + self.get_bias(last_uA_neg, b_l)
            else:
                uA, ub = None, 0

            return [(lA, uA), (None, None)], lb, ub
        else:
            raise NotImplementedError(f'Exponent {y} is not supported yet')

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        exp = v[1][0]
        assert exp == int(exp)
        exp = int(exp)
        pl, pu = torch.pow(v[0][0], exp), torch.pow(v[0][1], exp)
        if exp % 2 == 1:
            return pl, pu
        else:
            pl, pu = torch.min(pl, pu), torch.max(pl, pu)
            mask = 1 - ((v[0][0] < 0) * (v[0][1] > 0)).to(pl.dtype)
            return pl * mask, pu


class BoundReciprocal(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return torch.reciprocal(x)

    def bound_relax(self, x):
        m = (x.lower + x.upper) / 2
        kl = -1 / m.pow(2)
        self.add_linear_relaxation(mask=None, type='lower', k=kl, x0=m, y0=1. / m)
        ku = -1. / (x.lower * x.upper)
        self.add_linear_relaxation(mask=None, type='upper', k=ku, x0=x.lower, y0=1. / x.lower)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0].float(), v[0][1].float()
        assert h_L.min() > 0, 'Only positive values are supported in BoundReciprocal'
        return torch.reciprocal(h_U), torch.reciprocal(h_L)


class BoundSqrt(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return torch.sqrt(x)

    def interval_propagate(self, *v):
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x):
        x_l = x.lower
        x_u = torch.max(x.upper, x.lower + 1e-8)
        sqrt_l = self.forward(x_l)
        sqrt_u = self.forward(x_u)
        k_l = (sqrt_u - sqrt_l) / (x_u - x_l).clamp(min=1e-8)
        b_l = sqrt_l - k_l * x_l

        x_m = (x_l + x_u) / 2
        sqrt_m = self.forward(x_m)
        k_u = -0.5 * torch.pow(x_m, -1.5)
        b_u = sqrt_m - k_u * x_m

        # TODO make this part a general function
        if last_lA is not None:
            last_lA_pos, last_lA_neg = last_lA.clamp(min=0), last_lA.clamp(max=0)
            lA = last_lA_pos * k_l + last_lA_neg * k_u
            lb = self.get_bias(last_lA_pos, b_l) + self.get_bias(last_lA_neg, b_u)
        else:
            lA, lb = None, 0
        if last_uA is not None:
            last_uA_pos, last_uA_neg = last_uA.clamp(min=0), last_uA.clamp(max=0)
            uA = last_uA_pos * k_u + last_uA_neg * k_l
            ub = self.get_bias(last_uA_pos, b_u) + self.get_bias(last_uA_neg, b_l)
        else:
            uA, ub = None, 0

        return [(lA, uA), (None, None)], lb, ub


class BoundSqr(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.nonlinear = True

    def forward(self, x):
        return x**2

    def bound_backward(self, last_lA, last_uA, x):
        x_L, x_U = x.lower, x.upper
        upper_k = x_U + x_L
        upper_b = x_L**2 - upper_k * x_L
        if last_uA is not None:
            # Special case if we only want the upper bound with non-negative
            # coefficients.
            if last_uA.min() >= 0:
                uA = last_uA * upper_k
                ubias = self.get_bias(last_uA, upper_b)
            else:
                raise NotImplementedError
        else:
            uA, ubias = None, 0
        if last_lA is not None:
            if last_lA.max() <= 0:
                lA = last_lA * upper_k
                lbias = self.get_bias(last_lA, upper_b)
            else:
                raise NotImplementedError
        else:
            lA, lbias = None, 0
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        lower = ((h_U < 0) * (h_U**2) + (h_L > 0) * (h_L**2))
        upper = torch.max(h_L**2, h_U**2)
        return lower, upper
