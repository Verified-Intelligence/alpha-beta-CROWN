""" Activation operators or other unary nonlinear operators"""
from .base import *

class BoundActivation(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True
        self.relaxed = False

    def _init_masks(self, x):
        self.mask_pos = torch.ge(x.lower, 0).to(torch.float)
        self.mask_neg = torch.le(x.upper, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg

    def _init_linear(self, x, dim_opt=None):
        self._init_masks(x)
        self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def _add_linear(self, mask, type, k, x0, y0):
        if mask is None:
            mask = 1
        if type == 'lower':
            w_out, b_out = self.lw, self.lb
        else:
            w_out, b_out = self.uw, self.ub
        w_out += mask * k
        b_out += mask * (-x0 * k + y0)

    def bound_relax(self, x):
        return not_implemented_op(self, 'bound_relax')
    
    def interval_propagate(self, *v):
        return self.default_interval_propagate(*v)    

    def bound_backward(self, last_lA, last_uA, x):
        if not self.relaxed:
            self._init_linear(x)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = (
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0), 
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0))
            else:
                w_pos, b_pos, w_neg, b_neg = (
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0), 
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0))
            if self.batch_dim == 0:
                _A = last_A.clamp(min=0) * w_pos + last_A.clamp(max=0) * w_neg
                _bias = last_A.clamp(min=0) * b_pos + last_A.clamp(max=0) * b_neg
                if _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=list(range(2, _bias.ndim)))
            elif self.batch_dim == -1:
                mask = torch.gt(last_A, 0.).to(torch.float)
                _A = last_A * (mask * w_pos.unsqueeze(1) +
                                (1 - mask) * w_neg.unsqueeze(1))
                _bias = last_A * (mask * b_pos.unsqueeze(1) +
                        (1 - mask) * b_neg.unsqueeze(1))
                if _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=list(range(2, _bias.ndim)))
            else:
                raise NotImplementedError

            return _A, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, x):
        if not self.relaxed:
            self._init_linear(x)
            self.bound_relax(x)

        if self.lw.ndim > 0:
            if x.lw is not None:
                lw = self.lw.unsqueeze(1).clamp(min=0) * x.lw + \
                     self.lw.unsqueeze(1).clamp(max=0) * x.uw
                uw = self.uw.unsqueeze(1).clamp(max=0) * x.lw + \
                     self.uw.unsqueeze(1).clamp(min=0) * x.uw
            else:
                lw = uw = None
        else:
            if x.lw is not None:
                lw = self.lw.unsqueeze(0).clamp(min=0) * x.lw + \
                     self.lw.unsqueeze(0).clamp(max=0) * x.uw
                uw = self.uw.unsqueeze(0).clamp(min=0) * x.lw + \
                     self.uw.unsqueeze(0).clamp(max=0) * x.uw
            else:
                lw = uw = None
        lb = self.lw.clamp(min=0) * x.lb + self.lw.clamp(max=0) * x.ub + self.lb
        ub = self.uw.clamp(max=0) * x.lb + self.uw.clamp(min=0) * x.ub + self.ub

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        return self.forward(h_L), self.forward(h_U)


class BoundOptimizableActivation(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        # Two stages: `init` (initializing parameters) and `opt` (optimizing parameters).
        # If `None`, it means activation optimization is currently not used.
        self.opt_stage = None

    """ Enter the stage for initializing bound optimization. Optimized bounds are not used in 
        this stage. """
    def opt_init(self):
        self.opt_stage = 'init'

    """ Start optimizing bounds """
    def opt_start(self):
        self.opt_stage = 'opt'

    """ start_nodes: a list of starting nodes [(node, size)] during 
    CROWN backward bound propagation"""
    def init_opt_parameters(self, start_nodes):
        raise NotImplementedError

    def _init_linear(self, x, dim_opt=None):
        self._init_masks(x)
        # The first dimension of size 2 is used for lA and uA respectively,
        # when computing intermediate bounds.
        if self.opt_stage == 'opt' and dim_opt:
            self.lw = torch.zeros(2, dim_opt, *x.lower.shape).to(x.lower)     
        else:
            self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()        

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        self._start = start_node

        if self.opt_stage != 'opt':
            return super().bound_backward(last_lA, last_uA, x)
        assert self.batch_dim == 0            

        if not self.relaxed:
            self._init_linear(x, dim_opt=start_shape)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = self.lw[0], self.lb[0], self.uw[0], self.ub[0]
            else:
                w_pos, b_pos, w_neg, b_neg = self.uw[1], self.ub[1], self.lw[1], self.lb[1]
            _A = last_A.clamp(min=0) * w_pos + last_A.clamp(max=0) * w_neg
            _bias = last_A.clamp(min=0) * b_pos + last_A.clamp(max=0) * b_neg 
            if _bias.ndim > 2:
                _bias = torch.sum(_bias, list(range(2, _bias.ndim)))
            return _A, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def _no_bound_parameters(self):
        raise AttributeError('Bound parameters have not been initialized.'
                            'Please call `compute_bounds` with `method=CROWN-optimized`'
                            ' at least once.')

class BoundLeakyRelu(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True
        self.options = options.get('relu')
        self.alpha = attr['alpha']

    @Bound.save_io_shape
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, start_shape=None):
        if x is not None:
            lb_r = x.lower.clamp(max=0)
            ub_r = x.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = (ub_r - self.alpha * lb_r) / (ub_r - lb_r)
        upper_b = - lb_r * upper_d + self.alpha * lb_r

        if self.options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float() + (upper_d < 1.0).float() * self.alpha
        elif self.options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float() + (upper_d <= 0.0).float() * self.alpha
        else:
            lower_d = (upper_d > 0.5).float() + (upper_d <= 0.5).float() * self.alpha

        upper_d = upper_d.unsqueeze(0)
        lower_d = lower_d.unsqueeze(0)
        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias = self.get_bias(pos_uA, upper_b)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias = self.get_bias(neg_lA, upper_b)
        return [(lA, uA)], lbias, ubias


class BoundRelu(BoundOptimizableActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.options = options
        self.relu_options = options.get('relu', 'adaptive')
        self.beta = self.beta_mask = self.masked_beta = self.sparse_beta = None
        self.split_beta_used = False
        self.history_beta_used = False
        self.flattened_nodes = None
        # Save patches size for each output node.
        self.patch_size = {}

    def init_opt_parameters(self, start_nodes):
        self.alpha = OrderedDict()
        ref = self.inputs[0].lower # a reference variable for getting the shape
        for ns, size_s in start_nodes:
            self.alpha[ns] = torch.empty([2, size_s, ref.size(0), *self.shape], 
                dtype=torch.float, device=ref.device, requires_grad=True)
        for k, v in self.alpha.items():
            v.data.copy_(self.lower_d.data)  # Initial from adaptive lower bounds.    

    @Bound.save_io_shape
    def forward(self, x):
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return F.relu(x)

    # Linear relaxation for nonlinear functions
    # Used for forward mode bound propagation
    def bound_relax(self, x):
        # FIXME maybe avoid using `mask` which looks inefficient
        # m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)
        self._add_linear(mask=self.mask_neg, type='lower',
                         k=torch.zeros_like(x.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_neg, type='upper',
                         k=torch.zeros_like(x.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='lower',
                         k=torch.ones_like(x.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='upper',
                         k=torch.ones_like(x.lower), x0=0, y0=0)
        upper = torch.max(x.upper, x.lower + 1e-8)
        delta = 1e-8
        r = (x.upper - x.lower).clamp(min=delta)
        upper_k = x.upper / r + delta / r
        self._add_linear(mask=self.mask_both, type='upper',
                         k=upper_k, x0=x.lower, y0=0)
        if self.relu_options == "same-slope":
            lower_k = upper_k
        elif self.relu_options == "zero-lb":
            lower_k = torch.zeros_like(upper_k)
        elif self.relu_options == "one-lb":
            lower_k = torch.ones_like(upper_k)
        elif self.opt_stage == 'opt':
            # Each actual alpha in the forward mode has shape (batch_size, *relu_node_shape]. 
            # But self.alpha has shape (2, output_shape, batch_size, *relu_node_shape]
            # and we do not need its first two dimensions.
            lower_k = alpha = self.alpha['_forward'][0, 0]
        else:
            # adaptive
            lower_k = torch.gt(torch.abs(x.upper), torch.abs(x.lower)).to(torch.float)
        # NOTE #FIXME Saved for initialization bounds for optimization.
        # In the backward mode, same-slope bounds are used.
        # But here it is using adaptive bounds which seem to be better
        # for nn4sys benchmark with loose input bounds. Need confirmation 
        # for other cases.
        self.d = lower_k.detach() # saved for initializing optimized bounds           
        self._add_linear(mask=self.mask_both, type='lower',
                         k=lower_k, x0=0., y0=0.)

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, beta_for_intermediate_layers=False, unstable_idx=None):
        if x is not None:
            # # only set lower and upper bound here when using neuron set version, ie, not ob_update_by_layer
            # if self.beta is not None and not self.options.get('optimize_bound_args', {}).get('ob_update_by_layer', False):
            #     if self.beta_mask.abs().sum() != 0:
            #         # set bound neuron-wise according to beta_mask
            #         x.lower = x.lower * (self.beta_mask != 1).to(torch.float32)
            #         x.upper = x.upper * (self.beta_mask != -1).to(torch.float32)

            lb_r = x.lower.clamp(max=0)
            ub_r = x.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)

        self.I = ((lb_r != 0) * (ub_r != 0)).detach()  # unstable neurons
        # print('unstable neurons:', self.I.sum())

        if hasattr(x, 'interval') and Interval.use_relative_bounds(x.interval):
            diff_x = x.interval.upper_offset - x.interval.lower_offset
            upper_d = (self.interval.upper_offset - self.interval.lower_offset) / diff_x.clamp(min=epsilon)
            mask_tiny_diff = (diff_x <= epsilon).float()
            upper_d = mask_tiny_diff * F.relu(x.upper) + (1 - mask_tiny_diff) * upper_d
        else:
            ub_r = torch.max(ub_r, lb_r + 1e-8)
            upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d

        flag_expand = False
        ub_lower_d = lb_lower_d = None
        if self.relu_options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.relu_options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.relu_options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        elif self.relu_options == "reversed-adaptive":
            lower_d = (upper_d < 0.5).float()
        elif self.opt_stage == 'opt':
            # Alpha-CROWN.
            lower_d = None
            # Each alpha has shape (2, output_shape, batch_size, *relu_node_shape].
            # If slope is shared, output_shape will be 1.
            if unstable_idx is not None and self.alpha[start_node.name].size(1) != 1:
                if isinstance(unstable_idx, tuple):
                    if isinstance(last_lA, torch.Tensor) or isinstance(last_uA, torch.Tensor):
                        # Patches mode converted to matrix. Need to select accross the spec dimension.
                        # For this node, since it is in matrix mode, the spec dimension is out_c * out_h * out_w
                        selected_alpha = self.alpha[start_node.name]
                        # Reshape the spec dimension to c*h*w so we can select based on unstable index.
                        selected_alpha = selected_alpha.view(-1, *start_node.output_shape[1:], *selected_alpha.shape[2:])
                        selected_alpha = selected_alpha[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]]
                    else:
                        # unstable index for patches mode. Need to choose based on unstable_idx.
                        # Selection is just based on output channel, and it will be selected when d_pos_unfolded_r and d_neg_unfolded_r are constructed.
                        selected_alpha = self.alpha[start_node.name]
                elif unstable_idx.ndim == 1:
                    # Only unstable neurons of the start_node neurons are used.
                    selected_alpha = self.non_deter_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                elif unstable_idx.ndim == 2:
                    # Each element in the batch selects different neurons.
                    selected_alpha = batched_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                else:
                    raise ValueError
            else:
                selected_alpha = self.alpha[start_node.name]
            # print(f'{self.name} selecting {start_node.name} alpha {selected_alpha.size()}')
            # The first dimension is lower/upper intermediate bound.
            if x is not None:
                lower = x.lower
                upper = x.upper
            else:
                lower = self.lower
                upper = self.upper
            lower_mask = lower > 0
            upper_mask = upper < 0
            if last_lA is not None:
                lb_lower_d = selected_alpha[0].clamp(min=0.0, max=1.0)
                lb_lower_d[:, lower_mask] = 1.0
                lb_lower_d[:, upper_mask] = 0.0
            if last_uA is not None:
                ub_lower_d = selected_alpha[1].clamp(min=0.0, max=1.0)
                ub_lower_d[:, lower_mask] = 1.0
                ub_lower_d[:, upper_mask] = 0.0
            self.zero_backward_coeffs_l = self.zero_backward_coeffs_u = upper_mask.all().item()
            flag_expand = True
        else:
            # adaptive
            lower_d = (upper_d > 0.5).float()

        # save for calculate babsr score
        self.d = upper_d
        self.lA = last_lA
        # Save for initialization bounds.
        self.lower_d = lower_d

        # assert self.I.sum() == torch.logical_and(0 < self.d, self.d < 1).sum()

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        upper_d = upper_d.unsqueeze(0)
        upper_b = upper_b.unsqueeze(0)
        if not flag_expand:
            if self.opt_stage == 'opt':
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)

        mode = "patches" if isinstance(last_lA, Patches) or isinstance(last_uA, Patches) else "matrix"

        # In patches mode, we need to unfold lower and upper slopes. In matrix mode we simply return.
        def _maybe_unfold(d_tensor, last_A):
            if mode == "matrix" or d_tensor is None or last_A is None:
                return d_tensor
            # Input are slopes with shape (spec, batch, input_c, input_h, input_w)
            # Here spec is the same as out_c.
            assert d_tensor.ndim == 5
            d_shape = d_tensor.size()
            # Reshape to 4-D tensor to unfold.
            d_tensor = d_tensor.view(-1, *d_shape[-3:])
            # unfold the slope matrix as patches. Patch shape is [spec * batch, out_h, out_w, in_c, H, W).
            d_unfolded = inplace_unfold(d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding)
            # Reshape to (spec, batch, out_h, out_w, in_c, H, W); here spec_size is out_c.
            d_unfolded_r = d_unfolded.view(*d_shape[:-3], *d_unfolded.shape[1:])
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

        # Choose upper or lower bounds based on the sign of last_A
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0

            if type(last_A) == torch.Tensor:
                # multiply according to sign of A (we use fused operation to save memory)
                # neg_A = last_A.clamp(max=0)
                # pos_A = last_A.clamp(min=0)
                # A = d_pos * pos_A + d_neg * neg_A
                A, pos_A, neg_A = self.clamp_mutiply(last_A, d_pos, d_neg)
                bias = 0
                if b_pos is not None:
                    bias = bias + torch.einsum('sb...,sb...->sb', pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + torch.einsum('sb...,sb...->sb', neg_A, b_neg)
                return A, bias
            elif type(last_A) == Patches:
                # if last_A is not an identity matrix
                assert last_A.identity == 0
                if last_A.identity == 0:
                    # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W]. Here out_c is the spec dimension.
                    # or (unstable_size, batch_size, in_c, H, W) when it is sparse.
                    patches = last_A.patches
                    prod, pos_A_patches, neg_A_patches = self.clamp_mutiply_non_contiguous(patches, d_pos, d_neg)
                    # prod has shape [out_c, batch_size, out_h, out_w, in_c, H, W] or (unstable_size, batch_size, in_c, H, W) when it is sparse.

                    # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                    if start_node is not None:
                        if last_A.unstable_idx is not None:
                            # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                            self.patch_size[start_node.name] = [last_A.output_shape[1], prod.size(1), last_A.output_shape[2], last_A.output_shape[3], prod.size(-3), prod.size(-2), prod.size(-1)]
                        else:
                            # Regular patches.
                            self.patch_size[start_node.name] = prod.size()

                    bias = 0
                    if b_pos is not None:
                        # For sparse patches the return bias size is (unstable_size, batch).
                        # For regular patches the return bias size is (spec, batch, out_h, out_w).
                        bias = bias + torch.einsum('sb...chw,sb...chw->sb...', b_pos, pos_A_patches)
                    if b_neg is not None:
                        bias = bias + torch.einsum('sb...chw,sb...chw->sb...', b_neg, neg_A_patches)
                    return Patches(prod, last_A.stride, last_A.padding, prod.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape), bias

        # In patches mode we might need an unfold.
        upper_d = _maybe_unfold(upper_d, last_lA if last_lA is not None else last_uA)
        lower_d = _maybe_unfold(lower_d, last_lA if last_lA is not None else last_uA)
        upper_b = _maybe_unfold(upper_b, last_lA if last_lA is not None else last_uA)
        ub_lower_d = _maybe_unfold(ub_lower_d, last_uA)
        lb_lower_d = _maybe_unfold(lb_lower_d, last_lA)

        uA, ubias = _bound_oneside(last_uA, upper_d, ub_lower_d if lower_d is None else lower_d, upper_b, None)
        lA, lbias = _bound_oneside(last_lA, lb_lower_d if lower_d is None else lower_d, upper_d, None, upper_b)

        self.masked_beta_lower = self.masked_beta_upper = None
        if self.options.get('optimize_bound_args', {}).get('ob_beta', False):
            if self.options.get('optimize_bound_args', {}).get('ob_single_node_split', False):
                # Beta-CROWN.
                A = last_uA if last_uA is not None else last_lA
                if type(A) is Patches:
                    # For patches mode, masked_beta will be used; sparse beta is not supported.
                    self.masked_beta = (self.beta[0] * self.beta_mask).requires_grad_()
                    A_patches = A.patches
                    # unfold the beta as patches, size (batch, out_h, out_w, in_c, H, W)
                    masked_beta_unfolded = inplace_unfold(self.masked_beta, kernel_size=A_patches.shape[-2:], padding=A.padding, stride=A.stride)
                    if A.unstable_idx is not None:
                        masked_beta_unfolded = masked_beta_unfolded.permute(1, 2, 0, 3, 4)
                        # After selection, the shape is (unstable_size, batch, in_c, H, W).
                        masked_beta_unfolded = masked_beta_unfolded[A.unstable_idx[1], A.unstable_idx[2]]
                    else:
                        # Add the spec (out_c) dimension.
                        masked_beta_unfolded = masked_beta_unfolded.unsqueeze(0)
                    if uA is not None:
                        uA = Patches(uA.patches + masked_beta_unfolded, uA.stride, uA.padding, uA.patches.shape, unstable_idx=uA.unstable_idx, output_shape=uA.output_shape)
                    if lA is not None:
                        lA = Patches(lA.patches - masked_beta_unfolded, lA.stride, lA.padding, lA.patches.shape, unstable_idx=lA.unstable_idx, output_shape=lA.output_shape)
                elif type(A) is torch.Tensor:
                    # For matrix mode, beta is sparse.
                    beta_values = (self.sparse_beta * self.sparse_beta_sign).expand(lA.size(0), -1, -1)
                    # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
                    beta_indices = self.sparse_beta_loc.unsqueeze(0).expand(lA.size(0), -1, -1)
                    # For conv layer, the last dimension is flattened in indices.
                    prev_size = A.size()
                    if uA is not None:
                        uA = self.non_deter_scatter_add(uA.view(uA.size(0), uA.size(1), -1), dim=2, index=beta_indices, src=beta_values)
                        uA = uA.view(prev_size)
                    if lA is not None:
                        lA = self.non_deter_scatter_add(lA.view(lA.size(0), lA.size(1), -1), dim=2, index=beta_indices, src=beta_values.neg())
                        lA = lA.view(prev_size)
                else:
                    raise RuntimeError(f"Unknown type {type(A)} for A")
            # The code block below is for debugging and will be removed (until the end of this function).
            elif not self.options.get('optimize_bound_args', {}).get('ob_single_node_split', True):
                A = uA if uA is not None else lA
                if type(A) == torch.Tensor:
                    device = A.device
                else:
                    device = A.patches.device
                print_time = False

                if self.single_beta_used or self.split_beta_used or self.history_beta_used:
                    start_time = time.time()
                    history_compute_time, split_compute_time, split_convert_time = 0, 0, 0
                    history_compute_time1, history_compute_time2 = 0, 0
                    # assert len(self.split_beta) > 0, "split_beta_used or history_beta_used is True means there have to be one relu in one batch is used in split constraints"
                    if self.single_beta_used:
                        if beta_for_intermediate_layers:
                            # We handle the refinement of intermediate layer after this split layer here. (the refinement for intermediate layers before the split is handled in compute_bounds().
                            # print(f'single node beta for {start_node.name} with beta shape {self.single_intermediate_betas[start_node.name]["ub"].size()}')
                            assert not self.history_beta_used
                            assert not self.history_beta_used
                            assert type(A) is not Patches
                            if uA is not None:
                                # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
                                single_intermediate_beta = self.single_intermediate_betas[start_node.name]['ub']
                                single_intermediate_beta = single_intermediate_beta.view(
                                    single_intermediate_beta.size(0), -1, single_intermediate_beta.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    single_intermediate_beta = self.non_deter_index_select(single_intermediate_beta, index=unstable_idx, dim=1)
                                # This is the sign.
                                single_intermediate_beta = single_intermediate_beta * self.single_beta_sign.unsqueeze(1)
                                # We now generate a large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
                                prev_size = uA.size()
                                # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
                                indices = self.single_beta_loc.unsqueeze(0).expand(uA.size(0), -1, -1)
                                # We update uA here directly using sparse operation. Note the spec dimension is at the first!
                                uA = self.non_deter_scatter_add(uA.view(uA.size(0), uA.size(1), -1), dim=2, index=indices, src=single_intermediate_beta.transpose(0,1))
                                uA = uA.view(prev_size)
                            if lA is not None:
                                # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
                                single_intermediate_beta = self.single_intermediate_betas[start_node.name]['lb']
                                single_intermediate_beta = single_intermediate_beta.view(
                                    single_intermediate_beta.size(0), -1, single_intermediate_beta.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    single_intermediate_beta = self.non_deter_index_select(single_intermediate_beta, index=unstable_idx, dim=1)
                                # This is the sign, for lower bound we need to negate.
                                single_intermediate_beta = single_intermediate_beta * ( - self.single_beta_sign.unsqueeze(1))
                                # We now generate a large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
                                prev_size = lA.size()
                                # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
                                indices = self.single_beta_loc.unsqueeze(0).expand(lA.size(0), -1, -1)
                                # We update lA here directly using sparse operation. Note the spec dimension is at the first!
                                lA = self.non_deter_scatter_add(lA.view(lA.size(0), lA.size(1), -1), dim=2, index=indices, src=single_intermediate_beta.transpose(0,1))
                                lA = lA.view(prev_size)
                        else:
                            self.masked_beta_lower = self.masked_beta_upper = self.masked_beta = self.beta * self.beta_mask

                    ############################
                    # sparse_coo version for history coeffs
                    if self.history_beta_used:
                        # history_compute_time = time.time()
                        if beta_for_intermediate_layers:
                            # print(f'history intermediate beta for {start_node.name} with beta shape {self.history_intermediate_betas[start_node.name]["ub"].size()}')
                            if uA is not None:
                                # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
                                history_intermediate_beta = self.history_intermediate_betas[start_node.name]['ub']
                                history_intermediate_beta = history_intermediate_beta.view(
                                    history_intermediate_beta.size(0), -1, history_intermediate_beta.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    history_intermediate_beta = self.non_deter_index_select(history_intermediate_beta, index=unstable_idx, dim=1)
                                # new_history_coeffs has shape (batch, prod(nodes), n_max_history_beta)
                                # new_history_c has shape (batch, n_max_history_beta)
                                # This can generate a quite large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
                                self.masked_beta_upper = torch.bmm(history_intermediate_beta, (
                                            self.new_history_coeffs * self.new_history_c.unsqueeze(1)).transpose(-1,
                                                                                                                 -2))
                            if lA is not None:
                                history_intermediate_beta = self.history_intermediate_betas[start_node.name]['lb']
                                history_intermediate_beta = history_intermediate_beta.view(
                                    history_intermediate_beta.size(0), -1, history_intermediate_beta.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    history_intermediate_beta = self.non_deter_index_select(history_intermediate_beta, index=unstable_idx, dim=1)
                                self.masked_beta_lower = torch.bmm(history_intermediate_beta, (
                                            self.new_history_coeffs * self.new_history_c.unsqueeze(1)).transpose(-1,
                                                                                                                 -2))
                        else:
                            # new_history_coeffs has shape (batch, prod(nodes), n_max_history_beta)
                            # new_history_beta has shape (batch, m_max_history_beta)
                            self.masked_beta_lower = self.masked_beta_upper = torch.bmm(self.new_history_coeffs, (
                                        self.new_history_beta * self.new_history_c).unsqueeze(-1)).squeeze(-1)

                    # new split constraint
                    if self.split_beta_used:
                        split_convert_time = time.time()
                        if self.split_coeffs["dense"] is None:
                            assert not hasattr(self, 'split_intermediate_betas')  # intermediate beta split must use the dense mode.
                            ##### we can use repeat to further save the conversion time
                            # since the new split constraint coeffs can be optimized, we can just save the index and assign optimized coeffs value to the sparse matrix
                            self.new_split_coeffs = torch.zeros(self.split_c.size(0), self.flattened_nodes,
                                                                dtype=torch.get_default_dtype(), device=device)
                            # assign coeffs value to the first half batch
                            self.new_split_coeffs[
                                (self.split_coeffs["nonzero"][:, 0], self.split_coeffs["nonzero"][:, 1])] = \
                            self.split_coeffs["coeffs"]
                            # # assign coeffs value to the rest half batch with the same values since split constraint shared the same coeffs for >0/<0
                            self.new_split_coeffs[(self.split_coeffs["nonzero"][:, 0] + int(self.split_c.size(0) / 2),
                                                   self.split_coeffs["nonzero"][:, 1])] = self.split_coeffs["coeffs"]
                        else:
                            # batch = int(self.split_c.size(0)/2)
                            # assign coeffs value to the first half batch and the second half batch
                            self.new_split_coeffs = self.split_coeffs["dense"].repeat(2, 1)
                        split_convert_time = time.time() - split_convert_time
                        split_compute_time = time.time()
                        if beta_for_intermediate_layers:
                            assert hasattr(self, 'split_intermediate_betas')
                            # print(f'split intermediate beta for {start_node.name} with beta shape {self.split_intermediate_betas[start_node.name]["ub"].size()}')
                            if uA is not None:
                                # upper bound betas for this set of intermediate neurons.
                                # Make an extra spec dimension. Now new_split_coeffs has size (batch, specs, #nodes). Specs is the number of intermediate neurons of start node. The same split will be applied to all specs in a batch element.
                                # masked_beta_upper has shape (batch, spec, #nodes)
                                split_intermediate_betas = self.split_intermediate_betas[start_node.name]['ub']
                                split_intermediate_betas = split_intermediate_betas.view(split_intermediate_betas.size(0), -1, split_intermediate_betas.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    split_intermediate_betas = self.non_deter_index_select(split_intermediate_betas, index=unstable_idx, dim=1)
                                self.split_masked_beta_upper = split_intermediate_betas * (
                                            self.new_split_coeffs * self.split_c).unsqueeze(1)
                            if lA is not None:
                                split_intermediate_betas = self.split_intermediate_betas[start_node.name]['lb']
                                split_intermediate_betas = split_intermediate_betas.view(split_intermediate_betas.size(0), -1, split_intermediate_betas.size(-1))
                                if unstable_idx is not None:
                                    # Only unstable neurons of the start_node neurons are used.
                                    split_intermediate_betas = self.non_deter_index_select(split_intermediate_betas, index=unstable_idx, dim=1)
                                self.split_masked_beta_lower = split_intermediate_betas * (
                                            self.new_split_coeffs * self.split_c).unsqueeze(1)
                        else:
                            # beta for final objective only. TODO: distinguish between lb and ub.
                            self.split_masked_beta_upper = self.split_masked_beta_lower = self.new_split_coeffs * (
                                        self.split_beta * self.split_c)
                        # add the new split constraint beta to the masked_beta
                        if self.masked_beta_upper is None:
                            self.masked_beta_upper = self.split_masked_beta_upper
                        else:
                            self.masked_beta_upper = self.masked_beta_upper + self.split_masked_beta_upper

                        if self.masked_beta_lower is None:
                            self.masked_beta_lower = self.split_masked_beta_lower
                        else:
                            self.masked_beta_lower = self.masked_beta_lower + self.split_masked_beta_lower
                        # For backwards compatibility - we originally only have one beta.
                        self.masked_beta = self.masked_beta_lower
                        split_compute_time = time.time() - split_compute_time

                    A = last_uA if last_uA is not None else last_lA
                    if type(A) is Patches:
                        assert not hasattr(self, 'split_intermediate_betas')
                        assert not hasattr(self, 'single_intermediate_betas')
                        A_patches = A.patches
                        # Reshape beta to image size.
                        self.masked_beta = self.masked_beta.view(self.masked_beta.size(0), *ub_r.size()[1:])
                        # unfold the beta as patches, size (batch, out_h, out_w, in_c, H, W)
                        masked_beta_unfolded = inplace_unfold(self.masked_beta, kernel_size=A_patches.shape[-2:], padding=A.padding, stride=A.stride)
                        if A.unstable_idx is not None:
                            masked_beta_unfolded = masked_beta_unfolded.permute(1, 2, 0, 3, 4)
                            # After selection, the shape is (unstable_size, batch, in_c, H, W).
                            masked_beta_unfolded = masked_beta_unfolded[A.unstable_idx[1], A.unstable_idx[2]]
                        else:
                            # Add the spec (out_c) dimension.
                            masked_beta_unfolded = masked_beta_unfolded.unsqueeze(0)
                        if uA is not None:
                            uA = Patches(uA.patches + masked_beta_unfolded, uA.stride, uA.padding, uA.patches.shape, unstable_idx=uA.unstable_idx, output_shape=uA.output_shape)
                        if lA is not None:
                            lA = Patches(lA.patches - masked_beta_unfolded, lA.stride, lA.padding, lA.patches.shape, unstable_idx=lA.unstable_idx, output_shape=lA.output_shape)
                    elif type(A) is torch.Tensor:
                        if uA is not None:
                            # print("uA", uA.shape, self.masked_beta.shape)
                            # uA/lA has shape (spec, batch, *nodes)
                            if beta_for_intermediate_layers:
                                if not self.single_beta_used:
                                    # masked_beta_upper has shape (batch, spec, #nodes)
                                    self.masked_beta_upper = self.masked_beta_upper.transpose(0, 1)
                                    self.masked_beta_upper = self.masked_beta_upper.view(self.masked_beta_upper.size(0),
                                                                                         self.masked_beta_upper.size(1),
                                                                                         *uA.shape[2:])
                            else:
                                # masked_beta_upper has shape (batch, #nodes)
                                self.masked_beta_upper = self.masked_beta_upper.reshape(uA[0].shape).unsqueeze(0)
                            if not self.single_beta_used or not beta_for_intermediate_layers:
                                # For intermediate layer betas witn single node split, uA has been modified above.
                                uA = uA + self.masked_beta_upper
                        if lA is not None:
                            # print("lA", lA.shape, self.masked_beta.shape)
                            if beta_for_intermediate_layers:
                                if not self.single_beta_used:
                                    # masked_beta_upper has shape (batch, spec, #nodes)
                                    self.masked_beta_lower = self.masked_beta_lower.transpose(0, 1)
                                    self.masked_beta_lower = self.masked_beta_lower.view(self.masked_beta_lower.size(0),
                                                                                         self.masked_beta_lower.size(1),
                                                                                         *lA.shape[2:])
                            else:
                                # masked_beta_upper has shape (batch, #nodes)
                                self.masked_beta_lower = self.masked_beta_lower.reshape(lA[0].shape).unsqueeze(0)
                            if not self.single_beta_used or not beta_for_intermediate_layers:
                                # For intermediate layer betas witn single node split, lA has been modified above.
                                lA = lA - self.masked_beta_lower
                    else:
                        raise RuntimeError(f"Unknown type {type(A)} for A")
                    # print("total:", time.time()-start_time, history_compute_time1, history_compute_time2, split_convert_time, split_compute_time)

        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            nominal = F.relu(v[0].nominal)
            mask_nominal = (nominal > 0).float()
            mask_l = (v[0].lower > 0).float()
            mask_u = (v[0].upper > 0).float()
            lower_offset = mask_nominal * (mask_l * v[0].lower_offset + (1 - mask_l) * (-nominal))
            upper_offset = mask_nominal * v[0].upper_offset + (1 - mask_nominal) * mask_u * v[0].upper
            return Interval(None, None, nominal, lower_offset, upper_offset)

        h_L, h_U = v[0][0], v[0][1]

        return F.relu(h_L), F.relu(h_U)

    def bound_forward(self, dim_in, x):
        return super().bound_forward(dim_in, x)

class BoundSqrt(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.sqrt(x)

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            nominal = self.forward(v[0].nominal)
            lower_offset = self.forward(v[0].nominal + v[0].lower_offset) - nominal
            upper_offset = self.forward(v[0].nominal + v[0].upper_offset) - nominal
            return Interval(None, None, nominal, lower_offset, upper_offset)            

        return super().interval_propagate(*v)

class BoundReciprocal(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.reciprocal(x)

    def bound_relax(self, x):
        m = (x.lower + x.upper) / 2
        kl = -1 / m.pow(2)
        self._add_linear(mask=None, type='lower', k=kl, x0=m, y0=1. / m)
        ku = -1. / (x.lower * x.upper)
        self._add_linear(mask=None, type='upper', k=ku, x0=x.lower, y0=1. / x.lower)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0].float(), v[0][1].float()
        assert h_L.min() > 0, 'Only positive values are supported in BoundReciprocal'
        return torch.reciprocal(h_U), torch.reciprocal(h_L)


class BoundSin(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.max_point = math.pi / 2
        self.min_point = math.pi * 3 / 2

    @Bound.save_io_shape
    def forward(self, x):
        return torch.sin(x)

    def interval_propagate(self, *v):
        # Check if a point is in [l, u], considering the 2pi period
        def check_crossing(ll, uu, point):
            return ((((uu - point) / (2 * math.pi)).floor() - ((ll - point) / (2 * math.pi)).floor()) > 0).float()
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

    def bound_backward(self, last_lA, last_uA, *x, start_node=None, start_shape=None):
        return not_implemented_op(self, 'bound_backward')


class BoundCos(BoundSin):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.max_point = 0.0
        self.min_point = math.pi

    @Bound.save_io_shape
    def forward(self, x):
        return torch.cos(x)



class BoundTanh(BoundOptimizableActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.precompute_relaxation('tanh', torch.tanh, self.dtanh)

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {} 
        self.tp_both_upper_init = {}

    def init_opt_parameters(self, start_nodes):
        self.alpha = OrderedDict()
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = l.shape
        for ns, size_s in start_nodes:
            self.alpha[ns] = torch.empty(4, 2, size_s, *shape, device=l.device)
            self.alpha[ns].data[:2] = ((l + u) / 2).unsqueeze(0).expand(2, 2, size_s, *shape)
            self.alpha[ns].data[2] = self.tp_both_lower_init[ns].expand(2, size_s, *shape)
            self.alpha[ns].data[3] = self.tp_both_upper_init[ns].expand(2, size_s, *shape)

    def dtanh(self, x):
        # to avoid bp error when cosh is too large
        # cosh(25.0)**2 > 1e21
        mask = torch.lt(torch.abs(x), 25.0).float()
        cosh = torch.cosh(mask * x + 1 - mask)
        return mask * (1. / cosh.pow(2))

    """Precompute relaxation parameters for tanh and sigmoid"""

    @torch.no_grad()
    def precompute_relaxation(self, name, func, dfunc):
        self.x_limit = 500
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for {}'.format(name))

        def check_lower(upper, d):
            k = dfunc(d)
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            k = dfunc(d)
            return k * (lower - d) + func(d) >= func(lower)

        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        r = torch.zeros_like(upper)
        l = -torch.ones_like(upper)
        while True:
            checked = check_lower(upper, l).int()
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel(): 
                break
        for t in range(max_iter):
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        self.d_lower = l.clone()

        lower = -self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        l = torch.zeros_like(upper)
        r = torch.ones_like(upper)
        while True:
            checked = check_upper(lower, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel(): 
                break
        for t in range(max_iter):
            m = (l + r) / 2
            checked = check_upper(lower, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        self.d_upper = r.clone()

        logger.debug('Done')

    @Bound.save_io_shape
    def forward(self, x):
        return torch.tanh(x)

    def bound_relax_impl(self, x, func, dfunc):
        # When self.x_limit is large enough, torch.tanh(self.x_limit)=1, 
        # and thus clipping is valid
        lower = x.lower.clamp(min=-self.x_limit)
        upper = x.upper.clamp(max=self.x_limit)
        y_l, y_u = func(lower), func(upper)

        min_preact = 1e-6
        mask_close = (upper - lower) < min_preact
        k_direct = k = torch.where(mask_close, 
            dfunc(upper), (y_u - y_l) / (upper - lower).clamp(min=min_preact))

        # Fixed bounds that cannot be optimized
        # upper bound for negative
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=lower, y0=y_l)
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=lower, y0=y_l)

        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            (upper / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_lower = torch.index_select(self.d_lower, 0, index).view(lower.shape)

        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            (lower / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_upper = torch.index_select(self.d_upper, 0, index).view(upper.shape)           

        ns = self._start.name

        # bound with tangent lines can be optimized
        if self.opt_stage == 'opt':
            if not hasattr(self, 'alpha'):
                self._no_bound_parameters()

            # Clipping is done here rather than after `opt.step()` call
            # because it depends on pre-activation bounds   
            self.alpha[ns].data[0, :] = torch.max(torch.min(self.alpha[ns][0, :], upper), lower)
            self.alpha[ns].data[1, :] = torch.max(torch.min(self.alpha[ns][1, :], upper), lower)
            self.alpha[ns].data[2, :] = torch.min(self.alpha[ns][2, :], d_lower)
            self.alpha[ns].data[3, :] = torch.max(self.alpha[ns][3, :], d_upper)

            tp_pos = self.alpha[ns][0]
            tp_neg = self.alpha[ns][1]
            tp_both_lower = self.alpha[ns][2]
            tp_both_upper = self.alpha[ns][3] 

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            mask_direct = self.mask_both * ( k_direct < dfunc(lower) )
            self._add_linear(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self._add_linear(mask=self.mask_both - mask_direct, type='lower', 
                k=dfunc(tp_both_lower), x0=tp_both_lower, 
                y0=self.forward(tp_both_lower))

            mask_direct = self.mask_both * ( k_direct < dfunc(upper) )   
            self._add_linear(mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self._add_linear(mask=self.mask_both - mask_direct, type='upper', 
                k=dfunc(tp_both_upper), x0=tp_both_upper, 
                y0=self.forward(tp_both_upper))

            self._add_linear(mask=self.mask_neg, type='lower', 
                k=dfunc(tp_neg), x0=tp_neg, y0=self.forward(tp_neg))
            self._add_linear(mask=self.mask_pos, type='upper', 
                k=dfunc(tp_pos), x0=tp_pos, y0=self.forward(tp_pos))
        else:
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # lower bound for negative
            self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
            # upper bound for positive
            self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

            k = dfunc(d_lower)
            y0 = func(d_lower)
            if self.opt_stage == 'init':
                self.tp_both_lower_init[ns] = d_lower.detach()
            mask_direct = self.mask_both * ( k_direct < dfunc(lower) )
            self._add_linear(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)                
            self._add_linear(mask=self.mask_both - mask_direct, type='lower', k=k, x0=d_lower, y0=y0)

            k = dfunc(d_upper)
            y0 = func(d_upper)
            if self.opt_stage == 'init':
                self.tp_both_upper_init[ns] = d_upper.detach()  
                self.tmp_lower = x.lower.detach()
                self.tmp_upper = x.upper.detach()
            mask_direct = self.mask_both * ( k_direct < dfunc(upper) )
            self._add_linear(mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)                
            self._add_linear(mask=self.mask_both - mask_direct, type='upper', k=k, x0=d_upper, y0=y0)

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.tanh, self.dtanh)


class BoundSigmoid(BoundTanh):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super(BoundTanh, self).__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.precompute_relaxation('sigmoid', torch.sigmoid, self.dsigmoid)

    @Bound.save_io_shape
    def forward(self, x):
        return torch.sigmoid(x)

    def dsigmoid(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.sigmoid, self.dsigmoid)


class BoundSoftplus(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super(BoundSoftplus, self).__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.softplus = nn.Softplus()

    @Bound.save_io_shape
    def forward(self, x):
        return self.softplus(x) 


class BoundExp(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.options = options.get('exp')
        self.max_input = 0

    @Bound.save_io_shape
    def forward(self, x):
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return torch.exp(x - self.max_input)
        return torch.exp(x)

    def interval_propagate(self, *v):
        assert (len(v) == 1)

        if Interval.use_relative_bounds(*v):
            assert not self.loss_fusion or self.options == 'no-max-input'
            nominal = torch.exp(v[0].nominal)
            return Interval(
                None, None,
                nominal,
                nominal * (torch.exp(v[0].lower_offset) - 1),
                nominal * (torch.exp(v[0].upper_offset) - 1)
            )

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
        self._add_linear(mask=None, type='lower', k=k, x0=m, y0=exp_m)
        min_val = -1e9  # to avoid (-inf)-(-inf) when both input.lower and input.upper are -inf
        epsilon = 1e-20
        close = (u - l < epsilon).int()
        k = close * exp_u + (1 - close) * (exp_u - exp_l) / (u - l + epsilon)
        self._add_linear(mask=None, type='upper', k=k, x0=l, y0=exp_l)


class BoundLog(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].forward_value, dim=-1) 
        return torch.log(x.clamp(min=epsilon))

    def bound_relax(self, x):
        rl, ru = self.forward(x.lower), self.forward(x.upper)
        ku = (ru - rl) / (x.upper - x.lower + epsilon)
        self._add_linear(mask=None, type='lower', k=ku, x0=x.lower, y0=rl)
        m = (x.lower + x.upper) / 2
        k = torch.reciprocal(m)
        rm = self.forward(m)
        self._add_linear(mask=None, type='upper', k=k, x0=m, y0=rm)

    def interval_propagate(self, *v):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            par = self.inputs[0].inputs[0].inputs[0]
            if Interval.use_relative_bounds(*v):
                lower = torch.logsumexp(par.interval.nominal + par.interval.lower_offset, dim=-1) 
                upper = torch.logsumexp(par.interval.nominal + par.interval.upper_offset, dim=-1) 
                return Interval.make_interval(lower, upper, nominal=self.forward_value, use_relative=True)
            else:
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


class BoundPow(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x, y):
        return torch.pow(x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)

        if Interval.use_relative_bounds(*v):
            exp = v[1].nominal
            assert exp == int(exp)
            exp = int(exp)
            h_L = v[0].nominal + v[0].lower_offset
            h_U = v[0].nominal + v[0].upper_offset
            lower, upper = torch.pow(h_L, exp), torch.pow(h_U, exp)
            if exp % 2 == 0:
                lower, upper = torch.min(lower, upper), torch.max(lower, upper)
                mask = 1 - ((h_L < 0) * (h_U > 0)).float()
                lower = lower * mask
            return Interval.make_interval(lower, upper, nominal=self.forward_value, use_relative=True)

        exp = v[1][0]
        assert exp == int(exp)
        exp = int(exp)
        pl, pu = torch.pow(v[0][0], exp), torch.pow(v[0][1], exp)
        if exp % 2 == 1:
            return pl, pu
        else:
            pl, pu = torch.min(pl, pu), torch.max(pl, pu)
            mask = 1 - ((v[0][0] < 0) * (v[0][1] > 0)).float()
            return pl * mask, pu
