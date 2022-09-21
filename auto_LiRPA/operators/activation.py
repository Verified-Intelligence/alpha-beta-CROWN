""" Activation operators or other unary nonlinear operators"""
from typing import Optional, Tuple
import torch
from torch import Tensor
from .base import *
from .clampmult import multiply_by_A_signs
from .solver_utils import grb
from ..utils import unravel_index, logger, prod


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class BoundActivation(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [0]
        self.relaxed = False

    def _init_masks(self, x):
        self.mask_pos = x.lower >= 0
        self.mask_neg = x.upper <= 0
        self.mask_both = torch.logical_not(torch.logical_or(self.mask_pos, self.mask_neg))

    def init_linear_relaxation(self, x, dim_opt=None):
        self._init_masks(x)
        self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def add_linear_relaxation(self, mask, type, k, x0, y0):
        if type == 'lower':
            w_out, b_out = self.lw, self.lb
        else:
            w_out, b_out = self.uw, self.ub

        if mask is None:
            if isinstance(k, Tensor) and k.ndim > 0:
                w_out[:] = k
            else:
                w_out.fill_(k)
        else:
            if isinstance(k, Tensor):
                w_out[..., mask] = k[..., mask].to(w_out)
            else:
                w_out[..., mask] = k

        if (not isinstance(x0, Tensor) and x0 == 0
                and not isinstance(y0, Tensor) and y0 == 0):
            pass
        else:
            b = -x0 * k + y0
            if mask is None:
                if b.ndim > 0:
                    b_out[:] = b
                else:
                    b_out.fill_(b)
            else:
                b_out[..., mask] = b[..., mask]

    def bound_relax(self, x):
        return not_implemented_op(self, 'bound_relax')

    def interval_propagate(self, *v):
        return self.default_interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x):
        if not self.relaxed:
            self.init_linear_relaxation(x)
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
            w_pos = maybe_unfold_patches(w_pos, last_A)
            w_neg = maybe_unfold_patches(w_neg, last_A)
            b_pos = maybe_unfold_patches(b_pos, last_A)
            b_neg = maybe_unfold_patches(b_neg, last_A)
            if self.batch_dim == 0:
                _A, _bias = multiply_by_A_signs(last_A, w_pos, w_neg, b_pos, b_neg)
            elif self.batch_dim == -1:
                # FIXME: why this is different from above?
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

    @staticmethod
    @torch.jit.script
    def bound_forward_w(
            relax_lw: Tensor, relax_uw: Tensor, x_lw: Tensor, x_uw: Tensor, dim: int):
        lw = (relax_lw.unsqueeze(dim).clamp(min=0) * x_lw +
              relax_lw.unsqueeze(dim).clamp(max=0) * x_uw)
        uw = (relax_uw.unsqueeze(dim).clamp(max=0) * x_lw +
              relax_uw.unsqueeze(dim).clamp(min=0) * x_uw)
        return lw, uw

    @staticmethod
    @torch.jit.script
    def bound_forward_b(
            relax_lw: Tensor, relax_uw: Tensor, relax_lb: Tensor,
            relax_ub: Tensor, x_lb: Tensor, x_ub: Tensor):
        lb = relax_lw.clamp(min=0) * x_lb + relax_lw.clamp(max=0) * x_ub + relax_lb
        ub = relax_uw.clamp(max=0) * x_lb + relax_uw.clamp(min=0) * x_ub + relax_ub
        return lb, ub

    def bound_forward(self, dim_in, x):
        if not self.relaxed:
            self.init_linear_relaxation(x)
            self.bound_relax(x)

        assert (x.lw is None) == (x.uw is None)

        dim = 1 if self.lw.ndim > 0 else 0

        if x.lw is not None:
            lw, uw = BoundActivation.bound_forward_w(self.lw, self.uw, x.lw, x.uw, dim)
        else:
            lw = uw = None
        lb, ub = BoundActivation.bound_forward_b(self.lw, self.uw, self.lb, self.ub, x.lb, x.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        return self.forward(h_L), self.forward(h_U)


class BoundOptimizableActivation(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # Stages:
        #   * `init`: initializing parameters
        #   * `opt`: optimizing parameters
        #   * `reuse`: not optimizing parameters but reuse saved values
        # If `None`, it means activation optimization is currently not used.
        self.opt_stage = None
        self.alpha = OrderedDict()
        # Save patch sizes during bound_backward() for each output_node.
        self.patch_size = {}
        # Location of batch dimension in self.alpha. Must be set by children.
        self.alpha_batch_dim = None
        # A torch.bool mask of shape Tensor([batch_size]) that conditions the sample of alpha and beta to update
        # If set to None, update all samples
        # If not None, select those corresponding to 1 to update
        self.alpha_beta_update_mask = None

    def opt_init(self):
        """Enter the stage for initializing bound optimization. Optimized bounds
        are not used in this stage."""
        self.opt_stage = 'init'

    def opt_start(self):
        """Start optimizing bounds."""
        self.opt_stage = 'opt'

    def opt_reuse(self):
        """ Reuse optimizing bounds """
        self.opt_stage = 'reuse'

    def opt_no_reuse(self):
        """ Finish reusing optimized bounds """
        if self.opt_stage == 'reuse':
            self.opt_stage = None

    def opt_end(self):
        """ End optimizing bounds """
        self.opt_stage = None

    def init_opt_parameters(self, start_nodes):
        """ start_nodes: a list of starting nodes [(node, size)] during
        CROWN backward bound propagation"""
        raise NotImplementedError

    def clip_alpha_(self):
        pass

    def init_linear_relaxation(self, x, dim_opt=None):
        self._init_masks(x)
        # The first dimension of size 2 is used for lA and uA respectively,
        # when computing intermediate bounds.
        if self.opt_stage in ['opt', 'reuse'] and dim_opt is not None:
            # For optimized bounds, we have independent lw for each output dimension for bound optimization.
            # If the output layer is a fully connected layer, len(dim_opt) = 1.
            # If the output layer is a conv layer, len(dim_opt) = 3 but we only use the out_c dimension to create slopes/bias.
            # Variables are shared among out_h, out_w dimensions so far.
            dim = dim_opt if isinstance(dim_opt, int) else dim_opt[0]
            self.lw = torch.zeros(2, dim, *x.lower.shape).to(x.lower)
        else:
            # Without optimized bounds, the lw, lb (slope, biase) etc only depend on intermediate layer bounds,
            # and are shared among different output dimensions.
            self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        self._start = start_node.name

        if self.opt_stage not in ['opt', 'reuse']:
            last_A = last_lA if last_lA is not None else last_uA
            # Returned [(lA, uA)], lbias, ubias
            As, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
            if isinstance(last_A, Patches):
                A_prod = As[0][1].patches if As[0][0] is None else As[0][1].patches
                # FIXME: Unify this function with BoundReLU
                # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                if start_node is not None:
                    if last_A.unstable_idx is not None:
                        # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                        self.patch_size[start_node.name] = [last_A.output_shape[1], A_prod.size(1), last_A.output_shape[2], last_A.output_shape[3], A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                    else:
                        # Regular patches.
                        self.patch_size[start_node.name] = A_prod.size()
            return As, lbias, ubias
        assert self.batch_dim == 0

        if not self.relaxed:
            self.init_linear_relaxation(x, dim_opt=start_shape)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = self.lw[0], self.lb[0], self.uw[0], self.ub[0]
            else:
                w_pos, b_pos, w_neg, b_neg = self.uw[1], self.ub[1], self.lw[1], self.lb[1]
            w_pos = maybe_unfold_patches(w_pos, last_A)
            w_neg = maybe_unfold_patches(w_neg, last_A)
            b_pos = maybe_unfold_patches(b_pos, last_A)
            b_neg = maybe_unfold_patches(b_neg, last_A)
            A_prod, _bias = multiply_by_A_signs(last_A, w_pos, w_neg, b_pos, b_neg)
            return A_prod, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def _no_bound_parameters(self):
        raise AttributeError('Bound parameters have not been initialized.'
                             'Please call `compute_bounds` with `method=CROWN-optimized`'
                             ' at least once.')

    def dump_optimized_params(self):
        raise NotImplementedError

    def restore_optimized_params(self):
        raise NotImplementedError

    def set_alpha_beta_update_mask(self, mask):
        self.alpha_beta_update_mask = mask

    def clean_alpha_beta_update_mask(self):
        self.alpha_beta_update_mask = None


class BoundRelu(BoundOptimizableActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.relu_options = options.get('relu', 'adaptive')  # FIXME: use better names.
        self.use_sparse_spec_alpha = options.get('sparse_spec_alpha', False)
        self.use_sparse_features_alpha = options.get('sparse_features_alpha', False)
        self.beta = self.beta_mask = self.masked_beta = self.sparse_beta = None
        self.split_beta_used = False
        self.history_beta_used = False
        self.flattened_nodes = None
        # Save patches size for each output node.
        self.patch_size = {}
        self.cut_used = False
        self.cut_module = None
        # Alpha dimension is  (2, output_shape, batch, *shape) for ReLU.
        self.alpha_batch_dim = 2

    def init_opt_parameters(self, start_nodes):
        ref = self.inputs[0].lower # a reference variable for getting the shape
        batch_size = ref.size(0)
        self.alpha = OrderedDict()
        self.alpha_lookup_idx = OrderedDict()  # For alpha with sparse spec dimention.
        self.alpha_indices = None  # indices of non-zero alphas.
        verbosity = self.options.get('verbosity', 0)

        # Alpha can be sparse in both spec dimension, and the C*H*W dimension.
        # We first deal with the sparse-feature alpha, which is sparse in the
        # C*H*W dimesnion of this layer.
        minimum_sparsity = self.options.get('minimum_sparsity', 0.9)
        if (hasattr(self.inputs[0], 'lower') and hasattr(self.inputs[0], 'upper')
                and self.use_sparse_features_alpha):
            # Pre-activation bounds available, we will store the alpha for unstable neurons only.
            # Since each element in a batch can have different unstable neurons,
            # for simplicity we find a super-set using any(dim=0).
            # This can be non-ideal if the x in a batch are very different.
            self.alpha_indices = torch.logical_and(
                self.inputs[0].lower < 0, self.inputs[0].upper > 0).any(dim=0).nonzero(as_tuple=True)
            total_neuron_size = self.inputs[0].lower.numel() // batch_size
            if self.alpha_indices[0].size(0) <= minimum_sparsity * total_neuron_size:
                # Shape is the number of unstable neurons in this layer.
                alpha_shape = [self.alpha_indices[0].size(0)]
                # Skip the batch, spec dimension, and find the lower slopes for all unstable neurons.
                if len(self.alpha_indices) == 1:
                    # This layer is after a linear layer.
                    alpha_init = self.lower_d[:, :, self.alpha_indices[0]]
                elif len(self.alpha_indices) == 3:
                    # This layer is after a conv layer.
                    alpha_init = self.lower_d[
                        :, :, self.alpha_indices[0], self.alpha_indices[1],
                        self.alpha_indices[2]]
                else:
                    raise ValueError
                if verbosity > 0:
                    print(f'layer {self.name} using sparse-features alpha with shape {alpha_shape}; unstable size {self.alpha_indices[0].size(0)}; total size {total_neuron_size} ({ref.shape})')
            else:
                alpha_shape = self.shape  # Full alpha.
                alpha_init = self.lower_d
                if verbosity > 0:
                    print(f'layer {self.name} using full alpha with shape {alpha_shape}; unstable size {self.alpha_indices[0].size(0)}; total size {total_neuron_size} ({ref.shape})')
                    self.alpha_indices = None  # Use full alpha.
        else:
            alpha_shape = self.shape  # Full alpha.
            alpha_init = self.lower_d
        # Now we start to create alphas for all start nodes.
        # When sparse-spec feature is enabled, alpha is created for only
        # unstable neurons in start node.
        for ns, output_shape, unstable_idx in start_nodes:
            if isinstance(output_shape, (list, tuple)):
                if len(output_shape) > 1:
                    size_s = prod(output_shape)  # Conv layers.
                else:
                    size_s = output_shape[0]
            else:
                size_s = output_shape
            # unstable_idx may be a tensor (dense layer or conv layer
            # with shared alpha), or tuple of 3-d tensors (conv layer with
            # non-sharing alpha).
            sparsity = float('inf') if unstable_idx is None else unstable_idx.size(0) if isinstance(unstable_idx, torch.Tensor) else unstable_idx[0].size(0)
            if sparsity <= minimum_sparsity * size_s and self.use_sparse_spec_alpha:
                if verbosity > 0:
                    print(f'layer {self.name} start_node {ns} using sparse-spec alpha with unstable size {sparsity} total_size {size_s} output_shape {output_shape}')
                # For fully connected layer, or conv layer with shared alpha per channel.
                # shape is (2, sparse_spec, batch, this_layer_shape)
                # We create sparse specification dimension, where the spec dimension of alpha only includes slopes for unstable neurons in start_node.
                self.alpha[ns] = torch.empty([2, sparsity + 1, batch_size, *alpha_shape],
                                             dtype=torch.float, device=ref.device, requires_grad=True)
                self.alpha[ns].data.copy_(alpha_init.data)  # This will broadcast to (2, sparse_spec) dimensions.
                # unstable_idx is a list of used neurons (or channels for BoundConv) for the start_node.
                assert unstable_idx.ndim == 1 if isinstance(unstable_idx, torch.Tensor) else unstable_idx[0].ndim == 1
                # We only need to the alpha for the unstable neurons in start_node.
                indices = torch.arange(1, sparsity + 1, device=alpha_init.device, dtype=torch.long)
                if isinstance(output_shape, int) or len(output_shape) == 1:
                    # Fully connected layers, or conv layer in patches mode with partially shared alpha (pixels in the same channel use the same alpha).
                    self.alpha_lookup_idx[ns] = torch.zeros(size_s, dtype=torch.long, device=alpha_init.device)
                    # This lookup table maps the unstable_idx to the actual alpha location in self.alpha[ns].
                    # Note that self.alpha[ns][:,0] is reserved for any unstable neurons that are not found in the lookup table. This usually should not
                    # happen, unless reference bounds are not properly set.
                    self.alpha_lookup_idx[ns].data[unstable_idx] = indices
                else:
                    # conv layer in matrix mode, or in patches mode but with non-shared alpha. The lookup table is 3-d.
                    assert len(output_shape) == 3
                    self.alpha_lookup_idx[ns] = torch.zeros(output_shape, dtype=torch.long, device=alpha_init.device)
                    if isinstance(unstable_idx, torch.Tensor):
                        # Convert the unstable index from flattend 1-d to 3-d. (matrix mode).
                        unstable_idx_3d = unravel_index(unstable_idx, output_shape)
                    else:
                        # Patches mode with non-shared alpha, unstable_idx is already 3d.
                        unstable_idx_3d = unstable_idx
                    # Build look-up table.
                    self.alpha_lookup_idx[ns].data[unstable_idx_3d[0], unstable_idx_3d[1], unstable_idx_3d[2]] = indices
            else:
                if verbosity > 0:
                    print(f'layer {self.name} start_node {ns} using full alpha with unstable size {sparsity if unstable_idx is not None else None} total_size {size_s} output_shape {output_shape}')
                # alpha shape is (2, spec, batch, this_layer_shape). "this_layer_shape" may still be sparse.
                self.alpha[ns] = torch.empty([2, size_s, batch_size, *alpha_shape],
                                             dtype=torch.float, device=ref.device, requires_grad=True)
                self.alpha[ns].data.copy_(alpha_init.data)  # This will broadcast to (2, spec) dimensions
                # alpha_lookup_idx can be used for checking if sparse alpha is used or not.
                self.alpha_lookup_idx[ns] = None

    def clip_alpha_(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def forward(self, x):
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return F.relu(x)

    def _forward_relaxation(self, x):
        self._init_masks(x)
        self.mask_pos = self.mask_pos.to(x.lower)
        self.mask_both = self.mask_both.to(x.lower)

        upper_k, upper_b = self._relu_upper_bound(x.lower, x.upper)
        self.uw = self.mask_pos + self.mask_both * upper_k
        self.ub = self.mask_both * upper_b

        if self.opt_stage in ['opt', 'reuse']:
            # Each actual alpha in the forward mode has shape (batch_size, *relu_node_shape].
            # But self.alpha has shape (2, output_shape, batch_size, *relu_node_shape]
            # and we do not need its first two dimensions.
            lower_k = alpha = self.alpha['_forward'][0, 0]
        elif self.relu_options == "same-slope":
            lower_k = upper_k
        elif self.relu_options == "zero-lb":
            lower_k = torch.zeros_like(upper_k)
        elif self.relu_options == "one-lb":
            lower_k = torch.ones_like(upper_k)
        else:
            # adaptive
            lower_k = torch.gt(torch.abs(x.upper), torch.abs(x.lower)).to(torch.float)
        # NOTE #FIXME Saved for initialization bounds for optimization.
        # In the backward mode, same-slope bounds are used.
        # But here it is using adaptive bounds which seem to be better
        # for nn4sys benchmark with loose input bounds. Need confirmation
        # for other cases.
        self.lower_d = lower_k.detach() # saved for initializing optimized bounds

        self.lw = self.mask_both * lower_k + self.mask_pos

    def bound_dynamic_forward(self, x, max_dim=None, offset=0):
        self._init_masks(x)
        self.mask_pos = self.mask_pos.to(x.lower)
        self.mask_both = self.mask_both.to(x.lower)

        upper_k, upper_b = self._relu_upper_bound(x.lower, x.upper)
        w_new = (self.mask_pos.unsqueeze(1) * x.lw
            + self.mask_both.unsqueeze(1) * upper_k.unsqueeze(1) * x.lw)
        upper_b = self.mask_both * upper_b / 2
        b_new = (self.mask_pos * x.lb
            + self.mask_both * upper_k * x.lb + upper_b)

        # Create new variables for unstable ReLU
        batch_size = w_new.shape[0]
        device = w_new.device
        unstable = self.mask_both.view(batch_size, -1)
        tot_unstable = int(unstable.sum(dim=-1).max())
        tot_dim = x.tot_dim + tot_unstable
        # logger.debug(f'Unstable: {tot_unstable}')

        if offset + w_new.shape[1] < x.tot_dim:
            return LinearBound(
                w_new, b_new, w_new, b_new, x_L=x.x_L, x_U=x.x_U, tot_dim=tot_dim)

        index = torch.cumsum(unstable, dim=-1).to(torch.int64)
        index = (index - (offset + w_new.shape[1] - x.tot_dim)).clamp(min=0)
        num_new_dim = int(index.max())
        num_new_dim_actual = min(num_new_dim, max_dim - w_new.shape[1])
        index = index.clamp(max=num_new_dim_actual+1)
        w_unstable = torch.zeros(batch_size, num_new_dim_actual + 2, unstable.size(-1), device=device)
        x_L_unstable = -torch.ones(batch_size, num_new_dim_actual, device=device)
        x_U_unstable = torch.ones(batch_size, num_new_dim_actual, device=device)
        w_unstable.scatter_(dim=1, index=index.unsqueeze(1), src=upper_b.view(batch_size, 1, -1), reduce='add')
        w_unstable = w_unstable[:, 1:-1].view(batch_size, num_new_dim_actual, *w_new.shape[2:])

        w_new = torch.cat([w_new, w_unstable], dim=1)
        x_L_new = torch.cat([x.x_L, x_L_unstable], dim=-1)
        x_U_new = torch.cat([x.x_U, x_U_unstable], dim=-1)

        return LinearBound(
            w_new, b_new, w_new, b_new, x_L=x_L_new, x_U=x_U_new, tot_dim=tot_dim)


    def bound_forward(self, dim_in, x):
        self._forward_relaxation(x)

        lb = self.lw * x.lb
        ub = self.uw * x.ub + self.ub

        if x.lw is not None:
            lw = self.lw.unsqueeze(1) * x.lw
        else:
            lw = None
        if x.uw is not None:
            uw = self.uw.unsqueeze(1) * x.uw
        else:
            uw = None

        if not lw.requires_grad:
            del self.mask_both, self.mask_pos
            del self.lw, self.uw, self.ub

        return LinearBound(lw, lb, uw, ub)

    @staticmethod
    @torch.jit.script
    def _relu_upper_bound(lb, ub):
        """Upper bound slope and intercept according to CROWN relaxation."""
        # TODO: pre-comple all JIT functions before run.
        lb_r = lb.clamp(max=0)
        ub_r = ub.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        return upper_d, upper_b

    @staticmethod
    def _relu_mask_alpha(lower, upper, lb_lower_d : Optional[Tensor], ub_lower_d : Optional[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        lower_mask = (lower >= 0).requires_grad_(False).to(lower.dtype)
        upper_mask = (upper <= 0).requires_grad_(False)
        zero_coeffs = upper_mask.all()
        no_mask = (1. - lower_mask) * (1. - upper_mask.to(upper.dtype))
        if lb_lower_d is not None:
            lb_lower_d = torch.clamp(lb_lower_d, min=0., max=1.) * no_mask + lower_mask
        if ub_lower_d is not None:
            ub_lower_d = torch.clamp(ub_lower_d, min=0., max=1.) * no_mask + lower_mask
        return lb_lower_d, ub_lower_d, zero_coeffs

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        if x is not None:
            lower = x.lower
            upper = x.upper
        else:
            lower = self.lower
            upper = self.upper

        # Upper bound slope and intercept according to CROWN relaxation.
        upper_d, upper_b = self._relu_upper_bound(lower, upper)

        flag_expand = False
        ub_lower_d = lb_lower_d = None
        lower_b = None  # ReLU does not have lower bound intercept (=0).
        alpha_lookup_idx = None  # For sparse-spec alpha.
        if self.opt_stage in ['opt', 'reuse']:
            # Alpha-CROWN.
            lower_d = None
            # Each alpha has shape (2, output_shape, batch_size, *relu_node_shape].
            # If slope is shared, output_shape will be 1.
            # The *relu_node_shape might be sparse (sparse-feature alpha), where the non-zero values are indicated by self.alpha_indices.
            # The out_shape might be sparse (sparse-spec alpha), where the non-zero values are indexed by self.alpha_lookup_idx.
            if unstable_idx is not None and self.alpha[start_node.name].size(1) != 1:
                # print(f'relu layer {self.name}, start_node {start_node}, unstable_idx {type(unstable_idx)} alpha idx {self.alpha_lookup_idx[start_node.name].size()}')
                alpha_lookup_idx = self.alpha_lookup_idx[start_node.name]
                if isinstance(unstable_idx, tuple):
                    # Start node is a conv node.
                    selected_alpha = self.alpha[start_node.name]
                    if isinstance(last_lA, Tensor) or isinstance(last_uA, Tensor):
                        # Start node is a conv node but we received tensors as A matrices.
                        # Patches mode converted to matrix, or matrix mode used. Need to select accross the spec dimension.
                        # For this node, since it is in matrix mode, the spec dimension is out_c * out_h * out_w
                        # Shape is [2, spec, batch, *this_layer_shape]
                        if alpha_lookup_idx is None:
                            # Reshape the spec dimension to c*h*w so we can select used alphas based on unstable index.
                            # Shape becomes [2, out_c, out_h, out_w, batch, *this_layer_shape]
                            selected_alpha = selected_alpha.view(selected_alpha.size(0), *start_node.output_shape[1:], *selected_alpha.shape[2:])
                            selected_alpha = selected_alpha[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]]
                        else:
                            assert alpha_lookup_idx.ndim == 3
                            # We only stored some alphas, and A is also sparse, so the unstable_idx must be first translated to real indices.
                            # alpha shape is (2, sparse_spec_shape, batch_size, *relu_node_shape) where relu_node_shape can also be sparse.
                            # We use sparse-spec alphas. Need to convert these unstable_idx[0], unstable_idx[1], unstable_idx[0] using lookup table.
                            _unstable_idx = alpha_lookup_idx[unstable_idx[0], unstable_idx[1], unstable_idx[2]]
                            selected_alpha = self.non_deter_index_select(selected_alpha, index=_unstable_idx, dim=1)
                    else:
                        # Patches mode. Alpha must be selected after unfolding, so cannot be done here.
                        # Selection is deferred to maybe_unfold() using alpha_lookup_idx.
                        # For partially shared alpha, its shape is (2, out_c, batch_size, *relu_node_shape).
                        # For full alpha, its shape is (2, out_c*out_h*out_w, batch_size, *relu_node_shape).
                        # Both the spec dimension and relu_node_shape dimensions can be sparse.
                        pass
                elif unstable_idx.ndim == 1:
                    # Start node is a FC node.
                    # Only unstable neurons of the start_node neurons are used.
                    assert alpha_lookup_idx is None or alpha_lookup_idx.ndim == 1
                    _unstable_idx = alpha_lookup_idx[unstable_idx] if alpha_lookup_idx is not None else unstable_idx
                    selected_alpha = self.non_deter_index_select(self.alpha[start_node.name], index=_unstable_idx, dim=1)
                elif unstable_idx.ndim == 2:
                    assert alpha_lookup_idx is None, "sparse spec alpha has not been implemented yet."
                    # Each element in the batch selects different neurons.
                    selected_alpha = batched_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                else:
                    raise ValueError
            else:
                # Spec dimension is dense. Alpha must not be created sparsely.
                assert self.alpha_lookup_idx[start_node.name] is None
                selected_alpha = self.alpha[start_node.name]
            # The first dimension is lower/upper intermediate bound.
            if last_lA is not None:
                lb_lower_d = selected_alpha[0]
            if last_uA is not None:
                ub_lower_d = selected_alpha[1]

            if self.alpha_indices is not None:
                # Sparse alpha on the hwc dimension. We store slopes for unstable neurons in this layer only.
                # Recover to full alpha first.
                def reconstruct_full_alpha(sparse_alpha, full_alpha_shape, alpha_indices):
                    full_alpha = torch.zeros(full_alpha_shape, dtype=sparse_alpha.dtype, device=sparse_alpha.device)
                    if len(alpha_indices) == 1:
                        # Relu after a dense layer.
                        full_alpha[:, :, alpha_indices[0]] = sparse_alpha
                    elif len(alpha_indices) == 3:
                        # Relu after a conv layer.
                        full_alpha[:, :, alpha_indices[0], alpha_indices[1], alpha_indices[2]] = sparse_alpha
                    else:
                        raise ValueError
                    return full_alpha
                sparse_alpha_shape = lb_lower_d.shape if lb_lower_d is not None else ub_lower_d.shape
                full_alpha_shape = sparse_alpha_shape[:-1] + self.shape
                if lb_lower_d is not None:
                    lb_lower_d = reconstruct_full_alpha(lb_lower_d, full_alpha_shape, self.alpha_indices)
                if ub_lower_d is not None:
                    ub_lower_d = reconstruct_full_alpha(ub_lower_d, full_alpha_shape, self.alpha_indices)

            # condition only on the masked part
            if self.alpha_beta_update_mask is not None:
                if lb_lower_d is not None:
                    lb_lower_d_new = lb_lower_d[:, self.alpha_beta_update_mask]
                else:
                    lb_lower_d_new = None
                if ub_lower_d is not None:
                    ub_lower_d_new = ub_lower_d[:, self.alpha_beta_update_mask]
                else:
                    ub_lower_d_new = None
                lb_lower_d, ub_lower_d, zero_coeffs = self._relu_mask_alpha(lower, upper, lb_lower_d_new, ub_lower_d_new)
            else:
                lb_lower_d, ub_lower_d, zero_coeffs = self._relu_mask_alpha(lower, upper, lb_lower_d, ub_lower_d)
            self.zero_backward_coeffs_l = self.zero_backward_coeffs_u = zero_coeffs
            flag_expand = True  # we already have the spec dimension.
        elif self.relu_options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.relu_options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).to(upper_d.dtype)
        elif self.relu_options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).to(upper_d.dtype)
        elif self.relu_options == "reversed-adaptive":
            lower_d = (upper_d < 0.5).to(upper_d.dtype)
        else:
            # adaptive
            lower_d = (upper_d > 0.5).to(upper_d.dtype)

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        upper_d = upper_d.unsqueeze(0)
        upper_b = upper_b.unsqueeze(0)
        if not flag_expand:
            if self.opt_stage in ['opt', 'reuse']:
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)
        return upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, alpha_lookup_idx

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, beta_for_intermediate_layers=False, unstable_idx=None):
        # Get element-wise CROWN linear relaxations.
        upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, alpha_lookup_idx = \
            self._backward_relaxation(last_lA, last_uA, x, start_node, unstable_idx)
        # save for calculate babsr score
        self.d = upper_d
        self.lA = last_lA
        # Save for initialization bounds.
        self.lower_d = lower_d

        # Choose upper or lower bounds based on the sign of last_A
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            # Obtain the new linear relaxation coefficients based on the signs in last_A.
            _A, _bias = multiply_by_A_signs(last_A, d_pos, d_neg, b_pos, b_neg)
            if isinstance(last_A, Patches):
                # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                A_prod = _A.patches
                if start_node is not None:
                    if last_A.unstable_idx is not None:
                        # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                        self.patch_size[start_node.name] = [last_A.output_shape[1], A_prod.size(1), last_A.output_shape[2], last_A.output_shape[3], A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                    else:
                        # Regular patches.
                        self.patch_size[start_node.name] = A_prod.size()
            return _A, _bias

        ######## A problem with patches mode for cut constraint start ##########
        # There are cases that  the node that is in the constraint but not selected by the patches for the output node
        # trick: only count the small patches that have all the split node coeffs[ci].sum() equal to coeffs_unfolded[ci][out_h, out_w, -1].sum()
        # we should force these beta to be 0 to disable the effect of these constraints
        A = last_lA if last_lA is not None else last_uA
        current_layer_shape = x.lower.size()[1:]
        if self.cut_used and type(A) is Patches:
            self.cut_module.patch_trick(start_node, self.name, A, current_layer_shape)
        ######## A problem with patches mode for cut constraint end ##########

        if self.cut_used:
            # propagate postrelu node in cut constraints
            last_lA, last_uA = self.cut_module.relu_cut(
                start_node, self.name, last_lA, last_uA, current_layer_shape, unstable_idx,
                batch_mask=self.alpha_beta_update_mask)

        # In patches mode we might need an unfold.
        # lower_d, upper_d, lower_b, upper_b: 1, batch, current_c, current_w, current_h or None
        upper_d = maybe_unfold_patches(upper_d, last_lA if last_lA is not None else last_uA)
        lower_d = maybe_unfold_patches(lower_d, last_lA if last_lA is not None else last_uA)
        upper_b = maybe_unfold_patches(upper_b, last_lA if last_lA is not None else last_uA)
        lower_b = maybe_unfold_patches(lower_b, last_lA if last_lA is not None else last_uA)  # for ReLU it is always None; keeping it here for completeness.
        # ub_lower_d and lb_lower_d might have sparse spec dimension, so they may need alpha_lookup_idx to convert to actual spec dim.
        ub_lower_d = maybe_unfold_patches(ub_lower_d, last_uA, alpha_lookup_idx=alpha_lookup_idx)
        # optimizable slope lb_lower_d: spec (only channels in spec layer), batch, current_c, current_w, current_h
        # patches mode lb_lower_d after unfold: unstable, batch, in_C, H, W
        lb_lower_d = maybe_unfold_patches(lb_lower_d, last_lA, alpha_lookup_idx=alpha_lookup_idx)

        if self.cut_used:
            I = (x.lower < 0) * (x.upper > 0)
            # propagate integer var of relu neuron (arelu) in cut constraints through relu layer
            lA, uA, lbias, ubias = self.cut_module.arelu_cut(
                start_node, self.name, last_lA, last_uA, lower_d, upper_d,
                lower_b, upper_b, lb_lower_d, ub_lower_d, I, x, self.patch_size,
                current_layer_shape, unstable_idx,
                batch_mask=self.alpha_beta_update_mask)
        else:
            uA, ubias = _bound_oneside(
                last_uA, upper_d, ub_lower_d if lower_d is None else lower_d,
                upper_b, lower_b)
            lA, lbias = _bound_oneside(
                last_lA, lb_lower_d if lower_d is None else lower_d, upper_d,
                lower_b, upper_b)

        # Regular Beta CROWN with single neuron split
        def _beta_crown_single_neuron_splits(A, uA, lA, unstable_idx):
            if type(A) is Patches:
                if self.options.get('enable_opt_interm_bounds', False):
                    # expand sparse_beta to full beta
                    beta_values = (self.sparse_beta[start_node.name] * self.sparse_beta_sign[start_node.name])
                    beta_indices = self.sparse_beta_loc[start_node.name]
                    self.masked_beta = torch.zeros(2, *self.shape).reshape(2, -1).to(A.patches.dtype)
                    self.non_deter_scatter_add(self.masked_beta, dim=1, index=beta_indices, src=beta_values.to(self.masked_beta.dtype))
                    self.masked_beta = self.masked_beta.reshape(2, *self.shape)
                else:
                    if self.beta is None:
                        # Beta not used.
                        return lA, uA
                    # For patches mode, masked_beta will be used; sparse beta is not supported.
                    self.masked_beta = (self.beta[0] * self.beta_mask).requires_grad_()
                # unfold the beta as patches, size (batch, out_h, out_w, in_c, H, W)
                A_patches = A.patches
                masked_beta_unfolded = inplace_unfold(self.masked_beta, kernel_size=A_patches.shape[-2:], padding=A.padding, stride=A.stride, inserted_zeros=A.inserted_zeros, output_padding=A.output_padding)
                if A.unstable_idx is not None:
                    masked_beta_unfolded = masked_beta_unfolded.permute(1, 2, 0, 3, 4, 5)
                    # After selection, the shape is (unstable_size, batch, in_c, H, W).
                    masked_beta_unfolded = masked_beta_unfolded[A.unstable_idx[1], A.unstable_idx[2]]
                else:
                    # Add the spec (out_c) dimension.
                    masked_beta_unfolded = masked_beta_unfolded.unsqueeze(0)
                if self.alpha_beta_update_mask is not None:
                    masked_beta_unfolded = masked_beta_unfolded[self.alpha_beta_update_mask]
                if uA is not None:
                    uA = uA.create_similar(uA.patches + masked_beta_unfolded)
                if lA is not None:
                    lA = lA.create_similar(lA.patches - masked_beta_unfolded)
            elif type(A) is Tensor:
                if self.options.get('enable_opt_interm_bounds', False):
                    # For matrix mode, beta is sparse.
                    beta_values = (self.sparse_beta[start_node.name] * self.sparse_beta_sign[start_node.name]).expand(lA.size(0), -1, -1)
                    # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
                    beta_indices = self.sparse_beta_loc[start_node.name].unsqueeze(0).expand(lA.size(0), -1, -1)
                else:
                    # For matrix mode, beta is sparse.
                    beta_values = (self.sparse_beta * self.sparse_beta_sign).expand(lA.size(0), -1, -1)
                    # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
                    beta_indices = self.sparse_beta_loc.unsqueeze(0).expand(lA.size(0), -1, -1)
                # For conv layer, the last dimension is flattened in indices.
                prev_size = A.size()
                if self.alpha_beta_update_mask is not None:
                    beta_indices = beta_indices[:, self.alpha_beta_update_mask]
                    beta_values = beta_values[:, self.alpha_beta_update_mask]
                if uA is not None:
                    uA = self.non_deter_scatter_add(uA.view(uA.size(0), uA.size(1), -1), dim=2, index=beta_indices, src=beta_values.to(uA.dtype))
                    uA = uA.view(prev_size)
                if lA is not None:
                    lA = self.non_deter_scatter_add(lA.view(lA.size(0), lA.size(1), -1), dim=2, index=beta_indices, src=beta_values.neg().to(lA.dtype))
                    lA = lA.view(prev_size)
            else:
                raise RuntimeError(f"Unknown type {type(A)} for A")
            return lA, uA

        if self.cut_used:
            # propagate prerelu node in cut constraints
            lA, uA = self.cut_module.pre_cut(start_node, self.name, lA, uA, current_layer_shape, unstable_idx,
                                             batch_mask=self.alpha_beta_update_mask)
        self.masked_beta_lower = self.masked_beta_upper = None
        if self.options.get('optimize_bound_args', {}).get('enable_beta_crown', False) and self.sparse_beta is not None:
            if self.options.get('optimize_bound_args', {}).get('single_node_split', False):
                # Beta-CROWN: each split constraint only has single neuron (e.g., second ReLU neuron > 0).
                A = lA if lA is not None else uA
                lA, uA = _beta_crown_single_neuron_splits(A, uA, lA, unstable_idx)
            # The code block below is for debugging and will be removed (until the end of this function).
            # elif False and not self.options.get('optimize_bound_args', {}).get('single_node_split', True):
            #     # Improved Beta-CROWN: (1) general split constraints: each split constraint have multiple neuron
            #     # (e.g., second ReLU neuron > 0); (2) intermediate Relu bounds refinement with the general split constraints.
            #     A = uA if uA is not None else lA
            #     lA, uA = _beta_crown_multi_neuron_splits(x, A, uA, lA, unstable_idx, start_node)
            # print(lA.sum(), uA.sum())
            # exit()

        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        return F.relu(h_L), F.relu(h_U)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (8,16,16)
        gvars_array = np.array(v[0])
        this_layer_shape = gvars_array.shape
        assert gvars_array.shape == self.output_shape[1:]

        pre_lbs = self.inputs[0].lower.cpu().detach().numpy().reshape(-1)
        pre_ubs = self.inputs[0].upper.cpu().detach().numpy().reshape(-1)

        new_layer_gurobi_vars = []
        relu_integer_vars = []
        new_relu_layer_constrs = []
        # predefined zero variable shared in the whole solver model
        zero_var = model.getVarByName("zero")

        for neuron_idx, pre_var in enumerate(gvars_array.reshape(-1)):
            pre_ub = pre_ubs[neuron_idx]
            pre_lb = pre_lbs[neuron_idx]

            if pre_lb >= 0:
                # ReLU is always passing
                var = pre_var
            elif pre_ub <= 0:
                var = zero_var
            else:
                ub = pre_ub

                var = model.addVar(ub=ub, lb=pre_lb,
                                   obj=0,
                                   vtype=grb.GRB.CONTINUOUS,
                                   name=f'ReLU{self.name}_{neuron_idx}')

                if model_type == "mip" or model_type == "lp_integer":
                    # binary indicator
                    if model_type == "mip":
                        a = model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{self.name}_{neuron_idx}')
                    elif model_type == "lp_integer":
                        a = model.addVar(ub=1, lb=0, vtype=grb.GRB.CONTINUOUS, name=f'aReLU{self.name}_{neuron_idx}')
                    relu_integer_vars.append(a)

                    new_relu_layer_constrs.append(
                        model.addConstr(pre_var - pre_lb * (1 - a) >= var,
                                        name=f'ReLU{self.name}_{neuron_idx}_a_0'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= pre_var, name=f'ReLU{self.name}_{neuron_idx}_a_1'))
                    new_relu_layer_constrs.append(
                        model.addConstr(pre_ub * a >= var, name=f'ReLU{self.name}_{neuron_idx}_a_2'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= 0, name=f'ReLU{self.name}_{neuron_idx}_a_3'))

                elif model_type == "lp":
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= 0, name=f'ReLU{self.name}_{neuron_idx}_a_0'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= pre_var, name=f'ReLU{self.name}_{neuron_idx}_a_1'))
                    new_relu_layer_constrs.append(model.addConstr(
                        pre_ub * pre_var - (pre_ub - pre_lb) * var >= pre_ub * pre_lb,
                        name=f'ReLU{self.name}_{neuron_idx}_a_2'))

                else:
                    print(f"gurobi model type {model_type} not supported!")

            new_layer_gurobi_vars.append(var)

        new_layer_gurobi_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape).tolist()
        if model_type in ["mip", "lp_integer"]:
            self.integer_vars = relu_integer_vars
        self.solver_vars = new_layer_gurobi_vars
        self.solver_constrs = new_relu_layer_constrs
        model.update()

    def dump_optimized_params(self):
        return {
            'alpha': self.alpha,
            'alpha_lookup_idx': self.alpha_lookup_idx,
            'alpha_indices': self.alpha_indices
        }

    def restore_optimized_params(self, opt_var_dict):
        self.alpha, self.alpha_lookup_idx, self.alpha_indices = \
            opt_var_dict['alpha'], opt_var_dict['alpha_lookup_idx'], opt_var_dict['alpha_indices']


class BoundLeakyRelu(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.options = options.get('relu')
        self.alpha = attr['alpha']

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
            lower_d = (upper_d >= 1.0).to(upper_d.dtype) + (upper_d < 1.0).to(upper_d.dtype) * self.alpha
        elif self.options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).to(upper_d.dtype)+ (upper_d <= 0.0).to(upper_d.dtype) * self.alpha
        else:
            lower_d = (upper_d > 0.5).to(upper_d.dtype) + (upper_d <= 0.5).to(upper_d.dtype)* self.alpha

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

    def dump_optimized_params(self):
        return self.alpha

    def restore_optimized_params(self, alpha):
        self.alpha = alpha


class BoundTanh(BoundOptimizableActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.precompute_relaxation('tanh', torch.tanh, self.dtanh)
        # Alpha dimension is (4, 2, output_shape, batch, *shape) for Tanh.
        self.alpha_batch_dim = 3

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {}
        self.tp_both_upper_init = {}

    def init_opt_parameters(self, start_nodes):
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = l.shape
        for ns, size_s, _ in start_nodes:
            if isinstance(size_s, torch.Size):
                size_s = prod(size_s)
            self.alpha[ns] = torch.empty(4, 2, size_s, *shape, device=l.device)
            self.alpha[ns].data[:2] = ((l + u) / 2).unsqueeze(0).expand(2, 2, size_s, *shape)
            self.alpha[ns].data[2] = self.tp_both_lower_init[ns].expand(2, size_s, *shape)
            self.alpha[ns].data[3] = self.tp_both_upper_init[ns].expand(2, size_s, *shape)

    def dtanh(self, x):
        # to avoid bp error when cosh is too large
        # cosh(25.0)**2 > 1e21
        mask = torch.lt(torch.abs(x), 25.0).to(x.dtype)
        cosh = torch.cosh(mask * x + 1 - mask)
        return mask * (1. / cosh.pow(2))

    @torch.no_grad()
    def precompute_relaxation(self, name, func, dfunc, x_limit = 500):
        """
        This function precomputes the tangent lines that will be used as lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for {}'.format(name))

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper), check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d >= lower), check if the slope at d will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        # Given an upper bound point (>=0), find a line that is guaranteed to be a lower bound of this function.
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        r = torch.zeros_like(upper)
        # Initial guess, the tangent line is at -1.
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to 0.
        for t in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower = l.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
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
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = torch.where(mask_close,
                                   dfunc(upper), (y_u - y_l) / (upper - lower).clamp(min=min_preact))

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(mask=self.mask_neg, type='upper', k=k, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(mask=self.mask_pos, type='lower', k=k, x0=lower, y0=y_l)

        # Indices of neurons with input upper bound >=0, whose optimal slope to lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0, they will be masked later.
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            (upper / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # Lookup the lower bound slope from the pre-computed table.
        d_lower = torch.index_select(self.d_lower, 0, index).view(lower.shape)

        # Indices of neurons with lower bound <=0, whose optimal slope to upper bound the function was pre-computed.
        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            (lower / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_upper = torch.index_select(self.d_upper, 0, index).view(upper.shape)

        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            # Clipping is done here rather than after `opt.step()` call
            # because it depends on pre-activation bounds
            self.alpha[ns].data[0, :] = torch.max(torch.min(self.alpha[ns][0, :], upper), lower)
            self.alpha[ns].data[1, :] = torch.max(torch.min(self.alpha[ns][1, :], upper), lower)
            self.alpha[ns].data[2, :] = torch.min(self.alpha[ns][2, :], d_lower)
            self.alpha[ns].data[3, :] = torch.max(self.alpha[ns][3, :], d_upper)

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0]
            tp_neg = self.alpha[ns][1]
            tp_both_lower = self.alpha[ns][2]
            tp_both_upper = self.alpha[ns][3]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(lower))
            self.add_linear_relaxation(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='lower',
                k=dfunc(tp_both_lower), x0=tp_both_lower,
                y0=self.forward(tp_both_lower))

            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(upper))
            self.add_linear_relaxation(mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='upper',
                k=dfunc(tp_both_upper), x0=tp_both_upper,
                y0=self.forward(tp_both_upper))

            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower',
                k=dfunc(tp_neg), x0=tp_neg, y0=self.forward(tp_neg))
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper',
                k=dfunc(tp_pos), x0=tp_pos, y0=self.forward(tp_pos))
        else:
            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
            k = dfunc(d_lower)
            y0 = func(d_lower)
            if self.opt_stage == 'init':
                # Initialize optimizable slope.
                ns = self._start
                self.tp_both_lower_init[ns] = d_lower.detach()
            # Another possibility is to use the direct line as the lower bound, when this direct line does not intersect with f.
            # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(lower))
            self.add_linear_relaxation(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='lower', k=k, x0=d_lower, y0=y0)

            # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
            k = dfunc(d_upper)
            y0 = func(d_upper)
            if self.opt_stage == 'init':
                ns = self._start
                self.tp_both_upper_init[ns] = d_upper.detach()
                self.tmp_lower = x.lower.detach()
                self.tmp_upper = x.upper.detach()
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(upper))
            self.add_linear_relaxation(mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='upper', k=k, x0=d_upper, y0=y0)

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.tanh, self.dtanh)

    def dump_optimized_params(self):
        return self.alpha

    def restore_optimized_params(self, alpha):
        self.alpha = alpha


class BoundSigmoid(BoundTanh):
    def __init__(self, attr, inputs, output_index, options):
        super(BoundTanh, self).__init__(attr, inputs, output_index, options)
        self.precompute_relaxation('sigmoid', torch.sigmoid, self.dsigmoid)
        # Alpha dimension is  (4, 2, output_shape, batch, *shape) for S-shaped functions.
        self.alpha_batch_dim = 3

    def forward(self, x):
        return torch.sigmoid(x)

    def dsigmoid(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.sigmoid, self.dsigmoid)


class BoundSoftplus(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super(BoundSoftplus, self).__init__(attr, inputs, output_index, options)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)


class BoundAbs(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return x.abs()

    def bound_backward(self, last_lA, last_uA, x):
        x_L = x.lower.clamp(max=0)
        x_U = torch.max(x.upper.clamp(min=0), x_L + 1e-8)
        mask_neg = x_U <= 0
        mask_pos = x_L >= 0
        y_L = x_L.abs()
        y_U = x_U.abs()
        upper_k = (y_U - y_L) / (x_U - x_L)
        upper_b = y_L - upper_k * x_L
        lower_k = (mask_neg * (-1.0) + mask_pos * 1.0)
        lower_b = (mask_neg + mask_pos) * ( y_L - lower_k * x_L )
        if last_uA is not None:
            # Special case if we only want the upper bound with non-negative coefficients
            if last_uA.min() >= 0:
                uA = last_uA * upper_k
                ubias = self.get_bias(last_uA, upper_b)
            else:
                last_uA_pos = last_uA.clamp(min=0)
                last_uA_neg = last_uA.clamp(max=0)
                uA = last_uA_pos * upper_k + last_uA_neg * lower_k
                ubias = (self.get_bias(last_uA_pos, upper_b)
                         + self.get_bias(last_uA_neg, lower_b))
        else:
            uA, ubias = None, 0
        if last_lA is not None:
            if last_lA.max() <= 0:
                lA = last_lA * upper_k
                lbias = self.get_bias(last_lA, upper_b)
            else:
                last_lA_pos = last_lA.clamp(min=0)
                last_lA_neg = last_lA.clamp(max=0)
                lA = last_lA_pos * lower_k + last_lA_neg * upper_k
                lbias = (self.get_bias(last_lA_pos, lower_b)
                         + self.get_bias(last_lA_neg, upper_b))
        else:
            lA, lbias = None, 0
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        lower = ((h_U < 0) * h_U.abs() + (h_L > 0) * h_L.abs())
        upper = torch.max(h_L.abs(), h_U.abs())
        return lower, upper


class BoundATenHeaviside(BoundOptimizableActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.alpha_batch_dim = 2

    def forward(self, *x):
        self.input_shape = x[0].shape
        # x[0]: input; x[1]: value when x == 0
        return torch.heaviside(x[0], x[1])

    def init_opt_parameters(self, start_nodes):
        l = self.inputs[0].lower
        for ns, size_s, _ in start_nodes:
            self.alpha[ns] = torch.zeros_like(l).unsqueeze(0).repeat(2, *[1] * l.ndim).requires_grad_(True)

    def clip_alpha_(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def bound_backward(self, last_lA, last_uA, *x, start_node=None, start_shape=None):
        x = x[0]
        if x is not None:
            lb_r = x.lower
            ub_r = x.upper
        else:
            lb_r = self.lower
            ub_r = self.upper

        if self.opt_stage not in ['opt', 'reuse']:
            # zero slope:
            upper_d = torch.zeros_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
            lower_d = torch.zeros_like(ub_r, device=ub_r.device, dtype=ub_r.dtype)
        else:
            upper_d = self.alpha[start_node.name][0].clamp(0, 1) * (1. / (-lb_r).clamp(min=1e-3))
            lower_d = self.alpha[start_node.name][1].clamp(0, 1) * (1. / (ub_r.clamp(min=1e-3)))

        upper_b = torch.ones_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
        lower_b = torch.zeros_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
        # For stable neurons, set fixed slope and bias.
        ub_mask = (ub_r <= 0).to(dtype=ub_r.dtype)
        lb_mask = (lb_r >= 0).to(dtype=lb_r.dtype)
        upper_b = upper_b - upper_b * ub_mask
        lower_b = lower_b * (1. - lb_mask) + lb_mask
        upper_d = upper_d - upper_d * ub_mask - upper_d * lb_mask
        lower_d = lower_d - lower_d * lb_mask - lower_d * ub_mask
        upper_d = upper_d.unsqueeze(0)
        lower_d = lower_d.unsqueeze(0)
        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias = (pos_uA * upper_b + neg_uA * lower_b).flatten(2).sum(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias = (pos_lA * lower_b + neg_lA * upper_b).flatten(2).sum(-1)

        return [(lA, uA), (None, None)], lbias, ubias
