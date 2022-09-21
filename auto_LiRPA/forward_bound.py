from auto_LiRPA.beta_crown import print_optimized_beta
import torch
from torch import Tensor
import warnings
from .bound_ops import *
from .utils import *
from .backward_bound import batched_backward
from .linear_bound import LinearBound
from .perturbations import PerturbationLpNorm

import sys
sys.setrecursionlimit(1000000)

def forward_general(self, C=None, node=None, concretize=False, offset=0):
    if self.bound_opts['dynamic_forward']:
        return self.forward_general_dynamic(C, node, concretize, offset)

    if C is None:
        if hasattr(node, 'linear'):
            return node.linear.lower, node.linear.upper
        if not node.from_input:
            node.linear = LinearBound(None, node.value, None, node.value, node.value, node.value)
            return node.value, node.value
        if not node.perturbed:
            node.lower = node.upper = self.get_forward_value(node)
        if hasattr(node, 'lower'):
            node.linear = LinearBound(None, node.lower, None, node.upper, node.lower, node.upper)
            return node.lower, node.upper

    for l_pre in node.inputs:
        if not hasattr(l_pre, 'linear'):
            self.forward_general(node=l_pre, offset=offset)
    inp = [l_pre.linear for l_pre in node.inputs]
    node._start = '_forward'
    if (C is not None and isinstance(node, BoundLinear) and 
            not node.is_input_perturbed(1) and not node.is_input_perturbed(2)):
        linear = node.bound_forward(self.dim_in, *inp, C=C)
        C_merged = True
    else:
        linear = node.linear = node.bound_forward(self.dim_in, *inp)
        C_merged = False

    lw, uw = linear.lw, linear.uw
    lower, upper = linear.lb, linear.ub

    if C is not None and not C_merged:
        # FIXME use bound_forward of BoundLinear
        C_pos, C_neg = C.clamp(min=0), C.clamp(max=0)
        _lw = torch.matmul(lw, C_pos.transpose(-1, -2)) + torch.matmul(uw, C_neg.transpose(-1, -2))
        _uw = torch.matmul(uw, C_pos.transpose(-1, -2)) + torch.matmul(lw, C_neg.transpose(-1, -2))
        lw, uw = _lw, _uw
        _lower = torch.matmul(lower.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                    torch.matmul(upper.unsqueeze(1), C_neg.transpose(-1, -2))
        _upper = torch.matmul(upper.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                    torch.matmul(lower.unsqueeze(1), C_neg.transpose(-1, -2))
        lower, upper = _lower.squeeze(1), _upper.squeeze(1)

    logger.debug(f'Forward bounds to {node}')

    if concretize:
        if lw is not None or uw is not None:
            prev_dim_in = 0
            batch_size = lw.shape[0]
            assert (lw.ndim > 1)
            lA = lw.reshape(batch_size, self.dim_in, -1).transpose(1, 2)
            uA = uw.reshape(batch_size, self.dim_in, -1).transpose(1, 2)
            for i in range(len(self.root)):
                if hasattr(self.root[i], 'perturbation') and self.root[i].perturbation is not None:
                    _lA = lA[:, :, prev_dim_in : (prev_dim_in + self.root[i].dim)]
                    _uA = uA[:, :, prev_dim_in : (prev_dim_in + self.root[i].dim)]
                    lower = lower + self.root[i].perturbation.concretize(
                        self.root[i].center, _lA, sign=-1, aux=self.root[i].aux).view(lower.shape)
                    upper = upper + self.root[i].perturbation.concretize(
                        self.root[i].center, _uA, sign=+1, aux=self.root[i].aux).view(upper.shape)
                    prev_dim_in += self.root[i].dim
        linear.lower, linear.upper = lower, upper

        if C is None:          
            node.linear = linear
            node.lower, node.upper = lower, upper
        
        if self.bound_opts['forward_refinement']:
            need_refinement = False
            for out in node.output_name:
                out_node = self[out]
                if getattr(out_node, 'nonlinear', False):
                    need_refinement = True
                for i in getattr(out_node, 'requires_input_bounds', []):
                    if out_node.inputs[i] == node:
                        need_refinement = True
                        break
            if need_refinement:
                forward_refinement(self, node)
        return lower, upper


def forward_general_dynamic(
        self, C=None, node=None, concretize=False, offset=0):
    max_dim = self.bound_opts['forward_max_dim']
    
    if C is None:
        if hasattr(node, 'linear'):
            assert not concretize

            linear = node.linear
            if offset == 0:
                if linear.lw is None:
                    return linear
                elif linear.lw.shape[1] <= max_dim:
                    return linear
            if linear.lw is not None:
                lw = linear.lw[:, offset:offset+max_dim]
                x_L = linear.x_L[:, offset:offset+max_dim]
                x_U = linear.x_U[:, offset:offset+max_dim]
                tot_dim = linear.tot_dim
                if offset == 0:
                    lb = linear.lb
                else:
                    lb = torch.zeros_like(linear.lb)
            else:
                lw = x_L = x_U = None
                tot_dim = 0
                lb = linear.lb
            return LinearBound(
                lw, lb, lw, lb, x_L=x_L, x_U=x_U,
                offset=offset, tot_dim=tot_dim,
            )

        # These cases have no coefficient tensor
        if not node.from_input:
            if concretize:
                return node.value, node.value
            else:
                node.linear = LinearBound(
                    None, node.value, None, node.value, node.value, node.value)
                return node.linear
        if not node.perturbed:
            if not hasattr(node, 'lower'):
                node.lower = node.upper = self.get_forward_value(node)
            raise NotImplementedError
            if concretize:
                return node.lower, node.upper
            else:
                if offset > 0:
                    lb = torch.zeros_like(node.lower)                
                else:
                    lb = node.lower
                node.linear = LinearBound(None, lb, None, lb, node.lower, node.upper)
                return node.linear

    if offset == 0:
        logger.debug(f'forward_general_dynamic: node={node}')

    inp = []
    for l_pre in node.inputs:
        linear_inp = self.forward_general_dynamic(node=l_pre, offset=offset)
        linear_inp.lower = getattr(l_pre, 'lower', None)
        linear_inp.upper = getattr(l_pre, 'upper', None)
        inp.append(linear_inp)
    node._start = '_forward'
    if (C is not None and isinstance(node, BoundLinear) and 
            not node.is_input_perturbed(1) and not node.is_input_perturbed(2)):
        linear = node.bound_dynamic_forward(
            *inp, C=C, max_dim=max_dim, offset=offset)
        C_merged = True
    else:
        linear = node.bound_dynamic_forward(
            *inp, max_dim=max_dim, offset=offset)
        C_merged = False
    if offset > 0:
        linear.lb = linear.ub = torch.zeros_like(linear.lb)

    lw, lb, tot_dim = linear.lw, linear.lb, linear.tot_dim
    #logger.debug(f'forward_general_dynamic: node={node}, w_size={lw.shape[1]}, tot_dim={tot_dim}')

    if C is not None and not C_merged:
        # FIXME use bound_forward of BoundLinear
        lw = torch.matmul(lw, C.transpose(-1, -2))
        lb = torch.matmul(lb.unsqueeze(1), C.transpose(-1, -2)).squeeze(1)

    if concretize:
        lower = upper = lb
        if lw is not None:
            batch_size = lw.shape[0]
            assert (lw.ndim > 1)
            if lw.shape[1] > 0:
                A = lw.reshape(batch_size, lw.shape[1], -1).transpose(1, 2)
                ptb = PerturbationLpNorm(x_L=linear.x_L, x_U=linear.x_U)
                lower = lower + ptb.concretize(x=None, A=A, sign=-1).view(lb.shape)
                upper = upper + ptb.concretize(x=None, A=A, sign=1).view(lb.shape)
            offset_next = offset + max_dim
            more = offset_next < tot_dim
        else:
            more = False

        if C is None and offset == 0 and not more:
            node.linear = linear

        if more:
            if lw is not None and lw.shape[1] > 0:
                del A
                del ptb
                del lw
                del linear
            del inp
            # TODO make it non-recursive
            lower_next, upper_next = self.forward_general_dynamic(
                C, node, concretize=True, offset=offset_next)
            lower = lower + lower_next
            upper = upper + upper_next

        if C is None:
            node.lower, node.upper = lower, upper
        
        return lower, upper
    else:
        return linear


def clean_memory(self, node):
    """ Remove linear bounds that are no longer needed. """
    # TODO add an option to retain these bounds
    
    for inp in node.inputs:
        if hasattr(inp, 'linear') and inp.linear is not None:
            clean = True
            for out in inp.output_name:
                out_node = self[out]
                if not (hasattr(out_node, 'linear') and out_node.linear is not None):
                    clean = False
            if clean:
                if isinstance(inp.linear, tuple):
                    for item in inp.linear:
                        del item
                delattr(inp, 'linear')


def forward_refinement(self, node):
    """ Refine forward bounds with backward bound propagation 
    (only refine unstable positions). """
    unstable_size_before = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    if unstable_size_before == 0:
        return
    unstable_idx, unstable_size = self.get_unstable_locations(
        node.lower, node.upper, conv=isinstance(node, BoundConv))
    logger.debug(f'Forward refinement for {node}')
    batch_size = node.lower.shape[0]
    ret = self.batched_backward(
        node, C=None, unstable_idx=unstable_idx, batch_size=batch_size)
    self.restore_sparse_bounds(
        node, unstable_idx, unstable_size, node.lower, node.upper,
        new_lower=ret[0], new_upper=ret[1])
    unstable_size_after = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    logger.debug(f'  Unstable neurons: {unstable_size_before} -> {unstable_size_after}')
    # TODO also update linear bounds?


def init_forward(self, root, dim_in):
    if dim_in == 0:
        raise ValueError("At least one node should have a specified perturbation")
    prev_dim_in = 0
    # Assumption: root[0] is the input node which implies batch_size
    batch_size = root[0].value.shape[0]
    dynamic = self.bound_opts['dynamic_forward']
    for i in range(len(root)):
        if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
            shape = root[i].linear.lw.shape
            if dynamic:
                if shape[1] != dim_in:
                    raise NotImplementedError('Dynamic forward bound is not supported yet when there are multiple perturbed inputs.')
                ptb = root[i].perturbation
                if (type(ptb) != PerturbationLpNorm or ptb.norm < np.inf
                        or ptb.x_L is None or ptb.x_U is None):
                    raise NotImplementedError(
                        'For dynamic forward bounds, only Linf (box) perturbations are supported, and x_L and x_U must be explicitly provided.')
                root[i].linear.x_L = (
                    ptb.x_L_sparse.view(batch_size, -1) if ptb.sparse 
                    else ptb.x_L.view(batch_size, -1))
                root[i].linear.x_U = (
                    ptb.x_U_sparse.view(batch_size, -1) if ptb.sparse 
                    else ptb.x_U.view(batch_size, -1))
            else:
                lw = torch.zeros(shape[0], dim_in, *shape[2:]).to(root[i].linear.lw)
                lw[:, prev_dim_in:(prev_dim_in+shape[1])] = root[i].linear.lw
                if root[i].linear.lw.data_ptr() == root[i].linear.uw.data_ptr():
                    uw = lw
                else:
                    uw = torch.zeros(shape[0], dim_in, *shape[2:]).to(root[i].linear.uw)
                    uw[:, prev_dim_in:(prev_dim_in+shape[1])] = root[i].linear.uw
                root[i].linear.lw = lw
                root[i].linear.uw = uw
            if i >= self.num_global_inputs:
                root[i].forward_value = root[i].forward_value.unsqueeze(0).repeat(
                    *([batch_size] + [1] * self.forward_value.ndim))
            prev_dim_in += shape[1]
        else:
            b = fv = root[i].forward_value
            shape = fv.shape
            if root[i].from_input:
                w = torch.zeros(shape[0], dim_in, *shape[1:], device=self.device)
                warnings.warn(f'Creating a LinearBound with zero weights with shape {w.shape}')
            else:
                w = None
            root[i].linear = LinearBound(w, b, w, b, b, b)
            root[i].lower = root[i].upper = b
            root[i].interval = (root[i].lower, root[i].upper)
