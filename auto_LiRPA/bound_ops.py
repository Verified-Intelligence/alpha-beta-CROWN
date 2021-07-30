import copy
import os
import time
from itertools import chain
import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d, \
    AdaptiveAvgPool2d, AvgPool2d, Tanh
import math
from collections import OrderedDict

from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm, PerturbationSynonym, PerturbationL0Norm
from auto_LiRPA.utils import eyeC, OneHotC, LinearBound, user_data_dir, isnan, Patches, logger, Benchmarking, prod, batched_index_select, patchesToMatrix, check_padding

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

epsilon = 1e-12


def not_implemented_op(node, func):
    message = ("Function `{}` of `{}` is not supported yet."
            " Please help to open an issue at https://github.com/KaidiXu/auto_LiRPA"
            " or implement this function in auto_LiRPA/bound_ops.py by yourself.".format(func, node))
    raise NotImplementedError(message)


"""Interval object. Used for interval bound propagation."""
class Interval(tuple):
    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, nominal=None, lower_offset=None, upper_offset=None, ptb=None):
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, nominal=None, lower_offset=None, upper_offset=None, ptb=None):
        self.nominal = nominal
        self.lower_offset = lower_offset
        self.upper_offset = upper_offset

        if ptb is None:
            self.ptb = None
            # If relative bounds are not used, `self.ptb == None` means that this interval
            # is not perturbed and it shall be treated as a constant and lb = ub.
            # But if relative bounds are used, every node in IBP is supposed to have an `Interval` object
            # even if this node is perturbed.
            if nominal is None:
                # To avoid mistakes, in this case the caller must make sure lb and ub are the same object.
                assert lb is ub
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb

    def __str__(self):
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

    @property
    def lower(self):
        return self.nominal + self.lower_offset

    @property
    def upper(self):
        return self.nominal + self.upper_offset

    """Checking if the other interval is tuple, keep the perturbation."""

    @staticmethod
    def make_interval(lb, ub, other=None, nominal=None, use_relative=False):
        if isinstance(other, Interval):
            return Interval(lb, ub, ptb=other.ptb)
        else:
            if use_relative:
                if nominal is None:
                    return Interval(
                        None, None, (lb + ub) / 2, (lb - ub) / 2, (ub - lb) / 2)
                else:
                    return Interval(None, None, nominal, lb - nominal, ub - nominal)
            else:
                return (lb, ub)

    """Given a tuple or Interval object, returns the norm and eps."""

    @staticmethod
    def get_perturbation(interval):
        if isinstance(interval, Interval) and interval.ptb is not None:
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return np.inf, 1.0
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            # elif interval.ptb is None:
            #     raise RuntimeError("get_perturbation() encountered an interval that is not perturbed.")
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return np.inf, np.nan

    """Checking if a Interval or tuple object has perturbation enabled."""

    @staticmethod
    def is_perturbed(interval):
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True

    @staticmethod
    def use_relative_bounds(*intervals):
        using = True
        for interval in intervals:
            using = using and (
                isinstance(interval, Interval) and 
                interval.nominal is not None and 
                interval.lower_offset is not None and interval.upper_offset is not None)
        return using


class Bound(nn.Module):
    r"""
    Base class for supporting the bound computation of an operator.

    Args:
        input_name (list): The name of input nodes.

        name (str): The name of this node.

        ori_name (str): Name in the original model.

        attr (dict): Attributes of the operator.

        inputs (list): A list of input nodes.

        output_index (int): The index in the output if the operator has multiple outputs. Usually output_index=0.

        options (dict): Bound options.

        device (str or torch.device): Device of the bounded module.

    Be sure to run `super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)`
    first in the `__init__` function.
    """

    def __init__(self, input_name, name, ori_name, attr={}, inputs=[], output_index=0, options={}, device=None):
        super().__init__()
        self.output_name = []
        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.options, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, options, device
        self.forward_value = None
        self.from_input = False
        self.bounded = False
        self.IBP_rets = None
        # Determine if this node has a perturbed output or not. The function BoundedModule._mark_perturbed_nodes() will set this property.
        self.perturbed = False
        if options is not None and 'loss_fusion' in options:
            self.loss_fusion = options['loss_fusion']
        else:
            self.loss_fusion = False
        self.options = options
        # Use `default_interval_propagate`
        self.use_default_ibp = False 

    """Check if the i-th input is with perturbation or not."""
    def is_input_perturbed(self, i=0):
        return self.inputs[i].perturbed

    def forward(self, *x):
        r"""
        Function for standard/clean forward. 

        Args: 
            x: A list of input values. The length of the list is equal to the number of input nodes.

        Returns:
            output (Tensor): The standard/clean output of this node.
        """
        return not_implemented_op(self, 'forward')

    def interval_propagate(self, *v):
        r"""
        Function for interval bound propagation (IBP) computation.

        There is a default function `self.default_interval_propagate(*v)` in the base class, 
        which can be used if the operator is *monotonic*. To use it, set `self.use_default_ibp = True`
        in the `__init__` function, and the implementation of this function can be skipped.

        Args: 
            v: A list of the interval bound of input nodes. 
            Generally, for each element `v[i]`, `v[i][0]` is the lower interval bound,
            and `v[i][1]` is the upper interval bound.

        Returns:
            bound: The interval bound of this node, in a same format as v[i].
        """        
        if self.use_default_ibp:
            return self.default_interval_propagate(*v)
        else:
            return not_implemented_op(self, 'interval_propagate')
        
    """For unary monotonous functions or functions for altering shapes only but not values"""
    def default_interval_propagate(self, *v):
        if len(v) == 0:
            return Interval.make_interval(self.forward(), self.forward())
        elif len(v) == 1:
            if Interval.use_relative_bounds(v[0]):
                return Interval(
                    None, None,
                    self.forward(v[0].nominal), 
                    self.forward(v[0].lower_offset), 
                    self.forward(v[0].upper_offset)
                )
            else:
                return Interval.make_interval(self.forward(v[0][0]), self.forward(v[0][1]), v[0])
        else:
            raise NotImplementedError('default_interval_propagate only supports no more than 1 input node')


    def bound_forward(self, dim_in, *x):
        r"""
        Function for forward mode bound propagation. Forward mode LiRPA computs a `LinearBound`
        instance representing the linear bound for each involved node. Major attributes of `LinearBound` include
        `lw`, `uw`, `lb`, `ub`, `lower`, and `upper`.

        `lw` and `uw` are coefficients of linear bounds w.r.t. model input. 
        Their shape is `(batch_size, dim_in, *standard_shape)`, where `dim_in` is the total dimension 
        of perturbed input nodes of the model, and `standard_shape` is the shape of the standard/clean output.
        `lb` and `ub` are bias terms of linear bounds, and their shape is equal to the shape of standard/clean output.
        `lower` and `upper` are concretized lower and upper bounds that will be computed later in BoundedModule.

        Args: 
            dim_in (int): Total dimension of perturbed input nodes of the model.
            
            x: A list of the linear bound of input nodes. Each element in x is a `LinearBound` instance.

        Returns:
            bound (LinearBound): The linear bound of this node.
        """                
        return not_implemented_op(self, 'bound_forward')

    def bound_backward(self, last_lA, last_uA, *x):
        r"""
        Function for backward mode bound propagation.

        Args: 
            last_lA (Tensor): `A` matrix for lower bound computation propagated to this node. It can be `None` if lower bound is not needed.
            
            last_uA (Tensor): `A` matrix for upper bound computation propagated to this node. It can be `None` if upper bound is not needed.
            
            x: A list of input nodes, with x[i].lower and x[i].upper that can be used as pre-activation bounds.

        Returns:
            A: A list of A matrices for the input nodes. Each element is a tuple (lA, uA).

            lbias (Tensor): The bias term for lower bound computation, introduced by the linear relaxation of this node. .

            ubias (Tensor): The bias term for upper bound computation, introduced by the linear relaxation of this node. 
        """        
        return not_implemented_op(self, 'bound_backward')

    def infer_batch_dim(self, batch_size, *x):
        return not_implemented_op(self.type, 'infer_batch_dim')

    def broadcast_backward(self, A, x):
        shape = x.output_shape
        batch_dim = max(self.batch_dim, 0)
                
        if isinstance(A, torch.Tensor):
            if x.batch_dim == -1:
                # final shape of input
                shape = torch.Size([A.shape[batch_dim + 1]] + list(shape))
                dims = []
                cnt_sum = A.ndim - len(shape) - 1
                for i in range(1, A.ndim): # merge the output dimensions?
                    if i != self.batch_dim + 1 and cnt_sum > 0:
                        dims.append(i)
                        cnt_sum -= 1
                if dims:
                    A = torch.sum(A, dim=dims)
            else:
                dims = list(range(1, 1 + A.ndim - 1 - len(shape)))
                if dims:
                    A = torch.sum(A, dim=dims)
            dims = []
            for i in range(len(shape)):
                # Skip the batch dimension.
                if shape[i] == 1 and A.shape[i + 1] != 1 and i != batch_dim:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            assert (A.shape[2:] == shape[1:])  # skip the spec and batch dimension.
        elif type(A) == Patches:
            pass

        return A

    @staticmethod
    def broadcast_forward(dim_in, x, shape_res):
        lw, lb, uw, ub = x.lw, x.lb, x.uw, x.ub
        shape_x, shape_res = list(x.lb.shape), list(shape_res)
        if lw is None:
            lw = uw = torch.zeros(dim_in, *shape_x, device=lb.device)
            has_batch_size = False
        else:
            has_batch_size = True
        while len(shape_x) < len(shape_res):
            if not has_batch_size:
                lw, uw = lw.unsqueeze(0), uw.unsqueeze(0)
                lb, ub = lb.unsqueeze(0), ub.unsqueeze(0)
                shape_x = [1] + shape_x
                has_batch_size = True
            else:
                lw, uw = lw.unsqueeze(2), uw.unsqueeze(2)
                lb, ub = lb.unsqueeze(1), ub.unsqueeze(1)
                shape_x = [shape_x[0], 1] + shape_x[1:]
        repeat = [(shape_res[i] // shape_x[i]) for i in range(len(shape_x))]
        lb, ub = lb.repeat(*repeat), ub.repeat(*repeat)
        repeat = repeat[:1] + [1] + repeat[1:]
        lw, uw = lw.repeat(*repeat), uw.repeat(*repeat)
        return lw, lb, uw, ub

    def get_bias(self, A, bias):
        if A is None:
            return 0
        if not Benchmarking:
            assert not isnan(A)
            assert not isnan(bias)

        if isinstance(A, torch.Tensor):
            if torch.norm(A, p=1) < epsilon:
                return 0
            output_dim = A.shape[0]
            if self.batch_dim != -1:
                batch_size = A.shape[self.batch_dim + 1]
                A_shape = [
                    A.shape[0],
                    np.prod(A.shape[1:self.batch_dim + 1]).astype(np.int32),
                    batch_size,
                    np.prod(A.shape[self.batch_dim + 2:]).astype(np.int32)
                ]
                A = A.reshape(*A_shape).permute(2, 0, 1, 3).reshape(batch_size, output_dim, -1)
                bias = bias.reshape(*A_shape[1:]).transpose(0, 1).reshape(batch_size, -1, 1)
                # FIXME avoid .transpose(0, 1) back
                bias_new = A.matmul(bias).squeeze(-1).transpose(0, 1)
            else:
                batch_size = A.shape[1]
                A = A.view(output_dim, batch_size, -1)
                bias_new = A.matmul(bias.view(-1))
            if isnan(bias_new):
                # NaN can be caused by 0 * inf, if 0 appears in `A` and inf appears in `bias`.
                # Force the whole bias to be 0, to avoid gradient issues. 
                # FIXME maybe find a more robust solution.
                return 0
            else:
                return bias_new
        elif type(A) == Patches:
            if torch.norm(A.patches, p = 1) < epsilon:
                return 0

            # the shape of A.patches is [batch, L, out_c, in_c, K, K]
            
            if self.batch_dim != -1:
                batch_size = bias.shape[0]
                bias, padding = check_padding(bias, A.padding)
                bias = F.unfold(bias, kernel_size=A.patches.size(-1), padding=padding, stride=A.stride).transpose(-2, -1).unsqueeze(-2)
                # Here the size of bias is [batch_size, L, 1, in_c * K * K]
                L = bias.size(1)

                # now the shape is [batch_size, L, out_c, in_c * K * K]
                patches = A.patches.reshape(A.patches.size(0), A.patches.size(1), A.patches.size(-4), A.patches.size(-1) * A.patches.size(-2) * A.patches.size(-3))
                # torch.cuda.empty_cache()
                # use torch.einsum() to save memory, equal to:
                # bias_new = (bias * patches).sum(-1).transpose(-2, -1)
                if bias.shape[1] == patches.shape[1]:
                    if bias.shape[0] == patches.shape[0]:
                        bias_new = torch.einsum('naik,najk->nja', bias, patches)
                    else:
                        bias_new = torch.einsum('naik,majk->nja', bias, patches)
                else:
                    bias_new = torch.einsum('naik,mbjk->nja', bias, patches)
                # assert torch.allclose((bias * patches).sum(-1).transpose(-2, -1), bias_new)
                bias_new = bias_new.reshape(batch_size, bias_new.size(-2), int(math.sqrt(bias_new.size(-1))), int(math.sqrt(bias_new.size(-1))))
            else:
                # Similar to BoundConstant
                patches = A.patches
                patches_reshape = torch.sum(patches, dim=(-1, -2, -3)) * bias.to(self.device)
                patches_reshape = patches_reshape.transpose(-1, -2)
                return patches_reshape.view(patches_reshape.size(0), patches_reshape.size(1), int(math.sqrt(patches_reshape.size(2))), -1).transpose(0, 1)

            return bias_new
        else:
            return NotImplementedError()

    @staticmethod
    @torch.jit.script
    def clamp_mutiply(A, pos, neg):
        Apos = A.clamp(min=0)
        Aneg = A.clamp(max=0)
        return pos.contiguous() * Apos + neg.contiguous() * Aneg, Apos, Aneg

    """save input and output shapes uniformly by the decorator"""
    @staticmethod
    def save_io_shape(func):
        def wrapper(self, *args, **kwargs):
            if len(args) > 0:
                self.input_shape = args[0].shape  # x should always be the first input

            output = func(self, *args, **kwargs)

            if isinstance(output, torch.Tensor):
                self.output_shape = output.shape
            return output

        return wrapper


    """Some operations are non-deterministic and deterministic mode will fail. So we temporary disable it."""
    def non_deter_wrapper(self, op, *args, **kwargs):
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)

class BoundReshape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = prod(x.shape) // int(prod(shape[:i]) * prod(shape[(i + 1):]))
        self.shape = shape
        return x.reshape(shape)

    def bound_backward(self, last_lA, last_uA, x, shape):
        def _bound_oneside(A):
            if A is None:
                return None
            # A shape is (spec, batch, *node_shape)
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape[1:])
        uw = x.uw.reshape(batch_size, dim_in, *self.shape[1:])
        lb = x.lb.reshape(batch_size, *self.shape[1:])
        ub = x.ub.reshape(batch_size, *self.shape[1:])
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                v[0].nominal.reshape(self.shape),
                v[0].lower_offset.reshape(self.shape),
                v[0].upper_offset.reshape(self.shape),
                ptb=v[0].ptb
            )
        return Interval.make_interval(v[0][0].reshape(*v[1][0]), v[0][1].reshape(*v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.input_shape[x[0]] == self.shape[x[0]]:
            return x[0]
        raise NotImplementedError('input shape {}, new shape {}, input batch dim {}'.format(
            self.input_shape, self.shape, x[0]
        ))


class BoundInput(Bound):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name)
        self.value = value
        self.perturbation = perturbation
        self.from_input = True

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Update perturbed property based on the perturbation set.
        if key == "perturbation":
            if self.perturbation is not None:
                self.perturbed = True
            else:
                self.perturbed = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            if len(prefix.split('.')) == 2:
                key = prefix + name
            else:
                # change key to prefix + self.ori_name when calling load_state_dict()
                key = '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if param.ndim == 0 and input_param.ndim == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = param if keep_vars else param.detach()
                else:
                    # change parameters' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = buf if keep_vars else buf.detach()
                else:
                    # change buffers' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = buf if keep_vars else buf.detach()

    @Bound.save_io_shape
    def forward(self):
        return self.value

    def bound_forward(self, dim_in):
        assert 0

    def bound_backward(self, last_lA, last_uA):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def interval_propagate(self, *v):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def infer_batch_dim(self, batch_size, *x):
        shape = self.forward_value.shape
        for i in range(len(shape)):
            if shape[i] == batch_size:
                return i
        return -1


class BoundParams(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_parameter('param', value)
        self.from_input = False
        self.initializing = False

    """Override register_parameter() hook to register only needed parameters."""

    def register_parameter(self, name, param):
        if name == 'param':
            # self._parameters[name] = param  # cannot contain '.' in name, it will cause error when loading state_dict
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def init(self, initializing=False):
        self.initializing = initializing

    @Bound.save_io_shape
    def forward(self):
        if self.initializing:
            return self.param_init
        else:
            return self.param

    def infer_batch_dim(self, batch_size, *x):
        return -1


class BoundBuffers(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_buffer('buffer', value.clone().detach())

    @Bound.save_io_shape
    def forward(self):
        return self.buffer


class BoundLinear(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # Gemm:
        # A = A if transA == 0 else A.T
        # B = B if transB == 0 else B.T
        # C = C if C is not None else np.array(0)
        # Y = alpha * np.dot(A, B) + beta * C
        # return Y

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        # Defaults in ONNX
        self.transA = 0
        self.transB = 0
        self.alpha = 1.0
        self.beta = 1.0
        if attr is not None:
            self.transA = attr['transA'] if 'transA' in attr else self.transA
            self.transB = attr['transB'] if 'transB' in attr else self.transB
            self.alpha = attr['alpha'] if 'alpha' in attr else self.alpha
            self.beta = attr['beta'] if 'beta' in attr else self.beta

        self.opt_matmul = options.get('matmul')

    """Handle tranpose and linear coefficients."""
    def _preprocess(self, a, b, c=None):
        if self.transA and isinstance(a, torch.Tensor):
            a = a.transpose(-2,-1)
        if self.alpha != 1.0:
            a = self.alpha * a
        if not self.transB and isinstance(b, torch.Tensor):
            # our code assumes B is transposed (common case), so we transpose B only when it is not transposed in gemm.
            b = b.transpose(-2,-1)
        if c is not None:
            if self.beta != 1.0:
                c = self.beta * c
        return a, b, c

    @Bound.save_io_shape
    def forward(self, x, w, b=None):
        x, w, b = self._preprocess(x, w, b)
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res

    """Multiply weight matrix with a diagonal matrix with selected rows."""
    def onehot_mult(self, weight, bias, C, batch_size):

        if C is None:
            return None, 0.0

        new_weight = None
        new_biase = 0.0
        if C.index.ndim == 1:
            # Every element in the batch shares the same rows.
            if weight is not None:
                new_weight = self.non_deter_index_select(weight, dim=0, index=C.index).unsqueeze(1).expand([-1, batch_size] + [-1] * (weight.ndim - 1))
            if bias is not None:
                new_bias = self.non_deter_index_select(bias, dim=0, index=C.index).unsqueeze(1).expand(-1, batch_size)
        elif C.index.ndim == 2:
            # Every element in the batch has different rows, but the number of rows are the same. This essentially needs a batched index_select function.
            if weight is not None:
                new_weight = batched_index_select(weight.unsqueeze(0), dim=1, index=C.index)
            if bias is not None:
                new_bias = batched_index_select(bias.unsqueeze(0), dim=1, index=C.index)
        if C.coeffs is not None:
            if weight is not None:
                new_weight = new_weight * C.coeffs.unsqueeze(-1)
            if bias is not None:
                new_bias = new_bias * C.coeffs
        if C.index.ndim == 2:
            # Eventually, the shape of A is [spec, batch, *node] so need a transpose.
            new_weight = new_weight.transpose(0, 1)
            new_bias = new_bias.transpose(0, 1)
        return new_weight, new_bias


    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2 or len(x) == 3
        has_bias = len(x) == 3
        # x[0]: input node, x[1]: weight, x[2]: bias
        input_lb = [xi.lower if hasattr(xi, 'lower') else None for xi in x]
        input_ub = [xi.upper if hasattr(xi, 'upper') else None for xi in x]
        # transpose and scale each term if necessary.
        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if last_lA is not None else last_uA.shape[1]

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat([1, batch_size] + [1] * (input_lb[1].ndim - 1))
                # Bias will be directly added to output.
                if has_bias:
                    lbias = ubias = input_lb[2].unsqueeze(1).repeat(1, batch_size)
            elif isinstance(last_lA, OneHotC) or isinstance(last_uA, OneHotC):
                # We need to select several rows from the weight matrix (its shape is output_size * input_size).
                lA_x, lbias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_lA, batch_size)
                if last_lA is last_uA:
                    uA_x = lA_x
                    ubias = lbias
                else:
                    uA_x, ubias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_uA, batch_size)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    # Just multiply this layer's weight into bound matrices, and produce biases.
                    next_A = last_A.to(input_lb[1]).matmul(input_lb[1])
                    sum_bias = (last_A.to(input_lb[2]).matmul(input_lb[2]) 
                        if has_bias else 0.0)
                    return next_A, sum_bias

                lA_x, lbias = _bound_oneside(last_lA)
                uA_x, ubias = _bound_oneside(last_uA)

        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            # Obtain relaxations for matrix multiplication.
            [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.bound_backward_with_weight(last_lA, last_uA, input_lb, input_ub, x[0], x[1])
            if has_bias:
                if x[2].perturbation is not None:
                    # Bias is also perturbed. Since bias is directly added to the output, in backward mode it is treated
                    # as an input with last_lA and last_uA as associated bounds matrices.
                    # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
                    lA_bias = last_lA
                    uA_bias = last_uA
                else:
                    # Bias not perturbed, so directly adding the bias of this layer to the final bound bias term.
                    if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                        # Bias will be directly added to output.
                        lbias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                        ubias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                    else:
                        if last_lA is not None:
                            lbias += last_lA.matmul(input_lb[2])
                        if last_uA is not None:
                            ubias += last_uA.matmul(input_lb[2])
            # If not has_bias, no need to compute lA_bias and uA_bias

        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat([1, batch_size] + [1] * (input_lb[1].ndim - 1))
            else:
                lA_x = last_lA.matmul(input_lb[1])
                uA_x = last_uA.matmul(input_lb[1])
            # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
            lA_bias = last_lA
            uA_bias = last_uA

        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def _reshape(self, x_l, x_u, y_l, y_u):
        x_shape, y_shape = self.input_shape, self.y_shape

        # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)

        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        return x_l, x_u, y_l, y_u

    def _relax(self, input_lb, input_ub):
        return BoundMul.get_bound_mul(*self._reshape(input_lb[0], input_ub[0], input_lb[1], input_ub[1]))

    def bound_backward_with_weight(self, last_lA, last_uA, input_lb, input_ub, x, y):
        # Note: x and y are not tranposed or scaled, and we should avoid using them directly.
        # Use input_lb and input_ub instead.
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(input_lb, input_ub)

        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)
        x_shape, y_shape = input_lb[0].size(), input_lb[1].size()
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(x_shape[0], -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(x_shape[0], -1, 1)

        if len(x.output_shape) != 2 and len(x.output_shape) == len(y.output_shape):
            dim_y = [-3]
        elif len(y.output_shape) == 2:
            dim_y = list(range(2, 2 + len(x_shape) - 2))
        else:
            raise NotImplementedError

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            if isinstance(last_A, eyeC):
                A_x = alpha_pos.squeeze(0).permute(1, 0, 2).repeat(1, last_A.shape[1], 1)
                A_y = beta_pos * torch.eye(last_A.shape[2], device=last_A.device) \
                    .view((last_A.shape[2], 1, last_A.shape[2], 1))
                if len(dim_y) != 0:
                    A_y = torch.sum(beta_pos, dim=dim_y)
                bias = gamma_pos.transpose(0, 1)
            else:
                # last_uA has size (batch, spec, output)
                last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
                last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
                # alpha_u has size (batch, spec, output, input)
                # uA_x has size (batch, spec, input).
                # uA_x = torch.sum(last_uA_pos * alpha_u + last_uA_neg * alpha_l, dim=-2)
                A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) + \
                       alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
                # beta_u has size (batch, spec, output, input)
                # uA_y is for weight matrix, with parameter size (output, input)
                # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
                A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
                if len(dim_y) != 0:
                    A_y = torch.sum(A_y, dim=dim_y)
                # last_uA has size (batch, spec, output)
                _last_A_pos = last_A_pos.reshape(last_A.shape[0], last_A.shape[1], -1)
                _last_A_neg = last_A_neg.reshape(last_A.shape[0], last_A.shape[1], -1)
                # gamma_u has size (batch, output, 1)
                # ubias has size (batch, spec, 1)
                bias = _last_A_pos.transpose(0, 1).matmul(gamma_pos).transpose(0, 1) + \
                       _last_A_neg.transpose(0, 1).matmul(gamma_neg).transpose(0, 1)
            bias = bias.squeeze(-1)
            return A_x, A_y, bias

        lA_x, lA_y, lbias = _bound_oneside(last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def _propagate_Linf(x, w):
        if Interval.use_relative_bounds(x):
            if len(x.nominal.shape) == 2 and w.ndim == 3:
                nominal = torch.bmm(x.nominal.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                lower_offset = (
                    torch.bmm(x.lower_offset.unsqueeze(1), w.clamp(min=0).transpose(-1, -2)) + 
                    torch.bmm(x.upper_offset.unsqueeze(1), w.clamp(max=0).transpose(-1, -2))).squeeze(1)
                upper_offset = (
                    torch.bmm(x.lower_offset.unsqueeze(1), w.clamp(max=0).transpose(-1, -2)) + 
                    torch.bmm(x.upper_offset.unsqueeze(1), w.clamp(min=0).transpose(-1, -2))).squeeze(1)
            else:
                nominal = x.nominal.matmul(w.transpose(-1, -2))
                lower_offset = (
                    x.lower_offset.matmul(w.clamp(min=0).transpose(-1, -2)) + 
                    x.upper_offset.matmul(w.clamp(max=0).transpose(-1, -2)))
                upper_offset = (
                    x.lower_offset.matmul(w.clamp(max=0).transpose(-1, -2)) + 
                    x.upper_offset.matmul(w.clamp(min=0).transpose(-1, -2)))
            return Interval(None, None, nominal, lower_offset, upper_offset)

        h_L, h_U = x
        mid = (h_L + h_U) / 2
        diff = (h_U - h_L) / 2
        w_abs = w.abs()
        if mid.ndim == 2 and w.ndim == 3:
            center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
            deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
        else:
            center = mid.matmul(w.transpose(-1, -2))
            deviation = diff.matmul(w_abs.transpose(-1, -2))
        return center, deviation

    def interval_propagate(self, *v, C=None, w=None):
        has_bias = len(v) == 3
        if self is not None:
            # This will convert an Interval object to tuple. We need to add perturbation property later.
            if Interval.use_relative_bounds(v[0]):
                v_nominal = self._preprocess(v[0].nominal, v[1].nominal, v[2].nominal)
                v_lower_offset = self._preprocess(v[0].lower_offset, v[1].lower_offset, v[2].lower_offset)
                v_upper_offset = self._preprocess(v[0].upper_offset, v[1].upper_offset, v[2].upper_offset)
                v = [Interval(None, None, bounds[0], bounds[1], bounds[2]) 
                    for bounds in zip(v_nominal, v_lower_offset, v_upper_offset)]
            else:
                v_lb, v_ub = zip(*v)
                v_lb = self._preprocess(*v_lb)
                v_ub = self._preprocess(*v_ub)
                # After preprocess the lower and upper bounds, we make them Intervals again.
                v = [Interval.make_interval(bounds[0], bounds[1], bounds[2]) for bounds in zip(v_lb, v_ub, v)]
        if w is None and self is None:
            # Use C as the weight, no bias.
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                # No specified weight, use this layer's weight.
                if self.is_input_perturbed(1):  # input index 1 is weight.
                    # w is a perturbed tensor. Use IBP with weight perturbation.
                    # C matrix merging not supported.
                    assert C is None
                    res = self.interval_propagate_with_weight(*v)
                    if Interval.use_relative_bounds(res):
                        if has_bias:
                            raise NotImplementedError
                        else:
                            return res
                    else:
                        l, u = res
                        if has_bias:
                            return l + v[2][0], u + v[2][1]
                        else:
                            return l, u
                else:
                    # Use weight 
                    if Interval.use_relative_bounds(v[1]):
                        w = v[1].nominal
                    else:
                        w = v[1][0]
            if has_bias:
                lb, ub = (v[2].lower, v[2].upper) if Interval.use_relative_bounds(v[2]) else v[2]
            else:
                lb = ub = 0.0

            if C is not None:
                w = C.matmul(w)
                lb = C.matmul(lb) if not isinstance(lb, float) else lb
                ub = C.matmul(ub) if not isinstance(ub, float) else ub

        # interval_propagate() of the Linear layer may encounter input with different norms.
        norm, eps = Interval.get_perturbation(v[0])[:2]
        if norm == np.inf:
            interval = BoundLinear._propagate_Linf(v[0], w)
            if isinstance(interval, Interval):
                b_center = (lb + ub) / 2
                interval.nominal += b_center
                interval.lower_offset += lb - b_center
                interval.upper_offset += ub - b_center
                return interval
            else:
                center, deviation = interval
        elif norm > 0:
            # General Lp norm.
            norm, eps = Interval.get_perturbation(v[0])
            mid = v[0][0]
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            deviation = w.norm(dual_norm, dim=-1) * eps
        else: # here we calculate the L0 norm IBP bound of Linear layers, using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = v[0][0]
            weight_abs = w.abs()
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            # L0 norm perturbation
            k = int(eps)
            deviation = torch.sum(torch.topk(weight_abs, k)[0], dim=1) * ratio

        lower, upper = center - deviation + lb, center + deviation + ub

        return (lower, upper)

    def interval_propagate_with_weight(self, *v):
        input_norm, input_eps = Interval.get_perturbation(v[0])
        weight_norm, weight_eps = Interval.get_perturbation(v[1])        

        if Interval.use_relative_bounds(*v):
            assert input_norm == weight_norm == np.inf
            assert self.opt_matmul == 'economic'
            
            x, y = v[0], v[1]

            nominal = x.nominal.matmul(y.nominal.transpose(-1, -2))

            matmul_offset = torch.matmul(
                torch.max(x.lower_offset.abs(), x.upper_offset.abs()),
                torch.max(y.upper_offset.abs(), y.lower_offset.abs()).transpose(-1, -2))

            lower_offset = (
                x.nominal.clamp(min=0).matmul(y.lower_offset.transpose(-1, -2)) + 
                x.nominal.clamp(max=0).matmul(y.upper_offset.transpose(-1, -2)) + 
                x.lower_offset.matmul(y.nominal.clamp(min=0).transpose(-1, -2)) + 
                x.upper_offset.matmul(y.nominal.clamp(max=0).transpose(-1, -2)) - matmul_offset)
            
            upper_offset = (
                x.nominal.clamp(min=0).matmul(y.upper_offset.transpose(-1, -2)) + 
                x.nominal.clamp(max=0).matmul(y.lower_offset.transpose(-1, -2)) + 
                x.upper_offset.matmul(y.nominal.clamp(min=0).transpose(-1, -2)) + 
                x.lower_offset.matmul(y.nominal.clamp(max=0).transpose(-1, -2)) + matmul_offset)

            return Interval(None, None, nominal, lower_offset, upper_offset)

        self.x_shape = v[0][0].shape
        self.y_shape = v[1][0].shape

        if input_norm == np.inf and weight_norm == np.inf:
            # A memory-efficient implementation without expanding all the elementary multiplications
            if self.opt_matmul == 'economic':
                x_l, x_u = v[0][0], v[0][1]
                y_l, y_u = v[1][0].transpose(-1, -2), v[1][1].transpose(-1, -2)

                dx, dy = F.relu(x_u - x_l), F.relu(y_u - y_l)
                base = x_l.matmul(y_l)

                mask_xp, mask_xn = (x_l > 0).float(), (x_u < 0).float()
                mask_xpn = 1 - mask_xp - mask_xn
                mask_yp, mask_yn = (y_l > 0).float(), (y_u < 0).float()
                mask_ypn = 1 - mask_yp - mask_yn

                lower, upper = base.clone(), base.clone()

                lower += dx.matmul(y_l.clamp(max=0)) - (dx * mask_xn).matmul(y_l * mask_ypn)
                upper += dx.matmul(y_l.clamp(min=0)) + (dx * mask_xp).matmul(y_l * mask_ypn)

                lower += x_l.clamp(max=0).matmul(dy) - (x_l * mask_xpn).matmul(dy * mask_yn)
                upper += x_l.clamp(min=0).matmul(dy) + (x_l * mask_xpn).matmul(dy * mask_yp)

                lower += (dx * mask_xn).matmul(dy * mask_yn)
                upper += (dx * (mask_xpn + mask_xp)).matmul(dy * (mask_ypn + mask_yp))
            else:
                # Both input data and weight are Linf perturbed (with upper and lower bounds).
                # We need a x_l, x_u for each row of weight matrix.
                x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
                y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
                # Reuse the multiplication bounds and sum over results.
                lower, upper = BoundMul.interval_propagate(*[(x_l, x_u), (y_l, y_u)])
                lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)

            return lower, upper
        elif input_norm == np.inf and weight_norm == 2:
            # This eps is actually the epsilon per row, as only one row is involved for each output element.
            eps = weight_eps
            # Input data and weight are Linf perturbed (with upper and lower bounds).
            h_L, h_U = v[0]
            # First, handle non-perturbed weight with Linf perturbed data.
            center, deviation = BoundLinear._propagate_Linf(v[0], v[1][0])
            # Compute the maximal L2 norm of data. Size is [batch, 1].
            max_l2 = torch.max(h_L.abs(), h_U.abs()).norm(2, dim=-1).unsqueeze(-1)
            # Add the L2 eps to bounds.
            lb, ub = center - deviation - max_l2 * eps, center + deviation + max_l2 * eps
            return lb, ub
        else:
            raise NotImplementedError(
                "Unsupported perturbation combination: data={}, weight={}".format(input_norm, weight_norm))

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None, C=None):
        has_bias = b is not None
        x, w, b = self._preprocess(x, w, b)

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            if isinstance(w, LinearBound):
                w = w.lower
            if isinstance(b, LinearBound):
                b = b.lower
            if C is not None:
                w = C.to(w).matmul(w).transpose(-1, -2)
                if b is not None:
                    b = C.to(b).matmul(b)
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = (x.lb.unsqueeze(1).matmul(w_pos) + x.ub.unsqueeze(1).matmul(w_neg)).squeeze(1)
                ub = (x.ub.unsqueeze(1).matmul(w_pos) + x.lb.unsqueeze(1).matmul(w_neg)).squeeze(1)
            else:               
                w = w.t()
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = x.lb.matmul(w_pos) + x.ub.matmul(w_neg)
                ub = x.ub.matmul(w_pos) + x.lb.matmul(w_neg)
            lw = x.lw.matmul(w_pos) + x.uw.matmul(w_neg)
            uw = x.uw.matmul(w_pos) + x.lw.matmul(w_neg)
            if b is not None:
                lb += b
                ub += b
        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            if C is not None:
                raise NotImplementedError
            res = self.bound_forward_with_weight(dim_in, x, w)
            if has_bias:
                raise NotImplementedError
            lw, lb, uw, ub = res.lw, res.lb, res.uw, res.ub
        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError

        return LinearBound(lw, lb, uw, ub)

    def bound_forward_with_weight(self, dim_in, x, y):
        x_unsqueeze = LinearBound(
            x.lw.unsqueeze(-2),
            x.lb.unsqueeze(-2),
            x.uw.unsqueeze(-2),
            x.ub.unsqueeze(-2),
            x.lower.unsqueeze(-2),
            x.upper.unsqueeze(-2),
        )
        y_unsqueeze = LinearBound(
            y.lw.unsqueeze(-3),
            y.lb.unsqueeze(-3),
            y.uw.unsqueeze(-3),
            y.ub.unsqueeze(-3),
            y.lower.unsqueeze(-3),
            y.upper.unsqueeze(-3),
        )
        res_mul = BoundMul.bound_forward(dim_in, x_unsqueeze, y_unsqueeze)
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1)
        )

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundBatchNormalization(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device, training):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.eps = attr['epsilon']
        self.momentum = round(1 - attr['momentum'], 5)  # take care!
        self.mode = options.get("conv_mode", "matrix")
        self.bn_mode = options.get("bn", "forward")
        # self.num_batches_tracked = 0 # not support yet
        self.to(device)
        self.training = training

    @Bound.save_io_shape
    def forward(self, x, w, b, m, v):
        if self.training:
            dim = [0] + list(range(2, x.ndim))
            self.current_mean = x.mean(dim)
            self.current_var = x.var(dim, unbiased=False)
            return F.batch_norm(x, m, v, w, b, self.training, self.momentum, self.eps)
        else:
            self.current_mean = m.data.clone()
            self.current_var = v.data.clone()
            return F.batch_norm(x, m, v, w, b, self.training, self.momentum, self.eps)

    def bound_backward(self, last_lA, last_uA, *x):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        # x[0]: input, x[1]: weight, x[2]: bias, x[3]: running_mean, x[4]: running_var
        weight, bias = x[1].param, x[2].param

        if not self.training:
            self.current_mean = x[3].value
            self.current_var = x[4].value

        tmp_bias = bias - self.current_mean / torch.sqrt(self.current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) == torch.Tensor:
                next_A = last_A * tmp_weight.view(*((1, 1, -1) + (1,) * (last_A.ndim - 3)))
                if last_A.ndim > 3:
                    sum_bias = (last_A.sum(tuple(range(3, last_A.ndim))) * tmp_bias).sum(2)
                else:
                    sum_bias = (last_A * tmp_bias).sum(2)
            elif type(last_A) == Patches:
                # TODO Only 4-dim BN supported in the Patches mode
                if last_A.identity == 0:
                    patches = last_A.patches

                    patches = patches * tmp_weight.view(-1, 1, 1)
                    next_A = Patches(patches, last_A.stride, last_A.padding, last_A.shape, identity=0)
                    
                    bias = tmp_bias.view(1,-1,1,1).expand(self.input_shape)
                    tmp_shape = [self.input_shape[0], self.input_shape[-1]*self.input_shape[-2]] if last_A.shape[1] == 1 else list(last_A.shape[:2])
                    bias = F.unfold(bias, last_A.patches.size(-1), padding=last_A.padding, stride=last_A.stride).transpose(-1,-2).view(tmp_shape + [1] + list(last_A.shape[3:]))
                    sum_bias = (last_A.patches * bias).sum((-1,-2,-3)).transpose(-1, -2)
                    sum_bias = sum_bias.view(sum_bias.size(0), sum_bias.size(1), int(math.sqrt(sum_bias.size(2))), int(math.sqrt(sum_bias.size(2)))).transpose(0, 1)
                else:
                    # we should create a real identity Patch
                    num_channel = tmp_weight.view(-1).size(0)
                    batch_size = last_A.shape[0]
                    w = int(math.sqrt(last_A.shape[1]))
                    patches = (torch.eye(num_channel, device=tmp_weight.device) * tmp_weight.view(-1)).unsqueeze(0).unsqueeze(0).unsqueeze(4).unsqueeze(5) # now [1 * 1 * in_C * in_C * 1 * 1]
                    patches = patches.expand(batch_size, w*w, num_channel, num_channel, 1, 1)
                    
                    next_A = Patches(patches, 1, 0, last_A.shape)
                    
                    sum_bias = tmp_bias.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(num_channel, batch_size, w, w) # squeezing batch dim, now [C * 1 * 1 * 1]
            else:
                raise NotImplementedError()
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)

        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], lbias, ubias

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        h_L, h_U = v[0]
        weight, bias = v[1][0], v[2][0]

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        # Use `mid` in IBP to compute mean and variance for BN.
        # In this case, `forward` should not have been called.
        if self.bn_mode == 'ibp' and not hasattr(self, 'forward_value'):
            m, v, w, b = tuple(self.inputs[i].forward() for i in range(1, 5))
            self.forward(mid, m, v, w, b)

        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = bias - self.current_mean * tmp_weight

        shape = (1, -1) + (1,) * (mid.ndim - 2)
        center = tmp_weight.view(*shape) * mid + tmp_bias.view(*shape)
        deviation = tmp_weight_abs.view(*shape) * diff
        lower = center - deviation
        upper = center + deviation

        return lower, upper

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundConv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False
        self.to(device)
        self.mode = options.get("conv_mode", "matrix")

    @Bound.save_io_shape
    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv2d(x[0], x[1], bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].lower

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) is OneHotC:
                # Conv layer does not support the OneHotC fast path. We have to create a dense matrix instead.
                assert OneHotC.index.ndim == 2

            if type(last_A) == torch.Tensor:
                shape = last_A.size()
                # when (W−F+2P)%S != 0, construct the output_padding
                output_padding0 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[0] + 2 * \
                                self.padding[0] - 1 - (int(weight.size()[2] - 1) * self.dilation[0])
                output_padding1 = int(self.input_shape[3]) - (int(self.output_shape[3]) - 1) * self.stride[1] + 2 * \
                                self.padding[1] - 1 - (int(weight.size()[3] - 1) * self.dilation[0])
                next_A = F.conv_transpose2d(last_A.reshape(shape[0] * shape[1], *shape[2:]), weight, None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, output_padding=(output_padding0, output_padding1))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                if self.has_bias:
                    sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
                else:
                    sum_bias = 0
                return next_A, sum_bias
            elif type(last_A) == Patches:
                # Here we build and propagate a Patch object with (patches, stride, padding)
                # shape of the patches is [batch_size, L, out_c, in_c, K * K]

                # Now we only consider last_A that are Patches objects, which means that we only consider conv layers
                assert type(last_A) == Patches
                if last_A.identity == 0:
                    # [batch_size, L, out_c, in_c, K, K]
                    patches = last_A.patches
                    batch_size = last_A.patches.size(0)
                    L = patches.size(1)
                    out_c = patches.size(2)
                    patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))

                    pieces = F.conv_transpose2d(patches, weight, stride=self.stride)
                    pieces = pieces.view(batch_size, L, out_c, pieces.size(-3), pieces.size(-2), pieces.size(-1))

                    if self.has_bias:
                        # use torch.einsum() to save memory, equal to:
                        # sum_bias = (last_A.patches.sum((-1, -2)) * x[2].lower).sum(-1).transpose(-2, -1)

                        # sum_bias = torch.einsum('naik...,k->nia', last_A.patches, x[2].lower)  # this even worse, why?
                        sum_bias = torch.einsum('naik,k->nia', last_A.patches.sum((-1, -2)), x[2].lower)
                        # assert torch.allclose((last_A.patches.sum((-1, -2)) * x[2].lower).sum(-1).transpose(-2, -1),  sum_bias)
                        sum_bias = sum_bias.view(batch_size, -1, int(math.sqrt(L)), int(math.sqrt(L))).transpose(0, 1)
                        # shape of the bias is [batch_size, L, out_channel]
                    else:
                        sum_bias = 0
                elif last_A.identity == 1:
                    pieces = weight.view(1, 1, weight.size(0), weight.size(1), weight.size(2), weight.size(3)).expand(last_A.shape[0], last_A.shape[1], weight.size(0), weight.size(1), weight.size(2), weight.size(3))
                    # Here we should transpose sum_bias to set the batch dim to 1, which is aimed to keep consistent with the matrix version
                    sum_bias = x[2].lower.unsqueeze(0).unsqueeze(2).unsqueeze(3).transpose(0, 1)
                else:
                    raise NotImplementedError()
                padding = last_A.padding if last_A is not None else (0, 0, 0, 0)  # (left, right, top, bottom)
                stride = last_A.stride if last_A is not None else 1

                if type(padding) == int:
                    padding = padding * self.stride[0] + self.padding[0]
                else:
                    padding = tuple(p * self.stride[0] + self.padding[0] for p in padding)
                stride *= self.stride[0]

                if pieces.shape[-1] > self.input_shape[-1]: # the patches is too large and from now on, we will use matrix mode instead of patches mode.
                    # This is our desired matrix: the input will be flattend to (batch_size, input_channel*input_x * input_y) and multiplies on this matrix.
                    # After multiplication, the desired output is (batch_size, out_channel, output_x*output_y) where total_patches=output_x*output_y
                    A_matrix = patchesToMatrix(pieces, self.input_shape[1:], stride, padding)
                    if type(sum_bias) == torch.Tensor:
                        sum_bias = sum_bias.transpose(0, 1)
                        sum_bias = sum_bias.view(sum_bias.size(0), -1).transpose(0,1)
                    return A_matrix.transpose(0,1), sum_bias
                
                return Patches(pieces, stride, padding, pieces.shape), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]
        input_dim = x.lb.shape[-2] * x.lb.shape[-1]
        wshape = x.lw.shape
        wshape_conv = (wshape[0] * wshape[1], *wshape[2:])        
        eye = torch.eye(input_dim).view(input_dim, 1, *x.lb.shape[-2:])
        weight = F.conv2d(eye, weight, None, self.stride, self.padding, self.dilation, self.groups)
        weight = weight.view(input_dim, -1)
        output_dim = weight.shape[-1]
        bias = bias.view(1, -1, 1).repeat(1, 1, output_dim // bias.shape[0]).view(*self.output_shape[1:])
        batch_size = x.lb.shape[0]

        lw = (x.lw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(min=0)) + 
            x.uw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, dim_in, *self.output_shape[1:])
        uw = (x.uw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(min=0)) + 
            x.lw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, dim_in, *self.output_shape[1:])
        
        lb = (x.lb.reshape(batch_size, -1).matmul(weight.clamp(min=0)) + 
            x.ub.reshape(batch_size, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, *self.output_shape[1:]) + bias
        ub = (x.ub.reshape(batch_size, -1).matmul(weight.clamp(min=0)) + 
            x.lb.reshape(batch_size, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, *self.output_shape[1:]) + bias

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]

        if Interval.use_relative_bounds(*v):
            bias = v[2].nominal if self.has_bias else None
            if norm == np.inf:
                weight = v[1].nominal
                nominal = F.conv2d(
                    v[0].nominal, weight, bias, 
                    self.stride, self.padding, self.dilation, self.groups)
                lower_offset = (F.conv2d(
                                    v[0].lower_offset, weight.clamp(min=0), None,
                                    self.stride, self.padding, self.dilation, self.groups) + 
                                F.conv2d(
                                    v[0].upper_offset, weight.clamp(max=0), None,
                                    self.stride, self.padding, self.dilation, self.groups))
                upper_offset = (F.conv2d(
                                    v[0].upper_offset, weight.clamp(min=0), None,
                                    self.stride, self.padding, self.dilation, self.groups) + 
                                F.conv2d(
                                    v[0].lower_offset, weight.clamp(max=0), None,
                                    self.stride, self.padding, self.dilation, self.groups))
                return Interval(
                    None, None, nominal=nominal, 
                    lower_offset=lower_offset, upper_offset=upper_offset
                )
            else:
                raise NotImplementedError

        h_L, h_U = v[0]
        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        elif norm > 0:
            norm, eps = Interval.get_perturbation(v[0])
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else: # Here we calculate the L0 norm IBP bound using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = h_U
            k = int(eps)
            weight_sum = torch.sum(weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * ratio

            if self.has_bias:
                center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
            else:
                center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])
        
        center = F.conv2d(mid, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        upper = center + deviation
        lower = center - deviation
        return lower, upper

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]

        mid_w = (x.lw + x.uw) / 2
        mid_b = (x.lb + x.ub) / 2
        diff_w = (x.uw - x.lw) / 2
        diff_b = (x.ub - x.lb) / 2
        weight_abs = weight.abs()
        shape = mid_w.shape
        shape_wconv = [shape[0] * shape[1]] + list(shape[2:])
        deviation_w = F.conv2d(
            diff_w.reshape(shape_wconv), weight_abs, None, 
            self.stride, self.padding, self.dilation, self.groups)
        deviation_b = F.conv2d(
            diff_b, weight_abs, None, 
            self.stride, self.padding, self.dilation, self.groups)
        center_w = F.conv2d(
            mid_w.reshape(shape_wconv), weight, None, 
            self.stride, self.padding, self.dilation, self.groups)
        center_b =  F.conv2d(
            mid_b, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups)
        deviation_w = deviation_w.reshape(shape[0], -1, *deviation_w.shape[1:])
        center_w = center_w.reshape(shape[0], -1, *center_w.shape[1:])

        return LinearBound(
            lw = center_w - deviation_w,
            lb = center_b - deviation_b,
            uw = center_w + deviation_w,
            ub = center_b + deviation_b)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return x[0]
class BoundAveragePool(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # assumptions: ceil_mode=False, count_include_pad=True
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])
      
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.count_include_pad = True
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size) / (
                np.prod(self.kernel_size))
            next_A = F.pad(next_A, (0, self.input_shape[-2] - next_A.shape[-2], 0, self.input_shape[-1] - next_A.shape[-1]))
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            return next_A, 0

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundGlobalAveragePool(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x):
        output = AdaptiveAvgPool2d((1, 1)).forward(x)  # adaptiveAveragePool with output size (1, 1)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        H, W = self.input_shape[-2], self.input_shape[-1]

        lA = last_lA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)
        uA = last_uA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)

        return [(lA, uA)], 0, 0

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        h_L = F.adaptive_avg_pool2d(h_L, (1, 1))
        h_U = F.adaptive_avg_pool2d(h_U, (1, 1))
        return h_L, h_U

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundConcat(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.IBP_rets = None

    @Bound.save_io_shape
    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, torch.Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        if self.axis < 0:
            self.axis = x[0].ndim + self.axis
        return torch.cat(x, dim=int(self.axis))

    def interval_propagate(self, *v):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == np.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))

        if Interval.use_relative_bounds(*v):
            assert all_inf # Only LINF supported for now
            return Interval(
                None, None,
                self.forward(*[_v.nominal for _v in v]),
                self.forward(*[_v.lower_offset for _v in v]),
                self.forward(*[_v.upper_offset for _v in v]),
            )

        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb=ptb)
        else:
            raise RuntimeError("BoundConcat does not support inputs with norm {}".format(norms))

    def bound_backward(self, last_lA, last_uA, *x):
        if self.axis < 0:
            self.axis = len(self.output_shape) + self.axis
        assert (self.axis > 0)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.split(last_A, self.input_size, dim=self.axis + 1)

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axis < 0:
            self.axis = x[0].lb.ndim + self.axis
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert np.min(x) == np.max(x)
        assert x[0] != self.axis
        return x[0]


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


class BoundPad(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        if len(attr) == 1:
            self.padding = [0, 0, 0, 0]
            self.value = 0.0
        else:
            self.padding = attr['pads'][2:4] + attr['pads'][6:8]
            self.value = attr['value']
        assert self.padding == [0, 0, 0, 0]
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, pad, value=0.0):
        # TODO: padding for 3-D or more dimensional inputs.
        assert x.ndim == 4
        # x[1] should be [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
        pad = [int(pad[3]), int(pad[7]), int(pad[2]), int(pad[6])]
        final = F.pad(x, pad, value=value)
        self.padding, self.value = pad, value
        return final

    def interval_propagate(self, *v):
        l, u = zip(*v)
        return Interval.make_interval(self.forward(*l), self.forward(*u), v[0])

    def bound_backward(self, last_lA, last_uA, *x):
        # TODO: padding for 3-D or more dimensional inputs.
        pad = self.padding
        left, right, top, bottom = self.padding
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            assert type(last_A) is Patches or last_A.ndim == 5
            if type(last_A) is Patches:
                if isinstance(last_A.padding, tuple):
                    new_padding = (last_A.padding[0] + left, last_A.padding[1] + right, last_A.padding[2] + top, last_A.padding[3] + bottom)
                else:
                    new_padding = (last_A.padding + left, last_A.padding + right, last_A.padding + top, last_A.padding + bottom)
                return Patches(last_A.patches, last_A.stride, new_padding, last_A.shape, last_A.identity)
            else:
                shape = last_A.size()
                return last_A[:, :, :, top:(shape[3] - bottom), left:(shape[4] - right)]
        last_lA = _bound_oneside(last_lA)
        last_uA = _bound_oneside(last_uA)
        return [(last_lA, last_uA), (None, None), (None, None)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


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

            if self.batch_dim == 0:
                if sign == -1:
                    _A = last_A.clamp(min=0) * self.lw.unsqueeze(0) + last_A.clamp(max=0) * self.uw.unsqueeze(0)
                    _bias = last_A.clamp(min=0) * self.lb.unsqueeze(0) + last_A.clamp(max=0) * self.ub.unsqueeze(0)
                elif sign == 1:
                    _A = last_A.clamp(min=0) * self.uw.unsqueeze(0) + last_A.clamp(max=0) * self.lw.unsqueeze(0)
                    _bias = last_A.clamp(min=0) * self.ub.unsqueeze(0) + last_A.clamp(max=0) * self.lb.unsqueeze(0)
                while _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=-1)
            elif self.batch_dim == -1:
                mask = torch.gt(last_A, 0.).to(torch.float)
                if sign == -1:
                    _A = last_A * (mask * self.lw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.uw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.lb.unsqueeze(0).unsqueeze(1) +
                                      (1 - mask) * self.ub.unsqueeze(0).unsqueeze(1))
                elif sign == 1:
                    _A = last_A * (mask * self.uw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.lw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.ub.unsqueeze(0).unsqueeze(1) +
                                      (1 - mask) * self.lb.unsqueeze(0).unsqueeze(1))
                while _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=-1)
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

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


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
                _A = last_A.clamp(min=0) * self.lw[0] + last_A.clamp(max=0) * self.uw[0]
                _bias = last_A.clamp(min=0) * self.lb[0] + last_A.clamp(max=0) * self.ub[0]
            elif sign == 1:
                _A = last_A.clamp(min=0) * self.uw[1] + last_A.clamp(max=0) * self.lw[1]
                _bias = last_A.clamp(min=0) * self.ub[1] + last_A.clamp(max=0) * self.lb[1]
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

        use_lower_b = False
        flag_expand = False
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
            if unstable_idx is not None:
                if unstable_idx.ndim == 1:
                    # Only unstable neurons of the start_node neurons are used.
                    selected_alpha = self.non_deter_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                elif unstable_idx.ndim == 2:
                    # Each element in the batch selects different neurons.
                    selected_alpha = batched_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                else:
                    raise ValueError
            else:
                selected_alpha = self.alpha[start_node.name]
            # print(f'{self.name} alpha {selected_alpha.size()} lA {"none" if last_lA is None else last_lA.size()} uA {"none" if last_uA is None else last_uA.size()} unstable {"none" if unstable_idx is None else unstable_idx.size()}')
            # The first dimension is lower/upper intermediate bound.
            lb_lower_d = selected_alpha[0].clamp(min=0.0, max=1.0)
            ub_lower_d = selected_alpha[1].clamp(min=0.0, max=1.0)
            if x is not None:
                lower = x.lower
                upper = x.upper
            else:
                lower = self.lower
                upper = self.upper

            lb_lower_d[:, lower > 0] = 1.0
            lb_lower_d[:, upper < 0] = 0.0
            ub_lower_d[:, lower > 0] = 1.0
            ub_lower_d[:, upper < 0] = 0.0
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
        if not flag_expand:
            if self.opt_stage == 'opt':
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0)
                ub_lower_d = ub_lower_d.unsqueeze(0)
            else:
                lower_d = lower_d.unsqueeze(0)

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
                    bias = bias + self.get_bias(pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + self.get_bias(neg_A, b_neg)
                return A, bias
            elif type(last_A) == Patches:
                # if last_A is not an identity matrix
                assert last_A.identity == 0
                if last_A.identity == 0:
                    # Expected shape for d is [spec, batch, C, H, W]. Since the unfold function only supports 4-D vector, we need to merge spec and batch dimension.
                    assert d_pos.ndim == d_neg.ndim == 5
                    d_pos_shape = d_pos.size()
                    d_neg_shape = d_neg.size()
                    d_pos = d_pos.view(-1, *d_pos_shape[2:])
                    d_neg = d_neg.view(-1, *d_neg_shape[2:])

                    # unfold the slope matrix as patches
                    d_pos, padding_p = check_padding(d_pos, last_A.padding)
                    d_neg, padding_n = check_padding(d_neg, last_A.padding)
                    d_pos_unfolded = F.unfold(d_pos, kernel_size=last_A.patches.size(-1), stride=last_A.stride, padding=padding_p)
                    d_neg_unfolded = F.unfold(d_neg, kernel_size=last_A.patches.size(-1), stride=last_A.stride, padding=padding_n)


                    # Reshape the unfolded patches as [batch_size, L, spec_size, in_c, H, W]; here spec_size is out_c.
                    # The output of unfold has the L dimension at last. Transpose it first.
                    d_pos_unfolded_r = d_pos_unfolded.transpose(-2, -1)
                    # Reshape to (spec, batch, L, in_c, H, W]
                    d_pos_unfolded_r = d_pos_unfolded_r.view(d_pos_shape[0], d_pos_shape[1], d_pos_unfolded_r.size(1), last_A.patches.size(-3), last_A.patches.size(-2), last_A.patches.size(-1))
                    d_pos_unfolded_r = d_pos_unfolded_r.permute(1, 2, 0, 3, 4, 5)
                    d_neg_unfolded_r = d_neg_unfolded.transpose(-2, -1)
                    d_neg_unfolded_r = d_neg_unfolded_r.view(d_neg_shape[0], d_neg_shape[1], d_neg_unfolded_r.size(1), last_A.patches.size(-3), last_A.patches.size(-2), last_A.patches.size(-1))
                    d_neg_unfolded_r = d_neg_unfolded_r.permute(1, 2, 0, 3, 4, 5)

                    # print(f'{self.name} {d_neg_shape} {d_neg.size()} {d_neg_unfolded.size()} {d_neg_unfolded_r.size()} {last_A.patches.size()}')

                    # multiply according to sign of A (we use fused operation to save memory)
                    # prod = d_pos_unfolded_r * pos_A_patches + d_neg_unfolded_r * neg_A_patches
                    # neg_A_patches = last_A.patches.clamp(max=0)
                    # pos_A_patches = last_A.patches.clamp(min=0)

                    # Expected last_A shape: [batch_size, L, out_c, in_c, H, W]. Here out_c is the spec dimension.
                    prod, pos_A_patches, neg_A_patches = self.clamp_mutiply(last_A.patches, d_pos_unfolded_r, d_neg_unfolded_r)

                    # Save the patch size, which will be used in init_slope().
                    if start_node is not None:
                        self.patch_size[start_node.name] = prod.size()

                    bias = 0
                    if b_pos is not None:
                        bias = bias + self.get_bias(Patches(pos_A_patches, last_A.stride, last_A.padding, last_A.shape), b_pos)
                    if b_neg is not None:
                        bias = bias + self.get_bias(Patches(neg_A_patches, last_A.stride, last_A.padding, last_A.shape), b_neg)

                    return Patches(prod, last_A.stride, last_A.padding, prod.shape, 0), bias.transpose(0, 1)

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
                    # unfold the beta as patches
                    masked_beta_unfolded = F.unfold(self.masked_beta, kernel_size=A_patches.size(-1), padding=A.padding, stride=A.stride)
                    # reshape the unfolded patches as [batch_size, L, 1, in_c, H, W]
                    masked_beta_unfolded = masked_beta_unfolded.transpose(-2, -1)
                    masked_beta_unfolded = masked_beta_unfolded.view(
                            masked_beta_unfolded.size(0), masked_beta_unfolded.size(1), A_patches.size(-3), A_patches.size(-2), A_patches.size(-1)).unsqueeze(2)
                    if uA is not None:
                        uA = Patches(uA.patches + masked_beta_unfolded, uA.stride, uA.padding, uA.patches.shape, 0)
                    if lA is not None:
                        lA = Patches(lA.patches - masked_beta_unfolded, lA.stride, lA.padding, lA.patches.shape, 0)
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
                        # unfold the beta as patches
                        masked_beta_unfolded = F.unfold(self.masked_beta, kernel_size=A_patches.size(-1),
                                                        padding=A.padding, stride=A.stride)
                        # reshape the unfolded patches as [batch_size, L, 1, in_c, H, W]
                        masked_beta_unfolded = masked_beta_unfolded.transpose(-2, -1)
                        masked_beta_unfolded = masked_beta_unfolded.view(
                            masked_beta_unfolded.size(0), masked_beta_unfolded.size(1), A_patches.size(-3),
                            A_patches.size(-2), A_patches.size(-1)).unsqueeze(2)
                        if uA is not None:
                            uA = Patches(uA.patches + masked_beta_unfolded, uA.stride, uA.padding, uA.patches.shape, 0)
                        if lA is not None:
                            lA = Patches(lA.patches - masked_beta_unfolded, lA.stride, lA.padding, lA.patches.shape, 0)
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


class BoundUnsqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        if self.axes < 0:
            self.axes = len(self.input_shape) + self.axes + 1
        return x.unsqueeze(self.axes) 

    def bound_backward(self, last_lA, last_uA, x):
        if self.axes == 0:
            return last_lA, 0, last_uA, 0
        else:
            return [(last_lA.squeeze(self.axes + 1) if last_lA is not None else None,
                     last_uA.squeeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        if len(self.input_shape) == 0:
            lw, lb = x.lw.unsqueeze(1), x.lb.unsqueeze(0)
            uw, ub = x.uw.unsqueeze(1), x.ub.unsqueeze(0)
        else:
            lw, lb = x.lw.unsqueeze(self.axes + 1), x.lb.unsqueeze(self.axes)
            uw, ub = x.uw.unsqueeze(self.axes + 1), x.ub.unsqueeze(self.axes)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.axes > x[0]:
            return x[0]
        raise NotImplementedError
    


class BoundSqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return x.squeeze(self.axes) 

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axes != 0)
        return [(last_lA.unsqueeze(self.axes + 1) if last_lA is not None else None,
                 last_uA.unsqueeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif x[0] < self.axes:
            return x[0]
        elif x[0] > self.axes:
            return x[0] - 1
        else:
            assert 0


class BoundConstantOfShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.device = device
        self.value = attr['value'].to(self.device)

    @Bound.save_io_shape
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
        value = torch.ones(list(v[0][0]), device=self.device) * self.value
        return value, value

    def infer_batch_dim(self, batch_size, *x):
        # FIXME Should avoid referring to batch_size; Treat `torch.Size` results differently
        if self.x[0] == batch_size:
            return 0
        else:
            return -1


class BoundConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.value = attr['value'].to(self.device)
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self):
        return self.value.to(self.device)

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def bound_backward(self, last_lA, last_uA):
        def _bound_oneside(A):
            if A is None:
                return 0.0

            if type(A) == torch.Tensor:
                while A.ndim > 2:
                    A = torch.sum(A, dim=-1)
            elif type(A) == Patches:
                patches = A.patches
                patches_reshape = torch.sum(patches, dim=(-1, -2, -3)) * self.value.to(self.device)
                patches_reshape = patches_reshape.transpose(-1, -2)
                return patches_reshape.view(patches_reshape.size(0), patches_reshape.size(1), int(math.sqrt(patches_reshape.size(2))), -1).transpose(0, 1)

            return A * self.value.to(self.device)

        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(dim_in, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub)


class BoundShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True        

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, torch.Tensor) else torch.tensor(x).shape

    def forward(self, x):
        self.from_input = False
        return BoundShape.shape(x)

    def bound_forward(self, dim_in, x):
        return self.forward_value

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            shape = self.forward(v[0].nominal)
            if not isinstance(shape, torch.Tensor):
                shape = torch.tensor(shape, device=self.device)
            return Interval(
                None, None,
                shape, torch.zeros_like(shape), torch.zeros_like(shape)
            )

        return super().interval_propagate(*v)


class BoundGather(Bound):
    def __init__(self, input_name, name, ori_name, attr, x, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, x, output_index, options, device)
        self.axis = attr['axis'] if 'axis' in attr else 0
        self.nonlinear = False  # input shape required

    @Bound.save_io_shape
    def forward(self, x, indices):
        self.indices = indices
        x = x.to(self.indices.device)  # BoundShape.shape() will return value on cpu only
        if indices.ndim == 0:
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
        elif self.axis == 0:
            return torch.index_select(x, dim=self.axis, index=indices.reshape(-1)) \
                .reshape(*indices.shape, x.shape[-1])
        raise ValueError(
            'Unsupported shapes in Gather: data {}, indices {}, axis {}'.format(x.shape, indices.shape, self.axis))

    def bound_backward(self, last_lA, last_uA, x, indices):
        assert self.from_input
        assert self.indices.ndim == 0  # TODO

        def _bound_oneside(A):
            if A is None:
                return None
            assert (self.indices.ndim == 0)

            A = A.unsqueeze(self.axis + 1)
            idx = int(self.indices)
            tensors = []
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[self.axis + 1] *= idx
                tensors.append(torch.zeros(shape_pre, device=self.device))
            tensors.append(A)
            if self.input_shape[self.axis] - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[self.axis + 1] *= self.input_shape[self.axis] - idx - 1
                tensors.append(torch.zeros(shape_next, device=self.device))
            return torch.cat(tensors, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        assert self.indices.ndim == 0  # TODO

        if isinstance(x, torch.Size):
            lw = uw = torch.zeros(dim_in, device=self.device)
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=self.indices).squeeze(self.axis)
        else:
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            uw = torch.index_select(x.uw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            lb = torch.index_select(x.lb, dim=self.axis, index=self.indices).squeeze(self.axis)
            ub = torch.index_select(x.ub, dim=self.axis, index=self.indices).squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)

        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                self.forward(v[0].nominal, v[1].nominal),
                self.forward(v[0].lower_offset, v[1].nominal),
                self.forward(v[0].upper_offset, v[1].nominal)
            )
        
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]


class BoundGatherElements(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.axis = attr['axis']

    @Bound.save_io_shape
    def forward(self, x, index):
        self.index = index
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index):
        assert self.from_input
        
        dim = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.output_shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1,
                index=self.index.unsqueeze(0).repeat(A.shape[0], *([1] * (A.ndim - 1))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)

        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                self.forward(v[0].nominal, v[1].nominal),
                self.forward(v[0].lower_offset, v[1].nominal),
                self.forward(v[0].upper_offset, v[1].nominal)
            )

        return self.forward(v[0][0], v[1][0]), \
               self.forward(v[0][1], v[1][1])

    def bound_forward(self, dim_in, x, index):
        assert self.axis != 0
        dim = self._get_dim()
        return LinearBound(
            torch.gather(x.lw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.lb, dim=dim, index=self.index),
            torch.gather(x.uw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.ub, dim=dim, index=self.index))

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]
    
    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.output_shape) + dim
        return dim


class BoundPrimConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.value = attr['value']

    @Bound.save_io_shape
    def forward(self):
        return torch.tensor([], device=self.device)


class BoundRNN(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.complex = True
        self.output_index = output_index

    @Bound.save_io_shape
    def forward(self, x, weight_input, weight_recurrent, bias, sequence_length, initial_h):
        assert (torch.sum(torch.abs(initial_h)) == 0)

        self.input_size = x.shape[-1]
        self.hidden_size = weight_input.shape[-2]

        class BoundRNNImpl(nn.Module):
            def __init__(self, input_size, hidden_size,
                         weight_input, weight_recurrent, bias, output_index, options, device):
                super().__init__()

                self.input_size = input_size
                self.hidden_size = hidden_size

                self.cell = torch.nn.RNNCell(
                    input_size=input_size,
                    hidden_size=hidden_size
                )

                self.cell.weight_ih.data.copy_(weight_input.squeeze(0).data)
                self.cell.weight_hh.data.copy_(weight_recurrent.squeeze(0).data)
                self.cell.bias_ih.data.copy_((bias.squeeze(0))[:hidden_size].data)
                self.cell.bias_hh.data.copy_((bias.squeeze(0))[hidden_size:].data)

                self.output_index = output_index

            def forward(self, x):
                length = x.shape[0]
                outputs = []
                hidden = torch.zeros(x.shape[1], self.hidden_size, device=self.device)
                for i in range(length):
                    hidden = self.cell(x[i, :], hidden)
                    outputs.append(hidden.unsqueeze(0))
                outputs = torch.cat(outputs, dim=0)

                if self.output_index == 0:
                    return outputs
                else:
                    return hidden

        self.model = BoundRNNImpl(
            self.input_size, self.hidden_size,
            weight_input, weight_recurrent, bias,
            self.output_index, self.device)
        self.input = (x,)

        return self.model(self.input)


class BoundTranspose(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.perm = attr['perm']
        self.perm_inv_inc_one = [-1] * (len(self.perm) + 1)
        self.perm_inv_inc_one[0] = 0
        for i in range(len(self.perm)):
            self.perm_inv_inc_one[self.perm[i] + 1] = i + 1
        self.use_default_ibp = True            

    @Bound.save_io_shape
    def forward(self, x):
        return x.permute(*self.perm)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return last_A.permute(self.perm_inv_inc_one)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        if self.input_shape[0] != 1:
            perm = [0] + [(p + 1) for p in self.perm]
        else:
            assert (self.perm[0] == 0)
            perm = [0, 1] + [(p + 1) for p in self.perm[1:]]
        lw, lb = x.lw.permute(*perm), x.lb.permute(self.perm)
        uw, ub = x.uw.permute(*perm), x.ub.permute(self.perm)

        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            return self.perm.index(x[0])


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


class BoundNeg(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x):
        return -x

    def bound_backward(self, last_lA, last_uA, x):
        return [(-last_lA if last_lA is not None else None,
                 -last_uA if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(-x.uw, -x.ub, -x.lw, -x.lb)

    def interval_propagate(self, *v):
        return -v[0][1], -v[0][0]

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundMatMul(BoundLinear):
    # Reuse most functions from BoundLinear.
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.transA = 0
        self.transB = 1  # MatMul assumes B is transposed.
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def interval_propagate(self, *v):
        w_l = v[1][0].transpose(-1, -2)
        w_u = v[1][1].transpose(-1, -2)
        lower, upper = super().interval_propagate(v[0], (w_l, w_u))
        return lower, upper   

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2
        # BoundLinear has W transposed.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        results = super().bound_backward(last_lA, last_uA, *x)
        # Transpose input back.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        # Transpose result on A.
        return [results[0][0], (lA_y, uA_y), results[0][2]], results[1], results[2]

    def bound_forward(self, dim_in, x, y):
        return super().bound_forward(dim_in, x, LinearBound(
            y.lw.transpose(-1, -2) if y.lw is not None else None,
            y.lb.transpose(-1, -2) if y.lb is not None else None,
            y.uw.transpose(-1, -2) if y.uw is not None else None,
            y.ub.transpose(-1, -2) if y.ub is not None else None,
            y.lower.transpose(-1, -2) if y.lower is not None else None,
            y.upper.transpose(-1, -2) if y.upper is not None else None
        ))

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


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

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


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


class BoundReduceMax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        # for torch.max, `dim` must be an int
        if isinstance(self.axis, list):
            assert len(self.axis) == 1
            self.axis = self.axis[0]
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        self.use_default_ibp = True      

        """Assume that the indexes with the maximum values are not perturbed. 
        This generally doesn't hold true, but can still be used for the input shift 
        in Softmax of Transformers."""   
        self.fixed_max_index = options.get('fixed_reducemax_index', False)

    @Bound.save_io_shape
    def forward(self, x):
        if self.axis < 0:
            self.axis += len(self.input_shape)
        assert self.axis > 0
        res = torch.max(x, dim=self.axis, keepdim=self.keepdim)
        self.indices = res.indices
        return res.values

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]

    def bound_backward(self, last_lA, last_uA, x):
        if self.fixed_max_index:	
            def _bound_oneside(last_A):	
                if last_A is None:	
                    return None	
                indices = self.indices.unsqueeze(0)	
                if not self.keepdim:	
                    assert (self.from_input)	
                    last_A = last_A.unsqueeze(self.axis + 1)	
                    indices = indices.unsqueeze(self.axis + 1)	
                shape = list(last_A.shape)	
                shape[self.axis + 1] *= self.input_shape[self.axis]	
                A = torch.zeros(shape, device=last_A.device)	
                A.scatter_(dim=self.axis + 1, index=indices, src=last_A)	
                return A	

            return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0	
        else:
            raise NotImplementedError('`bound_backward` for BoundReduceMax with perturbed maximum indexes is not implemented.')


class BoundReduceMean(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        self.use_default_ibp = True        

    @Bound.save_io_shape
    def forward(self, x):
        return torch.mean(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * last_A.ndim
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat) / size
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.keepdim)
        assert (len(self.axis) == 1)
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        size = self.input_shape[axis]
        lw = x.lw.sum(dim=axis + 1, keepdim=True) / size
        lb = x.lb.sum(dim=axis, keepdim=True) / size
        uw = x.uw.sum(dim=axis + 1, keepdim=True) / size
        ub = x.ub.sum(dim=axis, keepdim=True) / size
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] in self.axis:
            assert not self.perturbed
            return -1
        return x[0]


class BoundReduceSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes'] if 'axes' in attr else None
        self.keepdim = bool(attr['keepdims'])
        self.use_default_ibp = True        

    @Bound.save_io_shape
    def forward(self, x):
        if self.axis is not None:
            return torch.sum(x, dim=self.axis, keepdim=self.keepdim)
        else:
            return torch.sum(x)
            
    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * last_A.ndim
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat)
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert self.keepdim
        assert len(self.axis) == 1
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        lw, lb = x.lw.sum(dim=axis + 1, keepdim=True), x.lb.sum(dim=axis, keepdim=True)
        uw, ub = x.uw.sum(dim=axis + 1, keepdim=True), x.ub.sum(dim=axis, keepdim=True)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert not x[0] in self.axis
        return x[0]

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

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundSplit(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.split = attr['split']
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.split(x, self.split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axis > 0)
        pre = sum(self.split[:self.output_index])
        suc = sum(self.split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.axis > 0 and self.from_input)
        lw = torch.split(x.lw, self.split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, self.split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, self.split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, self.split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]


class BoundEqual(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, y):
        return x == y

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


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

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundSqrt(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.sqrt(x)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            nominal = self.forward(v[0].nominal)
            lower_offset = self.forward(v[0].nominal + v[0].lower_offset) - nominal
            upper_offset = self.forward(v[0].nominal + v[0].upper_offset) - nominal
            return Interval(None, None, nominal, lower_offset, upper_offset)            

        return super().interval_propagate(*v)


class BoundExpand(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, y):
        y = y.clone()
        assert y.ndim == 1
        n, m = x.ndim, y.shape[0]
        assert n <= m
        for i in range(n):
            if y[m - n + i] == 1:
                y[m - n + i] = x.shape[i]
            else:
                assert x.shape[i] == 1 or x.shape[i] == y[m - n + i]
        return x.expand(*list(y))

    def infer_batch_dim(self, batch_size, *x):
        # FIXME should avoid referring to batch_size
        if self.forward_value.shape[0] != batch_size:
            return -1
        else:
            raise NotImplementedError('forward_value shape {}'.format(self.forward_value.shape))


class BoundWhere(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(0)

        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                self.forward(v[0].nominal, v[1].nominal, v[2].nominal),
                self.forward(v[0].nominal, v[1].lower_offset, v[2].lower_offset),
                self.forward(v[0].nominal, v[1].upper_offset, v[2].upper_offset)
            )

        condition = v[0][0]
        return tuple([torch.where(condition, v[1][j], v[2][j]) for j in range(2)])

    def bound_backward(self, last_lA, last_uA, condition, x, y):
        assert torch.allclose(condition.lower.float(), condition.upper.float())
        assert self.from_input
        mask = condition.lower.float()

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            assert last_A.ndim > 1
            A_x = self.broadcast_backward(mask.unsqueeze(0) * last_A, x)
            A_y = self.broadcast_backward((1 - mask).unsqueeze(0) * last_A, y)
            return A_x, A_y

        lA_x, lA_y = _bound_oneside(last_lA)
        uA_x, uA_y = _bound_oneside(last_uA)

        return [(None, None), (lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x[1:])


class BoundNot(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x):
        return x.logical_not()

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundCumSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x, axis):
        self.axis = axis
        return torch.cumsum(x, axis)

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]


class BoundSlice(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.start = attr["starts"][0] if "starts" in attr else None
        self.end = attr["ends"][0] if "ends" in attr else None
        self.axes = attr["axes"][0] if "axes" in attr else None
        self.use_default_ibp = True

    # Older Pytorch version only passes steps as input.
    @Bound.save_io_shape
    def forward(self, x, start=None, end=None, axes=None, steps=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        axes = self.axes if axes is None else axes
        assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape if isinstance(x, torch.Tensor) else [len(x)]
        if start < 0:
            start += shape[axes]
        if end < 0:
            if end == -9223372036854775807:  # -inf in ONNX
                end = 0  # only possible when step == -1
            else:
                end += shape[axes]
        if steps == -1:        
            start, end = end, start + 1  # TODO: more test more negative step size.
        end = min(end, shape[axes])
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))
        return final

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            assert self.axes != x[0]
            return x[0]


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

    def infer_batch_dim(self, batch_size, *x):
        return x[0]

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
        

class BoundRange(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, start, end, step):
        if start.dtype == end.dtype == step.dtype == torch.int64:
            return torch.arange(start, end, step, dtype=torch.int64, device=self.device)
        else:
            return torch.arange(start, end, step, device=self.device)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == x[1] == x[2] == -1
        return -1


"""All supported ONNX operations go below."""
class BoundATenDiag(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, diagonal=0):
        return torch.diag(x, diagonal=diagonal)

    def interval_propagate(self, *v):
        return Interval.make_interval(torch.diag(v[0][0], v[1][0]), torch.diag(v[0][1], v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        return 1  # This is not a batch operation.


class BoundATenDiagonal(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, offset=0, dim1=0, dim2=1):
        return torch.diagonal(x, offset=offset, dim1=dim1, dim2=dim2)

    def interval_propagate(self, *v):
        params = (v[1][0], v[2][0], v[3][0])
        return Interval.make_interval(torch.diagonal(v[0][0], *params), torch.diagonal(v[0][1], *params), v[0])

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundFlatten(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True
        self.axis = attr['axis']

    @Bound.save_io_shape
    def forward(self, x):
        return torch.flatten(x, self.axis)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0


class BoundMaxPool(BoundOptimizableActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.nonlinear = True
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        # assert self.padding[0] == 0 and self.padding[1] == 0 # currently only support padding=0
        self.ceil_mode = False
        self.use_default_ibp = True
        self.alpha = None
        self.init = {}

    @Bound.save_io_shape
    def forward(self, x):
        output, _ = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        return output

    def project_simplex(self, patches):
        sorted = torch.flatten(patches, -2)
        sorted, _ = torch.sort(sorted, -1, descending=True)
        rho_sum = torch.cumsum(sorted, -1)
        rho_value = 1 - rho_sum
        rho_value = (sorted + rho_value/torch.tensor(range(1, sorted.size(-1)+1), dtype=torch.float, device=sorted.device)) > 0
        _, rho_index = torch.max(torch.cumsum(rho_value, -1), -1)
        rho_sum = torch.gather(rho_sum, -1, rho_index.unsqueeze(-1)).squeeze(-1)
        lbd = 1/(rho_index+1)* (1-rho_sum)

        return torch.clamp(patches + lbd.unsqueeze(-1).unsqueeze(-1), min=0)

    def init_opt_parameters(self, start_nodes):
        batch_size, channel, h, w = self.input_shape
        o_h, o_w = self.output_shape[-2:]
        # batch_size, out_c, out_h, out_w, k, k

        self.alpha = OrderedDict()
        ref = self.inputs[0].lower # a reference variable for getting the shape
        for ns, size_s in start_nodes:
            self.alpha[ns] = torch.empty([1, size_s, self.input_shape[0], self.input_shape[1], self.output_shape[-2], self.output_shape[-1], self.kernel_size[0], self.kernel_size[1]], 
                dtype=torch.float, device=ref.device, requires_grad=True)
            self.init[ns] = False



    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        paddings = tuple(self.padding + self.padding)

        A_shape = last_lA.shape if last_lA is not None else last_uA.shape
        # batch_size, input_c, x, y
        upper_d = torch.zeros((list(self.input_shape)), device=x.device)
        lower_d = torch.zeros((list(self.input_shape)), device=x.device)

        upper_d = F.pad(upper_d, paddings)
        lower_d = F.pad(lower_d, paddings)

        # batch_size, output_c, x, y
        upper_b = torch.zeros((list(self.output_shape)), device=x.device)
        lower_b = torch.zeros((list(self.output_shape)), device=x.device)

        # 1. find the index i where li > uj for all j, then set upper_d = lower_d = 1
        max_lower, max_lower_index = F.max_pool2d(x.lower, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        delete_upper = torch.scatter(torch.flatten(F.pad(x.upper, paddings), -2), -1, torch.flatten(max_lower_index, -2), -np.inf).view(upper_d.shape)
        max_upper, _ = F.max_pool2d(delete_upper, self.kernel_size, self.stride, 0, return_indices=True, ceil_mode=self.ceil_mode)
        
        values = torch.zeros_like(max_lower)
        values[max_lower >= max_upper] = 1.0
        upper_d = torch.scatter(torch.flatten(upper_d, -2), -1, torch.flatten(max_lower_index, -2), torch.flatten(values, -2)).view(upper_d.shape)

        if self.opt_stage == 'opt':
            alpha = self.alpha[start_node.name]
            if self.init[start_node.name] == False:
                lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
                lower_d_unfold = F.unfold(lower_d, self.kernel_size, 1, stride=self.stride)

                alpha_data = lower_d_unfold.view(lower_d.shape[0], lower_d.shape[1], self.kernel_size[0], self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])
                alpha.data.copy_(alpha_data.permute((0,1,4,5,2,3)).clone().detach())
                self.init[start_node.name] = True
                if self.padding[0] > 0:
                    lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

            alpha.data = self.project_simplex(alpha.data).clone().detach()
            alpha = alpha.permute((0,1,2,3,6,7,4,5))
            alpha_shape = alpha.shape
            alpha = alpha.reshape((alpha_shape[0]*alpha_shape[1]*alpha_shape[2], -1, alpha_shape[-2]*alpha_shape[-1]))
            lower_d = F.fold(alpha, self.input_shape[-2:], self.kernel_size, 1, self.padding, self.stride)
            lower_d = lower_d.view(alpha_shape[0], alpha_shape[1], alpha_shape[2], *lower_d.shape[1:])
            lower_d = lower_d.squeeze(0)
        else:
            lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
            if self.padding[0] > 0:
                lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        values[:] = 0.0
        max_upper_, _ = F.max_pool2d(x.upper, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        values[max_upper > max_lower] = max_upper_[max_upper > max_lower]
        upper_b = values

        assert type(last_lA) == torch.Tensor or type(last_uA) == torch.Tensor
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            pos_A = last_A.clamp(min=0)
            neg_A = last_A.clamp(max=0)

            bias = 0
            if b_pos is not None:
                bias = bias + self.get_bias(pos_A, b_pos)
            if b_neg is not None:
                bias = bias + self.get_bias(neg_A, b_neg)

            shape = last_A.size()
            pos_A = F.interpolate(pos_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
            pos_A = F.pad(pos_A, (0, self.input_shape[-2] - pos_A.shape[-2], 0, self.input_shape[-1] - pos_A.shape[-1]))
            pos_A = pos_A.view(shape[0], shape[1], *pos_A.shape[1:])

            neg_A = F.interpolate(neg_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
            neg_A = F.pad(neg_A, (0, self.input_shape[-2] - neg_A.shape[-2], 0, self.input_shape[-1] - neg_A.shape[-1]))
            neg_A = neg_A.view(shape[0], shape[1], *neg_A.shape[1:])

            next_A = pos_A * d_pos + neg_A * d_neg
            return next_A, bias

        if self.padding[0] > 0:
            upper_d = upper_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        uA, ubias = _bound_oneside(last_uA, upper_d, lower_d, upper_b, lower_b)
        lA, lbias = _bound_oneside(last_lA, lower_d, upper_d, lower_b, upper_b)

        return [(lA, uA)], lbias, ubias

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0
