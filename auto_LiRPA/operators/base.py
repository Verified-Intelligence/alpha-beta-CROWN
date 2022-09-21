""" Base class and functions for implementing bound operators"""
import copy
import os
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from itertools import chain
from numpy.lib.arraysetops import isin
from collections import OrderedDict

from ..perturbations import *
from ..utils import *
from ..patches import *
from ..linear_bound import LinearBound

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

epsilon = 1e-12


def not_implemented_op(node, func):
    message = ("Function `{}` of `{}` is not supported yet."
            " Please help to open an issue at https://github.com/KaidiXu/auto_LiRPA"
            " or implement this function in auto_LiRPA/bound_ops.py"
            " or auto_LiRPA/operators by yourself.".format(func, node))
    raise NotImplementedError(message)


class Interval(tuple):
    """Interval object for interval bound propagation."""

    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, ptb=None):
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, ptb=None):
        if ptb is None:
            self.ptb = None
            # `self.ptb == None` means that this interval
            # is not perturbed and it shall be treated as a constant and lb = ub.
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

    @staticmethod
    def make_interval(lb, ub, other=None):
        """Checking if the other interval is tuple, keep the perturbation."""
        if isinstance(other, Interval):
            return Interval(lb, ub, ptb=other.ptb)
        else:
            return (lb, ub)

    @staticmethod
    def get_perturbation(interval):
        """Given a tuple or Interval object, returns the norm and eps."""
        if isinstance(interval, Interval) and interval.ptb is not None:
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return np.inf, 1.0
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return np.inf, np.nan


    @staticmethod
    def is_perturbed(interval):
        """Checking if a Interval or tuple object has perturbation enabled."""
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True


class Bound(nn.Module):
    r"""
    Base class for supporting the bound computation of an operator. Please see examples
    at `auto_LiRPA/operators`.

    Args:
        attr (dict): Attributes of the operator.

        inputs (list): A list of input nodes.

        output_index (int): The index in the output if the operator has multiple outputs. Usually output_index=0.

        options (dict): Bound options.

    Be sure to run `super().__init__(attr, inputs, output_index, options, device)`
    first in the `__init__` function.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__()
        attr = {} if attr is None else attr
        inputs = [] if inputs is None else inputs
        options = {} if options is None else options
        self.name = None
        self.output_name = []
        self.device = attr.get('device')
        self.attr, self.inputs, self.output_index, self.options = \
            attr, inputs, output_index, options
        self.forward_value = None
        self.output_shape = None
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
        # If set to true, the backward bound output of this node is 0.
        self.zero_backward_coeffs_l = False
        self.zero_backward_coeffs_u = False
        # If set to true, the A matrix accumulated on this node is 0.
        self.zero_lA_mtx = False
        self.zero_uA_mtx = False

        self.patches_start = False

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def is_input_perturbed(self, i=0):
        r"""Check if the i-th input is with perturbation or not."""
        return i < len(self.inputs) and self.inputs[i].perturbed

    def clear(self):
        """ Clear attributes when there is a new input to the network"""
        pass

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

    def default_interval_propagate(self, *v):
        """For unary monotonous functions or functions for altering shapes only but not values"""
        if len(v) == 0:
            return Interval.make_interval(self.forward(), self.forward())
        elif len(v) == 1:
            return Interval.make_interval(
                self.forward(v[0][0]), self.forward(v[0][1]), v[0])
        else:
            raise NotImplementedError('default_interval_propagate only supports no more than 1 input node')


    def bound_forward(self, dim_in, *x):
        r"""
        Function for forward mode bound propagation. Forward mode LiRPA computs a `LinearBound`
        instance representing the linear bound for each involved node.
        Major attributes of `LinearBound` include
        `lw`, `uw`, `lb`, `ub`, `lower`, and `upper`.

        `lw` and `uw` are coefficients of linear bounds w.r.t. model input.
        Their shape is `(batch_size, dim_in, *standard_shape)`,
        where `dim_in` is the total dimension of perturbed input nodes of the model,
        and `standard_shape` is the shape of the standard/clean output.
        `lb` and `ub` are bias terms of linear bounds, and their shape is equal
        to the shape of standard/clean output.
        `lower` and `upper` are concretized lower and upper bounds that will be
        computed later in BoundedModule.

        Args:
            dim_in (int): Total dimension of perturbed input nodes of the model.

            x: A list of the linear bound of input nodes. Each element in x is a `LinearBound` instance.

        Returns:
            bound (LinearBound): The linear bound of this node.
        """
        return not_implemented_op(self, 'bound_forward')

    def bound_dynamic_forward(self, *x, max_dim=None, offset=0):
        raise NotImplementedError(f'bound_dynamic_forward is not implemented for {self}.')

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
        # Default implementation assuming the batch dimension is always at 0.
        # Do not use it if the operator can alter the shape
        assert x[0] in [0, -1]
        return x[0]

    def broadcast_backward(self, A, x):
        shape = x.output_shape
        batch_dim = max(self.batch_dim, 0)

        if isinstance(A, Tensor):
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
                # FIXME (05/11/2022): the following condition is not always correct. We should not rely on checking dimension is "1" or not.
                if shape[i] == 1 and A.shape[i + 1] != 1 and i != batch_dim:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            assert (A.shape[2:] == shape[1:])  # skip the spec and batch dimension.
        else:
            pass
        return A

    def get_bias(self, A, bias):
        if A is None:
            return 0
        if not Benchmarking:
            assert not isnan(A)
            assert not isnan(bias)
        if torch.isinf(bias).any():
            warnings.warn('There is an inf value in the bias of LiRPA bounds.')

        if isinstance(A, Tensor):
            if self.batch_dim != -1:
                bias_new = torch.einsum('sb...,b...->sb', A, bias)
            else:
                bias_new = torch.einsum('sb...,...->sb', A, bias)
            if isnan(bias_new):
                # NaN can be caused by 0 * inf, if 0 appears in `A` and inf appears in `bias`.
                # Force the whole bias to be 0, to avoid gradient issues.
                # FIXME maybe find a more robust solution.
                return 0
            else:
                # FIXME (09/17): handle the case for pieces.unstable_idx.
                return bias_new
        elif isinstance(A, eyeC):
            batch_size = A.shape[1]
            if self.batch_dim != -1:
                return bias.reshape(batch_size, -1).t()
            else:
                return bias.reshape(-1).unsqueeze(-1).repeat(1, batch_size)
        elif type(A) == Patches:
            # the shape of A.patches is [batch, L, out_c, in_c, K, K]
            if self.batch_dim != -1:
                # Input A patches has shape (spec, batch, out_h, out_w, in_c, H, W) or (unstable_size, batch, in_c, H, W).
                patches = A.patches
                # Here the size of bias is [batch_size, out_h, out_w, in_c, H, W]
                bias = inplace_unfold(bias, kernel_size=A.patches.shape[-2:], stride=A.stride, padding=A.padding, inserted_zeros=A.inserted_zeros, output_padding=A.output_padding)
                if A.unstable_idx is not None:
                    # Sparse bias has shape [unstable_size, batch_size, in_c, H, W]. No need to select over the out_c dimension.
                    bias = bias[:, A.unstable_idx[1], A.unstable_idx[2]]
                    # bias_new has shape (unstable_size, batch).
                    bias_new = torch.einsum('bschw,sbchw->sb', bias, patches)
                else:
                    # Sum over the in_c, H, W dimension. Use torch.einsum() to save memory, equal to:
                    # bias_new = (bias * patches).sum(-1,-2,-3).transpose(-2, -1)
                    # bias_new has shape (spec, batch, out_h, out_w).
                    bias_new = torch.einsum('bijchw,sbijchw->sbij', bias, patches)
            else:
                # Similar to BoundConstant. (BoundConstant does not have batch_dim).
                # FIXME (09/16): bias size is different for BoundConstant. We should use the same size!
                patches = A.patches
                bias_new = torch.sum(patches, dim=(-1, -2, -3)) * bias.to(self.device)
                # Return shape is (spec, batch, out_h, out_w) or (unstable_size, batch).
                return bias_new
            return bias_new
        else:
            return NotImplementedError()

    def make_axis_non_negative(self, axis, shape='input'):
        if shape == 'input':
            shape = self.input_shape
        elif shape == 'output':
            shape = self.output_shape
        else:
            assert isinstance(shape, torch.Size)
        if axis < 0:
            return axis + len(shape)
        else:
            return axis

    def non_deter_wrapper(self, op, *args, **kwargs):
        """Some operations are non-deterministic and deterministic mode will fail. So we temporary disable it."""
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
