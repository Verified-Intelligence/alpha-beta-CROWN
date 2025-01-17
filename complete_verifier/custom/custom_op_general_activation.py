#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
r"""
This file is an example of defining a custom operation and providing its
relaxations for bound computation. Here we consider a modified activation
function sigma which is a mixture between a linear function and non-linear
function:
             / a_i x_i + b_i   if m_i = 1,
    f(x_i) = |
             \ sigmoid(x_i)       if m_i = 0.
where a, b are element-wise slopes and biases when the function is linear,
and m is the mask controlling the behavior of this function. We consider
perturbations on x.

This example is a slight extension of `custom_op.py`.
"""
import torch
import torch.nn as nn
from auto_LiRPA import register_custom_op
from auto_LiRPA.bound_ops import BoundSigmoid, Interval


class LinearMaskedSigmoidOp(torch.autograd.Function):
    """A sigmoid function with some neurons replaced with linear operations."""
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor, slope: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using Sigmoid
        s = torch.sigmoid(input)
        ctx.save_for_backward(input, mask, slope, bias, s)
        return s * (1.0 - mask) + (input * slope + bias) * mask

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, mask, slope, bias, s = ctx.saved_tensors
        sigmoid_grad = grad_output * s * (1.0 - s)
        grad_input = sigmoid_grad * (1.0 - mask) + grad_output * mask * slope
        grad_slope = grad_output * input * mask
        grad_bias = grad_output * mask
        grad_mask = -s + (input * slope + bias)
        return grad_input, grad_mask, grad_slope, grad_bias

    @staticmethod
    def symbolic(g, input, mask, weight, bias):
        # This will be parsed as a custom operation when doing the ONNX conversion.
        return g.op("customOp::LinearMaskedSigmoid", input, mask, weight, bias)


class LinearMaskedSigmoid(nn.Module):
    """Create a module to wrap the parameters for LinearMaskedSigmoidOp."""
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, )
        # All mask, slope and bias are element-wise.
        self.register_buffer('mask', (torch.rand(size=size) > 0).to(dtype=torch.get_default_dtype()))
        self.register_buffer('alpha', torch.rand(size=size, dtype=torch.get_default_dtype()))
        self.register_buffer('bias', torch.rand(size=size, dtype=torch.get_default_dtype()))

    def forward(self, input):
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using Sigmoid.
        return LinearMaskedSigmoidOp.apply(input, self.mask, self.slope, self.bias)


class BoundLinearMaskedSigmoid(BoundSigmoid):
    """This class defines how we compute the bounds for our customized Sigmoid function."""

    def forward(self, x, mask=None, slope=None, bias=None):
        """Regular forward propagation (e.g., for evaluating clean accuracy)."""
        if mask is None or slope is None or bias is None:
            # If mask is not given, then just act like a sigmoid.
            return super().forward(x)
        return LinearMaskedSigmoidOp.apply(x, mask, slope, bias)

    def interval_propagate(self, x, mask, slope, bias):
        """Interval bound propagation (IBP)."""
        # Each x, mask, slope, bias is a tuple, or a Interval object representing lower and upper bounds.
        # We assume Linf norm perturbation on input.
        assert Interval.get_perturbation(x)[0] == float("inf")
        x_L, x_U = x[0], x[1]  # The inputs (x)
        # We assume no perturbations on mask, slope and bias.
        mask, slope, bias = mask[0], slope[0], bias[0]
        # Lower and upper bounds when Sigmoid is selected.
        sigmoid_lb = torch.sigmoid(x_L)
        sigmoid_ub = torch.sigmoid(x_U)
        # Lower and upper bounds when linear coefficients are selected.
        pos_slope = (slope >= 0).to(dtype=torch.get_default_dtype())
        neg_slope = 1.0 - pos_slope
        linear_lb = pos_slope * slope * x_L + neg_slope * slope * x_U + bias
        linear_ub = pos_slope * slope * x_U + neg_slope * slope * x_L + bias
        # Select the final bounds according to the mask.
        final_lb = mask * linear_lb + (1.0 - mask) * sigmoid_lb
        final_ub = mask * linear_ub + (1.0 - mask) * sigmoid_ub
        return final_lb, final_ub

    def bound_relax(self, x, init=False, dim_opt=None):
        """Element-wise CROWN relaxation for our special Sigmoid activation function."""
        # Call parent class to relax Sigmoid neurons.
        super().bound_relax(x, init=init, dim_opt=dim_opt)
        # Modify the relaxation coefficients for these linear neurons.
        neg_mask = 1.0 - self._mask
        masked_slope = self._mask * self._slope
        masked_bias = self._mask * self._bias
        self.uw = masked_slope + neg_mask * self.uw
        self.ub = masked_bias + neg_mask * self.ub
        self.lw = masked_slope + neg_mask * self.lw
        self.lb = masked_bias + neg_mask * self.lb

    def bound_backward(self, last_lA, last_uA, x, mask, slope, bias, **kwargs):
        """Backward LiRPA (CROWN) bound propagation."""
        # These are additional variables that will be used in _backward_relaxation(), so we save them here.
        self._mask = mask.buffer  # These are registered as buffers; see class BoundBuffer.
        self._slope = slope.buffer
        self._bias = bias.buffer
        # The parent class will call _backward_relaxation() and obtain the relaxations,
        # and that's all we need; after obtaining linear relaxations for each neuron, other
        # parts of class BoundSigmoid can be reused.
        As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
        # Returned As = [(lA, uA)]; these A matrices are for input x.
        # Our customized Sigmoid has three additional buffers as inputs; we need to set their
        # corresponding A matrices to None. The length of As must match the number of inputs
        # of this customize function.
        As += [(None, None), (None, None), (None, None)]
        return As, lbias, ubias


class linear_masked_sigmoid_model(nn.Module):
    """Model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=0)
        # Using our customized Sigmoid function to replace the original Sigmoid function.
        self.linear_masked_sigmoid1 = LinearMaskedSigmoid(size=(16,15,15))
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=0)
        self.linear_masked_sigmoid2 = LinearMaskedSigmoid(size=(32,6,6))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*6*6,100)
        self.linear_masked_sigmoid3 = LinearMaskedSigmoid(100)
        self.linear2 = nn.Linear(100, 10)
        # Register the customized op in auto_LiRPA library.
        register_custom_op("customOp::LinearMaskedSigmoid", BoundLinearMaskedSigmoid)

    def forward(self, x):
        out = self.conv1(x)
        out = self.linear_masked_sigmoid1(out)
        # out = torch.nn.functional.sigmoid(out)
        out = self.conv2(out)
        out = self.linear_masked_sigmoid2(out)
        # out = torch.nn.functional.sigmoid(out)
        out = self.flatten(out)  # Flatten must be after activation for most efficient computation.
        out = self.linear1(out)
        out = self.linear_masked_sigmoid3(out)
        # out = torch.nn.functional.sigmoid(out)
        out = self.linear2(out)
        return out

