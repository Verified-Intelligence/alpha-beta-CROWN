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
relaxations for bound computation. Here we consider a modified ReLU
function which is a mixture between a linear function and ReLU function:
             / a_i x_i + b_i   if m_i = 1,
    f(x_i) = |
             \ ReLU(x_i)       if m_i = 0.
where a, b are element-wise slopes and biases when the function is linear,
and m is the mask controlling the behavior of this function. We consider
perturbations on x.

An example command to run verification on this customized model:

python abcrown.py --config exp_configs/tutorial_examples/custom_op_example.yaml --complete_verifier skip --batch_size 256

Note that if you also want to conduct branch and bound on your customized
op, you may also need to customize BaB code, so the complete verifier is
skipped here.
"""
import torch
import torch.nn as nn
from auto_LiRPA import register_custom_op
from auto_LiRPA.bound_ops import BoundRelu, Interval


class LinearMaskedReluOp(torch.autograd.Function):
    """A relu function with some neurons replaced with linear operations."""
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor, slope: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using ReLU
        ctx.save_for_backward(input, mask, slope, bias)
        return input.clamp(min=0) * (1.0 - mask) + (input * slope + bias) * mask

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, mask, slope, bias = ctx.saved_tensors
        relu_grad = grad_output.clone()
        relu_grad[input < 0] = 0
        grad_input = relu_grad * (1.0 - mask) + grad_output * mask * slope
        grad_slope = grad_output * input * mask
        grad_bias = grad_output * mask
        grad_mask = -input.clamp(min=0) + (input * slope + bias)
        return grad_input, grad_mask, grad_slope, grad_bias

    @staticmethod
    def symbolic(g, input, mask, weight, bias):
        # This will be parsed as a custom operation when doing the ONNX conversion.
        return g.op("customOp::LinearMaskedRelu", input, mask, weight, bias)


class LinearMaskedRelu(nn.Module):
    """Create a module to wrap the parameters for LinearMaskedReluOp."""
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, )
        # All mask, slope and bias are element-wise.
        self.register_buffer('mask', (torch.rand(size=size) > 0).to(dtype=torch.get_default_dtype()))
        self.register_buffer('alpha', torch.rand(size=size, dtype=torch.get_default_dtype()))
        self.register_buffer('bias', torch.rand(size=size, dtype=torch.get_default_dtype()))

    def forward(self, input):
        # mask = 1 => using linear operation input * slope + bias, mask = 0 => using ReLU
        return LinearMaskedReluOp.apply(input, self.mask, self.slope, self.bias)


class BoundLinearMaskedRelu(BoundRelu):
    """This class defines how we compute the bounds for our customized Relu function."""

    def forward(self, x, mask, slope, bias):
        """Regular forward propagation (e.g., for evaluating clean accuracy)."""
        # Save the shape, which will be used in other parts of the verifier.
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return LinearMaskedReluOp.apply(x, mask, slope, bias)

    def interval_propagate(self, x, mask, slope, bias):
        """Interval bound propagation (IBP)."""
        # Each x, mask, slope, bias is a tuple, or a Interval object representing lower and upper bounds.
        # We assume Linf norm perturbation on input.
        assert Interval.get_perturbation(x)[0] == float("inf")
        x_L, x_U = x[0], x[1]  # The inputs (x)
        # We assume no perturbations on mask, slope and bias.
        mask, slope, bias = mask[0], slope[0], bias[0]
        # Lower and upper bounds when ReLU is selected.
        relu_lb = x_L.clamp(min=0)
        relu_ub = x_U.clamp(min=0)
        # Lower and upper bounds when linear coefficients are selected.
        pos_slope = (slope >= 0).to(dtype=torch.get_default_dtype())
        neg_slope = 1.0 - pos_slope
        linear_lb = pos_slope * slope * x_L + neg_slope * slope * x_U + bias
        linear_ub = pos_slope * slope * x_U + neg_slope * slope * x_L + bias
        # Select the final bounds according to the mask.
        final_lb = mask * linear_lb + (1.0 - mask) * relu_lb
        final_ub = mask * linear_ub + (1.0 - mask) * relu_ub
        return final_lb, final_ub

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        """Element-wise CROWN relaxation for our special ReLU activation function."""
        # Call parent class to relax ReLU neurons.
        upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, alpha_lookup_idx = super()._backward_relaxation(
                last_lA, last_uA, x, start_node, unstable_idx)
        # Modify the relaxation coefficients for these linear neurons.
        neg_mask = 1.0 - self._mask
        masked_slope = self._mask * self._slope
        masked_bias = self._mask * self._bias
        upper_d = masked_slope + neg_mask * upper_d
        upper_b = masked_bias + neg_mask * upper_b
        if lower_d is not None:
            # Shared slope between lower and upper bounds.
            lower_d = masked_slope + neg_mask * lower_d
        else:
            # Not shared slopes: we have two set of slopes one for lA, one for uA.
            # One of them might be not necessary (None), if only lower or upper bound is computed.
            lb_lower_d = masked_slope + neg_mask * lb_lower_d if lb_lower_d is not None else None
            ub_lower_d = masked_slope + neg_mask * ub_lower_d if ub_lower_d is not None else None
        assert lower_b is None  # For ReLU, there is no lower bias (=0)
        # The required dimension is (batch, spec, C, H, W). The size of masked_bias is (C,H,W),
        # and we need to expand other dimensions.
        lower_b = masked_bias.unsqueeze(0).unsqueeze(0).expand(upper_b.size())
        return upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, alpha_lookup_idx

    def bound_backward(self, last_lA, last_uA, x, mask, slope, bias, **kwargs):
        """Backward LiRPA (CROWN) bound propagation."""
        # These are additional variables that will be used in _backward_relaxation(), so we save them here.
        self._mask = mask.buffer  # These are registered as buffers; see class BoundBuffer.
        self._slope = slope.buffer
        self._bias = bias.buffer
        # The parent class will call _backward_relaxation() and obtain the relaxations,
        # and that's all we need; after obtaining linear relaxations for each neuron, other
        # parts of class BoundRelu can be reused.
        As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
        # Returned As = [(lA, uA)]; these A matrices are for input x.
        # Our customized ReLU has three additional buffers as inputs; we need to set their
        # corresponding A matrices to None. The length of As must match the number of inputs
        # of this customize function.
        As += [(None, None), (None, None), (None, None)]
        return As, lbias, ubias


class linear_masked_relu_model(nn.Module):
    """Model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        # Using our customized ReLU function to replace the original ReLU function.
        self.linear_masked_relu1 = LinearMaskedRelu(size=(16,16,16))
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(32,8,8))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*8*8,100)
        self.linear_masked_relu3 = LinearMaskedRelu(100)
        self.linear2 = nn.Linear(100, 10)
        # Register the customized op in auto_LiRPA library.
        register_custom_op("customOp::LinearMaskedRelu", BoundLinearMaskedRelu)

    def forward(self, x):
        out = self.conv1(x)
        out = self.linear_masked_relu1(out)
        # out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.linear_masked_relu2(out)
        # out = torch.nn.functional.relu(out)
        out = self.flatten(out)  # Flatten must be after activation for most efficient computation.
        out = self.linear1(out)
        out = self.linear_masked_relu3(out)
        # out = torch.nn.functional.relu(out)
        out = self.linear2(out)
        return out

