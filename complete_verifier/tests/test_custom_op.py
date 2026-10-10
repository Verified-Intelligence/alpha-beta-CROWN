#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Unit tests for custom/custom_op.py"""
import os
import sys
import unittest

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLinearMaskedReluOp(unittest.TestCase):
    """Tests for LinearMaskedReluOp function."""

    def test_forward_all_relu(self):
        """Test forward when all neurons use ReLU (mask=0)."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        mask = torch.zeros(5)
        slope = torch.ones(5)
        bias = torch.zeros(5)
        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_all_linear(self):
        """Test forward when all neurons use linear (mask=1)."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        mask = torch.ones(5)
        slope = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        expected = input_tensor * 2.0 + 1.0
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_mixed_mask(self):
        """Test forward with mixed mask."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.tensor([-1.0, 1.0])
        mask = torch.tensor([0.0, 1.0])
        slope = torch.tensor([2.0, 2.0])
        bias = torch.tensor([0.5, 0.5])
        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        # First element: ReLU(-1) = 0
        # Second element: 1.0 * 2.0 + 0.5 = 2.5
        expected = torch.tensor([0.0, 2.5])
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_batch(self):
        """Test forward with batch input."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.randn(4, 10)
        mask = (torch.rand(10) > 0.5).float()
        slope = torch.rand(10)
        bias = torch.rand(10)
        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        self.assertEqual(result.shape, input_tensor.shape)


class TestLinearMaskedRelu(unittest.TestCase):
    """Tests for LinearMaskedRelu module."""

    def test_init_with_int(self):
        """Test initialization with integer size."""
        from custom.custom_op import LinearMaskedRelu
        module = LinearMaskedRelu(10)
        self.assertEqual(module.mask.shape, (10,))
        self.assertEqual(module.slope.shape, (10,))
        self.assertEqual(module.bias.shape, (10,))

    def test_init_with_tuple(self):
        """Test initialization with tuple size."""
        from custom.custom_op import LinearMaskedRelu
        module = LinearMaskedRelu((5, 3))
        self.assertEqual(module.mask.shape, (5, 3))
        self.assertEqual(module.slope.shape, (5, 3))
        self.assertEqual(module.bias.shape, (5, 3))

    def test_forward(self):
        """Test forward pass."""
        from custom.custom_op import LinearMaskedRelu
        module = LinearMaskedRelu(10)
        input_tensor = torch.randn(4, 10)
        output = module(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)


class TestLinearMaskedReluOpBackward(unittest.TestCase):
    """Tests for LinearMaskedReluOp backward pass."""

    def test_backward_gradients(self):
        """Test that backward pass computes gradients."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.tensor([-1.0, 0.5, 1.0], requires_grad=True)
        mask = torch.tensor([0.0, 1.0, 0.0])
        slope = torch.tensor([1.0, 2.0, 1.0], requires_grad=True)
        bias = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

    def test_backward_relu_gradient(self):
        """Test backward gradient for ReLU part."""
        from custom.custom_op import LinearMaskedReluOp
        input_tensor = torch.tensor([-1.0, 1.0], requires_grad=True)
        mask = torch.tensor([0.0, 0.0])  # All ReLU
        slope = torch.tensor([1.0, 1.0])
        bias = torch.tensor([0.0, 0.0])

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        result.sum().backward()

        # For ReLU: grad = 0 when input < 0, grad = 1 when input > 0
        expected_grad = torch.tensor([0.0, 1.0])
        self.assertTrue(torch.allclose(input_tensor.grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
