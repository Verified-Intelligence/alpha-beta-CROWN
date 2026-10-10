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
"""Unit tests for heuristics/utils.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.utils import compute_ratio, get_preact_params, get_babsr_biases
from auto_LiRPA.bound_ops import BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd


class TestComputeRatio(unittest.TestCase):
    """Tests for compute_ratio function."""

    def test_basic_ratio(self):
        """Test basic ratio computation with simple bounds."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])
        slope_ratio, intercept = compute_ratio(lower, upper)
        # slope = upper / (upper - lower) = 1 / (1 - (-1)) = 0.5
        self.assertAlmostEqual(slope_ratio.item(), 0.5, places=5)
        # intercept = -lower * slope = -(-1) * 0.5 = 0.5
        self.assertAlmostEqual(intercept.item(), 0.5, places=5)

    def test_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([4.0])
        slope_ratio, intercept = compute_ratio(lower, upper)
        # slope = 4 / (4 - (-2)) = 4/6 = 2/3
        self.assertAlmostEqual(slope_ratio.item(), 4.0/6.0, places=5)
        # intercept = -(-2) * (4/6) = 2 * 4/6 = 8/6 = 4/3
        self.assertAlmostEqual(intercept.item(), 8.0/6.0, places=5)

    def test_positive_lower_bound(self):
        """Test when lower bound is positive (ReLU is always active)."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])
        slope_ratio, intercept = compute_ratio(lower, upper)
        # lower_temp = clamp(1.0, max=0) = 0
        # slope = 2 / (2 - 0) = 1
        self.assertAlmostEqual(slope_ratio.item(), 1.0, places=5)
        # intercept = -0 * 1 = 0
        self.assertAlmostEqual(intercept.item(), 0.0, places=5)

    def test_negative_upper_bound(self):
        """Test when upper bound is negative (ReLU is always inactive)."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-1.0])
        slope_ratio, intercept = compute_ratio(lower, upper)
        # upper_temp = relu(-1) = 0
        # slope = 0 / (0 - (-2)) = 0
        self.assertAlmostEqual(slope_ratio.item(), 0.0, places=5)
        # intercept = -(-2) * 0 = 0
        self.assertAlmostEqual(intercept.item(), 0.0, places=5)

    def test_batch_computation(self):
        """Test with batch of bounds."""
        lower = torch.tensor([-1.0, -2.0, 0.5])
        upper = torch.tensor([1.0, 2.0, 1.5])
        slope_ratio, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope_ratio.shape, (3,))
        self.assertEqual(intercept.shape, (3,))
        # First element: slope = 1/(1-(-1)) = 0.5
        self.assertAlmostEqual(slope_ratio[0].item(), 0.5, places=5)
        # Second element: slope = 2/(2-(-2)) = 0.5
        self.assertAlmostEqual(slope_ratio[1].item(), 0.5, places=5)
        # Third element: lower is positive, so slope = 1
        self.assertAlmostEqual(slope_ratio[2].item(), 1.0, places=5)

    def test_multidimensional(self):
        """Test with multidimensional tensors."""
        lower = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        upper = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        slope_ratio, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope_ratio.shape, (2, 2))
        self.assertEqual(intercept.shape, (2, 2))

    def test_zero_crossing(self):
        """Test bounds that cross zero."""
        lower = torch.tensor([-0.5])
        upper = torch.tensor([0.5])
        slope_ratio, intercept = compute_ratio(lower, upper)
        # slope = 0.5 / (0.5 - (-0.5)) = 0.5
        self.assertAlmostEqual(slope_ratio.item(), 0.5, places=5)
        # intercept = -(-0.5) * 0.5 = 0.25
        self.assertAlmostEqual(intercept.item(), 0.25, places=5)


def create_mock_bound_node(bound_class, inputs):
    """Create a mock bound node of the exact type using __new__.

    The get_babsr_biases function uses type() equality checks, so subclasses
    won't work. We use __new__ to create instances without calling __init__.
    """
    obj = object.__new__(bound_class)
    obj.inputs = inputs
    return obj


class MockParam:
    """Mock parameter wrapper."""
    def __init__(self, value):
        self.param = torch.tensor(value, dtype=torch.float32)

    def detach(self):
        return self.param


class TestGetBabsrBiases(unittest.TestCase):
    """Tests for get_babsr_biases function."""

    def _create_mock_param(self, value):
        """Create a mock input with param attribute."""
        return MockParam(value)

    def test_bound_conv_with_bias(self):
        """Test BoundConv layer with bias."""
        mock_act = MagicMock()

        # BoundConv with bias: inputs has > 2 elements, last is bias
        mock_bias = self._create_mock_param([1.0, 2.0, 3.0])
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock(), mock_bias])
        mock_act.inputs = [mock_conv]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        # Bias should be unsqueezed twice for conv
        self.assertEqual(biases[0].shape, (3, 1, 1))

    def test_bound_conv_without_bias(self):
        """Test BoundConv layer without bias."""
        mock_act = MagicMock()

        # BoundConv without bias: inputs has <= 2 elements
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock()])
        mock_act.inputs = [mock_conv]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertEqual(biases[0], 0)

    def test_bound_linear(self):
        """Test BoundLinear layer."""
        mock_act = MagicMock()

        # BoundLinear: bias is inputs[2]
        mock_bias = self._create_mock_param([0.5, 1.5])
        mock_linear = create_mock_bound_node(BoundLinear, [MagicMock(), MagicMock(), mock_bias])
        mock_act.inputs = [mock_linear]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertTrue(torch.allclose(biases[0], torch.tensor([0.5, 1.5])))

    def test_bound_add_simple(self):
        """Test BoundAdd layer without nested convolutions."""
        mock_act = MagicMock()

        # BoundAdd with no conv or bn inputs
        mock_add = create_mock_bound_node(BoundAdd, [MagicMock(), MagicMock()])
        mock_act.inputs = [mock_add]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertEqual(biases[0], 0)

    def test_bound_add_with_conv(self):
        """Test BoundAdd layer with nested BoundConv."""
        mock_act = MagicMock()

        # Create a nested BoundConv with bias
        mock_bias = self._create_mock_param([2.0, 3.0])
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock(), mock_bias])
        mock_add = create_mock_bound_node(BoundAdd, [mock_conv])
        mock_act.inputs = [mock_add]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertTrue(torch.allclose(biases[0], torch.tensor([2.0, 3.0])))

    def test_bound_batchnorm(self):
        """Test BoundBatchNormalization layer."""
        mock_act = MagicMock()

        # BN: bias is inputs[-3]
        mock_bias = self._create_mock_param([0.1, 0.2, 0.3])
        mock_bn = create_mock_bound_node(BoundBatchNormalization,
                                          [MagicMock(), MagicMock(), mock_bias, MagicMock(), MagicMock()])
        mock_act.inputs = [mock_bn]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertTrue(torch.allclose(biases[0], torch.tensor([0.1, 0.2, 0.3])))

    def test_unknown_layer_with_zero_default(self):
        """Test unknown layer type with zero_default=True."""
        mock_act = MagicMock()
        mock_unknown = MagicMock()
        mock_act.inputs = [mock_unknown]

        with patch('builtins.print') as mock_print:
            biases = get_babsr_biases(mock_act, zero_default=True)
            self.assertEqual(len(biases), 1)
            self.assertEqual(biases[0], 0)
            mock_print.assert_called()

    def test_unknown_layer_raises_error(self):
        """Test unknown layer type raises NotImplementedError."""
        mock_act = MagicMock()
        mock_unknown = MagicMock()
        mock_act.inputs = [mock_unknown]

        with self.assertRaises(NotImplementedError):
            get_babsr_biases(mock_act, zero_default=False)

    def test_multiple_inputs(self):
        """Test activation with multiple input nodes."""
        mock_act = MagicMock()

        # Create a BoundLinear
        mock_bias1 = self._create_mock_param([1.0])
        mock_linear = create_mock_bound_node(BoundLinear, [MagicMock(), MagicMock(), mock_bias1])

        # Create a BoundConv without bias
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock()])

        mock_act.inputs = [mock_linear, mock_conv]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 2)
        self.assertTrue(torch.allclose(biases[0], torch.tensor([1.0])))
        self.assertEqual(biases[1], 0)


class TestGetPreactParams(unittest.TestCase):
    """Tests for get_preact_params function."""

    def _create_mock_param(self, value):
        """Create a mock input with param attribute."""
        return MockParam(value)

    def test_single_input(self):
        """Test get_preact_params with single input."""
        mock_act = MagicMock()

        mock_bias = self._create_mock_param([0.5])
        mock_linear = create_mock_bound_node(BoundLinear, [MagicMock(), MagicMock(), mock_bias])
        mock_act.inputs = [mock_linear]

        result = get_preact_params(mock_act)
        self.assertTrue(torch.allclose(result, torch.tensor([0.5])))

    def test_multiple_inputs_raises_assertion(self):
        """Test that multiple inputs raises assertion error."""
        mock_act = MagicMock()
        mock_act.inputs = [MagicMock(), MagicMock()]

        with self.assertRaises(AssertionError):
            get_preact_params(mock_act)

    def test_zero_default_propagates(self):
        """Test that zero_default parameter is passed to get_babsr_biases."""
        mock_act = MagicMock()
        mock_unknown = MagicMock()
        mock_act.inputs = [mock_unknown]

        with patch('builtins.print'):
            result = get_preact_params(mock_act, zero_default=True)
            self.assertEqual(result, 0)


class TestBoundAddNestedCases(unittest.TestCase):
    """Additional tests for BoundAdd with nested structures."""

    def _create_mock_param(self, value):
        """Create a mock input with param attribute."""
        return MockParam(value)

    def test_bound_add_with_batchnorm(self):
        """Test BoundAdd with nested BoundBatchNormalization."""
        mock_act = MagicMock()

        # Create nested BoundBatchNormalization
        mock_bn = create_mock_bound_node(BoundBatchNormalization, [MagicMock()])
        mock_add = create_mock_bound_node(BoundAdd, [mock_bn])
        mock_act.inputs = [mock_add]

        biases = get_babsr_biases(mock_act)
        # BN in BoundAdd just adds 0
        self.assertEqual(len(biases), 1)
        self.assertEqual(biases[0], 0)

    def test_bound_add_with_nested_add(self):
        """Test BoundAdd with nested BoundAdd containing BoundConv."""
        mock_act = MagicMock()

        # Create BoundConv in inner add with bias
        mock_bias = self._create_mock_param([5.0, 6.0])
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock(), mock_bias])

        # Create nested BoundAdd
        mock_inner_add = create_mock_bound_node(BoundAdd, [mock_conv])
        mock_outer_add = create_mock_bound_node(BoundAdd, [mock_inner_add])
        mock_act.inputs = [mock_outer_add]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        self.assertTrue(torch.allclose(biases[0], torch.tensor([5.0, 6.0])))

    def test_bound_add_with_nested_add_no_bias_conv(self):
        """Test BoundAdd with nested BoundAdd containing BoundConv without bias."""
        mock_act = MagicMock()

        # Create BoundConv without bias (only 2 inputs)
        mock_conv = create_mock_bound_node(BoundConv, [MagicMock(), MagicMock()])

        # Create nested BoundAdd
        mock_inner_add = create_mock_bound_node(BoundAdd, [mock_conv])
        mock_outer_add = create_mock_bound_node(BoundAdd, [mock_inner_add])
        mock_act.inputs = [mock_outer_add]

        biases = get_babsr_biases(mock_act)
        self.assertEqual(len(biases), 1)
        # Should be 0 since conv has no bias
        self.assertEqual(biases[0], 0)


if __name__ == '__main__':
    unittest.main()
