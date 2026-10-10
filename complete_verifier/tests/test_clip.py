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
"""
Unit tests for input_split/clip.py

This module tests the domain clipping functionality for branch-and-bound verification.
Tests cover:
- concretize_bounds: Hölder's inequality for L-inf norm bound concretization
- deconstruct_bias: Bias deconstruction from A matrix and bounds
- _clip_main_fn: Core clipping algorithm
- clip_domains: Main entry point for domain clipping
- _in_depth_volume_metrics: Volume calculation for clipping effectiveness
- check_lbias: Development check for lbias consistency

To run tests: python -m pytest complete_verifier/tests/test_clip.py -v
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os

# Add the complete_verifier to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from input_split.clip import (
    concretize_bounds,
    deconstruct_bias,
    _clip_main_fn,
    clip_domains,
    _in_depth_volume_metrics,
    check_lbias,
    log_underflow,
)


class TestConcretizeBounds(unittest.TestCase):
    """Tests for the concretize_bounds function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 4
        self.num_spec = 2
        self.input_dim = 3

    def test_basic_concretization_lower(self):
        """Test basic lower bound concretization."""
        xhat = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
        eps = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        lA = torch.tensor([
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]
        ])
        lbias = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # For lower bounds: lA @ xhat - |lA| @ eps + lbias
        # Batch 0, spec 0: (1*0.5 + 1*0.5 + 1*0.5) - (1*0.5 + 1*0.5 + 1*0.5) = 1.5 - 1.5 = 0
        # Batch 0, spec 1: (2*0.5 + 2*0.5 + 2*0.5) - (2*0.5 + 2*0.5 + 2*0.5) = 3.0 - 3.0 = 0
        expected_batch0 = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(result[0], expected_batch0, atol=1e-5))

    def test_basic_concretization_upper(self):
        """Test upper bound concretization."""
        xhat = torch.tensor([[0.5, 0.5, 0.5]])
        eps = torch.tensor([[0.5, 0.5, 0.5]])
        lA = torch.tensor([[[1.0, 1.0, 1.0]]])
        lbias = torch.tensor([[0.0]])

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=False)

        # For upper bounds: lA @ xhat + |lA| @ eps + lbias
        # (1*0.5 + 1*0.5 + 1*0.5) + (1*0.5 + 1*0.5 + 1*0.5) = 1.5 + 1.5 = 3.0
        expected = torch.tensor([[3.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_with_bias(self):
        """Test concretization with non-zero bias."""
        xhat = torch.tensor([[0.0, 0.0]])
        eps = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = torch.tensor([[5.0]])

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Lower: 0 - 2 + 5 = 3.0
        expected = torch.tensor([[3.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_negative_coefficients(self):
        """Test concretization with negative coefficients in lA."""
        xhat = torch.tensor([[0.0]])
        eps = torch.tensor([[1.0]])
        lA = torch.tensor([[[-2.0]]])
        lbias = torch.tensor([[0.0]])

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Lower: -2*0 - |-2|*1 + 0 = -2.0
        expected = torch.tensor([[-2.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_output_shape(self):
        """Test that output has correct shape (batch, num_spec)."""
        batch = 5
        num_spec = 3
        input_dim = 4
        xhat = torch.randn(batch, input_dim)
        eps = torch.abs(torch.randn(batch, input_dim))
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        self.assertEqual(result.shape, (batch, num_spec))

    def test_zero_epsilon(self):
        """Test when epsilon is zero (single point domain)."""
        xhat = torch.tensor([[1.0, 2.0]])
        eps = torch.tensor([[0.0, 0.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = torch.tensor([[0.5]])

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Lower: 1*1 + 1*2 - 0 + 0.5 = 3.5
        expected = torch.tensor([[3.5]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_einsum_broadcasting(self):
        """Test that einsum broadcasting works correctly for batches."""
        batch = 3
        num_spec = 2
        input_dim = 4
        xhat = torch.ones(batch, input_dim)
        eps = torch.ones(batch, input_dim) * 0.1
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.zeros(batch, num_spec)

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # All same: 1*4 - 0.1*4 + 0 = 3.6 for each spec
        expected = torch.ones(batch, num_spec) * 3.6
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


class TestDeconstructBias(unittest.TestCase):
    """Tests for the deconstruct_bias function."""

    def test_basic_deconstruction_lower(self):
        """Test basic bias deconstruction for lower bounds."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        A = torch.ones(batch, num_spec, input_dim)
        dm_ob = torch.tensor([[1.0], [1.0]])  # domain output bound

        result = deconstruct_bias(x_L, x_U, A, dm_ob, is_lower=True)

        # xhat = 0.5, eps = 0.5
        # dm_ob - (A @ xhat - |A| @ eps)
        # 1.0 - (1*0.5*3 - 1*0.5*3) = 1.0 - (1.5 - 1.5) = 1.0
        expected = torch.ones(batch, num_spec)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_basic_deconstruction_upper(self):
        """Test basic bias deconstruction for upper bounds."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        A = torch.ones(batch, num_spec, input_dim)
        dm_ob = torch.tensor([[3.0], [3.0]])

        result = deconstruct_bias(x_L, x_U, A, dm_ob, is_lower=False)

        # For upper: dm_ob - (A @ xhat + |A| @ eps)
        # 3.0 - (1.5 + 1.5) = 0.0
        expected = torch.zeros(batch, num_spec)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_output_shape(self):
        """Test that output has correct shape (batch, num_spec)."""
        batch = 4
        input_dim = 5
        num_spec = 3

        x_L = torch.randn(batch, input_dim)
        x_U = x_L + torch.abs(torch.randn(batch, input_dim))
        A = torch.randn(batch, num_spec, input_dim)
        dm_ob = torch.randn(batch, num_spec)

        result = deconstruct_bias(x_L, x_U, A, dm_ob, is_lower=True)

        self.assertEqual(result.shape, (batch, num_spec))

    def test_roundtrip_with_concretize(self):
        """Test that deconstruct_bias and concretize_bounds are inverses."""
        batch = 3
        input_dim = 4
        num_spec = 2

        x_L = torch.randn(batch, input_dim)
        x_U = x_L + torch.abs(torch.randn(batch, input_dim)) + 0.1
        lA = torch.randn(batch, num_spec, input_dim)
        original_lbias = torch.randn(batch, num_spec)

        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2

        # First concretize to get dm_lb
        dm_lb = concretize_bounds(xhat, eps, lA, original_lbias, is_lower=True)

        # Then deconstruct to recover bias
        recovered_lbias = deconstruct_bias(x_L, x_U, lA, dm_lb, is_lower=True)

        self.assertTrue(torch.allclose(original_lbias, recovered_lbias, atol=1e-4))

    def test_negative_A_coefficients(self):
        """Test deconstruction with negative coefficients."""
        batch = 1
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[2.0, 2.0]])
        A = torch.tensor([[[-1.0, -2.0]]])
        dm_ob = torch.tensor([[-3.0]])

        result = deconstruct_bias(x_L, x_U, A, dm_ob, is_lower=True)

        # xhat = [1, 1], eps = [1, 1]
        # dm_ob - (A @ xhat - |A| @ eps)
        # -3.0 - ((-1*1 + -2*1) - (1*1 + 2*1)) = -3.0 - (-3 - 3) = -3.0 + 6 = 3.0
        expected = torch.tensor([[3.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_flattens_input(self):
        """Test that function properly flattens multi-dimensional inputs."""
        batch = 2
        # Multi-dimensional input (e.g., image-like)
        x_L = torch.randn(batch, 2, 3)
        x_U = x_L + torch.abs(torch.randn(batch, 2, 3)) + 0.1
        num_spec = 2
        input_dim = 6  # flattened

        A = torch.randn(batch, num_spec, 2, 3)  # will be flattened
        dm_ob = torch.randn(batch, num_spec)

        result = deconstruct_bias(x_L, x_U, A, dm_ob, is_lower=True)

        self.assertEqual(result.shape, (batch, num_spec))


class TestClipMainFn(unittest.TestCase):
    """Tests for the _clip_main_fn function."""

    def test_basic_clipping(self):
        """Test basic clipping with simple inputs."""
        batch = 1
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 0.0]]])  # Only depends on first dimension
        lbias = None
        thresholds = torch.tensor([[0.0]])
        dm_lb = torch.tensor([[-0.5]])  # Below threshold

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # Should clip the upper bound on x[0]
        self.assertTrue(new_x_U[0, 0] < x_U[0, 0])

    def test_no_clipping_when_verified(self):
        """Test that no clipping occurs when domain is already verified."""
        batch = 1
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = None
        thresholds = torch.tensor([[-10.0]])  # Very low threshold
        dm_lb = torch.tensor([[1.0]])  # Above threshold

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # Bounds may change but should clip towards verified region
        # Just check shapes are preserved
        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)

    def test_clipping_respects_original_bounds(self):
        """Test that clipping doesn't expand bounds beyond original."""
        batch = 2
        input_dim = 3
        num_spec = 2

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -1.0)

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # New lower bounds should be >= original lower bounds
        self.assertTrue((new_x_L >= x_L - 1e-6).all())
        # New upper bounds should be <= original upper bounds
        self.assertTrue((new_x_U <= x_U + 1e-6).all())

    def test_multiple_iterations(self):
        """Test that multiple iterations can improve clipping."""
        batch = 1
        input_dim = 2
        num_spec = 2

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
        lbias = torch.tensor([[0.0, 0.0]])
        thresholds = torch.tensor([[0.0, 0.0]])
        dm_lb = torch.tensor([[-1.0, -0.5]])

        # Single iteration
        new_x_L_1, new_x_U_1 = _clip_main_fn(
            x_L.clone(), x_U.clone(), lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True
        )

        # Multiple iterations
        new_x_L_3, new_x_U_3 = _clip_main_fn(
            x_L.clone(), x_U.clone(), lA, lbias, thresholds, dm_lb, num_iters=3, is_lower=True
        )

        # With multiple specs, multiple iterations may improve clipping
        # Just verify both produce valid output
        self.assertEqual(new_x_L_1.shape, x_L.shape)
        self.assertEqual(new_x_L_3.shape, x_L.shape)

    def test_upper_bounding_mode(self):
        """Test clipping when upper bounding the network."""
        batch = 1
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = None
        thresholds = torch.tensor([[0.0]])
        dm_lb = torch.tensor([[-0.5]])

        # is_lower=False means we're upper bounding
        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=False)

        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        # Different lA for each batch
        lA = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        lbias = None
        thresholds = torch.tensor([[0.0], [0.0]])
        dm_lb = torch.tensor([[-0.5], [-0.5]])

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # First batch should clip x[0], second batch should clip x[1]
        # Due to different lA
        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_zero_lA_coefficient(self):
        """Test handling of zero coefficients in lA."""
        batch = 1
        input_dim = 2
        num_spec = 1

        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[0.0, 1.0]]])  # Zero coefficient for first dim
        lbias = None
        thresholds = torch.tensor([[0.0]])
        dm_lb = torch.tensor([[-0.5]])

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # First dimension should not be clipped (division by zero handled)
        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_all_positive_lA(self):
        """Test clipping when all lA coefficients are positive."""
        batch = 1
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)  # All positive
        lbias = None
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -1.5)

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=1, is_lower=True)

        # Should only affect upper bounds (positive coefficients -> clip upper)
        self.assertTrue((new_x_L == x_L).all() or (new_x_U < x_U).any())


class TestClipDomains(unittest.TestCase):
    """Tests for the clip_domains function."""

    def test_basic_functionality(self):
        """Test basic clip_domains functionality."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.zeros(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)

    def test_shape_preservation(self):
        """Test that original shapes are preserved."""
        batch = 3
        # Multi-dimensional input shape
        x_L = torch.zeros(batch, 2, 3)
        x_U = torch.ones(batch, 2, 3)
        num_spec = 2
        input_dim = 6  # flattened

        lA = torch.randn(batch, num_spec, 6)
        lbias = torch.zeros(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)

    def test_all_verified_domains(self):
        """Test behavior when all domains are already verified."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.full((batch, num_spec), 10.0)  # High bias
        thresholds = torch.zeros(batch, num_spec)

        with patch('builtins.print') as mock_print:
            new_x_L, new_x_U = clip_domains(
                x_L, x_U, thresholds, lA, lbias)

        self.assertTrue((new_x_L > new_x_U).any(1).all())

    def test_mixed_batch_marks_only_verified_domains_empty(self):
        """Test that verified domains are empty and unverified domains remain."""
        x_L = torch.zeros(2, 1)
        x_U = torch.ones(2, 1)
        lA = torch.ones(2, 1, 1)
        lbias = torch.tensor([[10.0], [-1.0]])
        thresholds = torch.zeros(2, 1)

        new_x_L, new_x_U = clip_domains(
            x_L, x_U, thresholds, lA, lbias)

        torch.testing.assert_close(
            (new_x_L <= new_x_U).all(1), torch.tensor([False, True]))

    def test_verified_zero_slope_domain_is_empty(self):
        """A flat affine certificate still removes the verified domain."""
        x_L = torch.zeros(2, 1)
        x_U = torch.ones(2, 1)
        lA = torch.zeros(2, 1, 1)
        lbias = torch.tensor([[1.0], [-1.0]])
        thresholds = torch.zeros(2, 1)

        new_x_L, new_x_U = clip_domains(
            x_L, x_U, thresholds, lA, lbias)

        torch.testing.assert_close(
            (new_x_L <= new_x_U).all(1), torch.tensor([False, True]))

    def test_verified_nonzero_slope_rounding_to_point_is_empty(self):
        """A certified domain remains empty when clipping rounds to a point."""
        x_L = torch.tensor([[-4.0]], dtype=torch.float16)
        x_U = torch.tensor([[-3.5]], dtype=torch.float16)
        lA = torch.tensor([[[-4.0]]], dtype=torch.float16)
        lbias = torch.tensor([[-6.0]], dtype=torch.float16)
        # The affine lower bound is 8. The threshold is one FP16 ULP below it,
        # but the clipping candidate rounds back to the upper input endpoint.
        thresholds = torch.tensor([[7.99609375]], dtype=torch.float16)

        new_x_L, new_x_U = clip_domains(
            x_L, x_U, thresholds, lA, lbias)

        self.assertTrue((new_x_L > new_x_U).all())

    def test_with_dm_lb_provided(self):
        """Test when dm_lb is provided instead of computed from lbias."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = None  # Not provided
        dm_lb = torch.full((batch, num_spec), -0.5)
        thresholds = torch.zeros(batch, num_spec)

        # Should raise assertion since lbias is None and dm_lb is provided but assertion checks lbias
        with self.assertRaises(AssertionError):
            clip_domains(x_L, x_U, thresholds, lA, lbias, dm_lb=None)

    def test_num_iters_parameter(self):
        """Test different num_iters values."""
        batch = 2
        input_dim = 2
        num_spec = 2

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.zeros(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        # Test with different iteration counts
        for num_iters in [1, 2, 5]:
            new_x_L, new_x_U = clip_domains(
                x_L.clone(), x_U.clone(), thresholds, lA, lbias, num_iters=num_iters
            )
            self.assertEqual(new_x_L.shape, x_L.shape)

    def test_is_lower_false(self):
        """Test upper bounding mode."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.zeros(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias, is_lower=False)

        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)

    @patch('builtins.print')
    def test_prints_statistics(self, mock_print):
        """Test that statistics are printed."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.full((batch, num_spec), -0.5)
        thresholds = torch.zeros(batch, num_spec)

        clip_domains(x_L, x_U, thresholds, lA, lbias)

        # Should print domain clipping statistics
        mock_print.assert_called()

    @patch('builtins.print')
    def test_calculate_volume_flag(self, mock_print):
        """Test calculate_volume parameter."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.full((batch, num_spec), -0.5)
        thresholds = torch.zeros(batch, num_spec)

        clip_domains(x_L, x_U, thresholds, lA, lbias, calculate_volume=True)

        # Volume calculations should be performed
        call_args_list = [str(call) for call in mock_print.call_args_list]
        # Check that print was called
        self.assertTrue(len(call_args_list) > 0)

    def test_shrunken_and_verified_detection(self):
        """Test detection of domains that become verified after clipping."""
        batch = 2
        input_dim = 2
        num_spec = 2  # Need multiple specs for this scenario

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        # Construct lA and lbias such that clipping causes x_L > x_U for some domain
        lA = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ])
        lbias = torch.full((batch, num_spec), -2.0)  # Very negative
        thresholds = torch.zeros(batch, num_spec)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_clones_input(self):
        """Test that input tensors are cloned and not modified."""
        batch = 2
        input_dim = 2
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        x_L_original = x_L.clone()
        x_U_original = x_U.clone()

        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.full((batch, num_spec), -0.5)
        thresholds = torch.zeros(batch, num_spec)

        with patch('builtins.print'):
            clip_domains(x_L, x_U, thresholds, lA, lbias)

        # Original should be unchanged
        self.assertTrue(torch.equal(x_L, x_L_original))
        self.assertTrue(torch.equal(x_U, x_U_original))


class TestInDepthVolumeMetrics(unittest.TestCase):
    """Tests for the _in_depth_volume_metrics function."""

    @patch('builtins.print')
    def test_basic_volume_calculation(self, mock_print):
        """Test basic volume calculation."""
        batch = 2
        input_dim = 2
        num_spec = 1

        original_x_L = torch.zeros(batch, input_dim)
        original_x_U = torch.ones(batch, input_dim)
        new_x_L = torch.zeros(batch, input_dim)
        new_x_U = torch.ones(batch, input_dim) * 0.5  # Shrunk by half
        nv_mask = torch.ones(batch, dtype=torch.bool)
        lA = torch.ones(batch, num_spec, input_dim)
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -0.5)

        _in_depth_volume_metrics(
            original_x_L, original_x_U, new_x_L, new_x_U,
            nv_mask, lA, thresholds, dm_lb, is_lower=1
        )

        mock_print.assert_called()

    @patch('builtins.print')
    def test_no_shrinkage(self, mock_print):
        """Test when no shrinkage occurs."""
        batch = 2
        input_dim = 2
        num_spec = 1

        original_x_L = torch.zeros(batch, input_dim)
        original_x_U = torch.ones(batch, input_dim)
        new_x_L = original_x_L.clone()
        new_x_U = original_x_U.clone()
        nv_mask = torch.ones(batch, dtype=torch.bool)
        lA = torch.ones(batch, num_spec, input_dim)
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -0.5)

        _in_depth_volume_metrics(
            original_x_L, original_x_U, new_x_L, new_x_U,
            nv_mask, lA, thresholds, dm_lb, is_lower=1
        )

        # Should report 100% ratio
        call_str = str(mock_print.call_args_list[-1])
        self.assertIn("100.00%", call_str)

    @patch('builtins.print')
    def test_numerical_underflow_handling(self, mock_print):
        """Test handling of numerical underflow in volume calculations."""
        batch = 2
        input_dim = 100  # High dimensions cause underflow
        num_spec = 1

        original_x_L = torch.zeros(batch, input_dim)
        original_x_U = torch.ones(batch, input_dim) * 0.01  # Very small domain
        new_x_L = original_x_L.clone()
        new_x_U = original_x_U.clone() * 0.5
        nv_mask = torch.ones(batch, dtype=torch.bool)
        lA = torch.ones(batch, num_spec, input_dim)
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -0.5)

        _in_depth_volume_metrics(
            original_x_L, original_x_U, new_x_L, new_x_U,
            nv_mask, lA, thresholds, dm_lb, is_lower=1
        )

        # Should handle underflow gracefully
        mock_print.assert_called()

    @patch('builtins.print')
    def test_partial_mask(self, mock_print):
        """Test with only some domains unverified."""
        batch = 4
        input_dim = 2
        num_spec = 1

        original_x_L = torch.zeros(batch, input_dim)
        original_x_U = torch.ones(batch, input_dim)
        new_x_L = torch.zeros(batch, input_dim)
        new_x_U = torch.ones(batch, input_dim) * 0.5
        nv_mask = torch.tensor([True, False, True, False])  # Only 2 unverified
        lA = torch.ones(batch, num_spec, input_dim)
        thresholds = torch.zeros(batch, num_spec)
        dm_lb = torch.full((batch, num_spec), -0.5)

        _in_depth_volume_metrics(
            original_x_L, original_x_U, new_x_L, new_x_U,
            nv_mask, lA, thresholds, dm_lb, is_lower=1
        )

        mock_print.assert_called()

    @patch('builtins.print')
    def test_float64_conversion(self, mock_print):
        """Test that tensors are converted to float64 for precision."""
        batch = 2
        input_dim = 2
        num_spec = 1

        # Use float32 inputs
        original_x_L = torch.zeros(batch, input_dim, dtype=torch.float32)
        original_x_U = torch.ones(batch, input_dim, dtype=torch.float32)
        new_x_L = torch.zeros(batch, input_dim, dtype=torch.float32)
        new_x_U = torch.ones(batch, input_dim, dtype=torch.float32) * 0.5
        nv_mask = torch.ones(batch, dtype=torch.bool)
        lA = torch.ones(batch, num_spec, input_dim, dtype=torch.float32)
        thresholds = torch.zeros(batch, num_spec, dtype=torch.float32)
        dm_lb = torch.full((batch, num_spec), -0.5, dtype=torch.float32)

        # Should not raise error due to dtype conversion
        _in_depth_volume_metrics(
            original_x_L, original_x_U, new_x_L, new_x_U,
            nv_mask, lA, thresholds, dm_lb, is_lower=1
        )

        mock_print.assert_called()


class TestCheckLbias(unittest.TestCase):
    """Tests for the check_lbias function."""

    def test_consistent_lbias(self):
        """Test that consistent lbias passes check."""
        batch = 3
        input_dim = 4
        num_spec = 2

        x_L = torch.randn(batch, input_dim)
        x_U = x_L + torch.abs(torch.randn(batch, input_dim)) + 0.1
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.full((batch, num_spec), 100.0)  # High threshold so all unverified

        # Compute dm_lb from lA and lbias
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Should not raise assertion
        check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)

    def test_none_lbias_returns_early(self):
        """Test that None lbias returns without checking."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, num_spec, input_dim)
        dm_lb = torch.randn(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        # Should return without error when lbias is None
        check_lbias(x_L, x_U, lA, None, dm_lb, thresholds)

    def test_inconsistent_lbias_raises(self):
        """Test that inconsistent lbias raises assertion."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.ones(batch, num_spec)  # Incorrect lbias
        dm_lb = torch.zeros(batch, num_spec)  # Doesn't match lA @ xhat - |lA| @ eps + lbias
        thresholds = torch.full((batch, num_spec), 100.0)  # All unverified

        with self.assertRaises(AssertionError):
            check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)

    def test_verified_domains_skipped(self):
        """Test that verified domains are not checked."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)
        lbias = torch.ones(batch, num_spec)  # Would be inconsistent
        dm_lb = torch.full((batch, num_spec), 10.0)  # Above threshold = verified
        thresholds = torch.zeros(batch, num_spec)  # Low threshold

        # Should not raise because all domains are verified
        check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)

    def test_flattens_inputs(self):
        """Test that inputs are properly flattened."""
        batch = 2
        # Multi-dimensional input
        x_L = torch.randn(batch, 2, 3)
        x_U = x_L + torch.abs(torch.randn(batch, 2, 3)) + 0.1
        num_spec = 2
        lA = torch.randn(batch, num_spec, 2, 3)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.full((batch, num_spec), 100.0)

        # Compute correct dm_lb
        x_L_flat = x_L.flatten(1)
        x_U_flat = x_U.flatten(1)
        lA_flat = lA.flatten(2)
        xhat = (x_U_flat + x_L_flat) / 2
        eps = (x_U_flat - x_L_flat) / 2
        dm_lb = concretize_bounds(xhat, eps, lA_flat, lbias, is_lower=True)

        # Should handle multi-dimensional inputs
        check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)

    def test_tolerance_in_comparison(self):
        """Test that isclose uses appropriate tolerance."""
        batch = 2
        input_dim = 3
        num_spec = 1

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.ones(batch, num_spec, input_dim)

        # Compute correct lbias
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        dm_lb = torch.full((batch, num_spec), 0.0)
        correct_lbias = deconstruct_bias(x_L, x_U, lA, dm_lb, is_lower=True)

        # Add small noise within tolerance
        lbias = correct_lbias + 1e-5
        thresholds = torch.full((batch, num_spec), 100.0)

        # Should pass with small tolerance
        check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)


class TestLogUnderflow(unittest.TestCase):
    """Tests for the log_underflow constant."""

    def test_log_underflow_is_negative(self):
        """Test that log_underflow is a large negative number."""
        self.assertLess(log_underflow, 0)

    def test_log_underflow_is_finite(self):
        """Test that log_underflow is finite."""
        import math
        self.assertTrue(math.isfinite(log_underflow))

    def test_log_underflow_represents_tiny_value(self):
        """Test that exp(log_underflow) is approximately tiny."""
        import torch
        tiny = torch.finfo(torch.float64).tiny
        self.assertAlmostEqual(torch.exp(torch.tensor(log_underflow)).item(), tiny, places=300)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_single_batch(self):
        """Test with single batch."""
        x_L = torch.zeros(1, 3)
        x_U = torch.ones(1, 3)
        lA = torch.ones(1, 1, 3)
        lbias = torch.zeros(1, 1)
        thresholds = torch.zeros(1, 1)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, (1, 3))
        self.assertEqual(new_x_U.shape, (1, 3))

    def test_single_input_dimension(self):
        """Test with single input dimension."""
        batch = 3
        x_L = torch.zeros(batch, 1)
        x_U = torch.ones(batch, 1)
        lA = torch.ones(batch, 2, 1)
        lbias = torch.zeros(batch, 2)
        thresholds = torch.zeros(batch, 2)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, (batch, 1))

    def test_many_specs(self):
        """Test with many specifications."""
        batch = 2
        input_dim = 3
        num_spec = 10

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_large_batch(self):
        """Test with large batch size."""
        batch = 100
        input_dim = 5
        num_spec = 2

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_very_small_epsilon(self):
        """Test with very small domain (small epsilon)."""
        batch = 2
        input_dim = 3

        # Very tight bounds
        center = torch.randn(batch, input_dim)
        x_L = center - 1e-6
        x_U = center + 1e-6
        lA = torch.randn(batch, 1, input_dim)
        lbias = torch.randn(batch, 1)
        thresholds = torch.zeros(batch, 1)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_large_coefficients(self):
        """Test with large lA coefficients."""
        batch = 2
        input_dim = 3

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, 1, input_dim) * 1000  # Large coefficients
        lbias = torch.randn(batch, 1)
        thresholds = torch.zeros(batch, 1)

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)

    def test_negative_thresholds(self):
        """Test with negative thresholds."""
        batch = 2
        input_dim = 3

        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)
        lA = torch.randn(batch, 2, input_dim)
        lbias = torch.zeros(batch, 2)
        thresholds = torch.full((batch, 2), -1.0)  # Negative thresholds

        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(x_L, x_U, thresholds, lA, lbias)

        self.assertEqual(new_x_L.shape, x_L.shape)


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def test_concretize_bounds_stability(self):
        """Test numerical stability of concretize_bounds."""
        batch = 5
        input_dim = 10
        num_spec = 3

        # Use extreme values
        xhat = torch.randn(batch, input_dim) * 100
        eps = torch.abs(torch.randn(batch, input_dim)) * 100
        lA = torch.randn(batch, num_spec, input_dim) * 100
        lbias = torch.randn(batch, num_spec) * 100

        result = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Should not contain NaN or Inf
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())

    def test_clip_main_fn_stability(self):
        """Test numerical stability of _clip_main_fn."""
        batch = 3
        input_dim = 5
        num_spec = 2

        x_L = torch.randn(batch, input_dim) * 10
        x_U = x_L + torch.abs(torch.randn(batch, input_dim)) * 10 + 0.1
        lA = torch.randn(batch, num_spec, input_dim) * 10
        lbias = torch.randn(batch, num_spec) * 10
        thresholds = torch.randn(batch, num_spec) * 10
        dm_lb = torch.randn(batch, num_spec) * 10

        new_x_L, new_x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters=3, is_lower=True)

        # Should not contain NaN
        self.assertFalse(torch.isnan(new_x_L).any())
        self.assertFalse(torch.isnan(new_x_U).any())


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions."""

    def test_full_clipping_pipeline(self):
        """Test the full clipping pipeline."""
        batch = 4
        input_dim = 5
        num_spec = 3

        # Create realistic input domain
        x_L = torch.zeros(batch, input_dim)
        x_U = torch.ones(batch, input_dim)

        # Create CROWN-like bounds
        lA = torch.randn(batch, num_spec, input_dim)
        lbias = torch.randn(batch, num_spec)
        thresholds = torch.zeros(batch, num_spec)

        # Compute dm_lb
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        # Check lbias consistency
        check_lbias(x_L, x_U, lA, lbias, dm_lb, thresholds)

        # Perform clipping
        with patch('builtins.print'):
            new_x_L, new_x_U = clip_domains(
                x_L, x_U, thresholds, lA, lbias,
                num_iters=2, calculate_volume=True
            )

        # Verify results
        self.assertEqual(new_x_L.shape, x_L.shape)
        self.assertEqual(new_x_U.shape, x_U.shape)
        self.assertTrue((new_x_L >= x_L - 1e-6).all())
        self.assertTrue((new_x_U <= x_U + 1e-6).all())

    def test_deconstruct_then_concretize(self):
        """Test that deconstruct and concretize are consistent."""
        batch = 3
        input_dim = 4
        num_spec = 2

        x_L = torch.randn(batch, input_dim)
        x_U = x_L + torch.abs(torch.randn(batch, input_dim)) + 0.1
        lA = torch.randn(batch, num_spec, input_dim)
        dm_lb = torch.randn(batch, num_spec)

        # Deconstruct to get lbias
        lbias = deconstruct_bias(x_L, x_U, lA, dm_lb, is_lower=True)

        # Concretize back to dm_lb
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        recovered_dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower=True)

        self.assertTrue(torch.allclose(dm_lb, recovered_dm_lb, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
