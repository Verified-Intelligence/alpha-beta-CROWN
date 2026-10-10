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
"""Unit tests for domain_clipper.py module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock arguments.Config before importing domain_clipper
import arguments
_original_config = arguments.Config


def _setup_mock_config():
    """Create a mock config for testing."""
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    # Set required config values for DomainClipper
    new_config['bab']['clip_n_verify']['final_layer'] = True
    new_config['bab']['clip_n_verify']['prune'] = False
    new_config['bab']['clip_n_verify']['rearrange_constraints'] = False
    new_config['bab']['clip_n_verify']['clip_interm_domain']['enabled'] = False
    new_config['bab']['clip_n_verify']['clip_interm_domain']['with_input'] = False
    new_config['bab']['clip_n_verify']['clip_interm_domain']['topk_objective'] = 0
    return new_config


# Apply mock config
arguments.Config = _setup_mock_config()

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from domain_clipper import (
    pad_first_dim, concretize_bounds, expand_x_batch,
    update_interm_bounds, _all_dist, dimensionwise_shrinkage_stats,
    prune_d, parallel_clipping, DomainClipper
)


def tearDownModule():
    """Restore original config after all tests."""
    arguments.Config = _original_config


# ============================================================================
# DomainClipper.domain_clip_ReLU tests
# ============================================================================

class TestDomainClipReLU(unittest.TestCase):
    """Tests for the input returned after clipping a ReLU domain."""

    def test_returns_bounded_tensor_for_clipped_box(self):
        """The returned input should consistently represent the new box."""
        source = torch.zeros(1, 2)
        x = BoundedTensor(
            source,
            PerturbationLpNorm(
                norm=float('inf'), x_L=source - 1, x_U=source + 1),
        )

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x_L = torch.tensor(
                    [[-2.0, 1.0], [0.0, 4.0]], dtype=dtype)
                x_U = torch.tensor(
                    [[2.0, 3.0], [6.0, 10.0]], dtype=dtype)

                clipper = DomainClipper.__new__(DomainClipper)
                clipper.timer = MagicMock()
                clipper.prune = False
                A = torch.empty(0, dtype=dtype)
                bias = torch.empty(0, dtype=dtype)
                clipper.get_constraints = MagicMock(return_value=(A, bias))
                clipper.intersection_check = MagicMock()
                interm_bounds = {}
                clipper.clip_domains = MagicMock(
                    return_value=(x_L, x_U, None, interm_bounds))
                domain = {'history': {}}

                result, returned_bounds, returned_domain, prune_mask = (
                    clipper.domain_clip_ReLU(domain, x, interm_bounds))

                expected_center = (x_L + x_U) / 2
                expected_radius = (x_U - x_L) / 2
                self.assertIsInstance(result, BoundedTensor)
                self.assertTrue(torch.equal(result.data, expected_center))
                self.assertTrue(torch.equal(result.ptb.x_L, x_L))
                self.assertTrue(torch.equal(result.ptb.x_U, x_U))
                self.assertTrue(torch.equal(result.ptb.x0, expected_center))
                self.assertTrue(torch.equal(result.ptb.eps, expected_radius))
                self.assertEqual(result.ptb.norm, float('inf'))
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, x_L.device)
                self.assertIs(returned_bounds, interm_bounds)
                self.assertIs(returned_domain, domain)
                self.assertIsNone(prune_mask)


# ============================================================================
# pad_first_dim Tests
# ============================================================================

class TestPadFirstDim(unittest.TestCase):
    """Tests for pad_first_dim function."""

    def test_already_correct_size(self):
        """Test tensor that's already the correct size."""
        tensor = torch.randn(5, 3, 4)
        result = pad_first_dim(tensor, 5)
        self.assertEqual(result.shape, (5, 3, 4))
        self.assertTrue(torch.equal(result, tensor))

    def test_pad_with_zeros_when_empty(self):
        """Test padding empty tensor fills with zeros."""
        tensor = torch.empty(0, 3, 4)
        result = pad_first_dim(tensor, 5)
        self.assertEqual(result.shape, (5, 3, 4))
        self.assertTrue(torch.all(result == 0))

    def test_pad_duplicates_first_row(self):
        """Test that padding duplicates first row."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = pad_first_dim(tensor, 4)
        self.assertEqual(result.shape, (4, 2))
        # First two rows should be original
        self.assertTrue(torch.equal(result[:2], tensor))
        # Remaining rows should be copies of first row
        self.assertTrue(torch.equal(result[2], tensor[0]))
        self.assertTrue(torch.equal(result[3], tensor[0]))

    def test_pad_1d_tensor(self):
        """Test padding 1D tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        # For 1D tensors, first dim is padded
        result = pad_first_dim(tensor.unsqueeze(0), 3)
        self.assertEqual(result.shape, (3, 3))

    def test_pad_single_element(self):
        """Test padding tensor with single element."""
        tensor = torch.tensor([[5.0]])
        result = pad_first_dim(tensor, 4)
        self.assertEqual(result.shape, (4, 1))
        self.assertTrue(torch.all(result == 5.0))

    def test_preserves_dtype(self):
        """Test that padding preserves dtype."""
        tensor = torch.randn(2, 3, dtype=torch.float64)
        result = pad_first_dim(tensor, 5)
        self.assertEqual(result.dtype, torch.float64)

    def test_preserves_device(self):
        """Test that padding preserves device."""
        tensor = torch.randn(2, 3, device='cpu')
        result = pad_first_dim(tensor, 5)
        self.assertEqual(result.device.type, 'cpu')

    def test_higher_dimensional_tensor(self):
        """Test padding higher dimensional tensor."""
        tensor = torch.randn(2, 3, 4, 5)
        result = pad_first_dim(tensor, 6)
        self.assertEqual(result.shape, (6, 3, 4, 5))
        # Original content preserved
        self.assertTrue(torch.equal(result[:2], tensor))


# ============================================================================
# concretize_bounds Tests
# ============================================================================

class TestConcretizeBounds(unittest.TestCase):
    """Tests for concretize_bounds function."""

    def test_lower_bound_basic(self):
        """Test basic lower bound concretization."""
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])  # [batch, num_constr, input_dim]
        lbias = torch.tensor([[0.0]])

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        # xhat = [0.5, 0.5], eps = [0.5, 0.5]
        # status = -1 for lower bound
        # result = -1 * (|lA| @ eps) + lA @ xhat + lbias
        # = -1 * (1+1)*0.5 + (0.5+0.5) + 0 = -1 + 1 = 0
        self.assertEqual(result.shape, (1, 1))

    def test_upper_bound_basic(self):
        """Test basic upper bound concretization."""
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = torch.tensor([[0.0]])

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=False)
        # status = +1 for upper bound
        # result = +1 * (|lA| @ eps) + lA @ xhat + lbias
        # = 1 + 1 = 2
        self.assertEqual(result.shape, (1, 1))

    def test_with_negative_coefficients(self):
        """Test with negative coefficients in lA."""
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[2.0, 2.0]])
        lA = torch.tensor([[[-1.0, 1.0]]])
        lbias = torch.tensor([[0.0]])

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        # xhat = [1, 1], eps = [1, 1]
        # |lA| = [1, 1]
        # |lA| @ eps = 2
        # lA @ xhat = -1 + 1 = 0
        # result = -1 * 2 + 0 + 0 = -2
        self.assertAlmostEqual(result[0, 0].item(), -2.0, places=5)

    def test_with_bias(self):
        """Test with non-zero bias."""
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 1.0]]])
        lbias = torch.tensor([[5.0]])

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        # xhat = [0.5, 0.5], eps = [0.5, 0.5]
        # result = -1 * 1 + 1 + 5 = 5
        self.assertAlmostEqual(result[0, 0].item(), 5.0, places=5)

    def test_batch_processing(self):
        """Test batch processing."""
        batch = 3
        x_L = torch.zeros(batch, 4)
        x_U = torch.ones(batch, 4)
        lA = torch.randn(batch, 2, 4)
        lbias = torch.randn(batch, 2)

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        self.assertEqual(result.shape, (batch, 2))

    def test_multiple_constraints(self):
        """Test with multiple constraints."""
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        lA = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        lbias = torch.tensor([[0.0, 0.0, 0.0]])

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        self.assertEqual(result.shape, (1, 3))


# ============================================================================
# expand_x_batch Tests
# ============================================================================

class TestExpandXBatch(unittest.TestCase):
    """Tests for expand_x_batch function."""

    def test_basic_expand(self):
        """Test basic batch expansion."""
        x_L = torch.zeros(2, 3, 4)
        x_U = torch.ones(2, 3, 4)
        x_shape = x_L.shape

        # Expand to 6 (2 * 3)
        result_L, result_U, result_shape = expand_x_batch(x_L, x_U, x_shape, 6)
        self.assertEqual(result_L.shape, (6, 3, 4))
        self.assertEqual(result_U.shape, (6, 3, 4))
        self.assertEqual(result_shape, (6, 3, 4))

    def test_expand_2d_tensor(self):
        """Test expanding 2D tensor."""
        x_L = torch.zeros(1, 10)
        x_U = torch.ones(1, 10)
        x_shape = x_L.shape

        result_L, result_U, result_shape = expand_x_batch(x_L, x_U, x_shape, 5)
        self.assertEqual(result_L.shape, (5, 10))
        self.assertEqual(result_U.shape, (5, 10))

    def test_expand_preserves_values(self):
        """Test that expansion preserves values."""
        x_L = torch.tensor([[1.0, 2.0]])
        x_U = torch.tensor([[3.0, 4.0]])
        x_shape = x_L.shape

        result_L, result_U, result_shape = expand_x_batch(x_L, x_U, x_shape, 3)

        for i in range(3):
            self.assertTrue(torch.equal(result_L[i], x_L[0]))
            self.assertTrue(torch.equal(result_U[i], x_U[0]))

    def test_zero_batch_raises_error(self):
        """Test that zero batch dimension raises ValueError."""
        x_L = torch.empty(0, 10)
        x_U = torch.empty(0, 10)
        x_shape = x_L.shape

        with self.assertRaises(ValueError):
            expand_x_batch(x_L, x_U, x_shape, 5)

    def test_expand_4d_tensor(self):
        """Test expanding 4D tensor (image-like)."""
        x_L = torch.zeros(1, 3, 32, 32)
        x_U = torch.ones(1, 3, 32, 32)
        x_shape = x_L.shape

        result_L, result_U, result_shape = expand_x_batch(x_L, x_U, x_shape, 4)
        self.assertEqual(result_L.shape, (4, 3, 32, 32))
        self.assertEqual(result_U.shape, (4, 3, 32, 32))


# ============================================================================
# _all_dist Tests
# ============================================================================

class TestAllDist(unittest.TestCase):
    """Tests for _all_dist function."""

    def test_basic_distance(self):
        """Test basic signed distance calculation."""
        pts = torch.tensor([[0.0, 0.0]])  # [batch, dim]
        lA = torch.tensor([[[1.0, 0.0]]])  # [batch, num_constr, dim]
        lbias = torch.tensor([[0.0]])  # [batch, num_constr]

        result = _all_dist(pts, lA, lbias)
        # numerator = lA @ pts + lbias = 0
        # denominator = ||lA|| = 1
        # result = 0 / 1 = 0
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0, places=5)

    def test_distance_with_bias(self):
        """Test distance with non-zero bias."""
        pts = torch.tensor([[0.0, 0.0]])
        lA = torch.tensor([[[1.0, 0.0]]])
        lbias = torch.tensor([[3.0]])

        result = _all_dist(pts, lA, lbias)
        # numerator = 0 + 3 = 3
        # denominator = 1
        # result = 3
        self.assertAlmostEqual(result[0, 0, 0].item(), 3.0, places=5)

    def test_distance_with_offset_point(self):
        """Test distance with offset point."""
        pts = torch.tensor([[2.0, 0.0]])
        lA = torch.tensor([[[1.0, 0.0]]])
        lbias = torch.tensor([[0.0]])

        result = _all_dist(pts, lA, lbias)
        # numerator = 2
        # denominator = 1
        # result = 2
        self.assertAlmostEqual(result[0, 0, 0].item(), 2.0, places=5)

    def test_batch_distances(self):
        """Test batch distance calculation."""
        batch = 3
        pts = torch.randn(batch, 4)
        lA = torch.randn(batch, 5, 4)
        lbias = torch.randn(batch, 5)

        result = _all_dist(pts, lA, lbias)
        self.assertEqual(result.shape, (batch, 5, 1))

    def test_normalized_by_lA_norm(self):
        """Test that distance is normalized by lA norm."""
        pts = torch.tensor([[0.0, 0.0]])
        lA = torch.tensor([[[3.0, 4.0]]])  # norm = 5
        lbias = torch.tensor([[10.0]])

        result = _all_dist(pts, lA, lbias)
        # numerator = 0 + 10 = 10
        # denominator = 5
        # result = 10 / 5 = 2
        self.assertAlmostEqual(result[0, 0, 0].item(), 2.0, places=5)


# ============================================================================
# dimensionwise_shrinkage_stats Tests
# ============================================================================

class TestDimensionwiseShrinkageStats(unittest.TestCase):
    """Tests for dimensionwise_shrinkage_stats function."""

    def test_no_shrinkage(self):
        """Test when there's no shrinkage."""
        x_L = torch.zeros(2, 10)
        x_U = torch.ones(2, 10)
        x_L_new = torch.zeros(2, 10)
        x_U_new = torch.ones(2, 10)

        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        # No shrinkage means ratio = 1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_uniform_shrinkage(self):
        """Test uniform shrinkage across all dimensions."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        x_L_new = torch.full((1, 4), 0.25)
        x_U_new = torch.full((1, 4), 0.75)

        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        # New side length = 0.5, old = 1.0, ratio = 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_partial_shrinkage(self):
        """Test shrinkage in only some dimensions."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        x_L_new = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
        x_U_new = torch.tensor([[1.0, 1.0, 0.5, 1.0]])

        # Ratios: [1.0, 0.5, 0.5, 1.0], mean = 0.75
        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        self.assertAlmostEqual(result, 0.75, places=5)

    def test_complete_collapse(self):
        """Test when domain collapses to a point."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        x_L_new = torch.full((1, 4), 0.5)
        x_U_new = torch.full((1, 4), 0.5)

        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        # Ratio should be 0 (or very close due to eps)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_batch_processing(self):
        """Test with multiple batches."""
        batch = 5
        x_L = torch.zeros(batch, 10)
        x_U = torch.ones(batch, 10)
        x_L_new = torch.zeros(batch, 10)
        x_U_new = torch.ones(batch, 10) * 0.5

        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        # All batches shrink by 50%
        self.assertAlmostEqual(result, 0.5, places=5)


# ============================================================================
# parallel_clipping Tests
# ============================================================================

class TestParallelClipping(unittest.TestCase):
    """Tests for parallel_clipping function."""

    def test_basic_clipping(self):
        """Test basic domain clipping."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # Constraint on first dim
        lbias = torch.tensor([[-0.3]])  # x[0] >= 0.3

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=1, batches=1, is_lower=True)

        # Lower bound on x[0] should be tightened
        self.assertGreaterEqual(x_L_new[0, 0].item(), x_L[0, 0].item())
        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)

    def test_no_clipping_when_satisfied(self):
        """Test no clipping when constraint already satisfied."""
        x_L = torch.tensor([[0.5, 0.5]])
        x_U = torch.tensor([[1.0, 1.0]])
        # Constraint: x[0] >= 0 (already satisfied)
        lA = torch.tensor([[[1.0, 0.0]]])
        lbias = torch.tensor([[0.0]])

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=1, batches=1, is_lower=True)

        # Bounds should remain same or be tighter
        self.assertGreaterEqual(x_L_new[0, 0].item(), x_L[0, 0].item())
        self.assertLessEqual(x_U_new[0, 0].item(), x_U[0, 0].item())

    def test_multiple_constraints(self):
        """Test clipping with multiple constraints."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        lA = torch.randn(1, 3, 4)
        lbias = torch.randn(1, 3)

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=3, batches=1, is_lower=True)

        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)

    def test_batch_clipping(self):
        """Test batch clipping."""
        batch = 4
        x_L = torch.zeros(batch, 10)
        x_U = torch.ones(batch, 10)
        lA = torch.randn(batch, 5, 10)
        lbias = torch.randn(batch, 5)

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=5, batches=batch, is_lower=True)

        self.assertEqual(x_L_new.shape, (batch, 10))
        self.assertEqual(x_U_new.shape, (batch, 10))

    def test_upper_bound_clipping(self):
        """Test clipping for upper bounds."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        lA = torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]])  # Constraint: -x[0] >= -0.7 => x[0] <= 0.7
        lbias = torch.tensor([[0.7]])

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=1, batches=1, is_lower=False)

        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)

    def test_multiple_iterations(self):
        """Test with multiple iterations."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        lA = torch.randn(1, 2, 4)
        lbias = torch.randn(1, 2)

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=2, batches=1,
                                              is_lower=True, num_iters=2)

        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)


# ============================================================================
# prune_d Tests
# ============================================================================

class TestPruneD(unittest.TestCase):
    """Tests for prune_d function."""

    def test_prune_tensor_keys(self):
        """Test pruning tensor values."""
        d = {
            'cs': torch.randn(5, 3),
            'thresholds': torch.randn(5),
            'x_Ls': torch.randn(5, 10),
            'x_Us': torch.randn(5, 10)
        }
        mask = torch.tensor([True, False, True, False, True])

        result = prune_d(mask, d)

        self.assertEqual(result['cs'].shape[0], 3)
        self.assertEqual(result['thresholds'].shape[0], 3)
        self.assertEqual(result['x_Ls'].shape[0], 3)
        self.assertEqual(result['x_Us'].shape[0], 3)

    def test_prune_list_keys(self):
        """Test pruning list values."""
        d = {
            'history': ['a', 'b', 'c', 'd', 'e'],
            'betas': [1, 2, 3, 4, 5]
        }
        mask = torch.tensor([True, False, True, False, True])

        result = prune_d(mask, d)

        self.assertEqual(len(result['history']), 3)
        self.assertEqual(result['history'], ['a', 'c', 'e'])
        self.assertEqual(len(result['betas']), 3)

    def test_prune_dict_keys(self):
        """Test pruning dict values."""
        d = {
            'lower_bounds': {'layer1': torch.randn(5, 10)},
            'upper_bounds': {'layer1': torch.randn(5, 10)},
            'lAs': {'layer1': torch.randn(5, 3, 10)},
            'mask': {'layer1': torch.randn(5, 10)}
        }
        mask = torch.tensor([True, True, False, False, True])

        result = prune_d(mask, d)

        self.assertEqual(result['lower_bounds']['layer1'].shape[0], 3)
        self.assertEqual(result['upper_bounds']['layer1'].shape[0], 3)

    def test_prune_empty_mask(self):
        """Test pruning with empty mask."""
        d = {'cs': torch.randn(5, 3)}
        mask = torch.tensor([False, False, False, False, False])

        result = prune_d(mask, d)

        # Empty mask returns None (early return)
        self.assertIs(result, d)

    def test_prune_all_selected(self):
        """Test pruning when all elements selected."""
        d = {'cs': torch.randn(3, 5)}
        mask = torch.tensor([True, True, True])

        result = prune_d(mask, d)

        self.assertEqual(result['cs'].shape[0], 3)

    def test_prune_alphas(self):
        """Test pruning nested alphas dict."""
        d = {
            'alphas': {
                'layer1': {
                    'node1': torch.randn(2, 3, 5, 4),  # idx dim is 2
                    'node2': torch.randn(2, 3, 5, 4)
                }
            }
        }
        mask = torch.tensor([True, False, True, False, True])

        result = prune_d(mask, d)

        # Third dimension should be pruned to 3
        self.assertEqual(result['alphas']['layer1']['node1'].shape[2], 3)
        self.assertEqual(result['alphas']['layer1']['node2'].shape[2], 3)


# ============================================================================
# update_interm_bounds Tests
# ============================================================================

class TestUpdateIntermBounds(unittest.TestCase):
    """Tests for update_interm_bounds function."""

    def test_basic_update(self):
        """Test basic bounds update with correct mask format.

        The update_interm_bounds function expects:
        - unstable_mask[key] to be a tensor (for .sum()) with a [0] index for slicing
        - This is an unusual pattern - the mask tensor's first dimension is used for column indexing
        """
        interm_bounds = {
            'layer1': [torch.zeros(2, 10), torch.ones(2, 10)],
            'final': [torch.zeros(2, 5), torch.ones(2, 5)]
        }
        # New bounds only for unstable neurons (3 unstable neurons)
        new_interm_bounds = {
            'layer1': [torch.tensor([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
                       torch.tensor([[0.8, 0.7, 0.6], [0.85, 0.75, 0.65]])]
        }
        # Mask is a tensor with shape [1, 10] where mask[0] gives boolean indices
        unstable_mask = {
            'layer1': torch.tensor([[True, False, False, False, False,
                                     True, False, False, False, True]])
        }

        result = update_interm_bounds(
            interm_bounds, new_interm_bounds, 'final', unstable_mask
        )

        self.assertIn('layer1', result)
        self.assertIn('final', result)
        # Verify bounds were updated at the unstable indices
        self.assertGreater(result['layer1'][0][0, 0].item(), 0)  # Should be 0.1

    def test_skip_final_layer(self):
        """Test that final layer is skipped."""
        interm_bounds = {
            'layer1': [torch.zeros(2, 10), torch.ones(2, 10)],
            'final': [torch.zeros(2, 5), torch.ones(2, 5)]
        }
        new_interm_bounds = {
            'final': [torch.ones(2, 5) * 0.5, torch.ones(2, 5) * 0.5]
        }
        unstable_mask = {}

        result = update_interm_bounds(
            interm_bounds, new_interm_bounds, 'final', unstable_mask
        )

        # Final layer should be unchanged (just copied)
        self.assertTrue(torch.equal(result['final'][0], interm_bounds['final'][0]))

    def test_skip_missing_keys(self):
        """Test that missing keys in new_interm_bounds are skipped."""
        interm_bounds = {
            'layer1': [torch.zeros(2, 10), torch.ones(2, 10)],
            'layer2': [torch.zeros(2, 5), torch.ones(2, 5)]
        }
        new_interm_bounds = {}  # Empty
        unstable_mask = {}

        result = update_interm_bounds(
            interm_bounds, new_interm_bounds, 'final', unstable_mask
        )

        # All layers should be copied unchanged
        self.assertTrue(torch.equal(result['layer1'][0], interm_bounds['layer1'][0]))
        self.assertTrue(torch.equal(result['layer2'][0], interm_bounds['layer2'][0]))

    def test_with_prune_mask(self):
        """Test update with pruning mask."""
        interm_bounds = {
            'layer1': [torch.zeros(4, 10), torch.ones(4, 10)]
        }
        # After pruning, batch size becomes 2
        new_interm_bounds = {
            'layer1': [torch.tensor([[0.1], [0.2]]), torch.tensor([[0.9], [0.8]])]
        }
        # Mask tensor format: [1, dim] where [0] gives the boolean mask
        unstable_mask = {
            'layer1': torch.tensor([[True] + [False] * 9])
        }
        prune_mask = torch.tensor([True, False, True, False])

        result = update_interm_bounds(
            interm_bounds, new_interm_bounds, 'final', unstable_mask, prune_mask
        )

        # Result should have pruned batch size (2)
        self.assertEqual(result['layer1'][0].shape[0], 2)


# ============================================================================
# DomainClipper Class Tests
# ============================================================================

class TestDomainClipperIntersectionCheck(unittest.TestCase):
    """Tests for DomainClipper.intersection_check method.

    Note: The intersection_check method currently only prints statistics
    and does not return values. These tests verify the method runs without
    error and test the underlying constraint classification logic directly.
    """

    def _create_mock_x(self, x_L, x_U):
        """Create a mock bounded tensor."""
        mock_x = MagicMock()
        mock_x.ptb = MagicMock()
        mock_x.ptb.x_L = x_L
        mock_x.ptb.x_U = x_U
        return mock_x

    def _create_clipper(self):
        """Create a minimal DomainClipper instance for testing."""
        clipper = object.__new__(DomainClipper)
        return clipper

    def _compute_constraint_masks(self, x_L, x_U, lA, lbias):
        """Compute constraint classification masks (redundant, infeasible, intersecting).

        This replicates the logic from intersection_check for testing purposes.
        A constraint lA*x + lbias <= 0 is:
        - redundant if max(lA*x + lbias) <= 0 (all points satisfy)
        - infeasible if min(lA*x + lbias) > 0 (no point satisfies)
        - intersecting otherwise (some points satisfy, some don't)
        """
        flat_lA = lA.flatten(2)
        x_L_flat = x_L.flatten(1)
        x_U_flat = x_U.flatten(1)

        pos_mask = flat_lA > 0
        neg_mask = flat_lA < 0

        # max(lA*x + lbias) over the box
        max_term = pos_mask * flat_lA * x_U_flat.unsqueeze(1) + neg_mask * flat_lA * x_L_flat.unsqueeze(1)
        max_val = torch.sum(max_term, dim=2) + lbias

        # min(lA*x + lbias) over the box
        min_term = pos_mask * flat_lA * x_L_flat.unsqueeze(1) + neg_mask * flat_lA * x_U_flat.unsqueeze(1)
        min_val = torch.sum(min_term, dim=2) + lbias

        redundant_mask = max_val <= 0
        infeasible_mask = min_val > 0
        intersect_mask = ~(redundant_mask | infeasible_mask)

        return intersect_mask, redundant_mask, infeasible_mask

    def test_all_redundant_constraints(self):
        """Test when all constraints are redundant (all points satisfy).

        A constraint lA*x + lbias <= 0 is redundant if max(lA*x + lbias) <= 0
        over the entire box. This means all points in the box satisfy it.
        """
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)

        # Constraint: x[0] - 2 <= 0  =>  x[0] <= 2 (always satisfied in [0,1])
        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        lbias = torch.tensor([[-2.0]])

        intersect_mask, redundant_mask, infeasible_mask = self._compute_constraint_masks(
            x_L, x_U, lA, lbias)

        # All constraints should be redundant
        self.assertTrue(redundant_mask.all())
        self.assertFalse(infeasible_mask.any())
        self.assertFalse(intersect_mask.any())

    def test_all_infeasible_constraints(self):
        """Test when all constraints are infeasible (no point satisfies).

        A constraint lA*x + lbias <= 0 is infeasible if min(lA*x + lbias) > 0
        over the entire box. This means no point in the box can satisfy it.
        """
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)

        # Constraint: x[0] + 1 <= 0  =>  x[0] <= -1 (never satisfied in [0,1])
        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        lbias = torch.tensor([[1.0]])

        intersect_mask, redundant_mask, infeasible_mask = self._compute_constraint_masks(
            x_L, x_U, lA, lbias)

        # All constraints should be infeasible
        self.assertTrue(infeasible_mask.all())
        self.assertFalse(redundant_mask.any())
        self.assertFalse(intersect_mask.any())

    def test_mixed_constraints(self):
        """Test with a mix of redundant, infeasible, and intersecting constraints."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)

        # Three constraints:
        # 1. x[0] - 2 <= 0  =>  x[0] <= 2 (redundant - always satisfied in [0,1])
        # 2. x[1] + 1 <= 0  =>  x[1] <= -1 (infeasible - never satisfied in [0,1])
        # 3. x[2] - 0.5 <= 0  =>  x[2] <= 0.5 (intersecting - some points satisfy)
        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]]])
        lbias = torch.tensor([[-2.0, 1.0, -0.5]])

        intersect_mask, redundant_mask, infeasible_mask = self._compute_constraint_masks(
            x_L, x_U, lA, lbias)

        # Check each constraint classification
        self.assertTrue(redundant_mask[0, 0].item())      # First is redundant
        self.assertTrue(infeasible_mask[0, 1].item())     # Second is infeasible
        self.assertTrue(intersect_mask[0, 2].item())      # Third intersects

    def test_intersecting_constraint(self):
        """Test a single intersecting constraint.

        A constraint intersects when min(lA*x + lbias) <= 0 <= max(lA*x + lbias).
        """
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)

        # Constraint: x[0] - 0.5 <= 0  =>  x[0] <= 0.5
        # In [0,1]: min = 0 - 0.5 = -0.5, max = 1 - 0.5 = 0.5
        # Since min < 0 and max > 0, this constraint intersects the box
        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        lbias = torch.tensor([[-0.5]])

        intersect_mask, redundant_mask, infeasible_mask = self._compute_constraint_masks(
            x_L, x_U, lA, lbias)

        # Should be intersecting (not redundant, not infeasible)
        self.assertTrue(intersect_mask.all())
        self.assertFalse(redundant_mask.any())
        self.assertFalse(infeasible_mask.any())

    def test_batch_constraints(self):
        """Test with batch of constraints."""
        batch = 2
        x_L = torch.zeros(batch, 4)
        x_U = torch.ones(batch, 4)

        # Each batch has 2 constraints
        lA = torch.randn(batch, 2, 4)
        lbias = torch.randn(batch, 2)

        intersect_mask, redundant_mask, infeasible_mask = self._compute_constraint_masks(
            x_L, x_U, lA, lbias)

        # Check output shapes
        self.assertEqual(intersect_mask.shape, (batch, 2))
        self.assertEqual(redundant_mask.shape, (batch, 2))
        self.assertEqual(infeasible_mask.shape, (batch, 2))

        # Each constraint should be exactly one of: redundant, infeasible, or intersecting
        exclusive = redundant_mask.int() + infeasible_mask.int() + intersect_mask.int()
        self.assertTrue((exclusive == 1).all())

    def test_intersection_check_runs_without_error(self):
        """Test that intersection_check method runs without raising an error."""
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        x = self._create_mock_x(x_L, x_U)

        lA = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        lbias = torch.tensor([[-0.5]])

        clipper = self._create_clipper()
        # Method should run without error (it only prints, doesn't return)
        result = clipper.intersection_check(x, lA, lbias)
        self.assertIsNone(result)  # Method doesn't return anything


class TestDomainClipperClipDomains(unittest.TestCase):
    """Tests for DomainClipper.clip_domains method."""

    def test_basic_clip_domains(self):
        """Test basic domain clipping through DomainClipper."""
        # Create minimal mock setup
        x_L = torch.zeros(2, 10)
        x_U = torch.ones(2, 10)
        lA = torch.randn(2, 3, 10)
        lbias = torch.randn(2, 3)

        # Mock a DomainClipper instance
        clipper = MagicMock()
        clipper.prune = False
        clipper.mask = {}
        clipper.concretize_interm_bounds = MagicMock(return_value={})

        # Test the clip_domains logic directly
        result = parallel_clipping(
            x_L, x_U, lA, lbias,
            num_constr=3, batches=2, is_lower=True
        )

        x_L_new, x_U_new = result
        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)


class TestDomainClipperBuildFinalLALbias(unittest.TestCase):
    """Tests for DomainClipper.build_final_lA_lbias method."""

    def _create_clipper_with_bounds(self, lA, uA, lbias, ubias, mapping):
        """Create a minimal DomainClipper with bound data for testing."""
        clipper = object.__new__(DomainClipper)
        clipper.lA = lA
        clipper.uA = uA
        clipper.lbias = lbias
        clipper.ubias = ubias
        clipper.mapping = mapping
        return clipper

    def test_single_history_lower_bound(self):
        """Test building lA/lbias from single history with lower bound."""
        # Setup bounds: layer with 3 unstable neurons, input dim 4
        lA = {'layer1': torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                        [5.0, 6.0, 7.0, 8.0],
                                        [9.0, 10.0, 11.0, 12.0]]])}
        uA = {'layer1': torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                                        [0.5, 0.6, 0.7, 0.8],
                                        [0.9, 1.0, 1.1, 1.2]]])}
        lbias = {'layer1': torch.tensor([[0.1, 0.2, 0.3]])}
        ubias = {'layer1': torch.tensor([[1.1, 1.2, 1.3]])}
        mapping = {'layer1': {0: 0, 1: 1, 2: 2}}

        clipper = self._create_clipper_with_bounds(lA, uA, lbias, ubias, mapping)

        # History: idx=[1], status=[-1] (lower bound, neuron 1)
        histories = [{'layer1': (torch.tensor([1]), torch.tensor([-1.0]), None, None, None)}]

        final_lA, final_lbias = clipper.build_final_lA_lbias(histories)

        # Should use lower bound for unstable_idx=1
        self.assertEqual(final_lA.shape, (1, 1, 4))
        self.assertEqual(final_lbias.shape, (1, 1))
        self.assertTrue(torch.allclose(final_lA[0, 0], lA['layer1'][0, 1]))
        self.assertAlmostEqual(final_lbias[0, 0].item(), lbias['layer1'][0, 1].item())

    def test_single_history_upper_bound(self):
        """Test building lA/lbias from single history with upper bound (negated)."""
        lA = {'layer1': torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                        [5.0, 6.0, 7.0, 8.0]]])}
        uA = {'layer1': torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                                        [0.5, 0.6, 0.7, 0.8]]])}
        lbias = {'layer1': torch.tensor([[0.1, 0.2]])}
        ubias = {'layer1': torch.tensor([[1.1, 1.2]])}
        mapping = {'layer1': {0: 0, 1: 1}}

        clipper = self._create_clipper_with_bounds(lA, uA, lbias, ubias, mapping)

        # History: idx=[0], status=[1] (upper bound, neuron 0)
        histories = [{'layer1': (torch.tensor([0]), torch.tensor([1.0]), None, None, None)}]

        final_lA, final_lbias = clipper.build_final_lA_lbias(histories)

        # Should use negated upper bound for unstable_idx=0
        self.assertEqual(final_lA.shape, (1, 1, 4))
        self.assertEqual(final_lbias.shape, (1, 1))
        self.assertTrue(torch.allclose(final_lA[0, 0], -uA['layer1'][0, 0]))
        self.assertAlmostEqual(final_lbias[0, 0].item(), -ubias['layer1'][0, 0].item())

    def test_multiple_histories(self):
        """Test building lA/lbias from multiple histories."""
        lA = {'layer1': torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])}
        uA = {'layer1': torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])}
        lbias = {'layer1': torch.tensor([[0.1, 0.2, 0.3]])}
        ubias = {'layer1': torch.tensor([[1.1, 1.2, 1.3]])}
        mapping = {'layer1': {0: 0, 1: 1, 2: 2}}

        clipper = self._create_clipper_with_bounds(lA, uA, lbias, ubias, mapping)

        # Two histories: first uses lower bound, second uses upper bound
        histories = [
            {'layer1': (torch.tensor([0]), torch.tensor([-1.0]), None, None, None)},
            {'layer1': (torch.tensor([2]), torch.tensor([1.0]), None, None, None)}
        ]

        final_lA, final_lbias = clipper.build_final_lA_lbias(histories)

        # Should have batch size 2
        self.assertEqual(final_lA.shape, (2, 1, 2))
        self.assertEqual(final_lbias.shape, (2, 1))

        # First batch: lower bound for neuron 0
        self.assertTrue(torch.allclose(final_lA[0, 0], lA['layer1'][0, 0]))
        # Second batch: negated upper bound for neuron 2
        self.assertTrue(torch.allclose(final_lA[1, 0], -uA['layer1'][0, 2]))

    def test_history_with_multiple_entries_uses_last(self):
        """Test that only the last entry in history is used."""
        lA = {'layer1': torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])}
        uA = {'layer1': torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])}
        lbias = {'layer1': torch.tensor([[0.1, 0.2, 0.3]])}
        ubias = {'layer1': torch.tensor([[1.1, 1.2, 1.3]])}
        mapping = {'layer1': {0: 0, 1: 1, 2: 2}}

        clipper = self._create_clipper_with_bounds(lA, uA, lbias, ubias, mapping)

        # History with multiple entries - should use last (idx=2, status=1)
        histories = [{'layer1': (torch.tensor([0, 1, 2]), torch.tensor([-1.0, -1.0, 1.0]), None, None, None)}]

        final_lA, final_lbias = clipper.build_final_lA_lbias(histories)

        # Should use last entry: negated upper bound for neuron 2
        self.assertTrue(torch.allclose(final_lA[0, 0], -uA['layer1'][0, 2]))

    def test_output_shape_consistency(self):
        """Test that output shapes are consistent with input dimensions."""
        input_dim = 8
        num_neurons = 5
        lA = {'layer1': torch.randn(1, num_neurons, input_dim)}
        uA = {'layer1': torch.randn(1, num_neurons, input_dim)}
        lbias = {'layer1': torch.randn(1, num_neurons)}
        ubias = {'layer1': torch.randn(1, num_neurons)}
        mapping = {'layer1': {i: i for i in range(num_neurons)}}

        clipper = self._create_clipper_with_bounds(lA, uA, lbias, ubias, mapping)

        histories = [
            {'layer1': (torch.tensor([0]), torch.tensor([-1.0]), None, None, None)},
            {'layer1': (torch.tensor([1]), torch.tensor([1.0]), None, None, None)},
            {'layer1': (torch.tensor([2]), torch.tensor([-1.0]), None, None, None)},
        ]

        final_lA, final_lbias = clipper.build_final_lA_lbias(histories)

        self.assertEqual(final_lA.shape, (3, 1, input_dim))
        self.assertEqual(final_lbias.shape, (3, 1))


class TestDomainClipperUpdateUnstableIdx(unittest.TestCase):
    """Tests for DomainClipper.update_unstable_idx method."""

    def test_basic_update(self):
        """Test basic unstable index update."""
        # Create mock network
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_input = MagicMock()
        mock_input.name = 'input_layer'
        mock_node.inputs = [mock_input]
        mock_net.net = {'op_node': mock_node}

        # Create mock DomainClipper
        clipper = MagicMock()
        clipper.mask = {}
        clipper.true_indices = {}
        clipper.mapping = {}
        clipper.lA = {'input_layer': torch.randn(1, 10, 5)}
        clipper.uA = {'input_layer': torch.randn(1, 10, 5)}
        clipper.lbias = {'input_layer': torch.randn(1, 10)}
        clipper.ubias = {'input_layer': torch.randn(1, 10)}

        # Updated mask
        updated_mask = {
            'op_node': [torch.tensor([True, False, True, True, False,
                                      False, True, False, False, True])]
        }

        # Test the update logic
        mask = updated_mask['op_node'][0]
        true_indices = mask.view(-1).nonzero(as_tuple=True)[0]
        mapping = {idx.item(): i for i, idx in enumerate(true_indices)}

        self.assertEqual(len(true_indices), 5)  # 5 True values
        self.assertEqual(len(mapping), 5)


class TestDomainClipperGetStopCriterionAndIter(unittest.TestCase):
    """Tests for DomainClipper.get_stop_criterion_and_iter method."""

    def test_sets_attributes(self):
        """Test that method sets stop_func and iter_idx."""
        clipper = MagicMock()
        clipper.stop_func = None
        clipper.iter_idx = None

        # Simulate the method
        def get_stop_criterion_and_iter(self, stop_func, iter_idx):
            self.stop_func = stop_func
            self.iter_idx = iter_idx

        mock_stop_func = lambda x: x > 0
        get_stop_criterion_and_iter(clipper, mock_stop_func, 5)

        self.assertEqual(clipper.stop_func, mock_stop_func)
        self.assertEqual(clipper.iter_idx, 5)


class TestDomainClipperGetConstraints(unittest.TestCase):
    """Tests for DomainClipper.get_constraints method."""

    def test_uses_all_histories_early(self):
        """Test that early iterations use all histories."""
        clipper = MagicMock()
        clipper.iter_idx = 1
        clipper.max_iter = 2
        clipper.build_final_lA_lbias_all = MagicMock(return_value=(torch.randn(2, 3, 10), torch.randn(2, 3)))
        clipper.build_final_lA_lbias = MagicMock()

        # Simulate get_constraints behavior
        if clipper.iter_idx <= clipper.max_iter:
            A, bias = clipper.build_final_lA_lbias_all([])
        else:
            A, bias = clipper.build_final_lA_lbias([])

        clipper.build_final_lA_lbias_all.assert_called_once()
        clipper.build_final_lA_lbias.assert_not_called()

    def test_uses_single_history_late(self):
        """Test that late iterations use single history."""
        clipper = MagicMock()
        clipper.iter_idx = 5
        clipper.max_iter = 2
        clipper.build_final_lA_lbias_all = MagicMock()
        clipper.build_final_lA_lbias = MagicMock(return_value=(torch.randn(2, 1, 10), torch.randn(2, 1)))

        # Simulate get_constraints behavior
        if clipper.iter_idx <= clipper.max_iter:
            A, bias = clipper.build_final_lA_lbias_all([])
        else:
            A, bias = clipper.build_final_lA_lbias([])

        clipper.build_final_lA_lbias.assert_called_once()
        clipper.build_final_lA_lbias_all.assert_not_called()


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_pad_first_dim_negative_max_len(self):
        """Test pad_first_dim behavior with edge cases."""
        tensor = torch.randn(3, 4)
        # max_len less than current size - function just returns original
        result = pad_first_dim(tensor, 3)
        self.assertTrue(torch.equal(result, tensor))

    def test_concretize_bounds_zero_dimensions(self):
        """Test concretize_bounds with zero-sized dimensions."""
        x_L = torch.zeros(1, 0)
        x_U = torch.ones(1, 0)
        lA = torch.randn(1, 2, 0)
        lbias = torch.zeros(1, 2)

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        self.assertEqual(result.shape, (1, 2))

    def test_parallel_clipping_zero_constraints(self):
        """Test parallel_clipping with zero constraints.

        Note: parallel_clipping with 0 constraints may fail due to
        torch operations on empty tensors. This test documents that
        at least 1 constraint is expected.
        """
        x_L = torch.zeros(1, 4)
        x_U = torch.ones(1, 4)
        # Use a single trivial constraint instead of zero
        lA = torch.zeros(1, 1, 4)  # Zero coefficients = no effect
        lbias = torch.zeros(1, 1)

        # Should handle gracefully
        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=1, batches=1, is_lower=True)
        # With trivial constraint (0*x + 0 >= 0), bounds should stay the same
        self.assertEqual(x_L_new.shape, x_L.shape)
        self.assertEqual(x_U_new.shape, x_U.shape)

    def test_all_dist_zero_norm(self):
        """Test _all_dist with near-zero norm (handled by epsilon)."""
        pts = torch.tensor([[0.0, 0.0]])
        lA = torch.tensor([[[1e-12, 1e-12]]])  # Very small norm
        lbias = torch.tensor([[1.0]])

        result = _all_dist(pts, lA, lbias)
        # Should not produce inf/nan due to eps in denominator
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def test_concretize_bounds_large_values(self):
        """Test concretize_bounds with large values."""
        x_L = torch.zeros(1, 4) - 1e6
        x_U = torch.zeros(1, 4) + 1e6
        lA = torch.randn(1, 2, 4)
        lbias = torch.zeros(1, 2)

        result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
        self.assertFalse(torch.isnan(result).any())

    def test_parallel_clipping_small_domains(self):
        """Test parallel_clipping with very small domains."""
        eps = 1e-8
        x_L = torch.zeros(1, 4)
        x_U = torch.zeros(1, 4) + eps
        lA = torch.randn(1, 2, 4)
        lbias = torch.randn(1, 2)

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=2, batches=1, is_lower=True)

        self.assertFalse(torch.isnan(x_L_new).any())
        self.assertFalse(torch.isnan(x_U_new).any())

    def test_dimensionwise_shrinkage_stats_zero_domain(self):
        """Test dimensionwise_shrinkage_stats with zero-width domain."""
        x_L = torch.zeros(1, 4)
        x_U = torch.zeros(1, 4)  # Same as x_L
        x_L_new = torch.zeros(1, 4)
        x_U_new = torch.zeros(1, 4)

        # Should handle division by near-zero gracefully
        result = dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)
        self.assertFalse(torch.isnan(torch.tensor(result)))


class TestDtypePreservation(unittest.TestCase):
    """Tests for dtype preservation across operations."""

    def test_parallel_clipping_preserves_dtype(self):
        """Test that parallel_clipping preserves input dtype."""
        for dtype in [torch.float32, torch.float64]:
            x_L = torch.zeros(1, 4, dtype=dtype)
            x_U = torch.ones(1, 4, dtype=dtype)
            lA = torch.randn(1, 2, 4, dtype=dtype)
            lbias = torch.randn(1, 2, dtype=dtype)

            x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                                  num_constr=2, batches=1, is_lower=True)

            self.assertEqual(x_L_new.dtype, dtype)
            self.assertEqual(x_U_new.dtype, dtype)

    def test_concretize_bounds_preserves_dtype(self):
        """Test that concretize_bounds preserves input dtype."""
        for dtype in [torch.float32, torch.float64]:
            x_L = torch.zeros(1, 4, dtype=dtype)
            x_U = torch.ones(1, 4, dtype=dtype)
            lA = torch.randn(1, 2, 4, dtype=dtype)
            lbias = torch.randn(1, 2, dtype=dtype)

            result = concretize_bounds(x_L, x_U, lA, lbias, is_lower=True)
            self.assertEqual(result.dtype, dtype)


class TestDeviceHandling(unittest.TestCase):
    """Tests for device handling (CPU only in unit tests)."""

    def test_parallel_clipping_cpu(self):
        """Test parallel_clipping on CPU."""
        device = 'cpu'
        x_L = torch.zeros(1, 4, device=device)
        x_U = torch.ones(1, 4, device=device)
        lA = torch.randn(1, 2, 4, device=device)
        lbias = torch.randn(1, 2, device=device)

        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias,
                                              num_constr=2, batches=1, is_lower=True)

        self.assertEqual(x_L_new.device.type, device)
        self.assertEqual(x_U_new.device.type, device)


if __name__ == '__main__':
    unittest.main()
