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
"""Unit tests for input_split/branching_heuristics.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_module():
    """Setup arguments.Config for all tests."""
    import arguments
    global original_config
    original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestInputSplitBranchingNaive(unittest.TestCase):
    """Tests for naive branching method."""

    def test_naive_method_returns_correct_shape(self):
        """Test that naive method returns correct output shape."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 4
        input_dim = 10
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.randn(batch_size, 5)
        thresholds = torch.zeros(batch_size, 5)
        lA = torch.randn(batch_size, 5, input_dim)
        split_depth = 2

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=split_depth
        )

        self.assertEqual(result.shape, (batch_size, split_depth))

    def test_naive_method_selects_largest_edges(self):
        """Test that naive method selects dimensions with largest x_U - x_L."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 5
        x_L = torch.zeros(batch_size, input_dim)
        # Make dimensions 3 and 1 have the largest perturbations
        x_U = torch.tensor([
            [0.1, 0.5, 0.2, 0.9, 0.3],
            [0.2, 0.8, 0.1, 0.7, 0.4]
        ])
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)
        split_depth = 2

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=split_depth
        )

        # For first batch: largest edges at indices 3 (0.9) and 1 (0.5)
        self.assertIn(3, result[0].tolist())
        self.assertIn(1, result[0].tolist())
        # For second batch: largest edges at indices 1 (0.8) and 3 (0.7)
        self.assertIn(1, result[1].tolist())
        self.assertIn(3, result[1].tolist())

    def test_naive_method_single_split(self):
        """Test naive method with single split."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 3
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.tensor([
            [0.1, 0.5, 0.2, 0.9],
            [0.3, 0.1, 0.8, 0.2],
            [0.7, 0.2, 0.3, 0.4]
        ])
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        self.assertEqual(result.shape, (batch_size, 1))
        self.assertEqual(result[0].item(), 3)  # Index 3 has largest edge 0.9
        self.assertEqual(result[1].item(), 2)  # Index 2 has largest edge 0.8
        self.assertEqual(result[2].item(), 0)  # Index 0 has largest edge 0.7


class TestInputSplitBranchingSb(unittest.TestCase):
    """Tests for smart branching (sb) method."""

    def test_sb_method_returns_correct_shape(self):
        """Test that sb method returns correct output shape."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        # Set config values
        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.001
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 4
        input_dim = 10
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.randn(batch_size, 5)
        thresholds = torch.zeros(batch_size, 5)
        lA = torch.randn(batch_size, 5, input_dim)
        split_depth = 3

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='sb', split_depth=split_depth
        )

        self.assertEqual(result.shape, (batch_size, split_depth))

    def test_sb_method_uses_lA_for_scoring(self):
        """Test that sb method uses lA coefficients for scoring."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)  # Same perturbation for all
        dom_lb = torch.zeros(batch_size, 1)
        thresholds = torch.zeros(batch_size, 1)
        # Make lA have large value at index 2
        lA = torch.zeros(batch_size, 1, input_dim)
        lA[:, :, 2] = 10.0

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='sb', split_depth=1
        )

        # Index 2 should be selected due to large lA coefficient
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 2)


class TestInputSplitHeuristicSb(unittest.TestCase):
    """Direct tests for input_split_heuristic_sb function."""

    def test_sb_heuristic_basic(self):
        """Test basic sb heuristic functionality."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 5
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 3)
        thresholds = torch.zeros(batch_size, 3)
        lA = torch.randn(batch_size, 3, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=2)

        self.assertEqual(result.shape, (batch_size, 2))

    def test_sb_heuristic_with_sb_sum_true(self):
        """Test sb heuristic with sb_sum=True."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        # lA has largest absolute values at index 0
        lA = torch.tensor([
            [[10.0, 1.0, 2.0, 3.0], [5.0, 1.0, 1.0, 1.0]],
            [[8.0, 2.0, 1.0, 1.0], [4.0, 1.0, 1.0, 1.0]]
        ]).float()

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        # Index 0 should be selected due to largest sum of abs(lA)
        self.assertEqual(result[0].item(), 0)
        self.assertEqual(result[1].item(), 0)

    def test_sb_heuristic_with_sb_sum_false(self):
        """Test sb heuristic with sb_sum=False (uses margin weight)."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 1.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = False
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.ones(batch_size, 2, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        self.assertEqual(result.shape, (batch_size, 1))

    def test_sb_heuristic_with_primary_spec(self):
        """Test sb heuristic with sb_primary_spec set."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = False
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = 1
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        num_specs = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, num_specs)
        thresholds = torch.zeros(batch_size, num_specs)
        # Make spec 1 have highest score at index 2
        lA = torch.zeros(batch_size, num_specs, input_dim)
        lA[:, 0, 0] = 10.0  # Spec 0 highest at index 0
        lA[:, 1, 2] = 15.0  # Spec 1 highest at index 2
        lA[:, 2, 3] = 20.0  # Spec 2 highest at index 3

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        # With sb_primary_spec=1, should use spec 1 which has highest at index 2
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 2)

    def test_sb_heuristic_with_touch_zero_score(self):
        """Test sb heuristic with touch_zero_score bonus."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 100.0

        batch_size = 2
        input_dim = 4
        # x_L touches zero at index 1, x_U touches zero at index 2
        x_L = torch.tensor([[0.0, 0.0, -0.5, -0.5], [0.0, 0.0, -0.5, -0.5]])
        x_U = torch.tensor([[0.5, 0.5, 0.0, 0.5], [0.5, 0.5, 0.0, 0.5]])
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        # Small lA values so touch_zero_score dominates
        lA = torch.ones(batch_size, 2, input_dim) * 0.01

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        # Touch zero at indices 0, 1, 2 - should select one of them
        self.assertIn(result[0].item(), [0, 1, 2])

    def test_sb_heuristic_with_coeff_thresh(self):
        """Test sb heuristic with lA clamping threshold."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 1.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        # Different perturbations to differentiate scores
        x_U = torch.tensor([[0.1, 0.5, 0.2, 0.9], [0.1, 0.5, 0.2, 0.9]])
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        # Small lA values that will be clamped to 1.0
        lA = torch.ones(batch_size, 2, input_dim) * 0.01

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        # With clamped lA, perturbation size determines score - index 3 has largest
        self.assertEqual(result[0].item(), 3)


class TestInputSplitBranchingBruteForce(unittest.TestCase):
    """Tests for brute-force branching method."""

    def _create_mock_net(self, batch_size, num_specs, output_dim, input_dim):
        """Create a mock network for brute-force testing."""
        mock_net = MagicMock()
        mock_net.c = torch.randn(batch_size, num_specs, output_dim)

        # Mock the net.net.compute_bounds method
        def mock_compute_bounds(x, C, method, bound_upper=True, reference_bounds=None):
            batch = x[0].shape[0]
            return (torch.randn(batch, num_specs),)

        mock_net.net = MagicMock()
        mock_net.net.compute_bounds = mock_compute_bounds

        # Mock nodes for reference_interm_bounds
        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_net.net.nodes = MagicMock(return_value=[mock_node])

        return mock_net

    def test_brute_force_assertion_split_depth(self):
        """Test that brute-force requires split_depth == 1."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['bf_iters'] = 10

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        with self.assertRaises(AssertionError):
            input_split_branching(
                None, dom_lb, x_L, x_U, lA, thresholds,
                branching_method='brute-force', split_depth=2, num_iter=0
            )

    def test_brute_force_falls_back_to_sb_after_bf_iters(self):
        """Test that brute-force falls back to sb after bf_iters."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['bf_iters'] = 5
        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        # num_iter > bf_iters should fall back to sb
        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='brute-force', split_depth=1, num_iter=10
        )

        self.assertEqual(result.shape, (batch_size, 1))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bf heuristic (uses .cuda() internally)")
class TestInputSplitHeuristicBf(unittest.TestCase):
    """Direct tests for input_split_heuristic_bf function.

    Note: These tests require CUDA because the bf heuristic has a hardcoded
    .cuda() call in the source code (line 138 in branching_heuristics.py).
    """

    def _create_mock_net(self, batch_size, num_specs, output_dim, input_dim, lb_values=None, device='cuda'):
        """Create a mock network for brute-force testing."""
        mock_net = MagicMock()
        mock_net.c = torch.randn(1, num_specs, output_dim, device=device)

        # Mock the net.net.compute_bounds method
        def mock_compute_bounds(x, C, method, bound_upper=True, reference_bounds=None):
            expanded_batch = x[0].shape[0]
            if lb_values is not None:
                return (lb_values.expand(expanded_batch, -1),)
            return (torch.randn(expanded_batch, num_specs, device=device),)

        mock_net.net = MagicMock()
        mock_net.net.compute_bounds = mock_compute_bounds

        # Mock nodes for reference_interm_bounds
        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5, device=device)
        mock_node.upper = torch.ones(batch_size, 5, device=device)
        mock_node.name = 'node1'
        mock_net.net.nodes = MagicMock(return_value=[mock_node])

        return mock_net

    def test_bf_heuristic_returns_indices(self):
        """Test that bf heuristic returns valid indices."""
        from input_split.branching_heuristics import input_split_heuristic_bf
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_backup_thresh'] = 0.1
        arguments.Config['bab']['branching']['input_split']['bf_rhs_offset'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_zero_crossing_score'] = False
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4
        device = 'cuda'
        x_L = torch.zeros(batch_size, input_dim, device=device)
        x_U = torch.ones(batch_size, input_dim, device=device)
        dom_lb = torch.tensor([[-0.5, -0.3], [-0.4, -0.2]], device=device)
        thresholds = torch.zeros(1, num_specs, device=device)
        lA = torch.randn(batch_size, num_specs, input_dim, device=device)

        mock_net = self._create_mock_net(batch_size, num_specs, output_dim, input_dim, device=device)

        result = input_split_heuristic_bf(
            mock_net, x_L, x_U, dom_lb, thresholds, lA
        )

        self.assertEqual(result.shape, (batch_size, 1))
        # All indices should be valid (0 to input_dim-1)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < input_dim).all())

    def test_bf_heuristic_with_zero_crossing_score(self):
        """Test bf heuristic with zero crossing score enabled."""
        from input_split.branching_heuristics import input_split_heuristic_bf
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_backup_thresh'] = 0.1
        arguments.Config['bab']['branching']['input_split']['bf_rhs_offset'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_zero_crossing_score'] = True
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4
        device = 'cuda'
        # Index 1 crosses zero (x_L < 0 and x_U > 0)
        x_L = torch.tensor([[0.1, -0.5, 0.2], [0.1, -0.3, 0.2]], device=device)
        x_U = torch.tensor([[0.5, 0.5, 0.6], [0.5, 0.3, 0.6]], device=device)
        dom_lb = torch.tensor([[-0.5, -0.3], [-0.4, -0.2]], device=device)
        thresholds = torch.zeros(1, num_specs, device=device)
        lA = torch.randn(batch_size, num_specs, input_dim, device=device)

        mock_net = self._create_mock_net(batch_size, num_specs, output_dim, input_dim, device=device)

        result = input_split_heuristic_bf(
            mock_net, x_L, x_U, dom_lb, thresholds, lA
        )

        self.assertEqual(result.shape, (batch_size, 1))

    def test_bf_heuristic_with_touch_zero_score(self):
        """Test bf heuristic with touch zero score enabled."""
        from input_split.branching_heuristics import input_split_heuristic_bf
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_backup_thresh'] = 0.1
        arguments.Config['bab']['branching']['input_split']['bf_rhs_offset'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_zero_crossing_score'] = False
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 100.0

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4
        device = 'cuda'
        # Index 0 touches zero at lower bound, index 2 touches zero at upper bound
        x_L = torch.tensor([[0.0, 0.1, -0.5], [0.0, 0.1, -0.5]], device=device)
        x_U = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], device=device)
        dom_lb = torch.tensor([[-0.5, -0.3], [-0.4, -0.2]], device=device)
        thresholds = torch.zeros(1, num_specs, device=device)
        lA = torch.randn(batch_size, num_specs, input_dim, device=device)

        mock_net = self._create_mock_net(batch_size, num_specs, output_dim, input_dim, device=device)

        result = input_split_heuristic_bf(
            mock_net, x_L, x_U, dom_lb, thresholds, lA
        )

        self.assertEqual(result.shape, (batch_size, 1))


class TestInputSplitBranchingUnsupportedMethod(unittest.TestCase):
    """Tests for unsupported branching methods."""

    def test_unsupported_method_raises_error(self):
        """Test that unsupported method raises NameError."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        with self.assertRaises(NameError) as context:
            input_split_branching(
                None, dom_lb, x_L, x_U, lA, thresholds,
                branching_method='invalid_method', split_depth=1
            )

        self.assertIn('Unsupported branching method', str(context.exception))
        self.assertIn('invalid_method', str(context.exception))


class TestInputSplitBranchingInputFlattening(unittest.TestCase):
    """Tests for input flattening in branching methods."""

    def test_3d_input_flattened(self):
        """Test that 3D inputs are properly flattened."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        # 3D input shape: (batch, channels, width)
        x_L = torch.zeros(batch_size, 3, 4)  # Flattened = 12
        x_U = torch.ones(batch_size, 3, 4)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, 3, 4)  # Will be flattened to (batch, 2, 12)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        self.assertEqual(result.shape, (batch_size, 1))
        # Index should be in range [0, 11] for flattened input
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < 12).all())

    def test_4d_input_flattened(self):
        """Test that 4D inputs are properly flattened."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        # 4D input shape: (batch, channels, height, width)
        x_L = torch.zeros(batch_size, 1, 3, 3)  # Flattened = 9
        x_U = torch.ones(batch_size, 1, 3, 3)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, 1, 3, 3)  # Will be flattened to (batch, 2, 9)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        self.assertEqual(result.shape, (batch_size, 1))
        # Index should be in range [0, 8] for flattened input
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < 9).all())


class TestInputSplitBranchingSplitDepth(unittest.TestCase):
    """Tests for different split depths."""

    def test_split_depth_equals_input_dim(self):
        """Test when split_depth equals input dimension."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=input_dim
        )

        self.assertEqual(result.shape, (batch_size, input_dim))
        # Should return all indices in some order
        for i in range(batch_size):
            self.assertEqual(set(result[i].tolist()), set(range(input_dim)))

    def test_split_depth_one(self):
        """Test with split_depth = 1."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 3
        input_dim = 10
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        self.assertEqual(result.shape, (batch_size, 1))


class TestInputSplitBranchingDeviceHandling(unittest.TestCase):
    """Tests for device handling."""

    def test_cpu_tensors(self):
        """Test with CPU tensors."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim, device='cpu')
        x_U = torch.ones(batch_size, input_dim, device='cpu')
        dom_lb = torch.zeros(batch_size, 2, device='cpu')
        thresholds = torch.zeros(batch_size, 2, device='cpu')
        lA = torch.randn(batch_size, 2, input_dim, device='cpu')

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='sb', split_depth=1
        )

        self.assertEqual(result.device.type, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tensors(self):
        """Test with CUDA tensors."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim, device='cuda')
        x_U = torch.ones(batch_size, input_dim, device='cuda')
        dom_lb = torch.zeros(batch_size, 2, device='cuda')
        thresholds = torch.zeros(batch_size, 2, device='cuda')
        lA = torch.randn(batch_size, 2, input_dim, device='cuda')

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='sb', split_depth=1
        )

        self.assertEqual(result.device.type, 'cuda')


class TestInputSplitBranchingBatchSizes(unittest.TestCase):
    """Tests for various batch sizes."""

    def test_single_batch(self):
        """Test with batch size of 1."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 1
        input_dim = 5
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=2
        )

        self.assertEqual(result.shape, (1, 2))

    def test_large_batch(self):
        """Test with large batch size."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 100
        input_dim = 10
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=3
        )

        self.assertEqual(result.shape, (batch_size, 3))


class TestInputSplitHeuristicSbEdgeCases(unittest.TestCase):
    """Edge case tests for sb heuristic."""

    def test_sb_with_all_zero_lA(self):
        """Test sb heuristic when lA is all zeros."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.zeros(batch_size, 2, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        self.assertEqual(result.shape, (batch_size, 1))

    def test_sb_with_negative_dom_lb(self):
        """Test sb heuristic with negative domain lower bounds."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 1.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = False
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.ones(batch_size, 2, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        self.assertEqual(result.shape, (batch_size, 1))

    def test_sb_with_positive_thresholds(self):
        """Test sb heuristic with positive threshold values."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 1.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = False
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.tensor([[0.5, 1.0], [0.3, 0.7]])
        lA = torch.ones(batch_size, 2, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1)

        self.assertEqual(result.shape, (batch_size, 1))

    def test_sb_with_single_spec(self):
        """Test sb heuristic with single specification."""
        from input_split.branching_heuristics import input_split_heuristic_sb
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 3
        input_dim = 5
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 1)
        thresholds = torch.zeros(batch_size, 1)
        lA = torch.randn(batch_size, 1, input_dim)

        result = input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=2)

        self.assertEqual(result.shape, (batch_size, 2))


class TestInputSplitBranchingNoGrad(unittest.TestCase):
    """Tests for torch.no_grad decorator."""

    def test_no_grad_context(self):
        """Test that function runs without gradient computation."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim, requires_grad=True)
        x_U = torch.ones(batch_size, input_dim, requires_grad=True)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim, requires_grad=True)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        # Result should not require gradients
        self.assertFalse(result.requires_grad)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bf heuristic (uses .cuda() internally)")
class TestInputSplitHeuristicBfBackup(unittest.TestCase):
    """Tests for bf heuristic backup behavior.

    Note: These tests require CUDA because the bf heuristic has a hardcoded
    .cuda() call in the source code (line 138 in branching_heuristics.py).
    """

    def _create_mock_net_with_bad_bounds(self, batch_size, num_specs, output_dim, input_dim, device='cuda'):
        """Create mock net that returns bounds triggering backup threshold."""
        mock_net = MagicMock()
        mock_net.c = torch.randn(1, num_specs, output_dim, device=device)

        # Return very negative bounds to trigger backup
        def mock_compute_bounds(x, C, method, bound_upper=True, reference_bounds=None):
            expanded_batch = x[0].shape[0]
            return (torch.ones(expanded_batch, num_specs, device=device) * -100.0,)

        mock_net.net = MagicMock()
        mock_net.net.compute_bounds = mock_compute_bounds

        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5, device=device)
        mock_node.upper = torch.ones(batch_size, 5, device=device)
        mock_node.name = 'node1'
        mock_net.net.nodes = MagicMock(return_value=[mock_node])

        return mock_net

    def test_bf_backup_threshold_triggers_sb_fallback(self):
        """Test that low objective triggers SB fallback."""
        from input_split.branching_heuristics import input_split_heuristic_bf
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_backup_thresh'] = 1e10  # Very high threshold
        arguments.Config['bab']['branching']['input_split']['bf_rhs_offset'] = 0.0
        arguments.Config['bab']['branching']['input_split']['bf_zero_crossing_score'] = False
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4
        device = 'cuda'
        x_L = torch.zeros(batch_size, input_dim, device=device)
        x_U = torch.ones(batch_size, input_dim, device=device)
        dom_lb = torch.tensor([[-0.5, -0.3], [-0.4, -0.2]], device=device)
        thresholds = torch.zeros(1, num_specs, device=device)
        lA = torch.randn(batch_size, num_specs, input_dim, device=device)

        mock_net = self._create_mock_net_with_bad_bounds(batch_size, num_specs, output_dim, input_dim, device=device)

        result = input_split_heuristic_bf(
            mock_net, x_L, x_U, dom_lb, thresholds, lA
        )

        # Should still return valid indices
        self.assertEqual(result.shape, (batch_size, 1))
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < input_dim).all())


class TestInputSplitBranchingSpecDimensions(unittest.TestCase):
    """Tests for various specification dimensions."""

    def test_multiple_specs(self):
        """Test with multiple specifications."""
        from input_split.branching_heuristics import input_split_branching
        import arguments

        arguments.Config['bab']['branching']['input_split']['sb_coeff_thresh'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_margin_weight'] = 0.0
        arguments.Config['bab']['branching']['input_split']['sb_sum'] = True
        arguments.Config['bab']['branching']['input_split']['sb_primary_spec'] = None
        arguments.Config['bab']['branching']['input_split']['touch_zero_score'] = 0.0

        batch_size = 2
        input_dim = 6
        num_specs = 5
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, num_specs)
        thresholds = torch.zeros(batch_size, num_specs)
        lA = torch.randn(batch_size, num_specs, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='sb', split_depth=2
        )

        self.assertEqual(result.shape, (batch_size, 2))


class TestInputSplitBranchingSmallInputs(unittest.TestCase):
    """Tests for small input dimensions."""

    def test_single_input_dimension(self):
        """Test with single input dimension."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 1
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=1
        )

        self.assertEqual(result.shape, (batch_size, 1))
        # Only index 0 is possible
        self.assertTrue((result == 0).all())

    def test_two_input_dimensions(self):
        """Test with two input dimensions."""
        from input_split.branching_heuristics import input_split_branching

        batch_size = 2
        input_dim = 2
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.tensor([[0.3, 0.7], [0.8, 0.2]])  # Different perturbation sizes
        dom_lb = torch.zeros(batch_size, 2)
        thresholds = torch.zeros(batch_size, 2)
        lA = torch.randn(batch_size, 2, input_dim)

        result = input_split_branching(
            None, dom_lb, x_L, x_U, lA, thresholds,
            branching_method='naive', split_depth=2
        )

        self.assertEqual(result.shape, (batch_size, 2))
        # For first batch: index 1 (0.7) > index 0 (0.3)
        self.assertEqual(result[0, 0].item(), 1)
        self.assertEqual(result[0, 1].item(), 0)
        # For second batch: index 0 (0.8) > index 1 (0.2)
        self.assertEqual(result[1, 0].item(), 0)
        self.assertEqual(result[1, 1].item(), 1)


if __name__ == '__main__':
    setup_module()
    unittest.main()
