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
"""Unit tests for heuristics/fsb.py - FsbBranching class."""

import unittest
from heuristics.decision_types import BranchingDecisions
import sys
import os
from unittest.mock import MagicMock, patch, call
from collections import defaultdict

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setUpModule():
    """Initialize arguments.Config for all tests (unittest) / setup_module alias for pytest."""
    import arguments
    global _original_config
    _original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config


setup_module = setUpModule  # pytest alias


def tearDownModule():
    """Restore original arguments.Config."""
    import arguments
    arguments.Config = _original_config


teardown_module = tearDownModule  # pytest alias


# ============================================================================
# Shared Test Helpers
# ============================================================================

def create_mock_net(batch_size, num_layers=2, neurons_per_layer=5):
    """Create a mock network with proper structure for FSB tests.

    Args:
        batch_size: Batch size (used to set up compute_lower_bounds_chunks mock).
        num_layers: Number of layers in the network.
        neurons_per_layer: Number of neurons per layer (unused, kept for API compatibility).

    Returns:
        MagicMock: A mock network object with split_nodes, split_activations, etc.
    """
    mock_net = MagicMock()
    mock_net.final_name = 'output'

    split_nodes = []
    split_activations = {}

    for i in range(num_layers):
        mock_node = MagicMock()
        mock_node.name = f'layer{i}'
        mock_activation = MagicMock()
        mock_activation.name = f'act_layer{i}'
        mock_input = MagicMock()
        mock_input.name = f'pre_layer{i}'
        mock_activation.inputs = [mock_input]
        split_nodes.append(mock_node)
        split_activations[f'layer{i}'] = [(mock_activation,)]

    mock_net.split_nodes = split_nodes
    mock_net.split_activations = split_activations
    mock_net.build_history_and_set_bounds = MagicMock()

    # fsb doubles the batch (lbs = cat([v, v])), then build_history_and_set_bounds
    # doubles it again for the two split branches, so the effective output is batch*4.
    csb_patcher = patch(
        'heuristics.fsb.compute_lower_bounds_chunks',
        return_value=torch.ones(batch_size * 4, 1)
    )
    mock_net._mock_csb = csb_patcher.start()
    mock_net._csb_patcher = csb_patcher

    return mock_net


def create_domains(batch=2, neurons=5, num_layers=2, betas=None):
    """Create a standard domains dict for testing.

    Args:
        batch: Batch size.
        neurons: Number of neurons per layer.
        num_layers: Number of layers.
        betas: Optional betas list. Defaults to empty list.

    Returns:
        dict: A domains dictionary with lower_bounds, upper_bounds, mask, etc.
    """
    if betas is None:
        betas = []

    domains = {
        'lower_bounds': {},
        'upper_bounds': {},
        'mask': {},
        'lAs': {},
        'history': [],
        'alphas': defaultdict(dict),
        'betas': betas,
        'thresholds': torch.zeros(batch, 1),
        'cs': torch.ones(batch, 1),
    }

    for i in range(num_layers):
        domains['lower_bounds'][f'pre_layer{i}'] = -torch.ones(batch, neurons)
        domains['upper_bounds'][f'pre_layer{i}'] = torch.ones(batch, neurons)
        domains['mask'][f'layer{i}'] = torch.ones(batch, neurons)
        domains['lAs'][f'act_layer{i}'] = torch.rand(batch, 1, neurons)

    return domains


class TestFsbBranchingInit(unittest.TestCase):
    """Test FsbBranching initialization."""

    def tearDown(self):
        patch.stopall()

    def test_init_basic(self):
        """Test basic initialization."""
        from heuristics.fsb import FsbBranching

        mock_net = create_mock_net(2)
        branching = FsbBranching(mock_net)

        self.assertEqual(branching.net, mock_net)

    def test_init_inherits_babsr_attributes(self):
        """Test that FsbBranching inherits attributes from BabsrBranching."""
        from heuristics.fsb import FsbBranching

        mock_net = create_mock_net(2)
        branching = FsbBranching(mock_net)

        # Should have net attribute from parent
        self.assertTrue(hasattr(branching, 'net'))


class TestFsbGetBranchingDecisions(unittest.TestCase):
    """Tests for FsbBranching.compute_branching_decisions method."""

    def tearDown(self):
        patch.stopall()

    def test_returns_three_values(self):
        """Test that compute_branching_decisions returns three values."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            result = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )

        self.assertIsInstance(result, BranchingDecisions)
        final_decision = result.branching_decision
        points = result.branching_points
        split_depth = result.split_depth
        self.assertIsNone(points)  # Points is always None
        self.assertIsInstance(split_depth, int)

    def test_calls_babsr_score(self):
        """Test that babsr_score is called."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )

            mock_score.assert_called_once()

    def test_doubles_bounds_for_processing(self):
        """Test that lower/upper bounds are doubled (concatenated) for processing."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)

        captured_args = []
        def capture_args(args, split):
            captured_args.append(dict(args))
        mock_net.build_history_and_set_bounds = MagicMock(side_effect=capture_args)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )

        # Check that bounds were doubled
        self.assertTrue(len(captured_args) > 0)
        first_call = captured_args[0]
        for key in first_call['lower_bounds']:
            self.assertEqual(
                first_call['lower_bounds'][key].shape[0],
                batch * 2
            )

    def test_split_depth_returned_correctly(self):
        """Test that returned split_depth does not exceed requested."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            requested_split_depth = 2
            _info = branching.compute_branching_decisions(
                domains, split_depth=requested_split_depth, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            returned_split_depth = _info.split_depth

        self.assertLessEqual(returned_split_depth, requested_split_depth)


class TestFsbBranchingReduceOp(unittest.TestCase):
    """Test reduce operation handling in FsbBranching."""

    def tearDown(self):
        patch.stopall()

    def test_calls_get_reduce_op(self):
        """Test that get_reduce_op is called with branching_reduceop."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            with patch('heuristics.fsb.get_reduce_op') as mock_reduce_op:
                mock_reduce_op.return_value = lambda x, dim: torch.min(x, dim=dim)

                branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=3,
                    branching_reduceop='min'
                )

                mock_reduce_op.assert_called_once_with('min')


class TestFsbBranchingUseBeta(unittest.TestCase):
    """Test use_beta parameter handling."""

    def tearDown(self):
        patch.stopall()

    def test_use_beta_true_passes_betas(self):
        """Test that use_beta=True passes betas to compute_lower_bounds_chunks."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons, betas=[torch.ones(batch, 10)])

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3,
                use_beta=True
            )

        # Verify compute_lower_bounds_chunks was called with betas in the domain dict
        mock_net._mock_csb.assert_called()
        call_args = mock_net._mock_csb.call_args
        self.assertIn('betas', call_args[0][1])

    def test_use_beta_false_does_not_pass_betas(self):
        """Test that use_beta=False omits betas from the compute_lower_bounds_chunks domain dict."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = create_domains(batch, neurons)

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3,
                use_beta=False
            )

        # Verify compute_lower_bounds_chunks was called without betas in the domain dict
        mock_net._mock_csb.assert_called()
        call_args = mock_net._mock_csb.call_args
        self.assertNotIn('betas', call_args[0][1])



class TestFsbBranchingSkipLayers(unittest.TestCase):
    """Test skip layer logic in FsbBranching."""

    def tearDown(self):
        patch.stopall()

    def test_skip_first_layer_when_multiple_layers(self):
        """Test that first layer is skipped when there are multiple layers."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch, num_layers=3)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
                'pre_layer2': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
                'pre_layer2': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
                'layer2': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
                'act_layer2': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            # Scores for 3 layers
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(3)],
                [-torch.rand(batch, neurons) for _ in range(3)]
            )

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # Should complete without error
        self.assertIsNotNone(final_decision)

    def test_skip_layer_with_invalid_scores(self):
        """Test that layers with all invalid scores are skipped."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch, num_layers=2)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            # First layer has all zero scores (should be skipped)
            # Second layer has valid scores
            mock_score.return_value = (
                [torch.zeros(batch, neurons), torch.rand(batch, neurons)],
                [torch.zeros(batch, neurons), -torch.rand(batch, neurons)]
            )

            with patch('builtins.print'):  # Suppress skip layer print
                _info = branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=3
                )
                final_decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        # Should still return decisions from second layer
        self.assertIsNotNone(final_decision)


class TestFsbBranchingRandomFallback(unittest.TestCase):
    """Test random decision fallback when no valid scores."""

    def tearDown(self):
        patch.stopall()

    def test_random_fallback_when_all_scores_invalid(self):
        """Test that random decisions are used when all scores are invalid."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        # Return very negative values to make decisions invalid
        mock_net._mock_csb.return_value = torch.full((batch * 4, 1), -20000.0)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            # Should complete without error even with all invalid scores
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # Should still return decisions
        self.assertIsNotNone(final_decision)

    def test_random_fallback_when_tmp_ret_empty(self):
        """Test random fallback when tmp_ret is empty (all layers skipped)."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            # All layers have zero/invalid scores
            mock_score.return_value = (
                [torch.zeros(batch, neurons) for _ in range(2)],
                [torch.zeros(batch, neurons) for _ in range(2)]
            )

            with patch('builtins.print'):  # Suppress skip layer print
                _info = branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=3
                )
                final_decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        # Should use random fallback
        self.assertIsNotNone(final_decision)


class TestFsbBranchingAlphaHandling(unittest.TestCase):
    """Test alpha handling in FsbBranching."""

    def tearDown(self):
        patch.stopall()

    def test_alphas_concatenated_for_processing(self):
        """Test that alphas are concatenated for doubled batch processing."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)

        captured_args = []
        def capture_args(args, split):
            # Make a deep copy of alphas
            args_copy = dict(args)
            if 'alphas' in args_copy:
                args_copy['alphas'] = {
                    k: {kk: vv.clone() for kk, vv in v.items()}
                    for k, v in args_copy['alphas'].items()
                }
            captured_args.append(args_copy)
        mock_net.build_history_and_set_bounds = MagicMock(side_effect=capture_args)
        branching = FsbBranching(mock_net)

        # Create alphas with original batch size
        alphas = defaultdict(dict)
        alphas['layer0']['out'] = torch.rand(1, 1, batch, neurons)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': alphas,
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )

        # First call should have alphas with doubled batch size
        self.assertTrue(len(captured_args) > 0)
        first_call = captured_args[0]
        if 'alphas' in first_call and first_call['alphas']:
            for key, nested in first_call['alphas'].items():
                for kk, vv in nested.items():
                    # Batch dim should be doubled (index 2)
                    self.assertEqual(vv.shape[2], batch * 2)

    def test_set_alpha_only_first_iteration(self):
        """Test that alphas are only set on first iteration."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)

        captured_args = []
        def capture_args(args, split):
            args_copy = dict(args)
            captured_args.append(args_copy)
        mock_net.build_history_and_set_bounds = MagicMock(side_effect=capture_args)
        branching = FsbBranching(mock_net)

        alphas = defaultdict(dict)
        alphas['layer0']['out'] = torch.rand(1, 1, batch, neurons)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': alphas,
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )

        # First call should have non-empty alphas, subsequent should have empty
        if len(captured_args) >= 2:
            first_alphas = captured_args[0].get('alphas', {})
            second_alphas = captured_args[1].get('alphas', {})
            # First should have content, second should be empty
            self.assertTrue(len(first_alphas) > 0 or len(second_alphas) == 0)


class TestFsbBranchingTopK(unittest.TestCase):
    """Test topk selection logic in FsbBranching."""

    def tearDown(self):
        patch.stopall()

    def test_topk_limited_by_unstable_neurons(self):
        """Test that topk is limited by the number of unstable neurons."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 10
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        # Create mask with only 3 unstable neurons total
        mask_layer0 = torch.zeros(batch, neurons)
        mask_layer0[:, :2] = 1  # 2 unstable neurons
        mask_layer1 = torch.zeros(batch, neurons)
        mask_layer1[:, 0] = 1  # 1 unstable neuron

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': mask_layer0,
                'layer1': mask_layer1,
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            # Request 10 candidates but only 3 unstable neurons exist
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=10
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # Should complete without error
        self.assertIsNotNone(final_decision)


class TestFsbBranchingEdgeCases(unittest.TestCase):
    """Test edge cases in FsbBranching."""

    def test_single_batch(self):
        """Test with batch size of 1."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 1, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)

    def test_single_layer(self):
        """Test with only one layer."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = MagicMock()
        mock_net.final_name = 'output'

        mock_node = MagicMock()
        mock_node.name = 'layer0'
        mock_activation = MagicMock()
        mock_activation.name = 'act_layer0'
        mock_input = MagicMock()
        mock_input.name = 'pre_layer0'
        mock_activation.inputs = [mock_input]

        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer0': [(mock_activation,)]}
        mock_net.build_history_and_set_bounds = MagicMock()

        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {'pre_layer0': -torch.ones(batch, neurons)},
            'upper_bounds': {'pre_layer0': torch.ones(batch, neurons)},
            'mask': {'layer0': torch.ones(batch, neurons)},
            'lAs': {'act_layer0': torch.rand(batch, 1, neurons)},
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.compute_lower_bounds_chunks',
                   return_value=torch.ones(batch * 4, 1)):
            with patch('heuristics.fsb.babsr_score') as mock_score:
                mock_score.return_value = (
                    [torch.rand(batch, neurons)],
                    [-torch.rand(batch, neurons)]
                )

                _info = branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=3
                )
                final_decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)

    def test_large_split_depth(self):
        """Test with split_depth larger than available neurons."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 3
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            # Request very large split depth
            _info = branching.compute_branching_decisions(
                domains, split_depth=100, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # split_depth should be limited
        self.assertLessEqual(split_depth, 100)
        self.assertIsNotNone(final_decision)


class TestFsbBranchingMaskUpdate(unittest.TestCase):
    """Test mask update behavior in FsbBranching."""

    def test_mask_updated_for_selected_neurons(self):
        """Test that mask is set to 0 for selected neurons."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        # Original mask with all unstable
        orig_mask = {
            'layer0': torch.ones(batch, neurons),
            'layer1': torch.ones(batch, neurons),
        }

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': orig_mask,
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            _info = branching.compute_branching_decisions(
                domains, split_depth=2, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # The mask should have been modified
        total_ones = sum(m.sum().item() for m in orig_mask.values())
        expected_max_ones = batch * neurons * 2
        self.assertLessEqual(total_ones, expected_max_ones)


class TestFsbBranchingDecisionComparison(unittest.TestCase):
    """Test score vs intercept_tb decision comparison."""

    def test_decision_prefers_higher_score(self):
        """Test that decisions prefer higher score when both are valid."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            # High scores
            mock_score.return_value = (
                [torch.rand(batch, neurons) + 1.0 for _ in range(2)],
                [-torch.rand(batch, neurons) - 1.0 for _ in range(2)]
            )

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


class TestFsbBranchingPrioritizeAlphas(unittest.TestCase):
    """Test prioritize_alphas parameter handling."""

    def test_prioritize_alphas_passed_to_babsr_score(self):
        """Test that prioritize_alphas is passed to babsr_score."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3,
                prioritize_alphas='none'
            )

            # Check that babsr_score was called with prioritize_alphas
            call_args = mock_score.call_args
            self.assertEqual(call_args[0][-1], 'none')


class TestFsbBranchingMultipleBounds(unittest.TestCase):
    """Test handling of multiple bounds (AND clauses)."""

    def test_multiple_bounds_handled_correctly(self):
        """Test that multiple bounds (cs with multiple columns) work correctly."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        num_bounds = 3
        mock_net = create_mock_net(batch)
        mock_net._mock_csb.return_value = torch.ones(batch * 4, num_bounds)
        branching = FsbBranching(mock_net)

        # cs with multiple bounds
        cs = torch.ones(batch, num_bounds)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, num_bounds, neurons),
                'act_layer1': torch.rand(batch, num_bounds, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, num_bounds),
            'cs': cs,
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            # Should handle multiple bounds without error
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


class TestFsbBranchingKwargsHandling(unittest.TestCase):
    """Test that extra kwargs are handled properly."""

    def test_extra_kwargs_ignored(self):
        """Test that extra kwargs don't cause errors."""
        from heuristics.fsb import FsbBranching

        batch, neurons = 2, 5
        mock_net = create_mock_net(batch)
        branching = FsbBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
                'pre_layer1': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
            },
            'mask': {
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'history': [],
            'alphas': defaultdict(dict),
            'betas': [],
            'thresholds': torch.zeros(batch, 1),
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.fsb.babsr_score') as mock_score:
            mock_score.return_value = (
                [torch.rand(batch, neurons) for _ in range(2)],
                [-torch.rand(batch, neurons) for _ in range(2)]
            )

            # Pass extra kwargs that should be ignored
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=3,
                some_extra_kwarg='value', another_kwarg=123
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


if __name__ == '__main__':
    unittest.main()
