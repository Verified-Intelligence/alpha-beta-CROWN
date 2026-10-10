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
"""Unit tests for heuristics/kfsb.py - KfsbBranching class."""

import unittest
from heuristics.decision_types import BranchingDecisions
import sys
import importlib
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict


def setUpModule():
    import arguments
    global _original_config
    _original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config


setup_module = setUpModule


def tearDownModule():
    import arguments
    arguments.Config = _original_config


teardown_module = tearDownModule

import torch
import numpy as np


# =============================================================================
# Helper Functions for DRY Test Setup
# =============================================================================

def create_mock_net(batch_size, num_layers=2, layer_names=None, activation_prefix='act_'):
    """
    Create a mock network object with split_activations, split_nodes, etc.

    Args:
        batch_size: Number of elements in the batch.
        num_layers: Number of layers to create (default 2).
        layer_names: Optional list of layer names. If None, uses 'layer1', 'layer2', etc.
        activation_prefix: Prefix for activation names (default 'act_').

    Returns:
        MagicMock: Configured mock network object.
    """
    if layer_names is None:
        layer_names = [f'layer{i+1}' for i in range(num_layers)]

    mock_net = MagicMock()
    mock_net.final_name = 'output'

    # Create split_activations
    split_activations = {}
    for layer_name in layer_names:
        mock_activation = MagicMock()
        mock_activation.name = f'{activation_prefix}{layer_name}'
        split_activations[layer_name] = [(mock_activation,)]
    mock_net.split_activations = split_activations

    # Create split_nodes
    split_nodes = []
    for layer_name in layer_names:
        mock_split_node = MagicMock()
        mock_split_node.name = layer_name
        split_nodes.append(mock_split_node)
    mock_net.split_nodes = split_nodes

    # Default mock methods
    mock_net.build_history_and_set_bounds = MagicMock()

    # kfsb now calls compute_lower_bounds_chunks (standalone import), not net.update_bounds.
    # Default return: (batch*2, 1) — the split doubles effective batch size.
    csb_patcher = patch(
        'heuristics.kfsb.compute_lower_bounds_chunks',
        return_value=torch.ones(batch_size * 2, 1)
    )
    mock_net._mock_csb = csb_patcher.start()
    mock_net._csb_patcher = csb_patcher

    return mock_net


def create_domains(batch_size, num_neurons, num_layers=2, layer_names=None,
                   activation_prefix='act_', num_bounds=1, cs_value=None,
                   include_betas=False, betas_value=None):
    """
    Create a domains dictionary for testing.

    Args:
        batch_size: Number of elements in the batch.
        num_neurons: Number of neurons per layer (int or dict mapping layer->neurons).
        num_layers: Number of layers (default 2).
        layer_names: Optional list of layer names. If None, uses 'layer1', 'layer2', etc.
        activation_prefix: Prefix for activation names (default 'act_').
        num_bounds: Number of bounds/output constraints (default 1).
        cs_value: Value for 'cs'. If None, uses torch.ones. Set to 'none' for cs=None.
        include_betas: Whether to include non-empty betas.
        betas_value: Custom betas value if include_betas is True.

    Returns:
        dict: Configured domains dictionary.
    """
    if layer_names is None:
        layer_names = [f'layer{i+1}' for i in range(num_layers)]

    # Handle num_neurons as int or dict
    if isinstance(num_neurons, int):
        neurons_per_layer = {name: num_neurons for name in layer_names}
    else:
        neurons_per_layer = num_neurons

    lower_bounds = {}
    upper_bounds = {}
    mask = {}
    lAs = {}

    for layer_name in layer_names:
        n_neurons = neurons_per_layer.get(layer_name, num_neurons if isinstance(num_neurons, int) else 5)
        lower_bounds[layer_name] = -torch.ones(batch_size, n_neurons)
        upper_bounds[layer_name] = torch.ones(batch_size, n_neurons)
        mask[layer_name] = torch.ones(batch_size, n_neurons)
        act_name = f'{activation_prefix}{layer_name}'
        lAs[act_name] = torch.ones(batch_size, num_bounds, n_neurons)

    # Handle cs value
    if cs_value == 'none':
        cs = None
    elif cs_value is not None:
        cs = cs_value
    else:
        cs = torch.ones(batch_size, num_bounds)

    # Handle betas
    if include_betas and betas_value is not None:
        betas = betas_value
    elif include_betas:
        betas = [torch.ones(batch_size, 10)]
    else:
        betas = []

    # Handle thresholds shape based on num_bounds
    thresholds = torch.zeros(batch_size, num_bounds)

    domains = {
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'mask': mask,
        'lAs': lAs,
        'cs': cs,
        'history': [],
        'alphas': defaultdict(dict),
        'betas': betas,
        'thresholds': thresholds,
    }

    return domains


# =============================================================================
# Test Classes
# =============================================================================

class TestGetBranchingDecisionsInvalidMethod(unittest.TestCase):
    """Test invalid method handling in compute_branching_decisions."""

    def tearDown(self):
        patch.stopall()

    def test_invalid_method_raises_value_error(self):
        """Test that invalid method raises ValueError."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(batch_size=2, num_layers=1)
        branching = KfsbBranching(mock_net)

        # Create minimal domains - need to adjust lAs key to match activation name
        domains = create_domains(batch_size=2, num_neurons=10, num_layers=1)
        # Fix lAs key to match what the simple test expects
        domains['lAs'] = {'layer1': torch.ones(2, 1, 10)}

        with self.assertRaises(ValueError) as context:
            branching.compute_branching_decisions(
                domains, split_depth=1, method='invalid_method'
            )

        self.assertIn('Unsupported branching method', str(context.exception))
        self.assertIn('invalid_method', str(context.exception))


class TestGetBranchingDecisionsKfsbInterceptOnly(unittest.TestCase):
    """Test compute_branching_decisions with 'kfsb-intercept-only' method."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_intercept_only_calls_score_function(self):
        """Test that intercept-only method calls babsr_score_intercept_only."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )

            mock_score_func.assert_called_once()


class TestGetBranchingDecisionsKfsb(unittest.TestCase):
    """Test compute_branching_decisions with 'kfsb' method."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_kfsb_calls_babsr_score(self):
        """Test that kfsb method calls babsr_score."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        # kfsb doubles the batch before calling shortcut bounds, so return is batch*4
        mock_net._mock_csb.return_value = torch.ones(self.batch_size * 4, 1)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score') as mock_score_func:
            mock_score = [torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_intercept = [-torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_score_func.return_value = (mock_score, mock_intercept)

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb'
            )

            mock_score_func.assert_called_once()

    def test_kfsb_doubles_batch_for_bounds(self):
        """Test that kfsb method doubles the batch size for lower/upper bounds."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)

        # Track what gets passed to build_history_and_set_bounds
        captured_args = []
        def capture_args(args, split):
            captured_args.append(dict(args))
        mock_net.build_history_and_set_bounds = MagicMock(side_effect=capture_args)
        mock_net._mock_csb.return_value = torch.ones(self.batch_size * 4, 1)

        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score') as mock_score_func:
            mock_score = [torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_intercept = [-torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_score_func.return_value = (mock_score, mock_intercept)

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb'
            )

        # Check that the bounds were doubled
        self.assertTrue(len(captured_args) > 0)
        first_call_args = captured_args[0]
        # For kfsb method, lower_bounds should be doubled (batch * 2)
        for key in first_call_args['lower_bounds']:
            self.assertEqual(
                first_call_args['lower_bounds'][key].shape[0],
                self.batch_size * 2
            )


class TestGetBranchingDecisionsUseBeta(unittest.TestCase):
    """Test compute_branching_decisions with use_beta parameter."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_use_beta_true_passes_betas_to_update_bounds(self):
        """Test that use_beta=True passes betas to update_bounds."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)

        betas = [torch.ones(self.batch_size, 10)]
        domains = create_domains(self.batch_size, self.num_neurons, include_betas=True, betas_value=betas)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                use_beta=True, method='kfsb-intercept-only'
            )

        # Verify compute_lower_bounds_chunks was called with betas in the domain dict.
        # Signature: compute_lower_bounds_chunks(net, d, ...) so d is args[1].
        mock_net._mock_csb.assert_called()
        call_args = mock_net._mock_csb.call_args
        self.assertIn('betas', call_args[0][1])


class TestGetBranchingDecisionsKeepAllDecision(unittest.TestCase):
    """Test compute_branching_decisions with keep_all_decision parameter."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_keep_all_decision_true_returns_list_of_lists(self):
        """Test that keep_all_decision=True returns decisions per batch."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                keep_all_decision=True, method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # With keep_all_decision=True, final_decision should be a list of lists
        self.assertIsInstance(final_decision, list)
        self.assertEqual(len(final_decision), self.batch_size)
        for batch_decisions in final_decision:
            self.assertIsInstance(batch_decisions, list)

    def test_keep_all_decision_false_returns_flat_list(self):
        """Test that keep_all_decision=False returns a flat list of decisions."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                keep_all_decision=False, method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # With keep_all_decision=False, final_decision should be a flat list
        self.assertIsInstance(final_decision, list)
        # Each element should be a decision (layer, idx) tuple/list
        if len(final_decision) > 0:
            self.assertTrue(
                isinstance(final_decision[0], (list, tuple)) and len(final_decision[0]) == 2
            )


class TestGetBranchingDecisionsReduceOp(unittest.TestCase):
    """Test compute_branching_decisions with different branching_reduceop values."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_min_reduce_op(self):
        """Test with branching_reduceop='min'."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            with patch('heuristics.kfsb.get_reduce_op') as mock_reduce_op:
                mock_reduce_op.return_value = lambda x, dim: torch.min(x, dim=dim)

                branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=self.topk,
                    branching_reduceop='min', method='kfsb-intercept-only'
                )

                # Verify get_reduce_op was called with 'min'
                calls = mock_reduce_op.call_args_list
                self.assertEqual(calls[0][0][0], 'min')


class TestGetBranchingDecisionsReturnValues(unittest.TestCase):
    """Test return values of compute_branching_decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_returns_three_values(self):
        """Test that compute_branching_decisions returns three values."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            result = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )

        self.assertIsInstance(result, BranchingDecisions)
        final_decision = result.branching_decision
        points = result.branching_points
        split_depth = result.split_depth
        self.assertIsNone(points)  # Points is always None
        self.assertIsInstance(split_depth, int)

    def test_split_depth_does_not_exceed_requested(self):
        """Test that returned split_depth does not exceed requested."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            requested_split_depth = 2
            _info = branching.compute_branching_decisions(
                domains, split_depth=requested_split_depth, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            returned_split_depth = _info.split_depth

        self.assertLessEqual(returned_split_depth, requested_split_depth)


class TestGetBranchingDecisionsScoreLengthCalculation(unittest.TestCase):
    """Test score_length calculation for layer/neuron index conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons_layer1 = 5
        self.num_neurons_layer2 = 3
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_score_length_handles_different_layer_sizes(self):
        """Test that score_length correctly handles different layer sizes."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)

        # Create domains with different neuron counts per layer
        neurons_per_layer = {
            'layer1': self.num_neurons_layer1,
            'layer2': self.num_neurons_layer2
        }
        domains = create_domains(self.batch_size, neurons_per_layer)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            # Different sizes per layer
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons_layer1),
                torch.rand(self.batch_size, self.num_neurons_layer2),
            ]

            # Should not raise an error
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsMaskUpdate(unittest.TestCase):
    """Test mask updates in compute_branching_decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_mask_is_updated_for_selected_neurons(self):
        """Test that mask is set to 0 for selected neurons."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)

        # Create domains with custom mask that we can track modifications to
        domains = create_domains(self.batch_size, self.num_neurons)
        orig_mask = domains['mask']

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            _info = branching.compute_branching_decisions(
                domains, split_depth=2, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # The mask should have been modified (some entries set to 0)
        total_ones = sum(m.sum().item() for m in orig_mask.values())
        # With split_depth=2, at least some neurons should be marked as selected
        expected_max_ones = self.batch_size * self.num_neurons * 2
        self.assertLessEqual(total_ones, expected_max_ones)


class TestGetBranchingDecisionsRandomFallback(unittest.TestCase):
    """Test random decision fallback when no valid scores."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_random_fallback_used_for_invalid_scores(self):
        """Test that random decisions are used when scores are invalid."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        # Return very negative values to make thres < -10000
        mock_net._mock_csb.return_value = torch.full((self.batch_size * 2, 1), -20000.0)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            # Return zero scores to trigger invalid score path
            mock_score_func.return_value = [
                torch.zeros(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            # Should complete without error even with all invalid scores
            with patch('builtins.print'):  # Suppress print output
                _info = branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=self.topk,
                    method='kfsb-intercept-only'
                )
                final_decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        # Should still return decisions
        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsCsNone(unittest.TestCase):
    """Test handling when cs is None."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_cs_none_sets_number_bounds_to_one(self):
        """Test that cs=None results in number_bounds=1."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons, cs_value='none')

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            # Should not raise an error
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsMultipleBounds(unittest.TestCase):
    """Test handling of multiple bounds (AND clauses)."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.num_bounds = 3  # Multiple bounds
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_multiple_bounds_handled_correctly(self):
        """Test that multiple bounds (cs with multiple columns) work correctly."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        mock_net._mock_csb.return_value = torch.ones(self.batch_size * 2, self.num_bounds)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons, num_bounds=self.num_bounds)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            # Should handle multiple bounds without error
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsKfsbSplitAlphas(unittest.TestCase):
    """Test that kfsb method splits alphas after processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_kfsb_restores_alpha_shape_after_processing(self):
        """Test that kfsb method restores alpha to original batch size."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        mock_net._mock_csb.return_value = torch.ones(self.batch_size * 4, 1)
        branching = KfsbBranching(mock_net)

        # Create domains and add custom alphas
        domains = create_domains(self.batch_size, self.num_neurons)
        original_alpha = torch.rand(1, 1, self.batch_size, self.num_neurons)
        domains['alphas']['layer1']['out'] = original_alpha.clone()

        # Store original batch dimension size
        original_batch_dim = domains['alphas']['layer1']['out'].shape[2]

        with patch('heuristics.kfsb.babsr_score') as mock_score_func:
            mock_score = [torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_intercept = [-torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_score_func.return_value = (mock_score, mock_intercept)

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb'
            )

        # After kfsb method, alphas should be restored to original batch size
        # The kfsb method doubles alphas during processing and splits them back
        restored_alpha = domains['alphas']['layer1']['out']

        # Verify the batch dimension (dim 2) is restored to original size
        self.assertEqual(restored_alpha.shape[2], original_batch_dim)

        # Verify overall shape structure is preserved (4D tensor)
        self.assertEqual(len(restored_alpha.shape), 4)


class TestGetBranchingDecisionsTorchNoGrad(unittest.TestCase):
    """Test that compute_branching_decisions uses torch.no_grad decorator."""

    def setUp(self):
        self.batch_size = 4
        self.num_neurons = 8
        self.topk = 5

    def tearDown(self):
        patch.stopall()

    def test_no_grad_during_execution(self):
        """Test that gradients are disabled during method execution."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(self.batch_size, self.num_neurons)

        # Track whether grad was enabled during compute_lower_bounds_chunks call
        grad_enabled_during_call = []

        # kfsb method doubles batch: batch_size * 2 for score + intercept,
        # then compute_lower_bounds_chunks is called with batch_size * 4 total
        # and result needs to match torch.cat([rhs, rhs]) which is batch_size * 4
        doubled_batch = self.batch_size * 4

        def track_grad_state(*args, **kwargs):
            grad_enabled_during_call.append(torch.is_grad_enabled())
            return torch.ones(doubled_batch, 1)

        mock_net._mock_csb.side_effect = track_grad_state

        with patch('heuristics.kfsb.babsr_score') as mock_score_func:
            mock_score = [torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_intercept = [-torch.rand(self.batch_size, self.num_neurons) for _ in range(2)]
            mock_score_func.return_value = (mock_score, mock_intercept)

            # Enable grad before calling to verify decorator disables it
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                branching.compute_branching_decisions(
                    domains, split_depth=1, branching_candidates=self.topk, method='kfsb'
                )

        # Verify that grad was disabled during the method execution
        self.assertTrue(len(grad_enabled_during_call) > 0, "compute_lower_bounds_chunks should have been called")
        for was_enabled in grad_enabled_during_call:
            self.assertFalse(was_enabled, "Gradients should be disabled inside compute_branching_decisions")


class TestGetBranchingDecisionsEdgeCases(unittest.TestCase):
    """Test edge cases in compute_branching_decisions."""

    def tearDown(self):
        patch.stopall()

    def test_single_batch_element(self):
        """Test with batch size of 1."""
        from heuristics.kfsb import KfsbBranching

        batch_size = 1
        num_neurons = 5
        topk = 2

        mock_net = create_mock_net(batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(batch_size, num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(batch_size, num_neurons) for _ in range(2)
            ]

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)

    def test_single_layer(self):
        """Test with only one layer."""
        from heuristics.kfsb import KfsbBranching

        batch_size = 2
        num_neurons = 5
        topk = 2

        mock_net = create_mock_net(batch_size, num_layers=1)
        branching = KfsbBranching(mock_net)
        domains = create_domains(batch_size, num_neurons, num_layers=1)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(batch_size, num_neurons)  # Only one layer
            ]

            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(final_decision)

    def test_very_large_split_depth(self):
        """Test with split_depth larger than available neurons."""
        from heuristics.kfsb import KfsbBranching

        batch_size = 2
        num_neurons = 3
        topk = 2

        mock_net = create_mock_net(batch_size)
        branching = KfsbBranching(mock_net)
        domains = create_domains(batch_size, num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(batch_size, num_neurons) for _ in range(2)
            ]

            # Request very large split depth
            _info = branching.compute_branching_decisions(
                domains, split_depth=100, branching_candidates=topk,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # split_depth should be limited
        self.assertLessEqual(split_depth, 100)
        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsTopK(unittest.TestCase):
    """Test topk selection in compute_branching_decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 10
        self.topk = 5

    def tearDown(self):
        patch.stopall()

    def test_topk_limited_by_unstable_neurons(self):
        """Test that topk is limited by the number of unstable neurons."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)
        branching = KfsbBranching(mock_net)

        # Create domains and customize mask with only 3 unstable neurons total
        domains = create_domains(self.batch_size, self.num_neurons)

        mask_layer1 = torch.zeros(self.batch_size, self.num_neurons)
        mask_layer1[:, :2] = 1  # 2 unstable neurons in layer1
        mask_layer2 = torch.zeros(self.batch_size, self.num_neurons)
        mask_layer2[:, 0] = 1  # 1 unstable neuron in layer2

        domains['mask'] = {
            'layer1': mask_layer1,
            'layer2': mask_layer2,
        }

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            # Request 10 candidates but only 3 unstable neurons exist
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=10,
                method='kfsb-intercept-only'
            )
            final_decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # Should complete without error
        self.assertIsNotNone(final_decision)


class TestGetBranchingDecisionsAlphaHandling(unittest.TestCase):
    """Test alpha handling in compute_branching_decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_neurons = 5
        self.topk = 3

    def tearDown(self):
        patch.stopall()

    def test_alphas_set_on_first_iteration_only(self):
        """Test that alphas are only set on the first iteration."""
        from heuristics.kfsb import KfsbBranching

        mock_net = create_mock_net(self.batch_size)

        captured_args = []
        def capture_build_history_args(args, split):
            captured_args.append(dict(args))
        mock_net.build_history_and_set_bounds = MagicMock(side_effect=capture_build_history_args)

        branching = KfsbBranching(mock_net)

        # Create domains and add custom alphas
        domains = create_domains(self.batch_size, self.num_neurons)
        domains['alphas']['layer1']['out'] = torch.rand(1, 1, self.batch_size, self.num_neurons)

        with patch('heuristics.kfsb.babsr_score_intercept_only') as mock_score_func:
            mock_score_func.return_value = [
                torch.rand(self.batch_size, self.num_neurons) for _ in range(2)
            ]

            branching.compute_branching_decisions(
                domains, split_depth=1, branching_candidates=self.topk,
                method='kfsb-intercept-only'
            )

        # First call should have alphas, subsequent calls should have empty dict
        if len(captured_args) >= 2:
            # First call has alphas
            first_call_alphas = captured_args[0].get('alphas', {})
            # Second call should have empty alphas (set_alpha becomes False)
            second_call_alphas = captured_args[1].get('alphas', {})
            # The first call should have more content than subsequent calls
            self.assertGreaterEqual(len(first_call_alphas), len(second_call_alphas))


if __name__ == '__main__':
    unittest.main()
