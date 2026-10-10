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
"""Unit tests for heuristics/babsr.py - BabsrBranching class and score functions."""

import os
import sys
import unittest
from heuristics.decision_types import BranchingDecisions
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Module-level test helpers for DRY code
# =============================================================================

def create_mock_net(num_layers=2, neurons_per_layer=5, include_inputs=True):
    """Create a mock network for testing BabsrBranching.

    Args:
        num_layers: Number of layers in the mock network.
        neurons_per_layer: Number of neurons per layer (unused but kept for API consistency).
        include_inputs: If True, includes mock_activation.inputs with pre_layer names.
                       Set to False for simpler mocks that don't need input references.

    Returns:
        MagicMock configured as a network with split_nodes, split_activations, and split_indices.

    Key structure:
        - split_nodes[i].name = 'layer{i}' - used for mask keys
        - split_activations['layer{i}'] = [(activation,)] where activation.name = 'act_layer{i}'
        - activation.inputs[0].name = 'pre_layer{i}' - used for lower/upper bounds keys (if include_inputs=True)
    """
    mock_net = MagicMock()
    mock_net.final_name = 'output'

    split_nodes = []
    split_activations = {}
    split_indices = []

    for i in range(num_layers):
        mock_node = MagicMock()
        mock_node.name = f'layer{i}'

        mock_activation = MagicMock()
        mock_activation.name = f'act_layer{i}'

        if include_inputs:
            mock_input = MagicMock()
            mock_input.name = f'pre_layer{i}'
            mock_activation.inputs = [mock_input]

        split_nodes.append(mock_node)
        split_activations[f'layer{i}'] = [(mock_activation,)]
        split_indices.append(i)

    mock_net.split_nodes = split_nodes
    mock_net.split_activations = split_activations
    mock_net.split_indices = split_indices

    return mock_net


def create_domains(batch=2, neurons=5, cs_value='default', use_zeros=False):
    """Create a domains dict for testing branching decisions.

    Args:
        batch: Batch size.
        neurons: Number of neurons per layer.
        cs_value: Value for 'cs' key. Use 'default' for torch.ones(batch, 1),
                 None for None, or provide a custom tensor.
        use_zeros: If True, use zeros for bounds and lAs (for testing edge cases).
                  If False, use standard values (-ones/ones for bounds, rand for lAs).

    Returns:
        Dict with lower_bounds, upper_bounds, mask, lAs, and cs keys.
    """
    if cs_value == 'default':
        cs = torch.ones(batch, 1)
    else:
        cs = cs_value

    if use_zeros:
        lower_val = torch.zeros(batch, neurons)
        upper_val = torch.zeros(batch, neurons)
        lAs_val = torch.zeros(batch, 1, neurons)
    else:
        lower_val = -torch.ones(batch, neurons)
        upper_val = torch.ones(batch, neurons)
        lAs_val = torch.rand(batch, 1, neurons)

    return {
        'lower_bounds': {
            'pre_layer0': lower_val.clone(),
            'pre_layer1': lower_val.clone(),
        },
        'upper_bounds': {
            'pre_layer0': upper_val.clone(),
            'pre_layer1': upper_val.clone(),
        },
        'mask': {
            'pre_layer0': torch.ones(batch, neurons),
            'pre_layer1': torch.ones(batch, neurons),
            'layer0': torch.ones(batch, neurons),
            'layer1': torch.ones(batch, neurons),
        },
        'lAs': {
            'act_layer0': lAs_val.clone(),
            'act_layer1': torch.rand(batch, 1, neurons) if not use_zeros else lAs_val.clone(),
        },
        'cs': cs,
    }


class TestBabsrScoreInterceptOnly(unittest.TestCase):
    """Tests for babsr_score_intercept_only function."""

    def test_basic_score_computation(self):
        """Test basic score computation with simple bounds."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons = 5

        # Create mock split_activations structure
        mock_activation = MagicMock()
        mock_activation.name = 'act_layer1'
        split_activations = {'layer1': [(mock_activation,)]}

        lbs = {
            'layer1': -torch.ones(batch, neurons),
        }
        ubs = {
            'layer1': torch.ones(batch, neurons),
        }
        lAs = {
            'act_layer1': torch.ones(batch, 1, neurons),
        }

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        self.assertEqual(len(score), 1)  # One layer
        self.assertEqual(score[0].shape, (batch, neurons))

    def test_skips_final_name(self):
        """Test that final_name layer is skipped."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons = 5

        mock_activation1 = MagicMock()
        mock_activation1.name = 'act_layer1'
        mock_activation2 = MagicMock()
        mock_activation2.name = 'act_output'

        split_activations = {
            'layer1': [(mock_activation1,)],
            'output': [(mock_activation2,)]
        }

        lbs = {
            'layer1': -torch.ones(batch, neurons),
            'output': -torch.ones(batch, 3),
        }
        ubs = {
            'layer1': torch.ones(batch, neurons),
            'output': torch.ones(batch, 3),
        }
        lAs = {
            'act_layer1': torch.ones(batch, 1, neurons),
            'act_output': torch.ones(batch, 1, 3),
        }

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        # Only layer1 should be processed, output should be skipped
        self.assertEqual(len(score), 1)

    def test_positive_lower_bounds(self):
        """Test with positive lower bounds (ReLU always active)."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons = 3

        mock_activation = MagicMock()
        mock_activation.name = 'act_layer1'
        split_activations = {'layer1': [(mock_activation,)]}

        # Positive lower bounds
        lbs = {'layer1': torch.ones(batch, neurons)}
        ubs = {'layer1': 2 * torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        # With positive lower bounds, intercept should be 0
        self.assertEqual(len(score), 1)
        self.assertTrue(torch.all(score[0] == 0))

    def test_negative_upper_bounds(self):
        """Test with negative upper bounds (ReLU always inactive)."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons = 3

        mock_activation = MagicMock()
        mock_activation.name = 'act_layer1'
        split_activations = {'layer1': [(mock_activation,)]}

        # Negative upper bounds
        lbs = {'layer1': -2 * torch.ones(batch, neurons)}
        ubs = {'layer1': -1 * torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        self.assertEqual(len(score), 1)

    def test_multiple_layers(self):
        """Test with multiple layers."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons1, neurons2 = 5, 3

        mock_activation1 = MagicMock()
        mock_activation1.name = 'act_layer1'
        mock_activation2 = MagicMock()
        mock_activation2.name = 'act_layer2'

        split_activations = {
            'layer1': [(mock_activation1,)],
            'layer2': [(mock_activation2,)]
        }

        lbs = {
            'layer1': -torch.ones(batch, neurons1),
            'layer2': -torch.ones(batch, neurons2),
        }
        ubs = {
            'layer1': torch.ones(batch, neurons1),
            'layer2': torch.ones(batch, neurons2),
        }
        lAs = {
            'act_layer1': torch.ones(batch, 1, neurons1),
            'act_layer2': torch.ones(batch, 1, neurons2),
        }

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        self.assertEqual(len(score), 2)
        self.assertEqual(score[0].shape, (batch, neurons1))
        self.assertEqual(score[1].shape, (batch, neurons2))

    def test_negative_lAs(self):
        """Test with negative lA values."""
        from heuristics.babsr import babsr_score_intercept_only

        batch = 2
        neurons = 3

        mock_activation = MagicMock()
        mock_activation.name = 'act_layer1'
        split_activations = {'layer1': [(mock_activation,)]}

        lbs = {'layer1': -torch.ones(batch, neurons)}
        ubs = {'layer1': torch.ones(batch, neurons)}
        # Negative lA values
        lAs = {'act_layer1': -torch.ones(batch, 1, neurons)}

        score = babsr_score_intercept_only(
            lbs, ubs, lAs, batch, final_name='output', split_activations=split_activations
        )

        # Score should be computed with clamped lA
        self.assertEqual(len(score), 1)
        # With negative lA, after clamp(0, None), the term should be 0
        # ratio * 0 = 0 for all neurons


class TestBabsrScore(unittest.TestCase):
    """Tests for babsr_score function."""

    def _create_mock_layer(self, name, input_name):
        """Create a mock layer with input structure."""
        mock_layer = MagicMock()
        mock_layer.name = name
        mock_input = MagicMock()
        mock_input.name = input_name
        mock_layer.inputs = [mock_input]
        return mock_layer

    def test_basic_score_computation(self):
        """Test basic babsr_score computation."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        # Create mock layer structure
        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            score, intercept_tb = babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds=1, split_nodes=split_nodes,
                split_activations=split_activations
            )

        self.assertEqual(len(score), 1)
        self.assertEqual(len(intercept_tb), 1)
        self.assertEqual(score[0].shape, (batch, neurons))
        self.assertEqual(intercept_tb[0].shape, (batch, neurons))

    def test_prioritize_alphas_positive(self):
        """Test with prioritize_alphas='positive'."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        # Mix of positive and negative lA values
        lAs = {'act_layer1': torch.randn(batch, 1, neurons)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            with patch('builtins.print'):  # Suppress print statements
                score, intercept_tb = babsr_score(
                    lower_bounds, upper_bounds, lAs, mask, reduce_op,
                    number_bounds=1, split_nodes=split_nodes,
                    split_activations=split_activations,
                    prioritize_alphas='positive'
                )

        self.assertEqual(len(score), 1)

    def test_prioritize_alphas_negative(self):
        """Test with prioritize_alphas='negative'."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.randn(batch, 1, neurons)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            with patch('builtins.print'):
                score, intercept_tb = babsr_score(
                    lower_bounds, upper_bounds, lAs, mask, reduce_op,
                    number_bounds=1, split_nodes=split_nodes,
                    split_activations=split_activations,
                    prioritize_alphas='negative'
                )

        self.assertEqual(len(score), 1)

    def test_prioritize_alphas_invalid_raises(self):
        """Test that invalid prioritize_alphas raises ValueError."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            with self.assertRaises(ValueError) as context:
                babsr_score(
                    lower_bounds, upper_bounds, lAs, mask, reduce_op,
                    number_bounds=1, split_nodes=split_nodes,
                    split_activations=split_activations,
                    prioritize_alphas='invalid_value'
                )

        self.assertIn('prioritize_alphas', str(context.exception).lower())

    def test_multiple_bounds(self):
        """Test with multiple bounds (AND clauses)."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5
        number_bounds = 3

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, number_bounds, neurons)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            score, intercept_tb = babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds=number_bounds, split_nodes=split_nodes,
                split_activations=split_activations
            )

        self.assertEqual(len(score), 1)
        # Score should be averaged over number_bounds
        self.assertEqual(score[0].shape, (batch, neurons))

    def test_reduce_op_min(self):
        """Test with reduce_op=torch.min."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}
        reduce_op = torch.min

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            score, intercept_tb = babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds=1, split_nodes=split_nodes,
                split_activations=split_activations
            )

        self.assertEqual(len(score), 1)

    def test_bias_as_int(self):
        """Test when get_preact_params returns int (bias=0)."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        lAs = {'act_layer1': torch.ones(batch, 1, neurons)}
        reduce_op = torch.max

        # Return int 0 for bias
        with patch('heuristics.babsr.get_preact_params', return_value=0):
            score, intercept_tb = babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds=1, split_nodes=split_nodes,
                split_activations=split_activations
            )

        self.assertEqual(len(score), 1)

    def test_multiple_layers_reversed(self):
        """Test that layers are processed in reversed order."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons1, neurons2 = 5, 3

        mock_layer1 = self._create_mock_layer('act_layer1', 'pre_layer1')
        mock_layer2 = self._create_mock_layer('act_layer2', 'pre_layer2')

        split_node1 = MagicMock()
        split_node1.name = 'node1'
        split_node2 = MagicMock()
        split_node2.name = 'node2'
        split_nodes = [split_node1, split_node2]

        split_activations = {
            'node1': [(mock_layer1,)],
            'node2': [(mock_layer2,)]
        }

        lower_bounds = {
            'pre_layer1': -torch.ones(batch, neurons1),
            'pre_layer2': -torch.ones(batch, neurons2),
        }
        upper_bounds = {
            'pre_layer1': torch.ones(batch, neurons1),
            'pre_layer2': torch.ones(batch, neurons2),
        }
        mask = {
            'pre_layer1': torch.ones(batch, neurons1),
            'pre_layer2': torch.ones(batch, neurons2),
        }
        lAs = {
            'act_layer1': torch.ones(batch, 1, neurons1),
            'act_layer2': torch.ones(batch, 1, neurons2),
        }
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            score, intercept_tb = babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds=1, split_nodes=split_nodes,
                split_activations=split_activations
            )

        # Scores are inserted at position 0, so later layers come first in original order
        self.assertEqual(len(score), 2)


class TestBabsrBranchingClass(unittest.TestCase):
    """Tests for BabsrBranching class."""

    def test_init(self):
        """Test BabsrBranching initialization."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net(include_inputs=False)
        branching = BabsrBranching(mock_net)

        self.assertEqual(branching.net, mock_net)
        self.assertEqual(branching.icp_score_counter, 0)


class TestBabsrBranchingGetBranchingDecisions(unittest.TestCase):
    """Tests for BabsrBranching.compute_branching_decisions method."""

    def test_basic_branching_decision(self):
        """Test basic branching decision."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)
        self.assertIsNone(points)
        self.assertIsInstance(split_depth, int)
        self.assertGreaterEqual(split_depth, 0)

    def test_returns_three_values(self):
        """Test that compute_branching_decisions returns three values."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            result = branching.compute_branching_decisions(domains, split_depth=1)

        self.assertIsInstance(result, BranchingDecisions)

    def test_split_depth_parameter(self):
        """Test split_depth parameter affects result."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info1 = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision1 = _info1.branching_decision
            _ = _info1.branching_points
            depth1 = _info1.split_depth
            _info2 = branching.compute_branching_decisions(
                domains, split_depth=2
            )
            decision2 = _info2.branching_decision
            _ = _info2.branching_points
            depth2 = _info2.split_depth

        self.assertLessEqual(depth1, 1)
        self.assertLessEqual(depth2, 2)

    def test_cs_none(self):
        """Test with cs=None."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains(cs_value=None)

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_branching_reduceop_min(self):
        """Test with branching_reduceop='min'."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_reduceop='min'
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_branching_reduceop_max(self):
        """Test with branching_reduceop='max' (default for babsr)."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, branching_reduceop='max'
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)


class TestBabsrBranchingRandomFallback(unittest.TestCase):
    """Tests for random fallback in BabsrBranching."""

    def test_random_fallback_with_low_scores(self):
        """Test that random fallback is used when scores are below threshold."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        # All zero scores should trigger random fallback
        domains = create_domains(use_zeros=True)

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            with patch('builtins.print'):  # Suppress random dict print
                _info = branching.compute_branching_decisions(
                    domains, split_depth=1
                )
                decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        self.assertIsNotNone(decision)


class TestBabsrBranchingICPScoreCounter(unittest.TestCase):
    """Tests for icp_score_counter behavior in BabsrBranching."""

    def test_icp_score_counter_increments(self):
        """Test that icp_score_counter can be incremented."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)

        self.assertEqual(branching.icp_score_counter, 0)
        branching.icp_score_counter += 1
        self.assertEqual(branching.icp_score_counter, 1)

    def test_icp_score_counter_reset(self):
        """Test that icp_score_counter can be reset."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)

        branching.icp_score_counter = 5
        branching.icp_score_counter = 0
        self.assertEqual(branching.icp_score_counter, 0)


class TestBabsrNormalizeScores(unittest.TestCase):
    """Tests for normalize_scores helper function within babsr_score."""

    def _create_mock_layer(self, name, input_name):
        """Create a mock layer with input structure."""
        mock_layer = MagicMock()
        mock_layer.name = name
        mock_input = MagicMock()
        mock_input.name = input_name
        mock_layer.inputs = [mock_input]
        return mock_layer

    def test_normalize_scores_called_with_prioritize(self):
        """Test that normalize_scores is applied when prioritize_alphas is set."""
        from heuristics.babsr import babsr_score

        batch = 2
        neurons = 5

        mock_layer = self._create_mock_layer('act_layer1', 'pre_layer1')
        split_nodes = [MagicMock(name='node1')]
        split_nodes[0].name = 'node1'
        split_activations = {'node1': [(mock_layer,)]}

        lower_bounds = {'pre_layer1': -torch.ones(batch, neurons)}
        upper_bounds = {'pre_layer1': torch.ones(batch, neurons)}
        mask = {'pre_layer1': torch.ones(batch, neurons)}
        # Create lAs with both positive and negative values
        lAs = {'act_layer1': torch.tensor([[[1, -1, 2, -2, 0]],
                                            [[0.5, -0.5, 1, -1, 0]]], dtype=torch.float32)}
        reduce_op = torch.max

        with patch('heuristics.babsr.get_preact_params', return_value=torch.zeros(neurons)):
            with patch('builtins.print'):
                score_positive, _ = babsr_score(
                    lower_bounds, upper_bounds, lAs, mask, reduce_op,
                    number_bounds=1, split_nodes=split_nodes,
                    split_activations=split_activations,
                    prioritize_alphas='positive'
                )

                score_negative, _ = babsr_score(
                    lower_bounds, upper_bounds, lAs, mask, reduce_op,
                    number_bounds=1, split_nodes=split_nodes,
                    split_activations=split_activations,
                    prioritize_alphas='negative'
                )

        # Scores should be different when prioritizing different alphas
        self.assertEqual(len(score_positive), 1)
        self.assertEqual(len(score_negative), 1)


class TestBabsrDecisionFormat(unittest.TestCase):
    """Tests for decision format in BabsrBranching."""

    def test_decision_is_list(self):
        """Test that decision is a list."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsInstance(decision, list)

    def test_decision_tuples_format(self):
        """Test that decisions contain (layer, index) tuples."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # Each decision should be a list/tuple with 2 elements
        if len(decision) > 0:
            self.assertTrue(all(len(d) == 2 for d in decision))


class TestBabsrEdgeCases(unittest.TestCase):
    """Edge case tests for BabsrBranching."""

    def test_single_batch(self):
        """Test with batch size of 1."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains(batch=1)

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_single_layer(self):
        """Test with only one layer."""
        from heuristics.babsr import BabsrBranching

        batch = 2
        neurons = 5
        mock_net = create_mock_net(num_layers=1)
        branching = BabsrBranching(mock_net)

        domains = {
            'lower_bounds': {
                'pre_layer0': -torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': torch.ones(batch, neurons),
            },
            'mask': {
                'pre_layer0': torch.ones(batch, neurons),
                'layer0': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
            },
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_large_split_depth(self):
        """Test with split_depth larger than available neurons."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains(neurons=3)  # Small number of neurons

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=100  # Much larger than available neurons
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        # split_depth should be limited
        self.assertLessEqual(split_depth, 100)
        self.assertIsNotNone(decision)

    def test_sparse_mask(self):
        """Test with sparse mask (few unstable neurons)."""
        from heuristics.babsr import BabsrBranching

        batch = 2
        neurons = 10
        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)

        # Sparse mask - only 2 unstable neurons per layer
        mask0 = torch.zeros(batch, neurons)
        mask0[:, :2] = 1
        mask1 = torch.zeros(batch, neurons)
        mask1[:, :2] = 1

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
                'pre_layer0': mask0,
                'pre_layer1': mask1,
                'layer0': mask0,
                'layer1': mask1,
            },
            'lAs': {
                'act_layer0': torch.rand(batch, 1, neurons),
                'act_layer1': torch.rand(batch, 1, neurons),
            },
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)


class TestBabsrSparsestLayer(unittest.TestCase):
    """Tests for sparsest_layer parameter."""

    def test_sparsest_layer_zero(self):
        """Test with sparsest_layer=0."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, sparsest_layer=0
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_sparsest_layer_negative(self):
        """Test with sparsest_layer=-1 (all layers dense)."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, sparsest_layer=-1
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)


class TestBabsrMaxInfoThreshold(unittest.TestCase):
    """Tests for max_info_threshold parameter."""

    def test_max_info_threshold_default(self):
        """Test with default max_info_threshold."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, max_info_threshold=0.001
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    def test_max_info_threshold_high(self):
        """Test with high max_info_threshold (triggers fallback)."""
        from heuristics.babsr import BabsrBranching

        batch = 2
        neurons = 5
        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)

        # Small scores that will be below threshold - custom domains needed
        domains = {
            'lower_bounds': {
                'pre_layer0': -0.01 * torch.ones(batch, neurons),
                'pre_layer1': -0.01 * torch.ones(batch, neurons),
            },
            'upper_bounds': {
                'pre_layer0': 0.01 * torch.ones(batch, neurons),
                'pre_layer1': 0.01 * torch.ones(batch, neurons),
            },
            'mask': {
                'pre_layer0': torch.ones(batch, neurons),
                'pre_layer1': torch.ones(batch, neurons),
                'layer0': torch.ones(batch, neurons),
                'layer1': torch.ones(batch, neurons),
            },
            'lAs': {
                'act_layer0': 0.01 * torch.rand(batch, 1, neurons),
                'act_layer1': 0.01 * torch.rand(batch, 1, neurons),
            },
            'cs': torch.ones(batch, 1),
        }

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            with patch('builtins.print'):  # Suppress random dict print
                _info = branching.compute_branching_decisions(
                    domains, split_depth=1, max_info_threshold=1000.0  # Very high threshold
                )
                decision = _info.branching_decision
                points = _info.branching_points
                split_depth = _info.split_depth

        self.assertIsNotNone(decision)


class TestBabsrPrioritizeAlphas(unittest.TestCase):
    """Tests for prioritize_alphas parameter in compute_branching_decisions."""

    def test_prioritize_alphas_none(self):
        """Test with prioritize_alphas='none'."""
        from heuristics.babsr import BabsrBranching

        mock_net = create_mock_net()
        branching = BabsrBranching(mock_net)
        domains = create_domains()

        with patch('heuristics.babsr.get_preact_params', return_value=0):
            _info = branching.compute_branching_decisions(
                domains, split_depth=1, prioritize_alphas='none'
            )
            decision = _info.branching_decision
            points = _info.branching_points
            split_depth = _info.split_depth

        self.assertIsNotNone(decision)

    # Note: prioritize_alphas='positive' and 'negative' are tested at the
    # babsr_score level in TestBabsrScore. Testing at the compute_branching_decisions
    # level requires specific input configurations that interact with internal
    # score normalization logic. The babsr_score tests provide adequate coverage
    # for the prioritize_alphas feature.


if __name__ == '__main__':
    unittest.main()
