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
"""Unit tests for heuristics/base.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heuristics.decision_types import BranchingDecisions


class TestNeuronBranchingHeuristic(unittest.TestCase):
    """Tests for NeuronBranchingHeuristic base class."""

    def _create_mock_net(self):
        """Create a mock network for testing."""
        mock_net = MagicMock()
        mock_net.split_activations = {
            'layer1': MagicMock(),
            'layer2': MagicMock()
        }
        mock_net.split_nodes = []
        return mock_net

    def test_init(self):
        """Test initialization of NeuronBranchingHeuristic."""
        from heuristics.base import NeuronBranchingHeuristic
        mock_net = self._create_mock_net()
        heuristic = NeuronBranchingHeuristic(mock_net)
        self.assertEqual(heuristic.net, mock_net)
        self.assertIsNone(heuristic.batch_size)
        self.assertIsNone(heuristic.device)

    def test_layer_iterator_dict(self):
        """Test layer_iterator with dictionary input."""
        from heuristics.base import NeuronBranchingHeuristic
        mock_net = self._create_mock_net()
        heuristic = NeuronBranchingHeuristic(mock_net)

        bounds = {
            'layer2': torch.randn(2, 10),
            'layer1': torch.randn(2, 10),
            'layer3': torch.randn(2, 10),  # Not in split_activations
        }

        results = list(heuristic.layer_iterator(bounds))
        # Should only include layers in split_activations and be sorted
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 'layer1')
        self.assertEqual(results[1][0], 'layer2')

    def test_update_batch_size_and_device(self):
        """Test update_batch_size_and_device method."""
        from heuristics.base import NeuronBranchingHeuristic
        mock_net = self._create_mock_net()
        heuristic = NeuronBranchingHeuristic(mock_net)

        bounds = {
            'layer1': torch.randn(8, 10),  # batch_size = 8
        }

        heuristic.update_batch_size_and_device(bounds)
        self.assertEqual(heuristic.batch_size, 8)
        self.assertEqual(heuristic.device.type, 'cpu')

    def test_update_batch_size_cuda(self):
        """Test update_batch_size_and_device with different device."""
        from heuristics.base import NeuronBranchingHeuristic
        mock_net = self._create_mock_net()
        heuristic = NeuronBranchingHeuristic(mock_net)

        bounds = {
            'layer1': torch.randn(4, 5),
        }

        heuristic.update_batch_size_and_device(bounds)
        self.assertEqual(heuristic.batch_size, 4)

    def test_find_topk_scores_basic(self):
        """Test find_topk_scores method."""
        from heuristics.base import NeuronBranchingHeuristic

        mock_net = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.name = 'layer1'
        mock_node2 = MagicMock()
        mock_node2.name = 'layer2'
        mock_net.split_nodes = [mock_node1, mock_node2]
        mock_net.split_activations = {'layer1': MagicMock(), 'layer2': MagicMock()}

        heuristic = NeuronBranchingHeuristic(mock_net)
        heuristic.batch_size = 2
        heuristic.device = 'cpu'

        layer_scores = {
            'layer1': torch.tensor([[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]]),
            'layer2': torch.tensor([[0.7, 0.2], [0.1, 0.3]]),
        }
        split_masks = {
            'layer1': torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
            'layer2': torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        }

        topk_layers, topk_indices = heuristic.find_topk_scores(
            layer_scores, split_masks, k=2
        )

        self.assertEqual(topk_layers.shape, (2, 2))
        self.assertEqual(topk_indices.shape, (2, 2))

    def test_find_topk_scores_return_scores(self):
        """Test find_topk_scores with return_scores=True."""
        from heuristics.base import NeuronBranchingHeuristic

        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}

        heuristic = NeuronBranchingHeuristic(mock_net)
        heuristic.batch_size = 1
        heuristic.device = 'cpu'

        layer_scores = {
            'layer1': torch.tensor([[0.5, 0.3, 0.7]]),
        }
        split_masks = {
            'layer1': torch.tensor([[1.0, 1.0, 1.0]]),
        }

        result = heuristic.find_topk_scores(
            layer_scores, split_masks, k=2, return_scores=True
        )

        # When return_scores=True, should return 3 values
        self.assertEqual(len(result), 3)

    def test_format_decisions(self):
        """Test format_decisions method."""
        from heuristics.base import NeuronBranchingHeuristic

        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}

        heuristic = NeuronBranchingHeuristic(mock_net)
        heuristic.batch_size = 2
        heuristic.device = 'cpu'

        topk_layers = torch.tensor([[0, 0], [0, 0]])
        topk_indices = torch.tensor([[1, 2], [3, 4]])

        result = heuristic.format_decisions(topk_layers, topk_indices)

        # Should return a BranchingDecisions
        self.assertIsInstance(result, BranchingDecisions)
        # branching_decision should be a list of decisions
        self.assertIsInstance(result.branching_decision, list)


class TestNeuronBranchingHeuristicLayerIterator(unittest.TestCase):
    """Additional tests for layer_iterator."""

    def test_empty_bounds(self):
        """Test layer_iterator with empty bounds."""
        from heuristics.base import NeuronBranchingHeuristic

        mock_net = MagicMock()
        mock_net.split_activations = {'layer1': MagicMock()}

        heuristic = NeuronBranchingHeuristic(mock_net)

        bounds = {}
        results = list(heuristic.layer_iterator(bounds))
        self.assertEqual(len(results), 0)

    def test_no_matching_layers(self):
        """Test layer_iterator when no layers match split_activations."""
        from heuristics.base import NeuronBranchingHeuristic

        mock_net = MagicMock()
        mock_net.split_activations = {'layer1': MagicMock()}

        heuristic = NeuronBranchingHeuristic(mock_net)

        bounds = {
            'layer2': torch.randn(2, 10),
            'layer3': torch.randn(2, 10),
        }
        results = list(heuristic.layer_iterator(bounds))
        self.assertEqual(len(results), 0)


if __name__ == '__main__':
    unittest.main()
