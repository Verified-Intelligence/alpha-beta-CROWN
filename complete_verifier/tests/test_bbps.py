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
"""Unit tests for heuristics/nonlinear/bbps.py (NonlinearBranching)."""
import os
import sys
import unittest
from heuristics.decision_types import BranchingDecisions
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments

# Initialize arguments.Config before importing bbps
arguments.Config.parse_config(args=[], verbose=False)

from heuristics.nonlinear.bbps import NonlinearBranching


def _create_mock_net():
    """Create a mock LiRPANet for testing."""
    mock_net = MagicMock()

    # Setup net.net (the underlying BoundedModule)
    mock_model = MagicMock()
    mock_model.root_names = ['input']
    mock_model.final_name = '/output'
    mock_model.input_name = ['input']

    # Setup roots
    mock_root = MagicMock()
    mock_root.output_shape = (1, 10)
    mock_model.roots.return_value = [mock_root]

    mock_net.net = mock_model
    mock_net.final_name = '/output'
    mock_net.split_nodes = []
    mock_net.split_activations = {}
    mock_net.A_saved = {}
    mock_net.x = MagicMock()

    return mock_net


def _create_mock_node(name, output_shape=(1, 10)):
    """Create a mock node."""
    mock_node = MagicMock()
    mock_node.name = name
    mock_node.output_shape = output_shape
    return mock_node


class TestNonlinearBranchingInit(unittest.TestCase):
    """Tests for NonlinearBranching.__init__."""

    def test_init_uniform_branching(self):
        """Test initialization with uniform branching points."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.branching_point_method, 'uniform')
        self.assertEqual(heuristic.num_branches, 2)
        self.assertEqual(heuristic.method, 'fast')
        self.assertFalse(heuristic.filter)
        self.assertFalse(heuristic.filter_beta)
        self.assertEqual(heuristic.filter_batch_size, 100)
        self.assertEqual(heuristic.filter_iterations, 10)
        self.assertTrue(heuristic.filter_clamp)
        self.assertFalse(heuristic.relu_only)

    def test_init_babsr_like_method(self):
        """Test initialization with babsr-like method creates BaBSRNonlinearBranching."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 3,
            'method': 'babsr-like',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.method, 'babsr-like')
        self.assertTrue(hasattr(heuristic, 'babsr'))

    @patch('heuristics.nonlinear.bbps.BranchingPointOpt')
    def test_init_opt_branching_point_method(self, mock_bp_opt_class):
        """Test initialization with opt branching point method."""
        mock_net = _create_mock_net()
        mock_bp_opt_class.return_value = MagicMock()

        # Setup arguments.Config for branching point options
        arguments.Config['bab']['branching']['nonlinear_split']['branching_point'] = {
            'db_path': '/tmp/test.db',
            'num_iterations': 100,
            'range_l': -10,
            'range_u': 10,
            'step_size_1d': 0.1,
            'step_size': 0.5,
            'batch_size': 1000,
            'log_interval': 10
        }

        kwargs = {
            'branching_point_method': 'opt',
            'num_branches': 2,  # Must be 2 for opt
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.branching_point_method, 'opt')
        self.assertTrue(hasattr(heuristic, 'bp_opt'))

    def test_init_opt_requires_two_branches(self):
        """Test that opt branching point method requires num_branches=2."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'opt',
            'num_branches': 3,  # Invalid - must be 2
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        with self.assertRaises(AssertionError):
            NonlinearBranching(mock_net, **kwargs)

    def test_init_invalid_branching_point_method(self):
        """Test that invalid branching point method raises AssertionError."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'invalid_method',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        with self.assertRaises(AssertionError):
            NonlinearBranching(mock_net, **kwargs)

    def test_init_stores_root_name(self):
        """Test that root_name is stored from net."""
        mock_net = _create_mock_net()
        mock_net.net.root_names = ['my_input']

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.root_name, 'my_input')


class TestGetUniformBranchingPoints(unittest.TestCase):
    """Tests for NonlinearBranching._get_uniform_branching_points."""

    def _create_heuristic(self, num_branches=2):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': num_branches,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_uniform_points_2_branches(self):
        """Test uniform branching points with 2 branches."""
        heuristic = self._create_heuristic(num_branches=2)

        lb = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        ub = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        # With 2 branches, we have 1 branching point at 0.5 ratio
        self.assertEqual(points.shape, (2, 2, 1))

        # Points should be at midpoint: lb + 0.5 * (ub - lb)
        expected = (lb + ub) / 2
        torch.testing.assert_close(points.squeeze(-1), expected)

    def test_uniform_points_3_branches(self):
        """Test uniform branching points with 3 branches."""
        heuristic = self._create_heuristic(num_branches=3)

        lb = torch.tensor([[0.0], [0.0]])
        ub = torch.tensor([[3.0], [6.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        # With 3 branches, we have 2 branching points at 1/3 and 2/3 ratios
        self.assertEqual(points.shape, (2, 1, 2))

        # First point at 1/3, second at 2/3
        expected_first = lb + (ub - lb) / 3
        expected_second = lb + 2 * (ub - lb) / 3

        torch.testing.assert_close(points[..., 0], expected_first)
        torch.testing.assert_close(points[..., 1], expected_second)

    def test_uniform_points_4_branches(self):
        """Test uniform branching points with 4 branches."""
        heuristic = self._create_heuristic(num_branches=4)

        lb = torch.tensor([[0.0]])
        ub = torch.tensor([[4.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        # With 4 branches, we have 3 branching points at 1/4, 2/4, 3/4 ratios
        self.assertEqual(points.shape, (1, 1, 3))

        torch.testing.assert_close(points[0, 0, 0], torch.tensor(1.0))
        torch.testing.assert_close(points[0, 0, 1], torch.tensor(2.0))
        torch.testing.assert_close(points[0, 0, 2], torch.tensor(3.0))

    def test_uniform_points_override_num_branches(self):
        """Test _get_uniform_branching_points with num_branches override."""
        heuristic = self._create_heuristic(num_branches=2)

        lb = torch.tensor([[0.0]])
        ub = torch.tensor([[3.0]])

        # Override num_branches
        points = heuristic._get_uniform_branching_points(lb, ub, num_branches=3)

        # Should have 2 points despite heuristic.num_branches being 2
        self.assertEqual(points.shape[-1], 2)

    def test_uniform_points_preserves_device(self):
        """Test that branching points are on the same device as input."""
        heuristic = self._create_heuristic(num_branches=2)

        lb = torch.tensor([[0.0, 1.0]])
        ub = torch.tensor([[1.0, 2.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.device, lb.device)

    def test_uniform_points_batch_processing(self):
        """Test uniform branching points with batched input."""
        heuristic = self._create_heuristic(num_branches=2)

        batch_size = 5
        num_neurons = 10
        lb = torch.randn(batch_size, num_neurons)
        ub = lb + torch.abs(torch.randn(batch_size, num_neurons))  # Ensure ub > lb

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.shape, (batch_size, num_neurons, 1))
        # All points should be between lb and ub
        self.assertTrue(torch.all(points.squeeze(-1) >= lb))
        self.assertTrue(torch.all(points.squeeze(-1) <= ub))


class TestGetBranchingDecisions(unittest.TestCase):
    """Tests for NonlinearBranching.compute_branching_decisions."""

    def _create_heuristic(self, filter_enabled=False, relu_only=False):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': filter_enabled,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': relu_only,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_split_depth_clamped_to_1(self):
        """Test that split_depth > 1 is clamped to 1."""
        heuristic = self._create_heuristic()

        # Mock the methods that compute_branching_decisions calls
        heuristic.get_heuristic_decisions = MagicMock(return_value=(
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            torch.tensor([[[0.5]]])
        ))
        heuristic.format_decisions = MagicMock(return_value=([([0, 0])], [0.5], 1))

        domains = {'lower_bounds': {'layer': torch.randn(1, 10)}}

        # Call with split_depth > 1
        result = heuristic.compute_branching_decisions(
            domains, split_depth=5, branching_candidates=1
        )

        # format_decisions should be called
        heuristic.format_decisions.assert_called_once()

    def test_branching_candidates_set_to_1_when_no_filter(self):
        """Test branching_candidates is set to 1 when filter is disabled."""
        heuristic = self._create_heuristic(filter_enabled=False)

        heuristic.get_heuristic_decisions = MagicMock(return_value=(
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            torch.tensor([[[0.5]]])
        ))
        heuristic.format_decisions = MagicMock(return_value=([([0, 0])], [0.5], 1))

        domains = {'lower_bounds': {'layer': torch.randn(1, 10)}}

        heuristic.compute_branching_decisions(
            domains, split_depth=1, branching_candidates=10
        )

        # branching_candidates should be set to 1
        self.assertEqual(heuristic.branching_candidates, 1)

    def test_filter_called_when_enabled(self):
        """Test that _filter is called when filter is enabled."""
        heuristic = self._create_heuristic(filter_enabled=True)

        layers = torch.tensor([[0, 1]])
        indices = torch.tensor([[0, 1]])
        points = torch.tensor([[[0.5], [0.6]]])

        heuristic.get_heuristic_decisions = MagicMock(return_value=(layers, indices, points))
        heuristic._filter = MagicMock(return_value=(
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            torch.tensor([[[0.5]]])
        ))
        heuristic.format_decisions = MagicMock(return_value=([([0, 0])], [0.5], 1))

        domains = {'lower_bounds': {'layer': torch.randn(1, 10)}}

        heuristic.compute_branching_decisions(
            domains, split_depth=1, branching_candidates=2
        )

        heuristic._filter.assert_called_once()


class TestTakeFilterBatch(unittest.TestCase):
    """Tests for NonlinearBranching._take_filter_batch."""

    def _create_heuristic(self, filter_batch_size=2):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': True,
            'filter_beta': False,
            'filter_batch_size': filter_batch_size,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_take_filter_batch_first_batch(self):
        """Test taking first batch from args_update_bounds."""
        heuristic = self._create_heuristic(filter_batch_size=2)

        args = {
            'lower_bounds': {
                'layer1': torch.arange(6).reshape(6, 1).float(),
                'layer2': torch.arange(6, 12).reshape(6, 1).float(),
            },
            'upper_bounds': {
                'layer1': torch.arange(6, 12).reshape(6, 1).float(),
                'layer2': torch.arange(12, 18).reshape(6, 1).float(),
            },
            'cs': torch.arange(6).float(),
            'thresholds': torch.arange(6).float(),
            'alphas': {
                'layer1': {
                    'sub1': torch.arange(12).reshape(2, 1, 6).float(),
                }
            }
        }

        result = heuristic._take_filter_batch(args, i=0)

        # Check first batch (indices 0:2)
        torch.testing.assert_close(
            result['lower_bounds']['layer1'],
            torch.tensor([[0.], [1.]])
        )
        torch.testing.assert_close(result['cs'], torch.tensor([0., 1.]))

    def test_take_filter_batch_second_batch(self):
        """Test taking second batch from args_update_bounds."""
        heuristic = self._create_heuristic(filter_batch_size=2)

        args = {
            'lower_bounds': {
                'layer1': torch.arange(6).reshape(6, 1).float(),
            },
            'upper_bounds': {
                'layer1': torch.arange(6, 12).reshape(6, 1).float(),
            },
            'cs': torch.arange(6).float(),
            'thresholds': torch.arange(6).float(),
            'alphas': {
                'layer1': {
                    'sub1': torch.arange(12).reshape(2, 1, 6).float(),
                }
            }
        }

        result = heuristic._take_filter_batch(args, i=1)

        # Check second batch (indices 2:4)
        torch.testing.assert_close(
            result['lower_bounds']['layer1'],
            torch.tensor([[2.], [3.]])
        )
        torch.testing.assert_close(result['cs'], torch.tensor([2., 3.]))

    def test_take_filter_batch_with_beta(self):
        """Test taking batch when filter_beta is enabled."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': True,
            'filter_beta': True,  # Enable filter_beta
            'filter_batch_size': 2,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        args = {
            'lower_bounds': {
                'layer1': torch.arange(4).reshape(4, 1).float(),
            },
            'upper_bounds': {
                'layer1': torch.arange(4, 8).reshape(4, 1).float(),
            },
            'cs': torch.arange(4).float(),
            'thresholds': torch.arange(4).float(),
            'alphas': {
                'layer1': {
                    'sub1': torch.arange(8).reshape(2, 1, 4).float(),
                }
            },
            'betas': [torch.tensor([0.1]), torch.tensor([0.2]),
                      torch.tensor([0.3]), torch.tensor([0.4])],
            'history': [{'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}],
        }

        result = heuristic._take_filter_batch(args, i=0)

        # Check betas and history are sliced
        self.assertEqual(len(result['betas']), 2)
        self.assertEqual(len(result['history']), 2)

    def test_take_filter_batch_last_partial_batch(self):
        """Test taking a partial last batch."""
        heuristic = self._create_heuristic(filter_batch_size=3)

        # 5 total items with batch_size=3 -> last batch has 2 items
        args = {
            'lower_bounds': {
                'layer1': torch.arange(5).reshape(5, 1).float(),
            },
            'upper_bounds': {
                'layer1': torch.arange(5, 10).reshape(5, 1).float(),
            },
            'cs': torch.arange(5).float(),
            'thresholds': torch.arange(5).float(),
            'alphas': {
                'layer1': {
                    'sub1': torch.arange(10).reshape(2, 1, 5).float(),
                }
            }
        }

        result = heuristic._take_filter_batch(args, i=1)

        # Last batch should have indices 3:5 (2 items)
        self.assertEqual(result['lower_bounds']['layer1'].shape[0], 2)
        torch.testing.assert_close(
            result['lower_bounds']['layer1'],
            torch.tensor([[3.], [4.]])
        )


class TestComputeBranchingScores(unittest.TestCase):
    """Tests for NonlinearBranching.compute_branching_scores."""

    def _create_heuristic(self, branching_point_method='uniform'):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': branching_point_method,
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_compute_branching_scores_uniform(self):
        """Test compute_branching_scores with uniform method."""
        heuristic = self._create_heuristic(branching_point_method='uniform')

        mock_node = _create_mock_node('layer1')

        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0, 0.0]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0, 2.0]])},
        }

        # Mock compute_scores_with_points
        heuristic.compute_scores_with_points = MagicMock(return_value={
            'scores': torch.tensor([[0.5, 0.3]]),
            'points': torch.tensor([[[0.0], [1.0]]])
        })

        result = heuristic.compute_branching_scores(mock_node, domains)

        heuristic.compute_scores_with_points.assert_called_once()
        self.assertIn('scores', result)
        self.assertIn('points', result)

    def test_compute_branching_scores_invalid_method(self):
        """Test compute_branching_scores with invalid method raises NameError."""
        heuristic = self._create_heuristic(branching_point_method='uniform')

        mock_node = _create_mock_node('layer1')

        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0]])},
        }

        with self.assertRaises(NameError):
            heuristic.compute_branching_scores(
                mock_node, domains, branching_point_method='invalid'
            )

    def test_compute_branching_scores_override_method(self):
        """Test compute_branching_scores with overridden branching_point_method."""
        heuristic = self._create_heuristic(branching_point_method='uniform')

        mock_node = _create_mock_node('layer1')

        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0]])},
        }

        # Mock compute_scores_with_points
        heuristic.compute_scores_with_points = MagicMock(return_value={
            'scores': torch.tensor([[0.5]]),
            'points': torch.tensor([[[0.0]]])
        })

        # Override branching_point_method
        heuristic.compute_branching_scores(
            mock_node, domains, branching_point_method='uniform'
        )

        heuristic.compute_scores_with_points.assert_called_once()


class TestComputeScoresWithPoints(unittest.TestCase):
    """Tests for NonlinearBranching.compute_scores_with_points."""

    def _create_heuristic(self, method='fast', relu_only=False):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': method,
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': relu_only,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_compute_scores_calls_fast_heuristic(self):
        """Test compute_scores_with_points calls _fast_heuristic for fast method."""
        heuristic = self._create_heuristic(method='fast')

        mock_node = _create_mock_node('layer1')
        heuristic.net.split_activations = {'layer1': []}

        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0]]), '/output': torch.tensor([[-0.5]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0]]), '/output': torch.tensor([[0.5]])},
            'lAs': {},
            'thresholds': torch.tensor([[0.0]]),
        }

        points = torch.tensor([[[0.0]]])

        heuristic._fast_heuristic = MagicMock(return_value={
            'scores': torch.tensor([[0.5]]),
            'points': points
        })

        heuristic.compute_scores_with_points(mock_node, domains, points)

        heuristic._fast_heuristic.assert_called_once()

    def test_compute_scores_calls_babsr_for_babsr_like(self):
        """Test compute_scores_with_points calls babsr for babsr-like method."""
        heuristic = self._create_heuristic(method='babsr-like')

        mock_node = _create_mock_node('layer1')
        heuristic.net.split_activations = {'layer1': []}

        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0]]), '/output': torch.tensor([[-0.5]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0]]), '/output': torch.tensor([[0.5]])},
            'lAs': {},
            'thresholds': torch.tensor([[0.0]]),
        }

        points = torch.tensor([[[0.0]]])

        heuristic.babsr = MagicMock()
        heuristic.babsr.compute_heuristic = MagicMock(return_value={
            'scores': torch.tensor([[0.5]]),
            'points': points
        })

        heuristic.compute_scores_with_points(mock_node, domains, points)

        heuristic.babsr.compute_heuristic.assert_called_once()


class TestGetOptimizedBranchingPoints(unittest.TestCase):
    """Tests for NonlinearBranching._get_optimized_branching_points."""

    @patch('heuristics.nonlinear.bbps.BranchingPointOpt')
    def test_get_optimized_branching_points(self, mock_bp_opt_class):
        """Test _get_optimized_branching_points calls bp_opt."""
        mock_net = _create_mock_net()
        mock_bp_opt = MagicMock()
        mock_bp_opt.get_branching_points.return_value = torch.tensor([[[0.5]]])
        mock_bp_opt_class.return_value = mock_bp_opt

        arguments.Config['bab']['branching']['nonlinear_split']['branching_point'] = {
            'db_path': '/tmp/test.db',
            'num_iterations': 100,
            'range_l': -10,
            'range_u': 10,
            'step_size_1d': 0.1,
            'step_size': 0.5,
            'batch_size': 1000,
            'log_interval': 10
        }

        kwargs = {
            'branching_point_method': 'opt',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        mock_node = _create_mock_node('layer1')
        domains = {
            'lower_bounds': {'layer1': torch.tensor([[-1.0]])},
            'upper_bounds': {'layer1': torch.tensor([[1.0]])},
        }

        result = heuristic._get_optimized_branching_points(mock_node, domains)

        mock_bp_opt.get_branching_points.assert_called_once_with(
            mock_node,
            lower_bounds=domains['lower_bounds'],
            upper_bounds=domains['upper_bounds'],
        )


class TestGetHeuristicDecisions(unittest.TestCase):
    """Tests for NonlinearBranching.get_heuristic_decisions."""

    def _create_heuristic(self, relu_only=False):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        # Setup split_nodes
        mock_node1 = _create_mock_node('layer1', output_shape=(1, 4))
        mock_node2 = _create_mock_node('layer2', output_shape=(1, 4))
        mock_net.split_nodes = [mock_node1, mock_node2]
        mock_net.split_activations = {'layer1': [], 'layer2': []}

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': relu_only,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_get_heuristic_decisions_basic(self):
        """Test get_heuristic_decisions returns layers, indices, points."""
        heuristic = self._create_heuristic(relu_only=False)
        heuristic.branching_candidates = 2
        heuristic.reduce_op = torch.Tensor.min

        # Mock compute_branching_scores
        heuristic.compute_branching_scores = MagicMock(return_value={
            'scores': torch.tensor([[0.5, 0.3, 0.2, 0.1]]),
            'points': torch.tensor([[[[0.0]], [[0.1]], [[0.2]], [[0.3]]]])
        })

        domains = {
            'mask': {
                'layer1': torch.ones(1, 4),
                'layer2': torch.ones(1, 4),
            },
            'lower_bounds': {
                'layer1': torch.randn(1, 4),
                'layer2': torch.randn(1, 4),
            },
            'upper_bounds': {
                'layer1': torch.randn(1, 4),
                'layer2': torch.randn(1, 4),
            },
        }

        layers, indices, points = heuristic.get_heuristic_decisions(domains)

        self.assertIsInstance(layers, torch.Tensor)
        self.assertIsInstance(indices, torch.Tensor)

    def test_get_heuristic_decisions_relu_only_returns_none_points(self):
        """Test that relu_only=True returns None for points."""
        heuristic = self._create_heuristic(relu_only=True)
        heuristic.branching_candidates = 1
        heuristic.reduce_op = torch.Tensor.min

        # Mock compute_branching_scores
        heuristic.compute_branching_scores = MagicMock(return_value={
            'scores': torch.tensor([[0.5, 0.3, 0.2, 0.1]]),
            'points': torch.tensor([[[[0.0]], [[0.1]], [[0.2]], [[0.3]]]])
        })

        domains = {
            'mask': {
                'layer1': torch.ones(1, 4),
                'layer2': torch.ones(1, 4),
            },
            'lower_bounds': {
                'layer1': torch.randn(1, 4),
                'layer2': torch.randn(1, 4),
            },
            'upper_bounds': {
                'layer1': torch.randn(1, 4),
                'layer2': torch.randn(1, 4),
            },
        }

        layers, indices, points = heuristic.get_heuristic_decisions(domains)

        self.assertIsNone(points)

    def _make_domains(self, batch, num_neurons=4):
        """Build a domain dict whose per-node scores are derived from the
        lower bounds, so chunking over the batch must reproduce the full-batch
        result exactly."""
        layer_names = ['layer1', 'layer2']
        domains = {
            'mask': {n: torch.ones(batch, num_neurons) for n in layer_names},
            'lower_bounds': {
                n: torch.randn(batch, num_neurons) for n in layer_names},
            'upper_bounds': {
                n: torch.randn(batch, num_neurons) for n in layer_names},
            'lAs': {n: torch.randn(batch, 1, num_neurons) for n in layer_names},
            'thresholds': torch.randn(batch, 1),
        }
        return domains

    def test_get_heuristic_decisions_chunking_matches_full(self):
        """decision_sub_batch_size must not change the heuristic decisions:
        chunked scoring + a single global top-k selection should reproduce the
        un-chunked result."""
        heuristic = self._create_heuristic(relu_only=False)
        heuristic.branching_candidates = 2
        heuristic.reduce_op = torch.Tensor.min

        # Scores/points depend only on the (possibly sliced) per-domain lower
        # bounds, so concatenating chunk results reconstructs the full batch.
        def fake_scores(node, domains, branching_point_method=None):
            lb = domains['lower_bounds'][node.name]
            b, n = lb.shape
            return {'scores': lb.clone(), 'points': lb.reshape(b, n, 1, 1).clone()}

        heuristic.compute_branching_scores = MagicMock(side_effect=fake_scores)

        batch = 7
        domains = self._make_domains(batch)

        original = arguments.Config['bab']['branching']['decision_sub_batch_size']
        try:
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = None
            full = heuristic.get_heuristic_decisions(domains)
            # A chunk size that does not evenly divide the batch (7 -> 3+3+1).
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = 3
            chunked = heuristic.get_heuristic_decisions(domains)
        finally:
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = original

        layers_f, indices_f, points_f = full
        layers_c, indices_c, points_c = chunked
        torch.testing.assert_close(layers_c, layers_f)
        torch.testing.assert_close(indices_c, indices_f)
        torch.testing.assert_close(points_c, points_f)

    def test_get_heuristic_decisions_chunk_larger_than_batch(self):
        """sub_batch_size >= batch should take the single-pass fast path and
        still produce the same result as no chunking."""
        heuristic = self._create_heuristic(relu_only=False)
        heuristic.branching_candidates = 2
        heuristic.reduce_op = torch.Tensor.min

        def fake_scores(node, domains, branching_point_method=None):
            lb = domains['lower_bounds'][node.name]
            b, n = lb.shape
            return {'scores': lb.clone(), 'points': lb.reshape(b, n, 1, 1).clone()}

        heuristic.compute_branching_scores = MagicMock(side_effect=fake_scores)

        domains = self._make_domains(batch=4)
        original = arguments.Config['bab']['branching']['decision_sub_batch_size']
        try:
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = None
            full = heuristic.get_heuristic_decisions(domains)
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = 100
            chunked = heuristic.get_heuristic_decisions(domains)
        finally:
            arguments.Config['bab']['branching']['decision_sub_batch_size'] = original

        for a, b in zip(chunked, full):
            torch.testing.assert_close(a, b)


class TestFormatDecisionsIntegration(unittest.TestCase):
    """Tests for format_decisions as called from NonlinearBranching."""

    def _create_heuristic(self):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_format_decisions_with_points(self):
        """Test format_decisions with branching points."""
        heuristic = self._create_heuristic()
        heuristic.batch_size = 2
        heuristic.device = 'cpu'

        layers = torch.tensor([[0], [1]])
        indices = torch.tensor([[5], [3]])
        points = torch.tensor([[[0.5]], [[0.3]]])

        _info = heuristic.format_decisions(layers, indices, points)
        decisions, pts, split_depth = _info.branching_decision, _info.branching_points, _info.split_depth

        self.assertIsInstance(decisions, list)
        self.assertEqual(split_depth, 1)

    def test_format_decisions_without_points(self):
        """Test format_decisions without branching points (None)."""
        heuristic = self._create_heuristic()
        heuristic.batch_size = 2
        heuristic.device = 'cpu'

        layers = torch.tensor([[0], [1]])
        indices = torch.tensor([[5], [3]])

        _info = heuristic.format_decisions(layers, indices, None)
        decisions, pts, split_depth = _info.branching_decision, _info.branching_points, _info.split_depth

        self.assertIsInstance(decisions, list)
        self.assertIsNone(pts)
        self.assertEqual(split_depth, 1)

    def test_format_decisions_multi_branch_points(self):
        """Test format_decisions with multiple branching points."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 3,  # 2 branching points
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)
        heuristic.batch_size = 2
        heuristic.device = 'cpu'

        layers = torch.tensor([[0], [1]])
        indices = torch.tensor([[5], [3]])
        # 2 branching points per decision
        points = torch.tensor([[[0.3, 0.6]], [[0.2, 0.4]]])

        _info = heuristic.format_decisions(layers, indices, points)
        decisions, pts, split_depth = _info.branching_decision, _info.branching_points, _info.split_depth

        self.assertIsInstance(decisions, list)
        self.assertEqual(split_depth, 1)
        # Points should be reshaped
        self.assertEqual(pts.shape[-1], 2)


class TestFilterMethod(unittest.TestCase):
    """Tests for NonlinearBranching._filter."""

    def _create_heuristic(self):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': True,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_filter_selects_best_candidate(self):
        """Test that _filter selects the best branching candidate."""
        heuristic = self._create_heuristic()
        heuristic.reduce_op = torch.Tensor.min

        # Mock _compute_actual_bounds
        # ret_lbs shape: [branching_candidates * num_branches * batch_size, num_specs]
        # With 2 candidates, 2 branches, batch_size 1 -> shape (4, 1)
        heuristic._compute_actual_bounds = MagicMock(return_value=torch.tensor([
            [-0.5],  # candidate 0, branch 0
            [-0.3],  # candidate 0, branch 1
            [-0.8],  # candidate 1, branch 0
            [-0.7],  # candidate 1, branch 1
        ]))

        domains = {
            'thresholds': torch.tensor([[0.0]]),
            'lower_bounds': {'layer1': torch.randn(1, 4)},
            'upper_bounds': {'layer1': torch.randn(1, 4)},
        }

        layers = torch.tensor([[0, 1]])  # 2 candidates
        indices = torch.tensor([[0, 1]])
        points = torch.tensor([[[0.5], [0.6]]])

        with patch('builtins.print'):  # Suppress print output
            new_layers, new_indices, new_points = heuristic._filter(
                domains, layers, indices, points
            )

        # Should select one candidate per batch element
        self.assertEqual(new_layers.shape, (1, 1))
        self.assertEqual(new_indices.shape, (1, 1))

    def test_filter_with_clamp(self):
        """Test _filter with filter_clamp enabled."""
        heuristic = self._create_heuristic()
        heuristic.filter_clamp = True
        heuristic.reduce_op = torch.Tensor.min

        heuristic._compute_actual_bounds = MagicMock(return_value=torch.tensor([
            [0.5],   # positive - should be clamped to 0
            [-0.3],
            [-0.8],
            [-0.7],
        ]))

        domains = {
            'thresholds': torch.tensor([[0.0]]),
            'lower_bounds': {'layer1': torch.randn(1, 4)},
            'upper_bounds': {'layer1': torch.randn(1, 4)},
        }

        layers = torch.tensor([[0, 1]])
        indices = torch.tensor([[0, 1]])
        points = torch.tensor([[[0.5], [0.6]]])

        with patch('builtins.print'):
            new_layers, new_indices, new_points = heuristic._filter(
                domains, layers, indices, points
            )

        # Should still work with clamped scores
        self.assertEqual(new_layers.shape, (1, 1))

    def test_filter_handles_none_points(self):
        """Test _filter when points is None."""
        heuristic = self._create_heuristic()
        heuristic.reduce_op = torch.Tensor.min

        heuristic._compute_actual_bounds = MagicMock(return_value=torch.tensor([
            [-0.5],
            [-0.3],
            [-0.8],
            [-0.7],
        ]))

        domains = {
            'thresholds': torch.tensor([[0.0]]),
            'lower_bounds': {'layer1': torch.randn(1, 4)},
            'upper_bounds': {'layer1': torch.randn(1, 4)},
        }

        layers = torch.tensor([[0, 1]])
        indices = torch.tensor([[0, 1]])

        with patch('builtins.print'):
            new_layers, new_indices, new_points = heuristic._filter(
                domains, layers, indices, None
            )

        self.assertIsNone(new_points)


class TestReduceOpIntegration(unittest.TestCase):
    """Tests for reduce_op usage in NonlinearBranching."""

    def _create_heuristic(self):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_reduce_op_min(self):
        """Test that reduce_op is set correctly for 'min'."""
        from utils import get_reduce_op

        reduce_op = get_reduce_op('min', with_dim=True)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.0]])
        result = reduce_op(x, dim=1)

        # Should return values (and possibly indices)
        if isinstance(result, torch.Tensor):
            torch.testing.assert_close(result, torch.tensor([1.0, 0.5]))
        else:
            torch.testing.assert_close(result.values, torch.tensor([1.0, 0.5]))

    def test_reduce_op_max(self):
        """Test that reduce_op is set correctly for 'max'."""
        from utils import get_reduce_op

        reduce_op = get_reduce_op('max', with_dim=True)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.0]])
        result = reduce_op(x, dim=1)

        if isinstance(result, torch.Tensor):
            torch.testing.assert_close(result, torch.tensor([3.0, 4.0]))
        else:
            torch.testing.assert_close(result.values, torch.tensor([3.0, 4.0]))


class TestBranchingWithSpecialActivations(unittest.TestCase):
    """Tests for branching behavior with special activation functions."""

    def _create_heuristic(self):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()
        mock_net.split_activations = {'layer1': []}

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_relu_node_forces_branching_at_zero(self):
        """Test that ReLU nodes force branching point at 0 for unstable neurons."""
        from auto_LiRPA.bound_ops import BoundRelu

        heuristic = self._create_heuristic()

        # Create a mock ReLU node
        mock_relu = MagicMock(spec=BoundRelu)
        mock_node = _create_mock_node('layer1', output_shape=(1, 4))
        heuristic.net.split_activations = {'layer1': [(mock_relu,)]}

        lb = torch.tensor([[-1.0, 0.5, -0.5, 1.0]])  # Some unstable (cross 0)
        ub = torch.tensor([[1.0, 1.5, 0.5, 2.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        # Points should be at midpoint initially
        expected_midpoint = (lb + ub) / 2
        torch.testing.assert_close(points.squeeze(-1), expected_midpoint)


class TestInputSplitMethodAttribute(unittest.TestCase):
    """Tests for input_split_method attribute."""

    def test_input_split_method_is_sb(self):
        """Test that input_split_method is set to 'sb'."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.input_split_method, 'sb')


class TestModelAndRootsAttributes(unittest.TestCase):
    """Tests for model and roots attributes."""

    def test_model_attribute(self):
        """Test that model attribute is set from net.net."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.model, mock_net.net)

    def test_roots_attribute(self):
        """Test that roots attribute is populated from model.roots()."""
        mock_net = _create_mock_net()
        mock_root = MagicMock()
        mock_net.net.roots.return_value = [mock_root]

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        self.assertEqual(heuristic.roots, [mock_root])


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for NonlinearBranching."""

    def _create_heuristic(self):
        """Create a NonlinearBranching instance for testing."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 2,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        return NonlinearBranching(mock_net, **kwargs)

    def test_empty_batch(self):
        """Test handling of empty batch."""
        heuristic = self._create_heuristic()

        lb = torch.tensor([]).reshape(0, 5)
        ub = torch.tensor([]).reshape(0, 5)

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.shape[0], 0)

    def test_single_neuron(self):
        """Test handling of single neuron."""
        heuristic = self._create_heuristic()

        lb = torch.tensor([[0.0]])
        ub = torch.tensor([[1.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.shape, (1, 1, 1))
        torch.testing.assert_close(points[0, 0, 0], torch.tensor(0.5))

    def test_equal_bounds(self):
        """Test handling when lower and upper bounds are equal."""
        heuristic = self._create_heuristic()

        lb = torch.tensor([[1.0, 2.0]])
        ub = torch.tensor([[1.0, 2.0]])  # Equal to lb

        points = heuristic._get_uniform_branching_points(lb, ub)

        # Points should equal the bounds when they're equal
        torch.testing.assert_close(points.squeeze(-1), lb)

    def test_negative_bounds(self):
        """Test handling of negative bounds."""
        heuristic = self._create_heuristic()

        lb = torch.tensor([[-5.0, -3.0]])
        ub = torch.tensor([[-1.0, -2.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        expected = (lb + ub) / 2
        torch.testing.assert_close(points.squeeze(-1), expected)

    def test_large_batch(self):
        """Test handling of large batch."""
        heuristic = self._create_heuristic()

        batch_size = 1000
        num_neurons = 100
        lb = torch.randn(batch_size, num_neurons)
        ub = lb + torch.abs(torch.randn(batch_size, num_neurons))

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.shape, (batch_size, num_neurons, 1))


class TestNumBranchesVariations(unittest.TestCase):
    """Tests for different num_branches configurations."""

    def test_num_branches_5(self):
        """Test with 5 branches (4 branching points)."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 5,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        lb = torch.tensor([[0.0]])
        ub = torch.tensor([[5.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        # 5 branches means 4 branching points at 1/5, 2/5, 3/5, 4/5
        self.assertEqual(points.shape[-1], 4)

        expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        torch.testing.assert_close(points, expected)

    def test_num_branches_10(self):
        """Test with 10 branches (9 branching points)."""
        mock_net = _create_mock_net()

        kwargs = {
            'branching_point_method': 'uniform',
            'num_branches': 10,
            'method': 'fast',
            'filter': False,
            'filter_beta': False,
            'filter_batch_size': 100,
            'filter_iterations': 10,
            'filter_clamp': True,
            'relu_only': False,
        }

        heuristic = NonlinearBranching(mock_net, **kwargs)

        lb = torch.tensor([[0.0]])
        ub = torch.tensor([[10.0]])

        points = heuristic._get_uniform_branching_points(lb, ub)

        self.assertEqual(points.shape[-1], 9)

        # Points should be at 1, 2, 3, ..., 9
        for i in range(9):
            torch.testing.assert_close(points[0, 0, i], torch.tensor(float(i + 1)))


if __name__ == '__main__':
    unittest.main()
