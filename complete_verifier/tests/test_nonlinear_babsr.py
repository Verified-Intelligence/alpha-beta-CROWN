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
"""Unit tests for heuristics/nonlinear/babsr.py - BaBSRNonlinearBranching class."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import deque

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBaBSRNonlinearBranchingInit(unittest.TestCase):
    """Tests for BaBSRNonlinearBranching.__init__ method."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        return mock_net

    def test_init_stores_net(self):
        """Test that __init__ stores the net parameter."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        self.assertEqual(branching.net, mock_net)

    def test_init_stores_model(self):
        """Test that __init__ stores net.net as model."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        self.assertEqual(branching.model, mock_net.net)

    def test_init_stores_num_branches(self):
        """Test that __init__ stores the num_branches parameter."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=3)

        self.assertEqual(branching.num_branches, 3)


class ComputeHeuristicTestBase:
    """Base class with common setup for compute_heuristic tests."""

    @staticmethod
    def create_mock_net():
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_activation = MagicMock()
        mock_activation.name = 'act_node'
        mock_net.split_activations = {'node': [mock_activation]}
        mock_net.x = MagicMock()
        return mock_net

    @staticmethod
    def create_mock_node(name='node'):
        """Create a mock node."""
        mock_node = MagicMock()
        mock_node.name = name
        return mock_node

    @staticmethod
    def setup_branching_mocks(branching, batch_size, num_neurons, num_bounds):
        """Setup standard mocks for compute_heuristic tests.

        Key dimensions:
        - bounds_before: (batch_size, num_bounds)
        - bounds_after: (batch_size, num_neurons, num_bounds)
        - margin_before: (batch_size, 1, num_bounds) - note the unsqueeze(1)
        - margin_after: (batch_size, num_neurons) after amax(dim=-1)
        - scores: (batch_size, num_neurons)
        """
        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size, num_bounds)
        )
        branching.interm_bound_required = set()
        branching._get_bounds_after_branching = MagicMock(
            return_value=torch.rand(batch_size, num_neurons, num_bounds)
        )

    @staticmethod
    def create_margin_before(batch_size, num_bounds):
        """Create margin_before tensor with correct shape (batch, 1, bounds)."""
        return torch.rand(batch_size, 1, num_bounds)

    @staticmethod
    def create_test_domains(batch_size, num_neurons):
        """Create standard test domains."""
        return {
            'lAs': {'act_node': torch.rand(batch_size, 1, num_neurons)},
            'lower_bounds': {'node': -torch.ones(batch_size, num_neurons)},
            'upper_bounds': {'node': torch.ones(batch_size, num_neurons)},
        }


class TestBaBSRNonlinearBranchingComputeHeuristic(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for BaBSRNonlinearBranching.compute_heuristic method."""

    def test_compute_heuristic_returns_dict(self):
        """Test that compute_heuristic returns a dictionary."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertIsInstance(result, dict)

    def test_compute_heuristic_returns_scores_key(self):
        """Test that compute_heuristic returns dict with 'scores' key."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertIn('scores', result)

    def test_compute_heuristic_returns_points_key(self):
        """Test that compute_heuristic returns dict with 'points' key."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertIn('points', result)
        self.assertTrue(torch.equal(result['points'], points))

    def test_compute_heuristic_scores_shape(self):
        """Test that scores have correct shape."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))

    def test_compute_heuristic_calls_get_partial_bounds_batch(self):
        """Test that compute_heuristic calls get_partial_bounds_batch."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        branching.compute_heuristic(node, points, domains, margin_before)

        branching.get_partial_bounds_batch.assert_called()


class TestBaBSRNonlinearBranchingGetBoundsAfterBranching(unittest.TestCase):
    """Tests for BaBSRNonlinearBranching._get_bounds_after_branching method."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        return mock_net

    def test_returns_tensor(self):
        """Test that _get_bounds_after_branching returns a tensor."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size = 2
        num_neurons = 5
        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size * num_neurons, num_neurons)
        )

        branched_neurons = torch.arange(num_neurons).expand(batch_size, num_neurons)
        lAs = {'act': torch.rand(batch_size * num_neurons, 1, num_neurons)}
        lb = {'other': -torch.ones(batch_size * num_neurons, num_neurons)}
        ub = {'other': torch.ones(batch_size * num_neurons, num_neurons)}
        lb_ori = -torch.ones(batch_size, num_neurons)
        ub_ori = torch.ones(batch_size, num_neurons)
        lb_branched = -0.5 * torch.ones(batch_size, num_neurons)
        ub_branched = 0.5 * torch.ones(batch_size, num_neurons)
        node_name = 'node'
        start_nodes = [MagicMock()]

        result = branching._get_bounds_after_branching(
            branched_neurons, lAs, lb, ub,
            lb_ori, ub_ori, lb_branched, ub_branched,
            node_name, start_nodes
        )

        self.assertIsInstance(result, torch.Tensor)

    def test_output_shape(self):
        """Test that output has correct shape [batch_size, num_neurons, -1]."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size = 2
        num_neurons = 5
        partial_bounds_size = 3
        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size * num_neurons, partial_bounds_size)
        )

        branched_neurons = torch.arange(num_neurons).expand(batch_size, num_neurons)
        lAs = {'act': torch.rand(batch_size * num_neurons, 1, num_neurons)}
        lb = {'other': -torch.ones(batch_size * num_neurons, num_neurons)}
        ub = {'other': torch.ones(batch_size * num_neurons, num_neurons)}
        lb_ori = -torch.ones(batch_size, num_neurons)
        ub_ori = torch.ones(batch_size, num_neurons)
        lb_branched = -0.5 * torch.ones(batch_size, num_neurons)
        ub_branched = 0.5 * torch.ones(batch_size, num_neurons)
        node_name = 'node'
        start_nodes = [MagicMock()]

        result = branching._get_bounds_after_branching(
            branched_neurons, lAs, lb, ub,
            lb_ori, ub_ori, lb_branched, ub_branched,
            node_name, start_nodes
        )

        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], num_neurons)


class TestBaBSRNonlinearBranchingGetPartialBoundsBatch(unittest.TestCase):
    """Tests for BaBSRNonlinearBranching.get_partial_bounds_batch method."""

    def _create_mock_net(self):
        """Create a mock network object with required attributes."""
        mock_net = MagicMock()
        mock_model = MagicMock()
        mock_net.net = mock_model
        mock_net.x = MagicMock()

        # Setup model attributes
        mock_model.root_names = ['input']
        mock_model.final_name = 'output'

        return mock_net, mock_model

    def test_output_does_not_track_gradients(self):
        """Partial heuristic bounds should not build an autograd graph."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()
        source = torch.ones(2, 3, requires_grad=True)
        start_node = MagicMock()
        start_node.name = 'start'
        start_node.inputs = []
        start_node.perturbed = True
        start_node.zero_uA_mtx = False
        start_node.zero_lA_mtx = False
        mock_model.nodes.return_value = [start_node]
        mock_model.__getitem__ = MagicMock(return_value=MagicMock())

        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)
        with patch(
            'heuristics.nonlinear.babsr.expand_batch',
            return_value=MagicMock(device='cpu'),
        ), patch(
            'heuristics.nonlinear.babsr.get_degrees', return_value={},
        ), patch.object(
            branching, '_backward_propagate', return_value=source * 2,
        ):
            result = branching.get_partial_bounds_batch(
                lb={'input': source},
                ub={'input': source + 1},
                lAs={'start': torch.ones(2, 1)},
                start_nodes=[start_node],
            )

        self.assertFalse(result.requires_grad)

    def test_sets_interm_bound_required(self):
        """Test that method initializes interm_bound_required as empty set."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()

        mock_node = MagicMock()
        mock_node.perturbed = False
        mock_node.name = 'test_node'
        mock_node.lA = None
        mock_node.uA = None
        mock_model.nodes.return_value = [mock_node]
        mock_model.get_forward_value.return_value = torch.tensor(0.0)
        mock_model.__getitem__ = MagicMock(return_value=mock_node)

        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        # Set a non-empty value to verify it gets reset
        branching.interm_bound_required = {'old_value'}

        batch_size = 2
        num_neurons = 5
        lb = {'test': -torch.ones(batch_size, num_neurons)}
        ub = {'test': torch.ones(batch_size, num_neurons)}
        lAs = {}
        start_nodes = []

        with patch('heuristics.nonlinear.babsr.expand_batch', return_value=MagicMock(device='cpu')):
            with patch('heuristics.nonlinear.babsr.get_degrees', return_value={}):
                try:
                    branching.get_partial_bounds_batch(lb, ub, lAs, start_nodes)
                except Exception:
                    pass

        # Verify it's a set
        self.assertIsInstance(branching.interm_bound_required, set)
        # Verify old value was cleared (set is re-initialized, not appended to)
        self.assertNotIn('old_value', branching.interm_bound_required)


class TestBaBSRNonlinearBranchingBackwardPropagate(unittest.TestCase):
    """Tests for BaBSRNonlinearBranching._backward_propagate method."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_model = MagicMock()
        mock_net.net = mock_model
        mock_model.final_name = 'output'
        return mock_net, mock_model

    def test_adds_to_interm_bound_required(self):
        """Test that method adds input names to interm_bound_required."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)
        branching.interm_bound_required = set()

        mock_node = MagicMock()
        mock_node.requires_input_bounds = [0]
        mock_input = MagicMock()
        mock_input.name = 'input_node'
        mock_node.inputs = [mock_input]
        mock_node.lA = torch.rand(2, 5)
        mock_node.bound_backward = MagicMock(return_value=(
            [(torch.rand(2, 5), None)],
            torch.tensor(0.0),
            None
        ))

        start_nodes = []

        branching._backward_propagate(mock_node, start_nodes)

        self.assertIn('input_node', branching.interm_bound_required)

    def test_returns_lb(self):
        """Test that method returns lower bound."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)
        branching.interm_bound_required = set()

        mock_node = MagicMock()
        mock_node.requires_input_bounds = []
        mock_node.inputs = []
        mock_node.lA = torch.rand(2, 5)
        expected_lb = torch.tensor(1.5)
        mock_node.bound_backward = MagicMock(return_value=(
            [],
            expected_lb,
            None
        ))

        start_nodes = []

        result = branching._backward_propagate(mock_node, start_nodes)

        self.assertTrue(torch.equal(result, expected_lb))

    def test_stops_propagation_when_lower_b_is_tensor_not_in_start_nodes(self):
        """Test that propagation stops when lower_b is tensor and node not in start_nodes."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)
        branching.interm_bound_required = set()

        mock_node = MagicMock()
        mock_node.requires_input_bounds = []
        mock_input = MagicMock()
        mock_node.inputs = [mock_input]
        mock_node.lA = torch.rand(2, 5)
        lower_b = torch.tensor([1.0, 2.0])
        mock_node.bound_backward = MagicMock(return_value=(
            [(torch.rand(2, 5), None)],
            lower_b,
            None
        ))

        start_nodes = []

        result = branching._backward_propagate(mock_node, start_nodes)

        self.assertTrue(torch.equal(result, lower_b))

    def test_continues_propagation_when_node_in_start_nodes(self):
        """Test that propagation continues when node is in start_nodes."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net, mock_model = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)
        branching.interm_bound_required = set()

        mock_node = MagicMock()
        mock_node.requires_input_bounds = []
        mock_input = MagicMock()
        mock_node.inputs = [mock_input]
        mock_node.lA = torch.rand(2, 5)
        lower_b = torch.tensor([1.0, 2.0])
        mock_node.bound_backward = MagicMock(return_value=(
            [(torch.rand(2, 5), None)],
            lower_b,
            None
        ))

        start_nodes = [mock_node]

        with patch('heuristics.nonlinear.babsr.add_bound') as mock_add_bound:
            branching._backward_propagate(mock_node, start_nodes)
            mock_add_bound.assert_called()


class TestBaBSRNonlinearBranchingNumBranches(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for num_branches parameter behavior."""

    def test_num_branches_affects_iterations(self):
        """Test that num_branches controls the number of branch iterations."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        batch_size, num_neurons, num_bounds = 2, 5, 3

        for num_branches in [2, 3, 4]:
            mock_net = self.create_mock_net()
            branching = BaBSRNonlinearBranching(mock_net, num_branches=num_branches)

            call_count = [0]

            def mock_get_bounds(*args, **kwargs):
                call_count[0] += 1
                return torch.rand(batch_size, num_neurons, num_bounds)

            branching.get_partial_bounds_batch = MagicMock(
                return_value=torch.rand(batch_size, num_bounds)
            )
            branching.interm_bound_required = set()
            branching._get_bounds_after_branching = MagicMock(side_effect=mock_get_bounds)

            node = self.create_mock_node()
            points = torch.rand(batch_size, num_neurons, num_branches - 1)
            domains = self.create_test_domains(batch_size, num_neurons)
            margin_before = self.create_margin_before(batch_size, num_bounds)

            branching.compute_heuristic(node, points, domains, margin_before)

            self.assertEqual(call_count[0], num_branches)


class TestBaBSRNonlinearBranchingMarginComputation(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for margin computation in compute_heuristic."""

    def test_margin_after_is_min_across_branches(self):
        """Test that margin_after uses torch.min across branches.

        With num_branches=3, _get_bounds_after_branching is called 3 times.
        We return decreasing values: 9.0, 8.0, 7.0.
        The minimum (7.0) should be used for margin_after.

        The score formula is: margin_after - margin_before.amax(dim=-1)
        With bounds_before=0 and margin_before=0, the margin_after_ for each branch is:
            margin_after_ = (margin_before + (bounds_after - bounds_before)).amax(dim=-1)
                          = (0 + bounds_after).amax(dim=-1)
                          = bounds_after.amax(dim=-1)
        So margin_after_ values are 9.0, 8.0, 7.0 (after amax over num_bounds).
        After torch.min: margin_after = 7.0
        Score = margin_after - margin_before.amax(dim=-1) = 7.0 - 0 = 7.0
        """
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=3)

        batch_size, num_neurons, num_bounds = 2, 5, 3

        # bounds_before = 0, so (bounds_after - bounds_before) = bounds_after
        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.zeros(batch_size, num_bounds)
        )
        branching.interm_bound_required = set()

        call_count = [0]

        def mock_get_bounds(*args, **kwargs):
            call_count[0] += 1
            # Returns 9.0, 8.0, 7.0 for successive calls
            return torch.full((batch_size, num_neurons, num_bounds), 10.0 - call_count[0])

        branching._get_bounds_after_branching = MagicMock(side_effect=mock_get_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 2)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = torch.zeros(batch_size, 1, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        # The minimum across branches (7.0) should be used
        # Score = margin_after - margin_before.amax() = 7.0 - 0 = 7.0
        expected_score = 7.0
        self.assertTrue(torch.allclose(result['scores'],
                                       torch.full((batch_size, num_neurons), expected_score)))


class TestBaBSRNonlinearBranchingRepeatHelper(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for the _repeat helper function in compute_heuristic."""

    def _setup_net_with_getitem(self):
        """Create mock network with __getitem__ support.

        The filtering logic in compute_heuristic line 50-51 is:
            lAs = {k: _repeat(v) for k, v in lAs.items()
                    if self.net.net[k] in start_nodes}

        And start_nodes is computed at line 38:
            start_nodes = [act[0] for act in self.net.split_activations[name]]

        So we need:
        - mock_activation[0] to return a specific object (the "start node")
        - mock_net.net['act_node'] to return that same object
        """
        mock_net = MagicMock()
        mock_net.net = MagicMock()

        # Create the start_node object that will be checked in the filter
        start_node = MagicMock(name='start_node_object')

        # Create mock activation where act[0] returns start_node
        mock_activation = MagicMock()
        mock_activation.name = 'act_node'
        mock_activation.__getitem__ = MagicMock(return_value=start_node)

        mock_net.split_activations = {'node': [mock_activation]}
        mock_net.x = MagicMock()

        # Make net.net['act_node'] return the same start_node
        mock_net.net.__getitem__ = MagicMock(return_value=start_node)

        return mock_net

    def test_lAs_are_filtered_by_start_nodes(self):
        """Test that lAs are filtered to only include start_nodes."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._setup_net_with_getitem()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3

        captured_lAs = []

        def capture_get_bounds(branched_neurons, lAs, *args, **kwargs):
            captured_lAs.append(dict(lAs))
            return torch.rand(batch_size, num_neurons, num_bounds)

        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size, num_bounds)
        )
        branching.interm_bound_required = set()
        branching._get_bounds_after_branching = MagicMock(side_effect=capture_get_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = {
            'lAs': {
                'act_node': torch.rand(batch_size, 1, num_neurons),
                'other_node': torch.rand(batch_size, 1, num_neurons),
            },
            'lower_bounds': {'node': -torch.ones(batch_size, num_neurons)},
            'upper_bounds': {'node': torch.ones(batch_size, num_neurons)},
        }
        margin_before = self.create_margin_before(batch_size, num_bounds)

        branching.compute_heuristic(node, points, domains, margin_before)

        self.assertTrue(len(captured_lAs) > 0)
        self.assertIn('act_node', captured_lAs[0])


class TestBaBSRNonlinearBranchingBoundsFiltering(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for bounds filtering in compute_heuristic."""

    def _setup_net_with_getitem(self):
        """Create mock network with __getitem__ support."""
        mock_net = self.create_mock_net()
        mock_activation = mock_net.split_activations['node'][0]
        mock_net.net.__getitem__ = MagicMock(return_value=mock_activation)
        return mock_net

    def test_node_name_excluded_from_lb_ub(self):
        """Test that node.name is excluded from lb and ub dictionaries.

        The compute_heuristic method filters lb/ub to exclude the key matching
        node.name (stored in local variable 'name'). This test verifies that
        'node' (the mock node's name) is not present in the filtered lb dict.
        """
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._setup_net_with_getitem()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3

        captured_lb = []

        def capture_get_bounds(branched_neurons, lAs, lb, ub, *args, **kwargs):
            captured_lb.append(set(lb.keys()))
            return torch.rand(batch_size, num_neurons, num_bounds)

        branching.interm_bound_required = {'other_bound'}
        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size, num_bounds)
        )
        branching._get_bounds_after_branching = MagicMock(side_effect=capture_get_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = {
            'lAs': {'act_node': torch.rand(batch_size, 1, num_neurons)},
            'lower_bounds': {
                'node': -torch.ones(batch_size, num_neurons),
                'other_bound': -0.5 * torch.ones(batch_size, num_neurons),
            },
            'upper_bounds': {
                'node': torch.ones(batch_size, num_neurons),
                'other_bound': 0.5 * torch.ones(batch_size, num_neurons),
            },
        }
        margin_before = self.create_margin_before(batch_size, num_bounds)

        branching.compute_heuristic(node, points, domains, margin_before)

        self.assertTrue(len(captured_lb) > 0)
        for lb_keys in captured_lb:
            self.assertNotIn('node', lb_keys)


class TestBaBSRNonlinearBranchingEdgeCases(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for edge cases in BaBSRNonlinearBranching."""

    def test_single_batch_size(self):
        """Test with batch_size=1."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 1, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))

    def test_single_neuron(self):
        """Test with num_neurons=1."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 1, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))

    def test_large_num_branches(self):
        """Test with large num_branches value."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        num_branches = 10
        branching = BaBSRNonlinearBranching(mock_net, num_branches=num_branches)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, num_branches - 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))


class TestBaBSRNonlinearBranchingIntegration(unittest.TestCase, ComputeHeuristicTestBase):
    """Integration tests for BaBSRNonlinearBranching."""

    def test_full_flow_with_mocked_dependencies(self):
        """Test the full compute_heuristic flow with all dependencies mocked."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 3, 4, 2
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertIn('scores', result)
        self.assertIn('points', result)
        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))
        self.assertTrue(torch.equal(result['points'], points))

    def test_scores_dtype_is_float(self):
        """Test that scores are float tensors."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3
        self.setup_branching_mocks(branching, batch_size, num_neurons, num_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertTrue(result['scores'].dtype in [torch.float32, torch.float64])


class TestBaBSRNonlinearBranchingBranchHelper(unittest.TestCase):
    """Tests for the _branch helper function in _get_bounds_after_branching."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        return mock_net

    def test_branch_helper_uses_scatter(self):
        """Test that _branch uses scatter to update specific neurons."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size = 2
        num_neurons = 5

        captured_lb = []

        def mock_get_partial(lb, ub, lAs, start_nodes):
            captured_lb.append({k: v.clone() for k, v in lb.items()})
            return torch.rand(lb[next(iter(lb.keys()))].shape[0], num_neurons)

        branching.get_partial_bounds_batch = MagicMock(side_effect=mock_get_partial)

        branched_neurons = torch.arange(num_neurons).expand(batch_size, num_neurons)
        lAs = {'act': torch.rand(batch_size * num_neurons, 1, num_neurons)}
        lb = {}
        ub = {}
        lb_ori = -torch.ones(batch_size, num_neurons)
        ub_ori = torch.ones(batch_size, num_neurons)
        lb_branched = torch.zeros(batch_size, num_neurons)
        ub_branched = 0.5 * torch.ones(batch_size, num_neurons)
        node_name = 'node'
        start_nodes = [MagicMock()]

        branching._get_bounds_after_branching(
            branched_neurons, lAs, lb, ub,
            lb_ori, ub_ori, lb_branched, ub_branched,
            node_name, start_nodes
        )

        self.assertTrue(len(captured_lb) > 0)
        self.assertIn(node_name, captured_lb[0])


class TestBaBSRNonlinearBranchingDeviceHandling(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for device handling in BaBSRNonlinearBranching."""

    def test_branched_neurons_device_matches_bounds(self):
        """Test that branched_neurons tensor is on the same device as bounds."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3

        bounds_before = torch.rand(batch_size, num_bounds)

        captured_branched_neurons = []

        def capture_get_bounds_after(branched_neurons, *args, **kwargs):
            captured_branched_neurons.append(branched_neurons.clone())
            return torch.rand(batch_size, num_neurons, num_bounds)

        branching.get_partial_bounds_batch = MagicMock(return_value=bounds_before)
        branching.interm_bound_required = set()
        branching._get_bounds_after_branching = MagicMock(side_effect=capture_get_bounds_after)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = self.create_test_domains(batch_size, num_neurons)
        margin_before = self.create_margin_before(batch_size, num_bounds)

        branching.compute_heuristic(node, points, domains, margin_before)

        self.assertTrue(len(captured_branched_neurons) > 0)
        self.assertEqual(captured_branched_neurons[0].device, bounds_before.device)


class TestBaBSRNonlinearBranchingMultidimensionalPoints(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for handling multi-dimensional points in compute_heuristic."""

    def test_points_shape_with_2d_neurons(self):
        """Test with 2D neuron layout (e.g., convolutional layer)."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self.create_mock_net()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size = 2
        height, width = 3, 4
        num_neurons = height * width
        num_bounds = 3

        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size, num_bounds)
        )
        branching.interm_bound_required = set()
        branching._get_bounds_after_branching = MagicMock(
            return_value=torch.rand(batch_size, num_neurons, num_bounds)
        )

        node = self.create_mock_node()
        points = torch.rand(batch_size, height, width, 1)
        domains = {
            'lAs': {'act_node': torch.rand(batch_size, 1, height, width)},
            'lower_bounds': {'node': -torch.ones(batch_size, height, width)},
            'upper_bounds': {'node': torch.ones(batch_size, height, width)},
        }
        margin_before = self.create_margin_before(batch_size, num_bounds)

        result = branching.compute_heuristic(node, points, domains, margin_before)

        self.assertEqual(result['scores'].shape, (batch_size, num_neurons))


class TestBaBSRNonlinearBranchingStartNodesHandling(unittest.TestCase, ComputeHeuristicTestBase):
    """Tests for start_nodes handling in compute_heuristic."""

    def _create_mock_net_multiple_activations(self):
        """Create mock network with multiple activations."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()

        mock_act1 = MagicMock()
        mock_act1.name = 'act_node1'
        mock_act2 = MagicMock()
        mock_act2.name = 'act_node2'

        mock_net.split_activations = {'node': [mock_act1, mock_act2]}
        mock_net.x = MagicMock()
        return mock_net

    def test_multiple_start_nodes(self):
        """Test handling of multiple start nodes from split_activations."""
        from heuristics.nonlinear.babsr import BaBSRNonlinearBranching

        mock_net = self._create_mock_net_multiple_activations()
        branching = BaBSRNonlinearBranching(mock_net, num_branches=2)

        batch_size, num_neurons, num_bounds = 2, 5, 3

        captured_start_nodes = []

        def capture_get_bounds(branched_neurons, lAs, lb, ub, lb_ori, ub_ori,
                               lb_branched, ub_branched, node_name, start_nodes):
            captured_start_nodes.append(list(start_nodes))
            return torch.rand(batch_size, num_neurons, num_bounds)

        branching.get_partial_bounds_batch = MagicMock(
            return_value=torch.rand(batch_size, num_bounds)
        )
        branching.interm_bound_required = set()
        branching._get_bounds_after_branching = MagicMock(side_effect=capture_get_bounds)

        node = self.create_mock_node()
        points = torch.rand(batch_size, num_neurons, 1)
        domains = {
            'lAs': {
                'act_node1': torch.rand(batch_size, 1, num_neurons),
                'act_node2': torch.rand(batch_size, 1, num_neurons),
            },
            'lower_bounds': {'node': -torch.ones(batch_size, num_neurons)},
            'upper_bounds': {'node': torch.ones(batch_size, num_neurons)},
        }
        margin_before = self.create_margin_before(batch_size, num_bounds)

        branching.compute_heuristic(node, points, domains, margin_before)

        self.assertTrue(len(captured_start_nodes) > 0)
        self.assertEqual(len(captured_start_nodes[0]), 2)


if __name__ == '__main__':
    unittest.main()
