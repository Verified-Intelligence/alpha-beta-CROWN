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
"""Unit tests for heuristics/nonlinear/utils.py module."""
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import torch.nn as nn
import numpy as np

sys.path.append('/storage/project/r-rbates8-0/rbates8/src/Verifier_Development/complete_verifier')


class TestPrecomputeABasic(unittest.TestCase):
    """Basic tests for precompute_A function."""

    def _create_mock_net(self, batch_size=2, device='cpu'):
        """Create a mock network for testing."""
        mock_net = MagicMock()
        mock_net.batch_size = batch_size
        mock_net.device = device
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']
        mock_net.split_activations = {}
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {}))
        return mock_net

    def test_empty_split_activations(self):
        """Test precompute_A with empty split_activations."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = self._create_mock_net()
        mock_net.split_activations = {}
        A = {}
        x = MagicMock()
        interm_bounds = {}

        # Should not raise any error
        precompute_A(mock_net, A, x, interm_bounds)

        # compute_bounds should not be called since need_A is empty
        mock_net.compute_bounds.assert_not_called()

    def test_node_already_in_A(self):
        """Test precompute_A when node is already in A dict."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = self._create_mock_net()

        # Create mock input node
        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True

        # Create mock node that references the input
        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.root_names = ['input']

        # A already contains the node
        A = {'layer1': torch.rand(2, 3, 4)}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # compute_bounds should not be called since node is already in A
        mock_net.compute_bounds.assert_not_called()

    def test_node_in_root_names_skipped(self):
        """Test that input nodes whose names are in root_names are not added to need_A.

        The precompute_A function iterates over split_activations and checks each
        node's inputs. If an input node's name is in net.root_names, that input
        is filtered out (not added to need_A). Since need_A ends up empty,
        compute_bounds is never called.
        """
        from heuristics.nonlinear.utils import precompute_A

        mock_net = self._create_mock_net()

        # Create mock input node whose name IS in root_names
        mock_input = MagicMock()
        mock_input.name = 'input'
        mock_input.perturbed = True

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.root_names = ['input']  # mock_input.name is in this list

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # compute_bounds should not be called since mock_input was filtered out
        # (its name 'input' is in root_names), leaving need_A empty
        mock_net.compute_bounds.assert_not_called()

    def test_unperturbed_node_skipped(self):
        """Test that input nodes with perturbed=False are not added to need_A.

        The precompute_A function iterates over split_activations and checks each
        node's inputs. If an input node has perturbed=False, that input is
        filtered out (not added to need_A). Since need_A ends up empty,
        compute_bounds is never called.
        """
        from heuristics.nonlinear.utils import precompute_A

        mock_net = self._create_mock_net()

        # Create mock input node with perturbed=False
        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = False  # This causes filtering

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.root_names = ['input']  # 'layer1' is NOT in root_names

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # compute_bounds should not be called since mock_input was filtered out
        # (perturbed=False), leaving need_A empty
        mock_net.compute_bounds.assert_not_called()


class TestPrecomputeAComputeBounds(unittest.TestCase):
    """Tests for precompute_A compute_bounds calls."""

    def _create_mock_net_with_node(self, batch_size=2, dim_output=10, device='cpu'):
        """Create a mock network with a node that needs A computation."""
        mock_net = MagicMock()
        mock_net.batch_size = batch_size
        mock_net.device = device
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        # Create mock input node
        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True
        mock_input.output_shape = [batch_size, dim_output]

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}

        # Set up compute_bounds to return updated A
        new_A = {'layer1': torch.rand(batch_size, dim_output, dim_output)}
        mock_net.compute_bounds = MagicMock(return_value=(None, None, new_A))

        return mock_net, mock_input

    def test_compute_bounds_called_for_missing_node(self):
        """Test that compute_bounds is called for nodes not in A."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        mock_net.compute_bounds.assert_called_once()

    def test_compute_bounds_uses_crown_method(self):
        """Test that compute_bounds uses CROWN method."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        self.assertEqual(call_kwargs['method'], 'CROWN')

    def test_compute_bounds_uses_return_A_true(self):
        """Test that compute_bounds uses return_A=True."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        self.assertTrue(call_kwargs['return_A'])

    def test_compute_bounds_uses_correct_final_node_name(self):
        """Test that compute_bounds uses correct final_node_name."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        self.assertEqual(call_kwargs['final_node_name'], 'layer1')

    def test_compute_bounds_uses_correct_needed_A_dict(self):
        """Test that compute_bounds uses correct needed_A_dict."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        expected_needed_A_dict = {'layer1': ['input']}
        self.assertEqual(call_kwargs['needed_A_dict'], expected_needed_A_dict)

    def test_A_updated_after_compute_bounds(self):
        """Test that A dict is updated after compute_bounds."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net, mock_input = self._create_mock_net_with_node()

        # compute_bounds returns new A entries
        new_A_entry = torch.rand(2, 10, 10)
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {'layer1': new_A_entry}))

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        self.assertIn('layer1', A)
        self.assertTrue(torch.equal(A['layer1'], new_A_entry))

    def test_identity_matrix_shape(self):
        """Test that identity matrix C has correct shape."""
        from heuristics.nonlinear.utils import precompute_A

        batch_size = 3
        dim_output = 5
        mock_net, mock_input = self._create_mock_net_with_node(
            batch_size=batch_size, dim_output=dim_output
        )

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # Check C shape in the call
        call_kwargs = mock_net.compute_bounds.call_args[1]
        C = call_kwargs['C']
        self.assertEqual(C.shape, (batch_size, dim_output, dim_output))

    def test_identity_matrix_values(self):
        """Test that C is an identity matrix."""
        from heuristics.nonlinear.utils import precompute_A

        batch_size = 2
        dim_output = 4
        mock_net, mock_input = self._create_mock_net_with_node(
            batch_size=batch_size, dim_output=dim_output
        )

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        C = call_kwargs['C']

        # Each batch should be identity matrix
        for b in range(batch_size):
            self.assertTrue(torch.allclose(C[b], torch.eye(dim_output)))


class TestPrecomputeAMultipleNodes(unittest.TestCase):
    """Tests for precompute_A with multiple nodes."""

    def test_multiple_split_activations(self):
        """Test precompute_A with multiple split_activations."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        # Create two different input nodes
        mock_input1 = MagicMock()
        mock_input1.name = 'layer1'
        mock_input1.perturbed = True
        mock_input1.output_shape = [2, 5]

        mock_input2 = MagicMock()
        mock_input2.name = 'layer2'
        mock_input2.perturbed = True
        mock_input2.output_shape = [2, 3]

        mock_node1 = MagicMock()
        mock_node1.inputs = [mock_input1]

        mock_node2 = MagicMock()
        mock_node2.inputs = [mock_input2]

        mock_net.split_activations = {
            'act1': [(mock_node1,)],
            'act2': [(mock_node2,)]
        }

        # Track compute_bounds calls
        call_count = [0]
        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            node_name = kwargs['final_node_name']
            return (None, None, {node_name: torch.rand(2, 5, 5)})

        mock_net.compute_bounds = MagicMock(side_effect=mock_compute_bounds)

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # Should be called for both nodes
        self.assertEqual(call_count[0], 2)

    def test_duplicate_nodes_processed_once(self):
        """Test that duplicate nodes are only processed once."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        # Create same input node referenced multiple times
        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True
        mock_input.output_shape = [2, 5]

        mock_node1 = MagicMock()
        mock_node1.inputs = [mock_input]

        mock_node2 = MagicMock()
        mock_node2.inputs = [mock_input]  # Same input

        mock_net.split_activations = {
            'act1': [(mock_node1,)],
            'act2': [(mock_node2,)]
        }

        mock_net.compute_bounds = MagicMock(return_value=(None, None, {'layer1': torch.rand(2, 5, 5)}))

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        # Should only be called once since it's the same node
        mock_net.compute_bounds.assert_called_once()


class TestPrecomputeAEdgeCases(unittest.TestCase):
    """Edge case tests for precompute_A."""

    def test_multidimensional_output_shape(self):
        """Test with multidimensional output shape (e.g., conv layers)."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        mock_input = MagicMock()
        mock_input.name = 'conv1'
        mock_input.perturbed = True
        mock_input.output_shape = [2, 3, 4, 4]  # batch, channels, height, width

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}

        dim_output = 3 * 4 * 4  # 48
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {'conv1': torch.rand(2, dim_output, 10)}))

        A = {}
        x = MagicMock()
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        C = call_kwargs['C']
        self.assertEqual(C.shape, (2, dim_output, dim_output))

    def test_interm_bounds_passed_to_compute_bounds(self):
        """Test that interm_bounds is passed to compute_bounds."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True
        mock_input.output_shape = [2, 5]

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {}))

        A = {}
        x = MagicMock()
        interm_bounds = {'layer0': (torch.zeros(2, 3), torch.ones(2, 3))}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        self.assertEqual(call_kwargs['interm_bounds'], interm_bounds)


class TestSetRootsBasic(unittest.TestCase):
    """Basic tests for set_roots function."""

    def _create_mock_bound_input(self):
        """Create a mock BoundInput node."""
        from auto_LiRPA.bound_ops import BoundInput
        return MagicMock(spec=BoundInput)

    def _create_mock_bound_constant(self):
        """Create a mock BoundConstant node."""
        from auto_LiRPA.bound_ops import BoundConstant
        mock_const = MagicMock(spec=BoundConstant)
        return mock_const

    def _create_mock_bound_params(self):
        """Create a mock BoundParams node."""
        from auto_LiRPA.bound_ops import BoundParams
        mock_params = MagicMock(spec=BoundParams)
        mock_params.param = nn.Parameter(torch.rand(3, 3))
        return mock_params

    def _create_mock_bound_buffers(self):
        """Create a mock BoundBuffers node."""
        from auto_LiRPA.bound_ops import BoundBuffers
        mock_buffers = MagicMock(spec=BoundBuffers)
        return mock_buffers

    def test_set_roots_with_single_bound_input(self):
        """Test set_roots with only BoundInput."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        self.assertTrue(torch.equal(bound_input.center, x))
        self.assertIs(bound_input.perturbation, x.ptb)
        self.assertIsNone(bound_input.aux)
        self.assertIsNone(bound_input.uA)

    def test_set_roots_sets_lA(self):
        """Test that set_roots sets lA correctly."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        expected_lA = A.sum(dim=2)
        self.assertTrue(torch.allclose(bound_input.lA, expected_lA))

    def test_set_roots_with_bound_constant(self):
        """Test set_roots with BoundInput and BoundConstant."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        bound_const = self._create_mock_bound_constant()
        roots = [bound_input, bound_const]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        self.assertIsNone(bound_const.perturbation)
        self.assertIsNone(bound_const.lA)
        self.assertIsNone(bound_const.uA)

    def test_set_roots_with_bound_params(self):
        """Test set_roots with BoundInput and BoundParams."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        bound_params = self._create_mock_bound_params()
        roots = [bound_input, bound_params]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        self.assertTrue(torch.equal(bound_params.center, bound_params.param))
        self.assertIsNone(bound_params.perturbation)
        self.assertIsNone(bound_params.lA)
        self.assertIsNone(bound_params.uA)

    def test_set_roots_with_bound_buffers(self):
        """Test set_roots with BoundInput and BoundBuffers."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        bound_buffers = self._create_mock_bound_buffers()
        roots = [bound_input, bound_buffers]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        self.assertIsNone(bound_buffers.perturbation)
        self.assertIsNone(bound_buffers.lA)
        self.assertIsNone(bound_buffers.uA)


class TestSetRootsAssertions(unittest.TestCase):
    """Tests for set_roots assertions."""

    def test_first_root_must_be_bound_input(self):
        """Test that first root must be BoundInput."""
        from heuristics.nonlinear.utils import set_roots
        from auto_LiRPA.bound_ops import BoundConstant

        mock_const = MagicMock(spec=BoundConstant)
        roots = [mock_const]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        with self.assertRaises(AssertionError):
            set_roots(roots, x, A)

    def test_subsequent_roots_must_be_valid_types(self):
        """Test that subsequent roots must be BoundConstant, BoundParams, or BoundBuffers."""
        from heuristics.nonlinear.utils import set_roots
        from auto_LiRPA.bound_ops import BoundInput

        mock_input = MagicMock(spec=BoundInput)

        # Create invalid second root (not BoundConstant, BoundParams, or BoundBuffers)
        invalid_root = MagicMock()
        roots = [mock_input, invalid_root]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        with self.assertRaises(AssertionError):
            set_roots(roots, x, A)

    def test_bound_params_param_must_be_nn_parameter(self):
        """Test that BoundParams.param must be nn.Parameter."""
        from heuristics.nonlinear.utils import set_roots
        from auto_LiRPA.bound_ops import BoundInput, BoundParams

        mock_input = MagicMock(spec=BoundInput)

        mock_params = MagicMock(spec=BoundParams)
        mock_params.param = torch.rand(3, 3)  # Regular tensor, not nn.Parameter

        roots = [mock_input, mock_params]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        with self.assertRaises(AssertionError):
            set_roots(roots, x, A)


class TestSetRootsMultipleRoots(unittest.TestCase):
    """Tests for set_roots with multiple root nodes."""

    def _create_mock_bound_input(self):
        """Create a mock BoundInput node."""
        from auto_LiRPA.bound_ops import BoundInput
        return MagicMock(spec=BoundInput)

    def _create_mock_bound_constant(self):
        """Create a mock BoundConstant node."""
        from auto_LiRPA.bound_ops import BoundConstant
        return MagicMock(spec=BoundConstant)

    def _create_mock_bound_params(self):
        """Create a mock BoundParams node."""
        from auto_LiRPA.bound_ops import BoundParams
        mock_params = MagicMock(spec=BoundParams)
        mock_params.param = nn.Parameter(torch.rand(3, 3))
        return mock_params

    def _create_mock_bound_buffers(self):
        """Create a mock BoundBuffers node."""
        from auto_LiRPA.bound_ops import BoundBuffers
        return MagicMock(spec=BoundBuffers)

    def test_mixed_root_types(self):
        """Test set_roots with mixed root types."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        bound_const = self._create_mock_bound_constant()
        bound_params = self._create_mock_bound_params()
        bound_buffers = self._create_mock_bound_buffers()

        roots = [bound_input, bound_const, bound_params, bound_buffers]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        # Should not raise
        set_roots(roots, x, A)

        # Verify all roots were processed
        self.assertTrue(torch.equal(bound_input.center, x))
        self.assertIsNone(bound_const.perturbation)
        self.assertTrue(torch.equal(bound_params.center, bound_params.param))
        self.assertIsNone(bound_buffers.perturbation)

    def test_multiple_bound_params(self):
        """Test set_roots with multiple BoundParams."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        bound_params1 = self._create_mock_bound_params()
        bound_params2 = self._create_mock_bound_params()

        roots = [bound_input, bound_params1, bound_params2]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)

        set_roots(roots, x, A)

        self.assertTrue(torch.equal(bound_params1.center, bound_params1.param))
        self.assertTrue(torch.equal(bound_params2.center, bound_params2.param))


class TestSetRootsLAComputation(unittest.TestCase):
    """Tests for lA computation in set_roots."""

    def _create_mock_bound_input(self):
        """Create a mock BoundInput node."""
        from auto_LiRPA.bound_ops import BoundInput
        return MagicMock(spec=BoundInput)

    def test_lA_sum_over_correct_dimension(self):
        """Test that lA is computed as sum over dim=2."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.rand(4, 2, 3)  # (dim_output, batch_size, dim_input)

        set_roots(roots, x, A)

        expected_lA = A.sum(dim=2)  # Sum over dim_input
        self.assertEqual(bound_input.lA.shape, (4, 2))  # (dim_output, batch_size)
        self.assertTrue(torch.allclose(bound_input.lA, expected_lA))

    def test_lA_with_different_shapes(self):
        """Test lA computation with different A shapes."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(3, 5)
        x.ptb = MagicMock()
        A = torch.rand(10, 3, 5)  # Different shape

        set_roots(roots, x, A)

        expected_lA = A.sum(dim=2)
        self.assertEqual(bound_input.lA.shape, (10, 3))
        self.assertTrue(torch.allclose(bound_input.lA, expected_lA))


class TestSetRootsEdgeCases(unittest.TestCase):
    """Edge case tests for set_roots."""

    def _create_mock_bound_input(self):
        """Create a mock BoundInput node."""
        from auto_LiRPA.bound_ops import BoundInput
        return MagicMock(spec=BoundInput)

    def test_single_element_A(self):
        """Test with single-element A tensor."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(1, 1)
        x.ptb = MagicMock()
        A = torch.rand(1, 1, 1)

        set_roots(roots, x, A)

        self.assertEqual(bound_input.lA.shape, (1, 1))

    def test_large_A_tensor(self):
        """Test with large A tensor."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(16, 100)
        x.ptb = MagicMock()
        A = torch.rand(50, 16, 100)

        set_roots(roots, x, A)

        self.assertEqual(bound_input.lA.shape, (50, 16))

    def test_zeros_A_tensor(self):
        """Test with all-zeros A tensor."""
        from heuristics.nonlinear.utils import set_roots

        bound_input = self._create_mock_bound_input()
        roots = [bound_input]

        x = torch.rand(2, 3)
        x.ptb = MagicMock()
        A = torch.zeros(4, 2, 3)

        set_roots(roots, x, A)

        self.assertTrue(torch.allclose(bound_input.lA, torch.zeros(4, 2)))


class TestPrecomputeAPrintMessage(unittest.TestCase):
    """Tests for precompute_A print message."""

    def test_prints_message_for_missing_A(self):
        """Test that a message is printed for missing A entries."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True
        mock_input.output_shape = [2, 5]

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {'layer1': torch.rand(2, 5, 5)}))

        A = {}
        x = MagicMock()
        interm_bounds = {}

        with patch('heuristics.nonlinear.utils.print') as mock_print:
            precompute_A(mock_net, A, x, interm_bounds)
            mock_print.assert_called()
            call_args = mock_print.call_args[0][0]
            self.assertIn('Missing A', call_args)
            self.assertIn('CROWN call', call_args)


class TestPrecomputeAXPassedCorrectly(unittest.TestCase):
    """Tests for x parameter handling in precompute_A."""

    def test_x_passed_as_tuple(self):
        """Test that x is passed as tuple to compute_bounds."""
        from heuristics.nonlinear.utils import precompute_A

        mock_net = MagicMock()
        mock_net.batch_size = 2
        mock_net.device = 'cpu'
        mock_net.root_names = ['input']
        mock_net.input_name = ['input']

        mock_input = MagicMock()
        mock_input.name = 'layer1'
        mock_input.perturbed = True
        mock_input.output_shape = [2, 5]

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        mock_net.split_activations = {'act': [(mock_node,)]}
        mock_net.compute_bounds = MagicMock(return_value=(None, None, {}))

        A = {}
        x = torch.rand(2, 3)
        interm_bounds = {}

        precompute_A(mock_net, A, x, interm_bounds)

        call_kwargs = mock_net.compute_bounds.call_args[1]
        self.assertEqual(call_kwargs['x'], (x,))


if __name__ == '__main__':
    unittest.main()
