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
"""Unit tests for input_split/input_split_on_relu_domains.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict

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
    # Set interm_transfer to True for InputReluSplitter
    new_config['bab']['interm_transfer'] = True
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestInputReluSplitterInit(unittest.TestCase):
    """Tests for InputReluSplitter initialization."""

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_without_interm_transfer_raises_error(self, mock_config):
        """Test that interm_transfer=False raises ValueError."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': False,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 5,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        with self.assertRaises(ValueError) as context:
            InputReluSplitter()

        self.assertIn('interm_transfer', str(context.exception))

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_sets_relu_split_iterations(self, mock_config):
        """Test that init sets relu_split_iterations correctly."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 10,
                'branching_input_iterations': 5,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.relu_split_iterations, 10)

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_sets_input_split_iterations(self, mock_config):
        """Test that init sets input_split_iterations correctly."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 10,
                'branching_input_iterations': 7,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.input_split_iterations, 7)

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_sets_split_order(self, mock_config):
        """Test that init sets split_order correctly."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 5,
                'branching_input_and_activation_order': ['activation', 'input']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.split_order, ['activation', 'input'])

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_computes_reseting_round(self, mock_config):
        """Test that init computes reseting_round correctly."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 8,
                'branching_input_iterations': 4,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.reseting_round, 12)  # 8 + 4


class TestInputReluSplitterSplitConditionInputFirst(unittest.TestCase):
    """Tests for split_condition when input split comes first."""

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_input_first_round_0(self, mock_print, mock_config):
        """Test split_condition returns input split at round 0 when input first."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        # Round 0: r % 8 = 0 < 3 (input_split_iterations), so input split
        result = splitter.split_condition(0)
        # The method currently returns None (no return statement)
        self.assertIsNone(result)
        # Verify the correct print output for input split
        mock_print.assert_called_with('Round 0, using input split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_input_first_round_at_boundary(self, mock_print, mock_config):
        """Test split_condition at boundary between input and relu."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        # Round 3: r % 8 = 3, which is NOT < 3, so activation split
        result = splitter.split_condition(3)
        self.assertIsNone(result)
        # Verify the correct print output for activation split
        mock_print.assert_called_with('Round 3, using activation split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_input_first_round_wraps(self, mock_print, mock_config):
        """Test split_condition wraps around correctly."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        # Round 8: r % 8 = 0 < 3, so input split (wrapped)
        result = splitter.split_condition(8)
        self.assertIsNone(result)
        # Verify the correct print output for input split (wrapped around)
        mock_print.assert_called_with('Round 8, using input split.')


class TestInputReluSplitterSplitConditionActivationFirst(unittest.TestCase):
    """Tests for split_condition when activation split comes first."""

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_activation_first_round_0(self, mock_print, mock_config):
        """Test split_condition returns activation split at round 0 when activation first."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['activation', 'input']
            }
        }

        splitter = InputReluSplitter()

        # Round 0: r % 8 = 0, NOT >= 5 (relu_split_iterations), so activation split
        result = splitter.split_condition(0)
        self.assertIsNone(result)
        # Verify the correct print output for activation split
        mock_print.assert_called_with('Round 0, using activation split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_activation_first_round_at_boundary(self, mock_print, mock_config):
        """Test split_condition at boundary when activation comes first."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['activation', 'input']
            }
        }

        splitter = InputReluSplitter()

        # Round 5: r % 8 = 5 >= 5, so input split
        result = splitter.split_condition(5)
        self.assertIsNone(result)
        # Verify the correct print output for input split
        mock_print.assert_called_with('Round 5, using input split.')


class TestInputBranchingDecisions(unittest.TestCase):
    """Tests for input_branching_decisions function."""

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_calls_input_split_branching(self, mock_branching):
        """Test that input_branching_decisions calls input_split_branching."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0, 1, 2], [1, 2, 0]])

        batch_size = 2
        input_dim = 4
        num_specs = 2

        wrapped_net = MagicMock()
        global_lbs = torch.randn(batch_size, num_specs)
        lAs = torch.randn(batch_size, num_specs, input_dim)
        x_Ls = torch.zeros(batch_size, input_dim)
        x_Us = torch.ones(batch_size, input_dim)
        rhs = torch.zeros(batch_size, num_specs)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        mock_branching.assert_called_once()
        call_kwargs = mock_branching.call_args[1]
        self.assertEqual(call_kwargs['branching_method'], 'sb')
        self.assertEqual(call_kwargs['split_depth'], 3)

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_returns_split_indices(self, mock_branching):
        """Test that input_branching_decisions returns split indices."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        expected_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        mock_branching.return_value = expected_indices

        batch_size = 2
        input_dim = 4
        num_specs = 2

        wrapped_net = MagicMock()
        global_lbs = torch.randn(batch_size, num_specs)
        lAs = torch.randn(batch_size, num_specs, input_dim)
        x_Ls = torch.zeros(batch_size, input_dim)
        x_Us = torch.ones(batch_size, input_dim)
        rhs = torch.zeros(batch_size, num_specs)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        self.assertTrue(torch.equal(result, expected_indices))

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_passes_correct_parameters(self, mock_branching):
        """Test that correct parameters are passed to input_split_branching."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0]])

        batch_size = 1
        input_dim = 3
        num_specs = 2

        wrapped_net = MagicMock()
        global_lbs = torch.randn(batch_size, num_specs)
        lAs = torch.randn(batch_size, num_specs, input_dim)
        x_Ls = torch.zeros(batch_size, input_dim)
        x_Us = torch.ones(batch_size, input_dim)
        rhs = torch.zeros(batch_size, num_specs)

        input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        call_kwargs = mock_branching.call_args[1]
        self.assertEqual(call_kwargs['net'], wrapped_net)
        self.assertTrue(torch.equal(call_kwargs['dom_lb'], global_lbs))
        self.assertTrue(torch.equal(call_kwargs['x_L'], x_Ls))
        self.assertTrue(torch.equal(call_kwargs['x_U'], x_Us))
        self.assertTrue(torch.equal(call_kwargs['lA'], lAs))
        self.assertTrue(torch.equal(call_kwargs['thresholds'], rhs))


class TestInputBranchingDecisionsNoGrad(unittest.TestCase):
    """Tests for torch.no_grad decorator on input_branching_decisions."""

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_no_grad_context(self, mock_branching):
        """Test that function runs without gradient computation."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0, 1, 2]])

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 2, requires_grad=True)
        lAs = torch.randn(1, 2, 3, requires_grad=True)
        x_Ls = torch.zeros(1, 3, requires_grad=True)
        x_Us = torch.ones(1, 3, requires_grad=True)
        rhs = torch.zeros(1, 2, requires_grad=True)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        # Result should not require gradients
        self.assertFalse(result.requires_grad)


class TestInputReluSplitterSplitConditionPrinting(unittest.TestCase):
    """Tests for split_condition output messages."""

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_prints_input_split_message(self, mock_print, mock_config):
        """Test that split_condition prints correct message for input split."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()
        splitter.split_condition(0)

        mock_print.assert_called_with('Round 0, using input split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_prints_activation_split_message(self, mock_print, mock_config):
        """Test that split_condition prints correct message for activation split."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()
        # Round 3 should trigger activation split (r % 8 = 3, not < 3)
        splitter.split_condition(3)

        mock_print.assert_called_with('Round 3, using activation split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_prints_activation_first_input_message(self, mock_print, mock_config):
        """Test that split_condition prints correct message when activation comes first."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['activation', 'input']
            }
        }

        splitter = InputReluSplitter()
        # Round 5 should trigger input split (r % 8 = 5 >= 5)
        splitter.split_condition(5)

        mock_print.assert_called_with('Round 5, using input split.')


class TestInputReluSplitterEdgeCases(unittest.TestCase):
    """Edge case tests for InputReluSplitter."""

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_with_zero_relu_iterations(self, mock_config):
        """Test initialization with zero relu iterations."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 0,
                'branching_input_iterations': 5,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.relu_split_iterations, 0)
        self.assertEqual(splitter.reseting_round, 5)

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    def test_init_with_zero_input_iterations(self, mock_config):
        """Test initialization with zero input iterations."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 0,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        self.assertEqual(splitter.input_split_iterations, 0)
        self.assertEqual(splitter.reseting_round, 5)

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_large_round_number(self, mock_print, mock_config):
        """Test split_condition with a large round number."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 5,
                'branching_input_iterations': 3,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()
        # Large round 1000 % 8 = 0, should be input split
        result = splitter.split_condition(1000)
        self.assertIsNone(result)
        # Verify the correct print output for input split (1000 % 8 = 0 < 3)
        mock_print.assert_called_with('Round 1000, using input split.')

    @patch('input_split.input_split_on_relu_domains.arguments.Config', new_callable=dict)
    @patch('builtins.print')
    def test_split_condition_at_each_round_in_cycle(self, mock_print, mock_config):
        """Test split_condition for each round in a complete cycle."""
        from input_split.input_split_on_relu_domains import InputReluSplitter

        mock_config['bab'] = {
            'interm_transfer': True,
            'branching': {
                'branching_relu_iterations': 4,
                'branching_input_iterations': 2,
                'branching_input_and_activation_order': ['input', 'activation']
            }
        }

        splitter = InputReluSplitter()

        # Complete cycle is 6 rounds (4 + 2)
        # Input first: rounds 0, 1 should be input; rounds 2, 3, 4, 5 should be activation
        expected_calls = [
            'Round 0, using input split.',
            'Round 1, using input split.',
            'Round 2, using activation split.',
            'Round 3, using activation split.',
            'Round 4, using activation split.',
            'Round 5, using activation split.',
        ]
        for r in range(6):
            result = splitter.split_condition(r)
            self.assertIsNone(result)
            mock_print.assert_called_with(expected_calls[r])


class TestInputBranchingDecisionsEdgeCases(unittest.TestCase):
    """Edge case tests for input_branching_decisions."""

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_single_batch_single_input(self, mock_branching):
        """Test with single batch and single input dimension."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        expected_indices = torch.tensor([[0, 0, 0]])
        mock_branching.return_value = expected_indices

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 1)
        lAs = torch.randn(1, 1, 1)
        x_Ls = torch.zeros(1, 1)
        x_Us = torch.ones(1, 1)
        rhs = torch.zeros(1, 1)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        self.assertTrue(torch.equal(result, expected_indices))

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_large_batch_size(self, mock_branching):
        """Test with large batch size."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        batch_size = 100
        input_dim = 10
        num_specs = 5

        expected_indices = torch.randint(0, input_dim, (batch_size, 3))
        mock_branching.return_value = expected_indices

        wrapped_net = MagicMock()
        global_lbs = torch.randn(batch_size, num_specs)
        lAs = torch.randn(batch_size, num_specs, input_dim)
        x_Ls = torch.zeros(batch_size, input_dim)
        x_Us = torch.ones(batch_size, input_dim)
        rhs = torch.zeros(batch_size, num_specs)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        self.assertEqual(result.shape, (batch_size, 3))

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_with_negative_bounds(self, mock_branching):
        """Test with negative lower and upper bounds."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0, 1, 2]])

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 2)
        lAs = torch.randn(1, 2, 4)
        x_Ls = torch.tensor([[-1.0, -2.0, -0.5, -1.5]])
        x_Us = torch.tensor([[-0.5, -1.0, 0.0, -0.5]])
        rhs = torch.zeros(1, 2)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        # Should still return the expected indices
        self.assertEqual(result.shape, (1, 3))

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_with_asymmetric_bounds(self, mock_branching):
        """Test with asymmetric bounds (different ranges for each input)."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[1, 0, 2]])

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 2)
        lAs = torch.randn(1, 2, 3)
        # Different ranges: [0,1], [-10,10], [0,0.001]
        x_Ls = torch.tensor([[0.0, -10.0, 0.0]])
        x_Us = torch.tensor([[1.0, 10.0, 0.001]])
        rhs = torch.zeros(1, 2)

        result = input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        self.assertEqual(result.shape, (1, 3))


class TestInputBranchingDecisionsSplitDepth(unittest.TestCase):
    """Tests for split_depth parameter in input_branching_decisions."""

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_split_depth_is_three(self, mock_branching):
        """Test that split_depth is always 3."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0, 1, 2]])

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 2)
        lAs = torch.randn(1, 2, 5)
        x_Ls = torch.zeros(1, 5)
        x_Us = torch.ones(1, 5)
        rhs = torch.zeros(1, 2)

        input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        call_kwargs = mock_branching.call_args[1]
        self.assertEqual(call_kwargs['split_depth'], 3)


class TestInputBranchingDecisionsBranchingMethod(unittest.TestCase):
    """Tests for branching_method parameter in input_branching_decisions."""

    @patch('input_split.input_split_on_relu_domains.input_split_branching')
    def test_branching_method_is_sb(self, mock_branching):
        """Test that branching_method is always 'sb' (smart branching)."""
        from input_split.input_split_on_relu_domains import input_branching_decisions

        mock_branching.return_value = torch.tensor([[0, 1, 2]])

        wrapped_net = MagicMock()
        global_lbs = torch.randn(1, 2)
        lAs = torch.randn(1, 2, 5)
        x_Ls = torch.zeros(1, 5)
        x_Us = torch.ones(1, 5)
        rhs = torch.zeros(1, 2)

        input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs)

        call_kwargs = mock_branching.call_args[1]
        self.assertEqual(call_kwargs['branching_method'], 'sb')


if __name__ == '__main__':
    setup_module()
    unittest.main()
