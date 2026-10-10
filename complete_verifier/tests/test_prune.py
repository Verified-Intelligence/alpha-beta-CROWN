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
"""Unit tests for prune.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPruneAlphas(unittest.TestCase):
    """Tests for prune_alphas function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        from prune import prune_alphas
        result = prune_alphas(None, ['layer1', 'layer2'])
        self.assertIsNone(result)

    def test_empty_alphas(self):
        """Test with empty alphas dict."""
        from prune import prune_alphas
        result = prune_alphas({}, ['layer1'])
        self.assertEqual(result, {})

    def test_prune_keeps_selected_layers(self):
        """Test that only selected layers are kept."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
                'layer2': torch.randn(2, 3),
                'layer3': torch.randn(2, 3),
            }
        }
        kept_names = ['layer1', 'layer3']
        result = prune_alphas(alpha, kept_names)

        self.assertIn('node1', result)
        self.assertIn('layer1', result['node1'])
        self.assertNotIn('layer2', result['node1'])
        self.assertIn('layer3', result['node1'])

    def test_prune_all_layers(self):
        """Test pruning when no layers are kept."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
                'layer2': torch.randn(2, 3),
            }
        }
        kept_names = []
        result = prune_alphas(alpha, kept_names)

        self.assertIn('node1', result)
        self.assertEqual(len(result['node1']), 0)

    def test_prune_multiple_nodes(self):
        """Test pruning with multiple nodes."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
                'layer2': torch.randn(2, 3),
            },
            'node2': {
                'layer1': torch.randn(2, 3),
                'layer3': torch.randn(2, 3),
            }
        }
        kept_names = ['layer1']
        result = prune_alphas(alpha, kept_names)

        self.assertIn('node1', result)
        self.assertIn('node2', result)
        self.assertIn('layer1', result['node1'])
        self.assertIn('layer1', result['node2'])
        self.assertNotIn('layer2', result['node1'])
        self.assertNotIn('layer3', result['node2'])

    def test_prune_keeps_tensor_values(self):
        """Test that kept tensor values are unchanged."""
        from prune import prune_alphas
        tensor = torch.randn(2, 3)
        alpha = {
            'node1': {
                'layer1': tensor,
            }
        }
        result = prune_alphas(alpha, ['layer1'])
        self.assertTrue(torch.equal(result['node1']['layer1'], tensor))

    def test_prune_nonexistent_layers(self):
        """Test keeping layers that don't exist in alpha."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
            }
        }
        kept_names = ['layer1', 'nonexistent_layer']
        result = prune_alphas(alpha, kept_names)

        self.assertIn('layer1', result['node1'])
        self.assertNotIn('nonexistent_layer', result['node1'])


class TestPruneAfterCROWNRecoverData(unittest.TestCase):
    """Tests for PruneAfterCROWN._recover_data method."""

    def test_recover_data_basic(self):
        """Test basic data recovery."""
        from prune import PruneAfterCROWN

        # Create a mock instance with unverified_indices
        mock_instance = MagicMock()
        mock_instance.unverified_indices = torch.tensor([0, 2])

        # Test data
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        full_size = 4
        fill_value = float('inf')
        recover_dim = 0

        # Call _recover_data as an instance method
        result = PruneAfterCROWN._recover_data(
            mock_instance, data, full_size, fill_value, recover_dim
        )

        self.assertEqual(result.shape, (4, 2))
        # Check that indices 0 and 2 have the original data
        self.assertTrue(torch.allclose(result[0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(result[2], torch.tensor([3.0, 4.0])))
        # Check that indices 1 and 3 have fill_value
        self.assertTrue((result[1] == float('inf')).all())
        self.assertTrue((result[3] == float('inf')).all())

    def test_recover_data_different_dim(self):
        """Test data recovery along different dimension."""
        from prune import PruneAfterCROWN

        mock_instance = MagicMock()
        mock_instance.unverified_indices = torch.tensor([1])

        data = torch.tensor([[1.0, 2.0, 3.0]])
        full_size = 3
        fill_value = 0.0
        recover_dim = 0

        result = PruneAfterCROWN._recover_data(
            mock_instance, data, full_size, fill_value, recover_dim
        )

        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.allclose(result[1], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue((result[0] == 0.0).all())
        self.assertTrue((result[2] == 0.0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_recover_data_accepts_indices_from_another_device(self):
        """Recovery follows the data device after results are moved to CPU."""
        from prune import PruneAfterCROWN

        mock_instance = MagicMock()
        mock_instance.unverified_indices = torch.tensor(
            [1, 3], device='cuda')
        data = torch.tensor([[10.0], [20.0]], device='cpu')

        result = PruneAfterCROWN._recover_data(
            mock_instance, data, full_size=4, fill_value=0, recover_dim=0)

        expected = torch.tensor([[0.0], [10.0], [0.0], [20.0]])
        self.assertEqual(result.device.type, 'cpu')
        self.assertTrue(torch.equal(result, expected))


class TestPruneAfterCROWNGetPrunedData(unittest.TestCase):
    """Tests for PruneAfterCROWN.get_pruned_data method."""

    def test_get_pruned_data_returns_tuple(self):
        """Test that get_pruned_data returns correct tuple."""
        from prune import PruneAfterCROWN

        # Create a mock instance with required attributes
        mock_instance = MagicMock()
        mock_instance.x_pruned = torch.randn(2, 3)
        mock_instance.c_pruned = torch.randn(2, 4)
        mock_instance.rhs_pruned = torch.randn(2, 4)
        mock_instance.or_spec_size_pruned = None
        mock_instance.stop_criterion_func_pruned = lambda x: x

        # Call get_pruned_data as an instance method
        result = PruneAfterCROWN.get_pruned_data(mock_instance)

        self.assertEqual(len(result), 5)
        self.assertIs(result[0], mock_instance.x_pruned)
        self.assertIs(result[1], mock_instance.c_pruned)
        self.assertIs(result[2], mock_instance.rhs_pruned)
        self.assertIsNone(result[3])


class TestPruneAlphasEdgeCases(unittest.TestCase):
    """Additional edge case tests for prune_alphas."""

    def test_prune_with_various_value_types(self):
        """Test pruning with various value types (not just tensors)."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
                'layer2': [1, 2, 3],  # Not a tensor
            }
        }
        kept_names = ['layer1', 'layer2']
        result = prune_alphas(alpha, kept_names)

        self.assertIn('layer1', result['node1'])
        self.assertIn('layer2', result['node1'])

    def test_prune_single_layer_single_node(self):
        """Test pruning with single layer and single node."""
        from prune import prune_alphas
        alpha = {
            'node1': {
                'layer1': torch.randn(2, 3),
            }
        }
        kept_names = ['layer1']
        result = prune_alphas(alpha, kept_names)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result['node1']), 1)

    def test_prune_empty_kept_names_different_structure(self):
        """Test pruning with empty kept_names on complex structure."""
        from prune import prune_alphas
        alpha = {
            'node1': {'layer1': torch.randn(2, 3)},
            'node2': {'layer1': torch.randn(2, 3), 'layer2': torch.randn(2, 3)},
            'node3': {},
        }
        kept_names = []
        result = prune_alphas(alpha, kept_names)

        self.assertEqual(len(result['node1']), 0)
        self.assertEqual(len(result['node2']), 0)
        self.assertEqual(len(result['node3']), 0)


class TestPruneAfterCROWNInit(unittest.TestCase):
    """Tests for PruneAfterCROWN initialization with mocks."""

    @patch('prune.arguments')
    def test_init_with_stop_criterion_batch_any(self, mock_arguments):
        """Test initialization with stop_criterion_batch_any."""
        from prune import PruneAfterCROWN
        from auto_LiRPA.utils import stop_criterion_batch_any

        mock_arguments.Config = {'solver': {'optimize_disjuncts_separately': True}}

        # Create mocks
        mock_net = MagicMock()
        mock_net.final_name = 'output'
        mock_net.get_enabled_opt_act.return_value = []

        mock_x = MagicMock()
        mock_x.ptb = MagicMock()
        mock_x.ptb.x_L = torch.randn(4, 3)
        mock_x.ptb.x_U = torch.randn(4, 3)
        mock_x.data = torch.randn(4, 3)

        c = torch.randn(4, 2)
        rhs = torch.zeros(4, 2)
        lb = torch.tensor([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

        aux_reference_bounds = {}

        # Create instance
        pruner = PruneAfterCROWN(
            mock_net, mock_x, c, rhs, lb, aux_reference_bounds,
            stop_criterion=stop_criterion_batch_any, or_spec_size=None
        )

        self.assertIsNotNone(pruner.unverified_indices)
        self.assertEqual(pruner.prune_dim, 0)

    @patch('prune.arguments')
    def test_init_with_stop_criterion_all(self, mock_arguments):
        """Test initialization with stop_criterion_all."""
        from prune import PruneAfterCROWN
        from auto_LiRPA.utils import stop_criterion_all

        mock_arguments.Config = {'solver': {'optimize_disjuncts_separately': False}}

        mock_net = MagicMock()
        mock_net.final_name = 'output'
        mock_net.get_enabled_opt_act.return_value = []

        mock_x = MagicMock()
        mock_x.ptb = MagicMock()

        c = torch.randn(1, 4)
        rhs = torch.zeros(1, 4)
        lb = torch.tensor([[0.5, -0.5, 0.5, -0.5]])

        aux_reference_bounds = {}

        pruner = PruneAfterCROWN(
            mock_net, mock_x, c, rhs, lb, aux_reference_bounds,
            stop_criterion=stop_criterion_all, or_spec_size=None
        )

        self.assertIsNotNone(pruner.unverified_indices)
        self.assertEqual(pruner.prune_dim, 1)


class TestPruneAfterCROWNRecover(unittest.TestCase):
    """Tests for PruneAfterCROWN.recover method."""

    def test_recover_with_full_alpha_info_true(self):
        """Test recover method with full_alpha_info=True calls _recover_data correctly.

        The recover method should call _recover_data for:
        - lb and ub entries matching final_name
        - all lA entries
        - alphas with full_alpha_info format (node -> 'alpha' -> spec_name)
        - mask entries matching final_name
        - input_split_idx entries matching final_name
        """
        from prune import PruneAfterCROWN

        mock_instance = MagicMock()
        mock_instance.prune_dim = 0
        mock_instance.prune_dim_alpha = 2
        mock_instance.optimize_disjuncts_separately = False
        mock_instance.lb_ori = torch.randn(4, 3)
        mock_instance.unverified_indices = torch.tensor([0, 2])
        mock_instance.net = MagicMock()
        mock_instance.net.final_name = 'output'

        # Use MagicMock to track calls to _recover_data
        mock_instance._recover_data = MagicMock(side_effect=lambda data, *args: data)

        lb = {'output': torch.randn(2, 3)}
        ub = {'output': torch.randn(2, 3)}
        lA = {'layer1': torch.randn(2, 5)}
        alphas = {
            'node1': {
                'alpha': {'output': torch.randn(2, 4, 2, 3)}
            }
        }
        mask = {'output': [torch.ones(2, 3, dtype=torch.bool)]}
        input_split_idx = {'output': torch.zeros(2, dtype=torch.long)}

        # Call recover (using unbound method)
        PruneAfterCROWN.recover(
            mock_instance, lb, ub, lA, alphas, mask, input_split_idx,
            full_alpha_info=True
        )

        # Verify _recover_data was called
        self.assertTrue(mock_instance._recover_data.called,
                        "_recover_data should be called by recover method")

        # Should be called for: lb['output'], ub['output'], lA['layer1'],
        # alphas['node1']['alpha']['output'], mask['output'][0], input_split_idx['output']
        # Total: 6 calls
        self.assertEqual(mock_instance._recover_data.call_count, 6)

    def test_recover_with_full_alpha_info_false(self):
        """Test recover method with full_alpha_info=False calls _recover_data correctly.

        When full_alpha_info=False, alphas have format: node -> spec_name -> tensor
        (without the intermediate 'alpha' key).
        """
        from prune import PruneAfterCROWN

        mock_instance = MagicMock()
        mock_instance.prune_dim = 0
        mock_instance.prune_dim_alpha = 2
        mock_instance.optimize_disjuncts_separately = False
        mock_instance.lb_ori = torch.randn(4, 3)
        mock_instance.unverified_indices = torch.tensor([0, 2])
        mock_instance.net = MagicMock()
        mock_instance.net.final_name = 'output'

        # Use MagicMock to track calls to _recover_data
        mock_instance._recover_data = MagicMock(side_effect=lambda data, *args: data)

        lb = {'output': torch.randn(2, 3)}
        ub = {'output': torch.randn(2, 3)}
        lA = {'layer1': torch.randn(2, 5)}
        alphas = {
            'node1': {'output': torch.randn(2, 4, 2, 3)}
        }
        mask = {'output': [torch.ones(2, 3, dtype=torch.bool)]}
        input_split_idx = {'output': torch.zeros(2, dtype=torch.long)}

        PruneAfterCROWN.recover(
            mock_instance, lb, ub, lA, alphas, mask, input_split_idx,
            full_alpha_info=False
        )

        # Verify _recover_data was called
        self.assertTrue(mock_instance._recover_data.called,
                        "_recover_data should be called by recover method")

        # Same number of calls as full_alpha_info=True for this test setup
        self.assertEqual(mock_instance._recover_data.call_count, 6)


class TestPruneAfterCROWNAttributes(unittest.TestCase):
    """Tests for PruneAfterCROWN attribute setting."""

    @patch('prune.arguments')
    def test_prune_dim_alpha_calculation(self, mock_arguments):
        """Test that prune_dim_alpha is calculated correctly."""
        from prune import PruneAfterCROWN
        from auto_LiRPA.utils import stop_criterion_batch_any

        mock_arguments.Config = {'solver': {'optimize_disjuncts_separately': True}}

        mock_net = MagicMock()
        mock_net.final_name = 'output'
        mock_net.get_enabled_opt_act.return_value = []

        mock_x = MagicMock()
        mock_x.ptb = MagicMock()
        mock_x.ptb.x_L = torch.randn(4, 3)
        mock_x.ptb.x_U = torch.randn(4, 3)
        mock_x.data = torch.randn(4, 3)

        c = torch.randn(4, 2)
        rhs = torch.zeros(4, 2)
        lb = torch.randn(4, 2)

        pruner = PruneAfterCROWN(
            mock_net, mock_x, c, rhs, lb, {},
            stop_criterion=stop_criterion_batch_any
        )

        # prune_dim_alpha = 2 - prune_dim
        # When prune_dim = 0, prune_dim_alpha = 2
        self.assertEqual(pruner.prune_dim_alpha, 2 - pruner.prune_dim)

    @patch('prune.arguments')
    def test_original_bounds_stored(self, mock_arguments):
        """Test that original lb and rhs are stored."""
        from prune import PruneAfterCROWN
        from auto_LiRPA.utils import stop_criterion_batch_any

        mock_arguments.Config = {'solver': {'optimize_disjuncts_separately': True}}

        mock_net = MagicMock()
        mock_net.final_name = 'output'
        mock_net.get_enabled_opt_act.return_value = []

        mock_x = MagicMock()
        mock_x.ptb = MagicMock()
        mock_x.ptb.x_L = torch.randn(4, 3)
        mock_x.ptb.x_U = torch.randn(4, 3)
        mock_x.data = torch.randn(4, 3)

        c = torch.randn(4, 2)
        rhs = torch.randn(4, 2)
        lb = torch.randn(4, 2)

        pruner = PruneAfterCROWN(
            mock_net, mock_x, c, rhs, lb, {},
            stop_criterion=stop_criterion_batch_any
        )

        # Check that original values are stored
        self.assertTrue(torch.equal(pruner.lb_ori, lb))
        self.assertTrue(torch.equal(pruner.rhs_ori, rhs))


if __name__ == '__main__':
    unittest.main()
