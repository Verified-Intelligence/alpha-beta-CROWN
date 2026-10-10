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
"""
Unit tests for complete_verifier_func.py

This module tests the complete verifier main interface functions:
- bab: Branch-and-bound verification main function
- complete_verifier: Complete verification interface
- prepare_for_act_bab: Prepare batches for activation BaB
- _format_result_act_bab: Format BaB results

To run tests: python -m pytest complete_verifier/tests/test_complete_verifier_func.py -v
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import sys
import os

# Add the complete_verifier to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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


class TestPrepareForActBab(unittest.TestCase):
    """Tests for the prepare_for_act_bab function."""

    def test_basic_batch_preparation(self):
        """Test basic batch preparation with simple inputs."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 4
        total_specs = 8
        input_dim = 10
        output_dim = 5

        x = torch.randn(total_specs, input_dim)
        c = torch.randn(total_specs, output_dim)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'global_lb': None,
            'global_ub': None,
            'lower_bounds': None,
            'upper_bounds': None,
            'lA': None,
            'alphas': None,
            'mask': None,
            'refined_betas': None,
            'attack_examples': None,
            'attack_margins': None,
            'history': None,
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        self.assertEqual(batch_x.shape[0], batch_size)
        self.assertEqual(batch_c.shape[0], batch_size)
        self.assertEqual(batch_rhs.shape[0], batch_size)

    def test_batch_preparation_second_batch(self):
        """Test batch preparation for second batch."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 3
        total_specs = 8
        input_dim = 5

        x = torch.arange(total_specs * input_dim).float().view(total_specs, input_dim)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        # Second batch should start at index 3
        self.assertEqual(batch_x.shape[0], batch_size)
        # Verify the values are from the second batch
        expected_start = batch_size * input_dim
        self.assertEqual(batch_x[0, 0].item(), expected_start)

    def test_batch_preparation_with_global_bounds(self):
        """Test batch preparation with global lower/upper bounds."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'global_lb': torch.randn(total_specs, 3),
            'global_ub': torch.randn(total_specs, 3),
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        self.assertEqual(batch_ref['global_lb'].shape[0], batch_size)
        self.assertEqual(batch_ref['global_ub'].shape[0], batch_size)

    def test_batch_preparation_with_lower_upper_bounds(self):
        """Test batch preparation with intermediate bounds."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'lower_bounds': {
                'layer1': torch.randn(total_specs, 10),
                'layer2': torch.randn(total_specs, 5),
            },
            'upper_bounds': {
                'layer1': torch.randn(total_specs, 10),
                'layer2': torch.randn(total_specs, 5),
            },
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        self.assertEqual(batch_ref['lower_bounds']['layer1'].shape[0], batch_size)
        self.assertEqual(batch_ref['upper_bounds']['layer2'].shape[0], batch_size)

    def test_batch_preparation_with_lA(self):
        """Test batch preparation with lA dictionary."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'lA': {
                'layer1': torch.randn(total_specs, 3, 10),
                'layer2': torch.randn(total_specs, 3, 5),
            },
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        self.assertEqual(batch_ref['lA']['layer1'].shape[0], batch_size)
        self.assertEqual(batch_ref['lA']['layer2'].shape[0], batch_size)

    def test_batch_preparation_with_alphas(self):
        """Test batch preparation with alphas dictionary."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'alphas': {
                'layer1': {
                    'alpha': {
                        'key1': torch.randn(2, 3, total_specs, 5),
                        'key2': torch.randn(2, 3, total_specs, 10),
                    },
                    'alpha_lookup_idx': [0, 1, 2],
                    'alpha_indices': torch.tensor([1, 2]),
                    'init': True,
                }
            }
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        # Alpha batching is along dim 2
        self.assertEqual(batch_ref['alphas']['layer1']['alpha']['key1'].shape[2], batch_size)
        # Other items should be preserved
        self.assertEqual(batch_ref['alphas']['layer1']['alpha_lookup_idx'], [0, 1, 2])
        self.assertTrue(batch_ref['alphas']['layer1']['init'])

    def test_batch_preparation_with_mask(self):
        """Test batch preparation with mask dictionary."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'mask': {
                'layer1': [
                    torch.randn(total_specs, 10),  # Normal mask
                    None,  # None mask
                    torch.randn(1, 5),  # Single batch mask
                ]
            }
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        # First mask should be batched
        self.assertEqual(batch_ref['mask']['layer1'][0].shape[0], batch_size)
        # None mask should remain None
        self.assertIsNone(batch_ref['mask']['layer1'][1])
        # Single batch mask should remain unchanged
        self.assertEqual(batch_ref['mask']['layer1'][2].shape[0], 1)

    def test_batch_preparation_with_attack_examples(self):
        """Test batch preparation with attack examples and margins."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {
            'attack_examples': torch.randn(total_specs, 1, 5),
            'attack_margins': torch.randn(total_specs, 3),
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        self.assertEqual(batch_ref['attack_examples'].shape[0], batch_size)
        self.assertEqual(batch_ref['attack_margins'].shape[0], batch_size)

    def test_batch_preparation_with_refined_betas(self):
        """Test batch preparation preserves refined_betas."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        refined_betas = {'layer1': torch.randn(10)}
        reference_dict = {
            'refined_betas': refined_betas,
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        # refined_betas should be preserved (not batched)
        self.assertEqual(batch_ref['refined_betas'], refined_betas)

    def test_batch_preparation_with_history(self):
        """Test batch preparation preserves history."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        history = [{'step': 1}, {'step': 2}]
        reference_dict = {
            'history': history,
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        # history should be preserved (not batched)
        self.assertEqual(batch_ref['history'], history)

    def test_batch_preparation_empty_reference_dict(self):
        """Test batch preparation with empty reference dict."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=0
        )

        # All batch reference dict values should be None
        self.assertIsNone(batch_ref['global_lb'])
        self.assertIsNone(batch_ref['global_ub'])

    def test_batch_preparation_last_partial_batch(self):
        """Test batch preparation for last partial batch."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 3
        total_specs = 5  # Last batch will only have 2 elements

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)
        reference_dict = {}

        # Get last batch (idx 1)
        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        # Last batch should have remaining 2 elements
        self.assertEqual(batch_x.shape[0], 2)


class TestFormatResultActBab(unittest.TestCase):
    """Tests for the _format_result_act_bab function."""

    def test_format_safe_result(self):
        """Test formatting a safe result."""
        from complete_verifier_func import _format_result_act_bab

        result = (0.5, 100, 'safe')
        mock_model = MagicMock()
        vnnlib = MagicMock()

        formatted = _format_result_act_bab(result, mock_model, vnnlib)

        self.assertEqual(formatted, (0.5, 100, 'safe'))

    def test_format_unknown_result(self):
        """Test formatting an unknown result."""
        from complete_verifier_func import _format_result_act_bab

        result = (-0.1, 500, 'unknown')
        mock_model = MagicMock()
        vnnlib = MagicMock()

        formatted = _format_result_act_bab(result, mock_model, vnnlib)

        self.assertEqual(formatted, (-0.1, 500, 'unknown'))

    @patch('complete_verifier_func.check_and_save_cex')
    def test_format_unsafe_bab_result(self, mock_check_cex):
        """Test formatting an unsafe_bab result."""
        from complete_verifier_func import _format_result_act_bab

        mock_stats = MagicMock()
        mock_stats.counterexample = torch.randn(1, 10)

        result = (-1.0, 200, 'unsafe_bab', mock_stats)
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model.net.return_value = torch.randn(1, 5)
        vnnlib = MagicMock()

        mock_check_cex.return_value = ('unsafe', None)

        formatted = _format_result_act_bab(result, mock_model, vnnlib)

        self.assertEqual(formatted[0], -1.0)
        self.assertEqual(formatted[1], 200)
        self.assertEqual(formatted[2], 'unsafe')
        mock_check_cex.assert_called_once()

    @patch('complete_verifier_func.check_and_save_cex')
    @patch('complete_verifier_func.arguments.Config', {
        'attack': {'cex_path': '/tmp/cex'}
    })
    def test_format_unsafe_bab_uses_cex_path(self, mock_check_cex):
        """Test that unsafe_bab result uses correct cex_path."""
        from complete_verifier_func import _format_result_act_bab

        mock_stats = MagicMock()
        mock_stats.counterexample = torch.randn(1, 10)

        result = (-1.0, 200, 'unsafe_bab', mock_stats)
        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model.net.return_value = torch.randn(1, 5)
        vnnlib = MagicMock()

        mock_check_cex.return_value = ('unsafe', None)

        _format_result_act_bab(result, mock_model, vnnlib)

        # Check that cex_path was passed
        call_args = mock_check_cex.call_args
        self.assertEqual(call_args[0][3], '/tmp/cex')

    def test_format_result_truncates_extra_elements(self):
        """Test that format result truncates to 3 elements for non-unsafe."""
        from complete_verifier_func import _format_result_act_bab

        # Result with extra data
        result = (0.5, 100, 'safe', {'extra': 'data'})
        mock_model = MagicMock()
        vnnlib = MagicMock()

        formatted = _format_result_act_bab(result, mock_model, vnnlib)

        self.assertEqual(len(formatted), 3)




class TestCompleteVerifier(unittest.TestCase):
    """Tests for the complete_verifier function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_abcrown = MagicMock()
        self.mock_all_specs = MagicMock()
        self.mock_abcrown.vnnlib_handler.all_specs = self.mock_all_specs

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_safe_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with safe result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1  # Fixed time for simplicity

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'safe')
        self.assertEqual(len(bab_ret), 1)
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_unsafe_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with unsafe result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(-1.0, 50, 'unsafe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'unsafe-bab')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_unknown_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with unknown result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(-0.5, 200, 'unknown'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'unknown')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_timeout(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier when timeout is exceeded."""
        from complete_verifier_func import complete_verifier

        # Simulate that bab takes longer than timeout - use side_effect with enough values
        mock_time.side_effect = [0, 1, 102, 103, 104, 105]  # Last time > timeout_threshold

        self.mock_abcrown.bab = MagicMock(return_value=(0.1, 500, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Should return unknown due to timeout
        self.assertEqual(result, 'unknown')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_invalid_return_raises(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier raises on invalid bab return value."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'invalid_status'))
        model_ori = MagicMock()
        bab_ret = []

        with self.assertRaises(ValueError) as context:
            complete_verifier(
                self.mock_abcrown, model_ori,
                index=0, timeout_threshold=100,
                bab_ret=bab_ret, reference_dict={}
            )

        self.assertIn('invalid_status', str(context.exception))

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_records_bab_stats(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier records stats in bab_ret."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        complete_verifier(
            self.mock_abcrown, model_ori,
            index=5, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Check bab_ret contains [index, lb, num_domains, time]
        self.assertEqual(len(bab_ret), 1)
        self.assertEqual(bab_ret[0][0], 5)  # index
        self.assertEqual(bab_ret[0][1], 0.5)  # lb
        self.assertEqual(bab_ret[0][2], 100)  # num_domains
        # time is 0 because mock_time.return_value is constant
        self.assertIsInstance(bab_ret[0][3], (int, float))  # time

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': True}},  # Will trigger assertion
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_asserts_opt_interm_bounds(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier asserts enable_opt_interm_bounds is False."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        with self.assertRaises(AssertionError):
            complete_verifier(
                self.mock_abcrown, model_ori,
                index=0, timeout_threshold=100,
                bab_ret=bab_ret, reference_dict={}
            )

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    @patch('builtins.print')
    def test_complete_verifier_prints_progress(self, mock_print, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier prints verification progress."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        complete_verifier(
            self.mock_abcrown, model_ori,
            index=3, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Check that progress was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        found_instance_print = any('Instance 3' in call for call in print_calls)
        self.assertTrue(found_instance_print)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_prepare_for_act_bab_single_spec(self):
        """Test prepare_for_act_bab with single specification."""
        from complete_verifier_func import prepare_for_act_bab

        x = torch.randn(1, 10)
        c = torch.randn(1, 5)
        rhs = torch.randn(1)
        reference_dict = {}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size=1, batch_idx=0
        )

        self.assertEqual(batch_x.shape[0], 1)

    def test_prepare_for_act_bab_large_batch(self):
        """Test prepare_for_act_bab with large batch size."""
        from complete_verifier_func import prepare_for_act_bab

        total_specs = 100
        batch_size = 50

        x = torch.randn(total_specs, 10)
        c = torch.randn(total_specs, 5)
        rhs = torch.randn(total_specs)
        reference_dict = {'global_lb': torch.randn(total_specs, 5)}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        self.assertEqual(batch_x.shape[0], batch_size)
        self.assertEqual(batch_ref['global_lb'].shape[0], batch_size)

    def test_prepare_for_act_bab_empty_alphas(self):
        """Test prepare_for_act_bab with empty alphas dict."""
        from complete_verifier_func import prepare_for_act_bab

        x = torch.randn(4, 10)
        c = torch.randn(4, 5)
        rhs = torch.randn(4)
        reference_dict = {'alphas': {}}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size=2, batch_idx=0
        )

        self.assertEqual(batch_ref['alphas'], {})

    def test_prepare_for_act_bab_empty_mask(self):
        """Test prepare_for_act_bab with empty mask dict."""
        from complete_verifier_func import prepare_for_act_bab

        x = torch.randn(4, 10)
        c = torch.randn(4, 5)
        rhs = torch.randn(4)
        reference_dict = {'mask': {}}

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size=2, batch_idx=0
        )

        self.assertEqual(batch_ref['mask'], {})


class TestBatchingCorrectness(unittest.TestCase):
    """Tests for verifying batching correctness."""

    def test_batch_values_are_correct(self):
        """Test that batched values are actually from the correct batch."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 6

        # Create identifiable data
        x = torch.arange(total_specs * 5).float().view(total_specs, 5)
        c = torch.arange(total_specs * 3).float().view(total_specs, 3)
        rhs = torch.arange(total_specs).float()
        reference_dict = {}

        # Get batch 1 (indices 2, 3)
        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        # Verify first element of batch_x is from x[2]
        self.assertTrue(torch.equal(batch_x[0], x[2]))
        self.assertTrue(torch.equal(batch_x[1], x[3]))

        # Verify rhs values
        self.assertEqual(batch_rhs[0].item(), 2.0)
        self.assertEqual(batch_rhs[1].item(), 3.0)

    def test_alpha_batching_along_correct_dim(self):
        """Test that alphas are batched along dimension 2."""
        from complete_verifier_func import prepare_for_act_bab

        batch_size = 2
        total_specs = 4

        x = torch.randn(total_specs, 5)
        c = torch.randn(total_specs, 3)
        rhs = torch.randn(total_specs)

        # Create alpha with identifiable values
        alpha_data = torch.arange(2 * 3 * total_specs * 5).float().view(2, 3, total_specs, 5)
        reference_dict = {
            'alphas': {
                'layer': {
                    'alpha': {'key': alpha_data}
                }
            }
        }

        batch_x, batch_c, batch_rhs, batch_ref = prepare_for_act_bab(
            x, c, rhs, reference_dict, batch_size, batch_idx=1
        )

        # Should take indices 2, 3 along dim 2
        expected = alpha_data[:, :, 2:4, :]
        self.assertTrue(torch.equal(batch_ref['alphas']['layer']['alpha']['key'], expected))


if __name__ == '__main__':
    unittest.main()
