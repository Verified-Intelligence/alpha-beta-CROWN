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
"""Unit tests for incomplete_verifier_func.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestSpecHandlerSetUnverifiedOrMask(unittest.TestCase):
    """Tests for SpecHandler.set_unverified_or_mask method."""

    def _create_mock_spec_handler(self, spec_type, or_spec_size):
        """Create a mock SpecHandler with minimal setup."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType
        from auto_LiRPA.utils import stop_criterion_batch_any, stop_criterion_all, stop_criterion_general

        mock_handler = MagicMock(spec=SpecHandler)
        mock_handler.spec_type = spec_type
        mock_handler.or_spec_size = or_spec_size
        mock_handler.rhs = torch.zeros(or_spec_size.shape[0], or_spec_size.max().item())

        if spec_type == SpecType.SINGLE_OR:
            mock_handler.stop_criterion = stop_criterion_batch_any
        elif spec_type == SpecType.SINGLE_AND_IN_MULTI_ORS:
            mock_handler.stop_criterion = stop_criterion_batch_any
        elif spec_type == SpecType.FIXED_NUM_ANDS_IN_MULTI_ORS:
            mock_handler.stop_criterion = stop_criterion_batch_any
        else:
            mock_handler.stop_criterion = stop_criterion_general

        return mock_handler

    def test_set_unverified_or_mask_batch_any_all_verified(self):
        """Test set_unverified_or_mask with batch_any when all verified."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        # Create a real SpecHandler-like object
        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([[0.0], [0.0]])  # 2 ORs, 1 AND each
        handler.or_spec_size = torch.tensor([1, 1])

        # lb > rhs means verified
        lb = torch.tensor([[1.0], [1.0]])  # Both verified

        # Call the actual method
        SpecHandler.set_unverified_or_mask(handler, lb)

        # All should be verified (unverified_or_mask should be False)
        self.assertFalse(handler.unverified_or_mask.any())

    def test_set_unverified_or_mask_batch_any_none_verified(self):
        """Test set_unverified_or_mask with batch_any when none verified."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([[0.0], [0.0]])
        handler.or_spec_size = torch.tensor([1, 1])

        # lb <= rhs means not verified
        lb = torch.tensor([[-1.0], [-1.0]])  # Neither verified

        SpecHandler.set_unverified_or_mask(handler, lb)

        # All should be unverified
        self.assertTrue(handler.unverified_or_mask.all())

    def test_set_unverified_or_mask_batch_any_partial_verified(self):
        """Test set_unverified_or_mask with batch_any when partially verified."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([[0.0], [0.0]])
        handler.or_spec_size = torch.tensor([1, 1])

        # First verified, second not
        lb = torch.tensor([[1.0], [-1.0]])

        SpecHandler.set_unverified_or_mask(handler, lb)

        # First OR verified (mask=False), second not (mask=True)
        self.assertFalse(handler.unverified_or_mask[0].item())
        self.assertTrue(handler.unverified_or_mask[1].item())

    def test_set_unverified_or_mask_all_criterion(self):
        """Test set_unverified_or_mask with stop_criterion_all."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_all

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_all
        handler.rhs = torch.tensor([[0.0, 0.0]])  # 1 batch, 2 specs
        handler.or_spec_size = torch.tensor([2])

        # Both specs verified
        lb = torch.tensor([[1.0, 1.0]])

        SpecHandler.set_unverified_or_mask(handler, lb)

        # All verified (mask should be False)
        self.assertFalse(handler.unverified_or_mask.any())

    def test_set_unverified_or_mask_general_criterion(self):
        """Test set_unverified_or_mask with stop_criterion_general."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_general

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_general
        handler.rhs = torch.tensor([[0.0, 0.0, 0.0]])  # 3 specs flattened
        handler.or_spec_size = torch.tensor([2, 1])  # First OR has 2 ANDs, second has 1

        # First OR: both ANDs fail -> unverified
        # Second OR: single AND passes -> verified
        lb = torch.tensor([[-1.0, -1.0, 1.0]])

        SpecHandler.set_unverified_or_mask(handler, lb, or_spec_size=handler.or_spec_size)

        # First unverified, second verified
        self.assertTrue(handler.unverified_or_mask[0].item())
        self.assertFalse(handler.unverified_or_mask[1].item())

    def test_set_unverified_or_mask_unknown_criterion_raises(self):
        """Test set_unverified_or_mask raises for unknown stop criterion."""
        from incomplete_verifier_func import SpecHandler

        handler = MagicMock()
        handler.stop_criterion = lambda x: x  # Unknown criterion
        handler.rhs = torch.tensor([[0.0]])
        handler.or_spec_size = torch.tensor([1])

        lb = torch.tensor([[1.0]])

        with self.assertRaises(ValueError):
            SpecHandler.set_unverified_or_mask(handler, lb)

    def test_set_unverified_or_mask_custom_rhs(self):
        """Test set_unverified_or_mask with custom rhs parameter."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([[0.0]])  # Default rhs
        handler.or_spec_size = torch.tensor([1])

        lb = torch.tensor([[0.5]])
        custom_rhs = torch.tensor([[1.0]])  # Higher threshold

        # With default rhs (0.0), lb=0.5 is verified
        # With custom rhs (1.0), lb=0.5 is NOT verified
        SpecHandler.set_unverified_or_mask(handler, lb, rhs=custom_rhs)

        self.assertTrue(handler.unverified_or_mask[0].item())


class TestSpecHandlerPrune(unittest.TestCase):
    """Tests for SpecHandler._prune method."""

    def test_prune_basic(self):
        """Test _prune basic functionality."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([1, 3])
        handler.spec_type = SpecType.SINGLE_OR
        handler.or_spec_size = torch.tensor([2, 2, 2, 2])

        # Data with 4 ORs
        data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        result = SpecHandler._prune(handler, data, dim=0)

        # Should select indices 1 and 3
        expected = torch.tensor([[3, 4], [7, 8]])
        self.assertTrue(torch.equal(result, expected))

    def test_prune_along_dim_2(self):
        """Test _prune along dimension 2 (for alphas)."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([0, 2])
        handler.spec_type = SpecType.SINGLE_OR
        handler.or_spec_size = torch.tensor([1, 1, 1])

        # Alpha-like data: [alpha_size, spec, num_or, output]
        data = torch.randn(2, 1, 3, 5)

        result = SpecHandler._prune(handler, data, dim=2)

        self.assertEqual(result.shape, (2, 1, 2, 5))

    def test_prune_with_and_size_variable(self):
        """Test _prune with prune_and_size for variable AND sizes."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([0, 1])
        handler.spec_type = SpecType.VARIABLE_NUM_ANDS_IN_EACH_OR
        handler.or_spec_size = torch.tensor([2, 1])  # Max is 2 after prune

        # Data padded to max AND size of 3
        data = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]]
        ])  # Shape: [3, 2, 3]

        result = SpecHandler._prune(handler, data, dim=0, prune_and_size=True)

        # Should prune to indices [0, 1] and reduce AND dim to max(2, 1) = 2
        self.assertEqual(result.shape[0], 2)  # 2 unverified ORs
        self.assertEqual(result.shape[1], 2)  # max AND size is 2

    def test_prune_without_and_size_variable(self):
        """Test _prune without prune_and_size doesn't reduce AND dim."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([0, 1])
        handler.spec_type = SpecType.VARIABLE_NUM_ANDS_IN_EACH_OR
        handler.or_spec_size = torch.tensor([2, 1])

        data = torch.randn(3, 3, 5)

        result = SpecHandler._prune(handler, data, dim=0, prune_and_size=False)

        # Should not reduce AND dimension
        self.assertEqual(result.shape[1], 3)


class TestSpecHandlerExpandIntermediate(unittest.TestCase):
    """Tests for SpecHandler.expand_intermediate method."""

    def test_expand_intermediate_does_nothing_when_separate(self):
        """Test expand_intermediate does nothing when optimize_disjuncts_separately."""
        from incomplete_verifier_func import SpecHandler

        handler = MagicMock()
        handler.optimize_disjuncts_separately = True

        ret = {'lower_bounds': {}, 'upper_bounds': {}, 'alphas': {}}

        SpecHandler.expand_intermediate(handler, ret)

        # Should return without modification
        self.assertEqual(ret['lower_bounds'], {})

    def test_expand_intermediate_expands_bounds(self):
        """Test expand_intermediate expands intermediate bounds."""
        from incomplete_verifier_func import SpecHandler

        handler = MagicMock()
        handler.optimize_disjuncts_separately = False
        handler.num_or = 3

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'model': mock_model,
            'lower_bounds': {
                'layer1': torch.randn(1, 10),  # Shared
                'output': torch.randn(3, 5),   # Final - don't expand
            },
            'upper_bounds': {
                'layer1': torch.randn(1, 10),
                'output': torch.randn(3, 5),
            },
            'alphas': None
        }

        SpecHandler.expand_intermediate(handler, ret)

        # Intermediate should be expanded
        self.assertEqual(ret['lower_bounds']['layer1'].shape[0], 3)
        # Final should not be expanded
        self.assertEqual(ret['lower_bounds']['output'].shape[0], 3)

    def test_expand_intermediate_expands_alphas(self):
        """Test expand_intermediate expands alpha values."""
        from incomplete_verifier_func import SpecHandler

        handler = MagicMock()
        handler.optimize_disjuncts_separately = False
        handler.num_or = 4

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'model': mock_model,
            'lower_bounds': {'layer1': torch.randn(1, 10), 'output': torch.randn(4, 5)},
            'upper_bounds': {'layer1': torch.randn(1, 10), 'output': torch.randn(4, 5)},
            'alphas': {
                'layer1': {
                    'alpha': {
                        'layer1': torch.randn(2, 3, 1, 10),  # Shared intermediate
                        'output': torch.randn(2, 3, 4, 5),   # Final - don't expand
                    }
                }
            }
        }

        SpecHandler.expand_intermediate(handler, ret)

        # Intermediate alpha should be expanded
        self.assertEqual(ret['alphas']['layer1']['alpha']['layer1'].shape[2], 4)
        # Final alpha should not change
        self.assertEqual(ret['alphas']['layer1']['alpha']['output'].shape[2], 4)


class TestSpecHandlerPruneAttackRet(unittest.TestCase):
    """Tests for SpecHandler.prune_attack_ret method."""

    def test_prune_attack_ret_all_none(self):
        """Test prune_attack_ret when all inputs are None."""
        from incomplete_verifier_func import SpecHandler

        handler = MagicMock()

        result = SpecHandler.prune_attack_ret(handler, None, None, None)

        self.assertEqual(result, (None, None, None))

    def test_prune_attack_ret_with_data(self):
        """Test prune_attack_ret with actual data."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([1, 2])
        handler.spec_type = SpecType.SINGLE_OR
        handler.or_spec_size = torch.tensor([1, 1, 1])

        # Mock _prune to return sliced data
        def mock_prune(data, dim, prune_and_size):
            return data.index_select(dim, handler.unverified_or_indices)

        handler._prune = mock_prune

        attack_examples = torch.randn(3, 1, 28, 28)
        attack_margins = torch.randn(3, 10)
        all_adv_candidates = torch.randn(3, 5, 28, 28)

        result = SpecHandler.prune_attack_ret(
            handler, attack_examples, attack_margins, all_adv_candidates
        )

        ex, margins, adv = result
        self.assertEqual(ex.shape[0], 2)
        self.assertEqual(margins.shape[0], 2)
        self.assertEqual(adv.shape[0], 2)

    def test_prune_attack_ret_partial_none(self):
        """Test prune_attack_ret when some inputs are None."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.unverified_or_indices = torch.tensor([0])
        handler.spec_type = SpecType.SINGLE_OR
        handler.or_spec_size = torch.tensor([1, 1])

        def mock_prune(data, dim, prune_and_size):
            return data.index_select(dim, handler.unverified_or_indices)

        handler._prune = mock_prune

        attack_examples = torch.randn(2, 1, 10)

        result = SpecHandler.prune_attack_ret(handler, attack_examples, None, None)

        ex, margins, adv = result
        self.assertEqual(ex.shape[0], 1)
        self.assertIsNone(margins)
        self.assertIsNone(adv)


class TestSpecHandlerAdhocProcessForMip(unittest.TestCase):
    """Tests for SpecHandler.adhoc_process_for_mip method."""

    def test_adhoc_process_for_mip_single_and_multi_ors(self):
        """Test adhoc_process_for_mip for SINGLE_AND_IN_MULTI_ORS."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_AND_IN_MULTI_ORS
        handler.same_x_range = True
        handler.optimize_disjuncts_separately = False
        handler.num_or = 2

        mock_all_specs = MagicMock()
        mock_all_specs.get.return_value = (
            torch.randn(2, 10),  # x
            torch.randn(2, 1, 5),  # c
            torch.zeros(2, 1),  # rhs
            torch.tensor([1, 1]),  # or_spec_size
            True,  # same_x_range
            True,  # same_or_spec_size
        )
        handler.vnnlib_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model.final_name = 'output'
        mock_model.net.__getitem__ = MagicMock(return_value=MagicMock())

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(2, 1),
        }

        SpecHandler.adhoc_process_for_mip(handler, ret)

        # Model x should be [1, ...]
        self.assertEqual(mock_model.x.shape[0], 1)

    def test_adhoc_process_for_mip_single_or_single_and(self):
        """Test adhoc_process_for_mip for SINGLE_OR with single AND."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_OR
        handler.same_x_range = True
        handler.optimize_disjuncts_separately = False
        handler.num_or = 1

        mock_all_specs = MagicMock()
        mock_all_specs.get.return_value = (
            torch.randn(1, 10),
            torch.randn(1, 1, 5),
            torch.zeros(1, 1),
            torch.tensor([1]),
            True,
            True,
        )
        handler.vnnlib_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model.final_name = 'output'
        mock_model.net.__getitem__ = MagicMock(return_value=MagicMock())

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(1, 1),
        }

        SpecHandler.adhoc_process_for_mip(handler, ret)

        # Should set model.x and model.c
        self.assertIsNotNone(mock_model.x)
        self.assertIsNotNone(mock_model.c)

    def test_adhoc_process_for_mip_single_or_multiple_ands(self):
        """Test adhoc_process_for_mip for SINGLE_OR with multiple ANDs."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_OR
        handler.same_x_range = True
        handler.optimize_disjuncts_separately = False
        handler.num_or = 1

        mock_all_specs = MagicMock()
        x = torch.randn(1, 10)
        c = torch.randn(1, 3, 5)  # 3 ANDs
        mock_all_specs.get.return_value = (
            x, c, torch.zeros(1, 3),
            torch.tensor([3]),  # 3 ANDs in single OR
            True, True,
        )
        handler.vnnlib_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_model.final_name = 'output'
        mock_model.net.__getitem__ = MagicMock(return_value=MagicMock())

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(1, 3),  # 3 ANDs
        }

        SpecHandler.adhoc_process_for_mip(handler, ret)

        # For SINGLE_OR with multiple ANDs, should use original x and c
        self.assertTrue(torch.equal(mock_model.x, x))
        self.assertTrue(torch.equal(mock_model.c, c))

    def test_adhoc_process_for_mip_unsupported_spec_raises(self):
        """Test adhoc_process_for_mip raises for unsupported spec type."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.FIXED_NUM_ANDS_IN_MULTI_ORS
        handler.same_x_range = True
        handler.optimize_disjuncts_separately = False

        mock_all_specs = MagicMock()
        mock_all_specs.get.return_value = (
            torch.randn(2, 10), torch.randn(2, 2, 5), torch.zeros(2, 2),
            torch.tensor([2, 2]), True, True,
        )
        handler.vnnlib_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(2, 2),
        }

        with self.assertRaises(ValueError):
            SpecHandler.adhoc_process_for_mip(handler, ret)

    def test_adhoc_process_for_mip_asserts_same_x_range(self):
        """Test adhoc_process_for_mip asserts same_x_range is True."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_OR
        handler.same_x_range = False  # Will fail assertion
        handler.optimize_disjuncts_separately = False

        ret = {'model': MagicMock(), 'global_lb': torch.randn(1, 1)}

        with self.assertRaises(AssertionError):
            SpecHandler.adhoc_process_for_mip(handler, ret)


class TestSpecHandlerAdhocPostProcessForMip(unittest.TestCase):
    """Tests for SpecHandler.adhoc_post_process_for_mip method."""

    def test_adhoc_post_process_for_mip_single_and(self):
        """Test adhoc_post_process_for_mip for single AND specs."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_AND_IN_MULTI_ORS
        handler.num_or = 2
        handler.unverified_or_mask = torch.tensor([True, False])
        handler.unverified_or_indices = torch.tensor([0])
        handler.or_spec_size = torch.tensor([1, 1])

        mock_all_specs = MagicMock()
        mock_all_specs.rhs = torch.tensor([[0.0], [0.0]])
        handler.vnnlib_handler.all_specs = mock_all_specs
        handler.vnnlib_handler.prune_verified_or_specs = MagicMock()

        mock_model = MagicMock()
        mock_model.final_name = 'output'
        mock_node = MagicMock()
        mock_model.net.__getitem__ = MagicMock(return_value=mock_node)

        ret = {
            'model': mock_model,
            'global_lb': torch.tensor([[1.0], [-1.0]]),  # First verified, second not
            'global_ub': torch.tensor([[2.0], [0.0]]),
            'lower_bounds': {'output': torch.tensor([[1.0], [-1.0]])},
            'upper_bounds': {'output': torch.tensor([[2.0], [0.0]])},
            'lA': None,
            'alphas': None,
            'mask': None,
            'input_split_idx': None,
        }

        # Mock the methods that get called
        handler.set_unverified_or_mask = MagicMock()
        handler.prune_verified_or_specs = MagicMock()

        SpecHandler.adhoc_post_process_for_mip(handler, ret)

        # Should set lower on model
        handler.set_unverified_or_mask.assert_called_once()
        handler.prune_verified_or_specs.assert_called_once()

    def test_adhoc_post_process_for_mip_single_or_multiple_ands(self):
        """Test adhoc_post_process_for_mip for SINGLE_OR with multiple ANDs."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.spec_type = SpecType.SINGLE_OR
        handler.num_or = 1

        mock_all_specs = MagicMock()
        mock_all_specs.rhs = torch.tensor([[0.0, 0.0, 0.0]])  # 3 ANDs
        handler.vnnlib_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.final_name = 'output'
        mock_node = MagicMock()
        mock_model.net.__getitem__ = MagicMock(return_value=mock_node)

        ret = {
            'model': mock_model,
            'global_lb': torch.tensor([[1.0, 0.5, -0.5]]),  # 3 ANDs
        }

        handler.set_unverified_or_mask = MagicMock()
        handler.prune_verified_or_specs = MagicMock()

        SpecHandler.adhoc_post_process_for_mip(handler, ret)

        # Should set model.net[final_name].lower
        self.assertTrue(torch.equal(mock_node.lower, ret['global_lb']))
        # Should call set_unverified_or_mask and prune_verified_or_specs
        handler.set_unverified_or_mask.assert_called_once()
        handler.prune_verified_or_specs.assert_called_once()


class TestSpecHandlerPostProcess(unittest.TestCase):
    """Tests for SpecHandler.post_process method."""

    def test_post_process_none_type(self):
        """Test post_process with NONE post-processing type."""
        from incomplete_verifier_func import SpecHandler, PostProcessingType, SpecType

        handler = MagicMock()
        handler.post_processing_type = PostProcessingType.NONE
        handler.num_or = 2
        handler.or_spec_size = torch.tensor([1, 1])
        handler.x = torch.randn(2, 10)
        handler.c = torch.randn(2, 1, 5)
        handler.rhs = torch.zeros(2, 1)
        handler.unverified_or_mask = torch.tensor([True, True])
        handler.unverified_or_indices = torch.tensor([0, 1])
        handler.optimize_disjuncts_separately = True
        handler.vnnlib_handler = MagicMock()

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'global_lb': torch.randn(2, 1),
            'global_ub': torch.randn(2, 1),
            'lower_bounds': {'output': torch.randn(2, 1)},
            'upper_bounds': {'output': torch.randn(2, 1)},
            'lA': None,
            'alphas': None,
            'mask': None,
            'input_split_idx': None,
        }

        # Mock prune_verified_or_specs
        handler.prune_verified_or_specs = MagicMock()
        handler._prune = lambda data, dim, prune_and: data

        result = SpecHandler.post_process(handler, mock_model, ret)

        self.assertIn('model', result)
        self.assertEqual(result['model'], mock_model)

    def test_post_process_reshape_type(self):
        """Test post_process with RESHAPE post-processing type."""
        from incomplete_verifier_func import SpecHandler, PostProcessingType

        handler = MagicMock()
        handler.post_processing_type = PostProcessingType.RESHAPE
        handler.num_or = 3
        handler.or_spec_size = torch.tensor([2, 2, 2])  # Each OR has 2 ANDs
        handler.x = torch.randn(1, 10)
        handler.c = torch.randn(1, 6, 5)
        handler.rhs = torch.zeros(1, 6)
        handler.unverified_or_mask = torch.tensor([True, True, True])
        handler.unverified_or_indices = torch.tensor([0, 1, 2])
        handler.optimize_disjuncts_separately = False
        handler.vnnlib_handler = MagicMock()

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'global_lb': torch.randn(1, 6),  # Flattened
            'global_ub': torch.randn(1, 6),
            'lower_bounds': {'output': torch.randn(1, 6)},
            'upper_bounds': {'output': torch.randn(1, 6)},
            'lA': {'layer1': torch.randn(1, 6, 10)},
            'alphas': None,
            'mask': None,
            'input_split_idx': None,
        }

        handler.prune_verified_or_specs = MagicMock()
        handler._prune = lambda data, dim, prune_and: data

        result = SpecHandler.post_process(handler, mock_model, ret)

        # Should reshape from [1, 6] to [3, 2]
        self.assertEqual(result['global_lb'].shape, (3, 2))
        self.assertEqual(result['lA']['layer1'].shape, (3, 2, 10))


class TestSpecHandlerInitSpecTypes(unittest.TestCase):
    """Tests for SpecHandler initialization with different spec types."""

    def _create_mock_vnnlib_handler(self, x, c, rhs, or_spec_size, same_x_range, same_or_spec_size):
        """Create a mock vnnlib handler with all_specs."""
        mock_handler = MagicMock()
        mock_all_specs = MagicMock()
        mock_all_specs.get.return_value = (x, c, rhs, or_spec_size, same_x_range, same_or_spec_size)
        mock_handler.all_specs = mock_all_specs
        return mock_handler

    @patch('incomplete_verifier_func.arguments')
    def test_init_single_or(self, mock_arguments):
        """Test SpecHandler init with SINGLE_OR spec type."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': False,
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(1, 10)
        c = torch.randn(1, 2, 5)  # 1 OR, 2 ANDs
        rhs = torch.zeros(1, 2)
        or_spec_size = torch.tensor([2])

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, True, True
        )

        handler = SpecHandler(mock_vnnlib)

        self.assertEqual(handler.spec_type, SpecType.SINGLE_OR)
        self.assertEqual(handler.post_processing_type, PostProcessingType.NONE)

    @patch('incomplete_verifier_func.arguments')
    def test_init_single_and_in_multi_ors_separate(self, mock_arguments):
        """Test SpecHandler init with SINGLE_AND_IN_MULTI_ORS, optimize separately."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': True,
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(3, 10)  # 3 ORs
        c = torch.randn(3, 1, 5)  # Each OR has 1 AND
        rhs = torch.zeros(3, 1)
        or_spec_size = torch.tensor([1, 1, 1])

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, True, True
        )

        handler = SpecHandler(mock_vnnlib)

        self.assertEqual(handler.spec_type, SpecType.SINGLE_AND_IN_MULTI_ORS)
        self.assertEqual(handler.post_processing_type, PostProcessingType.NONE)
        self.assertTrue(handler.optimize_disjuncts_separately)

    @patch('incomplete_verifier_func.arguments')
    def test_init_single_and_in_multi_ors_together(self, mock_arguments):
        """Test SpecHandler init with SINGLE_AND_IN_MULTI_ORS, optimize together."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': False,
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(3, 10)
        c = torch.randn(3, 1, 5)
        rhs = torch.zeros(3, 1)
        or_spec_size = torch.tensor([1, 1, 1])

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, True, True
        )

        handler = SpecHandler(mock_vnnlib)

        self.assertEqual(handler.spec_type, SpecType.SINGLE_AND_IN_MULTI_ORS)
        self.assertEqual(handler.post_processing_type, PostProcessingType.RESHAPE)
        # x should be reshaped to batch size 1
        self.assertEqual(handler.x.shape[0], 1)

    @patch('incomplete_verifier_func.arguments')
    def test_init_fixed_num_ands_multi_ors(self, mock_arguments):
        """Test SpecHandler init with FIXED_NUM_ANDS_IN_MULTI_ORS."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': True,
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(2, 10)  # 2 ORs
        c = torch.randn(2, 3, 5)  # Each OR has 3 ANDs
        rhs = torch.zeros(2, 3)
        or_spec_size = torch.tensor([3, 3])

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, True, True
        )

        handler = SpecHandler(mock_vnnlib)

        self.assertEqual(handler.spec_type, SpecType.FIXED_NUM_ANDS_IN_MULTI_ORS)

    @patch('incomplete_verifier_func.arguments')
    def test_init_variable_num_ands(self, mock_arguments):
        """Test SpecHandler init with VARIABLE_NUM_ANDS_IN_EACH_OR."""
        from incomplete_verifier_func import SpecHandler, SpecType, PostProcessingType

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': True,
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(2, 10)
        c = torch.randn(2, 3, 5)  # Padded to max AND size
        rhs = torch.zeros(2, 3)
        or_spec_size = torch.tensor([2, 3])  # Different AND sizes

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, True, False  # same_or_spec_size=False
        )

        handler = SpecHandler(mock_vnnlib)

        self.assertEqual(handler.spec_type, SpecType.VARIABLE_NUM_ANDS_IN_EACH_OR)

    @patch('incomplete_verifier_func.arguments')
    def test_init_different_x_range_forces_separate(self, mock_arguments):
        """Test SpecHandler init forces optimize_disjuncts_separately when x ranges differ."""
        from incomplete_verifier_func import SpecHandler

        mock_arguments.Config = {
            'solver': {
                'prune_after_crown': False,
                'bound_prop_method': 'crown',
                'optimize_disjuncts_separately': False,  # Will be overridden
                'invprop': {'apply_output_constraints_to': []},
            },
            'general': {
                'store_all_specs_on_cpu': False,
                'device': 'cpu',
            }
        }

        x = torch.randn(2, 10)
        c = torch.randn(2, 1, 5)
        rhs = torch.zeros(2, 1)
        or_spec_size = torch.tensor([1, 1])

        mock_vnnlib = self._create_mock_vnnlib_handler(
            x, c, rhs, or_spec_size, False, True  # same_x_range=False
        )

        with patch('builtins.print'):
            handler = SpecHandler(mock_vnnlib)

        # Should be forced to True
        self.assertTrue(handler.optimize_disjuncts_separately)


class TestSpecHandlerPruneVerifiedOrSpecs(unittest.TestCase):
    """Tests for SpecHandler.prune_verified_or_specs method."""

    def test_prune_verified_or_specs_updates_num_or(self):
        """Test prune_verified_or_specs updates num_or."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.num_or = 4
        handler.unverified_or_mask = torch.tensor([True, False, True, False])
        handler.unverified_or_indices = torch.tensor([0, 2])
        handler.or_spec_size = torch.tensor([1, 1, 1, 1])
        handler.optimize_disjuncts_separately = True
        handler.spec_type = SpecType.SINGLE_OR
        handler.vnnlib_handler = MagicMock()

        def mock_prune(data, dim, prune_and_size):
            return data.index_select(dim, handler.unverified_or_indices)

        handler._prune = mock_prune

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(4, 1),
            'global_ub': torch.randn(4, 1),
            'lower_bounds': {'output': torch.randn(4, 1)},
            'upper_bounds': {'output': torch.randn(4, 1)},
            'lA': None,
            'alphas': None,
            'mask': None,
            'input_split_idx': None,
        }

        with patch('builtins.print'):
            SpecHandler.prune_verified_or_specs(handler, ret)

        # num_or should be updated to count of True in unverified_or_mask
        self.assertEqual(handler.num_or, 2)

    def test_prune_verified_or_specs_prunes_bounds(self):
        """Test prune_verified_or_specs prunes bounds correctly."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.num_or = 3
        handler.unverified_or_mask = torch.tensor([False, True, True])  # First verified
        handler.unverified_or_indices = torch.tensor([1, 2])
        handler.or_spec_size = torch.tensor([1, 1, 1])
        handler.optimize_disjuncts_separately = True
        handler.spec_type = SpecType.SINGLE_OR
        handler.vnnlib_handler = MagicMock()

        def mock_prune(data, dim, prune_and_size):
            return data.index_select(dim, handler.unverified_or_indices)

        handler._prune = mock_prune

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'model': mock_model,
            'global_lb': torch.tensor([[1.0], [-1.0], [-2.0]]),
            'global_ub': torch.tensor([[2.0], [0.0], [-1.0]]),
            'lower_bounds': {'output': torch.tensor([[1.0], [-1.0], [-2.0]])},
            'upper_bounds': {'output': torch.tensor([[2.0], [0.0], [-1.0]])},
            'lA': None,
            'alphas': None,
            'mask': None,
            'input_split_idx': None,
        }

        with patch('builtins.print'):
            SpecHandler.prune_verified_or_specs(handler, ret)

        # Should only have 2 unverified ORs
        self.assertEqual(ret['global_lb'].shape[0], 2)

    def test_prune_verified_or_specs_prunes_alphas(self):
        """Test prune_verified_or_specs prunes alphas correctly."""
        from incomplete_verifier_func import SpecHandler, SpecType

        handler = MagicMock()
        handler.num_or = 2
        handler.unverified_or_mask = torch.tensor([False, True])
        handler.unverified_or_indices = torch.tensor([1])
        handler.or_spec_size = torch.tensor([1, 1])
        handler.optimize_disjuncts_separately = True
        handler.spec_type = SpecType.SINGLE_OR
        handler.vnnlib_handler = MagicMock()

        def mock_prune(data, dim, prune_and_size):
            return data.index_select(dim, handler.unverified_or_indices)

        handler._prune = mock_prune

        mock_model = MagicMock()
        mock_model.final_name = 'output'

        ret = {
            'model': mock_model,
            'global_lb': torch.randn(2, 1),
            'global_ub': torch.randn(2, 1),
            'lower_bounds': {'output': torch.randn(2, 1), 'layer1': torch.randn(2, 10)},
            'upper_bounds': {'output': torch.randn(2, 1), 'layer1': torch.randn(2, 10)},
            'lA': {'layer1': torch.randn(2, 1, 10)},
            'alphas': {
                'layer1': {
                    'alpha': {
                        'output': torch.randn(2, 1, 2, 5),
                        'layer1': torch.randn(2, 3, 2, 10),
                    }
                }
            },
            'mask': None,
            'input_split_idx': None,
        }

        with patch('builtins.print'):
            SpecHandler.prune_verified_or_specs(handler, ret)

        # Alphas should be pruned along dim 2
        self.assertEqual(ret['alphas']['layer1']['alpha']['output'].shape[2], 1)


class TestSetUnverifiedOrMaskEdgeCases(unittest.TestCase):
    """Edge case tests for set_unverified_or_mask."""

    def test_set_unverified_or_mask_empty_lb(self):
        """Test set_unverified_or_mask with empty lower bounds."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([]).reshape(0, 1)
        handler.or_spec_size = torch.tensor([])

        lb = torch.tensor([]).reshape(0, 1)

        SpecHandler.set_unverified_or_mask(handler, lb)

        self.assertEqual(handler.unverified_or_mask.shape[0], 0)

    def test_set_unverified_or_mask_single_element(self):
        """Test set_unverified_or_mask with single element."""
        from incomplete_verifier_func import SpecHandler
        from auto_LiRPA.utils import stop_criterion_batch_any

        handler = MagicMock()
        handler.stop_criterion = stop_criterion_batch_any
        handler.rhs = torch.tensor([[0.0]])
        handler.or_spec_size = torch.tensor([1])

        lb = torch.tensor([[0.5]])  # Verified

        SpecHandler.set_unverified_or_mask(handler, lb)

        self.assertFalse(handler.unverified_or_mask[0].item())
        self.assertEqual(handler.unverified_or_indices.shape[0], 0)


if __name__ == '__main__':
    unittest.main()
