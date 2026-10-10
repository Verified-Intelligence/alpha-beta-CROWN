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
"""Unit tests for specifications.py"""
import os
import sys
import unittest
import pytest
from types import SimpleNamespace

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specifications import (
    shrink_vnnlib, add_rhs_offset_legacy, BatchedSpecs,
    Specification, SpecificationVerifiedAcc, SpecificationTarget,
    SpecificationRunnerup, SpecificationAllPositive, construct_vnnlib,
    VNNLibEntries, vnnlibHandler
)


@pytest.fixture(autouse=True)
def setup_arguments():
    """Setup arguments.Config for testing."""
    import arguments
    original_config = arguments.Config
    try:
        new_config = arguments.ConfigHandler()
        new_config.construct_config_dict(new_config.default_args)
        new_config.file = None
        # Set required config values
        new_config['data']['num_outputs'] = 10
        new_config['bab']['decision_thresh'] = 0.0
        new_config['specification']['type'] = 'lp'
        new_config['specification']['norm'] = float('inf')
        new_config['specification']['robustness_type'] = 'verified-acc'
        new_config['specification']['shrink_eps'] = None
        new_config['general']['store_all_specs_on_cpu'] = False
        new_config['general']['device'] = 'cpu'
        arguments.Config = new_config
        yield
    finally:
        arguments.Config = original_config


class TestShrinkVnnlib(unittest.TestCase):
    """Tests for shrink_vnnlib function."""

    def test_shrink_basic(self):
        """Test basic shrinking of vnnlib input ranges."""
        vnnlib = [
            [[(0.0, 1.0), (0.0, 1.0)], []]
        ]
        shrink_eps = 0.1
        result = shrink_vnnlib(vnnlib, shrink_eps)
        # Lower bound should increase, upper bound should decrease
        self.assertAlmostEqual(result[0][0][0][0], 0.1)
        self.assertAlmostEqual(result[0][0][0][1], 0.9)
        self.assertAlmostEqual(result[0][0][1][0], 0.1)
        self.assertAlmostEqual(result[0][0][1][1], 0.9)

    def test_shrink_multiple_specs(self):
        """Test shrinking with multiple specifications."""
        vnnlib = [
            [[(0.0, 2.0)], []],
            [[(-1.0, 1.0)], []]
        ]
        shrink_eps = 0.5
        result = shrink_vnnlib(vnnlib, shrink_eps)
        self.assertAlmostEqual(result[0][0][0][0], 0.5)
        self.assertAlmostEqual(result[0][0][0][1], 1.5)
        self.assertAlmostEqual(result[1][0][0][0], -0.5)
        self.assertAlmostEqual(result[1][0][0][1], 0.5)

    def test_shrink_requires_positive_eps(self):
        """Test the positive epsilon precondition."""
        vnnlib = [[[(0.0, 1.0)], []]]
        for shrink_eps in (0.0, -0.1):
            with self.subTest(shrink_eps=shrink_eps):
                with self.assertRaises(AssertionError):
                    shrink_vnnlib(vnnlib, shrink_eps)


class TestAddRhsOffsetLegacy(unittest.TestCase):
    """Tests for add_rhs_offset_legacy function."""

    def test_none_offset_returns_original(self):
        """Test that None offset returns original vnnlib."""
        vnnlib = [
            (np.array([[0, 1]]), [(np.array([[1, 0]]), np.array([0.5]))])
        ]
        result = add_rhs_offset_legacy(vnnlib, None)
        self.assertIs(result, vnnlib)

    def test_scalar_float_offset(self):
        """Test scalar float offset."""
        c = np.array([[1, 0]])
        rhs = np.array([0.5])
        vnnlib = [
            (np.array([[0, 1]]), [(c, rhs)])
        ]
        result = add_rhs_offset_legacy(vnnlib, 0.1)
        self.assertAlmostEqual(result[0][1][0][1][0], 0.6, places=5)

    def test_tensor_scalar_offset(self):
        """Test tensor scalar offset."""
        c = np.array([[1, 0]])
        rhs = np.array([0.5])
        vnnlib = [
            (np.array([[0, 1]]), [(c, rhs)])
        ]
        offset = torch.tensor([0.1])
        result = add_rhs_offset_legacy(vnnlib, offset)
        # Should add offset + delta (1e-3)
        expected = 0.5 + 0.1 + 1e-3
        self.assertAlmostEqual(result[0][1][0][1][0], expected, places=5)

    def test_multiple_clauses(self):
        """Test with multiple clauses."""
        c1 = np.array([[1, 0]])
        c2 = np.array([[0, 1]])
        rhs1 = np.array([0.5])
        rhs2 = np.array([0.3])
        vnnlib = [
            (np.array([[0, 1]]), [(c1, rhs1), (c2, rhs2)])
        ]
        result = add_rhs_offset_legacy(vnnlib, 0.2)
        self.assertAlmostEqual(result[0][1][0][1][0], 0.7, places=5)
        self.assertAlmostEqual(result[0][1][1][1][0], 0.5, places=5)


class TestBatchedSpecs(unittest.TestCase):
    """Tests for BatchedSpecs dataclass."""

    def _create_mock_bounded_tensor(self, shape):
        """Create a mock BoundedTensor-like object for testing."""
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        data = torch.randn(shape)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        return BoundedTensor(data, ptb)

    def test_init(self):
        """Test BatchedSpecs initialization."""
        x = self._create_mock_bounded_tensor((4, 3, 32, 32))
        c = torch.randn(4, 2, 10)
        rhs = torch.randn(4, 2)
        or_spec_size = torch.tensor([2, 2, 2, 2])

        specs = BatchedSpecs(
            batch_size=4,
            x=x,
            c=c,
            rhs=rhs,
            or_spec_size=or_spec_size,
            same_x_range=True,
            same_or_spec_size=True
        )

        self.assertEqual(specs.batch_size, 4)
        self.assertTrue(specs.same_x_range)
        self.assertTrue(specs.same_or_spec_size)

    def test_get_moves_to_device(self):
        """Test that get() moves tensors to specified device."""
        x = self._create_mock_bounded_tensor((2, 3, 8, 8))
        c = torch.randn(2, 1, 5)
        rhs = torch.randn(2, 1)
        or_spec_size = torch.tensor([1, 1])

        specs = BatchedSpecs(
            batch_size=2,
            x=x,
            c=c,
            rhs=rhs,
            or_spec_size=or_spec_size,
            same_x_range=False,
            same_or_spec_size=True
        )

        result = specs.get(device='cpu')
        self.assertEqual(len(result), 6)
        x_out, c_out, rhs_out, or_size_out, same_x, same_or = result
        self.assertEqual(c_out.device.type, 'cpu')
        self.assertEqual(rhs_out.device.type, 'cpu')

    def test_get_without_single_x_range(self):
        """Test get() without single_x_range returns full batch."""
        x = self._create_mock_bounded_tensor((4, 3, 8, 8))
        c = torch.randn(4, 1, 5)
        rhs = torch.randn(4, 1)
        or_spec_size = torch.tensor([1, 1, 1, 1])

        specs = BatchedSpecs(
            batch_size=4,
            x=x,
            c=c,
            rhs=rhs,
            or_spec_size=or_spec_size,
            same_x_range=True,
            same_or_spec_size=True
        )

        result = specs.get(device='cpu', single_x_range=False)
        x_out = result[0]
        # Without single_x_range, x should keep original batch size
        self.assertEqual(x_out.shape[0], 4)


class TestBatchedSpecsAttributes(unittest.TestCase):
    """Additional tests for BatchedSpecs attributes."""

    def test_batch_size_attribute(self):
        """Test batch_size attribute."""
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        data = torch.randn(8, 3, 16, 16)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        specs = BatchedSpecs(
            batch_size=8,
            x=x,
            c=torch.randn(8, 3, 10),
            rhs=torch.randn(8, 3),
            or_spec_size=torch.tensor([3] * 8),
            same_x_range=False,
            same_or_spec_size=True
        )

        self.assertEqual(specs.batch_size, 8)

    def test_same_or_spec_size_false(self):
        """Test with same_or_spec_size=False."""
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        data = torch.randn(3, 1, 4)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        specs = BatchedSpecs(
            batch_size=3,
            x=x,
            c=torch.randn(3, 2, 5),  # padded
            rhs=torch.randn(3, 2),
            or_spec_size=torch.tensor([1, 2, 1]),  # different sizes
            same_x_range=True,
            same_or_spec_size=False
        )

        self.assertFalse(specs.same_or_spec_size)


# ============================================================================
# Specification Base Class Tests (pytest)
# ============================================================================

class TestSpecificationBase:
    """Tests for Specification base class."""

    def test_specification_init(self):
        """Test Specification initialization."""
        import arguments

        spec = Specification()
        assert spec.num_outputs == arguments.Config['data']['num_outputs']
        assert isinstance(spec.rhs, np.ndarray)
        assert spec.rhs[0] == arguments.Config['bab']['decision_thresh']

# ============================================================================
# SpecificationVerifiedAcc Tests (pytest)
# ============================================================================

class TestSpecificationVerifiedAccPytest:
    """Tests for SpecificationVerifiedAcc class."""

    def test_construct_vnnlib_single_example(self):
        """Test constructing vnnlib for single example."""
        import arguments

        arguments.Config['data']['num_outputs'] = 3

        spec = SpecificationVerifiedAcc()
        dataset = {
            'labels': torch.tensor([0, 1, 2])
        }
        x_range = [np.array([[0.0, 1.0], [0.0, 1.0]])]
        example_idx_list = [0]

        result = spec.construct_vnnlib(dataset, x_range, example_idx_list)

        assert len(result) == 1
        assert len(result[0]) == 1
        # Should have num_outputs - 1 specifications
        assert len(result[0][0][1]) == 2  # 3 outputs - 1

    def test_construct_vnnlib_multiple_examples(self):
        """Test constructing vnnlib for multiple examples."""
        import arguments

        arguments.Config['data']['num_outputs'] = 4

        spec = SpecificationVerifiedAcc()
        dataset = {
            'labels': torch.tensor([0, 1, 2, 3])
        }
        x_range = [
            np.array([[0.0, 1.0]]),
            np.array([[0.0, 1.0]]),
        ]
        example_idx_list = [0, 1]

        result = spec.construct_vnnlib(dataset, x_range, example_idx_list)

        assert len(result) == 2
        for vnn in result:
            assert len(vnn[0][1]) == 3  # 4 outputs - 1


# ============================================================================
# SpecificationTarget Tests (pytest)
# ============================================================================

class TestSpecificationTargetPytest:
    """Tests for SpecificationTarget class."""

    def test_construct_vnnlib(self):
        """Test constructing vnnlib for targeted attack."""
        import arguments

        arguments.Config['data']['num_outputs'] = 5

        spec = SpecificationTarget()
        dataset = {
            'labels': torch.tensor([0, 1]),
            'target_label': torch.tensor([1, 0])
        }
        x_range = [
            np.array([[0.0, 1.0]]),
            np.array([[0.0, 1.0]]),
        ]
        example_idx_list = [0, 1]

        result = spec.construct_vnnlib(dataset, x_range, example_idx_list)

        assert len(result) == 2
        # Each should have exactly 1 constraint
        assert len(result[0][0][1]) == 1
        assert len(result[1][0][1]) == 1


# ============================================================================
# SpecificationRunnerup Tests (pytest)
# ============================================================================

class TestSpecificationRunnerupPytest:
    """Tests for SpecificationRunnerup class."""

    def test_construct_vnnlib(self):
        """Test constructing vnnlib for runnerup verification."""
        import arguments

        arguments.Config['data']['num_outputs'] = 3

        spec = SpecificationRunnerup()
        dataset = {
            'labels': torch.tensor([0, 1]),
            'runnerup': torch.tensor([1, 2])
        }
        x_range = [
            np.array([[0.0, 1.0]]),
            np.array([[0.0, 1.0]]),
        ]
        example_idx_list = [0, 1]

        result = spec.construct_vnnlib(dataset, x_range, example_idx_list)

        assert len(result) == 2
        assert len(result[0][0][1]) == 1


# ============================================================================
# SpecificationAllPositive Tests (pytest)
# ============================================================================

class TestSpecificationAllPositivePytest:
    """Tests for SpecificationAllPositive class."""

    def test_construct_vnnlib(self):
        """Test constructing vnnlib for all-positive specification."""
        import arguments

        arguments.Config['data']['num_outputs'] = 4

        spec = SpecificationAllPositive()
        dataset = {}
        x_range = [np.array([[0.0, 1.0]])]
        example_idx_list = [0]

        result = spec.construct_vnnlib(dataset, x_range, example_idx_list)

        assert len(result) == 1
        assert len(result[0][0][1]) == 4


# ============================================================================
# construct_vnnlib Function Tests (pytest)
# ============================================================================

class TestConstructVnnlibPytest:
    """Tests for construct_vnnlib function."""

    def test_lp_linf_spec(self):
        """Test construct_vnnlib with Linf Lp specification."""
        import arguments

        arguments.Config['specification']['type'] = 'lp'
        arguments.Config['specification']['norm'] = float('inf')
        arguments.Config['specification']['robustness_type'] = 'verified-acc'
        arguments.Config['data']['num_outputs'] = 3

        dataset = {
            'X': torch.randn(2, 1, 4, 4),
            'labels': torch.tensor([0, 1]),
            'eps': 0.1,
        }
        example_idx_list = [0]

        result = construct_vnnlib(dataset, example_idx_list)

        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_lp_linf_spec_with_data_bounds(self):
        """Test construct_vnnlib with Linf and data bounds."""
        import arguments

        arguments.Config['specification']['type'] = 'lp'
        arguments.Config['specification']['norm'] = float('inf')
        arguments.Config['specification']['robustness_type'] = 'verified-acc'
        arguments.Config['data']['num_outputs'] = 3

        dataset = {
            'X': torch.randn(2, 1, 4, 4),
            'labels': torch.tensor([0, 1]),
            'eps': 0.1,
            'data_min': 0.0,
            'data_max': 1.0,
        }
        example_idx_list = [0]

        result = construct_vnnlib(dataset, example_idx_list)

        assert len(result) == 1

    def test_lp_l2_spec(self):
        """Test construct_vnnlib with L2 Lp specification."""
        import arguments

        arguments.Config['specification']['type'] = 'lp'
        arguments.Config['specification']['norm'] = 2
        arguments.Config['specification']['robustness_type'] = 'verified-acc'
        arguments.Config['data']['num_outputs'] = 3

        dataset = {
            'X': torch.randn(2, 1, 4, 4),
            'labels': torch.tensor([0, 1]),
            'eps': 0.5,
            'norm': 2,
        }
        example_idx_list = [0]

        result = construct_vnnlib(dataset, example_idx_list)

        assert len(result) == 1
        # For L2, x_range is a dict
        assert isinstance(result[0][0][0], dict)

    def test_lp_with_eps_list(self):
        """Test construct_vnnlib with list of eps values (L2 norm)."""
        import arguments

        # Use L2 norm since Linf with list eps has shape mismatch issues
        arguments.Config['specification']['type'] = 'lp'
        arguments.Config['specification']['norm'] = 2
        arguments.Config['specification']['robustness_type'] = 'verified-acc'
        arguments.Config['data']['num_outputs'] = 3

        dataset = {
            'X': torch.randn(3, 1, 4, 4),
            'labels': torch.tensor([0, 1, 2]),
            'eps': [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.15])],
            'norm': 2,
        }
        example_idx_list = [0, 1]

        result = construct_vnnlib(dataset, example_idx_list)

        assert len(result) == 2

    def test_box_spec(self):
        """Test construct_vnnlib with box specification."""
        import arguments

        arguments.Config['specification']['type'] = 'box'
        arguments.Config['specification']['robustness_type'] = 'all-positive'
        arguments.Config['data']['num_outputs'] = 2

        dataset = {
            'X': torch.randn(2, 4),
            'data_min': torch.zeros(2, 4),
            'data_max': torch.ones(2, 4),
        }
        example_idx_list = [0]

        result = construct_vnnlib(dataset, example_idx_list)

        assert len(result) == 1

    def test_unsupported_spec_type(self):
        """Test construct_vnnlib with unsupported specification type."""
        import arguments

        arguments.Config['specification']['type'] = 'unsupported'

        dataset = {
            'X': torch.randn(2, 4),
        }

        with pytest.raises(ValueError, match='Unsupported perturbation type'):
            construct_vnnlib(dataset, [0])

    def test_unsupported_robustness_type(self):
        """Test construct_vnnlib with unsupported robustness type."""
        import arguments

        arguments.Config['specification']['type'] = 'lp'
        arguments.Config['specification']['norm'] = float('inf')
        arguments.Config['specification']['robustness_type'] = 'unknown-type'
        arguments.Config['data']['num_outputs'] = 3

        dataset = {
            'X': torch.randn(2, 4),
            'labels': torch.tensor([0, 1]),
            'eps': 0.1,
        }

        with pytest.raises(ValueError):
            construct_vnnlib(dataset, [0])


# ============================================================================
# vnnlibHandler Tests (pytest)
# ============================================================================

class TestVnnlibHandlerPytest:
    """Tests for vnnlibHandler class."""

    def test_init_linf_norm(self):
        """Test initialization with Linf norm vnnlib."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        assert handler.is_linf_norm is True
        assert handler.total_num_or == 1
        assert handler.num_output == 3

    def test_init_dict_format(self):
        """Test initialization with dict format (general Lp norm)."""
        import arguments

        arguments.Config['specification']['norm'] = 2

        x = torch.randn(1, 4)
        x_item = {
            'X': x,
            'data_min': x - 0.1,
            'data_max': x + 0.1,
            'eps': torch.tensor(0.1),
        }
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_item, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        assert handler.is_linf_norm is False
        assert handler.total_num_or == 1

    def test_pop_single_batch(self):
        """Test _pop method for single batch."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)
        result = handler._pop(1)

        assert result is not None
        assert result.batch_size == 1

    def test_pop_exhausted(self):
        """Test _pop returns None when exhausted."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)
        handler._pop(1)
        result = handler._pop(1)

        assert result is None

    def test_add_rhs_offset_float(self):
        """Test add_rhs_offset with float."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)
        original_rhs = handler.rhs.clone()
        handler.add_rhs_offset(0.5)

        assert not torch.equal(handler.rhs, original_rhs)

    def test_add_rhs_offset_none(self):
        """Test add_rhs_offset with None."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)
        original_rhs = handler.rhs.clone()
        handler.add_rhs_offset(None)

        assert torch.equal(handler.rhs, original_rhs)

    def test_prune_verified_or_specs(self):
        """Test prune_verified_or_specs method."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range1 = [[0.0, 1.0]] * 4
        x_range2 = [[0.0, 0.5]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [
            (x_range1, [(c1, rhs1)]),
            (x_range2, [(c1, rhs1)]),
        ]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)
        assert handler.total_num_or == 2

        unverified_mask = torch.tensor([False, True])
        handler.prune_verified_or_specs(unverified_mask)

        assert handler.total_num_or == 1

    def test_update_input_bounds(self):
        """Test update_input_bounds method."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        rhs1 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        new_min = torch.zeros(1, 4)
        new_max = torch.ones(1, 4) * 0.5
        handler.update_input_bounds(new_min, new_max)

        assert torch.equal(handler.data_min, new_min)
        assert torch.equal(handler.data_max, new_max)

    def test_multiple_or_specs(self):
        """Test handler with multiple OR specs per input."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0]])
        c2 = np.array([[1.0, 0.0, -1.0]])
        rhs1 = np.array([0.0])
        rhs2 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1), (c2, rhs2)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        assert handler.total_num_or == 2

    def test_different_and_sizes(self):
        """Test handler with different AND sizes in OR specs."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = [[0.0, 1.0]] * 4
        c1 = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])  # 2 ANDs
        c2 = np.array([[1.0, -1.0, 0.0]])  # 1 AND
        rhs1 = np.array([0.0, 0.0])
        rhs2 = np.array([0.0])
        vnnlib = [(x_range, [(c1, rhs1), (c2, rhs2)])]
        vnnlib_shape = [-1, 4]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        assert handler.total_num_or == 2
        assert handler.or_spec_size[0].item() == 2
        assert handler.or_spec_size[1].item() == 1

    def test_open_input_bounds_shift_without_changing_rhs(self):
        """Test input strictness consumed by the closed verifier box format."""
        import arguments

        arguments.Config['specification']['norm'] = float('inf')

        x_range = np.array([[0.0, 1.0], [2.0, 3.0]])
        c = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        branch_strictness = SimpleNamespace(
            input_lower=np.array([True, False]),
            input_upper=np.array([False, True]),
            specs=[np.array([True, False])],
        )
        metadata = SimpleNamespace(strictness=[branch_strictness])
        vnnlib = VNNLibEntries([(x_range, [(c, rhs)])], metadata)
        vnnlib_shape = [-1, 2]

        handler = vnnlibHandler(vnnlib, vnnlib_shape)

        np_dtype = (
            np.float64 if torch.get_default_dtype() == torch.float64
            else np.float32
        )
        expected_data_min = torch.tensor(
            [[np.nextafter(np_dtype(0.0), np_dtype(np.inf)), 2.0]])
        expected_data_max = torch.tensor(
            [[1.0, np.nextafter(np_dtype(3.0), np_dtype(-np.inf))]])
        expected_rhs = torch.tensor([0.0, 1.0])

        torch.testing.assert_close(handler.data_min, expected_data_min)
        torch.testing.assert_close(handler.data_max, expected_data_max)
        torch.testing.assert_close(handler.rhs, expected_rhs)


# ============================================================================
# Additional BatchedSpecs Tests (pytest)
# ============================================================================

class TestBatchedSpecsPytest:
    """Additional pytest-style tests for BatchedSpecs."""

    def test_batched_specs_print_stats(self, capsys):
        """Test BatchedSpecs print_stats method."""
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        x = torch.randn(2, 4)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=x - 0.1, x_U=x + 0.1)
        bounded_x = BoundedTensor(x, ptb)

        specs = BatchedSpecs(
            batch_size=2,
            x=bounded_x,
            c=torch.randn(2, 1, 3),
            rhs=torch.zeros(2, 1),
            or_spec_size=torch.tensor([1, 1]),
            same_x_range=True,
            same_or_spec_size=True,
        )

        specs.print_stats()
        captured = capsys.readouterr()

        assert "Batch size: 2" in captured.out
        assert "First 10 spec matrices" in captured.out


if __name__ == '__main__':
    unittest.main()
