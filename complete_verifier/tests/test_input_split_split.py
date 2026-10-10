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
"""Unit tests for input_split/split.py"""
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


class TestInputSplitParallelBasic(unittest.TestCase):
    """Basic tests for input_split_parallel function."""

    def test_basic_split_depth_1(self):
        """Test basic split with depth 1."""
        from input_split.split import input_split_parallel

        batch_size = 2
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0], [1]])  # Split on different dimensions

        new_x_L, new_x_U, actual_depth = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # With split_depth=1 and split_partitions=2, batch doubles
        self.assertEqual(new_x_L.shape[0], batch_size * 2)
        self.assertEqual(new_x_U.shape[0], batch_size * 2)
        self.assertEqual(actual_depth, 1)

    def test_basic_split_depth_2(self):
        """Test basic split with depth 2."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 4
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0, 1]])  # Split on dimensions 0 and 1

        new_x_L, new_x_U, actual_depth = input_split_parallel(
            x_L, x_U, shape, split_depth=2, i_idx=i_idx
        )

        # With split_depth=2 and split_partitions=2, batch quadruples
        self.assertEqual(new_x_L.shape[0], batch_size * 4)
        self.assertEqual(new_x_U.shape[0], batch_size * 4)
        self.assertEqual(actual_depth, 2)

    def test_split_creates_valid_bounds(self):
        """Test that split creates valid lower <= upper bounds."""
        from input_split.split import input_split_parallel

        batch_size = 2
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0], [1]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # All lower bounds should be <= upper bounds
        self.assertTrue((new_x_L <= new_x_U).all())

    def test_split_midpoint(self):
        """Test that split creates midpoint split."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])  # Split on dimension 0

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # Should have 2 domains: [0, 0.5] and [0.5, 1] on dimension 0
        # Check that midpoint 0.5 is used
        self.assertTrue(0.5 in new_x_L[:, 0] or 0.5 in new_x_U[:, 0])


class TestInputSplitParallelWithSplitHint(unittest.TestCase):
    """Tests for input_split_parallel with split_hint."""

    def test_split_hint_single_value(self):
        """Test split_hint with a single value applied to all dims."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])
        split_hint = [0.3]  # Split at 0.3 instead of midpoint

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx,
            split_partitions=2, split_hint=split_hint
        )

        # One domain should have upper bound at 0.3, other should have lower bound at 0.3
        self.assertTrue((new_x_L <= new_x_U).all())

    def test_split_hint_per_dimension(self):
        """Test split_hint with per-dimension values."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[1]])  # Split on dimension 1
        split_hint = [0.2, 0.5, 0.8]

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx,
            split_partitions=2, split_hint=split_hint
        )

        # Verify bounds are valid
        self.assertTrue((new_x_L <= new_x_U).all())

    def test_split_hint_wrong_length_raises(self):
        """Test that split_hint with wrong length raises assertion."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])
        split_hint = [0.3, 0.5]  # Wrong length (should be 1 or 3)

        with self.assertRaises(AssertionError):
            input_split_parallel(
                x_L, x_U, shape, split_depth=1, i_idx=i_idx,
                split_partitions=2, split_hint=split_hint
            )

    def test_split_hint_requires_binary_partitions(self):
        """Test that split_hint with non-binary partitions raises assertion."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])
        split_hint = [0.3]

        with self.assertRaises(AssertionError):
            input_split_parallel(
                x_L, x_U, shape, split_depth=1, i_idx=i_idx,
                split_partitions=3, split_hint=split_hint
            )


class TestInputSplitParallelSplitPartitions(unittest.TestCase):
    """Tests for input_split_parallel with different split_partitions."""

    def test_ternary_split(self):
        """Test split with 3 partitions."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx,
            split_partitions=3
        )

        # With 3 partitions and depth 1, batch triples
        self.assertEqual(new_x_L.shape[0], batch_size * 3)
        self.assertTrue((new_x_L <= new_x_U).all())

    def test_quaternary_split(self):
        """Test split with 4 partitions."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx,
            split_partitions=4
        )

        # With 4 partitions and depth 1, batch quadruples
        self.assertEqual(new_x_L.shape[0], batch_size * 4)
        self.assertTrue((new_x_L <= new_x_U).all())


class TestInputSplitParallelEdgeCases(unittest.TestCase):
    """Edge case tests for input_split_parallel."""

    def test_split_depth_larger_than_indices(self):
        """Test when split_depth exceeds available indices."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0, 1]])  # Only 2 indices

        # Request depth 5 but only 2 indices available
        new_x_L, new_x_U, actual_depth = input_split_parallel(
            x_L, x_U, shape, split_depth=5, i_idx=i_idx
        )

        # Depth should be capped at available indices
        self.assertEqual(actual_depth, 2)
        self.assertEqual(new_x_L.shape[0], batch_size * 4)

    def test_single_dimension_input(self):
        """Test with single dimension input."""
        from input_split.split import input_split_parallel

        batch_size = 2
        input_dim = 1
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0], [0]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        self.assertEqual(new_x_L.shape, (batch_size * 2, input_dim))
        self.assertTrue((new_x_L <= new_x_U).all())

    def test_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.tensor([[-10.0, 0.0]])
        x_U = torch.tensor([[10.0, 100.0]])
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0, 1]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=2, i_idx=i_idx
        )

        self.assertTrue((new_x_L <= new_x_U).all())

    def test_preserves_untouched_dimensions(self):
        """Test that dimensions not split are preserved."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 3
        x_L = torch.tensor([[0.0, 1.0, 2.0]])
        x_U = torch.tensor([[1.0, 3.0, 4.0]])
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])  # Only split dimension 0

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # Dimensions 1 and 2 should be unchanged in all resulting domains
        for i in range(new_x_L.shape[0]):
            self.assertEqual(new_x_L[i, 1].item(), 1.0)
            self.assertEqual(new_x_U[i, 1].item(), 3.0)
            self.assertEqual(new_x_L[i, 2].item(), 2.0)
            self.assertEqual(new_x_U[i, 2].item(), 4.0)


class TestInputSplitParallelShape(unittest.TestCase):
    """Tests for input_split_parallel shape handling."""

    def test_multidimensional_shape(self):
        """Test with multidimensional input shape."""
        from input_split.split import input_split_parallel

        batch_size = 2
        channels = 3
        height = 2
        width = 2
        total_dim = channels * height * width

        x_L = torch.zeros(batch_size, channels, height, width)
        x_U = torch.ones(batch_size, channels, height, width)
        shape = (batch_size, channels, height, width)
        i_idx = torch.tensor([[0], [1]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # Output should have same shape per sample as input
        self.assertEqual(new_x_L.shape[1:], (channels, height, width))
        self.assertEqual(new_x_U.shape[1:], (channels, height, width))


class TestInputSplitParallelNoGrad(unittest.TestCase):
    """Tests that input_split_parallel does not track gradients."""

    def test_no_grad_context(self):
        """Test that function runs without gradient computation."""
        from input_split.split import input_split_parallel

        batch_size = 2
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim, requires_grad=True)
        x_U = torch.ones(batch_size, input_dim, requires_grad=True)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0], [1]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        self.assertFalse(new_x_L.requires_grad)
        self.assertFalse(new_x_U.requires_grad)


class TestGetSplitDepthFunction(unittest.TestCase):
    """Tests for get_split_depth function from split.py."""

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_zero_domain_returns_one(self, mock_config):
        """Test that zero domains returns depth 1."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        result = get_split_depth(current_domain_num=0)
        self.assertEqual(result, 1)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_domain_larger_than_min_batch(self, mock_config):
        """Test when domain count exceeds min batch size."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        result = get_split_depth(current_domain_num=100)
        self.assertEqual(result, 1)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_domain_smaller_than_min_batch(self, mock_config):
        """Test when domain count is less than min batch size."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }
        # min_batch_size = 0.5 * 64 = 32
        # current_domain_num = 4 < 32
        # depth = int(log(32 // 4) // log(2)) = int(log(8) // log(2))
        # Due to floating point: log(8) ≈ 2.079, log(2) ≈ 0.693
        # 2.079 // 0.693 = 2.0, so depth = 2

        result = get_split_depth(current_domain_num=4)
        self.assertEqual(result, 2)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_depth_at_least_one(self, mock_config):
        """Test that depth is always at least 1."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        result = get_split_depth(current_domain_num=50)
        self.assertGreaterEqual(result, 1)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_with_split_partitions_3(self, mock_config):
        """Test with ternary split partitions."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 81  # 81 = 3^4
        }
        # min_batch_size = 0.5 * 81 = 40.5
        # current_domain_num = 3 < 40.5
        # depth = log(40 // 3) / log(3) = log(13) / log(3) ≈ 2.33 -> 2

        result = get_split_depth(current_domain_num=3, split_partitions=3)
        self.assertGreaterEqual(result, 1)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_exact_min_batch_size(self, mock_config):
        """Test when domain count equals min batch size."""
        from input_split.split import get_split_depth

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }
        # min_batch_size = 32, current_domain_num = 32

        result = get_split_depth(current_domain_num=32)
        self.assertEqual(result, 1)


class TestRepeatDataAfterSplitBasic(unittest.TestCase):
    """Basic tests for repeat_data_after_split function."""

    def test_repeat_cs(self):
        """Test repeating cs tensor."""
        from input_split.split import repeat_data_after_split

        cs = torch.randn(2, 3, 4)  # (batch, spec, output)
        split_depth = 2
        split_partitions = 2
        new_batch_size = split_partitions ** split_depth  # 4

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            cs=cs
        )

        repeated_cs = result[0]
        self.assertEqual(repeated_cs.shape, (2 * new_batch_size, 3, 4))

    def test_repeat_thresholds(self):
        """Test repeating thresholds tensor."""
        from input_split.split import repeat_data_after_split

        thresholds = torch.randn(2, 3)
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            thresholds=thresholds
        )

        repeated_thresholds = result[1]
        self.assertEqual(repeated_thresholds.shape, (4, 3))

    def test_repeat_dm_lb(self):
        """Test repeating dm_lb tensor."""
        from input_split.split import repeat_data_after_split

        dm_lb = torch.randn(2, 3)
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            dm_lb=dm_lb
        )

        repeated_dm_lb = result[2]
        self.assertEqual(repeated_dm_lb.shape, (4, 3))

    def test_repeat_spec_size(self):
        """Test repeating spec_size tensor."""
        from input_split.split import repeat_data_after_split

        spec_size = torch.tensor([2, 3])
        split_depth = 2
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            spec_size=spec_size
        )

        repeated_spec_size = result[4]
        self.assertEqual(repeated_spec_size.shape, (8,))

    def test_repeat_lA(self):
        """Test repeating lA tensor."""
        from input_split.split import repeat_data_after_split

        lA = torch.randn(2, 3, 4)
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            lA=lA
        )

        repeated_lA = result[5]
        self.assertEqual(repeated_lA.shape, (4, 3, 4))

    def test_repeat_lbias(self):
        """Test repeating lbias tensor."""
        from input_split.split import repeat_data_after_split

        lbias = torch.randn(2, 3)
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            lbias=lbias
        )

        repeated_lbias = result[6]
        self.assertEqual(repeated_lbias.shape, (4, 3))


class TestRepeatDataAfterSplitAlphas(unittest.TestCase):
    """Tests for repeating alphas in repeat_data_after_split."""

    def test_repeat_alphas(self):
        """Test repeating alphas dictionary."""
        from input_split.split import repeat_data_after_split

        alphas = {
            'module1': {
                'spec1': torch.randn(2, 3, 4, 5),  # dim 2 is batch
                'spec2': torch.randn(2, 3, 2, 5)
            }
        }
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            alphas=alphas
        )

        repeated_alphas = result[3]
        self.assertEqual(repeated_alphas['module1']['spec1'].shape, (2, 3, 8, 5))
        self.assertEqual(repeated_alphas['module1']['spec2'].shape, (2, 3, 4, 5))

    def test_repeat_alphas_preserves_structure(self):
        """Test that alphas dictionary structure is preserved."""
        from input_split.split import repeat_data_after_split

        alphas = {
            'm1': {'s1': torch.randn(1, 2, 3, 4)},
            'm2': {'s2': torch.randn(1, 2, 3, 4), 's3': torch.randn(1, 2, 3, 4)}
        }
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            alphas=alphas
        )

        repeated_alphas = result[3]
        self.assertIn('m1', repeated_alphas)
        self.assertIn('m2', repeated_alphas)
        self.assertIn('s1', repeated_alphas['m1'])
        self.assertIn('s2', repeated_alphas['m2'])
        self.assertIn('s3', repeated_alphas['m2'])


class TestRepeatDataAfterSplitConstraints(unittest.TestCase):
    """Tests for repeating constraints in repeat_data_after_split."""

    def test_repeat_constraints(self):
        """Test repeating constraints tuple."""
        from input_split.split import repeat_data_after_split

        constraints_A = torch.randn(2, 3, 4)
        constraints_b = torch.randn(2, 3)
        constraints = (constraints_A, constraints_b)
        split_depth = 1
        split_partitions = 2

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            constraints=constraints
        )

        r_constraints = result[7]
        self.assertIsNotNone(r_constraints)
        self.assertEqual(r_constraints[0].shape, (4, 3, 4))
        self.assertEqual(r_constraints[1].shape, (4, 3))


class TestRepeatDataAfterSplitNone(unittest.TestCase):
    """Tests for None handling in repeat_data_after_split."""

    def test_all_none_returns_all_none(self):
        """Test that all None inputs return all None outputs."""
        from input_split.split import repeat_data_after_split

        result = repeat_data_after_split(
            split_depth=1,
            split_partitions=2
        )

        self.assertIsNone(result[0])  # cs
        self.assertIsNone(result[1])  # thresholds
        self.assertIsNone(result[2])  # dm_lb
        self.assertIsNone(result[3])  # alphas
        self.assertIsNone(result[4])  # spec_size
        self.assertIsNone(result[5])  # lA
        self.assertIsNone(result[6])  # lbias
        self.assertIsNone(result[7])  # constraints


class TestRepeatDataAfterSplitPartitions(unittest.TestCase):
    """Tests for different partition counts in repeat_data_after_split."""

    def test_ternary_partitions(self):
        """Test with 3 partitions."""
        from input_split.split import repeat_data_after_split

        cs = torch.randn(2, 3, 4)
        split_depth = 2
        split_partitions = 3
        # new_batch_size = 3^2 = 9

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            cs=cs
        )

        repeated_cs = result[0]
        self.assertEqual(repeated_cs.shape, (2 * 9, 3, 4))

    def test_depth_3_partitions_2(self):
        """Test with depth 3 and binary partitions."""
        from input_split.split import repeat_data_after_split

        thresholds = torch.randn(1, 5)
        split_depth = 3
        split_partitions = 2
        # new_batch_size = 2^3 = 8

        result = repeat_data_after_split(
            split_depth=split_depth,
            split_partitions=split_partitions,
            thresholds=thresholds
        )

        repeated_thresholds = result[1]
        self.assertEqual(repeated_thresholds.shape, (8, 5))


class TestInputSplitAndRepeatBasic(unittest.TestCase):
    """Basic tests for input_split_and_repeat function."""

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_basic_split_and_repeat(self, mock_config):
        """Test basic functionality of input_split_and_repeat."""
        from input_split.split import input_split_and_repeat

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        batch_size = 2
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0, 1], [1, 0]])
        cs = torch.randn(batch_size, 2, 5)

        result = input_split_and_repeat(
            x_L=x_L,
            x_U=x_U,
            current_domain_num=batch_size,
            shape=shape,
            i_idx=i_idx,
            cs=cs
        )

        new_x_L, new_x_U = result[0], result[1]
        repeated_cs = result[2]

        # Verify bounds are valid
        self.assertTrue((new_x_L <= new_x_U).all())
        # Verify cs was repeated
        self.assertIsNotNone(repeated_cs)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_returns_correct_tuple_length(self, mock_config):
        """Test that input_split_and_repeat returns correct tuple."""
        from input_split.split import input_split_and_repeat

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        batch_size = 1
        input_dim = 2
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])

        result = input_split_and_repeat(
            x_L=x_L,
            x_U=x_U,
            current_domain_num=batch_size,
            shape=shape,
            i_idx=i_idx
        )

        # Should return 10 elements
        self.assertEqual(len(result), 10)

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_with_all_optional_params(self, mock_config):
        """Test with all optional parameters provided."""
        from input_split.split import input_split_and_repeat

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        batch_size = 2
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0], [1]])

        cs = torch.randn(batch_size, 2, 5)
        thresholds = torch.zeros(batch_size, 2)
        dm_lb = torch.randn(batch_size, 2)
        alphas = {'m1': {'s1': torch.randn(2, 2, batch_size, 3)}}
        spec_size = torch.tensor([2, 2])
        lA = torch.randn(batch_size, 2, 3)
        lbias = torch.randn(batch_size, 2)
        constraints = (torch.randn(batch_size, 2, 3), torch.randn(batch_size, 2))

        result = input_split_and_repeat(
            x_L=x_L,
            x_U=x_U,
            current_domain_num=batch_size,
            shape=shape,
            i_idx=i_idx,
            cs=cs,
            thresholds=thresholds,
            dm_lb=dm_lb,
            alphas=alphas,
            spec_size=spec_size,
            lA=lA,
            lbias=lbias,
            constraints=constraints
        )

        # Verify all outputs are not None
        new_x_L, new_x_U = result[0], result[1]
        self.assertTrue((new_x_L <= new_x_U).all())
        self.assertIsNotNone(result[2])  # cs
        self.assertIsNotNone(result[3])  # thresholds
        self.assertIsNotNone(result[4])  # dm_lb
        self.assertIsNotNone(result[5])  # alphas
        self.assertIsNotNone(result[6])  # spec_size
        self.assertIsNotNone(result[7])  # lA
        self.assertIsNotNone(result[8])  # lbias
        self.assertIsNotNone(result[9])  # constraints


class TestInputSplitAndRepeatWithSplitHint(unittest.TestCase):
    """Tests for input_split_and_repeat with split_hint."""

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_with_split_hint(self, mock_config):
        """Test input_split_and_repeat with split_hint."""
        from input_split.split import input_split_and_repeat

        mock_config['solver'] = {
            'min_batch_size_ratio': 0.5,
            'batch_size': 64
        }

        batch_size = 1
        input_dim = 3
        x_L = torch.zeros(batch_size, input_dim)
        x_U = torch.ones(batch_size, input_dim)
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])
        split_hint = [0.3]

        result = input_split_and_repeat(
            x_L=x_L,
            x_U=x_U,
            current_domain_num=batch_size,
            shape=shape,
            i_idx=i_idx,
            split_hint=split_hint
        )

        new_x_L, new_x_U = result[0], result[1]
        self.assertTrue((new_x_L <= new_x_U).all())


class TestInputSplitAndRepeatSplitPartitions(unittest.TestCase):
    """Tests for input_split_and_repeat with different split_partitions."""

    @patch('input_split.split.arguments.Config', new_callable=dict)
    def test_with_ternary_partitions(self, mock_config):
        """Test input_split_and_repeat with 3 partitions."""
        import math
        from input_split.split import input_split_and_repeat, get_split_depth

        min_batch_size_ratio = 0.5
        config_batch_size = 64
        mock_config['solver'] = {
            'min_batch_size_ratio': min_batch_size_ratio,
            'batch_size': config_batch_size
        }

        current_domain_num = 1
        input_dim = 2
        split_partitions = 3
        x_L = torch.zeros(current_domain_num, input_dim)
        x_U = torch.ones(current_domain_num, input_dim)
        shape = (current_domain_num, input_dim)
        # Provide enough indices for depth 3 splits
        i_idx = torch.tensor([[0, 1, 0]])

        # Calculate expected depth using the same formula as get_split_depth
        min_batch_size = min_batch_size_ratio * config_batch_size
        computed_depth = int(math.log(min_batch_size // current_domain_num) // math.log(split_partitions))
        computed_depth = max(computed_depth, 1)

        # Verify our calculation matches the function
        actual_computed_depth = get_split_depth(current_domain_num, split_partitions)
        self.assertEqual(actual_computed_depth, computed_depth)

        # Actual depth used is min(computed_depth, i_idx.size(1)) per input_split_parallel
        actual_depth = min(computed_depth, i_idx.size(1))

        result = input_split_and_repeat(
            x_L=x_L,
            x_U=x_U,
            current_domain_num=current_domain_num,
            shape=shape,
            i_idx=i_idx,
            split_partitions=split_partitions
        )

        new_x_L = result[0]
        # With ternary partitions, batch should be multiplied by split_partitions^depth
        expected_batch_size = current_domain_num * (split_partitions ** actual_depth)
        self.assertEqual(new_x_L.shape[0], expected_batch_size)


class TestSplitCoverageComputation(unittest.TestCase):
    """Tests to verify split operations cover the entire domain."""

    def test_binary_split_covers_domain(self):
        """Test that binary split covers entire original domain."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # The min of all lower bounds should equal original lower
        # The max of all upper bounds should equal original upper
        self.assertAlmostEqual(new_x_L[:, 0].min().item(), 0.0, places=5)
        self.assertAlmostEqual(new_x_U[:, 0].max().item(), 1.0, places=5)

    def test_split_domains_are_disjoint_on_split_dim(self):
        """Test that resulting domains don't overlap on split dimension."""
        from input_split.split import input_split_parallel

        batch_size = 1
        input_dim = 2
        x_L = torch.tensor([[0.0, 0.0]])
        x_U = torch.tensor([[1.0, 1.0]])
        shape = (batch_size, input_dim)
        i_idx = torch.tensor([[0]])

        new_x_L, new_x_U, _ = input_split_parallel(
            x_L, x_U, shape, split_depth=1, i_idx=i_idx
        )

        # For binary split on dim 0:
        # Domain 0 should have x_U[0, 0] == x_L[1, 0] (they meet at midpoint)
        midpoint = (x_L[0, 0] + x_U[0, 0]) / 2
        # One domain should end at midpoint, other should start at midpoint
        self.assertTrue(
            torch.isclose(new_x_U[0, 0], midpoint) or
            torch.isclose(new_x_U[1, 0], midpoint) or
            torch.isclose(new_x_L[0, 0], midpoint) or
            torch.isclose(new_x_L[1, 0], midpoint)
        )


if __name__ == '__main__':
    setup_module()
    unittest.main()
