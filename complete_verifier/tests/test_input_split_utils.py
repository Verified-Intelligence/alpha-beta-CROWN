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
"""Unit tests for input_split/utils.py"""
import os
import sys
import unittest
from unittest.mock import patch, call

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_split.utils import initial_verify_criterion, Timer, Stats


class TestInitialVerifyCriterion(unittest.TestCase):
    """Tests for initial_verify_criterion function."""

    def test_all_verified(self):
        """Test when all samples are verified."""
        # lbs > rhs means verified
        lbs = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        rhs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertTrue(verified)
        self.assertEqual(len(unverified_idx), 0)

    def test_none_verified(self):
        """Test when no samples are verified."""
        # lbs <= rhs means not verified
        lbs = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
        rhs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertFalse(verified)
        self.assertEqual(len(unverified_idx), 2)
        self.assertTrue(0 in unverified_idx)
        self.assertTrue(1 in unverified_idx)

    def test_partial_verified(self):
        """Test when some samples are verified."""
        # First sample verified (has one lb > rhs), second not verified
        lbs = torch.tensor([[1.0, -1.0], [-0.5, -1.5]])
        rhs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertFalse(verified)  # Not all verified
        self.assertEqual(len(unverified_idx), 1)
        self.assertEqual(unverified_idx[0].item(), 1)

    def test_single_sample_verified(self):
        """Test single sample that is verified."""
        lbs = torch.tensor([[0.1, 0.2, 0.3]])
        rhs = torch.tensor([[0.0, 0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertTrue(verified)

    def test_single_sample_not_verified(self):
        """Test single sample that is not verified."""
        lbs = torch.tensor([[-0.1, -0.2, -0.3]])
        rhs = torch.tensor([[0.0, 0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertFalse(verified)
        self.assertEqual(len(unverified_idx), 1)

    def test_boundary_case(self):
        """Test boundary case where lbs == rhs."""
        lbs = torch.tensor([[0.0, 0.0]])
        rhs = torch.tensor([[0.0, 0.0]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        # lbs - rhs = 0, not > 0, so not verified
        self.assertFalse(verified)

    def test_with_nonzero_rhs(self):
        """Test with non-zero rhs values."""
        lbs = torch.tensor([[1.0, 2.0]])
        rhs = torch.tensor([[0.5, 0.5]])
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertTrue(verified)


class TestInputSplitTimer(unittest.TestCase):
    """Tests for input_split Timer class."""

    def test_default_values(self):
        """Test Timer default field values."""
        timer = Timer()
        self.assertEqual(timer.total_func_time, 0.0)
        self.assertEqual(timer.total_prepare_time, 0.0)
        self.assertEqual(timer.total_bound_time, 0.0)
        self.assertEqual(timer.total_beta_bound_time, 0.0)
        self.assertEqual(timer.total_transfer_time, 0.0)
        self.assertEqual(timer.total_finalize_time, 0.0)

    def test_start_add_cycle(self):
        """Test start and add methods."""
        timer = Timer()
        timer.start('test_op')
        # Perform some operation
        _ = [i**2 for i in range(100)]
        timer.add('test_op')
        self.assertIn('test_op', timer.time_last)
        self.assertIn('test_op', timer.time_sum)
        self.assertGreaterEqual(timer.time_last['test_op'], 0)
        self.assertGreaterEqual(timer.time_sum['test_op'], 0)

    def test_time_accumulation(self):
        """Test that times accumulate across multiple start/add cycles."""
        timer = Timer()
        timer.start('op')
        timer.add('op')
        first_total = timer.time_sum['op']

        timer.start('op')
        timer.add('op')
        self.assertGreaterEqual(timer.time_sum['op'], first_total)


class TestInputSplitStats(unittest.TestCase):
    """Tests for input_split Stats class."""

    def test_initial_values(self):
        """Test Stats initial values."""
        stats = Stats()
        self.assertEqual(stats.visited, 0)
        self.assertEqual(stats.storage_depth, 0)
        self.assertIsInstance(stats.timer, Timer)

    def test_visited_increment(self):
        """Test incrementing visited counter."""
        stats = Stats()
        stats.visited += 10
        self.assertEqual(stats.visited, 10)

    def test_storage_depth_update(self):
        """Test updating storage_depth."""
        stats = Stats()
        stats.storage_depth = 5
        self.assertEqual(stats.storage_depth, 5)


class TestTimerPrint(unittest.TestCase):
    """Tests for Timer print method."""

    @patch('builtins.print')
    def test_print_no_error(self, mock_print):
        """Test that print method outputs correct format."""
        timer = Timer()
        timer.start('op1')
        timer.add('op1')
        timer.print()

        # Should have 4 print calls: 'Time: ', 'op1 X.XXXX', '', 'Accumulated time: ', 'op1 X.XXXX', ''
        # The Timer.print() method uses print with end='' for formatting
        self.assertGreaterEqual(mock_print.call_count, 4)

        # Check that 'Time: ' is printed first
        first_call = mock_print.call_args_list[0]
        self.assertEqual(first_call, call('Time: ', end=''))

        # Check that 'Accumulated time: ' is printed
        accumulated_calls = [c for c in mock_print.call_args_list if 'Accumulated time: ' in str(c)]
        self.assertEqual(len(accumulated_calls), 1)

    @patch('builtins.print')
    def test_print_empty_timer(self, mock_print):
        """Test print with no recorded times."""
        timer = Timer()
        timer.print()

        # Should still print headers even with empty dictionaries
        # 'Time: ', newline, 'Accumulated time: ', newline
        self.assertGreaterEqual(mock_print.call_count, 4)

        # Check 'Time: ' header is printed
        time_calls = [c for c in mock_print.call_args_list if c == call('Time: ', end='')]
        self.assertEqual(len(time_calls), 1)

        # Check 'Accumulated time: ' header is printed
        accum_calls = [c for c in mock_print.call_args_list if c == call('Accumulated time: ', end='')]
        self.assertEqual(len(accum_calls), 1)

    @patch('builtins.print')
    def test_print_multiple_operations(self, mock_print):
        """Test print output with multiple operations."""
        timer = Timer()
        timer.start('op1')
        timer.add('op1')
        timer.start('op2')
        timer.add('op2')
        timer.print()

        # Convert all calls to strings for easier searching
        call_strs = [str(c) for c in mock_print.call_args_list]

        # Should contain both operation names in the output
        op1_in_output = any('op1' in s for s in call_strs)
        op2_in_output = any('op2' in s for s in call_strs)
        self.assertTrue(op1_in_output, "op1 should appear in print output")
        self.assertTrue(op2_in_output, "op2 should appear in print output")


class TestTimerMultipleOperations(unittest.TestCase):
    """Extended tests for Timer with multiple operations."""

    def test_multiple_operations(self):
        """Test tracking multiple different operations."""
        timer = Timer()
        timer.start('op1')
        timer.add('op1')
        timer.start('op2')
        timer.add('op2')
        timer.start('op3')
        timer.add('op3')

        self.assertEqual(len(timer.time_sum), 3)
        self.assertIn('op1', timer.time_sum)
        self.assertIn('op2', timer.time_sum)
        self.assertIn('op3', timer.time_sum)

    def test_time_last_updates(self):
        """Test that time_last is updated correctly."""
        timer = Timer()
        timer.start('op')
        timer.add('op')
        first_last = timer.time_last['op']

        timer.start('op')
        # Introduce small delay
        _ = sum(range(10000))
        timer.add('op')

        # time_last should be the most recent time, not accumulated
        self.assertIsNotNone(timer.time_last['op'])
        # Both should be non-negative
        self.assertGreaterEqual(first_last, 0)
        self.assertGreaterEqual(timer.time_last['op'], 0)

    def test_start_initializes_time_sum_if_not_exists(self):
        """Test that start creates entry in time_sum if it doesn't exist."""
        timer = Timer()
        self.assertNotIn('new_op', timer.time_sum)

        timer.start('new_op')
        self.assertIn('new_op', timer.time_sum)
        self.assertEqual(timer.time_sum['new_op'], 0)

    def test_start_does_not_reset_time_sum(self):
        """Test that start does not reset existing time_sum."""
        timer = Timer()
        timer.start('op')
        timer.add('op')
        first_sum = timer.time_sum['op']

        timer.start('op')  # Second start
        # time_sum should not be reset
        self.assertEqual(timer.time_sum['op'], first_sum)


class TestInitialVerifyCriterionEdgeCases(unittest.TestCase):
    """Additional edge case tests for initial_verify_criterion."""

    def test_large_batch(self):
        """Test with a large batch of samples."""
        batch_size = 100
        lbs = torch.randn(batch_size, 10)
        rhs = torch.zeros(batch_size, 10)

        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        # Check that results are consistent
        verified_by_check = torch.any((lbs - rhs) > 0, dim=-1).all().item()
        self.assertEqual(verified, verified_by_check)

    def test_single_class(self):
        """Test with single output class."""
        lbs = torch.tensor([[0.5], [0.0], [-0.5]])
        rhs = torch.tensor([[0.0], [0.0], [0.0]])

        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertFalse(verified)
        # First sample verified (0.5 > 0), second not (0 not > 0), third not (-0.5 not > 0)
        self.assertEqual(len(unverified_idx), 2)

    def test_negative_rhs(self):
        """Test with negative rhs values."""
        lbs = torch.tensor([[0.0, 0.0]])
        rhs = torch.tensor([[-1.0, -1.0]])
        # lbs - rhs = [[1.0, 1.0]] > 0, so verified
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertTrue(verified)

    def test_mixed_rhs(self):
        """Test with mixed positive and negative rhs values."""
        lbs = torch.tensor([[0.5, -0.5], [0.5, 0.5]])
        rhs = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

        # First sample: 0.5 > 0 (True), -0.5 > 0 (False) -> any True -> verified
        # Second sample: 0.5 > 0 (True), 0.5 > 1.0 (-0.5 > 0) (False) -> any True -> verified
        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertTrue(verified)

    def test_outputs_correct_unverified_indices(self):
        """Test that correct unverified indices are returned."""
        lbs = torch.tensor([
            [1.0, 1.0],   # verified (any > 0)
            [-1.0, -1.0], # not verified
            [0.5, 0.5],   # verified
            [-2.0, -2.0], # not verified
        ])
        rhs = torch.zeros(4, 2)

        verified, unverified_idx = initial_verify_criterion(lbs, rhs)
        self.assertFalse(verified)
        self.assertEqual(len(unverified_idx), 2)
        self.assertIn(1, unverified_idx.tolist())
        self.assertIn(3, unverified_idx.tolist())


class TestStatsExtended(unittest.TestCase):
    """Extended tests for Stats class."""

    def test_timer_is_timer_instance(self):
        """Test that stats.timer is properly initialized."""
        stats = Stats()
        self.assertIsInstance(stats.timer, Timer)
        self.assertEqual(stats.timer.total_func_time, 0.0)

    def test_timer_can_be_used(self):
        """Test that the timer in Stats is functional."""
        stats = Stats()
        stats.timer.start('test')
        stats.timer.add('test')
        self.assertIn('test', stats.timer.time_sum)

    def test_multiple_stats_independent(self):
        """Test that multiple Stats instances are independent."""
        stats1 = Stats()
        stats2 = Stats()

        stats1.visited = 10
        stats1.storage_depth = 5

        self.assertEqual(stats2.visited, 0)
        self.assertEqual(stats2.storage_depth, 0)


if __name__ == '__main__':
    unittest.main()
