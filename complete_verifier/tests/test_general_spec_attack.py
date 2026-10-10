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
"""Unit tests for attack/general_spec_attack.py"""

import unittest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import torch.nn as nn
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock tqdm before importing general_spec_attack
mock_tqdm = MagicMock(side_effect=lambda x, *args, **kwargs: x)
sys.modules['tqdm'] = MagicMock()
sys.modules['tqdm'].tqdm = mock_tqdm


# ============================================================================
# Shared Test Configuration
# ============================================================================

# Standard attack configuration used by most PGD attack tests
STANDARD_ATTACK_CONFIG = {
    'attack': {
        'pgd_steps': 2,
        'pgd_restarts': 2,
        'pgd_lr_decay': 0.99,
        'pgd_early_stop': False,
        'pgd_restart_when_stuck': False,
        'attack_tolerance': 0.0,
        'gama_lambda': 0.5,
        'gama_decay': 0.9,
        'pgd_loss': 'default_pgd_loss',
        'pgd_loss_mode': 'hinge',
        'adv_example_finalizer': 'default_adv_example_finalizer',
    },
    'bab': {'attack': {'enabled': False}},
}


def make_config_patch(**overrides):
    """Create a config dict with optional overrides.

    Args:
        **overrides: Key-value pairs to override in the config.
            Nested keys can be specified as 'attack.pgd_steps': 5.

    Returns:
        dict: A new config dict with overrides applied.
    """
    import copy
    config = copy.deepcopy(STANDARD_ATTACK_CONFIG)
    for key, value in overrides.items():
        parts = key.split('.')
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return config


# Pre-built config variants for common test scenarios
CONFIG_BAB_ATTACK_ENABLED = make_config_patch(**{'bab.attack.enabled': True})
CONFIG_RESTART_WHEN_STUCK = make_config_patch(**{'attack.pgd_restart_when_stuck': True})
CONFIG_EARLY_STOP = make_config_patch(**{
    'attack.pgd_steps': 10,
    'attack.pgd_early_stop': True,
    'attack.early_stop_condition': 'default_early_stop_condition',
})
CONFIG_GAMA_LOSS = make_config_patch(**{'attack.pgd_steps': 3})
CONFIG_DEFAULTS_TEST = make_config_patch(**{
    'attack.pgd_steps': 5,
    'attack.pgd_restarts': 3,
})
CONFIG_EARLY_STOP_UNEQUAL = make_config_patch(**{
    'attack.pgd_steps': 5,
    'attack.pgd_early_stop': True,
    'attack.early_stop_condition': 'default_early_stop_condition',
})


class TestDefaultPgdLoss(unittest.TestCase):
    """Tests for default_pgd_loss function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 3
        self.num_and = 2
        self.num_restarts = 4
        self.num_output = 5

        # or_spec_size defines the number of AND clauses in each OR clause
        self.or_spec_size = torch.tensor([2, 2, 2])  # Total specs = 6
        self.num_spec = self.or_spec_size.sum().item()

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_basic_loss_computation(self):
        """Test basic loss computation without GAMA loss."""
        from attack.general_spec_attack import default_pgd_loss

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        loss, total_loss = default_pgd_loss(
            None, output, C_mat, rhs_mat, self.or_spec_size,
            gama_lambda=0, threshold=-1e-5, mode='hinge'
        )

        # Loss should have shape [batch, num_spec, num_restarts]
        self.assertEqual(loss.shape, (self.batch_size, self.num_spec, self.num_restarts))
        # total_loss should be a scalar
        self.assertEqual(total_loss.dim(), 0)

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_loss_with_gama(self):
        """Test loss computation with GAMA loss.

        Note: The GAMA loss in default_pgd_loss requires origin_out to have the same
        shape as the repeated output [batch, num_spec, num_restarts, num_output],
        since output is repeat_interleave'd before the GAMA loss is computed.
        """
        from attack.general_spec_attack import default_pgd_loss

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        # origin_out needs to match the shape after repeat_interleave
        origin_out = torch.randn(self.batch_size, self.num_spec, self.num_restarts, self.num_output)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        loss, total_loss = default_pgd_loss(
            origin_out, output, C_mat, rhs_mat, self.or_spec_size,
            gama_lambda=0.5, threshold=-1e-5, mode='hinge'
        )

        self.assertEqual(loss.shape, (self.batch_size, self.num_spec, self.num_restarts))
        self.assertEqual(total_loss.dim(), 0)

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_loss_sum_mode(self):
        """Test loss computation in sum mode."""
        from attack.general_spec_attack import default_pgd_loss

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        loss, total_loss = default_pgd_loss(
            None, output, C_mat, rhs_mat, self.or_spec_size,
            gama_lambda=0, threshold=-1e-5, mode='sum'
        )

        # In sum mode, non-negative losses are set to 1.0
        self.assertEqual(loss.shape, (self.batch_size, self.num_spec, self.num_restarts))

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_loss_threshold_clipping(self):
        """Test that threshold clipping works correctly."""
        from attack.general_spec_attack import default_pgd_loss

        # Create controlled output to get predictable margins
        output = torch.zeros(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        # Build C_mat properly: [batch, num_spec, num_output]
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        for i in range(self.num_spec):
            C_mat[:, i, i % self.num_output] = 1.0
        # Set rhs large enough to create negative margins
        rhs_mat = torch.ones(self.batch_size, self.num_spec) * 10

        threshold = -5.0
        loss, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat, self.or_spec_size,
            gama_lambda=0, threshold=threshold, mode='hinge'
        )

        # After clamping and negation, loss should be clipped
        self.assertTrue((loss <= -threshold).all())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 1e-3}
    })
    def test_attack_tolerance_applied(self):
        """Test that attack tolerance is applied to loss."""
        from attack.general_spec_attack import default_pgd_loss

        output = torch.zeros(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.zeros(self.batch_size, self.num_spec)

        loss, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat, self.or_spec_size,
            gama_lambda=0, threshold=-1.0, mode='hinge'
        )

        # raw_margin = C @ output - rhs = 0. With tolerance 1e-3,
        # adjusted_margin = raw_margin - tolerance = -1e-3.
        # No clipping occurs, so the negated loss is exactly 1e-3.
        expected = torch.full_like(loss, 1e-3)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))


class TestTestConditions(unittest.TestCase):
    """Tests for test_conditions function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.num_restarts = 3
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2])
        self.num_spec = 4

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_successful_attack_detected(self):
        """Test that successful attack is detected."""
        from attack.general_spec_attack import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_or, self.num_restarts, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)

        # Create output and specs such that margin is negative
        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.ones(self.batch_size, self.num_spec) * 100  # Very large rhs

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, self.or_spec_size)

        # With large negative margins, attack should succeed
        self.assertEqual(result.shape, (self.batch_size,))
        self.assertTrue(result.all())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_failed_attack_detected(self):
        """Test that failed attack is detected."""
        from attack.general_spec_attack import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_or, self.num_restarts, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.ones(self.batch_size, self.num_spec) * -100  # Very negative rhs

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, self.or_spec_size)

        # With large positive margins, attack should fail
        self.assertFalse(result.all())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 1e-3}
    })
    def test_attack_tolerance_relaxes_success_condition(self):
        """A positive margin within attack_tolerance is accepted."""
        from attack.general_spec_attack import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_or, self.num_restarts, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)
        output = torch.zeros(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.full((self.batch_size, self.num_spec), -5e-4)

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, self.or_spec_size)

        self.assertTrue(result.all())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': -1e-3}
    })
    def test_negative_attack_tolerance_makes_success_condition_stricter(self):
        """A negative tolerance preserves the legacy stricter attack check."""
        from attack.general_spec_attack import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_or, self.num_restarts, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)
        output = torch.zeros(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.full((self.batch_size, self.num_spec), 5e-4)

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, self.or_spec_size)

        self.assertFalse(result.any())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_return_best_idx(self):
        """Test that best OR index is returned when requested."""
        from attack.general_spec_attack import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_or, self.num_restarts, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        result, best_idx = test_conditions(input_tensor, output, data_min, data_max,
                                           C_mat, rhs_mat, self.or_spec_size, return_best_idx=True)

        self.assertEqual(result.shape, (self.batch_size,))
        self.assertEqual(best_idx.shape, (self.batch_size,))
        self.assertTrue((best_idx >= 0).all() and (best_idx < self.num_or).all())

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_out_of_range_input_penalized(self):
        """Test that inputs outside data bounds get inf margin."""
        from attack.general_spec_attack import test_conditions

        # Create input outside bounds
        input_tensor = torch.ones(self.batch_size, self.num_or, self.num_restarts, *self.input_shape) * 2.0
        data_min = torch.zeros(self.batch_size, self.num_or, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, 1, *self.input_shape)

        output = torch.randn(self.batch_size, self.num_or, self.num_restarts, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.ones(self.batch_size, self.num_spec) * 100

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, self.or_spec_size)

        # Out of range inputs should not be considered valid attacks
        self.assertFalse(result.all())


class TestDefaultEarlyStopCondition(unittest.TestCase):
    """Tests for default_early_stop_condition function."""

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_delegates_to_test_conditions(self):
        """Test that default_early_stop_condition delegates to test_conditions."""
        from attack.general_spec_attack import default_early_stop_condition

        batch_size = 2
        num_or = 2
        num_restarts = 1
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4

        inputs = torch.rand(batch_size, num_or, num_restarts, *input_shape)
        output = torch.randn(batch_size, num_or, num_restarts, num_output)
        data_min = torch.zeros(batch_size, num_or, 1, *input_shape)
        data_max = torch.ones(batch_size, num_or, 1, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        model = MagicMock()

        result = default_early_stop_condition(
            inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size, model
        )

        self.assertEqual(result.shape, (batch_size,))


class TestFindOptimalLoss(unittest.TestCase):
    """Tests for find_optimal_loss function."""

    def test_finds_best_loss_per_or(self):
        """Test that find_optimal_loss correctly finds best losses."""
        from attack.general_spec_attack import find_optimal_loss

        batch_size = 2
        num_spec = 6
        num_restarts = 4
        num_or = 3
        or_spec_size = torch.tensor([2, 2, 2])

        # Create loss tensor with known values
        loss = torch.randn(batch_size, num_spec, num_restarts)

        # Precompute group indices
        group_indices = torch.repeat_interleave(
            torch.arange(num_or), or_spec_size
        )

        best_losses, restart_indices = find_optimal_loss(loss, group_indices, num_or)

        self.assertEqual(best_losses.shape, (batch_size, num_or))
        self.assertEqual(restart_indices.shape, (batch_size, num_or))
        # Restart indices should be valid
        self.assertTrue((restart_indices >= 0).all() and (restart_indices < num_restarts).all())

    def test_worst_among_ands_best_among_restarts(self):
        """Test that function finds worst among ANDs and best among restarts."""
        from attack.general_spec_attack import find_optimal_loss

        batch_size = 1
        num_or = 2
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        num_restarts = 2

        # Create specific loss values
        # OR 0: ANDs at indices 0, 1
        # OR 1: ANDs at indices 2, 3
        loss = torch.tensor([[[
            [1.0, 2.0],  # AND 0: restarts [1, 2] -> worst across restarts (for each restart)
            [0.5, 1.5],  # AND 1: restarts [0.5, 1.5]
            [3.0, 0.1],  # AND 2: restarts [3, 0.1]
            [2.0, 0.2],  # AND 3: restarts [2, 0.2]
        ]]]).squeeze(0)  # Shape: [1, 4, 2]

        group_indices = torch.repeat_interleave(
            torch.arange(num_or), or_spec_size
        )

        best_losses, restart_indices = find_optimal_loss(loss, group_indices, num_or)

        # For OR 0 (ANDs 0,1):
        #   restart 0: min(1.0, 0.5) = 0.5
        #   restart 1: min(2.0, 1.5) = 1.5
        #   best among restarts: max(0.5, 1.5) = 1.5 at restart 1
        self.assertEqual(best_losses.shape, (batch_size, num_or))
        self.assertEqual(restart_indices.shape, (batch_size, num_or))


class TestGeneralSpecDefaultAdvExampleFinalizer(unittest.TestCase):
    """Tests for default_adv_example_finalizer function in general_spec_attack module."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 3
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2, 2])
        self.num_spec = 6

    @patch('builtins.print')
    def test_returns_correct_shapes(self, mock_print):
        """Test that finalizer returns tensors with correct shapes."""
        from attack.general_spec_attack import default_adv_example_finalizer

        model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

        x = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        best_deltas = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        adv_input, adv_output, adv_margin, adv_margin_per_or = default_adv_example_finalizer(
            model, x, best_deltas, data_min, data_max, C_mat, rhs_mat, self.or_spec_size
        )

        self.assertEqual(adv_input.shape, (self.batch_size, self.num_or, *self.input_shape))
        self.assertEqual(adv_output.shape, (self.batch_size, self.num_or, self.num_output))
        self.assertEqual(adv_margin.shape, (self.batch_size, self.num_spec))
        self.assertEqual(adv_margin_per_or.shape, (self.batch_size, self.num_or))

    @patch('builtins.print')
    def test_clamps_adversarial_input(self, mock_print):
        """Test that adversarial input is clamped to data bounds."""
        from attack.general_spec_attack import default_adv_example_finalizer

        model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

        x = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Large deltas that would exceed bounds
        best_deltas = torch.ones(self.batch_size, self.num_or, *self.input_shape) * 10
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        adv_input, _, _, _ = default_adv_example_finalizer(
            model, x, best_deltas, data_min, data_max, C_mat, rhs_mat, self.or_spec_size
        )

        # Adversarial input should be within bounds
        self.assertTrue((adv_input >= data_min).all())
        self.assertTrue((adv_input <= data_max).all())

    @patch('builtins.print')
    def test_shape_assertion_num_or(self, mock_print):
        """Test that num_or shape assertion works."""
        from attack.general_spec_attack import default_adv_example_finalizer

        model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

        # x with wrong num_or dimension (not matching or_spec_size)
        x = torch.rand(self.batch_size, 5, *self.input_shape)  # 5 != num_or=3 and != 1
        best_deltas = torch.zeros(self.batch_size, 5, *self.input_shape)
        data_min = torch.zeros(self.batch_size, 5, *self.input_shape)
        data_max = torch.ones(self.batch_size, 5, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)

        with self.assertRaises(AssertionError):
            default_adv_example_finalizer(
                model, x, best_deltas, data_min, data_max, C_mat, rhs_mat, self.or_spec_size
            )


class TestPgdAttackWithGeneralSpecs(unittest.TestCase):
    """Tests for pgd_attack_with_general_specs function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2])
        self.num_spec = 4

        # Simple model for testing
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_uniform_initialization(self, mock_print):
        """Test PGD attack with uniform initialization."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.attack_success.shape, (self.batch_size,))
        self.assertEqual(result.best_or_idx.shape, (self.batch_size,))
        self.assertEqual(result.adv_input_per_or.shape, (self.batch_size, self.num_or, *self.input_shape))

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_none_initialization(self, mock_print):
        """Test PGD attack with none (zero) initialization."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='none', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_invalid_initialization_raises(self, mock_print):
        """Test that invalid initialization raises ValueError."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        with self.assertRaises(ValueError):
            pgd_attack_with_general_specs(
                self.model, X, data_min, data_max, C_mat, rhs_mat,
                self.or_spec_size, alpha, use_adam=True,
                initialization='invalid_method', num_restarts=2, pgd_steps=2
            )

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_shape_assertion(self, mock_print):
        """Test that shape assertion fails for mismatched X, data_min, data_max."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Mismatched shapes
        data_min = torch.zeros(self.batch_size, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        with self.assertRaises(AssertionError):
            pgd_attack_with_general_specs(
                self.model, X, data_min, data_max, C_mat, rhs_mat,
                self.or_spec_size, alpha, use_adam=True,
                initialization='uniform', num_restarts=2, pgd_steps=2
            )

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_BAB_ATTACK_ENABLED)
    def test_adv_input_all_returned_when_bab_attack_enabled(self, mock_print):
        """Test that adv_input_all is returned when bab attack is enabled."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result.adv_input_all)

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_without_adam(self, mock_print):
        """Test PGD attack without Adam optimizer (using sign gradient)."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        # Alpha needs to be a tensor with proper shape for non-Adam
        alpha = torch.full((self.num_or, *self.input_shape), 0.01)

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=False,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_RESTART_WHEN_STUCK)
    def test_restart_when_stuck(self, mock_print):
        """Test PGD attack with restart_when_stuck enabled."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_GAMA_LOSS)
    def test_gama_loss(self, mock_print):
        """Test PGD attack with GAMA loss.

        Note: This test uses equal or_spec_size so that the GAMA loss computation
        works correctly. When num_or == num_spec (each OR has 1 AND), there's no
        dimension mismatch after repeat_interleave.
        """
        from attack.general_spec_attack import pgd_attack_with_general_specs

        # Use single AND per OR to avoid dimension mismatch in GAMA loss
        or_spec_size = torch.tensor([1, 1])
        num_spec = 2

        X = torch.rand(self.batch_size, len(or_spec_size), *self.input_shape)
        data_min = torch.zeros(self.batch_size, len(or_spec_size), *self.input_shape)
        data_max = torch.ones(self.batch_size, len(or_spec_size), *self.input_shape)
        C_mat = torch.randn(self.batch_size, num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', GAMA_loss=True,
            num_restarts=2, pgd_steps=3
        )

        self.assertIsNotNone(result)


class TestPgdAttackWithOSIInitialization(unittest.TestCase):
    """Tests for OSI initialization in pgd_attack_with_general_specs."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2])
        self.num_spec = 4
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

    @patch('attack.general_spec_attack.OSI_init_C')
    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_osi_initialization_calls_osi_init_c(self, mock_print, mock_osi):
        """Test that OSI initialization calls OSI_init_C."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Set up mock to return proper shaped tensor
        X_expanded = X.unsqueeze(2).expand(self.batch_size, self.num_or, 2, *self.input_shape)
        mock_osi.return_value = X_expanded + torch.randn_like(X_expanded) * 0.1

        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='osi', num_restarts=2, pgd_steps=2
        )

        mock_osi.assert_called_once()
        self.assertIsNotNone(result)


class TestPgdAttackWithBoundaryInitialization(unittest.TestCase):
    """Tests for boundary initialization in pgd_attack_with_general_specs."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2])
        self.num_spec = 4
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

    @patch('attack.general_spec_attack.boundary_attack')
    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_boundary_initialization_fallback_to_uniform(self, mock_print, mock_boundary):
        """Test that boundary initialization falls back to uniform when it returns None."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        mock_boundary.return_value = None

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='boundary', num_restarts=2, pgd_steps=2
        )

        mock_boundary.assert_called_once()
        self.assertIsNotNone(result)


class TestPgdAttackEarlyStop(unittest.TestCase):
    """Tests for early stopping in pgd_attack_with_general_specs."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.input_shape = (3, 4, 4)
        self.num_output = 5
        self.or_spec_size = torch.tensor([2, 2])
        self.num_spec = 4
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(48, self.num_output))

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_EARLY_STOP)
    def test_early_stop_enabled(self, mock_print):
        """Test PGD attack with early stop enabled."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=10
        )

        self.assertIsNotNone(result)

    @patch('builtins.print')
    @patch('arguments.Config', make_config_patch(**{
        'attack.pgd_steps': 3,
        'attack.pgd_early_stop': True,
        'attack.early_stop_condition': 'always_early_stop_condition',
    }))
    def test_early_stop_rejected_candidate_continues_pgd(self, mock_print):
        """Concrete-validation failure should not stop PGD early."""
        import attack.general_spec_attack as general_spec_attack

        def always_early_stop_condition(inputs, *args, **kwargs):
            return torch.ones(
                inputs.shape[0], dtype=torch.bool, device=inputs.device)

        general_spec_attack.always_early_stop_condition = (
            always_early_stop_condition)
        try:
            X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
            data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
            data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
            C_mat = torch.randn(self.batch_size, self.num_spec, self.num_output)
            rhs_mat = torch.randn(self.batch_size, self.num_spec)
            alpha = 0.01

            with patch.object(
                    general_spec_attack.attack_utils,
                    'are_attack_counterexamples_valid',
                    return_value=False) as validator:
                result = general_spec_attack.pgd_attack_with_general_specs(
                    self.model, X, data_min, data_max, C_mat, rhs_mat,
                    self.or_spec_size, alpha, use_adam=True,
                    initialization='uniform', num_restarts=2, pgd_steps=3,
                    vnnlib=[([[0.0, 1.0]], [])],
                )
        finally:
            del general_spec_attack.always_early_stop_condition

        self.assertIsNotNone(result)
        self.assertEqual(validator.call_count, 3)
        mock_print.assert_any_call(
            "PGD early stop candidate rejected by concrete validation, continuing PGD.")


class TestPgdAttackResultDataclass(unittest.TestCase):
    """Tests for PGDAttackResult dataclass structure."""

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_result_has_all_expected_fields(self, mock_print):
        """Test that PGDAttackResult has all expected fields."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 2
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        # Check all expected fields exist
        self.assertTrue(hasattr(result, 'attack_success'))
        self.assertTrue(hasattr(result, 'best_or_idx'))
        self.assertTrue(hasattr(result, 'adv_input_per_or'))
        self.assertTrue(hasattr(result, 'adv_output_per_or'))
        self.assertTrue(hasattr(result, 'adv_margin_per_or'))
        self.assertTrue(hasattr(result, 'adv_input_best'))
        self.assertTrue(hasattr(result, 'adv_output_best'))
        self.assertTrue(hasattr(result, 'adv_margin_best'))
        self.assertTrue(hasattr(result, 'adv_margin_per_spec'))
        self.assertTrue(hasattr(result, 'adv_input_all'))


class TestLossComputationMathematical(unittest.TestCase):
    """Mathematical correctness tests for loss computation."""

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_loss_computation_correctness(self):
        """Test that loss is computed correctly: -clamp(C*output - rhs, min=threshold)."""
        from attack.general_spec_attack import default_pgd_loss

        batch_size = 1
        num_or = 2
        num_restarts = 1
        num_output = 3
        or_spec_size = torch.tensor([1, 1])
        num_spec = 2

        # Simple case: output = [1, 2, 3], C = eye, rhs = [2, 5]
        output = torch.tensor([[[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]]])  # [1, 2, 1, 3]
        C_mat = torch.eye(num_output)[:num_spec].unsqueeze(0)  # [1, 2, 3]
        rhs_mat = torch.tensor([[2.0, 5.0]])  # [1, 2]

        # Use very low threshold so margins don't get clamped
        threshold = -100.0
        loss, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat, or_spec_size,
            gama_lambda=0, threshold=threshold, mode='hinge'
        )

        # For spec 0: C[0] * output = 1*1 + 0*2 + 0*3 = 1, margin = 1 - 2 = -1
        # For spec 1: C[1] * output = 0*1 + 1*2 + 0*3 = 2, margin = 2 - 5 = -3
        # After clamp(min=-100): -1, -3 (no clamping)
        # After negation: 1, 3
        expected_loss = torch.tensor([[[1.0], [3.0]]])
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-5))


class TestTestConditionsMathematical(unittest.TestCase):
    """Mathematical correctness tests for test_conditions."""

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_or_and_logic(self):
        """Test that OR-AND logic is computed correctly."""
        from attack.general_spec_attack import test_conditions

        batch_size = 1
        num_or = 2
        num_restarts = 1
        input_shape = (1,)
        num_output = 2

        # Two OR clauses, each with 2 AND clauses
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4

        input_tensor = torch.zeros(batch_size, num_or, num_restarts, *input_shape)
        data_min = torch.zeros(batch_size, num_or, 1, *input_shape)
        data_max = torch.ones(batch_size, num_or, 1, *input_shape)

        # Output: [1, 2]
        output = torch.tensor([[[[1.0, 2.0]], [[1.0, 2.0]]]])  # [1, 2, 1, 2]

        # C_mat: identity-like, so C*output selects output values
        # rhs_mat: [0.5, 0.5, 3.0, 3.0]
        # For OR 0 (ANDs 0,1): margins = [1-0.5, 2-0.5] = [0.5, 1.5] -> max = 1.5 > 0 (fail)
        # For OR 1 (ANDs 2,3): margins = [1-3, 2-3] = [-2, -1] -> max = -1 <= 0 (success)
        # Since at least one OR succeeds, overall success = True
        C_mat = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
        rhs_mat = torch.tensor([[0.5, 0.5, 3.0, 3.0]])

        result = test_conditions(input_tensor, output, data_min, data_max,
                                 C_mat, rhs_mat, or_spec_size)

        self.assertTrue(result.all())


class TestDefaultPgdLossWithDifferentOrSpecSizes(unittest.TestCase):
    """Tests for default_pgd_loss with varying OR spec sizes."""

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_unequal_or_spec_sizes(self):
        """Test loss computation with unequal OR spec sizes."""
        from attack.general_spec_attack import default_pgd_loss

        batch_size = 2
        num_restarts = 3
        num_output = 5

        # Unequal sizes: OR 0 has 3 ANDs, OR 1 has 2 ANDs
        or_spec_size = torch.tensor([3, 2])
        num_or = 2
        num_spec = 5

        output = torch.randn(batch_size, num_or, num_restarts, num_output)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)

        loss, total_loss = default_pgd_loss(
            None, output, C_mat, rhs_mat, or_spec_size,
            gama_lambda=0, threshold=-1e-5, mode='hinge'
        )

        self.assertEqual(loss.shape, (batch_size, num_spec, num_restarts))


class TestFindOptimalLossEdgeCases(unittest.TestCase):
    """Edge case tests for find_optimal_loss."""

    def test_single_restart(self):
        """Test find_optimal_loss with single restart."""
        from attack.general_spec_attack import find_optimal_loss

        batch_size = 2
        num_or = 3
        or_spec_size = torch.tensor([2, 2, 2])
        num_spec = 6
        num_restarts = 1

        loss = torch.randn(batch_size, num_spec, num_restarts)
        group_indices = torch.repeat_interleave(torch.arange(num_or), or_spec_size)

        best_losses, restart_indices = find_optimal_loss(loss, group_indices, num_or)

        self.assertEqual(best_losses.shape, (batch_size, num_or))
        # With single restart, all indices should be 0
        self.assertTrue((restart_indices == 0).all())

    def test_single_and_per_or(self):
        """Test find_optimal_loss with single AND per OR."""
        from attack.general_spec_attack import find_optimal_loss

        batch_size = 2
        num_or = 3
        or_spec_size = torch.tensor([1, 1, 1])
        num_spec = 3
        num_restarts = 4

        loss = torch.randn(batch_size, num_spec, num_restarts)
        group_indices = torch.repeat_interleave(torch.arange(num_or), or_spec_size)

        best_losses, restart_indices = find_optimal_loss(loss, group_indices, num_or)

        self.assertEqual(best_losses.shape, (batch_size, num_or))
        self.assertEqual(restart_indices.shape, (batch_size, num_or))


class TestPgdAttackDeviceHandling(unittest.TestCase):
    """Tests for device handling in pgd_attack_with_general_specs."""

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_cpu_device(self, mock_print):
        """Test that attack works on CPU."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 2
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape, device='cpu')
        data_min = torch.zeros(batch_size, num_or, *input_shape, device='cpu')
        data_max = torch.ones(batch_size, num_or, *input_shape, device='cpu')
        C_mat = torch.randn(batch_size, num_spec, num_output, device='cpu')
        rhs_mat = torch.randn(batch_size, num_spec, device='cpu')
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertEqual(result.attack_success.device.type, 'cpu')


class TestDefaultPgdLossGradients(unittest.TestCase):
    """Tests for gradient computation through default_pgd_loss."""

    @patch('arguments.Config', {
        'attack': {'attack_tolerance': 0.0}
    })
    def test_gradients_flow(self):
        """Test that gradients flow through loss computation."""
        from attack.general_spec_attack import default_pgd_loss

        batch_size = 2
        num_or = 2
        num_restarts = 3
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4

        output = torch.randn(batch_size, num_or, num_restarts, num_output, requires_grad=True)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)

        _, total_loss = default_pgd_loss(
            None, output, C_mat, rhs_mat, or_spec_size,
            gama_lambda=0, threshold=-1e-5, mode='hinge'
        )

        total_loss.backward()

        # Gradients should flow to output
        self.assertIsNotNone(output.grad)
        self.assertEqual(output.grad.shape, output.shape)


class TestNormalizeFunction(unittest.TestCase):
    """Tests for custom normalize function in pgd_attack_with_general_specs."""

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_custom_normalize_function(self, mock_print):
        """Test PGD attack with custom normalize function."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 2
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        # Custom normalization
        def custom_normalize(x):
            return (x - 0.5) / 0.5

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            normalize=custom_normalize,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)


class TestPgdAttackSingleBatch(unittest.TestCase):
    """Tests for pgd_attack_with_general_specs with single batch."""

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_single_batch(self, mock_print):
        """Test PGD attack with single batch element."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 1
        num_or = 2
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertEqual(result.attack_success.shape, (batch_size,))


class TestPgdAttackSingleOr(unittest.TestCase):
    """Tests for pgd_attack_with_general_specs with single OR clause."""

    @patch('builtins.print')
    @patch('arguments.Config', STANDARD_ATTACK_CONFIG)
    def test_single_or_clause(self, mock_print):
        """Test PGD attack with single OR clause."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 1
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([3])
        num_spec = 3
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertEqual(result.adv_margin_per_or.shape, (batch_size, num_or))


class TestConfigDefaults(unittest.TestCase):
    """Tests for using config defaults when parameters are None."""

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_DEFAULTS_TEST)
    def test_uses_config_defaults(self, mock_print):
        """Test that config defaults are used when parameters are None."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 2
        input_shape = (3, 4, 4)
        num_output = 5
        or_spec_size = torch.tensor([2, 2])
        num_spec = 4
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        # Pass None for num_restarts and pgd_steps to use config defaults
        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=None, pgd_steps=None
        )

        self.assertIsNotNone(result)


class TestEarlyStopConditionWithDifferentOrSpecSizes(unittest.TestCase):
    """Tests for early stop with different OR spec sizes."""

    @patch('builtins.print')
    @patch('arguments.Config', CONFIG_EARLY_STOP_UNEQUAL)
    def test_early_stop_with_unequal_or_spec_sizes(self, mock_print):
        """Test early stop with unequal OR spec sizes."""
        from attack.general_spec_attack import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 3
        input_shape = (3, 4, 4)
        num_output = 5
        # Unequal sizes
        or_spec_size = torch.tensor([1, 2, 3])
        num_spec = 6
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_spec, num_output)
        rhs_mat = torch.randn(batch_size, num_spec)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=5
        )

        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
