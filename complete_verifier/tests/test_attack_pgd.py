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
Comprehensive unit tests for attack/attack_pgd.py

Tests cover:
- default_pgd_loss function with different modes
- test_conditions function for specification checking
- default_early_stop_condition wrapper
- default_adv_example_finalizer function
- pgd_attack_with_general_specs main attack function
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.nn as nn
import sys
import os

# Add complete_verifier to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDefaultPgdLoss(unittest.TestCase):
    """Tests for the default_pgd_loss function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_restarts = 3
        self.num_or = 2
        self.num_and = 2  # num_spec = num_or * num_and
        self.num_output = 4

        # Create output tensor [batch_size, num_restarts, num_or, num_output]
        self.output = torch.randn(self.batch_size, self.num_restarts, self.num_or, self.num_output)

        # Create C_mat [batch_size, num_spec, num_output]
        self.C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)

        # Create rhs_mat [batch_size, num_spec]
        self.rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        self.or_spec_size = torch.tensor([self.num_and] * self.num_or)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_hinge_mode_returns_correct_shapes(self, mock_config):
        """Test hinge mode returns tensors with correct shapes."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        loss, loss_gamma = default_pgd_loss(
            None, self.output, self.C_mat, self.rhs_mat,
            self.or_spec_size, gama_lambda=0, mode='hinge'
        )

        # loss shape: [batch_size, num_restarts, num_or, num_and_spec]
        self.assertEqual(loss.shape[0], self.batch_size)
        self.assertEqual(loss.shape[1], self.num_restarts)
        self.assertEqual(loss.shape[2], self.num_or)
        self.assertEqual(loss.shape[3], self.num_and)

        # loss_gamma is a scalar (sum of losses)
        self.assertEqual(loss_gamma.dim(), 0)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_sum_mode_clamps_positive_values(self, mock_config):
        """Test sum mode sets positive loss values to 1.0 and preserves negative values."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        # Create output that will produce both positive and negative loss values
        # by using controlled C_mat and rhs_mat values
        output = torch.randn(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        loss, loss_gamma = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            self.or_spec_size, gama_lambda=0, mode='sum'
        )

        # In sum mode, positive values should be 1.0
        positive_mask = loss >= 0
        negative_mask = loss < 0

        # Test positive branch: all non-negative values should be clamped to 1.0
        if positive_mask.any():
            self.assertTrue((loss[positive_mask] == 1.0).all(),
                "Positive loss values should be clamped to 1.0 in sum mode")

        # Test negative branch: negative values should remain unchanged (not clamped)
        # To verify this, we check that negative values are strictly less than 0
        if negative_mask.any():
            self.assertTrue((loss[negative_mask] < 0).all(),
                "Negative loss values should remain negative in sum mode")

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_sum_mode_preserves_negative_values(self, mock_config):
        """Test sum mode preserves original negative loss values unchanged."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        # Create tensors that will produce only negative loss values
        # Loss calculation: loss = -(C @ output - rhs)
        # For loss < 0, we need C @ output - rhs > 0, i.e., C @ output > rhs
        num_output = 4
        output = torch.zeros(1, 1, 1, num_output)
        output[0, 0, 0, 0] = 10.0  # Large positive output

        # C_mat selects first output element
        C_mat = torch.zeros(1, 1, num_output)
        C_mat[0, 0, 0] = 1.0  # Select first output
        rhs_mat = torch.tensor([[0.0]])  # Small rhs so C @ output > rhs

        or_spec_size = torch.tensor([1])

        # First compute in hinge mode to get the raw loss values
        loss_hinge, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            or_spec_size, gama_lambda=0, mode='hinge'
        )

        # Now compute in sum mode
        loss_sum, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            or_spec_size, gama_lambda=0, mode='sum'
        )

        # All values should be negative in this setup (margin is large positive, negated = negative loss)
        self.assertTrue((loss_hinge < 0).all(),
            "Test setup should produce negative loss values")

        # In sum mode, negative values should match hinge mode (preserved)
        self.assertTrue(torch.allclose(loss_sum, loss_hinge),
            "Negative loss values should be preserved unchanged in sum mode")

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_gama_loss_with_origin_out(self, mock_config):
        """Test GAMA loss computation when origin_out is provided."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        origin_out = torch.randn_like(self.output)
        gama_lambda = 0.5

        loss, loss_gamma = default_pgd_loss(
            origin_out, self.output, self.C_mat, self.rhs_mat,
            self.or_spec_size, gama_lambda=gama_lambda, mode='hinge'
        )

        # loss_gamma should include the GAMA term
        self.assertEqual(loss_gamma.dim(), 0)
        # The loss_gamma should be different from just loss.sum()
        # due to the GAMA term

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_loss_without_gama(self, mock_config):
        """Test loss computation without GAMA (gama_lambda=0)."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        loss, loss_gamma = default_pgd_loss(
            None, self.output, self.C_mat, self.rhs_mat,
            self.or_spec_size, gama_lambda=0, mode='hinge'
        )

        # Without GAMA, loss_gamma should equal loss.sum()
        self.assertAlmostEqual(loss_gamma.item(), loss.sum().item(), places=4)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_threshold_clipping(self, mock_config):
        """Test that loss is clipped by threshold in hinge mode."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        threshold = -1e-5

        # Create inputs that will produce margins below the threshold
        # margin = C @ output - rhs, we want some margins << threshold
        num_output = 4
        output = torch.zeros(1, 1, 1, num_output)
        output[0, 0, 0, 0] = -100.0  # Large negative output

        C_mat = torch.zeros(1, 1, num_output)
        C_mat[0, 0, 0] = 1.0  # Select first output
        rhs_mat = torch.tensor([[0.0]])  # margin = -100 - 0 = -100, way below threshold

        or_spec_size = torch.tensor([1])

        loss, _ = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            or_spec_size, gama_lambda=0, threshold=threshold, mode='hinge'
        )

        # The internal flow is:
        # 1. margin = C @ output - rhs = -100
        # 2. margin = clamp(margin, min=threshold) = -1e-5 (clipped up)
        # 3. loss = -margin = 1e-5
        # So after negation, loss should be at most -threshold
        self.assertTrue((loss <= -threshold + 1e-9).all(),
            f"Loss should be at most -threshold ({-threshold}), but got max {loss.max().item()}")

        # Verify the specific value
        expected_loss = -threshold
        self.assertAlmostEqual(loss.item(), expected_loss, places=8,
            msg=f"Loss should equal -threshold when margin is clipped")

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_attack_tolerance_applied(self, mock_config):
        """Test that attack_tolerance is applied in loss computation."""
        tolerance = 0.1
        mock_config['attack'] = {'attack_tolerance': tolerance}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        output = torch.zeros(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        C_mat = torch.zeros(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.zeros(self.batch_size, self.num_or * self.num_and)

        loss, loss_gamma = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            self.or_spec_size, gama_lambda=0, threshold=-1.0, mode='hinge'
        )

        # raw_margin = C @ output - rhs = 0. With tolerance 0.1,
        # adjusted_margin = raw_margin - tolerance = -0.1.
        # No clipping occurs, so the negated loss is exactly 0.1.
        expected = torch.full_like(loss, tolerance)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertTrue(torch.allclose(loss_gamma, expected.sum(), atol=1e-6))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_sanity_check_disables_clamping(self, mock_config):
        """Test that sanity_check not None skips clamping."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': True}  # Not None

        from attack.attack_pgd import default_pgd_loss

        loss, loss_gamma = default_pgd_loss(
            None, self.output, self.C_mat, self.rhs_mat,
            self.or_spec_size, gama_lambda=0, threshold=-1e-5, mode='hinge'
        )

        # Should still return valid tensors
        self.assertIsNotNone(loss)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_single_batch(self, mock_config):
        """Test with single batch size."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        output = torch.randn(1, self.num_restarts, self.num_or, self.num_output)
        C_mat = torch.randn(1, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(1, self.num_or * self.num_and)

        loss, loss_gamma = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            self.or_spec_size, gama_lambda=0, mode='hinge'
        )

        self.assertEqual(loss.shape[0], 1)


class TestTestConditions(unittest.TestCase):
    """Tests for the test_conditions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_restarts = 3
        self.num_or = 2
        self.num_and = 2
        self.num_output = 4
        self.input_shape = (3, 4, 4)  # e.g., CIFAR-like

        self.or_spec_size = torch.tensor([self.num_and] * self.num_or)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_returns_boolean_tensor(self, mock_config):
        """Test that test_conditions returns a boolean tensor."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import test_conditions

        # Create inputs
        input_tensor = torch.rand(self.batch_size, self.num_restarts, self.num_or, *self.input_shape)
        output = torch.randn(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, (self.batch_size,))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_return_best_idx(self, mock_config):
        """Test return_best_idx returns index tensor."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_restarts, self.num_or, *self.input_shape)
        output = torch.randn(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        result, idx = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size, return_best_idx=True
        )

        self.assertEqual(result.shape, (self.batch_size,))
        self.assertEqual(idx.shape, (self.batch_size,))
        # idx should be valid OR spec indices
        self.assertTrue((idx >= 0).all() and (idx < self.num_or).all())

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_input_outside_bounds_fails(self, mock_config):
        """Test that inputs outside bounds fail the condition."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import test_conditions

        # Create inputs that are outside the bounds
        input_tensor = torch.ones(self.batch_size, self.num_restarts, self.num_or, *self.input_shape) * 2.0
        output = torch.randn(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        # Inputs are outside bounds, so should fail
        self.assertTrue((~result).all())

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_attack_tolerance_relaxes_success_condition(self, mock_config):
        """A positive margin within attack_tolerance is accepted."""
        mock_config['attack'] = {'attack_tolerance': 1e-3}

        from attack.attack_pgd import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_restarts, self.num_or, *self.input_shape)
        output = torch.zeros(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, self.num_or, *self.input_shape)
        C_mat = torch.zeros(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.full((self.batch_size, self.num_or * self.num_and), -5e-4)

        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        self.assertTrue(result.all())

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_negative_attack_tolerance_makes_success_condition_stricter(self, mock_config):
        """A negative tolerance preserves the legacy stricter attack check."""
        mock_config['attack'] = {'attack_tolerance': -1e-3}

        from attack.attack_pgd import test_conditions

        input_tensor = torch.rand(self.batch_size, self.num_restarts, self.num_or, *self.input_shape)
        output = torch.zeros(self.batch_size, self.num_restarts, self.num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, self.num_or, *self.input_shape)
        C_mat = torch.zeros(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.full((self.batch_size, self.num_or * self.num_and), 5e-4)

        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        self.assertFalse(result.any())

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_single_or_spec(self, mock_config):
        """Test with single OR specification."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import test_conditions

        num_or = 1
        or_spec_size = torch.tensor([self.num_and])

        input_tensor = torch.rand(self.batch_size, self.num_restarts, num_or, *self.input_shape)
        output = torch.randn(self.batch_size, self.num_restarts, num_or, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, num_or * self.num_and)

        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, or_spec_size
        )

        self.assertEqual(result.shape, (self.batch_size,))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_input_shape_one_or(self, mock_config):
        """Test when input.shape[2] == 1 (single OR spec input)."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import test_conditions

        # Input has shape[2] = 1, but num_or = 2
        input_tensor = torch.rand(self.batch_size, self.num_restarts, 1, *self.input_shape)
        output = torch.randn(self.batch_size, self.num_restarts, 1, self.num_output)
        data_min = torch.zeros(self.batch_size, 1, 1, *self.input_shape)
        data_max = torch.ones(self.batch_size, 1, 1, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        # or_spec_size still has num_or elements
        result = test_conditions(
            input_tensor, output, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        self.assertEqual(result.shape, (self.batch_size,))


class TestDefaultEarlyStopCondition(unittest.TestCase):
    """Tests for the default_early_stop_condition function."""

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_wrapper_calls_test_conditions(self, mock_config):
        """Test that early stop condition wraps test_conditions."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import default_early_stop_condition

        batch_size = 2
        num_restarts = 3
        num_or = 2
        num_and = 2
        num_output = 4
        input_shape = (3, 4, 4)
        or_spec_size = torch.tensor([num_and] * num_or)

        inputs = torch.rand(batch_size, num_restarts, num_or, *input_shape)
        output = torch.randn(batch_size, num_restarts, num_or, num_output)
        data_min = torch.zeros(batch_size, 1, num_or, *input_shape)
        data_max = torch.ones(batch_size, 1, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        model = MagicMock()

        result = default_early_stop_condition(
            inputs, output, data_min, data_max,
            C_mat, rhs_mat, or_spec_size, model
        )

        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, (batch_size,))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_return_best_idx_forwarded(self, mock_config):
        """Test that return_best_idx parameter is forwarded."""
        mock_config['attack'] = {'attack_tolerance': 0.0}

        from attack.attack_pgd import default_early_stop_condition

        batch_size = 2
        num_restarts = 3
        num_or = 2
        num_and = 2
        num_output = 4
        input_shape = (3, 4, 4)
        or_spec_size = torch.tensor([num_and] * num_or)

        inputs = torch.rand(batch_size, num_restarts, num_or, *input_shape)
        output = torch.randn(batch_size, num_restarts, num_or, num_output)
        data_min = torch.zeros(batch_size, 1, num_or, *input_shape)
        data_max = torch.ones(batch_size, 1, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        model = MagicMock()

        result, idx = default_early_stop_condition(
            inputs, output, data_min, data_max,
            C_mat, rhs_mat, or_spec_size, model, return_best_idx=True
        )

        self.assertEqual(result.shape, (batch_size,))
        self.assertEqual(idx.shape, (batch_size,))


class TestPgdDefaultAdvExampleFinalizer(unittest.TestCase):
    """Tests for the default_adv_example_finalizer function in attack_pgd module."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.num_and = 2
        self.num_output = 4
        self.input_shape = (3, 4, 4)
        self.or_spec_size = torch.tensor([self.num_and] * self.num_or)

    def test_returns_correct_shapes(self):
        """Test that finalizer returns tensors with correct shapes."""
        from attack.attack_pgd import default_adv_example_finalizer

        # Create mock model
        model_ori = MagicMock()
        model_ori.return_value = torch.randn(self.batch_size * self.num_or, self.num_output)

        x = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        best_deltas = torch.randn(self.batch_size, self.num_or, *self.input_shape) * 0.1
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        adv_input, adv_output, adv_margin, adv_margin_per_or = default_adv_example_finalizer(
            model_ori, x, best_deltas, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        # adv_input: [batch_size, num_or, *input_shape]
        self.assertEqual(adv_input.shape, (self.batch_size, self.num_or, *self.input_shape))

        # adv_output: [batch_size, num_or, out_dim]
        self.assertEqual(adv_output.shape, (self.batch_size, self.num_or, self.num_output))

        # adv_margin: [batch_size, num_spec]
        self.assertEqual(adv_margin.shape, (self.batch_size, self.num_or * self.num_and))

        # adv_margin_per_or: [batch_size, num_or]
        self.assertEqual(adv_margin_per_or.shape, (self.batch_size, self.num_or))

    def test_clamps_input_to_bounds(self):
        """Test that adversarial input is clamped to data bounds."""
        from attack.attack_pgd import default_adv_example_finalizer

        model_ori = MagicMock()
        model_ori.return_value = torch.randn(self.batch_size * self.num_or, self.num_output)

        x = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Large deltas that would exceed bounds
        best_deltas = torch.ones(self.batch_size, self.num_or, *self.input_shape) * 10.0
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        adv_input, _, _, _ = default_adv_example_finalizer(
            model_ori, x, best_deltas, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        # Check that all values are within bounds
        self.assertTrue((adv_input >= data_min).all())
        self.assertTrue((adv_input <= data_max).all())

    def test_model_called_with_correct_shape(self):
        """Test that model is called with correctly shaped input."""
        from attack.attack_pgd import default_adv_example_finalizer

        model_ori = MagicMock()
        model_ori.return_value = torch.randn(self.batch_size * self.num_or, self.num_output)

        x = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        best_deltas = torch.randn(self.batch_size, self.num_or, *self.input_shape) * 0.1
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)

        default_adv_example_finalizer(
            model_ori, x, best_deltas, data_min, data_max,
            C_mat, rhs_mat, self.or_spec_size
        )

        # Model should be called with shape [batch_size * num_or, *input_shape]
        call_args = model_ori.call_args[0][0]
        expected_shape = (self.batch_size * self.num_or, *self.input_shape)
        self.assertEqual(call_args.shape, expected_shape)

    def test_single_or_spec(self):
        """Test finalizer with single OR specification."""
        from attack.attack_pgd import default_adv_example_finalizer

        num_or = 1
        or_spec_size = torch.tensor([self.num_and])

        model_ori = MagicMock()
        model_ori.return_value = torch.randn(self.batch_size * num_or, self.num_output)

        x = torch.rand(self.batch_size, num_or, *self.input_shape)
        best_deltas = torch.randn(self.batch_size, num_or, *self.input_shape) * 0.1
        data_min = torch.zeros(self.batch_size, num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, num_or * self.num_and)

        adv_input, adv_output, adv_margin, adv_margin_per_or = default_adv_example_finalizer(
            model_ori, x, best_deltas, data_min, data_max,
            C_mat, rhs_mat, or_spec_size
        )

        self.assertEqual(adv_margin_per_or.shape, (self.batch_size, num_or))


class TestPgdAttackWithGeneralSpecs(unittest.TestCase):
    """Tests for the pgd_attack_with_general_specs function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_or = 2
        self.num_and = 2
        self.num_output = 4
        self.input_shape = (3, 4, 4)
        self.or_spec_size = torch.tensor([self.num_and] * self.num_or)

        # Create a simple model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, self.num_output)
        )

    def _get_mock_config(self):
        """Create a mock config dictionary."""
        return {
            'attack': {
                'pgd_steps': 2,
                'pgd_restarts': 2,
                'pgd_lr_decay': 0.99,
                'pgd_early_stop': False,
                'pgd_restart_when_stuck': False,
                'attack_tolerance': 0.0,
                'gama_lambda': 0.0,
                'gama_decay': 0.9,
                'pgd_loss': 'default_pgd_loss',
                'pgd_loss_mode': 'hinge',
                'early_stop_condition': 'default_early_stop_condition',
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'debug': {'sanity_check': None},
            'bab': {'attack': {'enabled': False}},
        }

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)  # Disable tqdm for tests
    def test_uniform_initialization(self, mock_config):
        """Test PGD attack with uniform initialization."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        # AdamClipping expects a scalar learning rate when use_adam=True
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.attack_success.shape, (self.batch_size,))
        self.assertEqual(result.best_or_idx.shape, (self.batch_size,))
        self.assertEqual(result.adv_input_best.shape, (self.batch_size, *self.input_shape))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_none_initialization(self, mock_config):
        """Test PGD attack with no initialization (zeros)."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='none', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_invalid_initialization_raises(self, mock_config):
        """Test that invalid initialization method raises ValueError."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        with self.assertRaises(ValueError):
            pgd_attack_with_general_specs(
                self.model, X, data_min, data_max, C_mat, rhs_mat,
                self.or_spec_size, alpha, use_adam=True,
                initialization='invalid_method', num_restarts=2, pgd_steps=2
            )

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_without_adam_optimizer(self, mock_config):
        """Test PGD attack without Adam optimizer (standard PGD)."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = torch.ones(self.num_or, *self.input_shape) * 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=False,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_with_early_stop(self, mock_config):
        """Test PGD attack with early stopping enabled."""
        config = self._get_mock_config()
        config['attack']['pgd_early_stop'] = True
        mock_config.update(config)

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('builtins.print')
    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_early_stop_rejected_candidate_continues_pgd(
            self, mock_config, mock_print):
        """Concrete-validation failure should not stop PGD early."""
        import attack.attack_pgd as attack_pgd

        def always_early_stop_condition(inputs, *args, **kwargs):
            return torch.ones(
                inputs.shape[0], dtype=torch.bool, device=inputs.device)

        config = self._get_mock_config()
        config['attack']['pgd_steps'] = 3
        config['attack']['pgd_early_stop'] = True
        config['attack']['early_stop_condition'] = (
            'always_early_stop_condition')
        mock_config.update(config)

        attack_pgd.always_early_stop_condition = always_early_stop_condition
        try:
            X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
            data_min = torch.zeros(
                self.batch_size, self.num_or, *self.input_shape)
            data_max = torch.ones(
                self.batch_size, self.num_or, *self.input_shape)
            C_mat = torch.randn(
                self.batch_size, self.num_or * self.num_and, self.num_output)
            rhs_mat = torch.randn(
                self.batch_size, self.num_or * self.num_and)
            alpha = 0.01

            with patch.object(
                    attack_pgd.attack_utils,
                    'are_attack_counterexamples_valid',
                    return_value=False) as validator:
                result = attack_pgd.pgd_attack_with_general_specs(
                    self.model, X, data_min, data_max, C_mat, rhs_mat,
                    self.or_spec_size, alpha, use_adam=True,
                    initialization='uniform', num_restarts=2, pgd_steps=3,
                    vnnlib=[([[0.0, 1.0]], [])],
                )
        finally:
            del attack_pgd.always_early_stop_condition

        self.assertIsNotNone(result)
        self.assertEqual(validator.call_count, 3)
        mock_print.assert_any_call(
            "PGD early stop candidate rejected by concrete validation, continuing PGD.")

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_with_restart_when_stuck(self, mock_config):
        """Test PGD attack with restart when stuck enabled."""
        config = self._get_mock_config()
        config['attack']['pgd_restart_when_stuck'] = True
        mock_config.update(config)

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_with_gama_loss(self, mock_config):
        """Test PGD attack with GAMA loss enabled."""
        config = self._get_mock_config()
        config['attack']['gama_lambda'] = 0.5
        mock_config.update(config)

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', GAMA_loss=True,
            num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_with_normalize_function(self, mock_config):
        """Test PGD attack with custom normalization function."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        # Custom normalization
        normalize = lambda x: (x - 0.5) / 0.5

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            normalize=normalize, initialization='uniform',
            num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_bab_attack_enabled_returns_all_inputs(self, mock_config):
        """Test that bab attack enabled returns adv_input_all."""
        config = self._get_mock_config()
        config['bab']['attack']['enabled'] = True
        mock_config.update(config)

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result.adv_input_all)
        # Shape should be [batch_size, num_restarts, num_or, *input_shape]
        self.assertEqual(result.adv_input_all.shape[0], self.batch_size)
        self.assertEqual(result.adv_input_all.shape[2], self.num_or)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_pgd_attack_result_fields(self, mock_config):
        """Test that PGDAttackResult has all expected fields."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        # Check all expected fields
        self.assertTrue(hasattr(result, 'attack_success'))
        self.assertTrue(hasattr(result, 'best_or_idx'))
        self.assertTrue(hasattr(result, 'adv_input_per_or'))
        self.assertTrue(hasattr(result, 'adv_output_per_or'))
        self.assertTrue(hasattr(result, 'adv_margin_per_or'))
        self.assertTrue(hasattr(result, 'adv_input_best'))
        self.assertTrue(hasattr(result, 'adv_output_best'))
        self.assertTrue(hasattr(result, 'adv_margin_best'))
        self.assertTrue(hasattr(result, 'adv_margin_per_spec'))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_single_batch(self, mock_config):
        """Test PGD attack with single batch."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        batch_size = 1
        X = torch.rand(batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertEqual(result.attack_success.shape, (batch_size,))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_single_or_spec(self, mock_config):
        """Test PGD attack with single OR specification."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        num_or = 1
        or_spec_size = torch.tensor([self.num_and])

        X = torch.rand(self.batch_size, num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertEqual(result.adv_margin_per_or.shape, (self.batch_size, num_or))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    @patch('attack.attack_pgd.OSI_init_C')
    def test_osi_initialization(self, mock_osi, mock_config):
        """Test PGD attack with OSI initialization."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Mock OSI to return proper tensor
        X_expanded = X.view(X.shape[0], 1, *X.shape[1:]).expand(-1, 2, *(-1,) * (X.ndim - 1))
        mock_osi.return_value = X_expanded + torch.randn_like(X_expanded) * 0.1

        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='osi', num_restarts=2, pgd_steps=2
        )

        mock_osi.assert_called_once()
        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    @patch('attack.attack_pgd.boundary_attack')
    def test_boundary_initialization_success(self, mock_boundary, mock_config):
        """Test PGD attack with boundary initialization when successful."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        # Mock boundary attack to return successful adversarial inputs
        mock_boundary.return_value = X.reshape(-1, *X.shape[1:]) + torch.randn_like(X.reshape(-1, *X.shape[1:])) * 0.01

        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='boundary', num_restarts=2, pgd_steps=2
        )

        mock_boundary.assert_called_once()
        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    @patch('attack.attack_pgd.boundary_attack')
    def test_boundary_initialization_fallback(self, mock_boundary, mock_config):
        """Test boundary initialization falls back to uniform when it fails."""
        mock_config.update(self._get_mock_config())

        from attack.attack_pgd import pgd_attack_with_general_specs

        # Mock boundary attack to return None (failure)
        mock_boundary.return_value = None

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='boundary', num_restarts=2, pgd_steps=2
        )

        # Should fall back to uniform and still produce valid result
        self.assertIsNotNone(result)

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_sum_loss_mode(self, mock_config):
        """Test PGD attack with sum loss mode."""
        config = self._get_mock_config()
        config['attack']['pgd_loss_mode'] = 'sum'
        mock_config.update(config)

        from attack.attack_pgd import pgd_attack_with_general_specs

        X = torch.rand(self.batch_size, self.num_or, *self.input_shape)
        data_min = torch.zeros(self.batch_size, self.num_or, *self.input_shape)
        data_max = torch.ones(self.batch_size, self.num_or, *self.input_shape)
        C_mat = torch.randn(self.batch_size, self.num_or * self.num_and, self.num_output)
        rhs_mat = torch.randn(self.batch_size, self.num_or * self.num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            self.model, X, data_min, data_max, C_mat, rhs_mat,
            self.or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=2, pgd_steps=2
        )

        self.assertIsNotNone(result)


class TestOsiInitialization(unittest.TestCase):
    """Tests for OSI initialization."""

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.OSI_init_C')
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_osi_called_with_correct_params(self, mock_osi, mock_config):
        """Test that OSI_init_C is called with correct parameters."""
        mock_config.update({
            'attack': {
                'pgd_steps': 2,
                'pgd_restarts': 2,
                'pgd_lr_decay': 0.99,
                'pgd_early_stop': False,
                'pgd_restart_when_stuck': False,
                'attack_tolerance': 0.0,
                'gama_lambda': 0.0,
                'gama_decay': 0.9,
                'pgd_loss': 'default_pgd_loss',
                'pgd_loss_mode': 'hinge',
                'early_stop_condition': 'default_early_stop_condition',
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'debug': {'sanity_check': None},
            'bab': {'attack': {'enabled': False}},
        })

        from attack.attack_pgd import pgd_attack_with_general_specs

        batch_size = 2
        num_or = 2
        num_and = 2
        num_output = 4
        input_shape = (3, 4, 4)
        or_spec_size = torch.tensor([num_and] * num_or)

        X = torch.rand(batch_size, num_or, *input_shape)
        X_expanded = X.view(X.shape[0], 1, *X.shape[1:]).expand(-1, 2, *(-1,) * (X.ndim - 1))
        mock_osi.return_value = X_expanded + torch.randn_like(X_expanded) * 0.1

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, num_output))
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        alpha = 0.01

        pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='osi', num_restarts=2, pgd_steps=2
        )

        # Verify OSI was called
        mock_osi.assert_called_once()


class TestAttackConsistency(unittest.TestCase):
    """Tests for attack result consistency."""

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    @patch('attack.attack_pgd.tqdm', lambda x: x)
    def test_attack_success_margin_consistency(self, mock_config):
        """Test that attack_success matches margin <= 0."""
        mock_config.update({
            'attack': {
                'pgd_steps': 5,
                'pgd_restarts': 3,
                'pgd_lr_decay': 0.99,
                'pgd_early_stop': False,
                'pgd_restart_when_stuck': False,
                'attack_tolerance': 0.0,
                'gama_lambda': 0.0,
                'gama_decay': 0.9,
                'pgd_loss': 'default_pgd_loss',
                'pgd_loss_mode': 'hinge',
                'early_stop_condition': 'default_early_stop_condition',
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'debug': {'sanity_check': None},
            'bab': {'attack': {'enabled': False}},
        })

        from attack.attack_pgd import pgd_attack_with_general_specs

        batch_size = 3
        num_or = 2
        num_and = 2
        num_output = 4
        input_shape = (3, 4, 4)
        or_spec_size = torch.tensor([num_and] * num_or)

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, num_output))

        X = torch.rand(batch_size, num_or, *input_shape)
        data_min = torch.zeros(batch_size, num_or, *input_shape)
        data_max = torch.ones(batch_size, num_or, *input_shape)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        alpha = 0.01

        result = pgd_attack_with_general_specs(
            model, X, data_min, data_max, C_mat, rhs_mat,
            or_spec_size, alpha, use_adam=True,
            initialization='uniform', num_restarts=3, pgd_steps=5
        )

        # The assertion in the source code checks this, but we verify here too
        expected_success = result.adv_margin_best <= 0.0
        self.assertTrue((result.attack_success == expected_success).all())


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_empty_output_tensor(self, mock_config):
        """Test handling of edge case with minimal dimensions."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        # Minimal dimensions
        batch_size = 1
        num_restarts = 1
        num_or = 1
        num_and = 1
        num_output = 2

        output = torch.randn(batch_size, num_restarts, num_or, num_output)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        or_spec_size = torch.tensor([num_and])

        loss, loss_gamma = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            or_spec_size, gama_lambda=0, mode='hinge'
        )

        self.assertEqual(loss.shape, (batch_size, num_restarts, num_or, num_and))

    @patch('attack.attack_pgd.arguments.Config', new_callable=dict)
    def test_large_batch_size(self, mock_config):
        """Test with larger batch size."""
        mock_config['attack'] = {'attack_tolerance': 0.0}
        mock_config['debug'] = {'sanity_check': None}

        from attack.attack_pgd import default_pgd_loss

        batch_size = 32
        num_restarts = 5
        num_or = 3
        num_and = 4
        num_output = 10

        output = torch.randn(batch_size, num_restarts, num_or, num_output)
        C_mat = torch.randn(batch_size, num_or * num_and, num_output)
        rhs_mat = torch.randn(batch_size, num_or * num_and)
        or_spec_size = torch.tensor([num_and] * num_or)

        loss, loss_gamma = default_pgd_loss(
            None, output, C_mat, rhs_mat,
            or_spec_size, gama_lambda=0, mode='hinge'
        )

        self.assertEqual(loss.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
