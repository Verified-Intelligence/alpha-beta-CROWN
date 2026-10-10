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
"""Unit tests for input_split/attack_in_input_split.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPgdAttackOnDomainsStageInit(unittest.TestCase):
    """Tests for pgd_attack_on_domains with stage='init'."""

    def _create_mock_domains(self, batch_size=2, input_dim=3, output_dim=4, num_and=2):
        """Create mock domains object for init stage."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        # __getitem__ returns: lb, dm_l, dm_u, c, rhs, spec_size
        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=2, input_dim=3, output_dim=4, attack_success=False):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([attack_success]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_uses_all_domains(self, mock_pgd_attack, mock_config):
        """Test that init stage uses all domains."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim, output_dim = 3, 5, 4
        mock_domains = self._create_mock_domains(batch_size, input_dim, output_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(
            batch_size, input_dim, output_dim, attack_success=False
        )

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertFalse(result)
        mock_pgd_attack.assert_called_once()
        self.assertIs(mock_pgd_attack.call_args.kwargs['vnnlib'], vnnlib)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_computes_center(self, mock_pgd_attack, mock_config):
        """Test that center_x is computed as (dm_l + dm_u) / 2."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim = 2, 3
        mock_domains = self._create_mock_domains(batch_size, input_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size, input_dim)

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_args = mock_pgd_attack.call_args
        center_x = call_args[0][1]  # second positional arg
        dm_l = call_args[0][2]
        dm_u = call_args[0][3]

        expected_center = (dm_l + dm_u) / 2
        self.assertTrue(torch.allclose(center_x, expected_center))

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_auto_alpha(self, mock_pgd_attack, mock_config):
        """Test that auto alpha is computed from (dm_u - dm_l).max() / 8."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 50,
                'pgd_restarts': 10,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim = 2, 3
        mock_domains = self._create_mock_domains(batch_size, input_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size, input_dim)

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_kwargs = mock_pgd_attack.call_args[1]
        self.assertIn('alpha', call_kwargs)
        # alpha should be (dm_u - dm_l).max() / 8
        self.assertIsInstance(call_kwargs['alpha'], (float, torch.Tensor))

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_manual_alpha(self, mock_pgd_attack, mock_config):
        """Test that manual alpha value is used."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': '0.01',
                'pgd_steps': 50,
                'pgd_restarts': 10,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_kwargs = mock_pgd_attack.call_args[1]
        self.assertEqual(call_kwargs['alpha'], 0.01)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_pgd_steps_passed(self, mock_pgd_attack, mock_config):
        """Test that pgd_steps is passed correctly."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 75,
                'pgd_restarts': 10,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_kwargs = mock_pgd_attack.call_args[1]
        self.assertEqual(call_kwargs['pgd_steps'], 75)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_init_stage_num_restarts_passed(self, mock_pgd_attack, mock_config):
        """Test that num_restarts is passed correctly."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 50,
                'pgd_restarts': 25,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_kwargs = mock_pgd_attack.call_args[1]
        self.assertEqual(call_kwargs['num_restarts'], 25)


class TestPgdAttackOnDomainsStageBab(unittest.TestCase):
    """Tests for pgd_attack_on_domains with stage='bab'."""

    def _create_mock_domains(self, batch_size=5, input_dim=3, output_dim=4, num_and=2):
        """Create mock domains object for bab stage."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        mock_domains.get_topk_indices = MagicMock(
            return_value=torch.tensor([0, 1, 2])  # Return first 3 indices
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=2, input_dim=3, output_dim=4, attack_success=False):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([attack_success]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_bab_stage_uses_worst_domains(self, mock_pgd_attack, mock_config):
        """Test that bab stage uses only worst k domains."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 3,
                'pgd_alpha': 'auto',
                'pgd_steps': 5,
                'pgd_restarts': 5,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size = 10
        mock_domains = self._create_mock_domains(batch_size)
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'bab', vnnlib)

        mock_domains.get_topk_indices.assert_called_once()
        call_kwargs = mock_domains.get_topk_indices.call_args[1]
        self.assertEqual(call_kwargs['k'], 3)  # max_num_domains

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_bab_stage_limits_domains_by_config(self, mock_pgd_attack, mock_config):
        """Test that bab stage limits domains to max_num_domains."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 5,
                'pgd_alpha': 'auto',
                'pgd_steps': 5,
                'pgd_restarts': 5,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size = 20
        mock_domains = self._create_mock_domains(batch_size)
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'bab', vnnlib)

        call_kwargs = mock_domains.get_topk_indices.call_args[1]
        self.assertEqual(call_kwargs['k'], 5)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_bab_stage_uses_min_of_max_domains_and_len(self, mock_pgd_attack, mock_config):
        """Test that bab stage uses min(max_num_domains, len(domains))."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 100,  # More than actual domains
                'pgd_alpha': 'auto',
                'pgd_steps': 5,
                'pgd_restarts': 5,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size = 5  # Less than max_num_domains
        mock_domains = self._create_mock_domains(batch_size)
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'bab', vnnlib)

        call_kwargs = mock_domains.get_topk_indices.call_args[1]
        self.assertEqual(call_kwargs['k'], 5)  # min(100, 5) = 5

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_bab_stage_uses_input_split_check_adv_config(self, mock_pgd_attack, mock_config):
        """Test that bab stage uses input_split_check_adv config."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 3,
                'pgd_alpha': '0.02',
                'pgd_steps': 10,
                'pgd_restarts': 8,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'bab', vnnlib)

        call_kwargs = mock_pgd_attack.call_args[1]
        self.assertEqual(call_kwargs['alpha'], 0.02)
        self.assertEqual(call_kwargs['pgd_steps'], 10)
        self.assertEqual(call_kwargs['num_restarts'], 8)


class TestPgdAttackOnDomainsInvalidStage(unittest.TestCase):
    """Tests for pgd_attack_on_domains with invalid stage."""

    def test_invalid_stage_raises_value_error(self):
        """Test that invalid stage raises ValueError."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_domains = MagicMock()
        model = MagicMock()
        vnnlib = MagicMock()

        with self.assertRaises(ValueError) as context:
            pgd_attack_on_domains(model, mock_domains, 'invalid_stage', vnnlib)

        self.assertIn('invalid_stage', str(context.exception))

    def test_empty_stage_raises_value_error(self):
        """Test that empty stage raises ValueError."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_domains = MagicMock()
        model = MagicMock()
        vnnlib = MagicMock()

        with self.assertRaises(ValueError):
            pgd_attack_on_domains(model, mock_domains, '', vnnlib)

    def test_none_stage_raises_value_error(self):
        """Test that None stage raises error."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_domains = MagicMock()
        model = MagicMock()
        vnnlib = MagicMock()

        with self.assertRaises((ValueError, TypeError)):
            pgd_attack_on_domains(model, mock_domains, None, vnnlib)


class TestPgdAttackOnDomainsAttackSuccess(unittest.TestCase):
    """Tests for pgd_attack_on_domains when attack succeeds."""

    def _create_mock_domains(self, batch_size=2, input_dim=3, output_dim=4, num_and=2):
        """Create mock domains object."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=2, input_dim=3, output_dim=4, attack_success=True):
        """Create mock PGDAttackResult with successful attack."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([attack_success]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    @patch('input_split.attack_in_input_split.check_and_save_cex')
    def test_attack_success_calls_check_and_save_cex(self, mock_check_cex, mock_pgd_attack, mock_config):
        """Test that successful attack calls check_and_save_cex."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result(attack_success=True)
        mock_check_cex.return_value = ('unsafe', True)

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        mock_check_cex.assert_called_once()

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    @patch('input_split.attack_in_input_split.check_and_save_cex')
    def test_attack_success_returns_verified_success(self, mock_check_cex, mock_pgd_attack, mock_config):
        """Test that successful attack returns verified_success from check_and_save_cex."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result(attack_success=True)
        mock_check_cex.return_value = ('unsafe', True)  # Returns (status, verified_success)

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertTrue(result)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    @patch('input_split.attack_in_input_split.check_and_save_cex')
    def test_attack_success_returns_false_when_verification_fails(self, mock_check_cex, mock_pgd_attack, mock_config):
        """Test that verified_success=False is returned when check_and_save_cex fails."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result(attack_success=True)
        mock_check_cex.return_value = ('unsafe', False)  # verification failed

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertFalse(result)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_attack_failure_returns_false(self, mock_pgd_attack, mock_config):
        """Test that failed attack returns False."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result(attack_success=False)

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertFalse(result)


class TestPgdAttackOnDomainsSameSpecSize(unittest.TestCase):
    """Tests for pgd_attack_on_domains when all domains have same spec size."""

    def _create_mock_domains_same_spec(self, batch_size=3, input_dim=4, output_dim=5, num_and=2):
        """Create mock domains with same spec size."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)  # All same

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=3, input_dim=4, output_dim=5):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_same_spec_size_reshapes_c_correctly(self, mock_pgd_attack, mock_config):
        """Test that c is reshaped correctly when spec_size is same."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim, output_dim, num_and = 3, 4, 5, 2
        mock_domains = self._create_mock_domains_same_spec(batch_size, input_dim, output_dim, num_and)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size, input_dim, output_dim)

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_args = mock_pgd_attack.call_args[0]
        c = call_args[4]  # c is 5th arg

        # c should be [1, num_and * num_domains, output_dim]
        self.assertEqual(c.shape[0], 1)
        self.assertEqual(c.shape[1], num_and * batch_size)
        self.assertEqual(c.shape[2], output_dim)


class TestPgdAttackOnDomainsDifferentSpecSize(unittest.TestCase):
    """Tests for pgd_attack_on_domains when domains have different spec sizes."""

    def _create_mock_domains_diff_spec(self, batch_size=3, input_dim=4, output_dim=5, max_num_and=3):
        """Create mock domains with different spec sizes."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, max_num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, max_num_and, output_dim)
        rhs = torch.zeros(batch_size, max_num_and)
        spec_size = torch.tensor([1, 2, 3], dtype=torch.int64)  # Different sizes

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=3, input_dim=4, output_dim=5):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, 6),  # sum of spec_sizes
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    @patch('input_split.attack_in_input_split.unpad_to_list_of_tensors')
    def test_different_spec_size_calls_unpad(self, mock_unpad, mock_pgd_attack, mock_config):
        """Test that unpad_to_list_of_tensors is called for different spec sizes."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim, output_dim = 3, 4, 5
        mock_domains = self._create_mock_domains_diff_spec(batch_size, input_dim, output_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size, input_dim, output_dim)

        # Mock unpad_to_list_of_tensors to return list of tensors
        mock_unpad.return_value = [torch.randn(1, 2, output_dim)]

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertTrue(mock_unpad.called)


class TestUpdateRhsWithAttackBasic(unittest.TestCase):
    """Basic tests for update_rhs_with_attack function."""

    def _create_mock_model(self, output_dim=5):
        """Create mock model."""
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, output_dim)
        return mock_model

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult with controlled output for valid gaps."""
        from attack import PGDAttackResult

        if adv_output is None:
            # Use positive outputs to ensure upper bounds are reasonable
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_returns_tensor(self, mock_pgd_attack, mock_config):
        """Test that update_rhs_with_attack returns a tensor."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 3, 4, 5, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        # Use identity-like cs to get predictable upper bounds
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)  # High thresholds
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)  # Very low lower bounds for safe gap

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = self._create_mock_model(output_dim)

        result = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        self.assertIsInstance(result, torch.Tensor)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_limits_domains_by_max_num_domains(self, mock_pgd_attack, mock_config):
        """Test that domains are limited by max_num_domains."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 2,  # Limit to 2
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 5, 4, 5, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(2, input_dim, output_dim)

        model = self._create_mock_model(output_dim)

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        # Check that pgd_attack was called with limited domains
        call_args = mock_pgd_attack.call_args[0]
        x = call_args[1]
        self.assertEqual(x.shape[1], 2)  # num_domains limited to 2

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_computes_center_correctly(self, mock_pgd_attack, mock_config):
        """Test that center x is computed as (x_L + x_U) / 2."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = self._create_mock_model(output_dim)

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        call_args = mock_pgd_attack.call_args[0]
        x = call_args[1]  # center x
        data_min = call_args[2]
        data_max = call_args[3]

        expected_center = (data_min + data_max) / 2
        self.assertTrue(torch.allclose(x, expected_center))


class TestUpdateRhsWithAttackRhsUpdate(unittest.TestCase):
    """Tests for RHS update logic in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_rhs_is_updated_with_min(self, mock_pgd_attack, mock_config):
        """Test that RHS is updated with min(rhs, upper_bound)."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)  # High thresholds
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)  # Very low dm_lb for safe gap

        # Create result with specific output values
        adv_output = torch.ones(1, num_domains, output_dim)  # Will give upper_bound = 1.0 per spec
        mock_result = self._create_mock_pgd_result(num_domains, input_dim, output_dim, adv_output)
        mock_pgd_attack.return_value = mock_result

        model = MagicMock()

        result = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        # Result should have same shape as input thresholds
        self.assertEqual(result.shape, thresholds.shape)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_modifies_thresholds_in_place(self, mock_pgd_attack, mock_config):
        """Test that thresholds tensor is modified in place."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        adv_output = torch.ones(1, num_domains, output_dim)
        mock_result = self._create_mock_pgd_result(num_domains, input_dim, output_dim, adv_output)
        mock_pgd_attack.return_value = mock_result

        model = MagicMock()
        original_thresholds_id = id(thresholds)

        result = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        # Result should be same object as input thresholds
        self.assertEqual(id(result), original_thresholds_id)


class TestUpdateRhsWithAttackAlphaComputation(unittest.TestCase):
    """Tests for alpha computation in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_alpha_computed_from_bounds(self, mock_pgd_attack, mock_config):
        """Test that alpha is computed as (data_max - data_min).max() / 8."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = 2.0 * torch.ones(num_domains, input_dim)  # Range is 2
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = MagicMock()

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        call_kwargs = mock_pgd_attack.call_args[1]
        expected_alpha = 2.0 / 8  # (x_U - x_L).max() / 8
        self.assertAlmostEqual(call_kwargs['alpha'].item(), expected_alpha, places=5)


class TestUpdateRhsWithAttackSpecSize(unittest.TestCase):
    """Tests for spec_size handling in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_spec_size_is_full_tensor(self, mock_pgd_attack, mock_config):
        """Test that spec_size is a full tensor with cs.shape[1] for each domain."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 3, 4, 5, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = MagicMock()

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        call_args = mock_pgd_attack.call_args[0]
        spec_size = call_args[6]

        self.assertEqual(spec_size.shape[0], num_domains)
        self.assertTrue((spec_size == num_spec).all())


class TestUpdateRhsWithAttackGapAssertion(unittest.TestCase):
    """Tests for gap assertion in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult with specific adv_output."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_valid_gap_does_not_raise(self, mock_pgd_attack, mock_config):
        """Test that valid gap (rhs >= dm_lb) does not raise."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)  # High thresholds
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)  # Very low lower bound

        # Create output that produces valid upper bounds (positive values)
        adv_output = torch.ones(1, num_domains, output_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(
            num_domains, input_dim, output_dim, adv_output
        )

        model = MagicMock()

        # Should not raise
        result = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)
        self.assertIsNotNone(result)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_negative_gap_raises_assertion(self, mock_pgd_attack, mock_config):
        """Test that negative gap (rhs < dm_lb - 1e-3) raises AssertionError."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = -10.0 * torch.ones(num_domains, num_spec)  # Very low thresholds
        dm_lb = 10.0 * torch.ones(num_domains, num_spec)  # High dm_lb (will cause negative gap)

        # Create output that produces low upper bounds (identity * negative = negative)
        adv_output = -1000.0 * torch.ones(1, num_domains, output_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(
            num_domains, input_dim, output_dim, adv_output
        )

        model = MagicMock()

        with self.assertRaises(AssertionError) as context:
            update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        self.assertIn('Gap', str(context.exception))


class TestUpdateRhsWithAttackPrint(unittest.TestCase):
    """Tests for print statements in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    @patch('builtins.print')
    def test_prints_domain_count(self, mock_print, mock_pgd_attack, mock_config):
        """Test that domain count is printed."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 5, 3, 4, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = MagicMock()

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        # Check that print was called with domain count
        print_calls = [str(call) for call in mock_print.call_args_list]
        domain_count_printed = any(f'{num_domains} domains' in str(call) for call in print_calls)
        self.assertTrue(domain_count_printed)


class TestUpdateRhsWithAttackMatrixOps(unittest.TestCase):
    """Tests for matrix operations in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, adv_output=None):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim)

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, num_domains, input_dim),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, num_domains * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_c_mat_shape(self, mock_pgd_attack, mock_config):
        """Test that C_mat has correct shape [1, num_domains * num_and, output_dim]."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 3, 4, 5, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = MagicMock()

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        call_args = mock_pgd_attack.call_args[0]
        C_mat = call_args[4]

        self.assertEqual(C_mat.shape[0], 1)
        self.assertEqual(C_mat.shape[1], num_domains * num_spec)
        self.assertEqual(C_mat.shape[2], output_dim)

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_rhs_mat_shape(self, mock_pgd_attack, mock_config):
        """Test that rhs_mat has correct shape [1, num_domains * num_and]."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 3, 4, 5, 2
        x_L = torch.zeros(num_domains, input_dim)
        x_U = torch.ones(num_domains, input_dim)
        cs = torch.eye(num_spec, output_dim).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim)

        model = MagicMock()

        update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        call_args = mock_pgd_attack.call_args[0]
        rhs_mat = call_args[5]

        self.assertEqual(rhs_mat.shape[0], 1)
        self.assertEqual(rhs_mat.shape[1], num_domains * num_spec)


class TestUpdateRhsWithAttackDeviceHandling(unittest.TestCase):
    """Tests for device handling in update_rhs_with_attack."""

    def _create_mock_pgd_result(self, num_domains=2, input_dim=3, output_dim=5, device='cpu', adv_output=None):
        """Create mock PGDAttackResult on specified device."""
        from attack import PGDAttackResult

        if adv_output is None:
            adv_output = torch.ones(1, num_domains, output_dim, device=device)

        return PGDAttackResult(
            attack_success=torch.tensor([False], device=device),
            best_or_idx=torch.tensor([0], device=device),
            adv_input_per_or=torch.rand(1, num_domains, input_dim, device=device),
            adv_output_per_or=adv_output,
            adv_margin_per_or=torch.rand(1, num_domains, device=device),
            adv_input_best=torch.rand(1, input_dim, device=device),
            adv_output_best=torch.rand(1, output_dim, device=device),
            adv_margin_best=torch.rand(1, device=device),
            adv_margin_per_spec=torch.rand(1, num_domains * 2, device=device),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_preserves_input_device(self, mock_pgd_attack, mock_config):
        """Test that output is on same device as input."""
        from input_split.attack_in_input_split import update_rhs_with_attack

        mock_config['attack'] = {
            'input_split_check_adv': {
                'max_num_domains': 10,
                'pgd_steps': 5,
            }
        }

        num_domains, input_dim, output_dim, num_spec = 2, 3, 4, 2
        device = 'cpu'
        x_L = torch.zeros(num_domains, input_dim, device=device)
        x_U = torch.ones(num_domains, input_dim, device=device)
        cs = torch.eye(num_spec, output_dim, device=device).unsqueeze(0).expand(num_domains, -1, -1).clone()
        thresholds = 10.0 * torch.ones(num_domains, num_spec, device=device)
        dm_lb = -100.0 * torch.ones(num_domains, num_spec, device=device)

        mock_pgd_attack.return_value = self._create_mock_pgd_result(num_domains, input_dim, output_dim, device)

        model = MagicMock()

        result = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model)

        self.assertEqual(result.device.type, device)


class TestPgdAttackOnDomainsDmUnsqueeze(unittest.TestCase):
    """Tests for dm_l/dm_u unsqueezing in pgd_attack_on_domains."""

    def _create_mock_domains(self, batch_size=2, input_dim=3, output_dim=4, num_and=2):
        """Create mock domains."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=2, input_dim=3, output_dim=4):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_dm_l_dm_u_are_unsqueezed(self, mock_pgd_attack, mock_config):
        """Test that dm_l and dm_u are unsqueezed to [1, num_domains, input_shape]."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        batch_size, input_dim = 3, 5
        mock_domains = self._create_mock_domains(batch_size, input_dim)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size, input_dim)

        model = MagicMock()
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_args = mock_pgd_attack.call_args[0]
        dm_l = call_args[2]
        dm_u = call_args[3]

        # Should be [1, num_domains, input_dim]
        self.assertEqual(dm_l.shape[0], 1)
        self.assertEqual(dm_l.shape[1], batch_size)
        self.assertEqual(dm_u.shape[0], 1)
        self.assertEqual(dm_u.shape[1], batch_size)


class TestPgdAttackOnDomainsModelPassing(unittest.TestCase):
    """Tests for model passing in pgd_attack_on_domains."""

    def _create_mock_domains(self, batch_size=2, input_dim=3, output_dim=4, num_and=2):
        """Create mock domains."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=2, input_dim=3, output_dim=4):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size * 2),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_model_passed_to_pgd_attack(self, mock_pgd_attack, mock_config):
        """Test that model is passed as first argument to pgd_attack_with_general_specs."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains()
        mock_pgd_attack.return_value = self._create_mock_pgd_result()

        model = MagicMock(name='test_model')
        vnnlib = MagicMock()

        pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        call_args = mock_pgd_attack.call_args[0]
        self.assertIs(call_args[0], model)


class TestPgdAttackOnDomainsEdgeCases(unittest.TestCase):
    """Edge case tests for pgd_attack_on_domains."""

    def _create_mock_domains(self, batch_size=1, input_dim=2, output_dim=3, num_and=1):
        """Create mock domains for edge cases."""
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=batch_size)

        lb = -torch.rand(batch_size, num_and)
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        c = torch.randn(batch_size, num_and, output_dim)
        rhs = torch.zeros(batch_size, num_and)
        spec_size = torch.full([batch_size], num_and, dtype=torch.int64)

        mock_domains.__getitem__ = MagicMock(
            return_value=(lb, dm_l, dm_u, c, rhs, spec_size)
        )
        mock_domains.get_topk_indices = MagicMock(return_value=torch.tensor([0]))
        return mock_domains

    def _create_mock_pgd_result(self, batch_size=1, input_dim=2, output_dim=3):
        """Create mock PGDAttackResult."""
        from attack import PGDAttackResult

        return PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.rand(1, batch_size, input_dim),
            adv_output_per_or=torch.rand(1, batch_size, output_dim),
            adv_margin_per_or=torch.rand(1, batch_size),
            adv_input_best=torch.rand(1, input_dim),
            adv_output_best=torch.rand(1, output_dim),
            adv_margin_best=torch.rand(1),
            adv_margin_per_spec=torch.rand(1, batch_size),
        )

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_single_domain(self, mock_pgd_attack, mock_config):
        """Test with single domain."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains(batch_size=1)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size=1)

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertFalse(result)
        mock_pgd_attack.assert_called_once()

    @patch('input_split.attack_in_input_split.arguments.Config', new_callable=dict)
    @patch('input_split.attack_in_input_split.pgd_attack_with_general_specs')
    def test_single_spec(self, mock_pgd_attack, mock_config):
        """Test with single AND specification."""
        from input_split.attack_in_input_split import pgd_attack_on_domains

        mock_config['attack'] = {
            'input_split': {
                'pgd_alpha': 'auto',
                'pgd_steps': 100,
                'pgd_restarts': 30,
            },
            'cex_path': '/tmp/cex'
        }

        mock_domains = self._create_mock_domains(batch_size=2, num_and=1)
        mock_pgd_attack.return_value = self._create_mock_pgd_result(batch_size=2)

        model = MagicMock()
        vnnlib = MagicMock()

        result = pgd_attack_on_domains(model, mock_domains, 'init', vnnlib)

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
