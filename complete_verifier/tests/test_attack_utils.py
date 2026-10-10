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
"""Unit tests for attack/attack_utils.py"""
import os
import sys
import unittest

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAttackStats(unittest.TestCase):
    """Tests for Stats class in attack_utils."""

    def test_stats_init(self):
        """Test Stats initialization."""
        from attack.attack_utils import Stats
        stats = Stats()
        self.assertEqual(stats.num_restarts, 0)
        self.assertEqual(stats.num_steps, 0)

    def test_stats_reset(self):
        """Test Stats reset."""
        from attack.attack_utils import Stats
        stats = Stats()
        stats.num_restarts = 10
        stats.num_steps = 100
        stats.reset()
        self.assertEqual(stats.num_restarts, 0)
        self.assertEqual(stats.num_steps, 0)

    def test_stats_report(self):
        """Test Stats report."""
        from attack.attack_utils import Stats
        stats = Stats()
        stats.num_restarts = 5
        stats.num_steps = 50
        report = stats.report()
        self.assertEqual(report['num_restarts'], 5)
        self.assertEqual(report['num_steps'], 50)

    def test_stats_accumulate(self):
        """Test Stats accumulate."""
        from attack.attack_utils import Stats
        stats = Stats()
        stats.accumulate(3, 30)
        self.assertEqual(stats.num_restarts, 3)
        self.assertEqual(stats.num_steps, 30)
        stats.accumulate(2, 20)
        self.assertEqual(stats.num_restarts, 5)
        self.assertEqual(stats.num_steps, 50)


class TestPGDAttackResult(unittest.TestCase):
    """Tests for PGDAttackResult dataclass."""

    def test_pgd_attack_result_creation(self):
        """Test PGDAttackResult can be created."""
        from attack.attack_utils import PGDAttackResult

        attack_success = torch.tensor([True, False])
        best_or_idx = torch.tensor([0, 1])
        adv_input_per_or = torch.randn(2, 2, 3, 32, 32)
        adv_output_per_or = torch.randn(2, 2, 10)
        adv_margin_per_or = torch.randn(2, 2)
        adv_input_best = torch.randn(2, 3, 32, 32)
        adv_output_best = torch.randn(2, 10)
        adv_margin_best = torch.randn(2)
        adv_margin_per_spec = torch.randn(2, 4)

        result = PGDAttackResult(
            attack_success=attack_success,
            best_or_idx=best_or_idx,
            adv_input_per_or=adv_input_per_or,
            adv_output_per_or=adv_output_per_or,
            adv_margin_per_or=adv_margin_per_or,
            adv_input_best=adv_input_best,
            adv_output_best=adv_output_best,
            adv_margin_best=adv_margin_best,
            adv_margin_per_spec=adv_margin_per_spec
        )

        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result.attack_success, attack_success))
        self.assertTrue(torch.equal(result.best_or_idx, best_or_idx))
        self.assertTrue(torch.equal(result.adv_input_per_or, adv_input_per_or))
        self.assertTrue(torch.equal(result.adv_output_per_or, adv_output_per_or))
        self.assertTrue(torch.equal(result.adv_margin_per_or, adv_margin_per_or))
        self.assertTrue(torch.equal(result.adv_input_best, adv_input_best))
        self.assertTrue(torch.equal(result.adv_output_best, adv_output_best))
        self.assertTrue(torch.equal(result.adv_margin_best, adv_margin_best))
        self.assertTrue(torch.equal(result.adv_margin_per_spec, adv_margin_per_spec))
        self.assertIsNone(result.adv_input_all)

    def test_pgd_attack_result_optional_field(self):
        """Test PGDAttackResult with optional adv_input_all."""
        from attack.attack_utils import PGDAttackResult

        result = PGDAttackResult(
            attack_success=torch.tensor([True]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.randn(1, 1, 3, 32, 32),
            adv_output_per_or=torch.randn(1, 1, 10),
            adv_margin_per_or=torch.randn(1, 1),
            adv_input_best=torch.randn(1, 3, 32, 32),
            adv_output_best=torch.randn(1, 10),
            adv_margin_best=torch.randn(1),
            adv_margin_per_spec=torch.randn(1, 2),
            adv_input_all=torch.randn(1, 5, 1, 3, 32, 32)
        )

        self.assertIsNotNone(result.adv_input_all)


if __name__ == '__main__':
    unittest.main()
