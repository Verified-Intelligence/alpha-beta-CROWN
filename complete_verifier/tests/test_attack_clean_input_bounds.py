"""Regression tests for clean-input checking in the PGD attack interface."""

import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.attack_interface import attack_with_general_specs


class TestCleanInputBounds(unittest.TestCase):
    def test_clean_counterexample_preserves_input_inside_bounds(self):
        # The clean input is a valid counterexample. It must not be replaced
        # by a bound during the initial adversarial-example finalization.
        x = torch.tensor([[0.4]])
        data_min = torch.zeros_like(x)
        data_max = torch.ones_like(x)
        c_mat = torch.ones((1, 1))
        rhs_mat = torch.tensor([0.6])
        or_spec_size = torch.tensor([1])

        config = {
            "specification": {"norm": float("inf")},
            "bab": {"attack": {"enabled": False}},
            "attack": {
                "general_attack": True,
                "pgd_alpha": "auto",
                "pgd_alpha_scale": False,
                "pgd_restarts": 1,
                "pgd_batch_size": 1,
                "pgd_steps": 1,
                "adv_example_finalizer": "default_adv_example_finalizer",
                "attack_tolerance": 0.0,
            },
        }

        with (
            patch("attack.attack_interface.arguments.Config", config),
            patch(
                "attack.attack_interface.invalid_attack_success_mask",
                return_value=torch.zeros((1, 1), dtype=torch.bool),
            ),
            patch(
                "attack.general_spec_attack.pgd_attack_with_general_specs",
                side_effect=AssertionError("PGD should not run for a clean counterexample"),
            ),
        ):
            success, _, adv_input, adv_output, _, _ = attack_with_general_specs(
                torch.nn.Identity(),
                x,
                data_min,
                data_max,
                c_mat,
                rhs_mat,
                or_spec_size,
            )

        self.assertTrue(success.all().item())
        torch.testing.assert_close(adv_input, x)
        torch.testing.assert_close(adv_output, x)


if __name__ == "__main__":
    unittest.main()
