"""Tests for interval-sensitivity branching."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from heuristics.interval_sensitivity import IntervalSensitivityBranching


class TestIntervalSensitivity(unittest.TestCase):

    def setUp(self):
        activation = SimpleNamespace(name="relu1")
        node = SimpleNamespace(name="layer1")

        self.net = SimpleNamespace(
            split_nodes=[node],
            split_activations={"layer1": [(activation, None)]},
            nonlinear_split=False,
        )

        self.config_patch = patch(
            "arguments.Config.all_args",
            {"bab": {"branching": {"decision_sub_batch_size": None}}},
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.heuristic = IntervalSensitivityBranching(self.net)

    def domains(self, lower, upper, coefficients, mask):
        return {
            "lower_bounds": {"layer1": torch.tensor([lower])},
            "upper_bounds": {"layer1": torch.tensor([upper])},
            "mask": {"layer1": torch.tensor([mask])},
            "lAs": {"relu1": torch.tensor([coefficients])},
        }

    def test_uncertainty_and_sensitivity(self):
        domains = self.domains(
            [-1., -2.],
            [1., 2.],
            [[1., 3.]],
            [1., 1.],
        )

        scores = self.heuristic.compute_neuron_scores(domains)
        torch.testing.assert_close(
            scores["layer1"],
            torch.tensor([[0.5, 3.0]]),
        )

    def test_stable_neurons_have_zero_score(self):
        domains = self.domains(
            [1., -2., -1.],
            [2., -1., 1.],
            [[5., 5., 5.]],
            [0., 0., 1.],
        )

        scores = self.heuristic.compute_neuron_scores(domains)
        torch.testing.assert_close(
            scores["layer1"],
            torch.tensor([[0., 0., 2.5]]),
        )

    def test_multiple_specifications(self):
        domains = self.domains(
            [-1., -1.],
            [1., 1.],
            [[2., 4.], [4., 8.]],
            [1., 1.],
        )

        scores = self.heuristic.compute_neuron_scores(domains)
        torch.testing.assert_close(
            scores["layer1"],
            torch.tensor([[1.5, 3.0]]),
        )

    def test_mask_excludes_neuron(self):
        domains = self.domains(
            [-1., -1.],
            [1., 1.],
            [[100., 1.]],
            [0., 1.],
        )

        scores = self.heuristic.compute_neuron_scores(domains)
        self.assertEqual(scores["layer1"][0, 0].item(), 0.)
        self.assertGreater(scores["layer1"][0, 1].item(), 0.)

    def test_branching_selects_highest_scoring_neuron(self):
        domains = self.domains(
            [-1., -2., -1.],
            [1., 2., 1.],
            [[1., 4., 2.], [1., 6., 2.]],
            [1., 1., 1.],
        )
        result = self.heuristic.compute_branching_decisions(
            domains, split_depth=1
        )
        self.assertEqual(result.branching_decision, [[0, 1]])
        self.assertEqual(result.split_depth, 1)
        self.assertEqual(result.batch_size, 1)

    def test_factory_registration(self):
        from heuristics.branching_heuristics import (
            get_branching_heuristic,
        )

        with patch(
            "heuristics.branching_heuristics.resolve_branching_method",
            return_value="interval-sensitivity",
        ):
            obj = get_branching_heuristic(self.net)

        self.assertIsInstance(obj, IntervalSensitivityBranching)


if __name__ == "__main__":
    unittest.main()
