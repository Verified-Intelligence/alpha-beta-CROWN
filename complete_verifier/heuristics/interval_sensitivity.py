"""Interval-sensitivity branching for ReLU verification."""

import torch

from heuristics.base import NeuronBranchingHeuristic


class IntervalSensitivityBranching(NeuronBranchingHeuristic):
    """Rank ReLU splits by relaxation uncertainty and output sensitivity."""

    @torch.no_grad()
    def compute_neuron_scores(self, domains, **kwargs):
        scores = {}

        for name, lower in self.layer_iterator(domains["lower_bounds"]):
            upper = domains["upper_bounds"][name]
            mask = domains["mask"][name]

            activation = self.net.split_activations[name][0][0]
            coefficients = domains["lAs"][activation.name]

            # ReLU relaxation intercept for unstable neurons.
            uncertainty = (
                (-lower).clamp_min(0) * upper.clamp_min(0)
                / (upper - lower).clamp_min(1e-12)
            )

            # Average absolute sensitivity across specifications.
            sensitivity = coefficients.abs().mean(dim=1)

            score = uncertainty.flatten(1) * sensitivity.flatten(1)
            score = torch.nan_to_num(
                score, nan=0.0, posinf=0.0, neginf=0.0
            )

            scores[name] = score * mask.flatten(1)

        return scores
