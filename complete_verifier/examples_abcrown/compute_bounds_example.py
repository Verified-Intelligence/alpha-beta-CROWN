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
"""Compute bounds demo with a moderately sized graph and multiple objectives."""

from __future__ import annotations

import torch
import torch.nn as nn

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    IOConstraints,
    input_vars,
    output_vars,
)


class ResidualReachabilityGraph(nn.Module):
    """A small residual MLP used as a computation graph for API demonstration."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 32, out_dim: int = 5) -> None:
        super().__init__()
        self.block1 = nn.Linear(in_dim, hidden_dim)
        self.block2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(7)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.6)
                nn.init.uniform_(module.bias, -0.15, 0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.block1(x))
        h2 = torch.relu(self.block2(h1))
        # Residual combination keeps the graph non-trivial while staying verifier-friendly.
        return self.head(h1 + 0.25 * h2)


def main() -> None:
    model = ResidualReachabilityGraph()
    x = input_vars(6)
    y = output_vars(5)

    input_constraint = (
        (x >= [-1.2, -0.8, -1.0, -0.6, -1.1, -0.9]) &
        (x <= [1.1, 0.9, 0.7, 1.0, 0.8, 1.2])
    )

    constraints = IOConstraints(
        input_vars=x,
        input_constraint=input_constraint,
    )

    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(model, x, y, config=cfg)

    objectives = [
        y[0],
        y[1] - y[3],
        0.75 * y[2] + 0.2 * y[4],
        -0.4 * y[0] + y[3] - 0.5 * y[4],
        y[0] + y[1] + y[2] - y[4],
    ]
    bounds_result = solver.compute_bounds(
        constraints=constraints,
        objective=objectives,
        return_linear_bounds=True,
    )

    print("[compute_bounds] lower:", bounds_result.lower)
    print("[compute_bounds] upper:", bounds_result.upper)
    print("[compute_bounds] linear bounds:", bounds_result.linear_bounds)


if __name__ == "__main__":
    main()
