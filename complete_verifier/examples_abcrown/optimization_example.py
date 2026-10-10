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
"""Optimization demo for ABCrownSolver.minimize/maximize (moderate complexity)."""

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


class ProcessControlGraph(nn.Module):
    """A small nonlinear graph for optimization examples."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 24, out_dim: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(23)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                nn.init.uniform_(module.bias, -0.12, 0.12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(0.6 * h1 + 0.4 * h2)


def main() -> None:
    # 1) Build model and symbolic variables.
    model = ProcessControlGraph()
    u = input_vars(6)
    y = output_vars(3)  # y[0]=energy cost, y[1]=quality penalty, y[2]=throughput score

    # 2) Input box constraints.
    input_constraint = (
        (u >= [-1.4, -1.2, -1.0, -0.8, -1.1, -0.9]) &
        (u <= [1.2, 1.0, 1.1, 0.9, 1.3, 1.0])
    )
    constraints = IOConstraints(
        input_vars=u,
        input_constraint=input_constraint,
    )

    # 3) Configure optimizer backends.
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(model, u, y, config=cfg)

    # 4) Minimize a weighted operating objective.
    total_operating_cost = y[0] + 0.35 * y[1] - 0.25 * y[2]
    min_result = solver.minimize(
        objective=total_operating_cost,
        constraints=constraints,
    )

    # 5) Maximize a productivity objective under the same constraints.
    productivity_score = y[2] - 0.2 * y[1]
    max_result = solver.maximize(
        objective=productivity_score,
        constraints=constraints,
    )

    print("[minimize] primal:", min_result.primal_value)
    print("[minimize] best:", min_result.x_best)
    print()
    print("[maximize] primal:", max_result.primal_value)
    print("[maximize] best:", max_result.x_best)


if __name__ == "__main__":
    main()
