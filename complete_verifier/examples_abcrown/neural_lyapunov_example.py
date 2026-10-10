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
"""Verify the tutorial Lyapunov computation graph directly with the new API."""

from __future__ import annotations

import os

import torch

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    IOConstraints,
    input_vars,
    output_vars,
)
from neural_lyapunov_dependency.computation_graph import (
    Controller,
    Lyapunov,
    VanDerPolDynamics,
    LyapunovComputationGraph,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dynamics = VanDerPolDynamics()
    controller = Controller(
        dims=[2, 10, 10, 1],
        x_equilibrium=dynamics.x_equilibrium,
        u_equilibrium=dynamics.u_equilibrium,
        scale=1.0,
    )
    lyapunov = Lyapunov(dims=[2, 40, 40, 1])
    model = LyapunovComputationGraph(dynamics, controller, lyapunov)
    figure_dir = os.path.join(os.path.dirname(__file__), "neural_lyapunov_dependency")
    ckpt_path = os.path.join(figure_dir, "seed_0.pth")
    v_min = 0.0106
    v_max = 0.989
    # Match the small numerical slack used by the Jacobian verifier near V_dot = 0.
    v_dot_tolerance = 3e-6
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    x = input_vars(2)
    y = output_vars(2)  # y[0] = V(x), y[1] = V_dot
    input_constraint = (x >= [-4.8, -10.8]) & (x <= [4.8, 10.8])
    output_constraint = (y[0] < v_min) | (y[0] > v_max) | (y[1] < v_dot_tolerance)
    constraints = IOConstraints(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    cfg = ConfigBuilder.from_defaults()
    cfg = (
        cfg.set("attack/pgd_order", "before")
        .set("attack/pgd_steps", 20)
        .set("attack/pgd_restarts", 10)
    )
    solver = ABCrownSolver(model, x, y, config=cfg)
    result = solver.verify(constraints=constraints)

    print("[info] verifying Lyapunov tutorial graph with ABCrown API")
    print(f"status={result.status}, success={result.success}")


if __name__ == "__main__":
    main()
