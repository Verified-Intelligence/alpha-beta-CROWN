#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Small API compatibility smoke tests for ABCrownSolver."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Sequence

import torch

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    VerificationSpec,
    input_vars,
    output_vars,
)

class ConstantPositive(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        return self.linear(x_flat)


class FixedLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor([2.0, 0.0, -1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        return self.linear(x_flat[:, :1])


class SumSquares(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return (x ** 2).sum(dim=1, keepdim=True)


class MultiOutputAffines(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        y0 = x[:, 0] + 0.5 * x[:, 1]
        y1 = -0.3 * x[:, 0] + x[:, 1]
        y2 = x.sum(dim=1)
        return torch.stack([y0, y1, y2], dim=1)


def test_single_input_constant() -> None:
    torch.manual_seed(0)
    model = ConstantPositive()
    x = input_vars(1)
    y = output_vars(1)
    input_constraint = (x[0] == 0.0)
    output_constraint = y[0] > 0.0
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[single_input] status=", result.status, "success=", result.success)


def test_vectorized_classification() -> None:
    torch.manual_seed(0)
    model = FixedLogits()
    x = input_vars(28 * 28)
    y = output_vars(3)
    eps = 0.1
    input_constraint = (x >= -eps) & (x <= eps)
    output_constraint = (y[0] > y[1]) & (y[0] > y[2])
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    cfg = ConfigBuilder.from_defaults().set(
        attack__pgd_order="skip",
        general__complete_verifier="skip",
        general__enable_incomplete_verification=True,
        general__enable_complete_verification=False,
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[vectorized] status=", result.status, "success=", result.success)


def test_vectorized_expression_spec() -> None:
    torch.manual_seed(0)
    model = FixedLogits()
    x = input_vars(28 * 28)
    y = output_vars(3)
    eps = 0.1
    input_constraint = (x >= -eps) & (x <= eps)
    output_constraint = (y[0] > y[1]) & (y[0] > y[2])
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    cfg = ConfigBuilder.from_defaults().set(
        attack__pgd_order="skip",
        general__complete_verifier="skip",
        general__enable_incomplete_verification=True,
        general__enable_complete_verification=False,
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[vectorized_expression] status=", result.status, "success=", result.success)


def test_expression_spec() -> None:
    torch.manual_seed(0)
    x = input_vars(2)
    y = output_vars(1)
    input_constraint = (
        (x[0] >= -0.1) & (x[0] <= 0.1) & (x[1] >= -0.1) & (x[1] <= 0.1)
    )
    output_constraint = y[0] > 0.0
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    model = SumSquares()
    cfg = ConfigBuilder.from_defaults().set(
        attack__pgd_order="skip",
        general__complete_verifier="skip",
        general__enable_incomplete_verification=True,
        general__enable_complete_verification=False,
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[expression] status=", result.status, "success=", result.success)


def test_clause_or_spec() -> None:
    torch.manual_seed(0)
    model = FixedLogits()
    x = input_vars(28 * 28)
    y = output_vars(3)
    eps = 0.1
    input_constraint = (x >= -eps) & (x <= eps)
    case_a = (y[0] > y[1]) & (y[0] > y[2])
    case_b = (y[1] > y[0]) & (y[1] > y[2])
    output_constraint = case_a | case_b
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[clauses_or] status=", result.status, "success=", result.success)


def test_mixed_and_or_spec() -> None:
    """Mixed AND/OR expression spec to exercise DNF expansion."""
    torch.manual_seed(0)
    model = MultiOutputAffines()
    x = input_vars(2)
    y = output_vars(3)

    # Input: simple conjunction of per-dim boxes
    input_constraint = (
        (x[0] >= -0.1) & (x[0] <= 0.1) &
        (x[1] >= -0.2) & (x[1] <= 0.2)
    )

    # Output: (y0 > 0 AND y1 < 0.2) AND ( (y2 > 0.1) OR (y0 - y1 > -0.1) )
    base_pred = (y[0] > 0.0) & (y[1] < 0.2)
    branch_a = y[2] > 0.1
    branch_b = (y[0] - y[1]) > -0.1
    output_constraint = base_pred & (branch_a | branch_b)

    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[mixed_and_or] status=", result.status, "success=", result.success)


def test_config_update() -> None:
    torch.manual_seed(0)
    model = ConstantPositive()
    x = input_vars(1)
    y = output_vars(1)
    input_constraint = (x[0] >= 0.0) & (x[0] <= 0.0)
    output_constraint = y[0] > 0.0
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    cfg = (
        ConfigBuilder.from_defaults()
        .set(
            attack__pgd_order="skip",
            general__complete_verifier="skip",
            general__enable_incomplete_verification=True,
            general__enable_complete_verification=False,
        )
        .update({"solver": {"batch_size": 4}})
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.solve()
    print("[config_update] status=", result.status, "success=", result.success)


TESTS: Dict[str, Callable[[], None]] = {
    "single": test_single_input_constant,
    "vectorized": test_vectorized_classification,
    "vectorized_expression": test_vectorized_expression_spec,
    "expression": test_expression_spec,
    "clauses_or": test_clause_or_spec,
    "config_update": test_config_update,
    "mixed_and_or": test_mixed_and_or_spec,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="API smoke tests.")
    parser.add_argument(
        "--tests",
        type=str,
        default=",".join(TESTS.keys()),
        help=f"Comma-separated list of tests to run. Available: {', '.join(TESTS.keys())}",
    )
    args = parser.parse_args(argv)
    to_run = [name.strip() for name in args.tests.split(",") if name.strip()]
    for name in to_run:
        fn = TESTS.get(name)
        if fn is None:
            print(f"[warn] unknown test '{name}', skipping.")
            continue
        fn()


if __name__ == "__main__":
    main()
