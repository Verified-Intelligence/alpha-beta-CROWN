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
"""Small API compatibility smoke tests for ABCrownSolver."""

from __future__ import annotations

import argparse
import traceback
from typing import Callable, Dict, Sequence

import torch

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    IOConstraints,
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
    result = solver.verify()
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
    cfg = (
        ConfigBuilder.from_defaults()
        .set("attack/pgd_order", "skip")
        .set("general/complete_verifier", "skip")
        .set("general/enable_complete_verification", False)
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.verify()
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
    cfg = (
        ConfigBuilder.from_defaults()
        .set("attack/pgd_order", "skip")
        .set("general/complete_verifier", "skip")
        .set("general/enable_complete_verification", False)
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.verify()
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
    cfg = (
        ConfigBuilder.from_defaults()
        .set("attack/pgd_order", "skip")
        .set("general/complete_verifier", "skip")
        .set("general/enable_complete_verification", False)
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.verify()
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
    result = solver.verify()
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
    result = solver.verify()
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
        .set("attack/pgd_order", "skip")
        .set("general/complete_verifier", "skip")
        .set("general/enable_complete_verification", False)
        .update({"solver": {"batch_size": 4}})
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    result = solver.verify()
    print("[config_update] status=", result.status, "success=", result.success)


def test_compute_bounds_linear_bounds() -> None:
    """Unit-style smoke test for compute_bounds with vector objectives and linear-bound output.

    This verifies that compute_bounds accepts IOConstraints without output constraints,
    computes batched lower/upper bounds for multiple objective expressions, enables the
    API's internal sort-domain default, and returns the structured affine relaxation
    object on the supported default verifier path when return_linear_bounds=True.
    """
    torch.manual_seed(0)
    model = MultiOutputAffines()
    x = input_vars(2)
    y = output_vars(3)
    constraints = IOConstraints(
        input_vars=x,
        input_constraint=(x >= [-0.1, -0.2]) & (x <= [0.1, 0.2]),
    )
    cfg = ConfigBuilder.from_defaults()
    solver = ABCrownSolver(model, x, y, config=cfg)
    result = solver.compute_bounds(
        constraints=constraints,
        objective=[y[0], y[1] - y[2]],
        return_linear_bounds=True,
    )
    assert result.lower.shape == result.upper.shape == torch.Size([2])
    assert result.linear_bounds is not None
    assert set(result.linear_bounds.keys()) == {
        "lower_A", "lower_bias", "upper_A", "upper_bias", "subdomains"
    }
    assert result.linear_bounds["lower_A"].shape == torch.Size([2, 2])
    assert result.linear_bounds["upper_A"].shape == torch.Size([2, 2])
    assert result.linear_bounds["lower_bias"].shape == torch.Size([2])
    assert result.linear_bounds["upper_bias"].shape == torch.Size([2])
    assert torch.allclose(
        result.linear_bounds["lower_bias"],
        torch.zeros_like(result.linear_bounds["lower_bias"]),
        atol=1e-6,
    )
    assert torch.allclose(
        result.linear_bounds["upper_bias"],
        torch.zeros_like(result.linear_bounds["upper_bias"]),
        atol=1e-6,
    )
    assert result.stats["sort_domain_interval"] == 1
    print("[compute_bounds] lower=", result.lower, "upper=", result.upper)
    print("[compute_bounds] summary= bounds shape [2], linear_bounds keys= lower_A/lower_bias/upper_A/upper_bias/subdomains")


class _ResidualReachabilityGraph(torch.nn.Module):
    """Small residual MLP matching ``compute_bounds_example.py``."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 32, out_dim: int = 5) -> None:
        super().__init__()
        self.block1 = torch.nn.Linear(in_dim, hidden_dim)
        self.block2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(7)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.6)
                torch.nn.init.uniform_(module.bias, -0.15, 0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.block1(x))
        h2 = torch.relu(self.block2(h1))
        return self.head(h1 + 0.25 * h2)


def _objective_coefficient_rows() -> torch.Tensor:
    """Coefficients of the 5 objectives used by the soundness test, over y[0..4]."""
    rows = torch.zeros((5, 5), dtype=torch.float32)
    rows[0, 0] = 1.0
    rows[1, 1] = 1.0
    rows[1, 3] = -1.0
    rows[2, 2] = 0.75
    rows[2, 4] = 0.20
    rows[3, 0] = -0.4
    rows[3, 3] = 1.0
    rows[3, 4] = -0.5
    rows[4, 0] = 1.0
    rows[4, 1] = 1.0
    rows[4, 2] = 1.0
    rows[4, 4] = -1.0
    return rows


_OBJECTIVE_NAMES = [
    "y[0]",
    "y[1] - y[3]",
    "0.75*y[2] + 0.2*y[4]",
    "-0.4*y[0] + y[3] - 0.5*y[4]",
    "y[0] + y[1] + y[2] - y[4]",
]

_SOUNDNESS_INPUT_LOWER = [-1.2, -0.8, -1.0, -0.6, -1.1, -0.9]
_SOUNDNESS_INPUT_UPPER = [1.1, 0.9, 0.7, 1.0, 0.8, 1.2]


def _sampled_inner_envelope(
    model: torch.nn.Module,
    num_random: int = 20_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(min, max)`` of objective values seen by sampling.

    Combines uniform Monte Carlo over the input box and *all* 2^d corner
    points. The result is contained inside the true reachable set, so any
    sound bound must contain it.
    """
    import itertools

    lower = torch.tensor(_SOUNDNESS_INPUT_LOWER, dtype=torch.float32)
    upper = torch.tensor(_SOUNDNESS_INPUT_UPPER, dtype=torch.float32)
    rows = _objective_coefficient_rows()

    model.eval()
    gen = torch.Generator().manual_seed(123)
    x_rand = torch.rand(num_random, lower.numel(), generator=gen) * (upper - lower) + lower
    corner_iter = itertools.product(
        *[(float(lower[i]), float(upper[i])) for i in range(lower.numel())]
    )
    x_corners = torch.tensor(list(corner_iter), dtype=torch.float32)
    x_all = torch.cat([x_rand, x_corners], dim=0)
    with torch.no_grad():
        y = model(x_all)
    obj_vals = y @ rows.T  # shape (N, 5)
    return obj_vals.min(dim=0).values, obj_vals.max(dim=0).values


def test_compute_bounds_sound_multi_objective() -> None:
    """Regression: ``compute_bounds`` must return SOUND per-objective bounds.

    Historically, the API packed several objectives into a single AND
    clause and returned UNSOUND (too-tight) bounds: actual model outputs at
    some inputs in the region landed OUTSIDE the reported ``[lb, ub]``.
    The current implementation lays each objective out as its own input
    batch and uses an ``api``-side monkey-patch around
    ``input_bab_parallel`` (``ABCrownSolver._capture_per_objective_lb``) to
    extract per-objective BaB-tightened ``lb-rhs`` from the single
    parallel BaB run. This test guards both invariants:

    * **Sound**: for every objective i, ``lb_i <= min_x obj_i(x)`` and
      ``ub_i >= max_x obj_i(x)`` against a Monte Carlo + box-corner
      sample (an inner envelope of the reachable set).
    * **Multi-objective shape**: ``BoundsResult.lower/upper`` is a length-5
      1-D tensor when 5 independent objectives are submitted.

    Any future change that re-introduces AND-coupling of objectives, drops
    the per-objective capture context, or otherwise lets the BaB scalar
    worst-domain lb leak into the per-objective vector will trip this test.
    """
    torch.manual_seed(0)
    model = _ResidualReachabilityGraph()
    obs_min, obs_max = _sampled_inner_envelope(model)

    x = input_vars(6)
    y = output_vars(5)
    constraints = IOConstraints(
        input_vars=x,
        input_constraint=(x >= _SOUNDNESS_INPUT_LOWER) & (x <= _SOUNDNESS_INPUT_UPPER),
    )

    # A short BaB budget keeps the smoke test fast. Soundness must hold
    # regardless of BaB depth: at 0 iterations bounds are just incomplete
    # CROWN (loose but sound); more iterations tighten while staying sound.
    cfg = (
        ConfigBuilder.from_defaults()
        .set("bab/timeout", 5.0)
        .set("bab/override_timeout", 5.0)
    )
    solver = ABCrownSolver(model, x, y, config=cfg)

    objectives = [
        y[0],
        y[1] - y[3],
        0.75 * y[2] + 0.2 * y[4],
        -0.4 * y[0] + y[3] - 0.5 * y[4],
        y[0] + y[1] + y[2] - y[4],
    ]
    result = solver.compute_bounds(
        constraints=constraints,
        objective=objectives,
        return_linear_bounds=False,
    )

    assert result.lower.shape == torch.Size([5]), (
        f"Expected lower of shape [5], got {tuple(result.lower.shape)}"
    )
    assert result.upper.shape == torch.Size([5]), (
        f"Expected upper of shape [5], got {tuple(result.upper.shape)}"
    )

    tol = 1e-5
    failures: list[str] = []
    for i, name in enumerate(_OBJECTIVE_NAMES):
        lb_i = float(result.lower[i])
        ub_i = float(result.upper[i])
        obs_lo = float(obs_min[i])
        obs_hi = float(obs_max[i])
        if not (lb_i <= ub_i + tol):
            failures.append(
                f"  objective {i} ({name}): degenerate, lb={lb_i:+.6f} > ub={ub_i:+.6f}"
            )
        if lb_i > obs_lo + tol:
            failures.append(
                f"  objective {i} ({name}): UNSOUND lower, "
                f"lb={lb_i:+.6f} > sampled min={obs_lo:+.6f}"
            )
        if ub_i < obs_hi - tol:
            failures.append(
                f"  objective {i} ({name}): UNSOUND upper, "
                f"ub={ub_i:+.6f} < sampled max={obs_hi:+.6f}"
            )
        print(
            f"[compute_bounds_sound] obj {i} ({name}): "
            f"reported=[{lb_i:+.6f}, {ub_i:+.6f}] "
            f"sampled=[{obs_lo:+.6f}, {obs_hi:+.6f}]"
        )

    if failures:
        raise AssertionError(
            "compute_bounds produced unsound bounds against sampled inner envelope:\n"
            + "\n".join(failures)
        )
    print("[compute_bounds_sound] all 5 objectives are SOUND against sampling envelope.")


def test_optimization_history() -> None:
    """Smoke-test the returned primal/dual trace from minimize(..., return_bound_history=True)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 1))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1.0, 0.5]]))
        model[1].bias.zero_()
    x = input_vars(2)
    y = output_vars(1)
    constraints = IOConstraints(
        input_vars=x,
        input_constraint=(x >= [-1.0, -1.0]) & (x <= [1.0, 1.0]),
    )
    cfg = (
        ConfigBuilder.from_defaults()
        .set("general/device", "cpu")
        .set("attack/input_split_check_adv/pgd_steps", 20)
        .set("attack/input_split_check_adv/max_num_domains", 16)
        .set("bab/timeout", 0.5)
        .set("bab/override_timeout", 0.5)
        .set("bab/max_iterations", 4)
    )
    solver = ABCrownSolver(model, x, y, config=cfg)
    result = solver.minimize(
        objective=y[0],
        constraints=constraints,
        return_bound_history=True,
    )

    assert result.success, f"Expected success=True, got status={result.status}"
    assert result.bound_history is not None, "Expected non-empty bound_history"
    primal_values = result.bound_history["primal_values"]
    dual_bounds = result.bound_history["dual_bounds"]
    assert len(primal_values) > 0, "Expected at least one primal history entry"
    assert len(primal_values) == len(dual_bounds), "Primal and dual histories must align"
    assert all(
        primal_values[i] <= primal_values[i - 1] + 1e-6
        for i in range(1, len(primal_values))
    ), f"Primal history must be non-increasing, got {primal_values}"
    assert all(
        dual_bounds[i] + 1e-6 >= dual_bounds[i - 1]
        for i in range(1, len(dual_bounds))
    ), f"Dual history must be non-decreasing, got {dual_bounds}"
    print(
        "[optimization_history] primal_values=",
        primal_values,
        "dual_bounds=",
        dual_bounds,
    )


TESTS: Dict[str, Callable[[], None]] = {
    "single": test_single_input_constant,
    "vectorized": test_vectorized_classification,
    "vectorized_expression": test_vectorized_expression_spec,
    "expression": test_expression_spec,
    "clauses_or": test_clause_or_spec,
    "config_update": test_config_update,
    "mixed_and_or": test_mixed_and_or_spec,
    "compute_bounds": test_compute_bounds_linear_bounds,
    "compute_bounds_sound": test_compute_bounds_sound_multi_objective,
    "optimization_history": test_optimization_history,
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
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for name in to_run:
        fn = TESTS.get(name)
        if fn is None:
            print(f"[warn] unknown test '{name}', skipping.")
            skipped.append(name)
            continue
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"[fail] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed.append(name)
            print(f"[pass] {name}")

    print(
        "[summary] "
        f"total={len(passed) + len(failed)}, "
        f"passed={len(passed)}, "
        f"failed={len(failed)}, "
        f"skipped={len(skipped)}"
    )
    if passed:
        print("[summary] passed tests:", ", ".join(passed))
    if failed:
        print("[summary] failed tests:", ", ".join(failed))
        raise SystemExit(1)
    if skipped:
        print("[summary] skipped tests:", ", ".join(skipped))


if __name__ == "__main__":
    main()
