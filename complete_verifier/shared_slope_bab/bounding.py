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
"""A fast per-node lower bound for the shared-slope substitution BaB."""
from __future__ import annotations
import numpy as np
import torch

TOL = 1e-9
_COMPILED_LOWER_BOUND = {}


def _log_bucket_line_search(rc, da, width, slope0, num_buckets, upper=None):
    """Choose a dual step size by bucketing its coordinate breakpoints.

    Shapes, where ``n`` is the number of coefficient columns:
        ``rc``, ``da``   [B, n]
        ``width``        scalar
        ``slope0``       [B]
        ``upper``        [B] or None

    Returns ``eta`` and ``emax`` with shape [B], plus ``crossed`` [B,
    num_buckets].
    """
    positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
    eta_j = torch.where(positive, -rc / da, torch.full_like(rc, float('inf')))
    max_eta = torch.where(positive, eta_j, torch.full_like(eta_j, -float('inf'))).max(-1).values
    has_breakpoint = positive.any(-1)
    if upper is None:
        emax = torch.where(
            has_breakpoint, max_eta,
            torch.zeros(rc.shape[0], device=rc.device, dtype=rc.dtype))
    else:
        finite_upper = torch.isfinite(upper)
        fallback = torch.where(finite_upper, upper, torch.zeros_like(upper))
        emax = torch.where(has_breakpoint, max_eta, fallback)
        emax = torch.where(finite_upper, torch.minimum(emax, upper), emax)
    active = positive & (eta_j <= emax.unsqueeze(-1)) & (eta_j > 0)
    has_active = active.any(-1)
    emin = torch.where(
        has_active,
        torch.where(active, eta_j, torch.full_like(eta_j, float('inf'))).min(-1).values,
        torch.zeros(rc.shape[0], device=rc.device, dtype=rc.dtype))
    safe_emin = emin.clamp_min(1e-30)
    safe_emax = torch.maximum(emax, safe_emin).clamp_min(1e-30)
    le = torch.log(safe_emin)
    lw = (torch.log(safe_emax) - le).clamp_min(1e-30) / num_buckets
    eta_for_index = torch.where(active, eta_j.clamp_min(1e-30), safe_emin.unsqueeze(-1))
    bidx = (((torch.log(eta_for_index) - le.unsqueeze(-1)) / lw.unsqueeze(-1))
            .floor().long()).clamp(0, num_buckets - 1)
    bidx = torch.where(active, bidx, torch.full_like(bidx, num_buckets))
    decr = torch.where(active, da.abs() * width, torch.zeros_like(da))
    buckets = torch.zeros(rc.shape[0], num_buckets + 1, device=rc.device, dtype=rc.dtype)
    buckets.scatter_add_(1, bidx, decr)
    crossed = buckets[:, :num_buckets].cumsum(-1) >= slope0.unsqueeze(-1)
    eta = torch.exp(le + (crossed.float().argmax(-1).to(rc.dtype) + 0.5) * lw)
    eta = torch.where(crossed.any(-1), eta, emax)
    return torch.where(slope0 > TOL, eta, torch.zeros_like(eta)).clamp_min(0.0), emax, crossed


def get_lower_bound(problem: dict, sides: torch.Tensor, phase_dual: torch.Tensor,
                    box_dual: torch.Tensor, iteration: int,
                    line_search_buckets: int = 256):
    """Warm-started dual lower bound for a batch of branch domains.

    For D fixed splits and n coefficient columns, ``problem['a_g']`` is
    [D, n], ``problem['a_l1']`` is [D], and its remaining ``ratio_*``,
    ``c0_*``, ``c_in``, and ``z_*`` entries are [D]. ``sides``,
    ``phase_dual``, and ``box_dual`` are [B, D]. The returned bound is [B],
    and the updated duals are [B, D].
    """
    a_g = problem['a_g']
    B = sides.shape[0]
    D = sides.shape[1]
    dtype = a_g.dtype
    sign = torch.where(sides == 0, torch.ones_like(sides), -torch.ones_like(sides)).to(dtype)
    rho = torch.where(sides == 0, problem['ratio_off'], problem['ratio_on'])
    c0_path = problem['c0'] + torch.where(sides == 0, problem['c0_off'], problem['c0_on']).sum(1)
    cin, zlo, zhi = problem['c_in'], problem['z_lo'], problem['z_hi']
    b0 = torch.where(sides == 0, cin.expand(B, D), (-cin).expand(B, D))
    b1 = torch.where(sides == 0, (-zlo).expand(B, D), (-zhi).expand(B, D))
    a_l1 = problem['a_l1']
    infeasible = ((b0 > a_l1.unsqueeze(0) + 1e-6)
                  | (b1 > a_l1.unsqueeze(0) + 1e-6)).any(1)

    path_base = problem['d_base'].unsqueeze(0) + rho @ a_g
    rc = path_base + (sign * (phase_dual - box_dual)) @ a_g
    best = c0_path - rc.abs().sum(-1) + (phase_dual * b0 + box_dual * b1).sum(-1)
    best = torch.maximum(best, c0_path - path_base.abs().sum(-1))

    for _ in range(iteration):
        x = torch.where(rc < 0, torch.ones_like(rc), -torch.ones_like(rc))
        p = x @ a_g.t()
        s0 = b0 + sign * p
        s1 = b1 - sign * p
        sp0 = torch.where((phase_dual <= TOL) & (s0 < 0), torch.zeros_like(s0), s0)
        sp1 = torch.where((box_dual <= TOL) & (s1 < 0), torch.zeros_like(s1), s1)
        slope0 = (sp0 * s0 + sp1 * s1).sum(-1)
        da = (sign * (sp0 - sp1)) @ a_g
        w0 = torch.where(sp0 < 0, -phase_dual / sp0, torch.full_like(phase_dual, float('inf')))
        w1 = torch.where(sp1 < 0, -box_dual / sp1, torch.full_like(box_dual, float('inf')))
        ebnd = torch.minimum(w0.min(-1).values, w1.min(-1).values)
        eta, _, _ = _log_bucket_line_search(
            rc, da, 2.0, slope0, line_search_buckets, upper=ebnd)
        eta = torch.minimum(eta, torch.where(torch.isfinite(ebnd), ebnd, eta)).clamp_min(0.0)
        phase_dual = (phase_dual + eta.unsqueeze(-1) * sp0).clamp_min(0.0)
        box_dual = (box_dual + eta.unsqueeze(-1) * sp1).clamp_min(0.0)
        rc = path_base + (sign * (phase_dual - box_dual)) @ a_g
        best = torch.maximum(
            best,
            c0_path - rc.abs().sum(-1) + (phase_dual * b0 + box_dual * b1).sum(-1))

    best = torch.where(infeasible, torch.full_like(best, float('inf')), best)
    return best, phase_dual, box_dual


def problem_at_depth(problem, depth):
    """Slice split-indexed rows and constants to the first ``depth`` splits.

    The returned ``a_g`` is [depth, n_columns], ``d_base`` is
    [n_columns], and the split constants are [depth].
    """
    d_base = problem['spec_coeffs'].clone()
    d_base[problem['relu_error_col'][:depth]] = 0.0
    a_g = problem['preact_coeffs'][:depth].contiguous()
    return {
        'a_g': a_g,
        'a_l1': a_g.abs().sum(-1),
        'd_base': d_base,
        'c0': torch.as_tensor(problem['spec_center'], device=a_g.device, dtype=a_g.dtype),
        'ratio_off': problem['subst_scale'][:depth, 0].contiguous(),
        'ratio_on': problem['subst_scale'][:depth, 1].contiguous(),
        'c0_off': problem['subst_const_shift'][:depth, 0].contiguous(),
        'c0_on': problem['subst_const_shift'][:depth, 1].contiguous(),
        'c_in': problem['preact_center'][:depth].contiguous(),
        'z_lo': problem['inactive_lower_shift'][:depth].contiguous(),
        'z_hi': problem['active_upper_shift'][:depth].contiguous(),
    }


def get_compiled_lower_bound(iteration, line_search_buckets):
    """Return a cached compiled lower-bound function for this configuration."""
    key = (int(iteration), int(line_search_buckets))
    if key not in _COMPILED_LOWER_BOUND:
        def kern(F, sides, phase_dual, box_dual):
            return get_lower_bound(
                F, sides, phase_dual, box_dual, iteration=iteration,
                line_search_buckets=line_search_buckets)
        _COMPILED_LOWER_BOUND[key] = torch.compile(kern, dynamic=True)
    return _COMPILED_LOWER_BOUND[key]


def build_problem(state, spec_output_coeffs, spec_bias, order, device, dtype=torch.float32):
    """Build substitution constants in branching order.

    The result has ``preact_coeffs`` [S, n_columns], ``spec_coeffs``
    [n_columns], and each remaining per-split tensor has leading size S.
    """
    n_columns = state['n_columns']
    unstable_neurons = state['unstable_list']
    spec_output_coeffs_t = torch.as_tensor(
        spec_output_coeffs, device=device, dtype=state['output_coeffs'].dtype)
    spec_coeffs = (spec_output_coeffs_t @ state['output_coeffs']).to(dtype)
    spec_center = float(spec_output_coeffs_t @ state['output_center'] + spec_bias)
    spec_coeffs_np = spec_coeffs.detach().cpu().numpy()

    num_splits = len(order)
    preact_coeffs = torch.zeros(num_splits, n_columns, device=device, dtype=dtype)
    subst_scale = torch.empty(num_splits, 2, device=device, dtype=dtype)
    subst_const_shift = torch.empty(num_splits, 2, device=device, dtype=dtype)
    preact_center = torch.empty(num_splits, device=device, dtype=dtype)
    inactive_lower_shift = torch.empty(num_splits, device=device, dtype=dtype)
    active_upper_shift = torch.empty(num_splits, device=device, dtype=dtype)
    relu_error_col = torch.empty(num_splits, device=device, dtype=torch.long)

    for split_index, unstable_index in enumerate(order):
        unstable_neuron = unstable_neurons[unstable_index]
        relu_slope = float(unstable_neuron['relu_slope'])
        half_error = float(unstable_neuron['half_error'])
        preact_center_value = float(unstable_neuron['preact_center'])
        relu_error_col_idx = int(unstable_neuron['relu_error_col'])
        row_indices = np.asarray(unstable_neuron['row_indices'], np.int64)
        row_values = np.asarray(unstable_neuron['row_values'], np.float64)

        preact_row = torch.zeros(n_columns, dtype=dtype)
        preact_row[torch.from_numpy(row_indices)] = torch.from_numpy(row_values).to(dtype)
        preact_row[relu_error_col_idx] = 0.0
        preact_coeffs[split_index] = preact_row.to(device)

        has_band = half_error > 1e-12
        inv_half_error = 1.0 / (half_error if has_band else 1.0)
        eliminated_error_coeff = float(spec_coeffs_np[relu_error_col_idx])

        subst_scale[split_index, 0] = (
            -(relu_slope * inv_half_error) * eliminated_error_coeff
        ) if has_band else 0.0
        subst_const_shift[split_index, 0] = (
            -(1 + relu_slope * preact_center_value * inv_half_error)
            * eliminated_error_coeff
        ) if has_band else -eliminated_error_coeff

        subst_scale[split_index, 1] = (
            ((1 - relu_slope) * inv_half_error) * eliminated_error_coeff
        ) if has_band else 0.0
        subst_const_shift[split_index, 1] = (
            ((1 - relu_slope) * preact_center_value * inv_half_error - 1)
            * eliminated_error_coeff
        ) if has_band else -eliminated_error_coeff

        preact_center[split_index] = preact_center_value
        inactive_lower_shift[split_index] = (
            2 * half_error / max(relu_slope, 1e-30) + preact_center_value
        ) if (has_band and relu_slope > 1e-12) else 1e9
        active_upper_shift[split_index] = (
            2 * half_error / max(1 - relu_slope, 1e-30) - preact_center_value
        ) if (has_band and (1 - relu_slope) > 1e-12) else 1e9
        relu_error_col[split_index] = relu_error_col_idx

    return dict(
        n_columns=n_columns,
        preact_coeffs=preact_coeffs,
        spec_coeffs=spec_coeffs,
        spec_center=spec_center,
        subst_scale=subst_scale,
        subst_const_shift=subst_const_shift,
        preact_center=preact_center,
        inactive_lower_shift=inactive_lower_shift,
        active_upper_shift=active_upper_shift,
        relu_error_col=relu_error_col,
        S=num_splits,
        dtype=dtype,
    )


def reorder_problem(problem, order):
    """
    Reorder split-indexed tensors without rebuilding dense coefficient rows.
    """
    order_t = torch.as_tensor(order, device=problem['preact_coeffs'].device, dtype=torch.long)
    for key in (
            'preact_coeffs', 'subst_scale', 'subst_const_shift',
            'preact_center', 'inactive_lower_shift', 'active_upper_shift',
            'relu_error_col'):
        problem[key] = problem[key].index_select(0, order_t)
    return problem
