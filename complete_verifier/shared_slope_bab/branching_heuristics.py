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
"""Branching scores for the shared-slope BaB."""
import numpy as np
import torch


def shared_slope_branching(state, spec_output_coeffs, spec_bias):
    """Return unstable-neuron indices ordered by the box plus one-half-space score."""
    spec_output_coeffs_t = torch.as_tensor(
        spec_output_coeffs, device=state['output_coeffs'].device,
        dtype=state['output_coeffs'].dtype)
    d_obj_base = (spec_output_coeffs_t @ state['output_coeffs']).detach().cpu().double()
    d_obj_base = d_obj_base.numpy().reshape(-1)
    c0_obj_base = float(
        (spec_output_coeffs_t @ state['output_center']).detach().cpu().item()
        + spec_bias)
    base_abs = np.abs(d_obj_base)
    base_abs_sum = float(base_abs.sum())
    baseline_lb = c0_obj_base - base_abs_sum

    def _lmin_sparse(supp, a_supp, d_supp, d_enc, enc, c0, beta):
        off_abs = (
            base_abs_sum
            - float(base_abs[supp].sum())
            - float(base_abs[enc])
            + float(np.abs(d_supp).sum())
            + abs(float(d_enc))
        )
        g0 = c0 - off_abs
        eff = np.where(d_supp != 0, np.sign(d_supp), np.sign(a_supp))
        gp0 = -beta - float((a_supp * eff).sum())
        if gp0 <= 0:
            return g0
        valid = (a_supp != 0) & (d_supp != 0)
        if not valid.any():
            return float('inf')
        lam_v = -d_supp[valid] / a_supp[valid]
        a_v = a_supp[valid]
        pos = lam_v > 0
        if not pos.any():
            return float('inf')
        lam_pos = lam_v[pos]
        decr = 2.0 * np.abs(a_v[pos])
        order = np.argsort(lam_pos)
        lam_s = lam_pos[order]
        decr_s = decr[order]
        cum_before = np.concatenate(([0.0], np.cumsum(decr_s)))
        gaps = np.diff(np.concatenate(([0.0], lam_s)))
        slope_j = gp0 - cum_before[:-1]
        contrib = np.cumsum(slope_j * gaps)
        m = int(np.searchsorted(cum_before, gp0, side='left'))
        if m > lam_s.size:
            return float('inf')
        return g0 + float(contrib[m - 1])

    scores = np.zeros(len(state['unstable_list']), dtype=np.float64)
    for idx, u in enumerate(state['unstable_list']):
        lam = float(u['relu_slope'])
        mu = float(u['half_error'])
        c_in = float(u['preact_center'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)
        row_val = np.asarray(u['row_values'], dtype=np.float64)
        e_new_col = int(u['relu_error_col'])
        if row_idx.size == 0:
            continue
        if mu > 1e-12:
            ew_k = float(d_obj_base[e_new_col]) / mu
        else:
            continue
        db_supp = d_obj_base[row_idx]
        d_enc = float(d_obj_base[e_new_col]) - ew_k * mu

        d_off_supp = db_supp - ew_k * lam * row_val
        c0_off = c0_obj_base - ew_k * (lam * c_in + mu)
        lb_off = _lmin_sparse(row_idx, row_val, d_off_supp, d_enc, e_new_col,
                              c0_off, -c_in)

        d_on_supp = db_supp + ew_k * (1.0 - lam) * row_val
        c0_on = c0_obj_base + ew_k * ((1.0 - lam) * c_in - mu)
        lb_on = _lmin_sparse(row_idx, -row_val, d_on_supp, d_enc, e_new_col,
                             c0_on, c_in)
        scores[idx] = float(min(lb_off, lb_on)) - baseline_lb
    return np.argsort(-scores).tolist()
