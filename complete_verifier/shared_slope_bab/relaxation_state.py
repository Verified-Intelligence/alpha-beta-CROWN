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
"""Build the shared-slope BaB state: the centre and generators of the output
and of every unstable pre-activation, from one dynamic-forward compute_bounds
call, optionally after a search for the ReLU slopes."""
from __future__ import annotations
import numpy as np
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm


def build_relaxation_state(net, ret, x, device, dtype=torch.float64,
                           slope_opt_spec_coeffs=None, slope_iteration=80,
                           lr_slope=0.03, bidx=0, verbose=True):
    """
    Build the relaxation state for one input box.

    Every node of the relaxed network is `c + G e`, its centre and generators
    over the variables `e in [-1, 1]^n`: one per input coordinate the box
    moves, then one per unstable ReLU neuron (`n_columns` in all). The state
    holds them in the form the BaB solver reads:

        output_coeffs  [n_out, n_columns]   the output's generators G
        output_center  [n_out]              the output's centre c
        var_lb, var_hi [n_columns]          -1 and +1, each variable's range
        unstable_list  per unstable neuron: its input bounds, its slope and
                       half-width (`relu_slope`, `half_error`), the column of
                       its own variable, and its pre-activation's generators,
                       stored sparse as (row_indices, row_values)
        relu_error_col (relu name, flat neuron index) -> column of its variable

    The pre-activation bounds are the BaB initialization's (`ret`), held
    fixed. With `slope_opt_spec_coeffs` (`[n_query, n_out]`) and
    `slope_iteration > 0`, the slopes of the unstable ReLUs are searched to
    maximize the worst query's lower bound
    (`compute_bounds(method='dynamic-forward-optimized')`); otherwise each is
    the default `u / (u - l)`. The same compute_bounds call returns the state
    under those slopes.

    Raises NotImplementedError if dynamic forward cannot read the graph.
    """
    bm = net.net
    ptb = x.ptb
    x_L, x_U = ptb.x_L[bidx:bidx + 1], ptb.x_U[bidx:bidx + 1]
    if not (torch.equal(ptb.x_L, x_L.expand_as(ptb.x_L))
            and torch.equal(ptb.x_U, x_U.expand_as(ptb.x_U))):
        raise NotImplementedError(
            f'the relaxation state describes the single box at bidx={bidx}, '
            f'but the batch holds {ptb.x_L.shape[0]} different ones')
    x1 = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        eps=ptb.eps, norm=ptb.norm, x_L=x_L, x_U=x_U))
    final_node = bm[net.final_name]
    n_out = int(np.prod([int(d) for d in final_node.output_shape[1:]]))
    preacts = {r.name: r.inputs[0] for r in bm.relus}
    interm = {pre.name: (ret['lower_bounds'][pre.name][bidx:bidx + 1].to(x_L),
                         ret['upper_bounds'][pre.name][bidx:bidx + 1].to(x_L))
              for pre in preacts.values()}
    # relu name -> flat indices of its unstable neurons, in column order
    unstable = {name: torch.logical_and(
        interm[pre.name][0] < 0, interm[pre.name][1] > 0
    ).flatten().nonzero(as_tuple=True)[0] for name, pre in preacts.items()}
    search = (slope_opt_spec_coeffs is not None and slope_iteration > 0
              and any(u.numel() for u in unstable.values()))
    spec = None
    if search:
        spec = torch.atleast_2d(torch.as_tensor(
            slope_opt_spec_coeffs, device=x_L.device, dtype=x_L.dtype))[None]
    # Search settings belong to this build; beta-CROWN retains its own options.
    options = bm.bound_opts['optimize_bound_args']
    search_options = {'iteration': slope_iteration, 'lr_alpha': lr_slope,
                      'early_stop_patience': slope_iteration,
                      'stop_criterion_func': lambda _: False}
    previous = {key: options[key] for key in search_options}
    try:
        if search:
            bm.set_bound_opts({'optimize_bound_args': search_options})
        with torch.no_grad():
            _, _, found = bm.compute_bounds(
                x=(x1,), C=spec, interm_bounds=interm,
                method='dynamic-forward-optimized' if search else 'dynamic-forward',
                return_generators=True,
                needed_generators_dict={
                    final_node.name: None,
                    **{pre.name: unstable[name] for name, pre in preacts.items()}})
    finally:
        options.update(previous)
    slopes = bm.dynamic_forward_slopes or {}
    # one box: G [1, elements, n] -> [elements, n], c [1, elements] -> [elements]
    generators = {name: (G[0], c[0]) for name, (G, c) in found.items()}
    # which columns each node's variables take
    ranges = bm[bm.input_name[0]].linear.dynamic_variable_ranges
    output_coeffs, output_center = (
        t.to(dtype) for t in generators[final_node.name])
    n_columns = int(output_coeffs.shape[1])
    # the input's variables come first: one per coordinate its box moves
    n_input = ranges[bm.input_name[0]][1]

    # A record per unstable neuron, in column order. A pre-activation never
    # depends on its own ReLU's variable, which only appears after the ReLU.
    relu_error_col = {}
    unstable_list = []
    for name, pre in preacts.items():
        u_idx = unstable[name]
        col, count = ranges[name]
        if count != int(u_idx.numel()):
            raise NotImplementedError(
                f'{name}: the pass opened {count} columns for '
                f'{int(u_idx.numel())} unstable neurons')
        if not count:
            continue
        lo, hi = interm[pre.name]
        slope, half = BoundRelu.dynamic_forward_relaxation(
            lo, hi, slopes.get(name))
        lo_u, hi_u, slope_u, half_u = torch.stack(
            [t.flatten()[u_idx] for t in (lo, hi, slope, half)]
        ).to(dtype).cpu().tolist()
        # the pre-activation's generators at its unstable neurons, one row
        # per neuron: [count, number of variables it depends on]
        pre_G, preact_center = generators[pre.name]
        pre_G = pre_G.to(dtype).cpu()
        # Store each row sparse: nonzero() lists the entries row by row, so
        # splitting at the per-row counts gives each row's columns and values.
        row_of, column = pre_G.nonzero(as_tuple=True)
        splits = np.cumsum(np.bincount(row_of.numpy(), minlength=count))[:-1]
        row_indices = np.split(column.numpy(), splits)
        row_values = np.split(
            pre_G[row_of, column].numpy().astype(np.float64, copy=False),
            splits)
        centres = preact_center.to(dtype).cpu().tolist()
        for i, j in enumerate(u_idx.tolist()):
            relu_error_col[(name, j)] = col + i
            unstable_list.append(dict(
                layer_idx=name, neuron_idx=j,
                preact_center=centres[i], lo=lo_u[i], hi=hi_u[i],
                relu_slope=slope_u[i], half_error=half_u[i],
                relu_error_col=col + i,
                row_indices=row_indices[i], row_values=row_values[i]))
    if sum(count for _, count in ranges.values()) != n_columns:
        # The BaB solver needs every variable in the output's generators.
        raise NotImplementedError(
            f'some variables do not reach the output: '
            f'{ranges}, output width {n_columns}')

    if verbose:
        print(f'[Shared-slope BaB] n_input={n_input} '
              f'n_unstable={n_columns - n_input} n_columns={n_columns}')

    return dict(
        n_columns=n_columns, n_input=n_input, n_out=n_out,
        unstable_list=unstable_list,
        output_coeffs=output_coeffs, output_center=output_center,
        var_lb=torch.full((n_columns,), -1.0, device=device, dtype=dtype),
        var_hi=torch.full((n_columns,), 1.0, device=device, dtype=dtype),
        relu_error_col=relu_error_col, relu_order=list(preacts),
    )


def root_bound(state, spec_output_coeffs, spec_bias):
    """
    Concretized root LP bound:
    min_{e in box} (spec_output_coeffs @ output_coeffs) e
    + (spec_output_coeffs @ output_center + spec_bias).
    spec_output_coeffs: [n_out] or [n_spec, n_out].  Returns the box minimum (scalar / [n_spec]).
    """
    single = torch.as_tensor(spec_output_coeffs).dim() == 1
    lower_bound = root_decomposition(
        state, spec_output_coeffs, spec_bias)['lower_bound']
    return lower_bound[0] if single else lower_bound


def root_decomposition(state, spec_output_coeffs, spec_bias):
    """
    Root-level coefficient diagnostics for one or more output specs.
    """
    output_coeffs = state['output_coeffs']
    output_center = state['output_center']
    var_hi = state['var_hi']
    spec_output_coeffs = torch.as_tensor(
        spec_output_coeffs, device=output_coeffs.device, dtype=output_coeffs.dtype)
    if spec_output_coeffs.dim() == 1:
        spec_output_coeffs = spec_output_coeffs.unsqueeze(0)
    spec_coeffs = spec_output_coeffs @ output_coeffs  # [n_spec, n_columns]
    spec_center = spec_output_coeffs @ output_center + torch.as_tensor(
        spec_bias, device=output_coeffs.device, dtype=output_coeffs.dtype)
    abs_contribs = spec_coeffs.abs() * var_hi
    n_input = int(state['n_input'])
    unstable_error_cols = [int(u['relu_error_col']) for u in state['unstable_list']]
    input_abs_contrib = abs_contribs[:, :n_input].sum(-1)
    if unstable_error_cols:
        unstable_error_cols_t = torch.tensor(
            unstable_error_cols, device=output_coeffs.device, dtype=torch.long)
        unstable_error_coeffs = spec_coeffs.index_select(1, unstable_error_cols_t)
        relu_error_abs_contrib = abs_contribs.index_select(1, unstable_error_cols_t).sum(-1)
    else:
        unstable_error_coeffs = spec_coeffs.new_zeros((spec_coeffs.shape[0], 0))
        relu_error_abs_contrib = spec_coeffs.new_zeros((spec_coeffs.shape[0],))
    total_abs_contrib = abs_contribs.sum(-1)
    lower_bound = spec_center - total_abs_contrib
    return {
        'spec_center': spec_center,
        'spec_coeffs': spec_coeffs,
        'abs_contribs': abs_contribs,
        'input_abs_contrib': input_abs_contrib,
        'relu_error_abs_contrib': relu_error_abs_contrib,
        'total_abs_contrib': total_abs_contrib,
        'lower_bound': lower_bound,
        'unstable_error_cols': unstable_error_cols,
        'unstable_error_coeffs': unstable_error_coeffs,
    }


def intermediate_bounds_table(state):
    """
    Return one diagnostic row per unstable ReLU neuron.
    """
    rows = []
    for unstable_neuron in state['unstable_list']:
        rows.append({
            'layer_idx': unstable_neuron['layer_idx'],
            'neuron_idx': int(unstable_neuron['neuron_idx']),
            'lo': float(unstable_neuron['lo']),
            'hi': float(unstable_neuron['hi']),
            'preact_center': float(unstable_neuron['preact_center']),
            'relu_slope': float(unstable_neuron['relu_slope']),
            'half_error': float(unstable_neuron['half_error']),
            'relu_error_col': int(unstable_neuron['relu_error_col']),
        })
    return rows
