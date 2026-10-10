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
"""Entry point and BaB loop for the shared-slope verifier."""
from __future__ import annotations

import time

import torch
import arguments
from shared_slope_bab.relaxation_state import build_relaxation_state, root_bound
from shared_slope_bab.bounding import (
    TOL, problem_at_depth, get_compiled_lower_bound,
    build_problem, reorder_problem)
from shared_slope_bab.branching_heuristics import shared_slope_branching
from shared_slope_bab.branching_domains import SharedSlopeDomainList


def _relaxation_state_bab(state, spec_output_coeffs, spec_bias, device,
                          timeout=None, max_domains=None):
    """Run the shared-slope dual BaB for every open specification."""
    ssb_args = arguments.Config['bab']['shared_slope_bab']
    iteration = ssb_args.get('iteration', 1)
    line_search_buckets = ssb_args.get('line_search_buckets', 256)
    n_open = spec_output_coeffs.shape[0]
    infos = []
    for spec_idx in range(n_open):
        if timeout is not None and timeout <= 0:
            return 'unknown', {'specs': infos, 'failed': spec_idx, 'reason': 'timeout'}
        status, info = single_spec_bab(
            state, spec_output_coeffs[spec_idx], float(spec_bias[spec_idx]), device,
            timeout=timeout, iteration=iteration, max_domains=max_domains,
            verbose=True, line_search_buckets=line_search_buckets)
        infos.append((spec_idx, status, info))
        print(f'[Shared-slope BaB] spec {spec_idx}: {status} nodes={info.get("nodes")} '
              f'depth={info.get("depth")} peak={info.get("peak")} '
              f'reason={info.get("reason", "done")} wall={info.get("wall", 0):.2f}s')
        if status != 'safe':
            return 'unknown', {'specs': infos, 'failed': spec_idx}
    return 'safe', {'specs': infos}


def _visited(info):
    """Count the BaB nodes visited for the specifications in ``info``."""
    return sum(spec_info.get('nodes', 0) for _, _, spec_info in info.get('specs', []))


def shared_slope_bab(net, x, c, rhs, ret, *, timeout=None):
    """Try the shared-slope BaB before the beta-CROWN BaB loop.

    The incoming bounds in ``ret`` are the BaB initialization. This builds a
    same-slope relaxation from those bounds and runs the substitution BaB
    on every open specification.

    Returns ``(global_lb, visited, status)``. The lower bounds keep the shape in
    ``ret``; status is ``safe`` only when every open specification certifies.
    """
    bab_args = arguments.Config['bab']
    ssb_args = bab_args['shared_slope_bab']
    global_lower_bound = ret['global_lb']                   # [B, n_spec]
    visited = 0
    if c is None:
        return global_lower_bound, visited, 'unknown'

    device = net.device
    rhs_2d = rhs[:, :, 0] if rhs.dim() == 3 else rhs        # [B, n_spec]
    batch_size = x.ptb.x_L.shape[0]

    if batch_size > 1:
        same_input_box = bool(torch.allclose(x.ptb.x_L, x.ptb.x_L[0:1])) and \
                         bool(torch.allclose(x.ptb.x_U, x.ptb.x_U[0:1]))
        if not same_input_box:
            print(f'[Shared-slope BaB] skip: batched spec with DISTINCT input boxes '
                  f'(B={batch_size}) -> fall through to beta-CROWN')
            return global_lower_bound, visited, 'unknown'

    # Open (batch, spec) pairs that the BaB initialization bounds have not verified.
    open_pairs = (global_lower_bound - rhs_2d <= 0).nonzero(as_tuple=False)   # [n_open, 2]
    if open_pairs.shape[0] == 0:
        return global_lower_bound, visited, 'safe'

    # Each open pair becomes a query over the network output, with bias = -rhs.
    open_batch_idx = open_pairs[:, 0]
    open_spec_idx = open_pairs[:, 1]
    c_batch_idx = open_batch_idx.to(c.device)
    c_spec_idx = open_spec_idx.to(c.device)
    spec_output_coeffs = c[c_batch_idx, c_spec_idx].to(device)   # [n_open, n_out]
    spec_bias = (-rhs_2d[open_batch_idx, open_spec_idx]).to(device)  # [n_open]
    init_margin = (
        global_lower_bound[open_batch_idx, open_spec_idx]
        - rhs_2d[open_batch_idx, open_spec_idx]).to(device)

    # Share relaxation slopes across all open queries when building the state.
    slope_iteration = ssb_args.get('slope_iteration', 80)
    num_queries = spec_output_coeffs.shape[0]
    shared_slope_iteration = ssb_args.get('shared_slope_iteration', -1)
    if shared_slope_iteration < 0:
        shared_slope_iteration = 0 if num_queries > 1 else slope_iteration
    shared_max_domains = ssb_args.get('shared_max_domains', 250000)
    bab_start = time.time()
    slope_opt_spec_coeffs = spec_output_coeffs.detach().cpu().double().numpy()
    remaining = None if timeout is None else timeout - (time.time() - bab_start)
    if remaining is not None and remaining <= 0:
        print('[Shared-slope BaB] timeout before shared relaxation state build; '
              'fall through to beta-CROWN')
        return global_lower_bound, visited, 'unknown'
    t_state = time.time()
    try:
        state = build_relaxation_state(
            net, ret, x, device, bidx=0,
            slope_opt_spec_coeffs=slope_opt_spec_coeffs,
            slope_iteration=shared_slope_iteration, verbose=False)
    except NotImplementedError as exc:
        print(f'[Shared-slope BaB] skip: {exc} -> fall through to beta-CROWN')
        return global_lower_bound, visited, 'unknown'
    print(f'[Shared-slope BaB] shared build for {num_queries} queries '
          f'{time.time()-t_state:.2f}s n_columns={state["n_columns"]} '
          f'unstable={len(state["unstable_list"])} '
          f'slope_iteration={shared_slope_iteration}')

    # Verify each open query.
    for query_idx in range(num_queries):
        remaining = None if timeout is None else timeout - (time.time() - bab_start)
        if remaining is not None and remaining <= 0:
            print('[Shared-slope BaB] timeout before query build; fall through to beta-CROWN')
            return global_lower_bound, visited, 'unknown'
        query_spec_output_coeffs = spec_output_coeffs[query_idx:query_idx + 1].to(
            dtype=state['output_coeffs'].dtype)
        query_spec_bias = spec_bias[query_idx:query_idx + 1].to(
            dtype=state['output_coeffs'].dtype)
        root_bound_value = float(
            root_bound(state, query_spec_output_coeffs, query_spec_bias).item())
        print(f'[Shared-slope BaB] query {query_idx+1}/{num_queries} '
              f'root={root_bound_value:.4f} init-margin={float(init_margin[query_idx]):.4f}')
        remaining = None if timeout is None else timeout - (time.time() - bab_start)
        status, info = _relaxation_state_bab(
            state, query_spec_output_coeffs, query_spec_bias, device,
            timeout=remaining,
            max_domains=(shared_max_domains if num_queries > 1 else None))
        visited += _visited(info)
        if status != 'safe':
            first_info = (info.get('specs', [(None, None, {})])[-1][2]
                          if isinstance(info, dict) else {})
            if (first_info.get('reason') == 'max_domains'
                    and slope_iteration > shared_slope_iteration):
                remaining = None if timeout is None else timeout - (time.time() - bab_start)
                if remaining is not None and remaining <= 0:
                    print('[Shared-slope BaB] timeout before per-query fallback; '
                          'fall through to beta-CROWN')
                    return global_lower_bound, visited, 'unknown'
                t_fallback = time.time()
                fallback_spec = (spec_output_coeffs[query_idx:query_idx + 1]
                                 .detach().cpu().double().numpy())
                try:
                    fallback_state = build_relaxation_state(
                        net, ret, x, device, bidx=0,
                        slope_opt_spec_coeffs=fallback_spec,
                        slope_iteration=slope_iteration, verbose=False)
                except NotImplementedError as exc:
                    print(f'[Shared-slope BaB] skip: per-query fallback build failed '
                          f'({exc}) -> fall through to beta-CROWN')
                    return global_lower_bound, visited, 'unknown'
                fallback_root = float(root_bound(
                    fallback_state,
                    spec_output_coeffs[query_idx:query_idx + 1].to(
                        dtype=fallback_state['output_coeffs'].dtype),
                    spec_bias[query_idx:query_idx + 1].to(
                        dtype=fallback_state['output_coeffs'].dtype)).item())
                print(f'[Shared-slope BaB] query {query_idx+1}/{num_queries} '
                      f'fallback build {time.time()-t_fallback:.2f}s '
                      f'root={fallback_root:.4f} slope_iteration={slope_iteration}')
                remaining = None if timeout is None else timeout - (time.time() - bab_start)
                status, info = _relaxation_state_bab(
                    fallback_state, query_spec_output_coeffs, query_spec_bias,
                    device, timeout=remaining, max_domains=None)
                visited += _visited(info)
        print(f'[Shared-slope BaB] query {query_idx+1}/{num_queries}: '
              f'{status}  {info.get("specs", info)}')
        if status != 'safe':
            return global_lower_bound, visited, 'unknown'

    updated_global_lower_bound = global_lower_bound.clone()
    updated_global_lower_bound[:, :] = rhs_2d + 1.0
    return updated_global_lower_bound, visited, 'safe'

def _print_bab_round(info):
    print(f'BaB round {info["round"]}')
    print(f'batch: {info["batch"]}')
    print('splitting decisions:')
    print(f'split level 0: {info["decision"]}')
    print(f'best_l after optimization: {info["best_l"]}')
    print(f'beta sum per layer: {info["beta_per_layer"]}')
    print(f'alpha/beta optimization time: {info["bound_time"]}')
    print('pruning_in_iteration open status: True')
    print(f'ratio of positive domain = {info["n_pruned"]} / {info["batch"]} = {info["ratio"]}')
    print(f'pruning-in-iteration extra time: {info["prune_time"]}')
    print('Time: ' + '    '.join(f'{k} {v:.4f}' for k, v in info['round_time'].items()))
    print('Accumulated time: ' + '    '.join(f'{k} {v:.4f}' for k, v in info['acc_time'].items()))
    print(f'max lb {info["max_lb"]} min lb {info["min_lb"]}')
    print(f'Number of Verified Splits: {info["n_pruned"]} of {info["batch"]}')
    print('Current worst splitting domains lb-rhs (depth):')
    print(', '.join(f'{b:.5f} ({d})' for b, d in info['worst_domains']))
    print(f'Current (lb-rhs): {info["worst"]}')
    print(f'{info["nodes"]} domains visited')
    print(f'Length of domains: {info["n_open"]}')
    print(f'Cumulative time: {info["cum_time"]}')
    print()


def single_spec_bab(state, spec_output_coeffs, spec_bias, device, timeout=None,
                    iteration=1, batch_size=16384, max_domains=None,
                    bab_dtype=torch.float32, verbose=True, line_search_buckets=256):
    """
    Relaxation substitution BaB with the factored dual kernel.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    # Branch order uses the box plus one half-space worst-child score.
    unordered_problem = build_problem(
        state, spec_output_coeffs, spec_bias, list(range(len(state['unstable_list']))),
        device, bab_dtype)
    order = shared_slope_branching(state, spec_output_coeffs, spec_bias)
    S = len(order)
    problem = reorder_problem(unordered_problem, order)

    box_minimizer = torch.where(
        problem['spec_coeffs'] < 0,
        torch.ones_like(problem['spec_coeffs']),
        -torch.ones_like(problem['spec_coeffs']))
    g_root = float((problem['spec_coeffs'] * box_minimizer).sum()) + problem['spec_center']
    if verbose:
        print(f'initial shared-slope substitution BaB: {S} unstable neurons, '
              f'iteration={iteration}, '
              f'root (lb-rhs)={g_root:.4f}')
    if g_root > TOL:
        return 'safe', dict(nodes=1, depth=0, peak=0, wall=0.0)
    if S == 0:
        # Nothing to split and the root did not clear: the root bound is all
        # this kernel can say.
        return 'unknown', dict(nodes=1, depth=0, peak=0, wall=0.0,
                               reason='no_splits')

    # per-split (in branch order) relu layer name / neuron, for the round log:
    # the neuron split that produced round d's batch is order[d-1].  layer_idx is the
    # relu's graph node name (e.g. '/input-72'), matching the native [/input-72, 28] style.
    unstable_list = state['unstable_list']
    split_layer = [str(unstable_list[order[k]]['layer_idx']) for k in range(S)]
    split_neuron = [int(unstable_list[order[k]]['neuron_idx']) for k in range(S)]
    uniq_layers = sorted(set(split_layer))
    layer_id = {name: i for i, name in enumerate(uniq_layers)}

    compiled_lower_bound = get_compiled_lower_bound(iteration, line_search_buckets)
    for warm_depth in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        if warm_depth <= S:
            Fw = problem_at_depth(problem, warm_depth)
            zw = torch.zeros(8, warm_depth, device=device, dtype=bab_dtype)
            sw = torch.zeros(8, warm_depth, device=device, dtype=torch.long)
            compiled_lower_bound(Fw, sw, zw, zw.clone())
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    domains = SharedSlopeDomainList.root(device, bab_dtype)
    nodes = 1
    depth = 1
    peak = 2
    round_no = 0
    acc = {'solve': 0.0, 'prune': 0.0, 'add': 0.0, 'bab_iteration': 0.0}
    while len(domains) > 0 and depth <= S:
        if timeout is not None and time.perf_counter() - t0 > timeout:
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=len(domains),
                                   reason='timeout',
                                   wall=time.perf_counter() - t0)
        B = len(domains)
        nodes += B
        peak = max(peak, B)
        round_no += 1
        best_bound = torch.empty(B, device=device, dtype=bab_dtype)
        next_phase_dual = torch.empty(B, depth, device=device, dtype=bab_dtype)
        next_box_dual = torch.empty(B, depth, device=device, dtype=bab_dtype)
        t_solve = time.perf_counter()
        i = 0
        cs = batch_size
        while i < B:
            s = min(cs, B - i)
            while True:
                try:
                    bound_chunk, phase_dual_chunk, box_dual_chunk = compiled_lower_bound(
                        problem_at_depth(problem, depth),
                        domains.sides[i:i+s].to(torch.long).contiguous(),
                        domains.phase_dual[i:i+s],
                        domains.box_dual[i:i+s])
                    break
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if s <= 128:
                        raise
                    s //= 2
                    cs = max(128, min(cs, s))
            best_bound[i:i+s] = bound_chunk
            next_phase_dual[i:i+s] = phase_dual_chunk
            next_box_dual[i:i+s] = box_dual_chunk
            i += s
        round_solve_time = time.perf_counter() - t_solve

        t_prune = time.perf_counter()
        keep = best_bound <= TOL
        n_pruned = int((~keep).sum())
        open_domains = SharedSlopeDomainList(
            domains.sides, next_phase_dual, next_box_dual).filter_verified_domains(keep)
        n_open = len(open_domains)
        round_prune_time = time.perf_counter() - t_prune

        if (max_domains and max_domains > 0 and n_open > 0
                and depth < S and 2 * n_open > max_domains):
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=int(2 * n_open), reason='max_domains',
                                   wall=time.perf_counter() - t0)

        # build next-round frontier now (so its cost shows in this round's 'add' time)
        t_add = time.perf_counter()
        will_expand = n_open > 0 and depth < S
        next_domains = None
        if will_expand:
            next_domains = open_domains.split_domains()
        round_add_time = time.perf_counter() - t_add

        cum = time.perf_counter() - t0
        acc['solve'] += round_solve_time
        acc['prune'] += round_prune_time
        acc['add'] += round_add_time
        acc['bab_iteration'] = cum

        if verbose:
            # beta sum per layer = dual mass (phase_dual+box_dual) grouped by relu layer
            beta_per_layer = [0.0 for _ in uniq_layers]
            for col, layer_name in enumerate(split_layer[:depth]):
                beta_per_layer[layer_id[layer_name]] += float(
                    next_phase_dual[:, col].abs().sum()
                    + next_box_dual[:, col].abs().sum())
            beta_per_layer = [round(v, 4) for v in beta_per_layer]
            finite = best_bound[torch.isfinite(best_bound)]
            max_lb = float(finite.max()) if finite.numel() else float('inf')
            best_l = float(finite.sum()) if finite.numel() else float('inf')
            open_bounds = best_bound[keep]
            k = min(20, n_open)
            worst_vals = (torch.topk(open_bounds, k, largest=False).values.tolist()
                          if k > 0 else [])
            # worst still-open (lb-rhs); at the closing round (none open) fall back
            # to the smallest now-verified bound so the line stays meaningful.
            worst = float(open_bounds.min()) if n_open else float(best_bound.min())
            _print_bab_round(dict(
                round=round_no, batch=B,
                decision=f'[layer {split_layer[depth-1]}, neuron {split_neuron[depth-1]}]'
                         f'  (uniform: all {B} domains, global branch order)',
                best_l=best_l, beta_per_layer=beta_per_layer,
                bound_time=round_solve_time, n_pruned=n_pruned, ratio=n_pruned / B,
                prune_time=round_prune_time,
                round_time={'solve': round_solve_time, 'prune': round_prune_time,
                            'add': round_add_time,
                            'bab_iteration': round_solve_time + round_prune_time + round_add_time},
                acc_time=acc,
                max_lb=max_lb, min_lb=float(best_bound.min()),
                worst_domains=[(v, depth) for v in worst_vals], worst=worst,
                nodes=nodes, n_open=n_open, cum_time=cum))

        if n_open == 0:
            return 'safe', dict(nodes=nodes, depth=depth, peak=peak,
                                wall=time.perf_counter() - t0)
        if depth >= S:
            return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                                   open=n_open, reason='splits_exhausted',
                                   wall=time.perf_counter() - t0)
        domains = next_domains
        depth += 1
    # Every exit above returns; falling out of the loop means no round ran, and
    # that is never a proof.
    return 'unknown', dict(nodes=nodes, depth=depth, peak=peak,
                           reason='no_round', wall=time.perf_counter() - t0)
