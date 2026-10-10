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
"""Sanity tests for the shared-slope substitution BaB.

The first group uses the hand-derived section 2 example, a 2-2-2-1 ReLU net:
  * root relaxation bound = -1/2 (s2.4);
  * splitting neuron (2,1)=e3 gives both children's exact LP bound 0 (s2.5);
  * the property f >= 0 is proved after one split.

Covered:
  * build_relaxation_state on a real tiny BoundedModule reproduces the
    affine relaxation equations (CROWN backward vs hand derivation);
  * the factored dual solver is sound (bound <= exact LP, via a scipy oracle);
  * the factored single-constraint infeasibility prune is SOUND (a feasible node is
    never falsely certified) -- this test catches the sign-bug fixed on 2026-06-28.

Run: python -m pytest tests/test_shared_slope_bab.py -v
"""
from pathlib import Path

import numpy as np
import torch
import pytest

from shared_slope_bab.bounding import build_problem

_DT = torch.float64


def _doc_state(device):
    """The hand-derived section-2 example as a same-slope relaxation state.

    coefficient columns: [e1^0, e2^0, e1, e2, e3, e4] (n_columns=6, n_input=2).
    Output affine form (eq 14): f = .5 e1^0 + .5 e2^0 + .5 e1 + 0 e2 + .5 e3 + .5 e4 + 2.
    """
    output_coeffs = torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.5, 0.5]], dtype=_DT, device=device)
    output_center = torch.tensor([2.0], dtype=_DT, device=device)

    def neuron(row, relu_error_col_idx):
        return dict(layer_idx=0, neuron_idx=relu_error_col_idx, preact_center=0.0, lo=-2.0, hi=2.0,
                    relu_slope=0.5, half_error=0.5, relu_error_col=relu_error_col_idx,
                    row_indices=np.array([i for i, v in enumerate(row) if v != 0],
                                         dtype=np.int64),
                    row_values=np.array([v for v in row if v != 0], dtype=np.float64))

    ul = [
        neuron([1, 1, 0, 0, 0, 0], 2),      # L1n1 (e1): x = e1^0 + e2^0
        neuron([1, -1, 0, 0, 0, 0], 3),     # L1n2 (e2): x = e1^0 - e2^0
        neuron([1, 0, 0.5, 0.5, 0, 0], 4),  # L2n1 (e3): x = e1^0 + .5e1 + .5e2
        neuron([0, 1, 0.5, -0.5, 0, 0], 5),  # L2n2 (e4): x = e2^0 + .5e1 - .5e2
    ]
    return dict(n_columns=6, n_input=2, n_out=1, output_coeffs=output_coeffs, output_center=output_center,
                var_lb=torch.full((6,), -1.0, dtype=_DT, device=device),
                var_hi=torch.full((6,), 1.0, dtype=_DT, device=device),
                unstable_list=ul)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_shared_slope_bab_has_package():
    """The implementation modules should live in the shared_slope_bab package."""
    root = Path(__file__).resolve().parents[1]
    package = root / 'shared_slope_bab'
    assert (package / 'branch_and_bound.py').exists()
    assert (package / 'bounding.py').exists()
    assert (package / 'relaxation_state.py').exists()


def test_root_bound_matches_doc():
    """Root relaxation bound = -1/2 (hand-derived affine example section 2.4)."""
    dev = _device()
    state = _doc_state(dev)
    problem = build_problem(state, np.array([1.0]), 0.0, list(range(4)), dev, _DT)
    # root LP min_e d.e + spec_center over the box [-1,1] = spec_center - sum|d|  (eq 13)
    root = float(problem['spec_center']) - float(problem['spec_coeffs'].abs().sum())
    assert abs(root - (-0.5)) < 1e-9, root


def test_root_decomposition_exposes_center_coefficients_and_error_terms():
    """Root diagnostics should isolate the center, coefficients, and ReLU error terms."""
    from shared_slope_bab.relaxation_state import root_decomposition
    dev = _device()
    state = _doc_state(dev)
    diag = root_decomposition(state, np.array([1.0]), 0.0)
    assert torch.allclose(diag['spec_center'], torch.tensor([2.0], dtype=_DT, device=dev))
    assert torch.allclose(
        diag['spec_coeffs'][0],
        torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 0.5], dtype=_DT, device=dev))
    assert torch.allclose(diag['input_abs_contrib'], torch.tensor([1.0], dtype=_DT, device=dev))
    assert torch.allclose(diag['relu_error_abs_contrib'], torch.tensor([1.5], dtype=_DT, device=dev))
    assert torch.allclose(diag['lower_bound'], torch.tensor([-0.5], dtype=_DT, device=dev))
    assert diag['unstable_error_cols'] == [2, 3, 4, 5]
    assert torch.allclose(
        diag['unstable_error_coeffs'][0],
        torch.tensor([0.5, 0.0, 0.5, 0.5], dtype=_DT, device=dev))


def test_intermediate_bounds_table_exposes_per_unstable_neuron_rows():
    """Bounds diagnostics should give one alignment row per unstable neuron."""
    from shared_slope_bab.relaxation_state import intermediate_bounds_table
    dev = _device()
    rows = intermediate_bounds_table(_doc_state(dev))
    assert len(rows) == 4
    assert rows[0]['layer_idx'] == 0
    assert rows[0]['neuron_idx'] == 2
    assert rows[0]['lo'] == -2.0
    assert rows[0]['hi'] == 2.0
    assert rows[0]['preact_center'] == 0.0
    assert rows[0]['relu_slope'] == 0.5
    assert rows[0]['half_error'] == 0.5
    assert rows[0]['relu_error_col'] == 2


# --- factored dual kernel ---

def test_factored_dual_certifies_doc_example():
    """The factored dual kernel certifies the worked example."""
    from shared_slope_bab.branch_and_bound import single_spec_bab as dverify
    dev = _device()
    state = _doc_state(dev)
    verdict, info = dverify(state, np.array([1.0]), 0.0, dev, timeout=10,
                            iteration=1, bab_dtype=_DT, verbose=False)
    assert verdict == 'safe', (verdict, info)


# =============================================================================
# build_relaxation_state on a real BoundedModule (CROWN backward vs the hand-derived example)
# =============================================================================

def _doc_net_relaxation_state(device, slope_opt_spec_coeffs=None, slope_iteration=0):
    """Build the hand-derived example's (2-2-2-1) net as an auto_LiRPA BoundedModule and
    run build_relaxation_state with the doc's assumed pre-act bounds [-2,2].  Returns the
    relaxation state.  The default slope on [-2,2] is hi/(hi-lo)=1/2, exactly the doc's
    c=1/2, so build_relaxation_state must reproduce eq 14 / eq 15 exactly."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.ReLU(),
        torch.nn.Linear(2, 2), torch.nn.ReLU(),
        torch.nn.Linear(2, 1)).to(device).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1., 1.], [1., -1.]])); net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1., 1.], [1., -1.]]))
        net[2].bias.copy_(torch.tensor([-1., 0.]))
        net[4].weight.copy_(torch.tensor([[1., 1.]])); net[4].bias.copy_(torch.tensor([1.]))
    box_minimizer = torch.zeros(1, 2, device=device, dtype=_DT)
    bm = BoundedModule(net, box_minimizer, device=device)
    ptb = PerturbationLpNorm(eps=1.0,
                             x_L=torch.full((1, 2), -1., device=device, dtype=_DT),
                             x_U=torch.full((1, 2), 1., device=device, dtype=_DT))
    bx = BoundedTensor(box_minimizer, ptb)
    bm.compute_bounds(x=(bx,), method='backward')      # populate graph + names
    lower_bounds, upper_bounds = {}, {}
    for r in bm.relus:                                  # force the doc's [-2,2]
        pre = r.inputs[0].name
        sz = 1
        for d in r.inputs[0].output_shape[1:]:
            sz *= d
        lower_bounds[pre] = torch.full((1, sz), -2., device=device, dtype=_DT)
        upper_bounds[pre] = torch.full((1, sz), 2., device=device, dtype=_DT)
    ret = {'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}
    wrap = type('N', (), {})()
    wrap.net = bm; wrap.final_name = bm.final_name; wrap.device = device
    return build_relaxation_state(
        wrap, ret, bx, device, slope_opt_spec_coeffs=slope_opt_spec_coeffs,
        slope_iteration=slope_iteration, verbose=False)


def test_build_relaxation_state_reproduces_doc_eq14():
    """The CROWN-backward relaxation state matches the hand derivation:
    output_center=2, output_coeffs=[.5,.5,.5,0,.5,.5] (eq 14), and each unstable neuron's row
    matches its input-side relaxation (eq 15 for e3: e1^0 + .5 e1 + .5 e2)."""
    dev = _device()
    st = _doc_net_relaxation_state(dev)
    assert st['n_columns'] == 6 and st['n_input'] == 2
    assert abs(float(st['output_center'][0]) - 2.0) < 1e-9
    assert np.allclose(st['output_coeffs'][0].cpu().numpy(),
                       [0.5, 0.5, 0.5, 0.0, 0.5, 0.5], atol=1e-9)
    # Every unstable neuron has relu_slope=half_error=1/2 and preact_center=0.
    rows = {}
    for u in st['unstable_list']:
        assert abs(u['relu_slope'] - 0.5) < 1e-9 and abs(u['half_error'] - 0.5) < 1e-9
        assert abs(u['preact_center']) < 1e-9
        rows[u['relu_error_col']] = dict(zip(u['row_indices'].tolist(),
                                        np.round(u['row_values'], 6).tolist()))
    assert rows[2] == {0: 1.0, 1: 1.0}          # L1n1: e1^0 + e2^0
    assert rows[3] == {0: 1.0, 1: -1.0}         # L1n2: e1^0 - e2^0
    assert rows[4] == {0: 1.0, 2: 0.5, 3: 0.5}  # L2n1 = e3: e1^0 + .5e1 + .5e2 (eq 15)
    assert rows[5] == {1: 1.0, 2: 0.5, 3: -0.5}  # L2n2 = e4: e2^0 + .5e1 - .5e2


# ---------------------------------------------------------------------------
# build_relaxation_state on a multi-layer net (beyond the 2-2-2-1 doc net): the
# CROWN-backward extraction and Adam slope search give a sound, tighter root.
# ---------------------------------------------------------------------------
def test_build_relaxation_state_adam_sound_and_tightening_on_mlp():
    """On a random 5-8-8-3 MLP: the root bound is valid over the input box, and
    Adam slope search only
    improves it vs the default slope init (tightening)."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state, root_bound
    dev = _device(); torch.manual_seed(3)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 3)).to(dev).double()
    x0 = torch.zeros(1, 5, dtype=_DT, device=dev)
    bm = BoundedModule(net, x0, device=dev)
    eps = 0.5
    ptb = PerturbationLpNorm(eps=eps, x_L=x0 - eps, x_U=x0 + eps)
    bx = BoundedTensor(x0, ptb)
    bm.compute_bounds(x=(bx,), method='backward')
    lower, upper = {}, {}
    for r in bm.relus:
        pre = r.inputs[0].name
        lower[pre] = r.inputs[0].lower.detach(); upper[pre] = r.inputs[0].upper.detach()
    ret = {'lower_bounds': lower, 'upper_bounds': upper}
    wrap = type('N', (), {})(); wrap.net = bm; wrap.final_name = bm.final_name; wrap.device = dev
    spec = np.array([1.0, -1.0, 0.0])                      # a linear spec over 3 outputs
    base_bound = float(root_bound(
        build_relaxation_state(wrap, ret, bx, dev, dtype=_DT, verbose=False), spec, 0.0))
    adam = float(root_bound(
        build_relaxation_state(wrap, ret, bx, dev, dtype=_DT,
                        slope_opt_spec_coeffs=spec[None], slope_iteration=40, verbose=False), spec, 0.0))
    assert adam >= base_bound - 1e-6, (base_bound, adam)  # Adam tightens (or ties)
    # soundness: root <= true min of (spec . net(x)) over the box (sampled)
    xs = x0 + eps * (2 * torch.rand(4000, 5, dtype=_DT, device=dev) - 1)
    fx = net(xs) @ torch.tensor(spec, dtype=_DT, device=dev)
    assert adam <= float(fx.min()) + 1e-6, (adam, float(fx.min()))


def test_relaxation_state_root_is_sound_lower_bound():
    """The relaxation state root bound must be a valid lower bound of the true network
    minimum. For this net the true min over [-1,1]^2 is >= 1 (f(0,-1)=1); the
    relaxation root is -1/2 <= 1. Also sample-check no input beats
    the certified-after-split value 0."""
    from shared_slope_bab.relaxation_state import root_bound
    dev = _device()
    st = _doc_net_relaxation_state(dev)
    rb = float(root_bound(st, np.array([1.0]), 0.0))
    assert abs(rb - (-0.5)) < 1e-6, rb                 # matches doc s2.4
    # brute-force the true min of f on a grid -> must be >= the sound bound
    g = torch.linspace(-1, 1, 41, dtype=_DT)
    xx, yy = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([xx.flatten(), yy.flatten()], 1)
    h1 = torch.relu(pts @ torch.tensor([[1., 1.], [1., -1.]], dtype=_DT).T)
    h2 = torch.relu(h1 @ torch.tensor([[1., 1.], [1., -1.]], dtype=_DT).T
                    + torch.tensor([-1., 0.], dtype=_DT))
    f = h2 @ torch.tensor([1., 1.], dtype=_DT) + 1.0
    assert rb <= float(f.min()) + 1e-9                 # sound: root <= true min


def test_a_pre_activation_does_not_depend_on_its_own_relu_variable():
    """An unstable pre-activation z_k is computed before its ReLU, so its
    generators have no entry in the column of that ReLU's variable."""
    dev = _device()
    st = _doc_net_relaxation_state(dev)
    for u in st['unstable_list']:
        assert u['relu_error_col'] not in set(u['row_indices'].tolist())


def test_slope_opt_does_not_lower_root():
    """Per-neuron slope optimization can only improve (or keep) the root bound."""
    from shared_slope_bab.relaxation_state import root_bound
    dev = _device()
    spec_output_coeffs = np.array([1.0])
    base = float(root_bound(_doc_net_relaxation_state(dev), spec_output_coeffs, 0.0))
    opt = float(root_bound(_doc_net_relaxation_state(dev, slope_opt_spec_coeffs=spec_output_coeffs, slope_iteration=30), spec_output_coeffs, 0.0))
    assert opt >= base - 1e-6, (base, opt)


# =============================================================================
# Dual solver soundness
# =============================================================================

def _exact_node_lp(problem, split_sides):
    """Ground-truth node LP min via scipy: min_e path_center + path_coeffs.e s.t. split
    halfspaces, e in [-1,1].  Returns +inf if infeasible.  (scipy = oracle.)"""
    from scipy.optimize import linprog
    preact_coeffs = problem['preact_coeffs'].cpu().numpy(); preact_center_value = problem['preact_center'].cpu().numpy()
    inactive_lower_shift_value = problem['inactive_lower_shift'].cpu().numpy(); active_upper_shift_value = problem['active_upper_shift'].cpu().numpy()
    subst_scale = problem['subst_scale'].cpu().numpy(); subst_const_shift = problem['subst_const_shift'].cpu().numpy()
    spec_coeffs = problem['spec_coeffs'].cpu().numpy(); spec_center = problem['spec_center']; relu_error_col_idx = problem['relu_error_col'].cpu().numpy()
    n_columns = problem['n_columns']; depth = len(split_sides)
    path_coeffs = spec_coeffs.copy()
    path_coeffs[relu_error_col_idx[:depth]] = 0.0
    path_center = spec_center
    constraint_coeffs, constraint_bias = [], []
    for split_index, split_side in enumerate(split_sides):
        phase_sign = 1.0 if split_side == 0 else -1.0
        path_coeffs = (
            path_coeffs
            + subst_scale[split_index, split_side] * preact_coeffs[split_index])
        path_center += subst_const_shift[split_index, split_side]
        constraint_coeffs.append(phase_sign * preact_coeffs[split_index])
        constraint_bias.append(
            preact_center_value[split_index] if split_side == 0 else -preact_center_value[split_index])
        constraint_coeffs.append(-phase_sign * preact_coeffs[split_index])
        constraint_bias.append(
            -inactive_lower_shift_value[split_index]
            if split_side == 0 else -active_upper_shift_value[split_index])
    r = linprog(path_coeffs, A_ub=np.array(constraint_coeffs), b_ub=-np.array(constraint_bias),
                bounds=[(-1, 1)] * n_columns, method='highs')
    return (float(r.fun) + path_center) if r.success else float('inf')


def test_warm_logbucket_node_bounds_sound_vs_exact_lp():
    """The warm logbucket kernel is sound against the scipy LP oracle."""
    pytest.importorskip("scipy")
    from shared_slope_bab.bounding import (
        build_problem as dual_build, problem_at_depth, get_lower_bound)
    from shared_slope_bab.branch_and_bound import single_spec_bab as dverify
    dev = _device()
    state = _doc_state(dev)
    problem = dual_build(state, np.array([1.0]), 0.0, list(range(4)), dev, _DT)
    for depth in (1, 2, 3, 4):
        F = problem_at_depth(problem, depth)
        B = 1 << depth
        for bb in range(B):
            split_sides = [(bb >> j) & 1 for j in range(depth)]
            ex = _exact_node_lp(problem, split_sides)
            st = torch.tensor([split_sides], device=dev, dtype=torch.long)
            z = torch.zeros(1, depth, device=dev, dtype=_DT)
            dual = float(get_lower_bound(
                F, st, z, z, iteration=3, line_search_buckets=64)[0])
            assert dual <= ex + 1e-6, ('warm dual>exact', split_sides, dual, ex)
            if np.isfinite(ex):
                assert np.isfinite(dual), ('warm dual false +inf', split_sides, ex)
    verdict, info = dverify(
        state, np.array([1.0]), 0.0, dev, timeout=10,
        iteration=1, bab_dtype=_DT, verbose=False, line_search_buckets=64)
    assert verdict == 'safe', (verdict, info)


def test_infeasibility_prune_is_sound():
    """The factored single-constraint infeasibility prune must NOT certify a
    feasible node (regression for the 2026-06-28 sign-bug).  Construct a 1-neuron
    problem: OFF constraint e0 + preact_center <= 0 with |preact_coeffs|_1 = 1.
      preact_center=0.5  -> feasible (e0 <= -0.5); true OFF bound = -1, NOT certified.
      preact_center=2.0  -> infeasible (e0 <= -2 impossible); certified (+inf)."""
    from shared_slope_bab.bounding import build_problem as dual_build
    from shared_slope_bab.bounding import problem_at_depth, get_lower_bound
    dev = _device()

    def one_neuron(preact_center_value):
        state = dict(
            n_columns=4, n_input=3, n_out=1,
            output_coeffs=torch.tensor([[1., 0., 0., 0.]], dtype=_DT, device=dev),
            output_center=torch.tensor([0.0], dtype=_DT, device=dev),
            var_lb=torch.full((4,), -1., dtype=_DT, device=dev),
            var_hi=torch.full((4,), 1., dtype=_DT, device=dev),
            unstable_list=[dict(layer_idx=0, neuron_idx=3, preact_center=preact_center_value, lo=-2., hi=2.,
                                relu_slope=0.5, half_error=0.5, relu_error_col=3,
                                row_indices=np.array([0], np.int64),
                                row_values=np.array([1.0]))])
        return dual_build(state, np.array([1.0]), 0.0, [0], dev, _DT)

    split_sides = torch.tensor([[0]], device=dev, dtype=torch.int8)   # OFF
    z = torch.zeros(1, 1, device=dev, dtype=_DT)
    feas, _, _ = get_lower_bound(
        problem_at_depth(one_neuron(0.5), 1), split_sides.to(torch.long), z, z,
        iteration=3, line_search_buckets=64)
    assert torch.isfinite(feas).all(), feas
    infe, _, _ = get_lower_bound(
        problem_at_depth(one_neuron(2.0), 1), split_sides.to(torch.long), z.clone(), z.clone(),
        iteration=3, line_search_buckets=64)
    assert torch.isinf(infe[0]) and infe[0] > 0, infe


def test_single_spec_bab_does_not_certify_without_a_split():
    """With nothing to split the root bound is the whole answer. A property
    false at a sampled input must come back 'unknown', not fall through the
    empty BaB loop to 'safe'. An all-stable ReLU net reaches S = 0."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state
    from shared_slope_bab.branch_and_bound import single_spec_bab
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(),
                                torch.nn.Linear(6, 2)).double()
    with torch.no_grad():
        model[0].bias.fill_(10.0)           # every ReLU active over the box
    x0 = torch.zeros(1, 3, dtype=_DT)
    bm = BoundedModule(model, x0, device='cpu')
    bx = BoundedTensor(x0, PerturbationLpNorm(
        norm=float('inf'), x_L=x0 - 1.5, x_U=x0 + 1.5))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                            for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                            for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net, wrap.final_name = bm, bm.final_name
    state = build_relaxation_state(wrap, ret, bx, 'cpu', dtype=_DT, verbose=False)
    assert len(state['unstable_list']) == 0
    pts = x0 + (torch.rand(50000, 3, dtype=_DT,
                           generator=torch.Generator().manual_seed(0)) * 2 - 1) * 1.5
    with torch.no_grad():
        smin = float((model(pts) @ torch.tensor([1., -1.], dtype=_DT)).min())
    verdict, info = single_spec_bab(
        state, np.array([1., -1.]), -smin - 0.05, 'cpu',
        timeout=5, iteration=1, bab_dtype=_DT, verbose=False)
    assert verdict == 'unknown', info


def _tiny_state(model, n_in, device, eps=0.5, **kwargs):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state

    # The box is deliberately not centred on zero: the input variables move
    # around the box's centre, and a box centred on zero would hide code that
    # forgets that centre.
    x0 = torch.linspace(-1., 1., n_in, dtype=_DT, device=device).reshape(1, n_in)
    bm = BoundedModule(model, x0, device=device)
    bx = BoundedTensor(x0, PerturbationLpNorm(eps=eps, x_L=x0 - eps, x_U=x0 + eps))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach()
                            for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach()
                            for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net = bm
    wrap.final_name = bm.final_name
    wrap.device = device
    state = build_relaxation_state(wrap, ret, bx, device, dtype=_DT, verbose=False,
                            **kwargs)
    interm = {k: (v, ret['upper_bounds'][k]) for k, v in ret['lower_bounds'].items()}
    return state, bm, bx, interm


class _Residual(torch.nn.Module):
    """Two branches that merge, so both must use one numbering of the
    variables."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(4, 5)
        self.h = torch.nn.Linear(5, 5)
        self.skip = torch.nn.Linear(4, 5)
        self.out = torch.nn.Linear(5, 2)

    def forward(self, x):
        return self.out(torch.relu(self.h(torch.relu(self.a(x))) + self.skip(x)))


def test_build_relaxation_state_on_a_residual_net_is_sound():
    """Both branches give their ReLUs variables, and adding the branches is
    only sound because a column is the same variable on both sides."""
    from shared_slope_bab.relaxation_state import root_bound
    dev = _device()
    model = _Residual().to(dev).double()
    st, _, bx, _ = _tiny_state(model, 4, dev)
    spec = np.array([1.0, -1.0])
    root = float(root_bound(st, spec, 0.0))
    x_L, x_U = bx.ptb.x_L, bx.ptb.x_U
    xs = x_L + (x_U - x_L) * torch.rand(20000, 4, dtype=_DT, device=dev)
    fx = model(xs) @ torch.tensor(spec, dtype=_DT, device=dev)
    assert root <= float(fx.min()) + 1e-9, (root, float(fx.min()))
    assert st['n_columns'] > st['n_input'], st


@pytest.mark.parametrize('fail', [False, True])
def test_build_restores_the_modules_search_options(fail, monkeypatch):
    from shared_slope_bab.relaxation_state import build_relaxation_state
    device = torch.device('cpu')
    _, bm, bx, interm = _tiny_state(_Residual().double(), 4, device)
    bm.set_bound_opts({'optimize_bound_args': {
        'iteration': 3, 'lr_alpha': .07, 'early_stop_patience': 2,
        'stop_criterion_func': lambda _: True}})
    options = bm.bound_opts['optimize_bound_args']
    previous = options.copy()
    generator_options = bm.bound_opts['dynamic_forward'].copy()
    ret = {'lower_bounds': {name: lo for name, (lo, _) in interm.items()},
           'upper_bounds': {name: hi for name, (_, hi) in interm.items()}}
    wrap = type('N', (), {})()
    wrap.net, wrap.final_name = bm, bm.final_name
    if fail:
        def broken(*args, **kwargs):
            raise RuntimeError('optimizer failure')
        monkeypatch.setattr(bm, '_get_optimized_bounds', broken)
    kwargs = dict(slope_opt_spec_coeffs=np.array([[1., -1.]]),
                  slope_iteration=2, lr_slope=.03, verbose=False)
    if fail:
        with pytest.raises(RuntimeError, match='optimizer failure'):
            build_relaxation_state(wrap, ret, bx, device, **kwargs)
    else:
        state = build_relaxation_state(wrap, ret, bx, device, **kwargs)
        assert torch.isfinite(state['output_coeffs']).all()
    assert bm.bound_opts['optimize_bound_args'] is options
    assert options == previous
    assert bm.bound_opts['dynamic_forward'] == generator_options


def test_n_input_counts_the_coordinates_the_box_moves():
    """A box that fixes most coordinates has a variable only for each
    coordinate it moves, and n_input counts exactly those."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state, root_decomposition
    dev = _device()
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(dev).double()
    x0 = torch.linspace(-1., 1., 40, dtype=_DT, device=dev).reshape(1, 40)
    x_L, x_U = x0.clone(), x0.clone()
    x_L[:, :5] -= 0.5
    x_U[:, :5] += 0.5
    bm = BoundedModule(model, x0, device=dev)
    bx = BoundedTensor(x0, PerturbationLpNorm(eps=0.5, x_L=x_L, x_U=x_U))
    bm.compute_bounds(x=(bx,), method='backward')
    ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach() for r in bm.relus},
           'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach() for r in bm.relus}}
    wrap = type('N', (), {})()
    wrap.net, wrap.final_name, wrap.device = bm, bm.final_name, dev
    st = build_relaxation_state(wrap, ret, bx, dev, dtype=_DT, verbose=False)

    assert st['n_input'] == 5, st['n_input']
    assert st['n_columns'] == st['n_input'] + len(st['unstable_list'])
    diag = root_decomposition(st, np.array([1., -1.]), 0.)
    torch.testing.assert_close(
        diag['input_abs_contrib'] + diag['relu_error_abs_contrib'],
        diag['total_abs_contrib'], rtol=0, atol=1e-12)


def test_build_relaxation_state_refuses_a_graph_dynamic_forward_cannot_propagate():
    class Cat(torch.nn.Module):
        """Concat has no dynamic-forward construction."""

        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.a = torch.nn.Linear(4, 5)
            self.out = torch.nn.Linear(9, 2)

        def forward(self, x):
            return self.out(torch.cat([torch.relu(self.a(x)), x], dim=1))

    dev = _device()
    with pytest.raises(NotImplementedError):
        _tiny_state(Cat().to(dev).double(), 4, dev)


class _ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.c = torch.nn.Conv2d(1, 3, 3, stride=2, padding=1)
        self.l = torch.nn.Linear(3 * 4 * 4, 6)
        self.o = torch.nn.Linear(6, 2)

    def forward(self, x):
        h = torch.relu(self.c(x)).flatten(1)
        return self.o(torch.relu(self.l(h)))


@pytest.mark.parametrize('slope_iteration', [0, 5])
def test_the_relaxation_state_does_not_depend_on_the_slice_size(slope_iteration):
    """abcrown caps a pass at forward_max_dim variables, fewer than a real
    network has; the state read slice by slice has to be the one read in a
    single slice."""
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.relaxation_state import build_relaxation_state
    dev = _device()
    model = _ConvNet().to(dev).double()
    x0 = torch.linspace(-1., 1., 49, dtype=_DT, device=dev).reshape(1, 1, 7, 7)
    states = []
    for max_dim in (int(1e9), 7):
        bm = BoundedModule(model, x0, device=dev,
                           bound_opts={'forward_max_dim': max_dim})
        bx = BoundedTensor(x0, PerturbationLpNorm(eps=0.5, x_L=x0 - 0.5, x_U=x0 + 0.5))
        bm.compute_bounds(x=(bx,), method='backward')
        ret = {'lower_bounds': {r.inputs[0].name: r.inputs[0].lower.detach() for r in bm.relus},
               'upper_bounds': {r.inputs[0].name: r.inputs[0].upper.detach() for r in bm.relus}}
        wrap = type('N', (), {})()
        wrap.net, wrap.final_name, wrap.device = bm, bm.final_name, dev
        states.append(build_relaxation_state(
            wrap, ret, bx, dev, dtype=_DT, verbose=False,
            slope_opt_spec_coeffs=np.array([[1., -1.]]), slope_iteration=slope_iteration))
    one_slice, sliced = states
    assert sliced['n_columns'] > 7, 'slices of 7 variables have to need several passes'
    assert one_slice['n_columns'] == sliced['n_columns']
    assert one_slice['relu_order'] == sliced['relu_order']
    assert one_slice['relu_error_col'] == sliced['relu_error_col']
    for key in ('output_coeffs', 'output_center'):
        torch.testing.assert_close(sliced[key], one_slice[key], rtol=0, atol=1e-9)
    assert len(one_slice['unstable_list']) == len(sliced['unstable_list'])
    for a, b in zip(one_slice['unstable_list'], sliced['unstable_list']):
        np.testing.assert_array_equal(a['row_indices'], b['row_indices'])
        np.testing.assert_allclose(a['row_values'], b['row_values'], rtol=0, atol=1e-9)
        assert (a['layer_idx'], a['neuron_idx'], a['relu_error_col']) == (
            b['layer_idx'], b['neuron_idx'], b['relu_error_col'])
        for key in ('preact_center', 'lo', 'hi', 'relu_slope', 'half_error'):
            assert abs(a[key] - b[key]) <= 1e-9, (key, a, b)


def test_shared_slope_bab_runs_with_intermediate_clipping_disabled():
    """The state builder needs input and pre-activation bounds, not clip data."""
    import arguments
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from shared_slope_bab.branch_and_bound import shared_slope_bab

    original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    new_config['bab']['clip_n_verify']['clip_interm_domain']['enabled'] = False
    new_config['bab']['shared_slope_bab']['iteration'] = 1
    new_config['bab']['shared_slope_bab']['slope_iteration'] = 0
    new_config['bab']['shared_slope_bab']['shared_slope_iteration'] = 0
    new_config['bab']['shared_slope_bab']['line_search_buckets'] = 32
    arguments.Config = new_config
    try:
        model = torch.nn.Linear(1, 2).double()
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.tensor([1.0, 0.0], dtype=_DT))
        x0 = torch.zeros(1, 1, dtype=_DT)
        bm = BoundedModule(model, x0, device='cpu')
        bx = BoundedTensor(x0, PerturbationLpNorm(
            norm=float('inf'), x_L=x0 - 0.25, x_U=x0 + 0.25))
        bm.compute_bounds(x=(bx,), method='backward')
        wrapped = type('N', (), {})()
        wrapped.net = bm
        wrapped.final_name = bm.final_name
        wrapped.device = torch.device('cpu')
        ret = {
            'global_lb': torch.tensor([[-1.0]], dtype=_DT),
            'lower_bounds': {},
            'upper_bounds': {},
        }
        coeffs = torch.tensor([[[1.0, -1.0]]], dtype=_DT)
        rhs = torch.zeros((1, 1), dtype=_DT)

        _, visited, verdict = shared_slope_bab(
            wrapped, bx, coeffs, rhs, ret, timeout=10)

        assert verdict == 'safe'
        assert visited == 1
    finally:
        arguments.Config = original_config
