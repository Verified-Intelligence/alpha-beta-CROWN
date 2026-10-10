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
"""Attack the narrow counterexample basins in SoundnessBench residual models.

OSI initialization and multi-step-size PGD locate candidates; polishing and
float32 ONNXRuntime evaluation refine them for the official counterexample
checker. Returns the verifier's attack_with_general_specs() result tuple.
"""
import time
import numpy as np
import torch

import arguments

# ---- onnxruntime validation (matches the official VNN-COMP checker) ----
_ort_sess = None
_ort_input_name = None
_ort_onnx_path = None
_ort_intra_op = 4   # onnxruntime intra-op threads; set from the config kwarg
                    # ort_intra_op by soundness_attack(). 4 is a good default on the
                    # A10G (1 instance owns the cores); lower it (or set ORT_INTRA_OP)
                    # when running several instances in parallel to avoid CPU
                    # oversubscription.


def _get_ort():
    # The session is cached so the many polish/pattern/manifold evals within one
    # instance don't rebuild it. But it MUST be rebuilt whenever the onnx model
    # changes: a continuous run mixes models (this benchmark has both model.onnx
    # and model_residual.onnx), and reusing a session built for a *different*
    # network silently validates the counterexample against the wrong model --
    # the surrogate margin looks valid while the onnxruntime margin is garbage.
    # So key the cache on onnx_path, not on "is None".
    global _ort_sess, _ort_input_name, _ort_onnx_path
    onnx_path = arguments.Config['model']['onnx_path']
    if _ort_sess is None or onnx_path != _ort_onnx_path:
        import onnxruntime as ort
        from load_model import unzip_and_optimize_onnx
        # Build the session IDENTICALLY to abcrown's _run_onnx_outputs so our
        # validation matches the check_and_save_cex / saver computation exactly.
        so = ort.SessionOptions()
        # 4 intra-op threads: the endgame is dominated by these onnxruntime forward
        # passes and the A10G per-core CPU is slow, so a single thread is the real
        # bottleneck (OMP has ~no effect since the numpy/torch ops are not the
        # bottleneck). 4 gave the bulk of the speedup on the A10G (idx 9: 124s->110s);
        # more plateaus. Overridable via ORT_INTRA_OP for parallel sweeps.
        import os as _os
        so.intra_op_num_threads = int(_os.environ.get('ORT_INTRA_OP', str(_ort_intra_op)))
        so.inter_op_num_threads = 1
        _ort_sess = ort.InferenceSession(
            unzip_and_optimize_onnx(onnx_path).SerializeToString(),
            sess_options=so)
        _ort_input_name = _ort_sess.get_inputs()[0].name
        _ort_onnx_path = onnx_path
    return _ort_sess, _ort_input_name


def _ort_out(X_np):
    sess, name = _get_ort()
    X_np = np.ascontiguousarray(X_np.astype(np.float32))
    flat = X_np.reshape(X_np.shape[0], -1)
    return sess.run(None, {name: flat})[0].reshape(X_np.shape[0], -1)


def _parse_vnnlib_input_box(path, n_in):
    """Parse the exact (full-precision) input box from the .vnnlib text.

    The official checker reads these float64 bounds; matching them exactly lets
    us keep the counterexample strictly inside the box (not just within
    tolerance)."""
    import re
    with open(path) as _f:
        txt = _f.read()
    lb = {}; ub = {}
    for m in re.finditer(r'\(\s*assert\s*\(\s*>=\s*X_(\d+)\s+(-?[\d.eE+]+)\s*\)\s*\)', txt):
        lb[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r'\(\s*assert\s*\(\s*<=\s*X_(\d+)\s+(-?[\d.eE+]+)\s*\)\s*\)', txt):
        ub[int(m.group(1))] = float(m.group(2))
    if len(lb) != n_in or len(ub) != n_in:
        return None, None
    L64 = np.array([lb[i] for i in range(n_in)], dtype=np.float64)
    U64 = np.array([ub[i] for i in range(n_in)], dtype=np.float64)
    return L64, U64


def _largest_inner_float32_box(L64, U64):
    """Return float32 bounds (Lin, Uin) with float64(Lin) >= L64 and
    float64(Uin) <= U64, nudging via nextafter so a float32 point clamped to
    [Lin, Uin] is *strictly inside* the float64 box the checker enforces."""
    Lin = L64.astype(np.float32)
    bad = Lin.astype(np.float64) < L64
    while bad.any():
        Lin[bad] = np.nextafter(Lin[bad], np.float32(np.inf))
        bad = Lin.astype(np.float64) < L64
    Uin = U64.astype(np.float32)
    bad = Uin.astype(np.float64) > U64
    while bad.any():
        Uin[bad] = np.nextafter(Uin[bad], np.float32(-np.inf))
        bad = Uin.astype(np.float64) > U64
    return Lin, Uin


def soundness_attack(model, x, data_min, data_max, C_mat, rhs_mat, or_spec_size,
                     initialization="uniform", GAMA_loss=False, vnnlib=None,
                     time_budget=100.0, margin_buffer=5e-8, seed_offset=0,
                     endgame_cands=1, ils_hops=8, ils_keep=128, ils_hop_steps=600,
                     ils_jump=0.4, ils_jumps=(0.1, 0.2, 0.4, 0.6),
                     drill_steps=3000, drill_lr=1e-4, robust_target=2e-7,
                     ils_check_every=4, ort_intra_op=4, verbose=True):
    """Custom attack_func; signature matches attack_with_general_specs.

    The extra keyword arguments are set from the config via Customized(), e.g.
        attack_func: Customized("custom_soundness_attack", "soundness_attack",
                                time_budget=110.0)

    :param time_budget: wall-clock budget (s) for the whole attack. Keep it
        below the per-instance timeout minus a small margin.
    :param margin_buffer: required onnxruntime worst-margin. The official
        VNN-COMP checker recomputes the (multi-threaded) onnxruntime output and
        accepts a strict violation; empirically a worst-margin of +7.5e-8 is
        accepted and our session matches it to ~2.5e-9, so 5e-8 is a safe floor.
    :param seed_offset: shift the deterministic per-round restart-seed sequence
        so a stochastically-missed instance can be retried with a fresh draw.
    """
    device = x.device
    cfg = arguments.Config["attack"]
    steps = int(cfg["pgd_steps"])
    restarts = int(cfg["pgd_restarts"])
    lr_decay = float(cfg["pgd_lr_decay"])
    time_budget = float(time_budget)
    buffer = float(margin_buffer)
    seed_offset = int(seed_offset)
    # How many polished entries to run the precision endgame (pattern +
    # manifold) from each round. 1 is best: the manifold/pattern are expensive,
    # so >1 trades PGD rounds for within-round entry diversity, which is a bad
    # trade on hard instances (more rounds with fresh seeds find better basins
    # than more candidates from a single PGD run, which are near-duplicates).
    endgame_cands = max(1, int(endgame_cands))
    # Iterated-local-search hops: when standard PGD stays stuck in a "decoy"
    # basin (worst-margin far below 0), jump the best points out by a large
    # perturbation and re-optimize. The big jump escapes decoys that trap
    # gradient PGD and finds the true (well-hidden) basins. Adaptive: only runs
    # when the round did NOT get close to the basin.
    ils_hops = int(ils_hops); ils_keep = int(ils_keep)
    ils_hop_steps = int(ils_hop_steps); ils_jump = float(ils_jump)
    # Deep-drill (ported from the strict-cex-polish branch): after the endgame
    # reaches the basin, an aggressive gradient drill pushes the worst margin DEEP
    # past 0 so the saved cex is robust to the checker's float32 onnxruntime
    # recompute (not a razor-thin +5e-8), AND it converts a round-1 near-miss into
    # a valid cex so hard instances solve in ONE round instead of needing an ILS
    # re-seed round. robust_target is the surrogate worst-margin we drill toward.
    drill_steps = int(drill_steps); drill_lr = float(drill_lr)
    robust_target = float(robust_target); ils_check_every = max(1, int(ils_check_every))
    # Publish the ort thread count (config kwarg) for _get_ort. Rebuild the cached
    # session if it changed so the new setting takes effect.
    global _ort_intra_op, _ort_sess
    if int(ort_intra_op) != _ort_intra_op:
        _ort_intra_op = int(ort_intra_op); _ort_sess = None
    # Different hard instances want different escape distances (idx 8 wants
    # 0.6*eps, idx 9 wants 0.1*eps, others 0.4*eps), so cycle the jump scale
    # across hops -- one config then escapes them all in a single round.
    if ils_jumps is None:
        ils_jumps = [ils_jump]
    elif isinstance(ils_jumps, str):
        ils_jumps = [float(t) for t in ils_jumps.split(",")]
    else:
        ils_jumps = [float(t) for t in ils_jumps]
    # multi-alpha fractions of eps (absolute step = frac * eps). A finer grid
    # covers the per-instance variation in basin width (some basins are only
    # reachable at a specific step size).
    alpha_fracs = [0.0035, 0.005, 0.0075, 0.01, 0.015, 0.025]
    osi_iters = 50
    kappa = 1e-5
    t_start = time.time()
    # Search rounds share 90% of the attack budget; the remaining 10% belongs
    # to the final ORT backstop. Every iterative refinement respects its phase.
    round_budget = time_budget * 0.9
    phase_deadline = t_start + round_budget

    def deadline_reached(deadline=None):
        limit = phase_deadline if deadline is None else min(phase_deadline, deadline)
        return time.time() >= limit

    def log(msg):
        if verbose:
            print(f"[soundness {time.time()-t_start:6.1f}s] {msg}", flush=True)

    num_or = or_spec_size.shape[0]
    # This benchmark uses a single OR clause (conjunction of ANDs). Attack it.
    assert num_or == 1, f"custom_soundness_attack supports single-OR specs (got {num_or})"
    in_shape = x.shape[1:]
    n_in = int(np.prod([int(s) for s in in_shape]))
    n_out = C_mat.shape[-1]
    # ------------------------- DIMENSION GLOSSARY -------------------------
    # Four axes recur throughout this file; tensor-shape comments name them:
    #   n_in     (= 128) flattened input dimension. The Linf box lives in R^n_in.
    #   n_out    (= 384) raw network outputs. y = model(x) has width n_out.
    #   n_spec   (=  12) AND-inequalities in the specification (a.k.a. num_spec).
    #            The property is a CONJUNCTION; a counterexample must violate ALL
    #            n_spec of them at once.
    #   restarts (= 512) parallel PGD restarts. PGD optimizes this many independent
    #            candidate inputs simultaneously; it is the batch axis that appears
    #            as B / N / Rb / k in the helpers below.
    # MARGIN CONVENTION (see abc_margin): for spec j, g_j = rhs_j - (C y)_j.
    #   g_j >= 0  <=>  spec j is violated  <=>  good for us (counterexample side).
    #   The "worst margin" of a point is  min_j g_j  over the n_spec specs. A point
    #   is a valid counterexample iff its worst margin >= 0; we actually require
    #   >= `buffer` (5e-8) so the official onnxruntime checker cannot round it
    #   negative. The entire attack is one big "maximize the worst margin" search.

    Lnp0 = data_min[0].reshape(-1).detach().cpu().numpy().astype(np.float32)
    Unp0 = data_max[0].reshape(-1).detach().cpu().numpy().astype(np.float32)
    # Confine the attack to the largest float32 box STRICTLY inside the exact
    # float64 vnnlib bounds. Otherwise a corner coordinate clamps to float32(L)
    # which can be a fraction of a ULP below the float64 L the official checker
    # enforces -> "correct_up_to_tolerance" (input out of box) instead of
    # "correct". This makes the saved input strictly in-box.
    try:
        _vpath = arguments.Config['specification']['vnnlib_path']
        L64, U64 = _parse_vnnlib_input_box(_vpath, n_in)
    except Exception:
        L64 = U64 = None
    if L64 is not None:
        Lnp0, Unp0 = _largest_inner_float32_box(L64, U64)
    else:
        # Fallback: nudge the float32 bounds 2 ULPs inward.
        Lnp0 = np.nextafter(np.nextafter(Lnp0, np.float32(np.inf)), np.float32(np.inf))
        Unp0 = np.nextafter(np.nextafter(Unp0, np.float32(-np.inf)), np.float32(-np.inf))
    L = torch.tensor(Lnp0, device=device, dtype=torch.float32)
    U = torch.tensor(Unp0, device=device, dtype=torch.float32)
    # C_mat is [num_spec, n_out] (already flattened across ORs), rhs_mat is [num_spec].
    Cm = C_mat.to(device=device, dtype=torch.float32)         # [num_spec, n_out]
    rh = rhs_mat.to(device=device, dtype=torch.float32)       # [num_spec]
    eps_v = (U - L) / 2.0
    Cnp = Cm.detach().cpu().numpy().astype(np.float64)
    rhnp = rh.detach().cpu().numpy().astype(np.float64)
    epsn = eps_v.detach().cpu().numpy()
    Lnp = L.detach().cpu().numpy()
    Unp = U.detach().cpu().numpy()

    for p in model.parameters():
        p.requires_grad_(False)

    n_spec = int(rh.shape[0])
    _box_src = "exact-vnnlib" if L64 is not None else "ulp-nudged-fallback"
    log(f"=== soundness_attack: dim={n_in} outputs={n_out} spec=AND of {n_spec} "
        f"ineqs | eps[min/mean/max]={epsn.min():.3f}/{epsn.mean():.3f}/{epsn.max():.3f}")
    log(f"    restarts={restarts} steps={steps} lr_decay={lr_decay} init={initialization} "
        f"alpha_fracs={alpha_fracs}")
    log(f"    time_budget={time_budget}s buffer={buffer:.1e} seed_offset={seed_offset} "
        f"box={_box_src}")

    def fwd(xb):
        # Forward pass on the torch surrogate.  xb: [B, n_in] -> ret: [B, n_out].
        return model(xb.view(-1, *in_shape)).view(xb.shape[0], n_out)

    def abc_margin(xb):
        # Per-spec margin on the TORCH surrogate (the onnx2pytorch model). This is
        # the differentiable objective used by every gradient step in the file.
        # abcrown convention: a counterexample needs (C y - rhs) <= 0 for EVERY
        # spec, i.e. g = rhs - C y >= 0 for every spec; we maximize min_j g_j.
        #   xb : [B, n_in]    B candidate inputs (a restart pool or an endgame batch)
        #   y  : [B, n_out]   network outputs
        #   ret: [B, n_spec]  margin per spec; a >=0 entry means that spec is violated.
        y = fwd(xb)
        return rh.unsqueeze(0) - y @ Cm.t()        # [B, num_spec], >=0 satisfied

    def ort_worst(X_np):
        # The SAME worst-margin as abc_margin, but computed on the EXACT float32
        # onnxruntime model -- the ground truth the official VNN-COMP checker
        # uses. The torch surrogate and onnxruntime agree only to ~1e-5 (with
        # TF32 off); the planted basin is thinner than that gap, so EVERY
        # accept/reject decision is taken here, never on the torch margin.
        #   X_np: [N, n_in]  ->  g: [N, n_spec]  ->  ret: [N]  worst margin per input
        y = _ort_out(X_np).astype(np.float64)       # [N, n_out]
        g = rhnp[None] - y @ Cnp.T                   # [N, num_spec]
        return g.min(axis=1)

    def ort_worst1(x1):
        # onnxruntime worst margin of a single point x1: [n_in] -> float.
        return float(ort_worst(x1.reshape(1, -1))[0])

    # ---------------- polish: maximize worst (min) margin -----------------
    def polish(X0, psteps=1200, lr=3e-4, return_all=False, deadline=None):
        # Batched Adam ascent that drives points to the DEEPEST part of the basin
        # (the largest worst-margin), giving the onnxruntime endgame a buffer to
        # work with. Whereas the main PGD uses sign steps to FIND the basin, this
        # uses smooth Adam to settle to its bottom. Tracks the best (deepest)
        # point each row ever reaches (the iterate oscillates), and returns the
        # single global best, or -- with return_all -- the best-per-row batch so
        # the caller can keep several diverse polished candidates.
        #   X0  : [B, n_in]   B start points, polished independently in one batch
        #   ret : (x[n_in], margin float) for the deepest row, OR
        #         ([B, n_in], [B]) per-row best point + margin when return_all.
        X0 = np.atleast_2d(X0).astype(np.float32)
        xp = torch.tensor(X0, device=device).clone().requires_grad_(True)
        opt = torch.optim.Adam([xp], lr=lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
        # Include the starting points so a deadline before the first step still
        # returns candidates with their actual margins, not sentinel scores.
        best_row = abc_margin(xp).min(dim=1).values.detach()  # [B] best worst-margin/row
        best_rowx = xp.detach().clone()                     # [B, n_in] its argmax point
        for s in range(psteps):
            if deadline_reached(deadline):
                break
            opt.zero_grad()
            g = abc_margin(xp)                               # [B, n_spec]
            # The hard worst-margin min_j g_j is non-smooth (kinks where the
            # binding spec switches), so ascend a temperature-T SOFT-min instead:
            #   -T * logsumexp(-g / T)  ->  min_j g_j   as T -> 0.
            # Anneal T from 3e-3 down to 1e-4 so the objective starts smooth
            # (averages over specs) and gradually focuses on the single deepest
            # binding spec.
            T = max(1e-4, 3e-3 * (0.5 ** (s / 300.0)))
            softmin = -T * torch.logsumexp(-g / T, dim=1)   # [B] smooth worst margin
            # Maximize the soft worst-margin (negated -> a loss), plus a hinge
            # that strongly (x10) pushes any still-negative spec back up toward 0.
            loss = -(softmin.sum()) - 10.0 * torch.clamp(g, max=0).sum()
            loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                xp.clamp_(L, U)                             # project back into the box
                wm = abc_margin(xp).min(dim=1).values       # [B] true (hard) worst margin
                imp = wm > best_row                         # [B] rows that improved
                best_rowx = torch.where(imp.unsqueeze(1), xp.detach(), best_rowx)
                best_row = torch.maximum(best_row, wm)
        rowx = best_rowx.cpu().numpy(); rowm = best_row.cpu().numpy()
        if return_all:
            return rowx, rowm
        bi = int(rowm.argmax())
        return rowx[bi], float(rowm[bi])

    def deep_drill(X0, steps=None, lr=None, robust=None):
        # Aggressive Adam drill (ported from the strict-cex-polish branch's drilling
        # polish). Same soft-worst-margin ascent as polish(), but many more steps, a
        # stronger hinge that forces EVERY spec strictly positive, and a colder final
        # temperature so it focuses on the single deepest binding spec. Two jobs:
        #   (1) CONVERT a round-1 near-miss (worst-margin a little below 0) into a
        #       valid counterexample, so a hard instance solves in ONE round; and
        #   (2) DEEPEN a cex well past 0 to a robust margin, so it survives the
        #       official checker's float32 onnxruntime recompute.
        # Surrogate-only (fast, gradient-based); the caller ONNX-validates/refines
        # the result -- so if the pytorch/onnx gap is large (e.g. idx 31) the drilled
        # point is simply not adopted and the onnxruntime ladder handles it instead.
        #   X0     : [B, n_in]   near-miss / basin start points
        #   robust : stop once every row's surrogate worst-margin reaches this
        #   ret    : ([B, n_in] best-per-row points, [B] their surrogate worst-margins)
        steps = drill_steps if steps is None else int(steps)
        lr = drill_lr if lr is None else float(lr)
        robust = robust_target if robust is None else float(robust)
        X0 = np.atleast_2d(X0).astype(np.float32)
        xp = torch.tensor(X0, device=device).clone().requires_grad_(True)
        opt = torch.optim.Adam([xp], lr=lr)
        best_row = abc_margin(xp).min(dim=1).values.detach()  # [B] best worst-margin/row
        best_rowx = xp.detach().clone()                     # [B, n_in]
        prev_best = -1e30; last_gain = 0    # patience: bail FAST on a stuck decoy so
                                            # the drill only spends its budget where it
                                            # is actually converting/deepening
        for s in range(steps):
            if deadline_reached():
                break
            opt.zero_grad()
            g = abc_margin(xp)                              # [B, n_spec]
            # anneal soft-min temperature from 1e-3 to a cold 3e-5 floor (colder than
            # polish's 1e-4) so the drill sharpens onto the single deepest spec
            T = max(3e-5, 1e-3 * (0.5 ** (s / 700.0)))
            softmin = -T * torch.logsumexp(-g / T, dim=1)  # [B] smooth worst margin
            # maximize the soft worst-margin; the x20 hinge (vs polish's x10) drives
            # ANY still-negative spec strictly positive -> deep, all-rows violation
            loss = -(softmin.sum()) - 20.0 * torch.clamp(g, max=0).sum()
            loss.backward(); opt.step()
            with torch.no_grad():
                xp.clamp_(L, U)
                wm = abc_margin(xp).min(dim=1).values      # [B] true (hard) worst margin
                imp = wm > best_row
                best_rowx = torch.where(imp.unsqueeze(1), xp.detach(), best_rowx)
                best_row = torch.maximum(best_row, wm)
            if s % 100 == 0:
                cur = float(best_row.max())                # the BEST candidate in the batch
                if cur >= robust:
                    break                                  # some candidate is robustly deep -> keep it
                # Batch parallelism dissolves the decoy-vs-near-miss guess: a genuine
                # near-miss in the batch crosses 0 EARLY, so if NOT ONE of the ~12
                # candidates is a cex by now they are all decoys -> bail and let the
                # ILS escape (this is the ILS-needing instances like idx 6). A late
                # single-point converge no longer traps us, because some other point
                # in the batch already converged if the basin was reachable.
                if cur > prev_best + 1e-9:
                    prev_best = cur; last_gain = s
                elif s - last_gain >= 500:
                    break                                  # whole batch stalled (all decoys/ceiling)
        return best_rowx.cpu().numpy(), best_row.cpu().numpy()

    # ----- exact float32-ORT precision search (greedy coord + grad line) ---
    def ort_pattern(x0, sweeps=24, target=None, label="pattern",
                    hs=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)):
        # Greedy "pattern search" that MAXIMIZES the worst onnxruntime margin
        # directly. polish/PGD optimize the torch surrogate, but the surrogate and
        # onnxruntime differ by ~1e-5, which is larger than the basin depth -- so
        # the last leg must climb on the true onnxruntime margin. onnxruntime is a
        # black box with no gradient, so this is GRADIENT-FREE: each sweep probes
        # a cloud of nearby points, scores them all with one onnxruntime batch
        # eval, and greedily steps to the best. The torch gradient is used only as
        # one extra search DIRECTION (a hint), never for the accept decision.
        #   x0   : [n_in] start point (already near the basin, e.g. from polish)
        #   hs   : per-coordinate probe sizes, as fractions of eps
        #   ret  : (cur[n_in], onnxruntime worst-margin float)
        cur = x0.astype(np.float32).copy()
        cm0 = cm = ort_worst1(cur); used = 0
        for _ in range(sweeps):
            if deadline_reached():
                break
            used += 1
            cands = [cur[None]]                            # list of [*, n_in] blocks
            for h in hs:
                # step = diag(h): [n_in, n_in]. cur[None] + step*epsn[None] is then
                # [n_in, n_in] whose row i is `cur` with ONLY coordinate i nudged by
                # +h*eps_i. So this block = "perturb each coordinate one at a time".
                step = np.zeros((n_in, n_in), np.float32); np.fill_diagonal(step, h)
                cands.append(np.clip(cur[None] + step * epsn[None], Lnp, Unp).astype(np.float32))
                cands.append(np.clip(cur[None] - step * epsn[None], Lnp, Unp).astype(np.float32))  # and the -h side
            # Extra candidate direction: the torch soft-min ascent direction d,
            # sampled at several step sizes. Cheap hint; onnxruntime still judges.
            xt = torch.tensor(cur, device=device).reshape(1, -1).requires_grad_(True)
            g = abc_margin(xt); sm = -1e-3 * torch.logsumexp(-g / 1e-3, dim=1)
            grd, = torch.autograd.grad(sm.sum(), xt)
            d = grd.reshape(-1).cpu().numpy(); nrm = np.linalg.norm(d)
            if nrm > 0:
                d = d / nrm
                ts = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])[:, None]   # [7,1] line steps
                cands.append(np.clip(cur[None] + ts * d[None] * epsn[None], Lnp, Unp).astype(np.float32))
            allc = np.concatenate(cands, 0)               # [num_cand, n_in] all probes
            oms = ort_worst(allc)                         # [num_cand] onnxruntime margins
            j = int(oms.argmax())
            if oms[j] > cm + 1e-12:
                cm = oms[j]; cur = allc[j].copy()         # greedy step to the best probe
            else:
                break                                     # no probe improved -> local max
            if target is not None and cm >= target:
                break                                     # reached the buffer; done
        if verbose and cm > cm0 + 1e-12:
            log(f"    {label}: ORT margin {cm0:+.3e} -> {cm:+.3e} ({used} sweeps)")
        return cur, cm

    def ort_random(x0, iters=50, B=192):
        # A random-search SIBLING of ort_pattern, kept as an available tool (the
        # default endgame uses ort_pattern + manifold_search and does not call
        # this). Same idea -- greedily climb the worst onnxruntime margin with no
        # onnxruntime gradient -- but probe with random directions instead of
        # axis-aligned coordinates. Each iteration scores B candidates:
        #   line[B/2]: along the torch soft-min direction d, log-uniform step
        #              sizes 10^[-6,-2.5] * eps;
        #   rand[B/2]: isotropic Gaussian directions, log-uniform scales
        #              10^[-6,-3] * eps (multi-coordinate moves).
        #   x0 -> ret: (cur[n_in], onnxruntime worst-margin float); stops once >=0.
        cur = x0.astype(np.float32).copy(); cm = ort_worst1(cur)
        for _ in range(iters):
            if deadline_reached():
                break
            xt = torch.tensor(cur, device=device).reshape(1, -1).requires_grad_(True)
            g = abc_margin(xt); sm = -1e-3 * torch.logsumexp(-g / 1e-3, dim=1)
            grd, = torch.autograd.grad(sm.sum(), xt)
            d = grd.reshape(-1).cpu().numpy(); nrm = np.linalg.norm(d)
            if nrm > 0:
                d = d / nrm
            line = np.clip(cur[None] + (10 ** np.random.uniform(-6, -2.5, (B // 2, 1))) * d[None] * epsn[None], Lnp, Unp).astype(np.float32)
            rand = np.clip(cur[None] + (10 ** np.random.uniform(-6, -3, (B // 2, 1))) * np.random.randn(B // 2, n_in) * epsn[None], Lnp, Unp).astype(np.float32)
            cand = np.concatenate([line, rand], 0)        # [B, n_in]
            oms = ort_worst(cand); j = int(oms.argmax())  # [B] onnxruntime margins
            if oms[j] > cm: cm = oms[j]; cur = cand[j].copy()
            if cm >= 0: break
        return cur, cm

    def manifold_search(x0, iters=150, B=256, target=None, deadline=None):
        # For instances where PGD/polish REACH the basin but its worst margin tops
        # out at ~0 on onnxruntime (idx 31, 47): the bottom of the basin is a thin
        # MANIFOLD, the set of points where the few "binding" specs sit at the
        # worst margin. Along that manifold the binding torch margins barely move,
        # but the tiny torch<->onnxruntime gap does -- so somewhere on it the
        # onnxruntime margin is positive. We walk the manifold and onnxruntime-
        # evaluate to find that point.
        #
        # Each iteration:
        #   1. find the binding specs   (those within 5e-5 of the worst margin);
        #   2. build their Jacobian J = d(margin)/dx  -> [b, n_in];
        #   3. project random directions onto the NULL-SPACE of J, i.e. the basin's
        #      tangent space  proj = rdir - (rdir J^T)(J J^T)^-1 J  =  rdir (I - J^+ J).
        #      Moving along proj keeps the binding margins ~unchanged (stays in the
        #      basin) while varying everything else;
        #   4. step B candidates along proj at log-uniform scales, score on
        #      onnxruntime, keep the best (occasionally re-settle it with polish).
        if target is None: target = buffer
        cur = x0.astype(np.float32).copy(); cm0 = cm = ort_worst1(cur); best = cur.copy()
        nbind0 = 0; it = 0; floor_since = None
        for it in range(iters):
            # Time-box: the manifold walk is ~0.7s/iter on the A10G, so 140 iters can
            # eat ~95s and STARVE the ILS (idx 6/47 reach a shallow non-basin point,
            # the manifold grinds the whole budget from it and never solves, and the
            # ILS that would have found the real basin never runs). Stop at the caller's
            # deadline so the ILS keeps its share of the budget.
            if deadline_reached(deadline):
                break
            # Floor-grind cap: once the walk is at/near the float32 quantization floor
            # (worst margin within ~2.5 ULP of 0), it has REACHED the basin bottom. If a
            # few dozen more iters do not carry the true min across +1 ULP, they never
            # will -- 0.0 is a local max that does not cross even at a 200s budget (idx
            # 8/37). The counterexample instead comes from a FRESH PGD restart landing a
            # DIFFERENT, crossable basin (idx 47 crosses in round 2). So bail after
            # `floor_grind` iters at the floor and hand the budget back to the round loop
            # for another restart. A genuinely manifold-crossable instance (idx 31)
            # crosses well within this window, so it is unaffected.
            if cm > -1.2e-6:
                if floor_since is None:
                    floor_since = it
                elif it - floor_since >= 50:
                    break
            else:
                floor_since = None      # dropped back below the floor band; reset
            xt = torch.tensor(cur, device=device).reshape(1, -1).requires_grad_(True)
            g = abc_margin(xt).reshape(-1)                             # [n_spec]
            thr = g.min().item() + 5e-5                                # "binding" cutoff
            bind = torch.where(g <= thr)[0].tolist()                   # b binding spec indices
            nbind0 = len(bind)
            J = []
            for k in bind:
                gk, = torch.autograd.grad(g[k], xt, retain_graph=True)  # d g_k / dx : [n_in]
                J.append(gk.reshape(-1))
            Jn = torch.stack(J, 0).cpu().numpy().astype(np.float64)   # [b, n_in] binding Jacobian
            JJt = Jn @ Jn.T + 1e-9 * np.eye(Jn.shape[0])              # [b, b] (+ ridge for stability)
            Pinv = np.linalg.solve(JJt, Jn)                            # (J J^T)^-1 J : [b, n_in]
            rdir = np.random.randn(B, n_in)                           # [B, n_in] random directions
            proj = rdir - (rdir @ Jn.T) @ Pinv                        # [B, n_in] null-space part (basin tangent)
            proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
            scales = 10 ** np.random.uniform(-3.0, -1.0, size=(B, 1))  # [B, 1] step sizes (frac of eps)
            cand = np.clip(cur[None] + scales * proj * epsn[None], Lnp, Unp).astype(np.float32)  # [B, n_in]
            oms = ort_worst(cand); j = int(oms.argmax())              # [B] onnxruntime margins
            cand_best, ob = cand[j], oms[j]
            # Re-settle the best candidate to the basin bottom when it improves so the
            # walk keeps the worst margin deep. This 300-step polish at 30% frequency
            # was the manifold's dominant cost (~0.7s/iter, ~95s/call on the A10G);
            # 120 steps only when we actually improved (plus a rare 8% probe) gives the
            # same walk quality at a fraction of the GPU time, so far more iterations
            # fit the budget -- which is what the manifold-bound tail (37, 47) needs.
            if oms[j] > cm or np.random.rand() < 0.08:
                px, _ = polish(cand[j], psteps=120, deadline=deadline); opx = ort_worst1(px)
                if opx > ob: cand_best, ob = px, opx
            if ob > cm:
                cm = ob; cur = cand_best.copy(); best = cur.copy()
                if verbose and it % 25 == 0:
                    log(f"      manifold it {it}: ORT margin -> {cm:+.3e} "
                        f"(binding={nbind0})")
            if cm >= target:
                break
            if deadline_reached(deadline):
                break
        if verbose and cm > cm0 + 1e-12:
            log(f"    manifold: ORT margin {cm0:+.3e} -> {cm:+.3e} "
                f"({it+1} iters, binding={nbind0})")
        return best, cm

    # --------------------------- main PGD loop ----------------------------
    g_rng = torch.Generator(device=device); g_rng.manual_seed(int(arguments.Config['general']['seed']))
    # Per-restart PGD step size = (cyclic multi-alpha fraction) * eps : [restarts, n_in].
    # Restart r uses alpha_fracs[r % len], so the pool spans several step scales at
    # once -- crucial because each instance's basin is reachable only near a
    # particular step size, and abcrown's default alpha=eps/4 is ~25x too coarse.
    step0 = torch.tensor([alpha_fracs[r % len(alpha_fracs)] for r in range(restarts)],
                         device=device, dtype=torch.float32).unsqueeze(1) * eps_v.unsqueeze(0)

    best_ort = -1e9; best_ort_x = None; rounds = 0; success = False
    best_stage = "none"   # which stage produced the current best margin
    def update_best(val, xv, stage):
        nonlocal best_ort, best_ort_x, best_stage
        if val > best_ort:
            best_ort = val; best_ort_x = xv.copy(); best_stage = stage
            return True
        return False
    topk = 32
    entry_pool = []   # list of (ort_margin, x) best near-basin polished entries
    def add_entry(mval, xv):
        entry_pool.append((float(mval), xv.copy()))
        entry_pool.sort(key=lambda e: -e[0])
        del entry_pool[6:]
    b1, b2, e = 0.9, 0.999, 1e-8

    def mini_pgd(xb_init, nsteps):
        # The PGD used INSIDE each ILS hop to re-optimize the jumped points. Same
        # recipe as the main loop's PGD: Adam moments but a SIGN update (the
        # .sign() makes every step a fixed-size eps-fraction, robust to the wildly
        # varying gradient scale near the trapdoor). Tracks the best (deepest)
        # point each row reaches.
        #   xb_init: [Rb, n_in]  ->  ret: (bwm[Rb] best worst-margin/row,
        #                                   bxb[Rb, n_in] its argmax point/row)
        Rb = xb_init.shape[0]
        # Per-row step size = (cyclic multi-alpha fraction) * eps : [Rb, n_in].
        s0 = torch.tensor([alpha_fracs[r % len(alpha_fracs)] for r in range(Rb)],
                          device=device, dtype=torch.float32).unsqueeze(1) * eps_v.unsqueeze(0)
        xb = xb_init.detach().requires_grad_(True)
        mm = torch.zeros_like(xb); vv = torch.zeros_like(xb); st = s0.clone()   # Adam m, v, step
        bwm = torch.full((Rb,), -1e30, device=device); bxb = xb.detach().clone()
        for t in range(1, nsteps + 1):
            if deadline_reached():
                break
            g = abc_margin(xb)                              # [Rb, n_spec]
            with torch.no_grad():
                wmt = g.min(dim=1).values                   # [Rb] worst margin now
                imp = wmt > bwm
                bxb = torch.where(imp.unsqueeze(1), xb.detach(), bxb)   # keep best-ever per row
                bwm = torch.maximum(bwm, wmt)
            # Hinge: only reward specs up to kappa, so optimization focuses on the
            # still-unsatisfied (most negative) specs rather than over-satisfying.
            loss = -torch.clamp(g, max=kappa).sum()
            gx, = torch.autograd.grad(loss, xb)
            with torch.no_grad():
                mm.mul_(b1).add_(gx, alpha=1 - b1); vv.mul_(b2).addcmul_(gx, gx, value=1 - b2)
                bc1 = 1 - b1 ** t; bc2 = 1 - b2 ** t        # Adam bias correction
                xb.sub_((st / bc1) * (mm / (vv.sqrt() / (bc2 ** 0.5) + e)).sign())   # SIGN of Adam dir
                xb.clamp_(L, U); st *= lr_decay             # project to box, decay the step
            xb = xb.detach().requires_grad_(True)
            if time.time() - t_start > round_budget:
                break
        with torch.no_grad():
            wmt = abc_margin(xb).min(dim=1).values          # fold in the final iterate too
            imp = wmt > bwm
            bxb = torch.where(imp.unsqueeze(1), xb.detach(), bxb)
            bwm = torch.maximum(bwm, wmt)
        return bwm, bxb

    def do_endgame(bxb, bwm, light=False):
        # light=True skips the SLOW onnxruntime manifold walk -- used for the cheap
        # in-ILS "are we done yet?" probes so the ILS can stop the moment the batched
        # drill has cracked a basin, instead of always running all its hops.
        # Turn the best PGD points into an onnxruntime-VALID counterexample via a
        # precision ladder of increasing cost:
        #   pgd-direct -> polish -> ort_pattern -> manifold_search
        # Each rung is only reached if the cheaper ones did not already clear the
        # buffer. update_best() records the global best; add_entry() stashes deep
        # near-basin points for the end-of-budget backstop.
        #   bxb: [restarts, n_in]  best-ever point per restart (from PGD/ILS)
        #   bwm: [restarts]        its surrogate worst-margin
        #   ret: tvn[k]            top-k surrogate margins (for the caller's logging)
        # Existing candidates may still be validated in the reserved portion;
        # skip their evaluation only when the total attack budget is exhausted.
        if time.time() >= t_start + time_budget:
            return np.empty(0)
        with torch.no_grad():
            tv, ti = torch.topk(bwm, min(topk, restarts))   # [k] the k deepest restarts
            cand = bxb[ti].detach().cpu().numpy()            # [k, n_in]
        tvn = tv.cpu().numpy()                               # [k] their surrogate margins
        # Manifold budget cap. If PGD is still stuck in a decoy (bwm < -1e-5) the ILS
        # will run this round, so the (slow, ~0.7s/iter on A10G) manifold must NOT
        # grind the whole round from a shallow drilled point and starve it -- that is
        # the idx 6/8 A10G regression (manifold ran to the full round budget, then the
        # ILS was skipped for lack of time). Cap it to 40% of the round there; when PGD
        # already reached the basin (bwm > -1e-5, no ILS pending) give it the full budget.
        # (The manifold ALSO self-limits its floor-grinding via `floor_grind` -- see
        # manifold_search -- so a non-crossable 0.0 hands the budget back for a fresh
        # round instead of grinding the whole round away, which is what lands idx 47.)
        m_dl = t_start + round_budget * (0.4 if float(bwm.max()) < -1e-5 else 1.0)
        # "near": close enough that the raw surrogate point may ALREADY be an
        # onnxruntime-valid cex -> just validate the top restarts on onnxruntime.
        near = tvn > -1e-4
        if near.any():
            oms = ort_worst(cand[near]); j = int(oms.argmax())
            if update_best(oms[j], cand[near][j], "pgd-direct"):
                log(f"    pgd-direct: top restart ORT worst-margin={oms[j]:+.3e}")
        # The round cutoff stops further optimization, not validation of a
        # candidate already found by the final PGD step.
        if deadline_reached():
            return tvn
        # "near2": within polishing range of the basin. Run the full precision
        # ladder from the closest restarts.
        near2 = tvn > -1e-3
        if near2.any() and endgame_cands <= 1:
            # Default path: one best candidate through polish -> pattern -> manifold.
            if drill_steps > 0:
                # Batch deep-drill ALL top polished candidates at once (one GPU op,
                # ~as cheap as drilling a single point). Genuine near-misses in the
                # batch convert past 0 while decoys stay stuck, so we simply keep the
                # candidate that converges deepest -- no per-point decoy-vs-near-miss
                # decision to get wrong. This is what lets a hard instance solve in
                # ONE round without an ILS re-seed.
                pool_pts, pool_m = polish(cand[near2][:12], return_all=True)  # [k,n_in],[k]
                bi0 = int(pool_m.argmax())
                px, ptm = pool_pts[bi0], float(pool_m[bi0])   # best polished (as before)
                om2 = ort_worst1(px)
                # ADDITIVE fast-convert: batch deep-drill ALL polished candidates and
                # adopt a drilled point ONLY IF it beats the polished one on the
                # ONNXRUNTIME margin (the checker's metric). This can only HELP and
                # never disrupts the polish->pattern->manifold ladder: a near-miss
                # instance (idx 5/8/38/44) converts to a deep cex here and skips the
                # slow manifold, while everything else just doesn't adopt the drill and
                # proceeds exactly as before.
                dpts, _ = deep_drill(pool_pts)
                doms = ort_worst(dpts); dbi = int(doms.argmax())
                if float(doms[dbi]) > om2:
                    px, om2 = dpts[dbi], float(doms[dbi])
            else:
                px, ptm = polish(cand[near2][:8]); om2 = ort_worst1(px)
            log(f"    polish+drill: surrogate worst-margin->{ptm:+.3e}  ORT worst-margin={om2:+.3e}")
            update_best(om2, px, "polish")
            cx, cm = ort_pattern(px, target=buffer, label="pattern")
            if cm > om2: om2 = cm; px = cx
            update_best(om2, px, "pattern")
            add_entry(om2, px)
            # Manifold walk only when pattern brought onnxruntime to ~0 (basin
            # reached) but still short of the buffer -- the expensive last resort
            # for the "basin found, onnxruntime max ~0" instances.
            if not light and om2 > -5e-6 and best_ort < buffer:
                log(f"    manifold: walking basin from ORT worst-margin={om2:+.3e} ...")
                mx, mm = manifold_search(px, iters=140, deadline=m_dl)
                update_best(mm, mx, "manifold")
                add_entry(mm, mx)
        elif near2.any():
            # Opt-in path (endgame_cands > 1): run the ladder from the top-N
            # polished candidates instead of just the best. Empirically a worse
            # use of budget than more PGD rounds, hence default endgame_cands = 1.
            pool_pts, pool_m = polish(cand[near2][:max(8, endgame_cands * 2)],
                                      return_all=True)
            oms = ort_worst(pool_pts)
            order = list(np.argsort(-oms)); j0 = order[0]
            log(f"    polish: {len(pool_pts)} entries | best surrogate={pool_m.max():+.3e} "
                f"ORT={oms[j0]:+.3e} | endgame on top {min(endgame_cands, len(order))}")
            update_best(oms[j0], pool_pts[j0], "polish")
            for rank, j in enumerate(order[:endgame_cands]):
                if best_ort >= buffer or time.time() - t_start > round_budget:
                    break
                px = pool_pts[j]; om2 = oms[j]
                cx, cm = ort_pattern(px, target=buffer, label=f"pattern[c{rank}]")
                if cm > om2: om2 = cm; px = cx
                update_best(om2, px, "pattern"); add_entry(om2, px)
                if om2 > -5e-6 and best_ort < buffer:
                    mx, mm = manifold_search(px, iters=140, deadline=m_dl)
                    update_best(mm, mx, "manifold"); add_entry(mm, mx)
        return tvn

    # Run PGD rounds. Each round: standard PGD -> endgame; if that did NOT solve
    # and PGD stayed stuck in a decoy basin, ILS-hop out and run the endgame
    # again. Endgame-first means instances the endgame can already crack do not
    # pay for the (expensive) ILS hops.
    while time.time() - t_start < round_budget and not success:
        # Re-seed per round with the round index for diverse, reproducible
        # restart pools. The seed_offset arg shifts the deterministic draw so a
        # stochastically-missed instance can be retried with a fresh sequence.
        g_rng.manual_seed(seed_offset * 100003 + rounds)
        # Seed numpy too. The endgame's manifold_search (and ort_random) draw their
        # search directions from np.random, so seeding ONLY the torch RNG left the
        # manifold-dependent instances (idx 31/47) non-deterministic -- fast some
        # runs, slow/fail on others. Seeding both per round makes the whole attack,
        # and each instance's per-round outcome, fully reproducible.
        np.random.seed((seed_offset * 100003 + rounds) % (2 ** 31))
        rounds += 1
        t_round = time.time()
        # Start the restart pool uniformly at random in the box: [restarts, n_in].
        xb = L + (U - L) * torch.rand(restarts, n_in, generator=g_rng, device=device)
        if initialization == 'osi':
            # Output-Sampling Initialization: give each restart a RANDOM linear
            # objective over the outputs (weights wd: [restarts, n_out] in [-1,1])
            # and sign-ascend it for a few steps. This scatters the restarts to
            # diverse, output-driven corners of the box -- a much better spread of
            # starting points than uniform noise, so more restarts land near the
            # (single, hidden) basin.
            wd = (torch.rand(restarts, n_out, generator=g_rng, device=device) * 2 - 1)
            xo = xb.clone().requires_grad_(True)
            for _ in range(osi_iters):
                if deadline_reached():
                    break
                proj = (fwd(xo) * wd).sum()                 # sum_r <wd_r, y_r>
                gx, = torch.autograd.grad(proj, xo)
                xo = (xo + step0 * gx.sign()).clamp(L, U).detach().requires_grad_(True)
            xb = xo.detach()
        xb = xb.detach().requires_grad_(True)
        m = torch.zeros_like(xb); v = torch.zeros_like(xb)
        b1, b2, e = 0.9, 0.999, 1e-8
        step = step0.clone(); done_steps = 0
        # Adam-with-sign + slow decay OSCILLATES around the basin, so the worst
        # margin goes up and back down. Keep the best (deepest) point each
        # restart ever reaches, not the final iterate.
        best_wm = torch.full((restarts,), -1e30, device=device)   # [restarts] best worst-margin/restart
        best_xb = xb.detach().clone()                             # [restarts, n_in] its argmax point
        for t in range(1, steps + 1):
            if deadline_reached():
                break
            g = abc_margin(xb)                              # [restarts, n_spec]
            with torch.no_grad():
                wm_t = g.min(dim=1).values                 # worst-margin per restart now
                imp = wm_t > best_wm
                best_xb = torch.where(imp.unsqueeze(1), xb.detach(), best_xb)
                best_wm = torch.maximum(best_wm, wm_t)
            loss = -torch.clamp(g, max=kappa).sum()
            gx, = torch.autograd.grad(loss, xb)
            with torch.no_grad():
                m.mul_(b1).add_(gx, alpha=1 - b1)
                v.mul_(b2).addcmul_(gx, gx, value=1 - b2)
                bc1 = 1 - b1 ** t; bc2 = 1 - b2 ** t
                denom = v.sqrt() / (bc2 ** 0.5) + e
                xb.sub_((step / bc1) * (m / denom).sign())
                xb.clamp_(L, U)
                step *= lr_decay
            xb = xb.detach().requires_grad_(True)
            done_steps = t
            if verbose and t % 200 == 0:
                log(f"  round {rounds} pgd step {t}/{steps}: surrogate worst-margin "
                    f"best-ever={best_wm.max().item():+.3e} current={wm_t.max().item():+.3e}")
            if time.time() - t_start > round_budget:
                break
        # also fold in the final iterate
        with torch.no_grad():
            wm_t = abc_margin(xb).min(dim=1).values
            imp = wm_t > best_wm
            best_xb = torch.where(imp.unsqueeze(1), xb.detach(), best_xb)
            best_wm = torch.maximum(best_wm, wm_t)
        t_pgd = time.time() - t_round
        n1 = int((best_wm > -1e-4).sum().item()); n2 = int((best_wm > -1e-3).sum().item())
        log(f"  round {rounds}: PGD {done_steps} steps in {t_pgd:.1f}s | best surrogate "
            f"worst-margin={best_wm.max().item():+.3e} | near-basin restarts: "
            f"{n1}(>-1e-4) {n2}(>-1e-3)")
        # ---- endgame first (cheap for instances it can already crack) ----
        do_endgame(best_xb, best_wm)
        if best_ort >= buffer:
            success = True
        # ---- ILS: only if the endgame did NOT solve it and PGD is stuck in a
        #      decoy basin. Hop out (cycling escape distances) and re-run the
        #      endgame from the escaped points. ----
        # ILS only when PGD is stuck in a DEEP decoy (best surrogate < -1e-5). The
        # batched deep-drill now converts SHALLOW near-misses (worst-margin close to
        # 0, e.g. idx 5) directly in the endgame, so they no longer need an ILS hop.
        # Keeping this gate is important: instances that reach the basin but sit at
        # ORT margin ~0 (idx 10/31/37) must be left to the manifold, NOT sent through
        # a budget-eating ILS that then starves the manifold and makes them time out.
        #
        # ALSO gate on the ORT margin: the surrogate can read "stuck in a decoy"
        # (best_wm < -1e-5) while onnxruntime has ALREADY reached the basin floor
        # (best_ort ~ 0), as for idx 47. Running the ILS there is pure waste -- it
        # burns ~50s escaping a decoy we have already escaped, and the floor it reaches
        # is a non-crossable local max anyway. The counterexample comes from a FRESH PGD
        # restart (a new round) landing a DIFFERENT, crossable basin. So once ORT is at
        # the floor (best_ort > -2e-6), SKIP the ILS and let the round loop start that
        # fresh round -- this is what lands idx 47 within budget.
        if not success and ils_hops > 0 and best_wm.max().item() < -1e-5 \
                and best_ort < -2e-6 \
                and time.time() - t_start < round_budget:
            # Sequential single-scale hops: each hop jumps the current top-k best
            # points OUT of their decoy and re-optimizes them, and the scale CYCLES
            # across hops (0.1/0.2/0.4/0.6*eps) so the different escape distances
            # each hard instance needs are covered without per-instance tuning.
            # (The per-hop refinement is essential -- batching all scales into one
            # wide pass to fill the GPU was tried and regresses; keep the iteration
            # and instead control GPU use via `ils_keep` (mini-PGD batch width).)
            hops_done = 0
            for hop in range(ils_hops):
                if time.time() - t_start > round_budget:
                    break
                k = min(ils_keep, restarts)
                _, ti = torch.topk(best_wm, k)             # the k current-best points
                cur = best_xb[ti]                          # [k, n_in]
                jscale = ils_jumps[hop % len(ils_jumps)]   # cycle escape distance
                jmp = (torch.rand(k, n_in, generator=g_rng, device=device) * 2 - 1) \
                    * jscale * eps_v.unsqueeze(0)          # [k, n_in] in [-jscale,jscale]*eps
                # Re-optimize the jumped points, then MERGE them with the existing
                # pool and keep the best `restarts` overall (elitist: a hop that
                # lands somewhere worse is simply discarded, a better basin sticks).
                hb_wm, hb_xb = mini_pgd((cur + jmp).clamp(L, U), ils_hop_steps)   # [k], [k, n_in]
                allx = torch.cat([best_xb, hb_xb], 0)      # [restarts + k, n_in]
                allw = torch.cat([best_wm, hb_wm], 0)      # [restarts + k]
                tv2, ti2 = torch.topk(allw, restarts)
                best_xb = allx[ti2]; best_wm = allw[ti2]   # back to [restarts, *]
                hops_done = hop + 1
                # Probe every few hops with a LIGHT endgame (batched drill, NO slow
                # manifold) and stop hopping as soon as the basin is in hand -- the full
                # 16 hops cost ~100s on the A10G, wasted once an early hop already found
                # the basin. Decide on the ONNXRUNTIME margin (best_ort), which is the
                # real test, NOT the surrogate (a shallow non-basin point reads near 0
                # on the surrogate without being a cex, e.g. idx 6 at hop 1):
                #   best_ort >= buffer -> solved outright (drill-convertible 6/8/9/10/44);
                #   best_ort > -1e-5   -> basin reached but needs the manifold, so stop
                #                         hopping and let the post-loop full endgame
                #                         finish it with the remaining budget (31/37/47).
                if best_wm.max().item() > -1e-4 and (hop + 1) % ils_check_every == 0 \
                        and time.time() - t_start < round_budget:
                    do_endgame(best_xb, best_wm, light=True)
                    if best_ort >= buffer:
                        success = True
                        break
                    # Late in the hop budget only: if the probe has reached the basin
                    # (ONNX margin ~0) but the drill couldn't convert it, stop hopping
                    # and hand the remaining budget to the post-loop manifold (the
                    # manifold-bound tail 31/37/47). Gated to the 2nd half of the hops
                    # so it can't fire on a drill-convertible instance still improving
                    # through a transient early near-0 (e.g. idx 9 at hop 1).
                    if best_ort > -1e-6 and (hop + 1) >= ils_hops // 2:
                        break
            log(f"  round {rounds}: ILS {hops_done} hops (jumps={ils_jumps}) | "
                f"best surrogate worst-margin {best_wm.max().item():+.3e}")
            if not success:
                do_endgame(best_xb, best_wm)
                if best_ort >= buffer:
                    success = True
        log(f"  round {rounds} done: best ORT worst-margin={best_ort:+.3e} via {best_stage} | "
            f"elapsed={time.time()-t_start:.1f}s success={success}")

    # ---- final backstop: more manifold walks from the best entries ----
    phase_deadline = t_start + time_budget
    if not success and entry_pool and entry_pool[0][0] > -5e-6:
        log(f"backstop phase: {len(entry_pool)} basin entries, best ORT margin="
            f"{entry_pool[0][0]:+.3e}, budget left={time_budget-(time.time()-t_start):.1f}s")
        ei = 0
        while time.time() - t_start < time_budget and best_ort < buffer:
            start_x = entry_pool[ei % len(entry_pool)][1]; ei += 1
            mx, mm = manifold_search(start_x, iters=140, deadline=t_start + time_budget)
            update_best(mm, mx, "manifold-backstop")
            cx, cm = ort_pattern(best_ort_x if best_ort_x is not None else start_x,
                                 target=buffer, label="backstop-pattern")
            update_best(cm, cx, "pattern-backstop")
        success = bool(best_ort >= buffer)

    # ----------------------- build abcrown return -------------------------
    if best_ort_x is None:
        best_ort_x = ((Lnp + Unp) / 2).astype(np.float32)
        log("no near-basin candidate found at all (PGD never reached the basin)")
    elif best_ort > -5e-5 and time.time() - t_start < time_budget - 15:
        # Push the onnxruntime margin deep for a ROBUST safety buffer -- but ONLY if
        # there is budget left for it. The deepen costs ~10s, and running it past the
        # budget pushed A10G solves over the 150s timeout (idx 8 found its cex at 141s
        # but the unconditional deepen ran to 153s and lost it). If we already have a
        # valid cex we just return it as-is; if we don't, this was the last chance.
        # First deep-drill the surrogate (cheap, gradient) to convert/deepen, adopt it
        # if its onnxruntime margin is better, then ort_pattern toward robust_target.
        if drill_steps > 0:
            dx, dm = deep_drill(best_ort_x[None, :])
            if float(dm[0]) > -5e-5:
                do2 = ort_worst1(dx[0])
                if do2 > best_ort:
                    best_ort = do2; best_ort_x = dx[0]
        bx, bm = ort_pattern(best_ort_x, sweeps=60,
                             target=max(robust_target, buffer), label="final-deepen")
        if bm > best_ort:
            best_ort = bm; best_ort_x = bx
    success = bool(best_ort >= buffer)

    adv = torch.tensor(best_ort_x, device=device, dtype=x.dtype).view(1, *in_shape)
    # adv_output MUST be the ONNXRUNTIME output: abcrown's check_and_save_cex
    # gate (is_specification_vio) validates the output we return here, and the
    # official VNN-COMP checker also uses onnxruntime. Returning the pytorch
    # (onnx2pytorch) output would let a razor-thin onnxruntime-valid point be
    # rejected by the gate.
    adv_out_np = _ort_out(best_ort_x.reshape(1, -1))            # [1, n_out] float32
    adv_output = torch.tensor(adv_out_np, device=device, dtype=x.dtype).view(1, n_out)
    adv_margin_per_spec = (Cm * adv_output).sum(-1) - rh        # [num_spec], abcrown convention
    attack_ret = torch.tensor([success], device=device)
    best_or_idx = torch.tensor([0], device=device, dtype=torch.long)
    adv_input = adv                                             # [1(num_or), *in_shape]
    # Per-spec margins from the ONNXRUNTIME output the checker will see; a valid
    # counterexample needs every C.y - rhs <= 0 (equivalently worst >= 0 here).
    ort_spec = (rhnp - adv_out_np.astype(np.float64) @ Cnp.T).reshape(-1)
    worst_idx = int(ort_spec.argmin())
    log(f"FINAL success={success} via {best_stage} | ORT worst-margin={best_ort:+.3e} "
        f"(binding spec #{worst_idx}) | rounds={rounds} time={time.time()-t_start:.1f}s")
    if verbose:
        log(f"    per-spec ORT margins (rhs - C.y, want all >=0): "
            f"[{', '.join(f'{m:+.2e}' for m in ort_spec)}]")
    print(f"[soundness_attack] FINAL success={success} ort_worst={best_ort:+.3e} "
          f"rounds={rounds} time={time.time()-t_start:.1f}s", flush=True)
    return attack_ret, best_or_idx, adv_input, adv_output, adv_margin_per_spec, None
