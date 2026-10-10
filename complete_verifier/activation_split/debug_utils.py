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
import torch
from auto_LiRPA.bound_ops import BoundInput

from state import WorkingIntermBoundsInfo

# enabled only if --check_mixed_decision_bound_soundness is given to to the argument
def _check_mixed_precision_bound_soundness(
    net, new_x, c, enable_opt_interm_bounds, lb, decision_thresh
):
    """
    Recompute lowerbounds in FP32 precision and compare to the mixed-precision
    lowerbounds. If they differ, raise an error.
    Meantime, keep the module's state exactly as it was before this function was called.
    """

    bm = net.net
    full_batch = lb.size(0)
    was_pruned = new_x.shape[0] != full_batch
    if was_pruned:
        x_L, x_U = new_x.ptb.x_L, new_x.ptb.x_U
        assert torch.equal(x_L, x_L[:1].expand_as(x_L)) and torch.equal(
            x_U, x_U[:1].expand_as(x_U)
        ), (
            "Mixed-precision soundness check: new_x was pruned to "
            f"{new_x.shape[0]}/{full_batch} domains and its per-domain input "
            "regions differ, so the full batch cannot be rebuilt by expansion. "
            "Pass the domain x_Ls/x_Us to reconstruct it (input-split BaB)."
        )

        def _to_full(t):
            return t[:1].expand(full_batch, *t.shape[1:]).contiguous()

        ptb = type(new_x.ptb)(norm=new_x.ptb.norm, x_L=_to_full(x_L), x_U=_to_full(x_U))
        new_x = type(new_x)(_to_full(new_x.data), ptb)

    # The recompute forwards new_x and runs a single (non-optimizing) backward
    # pass, which reassigns several module attributes. Snapshot them all so the
    # module is restored exactly afterwards. iteration=1 evaluates the current
    # alpha/beta with no optimizer step (the backward/opt.step in
    # optimized_bounds.py is guarded by `i != iteration - 1`), so the
    # optimizable parameters keep their values and need no snapshot.
    saved_opt_args = dict(bm.bound_opts["optimize_bound_args"])
    saved_enable_interm = bm.bound_opts.get("enable_opt_interm_bounds")
    saved_preserve_mask = bm.last_update_preserve_mask
    saved_roots = {}
    for root in bm.roots():
        cached_value = root.forward_value
        input_value = (bm.get_forward_value(root)
                       if type(root) is BoundInput else None)
        saved_roots[root] = (
            input_value, getattr(root, "perturbation", None), cached_value)
    saved_bounds = {l: (l.lower, l.upper) for l in bm.layers_requiring_bounds}
    saved_lA = {
        n: (getattr(n, "lA", None), getattr(n, "uA", None))
        for n in bm.get_splittable_activations()
    }
    saved_update_mask = {
        n: getattr(n, "alpha_beta_update_mask", None) for n in bm.nodes()
    }

    lb_ref = None
    try:
        bm.set_bound_opts(
            {
                "optimize_bound_args": {
                    "enable_beta_crown": True,
                    "fix_interm_bounds": True,
                    "pruning_in_iteration": False,
                    # Reuse the module's optimized (keep_best-restored) alpha
                    # instead of re-initializing it from scratch.
                    "init_alpha": False,
                    # Recompute a clean full-precision reference: disable mixed
                    # precision. (lb is itself a full-precision bound -- keep_best
                    # only updates on the full-precision rounds.)
                    "mixed_precision_dtype": None,
                    "low_precision_prune_in_iter": False,
                    "keep_best": False,
                    "iteration": 1,
                },
                "enable_opt_interm_bounds": enable_opt_interm_bounds,
            }
        )
        all_bounds = WorkingIntermBoundsInfo.from_net(net).to_dict_of_list()
        interm_bounds = all_bounds
        reference_bounds = all_bounds
        ref_ret_full_precision = bm.compute_bounds(
            x=(new_x,),
            C=c,
            method="CROWN-optimized",
            interm_bounds=interm_bounds,
            reference_bounds=reference_bounds,
            return_A=False,
            needed_A_dict=None,
            cutter=net.cutter,
            bound_upper=False,
            decision_thresh=decision_thresh,
        )
        lb_ref = ref_ret_full_precision[0]
    finally:
        # Restore the module exactly as we found it.
        for layer, (lo, up) in saved_bounds.items():
            layer.lower, layer.upper = lo, up
        for node, (lA, uA) in saved_lA.items():
            node.lA, node.uA = lA, uA
        for node, mask in saved_update_mask.items():
            node.alpha_beta_update_mask = mask
        for root, (value, perturbation, forward_value) in saved_roots.items():
            if type(root) is BoundInput:
                root.bind_input(value)
            root.perturbation = perturbation
            root.forward_value = forward_value
        bm.last_update_preserve_mask = saved_preserve_mask
        bm.set_bound_opts({"optimize_bound_args": saved_opt_args})
        if saved_enable_interm is not None:
            bm.set_bound_opts({"enable_opt_interm_bounds": saved_enable_interm})

    is_close = torch.allclose(lb_ref, lb, atol=1e-4, rtol=1e-4)
    if not is_close:
        abs_diff = (lb_ref - lb).abs()
        print("Warning: the computed bounds may not be stable under mixed precision.")
        print(f"Max abs difference in lb_ref - lb: {abs_diff.max().item()}")
        print(
            "Max rel difference in lb_ref - lb: "
            f"{(abs_diff / lb_ref.abs()).max().item()}"
        )
        print(f"Max suspicious gap in lb: {(lb - lb_ref).max().item()}")
        verified_mask_ref = torch.all(
            torch.max(lb_ref >= decision_thresh, dim=1, keepdim=True)[0], dim=1
        )
        verified_mask = torch.all(
            torch.max(lb >= decision_thresh, dim=1, keepdim=True)[0], dim=1
        )
        print(
            "Number of verified samples in ref verified: "
            f"{verified_mask_ref.sum().item()}. mixed precision verified: "
            f"{verified_mask.sum().item()}"
        )
        diff_mask = torch.nonzero(
            verified_mask_ref ^ verified_mask, as_tuple=False
        ).squeeze(-1)
        if diff_mask.numel() > 0:
            diff_abs = (lb_ref[diff_mask] - lb[diff_mask]).abs()
            print(
                "Max abs difference in lb_ref - lb for samples with "
                f"different verification result: {diff_abs.max().item()}"
            )
            print(
                "Max rel difference in lb_ref - lb for samples with "
                "different verification result: "
                f"{(diff_abs / lb_ref[diff_mask].abs()).max().item()}"
            )
            print(f"Raw bounds: (mixed precision) {lb[diff_mask]} ")
            print(f"Raw bounds: (full precision) {lb_ref[diff_mask]} ")
            # Record the disagreement and keep going so the whole run can be
            # logged in one pass, rather than aborting on the first one. Grep
            # the log for this tag to collect every flipped-decision domain.
            print(
                "[MIXED-PRECISION-UNSOUND] mixed precision and full precision "
                f"disagree on {diff_mask.numel()} domain(s); "
                f"max abs gap {diff_abs.max().item()}."
            )
            raise RuntimeError(
                "Mixed-precision bound computation is not sound: the computed "
                "bounds differ from full-precision bounds. See the log for "
                "details."
            )
    else:
        print("The computed bounds are close to full precision backward bounds.")
