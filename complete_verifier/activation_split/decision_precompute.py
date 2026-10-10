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
"""
precompute both branching and clip decisions.
"""

import torch
from math import log
import arguments
from typing import Tuple, TYPE_CHECKING
from heuristics.decision_types import (
    BranchingDecisions,
    BatchFirstBranchingDecisions,
)
from heuristics import BranchingHeuristicObj
from domain_clipper import (
    DomainClipScorer,
    SubDomainClipDecisions,
)
from activation_split.utils import (
    mask_tensor_first_dim,
    mask_tensor_first_dim_allow_none,
    mask_list,
    tf32_ctx,
)
from activation_split.return_types import (
    UpdateBoundCoreContext,
    UpdateBoundCoreReturn,
)
from state import (
    AlphaValueData,
    BetaFullData,
    BatchedlA,
    WorkingIntermBoundsInfo,
    IntermBoundsFactory,
)
from branching_domains import BatchedDomainList

if TYPE_CHECKING:
    from ..beta_CROWN_solver import LiRPANet


def get_split_depth(batch_size, min_batch_size, min_depth):
    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    if batch_size < min_batch_size:
        # Split multiple levels, to obtain at least min_batch_size domains in this batch.
        return max(
            min_depth,
            int(log(min_batch_size / max(min_depth, batch_size)) / log(2)),
        )
    else:
        return min_depth


def compute_first_iteration_decision(
    net: "LiRPANet",
    branching_heuristic: BranchingHeuristicObj,
    domain_clip_scorer: DomainClipScorer | None,
    domains: BatchedDomainList,
    domain_interm_factory: IntermBoundsFactory,
    *,
    timer,
    device,
    enable_clip_domains: bool,
    drop_lA: bool,
):
    """
    Precompute the decision for the initial batch in the domain list.
    """
    solver_args = arguments.Config["solver"]
    bab_args = arguments.Config["bab"]
    branch_args = bab_args["branching"]
    batch = solver_args["batch_size"]

    depth = 1
    batch_selected_domain = len(domains)
    if batch_selected_domain == 0:
        return
    first_d = domains.pick_out(batch_selected_domain, device)
    domain_interm_factory.construct_interm_bounds_in_d(first_d, net.unstable_mask)
    min_batch_size = min(
        solver_args["min_batch_size_ratio"] * solver_args["batch_size"],
        batch,
    )
    split_depth = get_split_depth(batch_selected_domain, min_batch_size, depth)
    first_d["mask"] = WorkingIntermBoundsInfo.from_two_dicts(
        first_d["lower_bounds"], first_d["upper_bounds"]
    ).compute_unstable_mask(net)
    branching_decisions = branching_heuristic.compute_branching_decisions(
            first_d,
            split_depth,
            method=branch_args["method"],
            branching_candidates=max(branch_args["candidates"], split_depth),
            branching_reduceop=branch_args["reduceop"],
            timer=timer,
        )
    domain_decision = BatchFirstBranchingDecisions.from_branching_decision(
        branching_decision=branching_decisions
    )
    if enable_clip_domains:
        assert domain_clip_scorer is not None
        # make a shallow copy
        copied_first_d = first_d.copy()
        net.build_history_and_set_bounds(
            d=copied_first_d,
            split={
                "decision": branching_decisions.branching_decision,
                "points": branching_decisions.branching_points,
            },
            mode="depth",
        )
        clip_decisions = domain_clip_scorer.compute_clip_decisions(
            domains=copied_first_d,
            split_activations=net.split_activations,
        )
        first_d["sub_domain_clip_decisions"] = SubDomainClipDecisions.from_clip_decisions(
            splitted_domain_clip_decisions=clip_decisions,
            split_depth=split_depth,
            batch_size=batch_selected_domain,
        )

    # Reconstruct unstable_bounds by extracting unstable neurons from full bounds.
    # This mirrors the logic in interm_bounds_construct_domain_from_working.
    first_d_unstable_bounds = {}
    if bab_args["interm_transfer"]:
        for layer_name, mask in domains.unstable_mask.items():
            if mask is not None and layer_name in first_d["lower_bounds"]:
                first_d_unstable_bounds[layer_name] = [
                    first_d["lower_bounds"][layer_name][:, mask[0]],
                    first_d["upper_bounds"][layer_name][:, mask[0]],
                ]

    ret_bounds = {
        "lower_bounds": {net.final_name: first_d["lower_bounds"][net.final_name]},
        "upper_bounds": {net.final_name: first_d["upper_bounds"][net.final_name]},
        "lAs": first_d["lAs"],
        "alphas": first_d["alphas"],
        "betas": first_d["betas"],
        "split_history": first_d["split_history"],
        "unstable_bounds": first_d_unstable_bounds,
        "primals": None,
        "c": first_d["cs"],
        "x_Ls": first_d["x_Ls"],
        "x_Us": first_d["x_Us"],
        "input_split_idx": first_d["input_split_idx"],
        "decision_info": domain_decision,
        "sub_domain_clip_decisions": first_d["sub_domain_clip_decisions"],
    }
    d_for_add = {
        "history": first_d["history"],
        "thresholds": first_d["thresholds"],
        "depths": first_d["depths"],
    }
    if drop_lA:
        ret_bounds["lAs"] = {}
        domains.drop_lAs()
    domains.add(ret_bounds, d_for_add, check_infeasibility=False)


def precompute_decision(
    net: "LiRPANet",
    # data needed for precompute
    lb: torch.Tensor,
    ub: torch.Tensor,
    working_interm_bounds: WorkingIntermBoundsInfo,
    *,
    masked_lAs: BatchedlA,
    working_alpha: AlphaValueData,
    domain_beta_this_iter,
    depths,
    thresholds,
    history,
    c,
    # batch sizes
    batch_size: int,
    batch_device_limit: int,
    # branching decisiions and clip decisions
    domain_clip_scorer,
    branching_heuristic,
    # flags that precompute need to know
    precompute_bfs_flag: bool,
    enable_clip_domains: bool,
) -> Tuple[BranchingDecisions, SubDomainClipDecisions]:
    """Input a batch of unverified domains and compute their split decisions and clip decisions.

    Handle all_node_split cases.

    Args:
        net: The LiRPANet network.
        lb: A batch of lower bounds of the final layer after c, computed by compute_bounds.
        ub: Currently unused. A batch of upper bounds of the final layer after c,
            computed by compute_bounds.
        working_interm_bounds: A batch of intermediate bounds in the network.
        masked_lAs: A batch of intermediate lAs in the network.
        working_alpha: A batch of alpha values.
        domain_beta_this_iter: A batch of beta in the domain's format, values given
            recent compute_bounds in solve.
        depths: A batch of depths, representing the number of activations split.
        thresholds: A batch of thresholds.
        history: A batch of domain histories, each representing an activation split history.
        c: A batch of c matrices.
        batch_size: The batch size of this batch.
        batch_device_limit: The device-limited batch size, used to determine split depth.
        domain_clip_scorer: Scorer for clip decisions.
        branching_heuristic: Heuristic used for branching decisions.
        precompute_bfs_flag: Marks whether this is in the BFS phase; should be False.
        enable_clip_domains: Marks whether domain clip decisions are needed.

    Returns:
        A tuple of (BranchingDecisions, SubDomainClipDecisions).
    """

    net.timer.start("decision_precompute")
    # biccos_args = arguments.Config["bab"]["cut"]["biccos"]
    solver_args = arguments.Config["solver"]
    bab_args = arguments.Config["bab"]
    branch_args = bab_args["branching"]
    enable_tf32_branching_decision = bab_args["mixed_precision"]["enable_tf32_branching_decision"]
    enable_tf32_clip_decision = bab_args["mixed_precision"]["enable_tf32_clip_decision"]
    device = lb.device

    # ==================================================
    #  1. select domain that need decision precompute
    # ==================================================

    # Select the domains that need decision precomputeing
    # i. unverified (ensured before passing) AND ii. not all-split

    mask_not_all_split = torch.Tensor(depths).to(device=device) < net.tot_ambi_nodes
    actual_needed_batch = int(mask_not_all_split.sum().item())
    mask_not_all_split_cpu = mask_not_all_split.to("cpu")

    # early exit
    if actual_needed_batch == 0:
        # contruct empty precomputeed decision
        net.timer.add("decision_precompute")
        return (
            BranchingDecisions.reconstruct_from_masked_result(
                branching_decision=BranchingDecisions(
                    branching_decision=[],
                    branching_points=None,
                    split_depth=0,
                    batch_size=0,
                ),
                original_batch_size=batch_size,
                used_batch_mask=mask_not_all_split_cpu,
            ),
            (
                SubDomainClipDecisions.reconstruct_from_masked_result(
                    masked_result=SubDomainClipDecisions.from_clip_decisions(
                        splitted_domain_clip_decisions=(
                            domain_clip_scorer.get_empty_clip_decisions(batch_size=0)
                        ),
                        batch_size=0,
                        split_depth=0,
                    ),
                    batch_mask_used=mask_not_all_split,
                    original_batch_size=batch_size,
                )
                if domain_clip_scorer
                else SubDomainClipDecisions({}, 0, 0, 0)
            ),
        )

    # ==================================================
    # 2. prepare `precompute_d` needed to get decisions
    # ==================================================

    if precompute_bfs_flag:
        raise NotImplementedError("MTS with precompute is not supported yet.")
    else:
        depth = 1

    min_batch_size = min(
        solver_args["min_batch_size_ratio"] * solver_args["batch_size"],
        batch_device_limit,
    )
    split_depth = get_split_depth(actual_needed_batch, min_batch_size, depth)

    # garbage collect lA in the net to save memory.
    BatchedlA.gc_lA_from_net(net)

    ############### constructing precompute_d for decision precopmute.

    skip_all_true_mask = bool(batch_size == actual_needed_batch)

    ########## fused computation for bounds and masks for performance.
    masked_lower_bounds: dict = working_interm_bounds.to_lower_bounds_dict(
        append_dict={net.final_name: lb.detach()}
    )
    masked_upper_bounds: dict = working_interm_bounds.to_upper_bounds_dict(
        append_dict={net.final_name: ub.detach()}
    )
    masked_lower_bounds = (
        masked_lower_bounds
        if skip_all_true_mask
        else mask_tensor_first_dim(
            masked_lower_bounds,
            mask_not_all_split,
        )
    )
    masked_upper_bounds = (
        masked_upper_bounds
        if skip_all_true_mask
        else mask_tensor_first_dim(
            masked_upper_bounds,
            mask_not_all_split,
        )
    )

    #### Compute unstable mask for this iteration
    masked_unstable_activation_mask = WorkingIntermBoundsInfo.from_two_dicts(
        masked_lower_bounds, masked_upper_bounds
    ).compute_unstable_mask(net)

    precompute_d = {
        "lower_bounds": masked_lower_bounds,  # newest
        "upper_bounds": masked_upper_bounds,  # newest
        "mask": masked_unstable_activation_mask,  # newest
        "lAs": (
            masked_lAs
            if skip_all_true_mask
            else masked_lAs.inplace_mask_batch_dim(
                mask_not_all_split,
            )
        ),  # newest
        "cs": (
            c
            if skip_all_true_mask
            else mask_tensor_first_dim(
                c,
                mask_not_all_split,
            )
        ),  # newest (constant)
        "history": (
            history
            if skip_all_true_mask
            else mask_list(
                history,  # pre_result.history,
                mask_not_all_split_cpu,
            )
        ),  # newest (from build history)
        "alphas": (
            working_alpha
            if skip_all_true_mask
            else working_alpha.mask_batch_dim(
                mask_not_all_split,
            )
        ),  # newest shape fixed
        "betas": (
            domain_beta_this_iter
            if skip_all_true_mask
            else mask_list(
                domain_beta_this_iter,
                mask_not_all_split_cpu,
            )
        ),  # newest shape fixed
        "thresholds": (
            thresholds
            if skip_all_true_mask
            else mask_tensor_first_dim(
                thresholds,
                mask_not_all_split,
            )
        ),  # constants
    }

    # ==================================================
    #  3. get both split decision and clip decision
    # ==================================================

    net.timer.start("decision_precompute.compute_branching_decisions")
    # Increase the maximum number of candidates for fsb and kfsb
    # if there are more splits needed.
    with tf32_ctx(enable_tf32_branching_decision):
        _precomputed_wd = branching_heuristic.compute_branching_decisions(
            precompute_d,
            split_depth,
            method=branch_args["method"],
            branching_candidates=max(branch_args["candidates"], split_depth),
            branching_reduceop=branch_args["reduceop"],
            timer=net.timer,
        )
    split_depth = _precomputed_wd.split_depth
    net.timer.add("decision_precompute.compute_branching_decisions")
    # reorganization
    branching_decision = BranchingDecisions.reconstruct_from_masked_result(
        branching_decision=_precomputed_wd,
        original_batch_size=batch_size,
        used_batch_mask=mask_not_all_split_cpu,
    )

    # get decisions for domain clippers
    if enable_clip_domains:
        net.timer.start("decision_precompute.clip_decision")
        assert domain_clip_scorer is not None
        domain_clip_precompute_d = precompute_d.copy()  # shallow copy.

        net.build_history_and_set_bounds(
            d=domain_clip_precompute_d,
            split={
                "decision": _precomputed_wd.branching_decision,
                "points": _precomputed_wd.branching_points,
            },
            mode="depth",
        )
        reconstruct_from_masked_result = (
            SubDomainClipDecisions.reconstruct_from_masked_result
        )
        with tf32_ctx(enable_tf32_clip_decision):
            clip_decisions = domain_clip_scorer.compute_clip_decisions(
                domains=domain_clip_precompute_d,
                split_activations=net.net.split_activations,
            )
        precompute_clip_decisions = reconstruct_from_masked_result(
            masked_result=SubDomainClipDecisions.from_clip_decisions(
                splitted_domain_clip_decisions=clip_decisions,
                split_depth=split_depth,
                batch_size=actual_needed_batch,
            ),
            batch_mask_used=mask_not_all_split,
            original_batch_size=batch_size,
        )

        net.timer.add("decision_precompute.clip_decision")
    else:
        precompute_clip_decisions = SubDomainClipDecisions({}, 0, 0, 0)

    ret = branching_decision, precompute_clip_decisions
    net.timer.add("decision_precompute")
    return ret


def update_bounds_precompute_extract(
    self: "LiRPANet",
    d,
    alpha: bool,
    beta: bool,
    *,
    enable_clip_domains: bool,
    core_ctx: UpdateBoundCoreContext,
) -> UpdateBoundCoreReturn:
    """
    Compute the split and clip decisions for unverified domains.
    Return only contains unverified domains' decisions.
    """

    ################ remove unused field in d to save memory.
    for k in list(d.keys()):
        if k not in ["history", "depths", "betas"]:
            d.pop(k)

    mask_unverified = torch.all(core_ctx.lb <= core_ctx.decision_thresh, dim=1)
    mask_unverified_cpu = mask_unverified.to("cpu")

    depths = torch.as_tensor(d["depths"], dtype=torch.int32)
    num_unverified = int(mask_unverified.sum().item())

    lb_final_max = core_ctx.lb.max().item()
    lb_final_min = core_ctx.lb.min().item()
    n_splits = core_ctx.lb.shape[0]
    n_verified = n_splits - num_unverified

    ################# Extract and Mask.

    ####### Fused Extract and Mask
    # Extract and mask working intermediate bounds.
    masked_working_interm_bounds = (
        WorkingIntermBoundsInfo.from_net(self, move=True).mask_batch_dim_inplace(
            mask_unverified
        )
    )

    # Extract and mask lA.
    if self.net.last_update_preserve_mask is None:
        # there was no pruning. need to mask the output lAs
        masked_lAs = BatchedlA.from_net(
            self,
            preserve_mask=None,
            tot_cells=None,
            device=core_ctx.lb.device,
            move=True,
        ).inplace_mask_batch_dim(mask_unverified)

        BatchedlA.gc_lA_from_net(self)
    else:
        # there is pruning. lAs are already masked by last_update_preserve_mask
        mask_unverified_in_prune = torch.all(
            core_ctx.lb[self.net.last_update_preserve_mask]
            <= core_ctx.decision_thresh[self.net.last_update_preserve_mask],
            dim=1,
        )
        masked_lAs = BatchedlA.from_net(
            self,
            preserve_mask=None,
            tot_cells=None,
            device=core_ctx.lb.device,
            move=True,
        ).inplace_mask_batch_dim(mask_unverified_in_prune)

        BatchedlA.gc_lA_from_net(self)

    masked_working_alpha = (
        AlphaValueData.from_net(
            self, starting_node_scope="part", move=True
        ).mask_batch_dim_inplace(mask_unverified)
        if alpha
        else ValueError("Alpha has not been used in this run.")
    )

    if beta:
        assert not isinstance(core_ctx.nums_effective_beta_per_domain, ValueError)

        masked_working_beta = BetaFullData.from_net(
            self, core_ctx.nums_effective_beta_per_domain[0].keys()
        ).mask_batch_dim_inplace(mask_unverified)

        masked_nums_effctive_beta_per_domain = (
            core_ctx.nums_effective_beta_per_domain.mask_batch_dim(mask_unverified_cpu)
        )

        masked_domain_beta_this_iter = (
            list(
                masked_working_beta.to_domain_dict(
                    masked_nums_effctive_beta_per_domain, device="cpu"
                )
            )
            if type(core_ctx.branching_heuristic).__name__ == "NonlinearBranching"
            else [ValueError("Only nonlinear branching needs beta from this iter")]
            * num_unverified
        )

    else:
        masked_working_beta = ValueError("Beta is truned off")

        masked_nums_effctive_beta_per_domain = ValueError(
            "Splits per example has not been used in this run."
        )

        masked_domain_beta_this_iter = [
            ValueError("Beta is truned off")
        ] * num_unverified

    masked_split_history = (
        self.get_cut_new_split_history(n_splits, mask_select=mask_unverified_cpu)
        if self.net.cut_used
        else [{} for _ in range(n_splits)]
    )

    ###### Simple mask
    masked_lb: torch.Tensor = mask_tensor_first_dim(core_ctx.lb, mask_unverified)
    masked_ub = mask_tensor_first_dim(core_ctx.ub, mask_unverified)
    masked_lb_last = mask_tensor_first_dim(core_ctx.lb_last, mask_unverified)
    masked_ub_last = mask_tensor_first_dim(core_ctx.ub_last, mask_unverified)
    masked_c = mask_tensor_first_dim(core_ctx.c, mask_unverified)
    masked_new_x_Ls = mask_tensor_first_dim_allow_none(
        core_ctx.new_x_Ls, mask_unverified
    )
    masked_new_x_Us = mask_tensor_first_dim_allow_none(
        core_ctx.new_x_Us, mask_unverified
    )
    masked_history = mask_list(d["history"], mask_unverified_cpu)
    masked_depths = mask_tensor_first_dim(depths, mask_unverified_cpu)
    masked_decision_thresh = mask_tensor_first_dim(
        core_ctx.decision_thresh, mask_unverified
    )

    ##### Safeguard for branching_input_and_activation being false.
    assert (
        core_ctx.primal_x is None
    ), "branching_input_and_activation is assumed always False"
    assert core_ctx.input_split_idx is None, "input_split_idx is assumed always None"
    assert core_ctx.x_Ls is None, "x_Ls is assumed always None"
    assert core_ctx.x_Us is None, "x_Us is assumed always None"

    self.timer.start("decision_precomputeing")

    ##### Precompute split and clip decisions
    branching_decision, precompute_clip_decisions = precompute_decision(
        net=self,
        # data needed for precompute
        lb=masked_lb,
        ub=masked_ub,
        working_interm_bounds=masked_working_interm_bounds,
        masked_lAs=masked_lAs,
        working_alpha=masked_working_alpha,
        domain_beta_this_iter=masked_domain_beta_this_iter,
        depths=masked_depths,
        thresholds=masked_decision_thresh,
        history=masked_history,
        c=masked_c,
        # batch sizes
        batch_size=num_unverified,
        batch_device_limit=core_ctx.batch_device_limit,
        # branching decisiions and clip decisions
        domain_clip_scorer=core_ctx.domain_clip_scorer,
        branching_heuristic=core_ctx.branching_heuristic,
        # flags that precompute need to know
        precompute_bfs_flag=core_ctx.precompute_bfs_flag,
        enable_clip_domains=enable_clip_domains,
    )
    self.timer.add("decision_precomputeing")

    emptied_lA = BatchedlA(
        {},
        is_emptied=True,
    )

    # modify required fields in d
    d["thresholds"] = masked_decision_thresh
    d["history"] = masked_history
    d["depths"] = masked_depths.tolist()

    # prune unnecessary fields in d to make sure no one uses them by mistake
    for k in list(d.keys()):
        if k not in ["history", "depths", "thresholds"]:
            d.pop(k)

    return UpdateBoundCoreReturn(
        lb=masked_lb,
        ub=masked_ub,
        lb_last=masked_lb_last,
        ub_last=masked_ub_last,
        nums_effective_beta_per_domain=masked_nums_effctive_beta_per_domain,
        input_split_idx=None,  # uesless
        primal_x=None,  # uesless
        x_Ls=None,
        x_Us=None,
        new_x_Ls=masked_new_x_Ls,
        new_x_Us=masked_new_x_Us,
        c=masked_c,
        working_beta=masked_working_beta,
        working_alpha=masked_working_alpha,
        working_interm_bounds=masked_working_interm_bounds,
        batched_lA=emptied_lA,
        branching_decision=branching_decision,
        sub_domain_clip_decisions=precompute_clip_decisions,
        decision_thresh=masked_decision_thresh,
        lb_final_max=lb_final_max,
        lb_final_min=lb_final_min,
        n_verified=n_verified,
        n_splits=n_splits,
        new_split_history=masked_split_history,
        depths=masked_depths,
        thresholds=masked_decision_thresh,
        history=masked_history,
    )


def update_bounds_extract_no_mask_no_precompute(
    self: "LiRPANet",
    alpha: bool,
    beta: bool,
    enable_opt_interm_bounds: bool,
    *,
    core_ctx: UpdateBoundCoreContext,
    depths,
    history,
) -> UpdateBoundCoreReturn:
    """
    Extract results from BoundedModule for the normal (non-precompute) path.
    Returns all domains without performing masking nor decision precomputation
    """
    # stats
    lb_final_max = core_ctx.lb.max().item()
    lb_final_min = core_ctx.lb.min().item()
    n_verified = int(
        torch.sum(core_ctx.lb > core_ctx.decision_thresh.to(core_ctx.lb.device)).item()
    )
    n_splits = core_ctx.lb.shape[0]

    ################## Extract values #########################
    batched_lA = BatchedlA.from_net(
        self,
        preserve_mask=self.net.last_update_preserve_mask,
        tot_cells=n_splits,
        move=True,
    )
    working_interm_bounds = WorkingIntermBoundsInfo.from_net(self, move=True)
    working_alpha = (
        AlphaValueData.from_net(
            self,
            starting_node_scope="part" if enable_opt_interm_bounds else "all",
            move=True,
        )
        if alpha
        else ValueError("Alpha has not been used in this run.")
    )
    working_beta = (
        BetaFullData.from_net(self, core_ctx.nums_effective_beta_per_domain[0].keys())
        if beta
        else ValueError("Beta has not been used in this run.")
    )
    new_split_history = (
        self.get_cut_new_split_history(core_ctx.original_size)
        if self.net.cut_used
        else [{}] * core_ctx.original_size
    )

    ################### Create empty decisions as spacers ################
    branching_decision = BranchingDecisions(
        branching_decision=[[]],
        branching_points=None,
        split_depth=0,
        batch_size=core_ctx.lb.shape[0],
    )
    precompute_clip_decisions = SubDomainClipDecisions({}, 0, 0, 0)

    return UpdateBoundCoreReturn(
        lb=core_ctx.lb,
        ub=core_ctx.ub,
        lb_last=core_ctx.lb_last,
        ub_last=core_ctx.ub_last,
        nums_effective_beta_per_domain=core_ctx.nums_effective_beta_per_domain,
        input_split_idx=core_ctx.input_split_idx,
        primal_x=core_ctx.primal_x,
        x_Ls=core_ctx.x_Ls,
        x_Us=core_ctx.x_Us,
        new_x_Ls=core_ctx.new_x_Ls,
        new_x_Us=core_ctx.new_x_Us,
        c=core_ctx.c,
        working_beta=working_beta,
        working_alpha=working_alpha,
        working_interm_bounds=working_interm_bounds,
        batched_lA=batched_lA,
        branching_decision=branching_decision,
        sub_domain_clip_decisions=precompute_clip_decisions,
        decision_thresh=core_ctx.decision_thresh,
        lb_final_max=lb_final_max,
        lb_final_min=lb_final_min,
        n_verified=n_verified,
        n_splits=n_splits,
        new_split_history=new_split_history,
        depths=depths,
        thresholds=core_ctx.decision_thresh,
        history=history,
    )


# TODO: this function can be potentially merged with update_bounds_precompute_extract.
def update_bounds_precompute_biccos_extract(
    self: "LiRPANet",
    d,
    alpha: bool,
    beta: bool,
    *,
    enable_clip_domains: bool,
    enable_cut: bool,
    enable_biccos: bool,
    iter_idx: int,
    visited_num: int,
    core_ctx: UpdateBoundCoreContext,
) -> UpdateBoundCoreReturn:
    """
    Compute the split and clip decisions for unverified domains.
    Return all domains' decisions.

    """

    lb_final_max = core_ctx.lb.max().item()
    lb_final_min = core_ctx.lb.min().item()
    n_splits = core_ctx.lb.shape[0]
    n_verified = n_splits - int(
        torch.all(core_ctx.lb <= core_ctx.decision_thresh, dim=1).sum().item()
    )

    mask_unverified = torch.all(core_ctx.lb <= core_ctx.decision_thresh, dim=1)
    num_unverified = int(mask_unverified.sum().item())
    mask_unverified_cpu = mask_unverified.to("cpu")

    ######## Fused extract-and-mask data from net
    #### need to be performed in advance because BICCOS
    ### will call compute_bounds again and destroy these data

    masked_working_interm_bounds = (
        WorkingIntermBoundsInfo.from_net(self, move=True).mask_batch_dim_inplace(
            mask_unverified
        )
    )

    if self.net.last_update_preserve_mask is None:
        masked_lAs = BatchedlA.from_net(
            self,
            preserve_mask=None,
            tot_cells=None,
            device=core_ctx.lb.device,
        ).inplace_mask_batch_dim(mask_unverified)

        BatchedlA.gc_lA_from_net(self)
    else:
        # there is pruning. lAs are already masked by last_update_preserve_mask
        mask_unverified_in_prune = torch.all(
            core_ctx.lb[self.net.last_update_preserve_mask]
            <= core_ctx.decision_thresh[self.net.last_update_preserve_mask],
            dim=1,
        )
        masked_lAs = BatchedlA.from_net(
            self,
            preserve_mask=None,
            tot_cells=None,
            device=core_ctx.lb.device,
        ).inplace_mask_batch_dim(mask_unverified_in_prune)

        BatchedlA.gc_lA_from_net(self)

    masked_working_alpha = (
        AlphaValueData.from_net(
            self, starting_node_scope="part", move=True
        ).mask_batch_dim_inplace(mask_unverified)
        if alpha
        else ValueError("Alpha has not been used in this run.")
    )

    if beta:
        assert not isinstance(core_ctx.nums_effective_beta_per_domain, ValueError)
        masked_working_beta = BetaFullData.from_net(
            self, core_ctx.nums_effective_beta_per_domain[0].keys()
        )
        new_beta_no_masked = masked_working_beta.to_domain_dict(
            core_ctx.nums_effective_beta_per_domain, device=core_ctx.lb.device
        )
        masked_working_beta.mask_batch_dim_inplace(mask_unverified)
        masked_nums_effctive_beta_per_domain = (
            core_ctx.nums_effective_beta_per_domain.mask_batch_dim(mask_unverified_cpu)
        )
        masked_domain_beta_this_iter = (
            list(
                masked_working_beta.to_domain_dict(
                    masked_nums_effctive_beta_per_domain, device="cpu"
                )
            )
            if type(core_ctx.branching_heuristic).__name__ == "NonlinearBranching"
            else [ValueError("Only nonlinear branching needs beta from this iter")]
            * num_unverified
        )

    else:
        masked_working_beta = ValueError("Beta is truned off")
        masked_nums_effctive_beta_per_domain = ValueError(
            "Splits per example has not been used in this run."
        )
        new_beta_no_masked = [ValueError("Beta is truned off")] * n_splits
        masked_domain_beta_this_iter = [
            ValueError("Beta is truned off")
        ] * num_unverified

    # BICCOS needs cuts info unmasked.
    new_split_history_no_masked = (
        self.get_cut_new_split_history(n_splits)
        if self.net.cut_used
        else [{} for _ in range(n_splits)]
    )
    masked_split_history = mask_list(new_split_history_no_masked, mask_unverified_cpu)

    ##### Safeguard for broken branching_input_and_activation
    assert (
        core_ctx.primal_x is None
    ), "branching_input_and_activation is assumed always False"
    assert core_ctx.input_split_idx is None, "input_split_idx is assumed always None"
    assert core_ctx.x_Ls is None, "x_Ls is assumed always None"
    assert core_ctx.x_Us is None, "x_Us is assumed always None"

    ########### Process biccos cuts before performing mask
    if enable_cut and enable_biccos:
        # We only enforce cut usage for multi-tree-searching
        enforce_cut_usage = core_ctx.is_multitree_bab

        bab_args = arguments.Config["bab"]
        biccos_args = bab_args["cut"]["biccos"]
        biccos_heuristic = biccos_args["heuristic"]

        # If disable_constraint_strengthening, set iter_idx to a very large value
        # to skip inference proceture
        effective_iter_idx = iter_idx if biccos_args["constraint_strengthening"] else float("inf")
        mock_ret = {
            "betas": new_beta_no_masked,
            "lower_bounds": {self.final_name: core_ctx.lb},
        }
        mock_d = {
            "lower_bounds": d["lower_bounds"],  # old lower_bounds
            "upper_bounds": d["upper_bounds"],  # old upper_bounds
            "betas": d["betas"],  # old betas
            "history": d["history"],  # new history
            "depths": d["depths"],  # new depths
            "alphas": d["alphas"],  # old alphas
            "thresholds": d["thresholds"],  # old thresholds
            "split_history": new_split_history_no_masked,  # new split history
            "lAs": {},  # referenced but unused
            "cs": d["cs"],
            "x_Ls": d["x_Ls"],
            "x_Us": d["x_Us"],
        }
        self.biccos.update_cut(
            d=mock_d,
            net=self,
            ret=mock_ret,
            recorder=core_ctx.recorder,
            enforce_usage=enforce_cut_usage,
            heuristic=biccos_heuristic,
            iter_idx=effective_iter_idx,
            domain_visited=visited_num,
        )

    ###### perform masking for other return values
    depths = torch.as_tensor(d["depths"], dtype=torch.int32)

    masked_lb: torch.Tensor = mask_tensor_first_dim(core_ctx.lb, mask_unverified)
    masked_ub = mask_tensor_first_dim(core_ctx.ub, mask_unverified)
    masked_lb_last = mask_tensor_first_dim(core_ctx.lb_last, mask_unverified)
    masked_ub_last = mask_tensor_first_dim(core_ctx.ub_last, mask_unverified)
    masked_c = mask_tensor_first_dim(core_ctx.c, mask_unverified)
    masked_new_x_Ls = mask_tensor_first_dim_allow_none(
        core_ctx.new_x_Ls, mask_unverified
    )
    masked_new_x_Us = mask_tensor_first_dim_allow_none(
        core_ctx.new_x_Us, mask_unverified
    )
    masked_history = mask_list(d["history"], mask_unverified_cpu)
    masked_depths = mask_tensor_first_dim(depths, mask_unverified_cpu)
    masked_decision_thresh = mask_tensor_first_dim(
        core_ctx.decision_thresh, mask_unverified
    )

    # modify required fields in d
    d["thresholds"] = masked_decision_thresh
    d["history"] = masked_history
    d["depths"] = masked_depths.tolist()

    # prune unnecessary fields in d to make sure no one uses them by mistake
    for k in list(d.keys()):
        if k not in ["history", "depths", "thresholds"]:
            d.pop(k)

    self.timer.start("decision_precomputeing")
    # use unverified domains to get split decision and clip decision
    branching_decision, precompute_clip_decisions = precompute_decision(
        net=self,
        # data needed for precompute
        lb=masked_lb,
        ub=masked_ub,
        working_interm_bounds=masked_working_interm_bounds,
        masked_lAs=masked_lAs,
        working_alpha=masked_working_alpha,
        domain_beta_this_iter=masked_domain_beta_this_iter,
        depths=masked_depths,
        thresholds=masked_decision_thresh,
        history=masked_history,
        c=masked_c,
        # batch sizes
        batch_size=num_unverified,
        batch_device_limit=core_ctx.batch_device_limit,
        # branching decisiions and clip decisions
        domain_clip_scorer=core_ctx.domain_clip_scorer,
        branching_heuristic=core_ctx.branching_heuristic,
        # flags that precompute need to know
        precompute_bfs_flag=core_ctx.precompute_bfs_flag,
        enable_clip_domains=enable_clip_domains,
    )
    self.timer.add("decision_precomputeing")

    emptied_lA = BatchedlA(
        {},
        is_emptied=True,
    )

    # return masked results
    return UpdateBoundCoreReturn(
        lb=masked_lb,
        ub=masked_ub,
        lb_last=masked_lb_last,
        ub_last=masked_ub_last,
        nums_effective_beta_per_domain=masked_nums_effctive_beta_per_domain,
        input_split_idx=None,  # uesless
        primal_x=None,  # uesless
        x_Ls=None,
        x_Us=None,
        new_x_Ls=masked_new_x_Ls,
        new_x_Us=masked_new_x_Us,
        c=masked_c,
        working_beta=masked_working_beta,
        working_alpha=masked_working_alpha,
        working_interm_bounds=masked_working_interm_bounds,
        batched_lA=emptied_lA,
        branching_decision=branching_decision,
        sub_domain_clip_decisions=precompute_clip_decisions,
        decision_thresh=masked_decision_thresh,
        lb_final_max=lb_final_max,
        lb_final_min=lb_final_min,
        n_verified=n_verified,
        n_splits=n_splits,
        new_split_history=masked_split_history,
        depths=masked_depths,
        thresholds=masked_decision_thresh,
        history=masked_history,
    )
