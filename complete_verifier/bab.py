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
"""Branch and bound for activation space split."""
import time
import numpy as np
import torch
import copy
import warnings
import functools
from typing import Optional
from branching_domains import (
    ShallowFirstBatchedDomainList,
    check_worst_domain,
)
from beta_CROWN_solver import LiRPANet
from auto_LiRPA.utils import (
    stop_criterion_batch_any,
    multi_spec_keep_func_all,
)
from domain_clipper import DomainClipScorer
from heuristics import get_branching_heuristic
from heuristics.decision_types import BranchingDecisions
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_utils import cplex_update_general_beta
from utils import print_splitting_decisions
from prune import prune_alphas
import arguments
from state import WorkingIntermBoundsInfo
from activation_split.update_bounds_phases import update_bounds_pre, update_bounds_core, update_bounds_post


def deprecated(message: str):
    """Simple local deprecation decorator to avoid depending on typing_extensions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_split_depth(batch_size, min_batch_size, min_depth):
    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    if batch_size < min_batch_size:
        # Split multiple levels, to obtain at least min_batch_size domains in this batch.
        return max(
            min_depth,
            int(np.log(min_batch_size / max(min_depth, batch_size)) / np.log(2)),
        )
    else:
        return min_depth


def determine_decision_precompute(is_multitree_bab):
    enable_decision_precompute = (
        not is_multitree_bab
        and (arguments.Config["bab"]["branching_decision_precompute"]["enable"])
        # arguments.Config['bab']['branching_decision_precompute']['enable'] is True by default
    )
    print(f"{enable_decision_precompute=}")
    return enable_decision_precompute

#! README: Deprecated. Do not write new code to this function. Read the following warning.
@deprecated(
    "split_domain is duplicate and will be removed in the future. "
    "it exists only for multi_tree_bab."
    "If you want to add new features to ReLU BaB, please write to"
    " activation_split.stage_preprocess/solve/postprocess"
)
def split_domain(
    net: LiRPANet,
    domains,
    d,
    device_batch_limit,
    iter_idx,
    stats,
    set_init_alpha=False,
    fix_interm_bounds=True,
    branching_heuristic=None,
    domain_clip_scorer: Optional[DomainClipScorer] = None,
    is_multitree_bab=False,
):
    solver_args = arguments.Config["solver"]
    bab_args = arguments.Config["bab"]
    branch_args = bab_args["branching"]
    biccos_args = bab_args["cut"]["biccos"]
    biccos_enable = biccos_args["enabled"]
    biccos_heuristic = biccos_args["heuristic"]
    enable_clip_domains = bab_args["clip_n_verify"]["clip_interm_domain"][
        "enabled"
    ] and not (isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs)
    stop_func = stop_criterion_batch_any

    min_batch_size = min(
        solver_args["min_batch_size_ratio"] * solver_args["batch_size"],
        device_batch_limit,
    )
    batch_selected_domain = next(iter(d["lower_bounds"].values())).shape[0]
    print("batch:", batch_selected_domain)

    stats.timer.start("decision")
    stats.timer.start("branching_decision")

    enable_decision_precompute = determine_decision_precompute(is_multitree_bab)

    # precompute branching decision if enabled.
    # The first iteration does not precompute.
    assert isinstance(iter_idx, int)
    if enable_decision_precompute:
        branching_decision = BranchingDecisions.from_batch_first_decisions(
            domain_decision=d["batch_first_branching_decisions"],
            strict=False,
        )
        branching_decision, branching_points, split_depth = (
            branching_decision.branching_decision,
            branching_decision.branching_points,
            branching_decision.split_depth,
        )

    else:
        # Only for Multi-Tree-Search, we need to calculate the depth of the BFS.
        # Check if 'domains' is an instance of ShallowFirstBatchedDomainList
        # and that we are using a BFS
        if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
            target_batch_size = biccos_args["multi_tree_branching"]["target_batch_size"]
            keep_n_best_domains = biccos_args["multi_tree_branching"]["keep_n_best_domains"]
            # Ensure that the target batch size is at least as large as
            # the number of domains we wish to keep.
            # This is a sanity check to prevent a configuration error.
            assert target_batch_size >= keep_n_best_domains
            # This prevents a division by zero error in the next step.
            assert batch_selected_domain >= 1
            # The 'depth' represent how many levels will be used in the MTS process.
            depth = target_batch_size // batch_selected_domain
            # Ensure that the computed depth is a positive integer.
            # This assertion guarantees that there will be at least one level of processing.
            assert depth > 0
        else:
            depth = 1
        # Calculate the split depth.
        # Increase the maximum number of candidates for fsb and kfsb
        # if there are more splits needed.
        split_depth = get_split_depth(batch_selected_domain, min_batch_size, depth)
        d["mask"] = WorkingIntermBoundsInfo.from_two_dicts(
            d["lower_bounds"], d["upper_bounds"]
        ).compute_unstable_mask(net)
        _branching_decision = branching_heuristic.compute_branching_decisions(
            d,
            split_depth,
            method=branch_args["method"],
            branching_candidates=max(branch_args["candidates"], split_depth),
            branching_reduceop=branch_args["reduceop"],
            timer=stats.timer,
        )
        branching_decision = _branching_decision.branching_decision
        branching_points = _branching_decision.branching_points
        split_depth = _branching_decision.split_depth
        # note: this condition is practically identical to
        # any(d["depths"] == net.tot_ambi_nodes).
        # compute_branching_decisions will return an empty
        # decision if any(d["depths"] == net.tot_ambi_nodes) met.
        if len(branching_decision) == 0:
            print("all nodes are split!!")
            print(f"{stats.visited} domains visited")
            stats.all_node_split = True
            stats.all_split_result = "unknown"
            if not solver_args["beta-crown"]["all_node_split_LP"]:
                global_lb = d["global_lb"][0] - d["thresholds"][0]
                for i in range(1, len(d["global_lb"])):
                    if max(d["global_lb"][i] - d["thresholds"][i]) <= max(global_lb):
                        global_lb = d["global_lb"][i] - d["thresholds"][i]
                stats.timer.add("branching_decision")
                return global_lb, torch.inf

    stats.timer.add("branching_decision")

    split = {
        "decision": branching_decision,
        "points": branching_points,
    }
    if split["points"] is not None and not bab_args["interm_transfer"]:
        raise NotImplementedError(
            "General branching points are not supported " "when interm_transfer==False"
        )
    print_splitting_decisions(
        net,
        d,
        split_depth,
        split,
        verbose=arguments.Config["debug"]["print_verbose_decisions"],
    )
    stats.timer.add("decision")

    stats.timer.start("set_bounds")
    if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
        net.build_history_and_set_bounds(d, split, mode="breadth")
    else:
        net.build_history_and_set_bounds(d, split, mode="depth")
    batch_size = len(split["decision"])
    stats.visited += batch_size
    stats.timer.add("set_bounds")
    if branch_args["save_visited_domains_to"] != "":
        stats.domainDB.add_records_from_d(d["history"], d["depths"])
    stats.timer.start("solve")
    # Caution: we use 'all' predicate to keep the domain when multiple specs
    # are present: all lbs should be <= threshold, otherwise pruned
    # maybe other 'keeping' criterion needs to be passed here
    if enable_clip_domains and net.domain_clipper is not None:
        net.domain_clipper.get_stop_criterion_and_iter(stop_func, iter_idx)

    # self.net.update_bounds is broken into 4 parts here:
    # preResults, coreResults, postResults

    # ============================ begin of replacing original update_bounds =====================

    preResults = update_bounds_pre(
        d=d,
        final_name=net.final_name,
        net_c=net.c,
        net_x=net.x,
        dtype_map=net.get_split_node_dtype_map(),
        timer=net.timer,
        device=net.device,
        beta_bias=branching_points is not None,
    )
    # ============================ logic above belongs to pre-processing =========================
    # Solve+Decision
    coreResults = update_bounds_core(
        net=net,
        pre_result=preResults,
        # flags
        fix_interm_bounds=fix_interm_bounds,
        precompute_bfs_flag=isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs,
        is_multitree_bab=is_multitree_bab,
        enable_clip_domains=enable_clip_domains,
        enable_decision_precompute=enable_decision_precompute,
        # numbers and limits
        iter_idx=iter_idx,
        visited_num=stats.visited,
        batch_device_limit=device_batch_limit,
        # functors or objects
        domain_clip_scorer=domain_clip_scorer,
        branching_heuristic=branching_heuristic,
        stop_criterion_func=stop_func(d["thresholds"]),
        multi_spec_keep_func=multi_spec_keep_func_all,
    )
    # ============================ logic below belongs to post-processing ========================
    postResults = update_bounds_post(
        core_result=coreResults,
        timer=net.timer,
        final_name=net.final_name,
        split_node_names=[n.name for n in net.net.split_nodes],
        layers_requiring_bounds_names=[n.name for n in net.net.layers_requiring_bounds],
        unstable_mask=net.unstable_mask,
        interm_transfer=net.interm_transfer,
    )
    ret = postResults.__dict__
    # ============================ end of replacing original update_bounds =========================

    stats.timer.add("solve")

    if solver_args["beta-crown"]["all_node_split_LP"] and torch.any(
        torch.tensor(d["depths"]) == net.tot_ambi_nodes
    ):
        assert net.solver_model_initialized, "already built in update_bounds_core"
        # FIXME build_history_and_set_bounds doesn't return correct split
        # (just dummy elements) when split_depth > 1
        stats.all_split_result = "unknown"
        for k in ["lower_bounds", "upper_bounds"]:
            if k not in d:
                d[k] = ValueError("previous lower/upper bounds are pruned.")
        if batch_verification_all_node_split_LP(
            net,
            d,
            ret,
            split,
            stats,
            working_interm_info=coreResults.working_interm_bounds,
        ):
            stats.all_node_split = True
            stats.all_split_result = "unsafe"
            return torch.inf

    # handle all_node_split case while all_node_split_LP is False.
    if (
        (not solver_args["beta-crown"]["all_node_split_LP"])
        and enable_decision_precompute
        and torch.any(torch.tensor(d["depths"]) == net.tot_ambi_nodes)
    ):
        # Handle all nodes are split before inserting domains to domain list
        print("all nodes are split!!")
        print(f"{stats.visited} domains visited")
        stats.all_node_split = True
        stats.all_split_result = "unknown"

        global_lb = torch.min(
            torch.max(
                ret["lower_bounds"][net.final_name] - d["thresholds"].cpu(), dim=1
            ).values,
            dim=0,
        ).values.item()

        return global_lb, torch.inf

    if set_init_alpha:
        print("Setting the initial alpha")
        ret["alphas"] = prune_alphas(ret["alphas"], net.alpha_start_nodes)
        # We just want the data structure here, not the values
        domains.init_alpha = {
            k: {
                kk: vv[:, :, :1]
                .detach()
                .clone()
                .to(net.device)
                .to(torch.get_default_dtype())
                for kk, vv in v.items()
            }
            for k, v in ret["alphas"].items()
        }
    else:
        if not fix_interm_bounds:
            ret["alphas"] = prune_alphas(ret["alphas"], net.alpha_start_nodes)

    # We have to add cuts now, because domains.add might modify the list of domains in ret
    if (
        ret and bab_args["cut"]["enabled"] and biccos_enable
        and not enable_decision_precompute  # for precompte has its own cut handling
    ):
        # We only enforce cut usage for multi-tree-searching
        enforce_cut_usage = (
            isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs
        )
        # If disable_constraint_strengthening, set iter_idx to a very large value
        # to skip inference procedure
        iter_idx = iter_idx if biccos_args["constraint_strengthening"] else float("inf")
        net.biccos.update_cut(
            d,
            net,
            ret,
            recorder=net.recorder,
            enforce_usage=enforce_cut_usage,
            heuristic=biccos_heuristic,
            iter_idx=iter_idx,
            domain_visited=stats.visited,
        )

    stats.timer.start("add")
    domains.add(ret, d, check_infeasibility=not fix_interm_bounds)
    domains.print()
    stats.timer.add("add")
    del d
    return ret

#! README: Deprecated. Do not write new code to this function. Read the following warning.
@deprecated(
    "act_split_round is duplicate and will be removed in the future, "
    "it exists only for multi_tree_bab."
    "Implementation has been broken down and moved "
    "to branch_and_bound_preprocess/solve/postprocess"
)
def act_split_round(
    domains,
    net: LiRPANet,
    device_batch_limit,
    iter_idx,
    stats=None,
    branching_heuristic=None,
    is_multitree_bab=False,
    domain_clip_scorer=None,
):
    bab_args = arguments.Config["bab"]
    sort_domain_iter = bab_args["sort_domain_interval"]
    recompute_interm = bab_args["recompute_interm"]
    spec_args = arguments.Config["specification"]

    stats.timer.start("pickout")
    d = domains.pick_out(batch=device_batch_limit, device=net.device)
    net.domain_interm_factory.construct_interm_bounds_in_d(d, net.unstable_mask)
    stats.timer.add("pickout")

    # when cplex cut is enabled, for domains with general_beta created for outdated cuts,
    # we need to rewrite it to general_beta for new cuts
    if bab_args["cut"]["enabled"] and bab_args["cut"]["cplex_cuts"]:
        cplex_update_general_beta(net, d)

    split_domain(
        net,
        domains,
        d,
        device_batch_limit,
        stats=stats,
        fix_interm_bounds=not recompute_interm,
        branching_heuristic=branching_heuristic,
        iter_idx=iter_idx,
        is_multitree_bab=is_multitree_bab,
        domain_clip_scorer=domain_clip_scorer,
    )

    print("Length of domains:", len(domains))

    if len(domains) == 0:
        print("No domains left, verification finished!")

    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        stats.timer.start("sort")
        domains.sort()
        stats.timer.add("sort")
    global_lb = check_worst_domain(domains)
    rhs_offset = spec_args["rhs_offset"]
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f"Current (lb-rhs): {global_lb}")
    else:
        print(f"Current (lb-rhs): {global_lb.max().item()}")
    print(f"{stats.visited} domains visited")

    stats.timer.print()
    return global_lb

# TODO: multi_tree_bab need to be refactored to
# branch_and_bound_preprocess/solve/postprocess as well
# TODO: multi_tree_bab currently requires the duplicated act_split_round and split_domain
def multi_tree_bab(
    net: LiRPANet,
    domains,
    batch,
    stop_criterion_func,
    biccos_args,
    stats,
    start_time,
    initial_bs_ratio,
):
    """
    Usually, BaB uses a single binary tree. In multi-tree search, keep track of multiple trees,
    and each node may have multiple children. This allows us to e.g. explore both the splits (A, B),
    (A, C) and (D, C) in parallel. By doing so, we can generate more diverse BICCOS cuts.
    After the multi-tree search terminates, we drop all but one tree, which is pruned to become a
    binary tree. This tree is then used for the rest of the BaB process.
    In each iteration, we select the best n leaf nodes and perform k splits each.

    input:
        net: LirpaNet
        domains: ShallowFirstBatchedDomainList
        batch: int
        stop_criterion_func: callable
        biccos_args: dict
        stats: Stats
        start_time: float
    """
    shallowbranching_heuristic = get_branching_heuristic(net, "kfsb")
    assert len(domains) == 1

    # At the end of the multi-tree search, we have to restore the initial domain
    initial_domain = domains.pick_out(batch=batch, device=net.device)
    net.domain_interm_factory.construct_interm_bounds_in_d(
        initial_domain, net.unstable_mask
    )
    initial_ret = net.update_bounds(
        initial_domain,
        fix_interm_bounds=True,
        stop_criterion_func=stop_criterion_func,
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=False,
        enable_clip_domains=False,
    )

    # We did not perform any decision yet.
    # Copy the decision info from initial_domain, which should also be empty.
    initial_ret["decision_info"] = initial_domain["batch_first_branching_decisions"]
    initial_ret["sub_domain_clip_decisions"] = initial_domain[
        "sub_domain_clip_decisions"
    ]

    domains.add(initial_ret, initial_domain, check_infeasibility=False)

    total_round = 0
    max_iter_shallow = biccos_args["multi_tree_branching"]["iterations"]
    num_domains = len(domains)
    # In rare cases, adding the initial domain back might prove it to be UNSAT.
    # This might happen due to randomness in the gradient updates.
    # If it happens, we're done and don't need to proceed with regular BaB.
    if num_domains == 0:
        global_lb = check_worst_domain(domains)
        return global_lb
    assert num_domains == 1

    # Proceed only if we haven't reached the maximum number of shallow iterations
    # AND either:
    #   1. There is at least one domain available (num_domains > 0)
    #      OR
    #   2. There is at least one backup domain in 'domains.mtb_backup'
    #      AND the backup's 'skip' counter (from the first backup entry) is less than 10,
    #         meaning we haven't skipped it too many times.
    while total_round < max_iter_shallow and (
        num_domains > 0
        or (len(domains.mtb_backup) > 0 and domains.mtb_backup[0]["skip"] < 10)
    ):
        # Increment the total number of iterations/rounds processed.
        total_round += 1

        # If there are no active domains left, try to restore one from the backup list.
        if num_domains == 0:
            # Ensure that there is at least one backup available.
            assert len(domains.mtb_backup) > 0

            # If the most recent backup entry has already been skipped 3 times
            # and there is more than one backup available, then remove it.
            # This prevents repeatedly attempting a backup that's already failed multiple times.
            if domains.mtb_backup[-1]["skip"] == 3 and len(domains.mtb_backup) > 1:
                del domains.mtb_backup[-1]

            # Increase the skip count of the current (last) backup entry.
            domains.mtb_backup[-1]["skip"] += 1

            # Output the current state: how many backup entries remain and the current skip count.
            print(
                "Going back, stack has",
                len(domains.mtb_backup),
                "entries left. Skipping",
                domains.mtb_backup[-1]["skip"],
            )

            try:
                # Attempt to restore a domain by adding the best candidate(s)
                # using a deep copy of the most recent backup entry.
                domains.add_best_k_lower_bounds(**copy.deepcopy(domains.mtb_backup[-1]))
            except ShallowFirstBatchedDomainList.EmptyKLower:
                # If there is nothing left to restore from this backup entry,
                # mark it as fully skipped by setting its skip count to 3.
                # This will force its removal in subsequent iterations.
                domains.mtb_backup[-1]["skip"] = 3
                # Skip the remainder of the current iteration and proceed with the next one.
                continue

        print(f"Shallow-BaB round {total_round}")
        global_lb = act_split_round(
            domains,
            net,
            batch,
            iter_idx=total_round,
            stats=stats,
            branching_heuristic=shallowbranching_heuristic,
            is_multitree_bab=True,
        )
        num_domains = len(domains)
        if num_domains == 0:
            print("No domains left, MTS early stop!")
            break
        print(f"Cumulative time: {time.time() - start_time}\n")

    # Drop current list of domains
    domains.use_bfs = False
    if len(domains) > 0:
        domains.pick_out(batch=len(domains), device=net.device)
        # no need to build interm. return value discarded anyways.

    if not biccos_args["multi_tree_branching"]["restore_best_tree"]:
        domains.add(initial_ret, initial_domain, check_infeasibility=False)
    else:
        print("Restoring the best tree")
        domains.restore_best_domains(initial_ret, initial_domain)
        # We might have added some domains that are UNSAT
        print("Shallow branching resets to n domains: ", len(domains))
        base_d = domains.pick_out(batch=len(domains), device=net.device)
        net.domain_interm_factory.construct_interm_bounds_in_d(base_d, net.unstable_mask)
        new_ret = net.update_bounds(
            base_d,
            fix_interm_bounds=True,
            stop_criterion_func=stop_criterion_func,
            multi_spec_keep_func=multi_spec_keep_func_all,
            beta_bias=False,
            enable_clip_domains=False,
        )
        new_ret["decision_info"] = base_d["batch_first_branching_decisions"]
        new_ret["sub_domain_clip_decisions"] = base_d[
            "sub_domain_clip_decisions"
        ]
        domains.add(new_ret, base_d, check_infeasibility=False)
        print("After pruning, left: ", len(domains))
    if not biccos_args["constraint_strengthening"]:
        arguments.Config["solver"]["min_batch_size_ratio"] = initial_bs_ratio
    print("\n   Back to Regular BaB\n")
    return global_lb
