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
from branching_domains import (
    BatchedDomainList,
    ShallowFirstBatchedDomainList,
)
from beta_CROWN_solver_utils import build_history_and_set_bounds_static
from heuristics.decision_types import BranchingDecisions
from utils import (
    print_splitting_decisions_impl,
)
from activation_split.protocols import PreprocessPacket
from activation_split.stage_arguments import (
    PreprocessConstArguments,
    PreprocessMutableArguments,
)
from activation_split.return_types import UpdateBoundPreReturn
from activation_split.update_bounds_phases import update_bounds_pre
import arguments


def branch_and_bound_preprocess(
    domains: BatchedDomainList,
    const_args: PreprocessConstArguments,
    mutable_args: PreprocessMutableArguments,
) -> PreprocessPacket:
    """
    Implementation of branch_and_bound_preprocess.

    Responsibility:
    - pickout domain from list.
    - split domain according to decisions and build interm bounds history.
    - call update_bounds_pre to prepare alpha, beta and other important data
      for update_bounds_solve to use.
    - wrap into a packet and send to next stage.

    Note: this function calls update_bounds_pre as an implementation detail
    rather than inlining the logic here, so that it can be reused without
    the surrounding domain-pick/split bookkeeping.
    """
    bab_args = arguments.Config["bab"]
    branch_args = bab_args["branching"]

    stats = mutable_args.stats

    stats.timer.start("pickout")
    d = domains.pick_out(batch=mutable_args.device_batch_limit, device=const_args.device)
    mutable_args.domain_interm_factory.construct_interm_bounds_in_d(
        d, const_args.unstable_mask
    )
    stats.timer.add("pickout")

    stats.timer.start("decision")
    stats.timer.start("branching_decision")

    branching_decision_data = BranchingDecisions.from_batch_first_decisions(
        domain_decision=d["batch_first_branching_decisions"],
        strict=False,
    )

    print("batch:", branching_decision_data.batch_size)

    stats.timer.add("branching_decision")
    split = {
        "decision": branching_decision_data.branching_decision,
        "points": branching_decision_data.branching_points,
    }
    print_splitting_decisions_impl(
        const_args.split_nodes_names,
        d["lower_bounds"],
        d["upper_bounds"],
        branching_decision_data.split_depth,
        split,
        verbose=arguments.Config["debug"]["print_verbose_decisions"],
    )
    if split["points"] is not None and not bab_args["interm_transfer"]:
        raise NotImplementedError(
            "General branching points are not supported " "when interm_transfer==False"
        )
    stats.timer.add("decision")
    stats.timer.start("set_bounds")
    if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
        build_history_and_set_bounds_static(
            d,
            split,
            mode="breadth",
            final_name=const_args.final_name,
            split_nodes_names=const_args.split_nodes_names,
        )
    else:
        build_history_and_set_bounds_static(
            d,
            split,
            mode="depth",
            final_name=const_args.final_name,
            split_nodes_names=const_args.split_nodes_names,
        )
    batch_size_after_split = len(split["decision"])
    stats.visited += batch_size_after_split
    stats.timer.add("set_bounds")

    if branch_args["save_visited_domains_to"] != "":
        stats.domainDB.add_records_from_d(d["history"], d["depths"])

    stats.timer.start("solve")

    preResults: UpdateBoundPreReturn = update_bounds_pre(
        d=d,
        final_name=const_args.final_name,
        net_c=const_args.net_c,
        net_x=const_args.net_x,
        dtype_map=const_args.dtype_map,
        timer=stats.timer,
        device=const_args.device,
        beta_bias=branching_decision_data.branching_points is not None,
    )

    return PreprocessPacket(
        preResults=preResults,
        # book keeping infos
        iter_idx=mutable_args.iter_idx,
        num_visited_domains=stats.visited,
        device_batch_limit=mutable_args.device_batch_limit,
        exit_status="normal",
    )
