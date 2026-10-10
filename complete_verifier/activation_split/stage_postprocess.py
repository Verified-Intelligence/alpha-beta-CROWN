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
from branching_domains import BatchedDomainList, check_worst_domain
from activation_split.update_bounds_phases import update_bounds_post
from activation_split.stage_arguments import PostprocessConstArguments, PostprocessMutableArguments
from activation_split.protocols import SolvePacket, PostprocessPacket
from activation_split.return_types import UpdateBoundPostReturn
import arguments


def branch_and_bound_postprocess(
    domains: BatchedDomainList,
    solve_packet: SolvePacket,
    const_args: PostprocessConstArguments,
    mutable_args: PostprocessMutableArguments,
) -> PostprocessPacket:
    """
    Implementation for branch_and_bound_postprocess stage.

    Responsibility of this stage:
    - accept packet from solve.
    - finalize and transfer GPU data to CPU.
    - format, reorganization.
    - add to domain list.
    """
    bab_args = arguments.Config["bab"]
    spec_args = arguments.Config["specification"]
    sort_domain_iter = bab_args["sort_domain_interval"]
    core_result = solve_packet.coreResults

    # Finalize and transfer raw results from solve_packet to domains.add formats.
    postResults: UpdateBoundPostReturn = update_bounds_post(
        core_result=core_result,
        timer=mutable_args.stats.timer,
        final_name=const_args.final_name,
        split_node_names=const_args.split_nodes_names,
        layers_requiring_bounds_names=const_args.layers_requiring_bounds_names,
        unstable_mask=const_args.unstable_mask,
        interm_transfer=const_args.interm_transfer,
    )
    ret = postResults.__dict__

    mutable_args.stats.timer.add("solve")
    mutable_args.stats.timer.start("add")

    domains.add(
        ret,
        {
            "history": core_result.history,
            "depths": core_result.depths,
            "thresholds": core_result.thresholds,
        },
        check_infeasibility=False,
    )
    domains.print()
    mutable_args.stats.timer.add("add")

    if sort_domain_iter > 0 and mutable_args.iter_idx % sort_domain_iter == 0:
        print("Sorting domains...")
        mutable_args.stats.timer.start("sort")
        domains.sort()
        mutable_args.stats.timer.add("sort")

    global_lb = check_worst_domain(domains)
    rhs_offset = spec_args["rhs_offset"]
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f"Current (lb-rhs): {global_lb}")
    else:
        print(f"Current (lb-rhs): {global_lb.max().item()}")
    print(f"{mutable_args.stats.visited} domains visited")
    print("Length of domains:", len(domains))

    return PostprocessPacket(
        global_lb=global_lb,
        exit_status="normal",
    )
