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
from beta_CROWN_solver import LiRPANet, stop_criterion_batch_any
from auto_LiRPA.utils import multi_spec_keep_func_all
from activation_split.protocols import PreprocessPacket, SolvePacket
from activation_split.stage_arguments import SolveConstArguments, SolveMutableArguments
from activation_split.update_bounds_phases import update_bounds_core
import arguments

from cuts.cut_utils import fetch_cut_from_cplex, cplex_update_general_beta
from lp_mip_solver import batch_verification_all_node_split_LP
import torch


def branch_and_bound_solve(
    net: LiRPANet,
    prePacket: PreprocessPacket,
    const_args: SolveConstArguments,
    mutable_args: SolveMutableArguments,
) -> SolvePacket:
    """
    Implementation for branch_and_bound_solve.

    Responsibility:
    - accept packets from preprocess.
    - fetch cuts, apply cuts.
    - execute update_bounds_core to get lower bounds of each domain.
    - check all_node_split status, call LP if needed. Determine early exit status.
    - pass packet to postprocess.
    """
    # ============ Retrieve Arguments ============
    bab_args = arguments.Config["bab"]
    solver_args = arguments.Config["solver"]
    biccos_args = bab_args["cut"]["biccos"]

    # ============ Retrieve Flags ============
    domain_bfs_flag = False
    enable_clip_domains = (
        bab_args["clip_n_verify"]["clip_interm_domain"]["enabled"]
        and not domain_bfs_flag
    )
    fix_interm_bounds = True
    enable_decision_precompute = True

    # ============ Fetch and Process Cuts ============
    if (
        bab_args["cut"]["enabled"]
        and bab_args["cut"]["cplex_cuts"]
        and not biccos_args["enabled"]
    ):
        fetch_cut_from_cplex(net, recorder=net.recorder)
    # here to handle the case where the split node happen to be in the
    # cut constraint !!!

    # when cplex cut is enabled, for domains with general_beta created for outdated cuts,
    # we need to rewrite it to general_beta for new cuts
    if bab_args["cut"]["enabled"] and bab_args["cut"]["cplex_cuts"]:
        cplex_update_general_beta(net, prePacket.preResults.d_dict)

    stop_func = stop_criterion_batch_any
    if enable_clip_domains and net.domain_clipper is not None:
        net.domain_clipper.get_stop_criterion_and_iter(stop_func, prePacket.iter_idx)

    # ============================================================
    # Compute lower bounds for each domain.
    # ============================================================
    coreResults = update_bounds_core(
        net=net,
        pre_result=prePacket.preResults,
        # flags
        fix_interm_bounds=fix_interm_bounds,
        precompute_bfs_flag=domain_bfs_flag,
        is_multitree_bab=False,
        enable_clip_domains=enable_clip_domains,
        enable_decision_precompute=enable_decision_precompute,
        # numbers and limits
        iter_idx=prePacket.iter_idx,
        visited_num=prePacket.num_visited_domains,
        batch_device_limit=prePacket.device_batch_limit,
        # functors or objects
        domain_clip_scorer=const_args.domain_clip_scorer,
        branching_heuristic=const_args.branching_heuristic,
        stop_criterion_func=stop_func(prePacket.preResults.d_dict["thresholds"]),
        multi_spec_keep_func=multi_spec_keep_func_all,
    )

    # ============ Determine all_node_split status for early exit ============
    all_node_split_flag = torch.any(
        torch.tensor(prePacket.preResults.d_dict["depths"]) == net.tot_ambi_nodes
    )

    # early exits
    if solver_args["beta-crown"]["all_node_split_LP"] and all_node_split_flag:
        assert (
            net.solver_model_initialized
        ), "should already built in update_bounds_core"

        fake_split = {}
        temp_lb = coreResults.lb.cpu()
        temp_ub = coreResults.ub.cpu()
        input_output_ret = {
            "lower_bounds": {net.final_name: temp_lb},
            "upper_bounds": {net.final_name: temp_ub},
        }
        temp_d_dict = prePacket.preResults.d_dict.copy()  # make a shallowcopy
        if arguments.Config["bab"]["cut"]["manual_cuts"]:
            temp_d_dict["lower_bounds"] = (
                coreResults.working_interm_bounds.to_lower_bounds_dict()
            )
            temp_d_dict["upper_bounds"] = (
                coreResults.working_interm_bounds.to_upper_bounds_dict()
            )
        else:
            temp_d_dict["lower_bounds"] = {}
            temp_d_dict["upper_bounds"] = {}
        if batch_verification_all_node_split_LP(
            net,
            temp_d_dict,
            input_output_ret,
            fake_split,
            mutable_args.stats,
            working_interm_info=coreResults.working_interm_bounds,
        ):
            early_exit_status = "all_node_split_LP_unsafe"
        else:
            early_exit_status = "normal_exit"
        coreResults.lb = temp_lb.to(coreResults.lb.device)
        coreResults.ub = temp_ub.to(coreResults.ub.device)
    elif (not solver_args["beta-crown"]["all_node_split_LP"]) and all_node_split_flag:
        print("all nodes are split!!")
        print(f"{mutable_args.stats.visited} domains visited")
        early_exit_status = "all_node_split_unknown"
    else:
        early_exit_status = "normal_exit"

    # Return results and early exit status.
    return SolvePacket(
        coreResults=coreResults,
        early_exit_status=early_exit_status,
    )
