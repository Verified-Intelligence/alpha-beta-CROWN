#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for input space split."""
from typing import Tuple, Union, Callable, List, Optional
import time
import torch
from torch import Tensor
import math
import sys
import os

import arguments
from beta_CROWN_solver import LiRPANet
from auto_LiRPA.utils import (stop_criterion_batch_any, AutoBatchSize)
from auto_LiRPA import BoundedTensor
from auto_LiRPA.concretize_func import construct_constraints

from utils import check_auto_enlarge_batch_size, Stats

from input_split.attack_in_input_split import (pgd_attack_on_domains,
                                            update_rhs_with_attack)
from input_split.branching_heuristics import input_split_branching
from input_split.clip import clip_domains, deconstruct_bias
from input_split.split import input_split_and_repeat
from input_split.utils import initial_verify_criterion
from input_split.branching_domains import UnsortedInputDomainList, UnsortedMultiSpecInputDomainList

def batch_verification_input_split(
        d, net, batch, num_iter, decision_thresh, shape=None,
        bounding_method="crown", branching_method="sb",
        stop_criterion=stop_criterion_batch_any, split_partitions=2,
        use_reordered_bab=False, stats=None):
    """
    General function of the batch_verification_input_split method.

    @param d:                   Domain list
    @param net:                 Bounded neural network
    @param batch:               Number of effective batches to evaluate
    @param num_iter:            The current iteration number of the input BaB run
    @param decision_thresh:     The specification threshold to verify against
    @param shape:               The shape of the network's input
    @param bounding_method:     The method to use when bounding the subdomains of the network
    @param branching_method:    The branching heuristic to use when splitting on input dimensions
    @param stop_criterion:      Criterion to stop naive lower bound of network
    @param split_partitions:    The number of partitions to create for subdomains, currently is always 2 for input split
    @param use_reordered_bab:   Whether to use the reordered BaB order
        - If True, the order of operations is:
            - Reorder:
                pickout -> (attack) -> bounding -> filter -> split -> decision -> (clip) -> add_domain
            - Regular:
                pickout -> (attack) -> split -> bounding -> filter -> (clip) -> decision -> add_domain
    @param stats:               Statistics object to record the time and visited domains

    @return:                   The global lower bound of the current batch
    """
    # --- INITIALIZATION STEP --- #
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    split_hint = input_split_args['split_hint']
    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    enable_clip_domains = clip_input_args['enabled']
    clip_iters = clip_input_args['clip_iterations']
    clip_calculate_volume_metrics = clip_input_args['clip_calculate_volume_metrics']
    enable_constrained_concretize = enable_clip_domains and (clip_input_args["clip_type"] == "complete")

    # --- PICKOUT STEP --- #
    stats.timer.start("pickout")
    ret = d.pick_out_batch(batch, device=net.device)
    alphas, dm_lb, x_L, x_U, cs, thresholds, constraints, spec_sizes, split_idx = ret
    pickout_batch = len(x_L)
    print(f"Current pickout batch: {pickout_batch}")
    stats.timer.add('pickout')

    # --- ATTACK STEP (If required) --- #
    if input_split_args["update_rhs_with_attack"]:
        assert spec_sizes.unique().numel() == 1, "Currently only support same number of AND specs for update_rhs_with_attack"
        stats.timer.start('update_rhs_with_attack')
        if arguments.Config['model']['with_jacobian']:
            model_to_attack = net.net
        else:
            model_to_attack = net.model_ori
        thresholds = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model_to_attack)
        stats.timer.add('update_rhs_with_attack')

    # --- SPLIT STEP (Normal) --- #
    if not use_reordered_bab:
        stats.timer.start('split')
        if not input_split_args["compare_with_old_bounds"]:
            dm_lb = None
        x_L, x_U, cs, thresholds, dm_lb, alphas, spec_sizes, _, _, constraints = input_split_and_repeat(
            x_L, x_U, pickout_batch, shape, split_idx, split_partitions, split_hint,
            cs, thresholds, dm_lb, alphas, spec_sizes, None, None, constraints)
        stats.timer.add('split')

    # --- BOUNDING STEP --- #
    dm_lb, alphas, lA, lbias = get_bounds(
        net, dm_lb, x_L, x_U, alphas, cs, thresholds, constraints,
        bounding_method, branching_method, stop_criterion, stats
    )

    assert not dm_lb.isnan().any()
    # When the clip type is complete, linear constraints will be applied when concretizing.
    if enable_constrained_concretize:
        # --- Initialize constraints
        batch_size = x_L.shape[0]
        x_dim = x_L.flatten(1).shape[1]
        constraints = construct_constraints(lA, lbias, thresholds, batch_size, x_dim)
    else:
        constraints = None

    # --- Filter out verified subdomains --- #
    stats.timer.start('filtering')
    # Since we have only bounded the domains and not clipped them, we only need to check thresholds
    ret_filt = UnsortedInputDomainList.filter_verified_domains(len(x_L), dm_lb, x_L, x_U,
                                        alphas, cs, thresholds, lA, lbias, constraints, spec_sizes = spec_sizes,
                                        check_dm_lbs=True, check_input_boxes=False, use_alpha=d.use_alpha)
    num_unverified_domains, dm_lb, x_L, x_U, alphas, cs, thresholds, lA, lbias, constraints, _, spec_sizes = ret_filt
    stats.timer.add('filtering')

    # When num_unverified_domains > 0, there are still unverified subdomains after filtering.
    # It will be processed to decision and add to domain in the following steps.
    if num_unverified_domains > 0:

        # --- CLIP STEP (Normal) (If required) --- #
        if not use_reordered_bab and enable_clip_domains:
            stats.timer.start("clip")
            x_L, x_U = clip_domains(x_L, x_U, thresholds, lA, lbias,
                                num_iters=clip_iters,
                                calculate_volume=clip_calculate_volume_metrics)
            stats.timer.add("clip")

        # --- DECISION STEP --- #
        stats.timer.start("decision")
        split_idx = input_split_branching(
                net, dm_lb, x_L, x_U, lA, thresholds,
                branching_method, stats.storage_depth, num_iter=num_iter
            )
        stats.timer.add("decision")

        if use_reordered_bab:

            # --- SPLIT STEP (Reordered) --- #
            stats.timer.start('split')
            # This workload consists of all pending items (num_unverified_domains + len(d))
            # but it will not exceed the system's batch limit.
            if not input_split_args["compare_with_old_bounds"]:
                dm_lb = None
            x_L, x_U, cs, thresholds, dm_lb, alphas, spec_sizes, lA, lbias, constraints = input_split_and_repeat(
                    x_L, x_U, num_unverified_domains, shape, split_idx, split_partitions,
                    split_hint, cs, thresholds, dm_lb, alphas, spec_sizes, lA, lbias, constraints)
            stats.timer.add('split')

            # --- CLIP STEP (Reordered) (If required) --- #
            if enable_clip_domains:
                stats.timer.start('clip')
                x_L, x_U = clip_domains(x_L, x_U, thresholds, lA, lbias,
                                        num_iters=clip_iters,
                                        calculate_volume=clip_calculate_volume_metrics)
                stats.timer.add('clip')

        # --- ADD DOMAIN STEP --- #
        stats.timer.start('add_domain')
        # Clipping only updates the input bounds but not the dm_lb
        split_idx = None if use_reordered_bab else split_idx
        d.add(dm_lb, x_L.detach(), x_U.detach(),
                alphas, cs, thresholds,
                constraints=constraints, split_idx=split_idx,
                check_dm_lbs=False, check_input_boxes=enable_clip_domains,
                spec_sizes = spec_sizes)
        stats.timer.add('add_domain')

    # --- SUMMARIZATION STEP --- #
    stats.timer.start('summary')
    len_domains = len(d)

    if len_domains == 0:
        print("No domains left, verification finished!")
        if dm_lb is not None and len(dm_lb) > 0:
            dm_lb_min = dm_lb.min().item()
            print(f"The lower bound of last batch is {dm_lb_min}")
        _print_final_results(stats, 0)
        return decision_thresh.max() + 1e-7
    else:
        if input_split_args["skip_getting_worst_domain"]:
            # It can be costly to call get_topk_indices when the domain list is long
            worst_idx = 0
        else:
            worst_idx = d.get_topk_indices()
        worst_val = d[worst_idx]
        # worst_val[0] is the lower bound, worst_val[-2] is the rhs
        global_lb = worst_val[0] - worst_val[-2]
        _print_final_results(stats, len_domains)
        if not input_split_args["skip_getting_worst_domain"]:
            if 1 <= global_lb.numel() <= 5:
                print(f"Current (lb-rhs): {global_lb}")
            else:
                print(f"Current (lb-rhs): {global_lb.max().item()}")

    if input_split_args["show_progress"]:
        print(f"Progress: {d.get_progess():.10f}")
    sys.stdout.flush()

    return global_lb

def _print_final_results(stats: Stats, len_domains: int):
    stats.timer.add('summary')
    stats.timer.print()
    print("Length of domains:", len_domains)
    print(f"{stats.visited} domains visited")

def get_bounds(net: LiRPANet, dm_lb: Tensor, x_L: Tensor, x_U: Tensor, alphas: List, cs: Tensor,
               thresholds: Tensor, constraints: Optional[tuple],
               bounding_method: str, branching_method: str, stop_func: Callable, stats: Stats
              ) -> Tuple[Tensor, Tensor, Tensor, List, Tensor]:
    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    enable_clip_domains = clip_input_args["enabled"]
    # need lA and lbias to shrink domains
    return_A = branching_method != 'naive' or enable_clip_domains
    # As of now, no method requires lbias so this is set to False. In the future, this can be set to
    # True based on any heuristic/method that requires it
    return_b = enable_clip_domains
    stats.timer.start("bounding")

    stats.visited += len(x_L)

    ret = net.get_lower_bound_naive(
        dm_lb=dm_lb if input_split_args["compare_with_old_bounds"] else None,
        dm_l=x_L, dm_u=x_U, alphas=alphas,
        bounding_method=bounding_method,
        C=cs, stop_criterion=stop_func, thresholds=thresholds,
        return_A=return_A, return_b=return_b,
        constraints=constraints, stats=stats)
    new_dm_lb, alphas, lA, lbias = ret  # here alphas is a dict
    new_dm_lb = new_dm_lb.to(device=thresholds.device)  # ensures it is on the same device as it may be different
    assert not new_dm_lb.isnan().any()
    stats.timer.add("bounding")
    # Visited domain is now incremented based on the input of get_lower_bound_naive.

    return new_dm_lb, alphas, lA, lbias


def input_bab_parallel(net: LiRPANet, x: BoundedTensor, c: Tensor, rhs: Tensor,
                       or_spec_size: Tensor,
                       reference_dict:Optional[dict]=None,
                       timeout:Optional[float]=None, max_iterations: Optional[int]=None,
                       vnnlib=None, return_domains:bool=False, index: Optional[int] = None,
                       ):
    """Run input split bab.

    """
    stats = Stats()
    start = time.time()

    bab_args = arguments.Config["bab"]
    clip_n_verify_args = bab_args["clip_n_verify"]
    clip_input_args = clip_n_verify_args["clip_input_domain"]
    branching_args = bab_args["branching"]
    input_split_args = branching_args["input_split"]

    batch = arguments.Config["solver"]["batch_size"]
    bounding_method = arguments.Config["solver"]["bound_prop_method"]
    init_bounding_method = arguments.Config["solver"]["init_bound_prop_method"]

    bab_args = arguments.Config["bab"]
    timeout = timeout or bab_args["timeout"]
    max_iterations = max_iterations or bab_args["max_iterations"]
    sort_domain_iter = bab_args["sort_domain_interval"]

    branching_args = bab_args["branching"]
    branching_method = branching_args["method"]

    input_split_args = branching_args["input_split"]
    adv_check = input_split_args["adv_check"]
    split_partitions = input_split_args["split_partitions"]
    presplit_domains = input_split_args["presplit_domains"]
    use_reordered_bab = input_split_args["reorder_bab"]
    split_hint = input_split_args["split_hint"]

    clip_n_verify_args = bab_args["clip_n_verify"]
    clip_input_args = clip_n_verify_args["clip_input_domain"]
    use_clip_domains = clip_input_args["enabled"]
    clip_calculate_volume_metrics = clip_input_args["clip_calculate_volume_metrics"]
    constrained_concretize = use_clip_domains and clip_input_args["clip_type"] == "complete"
    clip_neuron_selection_value = clip_input_args["clip_neuron_selection_value"]
    clip_neuron_selection_type = clip_input_args["clip_neuron_selection_type"]

    pgd_order = arguments.Config["attack"]["pgd_order"]

    enable_check_adv = arguments.Config["attack"]["input_split_check_adv"]["enabled"]
    enable_check_adv = (
        pgd_order != "skip" if enable_check_adv == "auto"
        else enable_check_adv == "true"
    )

    # In regular input BaB, the total number of subquestions equals batch_size * 2 after splitting.
    # In reordered BaB, we also ensure the total subquestions equal batch_size * 2,
    # but the input domains are split after the bounding step.
    batch = 2 * batch if use_reordered_bab else batch

    if init_bounding_method == "same":
        init_bounding_method = bounding_method

    use_alpha = init_bounding_method.lower() == "alpha-crown" or bounding_method == "alpha-crown"

    stop_criterion = stop_criterion_batch_any

    # get input domain lower/upper limits
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    def _broadcast_dm(dm):
        if dm.shape[0] == 1 and c.shape[0] > 1:
            dm = dm.expand(c.shape[0], *[-1] * (dm.ndim - 1))
        assert dm.shape[0] == c.shape[0]
        return dm

    # c is expanded in build(). dm_l(u) must be expanded as well
    dm_l = _broadcast_dm(dm_l)
    dm_u = _broadcast_dm(dm_u)

    if (dm_u - dm_l > 0).int().sum() == 1:
        branching_method = "naive"

    # Since we always enable incomplete verification by default,
    # it always has reference_dict.
    if reference_dict:
        ret = reference_dict
        global_lb = ret['global_lb']
        # format alphas. extract only alpha values from the complete alpha dict
        alphas = {}
        if use_alpha:
            alphas = {k: {kk: vv for kk, vv in v['alpha'].items()} for k, v in ret['alphas'].items()}

    else:
        # FIXME: This branch should not be used by default and is only for backup.
        # Maybe it can be removed in the future.
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        global_lb, ret = net.build(
            x, c, rhs, stop_criterion,
            bounding_method=init_bounding_method, return_A=False)
        if use_alpha:
            alphas = ret['alphas']

    stats.visited += len(global_lb)

    # check if the initial global lower bound is verified
    initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)
    if initial_verified:
        return global_lb.max(), stats.visited, "safe"

    result = "unknown"

    # Set the output activations for heuristic driven constrained concretize.
    # Please check auto_LiRPA/concretize_func.py
    for node in net.net.get_splittable_activations():
        for preact_node in node.inputs:
            if preact_node.output_activations is not None:
                preact_node.output_activations.append(node)
            else:
                preact_node.output_activations = [node]

    # Set the neuron_selection_value for complete clipping
    net.net.clip_neuron_selection_value = clip_neuron_selection_value
    net.net.clip_neuron_selection_type = clip_neuron_selection_type
    if clip_neuron_selection_type == "ratio":
        assert clip_neuron_selection_value <= 1.0, "Neuron selection ratio should be smaller than 1.0! "
    else:
        assert isinstance(clip_neuron_selection_value, int), "Neuron selection number should be an integer! "

    # compute storage depth
    min_batch_size = (
            arguments.Config["solver"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(split_partitions)), 1)
    stats.storage_depth = min(max_depth, dm_l.shape[-1])

    # get lA if needed
    lA = ret['lA'].get(net.net.input_name[0], None) # IF bound_prop_method is IBP, lA is None
    if lA is None:
        assert branching_method != "sb", "sb heuristic cannot be used without lA."
        assert not use_clip_domains, "clip domains cannot be used without lA."

    alphas = {}
    if use_alpha:
        alphas = {k: {kk: vv for kk, vv in v['alpha'].items()} for k, v in ret['alphas'].items()}

    # initialize the input domain list
    same_or_spec_size = (or_spec_size == or_spec_size[0]).all()
    if same_or_spec_size:
        domains = UnsortedInputDomainList(
            stats.storage_depth,
            arguments.Config["general"]["device"],
            use_alpha=use_alpha,
            sort_index=input_split_args["sort_index"],
            sort_descending=input_split_args["sort_descending"],
            use_split_idx=not use_reordered_bab
        )
    else:
        print(f"Handling general specifications in input BaB.")
        assert not constrained_concretize, "Currently constrained concretization is not supported for general specifications."
        # the remaining properties have different number of AND clauses
        domains = UnsortedMultiSpecInputDomainList(
            or_spec_size,
            x.shape[1:],  # input shape of the network
            c.shape[2],  # output dimension of the network before applying C
            stats.storage_depth,
            arguments.Config["general"]["device"],
            use_alpha=use_alpha,
            sort_index=input_split_args["sort_index"],
            sort_descending=input_split_args["sort_descending"],
            use_split_idx=not use_reordered_bab,
            alpha_final_name=net.final_name if use_alpha else None,
        )
    if use_reordered_bab:
        # filter out verified subdomains
        lbias = deconstruct_bias(dm_l, dm_u, lA, global_lb)

        constraints = None
        # Construct constraints matrices if use complete clip.
        if constrained_concretize:
            # Constraints matrices will come from the output node A matrices and b matrics from last CROWN call
            current_batch_size = dm_l.shape[0]
            current_x_dim = dm_l.view((current_batch_size, -1)).shape[1]
            constraints = construct_constraints(lA, lbias, rhs, current_batch_size, current_x_dim)

        num_unverified_domains = len(remaining_index)
        spec_sizes = or_spec_size

        # compute initial split idx
        split_idx = input_split_branching(
            net, global_lb, dm_l, dm_u, lA, rhs, branching_method, stats.storage_depth, num_iter=0)

        # perform the initial split on the domains
        dm_l, dm_u, c, rhs, global_lb, alphas, spec_sizes, lA, lbias, constraints = input_split_and_repeat(
            dm_l, dm_u, num_unverified_domains, x.shape, split_idx, split_partitions, split_hint,
            c, rhs, global_lb, alphas, spec_sizes, lA, lbias, constraints)
        # shrink the initial dm_l and dm_u
        if use_clip_domains:
            dm_l, dm_u = clip_domains(dm_l, dm_u, rhs, lA, lbias, calculate_volume=clip_calculate_volume_metrics)
        domains.add(global_lb, dm_l.detach(), dm_u.detach(),
                    alphas, c, rhs, constraints=constraints, split_idx=None, remaining_index=None,
                    check_dm_lbs=False, check_input_boxes=use_clip_domains,
                    spec_sizes=spec_sizes
                    )
    else:
        # compute initial split idx
        split_idx = input_split_branching(
            net, global_lb, dm_l, dm_u, lA, rhs, branching_method, stats.storage_depth, num_iter=0)

        constraints = None
        # Construct constraints matrices if constrained concretize is enabled.
        if constrained_concretize:
            # Constraints matrices will come from the output node A matrices and b matrics from last CROWN call
            lbias = deconstruct_bias(dm_l, dm_u, lA, global_lb)
            current_batch_size = dm_l.shape[0]
            current_x_dim = dm_l.view((current_batch_size, -1)).shape[1]
            constraints = construct_constraints(lA, lbias, rhs, current_batch_size, current_x_dim)

        if use_clip_domains and not use_reordered_bab:
            dm_l, dm_u = clip_domains(dm_l, dm_u, rhs, lA, None, global_lb, calculate_volume=clip_calculate_volume_metrics)
        domains.add(global_lb, dm_l.detach(), dm_u.detach(),
                    alphas, c, rhs, constraints=constraints, split_idx=split_idx,
                    remaining_index=remaining_index, check_input_boxes=use_clip_domains,
                    spec_sizes=or_spec_size
                    )

    if pgd_order == "after":
        if arguments.Config['model']['with_jacobian']:
            model_to_attack = net.net
        else:
            model_to_attack = net.model_ori
        # The domain list currently only has the original unverified domains, we can perform
        # pgd attacks on them directly
        if pgd_attack_on_domains(model_to_attack, domains, 'init', vnnlib):
            print("pgd attack succeed in input_bab_parallel")
            result = "unsafe"

    if presplit_domains:
        assert not use_alpha
        assert same_or_spec_size, "Currently only support same number of AND specs"
        load_presplit_domains(
            domains, net, bounding_method, branching_method, stop_criterion, stats, index=index,
        )

    auto_batch_size = AutoBatchSize(
        batch, net.device,
        enable=arguments.Config["solver"]["auto_enlarge_batch_size"])

    num_iter = 1
    while (result == "unknown" and len(domains) > 0
           and (max_iterations == -1 or num_iter <= max_iterations)):
        print(f"Iteration {num_iter}")
        # sort the domains every certain number of iterations
        if sort_domain_iter > 0 and num_iter % sort_domain_iter == 0:
            stats.timer.start("sort_domains")
            domains.sort()
            stats.timer.add("sort_domains")

        if enable_check_adv:
            if adv_check != -1 and stats.visited >= adv_check:
                stats.timer.start("adv_check")
                # check whether adv example found
                if arguments.Config['model']['with_jacobian']:
                    model_to_attack = net.net
                else:
                    model_to_attack = net.model_ori
                if pgd_attack_on_domains(model_to_attack, domains, 'bab', vnnlib):
                    return global_lb.max(), stats.visited, "unsafe"
                stats.timer.add("adv_check")

        batch_ = batch
        if branching_method == "brute-force" and num_iter <= input_split_args["bf_iters"]:
            batch_ = input_split_args["bf_batch_size"]
        auto_batch_size.record_actual_batch_size(min(batch_, len(domains)))
        global_lb = batch_verification_input_split(
            domains, net, batch_,
            num_iter=num_iter, decision_thresh=rhs, shape=x.shape,
            bounding_method=bounding_method, branching_method=branching_method,
            stop_criterion=stop_criterion, split_partitions=split_partitions,
            use_reordered_bab=use_reordered_bab, stats=stats)
        batch = check_auto_enlarge_batch_size(auto_batch_size)


        if time.time() - start > timeout:
            print("Time out!")
            break

        print(f"Cumulative time: {time.time() - start}\n")
        num_iter += 1

    if result == "unknown" and len(domains) == 0:
        result = "safe"

    if return_domains:
        assert same_or_spec_size, "Currently only support same number of AND specs"
        # Thresholds may have been updated by PGD attack so that different
        # domains may have different thresholds. Restore thresholds to the
        # default RHS for the sorting.
        # FIXME:
        #   We cannot return domains when using UnsortedMultiSpecInputDomainList with this
        #   sort of indexing.
        domains.threshold._storage.data[:] = rhs
        domains.sort()
        if return_domains == -1:
            return_domains = len(domains)
        lower_bound, x_L, x_U = domains.pick_out_batch(
            return_domains, device="cpu")[1:4]
        return lower_bound, x_L, x_U
    else:
        del domains
        return global_lb.max(), stats.visited, result


def load_presplit_domains(domains: Union[UnsortedInputDomainList, UnsortedMultiSpecInputDomainList],
                          net: LiRPANet, bounding_method: str, branching_method: str,
                          stop_criterion: Callable, stats: Stats, index: int):
    """

    :param domains:             Input domain list for storing subdomains during BaB.
    :param net:                 AutoLiRPA net object. This is a mutable object.
    :param bounding_method:     Bounding method to be used the original domain has been bounded and split.
    :param branching_method:    Input BaB branching method i.e. naive, sb, etc.
    :param stop_criterion:           lambda function defining stopping (verification) criteria
    """
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    use_reordered_bab = input_split_args["reorder_bab"]

    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    enable_clip_domains = clip_input_args["enabled"]
    batch_size = arguments.Config["solver"]["batch_size"]
    batch_size = batch_size*2 if use_reordered_bab else batch_size
    ret = domains.pick_out_batch(len(domains))
    alphas, dm_lb, x_L, x_U, cs, thresholds, _, _, split_idx = ret
    batch_size = max(1, batch_size // dm_lb.shape[0])

    if alphas:
        raise NotImplementedError

    if input_split_args["presplit_domains"].endswith(".pt"):
        # There should be only a single instance if the presplit domains is a single file
        assert index == 0
        load_path = input_split_args["presplit_domains"]
    else:
        # One presplit domain list for each instance
        load_path = os.path.join(input_split_args["presplit_domains"], f"{index}.pt")
    print("Loading pre-split domains from", load_path)
    presplit_dm_l, presplit_dm_u = torch.load(load_path)
    presplit_dm_l = presplit_dm_l.to(dm_lb)
    presplit_dm_u = presplit_dm_u.to(dm_lb)
    num_presplit_domains = presplit_dm_l.shape[0]
    print(f"Loaded {num_presplit_domains} pre-split domains")

    num_batches = (num_presplit_domains + batch_size - 1) // batch_size
    for i in range(num_batches):
        print(f"Pre-split domains batch {i+1}/{num_batches}:")
        x_L = presplit_dm_l[i*batch_size:(i+1)*batch_size]
        x_U = presplit_dm_u[i*batch_size:(i+1)*batch_size]
        size = x_L.shape[0]

        x_L = x_L.view(x_L.shape[0], -1, x_L.shape[-1]).repeat(1, dm_lb.shape[0], 1).view(-1, x_L.shape[-1])
        x_U = x_U.view(x_U.shape[0], -1, x_U.shape[-1]).repeat(1, dm_lb.shape[0], 1).view(-1, x_U.shape[-1])
        dm_lb_ = dm_lb.repeat(size, 1)
        cs_ = cs.repeat(size, 1, 1)
        thresholds_ = thresholds.repeat(size, 1)

        new_dm_lb, alphas, lA, lbias = get_bounds(
            net, dm_lb_, x_L, x_U, alphas, cs_, thresholds_, None,
            bounding_method, branching_method, stop_criterion, stats=stats
        )

        if enable_clip_domains:
            clip_iters = clip_input_args["clip_iterations"]
            clip_calculate_volume_metrics = clip_input_args["clip_calculate_volume_metrics"]
            stats.timer.start("clip")
            ret = clip_domains(x_L, x_U, thresholds_, lA, lbias,
                               dm_lb=new_dm_lb, num_iters=clip_iters,
                               calculate_volume=clip_calculate_volume_metrics)
            x_L, x_U = ret
            stats.timer.add("clip")

        stats.timer.start("decision")
        split_idx = input_split_branching(
            net, new_dm_lb, x_L, x_U, lA, thresholds_,
            branching_method, stats.storage_depth, num_iter=1
        )
        stats.timer.add("decision")

        num_domains_pre = len(domains)
        domains.add(new_dm_lb, x_L, x_U, alphas, cs_, thresholds_,
                    split_idx = None if use_reordered_bab else split_idx)
        print(f"{len(domains) - num_domains_pre} domains added, "
              f"{len(domains)} in total")
        print()

    print(f"{len(domains)} pre-split domains added out of {presplit_dm_l.shape[0]}")
    verified_ratio = 1 - len(domains) * 1. / presplit_dm_l.shape[0]
    print(f"Verified ratio: {verified_ratio}")
