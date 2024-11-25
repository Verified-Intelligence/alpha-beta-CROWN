#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for activation space split."""
import time
import numpy as np
import torch
import copy

from auto_LiRPA.utils import stop_criterion_batch_any, multi_spec_keep_func_all
from branching_domains import BatchedDomainList, check_worst_domain
from attack.domains import SortedReLUDomainList
from attack.bab_attack import bab_loop_attack
from heuristics import get_branching_heuristic
from input_split.input_split_on_relu_domains import input_split_on_relu_domains, InputReluSplitter
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_verification import cut_verification, get_impl_params
from cuts.cut_utils import fetch_cut_from_cplex, clean_net_mps_process, cplex_update_general_beta
from utils import (print_splitting_decisions, print_average_branching_neurons,
                   Stats, get_unstable_neurons)
from prune import prune_alphas
import arguments


def get_split_depth(batch_size, min_batch_size):
    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    if batch_size < min_batch_size:
        # Split multiple levels, to obtain at least min_batch_size domains in this batch.
        return max(1, int(
            np.log(min_batch_size / max(1, batch_size)) / np.log(2)))
    else:
        return 1


def split_domain(net, domains, d, batch, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    stop_func = stop_criterion_batch_any

    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    batch = next(iter(d['lower_bounds'].values())).shape[0]
    print('batch:', batch)

    stats.timer.start('decision')
    split_depth = get_split_depth(batch, min_batch_size)
    # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
    branching_decision, branching_points, split_depth = (
        branching_heuristic.get_branching_decisions(
            d, split_depth, method=branch_args['method'],
            branching_candidates=max(branch_args['candidates'], split_depth),
            branching_reduceop=branch_args['reduceop'],
            iter_idx=iter_idx, num_all_domains=len(domains)))
    print_average_branching_neurons(
        branching_decision, stats.implied_cuts, impl_params=impl_params)
    if len(branching_decision) < len(next(iter(d['mask'].values()))):
        print('all nodes are split!!')
        print(f'{stats.visited} domains visited')
        stats.all_node_split = True
        if not solver_args['beta-crown']['all_node_split_LP']:
            global_lb = d['global_lb'][0] - d['thresholds'][0]
            for i in range(1, len(d['global_lb'])):
                if max(d['global_lb'][i] - d['thresholds'][i]) <= max(global_lb):
                    global_lb = d['global_lb'][i] - d['thresholds'][i]
            return global_lb, torch.inf
    split = {
        'decision': branching_decision,
        'points': branching_points,
    }
    if split['points'] is not None and not bab_args['interm_transfer']:
        raise NotImplementedError(
            'General branching points are not supported '
            'when interm_transfer==False')
    print_splitting_decisions(
        net, d, split_depth, split,
        verbose=arguments.Config['debug']['print_verbose_decisions'])
    stats.timer.add('decision')

    stats.timer.start('set_bounds')
    net.build_history_and_set_bounds(d, split, impl_params=impl_params)
    stats.timer.add('set_bounds')
    batch = len(split['decision'])
    stats.timer.start('solve')
    # Caution: we use "all" predicate to keep the domain when multiple specs
    # are present: all lbs should be <= threshold, otherwise pruned
    # maybe other "keeping" criterion needs to be passed here
    ret = net.update_bounds(
        d, fix_interm_bounds=fix_interm_bounds,
        stop_criterion_func=stop_func(d['thresholds']),
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=branching_points is not None)
    stats.timer.add('solve')

    if solver_args['beta-crown']['all_node_split_LP']:
        # FIXME build_history_and_set_bounds doesn't return correct split
        # (just dummy elements) when split_depth > 1
        ret_lp = batch_verification_all_node_split_LP(net, d, ret, split)
        if ret_lp is not None:
            # lp_status == "unsafe"
            # unsafe cases still needed to be handled! set to be unknown for now!
            stats.all_node_split = True
            return ret_lp, torch.inf

    if set_init_alpha:
        print('Setting the initial alpha')
        ret['alphas'] = prune_alphas(ret['alphas'], net.alpha_start_nodes)
        # We just want the data structure here, not the values
        domains.init_alpha = {
            k: {kk: vv[:,:,:1].detach().clone().to(net.x.device).to(
                torch.get_default_dtype()) for kk, vv in v.items()}
            for k, v in ret['alphas'].items()
        }
    else:
        if not fix_interm_bounds:
            ret['alphas'] = prune_alphas(ret['alphas'], net.alpha_start_nodes)

    stats.timer.start('add')
    old_d_len = len(domains)
    # If intermediate layers are not refined or updated,
    # we do not need to check infeasibility when adding new domains.
    domains.add(ret, d, check_infeasibility=not fix_interm_bounds)
    stats.visited += len(domains) - old_d_len
    domains.print()
    stats.timer.add('add')

    return ret


def act_split_round(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']

    stats.timer.start('pickout')
    d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    if vanilla_crown:
        d['history'] = None
    stats.timer.add('pickout')

    # when cplex cut is enabled, for domains with general_beta created for outdated cuts,
    # we need to rewrite it to general_beta for new cuts
    if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
        cplex_update_general_beta(net, d)

    if d['mask'] is not None:
        split_domain(net, domains, d, batch, impl_params=impl_params,
                     stats=stats, fix_interm_bounds=not recompute_interm,
                     branching_heuristic=branching_heuristic, iter_idx=iter_idx)
        print('Length of domains:', len(domains))
        stats.timer.print()

    if len(domains) == 0:
        print('No domains left, verification finished!')

    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()
    global_lb = check_worst_domain(domains)
    rhs_offset = arguments.Config['specification']['rhs_offset']
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f'Current (lb-rhs): {global_lb}')
    else:
        print(f'Current (lb-rhs): {global_lb.max().item()}')
    print(f'{stats.visited} domains visited')

    return global_lb


def general_bab(net, domain, x, refined_lower_bounds=None,
            refined_upper_bounds=None, activation_opt_params=None,
            reference_alphas=None, reference_lA=None, attack_images=None,
            timeout=None, max_iterations=None, refined_betas=None, rhs=0,
            model_incomplete=None, time_stamp=0):
    # the crown_lower/upper_bounds are present for initializing the unstable
    # indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN
    # process again which is slightly slower
    start_time = time.time()
    stats = Stats()

    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    timeout = timeout or bab_args['timeout']
    max_domains = bab_args['max_domains']
    batch = solver_args['batch_size']
    use_bab_attack = bab_args['attack']['enabled']
    cut_enabled = bab_args['cut']['enabled']
    branching_input_and_activation = branch_args['branching_input_and_activation']
    max_iterations = max_iterations or bab_args['max_iterations']

    input_relu_splitter = (InputReluSplitter() if branching_input_and_activation
                           else None)

    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)
    stop_criterion = stop_criterion_batch_any(rhs)

    if refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        global_lb, ret = net.build(
            domain, x, stop_criterion_func=stop_criterion, decision_thresh=rhs)
        updated_mask, lA, alpha = (ret['mask'], ret['lA'], ret['alphas'])
        global_ub = global_lb + torch.inf
    else:
        ret = net.build_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds,
            activation_opt_params, reference_lA=reference_lA,
            reference_alphas=reference_alphas, stop_criterion_func=stop_criterion,
            cutter=net.cutter, refined_betas=refined_betas, decision_thresh=rhs)
        (global_ub, global_lb, updated_mask, lA, alpha) = (
            ret['global_ub'], ret['global_lb'], ret['mask'], ret['lA'],
            ret['alphas'])
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()

    # Transfer A_saved to the new LiRPANet
    if hasattr(model_incomplete, 'A_saved'):
        net.A_saved = model_incomplete.A_saved

    cut_enabled = arguments.Config['bab']['cut']['enabled']
    if cut_enabled:
        net.set_cuts(model_incomplete.A_saved, x, ret['lower_bounds'], ret['upper_bounds'])
    impl_params = get_impl_params(net, model_incomplete, time_stamp)

    if solver_args['beta-crown']['all_node_split_LP']:
        timeout = bab_args['timeout']
        net.build_solver_model(timeout, model_type='lp')
    # tell the AutoLiRPA class not to transfer intermediate bounds to save time
    net.interm_transfer = bab_args['interm_transfer']

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP", "MIP']:
        return all_label_global_lb, 0, 'unknown'

    if stop_criterion(global_lb).all():
        return all_label_global_lb, 0, 'safe'

    # If we are not optimizing intermediate layer bounds, we do not need to
    # save all the intermediate alpha.
    # We only keep the alpha for the last layer.
    if not solver_args['beta-crown']['enable_opt_interm_bounds']:
        # new_alpha shape:
        # [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}]
        # for each sample in batch]
        alpha = prune_alphas(alpha, net.alpha_start_nodes)

    DomainClass = SortedReLUDomainList if use_bab_attack else BatchedDomainList
    # This is the first (initial) domain.
    domains = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net,
        x=x, branching_input_and_activation=branching_input_and_activation)
    num_domains = len(domains)

    # after domains are added, we replace global_lb, global_ub with the multiple
    # targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask)
    net.tot_ambi_nodes = tot_ambi_nodes

    if cut_enabled and impl_params is None:
        cut_verification(net, domains)

    if use_bab_attack:
        return bab_loop_attack(
            domains, net, batch, rhs, start_time, timeout,
            updated_mask, attack_images, all_label_global_ub)

    branching_heuristic = get_branching_heuristic(net)

    total_round = 0
    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        total_round += 1
        global_lb = None
        print(f'BaB round {total_round}')

        if input_relu_splitter and input_relu_splitter.split_condition(
                total_round-1):
            global_lb = input_split_on_relu_domains(
                domains, wrapped_net=net, batch_size=batch)
        else:
            if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
                fetch_cut_from_cplex(net)
            global_lb = act_split_round(
                domains, net, batch, iter_idx=total_round,
                impl_params=impl_params, stats=stats,
                branching_heuristic=branching_heuristic)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()

        num_domains = len(domains)

        result = None
        if stats.all_node_split:
            stats.all_node_split = False
            result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif time.time() - start_time > timeout:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')

    if not result:
        # No domains left and not timed out.
        result = 'safe'

    del domains
    clean_net_mps_process(net)

    return global_lb, stats.visited, result
