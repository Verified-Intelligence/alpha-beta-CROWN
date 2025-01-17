#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
'''Branch and bound for activation space split.'''
import time
import numpy as np
import torch
import copy

from branching_domains import BatchedDomainList, ShallowFirstBatchedDomainList, check_worst_domain
from auto_LiRPA.utils import (stop_criterion_batch_any, multi_spec_keep_func_all,
                              AutoBatchSize)
from auto_LiRPA.bound_ops import (BoundInput)
from attack.domains import SortedReLUDomainList
from attack.bab_attack import bab_loop_attack
from heuristics import get_branching_heuristic
from input_split.input_split_on_relu_domains import input_split_on_relu_domains, InputReluSplitter
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_verification import cut_verification, get_impl_params
from cuts.cut_utils import fetch_cut_from_cplex, clean_net_mps_process, cplex_update_general_beta
from cuts.infered_cuts import BICCOS
from utils import (print_splitting_decisions, print_average_branching_neurons,
                   Stats, get_unstable_neurons, check_auto_enlarge_batch_size)
from prune import prune_alphas
import arguments


def get_split_depth(batch_size, min_batch_size, min_depth):
    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    if batch_size < min_batch_size:
        # Split multiple levels, to obtain at least min_batch_size domains in this batch.
        return max(min_depth, int(
            np.log(min_batch_size / max(min_depth, batch_size)) / np.log(2)))
    else:
        return min_depth

def split_domain(net, domains, d, batch, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    biccos_args = bab_args['cut']['biccos']
    biccos_enable = biccos_args['enabled']
    biccos_heuristic = biccos_args['heuristic']
    stop_func = stop_criterion_batch_any

    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    batch = next(iter(d['lower_bounds'].values())).shape[0]
    print('batch:', batch)

    stats.timer.start('decision')
    if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
        depth = biccos_args['multi_tree_branching']['k_splits']
    else:
        depth = 1
    split_depth = get_split_depth(batch, min_batch_size, depth)
    # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
    branching_decision, branching_points, split_depth = (
        branching_heuristic.get_branching_decisions(
            d, split_depth, method=branch_args['method'],
            branching_candidates=max(branch_args['candidates'], split_depth),
            branching_reduceop=branch_args['reduceop']))
    print_average_branching_neurons(
        branching_decision, stats.implied_cuts, impl_params=impl_params)
    if len(branching_decision) < len(next(iter(d['mask'].values()))):
        print('all nodes are split!!')
        print(f'{stats.visited} domains visited')
        stats.all_node_split = True
        stats.all_split_result = 'unknown'
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
    if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
        net.build_history_and_set_bounds(d, split, impl_params=impl_params, mode='breadth')
    else:
        net.build_history_and_set_bounds(d, split, impl_params=impl_params, mode='depth')
    stats.timer.add('set_bounds')
    batch = len(split['decision'])
    stats.timer.start('solve')
    # Caution: we use 'all' predicate to keep the domain when multiple specs
    # are present: all lbs should be <= threshold, otherwise pruned
    # maybe other 'keeping' criterion needs to be passed here
    ret = net.update_bounds(
        d, fix_interm_bounds=fix_interm_bounds,
        stop_criterion_func=stop_func(d['thresholds']),
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=branching_points is not None)
    stats.timer.add('solve')

    if (solver_args['beta-crown']['all_node_split_LP']
            and torch.any(torch.tensor(d['depths']) == net.tot_ambi_nodes)):
        # FIXME build_history_and_set_bounds doesn't return correct split
        # (just dummy elements) when split_depth > 1
        stats.all_split_result = 'unknown'
        if batch_verification_all_node_split_LP(net, d, ret, split, stats):
            stats.all_node_split = True
            stats.all_split_result = 'unsafe'
            return torch.inf

    if set_init_alpha:
        print('Setting the initial alpha')
        ret['alphas'] = prune_alphas(ret['alphas'], net.alpha_start_nodes)
        # We just want the data structure here, not the values
        domains.init_alpha = {
            k: {kk: vv[:, :, :1].detach().clone().to(net.x.device).to(
                torch.get_default_dtype()) for kk, vv in v.items()}
            for k, v in ret['alphas'].items()
        }
    else:
        if not fix_interm_bounds:
            ret['alphas'] = prune_alphas(ret['alphas'], net.alpha_start_nodes)

    # We have to add cuts now, because domains.add might modify the list of domains in ret
    if ret and bab_args['cut']['enabled'] and biccos_enable:
        # We only enforce cut usage for multi-tree-searching
        enforce_cut_usage = (
        isinstance(domains, ShallowFirstBatchedDomainList)
        and domains.use_bfs)
        # If disable_constraint_strengthening, set iter_idx to a very large value
        # to skip inference proceture
        iter_idx = iter_idx if biccos_args['constraint_strengthening'] else float('inf')
        net.biccos.update_cut(d, net, ret,
                            enforce_usage=enforce_cut_usage,
                            heuristic=biccos_heuristic,
                            iter_idx=iter_idx)

    stats.timer.start('add')
    old_d_len = len(domains)
    domains.add(ret, d, check_infeasibility=not fix_interm_bounds)
    stats.visited += len(domains) - old_d_len
    domains.print()
    stats.timer.add('add')
    del d
    return ret


def act_split_round(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

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
    rhs_offset = spec_args['rhs_offset']
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f'Current (lb-rhs): {global_lb}')
    else:
        print(f'Current (lb-rhs): {global_lb.max().item()}')
    print(f'{stats.visited} domains visited')


    return global_lb

def multi_tree_bab(net, domains, batch,
    stop_criterion, biccos_args, impl_params,
    stats, start_time):
    '''
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
        stop_criterion: callable
        biccos_args: dict
        impl_params: dict
        stats: Stats
        start_time: float
    '''
    shallowbranching_heuristic = get_branching_heuristic(net, 'kfsb')
    assert len(domains) == 1

    # At the end of the multi-tree search, we have to restore the initial domain
    initial_domain = domains.pick_out(batch=batch, device=net.x.device)
    initial_ret = net.update_bounds(
        initial_domain,
        fix_interm_bounds=True,
        stop_criterion_func=stop_criterion,
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=False
    )
    domains.add(initial_ret, initial_domain, check_infeasibility=False)

    total_round = 0
    max_iter_shallow = biccos_args['multi_tree_branching']['iterations']
    num_domains = len(domains)
    # In rare cases, adding the initial domain back might prove it to be UNSAT.
    # This might happen due to randomnes in the gradient updates.
    # If it happens, we're done and don't need to proceed with regular BaB.
    if num_domains == 0:
        return
    assert num_domains == 1

    while (num_domains > 0 and total_round < max_iter_shallow):
        total_round += 1
        print(f'Shallow-BaB round {total_round}')
        act_split_round(domains, net, batch, iter_idx=total_round,
                impl_params=impl_params, stats=stats,
                branching_heuristic=shallowbranching_heuristic)
        num_domains = len(domains)
        print(f'Cumulative time: {time.time() - start_time}\n')

    # Drop current list of domains
    domains.use_bfs = False
    if len(domains) > 0:
        domains.pick_out(batch=len(domains), device=net.x.device)

    if not biccos_args['multi_tree_branching']['restore_best_tree']:
        domains.add(initial_ret, initial_domain, check_infeasibility=False)
    else:
        domains.restore_best_domains(initial_ret, initial_domain)
        # We might have added some domains that are UNSAT
        print('Shallow branching resets to n domains: ', len(domains))
        base_d = domains.pick_out(batch=len(domains), device=net.x.device)
        new_ret = net.update_bounds(
                base_d,
                fix_interm_bounds=True,
                stop_criterion_func=stop_criterion,
                multi_spec_keep_func=multi_spec_keep_func_all,
                beta_bias=False
            )
        domains.add(new_ret, base_d, check_infeasibility=False)
        print('After pruning, left: ', len(domains))

    print('\nBack to Regular BaB\n')

def general_bab(net, domain, x, refined_lower_bounds=None,
                refined_upper_bounds=None, activation_opt_params=None,
                reference_alphas=None, reference_lA=None, attack_images=None,
                timeout=None, max_iterations=None, refined_betas=None, rhs=0,
                model_incomplete=None, time_stamp=0, property_idx=None):
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
    cut_enabled = bab_args['cut']['enabled']
    biccos_args = bab_args['cut']['biccos']
    multi_tree_branching_enabled = (cut_enabled and biccos_args['enabled'] and
                                    biccos_args['multi_tree_branching']['enabled'])
    max_iterations = max_iterations or bab_args['max_iterations']

    input_relu_splitter = (InputReluSplitter() if
                branch_args['branching_input_and_activation'] else None)

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

    if cut_enabled:
        net.set_cuts(model_incomplete.A_saved, x, ret['lower_bounds'], ret['upper_bounds'])
        if biccos_args['enabled']:
            print('Inferred cuts enabled')
            print('Warning: The mininal batch size ratio is set to 0')
            initial_bs_ratio = arguments.Config['solver']['min_batch_size_ratio']
            arguments.Config['solver']['min_batch_size_ratio'] = 0
            net.biccos = BICCOS(ret, property_idx, rhs)

    impl_params = get_impl_params(net, model_incomplete, time_stamp)

    if solver_args['beta-crown']['all_node_split_LP']:
        # Initialize the LP solver model and pre-store the names of the layers
        timeout = bab_args['timeout']
        net.build_solver_model(timeout, model_type='lp')
        net.pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in net.net.relus]
        net.relu_layer_names = [relu_layer.name for relu_layer in net.net.relus]
        input_name = [name for name in net.net.input_name if type(net.net[name]) == BoundInput]
        assert len(input_name) == 1, 'there should be only 1 BoundInput!'
        input_name = input_name[0]
        def extract_var_names(solver_vars):
            if isinstance(solver_vars, list):
                return [extract_var_names(sub_solver_vars) for sub_solver_vars in solver_vars]
            else:
                return solver_vars.VarName
        net.input_name = extract_var_names(net.net[input_name].solver_vars)
    # tell the AutoLiRPA class not to transfer intermediate bounds to save time
    net.interm_transfer = bab_args['interm_transfer']
    if not bab_args['interm_transfer']:
        # Branching domains cannot support
        # bab_args['interm_transfer'] == False and bab_args['sort_domain_interval'] > 0
        # at the same time.
        assert bab_args['sort_domain_interval'] == -1

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP', 'MIP']:
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

    if bab_args['attack']['enabled']:
        DomainClass = SortedReLUDomainList
    elif multi_tree_branching_enabled:
        DomainClass = ShallowFirstBatchedDomainList
    else:
        DomainClass = BatchedDomainList
    # This is the first (initial) domain.
    domains = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    num_domains = len(domains)

    # after domains are added, we replace global_lb, global_ub with the multile
    # targets 'real' global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask, net)
    net.tot_ambi_nodes = tot_ambi_nodes

    if cut_enabled and impl_params is None:
        cut_verification(net, domains)

    if bab_args['attack']['enabled']:
        return bab_loop_attack(
            domains, net, batch, rhs, start_time, timeout,
            updated_mask, attack_images, all_label_global_ub)

    branching_heuristic = get_branching_heuristic(net)

    # If we are using shallow branching, we need to do the multi-tree search
    # as the pre-solve part for BICCOS.
    if isinstance(domains, ShallowFirstBatchedDomainList):
        multi_tree_bab(
            net, domains, batch, stop_criterion, biccos_args,
            impl_params, stats, start_time)

    num_domains = len(domains)
    vram_ratio = 0.85 if cut_enabled else 0.9
    auto_batch_size = AutoBatchSize(
        batch, net.device, vram_ratio,
        enable=arguments.Config['solver']['auto_enlarge_batch_size'])

    total_round = 0
    result = None
    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        total_round += 1
        global_lb = None
        print(f'BaB round {total_round}')

        if (cut_enabled and biccos_args['enabled']
            and total_round - 1 == net.biccos.max_iter):
            print('Cut Inference reaches max iterations. Recover the setting')
            arguments.Config['solver']['min_batch_size_ratio'] = initial_bs_ratio  # pylint: disable=used-before-assignment

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
        if input_relu_splitter and input_relu_splitter.split_condition(
                total_round-1):
            global_lb = input_split_on_relu_domains(
                domains, wrapped_net=net, batch_size=batch)
        else:
            if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                and not biccos_args['enabled']):
                fetch_cut_from_cplex(net)
            global_lb = act_split_round(
                domains, net, batch, iter_idx=total_round,
                impl_params=impl_params, stats=stats,
                branching_heuristic=branching_heuristic)
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()

        num_domains = len(domains)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
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

    return global_lb, stats.visited, result, stats
