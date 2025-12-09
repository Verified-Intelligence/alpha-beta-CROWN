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
'''Branch and bound for activation space split.'''
import time
import numpy as np
import torch
import copy

from branching_domains import BatchedDomainList, ShallowFirstBatchedDomainList, check_worst_domain
from beta_CROWN_solver import LiRPANet
from auto_LiRPA.utils import (stop_criterion_batch_any, multi_spec_keep_func_all,
                              AutoBatchSize)
from auto_LiRPA.bound_ops import (BoundInput)
from attack.domains import SortedReLUDomainList
from attack.bab_attack import bab_loop_attack
from heuristics import get_branching_heuristic
from input_split.input_split_on_relu_domains import input_split_on_relu_domains, InputReluSplitter
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_verification import cut_verification
from cuts.cut_utils import fetch_cut_from_cplex, cplex_update_general_beta
from cuts.infered_cuts import BICCOS
from utils import (print_splitting_decisions, Stats, get_unstable_neurons,
                   check_auto_enlarge_batch_size)
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

def split_domain(net: LiRPANet, domains, d, batch, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    biccos_args = bab_args['cut']['biccos']
    biccos_enable = biccos_args['enabled']
    biccos_heuristic = biccos_args['heuristic']
    enable_clip_domains = (bab_args['clip_n_verify']['clip_interm_domain']['enabled']
        and not (isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs))
    stop_func = stop_criterion_batch_any

    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    batch = next(iter(d['lower_bounds'].values())).shape[0]
    print('batch:', batch)

    stats.timer.start('decision')
    # Only for Multi-Tree-Search, we need to calculate the depth of the BFS.
    # Check if 'domains' is an instance of ShallowFirstBatchedDomainList and that we are using a BFS
    if isinstance(domains, ShallowFirstBatchedDomainList) and domains.use_bfs:
        target_batch_size = biccos_args['multi_tree_branching']['target_batch_size']
        keep_n_best_domains = biccos_args['multi_tree_branching']['keep_n_best_domains']
        # Ensure that the target batch size is at least as large as
        # the number of domains we wish to keep.
        # This is a sanity check to prevent a configuration error.
        assert target_batch_size >= keep_n_best_domains
        # This prevents a division by zero error in the next step.
        assert batch >= 1
        # The 'depth' represent how many levels will be used in the MTS process.
        depth = target_batch_size // batch
        # Ensure that the computed depth is a positive integer.
        # This assertion guarantees that there will be at least one level of processing.
        assert depth > 0
    else:
        depth = 1
    # Calculate the split depth.
    split_depth = get_split_depth(batch, min_batch_size, depth)
    # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
    branching_decision, branching_points, split_depth = (
        branching_heuristic.get_branching_decisions(
            d, split_depth, method=branch_args['method'],
            branching_candidates=max(branch_args['candidates'], split_depth),
            branching_reduceop=branch_args['reduceop']))
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
        net.build_history_and_set_bounds(d, split, mode='breadth')
    else:
        net.build_history_and_set_bounds(d, split, mode='depth')
    batch = len(split['decision'])
    stats.visited += batch
    stats.timer.add('set_bounds')
    stats.timer.start('solve')
    # Caution: we use 'all' predicate to keep the domain when multiple specs
    # are present: all lbs should be <= threshold, otherwise pruned
    # maybe other 'keeping' criterion needs to be passed here
    if enable_clip_domains and net.domain_clipper is not None:
        net.domain_clipper.get_stop_criterion_and_iter(stop_func, iter_idx)
    ret = net.update_bounds(
        d, fix_interm_bounds=fix_interm_bounds,
        stop_criterion_func=stop_func(d['thresholds']),
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=branching_points is not None,
        enable_clip_domains=enable_clip_domains,
    )
    stats.timer.add('solve')

    if (solver_args['beta-crown']['all_node_split_LP']
            and torch.any(torch.tensor(d['depths']) == net.tot_ambi_nodes)):
        if not hasattr(net, 'solver_model_initialized') or not net.solver_model_initialized:
            initialize_lp_solver_for_bab(net)
            net.solver_model_initialized = True
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
            k: {kk: vv[:, :, :1].detach().clone().to(net.device).to(
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
                            iter_idx=iter_idx,
                            domain_visited=stats.visited)

    stats.timer.start('add')
    domains.add(ret, d, check_infeasibility=not fix_interm_bounds)
    domains.print()
    stats.timer.add('add')
    del d
    return ret


def act_split_round(domains, net: LiRPANet, batch, iter_idx, stats=None,
                    branching_heuristic=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

    stats.timer.start('pickout')
    d = domains.pick_out(batch=batch, device=net.device)
    if vanilla_crown:
        d['history'] = None
    stats.timer.add('pickout')

    # when cplex cut is enabled, for domains with general_beta created for outdated cuts,
    # we need to rewrite it to general_beta for new cuts
    if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
        cplex_update_general_beta(net, d)

    if d['mask'] is not None:
        split_domain(net, domains, d, batch, stats=stats, fix_interm_bounds=not recompute_interm,
                     branching_heuristic=branching_heuristic, iter_idx=iter_idx)

        print('Length of domains:', len(domains))

    if len(domains) == 0:
        print('No domains left, verification finished!')

    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        stats.timer.start('sort')
        domains.sort()
        stats.timer.add('sort')
    global_lb = check_worst_domain(domains)
    rhs_offset = spec_args['rhs_offset']
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f'Current (lb-rhs): {global_lb}')
    else:
        print(f'Current (lb-rhs): {global_lb.max().item()}')
    print(f'{stats.visited} domains visited')

    stats.timer.print()
    return global_lb

def multi_tree_bab(net: LiRPANet, domains, batch,
    stop_criterion_func, biccos_args,
    stats, start_time, initial_bs_ratio):
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
        stop_criterion_func: callable
        biccos_args: dict
        stats: Stats
        start_time: float
    '''
    shallowbranching_heuristic = get_branching_heuristic(net, 'kfsb')
    assert len(domains) == 1

    # At the end of the multi-tree search, we have to restore the initial domain
    initial_domain = domains.pick_out(batch=batch, device=net.device)
    initial_ret = net.update_bounds(
        initial_domain,
        fix_interm_bounds=True,
        stop_criterion_func=stop_criterion_func,
        multi_spec_keep_func=multi_spec_keep_func_all,
        beta_bias=False,
        enable_clip_domains=False
    )
    domains.add(initial_ret, initial_domain, check_infeasibility=False)

    total_round = 0
    max_iter_shallow = biccos_args['multi_tree_branching']['iterations']
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
    while (total_round < max_iter_shallow
           and (num_domains > 0
                or (len(domains.mtb_backup) > 0
                    and domains.mtb_backup[0]['skip'] < 10))):
        # Increment the total number of iterations/rounds processed.
        total_round += 1

        # If there are no active domains left, try to restore one from the backup list.
        if num_domains == 0:
            # Ensure that there is at least one backup available.
            assert len(domains.mtb_backup) > 0

            # If the most recent backup entry has already been skipped 3 times
            # and there is more than one backup available, then remove it.
            # This prevents repeatedly attempting a backup that's already failed multiple times.
            if domains.mtb_backup[-1]['skip'] == 3 and len(domains.mtb_backup) > 1:
                del domains.mtb_backup[-1]

            # Increase the skip count of the current (last) backup entry.
            domains.mtb_backup[-1]['skip'] += 1

            # Output the current state: how many backup entries remain and the current skip count.
            print('Going back, stack has', len(domains.mtb_backup),
                'entries left. Skipping', domains.mtb_backup[-1]['skip'])

            try:
                # Attempt to restore a domain by adding the best candidate(s)
                # using a deep copy of the most recent backup entry.
                domains.add_best_k_lower_bounds(**copy.deepcopy(domains.mtb_backup[-1]))
            except ShallowFirstBatchedDomainList.EmptyKLower:
                # If there is nothing left to restore from this backup entry,
                # mark it as fully skipped by setting its skip count to 3.
                # This will force its removal in subsequent iterations.
                domains.mtb_backup[-1]['skip'] = 3
                # Skip the remainder of the current iteration and proceed with the next one.
                continue

        print(f'Shallow-BaB round {total_round}')
        global_lb = act_split_round(domains, net, batch, iter_idx=total_round,
                stats=stats, branching_heuristic=shallowbranching_heuristic)
        num_domains = len(domains)
        if num_domains == 0:
            print('No domains left, MTS early stop!')
            break
        print(f'Cumulative time: {time.time() - start_time}\n')

    # Drop current list of domains
    domains.use_bfs = False
    if len(domains) > 0:
        domains.pick_out(batch=len(domains), device=net.device)

    if not biccos_args['multi_tree_branching']['restore_best_tree']:
        domains.add(initial_ret, initial_domain, check_infeasibility=False)
    else:
        print('Restoring the best tree')
        domains.restore_best_domains(initial_ret, initial_domain)
        # We might have added some domains that are UNSAT
        print('Shallow branching resets to n domains: ', len(domains))
        base_d = domains.pick_out(batch=len(domains), device=net.device)
        new_ret = net.update_bounds(
                base_d,
                fix_interm_bounds=True,
                stop_criterion_func=stop_criterion_func,
                multi_spec_keep_func=multi_spec_keep_func_all,
                beta_bias=False,
                enable_clip_domains=False
            )
        domains.add(new_ret, base_d, check_infeasibility=False)
        print('After pruning, left: ', len(domains))
    if not biccos_args['constraint_strengthening']:
        arguments.Config['solver']['min_batch_size_ratio'] = initial_bs_ratio
    print('\n   Back to Regular BaB\n')
    return global_lb


def initialize_lp_solver_for_bab(net: LiRPANet):
    # Initialize the LP solver model and pre-store the names of the layers
    timeout = arguments.Config['bab']['timeout']
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


def general_bab(net: LiRPANet, x, c, rhs,
                reference_dict=None,
                timeout=None, max_iterations=None):
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
    enable_clip_domains = (bab_args['clip_n_verify']['clip_input_domain']['enabled']
        or bab_args['clip_n_verify']['clip_interm_domain']['enabled'])
    max_iterations = max_iterations or bab_args['max_iterations']
    MTS_enabled = (
        biccos_args['enabled'] and biccos_args['multi_tree_branching']['enabled'])

    input_relu_splitter = (InputReluSplitter() if
                branch_args['branching_input_and_activation'] else None)

    stop_criterion = stop_criterion_batch_any
    stop_criterion_func = stop_criterion(rhs)

    if reference_dict is None:
        reference_dict = {}
    refined_lower_bounds = reference_dict.get('lower_bounds', None)
    refined_upper_bounds = reference_dict.get('upper_bounds', None)
    reference_lA = reference_dict.get('lA', None)
    reference_alphas = reference_dict.get('alphas', None)
    refined_betas = reference_dict.get('refined_betas', None)
    attack_examples = reference_dict.get('attack_examples', None)

    # Since we always enable incomplete verification by default,
    # it always has refined bounds.
    if refined_lower_bounds is None or refined_upper_bounds is None:
        # FIXME: This branch should not be used by default and is only for backup.
        # Maybe it can be removed in the future.
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        _, ret = net.build(x, c, rhs, stop_criterion)
    else:
        ret = net.build_with_refined_bounds(
            x, c, rhs, stop_criterion,
            refined_lower_bounds, refined_upper_bounds,
            reference_lA, reference_alphas,
            refined_betas)

    (global_ub, global_lb, updated_mask, lA, alpha) = (
        ret['global_ub'], ret['global_lb'], ret['mask'], ret['lA'],
        ret['alphas'])

    if cut_enabled:
        # Always reset the cut module if it exists
        net.net.cut_timestamp = -1
        net.net.cut_module = None
        # All intermediate bounds are set during the incomplete verification phase.
        # We only need to set the final layer bounds here.
        net.net[net.net.final_name].lower = global_lb
        net.net[net.net.final_name].upper = global_ub
        net.set_cuts()
        if biccos_args['enabled']:
            net.biccos = BICCOS(ret, rhs, net.final_name)
            initial_bs_ratio, MTS_enabled = net.biccos.set_auto_params()

    net.interm_transfer = bab_args['interm_transfer']

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP', 'MIP']:
        return all_label_global_lb, 0, 'unknown'

    if stop_criterion_func(global_lb).all():
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
    elif MTS_enabled:
        DomainClass = ShallowFirstBatchedDomainList
    else:
        DomainClass = BatchedDomainList
    # This is the first (initial) domain.
    domains = DomainClass(
        ret, c, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'],
    )
    num_domains = len(domains)

    # after domains are added, we replace global_lb, global_ub with the multile
    # targets 'real' global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask, net)
    net.tot_ambi_nodes = tot_ambi_nodes
    domains.update_unstable_mask(updated_mask)
    net.unstable_mask = domains.unstable_mask
    if enable_clip_domains:
        net.domain_clipper.update_unstable_idx(updated_mask, net)

    if cut_enabled:
        cut_verification(net, domains)

    if bab_args['attack']['enabled']:
        return bab_loop_attack(
            domains, net, batch, rhs, start_time, timeout,
            updated_mask, attack_examples, all_label_global_ub)

    branching_heuristic = get_branching_heuristic(net)

    # If we are using shallow branching, we need to do the multi-tree search
    # as the pre-solve part for BICCOS.
    if isinstance(domains, ShallowFirstBatchedDomainList):
        global_lb = multi_tree_bab(
            net, domains, batch, stop_criterion_func, biccos_args,
            stats, start_time, initial_bs_ratio)  # pylint: disable=used-before-assignment

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
        print(f'BaB round {total_round}')

        if (cut_enabled and biccos_args['enabled']
            and total_round - 1 == net.biccos.max_iter
            and biccos_args['heuristic'] == 'neuron_influence_score'):
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
                domains, net, batch, iter_idx=total_round, stats=stats,
                branching_heuristic=branching_heuristic)
        batch = check_auto_enlarge_batch_size(auto_batch_size)

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

    return global_lb.max(), stats.visited, result, stats
