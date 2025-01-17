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
"""Branch and bound based adversarial attacks."""

import copy
import time
from collections import defaultdict, Counter
import numpy as np
import torch
import arguments
from sortedcontainers import SortedList
from attack.adv_domains import AdvExamplePool
from attack.attack_pgd import attack_with_general_specs
from attack.domains import ReLUDomain
from heuristics import KfsbBranching
from utils import check_infeasible_bounds, Stats
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor
from cuts.cut_utils import clean_net_mps_process


diving_Visited = 0  # FIXME (10/21): Do not use global variable here.


def bab_loop_attack(domains, net, batch, rhs, start_time, timeout,
                    updated_mask, attack_images, all_label_global_ub):
    bab_args = arguments.Config['bab']
    solver_args = arguments.Config['solver']
    get_upper_bound = bab_args['get_upper_bound']
    max_domains = bab_args['max_domains']
    stats = Stats()

    # Beam search based BaB enabled. We need to construct the MIP model.
    print('Building MIP for beam search...')
    net.build_solver_model(
        timeout=bab_args['attack']['mip_timeout'],
        mip_multi_proc=solver_args['mip']['parallel_solvers'],
        mip_threads=solver_args['mip']['solver_threads'], model_type='mip')

    # BaB-attack code still uses a legacy sorted domain list.
    domains = to_sorted_list(domains)
    adv_pool = init_bab_attack(net, updated_mask, attack_images)
    global_ub = min(all_label_global_ub, adv_pool.adv_pool[0].obj)

    total_round = 0
    while len(domains):
        total_round += 1
        global_lb = None
        print(f'BaB round {total_round}')

        global_lb, batch_ub, domains = bab_attack(
            domains, net, batch, adv_pool=adv_pool)

        if get_upper_bound:
            print(f'Global ub: {global_ub}, batch ub: {batch_ub}')
        global_ub = min(global_ub, batch_ub)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        if isinstance(global_ub, torch.Tensor):
            global_ub = global_ub.min().item()

        result = None
        if stats.all_node_split:
            stats.all_node_split = False
            result = 'unknown'
        elif len(domains) > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif global_ub < rhs:
            print('Attack success during branch and bound.')
            result = 'unsafe'
        elif time.time() - start_time > timeout:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')

    if not result:
        # No domains left and no ub < 0 found.
        result = 'unknown'

    if beam_mip_attack.started:
        print('Terminating MIP processes...')
        net.pool_termination_flag.value = 1
    del domains
    clean_net_mps_process(net)

    return global_lb, stats.visited, result


def history_to_splits(history):
    splits = []
    coeffs = []
    for layer_idx, layer_history in enumerate(history):
        for s, c in zip(*layer_history):
            splits.append([layer_idx, s])
            coeffs.append(c)
    return splits, coeffs


@torch.no_grad()
def find_promising_domains(adv_pool, dive_domains, candidates, start_iter, max_dive_fix, min_local_free):
    find_promising_domains.counter += 1
    all_splits = []
    all_coeffs = []
    all_advs = []
    all_act_patterns = []
    # Skip the earliest a few iterations.
    if find_promising_domains.counter < start_iter:
        print(f'Current iteration {find_promising_domains.counter}, MIP will start at iteration {start_iter}')
        return all_splits, all_coeffs, all_advs, all_act_patterns
    n_domains = candidates - 1
    if find_promising_domains.current_method == 'top-down':
        # automatically adjust top down max_dive_fix_ratio
        max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
        if max_dive_fix_ratio > 0 and find_promising_domains.topdown_status == "infeas":
            # still run top down for next sub-mip round because the previous one has too large dive and inf sub-mips
            find_promising_domains.current_method = "bottom-up"
            print(f"orig max_dive_fix_ratio: {max_dive_fix_ratio}, fix: {max_dive_fix}")
            new_max_dive_fix_ratio = max(max_dive_fix_ratio - 0.2, 0)
            max_dive_fix = int(max_dive_fix / max_dive_fix_ratio * new_max_dive_fix_ratio)
            arguments.Config["bab"]["attack"]["max_dive_fix_ratio"] = new_max_dive_fix_ratio
            print(f"### topdown most inf! reduce max_dive_fix_ratio to {new_max_dive_fix_ratio}, now fix: {max_dive_fix}")
        if max_dive_fix_ratio < 1 and find_promising_domains.topdown_status == "timeout":
            # wait for next round topdown dive with the increased max_dive_fix_ratio
            print(f"orig max_dive_fix_ratio: {max_dive_fix_ratio}, fix: {max_dive_fix}")
            new_max_dive_fix_ratio = min(max_dive_fix_ratio + 0.2, 1.)
            arguments.Config["bab"]["attack"]["max_dive_fix_ratio"] = new_max_dive_fix_ratio
            print(f"### topdown most timeout! increase max_dive_fix_ratio to {new_max_dive_fix_ratio}")
        # set back status to normal
        find_promising_domains.topdown_status = "normal"

    if find_promising_domains.current_method == 'bottom-up':
        # automatically adjust bottom up min_local_free_ratio
        min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
        if min_local_free_ratio > 0. and find_promising_domains.bottomup_status == "timeout":
            # rerun bottomup search with the decreased min_local_free_ratio
            print(f"orig min_local_free_ratio: {min_local_free_ratio}, fix: {min_local_free}")
            new_min_local_free_ratio = max(min_local_free_ratio - 0.1, 0.01)
            min_local_free = int(min_local_free / min_local_free_ratio * new_min_local_free_ratio)
            arguments.Config["bab"]["attack"]["min_local_free_ratio"] = new_min_local_free_ratio
            print(f"### bottom-up most timeout! decrease min_local_free_ratio to {new_min_local_free_ratio}, fix: {min_local_free}")
        find_promising_domains.bottomup_status = "normal"

    if find_promising_domains.current_method == 'top-down':
        # Try adversarial example local search.
        # Common adversarial pattern. Always used.
        find_promising_domains.current_method = 'bottom-up'
        print('Bottom-Up: Constructing sub-MIPs from current adversarial examples.')
        adv_split, adv_coeff = adv_pool.get_activation_pattern_from_pool()
        all_splits = [adv_split.tolist()]
        all_coeffs = [adv_coeff.tolist()]
        all_advs = [adv_pool.adv_pool[0].x.unsqueeze(0)]
        all_act_patterns = [[p.unsqueeze(0) for p in adv_pool.adv_pool[0].activation_pattern]]
        for i in range(n_domains - 1):
            # Add an adversarial example.
            if i < len(adv_pool.adv_pool):
                uncommon_split, _ = adv_pool.get_ranked_activation_pattern(n_activations=min_local_free, find_uncommon=True, random_keep=True)
                adv_s, adv_c = adv_pool.get_activation_pattern(adv_pool.adv_pool[i], blacklist=uncommon_split)
                all_splits.append(adv_s)
                all_coeffs.append(adv_c)
                all_advs.append(adv_pool.adv_pool[i].x.unsqueeze(0))
                all_act_patterns.append([p.unsqueeze(0) for p in adv_pool.adv_pool[i].activation_pattern])
    else:
        find_promising_domains.current_method = 'top-down'
        print('Top-Down: Constructing sub-MIPs from beam search domains.')
        diving_fix = []
        # Try diving domains.
        for i in range(n_domains):
            # Add this domain for MIP solving.
            if i < len(dive_domains):
                s, c = history_to_splits(dive_domains[i].history)
                diving_fix.append(max(max_dive_fix - len(s), 0))
                # Add adv diving.
                if max_dive_fix > len(s):
                    adv_s, adv_c = adv_pool.get_ranked_activation_pattern(n_activations=max_dive_fix - len(s), blacklist=s, random_keep=True)
                    s = torch.cat([torch.tensor(s), adv_s], dim=0)
                    c = torch.cat([torch.tensor(c), adv_c], dim=0)
                all_splits.append(s)
                all_coeffs.append(c)
                all_advs.append(None)
                all_act_patterns.append(None)
        print(f"{diving_fix} neurons fixed by diving.")
    print(f"Generating sub-MIPs with {[len(s) for s in all_splits]} fixed neurons.")
    return all_splits, all_coeffs, all_advs, all_act_patterns


def beam_alpha_crown(net, splits, coeffs):

    batch = len(splits)
    # reset beta to None
    for m in net.net.relus:
        m.sparse_betas = None

    # set alpha alpha
    alpha = net.refined_alpha
    spec_name = net.net.final_name
    for m in net.net.relus:
        for spec_name in alpha[m.name].keys():
            m.alpha[spec_name] = alpha[m.name][spec_name].repeat(1, 1, batch, *([1] * (alpha[m.name][spec_name].ndim - 3))).detach().requires_grad_()

    # repeat lower and upper bounds according to batch size
    lower_bounds, upper_bounds = [], []
    for refined_lower_bounds, refined_upper_bounds in zip(net.refined_lower_bounds, net.refined_upper_bounds):
        if refined_lower_bounds.ndim == 4:
            lower_bounds.append(refined_lower_bounds.repeat(batch,1,1,1).detach())
            upper_bounds.append(refined_upper_bounds.repeat(batch,1,1,1).detach())
        else:
            lower_bounds.append(refined_lower_bounds.repeat(batch,1).detach())
            upper_bounds.append(refined_upper_bounds.repeat(batch,1).detach())

    # update bounds with splits
    for bi in range(batch):
        split, coeff = splits[bi], coeffs[bi]
        for s, c in zip(split, coeff):
            # splits for each batch
            relu_layer, neuron_idx = s
            if c == 1:
                lower_bounds[relu_layer].view(batch, -1)[bi, neuron_idx] = 0
            else:
                upper_bounds[relu_layer].view(batch, -1)[bi, neuron_idx] = 0

    interm_bounds, reference_bounds = {}, {}
    for i, layer in enumerate(net.net.relus):
        nd = layer.inputs[0].name
        if i == 0:
            print("new intermediate layer bounds:", i, nd, net.refined_lower_bounds[i].shape, lower_bounds[i].shape)
            interm_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
        else:
            print("reference bounds:", i, nd, net.refined_lower_bounds[i].shape, lower_bounds[i].shape)
            reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]

    ptb = PerturbationLpNorm(norm=net.x.ptb.norm, eps=net.x.ptb.eps,
                                x_L=net.x.ptb.x_L.repeat(batch, 1, 1, 1),
                                x_U=net.x.ptb.x_U.repeat(batch, 1, 1, 1))
    new_x = BoundedTensor(net.x.data.repeat(batch, 1, 1, 1), ptb)
    c = None if net.c is None else net.c.repeat(new_x.shape[0], 1, 1)

    lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
    init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
    optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
    lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]

    net.net.set_bound_opts({'verbosity': 1})
    net.net.set_bound_opts({'optimize_bound_args': {
        'iteration': init_iteration, 'enable_beta_crown': False, 'enable_alpha_crown': True,
        'use_shared_alpha': False, 'optimizer': optimizer,
        'fix_interm_bounds': True,
        'lr_alpha': lr_init_alpha, 'init_alpha': False,
        'lr_decay': lr_decay}})
    lb, _ = net.net.compute_bounds(x=(new_x,), C=c, method='crown-optimized',
                                interm_bounds=interm_bounds,
                                reference_bounds=reference_bounds,
                                bound_upper=False, needed_A_dict=net.needed_A_dict)
    lower_bounds_new, upper_bounds_new = net.get_candidate_parallel(lb, lb + 99)

    return lower_bounds_new, upper_bounds_new


def beam_mip_attack(net, adv_pool, dive_domains, submip_start_iteration, max_dive_fix, min_local_free, finalize=False):
    def parse_results(res):
        solver_results = res.get()
        upper_bounds, lower_bounds, status, solutions = zip(*solver_results)
        solutions = torch.cat(solutions, dim=0)  # Each MIP worker may return 0 or 1 solution.
        print('Sub-MIP Method:', find_promising_domains.current_method)
        print('Got MIP ub:', [f"{ub:.5f}" for ub in upper_bounds])
        print('Got MIP lb:', [f"{lb:.5f}" for lb in lower_bounds])
        print('Got MIP status:', status)
        print('Got MIP solutions:', solutions.size())
        # collect status for submip
        mip_status = "normal"
        inf_cnt = 0
        timeout_cnt = 0
        for st in status:
            if st == 9:
                timeout_cnt += 1
            if st == 3:
                inf_cnt += 1
        if inf_cnt >= (len(upper_bounds) // 2):
            mip_status = "infeas"
        if timeout_cnt >= (len(upper_bounds) // 2):
            mip_status = "timeout"

        if find_promising_domains.current_method == "top-down":
            find_promising_domains.topdown_status = "normal"
            find_promising_domains.topdown_status = mip_status
            print(f"### topdown status {find_promising_domains.topdown_status}")
        else:
            find_promising_domains.bottomup_status = "normal"
            find_promising_domains.bottomup_status = mip_status
            print(f"### bottomup status {find_promising_domains.bottomup_status}")
        return min(upper_bounds), solutions
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    # Wait for last batch of MIP to finish.
    attack_success = False
    min_bound = float("inf")
    solutions = None
    if finalize and beam_mip_attack.started:
        print('Waiting MIP Solver to finalize...')
        return parse_results(net.pool_result)
    else:
        if net.pool_result is not None:
            # Checking if last batch of MIP has finished.
            if not net.pool_result.ready():
                print('MIP solver still running. Waiting for the next iteration.')
                return float("inf"), None
            else:
                # Get results from the last batch and run a new batch.
                min_bound, solutions = parse_results(net.pool_result)
                attack_success = min_bound < 0

        if solutions is not None and solutions.size(0) != 0:
            # Add MIP solutions to the pool.
            solutions = solutions.to(net.net.device)
            mip_pred = net.net(solutions, reset_perturbed_nodes=False).cpu()
            mip_margins = mip_pred.matmul(net.c.cpu()[0].transpose(-1, -2)).squeeze(-1)
            # Convert to margin via the C matrix.
            attack_ret, attack_images, attack_margin, _ = attack_with_general_specs(
                    model=net.model_ori,
                    x=solutions,
                    # shape [batch, num_spec, *input_shape]
                    data_min=net.x.ptb.x_L[0].expand(solutions.size(0), *[-1] * (net.x.ptb.x_L.ndim - 1)).unsqueeze(1),
                    data_max=net.x.ptb.x_U[0].expand(solutions.size(0), *[-1] * (net.x.ptb.x_U.ndim - 1)).unsqueeze(1),
                    list_target_label_arrays=solutions.size(0)*[((net.c[0].cpu(), torch.tensor([0.])),)],  # (prop_mat, rhs).
                    initialization="none")
            # attack_images has shape (batch, spec, c, h, w).
            adv_pool.add_adv_images(attack_images.squeeze(1))  # Squeeze the spec dimension, since we only have 1 spec.
            adv_pool.print_pool_status()
            print(f'mip ub: {min(mip_margins.view(-1)).item()} -> mip ub (PGD): {min(attack_margin.view(-1)).item()}, best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')

        if not attack_success:
            # Run new MIPs based on selected domains.
            splits, coeffs, advs, act_patterns = find_promising_domains(adv_pool, dive_domains, mip_multi_proc, submip_start_iteration, max_dive_fix, min_local_free)
            if len(splits) > 0:
                print('Start to run MIP!')
                refined_lower_bounds, refined_upper_bounds = None, None
                if arguments.Config["bab"]["attack"]["refined_mip_attacker"]:
                    refined_batch_size = arguments.Config["bab"]["attack"]["refined_batch_size"]
                    batch = len(splits)
                    if refined_batch_size is None or batch <= refined_batch_size:
                        refined_lower_bounds, refined_upper_bounds = beam_alpha_crown(net, splits, coeffs)
                    else:
                        refined_lower_bounds = [[] for _ in net.refined_lower_bounds]
                        refined_upper_bounds = [[] for _ in net.refined_upper_bounds]
                        start_batch, end_batch = 0, refined_batch_size
                        while start_batch < batch:
                            rlbs, rubs = beam_alpha_crown(net, splits[start_batch: end_batch], coeffs[start_batch: end_batch])
                            for relu_idx, (rlb, rub) in enumerate(zip(rlbs, rubs)):
                                refined_lower_bounds[relu_idx].append(rlb)
                                refined_upper_bounds[relu_idx].append(rub)
                            start_batch += refined_batch_size
                            end_batch += refined_batch_size

                        for relu_idx in range(len(refined_lower_bounds)):
                            refined_lower_bounds[relu_idx] = torch.cat(refined_lower_bounds[relu_idx])
                            refined_upper_bounds[relu_idx] = torch.cat(refined_upper_bounds[relu_idx])
                            assert refined_lower_bounds[relu_idx].size(0) == batch, f"refined_batch_size process wrong, {relu_idx}, {refined_lower_bounds[relu_idx].size(0)} != {batch}!"

                net.update_mip_model_fix_relu(splits, coeffs, target=None,
                        async_mip=True, best_adv=advs, adv_activation_pattern=act_patterns,
                        refined_lower_bounds=refined_lower_bounds, refined_upper_bounds=refined_upper_bounds)
                beam_mip_attack.started = True
    if min_bound > 0:
        # ObjBound was returned. This is not an upper bound.
        return float("inf"), solutions
    else:
        return min_bound, solutions


def probabilistic_select_domains(dive_domains, candidates_number):
    softmax_temperature = 0.1
    new_domains = type(dive_domains)()
    # Always Keep domains with non-zero priorities.
    removed_domains = []
    for i, d in enumerate(dive_domains):
        if d.priority > 0:
            # Shallow copy this domain.
            new_domains.add(copy.copy(d))
            # Make sure this domain will not be selected again later.
            removed_domains.append(i)
    for r in reversed(removed_domains):
        dive_domains.pop(r)
    lbs = torch.tensor([d.lower_bound for d in dive_domains])
    # Select candidates_number - domains with priority.
    remain_domains = min(len(dive_domains), candidates_number) - len(new_domains)
    if remain_domains > 0:
        # probs = -lbs / lbs.abs().sum()
        normalized_lbs = -lbs / lbs.neg().max()
        probs = torch.nn.functional.softmax(normalized_lbs / softmax_temperature, dim=0)
        # Choose domains based on sampling probability.
        selected_indices = probs.multinomial(remain_domains, replacement=False)
        for i in selected_indices:
            new_domains.add(dive_domains[i])
        print(f'Probabilistic domain selection: probability are {probs[len(probs)//100]}@0.01, '
              f'{probs[len(probs)//20]}@0.05, {probs[len(probs)//10]}@0.1, {probs[len(probs)//5]}@0.2, {probs[len(probs)//2]}@0.5')
    del dive_domains
    return new_domains


def bab_attack(dive_domains, net, batch, fix_interm_bounds=True, adv_pool=None):
    if isinstance(arguments.Config["bab"]["decision_thresh"], torch.Tensor):
        decision_thresh = arguments.Config["bab"]["decision_thresh"].unique().item()
    else:
        decision_thresh = arguments.Config["bab"]["decision_thresh"]
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    candidates_number = arguments.Config["bab"]["attack"]["beam_candidates"]
    split_depth = arguments.Config["bab"]["attack"]["beam_depth"]
    submip_start_iteration = arguments.Config["bab"]["attack"]["mip_start_iteration"]
    max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
    min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]

    tot_ambi_nodes = net.tot_ambi_nodes
    max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
    min_local_free = int(min_local_free_ratio * tot_ambi_nodes)

    def merge_split_decisions(all_decisions, top_k=7):
        """Merge a list of list of decisions, and pick the top-k of decisions."""
        n_examples = len(all_decisions)
        flat_decisions = [tuple(decision) for example_decisions in all_decisions for decision in example_decisions]
        counter = Counter(flat_decisions)
        return [[list(c[0]) for c in counter.most_common(top_k)]] * n_examples

    # dive_domains store the dive domains
    global diving_Visited   # FIXME (10/21): Do not use global variables.

    total_time = time.time()

    pickout_time = time.time()
    print(f"iteration starts with beam search domain length: {len(dive_domains)}")
    dive_domains = probabilistic_select_domains(dive_domains, candidates_number)
    print(f"prune beam search domains to be length {len(dive_domains)}")
    # pickout the worst candidates_number domains
    domains_params = pickout_dive_domains(dive_domains, batch=candidates_number, device=net.x.device)
    mask, lAs, orig_lbs, orig_ubs, alphas, betas, _, selected_domains, cs, rhs = domains_params

    ###### Maybe we can apply integer fix here #######

    # throw away all the rest dive domains
    dive_domains = SortedList()
    pickout_time = time.time() - pickout_time

    # for each domain in dive_domains, select k (7 or 10) split decisions with kfsb
    decision_time = time.time()
    history = [sd.history for sd in selected_domains]
    split_history = [sd.split_history for sd in selected_domains]
    assert branching_method == "kfsb"
    # we need to select k decisions for each dive domain

    domains = {
        'lower_bounds': orig_lbs, 'upper_bounds': orig_ubs,
        'mask': mask, 'lAs': lAs, 'cs': cs, 'thresholds': rhs,
        'alphas': alphas, 'betas': betas, 'history': history
    }

    dive_decisions, _, _ = (
        KfsbBranching(net).get_branching_decisions(domains, split_depth,
                                                   branching_candidates=split_depth,
                                                   branching_reduceop=branching_reduceop,
                                                   method=branching_method,
                                                   prioritize_alphas="none",
                                                   # Change to "negative" to branch on upper bound first.
                                                   ))
    # In beam search, we use the same branching decisions for all nodes, so we merge branching decisions for different nodes into 1.
    merged_decisions = merge_split_decisions(dive_decisions, len(dive_decisions[0]))
    print('splitting decisions: {}'.format(merged_decisions[0]))
    if len(merged_decisions[0]) == 0:
        print("No beam search domains left, attack failed, please increase search candidates by increasing beam_candidates.")
        return torch.tensor(-np.inf), np.inf, dive_domains.clear()
    decision_time = time.time() - decision_time

    dive_time = time.time()
    # fill up the dive domains to be max_dive_domains (1024)
    # TODO: we should allow a loop here to run very large batch size. This is useful for CIFAR.
    domains_params = add_dive_domain_from_dive_decisions(selected_domains, merged_decisions, mask=mask, device=net.x.device)
    mask, orig_lbs, orig_ubs, orig_alphas, orig_betas, selected_domains = domains_params
    orig_history = [sd.history for sd in selected_domains]
    orig_split_history = [[] for i in range(len(orig_history))]  # This is not used in diving.
    dive_time = time.time() - dive_time

    solve_time = time.time()
    dom_ub, dom_lb, lAs, dom_lb_all, dom_ub_all, alphas, split_history, betas, cs = [], [], [], [], [], [], [], [], []
    primals = None
    # Divide all domains into multiple batches.
    batch_boundaries = torch.arange(0, len(orig_betas), batch).tolist() + [len(orig_betas)]
    for i in range(len(batch_boundaries) - 1):
        batch_start, batch_end = batch_boundaries[i], batch_boundaries[i+1]
        # orig_lbs and orig_ubs are organized per layer.
        batch_orig_lbs = [single_lbs[batch_start:batch_end] for single_lbs in orig_lbs]
        batch_orig_ubs = [single_ubs[batch_start:batch_end] for single_ubs in orig_ubs]

        batch_alphas = defaultdict.fromkeys(orig_alphas.keys(), None)
        for m_name in list(orig_alphas.keys()):
            for spec_name in orig_alphas[m_name]:
                batch_alphas[m_name] = {spec_name: orig_alphas[m_name][spec_name][:, :, batch_start:batch_end]}

        batch_history = orig_history[batch_start:batch_end]
        batch_split_history = orig_split_history[batch_start:batch_end]  # Not used.
        batch_betas = orig_betas[batch_start:batch_end]
        batch_split = {'decision': [], 'coeffs': [], 'diving': len(batch_history)}

        ret = net.update_bounds(
            {
              'lower_bounds': batch_orig_lbs, 'upper_bounds': batch_orig_ubs,
              'alphas': batch_alphas, 'betas': batch_betas,
              'history': batch_history, 'split_history': batch_split_history,
            },
            fix_interm_bounds=fix_interm_bounds, beta_bias=False)
        batch_dom_ub, batch_dom_lb, batch_lAs, batch_dom_lb_all, batch_dom_ub_all, batch_alphas, batch_split_history, batch_betas, batch_primals, batch_cs = (
            ret['upper_bounds'][-1], ret['lower_bounds'][-1], ret['lAs'],
            ret['lower_bounds'], ret['upper_bounds'], ret['alphas'],
            ret['split_history'], ret['betas'], ret['primals'], ret['c']
        )

        batch_cs = batch_cs.split(1, dim=0)
        for full_list, partial_list in ((dom_ub, batch_dom_ub), (dom_lb, batch_dom_lb), (cs, batch_cs),
                                        (split_history, batch_split_history), (betas, batch_betas)):
            full_list.extend(partial_list)

        # alphas must be handled specially as a nested dictionary.
        single_item_alphas = [defaultdict(dict) for i in range(len(batch_history))]
        for m_name in list(orig_alphas.keys()):
            for spec_name in orig_alphas[m_name]:
                # Split across batch dimension.
                all_alphas = batch_alphas[m_name][spec_name].split(1, dim=2)
                for i, s in enumerate(all_alphas):
                    single_item_alphas[i][m_name][spec_name] = all_alphas[i]
        alphas.extend(single_item_alphas)
        # per layer lower and upper bounds is a list of tensors; need to reshape to tensor of lists.
        def to_list_of_tensors(all_items):
            single_item_list = [[None] * len(all_items) for i in range(all_items[0].size(0))]
            for i, layer_item in enumerate(all_items):  # layer.
                for j, batch_item in enumerate(layer_item):  # batch.
                    single_item_list[j][i] = batch_item.unsqueeze(0)  # Add extra batch dimension.
            return single_item_list
        # lA is a list of tensors; need to reshape it.
        lAs.extend(to_list_of_tensors(batch_lAs))
        dom_lb_all.extend(to_list_of_tensors(batch_dom_lb_all))
        dom_ub_all.extend(to_list_of_tensors(batch_dom_ub_all))
        # "batch_primals" is a tensor.
        if primals is None:
            primals = batch_primals
        else:
            primals = torch.cat([primals, batch_primals], dim=0)

    if adv_pool is not None:
        adv_imgs = primals[np.argsort(dom_ub)[:50]]  # we only select best adv_imgs according to their upper bounds
        attack_ret, attack_images, attack_margin, _ = attack_with_general_specs(
                model=net.model_ori,
                x=adv_imgs,
                # shape [batch, num_spec, *input_shape]
                data_min=net.x.ptb.x_L[0].expand(adv_imgs.size(0), *[-1] * (net.x.ptb.x_L.ndim - 1)).unsqueeze(1),
                data_max=net.x.ptb.x_U[0].expand(adv_imgs.size(0), *[-1] * (net.x.ptb.x_U.ndim - 1)).unsqueeze(1),
                list_target_label_arrays=adv_imgs.size(0)*[((net.c[0].cpu(), torch.tensor([0.])),)],  # (prop_mat, rhs).
                initialization="none")
        # attack_images has shape (batch, spec, c, h, w).
        adv_pool.add_adv_images(attack_images.squeeze(1))  # Squeeze the spec dimension, since we only have 1 spec.
        adv_pool.print_pool_status()
        print(f'ub: {min(dom_ub)} -> ub (PGD): {min(attack_margin)}, best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')

    solve_time = time.time() - solve_time
    add_time = time.time()

    # See how these neurons are set in adv. examples, and we keep that domain for searching.
    activations = adv_pool.find_most_likely_activation(merged_decisions[0])
    # Get all the domains with this specific activations.
    activations = [str((aa + 1) // 2) for aa in activations]  # convert to 1, 0 instead of +1, -1
    domain_idx = int("".join(activations), base=2)
    priorities = torch.zeros(len(dom_lb))
    priorities[domain_idx] = 1.0
    print(f'decision in adv example {activations}, domain size {len(dom_lb)}')

    # add all 1024 domains back into dive domains
    ####### will not add extra split constraints here by default, set add_constraints=True if want to ######
    diving_unsat_list = add_dive_domain_parallel(lA=lAs, lb=dom_lb, ub=dom_ub,
                                            lb_all=dom_lb_all, ub_all=dom_ub_all,
                                            dive_domains=dive_domains, selected_domains=selected_domains,
                                            alpha=alphas, beta=betas, decision_thresh=decision_thresh,
                                            split_history=split_history, check_infeasibility=False,
                                            primals=primals,
                                            priorities=priorities, cs=cs)

    print('Beam search domains [lb, ub] (depth):')
    for i in dive_domains[:50]:
        if i.priority != 0:
            prio = f'(prio={i.priority:.2f})'
        else:
            prio = ""
        print(f'[{i.lower_bound:.5f}, {i.upper_bound:.5f}] ({i.depth}){prio}', end=', ')
    print()

    diving_Visited += len(selected_domains)
    print('Current worst domains:', [i.lower_bound for i in dive_domains[:10]])
    add_time = time.time() - add_time

    total_time = time.time() - total_time
    print('length of beam search domains:', len(dive_domains))
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t dive: {dive_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')

    dive_domains = probabilistic_select_domains(dive_domains, max(candidates_number, arguments.Config["solver"]["mip"]["parallel_solvers"]))
    print(f"prune beam search domains to {len(dive_domains)} according to probabilistic_select_domains()")

    # Run MIP for final solving adversarial examples.
    mip_ub, solutions = beam_mip_attack(net, adv_pool, dive_domains, submip_start_iteration, max_dive_fix, min_local_free, finalize=len(dive_domains) == 0)
    if len(dive_domains) > 0:
        global_lb = dive_domains[0].lower_bound
    else:
        print("No beam search domains left, attack failed, please increase search candidates by increasing beam_candidates.")
        return torch.tensor(-np.inf), np.inf, dive_domains

    # check dom_ub for adv
    batch_ub = np.inf
    if get_upper_bound:
        if adv_pool is not None:
            batch_ub = min(min(dom_ub), mip_ub, adv_pool.adv_pool[0].obj)
        else:
            batch_ub = min(dom_ub)
        print(f"Current lb:{global_lb}, ub:{batch_ub}")
    else:
        print(f"Current lb:{global_lb}")

    print(f'{diving_Visited} domains visited')

    return global_lb, batch_ub, dive_domains


def count_domain_unstable(domain):
    unstable_cnt = []
    for (lb, ub) in zip(domain.lower_all[:-1],  domain.upper_all[:-1]):
        unstable_cnt.append(torch.logical_and(lb < 0, ub > 0).sum().item())
    print("remaining unstable neurons:", unstable_cnt)


def bfs_splits_coeffs(num_splits):
    # parse all possible coeffs combinations consider the number of splits
    return [[int((float(c) - 0.5) * 2) for c in f"{{:0{num_splits}b}}".format(i)] for i in range(2**num_splits)]


def clone_to_dive(self):
    lower_all = upper_all = None   # These should not be used in beam search.
    beta = None  # This should not be used.
    history = [[None, None] for i in range(len(self.history))]  # Create an empty history for each layer.
    ####### Need to make sure we do not need to clone primals #######
    dive_d = ReLUDomain(lA=self.lA, lb=self.lower_bound, ub=self.upper_bound,
                lb_all=lower_all, up_all=upper_all, alpha=self.alpha,
                beta=beta, depth=self.depth,
                split_history=[], history=history, intermediate_betas=[],
                primals=self.primals, priority=self.priority, c=self.c)

    return dive_d


def to_sorted_list(self):
    """
        This function is only for supporting legacy code. It is slow. Avoid to use it frequently!
    :return:
    """
    now_len = len(self)
    ret = SortedList()
    for i in range(now_len):
        ret.add(self[i])
    return ret


def add_dive_domain_from_dive_decisions(dive_domains, dive_decisions, mask=None, device='cuda'):
    new_dive_domains = []

    dive_coeffs = {}

    # merged_new_lAs = []
    merged_lower_bounds = []
    merged_upper_bounds = []
    betas_all = []
    alphas_all = []
    ret_s = defaultdict.fromkeys(dive_domains[0].alpha.keys(), None)
    # intermediate_betas_all = []

    for di, dive_d in enumerate(dive_domains):
        decision = torch.tensor(dive_decisions[di], device='cpu').long()
        num_splits = len(dive_decisions[di])
        repeats = 2 ** num_splits

        expand_lb = [dive_d.lower_all[i].repeat(repeats, *([1] * (dive_d.lower_all[i].ndim - 1))) for i in range(len(dive_d.lower_all))]
        expand_ub = [dive_d.upper_all[i].repeat(repeats, *([1] * (dive_d.upper_all[i].ndim - 1))) for i in range(len(dive_d.upper_all))]

        if num_splits not in dive_coeffs:
            dive_coeffs[num_splits] = bfs_splits_coeffs(num_splits)
        # Generate beta. All subdomains generated by this domain has the same beta.
        # All subdomains also share the same decisions. They just have different coeffs in history.
        if dive_d.beta is None:
            # No existing beta, so generate a new one.
            new_beta = [torch.zeros(size=(0,), device=device) for _ in range(len(dive_d.history))]
        else:
            # Reuse existing beta in dive_d. This should be a cuda tensor.
            new_beta = dive_d.beta
        # For all subdomains we add the same decisions.
        decision_to_add_per_layer = []
        # Store which neurons are selected per layer.
        layer_idx_mask = []
        for layer_idx in range(len(dive_d.history)):
            # Adding new decisions for this layer.
            idx_mask = (decision[:, 0] == layer_idx)
            # Save this mask to be used later.
            layer_idx_mask.append(idx_mask)
            if idx_mask.size(0) == 0:
                # no node selected in this layer
                decision_to_add_per_layer.append(None)
                continue
            # Finding the location of splits in this layer.
            dive_loc = decision[idx_mask][:, 1].long()
            decision_to_add_per_layer.append(dive_d.history[layer_idx][0] + dive_loc.tolist())
            # Adding zeros to beta.
            new_beta[layer_idx] = torch.cat([new_beta[layer_idx], torch.zeros(dive_loc.size(0), device=device)])
        # Repeat beta for each layer, and add views to the list.
        new_beta = [b.view(1, -1).repeat(repeats, 1) for b in new_beta]
        for i in range(repeats):
            betas_all.append([b[i] for b in new_beta])  # This is just a view and will be added very quickly.

        # Deal with split history. This has to be done per-domain, however there are very few tensor operations here.
        dive_coeffs_t = torch.tensor(dive_coeffs[num_splits], device='cpu')
        for i in range(repeats):
            new_dive_d = clone_to_dive(dive_d)  # This will copy nothing.
            # This is just for generating the history. In each subdomain, only the history is different.
            for layer_idx in range(len(dive_d.history)):
                idx_mask = layer_idx_mask[layer_idx]
                if idx_mask.size(0) == 0:
                    # no node selected in this layer
                    continue
                this_layer_dive_coeffs = dive_coeffs_t[i][idx_mask]
                # add new dive constraints to dive domain history.
                new_dive_d.history[layer_idx][0] = decision_to_add_per_layer[layer_idx]
                new_dive_d.history[layer_idx][1] = dive_d.history[layer_idx][1] + this_layer_dive_coeffs.tolist()
            new_dive_d.depth += decision.size(0)
            new_dive_domains.append(new_dive_d)

        coeffs = np.array(dive_coeffs[num_splits])
        decisions = np.repeat(np.expand_dims(np.array(dive_decisions[di]), 0), repeats, axis=0)  # 1024, 10, 2
        zero_coeffs = np.argwhere(coeffs == -1)  # 5120, 2
        one_coeffs = np.argwhere(coeffs == 1)  # 5120, 2

        zero_idx = decisions[zero_coeffs[:, 0], zero_coeffs[:, 1]]  # 5120, 2
        one_idx = decisions[one_coeffs[:, 0], one_coeffs[:, 1]]  # 5120, 2
        for i in range(len(expand_lb)):
            selected_one_idx = np.argwhere(one_idx[:, 0] == i)
            selected_zero_idx = np.argwhere(zero_idx[:, 0] == i)

            if len(selected_one_idx) == 0 and len(selected_zero_idx) == 0:
                continue
            expand_lb[i].view(repeats, -1)[one_coeffs[:, 0][selected_one_idx.squeeze()], one_idx[selected_one_idx.squeeze()][:, 1]] = 0
            expand_ub[i].view(repeats, -1)[zero_coeffs[:, 0][selected_zero_idx.squeeze()], zero_idx[selected_zero_idx.squeeze()][:, 1]] = 0

        merged_lower_bounds.append(expand_lb)
        merged_upper_bounds.append(expand_ub)

        # for j in range(len(new_dive_d.lA)):
        #     new_lAs.append(new_dive_d.lA[j].repeat(repeats, *([1] * (len(new_dive_d.lA[j].shape) - 1))))
        # merged_new_lAs.append(new_lAs)

        assert isinstance(new_dive_d.alpha, dict)
        tmp_alpha = defaultdict.fromkeys(new_dive_d.alpha.keys(), None)
        for m_name in list(new_dive_d.alpha.keys()):
            for spec_name in new_dive_d.alpha[m_name]:
                tmp_alpha[m_name] = {spec_name: new_dive_d.alpha[m_name][spec_name].repeat(1, 1, repeats,  *([1] * (new_dive_d.alpha[m_name][spec_name].ndim - 3)))}
        alphas_all.append(tmp_alpha)

        # intermediate_betas_all += [new_dive_d.intermediate_betas] * repeats

    ret_lbs = []
    for j in range(len(merged_lower_bounds[0])):
        ret_lbs.append(torch.cat([merged_lower_bounds[i][j] for i in range(len(merged_lower_bounds))]))

    ret_ubs = []
    for j in range(len(merged_upper_bounds[0])):
        ret_ubs.append(torch.cat([merged_upper_bounds[i][j] for i in range(len(merged_upper_bounds))]))

    ret_lbs = [t.to(device=device, non_blocking=True) for t in ret_lbs]
    ret_ubs = [t.to(device=device, non_blocking=True) for t in ret_ubs]

    for m_name in list(alphas_all[0].keys()):
        for spec_name in alphas_all[0][m_name]:
            ret_s[m_name] = {spec_name: torch.cat([alphas_all[i][m_name][spec_name] for i in range(len(alphas_all))], dim=2)}

    # Recompute the mask on GPU.
    new_masks = []
    for j in range(len(ret_lbs) - 1):  # Exclude the final output layer.
        new_masks.append(
            torch.logical_and(ret_lbs[j] < 0, ret_ubs[j] > 0).view(ret_lbs[0].size(0), -1).float())
    print(f"expand original {len(dive_domains)} selected domains to {len(new_dive_domains)} with {num_splits} splits")

    return new_masks, ret_lbs, ret_ubs, ret_s, betas_all, new_dive_domains


def add_dive_domain_parallel(lA, lb, ub, lb_all, ub_all, dive_domains, selected_domains, alpha, beta,
                        split_history=None, decision_thresh=0,
                        check_infeasibility=True, primals=None, priorities=None,
                        cs=None):
    """
    Add current explored domains for beam search in the next iteration.
    """
    unsat_list = []
    batch = len(selected_domains)
    if primals.is_cuda:
        primals = primals.cpu()
    if isinstance(decision_thresh, torch.Tensor):
        decision_thresh = decision_thresh.to(lb[0].device)
    for i in range(batch):
        infeasible = False
        if lb[i] < decision_thresh:
            if check_infeasibility:
                if check_infeasible_bounds(lb_all[i], ub_all[i], reduce=True):
                    infeasible = True

            if not infeasible:
                priority=0 if priorities is None else priorities[i].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                dive_primals = primals[i] if primals is not None else None
                dive_d = ReLUDomain(lA[i], lb[i].item(), ub[i].item(), lb_all[i], ub_all[i], alpha[i], beta[i],
                                  selected_domains[i].depth+0, split_history=split_history[i],  # depth has been added during diving.
                                  history=new_history,
                                  primals=dive_primals, priority=priority,
                                  c=cs[i] if cs is not None else None)

                dive_domains.add(dive_d)

    return unsat_list


def pickout_dive_domains(domains, batch, device='cuda', diving=False):
    """
    Pick the first batch of domains in the `domains` sequence
    that has still not meet verify_criterion().
    dive_rate: how many times of dive domains over selected domains picked out
    Returns: CandidateDomain with the lowest reference_value.
    """
    assert batch > 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

    idx, idx2 = 0, 0
    batch = min(len(domains), batch)
    lAs, lower_all, upper_all, alphas_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
    dm_l_all, dm_u_all = [], []
    c_all, thresholds_all = [], []
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        # Pop out domains from the list one by one (SLOW).
        if len(domains) == 0:
            print(f"No domain left to pick from. Batch limit {batch} current batch: {idx}")
            break
        if idx2 == len(domains): break  # or len(domains)-1?
        if domains[idx2].split is True:
            idx2 += 1
            continue
        selected_candidate_domain = domains.pop(idx2)
        # idx2 -= 1
        if not selected_candidate_domain.verify_criterion() and selected_candidate_domain.valid is True:
            # unique = [x for i, x in enumerate(selected_candidate_domain.history) if i == selected_candidate_domain.history.index(x)]
            # assert len(unique) == len(selected_candidate_domain.history)
            # We transfer only some of the tensors directly to GPU. Other tensors will be transferred in batch later.
            selected_candidate_domain.to_device(device, partial=True)
            idx += 1
            lAs.append(selected_candidate_domain.lA)
            lower_all.append(selected_candidate_domain.lower_all)
            upper_all.append(selected_candidate_domain.upper_all)
            alphas_all.append(selected_candidate_domain.alpha)
            betas_all.append(selected_candidate_domain.beta)
            intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
            dm_l_all.append(selected_candidate_domain.dm_l)
            dm_u_all.append(selected_candidate_domain.dm_u)
            c_all.append(selected_candidate_domain.c)
            thresholds_all.append(selected_candidate_domain.threshold)
            selected_candidate_domains.append(selected_candidate_domain)
            selected_candidate_domain.valid = False  # set False to avoid another pop
            if idx == batch: break
        # else:
        #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate_domain.lower_bound, selected_candidate_domain.split))
        selected_candidate_domain.valid = False   # set False to avoid another pop

    batch = idx

    lower_bounds = []
    upper_bounds = []
    new_lAs = []
    new_masks = []
    for j in range(len(lower_all[0])):
        lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
    lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

    for j in range(len(upper_all[0])):
        upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
    upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

    # Reshape to batch first in each list.
    for j in range(len(lAs[0])):
        new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
    # Transfer to GPU.
    new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

    # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
    lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
    upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]

    # Recompute the mask on GPU.
    for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
        new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())

    thresholds = torch.tensor(thresholds_all).to(device=device, non_blocking=True)

    # aggregate C to shape (batch, 1, num_outputs)
    cs = torch.cat(c_all).to(device=device, non_blocking=True)
    if not cs.is_contiguous():
        cs = cs.contiguous()

    alphas = defaultdict(dict)
    # alphas_all is a two level dictionary.
    for k, v_map in alphas_all[0].items():
        alphas[k] = {}
        for kk, vv in v_map.items():
            alphas_for_this_layer = []
            for s in alphas_all:
                # Go over the batch dimension and add all alphas for this key pair.
                alphas_for_this_layer.append(s[k][kk])
            alphas[k][kk] = torch.cat(alphas_for_this_layer, dim=2).to(device=device, non_blocking=True)

    return new_masks, new_lAs, lower_bounds, upper_bounds, alphas, betas_all, intermediate_betas_all, selected_candidate_domains, cs, thresholds


def init_bab_attack(net, mask, attack_images):
    adv_pool = AdvExamplePool(net.net, mask, C=net.c)
    adv_pool.add_adv_images(attack_images)
    print(f'best adv in pool: {adv_pool.adv_pool[0].obj}, '
          f'worst {adv_pool.adv_pool[-1].obj}')
    adv_pool.print_pool_status()

    find_promising_domains.counter = 0
    find_promising_domains.current_method = "top-down"
    find_promising_domains.topdown_status = "normal"
    find_promising_domains.bottomup_status = "normal"
    beam_mip_attack.started = False

    return adv_pool
