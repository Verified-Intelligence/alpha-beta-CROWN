#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import time
import random
import numpy as np
import torch
from collections import defaultdict, Counter

from auto_LiRPA.utils import stop_criterion_sum
from branching_domains import pick_out_batch, add_domain_parallel, ReLUDomain, SortedList, DFS_SortedList, merge_domains_params
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB
import arguments


Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False
global_ub = np.inf
DFS_enabled = False


def batch_verification(d, net, batch, pre_relu_indices, growth_rate, layer_set_bound=True,
                       dive_domains=[], adv_pool=None):
    global Visited, Flag_first_split
    global Use_optimized_split
    global global_ub
    global DFS_enabled

    opt_intermediate_beta = False
    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    DFS_percent = arguments.Config["bab"]["dfs_percent"]
    branching_candidates = arguments.Config["bab"]["branching"]["candidates"]
    dive_rate = 0

    total_time = time.time()

    pickout_time = time.time()
    #### add new domains into dive_domains

    domains_params = pick_out_batch(d, decision_thresh, batch=batch * (1 - dive_rate), device=net.x.device, DFS_percent=DFS_percent if DFS_enabled else 0)
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = domains_params

    pickout_time = time.time() - pickout_time

    if mask is not None:
        decision_time = time.time()

        # print('history', selected_domains[0].history)
        history = [sd.history for sd in selected_domains]
        split_history = [sd.split_history for sd in selected_domains]

        if branching_method == 'babsr':
            branching_decision = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                                            batch=batch, branching_reduceop=branching_reduceop)
        elif branching_method == 'fsb':
            branching_decision = choose_node_parallel_FSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                            branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                            slopes=slopes, betas=betas, history=history)
        elif branching_method == 'kfsb':
            branching_decision = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                            branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                            slopes=slopes, betas=betas, history=history)
        else:
            raise NotImplementedError

        if len(branching_decision) < len(mask[0]):
            print('all nodes are split!!')
            global all_node_split
            all_node_split = True
            return selected_domains[0].lower_bound, global_ub


        # Use_optimized_split = not Use_optimized_split
        print('splitting decisions: {}'.format(branching_decision[:10]))
        # print("splitting coeffs: {}".format(split["coeffs"]))

        if not Use_optimized_split:
            split = {}
            # split["decision"]: selected domains (next batch/2)->node list->node: [layer, idx]
            split["decision"] = [[bd] for bd in branching_decision]
            # split["split"]: selected domains (next batch/2)->node list->float coefficients
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]
        else:
            split = {}
            split["decision"] = [[[2, i] for i in range(100)] for bd in branching_decision]
            split["coeffs"] = [[random.random() * 0.001 - 0.0005 for j in range(100)] for i in
                               range(len(branching_decision))]
        split["diving"] = 0

        decision_time = time.time() - decision_time

        solve_time = time.time()
        single_node_split = True
        ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,
                                split_history=split_history, layer_set_bound=layer_set_bound, betas=betas,
                                single_node_split=single_node_split, intermediate_betas=intermediate_betas)
        dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = ret

        if adv_pool is not None:
            adv_pool.add_adv_images(primals)

        solve_time = time.time() - solve_time
        add_time = time.time()
        batch, diving_batch = len(branching_decision), split["diving"]
        # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
        check_infeasibility = not (single_node_split and layer_set_bound)
        unsat_list = add_domain_parallel(lA=lAs[:2*batch], lb=dom_lb[:2*batch], ub=dom_ub[:2*batch], lb_all=dom_lb_all[:2*batch], up_all=dom_ub_all[:2*batch],
                                         domains=d, selected_domains=selected_domains[:batch], slope=slopes[:2*batch], beta=betas[:2*batch],
                                         growth_rate=growth_rate, branching_decision=branching_decision, decision_thresh=decision_thresh,
                                         split_history=split_history[:2*batch], intermediate_betas=intermediate_betas[:2*batch],
                                         check_infeasibility=check_infeasibility, primals=primals[:2*batch] if primals is not None else None)

        Visited += (len(selected_domains) - diving_batch - len(unsat_list)) * 2  # one unstable neuron split to two nodes
        print('Current worst splitting domains [lb, ub] (depth):')
        for i in d[:20]:
            print(f'[{i.lower_bound:.5f}, {i.upper_bound:5f}] ({i.depth})', end=', ')
        print()
        if hasattr(d, 'sublist'):
            print(f'Max depth domain: [{d.sublist[0].domain.lower_bound}, {d.sublist[0].domain.upper_bound}] ({d.sublist[0].domain.depth})')
        add_time = time.time() - add_time

    total_time = time.time() - total_time
    print('length of domains:', len(d))
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')

    if len(d) > 0:
        global_lb = d[0].lower_bound
    else:
        print("No domains left, verification finished!")
        return torch.tensor(arguments.Config["bab"]["decision_thresh"] + 1e-7), global_ub

    if get_upper_bound:
        batch_ub = min(dom_ub)
        if batch_ub < global_ub:
            global_ub = batch_ub
        print(f"Current lb:{global_lb}, ub:{global_ub}")
    else:
        print(f"Current lb:{global_lb}")

    print('{} neurons visited'.format(Visited))

    return global_lb, global_ub



def relu_bab_parallel(net, domain, x, use_neuron_set_strategy=False, refined_lower_bounds=None,
                      refined_upper_bounds=None, reference_slopes=None, attack_images=None):
    start = time.time()
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split, global_ub, DFS_enabled

    opt_intermediate_beta = False
    timeout = arguments.Config["bab"]["timeout"]
    max_domains = arguments.Config["bab"]["max_domains"]
    batch = arguments.Config["solver"]["beta-crown"]["batch_size"]
    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    record = arguments.Config["general"]["record_bounds"]
    lp_test = arguments.Config["debug"]["lp_test"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    DFS_percent = arguments.Config["bab"]["dfs_percent"]

    attack_args = getattr(arguments, 'attack_args', None)

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    DFS_enabled = False
    adv_pool = None

    if arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model_with_refined_bounds(
            domain, x, None, None, stop_criterion_func=stop_criterion_sum(decision_thresh), reference_slopes=None)
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model(
            domain, x, stop_criterion_func=stop_criterion_sum(decision_thresh))
    else:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, stop_criterion_func=stop_criterion_sum(decision_thresh),reference_slopes=reference_slopes)

    if lp_test in ["LP_intermediate_refine", "MIP_intermediate_refine"]:
        glb = net.build_the_model_lp(lower_bounds, upper_bounds)
        print("initial LP:", glb)

    if isinstance(global_lb, torch.Tensor): global_lb = global_lb.item()
    if lp_test in ["LP", "MIP"]:
        return global_lb, global_ub, [[time.time()-start, global_lb]], 0
    # return global_lb, global_ub, [[time.time()-start, global_lb]], 0

    print(global_lb)
    if global_lb > decision_thresh:
        return global_lb, global_ub, [[time.time()-start, global_lb]], 0

    if True:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        new_slope = defaultdict(dict)
        output_layer_name = net.net.final_name
        for relu_layer, alphas in slope.items():
            new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
        slope = new_slope

    # This is the first (initial) domain.
    candidate_domain = ReLUDomain(lA, global_lb, global_ub, lower_bounds, upper_bounds, slope, history=history, depth=0, primals=primals).to_cpu()
    domains = DFS_SortedList() if DFS_percent > 0 else SortedList()
    dive_domains = DFS_SortedList() if DFS_percent > 0 else SortedList()
    domains.add(candidate_domain)

    tot_ambi_nodes = 0
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print('# of unstable neurons:', tot_ambi_nodes)


    glb_record = [[time.time()-start, global_lb]]
    stop_condition = len(domains) > 0
    # while len(domains) > 0:
    while stop_condition:


        if True:
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:  # do two batch of neuron set bounds  per 10000 domains
                # neuron set  bounds cost more memory, we set a smaller batch here
                global_lb, global_ub = batch_verification(domains, net, int(batch/2), pre_relu_indices, 0, layer_set_bound=False,
                                            dive_domains=dive_domains,
                                            adv_pool=adv_pool)
            else:
                global_lb, global_ub = batch_verification(domains, net, batch, pre_relu_indices, 0,
                                            layer_set_bound=not opt_intermediate_beta,
                                            dive_domains=dive_domains,
                                            adv_pool=adv_pool)
        stop_condition = len(domains) > 0

        if isinstance(global_lb, torch.Tensor): global_lb = global_lb.item()

        if all_node_split:
            del domains
            all_node_split = False
            return global_lb, global_ub, glb_record, Visited

        if len(domains) > max_domains:
            print("No enough memory for the domain list!!!!!!!!")
            del domains
            return global_lb, global_ub, glb_record, Visited

        if get_upper_bound:
            if global_ub < decision_thresh:
                print("Attack success during bab!!!!!!!!")
                # Terminate MIP if it has been started.
                if beam_mip_attack.started:
                    print('Terminating MIP processes...')
                    net.pool_termination_flag.value = 1
                del domains
                return global_lb, global_ub, glb_record, Visited

        if time.time() - start > timeout:
            print('Time out!!!!!!!!')
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, global_ub, glb_record, Visited

        if record:
            glb_record.append([time.time() - start, global_lb])
        print(f'Cumulative time: {time.time() - start}\n')

    del domains
    return global_lb, global_ub, glb_record, Visited
