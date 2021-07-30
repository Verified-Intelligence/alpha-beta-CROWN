import time
import random
import numpy as np
import torch
from collections import defaultdict
from sortedcontainers import SortedList

from auto_LiRPA.utils import stop_criterion_sum
from branching_domains import pick_out_batch, add_domain_parallel, ReLUDomain
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB

Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False


def batch_verification(d, net, batch, pre_relu_indices, no_LP, growth_rate, decision_thresh = 0, layer_set_bound=True,
                       beta=True, branching_method='sb-min', lr_alpha=0.1, lr_beta=0.05, optimizer="adam",
                       iteration=20, beta_warmup=True, opt_coeffs=True, opt_bias=True, lp_test=None,
                       opt_intermediate_beta=True, intermediate_refinement_layers=None, branching_reduceop='min', branching_candidates=1):
    global Visited, Flag_first_split
    global Use_optimized_split

    total_time = time.time()

    pickout_time = time.time()
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = pick_out_batch(d, decision_thresh, batch, device=net.x.device)
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
                                                          branching_candidates=branching_candidates, branching_reduceop=branching_reduceop, slopes=slopes, betas=betas, history=history)
        elif branching_method == 'kfsb':
            branching_decision = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                                           branching_candidates=branching_candidates, branching_reduceop=branching_reduceop, slopes=slopes, betas=betas, history=history)
        else:
            raise NotImplementedError

        if len(branching_decision) < len(mask[0]):
            print('all nodes are split!!')
            global all_node_split
            all_node_split = True
            return selected_domains[0].lower_bound

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
        # Use_optimized_split = not Use_optimized_split
        print('splitting decisions: {}'.format(branching_decision[:10]))
        # print("splitting coeffs: {}".format(split["coeffs"]))

        decision_time = time.time() - decision_time

        solve_time = time.time()
        single_node_split = not opt_coeffs and not opt_bias and not opt_intermediate_beta
        print('single_node_split:', single_node_split)
        ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history, split_history=split_history, iteration=iteration,
                                  decision_thresh=decision_thresh, layer_set_bound=layer_set_bound, beta=beta, betas=betas,
                                  lr_alpha=lr_alpha, lr_beta=lr_beta, optimizer=optimizer,
                                  beta_warmup=beta_warmup, opt_coeffs=opt_coeffs, opt_bias=opt_bias, lp_test=lp_test, single_node_split=single_node_split,
                                  opt_intermediate_beta=opt_intermediate_beta, intermediate_refinement_layers=intermediate_refinement_layers, intermediate_betas=intermediate_betas)
        dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas = ret
        solve_time = time.time() - solve_time
        add_time = time.time()
        # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
        check_infeasibility = not (single_node_split and layer_set_bound)
        unsat_list = add_domain_parallel(lA=lAs, lb=dom_lb, ub=dom_ub, lb_all=dom_lb_all, up_all=dom_ub_all,
                                         domains=d, selected_domains=selected_domains, slope=slopes, beta=betas,
                                         growth_rate=growth_rate, branching_decision=branching_decision, decision_thresh=decision_thresh,
                                         split_history=split_history, intermediate_betas=intermediate_betas,
                                         check_infeasibility=check_infeasibility)
        Visited += (len(selected_domains) - len(unsat_list)) * 2  # one unstable neuron split to two nodes
        print('Current worst domains:', [i.lower_bound for i in d[:10]])
        add_time = time.time() - add_time

    total_time = time.time() - total_time
    print('length of domains:', len(d))
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')

    if len(d) > 0:
        global_lb = d[0].lower_bound
    else:
        print("No domains left, verification finished!")
        return torch.tensor(999)

    print(f"Current lb:{global_lb}")

    print('{} neurons visited\n'.format(Visited))

    return global_lb


def relu_bab_parallel(net, domain, x, args, no_LP=False, use_neuron_set_strategy=False, record=False, refined_lower_bounds=None, refined_upper_bounds=None):
    start = time.time()
    # All supported arguments.
    batch = args.batch_size
    decision_thresh = args.decision_thresh
    beta = not args.no_beta
    max_subproblems_list = args.max_subproblems_list
    iteration = args.iteration
    timeout = args.timeout
    lr_alpha = args.lr_alpha
    lr_beta = args.lr_beta
    lr_init_alpha = args.lr_init_alpha
    branching_method = args.branching_method
    beta_warmup  =  args.beta_warmup
    opt_coeffs = args.opt_coeffs
    opt_bias = args.opt_bias
    lp_test = args.lp_test
    opt_intermediate_beta = args.opt_intermediate_beta
    intermediate_refinement_layers = args.intermediate_refinement_layers
    share_slopes = args.share_slopes
    branching_candidates = args.branching_candidates
    branching_reduceop = args.branching_reduceop
    global Visited, Flag_first_split, all_node_split
    optimizer = args.optimizer
    Visited, Flag_first_split = 0, True
    if refined_lower_bounds is None or refined_upper_bounds is None:
        global_ub, global_lb, _, _, _, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model(
            domain, x, no_lp=no_LP, stop_criterion_func=stop_criterion_sum(decision_thresh),
            lr_init_alpha=lr_init_alpha, optimizer=optimizer, share_slopes=share_slopes)
    else:
        global_ub, global_lb, _, _, _, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, no_lp=no_LP, stop_criterion_func=stop_criterion_sum(decision_thresh),
            lr_init_alpha=lr_init_alpha, optimizer=optimizer, share_slopes=share_slopes)

    if isinstance(global_lb, torch.Tensor): global_lb = global_lb.item()
    if lp_test in ["LP", "MIP"]:
        return global_lb, global_ub, [[time.time()-start, global_lb]], 0
    # return global_lb, global_ub, [[time.time()-start, global_lb]], 0

    if not opt_intermediate_beta:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        new_slope = defaultdict(dict)
        output_layer_name = net.net.final_name
        for relu_layer, alphas in slope.items():
            new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
        slope = new_slope

    print(global_lb)
    if global_lb > decision_thresh:
        return global_lb, global_ub, [[time.time()-start, global_lb]], 0

    # This is the first (initial) domain.
    candidate_domain = ReLUDomain(lA, global_lb, global_ub, lower_bounds, upper_bounds, slope, history=history, depth=0).to_cpu()
    domains = SortedList()
    domains.add(candidate_domain)
    tot_ambi_nodes = 0
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = torch.sum(layer_mask == -1).item()
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print('# of unstable neurons:', tot_ambi_nodes)
    random_order = np.arange(tot_ambi_nodes)
    np.random.shuffle(random_order)

    glb_record = [[time.time()-start, global_lb]]
    while len(domains) > 0:

        if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:  # do two batch of neuron set bounds  per 10000 domains
            # neuron set  bounds cost more memory, we set a smaller batch here
            global_lb = batch_verification(domains, net, int(batch/2), pre_relu_indices, no_LP, 0, iteration=iteration,
                                           decision_thresh=decision_thresh, layer_set_bound=False, beta=beta,
                                           lr_alpha=lr_alpha, lr_beta=lr_beta, optimizer=optimizer,
                                           branching_method=branching_method, beta_warmup=beta_warmup,
                                           opt_coeffs=opt_coeffs, opt_bias=opt_bias, lp_test=lp_test,
                                           branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                           opt_intermediate_beta=opt_intermediate_beta, intermediate_refinement_layers=intermediate_refinement_layers)
        else:
            global_lb = batch_verification(domains, net, batch, pre_relu_indices, no_LP, 0, decision_thresh=decision_thresh, iteration=iteration,
                                           beta=beta, lr_alpha=lr_alpha, lr_beta=lr_beta,
                                           branching_method=branching_method, beta_warmup=beta_warmup,
                                           opt_coeffs=opt_coeffs, opt_bias=opt_bias, lp_test=lp_test,
                                           branching_candidates=branching_candidates, branching_reduceop=branching_reduceop, optimizer=optimizer,
                                           layer_set_bound=not opt_intermediate_beta,
                                           opt_intermediate_beta=opt_intermediate_beta,
                                           intermediate_refinement_layers=intermediate_refinement_layers
                                           )
        
        if isinstance(global_lb, torch.Tensor): global_lb = global_lb.item()

        if all_node_split:
            del domains
            all_node_split = False
            return global_lb, np.inf, glb_record, Visited

        if len(domains) > max_subproblems_list:
            print("no enough memory for the domain list")
            del domains
            return global_lb, np.inf, glb_record, Visited

        # if Visited >= 180:
        #     # if Visited>=8:
        #     del domains
        #     return global_lb, np.inf, glb_record, Visited

        if time.time() - start > timeout:
            print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, np.inf, glb_record, Visited

        if record: 
            glb_record.append([time.time() - start, global_lb])

    del domains
    return global_lb, np.inf, glb_record, Visited
