#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for input space split."""

import time
import numpy as np
import torch
import math

import arguments
from auto_LiRPA.utils import stop_criterion_batch
from branching_domains_input_split import (
    UnsortedInputDomainList,
    SortedInputDomainList,
)

from branching_heuristics import input_split_parallel, input_split_branching, get_split_depth
from attack_pgd import pgd_attack_with_general_specs, test_conditions, gen_adv_example

Visited, Solve_slope, storage_depth = 0, False, 0


def batch_verification_input_split(
    d,
    net,
    batch,
    decision_thresh,
    shape=None,
    bounding_method="crown",
    branching_method="sb",
    stop_func=stop_criterion_batch,
):
    split_start_time = time.time()
    global Visited

    # STEP 1: find the neuron to split and create new split domains.
    pickout_start_time = time.time()
    ret = d.pick_out_batch(batch, device=net.x.device)
    dom_ub = dom_lb = None
    pickout_time = time.time() - pickout_start_time

    # STEP 2: find the neuron to split and create new split domains.
    decision_start_time = time.time()
    slopes, dm_l_all, dm_u_all, cs, thresholds, split_idx = ret

    split_depth = get_split_depth(dm_l_all)
    new_dm_l_all, new_dm_u_all, cs, thresholds, split_depth = input_split_parallel(
        dm_l_all, dm_u_all, shape, cs, thresholds, split_depth=split_depth, i_idx=split_idx)

    slopes = slopes * (2 ** (split_depth - 1))

    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains.
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes,
        bounding_method=bounding_method, C=cs,
        stop_criterion_func=stop_func(thresholds),
    )
    # here slopes is a dict
    dom_lb, dom_ub, slopes, lA = ret
    bounding_time = time.time() - bounding_start_time
    decision_time -= time.time()

    split_idx = input_split_branching(net, dom_lb, new_dm_l_all, new_dm_u_all, lA, thresholds, branching_method, None, shape, slopes, storage_depth)
    decision_time += time.time()
    # STEP 4: Add new domains back to domain list.
    adddomain_start_time = time.time()
    d.add(
        dom_lb,
        new_dm_l_all.detach(),
        new_dm_u_all.detach(),
        slopes,
        cs,
        thresholds,
        split_idx,
    )
    adddomain_time = time.time() - adddomain_start_time

    total_time = time.time() - split_start_time
    print(
        f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  add_domain: {adddomain_time:.4f}"
    )
    print("length of domains:", len(d))

    if len(d) == 0:
        print("No domains left, verification finished!")
        if dom_lb is not None:
            print(f"The lower bound of last batch is {dom_lb.min().item()}")
        return decision_thresh.max() + 1e-7
    else:
        worst_idx = d.get_topk_indices().item()
        worst_val = d[worst_idx]
        global_lb = worst_val[0] - worst_val[-1]

    Visited += len(new_dm_l_all)
    print(f"Current (lb-rhs): {global_lb.max().item()}")
    print("{} branch and bound domains visited\n".format(Visited))

    return global_lb


def input_bab_parallel(
    net,
    init_domain,
    x,
    model_ori=None,
    all_prop=None,
    rhs=None,
    timeout=None,
    branching_method="naive",
):
    global storage_depth

    # the crown_lower/upper_bounds are dummy arguments here --- similar to refined_lower/upper_bounds, they are not used
    """ run input split bab """
    prop_mat_ori = net.c[0]

    start = time.time()
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled

    timeout = timeout or arguments.Config["bab"]["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    record = arguments.Config["general"]["record_bounds"]
    bounding_method = arguments.Config["solver"]["bound_prop_method"]
    adv_check = arguments.Config['bab']['branching']['input_split']['adv_check']

    stop_func = stop_criterion_batch

    Visited, Flag_first_split, global_ub = 0, True, np.inf

    (
        global_ub,
        global_lb,
        _,
        _,
        primals,
        updated_mask,
        lA,
        lower_bounds,
        upper_bounds,
        pre_relu_indices,
        slope,
        history,
        attack_image,
    ) = net.build_the_model(
        init_domain,
        x,
        stop_criterion_func=stop_func(rhs),
        bounding_method=bounding_method,
    )
    if hasattr(net.net[net.net.input_name[0]], 'lA'):
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U
    split_depth = get_split_depth(dm_l)

    # compute storage depth
    use_slope = arguments.Config["solver"]["bound_prop_method"] == "alpha-crown"
    min_batch_size = (
        arguments.Config["solver"]["min_batch_size_ratio"]
        * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1])

    # compute initial split idx
    split_idx = input_split_branching(net, global_lb, dm_l, dm_u, lA, rhs, branching_method, None, None, None, storage_depth)

    initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)
    if initial_verified:
        return (
            global_lb.max(),
            None,
            [[time.time() - start, global_lb.max().item()]],
            0,
            "safe",
        )

    if arguments.Config["bab"]["batched_domain_list"]:
        domains = UnsortedInputDomainList(storage_depth, use_slope=use_slope)
    else:
        domains = SortedInputDomainList()

    domains.add(
        global_lb,
        dm_l.detach(),
        dm_u.detach(),
        slope,
        net.c,
        rhs,
        split_idx,
        remaining_index,
    )

    glb_record = [[time.time() - start, (global_lb - rhs).max().item()]]

    if arguments.Config["attack"]["pgd_order"] == "after":
        ## pack the domain list
        lbs, ubs, Cs, rhs = [], [], [], []
        for idx in range(len(domains)):
            val = domains[idx]
            lbs.append(val[1][None, ...])
            ubs.append(val[2][None, ...])
            Cs.append(val[3][None, ...])
            rhs.append(val[4][None, ...])

        lbs = torch.cat(lbs, dim=0)
        # [num_or_spec, input_shape]
        ubs = torch.cat(ubs, dim=0)
        # [num_or_spec, input_shape]
        Cs = torch.cat(Cs, dim=0)
        # [num_or_spec, num_and_spec, output_dim]
        rhs = torch.cat(rhs, dim=0)
        # [num_or_spec, num_and_spec]

        cond_mat = [[Cs.shape[1] for i in range(Cs.shape[0])]]
        Cs = Cs.view(1, -1, Cs.shape[-1])
        # [num_example, num_spec, num_output]
        rhs = rhs.view(1, -1)
        # [num_example, num_spec]
        lbs = lbs.unsqueeze(0)
        ubs = ubs.unsqueeze(0)
        # [num_example, num_or_spec, input_shape]

        if arguments.Config["attack"]["input_split"]["pgd_alpha"] == "auto":
            alpha = (ubs - lbs).max() / 4
        else:
            alpha = float(arguments.Config["attack"]["input_split"]["pgd_alpha"])
        # pack the domains as a large spec matrix

        num_restarts = arguments.Config["attack"]["input_split"]["pgd_restarts"]
        num_steps = arguments.Config["attack"]["input_split"]["pgd_steps"]

        device = x.device
        lbs = lbs.to(device)
        ubs = ubs.to(device)
        rhs = rhs.to(device)
        Cs = Cs.to(device)

        attack_x = ((lbs + ubs)/2).squeeze(0)

        best_deltas, _ = pgd_attack_with_general_specs(model_ori, attack_x, lbs, ubs, Cs, rhs, cond_mat, same_number_const=True, alpha=alpha, set_pgd_steps=num_steps, set_num_restarts=num_restarts)
        attack_image, attack_output, _ = gen_adv_example(model_ori, attack_x, best_deltas, ubs, lbs, Cs, rhs, cond_mat)

        if test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1), Cs, rhs, cond_mat, True, ubs, lbs).all():
            print("pgd attack succeed in input_bab_parallel")
            return global_lb.max(), np.inf, glb_record, Visited, "unsafe"

    num_iter = 1
    sort_domain_iter = arguments.Config["bab"]["branching"]["input_split"]["sort_domain_interval"]
    enhanced_bound_initialized = False
    while len(domains) > 0:
        # sort the domains every certain number of iterations
        if (isinstance(domains, UnsortedInputDomainList) and
                sort_domain_iter > 0 and num_iter % sort_domain_iter == 0):
            sort_start_time = time.time()
            domains.sort()
            sort_time = time.time() - sort_start_time
            print(f"Sorting domains used {sort_time:.4f}s\n")

        last_glb = global_lb.max()

        if Visited > adv_check:
            adv_check_start_time = time.time()
            # check whether adv example found
            ub, ret_adv = check_adv(domains, model_ori)
            if ret_adv:
                return global_lb.max(), ub, glb_record, Visited, "unsafe"
            adv_check_time = time.time() - adv_check_start_time
            print(f"Adv attack time: {adv_check_time:.4f}s")

        global_lb = batch_verification_input_split(
            domains,
            net,
            batch,
            decision_thresh=rhs,
            shape=x.shape,
            bounding_method=bounding_method,
            branching_method=branching_method,
            stop_func=stop_func,
        )

        # once the lower bound stop improving we change to solve slope mode
        if (
            arguments.Config["solver"]["bound_prop_method"]
            != arguments.Config["bab"]["branching"]["input_split"][
                "enhanced_bound_prop_method"
            ]
            and time.time() - start
            > arguments.Config["bab"]["branching"]["input_split"][
                "enhanced_bound_patience"
            ]
            and global_lb.max().cpu() <= last_glb.cpu()
            and bounding_method != "alpha-crown"
            and not enhanced_bound_initialized
        ):
            branching_method = arguments.Config["bab"]["branching"]["input_split"]["enhanced_branching_method"]
            bounding_method = arguments.Config["bab"]["branching"]["input_split"]["enhanced_bound_prop_method"]
            print(f'Using enhanced bound propagation method {bounding_method} with {branching_method} branching.')
            enhanced_bound_initialized = True

            (
                global_ub,
                global_lb,
                _,
                _,
                primals,
                updated_mask,
                lA,
                lower_bounds,
                upper_bounds,
                pre_relu_indices,
                slope,
                history,
                attack_image,
            ) = net.build_the_model(
                init_domain,
                x,
                stop_criterion_func=stop_func(rhs),
                bounding_method=bounding_method,
            )
            if hasattr(net.net[net.net.input_name[0]], 'lA'):
                lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
            else:
                raise ValueError("sb heuristic cannot be used without lA.")
            dm_l = x.ptb.x_L
            dm_u = x.ptb.x_U

            # compute initial split idx for the enhanced method
            split_idx = input_split_branching(net, global_lb, dm_l, dm_u, lA, rhs, branching_method, None, None, None, storage_depth)

            if arguments.Config["bab"]["batched_domain_list"]:
                use_slope = (
                    arguments.Config["bab"]["branching"]["input_split"][
                        "enhanced_bound_prop_method"
                    ]
                    == "alpha-crown"
                )
                domains = UnsortedInputDomainList(storage_depth, use_slope=use_slope)
            else:
                domains = SortedInputDomainList()
            # This is the first batch of initial domain(s) after the branching method changed.
            domains.add(
                global_lb,
                dm_l.detach(),
                dm_u.detach(),
                slope,
                net.c,
                rhs,
                split_idx,
            )
            global_lb = global_lb.max()

        if (
            time.time() - start
            > arguments.Config["bab"]["branching"]["input_split"]["attack_patience"]
        ):
            print("Perform PGD attack with massively random starts finally.")
            ub, ret_adv = massive_pgd_attack(
                init_domain, rhs, x, model_ori, prop_mat_ori
            )
            if ret_adv:
                del domains
                return global_lb.max(), ub.max(), glb_record, Visited, "unsafe"

        if time.time() - start > timeout:
            print("time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb.max(), None, glb_record, Visited, "unknown"

        if record:
            glb_record.append([time.time() - start, global_lb.cpu().numpy()])

        num_iter += 1

    del domains
    return global_lb.max(), None, glb_record, Visited, "safe"


def massive_pgd_attack(init_domain, rhs_mat, x, model_ori, C_mat):
    """pgd attack with very large number of random starts
    init_domain: [input_shape, 2]
    rhs_mat: [num_or(1), num_and]
    x: [batch(1), input_shape]
    C_mat: [num_and, out_dim]
    """
    # only attack the first specification when we have multiple specification in this domain
    if init_domain.ndim == 3:
        rhs_mat = rhs_mat[0:1]
        init_domain = init_domain[0]
        x = x[0:1]

    num_restarts = arguments.Config["attack"]["input_split_enhanced"]["pgd_restarts"]
    num_steps = arguments.Config["attack"]["input_split_enhanced"]["pgd_steps"]
    # pgd_steps is the same as the main pgd attack: before
    data_max, data_min = init_domain[..., 1], init_domain[..., 0]
    data_max = data_max.unsqueeze(0).unsqueeze(1)
    data_min = data_min.unsqueeze(0).unsqueeze(1)

    C_mat = C_mat.unsqueeze(0)

    if arguments.Config["attack"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split_enhanced"]["pgd_alpha"])
    # pack the domains as a large spec matrix

    cond_mat = [[C_mat.shape[1] for i in range(C_mat.shape[0])]]

    best_deltas, _ = pgd_attack_with_general_specs(
        model_ori,
        x,
        data_min,
        data_max,
        C_mat,
        rhs_mat,
        cond_mat,
        same_number_const=True,
        alpha=alpha,
        set_num_restarts=num_restarts,
        set_pgd_steps=num_steps,
    )
    attack_image, attack_output, attack_margin = gen_adv_example(
        model_ori, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat
    )

    if test_conditions(
        attack_image.unsqueeze(1),
        attack_output,
        C_mat,
        rhs_mat,
        cond_mat,
        True,
        data_max,
        data_min,
    ).all():
        print("pgd attack succeed in massive attack")
        return attack_margin, True
    else:
        return None, False


def initial_verify_criterion(lbs, rhs):
    """check whether verify successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    verified_idx = torch.any(
        (lbs - rhs) > 0, dim=-1
    )  # return bolling results in x's batch-wise
    if verified_idx.all():  # check whether all x verified
        print("Verified by initial bound!")
        return True, torch.where(verified_idx == 0)[0]
    else:
        return False, torch.where(verified_idx == 0)[0]


def check_adv(domains, model_ori):
    """check whether exiting domains have adv example or not. By naively forward inputs' lower and upper bound."""
    device = list(model_ori.parameters())[0].device
    worst_indices = domains.get_topk_indices(k=min(10, len(domains)))
    best_idx = domains.get_topk_indices(largest=True).item()
    indices = list(worst_indices.numpy()) + [best_idx]

    dm_l, dm_u, c, threshold = [], [], [], []
    for idx in indices:
        val = domains[idx]
        dm_l.append(val[1][None, ...].detach().cpu())
        dm_u.append(val[2][None, ...].detach().cpu())
        c.append(val[3][None, ...].detach().cpu())
        threshold.append(val[4].detach().cpu())

    adv_example = torch.cat(
        [torch.cat([dm_l[i], dm_u[i]]) for i in range(len(worst_indices))]
    )
    adv_example = torch.cat([adv_example, dm_l[-1], dm_u[-1]])
    adv_example = adv_example.unsqueeze(0).to(device, non_blocking=True)
    # [num_or, input_shape]

    prop_mat = torch.cat([torch.cat([c[i], c[i]]) for i in range(len(worst_indices))])
    prop_mat = torch.cat([prop_mat, c[-1], c[-1]]).to(device, non_blocking=True)
    # [num_or, num_and, output_dim]

    prop_rhs = [threshold[i] for i in range(len(worst_indices))]
    prop_rhs.append(threshold[-1])
    prop_rhs = torch.stack(prop_rhs).repeat_interleave(2, dim=0)
    # [num_or, num_and]

    cond_mat = [[prop_mat.shape[1] for i in range(prop_mat.shape[0])]]
    # [1, num_or, input_shape]
    prop_mat = prop_mat.view(1, -1, prop_mat.shape[-1])
    # [1, num_spec, output_dim]
    prop_rhs = prop_rhs.view(1, -1).to(device, non_blocking=True)
    # [1, num_spec]

    data_max = torch.cat(
        [torch.cat([dm_u[i], dm_u[i]]) for i in range(len(worst_indices))]
    )
    data_max = torch.cat([data_max, dm_u[-1], dm_u[-1]])
    data_max = data_max.unsqueeze(0).to(device, non_blocking=True)

    data_min = torch.cat(
        [torch.cat([dm_l[i], dm_l[i]]) for i in range(len(worst_indices))]
    )
    data_min = torch.cat([data_min, dm_l[-1], dm_l[-1]])
    data_min = data_min.unsqueeze(0).to(device, non_blocking=True)

    alpha = (data_max - data_min).max() / 4

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]
    num_restarts = arguments.Config["attack"]["input_split_check_adv"]["pgd_restarts"]

    if arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"])

    data_min = data_min.to(device)
    data_max = data_max.to(device)
    prop_rhs = prop_rhs.to(device)
    prop_mat = prop_mat.to(device)

    best_deltas, _ = pgd_attack_with_general_specs(model_ori, adv_example, data_min, data_max, prop_mat, prop_rhs, cond_mat, same_number_const=True, alpha=alpha, set_pgd_steps=pgd_steps, only_replicate_restarts=True)

    attack_image = best_deltas + adv_example.squeeze(1)
    attack_image = torch.min(torch.max(attack_image, data_min), data_max)

    attack_output = model_ori(attack_image.view(-1, *attack_image.shape[2:])).view(
        *attack_image.shape[:2], -1
    )
    # [1, num_or_spec, output_dim]

    if test_conditions(
        attack_image.unsqueeze(1),
        attack_output.unsqueeze(1),
        prop_mat.unsqueeze(1),
        prop_rhs,
        cond_mat,
        True,
        data_max,
        data_min,
    ).all():
        print("pgd attack succeed in check_adv")
        return None, True

    return None, False
