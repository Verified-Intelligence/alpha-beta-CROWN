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
import torch
import arguments
from attack.attack_pgd import (
    pgd_attack_with_general_specs, default_adv_example_finalizer, test_conditions, check_and_save_cex,
    process_vnn_lib_attack, build_conditions)
from load_model import Customized


def attack_in_input_bab_parallel(model_ori, domains, x, vnnlib=None):
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

    cond_mat = [[Cs.shape[1]]*Cs.shape[0]]
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

    best_deltas = pgd_attack_with_general_specs(
        model_ori, attack_x, lbs, ubs, Cs, rhs, cond_mat,
        same_number_const=True, alpha=alpha,
        pgd_steps=num_steps, num_restarts=num_restarts)[0]
    attack_image, attack_output, _ = eval(
        arguments.Config["attack"]["adv_example_finalizer"]
    )(model_ori, attack_x, best_deltas, ubs, lbs, Cs, rhs, cond_mat)

    res, idx = test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1),
                           Cs, rhs, cond_mat, True, ubs, lbs, return_success_idx=True)
    if res.all():
        print("pgd attack succeed in input bab parallel, with idx:", idx)
        _, verified_success = check_and_save_cex(attack_image[:, idx], attack_output[:, idx], vnnlib,
                                                 arguments.Config["attack"]["cex_path"], "unsafe")
                
        return verified_success
        
    return False


def massive_pgd_attack(x, model_ori, vnnlib=None):
    """pgd attack with very large number of random starts
    init_domain: [input_shape, 2]
    rhs_mat: [num_or(1), num_and]
    x: [batch(1), input_shape]
    C_mat: [num_and, out_dim]
    """

    num_restarts = arguments.Config["attack"]["input_split_enhanced"]["pgd_restarts"]
    num_steps = arguments.Config["attack"]["input_split_enhanced"]["pgd_steps"]

    list_target_label_arrays, data_min, data_max = process_vnn_lib_attack(vnnlib, x)
    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)
    data_min = data_min[:, :len(list_target_label_arrays[0]), ...]
    data_max = data_max[:, :len(list_target_label_arrays[0]), ...]

    if arguments.Config["attack"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split_enhanced"]["pgd_alpha"])

    best_deltas = pgd_attack_with_general_specs(
        model_ori, x, data_min, data_max, C_mat, rhs_mat, cond_mat,
        same_number_const=True, alpha=alpha, num_restarts=num_restarts,
        pgd_steps=num_steps,
    )[0]

    attack_image, attack_output, attack_margin = eval(arguments.Config["attack"]["adv_example_finalizer"])(
        model_ori, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat)
     
    if test_conditions(attack_image.unsqueeze(1), attack_output,
                       C_mat, rhs_mat, cond_mat, True, data_max, data_min).all():
        print("pgd attack succeed in massive attack")
        # attack_image has shape (batch, specs, c, h, w)
        _, verified_success = check_and_save_cex(attack_image[:, 0:1].squeeze(1), attack_output[:, 0:1].squeeze(1),
                                                 vnnlib, arguments.Config["attack"]["cex_path"], "unsafe")
        return attack_margin, verified_success
    return attack_margin, False


def check_adv(domains, model_ori, x, vnnlib=None):
    """check whether exiting domains have adv example or not.
    By using inputs' lower and upper bound as attack starting points."""
    if len(vnnlib) != 1:
        print('Multiple x in check_adv() is not supported so far!')
        return False

    device = x.device
    max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
    worst_indices = domains.get_topk_indices(k=min(max_num_domains, len(domains)))
    best_idx = domains.get_topk_indices(largest=True).item()
    indices = list(worst_indices.numpy()) + [best_idx]

    dm_l, dm_u, c, threshold = [], [], [], []
    for idx in indices:
        val = domains[idx]
        dm_l.append(val[1][None, ...].detach().cpu())
        dm_u.append(val[2][None, ...].detach().cpu())
        c.append(val[3][None, ...].detach().cpu())
        threshold.append(val[4].detach().cpu())

    # we pick the worst domains (smallest lower bounds) since they are less likely to be verified.
    # we use their input range: dm_l and dm_u as attacking starting points.
    starting_points = torch.cat([torch.cat([dm_l[i], dm_u[i]]) for i in range(len(worst_indices))])
    # we also include the most recent added domain to have a try.
    starting_points = torch.cat([starting_points, dm_l[-1], dm_u[-1]])
    starting_points = starting_points.unsqueeze(0).to(device, non_blocking=True)
    # [1, num_starting_points, *input_shape], num_starting_points = 2 * (worst_indices + 1)

    C_mat = torch.cat([torch.cat([c[i], c[i]]) for i in range(len(worst_indices))])
    C_mat = torch.cat([C_mat, c[-1], c[-1]]).to(device, non_blocking=True)
    # [num_starting_points, num_and, output_dim]

    rhs_mat = [threshold[i] for i in range(len(worst_indices))]
    rhs_mat.append(threshold[-1])
    rhs_mat = torch.stack(rhs_mat).repeat_interleave(2, dim=0)
    # [num_starting_points, num_and]

    # we need to manually construct condition/property/rhs matrix with the PGD random starts as num_starting_points
    cond_mat = [[C_mat.shape[1]] * C_mat.shape[0]]
    # list: [num_and, num_and, num_and ,...] with the length of num_starting_points
    prop_mat = C_mat.view(1, -1, C_mat.shape[-1])
    # [1, num_starting_points * num_and, output_dim]
    rhs_mat = rhs_mat.view(1, -1).to(device, non_blocking=True)
    # [1, num_starting_points * num_and]

    data_min = x.ptb.x_L.unsqueeze(1)
    data_max = x.ptb.x_U.unsqueeze(1)
    # [1, 1, *input_shape], all attack_images share the same data_min/max

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]

    if arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"])

    best_deltas = pgd_attack_with_general_specs(
        model_ori, starting_points, data_min, data_max, prop_mat, rhs_mat,
        cond_mat, same_number_const=True, alpha=alpha,
        pgd_steps=pgd_steps, only_replicate_restarts=True)[0]

    attack_image = best_deltas + starting_points.squeeze(1)
    attack_image = torch.min(torch.max(attack_image, data_min), data_max)
    # [1, num_starting_points, *input_shape]

    attack_output = model_ori(attack_image.view(-1, *attack_image.shape[2:])).view(
        *attack_image.shape[:2], -1)
    # [1, num_starting_points, output_dim]

    # in test_conditions() the attack_image and attack_output requires the shape:
    # [num_example, num_restarts, num_or_spec, *input_shape]
    # We currently don't have num_restarts dim, so we unsqueeze(1) for them.
    res, idx = test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1), prop_mat.unsqueeze(1), rhs_mat,
                               cond_mat, True, data_max.unsqueeze(1), data_min.unsqueeze(1), return_success_idx=True)
    if res.all():
        print("pgd attack succeed in check_adv, with idx:", idx)
        _, verified_success = check_and_save_cex(attack_image[:, idx], attack_output[:, idx], vnnlib,
                                                 arguments.Config["attack"]["cex_path"], "unsafe")
        return verified_success

    return False


def update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model_ori):
    device = x_L.device
    max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
    num_domains = min(max_num_domains, x_L.shape[0])
    print(f'Running PGD attack on {num_domains} domains')
    x_L = x_L[:num_domains]
    x_U = x_U[:num_domains]
    cs = cs[:num_domains]
    rhs = thresholds[:num_domains]
    dm_lb = dm_lb[:num_domains]

    adv_example = ((x_L + x_U) / 2).unsqueeze(0)
    cond_mat = [[cs.shape[1]] * cs.shape[0]]
    prop_mat = cs.view(1, -1, cs.shape[-1])
    prop_rhs = rhs.view(1, -1)
    data_max = x_U.unsqueeze(0)
    data_min = x_L.unsqueeze(0)
    alpha = (data_max - data_min).max() / 4

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]
    best_deltas = pgd_attack_with_general_specs(
        model_ori, adv_example, data_min, data_max, prop_mat, prop_rhs,
        cond_mat, same_number_const=True, alpha=alpha, pgd_steps=pgd_steps,
        only_replicate_restarts=True)[0]

    attack_image = best_deltas + adv_example.squeeze(1)
    attack_image = torch.min(torch.max(attack_image, data_min), data_max)
    attack_output = model_ori(attack_image.view(-1, *attack_image.shape[2:])).view(
        *attack_image.shape[:2], -1
    )
    attack_output = attack_output.view(cs.shape[0], -1, 1)
    upper_bound = cs.matmul(attack_output).squeeze(-1)
    print('Trying to update RHS with attack')
    print(f'  Current RHS: mean {rhs.mean().item()}')
    print(f'  New upper bound: mean {upper_bound.mean().item()}')
    print(f'  Number of updated RHS: {(upper_bound < rhs).sum()}/{rhs.numel()}')
    rhs = torch.min(rhs, upper_bound)
    thresholds[:num_domains] = rhs
    gap = rhs - dm_lb
    min_gap = gap.min()
    print('  Gap between lower/upper bounds: '
          f'mean {gap.mean().item()}, min {min_gap.item()}')
    assert min_gap >= -1e-3, 'Gap between lower and upper bounds is negative'

    return thresholds
