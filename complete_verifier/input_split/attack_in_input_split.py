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
import torch
import arguments
from attack import (
            pgd_attack_with_general_specs, PGDAttackResult, 
            check_and_save_cex)
from load_model import Customized
from utils import unpad_to_list_of_tensors


def pgd_attack_on_domains(model_ori, domains, stage, vnnlib):
    # dm_l, dm_u: [num_domains, *input_shape]
    # c: [num_domains, max_num_and, output_dim], rhs: [num_domains, max_num_and]
    # spec_size: [num_and_1, num_and_2, num_and_3, ...], with the length of num_domains
    if stage == 'init':
        # Use all domains
        attack_config = arguments.Config["attack"]["input_split"]
        _, dm_l, dm_u, c, rhs, spec_size = domains[:len(domains)]
    elif stage == 'bab':
        # Use only the worst domains
        max_domains = arguments.Config["attack"]["input_split_check_adv"]["max_num_domains"]
        worst_indices = domains.get_topk_indices(k=min(max_domains, len(domains)))
        indices = torch.cat([worst_indices, torch.tensor([len(domains) - 1], device=worst_indices.device)])
        attack_config = arguments.Config["attack"]["input_split_check_adv"]
        _, dm_l, dm_u, c, rhs, spec_size = domains[indices]
    else:
        raise ValueError(f"Invalid stage '{stage}' for PGD attack")

    # [1, num_domains, input_shape]
    dm_l = dm_l.unsqueeze(0)
    dm_u = dm_u.unsqueeze(0)
    center_x = (dm_l + dm_u) / 2

    same_spec_size = (spec_size == spec_size[0]).all()
    if same_spec_size:
        # All domains have the same number of AND specifications
        # c: [1, num_and * num_domains, output_dim], rhs: [1, num_and * num_domains]
        c = c.view(1, -1, c.shape[-1])
        rhs = rhs.view(1, -1)
    else:
        # Different domains may have different number of AND specifications
        # c: [1, num_and_1 + num_and_2 + ..., output_dim], rhs: [1, num_and_1 + num_and_2 + ...]
        c = torch.cat(unpad_to_list_of_tensors(c, 0, 1, spec_size, True), dim=1)
        rhs = torch.cat(unpad_to_list_of_tensors(rhs, 0, 1, spec_size, True), dim=1)

    alpha = (dm_u - dm_l).max() / 8 if attack_config["pgd_alpha"] == "auto" else float(attack_config["pgd_alpha"])
    pgd_steps = attack_config["pgd_steps"]
    num_restarts = attack_config["pgd_restarts"]

    ret: PGDAttackResult = pgd_attack_with_general_specs(
        model_ori, center_x, dm_l, dm_u, c, rhs, spec_size,
        alpha=alpha, pgd_steps=pgd_steps, num_restarts=num_restarts
    )

    adv_input, adv_output = ret.adv_input_per_or, ret.adv_output_per_or
    # [1, num_domains, *input_shape], [1, num_domains, output_dim]
    attack_success, best_or_idx = ret.attack_success, ret.best_or_idx
    # [1], [1]

    if attack_success.all():
        print(f"pgd attack succeed in stage {stage}, with index:", best_or_idx)
        _, verified_success = check_and_save_cex(
            adv_input[:, best_or_idx].squeeze(1),
            adv_output[:, best_or_idx].squeeze(1),
            vnnlib, arguments.Config["attack"]["cex_path"], "unsafe"
        )
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

    data_max = x_U.unsqueeze(0)
    data_min = x_L.unsqueeze(0)
    x = (data_min + data_max) / 2
    # [1, num_domains, *input_shape]

    spec_size = torch.full([cs.shape[0]], cs.shape[1], dtype=torch.int64, device=device)
    C_mat = cs.view(1, -1, cs.shape[-1])
    # [1, num_domains * num_and, output_dim]
    rhs_mat = rhs.view(1, -1)
    # [1, num_domains * num_and]

    alpha = (data_max - data_min).max() / 8

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]
    ret: PGDAttackResult = pgd_attack_with_general_specs(
        model_ori, x, data_min, data_max, C_mat, rhs_mat,
        spec_size, alpha=alpha, pgd_steps=pgd_steps
        )

    adv_input, adv_output = ret.adv_input_per_or, ret.adv_output_per_or
    # [1, num_domains, *input_shape], [1, num_domains, output_dim]
    adv_output = adv_output.view(cs.shape[0], -1, 1)
    # [num_domains, output_dim, 1]
    upper_bound = cs.matmul(adv_output).squeeze(-1)
    # [num_domains, num_spec], 

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
