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
import time

import arguments
from load_model import Customized
from auto_LiRPA import BoundedTensor
from attack.attack_utils import (default_adv_saver, default_adv_verifier,
                                process_data_for_attack, check_and_save_cex)


def attack(model_ori, x: BoundedTensor, c: torch.Tensor, rhs: torch.Tensor, or_spec_size: torch.Tensor, 
           vnnlib, verified_status, verified_success):
    
    attack_config = arguments.Config["attack"]
    initialization = 'uniform'
    GAMA_loss = False
    if "diversed" in attack_config["attack_mode"]:
        initialization = "osi"
    if "GAMA" in attack_config["attack_mode"]:
        GAMA_loss = True

    # format data
    x, data_min, data_max, c, rhs = process_data_for_attack(x, c, rhs, or_spec_size)

    attack_function = eval(attack_config["attack_func"])
    attack_ret, best_or_idx, adv_input, adv_output, adv_margins, all_adv_candidates = attack_function(
        model_ori, x, data_min, data_max, c, rhs, or_spec_size,
        initialization=initialization, GAMA_loss=GAMA_loss)
    # attack_ret, best_or_idx: [1].
    # adv_input shape: [num_or, *input_shape], adv_output shape: [num_or, num_output]
    # adv_margins shape: [num_spec]

    if attack_ret:
        # if attack is successful, we compare adv example of the best or spec
        # on pytorch and onnxruntime after attack and before saving
        # [1, *input_shape], [1, num_output]
        verified_status, verified_success = check_and_save_cex(adv_input[best_or_idx],
                                                                adv_output[best_or_idx],
                                                                vnnlib,
                                                                arguments.Config["attack"]["cex_path"], "unsafe-pgd")

    print("verified_status", verified_status)
    print("verified_success", verified_success)

    return verified_status, verified_success, adv_input, adv_margins, all_adv_candidates


def attack_with_general_specs(model, x, data_min, data_max, C_mat, rhs_mat, or_spec_size,
                              initialization="uniform", GAMA_loss=False):
    r""" Interface to PGD attack.

        :param model (torch.nn.Module): PyTorch model under attack.
        :param x: Input (x_0).[num_or, *x_shape]
        :param data_min: Lower bounds of data input. (e.g., 0 for mnist)
                shape: [num_or, *input_shape]
        :param data_max: Upper bounds of data input. (e.g., 1 for mnist)
                shape: [num_or, *input_shape]
        :param C_mat: [num_spec, num_output]
        :param rhs: [num_spec]
        :param or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or.
        :param initialization (string): initialization of PGD attack, chosen from 'uniform' and 'osi'
        :param GAMA_loss (boolean): whether to use GAMA (Guided adversarial attack) loss in PGD attack
    """
    attack_start_time = time.time()
    assert arguments.Config["specification"]["norm"] == float("inf"), print('We only support Linf-norm attack.')

    attack_config = arguments.Config["attack"]
    general_attack = attack_config["general_attack"]

    # Step 1: Import attack functions based on version
    # Now we have use the new version of general attack by default.
    # We keep both versions for compatibility and debugging purposes.
    if general_attack:
        from attack.general_spec_attack import (
            pgd_attack_with_general_specs,
            test_conditions,
            default_adv_example_finalizer,
        )
    else:
        from attack.attack_pgd import (
            pgd_attack_with_general_specs,
            test_conditions,
            default_adv_example_finalizer,
        )

    # Step 2: Determine attack parameters
    use_adam = True
    alpha = attack_config["pgd_alpha"]
    alpha_scale = attack_config["pgd_alpha_scale"]
    if alpha_scale:
        # [num_or, *input_shape]
        alpha = (data_max - data_min) * float(alpha)
        use_adam = False
    else:
        if alpha == 'auto':
            max_eps = torch.max(data_max - data_min).item() / 2
            alpha = max_eps / 4
        else:
            alpha = float(alpha)
    num_restarts = arguments.Config["attack"]["pgd_restarts"]
    batch_size = arguments.Config["attack"]["pgd_batch_size"]

    # When BaB-attack is enabled, we only support 1 batch in the main attack loop.
    enable_bab_attack = arguments.Config['bab']['attack']['enabled']
    if enable_bab_attack:
        assert num_restarts <= batch_size

    print(f'Attack parameters: initialization={initialization}, steps={arguments.Config["attack"]["pgd_steps"]}, restarts={arguments.Config["attack"]["pgd_restarts"]}, alpha={alpha}, GAMA={GAMA_loss}')

    # Step 3: Set all model parameters without gradient
    grad_status = {}
    for p in model.parameters():
        grad_status[p] = p.requires_grad
        p.requires_grad_(False)

    # Step 4: Format data
    # [1, num_or, *input_shape]
    batched_x = x.unsqueeze(0)
    batched_data_min = data_min.unsqueeze(0)
    batched_data_max = data_max.unsqueeze(0)

    # [1, num_spec, num_output]
    batched_C_mat = C_mat.unsqueeze(0)
    # [1, num_spec]
    batched_rhs_mat = rhs_mat.unsqueeze(0)

    # Step 5: Check if the clean input is already adversarial
    # [1, num_or, *input_shape], [1, num_or, num_output], [1, num_spec], [1, num_or]
    adv_input_per_or, adv_output_per_or, adv_margin_per_spec, adv_margin_per_or = eval(arguments.Config["attack"]["adv_example_finalizer"])(
        model, batched_x, torch.zeros_like(batched_x), batched_data_max, batched_data_min, batched_C_mat, batched_rhs_mat, or_spec_size
    )

    # Unsqueeze data to match API requirements of different versions of test_conditions().
    # In the original version, the dim of restart is 1.
    #   [1, 1, num_or, *input_shape] for input, [1, 1, num_or, num_output] for output,
    # while in the general attack version, it is 2.
    #   [1, num_or, 1, *input_shape] for input, [1, num_or, 1, num_output] for output.
    restart_dim = 2 if general_attack else 1
    adv_input_test = adv_input_per_or.unsqueeze(restart_dim)
    data_min_test = batched_data_min.unsqueeze(restart_dim)
    data_max_test = batched_data_max.unsqueeze(restart_dim)
    adv_output_test = adv_output_per_or.unsqueeze(restart_dim)

    # attack_success: [1], best_or_idx: [1]
    attack_success, best_or_idx = test_conditions(
        adv_input_test, adv_output_test,
        data_min_test, data_max_test,
        batched_C_mat, batched_rhs_mat, or_spec_size,
        return_best_idx=True
    )

    # If the clean input is already adversarial, we skip the attack.
    if attack_success.all():
        print("Clean input adversarial, attack skipped.")
        return (attack_success, best_or_idx,
                adv_input_per_or.squeeze(0).detach(), adv_output_per_or.squeeze(0).detach(),
                adv_margin_per_or.squeeze(0).detach(), None)

    # Main attack loop
    total_batches = (num_restarts + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        # Step 6: Preform PGD attack by batches of restarts
        ret = pgd_attack_with_general_specs(
            model, batched_x, batched_data_min, batched_data_max,
            batched_C_mat, batched_rhs_mat, or_spec_size, alpha,
            initialization=initialization, GAMA_loss=GAMA_loss,
            use_adam=use_adam, num_restarts=min(batch_size, num_restarts)
        )

        # Step 7: Save the best adversarial examples
        update_mask_per_or = ret.adv_margin_per_or < adv_margin_per_or
        adv_input_per_or[update_mask_per_or] = ret.adv_input_per_or[update_mask_per_or]
        adv_output_per_or[update_mask_per_or] = ret.adv_output_per_or[update_mask_per_or]
        adv_margin_per_or[update_mask_per_or] = ret.adv_margin_per_or[update_mask_per_or]

        update_mask_per_spec = update_mask_per_or.repeat_interleave(or_spec_size, dim=-1)
        adv_margin_per_spec[update_mask_per_spec] = ret.adv_margin_per_spec[update_mask_per_spec]

        adv_margin_best, best_or_idx = adv_margin_per_or.min(dim=-1)

        # Step 8: Check if the attack is successful
        attack_success = adv_margin_best <= 0.0
        assert (attack_success == ret.attack_success).all(), "inconsistent attack success status"
        # If the attack is successful, we can stop early
        if attack_success.all():
            break

        # Decrease the remaining number of restarts
        num_restarts -= batch_size

    # Step 9: Restore grad status
    for p in model.parameters():
        p.requires_grad_(grad_status[p])

    attack_time = time.time() - attack_start_time
    print(f'Attack finished in {attack_time:.4f} seconds.')
    print("PGD attack succeeded!" if attack_success.all() else "PGD attack failed")

    # Unify shape of adv_input_all based on version
    if ret.adv_input_all is not None:
        adv_input_all = ret.adv_input_all.squeeze(0)
        if not general_attack:
            adv_input_all = adv_input_all.transpose(0, 1)
        adv_input_all = adv_input_all.detach()
    else:
        adv_input_all = None

    return (attack_success, best_or_idx,
            adv_input_per_or.squeeze(0).detach(), adv_output_per_or.squeeze(0).detach(),
            adv_margin_per_spec.squeeze(0).detach(), adv_input_all)
