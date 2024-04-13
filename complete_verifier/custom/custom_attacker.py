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
from beta_CROWN_solver import LiRPANet
from attack.attack_pgd import attack_with_general_specs

def use_LiRPANet(model, x, data_min, data_max,
                    list_target_label_arrays,
                    initialization="uniform", GAMA_loss=False):
    wrapped_model = LiRPANet(model, in_size=x.shape)

    model_layers = wrapped_model.net._modules.keys()
    signmerge_layers = []
    for layer_id in model_layers:
        if "/merge" in layer_id:
            signmerge_layers.append(layer_id)

    # For the smallest model, we tend to use the loose approximation
    loose_approx = True if len(signmerge_layers) == 2 else False

    num_attacks = 40
    for _ in range(num_attacks):
        if loose_approx:
            for layer_id in signmerge_layers:
                wrapped_model.net[layer_id].signmergefunction = wrapped_model.net[layer_id].loose_function
        else:
            for layer_id in signmerge_layers:
                wrapped_model.net[layer_id].signmergefunction = wrapped_model.net[layer_id].tight_function

        res, attack_image, attack_margin, all_adv_candidates = attack_with_general_specs(wrapped_model.net, x, data_min, data_max,
                        list_target_label_arrays,
                        initialization, GAMA_loss)

        if res:
            break

        loose_approx = not loose_approx


    return res, attack_image, attack_margin, all_adv_candidates