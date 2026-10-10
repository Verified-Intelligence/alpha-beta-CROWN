#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
import time

from beta_CROWN_solver import LiRPANet
from attack import (
    attack_with_general_specs, verify_onnxruntime_output_matches_pytorch)
import arguments
from load_model import Customized
from custom.custom_adv_verifier import prepare_gtrsb_onnxruntime_counterexample


def use_LiRPANet(model, x, data_min, data_max, C_mat, rhs_mat, or_spec_size,
                    initialization="uniform", GAMA_loss=False, vnnlib=None):
    wrapped_model = LiRPANet(model, in_size=x.shape)

    model_layers = wrapped_model.net._modules.keys()
    signmerge_layers = []
    for layer_id in model_layers:
        if "/merge" in layer_id:
            signmerge_layers.append(layer_id)

    # For the smallest model, we tend to use the loose approximation
    loose_approx = True if len(signmerge_layers) == 2 else False

    min_attacks = 40
    timeout_buffer = 30.0
    deadline = time.time() + max(
        float(arguments.Config["bab"]["timeout"]) - timeout_buffer, 0.0)
    attack_count = 0
    while attack_count < min_attacks or time.time() < deadline:
        attack_count += 1
        if loose_approx:
            for layer_id in signmerge_layers:
                wrapped_model.net[layer_id].signmergefunction = wrapped_model.net[layer_id].loose_function
        else:
            for layer_id in signmerge_layers:
                wrapped_model.net[layer_id].signmergefunction = wrapped_model.net[layer_id].tight_function

        res, idx, attack_image, attack_output, attack_margin, all_adv_candidates = attack_with_general_specs(
            wrapped_model.net, x, data_min, data_max,
            C_mat, rhs_mat, or_spec_size,
            initialization, GAMA_loss
            )

        if res:
            res_comparison = True
            if arguments.Config['general']['verify_onnxruntime_output']:
                onnx_attack_image, onnx_attack_output = (
                    prepare_gtrsb_onnxruntime_counterexample(
                        attack_image[idx], attack_output[idx]))
                res_comparison = verify_onnxruntime_output_matches_pytorch(
                    onnx_attack_image, onnx_attack_output)
            if not res_comparison:
                res = False

            break

        loose_approx = not loose_approx

    print(f"GTRSB attack attempts: {attack_count}")
    return res, idx, attack_image, attack_output, attack_margin, all_adv_candidates
