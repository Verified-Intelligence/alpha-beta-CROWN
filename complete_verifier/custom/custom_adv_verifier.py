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
import torch
from attack import default_adv_verifier


def prepare_gtrsb_onnxruntime_counterexample(attack_image, attack_output):
    """Convert an internal GTSRB counterexample to original ONNX format."""
    attack_image = attack_image.permute(0, 2, 3, 1).contiguous()
    attack_output = torch.nn.functional.softmax(attack_output[0:1, :], dim=1)
    return attack_image, attack_output


def customized_gtrsb_adv_verifier(
        attack_image, attack_output, vnnlib, verify_with_onnxruntime):
    ori_shape = attack_image.shape
    attack_image, attack_output = prepare_gtrsb_onnxruntime_counterexample(
        attack_image, attack_output)
    if vnnlib is not None:
        # reversing format transform in custom_gtrsb_loader to adjust to permuted attack_image
        vnnlib[0] = (torch.tensor(vnnlib[0][0]).reshape(*ori_shape[1:], 2).permute(1, 2, 0, 3).reshape(-1, 2).tolist(), vnnlib[0][1])
    return default_adv_verifier(
        attack_image, attack_output, vnnlib, verify_with_onnxruntime)
