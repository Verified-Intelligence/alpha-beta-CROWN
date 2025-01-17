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
from complete_verifier.attack.attack_pgd import default_adv_verifier


def customized_gtrsb_adv_verifier(attack_image, attack_output, vnnlib, check_output):
    ori_shape = attack_image.shape
    # make sure to call contiguous otherwize view(-1) may fail because of non-contiguous space
    attack_image = attack_image.permute(0, 2, 3, 1).contiguous() 
    attack_output = torch.nn.functional.softmax(attack_output[0:1, :], dim=1)
    if vnnlib is not None:
        # reversing format transform in custom_gtrsb_loader to adjust to permuted attack_image
        vnnlib[0] = (torch.tensor(vnnlib[0][0]).reshape(*ori_shape[1:], 2).permute(1, 2, 0, 3).reshape(-1, 2).tolist(), vnnlib[0][1])
    return default_adv_verifier(attack_image, attack_output, vnnlib, check_output)