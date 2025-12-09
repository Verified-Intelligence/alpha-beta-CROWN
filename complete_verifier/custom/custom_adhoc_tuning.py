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
import arguments


def customized_vit_tuning(model_ori, vnnlib_handler):
    num_matmul = len([item for item in model_ori.modules()
                    if 'MatMul' in str(type(item))])
    if num_matmul >= 6:
        print('Sharing alpha due to model size')
        arguments.Config['solver']['alpha-crown']['share_alphas'] = True


def customized_nn4sys_tuning(model_ori, vnnlib_handler):
    num_params = sum(p.numel() for p in model_ori.parameters() if p.requires_grad)
    num_specs = vnnlib_handler.all_specs.batch_size
    if num_params > 1e7 and num_specs > 1000:
        print(f"Shrinking build_batch_size on model with {num_params} params and {num_specs} specs")
        arguments.Config['solver']['build_batch_size'] = 1000


def customized_vggnet16_tuning(model_ori, vnnlib_handler):
    perturbed = (vnnlib_handler.data_max - vnnlib_handler.data_min > 0).sum()
    print('Number of perturbed inputs:', int(perturbed))
    if perturbed > 10000:
        print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
        print('Setting arguments.Config["attack"]["pgd_order"] to "before"')
        arguments.Config['attack']['pgd_order'] = 'before'
    if perturbed > 100:
        print('Setting bound_prop_method to crown')
        arguments.Config['solver']['bound_prop_method'] = 'crown'
