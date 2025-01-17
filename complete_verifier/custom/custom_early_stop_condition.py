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
from attack.attack_pgd import test_conditions


def customized_gtrsb_condition(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const,
    data_max, data_min, model, indices, num_or_spec, return_success_idx=False):
    # condition based on base size 1

    test_input = inputs[:, indices.item() // num_or_spec, indices.item() % num_or_spec, :]
    test_output = model(test_input)
    test_input = test_input.unsqueeze(0).unsqueeze(0)
    test_output = test_output.unsqueeze(0).unsqueeze(0)
    return test_conditions(test_input, test_output, C_mat, rhs_mat, cond_mat, same_number_const,
        data_max, data_min, return_success_idx)