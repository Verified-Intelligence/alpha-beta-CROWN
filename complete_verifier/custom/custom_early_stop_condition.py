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
from attack import default_early_stop_condition


def customized_gtrsb_condition(inputs, output, data_min, data_max, 
                                C_mat, rhs_mat, or_spec_size,
                                model, return_best_idx=False):
    # inputs: [batch, 1, 1, *input_shape], output: [batch, 1, 1, output_dim]
    assert inputs.shape[:3] == output.shape[:3] == data_min.shape[:3] == data_max.shape[:3]
    assert inputs.shape[0] == 1 and inputs.shape[1] == 1 and inputs.shape[2] == 1
    # we recalculate the output of the model in gtrsb for numerical stability
    output = model(inputs.view(-1, *inputs.shape[3:])).unsqueeze(1).unsqueeze(1)
    return default_early_stop_condition(inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size, model, return_best_idx)