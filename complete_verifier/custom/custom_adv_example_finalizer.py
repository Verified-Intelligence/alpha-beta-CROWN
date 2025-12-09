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
from attack import default_adv_example_finalizer

def customized_gtrsb_adv_example_finalizer(model_ori, x, best_deltas, data_min, data_max, C_mat, rhs_mat, or_spec_size):
    # x, best_deltas, data_min and data_max have shape (batch, num_or, *input_shape).
    assert x.shape == best_deltas.shape == data_min.shape == data_max.shape
    assert x.shape[0] == C_mat.shape[0] == rhs_mat.shape[0] == 1, "batch size should be 1"
    num_or = or_spec_size.shape[0]
    assert C_mat.shape[1] == rhs_mat.shape[1] == num_or, "Only single AND in every OR is supported for gtrsb"

    # first, get the batched results over num_or dim.
    _, _, _, adv_margin_per_or = default_adv_example_finalizer(
        model_ori, x, best_deltas,data_min, data_max, C_mat, rhs_mat, or_spec_size
    )
    # [batch_size, num_or]
    
    # gtrsb has a lot of numerical issues, and is very sensitive to batch size
    # so we select the OR spec with the best (smallest) margin,
    # and re check satisfaction
    best_or_idx = adv_margin_per_or.argmin(dim=1)
    # [batch_size (1)]

    adv_example, adv_output, adv_margin, adv_margin_per_or = default_adv_example_finalizer(
        model_ori, x[:, best_or_idx], best_deltas[:, best_or_idx], 
        data_min[:, best_or_idx], data_max[:, best_or_idx],
        C_mat[:, best_or_idx], rhs_mat[:, best_or_idx], or_spec_size[best_or_idx]
    )
    # [batch_size, 1, *input_shape], [batch_size, 1, output_dim], [batch_size, 1]

    # expand the results to align with the default adv_example_finalizer API
    adv_example = adv_example.expand(-1, num_or, *x.shape[2:])
    adv_output = adv_output.expand(-1, num_or, -1)
    adv_margin = adv_margin.expand(-1, num_or)
    adv_margin_per_or = adv_margin_per_or.expand(-1, num_or)

    # [batch_size, num_or, *input_shape], [batch_size, num_or, output_dim], [batch_size, num_or]
    return adv_example, adv_output, adv_margin, adv_margin_per_or
