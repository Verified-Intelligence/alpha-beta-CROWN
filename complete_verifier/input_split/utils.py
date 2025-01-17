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


def initial_verify_criterion(lbs, rhs):
    """check whether verify successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    verified_idx = torch.any(
        (lbs - rhs) > 0, dim=-1
    )  # return bolling results in x's batch-wise
    if verified_idx.all():  # check whether all x verified
        print("Verified by initial bound!")
        return True, torch.where(verified_idx == 0)[0]
    else:
        return False, torch.where(verified_idx == 0)[0]


def transpose_c_back(lA, global_lb, rhs, dm_l, dm_u, ret, net):
    # Here, we transpose c matrix shape back from [1, spec_dim, ...] to [spec_dim, 1, ...],
    # so we should recover lA, lb, x_LB, x_UB, alphas as if they are computed with c shape [spec_dim, 1, ...],
    # to prepare for input domain bab.
    # More info can be found in function beginning comment.
    lA = lA.transpose(0, 1)
    global_lb = global_lb.transpose(0, 1)
    rhs = rhs.transpose(0, 1)
    net.c = net.c.transpose(0, 1)
    dm_l = dm_l.expand([net.c.shape[0]] + list(dm_l.shape[1:]))
    dm_u = dm_u.expand([net.c.shape[0]] + list(dm_u.shape[1:]))
    for start_node in ret['alphas']:
        for end_node in ret['alphas'][start_node]:
            if end_node == net.final_name:
                ret['alphas'][start_node][end_node] = ret[
                    'alphas'][start_node][end_node].transpose(1, 2)
            else:
                new_shape = list([
                    1 for _ in ret['alphas'][start_node][end_node].shape])
                assert ret['alphas'][start_node][end_node].shape[2] == 1
                new_shape[2] = net.c.shape[0]
                ret['alphas'][start_node][end_node] = ret[
                    'alphas'][start_node][end_node].repeat(new_shape)
    return lA, global_lb, rhs, dm_l, dm_u
