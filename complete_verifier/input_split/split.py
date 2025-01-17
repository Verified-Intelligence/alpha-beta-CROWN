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
import math
import torch
from torch import Tensor
from typing import Union, Tuple

import arguments


@torch.no_grad()
def input_split_parallel(
        x_L: Tensor,
        x_U: Tensor,
        shape: tuple = None,
        cs: Union[Tensor, None] = None,
        thresholds: Union[Tensor, None] = None,
        dm_lb: Union[Tensor, None] = None,
        lA: Union[Tensor, None] = None,
        lbias: Union[Tensor, None] = None,
        split_depth: int = 1,
        i_idx: Union[Tensor, None] = None,
        split_partitions: int = 2,
        split_hint: Union[list[float], None] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, Union[Tensor, None], Union[Tensor, int]]:
    """
    Split the x_L and x_U given split_idx and split_depth.
    @param x_L:                 The lower bound on the inputs of the subdomains
    @param x_U:                 The upper bound on the inputs of the subdomains
    @param shape:
    @param cs:
    @param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    @param split_depth:         How many splits we wish to consider for all subdomains where split_depth <= input_dim
    @param i_idx:               Input indices to split on for each batch
    @param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    @param split_hint:          Only valid when split_partitions=2, the domains get split at split_hint rather than the
                                midpoint
                                (lb + ub)/2. This is beneficial when clipping domains.
    @return:
    """
    # FIXME: this function should not be in this file.
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    x_L_cp = x_L.clone()
    x_U_cp = x_U.clone()

    split_depth = min(split_depth, i_idx.size(1))
    remaining_depth = split_depth
    input_dim = x_L.shape[1]

    if split_hint is not None:
        assert split_partitions == 2, "Can only handle split_hint with split_partitions==2"
        # convert split_hint to a tensor with length input_dim
        if len(split_hint) == 1:
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype).expand(input_dim)
        else:
            assert len(split_hint) == input_dim, f"split_dim has dimension {len(split_hint)} when input_dim is {input_dim}"
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype)

    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(x_L_cp.shape[0])
            copy_num = x_L_cp.shape[0]//x_L.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            has_crossing = None
            if split_hint is not None:
                # find dimensions that have a crossing at split_hint
                has_crossing = torch.logical_and(x_L_cp[indices, idx] < split_hint[idx], x_U_cp[indices, idx] > split_hint[idx])

            x_L_cp_list, x_U_cp_list = [], []
            for partition in range(split_partitions):
                x_L_cp_tmp = x_L_cp.clone()
                x_U_cp_tmp = x_U_cp.clone()

                lrange = ((partition + 1) * x_L_cp[indices, idx] +
                          (split_partitions - partition - 1) * x_U_cp[indices, idx]) / split_partitions
                urange = (partition * x_L_cp[indices, idx] +
                          (split_partitions - partition) * x_U_cp[indices, idx]) / split_partitions

                if split_hint is not None:
                    if partition == 0:
                        # creating upper subdomains with split_hint
                        x_L_cp_tmp[indices, idx] = torch.where(has_crossing, split_hint[idx], lrange)
                        x_U_cp_tmp[indices, idx] = urange
                    else:
                        # creating lower subdomains with split_hint
                        x_L_cp_tmp[indices, idx] = lrange
                        x_U_cp_tmp[indices, idx] = torch.where(has_crossing, split_hint[idx], urange)
                else:
                    x_L_cp_tmp[indices, idx] = lrange
                    x_U_cp_tmp[indices, idx] = urange

                x_L_cp_list.append(x_L_cp_tmp)
                x_U_cp_list.append(x_U_cp_tmp)

            x_L_cp = torch.cat(x_L_cp_list)
            x_U_cp = torch.cat(x_U_cp_list)

        remaining_depth -= min(input_dim, remaining_depth)

    split_depth = split_depth - remaining_depth

    new_x_L = x_L_cp.reshape(-1, *shape[1:])
    new_x_U = x_U_cp.reshape(-1, *shape[1:])

    new_batch_size = split_partitions ** split_depth
    if cs is not None:
        cs_shape = [new_batch_size] + [1] * (len(cs.shape) - 1)
        cs = cs.repeat(*cs_shape)
    if thresholds is not None:
        thresholds = thresholds.repeat(new_batch_size, 1)
    if dm_lb is not None:
        dm_lb = dm_lb.repeat(new_batch_size, 1)
    if lA is not None:
        prev_shape = lA.shape
        # flattens shape to (batch, spec, dim_in)
        lA = lA.flatten(2)
        lA = lA.repeat(new_batch_size, 1, 1)
        # transforms back to original shape with extended batch dimension
        lA = lA.reshape(prev_shape[0] * new_batch_size, *prev_shape[1:])
    if lbias is not None:
        lbias = lbias.repeat(new_batch_size, 1)

    return new_x_L, new_x_U, cs, thresholds, split_depth, dm_lb, lA, lbias


def get_split_depth(x_L, split_partitions=2):
    split_depth = 1
    min_batch_size_ratio = arguments.Config["solver"]["min_batch_size_ratio"]
    batch_size = arguments.Config["solver"]["batch_size"]
    if len(x_L) < min_batch_size_ratio * batch_size:
        min_batch_size = min_batch_size_ratio * batch_size
        split_depth = int(math.log(min_batch_size//len(x_L))//math.log(split_partitions))
        split_depth = max(split_depth, 1)
    return split_depth
