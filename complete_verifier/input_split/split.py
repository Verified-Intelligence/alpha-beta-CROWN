import math
import torch

import arguments


@torch.no_grad()
def input_split_parallel(x_L, x_U, shape=None,
                         cs=None, thresholds=None, split_depth=1, i_idx=None,
                         split_partitions=2):
    """
    Split the x_L and x_U given split_idx and split_depth.
    """
    # FIXME: this function should not be in this file.
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    x_L_cp = x_L.clone()
    x_U_cp = x_U.clone()

    split_depth = min(split_depth, i_idx.size(1))
    remaining_depth = split_depth
    input_dim = x_L.shape[1]
    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(x_L_cp.shape[0])
            copy_num = x_L_cp.shape[0]//x_L.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            x_L_cp_list, x_U_cp_list = [], []
            for partition in range(split_partitions):
                x_L_cp_tmp = x_L_cp.clone()
                x_U_cp_tmp = x_U_cp.clone()

                lrange = ((partition + 1) * x_L_cp[indices, idx] +
                          (split_partitions - partition - 1) * x_U_cp[indices, idx]) / split_partitions
                urange = (partition * x_L_cp[indices, idx] +
                          (split_partitions - partition) * x_U_cp[indices, idx]) / split_partitions

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

    if cs is not None:
        cs_shape = [split_partitions ** split_depth] + [1] * (len(cs.shape) - 1)
        cs = cs.repeat(*cs_shape)
    if thresholds is not None:
        thresholds = thresholds.repeat(split_partitions ** split_depth, 1)
    split_idx = i_idx.repeat(split_partitions ** split_depth, 1)
    return new_x_L, new_x_U, cs, thresholds, split_depth, split_idx


def get_split_depth(x_L, split_partitions=2):
    split_depth = 1
    min_batch_size_ratio = arguments.Config["solver"]["min_batch_size_ratio"]
    batch_size = arguments.Config["solver"]["batch_size"]
    if len(x_L) < min_batch_size_ratio * batch_size:
        min_batch_size = min_batch_size_ratio * batch_size
        split_depth = int(math.log(min_batch_size//len(x_L))//math.log(split_partitions))
        split_depth = max(split_depth, 1)
    return split_depth
