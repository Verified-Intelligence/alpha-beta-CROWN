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
import arguments
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def set_beta(self: 'LiRPANet', d, bias=True):
    # count how many split nodes in each batch example (batch, num of layers)
    splits_per_example = []
    max_splits_per_layer = {}
    batch = len(d['history'])
    for bi in range(batch):
        splits_per_example.append({})
        for k, v in d['history'][bi].items():
            # First element of layer_splits is a list of split neuron IDs.
            splits_per_example[bi][k] = len(v[0])
            max_splits_per_layer[k] = max(
                max_splits_per_layer.get(k, 0), splits_per_example[bi][k])

    # Create and load warmup beta.
    self.reset_beta(batch, max_splits_per_layer, betas=d['betas'], bias=bias)

    for node in self.split_nodes:
        if node.sparse_betas is None:
            continue
        sparse_betas = (node.sparse_betas
                        if isinstance(node.sparse_betas, list)
                        else node.sparse_betas.values())
        for sparse_beta in sparse_betas:
            sparse_beta.apply_splits(d['history'], node.name)

    return splits_per_example


def reset_beta(self: 'LiRPANet', batch, max_splits_per_layer, betas=None,
               bias=False):
    beta_crown_args = arguments.Config["solver"]["beta-crown"]
    enable_opt_interm_bounds = beta_crown_args['enable_opt_interm_bounds']
    for layer_name in max_splits_per_layer:
        layer = self.net[layer_name]
        if enable_opt_interm_bounds:
            start_nodes = []
            for act in self.split_activations[layer_name]:
                start_nodes.extend(list(act[0].alpha.keys()))
            start_nodes = list(set(start_nodes))
        else:
            start_nodes = None
        shape = (batch, max_splits_per_layer[layer_name])
        if betas is not None and betas[0] is not None and layer_name in betas[0]:
            betas_ = [(betas[bi][layer_name] if betas[bi] is not None else None)
                      for bi in range(batch)]
        else:
            betas_ = [None for _ in range(batch)]
        self.net.reset_beta(layer, shape, betas_, bias=bias,
                            start_nodes=start_nodes)


def get_beta(self: 'LiRPANet', splits_per_example, device=None):
    # split_per_example only has half of the examples.
    beta_crown_args = arguments.Config["solver"]["beta-crown"]
    enable_opt_interm_bounds = beta_crown_args['enable_opt_interm_bounds']
    ret = []
    betas_cpu = {}
    # Setting non_blocking to False ensures that data is fully transferred from the GPU to the CPU before proceeding. 
    for k in splits_per_example[0]:
        if not enable_opt_interm_bounds:
            betas_cpu[k] = self._transfer(
                self.net[k].sparse_betas[0].val, device, non_blocking=False)
        else:
            betas_cpu[k] = []
            for sparse_beta in self.net[k].sparse_betas.values():
                betas_cpu[k].append(self._transfer(
                    sparse_beta.val, device, non_blocking=False))
    for i in range(len(splits_per_example)):
        betas = {}
        for k in splits_per_example[i]:
            if not enable_opt_interm_bounds:
                betas[k] = betas_cpu[k][i, :splits_per_example[i][k]]
            else:
                betas[k] = []
                for item in betas_cpu[k]:
                    betas[k].append(item[i, :splits_per_example[i][k]])
        ret.append(betas)
    return ret
