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
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedTensor
from auto_LiRPA.bound_ops import BoundConstant, BoundParams, BoundBuffers, BoundInput


def precompute_A(net, A, x, interm_bounds):
    # TODO maybe some can be pruned
    need_A = set()
    for nodes in net.split_activations.values():
        for node in nodes:
            for inp in node[0].inputs:
                if inp.name not in net.root_names and inp.perturbed:
                    need_A.add(inp)
    for node in need_A:
        if node.name not in A:
            print(f'Missing A for {node}. Making an additional CROWN call.')
            batch_size = node.output_shape[0]
            dim_output = int(np.prod(node.output_shape[1:]))
            C = torch.eye(dim_output, device=net.device).unsqueeze(0).expand(batch_size, -1, -1)
            ret = net.compute_bounds(
                x=(x,), C=C, method='CROWN', final_node_name=node.name,
                return_A=True,
                needed_A_dict={node.name: [net.input_name[0]]},
                interm_bounds=interm_bounds)
            A.update(ret[-1])


def set_roots(roots, x, A):
    assert isinstance(roots[0], BoundInput)
    assert isinstance(roots[0].value, BoundedTensor)
    roots[0].center = x
    roots[0].perturbation = x.ptb
    roots[0].aux = None
    roots[0].uA = None
    for r in roots[1:]:
        assert isinstance(r, (BoundConstant, BoundParams, BoundBuffers))
        if isinstance(r, BoundParams):
            assert isinstance(r.param, nn.Parameter)
            r.center = r.param
        r.perturbation = None
        r.lA = r.uA = None
    # (dim_output, batch_size, dim_input)
    roots[0].lA = A.sum(dim=2)
