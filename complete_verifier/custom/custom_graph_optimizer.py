#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
This file shows how to use a customized graph optimizers, and it defines a
few customized optimizers for VNN-COMP models as examples. See example
configuration files in the `exp_configs` folder, e.g.,
`exp_configs/vnncomp23/gtrsb.yaml`.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import onnx
from auto_LiRPA import BoundedModule
from auto_LiRPA.bound_ops import BoundSign, BoundAdd, BoundSignMerge


def default_optimizer(model: BoundedModule):
    return model


def merge_sign(model: BoundedModule):
    nodes = list(model.nodes())
    for i, node in enumerate(nodes):
        if (i+2 < len(nodes) and type(node) == BoundSign
            and type(nodes[i+1]) == BoundAdd and type(nodes[i+2]) == BoundSign):
            print('Merging Sign node: %s', node)
            node_merge = BoundSignMerge(inputs=[node.inputs[0]],
                                options=model.bound_opts)
            node_merge.name = f'{node.name}/merge'
            model.add_nodes([node_merge])
            model.replace_node(node, node_merge)
            model.replace_node(nodes[i+1], node_merge)
            model.replace_node(nodes[i+2], node_merge)
    return model