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
"""
This file shows how to use a customized graph optimizers, and it defines a
few customized optimizers for VNN-COMP models as examples. See example
configuration files in the `exp_configs` folder, e.g.,
`exp_configs/vnncomp23/gtrsb.yaml`.
"""

import arguments
import torch
from auto_LiRPA import BoundedModule
from auto_LiRPA.bound_ops import (BoundSign, BoundAdd, BoundSignMerge,
                                  BoundReduceSum, BoundRelu, BoundConstant,
                                  BoundBuffers, BoundMul, BoundSub,
                                  BoundUnsqueeze, BoundMultiPiecewiseNonlinear,
                                  BoundMaxPool, BoundConv, BoundParams)


def default_optimizer(model: BoundedModule):
    pass


def maxpool_to_relu(model: BoundedModule):
    if arguments.Config['solver']['bound_prop_method'] == 'crown':
        residual = True
    else:
        # Residual connections are not supported in the dynamic forward method
        residual = False

    def _create_conv(node_input, node_weight, kernel_size):
        node_conv = BoundConv(
            attr={
                'pads': [0, 0, 0, 0],
                'strides': [kernel_size, kernel_size],
                'kernel_shape': [kernel_size, kernel_size],
                'group': 1,
                'dilations': [1, 1],
            },
            inputs=[node_input, node_weight],
            options=model.bound_opts,
        )
        return node_conv

    nodes = list(model.nodes())
    for i, node in enumerate(nodes):
        if isinstance(node, BoundMaxPool):
            assert node.kernel_size == node.stride == [2, 2]
            assert node.padding == [0, 0]
            assert not node.ceil_mode

            channels = node.output_shape[1]
            eye = torch.eye(channels, device=node.device)

            if residual:
                conv_1_w = torch.zeros(
                    channels * 2, channels, 2, 2, device=node.device)
                conv_1_w[:channels, :, 0, 0] = eye
                conv_1_w[:channels, :, 0, 1] = -eye
                conv_1_w[channels:, :, 1, 0] = eye
                conv_1_w[channels:, :, 1, 1] = -eye
                conv_1_w = torch.nn.Parameter(conv_1_w)
                node_conv_1_w = BoundParams(f'{node.name}/conv_1_w', conv_1_w)
                node_conv_1_w.name = f'{node.name}/conv_1_w'
                node_conv_1 = _create_conv(node.inputs[0], node_conv_1_w, 2)
                node_conv_1.name = f'{node.name}/conv_1'
                node_conv_1_relu = BoundRelu(inputs=[node_conv_1], options=node.options)
                node_conv_1_relu.name = f'{node.name}/conv_1_relu'

                conv_2_w = torch.zeros(
                    channels * 2, channels, 2, 2, device=node.device)
                conv_2_w[:channels, :, 0, 1] = eye
                conv_2_w[channels:, :, 1, 1] = eye
                conv_2_w = torch.nn.Parameter(conv_2_w)
                node_conv_2_w = BoundParams(f'{node.name}/conv_2_w', conv_2_w)
                node_conv_2_w.name = f'{node.name}/conv_2_w'
                node_conv_2 = _create_conv(node.inputs[0], node_conv_2_w, 2)
                node_conv_2.name = f'{node.name}/conv_2'
                node_add_1_2 = BoundAdd(inputs=[node_conv_1_relu, node_conv_2])
                node_add_1_2.name = f'{node.name}/add_1_2'

                conv_3_w = torch.zeros(
                    channels, channels * 2, 1, 1, device=node.device)
                conv_3_w[:, :channels, 0, 0] = eye
                conv_3_w[:, channels:, 0, 0] = -eye
                conv_3_w = torch.nn.Parameter(conv_3_w)
                node_conv_3_w = BoundParams(f'{node.name}/conv_3_w', conv_3_w)
                node_conv_3_w.name = f'{node.name}/conv_3_w'
                node_conv_3 = _create_conv(node_add_1_2, node_conv_3_w, 1)
                node_conv_3.name = f'{node.name}/conv_3'
                node_conv_3_relu = BoundRelu(inputs=[node_conv_3], options=node.options)
                node_conv_3_relu.name = f'{node.name}/conv_3_relu'

                conv_4_w = torch.zeros(
                    channels, channels * 2, 1, 1, device=node.device)
                conv_4_w[:, channels:, 0, 0] = eye
                conv_4_w = torch.nn.Parameter(conv_4_w)
                node_conv_4_w = BoundParams(f'{node.name}/conv_4_w', conv_4_w)
                node_conv_4_w.name = f'{node.name}/conv_4_w'
                node_conv_4 = _create_conv(node_add_1_2, node_conv_4_w, 1)
                node_conv_4.name =  f'{node.name}/conv_4'
                node_add_3_4 = BoundAdd(inputs=[node_conv_3_relu, node_conv_4])
                node_add_3_4.name = f'{node.name}/add_3_4'

                model.add_nodes([
                    node_conv_1,
                    node_conv_1_relu,
                    node_conv_1_w,
                    node_conv_2,
                    node_conv_2_w,
                    node_conv_3,
                    node_conv_3_relu,
                    node_conv_3_w,
                    node_conv_4,
                    node_conv_4_w,
                    node_add_1_2,
                    node_add_3_4,
                ])
                model.replace_node(node, node_add_3_4)
            else:
                # No residual connection but more channels
                conv_1_w = torch.zeros(
                    channels * 6, channels, 2, 2, device=node.device)

                conv_1_w[:channels, :, 0, 0] = eye
                conv_1_w[:channels, :, 0, 1] = -eye
                conv_1_w[channels:channels*2, :, 0, 1] = eye
                conv_1_w[channels*2:channels*3, :, 0, 1] = -eye

                conv_1_w[channels*3:channels*4, :, 1, 0] = eye
                conv_1_w[channels*3:channels*4, :, 1, 1] = -eye
                conv_1_w[channels*4:channels*5, :, 1, 1] = eye
                conv_1_w[channels*5:channels*6, :, 1, 1] = -eye

                conv_1_w = torch.nn.Parameter(conv_1_w)
                node_conv_1_w = BoundParams(f'{node.name}/conv_1_w', conv_1_w)
                node_conv_1_w.name = f'{node.name}/conv_1_w'
                node_conv_1 = _create_conv(node.inputs[0], node_conv_1_w, 2)
                node_conv_1.name = f'{node.name}/conv_1'
                node_conv_1_relu = BoundRelu(inputs=[node_conv_1], options=node.options)
                node_conv_1_relu.name = f'{node.name}/conv_1_relu'

                eye_triple = eye.repeat(1, 3)
                conv_2_w = torch.zeros(
                    channels * 3, channels * 6, 1, 1, device=node.device)
                conv_2_w[:channels, :channels*3, 0, 0] = eye_triple
                conv_2_w[:channels, channels*3:, 0, 0] = -eye_triple
                conv_2_w[channels:channels*2, channels*3:, 0, 0] = eye_triple
                conv_2_w[channels*2:, channels*3:, 0, 0] = -eye_triple
                conv_2_w = torch.nn.Parameter(conv_2_w)
                node_conv_2_w = BoundParams(f'{node.name}/conv_2_w', conv_2_w)
                node_conv_2_w.name = f'{node.name}/conv_2_w'
                node_conv_2 = _create_conv(node_conv_1_relu, node_conv_2_w, 1)
                node_conv_2.name = f'{node.name}/conv_2'
                node_conv_2_relu = BoundRelu(inputs=[node_conv_2], options=node.options)
                node_conv_2_relu.name = f'{node.name}/conv_2_relu'

                conv_3_w = torch.zeros(
                    channels, channels * 3, 1, 1, device=node.device)
                conv_3_w[:, :, 0, 0] = eye_triple
                conv_3_w = torch.nn.Parameter(conv_3_w)
                node_conv_3_w = BoundParams(f'{node.name}/conv_3_w', conv_3_w)
                node_conv_3_w.name = f'{node.name}/conv_3_w'
                node_conv_3 = _create_conv(node_conv_2_relu, node_conv_3_w, 1)
                node_conv_3.name =  f'{node.name}/conv_4'

                model.add_nodes([
                    node_conv_1,
                    node_conv_1_relu,
                    node_conv_1_w,
                    node_conv_2,
                    node_conv_2_relu,
                    node_conv_2_w,
                    node_conv_3,
                    node_conv_3_w,
                ])
                model.replace_node(node, node_conv_3)


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


def merge_relu_lookup_table(model: BoundedModule):
    nodes = list(model.nodes())
    for node in nodes:
        if (isinstance(node, BoundReduceSum)
                and isinstance(node.inputs[0], BoundMul)
                and isinstance(node.inputs[1], BoundConstant)
                and node.inputs[1].value.item() == -1):
            node_mul = node.inputs[0]
            if (isinstance(node_mul.inputs[1], BoundBuffers)
                    and node_mul.inputs[1].buffer.ndim == 1
                    and isinstance(node_mul.inputs[0], BoundRelu)
                    and isinstance(node_mul.inputs[0].inputs[0], BoundSub)
            ):
                node_sub = node_mul.inputs[0].inputs[0]
                node_weight = node_mul.inputs[1]
                if (isinstance(node_sub.inputs[1], BoundBuffers)
                        and node_sub.inputs[1].buffer.ndim == 1
                        and isinstance(node_sub.inputs[0], BoundUnsqueeze)
                        and not node_sub.inputs[0].inputs[1].perturbed
                        and node_sub.inputs[0].inputs[1].value == -1
                        and len(node_sub.inputs[0].inputs[0].output_shape) == 2
                ):
                    node_offset = node_sub.inputs[1]
                    node_input = node_sub.inputs[0].inputs[0]
                    # (weight*ReLU(node_input.unsqueeze(-1)-offset)).sum(-1)
                    print('Found a ReLU-based lookup table')
                    print('Input node:', node_input)
                    print('Offset:', node_offset.buffer)
                    print('Weight:', node_weight.buffer)
                    node_merged = BoundMultiPiecewiseNonlinear(
                        inputs=[node_input, node_weight, node_offset])
                    node_merged.name = f'{node.name}/merged'
                    model.add_nodes([node_merged])
                    model.replace_node(node, node_merged)
                    print('New node created:', node_merged)
