#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Optimizing computation graph in onnx file."""
from typing import List

import torch
import onnx
from onnx import numpy_helper as nh
import onnxruntime as ort
import numpy as np


def convert_onnx_to_double(onnx_file):
    """Convert onnx file from single precision to double precision."""
    onnx_model = onnx.load(onnx_file)

    def _convert_data(obj):
        fp64_initializer = nh.to_array(obj).astype(np.float64)
        obj.data_type = onnx.TensorProto.DOUBLE
        obj.raw_data = nh.from_array(fp64_initializer).raw_data

    # Change all initializer to fp64.
    for node in onnx_model.graph.initializer:
        _convert_data(node)
    # Change all inputs to fp64.
    for node in onnx_model.graph.input:
        node.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    # Change all outputs to fp64.
    for node in onnx_model.graph.output:
        node.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    # Change node attributes to fp64.
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if hasattr(attr, 't') and hasattr(attr.t, 'raw_data') and attr.t.raw_data:
                _convert_data(attr.t)
    # for node in onnx_model.graph.nodes:
    print('Model converted to double precision.')
    return onnx_model


def fuse_conv_and_bn(conv, bn):
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    #
    # we're done
    return fusedconv


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = None
) -> onnx.TensorProto:

    if data_type is None:
        if tensor_array.dtype in ['float32', 'float64']:
            data_type = onnx.TensorProto.FLOAT
        elif tensor_array.dtype == 'int64':
            data_type = onnx.TensorProto.INT64
        else:
            raise NotImplementedError(tensor_array.dtype)

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


def create_new_initializers(node, initializers):
    new_initializers = []
    for old_init in node.input:
        if old_init in initializers:
            cur_init = create_initializer_tensor(
                name=old_init,
                tensor_array=initializers[old_init])
            new_initializers.append(cur_init)
    return new_initializers


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def strided_conversion(input_feature, kernel, padding, stride):
    output_channel, input_channel, kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_feature.shape[-2:]
    output_h, output_w = conv_output_shape(h_w=(input_h, input_w), kernel_size=(kernel_h, kernel_w), stride=stride,
                                           pad=padding)
    total_locs = output_h * output_w  # total number of output locations.

    converted_matrix = torch.zeros(output_channel, total_locs, input_channel,
                                   (input_h + 2 * padding) * (input_w + 2 * padding)).type(torch.DoubleTensor)
    orig_stride = converted_matrix.stride()

    matrix_strided = torch.as_strided(converted_matrix,
                                      [output_channel, total_locs, output_h, output_w, input_channel, kernel_h,
                                       kernel_w],
                                      [orig_stride[0], orig_stride[1], (input_h + 2 * padding) * stride, stride,
                                       orig_stride[2], input_h + 2 * padding, 1])

    first_indices = torch.arange(total_locs)
    second_indices = torch.div(first_indices, output_h, rounding_mode="trunc")
    third_indices = torch.fmod(first_indices, output_h)
    matrix_strided[:, first_indices, second_indices, third_indices, :, :, :] = kernel.reshape(output_channel, 1,
                                                                                              input_channel, kernel_h,
                                                                                              kernel_w)
    converted_matrix = converted_matrix.view(output_channel * total_locs, input_channel, input_h + 2 * padding,
                                             input_w + 2 * padding)
    if padding > 0:
        converted_matrix = converted_matrix[:, :, padding:-padding, padding:-padding]
    # Output shape is (outC, outH, outW, inC, inH, inW)
    return converted_matrix.reshape(output_channel * output_h * output_w,
                                    converted_matrix.shape[1] * converted_matrix.shape[2] * converted_matrix.shape[3])


def fuse_cgan_gemm_reshape_bn(onnx_model, initializers):
    old_linear_weight = initializers[onnx_model.graph.node[0].input[1]]
    old_linear_bias = initializers[onnx_model.graph.node[0].input[2]]
    old_bn_weight = initializers[onnx_model.graph.node[3].input[1]]
    old_bn_bias = initializers[onnx_model.graph.node[3].input[2]]
    old_bn_mean = initializers[onnx_model.graph.node[3].input[3]]
    old_bn_var = initializers[onnx_model.graph.node[3].input[4]]

    dim_flattened = old_linear_bias.shape[0]
    dim_channel = old_bn_weight.shape[0]
    new_linear_weight = old_linear_weight.reshape(dim_channel, int(dim_flattened/dim_channel), -1).transpose(2, 1, 0) * old_bn_weight / np.sqrt(old_bn_var + onnx_model.graph.node[3].attribute[0].f)
    new_linear_bias = old_linear_bias.reshape(dim_channel, int(dim_flattened/dim_channel)).transpose(1, 0) * old_bn_weight / np.sqrt(old_bn_var + onnx_model.graph.node[3].attribute[0].f) + old_bn_bias - old_bn_weight * old_bn_mean / np.sqrt(old_bn_var + onnx_model.graph.node[3].attribute[0].f)
    new_linear_weight = new_linear_weight.transpose(0, 2, 1).reshape(-1, dim_flattened).transpose(1, 0)
    new_linear_bias = new_linear_bias.transpose(1, 0).reshape(dim_flattened)

    return new_linear_weight, new_linear_bias


def fuse_cgan_bn_reshape_gemm(onnx_model, initializers):
    old_linear_weight = initializers[onnx_model.graph.node[-1].input[1]]
    old_linear_bias = initializers[onnx_model.graph.node[-1].input[2]]
    old_bn_weight = initializers[onnx_model.graph.node[-4].input[1]]
    old_bn_bias = initializers[onnx_model.graph.node[-4].input[2]]
    old_bn_mean = initializers[onnx_model.graph.node[-4].input[3]]
    old_bn_var = initializers[onnx_model.graph.node[-4].input[4]]

    dim_flattened = old_linear_weight.shape[1]
    old_bn_weight_flatten = old_bn_weight[:, None].repeat(4, axis=-1).reshape(dim_flattened)
    old_bn_bias_flatten = old_bn_bias[:, None].repeat(4, axis=-1).reshape(dim_flattened)
    old_bn_mean_flatten = old_bn_mean[:, None].repeat(4, axis=-1).reshape(dim_flattened)
    old_bn_var_flatten = old_bn_var[:, None].repeat(4, axis=-1).reshape(dim_flattened)

    new_linear_weight = old_linear_weight * old_bn_weight_flatten / np.sqrt(old_bn_var_flatten + onnx_model.graph.node[-4].attribute[0].f)
    new_linear_bias = np.matmul(old_linear_weight, (old_bn_bias_flatten - old_bn_weight_flatten * old_bn_mean_flatten / np.sqrt(old_bn_var_flatten + onnx_model.graph.node[-4].attribute[0].f))) + old_linear_bias

    return new_linear_weight, new_linear_bias


def compress_onnx(onnx_model, old_path, save_path, onnx_optimization_flags: List[str], debug=False):
    assert len(onnx_optimization_flags) > 0
    if debug:
        place_holder = []
        for x in onnx_model.graph.input[0].type.tensor_type.shape.dim:
            place_holder.append(x.dim_value)

        place_holder[0] = 1
        place_holder = tuple(place_holder)

    cur_W, cur_b = None, None
    source = onnx_model.graph.input[0].name

    initializers = {}
    for onnx_module in onnx_model.graph.initializer:
        initializers[onnx_module.name] = nh.to_array(onnx_module)

    new_initializers = []
    new_nodes = []

    cnt, convcnt = 0, 0

    if "merge_gemm" in onnx_optimization_flags:  # bias only. need to generalize this later.
        W1 = initializers[onnx_model.graph.node[0].input[1]]
        b1 = initializers[onnx_model.graph.node[0].input[2]]

        W2 = initializers[onnx_model.graph.node[3].input[1]]
        b2 = initializers[onnx_model.graph.node[3].input[2]]

        input_feature = torch.randn((1, 3, 32, 32)).type(torch.DoubleTensor)
        pads = 1
        strides = 1

        nW = strided_conversion(input_feature, torch.from_numpy(W2).type(torch.DoubleTensor), pads,
                                strides).detach().numpy().astype(np.float32)
        nb = np.repeat(b2, 32 * 32)

        W3 = np.matmul(nW, W1)
        b3 = np.matmul(nW, b1) + nb

        cur_linear_W = create_initializer_tensor(
            name='linear_W',
            tensor_array=W3)

        new_initializers.append(cur_linear_W)

        cur_linear_b = create_initializer_tensor(
            name='linear_b',
            tensor_array=b3)

        new_initializers.append(cur_linear_b)

        cur_node = onnx.helper.make_node(
            name='linear_MatMul',
            op_type='Gemm',
            inputs=[onnx_model.graph.node[0].input[0], 'linear_W', 'linear_b'],
            outputs=[onnx_model.graph.node[3].output[0]],
            alpha=1.0,
            beta=1.0,
            transB=1
        )

        new_nodes.append(cur_node)

        re_node = onnx.helper.make_node(
            name='Relu_4',
            op_type='Relu',
            inputs=[onnx_model.graph.node[3].output[0]],
            outputs=['after_relu']
        )

        new_nodes.append(re_node)

        values = np.array([1, 16, 32, 32])
        cnode = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=values.shape,
                vals=values.flatten().astype(int),
            ),
        )

        new_nodes.append(cnode)

        rnode = onnx.helper.make_node(
            'Reshape',
            inputs=['after_relu', 'values'],
            outputs=[onnx_model.graph.node[4].output[0]],
        )

        new_nodes.append(rnode)

        started_node = 5

    elif "merge_gemm_reshape_bn" in onnx_optimization_flags:
        # TODO (assigned to Zhuowen): generalize the code to allow the merge of gemm + BN for any general onnx.
        merge_case = (onnx_model.graph.node[0].op_type == "Gemm" and 3 < len(onnx_model.graph.node) and
                onnx_model.graph.node[1].op_type == "Constant" and
                onnx_model.graph.node[2].op_type == "Reshape" and onnx_model.graph.node[3].op_type == "BatchNormalization")
        if merge_case:
            new_linear_weight, new_linear_bias = fuse_cgan_gemm_reshape_bn(onnx_model, initializers)

            linear_W_initializer = create_initializer_tensor(
                name='linear_W',
                tensor_array=new_linear_weight)
            new_initializers.append(linear_W_initializer)
            linear_b_initializer = create_initializer_tensor(
                name='linear_b',
                tensor_array=new_linear_bias)
            new_initializers.append(linear_b_initializer)

            cur_node = onnx.helper.make_node(
                op_type='Gemm',
                inputs=[onnx_model.graph.node[0].input[0], 'linear_W', 'linear_b'],
                outputs=['after_linear'],
                alpha=1.0,
                beta=1.0,
                transB=1
            )
            new_nodes.append(cur_node)

            values = nh.to_array(onnx_model.graph.node[1].attribute[0].t)
            cnode = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['values'],
                value=onnx.helper.make_tensor(
                    name='const_tensor',
                    data_type=onnx.TensorProto.INT64,
                    dims=values.shape,
                    vals=values.flatten().astype(int),
                ),
            )
            new_nodes.append(cnode)
            rnode = onnx.helper.make_node(
                'Reshape',
                inputs=['after_linear', 'values'],
                outputs=[onnx_model.graph.node[3].output[0]],
            )
            new_nodes.append(rnode)

            started_node = 4
        else:
            started_node = 0
    else:
        started_node = 0

    skipped = {}
    const_var = {}
    added_const = {}

    def trace(node_name):
        if node_name in skipped:
            return trace(skipped[node_name])
        return node_name

    if "remove_matmul_inplace" in onnx_optimization_flags:
        to_transpose_const, temp_dict = {}, {}
        for i, node in enumerate(onnx_model.graph.node[started_node:]):
            if node.op_type == "Constant":
                temp_dict[node.output[0]] = i
            matmul_case1 = (node.op_type == "MatMul" and i - 1 >= 0 and i + 1 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i - 1].op_type == "Transpose" and onnx_model.graph.node[i + 1].op_type == "Transpose" and
                    onnx_model.graph.node[i - 1].output[0] == node.input[1])
            matmul_case2 = (node.op_type == "MatMul" and i - 2 >= 0 and i + 1 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i - 2].op_type == "Transpose" and onnx_model.graph.node[i + 1].op_type == "Transpose" and
                    onnx_model.graph.node[i - 2].output[0] == node.input[1])
            if matmul_case1 or matmul_case2:
                to_transpose_const[i] = temp_dict[node.input[0]]
    elif "merge_vit" in onnx_optimization_flags:
        to_merge_linears, search_list, to_merge_others = [], [], []
        for i, node in enumerate(onnx_model.graph.node[started_node:]):
            if (node.op_type == "Transpose" and onnx_model.graph.node[i - 1].op_type == "BatchNormalization" and
                onnx_model.graph.node[i - 2].op_type == "Transpose"):
                search_list.append(node.output[0])
            elif node.op_type == "MatMul" and i + 1 < len(onnx_model.graph.node) and onnx_model.graph.node[i + 1].op_type == "Add":
                if node.input[0] in search_list:
                    to_merge_linears.append(i)
            elif node.op_type == "Shape" and node.input[0] in search_list:
                to_merge_others.append(i)

    for i, node in enumerate(onnx_model.graph.node[started_node:]):
        if "remove_ineffective_layers" in onnx_optimization_flags:
            if node.op_type == "Constant":
                val = nh.to_array(node.attribute[0].t)
                mn, mx = np.min(val.reshape(-1)), np.max(val.reshape(-1))
                if (mx - mn <= 1e-9).all():
                    const_var[node.output[0]] = (val, node)
                else:
                    added_const[node.output[0]] = True
                    new_nodes.append(node)
                continue
            elif node.op_type == "Sub":
                if node.input[1] in const_var:
                    if (np.abs(const_var[node.input[1]][0]) <= 1e-9).all():
                        skipped[node.output[0]] = trace(node.input[0])
                        print('Remove ineffective operation: sub(0).')
                        continue
                if node.input[1] not in added_const:
                    added_const[node.input[1]] = True
                    new_nodes.append(const_var[node.input[1]][1])
                # node.input[0] = trace(node.input[0])
                new_nodes.append(node)
                continue
            elif node.op_type == "Div":
                if node.input[1] in const_var:
                    if np.abs(const_var[node.input[1]][0] - 1) <= 1e-9:
                        skipped[node.output[0]] = trace(node.input[0])
                        print('Remove ineffective operation: div(0).')
                        continue
                if node.input[1] not in added_const:
                    added_const[node.input[1]] = True
                    new_nodes.append(const_var[node.input[1]][1])
                node.input[0] = trace(node.input[0])
                new_nodes.append(node)
                continue
        if "merge_bn" in onnx_optimization_flags:
            if node.op_type == "Conv":
                if i + 1 < len(onnx_model.graph.node) and onnx_model.graph.node[i + 1].op_type == "BatchNormalization":
                    bn_node = onnx_model.graph.node[i + 1]
                    w_conv = initializers[node.input[1]]
                    bn_weight = initializers[bn_node.input[1]]
                    bn_bias = initializers[bn_node.input[2]]
                    bn_mean = initializers[bn_node.input[3]]
                    bn_var = initializers[bn_node.input[4]]
                    w_shape = w_conv.shape
                    bn_eps = bn_node.attribute[0].f
                    w_bn = np.diag(bn_weight / (np.sqrt(bn_eps + bn_var)))

                    w_conv = w_conv.reshape(w_shape[0], -1)
                    if len(node.input) == 3:
                        b_conv = node.input[2]
                    else:
                        b_conv = np.zeros(w_shape[0])

                    b_bn = bn_bias - bn_weight * bn_mean / np.sqrt(bn_var + bn_eps)

                    W = np.matmul(w_bn, w_conv).reshape(*w_shape)
                    b = np.matmul(w_bn, b_conv) + b_bn

                    triggered = True

                    conv_W_initializer = create_initializer_tensor(
                        name=f"conv{convcnt}_W",
                        tensor_array=W)

                    conv_b_initializer = create_initializer_tensor(
                        name=f"conv{convcnt}_b",
                        tensor_array=b
                    )
                    conv_node = onnx.helper.make_node(
                        name=f"conv{convcnt}",
                        op_type="Conv",
                        inputs=[node.input[0], f"conv{convcnt}_W", f"conv{convcnt}_b"],
                        outputs=[bn_node.output[0]],
                        dilations=tuple(node.attribute[0].ints),
                        kernel_shape=tuple(node.attribute[2].ints),
                        pads=tuple(node.attribute[3].ints),
                        strides=tuple(node.attribute[4].ints)
                    )
                    new_nodes.append(conv_node)
                    new_initializers.append(conv_W_initializer)
                    new_initializers.append(conv_b_initializer)
                    convcnt += 1
                else:
                    new_nodes.append(node)
                    source = node.output[0]
                    new_initializers.extend(create_new_initializers(node, initializers))
                    triggered = False
            elif node.op_type == "BatchNormalization":
                if triggered:
                    triggered = False
                    continue
                new_nodes.append(node)
                source = node.output[0]
                new_initializers.extend(create_new_initializers(node, initializers))
            else:
                new_nodes.append(node)
                source = node.output[0]
                new_initializers.extend(create_new_initializers(node, initializers))
        elif "merge_linear" in onnx_optimization_flags:
            if node.op_type == "MatMul" or node.op_type == "Add":
                if node.input[-1] in initializers:
                    layer_mat = initializers[node.input[-1]]
                else:
                    new_nodes.append(node)
                    source = node.output[0]
                    new_initializers.extend(create_new_initializers(node, initializers))
                    continue
                if node.op_type == "MatMul":
                    if cur_W is None:
                        cur_W = layer_mat
                    else:
                        cur_W = np.matmul(cur_W, layer_mat)
                        if cur_b is not None: cur_b = np.matmul(cur_b, layer_mat)
                elif node.op_type == "Add":
                    if cur_b is None:
                        cur_b = layer_mat
                    else:
                        layer_b = layer_mat
                        cur_b += layer_b
                target = node.output[0]

                if (i == len(onnx_model.graph.node) - 1 or (
                        onnx_model.graph.node[i + 1].op_type != "MatMul" and
                        onnx_model.graph.node[i + 1].op_type != "Add")):
                    if cur_W is not None:
                        cur_linear_W = create_initializer_tensor(
                            name=f'linear{cnt}_W',
                            tensor_array=cur_W)
                        output_node = target
                        if cur_b is not None: output_node = f'linear{cnt}_intermediate'
                        cur_node = onnx.helper.make_node(
                            name=f'linear{cnt}_MatMul',
                            op_type='MatMul',
                            inputs=[source, f'linear{cnt}_W'],
                            outputs=[output_node]
                        )
                        new_initializers.append(cur_linear_W)
                        new_nodes.append(cur_node)
                    if cur_b is not None:
                        cur_linear_b = create_initializer_tensor(
                            name=f'linear{cnt}_b',
                            tensor_array=cur_b)

                        input_node = source
                        if cur_W is not None: input_node = f'linear{cnt}_intermediate'

                        cur_node = onnx.helper.make_node(
                            name=f'linear{cnt}_Add',
                            op_type='Add',
                            inputs=[input_node, f'linear{cnt}_b'],
                            outputs=[target]
                        )
                        new_initializers.append(cur_linear_b)
                        new_nodes.append(cur_node)
                    if cur_W is not None or cur_b is not None: cnt += 1
                    cur_W = None
                    cur_b = None
                    source = target
            else:
                for input_node in node.input:
                    if input_node in const_var:
                        if input_node not in added_const:
                            added_const[input_node] = True
                            new_nodes.append(const_var[input_node][1])
                new_nodes.append(node)
                source = node.output[0]
                new_initializers.extend(create_new_initializers(node, initializers))
        elif "merge_vit" in onnx_optimization_flags:
            if node.op_type == "BatchNormalization":
                if onnx_model.graph.node[i - 1].op_type == "Transpose" and onnx_model.graph.node[i + 1].op_type == "Transpose":
                    source_input = onnx_model.graph.node[i - 1].input[0]
                else:
                    source_input = node.input[0]
                bn_node = node
                bn_weight = initializers[bn_node.input[1]]
                bn_bias = initializers[bn_node.input[2]]
                bn_mean = initializers[bn_node.input[3]]
                bn_var = initializers[bn_node.input[4]]
                bn_eps = bn_node.attribute[0].f
                w_bn = np.diag(bn_weight / (np.sqrt(bn_eps + bn_var)))
                b_bn = bn_bias - bn_weight * bn_mean / np.sqrt(bn_var + bn_eps)
                ori_w = w_bn
                ori_b = b_bn

                if i + 2 not in to_merge_linears and i + 2 not in to_merge_others:
                    linear_W_initializer = create_initializer_tensor(
                        name=f"linear{cnt}_W",
                        tensor_array=ori_w)

                    linear_b_initializer = create_initializer_tensor(
                        name=f"linear{cnt}_b",
                        tensor_array=ori_b
                    )
                    matmul_node = onnx.helper.make_node(
                        name=f'linear{cnt}_MatMul',
                        op_type='MatMul',
                        inputs=[source_input, f'linear{cnt}_W'],
                        outputs=[f'linear{cnt}_intermediate']
                    )
                    add_node = onnx.helper.make_node(
                        name=f'linear{cnt}_Add',
                        op_type='Add',
                        inputs=[f'linear{cnt}_intermediate', f'linear{cnt}_b'],
                        outputs=[node.output[0]]
                    )
                    cnt += 1
                    new_nodes.append(matmul_node)
                    new_nodes.append(add_node)
                    new_initializers.append(linear_W_initializer)
                    new_initializers.append(linear_b_initializer)
            elif i in to_merge_others:
                node.input[0] = source_input
                new_nodes.append(node)
            elif i in to_merge_linears:
                w_cur = initializers[node.input[1]]
                b_cur = initializers[onnx_model.graph.node[i + 1].input[0]]
                W = np.matmul(ori_w, w_cur)
                b = np.matmul(ori_b, w_cur) + b_cur

                linear_W_initializer = create_initializer_tensor(
                    name=f"linear{cnt}_W",
                    tensor_array=W
                )
                linear_b_initializer = create_initializer_tensor(
                    name=f"linear{cnt}_b",
                    tensor_array=b
                )
                matmul_node = onnx.helper.make_node(
                    name=f'linear{cnt}_MatMul',
                    op_type='MatMul',
                    inputs=[source_input, f'linear{cnt}_W'],
                    outputs=[f'linear{cnt}_intermediate']
                )
                add_node = onnx.helper.make_node(
                    name=f'linear{cnt}_Add',
                    op_type='Add',
                    inputs=[f'linear{cnt}_intermediate', f'linear{cnt}_b'],
                    outputs=[onnx_model.graph.node[i + 1].output[0]]
                )
                cnt += 1
                new_nodes.append(matmul_node)
                new_nodes.append(add_node)
                new_initializers.append(linear_W_initializer)
                new_initializers.append(linear_b_initializer)
            elif (i - 1 not in to_merge_linears and not (node.op_type == "Transpose"
                    and (onnx_model.graph.node[i - 1].op_type == "BatchNormalization"
                    or onnx_model.graph.node[i + 1].op_type == "BatchNormalization"))):
                new_nodes.append(node)
                source = node.output[0]
                new_initializers.extend(create_new_initializers(node, initializers))
        elif "remove_matmul_inplace" in onnx_optimization_flags:
            transpose_case1 = (node.op_type == "Transpose" and i - 2 >= 0 and
                    onnx_model.graph.node[i - 1].op_type == "MatMul" and onnx_model.graph.node[i - 2].op_type == "Transpose")
            transpose_case2 = (node.op_type == "Transpose" and i - 3 >= 0 and
                    onnx_model.graph.node[i - 1].op_type == "MatMul" and onnx_model.graph.node[i - 3].op_type == "Transpose")
            transpose_case3 = (node.op_type == "Transpose" and i + 2 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i + 1].op_type == "MatMul" and onnx_model.graph.node[i + 2].op_type == "Transpose")
            transpose_case4 = (node.op_type == "Transpose" and i + 3 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i + 2].op_type == "MatMul" and onnx_model.graph.node[i + 3].op_type == "Transpose")
            matmul_case1 = (node.op_type == "MatMul" and i - 1 >= 0 and i + 1 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i - 1].op_type == "Transpose" and onnx_model.graph.node[i + 1].op_type == "Transpose" and
                    onnx_model.graph.node[i - 1].output[0] == node.input[1])
            matmul_case2 = (node.op_type == "MatMul" and i - 2 >= 0 and i + 1 < len(onnx_model.graph.node) and
                    onnx_model.graph.node[i - 2].op_type == "Transpose" and onnx_model.graph.node[i + 1].op_type == "Transpose" and
                    onnx_model.graph.node[i - 2].output[0] == node.input[1])

            if not (transpose_case1 or transpose_case2 or transpose_case3 or transpose_case4):
                if node.op_type == "Constant" and i in list(to_transpose_const.values()):
                    continue
                elif matmul_case1 or matmul_case2:
                    if matmul_case1:
                        source_input = onnx_model.graph.node[i - 1].input[0]
                    else:
                        source_input = onnx_model.graph.node[i - 2].input[0]
                    const_node = onnx_model.graph.node[to_transpose_const[i]]
                    val = nh.to_array(const_node.attribute[0].t).transpose(1, 0)
                    new_tensor = create_initializer_tensor(
                        name=f"Constant{cnt}",
                        tensor_array=val
                    )
                    matmul_node = onnx.helper.make_node(
                        name=f'linear{cnt}_MatMul',
                        op_type='MatMul',
                        inputs=[source_input, f"Constant{cnt}"],
                        outputs=[onnx_model.graph.node[i + 1].output[0]]
                    )
                    new_nodes.append(matmul_node)
                    new_initializers.append(new_tensor)
                    cnt += 1
                else:
                    new_nodes.append(node)
                    source = node.output[0]
                    new_initializers.extend(create_new_initializers(node, initializers))
        elif "fix_gtrsb" in onnx_optimization_flags:
            if node.op_type == "Transpose":
                onnx_model.graph.node[i+1].input[0] = node.input[0]
            elif node.op_type == "MatMul" and cnt == 0:
                w_cur = initializers[node.input[1]]
                shape_now = w_cur.shape
                if shape_now[0] == 23328:
                    w_cur = w_cur.reshape(1, 27, 27, 32, -1)
                elif shape_now[0] == 3136:
                    w_cur = w_cur.reshape(1, 7, 7, 64, -1)
                elif shape_now[0] == 1600:
                    w_cur = w_cur.reshape(1, 5, 5, 64, -1)
                else:
                    raise ValueError
                W = w_cur.transpose(0, 3, 1, 2, 4).reshape(-1, shape_now[-1])
                linear_W_initializer = create_initializer_tensor(
                    name=f"matmul{cnt}_W",
                    tensor_array=W)
                matmul_node = onnx.helper.make_node(
                    name=f'matmul{cnt}_MatMul',
                    op_type='MatMul',
                    inputs=[node.input[0], f"matmul{cnt}_W"],
                    outputs=[node.output[0]]
                )
                cnt += 1
                new_nodes.append(matmul_node)
                new_initializers.append(linear_W_initializer)
            else:
                for i in range(len(node.input)): node.input[i] = trace(node.input[i])
                new_nodes.append(node)
                source = node.output[0]
                for old_init in node.input:
                    if old_init in initializers and old_init not in [i.name for i in new_initializers]:
                        cur_init = create_initializer_tensor(
                            name=old_init,
                            tensor_array=initializers[old_init])
                        new_initializers.append(cur_init)
        else:
            # TODO (assigned to Zhuowen): generalize the code to allow the merge of BN + gemm for any general onnx.
            if "merge_bn_reshape_gemm" in onnx_optimization_flags and (i == len(onnx_model.graph.node[started_node:]) - 4) and (node.op_type == "BatchNormalization" and
                    onnx_model.graph.node[-3].op_type == "Constant" and onnx_model.graph.node[-2].op_type == "Reshape" and onnx_model.graph.node[-1].op_type == "Gemm"):
                new_linear_weight, new_linear_bias = fuse_cgan_bn_reshape_gemm(onnx_model, initializers)

                values = nh.to_array(onnx_model.graph.node[-3].attribute[0].t)
                cnode = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['values1'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.INT64,
                        dims=values.shape,
                        vals=values.flatten().astype(int),
                    ),
                )
                new_nodes.append(cnode)
                rnode = onnx.helper.make_node(
                    'Reshape',
                    inputs=[onnx_model.graph.node[-5].output[0], 'values1'],
                    outputs=['reshape_output'],
                )
                new_nodes.append(rnode)

                linear_W_initializer = create_initializer_tensor(
                    name=f'linear_W1',
                    tensor_array=new_linear_weight)
                new_initializers.append(linear_W_initializer)
                linear_b_initializer = create_initializer_tensor(
                    name=f'linear_b1',
                    tensor_array=new_linear_bias)
                new_initializers.append(linear_b_initializer)

                cur_node = onnx.helper.make_node(
                    op_type='Gemm',
                    inputs=['reshape_output', 'linear_W1', 'linear_b1'],
                    outputs=['Y'],
                    alpha=1.0,
                    beta=1.0,
                    transB=1
                )
                new_nodes.append(cur_node)
                break

            for i in range(len(node.input)):
                node.input[i] = trace(node.input[i])
            new_nodes.append(node)
            source = node.output[0]
            for old_init in node.input:
                if old_init in initializers:
                    value = initializers[old_init]
                    cur_init = create_initializer_tensor(
                        name=old_init, tensor_array=value)
                    new_initializers.append(cur_init)

    model_outputs = onnx_model.graph.output
    input_node = onnx_model.graph.input[0]

    if 'check_duplicate_upsample_initializers' in onnx_optimization_flags:
        # For cGAN transformers, duplicate initializers appear after using
        # https://github.com/daquexian/onnx-simplifier,
        # which cannot pass onnx.checker.check.
        for i in range(len(new_nodes)):
            if ('Upsample' in new_nodes[i].name
                    and len(new_nodes[i].input) == 2
                    and new_nodes[i].input[1] in initializers):
                for j in range(i + 1, len(new_nodes)):
                    if ('Upsample' in new_nodes[j].name
                            and len(new_nodes[j].input) == 2
                            and new_nodes[i].input[1] == new_nodes[j].input[1]):
                        new_name = f'{new_nodes[j].input[1]}.{j}'
                        assert new_name not in initializers
                        new_init = create_initializer_tensor(
                            name=new_name,
                            tensor_array=initializers[new_nodes[j].input[1]])
                        new_nodes[j].input[1] = new_init.name
                        new_initializers.append(new_init)
        new_initializers_dup = new_initializers
        new_initializers = []
        new_initializers_names = []
        for init in new_initializers_dup:
            if init.name not in new_initializers_names:
                new_initializers_names.append(init.name)
                new_initializers.append(init)

    if "remove_squeeze_in_last_layer" in onnx_optimization_flags:
        if new_nodes[-1].op_type == "Squeeze" and len(new_nodes[-1].attribute) == 0:
            new_nodes = new_nodes[:-1]
            model_outputs = [onnx.helper.make_value_info(
                new_nodes[-1].output[0],
                onnx.helper.make_tensor_type_proto(1, [1, 1]))]
            print("Remove the squeeze in the last layer.")
        elif (len(new_nodes) >= 2
                and new_nodes[-2].op_type == "Squeeze"
                and len(new_nodes[-2].attribute) == 0
                and new_nodes[-1].op_type == "Reshape"
                and new_nodes[-1].input[0] == new_nodes[-2].output[0]
                and new_nodes[-1].input[1] in initializers
                and initializers[new_nodes[-1].input[1]].tolist() == [1, 1]):
            reshape_shape = new_nodes[-1].input[1]
            squeeze_input = new_nodes[-2].input[0]
            new_nodes = new_nodes[:-2]
            model_outputs = [onnx.helper.make_tensor_value_info(
                squeeze_input, onnx.TensorProto.FLOAT, [1, 1])]
            new_initializers = [
                initializer for initializer in new_initializers
                if initializer.name != reshape_shape]
            print("Remove the Squeeze-Reshape output wrapper.")

    if "remove_relu_in_last_layer" in onnx_optimization_flags:
        if new_nodes[-1].op_type == "Relu":
            new_nodes = new_nodes[:-1]
            model_outputs[0].name = new_nodes[-1].output[0]
            print("Remove the relu in the last layer.")
    if "fix_gtrsb" in onnx_optimization_flags:
        for i, node in enumerate(new_nodes):
            if node.op_type == "Reshape":
                tmp = node.input[0]
                node.input[0] = new_nodes[i+3].output[0]
                new_nodes[i+1].input[0] = tmp
                new_nodes[i+4].input[0] = node.output[0]
                new_nodes[i] = new_nodes[i+1]
                new_nodes[i+1] = new_nodes[i+2]
                new_nodes[i+2] = new_nodes[i+3]
                new_nodes[i+3] = node
                break
        input_node = onnx.helper.make_tensor_value_info(
            name=onnx_model.graph.input[0].name,
            elem_type=1,
            shape=("unk__195", place_holder[3], place_holder[1], place_holder[2])
        )

    """
    # we should do flatten here, but the collins_yolo_robustness has incompatible onnx version
    # that make the conversion fail. We temporally flatten outside.
    if "flatten_final_output" in onnx_optimization_flags:
        flatten_node = onnx.helper.make_node(
            op_type='Flatten',
            inputs=[new_nodes[-1].output[0]],
            outputs=['Flattened_final_output'],
        )
        new_nodes.append(flatten_node)
        model_outputs[0].name = new_nodes[-1].output[0]
        print("Flattened the final output.")
    """

    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name="CompressedNet",
        inputs=[input_node],
        outputs=model_outputs,
        initializer=new_initializers
    )
    model_def = onnx.helper.make_model(new_graph, producer_name="onnx_example")
    if ('merge_vit' in onnx_optimization_flags
            or 'check_duplicate_upsample_initializers' in onnx_optimization_flags):
        model_def.opset_import[0].version = 9
    else:
        model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)

    if "optimize_relu_relationship" in onnx_optimization_flags:
        model_def = optimize_relu_relationship(model_def)

    onnx.checker.check_model(model_def)
    onnx.save(model_def, save_path)

    if debug:
        errors = np.zeros(100)
        # Workaround for onnx bug, see issue #150
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        for _ in range(100):
            inputs = torch.randn(place_holder).numpy()
            ort_sess = ort.InferenceSession(old_path, sess_options=options)
            input_name = ort_sess.get_inputs()[0].name
            output_name = ort_sess.get_outputs()[0].name
            output1 = ort_sess.run([output_name], {input_name: inputs})

            if "fix_gtrsb" in onnx_optimization_flags:
                inputs = inputs.transpose(0, 3, 1, 2)
            ort_sess = ort.InferenceSession(save_path, sess_options=options)
            input_name = ort_sess.get_inputs()[0].name
            output_name = ort_sess.get_outputs()[0].name
            output2 = ort_sess.run([output_name], {input_name: inputs})

        if "remove_relu_in_last_layer" in onnx_optimization_flags:
            output2 = np.array(output2).clip(min=0)

        errors[_] = np.sum(np.abs(np.array(output1) - np.array(output2)).reshape(-1))

        print(f"Compressed {len(onnx_model.graph.node) - len(model_def.graph.node)} Onnx nodes")
        print('Sum/mean of errors during compression: '
              f'{np.sum(errors)}/{np.mean(errors)}')

    return model_def


def optimize_relu_relationship(onnx_model, atol=1e-6, rtol=1e-4, sanity_n=1000):
    """Collapse a recognized affine-ReLU block into an equivalent smaller block.
    """
    import copy
    import time
    t_pass_start = time.time()
    original_model = copy.deepcopy(onnx_model)

    initializers = {init.name: init for init in onnx_model.graph.initializer}

    def _w(name):
        if name in initializers:
            return nh.to_array(initializers[name])
        return None

    def _gemm_attrs(node):
        attrs = {a.name: a for a in node.attribute}
        transA = int(attrs['transA'].i) if 'transA' in attrs else 0
        transB = int(attrs['transB'].i) if 'transB' in attrs else 0
        alpha = float(attrs['alpha'].f) if 'alpha' in attrs else 1.0
        beta = float(attrs['beta'].f) if 'beta' in attrs else 1.0
        return transA, transB, alpha, beta

    def _identify(W2_row, b2_i, W1, b1):
        nz = np.where(np.abs(W2_row) > atol)[0]
        if len(nz) != 2:
            return None
        idx_a, idx_b = int(nz[0]), int(nz[1])
        va, vb = float(W2_row[idx_a]), float(W2_row[idx_b])
        if va > 0 and vb < 0:
            idx1, idx2 = idx_a, idx_b
        elif va < 0 and vb > 0:
            idx1, idx2 = idx_b, idx_a
        else:
            return None
        s_pos = 1.0 / float(W2_row[idx1])
        s_neg = 1.0 / float(W2_row[idx2])
        tau = float(b2_i)
        w_o = W1[idx1] / s_pos
        b_o = tau + float(b1[idx1]) / s_pos
        w_chk = W1[idx2] / s_neg
        b_chk = tau + float(b1[idx2]) / s_neg
        if not np.allclose(w_o, w_chk, atol=atol, rtol=rtol):
            return None
        if abs(b_o - b_chk) > atol + rtol * max(abs(b_o), abs(b_chk), 1.0):
            return None
        return idx1, idx2, w_o, b_o

    nodes = list(onnx_model.graph.node)
    new_nodes = []
    new_inits = []
    i = 0
    n_merged = 0

    while i < len(nodes):
        if i + 3 < len(nodes):
            A, B, C, D = nodes[i:i + 4]
            if (A.op_type == "Gemm" and B.op_type == "Relu"
                    and C.op_type == "Gemm" and D.op_type == "Relu"
                    and len(B.input) == 1 and B.input[0] == A.output[0]
                    and C.input[0] == B.output[0]
                    and len(D.input) == 1 and D.input[0] == C.output[0]):

                tA1, tB1, al1, be1 = _gemm_attrs(A)
                tA2, tB2, al2, be2 = _gemm_attrs(C)
                # The rewrite is derived for the common fully connected Gemm
                # convention: Y = X @ W.T + b.
                if not (tA1 == 0 and tB1 == 1 and tA2 == 0 and tB2 == 1
                        and al1 == 1.0 and be1 == 1.0
                        and al2 == 1.0 and be2 == 1.0):
                    new_nodes.append(nodes[i]); i += 1; continue

                W1 = _w(A.input[1]) if len(A.input) >= 2 else None
                W2 = _w(C.input[1]) if len(C.input) >= 2 else None
                if W1 is None or W2 is None or W1.ndim != 2 or W2.ndim != 2:
                    new_nodes.append(nodes[i]); i += 1; continue

                b1 = (_w(A.input[2]) if len(A.input) >= 3
                      else np.zeros(W1.shape[0], dtype=W1.dtype))
                b2 = (_w(C.input[2]) if len(C.input) >= 3
                      else np.zeros(W2.shape[0], dtype=W2.dtype))

                n_out, n_mid = W2.shape
                if W1.shape[0] != n_mid or n_mid != 2 * n_out:
                    new_nodes.append(nodes[i]); i += 1; continue

                pairs = []
                used = set()
                ok = True
                for k in range(n_out):
                    r = _identify(W2[k], b2[k], W1, b1)
                    if r is None:
                        ok = False; break
                    idx1, idx2, w_o, b_o = r
                    if idx1 in used or idx2 in used:
                        ok = False; break
                    used.add(idx1); used.add(idx2)
                    pairs.append((k, w_o, b_o))

                if not ok or len(used) != n_mid:
                    new_nodes.append(nodes[i]); i += 1; continue

                n_in = W1.shape[1]
                W_new = np.empty((n_out, n_in), dtype=W1.dtype)
                b_new = np.empty((n_out,), dtype=b1.dtype)
                for k, w_o, b_o in pairs:
                    W_new[k] = w_o
                    b_new[k] = b_o

                W_new_name = f"{A.name}/merged/W"
                b_new_name = f"{A.name}/merged/b"
                new_inits.append(nh.from_array(W_new, name=W_new_name))
                new_inits.append(nh.from_array(b_new, name=b_new_name))

                fused_gemm = onnx.helper.make_node(
                    "Gemm",
                    inputs=[A.input[0], W_new_name, b_new_name],
                    outputs=[C.output[0]],
                    name=f"{A.name}/merged",
                    alpha=1.0, beta=1.0, transA=0, transB=1,
                )
                fused_relu = onnx.helper.make_node(
                    "Relu",
                    inputs=[C.output[0]],
                    outputs=[D.output[0]],
                    name=f"{A.name}/merged/relu",
                )
                new_nodes.append(fused_gemm)
                new_nodes.append(fused_relu)
                i += 4
                n_merged += 1
                continue

        new_nodes.append(nodes[i])
        i += 1

    if n_merged > 0:
        del onnx_model.graph.node[:]
        onnx_model.graph.node.extend(new_nodes)
        onnx_model.graph.initializer.extend(new_inits)
        t_rewrite = time.time() - t_pass_start
        print(f"[optimize_relu_relationship] merged {n_merged} two-layer block(s) into one "
              f"in {t_rewrite*1000:.1f} ms")

    onnx_model.ir_version = onnx.helper.find_min_ir_version_for(
        list(onnx_model.opset_import))

    if n_merged > 0:
        original_model.ir_version = onnx.helper.find_min_ir_version_for(
            list(original_model.opset_import))
        inp = original_model.graph.input[0]
        shape = [d.dim_value if d.dim_value > 0 else 1
                 for d in inp.type.tensor_type.shape.dim]

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  
        sess_orig = ort.InferenceSession(
            original_model.SerializeToString(),
            sess_options=opts, providers=['CPUExecutionProvider'])
        sess_new = ort.InferenceSession(
            onnx_model.SerializeToString(),
            sess_options=opts, providers=['CPUExecutionProvider'])
        in_name = sess_orig.get_inputs()[0].name

        np.random.seed(0)
        max_diff = 0.0
        for _ in range(sanity_n):
            x = np.random.randn(*shape).astype(np.float32)
            y_orig = sess_orig.run(None, {in_name: x})[0]
            y_new = sess_new.run(None, {in_name: x})[0]
            d = float(np.max(np.abs(y_orig - y_new)))
            if d > max_diff:
                max_diff = d
        if max_diff >= 1e-5:
            raise RuntimeError(
                f"[optimize_relu_relationship] sanity check FAILED: max forward diff over "
                f"{sanity_n} random inputs = {max_diff:.3e} exceeds tolerance 1e-5; "
                f"the rewrite is not numerically equivalent to the "
                f"original model.")
        print(f"[optimize_relu_relationship] sanity check OK: "
              f"max forward diff over {sanity_n} random inputs = {max_diff:.3e} ")

    return onnx_model


if __name__ == '__main__':
    old_path = "../vnncomp2023_benchmarks/benchmarks/cgan/onnx/cGAN_imgSz32_nCh_1.onnx"
    onnx_model = onnx.load(old_path)
    npath = "test.onnx"
    onnx_optimization_flags = ["merge_gemm_reshape_bn", "merge_bn_reshape_gemm"]
    compress_onnx(onnx_model, old_path, npath, onnx_optimization_flags, True)
