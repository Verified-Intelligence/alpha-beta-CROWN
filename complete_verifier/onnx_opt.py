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
"""Optimizing computation graph in onnx file."""
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import onnx.numpy_helper

import torch.nn.functional as F
from onnx import numpy_helper as nh


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
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


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


def compress_onnx(onnx_model, old_path, save_path, onnx_optimization_flags, debug=False):
    assert onnx_optimization_flags != 'none'
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
    #	print(onnx_module.name, nh.to_array(onnx_module).shape)

    # for i, node in enumerate(onnx_model.graph.node):
    #	print(node)

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
            name=f'linear_W',
            tensor_array=W3,
            data_type=onnx.TensorProto.FLOAT)

        new_initializers.append(cur_linear_W)

        cur_linear_b = create_initializer_tensor(
            name=f'linear_b',
            tensor_array=b3,
            data_type=onnx.TensorProto.FLOAT)

        new_initializers.append(cur_linear_b)

        cur_node = onnx.helper.make_node(
            name=f'linear_MatMul',
            op_type='Gemm',
            inputs=[onnx_model.graph.node[0].input[0], f'linear_W', f'linear_b'],
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

    else:
        started_node = 0

    skipped = {}
    const_var = {}
    added_const = {}

    def trace(node_name):
        if node_name in skipped:
            return trace(skipped[node_name])
        return node_name

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
                        tensor_array=W,
                        data_type=onnx.TensorProto.FLOAT)

                    conv_b_initializer = create_initializer_tensor(
                        name=f"conv{convcnt}_b",
                        tensor_array=b,
                        data_type=onnx.TensorProto.FLOAT
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
                    for old_init in node.input:
                        if old_init in initializers:
                            cur_init = create_initializer_tensor(
                                name=old_init,
                                tensor_array=initializers[old_init],
                                data_type=onnx.TensorProto.FLOAT)
                            new_initializers.append(cur_init)
                    triggered = False
            elif node.op_type == "BatchNormalization":
                if triggered == True:
                    triggered = False
                    continue
                new_nodes.append(node)
                source = node.output[0]
                for old_init in node.input:
                    if old_init in initializers:
                        cur_init = create_initializer_tensor(
                            name=old_init,
                            tensor_array=initializers[old_init],
                            data_type=onnx.TensorProto.FLOAT)
                        new_initializers.append(cur_init)
            else:
                new_nodes.append(node)
                source = node.output[0]
                for old_init in node.input:
                    if old_init in initializers:
                        cur_init = create_initializer_tensor(
                            name=old_init,
                            tensor_array=initializers[old_init],
                            data_type=onnx.TensorProto.FLOAT)
                        new_initializers.append(cur_init)
        elif "merge_linear" in onnx_optimization_flags:
            if node.op_type == "MatMul" or node.op_type == "Add":
                if node.input[-1] in initializers:
                    layer_mat = initializers[node.input[-1]]
                else:
                    new_nodes.append(node)
                    source = node.output[0]
                    for old_init in node.input:
                        if old_init in initializers:
                            cur_init = create_initializer_tensor(
                                name=old_init,
                                tensor_array=initializers[old_init],
                                data_type=onnx.TensorProto.FLOAT)
                            new_initializers.append(cur_init)
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
                            tensor_array=cur_W,
                            data_type=onnx.TensorProto.FLOAT)
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
                            tensor_array=cur_b,
                            data_type=onnx.TensorProto.FLOAT)

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
                for old_init in node.input:
                    if old_init in initializers:
                        cur_init = create_initializer_tensor(
                            name=old_init,
                            tensor_array=initializers[old_init],
                            data_type=onnx.TensorProto.FLOAT)
                        new_initializers.append(cur_init)

        else:
            for i in range(len(node.input)): node.input[i] = trace(node.input[i])
            new_nodes.append(node)
            source = node.output[0]
            for old_init in node.input:
                if old_init in initializers:
                    cur_init = create_initializer_tensor(
                        name=old_init,
                        tensor_array=initializers[old_init],
                        data_type=onnx.TensorProto.FLOAT)
                    new_initializers.append(cur_init)

    model_outputs = onnx_model.graph.output

    if "remove_relu_in_last_layer" in onnx_optimization_flags:
        if new_nodes[-1].op_type == "Relu":
            new_nodes = new_nodes[:-1]
            model_outputs[0].name = new_nodes[-1].output[0]
            print("Remove the relu in the last layer.")

    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name="CompressedNet",
        inputs=onnx_model.graph.input,
        outputs=model_outputs,
        initializer=new_initializers
    )
    model_def = onnx.helper.make_model(new_graph, producer_name="onnx_example")
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)

    # for i, node in enumerate(model_def.graph.node):
    #	print(node)

    onnx.checker.check_model(model_def)
    onnx.save(model_def, save_path)

    if debug:
        errors = np.zeros(100)
        for _ in range(100):
            inputs = torch.randn(place_holder).numpy()
            ort_sess = ort.InferenceSession(old_path)
            input_name = ort_sess.get_inputs()[0].name
            output_name = ort_sess.get_outputs()[0].name
            output1 = ort_sess.run([output_name], {input_name: inputs})

            ort_sess = ort.InferenceSession(save_path)
            input_name = ort_sess.get_inputs()[0].name
            output_name = ort_sess.get_outputs()[0].name
            output2 = ort_sess.run([output_name], {input_name: inputs})

        if "remove_relu_in_last_layer" in onnx_optimization_flags:
            output2 = np.array(output2).clip(min=0)

        errors[_] = np.sum(np.abs(np.array(output1) - np.array(output2)).reshape(-1))

        print(f"Compressed {len(onnx_model.graph.node) - len(model_def.graph.node)} Onnx nodes")
        print('Sum/mean of errors during compression: {}/{}'.format(np.sum(errors), np.mean(errors)))

    return model_def


if __name__ == '__main__':
    old_path = "benchmark/onnx/cifar_bias_field_0.onnx"
    onnx_model = onnx.load(old_path)
    npath = "./test"
    onnx_optimization_flags = "merge_gemm"
    compress_onnx(onnx_model, old_path, npath, onnx_optimization_flags, True)
