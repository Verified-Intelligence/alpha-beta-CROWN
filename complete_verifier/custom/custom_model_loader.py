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
This file shows how to use a customized model loader, and it defines a
few customized loaders for VNN-COMP models as examples. See example
configuration files in the `exp_configs` folder, e.g.,
`exp_configs/vnncomp21/marabou_cifar10.yaml`.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import numpy_helper as nh
import onnx2pytorch
import arguments

from onnx_opt import create_initializer_tensor
from load_model import load_model_onnx
from loading import joint_optimization_with_onnx_and_vnnlib
from read_vnnlib import read_vnnlib
from model_defs import Step_carvana


def transpose_nhwc(model_ori, vnnlib, shape):
    """Transpose vnnlib when NHWC is used (marabou-cifar10)."""
    # Only sequential models are supported and there must be a transpose
    # layer in the beginning.
    assert isinstance(model_ori, nn.Sequential)
    assert 'Transpose' in str(type(model_ori._modules['0']))
    model_ori = nn.Sequential(*(list(model_ori._modules.values())[1:]))
    assert len(vnnlib) == 1 and len(vnnlib[0]) == 2
    x = torch.tensor(vnnlib[0][0])
    assert x.shape == (shape[1] * shape[2] * shape[3], 2)
    x = x.reshape(shape[2], shape[3], shape[1], 2).permute(2, 0, 1, 3)
    vnnlib[0] = (x.reshape(-1, 2).numpy().tolist(), vnnlib[0][1])
    print('Model converted to NCHW format:', model_ori)
    return model_ori, vnnlib


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
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


def transpose_linear_layers(model_ori):
    """
    For test models in VNNComp.
    It looks like `in_features` and `out_features` are in the wrong order  after converting the onnx model to pytorch model.
    Return swapped model.
    """

    modules = []
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            layer = nn.Linear(m.in_features, m.out_features)  # Fix a bug in onnx converter for test models.
            layer.weight.data = m.weight.data.to(torch.float)
            layer.bias.data = m.bias.data.to(torch.float) if m.bias is not None else torch.zeros_like(layer.bias.data)
            modules.append(layer)
        else:
            modules.append(m)
    model_ori = nn.Sequential(*modules)
    print('Linear layers are transposed in this model.')

    return model_ori


def convert_carvana_model_vnnlib(model, vnnlib, c_mode='naive'):
    """
    We provide three ways to convert vnnlib in Carvana.
    naive: return the raw vnnlib
    one_by_one: return the vnnlib which try to verify all properties one by one (similar with standard verified accuracy)
    together: return the vnnlib which try to verify all properties at the same time (universal perturbation)
    """
    gt = np.array(vnnlib[0][0][-31*47:])  # binary mask ground truth
    new_x = vnnlib[0][0][:3*31*47]  # real input image
    assert np.all((gt[:, 0] == gt[:, 1]))   # make sure the mask ground truth has no perturbation
    assert len(np.unique(gt)) == 2  # make sure the mask ground truth is binary

    new_c = []
    if c_mode == 'one_by_one':
        # generate a spec which try to verify all properties one by one (similar with standard cifar-10 verified accuracy)
        for idx, gt_i in enumerate(gt):
            this_c = np.zeros((1, 2914))  # 2914 = 2*31*47, output size
            if gt_i[0] == 0:  # dim 0 > dim 1
                this_c[0, idx] = 1  # ground truth idx
                this_c[0, idx + 1457] = -1  # target idx
            else:  # dim 0 < dim 1
                this_c[0, idx] = -1  # target idx
                this_c[0, idx + 1457] = 1  # ground truth idx
            new_c.append((this_c, np.array([0])))
        model = nn.Sequential(model, nn.Flatten())

    elif c_mode == 'together':
        # generate a spec which try to verify all properties together
        this_c = np.zeros((1457, 2914))  # 2914 = 2*31*47, output size
        for idx, gt_i in enumerate(gt):
            if gt_i[0] == 0:  # dim 0 > dim 1
                this_c[idx, idx] = 1  # ground truth idx
                this_c[idx, idx + 1457] = -1  # target idx
            else:  # dim 0 < dim 1
                this_c[idx, idx] = -1  # target idx
                this_c[idx, idx + 1457] = 1  # ground truth idx
        new_c.append((this_c, np.zeros([1457])))
        model = nn.Sequential(model, nn.Flatten())

    elif c_mode == 'naive':
        # original spec, count correct classified pixels
        new_c.append(vnnlib[0][1][0])
        model = Step_carvana(model, gt[:, 0])
    else:
        raise NotImplementedError

    new_vnnlib = [(new_x, new_c)]
    return model, new_vnnlib


def split_carvana(path):
    # upsample: Conv_7; Conv_82
    # simp: Conv_7; Conv_13

    model_name = path

    onnx_model = onnx.load(model_name)

    if ("simp" in model_name):
        st_node, ed_node = "Conv_7", "Conv_13"
    elif ("upsample" in model_name):
        st_node, ed_node = "Conv_7", "Conv_82"
    else:
        raise NotImplementedError

    new_initializers = []
    new_nodes = []

    initializers = {}
    for onnx_module in onnx_model.graph.initializer:
        initializers[onnx_module.name] = nh.to_array(onnx_module)

    trigger = False
    for i, node in enumerate(onnx_model.graph.node):
        if (node.name == st_node): trigger = True
        if (trigger == True):
            for init_name in node.input:
                if (init_name not in initializers): continue
                conv_initializer = create_initializer_tensor(
                    name=init_name,
                    tensor_array=initializers[init_name],
                    data_type=onnx.TensorProto.FLOAT
                )
                new_initializers.append(conv_initializer)
            new_nodes.append(node)
        if (node.name == ed_node): break

    model_input_name = new_nodes[0].input[0]
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [None, 3, 31, 47])
    model_output_name = new_nodes[-1].output[0]
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [None, 2, 31, 47])

    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name="SplitNet",
        inputs=[X],
        outputs=[Y],
        initializer=onnx_model.graph.initializer
    )

    model_def = onnx.helper.make_model(new_graph, producer_name="onnx_example")

    model_def.opset_import[0].version = 12
    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)
    new_model_name = model_name[:-5] + "_split.onnx"
    onnx.save(model_def, new_model_name)
    print('New Carvana model saved to', new_model_name)


def customized_NN4SYS_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized NN4SYS loader.
    We split the model into v1 and v2 models to resolve numerical issues
    """

    def get_path(onnx_file):
        return f'{os.path.basename(onnx_file)}.pt'

    def convert_and_save_nn4sys(onnx_file):
        model_ori, _ = load_model_onnx(onnx_file)
        model_ori = nn.Sequential(*list(model_ori._modules.values()))
        # Split the model into v1 and v2 models to resolve numerical issues
        modules_v1 = []
        modules_v2 = []
        stage = 1
        for m in model_ori._modules.values():
            if isinstance(m, nn.Linear):
                if m.weight.abs().max() > 1e9:
                    stage = 2 if len(modules_v2) == 0 else 3
                    continue
            else:
                continue
            if stage == 1:
                modules_v1 += [m, nn.ReLU(inplace=True)]
            elif stage == 2:
                dim = modules_v1[-2].out_features - 1
                lin = nn.Linear(m.in_features - dim, m.out_features - dim)
                lin.weight.data = m.weight[:lin.out_features, :lin.in_features]
                lin.weight.data = lin.weight.to(dtype=torch.float64)
                lin.bias.data = m.bias[:lin.out_features]
                lin.bias.data = lin.bias.to(dtype=torch.float64)
                modules_v2 += [lin, nn.ReLU(inplace=True)]
        x = torch.tensor([[119740.8]], dtype=torch.float64)
        modules_v1 = modules_v1[:-1]
        model_v1 = nn.Sequential(*modules_v1)
        y = model_v1(x)
        dim = y.size(-1) - 1
        modules_v2 = modules_v2[:-1]
        linear_ident = nn.Linear(1, dim, bias=False)
        linear_ident.weight.data = torch.ones_like(linear_ident.weight, dtype=torch.float64)
        modules_v2.insert(0, linear_ident)
        model_v2 = nn.Sequential(*modules_v2)
        y[:, :-2] *= (y[:, 1:-1] <= 0).int()
        select = (y[:, :-1] > 0).int()
        y2 = model_v2(x)
        y2 = y2[:] * select
        res = y2.sum(dim=-1, keepdim=True)
        res_ref = model_ori(x)
        model_ori = (model_v1, model_v2, model_ori)
        name = get_path(onnx_file)
        torch.save(model_ori, name)
        print(f'Converted model saved to {name}')

    shape = (-1, 1)
    path = get_path(os.path.join(file_root, onnx_path))
    if not os.path.exists(path):
        convert_and_save_nn4sys(os.path.join(file_root, onnx_path))
    # load pre-converted model
    model_ori = torch.load(path)
    print(f'Loaded from {path}')
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path), regression=True)
    return model_ori, shape, vnnlib


def customized_Carvana_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized Carvana loader.
    We split the model for verification necessarily part only.
    """
    shape = (-1, 3, 31, 47)
    path = os.path.join(file_root, onnx_path[:-5] + '_split.onnx')
    if not os.path.exists(path):
        print('Split carvana model from:', os.path.join(file_root, onnx_path))
        split_carvana(os.path.join(file_root, onnx_path))
    else:
        print(f'Loaded split model from {path}')
    model_ori, _ = load_model_onnx(path)
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
    model_ori, vnnlib = convert_carvana_model_vnnlib(model_ori, vnnlib, c_mode='naive')
    return model_ori, shape, vnnlib


def customized_Marabou_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized Marabou loader.
    Transpose Conv layers from NHWC to NCHW format.
    Also transpose vnnlib in the same way.
    """
    shape = (-1, 3, 32, 32)
    model_ori, _ = load_model_onnx(os.path.join(file_root, onnx_path))
    conv_c, conv_h, conv_w = shape[1:]
    modules = list(model_ori.modules())[1:]
    new_modules = []
    for mi, m in enumerate(modules):
        if isinstance(m, torch.nn.Conv2d):
            # Infer the output size of conv.
            conv_h, conv_w = conv_output_shape((conv_h, conv_w), m.kernel_size, m.stride, m.padding)
            # conv_c = m.weight.size(0)
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
        elif isinstance(m, torch.nn.ReLU) and mi == (len(modules)-1):
            # not add relu if last layer is relu
            pass
        else:
            new_modules.append(m)
    model_ori = nn.Sequential(*new_modules)
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
    # Transpose vnnlib when NHWC is used (marabou-cifar10).
    model_ori, vnnlib = transpose_nhwc(model_ori, vnnlib, shape)
    return model_ori, shape, vnnlib


def customized_Gtrsb_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized GTRSB loader.
    Transpose vnnlib from NHWC to NCHW format.
    """
    model_ori, onnx_shape = load_model_onnx(os.path.join(file_root, onnx_path))
    shape = (1, *onnx_shape)
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
    x = torch.tensor(vnnlib[0][0])
    x = x.reshape(shape[2], shape[3], shape[1], 2).permute(2, 0, 1, 3)
    vnnlib[0] = (x.reshape(-1, 2).numpy().tolist(), vnnlib[0][1])
    model_ori, vnnlib = joint_optimization_with_onnx_and_vnnlib(model_ori, shape, vnnlib,
                                                                arguments.Config['model']['onnx_vnnlib_joint_optimization_flags'])
    return model_ori, shape, vnnlib


def customized_TEST_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized test model loader in VNNComp.
    The linear layers in this model seems in the wrong order.
    """
    model_ori, onnx_shape = load_model_onnx(os.path.join(file_root, onnx_path))
    shape = (-1, *onnx_shape)  # add the batch dim to onnx_shape
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
    model_ori = transpose_linear_layers(model_ori)
    return model_ori, shape, vnnlib
