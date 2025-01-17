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
import os
import csv
import re
import torch
import numpy as np
from load_model import load_model, load_model_onnx, Customized  # pylint: disable=unused-import
from read_vnnlib import read_vnnlib
from data_utils import load_eran_dataset, load_sdp_dataset
from data_utils import load_sampled_dataset, load_generic_dataset
from data_utils import load_pkl_dataset
from utils import get_save_path, expand_path
from specifications import construct_vnnlib


def default_onnx_and_vnnlib_loader(file_root, onnx_path, vnnlib_path):
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
    model_ori, onnx_shape = load_model_onnx(
        os.path.join(file_root, onnx_path), x=torch.tensor(vnnlib[0][0])[:, 0])
    shape = (-1, *onnx_shape)  # add the batch dim to onnx_shape
    model_ori, vnnlib = joint_optimization_with_onnx_and_vnnlib(model_ori, shape, vnnlib,
                                                                arguments.Config['model']['onnx_vnnlib_joint_optimization_flags'])
    return model_ori, shape, vnnlib


# TODO (2/17/23): Move it `loading.py` after the cleanup.
def load_verification_dataset():
    spec = arguments.Config['specification']

    if arguments.Config["data"]["pkl_path"] is not None:
        dataset_method = load_pkl_dataset
    elif arguments.Config['data']['dataset'].startswith('Customized('):
        dataset_method = eval(arguments.Config["data"]["dataset"])  # pylint: disable=eval-used
    elif "ERAN" in arguments.Config["data"]["dataset"] or "MADRY" in arguments.Config["data"]["dataset"]:
        dataset_method = load_eran_dataset
    elif "SDP" in arguments.Config["data"]["dataset"]:
        dataset_method = load_sdp_dataset
    elif "SAMPLE" in arguments.Config["data"]["dataset"]:
        # Sampled datapoints (a small subset of MNIST/CIFAR), only for reproducing some paper results.
        dataset_method = load_sampled_dataset
    elif "CIFAR" in arguments.Config["data"]["dataset"] or "MNIST" in arguments.Config["data"]["dataset"]:
        dataset_method = load_generic_dataset
    else:
        raise NotImplementedError(
            'Dataset not supported in this file! '
            'Please customize load_verification_dataset() function in utils.py.')

    # FIXME (01/10/22): fully document customized data loader.
    # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
    # X is the data matrix in (batch, ...).
    # labels are the ground truth labels, a tensor of integers.
    # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
    # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
    # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
    # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
    # Target label is the targeted attack label; can be set to None.
    data_config = dataset_method(spec)
    if isinstance(data_config, dict):
        ret = data_config
    else:
        # TODO (2/17/2023) deprecate
        runnerup, target_labels = None, None
        if len(data_config) == 5:
            X, labels, data_max, data_min, eps_new = data_config
        elif len(data_config) == 6:
            X, labels, data_max, data_min, eps_new, runnerup = data_config
        elif len(data_config) == 7:
            X, labels, data_max, data_min, eps_new, runnerup, target_labels = data_config
        else:
            raise NotImplementedError("Data config types not correct!")
        ret = {
            'X': X, 'labels': labels, 'runnerup': runnerup,
            'data_max': data_max, 'data_min': data_min,
            'eps': eps_new, 'norm': arguments.Config['specification']['norm'],
            'target_label': target_labels
        }
    if 'labels' in ret:
        assert ret['X'].size(0) == ret['labels'].size(0), "batch size of X and labels should be the same!"
    if 'data_max' in ret and 'data_min' in ret:
        assert (ret['data_max'] - ret['data_min']).min() >= 0, "data_max should always be larger or equal to data_min!"
    if arguments.Config["specification"]["norm"] != np.inf:
        if isinstance(arguments.Config["data"]["std"], (list, tuple)):
            assert arguments.Config["data"]["std"].count(arguments.Config["data"]["std"][0]) == len(
                arguments.Config["data"]["std"]), ('For non-Linf norm, we support only 1-d eps (all channels with the same perturbation). '
                'If you have more complex, per-channel eps (e.g., an ellipsoid L2 perturbation, you can '
                'add the data normalization into part of the model.')
            arguments.Config["data"]["std"] = arguments.Config["data"]["std"][0]
        else:
             arguments.Config["data"]["std"] = float(arguments.Config["data"]["std"])
        # FIXME need to ensure all the values are equal in the tensor
        if isinstance(ret['eps'], torch.Tensor):
            ret['eps'] = ret['eps'][0, 0, 0, 0]  # only support eps as a scalar for non-Linf norm

    return ret


def load_model_and_vnnlib(file_root, csv_item):
    onnx_path, vnnlib_path, arguments.Config["bab"]["timeout"] = csv_item
    onnx_path = os.path.join(
        arguments.Config["model"]["onnx_path_prefix"], onnx_path.strip())
    vnnlib_path = os.path.join(
        arguments.Config["specification"]["vnnlib_path_prefix"],
        vnnlib_path.strip())
    print(f'Using onnx {onnx_path}')
    print(f'Using vnnlib {vnnlib_path}')
    model_ori, shape, vnnlib = eval(  # pylint: disable=eval-used
        arguments.Config["model"]["onnx_loader"]
    )(file_root, onnx_path, vnnlib_path)
    return model_ori, shape, vnnlib, onnx_path


def adhoc_tuning(data_min, data_max, model_ori):
    if 'vgg' in arguments.Config['general']['root_path']:
        perturbed = (data_max - data_min > 0).sum()
        print('Number of perturbed inputs:', int(perturbed))
        if perturbed > 10000:
            print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
            print('Setting arguments.Config["attack"]["pgd_order"] to "before"')
            arguments.Config['attack']['pgd_order'] = 'before'
        if perturbed > 100:
            print('Setting bound_prop_method to crown')
            arguments.Config['solver']['bound_prop_method'] = 'crown'

    if 'nn4sys' in arguments.Config['general']['root_path']:
        if data_max.shape == torch.Size([1, 1]):
            print(f'Enlarging initial_max_domains for model with input shape {data_max.shape}')
            arguments.Config['bab']['initial_max_domains'] = 100000

    if 'vit' in arguments.Config['general']['root_path']:
        num_matmul = len([item for item in model_ori.modules()
                      if 'MatMul' in str(type(item))])
        if num_matmul >= 6:
            print('Sharing alpha due to model size')
            arguments.Config['solver']['alpha-crown']['share_alphas'] = True


def parse_run_mode():
    """ parse running by vnnlib or customized data
     if using customized data, we convert them to vnnlib format
     """
    model_ori = vnnlib_all = shape = None
    file_root = expand_path(arguments.Config['general']['root_path'])
    if (arguments.Config['general']['csv_name']
            and arguments.Config['specification']['vnnlib_path'] is None):
        # A CSV filename is specified, and we will go over all models and specs in this csv file.
        # Used for running VNN-COMP benchmarks in batch mode.
        # In this case, vnnlib_path should not be specified, otherwise we will run only a single model/spec.
        run_mode = 'csv_file'
        csv_path = os.path.join(file_root, arguments.Config['general']['csv_name'])
        with open(csv_path, newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')
            # In VNN-COMP each line of the csv contains 3 elements: model, vnnlib, timeout
            csv_file = [row for row in reader]

        if len(csv_file[0]) == 1:
            # Each row contains only one item, which is the vnnlib spec. So we load and return the model only once here.
            # This case is used when we have a batch of vnnlib specs to verify with one model either pytorch or ONNX.
            model_ori = load_model()

        arguments.Config['data']['end'] = min(arguments.Config['data']['end'], reader.line_num)
        if arguments.Config['data']['start'] != 0 or arguments.Config['data']['end'] != reader.line_num:
            assert (0 <= arguments.Config['data']['start'] <= reader.line_num
                    and arguments.Config['data']['end'] > arguments.Config['data']['start']
            ), 'specified --start or --end out of range: start={}, end={}, total_in_csv={}'.format(
                arguments.Config['data']['end'],
                arguments.Config['data']['start'], reader.line_num)
            print('customized start/end sample from instance {} to {} in {}'.format(
                arguments.Config['data']['start'], arguments.Config['data']['end'],
                arguments.Config['general']['csv_name']))
        else:
            print('no customized start/end sample, testing all samples in {}'.format(
                arguments.Config['general']['csv_name']))
            arguments.Config['data']['start'], arguments.Config['data']['end'] = 0, reader.line_num
        example_idx_list = csv_file[arguments.Config['data']['start']:arguments.Config['data']['end']]
    elif arguments.Config['model']['onnx_path'] is not None and arguments.Config['specification']['vnnlib_path'] is not None:
        # A onnx file and a vnnlib file is specified, run this onnx file with vnnlib, ignore csv file.
        # Used for VNN-COMP in single instance mode, will be used in run_instance.sh
        run_mode = 'single_vnnlib'
        arguments.Config['data']['start'], arguments.Config['data']['end'] = 0, 1
        csv_file = [(arguments.Config['model']['onnx_path'],
                     arguments.Config['specification']['vnnlib_path'],
                     arguments.Config['bab']['timeout'])]
        file_root = ''
        example_idx_list = csv_file[arguments.Config['data']['start']:arguments.Config['data']['end']]
    elif not arguments.Config['general']['csv_name']:
        # No CSV specified, we will use specifications defined in yaml file.
        # This part replaces the old robustness_verifier.py interface.
        run_mode = 'customized_data'
        # Load Pytorch or ONNX model depends on the model path or onnx_path is given.
        model_ori = load_model(weights_loaded=True)
        if arguments.Config['specification']['vnnlib_path'] is None:
            verification_dataset = load_verification_dataset()
            X = verification_dataset['X']
            # X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = verification_dataset
            if arguments.Config['data']['data_idx_file'] is not None:
                # Go over a list of data indices.
                with open(arguments.Config['data']['data_idx_file']) as f:
                    example_idx_list = re.split(r'[;|,|\n|\s]+', f.read().strip())
                    example_idx_list = [int(b_id) for b_id in example_idx_list]
                    print(f'Example indices (total {len(example_idx_list)}): {example_idx_list}')
            else:
                # By default, we go over all data.
                example_idx_list = list(range(X.shape[0]))
            example_idx_list = example_idx_list[arguments.Config['data']['start']:arguments.Config['data']['end']]
            vnnlib_all = construct_vnnlib(verification_dataset, example_idx_list)
            # X, labels, runnerup, data_max, data_min, perturb_epsilon,
            # target_label, example_idx_list)
            shape = [-1] + list(X.shape[1:])
        else:
            # Using vnnlib specification (e.g., loading a pytorch model and use vnnlib to define general specification).
            example_idx_list = [0]
            vnnlib = read_vnnlib(arguments.Config['specification']['vnnlib_path'])
            assert arguments.Config['model']['input_shape'] is not None, 'vnnlib does not have shape information, please specify by --input_shape'
            shape = arguments.Config['model']['input_shape']
            vnnlib_all = [vnnlib]  # Only 1 vnnlib file.
    else:
        raise NotImplementedError

    if arguments.Config['general']['results_file']:
        save_path = arguments.Config['general']['results_file']
    else:
        save_path = get_save_path(
            csv=arguments.Config['general']['csv_name'] is not None)

    print(f'Internal results will be saved to {save_path}.')
    # FIXME_NOW: model_ori should not be handled in this function! Do it in the utility function that loads models for all cases.
    return run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, shape


def joint_optimization_with_onnx_and_vnnlib(model, shape, vnnlib, flags):
    for opt_flag in flags:
        if opt_flag == "none":
            continue
        elif opt_flag == "peel_off_last_softmax_layer":
            def inv_sigmoid(c, rhs, optype):
                nonzero_c = [c[0][i] for i in c[0].nonzero()[0]]
                if optype == 'Sigmoid':
                    if len(nonzero_c) == 1:
                        # [ 0 ... 0 k 0 ...] <> rhs <=> ksigmoid(x_i) <> rhs <=> k x_i <> k * sigmoid^(-1) (rhs/k)
                        np_rhs = np.array(rhs)
                        np_c = np.array(c).reshape(np_rhs.shape)
                        # y = 1/(1+exp(-x)) => x = ln ( y / (1-y) )
                        new_rhs = np.log((np_rhs / np_c) / (- (np_rhs / np_c) + 1.)) * np_c
                        return new_rhs.tolist()
                    if len(nonzero_c) == 2 and 1 in nonzero_c and -1 in nonzero_c and rhs[0] == 0:
                        # [0 .. 0 1 0 ... 0 -1 0 ...] <> 0 <=> sigmoid(x_i) - sigmoid(x_j) <> 0 <=> x_i - x_j <> 0
                        return rhs
                elif optype == 'Softmax':
                    # [0 .. 0 1 0 ... 0 -1 0 ...] <> 0 <=> softmax(x_i) - softmax(x_j) <> 0 <=> x_i - x_j <> 0
                    assert len(nonzero_c) == 2 and 1 in nonzero_c and -1 in nonzero_c and rhs[0] == 0, \
                        f"Unsupported (c,rhs) for removing Softmax: ({c}, {rhs})"
                    return rhs
                else:
                    raise NotImplementedError

            ori_output_nodename = model.output_names[0]
            ori_output_nodeidx = 0
            new_output_nodename = None
            removed_op_type = None
            for node in model.onnx_model.graph.node:
                if (node.op_type == 'Sigmoid' or node.op_type == 'Softmax') and node.output[0] == ori_output_nodename:
                    new_output_nodename = node.input[0]
                    removed_op_type = node.op_type
                    break
                ori_output_nodeidx += 1

            if new_output_nodename is not None:
                print(f'Last {removed_op_type} node found, peel it off')
                # delete from onnx model
                model.output_names = [new_output_nodename]
                del model.onnx_model.graph.node[ori_output_nodeidx]
                # update the rhs in vnnlib
                new_vnnlib = [(vlib[0], [(c, inv_sigmoid(c, rhs, removed_op_type)) for c, rhs in vlib[1]]) for vlib in vnnlib]
                vnnlib = new_vnnlib

    return model, vnnlib
