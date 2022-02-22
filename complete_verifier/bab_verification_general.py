#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""alpha-beta-CROWN verifier with interface to handle vnnlib specifications used in VNN-COMP (except for ACASXu and nn4sys)."""

import socket
import random
import pickle
import os
import time
import gc
import csv
import torch
import numpy as np
from collections import defaultdict

import arguments
from beta_CROWN_solver import LiRPAConvNet
from lp_mip_solver import FSB_score
from batch_branch_and_bound import relu_bab_parallel
from utils import load_model_onnx, convert_test_model
from attack_pgd import pgd_attack
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import reduction_min, stop_criterion_min
from read_vnnlib import read_vnnlib_simple
from nn4sys_verification import nn4sys_verification

def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    arguments.Config.add_argument("--csv_name", type=str, default=None, help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).', hierarchy=h + ["csv_name"])
    arguments.Config.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.', hierarchy=h + ["onnx_path"])
    arguments.Config.add_argument("--vnnlib_path", type=str, default=None, help='Path to .vnnlib specification file.', hierarchy=h + ["vnnlib_path"])
    arguments.Config.add_argument("--results_file", type=str, default=None, help='Path to results file.', hierarchy=h + ["results_file"])
    arguments.Config.add_argument("--root_path", type=str, default=None, help='Root path of VNN-COMP benchmarks (VNN-COMP specific).', hierarchy=h + ["root_path"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="mnist_9_200", help='Model name.', hierarchy=h + ["name"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "CIFAR_SDP_FULL", "CIFAR_RESNET", "CIFAR_SAMPLE", "MNIST_SAMPLE", "CIFAR_ERAN", "MNIST_ERAN",
                                 "MNIST_ERAN_UN", "MNIST_SDP", "MNIST_MADRY_UN", "CIFAR_SDP", "CIFAR_UN", "NN4SYS", "TEST"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])

    h = ["attack"]
    arguments.Config.add_argument("--mip_attack", action='store_true', help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.', hierarchy=h + ["enable_mip_attack"])
    arguments.Config.add_argument('--pgd_steps', type=int, default=100, help="Steps of PGD attack.", hierarchy=h + ["pgd_steps"])
    arguments.Config.add_argument('--pgd_restarts', type=int, default=30, help="Number of random PGD restarts.", hierarchy= h + ["pgd_restarts"])
    arguments.Config.add_argument('--no_pgd_early_stop', action='store_false', dest='pgd_early_stop', help="Early stop PGD when an adversarial example is found.", hierarchy=h + ["pgd_early_stop"])
    arguments.Config.add_argument('--pgd_lr_decay', type=float, default=0.99, help='Learning rate decay factor used in PGD attack.', hierarchy= h + ["pgd_lr_decay"])
    arguments.Config.add_argument('--pgd_alpha', type=str, default="auto", help='Step size of PGD attack. Default (auto) is epsilon/4.', hierarchy=h + ["pgd_alpha"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option. Do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()


def get_labels(model_ori, x, vnnlib):
    pidx_list = []
    for prop_mat, prop_rhs in vnnlib[0][1]:
        if len(prop_rhs) > 1:
            # we only verify the easiest one
            output = model_ori(x).detach().cpu().numpy().flatten()
            print(output)
            vec = prop_mat.dot(output)
            selected_prop = prop_mat[vec.argmax()]
            y = int(np.where(selected_prop == 1)[0])  # true label
            pidx = int(np.where(selected_prop == -1)[0])  # target label
            arguments.Config["bab"]["decision_thresh"] = prop_rhs[vec.argmax()]
        else:
            assert len(prop_mat) == 1
            y = np.where(prop_mat[0] == 1)[0]
            if len(y) != 0:
                y = int(y)
            else:
                y = None
            pidx = np.where(prop_mat[0] == -1)[0]  # target label
            pidx = int(pidx) if len(pidx) != 0 else None  # Fix constant specification with no target label.
            if y is not None and pidx is None: y, pidx = pidx, y  # Fix vnnlib with >= const property.
            arguments.Config["bab"]["decision_thresh"] = prop_rhs[0]
        if pidx == y:
            raise NotImplementedError
        pidx_list.append(pidx)
    return y, pidx_list


def reshape_bounds(lower_bounds, upper_bounds, y, global_lb=None):
    with torch.no_grad():
        last_lower_bounds = torch.zeros(size=(1, lower_bounds[-1].size(1)+1), dtype=lower_bounds[-1].dtype, device=lower_bounds[-1].device)
        last_upper_bounds = torch.zeros(size=(1, upper_bounds[-1].size(1)+1), dtype=upper_bounds[-1].dtype, device=upper_bounds[-1].device)
        last_lower_bounds[:, :y] = lower_bounds[-1][:, :y]
        last_lower_bounds[:, y+1:] = lower_bounds[-1][:, y:]
        last_upper_bounds[:, :y] = upper_bounds[-1][:, :y]
        last_upper_bounds[:, y+1:] = upper_bounds[-1][:, y:]
        lower_bounds[-1] = last_lower_bounds
        upper_bounds[-1] = last_upper_bounds
        if global_lb is not None:
            last_global_lb = torch.zeros(size=(1, global_lb.size(1)+1), dtype=global_lb.dtype, device=global_lb.device)
            last_global_lb[:, :y] = global_lb[:, :y]
            last_global_lb[:, y+1:] = global_lb[:, y:]
            global_lb = last_global_lb
    return lower_bounds, upper_bounds, global_lb


def incomplete_verifier(model_ori, data, y, data_ub=None, data_lb=None, eps=0.0):
    norm = arguments.Config["specification"]["norm"]
    # LiRPA wrapper
    num_outputs = arguments.Config["data"]["num_outputs"]
    if y is not None:
        labels = torch.tensor([y]).long()
        if num_outputs == 1:
            # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
            c = (float(y) - 0.5) * 2 * torch.ones(size=(data.size(0), 1, 1))
        else:
            # Building a spec for all target labels.
            c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
            I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
            # Remove spec to self.
            c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))
    else:
        c = None
    model = LiRPAConvNet(model_ori, y, None, device=arguments.Config["general"]["device"],
                in_size=data.shape, deterministic=arguments.Config["general"]["deterministic"], simplify=False, c=c, conv_mode=arguments.Config["general"]["conv_mode"])
    print('Model prediction is:', model.net(data))
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    # global_ub, global_lb, _, _, _, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history
    _, global_lb, _, _, _, mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = model.build_the_model(
            domain, x, stop_criterion_func=stop_criterion_min(arguments.Config["bab"]["decision_thresh"]))

    if global_lb.min() >= arguments.Config["bab"]["decision_thresh"]:
        print("verified with init bound!")
        return "safe-incomplete", None, None, None

    # Save the alpha variables during optimization. Here the batch size is 1.
    saved_slopes = defaultdict(dict)
    for m in model.net.relus:
        for spec_name, alpha in m.alpha.items():
            # each slope size is (2, spec, 1, *shape); batch size is 1.
            saved_slopes[m.name][spec_name] = alpha.detach().clone()

    if y is not None and num_outputs > 1:
        # For the last layer, since we missed one label, we add them back here.
        assert lower_bounds[-1].size(0) == 1  # this function only handles batchsize = 1.
        lower_bounds, upper_bounds, global_lb = reshape_bounds(lower_bounds, upper_bounds, y, global_lb)
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)
    else:
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

    return "unknown", global_lb, saved_bounds, saved_slopes


def mip(saved_bounds, y, labels_to_verify=None):

    lirpa_model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA = saved_bounds

    if arguments.Config["general"]["complete_verifier"] == "mip":
        mip_global_lb, mip_global_ub, mip_status = lirpa_model.build_the_model_mip(lower_bounds, upper_bounds, labels_to_verify=labels_to_verify)

        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(0)  # Missing batch dimension.
        if mip_global_ub.ndim == 1:
            mip_global_ub = mip_global_ub.unsqueeze(0)  # Missing batch dimension.
        print(f'MIP solved lower bound: {mip_global_lb}')
        print(f'MIP solved upper bound: {mip_global_ub}')
        
        verified_status = "safe-mip"
        # Batch size is always 1.
        for pidx in range(len(mip_status)):
            if mip_global_lb[0, pidx] >=0:
                # Lower bound > 0, verified.
                continue
            # Lower bound < 0, now check upper bound.
            if mip_global_ub[0, pidx] <=0:
                # Must be 2 cases: solved with adv example, or early terminate with adv example.
                assert mip_status[pidx] in [2, 15]
                print("verified unsafe-mip with init mip!")
                return "unsafe-mip", mip_global_lb, None, None
            # Lower bound < 0 and upper bound > 0, must be a timeout.
            assert mip_status[pidx] == 9 or mip_status[pidx] == -1, "should only be timeout for label pidx"
            verified_status = "unknown-mip"
        
        print(f"verified {verified_status} with init mip!")
        return verified_status, mip_global_lb, None, None

    elif arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = FSB_score(lirpa_model.net, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

        refined_lower_bounds, refined_upper_bounds = lirpa_model.build_the_model_mip_refine(lower_bounds, upper_bounds,
                            score=score, stop_criterion_func=stop_criterion_min(1e-4))
        if arguments.Config["data"]["num_outputs"] > 1:
            lower_bounds, upper_bounds, _ = reshape_bounds(refined_lower_bounds, refined_upper_bounds, y)
        else:
            lower_bounds, upper_bounds, = refined_lower_bounds, refined_upper_bounds
        refined_global_lb = lower_bounds[-1]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())
        if refined_global_lb.min()>=0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", refined_global_lb, lower_bounds, upper_bounds

        return "unknown", refined_global_lb, lower_bounds, upper_bounds
    else:
        return "unknown", -float("inf"), lower_bounds, upper_bounds


def bab(unwrapped_model, data, target, y, eps=None, data_ub=None, data_lb=None, lower_bounds=None, upper_bounds=None, reference_slopes=None, attack_images=None):
    norm = arguments.Config["specification"]["norm"]
    if arguments.Config["specification"]["type"] == 'lp':
        if norm == np.inf:
            if data_ub is None:
                data_ub = data + eps
                data_lb = data - eps
            elif eps is not None:
                data_ub = torch.min(data + eps, data_ub)
                data_lb = torch.max(data - eps, data_lb)
        else:
            data_ub = data_lb = data
            assert torch.unique(eps).numel() == 1  # For other norms, the eps must be the same for each channel.
            eps = torch.mean(eps).item()
    elif arguments.Config["specification"]["type"] == 'bound':
        assert norm == np.inf
        # Use data_lb and data_ub directly.
    else:
        raise ValueError(f'Unsupported perturbation type {arguments.Config["specification"]["type"]}')

    if arguments.Config["debug"]["lp_test"] not in ["LP_intermediate_refine", "MIP_intermediate_refine"]:
        arguments.Config["debug"]["lp_test"] = None

    num_outputs = arguments.Config["data"]["num_outputs"]
    # assert num_outputs > 1
    # FIXME (01/12/22): Clean up specification matrix generation.
    if y is not None:
        if num_outputs > 1:
            c = torch.zeros((1, 1, num_outputs), device=arguments.Config["general"]["device"])  # we only support c with shape of (1, 1, n)
            c[0, 0, y] = 1
            c[0, 0, target] = -1
        else:
            # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
            c = (float(y) - 0.5) * 2 * torch.ones(size=(1, 1, 1))
    else:
        # if there is no ture label, we only verify the target output
        c = torch.zeros((1, 1, num_outputs), device=arguments.Config["general"]["device"])  # we only support c with shape of (1, 1, n)
        c[0, 0, target] = -1

    # This will use the refined bounds if the complete verifier is "bab-refine".
    model = LiRPAConvNet(unwrapped_model, y, target, device=arguments.Config["general"]["device"], in_size=data.shape,
                         deterministic=arguments.Config["general"]["deterministic"], conv_mode=arguments.Config["general"]["conv_mode"], c=c)
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()
        # c = c.cuda()
    print('Model prediction is:', model.net(data))


    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    # with torch.autograd.set_detect_anomaly(True):
    min_lb, min_ub, glb_record, nb_states = relu_bab_parallel(model, domain, x, refined_lower_bounds=lower_bounds,
                                                              refined_upper_bounds=upper_bounds, reference_slopes=reference_slopes, attack_images=attack_images)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    if isinstance(min_ub, torch.Tensor):
        min_ub = min_ub.item()
    if min_ub is None:
        min_ub = float('inf')

    return min_lb, min_ub, nb_states, glb_record


# MODEL_SIZE_THRESHOLD = 2.4e6
MODEL_LAYER_THRESHOLD = 3
def update_parameters(model):

    # if sum(p.numel() for p in model.parameters()) > MODEL_SIZE_THRESHOLD:
    #     # if the number of model parameters greater than the threshold
    #     print('arguments.Config["general"]["enable_incomplete_verification"] change: {} -> {}'.format(arguments.Config["general"]["enable_incomplete_verification"], False))
    #     arguments.Config["general"]["enable_incomplete_verification"] = False
    if sum(p.numel() for p in model.parameters()) < 15:
        # for TEST data only
        return

    if sum([type(m) == torch.nn.Sigmoid for m in list(model.modules())]) > 0:
        # if there is Sigmoid in model
        print('arguments.Config["general"]["loss_reduction_func"] change: {} -> {}'.format(arguments.Config["general"]["loss_reduction_func"], 'min'))
        arguments.Config["general"]["loss_reduction_func"] = 'min'
        print('arguments.Config["solver"]["alpha-crown"]["iteration"] change: {} -> {}'.format(arguments.Config["solver"]["alpha-crown"]["iteration"], 1000))
        arguments.Config["solver"]["alpha-crown"]["iteration"] = 1000
        print('arguments.Config["solver"]["beta-crown"]["lr_decay"] change: {} -> {}'.format(arguments.Config["solver"]["beta-crown"]["lr_decay"], 0.999))
        arguments.Config["solver"]["beta-crown"]["lr_decay"] = 0.999
        if arguments.Config["attack"]["pgd_order"] != 'skip':
            # It may be set to "skip" when testing on unsafe examples,
            # to check whether verification and attack are contradictory.
            print('arguments.Config["attack"]["pgd_order"] change: {} -> {}'.format(arguments.Config["attack"]["pgd_order"], 'before'))
            arguments.Config["attack"]["pgd_order"] = 'before'
        print('arguments.Config["general"]["complete_verifier"] change: {} -> {}'.format(arguments.Config["general"]["complete_verifier"], 'skip'))
        arguments.Config["general"]["complete_verifier"] = 'skip'
        return

    if sum([type(m) == torch.nn.ReLU for m in list(model.modules())]) < MODEL_LAYER_THRESHOLD:
        # if the number of ReLU layers < 3
        print('arguments.Config["general"]["complete_verifier"] change: {} -> {}'.format(arguments.Config["general"]["complete_verifier"], 'mip'))
        arguments.Config["general"]["complete_verifier"] = 'mip'
        return


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    if arguments.Config["general"]["device"] != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if arguments.Config["data"]["dataset"] == 'MNIST':
        shape = (1, 1, 28, 28)
    elif arguments.Config["data"]["dataset"] == 'CIFAR':
        shape = (1, 3, 32, 32)
    elif arguments.Config["data"]["dataset"] == 'ACASXU':
        shape = (1, 5)
    elif arguments.Config["data"]["dataset"] in ['NN4SYS', 'TEST']:
        shape = (1, 1)
    else:
        raise NotImplementedError

    if arguments.Config["specification"]["norm"] != np.inf and arguments.Config["attack"]["pgd_order"] != "skip":
        print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
        arguments.Config["attack"]["pgd_order"] = "skip"

    if arguments.Config["general"]["csv_name"] is not None:
        file_root = arguments.Config["general"]["root_path"]
        
        with open(os.path.join(file_root, arguments.Config["general"]["csv_name"]), newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            csv_file = []
            for row in reader:
                csv_file.append(row)

        save_path = 'vnn-comp_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npz'. \
            format(os.path.splitext(arguments.Config["general"]["csv_name"])[0], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
                   arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
                   arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"])
        print(f'saving results to {save_path}')

        arguments.Config["data"]["end"] = min(arguments.Config["data"]["end"], reader.line_num)
        if arguments.Config["data"]["start"] != 0 or arguments.Config["data"]["end"] != reader.line_num:
            assert arguments.Config["data"]["start"]>=0 and arguments.Config["data"]["start"]<=reader.line_num and arguments.Config["data"]["end"]>arguments.Config["data"]["start"],\
                "start or end sample error: {}, {}, {}".format(arguments.Config["data"]["end"], arguments.Config["data"]["start"], reader.line_num)
            print("customized start/end sample from {} to {}".format(arguments.Config["data"]["start"], arguments.Config["data"]["end"]))
        else:
            print("no customized start/end sample, testing for all samples")
            arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, reader.line_num
    else:
        # run in .sh
        arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, 1
        csv_file = [(arguments.Config["general"]["onnx_path"], arguments.Config["general"]["vnnlib_path"], arguments.Config["bab"]["timeout"])]
        save_path = arguments.Config["general"]["results_file"]
        file_root = ''

    verification_summary = defaultdict(list)
    time_per_sample_list = []
    status_per_sample_list = []
    bab_ret = []
    cnt = 0  # Number of examples in this run.
    bnb_ids = csv_file[arguments.Config["data"]["start"]:arguments.Config["data"]["end"]]

    for new_idx, csv_item in enumerate(bnb_ids):
        time_per_sample = time.time()
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        onnx_path, vnnlib_path, arguments.Config["bab"]["timeout"] = csv_item
        arguments.Config["bab"]["timeout"] = int(float(arguments.Config["bab"]["timeout"]))

        timeout_threshold = arguments.Config["bab"]["timeout"]  # In case arguments.Config["bab"]["timeout"] is changed later.

        # Convert ONNX model and read specifications.
        is_channel_last = False
        if arguments.Config["data"]["dataset"] == 'MNIST':
            model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(1,28,28))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 784, 10)
        elif arguments.Config["data"]["dataset"] == 'CIFAR':
            model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(3,32,32))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 3072, 10)
        elif arguments.Config["data"]["dataset"] == 'ACASXU':
            model_ori = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(5,))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 5, 5)
            model_ori = torch.nn.Sequential(*list(model_ori.modules())[1:])
        elif arguments.Config["data"]["dataset"] == 'TEST':
            model_ori = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(1,))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 1, 1)
            model_ori = convert_test_model(model_ori)
        elif arguments.Config["data"]["dataset"] == 'NN4SYS':
            from convert_nn4sys_model import convert_and_save_nn4sys, get_path
            path = get_path(os.path.join(file_root, onnx_path))
            if not os.path.exists(path):
                convert_and_save_nn4sys(os.path.join(file_root, onnx_path))
            # load pre-converted model
            model_ori = torch.load(path)
            print(f'Loaded from {path}')
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 1, 1, regression=True)

        model_ori.eval()
        if arguments.Config["data"]["dataset"] == 'NN4SYS':
            res = nn4sys_verification(model_ori, vnnlib, onnx_path=os.path.join(file_root, onnx_path))
            print(res)
            if res == 'unsafe':
                verified_status = "unsafe"
            elif res == 'timeout':
                verified_status = "unknown"
            elif res == 'verified':
                verified_status = "safe"
            else:
                raise ValueError(res)
        else:
            # auto tune args according to model_ori
            update_parameters(model_ori)

            # All other models.
            if is_channel_last:
                vnnlib_shape = shape[:1] + shape[2:] + shape[1:2]
                print(f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {vnnlib_shape}')
            else:
                vnnlib_shape = shape
            x_range = torch.tensor(vnnlib[0][0])
            data_min = x_range[:, 0].reshape(vnnlib_shape)
            data_max = x_range[:, 1].reshape(vnnlib_shape)
            x = x_range.mean(1).reshape(vnnlib_shape)  # only the shape of x is important.
            if is_channel_last:
                # The VNNlib file has X in NHWC order. We always use NCHW order.
                data_min = data_min.permute(0, 3, 1, 2).contiguous()
                data_max = data_max.permute(0, 3, 1, 2).contiguous()
                x = x.permute(0, 3, 1, 2).contiguous()
            eps_temp = 0.5 * (data_max - data_min).flatten(-2).mean(-1).reshape(1, -1, 1, 1)
            max_eps = eps_temp.max().item()
            model_ori, x, data_max, data_min = model_ori.to(arguments.Config["general"]["device"]), x.to(arguments.Config["general"]["device"]), data_max.to(arguments.Config["general"]["device"]), data_min.to(arguments.Config["general"]["device"])

            verified_status = "unknown"
            verified_success = False

            if arguments.Config["attack"]["pgd_order"] == "before":
                attack_ret, attack_images, attack_margin = pgd_attack(arguments.Config["data"]["dataset"], model_ori, x, max_eps, data_min, data_max, vnnlib=vnnlib)
                del attack_images, attack_margin
                if attack_ret:
                    # Attack success.
                    verified_status = "unsafe-pgd"
                    verified_success = True
            # continue  # uncomment for checking pgd attacking results

            init_global_lb = saved_bounds = saved_slopes = None

            # Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.
            if not verified_success and (arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
                y, pidx_list = get_labels(model_ori, x, vnnlib)
                verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(model_ori, x, y, data_ub=data_max, data_lb=data_min)
                verified_success = verified_status != "unknown"
                if not verified_success:
                    lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
            # continue  # uncomment for checking opt crown initial results

            if not verified_success and arguments.Config["attack"]["pgd_order"] == "after":
                attack_ret, attack_images, attack_margin = pgd_attack(arguments.Config["data"]["dataset"], model_ori, x, max_eps, data_min, data_max, vnnlib=vnnlib)
                del attack_images, attack_margin
                if attack_ret:
                    # Attack success.
                    verified_status = "unsafe-pgd"
                    verified_success = True

            # MIP or MIP refined bounds.
            if not verified_success and (arguments.Config["general"]["complete_verifier"] == "mip" or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
                start_mip = time.time()
                verified_status, init_global_lb, lower_bounds, upper_bounds = mip(saved_bounds=saved_bounds, y=y)
                verified_success = verified_status != "unknown"
                arguments.Config["bab"]["timeout"] -= (time.time() - start_mip)
                print("time threshold left for bab:", arguments.Config["bab"]["timeout"])

            # BaB bounds. (not do bab if unknown by mip solver for now)
            if not verified_success and arguments.Config["general"]["complete_verifier"] != "skip" and verified_status != "unknown-mip":
                for prop_mat, prop_rhs in vnnlib[0][1]:
                    if len(prop_rhs) > 1:
                        # Multiple properties in a "and" clause (e.g., marabou-cifar10). Only need to verify one of the easiest properties.
                        select_using_verified_bounds = True
                        if select_using_verified_bounds:
                            # FIXME: we should make sure the incomplete bound is using the prop_mat as the C matrix!
                            selection_metric = init_global_lb.detach().clone().squeeze(0)
                            selection_metric[y] = float("-inf")  # This this the groundtruth label (0 in the margin bounds).
                            pidx = selection_metric.argmax().item()  # Choose the label with bound closest to 0.
                        else:
                            # Check model output under input x.
                            output = model_ori(x).detach().cpu().numpy().flatten()
                            vec = prop_mat.dot(output)
                            if np.all(vec <= prop_rhs):
                                verified_status = "unsafe-clean"
                                verified_success = True
                                break
                            # we only verify the easiest one, based on clean input margin.
                            selected_prop = prop_mat[vec.argmax()]
                            y = int(np.where(selected_prop == 1)[0])  # true label
                            pidx = int(np.where(selected_prop == -1)[0])  # target label
                            arguments.Config["bab"]["decision_thresh"] = prop_rhs[vec.argmax()]
                    else:
                        assert len(prop_mat) == 1
                        y = np.where(prop_mat[0] == 1)[0]
                        if len(y) != 0:
                            y = int(y)
                        else:
                            y = None
                        pidx = np.where(prop_mat[0] == -1)[0]  # target label
                        pidx = int(pidx) if len(pidx) != 0 else None  # Fix constant specification with no target label.
                        if y is not None and pidx is None:
                            y, pidx = pidx, y  # Fix vnnlib with >= const property.
                        arguments.Config["bab"]["decision_thresh"] = prop_rhs[0]

                    print('##### [{}] True label: {}, Tested against: {}, onnx_path: {}, vnnlib_path: {} ######'.format(new_idx, y, pidx, onnx_path, vnnlib_path))
                    if pidx == y:
                        raise NotImplementedError

                    torch.cuda.empty_cache()
                    gc.collect()
                    # print(psutil.virtual_memory())

                    # Complete verification (BaB, BaB with refine, or MIP).
                    start = time.time()
                    if arguments.Config["general"]["enable_incomplete_verification"]:
                        # Reuse results from incomplete results, or from refined MIPs.
                        # skip the prop that already verified
                        rlb, rub = list(lower_bounds), list(upper_bounds)
                        rlb[-1] = rlb[-1][0, pidx]
                        rub[-1] = rub[-1][0, pidx]
                        if init_global_lb[0, pidx] >= arguments.Config["bab"]["decision_thresh"]:
                            print(f"init opt crown verified for label {pidx} with bound {init_global_lb[0, pidx]}")
                            l, nodes = rlb[-1].item(), 0
                        else:
                            # feed initialed bounds to save time
                            l, u, nodes, _ = bab(model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min,
                                                 lower_bounds=lower_bounds, upper_bounds=upper_bounds, reference_slopes=saved_slopes)
                    else:
                        assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
                        l, u, nodes, _ = bab(model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min)
                    time_cost = time.time() - start
                    print('Image {} against label {} verification end, Time cost: {}'.format(new_idx, pidx, time_cost))
                    bab_ret.append([new_idx, l, nodes, time_cost, pidx])
                    arguments.Config["bab"]["timeout"] -= time_cost  # total timeout - time_cost
                    if l < arguments.Config["bab"]["decision_thresh"]:
                        # timeout or all nodes are split. Break to run next sample save time if any label is not verified
                        break
                else:
                    # All target label verified.
                    verified_status = "safe-bab"
                    verified_success = True

            del init_global_lb, saved_bounds, saved_slopes

        # Summarize results.
        if arguments.Config["general"]["csv_name"] is None:
            ret_dict = {
                "unknown": "timeout",
                "unknown (timed out)": "timeout",
                "unknown-mip": "timeout",
                "safe": "UNSAT",
                "safe-bab": "UNSAT",
                "safe-mip": "UNSAT",
                "safe-incomplete": "UNSAT",
                "safe-incomplete-refine": "UNSAT",
                "unsafe": "SAT",
                "unsafe-mip": "SAT",
                "unsafe-clean": "SAT",
                "unsafe-pgd": "SAT",
            }
            with open(save_path, "w") as file:
                file.write(ret_dict.get(verified_status, 'timeout'))
                file.flush()
        else:
            cnt += 1
            if time.time() - time_per_sample > timeout_threshold:
                if 'unknown' not in verified_status:
                    verified_status += ' (timed out)'
            verification_summary[verified_status].append(new_idx)
            status_per_sample_list.append(verified_status)
            time_per_sample_list.append(time.time() - time_per_sample)
            with open(save_path, "wb") as f:
                pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "time": time_per_sample_list, "bab_ret": bab_ret}, f)
            print(f"Result: {verified_status} in {time_per_sample_list[-1]:.4f} seconds")

    if arguments.Config["general"]["csv_name"] is not None:
        # Finished all examples.
        num_timeout = sum("unknown" in s for s in status_per_sample_list)
        num_verified = sum("safe" in s and "unsafe" not in s for s in status_per_sample_list)
        num_unsafe = sum("unsafe" in s for s in status_per_sample_list)
        # Print BaB results.
        np.set_printoptions(suppress=True)
        bab_ret = np.array(bab_ret)
        print('\n')
        print(bab_ret)

        with open(save_path, "wb") as f:
            pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "time": time_per_sample_list, "bab_ret": bab_ret}, f)

        print("############# Summary #############")
        print("Final verified acc: {}% [total {} examples]".format(num_verified / len(bnb_ids) * 100., len(bnb_ids)))
        print("Total verification count:", num_verified + num_unsafe + num_timeout, ", total verified safe:", num_verified,
              ", verified unsafe:", num_unsafe, ", timeout:", num_timeout)
        if bab_ret.size > 0:
            print("mean time [total:{}]: {}".format(len(bnb_ids), bab_ret[:, 3].sum() / float(len(bnb_ids))))
            print("mean time [cnt:{}]: {}".format(cnt, bab_ret[:, 3].sum() / float(cnt)))
        print("max time", np.max(time_per_sample_list))
        for k, v in verification_summary.items():
            print(f"{k} (total {len(v)}):", v)


if __name__ == "__main__":
    config_args()
    main()
