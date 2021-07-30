import socket
import random
import pickle
import sys
import os
import time
import gc
import csv
import math
import torch
import numpy as np
from collections import defaultdict

from beta_CROWN_solver import LiRPAConvNet, set_mip_refine_timeout
from batch_branch_and_bound import relu_bab_parallel
from utils import *
from attack_pgd import attack_pgd
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import reduction_min, stop_criterion_min
from read_vnnlib import read_vnnlib_simple
from arguments import common_argparser
from nn4sys_verification import nn4sys_verification


def config_args():
    parser = common_argparser()
    parser.add_argument("--csv_name", type=str, default=None, help='name of .csv file')
    parser.add_argument("--onnx_path", type=str, default=None, help='path to .onnx file')
    parser.add_argument("--vnnlib_path", type=str, default=None, help='path to .vnnlib file')
    parser.add_argument("--results_file", type=str, default=None, help='path to results file')
    parser.add_argument("--data", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "ACASXU", "NN4SYS", "TEST"], help='dataset')
    parser.add_argument("--model", type=str, default="cresnet5_16_avg_bn", help='model name')
    parser.add_argument('--increase_TO', action='store_true', default=False, help='increase timeout when debugging')
    parser.add_argument('--pgd_order', choices=["before", "after", "skip"], default="after", help='Run PGD before/after incomplete verification, or skip it.')
    parser.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab", help='Complete verification verifier.')
    parser.add_argument('--no_incomplete', action='store_false', dest='incomplete', help='do not use init opt crown for all labels')
    args = parser.parse_args()

    set_mip_refine_timeout(args.mip_refine_timeout*args.timeout)
    return args


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
            args.decision_thresh = prop_rhs[vec.argmax()]
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
            args.decision_thresh = prop_rhs[0]
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


def incomplete_verifier(model_ori, data, norm, args, y, data_ub=None, data_lb=None):
    # LiRPA wrapper
    if y is not None:
        num_class = 10
        labels = torch.tensor([y]).long()
        # Building a spec for all target labels.
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        # Remove spec to self.
        c = (c[I].view(data.size(0), num_class - 1, num_class))
    else:
        c = None
    model = LiRPAConvNet(model_ori, y, None, solve_slope=args.solve_slope, device=args.device,
                in_size=data.shape, deterministic=args.deterministic, simplify=False,  c=c)
    print('Model prediction is:', model.net(data))                             
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    # global_ub, global_lb, _, _, _, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history
    _, global_lb, _, _, _, mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, _ = model.build_the_model(
            domain, x, 
            lr_init_alpha=args.lr_init_alpha, init_iteration=args.init_iteration, optimizer=args.optimizer,
            share_slopes=args.share_slopes, lr_decay=args.lr_decay,
            loss_reduction_func=args.loss_reduction_func, 
            stop_criterion_func=stop_criterion_min(args.decision_thresh))
    # print("initial opt crown bounds:", global_lb)

    if y is not None:
        # For the last layer, since we missed one label, we add them back here.
        assert lower_bounds[-1].size(0) == 1  # this function only handles batchsize = 1.
        lower_bounds, upper_bounds, global_lb = reshape_bounds(lower_bounds, upper_bounds, y, global_lb)
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)
    else:
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

    if global_lb.min() >= args.decision_thresh:
        print("verified with init bound!")
        return "safe-incomplete", global_lb, saved_bounds
    
    return "unknown", global_lb, saved_bounds


def mip(args, saved_bounds, y):

    lirpa_model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA = saved_bounds

    if args.complete_verifier == "mip":
        mip_global_lb, mip_status = lirpa_model.build_the_model_mip(lower_bounds, upper_bounds, args.timeout,
                        mip_multi_proc=args.mip_multi_proc, mip_threads=args.mip_threads)
        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(0)  # Missing batch dimension.
        print(f'MIP solved global bound={mip_global_lb}')
        if mip_global_lb.min()>=0:
            print("verified safe with init mip!")
            return "safe-mip", mip_global_lb, None, None
        elif 9 in mip_status:
            print("verified timeout with init mip!")
            return "unknown", mip_global_lb, lower_bounds, upper_bounds
        else:
            print("verified unsafe with init mip!")
            return "unsafe-mip", mip_global_lb, None, None

    elif args.complete_verifier == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = lirpa_model.FSB_score(lower_bounds, upper_bounds, mask, pre_relu_indices, lA)
        # branching_decision = choose_node_parallel_FSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                    # branching_candidates=branching_candidates, branching_reduceop=branching_reduceop, slopes=slopes, betas=betas, history=history)

        refined_lower_bounds, refined_upper_bounds = lirpa_model.build_the_model_mip_refine(lower_bounds, upper_bounds, score=score,
                    lr_decay=args.lr_decay, 
                    lr_init_alpha=args.lr_init_alpha,
                    loss_reduction_func=args.loss_reduction_func, 
                    stop_criterion_func=stop_criterion_min(1e-4),
                    mip_multi_proc=args.mip_multi_proc, mip_threads=args.mip_threads, 
                    mip_perneuron_refine_timeout=args.mip_perneuron_refine_timeout)
        lower_bounds, upper_bounds, global_lb = reshape_bounds(refined_lower_bounds, refined_upper_bounds, y)
        refined_global_lb = lower_bounds[-1]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())
        if refined_global_lb.min()>=0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", refined_global_lb, lower_bounds, upper_bounds

        return "unknown", refined_global_lb, lower_bounds, upper_bounds
    else:
        return "unknown", -float("inf"), lower_bounds, upper_bounds


def bab(model_ori, data, target, norm, args, y, data_ub=None, data_lb=None, lower_bounds=None, upper_bounds=None):
    # This will use the refined bounds if the complete verifier is "bab-refine".
    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, y, target, solve_slope=args.solve_slope, device=args.device, in_size=data.shape,
                         deterministic=args.deterministic, conv_mode=args.conv_mode)
    print('Model prediction is:', model.net(data))                             
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    # with torch.autograd.set_detect_anomaly(True):
    print('beta splits:', not args.no_beta)
    # TODO: do not use the name "lp_test"!
    args.lp_test = None
    min_lb, min_ub, glb_record, nb_states = relu_bab_parallel(model, domain, x, no_LP=True, args=args,
                        refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states


def pgd_attack(args, model_ori, x, vnnlib, max_eps, data_min, data_max):
    # start PGD filter
    only_target_attack = False
    if args.data in ["MNIST", "CIFAR"]:
        pidx_list = []
        for prop_mat, prop_rhs in vnnlib[0][1]:
            if len(prop_rhs) > 1:
                output = model_ori(x).detach().cpu().numpy().flatten()
                print(f'model output:', output)
                vec = prop_mat.dot(output)
                selected_prop = prop_mat[vec.argmax()]
                y = int(np.where(selected_prop == 1)[0])  # true label, whatever in target attack
                pidx = int(np.where(selected_prop == -1)[0])  # target label
                only_target_attack = True
            else:
                assert len(prop_mat) == 1
                y = np.where(prop_mat[0] == 1)[0]
                if len(y) != 0:
                    y = int(y)
                else:
                    y = None
                pidx = int(np.where(prop_mat[0] == -1)[0])  # target label
            if pidx == y:
                raise NotImplementedError
            pidx_list.append(pidx)

        print('##### PGD attack: True label: {}, Tested against: {} ######'.format(y, pidx_list))

        if only_target_attack:
            # Targeted attack PGD.
            perturbation = attack_pgd(model_ori, X=x, y=None, epsilon=float("inf"), alpha=max_eps,
                    attack_iters=500, num_restarts=100, upper_limit=data_max, lower_limit=data_min, multi_targeted=False, lr_decay=0.995, target=pidx_list[0])
        else:
            # Untargeted attack PGD.
            perturbation = attack_pgd(model_ori, X=x, y=torch.tensor([y], device=args.device), epsilon=float("inf"), alpha=max_eps / 4.0,
                    attack_iters=200, num_restarts=200, upper_limit=data_max, lower_limit=data_min, multi_targeted=True, lr_decay=0.99, target=None)

        attack_image = torch.max(torch.min(x + perturbation, data_max), data_min)
        assert (attack_image >= data_min).all()
        assert (attack_image <= data_max).all()
        # assert (attack_image-x).abs().max() <= eps_temp.max(), f"{(attack_image-x).abs().max()} <= {eps_temp.max()}"
        attack_output = model_ori(attack_image).squeeze(0)
        attack_label = attack_output.argmax()
        print("pgd prediction:", attack_output)

        if only_target_attack:
            # Targeted attack, must have one label.
            if attack_label == pidx_list[0]:
                assert len(pidx_list) == 1
                print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
                return True
            else:
                attack_logit = attack_output.data[pidx_list[0]].item()
                attack_output.data[pidx_list[0]] = -float("inf")
                attack_margin = attack_output.max().item() - attack_logit
                print(f"targeted pgd failed, margin {attack_margin}")
                return False
        else:
            # Untargeted attack, any non-groundtruth label is ok.
            groundtruth_logit = attack_output.data[y].item()
            attack_output.data[y] = -float("inf")
            attack_margin = groundtruth_logit - attack_output
            print(f"attack margin {attack_margin}")            
            # Untargeted attack, any non-groundtruth label is ok.
            if attack_label != y:
                print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
                return True
            else:
                print("untargeted pgd failed")
                return False
    else:
        print("pgd attack not supported for dataset", args.data)
        raise NotImplementedError


# MODEL_SIZE_THRESHOLD = 2.4e6
MODEL_LAYER_THRESHOLD = 3
def auto_check(model, args):

    # if sum(p.numel() for p in model.parameters()) > MODEL_SIZE_THRESHOLD:
    #     # if the number of model parameters greater than the threshold
    #     print('args.incomplete change: {} -> {}'.format(args.incomplete, False))
    #     args.incomplete = False
    if sum(p.numel() for p in model.parameters()) < 15:
        # for TEST data only
        return

    if sum([type(m) == torch.nn.Sigmoid for m in list(model.modules())]) > 0:
        # if there is Sigmoid in model
        print('args.loss_reduction_func change: {} -> {}'.format(args.loss_reduction_func, 'min'))
        args.loss_reduction_func = 'min'
        print('args.init_iteration change: {} -> {}'.format(args.init_iteration, 1000))
        args.init_iteration = 1000
        print('args.lr_decay change: {} -> {}'.format(args.lr_decay, 0.999))
        args.lr_decay = 0.999
        print('args.pgd_order change: {} -> {}'.format(args.pgd_order, 'before'))
        args.pgd_order = 'before'
        print('args.complete_verifier change: {} -> {}'.format(args.complete_verifier, 'skip'))
        args.complete_verifier = 'skip'
        return

    if sum([type(m) == torch.nn.ReLU for m in list(model.modules())]) < MODEL_LAYER_THRESHOLD:
        # if the number of ReLU layers < 3
        print('args.complete_verifier change: {} -> {}'.format(args.complete_verifier, 'mip'))
        args.complete_verifier = 'mip'
        return


def main(args):
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data == 'MNIST':
        shape = (1, 1, 28, 28)
    elif args.data == 'CIFAR':
        shape = (1, 3, 32, 32)
    elif args.data == 'ACASXU':
        shape = (1, 5)
    elif args.data in ['NN4SYS', 'TEST']:
        shape = (1, 1)
    else:
        raise NotImplementedError

    if args.csv_name is not None:
        file_root = args.load
        with open(os.path.join(file_root, args.csv_name), newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            csv_file = []
            for row in reader:
                csv_file.append(row)

        save_path = 'vnn-comp_[{}]_start={}_end={}_iter={}_b={}_int-beta={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npz'. \
            format(os.path.splitext(args.csv_name)[0], args.start, args.end, args.iteration, args.batch_size,
                   args.opt_intermediate_beta, args.timeout, args.branching_method, args.branching_reduceop, args.branching_candidates, args.lr_init_alpha, args.lr_alpha, args.lr_beta, args.pgd_order)
        print(f'saving results to {save_path}')

        args.end = min(args.end, reader.line_num)
        if args.start != 0 or args.end != reader.line_num:
            assert args.start>=0 and args.start<=reader.line_num and args.end>args.start,\
                "start or end sample error: {}, {}, {}".format(args.end, args.start, reader.line_num)
            print("customized start/end sample from {} to {}".format(args.start, args.end))
        else:
            print("no customized start/end sample, testing for all samples")
            args.start, args.end = 0, reader.line_num
    else:
        # run in .sh
        args.start, args.end = 0, 1
        csv_file = [(args.onnx_path, args.vnnlib_path, args.timeout)]
        save_path = args.results_file
        file_root = ''

    verification_summary = defaultdict(list)
    time_per_sample_list = []
    status_per_sample_list = []
    bab_ret = []
    cnt = 0  # Number of examples in this run.
    bnb_ids = csv_file[args.start:args.end]

    for new_idx, csv_item in enumerate(bnb_ids):
        time_per_sample = time.time()
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        onnx_path, vnnlib_path, args.timeout = csv_item
        args.timeout = int(float(args.timeout))

        if args.increase_TO:
            args.timeout = args.timeout/6*10  # debug only
            print('Debug only!!!! We increase the timeout to', args.timeout)
        timeout_threshold = args.timeout  # In case args.timeout is changed later.

        # Convert ONNX model and read specifications.
        is_channel_last = False
        if args.data == 'MNIST':
            model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(1,28,28))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 784, 10)
        elif args.data == 'CIFAR':
            model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(3,32,32))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 3072, 10)
        elif args.data == 'ACASXU':
            model_ori = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(5,))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 5, 5)
            model_ori = nn.Sequential(*list(model_ori.modules())[1:])
        elif args.data == 'TEST':
            model_ori = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(1,))
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 1, 1)
            model_ori = convert_test_model(model_ori)
        elif args.data == 'NN4SYS':
            from convert_nn4sys_model import convert_and_save_nn4sys, get_path
            path = get_path(os.path.join(file_root, onnx_path))
            if not os.path.exists(path):
                convert_and_save_nn4sys(os.path.join(file_root, onnx_path))
            # load pre-converted model
            model_ori = torch.load(path)
            print(f'Loaded from {path}')
            vnnlib = read_vnnlib_simple(os.path.join(file_root, vnnlib_path), 1, 1, regression=True)

        if args.data == 'NN4SYS':
            res = nn4sys_verification(model_ori, vnnlib, args, onnx_path=os.path.join(file_root, onnx_path))
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
            auto_check(model_ori, args)

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
            model_ori, x, data_max, data_min = model_ori.to(args.device), x.to(args.device), data_max.to(args.device), data_min.to(args.device)

            verified_status = "unknown"
            verified_success = False

            if args.pgd_order == "before" and pgd_attack(args, model_ori, x, vnnlib, max_eps, data_min, data_max):
                # Attack success.
                verified_status = "unsafe-pgd"
                verified_success = True
            # continue  # uncomment for checking pgd attacking results

            # Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.
            if not verified_success and (args.incomplete or args.complete_verifier == "bab-refine"):
                y, pidx_list = get_labels(model_ori, x, vnnlib)
                verified_status, init_global_lb, saved_bounds = incomplete_verifier(model_ori, x, args.norm,
                            args, y, data_ub=data_max, data_lb=data_min)
                verified_success = verified_status != "unknown"
                lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
            # continue  # uncomment for checking opt crown initial results

            if not verified_success and (args.pgd_order == "after" and pgd_attack(args, model_ori, x, vnnlib, max_eps, data_min, data_max)):
                # Attack success.
                verified_status = "unsafe-pgd"
                verified_success = True

            # MIP or MIP refined bounds.
            if not verified_success and (args.complete_verifier == "mip" or args.complete_verifier == "bab-refine"):
                verified_status, init_global_lb, lower_bounds, upper_bounds = mip(args, saved_bounds=saved_bounds, y=y)
                verified_success = verified_status != "unknown"

            # BaB bounds.
            if not verified_success and args.complete_verifier != "skip":
                for prop_mat, prop_rhs in vnnlib[0][1]:
                    if len(prop_rhs) > 1:
                        # Check model output under input x.
                        output = model_ori(x).detach().cpu().numpy().flatten()
                        vec = prop_mat.dot(output)
                        if np.all(vec <= prop_rhs):
                            verified_status = "unsafe-clean"
                            verified_success = True
                            break
                        # we only verify the easiest one
                        selected_prop = prop_mat[vec.argmax()]
                        y = int(np.where(selected_prop == 1)[0])  # true label
                        pidx = int(np.where(selected_prop == -1)[0])  # target label
                        args.decision_thresh = prop_rhs[vec.argmax()]
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
                        args.decision_thresh = prop_rhs[0]

                    print('##### [{}] True label: {}, Tested against: {}, onnx_path: {}, vnnlib_path: {} ######'.format(new_idx, y, pidx, onnx_path, vnnlib_path))
                    if pidx == y:
                        raise NotImplementedError

                    torch.cuda.empty_cache()
                    gc.collect()
                    # print(psutil.virtual_memory())

                    # Complete verification (BaB, BaB with refine, or MIP).
                    start = time.time()
                    if args.incomplete:
                        # Reuse results from incomplete results, or from refined MIPs.
                        # skip the prop that already verified
                        rlb, rub = list(lower_bounds), list(upper_bounds)
                        rlb[-1] = rlb[-1][0, pidx]
                        rub[-1] = rub[-1][0, pidx]
                        if init_global_lb[0, pidx] >= args.decision_thresh:
                            print(f"init opt crown verified for label {pidx} with bound {init_global_lb[0, pidx]}")
                            l, nodes = rlb[-1].item(), 0
                        else:
                            # feed initialed bounds to save time
                            l, nodes = bab(model_ori, x, pidx, args.norm, args, y, data_ub=data_max, data_lb=data_min,
                                           lower_bounds=lower_bounds, upper_bounds=upper_bounds)
                    else:
                        assert args.complete_verifier == "bab"  # for MIP and BaB-Refine.
                        l, nodes = bab(model_ori, x, pidx, args.norm, args, y, data_ub=data_max, data_lb=data_min)
                    time_cost = time.time() - start
                    print('Image {} against label {} verify end, Time cost: {}'.format(new_idx, pidx, time_cost))
                    bab_ret.append([new_idx, l, nodes, time_cost, pidx])
                    args.timeout -= time_cost  # total timeout - time_cost
                    if l < args.decision_thresh:
                        # timeout or all nodes are split. Break to run next sample save time if any label is not verified
                        break
                else:
                    # All target label verified.
                    verified_status = "safe-bab"
                    verified_success = True

        # Summarize results.
        if args.csv_name is None:
            ret_dict = {
                "unknown": "timeout",
                "unknown (timed out)": "timeout",
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
                if verified_status != 'unknown':
                    verified_status += ' (timed out)'
            verification_summary[verified_status].append(new_idx)
            status_per_sample_list.append(verified_status)
            time_per_sample_list.append(time.time() - time_per_sample)
            with open(save_path, "wb") as f:
                pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "time": time_per_sample_list, "bab_ret": bab_ret}, f)
            print(f"Result: {verified_status} in {time_per_sample_list[-1]:.4f} seconds")

    if args.csv_name is not None:
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
    args = config_args()
    main(args)
