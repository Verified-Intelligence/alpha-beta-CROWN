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
import argparse
import socket
import random
import time
import gc
import csv
import arguments

from beta_CROWN_solver_input_split import LiRPAConvNet
from batch_branch_and_bound_input_split import relu_bab_parallel

from utils import *

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from read_vnnlib import read_vnnlib_simple

parser = argparse.ArgumentParser()


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--csv_name", type=str, default=None, help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).', hierarchy=h + ["csv_name"])
    arguments.Config.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.', hierarchy=h + ["onnx_path"])
    arguments.Config.add_argument("--vnnlib_path", type=str, default=None, help='Path to .vnnlib specification file.', hierarchy=h + ["vnnlib_path"])
    arguments.Config.add_argument("--results_file", type=str, default=None, help='Path to results file.', hierarchy=h + ["results_file"])
    arguments.Config.add_argument("--root_path", type=str, default=None, help='Root path of VNN-COMP benchmarks (VNN-COMP specific).', hierarchy=h + ["root_path"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="ACASXU", choices=["ACASXU"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])

    h = ["bab"]
    arguments.Config.add_argument('--solve_slope', action='store_true', default=False, help='Optimize slope/alpha in compute bounds.', hierarchy=h + ["solve_slope"])

    arguments.Config.parse_config()


def bab(model_ori, data, target, norm, eps, decision_thresh, y, prop_mat, prop_rhs, data_ub=None, data_lb=None,
        c=None, shape=None, all_prop=None):

    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, y, target, solve_slope=arguments.Config["bab"]["solve_slope"], device=arguments.Config["general"]["device"], in_size=data.shape,
                         simplify=True if c is None else False, c=c)
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=None, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.cat([data_lb, data_ub])
    min_lb, min_input, glb_record, nb_states = relu_bab_parallel(model, domain, x, batch=arguments.Config["solver"]["beta-crown"]["batch_size"],
                                                                 decision_thresh=decision_thresh,
                                                                 iteration=arguments.Config["solver"]["beta-crown"]["iteration"], shape=shape,
                                                                 timeout=arguments.Config["bab"]["timeout"], lr_alpha=arguments.Config["solver"]["beta-crown"]["lr_alpha"],
                                                                 lr_init_alpha=arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
                                                                 branching_candidates=arguments.Config["bab"]["branching"]["candidates"],
                                                                 share_slopes=arguments.Config["solver"]["alpha-crown"]["share_slopes"],
                                                                 branching_method=arguments.Config["bab"]["branching"]["method"],
                                                                 prop_mat=prop_mat, prop_rhs=prop_rhs, model_ori=model_ori,
                                                                 all_prop=all_prop)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states, min_input


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    if arguments.Config["general"]["device"] != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])

    shape = (1, 5)

    if arguments.Config["general"]["csv_name"] is not None:
        file_root = arguments.Config["general"]["root_path"] + '/'
        csv_file = open(file_root + arguments.Config["general"]["csv_name"], newline='')
        reader = csv.reader(csv_file, delimiter=',')
        csv_file = []
        for row in reader:
            csv_file.append(row)
        save_path = 'vnn-comp_{}_alpha01_beta_{}_005_iter20_b{}_start_{}_branching_{}_to_{}.npy'. \
            format(arguments.Config["general"]["csv_name"], False, arguments.Config["solver"]["beta-crown"]["batch_size"], arguments.Config["data"]["start"], 'input_split', arguments.Config["bab"]["timeout"])
        arguments.Config["data"]["end"] = min(arguments.Config["data"]["end"], reader.line_num)
    else:
        # run in .sh
        arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, 1
        csv_file = [(arguments.Config["general"]["onnx_path"], arguments.Config["general"]["vnnlib_path"],
                     arguments.Config["bab"]["timeout"])]
        save_path = arguments.Config["general"]["results_file"]
        file_root = ''

    ret = []
    verified_acc = len(csv_file)
    verified_ret = []
    cnt = 0
    min_input = None

    bnb_ids = csv_file[arguments.Config["data"]["start"]:arguments.Config["data"]["end"]]

    for new_idx, csv_item in enumerate(bnb_ids):
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        new_idx += arguments.Config["data"]["start"]
        onnx_path, vnnlib_path, arguments.Config["bab"]["timeout"] = csv_item
        arguments.Config["bab"]["timeout"] = int(arguments.Config["bab"]["timeout"])

        model_ori = load_model_onnx(file_root + onnx_path, input_shape=(5,))
        model_ori = nn.Sequential(*list(model_ori.modules())[1:])
        vnnlib = read_vnnlib_simple(file_root + vnnlib_path, 5, 5)
        vnnlib_shape = shape

        cnt += 1
        # print(model_ori(x))
        pidx_all_verified = True
        start0 = time.time()

        for vnn in vnnlib:
            if not pidx_all_verified: break

            x_range = torch.tensor(vnn[0])
            data_min = x_range[:, 0].reshape(vnnlib_shape)
            data_max = x_range[:, 1].reshape(vnnlib_shape)
            x = x_range.mean(1).reshape(vnnlib_shape)  # only the shape of x is important.

            eps_temp = 0.5 * (data_max - data_min).flatten(-2).mean(-1).reshape(1, -1, 1, 1)

            c = None
            for prop_mat, prop_rhs in vnn[1]:
                if len(prop_rhs) > 1:
                    # we try to verify all props
                    c = torch.tensor(prop_mat).unsqueeze(0).type_as(data_max).to(arguments.Config["general"]["device"])
                    y = np.where(prop_mat == 1)[1]  # true label
                    pidx = np.where(prop_mat == -1)[1]  # target label
                    assert len(np.unique(prop_rhs)) == 1
                    decision_thresh = prop_rhs[0]
                else:
                    assert len(prop_mat) == 1
                    y = np.where(prop_mat[0] == 1)[0]
                    if len(y) != 0:
                        y = int(y)
                    else:
                        y = None  # no true label
                    pidx = int(np.where(prop_mat[0] == -1)[0])  # target label
                    decision_thresh = prop_rhs[0]  # already flipped in read_vnnlib_simple()

                print('##### [{}] True label: {}, Tested against: {}, onnx_path: {}, vnnlib_path: {} ######'.format(
                    new_idx, y, pidx, onnx_path, vnnlib_path))

                torch.cuda.empty_cache()
                gc.collect()
                # print(psutil.virtual_memory())

                model_ori.to('cpu')

                start = time.time()
                # Main function to run verification
                l, nodes, min_input = bab(model_ori, x, pidx, arguments.Config["specification"]["norm"], eps_temp, decision_thresh, y, prop_mat, prop_rhs,
                                          data_ub=data_max, data_lb=data_min, c=c, shape=shape, all_prop=vnn[1])
                time_cost = time.time() - start
                print('Image {} against {} verify end, Time cost: {}'.format(new_idx, pidx, time_cost))
                ret.append([new_idx, l, nodes, time_cost, pidx])
                arguments.Config["bab"]["timeout"] -= time_cost  # total timeout - time_cost
                # np.save(save_path, np.array(ret))
                if l < decision_thresh or min_input is not None:
                    pidx_all_verified = False
                    # break to run next sample save time if any label is not verified
                    break
                print(ret)
                # print(verified_ret)
                # break

        if time.time() - start0 > int(csv_item[2]):
            # time out
            verified_acc -= 1
            verified_ret.append([new_idx+1, 'timeout'])
        elif min_input is not None:
            verified_acc -= 1
            # adv example found
            verified_ret.append([new_idx+1, 'SAT'])
        else:
            # all props verified
            verified_ret.append([new_idx+1, 'UNSAT'])

        if arguments.Config["general"]["csv_name"] is None:
            with open(save_path, "w") as file:
                file.write(verified_ret[-1][-1])

    if arguments.Config["general"]["csv_name"] is not None:
        # some results analysis
        np.set_printoptions(suppress=True)
        ret = np.array(ret)
        verified_ret = np.array(verified_ret)
        print(ret)
        print(verified_ret)
        if ret.size > 0:
            print('time mean: {}, branches mean: {}'.format(ret[:, 3].mean(), ret[:, 2].mean()))

        print("final verified acc: {}%[{}]".format(verified_acc / len(bnb_ids) * 100., len(bnb_ids)))

        print("Total verification count:", cnt, "total verified:", verified_acc)
        if ret.size > 0:
            print("mean time [total:{}]: {}".format(len(bnb_ids), ret[:, 3].sum() / float(len(bnb_ids))))
            print("mean time [cnt:{}]: {}".format(cnt, ret[:, 3].sum() / float(cnt)))


if __name__ == "__main__":
    config_args()
    main()
