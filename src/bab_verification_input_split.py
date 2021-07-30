import argparse
import socket
import random
import sys
import time
import gc
import csv

from beta_CROWN_solver_input_split import LiRPAConvNet
from batch_branch_and_bound_input_split import relu_bab_parallel

from utils import *

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from read_vnnlib import read_vnnlib_simple

parser = argparse.ArgumentParser()

parser.add_argument('--no_solve_slope', action='store_false', dest='solve_slope', help='do not optimize slope/alpha in compute bounds')
parser.add_argument("--load", type=str, default=None, help='Load pretrained model')
parser.add_argument("--csv_name", type=str, default=None, help='name of .csv file')
parser.add_argument("--onnx_path", type=str, default=None, help='path to .onnx file')
parser.add_argument("--vnnlib_path", type=str, default=None, help='path to .vnnlib file')
parser.add_argument("--results_file", type=str, default=None, help='path to results file')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "ACASXU", "TEST"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--model", type=str, default="cresnet5_16_avg_bn", help='model name')
parser.add_argument("--batch_size", type=int, default=500, help='batch size')
parser.add_argument("--max_subproblems_list", type=int, default=200000, help='max length of sub-problems list')
parser.add_argument("--timeout", type=float, default=360, help='timeout for one property')
parser.add_argument("--start", type=int, default=0, help='start from i-th property')
parser.add_argument("--end", type=int, default=1000, help='end with (i-1)-th property')
parser.add_argument("--lr_init_alpha", type=float, default=0.1, help='learning rate for relu slopes/alpha at initial')
parser.add_argument("--lr_alpha", type=float, default=0.01, help='learning rate for relu slopes/alpha')
parser.add_argument("--iteration", type=int, default=10, help='iteration of optimization bounds')
parser.add_argument("--conv_mode", default="patches", choices=["patches", "matrix"], help='conv mode in BoundedModule')
parser.add_argument("--deterministic", action='store_true', help='Run code in CUDA deterministic mode, slower performance but better reproducibility.')
parser.add_argument("--double_fp", action='store_true', help='Use double precision floating point. GPUs with good double precision support are needed (NVIDIA P100, V100, A100; AMD Radeon Instinc MI50, MI100)')
parser.add_argument("--share_slopes", action='store_true', help='When --per_neuron_alpha is True, use shared alpha.')
parser.add_argument('--increase_TO', action='store_true', default=False, help='increase timeout when debugging')
parser.add_argument('--single_prop', action='store_true', default=False, help='only verify single prop')
parser.add_argument("--branching_method", default="sb", choices=["sb", "input_grad", "naive"], help='branching method')
parser.add_argument("--branching_candidates", type=int, default=3, help='select topK candidate per layer when using input_grad')

args = parser.parse_args()


def bab(model_ori, data, target, norm, eps, decision_thresh, args, y, prop_mat, prop_rhs, data_ub=None, data_lb=None,
        c=None, shape=None, all_prop=None):

    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, y, target, solve_slope=args.solve_slope, device=args.device, in_size=data.shape,
                         simplify=True if c is None else False, c=c)
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=None, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.cat([data_lb, data_ub])
    min_lb, min_input, glb_record, nb_states = relu_bab_parallel(model, domain, x, batch=args.batch_size,
                                                                 decision_thresh=decision_thresh,
                                                                 max_subproblems_list=args.max_subproblems_list,
                                                                 iteration=args.iteration, shape=shape,
                                                                 timeout=args.timeout, lr_alpha=args.lr_alpha,
                                                                 lr_init_alpha=args.lr_init_alpha,
                                                                 branching_candidates=args.branching_candidates,
                                                                 share_slopes=args.share_slopes, branching_method=args.branching_method,
                                                                 prop_mat=prop_mat, prop_rhs=prop_rhs, model_ori=model_ori,
                                                                 all_prop=all_prop)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states, min_input


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
        file_root = args.load + '/'
        csv_file = open(file_root + args.csv_name, newline='')
        reader = csv.reader(csv_file, delimiter=',')
        csv_file = []
        for row in reader:
            csv_file.append(row)
        save_path = 'vnn-comp_{}_alpha01_beta_{}_005_iter20_b{}_start_{}_branching_{}_to_{}.npy'. \
            format(args.csv_name, False, args.batch_size, args.start, 'input_split', args.timeout)
        args.end = min(args.end, reader.line_num)
    else:
        # run in .sh
        args.start, args.end = 0, 1
        csv_file = [(args.onnx_path, args.vnnlib_path, args.timeout)]
        save_path = args.results_file
        file_root = ''

    ret = []
    verified_acc = len(csv_file)
    verified_ret = []
    cnt = 0
    min_input = None

    bnb_ids = csv_file[args.start:args.end]

    for new_idx, csv_item in enumerate(bnb_ids):
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        new_idx += args.start
        onnx_path, vnnlib_path, args.timeout = csv_item
        args.timeout = int(args.timeout)

        if args.increase_TO:
            args.timeout = args.timeout / 6 * 10  # debug only

        is_channel_last = False
        if args.data == 'MNIST':
            model_ori, is_channel_last = load_model_onnx(file_root + onnx_path, input_shape=(1, 28, 28))
            vnnlib = read_vnnlib_simple(file_root + vnnlib_path, 784, 10)
        elif args.data == 'CIFAR':
            model_ori, is_channel_last = load_model_onnx(file_root + onnx_path, input_shape=(3, 32, 32))
            vnnlib = read_vnnlib_simple(file_root + vnnlib_path, 3072, 10)
        elif args.data == 'ACASXU':
            model_ori = load_model_onnx(file_root + onnx_path, input_shape=(5,))
            model_ori = nn.Sequential(*list(model_ori.modules())[1:])
            vnnlib = read_vnnlib_simple(file_root + vnnlib_path, 5, 5)
        elif args.data == 'TEST':
            model_ori = load_model_onnx(file_root + onnx_path, input_shape=(1,))
            vnnlib = read_vnnlib_simple(file_root + vnnlib_path, 1, 1)
            model_ori = convert_test_model(model_ori)

        if is_channel_last:
            vnnlib_shape = shape[:1] + shape[2:] + shape[1:2]
            print(
                f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {vnnlib_shape}')
        else:
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

            if is_channel_last:
                # The VNNlib file has X in NHWC order. We always use NCHW order.
                data_min = data_min.permute(0, 3, 1, 2).contiguous()
                data_max = data_max.permute(0, 3, 1, 2).contiguous()
                x = x.permute(0, 3, 1, 2).contiguous()
            eps_temp = 0.5 * (data_max - data_min).flatten(-2).mean(-1).reshape(1, -1, 1, 1)

            c = None
            for prop_mat, prop_rhs in vnn[1]:
                if len(prop_rhs) > 1:
                    if args.single_prop:
                        # we only verify the easiest one
                        output = model_ori(x).detach().numpy().flatten()
                        print(output)
                        vec = prop_mat.dot(output)
                        # sat = np.all(vec <= prop_rhs)
                        selected_prop = prop_mat[vec.argmax()]  # select the closet one to verify
                        y = int(np.where(selected_prop == 1)[0])  # true label
                        pidx = int(np.where(selected_prop == -1)[0])  # target label
                        decision_thresh = prop_rhs[vec.argmax()]
                    else:
                        # we try to verify all props
                        c = torch.tensor(prop_mat).unsqueeze(0).type_as(data_max).to(args.device)  # .type_as(data_max)
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
                l, nodes, min_input = bab(model_ori, x, pidx, args.norm, eps_temp, decision_thresh, args, y, prop_mat, prop_rhs,
                                          data_ub=data_max, data_lb=data_min, c=c, shape=shape, all_prop=vnn[1])
                time_cost = time.time() - start
                print('Image {} against {} verify end, Time cost: {}'.format(new_idx, pidx, time_cost))
                ret.append([new_idx, l, nodes, time_cost, pidx])
                args.timeout -= time_cost  # total timeout - time_cost
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

        if args.csv_name is None:
            with open(save_path, "w") as file:
                file.write(verified_ret[-1][-1])

    if args.csv_name is not None:
        # some results analysis
        np.set_printoptions(suppress=True)
        ret = np.array(ret)
        verified_ret = np.array(verified_ret)
        print(ret)
        print(verified_ret)
        if ret.size > 0:
            print('time mean: {}, branches mean: {}, number of timeout: {}'. \
                  format(ret[:, 3].mean(), ret[:, 2].mean(), (ret[:, 1] < 0).sum()))

        # if args.mode == "verified-acc":
        print("final verified acc: {}%[{}]".format(verified_acc / len(bnb_ids) * 100., len(bnb_ids)))
        # np.save('Verified-ret_{}_{}_start{}_end{}_{}.npy'. \
        #         format(args.model, args.data, args.start, args.end, verified_acc), verified_ret)

        print("Total verification count:", cnt, "total verified:", verified_acc)
        if ret.size > 0:
            print("mean time [total:{}]: {}".format(len(bnb_ids), ret[:, 3].sum() / float(len(bnb_ids))))
            print("mean time [cnt:{}]: {}".format(cnt, ret[:, 3].sum() / float(cnt)))

        print(args)


if __name__ == "__main__":
    main(args)
