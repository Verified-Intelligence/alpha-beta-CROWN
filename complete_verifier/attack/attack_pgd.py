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

import math
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import arguments
import os
import subprocess
from load_model import Customized
import sys
from load_model import inference_onnx

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def clamp(X, lower_limit=None, upper_limit=None):
    if lower_limit is None and upper_limit is None:
        return X
    if lower_limit is not None:
        return torch.max(X, lower_limit)
    if upper_limit is not None:
        return torch.min(X, upper_limit)
    return torch.max(torch.min(X, upper_limit), lower_limit)

def check_and_save_cex(adv_example, adv_output, vnnlib, res_path, expected_verified_status):
    """
    adv_example: [batch_size, *input_shape]
    adv_output: [batch_size, *input_shape]
    expected_verified_status: <"unsafe-pgd", "unsafe-bab", "unsafe", ...> 
    """
    print('\nChecking and Saving Counterexample in check_and_save_cex')
    assert adv_example.size(0) == 1, f'The batch_size of adv_example should be 1, but get {adv_example.size(0)}.'
    assert adv_output.dim() == 2, f'The adv_output should be in the shape of (batch_size, logits), but get {adv_output.shape}.'
    assert vnnlib is not None, f'Cached vnnlib should be used to enable check specs conditions.'
    # take only one in batch
    adv_output = adv_output[0:1].detach()
    verified_status = expected_verified_status
    verified_success = True

    if arguments.Config['general']['save_adv_example']:
        if eval(arguments.Config['attack']['adv_verifier'])(adv_example, adv_output, vnnlib, 
                                                            arguments.Config['general']['verify_onnxruntime_output']):
            try:
                print('Saving counterexample to', os.path.abspath(res_path))
                eval(arguments.Config['attack']['adv_saver'])(adv_example, adv_output, res_path) 
                verified_status = expected_verified_status
                verified_success = True
            except Exception as e:
                print(str(e))
                print('save adv example failed')
                verified_status = 'unknown'
                verified_success = False
        else:
            verified_status = 'unknown'
            verified_success = False
                
    if arguments.Config['general']['eval_adv_example']:
        onnx_path = arguments.Config['model']['onnx_path']
        vnnlib_path = arguments.Config['specification']['vnnlib_path']

        onnx_path = os.path.join(os.getcwd(), onnx_path)
        vnnlib_path = os.path.join(os.getcwd(), vnnlib_path)
        current_dir = os.path.dirname(__file__)
        script_path = os.path.join(current_dir, '../', 'check_counterexample.py')

        try:
            subprocess.run([sys.executable, script_path, onnx_path, vnnlib_path, res_path], check=True)
        except subprocess.CalledProcessError:
            print('Unexpected error in checking adv example')
            
    if arguments.Config['general']['show_adv_example']:
        print('Adv example:')
        print(adv_example[0, 0])
    print()
    return verified_status, verified_success


def default_adv_saver(adv_example, adv_output, res_path):
    x = adv_example.view(-1).detach().cpu()
    adv_output = adv_output.detach().cpu().numpy()
    with open(res_path, 'w+') as f:
        # f.write("; Counterexample with prediction: {}\n".format(attack_label))
        # f.write("\n")

        # for some cases the onnx input shape has no batch dim. (cctsdb_yolo)
        input_dim = len(adv_example) if adv_example.ndim == 1 else np.prod(adv_example[0].shape)
        # for i in range(input_dim):
        #     f.write("(declare-const X_{} Real)\n".format(i))
        #
        # for i in range(adv_output.shape[1]):
        #     f.write("declare-const Y_{} Real)\n".format(i))

        # f.write("; Input assignment:\n")
        f.write("(")
        for i in range(input_dim):
            f.write("(X_{}  {})\n".format(i, x[i].item()))

        # f.write("\n")
        # f.write("; Output obtained:\n")
        for i in range(adv_output.shape[1]):
            if i == 0:
                f.write("(Y_{} {})".format(i, adv_output[0,i]))
            else:
                f.write("\n(Y_{} {})".format(i, adv_output[0,i]))
        f.write(")")
        f.flush()

    if arguments.Config["general"]["eval_adv_example"]:
        onnx_path = arguments.Config["model"]["onnx_path"]
        vnnlib_path = arguments.Config["specification"]["vnnlib_path"]

        onnx_path = os.path.join(os.getcwd(), onnx_path)
        vnnlib_path = os.path.join(os.getcwd(), vnnlib_path)
        script_path = os.path.join('/'.join(__file__.split('/')[:-1]), '../', 'check_counterexample.py')

        # print(onnx_path, vnnlib_path, script_path)
        try:
            subprocess.run([sys.executable, script_path, onnx_path, vnnlib_path, res_path], check=True)
        except subprocess.CalledProcessError:
            print('Unexpected error in checking adv example')

        '''
        ## generate the specifications from C matrix and rhs_mat
        violated_C = C_mat[violate_index[:,0], violate_index[:,1], violate_index[:,2]] # [num_vio, num_and_spec, output_dim]
        rhs_mat = rhs_mat[violate_index[:,0], violate_index[:,2]] # [num_vio_or, num_and_spec]
        f.write("; Violated output constraints:\n")
        f.write("(assert (or\n")

        for or_index, _or in enumerate(violated_C):
            f.write('(and ')
            for and_index, spec in enumerate(_or):
                # f.write('(<= ')
                y_list = []
                for index, factor in enumerate(spec):
                    if factor == 1:
                        y_list.append((1, index))
                        break
                for index, factor in enumerate(spec):
                    if factor == -1:
                        y_list.append((-1, index))
                        break
                if rhs_mat[or_index, and_index] != 0:
                    y_list.append((0, rhs_mat[or_index, and_index].item()))

                if y_list[0][0] == 1:
                    f.write('(<= ')
                else:
                    f.write('(>= ')

                for yy in y_list:
                    if yy[0] != 0:
                        f.write("Y_{} ".format(yy[1]))
                    else:
                        f.write(str(yy[1] * y_list[0][0]))

                f.write(')')

            f.write(')\n')
        f.write("))")
        '''


def process_vnn_lib_attack(vnnlib, x):
    list_target_label_arrays = [[]]
    data_min_repeat = []
    data_max_repeat = []

    for vnn in vnnlib:
        data_range = torch.Tensor(vnn[0])
        spec_num = len(vnn[1])

        data_max_ = data_range[:,1].view(-1, *x.shape[1:]).to(x.device).expand(spec_num, *x.shape[1:]).unsqueeze(0)
        data_min_ = data_range[:,0].view(-1, *x.shape[1:]).to(x.device).expand(spec_num, *x.shape[1:]).unsqueeze(0)

        data_max_repeat.append(data_max_)
        data_min_repeat.append(data_min_)

        list_target_label_arrays[0].extend(list(vnn[1]))

    data_min_repeat = torch.cat(data_min_repeat, dim=1)
    data_max_repeat = torch.cat(data_max_repeat, dim=1)

    return list_target_label_arrays, data_min_repeat, data_max_repeat


def attack(model_ori, x, vnnlib, verified_status, verified_success,
           crown_filtered_constraints=None, initialization='uniform'):
    GAMA_loss = False
    if 'auto_attack' not in arguments.Config["attack"]["attack_mode"]:
        if "diversed" in arguments.Config["attack"]["attack_mode"]:
            initialization = "osi"
        if "GAMA" in arguments.Config["attack"]["attack_mode"]:
            GAMA_loss = True

        # In this file, we only consider batch_size == 1
        assert x.shape[0] == 1

        list_target_label_arrays, data_min_repeat, data_max_repeat = process_vnn_lib_attack(vnnlib, x)
        # list_target_label_arrays: a list of list of tuples: there are [batch, num_mats] tuples.
        # Each tuple representing a OR clause, which is an (prop_mat, prop_rhs).
        # The number of AND statements are the number of rows or prop_mat and prop_rhs.

        # TODO check if list_target_label_arrays can exactly match constraints in model_ori.c one by one.

        # data_min/max_repeat: [batch_size, spec_num, *input_shape]
        # list_target_label_arrays: [batch_size, spec_num, C_mat, rhs_mat]

        if crown_filtered_constraints is not None:
            # For attacking specs that cannot be verified by CROWN.
            assert len(list_target_label_arrays) == 1  # only support batch_size=1 cases
            list_target_label_arrays_new = [[]]
            for i in range(len(list_target_label_arrays[0])):
                if crown_filtered_constraints[i]:
                    continue
                list_target_label_arrays_new[0].append(list_target_label_arrays[0][i])
            list_target_label_arrays = list_target_label_arrays_new
            print(f"Remain {len(list_target_label_arrays[0])} labels need to be attacked.")

        attack_function = eval(arguments.Config["attack"]["attack_func"])
        attack_ret, attack_images, attack_margins, all_adv_candidates = attack_function(
            model_ori, x, data_min_repeat[:, :len(list_target_label_arrays[0]), ...],
            data_max_repeat[:, :len(list_target_label_arrays[0]), ...], list_target_label_arrays,
            initialization=initialization, GAMA_loss=GAMA_loss)

    else:
        raise NotImplementedError('Auto-attack interface has not been implemented yet.')
        # attack_ret, attack_images, attack_margins = auto_attack(model_ori, x, data_min=data_min, data_max=data_max, vnnlib=vnnlib)

    if attack_ret:
        # Still need to compare adv example on pytorch and onnxruntime after attack and before saving
        # attack_images is duplicate num_spec times due to the duplicated data_max and data_min
        # attack_images has shape (batch, spec, c, h, w).
        attack_output = model_ori(attack_images.view(-1, *x.shape[1:]))
        verified_status, verified_success = check_and_save_cex(attack_images[:, 0:1].squeeze(1), # squeeze spec
                                                               attack_output, vnnlib,
                                                               arguments.Config["attack"]["cex_path"], "unsafe-pgd")
            
    print("verified_status", verified_status)
    print("verified_success", verified_success)

    return verified_status, verified_success, attack_images, attack_margins, all_adv_candidates


def default_pgd_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const,
                     gama_lambda=0, threshold=-1e-5, mode='hinge', model=None):
    '''
    output: [num_example, num_restarts, num_or_spec, num_output]
    C_mat: [num_example, num_restarts, num_spec, num_output]
    rhs_mat: [num_example, num_spec]
    cond_mat: [[]] * num_examples
    gama_lambda: weight factor for gama loss. If true, sum the loss and return the sum of loss
    threshold: the threshold for hinge loss
    same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.
    '''
    if same_number_const:
        C_mat = C_mat.view(C_mat.shape[0], 1, output.shape[2], -1, C_mat.shape[-1])
        # [num_example, 1, num_or_spec, num_and_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, output.shape[2], -1)
        loss = C_mat.matmul(output.unsqueeze(-1)).squeeze(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"]
        loss = torch.clamp(loss, min=threshold)
        # [num_example, num_restarts, num_or_spec, num_and_spec]
        loss = -loss
    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        if origin_out is not None:
            origin_out = origin_out.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        loss = (C_mat * output).sum(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"]
        loss = torch.clamp(loss, min=threshold)
        loss = -loss

    if origin_out is not None:
        loss_gamma = loss.sum() + (gama_lambda * (output - origin_out)**2).sum(dim=3).sum()
    else:
        loss_gamma = loss.sum()
    # [num_example, num_restarts, num_or_spec, num_and_spec]

    if mode == "sum":
        loss[loss >= 0] = 1.0
    # loss is returned for best loss selection, loss_gamma is for gradient descent.
    return loss, loss_gamma


def test_conditions(input, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max, data_min, return_success_idx=False):
    '''
    Whether the output satisfies the specifiction conditions.
    If the output satisfies the specification for adversarial examples, this function returns True, otherwise False.

    input: [num_example, num_restarts, num_or_spec, *input_shape]
    output: [num_example, num_restarts, num_or_spec, num_output]
    C_mat: [num_example, num_restarts, num_spec, num_output] or [num_example, num_spec, num_output]
    rhs_mat: [num_example, num_spec]
    cond_mat: [[]] * num_examples
    same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.
    data_max & data_min: [num_example, 1, num_spec, *input_shape]
    '''
    if same_number_const:
        C_mat = C_mat.view(C_mat.shape[0], 1, len(cond_mat[0]), -1, C_mat.shape[-1])
        # [batch_size, restarts, num_or_spec, num_and_spec, output_dim]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, len(cond_mat[0]), -1)

        # apply a small tolerance to rhs so that we are more confident about the adv example
        cond = torch.matmul(C_mat, output.unsqueeze(-1)).squeeze(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"]

        valid = ((input <= data_max) & (input >= data_min))

        valid = valid.reshape(*valid.shape[:3], -1)
        # [num_example, restarts, num_all_spec, output_dim]
        valid = valid.all(-1).view(valid.shape[0], valid.shape[1], len(cond_mat[0]), -1)
        # [num_example, restarts, num_or_spec, num_and_spec]

        res = ((cond.amax(dim=-1, keepdim=True) < 0.0) & valid).any(dim=-1).any(dim=-1).any(dim=-1)

        if res.all() and return_success_idx:
            # invalid examples will not be selected by torch.min, shape: [num_example, restarts, num_all_spec, output_dim]
            vio_value = cond.amax(dim=-1, keepdim=True) * valid
            # index of the adv example with the largest violation
            idx = int(torch.min(torch.min(vio_value, dim=1).values, dim=1).indices)
            return res, idx

    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        # apply a small tolerance to rhs so that we are more confident about the adv example
        cond = torch.clamp((C_mat * output).sum(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"], min=0.0)
        # [num_example, 1, num_spec]

        group_C = torch.zeros(len(cond_mat[0]), C_mat.shape[2], device=cond.device) # [num_or_spec, num_total_spec]
        x_index = []
        y_index = []
        index = 0

        for i, num_cond in enumerate(cond_mat[0]):
            x_index.extend([i] * num_cond)
            y_index.extend([index+j] for j in range(num_cond))
            index += num_cond

        group_C[x_index, y_index] = 1.0

        # loss shape: [batch_size, num_restarts, num_total_spec]
        cond = group_C.matmul(cond.unsqueeze(-1)).squeeze(-1)

        valid = ((input <= data_max) & (input >= data_min))
        valid = valid.view(*valid.shape[:3], -1)
        # [num_example, restarts, num_all_spec, output_dim]
        valid = valid.all(-1).view(valid.shape[0], valid.shape[1], len(cond_mat[0]), -1)
        # [num_example, restarts, num_or_spec, num_and_spec]

        valid = valid.all(-1)

        # [num_example, num_restarts, num_or_example]
        res = ((cond == 0.0) & valid).any(dim=-1).any(dim=-1)

        if res and return_success_idx:
            # invalid examples will not be selected by torch.min, shape: [num_example, restarts, num_all_spec]
            vio_value = ((C_mat * output).sum(-1) - rhs_mat) * (cond == 0.0) * valid
            # index of the adv example with the largest violation
            idx = int(torch.min(torch.min(vio_value, dim=1).values, dim=1).indices)
            return res, idx

    if return_success_idx:
        # just return a dummy index, won't be used
        return res, float('nan')

    return res


def default_early_stop_condition(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const,
    data_max, data_min, model, indices, num_or_spec, return_success_idx=False):

    return test_conditions(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const,
        data_max, data_min, return_success_idx)


def build_conditions(x, list_target_label_arrays):
    '''
    parse C_mat, rhs_mat from the target_label_arrays
    '''
    batch_size = x.shape[0]

    cond_mat = [[] for _ in range(batch_size)]
    C_mat = [[] for _ in range(batch_size)]
    rhs_mat = [[] for _ in range(batch_size)]

    same_number_const = True
    const_num = None
    for i in range(batch_size):
        target_label_arrays = list_target_label_arrays[i]
        for prop_mat, prop_rhs in target_label_arrays:
            C_mat[i].append(torch.Tensor(prop_mat).to(x.device))
            rhs_mat[i].append(torch.Tensor(prop_rhs).to(x.device))
            cond_mat[i].append(prop_rhs.shape[0]) # mark the `and` group
            if const_num is not None and prop_rhs.shape[0] != const_num:
                same_number_const = False
            else:
                const_num = prop_rhs.shape[0]

        C_mat[i] = torch.cat(C_mat[i], dim=0).unsqueeze(0)
        rhs_mat[i] = torch.cat(rhs_mat[i], dim=0).unsqueeze(0)

        # C: [1, num_spec, num_output]
    try:
        # try to stack the specs for a batch of examples
        # C: [num_example, num_spec, num_output]
        C_mat = torch.cat(C_mat, dim=0)
        rhs_mat = torch.cat(rhs_mat, dim=0)
    except (RuntimeError, ValueError):
        # failed when the examples have different number of specs
        print("Only support batches when the examples have the same number of constraints.")
        assert False
    # C shape: [num_example, num_spec, num_output]
    # rhs shape: [num_example, num_spec]
    # cond_mat shape: [num_example, num_spec]

    return C_mat, rhs_mat, cond_mat, same_number_const


def default_adv_example_finalizer(model_ori, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat):
    # x and best_deltas has shape (batch, c, h, w).
    # data_min and data_max have shape (batch, spec, c, h, w).
    attack_image = torch.max(torch.min((x + best_deltas).unsqueeze(1), data_max), data_min)
    assert (attack_image >= data_min).all()
    assert (attack_image <= data_max).all()

    attack_output = model_ori(attack_image.view(-1, *x.shape[1:])).view(*attack_image.shape[:2], -1)
    # [batch_size, num_or_spec, out_dim]

    if arguments.Config['general']['save_output']:
        arguments.Globals['out']['pred_adv'] = attack_output[0][0].cpu()

    # only print out the first two random start outputs of the first two examples.
    print("Adv example prediction (first 2 examples and 2 restarts):\n", attack_output[:2,:2])

    attack_output_repeat = attack_output.unsqueeze(1).repeat_interleave(torch.tensor(cond_mat[0]).to(x.device), dim=2)
    # [num_example, num_restarts, num_spec, num_output]

    C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
    # [num_example, 1, num_spec, num_output]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
    # [num_example, 1, num_spec]

    attack_margin = (C_mat * attack_output_repeat).sum(-1) - rhs_mat
    # [num_example, num_restarts, num_spec]

    if arguments.Config['general']['save_output']:
        arguments.Globals['out']['attack_margin'] = attack_margin.cpu()

    print("PGD attack margin (first 2 examples and 10 specs):\n", attack_margin[:2, :, :10])
    print("number of violation: ", (attack_margin < 0).sum().item())
    # print the first 10 specifications for the first 2 examples

    return attack_image, attack_output, attack_margin


def OSI_init_C(model, X, alpha, output_dim, iter_steps=50, lower_limit=0.0, upper_limit=1.0):
    # the general version of OSI initialization.
    input_shape = X.shape
    # [batch_size, num_restarts, num_or_spec, *X_shape[1:]]
    X_init = X.clone().detach()
    # [batch_size, num_restarts, num_or_spec, *X_shape[1:]]
    X_init = X_init.view(-1, *X_init.shape[3:])
    X = X.view(-1, *X.shape[3:])
    # [batch_size, * num_restarts * num_or_spec, *X_shape[1:]]

    w_d = (torch.rand([X.shape[0], output_dim], device=X.device) - 0.5) * 2

    for i in range(iter_steps):
        X_init = X_init.detach().requires_grad_()
        output = model(X_init)

        # test whether we need to early stop here.

        dot = torch.einsum('...,...->', w_d, output)
        # dot = (w_d * output).sum()
        dot.backward()

        with torch.no_grad():
            X_init = X_init + alpha * torch.sign(X_init.grad)
            X_init = X_init.view(input_shape)
            X_init = torch.max(torch.min(X_init, upper_limit), lower_limit)
            X_init = X_init.view(-1, *X_init.shape[3:])

    X_init = X_init.view(input_shape)
    X = X.view(input_shape)

    assert (X_init <= upper_limit).all()
    assert (X_init >= lower_limit).all()

    return X_init


def OSI_init(model, X, y, eps, alpha, num_classes, iter_steps=50, lower_limit=0.0, upper_limit=1.0, extra_dim=None):
    input_shape = X.size()
    if extra_dim is not None:
        X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
    expand_shape = X.size()

    X_init = X.clone().detach()

    upper_limit = upper_limit.unsqueeze(1).unsqueeze(2)
    lower_limit = lower_limit.unsqueeze(1).unsqueeze(2)
    delta = (torch.empty_like(X).uniform_(-1, 1) * (upper_limit - lower_limit) + lower_limit)
    X_init = X_init + delta

    X = X.reshape(-1, *input_shape[1:])
    X_init = X_init.reshape(-1, *input_shape[1:])
    # Random vector from [-1, 1].
    w_d = (torch.rand([X.shape[0], num_classes], device=X.device) - 0.5) * 2

    if eps != float('inf'):
        lower_limit = torch.clamp(X-eps, min=lower_limit)
        upper_limit = torch.clamp(X+eps, max=upper_limit)

    for i in range(iter_steps):
        X_init = X_init.detach().requires_grad_()
        output = model(X_init)

        if (output.argmax(-1) != y).any():
            # return if attack succeeds.
            return X_init.view(expand_shape)

        dot = (w_d * output).sum()
        grad = torch.autograd.grad(dot, X_init)[0]

        X_init = X_init + alpha * torch.sign(grad)

        X_init = X_init.view(expand_shape)
        X_init = torch.max(torch.min(X_init, upper_limit), lower_limit)
        X = X.view(expand_shape)
        X_init = torch.max(torch.min(X_init, X+eps), X-eps)
        X_init = X_init.reshape(-1, *input_shape[1:])
        X = X.reshape(-1, *input_shape[1:])

    X_init = X_init.view(expand_shape)
    X = X.view(expand_shape)

    assert (X_init <= upper_limit).all()
    assert (X_init >= lower_limit).all()

    if eps is not None:
        assert (X_init <= X+eps).all()
        assert (X_init >= X-eps).all()

    return X_init


def pgd_attack_with_general_specs(model, X, data_min, data_max, C_mat, rhs_mat,
                                  cond_mat, same_number_const, alpha,
                                  use_adam=True, normalize=lambda x: x,
                                  initialization='uniform', GAMA_loss=False,
                                  num_restarts=None, pgd_steps=None,
                                  only_replicate_restarts=False,
                                  return_early_stopped=False):

    r''' the functional function for pgd attack

    Args:
        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)

        C_mat (torch.tensor): [num_example, num_spec, num_output]

        rhs_mat (torch.tensor): [num_example, num_spec]

        cond_mat (list): [[] * num_example] mark the group of conditions

        same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.

        alpha (float): alpha for pgd attack
    '''
    device = X.device
    attack_iters = arguments.Config["attack"]["pgd_steps"] if pgd_steps is None else pgd_steps
    num_restarts = arguments.Config["attack"]["pgd_restarts"] if num_restarts is None else num_restarts

    lr_decay=arguments.Config["attack"]["pgd_lr_decay"]
    early_stop=arguments.Config["attack"]["pgd_early_stop"]
    restart_when_stuck = arguments.Config["attack"]["pgd_restart_when_stuck"]

    if restart_when_stuck:
        total_replaced_deltas = 0

    if only_replicate_restarts:
        input_shape = (X.shape[0], *X.shape[2:])
    else:
        input_shape = X.size()
    num_classes = C_mat.shape[-1]

    num_or_spec = len(cond_mat[0])

    extra_dim = (num_restarts, num_or_spec) if only_replicate_restarts == False else (num_restarts,)
    # shape of x: [num_example, *shape_of_x]

    best_loss = torch.empty(X.size(0), device=device).fill_(float("-inf"))
    best_delta = torch.zeros(input_shape, device=device)

    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)
    # [1, 1, num_spec, *input_shape]

    X_ndim = X.ndim

    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X

    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim - 1))
    extra_dim = (X.shape[1], X.shape[2])

    if initialization == 'osi':
        # X_init = OSI_init(model, X, y, epsilon, alpha, num_classes, iter_steps=attack_iters, extra_dim=extra_dim, upper_limit=upper_limit, lower_limit=lower_limit)
        osi_start_time = time.time()
        X_init = OSI_init_C(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
        osi_time = time.time() - osi_start_time
        print(f'diversed PGD initialization time: {osi_time:.4f}')
    if initialization == 'boundary':
        boundary_adv_examples = boundary_attack(model, X[:,0,...].view(-1, *input_shape[1:]), data_min.view(*input_shape), data_max.view(*input_shape))
        if boundary_adv_examples is not None:
            X_init = boundary_adv_examples.view(X.shape[0], -1, *X.shape[2:])
            X = X[:,:X_init.shape[1],...]
            extra_dim = (X.shape[1], X.shape[2])
        else:
            initialization = 'uniform'

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    if initialization == 'osi' or initialization == 'boundary':
        delta = (X_init - X).detach().requires_grad_()
    elif initialization == 'uniform':
        delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
    elif initialization == 'none':
        delta = torch.zeros_like(X).requires_grad_()
    else:
        raise ValueError(f"Unknown initialization method {initialization}")

    if use_adam:
        opt = AdamClipping(params=[delta], lr=alpha)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    early_stopped = False

    for iteration in range(attack_iters):
        inputs = normalize(X + delta)
        output = model(inputs.view(-1, *input_shape[1:])).view(
            input_shape[0], *extra_dim, num_classes)

        if GAMA_loss:
            # Output on original model is needed if gama loss is used.
            origin_out = torch.softmax(model(normalize(X.reshape(-1, *input_shape[1:]))), 1)
            origin_out = origin_out.view(output.shape)
        else:
            origin_out = None

        loss, loss_gama = eval(arguments.Config["attack"]["pgd_loss"])(
            origin_out, output, C_mat, rhs_mat,
            cond_mat, same_number_const,
            gama_lambda if GAMA_loss else 0.0,
            mode=arguments.Config['attack']['pgd_loss_mode'], model=model)
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        # shape of loss: [num_example, num_restarts, num_or_spec]
        # or float when gama_lambda > 0

        loss_gama.sum().backward()

        with torch.no_grad():
            # Save the best loss so far.
            if same_number_const:
                loss = loss.amin(-1)
                # loss has shape [num_example, num_restarts, num_or_spec].
                # margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
            else:
                group_C = torch.zeros(len(cond_mat[0]), C_mat.shape[1]).to(loss.device) # [num_or_spec, num_total_spec]
                x_index = []
                y_index = []
                index = 0
                for i, cond in enumerate(cond_mat[0]):
                    for _ in range(cond):
                        x_index.append(i)
                        y_index.append(index)
                        index += 1
                group_C[x_index, y_index] = 1.0

                # loss shape: [batch_size, num_restarts, num_total_spec]
                loss = group_C.matmul(loss.unsqueeze(-1)).squeeze(-1)
                # loss shape: [batch_size, num_restarts, num_or_spec]

            loss = loss.view(loss.shape[0], -1)
            # all_loss and indices have shape (batch, ),
            # and this is the best loss over all restarts and number of classes.
            all_loss, indices = loss.max(1)
            # delta has shape (batch, restarts, num_class-1, c, h, w).
            # For each batch element, we want to select from the best over
            # (restarts, num_classes-1) dimension.
            # delta_targeted has shape (batch, c, h, w).
            delta_targeted = delta.view(
                delta.size(0), -1, *input_shape[1:]
            ).gather(
                dim=1, index=indices.view(
                    -1,1,*(1,) * (len(input_shape) - 1)).expand(
                        -1,-1,*input_shape[1:])
            ).squeeze(1)

            best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
            best_loss = torch.max(best_loss, all_loss)

        if early_stop:
            if eval(arguments.Config["attack"]["early_stop_condition"])(inputs, output, C_mat, rhs_mat,
                    cond_mat, same_number_const, data_max, data_min, model, indices, num_or_spec).all():
                print("pgd early stop")
                early_stopped = True
                break

        if restart_when_stuck:
            old_delta = delta.clone().detach()
        if use_adam:
            opt.step(clipping=True, lower_limit=delta_lower_limit,
                     upper_limit=delta_upper_limit, sign=1)
            opt.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            d = delta + alpha * torch.sign(delta.grad)
            d = torch.max(torch.min(d, delta_upper_limit), delta_lower_limit)
            delta = d.detach().requires_grad_()

        if restart_when_stuck:
            unchanged = ((delta - old_delta).abs().sum(list(range(2, delta.ndim)), keepdim=True) == 0).to(delta.dtype)
            total_replaced_deltas += int(unchanged.sum().item())
            new_init = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit)
            delta.data.copy_(delta * (1 - unchanged) + new_init * unchanged)

    if restart_when_stuck:
        total_num_deltas = X.size(0) * delta.size(1) * (iteration + 1)
        replaced_percentage = total_replaced_deltas / total_num_deltas * 100
        print(f'Attack batch size: {X.size(0)}, restarts: {delta.size(1)}, iterations: {iteration + 1} '
              f'replaced deltas {total_replaced_deltas} ({replaced_percentage}%)')

    if not early_stopped and 'Customized' in arguments.Config["attack"]["early_stop_condition"]:
        test_input = X[:, 0, 0, :] + best_delta
        test_output = model(test_input)
        test_input = test_input.unsqueeze(0).unsqueeze(0)
        test_output = test_output.unsqueeze(0).unsqueeze(0)
        if not test_conditions(test_input, test_output, C_mat, rhs_mat, cond_mat,
                           same_number_const, data_max, data_min).all():
            best_loss = torch.full(size=(1,), fill_value=float('-inf'), device=best_loss.device)

    if return_early_stopped:
        return best_delta, delta, best_loss, early_stopped
    else:
        return best_delta, delta, best_loss


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, num_restarts,
        multi_targeted=True, num_classes=10, use_adam=True, lr_decay=0.98,
        lower_limit=0.0, upper_limit=1.0, normalize=lambda x: x, early_stop=True, target=None,
        initialization='uniform', GAMA_loss=False, nn4sys=False):
    if initialization == 'osi':
        if multi_targeted:
            extra_dim = (num_restarts, num_classes - 1,)
        else:
            extra_dim = (num_restarts)
        X_init = OSI_init(model, X, y, epsilon, alpha, num_classes,
                          iter_steps=attack_iters, extra_dim=extra_dim,
                          upper_limit=upper_limit, lower_limit=lower_limit)

    best_loss = torch.empty(X.size(0), device=X.device).fill_(float("-inf"))
    best_delta = torch.zeros_like(X, device=X.device)

    input_shape = X.size()
    if multi_targeted:
        assert target is None  # Multi-targeted attack is for non-targeted attack only.
        extra_dim = (num_restarts, num_classes - 1,)
        # Add two extra dimensions for targets. Shape is (batch, restarts, target, ...).
        X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
        # Generate target label list for each example.
        E = torch.eye(num_classes, dtype=X.dtype, device=X.device)
        c = E.unsqueeze(0) - E[y].unsqueeze(1)
        # remove specifications to self.
        I = ~(y.unsqueeze(1) == torch.arange(num_classes, device=y.device).unsqueeze(0))
        # c has shape (batch, num_classes - 1, num_classes).
        c = c[I].view(input_shape[0], num_classes - 1, num_classes)
        # c has shape (batch, restarts, num_classes - 1, num_classes).
        c = c.unsqueeze(1).expand(-1, num_restarts, -1, -1)
        target_y = y.view(-1,*(1,) * len(extra_dim),1).expand(-1, *extra_dim, 1)
        # Restart is processed in a batch and no need to do individual restarts.
        num_restarts = 1
        # If element-wise lower and upper limits are given, we should reshape them to the same as X.
        if lower_limit.ndim == len(input_shape):
            lower_limit = lower_limit.unsqueeze(1).unsqueeze(1)
        if upper_limit.ndim == len(input_shape):
            upper_limit = upper_limit.unsqueeze(1).unsqueeze(1)
    else:
        if target is not None:
            # An attack target for targeted attack, in dimension (batch, ).
            target = torch.tensor(target, device='cuda').view(-1,1)
            target_index = target.view(-1,1,1).expand(-1, num_restarts, 1)
            # Add an extra dimension for num_restarts. Shape is (batch, num_restarts, ...).
            X = X.unsqueeze(1).expand(-1, num_restarts, *(-1,) * (X.ndim - 1))
            # Only run 1 restart, since we run all restarts together.
            extra_dim = (num_restarts, )
            num_restarts = 1
            # If element-wise lower and upper limits are given, we should reshape them to the same as X.
            if lower_limit.ndim == len(input_shape):
                lower_limit = lower_limit.unsqueeze(1)
            if upper_limit.ndim == len(input_shape):
                upper_limit = upper_limit.unsqueeze(1)
        elif nn4sys: # nn4sys
            # Add an extra dimension for num_restarts. Shape is (batch, num_restarts, ...).
            X = X.unsqueeze(1).expand(-1, num_restarts, *(-1,) * (X.ndim - 1))
            extra_dim = (num_restarts, )
            num_restarts = 1
            # If element-wise lower and upper limits are given, we should reshape them to the same as X.
            if lower_limit.ndim == len(input_shape):
                lower_limit = lower_limit.unsqueeze(1)
            if upper_limit.ndim == len(input_shape):
                upper_limit = upper_limit.unsqueeze(1)

    # This is the maximal/minimal delta values for each sample, each element.
    sample_lower_limit = torch.clamp(lower_limit - X, min=-epsilon)
    sample_upper_limit = torch.clamp(upper_limit - X, max=epsilon)

    success = False

    for _ in range(num_restarts):
        # one_label_loss = True if n % 2 == 0 else False  # random select target loss or marginal loss
        one_label_loss = False  # Temporarily disabled. Will do more tests.
        gama_lambda = arguments.Config["attack"]["gama_lambda"]
        if early_stop and success:
            break

        if initialization == 'osi':
            delta = (X_init - X).detach().requires_grad_()
        elif initialization == 'uniform':
            delta = (torch.empty_like(X).uniform_() * (sample_upper_limit - sample_lower_limit) + sample_lower_limit).requires_grad_()
        elif initialization == 'none':
            delta = torch.zeros_like(X).requires_grad_()
        else:
            raise ValueError(f"Unknown initialization method {initialization}")

        if use_adam:
            opt = AdamClipping(params=[delta], lr=alpha)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        for _ in range(attack_iters):
            inputs = normalize(X + delta)
            if multi_targeted or target is not None:
                output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)
            else:
                output = model(inputs)

            if not multi_targeted:
                # Not using the multi-targeted loss. Can be target or non-target attack.
                if nn4sys:
                    # [batch, num_restarts, num_class] -> [batch, num_restarts]
                    loss = (-y*output).sum(-1)
                elif target is not None:
                    # Targeted attack. In this case we have an extra (num_starts, ) dimension.
                    runnerup = output.scatter(dim=2, index=target_index, value=-float("inf")).max(2).values
                    # t = output.gather(dim=2, index=target_index).squeeze(-1)
                    t = output[:, :, target].squeeze(-1).squeeze(-1)
                    loss = (t - runnerup)
                else:
                    # Non-targeted attack.
                    if one_label_loss:
                        # Loss 1: simply reduce the loss groundtruth target.
                        loss = -output.gather(dim=1, index=y.view(-1,1)).squeeze(1)  # -groundtruth
                    else:
                        # Loss 2: reduce the margin between groundtruth and runner up label.
                        runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
                        groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
                        # Use the margin as the loss function.
                        loss = (runnerup - groundtruth)
                loss.sum().backward()
            else:
                if GAMA_loss:
                    origin_out = torch.softmax(model(normalize(X.reshape(-1, *input_shape[1:]))), 1)
                    origin_out = origin_out.view(output.shape)

                    # [batch, restarts, label - 1, logits]
                    out = torch.softmax(output, -1)

                    # runnerup = out.scatter(dim=3, index=target_y, value=-float("inf")).max(3).values
                    # groundtruth = out.gather(dim=3, index=target_y).squeeze(3)
                    # Use the margin as the loss function.
                    loss = torch.einsum('ijkl,ijkl->', c, output)
                    loss = loss.sum() + (gama_lambda * (out - origin_out)**2).sum(dim=3).sum()
                    # and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size / len(constraints)))
                    # and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size - len(and_idx))], axis=0)).to(device)
                    # out = torch.softmax(out,1)
                    # loss = (constraint_loss(out, constraints, and_idx=and_idx) + (gama_lambda * (out_X-out)**2).sum(dim=1)).sum()
                    gama_lambda *= arguments.Config["attack"]["gama_decay"]
                else:
                    # Non-targeted attack, using margins between groundtruth class and all target classes together.
                    # loss = torch.einsum('ijkl,ijkl->ijk', c, output)
                    loss = torch.einsum('ijkl,ijkl->', c, output)
                loss.backward()

            # print(loss.sum().item(), output.detach().cpu().numpy())
            # print(loss[:, :, 5])

            with torch.no_grad():
                # Save the best loss so far.
                if not multi_targeted:
                    # Not using multi-targeted loss.
                    if nn4sys:
                        all_loss, indices = loss.max(1)
                        delta_best = delta.gather(dim=1, index=indices.view(-1,1,*[1]*len(input_shape[1:])).expand(-1,-1,*input_shape[1:])).squeeze(1)
                        best_delta[all_loss >= best_loss] = delta_best[all_loss >= best_loss]
                        best_loss = torch.max(best_loss, all_loss)
                    elif target is not None:
                        # Targeted attack, need to check if the top-1 label is target label.
                        # Since we merged the random restart dimension, we need to find the best one among all random restarts.
                        all_loss, indices = loss.max(1)
                        # Gather the delta for the best loss in all random restarts.
                        delta_best = delta.gather(dim=1, index=indices.view(-1,1,1,1,1).expand(-1,-1,*input_shape[1:])).squeeze(1)
                        best_delta[all_loss >= best_loss] = delta_best[all_loss >= best_loss]
                        best_loss = torch.max(best_loss, all_loss)
                    else:
                        # Non-targeted attack. Success when the groundtruth is not top-1.
                        if one_label_loss:
                            runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
                            groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
                            # Use the margin as the loss function.
                            criterion = (runnerup - groundtruth)  # larger is better.
                        else:
                            criterion = loss
                        # Larger is better.
                        best_delta[criterion >= best_loss] = delta[criterion >= best_loss]
                        best_loss = torch.max(best_loss, criterion)
                else:
                    # if GAMA_loss:
                    #     # out = torch.softmax(output, -1).view(-1, num_classes)
                    #     out = torch.softmax(output, -1)
                    #     # runnerup = out.scatter(dim=3, index=target_y, value=-float("inf")).max(3).values
                    #     # groundtruth = out.gather(dim=3, index=target_y).squeeze(3)

                    #     # Use the margin as the loss function.
                    #     loss = torch.einsum('ijkl,ijkl->', c, output) + (gama_lambda/0.9 * (out - origin_out.view(output.shape))**2).sum(dim=3)
                    #     loss = loss.view(out.size(0), -1)
                    #     all_loss, indices = loss.max(1)
                    #     # and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size / len(constraints)))
                    #     # and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size - len(and_idx))], axis=0)).to(device)
                    #     # out = torch.softmax(out,1)
                    #     # loss = (constraint_loss(out, constraints, and_idx=and_idx) + (gama_lambda * (out_X-out)**2).sum(dim=1)).sum()
                    #     delta_targeted = delta.view(delta.size(0), -1, *input_shape[1:]).gather(dim=1, index=indices.view(-1,1,*(1,) * (len(input_shape) - 1)).expand(-1,-1,*input_shape[1:])).squeeze(1)
                    #     best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
                    #     best_loss = torch.max(best_loss, all_loss)
                    # else:
                    # Using multi-targeted loss. Need to find which label causes the worst case margin.
                    # Keep the one with largest margin.
                    # Note that we recompute the runnerup label here - the runnerup label might not be the target label.
                    # output has shape (batch, restarts, num_classes-1, num_classes).
                    # runnerup has shape (batch, restarts, num_classes-1).
                    runnerup = output.scatter(dim=3, index=target_y, value=-float("inf")).max(3).values
                    # groundtruth has shape (batch, restarts, num_classes-1).
                    groundtruth = output.gather(dim=3, index=target_y).squeeze(-1)
                    # margins has shape (batch, restarts * num_classes), ).
                    margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
                    # all_loss and indices have shape (batch, ), and this is the best loss over all restarts and number of classes.
                    all_loss, indices = margins.max(1)
                    # delta has shape (batch, restarts, num_class-1, c, h, w). For each batch element, we want to select from the best over (restarts, num_classes-1) dimension.
                    # delta_targeted has shape (batch, c, h, w).
                    delta_targeted = delta.view(delta.size(0), -1, *input_shape[1:]).gather(dim=1, index=indices.view(-1,1,*(1,) * (len(input_shape) - 1)).expand(-1,-1,*input_shape[1:])).squeeze(1)
                    best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
                    best_loss = torch.max(best_loss, all_loss)

                if early_stop:
                    if multi_targeted:
                        # Must be a untargeted attack. If any of the target succeed, that element in batch is successfully attacked.
                        # output has shape (batch, num_restarts, num_classes-1, num_classes,).
                        if (output.view(output.size(0), -1, num_classes).max(2).indices != y.unsqueeze(1)).any(1).all():
                            print('pgd early stop.')
                            success = True
                            break
                    elif target is not None:
                        # Targeted attack, the top-1 label of every element in batch must match the target.
                        # If any attack in some random restarts succeeds, the attack is successful.
                        if (output.max(2).indices == target).any(1).all():
                            print('pgd early stop.')
                            success = True
                            break
                    elif nn4sys:
                        if (output.view(output.size(0), -1, num_classes) * y <= target).any():
                            print('pgd early stop.')
                            success = True
                            break
                    else:
                        # Non-targeted attack, the top-1 label of every element in batch must not be the groundtruth label.
                        if (output.max(1).indices != y).all():
                            print('pgd early stop.')
                            success = True
                            break

                # Optimizer step.
                if use_adam:
                    opt.step(clipping=True, lower_limit=sample_lower_limit, upper_limit=sample_upper_limit, sign=1)
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                else:
                    d = delta + alpha * torch.sign(delta.grad)
                    d = torch.max(torch.min(d, sample_upper_limit), sample_lower_limit)
                    delta.copy_(d)
                    delta.grad = None

    return best_delta, delta


class AdamClipping(Optimizer):
    r"""Implements Adam algorithm, with per-parameter gradient clipping.
    The function is from PyTorch source code.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _clip_update(exp_avg : torch.Tensor, denom : torch.Tensor, step_size : float, clipping_step_eps : float, lower_limit : torch.Tensor, upper_limit : torch.Tensor, p : torch.Tensor):
        # Compute the Adam update.
        update = exp_avg / denom * step_size
        # update = p.grad
        # Linf norm, scale according to sign.
        scaled_update = torch.sign(update) * clipping_step_eps
        # Apply the update.
        d = p.data + scaled_update
        # Avoid out-of-boundary updates.
        d = torch.max(torch.min(d, upper_limit), lower_limit)
        p.copy_(d)

    @torch.no_grad()
    def step(self, clipping=None, lower_limit=None, upper_limit=None, sign=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Currently we only deal with 1 parameter group.
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if clipping:
                    assert sign == 1  # gradient ascent for adversarial attacks.
                    self._clip_update(exp_avg, denom, step_size, step_size, lower_limit, upper_limit, p)
                else:
                    # No clipping. Original Adam update.
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def boundary_attack(model, x, data_min, data_max):
    perturbation_index = ((data_max - data_min) != 0).view(data_max.shape[0], -1).nonzero()
    # index of the pixels perturbed
    if len(perturbation_index) > 5:
        print("Error: number of perturbed pixels is larger than 5, boundary attack is disabled.")
        return None

    data_max_flatten = data_max.view(data_max.shape[0], -1)
    data_min_flatten = data_min.view(data_min.shape[0], -1)

    adv_example = data_max_flatten
    for idx in perturbation_index:
        adv_example_neg = adv_example.clone()
        adv_example[:, idx] = data_max_flatten[0, idx]
        adv_example_neg[:, idx] = data_min_flatten[0, idx]
        adv_example = torch.cat([adv_example, adv_example_neg], dim=0)

    return adv_example.view(-1, *data_max.shape[1:])




def attack_with_general_specs(model, x, data_min, data_max,
                              list_target_label_arrays,
                              initialization="uniform", GAMA_loss=False):
    r""" Interface to PGD attack.

    Args:
        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).
        [batch_size, *x_shape]

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)
        shape: [batch_size, spec_num, *input_shape]

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)
        shape: [batch_size, spec_num, *input_shape]

        list_target_label_arrays: a list of list of tuples:
                We have N examples, and list_target_label_arrays is a list containing N lists.
                Each inner list contains the target_label_array for an example:
                    [(prop_mat_1, prop_rhs_1), (prop_mat_2, prop_rhs_2), ..., (prop_mat_n, prop_rhs_n)]
                    prop_mat is a numpy array with shape [num_and, num_output], prop_rhs is a numpy array with shape [num_and]

        initialization (string): initialization of PGD attack, chosen from 'uniform' and 'osi'

        GAMA_loss (boolean): whether to use GAMA (Guided adversarial attack) loss in PGD attack
    """
    attack_start_time = time.time()
    assert arguments.Config["specification"]["norm"] == np.inf, print('We only support Linf-norm attack.')
    use_adam = True

    device = x.device

    alpha = arguments.Config["attack"]["pgd_alpha"]
    alpha_scale = arguments.Config["attack"]["pgd_alpha_scale"]
    if alpha_scale:
        alpha = (data_max - data_min) * float(alpha)
        use_adam = False
    else:
        if alpha == 'auto':
            max_eps = torch.max(data_max - data_min).item()/2
            alpha = max_eps / 4
        else:
            alpha = float(alpha)

    print(f'Attack parameters: initialization={initialization}, steps={arguments.Config["attack"]["pgd_steps"]}, restarts={arguments.Config["attack"]["pgd_restarts"]}, alpha={alpha}, initialization={initialization}, GAMA={GAMA_loss}')

    # Set all parameters without gradient, this can speedup things significantly.
    grad_status = {}
    for p in model.parameters():
        grad_status[p] = p.requires_grad
        p.requires_grad_(False)

    output = model(x).detach()

    # FIXME conflict with clean prediction
    # if arguments.Config['general']['save_output']:
    #     arguments.Globals['out']['pred'] = output.cpu()

    print('Model output of first 5 examples:\n', output[:5])

    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)

    output = output.unsqueeze(1).unsqueeze(1).repeat(1, 1, len(cond_mat[0]), 1)

    if test_conditions(x, output, C_mat, rhs_mat, cond_mat, same_number_const,
                       data_max.unsqueeze(1), data_min.unsqueeze(1)).all():
        print("Clean prediction incorrect, attack skipped.")
        # Obtain attack margin.
        attack_image, _, attack_margin = eval(arguments.Config["attack"]["adv_example_finalizer"])(model, x, torch.zeros_like(x), data_max, data_min, C_mat, rhs_mat, cond_mat)
        print("clean attack image", attack_image, attack_image.shape)
        return True, attack_image.detach(), attack_margin.detach(), None

    data_min = data_min.to(device)
    data_max = data_max.to(device)
    rhs_mat = rhs_mat.to(device)
    C_mat = C_mat.to(device)
    num_restarts = arguments.Config["attack"]["pgd_restarts"]
    batch_size = arguments.Config["attack"]["pgd_batch_size"]
    best_deltas = None
    best_loss = None
    for _ in tqdm(range((num_restarts + batch_size - 1) // batch_size)):
        best_deltas_, last_deltas, best_loss_, early_stopped = pgd_attack_with_general_specs(
            model, x, data_min, data_max, C_mat, rhs_mat, cond_mat, same_number_const, alpha,
            initialization=initialization, GAMA_loss=GAMA_loss,
            use_adam=use_adam, num_restarts=min(batch_size, num_restarts), return_early_stopped=True)
        num_restarts -= batch_size
        if best_deltas is None:
            best_deltas = best_deltas_
            best_loss = best_loss_
        else:
            best_deltas[best_loss_ >= best_loss] = best_deltas_[
                best_loss_ >= best_loss]
            best_loss = torch.max(best_loss, best_loss_)
        if early_stopped:
            break

    attack_image, attack_output, attack_margin = eval(arguments.Config["attack"]["adv_example_finalizer"])(
        model, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat)

    # Adversarial images/candidates in all restarts and targets. Useful for BaB-attack.
    # last_deltas has shape [batch, num_restarts, specs, c, h, w]. Need the extra num_restarts and specs dim.
    # x has shape [batch, c, h, w] and data_min/data_max has shape [batch, num_specs, c, h, w].
    all_adv_candidates = torch.max(
            torch.min(x.unsqueeze(1).unsqueeze(1) + last_deltas,
                data_max.unsqueeze(1)), data_min.unsqueeze(1))

    # Go back to original requires_grad status.
    for p in model.parameters():
        p.requires_grad_(grad_status[p])

    attack_time = time.time() - attack_start_time
    print(f'Attack finished in {attack_time:.4f} seconds.')
    if test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1),
                       C_mat, rhs_mat, cond_mat, same_number_const,
                       data_max, data_min).all():
        print("PGD attack succeeded!")
        return True, attack_image.detach(), attack_margin.detach(), all_adv_candidates
    else:
        print("PGD attack failed")
        return False, attack_image.detach(), attack_margin.detach(), all_adv_candidates


def pgd_attack(dataset, model, x, max_eps, data_min, data_max, vnnlib=None, y=None,
               target=None, only_target_attack=False, initialization="uniform", GAMA_loss=False):
    # FIXME (01/11/2022): any parameter that can be read from config should not be passed in.
    r"""Interface to PGD attack.

    Args:
        dataset (str): The name of dataset. Each dataset might have different attack configurations.

        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).

        max_eps (float): Perturbation Epsilon. Assuming Linf perturbation for now. (e.g., 0.3 for MNIST)

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)

        vnnlib (list, optional): VNNLIB specifications. It will be used to extract attack target.

        y (int, optional): Groundtruth label. If specified, vnnlib will be ignored.

    Returns:
        success (bool): True if attack is successful. Otherwise False.
        attack_images (torch.tensor): last adversarial examples so far, may not be a real adversarial example if attack failed
    """
    assert arguments.Config["specification"]["norm"] == np.inf, print('We only support Linf-norm attack.')
    if dataset in ["MNIST", "CIFAR", "UNKNOWN"]:  # FIXME (01/11/2022): Make the attack function generic, not for the two datasets only!
        # FIXME (01/11/2022): Generic specification PGD.
        if y is not None and target is None:
            # Use y as the groundtruth label.
            pidx_list = ["all"]
            if arguments.Config["specification"]["type"] == "lp":
                if data_max is None:
                    data_max = x + max_eps
                    data_min = x - max_eps
                else:
                    data_max = torch.min(x + max_eps, data_max)
                    data_min = torch.max(x - max_eps, data_min)
            # If arguments.Config["specification"]["type"] == "bound", then we keep data_min and data_max.
        else:
            pidx_list = []
            if vnnlib is not None:
                # Extract attack target from vnnlib.
                for prop_mat, prop_rhs in vnnlib[0][1]:
                    if len(prop_rhs) > 1:
                        output = model(x).detach().cpu().numpy().flatten()
                        print('model output:', output)
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
            elif target is not None:
                pidx_list.append(target)
            else:
                raise NotImplementedError

        print('##### PGD attack: True label: {}, Tested against: {} ######'.format(y, pidx_list))
        if not isinstance(max_eps, float):
            max_eps = torch.max(max_eps).item()

        if only_target_attack:
            # Targeted attack PGD.
            if arguments.Config["attack"]["pgd_alpha"] == 'auto':
                alpha = max_eps
            else:
                alpha = float(arguments.Config["attack"]["pgd_alpha"])
            best_deltas, last_deltas = attack_pgd(model, X=x, y=None, epsilon=float("inf"), alpha=alpha, num_classes=arguments.Config["data"]["num_outputs"],
                    attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
                    multi_targeted=False, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=pidx_list[0], initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"], GAMA_loss=GAMA_loss)
        else:
            # Untargeted attack PGD.
            if arguments.Config["attack"]["pgd_alpha"] == 'auto':
                alpha = max_eps/4.0
            else:
                alpha = float(arguments.Config["attack"]["pgd_alpha"])
            best_deltas, last_deltas = attack_pgd(model, X=x, y=torch.tensor([y], device=x.device), epsilon=float("inf"), alpha=alpha, num_classes=arguments.Config["data"]["num_outputs"],
                    attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
                    multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=None, initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"], GAMA_loss=GAMA_loss)

        if x.shape[0] == 1:
            attack_image = torch.max(torch.min(x + best_deltas, data_max), data_min)
            assert (attack_image >= data_min).all()
            assert (attack_image <= data_max).all()
            # assert (attack_image-x).abs().max() <= eps_temp.max(), f"{(attack_image-x).abs().max()} <= {eps_temp.max()}"
            attack_output = model(attack_image).squeeze(0)

            # FIXME (10/02): This is not the best image for each attack target. We should save best attack delta for each target.
            all_targets_attack_image = torch.max(torch.min(x + last_deltas, data_max), data_min)

            attack_label = attack_output.argmax()
            print("pgd prediction:", attack_output)

            # FIXME (10/05): Cleanup.
            if only_target_attack:
                # Targeted attack, must have one label.
                attack_logit = attack_output.data[pidx_list[0]].item()
                attack_output.data[pidx_list[0]] = -float("inf")
                attack_margin = attack_output.max().item() - attack_logit
                print("attack margin", attack_margin)
                if attack_label == pidx_list[0]:
                    assert len(pidx_list) == 1
                    print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    # FIXME (10/05): Please check! attack_image is for one target only.
                    return True, attack_image.detach(), [attack_margin]
                else:
                    print(f"targeted pgd failed, margin {attack_margin}")
                    return False, attack_image.detach(), [attack_margin]
            else:
                # Untargeted attack, any non-groundtruth label is ok.
                groundtruth_logit = attack_output.data[y].item()
                attack_output.data[y] = -float("inf")
                attack_margin = groundtruth_logit - attack_output
                print("attack margin", attack_margin)
                # Untargeted attack, any non-groundtruth label is ok.
                if attack_label != y:
                    print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    return True, attack_image.detach(), attack_margin.detach().cpu().numpy()
                else:
                    print("untargeted pgd failed")
                    return False, attack_image.detach(), attack_margin.detach().cpu().numpy()
        else:
            # FIXME (10/02): please remove duplicated code!
            attack_images = torch.max(torch.min(x + best_deltas, data_max), data_min)
            attack_output = model(attack_images).squeeze(0)
            # do in batch attack
            attack_label = attack_output.argmax(1)

            if only_target_attack:
                # Targeted attack, must have one label.
                if (attack_label == pidx_list[0]).any():
                    # FIXME (10/02): remove duplicated code.
                    assert len(pidx_list) == 1
                    # print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    attack_logit = attack_output.data[:, pidx_list[0]].clone()
                    attack_output.data[:, pidx_list[0]] = -float("inf")
                    attack_margin = attack_output.max(1).values - attack_logit
                    return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
                else:
                    attack_logit = attack_output.data[:, pidx_list[0]].clone()
                    attack_output.data[:, pidx_list[0]] = -float("inf")
                    attack_margin = attack_output.max(1).values - attack_logit
                    # print(f"targeted pgd failed, margin {attack_margin}")
                    return False, attack_images.detach(), attack_margin.detach().cpu().numpy()
            else:
                raise NotImplementedError
                # TODO support batch
                # Untargeted attack, any non-groundtruth label is ok.
                groundtruth_logit = attack_output.data[y].item()
                attack_output.data[y] = -float("inf")
                attack_margin = groundtruth_logit - attack_output
                # Untargeted attack, any non-groundtruth label is ok.
                if attack_label != y:
                    print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
                else:
                    print("untargeted pgd failed")
                    return False, attack_images.detach(), attack_margin.detach().cpu().numpy()
    elif "NN4SYS" in dataset:
        # attack for nn4sys, attack the specs line by line
        # vnnlib:
        # [num_spec, num_X, 2(lower, upper)], a y <= b, num_spec
        specs = [vnnlib[i][0] for i in range(len(vnnlib))]
        y_sign = [vnnlib[i][1][0][0] for i in range(len(vnnlib))]
        y_upper = [vnnlib[i][1][0][1] for i in range(len(vnnlib))]

        # attack the specs in batch
        for i, spec in enumerate(specs):
            # print the top 10 specs for debugging.
            print('##### PGD attack: Batch {}, Threshold: {} ######'.format(i, y_upper[i][:10].squeeze() * y_sign[i][:10].squeeze()), y_sign[i][:10].squeeze())
            data_min = torch.Tensor(specs[i][:,:,0]).view(-1, *x.shape[1:]).to(x.device)
            data_max = torch.Tensor(specs[i][:,:,1]).view(-1, *x.shape[1:]).to(x.device)

            x = (data_min + data_max)/2

            eps = ((data_max - data_min)[(data_max - data_min)!=0]).mean().item()
            y = torch.Tensor(y_sign[i]).to(x.device)
            target = torch.Tensor(y_upper[i]).to(x.device)

            if arguments.Config["attack"]["pgd_alpha"] == 'auto':
                alpha = eps/4.0
            else:
                alpha = float(arguments.Config["attack"]["pgd_alpha"])

            best_deltas, last_deltas = attack_pgd(model, X=x, y=y, target=target, epsilon=float("inf"), alpha=alpha, num_classes=arguments.Config["data"]["num_outputs"],
                attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
                multi_targeted=False, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"], GAMA_loss=GAMA_loss, nn4sys=True)


            attack_images = torch.max(torch.min(x + best_deltas, data_max), data_min)
            attack_output = model(attack_images).squeeze(0)
            print("model output: ", model(x).squeeze(0)[:10].squeeze())
            print("attack output: ", attack_output[:10].squeeze())

            if (attack_output.detach().cpu().numpy().squeeze() * y_sign[i].squeeze() <= y_upper[i]).any():
                index = (attack_output.detach().cpu().numpy().squeeze() * y_sign[i].squeeze() <= y_upper[i]).nonzero()
                print("pgd succeed, against upper bound {} with ".format(y_upper[index[0]], attack_output[index[0]]))
                return True, attack_images[index[0]].detach(), attack_output[index[0]].detach().cpu().numpy()

        print("pgd failed")
        return False, attack_images.detach(), attack_output.detach().cpu().numpy()
    else:
        print("pgd attack not supported for dataset", dataset)
        raise NotImplementedError


def auto_attack(model_ori, x, max_eps=None, data_max=None, data_min=None, y=None, vnnlib=None):
    if max_eps is None or type(max_eps) != float:
        eps = (data_max - data_min)/2 if max_eps is None else max_eps
        eps = eps[:,:,0,0]

        standard_eps = eps.mean()
        factor = standard_eps/eps
        std = factor
        mean = torch.zeros(std.shape).view([1, -1, 1, 1]).to(x.device)
        std = (std).view([1, -1, 1, 1]).to(x.device)
        unormalized_x = x * std + mean
    else:
        standard_eps = max_eps
        mean = torch.zeros(std.shape).view([1, -1, 1, 1]).to(x.device)
        std = torch.ones(std.shape).view([1, -1, 1, 1]).to(x.device)
        unormalized_x = x * std + mean

    from autoattack import AutoAttack

    normalized = Normalization(mean, std, model_ori)
    normalized = normalized.to(x.device)
    adversary = AutoAttack(normalized, norm='Linf', eps=standard_eps.item(), version='standard')
    # adversary.attacks_to_run = ['apgd-ce']

    only_target_attack = False # untargeted attack by default

    if vnnlib is not None:
        # Extract attack target from vnnlib.
        for prop_mat, prop_rhs in vnnlib[0][1]:
            if len(prop_rhs) > 1:
                output = model_ori(x).detach().cpu().numpy().flatten()
                print('model output:', output)
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
    assert not only_target_attack # untargeted attack by default

    attack_images = adversary.run_standard_evaluation(unormalized_x, torch.Tensor([y]).long().to(x.device), bs=1)
    attack_images = torch.max(torch.min((attack_images - mean)/std, data_max), data_min)
    if max_eps is not None:
        attack_images = torch.max(torch.min(attack_images, x + max_eps), x - max_eps)
    attack_ret = model_ori(attack_images)
    attack_margin = attack_ret[:,y] - attack_ret[0]
    attack_margin[y] = float("inf")
    attack_label = attack_ret.argmax().item()
    attack_ret = not attack_ret.argmax().item() ==y

    attack_images = attack_images.detach()
    attack_margin = attack_margin.detach().cpu().numpy()

    if attack_ret:
        print("untargeted auto attack succeed, label {}, against label {}".format(y, attack_label))
    else:
        print("untargeted auto attack failed")

    return attack_ret, attack_images, attack_margin


def attack_after_crown(lb, vnnlib, model_ori, x, decision_thresh):
    # Run adversarial attack on those specs that cannot be verified by CROWN.
    crown_filtered_constraints = np.zeros(len(lb[-1]))
    for i in range(len(lb[-1])):
        if isinstance(decision_thresh, torch.Tensor):
            assert decision_thresh.ndim == 2
            assert decision_thresh.shape[0] == 1, "Batch size should be 1 for attack after CROWN."
            if lb[-1][i].item() > decision_thresh[0][i].item():
                crown_filtered_constraints[i] = True
        else:
            # Threshold is a float.
            if lb[-1][i].item() > decision_thresh:
                crown_filtered_constraints[i] = True

    _, verified_success, attack_images, _, _ = attack(
        model_ori, x, [vnnlib],
        verified_status="unknown", verified_success=False,
        crown_filtered_constraints=crown_filtered_constraints)

    return verified_success, attack_images

def default_adv_verifier(attack_image, attack_output, vnnlib=None, check_output=False):
    """ 
    Do two kinds of check on counterexample:
    
    1.check if the inference outptus are the same on both PyTorch and ONNXRuntime 
      (enabled when check_output is True)
    2.check if the output satisfied spec conditions (enabled when inputting vnnlib)
    
    Args:
        attack_output: [1, *intput_shape]
        attack_image: [1, *input_shape]
    """
    flatten_output = attack_output.view(-1)
    onnx_attack_image = attack_image.cpu().numpy()

    rel_tol = 1e-3
    abs_tol = 1e-4
    if check_output and not is_onnx_equal_to_pytorch_output(arguments.Config['model']['onnx_path'],
                                                            onnx_attack_image, 
                                                            flatten_output.detach().cpu().numpy(), rel_tol): 
        return False

    if vnnlib is not None and not is_specification_vio(vnnlib, attack_image.view(-1), flatten_output, abs_tol):
        return False

    return True


def is_onnx_equal_to_pytorch_output(onnx_path, onnx_attack_image, pytorch_y, rel_tol):
    """
    Compare the output of ONNX Runtime and PyTorch for given inputs.
    
    Args:
        onnx_path (str): Path to the ONNX model.
        onnx_attack_image (numpy.ndarray): Input image for ONNX model.
        pytorch_y (numpy.ndarray): Expected output from PyTorch model.
        rel_tol (float): Relative tolerance for comparison.
    
    Returns:
        bool: True if the outputs are similar within the given tolerance, False otherwise.
    """
    print('Checking if onnxruntime output is equal to pytorch')
    onnx_output = inference_onnx(onnx_path, onnx_attack_image)
    onnxruntime_y = onnx_output.flatten("C")
    
    try:
        diff = np.linalg.norm(onnxruntime_y - pytorch_y, ord=np.inf)
        norm = np.linalg.norm(pytorch_y, ord=np.inf)
        if norm < 1e-6:  # don't divide by zero
            rel_error = 0
        else:
            rel_error = diff / norm
    except ValueError as e:
        diff = 9999
        rel_error = 9999

    print(f'L-inf norm difference between onnx execution and pytorch output: {diff} (rel error: {rel_error});' f'(rel_limit: {rel_tol})')
    if rel_error > rel_tol:
        print('WARNING: Failed in the comparison between ONNX Runtime and PyTorch results.')
        return False
    print('Succeed in ONNX Runtime comparison check.')
    return True


def is_specification_vio(box_spec_list, x_list, expected_y, tol):
    """Check that the spec file was obeyed"""
    rv = False
    
    for i, box_spec in enumerate(box_spec_list):
        input_box, spec_list = box_spec
        assert len(input_box) == len(x_list), f"input box len: {len(input_box)}, x_in len: {len(x_list)}"

        input_box_tensor = torch.tensor(input_box, dtype=x_list.dtype, device=x_list.device)
        lb_tensor, ub_tensor = input_box_tensor[:, 0], input_box_tensor[:, 1]
        
        # Check if x_list is inside the input box using tensor operations
        inside_input_box = torch.all((x_list >= lb_tensor - tol) & (x_list <= ub_tensor + tol))

        if inside_input_box:
            # Check spec
            violated = False
                
            for j, (prop_mat, prop_rhs) in enumerate(spec_list):
                prop_mat_tensor = torch.tensor(prop_mat, dtype=expected_y.dtype, device=expected_y.device)
                prop_rhs_tensor = torch.tensor(prop_rhs, dtype=expected_y.dtype, device=expected_y.device)
                
                vec = torch.matmul(prop_mat_tensor, expected_y)
                sat = torch.all(vec <= prop_rhs_tensor + tol)

                if sat:
                    violated = True
                    break

            if violated:
                rv = True
                break
    
    if rv:
        print('Succeed in specification conditions check.')
    else:
        print('WARNING: Failed in specs conditions check.')
    return rv
