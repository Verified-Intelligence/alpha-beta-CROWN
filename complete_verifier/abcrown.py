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
"""α,β-CROWN (alpha-beta-CROWN) verifier main interface."""

import copy
import socket
import random
import pickle
import os
import time
import gc
import torch
import numpy as np
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_min
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPAConvNet
from lp_mip_solver import FSB_score
from attack_pgd import attack
from utils import Customized, default_onnx_and_vnnlib_loader, parse_run_mode
from nn4sys_verification import nn4sys_verification
from batch_branch_and_bound import relu_bab_parallel
from batch_branch_and_bound_input_split import input_bab_parallel

from read_vnnlib import batch_vnnlib, read_vnnlib
from cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching


def incomplete_verifier(model_ori, data, data_ub=None, data_lb=None, vnnlib=None):
    norm = arguments.Config["specification"]["norm"]
    # Generally, c should be constructed from vnnlib
    assert len(vnnlib) == 1
    vnnlib = vnnlib[0]
    c = torch.tensor(np.array([item[0] for item in vnnlib[1]])).to(data)
    c_transposed = False
    if c.shape[0] != 1 and data.shape[0] == 1:
        # TODO need a more general solution.
        # transpose c to share intermediate bounds
        c = c.transpose(0, 1)
        c_transposed = True
    arguments.Config["bab"]["decision_thresh"] = torch.tensor(np.array([item[1] for item in vnnlib[1]])).to(data)

    model = LiRPAConvNet(model_ori, in_size=data.shape, c=c)
    print('Model prediction is:', model.net(data))
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    eps = arguments.Globals["lp_perturbation_eps"]  # Perturbation value for non-Linf perturbations, None for all other cases.
    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    bound_prop_method = arguments.Config["solver"]["bound_prop_method"]

    _, global_lb, _, _, _, mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_images = model.build_the_model(
            domain, x, data_lb, data_ub, vnnlib, stop_criterion_func=stop_criterion_min(arguments.Config["bab"]["decision_thresh"]))

    if (global_lb > arguments.Config["bab"]["decision_thresh"]).all():
        print("verified with init bound!")
        return "safe-incomplete", None, None, None, None

    if arguments.Config["attack"]["pgd_order"] == "middle":
        if attack_images is not None:
            return "unsafe-pgd", None, None, None, None

    # Save the alpha variables during optimization. Here the batch size is 1.
    saved_slopes = defaultdict(dict)
    for m in model.net.relus:
        for spec_name, alpha in m.alpha.items():
            # each slope size is (2, spec, 1, *shape); batch size is 1.
            saved_slopes[m.name][spec_name] = alpha.detach().clone()

    if bound_prop_method == 'alpha-crown':
        # obtain and save relu alphas
        activation_opt_params = dict([(relu.name, relu.dump_optimized_params()) for relu in model.net.relus])
    else:
        activation_opt_params = None

    if c_transposed:
        lower_bounds[-1] = lower_bounds[-1].t()
        upper_bounds[-1] = upper_bounds[-1].t()
        global_lb = global_lb.t()

    saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

    return "unknown", global_lb, saved_bounds, saved_slopes, activation_opt_params


def mip(saved_bounds, labels_to_verify=None):

    lirpa_model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA = saved_bounds
    refined_betas = None

    if arguments.Config["general"]["complete_verifier"] == "mip":
        mip_global_lb, mip_global_ub, mip_status, mip_adv = lirpa_model.build_the_model_mip(labels_to_verify=labels_to_verify, save_adv=True)

        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(-1)  # Missing batch dimension.
        if mip_global_ub.ndim == 1:
            mip_global_ub = mip_global_ub.unsqueeze(-1)  # Missing batch dimension.
        print(f'MIP solved lower bound: {mip_global_lb}')
        print(f'MIP solved upper bound: {mip_global_ub}')

        verified_status = "safe-mip"
        # Batch size is always 1.
        labels_to_check = labels_to_verify if labels_to_verify is not None else range(len(mip_status))
        for pidx in labels_to_check:
            if mip_global_lb[pidx] >= 0:
                # Lower bound > 0, verified.
                continue
            # Lower bound < 0, now check upper bound.
            if mip_global_ub[pidx] <= 0:
                # Must be 2 cases: solved with adv example, or early terminate with adv example.
                assert mip_status[pidx] in [2, 15]
                print("verified unsafe-mip with init mip!")
                return "unsafe-mip", mip_global_lb, None, None, None
            # Lower bound < 0 and upper bound > 0, must be a timeout.
            assert mip_status[pidx] == 9 or mip_status[pidx] == -1, "should only be timeout for label pidx"
            verified_status = "unknown-mip"

        print(f"verified {verified_status} with init mip!")
        return verified_status, mip_global_lb, None, None, None

    elif arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = FSB_score(lirpa_model.net, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

        refined_lower_bounds, refined_upper_bounds, refined_betas = lirpa_model.build_the_model_mip_refine(lower_bounds, upper_bounds,
                            score=score, stop_criterion_func=stop_criterion_min(1e-4))

        # shape of the last layer should be (batch, 1) for verified-acc
        refined_lower_bounds[-1] = refined_lower_bounds[-1].reshape(lower_bounds[-1].shape)
        refined_upper_bounds[-1] = refined_upper_bounds[-1].reshape(upper_bounds[-1].shape)

        lower_bounds, upper_bounds, = refined_lower_bounds, refined_upper_bounds
        refined_global_lb = lower_bounds[-1]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())
        if refined_global_lb.min()>=0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", refined_global_lb, lower_bounds, upper_bounds, None

        return "unknown", refined_global_lb, lower_bounds, upper_bounds, refined_betas
    else:
        return "unknown", -float("inf"), lower_bounds, upper_bounds, refined_betas


def bab(unwrapped_model, data, targets, y, data_ub, data_lb,
        lower_bounds=None, upper_bounds=None, reference_slopes=None,
        attack_images=None, c=None, all_prop=None, cplex_processes=None,
        activation_opt_params=None, reference_lA=None, rhs=None, 
        model_incomplete=None, timeout=None, refined_betas=None):

    norm = arguments.Config["specification"]["norm"]
    eps = arguments.Globals["lp_perturbation_eps"]  # epsilon for non Linf perturbations, None for all other cases.
    if norm != float("inf"):
        # For non Linf norm, upper and lower bounds do not make sense, and they should be set to the same.
        assert torch.allclose(data_ub, data_lb)

    # This will use the refined bounds if the complete verifier is "bab-refine".
    # FIXME do not repeatedly create LiRPAConvNet which creates a new BoundedModule each time.
    model = LiRPAConvNet(unwrapped_model, in_size=data.shape if not targets.size > 1 else [len(targets)] + list(data.shape[1:]),
                         c=c, cplex_processes=cplex_processes)

    data = data.to(model.device)
    data_lb, data_ub = data_lb.to(model.device), data_ub.to(model.device)
    output = model.net(data).flatten()
    print('Model prediction is:', output)

    if arguments.Config['attack']['check_clean']:
        clean_rhs = c.matmul(output)
        print(f'Clean RHS: {clean_rhs}')
        if (clean_rhs < rhs).any():
            return -np.inf, np.inf, None, None, 'unsafe'

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    if cut_enabled:
        model.set_cuts(model_incomplete.A_saved, x, lower_bounds, upper_bounds)

    if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
        min_lb, min_ub, glb_record, nb_states, verified_ret = input_bab_parallel(
            model, domain, x, model_ori=unwrapped_model, all_prop=all_prop,
            rhs=rhs, timeout=timeout, branching_method=arguments.Config["bab"]["branching"]["method"])
    else:
        min_lb, min_ub, glb_record, nb_states, verified_ret = relu_bab_parallel(
            model, domain, x,
            refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
            activation_opt_params=activation_opt_params, reference_lA=reference_lA,
            reference_slopes=reference_slopes, attack_images=attack_images,
            timeout=timeout, refined_betas=refined_betas, rhs=rhs)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    if min_lb is None:
        min_lb = -np.inf
    if isinstance(min_ub, torch.Tensor):
        min_ub = min_ub.item()
    if min_ub is None:
        min_ub = np.inf

    return min_lb, min_ub, nb_states, glb_record, verified_ret


def update_parameters(model, data_min, data_max):
    if 'vggnet16_2022' in arguments.Config['general']['root_path']:
        perturbed = (data_max - data_min > 0).sum()
        if perturbed > 10000:
            print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
            print('Setting arguments.Config["attack"]["pgd_order"] to "before"')
            arguments.Config['attack']['pgd_order'] = 'before'


def sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub, scores, reference_slopes, lA, final_node_name, reverse=False):
    # TODO need minus rhs
    # To sort targets, this must be a classification task, and initial_max_domains
    # is set to 1.
    assert len(batched_vnnlib) == init_global_lb.shape[0] and init_global_lb.shape[1] == 1
    sorted_idx = scores.argsort(descending=reverse)
    batched_vnnlib = [batched_vnnlib[i] for i in sorted_idx]
    init_global_lb = init_global_lb[sorted_idx]
    init_global_ub = init_global_ub[sorted_idx]

    if reference_slopes is not None:
        for m, spec_dict in reference_slopes.items():
            for spec in spec_dict:
                if spec == final_node_name:
                    if spec_dict[spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = spec_dict[spec][:, sorted_idx]
                    else:
                        spec_dict[spec] = spec_dict[spec][:, :, sorted_idx]

    if lA is not None:
        lA = [lAitem[:, sorted_idx] for lAitem in lA]

    return batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx


def complete_verifier(
        model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
        init_global_lb, lower_bounds, upper_bounds, index,
        timeout_threshold, bab_ret=None, lA=None, cplex_processes=None,
        reference_slopes=None, activation_opt_params=None, refined_betas=None, attack_images=None,
        attack_margins=None):
    start_time = time.time()
    cplex_cuts = arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]
    sort_targets = arguments.Config["bab"]["sort_targets"]
    bab_attack_enabled = arguments.Config["bab"]["attack"]["enabled"]

    if arguments.Config["general"]["enable_incomplete_verification"]:
        init_global_ub = upper_bounds[-1]
        print('lA shape:', [lAitem.shape for lAitem in lA])
        if bab_attack_enabled:
            # Sort specifications based on adversarial attack margins.
            batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx = sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub,
                scores=attack_margins.flatten(), reference_slopes=reference_slopes, lA=lA, final_node_name=model_incomplete.net.final_node().name)
            attack_images = attack_images[:, :, sorted_idx]
        else:
            if sort_targets:
                assert not cplex_cuts
                # Sort specifications based on incomplete verifier bounds.
                batched_vnnlib, init_global_lb, init_global_ub, lA, _ = sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub,
                    scores=init_global_lb.flatten(), reference_slopes=reference_slopes, lA=lA, final_node_name=model_incomplete.net.final_node().name)
            if cplex_cuts:
                assert not sort_targets
                # need to sort pidx such that easier first according to initial alpha crown
                batched_vnnlib, init_global_lb, init_global_ub, lA, _ = sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub,
                    scores=init_global_lb.flatten(), reference_slopes=reference_slopes, lA=lA, final_node_name=model_incomplete.net.final_node().name, reverse=True)
        if reference_slopes is not None:
            reference_slopes_cp = copy.deepcopy(reference_slopes)

    solved_c_rows = []

    for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
        # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
        print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
        timeout = timeout_threshold - (time.time() - start_time)
        print(f'Remaining timeout: {timeout}')
        start_time_bab = time.time()

        x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
        data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
        data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
        x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.

        target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)

        assert len(target_label_arrays) == 1
        c, rhs, y, pidx = target_label_arrays[0]

        if bab_attack_enabled:
            if arguments.Config["bab"]["initial_max_domains"] != 1:
                raise ValueError('To run Bab-attack, please set initial_max_domains to 1. '
                                 f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')
            # Attack images has shape (batch, restarts, specs, c, h, w). The specs dimension should already be sorted.
            # Reshape it to (restarts, c, h, w) for this specification.
            this_spec_attack_images = attack_images[:, :, property_idx].view(attack_images.size(1), *attack_images.shape[3:])
        else:
            this_spec_attack_images = None
 
        if arguments.Config["general"]["enable_incomplete_verification"]:
            # extract lower bound by (sorted) init_global_lb and batch size of initial_max_domains
            this_batch_start_idx = property_idx * arguments.Config["bab"]["initial_max_domains"]
            lower_bounds[-1] = init_global_lb[this_batch_start_idx: this_batch_start_idx + c.shape[0]]
            upper_bounds[-1] = init_global_ub[this_batch_start_idx: this_batch_start_idx + c.shape[0]]

            # trim reference slope by batch size of initial_max_domains accordingly
            if reference_slopes is not None:
                for m, spec_dict in reference_slopes.items():
                    for spec in spec_dict:
                        if spec == model_incomplete.net.final_node().name:
                            if reference_slopes_cp[m][spec].size()[1] > 1:
                                # correspond to multi-x case
                                spec_dict[spec] = reference_slopes_cp[m][spec][:, this_batch_start_idx: this_batch_start_idx + c.shape[0]]
                            else:
                                spec_dict[spec] = reference_slopes_cp[m][spec][:, :, this_batch_start_idx: this_batch_start_idx + c.shape[0]]

            # trim lA by batch size of initial_max_domains accordingly
            if lA is not None:
                lA_trim = [Aitem[:, this_batch_start_idx: this_batch_start_idx + c.shape[0]] for Aitem in lA]

        print('##### Instance {} first 10 spec matrices: {}\nthresholds: {} ######'.format(index, c[:10],  rhs.flatten()[:10]))

        if np.array(pidx == y).any():
            raise NotImplementedError

        torch.cuda.empty_cache()
        gc.collect()

        c = torch.tensor(c, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
        rhs = torch.tensor(rhs, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])

        # extract cplex cut filename
        if cplex_cuts:
            assert arguments.Config["bab"]["initial_max_domains"] == 1

        # Complete verification (BaB, BaB with refine, or MIP).
        if arguments.Config["general"]["enable_incomplete_verification"]:
            assert not arguments.Config["bab"]["branching"]["input_split"]["enable"]
            # Reuse results from incomplete results, or from refined MIPs.
            # skip the prop that already verified
            rlb = list(lower_bounds)[-1]
            if arguments.Config["data"]["num_outputs"] != 1:
                init_verified_cond = rlb.flatten() > rhs.flatten()
                init_verified_idx = np.array(torch.where(init_verified_cond)[0].cpu())
                if init_verified_idx.size > 0:
                    print(f"Initial alpha-CROWN verified for spec index {init_verified_idx} with bound {rlb[init_verified_idx].squeeze()}.")
                    l, ret = init_global_lb[init_verified_idx].cpu().numpy().tolist(), 'safe'
                    bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                init_failure_idx = np.array(torch.where(~init_verified_cond)[0].cpu())
                if init_failure_idx.size == 0:
                    # This batch of x verified by init opt crown
                    continue
                print(f"Remaining spec index {init_failure_idx} with "
                      f"bounds {rlb[init_failure_idx]} need to verify.")
                assert len(np.unique(y)) == 1 and len(rhs.unique()) == 1
            else:
                init_verified_cond, init_failure_idx, y = torch.tensor([1]), np.array(0), np.array(0)

            if reference_slopes is not None:
                LiRPAConvNet.prune_reference_slopes(reference_slopes, ~init_verified_cond, model_incomplete.net.final_node().name)
            if lA is not None:
                lA_trim = LiRPAConvNet.prune_lA(lA_trim, ~init_verified_cond)

            lower_bounds[-1] = lower_bounds[-1][init_failure_idx]
            upper_bounds[-1] = upper_bounds[-1][init_failure_idx]
            # TODO change index [0:1] to [torch.where(~init_verified_cond)[0]] can handle more general vnnlib for multiple x
            l, u, nodes, glb_record, ret = bab(
                model_ori, x[0:1], init_failure_idx, y=np.unique(y),
                data_ub=data_max[0:1], data_lb=data_min[0:1],
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                c=c[torch.where(~init_verified_cond)[0]],
                reference_slopes=reference_slopes, cplex_processes=cplex_processes, rhs=rhs[0:1],
                activation_opt_params=activation_opt_params, reference_lA=lA_trim,
                model_incomplete=model_incomplete, timeout=timeout, refined_betas=refined_betas,
                attack_images=this_spec_attack_images)
            bab_ret.append([index, float(l), nodes, time.time() - start_time_bab, init_failure_idx.tolist()])
        else:
            assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
            assert not arguments.Config["bab"]["attack"]["enabled"], "BaB-attack must be used with incomplete verifier."
            # input split also goes here directly
            l, u, nodes, _, ret = bab(
                model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min, c=c,
                all_prop=target_label_arrays, cplex_processes=cplex_processes,
                rhs=rhs, timeout=timeout, attack_images=this_spec_attack_images)
            bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])

        # terminate the corresponding cut inquiry process if exists
        if cplex_cuts:
            solved_c_rows.append(c)
            terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

        timeout = timeout_threshold - (time.time() - start_time)
        if ret == 'unsafe':
            return 'unsafe-bab'
        if ret == 'unknown' or timeout < 0:
            return 'unknown'
        if ret != 'safe':
            raise ValueError(f'Unknown return value of bab: {ret}')
    else:
        return 'safe'


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    torch.set_printoptions(precision=8)
    device = arguments.Config["general"]["device"]
    if device != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)

    if arguments.Config["general"]["precompile_jit"]:
        precompile_jit_kernels()

    if arguments.Config["specification"]["norm"] != np.inf and arguments.Config["attack"]["pgd_order"] != "skip":
        print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
        arguments.Config["attack"]["pgd_order"] = "skip"

    run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, shape = parse_run_mode()
    verification_summary = defaultdict(list)
    time_all_instances = []
    status_per_sample_list = []
    bab_ret = []
    cnt = 0  # Number of examples in this run.
    select_instance = arguments.Config["data"]["select_instance"]

    for new_idx, csv_item in enumerate(example_idx_list):
        arguments.Globals["example_idx"] = new_idx
        vnnlib_id = new_idx + arguments.Config["data"]["start"]

        # Select some instances to verify
        if select_instance and not vnnlib_id in select_instance:
            continue

        start_time = time.time()
        print(f'\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx: {new_idx}, vnnlib ID: {vnnlib_id} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        if run_mode != 'customized_data':
            if len(csv_item) == 3:
                # model, vnnlib, timeout
                onnx_path, vnnlib_path, arguments.Config["bab"]["timeout"] = csv_item
                onnx_path = os.path.join(arguments.Config["model"]["onnx_path_prefix"], onnx_path.strip())
                vnnlib_path = os.path.join(arguments.Config["specification"]["vnnlib_path_prefix"], vnnlib_path.strip())
                print(f'Using onnx {onnx_path}')
                print(f'Using vnnlib {vnnlib_path}')
                model_ori, shape, vnnlib = eval(arguments.Config["model"]["onnx_loader"])(file_root, onnx_path, vnnlib_path)
            else:
                # Each line contains only 1 item, which is the vnnlib spec.
                vnnlib = read_vnnlib(csv_item[0])
                assert arguments.Config["model"]["input_shape"] is not None, 'vnnlib does not have shape information, please specify by --input_shape'
                shape = arguments.Config["model"]["input_shape"]
        else:
            vnnlib = vnnlib_all[new_idx]

        arguments.Config["bab"]["timeout"] = float(arguments.Config["bab"]["timeout"])
        if arguments.Config["bab"]["timeout_scale"] != 1:
            new_timeout = arguments.Config["bab"]["timeout"] * arguments.Config["bab"]["timeout_scale"]
            print(f'Scaling timeout: {arguments.Config["bab"]["timeout"]} -> {new_timeout}')
            arguments.Config["bab"]["timeout"] = new_timeout
        if arguments.Config["bab"]["override_timeout"] is not None:
            new_timeout = arguments.Config["bab"]["override_timeout"]
            print(f'Overriding timeout: {new_timeout}')
            arguments.Config["bab"]["timeout"] = new_timeout
        timeout_threshold = arguments.Config["bab"]["timeout"]  # In case arguments.Config["bab"]["timeout"] is changed later.


        if arguments.Config["data"]["dataset"] == 'NN4SYS':
            # FIXME_NOW: Remove this case. Should be handled generally.
            verified_status = res = nn4sys_verification(model_ori, vnnlib, onnx_path=os.path.join(file_root, onnx_path))
            print(res)
        else:
            model_ori.eval()
            vnnlib_shape = shape

            # FIXME attack and initial_incomplete_verification only works for assert len(vnnlib) == 1
            x_range = torch.tensor(vnnlib[0][0], dtype=torch.get_default_dtype())
            data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
            data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
            x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.

            # auto tune args
            update_parameters(model_ori, data_min, data_max)

            model_ori = model_ori.to(device)
            x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)

            verified_status = "unknown"
            verified_success = False                

            if arguments.Config["attack"]["pgd_order"] == "before":
                verified_status, verified_success, attack_images, attack_margins, all_adv_candidates = attack(
                    model_ori, x, data_min, data_max, vnnlib,
                    verified_status, verified_success)
            else:
                attack_images = attack_margins = all_adv_candidates = None

            init_global_lb = saved_bounds = saved_slopes = y = lower_bounds = upper_bounds = None
            activation_opt_params = model_incomplete = lA = cplex_processes = None

            # Incomplete verification is enabled by default. The intermediate lower
            # and upper bounds will be reused in bab and mip.
            if (not verified_success and (
                    arguments.Config["general"]["enable_incomplete_verification"]
                    or arguments.Config["general"]["complete_verifier"] == "bab-refine")):
                assert len(vnnlib) == 1
                verified_status, init_global_lb, saved_bounds, saved_slopes, activation_opt_params = \
                    incomplete_verifier(model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib)
                verified_success = verified_status != "unknown"
                if not verified_success:
                    model_incomplete, lower_bounds, upper_bounds = saved_bounds[:3]
                    lA = saved_bounds[-1]

            if not verified_success and arguments.Config["attack"]["pgd_order"] == "after":
                verified_status, verified_success, attack_images, attack_margins, all_adv_candidates = attack(
                    model_ori, x, data_min, data_max, vnnlib,
                    verified_status, verified_success)

            # MIP or MIP refined bounds.
            refined_betas = None
            if not verified_success and (arguments.Config["general"]["complete_verifier"] == "mip" or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
                # rhs = ? NEED TO SAVE TO LIRPA_MODULE
                verified_status, init_global_lb, lower_bounds, upper_bounds, refined_betas = mip(saved_bounds=saved_bounds)
                verified_success = verified_status != "unknown"

            # extract the process pool for cut inquiry
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                if saved_bounds is not None:
                    # use nullity of saved_bounds as an indicator of whether cut processes are launched
                    # saved_bounds[0] is the AutoLiRPA model instance
                    cplex_processes = saved_bounds[0].processes
                    mip_building_proc = saved_bounds[0].mip_building_proc

            # BaB bounds. (not do bab if unknown by mip solver for now)
            if not verified_success and arguments.Config["general"]["complete_verifier"] != "skip" and verified_status != "unknown-mip":
                batched_vnnlib = batch_vnnlib(vnnlib)
                verified_status = complete_verifier(
                    model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
                    init_global_lb, lower_bounds, upper_bounds, new_idx,
                    timeout_threshold=timeout_threshold - (time.time() - start_time),
                    bab_ret=bab_ret, lA=lA, cplex_processes=cplex_processes,
                    reference_slopes=saved_slopes, activation_opt_params=activation_opt_params,
                    refined_betas=refined_betas, attack_images=all_adv_candidates, attack_margins=attack_margins)

            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and saved_bounds is not None:
                terminate_mip_processes(mip_building_proc, cplex_processes)
                del cplex_processes

            del init_global_lb, saved_bounds, saved_slopes

        # Summarize results.
        if run_mode == 'single_vnnlib':
            # run in run_instance.sh
            if 'unknown' in verified_status or 'timeout' in verified_status or 'timed out' in verified_status:
                verified_status = 'timeout'
            elif 'unsafe' in verified_status:
                verified_status = 'sat'
            elif 'safe' in verified_status:
                verified_status = 'unsat'
            else:
                raise ValueError(f'Unknown verified_status {verified_status}')

            print('Result:', verified_status)
            print('Time:', time.time() - start_time)
            with open(save_path, "w") as file:
                file.write(verified_status)
                if arguments.Config["general"]["save_adv_example"]:
                    if verified_status == 'sat':
                        file.write('\n')
                        with open(arguments.Config["attack"]["cex_path"], "r") as adv_example:
                            file.write(adv_example.read())
                file.flush()
        else:
            cnt += 1
            if time.time() - start_time > timeout_threshold:
                if 'unknown' not in verified_status:
                    verified_status += ' (timed out)'
            verification_summary[verified_status].append(new_idx)
            status_per_sample_list.append([verified_status, time.time() - start_time])  # [status, time]
            with open(save_path, "wb") as f:
                pickle.dump({"summary": verification_summary, "results": status_per_sample_list,  "bab_ret": bab_ret}, f)
            print(f"Result: {verified_status} in {status_per_sample_list[-1][1]:.4f} seconds")

    if run_mode != 'single_vnnlib':
        # Finished all examples.
        time_timeout = [s[1] for s in status_per_sample_list if "unknown" in s[0]]
        time_verified = [s[1] for s in status_per_sample_list if "safe" in s[0] and "unsafe" not in s[0]]
        time_unsafe = [s[1] for s in status_per_sample_list if "unsafe" in s[0]]
        time_all_instances = [s[1] for s in status_per_sample_list]

        with open(save_path, "wb") as f:
            pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "bab_ret": bab_ret}, f)

        print("############# Summary #############")
        print("Final verified acc: {}% (total {} examples)".format(len(time_verified) / len(example_idx_list) * 100., len(example_idx_list)))
        print("Problem instances count:", len(time_verified) + len(time_unsafe) + len(time_timeout), ", total verified (safe/unsat):", len(time_verified),
              ", total falsified (unsafe/sat):", len(time_unsafe), ", timeout:", len(time_timeout))
        print(f"mean time for ALL instances (total {len(time_all_instances)}): {sum(time_all_instances)/(len(time_all_instances) + 1e-5)}, max time: {max(time_all_instances)}")
        if len(time_verified) > 0:
            print(f"mean time for verified SAFE instances (total {len(time_verified)}): "
                  f"{sum(time_verified) / len(time_verified)}, max time: {max(time_verified)}")
        if len(time_verified) > 0 and len(time_unsafe) > 0:
            print(f"mean time for verified (SAFE + UNSAFE) instances (total {(len(time_verified) + len(time_unsafe))}):"
                  f" {(sum(time_verified) + sum(time_unsafe)) / (len(time_verified) + len(time_unsafe))}, max time: "
                  f"{max(max(time_verified), max(time_unsafe))}")
        if len(time_verified) > 0 and len(time_timeout) > 0:
            print(f"mean time for verified SAFE + TIMEOUT instances (total {(len(time_verified) + len(time_timeout))}):"
                  f" {(sum(time_verified) + sum(time_timeout)) / (len(time_verified) + len(time_timeout))}, max time: "
                  f"{max(max(time_verified), max(time_timeout))}")
        if len(time_unsafe) > 0:
            print(f"mean time for verified UNSAFE instances (total {len(time_unsafe)}): "
                  f"{sum(time_unsafe) / len(time_unsafe)}, max time: {max(time_unsafe)}")

        for k, v in verification_summary.items():
            print(f"{k} (total {len(v)}), index:", v)


if __name__ == "__main__":
    arguments.Config.parse_config()
    main()
