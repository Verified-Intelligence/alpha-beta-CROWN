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
"""alpha-beta-CROWN verifier interface to handle robustness verification."""

import os
import re
import socket
import random
import time
import gc

from utils import get_test_acc, load_model, load_verification_dataset

import numpy as np
import pandas as pd

import torch
import arguments
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from bab_verification_general import mip, incomplete_verifier, bab
from attack_pgd import pgd_attack
from utils import Normalization, get_pgd_acc


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "crown-only-verified-acc", "alpha-crown-only-verified-acc", "ibp-only-verified-acc", "attack-only", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20). Mode can also be set as "crown-only-verified-acc" or "alpha-crown-only-verified-acc", which quickly computes the verified accuracy over the entire dataset via CROWN or alpha-CROWN.', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="please_specify_model_name", help='Name of model. Model must be defined in the load_verification_dataset() function in utils.py.', hierarchy=h + ["name"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "CIFAR_SDP_FULL", "CIFAR_RESNET", "CIFAR_SAMPLE", "MNIST_SAMPLE", "CIFAR_ERAN", "MNIST_ERAN",
                                 "MNIST_ERAN_UN", "MNIST_SDP", "MNIST_MADRY_UN", "CIFAR_SDP", "CIFAR_UN"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])
    arguments.Config.add_argument("--data_idx_file", type=str, default=None, help='A text file with a list of example IDs to run.', hierarchy=h + ["data_idx_file"])

    h = ["attack"]
    arguments.Config.add_argument("--mip_attack", action='store_true', help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.', hierarchy=h + ["enable_mip_attack"])
    arguments.Config.add_argument('--pgd_steps', type=int, default=100, help="Steps of PGD attack.", hierarchy=h + ["pgd_steps"])
    arguments.Config.add_argument('--pgd_restarts', type=int, default=30, help="Number of random PGD restarts.", hierarchy= h + ["pgd_restarts"])
    arguments.Config.add_argument('--no_pgd_early_stop', action='store_false', dest='pgd_early_stop', help="Early stop PGD when an adversarial example is found.", hierarchy=h + ["pgd_early_stop"])
    arguments.Config.add_argument('--pgd_lr_decay', type=float, default=0.99, help='Learning rate decay factor used in PGD attack.', hierarchy= h + ["pgd_lr_decay"])
    arguments.Config.add_argument('--pgd_alpha', type=str, default="auto", help='Step size of PGD attack. Default (auto) is epsilon/4.', hierarchy=h + ["pgd_alpha"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option, do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()


def get_statistics(model, image, true_label, eps, data_min, data_max, batch_size, method="CROWN"):
    """For quickly checking clean accuracy and CROWN verified accuracy."""
    assert method == "CROWN" or method == "alpha-CROWN" or method == "IBP"
    # Clearn accuracy
    predicted = model(image)
    n_correct = (predicted.argmax(dim=1) == true_label).sum().item()
    print(f'{n_correct} examples are correct, image range ({image.min()}, {image.max()})')

    # CROWN verified accuracy
    verified = 0
    N = image.size(0)
    num_outputs = arguments.Config["data"]["num_outputs"]
    norm = np.inf
    assert norm == arguments.Config["specification"]["norm"]  # TODO: make this function support more norms.
    model = BoundedModule(model, torch.empty_like(image[:batch_size]), device=arguments.Config["general"]["device"])
    if method == 'alpha-CROWN':
        # Set alpha-CROWN optimization parameters.
        lr_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        model.set_bound_opts({'optimize_bound_args': {'ob_iteration': iteration, 'ob_beta': False, 'ob_alpha': True,
                              'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                              'ob_lr': lr_alpha, 'ob_lr_decay': lr_decay}})
    batch_idx = 0
    all_start_time = time.time()
    while batch_idx * batch_size < N:
        start_time = time.time()
        start_idx, end_idx = batch_idx*batch_size, min(batch_idx*batch_size+batch_size, N)
        data, labels = image[start_idx:end_idx], torch.tensor(true_label[start_idx:end_idx])
        if arguments.Config["specification"]["type"] == "lp":
            # Linf norm only so far.
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
        else:
            # Per-example, per-element lower and upper bounds.
            data_ub = data_max[start_idx:end_idx]
            data_lb = data_min[start_idx:end_idx]
        data, data_lb, data_ub, labels = data.cuda(), data_lb.cuda(), data_ub.cuda(), labels.cuda()
        ptb = PerturbationLpNorm(norm=norm, eps=None, x_L=data_lb, x_U=data_ub)
        data = BoundedTensor(data, ptb)
        # labels = torch.argmax(pred, dim=1).cpu().detach().numpy()
        c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_outputs - 1, num_outputs)).cuda()
        if method == "CROWN" or method == "IBP":
            with torch.no_grad():
                lb, ub = model.compute_bounds(x=(data,), method=method, C=c, bound_upper=False)
        else:
            # alpha-CROWN requires gradient.
            lb, ub = model.compute_bounds(x=(data,), method="CROWN-optimized", C=c, bound_upper=False)
        verified += (lb.min(1)[0]>=0).sum().item()
        # Print some bounds for the first batch for debugging.
        duration = time.time() - start_time
        if batch_idx == 0:
            print("Bounds for first a few examples:")
            print(lb[:10].detach().cpu().numpy())
        print(f"batch: {batch_idx}, verified acc: {(lb.min(1)[0]>=0).sum().item()} / {data.size(0)}, time {duration}")
        del lb, ub
        batch_idx += 1

    full_duration = time.time() - all_start_time
    print(f"{method} verified acc: {verified/N * 100}%, {verified} verified, time {full_duration}")


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

    if arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)

    if arguments.Config["attack"]["pgd_order"] != "skip":
        if arguments.Config["specification"]["type"] == "lp" and arguments.Config["specification"]["norm"] != np.inf:
            print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
            arguments.Config["attack"]["pgd_order"] = "skip"

    model_ori = load_model(weights_loaded=True)

    if arguments.Config["specification"]["epsilon"] is not None:
        perturb_epsilon = torch.tensor(arguments.Config["specification"]["epsilon"], dtype=torch.get_default_dtype())
    else:
        print('No epsilon defined!')
        perturb_epsilon = None

    X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_verification_dataset(perturb_epsilon)

    if arguments.Config["general"]["mode"] == "clean-acc":
        print("Testing clean accuracy.")
        get_test_acc(model_ori, X=X, labels=labels, batch_size=arguments.Config["solver"]["beta-crown"]["batch_size"])
        return

    if "MNIST" in arguments.Config["data"]["dataset"]:
        attack_dataset = "MNIST"
    elif "CIFAR" in arguments.Config["data"]["dataset"]:
        attack_dataset = "CIFAR"
    else:
        attack_dataset = "UNKNOWN"

    if arguments.Config["specification"]["type"] == 'lp':
        if perturb_epsilon is None:
            raise ValueError("Perturbation epsilon is not set by data loader. Do you mean to use the 'bound' type specification? Try adding --spec_type bound")
        if arguments.Config["specification"]["epsilon"] is not None:
            print(f"epsilon after preprocessing: {perturb_epsilon}, data_max = {data_max}, data_min = {data_min}")
        if data_max.size(0) != 1 or data_min.size(0) != 1:
            raise ValueError("For 'lp' type specification, you need rabsolute (global) lower and upper bounds, not per example bounds.")
    elif arguments.Config["specification"]["type"] == 'bound':
        print(f'Loaded datasets with per-element lower and upper bounds: max = {data_max.max().item()}, min = {data_min.min().item()}')
        if data_max.size(0) != X.size(0) or data_min.size(0) != X.size(0):
            raise ValueError("For 'bound' type specification, you need per example lower and upper bounds.")
        if perturb_epsilon is None:
            perturb_epsilon = (data_max - data_min).mean() / 2.
            print(f'eps set to {perturb_epsilon}. This will not be used for certification, but will be used to determine PGD step size.')

    if arguments.Config["data"]["data_idx_file"] is not None:
        # Go over a list of data indices.
        with open(arguments.Config["data"]["data_idx_file"]) as f:
            bnb_ids = re.split(r'[;|,|\n|\s]+', f.read().strip())
            bnb_ids = [int(b_id) for b_id in bnb_ids]
            print(f'Example indices (total {len(bnb_ids)}): {bnb_ids}')
    else:
        # By default, we go over all data.
        bnb_ids = list(range(X.shape[0]))

    bnb_ids = bnb_ids[arguments.Config["data"]["start"]:  arguments.Config["data"]["end"]]
    print('Task length:', len(bnb_ids))

    save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npy'. \
        format(arguments.Config['model']['name'], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
               arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
               arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"])
    print(f'saving results to {save_path}')

    if arguments.Config["general"]["mode"] == "crown-only-verified-acc":
        get_statistics(model_ori, X, labels, perturb_epsilon, data_min, data_max, batch_size=arguments.Config["solver"]["beta-crown"]["batch_size"])
        return
    if arguments.Config["general"]["mode"] == "alpha-crown-only-verified-acc":
        get_statistics(model_ori, X, labels, perturb_epsilon, data_min, data_max, batch_size=arguments.Config["solver"]["beta-crown"]["batch_size"], method="alpha-CROWN")
        return
    if arguments.Config["general"]["mode"] == "ibp-only-verified-acc":
        get_statistics(model_ori, X, labels, perturb_epsilon, data_min, data_max, batch_size=arguments.Config["solver"]["beta-crown"]["batch_size"], method="IBP")
        return
    if arguments.Config["general"]["mode"] == "attack-only":
        get_pgd_acc(model_ori, X, labels, perturb_epsilon, data_min, data_max, batch_size=arguments.Config["solver"]["beta-crown"]["batch_size"])

    ret, lb_record, attack_success = [], [], []
    mip_unsafe, mip_safe, mip_unknown = [], [], []
    verified_acc = len(bnb_ids)
    verified_failed = []
    verified_success_list = []
    example_time = []
    skipped_examples = []
    nat_acc = len(bnb_ids)
    orig_timeout = arguments.Config["bab"]["timeout"]

    model_ori, all_data_max, all_data_min = model_ori.to(arguments.Config["general"]["device"]), data_max.to(arguments.Config["general"]["device"]), data_min.to(arguments.Config["general"]["device"])
    if isinstance(perturb_epsilon, torch.Tensor):
        perturb_eps = perturb_epsilon.to(arguments.Config["general"]["device"])

    for new_idx, imag_idx in enumerate(bnb_ids):
        arguments.Config["bab"]["timeout"] = orig_timeout
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, 'img ID:', imag_idx, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        torch.cuda.empty_cache()
        gc.collect()

        x, y = X[imag_idx], int(labels[imag_idx].item())
        x = x.unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
        if arguments.Config["specification"]["type"] == 'bound':
            data_min = all_data_min[imag_idx].unsqueeze(0)
            data_max = all_data_max[imag_idx].unsqueeze(0)
        else:
            data_min = all_data_min
            data_max = all_data_max
        # first check the model is correct at the input
        logit_pred = model_ori(x)[0]
        if logit_pred.size(0) > 1:
            # Multi-class.
            y_pred = torch.max(logit_pred, 0)[1].item()
        else:
            # Binary classifier: logit_pred > 0 => label 1, otherwise label 0.
            y_pred = int(logit_pred.item() > 0)

        if type(perturb_epsilon) is list:
            # Each image has different epsilon (e.g., OVAL 20).
            perturb_eps = perturb_epsilon[imag_idx].to(arguments.Config["general"]["device"])

        print(f'predicted label {y_pred}, correct label {y}, image norm {x.abs().sum().item()}, logits {logit_pred}')
        if y_pred != y:
            print(f'Result: image {imag_idx} prediction is incorrect, skipped.')
            skipped_examples.append(imag_idx)
            verified_acc -= 1
            nat_acc -= 1
            # attack_success.append(imag_idx)
            continue
        # else:
        #     # enable here to check clean acc
        #     continue

        verified_success = False
        verified_status = "unknown"
        attack_margin = None
        attack_images = None
        example_start_time = time.time()

        if arguments.Config["attack"]["pgd_order"] == "before":
            start_attack = time.time()
            attack_args = {'dataset': attack_dataset, 'model': model_ori, 'x': x, 'max_eps': perturb_eps, 'data_min': data_min, 'data_max': data_max, 'y': y}
            attack_ret, attack_images, attack_margin = pgd_attack(**attack_args)
            ret.append([imag_idx, 0, 0, time.time()-start_attack, new_idx, -3, np.inf, np.inf])
            if attack_ret:
                # Attack success.
                verified_status = "unsafe-pgd"
                verified_acc -= 1
                attack_success.append(imag_idx)
                print(f"Result: image {imag_idx} attack success!")
                example_time.append(time.time() - example_start_time)
                print(f'Wall time: {example_time[-1]}')
                continue
        # continue  # uncomment for checking pgd attacking results

        init_global_lb = saved_bounds = saved_slopes = None

        # Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.
        if not verified_success and (arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
            start_incomplete = time.time()
            data = x
            if arguments.Config["specification"]["type"] == 'lp':
                # Lp norm.
                if arguments.Config["specification"]["norm"] == np.inf:
                    if data_max is None:
                        data_ub = data + perturb_eps  # perturb_eps is already normalized.
                        data_lb = data - perturb_eps
                    else:
                        data_ub = torch.min(data + perturb_eps, data_max)
                        data_lb = torch.max(data - perturb_eps, data_min)
                else:
                    data_ub = data_lb = data
            elif arguments.Config["specification"]["type"] == 'bound':
                # Given lower and upper bounds *per example per pixel*.
                data_lb = data_min
                data_ub = data_max
            else:
                raise ValueError(f'Unexpected specification type {arguments.Config["specification"]["type"]}')
            verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(model_ori, x,
                        y, data_ub=data_ub, data_lb=data_lb, eps=perturb_eps)
            verified_success = verified_status != "unknown"
            if not verified_success:
                lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
            arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)
            ret.append([imag_idx, 0, 0, time.time()-start_incomplete, new_idx, -1, np.inf, np.inf])

        if verified_success:
            print(f"Result: image {imag_idx} verification success (with incomplete verifier)!")
            verified_success_list.append(imag_idx)
            example_time.append(time.time() - example_start_time)
            print(f'Wall time: {example_time[-1]}')
            continue

        if arguments.Config["attack"]["pgd_order"] == "after":
            start_attack = time.time()
            attack_args = {'dataset': attack_dataset, 'model': model_ori, 'x': x, 'max_eps': perturb_eps, 'data_min': data_min, 'data_max': data_max, 'y': y}
            attack_ret, attack_images, attack_margin = pgd_attack(**attack_args)
            ret.append([imag_idx, 0, 0, time.time()-start_attack, new_idx, -3, np.inf, np.inf])
            if attack_ret:
                # Attack success.
                verified_status = "unsafe-pgd"
                verified_acc -= 1
                attack_success.append(imag_idx)
                print(f"Result: image {imag_idx} attack success!")
                example_time.append(time.time() - example_start_time)
                print(f'Wall time: {example_time[-1]}')
                continue
            elif arguments.Config["attack"]["enable_mip_attack"]:
                c = torch.eye(arguments.Config["data"]["num_outputs"]).type_as(data)[[y]].unsqueeze(1) - torch.eye(arguments.Config["data"]["num_outputs"]).type_as(data).unsqueeze(0)
                lirpa_model, lower_bounds, upper_bounds, masks = saved_bounds[:4]
                lirpa_model.build_mip_model(lower_bounds, upper_bounds, arguments.Config["bab"]["timeout"], arguments.Config["solver"]["mip"]["parallel_solvers"], arguments.Config["solver"]["mip"]["solver_threads"])
                total_unstable = 0
                for layer_i, m in enumerate(masks):
                    unstable = int(m.sum().item())
                    total_unstable += unstable
                    print(f'layer {layer_i} has {unstable} unstable neurons')
                print(f'Total {total_unstable} unstable neurons.')

                attack_ret = False
                labels_to_verify = attack_margin.argsort().squeeze().tolist()
                print('Sorted order for labels to attack:', labels_to_verify)
                for target in labels_to_verify:
                    if target != y:
                        if init_global_lb[0][target].item() > 0:
                            print(f'Label {target} is already verified.')
                            continue
                        attack_image_target = target if target < y else target - 1
                        adv_pool = AdvExamplePool(lirpa_model.net, masks, C=c[:, target:target+1])
                        # Add adversarial image for the specific target only.
                        adv_pool.add_adv_images(attack_images[:, :, attack_image_target].view((-1, *attack_images.shape[-3:])))
                        neuron_idx, coeff = adv_pool.get_activation_pattern_from_pool()
                        # The initial starting point and activation pattern has a batch dimension because there can be multiple initializations.
                        selected_advs = adv_pool.adv_pool
                        best_adv = torch.stack([adv.x for adv in selected_advs], dim=0)
                        best_adv_pattern = [torch.stack([adv.activation_pattern[layer_i] for adv in selected_advs], dim=0) for layer_i in range(adv_pool.nlayers)]
                        print(f'Best adv example in pool: {adv_pool.adv_pool[0].obj}, worse {adv_pool.adv_pool[-1].obj}')
                        print(f'Target label {target} has {len(coeff)} out of {total_unstable} unstable neurons fixed.')
                        attack_ret, solver_results = lirpa_model.update_mip_model_fix_relu([neuron_idx], [coeff], target, arguments.Config["solver"]["mip"]["parallel_solvers"], arguments.Config["solver"]["mip"]["solver_threads"],
                                async_mip=False, best_adv=[best_adv], adv_activation_pattern=[best_adv_pattern])
                        with torch.no_grad():
                            pred = lirpa_model.net(solver_results[0][3].to(lirpa_model.net.device)).squeeze(0)
                            attack_margin = pred[y] - pred
                            print(f"attack margin: {attack_margin}, for label {target}: {pred[y] - pred[target]}")
                        if attack_ret:
                            break
                if attack_ret:
                    # Attack success.
                    verified_status = "unsafe-mip_attack"
                    verified_acc -= 1
                    attack_success.append(imag_idx)
                    print(f"Result: image {imag_idx} attack success!")
                    example_time.append(time.time() - example_start_time)
                    print(f'Wall time: {example_time[-1]}')
                    continue

        if arguments.Config["general"]["mode"] == "verified-acc":
            if arguments.Config["attack"]["pgd_order"] != "skip":
                # .reshape(-1) to handle the case where there is only 1 element.
                labels_to_verify = attack_margin.argsort().squeeze().reshape(-1).tolist()
                print('Sorted order for labels to attack:', labels_to_verify)
            elif arguments.Config["general"]["enable_incomplete_verification"]:
                # We have initial incomplete bounds.
                labels_to_verify = init_global_lb.argsort().squeeze(0).tolist()
            else:
                labels_to_verify = list(range(arguments.Config["data"]["num_outputs"]))
        elif arguments.Config["general"]["mode"] == "runnerup":
            labels_to_verify = [logit_pred.argsort(descending=True)[1]]
        elif arguments.Config["general"]["mode"] == "specify-target":
            labels_to_verify = [target_label[imag_idx]]
        else:
            raise ValueError("unknown verification mode")

        # MIP or MIP refined bounds.
        if not verified_success and (arguments.Config["general"]["complete_verifier"] == "mip" or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
            start_refine = time.time()
            verified_status, init_global_lb, lower_bounds, upper_bounds = mip(saved_bounds=saved_bounds, y=y, labels_to_verify=labels_to_verify)
            verified_success = "unknown" not in verified_status
            if verified_status == "unknown-mip" or verified_status == "unknown":
                verified_acc -= 1
                mip_unknown.append(imag_idx)
            elif verified_status == "unsafe-mip":
                verified_acc -= 1
                mip_unsafe.append(imag_idx)
            elif verified_status == "safe-mip" or verified_status == "safe-incomplete-refine":
                mip_safe.append(imag_idx)
            arguments.Config["bab"]["timeout"] -= (time.time()-start_refine)
            ret.append([imag_idx, 0, 0, time.time()-start_refine, new_idx, -2, np.inf, np.inf])
            print("time threshold left for bab:", arguments.Config["bab"]["timeout"])

        if verified_success:
            if verified_status == "safe-mip":
                print(f"Result: image {imag_idx} verification success (with mip)!")
                verified_success_list.append(imag_idx)
            elif verified_status == "safe-incomplete-refine":
                print(f"Result: image {imag_idx} verification success (with mip refine)!")
                verified_success_list.append(imag_idx)
            elif verified_status == "unsafe-mip":
                print(f"Result: image {imag_idx} attack success (with mip)!")
                attack_success.append(imag_idx)
            else:
                print(f"Warning: verified status {verified_status} not supported!")
                exit()
            example_time.append(time.time() - example_start_time)
            print(f'Wall time: {example_time[-1]}')
            continue
        elif arguments.Config["general"]["complete_verifier"] == 'skip':
            print(f"Result: image {imag_idx} verification failure (complete verifier skipped as requested).")
            verified_acc -= 1
            verified_failed.append(imag_idx)
            example_time.append(time.time() - example_start_time)
            print(f'Wall time: {example_time[-1]}')
            continue
        elif verified_status == "unknown-mip":
            print(f"Result: image {imag_idx} verification failure (with mip).")
            example_time.append(time.time() - example_start_time)
            print(f'Wall time: {example_time[-1]}')
            continue

        pidx_all_verified = True
        bab_attack_success = False
        for pidx in labels_to_verify:
            if isinstance(pidx, torch.Tensor):
                pidx = pidx.item()
            print('##### [{}:{}] Tested against {} ######'.format(new_idx, imag_idx, pidx))
            if pidx == y:
                print("groundtruth label, skip!")
                ret.append([imag_idx, 0, 0, 0, new_idx, pidx, np.inf, np.inf])
                continue

            torch.cuda.empty_cache()
            gc.collect()

            start_inner = time.time()

            # attack_images shape: (1, batch, restarts, num_outputs-1, c, h, w)
            # select target label attack_images according to pidx. New shape (restarts, c, h, w).
            targeted_attack_images = None
            if attack_images is not None:
                targeted_attack_images = attack_images[0, :, pidx if pidx < y else pidx - 1]
                attack_args.update({'target': pidx, 'only_target_attack': True})
                attack_args.update({'data_max': torch.min(x + perturb_eps, data_max)})
                attack_args.update({'data_min': torch.max(x - perturb_eps, data_min)})
                arguments.attack_args = attack_args
            else:
                arguments.attack_args = None

            try:
                if arguments.Config["general"]["enable_incomplete_verification"]:
                    # Reuse results from incomplete results, or from refined MIPs.
                    # skip the prop that already verified
                    rlb, rub = list(lower_bounds), list(upper_bounds)
                    rlb[-1] = rlb[-1][0, pidx]
                    rub[-1] = rub[-1][0, pidx]
                    # print(init_global_lb[0].min().item(), init_global_lb[0].min().item() - arguments.Config["bab"]["decision_thresh"] <= -100.)
                    if init_global_lb[0].min().item() - arguments.Config["bab"]["decision_thresh"] <= -100.:
                        print(f"Initial alpha-CROWN with poor bound {init_global_lb[0].min().item()}. We will run not branch and bound.")
                        l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                    elif init_global_lb[0, pidx] >= arguments.Config["bab"]["decision_thresh"]:
                        print(f"Initial alpha-CROWN verified for label {pidx} with bound {init_global_lb[0, pidx]}")
                        l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                    else:
                        if arguments.Config["bab"]["timeout"] < 0:
                            print(f"Image {imag_idx} verification failure (running out of time budget).")
                            l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                        else:
                            # feed initialed bounds to save time
                            l, u, nodes, glb_record = bab(model_ori, x, pidx, y=y, eps=perturb_eps, data_ub=data_max, data_lb=data_min,
                                           lower_bounds=lower_bounds, upper_bounds=upper_bounds, reference_slopes=saved_slopes, attack_images=targeted_attack_images)
                else:
                    assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
                    # Main function to run verification
                    l, u, nodes, glb_record = bab(model_ori, x, pidx, y=y, eps=perturb_eps,
                                                  data_ub=data_max, data_lb=data_min, attack_images=targeted_attack_images)
                time_cost = time.time() - start_inner
                print('Image {} label {} verification end, final lower bound {}, upper bound {}, time: {}'.format(imag_idx, pidx, l, u, time_cost))
                ret.append([imag_idx, l, nodes, time_cost, new_idx, pidx, u, attack_margin[pidx] if attack_margin is not None else np.inf])
                arguments.Config["bab"]["timeout"] -= time_cost
                lb_record.append([glb_record])
                print(imag_idx, l)
                np.save(save_path, np.array(ret))
                # np.save('lb_record_' + save_path, np.array(lb_record))
                if u < arguments.Config["bab"]["decision_thresh"]:
                    verified_status = "unsafe-bab"
                    pidx_all_verified = False
                    bab_attack_success = True
                    break
                elif l < arguments.Config["bab"]["decision_thresh"]:
                    pidx_all_verified = False
                    # break to run next sample save time if any label is not verified.
                    break
            except KeyboardInterrupt:
                print('time:', imag_idx, time.time()-start_inner, "\n",)
                print(ret)
                pidx_all_verified = False
                break

        example_time.append(time.time() - example_start_time)
        if not pidx_all_verified:
            verified_acc -= 1
            if bab_attack_success:
                attack_success.append(imag_idx)
                print(f'Result: image {imag_idx} attack success (with branch and bound)!')
            else:
                verified_failed.append(imag_idx)
                print(f'Result: image {imag_idx} verification failure (with branch and bound).')
        else:
            verified_success_list.append(imag_idx)
            print(f'Result: image {imag_idx} verification success (with branch and bound)!')
        # Make sure ALL tensors used in this loop are deleted here.
        del init_global_lb, saved_bounds, saved_slopes
        print(f'Wall time: {example_time[-1]}')

    # some results analysis
    np.set_printoptions(suppress=True)
    ret = np.array(ret)

    print(f'\nnumber of correctly classified examples: {nat_acc}')
    print(f'incorrectly classified idx (total {len(skipped_examples)}):', skipped_examples)
    print(f'attack success idx (total {len(attack_success)}):', attack_success)
    if len(attack_success) > 0:
        print('attack_success rate:', len(attack_success)/len(bnb_ids))
        np.save('Attack-success_{}_{}_start{}_end{}.npy'.
                format(arguments.Config['model']['name'], arguments.Config["data"]["dataset"], arguments.Config["data"]["start"], arguments.Config["data"]["end"]), np.array(attack_success))
    print(f'verification success idx (total {len(verified_success_list)}):', verified_success_list)
    print(f'verification failure idx (total {len(verified_failed)}):', verified_failed)
    if arguments.Config["general"]["complete_verifier"] == "mip":
        print("##### Complete MIP solver summary #####")
        print(f"mip verified safe idx: {mip_safe}")
        print(f"mip unsafe idx: {mip_unsafe}")
        print(f"mip unknown idx: {mip_unknown}")
        print(f"mip verified safe rate {len(mip_safe)/len(bnb_ids)}, "
                f"unsafe rate {len(mip_unsafe)/len(bnb_ids)}, "
                f"unknown rate {len(mip_unknown)/len(bnb_ids)}, "
                f"total {len(bnb_ids)}")

    print("final verified acc: {}%[{}]".format(verified_acc/len(bnb_ids)*100., len(bnb_ids)))
    np.save('Verified-acc_{}_{}_start{}_end{}_{}_branching_{}.npy'.
                    format(arguments.Config['model']['name'], arguments.Config["data"]["dataset"], arguments.Config["data"]["start"], arguments.Config["data"]["end"], verified_acc, arguments.Config["bab"]["branching"]["method"]), np.array(verified_failed))

    total_verification = len(verified_success_list) + len(verified_failed)
    print(f"verifier is called on {total_verification} examples.")
    print("total verified:", verified_acc)
    if ret.size > 0:
        # print("mean time [total:{}]: {}".format(len(bnb_ids), ret[:, 3].sum()/float(len(bnb_ids))))
        print("mean time [cnt:{}] (excluding attack success): {}".format(total_verification, ret[:, 3][ret[:, 5] != -3].sum()/float(total_verification if total_verification != 0 else "nan")))
        if len(attack_success) > 0:
            print("mean time [cnt:{}] (including attack success): {}".format(total_verification + len(attack_success), ret[:, 3].sum() / float(total_verification + len(attack_success))))


if __name__ == "__main__":
    config_args()
    main()
