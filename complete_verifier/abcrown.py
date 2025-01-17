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
"""α,β-CROWN (alpha-beta-CROWN) verifier main interface."""

import copy
import socket
import random
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any
from auto_LiRPA.operators.convolution import BoundConv
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPANet
from lp_mip_solver import mip, check_enable_refinement
from attack import attack
from attack.attack_pgd import check_and_save_cex
from utils import Logger, print_model
from specifications import (trim_batch, batch_vnnlib, sort_targets,
                            prune_by_idx, add_rhs_offset)
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning, Customized  # pylint: disable=unused-import
from bab import general_bab
from input_split.batch_branch_and_bound import input_bab_parallel
from read_vnnlib import read_vnnlib
from cuts.cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching
from lp_test import compare_optimized_bounds_against_lp_bounds


class ABCROWN:
    def __init__(self, args=None, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, list):
                args.append(f'--{k}')
                args.extend(list(map(str, v)))
            elif isinstance(v, bool):
                if v:
                    args.append(f'--{k}')
                else:
                    args.append(f'--no_{k}')
            else:
                args.append(f'--{k}={v}')
        arguments.Config.parse_config(args)

    def incomplete_verifier(
        self,
        model_ori,
        data,
        data_ub=None,
        data_lb=None,
        vnnlib=None,
        interm_bounds=None
    ):
        # Generally, c should be constructed from vnnlib
        assert len(vnnlib) == 1, 'incomplete_verifier only support single x spec'
        input_x, specs = vnnlib[0]
        c_transposed = False
        tighten_input_bounds = (
            arguments.Config['solver']['invprop']['tighten_input_bounds']
        )
        apply_output_constraints_to = (
            arguments.Config['solver']['invprop']['apply_output_constraints_to']
        )

        if len(specs) > 1:
            # single OR with many clauses (e.g., robustness verification)
            assert all([len(_[0]) == 1 for _ in specs]), \
                'for each property in OR, only one clause supported so far'
            c = torch.concat([
                item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
                for item in specs], dim=0).unsqueeze(1).to(data)  # c shape: (batch, 1, num_outputs)
            do_transpose = not arguments.Config['solver']['optimize_disjuncts_separately']
            rhs = torch.tensor(np.array([item[1] for item in specs])).to(data)  # (batch, 1)
            if do_transpose and c.shape[0] != 1 and data.shape[0] == 1:
                # transpose c to shape (1,batch,num_outputs) to share intermediate bounds
                assert len(apply_output_constraints_to) == 0, (
                    'To apply output constraints, set --optimize_disjuncts_separately'
                )
                c = c.transpose(0, 1)
                rhs = rhs.t()  # (1, batch)
                c_transposed = True
            else:
                if arguments.Config['solver']['prune_after_crown']:
                    raise NotImplementedError(
                        'To use optimize_disjuncts_separately=True, do not set '
                        'prune_after_crown=True'
                    )
            stop_func = stop_criterion_all(rhs)

        else:
            # single AND with many clauses (e.g., Yolo).
            # shape: (batch=1, num_clauses in AND, num_outputs)
            c = torch.tensor(specs[0][0]).unsqueeze(0).to(data)
            # shape: (1, num_clauses in AND)
            rhs = torch.tensor(specs[0][1], dtype=data.dtype, device=data.device).unsqueeze(0)
            stop_func = stop_criterion_batch_any(rhs)

        model = LiRPANet(model_ori, in_size=data.shape, c=c)

        bound_prop_method = arguments.Config['solver']['bound_prop_method']
        if len(apply_output_constraints_to) > 0:
            assert bound_prop_method == 'alpha-crown'
            model.net.constraints = torch.tensor([x[0] for x in specs])
            assert model.net.constraints.ndim == 3
            assert rhs.ndim ==  2
            if len(specs) == 1:
                assert rhs.size(0) == 1
                model.net.thresholds = rhs.squeeze(0)
            else:
                assert rhs.size(1) == 1
                model.net.thresholds = rhs.squeeze(1)

            # We need to use matrix mode for the layer that should utilize output constraints
            for node in model.net.nodes():
                if node.are_output_constraints_activated_for_layer(apply_output_constraints_to):
                    if isinstance(node, BoundConv) and node.mode == 'patches':
                        node.mode = 'matrix'

        if isinstance(input_x, dict):
            # Traditional Lp norm case. Still passed in as an vnnlib variable, but it is passed
            # in as a dictionary.
            ptb = PerturbationLpNorm(
                norm=input_x['norm'],
                eps=input_x['eps'], eps_min=input_x.get('eps_min', 0),
                x_L=data_lb, x_U=data_ub)
        else:
            norm = arguments.Config['specification']['norm']
            # Perturbation value for non-Linf perturbations, None for all other cases.
            ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb).to(data.device)
        output = model.net(x)
        print_model(model.net)
        print('Original output:', output)

        # save output
        if arguments.Config['general']['save_output']:
            arguments.Globals['out']['pred'] = output.cpu()

        domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
        # one of them is sufficient.
        global_lb, ret = model.build(
            domain, x, stop_criterion_func=stop_func, decision_thresh=rhs, vnnlib_ori=vnnlib,
            interm_bounds=interm_bounds)

        if arguments.Config['general']['return_optimized_model']:
            return model

        if c_transposed:
            # transpose back to get ready for general verified condition check and final outputs
            global_lb = global_lb.t()
            rhs = rhs.t()

        if torch.any((global_lb - rhs) > 0, dim=-1).all():
            # Any spec in AND verified means verified. Also check all() at batch dim.
            print('verified with init bound!')
            return 'safe-incomplete', {}, model

        if arguments.Config['attack']['pgd_order'] == 'middle':
            if ret['attack_images'] is not None:
                return 'unsafe-pgd', {}, model

        # Save the alpha variables during optimization. Here the batch size is 1.
        saved_alphas = defaultdict(dict)
        for m in model.net.optimizable_activations:
            for spec_name, alpha in m.alpha.items():
                # each alpha size is (2, spec, 1, *shape); batch size is 1.
                saved_alphas[m.name][spec_name] = alpha.detach().clone()

        # FIXME there may be some duplicate with saved_alphas
        if bound_prop_method == 'alpha-crown':
            ret['activation_opt_params'] = {
                node.name: node.dump_optimized_params()
                for node in model.net.optimizable_activations
            }

        if c_transposed:
            ret['lower_bounds'][model.final_name] = ret['lower_bounds'][model.final_name].t()
            ret['upper_bounds'][model.final_name] = ret['upper_bounds'][model.final_name].t()
            if ret['lA'] is not None:
                ret['lA'] = {k: v.transpose(0, 1) for k, v in ret['lA'].items()}

        ret.update({'model': model, 'global_lb': global_lb, 'alpha': saved_alphas})

        if tighten_input_bounds:
            perturbed_root = None
            for root in model.net.roots():
                if hasattr(root, 'perturbation') and root.perturbation is not None:
                    assert perturbed_root is None, (
                        'BaB based on tightened bounds currently supports only one input layer'
                    )
                    perturbed_root = root
            assert perturbed_root is not None
            ret['tightened_input_bounds'] = [
                perturbed_root.perturbation.x_L.detach(),
                perturbed_root.perturbation.x_U.detach(),
            ]
        return 'unknown', ret, model

    def bab(self, data_lb, data_ub, c, rhs,
            data=None, targets=None, vnnlib=None, timeout=None,
            time_stamp=0, data_dict=None, lower_bounds=None, upper_bounds=None,
            reference_alphas=None, attack_images=None, cplex_processes=None,
            activation_opt_params=None, reference_lA=None,
            model_incomplete=None, refined_betas=None,
            create_model=True, model=None, return_domains=False,
            max_iterations=None, property_idx=None, vnnlib_meta=None,
            orig_lirpa_model=None):
        # This will use the refined bounds if the complete verifier is 'bab-refine'.
        # FIXME do not repeatedly create LiRPANet which creates a new BoundedModule each time.

        # Save these arguments in case that they need to retrieved the next time
        # this function is called.
        if vnnlib_meta is None:
            vnnlib_meta = {
                'property_idx': 0, 'vnnlib_id': 0, 'benchmark_name': None
            }
        self.data_lb, self.data_ub, self.c, self.rhs = data_lb, data_ub, c, rhs
        self.data, self.targets, self.vnnlib = data, targets, vnnlib

        # if using input split, transpose C if there are multiple specs with shared input,
        # to improve efficiency when calling the incomplete verifier later
        if arguments.Config['bab']['branching']['input_split']['enable']:
            c_transposed = False
            if (data_lb.shape[0] == 1 and data_ub.shape[0] == 1 and c is not None
                    and c.shape[0] > 1 and c.shape[1] == 1):
                # multiple c instances (multiple vnnlibs) since c.shape[0] > 1,
                # but they share the same input (since data.shape[0] == 1）and
                # only single spec in each instance (c.shape[1] == 1)
                c = c.transpose(0, 1)
                rhs = rhs.transpose(0, 1)
                c_transposed = True

        if create_model:
            self.model = LiRPANet(
                model, c=c, cplex_processes=cplex_processes,
                in_size=(data_lb.shape if len(targets) <= 1
                        else [len(targets)] + list(data_lb.shape[1:])),
                mip_building_proc=(orig_lirpa_model.mip_building_proc
                                   if orig_lirpa_model is not None else None)
            )
            if not model_incomplete:
                print_model(self.model.net)

        data_lb, data_ub = data_lb.to(self.model.device), data_ub.to(self.model.device)
        norm = arguments.Config['specification']['norm']
        if data_dict is not None:
            assert isinstance(data_dict['eps'], float)
            ptb = PerturbationLpNorm(
                norm=norm, eps=data_dict['eps'],
                eps_min=data_dict.get('eps_min', 0), x_L=data_lb, x_U=data_ub)
        else:
            ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)

        if data is not None:
            data = data.to(self.model.device)
            x = BoundedTensor(data, ptb).to(data_lb.device)
            output = self.model.net(x).flatten()
            print('Model prediction is:', output)

            # save output:
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['pred'] = output.cpu()

            if arguments.Config['attack']['check_clean'] and not arguments.Config['debug'][
                'sanity_check']:
                clean_rhs = c.matmul(output)
                print(f'Clean RHS: {clean_rhs}')
                if (clean_rhs < rhs).any():
                    # add and set output batch_size dimension to 1
                    verified_status, _ = check_and_save_cex(
                        x.detach(), output.unsqueeze(0), vnnlib,
                        arguments.Config['attack']['cex_path'], 'unsafe')
                    return -torch.inf, None, verified_status
        else:
            x = BoundedTensor(data_lb, ptb).to(data_lb.device)

        self.domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
        if arguments.Config['bab']['branching']['input_split']['enable']:
            result = input_bab_parallel(
                self.model, self.domain, x, rhs=rhs,
                timeout=timeout, max_iterations=max_iterations,
                vnnlib=vnnlib, c_transposed=c_transposed,
                return_domains=return_domains, vnnlib_meta=vnnlib_meta
            )
            if return_domains:
                return result
        else:
            assert not return_domains, 'return_domains is only for input split for now'
            result = general_bab(
                self.model, self.domain, x,
                refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
                activation_opt_params=activation_opt_params, reference_lA=reference_lA,
                reference_alphas=reference_alphas, attack_images=attack_images,
                timeout=timeout, max_iterations=max_iterations,
                refined_betas=refined_betas, rhs=rhs, property_idx=property_idx,
                model_incomplete=model_incomplete, time_stamp=time_stamp)

        min_lb = result[0]
        if min_lb is None:
            min_lb = -torch.inf
        elif isinstance(min_lb, torch.Tensor):
            min_lb = min_lb.item()

        # If a counterexample is found in any node split LP, check and save it
        if result[2] == 'unsafe_bab':
            stats = result[3]
            adv_example = stats.counterexample.detach().to(self.model.device)
            verified_status, _ = check_and_save_cex(
                adv_example, self.model.net(adv_example), vnnlib,
                arguments.Config['attack']['cex_path'], 'unsafe')
            result = (min_lb, result[1], verified_status)
        else:
            result = (min_lb, *result[1:3])
        return result

    def complete_verifier(
            self, model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
            index, timeout_threshold, bab_ret=None, cplex_processes=None,
            attack_images=None, attack_margins=None, results=None, vnnlib_id=None,
            benchmark_name=None, orig_lirpa_model=None
    ):
        start_time = time.time()

        enable_incomplete = arguments.Config['general']['enable_incomplete_verification']
        init_global_lb = results.get('global_lb', None)
        lower_bounds = results.get('lower_bounds', None)
        upper_bounds = results.get('upper_bounds', None)
        reference_alphas = results.get('alpha', None)
        lA = results.get('lA', None)
        cplex_cuts = (arguments.Config['bab']['cut']['enabled']
                    and arguments.Config['bab']['cut']['cplex_cuts'])
        bab_attack_enabled = arguments.Config['bab']['attack']['enabled']

        reference_alphas_cp = None
        if enable_incomplete:
            final_name = model_incomplete.final_name
            init_global_ub = upper_bounds[final_name]
            print('lA shape:', [lAitem.shape for lAitem in lA.values()])
            (batched_vnnlib, init_global_lb, init_global_ub,
            lA, attack_images) = sort_targets(
                batched_vnnlib, init_global_lb, init_global_ub,
                attack_images, attack_margins, results, model_incomplete)
            if reference_alphas is not None:
                reference_alphas_cp = copy.deepcopy(reference_alphas)

        solved_c_rows = []

        time_stamp = 0
        rhs_offsets = arguments.Config['specification']['rhs_offset']
        for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
            # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
            print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
            timeout = timeout_threshold - (time.time() - start_time)
            print(f'Remaining timeout: {timeout}')
            start_time_bab = time.time()
            print(f'Verifying property {property_idx} with {len(properties[0])} instances.')
            if arguments.Config['debug']['sanity_check'] == 'Full':
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets[property_idx]
                timeout = timeout_threshold
                sanity_check_results = []
            else:
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets
            if (arguments.Config['bab']['cut']['enabled'] and
                arguments.Config['bab']['initial_max_domains'] == 1
                and not arguments.Config['debug']['sanity_check']):
                if init_global_lb[property_idx][0] > rhs_offset:
                    print('Verified by alpha-CROWN bound!')
                    continue

            if isinstance(properties[0][0], dict):
                def _get_item(properties, key):
                    return torch.concat([
                        item[key].unsqueeze(0) for item in properties[0]], dim=0)
                x = _get_item(properties, 'X')
                data_min = _get_item(properties, 'data_min')
                data_max = _get_item(properties, 'data_max')
                # A dict to store extra variables related to the data and specifications
                for item in properties[0]:
                    assert item['eps'] == properties[0][0]['eps']
                data_dict = {
                    'eps': properties[0][0]['eps'],
                    'eps_min': properties[0][0].get('eps_min', 0),
                }
            else:
                x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
                data_dict = None
            if 'tightened_input_bounds' in results:
                assert (
                    results['tightened_input_bounds'][0][property_idx:property_idx+1].shape
                    == data_min.shape
                )
                data_min = results['tightened_input_bounds'][0][property_idx:property_idx+1]
                data_max = results['tightened_input_bounds'][1][property_idx:property_idx+1]

            target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)
            assert len(target_label_arrays) == 1
            c, rhs, pidx = target_label_arrays[0]

            if bab_attack_enabled:
                if arguments.Config['bab']['initial_max_domains'] != 1:
                    raise ValueError(
                        'To run Bab-attack, please set initial_max_domains to 1. '
                        f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')
                # Attack images has shape (batch, restarts, specs, c, h, w).
                # The specs dimension should already be sorted.
                # Reshape it to (restarts, c, h, w) for this specification.
                this_spec_attack_images = attack_images[:, :, property_idx].view(
                    attack_images.size(1), *attack_images.shape[3:])
            else:
                this_spec_attack_images = None

            # FIXME Clean up.
            # Shape and type of rhs is very confusing
            rhs = torch.tensor(rhs, device=arguments.Config['general']['device'],
                               dtype=torch.get_default_dtype())
            if enable_incomplete and len(init_global_lb) > 1:
                # no need to trim_batch if batch = 1
                ret_trim = trim_batch(
                    model_incomplete, init_global_lb, init_global_ub,
                    reference_alphas_cp, lower_bounds, upper_bounds,
                    reference_alphas, lA, property_idx, c, rhs)
                lA_trim, rhs = ret_trim['lA'], ret_trim['rhs']
                trimmed_lower_bounds = ret_trim['lower_bounds']
                trimmed_upper_bounds = ret_trim['upper_bounds']
            else:
                lA_trim = lA.copy() if lA is not None else lA
                trimmed_lower_bounds = lower_bounds
                trimmed_upper_bounds = upper_bounds

            print(f'##### Instance {index} first 10 spec matrices: ')
            print(f'{c[:10]}\nthresholds: {rhs.flatten()[:10]} ######')

            torch.cuda.empty_cache()
            gc.collect()
            c = c.to(rhs)  # both device and dtype

            # compress the first dim of data_min, data_max based on duplication check
            if data_min.shape[0] > 1:
                l1_err_data_min = torch.norm((data_min[1:] - data_min[0:1]).view(-1), p=1)
                l1_err_data_max = torch.norm((data_max[1:] - data_max[0:1]).view(-1), p=1)
                if l1_err_data_min + l1_err_data_max < 1e-8:
                    # almost same x so we can use the first x
                    x, data_min, data_max = x[0:1], data_min[0:1], data_max[0:1]

            # Complete verification (BaB, BaB with refine, or MIP).
            time_stamp += 1
            input_split = arguments.Config['bab']['branching']['input_split']['enable']
            init_failure_idx = np.array([])
            if enable_incomplete and not input_split:
                if len(init_global_lb) > 1:  # if batch == 1, there is no need to filter here.
                    # Reuse results from incomplete results, or from refined MIPs.
                    # skip the prop that already verified
                    rlb = trimmed_lower_bounds[final_name]
                    # The following flatten is dangerous, each clause in OR only
                    # has one output bound.
                    assert len(rlb.shape) == len(rhs.shape) == 2
                    assert rlb.shape[1] == rhs.shape[1] == 1
                    init_verified_cond = rlb.flatten() > rhs.flatten()
                    init_verified_idx = torch.where(init_verified_cond)[0]
                    if len(init_verified_idx) > 0:
                        print('Initial alpha-CROWN verified for spec index '
                                f'{init_verified_idx} with bound '
                                f'{rlb[init_verified_idx].squeeze()}.')
                        l = init_global_lb[init_verified_idx].tolist()
                        bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                    init_failure_idx = torch.where(~init_verified_cond)[0]
                    if len(init_failure_idx) == 0:
                        # This batch of x verified by init opt crown
                        continue
                    print(f'Remaining spec index {init_failure_idx} with '
                            f'bounds {rlb[init_failure_idx]} need to verify.')

                    (reference_alphas, lA_trim, x, data_min, data_max,
                    trimmed_lower_bounds, trimmed_upper_bounds, c) = prune_by_idx(
                        reference_alphas, init_verified_cond, final_name, lA_trim, x,
                        data_min, data_max, lA is not None,
                        trimmed_lower_bounds, trimmed_upper_bounds, c)

                l, nodes, ret = self.bab(
                    data=x, targets=init_failure_idx, time_stamp=time_stamp,
                    data_ub=data_max, data_lb=data_min, data_dict=data_dict,
                    lower_bounds=trimmed_lower_bounds, upper_bounds=trimmed_upper_bounds,
                    c=c, reference_alphas=reference_alphas, cplex_processes=cplex_processes,
                    activation_opt_params=results.get('activation_opt_params', None),
                    refined_betas=results.get('refined_betas', None), rhs=rhs[0:1],
                    reference_lA=lA_trim, attack_images=this_spec_attack_images,
                    model_incomplete=model_incomplete, timeout=timeout, vnnlib=vnnlib,
                    model=model_ori, property_idx=property_idx,
                    vnnlib_meta={
                        'property_idx': property_idx,
                        'vnnlib_id': vnnlib_id,
                        'benchmark_name': benchmark_name
                    },
                    orig_lirpa_model=orig_lirpa_model,
                )
                bab_ret.append([index, float(l), nodes,
                                time.time() - start_time_bab,
                                init_failure_idx.tolist()])
            else:
                assert arguments.Config['general']['complete_verifier'] == 'bab'
                assert not arguments.Config['bab']['attack']['enabled'], (
                    'BaB-attack must be used with incomplete verifier.')
                # input split also goes here directly
                l, nodes, ret = self.bab(
                    data=x, targets=pidx, time_stamp=time_stamp,
                    data_ub=data_max, data_lb=data_min, c=c, data_dict=data_dict,
                    cplex_processes=cplex_processes,
                    rhs=rhs, timeout=timeout, attack_images=this_spec_attack_images,
                    vnnlib=vnnlib, model=model_ori, vnnlib_meta={
                        'property_idx': property_idx,
                        'vnnlib_id': vnnlib_id,
                        'benchmark_name': benchmark_name
                    },
                    orig_lirpa_model=orig_lirpa_model,
                )
                bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])

            # terminate the corresponding cut inquiry process if exists
            if cplex_cuts:
                solved_c_rows.append(c)
                terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

            timeout = timeout_threshold - (time.time() - start_time)
            if ret == 'unsafe':
                return 'unsafe-bab'
            elif ret == 'unknown' or timeout < 0:
                if arguments.Config['debug']['sanity_check'] == 'Full':
                    sanity_check_results.append(ret)
                    continue
                return 'unknown'
            elif ret != 'safe':
                raise ValueError(f'Unknown return value of bab: {ret}')
        else:
            if arguments.Config['debug']['sanity_check'] == 'Full':
                if len(sanity_check_results) == len(properties[0]):
                    print('Sanity check results:', sanity_check_results)
                    if all(result == 'unknown' for result in sanity_check_results):
                        return 'unknown'
                    else:
                        assert False, 'Sanity check failed. Something is wrong.'
            return 'safe'

    def attack(self, model_ori, x, vnnlib, verified_status, verified_success):
        if arguments.Config['model']['with_jacobian']:
            print('Using BoundedModule for attack for this model with JacobianOP')
            model = LiRPANet(model_ori, in_size=x.shape).net
        else:
            model = model_ori
        return attack(model, x, vnnlib, verified_status, verified_success)

    def main(self, interm_bounds=None):
        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
            # Always disable TF32 (precision is too low for verification).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if arguments.Config['general']['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        if arguments.Config['general']['double_fp']:
            torch.set_default_dtype(torch.float64)
        if arguments.Config['general']['precompile_jit']:
            precompile_jit_kernels()

        bab_args = arguments.Config['bab']
        debug_args = arguments.Config['debug']
        timeout_threshold = bab_args['timeout']
        interm_transfer_init = bab_args['interm_transfer']
        cut_usage_init = bab_args['cut']['enabled']
        select_instance = arguments.Config['data']['select_instance']
        complete_verifier = arguments.Config['general']['complete_verifier']
        p_a_crown_init = arguments.Config['solver']['prune_after_crown']
        if bab_args['backing_up_max_domain'] is None:
            arguments.Config['bab']['backing_up_max_domain'] = bab_args['initial_max_domains']
        (run_mode, save_path, file_root, example_idx_list, model_ori,
        vnnlib_all, shape) = parse_run_mode()
        self.logger = Logger(run_mode, save_path, timeout_threshold)

        if arguments.Config['general']['return_optimized_model']:
            assert len(example_idx_list) == 1, (
                'To return the optimized model, only one instance can be processed'
            )
        if arguments.Config['debug']['sanity_check']:
            print('Warning: Sanity Check Debugging is enabled.',
            'The PGD upper bound will be calculated and used as the RHS offset.')
            arguments.Config['attack']['pgd_order'] = 'before'

        for new_idx, csv_item in enumerate(example_idx_list):
            arguments.Globals['example_idx'] = new_idx
            vnnlib_id = new_idx + arguments.Config['data']['start']
            # Select some instances to verify
            if select_instance and not vnnlib_id in select_instance:
                continue
            self.logger.record_start_time()

            print(f'\n {"%"*35} idx: {new_idx}, vnnlib ID: {vnnlib_id} {"%"*35}')
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['idx'] = new_idx   # saved for test

            onnx_path = None
            if run_mode != 'customized_data':
                if len(csv_item) == 3:
                    # model, vnnlib, timeout
                    model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                        file_root, csv_item)
                    arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                    arguments.Config['specification']['vnnlib_path'] = os.path.join(
                        file_root, csv_item[1])
                else:
                    # Each line contains only 1 item, which is the vnnlib spec.
                    vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                    assert arguments.Config['model']['input_shape'] is not None, (
                        'vnnlib does not have shape information, '
                        'please specify by --input_shape')
                    shape = arguments.Config['model']['input_shape']
            else:
                vnnlib = vnnlib_all[new_idx]  # vnnlib_all is a list of all standard vnnlib

            # Skip running the actual verifier during preparation.
            if arguments.Config['general']['prepare_only']:
                continue

            # FIXME Don't write bab_args['timeout'] above.
            # Then these updates can be moved to arguments.update_arguments()
            bab_args['timeout'] = float(bab_args['timeout'])
            if bab_args['timeout_scale'] != 1:
                new_timeout = bab_args['timeout'] * bab_args['timeout_scale']
                print(f'Scaling timeout: {bab_args["timeout"]} -> {new_timeout}')
                bab_args['timeout'] = new_timeout
            if bab_args['override_timeout'] is not None:
                new_timeout = bab_args['override_timeout']
                print(f'Overriding timeout: {new_timeout}')
                bab_args['timeout'] = new_timeout
            timeout_threshold = bab_args['timeout']
            self.logger.update_timeout(timeout_threshold)

            if arguments.Config['general']['complete_verifier'].startswith('Customized'):
                res = eval(  # pylint: disable=eval-used
                    arguments.Config['general']['complete_verifier']
                )(model_ori, vnnlib, os.path.join(file_root, onnx_path))
                self.logger.summarize_results(res, new_idx)
                continue

            model_ori.eval()
            vnnlib_shape = shape
            attack_image = None

            # FIXME attack and initial_incomplete_verification only works for
            # assert len(vnnlib) == 1
            if isinstance(vnnlib[0][0], dict):
                x = vnnlib[0][0]['X'].reshape(vnnlib_shape)
                data_min = vnnlib[0][0]['data_min'].reshape(vnnlib_shape)
                data_max = vnnlib[0][0]['data_max'].reshape(vnnlib_shape)
            else:
                x_range = torch.tensor(vnnlib[0][0])
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
            adhoc_tuning(data_min, data_max, model_ori)

            rhs_offset_init = arguments.Config['specification']['rhs_offset']
            if rhs_offset_init is not None and not arguments.Config['debug']['sanity_check']:
                vnnlib = add_rhs_offset(vnnlib, rhs_offset_init)

            model_ori = model_ori.to(device)
            x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
            verified_status, verified_success = 'unknown', False

            if arguments.Config['debug']['sanity_check']:
                print('Warning: Sanity Check Debugging is enabled.',
                      'The PGD upper bound will be calculated and used as the RHS offset.')
                arguments.Config['attack']['pgd_order'] = 'before'

            if arguments.Config['attack']['pgd_order'] == 'before':
                (verified_status, verified_success, attack_image,
                 attack_margins, all_adv_candidates) = self.attack(
                    model_ori, x, vnnlib, verified_status, verified_success)
                if arguments.Config['debug']['sanity_check']:
                    if arguments.Config['debug']['sanity_check'] in ['Full', 'Full+Graph']:
                        rhs_offset = attack_margins.flatten().cpu().numpy()
                    else:
                        rhs_offset = attack_margins.min().item()
                    # changes the verification status back to unknown and the pgd_order is now skip
                    # so that now unsafe instances will also time out
                    print('Warning: Changing the RHS offset to the worst PGD '
                          'upper bound. If "rhs_offset" was set in the config/commandline, '
                          'it will be ignored.')
                    print(f'Using PGD upper bound:\n{rhs_offset}.')
                    print(f'Shape of attack_margins: {attack_margins.shape}')
                    print(f'Verified success: {verified_success} -> False')
                    print(f'Verified success: {verified_status} -> \'unknown\'')
                    vnnlib = add_rhs_offset(vnnlib, rhs_offset)
                    arguments.Config['attack']['pgd_order'] = 'skip'
                    verified_status, verified_success = 'unknown', False
            else:
                attack_margins = all_adv_candidates = None

            model_incomplete = cplex_processes = None
            ret = {}

            if arguments.Config['debug']['test_optimized_bounds']:
                compare_optimized_bounds_against_lp_bounds(
                    model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib
                )

            # Incomplete verification is enabled by default. The intermediate lower
            # and upper bounds will be reused in bab and mip.
            orig_lirpa_model = None
            if (not verified_success
                    and arguments.Config['general']['enable_incomplete_verification']):
                incomplete_verification_output = self.incomplete_verifier(
                    model_ori,
                    x,
                    data_ub=data_max,
                    data_lb=data_min,
                    vnnlib=vnnlib,
                    interm_bounds=interm_bounds
                )
                if arguments.Config['general']['return_optimized_model']:
                    return incomplete_verification_output
                verified_status, ret, orig_lirpa_model = incomplete_verification_output

                verified_success = verified_status != 'unknown'
                model_incomplete = ret.get('model', None)

                if arguments.Config['general']['complete_verifier'] == 'auto':
                    complete_verifier = check_enable_refinement(ret)
                    if complete_verifier in ['bab-refine', 'mip']:
                        arguments.Config['bab']['interm_transfer'] = True
                        arguments.Config['bab']['cut']['enabled'] = False
                        arguments.Config['solver']['prune_after_crown'] = False
                    else:
                        arguments.Config['bab']['interm_transfer'] = interm_transfer_init
                        arguments.Config['bab']['cut']['enabled'] = cut_usage_init
                        arguments.Config['solver']['prune_after_crown'] = p_a_crown_init

            if not verified_success and arguments.Config['attack']['pgd_order'] == 'after':
                (verified_status, verified_success, attack_image,
                 attack_margins, all_adv_candidates) = self.attack(
                    model_ori, x, vnnlib, verified_status, verified_success)
            # MIP or MIP refined bounds.
            if not verified_success and complete_verifier in ['bab-refine', 'mip']:
                # rhs = ? NEED TO SAVE TO LIRPA_MODULE
                mip_skip_unsafe = arguments.Config['solver']['mip']['skip_unsafe']
                verified_status, ret_mip = mip(
                    model_incomplete, ret, mip_skip_unsafe=mip_skip_unsafe, vnnlib=vnnlib,
                    pgd_attack_example=[attack_image, attack_margins], verifier=complete_verifier)
                verified_success = verified_status != 'unknown'
                ret.update(ret_mip)

            # extract the process pool for cut inquiry
            mip_building_proc = None
            if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
                assert arguments.Config['bab']['initial_max_domains'] == 1
                # use nullity of model_incomplete as an indicator of whether cut
                # processes are launched
                if model_incomplete is not None:
                    cplex_processes = model_incomplete.processes
                    print('Cut inquiry processes are launched.')
                    mip_building_proc = model_incomplete.mip_building_proc

            # BaB bounds. (not do bab if unknown by mip solver for now)
            if (not verified_success
                    and complete_verifier != 'skip'
                    and verified_status != 'unknown-mip'):
                batched_vnnlib = batch_vnnlib(vnnlib)  # [x, [(c, rhs, y, pidx)]] in batch-wise
                benchmark_name = (file_root.split('/')[-1]
                                  if debug_args['sanity_check'] is not None else None)
                verified_status = self.complete_verifier(
                    model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
                    new_idx, bab_ret=self.logger.bab_ret, cplex_processes=cplex_processes,
                    timeout_threshold=timeout_threshold - (time.time() - self.logger.start_time),
                    attack_images=all_adv_candidates,
                    attack_margins=attack_margins, results=ret, vnnlib_id=vnnlib_id,
                    benchmark_name=benchmark_name, orig_lirpa_model=orig_lirpa_model
                )

            if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                    and model_incomplete is not None):
                terminate_mip_processes(mip_building_proc, cplex_processes)
                del cplex_processes
            del ret

            if arguments.Config['debug']['sanity_check']:
                arguments.Config['specification']['rhs_offset'] = rhs_offset_init
            # Summarize results.
            self.logger.summarize_results(verified_status, new_idx)

        self.logger.finish()
        return self.logger.verification_summary


if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:])
    abcrown.main()
