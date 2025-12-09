#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""complete verifier main interface."""

import time
import gc
import torch

import arguments
from beta_CROWN_solver import LiRPANet
from attack import check_and_save_cex
from utils import Logger, print_model, take_batch
from specifications import BatchedSpecs
from bab import general_bab
from input_split.batch_branch_and_bound import input_bab_parallel
from cuts.cut_utils import terminate_mip_processes_by_c_matching, clean_net_mps_process

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from abcrown import ABCROWN

def bab(self: 'ABCROWN',
        reference_dict=None,
        model=None,
        timeout=None,
        return_domains=False,
        max_iterations=None,
        index=None
        ):

    if arguments.Config['general']['store_all_specs_on_cpu']:
        device = "cpu"
    else:
        device = arguments.Config['general']['device']
    all_specs: BatchedSpecs = self.vnnlib_handler.all_specs
    x, c, rhs, or_spec_size, _, _ = all_specs.get(device)
    vnnlib = self.vnnlib_handler.vnnlib

    enable_cuts = arguments.Config['bab']['cut']['enabled']
    cplex_cuts = (enable_cuts and arguments.Config['bab']['cut']['cplex_cuts'])
    all_node_split_LP = arguments.Config['solver']['beta-crown']['all_node_split_LP']
    bab_attack_enabled = arguments.Config['bab']['attack']['enabled']

    # initialize the model.
    # Since we always enable incomplete verification by default,
    # we will reuse the model.
    if arguments.Config['general']['enable_incomplete_verification']:
        assert reference_dict
        lirpa_model = reference_dict.get('model', None)
        assert lirpa_model is not None
    else:
        # FIXME: This branch should not be used by default and is only for backup.
        # Maybe it can be removed in the future.
        assert not cplex_cuts
        lirpa_model = LiRPANet(model, in_size=[1, *x.shape[1:]])
        print_model(lirpa_model.net)

    # save the output of the model from the last input
    # it is just for comparing with reference in tests
    output = lirpa_model.net(x[-1:].to(lirpa_model.device)).flatten()
    print('Model prediction is:', output)
    if arguments.Config['general']['save_output']:
        arguments.Globals['out']['pred'] = output.cpu()

    input_split = arguments.Config['bab']['branching']['input_split']['enable']
    if input_split:
        result = input_bab_parallel(
            lirpa_model, x, c, rhs, or_spec_size,
            reference_dict=reference_dict,
            timeout=timeout, max_iterations=max_iterations,
            vnnlib=vnnlib,
            return_domains=return_domains,
            index=index
        )
        if return_domains:
            return result
    else:
        assert not return_domains, 'return_domains is only for input split for now'
        total_num_or_spec = c.shape[0]
        if enable_cuts or all_node_split_LP or bab_attack_enabled:
            # these features only support batch size 1
            bab_batch_size = 1
            if cplex_cuts:
                solved_c_list = []
        else:
            # otherwise, we can use the whole batch
            bab_batch_size = total_num_or_spec
        num_batches = (total_num_or_spec + bab_batch_size - 1) // bab_batch_size

        result = [float("inf"), 0, "safe"]

        # this for loop will only execute once or total_num_or_spec times
        for batch_idx in range(num_batches):
            print(f'Activation BaB batch {batch_idx + 1}/{num_batches} ')
            batch_x, batch_c, batch_rhs, batch_reference_dict = prepare_for_act_bab(
                x, c, rhs, reference_dict, bab_batch_size, batch_idx)

            batch_result = general_bab(
                lirpa_model, batch_x, batch_c, batch_rhs,
                reference_dict=batch_reference_dict,
                timeout=timeout, max_iterations=max_iterations)

            if cplex_cuts:
                solved_c_list.append(batch_c)
                terminate_mip_processes_by_c_matching(lirpa_model.processes, solved_c_list)

            batch_result = _format_result_act_bab(batch_result, lirpa_model, vnnlib)

            # record the worst lb
            result[0] = min(result[0], batch_result[0])
            # account for the number of domains visited
            result[1] += batch_result[1]

            if batch_result[2] != 'safe':
                # can be 'unsafe_bab' or 'unknown'
                result[2] = batch_result[2]
                break
        if enable_cuts:
            clean_net_mps_process(lirpa_model)
    # return global lb, number of domains visited, and status
    return result

def complete_verifier(
        self: 'ABCROWN', model_ori,
        index, timeout_threshold, bab_ret=None,
        reference_dict=None
):
    start_time = time.time()

    torch.cuda.empty_cache()
    gc.collect()

    # FIXME: double check the implementation of enable_opt_interm_bounds
    # the current version does not imporve the result.
    assert not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']

    all_specs: BatchedSpecs = self.vnnlib_handler.all_specs

    print(f'###### Verifying property Instance {index} ######')
    all_specs.print_stats()
    timeout = timeout_threshold - (time.time() - start_time)
    print(f'Remaining timeout: {timeout}')
    start_time_bab = time.time()

    # Complete verification (BaB, BaB with refine, or MIP).
    l, num_domains_visited, ret = self.bab(
        reference_dict=reference_dict,
        model=model_ori,
        timeout=timeout,
        index=index,
    )
    bab_ret.append([index, l, num_domains_visited, time.time() - start_time_bab])

    timeout = timeout_threshold - (time.time() - start_time)

    if ret == 'unsafe':
        return 'unsafe-bab'
    elif ret == 'unknown' or timeout < 0:
        return 'unknown'
    elif ret != 'safe':
        raise ValueError(f'Unknown return value of bab: {ret}')
    return 'safe'

def prepare_for_act_bab(x, c, rhs, reference_dict, batch_size, batch_idx):
    batch_x = take_batch(x, batch_size, batch_idx)
    batch_c = take_batch(c, batch_size, batch_idx)
    batch_rhs = take_batch(rhs, batch_size, batch_idx)
    batch_reference_dict = {"global_lb": None, "global_ub": None,
                            "lower_bounds": None, "upper_bounds": None,
                            "lA": None, "alphas": None, "mask": None,
                            "refined_betas": None, "history": None,
                            "attack_examples": None, "attack_margins": None}

    if reference_dict.get('global_lb', None) is not None:
        batch_reference_dict['global_lb'] = take_batch(reference_dict['global_lb'], batch_size, batch_idx)
    if reference_dict.get('global_ub', None) is not None:
        batch_reference_dict['global_ub'] = take_batch(reference_dict['global_ub'], batch_size, batch_idx)
    if reference_dict.get('lower_bounds', None) is not None:
        batch_reference_dict['lower_bounds'] = {k: take_batch(v, batch_size, batch_idx)
                                                for k, v in reference_dict['lower_bounds'].items()}
    if reference_dict.get('upper_bounds', None) is not None:
        batch_reference_dict['upper_bounds'] = {k: take_batch(v, batch_size, batch_idx)
                                                for k, v in reference_dict['upper_bounds'].items()}
    if reference_dict.get('lA', None) is not None:
        batch_reference_dict['lA'] = {k: take_batch(v, batch_size, batch_idx)
                                                for k, v in reference_dict['lA'].items()}
    if reference_dict.get('alphas', None) is not None:
        batch_reference_dict['alphas'] = {}
        for k, v in reference_dict['alphas'].items():
            batch_reference_dict['alphas'][k] = {}
            batch_reference_dict['alphas'][k]['alpha'] = {kk: take_batch(vv, batch_size, batch_idx, batch_dim=2)
                                                          for kk, vv in v['alpha'].items()}
            for item in v.keys():
                if item != 'alpha':
                    # item can be 'alpha_lookup_idx', 'alpha_indices', 'init'.
                    batch_reference_dict['alphas'][k][item] = reference_dict['alphas'][k][item]
    if reference_dict.get('mask', None) is not None:
        batch_reference_dict['mask'] = {}
        for k, v in reference_dict['mask'].items():
            batch_reference_dict['mask'][k] = []
            for vv in v:
                # mask can be None if the corresponding input node is not perturbed.
                if vv is None or vv.shape[0] == 1:
                    batch_reference_dict['mask'][k].append(vv)
                else:
                    batch_reference_dict['mask'][k].append(take_batch(vv, batch_size, batch_idx, batch_dim=0))

    if reference_dict.get('refined_betas', None) is not None:
        batch_reference_dict['refined_betas'] = reference_dict['refined_betas']
    if reference_dict.get('attack_examples', None) is not None:
        batch_reference_dict['attack_examples'] = take_batch(reference_dict['attack_examples'], batch_size, batch_idx)
    if reference_dict.get('attack_margins', None) is not None:
        batch_reference_dict['attack_margins'] = take_batch(reference_dict['attack_margins'], batch_size, batch_idx)
    if reference_dict.get('history', None) is not None:
        batch_reference_dict['history'] = reference_dict['history']

    return batch_x, batch_c, batch_rhs, batch_reference_dict


def _format_result_act_bab(result, lirpa_model, vnnlib):
    # If a counterexample is found in any node split LP, check and save it
    if result[2] == 'unsafe_bab':
        stats = result[3]
        adv_example = stats.counterexample.detach().to(lirpa_model.device)
        verified_status, _ = check_and_save_cex(
            adv_example, lirpa_model.net(adv_example), vnnlib,
            arguments.Config['attack']['cex_path'], 'unsafe')
        result = (*result[:2], verified_status)
    else:
        result = result[:3]
    # return the result in the format of (global_lb, num_domains, result)
    return result
