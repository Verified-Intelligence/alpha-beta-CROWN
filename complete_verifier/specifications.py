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
"""Various kinds of specifications for verification."""
from numpy import ndarray

import arguments
import torch
import numpy as np
from typing import Union

from beta_CROWN_solver import LiRPANet

class Specification:
    def __init__(self):
        self.num_outputs = arguments.Config['data']['num_outputs']
        # FIXME Do not use numpy. Use torch instead.
        self.rhs = np.array([arguments.Config['bab']['decision_thresh']])

    def construct_vnnlib(self):
        raise NotImplementedError


class SpecificationVerifiedAcc(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            c = (torch.eye(self.num_outputs)[label].unsqueeze(1)
                - torch.eye(self.num_outputs).unsqueeze(0))
            I = (~(label.unsqueeze(1) == torch.arange(
                    self.num_outputs).type_as(label.data).unsqueeze(0)))
            c = c[I].view(1, self.num_outputs - 1, self.num_outputs)
            new_c = []
            for ii in range(self.num_outputs - 1):
                new_c.append((c[:, ii], self.rhs))
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationTarget(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            target_label = dataset['target_label'][example_idx_list[i]]
            c = torch.zeros([1, self.num_outputs])
            c[0, label] = 1
            c[0, target_label] = -1
            new_c = [(c, self.rhs)]
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationRunnerup(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            runnerup = dataset['runnerup'][example_idx_list[i]]
            c = torch.zeros([1, self.num_outputs])
            c[0, label] = 1
            c[0, runnerup] = -1
            new_c = [(c, self.rhs)]
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationAllPositive(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            this_x_range = x_range[i]
            c = torch.eye(self.num_outputs).unsqueeze(0)
            new_c = []
            for ii in range(self.num_outputs):
                new_c.append((c[:, ii], self.rhs))
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


def construct_vnnlib(dataset, example_idx_list):
    X = dataset['X']
    x_lower = x_upper = None
    if arguments.Config['specification']['type'] == 'lp':
        perturb_epsilon = dataset['eps']
        if type(perturb_epsilon) == list:
            # Each example has different perturbations.
            perturb_epsilon = torch.cat(perturb_epsilon)
            perturb_epsilon = perturb_epsilon[example_idx_list]
        assert perturb_epsilon is not None
        # FIXME why flatten?
        if arguments.Config['specification']['norm'] == float('inf'):
            if dataset.get('data_max', None) is None:
                # perturb_eps is already normalized.
                x_lower = (X[example_idx_list] - perturb_epsilon).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).flatten(1)
            else:
                x_lower = (X[example_idx_list] - perturb_epsilon).clamp(
                    min=dataset['data_min']).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).clamp(
                    max=dataset['data_max']).flatten(1)
            x_range = torch.stack([x_lower, x_upper], -1).numpy()
        else:
            # TODO create classes to handle it generally
            x_range = []
            for idx in example_idx_list:
                x_item = {
                    'X': X[idx],
                    'eps': dataset['eps'],
                    'norm': dataset['norm'],
                }
                if not isinstance(x_item['X'], torch.Tensor):
                    x_item['X'] = torch.tensor(x_item['X'])
                if 'eps_min' in dataset:
                    x_item['eps_min'] = dataset['eps_min']
                x_item['data_min'] = x_item['X'] - dataset['eps']
                x_item['data_max'] = x_item['X'] + dataset['eps']
                if dataset.get('data_min', None) is not None:
                    x_item['data_min'] = x_item['data_min'].clamp(
                        min=dataset['data_min'])
                if dataset.get('data_max', None) is not None:
                    x_item['data_max'] = x_item['data_max'].clamp(
                        min=dataset['data_max'])
                x_item['data_min'] = x_item['data_min']
                x_item['data_max'] = x_item['data_max']
                x_range.append(x_item)
    elif (arguments.Config['specification']['type'] == 'box' or
            # Some old config files use "bound"; keep for compatibility.
            arguments.Config['specification']['type'] == 'bound'):
        x_lower = dataset['data_min'].flatten(1)
        x_upper = dataset['data_max'].flatten(1)
        x_range = torch.stack([x_lower, x_upper], -1).numpy()
    else:
        raise ValueError('Unsupported perturbation type ' +
                         arguments.Config['specification']['type'])

    # TODO rename "robustness_type", since the verification objective may
    # not be related to robustness.
    robustness_type = arguments.Config['specification']['robustness_type']
    if robustness_type == 'verified-acc':
        specification = SpecificationVerifiedAcc()
    elif robustness_type == 'specify-target':
        specification = SpecificationTarget()
    elif robustness_type == 'runnerup':
        specification = SpecificationRunnerup()
    elif robustness_type == 'all-positive':
        specification = SpecificationAllPositive()
    else:
        raise ValueError(robustness_type)

    return specification.construct_vnnlib(
        dataset, x_range, example_idx_list)


def sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub, scores,
                     reference_alphas, lA, final_node_name, reverse=False):
    # TODO need minus rhs
    # To sort targets, this must be a classification task, and initial_max_domains
    # is set to 1.
    assert len(batched_vnnlib) == init_global_lb.shape[0] and init_global_lb.shape[1] == 1
    sorted_idx = scores.argsort(descending=reverse)
    batched_vnnlib = [batched_vnnlib[i] for i in sorted_idx]
    init_global_lb = init_global_lb[sorted_idx]
    init_global_ub = init_global_ub[sorted_idx]

    if reference_alphas is not None:
        for spec_dict in reference_alphas.values():
            for spec in spec_dict:
                if spec == final_node_name:
                    if spec_dict[spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = spec_dict[spec][:, sorted_idx]
                    else:
                        spec_dict[spec] = spec_dict[spec][:, :, sorted_idx]

    if lA is not None:
        lA = {k: v[sorted_idx] for k, v in lA.items()}

    return batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx


def trim_batch(model, init_global_lb, init_global_ub, reference_alphas_cp,
               orig_lower_bounds, orig_upper_bounds, reference_alphas, lA, property_idx,
               c, rhs):
    net = model.net
    optimize_disjuncts_separately = arguments.Config['solver']['optimize_disjuncts_separately']

    # FIXME (assigned to Kaidi, Jun 2023): this function might be wrong; it does not handles
    # the case with a few AND statements like yolo.
    # extract lower bound by (sorted) init_global_lb and batch size of initial_max_domains
    start_idx = property_idx * arguments.Config['bab']['initial_max_domains']
    lower_bounds = {}
    upper_bounds = {}
    if optimize_disjuncts_separately:
        for layer_name in orig_lower_bounds.keys():
            lower_bounds[layer_name] = orig_lower_bounds[layer_name][start_idx: start_idx + c.shape[0]]
            upper_bounds[layer_name] = orig_upper_bounds[layer_name][start_idx: start_idx + c.shape[0]]
        assert torch.all(lower_bounds[net.final_name] == init_global_lb[start_idx: start_idx + c.shape[0]])
        assert torch.all(upper_bounds[net.final_name] == init_global_ub[start_idx: start_idx + c.shape[0]])
    else:
        for layer_name in orig_lower_bounds.keys():
            lower_bounds[layer_name] = orig_lower_bounds[layer_name]
            upper_bounds[layer_name] = orig_upper_bounds[layer_name]
        lower_bounds[net.final_name] = init_global_lb[start_idx: start_idx + c.shape[0]]
        upper_bounds[net.final_name] = init_global_ub[start_idx: start_idx + c.shape[0]]
    if rhs.numel() > 1:
        if optimize_disjuncts_separately:
            raise NotImplementedError("Output constraints for disjunctions are not supported for rhs.numel() > 1")
    # trim reference slope by batch size of initial_max_domains accordingly
    if reference_alphas is not None:
        for m, spec_dict in reference_alphas.items():
            for spec in spec_dict:
                if spec == net.final_node().name:
                    if reference_alphas_cp[m][spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = reference_alphas_cp[m][spec][
                            :, start_idx: start_idx + c.shape[0]]
                    else:
                        spec_dict[spec] = reference_alphas_cp[m][spec][
                            :, :, start_idx: start_idx + c.shape[0]]
    # trim lA by batch size of initial_max_domains accordingly
    if lA is not None:
        lA = {k: v[start_idx: start_idx + c.shape[0]] for k, v in lA.items()}
    return {
        'lA': lA, 'rhs': rhs, 'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds
    }


def prune_by_idx(reference_alphas, init_verified_cond, final_name, lA_trim, x, data_min, data_max,
                 need_prune_lA, lower_bounds, upper_bounds, c):
    """
     Prune reference_alphas, lA_trim, x, data_min, data_max, lower_bounds, upper_bounds, c
     by init_verified_cond. Only keep unverified elements for next step bab or attack.
    """

    init_unverified_cond = ~init_verified_cond

    if reference_alphas is not None:
        LiRPANet.prune_reference_alphas(
            reference_alphas, init_unverified_cond, final_name)
    if need_prune_lA:
        lA_trim = LiRPANet.prune_lA(lA_trim, init_unverified_cond)

    if data_min.shape[0] > 1:
        # use [torch.where(~init_verified_cond)[0]] to prune x
        # when vnnlib has multiple different x
        # fixed: don't repeat x and then take [0:1]
        x, data_min, data_max = \
            x[torch.where(init_unverified_cond)[0]], \
            data_min[torch.where(init_unverified_cond)[0]], \
            data_max[torch.where(init_unverified_cond)[0]]

    lower_bounds[final_name] = lower_bounds[final_name][init_unverified_cond]
    upper_bounds[final_name] = upper_bounds[final_name][init_unverified_cond]
    c = c[torch.where(init_unverified_cond)[0]]

    return reference_alphas, lA_trim, x, data_min, data_max, lower_bounds, upper_bounds, c


def batch_vnnlib(vnnlib):
    """reorganize original vnnlib file, make x, c and rhs batch wise"""
    final_merged_rv = []

    init_d = {'x': [], 'c': [], 'rhs': [],
              'verify_criterion': [], 'attack_criterion': [] }
    target_labels = []

    for vnn in vnnlib:
        for mat, rhs in vnn[1]:
            if isinstance(vnn[0], dict):
                init_d['x'].append(vnn[0])
            else:
                init_d['x'].append(np.array(vnn[0]))
            init_d['c'].append(mat)
            init_d['rhs'].append(rhs)
            tmp_true_labels, tmp_target_labels = [], []
            for m in mat:

                target_label = np.where(m == -1)[-1]
                if len(target_label) != 0:
                    assert len(target_label) == 1
                    tmp_target_labels.append(target_label[0])
                else:
                    tmp_target_labels.append(None)

            target_labels.append(np.array(tmp_target_labels))

    if len(init_d['x']) == 0 or isinstance(init_d['x'][0], np.ndarray):
        # n, shape, 2; the batch dim n is necessary, even if n = 1
        init_d['x'] = np.array(init_d['x'])
    init_d['c'] = torch.concat(
        [(item if isinstance(item, torch.Tensor)
          else torch.tensor(item)).unsqueeze(0)
         for item in init_d['c']], dim=0)
    init_d['rhs'] = np.array(init_d['rhs'])  # n, n_output
    target_labels = np.array(target_labels)

    # batch_size = min(
    #         arguments.Config['solver']['batch_size'],
    #         arguments.Config['bab']['initial_max_domains'])
    # initial_max_domains can be much larger than batch_size if auto_enlarge_batch_size enabled
    batch_size = arguments.Config['bab']['initial_max_domains']

    total_batch = int(np.ceil(len(init_d['x']) / batch_size))
    print(f"Total VNNLIB file length: {len(init_d['x'])}, max property batch size: {batch_size}, total number of batches: {total_batch}")

    for i in range(total_batch):
        # [x, [(c, rhs, y, pidx)]], pidx can be none
        final_merged_rv.append([
            init_d['x'][i * batch_size: (i + 1) * batch_size],
            [(init_d['c'][i * batch_size: (i + 1) * batch_size],
              init_d['rhs'][i * batch_size: (i + 1) * batch_size],
              target_labels[i * batch_size: (i + 1) * batch_size]
            )]])

    return final_merged_rv


def sort_targets(batched_vnnlib, init_global_lb, init_global_ub,
                 attack_images=None, attack_margins=None, results=None,
                 model_incomplete=None):
    bab_attack_enabled = arguments.Config['bab']['attack']['enabled']
    sort_targets = arguments.Config['bab']['sort_targets']
    cplex_cuts = arguments.Config['bab']['cut']['enabled'] and arguments.Config['bab']['cut']['cplex_cuts']
    optimize_disjuncts_separately = arguments.Config['solver']['optimize_disjuncts_separately']
    reference_alphas = results.get('alpha', None)
    lA = results.get('lA', None)

    ret = None
    if bab_attack_enabled:
        # Sort specifications based on adversarial attack margins.
        ret = sort_targets_cls(
            batched_vnnlib, init_global_lb, init_global_ub, lA=lA,
            scores=attack_margins.flatten(), reference_alphas=reference_alphas,
            final_node_name=model_incomplete.net.final_node().name)
        attack_images = attack_images[:, :, ret[-1]]
    elif cplex_cuts:
        # need to sort pidx such that easier first according to initial alpha crown
        ret = sort_targets_cls(
            batched_vnnlib, init_global_lb, init_global_ub, lA=lA,
            scores=init_global_lb.flatten(), reference_alphas=reference_alphas,
            final_node_name=model_incomplete.net.final_node().name,
            reverse=True)
    elif sort_targets:
        # Sort specifications based on incomplete verifier bounds.
        ret = sort_targets_cls(
            batched_vnnlib, init_global_lb, init_global_ub, lA=lA,
            scores=init_global_lb.flatten(), reference_alphas=reference_alphas,
            final_node_name=model_incomplete.net.final_node().name)
    if ret:
        assert not optimize_disjuncts_separately, (
            "Sorting targets is currently not supported when disjuncts are optimized separately."
        )
        batched_vnnlib, init_global_lb, init_global_ub, lA = ret[:-1]

    return batched_vnnlib, init_global_lb, init_global_ub, lA, attack_images


def add_rhs_offset(
        vnnlib: list,
        rhs_offset: Union[np.ndarray, int, float] = None
) -> list:
    """
    Updates the second operand's offset value where rhs_offset is either a scalar that may be
    broadcast to ALL clauses, or rhs_offset is an array of offset values applied to each clause.
    @param vnnlib:      The vnnlib file formatted as a list object. Structure can be found in the
                        read_vnnlib.md.
    @param rhs_offset:  Scalar, array, or None. If array, it modifies the offsets in the clauses 
                        of the vnnlib file accordingly. If scalar, it is broadcast to all clauses.
    @return:            The modified vnnlib object
    """
    # If rhs_offset is None, return the original vnnlib
    if rhs_offset is None:
        return vnnlib

    # For debugging, add a print statement if sanity check is enabled
    if arguments.Config['debug']['sanity_check'] in ['Full', "Full+Graph"]:
        print('Add an offset to RHS for debugging:', rhs_offset)

    # Determine if rhs_offset is a scalar or array
    is_scalar = isinstance(rhs_offset, (int, float))
    
    updated_vnnlib = []
    k = 0  # Index counter if rhs_offset is an array

    for v in vnnlib:
        result = []
        for i in range(len(v[1])):
            if is_scalar:
                # If scalar, broadcast the same rhs_offset to all clauses
                item = (v[1][i][0], v[1][i][1] + rhs_offset + 1e-3)
            else:
                # If rhs_offset is an array, apply the offset to each clause
                item = (v[1][i][0], v[1][i][1] + rhs_offset[k:k + len(v[1][i][1])] + 1e-3)
                k += len(v[1][i][1])
            result.append(item)
        updated_vnnlib.append((v[0], result))
    
    return updated_vnnlib
