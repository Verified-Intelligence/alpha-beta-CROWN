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
import torch
import copy
import time
from auto_LiRPA.utils import stop_criterion_batch_any, multi_spec_keep_func_all
from cuts.cut_utils import fetch_cut_from_cplex, cut_analysis

import arguments

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet

class BICCOS:
    '''
    Branch-and-bound Inferred Cuts with Constraint Strengthening (BICCOS).

    How to use:

    bab:
        tree_traversal: breadth_first
        cut:
            enabled: True
            bab_cut: True
            number_cuts: 200
            biccos:
                enabled: True
                max_infer_iter: 20
                constraint_strengthening: True
                recursively_strengthening: True
                multi_tree_branching:
                    enabled: True
                    keep_n_best_domains: 1
                    k_splits: 200
                    iterations: 1
    '''
    def __init__(self, ret, property_idx, rhs):
        '''
        Initialize the BICCOS.
        ret: the return value of the beta crown solver, we only use this ret to get the 
            initial intermediate bounds and ceofficients (lA), and the mapping of the keys.
        '''
        self.biccos_cuts = []
        self.cumulative_time = 0
        self.decision_threshold = rhs.item()

        bab_args = arguments.Config['bab']
        biccos_args = bab_args['cut']['biccos']
        self.max_iter = biccos_args['max_infer_iter']
        self.max_cuts_num = bab_args['cut']['number_cuts']
        self.drop_ratio = biccos_args['drop_ratio']
        self.cplex_cuts_usage = bab_args['cut']['cplex_cuts']
        self.patches_cut = bab_args['cut']['patches_cut']
        self.recursively_strengthening = biccos_args['recursively_strengthening']
        self.save_cuts = biccos_args['save_biccos_cuts']
        self.bonus = biccos_args['verified_bonus']

        # let the lb, ub and lA corespond to the same key
        self.key_mapping = {key: index for index, key in enumerate(ret['lower_bounds'].keys())}
        self.key_mapping_lA = {key: index for index, key in enumerate(ret['lA'].keys())}

        self.lA_init = {self.key_mapping_lA[key]: value for key, value in ret['lA'].items()}
        self.lb_init = {self.key_mapping[key]: value for key, value in ret['lower_bounds'].items()}
        self.ub_init = {self.key_mapping[key]: value for key, value in ret['upper_bounds'].items()}

    def update_cut(self, d, net, ret, enforce_usage, heuristic=None, iter_idx=None):
        '''
        Main function to update the cuts.
        d: the dictionary contains the histories, bounds, betas, etc.
            {
            'mask': new_masks, 'lAs': new_lAs,
            'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds,
            'alphas': alphas, 'betas': betas_all,
            'intermediate_betas': intermediate_betas_all,
            'history': history, 'split_history': split_history,
            'global_lb': global_lb, 'depths': depths, 'cs': cs,
            'thresholds': thresholds, 'x_Ls': new_x_Ls, 'x_Us': new_x_Us,
            'input_split_idx': new_input_split_idx,
            }
        net: the beta crown solver
        ret: the return value of the bound_update() function in beta crown solver
            {
            'lower_bounds': ret_l, 'upper_bounds': ret_u,
            'lAs': lAs, 'alphas': ret_s,
            'betas': ret_b, 'split_history': new_split_history,
            'intermediate_betas': best_intermediate_betas,
            'primals': primal_x,
            'c': c, 'x_Ls': x_Ls, 'x_Us': x_Us,
            'input_split_idx': input_split_idx,
            }
        enforce_usage: whether to enforce the usage the original cuts
        heuristic: the heuristic to decide which neuron to drop
        iter_idx: the current iteration index
        '''
        print('======================Cut inference begins======================')
        start_cut_time = time.time() # record the start time
        inference_time = None
        unique_cuts_num = 0
        redundant_cuts_num = 0
        cplex_cuts = []
        self.final_name = net.final_name

        # extract the lower bound and the verified index
        lbs_final = ret['lower_bounds'][self.final_name]
        v_idx = torch.where(lbs_final > self.decision_threshold)[0]
        print("max lb", lbs_final.max(), "min lb", lbs_final.min())
        print(f'Number of Verified Splits: {len(v_idx)} of {len(lbs_final)}')

        if (iter_idx <= self.max_iter) or enforce_usage:
            if len(self.biccos_cuts) < self.max_cuts_num:

                # calculate the neuron influence score for all cases
                if heuristic == 'neuron_influence_score':
                    self.neuron_influence_score_cal(d['history'], d['lower_bounds'][self.final_name].to('cpu'), lbs_final)
                elif heuristic == 'random_drop':
                    print('Warning: Random drop heuristic used, performance may be bad.')
                elif heuristic == 'sparse_opt':
                    NotImplementedError('Sparse Optimization Heuristic is not implemented yet.')
                else:
                    print('Warning: No heuristic is used, performance may be bad.')

                if self.inference_condition(lbs_final) or (enforce_usage and (lbs_final > self.decision_threshold).any()):
                    inference_time = time.time() # record the preprocessing time
                    self.tmp_cuts = []
                    # Here we always use the constraint strengthening to strengthen the cuts.
                    # The cuts will be stored in self.tmp_cuts

                    if enforce_usage:
                        self.original_cut_inference(d, v_idx)
                    else:
                        self.constraint_strengthening(d, net, ret, v_idx, heuristic)

                    add_cuts_time = time.time() # record the inference time
                    # check the unique and redundant cuts
                    # here self.biccos_cuts is a list contains all the inferred cuts
                    # every iteration we inferred cuts and add the unique of them to self.biccos_cuts
                    for cut in self.tmp_cuts:
                        if cut not in self.biccos_cuts:
                            self.biccos_cuts.append(cut)
                            unique_cuts_num += 1
                        else:
                            redundant_cuts_num += 1
                    print(f'{len(self.tmp_cuts)} cuts inferred, {unique_cuts_num} unique cuts added, {redundant_cuts_num} redundant.')
                    cut_analysis_time = time.time() # record the analysis time
                else:
                    print('No cut inferred: All or none verified.')
            else:
                print('Stop inferencing: Max cuts number reached.')
        else:
            print('Stop inferencing: Max iteration reached.')

        # synchronize the cuts to the solver
        if self.cplex_cuts_usage:
            cplex_cuts, cut_timestamp = fetch_cut_from_cplex(net, sync_to_net=False)
            cplex_cuts = cplex_cuts if cplex_cuts is not None else []

        if unique_cuts_num > 0 or cplex_cuts:
            if cplex_cuts:
                net.net.cut_timestamp = cut_timestamp
                net.cutter.cuts = self.biccos_cuts + cplex_cuts
                print('BICCOS and MIP cuts are added to the cut module.')
            else:
                net.cutter.cuts = self.biccos_cuts
                print('BICCOS cuts are added to the cut module.')
            cut_module = net.cutter.construct_cut_module()
            net.net.cut_module = cut_module
            for m in net.net.relus:
                m.cut_module = cut_module

        if net.cutter.cuts:
            cut_analysis(net.cutter.cuts)
        else:
            cut_analysis(self.biccos_cuts)

        if self.save_cuts:
            self.save_biccos_cuts()

        stop_cut_time = time.time()
        self.cumulative_time += stop_cut_time - start_cut_time
        print('BICCOS time:', stop_cut_time - start_cut_time, '\nBICCOS Cumulative time:', self.cumulative_time)
        if inference_time:
            print('Preprocessing time:', inference_time - start_cut_time,
                  'Inference time:', add_cuts_time - inference_time,
                  'Add cuts time:', cut_analysis_time - add_cuts_time,
                  'Cut analysis time:', stop_cut_time - cut_analysis_time)
        print('======================Cut inference ends========================')

    def inference_condition(self, lbs_final):
        return not (lbs_final > self.decision_threshold).all() and (lbs_final > self.decision_threshold).any()

    def pick_d(self, v_idx, d):
        '''
        Pick the verified cases from the original d.
        Never use and change the original d in the following steps.
        '''
        d_new = {}
        max_idx = max(v_idx) + 1

        for key, value in d.items():
            if key in ['depths', 'history', 'betas']:
                d_new[key] = [copy.deepcopy(value[i]) for i in v_idx if i < len(value)]
            elif key in ['intermediate_betas', 'split_history']:
                d_new[key] = [value[i] for i in v_idx if i < len(value)]
            elif key in ['lower_bounds', 'upper_bounds']:
                d_new[key] = {k: copy.deepcopy(v[v_idx]) for k, v in value.items() if v.size(0) >= max_idx}
            elif key == 'lAs':
                d_new[key] = {k: v[v_idx] for k, v in value.items() if v.size(0) >= max_idx}
            elif key == 'alphas':
                d_new[key] = {sub_key: {tensor_key: tensor[:, :, v_idx, :]
                                        for tensor_key, tensor in sub_nested_dict.items() if tensor.size(2) >= max_idx}
                            for sub_key, sub_nested_dict in value.items()}
            elif key in ['cs', 'thresholds']:
                d_new[key] = value[v_idx]
        return d_new

    def constraint_strengthening(self, d, net, ret, v_idx, heuristic):
        '''
        Constraint Strengthening is used to strengthen the cuts.
        The verified cases are used to infer the cuts.
        Recursive call until no verified cases are left.

        input:
            d: the dictionary contains the verified cases.
            net: the beta crown solver.
            ret: the return value of the beta crown solver.
                we only need 'lower_bounds' and 'betas'.
            v_idx: the verified index.
            heuristic: the heuristic to decide which neuron to drop.

        update:
            self.tmp_cuts: a list of dictionaries contains the inferred cuts
        '''
        # infer the cut based on the influence socre, retrieve the verified domains and cuts
        d_revise, tmp_cuts, original_length = self.cut_inference(d, ret, v_idx, heuristic)

        # reverify the inferred cuts
        ret_revise = net.biccos_verification(d_revise,
                                        stop_criterion_func=stop_criterion_batch_any(d_revise['thresholds']),
                                        multi_spec_keep_func=multi_spec_keep_func_all)

        # extract the still verified index
        lbs_new = ret_revise['lower_bounds'][self.final_name]
        v_idx_new = torch.where(lbs_new > self.decision_threshold)[0]
        print(f'Number of Verified Cuts: {len(v_idx_new)} of {len(v_idx)}')

        # add the verified cuts
        for i in range(len(v_idx_new)):
            #if len(tmp_cuts[v_idx_new[i]]['arelu_coeffs']) < original_length[i]:
            self.tmp_cuts.append(tmp_cuts[v_idx_new[i]])

        # if there are still verified cases, do the constraint_strengthening recursively
        if self.recursively_strengthening and self.inference_condition(lbs_new):
            print('\nCuts Strengthening')
            self.constraint_strengthening(d_revise, net, ret_revise, v_idx_new, None)

    def _convert_history_from_list(self, history):
        '''
        Convert the history variables into tensors if they are lists.
        It is because some legacy code creates history as lists.

        return:
            history: a tuple of tensors
                history[0]: relu_idx
                history[1]: relu_status
                history[2]: relu_bias
                history[3]: relu_score
                history[4]: depths
        '''
        if isinstance(history[0], torch.Tensor):
            return history

        return (torch.tensor(history[0], dtype=torch.long),
                torch.tensor(history[1]),
                torch.tensor(history[2]),
                torch.tensor(history[3]),
                torch.tensor(history[4]))

    def original_cut_inference(self, d, v_idx):
        '''
        NOTUSE: This function is not used for the current version.

        Infer the original cuts based on the verified cases.
        Only used for the first iteration or the enforced usage.

        update:
            self.tmp_cuts: a list of dictionaries
        '''
        print('Original cuts are inferred.')
        for j in range(len(v_idx)):
            arelu_decision = []
            arelu_coeffs = []
            bias = 0
            for key in d['history'][v_idx[j]].keys():
                d['history'][v_idx[j]][key] = self._convert_history_from_list(d['history'][v_idx[j]][key])
            for key, (relu_idx, relu_status, _, _, _) in d['history'][v_idx[j]].items():
                # the relu_status is the split status of the neurons
                # 1 for >= 0 and -1 for <= 0
                key_int = self.key_mapping[key]
                for i in range(len(relu_idx)):
                    arelu_decision.append([key_int, relu_idx[i].item()])
                    arelu_coeffs.append(relu_status[i].item())
                    bias += relu_status[i].clamp(min=0).item()
            original_cut = self.generate_cut(relu_activation_decision=arelu_decision, relu_activation_coeffs=arelu_coeffs, b=bias-1)
            self.tmp_cuts.append(original_cut)

    def cut_inference(self, d, ret, v_idx, heuristic):
        '''
        Infer the cuts based on the verified cases.

        d, ret: the dictionary contains the verified cases and the return value of the beta crown solver.
        v_idx: the verified index.
        heuristic: the heuristic to decide which neuron to drop.

        Here we use heuristic to decide which neuron to drop.
        If the neuron is dropped, the corresponding bounds will be recovered, the histories and betas will be removed.
        Else, the neuron will be added to the cut.

        return:
            d: a dictionary contains the verified domains information 
            cuts: a list of dictionaries
        '''
        # deepcopy a new d only contains the verified cases
        d_revise = self.pick_d(v_idx, d)
        original_length = []
        tmp_cuts = []

        for j in range(len(v_idx)):
            relu_activation_decision = []
            relu_activation_coeffs = []
            bias = 0
            cut_length = 0
            # get the criterion for the j-th split history
            if heuristic == 'neuron_influence_score':
                criterion = self.influence_criterian_get(d['history'][v_idx[j]])
            # ensure that all history contains tensors
            for key in d['history'][v_idx[j]].keys():
                d['history'][v_idx[j]][key] = self._convert_history_from_list(d['history'][v_idx[j]][key])
            # inference procedure
            for key, (relu_idx, relu_status, relu_bias, relu_score, depths) in d['history'][v_idx[j]].items():
                key_int = self.key_mapping[key]
                cut_length += len(relu_idx)
                hist_index = []
                hist_split = []
                hist_bias = []
                hist_score = []
                hist_depths = []

                for i in range(len(relu_idx)):

                    if heuristic == 'random_drop':
                        condition = self.random_drop()
                    elif heuristic == 'neuron_influence_score':
                        condition = self.neuron_influence_score(relu_score[i], criterion)
                    elif heuristic == 'sparse_opt':
                        NotImplementedError('Sparse Optimization Heuristic is not implemented yet.')
                    else:
                        condition = False

                    # if (beta > 0 and upper bound used) or condition = True, keep the neuron to the cut, else drop the neuron
                    if (ret['betas'][v_idx[j]][key][i] > 0 and self.lA_init[key_int][0][0].flatten()[relu_idx[i]] <= 0) or condition:
                        # record the neuron and split status for cut
                        relu_activation_decision.append([key_int, relu_idx[i].item()])
                        relu_activation_coeffs.append(relu_status[i].item())
                        bias += relu_status[i].clamp(min=0).item()
                        # record the neuron and split status for history
                        hist_index.append(relu_idx[i].item())
                        hist_split.append(relu_status[i].item())
                        hist_bias.append(relu_bias[i].item())
                        hist_score.append(relu_score[i].item())
                        if depths is not None and len(relu_status) == len(depths):
                            hist_depths.append(depths[i].item())
                    else:
                        # drop the neuron, recover the bounds. The coresponding histories and betas removed.
                        d_revise['lower_bounds'][key][j].flatten()[relu_idx[i]] = self.lb_init[key_int][0].flatten()[relu_idx[i]]
                        d_revise['upper_bounds'][key][j].flatten()[relu_idx[i]] = self.ub_init[key_int][0].flatten()[relu_idx[i]]
                # we also recover the history and re-initialize the betas in 'd'
                d_revise['history'][j][key] = (torch.tensor(hist_index), torch.tensor(hist_split), torch.tensor(hist_bias), torch.tensor(hist_score), torch.tensor(hist_depths))
                # in the first bab round there may not be beta exist
                if d_revise['betas'][j] is not None:
                    d_revise['betas'][j][key] = torch.zeros_like(torch.tensor(hist_split))
            original_length.append(cut_length)
            tmp_cut = self.generate_cut(relu_activation_decision=relu_activation_decision, relu_activation_coeffs=relu_activation_coeffs, b=bias-1)
            tmp_cuts.append(tmp_cut)

        return d_revise, tmp_cuts, original_length

    def random_drop(self):
        from random import choice
        return choice([True, False])

    def neuron_influence_score(self, score, criterian):
        return score >= criterian

    def influence_criterian_get(self, hist):
        '''
        Get the criterion for the neuron influence score.

        The criterion is the quantile of the relu scores for all neurons in each layer.
        Defualt is the 0.5 quantile.

        return:
            criterion: the criterion for the neuron influence score
        '''
        relu_scores = [relu_score for _, (_, _, _, relu_score, _) in hist.items()]
        score = torch.cat(relu_scores).flatten()
        return score.quantile(self.drop_ratio, interpolation='midpoint')

    def neuron_influence_score_cal(self, d_hist, d_lbs_final, lbs_final):
        '''
        Calculate the neuron influence score based on the history and bounds.
        The neuron influence score is used to decide which neuron to drop.
        A bonus is added to the neurons with zero score to avoid the zero score.
        If the added neuron cause UNSAT, the bonus will be larger.

        update:
            d_hist: the history of the neurons
        '''
        lbs_score = lbs_final - d_lbs_final

        for j in range(len(d_lbs_final)):
            for key, (relu_idx, relu_status, relu_bias, relu_score, depths) in d_hist[j].items():
                relu_score = torch.tensor(relu_score)
                hist_score = relu_score.clone()
                # The newly added neuron's score is initialized to 0 and then assigned a value
                # with the newly calculated score (new - old) and given a bonus based on whether
                # it is verified or not
                hist_score[relu_score == 0] = lbs_score[j] + (self.bonus if lbs_final[j] > 0 else 1e-8)
                d_hist[j][key] = (relu_idx, relu_status, relu_bias, hist_score.flatten(), depths)

    def save_biccos_cuts(self, file_path='../../log/biccos_cuts'):
        '''
        Save the cuts to the file.
        '''
        with open(file_path, 'w') as file:
            for item in self.biccos_cuts:
                file.write(f"{item}\n")

    def generate_cut(self, input_decision=[], input_coeffs=[],
                     post_relu_decision=[], post_relu_coeffs=[],
                     relu_activation_decision=[], relu_activation_coeffs=[],
                     pre_relu_decision=[], pre_relu_coeffs=[],
                     b=0, c=-1):
        '''
        cut is a dictionary contains the cut information
        the logical cut is the cut only contains the relu_activation_decision and relu_activation_coeffs
        '''
        return {
            'x_decision': input_decision,
            'x_coeffs': input_coeffs,
            'relu_decision': post_relu_decision,
            'relu_coeffs': post_relu_coeffs,
            'arelu_decision': relu_activation_decision,
            'arelu_coeffs': relu_activation_coeffs,
            'pre_decision': pre_relu_decision,
            'pre_coeffs': pre_relu_coeffs,
            'bias': b,
            'c': c,
        }

def biccos_verification(self: 'LiRPANet', d, fix_interm_bounds=True,
                      stop_criterion_func=None,
                      multi_spec_keep_func=None):
    '''
    Verifing BICCOS cuts in GCP-CROWN.
    '''
    deterministic_opt = arguments.Config['general']['deterministic_opt']
    beta_args = arguments.Config['solver']['beta-crown']

    iteration = beta_args['iteration']
    enable_opt_interm_bounds = beta_args['enable_opt_interm_bounds']
    batch = d['upper_bounds'][self.final_name].shape[0]
    decision_thresh = d.get('thresholds', None)

    if self.net.cut_used:
        self.disable_cut_for_branching()

    splits_per_example = self.set_beta(d, bias=None)
    self.net.cut_used = (
                    arguments.Config['bab']['cut']['enabled']
                    and arguments.Config['bab']['cut']['bab_cut']
                    and getattr(self.net, 'cut_module', None) is not None)

    if self.net.cut_used:
        iteration = self.set_cut_params(
            batch, batch, d.get('split_history', None))

    ret = self._expand_tensors(d, batch)
    interm_bounds, lb_last, _, c, new_x, _, _ = ret
    self.set_alpha(d, set_all=enable_opt_interm_bounds)

    self.net.set_bound_opts({
        'optimize_bound_args': {
            'enable_beta_crown': True,
            'fix_interm_bounds': fix_interm_bounds,
            'stop_criterion_func': stop_criterion_func,
            'multi_spec_keep_func': multi_spec_keep_func,
            'iteration': iteration,
        },
        'enable_opt_interm_bounds': enable_opt_interm_bounds,
    })
    self.set_crown_bound_opts('beta')

    if fix_interm_bounds:
        reference_bounds = {}
        for name in self.alpha_start_nodes:
            if name in interm_bounds:
                reference_bounds[name] = interm_bounds[name]
                interm_bounds.pop(name)
    else:
        reference_bounds = interm_bounds
        interm_bounds = {}
    if len(reference_bounds):
        print('Recompute intermediate bounds for nodes:',
            ', '.join(list(reference_bounds.keys())))

    tmp_ret = self.net.compute_bounds(
            x=(new_x,), C=c, method='CROWN-optimized',
            interm_bounds=interm_bounds, reference_bounds=reference_bounds,
            return_A=False, needed_A_dict=self.needed_A_dict,
            cutter=self.cutter, bound_upper=False,
            decision_thresh=decision_thresh)

    lb, _ = tmp_ret
    ub = torch.full_like(lb, fill_value=torch.inf, device='cpu')  # dummy upper bound

    with torch.no_grad():
        # Move tensors to CPU for all elements in this batch.
        lb = lb.to(device='cpu')
        ret_b = self.get_beta(splits_per_example, device='cpu')

        # Reorganize tensors.
        ret_l, _ = self.get_candidate_parallel(lb, ub, device='cpu')
        ret_l[self.final_name] = torch.max(ret_l[self.final_name], lb_last.cpu())

    return {
            'lower_bounds': ret_l, 'betas': ret_b,
        }
