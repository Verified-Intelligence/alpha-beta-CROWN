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
import torch
import copy
import time
import json

from utils import convert_history_from_list
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
    '''
    def __init__(self, ret, rhs, final_name):
        '''
        Initialize the BICCOS.
        ret: the return value of the beta crown solver, we only use this ret to get the 
            initial intermediate bounds and ceofficients (lA), and the mapping of the keys.
        '''
        self.biccos_cuts = []
        self.cumulative_time = 0

        bab_args = arguments.Config['bab']
        debug_args = arguments.Config['debug']
        biccos_args = bab_args['cut']['biccos']
        self.max_iter = biccos_args['max_infer_iter']
        self.max_cuts_num = bab_args['cut']['number_cuts']
        self.max_domain = biccos_args['max_domain']
        self.drop_ratio = biccos_args['drop_ratio']
        self.cplex_cuts_usage = bab_args['cut']['cplex_cuts']
        self.patches_cut = bab_args['cut']['patches_cut']
        self.save_cuts = biccos_args['save_cuts']
        self.lb_ub_sanity_check = debug_args['sanity_check']
        self.heuristic = biccos_args['heuristic']
        self.auto_param = biccos_args['auto_param']
        self.final_name = final_name
        self.initial_bs_ratio = arguments.Config['solver']['min_batch_size_ratio']
        self.remaining_OR_spec_count = ret['lower_bounds'][self.final_name].shape[0]
        self.enable_constraint_strengthen = biccos_args['constraint_strengthening']
        self.cplex_cuts = []
        print('BICCOS cut inference is enabled. Initial parameters are set.')
        if self.lb_ub_sanity_check:
            print(' Warning: lb_ub_sanity_check is enabled, set the minimal batch size ratio be 0')
            self.decision_thresh = rhs.item()
        else:
            print('Set the minimal batch size ratio be 0 (default)')
            self.decision_thresh = 0

        # let the lb, ub and lA corresponding to the same key
        self.key_mapping = {key: index for index, key in enumerate(ret['lower_bounds'].keys())}
        self.key_mapping_lb = {index: key for index, key in enumerate(ret['lower_bounds'].keys())}
        self.key_mapping_lA = {index: key for index, key in enumerate(ret['lA'].keys())}
        self.lb_init = {self.key_mapping[key]: value for key, value in ret['lower_bounds'].items()}
        self.ub_init = {self.key_mapping[key]: value for key, value in ret['upper_bounds'].items()}

    def update_cut(self, d, net, ret, enforce_usage, domain_visited, heuristic=None, iter_idx=None):
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
        new_cplex_cuts = []
        self.final_name = net.final_name

        # extract the lower bound and the verified index
        lbs_final = ret['lower_bounds'][self.final_name]
        v_idx = torch.where(lbs_final > self.decision_thresh)[0]
        print("max lb", lbs_final.max(), "min lb", lbs_final.min())
        print(f'Number of Verified Splits: {len(v_idx)} of {len(lbs_final)}')

        # In this block, we store cuts as a dict, replicating a hash set
        # The keys are generated by dumping the cuts (they are dicts that contain lists) using json,
        # because that's a stable way to encode them (https://stackoverflow.com/a/22003440/1394336)
        # This is necessary, because we repeatedly need to check for duplicates, and comparing dicts is very slow
        new_cuts = dict()

        if (iter_idx <= self.max_iter) or enforce_usage:
            if domain_visited < self.max_domain or enforce_usage:

                # calculate the neuron influence score for all cases
                if heuristic == 'neuron_influence_score':
                    self.neuron_influence_score_cal(d['history'], d['lower_bounds'][self.final_name].to('cpu'), lbs_final)
                elif heuristic == 'random_drop':
                    print('Warning: Random drop heuristic used, performance may be bad.')
                elif heuristic == 'sparse_opt':
                    raise NotImplementedError('Sparse Optimization Heuristic is not implemented yet.')

                if self.inference_condition(lbs_final) or (enforce_usage and (lbs_final > self.decision_thresh).any()):
                    inference_time = time.time() # record the preprocessing time
                    self.tmp_cuts = []
                    # Here we always use the constraint strengthening to strengthen the cuts.
                    # The cuts will be stored in self.tmp_cuts
                    self.constraint_strengthening(d, net, ret, v_idx, heuristic)

                    add_cuts_time = time.time() # record the inference time
                    # check the unique and redundant cuts
                    # here self.biccos_cuts is a list contains all the inferred cuts
                    # every iteration we inferred cuts and add the unique of them to self.biccos_cuts
                    for cut in self.tmp_cuts:
                        cut_key = json.dumps(cut, sort_keys=True)
                        if cut not in self.biccos_cuts and cut_key not in new_cuts:
                            new_cuts[cut_key] = cut
                            unique_cuts_num += 1
                        else:
                            redundant_cuts_num += 1
                    self.biccos_cuts = self.merge_cuts(new_cuts)
                    print(f'{unique_cuts_num} unique cuts inferred, {redundant_cuts_num} redundant cuts found.')
                    cut_analysis_time = time.time() # record the analysis time
                else:
                    print('No cut inferred: All or none verified.')
            else:
                print('Stop inferring: Max cuts number / domain visited reached.')
                self.biccos_cuts = self.biccos_cuts[:self.max_cuts_num+1]
        else:
            print('Stop inferring: Max iteration reached.')

        # synchronize the cuts to the solver
        if self.cplex_cuts_usage:
            new_cplex_cuts, cut_timestamp = fetch_cut_from_cplex(net, sync_to_net=False)
            if new_cplex_cuts is not None:
                self.cplex_cuts = new_cplex_cuts

        if unique_cuts_num > 0 or new_cplex_cuts:
            # We always include the most recent cplex cuts, otherwise they would be dropped.
            net.cutter.cuts = self.biccos_cuts + self.cplex_cuts
            if new_cplex_cuts:
                net.net.cut_timestamp = cut_timestamp
                print('BICCOS and MIP cuts are added to the cut module.')
            else:
                print('BICCOS cuts are added to the cut module.')
            cut_module = net.cutter.construct_cut_module()
            net.net.cut_module = cut_module
            for m in net.net.relus:
                m.cut_module = cut_module

        if self.save_cuts and net.cutter.cuts is not None:
            self.save_biccos_cuts(net.cutter.cuts)

        if net.cutter.cuts:
            cut_analysis(net.cutter.cuts)
        else:
            cut_analysis(self.biccos_cuts)

        stop_cut_time = time.time()
        self.cumulative_time += stop_cut_time - start_cut_time
        print('BICCOS time:', stop_cut_time - start_cut_time, '\nBICCOS Cumulative time:', self.cumulative_time)
        if inference_time:
            print('Preprocessing time:', inference_time - start_cut_time,
                  'Inference time:', add_cuts_time - inference_time,
                  'Add cuts time:', cut_analysis_time - add_cuts_time,
                  'Cut analysis time:', stop_cut_time - cut_analysis_time)
        print('======================Cut inference ends========================')

    def merge_cuts(self, new_cuts_arg):
        '''
        The merge_cuts function combines cuts that differ by one decision component.
        Merging creates a more general (parent) cut. This is repeated iteratively.
        '''
        current_cuts = dict()
        # Initialize current_cuts with new_cuts_arg and self.biccos_cuts
        # Ensuring no mutation of the input dictionary if it's reused elsewhere.
        for k, v in new_cuts_arg.items():
            current_cuts[k] = v

        for cut_dict in self.biccos_cuts:
            cut_key = json.dumps(cut_dict, sort_keys=True)
            # The assert was useful for debugging, but in production, you might just overwrite or decide policy.
            # if cut_key in current_cuts:
            #     assert current_cuts[cut_key] == cut_dict 
            current_cuts[cut_key] = cut_dict

        continue_merging = True
        while continue_merging:
            continue_merging = False # Assume no merges in this pass initially

            # Cuts generated in this pass (parents) + cuts that didn't merge
            cuts_after_merging_pass = {}
            # Tracks keys of cuts already incorporated into a parent or processed as a sibling
            processed_in_pass = set()

            # Optional: Sorting here by 'arelu_decision' was in the original.
            # If not strictly necessary for merge correctness, removing it saves time.
            # For now, let's keep it to closely match original intent if there was one.
            # However, iterating directly over current_cuts.items() is often fine.
            
            # Create a list of items to iterate over, as current_cuts might be modified conceptually
            # by moving items to processed_in_pass or cuts_after_merging_pass
            # The sort key x[1]['arelu_decision'] for merging can be costly.
            # Iterating without this sort for merging phase for potential speedup:
            cuts_to_process_list = list(current_cuts.items())

            for cut_key, cut_data in cuts_to_process_list:
                if cut_key in processed_in_pass:
                    continue # Already merged as part of another pair

                cut_initiated_merge = False
                for i, coeff in enumerate(cut_data['arelu_coeffs']):
                    if coeff == -1.0: # Only positive coefficients look for negative siblings
                        continue

                    # Construct the potential sibling cut
                    sibling_arelu_coeffs = list(cut_data['arelu_coeffs']) # Make a mutable copy
                    sibling_arelu_coeffs[i] = -1.0

                    sibling_cut_dict = {
                        'x_decision': list(cut_data['x_decision']), # Create copies
                        'x_coeffs': list(cut_data['x_coeffs']),
                        'relu_decision': list(cut_data['relu_decision']),
                        'relu_coeffs': list(cut_data['relu_coeffs']),
                        'arelu_decision': list(cut_data['arelu_decision']),
                        'arelu_coeffs': sibling_arelu_coeffs,
                        'pre_decision': list(cut_data['pre_decision']),
                        'pre_coeffs': list(cut_data['pre_coeffs']),
                        'bias': cut_data['bias'] - 1, # Bias rule from original
                        'c': cut_data['c'], # 'c' value from original
                    }
                    sibling_cut_key = json.dumps(sibling_cut_dict, sort_keys=True)

                    if sibling_cut_key in current_cuts and sibling_cut_key not in processed_in_pass:
                        # Found a mergable sibling that hasn't been processed yet
                        processed_in_pass.add(cut_key)
                        processed_in_pass.add(sibling_cut_key)
                        cut_initiated_merge = True # Mark that the original cut initiated a merge

                        # Construct the parent cut by removing the differing decision
                        parent_arelu_decision = cut_data['arelu_decision'][:i] + cut_data['arelu_decision'][i+1:]
                        parent_arelu_coeffs = cut_data['arelu_coeffs'][:i] + cut_data['arelu_coeffs'][i+1:]

                        parent_cut_dict = {
                            'x_decision': list(cut_data['x_decision']),
                            'x_coeffs': list(cut_data['x_coeffs']),
                            'relu_decision': list(cut_data['relu_decision']),
                            'relu_coeffs': list(cut_data['relu_coeffs']),
                            'arelu_decision': parent_arelu_decision,
                            'arelu_coeffs': parent_arelu_coeffs,
                            'pre_decision': list(cut_data['pre_decision']),
                            'pre_coeffs': list(cut_data['pre_coeffs']),
                            'bias': cut_data['bias'] - 1, # Bias rule from original
                            'c': cut_data['c'],
                        }
                        parent_cut_key = json.dumps(parent_cut_dict, sort_keys=True)

                        cuts_after_merging_pass[parent_cut_key] = parent_cut_dict

                        # This cut has been merged; break from its coefficient loop
                        break

                if not cut_initiated_merge and cut_key not in processed_in_pass:
                    # If the cut didn't initiate a merge and wasn't consumed as a sibling
                    cuts_after_merging_pass[cut_key] = cut_data

            # Pruning redundant cuts
            # Sort by complexity (length of arelu_decision) so more general cuts are checked first
            merged_cuts_list_for_pruning = sorted(list(cuts_after_merging_pass.items()), key=lambda x: len(x[1]['arelu_decision']))

            pruned_cuts = {}
            for cut_key, cut_data in merged_cuts_list_for_pruning:
                is_redundant = False
                for _, potential_parent_cut in pruned_cuts.items():
                    if self.is_cut_a_parent(cut=cut_data, potential_parent=potential_parent_cut):
                        is_redundant = True
                        break
                if not is_redundant:
                    # Original code had: `and cut['arelu_decision']:`
                    # This means cuts with empty 'arelu_decision' are discarded.
                    if cut_data.get('arelu_decision', []): # Ensure it's not empty
                        pruned_cuts[cut_key] = cut_data

            # If the number of cuts changed, continue merging
            if len(pruned_cuts) < len(current_cuts):
                current_cuts = pruned_cuts
                continue_merging = True
            else:
                # If lengths are same, check if content actually changed.
                # This simplistic check might not catch all edge cases if keys are same but values differ.
                # A more robust check would be deep comparison or comparing key sets if values are guaranteed same by key.
                # For now, rely on length reduction as primary trigger.
                if set(pruned_cuts.keys()) != set(current_cuts.keys()):
                    current_cuts = pruned_cuts
                    continue_merging = True # Content changed even if length didn't (e.g. A,B -> P,C and P already existed)
                else:
                    # If keys are the same and length is the same, assume no effective change.
                    current_cuts = pruned_cuts # Final update
                    continue_merging = False
        return list(current_cuts.values())

    def is_cut_a_parent(self, cut, potential_parent):
        for potential_parent_decision, potential_parent_coeff in zip(potential_parent['arelu_decision'], potential_parent['arelu_coeffs']):
            potential_parent_decision_found = False
            for cut_decision, cut_coeff in zip(cut['arelu_decision'], cut['arelu_coeffs']):
                if potential_parent_decision == cut_decision and potential_parent_coeff == cut_coeff:
                    potential_parent_decision_found = True
                    break
            if not potential_parent_decision_found:
                return False
        return True

    def inference_condition(self, lbs_final):
        return not (lbs_final > self.decision_thresh).all() and (lbs_final > self.decision_thresh).any()

    def pick_d(self, v_idx, d):
        '''
        Pick the verified cases from the original d.
        Never use and change the original d in the following steps.
        '''
        d_new = {}
        max_idx = max(v_idx) + 1

        for key, value in d.items():
            # --- List of dictionaries ---
            # These keys will be modified
            if key in ['depths', 'history', 'betas']:
                d_new[key] = [copy.deepcopy(value[i]) for i in v_idx if i < len(value)]
            # These won't be modified
            elif key in ['intermediate_betas', 'split_history']:
                d_new[key] = [value[i] for i in v_idx if i < len(value)]
            # --- Dictionaries ---
            elif key in ['lower_bounds', 'upper_bounds']:
                d_new[key] = {k: copy.deepcopy(v[v_idx]) for k, v in value.items() if v.size(0) >= max_idx}
            elif key == 'lAs':
                d_new[key] = {k: v[v_idx] for k, v in value.items() if v.size(0) >= max_idx}
            elif key == 'alphas':
                d_new[key] = {sub_key: {tensor_key: tensor[:, :, v_idx, :]
                                        for tensor_key, tensor in sub_nested_dict.items() if tensor.size(2) >= max_idx}
                            for sub_key, sub_nested_dict in value.items()}
            # --- Tensors ---
            elif key in ['cs', 'thresholds']:
                d_new[key] = value[v_idx]
        return d_new

    def set_auto_params(self):
        '''
        Set the auto parameters for the BICCOS.
        The auto parameters are used to decide whether to enable the MTS.
        If the number of verified candidates is less than 3, enable the MTS.
        If the number of verified candidates is more than 2, disable the MTS.
        '''
        if self.auto_param:
            if self.remaining_OR_spec_count < 3:
                print('Warning: The number of verified candidates is less than 3, enable MTS.')
                arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled'] = True
            else:
                print('Warning: The number of verified candidates is more than 2, disable MTS.')
                arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled'] = False
        return self.initial_bs_ratio, arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled']

    def constraint_strengthening(self, d, net, ret, v_idx, heuristic):
        '''
        Constraint Strengthening is used to strengthen the cuts.
        The verified cases are used to infer the cuts.
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
        self.original_cut_inference(d['history'], ret['betas'], v_idx)
        if self.enable_constraint_strengthen:
            # infer the cut based on the influence score, retrieve the verified domains and cuts
            d_revise, tmp_cuts = self.cut_inference(d, ret, v_idx, heuristic)
            # reverify the inferred cuts
            ret_revise = net.biccos_verification(d_revise,
                                        stop_criterion_func=stop_criterion_batch_any(d_revise['thresholds']),
                                        multi_spec_keep_func=multi_spec_keep_func_all)
            # extract the still verified index
            lbs_new = ret_revise['lower_bounds'][self.final_name]
            v_idx_new = torch.where(lbs_new > self.decision_thresh)[0]
            # add the verified cuts
            for i in v_idx_new:
                self.tmp_cuts.append(tmp_cuts[i])

    def original_cut_inference(self, d_hists, ret_beta, v_idx):
        '''
        Infer the original cuts based on the verified cases.
        Only used for the first iteration or the enforced usage.
        update:
            self.tmp_cuts: a list of dictionaries
        '''
        for j in range(len(v_idx)):
            relu_activation_decision = []
            relu_activation_coeffs = []
            bias = 0
            for key in d_hists[v_idx[j]].keys():
                d_hists[v_idx[j]][key] = convert_history_from_list(d_hists[v_idx[j]][key])
            for key, (relu_idx, relu_status, _, _, _) in d_hists[v_idx[j]].items():
                # the relu_status is the split status of the neurons
                # 1 for >= 0 and -1 for <= 0
                key_int = self.key_mapping[key]
                for i in range(len(relu_idx)):
                    if ret_beta[v_idx[j]][key][i] > 0:
                        relu_activation_decision.append([key_int, relu_idx[i].item()])
                        relu_activation_coeffs.append(relu_status[i].item())
                        bias += relu_status[i].clamp(min=0).item()
            if len(relu_activation_coeffs) > 0:
                merged = [(d, c) for d, c in zip(relu_activation_decision, relu_activation_coeffs)]
                merged.sort()
                relu_activation_decision = [d for d, _ in merged]
                relu_activation_coeffs = [c for _, c in merged]
                original_cut = self.generate_cut(relu_activation_decision=relu_activation_decision, relu_activation_coeffs=relu_activation_coeffs, b=bias-1)
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
        tmp_cuts = []

        for j in range(len(v_idx)):
            relu_activation_decision = []
            relu_activation_coeffs = []
            bias = 0
            # get the criterion for the j-th split history
            if heuristic == 'neuron_influence_score':
                criterion = self.influence_criterian_get(d['history'][v_idx[j]])
            # ensure that all history contains tensors
            for key in d['history'][v_idx[j]].keys():
                d['history'][v_idx[j]][key] = convert_history_from_list(d['history'][v_idx[j]][key])
            # inference procedure
            for key, (relu_idx, relu_status, relu_bias, relu_score, depths) in d['history'][v_idx[j]].items():
                key_int = self.key_mapping[key]
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
                    if ret['betas'][v_idx[j]][key][i] > 0 or condition:
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
            tmp_cut = self.generate_cut(relu_activation_decision=relu_activation_decision, relu_activation_coeffs=relu_activation_coeffs, b=bias-1, c=-1)
            tmp_cuts.append(tmp_cut)
        return d_revise, tmp_cuts

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
                if isinstance(relu_score, torch.Tensor):
                    hist_score = relu_score.clone()
                else:
                    hist_score = torch.tensor(relu_score).clone()
                # The newly added neuron's score is initialized to 0 and then assigned a value
                # with the newly calculated score (new - old) and given a bonus based on whether
                # it is verified or not
                hist_score[relu_score == 0] = lbs_score[j] + 1e-6
                d_hist[j][key] = (relu_idx, relu_status, relu_bias, hist_score.flatten(), depths)

    def save_biccos_cuts(self, cuts, file_path='../../log/biccos_cuts'):
        '''
        Save the cuts to the file.
        '''
        with open(file_path, 'w') as file:
            for item in cuts:
                file.write(f"{item}\n")

    def load_biccos_cuts(self, file_path = f'../../log/biccos_cuts'):
        '''
        Load the cuts from the file.
        '''
        with open(file_path, 'r') as file:
            cuts = file.readlines()
        return cuts

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

def biccos_verification(self: 'LiRPANet', d, beta=True,
                      fix_interm_bounds=True,
                      stop_criterion_func=None,
                      multi_spec_keep_func=None,
                      iteration=50):
    '''
    Verifying BICCOS cuts in GCP-CROWN.
    '''
    beta_args = arguments.Config['solver']['beta-crown']
    enable_opt_interm_bounds = beta_args['enable_opt_interm_bounds']
    batch = d['upper_bounds'][self.final_name].shape[0]
    decision_thresh = d.get('thresholds', None)

    if beta:
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
    self.set_alpha(d['alphas'], set_all=enable_opt_interm_bounds)

    if beta:
        self.set_crown_bound_opts('beta')
    else:
        self.set_crown_bound_opts('alpha')

    self.net.set_bound_opts({
        'optimize_bound_args': {
            'enable_beta_crown': beta,
            'fix_interm_bounds': fix_interm_bounds,
            'stop_criterion_func': stop_criterion_func,
            'multi_spec_keep_func': multi_spec_keep_func,
            'iteration': iteration,
        },
        'enable_opt_interm_bounds': enable_opt_interm_bounds,
    })

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
        #ret_s = self.get_alpha(device='cpu')
        if beta:
            ret_b = self.get_beta(splits_per_example, device='cpu')
        else:
            ret_b = None

        # Reorganize tensors.
        ret_l, _, _ = self.get_candidate_parallel(lb, ub, device='cpu')
        ret_l[self.final_name] = torch.max(ret_l[self.final_name], lb_last.cpu())

    return {
            'lower_bounds': ret_l, 'betas': ret_b,
        }
