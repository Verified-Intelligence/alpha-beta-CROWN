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
import os
import ast
import copy
from collections import defaultdict

import torch
import arguments
import warnings

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.utils import (
        stop_criterion_placeholder, stop_criterion_all,
        stop_criterion_batch_any, stop_criterion_general,
        reduction_str2func, transfer)

from attack import attack
from input_split.input_split_on_relu_domains import input_branching_decisions
from utils import Timer, take_batch, expand_batch, transfer_obj
from load_model import Customized
from prune import PruneAfterCROWN
from domain_updater import (DomainUpdater, DomainUpdaterSimple)
from heuristics.nonlinear import precompute_A
from domain_clipper import DomainClipper


class LiRPANet:
    def __init__(self, model_ori, in_size, c=None, device=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        general_args = arguments.Config['general']
        solver_args = arguments.Config['solver']
        model_args = arguments.Config['model']
        bab_args = arguments.Config['bab']
        # c is not None only when LiRPANet is initialized from construct_mip_with_model()
        self.c = c
        self.model_ori = model_ori
        self.input_shape = in_size
        self.device = device or general_args['device']
        model_ori_state_dict = copy.deepcopy(model_ori.state_dict())
        bound_opts = {
            'deterministic': general_args['deterministic'],
            'conv_mode': general_args['conv_mode'],
            'sparse_features_alpha': general_args['sparse_alpha'],
            'sparse_spec_alpha': general_args['sparse_alpha'],
            'sparse_intermediate_bounds': general_args['sparse_interm'],
            'batched_crown_max_vram_ratio': solver_args['batched_crown_auto_enlarge_max_vram_ratio'],
            'crown_batch_size': solver_args['crown']['batch_size'],
            'max_crown_size': solver_args['crown']['max_crown_size'],
            'forward_refinement': solver_args['forward']['refine'],
            'forward_max_dim': solver_args['forward']['max_dim'],
            'use_full_conv_alpha': solver_args['alpha-crown']['full_conv_alpha'],
            'disable_optimization': solver_args['alpha-crown']['disable_optimization'],
            'fixed_reducemax_index': True,
            'matmul': {'share_alphas': solver_args['alpha-crown']['matmul_share_alphas']},
            'tanh': {'loose_threshold': bab_args['branching']['nonlinear_split']['loose_tanh_threshold']},
            'activation_bound_option': solver_args['crown']['activation_bound_option'],
            'buffers': {'has_batchdim': general_args['buffer_has_batchdim']},
            'optimize_bound_args': {
                'apply_output_constraints_to': solver_args['invprop']['apply_output_constraints_to'],
                'tighten_input_bounds': solver_args['invprop']['tighten_input_bounds'],
                'best_of_oc_and_no_oc': solver_args['invprop']['best_of_oc_and_no_oc'],
                'directly_optimize': solver_args['invprop']['directly_optimize'],
                'oc_lr': solver_args['invprop']['oc_lr'],
                'share_gammas': solver_args['invprop']['share_gammas'],
            },
            "optimize_graph": {
                "optimizer": eval(model_args['optimize_graph']) if model_args['optimize_graph'] else None,
            },
            "compare_crown_with_ibp": solver_args['crown']['compare_crown_with_ibp'],
            "forward_before_compute_bounds": solver_args['forward_before_compute_bounds'],
        }
        # Update with user-provided bound_opts if specified
        user_bound_opts_str = model_args['bound_opts']
        if user_bound_opts_str:
            try:
                user_bound_opts = ast.literal_eval(user_bound_opts_str)
            except Exception as e:
                raise ValueError(
                    f"Invalid --bound_opts argument: must be a valid Python dictionary literal.\n"
                    f"Error: {e}"
                )
            # Recursively update nested dictionaries
            def update_nested_dict(base, updates):
                for key, value in updates.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        update_nested_dict(base[key], value)
                    else:
                        base[key] = value
            update_nested_dict(bound_opts, user_bound_opts)
        self.net = BoundedModule(
            model_ori, torch.zeros(in_size, device=self.device),
            bound_opts=bound_opts,
            device=self.device
        )
        eval(general_args['graph_optimizer'])(self.net)
        self.root = self.net[self.net.root_names[0]]

        self.net.eval()
        self.return_A = False
        self.needed_A_dict = None
        self.cutter = None # class for generating and optimizing cuts
        self.biccos = None # class for BICCOS
        self.timer = Timer()

        # for fetching cplex in parallel
        self.mip_building_proc = None
        self.processes = None
        self.pool = self.pool_result = self.pool_termination_flag = None # For multi-process.

        # for recording whether we need to return intermediate bounds
        # after initial intermediate bounds fetching, this switch will be
        # aligned with arg.bab.interm_transfer
        self.interm_transfer = True

        self.domain_clipper = None # class for domains clipper

        self.final_name = self.net.final_name
        self.alpha_start_nodes = [self.final_name] + list(filter(
            lambda x: len(x.strip()) > 0,
            bab_args['optimized_interm'].split(',')))

        # If input split is enabled, we do not split nodes, self.nonlinear_split is False
        self.nonlinear_split = (
            not bab_args['branching']['input_split']['enable']
            and bab_args['branching']['method'] == 'nonlinear'
        )

        if arguments.Config['model']['with_jacobian']:
            print('Not checking the conversion correctness for this model with JacobianOP')
        else:
            # check conversion correctness
            dummy = torch.randn(in_size, device=self.device)
            try:
                assert torch.allclose(model_ori(dummy), self.net(dummy), atol=1e-7, rtol=5e-4)
            except AssertionError:
                print('torch allclose failed: norm '
                    f'{torch.norm(model_ori(dummy) - self.net(dummy))}')
        model_ori.load_state_dict(model_ori_state_dict, strict=False)

    @property
    def split_nodes(self):
        return self.net.split_nodes

    @torch.no_grad()
    def get_primal_upper_bound(self, A):
        assert self.x.ptb.norm == torch.inf, (
            'We only support to get primals for Linf norm perturbation so far')
        input_A_lower = A[self.net.output_name[0]][self.net.input_name[0]]['lA']
        batch = input_A_lower.shape[0]

        x_lb, x_ub = self.x.ptb.x_L, self.x.ptb.x_U
        input_primal = x_lb.clone().detach()
        input_primal[input_A_lower.squeeze(1) < 0] = x_ub[input_A_lower.squeeze(1) < 0]

        assert self.c.size(0) == 1
        return input_primal, self.model_ori(input_primal).matmul(self.c[0].transpose(-1, -2))

    # FIXME: should not pass lb and ub into function, they should be from self.net
    def get_interm_bounds(self, lb, ub=None, init=True, device=None):
        """Get the intermediate bounds.

        By default, we also add final layer bound after applying C
        (lb and lb+inf).
        """

        lower_bounds, upper_bounds = {}, {}
        unstable_bounds = {}

        # If input split is enabled, we do not need to get intermediate bounds
        # just return the bounds for the final layer.
        if arguments.Config['bab']['branching']['input_split']['enable']:
            lower_bounds[self.final_name] = lb.detach()
            if ub is None:
                ub = lb + torch.inf
            upper_bounds[self.final_name] = ub.detach()
            # input bab does not need to fetch intermediate bounds.
            return lower_bounds, upper_bounds, None

        if init:
            self.get_split_nodes()
            for layer in self.net.layers_requiring_bounds + self.net.split_nodes:
                if layer.lower is None and layer.upper is None:
                    continue
                lower_bounds[layer.name] = layer.lower.detach()
                upper_bounds[layer.name] = layer.upper.detach()
        elif self.interm_transfer:
            for layer in self.net.layers_requiring_bounds:
                if layer.lower is None and layer.upper is None:
                    continue
                # sometimes the layer may be all stable and
                # removed from the self.net.split_nodes
                # or its next node can not be split at all
                # see self.net.get_split_nodes()
                if layer not in self.net.split_nodes:
                    continue
                mask = self.unstable_mask[layer.name]
                if mask is not None:
                    unstable_bounds[layer.name] = [
                        transfer(layer.lower.detach()[:, mask[0]], device),
                        transfer(layer.upper.detach()[:, mask[0]], device)
                    ]

        # We have to set lower and upper bounds for the final layer here,
        # otherwise, the beta-crown test may fail.
        lower_bounds[self.final_name] = lb.detach()
        print(lower_bounds[self.final_name].shape)
        if ub is None:
            ub = lb + torch.inf
        upper_bounds[self.final_name] = ub.detach()

        return lower_bounds, upper_bounds, unstable_bounds

    def get_mask(self):
        masks = {}
        if arguments.Config['bab']['branching']['input_split']['enable']:
            # input bab does not need to fetch split masks.
            return masks
        for node in self.net.get_splittable_activations():
            mask = []
            for idx in node.requires_input_bounds:
                input_node = node.inputs[idx]
                if not input_node.perturbed or input_node.lower is None and input_node.upper is None:
                    mask.append(None)
                else:
                    mask.append(node.get_split_mask(
                        input_node.lower, input_node.upper, idx))
            masks[node.name] = mask
        return masks

    def get_lA(self, preserve_mask=None, tot_cells=None,
               transpose=True, device=None):
        lAs = {}

        if arguments.Config['bab']['branching']['input_split']['enable']:
            # lA of the input layer is needed for input bab.
            nodes = [self.net[self.net.input_name[0]]]
        else:
            nodes = list(self.net.get_splittable_activations())

        for node in nodes:
            lA = getattr(node, 'lA', None)
            if lA is None:
                continue
            if preserve_mask is not None:
                new_lA = torch.zeros(
                    [tot_cells, lA.shape[0]] + list(lA.shape[2:]),
                    dtype=lA.dtype, device=lA.device)
                new_lA[preserve_mask] = lA.transpose(0,1)
                lA = new_lA
            else:
                lA = lA.transpose(0, 1) if transpose else lA.squeeze(0)
            lAs[node.name] = transfer(lA, device)
        return lAs

    def get_candidate_parallel(self, lb, ub, device=None):
        """Get the intermediate bounds in the current model."""
        return self.get_interm_bounds(lb, ub, init=False, device=device)

    def expand_x_diff_batch(self, x_L, x_U):
        """Create a new BoundedTensor with the new of x_L and x_U."""
        new_data = (x_L + x_U) / 2
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, x_L=x_L, x_U=x_U)
        new_x = BoundedTensor(new_data, ptb)
        return new_x

    def prune_setting(self, d, beta, beta_bias, lb_last, ub_last,
                        batch_mask, enable_opt_interm_bounds):
        """Prune the setting after clipping the input domain."""
        c = d['cs']
        decision_thresh = d['thresholds']
        batch = c.shape[0]
        lb_last, ub_last = lb_last[batch_mask], ub_last[batch_mask]
        if beta:
            splits_per_example = self.set_beta(d, bias=beta_bias)
            self.set_alpha(d, set_all=enable_opt_interm_bounds)
            self.net.set_bound_opts({
                'optimize_bound_args': {
                    'stop_criterion_func': self.domain_clipper.stop_func(decision_thresh),
            }})
            self.set_crown_bound_opts('beta')
        return c, decision_thresh, batch, lb_last, ub_last, splits_per_example

    @torch.no_grad()
    def _expand_tensors(self, d, batch):
        lb, ub = d['lower_bounds'], d['upper_bounds']
        cs, x_Ls, x_Us = d.get('cs', None), d.get('x_Ls', None), d.get('x_Us', None)
        # Only the last element is used later.
        lb_last, ub_last = lb[self.final_name], ub[self.final_name]
        interm_bounds = {k: [lb[k], ub[k]] for k in lb if k != self.final_name}
        # create new_x here since batch may change
        new_x = expand_batch(self.x, batch, x_L=x_Ls, x_U=x_Us)
        if cs is None:
            assert self.c.size(0) == 1
            cs = None if self.c is None else self.c.expand(new_x.shape[0], -1, -1)
        return interm_bounds, lb_last, ub_last, cs, new_x, x_Ls, x_Us

    def update_bounds(self, d, beta=None, fix_interm_bounds=True,
                      shortcut=False, stop_criterion_func=stop_criterion_placeholder(),
                      multi_spec_keep_func=None, beta_bias=True, enable_clip_domains=False):
        """Main function for computing bounds after branch and bound in Beta-CROWN."""
        deterministic_opt = arguments.Config['general']['deterministic_opt']
        solver_args = arguments.Config['solver']
        beta_args = solver_args['beta-crown']
        bab_args = arguments.Config['bab']
        if beta is None:
            # might need to set beta False in FSB node selection
            beta = beta_args['beta']
        vanilla_crown = bab_args['vanilla_crown']
        if vanilla_crown:
            alpha = beta = False
        else:
            alpha = True

        iteration = beta_args['iteration']
        get_upper_bound = bab_args['get_upper_bound']
        enable_opt_interm_bounds = beta_args['enable_opt_interm_bounds']
        branching_input_and_activation = bab_args['branching']['branching_input_and_activation']
        batch = d['upper_bounds'][self.final_name].shape[0]
        decision_thresh = d.get('thresholds', None)

        self.timer.start('func')
        self.timer.start('prepare')

        if self.net.cut_used:
            self.disable_cut_for_branching()
        if beta and not vanilla_crown:
            splits_per_example = self.set_beta(d, bias=beta_bias)
            self.net.cut_used = (
                    arguments.Config['bab']['cut']['enabled']
                    and arguments.Config['bab']['cut']['bab_cut']
                    and getattr(self.net, 'cut_module', None) is not None)
            # even we need to use cut, maybe the cut is not fetched yet
            if self.net.cut_used:
                iteration = self.set_cut_params(
                    batch, batch, d.get('split_history', None))
            # here to handle the case where the split node happen to be in the
            # cut constraint !!!
        ret = self._expand_tensors(d, batch)
        interm_bounds, lb_last, ub_last, c, new_x, x_Ls, x_Us = ret
        new_x_Ls, new_x_Us = None, None

        if alpha:
            self.set_alpha(d['alphas'], set_all=enable_opt_interm_bounds)
        self.timer.add('prepare')

        self.timer.start('bound')
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
        self.set_crown_bound_opts('beta')

        if shortcut:
            args_compute_bounds = dict(x=(new_x,), C=c, reuse_alpha=True,
                interm_bounds=interm_bounds, bound_upper=False,
                decision_thresh=decision_thresh)
            with (torch.enable_grad() if beta else torch.no_grad()):
                lb, _, = self.net.compute_bounds(
                    method='CROWN-optimized' if beta else 'backward',
                    **args_compute_bounds)
            return lb

        # we need A matrix to construct adv example
        temp_return_A = get_upper_bound or branching_input_and_activation
        temp_needed_A_dict = self.needed_A_dict
        if enable_clip_domains and self.domain_clipper.using_final_layer:
            temp_return_A, temp_needed_A_dict = self._set_tmp_A(True, 'alpha-crown')
        original_size = new_x.shape[0]

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

        if vanilla_crown:
            method = 'CROWN'
        else:
            method = 'CROWN-optimized'

        ######### Clip and Verify Domains Start ########
        if enable_clip_domains and self.domain_clipper is not None:
            if self.domain_clipper.clip_input_domain:
                ret_clipper = self.domain_clipper.domain_clip_ReLU(d, new_x,
                                                        interm_bounds)
                new_x_Ls, new_x_Us, interm_bounds, d, batch_mask = ret_clipper
                new_x = self.expand_x_diff_batch(new_x_Ls, new_x_Us)
                if self.domain_clipper.prune and batch_mask is not None:
                    ret_prune = self.prune_setting(d, beta, beta_bias, lb_last, ub_last,
                                                batch_mask, enable_opt_interm_bounds)
                    (c, decision_thresh, batch,
                        lb_last, ub_last, splits_per_example) = ret_prune

            if self.domain_clipper.clip_interm_domain:
                interm_bounds = self.domain_clipper.optimize_interm_bounds(
                    d, new_x.ptb.x_L, new_x.ptb.x_U, interm_bounds,
                    self.split_activations)
        ######### Clip and Verify Domains End ##########

        tmp_ret = self.net.compute_bounds(
            x=(new_x,), C=c, method=method,
            interm_bounds=interm_bounds, reference_bounds=reference_bounds,
            return_A=temp_return_A, needed_A_dict=temp_needed_A_dict,
            cutter=self.cutter, bound_upper=False,
            decision_thresh=decision_thresh)

        self.timer.add('bound')

        A = tmp_ret[2] if temp_return_A else None
        lb, _ = tmp_ret[0], tmp_ret[1]

        # Using output constraints to clip input region.
        # TODO: clean up implementation, and make it more general.
        if enable_clip_domains and self.domain_clipper.using_final_layer:
            new_x_Ls, new_x_Us, interm_bounds = self.domain_clipper.domain_clip_outputs(A, new_x, interm_bounds)

        if get_upper_bound:
            primal_x, ub = self.get_primal_upper_bound(A)
        else:
            ub = torch.full_like(lb, fill_value=torch.inf)  # dummy upper bound
            primal_x = None
        # Use A matrix of the input, the find best neuron to branch in input space.

        input_split_idx = input_branching_decisions(
            self.net, lb,
            A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
            x_Ls, x_Us, decision_thresh
        ) if branching_input_and_activation else None

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            self.timer.start('transfer')
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            lAs = self.get_lA(
                self.net.last_update_preserve_mask, original_size,
                device='cpu', transpose=True)
            self.timer.add('transfer')
            self.timer.start('finalize')
            if alpha:
                ret_s = self.get_alpha(device='cpu', half=not deterministic_opt)
            else:
                ret_s = {}
            if beta:
                ret_b = self.get_beta(splits_per_example, device='cpu')
            else:
                ret_b = [{} for _ in range(batch)]
            # Reorganize tensors.
            ret_l, ret_u, unstable_bounds = self.get_candidate_parallel(lb, ub, device='cpu')
            if not deterministic_opt:
                ret_l[self.final_name] = torch.max(
                    ret_l[self.final_name], lb_last.cpu())
                if not get_upper_bound:
                    # Do not set to min so the primal is always corresponding
                    # to the upper bound.
                    ret_u[self.final_name] = torch.min(
                        ret_u[self.final_name], ub_last.cpu())
            self.timer.add('finalize')

        # Each key is corresponding to a pre-relu layer, and each value intermediate
        # beta values for neurons in that layer.
        new_split_history = [{} for _ in range(batch)]
        if self.net.cut_used:
            self.set_cut_new_split_history(new_split_history, batch)

        self.timer.add('func')
        self.timer.print()

        # Each key is corresponding to a pre-relu layer, and each value
        # intermediate beta values for neurons in that layer.,
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch)]

        lbs_final = ret_l[self.final_name]
        verified_elements = lbs_final > decision_thresh.to('cpu')
        v_idx = torch.where(torch.any(verified_elements, dim=1))[0]
        print("max lb", lbs_final.max(), "min lb", lbs_final.min())
        print(f'Number of Verified Splits: {len(v_idx)} of {len(lbs_final)}')

        return {
            'lower_bounds': ret_l, 'upper_bounds': ret_u,
            'lAs': lAs, 'alphas': ret_s,
            'betas': ret_b, 'split_history': new_split_history,
            'intermediate_betas': best_intermediate_betas,
            'unstable_bounds': unstable_bounds,
            'primals': primal_x,
            'c': c, 'x_Ls': x_Ls if new_x_Ls is None else new_x_Ls,
            'x_Us': x_Us if new_x_Us is None else new_x_Us,
            'input_split_idx': input_split_idx
        }

    def get_split_nodes(self, verbose=False):
        if arguments.Config['bab']['branching']['nonlinear_split']['relu_only']:
            for node in self.net.nodes():
                if node.splittable and not isinstance(node, BoundRelu):
                    node.splittable = False
                    node.force_not_splittable = True
        self.net.get_split_nodes()
        self.split_activations = self.net.split_activations
        if verbose:
            print('Split layers:')
            for layer in self.net.split_nodes:
                print(f'  {layer}: {self.split_activations[layer.name]}')
            print('Nonlinear functions:')
            for node in self.net.nodes():
                if node.perturbed and len(node.requires_input_bounds):
                    print('  ', node)
        if not self.nonlinear_split:
            for node in self.net.nodes():
                if node.splittable and not isinstance(node, BoundRelu):
                    self.nonlinear_split = True

    def build(self, x, c, rhs,
              stop_criterion,
              bounding_method=None, vnnlib_handler=None,
              interm_bounds=None, return_A=False,
              or_spec_size=None,
              full_alpha_info=False,
              ):
        # TODO merge with build_with_refined_bounds()
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        branching_args = bab_args['branching']
        share_alphas = solver_args['alpha-crown']['share_alphas']
        bounding_method = bounding_method or solver_args['bound_prop_method']
        batch_size_target = solver_args['build_batch_size']
        branching_input_and_activation = branching_args['branching_input_and_activation']
        enable_clip_domains = (bab_args["clip_n_verify"]["clip_interm_domain"]['enabled'])
        clip_in_alpha_crown = (
            bab_args["clip_n_verify"]["clip_interm_domain"]["clip_in_alpha_crown"])
        enable_opt_interm_bounds = arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
        enable_input_split = branching_args["input_split"]['enable']
        need_alphas = bounding_method in ['alpha-crown', 'alpha-forward']
        # get_all_alphas: we need both intermediate and final alphas
        get_all_alphas = enable_opt_interm_bounds or enable_input_split
        device = self.device
        # When full_alpha_info is True, we get all items related to alpha,
        #   including alpha values, some indices (like alpha_lookup_idx if sparse alpha is enabled)...
        #   the alpha dict is in the form of
        #       {node_name: {'alpha': {start_node_name: alpha_value},
        #                    'alpha_lookup_idx': {start_node_name: alpha_lookup_idx}, ...}}
        # When full_alpha_info is False, we only get alpha values.
        #   the alpha dict is in the form of
        #       {node_name: {start_node_name: alpha_value}}
        #   Compared with the dict if full_alpha_info is True, we reduce one level of nesting.
        batch_handler = BatchHandler(
            x, c, rhs, stop_criterion, or_spec_size,
            interm_bounds, batch_size_target, self.final_name,
            need_alphas, full_alpha_info, None, clip_in_alpha_crown
        )

        self.net.set_bound_opts({'verbosity': 1})
        self.set_crown_bound_opts('alpha')
        # If input split is enabled, we do not need to get split nodes.
        if not branching_args['input_split']['enable']:
            self.get_split_nodes(verbose=True)

        self._set_A_options(return_A=return_A or enable_clip_domains or clip_in_alpha_crown)
        if clip_in_alpha_crown:
            self.net.set_bound_opts({'clip_in_alpha_crown': True})

        total_batches = batch_handler.total_batches
        for now_batch in range(total_batches):
            print(f'build batch [{now_batch+1}/{total_batches}]')
            torch.cuda.empty_cache()

            # Get batch input.
            batch_x, batch_c, batch_rhs, stop_criterion_func, batch_interm_bounds, _, batch_or_spec_size = (
                batch_handler.get_batch_input(now_batch, device)
            )
            self.net.set_bound_opts({'optimize_bound_args': {'stop_criterion_func': stop_criterion_func}})
            self.x = batch_x
            self.c = batch_c

            # ---------- Compute bounds with different methods START ----------
            prune_after_crown = None
            if bounding_method == 'alpha-crown':
                # NOTE: There are some fast paths for alpha-crown.
                # 1. If we have only one batch, and we can verify it, we can return directly.
                # 2. If we enable attack with middle order and the attack is successful, we can return directly.
                # Besides, we can prune the specifications verified by initial CROWN bounds
                # to save the computational cost of alpha-crown.

                # first get CROWN bounds
                # Reference bounds are intermediate layer bounds from initial CROWN bounds.
                lb, ub, aux_reference_bounds = self.net.init_alpha(
                    (batch_x,), share_alphas=share_alphas, c=batch_c, bound_upper=False)
                print('initial CROWN bounds (first 10 items):', lb.flatten()[:10].tolist())

                if arguments.Config['general']['save_output']:
                    assert total_batches == 1
                    arguments.Globals['out']['init_crown_bounds'] = lb.cpu()

                if bab_args['cut']['enabled']:
                    assert total_batches == 1
                    self.enable_cuts()

                if stop_criterion_func(lb).all().item():
                    # Fast path. Initial CROWN bound can verify the network.
                    print('Verified with initial CROWN!')
                    ret = (lb, None)
                    if total_batches == 1:
                        # If we only have one batch, we can return the result directly.
                        return lb, {}
                else:
                    # Prune the specifications that can be verified by initial CROWN bounds.
                    if solver_args['prune_after_crown']:
                        prune_after_crown = PruneAfterCROWN(
                            self.net, batch_x, batch_c, batch_rhs, lb,
                            aux_reference_bounds,
                            stop_criterion,
                            or_spec_size=batch_or_spec_size)
                        # Update batched data after pruning.
                        batch_x, batch_c, batch_rhs, batch_or_spec_size, stop_criterion_func = prune_after_crown.get_pruned_data()
                        self.x = batch_x
                        self.c = batch_c
                        self.net.set_bound_opts({
                            'optimize_bound_args': {'stop_criterion_func': stop_criterion_func},
                        })
                    if arguments.Config['attack']['pgd_order'] == 'middle' and vnnlib_handler is not None:                        
                        _, verified_success, attack_examples, _, _ = attack(
                            self.model_ori, batch_x, batch_c, batch_rhs, batch_or_spec_size,
                            vnnlib_handler.vnnlib,
                            verified_status="unknown", verified_success=False)

                        if verified_success:
                            print("pgd attack succeed in middle order")
                            return None, {'attack_examples': attack_examples}

                    if enable_clip_domains or clip_in_alpha_crown:
                        print('Using alpha-CROWN with output constraints to initialize bounds.')
                    else:
                        print('Using alpha-CROWN to initialize bounds.')
                    ret = self.net.compute_bounds(
                        x=(batch_x,), C=batch_c, method='CROWN-Optimized',
                        return_A=self.return_A, needed_A_dict=self.needed_A_dict,
                        bound_upper=False, aux_reference_bounds=aux_reference_bounds,
                        cutter=self.cutter, interm_bounds=batch_interm_bounds,
                        decision_thresh=rhs)
            elif bounding_method == 'alpha-forward':
                warnings.warn('alpha-forward can only be used with input split for now')
                self.net.bound_opts['optimize_bound_args']['init_alpha'] = True
                ret = self.net.compute_bounds(
                    x=(batch_x,), C=batch_c, method='forward-optimized', bound_upper=False)
            else:
                with torch.no_grad():
                    if bounding_method == 'init-crown':
                        assert not self.return_A
                        lb, ub, _ = self.net.init_alpha(
                            (batch_x,), share_alphas=share_alphas, c=batch_c,
                            bound_upper=False
                            )
                        ret = lb, ub
                    else:
                        ret = self.net.compute_bounds(
                            x=(batch_x,), C=batch_c, method=bounding_method,
                            bound_upper=False, return_A=self.return_A,
                            needed_A_dict=self.needed_A_dict)
            # ----------- Compute bounds with different methods END -----------

            lb, ub = ret[0], ret[1]
            A = ret[-1]

            if branching_input_and_activation:
                assert total_batches == 1
                # Use A matrix of the input, the find best neuron to branch in input space.
                batch_input_split_idx = input_branching_decisions(
                    self.net, lb,
                    A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
                    self.x.ptb.x_L, self.x.ptb.x_U, rhs)
            else:
                batch_input_split_idx = {}

            print(f'initial {bounding_method} bounds (first 10 items):', lb.flatten()[:10].tolist())
            global_lb = lb.min().item()
            print(f'Global lower bound: {global_lb}')

            # DEBUG: check loose bounds
            if os.environ.get('ABCROWN_VIEW_INTERM', False):
                print('Intermediate bounds after initial alpha-CROWN:')
                self._print_interm_bounds()
                import pdb; pdb.set_trace()

            # Get batch results.
            if need_alphas:
                batch_alpha = self.get_alpha(get_all=get_all_alphas,
                    half=arguments.Config["solver"]["alpha-crown"]["alpha_dtype"] == "float16",
                    full_info=full_alpha_info, drop_unused=True
                )
            else:
                batch_alpha = None
            batch_lb, batch_ub, _ = self.get_interm_bounds(lb)  # primals are better upper bounds
            batch_mask = self.get_mask()
            batch_lA = self.get_lA()
            batch_mask = self.get_mask()
            if prune_after_crown:
                prune_after_crown.recover(batch_lb, batch_ub, batch_lA, batch_alpha, 
                                        batch_mask, batch_input_split_idx, full_alpha_info)

            # save initial alpha-crown for tests
            if arguments.Config['general']['save_output'] and bounding_method == 'alpha-crown':
                assert total_batches == 1
                arguments.Globals['out']['init_alpha_crown'] = batch_lb[self.final_name].cpu()


            batch_handler.add_batch_result(batch_lb, batch_ub, batch_lA, batch_alpha,
                batch_mask, batch_input_split_idx, A)

        if self.nonlinear_split and global_lb < 0 and self.return_A:
            assert total_batches == 1
            precompute_A(self.net, A, batch_x,
                         interm_bounds={k: (batch_lb[k], batch_ub[k]) for k in batch_lb})
        if bab_args['cut']['enabled']:
            assert total_batches == 1
            self.create_cutter(A, batch_x)

        # FIXME There is not only A here. There are also biases.
        # Need to rename.
        self.A_saved = A

        ####### Clip and Verify Domains Initialization #######
        if enable_clip_domains:
            self.domain_clipper = DomainClipper(batch_handler.batch_A[-1],
                    batch_x, self.final_name, self.net.input_name,
                    lb, ub, batch_mask)
        ###### Clip and Verify Domains Initialization END #######

        result = batch_handler.get_results(self.empty_history())

        return result['global_lb'], result

    def build_with_refined_bounds(
            self, x, c, rhs,
            stop_criterion,
            refined_lower_bounds, refined_upper_bounds,
            reference_lA,
            reference_alphas, refined_betas,
            or_spec_size=None,
            full_alpha_info=False
            ):
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        branch_args = bab_args['branching']
        share_alphas = solver_args['alpha-crown']['share_alphas']
        batch_size_target = solver_args['build_batch_size']
        branching_input_and_activation = branch_args['branching_input_and_activation']
        vanilla_crown = bab_args['vanilla_crown']
        enable_opt_interm_bounds = arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
        need_alphas = not vanilla_crown
        device = self.device
        assert refined_lower_bounds is not None and refined_upper_bounds is not None
        interm_bounds = {k: [refined_lower_bounds[k], refined_upper_bounds[k]]
                         for k in refined_lower_bounds if k != self.final_name}

        batch_handler = BatchHandler(
            x, c, rhs, stop_criterion, or_spec_size,
            interm_bounds, batch_size_target, self.final_name, 
            need_alphas, full_alpha_info, reference_alphas, reference_lA,
        )

        # FIXME: BaB-Attack requires refined_lower_bounds and refined_upper_bounds
        # Here we save them together into interm_bounds.
        # We need to update BaB-Attack in the future.
        self.interm_bounds = batch_handler.interm_bounds

        # reset the A options since they are set in the previous build() and then set it again
        self.needed_A_dict = None
        self.return_A = False
        self._set_A_options(bab=True)

        total_batches = batch_handler.total_batches
        # We assume that this function is only called in activation bab with not large instances.

        for now_batch in range(total_batches):
            print(f'build_with_refined_bounds batch [{now_batch+1}/{total_batches}]')
            torch.cuda.empty_cache()

            # Get batch input.
            batch_x, batch_c, batch_rhs, stop_criterion_func, batch_interm_bounds, batch_alphas, _ = (
                batch_handler.get_batch_input(now_batch, device)
            )
            self.net.set_bound_opts({'optimize_bound_args': {'stop_criterion_func': stop_criterion_func}})
            self.x = batch_x
            self.c = batch_c

            # ---------- Compute bounds with different methods START ----------
            skip_backward_pass = False
            if vanilla_crown:
                ret = self.net.compute_bounds(
                    x=(batch_x,), method='backward', C=batch_c,
                    return_A=self.return_A, #reuse_alpha=True,
                    interm_bounds=batch_interm_bounds,
                    needed_A_dict=self.needed_A_dict)
            else:
                self.net.init_alpha(
                    (batch_x,), share_alphas=share_alphas, c=batch_c,
                    interm_bounds=batch_interm_bounds,
                    reference_alphas=batch_alphas,
                    skip_bound_compute=True)

                self.set_crown_bound_opts('alpha')

                if solver_args['skip_with_refined_bound']:
                    print('all alpha initialized')
                    if not self.return_A:
                        # FIXME "A" is incorrect later when calling get_lA
                        skip_backward_pass = True
                        print('directly get lb and ub from refined bounds')
                        # Make sure the shape of reference_lA looks good so that we
                        # can recover the batch_lA
                        print('c shape:', batch_c.shape)
                        assert reference_lA is not None
                        batch_reference_lA = {k: batch_handler.take_batch(A, now_batch)
                                              for k, A in reference_lA.items()}
                        print('lA shapes:', [A.shape for A in batch_reference_lA.values()])
                        # A shape: [batch, num_output, *output_shape]
                        assert all([A.shape[0] == batch_c.shape[0] for A in batch_reference_lA.values()])
                        # Try to directly recover l and u from refined_lower_bounds
                        # and refined_upper_bounds without a backward crown pass
                        lb = batch_handler.take_batch(refined_lower_bounds[self.final_name], now_batch)
                        ub = batch_handler.take_batch(refined_upper_bounds[self.final_name], now_batch)
                        ret = (lb, ub)
                        # restore bounds back to the model only for all_node_split_LP
                        if solver_args['beta-crown']['all_node_split_LP']:
                            for node in batch_interm_bounds:
                                self.net[node].lower = batch_interm_bounds[node][0]
                                self.net[node].upper = batch_interm_bounds[node][1]
                            self.net[self.final_name].lower = lb
                            self.net[self.final_name].upper = ub
                    else:
                        # do a backward crown pass
                        print('true A is required, we do a full backward CROWN pass to obtain it')
                        ret = self.net.compute_bounds(
                            x=(batch_x,), method='backward', C=batch_c,
                            return_A=self.return_A, reuse_alpha=True,
                            interm_bounds=batch_interm_bounds,
                            needed_A_dict=self.needed_A_dict)
                else:
                    print('Restore to original setting since some alphas are not '
                        'initialized yet or being asked not to skip')
                    ret = self.net.compute_bounds(
                        x=(batch_x,), method='crown-optimized',
                        return_A=self.return_A, C=batch_c,
                        interm_bounds=batch_interm_bounds,
                        needed_A_dict=self.needed_A_dict)
            # ----------- Compute bounds with different methods END -----------

            lb, ub = ret[0], ret[1]
            A = ret[-1]

            if branching_input_and_activation:
                assert total_batches == 1
                # Use A matrix of the input, the find best neuron to branch in input space.
                batch_input_split_idx = input_branching_decisions(
                    self.net, lb,
                    A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
                    batch_x.ptb.x_L, batch_x.ptb.x_U, rhs)
            else:
                batch_input_split_idx = {}

            batch_lb, batch_ub, _ = self.get_interm_bounds(lb)  # primals are better upper bounds

            print('(alpha-)CROWN with fixed intermediate bounds:', lb, ub)
            print('Intermediate layers:', ','.join(list(batch_interm_bounds.keys())))
            if vanilla_crown:
                batch_alpha = None
            else:
                if arguments.Config['bab']['attack']['enabled']:
                    assert total_batches == 1
                    # Save all alphas, which will be further refined in bab-attack.
                    self.refined_alpha = reference_alphas
                batch_alpha = self.get_alpha(get_all=enable_opt_interm_bounds,
                        half=arguments.Config["solver"]["alpha-crown"]["alpha_dtype"] == "float16",
                        full_info=full_alpha_info, drop_unused=True
                    )

            batch_mask = self.get_mask()
            if skip_backward_pass:
                # If we skip the backward pass, we use the reference lA.
                batch_lA = None
            else:
                batch_lA = self.get_lA()

            batch_handler.add_batch_result(batch_lb, batch_ub, batch_lA, batch_alpha, batch_mask, batch_input_split_idx, A)
        if vanilla_crown:
            history = ret_b = None
        elif refined_betas is not None:
            assert total_batches == 1
            # only has batch size 1 for refined betas
            assert len(refined_betas[0]) == 1
            history, ret_b = refined_betas[0][0], refined_betas[1][0]
        else:
            history, ret_b = self.empty_history(), None

        result = batch_handler.get_results(history, ret_b)

        return result

    def build_history_and_set_bounds(self, d, split, mode='depth'):
        _, num_split = DomainUpdater.get_num_domain_and_split(
            d, split, self.final_name)
        args = (self.root, self.final_name, self.net.split_nodes)
        if num_split == 1 and (split.get('points', None) is None
                                 or split['points'].ndim == 1):
            domain_updater = DomainUpdaterSimple(*args)
        else:
            domain_updater = DomainUpdater(*args)

        domain_updater.set_branched_bounds(d, split, mode)

    def _set_A_options(self, bab=False, return_A=False):
        branching_args = arguments.Config['bab']['branching']
        input_and_act = branching_args['branching_input_and_activation']
        get_upper_bound = bab and arguments.Config['bab']['get_upper_bound']
        if get_upper_bound or input_and_act:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(
                self.net.input_name[0])
        if self.nonlinear_split or return_A:
            self.needed_A_dict = defaultdict(set)
            for node in self.net.nodes():
                if node != self.net.final_name:
                    self.needed_A_dict[node.name].add(self.net.input_name[0])
        # FIXME just use "self.needed_A_dict is not None" without the extra "self.return_A"
        if self.needed_A_dict is not None:
            self.return_A = True
        else:
            self.return_A = False

    def _set_tmp_A(self, enable_clip_domains, bounding_method):
        temp_return_A = self.return_A
        temp_needed_A_dict = defaultdict(set)
        if enable_clip_domains and bounding_method != 'alpha-forward':
            if not self.return_A:
                # clip domains needs lA and lbias
                temp_needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
                temp_return_A = True
            else:
                # if A_dict was already required, also get lA and lbias of the whole network
                temp_needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
                temp_needed_A_dict.update(self.needed_A_dict)
        return temp_return_A, temp_needed_A_dict

    def empty_history(self):
        '''
        history: a tuple of tensors
            history[0]: relu_idx
            history[1]: relu_status
            history[2]: relu_bias
            history[3]: relu_score
            history[4]: depths
        '''
        if arguments.Config['bab']['branching']['input_split']['enable']:
            # For input split, we do not need to track the history.
            return None
        return {layer.name: ([], [], [], [], []) for layer in self.net.split_nodes}

    def set_crown_bound_opts(self, crown_name):
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        crown_args = solver_args[f'{crown_name}-crown']
        opt_bound_args = {
            'deterministic': arguments.Config['general']['deterministic_opt'],
            'lr_alpha': crown_args['lr_alpha'],
            'iteration': crown_args['iteration'],
            'lr_decay': crown_args['lr_decay'],
            'use_float64_in_last_iteration': solver_args['use_float64_in_last_iteration'],
            'start_save_best': solver_args['start_save_best'],
            'loss_reduction_func': reduction_str2func(
                arguments.Config['general']['loss_reduction_func']),
        }
        if crown_name == 'alpha':
            opt_bound_args.update({
                'enable_alpha_crown': True, 'enable_beta_crown': False,
                'init_alpha': False, 'fix_interm_bounds': True,
                'use_shared_alpha': crown_args['share_alphas'],
                'early_stop_patience': solver_args['early_stop_patience'],
                'pruning_in_iteration': False,
                'max_time': crown_args['max_time'] * bab_args['timeout'],
            })
        elif crown_name == 'beta':
            opt_bound_args.update({
                'optimizer': crown_args['optimizer'],
                'lr_beta': crown_args['lr_beta'],
                'pruning_in_iteration': bab_args['pruning_in_iteration'],
                'pruning_in_iteration_threshold': bab_args['pruning_in_iteration_ratio'],
                'lr_cut_beta': bab_args['cut']['lr_beta'],
                'apply_output_constraints_to': [],
                'tighten_input_bounds': False,
                'directly_optimize': [],
                'share_gammas': False,
            })
        self.net.set_bound_opts({'optimize_bound_args': opt_bound_args})

    def _print_interm_bounds(self, lb=None, ub=None):
        if lb is not None:
            for k in lb:
                print(self.net[k])
                for out in self.net[k].output_name:
                    print('  ', self.net[out])
                print('  lower:', lb[k])
                if ub is not None and k in ub:
                    print('  upper:', ub[k])
                unstable = torch.logical_and(lb[k] < 0, ub[k] > 0).float()
                unstable = unstable.reshape(unstable.size(0), -1).sum(dim=-1)
                print('  unstable:', unstable, unstable.mean(), unstable.max())
        else:
            for node in self.net.nodes():
                if not node.perturbed:
                    continue
                if isinstance(node.lower, torch.Tensor):
                    print(node)
                    print('  lower:', node.lower.reshape(-1)[:10])
                    if isinstance(node.upper, torch.Tensor):
                        print('  upper:', node.upper.reshape(-1)[:10])
                        print(' Average gap:', (node.upper-node.lower).mean())

    from alpha import drop_unused_alpha, get_alpha, set_alpha
    from beta import get_beta, set_beta, reset_beta
    from lp_mip_solver import (
    build_solver_model, update_mip_model_fix_relu,
    build_the_model_mip_refine, build_the_model_mip_or,
    build_the_model_mip_and, all_node_split_LP, check_lp_cut, update_the_model_cut)
    from input_split.bounding import get_lower_bound_naive
    from cuts.cut_verification import (
        enable_cuts, create_cutter, set_cuts, create_mip_building_proc,
        set_cut_params, set_cut_new_split_history,
        disable_cut_for_branching)
    from cuts.infered_cuts import biccos_verification

class BatchHandler:
    """
    A helper class to handle batching of inputs and results in build() of LiRPANet.
    """
    def __init__(self, x: BoundedTensor, c, rhs, stop_criterion, or_spec_size, interm_bounds, 
                 batch_size_target, final_name, need_alphas, full_alpha_info,
                 reference_alphas=None, reference_lA=None, clip_in_alpha_crown=False):

        batch_size_ori = c.shape[0]
        same_x_range = (x[0:1] == x).all().item()
        # the batch size of x should be the same as c and rhs or 1 when the input range is the same.
        assert same_x_range or (x.shape[0] == batch_size_ori)

        if arguments.Config['general']['store_all_specs_on_cpu']:
            self.device = "cpu"
        else:
            self.device = arguments.Config['general']['device']

        self.x = x
        self.c = c
        self.rhs = rhs
        self.stop_criterion = stop_criterion
        self.or_spec_size = or_spec_size
        self.need_alphas = need_alphas
        self.full_alpha_info = full_alpha_info
        self.interm_bounds = interm_bounds
        self.batch_size_ori = batch_size_ori
        self.batch_size_target = batch_size_target
        self.final_name = final_name
        self.total_batches = (batch_size_ori + batch_size_target - 1) // batch_size_target
        self.reference_alphas = reference_alphas
        self.reference_lA = reference_lA

        # stop_criterion_general and stop_criterion_all are only used when
        # different OR specs are optimized together,
        # i.e., the original batch size should be exactly 1.
        if stop_criterion is stop_criterion_general or stop_criterion is stop_criterion_all:
            self.optimize_disjuncts_separately = False
            assert batch_size_ori == 1
        elif stop_criterion is stop_criterion_batch_any:
            self.optimize_disjuncts_separately = True

        if clip_in_alpha_crown:
            self.optimize_disjuncts_separately = True

        # when the x range is not the same, the alpha can be different for different x or batches.
        # example: BoundRelu. alpha shape, alpha_lookup_idx and alpha_indices can be different,
        # when sparse alpha is enabled, making merging batches infeasible.
        # so we only support one batch when the x range is not the same.
        if need_alphas:
            assert same_x_range or self.total_batches == 1

        self.batch_lb, self.batch_ub = [], []
        self.batch_lA, self.batch_alpha = [], []
        self.batch_mask, self.batch_input_split_idx = [], []
        self.batch_A = []

    def take_batch(self, data, batch_idx, device=None, batch_dim=0):
        return take_batch(data, self.batch_size_target, batch_idx, device, batch_dim)

    def get_batch_input(self, batch_idx, device):
        batch_x = self.take_batch(self.x, batch_idx, device=device)
        batch_c = self.take_batch(self.c, batch_idx, device)
        batch_rhs = self.take_batch(self.rhs, batch_idx, device)
        actual_batch_size = batch_c.shape[0]


        # update decision_thresh used in stop_criterion_func to the current batch
        stop_criterion = self.stop_criterion
        batch_or_spec_size = None
        if self.or_spec_size is not None:
            if self.optimize_disjuncts_separately:
                batch_or_spec_size = self.take_batch(self.or_spec_size, batch_idx, device)
            else:
                batch_or_spec_size = self.or_spec_size
        if stop_criterion is stop_criterion_general:
            assert self.or_spec_size is not None
            batch_or_spec_size = self.or_spec_size
            stop_criterion_func = stop_criterion(batch_or_spec_size, batch_rhs)
        elif (stop_criterion is stop_criterion_batch_any) or (stop_criterion is stop_criterion_all):
            stop_criterion_func = stop_criterion(batch_rhs)
        else:
            raise ValueError(f'Unknown stop criterion function: {self.stop_criterion_name}')

        batch_interm_bounds = None
        if self.interm_bounds is not None:
            batch_interm_bounds = {}
            for k in self.interm_bounds:
                if k != self.final_name:
                    interm_bound_batch_size = self.interm_bounds[k][0].shape[0]
                    if interm_bound_batch_size == 1:
                        # if the interm bounds are shared before, we expand them to the batch size.
                        batch_interm_bounds[k] = [
                            expand_batch(self.interm_bounds[k][0], actual_batch_size, device),
                            expand_batch(self.interm_bounds[k][1], actual_batch_size, device)
                        ]
                    else:
                        # otherwise, the batch size of the interm bounds should be the same as the batch size of c and rhs,
                        # and we should take the batch size from the interm bounds.
                        assert interm_bound_batch_size == self.batch_size_ori
                        batch_interm_bounds[k] = [
                            self.take_batch(self.interm_bounds[k][0], batch_idx, device),
                            self.take_batch(self.interm_bounds[k][1], batch_idx, device)
                        ]

        batch_alphas = None
        if self.reference_alphas is not None:
            batch_alphas = {}
            # since we assume when alphas are needed,
            # the x range is the same, or the total batches is 1,
            # here we should take batch alpha values from the reference alphas
            # and just copy other items like alpha_lookup_idx, alpha_indices... if exist.
            for node_name in self.reference_alphas:
                batch_alphas[node_name] = {}
                for kk in self.reference_alphas[node_name]:
                    if kk == 'alpha':
                        # alpha values are different for different batches.
                        batch_alphas[node_name][kk] = {}
                        for start_node_name, alpha_value in self.reference_alphas[node_name][kk].items():
                            # [alpha_size, prod(start_node_shape), batch_size, *node_shape]
                            alpha_batch_dim = 2
                            alpha_batch_size = alpha_value.shape[alpha_batch_dim]
                            if alpha_batch_size == 1:
                                # if the alpha is shared before, we expand it to the batch size.
                                batch_alphas[node_name][kk][start_node_name] = expand_batch(
                                    alpha_value, actual_batch_size, device, alpha_batch_dim
                                )
                            else:
                                # otherwise, the batch size of the alpha should be the same as the batch size of c and rhs,
                                # and we should take the batch size from the alpha.
                                assert alpha_batch_size == self.batch_size_ori
                                batch_alphas[node_name][kk][start_node_name] = self.take_batch(
                                    alpha_value, batch_idx, device, alpha_batch_dim
                                )
                    else:
                        # other items are the same for all batches.
                        batch_alphas[node_name][kk] = self.reference_alphas[node_name][kk]

        # rhs will be used to prune the verified or specs.
        self.curr_rhs = batch_rhs

        return batch_x, batch_c, batch_rhs, stop_criterion_func, batch_interm_bounds, batch_alphas, batch_or_spec_size

    def add_batch_result(self, lb, ub, lA, alphas, mask, input_split_idx, A):
        device = self.device
        self.batch_lb.append(transfer_obj(lb, device))
        self.batch_ub.append(transfer_obj(ub, device))
        self.batch_lA.append(transfer_obj(lA, device))
        self.batch_alpha.append(transfer_obj(alphas, device))
        self.batch_mask.append(transfer_obj(mask, device))
        self.batch_input_split_idx.append(transfer_obj(input_split_idx, device))
        if A is not None:
            self.batch_A.append(transfer_obj(A, device))

    def get_results(self, history, ret_b=None):
        lb = {k: torch.cat([item_lb[k] for item_lb in self.batch_lb])
              for k in self.batch_lb[0]}
        ub = {k: torch.cat([item_ub[k] for item_ub in self.batch_ub])
              for k in self.batch_ub[0]}
        lA = None
        if self.batch_lA[0] is not None:
            lA = {k: torch.cat([item_lA[k] for item_lA in self.batch_lA])
                for k in self.batch_lA[0]}

        if not self.need_alphas:
            alphas = None
        elif self.full_alpha_info:
            alphas = {}
            for k in self.batch_alpha[0]:
                alphas[k] = {}
                alphas[k]['alpha'] = {}
                for kk in self.batch_alpha[0][k]['alpha']:
                    alphas[k]['alpha'][kk] = torch.cat([item_alpha[k]['alpha'][kk] for item_alpha in self.batch_alpha], dim=2)
                for opt_key in self.batch_alpha[0][k]:
                    if opt_key != 'alpha':
                        # NOTE: assume all other keys are the same for all batches
                        # FIXME: assertion needed
                        alphas[k][opt_key] = self.batch_alpha[0][k][opt_key]
        else:
            alphas = {k: {kk: torch.cat([item_alpha[k][kk] for item_alpha in self.batch_alpha], dim=2)
                    for kk in self.batch_alpha[0][k]} for k in self.batch_alpha[0]}

        mask = {}
        for k in self.batch_mask[0]:
            for i in range(len(self.batch_mask[0][k])):
                if self.batch_mask[0][k][i] is None:
                    assert all(item_mask[k][i] is None for item_mask in self.batch_mask)
                    mask[k] = [None] * len(self.batch_mask[0][k])
                else:
                    mask[k] = [torch.cat([item_mask[k][i] for item_mask in self.batch_mask])
                            for i in range(len(self.batch_mask[0][k]))]

        input_split_idx = {k: torch.cat([item_input_split_idx[k] for item_input_split_idx in self.batch_input_split_idx])
                for k in self.batch_input_split_idx[0]}

        ret = {
            'mask': mask, 'lA': lA, 'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': alphas, 'history': history,
            'input_split_idx': input_split_idx,
            'global_lb': lb[self.final_name],
            'global_ub': ub[self.final_name],
        }
        if ret['lA'] is None:
            ret['lA'] = self.reference_lA

        ret['betas'] = ret_b

        return ret
