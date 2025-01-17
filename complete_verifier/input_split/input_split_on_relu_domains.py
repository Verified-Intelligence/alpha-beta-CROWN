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
"""Handle input space split on ReLU split domains."""
import time
from collections import defaultdict
import torch
import arguments

from auto_LiRPA.utils import stop_criterion_batch
from input_split.branching_heuristics import input_split_branching
from input_split.split import input_split_parallel, get_split_depth
from branching_domains import check_worst_domain


class InputReluSplitter:
    def __init__(self):
        bab_args = arguments.Config['bab']
        branch_args = bab_args['branching']
        if not bab_args['interm_transfer']:
            raise ValueError('Branching input and activation must be used when '
                             'interm_transfer is True')
        self.relu_split_iterations = branch_args['branching_relu_iterations']
        self.input_split_iterations = branch_args['branching_input_iterations']
        self.split_order = branch_args['branching_input_and_activation_order']
        self.reseting_round = (self.relu_split_iterations
                               + self.input_split_iterations)

    def split_condition(self, r):
        use_input_split = (
            r % self.reseting_round < self.input_split_iterations
            if self.split_order[0] == 'input'
            else r % self.reseting_round >= self.relu_split_iterations)
        if use_input_split:
            print(f'Round {r}, using input split.')
        else:
            print(f'Round {r}, using activation split.')


@torch.no_grad()
def input_branching_decisions(wrapped_net, global_lbs, lAs, x_Ls, x_Us, rhs):
    """
    Use smart branching given lA to find the best idx to split.
    """
    split_idx = input_split_branching(
        net=wrapped_net, dom_lb=global_lbs,
        dm_l_all=x_Ls, dm_u_all=x_Us,
        lA=lAs, thresholds=rhs, branching_method='sb', split_depth=3)
    return split_idx


@torch.no_grad()
def input_split_on_relu_domains(domains, wrapped_net, batch_size):
    sort_domain_iter = arguments.Config['bab']['sort_domain_interval']
    time_pickout = time_branching = time_bounding = 0.0
    time_transfer = time_add = 0.0
    # Get batch of domains. Some of these parameters are not used like betas,
    # because we are not optimizing the bounds in input split. However, we
    # will update intermediate lower and upper bounds.
    time_pickout = time.time()
    prev_domain_len = len(domains)
    (_, lAs, interm_lbs, interm_ubs, alphas, betas, intermediate_betas,
        history, split_history, _, depth, cs, rhs,
        x_Ls, x_Us, split_idx) = domains.pick_out(
        batch=batch_size, device=wrapped_net.x.device)
    time_pickout = time.time() - time_pickout

    # Given the current input bounds x_Ls, x_Us, we conduct input split
    # and update all intermediate layer bounds.
    # First we need to run branching heuristic for input split.
    time_branching = time.time()
    split_depth = get_split_depth(x_Ls)
    new_x_Ls, new_x_Us, new_cs, new_rhs, split_depth = input_split_parallel(
        x_Ls, x_Us, wrapped_net.x.shape, cs, rhs, split_depth=split_depth,
        i_idx=split_idx)
    n_repeat = 2 ** split_depth
    time_branching = time.time() - time_branching

    # Second, compute new bounds using input split. Using CROWN with alphas.
    # TODO: beta should also be used.
    time_bounding = time.time()
    dup_alphas = defaultdict(dict)
    for spec_key in alphas.keys():
        for layer_key in alphas[spec_key]:
            this_alpha = alphas[spec_key][layer_key]
            # alpha has shapes (2, output_dim, batch, ...).
            # The actual shape depends on whether sparse alpha is used and
            # the type of the layer before Relu (linear or conv).
            dup_alphas[spec_key][layer_key] = this_alpha.repeat(
                        1, 1, n_repeat, *([1]*(this_alpha.ndim - 3)))
    reference_interm_bounds = {}
    # FIXME: should not include last layer in interm_lbs.
    for i in range(len(interm_lbs) - 1):
        no_repeat_dim = [1] * (interm_lbs[i].ndim - 1)
        reference_interm_bounds[
                wrapped_net.name_dict[i]] = [
                        interm_lbs[i].repeat(n_repeat, *no_repeat_dim),
                        interm_ubs[i].repeat(n_repeat, *no_repeat_dim)]
    ret = wrapped_net.get_lower_bound_naive(
        dm_l=new_x_Ls, dm_u=new_x_Us, alphas=[],
        bounding_method='crown', C=new_cs,
        stop_criterion_func=stop_criterion_batch(rhs),
        reference_interm_bounds=reference_interm_bounds,
    )
    # FIXME: should be a dictionary key.
    new_final_layer_lb = interm_lbs[-1].repeat(
            n_repeat, *([1]*(interm_lbs[-1].ndim - 1)))
    new_final_layer_ub = interm_ubs[-1].repeat(
            n_repeat, *([1]*(interm_ubs[-1].ndim - 1)))
    if ret[0][0] is not None:
        new_final_layer_lb = torch.max(ret[0].cpu(), new_final_layer_lb.cpu())
    time_bounding = time.time() - time_bounding

    # Extract intermediate layer bounds from the network, and update.
    time_transfer = time.time()
    new_interm_lbs, new_interm_ubs = wrapped_net.get_candidate_parallel(
            new_final_layer_lb, new_final_layer_ub, device='cpu')

    # After input split, we have updated dom_lb and dom_ub.
    # Simply duplicate the domain parameters that are not updated.
    dup_lAs = [_lA.repeat(n_repeat, *([1] * (_lA.ndim - 1))) for _lA in lAs]
    dup_rhs = rhs.repeat(n_repeat, 1)  # Size is (batch, spec).
    dup_cs = cs.repeat(n_repeat, 1, 1)  # Size is (batch, spec, n_output).
    # History must be deepcopied, however we will deepcopy it during pickout.
    history = history * n_repeat
    depth = depth * n_repeat
    # split_history and intermediate_betas are not used.
    split_history = split_history * n_repeat
    intermediate_betas = intermediate_betas * n_repeat
    if isinstance(betas, list):
        # Note that this will not actually copy tensors. However, they will
        # be copied to the newly created sparse_beta in the next iteration.
        betas = betas * n_repeat
    else:
        # TODO: betas should be saved in dict, not list.
        raise NotImplementedError
    time_transfer = time.time() - time_transfer

    # Compute branching heuristic using returned lA.
    _start_time = time.time()
    input_lAs = ret[2]
    new_split_idx = input_branching_decisions(
        wrapped_net, new_final_layer_lb, input_lAs,
        new_x_Ls, new_x_Us, dup_rhs)
    time_branching += time.time() - _start_time

    time_add = time.time()
    domains.add(dup_lAs, new_final_layer_lb, new_final_layer_ub,
                new_interm_lbs, new_interm_ubs, history,
                depth, dup_alphas, betas,
                split_history, None, dup_rhs, intermediate_betas,
                False, dup_cs,
                n_repeat*batch_size, x_Ls=new_x_Ls, x_Us=new_x_Us,
                input_split_idx=new_split_idx,
                ignore_sides=True)  # No left/right side domains.
    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()
    global_lb = check_worst_domain(domains)
    print(f'Previous domain size: {prev_domain_len}, picked out: {batch_size} '
          f'added: {len(depth)}, '
          f'new size: {len(domains)}')
    domains.print()
    time_add = time.time() - time_add
    print(f'Input split batch time : pickout: {time_pickout:.4f}\t '
          f'branching: {time_branching:.4f}\t bound: {time_bounding:.4f}\t '
          f'transfer: {time_transfer:.4f}\t '
          f'add_to_domain: {time_add:.4f}')
    print(f"Current (lb-rhs): {global_lb.max()}")
    return global_lb
