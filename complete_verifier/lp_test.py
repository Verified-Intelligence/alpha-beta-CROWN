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
import copy
import torch
import numpy as np
from collections import deque

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.linear import BoundLinear
from beta_CROWN_solver import LiRPANet

def compare_optimized_bounds_against_lp_bounds(
        model_ori,
        data,
        data_ub=None,
        data_lb=None,
        vnnlib=None
    ):
    check_after_n_iterations = arguments.Config['debug']['test_optimized_bounds_after_n_iterations']
    include_output_constraint = arguments.Config['debug']['include_output_constraint']
    norm = arguments.Config['specification']['norm']
    # Generally, c should be constructed from vnnlib
    assert len(vnnlib) == 1
    vnnlib = vnnlib[0]
    c = torch.concat([
        # FIXME always use torch, not numpy
        item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
        for item in vnnlib[1]], dim=0).unsqueeze(1).to(data)
    if c.shape[0] != 1 and data.shape[0] == 1:
        # TODO need a more general solution.
        # transpose c to share intermediate bounds
        c = c.transpose(0, 1)
    rhs = torch.tensor(np.array([item[1] for item in vnnlib[1]])).to(data).t()
    model = LiRPANet(model_ori, in_size=data.shape, c=c)
    if isinstance(vnnlib[0], dict):
        ptb = PerturbationLpNorm(
            norm=vnnlib[0]['norm'],
            eps=vnnlib[0]['eps'], eps_min = vnnlib[0].get('eps_min', 0),
            x_L=data_lb, x_U=data_ub)
    else:
        # Perturbation value for non-Linf perturbations, None for all other cases.
        ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    solver_args = arguments.Config['solver']
    share_alphas = solver_args['alpha-crown']['share_alphas']

    model.x = x
    model.input_domain = domain
    model.net.set_bound_opts({
        'optimize_bound_args': {'stop_criterion_func': lambda x: False},
        'verbosity': 0,
    })
    model.set_crown_bound_opts('alpha')
    model.get_split_nodes(verbose=True)
    model._set_A_options()

    # Find all layers which bounds should be tested
    queue = deque([model.net.final_node()])
    test_list = []
    visited = set()
    while len(queue) > 0:
        l = queue.popleft()
        # Currently, the last layer must be linear
        # BoundConv should be tested, too. But first, issue #195 needs to be resolved.
        if isinstance(l, BoundLinear):
            test_list.append(l)
        for l_pre in l.inputs:
            if l_pre not in visited:
                queue.append(l_pre)
                visited.add(l_pre)

    # First, we need to get (very loose) bounds on all layers
    # If output constraints are used, both the LP solver and compute_bounds needs bounds
    # on all layers (including layers behind(!) the optimized layer).
    # compute_bounds would optimize all bounds of all layers, but that could mean it
    # gets better results than the LP solver which cannot tighten other layers
    # internally.
    known_layer_bounds = {}
    model.net.init_alpha(
        (x,),
        share_alphas=share_alphas,
        c=c,
        final_node_name=model.net.final_name
    )
    # Optimize for N iterations to get tighter bounds and check correctness at different
    # steps of the optimization process
    orig_iterations = model.net.bound_opts['optimize_bound_args']['iteration']
    model.net.bound_opts['optimize_bound_args']['iteration'] = check_after_n_iterations
    if check_after_n_iterations > 0:
        model.net.compute_bounds(
            x=(x,), C=c, method='CROWN-Optimized',
            return_A=False,
            bound_lower=True, bound_upper=True, aux_reference_bounds=None,
            final_node_name=model.net.final_name)
    model.net.bound_opts['optimize_bound_args']['iteration'] = orig_iterations
    for l in model.net.nodes():
        if l.is_lower_bound_current() or l.is_upper_bound_current():
            assert l.is_lower_bound_current()
            assert l.is_lower_bound_current()
            known_layer_bounds[l.name] = (l.lower, l.upper)

    for l in test_list[::-1]:
        print(f'Optimizing bounds for layer {l}')
        if model.net.final_name == l.name:
            used_c = c
        else:
            used_c = None
        assert l.name in known_layer_bounds
        filtered_known_layer_bounds = copy.deepcopy(known_layer_bounds)
        del filtered_known_layer_bounds[l.name]
        model.net.init_alpha(
            (x,),
            share_alphas=share_alphas,
            c=used_c,
            final_node_name=l.name,
            interm_bounds=filtered_known_layer_bounds,
        )
        optimized_lower_bound, optimized_upper_bound = model.net.compute_bounds(
            x=(x,), C=used_c, method='CROWN-Optimized',
            return_A=False,
            bound_lower=True, bound_upper=True, aux_reference_bounds=None,
            final_node_name=l.name,
            interm_bounds=filtered_known_layer_bounds)

        if include_output_constraint:
            # All layers need to have .lower and .upper set.
            # The above optimization has deleted those from all layers behind the
            # optimized one
            model.net.init_alpha(
                (x,),
                share_alphas=share_alphas,
                c=c,
                final_node_name=model.net.final_name,
                interm_bounds=known_layer_bounds,
            )
        optimized_lower_bound_np = optimized_lower_bound[0].detach().cpu().numpy()
        optimized_upper_bound_np = optimized_upper_bound[0].detach().cpu().numpy()

        print(f'Solving LP problems for layer {l}')
        optimal_lower_bounds_np = np.array(model.build_the_model_lp(
            optimized_layer_name=l.name,
            compute_upper_bound=False,
            include_output_constraint=include_output_constraint,
            rhs=rhs,
        ))
        optimal_upper_bounds_np = np.array(model.build_the_model_lp(
            optimized_layer_name=l.name,
            compute_upper_bound=True,
            include_output_constraint=include_output_constraint,
            rhs=rhs,
        ))
        print((optimized_lower_bound_np.sum(), optimal_lower_bounds_np.sum()))
        assert np.all(np.bitwise_or(
            np.isclose(optimal_lower_bounds_np, optimized_lower_bound_np),
            optimal_lower_bounds_np > optimized_lower_bound_np
        ))
        assert np.all(np.bitwise_or(
            np.isclose(optimal_upper_bounds_np, optimized_upper_bound_np),
            optimal_upper_bounds_np < optimized_upper_bound_np
        ))
        delta_lower_bounds = optimal_lower_bounds_np - optimized_lower_bound_np
        delta_upper_bounds = optimized_upper_bound_np - optimal_upper_bounds_np
        print(f'Delta between computed and optimal lower bound: '
              f' sum={delta_lower_bounds.sum()}, '
              f' max={delta_lower_bounds.max()}, '
              f' mean={delta_lower_bounds.mean()}')
        print(f'Delta between computed and optimal upper bound: '
              f' sum={delta_upper_bounds.sum()}, '
              f' max={delta_upper_bounds.max()}, '
              f' mean={delta_upper_bounds.mean()}')
        print()
        optimal_lower_bounds_torch = torch.tensor(optimal_lower_bounds_np)
        optimal_lower_bounds_torch = optimal_lower_bounds_torch.float().unsqueeze(0)
        optimal_upper_bounds_torch = torch.tensor(optimal_upper_bounds_np)
        optimal_upper_bounds_torch = optimal_upper_bounds_torch.float().unsqueeze(0)
        assert optimal_lower_bounds_torch.shape == optimized_lower_bound.shape
        assert optimal_upper_bounds_torch.shape == optimized_upper_bound.shape
        known_layer_bounds[l.name] = (
            optimal_lower_bounds_torch,
            optimal_upper_bounds_torch,
        )
