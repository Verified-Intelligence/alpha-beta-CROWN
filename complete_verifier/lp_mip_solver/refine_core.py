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
"""MIP refinement utilities extracted from legacy lp_mip_solver."""

import time
import sys
import os
import random
import numpy as np
import torch
import multiprocessing
import gurobipy as grb

import arguments
from auto_LiRPA.beta_crown import SparseBeta
from auto_LiRPA.bound_ops import BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd
from auto_LiRPA.utils import stop_criterion_min, reduction_str2func

from .utils import clamp, handle_gurobi_error


multiprocess_mip_model = None
mip_refine_time_start = None
mip_refine_timeout = 230


def build_the_model_mip_refine(m, lower_bounds, upper_bounds,
            stop_criterion_func=stop_criterion_min(1e-4), score=None, FSB_sort=True,
            topk_filter=1.):
    """
    Before the first branching, we build the model and create a mask matrix
    Output: relu_mask, current intermediate upper and lower bounds, a list of
            indices of the layers right before a Relu layer
            the constructed gurobi model
    NOTE: we keep all bounds as a list of tensors from now on.
            Only lower and upper bounds are kept in the same shape as layers' outputs.
    """

    # Get the layer and the bounds.
    m.layers = list(m.model_ori.children())

    # Intermediate bounds for mip_solver.
    reference_bounds = {}
    for i, _ in enumerate(m.net.relus):
        reference_bounds[m.net.relus[i].inputs[0].name] = [lower_bounds[m.net.relus[i].inputs[0].name], upper_bounds[m.net.relus[i].inputs[0].name]]
    reference_bounds[m.final_name] = [lower_bounds[m.final_name], upper_bounds[m.final_name]]

    x = m.x

    # Load the parameters.
    lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
    lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
    share_alphas = arguments.Config["solver"]["alpha-crown"]["share_alphas"]
    optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
    loss_reduction_func = reduction_str2func(arguments.Config["general"]["loss_reduction_func"])
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    mip_perneuron_refine_timeout = arguments.Config["solver"]["mip"]["refine_neuron_timeout"]
    mip_perneuron_refine_timeout_increasement = arguments.Config["solver"]["mip"]["refine_neuron_timeout_increasement"]
    timeout_neuron_threshold = arguments.Config["solver"]["mip"]["timeout_neuron_percentage"]
    remaining_timeout_threshold = arguments.Config["solver"]["mip"]["remaining_timeout_threshold"]
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
    sliding_window = arguments.Config["solver"]["mip"]["sliding_window"]
    remove_neurons_from_A = arguments.Config["solver"]["mip"]["remove_unstable_neurons"]

    # The refine degree depends on the timeout.
    global mip_refine_timeout
    mip_refine_timeout = arguments.Config["solver"]["mip"]["refine_neuron_time_percentage"] * arguments.Config["bab"]["timeout"]

    # Preset the args for incomplete full crown with refined bounds.
    m.net.init_alpha((m.x,), share_alphas=share_alphas, c=m.c)
    m.net.set_bound_opts({'verbosity': 1})
    m.net.set_bound_opts({'optimize_bound_args': {
        'iteration': 100, 'enable_beta_crown': False, 'enable_alpha_crown': True,
        'use_shared_alpha': share_alphas, 'optimizer': optimizer,
        'fix_interm_bounds': True,
        'lr_alpha': lr_init_alpha, 'init_alpha': False,
        'loss_reduction_func': loss_reduction_func,
        'stop_criterion_func': stop_criterion_func,
        'lr_decay': lr_decay}})
    lb_refined, ub_refined = None, None

    # Initialize the model.
    solver_model_vars = {}
    solver_vars = m.net.build_solver_module((x,), final_node_name = list(m.net.nodes())[0].name, interm_bounds = reference_bounds)
    solver_model_vars[-1] = solver_vars
    m.net.solver_model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
    m.net.solver_model.setParam("FeasibilityTol", 2e-5)

    #############
    # Config the hyperparameters for intermediate bounds refinement with mip.

    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    m.net.solver_model.setParam('TimeLimit', mip_perneuron_refine_timeout)
    m.net.solver_model.setParam('MIPGap', 1e-2)  # Relative gap between primal and dual.
    m.net.solver_model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between primal and dual.
    m.net.solver_model.setParam('Threads', mip_threads)

    global multiprocess_mip_model
    multiprocess_mip_model = m.net.solver_model
    print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads},"
            f"total threads used: {mip_multi_proc*mip_threads}, mip_perneuron_refine_timeout: {mip_perneuron_refine_timeout}")
    print(f"[total time budget for MIP: {mip_refine_timeout}]\n")

    refine_start_time = time.time()
    #############

    # Keep a record of model's information.
    m.gurobi_vars = []
    m.relu_constrs = []
    m.relu_indices_mask = []

    ## Refine the bounds with mip_solver.
    global mip_refine_time_start, mip_multiprocess_mip_model
    mip_refine_time_start = time.time()
    # Need to handle the cases where unstabled neurons are refined to stable.
    # (this relu_idx layer neuron idx, 1:>0, -1:<0)
    unstable_to_stable = [[] for _ in m.net.relus]
    last_relu_layer_refined = False
    any_relu_layer_refined = False
    final_alpha_crown_needed = True
    init_relu = 0
    device = m.net.device
    # This variable is to record whether an early stop action happened in the compute_bounds function. 
    early_stop = True
    unstable_neuron_filter = torch.tensor([])
    A_dict_ = {}
    # Set up the sliding window size.
    if sliding_window > 0:
        window_size = sliding_window
        print(f'using sliding window in mip solver and the window size is: {window_size}')
    else:
        window_size = None
    # Stop refinement by timeout.
    stop_by_timeout = False

    # Here the refinement will start from the first layer until the time is up.
    for relu_idx, layer in enumerate(m.net.relus[:]):
        print('refine the bounds for layer ', layer.name)
        # Change the constraints when sliding window is enabled.
        if window_size and relu_idx >= window_size:
            init_relu += 1
            print(f'initial constraints: {len(m.net.solver_model.getConstrs())}')
            layer_name = solver_model_vars[init_relu][1]
            # Remove the initial constraints.
            for constr in m.net.solver_model.getConstrs():
                if layer_name in constr.ConstrName:
                    m.net.solver_model.remove(constr)
            m.net.solver_model.update()
            print(f'updated constraints: {len(m.net.solver_model.getConstrs())}')
            # If the window makes the bounds much looser, adding extra constraints may help the tighten the bound.
            if arguments.Config["solver"]["mip"]["extra_constraint"] and not early_stop:
                vars = np.array(solver_model_vars[init_relu][0])
                input_vars = np.array(solver_model_vars[-1]).reshape(-1)
                target_length = vars.shape[0]
                input_length = input_vars.shape[0]
                lA = A_dict_[init_relu]['lA'].reshape(target_length, input_length)
                lbias = A_dict_[init_relu]['lbias'].reshape(target_length)
                uA = A_dict_[init_relu]['uA'].reshape(target_length, input_length)
                ubias = A_dict_[init_relu]['ubias'].reshape(target_length)
                # Add extra constraints between current start layer and the input based on alpha-crown.
                for i in range(target_length):
                    m.net.solver_model.addConstr(vars[i] >= grb.LinExpr(lA[i], input_vars)+lbias[i])
                    m.net.solver_model.addConstr(vars[i] <= grb.LinExpr(uA[i], input_vars)+ubias[i])
                m.net.solver_model.update()
            print(f'extra constraints from input: {len(m.net.solver_model.getConstrs())}')
        # As for the initial condition of the layers, the bounds are set to infinity.
        reference_bounds[layer.inputs[0].name] = [torch.full_like(reference_bounds[layer.inputs[0].name][0], -np.inf),
                                                torch.full_like(reference_bounds[layer.inputs[0].name][1], np.inf)]
        solver_vars = m.net.build_solver_module(x=(x,0), final_node_name = layer.inputs[0].name, interm_bounds = reference_bounds, set_input=False)
        solver_model_vars[relu_idx] = (solver_vars, layer.name)

        if relu_idx >= 1:
            print(f'current relu layer: {layer.name}, idx: {relu_idx}')
            print(f'timeout for this layer is {mip_perneuron_refine_timeout}')
            # Select the unstable neurons for Mip solver.
            removed_neurons = 0
            candidates_with_bounds = []
            candidates = []
            candidates_idx = []
            if type(layer.inputs[0]) == BoundLinear:
                for neuron_idx,v in enumerate(solver_vars):
                    out_lb = lower_bounds[layer.inputs[0].name][0, neuron_idx]
                    out_ub = upper_bounds[layer.inputs[0].name][0, neuron_idx]
                    # Since the neurons that has positive coefficients in alpha-crown won't affect the
                    # final lower bounds. So we don't have to refine them. But they may affect the bounds 
                    # of other intermediate layer bounds. Start on the third linear layer.
                    if remove_neurons_from_A and unstable_neuron_filter.numel() != 0 and unstable_neuron_filter[neuron_idx] == 0:
                        if out_lb * out_ub < 0:
                            removed_neurons += 1
                        continue
                    # For those stable neurons, skip for updates.
                    if out_lb * out_ub < 0:
                        product = out_lb * out_ub
                        candidates_with_bounds.append(((v.VarName, None, out_lb.cpu().tolist(), out_ub.cpu().tolist()), neuron_idx, product))

                candidates = [item[0] for item in candidates_with_bounds]
                candidates_idx = [item[1] for item in candidates_with_bounds]
                unstable_neuron_filter = torch.tensor([])
                print(f'Removed {removed_neurons} unstable neurons in MIP based on the coefficient matrix.')

                if score is not None and FSB_sort:
                    s = score[relu_idx].view(-1)[candidates_idx]
                    _, indices = s.sort(descending=True)
                    candidates = np.array(candidates)[indices.cpu().numpy()].tolist()
                    candidates_idx = np.array(candidates_idx)[indices.cpu().numpy()].tolist()
                    if topk_filter != 1.:
                        candidates = candidates[:int(len(candidates)*topk_filter)]
                        candidates_idx = candidates_idx[:int(len(candidates_idx)*topk_filter)]
            elif type(layer.inputs[0]) == BoundConv:
                for chan in range(len(solver_vars)):
                    for row in range(len(solver_vars[chan])):
                        for col in range(len(solver_vars[chan][row])):
                            candidates.append((solver_vars[chan][row][col].VarName, None, None, None))
                            candidates_idx.append([chan, row, col])
            # Multiprocess.
            multiprocess_mip_model = m.net.solver_model
            refine_time = time.time()
            if relu_idx == 1:
                if adv_warmup:
                    # Create pgd adv list as mip refinement warmup.
                    adv, max_values, min_values = _intermediate_PGD_attack(
                        m, relu_idx, restarts=3, attack_iters=50, alpha=None)
                    adv_list = []
                    layer_size = len(solver_vars)
                    for neuron_idx in candidates_idx:
                        adv_list.append((adv[neuron_idx + layer_size].cpu().tolist(), adv[neuron_idx].cpu().tolist())) # (low adv, max adv)
                    candidates = [(name, adv, out_lb, out_ub) for (name, _, out_lb, out_ub), adv in zip(candidates, adv_list)]
                with multiprocessing.Pool(mip_multi_proc) as pool:
                    solver_result = pool.starmap_async(mip_solver, candidates, chunksize=1)
                    if relu_idx + 1 < len(m.net.relus) and adv_warmup:
                        # Create adv list for next relu layer if still have next relu layer.
                        adv, max_values, min_values = _intermediate_PGD_attack(m, relu_idx + 1, restarts=3, attack_iters=50, alpha=None)
                    if not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
                        for relu_layer in m.net.relus:
                            relu_layer.sparse_betas = [SparseBeta((1, 0), device=device)]
                    else:
                        max_splits_per_layer = [0 for _ in range(len(m.net.relus))]
                        for relu_layer in m.net.relus:
                            relu_layer.sparse_betas = {}
                            for key in relu_layer.alpha.keys():
                                relu_layer.sparse_betas[key] = SparseBeta((1, 0), device=device)
                    # Find the A_dict for the linear layer correspond to the input.
                    if window_size:
                        lb_refined, ub_refined, A_dict = m.net.compute_bounds(x=(x,),
                                C=m.c, method='CROWN-optimized', return_A=True, needed_A_dict={layer.inputs[0].name:[list(m.net.nodes())[0].name]},
                                reference_bounds=reference_bounds, bound_upper=True)
                        print("alpha-CROWN with intermediate bounds by MIP:", lb_refined, ub_refined)
                        # Store the linear cofficient from alpha-crown for extra constraints in sliding window.
                        # If the compute_bounds stops early, it means no improvment and the A_dict will be empty.
                        if A_dict:
                            A_dict_[relu_idx] = A_dict[layer.inputs[0].name][list(m.net.nodes())[0].name]
                            early_stop = False
                    solver_result = solver_result.get()
            else:
                with multiprocessing.Pool(mip_multi_proc) as pool:
                    if adv_warmup:
                        adv_list = []
                        layer_size = len(solver_vars)
                        for neuron_idx in candidates_idx:
                            adv_list.append((adv[neuron_idx + layer_size].cpu().tolist(), adv[neuron_idx].cpu().tolist())) # (low adv, max adv)
                        candidates = [(name, adv, out_lb, out_ub) for (name, _, out_lb, out_ub), adv in zip(candidates, adv_list)]
                        solver_result = pool.starmap_async(mip_solver, candidates, chunksize=1)

                        if relu_idx + 1 < len(m.net.relus):
                            # Create adv list for next relu layer if still have next relu layer.
                            adv, max_values, min_values = _intermediate_PGD_attack(m, relu_idx + 1, restarts=3, attack_iters=50, alpha=None)
                    else:
                        solver_result = pool.starmap_async(mip_solver, candidates, chunksize=1)

                    # Config intermediate betas for last refined relu layer.
                    # We need to use beta crown to fully consider neurons that are refined from unstable to stable.
                    if last_relu_layer_refined and (time.time() - mip_refine_time_start < mip_refine_timeout):
                        print(f"Run alpha-CROWN after refining relu idx {relu_idx-1}")
                        max_splits_per_layer = len(unstable_to_stable[relu_idx-1])
                        refined_relu_layer = m.net.relus[relu_idx-1]
                        if not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
                            # Init all regular betas.
                            refined_relu_layer.sparse_betas = [SparseBeta((1, max_splits_per_layer), device=device)]
                            # Assign split constraint into regular betas.
                            for neuron_idx, (refined_neuron, sign) in enumerate(unstable_to_stable[relu_idx-1]):
                                refined_relu_layer.sparse_betas[0].loc[0, neuron_idx] = refined_neuron
                                refined_relu_layer.sparse_betas[0].sign[0, neuron_idx] = sign
                        else:
                            for key in refined_relu_layer.sparse_betas.keys():
                                # Init all intermediate betas.
                                refined_relu_layer.sparse_betas[key] = SparseBeta((1, max_splits_per_layer), device=device)
                            for neuron_idx, (stable_neuron_idx, sign) in enumerate(unstable_to_stable[relu_idx-1]):
                                for key in refined_relu_layer.sparse_betas.keys():
                                    # Assign split constraint into all intermdiate betas
                                    refined_relu_layer.sparse_betas[key].loc[0, neuron_idx] = stable_neuron_idx
                                    refined_relu_layer.sparse_betas[key].sign[0, neuron_idx] = sign
                        print("relu layer:", relu_idx-1, "has unstable to stable neurons:", unstable_to_stable[relu_idx-1])
                        # When use the convolutional layer, beta should be forbiddened by setting beta to false.
                        m.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': arguments.Config['solver']['beta-crown']['beta'],
                            "verbose": True}, 'enable_opt_interm_bounds': arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']})
                        # For the last layer, we find the neuron that has negative sign in alpha-crown coefficient.
                        A_dict_candidate = {m.net.final_node().name:layer.inputs[0].name}
                        if window_size:
                            A_dict_candidate[layer.inputs[0].name] = [list(m.net.nodes())[0].name]
                        lb_refined, ub_refined, A_dict = m.net.compute_bounds(x=(x,),
                            C=m.c, method='CROWN-optimized', return_A=True, needed_A_dict=A_dict_candidate,
                            reference_bounds=reference_bounds, bound_upper=True)
                        # If the compute_bounds stops early, it means no improvment and the A_dict will be empty.
                        if A_dict:
                            # Due to the early stop, the A_dict may be empty
                            if window_size and layer.inputs[0].name in A_dict:
                                A_dict_[relu_idx] = A_dict[layer.inputs[0].name][list(m.net.nodes())[0].name]
                                early_stop = False
                            # Find those neurons that have all positive cofficient with respect to output lower bounds.
                            if remove_neurons_from_A and m.net.final_node().name in A_dict:
                                unstable_neuron_filter = A_dict[m.net.final_node().name][layer.inputs[0].name]['lA'].squeeze(0)
                                unstable_neuron_filter = torch.sum(unstable_neuron_filter<=0, dim=0)
                        print("alpha-CROWN with intermediate bounds by MIP:", lb_refined, ub_refined)
                        final_alpha_crown_needed = False
                        # Early stop if no refine needed for this relu.
                        if lb_refined.min().item()>=0:
                            print(f"min of alpha-CROWN bounds {lb_refined.min().item()}>=0, verified!")
                            pool.terminate()
                            break
                    else:
                        final_alpha_crown_needed = True

                    solver_result = solver_result.get()

            refined_num = 0
            timeout_num = 0
            # Update bounds.
            for idx, (lb, ub, refined, timeout_check) in zip(candidates_idx, solver_result):
                if type(layer.inputs[0]) == BoundLinear:
                    if refined:
                        vlb = max(lb, lower_bounds[layer.inputs[0].name][0, idx])
                        vub = min(ub, upper_bounds[layer.inputs[0].name][0, idx])
                        lower_bounds[layer.inputs[0].name][0, idx] = vlb
                        upper_bounds[layer.inputs[0].name][0, idx] = vub
                        # We only care about the bounds right before the relu layer.
                        refined_num += 1
                        if vlb >= 0:
                            unstable_to_stable[relu_idx].append((idx, 1))
                        if vub <= 0:
                            unstable_to_stable[relu_idx].append((idx, -1))
                    v = solver_vars[idx]
                    v.LB, v.UB = lower_bounds[layer.inputs[0].name][0, idx], upper_bounds[layer.inputs[0].name][0, idx]
                    if timeout_check:
                        timeout_num += 1
                # For the convolutional layer, all the neurons will be refined.
                elif type(layer.inputs[0]) == BoundConv:
                    if refined:
                        refined_num += 1
                        v = solver_vars[idx[0]][idx[1]][idx[2]]
                        v.LB = lb
                        v.UB = ub
                    if timeout_check:
                        timeout_num += 1
            m.net.solver_model.update()
            refine_time = time.time() - refine_time
            print(f"MIP improved {refined_num} nodes out of {len(solver_result)} unstable nodes for layer {layer.name}, time {refine_time:.4f}")
            print(f"MIP got {timeout_num} timeout nodes out of {len(solver_result)} nodes for layer {layer.name}")
            last_relu_layer_refined = (refined_num > 0)
            any_relu_layer_refined = any_relu_layer_refined or last_relu_layer_refined
            # If too many timeout or remaining time is sufficient, just increase the time for next layer.
            if (timeout_num > len(solver_result) * timeout_neuron_threshold or
                mip_refine_timeout - time.time() - mip_refine_time_start > remaining_timeout_threshold * mip_perneuron_refine_timeout * (len(m.net.relus[:])-relu_idx)):
                mip_perneuron_refine_timeout += mip_perneuron_refine_timeout_increasement
            # Stop refine if timeout.
            if (time.time() - mip_refine_time_start >= mip_refine_timeout):
                # We don't break here, or the intermediate bounds of the current relu layer will not be updated.
                stop_by_timeout = True
            # Update the bounds for current relu layer and its previous layer.
        for i, _ in enumerate(m.net.relus):
            reference_bounds[m.net.relus[i].inputs[0].name] = [lower_bounds[m.net.relus[i].inputs[0].name], upper_bounds[m.net.relus[i].inputs[0].name]]
        # Set up Relu constraints for the Mip solver.
        solver_vars = m.net.build_solver_module(x=(x,), final_node_name = layer.name, interm_bounds=reference_bounds, set_input=False)
        print("maximum relu layer improved by MIP so far", relu_idx)
        if stop_by_timeout:
            print(f"Refine time {time.time() - mip_refine_time_start:.4f} exceeds the limit {mip_refine_timeout}, stop refining.")
            break

    print(f'MIP finished with {time.time() - refine_start_time}s')

    if any_relu_layer_refined and final_alpha_crown_needed:
        # We run alpha-CROWN if any relu layer is refined, but if we've run alpha-CROWN in the last iteration, we skip it.
        print(f"Run final alpha-CROWN after MIP solving on layer {len(m.layers)-1} and relu idx {len(m.net.relus)}")
        reference_bounds = {}
        for i, layer in enumerate(m.net.relus):
            # Only refined with the relu layers that are refined by mip before.
            # if i>=(maximum_refined_relu_layers+1): break
            nd = m.net.relus[i].inputs[0].name
            print(i, nd, lower_bounds[nd].shape)
            reference_bounds[nd] = [lower_bounds[nd], upper_bounds[nd]]

        lb_refined, ub_refined = m.net.compute_bounds(x=(x,), C=m.c,
            method='CROWN-optimized', reference_bounds=reference_bounds, bound_upper=False)
        print("alpha-CROWN with intermediate bounds improved by MIP:", lb_refined, ub_refined)

    # Creating history: batch, relu layers, [[loc neuron_idx],[coeff 1 if>=0 else -1]]
    history = [[[], [], [], [], []] for _ in m.net.relus]
    # Creating history betas: batch, relu layers, [beta tensor for this layer]
    if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
        retb = [[torch.tensor([])] for _ in m.net.relus]
    else:
        retb = [torch.tensor([]) for _ in m.net.relus]

    if lb_refined is None:
        for node in m.net.relus:
            del node.sparse_betas

        # The final layer bounds have the shape of (spec, 1)
        assert lower_bounds[m.final_name].shape[0] == m.c.shape[1], (
            "The final layer bounds should be the same as the dimension 1 of C matrix "
            "(i.e., the number of specifications).")
        lower_bounds[m.final_name] = lower_bounds[m.final_name].t()
        upper_bounds[m.final_name] = upper_bounds[m.final_name].t()
        return lower_bounds, upper_bounds, ([history], [retb])

    lb_refined, ub_refined, _ = m.get_interm_bounds(lb_refined)  # primals are better upper bounds
    ##### save refined betas to bab if not verified #####
    for mi, relu_layer in enumerate(m.net.relus):
        max_splits_per_layer = len(unstable_to_stable[mi])
        for neuron_idx, coeff in unstable_to_stable[mi]:
            history[mi][0].append(neuron_idx)
            history[mi][1].append(coeff)
            history[mi][2].append(0.0) # the split lb/ub is 0.0 in ReLU layer.
        # Save only used beta, discard padding beta.
        if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
            val_i = []
            for key in relu_layer.sparse_betas.keys():
                val_i.append(relu_layer.sparse_betas[key].val.cpu()[0, :max_splits_per_layer])
                # We only save betas for last layer optimization for now; the rest layer betas are not saved.
                # if key == m.net.final_name: val_i.append(relu_layer.sparse_betas[key].val.cpu()[0, :max_splits_per_layer])
            retb[mi] = val_i
        else:
            retb[mi] = relu_layer.sparse_betas[0].val.cpu()[0, :max_splits_per_layer]
        del relu_layer.sparse_betas
    return lb_refined, ub_refined, ([history], [retb])


def mip_solver(
        candidate,
        init=None,
        lower_bound: float = None,
        upper_bound: float = None
    ):
    """ Multiprocess worker for solving MIP models in build_the_model_mip_refine

    lower_bound and upper_bound are only used for logging in case the LP variable has
    no bounds associated (those might be inf for linear layers)
    """
    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9:
            # Timed out. Get current bound.
            bound = bound_type(grb_model.objbound, reference)
            refined = bound != reference
        elif grb_model.status == 2:
            # Optimally solved.
            bound = grb_model.objbound
            refined = True
        elif grb_model.status == 15:
            # We have find an lower bound >= 0 or upper bound <= 0, so this neuron becomes stable.
            bound = bound_type(1., -1.) * eps
            refined = True
        else:
            bound = reference
        return bound, refined, grb_model.status

    def solve_ub(model, v, out_ub, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
        status_ub_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminate as long as we find a negative upper bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min)
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
        status_lb_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminate as long as we find a positive lower bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max)
        return vlb, refined, status_lb, status_lb_r

    def init_x_start(model, init):
        init_array = np.array(init)
        init_shape = init_array.shape
        dim = 0
        for indices in np.ndindex(init_shape):
            v = model.getVarByName(f'inp_{dim}')
            v.Start = init_array[indices]
            dim += 1
        model.update()
        return

    init_lb, init_ub = None, None
    if init is not None:
        init_lb, init_ub = init
    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    out_lb, out_ub = v.LB, v.UB
    if lower_bound is not None:
        assert out_lb == -np.inf or out_lb == lower_bound
        out_lb = lower_bound
    if upper_bound is not None:
        assert out_ub == np.inf or out_ub == upper_bound
        out_ub = upper_bound
    refine_time = time.time()
    neuron_refined = False
    if time.time() - mip_refine_time_start >= mip_refine_timeout:
        return out_lb, out_ub, False, False
    eps = 1e-5

    if abs(out_lb) < abs(out_ub):
        # lb is tighter, solve lb first.
        vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps, init=init_lb)
        neuron_refined = neuron_refined or refined
        if vlb < 0:
            # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps, init=init_ub)
            neuron_refined = neuron_refined or refined
        else:
            # lb > 0, neuron is stable, we skip solving ub.
            vub, status_ub, status_ub_r = out_ub, -1, -1
    else:
        # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps, init=init_ub)
        neuron_refined = neuron_refined or refined
        if vub > 0:
            # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps, init=init_lb)
            neuron_refined = neuron_refined or refined
        else:
            # ub < 0, neuron is stable, we skip solving ub.
            vlb, status_lb, status_lb_r = out_lb, -1, -1

    print("Solving MIP for {},[{},{}]=>[{},{}] ({},{}; {},{}), time: {:.4f}s, #vars: {}, #constrs: {}, improved: {}".format(v.VarName, out_lb, out_ub, vlb, vub,
            status_lb, status_lb_r, status_ub, status_ub_r, time.time()-refine_time, model.NumVars, model.NumConstrs, neuron_refined))
    sys.stdout.flush()
    this_node_timeout = False
    if time.time() - refine_time >= float(model.getParamInfo('TimeLimit')[2]) * 2:
        this_node_timeout = True
    return vlb, vub, neuron_refined, this_node_timeout


def _intermediate_PGD_attack(m, relu_idx, restarts=3, attack_iters=50, alpha=None):
    lb, ub = m.x.ptb.x_L, m.x.ptb.x_U
    # make sure lb is not larger than ub
    lb = torch.min(lb, ub)
    X = m.x.detach().clone()

    # one clean forward to initialize all forward_value, but keep the bounds
    m.net(X, clear_forward_only=True)
    if relu_idx != "final":
        # pre-relu layer bounds
        forward_value = m.net.relus[relu_idx].inputs[0].forward_value
    else:
        # final output layer bounds
        forward_value = m.net.final_node().forward_value.mm(m.c[0].T)
    assert forward_value.size(0) == 1, "batch should be 1"
    layer_size = forward_value.reshape(-1).size(0)
    # maximize for the first half and minimize for the rest
    batch_size = layer_size * 2
    repeat_dims = [batch_size] + [1] * (X.ndim - 1)
    X = X.repeat(*repeat_dims)


    if alpha is None:
        alpha = (ub - lb).max() / 10.

    # print(forward_value[0, 10])
    device = m.net.device
    max_values = -torch.ones(layer_size, device=device) * 1e8
    min_values = torch.ones(layer_size, device=device) * 1e8
    best_delta = torch.zeros_like(X, device=device)
    for zz in range(restarts):
        delta = torch.zeros_like(X, device=device)
        delta.uniform_(0,1)
        delta = delta * (ub - lb) + lb
        delta = (delta - X).detach()
        delta.requires_grad = True
        for _ in range(attack_iters):
            m.net(X + delta, clear_forward_only=True)
            if relu_idx != "final":
                forward_value = m.net.relus[relu_idx].inputs[0].forward_value
            else:
                forward_value = m.net.final_node().forward_value.mm(m.c[0].T)
            maxv = forward_value[:layer_size].masked_select(torch.eye(layer_size, device=device).bool())
            minv = forward_value[layer_size:].masked_select(torch.eye(layer_size, device=device).bool())
            # print(maxv[218], minv[218])
            # print(zz, maxv[1], minv[1])
            loss = maxv.sum() - minv.sum()
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), lb - X, ub - X)
            delta.grad.zero_()
        m.net(X + delta, clear_forward_only=True)
        if relu_idx != "final":
            forward_value = m.net.relus[relu_idx].inputs[0].forward_value
        else:
            forward_value = m.net.final_node().forward_value.mm(m.c[0].T)
        maxv = forward_value[:layer_size].masked_select(torch.eye(layer_size, device=device).bool())
        minv = forward_value[layer_size:].masked_select(torch.eye(layer_size, device=device).bool())
        max_idx = (maxv >= max_values)
        min_idx = (minv <= min_values)
        best_delta[:layer_size][max_idx] = delta.detach()[:layer_size][max_idx]
        best_delta[layer_size:][min_idx] = delta.detach()[layer_size:][min_idx]
        max_values = torch.max(max_values, maxv)
        min_values = torch.min(min_values, minv)
        # print(max_values[218], min_values[218])
        # print(max_values[1], min_values[1])
    print(f"PGD done for relu layer {relu_idx}")
    # reset forward_value for each layer
    m.net(m.x, clear_forward_only=True)
    return X + best_delta, max_values, min_values
