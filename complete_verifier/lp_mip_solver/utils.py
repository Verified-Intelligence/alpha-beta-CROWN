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
"""Build, solve, and refine output bounds using gurobi LP/MIP solver based on the bounds obtained by auto_LiRPA."""

import time
import torch
import sys
import os
import subprocess
import multiprocessing
import numpy as np
import gurobipy as grb
import arguments

try:
    from scip_model import SCIPModel, EarlyStopEvent, GenerateCutsEvent
except:  # pylint: disable=bare-except
    pass

from auto_LiRPA.bound_ops import BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd
from utils import get_reduce_op, get_batch_size_from_masks
import torch.nn as nn


def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise grb.GurobiError(message)


def clamp(X, lower_limit, upper_limit):
    """Clamp tensor values to the specified range."""
    return torch.max(torch.min(X, upper_limit), lower_limit)


def compute_ratio(lower_bound, upper_bound):
    """
    helper function to calculate fsb score
    """
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = upper_bound.relu()
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept


def FSB_score(model, results, branching_reduceop='min'):
    """Use FSB to sort the order for mip refinement."""

    # FIXME duplicate code (with those in branching heuristics)

    batch = get_batch_size_from_masks(results['mask'])
    reduce_op = get_reduce_op(branching_reduceop)

    score = []
    intercept_tb = []
    relu_idx = -1
    number_bounds = results['lower_bounds'][model.final_name].shape[0]

    for layer in reversed(model.net.relus):
        ratio = results['lA'][layer.name]
        key = layer.inputs[0].name
        assert len(results['mask'][layer.name]) == 1
        this_layer_mask = results['mask'][layer.name][0].unsqueeze(1)
        ratio_temp_0, ratio_temp_1 = compute_ratio(
            results['lower_bounds'][key], results['upper_bounds'][key])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, (intercept_candidate.view(batch, number_bounds, -1) * this_layer_mask).mean(1))
        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, (abs(score_candidate).view(batch, number_bounds, -1) * this_layer_mask).mean(1))

        relu_idx -= 1

    return score


# Global multiprocessing variables
multiprocess_mip_model = None
stop_multiprocess = False
mip_solve_time_start = 0

# Concurrency helpers and MIP attack worker

class NoDaemonProcess(multiprocessing.Process):
    """Make 'daemon' attribute always return False."""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NestablePool:
    """Simplified pool class for backward compatibility with NoDaemonProcess support."""
    def __init__(self, *args, **kwargs):
        self.pool = multiprocessing.Pool(*args, **kwargs)
        self.Process = NoDaemonProcess
    
    def __getattr__(self, name):
        return getattr(self.pool, name)


def mip_solver_attack_init(flag):
    """Initialize the termination flag for MIP solver attack."""
    global termination_flag
    termination_flag = flag


def mip_solver_worker(model, input_shape, queue):
    """Helper function for mip_solver_attack to run MIP optimization in a separate process."""
    input_shape = list(input_shape)
    try:
        model.optimize()
        if model.solcount > 0:
            objval = model.objval
            solution = torch.empty([1,] + input_shape[1:], requires_grad=False)
            # Extract the current adv example.
            _, C, H, W = input_shape
            dim = 0
            for chan in range(C):
                for row in range(H):
                    for col in range(W):
                        v = model.getVarByName(f"inp_{dim}")
                        # v = model.getVarByName(f"inp_[{chan},{row},{col}]")
                        solution[0,chan,row,col] = v.X
                        dim += 1
        else:
            solution = torch.zeros([0,] + input_shape[1:], requires_grad=False)
            objval = float("inf")
        queue.put((model.status, objval, model.objbound, model.solcount, solution))
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)
        queue.put((-1, float("inf"), float("-inf"), 0, solution))
    # Make sure all results are sent back. This process will be killed.
    time.sleep(10)


def mip_solver_attack(new_splits):
    """Adversarial attack using MIP; negative objval implies success."""
    global termination_flag
    model = multiprocess_mip_model.copy()
    indices, relu_status, opt_var, input_shape, best_adv_input, relu_forward,\
            pre_relu_layer_names, relu_layer_names, lower_bounds, upper_bounds = new_splits
    input_shape = list(input_shape)

    if lower_bounds is not None:
        assert arguments.Config["bab"]["attack"]["refined_mip_attacker"]
        print("using full alpha crown intermediate bounds to refine mip solver attack!")
        from .bounds_core import update_model_bounds
        update_model_bounds(model, lower_bounds, upper_bounds, pre_relu_layer_names, relu_layer_names, model_type="mip")

    for [relu_idx, neuron_idx], s in zip(indices, relu_status):
        a = model.getVarByName(f"aReLU{relu_layer_names[relu_idx]}_{neuron_idx}")
        if a is None:
            continue
        if s == 1:
            a.LB = 1
            a.UB = 1
        else:
            a.LB = 0
            a.UB = 0
    model.update()

    if best_adv_input is not None and len(best_adv_input) > 0:
        model.NumStart = best_adv_input.size(0)
        C, H, W = best_adv_input.shape[-3:]
        for s in range(best_adv_input.size(0)):
            model.params.StartNumber = s
            dim = 0
            for chan in range(C):
                for row in range(H):
                    for col in range(W):
                        v = model.getVarByName(f"inp_{dim}")
                        v.Start = best_adv_input[s][chan][row][col]
                        dim += 1
            for relu_idx, relu in enumerate(relu_forward):
                for i in range(relu.size(1)):
                    v = model.getVarByName(f"aReLU{relu_layer_names[relu_idx]}_{i}")
                    if v is not None:
                        v.Start = int(relu[s][i] > 0)

    v = model.getVarByName(opt_var)
    if 'ALPHA_BETA_CROWN_MIP_DEBUG' in os.environ:
        model.setParam('OutputFlag', True)
    model.setParam('BestBdStop', 1e-5)
    model.setParam('BestObjStop', -1e-5)
    model.setParam('Heuristics', 0.5)
    model.setParam('MIPFocus', 1)
    model.setObjective(v, grb.GRB.MINIMIZE)

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=mip_solver_worker, args=(model, input_shape, queue,))
    p.start()
    while queue.empty():
        time.sleep(0.5)
        if termination_flag.value == 1:
            print('Early Stopping MIP solver (another MIP process may succeed; or we run out of time).')
            p.terminate()
            time.sleep(0.5)
            p.kill()
            return float("inf"), float("inf"), -1, torch.zeros([0,] + input_shape[1:], requires_grad=False)
    status, objval, objbound, solcount, solution = queue.get()
    if objval < 0:
        termination_flag.value = 1
    print("Solved MIP for {}, {} neurons fixed, status:{} ({} sols), lower={:.6f}, upper={:.6f}".format(
        v.VarName, len(indices), status, solcount, objbound, objval))
    sys.stdout.flush()
    p.terminate()
    p.kill()
    return objval, objbound, status, solution


def copy_model(model, basis=True, use_basis_warm_start=True, remove_constr_list=[], env=None):
    """
    deep copy a gurobi model together with variable historical results
    """
    model_split = model.copy() if env is None else model.copy(env=env)
    for rc in remove_constr_list:
        rcs = model_split.getConstrByName(rc.ConstrName)
        model_split.remove(rcs)
    model_split.update()
    if not basis:
        return model_split
    xvars = model.getVars()
    svars = model_split.getVars()
    for x, s in zip(xvars, svars):
        if use_basis_warm_start:
            s.VBasis = x.VBasis
        else:
            s.PStart = x.X
    xconstrs = model.getConstrs()
    sconstrs = model_split.getConstrs()
    for s in sconstrs:
        x = model.getConstrByName(s.ConstrName)
        if use_basis_warm_start:
            s.CBasis = x.CBasis
        else:
            s.DStart = x.Pi
    model_split.update()
    return model_split


def mip_solver_lb_ub(candidate, init=None, save_adv=None, mip_skip_unsafe=False):
    """ Solving MIP for adversarial attack/complete verification.
    init: warm up with given init which is usually found by pgd attack
    save_adv: a list of input names that we need to retrieve if an adv found
    """

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
    adv = None
    if init is not None:
        init_lb, init_ub = init
    
    # Use global model for multiprocessing
    if multiprocess_mip_model is not None:
        mip_model = multiprocess_mip_model.copy()
    else:
        print("Error: no model available for mip_solver_lb_ub")
        return None, None, -1, None
    v = mip_model.getVarByName(candidate)
    vlb = out_lb = v.LB
    vub = out_ub = v.UB
    global stop_multiprocess, mip_solve_time_start
    if stop_multiprocess:
        return out_lb, out_ub, -1, adv  # Solver skipped.
    refine_time = time.time()
    if arguments.Config["solver"]["mip"]["early_stop"]:
        if arguments.Config["solver"]["mip"]["mip_solver"] == 'gurobi':
            mip_model.setParam('BestBdStop', 1e-5)  # Terminate as long as we find a positive lower bound.
            mip_model.setParam('BestObjStop', -1e-5)  # Terminate as long as we find a adversarial example.
        elif arguments.Config["solver"]["mip"]["mip_solver"] == 'scip':
            mip_model.includeEventhdlr(EarlyStopEvent(), "EarlyStopEvent", "early stop handler")
        else:
            raise NotImplementedError

    mip_model.setObjective(v, grb.GRB.MINIMIZE)
    if init_lb is not None:
        init_x_start(mip_model, init_lb)
    try:
        mip_model.optimize()
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)

    vlb = max(mip_model.objbound, out_lb)
    if mip_model.solcount > 0:
        vub = min(mip_model.objval, out_ub)
    if vub < 0:
        # An adversarial example is found
        # print("stop: adv found!")
        if not mip_skip_unsafe:
            stop_multiprocess = True
        if save_adv:
            adv = [mip_model.getVarByName(var_name).X for var_name in save_adv]

    assert mip_model.status != 3, "should not be infeasible"
    print("solving MIP for {}, status:{}, [{}, {}]=>[{}, {}], time: {}s".format(v.VarName, mip_model.status,
                    out_lb, out_ub, vlb, vub, time.time()-refine_time))
    sys.stdout.flush()
    if time.time() - mip_solve_time_start > arguments.Config["bab"]["timeout"]:
        stop_multiprocess = True
    return vlb, vub, mip_model.status, adv


def mip_solver_lb_ub_and(candidate, save_adv=None, rhs=None):
    """Solving MIP for AND mode verification."""
    adv = None
    model = multiprocess_mip_model.copy()
    v = [model.getVarByName(c) for c in candidate]
    global stop_multiprocess, mip_solve_time_start
    refine_time = time.time()

    # All output variables should be less or equal to the corresponding rhs.
    for i, vi in enumerate(v):
        model.addConstr(vi <= rhs[i])

    model.setObjective(0, grb.GRB.MINIMIZE)

    model.optimize()
    if model.Status == grb.GRB.OPTIMAL:
        if save_adv:
            adv = [model.getVarByName(var_name).X for var_name in save_adv]
        print("Feasible solution found")
    elif model.Status == grb.GRB.INFEASIBLE:
        print("No feasible solution found")
    elif model.Status == grb.GRB.TIME_LIMIT:
        print("Time limit reached")
    else:
        raise RuntimeError(f"Unexpected Gurobi status: {model.Status}")
    print("solving MIP time: {}s".format(
        time.time() - refine_time))
    sys.stdout.flush()
    return None, None, model.Status, adv


def update_mip_model_fix_relu(m, relu_idx, status, target=None, async_mip=False, best_adv=None, adv_activation_pattern=None,
                              refined_lower_bounds=None, refined_upper_bounds=None):
    """Update MIP model by fixing ReLU activation patterns.
    
    Args:
        relu_idx: indices of relu to be fixed
        status: the status of the relu
    """
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    batch_num = len(relu_idx)
    model_candidates = []
    relu_layer_names = [relu_layer.name for relu_layer in m.net.relus]
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in m.net.relus]

    out_vars = m.net.final_node().solver_vars
    for b in range(batch_num):
        lb, ub = None, None
        if refined_lower_bounds is not None:
            lb = [lower_bound[b:b+1].cpu().detach().numpy() for lower_bound in refined_lower_bounds]
            ub = [upper_bound[b:b+1].cpu().detach().numpy() for upper_bound in refined_upper_bounds]
        m.relu_indices_mask = None
        if len(out_vars) == 1:
            model_candidates.append((relu_idx[b], status[b], out_vars[0].VarName, m.input_shape, best_adv[b],
                                     adv_activation_pattern[b], pre_relu_layer_names, relu_layer_names, lb, ub))
        else:
            if target is None:
                target = (m.c == -1).view(-1).nonzero().item()
            model_candidates.append((relu_idx[b], status[b], out_vars[target].VarName, m.input_shape, best_adv[b],
                                     adv_activation_pattern[b], pre_relu_layer_names, relu_layer_names, lb, ub))

    global multiprocess_mip_model, stop_multiprocess
    stop_multiprocess = False
    multiprocess_mip_model = m.net.solver_model

    if getattr(m, 'pool', None) is None:
        m.pool_termination_flag = multiprocessing.Value('i')
        pool = NestablePool(mip_multi_proc, initializer=mip_solver_attack_init, initargs=(m.pool_termination_flag,))
        m.pool = pool
    else:
        pool = m.pool
        m.pool_termination_flag.value = 0

    if async_mip:
        solver_result = pool.map_async(mip_solver_attack, model_candidates)
        m.pool_result = solver_result
        return solver_result
    else:
        solver_result = pool.map(mip_solver_attack, model_candidates)
        attack_result = any([a[0] < 0 for a in solver_result])
        return attack_result, solver_result


# Global variables for multiprocessing
termination_flag = None
multiprocess_mip_model = None
stop_multiprocess = False
mip_solve_time_start = None


def check_optimization_success(model, introduced_constrs_all=None):
    """
    check the status of the gurobi model, remove the newly added split constraints from model if infeasible
    """
    if model.status == 2:
        # Optimization successful, nothing to complain about
        pass
    elif model.status == 3:
        print("infeasible!")
        for introduced_cons_layer in introduced_constrs_all:
            model.remove(introduced_cons_layer)
        # The model is infeasible. We have made incompatible
        # assumptions, so this subdomain doesn't exist.
        # raise InfeasibleMaskException()
    else:
        print('\n')
        print(f'Gurobi model.status: {model.status}\n')
        raise NotImplementedError
