#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Build, solve, and refine output bounds using gurobi LP/MIP solver based on the bounds obtained by auto_LiRPA."""

import copy
import time
import random
from collections import defaultdict, OrderedDict

import torch
import arguments
from torch.nn import ZeroPad2d

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import (reduction_min, reduction_max, reduction_mean, reduction_sum,
                            stop_criterion_sum, stop_criterion_min)
from model_defs import Flatten
from auto_LiRPA.bound_ops import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd

import multiprocessing
import multiprocessing.pool
import sys
import os

try:
    import gurobipy as grb
except ModuleNotFoundError:
    pass


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func


def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise 


multiprocess_mip_model = None
multiprocess_lp_model = None
stop_multiprocess = False
mip_refine_time_start = None
mip_refine_timeout = 230


def mip_solver(candidate):
    """ Multiprocess worker for solving MIP models in build_the_model_mip_refine """
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

    def solve_ub(model, v, out_ub, eps=1e-5):
        status_ub_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        try:
            model.optimize()
        except grb.GurobiError as e: 
            handle_gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min)
        # assert status_ub != 3, "ub status 3"
        # if status_ub == 3:
        #     model_relaxed = model.copy()
        #     model_relaxed.reset()
        #     model_relaxed.setParam('BestBdStop', float("inf"))
        #     model_relaxed.setParam('MIPGap', 1e-4)
        #     model_relaxed.setParam('MIPGapAbs', 1e-10)
        #     relaxed_v = model_relaxed.getVarByName(candidate)
        #     model_relaxed.setObjective(relaxed_v, grb.GRB.MAXIMIZE)
        #     model_relaxed.feasRelaxS(0, True, True, False)
        #     model_relaxed.optimize()
        #     vub, refined, status_ub_r = get_grb_solution(model_relaxed, out_ub, min)
        #     del model_relaxed
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5):
        status_lb_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        try:
            model.optimize()
        except grb.GurobiError as e: 
            handle_gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max)
        # assert status_lb != 3, "lb status 3"
        # if status_lb == 3:
        #     # Deal with infeasibility caused by potential numerical issues.
        #     model_relaxed = model.copy()
        #     model_relaxed.reset()
        #     model_relaxed.setParam('BestBdStop', float("inf"))
        #     model_relaxed.setParam('MIPGap', 1e-4)
        #     model_relaxed.setParam('MIPGapAbs', 1e-10)
        #     relaxed_v = model_relaxed.getVarByName(candidate)
        #     model_relaxed.setObjective(relaxed_v, grb.GRB.MINIMIZE)
        #     model_relaxed.feasRelaxS(0, True, True, False)  # Must be done after setting the objective.
        #     model_relaxed.optimize()
        #     vlb, refined, status_lb_r = get_grb_solution(model_relaxed, out_lb, max)
        #     del model_relaxed
        return vlb, refined, status_lb, status_lb_r

    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    out_lb, out_ub = v.LB, v.UB
    refine_time = time.time()
    neuron_refined = False
    if time.time() - mip_refine_time_start >= mip_refine_timeout:
        return out_lb, out_ub, False
    eps = 1e-5

    if abs(out_lb) < abs(out_ub):
        # lb is tighter, solve lb first.
        vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
        neuron_refined = neuron_refined or refined
        if vlb < 0:
            # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
            neuron_refined = neuron_refined or refined
        else:
            # lb > 0, neuron is stable, we skip solving ub.
            vub, status_ub, status_ub_r = out_ub, -1, -1
    else:
        # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
        neuron_refined = neuron_refined or refined
        if vub > 0:
            # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
            neuron_refined = neuron_refined or refined
        else:
            # ub < 0, neuron is stable, we skip solving ub.
            vlb, status_lb, status_lb_r = out_lb, -1, -1

    print("Solving MIP for {}, [{},{}]=>[{},{}] ({},{}; {},{}), time: {:.4f}s, #vars: {}, #constrs: {}, improved: {}".format(v.VarName, out_lb, out_ub, vlb, vub,
            status_lb, status_lb_r, status_ub, status_ub_r, time.time()-refine_time, model.NumVars, model.NumConstrs, neuron_refined))
    sys.stdout.flush()

    return vlb, vub, neuron_refined


# https://blog.mbedded.ninja/programming/languages/python/python-multiprocessing/
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def mip_solver_attack_init(flag):
    global termination_flag
    termination_flag = flag


def mip_solver_worker(model, input_shape, queue):
    """ Helper function for mip_solver_attack """
    input_shape = list(input_shape)
    try:
        model.optimize()
        if model.solcount > 0:
            objval = model.objval
            solution = torch.empty([1,] + input_shape[1:])
            # Extract the current adv example.
            _, C, H, W = input_shape
            for chan in range(C):
                for row in range(H):
                    for col in range(W):
                        v = model.getVarByName(f"inp_[{chan},{row},{col}]")
                        solution[0,chan,row,col] = v.X
        else:
            solution = torch.zeros([0,] + input_shape[1:])
            objval = float("inf")
        queue.put((model.status, objval, model.objbound, model.solcount, solution))
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)
        queue.put((-1, float("inf"), float("-inf"), 0, solution))
    # Make sure all results are sent back. This process will be killed.
    time.sleep(10)


def mip_solver_attack(new_splits):
    """Adversarial attack using MIP. A negative returned bound indicates attack success regardless of model status."""
    global termination_flag
    # Modify the current MIP model with split constraints.
    model = multiprocess_mip_model.copy()

    indices, relu_status, opt_var, input_shape, best_adv_input, relu_forward = new_splits
    input_shape = list(input_shape)

    for [layer_idx, neuron_idx], s in zip(indices, relu_status):
        a = model.getVarByName(f"aReLU{layer_idx}_{neuron_idx}")
        if s == 1:
            a.LB = 1
            a.UB = 1
        else:
            a.LB = 0
            a.UB = 0
    model.update()

    # Hint start.
    if best_adv_input is not None and len(best_adv_input) > 0:
        model.NumStart=best_adv_input.size(0)
        C, H, W = best_adv_input.shape[-3:]
        for s in range(best_adv_input.size(0)):
            model.params.StartNumber = s
            for chan in range(C):
                for row in range(H):
                    for col in range(W):
                        v = model.getVarByName(f"inp_[{chan},{row},{col}]")
                        v.Start = best_adv_input[s][chan][row][col]

            for relu_idx, relu in enumerate(relu_forward):
                # Each tensor has size (n_start, n_neurons), flattened.
                for i in range(relu.size(1)):
                    v = model.getVarByName(f"aReLU{relu_idx}_{i}")
                    if v is not None:
                        v.Start = int(relu[s][i]>0)

    v = model.getVarByName(opt_var)
    if termination_flag.value == 1:
        return float("inf"), float("inf"), -1, torch.zeros([0,] + input_shape[1:])  # Solver skipped.
    refine_time = time.time()
    model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
    model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
    model.setParam('OutputFlag', False)
    model.setParam('Heuristics', 0.5)
    model.setParam('MIPFocus', 1)
    model.setObjective(v, grb.GRB.MINIMIZE)
    # Run the solver asynchronously with a background process, we check the termination flag every second.
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=mip_solver_worker, args=(model, input_shape, queue,))
    p.start()
    while queue.empty():
        time.sleep(0.5)
        if termination_flag.value == 1:
            print('Early Stopping MIP solver because another process has succeeded.')
            p.terminate()
            time.sleep(0.5)
            p.kill()
            return float("inf"), float("inf"), -1, torch.zeros([0,] + input_shape[1:])
    status, objval, objbound, solcount, solution = queue.get()
    if objval < 0:
        # An adversarial example is found
        termination_flag.value = 1

    print("Solved MIP for {}, {} neurons fixed, status:{} ({} sols), lower={:.6f}, upper={:.6f}, time: {:.6f}s".format(v.VarName, len(indices), status,
                    solcount, objbound, objval, time.time()-refine_time))
    sys.stdout.flush()
    p.terminate()
    p.kill()
    # If res < 0, we guarantee attack success. If res > 0, it might be an upper bound (status == 9) or a lower bound (status == 15).
    return objval, objbound, status, solution


def mip_solver_lb(candidate):
    """ Solving MIP for adversarial attack/complete verification. """
    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    vlb = out_lb = v.LB
    vub = out_ub = v.UB
    global stop_multiprocess, mip_solve_time_start
    if stop_multiprocess:
        return out_lb, out_ub, -1  # Solver skipped.
    refine_time = time.time()
    model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
    model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
    model.setObjective(v, grb.GRB.MINIMIZE)
    try:
        model.optimize()
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)

    vlb = max(model.objbound, out_lb)
    if model.solcount > 0:
        vub = min(model.objval, out_ub)
    if vub < 0:
        # An adversarial example is found
        # print("stop: adv found!")
        stop_multiprocess = True

    assert model.status != 3, "should not be infeasible"
    print("solving MIP for {}, status:{}, [{}, {}]=>[{}, {}], time: {}s".format(v.VarName, model.status,
                    out_lb, out_ub, vlb, vub, time.time()-refine_time))
    sys.stdout.flush()
    if time.time() - mip_solve_time_start > arguments.Config["bab"]["timeout"]:
        stop_multiprocess = True
    return vlb, vub, model.status


def lp_solver(candidate):
    """ Multiprocess worker for solving LP model in update_the_model_lp for intermediate bound refinements """
    global stop_multiprocess, multiprocess_lp_model
    model = multiprocess_lp_model.copy()
    v = model.getVarByName(candidate)
    out_lb, out_ub = v.LB, v.UB
    refined = False
    refine_time = time.time()
    if out_lb>=0 or out_ub<=0:
        return out_lb, out_ub, time.time()-refine_time, refined
    if stop_multiprocess: 
        return out_lb, out_ub, time.time()-refine_time, refined
    
    model.setObjective(v, grb.GRB.MINIMIZE)
    model.update()
    model.reset()
    try:
        model.optimize()
    except grb.GurobiError as e: 
        handle_gurobi_error(e.message)
    if model.status == 2:
        #vlb = model.objval
        vlb = v.X
        refined = True
    else:
        print("Warning: other model status happens!")
        #assert model.status != 3, "should not be infeasible"
        vlb = out_lb

    model.setObjective(v, grb.GRB.MAXIMIZE)
    model.update()
    model.reset()
    try:
        model.optimize()
    except grb.GurobiError as e: 
        handle_gurobi_error(e.message)
    if model.status == 2:
        #vub = model.objval
        vub = v.X
        refined = True
    else:
        print("Warning: other model status happens!")
        #assert model.status != 3, "should not be infeasible"
        vub = out_ub

    print_str = "Linear {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}, time={:3g}".format(v.VarName, out_lb, vlb, out_ub, vub, vlb-out_lb, out_ub-vub, time.time()-refine_time)
    print(print_str)
    sys.stdout.flush()
    return vlb, vub, print_str, refined


def build_solver_model(m, lower_bounds, upper_bounds, timeout, mip_multi_proc=None, mip_threads=1, input_domain=None, target=None, model_type="mip", simplified=False):
    """
    m is the instance of LiRPAConvNet
    Before the first branching, we build the model and create a mask matrix
    model_type ["mip", "lp", "lp_integer"]: three different types of guorbi solvers
    Output: relu_mask, current intermediate upper and lower bounds, a list of
            indices of the layers right before a Relu layer
            the constructed gurobi model
    NOTE: we keep all bounds as a list of tensors from now on.
            Only lower and upper bounds are kept in the same shape as layers' outputs.
            Mask is linearized
            Gurobi_var lists are lineariezd
            m.model_lower_bounds and m.model_upper_bounds are kepts mainly for
            debugging purpose and could be removed
    """
    if m.pool is not None:
        # Must close the pool because the old model shared to the pool workers is now stale.
        print("Closing MIP Pool...")
        m.pool.close()
        m.pool.terminate()
        m.pool.kill()
        m.pool = None
        m.pool_termination_flag = None
    new_relu_mask = []
    input_domain = input_domain if input_domain is not None else m.input_domain
    input_domain = input_domain.cpu().numpy()

    if simplified:
        # When simplying, the last lower/upper bounds must have dimension 1.
        # Otherwise we need to know the target label.
        assert target is not None or len(lower_bounds[-1]) == 1

    # setting for aws instance
    # mip_multi_proc = 4
    # mip_threads = 4
    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    # Initialize the model
    m.model = grb.Model()
    m.model.setParam('OutputFlag', False)
    m.model.setParam('Threads', mip_threads)
    m.model.setParam("FeasibilityTol", 2e-5)
    m.model.setParam('TimeLimit', timeout)
    print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads}, total threads used: {mip_multi_proc*mip_threads}")
    build_mip_time = time.time()

    # keep a record of model's information
    m.gurobi_vars = []
    m.integer_vars = []
    m.relu_constrs = []
    m.relu_indices_mask = []

    ## Do the input layer, which is a special case
    inp_gurobi_vars = []
    zero_var = m.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    if input_domain.ndim == 2:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(input_domain):
            v = m.model.addVar(lb=lb, ub=ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        assert input_domain.ndim == 4
        for chan in range(input_domain.shape[0]):
            chan_vars = []
            for row in range(input_domain.shape[1]):
                row_vars = []
                for col in range(input_domain.shape[2]):
                    lb = input_domain[chan, row, col, 0]
                    ub = input_domain[chan, row, col, 1]
                    v = m.model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)

    m.gurobi_vars.append(inp_gurobi_vars)

    ## Do the other layers, computing for each of the neuron, its upper
    ## bound and lower bound
    # layer_idx = 0 is input layer
    layer_idx = 1
    relu_idx = 0
    for layer in m.layers:
        # print("build mip:", layer, relu_idx, layer_idx)
        new_layer_gurobi_vars = []
        if type(layer) is nn.Linear:
            last_layer = False
            if relu_idx == len(m.net.relus):
                last_layer = True
            # Get the better estimates from KW and Interval Bounds
            # print("linear", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape)
            out_lbs = lower_bounds[relu_idx].squeeze(0).cpu().numpy()
            out_ubs = upper_bounds[relu_idx].squeeze(0).cpu().numpy()
            this_layer_bias = layer.bias.detach().cpu().numpy()
            this_layer_weight = layer.weight.detach().cpu().numpy()
            # for neuron_idx in range(layer.weight.size(0)):
            for neuron_idx in range(out_lbs.shape[0]):
                if layer == m.layers[-1]:
                    if simplified:
                        # if target is not None, the target label info should be contained in m.c
                        assert out_lbs.shape[0] == 1, "should be simplified with only one output"
                        lin_expr = m.c[0].mm(layer.bias.unsqueeze(1)).item()
                        coeffs = m.c[0].mm(layer.weight).view(-1)
                    else:
                        lin_expr = this_layer_bias[m.pred] - this_layer_bias[neuron_idx]
                        coeffs = this_layer_weight[m.pred, :] - this_layer_weight[neuron_idx, :]
                else:
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                
                lin_expr += grb.LinExpr(coeffs, m.gurobi_vars[-1])
                out_lb = out_lbs[neuron_idx]
                out_ub = out_ubs[neuron_idx]
                v = m.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                        vtype=grb.GRB.CONTINUOUS,
                                        name=f'lay{layer_idx}_{neuron_idx}')
                m.model.addConstr(lin_expr == v, name=f'lay{layer_idx}_{neuron_idx}_eq')
                new_layer_gurobi_vars.append(v)
        
        elif type(layer) is nn.AvgPool2d: # I implement avgpool mostly follow the Conv2d, but with constant weights.
            value = 1.0/(layer.kernel_size[0] * layer.kernel_size[1])
            assert layer.padding[0] == layer.padding[1]
            padding = layer.padding[0]
            input_x, input_y = len(m.gurobi_vars[-1][0]), len(m.gurobi_vars[-1][0][0])
            output_x = output_y = (2 * padding + input_x - (layer.stride[0] - 1))//layer.stride[0]
            chan_num = len(m.gurobi_vars[-1])
            for out_chan_idx in range(chan_num):
                out_chan_vars = []
                for out_row_idx in range(output_x):
                    out_row_vars = []
                    for out_col_idx in range(output_y):
                        # print(layer.bias.shape, out_chan_idx, out_lbs.size(1))
                        lin_expr = 0.0
                        for ker_row_idx in range(layer.kernel_size[0]):
                            in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                            if (in_row_idx < 0) or (in_row_idx == len(m.gurobi_vars[-1][out_chan_idx][ker_row_idx])):
                                # This is padding -> value of 0
                                continue
                            for ker_col_idx in range(layer.kernel_size[1]):
                                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                if (in_col_idx < 0) or (in_col_idx == input_y):
                                    # This is padding -> value of 0
                                    continue
                                coeff = value
                                lin_expr += coeff * m.gurobi_vars[-1][out_chan_idx][in_row_idx][in_col_idx]
                        v = m.model.addVar(lb=-float('inf'), ub=float('inf'),
                                                obj=0, vtype=grb.GRB.CONTINUOUS,
                                                name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        m.model.addConstr(lin_expr == v, name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]_eq')

                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)
        
        elif type(layer) is nn.Conv2d:
            assert layer.dilation == (1, 1)
            gvars_array = np.array(m.gurobi_vars[-1])
            if relu_idx == 0:
                pre_lb_size = np.expand_dims(gvars_array, axis=0).shape
            else:
                pre_lb_size = lower_bounds[relu_idx - 1].size()
            this_layer_shape = lower_bounds[relu_idx].shape
            out_lbs = lower_bounds[relu_idx].cpu().numpy()
            out_ubs = upper_bounds[relu_idx].cpu().numpy()
            # print("conv", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape, layer.bias.shape)

            this_layer_bias = layer.bias.detach().cpu().numpy()
            this_layer_weight = layer.weight.detach().cpu().numpy()
            weight_shape2, weight_shape3 = layer.weight.shape[2], layer.weight.shape[3]
            padding0, padding1 = layer.padding[0], layer.padding[1]
            stride0, stride1 = layer.stride[0], layer.stride[1]
            for out_chan_idx in range(this_layer_shape[1]):
                out_chan_vars = []
                for out_row_idx in range(this_layer_shape[2]):
                    out_row_vars = []
                    for out_col_idx in range(this_layer_shape[3]):
                        # print(layer.bias.shape, out_chan_idx, out_lbs.size(1))
                        lin_expr = this_layer_bias[out_chan_idx]

                        for in_chan_idx in range(layer.weight.shape[1]):

                            # new version of conv layer for building mip by skipping kernel loops
                            ker_row_min, ker_row_max = 0, weight_shape2
                            in_row_idx_min = -padding0 + stride0 * out_row_idx
                            in_row_idx_max = in_row_idx_min + weight_shape2 - 1
                            if in_row_idx_min < 0:
                                ker_row_min = -in_row_idx_min
                            if in_row_idx_max >= pre_lb_size[2]:
                                ker_row_max = ker_row_max - in_row_idx_max + pre_lb_size[2] -1
                            in_row_idx_min, in_row_idx_max = max(in_row_idx_min, 0), min(in_row_idx_max, pre_lb_size[2] - 1)

                            ker_col_min, ker_col_max = 0, weight_shape3
                            in_col_idx_min = -padding1 + stride1 * out_col_idx
                            in_col_idx_max = in_col_idx_min + weight_shape3 - 1
                            if in_col_idx_min < 0:
                                ker_col_min = -in_col_idx_min
                            if in_col_idx_max >= pre_lb_size[3]:
                                ker_col_max = ker_col_max - in_col_idx_max + pre_lb_size[3] -1
                            in_col_idx_min, in_col_idx_max = max(in_col_idx_min, 0), min(in_col_idx_max, pre_lb_size[3] - 1)

                            coeffs = this_layer_weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)

                            gvars = gvars_array[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                            lin_expr += grb.LinExpr(coeffs, gvars)

                        out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx]
                        out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx]
                        v = m.model.addVar(lb=out_lb, ub=out_ub,
                                                obj=0, vtype=grb.GRB.CONTINUOUS,
                                                name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        m.model.addConstr(lin_expr == v, name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]_eq')

                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)

        elif type(layer) is nn.ReLU:
            new_relu_layer_constr = []
            this_relu = m.net.relus[relu_idx]
            if isinstance(m.gurobi_vars[-1][0], list):
                # This is convolutional relu
                pre_lbs = lower_bounds[relu_idx].squeeze(0).cpu().numpy()
                pre_ubs = upper_bounds[relu_idx].squeeze(0).cpu().numpy()
                new_layer_mask = []
                iv = []
                # print("conv relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                out_chan, out_height, out_width = pre_lbs.shape
                for chan_idx, channel in enumerate(m.gurobi_vars[-1]):
                    chan_vars = []
                    for row_idx, row in enumerate(channel):
                        row_vars = []
                        for col_idx, pre_var in enumerate(row):
                            pre_ub = pre_ubs[chan_idx, row_idx, col_idx]
                            pre_lb = pre_lbs[chan_idx, row_idx, col_idx]

                            if pre_lb >= 0:
                                # ReLU is always passing
                                v = pre_var
                                new_layer_mask.append(1)
                            elif pre_ub <= 0:
                                v = zero_var
                                new_layer_mask.append(0)
                            else:
                                lb = 0
                                ub = pre_ub
                                new_layer_mask.append(-1)
                                neuron_idx = col_idx + row_idx * out_width + chan_idx * out_height * out_width

                                v = m.model.addVar(ub=ub, lb=pre_lb,
                                                        obj=0,
                                                        vtype=grb.GRB.CONTINUOUS,
                                                        name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')

                                if model_type == "mip" or model_type == "lp_integer":
                                    # binary indicator
                                    if model_type == "mip":
                                        a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{relu_idx}_{neuron_idx}')
                                    elif model_type == "lp_integer":
                                        a = m.model.addVar(ub=1, lb=0, vtype=grb.GRB.CONTINUOUS, name=f'aReLU{relu_idx}_{neuron_idx}')
                                    iv.append(a)

                                    new_relu_layer_constr.append(
                                        m.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                            name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                                    new_relu_layer_constr.append(
                                        m.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                    new_relu_layer_constr.append(
                                        m.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_3'))
                                
                                elif model_type == "lp":
                                    new_relu_layer_constr.append(
                                        m.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                                    new_relu_layer_constr.append(m.model.addConstr(
                                        pre_ub * pre_var - (pre_ub - pre_lb) * v >= pre_ub * pre_lb,
                                        name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                
                                else:
                                    print(f"gurobi model type {model_type} not supported!")

                            row_vars.append(v)
                        chan_vars.append(row_vars)
                    new_layer_gurobi_vars.append(chan_vars)
            else:
                # this is linear relu
                pre_lbs = lower_bounds[relu_idx].squeeze(0).cpu().numpy()
                pre_ubs = upper_bounds[relu_idx].squeeze(0).cpu().numpy()
                # print("linear relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                new_layer_mask = []
                iv = []
                assert isinstance(m.gurobi_vars[-1][0], grb.Var)
                for neuron_idx, pre_var in enumerate(m.gurobi_vars[-1]):
                    pre_ub = pre_ubs[neuron_idx]
                    pre_lb = pre_lbs[neuron_idx]

                    if pre_lb >= 0:
                        # The ReLU is always passing
                        v = pre_var
                        new_layer_mask.append(1)
                    elif pre_ub <= 0:
                        v = zero_var
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                        new_layer_mask.append(0)
                    else:
                        lb = 0
                        ub = pre_ub
                        # post-relu var
                        v = m.model.addVar(ub=ub, lb=pre_lb,
                                                obj=0,
                                                vtype=grb.GRB.CONTINUOUS,
                                                name=f'ReLU{layer_idx}_{neuron_idx}')
                        if model_type == "mip" or model_type == "lp_integer":
                            # binary indicator
                            if model_type == "mip":
                                a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{relu_idx}_{neuron_idx}')
                            elif model_type == "lp_integer":
                                a = m.model.addVar(ub=1, lb=0, vtype=grb.GRB.CONTINUOUS, name=f'aReLU{relu_idx}_{neuron_idx}')
                            iv.append(a)

                            new_relu_layer_constr.append(
                                m.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                    name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(
                                m.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_2'))
                            new_relu_layer_constr.append(
                                m.model.addConstr(v >= 0, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_3'))
                        
                        elif model_type == "lp":
                            new_relu_layer_constr.append(
                                m.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(m.model.addConstr(
                                pre_ub * pre_var - (pre_ub - pre_lb) * v >= pre_ub * pre_lb,
                                name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                
                        else:
                            print(f"gurobi model type {model_type} not supported!") 

                        new_layer_mask.append(-1)
                    new_layer_gurobi_vars.append(v)

            if model_type in ["mip", "lp_integer"]: m.integer_vars.append(iv)
            new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
            m.relu_constrs.append(new_relu_layer_constr)
            relu_idx += 1
        
        elif type(layer) is nn.MaxPool2d:
            input_x, input_y = len(m.gurobi_vars[-1][0]), len(m.gurobi_vars[-1][0][0])
            assert layer.padding[0] == layer.padding[1]
            padding = layer.padding[0]
            output_x = output_y = (2 * padding + input_x - (layer.stride[0] - 1))//layer.stride[0]
            chan_num = len(m.gurobi_vars[-1])

            pre_ubs = layer(F.relu(upper_bounds[relu_idx-1].squeeze(0))).cpu().numpy()

            for out_chan_idx in range(chan_num):
                out_chan_vars = []
                for out_row_idx in range(output_x):
                    out_row_vars = []
                    for out_col_idx in range(output_y):
                        a_sum = 0.0
                        v = m.model.addVar(lb=-float('inf'), ub=float('inf'),
                                                obj=0, vtype=grb.GRB.CONTINUOUS,
                                                name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        for ker_row_idx in range(layer.kernel_size[0]):
                            in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                            if (in_row_idx < 0) or (in_row_idx == len(m.gurobi_vars[-1][out_chan_idx][ker_row_idx])):
                                # This is padding -> value of 0
                                continue
                            for ker_col_idx in range(layer.kernel_size[1]):
                                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                if (in_col_idx < 0) or (in_col_idx == input_y):
                                    # This is padding -> value of 0
                                    continue
                                var = m.gurobi_vars[-1][out_chan_idx][in_row_idx][in_col_idx]
                                a = m.model.addVar(vtype=grb.GRB.BINARY)
                                a_sum += a
                                m.model.addConstr(v >= var)
                                m.model.addConstr(v <= var + (1-a)*pre_ubs[out_chan_idx,out_row_idx,out_col_idx])
                        m.model.addConstr(a_sum == 1, name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]_eq')
                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)
        
        elif type(layer) == Flatten or "Flatten" in str(type(layer)):
            for chan_idx in range(len(m.gurobi_vars[-1])):
                for row_idx in range(len(m.gurobi_vars[-1][chan_idx])):
                    new_layer_gurobi_vars.extend(m.gurobi_vars[-1][chan_idx][row_idx])
        
        elif isinstance(layer, nn.ZeroPad2d):
            left, right, top, bottom = layer.padding
            num_out_chan = len(m.gurobi_vars[-1])
            num_out_row = len(m.gurobi_vars[-1][0]) + left + right
            num_out_col = len(m.gurobi_vars[-1][0][0]) + top + bottom

            for out_chan_idx in range(num_out_chan):
                out_chan_vars = []
                for out_row_idx in range(num_out_row):
                    out_row_vars = []
                    row_pad = out_row_idx < left or out_row_idx >= num_out_row - right
                    for out_col_idx in range(num_out_col):
                        col_pad = out_col_idx < top or out_col_idx >= num_out_col - bottom
                        if row_pad or col_pad:
                            v = m.model.addVar(lb=0, ub=0,
                                        obj=0, vtype=grb.GRB.CONTINUOUS,
                                        name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        else:
                            v = m.gurobi_vars[-1][out_chan_idx][out_row_idx - left][out_col_idx - top]
                        # print(out_chan_idx, out_row_idx, out_col_idx, row_pad, col_pad, v.LB, v.UB)

                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)
        
        elif "Pad" in str(type(layer)):
            new_layer_gurobi_vars = []
            assert layer.padding[0] == layer.padding[1]
            padding = layer.padding[0]
            for chan_indx in range(len(m.gurobi_vars[-1])):
                out_chan_vars = []
                for i in range(padding):
                    out_row_vars = []
                    for j in range(padding*2 + len(m.gurobi_vars[-1][chan_indx][row_idx])):
                        out_row_vars.append(layer.value)
                    out_chan_vars.append(out_row_vars)
                for row_idx in range(len(m.gurobi_vars[-1][chan_indx])):
                    out_row_vars = []
                    for i in range(padding):
                        out_row_vars.append(layer.value)
                    for i in range(len(m.gurobi_vars[-1][chan_indx][row_idx])):
                        out_row_vars.append(m.gurobi_vars[-1][chan_indx][row_idx][i])
                    for i in range(padding):
                        out_row_vars.append(layer.value)
                    out_chan_vars.append(out_row_vars)
                for i in range(padding):
                    out_row_vars = []
                    for j in range(padding*2 + len(m.gurobi_vars[-1][chan_indx][row_idx])):
                        out_row_vars.append(layer.value)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)
        
        else:
            print("{} is not implemented".format(type(layer)))
            raise NotImplementedError

        m.gurobi_vars.append(new_layer_gurobi_vars)

        layer_idx += 1
    
    m.model.update()
    # save the indices for undecided relu-nodes
    m.relu_indices_mask = [(i == -1).nonzero().view(-1).tolist() for i in new_relu_mask]
    print("build_mip_time:", time.time()-build_mip_time)


def build_the_model_lp(m, lower_bounds, upper_bounds, using_integer=True, simplified=True):
    """
    Before the first branching, we build the model and create a mask matrix
    Output: relu_mask, current intermediate upper and lower bounds, a list of
            indices of the layers right before a Relu layer
            the constructed gurobi model
    NOTE: we keep all bounds as a list of tensors from now on.
            Only lower and upper bounds are kept in the same shape as layers' outputs.
            Mask is linearized
            Gurobi_var lists are lineariezd
            m.model_lower_bounds and m.model_upper_bounds are kepts mainly for
            debugging purpose and could be removed
            using_integer is True means to use MIP formulation but allow integer z to be 
            floating numbers from 0 to 1 which is equivalent to LP formulation
    """
    timeout = arguments.Config["bab"]["timeout"]
    model_type = "lp"
    if using_integer: model_type = "lp_integer"
    m.build_solver_model(lower_bounds, upper_bounds, timeout, model_type=model_type, simplified=simplified)

    # Assert that this is as expected a network with a single output
    assert simplified and len(m.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

    m.model.update()
    print('Finished building Gurobi LP model. Start solving the LP!')

    guro_start = time.time()

    m.gurobi_vars[-1][0].LB = -100000
    m.gurobi_vars[-1][0].UB = 100000
    m.model.setObjective(m.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
    # m.model.write("save.lp")
    try:
        m.model.optimize()
    except grb.GurobiError as e: 
        handle_gurobi_error(e.message)
    
    # for c in m.model.getConstrs():
    #     print('The dual value of %s : %g %g'%(c.constrName,c.pi, c.slack))

    assert m.model.status == 2, f"LP wasn't optimally solved status:{m.model.status}"

    guro_end = time.time()
    print('Gurobi solved the LP with time', guro_end - guro_start)

    glb = m.gurobi_vars[-1][0].X
    # lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)
    print("gurobi glb:", glb)

    # get the primal values for each layer from gurobi lp model
    primal_vars = []
    # mnist6_100 case:
    # primal_vars[0]: 784, input's primal values
    # primal_vars[1]: 100, z_hat (pre relu) primal value
    # primal_vars[2]: 100, z (post relu) primal value
    # primal_vars[11]: 1, last output primal value
    for layer_idx, vars in enumerate(m.gurobi_vars[1:]):
        # print(layer_idx, len(vars))
        pv = []
        if not isinstance(vars[0], list):
            for i in range(len(vars)):
                pv.append(vars[i].X)
        else:
            for chan in range(len(vars)):
                for row in range(len(vars[chan])):
                    for col in range(len(vars[chan][row])):
                        pv.append(vars[chan][row][col].X)
        # print(len(pv), pv, sum(pv), max(pv), min(pv))
        primal_vars.append(pv)
        # if layer_idx>=4: break
        # np.save(f"gurobi_primals/{layer_idx}.npy", np.array(pv))
    
    if using_integer:
        # get integer variables for unstable relu neurons in each relu layer
        integer_vars = []
        for relu_idx, vars in enumerate(m.integer_vars):
            iv = []
            for i in range(len(vars)):
                iv.append(vars[i].X)
            # np.save(f"gurobi_primals/z_relu{relu_idx}.npy", np.array(iv))
            integer_vars.append(iv)
        
    input_primal_gurobi = primal_vars[0]
    print("### Extracting primal values from gurobi lp model done ###")
    # m.solve_diving_lp(primal_vars, integer_vars, lower_bounds, upper_bounds)

    # get the primal input and calculate gub
    # inp_size = lower_bounds[0].size()
    # mini_inp = torch.zeros(inp_size).to(lower_bounds[0].device)
    # if len(inp_size) == 1:
    #     # This is a linear input.
    #     for i in range(inp_size[0]):
    #         mini_inp[i] = m.gurobi_vars[0][i].x
    # elif len(inp_size) == 0:
    #     mini_inp.data = torch.tensor(m.gurobi_vars[0][0].x).cuda()
    # else:
    #     for i in range(inp_size[0]):
    #         for j in range(inp_size[1]):
    #             for k in range(inp_size[2]):
    #                 mini_inp[i, j, k] = m.gurobi_vars[0][i][j][k].x
    # gub = m.net(mini_inp.unsqueeze(0)).item()
    # print("gub:", mini_inp, gub)

    # flatten high-dimensional gurobi var lists
    for l_idx, layer in enumerate(m.layers):
        if type(layer) is nn.Conv2d:
            flattened_gurobi = []
            for chan_idx in range(len(m.gurobi_vars[l_idx + 1])):
                for row_idx in range(len(m.gurobi_vars[l_idx + 1][chan_idx])):
                    flattened_gurobi.extend(m.gurobi_vars[l_idx + 1][chan_idx][row_idx])
            m.gurobi_vars[l_idx + 1] = flattened_gurobi
            if type(m.layers[l_idx + 1]) is nn.ReLU:
                flattened_gurobi = []
                for chan_idx in range(len(m.gurobi_vars[l_idx + 2])):
                    for row_idx in range(len(m.gurobi_vars[l_idx + 2][chan_idx])):
                        flattened_gurobi.extend(m.gurobi_vars[l_idx + 2][chan_idx][row_idx])
                m.gurobi_vars[l_idx + 2] = flattened_gurobi
        else:
            continue
    return glb


@torch.no_grad()
def build_the_model_mip(m, lower_bounds, upper_bounds, simplified=False, labels_to_verify=None):
    """
    Using the built gurobi model to solve mip formulation in parallel
    lower_bounds, upper_bounds: intermediate relu bounds from beta-crown
    simplified: only for target label if simplified, otherwise all labels remained
    Output: gurobi mip model solving lb and status
    """
    timeout = arguments.Config["bab"]["timeout"]
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]

    m.build_solver_model(lower_bounds, upper_bounds, timeout, mip_multi_proc, mip_threads)

    if simplified:
        # Assert that this is as expected a network with a single output
        assert len(m.gurobi_vars[-1]) == 1, "Network doesn't have scalar output if simplified"

        m.model.update()
        print('finished building Gurobi MIP model, calling optimize function')
        guro_start = time.time()
        # m.model.setParam("PreSolve", 0)
        # m.model.setParam("Method", 1)
        # m.model.setParam("FeasibilityTol", 2e-5)

        m.gurobi_vars[-1][0].LB = -100000
        m.gurobi_vars[-1][0].UB = 100000
        m.model.setObjective(m.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        # m.model.write("save.mip")
        try:
            m.model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        # for c in m.model.getConstrs():
        #     print('The dual value of %s : %g %g'%(c.constrName,c.pi, c.slack))

        assert m.model.status == 2, f"LP wasn't optimally solved status:{m.model.status}"

        guro_end = time.time()
        print('Gurobi solved the MIP with ', guro_end - guro_start, "seconds")

        glb = m.gurobi_vars[-1][0].X
        lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)
        print("gurobi glb:", glb)

        # record model information
        # indices for undecided relu-nodes
        m.relu_indices_mask = [(i == -1).nonzero().view(-1).tolist() for i in new_relu_mask]
        # flatten high-dimensional gurobi var lists
        for l_idx, layer in enumerate(m.layers):
            if type(layer) is nn.Conv2d:
                flattened_gurobi = []
                for chan_idx in range(len(m.gurobi_vars[l_idx + 1])):
                    for row_idx in range(len(m.gurobi_vars[l_idx + 1][chan_idx])):
                        flattened_gurobi.extend(m.gurobi_vars[l_idx + 1][chan_idx][row_idx])
                m.gurobi_vars[l_idx + 1] = flattened_gurobi
                if type(m.layers[l_idx + 1]) is nn.ReLU:
                    flattened_gurobi = []
                    for chan_idx in range(len(m.gurobi_vars[l_idx + 2])):
                        for row_idx in range(len(m.gurobi_vars[l_idx + 2][chan_idx])):
                            flattened_gurobi.extend(m.gurobi_vars[l_idx + 2][chan_idx][row_idx])
                    m.gurobi_vars[l_idx + 2] = flattened_gurobi
            else:
                continue
        return glb
    else:
        # not simplified directly after opt crown init bounds
        m.model.update()
        print('finished building Gurobi MIP model, calling optimize function')
        lb = lower_bounds[-1][0]
        ub = upper_bounds[-1][0]
        print('lower bounds for all target labels:', lb)
        candidates, candidate_neuron_ids = [], []
        if labels_to_verify is not None: # sort the labels
            for pidx in labels_to_verify:
                if lb[pidx] >= 0: continue # skip the label with intial bound >= 0
                candidates.append(m.gurobi_vars[-1][pidx].VarName)
                candidate_neuron_ids.append(pidx)
        else:
            for pidx, lbi in enumerate(lb):
                if lbi >= 0: continue
                candidates.append(m.gurobi_vars[-1][pidx].VarName)
                candidate_neuron_ids.append(pidx)
            # SINGLE THREAD
            # mip_time = time.time()
            # m.model.setObjective(m.gurobi_vars[-1][pidx], grb.GRB.MINIMIZE)
            # m.model.optimize()
            # assert m.model.status == 2, f"status: {m.model.status}"

            # glb = m.gurobi_vars[-1][pidx].X
            # print(f"mip: label {m.pred} target label {pidx}, orig {lbi}, mip {glb}, mip time {time.time()-mip_time}")
            # lb[pidx] = glb
            # if glb<0: break

        # Solve the worst label first.
        # candidates, candidate_neuron_ids = zip(*sorted(zip(candidates, candidate_neuron_ids), key=lambda x: lb[x[1]]))
        print('Starting MIP solver for these labels:', candidate_neuron_ids)

        # MULTITHREAD
        global multiprocess_mip_model, stop_multiprocess
        multiprocess_mip_model = m.model
        global mip_solve_time_start
        mip_solve_time_start = time.time()
        with multiprocessing.Pool(mip_multi_proc) as pool:
            solver_result = pool.map(mip_solver_lb, candidates, chunksize=1)
        multiprocess_mip_model = None
        stop_multiprocess = False

        status = [-1 for i in lb]
        for (vlb, vub, s), pidx in zip(solver_result, candidate_neuron_ids):
            lb[pidx] = vlb
            ub[pidx] = vub
            status[pidx] = s
        return lb, ub, status

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


def copy_model(model, basis=True, use_basis_warm_start=True, remove_constr_list=[]):
    """
    deep copy a gurobi model together with variable historical results
    """
    model_split = model.copy()

    # print(model_split.printStats())
    for rc in remove_constr_list:
        rcs = model_split.getConstrByName(rc.ConstrName)
        model_split.remove(rcs)
    model_split.update()

    if not basis:
        return model_split

    xvars = model.getVars()
    svars = model_split.getVars()
    # print(len(xvars), len(svars))
    for x, s in zip(xvars, svars):
        if use_basis_warm_start:
            s.VBasis = x.VBasis
        else:
            s.PStart = x.X

    xconstrs = model.getConstrs()
    sconstrs = model_split.getConstrs()
    # print(len(xconstrs), len(sconstrs))

    for s in sconstrs:
        x = model.getConstrByName(s.ConstrName)
        if use_basis_warm_start:
            s.CBasis = x.CBasis
        else:
            s.DStart = x.Pi
    model_split.update()
    return model_split


def compute_ratio(lower_bound, upper_bound):
    """
    helper function to calculate fsb score
    """
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio, intercept


def get_branching_op(branching_reduceop):
    """
    helper function to match reduce_op
    """
    if branching_reduceop == 'min':
        reduce_op = torch.min
    elif branching_reduceop == 'max':
        reduce_op = torch.max
    elif branching_reduceop == 'mean':
        reduce_op = torch.mean
    else:
        reduce_op = None
    return reduce_op


def FSB_score(net, lower_bounds, upper_bounds, orig_mask, pre_relu_indices, lAs, branching_candidates=5, 
            branching_reduceop='min', slopes=None):
    """
    Use FSB to sort the order for mip refinement
    """
    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = branching_candidates

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                    upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

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
            # print(input_node.inputs)
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
        score.insert(0, abs(score_candidate).view(batch, -1) * mask[relu_idx])

        relu_idx -= 1
    
    return score


def update_the_model_lp(m, lower_bounds, upper_bounds, decision, choice, using_integer=True, refine_intermediate_neurons=False):
    """
    The model updates upper and lower bounds after introducing a relu constraint and then update the gurobi model
    using these updated bounds
    input:
    lower_bounds, upper_bounds: lower and upper bounds of the parent domain
    decision: the index of the node where we make branching decision
    choice: force no-passing constraint (0) or all passing constraint (1)
    using_integer: use mip formulation but integer a is continous in [0, 1], same as lp results
    refine_intermediate_neurons: use lp solver to refine each intermediate neuron bounds, match with beta crown intermediate refinement
    output: lower bound from lp, refined lower bound if refine_intermediate_neurons
    """
    print("decision: {}, choice: {}".format(decision, choice))

    m.model_split = copy_model(m.model)

    relu_idx = 0
    # save the split relu layer number in m.replacing_bd_index
    for layer_idx, layer in enumerate(m.layers):
        if type(layer) is nn.ReLU:
            if relu_idx == decision[0]:
                m.replacing_bd_index = layer_idx
                break
            relu_idx+=1

    # m.replacing_bd_index = m.pre_relu_indices[decision[0]]
    # reintroduce ub and lb for gurobi constraints
    introduced_constrs = []
    rep_index = m.replacing_bd_index
    rep_relu_idx = decision[0]

    # build the lp model in gurobi
    for layer in m.layers[m.replacing_bd_index - 1:]:
        print(m.replacing_bd_index, layer)
        if type(layer) is nn.Linear:
            # print("linear", rep_index, rep_relu_idx, upper_bounds[rep_relu_idx].shape)
            for idx, var in enumerate(m.gurobi_vars[rep_index]):
                svar = m.model_split.getVarByName(var.VarName)
                svar.ub = upper_bounds[rep_relu_idx][0, idx].item()
                svar.lb = lower_bounds[rep_relu_idx][0, idx].item()

        elif type(layer) is nn.Conv2d:
            conv_ub = upper_bounds[rep_relu_idx].reshape(-1)
            conv_lb = lower_bounds[rep_relu_idx].reshape(-1)
            # print("conv", rep_index, rep_relu_idx, conv_lb.shape)
            for idx, var in enumerate(m.gurobi_vars[rep_index]):
                svar = m.model_split.getVarByName(var.VarName)
                svar.ub = conv_ub[idx].item()
                svar.lb = conv_lb[idx].item()

        elif type(layer) is nn.ReLU:
            # list to store integer variables for unstable relu nodes
            iv = []
            # locate relu index and remove all associated constraints
            pre_lbs = lower_bounds[rep_relu_idx].reshape(-1)
            pre_ubs = upper_bounds[rep_relu_idx].reshape(-1)
            # print("relu", rep_index, rep_relu_idx, pre_lbs.shape)
            for unstable_idx, unmasked_idx in enumerate(m.relu_indices_mask[rep_relu_idx]):
                pre_lb = pre_lbs[unmasked_idx].item()
                pre_ub = pre_ubs[unmasked_idx].item()
                var = m.gurobi_vars[rep_index][unmasked_idx]
                svar = m.model_split.getVarByName(var.VarName)
                pre_var = m.gurobi_vars[rep_index - 1][unmasked_idx]
                pre_svar = m.model_split.getVarByName(pre_var.VarName)

                if pre_lb >= 0 and pre_ub >= 0:
                    # ReLU is always passing
                    svar.lb = pre_lb
                    svar.ub = pre_ub
                    introduced_constrs.append(m.model_split.addConstr(pre_svar == svar))
                elif pre_lb <= 0 and pre_ub <= 0:
                    svar.lb = 0
                    svar.ub = 0
                else:
                    svar.lb = 0
                    svar.ub = pre_ub
                    if using_integer:
                        z = m.integer_vars[rep_relu_idx][unstable_idx]
                        sz = m.model_split.getVarByName(z.VarName)
                        # sz = m.model_split.getVarByName(f'z_ReLU{rep_index - 1}_{unmasked_idx}')
                        introduced_constrs.append(
                            m.model_split.addConstr(pre_svar - pre_lb * (1 - sz) >= svar))
                        introduced_constrs.append(
                            m.model_split.addConstr(svar >= pre_svar))
                        introduced_constrs.append(
                            m.model_split.addConstr(pre_ub * sz >= svar))
                        iv.append(sz)
                    else:
                        introduced_constrs.append(m.model_split.addConstr(svar >= pre_svar))
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        introduced_constrs.append(m.model_split.addConstr(svar <= slope * pre_svar + bias))
            
            rep_relu_idx += 1

        elif type(layer) is Flatten:
            pass
        else:
            raise NotImplementedError
        m.model_split.update()
        rep_index += 1

    # compute optimum
    assert len(m.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
    target_var = m.gurobi_vars[-1][0]
    target_svar = m.model_split.getVarByName(target_var.VarName)

    lp_time = time.time()

    m.model_split.update()
    # m.model.reset()
    m.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
    # m.model.setObjective(0, grb.GRB.MINIMIZE)
    m.model_split.optimize()
    # assert m.model.status == 2, "LP wasn't optimally solved"
    check_optimization_success(m.model_split, [introduced_constrs])

    glb = target_svar.X
    init_lp_glb = glb
    print(f"glb: {glb}, time: {time.time()-lp_time}")
    lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)

    # get the primal values for each layer from gurobi lp model
    primal_vars = []
    # mnist6_100 case:
    # primal_vars[0]: 784, input's primal values
    # primal_vars[1]: 100, z_hat (pre relu) primal value
    # primal_vars[2]: 100, z (post relu) primal value
    # primal_vars[11]: 1, last output primal value
    for layer_idx, vars in enumerate(m.gurobi_vars[1:]):
        # print(layer_idx, len(vars))
        pv = []
        if not isinstance(vars[0], list):
            for i in range(len(vars)):
                svar = m.model_split.getVarByName(vars[i].VarName)
                pv.append(svar.X)
        else:
            print("conv vars should be flattened at the end of build_the_model_lp, wrong!")
            exit()
            for chan in range(len(vars)):
                for row in range(len(vars[chan])):
                    for col in range(len(vars[chan][row])):
                        svar = m.model_split.getVarByName(vars[chan][row][col].VarName)
                        pv.append(svar.X)
        # print(len(pv), pv, sum(pv), max(pv), min(pv))
        primal_vars.append(pv)
        # if layer_idx>=4: break
        # np.save(f"gurobi_primals/c{choice}_{layer_idx}.npy", np.array(pv))
    
    if using_integer:
        # get the integer values for unstable relus for each relu layer
        integer_vars = []
        for relu_idx, vars in enumerate(m.integer_vars):
            iv = []
            for i in range(len(vars)):
                sz = m.model_split.getVarByName(vars[i].VarName)
                iv.append(sz.X)
            # np.save(f"gurobi_primals/c{choice}_z_relu{relu_idx}.npy", np.array(iv))
            integer_vars.append(iv)
        
    input_primal_gurobi = primal_vars[0]
    print("### Extracting primal values from gurobi lp model done ###")

    n_test_points = -1  # Test n_test_points neurons per layer.
    n_test_points = 3
    check_final_lp = True
    single = False 
    global multiprocess_lp_model
    lp_refine_multi_proc = multiprocessing.cpu_count()
    lp_threads = 1
    if n_test_points != -1:
        lp_refine_multi_proc = min(lp_refine_multi_proc, n_test_points)
    m.model_split.setParam('Threads', lp_threads)
    print(f"lp_refine_multi_proc: {lp_refine_multi_proc}, lp_threads: {lp_threads},"
            f"total threads used: {lp_refine_multi_proc*lp_threads}")

    refined_lp_glb = None
    if refine_intermediate_neurons:
        import datetime
        prefix = datetime.datetime.now().strftime('%m%d_%H%M%S')
        updated_svar, updated_lp_lbs, updated_lp_ubs = [], [], []
        relu_idx = 0
        for layer_idx, layer in enumerate(m.layers):
            if layer_idx == len(m.layers) - 1: break
            #if layer_idx != len(m.layers)-1: continue
            rep_index = layer_idx+1
            if type(layer) is nn.Linear:
                l = open(f"log_{prefix}_lp_dense_{layer_idx}.log", "w")
                # print(rep_index, len(m.gurobi_vars[rep_index]), upper_bounds[rep_index].reshape(-1).shape)
                # print(rep_index+1, len(m.gurobi_vars[rep_index+1]), upper_bounds[rep_index+1].reshape(-1).shape)
                # continue
                dense_ub = upper_bounds[relu_idx].reshape(-1)
                dense_lb = lower_bounds[relu_idx].reshape(-1)
                new_lb_bounds, new_up_bounds = [], []
                if n_test_points != -1:
                    # select some neurons for testing
                    selected_index = np.random.permutation(len(m.gurobi_vars[rep_index]))[:n_test_points]
                    print('selected neuron for testing: ', selected_index)
                    selected_neurons = [m.gurobi_vars[rep_index][si] for si in selected_index]
                else:
                    selected_neurons = m.gurobi_vars[rep_index]
                    selected_index = list(range(len(m.gurobi_vars[rep_index])))
                if single:
                    for idx, var in zip(selected_index, selected_neurons):
                        lp_refine_time = time.time()
                        svar = m.model_split.getVarByName(var.VarName)

                        if svar.UB<=0 or svar.LB>=0: continue

                        # svar.ub = dense_ub[idx].item()
                        # svar.lb = dense_lb[idx].item()
                        rtime = time.time()
                        m.model_split.setObjective(svar, grb.GRB.MINIMIZE)
                        m.model_split.update()
                        m.model_split.reset()
                        m.model_split.optimize()
                        glb = svar.X

                        m.model_split.setObjective(svar, grb.GRB.MAXIMIZE)
                        m.model_split.update()
                        m.model_split.reset()
                        m.model_split.optimize()
                        gub = svar.X

                        new_lb_bounds.append(glb)
                        new_up_bounds.append(gub)

                        ### use this line to update intermediate bounds for solving later nodes
                        svar.ub, svar.lb = gub, glb

                        print("Linear {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}, time={:3g}".format(var.VarName, dense_lb[idx].item(), glb, dense_ub[idx].item(), gub, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub, time.time()-lp_refine_time))
                        l.write("Linear {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}, time={:.3g}\n".format(var.VarName, dense_lb[idx].item(), glb, dense_ub[idx].item(), gub, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub, time.time()-lp_refine_time))
                        l.flush()
                        # l.write("{}: {}, {}\n".format(var, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub))
                        if check_final_lp:
                            updated_svar.append(svar)
                            updated_lp_lbs.append(glb)
                            updated_lp_ubs.append(gub)
                else:
                    m.model_split.update()
                    multiprocess_lp_model = m.model_split
                    with multiprocessing.Pool(lp_refine_multi_proc) as pool:
                        solver_result = pool.map(lp_solver, [var.VarName for var in selected_neurons], chunksize=1)

                    for (vlb, vub, print_str, refined), var in zip(solver_result, selected_neurons):
                        if not refined: continue
                        svar = m.model_split.getVarByName(var.VarName)
                        new_lb_bounds.append(vlb)
                        new_up_bounds.append(vub)
                        l.write(print_str+"\n")
                        l.flush()

                        ### use this line to update intermediate bounds for solving later nodes
                        svar.ub, svar.lb = vub, vlb
                        m.model_split.update()

                        if check_final_lp:
                            updated_svar.append(svar)
                            updated_lp_lbs.append(vlb)
                            updated_lp_ubs.append(vub)

                l.close()
                # m.model_split.update()

            elif type(layer) is nn.Conv2d:
                if n_test_points != -1:
                    # select some neurons for testing
                    selected_index = np.random.permutation(len(m.gurobi_vars[rep_index]))[:n_test_points]
                    print('selected neuron for testing: ', selected_index)
                    selected_neurons = [m.gurobi_vars[rep_index][si] for si in selected_index]
                else:
                    selected_neurons = m.gurobi_vars[rep_index]
                    selected_index = list(range(len(m.gurobi_vars[rep_index])))
                    # continue  
                l = open(f"log_{prefix}_lp_conv_{layer_idx}.log", "w")
                ##### Skip conv layers if add this continue #####
                # continue
                conv_ub = upper_bounds[relu_idx].reshape(-1)
                conv_lb = lower_bounds[relu_idx].reshape(-1)
                new_lb_bounds, new_up_bounds = [], []
                if single:
                    for idx, var in zip(selected_index, selected_neurons):
                        svar = m.model_split.getVarByName(var.VarName)
                        # print(svar, conv_lb[idx].item()-svar.lb, svar.ub-conv_ub[idx].item())
                        # svar.ub = conv_ub[idx].item()
                        # svar.lb = conv_lb[idx].item()
                        m.model_split.setObjective(svar, grb.GRB.MINIMIZE)
                        m.model_split.update()
                        m.model_split.optimize()
                        glb = svar.X

                        m.model_split.setObjective(svar, grb.GRB.MAXIMIZE)
                        m.model_split.update()
                        m.model_split.optimize()
                        gub = svar.X

                        new_lb_bounds.append(glb)
                        new_up_bounds.append(gub)

                        print("Conv {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}".format(var.VarName, conv_lb[idx].item(), glb, conv_ub[idx].item(), gub, glb-conv_lb[idx].item(), conv_ub[idx].item()-gub))
                        # print(var, glb-conv_lb[idx].item(), conv_ub[idx].item()-gub)
                        l.write("Conv {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}\n".format(var.VarName, conv_lb[idx].item(), glb, conv_ub[idx].item(), gub, glb-conv_lb[idx].item(), conv_ub[idx].item()-gub))
                        l.flush()
                        if check_final_lp:
                            updated_svar.append(svar)
                            updated_lp_lbs.append(glb)
                            updated_lp_ubs.append(gub)
                            ### use this line to update intermediate bounds for solving later nodes
                            svar.ub, svar.lb = gub, glb
                else:
                    m.model_split.update()
                    multiprocess_lp_model = m.model_split
                    with multiprocessing.Pool(lp_refine_multi_proc) as pool:
                        solver_result = pool.map(lp_solver, [var.VarName for var in selected_neurons], chunksize=1)

                    for (vlb, vub, print_str, refined), var in zip(solver_result, selected_neurons):
                        if not refined: continue
                        svar = m.model_split.getVarByName(var.VarName)
                        new_lb_bounds.append(vlb)
                        new_up_bounds.append(vub)
                        l.write(print_str+"\n")
                        l.flush()

                        ### use this line to update intermediate bounds for solving later nodes
                        svar.ub, svar.lb = vub, vlb
                        m.model_split.update()

                        if check_final_lp:
                            updated_svar.append(svar)
                            updated_lp_lbs.append(vlb)
                            updated_lp_ubs.append(vub)
                    
                l.close()
                # m.model_split.update()

            elif type(layer) is nn.ReLU:
                # locate relu index and remove all associated constraints
                for unmasked_idx in m.relu_indices_mask[relu_idx]:
                    var = m.gurobi_vars[layer_idx+1][unmasked_idx]
                    svar = m.model_split.getVarByName(var.VarName)
                    pre_var = m.gurobi_vars[layer_idx][unmasked_idx]
                    pre_svar = m.model_split.getVarByName(pre_var.VarName)
                    pre_lb, pre_ub = pre_svar.lb, pre_svar.ub

                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        svar.lb = pre_lb
                        svar.ub = pre_ub
                        introduced_constrs.append(m.model_split.addConstr(pre_svar == svar))
                    elif pre_lb <= 0 and pre_ub <= 0:
                        svar.lb = 0
                        svar.ub = 0
                    else:
                        svar.lb = 0
                        svar.ub = pre_ub
                        introduced_constrs.append(m.model_split.addConstr(svar >= pre_svar))
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        introduced_constrs.append(m.model_split.addConstr(svar <= slope * pre_svar + bias))

                relu_idx += 1
            else:
                pass

        if check_final_lp:
            final_lp_time = time.time()
            for si, svar in enumerate(updated_svar):
                # print(svar.VarName, svar.lb, svar.ub, updated_lp_ubs[si], updated_lp_lbs[si])
                svar.ub, svar.lb = updated_lp_ubs[si], updated_lp_lbs[si]
            print(f"total updated intermediate node bounds: {len(updated_svar)}")
            m.model_split.update()
            m.model_split.reset()
            '''
            # if we did not update the relu bounds in previous step, we should update relu constraints here
            relu_idx = 0
            for layer_idx, layer in enumerate(m.layers):
                if type(layer) is nn.ReLU:
                    for unmasked_idx in m.relu_indices_mask[relu_idx]:
                        var = m.gurobi_vars[layer_idx+1][unmasked_idx]
                        svar = m.model_split.getVarByName(var.VarName)
                        pre_var = m.gurobi_vars[layer_idx][unmasked_idx]
                        pre_svar = m.model_split.getVarByName(pre_var.VarName)
                        pre_lb, pre_ub = pre_svar.lb, pre_svar.ub

                        if pre_lb >= 0 and pre_ub >= 0:
                            # ReLU is always passing
                            svar.lb = pre_lb
                            svar.ub = pre_ub
                            introduced_constrs.append(m.model_split.addConstr(pre_svar == svar))
                        elif pre_lb <= 0 and pre_ub <= 0:
                            svar.lb = 0
                            svar.ub = 0
                        else:
                            svar.lb = 0
                            svar.ub = pre_ub
                            introduced_constrs.append(m.model_split.addConstr(svar >= pre_svar))
                            slope = pre_ub / (pre_ub - pre_lb)
                            bias = - pre_lb * slope
                            introduced_constrs.append(m.model_split.addConstr(svar <= slope * pre_svar + bias))

                        # m.model_split.update()
                        # print(pre_svar, pre_svar.lb, pre_svar.ub, pre_lb, pre_ub)
                        # print(svar, svar.lb, svar.ub)
                        # exit()
                    relu_idx += 1
            '''
            m.model_split.update()
            # m.model.reset()
            m.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
            # m.model.setObjective(0, grb.GRB.MINIMIZE)
            m.model_split.optimize()
            # assert m.model.status == 2, "LP wasn't optimally solved"

            if m.model_split.status == 2:
                glb = target_svar.X
            elif m.model_split.status == 3:
                print("infeasible!!!")
                glb = init_lp_glb
            else:
                print("Warning: model status", m.model_split.status)
                glb = init_lp_glb
            refined_lp_glb = glb
            print(f"new glb: {glb}, final lp time: {time.time()-final_lp_time}")
            l = open(f"log_{prefix}_lp_final.log", "w")
            l.write(f"\n***** new glb: {glb} *****\n\n")
            l.close()

        # exit()
    del m.model_split
    return init_lp_glb, refined_lp_glb


def update_the_model_mip(m, relu_mask, lower_bounds, upper_bounds, decision, choice):
    """
    The model updates upper and lower bounds after introducing a relu constraint and then update the gurobi model
    using these updated bounds
    input:
    relu_mask: the copied mask of the parent domain,
    pre_lb, pre_ub: lower and upper bounds of the parent domain
    decision: the index of the node where we make branching decision
    choice: force no-passing constraint (0) or all passing constraint (1)
    pre_relu_indices: indices of bounds that the layers prior to a relu_layer
    output: global lower bound, updated mask, updated lower and upper bounds
    """
    print("decision: {}, choice: {}".format(decision, choice))

    # m.model_split = copy_model(m.model)
    m.model_split = m.model.copy()
    # m.model_split.write("mip_save.lp")

    m.replacing_bd_index = m.pre_relu_indices[decision[0]]

    # reintroduce ub and lb for gurobi constraints
    introduced_constrs = []
    rep_index = m.replacing_bd_index
    for layer in m.layers[m.replacing_bd_index - 1:]:
        if type(layer) is nn.Linear:
            for idx, var in enumerate(m.gurobi_vars[rep_index]):
                svar = m.model_split.getVarByName(var.VarName)
                svar.ub = upper_bounds[rep_index][idx].item()
                svar.lb = lower_bounds[rep_index][idx].item()

        elif type(layer) is nn.Conv2d:
            conv_ub = upper_bounds[rep_index].reshape(-1)
            conv_lb = lower_bounds[rep_index].reshape(-1)
            for idx, var in enumerate(m.gurobi_vars[rep_index]):
                svar = m.model_split.getVarByName(var.VarName)
                svar.ub = conv_ub[idx].item()
                svar.lb = conv_lb[idx].item()

        elif type(layer) is nn.ReLU:
            # locate relu index and remove all associated constraints
            relu_idx = m.pre_relu_indices.index(rep_index - 1)
            # reintroduce relu constraints
            pre_lbs = lower_bounds[rep_index - 1].reshape(-1)
            pre_ubs = upper_bounds[rep_index - 1].reshape(-1)
            for unmasked_idx in m.relu_indices_mask[relu_idx]:
                pre_lb = pre_lbs[unmasked_idx].item()
                pre_ub = pre_ubs[unmasked_idx].item()
                var = m.gurobi_vars[rep_index][unmasked_idx]
                svar = m.model_split.getVarByName(var.VarName)
                sa = m.model_split.getVarByName("a" + var.VarName)
                pre_var = m.gurobi_vars[rep_index - 1][unmasked_idx]
                pre_svar = m.model_split.getVarByName(pre_var.VarName)
                # print(sa, svar, pre_var, pre_svar)

                if pre_lb >= 0 and pre_ub >= 0:
                    # ReLU is always passing
                    svar.lb = pre_lb
                    svar.ub = pre_ub
                    introduced_constrs.append(m.model_split.addConstr(pre_svar == svar))
                    relu_mask[relu_idx][unmasked_idx] = 1
                elif pre_lb <= 0 and pre_ub <= 0:
                    svar.lb = 0
                    svar.ub = 0
                    relu_mask[relu_idx][unmasked_idx] = 0
                else:
                    svar.lb = 0
                    svar.ub = pre_ub
                    introduced_constrs.append(m.model_split.addConstr(pre_svar - pre_lb * (1 - sa) >= svar))
                    introduced_constrs.append(m.model_split.addConstr(svar >= pre_svar))
                    introduced_constrs.append(m.model_split.addConstr(pre_ub * sa >= svar))
                    introduced_constrs.append(m.model_split.addConstr(svar >= 0))

        elif type(layer) is Flatten:
            pass
        else:
            raise NotImplementedError
        m.model_split.update()
        rep_index += 1

    # compute optimum
    assert len(m.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
    target_var = m.gurobi_vars[-1][0]
    target_svar = m.model_split.getVarByName(target_var.VarName)

    m.model_split.update()
    # m.model.reset()
    m.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
    # m.model.setObjective(0, grb.GRB.MINIMIZE)
    m.model_split.optimize()
    # assert m.model.status == 2, "LP wasn't optimally solved"
    check_optimization_success(m.model_split, [introduced_constrs])

    glb = target_svar.X
    print(f"mip glb: {glb}")
    lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)

    # get input variable values at which minimum is achieved
    inp_size = lower_bounds[0].size()
    mini_inp = torch.zeros(inp_size).to(lower_bounds[0].device)
    if len(inp_size) == 1:
        # This is a linear input.
        for i in range(inp_size[0]):
            var = m.gurobi_vars[0][i]
            svar = m.model_split.getVarByName(var.VarName)
            mini_inp[i] = svar.x

    else:
        for i in range(inp_size[0]):
            for j in range(inp_size[1]):
                for k in range(inp_size[2]):
                    var = m.gurobi_vars[0][i][j][k]
                    svar = m.model_split.getVarByName(var.VarName)
                    mini_inp[i, j, k] = svar.x

    del m.model_split
    return


def solve_diving_lp(m, primal_vars, integer_vars, lower_bounds, upper_bounds):
    """
    A customized lp gurobi model to get the dual and primal values for each constraint and neurons
    """
    # only support mlp models for now
    diving_model = m.model.copy()
    diving_model.reset()
    relu_idx = 0
    # Reconstruct \hat{nu}.
    all_nus = []
    all_nu_hats = []
    for i, layer in enumerate(m.layers):
        if type(layer) is nn.Linear:
            w = layer.weight.to("cpu")
            nu = torch.zeros(w.size(0))
            size = w.size(0)
            if layer == m.layers[-1]: size = lower_bounds[-1].shape[1]
            for neuron_idx in range(size):
                nu[neuron_idx] = m.model.getConstrByName(f'lay{i+1}_{neuron_idx}_eq').pi
            nu_hat = nu.unsqueeze(0).matmul(w).squeeze(0)
            all_nus.append(nu)
            all_nu_hats.append(nu_hat)
    # Find dual variables for ReLU neurons.
    for i, layer in enumerate(m.layers):
        if type(layer) is nn.Linear:
            pass
        elif type(layer) is nn.ReLU:
            xs = primal_vars[i - 1]
            hat_xs = primal_vars[i]
            pre_lbs = lower_bounds[relu_idx].squeeze(0)
            pre_ubs = upper_bounds[relu_idx].squeeze(0)
            nu_hats = all_nu_hats[relu_idx+1]
            pos_nu_hats = torch.clamp_min(nu_hats, 0)
            neg_nu_hats = torch.clamp_max(nu_hats, 0)
            unstable_idx = 0
            for neuron_idx in range(len(xs)):
                lb = pre_lbs[neuron_idx].item()
                ub = pre_ubs[neuron_idx].item()
                if lb < 0 and ub > 0:
                    # Unstable neuron
                    x = xs[neuron_idx]
                    hat_x = hat_xs[neuron_idx]
                    z = integer_vars[relu_idx][unstable_idx]
                    pi = m.model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_0').pi  # dual variable for upper bound \hat{x} <= x - l + l * z
                    gamma = m.model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_1').pi  # dual variable for lower bound \hat{x} >= x.
                    tau = m.model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_2').pi  # dual variable for another upper bound \hat{x} <= u * z
                    mu = m.model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_3').pi  # dual variable for lower bound \hat{x} >= 0.
                    nu_hat = nu_hats[neuron_idx].item()
                    pos = pos_nu_hats[neuron_idx].item()
                    neg = neg_nu_hats[neuron_idx].item()
                    if nu_hat < -1e-6:
                        alpha = gamma / (gamma + mu)
                    else:
                        alpha = float("nan")
                    # Derived dual variable, they should match gurobi.
                    new_tau = - lb / (ub - lb) * pos
                    new_pi = ub / (ub - lb) * pos
                    # Try to do diving.
                    upper_z = 1.0
                    lower_z = 0.0
                    neuron_set_name = ""
                    if nu_hat < -1e-6:
                        # Lower bounds are chosen.
                        neuron_set_name = f'z_ReLU{relu_idx}_{neuron_idx}'
                        z_var = diving_model.getVarByName(neuron_set_name)
                        # Binding on the lower bound case, z has flexibility.
                        if x > 0:  # (equivalently, gamma != 0, mu = 0).
                            # Any z above this lower bound won't change obj.
                            lower_z = alpha / ub * x
                            # In this case, we can set z to 1. (hat_x = x).
                            z_var.LB = 1.0
                            neuron_set_name += " set to 1"
                        elif x < 0:  # (equivalently, gamma = 0, mu != 0).
                            # Any z below this upper bound won't change obj.
                            upper_z = (alpha - 1) / lb * x + 1
                            # In this case, we can set z to 0. (hat_x = 0).
                            z_var.UB = 0.0
                            neuron_set_name += " set to 0"
                        elif mu > 0 and gamma > 0:
                            # In this case we can set z to either 0 or 1.
                            assert abs(x) < 1e-6
                            if random.random() > 0.5:
                                z_var.LB = 1.0
                                neuron_set_name += " set to 1 (random)"
                            else:
                                z_var.UB = 0.0
                                neuron_set_name += " set to 0 (random)"

                    print(f'layer {i:2d} neuron {neuron_idx:3d} l={lb:8.5f} ub={ub:8.5f} x={x:8.5f} hat_x={hat_x:8.5f} pi={pi:8.5f} ({new_pi:8.5f}), tau={tau:8.5f} ({new_tau:8.5f}), mu={mu:8.5f}, gamma={gamma:8.5f}, nu_hat={nu_hat:8.5f}, alpha={alpha:8.5f} z={z:8.5f} upper_z={upper_z:8.5f} lower_z={lower_z:8.5f} {neuron_set_name}')
                    unstable_idx += 1
            relu_idx += 1

    # diving_model.getVarByName('z_ReLU11_193').LB=1.0
    diving_model.update()
    diving_model.optimize()
    print(f'original obj: {m.model.objval}, new_obj: {diving_model.objval}, status: {diving_model.status}')
    return


def update_mip_model_fix_relu(m, relu_idx, status, target=None, async_mip=False, best_adv=None, adv_activation_pattern=None):
    '''
    relu_idx: indices of relu to be fixed
    status: the status of the relu
    '''
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]

    batch_num = len(relu_idx)
    model_candidates = []

    for b in range(batch_num):
        if len(m.gurobi_vars[-1]) == 1:
            # Only a single target.
            model_candidates.append((relu_idx[b], status[b], m.gurobi_vars[-1][0].VarName, m.input_shape, best_adv[b], adv_activation_pattern[b]))
        else:
            if target is None: target = (m.c == -1).view(-1).nonzero().item()
            # Multiple labels; need to choose the target label.
            model_candidates.append((relu_idx[b], status[b], m.gurobi_vars[-1][target].VarName, m.input_shape, best_adv[b], adv_activation_pattern[b]))

    # MULTITHREAD
    global multiprocess_mip_model, stop_multiprocess
    stop_multiprocess = False
    multiprocess_mip_model = m.model

    if m.pool is None:
        m.pool_termination_flag = multiprocessing.Value('i')
        pool = NestablePool(mip_multi_proc, initializer=mip_solver_attack_init, initargs=(m.pool_termination_flag,))
        m.pool = pool
    else:
        pool = m.pool
        m.pool_termination_flag.value = 0
    if async_mip:
        solver_result = pool.map_async(mip_solver_attack, model_candidates)
    else:
        solver_result = pool.map(mip_solver_attack, model_candidates)
    
    if async_mip:
        # Returns a AsyncResult object. The caller is responsible for checking the final results.
        # solver_result = solver_result.get()
        # solver_result = np.asarray([a[0]<0 for a in solver_result])
        m.pool_result = solver_result
        return solver_result
    else:
        attack_result = any([a[0] < 0 for a in solver_result])
        return attack_result, solver_result


def build_the_model_mip_refine(m, lower_bounds, upper_bounds, 
            stop_criterion_func=stop_criterion_min(1e-4), score=None, 
            FSB_sort=True, topk_filter=1.):
    """
    Before the first branching, we build the model and create a mask matrix
    Output: relu_mask, current intermediate upper and lower bounds, a list of
            indices of the layers right before a Relu layer
            the constructed gurobi model
    NOTE: we keep all bounds as a list of tensors from now on.
            Only lower and upper bounds are kept in the same shape as layers' outputs.
            Mask is linearized
            Gurobi_var lists are lineariezd
            m.model_lower_bounds and m.model_upper_bounds are kepts mainly for
            debugging purpose and could be removed
    """
    new_relu_mask = []
    x = m.x
    input_domain = m.input_domain

    lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
    lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
    share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
    optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
    loss_reduction_func = reduction_str2func(arguments.Config["general"]["loss_reduction_func"])
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    mip_perneuron_refine_timeout = arguments.Config["solver"]["mip"]["refine_neuron_timeout"]

    global mip_refine_timeout
    mip_refine_timeout = arguments.Config["solver"]["mip"]["refine_neuron_time_percentage"] * arguments.Config["bab"]["timeout"]

    # preset the args for incomplete full crown with refined bounds
    m.net.init_slope((m.x,), share_slopes=share_slopes, c=m.c)
    m.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': 100, 'ob_beta': False, 'ob_alpha': True,
                                    'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                    'ob_early_stop': False, 'ob_verbose': 0,
                                    'ob_keep_best': True, 'ob_update_by_layer': True,
                                    'ob_lr': lr_init_alpha, 'ob_init': False,
                                    'ob_loss_reduction_func': loss_reduction_func, 
                                    'ob_stop_criterion_func': stop_criterion_func, 
                                    'ob_lr_decay': lr_decay}})

    lb_refined, ub_refined = None, None

    # Initialize the model
    m.model = grb.Model()
    m.model.setParam('OutputFlag', False)
    m.model.setParam("FeasibilityTol", 2e-5)

    #############
    # Config the hyperparameters for intermeidate bounds refinement with mip

    # default setting for aws instance
    # mip_threads = 1
    # mip_multi_proc = 8
    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    m.model.setParam('TimeLimit', mip_perneuron_refine_timeout)
    m.model.setParam('MIPGap', 1e-2)  # Relative gap between primal and dual.
    m.model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between primal and dual.
    m.model.setParam('Threads', mip_threads)
    print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads},"
            f"total threads used: {mip_multi_proc*mip_threads}, mip_perneuron_refine_timeout: {mip_perneuron_refine_timeout}")
    print(f"[total time budget for MIP: {mip_refine_timeout}]\n")

    refine_start_time = time.time()
    #############

    # keep a record of model's information
    m.gurobi_vars = []
    m.relu_constrs = []
    m.relu_indices_mask = []

    ## Do the input layer, which is a special case
    inp_gurobi_vars = []
    zero_var = m.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    if input_domain.dim() == 2:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(input_domain):
            v = m.model.addVar(lb=lb, ub=ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        assert input_domain.dim() == 4
        for chan in range(input_domain.size(0)):
            chan_vars = []
            for row in range(input_domain.size(1)):
                row_vars = []
                for col in range(input_domain.size(2)):
                    lb = input_domain[chan, row, col, 0]
                    ub = input_domain[chan, row, col, 1]
                    v = m.model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)
    m.model.update()

    m.gurobi_vars.append(inp_gurobi_vars)

    ## Do the other layers, computing for each of the neuron, its upper
    ## bound and lower bound
    layer_idx = 1
    relu_idx = 0
    maximum_refined_relu_layers = 0
    need_refine = True
    global multiprocess_mip_model, mip_refine_time_start
    mip_refine_time_start = time.time()
    # print(len(m.layers), len(m.net.relus), len(lower_bounds))
    last_relu_layer_refined = False
    for layer in m.layers:
        this_layer_refined = False
        new_layer_gurobi_vars = []
        if type(layer) is nn.Linear:
            
            # Get the better estimates from KW and Interval Bounds
            # print("linear", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape)
            out_lbs = lower_bounds[relu_idx].squeeze(0)
            out_ubs = upper_bounds[relu_idx].squeeze(0)

            print(layer, relu_idx, layer_idx, out_lbs.shape)

            candidates = []
            candidate_neuron_ids = []
            for neuron_idx in range(layer.weight.size(0)):
                lin_expr = layer.bias[neuron_idx].item()
                coeffs = layer.weight[neuron_idx, :]
                lin_expr += grb.LinExpr(coeffs, m.gurobi_vars[-1])

                out_lb = out_lbs[neuron_idx].item()
                out_ub = out_ubs[neuron_idx].item()

                v = m.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                        vtype=grb.GRB.CONTINUOUS,
                                        name=f'lay{layer_idx}_{neuron_idx}')
                m.model.addConstr(lin_expr == v)
                m.model.update()

                # if relu_idx == 1 and (out_lb * out_ub < 0):
                if (relu_idx >= 1 and relu_idx < len(m.net.relus)) and (out_lb * out_ub < 0) and (time.time() - mip_refine_time_start<mip_refine_timeout):
                    candidates.append(v.VarName)
                    candidate_neuron_ids.append(neuron_idx)
                
                new_layer_gurobi_vars.append(v)

            if need_refine and (relu_idx >= 1 and relu_idx < len(m.net.relus)) and score is not None and FSB_sort:
                # sort (candidates, candidate_neuron_ids) according to score[relu_idx][candidate_neuron_ids]
                s = score[relu_idx].view(-1)[candidate_neuron_ids]
                _, indices = s.sort(descending=True)
                candidates = np.array(candidates)[indices.cpu().numpy()].tolist()
                candidate_neuron_ids = np.array(candidate_neuron_ids)[indices.cpu().numpy()].tolist()
                if topk_filter != 1.:
                    candidates = candidates[:int(len(candidates)*topk_filter)]
                    candidate_neuron_ids = candidate_neuron_ids[:int(len(candidate_neuron_ids)*topk_filter)]
                print("sorted candidates", candidates, "filter:", topk_filter)

            for vi in new_layer_gurobi_vars:
                vi.LB = -np.inf
                vi.UB = np.inf

            if need_refine and (relu_idx >= 1 and relu_idx < len(m.net.relus)) and (time.time() - mip_refine_time_start < mip_refine_timeout):
                multiprocess_mip_model = m.model
                refine_time = time.time()

                #####################
                # candidates = [candidates[ci] for ci in range(10)]
                # candidate_neuron_ids = [candidate_neuron_ids[ci] for ci in range(10)]
                #####################

                if relu_idx == 1:
                    # the second relu layer where mip refine starts
                    with multiprocessing.Pool(mip_multi_proc) as pool:
                        solver_result = pool.map(mip_solver, candidates, chunksize=1)

                    lb_refined_sum, ub_refined_sum, refined_num = 0., 0., 0
                    for (vlb, vub, refined), neuron_idx in zip(solver_result, candidate_neuron_ids):
                        if refined:
                            # v = new_layer_gurobi_vars[neuron_idx]
                            refined_num += 1
                            lb_refined_sum += vlb-lower_bounds[relu_idx][0, neuron_idx]
                            ub_refined_sum += upper_bounds[relu_idx][0, neuron_idx]-vub
                            lower_bounds[relu_idx][0, neuron_idx] = vlb
                            upper_bounds[relu_idx][0, neuron_idx] = vub
                            # v.LB = vlb
                            # v.UB = vub
                    refine_time = time.time() - refine_time
                    print(f"MIP improved {refined_num} nodes out of {len(candidates)} unstable nodes, lb improved {lb_refined_sum}, ub improved {ub_refined_sum}, time {refine_time:.4f}")
                    if refined_num > 0: 
                        maximum_refined_relu_layers = relu_idx
                        this_layer_refined = True
                        last_relu_layer_refined = True
                    else:
                        need_refine = False
                        last_relu_layer_refined = False
                    print("maximum relu layer improved by MIP so far", maximum_refined_relu_layers, "last_relu_layer_refined:", last_relu_layer_refined)
                    m.model.update()

                else:
                    with multiprocessing.Pool(mip_multi_proc) as pool:
                        solver_result = pool.map_async(mip_solver, candidates, chunksize=1)

                        if last_relu_layer_refined and (time.time() - mip_refine_time_start < mip_refine_timeout):
                            print(f"Run alpha-CROWN after refining layer {layer_idx-2} and relu idx {relu_idx-1}")
                            # using refined bounds with init opt crown for the previous optimized bounds
                            new_interval, reference_bounds = {}, {}
                            # for i, layer in enumerate(m.net.relus):
                            # only refined with the second relu layer
                            for i, layer in enumerate(m.net.relus):
                                # only refined with the relu layers that are refined by mip before
                                if i>=(maximum_refined_relu_layers+1): break
                                nd = m.net.relus[i].inputs[0].name
                                print(i, nd, lower_bounds[i].shape)
                                new_interval[nd] = [lower_bounds[i], upper_bounds[i]]
                                reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
                            # m.net.bound_opts['sparse_intermediate_bounds'] = False
                            lb_refined, ub_refined = m.net.compute_bounds(x=(x,), IBP=False, C=m.c, method='CROWN-optimized', return_A=False,
                                                        reference_bounds=reference_bounds, bound_upper=False)
                            # lb, ub = m.net.compute_bounds(x=(x,), IBP=False, C=m.c, method='CROWN-Optimized', return_A=False,
                                                                    # bound_upper=False)
                            print("alpha-CROWN with intermediate bounds by MIP:", lb_refined, ub_refined)

                            if lb_refined.min().item()>=0:
                                print(f"min of alpha-CROWN bounds {lb_refined.min().item()}>=0, verified!")
                                pool.terminate()
                                break
                            last_relu_layer_refined = False

                        solver_result = solver_result.get()

                    lb_refined_sum, ub_refined_sum, refined_num = 0., 0., 0
                    for (vlb, vub, refined), neuron_idx in zip(solver_result, candidate_neuron_ids):
                        if refined:
                            # v = new_layer_gurobi_vars[neuron_idx]
                            refined_num += 1
                            lb_refined_sum += vlb-lower_bounds[relu_idx][0, neuron_idx]
                            ub_refined_sum += upper_bounds[relu_idx][0, neuron_idx]-vub
                            lower_bounds[relu_idx][0, neuron_idx] = vlb
                            upper_bounds[relu_idx][0, neuron_idx] = vub
                            # v.LB = vlb
                            # v.UB = vub
                    refine_time = time.time() - refine_time
                    print(f"MIP improved {refined_num} nodes out of {len(candidates)} unstable nodes, lb improved {lb_refined_sum}, ub improved {ub_refined_sum}, time {refine_time:.4f}")
                    if refined_num>0: 
                        maximum_refined_relu_layers = relu_idx
                        this_layer_refined = True
                        last_relu_layer_refined = True
                    else:
                        need_refine = False
                        last_relu_layer_refined = False
                    print("maximum relu layer improved by MIP so far", maximum_refined_relu_layers)
                    m.model.update()

        elif type(layer) is nn.Conv2d:
            ###########
            # Refine the conv layers as well
            ###########
            # raise NotImplementedError
            assert layer.dilation == (1, 1)
            if relu_idx == 0:
                pre_lb_size = m.x.shape
            else:
                pre_lb_size = lower_bounds[relu_idx-1].size()
            out_lbs = lower_bounds[relu_idx]
            out_ubs = upper_bounds[relu_idx]
            # print("conv", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape, layer.bias.shape)
            gvars_array = np.array(m.gurobi_vars[-1])

            candidates = []
            candidate_neuron_ids = []
            for out_chan_idx in range(out_lbs.size(1)):
                out_chan_vars = []
                for out_row_idx in range(out_lbs.size(2)):
                    out_row_vars = []
                    for out_col_idx in range(out_lbs.size(3)):
                        # print(layer.bias.shape, out_chan_idx, out_lbs.size(1))
                        lin_expr = layer.bias[out_chan_idx].item()

                        for in_chan_idx in range(layer.weight.shape[1]):

                            # new version of conv layer for building mip by skipping kernel loops
                            ker_row_min, ker_row_max = 0, layer.weight.shape[2]
                            in_row_idx_min = -layer.padding[0] + layer.stride[0] * out_row_idx
                            in_row_idx_max = -layer.padding[0] + layer.stride[0] * out_row_idx + layer.weight.shape[2] - 1
                            if in_row_idx_min < 0: ker_row_min = 0 - in_row_idx_min
                            if in_row_idx_max >= pre_lb_size[2]: ker_row_max = ker_row_max - in_row_idx_max + pre_lb_size[2] -1
                            in_row_idx_min, in_row_idx_max = max(in_row_idx_min, 0), min(in_row_idx_max, pre_lb_size[2] - 1)

                            ker_col_min, ker_col_max = 0, layer.weight.shape[3]
                            in_col_idx_min = -layer.padding[1] + layer.stride[1] * out_col_idx
                            in_col_idx_max = -layer.padding[1] + layer.stride[1] * out_col_idx + layer.weight.shape[3] - 1
                            if in_col_idx_min < 0: ker_col_min = 0 - in_col_idx_min
                            if in_col_idx_max >= pre_lb_size[3]: ker_col_max = ker_col_max - in_col_idx_max + pre_lb_size[3] -1
                            in_col_idx_min, in_col_idx_max = max(in_col_idx_min, 0), min(in_col_idx_max, pre_lb_size[3] - 1)

                            coeffs = layer.weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)

                            gvars = gvars_array[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                            lin_expr += grb.LinExpr(coeffs, gvars)

                            # old version of conv layer for building mip
                            # for ker_row_idx in range(layer.weight.shape[2]):
                            #     in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                            #     if (in_row_idx < 0) or (in_row_idx == pre_lb_size[2]):
                            #         # This is padding -> value of 0
                            #         continue
                            #     for ker_col_idx in range(layer.weight.shape[3]):
                            #         in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                            #         if (in_col_idx < 0) or (in_col_idx == pre_lb_size[3]):
                            #             # This is padding -> value of 0
                            #             continue
                            #         # print(in_row_idx, in_col_idx)
                            #         coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                            #         lin_expr += coeff * m.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                        out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                        out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                        v = m.model.addVar(lb=out_lb, ub=out_ub,
                                                obj=0, vtype=grb.GRB.CONTINUOUS,
                                                name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        m.model.addConstr(lin_expr == v)
                        m.model.update()

                        if need_refine and (relu_idx >= 1) and (out_lb * out_ub < 0) and (time.time() - mip_refine_time_start < mip_refine_timeout):
                            candidates.append(v.VarName)
                            candidate_neuron_ids.append((out_chan_idx, out_row_idx, out_col_idx))

                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)

            ####### Comment out the following condition if disable refine for conv layers #######
            if False and need_refine and relu_idx >= 1 and (time.time() - mip_refine_time_start < mip_refine_timeout):
                multiprocess_mip_model = m.model
                refine_time = time.time()
                with multiprocessing.Pool(mip_multi_proc) as pool:
                    solver_result = pool.map(mip_solver, candidates)

                lb_refined_sum, ub_refined_sum, refined_num = 0., 0., 0
                for (vlb, vub, refined), (out_chan_idx, out_row_idx, out_col_idx) in zip(solver_result, candidate_neuron_ids):
                    if refined: 
                        # v = new_layer_gurobi_vars[out_chan_idx, out_row_idx, out_col_idx]
                        refined_num += 1
                        lb_refined_sum += vlb-lower_bounds[relu_idx][0, out_chan_idx, out_row_idx, out_col_idx]
                        ub_refined_sum += upper_bounds[relu_idx][0, out_chan_idx, out_row_idx, out_col_idx]-vub
                        lower_bounds[relu_idx][0, out_chan_idx, out_row_idx, out_col_idx] = vlb
                        upper_bounds[relu_idx][0, out_chan_idx, out_row_idx, out_col_idx] = vub
                        # v.LB = vlb
                        # v.UB = vub
                refine_time = time.time() - refine_time
                print(f"MIP improved {refined_num} nodes out of {len(candidates)} unstable nodes, lb improved {lb_refined_sum}, ub improved {ub_refined_sum}, time {refine_time:.4f}")
                if refined_num>0: 
                    maximum_refined_relu_layers = relu_idx
                    this_layer_refined = True
                print("maximum relu layer imporved by MIP so far", maximum_refined_relu_layers)
                m.model.update()

        elif type(layer) is nn.ReLU:
            new_relu_layer_constr = []
            this_relu = m.net.relus[relu_idx]
            if isinstance(m.gurobi_vars[-1][0], list):
                # This is convolutional relu
                pre_lbs = lower_bounds[relu_idx].squeeze(0)
                pre_ubs = upper_bounds[relu_idx].squeeze(0)
                new_layer_mask = []
                # print("conv relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                temp = pre_lbs.size()
                out_chain = temp[0]
                out_height = temp[1]
                out_width = temp[2]
                for chan_idx, channel in enumerate(m.gurobi_vars[-1]):
                    chan_vars = []
                    for row_idx, row in enumerate(channel):
                        row_vars = []
                        for col_idx, pre_var in enumerate(row):
                            pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()
                            pre_lb = pre_lbs[chan_idx, row_idx, col_idx].item()

                            if pre_lb >= 0:
                                # ReLU is always passing
                                v = pre_var
                                new_layer_mask.append(1)
                            elif pre_ub <= 0:
                                v = zero_var
                                new_layer_mask.append(0)
                            else:
                                lb = 0
                                ub = pre_ub
                                new_layer_mask.append(-1)
                                neuron_idx = col_idx + row_idx * out_width + chan_idx * out_height * out_width

                                v = m.model.addVar(ub=ub, lb=pre_lb,
                                                        obj=0,
                                                        vtype=grb.GRB.CONTINUOUS,
                                                        name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                # binary indicator
                                a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')

                                new_relu_layer_constr.append(
                                    m.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                            name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                                new_relu_layer_constr.append(
                                    m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                                new_relu_layer_constr.append(
                                    m.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                new_relu_layer_constr.append(
                                    m.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_3'))

                            row_vars.append(v)
                        chan_vars.append(row_vars)
                    new_layer_gurobi_vars.append(chan_vars)
            else:
                # this is linear relu
                pre_lbs = lower_bounds[relu_idx].squeeze(0)
                pre_ubs = upper_bounds[relu_idx].squeeze(0)
                # print("linear relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                new_layer_mask = []
                assert isinstance(m.gurobi_vars[-1][0], grb.Var)
                for neuron_idx, pre_var in enumerate(m.gurobi_vars[-1]):
                    pre_ub = pre_ubs[neuron_idx].item()
                    pre_lb = pre_lbs[neuron_idx].item()

                    if pre_lb >= 0:
                        # The ReLU is always passing
                        v = pre_var
                        new_layer_mask.append(1)
                    elif pre_ub <= 0:
                        v = zero_var
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                        new_layer_mask.append(0)
                    else:
                        lb = 0
                        ub = pre_ub
                        # post-relu var
                        v = m.model.addVar(ub=ub, lb=0,
                                                obj=0,
                                                vtype=grb.GRB.CONTINUOUS,
                                                name=f'ReLU{layer_idx}_{neuron_idx}')
                        # binary indicator
                        a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_{neuron_idx}')

                        new_relu_layer_constr.append(
                            m.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                    name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_0'))
                        new_relu_layer_constr.append(
                            m.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_1'))
                        new_relu_layer_constr.append(
                            m.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_2'))
                        # new_relu_layer_constr.append(
                        #     m.model.addConstr(v >= 0, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_3'))

                        new_layer_mask.append(-1)
                        #v.LB, v.UB = 0, np.inf

                    new_layer_gurobi_vars.append(v)

            new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
            m.relu_constrs.append(new_relu_layer_constr)
            relu_idx += 1

        elif type(layer) == Flatten or "Flatten" in str(type(layer)):
            for chan_idx in range(len(m.gurobi_vars[-1])):
                for row_idx in range(len(m.gurobi_vars[-1][chan_idx])):
                    new_layer_gurobi_vars.extend(m.gurobi_vars[-1][chan_idx][row_idx])
        else:
            raise NotImplementedError

        m.gurobi_vars.append(new_layer_gurobi_vars)

        layer_idx += 1

        if (time.time() - mip_refine_time_start >= mip_refine_timeout):
            break

    multiprocess_mip_model, mip_refine_time_start = None, None

    m.model.update()
    
    print(f'MIP finished with {time.time() - refine_start_time}s')

    slope_opt = None
    
    primals, duals, mini_inp = None, None, None

    if last_relu_layer_refined:
        print(f"Run final alpha-CROWN after MIP solving on layer {layer_idx-1} and relu idx {relu_idx}")
        # using refined bounds with init opt crown
        new_interval, reference_bounds = {}, {}
        # for i, layer in enumerate(m.net.relus):
        # only refined with the second relu layer
        for i, layer in enumerate(m.net.relus):
            # only refined with the relu layers that are refined by mip before
            if i>=(maximum_refined_relu_layers+1): break
            nd = m.net.relus[i].inputs[0].name
            print(i, nd, lower_bounds[i].shape)
            new_interval[nd] = [lower_bounds[i], upper_bounds[i]]
            reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
        lb_refined, ub_refined = m.net.compute_bounds(x=(x,), IBP=False, C=m.c, method='CROWN-optimized', return_A=False,
                                    reference_bounds=reference_bounds, bound_upper=False)
        print("alpha-CROWN with intermediate bounds improved by MIP:", lb_refined, ub_refined)
    
    if lb_refined is None:
        if lower_bounds[-1].shape[1] != m.c.shape[1]:
            # remove true label 0 bounds according to C matrix
            lower_bounds[-1] = lower_bounds[-1].mm(-m.c[0].T)
            upper_bounds[-1] = upper_bounds[-1].mm(-m.c[0].T)
        return lower_bounds, upper_bounds

    lb_refined, ub_refined, pre_relu_indices = m.get_candidate(m.net, lb_refined, lb_refined + 99)  # primals are better upper bounds
    mask, lA = m.get_mask_lA_parallel(m.net)
    return lb_refined, ub_refined

