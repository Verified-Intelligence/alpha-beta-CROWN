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

import torch.nn.functional as F
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import (reduction_min, reduction_max, reduction_mean, reduction_sum,
                            stop_criterion_sum, stop_criterion_min)
from model_defs import Flatten
from auto_LiRPA.bound_ops import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd
import beta_CROWN_solver

try:
    from scip_model import SCIPModel, EarlyStopEvent, GenerateCutsEvent
except:
    pass

import multiprocessing
import multiprocessing.pool
import sys
import os
import subprocess

CPLEX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CPLEX_cuts")
# CPLEX_FOLDER = "CPLEX_cuts"

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
mip_solve_time_start = None


def mip_solver(candidate, init=None):
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

    def solve_ub(model, v, out_ub, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
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

    def solve_lb(model, v, out_lb, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
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

    def init_x_start(model, init):
        # print("init with pgd adv!")
        dim = 0
        for chan in range(len(init)):
            for row in range(len(init[0])):
                for col in range(len(init[0][0])):
                    v = model.getVarByName(f'inp_{dim}')
                    v.Start = init[chan][row][col]
                    dim += 1
        model.update()
        return

    init_lb, init_ub = None, None
    if init is not None:
        init_lb, init_ub = init
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
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps, init=init_lb)
        neuron_refined = neuron_refined or refined
        if vub > 0:
            # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps, init=init_ub)
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
    """Adversarial attack using MIP. A negative returned bound indicates attack success regardless of model status."""
    global termination_flag
    # Modify the current MIP model with split constraints.
    model = multiprocess_mip_model.copy()

    indices, relu_status, opt_var, input_shape, best_adv_input, relu_forward,\
            pre_relu_layer_names, relu_layer_names, lower_bounds, upper_bounds = new_splits
    input_shape = list(input_shape)

    if lower_bounds is not None:
        # assign intermediate bounds from full alpha-crown with the splits
        assert arguments.Config["bab"]["attack"]["refined_mip_attacker"]
        print("using full alpha crown intermediate bounds to refine mip solver attack!")
        update_model_bounds(model, lower_bounds, upper_bounds, pre_relu_layer_names, relu_layer_names, model_type="mip")

    for [relu_idx, neuron_idx], s in zip(indices, relu_status):
        a = model.getVarByName(f"aReLU{relu_layer_names[relu_idx]}_{neuron_idx}")
        assert a is not None, f"Warning: aReLU{relu_layer_names[relu_idx]}_{neuron_idx}"

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
            dim = 0
            for chan in range(C):
                for row in range(H):
                    for col in range(W):
                        # v = model.getVarByName(f"inp_[{chan},{row},{col}]")
                        v = model.getVarByName(f"inp_{dim}")
                        v.Start = best_adv_input[s][chan][row][col]
                        dim += 1

            for relu_idx, relu in enumerate(relu_forward):
                # Each tensor has size (n_start, n_neurons), flattened.
                for i in range(relu.size(1)):
                    v = model.getVarByName(f"aReLU{relu_layer_names[relu_idx]}_{i}")
                    if v is not None:
                        v.Start = int(relu[s][i]>0)

    v = model.getVarByName(opt_var)

    ######### mip solve without async #########
    # refine_time = time.time()
    # model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
    # model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
    # model.setParam('OutputFlag', False)
    # model.setParam('Heuristics', 0.5)
    # model.setParam('MIPFocus', 1)

    # model.setObjective(v, grb.GRB.MINIMIZE)
    # model.optimize()

    # print("Solved MIP for {}, {} neurons fixed, status:{} ({} sols), lower={:.6f}, upper={:.6f}, time: {:.6f}s".format(v.VarName, len(indices), model.status,
    #                 model.solcount, model.objbound, model.objval, time.time()-refine_time))
    # return model.objval, model.objbound, model.status, None
    #########

    if termination_flag.value == 1:
        return float("inf"), float("inf"), -1, torch.zeros([0,] + input_shape[1:], requires_grad=False)  # Solver skipped.

    refine_time = time.time()
    model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
    model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
    model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
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
            print('Early Stopping MIP solver (another MIP process may succeed; or we run out of time).')
            p.terminate()
            time.sleep(0.5)
            p.kill()
            return float("inf"), float("inf"), -1, torch.zeros([0,] + input_shape[1:], requires_grad=False)
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


def mip_solver_lb_ub(candidate, init=None, save_adv=None):
    """ Solving MIP for adversarial attack/complete verification.
    init: warm up with given init which is usually found by pgd attack
    save_adv: a list of input names that we need to retrieve if an adv found
    """

    def init_x_start(model, init):
        # print("init with pgd adv!")
        dim = 0
        for chan in range(len(init)):
            for row in range(len(init[0])):
                for col in range(len(init[0][0])):
                    v = model.getVarByName(f'inp_{dim}')
                    v.Start = init[chan][row][col]
                    dim += 1
        model.update()
        return

    init_lb, init_ub = None, None
    adv = None
    if init is not None:
        init_lb, init_ub = init
    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    vlb = out_lb = v.LB
    vub = out_ub = v.UB
    global stop_multiprocess, mip_solve_time_start
    if stop_multiprocess:
        return out_lb, out_ub, -1, adv  # Solver skipped.
    refine_time = time.time()
    if arguments.Config["solver"]["mip"]["early_stop"]:
        if arguments.Config["solver"]["mip"]["mip_solver"] == 'gurobi':
            model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
            model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
        elif arguments.Config["solver"]["mip"]["mip_solver"] == 'scip':
            model.includeEventhdlr(EarlyStopEvent(), "EarlyStopEvent", "python event handler to early stop.")  # FIXME (01/13/22): threshold should be changable, based on specification.
        else:
            raise NotImplementedError
    if arguments.Config["solver"]["mip"]["mip_solver"] == 'scip':
        if bool(os.environ.get('ALPHA_BETA_CROWN_MIP_SHOW_CUTS', False)):
            model.includeEventhdlr(GenerateCutsEvent(), "GenerateCutsEvent", "python event handler to save cuts.")

    model.setObjective(v, grb.GRB.MINIMIZE)
    if init_lb is not None:
        init_x_start(model, init_lb)
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
        if save_adv:
            adv = [model.getVarByName(var_name).X for var_name in save_adv]

    assert model.status != 3, "should not be infeasible"
    print("solving MIP for {}, status:{}, [{}, {}]=>[{}, {}], time: {}s".format(v.VarName, model.status,
                    out_lb, out_ub, vlb, vub, time.time()-refine_time))
    sys.stdout.flush()
    if time.time() - mip_solve_time_start > arguments.Config["bab"]["timeout"]:
        stop_multiprocess = True
    return vlb, vub, model.status, adv


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


def build_solver_model(m, timeout, mip_multi_proc=None,
         mip_threads=1, model_type="mip", x=None, intermediate_bounds=None):
    """
    m is the instance of LiRPAConvNet
    model_type ["mip", "lp", "lp_integer"]: three different types of guorbi solvers
    lp_integer using mip formulation but continuous integer variable from 0 to 1 instead of
    binary variable as mip; lp_integer should have the same results as lp but allowing us to
    estimate integer variables.
    NOTE: we build lp/mip solver from computer graph
    """
    build_mip_start_time = time.time()
    if m.pool is not None:
        # Must close the pool because the old model shared to the pool workers is now stale.
        print("Closing MIP Pool...")
        m.pool.close()
        m.pool.terminate()
        m.pool.kill()
        m.pool = None
        m.pool_termination_flag = None
    # input_domain = input_domain if input_domain is not None else m.input_domain
    # input_domain = input_domain.cpu().numpy()
    m.net.model_type = model_type
    m.net.solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]

    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    ### Merge the current params to a new solver model build function
    # Initialize the model
    if m.net.solver_pkg == 'gurobi':
        m.net.model = grb.Model()
    elif m.net.solver_pkg == 'scip':
        m.net.model = SCIPModel()
    else:
        raise NotImplementedError

    ### Merge the current params to a new solver model config function
    m.net.model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
    m.net.model.setParam('Threads', mip_threads)
    m.net.model.setParam("FeasibilityTol", 2e-5)
    m.net.model.setParam('TimeLimit', timeout)
    cut_options = os.environ.get('ALPHA_BETA_CROWN_MIP_CUT_DEBUG', None)
    if cut_options is not None:
        net.model.setParam('Cuts', 0)
        for cut in cut_options.strip().split(','):
            cut, val = cut.strip().split('=')
            val = int(val)
            if cut == 'Gomory':
                suffix = 'Passes'
            else:
                suffix = 'Cuts'
            print(f'Setting {cut}{suffix} to {val}')
            net.model.setParam(f'{cut}{suffix}', val)
    print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads}, total threads used: {mip_multi_proc*mip_threads}")
    build_mip_time = time.time()

    # build model in auto_LiRPA
    out_vars = m.net.build_solver_module(
            x=x, C=m.c, intermediate_layer_bounds=intermediate_bounds, final_node_name=m.net.final_name, model_type=model_type, solver_pkg=m.net.solver_pkg)
    m.net.model.update()
    build_mip_time = time.time() - build_mip_start_time
    print(f"{model_type} solver model built in {build_mip_time:.4f} seconds.")
    return out_vars


# updated function using general computation graph to build lp model
def build_the_model_lp(m, using_integer=True, get_primals=False):
    """
    Before the first branching, we build the solver model
    using_integer:
        True: using mip formulation with continuous integer varariable (model_type="lp_integer")
        False: using triangle relaxation without integer varariable (model_type="lp")
    Output: the lower bound by solver model
    NOTE: We only consider one output node for now
    """
    timeout = arguments.Config["bab"]["timeout"]
    model_type = "lp"
    if using_integer: model_type = "lp_integer"

    # build the solver models
    m.build_solver_model(timeout, model_type=model_type)

    out_vars = m.net[m.net.final_name].solver_vars
    for obj in out_vars:
        guro_start = time.time()
        m.net.model.setObjective(obj, grb.GRB.MINIMIZE)
        try:
            m.net.model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)

        status = m.net.model.status
        assert status == 2, f"LP wasn't optimally solved status:{status}"
        print(f"[{obj}]- status: {status}, time: {time.time() - guro_start}")
        glb = obj.X if status != 3 else None

    if get_primals:
        # get the primal values for each layer from gurobi lp model
        primal_vars = []
        # mnist6_100 case:
        # primal_vars[0]: 784, input's primal values
        # primal_vars[1]: 100, z_hat (pre relu) primal value
        # primal_vars[2]: 100, z (post relu) primal value
        # primal_vars[11]: 1, last output primal value
        layers = [m.net.final_node()]
        node = m.net.final_node()
        while node.inputs:
            layers = [node.inputs[0]] + layers
            node = node.inputs[0]

        for layer in layers:
            pv = []
            vars = layer.solver_vars
            if not isinstance(vars[0], list):
                for i in range(len(vars)):
                    pv.append(vars[i].X)
            else:
                for chan in range(len(vars)):
                    for row in range(len(vars[chan])):
                        for col in range(len(vars[chan][row])):
                            pv.append(vars[chan][row][col].X)
            primal_vars.append(pv)

        if using_integer:
            # get integer variables for unstable relu neurons in each relu layer
            integer_vars = []
            for relu_idx, relu_layer in enumerate(m.net.relus):
                iv = []
                for relu_integer in relu_layer.integer_vars:
                    iv.append(relu_integer.X)
                # np.save(f"gurobi_primals/z_relu{relu_idx}.npy", np.array(iv))
                integer_vars.append(iv)

        input_primal_gurobi = primal_vars[0]
        print("### Extracting primal values from gurobi lp model done ###")
        # m.solve_diving_lp(primal_vars, integer_vars, lower_bounds, upper_bounds)

    return glb


def update_model_bounds(solver_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, model_type="lp_integer"):
    """update solver vars bounds with given lower and upper bounds
    Args:
        solver_model: copied solver model from m.net.model
        lower_bounds, upper_bounds: tightened bounds
        pre_relu_layer_names, relu_layer_names: the names of pre relu layers and relu layers, to get the vars
        model_type: m.net.model_type
    """
    # update pre relu neuron bounds as well as relu constraints
    for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
        this_layer_shape = lower_bounds[relu_idx].shape
        lbs, ubs = lower_bounds[relu_idx].reshape(-1), upper_bounds[relu_idx].reshape(-1)
        for neuron_idx in range(lbs.shape[0]):
            pre_var = solver_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
            pre_var.lb = pre_lb = lbs[neuron_idx]
            pre_var.ub = pre_ub = ubs[neuron_idx]
            var = solver_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
            # var is None if originally stable
            if var is not None:
                if pre_lb >= 0 and pre_ub >= 0:
                    # ReLU is always passing
                    var.lb = pre_lb
                    var.ub = pre_ub
                    solver_model.addConstr(pre_var == var)
                elif pre_lb <= 0 and pre_ub <= 0:
                    var.lb = 0
                    var.ub = 0
                else:
                    var.lb = 0
                    var.ub = pre_ub
                    if model_type in ["mip", "lp_integer"]:
                        a = solver_model.getVarByName(f"aReLU{relu_name}_{neuron_idx}")
                        solver_model.addConstr(pre_var - pre_lb * (1 - a) >= var)
                        solver_model.addConstr(var >= pre_var)
                        solver_model.addConstr(pre_ub * a >= var)
                    else:
                        solver_model.addConstr(var >= pre_var)
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        solver_model.addConstr(var <= slope * pre_var + bias)

    solver_model.update()
    return solver_model

def all_node_split_LP(m, lower_bounds, upper_bounds, rhs):
    m.all_node_model = copy_model(m.net.model, basis=False)

    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in m.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in m.net.relus]
    m.all_node_model = update_model_bounds(m.all_node_model, lower_bounds, upper_bounds,\
                pre_relu_layer_names, relu_layer_names, m.net.model_type)
    print('Finished building Gurobi LP model for all node split. Start solving the LP!')
    lp_status = "unsafe"
    adv = None

    assert lower_bounds[-1].size(0) == 1,  "only bounds with batch size 1"
    guro_start = time.time()
    # Assert that this is as expected a network with a single output
    orig_out_vars = m.net.final_node().solver_vars
    # assert len(orig_out_vars) == 1, "Network doesn't have scalar output"
    glbs = lower_bounds[-1].squeeze(0).clone()
    # import pdb; pdb.set_trace()
    assert len(orig_out_vars) == len(rhs), "out shape not matching!"
    for out_idx in range(len(orig_out_vars)):
        # objVar = m.all_node_model.getVarByName(orig_out_vars[0].VarName)
        objVar = m.all_node_model.getVarByName(orig_out_vars[out_idx].VarName)
        decision_threshold = rhs[out_idx]

        m.all_node_model.setObjective(objVar, grb.GRB.MINIMIZE)
        m.all_node_model.update()
        m.all_node_model.optimize()

        # assert m.model_cut.status == 2, f"model not optimized with status {m.model_cut.status}"
        if m.all_node_model.status == 2:
            glb = objVar.X
        elif m.all_node_model.status == 3:
            print("gurobi all node split lp model infeasible!")
            glb = float('inf')
        else:
            print(f"Warning: model status {m.all_node_model.status}!")
            glb = float('inf')
        print(f"# all node split glb [{objVar.VarName}], status {m.all_node_model.status}: {glbs[0]} -> {glb} vs rhs {decision_threshold}")

        guro_end = time.time()
        print('Gurobi solved the LP with time', guro_end - guro_start)
        glbs[out_idx] = glb
        if glb > decision_threshold:
            lp_status = "safe"
            break
    del m.all_node_model
    print(f"#### all node split glb:", glbs)
    return lp_status, glbs, adv



build_the_model_mip_proto_gurobi_model = None
def _build_the_model_mip_mps_save(args):
    candidate, fname = args
    print(f"parallel save mip model to {fname}.mps")
    now_model = build_the_model_mip_proto_gurobi_model.copy()
    v = now_model.getVarByName(candidate)
    now_model.setObjective(v, grb.GRB.MINIMIZE)
    now_model.update()
    now_model.write(f'{fname}.mps')

@torch.no_grad()
def construct_mip_with_model(unwrapped_model, x, input_shape, c, intermediate_bounds,
        save_mps=False, process_dict=None):
    """Construct a mip problem using just the model, input x (BoundedTensor), intermediate layer bounds and other parameters."""
    # Set threads to 1 to avoid a libgomp hang. Very important.
    # See https://github.com/pytorch/pytorch/issues/17199
    # See https://github.com/pytorch/pytorch/issues/58962
    # Bug still exists on Pytorch 1.11 with any tensor greater than  128 KBytes.
    torch.set_num_threads(1)
    # This will create the BoundedModule object at model.net.
    model = beta_CROWN_solver.LiRPAConvNet(unwrapped_model, in_size=input_shape, c=c, device='cpu')
    build_the_model_mip(model, labels_to_verify=None, save_mps=save_mps, process_dict=process_dict, x=x, intermediate_bounds=intermediate_bounds)


# updated function using general computation graph to build lp model
# @torch.no_grad()
def build_the_model_mip(m, labels_to_verify=None, save_mps=False, process_dict=None, save_adv=False, x=None, intermediate_bounds=None):
    """
    Using the built gurobi model to solve mip formulation in parallel
    lower_bounds, upper_bounds: intermediate relu bounds from beta-crown
    simplified: only for target label if simplified, otherwise all labels remained
    if process_dict is a dict, then we will dynamically record the cut getting processes into process_dict
        to support async sharing with the main thread
    Output: gurobi mip model solving lb and status
    """

    def gen_timestamp():
        return str(int(time.time() * 100.0) % 100000000)

    timeout = arguments.Config["bab"]["timeout"]
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]

    build_solver_model(m, timeout, mip_multi_proc=mip_multi_proc, 
                    mip_threads=mip_threads, model_type="mip", x=x, intermediate_bounds=intermediate_bounds)

    out_vars = m.net[m.net.final_name].solver_vars
    lb = m.net.final_node().lower[0].tolist()
    ub = [float('inf') for _ in lb]

    print('lower bounds for all target labels:', lb)
    candidates, candidate_neuron_ids, candidate_c_rows = [], [], []
    if labels_to_verify is not None: # sort the labels
        for pidx in labels_to_verify:
            if lb[pidx] >= 0: continue # skip the label with intial bound >= 0
            if solver_pkg == 'gurobi':
                candidates.append(out_vars[pidx].VarName)
            else:
                candidates.append(out_vars[pidx].name)
            candidate_neuron_ids.append(pidx)
            candidate_c_rows.append(m.c[:, pidx: pidx+1])
    else:
        for pidx, lbi in enumerate(lb):
            if lbi >= 0: continue
            if solver_pkg == 'gurobi':
                candidates.append(out_vars[pidx].VarName)
            else:
                candidates.append(out_vars[pidx].name)
            candidate_neuron_ids.append(pidx)
            candidate_c_rows.append(m.c[:, pidx: pidx+1])

    # Solve the worst label first.
    # candidates, candidate_neuron_ids = zip(*sorted(zip(candidates, candidate_neuron_ids), key=lambda x: lb[x[1]]))
    # candidate_neuron_ids = [1]
    # candidates = candidates[1:2]  # Change here to run just a specific label for testing.
    print('Starting MIP solver for these labels:', candidate_neuron_ids)
    if save_mps:
        ##### [CPLEX CUT] here we need to loop over all the candidate obj vars in asynchronized manner #####
        if process_dict is None:
            processes = {}
        else:
            processes = process_dict
        print("start creating model mps for candidates:", candidates)
        # have to be careful with the candidates name here!
        # our_vars has totally 9 obj vars and None for true label, var name after true label should add one:
        # [<gurobi.Var lay/16_0>, <gurobi.Var lay/16_1>, ... , <gurobi.Var lay/16_4>, None, <gurobi.Var lay/16_5>, ... , <gurobi.Var lay/16_8>]
        # candidate_neuron_ids is unverified pidx: [0, 1, 2, 3, 4, 6, 7, 8, 9]

        mps_pool_context = []
        model_filename_stamped_dict = {}
        model_c_row_dict = {}

        # FIXME: here we only consider Gurobi model yet
        global build_the_model_mip_proto_gurobi_model
        build_the_model_mip_proto_gurobi_model = m.net.model
        build_the_model_mip_proto_gurobi_model.update()
        for cidx, candidate in enumerate(candidates):
            pidx = candidate_neuron_ids[cidx]
            model_filename = os.path.join(CPLEX_FOLDER, f"cplexmip_lay{m.net.final_node().name.replace('/', '-')}_starttime={arguments.Globals['starting_timestamp']}_idx={arguments.Globals['example_idx']}_spec={pidx}")
            model_filename_stamped = model_filename + '_' + gen_timestamp()
            model_filename_stamped_dict[pidx] = model_filename_stamped
            model_c_row_dict[pidx] = candidate_c_rows[cidx]
            mps_pool_context.append((candidate, model_filename_stamped))

        save_mps_pool = multiprocessing.pool.Pool()
        save_mps_pool.map(_build_the_model_mip_mps_save, mps_pool_context)
        save_mps_pool.close()
        save_mps_pool.join()
        print('parallel mps save finish')

        for pidx in model_filename_stamped_dict:
            # import pdb; pdb.set_trace()
            # t1 = multiprocessing.Process(target=f"./{CPLEX_FOLDER}/get_cuts", args=[f"{model_filename}.mps", f"{CPLEX_FOLDER}/{model_filename}"])
            # t1 = multiprocessing.Process(target=run_get_cuts_subprocess, args=[CPLEX_FOLDER, model_filename])
            # t1.start()
            # processes[candidate_neuron_ids[cidx]] = t1
            model_filename_stamped = model_filename_stamped_dict[pidx]
            model_c_row = model_c_row_dict[pidx].detach().cpu()
            try:
                # use a try-catch block to try to ignore the interrupt signal between these two lines
                # in an effort to avoid process termination *between* two following lines
                # such termination results in absense of process records and ignorance of killing the launched process ->
                # a possible cause of orphan process
                proc, logfile = run_get_cuts_subprocess(model_filename_stamped)
                processes[pidx] = {'pid': proc.pid, '_logfile': logfile, '_fname_stamped': model_filename_stamped, 'c': model_c_row}
            except Exception as e:
                try:
                    proc.kill()
                except Exception as ee: pass
                raise e

        ###### run in a sequential way now, need to change to asynchronized manner later ######
        # for t1 in processes:
        #     t1.join()

        # subprocess.run([f"./{CPLEX_FOLDER}/get_cuts", f"{model_filename}.mps", f"{CPLEX_FOLDER}/{model_filename}"])
        del m.net.model
        # exit()
        return None, None, None, processes

    # MULTITHREAD to solve mip for all targeted labels
    global multiprocess_mip_model, stop_multiprocess
    multiprocess_mip_model = m.net.model
    global mip_solve_time_start
    mip_solve_time_start = time.time()
    input_names = None
    mip_adv = None
    if save_adv:
        input_shape = np.array(m.net.input_vars).shape
        input_names = [input_var.VarName for input_var in np.array(m.net.input_vars).reshape(-1).tolist()]

    if adv_warmup:
        adv, max_values, min_values = _intermediate_PGD_attack(m, "final", restarts=3, attack_iters=50, alpha=None)
        # convert 9 labels with m.c to 10 labels
        min_values, max_values = min_values.tolist(), max_values.tolist()
        layer_size = m.c.shape[1]
        adv = adv.cpu().tolist()
        adv_list = []
        for neuron_idx in candidate_neuron_ids:
            adv_list.append((adv[neuron_idx + layer_size], adv[neuron_idx])) # (low adv, max adv)
            assert min_values[neuron_idx] >= lb[neuron_idx]
        candidates = [(name, adv, input_names) for name, adv in zip(candidates, adv_list)]
    else:
        candidates = [(name, None, input_names) for name in candidates]

    with multiprocessing.Pool(mip_multi_proc) as pool:
        solver_result = pool.starmap(mip_solver_lb_ub, candidates, chunksize=1)
    # solver_result = mip_solver_lb_ub(candidates[1])

    multiprocess_mip_model = None
    stop_multiprocess = False

    ### NEED TO FIX: safe means obj > rhs instead of 0 ###
    status = [-1 for i in lb]
    for (vlb, vub, s, adv_new), pidx in zip(solver_result, candidate_neuron_ids):
        lb[pidx] = vlb
        ub[pidx] = vub
        status[pidx] = s
        if adv_new is not None: mip_adv = adv_new
    lb, ub = torch.tensor(lb), torch.tensor(ub)
    if save_adv and adv_new is not None:
        mip_adv = np.array(adv_new).reshape(input_shape).tolist()
    return lb, ub, status, mip_adv

def run_get_cuts_subprocess(model_filename):
    # remove legancy file to avoid collision
    cut_file_path = f"{model_filename}.cuts"
    idx_file_path = f"{model_filename}.indx"
    log_file_path = f"{model_filename}.log"
    if os.path.exists(cut_file_path):
        os.remove(cut_file_path)
    if os.path.exists(idx_file_path):
        os.remove(idx_file_path)
    # subprocess.run([f"./{CPLEX_FOLDER}/get_cuts", f"{model_filename}.mps", f"{CPLEX_FOLDER}/{model_filename}"])
    # return
    try:
        # only int file descriptor can be serialized across processes
        logfile = open(log_file_path, "w")
    except Exception as e:
        print('Cannot open log file for cuts solver.')
        print(e)
        logfile = None
    return subprocess.Popen([f"{CPLEX_FOLDER}/get_cuts", f"{model_filename}.mps", f"{model_filename}"],
            stderr=subprocess.STDOUT, stdout=logfile), logfile.fileno()  # Also returns the logfile handle so it can be closed later.


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
    upper_temp = upper_bound.relu()
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
    number_bounds = lower_bounds[-1].shape[0]

    for layer in reversed(net.relus):
        ratio = lAs[relu_idx]
        this_layer_mask = mask[relu_idx].unsqueeze(1)
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                    upper_bounds[pre_relu_indices[relu_idx]])
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
        score.insert(0, (abs(score_candidate).view(batch, number_bounds, -1) * this_layer_mask).mean(1))

        relu_idx -= 1

    return score


def solve_diving_lp(m, primal_vars, integer_vars, lower_bounds, upper_bounds):
    """
    A customized lp gurobi model to get the dual and primal values for each constraint and neurons
    """
    # only support mlp models for now
    orig_model = m.net.model
    diving_model = orig_model.copy()
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
                nu[neuron_idx] = orig_model.getConstrByName(f'lay{i+1}_{neuron_idx}_eq').pi
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
                    pi = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_0').pi  # dual variable for upper bound \hat{x} <= x - l + l * z
                    gamma = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_1').pi  # dual variable for lower bound \hat{x} >= x.
                    tau = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_2').pi  # dual variable for another upper bound \hat{x} <= u * z
                    mu = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_3').pi  # dual variable for lower bound \hat{x} >= 0.
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
    print(f'original obj: {orig_model.objval}, new_obj: {diving_model.objval}, status: {diving_model.status}')
    return


# update mip model by manually fixing intermediate relus
def update_mip_model_fix_relu(m, relu_idx, status, target=None, async_mip=False, best_adv=None, adv_activation_pattern=None,
                                refined_lower_bounds=None, refined_upper_bounds=None):
    '''
    relu_idx: indices of relu to be fixed
    status: the status of the relu
    '''
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]

    batch_num = len(relu_idx)
    model_candidates = []
    relu_layer_names = [relu_layer.name for relu_layer in m.net.relus]
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in m.net.relus]

    for b in range(batch_num):
        lb, ub = None, None
        if refined_lower_bounds is not None:
            lb = [lower_bound[b:b+1].cpu().detach().numpy() for lower_bound in refined_lower_bounds]
            ub = [upper_bound[b:b+1].cpu().detach().numpy() for upper_bound in refined_upper_bounds]
        out_vars = m.net.final_node().solver_vars
        m.relu_indices_mask = None
        # if len(m.gurobi_vars[-1]) == 1:
        if len(out_vars) == 1:
            # Only a single target.
            model_candidates.append((relu_idx[b], status[b], out_vars[0].VarName, m.input_shape, best_adv[b],
                                adv_activation_pattern[b], pre_relu_layer_names, relu_layer_names, lb, ub))
        else:
            # we only have 9 labels in out_vars for not considering predicted label
            if target is None: target = (m.c == -1).view(-1).nonzero().item()
            # Multiple labels; need to choose the target label.
            model_candidates.append((relu_idx[b], status[b], out_vars[target].VarName, m.input_shape, best_adv[b],
                                adv_activation_pattern[b], pre_relu_layer_names, relu_layer_names, lb, ub))

    # MULTITHREAD
    global multiprocess_mip_model, stop_multiprocess
    stop_multiprocess = False
    multiprocess_mip_model = m.net.model

    ##### testing cases with/without refined not using async parallel for debugging #####
    # import pdb; pdb.set_trace()
    # mip_solver_attack(model_candidates[0])
    # b = 0
    # model_candidates[0] = (relu_idx[b], status[b], out_vars[target].VarName, m.input_shape, best_adv[b],
    #                             adv_activation_pattern[b], pre_relu_layer_names, relu_layer_names, None, None)
    # mip_solver_attack(model_candidates[0])
    # exit()
    #####

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


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def _intermediate_PGD_attack(m, relu_idx, restarts=3, attack_iters=50, alpha=None):
    lb, ub = m.x.ptb.x_L, m.x.ptb.x_U
    X = m.x.detach().clone()

    m.net(X) # one clean forward to initialize all forward_value
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
    if X.ndim == 4:
        X = X.repeat(batch_size, 1, 1, 1)
        # LB, UB = lb.repeat(batch_size, 1, 1, 1), ub.repeat(batch_size, 1, 1, 1)
    elif X.ndim == 2:
        X = X.repeat(batch_size, 1)
        # LB, UB = lb.repeat(batch_size, 1), ub.repeat(batch_size, 1, 1, 1)
    else:
        print("Warning: no such X shape supported!")

    if alpha is None:
        alpha = (ub - lb).max() / 10.

    # print(forward_value[0, 10])
    max_values = -torch.ones(layer_size).cuda() * 1e8
    min_values = torch.ones(layer_size).cuda() * 1e8
    best_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(0,1)
        delta = delta * (ub - lb) + lb
        delta = (delta - X).detach()
        delta.requires_grad = True
        assert ((delta + X) > ub).sum() == 0
        assert ((delta + X) < lb).sum() == 0
        for _ in range(attack_iters):
            m.net(X + delta)
            if relu_idx != "final":
                forward_value = m.net.relus[relu_idx].inputs[0].forward_value
            else:
                forward_value = m.net.final_node().forward_value.mm(m.c[0].T)
            maxv = forward_value[:layer_size].masked_select(torch.eye(layer_size).bool().cuda())
            minv = forward_value[layer_size:].masked_select(torch.eye(layer_size).bool().cuda())
            # print(maxv[218], minv[218])
            # print(zz, maxv[1], minv[1])
            loss = maxv.sum() - minv.sum()
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), lb - X, ub - X)
            delta.grad.zero_()
        m.net(X + delta)
        if relu_idx != "final":
            forward_value = m.net.relus[relu_idx].inputs[0].forward_value
        else:
            forward_value = m.net.final_node().forward_value.mm(m.c[0].T)
        maxv = forward_value[:layer_size].masked_select(torch.eye(layer_size).bool().cuda())
        minv = forward_value[layer_size:].masked_select(torch.eye(layer_size).bool().cuda())
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
    m.net(m.x)
    return X + best_delta, max_values, min_values


##### still need to update for general graph!!! #####
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
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]

    global mip_refine_timeout
    mip_refine_timeout = arguments.Config["solver"]["mip"]["refine_neuron_time_percentage"] * arguments.Config["bab"]["timeout"]

    # preset the args for incomplete full crown with refined bounds
    m.net.init_slope((m.x,), share_slopes=share_slopes, c=m.c)
    m.net.set_bound_opts({'verbosity': 1})
    m.net.set_bound_opts({'optimize_bound_args': {'iteration': 100, 'enable_beta_crown': False, 'enable_alpha_crown': True,
                                                  'use_shared_alpha': share_slopes, 'optimizer': optimizer,
                                                  'early_stop': False,
                                                  'keep_best': True, 'fix_intermediate_layer_bounds': True,
                                                  'lr_alpha': lr_init_alpha, 'init_alpha': False,
                                                  'loss_reduction_func': loss_reduction_func,
                                                  'stop_criterion_func': stop_criterion_func,
                                                  'lr_decay': lr_decay}})
    lb_refined, ub_refined = None, None

    # Initialize the model
    m.model = grb.Model()
    m.model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
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
    # m.model.setParam('MIPFocus', 3)
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
        dim = 0
        for chan in range(input_domain.size(0)):
            chan_vars = []
            for row in range(input_domain.size(1)):
                row_vars = []
                for col in range(input_domain.size(2)):
                    lb = input_domain[chan, row, col, 0]
                    ub = input_domain[chan, row, col, 1]
                    v = m.model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'inp_{dim}')
                                            # name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                    dim += 1
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
    # need to handle the cases where unstabled neurons are refined to stable
    # (this relu_idx layer neuron idx, 1:>0, -1:<0)
    unstable_to_stable = [[] for _ in m.net.relus]
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

            weight = layer.weight.clone()
            bias = layer.bias.clone()
            if layer == m.layers[-1] and m.c is not None:
                weight = m.c.squeeze(0).mm(weight)
                bias = m.c.squeeze(0).mm(bias.unsqueeze(-1)).view(-1)

            print(layer, relu_idx, layer_idx, out_lbs.shape)

            candidates = []
            candidate_neuron_ids = []
            for neuron_idx in range(weight.size(0)):
                lin_expr = bias[neuron_idx].item()
                coeffs = weight[neuron_idx, :]

                if arguments.Config["solver"]["mip"]["mip_solver"] == 'gurobi':
                    lin_expr += grb.LinExpr(coeffs, m.gurobi_vars[-1])
                else:
                    for i in range(len(coeffs)):
                        lin_expr += coeffs[i] * m.gurobi_vars[-1][i]

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

            ######## update inf to all current layer bounds!!! #########
            m.model.update()
            ########

            refine_time = time.time()
            if need_refine and (relu_idx >= 1 and relu_idx < len(m.net.relus)) and (time.time() - mip_refine_time_start < mip_refine_timeout):
                multiprocess_mip_model = m.model

                if relu_idx == 1:
                    if adv_warmup:
                        # create pgd adv list as mip refinement warmup
                        adv, max_values, min_values = _intermediate_PGD_attack(m, relu_idx, restarts=3, attack_iters=50, alpha=None)
                        adv_list = []
                        layer_size = weight.size(0)
                        for neuron_idx in candidate_neuron_ids:
                            adv_list.append((adv[neuron_idx + layer_size].cpu().tolist(), adv[neuron_idx].cpu().tolist())) # (low adv, max adv)
                            # assert min_values[neuron_idx] >= lower_bounds[relu_idx][0, neuron_idx] and max_values[neuron_idx] <= upper_bounds[relu_idx][0, neuron_idx]
                        candidates = [(name, adv) for name, adv in zip(candidates, adv_list)]

                        # the second relu layer where mip refine starts
                        with multiprocessing.Pool(mip_multi_proc) as pool:
                            solver_result = pool.starmap_async(mip_solver, candidates, chunksize=1)

                            if relu_idx + 1 < len(m.net.relus):
                                # create adv list for next relu layer if still have next relu layer
                                adv, max_values, min_values = _intermediate_PGD_attack(m, relu_idx + 1, restarts=3, attack_iters=50, alpha=None)
                            solver_result = solver_result.get()
                    else:
                        # the second relu layer where mip refine starts
                        with multiprocessing.Pool(mip_multi_proc) as pool:
                            solver_result = pool.map(mip_solver, candidates, chunksize=1)

                    ##### serial test #####
                    # if adv_warmup:
                    #     mip_solver(candidates[10][0], candidates[10][1])
                    # else:
                    #     mip_solver(candidates[10])
                    # import pdb; pdb.set_trace()
                    ####################

                    lb_refined_sum, ub_refined_sum, refined_num = 0., 0., 0
                    for (vlb, vub, refined), neuron_idx in zip(solver_result, candidate_neuron_ids):
                        if refined:
                            # v = new_layer_gurobi_vars[neuron_idx]
                            vlb = max(vlb, lower_bounds[relu_idx][0, neuron_idx]) #
                            vub = min(vub, upper_bounds[relu_idx][0, neuron_idx]) #
                            refined_num += 1
                            lb_refined_sum += vlb - lower_bounds[relu_idx][0, neuron_idx]
                            ub_refined_sum += upper_bounds[relu_idx][0, neuron_idx] - vub
                            lower_bounds[relu_idx][0, neuron_idx] = vlb
                            upper_bounds[relu_idx][0, neuron_idx] = vub
                            if vlb >= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, 1))
                            if vub <= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, -1))

                        v = new_layer_gurobi_vars[neuron_idx]
                        v.LB, v.UB = lower_bounds[relu_idx][0, neuron_idx], upper_bounds[relu_idx][0, neuron_idx]
                    m.model.update()
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

                    # max_splits_per_layer = len(unstable_to_stable[relu_idx])
                    batch = 1
                    device = m.net.device
                    if not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
                        for relu_layer in m.net.relus:
                            relu_layer.sparse_beta = torch.zeros(size=(1, 0), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                            relu_layer.sparse_beta_loc = torch.zeros(size=(1, 0), dtype=torch.int64, device=device, requires_grad=False)
                            relu_layer.sparse_beta_sign = torch.zeros(size=(1, 0), dtype=torch.get_default_dtype(), device=device, requires_grad=False)
                    else:
                        max_splits_per_layer = [0 for _ in range(len(m.net.relus))]
                        # max_splits_per_layer[1] = len(unstable_to_stable[relu_idx])
                        for ridx, relu_layer in enumerate(m.net.relus):
                            relu_layer.sparse_beta, relu_layer.sparse_beta_loc, relu_layer.sparse_beta_sign = {}, {}, {}
                            for key in relu_layer.alpha.keys():
                                relu_layer.sparse_beta[key] = torch.zeros(size=(1, 0), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                                relu_layer.sparse_beta_loc[key] = torch.zeros(size=(1, 0), dtype=torch.int64, device=device, requires_grad=False)
                                relu_layer.sparse_beta_sign[key] = torch.zeros(size=(1, 0), dtype=torch.get_default_dtype(), device=device, requires_grad=False)

                    ##### debug the beta crown bound propagate with mip refined bounds #####
                    # max_splits_per_layer = len(unstable_to_stable[maximum_refined_relu_layers])
                    # refined_relu_layer = m.net.relus[maximum_refined_relu_layers]
                    # for key in refined_relu_layer.sparse_beta.keys():
                    #     # init all intermediate betas
                    #     refined_relu_layer.sparse_beta[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                    #     refined_relu_layer.sparse_beta_loc[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.int64, device=device, requires_grad=False)
                    #     refined_relu_layer.sparse_beta_sign[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=False)
                    # for neuron_idx, (stable_neuron_idx, sign) in enumerate(unstable_to_stable[maximum_refined_relu_layers]):
                    #     for key in refined_relu_layer.sparse_beta.keys():
                    #         # assign split constraint into all intermdiate betas
                    #         refined_relu_layer.sparse_beta_loc[key][0, neuron_idx] = stable_neuron_idx
                    #         refined_relu_layer.sparse_beta_sign[key][0, neuron_idx] = sign
                    # print("relu layer:", maximum_refined_relu_layers, "has unstable to stable neurons:", unstable_to_stable[maximum_refined_relu_layers])
                    # intermediate_layer_bounds, reference_bounds = {}, {}
                    # # for i, layer in enumerate(m.net.relus):
                    # # only refined with the second relu layer
                    # for i, layer in enumerate(m.net.relus):
                    #     # only refined with the relu layers that are refined by mip before
                    #     # if i>=(maximum_refined_relu_layers+1): break #### remember we don't need to set it in regular run!
                    #     nd = m.net.relus[i].inputs[0].name
                    #     print('reference bound:', i, nd, lower_bounds[i].shape)
                    #     intermediate_layer_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
                    #     reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]

                    # # print(f"{max_splits_per_layer} neurons are refined to stable!!!")
                    # m.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': True},
                    #             'enable_opt_interm_bounds': arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']})
                    # # m.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': False, 'lr_beta': 0.}})

                    # lb_refined, ub_refined = m.net.compute_bounds(x=(x,), IBP=False, C=m.c, method='CROWN-optimized', return_A=False, reference_bounds=reference_bounds, bound_upper=False)
                    # print("alpha-CROWN with intermediate bounds by MIP:", lb_refined, ub_refined)
                    # import pdb; pdb.set_trace()
                    ##########

                else:
                    with multiprocessing.Pool(mip_multi_proc) as pool:
                        if adv_warmup:
                            adv_list = []
                            layer_size = weight.size(0)
                            for neuron_idx in candidate_neuron_ids:
                                adv_list.append((adv[neuron_idx + layer_size].cpu().tolist(), adv[neuron_idx].cpu().tolist())) # (low adv, max adv)
                                # assert min_values[neuron_idx] >= lower_bounds[relu_idx][0, neuron_idx] and max_values[neuron_idx] <= upper_bounds[relu_idx][0, neuron_idx]
                            candidates = [(name, adv) for name, adv in zip(candidates, adv_list)]

                            solver_result = pool.starmap_async(mip_solver, candidates, chunksize=1)

                            if relu_idx + 1 < len(m.net.relus):
                                # create adv list for next relu layer if still have next relu layer
                                adv, max_values, min_values = _intermediate_PGD_attack(m, relu_idx + 1, restarts=3, attack_iters=50, alpha=None)
                        else:
                            solver_result = pool.map_async(mip_solver, candidates, chunksize=1)

                        if last_relu_layer_refined and (time.time() - mip_refine_time_start < mip_refine_timeout):
                            print(f"Run alpha-CROWN after refining layer {layer_idx-2} and relu idx {relu_idx-1}")
                            # using refined bounds with init opt crown for the previous optimized bounds
                            intermediate_layer_bounds, reference_bounds = {}, {}
                            # for i, layer in enumerate(m.net.relus):
                            # only refined with the second relu layer
                            for i, layer in enumerate(m.net.relus):
                                # only refined with the relu layers that are refined by mip before
                                # if i>=(maximum_refined_relu_layers+1): break
                                nd = m.net.relus[i].inputs[0].name
                                print(i, nd, lower_bounds[i].shape)
                                intermediate_layer_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
                                reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]

                            # config intermediate betas for last refined relu layer
                            # we need to use beta crown to fully consider neurons that are refined from unstable to stable
                            max_splits_per_layer = len(unstable_to_stable[maximum_refined_relu_layers])
                            refined_relu_layer = m.net.relus[maximum_refined_relu_layers]
                            if not arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
                                # init all regular betas
                                refined_relu_layer.sparse_beta = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                                refined_relu_layer.sparse_beta_loc = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.int64, device=device, requires_grad=False)
                                refined_relu_layer.sparse_beta_sign = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=False)
                                # assign split constraint into regular betas
                                for neuron_idx, (refined_neuron, sign) in enumerate(unstable_to_stable[maximum_refined_relu_layers]):
                                    refined_relu_layer.sparse_beta_loc[0, neuron_idx] = refined_neuron
                                    refined_relu_layer.sparse_beta_sign[0, neuron_idx] = sign
                            else:
                                for key in refined_relu_layer.sparse_beta.keys():
                                    # init all intermediate betas
                                    refined_relu_layer.sparse_beta[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                                    refined_relu_layer.sparse_beta_loc[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.int64, device=device, requires_grad=False)
                                    refined_relu_layer.sparse_beta_sign[key] = torch.zeros(size=(1, max_splits_per_layer), dtype=torch.get_default_dtype(), device=device, requires_grad=False)
                                for neuron_idx, (stable_neuron_idx, sign) in enumerate(unstable_to_stable[maximum_refined_relu_layers]):
                                    for key in refined_relu_layer.sparse_beta.keys():
                                        # assign split constraint into all intermdiate betas
                                        refined_relu_layer.sparse_beta_loc[key][0, neuron_idx] = stable_neuron_idx
                                        refined_relu_layer.sparse_beta_sign[key][0, neuron_idx] = sign
                            print("relu layer:", maximum_refined_relu_layers, "has unstable to stable neurons:", unstable_to_stable[maximum_refined_relu_layers])

                            m.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': True, "verbose": True},
                                'enable_opt_interm_bounds': arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']})

                            lb_refined, ub_refined = m.net.compute_bounds(x=(x,),
                                C=m.c, method='CROWN-optimized', return_A=False,
                                reference_bounds=reference_bounds, bound_upper=False)
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
                            vlb = max(vlb, lower_bounds[relu_idx][0, neuron_idx]) #
                            vub = min(vub, upper_bounds[relu_idx][0, neuron_idx]) #
                            refined_num += 1
                            lb_refined_sum += vlb - lower_bounds[relu_idx][0, neuron_idx]
                            ub_refined_sum += upper_bounds[relu_idx][0, neuron_idx] - vub
                            lower_bounds[relu_idx][0, neuron_idx] = vlb
                            upper_bounds[relu_idx][0, neuron_idx] = vub
                            if vlb >= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, 1))
                            if vub <= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, -1))

                        v = new_layer_gurobi_vars[neuron_idx]
                        v.LB, v.UB = lower_bounds[relu_idx][0, neuron_idx], upper_bounds[relu_idx][0, neuron_idx]
                    m.model.update()
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

                            if arguments.Config["solver"]["mip"]["mip_solver"] == 'gurobi':
                                lin_expr += grb.LinExpr(coeffs, gvars)
                            else:
                                lin_expr += coeffs@gvars

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
                                                        name=f'ReLU{relu_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                # binary indicator
                                a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{relu_idx}_[{chan_idx},{row_idx},{col_idx}]')

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
                                                name=f'ReLU{relu_idx}_{neuron_idx}')
                        # binary indicator
                        a = m.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{relu_idx}_{neuron_idx}')

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
        elif "Identity" in str(type(layer)):
            pass
        else:
            print(layer)
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
        intermediate_layer_bounds, reference_bounds = {}, {}
        # for i, layer in enumerate(m.net.relus):
        # only refined with the second relu layer
        for i, layer in enumerate(m.net.relus):
            # only refined with the relu layers that are refined by mip before
            # if i>=(maximum_refined_relu_layers+1): break
            nd = m.net.relus[i].inputs[0].name
            print(i, nd, lower_bounds[i].shape)
            intermediate_layer_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
            reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
        lb_refined, ub_refined = m.net.compute_bounds(x=(x,), C=m.c,
            method='CROWN-optimized', reference_bounds=reference_bounds, bound_upper=False)
        print("alpha-CROWN with intermediate bounds improved by MIP:", lb_refined, ub_refined)

    if lb_refined is None:
        if lower_bounds[-1].shape[1] != m.c.shape[1]:
            # remove true label 0 bounds according to C matrix
            lower_bounds[-1] = lower_bounds[-1].mm(-m.c[0].T)
            upper_bounds[-1] = upper_bounds[-1].mm(-m.c[0].T)
        return lower_bounds, upper_bounds

    lb_refined, ub_refined, pre_relu_indices = m.get_candidate(m.net, lb_refined, lb_refined + float('inf'))  # primals are better upper bounds
    # mask, lA = m.get_mask_lA_parallel(m.net)
    ##### save refined betas to bab if not verified #####
    # creating history: batch, relu layers, [[loc neuron_idx],[coeff 1 if>=0 else -1]]
    splits = [[[], []] for _ in m.net.relus]
    # creating history betas: batch, relu layers, [beta tensor for this layer]
    retb = []
    for mi, relu_layer in enumerate(m.net.relus):
        max_splits_per_layer = len(unstable_to_stable[mi])
        for neuron_idx, coeff in unstable_to_stable[mi]:
            splits[mi][0].append(neuron_idx)
            splits[mi][1].append(coeff)
        # Save only used beta, discard padding beta.
        if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
            val_i = []
            for key in relu_layer.sparse_beta.keys():
                # val_i.append([relu_layer.sparse_beta[key].cpu()[0, :max_splits_per_layer]])
                # we only save betas for last layer optimization for now; the rest layer betas are not saved.
                if key == m.net.final_name: val_i.append(relu_layer.sparse_beta[key].cpu()[0, :max_splits_per_layer])
            retb.append(val_i)
        else:
            retb.append(relu_layer.sparse_beta.cpu()[0, :max_splits_per_layer])
    return lb_refined, ub_refined, ([splits], [retb])

