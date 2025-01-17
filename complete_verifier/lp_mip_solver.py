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
"""Build, solve, and refine output bounds using gurobi LP/MIP solver based on the bounds obtained by auto_LiRPA."""

import time
import random
import multiprocessing
import multiprocessing.pool
import sys
import os
import shutil
import subprocess
import signal

import numpy as np

import torch
import torch.nn as nn

import arguments

from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import stop_criterion_min, reduction_str2func
from auto_LiRPA.bound_ops import (
    BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd,
    BoundRelu, BoundInput)
from auto_LiRPA.beta_crown import SparseBeta

import beta_CROWN_solver
from utils import get_reduce_op, get_batch_size_from_masks
try:
    from scip_model import SCIPModel, EarlyStopEvent, GenerateCutsEvent
except:  # pylint: disable=bare-except
    pass
try:
    import gurobipy as grb
except ModuleNotFoundError:
    pass

from typing import TYPE_CHECKING, Callable, List
if TYPE_CHECKING:
    from .beta_CROWN_solver import LiRPANet

from attack.attack_pgd import check_and_save_cex

CPLEX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuts/CPLEX_cuts")

def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise grb.GurobiError(message)


multiprocess_mip_model = None
multiprocess_lp_model = None
stop_multiprocess = False
mip_refine_time_start = None
mip_refine_timeout = 230
mip_solve_time_start = None



def check_enable_refinement(ret_incomplete_verification):
    model = ret_incomplete_verification['model'].net
    lower_bounds = ret_incomplete_verification['lower_bounds']
    upper_bounds = ret_incomplete_verification['upper_bounds']
    print('Checking if MIP refinement should be enabled')
    unstable = 0
    for node in model.nodes():
        if not node.perturbed:
            continue
        if isinstance(node, BoundConv):
            print(f'{node} is not supported in MIP refinement.')
            return 'bab'
        if len(node.requires_input_bounds) > 0:
            if not isinstance(node, BoundRelu):
                print(f'{node} is not supported in MIP refinement.')
                return 'bab'
            key = node.inputs[0].name
            unstable_ = ((lower_bounds[key] < 0) & (upper_bounds[key] > 0)).sum()
            print(f'Unstable ReLU neurons for {node}: {unstable_}')
            unstable += unstable_
    enable = unstable >= arguments.Config['solver']['mip']['unstable_neuron_threshold']
    if not enable:
        print('MIP refinement is disabled.')
        return 'bab'
    else:
        print('MIP refinement is enabled.')
        return 'bab-refine'

      
def mip(model, ret_incomplete, labels_to_verify=None, mip_skip_unsafe=False, vnnlib=None, pgd_attack_example=None, verifier=None):
    ret = {key: None for key in [
        'global_lb', 'lower_bounds', 'upper_bounds', 'refined_betas']}

    if verifier == "mip":
        mip_global_lb, mip_global_ub, mip_status, adv_example = model.build_the_model_mip(
            labels_to_verify=labels_to_verify, save_adv=arguments.Config["general"]["save_adv_example"], mip_skip_unsafe=mip_skip_unsafe, pgd_attack_example=pgd_attack_example)
        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(-1)  # Missing batch dimension.
        if mip_global_ub.ndim == 1:
            mip_global_ub = mip_global_ub.unsqueeze(-1)  # Missing batch dimension.
        print(f'MIP solved lower bound: {mip_global_lb}')
        print(f'MIP solved upper bound: {mip_global_ub}')
        ret['global_lb'] = mip_global_lb
        verified_status = "safe-mip"
        # Batch size is always 1.
        labels_to_check = labels_to_verify if labels_to_verify is not None else range(len(mip_status))
        for pidx in labels_to_check:
            if mip_global_lb[pidx] >= 0:
                # Lower bound > 0, verified.
                continue
            # Lower bound < 0, now check upper bound.
            if mip_global_ub[pidx] <= 0:
                # Must be 2 cases: solved with adv example, or early terminate with adv example.
                assert mip_status[pidx] in [2, 15]
                if mip_skip_unsafe:
                    return "unknown-mip", ret
                else:
                    print("verified unsafe-mip with init mip!")
                    verified_status = 'unsafe-mip'
                    # Save counterexample if the pgd_order is not skip.
                    if adv_example is not None:
                        # The vnnlib should be provided when save_adv is enableds
                        assert vnnlib is not None
                        # Reshape the counterexample for model. The shape of mip_adv is [batch_size, input_shape], here batch size should always be 1.
                        mip_adv = torch.tensor(adv_example).reshape(1, *model.x.shape).to(model.device)
                        attack_output = model.model_ori(mip_adv.view(-1, *model.x.shape[1:]))
                        verified_status, _ = check_and_save_cex(mip_adv, attack_output, vnnlib,
                                                                arguments.Config["attack"]["cex_path"], "unsafe-mip")
                    return verified_status, ret
            # Lower bound < 0 and upper bound > 0, must be a timeout.
            assert mip_status[pidx] == 9 or mip_status[pidx] == -1, "should only be timeout for label pidx"
            verified_status = "unknown-mip"
        print(f"verified {verified_status} with init mip!")
        return verified_status, ret
    elif verifier == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = FSB_score(model, ret_incomplete)
        refined_lower_bounds, refined_upper_bounds, refined_betas = model.build_the_model_mip_refine(
            ret_incomplete['lower_bounds'], ret_incomplete['upper_bounds'],
            score=score, stop_criterion_func=stop_criterion_min(1e-4))

        # To ensure compatibility with verified-acc, we need to reshape the final layer bound to [batch, 1].
        # Previously, the final layer bound was in the shape [1, batch].
        # Additionally, `ret_incomplete['lower_bounds'][model.final_name].shape` can be [pruned_batch, 1].
        # Therefore, we only need to transpose the final layer bound to achieve the desired shape.
        assert refined_lower_bounds[model.final_name].shape[0] == 1
        refined_lower_bounds[model.final_name] = refined_lower_bounds[model.final_name].T
        refined_upper_bounds[model.final_name] = refined_upper_bounds[model.final_name].T
        lower_bounds, upper_bounds, = refined_lower_bounds, refined_upper_bounds
        refined_global_lb = lower_bounds[model.final_name]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())

        # save output
        if arguments.Config['general']['save_output']:
            arguments.Globals['out']['refined_lb'] = refined_global_lb.cpu()

        ret.update({
            'global_lb': refined_global_lb,
            'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds,
            'refined_betas': refined_betas,
        })
        ret['refined_betas'] = tuple([
            { r.inputs[0].name: refined_betas[t][i][j]
             for j, r in enumerate(model.net.relus)}
            for i in range(len(refined_betas[t]))
        ] for t in range(2))
        if refined_global_lb.min()>=0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", ret
        else:
            return "unknown", ret
    else:
        ret.update({
            'global_lb': -float("inf"),
            'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds,
        })
        return "unknown", ret


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
    # model.setParam('BestBdStop', 1e-5)  # Terminate as long as we find a positive lower bound.
    # model.setParam('BestObjStop', -1e-5)  # Terminate as long as we find a adversarial example.
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
    model.setParam('BestBdStop', 1e-5)  # Terminate as long as we find a positive lower bound.
    model.setParam('BestObjStop', -1e-5)  # Terminate as long as we find a adversarial example.
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
            model.setParam('BestBdStop', 1e-5)  # Terminate as long as we find a positive lower bound.
            model.setParam('BestObjStop', -1e-5)  # Terminate as long as we find a adversarial example.
        elif arguments.Config["solver"]["mip"]["mip_solver"] == 'scip':
            model.includeEventhdlr(EarlyStopEvent(), "EarlyStopEvent", "python event handler to early stop.")  # FIXME (01/13/22): threshold should be changeable, based on specification.
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
        if not mip_skip_unsafe:
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


def build_solver_model(
    m: 'LiRPANet',
    timeout,
    mip_multi_proc=None,
    mip_threads=1,
    model_type="mip",
    x=None,
    intermediate_bounds=None,
    include_C=True,
    final_layer_name=None,
    model_modifier_callback: Callable = None,
):
    """
    m is the instance of LiRPANet
    model_type ["mip", "lp", "lp_integer"]: three different types of guorbi solvers
    lp_integer using mip formulation but continuous integer variable from 0 to 1 instead of
    binary variable as mip; lp_integer should have the same results as lp but allowing us to
    estimate integer variables.
    NOTE: we build lp/mip solver from computer graph
    """
    if include_C:
        C = m.c
    else:
        C = None
    if final_layer_name is None:
        final_layer_name = m.net.final_name
    build_mip_start_time = time.time()
    if m.pool is not None:
        # Must close the pool because the old model shared to the pool workers is now stale.
        print("Closing MIP Pool...")
        m.pool.close()
        m.pool.terminate()
        m.pool.kill()
        m.pool = None
        m.pool_termination_flag = None
    m.net.solver_model_type = model_type
    m.net.solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]

    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    ### Merge the current params to a new solver model build function
    # Initialize the model
    if m.net.solver_pkg == 'gurobi':
        m.net.solver_model = grb.Model()
    elif m.net.solver_pkg == 'scip':
        m.net.solver_model = SCIPModel()
    else:
        raise NotImplementedError
    # Layers must be reset, otherwise variables won't be recreated
    m.net._reset_solver_vars(m.net[final_layer_name])

    ### Merge the current params to a new solver model config function
    m.net.solver_model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
    m.net.solver_model.setParam('Threads', mip_threads)
    m.net.solver_model.setParam("FeasibilityTol", 2e-5)
    m.net.solver_model.setParam('TimeLimit', timeout)
    cut_options = os.environ.get('ALPHA_BETA_CROWN_MIP_CUT_DEBUG', None)
    if cut_options is not None:
        m.net.solver_model.setParam('Cuts', 0)
        for cut in cut_options.strip().split(','):
            cut, val = cut.strip().split('=')
            val = int(val)
            if cut == 'Gomory':
                suffix = 'Passes'
            else:
                suffix = 'Cuts'
            print(f'Setting {cut}{suffix} to {val}')
            m.net.solver_model.setParam(f'{cut}{suffix}', val)
    print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads}, total threads used: {mip_multi_proc*mip_threads}")

    # build model in auto_LiRPA
    out_vars = m.net.build_solver_module(
        x=x, C=C, interm_bounds=intermediate_bounds,
        final_node_name=final_layer_name, model_type=model_type,
        solver_pkg=m.net.solver_pkg)
    if model_modifier_callback is not None:
        model_modifier_callback(m.net.solver_model)
    m.net.solver_model.update()
    build_mip_time = time.time() - build_mip_start_time
    print(f"{model_type} solver model built in {build_mip_time:.4f} seconds.")
    return out_vars


# updated function using general computation graph to build lp model
def build_the_model_lp(
        m: 'LiRPANet',
        using_integer=True,
        get_primals=False,
        optimized_layer_name=None,
        final_layer_name=None,
        compute_upper_bound=False,
        include_output_constraint=False,
        rhs=None,
    ):
    """
    Before the first branching, we build the solver model
    using_integer:
        True: using mip formulation with continuous integer variable (model_type="lp_integer")
        False: using triangle relaxation without integer variable (model_type="lp")
    Output: the lower bound by solver model
    NOTE: We only consider one output node for now
    """
    if optimized_layer_name is None:
        optimized_layer_name = m.net.final_name
    if final_layer_name is None:
        final_layer_name = m.net.final_name
    if include_output_constraint:
        assert rhs is not None

    timeout = arguments.Config["bab"]["timeout"]
    model_type = "lp"
    if using_integer: model_type = "lp_integer"
    m.layers = list(m.model_ori.children())

    def add_output_constraint(model):
        final_layer_vars = m.net.final_node().solver_vars
        assert len(final_layer_vars) == 1, len(final_layer_vars)
        final_layer_var = final_layer_vars[0]
        assert rhs.shape == (1,1), rhs
        model.addConstr(final_layer_var <= rhs, name='output_constraint')

    # build the solver models
    m.build_solver_model(
        timeout,
        model_type=model_type,
        include_C=(optimized_layer_name == final_layer_name or include_output_constraint),
        final_layer_name=(final_layer_name if include_output_constraint else optimized_layer_name),
        model_modifier_callback=add_output_constraint if include_output_constraint else None,
    )

    out_vars = m.net[optimized_layer_name].solver_vars
    glbs = []
    for obj in out_vars:
        guro_start = time.time()
        if compute_upper_bound:
            m.net.solver_model.setObjective(obj, grb.GRB.MAXIMIZE)
        else:
            m.net.solver_model.setObjective(obj, grb.GRB.MINIMIZE)
        try:
            m.net.solver_model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)

        status = m.net.solver_model.status
        assert status == 2, f"LP wasn't optimally solved status:{status}"
        # print(f"[{obj}]- status: {status}, time: {time.time() - guro_start}")
        glb = obj.X if status != 3 else None
        glbs.append(glb)

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

    return glbs


def update_model_bounds(solver_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, model_type="lp_integer"):
    """update solver vars bounds with given lower and upper bounds
    Args:
        solver_model: copied solver model from m.net.solver_model
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


def all_node_split_LP(arg):  
    pre_relu_layer_names, relu_layer_names, orig_out_vars, lower_bounds, upper_bounds, rhs, dix = arg
    global input_name
    # Suppress logs from Gurobi
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    with grb.Env() as env:
        global termination_flag_lp
        global multiprocess_lp_model
        env.setParam('LogToConsole', 0)
        env.start()
        sys.stdout = old_stdout
        devnull.close()
        if termination_flag_lp.value == 1:
            # Stop if a counterexample is already found
            return 'unknown', dix, float('inf'), None  
        
        all_node_model = copy_model(multiprocess_lp_model, basis=False, env=env)
        all_node_model = update_model_bounds(all_node_model, lower_bounds, upper_bounds,\
                    pre_relu_layer_names, relu_layer_names, 'lp')

        counterexample = None
        assert lower_bounds[-1].size(0) == 1,  "only bounds with batch size 1"
        assert len(orig_out_vars) == len(rhs), "out shape not matching!"

        # Add constraints to the output variable to limit its value
        for out_idx in range(len(orig_out_vars)):
            objVar = all_node_model.getVarByName(orig_out_vars[out_idx])
            all_node_model.addConstr(objVar <= rhs[out_idx])

        # Perform LP only once since all constraints are enforced. Here we only optimize towards minimizing the first output,
        # but all outputs should satisfy all constraints as long as we find a counterexample.
        out_idx = 0
        objVar = all_node_model.getVarByName(orig_out_vars[out_idx])
        decision_threshold = rhs[out_idx]

        all_node_model.setObjective(objVar, grb.GRB.MINIMIZE)
        all_node_model.update()
        try:
            all_node_model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)

        # The optimization will either yield a result or fail to satisfy the constraints
        # with lock:
        if all_node_model.status == 2:
            glb = objVar.X
        else:
            glb = float('inf')

        if glb > decision_threshold:
            lp_status = "safe"
        else:
            lp_status = 'unsafe'
            # Ensure that we only have one counterexample, avoid conflict when testing
            if termination_flag_lp.value == 1:
                del all_node_model
                return 'unknown', dix, float('inf'), None
            def extract_value(input_name):
                if isinstance(input_name, list):
                    return [extract_value(sub_input_name) for sub_input_name in input_name]
                else:
                    return all_node_model.getVarByName(input_name).X
            counterexample = extract_value(input_name)
            print(f'Verified to be unsafe with input counterexample {counterexample}')
            termination_flag_lp.value = 1
        del all_node_model
        return lp_status, dix, float(glb), counterexample


def update_the_model_cut(m, cut, pre_lbs=None, pre_ubs=None, split=None, verbose=False):
    """
    recalculate the bound propagation using lp solver with cut constraints and split constraints
    """
    timeout = arguments.Config["bab"]["timeout"]
    m.model_cut = copy_model(m.net.solver_model)
    primal_verbose = False

    # Assert that this is as expected a network with a single output
    orig_out_vars = m.net.final_node().solver_vars
    assert len(orig_out_vars) == 1, "Network doesn't have scalar output"

    pre_relu_layer_idx = []
    layer_idx, relu_idx = 0, 0
    m.layers = list(m.model_ori.children())
    for layer_idx, layer in enumerate(m.layers):
        if type(layer) is nn.ReLU:
            pre_relu_layer_idx.append(layer_idx)
            relu_idx += 1

    lower_bounds, upper_bounds = None, None
    if pre_lbs is not None:
        lower_bounds = [lbs.clone().detach() for lbs in pre_lbs]
        upper_bounds = [ubs.clone().detach() for ubs in pre_ubs]

    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in m.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in m.net.relus]

    if split is not None and pre_lbs is not None:
        # only support single neuron split for now
        gurobi_splits = []
        for split_idx, node in enumerate(split['decision']):
            if split["choice"][split_idx] == 1:
                split_var = m.model_cut.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}")
                gurobi_splits.append(m.model_cut.addConstr(split_var >= 0, name=f"split{split_idx}"))
                print(f"split_expr:{split_var}>=0")
                # orig_v = lower_bounds[node[0]].view(-1)[node[1]].item()
                lower_bounds[node[0]].view(-1)[node[1]] = 0.
                # if primal_verbose: print(orig_v, lower_bounds[node[0]].view(-1)[node[1]])
            else:
                split_var = m.solver_model_cut.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}")
                gurobi_splits.append(m.model_cut.addConstr(split_var <= 0, name=f"split{split_idx}"))
                print(f"split_expr:{split_var}<=0")
                upper_bounds[node[0]].view(-1)[node[1]] = 0.
        m.model_cut.update()

    if pre_lbs is not None:
        m.model_cut = update_model_bounds(m.model_cut, lower_bounds, upper_bounds,\
                pre_relu_layer_names, relu_layer_names, m.net.solver_model_type)

    # post activation name: 'ReLU{relu_layer_names[relu_idx]}_{neuron_idx}'
    # integer name: 'aReLU{relu_layer_names[relu_idx]}_{neuron_idx}'
    # pre activation name: 'lay{pre_relu_layer_names[layer_idx]}_{neuron_idx}'
    gurobi_cuts = []
    # without cut, how many cut constraints are satisfied with previous primal values
    sat_cnt = 0
    if cut is None:
        cut = []
        print("warning: no cuts in update_the_model_cut")
    for cut_idx, ci in enumerate(cut):
        # ci is each individual cut
        lin_expr = -ci["bias"]
        # skip this constraint if any neuron is not unstable and not in guorbi model any more
        constr_value = -ci["bias"]
        constr_str = f"{-ci['bias']} + "
        skip = False
        for node, coeff in zip(ci["x_decision"], ci["x_coeffs"]):
            if m.model_cut.getVarByName(f"inp_{node[1]}") is None:
                print(f"Warning: inp_{node[1]} not exists!")
                skip = True
                continue
            z = m.net.solver_model.getVarByName(f"inp_{node[1]}")
            constr_str += f"{coeff} * {z.X:.3f} + "
            constr_value += coeff * z.X
            lin_expr += grb.LinExpr(coeff, m.model_cut.getVarByName(f"inp_{node[1]}"))
        for node, coeff in zip(ci["relu_decision"], ci["relu_coeffs"]):
            if m.model_cut.getVarByName(f"ReLU{relu_layer_names[node[0]]}_{node[1]}") is None:
                print(f"Warning: ReLU{relu_layer_names[node[0]]}_{node[1]} not exists!")
                skip = True
                continue
            z = m.net.solver_model.getVarByName(f"ReLU{relu_layer_names[node[0]]}_{node[1]}")
            constr_str += f"{coeff} * {z.X:.3f} + "
            constr_value += coeff * z.X
            lin_expr += grb.LinExpr(coeff, m.model_cut.getVarByName(f"ReLU{relu_layer_names[node[0]]}_{node[1]}"))
        for node, coeff in zip(ci["arelu_decision"], ci["arelu_coeffs"]):
            if m.model_cut.getVarByName(f"aReLU{relu_layer_names[node[0]]}_{node[1]}") is None:
                print(f"Warning: aReLU{relu_layer_names[node[0]]}_{node[1]} not exists!")
                skip = True
                continue
            z = m.net.solver_model.getVarByName(f"aReLU{relu_layer_names[node[0]]}_{node[1]}")
            constr_str += f"{coeff} * {z.X:.3f} + "
            constr_value += coeff * z.X
            lin_expr += grb.LinExpr(coeff, m.model_cut.getVarByName(f"aReLU{relu_layer_names[node[0]]}_{node[1]}"))
        for node, coeff in zip(ci["pre_decision"], ci["pre_coeffs"]):
            if m.model_cut.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}") is None:
                print(f"Warning: lay{pre_relu_layer_names[node[0]]}_{node[1]} not exists!")
                skip = True
                continue
            z = m.net.solver_model.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}")
            constr_str += f"{coeff} * {z.X:.3f} + "
            constr_value += coeff * z.X
            lin_expr += grb.LinExpr(coeff, m.model_cut.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}"))

        if not skip:
            constr_sat = False
            if ci["c"] == 1:
                gurobi_cuts.append(m.model_cut.addConstr(lin_expr >= 0, name=f"cut{cut_idx}"))
                if verbose:
                    constr_sat = True if constr_value >= 0 else False
                    if constr_sat is False:
                        print(f"{cut_idx}: lin_expr:{lin_expr} >= 0")
                        if primal_verbose: print(f"{constr_str[:-2]} ({constr_value}) >= 0; SAT:{constr_sat}\n")
            else:
                gurobi_cuts.append(m.model_cut.addConstr(lin_expr <= 0, name=f"cut{cut_idx}"))
                if verbose:
                    constr_sat = True if constr_value <= 0 else False
                    if constr_sat is False:
                        print(f"{cut_idx}: lin_expr:{lin_expr} <= 0")
                        if primal_verbose: print(f"{constr_str[:-2]} ({constr_value}) <= 0; SAT:{constr_sat}\n")
            if constr_sat: sat_cnt += 1
        else:
            pass

    m.model_cut.update()
    if verbose: print('Finished building Gurobi LP model. Start solving the LP!')

    guro_start = time.time()
    objVar = m.model_cut.getVarByName(orig_out_vars[0].VarName)

    m.model_cut.setObjective(objVar, grb.GRB.MINIMIZE)
    m.model_cut.update()
    # m.model_cut.setObjective(objVar, grb.GRB.MAXIMIZE)
    # m.model_cut.write("base_model_cut.lp")
    m.model_cut.optimize()

    # assert m.model_cut.status == 2, f"model not optimized with status {m.model_cut.status}"
    if m.model_cut.status == 2:
        glb = objVar.X
    elif m.model_cut.status == 3:
        print("warning, gurobi infeasible!")
        glb = float('inf')
    else:
        print("model status not supported!")
        exit()
    # lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)
    print("#### cut gurobi glb:", glb)

    if split is not None:
        for c in gurobi_splits:
            print('The dual value of split %s : %g %g'%(c.constrName, c.pi, c.slack))

    num_nonzero_beta = 0
    sum_beta = 0.
    for c in gurobi_cuts:
        if verbose and c.pi > 0:
            print('The dual value of %s : %g %g'%(c.constrName, c.pi, c.slack))
        if c.pi != 0:
            num_nonzero_beta += 1
            # dual var might be negative depends on its >= or <=
            sum_beta += c.pi if c.pi >0 else -c.pi
    print(f"cut gurobi nonzero betas: {num_nonzero_beta}/{len(gurobi_cuts)}, beta sum: {sum_beta}, no cut primal SAT: {sat_cnt}\n")
    guro_end = time.time()
    print('Gurobi solved the LP with time', guro_end - guro_start)
    del m.model_cut
    # exit()
    return glb

build_the_model_mip_proto_gurobi_model = None
def _build_the_model_mip_mps_save(args):
    candidate, fname = args
    print(f"parallel save mip model to {fname}.mps")
    now_model = build_the_model_mip_proto_gurobi_model.copy()
    v = now_model.getVarByName(candidate)
    now_model.setObjective(v, grb.GRB.MINIMIZE)
    now_model.update()
    now_model.write(f'{fname}.mps')


# This handler ensures that the pool is correctly shut down on program termination, to avoid warnings regarding unclosed file pointers
def _signal_handler(signum, frame):
    print(f"Process {os.getpid()} received signal {signum}. Terminate pool {save_mps_pool}")
    save_mps_pool.terminate()
    exit(0)

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
    model = beta_CROWN_solver.LiRPANet(unwrapped_model, in_size=input_shape, c=c, device='cpu')
    build_the_model_mip(model, labels_to_verify=None, save_mps=save_mps, process_dict=process_dict, x=x, intermediate_bounds=intermediate_bounds)

# Global pointer to the process pool used to save the mps
# Must be accessable by the _signal_handler function
save_mps_pool = None
# updated function using general computation graph to build lp model
def build_the_model_mip(m, labels_to_verify=None, save_mps=False, process_dict=None, save_adv=False, x=None, intermediate_bounds=None, mip_skip_unsafe=False, pgd_attack_example=None):
    """
    Using the built gurobi model to solve mip formulation in parallel
    lower_bounds, upper_bounds: intermediate relu bounds from beta-crown
    simplified: only for target label if simplified, otherwise all labels remained
    if process_dict is a dict, then we will dynamically record the cut getting processes into process_dict
        to support async sharing with the main thread
    Output: gurobi mip model solving lb and status
    """
    global save_mps_pool

    if (arguments.Config['bab']['cut']['cplex_cuts']
            and arguments.Config['bab']['cut']['cuts_path'] != CPLEX_FOLDER):
        os.makedirs(arguments.Config['bab']['cut']['cuts_path'], exist_ok=True)
        shutil.copy(os.path.join(CPLEX_FOLDER, "get_cuts"), os.path.join(arguments.Config['bab']['cut']['cuts_path'], "get_cuts"))

    def gen_timestamp():
        return str(int(time.time() * 100.0) % 100000000)

    timeout = arguments.Config["bab"]["timeout"]
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
    pgd_order = arguments.Config["attack"]["pgd_order"]

    model_type = "lp_integer" if arguments.Config["solver"]["mip"]["lp_solver"] else "mip"
    build_solver_model(m, timeout, mip_multi_proc=mip_multi_proc,
                    mip_threads=mip_threads, model_type=model_type, x=x, intermediate_bounds=intermediate_bounds)

    out_vars = m.net[m.net.final_name].solver_vars
    lb = m.net.final_node().lower[0].tolist()
    ub = [float('inf') for _ in lb]

    print('lower bounds for all target labels:', lb)
    candidates, candidate_neuron_ids, candidate_c_rows = [], [], []
    if labels_to_verify is not None: # sort the labels
        for pidx in labels_to_verify:
            if lb[pidx] >= 0: continue # skip the label with initial bound >= 0
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

    if arguments.Config["solver"]["mip"]["parallel_solvers"] is None:
        number_cpus = multiprocessing.cpu_count()
        # calculate the max number of mip_multi_proc needed based on the number of subproblems
        if len(candidate_neuron_ids) > number_cpus:
            mip_multi_proc = number_cpus
            mip_threads = 1
        else:
            mip_multi_proc = max(1, len(candidate_neuron_ids))
            mip_threads = number_cpus // mip_multi_proc
        m.net.solver_model.setParam('Threads', mip_threads)

        print("Number of cpus:", number_cpus, " Number of subproblems:", len(candidate_neuron_ids))
        print("Reassign each subproblems with number of thread:", mip_threads)

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
        build_the_model_mip_proto_gurobi_model = m.net.solver_model
        build_the_model_mip_proto_gurobi_model.update()
        for cidx, candidate in enumerate(candidates):
            pidx = candidate_neuron_ids[cidx]
            model_filename = os.path.join(arguments.Config['bab']['cut']['cuts_path'], f"cplexmip_lay{m.net.final_node().name.replace('/', '-')}_starttime={arguments.Globals['starting_timestamp']}_idx={arguments.Globals['example_idx']}_spec={pidx}")
            model_filename_stamped = model_filename + '_' + gen_timestamp()
            model_filename_stamped_dict[pidx] = model_filename_stamped
            model_c_row_dict[pidx] = candidate_c_rows[cidx]
            mps_pool_context.append((candidate, model_filename_stamped))

        save_mps_pool = multiprocessing.pool.Pool()
        signal.signal(signal.SIGTERM, _signal_handler)
        save_mps_pool.map(_build_the_model_mip_mps_save, mps_pool_context)
        save_mps_pool.close()
        save_mps_pool.join()
        print('parallel mps save finish')

        for pidx in model_filename_stamped_dict:
            model_filename_stamped = model_filename_stamped_dict[pidx]
            model_c_row = model_c_row_dict[pidx].detach().cpu()
            try:
                # use a try-catch block to try to ignore the interrupt signal between these two lines
                # in an effort to avoid process termination *between* two following lines
                # such termination results in absence of process records and ignorance of killing the launched process ->
                # a possible cause of orphan process
                proc, logfile = run_get_cuts_subprocess(model_filename_stamped)
                processes[pidx] = {'pid': proc.pid, '_logfile': logfile, '_fname_stamped': model_filename_stamped, 'c': model_c_row}
            except Exception as e:
                try:
                    proc.kill()
                except Exception as ee: pass
                raise e

        ###### run in a sequential way now, need to change to asynchronized manner later ######
        del m.net.solver_model
        # exit()
        return None, None, None, processes

    # MULTITHREAD to solve mip for all targeted labels
    global multiprocess_mip_model, stop_multiprocess
    multiprocess_mip_model = m.net.solver_model
    global mip_solve_time_start
    mip_solve_time_start = time.time()
    input_names = None
    mip_adv = None
    if save_adv:
        input_shape = np.array(m.net.input_vars).shape
        input_names = [input_var.VarName for input_var in np.array(m.net.input_vars).reshape(-1).tolist()]

    # If pgd_order is skip, no adversarial examples are available, we skip the adv_warmup in this case.
    if adv_warmup and pgd_order != 'skip':
        # Provide PGD attack adversarial candidate to MIP solver as initial feasible solutions.
        assert pgd_attack_example is not None
        adv, min_values = pgd_attack_example
        # Reshape adv to be [output_shape, input_shape].
        adv = adv.view(adv.shape[1], -1).cpu().tolist()
        adv_list = []
        for neuron_idx in candidate_neuron_ids:
            # The MIP process can solve a minimization and/or a maximization problem.
            # Here we use the minimization (lower bound) problem only, so we only provide the feasible solution for the minimization.
            adv_list.append((adv[neuron_idx], None))
            assert min_values.flatten()[neuron_idx] >= lb[neuron_idx]
        candidates = [(name, adv, input_names, mip_skip_unsafe) for name, adv in zip(candidates, adv_list)]
    else:
        candidates = [(name, None, input_names, mip_skip_unsafe) for name in candidates]

    with multiprocessing.Pool(mip_multi_proc) as pool:
        solver_result = pool.starmap(mip_solver_lb_ub, candidates, chunksize=1)
    # solver_result = mip_solver_lb_ub(*candidates[0])

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
    try:
        # only int file descriptor can be serialized across processes
        logfile = open(log_file_path, "w")
    except Exception as e:
        print('Cannot open log file for cuts solver.')
        print(e)
        logfile = None
    return subprocess.Popen([f"{arguments.Config['bab']['cut']['cuts_path']}/get_cuts", f"{model_filename}.mps", f"{model_filename}"],
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


def copy_model(model, basis=True, use_basis_warm_start=True, remove_constr_list=[], env=None):
    """
    deep copy a gurobi model together with variable historical results
    """
    model_split = model.copy() if env==None else model.copy(env=env)

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


def solve_diving_lp(m, primal_vars, integer_vars, lower_bounds, upper_bounds):
    """
    A customized lp gurobi model to get the dual and primal values for each constraint and neurons
    """
    # only support mlp models for now
    orig_model = m.net.solver_model
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
    multiprocess_mip_model = m.net.solver_model

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
    # make sure lb is not larger than ub
    lb = torch.min(lb, ub)
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
            m.net(X + delta)
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
        m.net(X + delta)
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
    m.net(m.x)
    return X + best_delta, max_values, min_values

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
    input_domain = m.input_domain

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
    need_refine = True
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
                        # Early stop if no refine needed for this relu.
                        if lb_refined.min().item()>=0:
                            print(f"min of alpha-CROWN bounds {lb_refined.min().item()}>=0, verified!")
                            pool.terminate()
                            need_refine = False
                            break
                        last_relu_layer_refined = False

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
            if refined_num > 0:
                last_relu_layer_refined = True
            else:
                need_refine = False
                last_relu_layer_refined = False
            # If too many timeout or remaining time is sufficient, just increase the time for next layer.
            if (timeout_num > len(solver_result) * timeout_neuron_threshold or
                mip_refine_timeout - time.time() - mip_refine_time_start > remaining_timeout_threshold * mip_perneuron_refine_timeout * (len(m.net.relus[:])-relu_idx)):
                mip_perneuron_refine_timeout += mip_perneuron_refine_timeout_increasement
            # Stop refine if timeout.
            if (time.time() - mip_refine_time_start >= mip_refine_timeout) or (not need_refine):
                break
            # Update the bounds for current relu layer and its previous layer.
        for i, _ in enumerate(m.net.relus):
            reference_bounds[m.net.relus[i].inputs[0].name] = [lower_bounds[m.net.relus[i].inputs[0].name], upper_bounds[m.net.relus[i].inputs[0].name]]
        # Set up Relu constraints for the Mip solver.
        solver_vars = m.net.build_solver_module(x=(x,), final_node_name = layer.name, interm_bounds=reference_bounds, set_input=False)
        print("maximum relu layer improved by MIP so far", relu_idx)

    print(f'MIP finished with {time.time() - refine_start_time}s')

    if last_relu_layer_refined:
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
    splits = [[[], [], [], [], []] for _ in m.net.relus]
    # Creating history betas: batch, relu layers, [beta tensor for this layer]
    retb = []

    if lb_refined is None:
        if lower_bounds[-1].shape[1] != m.c.shape[1]:
            # Remove true label 0 bounds according to C matrix.
            lower_bounds[-1] = lower_bounds[-1].mm(-m.c[0].T)
            upper_bounds[-1] = upper_bounds[-1].mm(-m.c[0].T)
        return lower_bounds, upper_bounds, ([splits], [retb])

    lb_refined, ub_refined = m.get_interm_bounds(lb_refined)  # primals are better upper bounds
    ##### save refined betas to bab if not verified #####
    for mi, relu_layer in enumerate(m.net.relus):
        max_splits_per_layer = len(unstable_to_stable[mi])
        for neuron_idx, coeff in unstable_to_stable[mi]:
            splits[mi][0].append(neuron_idx)
            splits[mi][1].append(coeff)
        # Save only used beta, discard padding beta.
        if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
            val_i = []
            for key in relu_layer.sparse_betas.keys():
                # val_i.append([relu_layer.sparse_beta[key].cpu()[0, :max_splits_per_layer]])
                # We only save betas for last layer optimization for now; the rest layer betas are not saved.
                if key == m.net.final_name: val_i.append(relu_layer.sparse_betas[key].val.cpu()[0, :max_splits_per_layer])
            retb.append(val_i)
        else:
            retb.append(relu_layer.sparse_betas[0].val.cpu()[0, :max_splits_per_layer])
    return lb_refined, ub_refined, ([splits], [retb])

def check_lp_cut(self, pre_lbs, pre_ubs, lower_bounds, split, history):
    if not (arguments.Config["bab"]["cut"]["enabled"]
            and arguments.Config["bab"]["cut"]["bab_cut"]):
        return
    assert arguments.Config["bab"]["interm_transfer"], "Cut does not support no-intermediate-bound-transfer yet"
    beta_crown_lbs = [i[-1] for i in lower_bounds]
    refine_time = time.time()

def batch_verification_all_node_split_LP(net, d, ret, split, stats):
    global termination_flag_lp
    global multiprocess_lp_model
    global input_name

    # Here we directly use the settings from mip
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]

    termination_flag_lp = multiprocessing.Value('i', 0)
    multiprocess_lp_model = net.net.solver_model
    pre_relu_layer_names = net.pre_relu_layer_names
    relu_layer_names = net.relu_layer_names
    input_name = net.input_name
    orig = net.net.final_node().solver_vars
    orig_out_vars = [orig[out_idx].VarName for out_idx in range(len(orig))]
    

    if mip_multi_proc is None:
        mip_multi_proc = multiprocessing.cpu_count()
        print("preset mip_multi_proc as default setting:", mip_multi_proc)

    depths = d['depths']
    rhs = d['thresholds']
    dom_lb_all = ret['lower_bounds']
    dom_ub_all = ret['upper_bounds']
    net.check_lp_cut(d['lower_bounds'], d['upper_bounds'],
                     dom_lb_all, split, d['history'])

    dom_lb = dom_lb_all[net.final_name]
    dom_lb_all = [dom_lb_all[layer.name] for layer in net.split_nodes] + [dom_lb_all[net.final_name]]
    dom_ub_all = [dom_ub_all[layer.name] for layer in net.split_nodes] + [dom_ub_all[net.final_name]]
    # Add unverified cases to the global waiting list
    all_node_model_para_list = []
    for domain_idx in range(len(depths)):
        dlb = [dlbs[domain_idx: domain_idx + 1] for dlbs in dom_lb_all]
        dub = [dubs[domain_idx: domain_idx + 1] for dubs in dom_ub_all]
        decision_threshold = rhs.to(dom_lb[0].device)[domain_idx if domain_idx < (len(dom_lb)//2) else domain_idx - (len(dom_lb)//2)]
        if depths[domain_idx] == net.tot_ambi_nodes and torch.any(dlb[-1] <= decision_threshold):
            all_node_model_para_list.append((pre_relu_layer_names, relu_layer_names, orig_out_vars, dlb, dub, decision_threshold, domain_idx))

    # Return False if all safe, return True when there exists a counterexample
    if len(all_node_model_para_list) == 0:
        return False

    # Use map to enable multiprocessing
    with multiprocessing.Pool(mip_multi_proc) as pool:
        solver_result = pool.map(all_node_split_LP, all_node_model_para_list)

    # Save the counterexample in stats if unsafe, otherwise adjust bounds to exclude safe cases
    for (lp_status, domain_idx, dlb, counterexample) in solver_result:
        if lp_status == "unsafe":
            stats.counterexample = torch.tensor([counterexample])
            return True
        dom_lb_all[-1][domain_idx] = torch.tensor([dlb])
        dom_lb[domain_idx] = torch.tensor([dlb])
    return False
