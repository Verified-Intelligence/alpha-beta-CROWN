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
import random
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


def lp_solver(candidate):
    """Multiprocess worker that refines per-neuron LP bounds."""
    from . import utils as solver_utils
    global multiprocess_lp_model

    model = multiprocess_lp_model.copy()
    v = model.getVarByName(candidate)
    out_lb, out_ub = v.LB, v.UB
    refined = False
    refine_time = time.time()

    # Early stop if neuron already stable or another worker finished.
    if out_lb >= 0 or out_ub <= 0:
        return out_lb, out_ub, time.time() - refine_time, refined
    if solver_utils.stop_multiprocess:
        return out_lb, out_ub, time.time() - refine_time, refined

    # Minimize to tighten lower bound.
    model.setObjective(v, grb.GRB.MINIMIZE)
    model.update()
    model.reset()
    try:
        model.optimize()
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)
    if model.status == grb.GRB.OPTIMAL:
        vlb = v.X
        refined = True
    else:
        print("Warning: LP solve did not finish optimally (lower bound phase).")
        vlb = out_lb

    # Maximize to tighten upper bound.
    model.setObjective(v, grb.GRB.MAXIMIZE)
    model.update()
    model.reset()
    try:
        model.optimize()
    except grb.GurobiError as e:
        handle_gurobi_error(e.message)
    if model.status == grb.GRB.OPTIMAL:
        vub = v.X
        refined = True
    else:
        print("Warning: LP solve did not finish optimally (upper bound phase).")
        vub = out_ub

    duration = time.time() - refine_time
    print_str = (
        f"Linear {v.VarName}: old_lb={out_lb:.7g}, new_lb={vlb:.7g}, "
        f"old_ub={out_ub:.7g}, new_ub={vub:.7g}, lb_diff={vlb - out_lb:.7g}, "
        f"ub_diff={out_ub - vub:.7g}, time={duration:3g}"
    )
    print(print_str)
    sys.stdout.flush()
    return vlb, vub, print_str, refined


def build_the_model_lp(
    m,
    using_integer=True,
    get_primals=False,
    optimized_layer_name=None,
    final_layer_name=None,
    compute_upper_bound=False,
    include_output_constraint=False,
    rhs=None,
):
    """Construct and optionally solve the LP relaxation for the network."""

    if optimized_layer_name is None:
        optimized_layer_name = m.net.final_name
    if final_layer_name is None:
        final_layer_name = m.net.final_name
    if include_output_constraint:
        assert rhs is not None

    timeout = arguments.Config["bab"]["timeout"]
    model_type = "lp_integer" if using_integer else "lp"
    m.layers = list(m.model_ori.children())

    def add_output_constraint(model):
        final_layer_vars = m.net.final_node().solver_vars
        assert len(final_layer_vars) == 1, len(final_layer_vars)
        final_layer_var = final_layer_vars[0]
        assert rhs.shape == (1, 1), rhs
        model.addConstr(final_layer_var <= rhs, name='output_constraint')

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
        model_sense = grb.GRB.MAXIMIZE if compute_upper_bound else grb.GRB.MINIMIZE
        m.net.solver_model.setObjective(obj, model_sense)
        try:
            m.net.solver_model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)

        status = m.net.solver_model.status
        assert status in (grb.GRB.OPTIMAL, grb.GRB.INFEASIBLE), f"Unexpected LP status: {status}"
        if status == grb.GRB.INFEASIBLE:
            glb = float("inf") if compute_upper_bound else -float("inf")
        else:
            glb = obj.X
        glbs.append(glb)

    if get_primals:
        # Extract primal values layer by layer.
        primal_vars = []
        layers = [m.net.final_node()]
        node = m.net.final_node()
        while node.inputs:
            layers = [node.inputs[0]] + layers
            node = node.inputs[0]

        for layer in layers:
            vars_ = layer.solver_vars
            pv = []
            if not isinstance(vars_[0], list):
                pv.extend(var.X for var in vars_)
            else:
                for chan in range(len(vars_)):
                    for row in range(len(vars_[chan])):
                        for col in range(len(vars_[chan][row])):
                            pv.append(vars_[chan][row][col].X)
            primal_vars.append(pv)

        if using_integer:
            integer_vars = []
            for relu_layer in m.net.relus:
                integer_vars.append([relu_integer.X for relu_integer in relu_layer.integer_vars])

        input_primal_gurobi = primal_vars[0]
        print("### Extracted primal values from Gurobi LP model ###")
        _ = input_primal_gurobi  # Preserve legacy side effect
        # The original implementation kept these variables for potential debugging.
        # m.solve_diving_lp(primal_vars, integer_vars, lower_bounds, upper_bounds)

    return glbs


def solve_diving_lp(m, primal_vars, integer_vars, lower_bounds, upper_bounds):
    """Rebuild diving LP used for extracting dual information (legacy helper)."""

    orig_model = m.net.solver_model
    diving_model = orig_model.copy()
    diving_model.reset()
    relu_idx = 0

    all_nus = []
    all_nu_hats = []
    for i, layer in enumerate(m.layers):
        if isinstance(layer, nn.Linear):
            w = layer.weight.to("cpu")
            nu = torch.zeros(w.size(0))
            size = w.size(0)
            if layer == m.layers[-1]:
                size = lower_bounds[-1].shape[1]
            for neuron_idx in range(size):
                nu[neuron_idx] = orig_model.getConstrByName(f'lay{i+1}_{neuron_idx}_eq').pi
            nu_hat = nu.unsqueeze(0).matmul(w).squeeze(0)
            all_nus.append(nu)
            all_nu_hats.append(nu_hat)

    for i, layer in enumerate(m.layers):
        if isinstance(layer, nn.Linear):
            continue
        if isinstance(layer, nn.ReLU):
            xs = primal_vars[i - 1]
            hat_xs = primal_vars[i]
            pre_lbs = lower_bounds[relu_idx].squeeze(0)
            pre_ubs = upper_bounds[relu_idx].squeeze(0)
            nu_hats = all_nu_hats[relu_idx + 1]
            pos_nu_hats = torch.clamp_min(nu_hats, 0)
            neg_nu_hats = torch.clamp_max(nu_hats, 0)
            unstable_idx = 0
            for neuron_idx in range(len(xs)):
                lb = pre_lbs[neuron_idx].item()
                ub = pre_ubs[neuron_idx].item()
                if lb < 0 and ub > 0:
                    x = xs[neuron_idx]
                    hat_x = hat_xs[neuron_idx]
                    z = integer_vars[relu_idx][unstable_idx]
                    pi = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_0').pi
                    gamma = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_1').pi
                    tau = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_2').pi
                    mu = orig_model.getConstrByName(f'ReLU{relu_idx}_{neuron_idx}_a_3').pi
                    nu_hat = nu_hats[neuron_idx].item()
                    pos = pos_nu_hats[neuron_idx].item()
                    neg = neg_nu_hats[neuron_idx].item()
                    if nu_hat < -1e-6:
                        alpha = gamma / (gamma + mu)
                    else:
                        alpha = float("nan")

                    new_tau = - lb / (ub - lb) * pos
                    new_pi = ub / (ub - lb) * pos

                    upper_z = 1.0
                    lower_z = 0.0
                    neuron_set_name = ""
                    if nu_hat < -1e-6:
                        neuron_set_name = f'z_ReLU{relu_idx}_{neuron_idx}'
                        z_var = diving_model.getVarByName(neuron_set_name)
                        if x > 0:
                            lower_z = alpha / ub * x
                            if z_var is not None:
                                z_var.LB = 1.0
                            neuron_set_name += " set to 1"
                        elif x < 0:
                            upper_z = (alpha - 1) / lb * x + 1
                            if z_var is not None:
                                z_var.UB = 0.0
                            neuron_set_name += " set to 0"
                        elif mu > 0 and gamma > 0:
                            assert abs(x) < 1e-6
                            if random.random() > 0.5:
                                if z_var is not None:
                                    z_var.LB = 1.0
                                neuron_set_name += " set to 1 (random)"
                            else:
                                if z_var is not None:
                                    z_var.UB = 0.0
                                neuron_set_name += " set to 0 (random)"
                        else:
                            if z_var is not None:
                                z_var.LB = lower_z
                                z_var.UB = upper_z
                    print(
                        f'layer {i:2d} neuron {neuron_idx:3d} l={lb:8.5f} ub={ub:8.5f} '
                        f'x={x:8.5f} hat_x={hat_x:8.5f} pi={pi:8.5f} ({new_pi:8.5f}), '
                        f'tau={tau:8.5f} ({new_tau:8.5f}), mu={mu:8.5f}, gamma={gamma:8.5f}, '
                        f'nu_hat={nu_hat:8.5f}, alpha={alpha:8.5f} z={z:8.5f} '
                        f'upper_z={upper_z:8.5f} lower_z={lower_z:8.5f} {neuron_set_name}'
                    )
                    unstable_idx += 1
            relu_idx += 1

    diving_model.update()
    diving_model.optimize()
    print(f'original obj: {orig_model.objval}, new_obj: {diving_model.objval}, status: {diving_model.status}')
    return


def update_model_bounds(solver_model, lower_bounds, upper_bounds,
                        pre_relu_layer_names, relu_layer_names, model_type="lp_integer"):
    """Update solver vars bounds with given lower and upper bounds.
    Args:
        solver_model: copied solver model from m.net.solver_model
        lower_bounds, upper_bounds: tightened bounds
        pre_relu_layer_names, relu_layer_names: the names of pre relu layers and relu layers, to get the vars
        model_type: model type string, one of ["mip", "lp_integer", "lp"]
    """

    # update pre relu neuron bounds as well as relu constraints
    for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
        this_layer_shape = lower_bounds[relu_idx].shape
        lbs = lower_bounds[relu_idx].reshape(-1)
        ubs = upper_bounds[relu_idx].reshape(-1)
        for neuron_idx in range(lbs.shape[0]):
            pre_var = solver_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
            pre_lb = lbs[neuron_idx]
            pre_ub = ubs[neuron_idx]
            pre_var.lb = pre_lb
            pre_var.ub = pre_ub
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
                        # Triangle relaxation parameters
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        if grb is not None:
                            solver_model.addConstr(var <= slope * pre_var + bias)
                        else:
                            # Fallback: try to add linear constraint via generic interface
                            solver_model.addConstr(var <= slope * pre_var + bias)

    solver_model.update()
    return solver_model


def all_node_split_LP(arg):  
    pre_relu_layer_names, relu_layer_names, orig_out_vars, lower_bounds, upper_bounds, rhs, dix = arg
    global input_name
    with grb.Env(empty=True) as env:
        global termination_flag_lp
        global multiprocess_lp_model
        env.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
        env.start()
        if termination_flag_lp.value == 1:
            # Stop if a counterexample is already found
            return 'unknown', dix, float('inf'), None  
        
        all_node_model = copy_model(multiprocess_lp_model, basis=False, env=env)
        all_node_model = update_model_bounds(all_node_model, lower_bounds, upper_bounds,
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
    dom_lb = ret['lower_bounds'][net.final_name]
    dom_ub = ret['upper_bounds'][net.final_name]
    # ret only contains the last layer bounds, we need to also collect all the intermediate bounds
    dom_lb_all, dom_ub_all, _ = net.get_interm_bounds(lb=dom_lb, ub=dom_ub, init=True)
    check_lp_cut(d['lower_bounds'], d['upper_bounds'], dom_lb_all, split, d['history'])
    dom_lb_all = [dom_lb_all[layer.name].to('cpu') for layer in net.split_nodes] + [dom_lb_all[net.final_name]]
    dom_ub_all = [dom_ub_all[layer.name].to('cpu') for layer in net.split_nodes] + [dom_ub_all[net.final_name]]
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


def check_lp_cut(pre_lbs, pre_ubs, lower_bounds, split, history):
    if not (arguments.Config["bab"]["cut"]["enabled"]
            and arguments.Config["bab"]["cut"]["bab_cut"]):
        return
    assert arguments.Config["bab"]["interm_transfer"], "Cut does not support no-intermediate-bound-transfer yet"
    beta_crown_lbs = [i[-1] for i in lower_bounds]
    refine_time = time.time()


def update_the_model_cut(m, cut, pre_lbs=None, pre_ubs=None, split=None, verbose=False):
    """Recalculate the bound propagation using lp solver with cut constraints and split constraints."""
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
                lower_bounds[node[0]].view(-1)[node[1]] = 0.
            else:
                split_var = m.model_cut.getVarByName(f"lay{pre_relu_layer_names[node[0]]}_{node[1]}")
                gurobi_splits.append(m.model_cut.addConstr(split_var <= 0, name=f"split{split_idx}"))
                print(f"split_expr:{split_var}<=0")
                upper_bounds[node[0]].view(-1)[node[1]] = 0.
        m.model_cut.update()

    if pre_lbs is not None:
        m.model_cut = update_model_bounds(m.model_cut, lower_bounds, upper_bounds,
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
    m.model_cut.optimize()

    if m.model_cut.status == 2:
        glb = objVar.X
    elif m.model_cut.status == 3:
        print("warning, gurobi infeasible!")
        glb = float('inf')
    else:
        print("model status not supported!")
        exit()
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
    return glb


# Global variables for LP multiprocessing
multiprocess_lp_model = None
termination_flag_lp = None
input_name = None


# Import dependencies
from .utils import copy_model, handle_gurobi_error
