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

import torch
import arguments
import gurobipy as grb
import os
import time
import shutil
import multiprocessing
import numpy as np
import subprocess
import signal
from attack import check_and_save_cex

# Import core functionality from modular files
from .mip_core import MIPSolver, VerificationResult, SolverResult
from .utils import (
    mip_solver_lb_ub, mip_solver_lb_ub_and, update_mip_model_fix_relu,
    mip_solver_attack, copy_model, handle_gurobi_error,
    clamp, compute_ratio, FSB_score, NoDaemonProcess, NestablePool
)
from .bounds_core import (
    update_model_bounds, all_node_split_LP, batch_verification_all_node_split_LP,
    check_lp_cut, update_the_model_cut, build_the_model_lp, lp_solver, solve_diving_lp
)
from .refine_core import build_the_model_mip_refine, _intermediate_PGD_attack
from auto_LiRPA.utils import stop_criterion_min
# Constants for backward compatibility
CPLEX_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuts/CPLEX_cuts")

# Global factory instance for backward compatibility
class DefaultSolverFactory:
    """Default solver factory for backward compatibility."""
    
    def __init__(self):
        pass

    def create_mip_solver(self, config=None):
        return MIPSolver(config)

    def create_model_builder(self, config=None):
        return MIPSolver(config)

    # Backward-compatible wrappers
    def build_solver_model(self, *args, **kwargs):
        solver = MIPSolver(None)
        return solver.build_solver_model(*args, **kwargs)

    def build_mip_or(self, *args, **kwargs):
        return build_the_model_mip_or(*args, **kwargs)

    def build_mip_and(self, *args, **kwargs):
        return build_the_model_mip_and(*args, **kwargs)

    def build_mip_refine(self, *args, **kwargs):
        return build_the_model_mip_refine(*args, **kwargs)

    def update_model_bounds(self, *args, **kwargs):
        return update_model_bounds(*args, **kwargs)

    def check_lp_cut(self, *args, **kwargs):
        return check_lp_cut(*args, **kwargs)

    def update_mip_model_fix_relu(self, *args, **kwargs):
        return update_mip_model_fix_relu(*args, **kwargs)

    def update_the_model_cut(self, *args, **kwargs):
        return update_the_model_cut(*args, **kwargs)

    def all_node_split_LP(self, *args, **kwargs):
        return all_node_split_LP(*args, **kwargs)

    def build_lp(self, *args, **kwargs):
        return build_the_model_lp(*args, **kwargs)

    def lp_solver(self, *args, **kwargs):
        return lp_solver(*args, **kwargs)

    def solve_diving_lp(self, *args, **kwargs):
        return solve_diving_lp(*args, **kwargs)

# Global factory instance
_default_factory = DefaultSolverFactory()

def create_factory(solver_type: str = "default"):
    """Create a solver factory of the specified type (backward compatibility)."""
    return DefaultSolverFactory()

def get_mip_solver(config=None):
    """Get MIP solver instance (backward compatibility)."""
    return _default_factory.create_mip_solver(config)

def get_model_builder(config=None):
    """Get model builder instance (backward compatibility)."""
    return _default_factory.create_model_builder(config)

def mip(model, ret_incomplete, vnnlib_handler, labels_to_verify=None,
        mip_skip_unsafe=False, pgd_attack_example=None, verifier=None):
    ret = {key: None for key in [
        'global_lb', 'lower_bounds', 'upper_bounds', 'refined_betas']}

    # TODO: Use data (x, c, rhs) from vnnlib_handler instead of model.
    _, _, _, or_spec_size, _, _ = vnnlib_handler.all_specs.get(model.device)
    vnnlib = vnnlib_handler.vnnlib

    if (or_spec_size == 1).all():
        # All OR specs have only one AND.
        mode_used = "or"
    elif or_spec_size.shape[0] == 1:
        # Only one OR spec.
        mode_used = "and"

    if verifier == "mip":
        # Use different MIP solvers based on specification type
        if mode_used == "and":
            # AND-style specifications (feasibility problem)
            mip_global_lb, mip_global_ub, mip_status, adv_example = model.build_the_model_mip_and(
                save_adv=arguments.Config["general"]["save_adv_example"],
                mip_skip_unsafe=mip_skip_unsafe, pgd_attack_example=pgd_attack_example,
                vnnlib=vnnlib)
            verified_status = "safe-mip"
            
            # Set ret['global_lb'] with appropriate shape for compatibility
            ret['global_lb'] = torch.full((1, 1), -float('inf'), dtype=torch.float32)
            
            if mip_status == grb.GRB.OPTIMAL:
                print("Verified unsafe-mip with AND-style specification!")
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
            elif mip_status == grb.GRB.INFEASIBLE:
                 verified_status = "safe-mip"
            elif mip_status == grb.GRB.TIME_LIMIT:
                verified_status = "unknown-mip"
            print(f"verified {verified_status} with AND-style specification!")
            return verified_status, ret
        else:  # mode_used == "or"
            # OR-style specifications (optimization problem)
            mip_global_lb, mip_global_ub, mip_status, adv_example= model.build_the_model_mip_or(
                labels_to_verify=labels_to_verify, save_adv=arguments.Config["general"]["save_adv_example"],
                mip_skip_unsafe=mip_skip_unsafe, pgd_attack_example=pgd_attack_example)

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
                        print("Verified unsafe-mip with OR-style specification!")
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
            print(f"verified {verified_status} with OR-style specification!")
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
        lower_bounds, upper_bounds = refined_lower_bounds, refined_upper_bounds
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

@torch.no_grad()
def construct_mip_with_model(model_ori, x, input_shape, c, intermediate_bounds, 
                            save_mps=False, process_dict=None):
    """Construct a mip problem using just the model, input x (BoundedTensor), intermediate layer bounds and other parameters."""
    torch.set_num_threads(1)
    # Delay import to avoid circular import
    import beta_CROWN_solver
    model = beta_CROWN_solver.LiRPANet(model_ori, in_size=input_shape, c=c, device='cpu')
    return build_the_model_mip_or(
        model,
        labels_to_verify=None,
        save_mps=save_mps,
        process_dict=process_dict,
        save_adv=False,
        x=x,
        intermediate_bounds=intermediate_bounds,
        mip_skip_unsafe=False,
        pgd_attack_example=None,
    )

# Import the main building functions
from .bounds_core import (
    update_model_bounds,
    all_node_split_LP,
    batch_verification_all_node_split_LP,
    check_lp_cut,
    update_the_model_cut
)

from .refine_core import (
    build_the_model_mip_refine,
    mip_solver,
    _intermediate_PGD_attack
)

from .utils import (
    mip_solver_lb_ub,
    mip_solver_lb_ub_and,
    mip_solver_attack,
    mip_solver_attack_init,
    mip_solver_worker,
    copy_model,
    handle_gurobi_error,
    clamp,
    compute_ratio,
    FSB_score,
    NoDaemonProcess,
    NestablePool,
    check_optimization_success,
    update_mip_model_fix_relu
)

from .mip_core import MIPSolver, VerificationResult, SolverResult

# Globals retained for legacy interfaces
multiprocess_mip_model = None
stop_multiprocess = False
mip_solve_time_start = None
build_the_model_mip_proto_gurobi_model = None
save_mps_pool = None

def build_the_model_mip_or(m, labels_to_verify=None, save_mps=False, process_dict=None,
                          save_adv=False, x=None, intermediate_bounds=None,
                          mip_skip_unsafe=False, pgd_attack_example=None):
    """Legacy OR-style MIP construction (copied from original implementation)."""

    global multiprocess_mip_model, stop_multiprocess, mip_solve_time_start
    global build_the_model_mip_proto_gurobi_model, save_mps_pool

    if (arguments.Config['bab']['cut']['cplex_cuts']
            and arguments.Config['bab']['cut']['cuts_path'] != CPLEX_FOLDER):
        os.makedirs(arguments.Config['bab']['cut']['cuts_path'], exist_ok=True)
        shutil.copy(os.path.join(CPLEX_FOLDER, "get_cuts"),
                    os.path.join(arguments.Config['bab']['cut']['cuts_path'], "get_cuts"))

    def gen_timestamp():
        return str(int(time.time() * 100.0) % 100000000)

    timeout = arguments.Config["bab"]["timeout"]
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
    pgd_order = arguments.Config["attack"]["pgd_order"]

    model_type = arguments.Config["solver"]["mip"]["formulation"]
    build_solver_model(m, timeout, mip_multi_proc=mip_multi_proc,
                       mip_threads=mip_threads, model_type=model_type,
                       x=x, intermediate_bounds=intermediate_bounds)

    out_vars = m.net[m.net.final_name].solver_vars
    assert m.net.final_node().lower.shape[0] == 1
    lb = m.net.final_node().lower[0].tolist()
    ub = [float('inf') for _ in lb]

    print('lower bounds for all target labels:', lb)
    candidates, candidate_neuron_ids, candidate_c_rows = [], [], []
    if labels_to_verify is not None:
        for pidx in labels_to_verify:
            if lb[pidx] >= 0:
                continue
            if solver_pkg == 'gurobi':
                candidates.append(out_vars[pidx].VarName)
            else:
                candidates.append(out_vars[pidx].name)
            candidate_neuron_ids.append(pidx)
            candidate_c_rows.append(m.c[:, pidx: pidx+1])
    else:
        for pidx, lbi in enumerate(lb):
            if lbi >= 0:
                continue
            if solver_pkg == 'gurobi':
                candidates.append(out_vars[pidx].VarName)
            else:
                candidates.append(out_vars[pidx].name)
            candidate_neuron_ids.append(pidx)
            candidate_c_rows.append(m.c[:, pidx: pidx+1])

    if arguments.Config["solver"]["mip"]["parallel_solvers"] is None:
        number_cpus = multiprocessing.cpu_count()
        if len(candidate_neuron_ids) > number_cpus:
            mip_multi_proc = number_cpus
            mip_threads = 1
        else:
            mip_multi_proc = max(1, len(candidate_neuron_ids))
            mip_threads = number_cpus // mip_multi_proc
        m.net.solver_model.setParam('Threads', mip_threads)
        print("Number of cpus:", number_cpus, " Number of subproblems:", len(candidate_neuron_ids))
        print("Reassign each subproblems with number of thread:", mip_threads)

    print('Starting MIP solver for these labels:', candidate_neuron_ids)

    if save_mps:
        global build_the_model_mip_proto_gurobi_model, save_mps_pool
        print("start creating model mps for candidates:", candidates)
        mps_pool_context = []
        model_filename_stamped_dict = {}
        model_c_row_dict = {}

        build_the_model_mip_proto_gurobi_model = m.net.solver_model
        build_the_model_mip_proto_gurobi_model.update()
        for cidx, candidate in enumerate(candidates):
            pidx = candidate_neuron_ids[cidx]
            model_filename = os.path.join(
                arguments.Config['bab']['cut']['cuts_path'],
                f"cplexmip_lay{m.net.final_node().name.replace('/', '-')}_starttime={arguments.Globals['starting_timestamp']}_idx={arguments.Globals['example_idx']}_spec={pidx}"
            )
            model_filename_stamped = model_filename + '_' + gen_timestamp()
            model_filename_stamped_dict[pidx] = model_filename_stamped
            model_c_row_dict[pidx] = candidate_c_rows[cidx]
            mps_pool_context.append((candidate, model_filename_stamped))

        save_mps_pool = multiprocessing.pool.Pool()
        signal.signal(signal.SIGTERM, _signal_handler)
        save_mps_pool.map(_build_the_model_mip_mps_save, mps_pool_context)
        save_mps_pool.close()
        save_mps_pool.join()
        save_mps_pool = None
        print('parallel mps save finish')

        processes = {} if process_dict is None else process_dict
        for pidx in model_filename_stamped_dict:
            model_filename_stamped = model_filename_stamped_dict[pidx]
            model_c_row = model_c_row_dict[pidx].detach().cpu()
            try:
                proc, logfile = run_get_cuts_subprocess(model_filename_stamped)
                processes[pidx] = {'pid': proc.pid, '_logfile': logfile,
                                   '_fname_stamped': model_filename_stamped, 'c': model_c_row}
            except Exception as exc:
                try:
                    proc.kill()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                raise exc

        del m.net.solver_model
        return None, None, None, processes

    multiprocess_mip_model = m.net.solver_model
    stop_multiprocess = False
    mip_solve_time_start = time.time()
    import lp_mip_solver.utils as utils
    utils.multiprocess_mip_model = multiprocess_mip_model
    utils.stop_multiprocess = False
    utils.mip_solve_time_start = mip_solve_time_start

    mip_adv = None
    input_names = None
    if save_adv:
        input_shape = np.array(m.net.input_vars).shape
        input_names = [input_var.VarName for input_var in np.array(m.net.input_vars).reshape(-1).tolist()]

    if adv_warmup and pgd_order in ['before', 'after']:
        assert pgd_attack_example is not None
        adv, min_values = pgd_attack_example
        assert adv.shape[0] == len(lb)
        assert min_values.ndim == 1 and min_values.shape[0] == len(lb)
        adv = adv.view(adv.shape[0], -1).cpu().tolist()
        adv_list = []
        for neuron_idx in candidate_neuron_ids:
            adv_list.append((adv[neuron_idx], None))
            assert min_values[neuron_idx] >= lb[neuron_idx]
        candidates = [(name, adv, input_names, mip_skip_unsafe) for name, adv in zip(candidates, adv_list)]
    else:
        candidates = [(name, None, input_names, mip_skip_unsafe) for name in candidates]

    with multiprocessing.Pool(mip_multi_proc) as pool:
        solver_result = pool.starmap(mip_solver_lb_ub, candidates, chunksize=1)

    multiprocess_mip_model = None
    stop_multiprocess = False
    utils.multiprocess_mip_model = None
    utils.stop_multiprocess = False

    status = [-1 for _ in lb]
    for (vlb, vub, s, adv_new), pidx in zip(solver_result, candidate_neuron_ids):
        lb[pidx] = vlb
        ub[pidx] = vub
        status[pidx] = s
        if adv_new is not None:
            mip_adv = adv_new

    lb, ub = torch.tensor(lb), torch.tensor(ub)
    return lb, ub, status, mip_adv


def _build_the_model_mip_mps_save(args):
    """Helper for parallel MPS dumping (legacy behaviour)."""
    candidate, fname = args
    print(f"parallel save mip model to {fname}.mps")
    global build_the_model_mip_proto_gurobi_model
    now_model = build_the_model_mip_proto_gurobi_model.copy()
    v = now_model.getVarByName(candidate)
    now_model.setObjective(v, grb.GRB.MINIMIZE)
    now_model.update()
    now_model.write(f'{fname}.mps')


def _signal_handler(signum, frame):
    """Ensure parallel MPS pool is closed on termination."""
    global save_mps_pool
    print(f"Process {os.getpid()} received signal {signum}. Terminate pool {save_mps_pool}")
    if save_mps_pool is not None:
        save_mps_pool.terminate()
    exit(0)


def run_get_cuts_subprocess(model_filename):
    """Launch external cuts solver (legacy helper)."""
    cut_file_path = f"{model_filename}.cuts"
    idx_file_path = f"{model_filename}.indx"
    log_file_path = f"{model_filename}.log"
    if os.path.exists(cut_file_path):
        os.remove(cut_file_path)
    if os.path.exists(idx_file_path):
        os.remove(idx_file_path)
    try:
        logfile = open(log_file_path, "w")
    except Exception as exc:  # pylint: disable=broad-except
        print('Cannot open log file for cuts solver.')
        print(exc)
        logfile = None
    proc = subprocess.Popen([
        f"{arguments.Config['bab']['cut']['cuts_path']}/get_cuts",
        f"{model_filename}.mps",
        f"{model_filename}"
    ], stderr=subprocess.STDOUT, stdout=logfile)
    logfile_fd = logfile.fileno() if logfile is not None else None
    return proc, logfile_fd


def build_the_model_mip_and(m, save_adv=False, x=None, intermediate_bounds=None,
                           mip_skip_unsafe=False, pgd_attack_example=None, vnnlib=None):
    """Legacy AND-style MIP construction."""

    global multiprocess_mip_model, stop_multiprocess, mip_solve_time_start

    timeout = arguments.Config["bab"]["timeout"]
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
    solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
    adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
    pgd_order = arguments.Config["attack"]["pgd_order"]

    model_type = arguments.Config["solver"]["mip"]["formulation"]
    build_solver_model(m, timeout, mip_multi_proc=mip_multi_proc,
                       mip_threads=mip_threads, model_type=model_type,
                       x=x, intermediate_bounds=intermediate_bounds)

    out_vars = m.net[m.net.final_name].solver_vars
    lb = m.net.final_node().lower[0].tolist()
    rhs = vnnlib[0][1][0][1]

    print('lower bounds for all target labels:', lb)
    candidates_name = []
    for pidx, _ in enumerate(lb):
        if solver_pkg == 'gurobi':
            candidates_name.append(out_vars[pidx].VarName)
        else:
            candidates_name.append(out_vars[pidx].name)

    multiprocess_mip_model = m.net.solver_model
    stop_multiprocess = False
    mip_solve_time_start = time.time()
    import lp_mip_solver.utils as utils
    utils.multiprocess_mip_model = multiprocess_mip_model
    utils.stop_multiprocess = False
    utils.mip_solve_time_start = mip_solve_time_start

    input_names = None
    if save_adv:
        input_names = [input_var.VarName for input_var in np.array(m.net.input_vars).reshape(-1).tolist()]

    if adv_warmup and pgd_order != 'skip':
        assert pgd_attack_example is not None
        adv, min_values = pgd_attack_example
        assert adv.shape[0] == 1
        assert min_values.ndim == 1 and min_values.shape[0] == 1
        adv = adv.view(adv.shape[0], -1).cpu().tolist()
        candidates = [(candidates_name, input_names, rhs)]
    else:
        candidates = [(candidates_name, input_names, rhs)]

    vlb, vub, status, adv_new = mip_solver_lb_ub_and(*candidates[0])
    multiprocess_mip_model = None
    stop_multiprocess = False
    utils.multiprocess_mip_model = None
    utils.stop_multiprocess = False

    return None, None, status, adv_new

def build_solver_model(model, timeout, mip_multi_proc=None, mip_threads=1,
                      model_type="mip", x=None, intermediate_bounds=None,
                      include_C=True, final_layer_name=None,
                      model_modifier_callback=None):
    """Build solver model (backward compatibility)."""
    solver = MIPSolver(None)
    return solver.build_solver_model(
        model, timeout, mip_multi_proc=mip_multi_proc, mip_threads=mip_threads,
        model_type=model_type, x=x, intermediate_bounds=intermediate_bounds,
        include_C=include_C, final_layer_name=final_layer_name,
        model_modifier_callback=model_modifier_callback
    )

# Export main functions and classes
__all__ = [
    'mip',
    'build_solver_model',
    'build_the_model_lp',
    'construct_mip_with_model',
    'CPLEX_FOLDER',
    'create_factory',
    'get_mip_solver',
    'get_model_builder',
    'MIPSolver',
    'VerificationResult',
    'SolverResult',
    'build_the_model_mip_or',
    'build_the_model_mip_and',
    'build_the_model_mip_refine',
    'update_model_bounds',
    'check_lp_cut',
    'update_mip_model_fix_relu',
    'update_the_model_cut',
    'all_node_split_LP',
    'batch_verification_all_node_split_LP',
    'mip_solver',
    'mip_solver_lb_ub',
    'mip_solver_lb_ub_and',
    'mip_solver_attack',
    'mip_solver_attack_init',
    'mip_solver_worker',
    'lp_solver',
    'solve_diving_lp',
    '_intermediate_PGD_attack',
    'copy_model',
    'handle_gurobi_error',
    'clamp',
    'compute_ratio',
    'FSB_score',
    'NoDaemonProcess',
    'NestablePool',
    '_build_the_model_mip_mps_save',
    '_signal_handler',
    'check_optimization_success'
] 
