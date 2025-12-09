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

from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass

try:
    from scip_model import SCIPModel, EarlyStopEvent, GenerateCutsEvent
except:  # pylint: disable=bare-except
    pass

from auto_LiRPA.bound_ops import BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd
from utils import get_reduce_op, get_batch_size_from_masks
import torch.nn as nn

# Global multiprocessing variables
multiprocess_mip_model = None
stop_multiprocess = False
mip_solve_time_start = 0

# Global multiprocessing variables for LP
multiprocess_lp_model = None
termination_flag_lp = None
input_name = None

# Global variables for refinement
mip_refine_time_start = 0
mip_refine_timeout = 0


class VerificationResult(Enum):
    """Verification result status."""
    SAFE_MIP = "safe-mip"
    UNSAFE_MIP = "unsafe-mip" 
    UNKNOWN_MIP = "unknown-mip"
    SAFE_LP = "safe-lp"
    UNSAFE_LP = "unsafe-lp"
    UNKNOWN_LP = "unknown-lp"
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SolverResult:
    """Result from solver execution."""
    status: VerificationResult
    global_lb: Optional[torch.Tensor] = None
    global_ub: Optional[torch.Tensor] = None
    lower_bounds: Optional[Dict[str, torch.Tensor]] = None
    upper_bounds: Optional[Dict[str, torch.Tensor]] = None
    adversarial_example: Optional[torch.Tensor] = None
    solve_time: float = 0.0
    solver_stats: Optional[Dict[str, Any]] = None
    refined_betas: Optional[Any] = None

    def to_legacy_format(self) -> Tuple[str, Dict[str, Any]]:
        """Convert to legacy (status_string, result_dict) format."""
        status_str = self.status.value
        
        result_dict = {
            'global_lb': self.global_lb,
            'lower_bounds': self.lower_bounds if self.lower_bounds is not None else {},
            'upper_bounds': self.upper_bounds if self.upper_bounds is not None else {},
            'refined_betas': self.refined_betas
        }
        
        if self.global_lb is None:
            result_dict['global_lb'] = torch.full((1, 1), float('-inf'))
            
        return status_str, result_dict


class MIPSolver:
    """MIP solver implementation."""
    
    def __init__(self, config):
        self.config = config
        self._start_time = None

    def _start_timer(self):
        self._start_time = time.time()
        
    def _get_elapsed_time(self):
        """Get elapsed time since timer was started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def build_solver_model(self, m, timeout, mip_multi_proc=None, mip_threads=1,
                          model_type="mip", x=None, intermediate_bounds=None,
                          include_C=True, final_layer_name=None, model_modifier_callback: Callable = None):
        """Build solver model for MIP/LP solving.
        
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

        # Initialize the model
        if m.net.solver_pkg == 'gurobi':
            m.net.solver_model = grb.Model()
        elif m.net.solver_pkg == 'scip':
            m.net.solver_model = SCIPModel()
        else:
            raise NotImplementedError
        # Layers must be reset, otherwise variables won't be recreated
        m.net._reset_solver_vars(m.net[final_layer_name])

        # Set model parameters
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

    def solve(self, model, ret_incomplete, specification, labels_to_verify=None,
             mip_skip_unsafe=False, pgd_attack_example=None, **kwargs) -> SolverResult:
        self._start_timer()
        
        # Determine mode
        mode = self._determine_solver_mode(specification, model)
        print(f"Unified MIP solver: detected {mode.upper()} mode")
        
        if mode == "and":
            return self._solve_and_mode(model, specification, mip_skip_unsafe, pgd_attack_example, **kwargs)
        else:  # mode == "or"
            return self._solve_or_mode(model, specification, labels_to_verify, mip_skip_unsafe, pgd_attack_example, **kwargs)

    def _determine_solver_mode(self, spec_handler, model) -> str:
        """Determine whether to use AND or OR mode based on specification structure."""
        try:
            _, _, _, or_spec_size, _, _ = spec_handler.all_specs.get(model.device if hasattr(model, 'device') else 'cpu')
            if (or_spec_size == 1).all():
                return "or"
            elif or_spec_size.shape[0] == 1:
                return "and"
            else:
                print("Complex spec detected, using OR mode")
                return "or"
        except:
            print("Could not determine mode, using OR mode")
            return "or"

    def _solve_and_mode(self, model, spec_handler, mip_skip_unsafe, pgd_attack_example, **kwargs) -> SolverResult:
        print("Solving AND specification (feasibility)")
        
        # Configuration
        timeout = arguments.Config["bab"]["timeout"]
        mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
        mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
        solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
        adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
        pgd_order = arguments.Config["attack"]["pgd_order"]
        model_type = "lp_integer" if arguments.Config["solver"]["mip"]["lp_solver"] else "mip"
        
        # Build solver model
        model.net.solver_model_type = model_type
        model.net.solver_pkg = solver_pkg
        
        if solver_pkg == 'gurobi':
            model.net.solver_model = grb.Model()
            model.net.solver_model.setParam('OutputFlag', False)
            model.net.solver_model.setParam('Threads', mip_threads)
            model.net.solver_model.setParam('FeasibilityTol', 2e-5)
            model.net.solver_model.setParam('TimeLimit', timeout)
        elif solver_pkg == 'scip':
            model.net.solver_model = SCIPModel()
            if arguments.Config["solver"]["mip"]["early_stop"]:
                model.net.solver_model.includeEventhdlr(EarlyStopEvent(), "EarlyStopEvent", "early stop handler")
            if bool(os.environ.get('ALPHA_BETA_CROWN_MIP_SHOW_CUTS', False)):
                model.net.solver_model.includeEventhdlr(GenerateCutsEvent(), "GenerateCutsEvent", "save cuts handler")
                
        # Reset and build model
        model.net._reset_solver_vars(model.net[model.net.final_name])
        model.net.build_solver_module(
            x=kwargs.get('x'),
            C=model.c,
            interm_bounds=kwargs.get('intermediate_bounds'),
            final_node_name=model.net.final_name,
            model_type=model_type,
            solver_pkg=solver_pkg
        )
        model.net.solver_model.update()
        
        # Get output variables and bounds
        out_vars = model.net[model.net.final_name].solver_vars
        lb = model.net.final_node().lower[0].tolist()
        rhs = spec_handler.vnnlib[0][1][0][1]
        
        print('lower bounds for all target labels:', lb)
        candidates_name = []
        for pidx, _ in enumerate(lb):
            if solver_pkg == 'gurobi':
                candidates_name.append(out_vars[pidx].VarName)
            else:
                candidates_name.append(out_vars[pidx].name)
        
        # Setup multiprocessing
        global multiprocess_mip_model, stop_multiprocess
        multiprocess_mip_model = model.net.solver_model
        global mip_solve_time_start
        mip_solve_time_start = time.time()
        
        # Also set global variables in utils module for mip_solver_lb_ub_and
        import lp_mip_solver.utils as utils
        utils.multiprocess_mip_model = model.net.solver_model
        utils.stop_multiprocess = False
        utils.mip_solve_time_start = mip_solve_time_start
        
        input_names = None
        save_adv = kwargs.get('save_adv', True)
        if save_adv:
            input_names = [input_var.VarName for input_var in np.array(model.net.input_vars).reshape(-1).tolist()]
        
        # Solve using AND-style solver
        from .utils import mip_solver_lb_ub_and
        solver_result = mip_solver_lb_ub_and(candidates_name, input_names, rhs)
        (vlb, vub, s, adv_new) = solver_result
        
        # Clean up global variables
        multiprocess_mip_model = None
        stop_multiprocess = False
        utils.multiprocess_mip_model = None
        utils.stop_multiprocess = False
        
        # Convert to SolverResult
        if s == 2:  # OPTIMAL - unsafe
            return SolverResult(
                status=VerificationResult.UNSAFE_MIP,
                adversarial_example=adv_new,
                global_lb=torch.tensor([[-float('inf')]]),
                solve_time=self._get_elapsed_time()
            )
        elif s == 3:  # INFEASIBLE - safe
            return SolverResult(
                status=VerificationResult.SAFE_MIP,
                global_lb=torch.tensor([[float('inf')]]),
                solve_time=self._get_elapsed_time()
            )
        else:  # timeout or other
            return SolverResult(
                status=VerificationResult.UNKNOWN_MIP,
                global_lb=torch.tensor([[-float('inf')]]),
                solve_time=self._get_elapsed_time()
            )

    def _solve_or_mode(self, model, spec_handler, labels_to_verify, mip_skip_unsafe, pgd_attack_example, **kwargs) -> SolverResult:
        """Solve OR specification (optimization problem)."""
        # Configuration
        timeout = arguments.Config["bab"]["timeout"]
        mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
        mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
        solver_pkg = arguments.Config["solver"]["mip"]["mip_solver"]
        adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
        pgd_order = arguments.Config["attack"]["pgd_order"]
        model_type = "lp_integer" if arguments.Config["solver"]["mip"]["lp_solver"] else "mip"
        
        # Build solver model
        model.net.solver_model_type = model_type
        model.net.solver_pkg = solver_pkg
        
        if solver_pkg == 'gurobi':
            model.net.solver_model = grb.Model()
            model.net.solver_model.setParam('OutputFlag', False)
            model.net.solver_model.setParam('Threads', mip_threads)
            model.net.solver_model.setParam('FeasibilityTol', 2e-5)
            model.net.solver_model.setParam('TimeLimit', timeout)
        elif solver_pkg == 'scip':
            model.net.solver_model = SCIPModel()
            if arguments.Config["solver"]["mip"]["early_stop"]:
                model.net.solver_model.includeEventhdlr(EarlyStopEvent(), "EarlyStopEvent", "early stop handler")
            if bool(os.environ.get('ALPHA_BETA_CROWN_MIP_SHOW_CUTS', False)):
                model.net.solver_model.includeEventhdlr(GenerateCutsEvent(), "GenerateCutsEvent", "save cuts handler")
                
        # Reset and build model
        model.net._reset_solver_vars(model.net[model.net.final_name])
        model.net.build_solver_module(
            x=kwargs.get('x'),
            C=model.c,
            interm_bounds=kwargs.get('intermediate_bounds'),
            final_node_name=model.net.final_name,
            model_type=model_type,
            solver_pkg=solver_pkg
        )
        model.net.solver_model.update()
        
        # Get output variables and bounds
        out_vars = model.net[model.net.final_name].solver_vars
        lb = model.net.final_node().lower[0].tolist()
        print('lower bounds for all target labels:', lb)
        
        # Prepare candidates for parallel solving
        candidates, candidate_neuron_ids, candidate_c_rows = [], [], []
        if labels_to_verify is not None:
            for pidx in labels_to_verify:
                if lb[pidx] >= 0: continue  # skip verified labels
                if solver_pkg == 'gurobi':
                    candidates.append(out_vars[pidx].VarName)
                else:
                    candidates.append(out_vars[pidx].name)
                candidate_neuron_ids.append(pidx)
                candidate_c_rows.append(model.c[:, pidx: pidx+1])
        else:
            for pidx, lbi in enumerate(lb):
                if lbi >= 0: continue
                if solver_pkg == 'gurobi':
                    candidates.append(out_vars[pidx].VarName)
                else:
                    candidates.append(out_vars[pidx].name)
                candidate_neuron_ids.append(pidx)
                candidate_c_rows.append(model.c[:, pidx: pidx+1])

        # Dynamic thread allocation
        if arguments.Config["solver"]["mip"]["parallel_solvers"] is None:
            number_cpus = multiprocessing.cpu_count()
            if len(candidate_neuron_ids) > number_cpus:
                mip_multi_proc = number_cpus
                mip_threads = 1
            else:
                mip_multi_proc = max(1, len(candidate_neuron_ids))
                mip_threads = number_cpus // mip_multi_proc
            model.net.solver_model.setParam('Threads', mip_threads)

            print("Number of cpus:", number_cpus, " Number of subproblems:", len(candidate_neuron_ids))
            print("Reassign each subproblems with number of thread:", mip_threads)
        
        print('Starting MIP solver for these labels:', candidate_neuron_ids)
        
        # Setup multiprocessing
        global multiprocess_mip_model, stop_multiprocess
        multiprocess_mip_model = model.net.solver_model
        global mip_solve_time_start
        mip_solve_time_start = time.time()
        input_names = None
        mip_adv = None
        save_adv = kwargs.get('save_adv', True)
        if save_adv:
            input_shape = np.array(model.net.input_vars).shape
            input_names = [input_var.VarName for input_var in np.array(model.net.input_vars).reshape(-1).tolist()]

        # Handle PGD warmup if available
        if adv_warmup and pgd_order != 'skip':
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

        # Parallel solve using multiprocessing
        from .utils import mip_solver_lb_ub
        
        # For multiprocessing, we need to pass the model through a different approach
        # Since Gurobi models can't be pickled, we'll use a workaround
        # Force single process mode to avoid global variable issues
        if mip_multi_proc == 1 or True:  # Always use single process for now
            # Single process - no multiprocessing needed
            # Set global variables directly in utils module
            import lp_mip_solver.utils as utils
            utils.multiprocess_mip_model = model.net.solver_model
            utils.stop_multiprocess = False
            utils.mip_solve_time_start = mip_solve_time_start
            
            solver_result = []
            for candidate in candidates:
                result = mip_solver_lb_ub(*candidate)
                solver_result.append(result)
            
            # Clean up global variables
            utils.multiprocess_mip_model = None
            utils.stop_multiprocess = False
        else:
            # Multiple processes - use global variable approach
            # Set global variable before creating pool (same as original)
            # Import and set global variables in utils module
            import lp_mip_solver.utils as utils
            
            # Set global variables in utils module
            utils.multiprocess_mip_model = model.net.solver_model
            utils.stop_multiprocess = False
            utils.mip_solve_time_start = mip_solve_time_start
            
            # Force fork mode for multiprocessing to ensure global variables are accessible
            import multiprocessing as mp
            if hasattr(mp, 'set_start_method'):
                try:
                    mp.set_start_method('fork', force=True)
                except RuntimeError:
                    pass  # Already set
            
            with multiprocessing.Pool(mip_multi_proc) as pool:
                solver_result = pool.starmap(mip_solver_lb_ub, candidates, chunksize=1)
            
            # Clean up global variables
            utils.multiprocess_mip_model = None
            utils.stop_multiprocess = False

        multiprocess_mip_model = None
        stop_multiprocess = False

        # Process results
        ub = [float('inf') for _ in lb]
        status = [-1 for i in lb]
        for (vlb, vub, s, adv_new), pidx in zip(solver_result, candidate_neuron_ids):
            lb[pidx] = vlb
            ub[pidx] = vub
            status[pidx] = s
            if adv_new is not None: 
                mip_adv = adv_new
        
        lb, ub = torch.tensor(lb), torch.tensor(ub)
        
        # Convert to SolverResult
        if mip_adv is not None:
            return SolverResult(
                status=VerificationResult.UNSAFE_MIP,
                adversarial_example=mip_adv,
                global_lb=lb.unsqueeze(-1),
                global_ub=ub.unsqueeze(-1),
                solve_time=self._get_elapsed_time(),
                solver_stats={'status_list': status}
            )
        elif (lb >= 0).all():
            return SolverResult(
                status=VerificationResult.SAFE_MIP,
                global_lb=lb.unsqueeze(-1),
                global_ub=ub.unsqueeze(-1),
                solve_time=self._get_elapsed_time(),
                solver_stats={'status_list': status}
            )
        else:
            return SolverResult(
                status=VerificationResult.UNKNOWN_MIP,
                global_lb=lb.unsqueeze(-1),
                global_ub=ub.unsqueeze(-1),
                solve_time=self._get_elapsed_time(),
                solver_stats={'status_list': status}
            )

    def cleanup(self):
        """Clean up MIP solver resources."""
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# Global variables for multiprocessing
multiprocess_mip_model = None
stop_multiprocess = False
mip_solve_time_start = None
