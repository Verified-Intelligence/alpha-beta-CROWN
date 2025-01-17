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

import multiprocessing
import os
import time
import copy
from collections import defaultdict

import torch

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from lp_mip_solver import CPLEX_FOLDER, construct_mip_with_model
from cuts.cut_utils import generate_cplex_cuts
from cuts.cutter import Cutter
from cuts.implication_graph import build_bound_implication_graph

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def get_impl_params(net, model_incomplete, time_stamp):
    if arguments.Config["bab"]["cut"]["implication"]:
        if time_stamp == 1:
            cut_info = net.set_dependencies(model_incomplete.A_saved)
            st_time = time.time()
            impl_params = {}
            implication_ret = build_bound_implication_graph(cut_info)
            impl_params["dependency_components"] = implication_ret[0]
            impl_params["index_mappings"] = implication_ret[1]
            impl_params["optimized_bound_indices"] = implication_ret[2]
            impl_params["optimized_bound_values"] = implication_ret[3]
            impl_params["unstable_idx"] = implication_ret[4]
            arguments.Config["bab"]["dep_time"] = time.time() - st_time
    else:
        impl_params = None
    return impl_params


def cut_verification(net, domains):
    cut_args = arguments.Config['bab']['cut']
    lp_cut_enabled = cut_args['lp_cut']
    cplex_cuts = cut_args['cplex_cuts']
    cplex_cuts_wait = cut_args['cplex_cuts_wait']

    print('======================Cut verification begins======================')
    start_cut = time.time()
    # enable lp solver
    if lp_cut_enabled:
        net.build_the_model_lp()
    if cplex_cuts:
        time.sleep(cplex_cuts_wait)
        generate_cplex_cuts(net)
    if len(domains) >= 1 and getattr(net.cutter, 'opt', False):
        # beta will be reused from split_history
        assert len(domains) == 1
        assert isinstance(domains[0].split_history['general_betas'], torch.Tensor)
        net.cutter.refine_cuts(split_history=domains[0].split_history)
    print('Cut time:', time.time() - start_cut)
    print('======================Cut verification ends======================')

def set_dependencies(self: 'LiRPANet', A):
    number_cuts = arguments.Config["bab"]["cut"]["number_cuts"]
    all_cuts, unstable_idx_list, ref_idx = add_implied_cuts(self, A, number_cuts=number_cuts, device=self.net.device)
    cut_info = {}
    cut_info["all_cuts"] = all_cuts
    cut_info["unstable_idx_list"] = unstable_idx_list
    cut_info["ref_idx"] = ref_idx
    return cut_info

def update_bounds_cut_naive(self: 'LiRPANet', d, split=None,
                            fix_interm_bounds=True, batchwise_out=True):
    pre_lb_all, pre_ub_all = d['lower_bounds'], d['upper_bounds'],
    alphas, history, cs = d['alphas'], d['history'], cs=d.get('cs', None)

    # batchwise_out: is to reshape the output into batchwise
    # True: used for get_lower_bound in bab; False: used for incomplete verifier
    solver_args = arguments.Config["solver"]
    bab_args = arguments.Config["bab"]
    early_stop_patience = solver_args["early_stop_patience"]
    start_save_best = solver_args["start_save_best"]
    beta = solver_args["beta-crown"]["beta"]
    optimizer = solver_args["beta-crown"]["optimizer"]
    lr_alpha = solver_args["beta-crown"]["lr_alpha"]
    lr_intermediate_beta = solver_args["intermediate_refinement"]["lr"]
    opt_coeffs = solver_args["intermediate_refinement"]["opt_coeffs"]
    opt_bias = solver_args["intermediate_refinement"]["opt_bias"]
    cut_iteration = bab_args["cut"]["iteration"]
    cut_lr_decay = bab_args["cut"]["lr_decay"]
    cut_lr_beta = bab_args["cut"]["lr_beta"]
    cut_early_stop_patience = bab_args["cut"]["early_stop_patience"]
    use_patches_cut = bab_args["cut"]["patches_cut"]
    cut_reference_bounds = bab_args["cut"]["cut_reference_bounds"]
    fix_intermediate_bounds = bab_args["cut"]["fix_intermediate_bounds"]

    if cut_early_stop_patience != -1:
        early_stop_patience = cut_early_stop_patience

    self.timer.start('func')
    ret_l, ret_u, ret_s = [[]], [[]], [[]]
    betas = [None]
    best_intermediate_betas = [defaultdict(dict)]
    new_split_history = [{}]
    self.net.beta_params = []
    self.net.single_beta_params = []
    self.net.single_beta_mask = []

    # get the cut version
    num_cuts = len(split["cut"])
    cut_timestamp = split["cut_timestamp"]
    self.net.cut_timestamp = cut_timestamp
    print("number of cut constraints:", num_cuts)
    print("cut timestamp:", cut_timestamp)
    self.timer.start('prepare')

    cut_module = self.cutter.construct_cut_module(use_patches_cut=use_patches_cut)
    self.net.cut_module = cut_module
    for m in self.net.splittable_activations:
        m.cut_module = cut_module

    # preset and compute bounds with the cut
    with torch.no_grad():
        upper_bounds = [i.clone() for i in pre_ub_all[:-1]]
        lower_bounds = [i.clone() for i in pre_lb_all[:-1]]
        pre_lb_all = [torch.cat([i]) for i in pre_lb_all]
        pre_ub_all = [torch.cat([i]) for i in pre_ub_all]

        # merge the inactive and active splits together
        new_interm_bounds = {}
        if cut_reference_bounds:
            for i, (uc, lc) in enumerate(zip(upper_bounds, lower_bounds)):
                # we set lower = 0 in first half batch, and upper = 0 in second half batch
                new_interm_bounds[self.name_dict[i]] = [lc, uc]

    # create new_x here since batch may change
    ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                x_L=self.x.ptb.x_L, x_U=self.x.ptb.x_U)
    new_x = BoundedTensor(self.x.data, ptb)
    self.net(new_x)  # batch may change, so we need to do forward to set some shapes here
    if cs is None:
        c = None if self.c is None else self.c
    else:
        c = cs
    self.timer.add('prepare')
    self.timer.start('bound')
    self.timer.start('beta_bound')
    # single node split True means only for single neuron split with regular beta crown
    self.net.set_bound_opts({'optimize_bound_args': {
        'enable_beta_crown': beta, 'opt_coeffs': opt_coeffs,
        'opt_bias': opt_bias, 'fix_interm_bounds': fix_interm_bounds,
        'iteration': cut_iteration, 'lr_decay': cut_lr_decay,
        'lr_alpha': lr_alpha, 'lr_cut_beta': cut_lr_beta,
        'lr_intermediate_beta': lr_intermediate_beta,
        'optimizer': optimizer, 'early_stop_patience': early_stop_patience,
        'start_save_best': start_save_best
    }})
    # set new interval if not want to run full beta crown with cut
    interm_bounds = new_interm_bounds if fix_intermediate_bounds else None

    self.cutter.construct_beta([item.shape for item in pre_lb_all])

    lb, _ = self.net.compute_bounds(x=(new_x,), C=c, method='CROWN-Optimized',
        reference_bounds=new_interm_bounds,
        interm_bounds=interm_bounds,
        bound_upper=False, cutter=self.cutter)
    print("##### cut lb:", lb[-1])
    self.timer.add('beta_bound')
    self.timer.add('bound')

    # save split and history constraints to new_split_history
    # new split history: [dict]
    with torch.no_grad():
        # only store the output obj start node betas
        new_split_history[0]["general_betas"] = cut_module.general_beta[self.net.final_name].detach()
        # need to attach timestamp of the cut for each domain
        new_split_history[0]["cut_timestamp"] = self.cutter.cut_timestamp

    if not arguments.Config["bab"]["cut"]["bab_cut"]:
        print("reset cut_enabled to False, disable cut in the following BaB")
        self.net.cut_used = False
        for m in self.net.splittable_activations:
            m.cut_used = False

    with torch.no_grad():
        if not batchwise_out:
            lb, ub = self.get_interm_bounds(lb)  # primals are better upper bounds
            mask = self.get_mask()
            lA = self.get_lA()
            alphas = self.get_alpha()[0]  # initial with one node only
            # FIXME ??? incompatible return value
            return ub[-1], lb[-1], None, None, None, mask[0], lA[0], lb, ub, None, alphas, history, new_split_history
        else:
            # Move tensors to CPU for all elements in this batch.
            self.timer.start('transfer')
            lb = lb.to(device='cpu')
            self.timer.add('transfer')
            self.timer.start('finalize')
            ub = torch.zeros_like(lb) + np.inf
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(
                lb, ub, device='cpu')
            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1].cpu())
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1].cpu())
            lAs = self.get_lA(device='cpu', transpose=False)
            # reshape the results to batch wise
            ret_l[0] = [j[:1] for j in lower_bounds_new]
            ret_u[0] = [j[:1] for j in upper_bounds_new]

            if len(alphas) > 0:
                ret_s = self.get_alpha(only_final=True, half=True, device='cpu')

    self.timer.add('func')
    self.timer.add('finalize')
    self.timer.print()

    # FIXME It looks like the return values here are no longer compatible with
    # beta_CROWN_solver since this commit (several items missing):
    # https://github.com/Verified-Intelligence/Verifier_Development/commit/99fa7f974434959f12dd093e82d123103fac06d2#diff-334e9c32db02844ccc46d2f267ddb6d766136b24b087c0ed88d60de72d519f19R128
    raise NotImplementedError('Please fix incompatible return values. '
                              'See comments in the code around this line.')

    return {
        'lower_bounds': ret_l, 'upper_bounds': ret_u,
        'lAs': lAs, 'alphas': ret_s, 'betas': betas,
        'split_history': new_split_history,
        'intermediate_betas': best_intermediate_betas,
        'c': c
    }


def set_cuts(self: 'LiRPANet', A, x, lower_bounds, upper_bounds,
             use_float64_in_last_iteration=False):
    assert len(lower_bounds) == len(upper_bounds) == len(self.net.splittable_activations) + 1
    for i, relu in enumerate(self.net.splittable_activations):
        relu.inputs[0].lower = lower_bounds[relu.inputs[0].name]
        relu.inputs[0].upper = upper_bounds[relu.inputs[0].name]
    self.net[self.net.final_name].lower = lower_bounds[self.net.final_name]
    self.net[self.net.final_name].upper = upper_bounds[self.net.final_name]

    cut_method = arguments.Config["bab"]["cut"]["method"]
    number_cuts = arguments.Config["bab"]["cut"]["number_cuts"]

    if arguments.Config["bab"]["cut"]["cplex_cuts"] or arguments.Config["bab"]["cut"]['biccos']["enabled"]:
        self.cutter = Cutter(self, A, x, number_cuts=number_cuts, device=self.net.device)
    
    cuts = None

    if cuts is None and not (arguments.Config["bab"]["cut"]["cplex_cuts"] or arguments.Config["bab"]["cut"]['biccos']["enabled"]):
        print("Warning: Cuts should either be automatically generated by enabling specifying --cut_method or manually given by --tmp_cuts")
        exit()


def create_mip_building_proc(self: 'LiRPANet', x):
    # throw error if "get_cuts" executable does not exist
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    if not is_exe(f'{CPLEX_FOLDER}/get_cuts'):
        raise Exception(f"CPLEX cutting planes are needed.\n"
                        f"However, the executable for generating them is not found, which should be in path '{CPLEX_FOLDER}/get_cuts'\n"
                        f"Please compile this executable by typing 'make' in directory {CPLEX_FOLDER}.")
    # (async) save gurobi mip model mps for each unverified labels and solve with cplex
    manager = multiprocessing.Manager()
    self.processes = manager.dict()
    intermediate_bounds = {}
    for name, layer in self.net._modules.items():
        layer_lower = layer.lower.clone().cpu() if layer.is_lower_bound_current() else None
        layer_upper = layer.upper.clone().cpu() if layer.is_upper_bound_current() else None
        if layer_lower is not None or layer_upper is not None:
            intermediate_bounds[name] = [layer_lower, layer_upper]  # Save its intermediate layer bounds in a dictionary.
    mip_building_proc = multiprocessing.Process(target=construct_mip_with_model, args=(
        copy.deepcopy(self.model_ori).cpu(), x.clone().to(device='cpu'), self.input_shape,
        self.c.clone().cpu(), intermediate_bounds, True, self.processes))
    mip_building_proc.start()
    self.mip_building_proc = mip_building_proc


def enable_cuts(self: 'LiRPANet'):
    self.return_A = True
    if self.needed_A_dict is None:
        self.needed_A_dict = defaultdict(set)
    self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
    for l in self.net.splittable_activations:
        self.needed_A_dict[l.inputs[0].name].add(self.net.input_name[0])


def create_cutter(self: 'LiRPANet', A, x, lb, ub, prune_after_crown):
    cut_args = arguments.Config["bab"]["cut"]
    if cut_args["cplex_cuts"] and self.mip_building_proc is None:
        if prune_after_crown:
            self.net.final_node().lower = lb[self.net.final_name]
            self.net.final_node().upper = ub[self.net.final_name]
        self.create_mip_building_proc(x)
        self.cutter = Cutter(
            self, A, x, number_cuts=cut_args["number_cuts"],
            device=self.net.device)
    # A for intermediate layers will be needed in cut construction
    self.A_saved = A


def set_cut_params(self: 'LiRPANet', batch_size, batch_base, split_history):
    cut_iteration = arguments.Config["bab"]["cut"]["bab_iteration"]

    num_constrs = self.net.cut_module.cut_bias.size(0)
    iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
    # Change the number of iterations during cuts.
    iteration = cut_iteration if cut_iteration > 0 else iteration

    # each general_beta: 2 (lA, uA), spec (out_c, out_h, out_w), batch, num_cuts
    general_beta = self.cutter.beta_init * torch.ones((2, 1, batch_size, num_constrs), device=self.net.device)
    if split_history is not None:
        # general beta warm up
        for batch_sh, sh in enumerate(split_history):
            if "general_betas" in sh and sh["cut_timestamp"] == self.net.cut_timestamp:
                if sh["general_betas"].shape[-1] == num_constrs:
                    # It only happens when using cplex cuts, the cuts are added before the
                    # bound computation, so that batch_base = len(split_history), we want to set the
                    # general beta for each batch.
                    assert batch_base == len(split_history)
                    general_beta[:, :, batch_sh: batch_sh+1, :] = sh["general_betas"].detach().clone()
                    general_beta[:, :, batch_sh+batch_base: batch_sh+batch_base+1, :] = sh["general_betas"].detach().clone()

    general_beta = general_beta.detach()
    general_beta.requires_grad = True
    general_betas = {self.net.final_name: general_beta}
    self.net.cut_beta_params = [general_betas[self.net.final_name]]
    for m in self.net.splittable_activations:
        m.cut_module = self.net.cut_module
        m.cut_used = True
    self.net.cut_module.general_beta = general_betas
    self.net.cut_module.cut_timestamps = [self.net.cut_timestamp for _ in range(batch_size)]
    print('cut re-enabled after branching node selection')

    return iteration


def set_cut_new_split_history(self: 'LiRPANet', new_split_history, batch_size):
    for i in range(batch_size):
        new_split_history[i]["general_betas"] = self.net.cut_module.general_beta[self.net.final_name][:, :, i:i + 1, :].detach()
        new_split_history[i]["cut_timestamp"] = self.net.cut_module.cut_timestamps[i]


def disable_cut_for_branching(self: 'LiRPANet'):
    """Disable cut_used for branching node selection.
    Reenable when beta is True.
    """
    print('cut disabled for branching node selection')
    self.net.cut_used = False
    for m in self.net.splittable_activations:
        m.cut_used = False
    self.net.cut_beta_params = []
