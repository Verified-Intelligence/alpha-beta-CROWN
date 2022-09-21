#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import copy
import time
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import numpy as np
import arguments
import warnings
import multiprocessing

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import stop_criterion_sum

from lp_mip_solver import *
from attack_pgd import attack
from cuts import Cutter


total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_transfer_time = total_finalize_time = 0.0


class LiRPAConvNet:
    def __init__(
            self, model_ori, in_size, c=None, device=None,
            cplex_processes=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.c = c
        self.model_ori = model_ori
        self.layers = layers
        self.input_shape = in_size
        self.device = device or arguments.Config["general"]["device"]
        self.net = BoundedModule(
            net, torch.zeros(in_size, device=self.device),
            bound_opts={
                'relu': 'adaptive',
                'deterministic': arguments.Config["general"]["deterministic"],
                'conv_mode': arguments.Config["general"]["conv_mode"],
                'sparse_features_alpha': arguments.Config["general"]["sparse_alpha"],
                'sparse_spec_alpha': arguments.Config["general"]["sparse_alpha"],
                'crown_batch_size': arguments.Config["solver"]["crown"]["batch_size"],
                'max_crown_size': arguments.Config["solver"]["crown"]["max_crown_size"],
                'forward_refinement': arguments.Config["solver"]["forward"]["refine"],
                'dynamic_forward': arguments.Config["solver"]["forward"]["dynamic"],
                'forward_max_dim': arguments.Config["solver"]["forward"]["max_dim"],
                'use_full_conv_alpha': arguments.Config["solver"]["alpha-crown"]["full_conv_alpha"],
            },
            device=self.device
        )
        self.net.eval()
        self.return_A = False
        self.needed_A_dict = None
        self.pool = self.pool_result = self.pool_termination_flag = None # For multi-process.
        self.cutter = None # class for generating and optimizing cuts

        # for fetching cplex in parallel
        self.mip_building_proc = None
        self.processes = None
        self.cplex_processes = cplex_processes

        # for recording whether we need to return intermediate bounds
        # after initial intermediate bounds fetching, this switch will be aligned with arg.bab.interm_transfer
        self.interm_transfer = True

        # check conversion correctness
        dummy = torch.randn(in_size, device=self.device)
        try:
            assert torch.allclose(net(dummy), self.net(dummy))
        except AssertionError:
            print(f'torch allclose failed: norm {torch.norm(net(dummy) - self.net(dummy))}')

    def get_lower_bound(self, pre_lbs, pre_ubs, split, slopes=None, betas=None, history=None, fix_intermediate_layer_bounds=True,
                        split_history=None, single_node_split=True, intermediate_betas=None, cs=None, decision_thresh=None, rhs=0,
                        stop_func=stop_criterion_sum(0), multi_spec_keep_func=None):

        """
        # (in) pre_lbs: layers list -> tensor(batch, layer shape)
        # (in) relu_mask: relu layers list -> tensor(batch, relu layer shape (view-1))
        # (in) slope: relu layers list -> tensor(batch, relu layer shape)
        # (out) lower_bounds: batch list -> layers list -> tensor(layer shape)
        # (out) masks_ret: batch list -> relu layers list -> tensor(relu layer shape)
        # (out) slope: batch list -> relu layers list -> tensor(relu layer shape)
        """
        if history is None:
            history = []
        start = time.time()

        if "cut" in split:
            ret = self.update_bounds_cut_naive(
                pre_lbs, pre_ubs, split, slopes,
                history=history,
                fix_intermediate_layer_bounds=fix_intermediate_layer_bounds,
                cs=cs)
        else:
            ret = self.update_bounds_parallel(
                pre_lbs, pre_ubs, split, slopes, betas=betas, history=history,
                fix_intermediate_layer_bounds=fix_intermediate_layer_bounds,
                split_history=split_history, cs=cs, decision_thresh=decision_thresh,
                stop_criterion_func=stop_func, multi_spec_keep_func=multi_spec_keep_func)

        # if get_upper_bound and single_node_split, primals have p and z values; otherwise None
        lower_bounds, upper_bounds, lAs, slopes, betas, split_history, best_intermediate_betas, primals, new_cs = ret

        beta_time = time.time()-start


        end = time.time()
        print('batch bounding time: ', end - start)

        return upper_bounds[-1], lower_bounds[-1], None, lAs, lower_bounds, \
               upper_bounds, slopes, split_history, betas, best_intermediate_betas, primals, new_cs


    def get_relu(self, model, idx):
        # find the i-th ReLU layer
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer


    def transfer_to_cpu(self, net, non_blocking=True, opt_intermediate_beta=False, transfer_items="all"):
        """Trasfer all necessary tensors to CPU in a batch."""

        # Create a data structure holding all the tensors we need to transfer.
        class TMP:
            pass
        cpu_net = TMP()
        cpu_net.perturbed_optimizable_activations = [None] * len (net.perturbed_optimizable_activations)
        for i in range(len(cpu_net.perturbed_optimizable_activations)):
            cpu_net.perturbed_optimizable_activations[i] = lambda : None
            cpu_net.perturbed_optimizable_activations[i].inputs = [lambda : None]
            cpu_net.perturbed_optimizable_activations[i].name = net.perturbed_optimizable_activations[i].name

        transfer_size = defaultdict(int)
        # Transfer data structures for each relu.
        # For get_candidate_parallel().
        if transfer_items == "all":
            if self.interm_transfer:
                for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
                    # For get_candidate_parallel.
                    cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
                    cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)
                    transfer_size['pre'] += layer.inputs[0].lower.numel() * 2
            # For get_lA_parallel().
            for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
                cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
                transfer_size['lA'] += layer.lA.numel()
        # For get_slope().
        if transfer_items == "all" or transfer_items == "slopes":
            for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
                # Per-neuron alpha.
                cpu_layer.alpha = OrderedDict()
                for spec_name, alpha in layer.alpha.items():
                    cpu_layer.alpha[spec_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)
                    transfer_size['alpha'] += alpha.numel()
        # For get_beta().
        if transfer_items == "all":
            for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
                if hasattr(layer, 'sparse_beta') and layer.sparse_beta is not None:
                    if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                        cpu_layer.sparse_beta = OrderedDict()
                        for key in layer.sparse_beta.keys():
                            cpu_layer.sparse_beta[key] = layer.sparse_beta[key].to(device='cpu', non_blocking=non_blocking)
                            transfer_size['beta'] += layer.sparse_beta[key].numel()
                    else:
                        cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)
                        transfer_size['beta'] += layer.sparse_beta.numel()
            # For intermediate beta.
            if opt_intermediate_beta and net.best_intermediate_betas is not None:
                cpu_net.best_intermediate_betas = OrderedDict()
                for split_layer, all_int_betas_this_layer in net.best_intermediate_betas.items():
                    # Single neuron split so far.
                    assert 'single' in all_int_betas_this_layer
                    assert 'history' not in all_int_betas_this_layer
                    assert 'split' not in all_int_betas_this_layer
                    cpu_net.best_intermediate_betas[split_layer] = {'single': defaultdict(dict)}
                    for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                        cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['lb'] = this_layer_intermediate_betas['lb'].to(device='cpu', non_blocking=non_blocking)
                        cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['ub'] = this_layer_intermediate_betas['ub'].to(device='cpu', non_blocking=non_blocking)
                        transfer_size['itermediate_beta'] += this_layer_intermediate_betas['lb'].numel() * 2
        print(f'Tensors transferred: {" ".join("{}={:.4f}M".format(k, v / 1048576) for (k, v) in transfer_size.items())}')
        return cpu_net


    def get_primal_upper_bound(self, A):
        with torch.no_grad():
            assert self.x.ptb.norm == np.inf, print('we only support to get primals for Linf norm perturbation so far')
            input_A_lower = A[self.net.output_name[0]][self.net.input_name[0]]["lA"]
            batch = input_A_lower.shape[0]

            x_lb, x_ub = self.x.ptb.x_L, self.x.ptb.x_U
            x_lb = x_lb.repeat(batch, 1, 1, 1)
            x_ub = x_ub.repeat(batch, 1, 1, 1)
            input_primal = x_lb.clone().detach()
            input_primal[input_A_lower.squeeze(1) < 0] = x_ub[input_A_lower.squeeze(1) < 0]

        return input_primal, self.model_ori(input_primal).matmul(self.c[0].transpose(-1, -2))


    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs
        # by default, we also add final layer bound after applying c, and this final layer bound should be passed
        # by lb and ub arguments (which is always passed in)

        # FIXME: these lower bounds, upper bounds should be dicts, not lists.
        lower_bounds = []
        upper_bounds = []
        self.pre_relu_indices = []
        i = 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {}

        for layer in model.perturbed_optimizable_activations:
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())
            self.name_dict[i] = layer.inputs[0].name
            if isinstance(layer, BoundRelu):
                self.pre_relu_indices.append(i)
            i += 1

        # Also add the bounds on the final thing
        lower_bounds.append(lb.flatten(1).detach())  # TODO merge with get_candidate_parallel to handle multi-x
        upper_bounds.append(ub.flatten(1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices


    def get_candidate_parallel(self, model, lb, ub, batch, diving_batch=0):
        # get the intermediate bounds in the current model
        lower_bounds = []
        upper_bounds = []

        for layer in model.perturbed_optimizable_activations:
            if self.interm_transfer:
                lower_bounds.append(layer.inputs[0].lower)
                upper_bounds.append(layer.inputs[0].upper)
            else:
                lower_bounds.append(None)
                upper_bounds.append(None)

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch + diving_batch, -1).detach())
        upper_bounds.append(ub.view(batch + diving_batch, -1).detach())

        return lower_bounds, upper_bounds


    def get_mask_lA_parallel(self, model):
        if len(model.perturbed_optimizable_activations) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.perturbed_optimizable_activations:
            # 1 is unstable neuron, 0 is stable neuron.
            mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float()
            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            if hasattr(this_relu, 'lA') and this_relu.lA is not None:
                lA.append(this_relu.lA.transpose(0, 1))
            else:
                # It might be skipped due to inactive neurons.
                lA.append(None)

        return mask, lA

    def get_lA_parallel(self, model, preserve_mask=None, tot_cells=None, to_cpu=False):
        if len(model.perturbed_optimizable_activations) == 0:
            return [None]
        # get lower A matrix of ReLU
        lA = []
        if preserve_mask is not None:
            for this_relu in model.perturbed_optimizable_activations:
                new_lA = torch.zeros([tot_cells, this_relu.lA.shape[0]] + list(this_relu.lA.shape[2:]),
                                     dtype=this_relu.lA.dtype,
                                     device=this_relu.lA.device)
                new_lA[preserve_mask] = this_relu.lA.transpose(0,1)
                lA.append(new_lA.to(device='cpu', non_blocking=False) if to_cpu else new_lA)
        else:
            for this_relu in model.perturbed_optimizable_activations:
                lA.append(this_relu.lA.transpose(0,1).to(device='cpu', non_blocking=False) if to_cpu else this_relu.lA.squeeze(0))

        return lA

    def get_beta(self, model, splits_per_example, diving_batch=0):
        # split_per_example only has half of the examples.
        batch = splits_per_example.size(0) - diving_batch
        retb = [[] for _ in range(batch * 2 + diving_batch)]
        for mi, m in enumerate(model.perturbed_optimizable_activations):
            if hasattr(m, 'sparse_beta'):
                # Save only used beta, discard padding beta.
                if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                    for i in range(batch):
                        val_i = []
                        val_i_plus_batch = []
                        for key in m.sparse_beta.keys():
                            val_i.append(m.sparse_beta[key][i, :splits_per_example[i, mi]])
                            val_i_plus_batch.append(m.sparse_beta[key][i + batch, :splits_per_example[i, mi]])
                        retb[i].append(val_i)
                        retb[i + batch].append(val_i_plus_batch)
                    for i in range(diving_batch):
                        retb[2 * batch + i].append(
                            m.sparse_beta[key][2 * batch + i, :splits_per_example[batch + i, mi]])
                else:
                    for i in range(batch):
                        retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                        retb[i + batch].append(m.sparse_beta[i + batch, :splits_per_example[i, mi]])
                    for i in range(diving_batch):
                        retb[2 * batch + i].append(m.sparse_beta[2 * batch + i, :splits_per_example[batch + i, mi]])
        return retb


    def get_slope(self, model, only_final=False):
        if len(model.perturbed_optimizable_activations) == 0:
            return {}

        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.

        ret = {}
        kept_layer_names = [self.net.final_name]
        kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
        for m in model.perturbed_optimizable_activations:
            ret[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                if not only_final or spec_name in kept_layer_names:
                    ret[m.name][spec_name] = alpha

        return ret

    def set_slope(self, model, slope, intermediate_refinement_layers=None, diving_batch=0, set_all=False):
        cleanup_intermediate_slope = isinstance(intermediate_refinement_layers, list) and len(intermediate_refinement_layers) == 0
        if cleanup_intermediate_slope:
            # Clean all intermediate betas if we are not going to refine intermeidate layer neurons anymore.
            del model.best_intermediate_betas
            for m in model.perturbed_optimizable_activations:
                if hasattr(m, 'single_intermediate_betas'):
                    print(f'deleting single_intermediate_betas for {m.name}')
                    del m.single_intermediate_betas
                if hasattr(m, 'history_intermediate_betas'):
                    print(f'deleting history_intermediate_betas for {m.name}')
                    del m.history_intermediate_betas
                if hasattr(m, 'split_intermediate_betas'):
                    print(f'deleting split_intermediate_betas for {m.name}')
                    del m.split_intermediate_betas
        kept_layer_names = [self.net.final_name]
        kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
        if type(slope) == list:
            for m in model.perturbed_optimizable_activations:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[0][m.name]:
                        if cleanup_intermediate_slope and spec_name not in kept_layer_names:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name in kept_layer_names or set_all:
                            if len(slope) - diving_batch > 0:
                                # Merge all slope vectors together in this batch. Size is (2, spec, batch, *shape).
                                m.alpha[spec_name] = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch)], dim=2)
                                # Duplicate for the second half of the batch.
                                m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3))).detach().requires_grad_()
                            if diving_batch > 0:
                                # create diving alpha
                                diving_alpha = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch, len(slope))], dim=2)
                                if diving_batch == len(slope):
                                    m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                                else:
                                    m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                                del diving_alpha
                            # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        elif type(slope) == defaultdict:
            for m in model.perturbed_optimizable_activations:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[m.name]:
                        if cleanup_intermediate_slope and spec_name not in kept_layer_names:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name in kept_layer_names or set_all:
                            slope_len = slope[m.name][spec_name].size(2)
                            if slope_len - diving_batch > 0:
                                # print(f'set_slope: non diving {m.name} {spec_name} size {slope[m.name][spec_name].size()}')
                                m.alpha[spec_name] = slope[m.name][spec_name]
                                # Duplicate for the second half of the batch.
                                m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3))).detach().requires_grad_()
                            if diving_batch > 0:
                                assert diving_batch == slope_len  # This is a temporary workaround to run bab-attack.
                                # print(f'set_slope: diving {m.name} {spec_name} size {slope[m.name][spec_name].size()}')
                                m.alpha[spec_name] = slope[m.name][spec_name].detach().requires_grad_()
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        else:
            raise NotImplementedError

    def reset_beta(self, batch, max_splits_per_layer=None, betas=None, diving_batch=0):
        # Recreate new beta with appropriate shape.
        for mi, m in enumerate(self.net.perturbed_optimizable_activations):
            if isinstance(m, BoundRelu):
                # Create only the non-zero beta. For each layer, it is padded to maximal length.
                # We create tensors on CPU first, and they will be transferred to GPU after initialized.
                if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                    m.sparse_beta, m.sparse_beta_loc, m.sparse_beta_sign = {}, {}, {}
                    for key in m.alpha.keys():
                        m.sparse_beta[key] = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                        m.sparse_beta_loc[key] = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                        m.sparse_beta_sign[key] = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    # Load beta from history.
                    for bi in range(batch):
                        if betas is not None and betas[bi] is not None:
                            # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                            for i, key in enumerate(m.sparse_beta.keys()):
                                valid_betas = len(betas[bi][mi][i])
                                m.sparse_beta[key][bi, :valid_betas] = betas[bi][mi][i]
                    # This is the beta variable to be optimized for this layer.
                    for key in m.sparse_beta.keys():
                        m.sparse_beta[key] = m.sparse_beta[key].repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()
                else:
                    m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                    m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    # Load beta from history.
                    for bi in range(batch):
                        if betas is not None and betas[bi] is not None:
                            # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                            valid_betas = len(betas[bi][mi])
                            m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
                    # This is the beta variable to be optimized for this layer.
                    m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()

                assert betas is None or batch + diving_batch == len(betas)
                if diving_batch != 0:
                    m.diving_sparse_beta = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    m.diving_sparse_beta_loc = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                    m.diving_sparse_beta_sign = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    # Load diving beta from history.
                    for dbi in range(diving_batch):
                        if betas[batch + dbi] is not None:
                            # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                            valid_betas = len(betas[batch + dbi][mi])
                            m.diving_sparse_beta[dbi, :valid_betas] = betas[batch + dbi][mi]
                    m.diving_sparse_beta = m.diving_sparse_beta.to(device=self.net.device, non_blocking=True)
                    m.sparse_beta = torch.cat([m.sparse_beta, m.diving_sparse_beta], dim=0).detach().\
                                to(device=self.net.device, non_blocking=True).requires_grad_()
                    del m.diving_sparse_beta

    """Main function for computing bounds after branch and bound in Beta-CROWN."""
    def update_bounds_parallel(
            self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None,
            beta=None, betas=None, history=None, fix_intermediate_layer_bounds=True, shortcut=False,
            split_history=None, cs=None, decision_thresh=None, stop_criterion_func=stop_criterion_sum(0),
            multi_spec_keep_func=None):
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time

        if beta is None:
            beta = arguments.Config["solver"]["beta-crown"]["beta"] # might need to set beta False in FSB node selection
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]['lr_alpha']
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        enable_opt_interm_bounds = arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']
        pruning_in_iteration = arguments.Config["bab"]["pruning_in_iteration"]
        pruning_in_iteration_threshold = arguments.Config["bab"]["pruning_in_iteration_ratio"]
        cut_iteration = arguments.Config["bab"]["cut"]["bab_iteration"]
        lr_cut_beta = arguments.Config["bab"]["cut"]["lr_beta"]
        cut_lr = arguments.Config["bab"]["cut"]["lr"]
        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0

        diving_batch = 0
        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)

        if "diving" in split:
            diving_batch = split["diving"]
            print(f"regular batch size: 2*{batch}, diving batch size 1*{diving_batch}")

        ret_s = [[] for _ in range(batch * 2 + diving_batch)]
        ret_b = [[] for _ in range(batch * 2 + diving_batch)]
        # Each key is corresponding to a pre-relu layer, and each value intermediate
        # beta values for neurons in that layer.
        new_split_history = [{} for _ in range(batch * 2 + diving_batch)]
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2 + diving_batch)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        start_prepare_time = time.time()
        # iteratively change upper and lower bound from former to later layer

        self.net.cut_beta_params = []
        if self.net.cut_used:
            # disable cut_used for branching node selection, reenable when beta is True
            print('cut disabled for branching node selection')
            self.net.cut_used = False
            for m in self.net.relus:
                m.cut_used = False
            self.net.cut_beta_params = []

        if beta:
            # count how many split nodes in each batch example (batch, num of layers)
            splits_per_example = torch.zeros(
                size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu',
                requires_grad=False)
            for bi in range(batch):
                d = decision[bi][0]
                for mi, layer_splits in enumerate(history[bi]):
                    splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.
            # This is the maximum number of split in each relu neuron for each batch.
            if batch > 0: max_splits_per_layer = splits_per_example.max(dim=0)[0]

            if diving_batch != 0:
                diving_splits_per_example = torch.zeros(size=(diving_batch, len(self.net.relus)),
                            dtype=torch.int64, device='cpu', requires_grad=False)
                for dbi in range(diving_batch):
                    # diving batch does not have decision splits but only have history splits
                    for mi, diving_layer_splits in enumerate(history[dbi + batch]):
                        diving_splits_per_example[dbi, mi] = len(diving_layer_splits[0])  # First element of layer_splits is a list of split neuron IDs.
                splits_per_example = torch.cat([splits_per_example, diving_splits_per_example], dim=0)
                max_splits_per_layer = splits_per_example.max(dim=0)[0]
                del diving_splits_per_example

            # Create and load warmup beta.
            self.reset_beta(batch, betas=betas, max_splits_per_layer=max_splits_per_layer, diving_batch=diving_batch)  # warm start beta

            if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                for bi in range(batch):
                    # Add history splits.
                    d, idx = decision[bi][0], decision[bi][1]
                    # Each history element has format [[[layer 1's split location], [layer 1's split coefficients +1/-1]], [[layer 2's split location], [layer 2's split coefficients +1/-1]], ...].
                    for mi, (split_locs, split_coeffs) in enumerate(history[bi]):
                        split_len = len(split_locs)
                        for key in self.net.relus[mi].sparse_beta.keys():
                            self.net.relus[mi].sparse_beta_sign[key][bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                            self.net.relus[mi].sparse_beta_loc[key][bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                        # Add current decision for positive splits.
                        if mi == d:
                            for key in self.net.relus[mi].sparse_beta.keys():
                                self.net.relus[mi].sparse_beta_sign[key][bi, split_len] = 1.0
                                self.net.relus[mi].sparse_beta_loc[key][bi, split_len] = idx
                # Duplicate split location.
                for m in self.net.relus:
                    for key in m.sparse_beta.keys():
                        m.sparse_beta_loc[key] = m.sparse_beta_loc[key].repeat(2, 1).detach()
                        m.sparse_beta_loc[key] = m.sparse_beta_loc[key].to(device=self.net.device, non_blocking=True)
                        m.sparse_beta_sign[key] = m.sparse_beta_sign[key].repeat(2, 1).detach()
                # Fixup the second half of the split (negative splits).
                for bi in range(batch):
                    d = decision[bi][0]  # layer of this split.
                    split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
                    for key in self.net.relus[d].sparse_beta_sign.keys():
                        self.net.relus[d].sparse_beta_sign[key][bi + batch, split_len] = -1.0
                # Transfer tensors to GPU.
                for m in self.net.relus:
                    for key in m.sparse_beta_sign.keys():
                        m.sparse_beta_sign[key] = m.sparse_beta_sign[key].to(device=self.net.device, non_blocking=True)
            else:
                for bi in range(batch):
                    # Add history splits.
                    d, idx = decision[bi][0], decision[bi][1]
                    # Each history element has format [[[layer 1's split location], [layer 1's split coefficients +1/-1]], [[layer 2's split location], [layer 2's split coefficients +1/-1]], ...].
                    for mi, (split_locs, split_coeffs) in enumerate(history[bi]):
                        split_len = len(split_locs)
                        self.net.relus[mi].sparse_beta_sign[bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                        self.net.relus[mi].sparse_beta_loc[bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                        # Add current decision for positive splits.
                        if mi == d:
                            self.net.relus[mi].sparse_beta_sign[bi, split_len] = 1.0
                            self.net.relus[mi].sparse_beta_loc[bi, split_len] = idx
                # Duplicate split location.
                for m in self.net.relus:
                    m.sparse_beta_loc = m.sparse_beta_loc.repeat(2, 1).detach()
                    m.sparse_beta_loc = m.sparse_beta_loc.to(device=self.net.device, non_blocking=True)
                    m.sparse_beta_sign = m.sparse_beta_sign.repeat(2, 1).detach()
                # Fixup the second half of the split (negative splits).
                for bi in range(batch):
                    d = decision[bi][0]  # layer of this split.
                    split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
                    self.net.relus[d].sparse_beta_sign[bi + batch, split_len] = -1.0
                # Transfer tensors to GPU.
                for m in self.net.relus:
                    m.sparse_beta_sign = m.sparse_beta_sign.to(device=self.net.device, non_blocking=True)

            if diving_batch > 0:
                # add diving domains history splits, no decision in diving domains
                for dbi in range(diving_batch):
                    for mi, (split_locs, split_coeffs) in enumerate(history[dbi + batch]):
                        split_len = len(split_locs)
                        self.net.relus[mi].diving_sparse_beta_sign[dbi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                        self.net.relus[mi].diving_sparse_beta_loc[dbi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                for m in self.net.relus:
                    # cat beta loc and sign to have the correct shape
                    m.diving_sparse_beta_loc = m.diving_sparse_beta_loc.to(device=self.net.device, non_blocking=True)
                    m.diving_sparse_beta_sign = m.diving_sparse_beta_sign.to(device=self.net.device, non_blocking=True)
                    m.sparse_beta_loc = torch.cat([m.sparse_beta_loc, m.diving_sparse_beta_loc], dim=0).detach()
                    m.sparse_beta_sign = torch.cat([m.sparse_beta_sign, m.diving_sparse_beta_sign], dim=0).detach()
                    # do no need to store the diving beta params any more
                    del m.diving_sparse_beta_loc, m.diving_sparse_beta_sign

            self.net.cut_used = arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["bab_cut"]
            # even we need to use cut, maybe the cut is not fetched yet

            batch_size = batch * 2 + diving_batch
            if self.net.cut_used and getattr(self.net, "cut_module", None) is not None:
                num_constrs = self.net.cut_module.cut_bias.size(0)
                # Change the number of iterations during cuts.
                iteration = cut_iteration if cut_iteration > 0 else iteration

                # each general_beta: 2 (lA, uA), spec (out_c, out_h, out_w), batch, num_cuts
                # print('init general_beta to 0')
                general_beta = self.cutter.beta_init * torch.ones((2, 1, batch_size, num_constrs), device=self.net.device)
                cut_timestamps = [self.net.cut_timestamp for _ in range(batch_size)]
                if split_history is not None:
                    # general beta warm up
                    for batch_sh, sh in enumerate(split_history):
                        if "general_betas" in sh and sh["cut_timestamp"] == self.net.cut_timestamp:
                            assert sh["general_betas"].shape[-1] == num_constrs
                            assert batch == len(split_history)
                            general_beta[:, :, batch_sh: batch_sh+1, :] = sh["general_betas"].detach().clone()
                            general_beta[:, :, batch_sh+batch: batch_sh+batch+1, :] = sh["general_betas"].detach().clone()

                general_beta = general_beta.detach()
                general_beta.requires_grad = True
                general_betas = {self.net.final_name: general_beta}
                self.net.cut_beta_params = [general_betas[self.net.final_name]]
                for m in self.net.relus:
                    m.cut_module = self.net.cut_module
                    m.cut_used = True
                self.net.cut_module.general_beta = general_betas
                print('cut re-enabled after branching node selection')
            ###### here to handle the case where the split node happen to be in the cut constraint !!! ######
        else:
            for m in self.net.relus:
                m.beta = None

        # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
        with torch.no_grad():
            # Setting the neuron upper/lower bounds with a split to 0.
            zero_indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
            zero_indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
            for i in range(batch):
                d, idx = decision[i][0], decision[i][1]
                # We save the batch, and neuron number for each split, and will set all corresponding elements in batch.
                zero_indices_batch[d].append(i)
                zero_indices_neuron[d].append(idx)
            zero_indices_batch = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_batch]
            zero_indices_neuron = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_neuron]

            # 2 * batch + diving_batch
            upper_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_ub_all[:-1]]
            lower_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_lb_all[:-1]]

            # 2 * cs
            if cs is not None:
                double_cs = torch.cat([cs[:batch], cs[:batch], cs[batch:]], dim=0)

            # Only the last element is used later.
            pre_lb_last = torch.cat([pre_lb_all[-1][:batch], pre_lb_all[-1][:batch], pre_lb_all[-1][batch:]])
            pre_ub_last = torch.cat([pre_ub_all[-1][:batch], pre_ub_all[-1][:batch], pre_ub_all[-1][batch:]])

            new_intermediate_layer_bounds = {}
            for d in range(len(lower_bounds)):
                # for each layer except the last output layer
                if len(zero_indices_batch[d]):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    lower_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                    upper_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                new_intermediate_layer_bounds[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]

        # create new_x here since batch may change
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                 x_L=self.x.ptb.x_L[0].expand(batch * 2 + diving_batch, *[-1]*(self.x.ptb.x_L.ndim-1)),
                                 x_U=self.x.ptb.x_U[0].expand(batch * 2 + diving_batch, *[-1]*(self.x.ptb.x_L.ndim-1)))
        new_x = BoundedTensor(self.x.data.expand(batch * 2 + diving_batch, *[-1]*(self.x.data.ndim-1)), ptb)
        if cs is None:
            c = None if self.c is None else self.c.expand(new_x.shape[0], -1, -1)
        else:
            # sample-wise C for supporting handling multiple targets in one batch
            c = double_cs

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes, diving_batch=diving_batch)

        if decision_thresh is not None and isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1:
            decision_thresh = torch.cat([decision_thresh, decision_thresh], dim=0)

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': beta, 'single_node_split': True,
                                                             'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds,
                                                             'optimizer':optimizer,
                                                             'pruning_in_iteration': pruning_in_iteration,
                                                             'pruning_in_iteration_threshold': pruning_in_iteration_threshold},
                                                             'enable_opt_interm_bounds': enable_opt_interm_bounds,})
            with torch.no_grad():
                lb, _, = self.net.compute_bounds(x=(new_x,), C=c, method='backward', reuse_alpha=True,
                                                 intermediate_layer_bounds=new_intermediate_layer_bounds, bound_upper=False)
            return lb

        return_A = True if get_upper_bound else False  # we need A matrix to construct adv example

        original_size = new_x.shape[0]

        if fix_intermediate_layer_bounds:
            start_beta_bound_time = time.time()
            self.net.set_bound_opts({'optimize_bound_args': {
                'enable_beta_crown': beta, 'single_node_split': True,
                'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds, 'iteration': iteration,
                'lr_alpha': lr_alpha, 'lr_decay': lr_decay, 'lr_beta': lr_beta,
                'optimizer': optimizer,
                'pruning_in_iteration': pruning_in_iteration,
                'pruning_in_iteration_threshold': pruning_in_iteration_threshold,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': multi_spec_keep_func},
                'enable_opt_interm_bounds': enable_opt_interm_bounds,
                'lr_cut_beta': lr_cut_beta,
            })
            kept_layer_names = list(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            for name in kept_layer_names:
                print(f'Removing intermediate layer bounds for layer {name}.')
                del new_intermediate_layer_bounds[name]
            print(new_x.shape, c.shape, decision_thresh.shape if decision_thresh is not None else None)
            tmp_ret = self.net.compute_bounds(
                x=(new_x,), C=c, method='CROWN-Optimized',
                intermediate_layer_bounds=new_intermediate_layer_bounds,
                return_A=return_A, needed_A_dict=self.needed_A_dict, cutter=self.cutter,
                bound_upper=False, decision_thresh=decision_thresh)
            beta_bound_time += time.time() - start_beta_bound_time
            # we don't care about the upper bound of the last layer
        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts({'optimize_bound_args': {
                'enable_beta_crown': beta, 'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds,
                'iteration': iteration, 'lr_alpha': lr_alpha, 'lr_decay': lr_decay,
                'lr_beta': lr_beta, 'optimizer': optimizer,
                'pruning_in_iteration': pruning_in_iteration,
                'pruning_in_iteration_threshold': pruning_in_iteration_threshold,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': multi_spec_keep_func},
                'enable_opt_interm_bounds': enable_opt_interm_bounds,
                'lr_cut_beta': lr_cut_beta,
            })
            tmp_ret = self.net.compute_bounds(
                x=(new_x,), C=c, method='CROWN-Optimized', intermediate_layer_bounds=new_intermediate_layer_bounds,
                return_A=return_A, needed_A_dict=self.needed_A_dict, cutter=self.cutter,
                bound_upper=False, decision_thresh=decision_thresh)

        if get_upper_bound:
            lb, _, A = tmp_ret
            primal_x, ub = self.get_primal_upper_bound(A)
        else:
            lb, _ = tmp_ret
            ub = torch.zeros_like(lb) + np.inf # dummy upper bound
            primal_x = None

        bound_time += time.time() - start_bound_time

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            # indexing on GPU seems to be faster, so get_lA_parallel() is conducted on GPU side then move to CPU
            lAs = self.get_lA_parallel(self.net, self.net.last_update_preserve_mask, original_size, to_cpu=True)
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()


            if len(slopes) > 0:
                ret_s = self.get_slope(transfer_net)

            if beta:
                ret_b = self.get_beta(transfer_net, splits_per_example, diving_batch=diving_batch)

            # Reorganize tensors.
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, batch * 2, diving_batch=diving_batch)

            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_last.cpu())
            if not get_upper_bound:
                # Do not set to min so the primal is always corresponding to the upper bound.
                upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_last.cpu())
            ret_l, ret_u = lower_bounds_new, upper_bounds_new

            finalize_time = time.time() - start_finalize_time

        if self.net.cut_used and getattr(self.net, "cut_module", None) is not None:
            for i in range(2 * batch + diving_batch):
                new_split_history[i]["general_betas"] = general_beta[:, :, i:i + 1, :].detach()
                new_split_history[i]["cut_timestamp"] = cut_timestamps[i]

        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {total_transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')

        # if primals is not None: ret_p = self.layer_wise_primals(primals)

        # assert (ret_p[1]['p'][0][0] == primal_x[1]).all()
        return ret_l, ret_u, lAs, ret_s, ret_b, new_split_history, best_intermediate_betas, primal_x, c

    def update_bounds_cut_naive(
            self, pre_lb_all=None, pre_ub_all=None, split=None,
            slopes=None, history=None, fix_intermediate_layer_bounds=True,
            batchwise_out=True, cs=None):
        # batchwise_out: is to reshape the output into batchwise
        # True: used for get_lower_bound in bab; False: used for incomplete verifier
        beta = arguments.Config["solver"]["beta-crown"]["beta"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        cut_iteration = arguments.Config["bab"]["cut"]["iteration"]
        cut_lr_decay = arguments.Config["bab"]["cut"]["lr_decay"]
        cut_lr_beta = arguments.Config["bab"]["cut"]["lr_beta"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]["lr_alpha"]
        lr_intermediate_beta = arguments.Config["solver"]["intermediate_refinement"]["lr"]
        opt_coeffs = arguments.Config["solver"]["intermediate_refinement"]["opt_coeffs"]
        opt_bias = arguments.Config["solver"]["intermediate_refinement"]["opt_bias"]
        early_stop_patience = arguments.Config["solver"]["early_stop_patience"]
        start_save_best = arguments.Config["solver"]["start_save_best"]
        cut_early_stop_patience = arguments.Config["bab"]["cut"]["early_stop_patience"]
        use_patches_cut = arguments.Config["bab"]["cut"]["patches_cut"]
        cut_reference_bounds = arguments.Config["bab"]["cut"]["cut_reference_bounds"]
        fix_intermediate_bounds = arguments.Config["bab"]["cut"]["fix_intermediate_bounds"]
        opt_intermediate_beta  = False

        if cut_early_stop_patience != -1:
            early_stop_patience = cut_early_stop_patience

        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time
        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0

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

        start_prepare_time = time.time()

        cut_module = self.cutter.construct_cut_module(use_patches_cut=use_patches_cut)
        self.net.cut_module = cut_module
        for m in self.net.relus:
            m.cut_module = cut_module

        # preset and compute bounds with the cut
        with torch.no_grad():
            upper_bounds = [i.clone() for i in pre_ub_all[:-1]]
            lower_bounds = [i.clone() for i in pre_lb_all[:-1]]
            pre_lb_all = [torch.cat([i]) for i in pre_lb_all]
            pre_ub_all = [torch.cat([i]) for i in pre_ub_all]

            # merge the inactive and active splits together
            new_intermediate_layer_bounds = {}
            if cut_reference_bounds:
                for i, (uc, lc) in enumerate(zip(upper_bounds, lower_bounds)):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    new_intermediate_layer_bounds[self.name_dict[i]] = [lc, uc]

        # create new_x here since batch may change
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                 x_L=self.x.ptb.x_L, x_U=self.x.ptb.x_U)
        new_x = BoundedTensor(self.x.data, ptb)
        self.net(new_x)  # batch may change, so we need to do forward to set some shapes here
        if cs is None:
            c = None if self.c is None else self.c
        else:
            c = cs

        prepare_time += time.time() - start_prepare_time

        start_bound_time = time.time()
        # single node split True means only for single neuron split with regular beta crown
        self.net.set_bound_opts({'optimize_bound_args': {
            'enable_beta_crown': beta, 'single_node_split': True,  'opt_coeffs': opt_coeffs,
            'opt_bias': opt_bias, 'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds,
            'iteration': cut_iteration, 'lr_decay': cut_lr_decay,
            'lr_alpha': lr_alpha, 'lr_cut_beta': cut_lr_beta,
            'lr_intermediate_beta': lr_intermediate_beta,
            'optimizer': optimizer, 'early_stop_patience': early_stop_patience,
            'start_save_best': start_save_best
        }})
        # set new interval if not want to run full beta crown with cut
        intermediate_layer_bounds = new_intermediate_layer_bounds if fix_intermediate_bounds else None

        self.cutter.construct_beta([item.shape for item in pre_lb_all])

        lb, _ = self.net.compute_bounds(x=(new_x,), C=c, method='CROWN-Optimized',
            reference_bounds=new_intermediate_layer_bounds,
            intermediate_layer_bounds=intermediate_layer_bounds,
            bound_upper=False, cutter=self.cutter)
        print("##### cut lb:", lb[-1])
        beta_bound_time += time.time() - start_bound_time
        bound_time += time.time() - start_bound_time

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
            for m in self.net.relus:
                m.cut_used = False

        with torch.no_grad():
            if not batchwise_out:
                ub = torch.zeros_like(lb) + np.inf
                lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, ub)  # primals are better upper bounds
                mask, lA = self.get_mask_lA_parallel(self.net)
                slope_opt = self.get_slope(self.net)[0]  # initial with one node only
                return ub[-1], lb[-1], None, None, None, mask[0], lA[0], lb, ub, None, slope_opt, history, new_split_history
            else:
                # Move tensors to CPU for all elements in this batch.
                start_transfer_time = time.time()
                lb = lb.to(device='cpu')
                transfer_net = self.transfer_to_cpu(self.net, non_blocking=False, opt_intermediate_beta=opt_intermediate_beta)
                transfer_time = time.time() - start_transfer_time

                start_finalize_time = time.time()
                ub = torch.zeros_like(lb) + np.inf
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, 1)
                lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1].cpu())
                upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1].cpu())
                lAs = self.get_lA_parallel(transfer_net)
                # reshape the results to batch wise
                ret_l[0] = [j[:1] for j in lower_bounds_new]
                ret_u[0] = [j[:1] for j in upper_bounds_new]

                if len(slopes) > 0:
                    ret_s = self.get_slope(transfer_net, only_final=True)


        finalize_time = time.time() - start_finalize_time
        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {total_transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')
        return ret_l, ret_u, lAs, ret_s, betas, new_split_history, best_intermediate_betas, None, c

    def set_cuts(self, A, x, lower_bounds, upper_bounds, use_float64_in_last_iteration=False):
        assert len(lower_bounds) == len(upper_bounds) == len(self.net.relus) + 1
        for i, relu in enumerate(self.net.relus):
            relu.inputs[0].lower = lower_bounds[i]
            relu.inputs[0].upper = upper_bounds[i]
        self.net[self.net.final_name].lower = lower_bounds[-1]
        self.net[self.net.final_name].upper = upper_bounds[-1]

        cut_method = arguments.Config["bab"]["cut"]["method"]
        number_cuts = arguments.Config["bab"]["cut"]["number_cuts"]

        if arguments.Config["bab"]["cut"]["cplex_cuts"] and self.mip_building_proc is None:
            self.cutter = Cutter(self, A, x, number_cuts=number_cuts, device=self.net.device)

        cuts = None

        if cuts is None and not arguments.Config["bab"]["cut"]["cplex_cuts"]:
            print("Warning: Cuts should either be automatically generated by enabling specifying --cut_method or manually given by --tmp_cuts")
            exit()

    def build_the_model(self, input_domain, x, data_lb=None, data_ub=None, vnnlib=None, stop_criterion_func=stop_criterion_sum(0), bounding_method=None):
        """
            return_crown_bounds is only used by incomplete_verifier
        """
        # TODO merge with build_the_model_with_refined_bounds()

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        lr_decay = arguments.Config["solver"]["alpha-crown"]["lr_decay"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        use_float64_in_last_iteration = arguments.Config["solver"]["use_float64_in_last_iteration"]
        early_stop_patience = arguments.Config["solver"]["early_stop_patience"]
        start_save_best = arguments.Config["solver"]["start_save_best"]
        bounding_method = bounding_method or arguments.Config["solver"]["bound_prop_method"]

        self.x = x
        self.input_domain = input_domain
        loss_reduction_func = reduction_str2func(loss_reduction_func)
        self._set_A_options()


        if arguments.Config["bab"]["cut"]["enabled"]:
            self.return_A = True
            if self.needed_A_dict is None:
                self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
            for l in self.net.relus:
                self.needed_A_dict[l.inputs[0].name].add(self.net.input_name[0])

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        self.net.set_bound_opts({'verbosity': 1})
        self.net.set_bound_opts({'optimize_bound_args': {
            'iteration': init_iteration, 'enable_beta_crown': False, 'enable_alpha_crown': True,
            'use_shared_alpha': share_slopes, 'optimizer': optimizer,
            'early_stop': False,
            'keep_best': True, 'fix_intermediate_layer_bounds': True,
            'lr_alpha': lr_init_alpha, 'init_alpha': False,
            'loss_reduction_func': loss_reduction_func,
            'stop_criterion_func': stop_criterion_func,
            'lr_decay': lr_decay, 'use_float64_in_last_iteration': use_float64_in_last_iteration,
            'early_stop_patience': early_stop_patience, 'start_save_best': start_save_best}})

        prune_after_crown_used = False

        if bounding_method == "alpha-crown":
            # first get CROWN bounds
            # Reference bounds are intermediate layer bounds from initial CROWN bounds.
            lb, ub, aux_reference_bounds = self.net.init_slope(
                (self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
            print('initial CROWN bounds:', lb, ub)

            if stop_criterion_func(lb).all().item():
                # Fast path. Initial CROWN bound can verify the network.
                return np.inf, lb, None, None, None, None, None, None, None, None, None, None, None

            if arguments.Config["attack"]["pgd_order"] == "middle" and vnnlib is not None:
                crown_filtered_constraints = np.zeros(len(lb[-1]))
                for i in range(len(lb[-1])):
                    if isinstance(arguments.Config["bab"]["decision_thresh"], torch.Tensor):
                        if arguments.Config["bab"]["decision_thresh"].shape[0] > 1:
                            if lb[-1][i].item() > arguments.Config["bab"]["decision_thresh"][i].item():
                                crown_filtered_constraints[i] = True
                        else:
                            if lb[-1][i].item() > arguments.Config["bab"]["decision_thresh"][0].item():
                                crown_filtered_constraints[i] = True
                    else:
                        if lb[-1][i].item() > arguments.Config["bab"]["decision_thresh"]:
                            crown_filtered_constraints[i] = True

                verified_status = "unknown"
                verified_success = False

                verified_status, verified_success, attack_images, _, _ = attack(
                    self.model_ori, x, data_lb, data_ub, [vnnlib], verified_status, verified_success, crown_filtered_constraints)

                if verified_success:  # Adversarial images are generated here.
                    print("PGD attack succeeded.")
                    return None, lb[-1], None, None, None, None, None, None, None, None, None, None, attack_images

            c_to_use = self.c
            if arguments.Config["solver"]["prune_after_crown"]:
                prune_after_crown_overhead = 0.
                stime = time.time()
                onedim_decision_thresh = arguments.Config["bab"]["decision_thresh"]
                assert not (isinstance(onedim_decision_thresh, torch.Tensor) and onedim_decision_thresh.shape[-1] > 1), \
                    "Multiple spec is not compatible with prune after CROWN optimization yet."
                if isinstance(onedim_decision_thresh, torch.Tensor):
                    onedim_decision_thresh = onedim_decision_thresh.view(-1)
                final_layer_lb = lb[-1]
                unverified_label_mask = (final_layer_lb <= onedim_decision_thresh).nonzero().view(-1)
                c_to_use = self.c[:, unverified_label_mask]
                # fix the slope shape
                for relu in self.net.relus:
                    if relu.alpha is not None and self.net.final_name in relu.alpha:
                        relu.alpha[self.net.final_name] = relu.alpha[self.net.final_name][:, unverified_label_mask].detach()
                prune_after_crown_used = True
                prune_after_crown_overhead += time.time() - stime
                print('prune_after_crown optimization in use: original label size =', final_layer_lb.shape[0], 'pruned label size =', len(unverified_label_mask))

            ret = self.net.compute_bounds(x=(x,), C=c_to_use, method='CROWN-Optimized',
                return_A=self.return_A, needed_A_dict=self.needed_A_dict,
                bound_upper=False, aux_reference_bounds=aux_reference_bounds, cutter=self.cutter)
        elif bounding_method == 'alpha-forward':
            warnings.warn('alpha-forward can only be used with input split for now')
            self.net.bound_opts['optimize_bound_args']['init_alpha'] = True
            ret = self.net.get_optimized_bounds(
                x=(x,), C=self.c, method='forward', bound_upper=False)
        elif bounding_method == 'init-crown':
            with torch.no_grad():
                lb, ub, aux_reference_bounds = self.net.init_slope(
                    (self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
                print('initial CROWN bounds:', lb, ub)
                lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + np.inf)
                mask, lA = self.get_mask_lA_parallel(self.net)
                history = [[[], []] for _ in range(len(self.net.relus))]
                slope_opt = self.get_slope(self.net)  # initial with one node only
            return ub[-1], lb[-1], None, None, None, mask, lA, lb, ub, pre_relu_indices, slope_opt, history, None
        else:
            with torch.no_grad():
                lb, _ = self.net.compute_bounds(
                    x=(x,), C=self.c, method=bounding_method, cutter=self.cutter, bound_upper=False)
                print(f'initial {bounding_method} bounds (first 10):', lb.flatten()[:10])
                lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + np.inf)
            return ub[-1], lb[-1], None, None, None, None, None, lb, ub, pre_relu_indices, None, None, None

        if self.return_A:
            lb, ub, A = ret
        else:
            lb, ub = ret; A = None

        if prune_after_crown_used:
            stime = time.time()
            # recover full shape lb, ub
            if lb is not None:
                # new_final_layer_lb = torch.zeros_like(final_layer_lb, device=final_layer_lb.device) + onedim_decision_thresh + 1e-2
                new_final_layer_lb = torch.full_like(final_layer_lb, float("inf"))
                new_final_layer_lb = new_final_layer_lb.unsqueeze(0)
                new_final_layer_lb[:, unverified_label_mask] = lb
                lb = new_final_layer_lb
            prune_after_crown_overhead += time.time() - stime

        print('initial alpha-CROWN bounds:', lb)
        initial_alpha_crown_bounds = lb.min().item()
        print('Worst class: (+ rhs)', initial_alpha_crown_bounds)

        slope_opt = self.get_slope(self.net)  # initial with one node only
        # for each pre-relu layer, we initial 2 lists for active and inactive split
        history = [[[], []] for _ in range(len(self.net.relus))]
        primals, duals, mini_inp = None, None, None
        ub = torch.zeros_like(lb) + np.inf
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, ub)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)

        if prune_after_crown_used:
            stime = time.time()
            with torch.no_grad():
                # handle lA
                newlA = []
                for Aitem in lA:
                    newAshape = list(Aitem.shape)
                    newAshape[1] = final_layer_lb.shape[0]
                    newA = torch.zeros(newAshape, device=Aitem.device, dtype=Aitem.dtype)
                    newA[:, unverified_label_mask] = Aitem
                    newlA.append(newA)
                lA = newlA
                # handle slope_opt
                for k, v in slope_opt.items():
                    if self.net.final_name in v:
                        oldslope = v[self.net.final_name]
                        slopeshape = list(oldslope.shape)
                        slopeshape[1] = final_layer_lb.shape[0]
                        newslope = torch.zeros(slopeshape, device=oldslope.device, dtype=oldslope.dtype)
                        newslope[:, unverified_label_mask] = oldslope
                        v[self.net.final_name] = newslope
                # since we may reread the slope from the network, we push the full shape slope back to net
                for m in self.net.relus:
                    if m.name in slope_opt:
                        m.alpha = slope_opt[m.name]
            prune_after_crown_overhead += time.time() - stime
            print('  prune after CROWN overhead:', prune_after_crown_overhead, 's')

        if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and self.mip_building_proc is None:
            # self.set_cuts(model_incomplete.A_saved, x, lower_bounds, upper_bounds)
            if prune_after_crown_used:
                self.net.final_node().lower = lb[-1]
                self.net.final_node().upper = ub[-1]
            self.create_mip_building_proc(x)
            self.cutter = Cutter(self, A, x, number_cuts=arguments.Config["bab"]["cut"]["number_cuts"], device=self.net.device)

        if arguments.Config["bab"]["cut"]["enabled"]:
            # A for intermediate layers will be needed in cut construction
            self.A_saved = A

        return ub[-1], lb[-1], mini_inp, duals, primals, mask, lA, lb, ub, pre_relu_indices, slope_opt, history, None

    def copy_alpha(
            self, reference_slopes, num_targets, target_batch_size=None,
            now_batch=None, intermediate_layer_bounds=None, batch_size=None):
        # alpha manipulation, since after init_slope all things are copied from alpha-CROWN and these alphas may have wrong shape
        opt_intermediate_beta  = False
        opt_interm_bounds = arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
        for m in self.net.relus:
            keys = list(m.alpha.keys())
            # when fixed intermediate bounds are available, since intermediate betas are not used anymore because we use fixed intermediate bounds later, we can delete these intermediate betas to save space
            if intermediate_layer_bounds is not None and not opt_interm_bounds and not opt_intermediate_beta:
                for k in keys:
                    if k != self.net.final_node().name:
                        del m.alpha[k]
            if (m.alpha[self.net.final_node().name].shape[1] != 1
                    or m.alpha[self.net.final_node().name].shape[2] != batch_size):
                # shape mismatch detected
                # pick the first slice with shape [2, 1, 1, ...], and repeat to [2, 1, batch_size, ...]
                repeat = [1 if i != 2 else batch_size for i in range(m.alpha[self.net.final_node().name].dim())]
                m.alpha[self.net.final_node().name] = (
                    m.alpha[self.net.final_node().name][:, 0:1, 0:1].repeat(*repeat))

        if reference_slopes is None:
            return False

        # We already have slopes available
        all_slope_initialized = True
        for m in self.net.relus:
            for spec_name, alpha in m.alpha.items():
                def not_setting_alpha():
                    print(f"not setting layer {m.name} start_node {spec_name} because shape mismatch ({alpha.size()} != {reference_alpha.size()})")
                # each slope size is (2, spec, batch_size, *shape); batch size is 1.
                if not spec_name in reference_slopes[m.name]:
                    continue
                reference_alpha = reference_slopes[m.name][spec_name]
                if spec_name == self.net.final_node().name:
                    target_start = now_batch * target_batch_size
                    target_end = min((now_batch + 1) * target_batch_size, num_targets)
                    if alpha.size()[2] == target_end - target_start:
                        print(f"setting alpha for layer {m.name} start_node {spec_name} with alignment adjustment")
                        # The reference alpha has deleted the pred class itself, while our alpha keeps that
                        # now align these two
                        # note: this part actually implements the following TODO (extract alpha according to different label)
                        if reference_alpha.size()[1] > 1:
                            # didn't apply multiple x in incomplete_verifier
                            alpha.data = reference_alpha[:, target_start:target_end].reshape_as(alpha.data)
                        else:
                            # applied multiple x in incomplete_verifier
                            alpha.data = reference_alpha[:, :, target_start:target_end].reshape_as(alpha.data)
                    else:
                        all_slope_initialized = False
                        not_setting_alpha()
                elif alpha.size() == reference_alpha.size():
                    print(f"setting alpha for layer {m.name} start_node {spec_name}")
                    alpha.data.copy_(reference_alpha)
                elif all([si == sj or ((d == 2) and sj == 1) for d, (si, sj) in enumerate(zip(alpha.size(), reference_alpha.size()))]):
                    print(f"setting alpha for layer {m.name} start_node {spec_name} with batch sample broadcasting")
                    alpha.data.copy_(reference_alpha)
                else:
                    # TODO extract alpha according to different label
                    all_slope_initialized = False
                    not_setting_alpha()

        return all_slope_initialized

    @staticmethod
    def prune_reference_slopes(reference_slopes, keep_condition, final_node_name):
        for m, spec_dict in reference_slopes.items():
            for spec in spec_dict:
                if spec == final_node_name:
                    if spec_dict[spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = spec_dict[spec][:, keep_condition]
                    else:
                        spec_dict[spec] = spec_dict[spec][:, :, keep_condition]

    @staticmethod
    def prune_lA(lA, keep_condition):
        return [lAitem[:, keep_condition] for lAitem in lA]

    def build_the_model_with_refined_bounds(self, input_domain, x, refined_lower_bounds, refined_upper_bounds,
            activation_opt_params=None, reference_lA=None,
            stop_criterion_func=stop_criterion_sum(0), reference_slopes=None,
            cutter=None, refined_betas=None):
        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        use_float64_in_last_iteration = arguments.Config["solver"]["use_float64_in_last_iteration"]
        target_batch_size = arguments.Config["solver"]["multi_class"]["label_batch_size"]
        start_save_best = arguments.Config["solver"]["start_save_best"]
        opt_intermediate_beta  = False

        self.x = x
        self.input_domain = input_domain
        self.cutter = cutter

        # expand x to align with C's batch size for multi target verification
        x_expand = x.clone()
        x_expand.data = x_expand.data.expand(*([self.c.size()[0]] + [-1] * (self.x.dim() - 1)))
        x_expand.ptb.x_L = x_expand.ptb.x_L.expand(*([self.c.size()[0]] + [-1] * (self.x.dim() - 1)))
        x_expand.ptb.x_U = x_expand.ptb.x_U.expand(*([self.c.size()[0]] + [-1] * (self.x.dim() - 1)))

        # also, we need to expand lower and upper bounds accordingly
        if refined_lower_bounds is not None and refined_upper_bounds is not None:
            # the intermediate bounds were shared in incomplete_verifier(), we expand them here
            refined_lower_bounds = [value[0:1].expand(*([self.c.size()[0]] + [-1] * (value.dim() - 1)))
                                    for value in refined_lower_bounds[:-1]] + [refined_lower_bounds[-1]]
            refined_upper_bounds = [value[0:1].expand(*([self.c.size()[0]] + [-1] * (value.dim() - 1)))
                                    for value in refined_upper_bounds[:-1]] + [refined_upper_bounds[-1]]

        loss_reduction_func = reduction_str2func(loss_reduction_func)
        self.refined_lower_bounds, self.refined_upper_bounds = refined_lower_bounds, refined_upper_bounds

        primals, duals, mini_inp = None, None, None

        self._set_A_options(get_upper_bound=get_upper_bound)

        # tot label batches
        tot_batches = (x_expand.size()[0] + target_batch_size - 1) // target_batch_size

        # batch results holder
        batch_lbs, batch_ubs, lA, slope_opts = [], [], [], {}

        for now_batch in range(tot_batches):
            print('build_the_model_with_refined_bounds batch [{}/{}]'.format(now_batch, tot_batches))

            # save gpu memory usage
            torch.cuda.empty_cache()

            batch_expand = BoundedTensor(
                x_expand.data[now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                PerturbationLpNorm(
                    x_expand.ptb.eps, x_expand.ptb.norm,
                    x_expand.ptb.x_L[now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                    x_expand.ptb.x_U[now_batch * target_batch_size: (now_batch + 1) * target_batch_size]))

            if refined_lower_bounds is not None and refined_upper_bounds is not None:
                # using refined bounds with init opt crown
                intermediate_layer_bounds = {}
                for i, layer in enumerate(self.net.relus):
                    nd = layer.inputs[0].name
                    intermediate_layer_bounds[nd] = [refined_lower_bounds[i][now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                                        refined_upper_bounds[i][now_batch * target_batch_size: (now_batch + 1) * target_batch_size]]
            else:
                intermediate_layer_bounds = None

            self.net.init_slope((batch_expand,), share_slopes=share_slopes,
                                c=self.c[now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                                intermediate_layer_bounds=intermediate_layer_bounds, activation_opt_params=activation_opt_params, skip_bound_compute=True)

            all_slope_initialized = self.copy_alpha(
                reference_slopes, num_targets=min((now_batch + 1) * target_batch_size, self.c.shape[0]) - now_batch * target_batch_size,
                target_batch_size=target_batch_size, now_batch=now_batch,
                intermediate_layer_bounds=intermediate_layer_bounds, batch_size=batch_expand.shape[0])

            self.net.set_bound_opts({'optimize_bound_args': {
                'iteration': init_iteration, 'enable_beta_crown': False, 'enable_alpha_crown': True,
                'use_shared_alpha': share_slopes, 'optimizer': optimizer,
                'early_stop': False, 'keep_best': True,
                'fix_intermediate_layer_bounds': True, 'lr_alpha': lr_init_alpha, 'init_alpha': False,
                'loss_reduction_func': loss_reduction_func,
                'stop_criterion_func': stop_criterion_func,
                'lr_decay': lr_decay, 'use_float64_in_last_iteration': use_float64_in_last_iteration,
                'start_save_best': start_save_best}})

            skip_backward_pass = False
            if all_slope_initialized and arguments.Config["solver"]["multi_class"]["skip_with_refined_bound"] is True:
                print('all slope initialized')
                bound_method = 'backward'
                if not self.return_A:
                    skip_backward_pass = True
                    print('directly get lb and ub from refined bounds')
                    # make sure the shape of reference_lA looks good so that we can recover the batch_lA
                    print('lA shapes:', [A.shape for A in reference_lA])
                    print('c shape:', self.c.shape)
                    assert all([A.shape[1] == self.c.shape[0] for A in reference_lA])
                    # try to directly recover l and u from refined_lower_bounds and refined_upper_bounds without a backward crown pass
                    # refined_lower/upper_bounds[-1]'s shape is [labels to verify, C]
                    # self.c's shape is [labels to verify, 1, C] where target labels have value -1.
                    lb = refined_lower_bounds[-1][now_batch * target_batch_size: (now_batch + 1) * target_batch_size]
                    ub = refined_upper_bounds[-1][now_batch * target_batch_size: (now_batch + 1) * target_batch_size]
                    ret = (lb, ub)
                else:
                    skip_backward_pass = False
                    # do a backward crown pass
                    print('true A is required, we do a full backward CROWN pass to obtain it')
                    ret = self.net.compute_bounds(
                        x=(batch_expand,), method=bound_method,
                        C=self.c[now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                        return_A=self.return_A, reuse_alpha=True, intermediate_layer_bounds=intermediate_layer_bounds,
                        needed_A_dict=self.needed_A_dict)
            else:
                print('restore to original setting since some slopes are not initialized yet or being asked not to skip')
                bound_method = 'crown-optimized'
                ret = self.net.compute_bounds(
                    x=(batch_expand,), method=bound_method, return_A=self.return_A,
                    C=self.c[now_batch * target_batch_size: (now_batch + 1) * target_batch_size],
                    intermediate_layer_bounds=intermediate_layer_bounds, needed_A_dict=self.needed_A_dict)

            if self.return_A:
                lb, ub, A = ret
            else:
                lb, ub = ret; A = None

            print("alpha-CROWN with fixed intermediate bounds:", lb, ub)
            slope_opt = self.get_slope(self.net)
            if arguments.Config["bab"]["attack"]["enabled"]:
                # Save all slopes, which will be further refined in bab-attack.
                self.refined_slope = reference_slopes
            batch_lb, batch_ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + np.inf)  # primals are better upper bounds


            mask, batch_lA = self.get_mask_lA_parallel(self.net)
            if skip_backward_pass:
                # change shape from incomplete verifier's lA because in incomplete verifier's [batch, spec, ...], spec is the current batch dim
                batch_lA = [item.transpose(0, 1) for item in reference_lA]

            history = [[[], []] for _ in range(len(self.net.relus))]
            ret_b = None
            if refined_betas is not None:
                # only has batch size 1 for refined betas
                assert len(refined_betas[0]) == 1
                history = refined_betas[0][0]
                ret_b = refined_betas[1][0]

            if get_upper_bound:
                print("opt crown:", lb[-1])
                primal_x, ub_x = self.get_primal_upper_bound(A)
                print("### Extracting primal values for inputs done ###")

            # early slope delete to save space
            if not opt_intermediate_beta:
                # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
                # We only keep the alpha for the last layer.
                new_slope_opt = {}
                kept_layer_names = [self.net.final_name]
                kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
                print(f'Keeping slopes for these layers: {kept_layer_names}')
                for relu_layer, alphas in slope_opt.items():
                    new_slope_opt[relu_layer] = {}
                    for layer_name in kept_layer_names:
                        new_slope_opt[relu_layer][layer_name] = alphas[layer_name]
            del slope_opt
            slope_opt = new_slope_opt

            batch_lbs.append(batch_lb)
            batch_ubs.append(batch_ub)
            if now_batch == 0:
                lA += batch_lA
            else:
                # need to accumulate itemwise over the 0 dim, since A's shape is [batch, spec=1, ...]
                lA = [torch.cat([accu_Aitem, new_Aitem], dim=0) for accu_Aitem, new_Aitem in zip(lA, batch_lA)]
            for k in slope_opt:
                if k not in slope_opts: slope_opts[k] = {}
                for kk, v in slope_opt[k].items():
                    if kk not in slope_opts[k]:
                        slope_opts[k][kk] = v
                    else:
                        slope_opts[k][kk] = torch.cat([slope_opts[k][kk], v], dim=2)

        # merge all things from the batch
        lb = [torch.cat([item_lb[i] for item_lb in batch_lbs]) for i in range(len(batch_lbs[0]))]
        ub = [torch.cat([item_ub[i] for item_ub in batch_ubs]) for i in range(len(batch_ubs[0]))]

        return ub[-1], lb[-1], mini_inp, duals, primals, mask, lA, lb, ub, pre_relu_indices, slope_opts, history, ret_b


    def get_lower_bound_naive(
            self, dm_l=None, dm_u=None, slopes=None,
            bounding_method="crown", C=None, stop_criterion_func=None):
        batch = len(dm_l)//2
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps, x_L=dm_l, x_U=dm_u)
        new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
        lA = None

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes, set_all=True)

        self.net.set_bound_opts({'optimize_bound_args': {
            'enable_beta_crown': False, 'single_node_split': True,
            'fix_intermediate_layer_bounds': True,
            'iteration': arguments.Config["solver"]["beta-crown"]["iteration"],
            'lr_alpha': arguments.Config["solver"]["beta-crown"]['lr_alpha'],
            'stop_criterion_func': stop_criterion_func,
        }})

        needed_A_dict = defaultdict(set)
        needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
        if bounding_method == "alpha-crown":
            lb, _, A_dict = self.net.compute_bounds(
                x=(new_x,), C=C, method='CROWN-Optimized', bound_upper=False,
                return_A=True, needed_A_dict=needed_A_dict)
            lA = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']
        elif bounding_method == 'alpha-forward':
            raise ValueError("Should not use alpha-forward.")
        else:
            with torch.no_grad():
                lb, _, A_dict = self.net.compute_bounds(x=(new_x,), C=C, method=bounding_method,
                        bound_upper=False, return_A=True, needed_A_dict=needed_A_dict)
                lA = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']

        with torch.no_grad():
            # Transfer everything to CPU.
            lb = lb.cpu()
            if bounding_method == "alpha-crown":
                transfer_net = self.transfer_to_cpu(self.net, non_blocking=False, transfer_items="slopes")
                ret_s = self.get_slope(transfer_net)
            else:
                ret_s = [None] * (batch * 2)

        # FIXME returning lb + np.inf is meaningless
        return lb, [None] * (batch * 2), ret_s, lA

    def get_neuron_primal(self, input_primal, lb, ub, slope_opt=None):
        # calculate the primal values for intermediate neurons
        # slope_opt is a list, each element has the dict for slopes of each batch

        if slope_opt is None:
            slope_opt = self.get_slope(self.net)

        batch_size = input_primal.shape[0]
        primal_values = [input_primal]
        # save the integer primal values in MIP constructions
        integer_primals = []
        primal = input_primal
        relu_idx = 0
        keys = list(slope_opt[0].keys())
        output_key = list(slope_opt[0][keys[0]].keys())[-1]
        # load saved primals from gurobi lp for debug
        # gurobi_primals = None
        # gurobi_primals = [np.load(f"gurobi_primals/{i}.npy") for i in range(10)]
        # gurobi_integer_primals = [np.load(f"gurobi_primals/z_relu{relu_idx}.npy") for relu_idx in range(5)]

        dual_values = torch.zeros((batch_size, 1), device=primal.device)

        for layer in self.layers:
            if not isinstance(layer, nn.ReLU):
                # just propagate the primal value if linear function or flatten layer
                primal = layer(primal)
            else:
                # only store input, pre_relu primal values, and output primals
                primal_values.append(primal.clone().detach())

                # handling nonlinear relus for primal propagations
                # we can use the lA from get_mask_lA_parallel but relu.lA is more straightforward
                # lA = lAs[0][relu_idx]
                lA = self.net.relus[relu_idx].lA.squeeze(0)

                # primal on lower boundary: lA<=0 & unstable
                u, l = ub[relu_idx].to(primal.device), lb[relu_idx].to(primal.device)
                unstable = (u > 0).logical_and(l < 0)

                # slope = slope_opt[which batch][keys[relu_idx]][output_key][0, 0]
                slope = self.net.relus[relu_idx].alpha[output_key][0, 0].to(primal.device)
                primal_l = primal * slope
                z_l =  primal / u
                z_l[z_l < 0] = 0

                # primal on upper boundary: lA>0 & unstable
                slope = (u / (u-l))
                bias = (-u * l / (u - l))
                primal_u = (primal * slope + bias).detach()
                z_u = (primal - l) / (u - l)
                # calculate z integer first, using last linear layer node primal values
                z = z_u
                z[(lA>0).logical_and(unstable)] = z_l[(lA>0).logical_and(unstable)]

                primal[(lA<=0).logical_and(unstable)] = primal_u[(lA<=0).logical_and(unstable)].detach()
                primal[(lA>0).logical_and(unstable)] = primal_l[(lA>0).logical_and(unstable)].detach()
                primal[(u<0)] = 0

                if self.net.relus[relu_idx].sparse_beta is not None and self.net.relus[relu_idx].sparse_beta.nelement() != 0:
                    beta_loc = self.net.relus[relu_idx].sparse_beta_loc
                    sparse_beta = self.net.relus[relu_idx].sparse_beta * self.net.relus[relu_idx].sparse_beta_sign

                    # we select split neuron from primal with tuple index
                    beta_loc_tuple = (torch.ones(beta_loc.shape).nonzero(as_tuple=True)[0], beta_loc.view(-1))
                    # we get the pre relu primal values for each split node
                    selected_primals = primal.view(batch_size, -1).gather(dim=1, index=beta_loc)
                    # we will add beta * split node pre relu primal to the eventual primal output obj
                    dual_values = dual_values + (sparse_beta * selected_primals).sum(1, keepdim=True).detach()
                    # for split node, we need to force choice 1 to be pre relu primal and 0 for choice 0
                    beta_c = (self.net.relus[relu_idx].sparse_beta_sign.view(-1) + 1) / 2
                    primal.view(batch_size, -1)[beta_loc_tuple] = primal_values[-1].view(batch_size, -1)[beta_loc_tuple] * beta_c
                    # force mixed integer z to be 1 and 0 for split nodes
                    z.view(batch_size, -1)[beta_loc_tuple] = beta_c

                # store the primal values of mixed integers
                # if z[unstable].view(-1).shape[0] % batch_size !=0:
                #     import pdb; pdb.set_trace()
                ip = torch.ones(z.shape, device=z.device) * (-1.)
                ip[unstable] = z[unstable]
                integer_primals.append(ip.view(batch_size, -1))

                # We should not force primal to be larger than 0, otherwise not correct !!!
                # primal = layer(primal)
                relu_idx += 1

            # primal_values.append(primal.clone().detach())

        primal_values.append(primal.clone().detach())
        primal_values[-1] = primal_values[-1] - dual_values

        integer_primals = [iv.to(device='cpu', non_blocking=True) for iv in integer_primals]
        primal_values = [pv.to(device='cpu', non_blocking=True) for pv in primal_values]

        return primal_values, integer_primals

    def layer_wise_primals(self, primals):
        # originally layer -> batch,
        # now need to be a list with batch elements
        neuron_primals, integer_primals = primals["p"], primals["z"]
        ret_p = []
        for bi in range(neuron_primals[0].size(0)):
            pv, iv = [], []
            for layer_idx in range(len(neuron_primals)):
                pv.append(neuron_primals[layer_idx][bi:bi + 1])
            for relu_idx in range(len(integer_primals)):
                iv.append(integer_primals[relu_idx][bi:bi + 1])
            ret_p.append({"p": pv, "z": iv})
        return ret_p

    def _set_A_options(self, get_upper_bound=False):
        if get_upper_bound:
            self.return_A = True
            if self.needed_A_dict is None:
                self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

    def create_mip_building_proc(self, x):
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
            layer_lower = layer.lower.clone().cpu() if hasattr(layer, 'lower') and isinstance(layer.lower, torch.Tensor) else None
            layer_upper = layer.upper.clone().cpu() if hasattr(layer, 'upper') and isinstance(layer.upper, torch.Tensor) else None
            if layer_lower is not None or layer_upper is not None:
                intermediate_bounds[name] = [layer_lower, layer_upper]  # Save its intermediate layer bounds in a dictionary.
        mip_building_proc = multiprocessing.Process(target=construct_mip_with_model, args=(
            copy.deepcopy(self.model_ori).cpu(), x.clone().to(device='cpu'), self.input_shape,
            self.c.clone().cpu(), intermediate_bounds, True, self.processes))
        mip_building_proc.start()
        self.mip_building_proc = mip_building_proc


    from lp_mip_solver import (build_solver_model, update_mip_model_fix_relu,
                            build_the_model_mip_refine, build_the_model_lp, build_the_model_mip,
                            all_node_split_LP)
