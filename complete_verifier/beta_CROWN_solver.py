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
import copy
import time
import random
from collections import defaultdict, OrderedDict

import torch
import arguments

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import reduction_sum, stop_criterion_sum, stop_criterion_min

from lp_mip_solver import *


total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_transfer_time = total_finalize_time = 0.0


class LiRPAConvNet:
    def __init__(self, model_ori, pred, test, device='cuda', simplify=False, in_size=(1, 3, 32, 32),
                 conv_mode='patches', deterministic=False, c=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.simplify = False
        self.c = c
        self.pred = pred
        self.layers = layers
        self.input_shape = in_size
        self.net = BoundedModule(net, torch.zeros(in_size, device=device), bound_opts={'relu': 'adaptive', 'deterministic': deterministic, 'conv_mode': conv_mode},
                                 device=device)
        self.net.eval()
        self.needed_A_dict = None
        self.pool = None   # For multi-process.
        self.pool_result = None
        self.pool_termination_flag = None

    
    def get_lower_bound(self, pre_lbs, pre_ubs, split, slopes=None, betas=None, history=None, layer_set_bound=True, 
                        split_history=None, single_node_split=True, intermediate_betas=None):

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

        lp_test = arguments.Config["debug"]["lp_test"]

        if single_node_split:
            ret = self.update_bounds_parallel(pre_lbs, pre_ubs, split, slopes, betas=betas, early_stop=False, history=history,
                                              layer_set_bound=layer_set_bound)
        else:
            ret = self.update_bounds_parallel_general(pre_lbs, pre_ubs, split, slopes, early_stop=False,
                                            history=history, split_history=split_history, 
                                            intermediate_betas=intermediate_betas, layer_set_bound=layer_set_bound)

        # if get_upper_bound and single_node_split, primals have p and z values; otherwise None
        lower_bounds, upper_bounds, lAs, slopes, betas, split_history, best_intermediate_betas, primals = ret

        beta_crown_lbs = [i[-1].item() for i in lower_bounds]
        beta_time = time.time()-start

        if lp_test == "LP_intermediate_refine":
            refine_time = time.time()
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                init_lp_glb0, refined_lp_glb0 = self.update_the_model_lp(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                init_lp_glb1, refined_lp_glb1 = self.update_the_model_lp(lower_bounds[bdi + total_batch],
                                         upper_bounds[bdi + total_batch], bd[0], choice=0)
                print("############ bound tightness summary ##############")
                print(f"init opt crown: {pre_lbs[-1][-1].item()}")
                print("beta crown for split:", beta_crown_lbs)
                print(f"init lp for split: [{init_lp_glb0}, {init_lp_glb1}]")
                print(f"lp intermediate refined for split: [{refined_lp_glb0}, {refined_lp_glb1}]")
                print("lp_refine time:", time.time() - refine_time, "beta crown time:", beta_time)
                exit()

        elif lp_test == "MIP_intermediate_refine":
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                self.update_the_model_mip(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                self.update_the_model_mip(lower_bounds[bdi + total_batch],
                                          upper_bounds[bdi + total_batch], bd[0], choice=0)

        end = time.time()
        print('batch bounding time: ', end - start)

        return [i[-1].item() for i in upper_bounds], [i[-1].item() for i in lower_bounds], None, lAs, lower_bounds, \
               upper_bounds, slopes, split_history, betas, best_intermediate_betas, primals


    def get_relu(self, model, idx):
        # find the i-th ReLU layer
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer


    """Trasfer all necessary tensors to CPU in a batch."""
    def transfer_to_cpu(self, net, non_blocking=True, opt_intermediate_beta=False):
        # Create a data structure holding all the tensors we need to transfer.
        cpu_net = lambda : None
        cpu_net.relus = [None] * len (net.relus)
        for i in range(len(cpu_net.relus)):
            cpu_net.relus[i] = lambda : None
            cpu_net.relus[i].inputs = [lambda : None]
            cpu_net.relus[i].name = net.relus[i].name

        # Transfer data structures for each relu.
        # For get_candidate_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # For get_candidate_parallel.
            cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
            cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)
        # For get_lA_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
        # For get_slope().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # Per-neuron alpha.
            cpu_layer.alpha = OrderedDict()
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha[spec_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)
        # For get_beta().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            if layer.sparse_beta is not None:
                cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)
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

        return cpu_net


    def get_primal_upper_bound(self, A):
        with torch.no_grad():
            assert self.x.ptb.norm == np.inf, print('we only support to get primals for Linf norm perturbation so far')
            input_A_lower = A[self.net.output_name[0]][self.net.input_name[0]]["lA"]
            batch = input_A_lower.shape[0]

            x_lb, x_ub, eps = self.x.ptb.x_L, self.x.ptb.x_U, self.x.ptb.eps
            x_lb = x_lb.repeat(batch, 1, 1, 1)
            x_ub = x_ub.repeat(batch, 1, 1, 1)
            input_primal = x_lb.clone().detach()
            input_primal[input_A_lower.squeeze(1) < 0] = x_ub[input_A_lower.squeeze(1) < 0]

        return input_primal, self.net(input_primal, clear_forward_only=True).matmul(self.c[0].transpose(-1, -2))


    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        lower_bounds = []
        upper_bounds = []
        self.pre_relu_indices = []
        i = 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {}

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())
            self.name_dict[i] = layer.inputs[0].name
            self.pre_relu_indices.append(i)
            i += 1

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(1, -1).detach())
        upper_bounds.append(ub.view(1, -1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices


    def get_candidate_parallel(self, model, lb, ub, batch, diving_batch=0):
        # get the intermediate bounds in the current model
        lower_bounds = []
        upper_bounds = []

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower)
            upper_bounds.append(layer.inputs[0].upper)

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch + diving_batch, -1).detach())
        upper_bounds.append(ub.view(batch + diving_batch, -1).detach())

        return lower_bounds, upper_bounds


    def get_mask_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.relus:
            # 1 is unstable neuron, 0 is stable neuron.
            mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float()
            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            if this_relu.lA is not None:
                lA.append(this_relu.lA.squeeze(0))
            else:
                # It might be skipped due to inactive neurons.
                lA.append(None)

        ret_mask, ret_lA = [], []
        for i in range(mask[0].size(0)):
            ret_mask.append([j[i:i+1] for j in mask])
            ret_lA.append([j[i:i+1] if j is not None else None for j in lA])
        return ret_mask, ret_lA


    def get_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None]
        # get lower A matrix of ReLU
        lA = []
        for this_relu in model.relus:
            lA.append(this_relu.lA.squeeze(0))

        ret_lA = []
        for i in range(lA[0].size(0)):
            ret_lA.append([j[i:i+1] for j in lA])
        return ret_lA


    def get_beta(self, model, splits_per_example, diving_batch=0):
        # split_per_example only has half of the examples.
        batch = splits_per_example.size(0) - diving_batch
        retb = [[] for i in range(batch * 2 + diving_batch)]
        for mi, m in enumerate(model.relus):
            for i in range(batch):
                # Save only used beta, discard padding beta.
                retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                retb[i + batch].append(m.sparse_beta[i + batch, :splits_per_example[i, mi]])
            for i in range(diving_batch):
                retb[2 * batch + i].append(m.sparse_beta[2 * batch + i, :splits_per_example[batch + i, mi]])
        return retb


    def get_slope(self, model):
        if len(model.relus) == 0:
            return [None]

        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.
        batch_size = next(iter(model.relus[0].alpha.values())).size(2)
        ret = [defaultdict(dict) for i in range(batch_size)]
        for m in model.relus:
            for spec_name, alpha in m.alpha.items():
                # print(f'save layer {m.name} start_node {spec_name} shape {alpha.size()} norm {alpha.abs().sum()}')
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:]
        return ret


    def set_slope(self, model, slope, intermediate_refinement_layers=None, diving_batch=0):
        cleanup_intermediate_slope = isinstance(intermediate_refinement_layers, list) and len(intermediate_refinement_layers) == 0
        if cleanup_intermediate_slope:
            # Clean all intermediate betas if we are not going to refine intermeidate layer neurons anymore.
            del model.best_intermediate_betas
            for m in model.relus:
                if hasattr(m, 'single_intermediate_betas'):
                    print(f'deleting single_intermediate_betas for {m.name}')
                    del m.single_intermediate_betas
                if hasattr(m, 'history_intermediate_betas'):
                    print(f'deleting history_intermediate_betas for {m.name}')
                    del m.history_intermediate_betas
                if hasattr(m, 'split_intermediate_betas'):
                    print(f'deleting split_intermediate_betas for {m.name}')
                    del m.split_intermediate_betas

        if type(slope) == list:
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[0][m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
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
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
                            # create diving alpha
                            diving_alpha = slope[m.name][spec_name]
                            assert diving_batch == diving_alpha.shape[2]
                            m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                            # else:
                            #     m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                            del diving_alpha
                        # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        else:
            raise NotImplementedError


    def reset_beta(self, model, batch, max_splits_per_layer=None, betas=None, diving_batch=0):
        # Recreate new beta with appropriate shape.
        for mi, m in enumerate(self.net.relus):
            # Create only the non-zero beta. For each layer, it is padded to maximal length.
            # We create tensors on CPU first, and they will be transferred to GPU after initialized.
            m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
            m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            # Load beta from history.
            # for bi in range(len(betas)):
            for bi in range(batch):
                if betas[bi] is not None:
                    # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                    valid_betas = len(betas[bi][mi])
                    m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
            # This is the beta variable to be optimized for this layer.
            m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()
            
            assert batch + diving_batch == len(betas)
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
    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None, beta=None, betas=None,
                        early_stop=True, history=None, layer_set_bound=True, shortcut=False):
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time

        if beta is None:
            beta = arguments.Config["solver"]["beta-crown"]["beta"] # might need to set beta False in FSB node selection
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]["lr_alpha"]
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]

        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0
        # update optimize-CROWN bounds in a parallel way

        # if history is None:
        #     history = []
        
        diving_batch = 0
        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)
        # print(split, "diving" in split)
        if "diving" in split:
            diving_batch = split["diving"]
            print(f"regular batch size: 2*{batch}, diving batch size 1*{diving_batch}")
            # betas and history: regular batch + diving batch number of constraints
            # new_sparse_betas: (2 * regular batch + diving batch number of constraints, max splits per layer)
            # print("betas", len(betas), betas)
            # print("history", len(history), history)
            # history = [[[[], []], [[], []], [[], []]], [[[], []], [[], []], [[], []]]]
            # betas = [None, None]

        # initial results with empty list
        ret_l = [[] for _ in range(batch * 2 + diving_batch)]
        ret_u = [[] for _ in range(batch * 2 + diving_batch)]
        ret_s = [[] for _ in range(batch * 2 + diving_batch)]
        ret_b = [[] for _ in range(batch * 2 + diving_batch)]
        new_split_history = [{} for _ in range(batch * 2 + diving_batch)]
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2 + diving_batch)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        start_prepare_time = time.time()
        # iteratively change upper and lower bound from former to later layer

        if beta:
            # count how many split nodes in each batch example (batch, num of layers)
            splits_per_example = torch.zeros(size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu', requires_grad=False)
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

                # import pdb; pdb.set_trace()
                splits_per_example = torch.cat([splits_per_example, diving_splits_per_example], dim=0)
                max_splits_per_layer = splits_per_example.max(dim=0)[0]
                del diving_splits_per_example

            # Create and load warmup beta.
            self.reset_beta(self.net, batch, betas=betas, max_splits_per_layer=max_splits_per_layer, diving_batch=diving_batch)  # warm start beta

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
        else:
            for m in self.net.relus:
                m.beta = None

        # if diving_batch > 0: import pdb; pdb.set_trace()

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

            # Only the last element is used later.
            pre_lb_last = torch.cat([pre_lb_all[-1][:batch], pre_lb_all[-1][:batch], pre_lb_all[-1][batch:]])
            pre_ub_last = torch.cat([pre_ub_all[-1][:batch], pre_ub_all[-1][:batch], pre_ub_all[-1][batch:]])

            new_candidate = {}
            for d in range(len(lower_bounds)):
                # for each layer except the last output layer
                if len(zero_indices_batch[d]):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    lower_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                    upper_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                new_candidate[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]

        # create new_x here since batch may change
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                 x_L=self.x.ptb.x_L.repeat(batch * 2 + diving_batch, 1, 1, 1),
                                 x_U=self.x.ptb.x_U.repeat(batch * 2 + diving_batch, 1, 1, 1))
        new_x = BoundedTensor(self.x.data.repeat(batch * 2 + diving_batch, 1, 1, 1), ptb)
        c = None if self.c is None else self.c.repeat(new_x.shape[0], 1, 1)
        # self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes, diving_batch=diving_batch)

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': beta, 'ob_single_node_split': True,
                'ob_update_by_layer': layer_set_bound, 'ob_optimizer':optimizer}})
            with torch.no_grad():
                lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='backward',
                                                 new_interval=new_candidate, bound_upper=False, return_A=False)
            return lb

        return_A = True if get_upper_bound else False  # we need A matrix to consturct adv example
        # FIXME (assigned to Shiqi): Please remove most parameters of this function, and use arguments.Config instead. Also change other places of this file.
        if layer_set_bound:
            start_beta_bound_time = time.time()
            self.net.set_bound_opts({'optimize_bound_args':
                                         {'ob_beta': beta, 'ob_single_node_split': True,
                                          'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                          'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                                          'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            # if diving_batch != 0: import pdb; pdb.set_trace()
            tmp_ret = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=return_A, bound_upper=False, needed_A_dict=self.needed_A_dict)
            beta_bound_time += time.time() - start_beta_bound_time

            # we don't care about the upper bound of the last layer

        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                    'ob_iteration': iteration, 'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                    'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            tmp_ret = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=return_A, bound_upper=False, needed_A_dict=self.needed_A_dict)

        if get_upper_bound:
            lb, _, A = tmp_ret
            primal_x, ub = self.get_primal_upper_bound(A)
        else:
            lb, _ = tmp_ret
            ub = lb + 99  # dummy upper bound
            primal_x = None

        bound_time += time.time() - start_bound_time

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()

            lAs = self.get_lA_parallel(transfer_net)
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
            # reshape the results based on batch.
            for i in range(batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]

                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]
            for i in range(2 * batch, 2 * batch + diving_batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]

            finalize_time = time.time() - start_finalize_time

        ret_p, primals = None, None
        if get_upper_bound:
            # print("opt crown:", lb)
            # primal_values, integer_primals = self.get_neuron_primal(primal_x, lb=lower_bounds_new, ub=upper_bounds_new)
            # correct intermediate primal should produce the correct primal output lb
            # print("primal lb with beta:", primal_values[-1])
            # print("### Extracting primal values and mixed integers with beta for intermeidate nodes done ###")
            # exit()
            # primals = {"p": primal_values, "z": integer_primals}
            pass

        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')

        # if primals is not None: ret_p = self.layer_wise_primals(primals)

        # assert (ret_p[1]['p'][0][0] == primal_x[1]).all()
        return ret_l, ret_u, lAs, ret_s, ret_b, new_split_history, best_intermediate_betas, primal_x


    def update_bounds_parallel_general(self, pre_lb_all=None, pre_ub_all=None, split=None, 
                            slopes=None, early_stop=True, split_history=None, history=None, 
                            intermediate_betas=None, layer_set_bound=True, debug=False):

        beta = arguments.Config["solver"]["beta-crown"]["beta"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]["lr_alpha"]
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        lr_intermediate_beta = arguments.Config["solver"]["intermediate_refinement"]["lr"]
        beta_warmup = arguments.Config["solver"]["beta-crown"]["beta_warmup"]
        opt_coeffs = arguments.Config["solver"]["intermediate_refinement"]["opt_coeffs"]
        opt_bias = arguments.Config["solver"]["intermediate_refinement"]["opt_bias"]
        opt_intermediate_beta = arguments.Config["solver"]["intermediate_refinement"]["enabled"]
        intermediate_refinement_layers = arguments.Config["solver"]["intermediate_refinement"]["layers"]
        
        if not hasattr(self, '_count'):  # for debugging.
            self._count = 0
        self._count += 1
        if self._count < 1:
            opt_intermediate_beta = False
            layer_set_bound = True
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time
        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0

        if split_history is None:
            split_history = []
        if history is None:
            history = []
        device = self.net.device

        # update optimize-CROWN bounds in a parallel way
        batch = len(split["decision"])

        # if any node in a layer is involved, it counts one constraint in this layer
        num_constr = [[0 for _ in range(2 * batch)] for _ in range(len(self.net.relus))]

        # A dictionary for converting node name to index.
        relu_node2idx = {}
        for i, m in enumerate(self.net.relus):
            relu_node2idx[m.name] = i

        # layers_need_change: keeps the smallest layer that involved in each batch
        # list (total_batch)-> earliest layer node of each batch
        layers_need_change = [np.inf] * batch
        for bi, bd in enumerate(split["decision"]):
            for node in bd:
                if num_constr[node[0]][bi] == 0:
                    num_constr[node[0]][bi] = 1
                    num_constr[node[0]][bi + batch] = 1
                if node[0] < layers_need_change[bi]:
                    layers_need_change[bi] = node[0]

        # print("@@num_constr", num_constr)
        # initial results with empty list
        ret_l = [[] for _ in range(batch * 2)]
        ret_u = [[] for _ in range(batch * 2)]
        ret_s = [[] for _ in range(batch * 2)]
        betas = [[] for _ in range(batch * 2)]  # Not actually used, since beta values are stored in the history.
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        if debug:
            if split_history[0]:
                for shi, sh in enumerate(split_history):
                    print(
                        f"##################################split_history batch {shi}##################################")
                    print("beta", sh["beta"])
                    print("c", sh["c"])
                    print("coeffs", sh["coeffs"])
                    print("bias", sh["bias"])
                    print("single_beta", sh["single_beta"])
            print("history", history)

        # new_split_history: store beta, c, coeffs tensors in this branch such that children can use
        # list (total_batch*2)->list (relu layers)->the beta/c/coeffs tensors for each layer (None if not used)
        new_split_history = [{"beta": [None for _ in range(len(self.net.relus))],
                              "c": [None for _ in range(len(self.net.relus))],
                              "coeffs": [None for _ in range(len(self.net.relus))],
                              "bias": [None for _ in range(len(self.net.relus))],
                              "single_beta": [None for _ in range(len(self.net.relus))]}
                             for _ in range(batch * 2)]

        start_prepare_time = time.time()
        # collect all the variables that need to be optimized here
        self.net.beta_params = []
        self.net.single_beta_params = []
        self.net.single_beta_mask = []
        if opt_coeffs:
            self.net.coeffs_params = []
            self.net.split_dense_coeffs_params = []
        if opt_bias: self.net.bias_params = []

        # reset the beta information of each layer
        for m in self.net.relus:
            # split ones are the new split constraint from split["deicsion"]: needs to optimize split_beta, split_coeffs, split_bias
            m.split_beta = [None for _ in range(2 * batch)]
            m.split_c = [0 for _ in range(2 * batch)]
            # split coeffs support either dense matrix or nonzero index paired with value
            m.split_coeffs = {"dense": None, "nonzero": [], "coeffs": []}
            m.split_bias = [None for _ in range(2 * batch)]

            # hisotry constraints: Only optimize history beta. All the others are copies from split_history
            m.history_beta = [None for _ in range(2 * batch)]
            m.history_c = [None for _ in range(2 * batch)]
            m.history_coeffs = [None for _ in range(2 * batch)]
            m.history_bias = [None for _ in range(2 * batch)]

            # masked_beta (batch, m.flattened_nodes) is the eventual coeffs mm (beta*c)
            m.masked_beta = None
            # m.split_beta_used: True if any of this layer node is used in the new split constraints
            m.split_beta_used = False
            # m.history_beta_used: True if any of this layer node is used in the history constraints
            m.history_beta_used = False

            # if any of the current/history split constraints are single node constraint, 
            # we save and optimize beta in m.beta and c in m.beta_mask
            m.single_beta_used = False
            m.beta = torch.zeros(2*batch, m.flattened_nodes, device=device)
            m.beta_mask = torch.zeros(2*batch, m.flattened_nodes, device=device)
            # The non-zero element position for single node split.
            m._single_beta_loc = [[] for _ in range(2*batch)]
            # The coefficients for non-zero element position for single node split.
            m._single_beta_sign = [[] for _ in range(2*batch)]

        ######################################### Collect the split and history constraints ########################################
        for lbi in range(batch):
            if len(split["decision"][lbi])==1:
                # this is a single node split for batch lbi, only assign beta and beta_mask
                node = split["decision"][lbi][0]
                m = self.net.relus[node[0]]
                m.beta_mask[lbi, node[1]] = 1
                m.beta_mask[lbi+batch, node[1]] = -1
                # Also save the location and split sign for later use.
                m._single_beta_loc[lbi].append(node[1])
                m._single_beta_loc[lbi+batch].append(node[1])
                m._single_beta_sign[lbi].append(1.0)
                m._single_beta_sign[lbi+batch].append(-1.0)
                m.single_beta_used = True
                # print(f'example {lbi} split {m.name} {m._single_beta_loc[lbi]}')
            else:
                # lbi is the index of small batch for assigning beta/c/coeffs for current split constraint
                for di in range(len(split["decision"][lbi])):
                    # index di iterates the nodes in the new split constraint of each batch
                    # tmp_d[lbi] map lbi idx to the total large batch
                    node = split["decision"][lbi][di]
                    coeff = split["coeffs"][lbi][di]
                    m = self.net.relus[node[0]]
                    m.split_beta_used = True
                    # need to assign coeffs value to sparse_coeffs twicie
                    m.split_coeffs["nonzero"].append([lbi, node[1]])
                    m.split_coeffs["coeffs"].append(coeff)
                    # m.split_c 1 means this constraint>0, 0 means not used in this layer, -1 means<0
                    m.split_c[lbi] = 1
                    m.split_c[lbi + batch] = -1

            if split_history[0]:
                # now we handle history constraints for each batch lbi
                # only the first split will have [[]] split history and not go into this if branch
                for lidx, m in enumerate(self.net.relus):
                    # lidx is the index of relu layers since history constraints could involve any layer nodes
                    beta_idx = split_history[lbi]["beta"][lidx]
                    c_idx = split_history[lbi]["c"][lidx]
                    coeffs_idx = split_history[lbi]["coeffs"][lidx]
                    single_beta_idx = split_history[lbi]["single_beta"][lidx]

                    if single_beta_idx is not None:
                        m.single_beta_used = True
                        nonzero_index = single_beta_idx["nonzero"]
                        m.beta_mask[lbi][nonzero_index] = m.beta_mask[lbi][nonzero_index] + single_beta_idx["c"]
                        m.beta_mask[lbi+batch][nonzero_index] = m.beta_mask[lbi+batch][nonzero_index] + single_beta_idx["c"]
                        m.beta[lbi][nonzero_index] = m.beta[lbi][nonzero_index] + single_beta_idx["value"]
                        m.beta[lbi+batch][nonzero_index] = m.beta[lbi+batch][nonzero_index] + single_beta_idx["value"]
                        # Also save the location and split sign for later use.
                        # Always put the current split to the last.
                        # print(f'example {lbi} history {m.name} {m._single_beta_loc[lbi]} {nonzero_index.squeeze(1).cpu().numpy().tolist()}')
                        m._single_beta_loc[lbi] = nonzero_index.squeeze(1).cpu().numpy().tolist() + m._single_beta_loc[lbi]
                        m._single_beta_loc[lbi+batch] = nonzero_index.squeeze(1).cpu().numpy().tolist() + m._single_beta_loc[lbi+batch]
                        m._single_beta_sign[lbi] = single_beta_idx["c"].squeeze(1).cpu().numpy().tolist() + m._single_beta_sign[lbi]
                        m._single_beta_sign[lbi+batch] = single_beta_idx["c"].squeeze(1).cpu().numpy().tolist() + m._single_beta_sign[lbi+batch]

                    if beta_idx is not None:
                        # it means batch lbi layer lidx has history general splits
                        m.history_beta_used = True

                        if beta_warmup:
                            m.history_beta[lbi] = beta_idx.detach().clone()
                            m.history_beta[lbi + batch] = beta_idx.detach().clone()
                        else:
                            m.history_beta[lbi] = beta_idx.detach().clone().zero_()
                            m.history_beta[lbi + batch] = beta_idx.detach().clone().zero_()

                        m.history_c[lbi] = c_idx.detach().clone()
                        m.history_c[lbi + batch] = c_idx.detach().clone()
                        m.history_c[lbi].requires_grad = False
                        m.history_c[lbi + batch].requires_grad = False

                        m.history_coeffs[lbi] = {"nonzero": coeffs_idx["nonzero"],
                                                 "coeffs": coeffs_idx["coeffs"].detach().clone()}
                        m.history_coeffs[lbi + batch] = {"nonzero": coeffs_idx["nonzero"],
                                                         "coeffs": coeffs_idx["coeffs"].detach().clone()}
                        m.history_coeffs[lbi]["coeffs"].requires_grad = False
                        m.history_coeffs[lbi + batch]["coeffs"].requires_grad = False

                        if opt_bias:
                            bias_idx = split_history[lbi]["bias"][lidx]
                            m.history_bias[lbi] = bias_idx.detach().clone()
                            m.history_bias[lbi + batch] = bias_idx.detach().clone()
                            m.history_bias[lbi].requires_grad = False
                            m.history_bias[lbi + batch].requires_grad = False

        ######################################### Process split and history constraints to be sparse matrix ########################################
        # m.split_c (2*batch, 1): 1 means this constraint>0, 0 means not used in this layer, -1 means<0; optimization: False
        # m.split_beta (2*batch, 1): init to be 0, beta for each constraint; optimization: True
        # m.split_bias (2*batch, 1): init to be 0; optimization: True
        # m.split_coeffs["dense"] (batch, m.flattened_nodes): the dense matrix for the new constraint; optimzation: True
        # m.split_coeffs["nonzero"] (# nonzero nodes, 2) ([batch index, node index]): the first batch index, using to assign value to sparse matrix m.new_split_coeffs; optimization: False
        # m.split_coeffs["coeffs"] (# nonzero nodes): the coeffs value; optimization: opt_coeffs
        # m.new_split_coeffs (2*batch, self.flattened_nodes): the sparse matrix of coeffs
        # m.bias (batch, 1): constraint + bias</>=0, reuse for the first and rest half batch; optimization: opt_bias
        # m.history_c (2*batch->[# constraints in each batch]): history c in each batch
        # m.history_beta (2*batch->[# constraints in each batch]): history beta in each batch
        # m.new_history_c (2*batch, max_nbeta): sparse matrix for c for all batches; optimization: False
        # m.new_history_beta (2*batch, max_nbeta): sparse matrix for beta for all batches optimization: True
        # m.history_coeffs["nonzero"] (2*batch->(# nonzero nodes, 2)) ([constraint index, node index]): the nonzero coeffs index in each batch
        # m.history_coeffs["coeffs"] (2*batch->(# nonzero nodes)): the coeffs value in each batch
        # m.new_history_coeffs (2*batch, m.flattened_node, max_nbeta): sparse matrix for coeffs; optimization: False
        # m.history_bias (2*batch->[# constraints in each batch]): history bias in each batch
        # m.new_history_bias (2*batch, max_nbeta): sparse matrix for bias for all batches; optimization: False

        for lidx, m in enumerate(self.net.relus):
            if m.single_beta_used:
                m.beta = m.beta.detach().requires_grad_(True)
                self.net.single_beta_params.append(m.beta)
                self.net.single_beta_mask.append(m.beta_mask)
                # Convert single_beta_loc and single_beta_sign to tensors.
                m.max_single_split = max([len(a) for a in m._single_beta_loc]) 
                m.single_beta_loc = torch.zeros(size=(2 * batch, m.max_single_split), dtype=torch.int64, device=device, requires_grad=False)
                m.single_beta_sign = torch.zeros(size=(2 * batch, m.max_single_split), dtype=torch.get_default_dtype(), device=device, requires_grad=True)
                for split_index, (beta_loc, beta_sign) in enumerate(zip(m._single_beta_loc, m._single_beta_sign)):
                    m.single_beta_loc[split_index].data[:len(beta_loc)] = torch.tensor(beta_loc, dtype=torch.int64, device=device)
                    m.single_beta_sign[split_index].data[:len(beta_sign)] = torch.tensor(beta_sign, dtype=torch.get_default_dtype(), device=device)  # Unassigned is 0.
                if m.max_single_split == 0:
                    m.single_beta_used = False

            if m.split_beta_used:
                ####### sparse coeffs and new_beta for split constraints #######
                m.split_c = torch.tensor(m.split_c, dtype=torch.get_default_dtype(), device=device,
                                         requires_grad=False).unsqueeze(-1)
                if m.split_c.abs().sum() > 0:
                    # there are nodes used the new split constraint in this layer
                    m.split_beta = torch.zeros(m.split_c.shape, dtype=torch.get_default_dtype(), device=device)
                    m.split_beta.requires_grad = True
                    self.net.beta_params.append(m.split_beta)

                    m.split_coeffs["nonzero"] = torch.tensor(m.split_coeffs["nonzero"], dtype=torch.long,
                                                             device=device, requires_grad=False)
                    m.split_coeffs["coeffs"] = torch.tensor(m.split_coeffs["coeffs"], dtype=torch.get_default_dtype(),
                                                            device=device)
                    # construct the dense matrix for the split coeffs
                    m.split_coeffs["dense"] = torch.zeros((batch, m.flattened_nodes), dtype=torch.get_default_dtype(),
                                                          device=device)
                    m.split_coeffs["dense"][(m.split_coeffs["nonzero"][:, 0], m.split_coeffs["nonzero"][:, 1])] = \
                    m.split_coeffs["coeffs"]
                    m.split_coeffs["dense"] = m.split_coeffs["dense"].detach()
                    dense_mask = torch.zeros((batch, m.flattened_nodes), dtype=torch.bool, device=device,
                                             requires_grad=False)
                    dense_mask[(m.split_coeffs["nonzero"][:, 0], m.split_coeffs["nonzero"][:, 1])] = True

                    if opt_coeffs:
                        # m.split_coeffs["coeffs"].requires_grad=True
                        # self.net.coeffs_params.append(m.split_coeffs["coeffs"])
                        m.split_coeffs["dense"].requires_grad = True
                        self.net.split_dense_coeffs_params.append(
                            {"dense": m.split_coeffs["dense"], "mask": dense_mask})
                    # coeffs_nonzero = (m.split_coeffs["nonzero"][:,0], m.split_coeffs["nonzero"][:,1])

                    if opt_bias:
                        m.split_bias = torch.zeros((batch, 1), dtype=torch.get_default_dtype(), device=device)
                        m.split_bias.requires_grad = True
                        self.net.bias_params.append(m.split_bias)

            if m.history_beta_used:
                ####### sparse_coeffs and new_beta for history constraints #######
                # Rebuild data-structure
                m.max_nbeta = 0  # max number of beta constraints in this batch (since each example in this batch can have different number of betas).
                num_elements = 0  # total number of coefficients in this batch.
                for batch_i in range(len(m.history_beta)):
                    if m.history_beta[batch_i] is not None:
                        m.max_nbeta = max(m.max_nbeta, m.history_beta[batch_i].size(1))
                        num_elements += m.history_coeffs[batch_i]["coeffs"].size(0)
                if m.max_nbeta == 0:
                    m.new_history_beta, m.new_history_coeffs = None, None
                    continue

                # We want create a coeffient tensor in size (batch, self.flattened_nodes, m.max_nbeta).
                # Since we know exactly how many elements are there in this sparse matrix, we pre-allocate entire indices arrays,
                # avoiding creating a lot of small (1,1) tensors and avoiding using the low-efficient torch.cat().
                # Do not hardcode device; our code needs to run on CPUs as well.
                batch_indices = torch.empty(size=(num_elements,), dtype=torch.long, device=device,
                                            requires_grad=False)
                node_indices = torch.empty(size=(num_elements,), dtype=torch.long, device=device,
                                           requires_grad=False)
                beta_indices = torch.empty(size=(num_elements,), dtype=torch.long, device=device,
                                           requires_grad=False)
                # In fact, we do not need gradient for history coefficients. We only need gradients for the last set of coefficients, which can be handled separatedly.
                # The sparse bmm() function does not support gradient to the sparse array, so we cannot obtain their gradients.
                values = torch.empty(size=(num_elements,), device=device, requires_grad=False)
                # Create a new beta tensor, with size (batch, m.max_nbeta).
                m.new_history_beta = torch.zeros(size=(len(m.history_beta), m.max_nbeta), device=device)
                m.new_history_c = torch.zeros(size=(len(m.history_c), m.max_nbeta), device=device)
                if opt_bias: m.new_history_bias = torch.zeros(size=(len(m.history_c), m.max_nbeta), device=device)
                index = 0
                for batch_i in range(len(m.history_beta)):
                    if m.history_beta[batch_i] is None:
                        continue
                    coeffs_indices = m.history_coeffs[batch_i][
                        "nonzero"]  # If you need torch.cat in the final code, make sure it is in update_bounds_parallel(), not here!
                    n_coeffs = coeffs_indices.size(0)  # number of coefficents for this batch element.
                    # which beta is this? e.g., first beta, second beta, etc.
                    beta_indices[index:index + n_coeffs] = coeffs_indices[:, 0].detach()
                    # insert the relu node indices for this split constraint.
                    node_indices[index:index + n_coeffs] = coeffs_indices[:, 1].detach()
                    # Set the batch indices to the batch ID.
                    batch_indices[index:index + n_coeffs] = batch_i
                    # The values of coefficients over all beta of this element.
                    values[index:index + n_coeffs] = m.history_coeffs[batch_i]["coeffs"]
                    # Move to the next elements.
                    index += n_coeffs
                    m.new_history_beta[batch_i][:m.history_beta[batch_i].size(1)] = m.history_beta[batch_i].squeeze(
                        0)
                    m.new_history_c[batch_i][:m.history_c[batch_i].size(1)] = m.history_c[batch_i].squeeze(0)
                    if opt_bias: m.new_history_bias[batch_i][:m.history_bias[batch_i].size(1)] = m.history_bias[
                        batch_i].squeeze(0)
                # we need the gradients for all the betas
                m.new_history_beta = m.new_history_beta.detach()
                m.new_history_beta.requires_grad = True
                self.net.beta_params.append(m.new_history_beta)
                # We don't need gradient for these coefficients.
                m.new_history_coeffs = torch.sparse_coo_tensor(
                    torch.stack([batch_indices, node_indices, beta_indices]), values,
                    (len(m.history_beta), m.flattened_nodes, m.max_nbeta), requires_grad=False, device=device)
                m.new_history_coeffs = m.new_history_coeffs.coalesce().to_dense()

        if debug:
            for mi, m in enumerate(self.net.relus):
                print(f"##################################layer{mi}##################################")
                print("split_beta", m.split_beta)
                print("split_c", m.split_c)
                print("split_coeffs", m.split_coeffs)
                print("split_bias", m.split_bias)
                print("history_beta", m.history_beta)
                print("history_c", m.history_c)
                print("history_coeffs", m.history_coeffs)
                print("history_bias", m.history_bias)
                print("single_beta", m.beta)
                print("single_beta_mask", m.beta_mask)

        ######################################### Process done, compute bounds! ########################################
        # idx is the index of relu layers, change_idx is the index of all layers

        with torch.no_grad():
            upper_bounds = [i.clone() for i in pre_ub_all[:-1]]
            lower_bounds = [i.clone() for i in pre_lb_all[:-1]]

            upper_bounds_cp = copy.deepcopy(upper_bounds)
            lower_bounds_cp = copy.deepcopy(lower_bounds)

            for i in range(len(lower_bounds)):
                if not lower_bounds[i].is_contiguous():
                    upper_bounds[i] = upper_bounds[i].contiguous()
                    lower_bounds[i] = lower_bounds[i].contiguous()

            for i in range(batch):
                if len(split["decision"][i]) == 1 and not opt_bias:
                    d, idx = split["decision"][i][0][0], split["decision"][i][0][1]
                    upper_bounds[d].view(batch, -1)[i][idx] = 0.0
                    lower_bounds[d].view(batch, -1)[i][idx] = 0.0

            pre_lb_all = [torch.cat(2 * [i]) for i in pre_lb_all]
            pre_ub_all = [torch.cat(2 * [i]) for i in pre_ub_all]

            # merge the inactive and active splits together
            new_candidate = {}
            for i, (l, uc, lc, u) in enumerate(zip(lower_bounds, upper_bounds_cp, lower_bounds_cp, upper_bounds)):
                # we set lower = 0 in first half batch, and upper = 0 in second half batch
                new_candidate[self.name_dict[i]] = [torch.cat((l, lc), dim=0), torch.cat((uc, u), dim=0)]

        # if not layer_set_bound:
        #     new_candidate_p = {}
        #     for i, (l, u) in enumerate(zip(pre_lb_all[:-1], pre_ub_all[:-1])):
        #         # we set lower = 0 in first half batch, and upper = 0 in second half batch
        #         new_candidate_p[self.name_dict[i]] = [l, u]

        # create new_x here since batch may change
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                 x_L=self.x.ptb.x_L.repeat(batch * 2, 1, 1, 1),
                                 x_U=self.x.ptb.x_U.repeat(batch * 2, 1, 1, 1))
        new_x = BoundedTensor(self.x.data.repeat(batch * 2, 1, 1, 1), ptb)
        self.net(new_x)  # batch may change, so we need to do forward to set some shapes here
        c = None if self.c is None else self.c.repeat(new_x.shape[0], 1, 1)

        if len(slopes) > 0:
            # set slope here again
            print(f'calling with {intermediate_refinement_layers}')
            self.set_slope(self.net, slopes, intermediate_refinement_layers=intermediate_refinement_layers)

        """
        for ii, example in enumerate(intermediate_betas):
            if example is not None:
                for kk in example.keys():
                    for kkk in example[kk].keys():
                        print(f'example intermediate_betas {ii} {kk} {kkk}')
            else:
                print(f'skipe example intermediate_betas {ii}')
        """

        if opt_intermediate_beta and intermediate_betas is not None:
            # selected_intermediate_betas = [intermediate_betas[i] for i in tmp_d]
            # Set it as the initial. Dupllicate for the second half of the batch.
            if len(intermediate_refinement_layers) == 0:
                del intermediate_betas  # Free GPU memory.
                self.net.init_intermediate_betas = None
            else:
                self.net.init_intermediate_betas = intermediate_betas + intermediate_betas

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if layer_set_bound and not opt_intermediate_beta:
            start_beta_bound_time = time.time()
            self.net.set_bound_opts({'optimize_bound_args':
                                         {'ob_beta': beta, 'ob_single_node_split': False,
                                          'ob_opt_coeffs': opt_coeffs, 'ob_opt_bias': opt_bias,
                                          'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                          'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta, 'ob_lr_intermediate_beta': lr_intermediate_beta,
                                          'ob_optimizer': optimizer}})
            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=False, bound_upper=False)
            beta_bound_time += time.time() - start_beta_bound_time

        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                                         'ob_iteration': iteration, 'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta,
                                         'ob_lr_intermediate_beta': lr_intermediate_beta,
                                         'ob_opt_coeffs': opt_coeffs, 'ob_opt_bias': opt_bias,
                                         'ob_single_node_split': False, 'ob_intermediate_beta': opt_intermediate_beta,
                                         'ob_intermediate_refinement_layers': intermediate_refinement_layers,
                                         'ob_optimizer': optimizer}})
            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=c, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=False, bound_upper=False)

        bound_time += time.time() - start_bound_time

        # print('best results of parent nodes', pre_lb_all[-1].repeat(2, 1))
        # print('finally, after optimization:', lower_bounds_new[-1])

        # Move tensors to CPU for all elements in this batch.
        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb = lb.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False, opt_intermediate_beta=opt_intermediate_beta)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, lb + 99, batch * 2)
            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1].cpu())
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1].cpu())

            lAs = self.get_lA_parallel(transfer_net)

            if len(slopes) > 0:
                ret_s = self.get_slope(transfer_net)

            # reshape the results
            for i in range(batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]

                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]

                # Save the best intermediate betas of this batch.
                if opt_intermediate_beta and self.net.best_intermediate_betas is not None:
                    # In self.net.best_intermediate_betas, relu layer name is the key, corresponds to the split in that layer.
                    # The value of the dict is 'split' or 'history' or 'single'
                    # For each 'split' and 'history' and 'single', there is a dictionary contains intermediates beta for all intermediate layers.
                    # And there are two sets of intermediate betas, one for lb and one for ub.
                    # each value is lb and ub with [batch, *layer_shape, n_splits]
                    # Example: self.net.best_intermediate_betas['/22']['split']['/9']['lb'].
                    # Note that since each batch element can have different number of splits, some splits can be dummy splits. We must skip these splits
                    # when saving best intermediate betas to the domain.
                    for split_layer, all_int_betas_this_layer in transfer_net.best_intermediate_betas.items():
                        if 'single' in all_int_betas_this_layer:
                            assert 'history' not in all_int_betas_this_layer
                            assert 'split' not in all_int_betas_this_layer
                            # Has single node split. Choose the used number of betas.
                            lidx = relu_node2idx[split_layer]
                            this_layer_history = split_history[i]["single_beta"][lidx] if split_history[i] is not None else None
                            # Can be None when there is no split in this layer.
                            n_split_this_layer = len(this_layer_history["nonzero"]) if this_layer_history is not None else 0
                            # The current split, if it is on the same layer.
                            split_node = split["decision"][i][0]
                            if split_node[0] == lidx:
                                # The current split is always the last row.
                                n_split_this_layer += 1
                            if n_split_this_layer > 0:
                                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                                    best_intermediate_betas[i][split_layer][intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i, ..., :n_split_this_layer],
                                        "ub": this_layer_intermediate_betas['ub'][i, ..., :n_split_this_layer],
                                    }
                                    # The other side of the split.
                                    best_intermediate_betas[i + batch][split_layer][
                                        intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i + batch, ..., :n_split_this_layer],
                                        "ub": this_layer_intermediate_betas['ub'][i + batch, ..., :n_split_this_layer],
                                    }
                                    # print(f'example {i} {i+batch} saved {split_layer} {intermediate_layer} with {n_split_this_layer} splits')
                        if 'history' in all_int_betas_this_layer:
                            # Has history split. Choose the used number of betas.
                            lidx = relu_node2idx[split_layer]
                            this_layer_history = split_history[i]["c"][lidx]
                            # Can be None when there is no split in this layer.
                            n_split_this_layer = len(this_layer_history) if this_layer_history is not None else 0
                            if n_split_this_layer > 0:
                                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['history'].items():
                                    best_intermediate_betas[i][split_layer][intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i, ..., :n_split_this_layer],
                                        "ub": this_layer_intermediate_betas['ub'][i, ..., :n_split_this_layer],
                                    }
                                    # The other side of the split.
                                    best_intermediate_betas[i + batch][split_layer][
                                        intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i + batch, ..., :n_split_this_layer],
                                        "ub": this_layer_intermediate_betas['ub'][i + batch, ..., :n_split_this_layer],
                                    }
                        # The currentv split with 1 beta.
                        if 'split' in all_int_betas_this_layer:
                            for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['split'].items():
                                if intermediate_layer in best_intermediate_betas[i][split_layer]:
                                    # Existing betas from history split need to be concatenated.
                                    best_intermediate_betas[i][split_layer][intermediate_layer] = {
                                        "lb": torch.cat((best_intermediate_betas[i][split_layer][
                                                             intermediate_layer]["lb"],
                                                         this_layer_intermediate_betas['lb'][i]), dim=-1),
                                        "ub": torch.cat((best_intermediate_betas[i][split_layer][
                                                             intermediate_layer]["ub"],
                                                         this_layer_intermediate_betas['ub'][i]), dim=-1),
                                    }
                                    # The other side of the split.
                                    best_intermediate_betas[i + batch][split_layer][
                                        intermediate_layer] = {
                                        "lb": torch.cat((
                                                        best_intermediate_betas[i + batch][split_layer][
                                                            intermediate_layer]["lb"],
                                                        this_layer_intermediate_betas['lb'][i + batch]), dim=-1),
                                        "ub": torch.cat((
                                                        best_intermediate_betas[i + batch][split_layer][
                                                            intermediate_layer]["ub"],
                                                        this_layer_intermediate_betas['ub'][i + batch]), dim=-1),
                                    }
                                else:
                                    best_intermediate_betas[i][split_layer][intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i],
                                        "ub": this_layer_intermediate_betas['ub'][i],
                                    }
                                    # The other side of the split.
                                    best_intermediate_betas[i + batch][split_layer][
                                        intermediate_layer] = {
                                        "lb": this_layer_intermediate_betas['lb'][i + batch],
                                        "ub": this_layer_intermediate_betas['ub'][i + batch],
                                    }

        ######################################### save split and history constraints to new_split_history ########################################
        with torch.no_grad():
            for lbi in range(batch):
                for lidx, m in enumerate(self.net.relus):

                    if m.single_beta_used:
                        # save the beta for single split constraints
                        nonzero_index = torch.tensor(m._single_beta_loc[lbi], dtype=torch.int64).unsqueeze(1)
                        new_split_history[lbi]["single_beta"][lidx] = {"nonzero": nonzero_index, "value":m.beta[lbi][nonzero_index], "c": m.beta_mask[lbi][nonzero_index]}
                        new_split_history[lbi+batch]["single_beta"][lidx] = {"nonzero": nonzero_index, "value":m.beta[lbi+batch][nonzero_index], "c": m.beta_mask[lbi+batch][nonzero_index]}

                    if m.history_beta[lbi] is not None and not m.split_beta_used:
                        new_split_history[lbi]["beta"][lidx] = m.history_beta[lbi]
                        new_split_history[lbi]["c"][lidx] = m.history_c[lbi]
                        new_split_history[lbi]["coeffs"][lidx] = m.history_coeffs[lbi]

                        new_split_history[lbi + batch]["beta"][lidx] = m.history_beta[lbi + batch]
                        new_split_history[lbi + batch]["c"][lidx] = m.history_c[lbi + batch]
                        new_split_history[lbi + batch]["coeffs"][lidx] = m.history_coeffs[lbi + batch]

                        if opt_bias:
                            new_split_history[lbi]["bias"][lidx] = m.history_bias[lbi]
                            new_split_history[lbi + batch]["bias"][lidx] = m.history_bias[lbi + batch]

                    elif m.split_beta_used and m.history_beta[lbi] is None:
                        new_split_history[lbi]["beta"][lidx] = m.split_beta[lbi].unsqueeze(0)
                        new_split_history[lbi]["c"][lidx] = m.split_c[lbi].unsqueeze(0)

                        batch_nonzero_index = (m.split_coeffs["nonzero"][:, 0] == lbi)
                        split_coeffs_nonzero = m.split_coeffs["nonzero"][batch_nonzero_index].detach().clone()
                        if m.split_coeffs["dense"] is None:
                            split_coeffs_value = m.split_coeffs["coeffs"][batch_nonzero_index].detach().clone()
                        else:
                            split_coeffs_value = m.split_coeffs["dense"][
                                (split_coeffs_nonzero[:, 0], split_coeffs_nonzero[:, 1])].detach().clone()
                        split_coeffs_nonzero[:, 0] = 0
                        split_coeffs_value.requires_grad = False
                        new_split_history[lbi]["coeffs"][lidx] = {"nonzero": split_coeffs_nonzero,
                                                                         "coeffs": split_coeffs_value}

                        new_split_history[lbi + batch]["beta"][lidx] = m.split_beta[lbi + batch].unsqueeze(0)
                        new_split_history[lbi + batch]["c"][lidx] = m.split_c[lbi + batch].unsqueeze(0)
                        new_split_history[lbi + batch]["coeffs"][lidx] = {"nonzero": split_coeffs_nonzero,
                                                                                       "coeffs": split_coeffs_value}

                        if opt_bias:
                            split_bias = m.split_bias[lbi].detach().clone()
                            split_bias.requires_grad = False
                            new_split_history[lbi]["bias"][lidx] = split_bias.unsqueeze(0)
                            new_split_history[lbi + batch]["bias"][lidx] = split_bias.unsqueeze(0)

                    elif m.split_beta_used and m.history_beta[lbi] is not None:
                        batch_nonzero_index = (m.split_coeffs["nonzero"][:, 0] == lbi)
                        split_coeffs_nonzero = m.split_coeffs["nonzero"][batch_nonzero_index].detach().clone()
                        if m.split_coeffs["dense"] is None:
                            split_coeffs_value = m.split_coeffs["coeffs"][batch_nonzero_index].detach().clone()
                        else:
                            split_coeffs_value = m.split_coeffs["dense"][
                                (split_coeffs_nonzero[:, 0], split_coeffs_nonzero[:, 1])].detach().clone()
                        split_coeffs_value.requires_grad = False
                        # insert the current split constraint before the history split constraints
                        split_coeffs_nonzero[:, 0] = 0
                        history_coeffs_nonzero, history_coeffs_value = m.history_coeffs[lbi][
                                                                           "nonzero"].detach().clone(), \
                                                                       m.history_coeffs[lbi]["coeffs"].detach().clone()
                        # move the current history constraints after the enw split constraint
                        history_coeffs_nonzero[:, 0] = history_coeffs_nonzero[:, 0] + 1

                        new_split_history[lbi]["beta"][lidx] = torch.cat(
                            (m.split_beta[lbi].unsqueeze(0), m.history_beta[lbi]), 1).detach()
                        new_split_history[lbi]["c"][lidx] = torch.cat(
                            (m.split_c[lbi].unsqueeze(0), m.history_c[lbi]), 1).detach()
                        new_split_history[lbi]["coeffs"][lidx] = {
                            "nonzero": torch.cat((split_coeffs_nonzero, history_coeffs_nonzero), 0).detach(),
                            "coeffs": torch.cat((split_coeffs_value, history_coeffs_value), 0).detach()}

                        new_split_history[lbi + batch]["beta"][lidx] = torch.cat(
                            (m.split_beta[lbi + batch].unsqueeze(0), m.history_beta[lbi + batch]), 1).detach()
                        new_split_history[lbi + batch]["c"][lidx] = torch.cat(
                            (m.split_c[lbi + batch].unsqueeze(0), m.history_c[lbi + batch]), 1).detach()
                        new_split_history[lbi + batch]["coeffs"][lidx] = {
                            "nonzero": torch.cat((split_coeffs_nonzero, history_coeffs_nonzero), 0).detach(),
                            "coeffs": torch.cat((split_coeffs_value, history_coeffs_value), 0).detach()}

                        if opt_bias:
                            split_bias = m.split_bias[lbi].detach().clone()
                            split_bias.requires_grad = False
                            new_split_history[lbi]["bias"][lidx] = torch.cat(
                                (split_bias.unsqueeze(0), m.history_bias[lbi]), 1).detach()
                            new_split_history[lbi + batch]["bias"][lidx] = torch.cat(
                                (split_bias.unsqueeze(0), m.history_bias[lbi + batch]), 1).detach()

        if debug:
            for shi, sh in enumerate(new_split_history):
                print(
                    f"##################################new_split_history batch {shi}##################################")
                print("beta", sh["beta"])
                print("c", sh["c"])
                print("coeffs", sh["coeffs"])
                print("bias", sh["bias"])
                print("single_beta", sh["single_beta"])

        finalize_time = time.time() - start_finalize_time
        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')
        return ret_l, ret_u, lAs, ret_s, betas, new_split_history, best_intermediate_betas, None


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

        for layer_idx, layer in enumerate(self.layers):
            # print(type(layer), primal.shape)
            # if gurobi_primals is not None and layer_idx < len(gurobi_primals):
            #     gp = torch.tensor(gurobi_primals[layer_idx]).float().to(primal.device)
            #     gip = torch.tensor(gurobi_integer_primals[relu_idx]).float().to(primal.device)
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
                # if relu_idx == 4: import pdb; pdb.set_trace()
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


    def build_the_model(self, input_domain, x, stop_criterion_func=stop_criterion_sum(0)):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        
        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        # first get CROWN bounds
        # Reference bounds are intermediate layer bounds from initial CROWN bounds.
        lb, ub, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
        print('initial CROWN bounds:', lb, ub)
        if stop_criterion_func(lb).all().item():
            # Fast path. Initial CROWN bound can verify the network.
            if not self.simplify:
                return None, lb[-1], None, None, None, None, None, None, None, None, None, None
            else:
                return None, lb[-1].item(), None, None, None, None, None, None, None, None, None, None
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                 'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                 'ob_early_stop': False, 'ob_verbose': 0,
                                 'ob_keep_best': True, 'ob_update_by_layer': True,
                                 'ob_lr': lr_init_alpha, 'ob_init': False,
                                 'ob_loss_reduction_func': loss_reduction_func,
                                 'ob_stop_criterion_func': stop_criterion_func,
                                 'ob_lr_decay': lr_decay}})
        lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
                                                 bound_upper=False, aux_reference_bounds=aux_reference_bounds)
        slope_opt = self.get_slope(self.net)[0]  # initial with one node only

        # build a complete A_dict
        # self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
        # self.layer_names.sort()

        # update bounds
        print('initial alpha-CROWN bounds:', lb, ub)
        primals, duals, mini_inp = None, None, None
        # mini_inp, primals = self.get_primals(self.A_dict)
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)

        if not self.simplify or stop_criterion_func(lb[-1]):
            history = [[[], []] for _ in range(len(self.net.relus))]
            return ub[-1], lb[-1], mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

        # for each pre-relu layer, we initial 2 lists for active and inactive split
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

    def build_the_model_with_refined_bounds(self, input_domain, x, refined_lower_bounds, refined_upper_bounds,
                                            stop_criterion_func=stop_criterion_sum(0), reference_slopes=None):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        no_joint_opt = arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        
        self.x = x
        self.input_domain = input_domain

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        slope_opt = None
        if not no_joint_opt:
            ######## using bab_verification_mip_refine.py ########
            lb, ub = refined_lower_bounds, refined_upper_bounds
        primals, duals, mini_inp = None, None, None

        return_A = True if get_upper_bound else False
        if get_upper_bound:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        # first get CROWN bounds
        self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c)
        # If we already have slopes available, we initialize them.
        if reference_slopes is not None:
            for m in self.net.relus:
                for spec_name, alpha in m.alpha.items():
                    # each slope size is (2, spec, 1, *shape); batch size is 1.
                    if spec_name in reference_slopes[m.name]:
                        reference_alpha = reference_slopes[m.name][spec_name]
                        if alpha.size() == reference_alpha.size():
                            print(f"setting alpha for layer {m.name} start_node {spec_name}")
                            alpha.data.copy_(reference_alpha)
                        else:
                            print(f"not setting layer {m.name} start_node {spec_name} because shape mismatch ({alpha.size()} != {reference_alpha.size()})")
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                 'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                 'ob_early_stop': False, 'ob_verbose': 0,
                                 'ob_keep_best': True, 'ob_update_by_layer': True,
                                 'ob_lr': lr_init_alpha, 'ob_init': False,
                                 'ob_loss_reduction_func': loss_reduction_func,
                                 'ob_stop_criterion_func': stop_criterion_func,
                                 'ob_lr_decay': lr_decay}})

        if no_joint_opt:
            # using init crown bounds as new_interval
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                new_interval[nd] = [layer.inputs[0].lower, layer.inputs[0].upper] ##!!
                # reference_bounds[nd] = [lb[i], ub[i]]
        else:
            # using refined bounds with init opt crown
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                print(i, nd, lb[i].shape)
                new_interval[nd] = [lb[i], ub[i]]
                # reference_bounds[nd] = [lb[i], ub[i]]
        ret = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='crown-optimized', return_A=return_A,
                                    new_interval=new_interval, bound_upper=False, needed_A_dict=self.needed_A_dict)
                                    #reference_bounds=reference_bounds, bound_upper=False)
        if return_A:
            lb, ub, A = ret
        else:
            lb, ub = ret

        print("alpha-CROWN with fixed intermediate bounds:", lb, ub)
        slope_opt = self.get_slope(self.net)[0]
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds

        if False and stop_criterion_func(lb[-1]):
            #################
            # using refined bounds with LP
            glb = self.build_the_model_lp(lb, ub)
            lb[-1] = torch.tensor([[glb]])
            print("LP with intermediate bounds from MIP:", lb[-1])
            # #################
            

        mask, lA = self.get_mask_lA_parallel(self.net)
        #import pdb; pdb.set_trace()
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            print("opt crown:", lb[-1])
            primal_x, ub_x = self.get_primal_upper_bound(A)
            print("### Extracting primal values for inputs done ###")

            # get the primal values for intermediate layers
            primal_values, integer_primals = self.get_neuron_primal(primal_x, lb, ub)
            # correct intermediate primal should produce the correct primal output lb
            print("primal lb:", primal_values[-1])
            print("### Extracting primal values and mixed integers for intermeidate nodes done ###")
        
            primals = {"p": primal_values, "z": integer_primals}
        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history


    def build_solver_model(self, lower_bounds, upper_bounds, timeout, mip_multi_proc=None, 
            mip_threads=1, input_domain=None, target=None, model_type="mip", simplified=False):
        # build the gurobi model according to the alpha-CROWN bounds
        return build_solver_model(self, lower_bounds, upper_bounds, timeout, mip_multi_proc, mip_threads, input_domain, target, model_type, simplified)


    def build_the_model_mip(self, lower_bounds, upper_bounds, simplified=False):
        # using the built gurobi model to solve mip formulation
        return build_the_model_mip(self, lower_bounds, upper_bounds, simplified)
    

    def build_the_model_lp(self, lower_bounds, upper_bounds, using_integer=True, simplified=True):
        # using the built gurobi model to solve lp formulation
        return build_the_model_lp(self, lower_bounds, upper_bounds, using_integer, simplified)


    def update_the_model_lp(self, lower_bounds, upper_bounds, decision, choice, using_integer=True, refine_intermediate_neurons=False):
        # solve split bounds with gurobi, set refine_intermeidate_neurons to be True if we need refinement
        return update_the_model_lp(self, lower_bounds, upper_bounds, decision, choice, using_integer, refine_intermediate_neurons)


    def update_the_model_mip(self, relu_mask, lower_bounds, upper_bounds, decision, choice):
        # solve split bounds with gurobi mip formulation
        return update_the_model_mip(self, relu_mask, lower_bounds, upper_bounds, decision, choice)


    def update_mip_model_fix_relu(self, relu_idx, status, target=None, async_mip=False, best_adv=None, adv_activation_pattern=None):
        # update mip model by manually fixing intermediate relus
        return update_mip_model_fix_relu(self, relu_idx, status, target, async_mip, best_adv, adv_activation_pattern)

    
    def build_the_model_mip_refine(m, lower_bounds, upper_bounds, stop_criterion_func=stop_criterion_min(1e-4), score=None, FSB_sort=True, topk_filter=1.):
            # using mip solver to refine the bounds of intermediate nodes
            return  build_the_model_mip_refine(m, lower_bounds, upper_bounds, stop_criterion_func, score, FSB_sort, topk_filter)
