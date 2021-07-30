import copy
import time
from collections import defaultdict, OrderedDict

import torch
from torch.nn import ZeroPad2d

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import (reduction_min, reduction_max, reduction_mean, reduction_sum,
                            stop_criterion_sum, stop_criterion_min)
from modules import Flatten

import multiprocessing
import sys
import os

try:
    import gurobipy as grb
except ModuleNotFoundError:
    pass

total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_transfer_time = total_finalize_time = 0.0

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

def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
    # TODO-finalize: remove this function! Use C instead.
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            joint_layer.to(layer.weight.device)
            new_all_layers.append(joint_layer)
    return new_all_layers


def add_single_prop(layers, gt, cls):
    """
    gt: ground truth lablel
    cls: class we want to verify against
    """
    # TODO-finalize: remove this function! Use C instead.
    if gt is not None:
        additional_lin_layer = nn.Linear(layers[-1].out_features, 1, bias=True)
        lin_weights = additional_lin_layer.weight.data
        lin_weights.fill_(0)
        lin_bias = additional_lin_layer.bias.data
        lin_bias.fill_(0)
        lin_weights[0, cls] = -1
        lin_weights[0, gt] = 1
        additional_lin_layer.to(layers[-1].weight.device)

        final_layers = [layers[-1], additional_lin_layer]
        final_layer = simplify_network(final_layers)
        verif_layers = layers[:-1] + final_layer
    else:
        # if there is no ture label, we only care about the target output
        final_layer = nn.Linear(layers[-1].in_features, 1, bias=True)
        final_layer.weight.data.copy_(layers[-1].weight.data[cls:cls+1] * -1)  # flip the weights since the spec is -1
        final_layer.bias.data.copy_(layers[-1].bias.data[cls])
        verif_layers = layers[:-1] + [final_layer]

    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


class LiRPAConvNet:
    def __init__(self, model_ori, pred, test, solve_slope=False, device='cuda', simplify=True, in_size=(1, 3, 32, 32),
                 conv_mode='patches', deterministic=False, c=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        if type(list(model_ori.modules())[-1]) is not torch.nn.Linear:
            simplify = False  # test_nano.onnx has relu layer at the last layer.
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.simplify = simplify
        if simplify: assert c is None, "c is None for simplified cases"
        self.c = c
        self.pred = pred
        if simplify:
            added_prop_layers = add_single_prop(layers, pred, test)
            self.layers = added_prop_layers
            net._modules[list(model_ori._modules)[-1]] = self.layers[-1]
        else:
            self.layers = layers
        self.solve_slope = solve_slope
        if solve_slope:
            self.net = BoundedModule(net, torch.zeros(in_size, device=device), bound_opts={'relu': 'adaptive', 'deterministic': deterministic, 'conv_mode': conv_mode},
                                     device=device)
        else:
            self.net = BoundedModule(net, torch.zeros(in_size, device=device), bound_opts={'relu': 'same-slope', 'conv_mode': conv_mode}, device=device)
        self.net.eval()

    def get_lower_bound(self, pre_lbs, pre_ubs, split, slopes=None, betas=None, history=None, decision_thresh=0, layer_set_bound=True,
                        beta=True, lr_alpha=0.1, lr_beta=0.05, optimizer="adam", iteration=20, beta_warmup=True,
                        opt_coeffs=True, opt_bias=True, lp_test=None, split_history=None, single_node_split=True,
                        intermediate_betas=None, opt_intermediate_beta=True, intermediate_refinement_layers=None):

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
        if single_node_split:
            ret = self.update_bounds_parallel(pre_lbs, pre_ubs, split, slopes, beta=beta, betas=betas, early_stop=False,
                                              optimizer=optimizer, iteration=iteration, history=history,
                                              layer_set_bound=layer_set_bound, lr_alpha=lr_alpha, lr_beta=lr_beta)
        else:
            ret = self.update_bounds_parallel_general(pre_lbs, pre_ubs, split, slopes, beta=beta, early_stop=False,
                                              optimizer=optimizer, iteration=iteration, history=history,
                                              split_history=split_history, intermediate_betas=intermediate_betas,
                                              layer_set_bound=layer_set_bound, lr_alpha=lr_alpha, lr_beta=lr_beta,
                                              beta_warmup=beta_warmup,
                                              opt_coeffs=opt_coeffs, opt_bias=opt_bias, opt_intermediate_beta=opt_intermediate_beta,
                                              intermediate_refinement_layers=intermediate_refinement_layers)

        lower_bounds, upper_bounds, lAs, slopes, betas, split_history, best_intermediate_betas = ret

        if lp_test == "LP":
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                self.update_the_model_lp(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                self.update_the_model_lp(lower_bounds[bdi + total_batch],
                                         upper_bounds[bdi + total_batch], bd[0], choice=0)
        elif lp_test == "MIP":
            for bdi, bd in enumerate(split["decision"]):
                total_batch = len(split["decision"])
                assert 2 * total_batch == len(lower_bounds)
                self.update_the_model_mip(lower_bounds[bdi], upper_bounds[bdi], bd[0], choice=1)
                self.update_the_model_mip(lower_bounds[bdi + total_batch],
                                          upper_bounds[bdi + total_batch], bd[0], choice=0)

        end = time.time()
        print('batch bounding time: ', end - start)
        return [i[-1].item() for i in upper_bounds], [i[-1].item() for i in lower_bounds], None, lAs, lower_bounds, \
               upper_bounds, slopes, split_history, betas, best_intermediate_betas

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
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha = OrderedDict()
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

    def get_candidate_parallel(self, model, lb, ub, batch):
        # get the intermediate bounds in the current model
        lower_bounds = []
        upper_bounds = []

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower)
            upper_bounds.append(layer.inputs[0].upper)

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch, -1).detach())
        upper_bounds.append(ub.view(batch, -1).detach())

        return lower_bounds, upper_bounds

    def get_mask_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.relus:
            mask_tmp = torch.zeros_like(this_relu.inputs[0].lower)
            unstable = ((this_relu.inputs[0].lower < 0) & (this_relu.inputs[0].upper > 0))
            mask_tmp[unstable] = -1
            active = (this_relu.inputs[0].lower >= 0)
            mask_tmp[active] = 1
            # otherwise 0, for inactive neurons

            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            lA.append(this_relu.lA.squeeze(0))

        ret_mask, ret_lA = [], []
        for i in range(mask[0].size(0)):
            ret_mask.append([j[i:i+1] for j in mask])
            ret_lA.append([j[i:i+1] for j in lA])
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

    def get_beta(self, model, splits_per_example):
        # split_per_example only has half of the examples.
        batch_size = splits_per_example.size(0)
        retb = [[] for i in range(batch_size * 2)]
        for mi, m in enumerate(model.relus):
            for i in range(batch_size):
                # Save only used beta, discard padding beta.
                retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                retb[i + batch_size].append(m.sparse_beta[i + batch_size, :splits_per_example[i, mi]])
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
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:]
        return ret

    def set_slope(self, model, slope):
        for m in model.relus:
            for spec_name in list(m.alpha.keys()):
                if spec_name in slope[0][m.name]:
                    # Merge all slope vectors together in this batch. Size is (2, spec, batch, *shape).
                    m.alpha[spec_name] = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope))], dim=2)
                    # Duplicate for the second half of the batch.
                    m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3)))
                else:
                    # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                    del m.alpha[spec_name]

    def reset_beta(self, model, batch, max_splits_per_layer=None, betas=None):
        # Recreate new beta with appropriate shape.
        for mi, m in enumerate(self.net.relus):
            # Create only the non-zero beta. For each layer, it is padded to maximal length.
            # We create tensors on CPU first, and they will be transferred to GPU after initialized.
            m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
            m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            # Load beta from history.
            for bi in range(len(betas)):
                if betas[bi] is not None:
                    # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                    valid_betas = len(betas[bi][mi])
                    m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
            # This is the beta variable to be optimized for this layer.
            m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()

    def check_optimization_success(self, introduced_constrs_all=None, model=None):
        if model is None:
            model = self.model
        if model.status == 2:
            # Optimization successful, nothing to complain about
            pass
        elif model.status == 3:
            for introduced_cons_layer in introduced_constrs_all:
                model.remove(introduced_cons_layer)
            # The model is infeasible. We have made incompatible
            # assumptions, so this subdomain doesn't exist.
            raise InfeasibleMaskException()
        else:
            print('\n')
            print(f'Gurobi model.status: {model.status}\n')
            raise NotImplementedError

    def copy_model(self, model, basis=True, use_basis_warm_start=True, remove_constr_list=[]):
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

    """Main function for computing bounds after branch and bound in Beta-CROWN."""
    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None, beta=True,
                               betas=None, early_stop=True, optimizer="adam", iteration=20, history=None,
                               layer_set_bound=True, lr_alpha=0.1, lr_beta=0.05, shortcut=False):
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time
        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0
        # update optimize-CROWN bounds in a parallel way

        # if history is None:
        #     history = []

        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)

        # layers_need_change = np.unique(decision[:, 0])
        # layers_need_change.sort()

        # initial results with empty list
        ret_l = [[] for _ in range(batch * 2)]
        ret_u = [[] for _ in range(batch * 2)]
        ret_s = [[] for _ in range(batch * 2)]
        ret_b = [[] for _ in range(batch * 2)]
        new_split_history = [{} for _ in range(batch * 2)]
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        start_prepare_time = time.time()
        # iteratively change upper and lower bound from former to later layer

        if beta:
            splits_per_example = torch.zeros(size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu', requires_grad=False)
            for bi in range(batch):
                d = decision[bi][0]
                for mi, layer_splits in enumerate(history[bi]):
                    splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.
            # This is the maximum number of split in each relu neuron for each batch.
            max_splits_per_layer = splits_per_example.max(dim=0)[0]

            # Create and load warmup beta.
            self.reset_beta(self.net, batch, betas=betas, max_splits_per_layer=max_splits_per_layer)  # warm start beta

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

            upper_bounds = [torch.cat((i, i), dim=0) for i in pre_ub_all[:-1]]
            lower_bounds = [torch.cat((i, i), dim=0) for i in pre_lb_all[:-1]]

            # pre_lb_all = [torch.cat(2 * [i]) for i in pre_lb_all]
            # pre_ub_all = [torch.cat(2 * [i]) for i in pre_ub_all]
            # Only the last element is used later.
            pre_lb_last = torch.cat([pre_lb_all[-1], pre_lb_all[-1]])
            pre_ub_last = torch.cat([pre_ub_all[-1], pre_ub_all[-1]])

            new_candidate = {}
            for d in range(len(lower_bounds)):
                if len(zero_indices_batch[d]):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    lower_bounds[d].view(2*batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                    upper_bounds[d].view(2*batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                new_candidate[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]

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
        # self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes)

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': beta, 'ob_single_node_split': True,
                'ob_update_by_layer': layer_set_bound, 'ob_optimizer':optimizer}})
            with torch.no_grad():
                lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                 new_interval=new_candidate, bound_upper=False, return_A=False)
            return lb

        if layer_set_bound:
            if self.solve_slope:
                start_beta_bound_time = time.time()
                self.net.set_bound_opts({'optimize_bound_args':
                                             {'ob_beta': beta, 'ob_single_node_split': True,
                                              'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                              'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
                lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                  new_interval=new_candidate, return_A=False, bound_upper=False)
                beta_bound_time += time.time() - start_beta_bound_time
            else:
                with torch.no_grad():
                    lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                      new_interval=new_candidate, bound_upper=False, return_A=False)

            # we don't care about the upper bound of the last layer

        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                    'ob_iteration': iteration, 'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=False, bound_upper=False)

        bound_time += time.time() - start_bound_time

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb = lb.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()
            # Reorganize tensors.
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, lb + 99, batch * 2)
            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_last.cpu())
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_last.cpu())
            # reshape the results based on batch.
            for i in range(batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]

                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]

            lAs = self.get_lA_parallel(transfer_net)
            if len(slopes) > 0:
                ret_s = self.get_slope(transfer_net)

            if beta:
                ret_b = self.get_beta(transfer_net, splits_per_example)

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

        return ret_l, ret_u, lAs, ret_s, ret_b, new_split_history, best_intermediate_betas

    def update_bounds_parallel_general(self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None,
                               beta=True, early_stop=True, optimizer="adam", iteration=20, split_history=None,
                               history=None, lr_alpha=0.1, lr_beta=0.05,  intermediate_betas=None,
                               layer_set_bound=True, beta_warmup=True, opt_coeffs=True,
                               opt_bias=True, debug=False, opt_intermediate_beta=True, intermediate_refinement_layers=None):
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
        betas = [[] for _ in range(batch * 2)]
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
                    upper_bounds[d].view(batch, -1)[i][idx] = 0  # 1e-10
                    lower_bounds[d].view(batch, -1)[i][idx] = 0  # -1e-10

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

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes)

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
            self.net.init_intermediate_betas = intermediate_betas + intermediate_betas

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if layer_set_bound and not opt_intermediate_beta:
            if self.solve_slope:
                start_beta_bound_time = time.time()
                self.net.set_bound_opts({'optimize_bound_args':
                                             {'ob_beta': beta, 'ob_single_node_split': False,
                                              'ob_opt_coeffs': opt_coeffs, 'ob_opt_bias': opt_bias,
                                              'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                              'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
                lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                  new_interval=new_candidate, return_A=False, bound_upper=False)
                beta_bound_time += time.time() - start_beta_bound_time
            else:
                with torch.no_grad():
                    lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                      new_interval=new_candidate, bound_upper=False, return_A=False)

            # we don't care about the upper bound of the last layer

        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                                         'ob_iteration': iteration, 'ob_lr': lr_alpha, 'ob_lr_beta': lr_beta,
                                         'ob_opt_coeffs': opt_coeffs, 'ob_opt_bias': opt_bias,
                                         'ob_single_node_split': False, 'ob_intermediate_beta': opt_intermediate_beta,
                                         'ob_intermediate_refinement_layers': intermediate_refinement_layers,
                                         'ob_optimizer': optimizer}})
            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                              new_interval=new_candidate, return_A=False, bound_upper=False)

        bound_time += time.time() - start_bound_time

        # print('best results of parent nodes', pre_lb_all[-1].repeat(2, 1))
        # print('finally, after optimization:', lower_bounds_new[-1])

        # Move tensors to CPU for all elements in this batch.
        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb = lb.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)
            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1].cpu())
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1].cpu())

            lAs = self.get_lA_parallel(self.net)

            if len(slopes) > 0:
                ret_s = self.get_slope(self.net)

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
                    for split_layer, all_int_betas_this_layer in self.net.best_intermediate_betas.items():
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
        return ret_l, ret_u, lAs, ret_s, betas, new_split_history, best_intermediate_betas

    def fake_forward(self, x):
        for layer in self.layers:
            if type(layer) is nn.Linear:
                x = F.linear(x, layer.weight, layer.bias)
            elif type(layer) is nn.Conv2d:
                x = F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
            elif type(layer) == nn.ReLU:
                x = F.relu(x)
            elif type(layer) == Flatten:
                x = x.reshape(x.shape[0], -1)
            elif type(layer) == nn.ZeroPad2d:
                x = F.pad(x, layer.padding)
            else:
                print(type(layer))
                raise NotImplementedError

        return x

    def get_primals(self, A, return_x=False):
        # get primal input by using A matrix
        input_A_lower = A[self.layer_names[-1]][self.net.input_name[0]][0]
        batch = input_A_lower.shape[1]
        l = self.input_domain[:, :, :, 0].repeat(batch, 1, 1, 1)
        u = self.input_domain[:, :, :, 1].repeat(batch, 1, 1, 1)
        diff = 0.5 * (l - u)  # already flip the sign by using lower - upper
        net_input = diff * torch.sign(input_A_lower.squeeze(0)) + self.x
        if return_x: return net_input

        primals = [net_input]
        for layer in self.layers:
            if type(layer) is nn.Linear:
                pre = primals[-1]
                primals.append(F.linear(pre, layer.weight, layer.bias))
            elif type(layer) is nn.Conv2d:
                pre = primals[-1]
                primals.append(F.conv2d(pre, layer.weight, layer.bias,
                                        layer.stride, layer.padding, layer.dilation, layer.groups))
            elif type(layer) == nn.ReLU:
                primals.append(F.relu(primals[-1]))
            elif type(layer) == Flatten:
                primals.append(primals[-1].reshape(primals[-1].shape[0], -1))
            else:
                print(type(layer))
                raise NotImplementedError

        # primals = primals[1:]
        primals = [i.detach().clone() for i in primals]
        # print('primals', primals[-1])

        return net_input, primals

    def build_the_model(self, input_domain, x, no_lp=True, 
                        lr_init_alpha=0.5, init_iteration=100, share_slopes=False, optimizer="adam",
                        loss_reduction_func=reduction_sum, stop_criterion_func=stop_criterion_sum(0), 
                        lr_decay=0.98):
        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        # first get CROWN bounds
        if self.solve_slope:
            self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c)
            self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                     'ob_early_stop': False, 'ob_verbose': 0,
                                     'ob_keep_best': True, 'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 'ob_init': False,
                                     'ob_loss_reduction_func': loss_reduction_func, 
                                     'ob_stop_criterion_func': stop_criterion_func, 
                                     'ob_lr_decay': lr_decay}})
            lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
                                                     bound_upper=False)
            slope_opt = self.get_slope(self.net)[0]  # initial with one node only
        else:
            with torch.no_grad():
                lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='backward', return_A=False)

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

        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history


    def build_the_model_with_refined_bounds(self, input_domain, x, refined_lower_bounds, refined_upper_bounds, 
                        no_lp=True, lr_init_alpha=0.5, init_iteration=100, share_slopes=False, optimizer="adam",
                        loss_reduction_func=reduction_sum, stop_criterion_func=stop_criterion_sum(0), lr_decay=0.98):
        self.x = x
        self.input_domain = input_domain

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        slope_opt = None
        ######## using bab_verification_mip_refine.py ########
        lb, ub = refined_lower_bounds, refined_upper_bounds
        primals, duals, mini_inp = None, None, None

        # first get CROWN bounds
        if self.solve_slope:
            self.net.init_slope((self.x,), share_slopes=share_slopes)
            self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                     'ob_early_stop': False, 'ob_verbose': 0,
                                     'ob_keep_best': True, 'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 'ob_init': False,
                                     'ob_loss_reduction_func': loss_reduction_func, 
                                     'ob_stop_criterion_func': stop_criterion_func, 
                                     'ob_lr_decay': lr_decay}})

            #################
            # using refined bounds with init opt crown
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                print(i, nd, lb[i].shape)
                new_interval[nd] = [lb[i], ub[i]]
                reference_bounds[nd] = [lb[i], ub[i]]
            lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='crown-optimized', return_A=False,
                                        new_interval=new_interval, bound_upper=False)
                                        #reference_bounds=reference_bounds, bound_upper=False)
            print("alpha-CROWN with intermediate bounds from MIP:", lb, ub)
            slope_opt = self.get_slope(self.net)[0]
            lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
            #################

            if False and stop_criterion_func(lb[-1]):
                #################
                # using refined bounds with LP
                glb = self.build_the_model_lp(lb, ub)
                lb[-1] = torch.tensor([[glb]])
                print("LP with intermediate bounds from MIP:", lb[-1])
                # #################
            
        else:
            with torch.no_grad():
                lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='backward', return_A=False)
            lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds

        mask, lA = self.get_mask_lA_parallel(self.net)
        #import pdb; pdb.set_trace()
        history = [[[], []] for _ in range(len(self.net.relus))]
        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

        
    def build_the_model_lp(self, lower_bounds, upper_bounds):
        """
        Before the first branching, we build the model and create a mask matrix
        Output: relu_mask, current intermediate upper and lower bounds, a list of
                indices of the layers right before a Relu layer
                the constructed gurobi model
        NOTE: we keep all bounds as a list of tensors from now on.
              Only lower and upper bounds are kept in the same shape as layers' outputs.
              Mask is linearized
              Gurobi_var lists are lineariezd
              self.model_lower_bounds and self.model_upper_bounds are kepts mainly for
              debugging purpose and could be removed
        """
        new_relu_mask = []
        x = self.x
        input_domain = self.input_domain

        # Initialize the model
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)

        # keep a record of model's information
        self.gurobi_vars = []
        self.relu_constrs = []
        self.relu_indices_mask = []

        ## Do the input layer, which is a special case
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
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
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        row_vars.append(v)
                    chan_vars.append(row_vars)
                inp_gurobi_vars.append(chan_vars)
        self.model.update()

        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        relu_idx = 0
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                # Get the better estimates from KW and Interval Bounds
                # print("linear", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape)
                out_lbs = lower_bounds[relu_idx].squeeze(0)
                out_ubs = upper_bounds[relu_idx].squeeze(0)
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, self.gurobi_vars[-1])

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(lin_expr == v)
                    self.model.update()

                    new_layer_gurobi_vars.append(v)

            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                if relu_idx == 0:
                    pre_lb_size = self.x.shape
                else:
                    pre_lb_size = lower_bounds[relu_idx-1].size()
                out_lbs = lower_bounds[relu_idx]
                out_ubs = upper_bounds[relu_idx]
                # print("conv", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape, layer.bias.shape)

                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            # print(layer.bias.shape, out_chan_idx, out_lbs.size(1))
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx == pre_lb_size[2]):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx == pre_lb_size[3]):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]
                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

            elif type(layer) is nn.ReLU:
                new_relu_layer_constr = []
                this_relu = self.net.relus[relu_idx]
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    new_layer_mask = []
                    # print("conv relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    temp = pre_lbs.size()
                    out_chain = temp[0]
                    out_height = temp[1]
                    out_width = temp[2]
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
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
                                    v = self.model.addVar(ub=ub, lb=pre_lb,
                                                          obj=0, vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    out_idx = col_idx + row_idx * out_width + chan_idx * out_height * out_width

                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{out_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{out_idx}_a_1'))
                                    new_relu_layer_constr.append(self.model.addConstr(
                                        pre_ub * pre_var - (pre_ub - pre_lb) * v >= pre_ub * pre_lb,
                                        name=f'ReLU{relu_idx}_{out_idx}_a_2'))
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    # this is linear relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    # print("linear relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    new_layer_mask = []
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
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
                            v = self.model.addVar(ub=ub, lb=pre_lb,
                                                  obj=0,
                                                  vtype=grb.GRB.CONTINUOUS,
                                                  name=f'ReLU{layer_idx}_{neuron_idx}')

                            new_relu_layer_constr.append(
                                self.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(pre_ub * pre_var - (pre_ub - pre_lb) * v >= pre_ub * pre_lb,
                                                     name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                            new_layer_mask.append(-1)

                        new_layer_gurobi_vars.append(v)

                new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
                self.relu_constrs.append(new_relu_layer_constr)
                relu_idx += 1

            elif type(layer) == Flatten or "Flatten" in str(type(layer)):
                for chan_idx in range(len(self.gurobi_vars[-1])):
                    for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"


        self.model.update()
        print('Finished building Gurobi LP model. Start solving the LP!')
        # import pdb; pdb.set_trace()
        guro_start = time.time()
        # self.model.setParam("PreSolve", 0)
        # self.model.setParam("Method", 1)

        self.gurobi_vars[-1][0].LB = -100000
        self.gurobi_vars[-1][0].UB = 100000
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        # self.model.write("save.lp")
        try:
            self.model.optimize()
        except grb.GurobiError as e: 
            handle_gurobi_error(e.message)


        # for c in self.model.getConstrs():
        #     print('The dual value of %s : %g %g'%(c.constrName,c.pi, c.slack))

        assert self.model.status == 2, f"LP wasn't optimally solved status:{self.model.status}"
        self.check_optimization_success()

        guro_end = time.time()
        print('Gurobi solved the LP with time', guro_end - guro_start)

        glb = self.gurobi_vars[-1][0].X
        lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)
        print("gurobi glb:", glb)

        # inp_size = lower_bounds[0].size()
        # mini_inp = torch.zeros(inp_size).to(lower_bounds[0].device)

        # if len(inp_size) == 1:
        #     # This is a linear input.
        #     for i in range(inp_size[0]):
        #         mini_inp[i] = self.gurobi_vars[0][i].x
        # elif len(inp_size) == 0:
        #     mini_inp.data = torch.tensor(self.gurobi_vars[0][0].x).cuda()
        # else:
        #     for i in range(inp_size[0]):
        #         for j in range(inp_size[1]):
        #             for k in range(inp_size[2]):
        #                 mini_inp[i, j, k] = self.gurobi_vars[0][i][j][k].x

        # self.relu_indices_mask = [(i == -1).nonzero().view(-1).tolist() for i in new_relu_mask]

        # gub = self.net(mini_inp.unsqueeze(0)).item()
        # print("gub:", mini_inp, gub)

        # record model information
        # indices for undecided relu-nodes
        self.relu_indices_mask = [(i == -1).nonzero().view(-1).tolist() for i in new_relu_mask]
        # flatten high-dimensional gurobi var lists
        for l_idx, layer in enumerate(self.layers):
            if type(layer) is nn.Conv2d:
                flattened_gurobi = []
                for chan_idx in range(len(self.gurobi_vars[l_idx + 1])):
                    for row_idx in range(len(self.gurobi_vars[l_idx + 1][chan_idx])):
                        flattened_gurobi.extend(self.gurobi_vars[l_idx + 1][chan_idx][row_idx])
                self.gurobi_vars[l_idx + 1] = flattened_gurobi
                if type(self.layers[l_idx + 1]) is nn.ReLU:
                    flattened_gurobi = []
                    for chan_idx in range(len(self.gurobi_vars[l_idx + 2])):
                        for row_idx in range(len(self.gurobi_vars[l_idx + 2][chan_idx])):
                            flattened_gurobi.extend(self.gurobi_vars[l_idx + 2][chan_idx][row_idx])
                    self.gurobi_vars[l_idx + 2] = flattened_gurobi
            else:
                continue
        return glb


    def build_the_model_mip(self, lower_bounds, upper_bounds, timeout, mip_multi_proc=None, mip_threads=1):
        """
        Before the first branching, we build the model and create a mask matrix
        Output: relu_mask, current intermediate upper and lower bounds, a list of
                indices of the layers right before a Relu layer
                the constructed gurobi model
        NOTE: we keep all bounds as a list of tensors from now on.
              Only lower and upper bounds are kept in the same shape as layers' outputs.
              Mask is linearized
              Gurobi_var lists are lineariezd
              self.model_lower_bounds and self.model_upper_bounds are kepts mainly for
              debugging purpose and could be removed
        """
        new_relu_mask = []
        x = self.x
        input_domain = self.input_domain

        # setting for aws instance
        # mip_multi_proc = 4
        # mip_threads = 4
        if mip_multi_proc is None:
            mip_multi_proc = multiprocessing.cpu_count()
            print("preset mip_multi_proc as default setting:", mip_multi_proc)

        # Initialize the model
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', mip_threads)
        self.model.setParam("FeasibilityTol", 2e-5)
        self.model.setParam('TimeLimit', timeout)
        print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads}, total threads used: {mip_multi_proc*mip_threads}")
        build_mip_time = time.time()

        # keep a record of model's information
        self.gurobi_vars = []
        self.relu_constrs = []
        self.relu_indices_mask = []

        ## Do the input layer, which is a special case
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
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
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        row_vars.append(v)
                    chan_vars.append(row_vars)
                inp_gurobi_vars.append(chan_vars)
        self.model.update()

        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        relu_idx = 0
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                last_layer = False
                if relu_idx == len(self.net.relus):
                    last_layer = True
                # Get the better estimates from KW and Interval Bounds
                # print("linear", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape)
                out_lbs = lower_bounds[relu_idx].squeeze(0)
                out_ubs = upper_bounds[relu_idx].squeeze(0)
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    if last_layer and not self.simplify:
                        lin_expr = layer.bias[self.pred].item()-layer.bias[neuron_idx].item()
                        coeffs = layer.weight[self.pred, :]-layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, self.gurobi_vars[-1])

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(lin_expr == v)
                    self.model.update()

                    new_layer_gurobi_vars.append(v)
            elif type(layer) is nn.AvgPool2d: # I implement avgpool mostly follow the Conv2d, but with constant weights.
                value = 1.0/(layer.kernel_size[0] * layer.kernel_size[1])
                assert layer.padding[0] == layer.padding[1]
                padding = layer.padding[0]
                input_x, input_y = len(self.gurobi_vars[-1][0]), len(self.gurobi_vars[-1][0][0])
                output_x = output_y = (2 * padding + input_x - (layer.stride[0] - 1))//layer.stride[0]
                chan_num = len(self.gurobi_vars[-1])
                for out_chan_idx in range(chan_num):
                    out_chan_vars = []
                    for out_row_idx in range(output_x):
                        out_row_vars = []
                        for out_col_idx in range(output_y):
                            # print(layer.bias.shape, out_chan_idx, out_lbs.size(1))
                            lin_expr = 0.0
                            for ker_row_idx in range(layer.kernel_size[0]):
                                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                if (in_row_idx < 0) or (in_row_idx == len(self.gurobi_vars[-1][out_chan_idx][ker_row_idx])):
                                    # This is padding -> value of 0
                                    continue
                                for ker_col_idx in range(layer.kernel_size[1]):
                                    in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                    if (in_col_idx < 0) or (in_col_idx == input_y):
                                        # This is padding -> value of 0
                                        continue
                                    coeff = value
                                    lin_expr += coeff * self.gurobi_vars[-1][out_chan_idx][in_row_idx][in_col_idx]
                            v = self.model.addVar(lb=-float('inf'), ub=float('inf'),
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)
            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                if relu_idx == 0:
                    pre_lb_size = self.x.shape
                else:
                    pre_lb_size = lower_bounds[relu_idx-1].size()
                out_lbs = lower_bounds[relu_idx]
                out_ubs = upper_bounds[relu_idx]
                # print("conv", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape, layer.bias.shape)
                gvars_array = np.array(self.gurobi_vars[-1])

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
                                # print(lin_expr)
                                # exit()

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
                                #         lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]
                                # print(lin_expr)
                                # exit()

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

            elif type(layer) is nn.ReLU:
                new_relu_layer_constr = []
                this_relu = self.net.relus[relu_idx]
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    new_layer_mask = []
                    # print("conv relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    temp = pre_lbs.size()
                    out_chain = temp[0]
                    out_height = temp[1]
                    out_width = temp[2]
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
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

                                    v = self.model.addVar(ub=ub, lb=pre_lb,
                                                          obj=0,
                                                          vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    # binary indicator
                                    a = self.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')

                                    new_relu_layer_constr.append(
                                        self.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                             name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_3'))

                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    # this is linear relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    # print("linear relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    new_layer_mask = []
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
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
                            v = self.model.addVar(ub=ub, lb=pre_lb,
                                                  obj=0,
                                                  vtype=grb.GRB.CONTINUOUS,
                                                  name=f'ReLU{layer_idx}_{neuron_idx}')
                            # binary indicator
                            a = self.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_{neuron_idx}')

                            new_relu_layer_constr.append(
                                self.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                     name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_2'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(v >= 0, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_3'))

                            new_layer_mask.append(-1)

                        new_layer_gurobi_vars.append(v)

                new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
                self.relu_constrs.append(new_relu_layer_constr)
                relu_idx += 1
            elif type(layer) is nn.MaxPool2d:
                input_x, input_y = len(self.gurobi_vars[-1][0]), len(self.gurobi_vars[-1][0][0])
                assert layer.padding[0] == layer.padding[1]
                padding = layer.padding[0]
                output_x = output_y = (2 * padding + input_x - (layer.stride[0] - 1))//layer.stride[0]
                chan_num = len(self.gurobi_vars[-1])

                pre_ubs = layer(F.relu(upper_bounds[relu_idx].squeeze(0)))

                for out_chan_idx in range(chan_num):
                    out_chan_vars = []
                    for out_row_idx in range(output_x):
                        out_row_vars = []
                        for out_col_idx in range(output_y):
                            a_sum = 0.0
                            v = self.model.addVar(lb=-float('inf'), ub=float('inf'),
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            for ker_row_idx in range(layer.kernel_size[0]):
                                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                if (in_row_idx < 0) or (in_row_idx == len(self.gurobi_vars[-1][out_chan_idx][ker_row_idx])):
                                    # This is padding -> value of 0
                                    continue
                                for ker_col_idx in range(layer.kernel_size[1]):
                                    in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                    if (in_col_idx < 0) or (in_col_idx == input_y):
                                        # This is padding -> value of 0
                                        continue
                                    var = self.gurobi_vars[-1][out_chan_idx][in_row_idx][in_col_idx]
                                    a = self.model.addVar(vtype=grb.GRB.BINARY)
                                    a_sum += a
                                    self.model.addConstr(v >= var)
                                    self.model.addConstr(v <= var + (1-a)*pre_ubs[out_chan_idx,out_row_idx,out_col_idx].item())
                                    self.model.update()
                            self.model.addConstr(a_sum == 1)
                            self.model.update()
                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)
            elif type(layer) == Flatten or "Flatten" in str(type(layer)):
                for chan_idx in range(len(self.gurobi_vars[-1])):
                    for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
            elif "Pad" in str(type(layer)):
                new_layer_gurobi_vars = []
                assert layer.padding[0] == layer.padding[1]
                padding = layer.padding[0]
                for chan_indx in range(len(self.gurobi_vars[-1])):
                    out_chan_vars = []
                    for i in range(padding):
                        out_row_vars = []
                        for j in range(padding*2 + len(self.gurobi_vars[-1][chan_indx][row_idx])):
                            out_row_vars.append(layer.value)
                        out_chan_vars.append(out_row_vars)
                    for row_idx in range(len(self.gurobi_vars[-1][chan_indx])):
                        out_row_vars = []
                        for i in range(padding):
                            out_row_vars.append(layer.value)
                        for i in range(len(self.gurobi_vars[-1][chan_indx][row_idx])):
                            out_row_vars.append(self.gurobi_vars[-1][chan_indx][row_idx][i])
                        for i in range(padding):
                            out_row_vars.append(layer.value)
                        out_chan_vars.append(out_row_vars)
                    for i in range(padding):
                        out_row_vars = []
                        for j in range(padding*2 + len(self.gurobi_vars[-1][chan_indx][row_idx])):
                            out_row_vars.append(layer.value)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)
            else:
                print("{} is not implemented".format(type(layer)))
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        print("build_mip_time:", time.time()-build_mip_time)

        if self.simplify:
            # Assert that this is as expected a network with a single output
            assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output if simplified"

            self.model.update()
            print('finished building Gurobi MIP model, calling optimize function')
            # import pdb; pdb.set_trace()
            guro_start = time.time()
            # self.model.setParam("PreSolve", 0)
            # self.model.setParam("Method", 1)
            # self.model.setParam("FeasibilityTol", 2e-5)

            self.gurobi_vars[-1][0].LB = -100000
            self.gurobi_vars[-1][0].UB = 100000
            self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
            # self.model.write("save.mip")
            try:
                self.model.optimize()
            except grb.GurobiError as e: 
                handle_gurobi_error(e.message)
            # for c in self.model.getConstrs():
            #     print('The dual value of %s : %g %g'%(c.constrName,c.pi, c.slack))

            assert self.model.status == 2, f"LP wasn't optimally solved status:{self.model.status}"
            self.check_optimization_success()

            guro_end = time.time()
            print('Gurobi solved the MIP with ', guro_end - guro_start, "seconds")

            glb = self.gurobi_vars[-1][0].X
            lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)
            print("gurobi glb:", glb)

            # record model information
            # indices for undecided relu-nodes
            self.relu_indices_mask = [(i == -1).nonzero().view(-1).tolist() for i in new_relu_mask]
            # flatten high-dimensional gurobi var lists
            for l_idx, layer in enumerate(self.layers):
                if type(layer) is nn.Conv2d:
                    flattened_gurobi = []
                    for chan_idx in range(len(self.gurobi_vars[l_idx + 1])):
                        for row_idx in range(len(self.gurobi_vars[l_idx + 1][chan_idx])):
                            flattened_gurobi.extend(self.gurobi_vars[l_idx + 1][chan_idx][row_idx])
                    self.gurobi_vars[l_idx + 1] = flattened_gurobi
                    if type(self.layers[l_idx + 1]) is nn.ReLU:
                        flattened_gurobi = []
                        for chan_idx in range(len(self.gurobi_vars[l_idx + 2])):
                            for row_idx in range(len(self.gurobi_vars[l_idx + 2][chan_idx])):
                                flattened_gurobi.extend(self.gurobi_vars[l_idx + 2][chan_idx][row_idx])
                        self.gurobi_vars[l_idx + 2] = flattened_gurobi
                else:
                    continue
            return glb
        else:
            # not simplified directly after opt crown init bounds
            self.model.update()
            print('finished building Gurobi MIP model, calling optimize function')
            lb = lower_bounds[-1][0]
            print(lb)
            candidates, candidate_neuron_ids = [], []
            for pidx, lbi in enumerate(lb):
                if lbi >= 0: continue
                candidates.append(self.gurobi_vars[-1][pidx].VarName)
                candidate_neuron_ids.append(pidx)
                # SINGLE THREAD
                # mip_time = time.time()
                # self.model.setObjective(self.gurobi_vars[-1][pidx], grb.GRB.MINIMIZE)
                # self.model.optimize()
                # assert self.model.status == 2, f"status: {self.model.status}"

                # glb = self.gurobi_vars[-1][pidx].X
                # print(f"mip: label {self.pred} target label {pidx}, orig {lbi}, mip {glb}, mip time {time.time()-mip_time}")
                # lb[pidx] = glb
                # if glb<0: break
            # MULTITHREAD
            global multiprocess_mip_model, stop_multiprocess
            multiprocess_mip_model = self.model
            with multiprocessing.Pool(mip_multi_proc) as pool:
                solver_result = pool.map(mip_solver_lb, candidates)
            
            multiprocess_mip_model = None
            stop_multiprocess = False

            status = [-1 for i in lb]
            for (vlb, s), pidx in zip(solver_result, candidate_neuron_ids):
                lb[pidx] = vlb
                status[pidx]  = s
            return lb, status


    def compute_ratio(self, lower_bound, upper_bound):
        lower_temp = lower_bound.clamp(max=0)
        upper_temp = F.relu(upper_bound)
        slope_ratio = upper_temp / (upper_temp - lower_temp)
        intercept = -1 * lower_temp * slope_ratio

        return slope_ratio, intercept


    def get_branching_op(self, branching_reduceop):
        if branching_reduceop == 'min':
            reduce_op = torch.min
        elif branching_reduceop == 'max':
            reduce_op = torch.max
        elif branching_reduceop == 'mean':
            reduce_op = torch.mean
        else:
            reduce_op = None
        return reduce_op


    def FSB_score(self, lower_bounds, upper_bounds, orig_mask, pre_relu_indices, lAs, branching_candidates=5, 
            branching_reduceop='min', slopes=None, ):
        from auto_LiRPA.bound_ops import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd
        batch = len(orig_mask[0])
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = self.get_branching_op(branching_reduceop)
        topk = branching_candidates

        score = []
        intercept_tb = []
        relu_idx = -1

        for layer in reversed(self.net.relus):
            ratio = lAs[relu_idx]
            ratio_temp_0, ratio_temp_1 = self.compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
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
        # print(len(score))
        # print([s.shape for s in score])
        # import pdb; pdb.set_trace()
        # exit()
        return score


    def build_the_model_mip_refine(self, lower_bounds, upper_bounds, 
                lr_init_alpha=0.5, share_slopes=False, optimizer="adam",
                loss_reduction_func=reduction_sum, 
                stop_criterion_func=stop_criterion_min(1e-4), lr_decay=0.98,
                score=None, FSB_sort=True, topk_filter=1., mip_multi_proc=None, mip_threads=1, mip_perneuron_refine_timeout=15):
        """
        Before the first branching, we build the model and create a mask matrix
        Output: relu_mask, current intermediate upper and lower bounds, a list of
                indices of the layers right before a Relu layer
                the constructed gurobi model
        NOTE: we keep all bounds as a list of tensors from now on.
              Only lower and upper bounds are kept in the same shape as layers' outputs.
              Mask is linearized
              Gurobi_var lists are lineariezd
              self.model_lower_bounds and self.model_upper_bounds are kepts mainly for
              debugging purpose and could be removed
        """
        new_relu_mask = []
        x = self.x
        input_domain = self.input_domain
        loss_reduction_func = reduction_str2func(loss_reduction_func)

        # preset the args for incomplete full crown with refined bounds
        self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c)
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': 100, 'ob_beta': False, 'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                     'ob_early_stop': False, 'ob_verbose': 0,
                                     'ob_keep_best': True, 'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 'ob_init': False,
                                     'ob_loss_reduction_func': loss_reduction_func, 
                                     'ob_stop_criterion_func': stop_criterion_func, 
                                     'ob_lr_decay': lr_decay}})

        lb_refined, ub_refined = None, None

        # Initialize the model
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam("FeasibilityTol", 2e-5)

        #############
        # Config the hyperparameters for intermeidate bounds refinement with mip

        # default setting for aws instance
        # mip_threads = 1
        # mip_multi_proc = 8
        if mip_multi_proc is None:
            mip_multi_proc = multiprocessing.cpu_count()
            print("preset mip_multi_proc as default setting:", mip_multi_proc)

        self.model.setParam('TimeLimit', mip_perneuron_refine_timeout)
        self.model.setParam('MIPGap', 1e-2)  # Relative gap between primal and dual.
        self.model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between primal and dual.
        self.model.setParam('Threads', mip_threads)
        print(f"mip_multi_proc: {mip_multi_proc}, mip_threads: {mip_threads},"
                f"total threads used: {mip_multi_proc*mip_threads}, mip_perneuron_refine_timeout: {mip_perneuron_refine_timeout}")
        print(f"[total time budget for MIP: {mip_refine_timeout}]\n")

        refine_start_time = time.time()
        #############

        # keep a record of model's information
        self.gurobi_vars = []
        self.relu_constrs = []
        self.relu_indices_mask = []

        ## Do the input layer, which is a special case
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
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
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        row_vars.append(v)
                    chan_vars.append(row_vars)
                inp_gurobi_vars.append(chan_vars)
        self.model.update()

        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        relu_idx = 0
        maximum_refined_relu_layers = 0
        need_refine = True
        global multiprocess_mip_model, mip_refine_time_start
        mip_refine_time_start = time.time()
        # print(len(self.layers), len(self.net.relus), len(lower_bounds))
        last_relu_layer_refined = False
        for layer in self.layers:
            this_layer_refined = False
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                
                # Get the better estimates from KW and Interval Bounds
                # print("linear", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape)
                out_lbs = lower_bounds[relu_idx].squeeze(0)
                out_ubs = upper_bounds[relu_idx].squeeze(0)

                print(layer, relu_idx, layer_idx, out_lbs.shape)
                # import pdb; pdb.set_trace()

                candidates = []
                candidate_neuron_ids = []
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, self.gurobi_vars[-1])

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()

                    v = self.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(lin_expr == v)
                    self.model.update()

                    # if relu_idx == 1 and (out_lb * out_ub < 0):
                    if (relu_idx >= 1 and relu_idx < len(self.net.relus)) and (out_lb * out_ub < 0) and (time.time() - mip_refine_time_start<mip_refine_timeout):
                        candidates.append(v.VarName)
                        candidate_neuron_ids.append(neuron_idx)
                    
                    new_layer_gurobi_vars.append(v)

                if need_refine and (relu_idx >= 1 and relu_idx < len(self.net.relus)) and score is not None and FSB_sort:
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

                if need_refine and (relu_idx >= 1 and relu_idx < len(self.net.relus)) and (time.time() - mip_refine_time_start<mip_refine_timeout):
                    multiprocess_mip_model = self.model
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
                        self.model.update()

                    else:
                        with multiprocessing.Pool(mip_multi_proc) as pool:
                            solver_result = pool.map_async(mip_solver, candidates, chunksize=1)

                            if last_relu_layer_refined and (time.time() - mip_refine_time_start<mip_refine_timeout):
                                print(f"Run alpha-CROWN after refining layer {layer_idx-2} and relu idx {relu_idx-1}")
                                # using refined bounds with init opt crown for the previous optimized bounds
                                new_interval, reference_bounds = {}, {}
                                # for i, layer in enumerate(self.net.relus):
                                # only refined with the second relu layer
                                for i, layer in enumerate(self.net.relus):
                                    # only refined with the relu layers that are refined by mip before
                                    if i>=(maximum_refined_relu_layers+1): break
                                    nd = self.net.relus[i].inputs[0].name
                                    print(i, nd, lower_bounds[i].shape)
                                    new_interval[nd] = [lower_bounds[i], upper_bounds[i]]
                                    reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
                                lb_refined, ub_refined = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-optimized', return_A=False,
                                                            reference_bounds=reference_bounds, bound_upper=False)
                                # lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
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
                        self.model.update()

            elif type(layer) is nn.Conv2d:
                ###########
                # Refine the conv layers as well
                ###########
                # raise NotImplementedError
                assert layer.dilation == (1, 1)
                if relu_idx == 0:
                    pre_lb_size = self.x.shape
                else:
                    pre_lb_size = lower_bounds[relu_idx-1].size()
                out_lbs = lower_bounds[relu_idx]
                out_ubs = upper_bounds[relu_idx]
                # print("conv", layer_idx, relu_idx, lower_bounds[relu_idx].shape, layer.weight.shape, layer.bias.shape)
                gvars_array = np.array(self.gurobi_vars[-1])

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
                                #         lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            if need_refine and (relu_idx >= 1) and (out_lb * out_ub < 0) and (time.time() - mip_refine_time_start<mip_refine_timeout):
                                candidates.append(v.VarName)
                                candidate_neuron_ids.append((out_chan_idx, out_row_idx, out_col_idx))

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

                ####### Comment out the following condition if disable refine for conv layers #######
                if False and need_refine and relu_idx >= 1 and (time.time() - mip_refine_time_start<mip_refine_timeout):
                    multiprocess_mip_model = self.model
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
                    self.model.update()

            elif type(layer) is nn.ReLU:
                new_relu_layer_constr = []
                this_relu = self.net.relus[relu_idx]
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    new_layer_mask = []
                    # print("conv relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    temp = pre_lbs.size()
                    out_chain = temp[0]
                    out_height = temp[1]
                    out_width = temp[2]
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
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

                                    v = self.model.addVar(ub=ub, lb=pre_lb,
                                                          obj=0,
                                                          vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    # binary indicator
                                    a = self.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')

                                    new_relu_layer_constr.append(
                                        self.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                             name=f'ReLU{relu_idx}_{neuron_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx}_{neuron_idx}_a_1'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx}_{neuron_idx}_a_2'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v >= 0, name=f'ReLU{relu_idx}_{neuron_idx}_a_3'))

                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    # this is linear relu
                    pre_lbs = lower_bounds[relu_idx].squeeze(0)
                    pre_ubs = upper_bounds[relu_idx].squeeze(0)
                    # print("linear relu", layer_idx, relu_idx, lower_bounds[relu_idx].shape)
                    new_layer_mask = []
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
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
                            v = self.model.addVar(ub=ub, lb=0,
                                                  obj=0,
                                                  vtype=grb.GRB.CONTINUOUS,
                                                  name=f'ReLU{layer_idx}_{neuron_idx}')
                            # binary indicator
                            a = self.model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{layer_idx}_{neuron_idx}')

                            new_relu_layer_constr.append(
                                self.model.addConstr(pre_var - pre_lb * (1 - a) >= v,
                                                     name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(v >= pre_var, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_1'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_2'))
                            # new_relu_layer_constr.append(
                            #     self.model.addConstr(v >= 0, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_3'))

                            new_layer_mask.append(-1)
                            #v.LB, v.UB = 0, np.inf

                        new_layer_gurobi_vars.append(v)

                new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
                self.relu_constrs.append(new_relu_layer_constr)
                relu_idx += 1

            elif type(layer) == Flatten or "Flatten" in str(type(layer)):
                for chan_idx in range(len(self.gurobi_vars[-1])):
                    for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

            if (time.time() - mip_refine_time_start>=mip_refine_timeout):
                break

        multiprocess_mip_model, mip_refine_time_start = None, None

        self.model.update()
        
        print(f'MIP finished with {time.time() - refine_start_time}s')

        slope_opt = None
        
        primals, duals, mini_inp = None, None, None

        if last_relu_layer_refined:
            print(f"Run final alpha-CROWN after MIP solving on layer {layer_idx-1} and relu idx {relu_idx}")
            # using refined bounds with init opt crown
            new_interval, reference_bounds = {}, {}
            # for i, layer in enumerate(self.net.relus):
            # only refined with the second relu layer
            for i, layer in enumerate(self.net.relus):
                # only refined with the relu layers that are refined by mip before
                if i>=(maximum_refined_relu_layers+1): break
                nd = self.net.relus[i].inputs[0].name
                print(i, nd, lower_bounds[i].shape)
                new_interval[nd] = [lower_bounds[i], upper_bounds[i]]
                reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]
            lb_refined, ub_refined = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-optimized', return_A=False,
                                        reference_bounds=reference_bounds, bound_upper=False)
            print("alpha-CROWN with intermediate bounds improved by MIP:", lb_refined, ub_refined)

        if lb_refined is None:
            return lower_bounds, upper_bounds

        lb_refined, ub_refined, pre_relu_indices = self.get_candidate(self.net, lb_refined, lb_refined + 99)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)
        return lb_refined, ub_refined


    def update_the_model_lp(self, relu_mask, lower_bounds, upper_bounds, decision, choice):
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

        self.model_split = self.copy_model(self.model)
        relu_idx = 0
        # save the split relu layer number in self.replacing_bd_index
        for layer_idx, layer in enumerate(self.layers):
            if type(layer) is nn.ReLU:
                if relu_idx == decision[0]:
                    self.replacing_bd_index = layer_idx
                    break
                relu_idx+=1

        # self.replacing_bd_index = self.pre_relu_indices[decision[0]]
        # reintroduce ub and lb for gurobi constraints
        introduced_constrs = []
        rep_index = self.replacing_bd_index
        rep_relu_idx = decision[0]
        for layer in self.layers[self.replacing_bd_index - 1:]:
            print(self.replacing_bd_index, layer)
            if type(layer) is nn.Linear:
                # print("linear", rep_index, rep_relu_idx, upper_bounds[rep_relu_idx].shape)
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = upper_bounds[rep_relu_idx][0, idx].item()
                    svar.lb = lower_bounds[rep_relu_idx][0, idx].item()

            elif type(layer) is nn.Conv2d:
                conv_ub = upper_bounds[rep_relu_idx].reshape(-1)
                conv_lb = lower_bounds[rep_relu_idx].reshape(-1)
                # print("conv", rep_index, rep_relu_idx, conv_lb.shape)
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = conv_ub[idx].item()
                    svar.lb = conv_lb[idx].item()

            elif type(layer) is nn.ReLU:
                # locate relu index and remove all associated constraints
                pre_lbs = lower_bounds[rep_relu_idx].reshape(-1)
                pre_ubs = upper_bounds[rep_relu_idx].reshape(-1)
                # print("relu", rep_index, rep_relu_idx, pre_lbs.shape)
                for unmasked_idx in self.relu_indices_mask[rep_relu_idx]:
                    pre_lb = pre_lbs[unmasked_idx].item()
                    pre_ub = pre_ubs[unmasked_idx].item()
                    var = self.gurobi_vars[rep_index][unmasked_idx]
                    svar = self.model_split.getVarByName(var.VarName)
                    pre_var = self.gurobi_vars[rep_index - 1][unmasked_idx]
                    pre_svar = self.model_split.getVarByName(pre_var.VarName)

                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        svar.lb = pre_lb
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(pre_svar == svar))
                    elif pre_lb <= 0 and pre_ub <= 0:
                        svar.lb = 0
                        svar.ub = 0
                    else:
                        svar.lb = 0
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(svar >= pre_svar))
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        introduced_constrs.append(self.model_split.addConstr(svar <= slope * pre_svar + bias))
                rep_relu_idx += 1

            elif type(layer) is Flatten:
                pass
            else:
                raise NotImplementedError
            self.model_split.update()
            rep_index += 1

        # compute optimum
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
        target_var = self.gurobi_vars[-1][0]
        target_svar = self.model_split.getVarByName(target_var.VarName)

        self.model_split.update()
        # self.model.reset()
        self.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
        # self.model.setObjective(0, grb.GRB.MINIMIZE)
        self.model_split.optimize()
        # assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success([introduced_constrs], model=self.model_split)

        glb = target_svar.X
        print(f"glb: {glb}")
        lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)

        # get input variable values at which minimum is achieved => the upper bound of the obj lb
        # inp_size = lower_bounds[0].size()
        # mini_inp = torch.zeros(inp_size).to(lower_bounds[0].device)
        # if len(inp_size) == 1:
        #     # This is a linear input.
        #     for i in range(inp_size[0]):
        #         var = self.gurobi_vars[0][i]
        #         svar = self.model_split.getVarByName(var.VarName)
        #         mini_inp[i] = svar.x
        # else:
        #     for i in range(inp_size[0]):
        #         for j in range(inp_size[1]):
        #             for k in range(inp_size[2]):
        #                 var = self.gurobi_vars[0][i][j][k]
        #                 svar = self.model_split.getVarByName(var.VarName)
        #                 mini_inp[i, j, k] = svar.x
        # gub = self.net(mini_inp.unsqueeze(0)).item()
        # print(gub, glb, self.model_split.status)
        # assert gub>=glb, "wrong constraints added, not sound!"

        n_test_points = -1  # Test n_test_points neurons per layer.
        check_final_lp = True 

        LOG = True
        if LOG:
            import datetime
            prefix = datetime.datetime.now().strftime('%m%d_%H%M%S')
            updated_svar, updated_lp_lbs, updated_lp_ubs = [], [], []
            relu_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                rep_index = layer_idx+1
                if type(layer) is nn.Linear:
                    l = open(f"log_{prefix}_lp_dense_{layer_idx}.log", "w")
                    # print(rep_index, len(self.gurobi_vars[rep_index]), upper_bounds[rep_index].reshape(-1).shape)
                    # print(rep_index+1, len(self.gurobi_vars[rep_index+1]), upper_bounds[rep_index+1].reshape(-1).shape)
                    # continue
                    dense_ub = upper_bounds[relu_idx].reshape(-1)
                    dense_lb = lower_bounds[relu_idx].reshape(-1)
                    new_lb_bounds, new_up_bounds = [], []
                    if n_test_points != -1:
                        # select some neurons for testing
                        selected_index = np.random.permutation(len(self.gurobi_vars[rep_index]))[:n_test_points]
                        print('selected neuron for testing: ', selected_index)
                        selected_neurons = [self.gurobi_vars[rep_index][si] for si in selected_index]
                    else:
                        selected_neurons = self.gurobi_vars[rep_index]
                        selected_index = list(range(len(self.gurobi_vars[rep_index])))
                    for idx, var in zip(selected_index, selected_neurons):
                        svar = self.model_split.getVarByName(var.VarName)
                        # svar.ub = dense_ub[idx].item()
                        # svar.lb = dense_lb[idx].item()
                        self.model_split.setObjective(svar, grb.GRB.MINIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
                        glb = svar.X

                        self.model_split.setObjective(svar, grb.GRB.MAXIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
                        gub = svar.X

                        new_lb_bounds.append(glb)
                        new_up_bounds.append(gub)

                        print("Linear {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}".format(var.VarName, dense_lb[idx].item(), glb, dense_ub[idx].item(), gub, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub))
                        l.write("Linear {}: old_lb={:.7g}, new_lb={:.7g}, old_ub={:.7g}, new_ub={:.7g}, lb_diff={:.7g}, ub_diff={:.7g}\n".format(var.VarName, dense_lb[idx].item(), glb, dense_ub[idx].item(), gub, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub))
                        l.flush()
                        # l.write("{}: {}, {}\n".format(var, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub))
                        if check_final_lp:
                            updated_svar.append(svar)
                            updated_lp_lbs.append(glb)
                            updated_lp_ubs.append(gub)
                            # svar.ub, svar.lb = gub, glb
                    l.close()
                    # self.model_split.update()

                elif type(layer) is nn.Conv2d:
                    if n_test_points != -1:
                        # select some neurons for testing
                        selected_index = np.random.permutation(len(self.gurobi_vars[rep_index]))[:n_test_points]
                        print('selected neuron for testing: ', selected_index)
                        selected_neurons = [self.gurobi_vars[rep_index][si] for si in selected_index]
                    else:
                        selected_neurons = self.gurobi_vars[rep_index]
                        selected_index = list(range(len(self.gurobi_vars[rep_index])))
                        continue  # Skip conv layers.
                    l = open(f"log_{prefix}_lp_conv_{layer_idx}.log", "w")
                    # continue
                    conv_ub = upper_bounds[relu_idx].reshape(-1)
                    conv_lb = lower_bounds[relu_idx].reshape(-1)
                    new_lb_bounds, new_up_bounds = [], []
                    for idx, var in zip(selected_index, selected_neurons):
                        # print(var)
                        svar = self.model_split.getVarByName(var.VarName)
                        # print(svar, conv_lb[idx].item()-svar.lb, svar.ub-conv_ub[idx].item())
                        # continue
                        # svar.ub = conv_ub[idx].item()
                        # svar.lb = conv_lb[idx].item()
                        self.model_split.setObjective(svar, grb.GRB.MINIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
                        glb = svar.X

                        self.model_split.setObjective(svar, grb.GRB.MAXIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
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
                            # svar.ub, svar.lb = gub, glb
                        # exit()
                    l.close()
                    # self.model_split.update()

                elif type(layer) is nn.ReLU:
                    # locate relu index and remove all associated constraints
                    relu_idx += 1
                else:
                    pass

            if check_final_lp:
                for si, svar in enumerate(updated_svar):
                    svar.ub, svar.lb = updated_lp_ubs[si], updated_lp_lbs[si]
                print(f"total updated intermediate node bounds: {len(updated_svar)}")
                self.model_split.update()

                relu_idx = 0
                for layer_idx, layer in enumerate(self.layers):
                    if type(layer) is nn.ReLU:
                        for unmasked_idx in self.relu_indices_mask[relu_idx]:
                            var = self.gurobi_vars[layer_idx+1][unmasked_idx]
                            svar = self.model_split.getVarByName(var.VarName)
                            pre_var = self.gurobi_vars[layer_idx][unmasked_idx]
                            pre_svar = self.model_split.getVarByName(pre_var.VarName)
                            pre_lb, pre_ub = pre_svar.lb, pre_svar.ub

                            if pre_lb >= 0 and pre_ub >= 0:
                                # ReLU is always passing
                                svar.lb = pre_lb
                                svar.ub = pre_ub
                                introduced_constrs.append(self.model_split.addConstr(pre_svar == svar))
                            elif pre_lb <= 0 and pre_ub <= 0:
                                svar.lb = 0
                                svar.ub = 0
                            else:
                                svar.lb = 0
                                svar.ub = pre_ub
                                introduced_constrs.append(self.model_split.addConstr(svar >= pre_svar))
                                slope = pre_ub / (pre_ub - pre_lb)
                                bias = - pre_lb * slope
                                introduced_constrs.append(self.model_split.addConstr(svar <= slope * pre_svar + bias))

                            # self.model_split.update()
                            # print(pre_svar, pre_svar.lb, pre_svar.ub, pre_lb, pre_ub)
                            # print(svar, svar.lb, svar.ub)
                            # exit()
                        relu_idx += 1

                self.model_split.update()
                # self.model.reset()
                self.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
                # self.model.setObjective(0, grb.GRB.MINIMIZE)
                self.model_split.optimize()
                # assert self.model.status == 2, "LP wasn't optimally solved"
                self.check_optimization_success([introduced_constrs], model=self.model_split)

                glb = target_svar.X
                print(f"new glb: {glb}")
                l = open(f"log_{prefix}_lp_final.log", "w")
                l.write(f"\n***** new glb: {glb} *****\n\n")
                l.close()

            # exit()
        del self.model_split
        return

    def update_the_model_mip(self, relu_mask, lower_bounds, upper_bounds, decision, choice):
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

        # self.model_split = self.copy_model(self.model)
        self.model_split = self.model.copy()
        # self.model_split.write("mip_save.lp")

        self.replacing_bd_index = self.pre_relu_indices[decision[0]]

        # reintroduce ub and lb for gurobi constraints
        introduced_constrs = []
        rep_index = self.replacing_bd_index
        for layer in self.layers[self.replacing_bd_index - 1:]:
            if type(layer) is nn.Linear:
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = upper_bounds[rep_index][idx].item()
                    svar.lb = lower_bounds[rep_index][idx].item()

            elif type(layer) is nn.Conv2d:
                conv_ub = upper_bounds[rep_index].reshape(-1)
                conv_lb = lower_bounds[rep_index].reshape(-1)
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = conv_ub[idx].item()
                    svar.lb = conv_lb[idx].item()

            elif type(layer) is nn.ReLU:
                # locate relu index and remove all associated constraints
                relu_idx = self.pre_relu_indices.index(rep_index - 1)
                # reintroduce relu constraints
                pre_lbs = lower_bounds[rep_index - 1].reshape(-1)
                pre_ubs = upper_bounds[rep_index - 1].reshape(-1)
                for unmasked_idx in self.relu_indices_mask[relu_idx]:
                    pre_lb = pre_lbs[unmasked_idx].item()
                    pre_ub = pre_ubs[unmasked_idx].item()
                    var = self.gurobi_vars[rep_index][unmasked_idx]
                    svar = self.model_split.getVarByName(var.VarName)
                    sa = self.model_split.getVarByName("a" + var.VarName)
                    pre_var = self.gurobi_vars[rep_index - 1][unmasked_idx]
                    pre_svar = self.model_split.getVarByName(pre_var.VarName)
                    # print(sa, svar, pre_var, pre_svar)

                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        svar.lb = pre_lb
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(pre_svar == svar))
                        relu_mask[relu_idx][unmasked_idx] = 1
                    elif pre_lb <= 0 and pre_ub <= 0:
                        svar.lb = 0
                        svar.ub = 0
                        relu_mask[relu_idx][unmasked_idx] = 0
                    else:
                        svar.lb = 0
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(pre_svar - pre_lb * (1 - sa) >= svar))
                        introduced_constrs.append(self.model_split.addConstr(svar >= pre_svar))
                        introduced_constrs.append(self.model_split.addConstr(pre_ub * sa >= svar))
                        introduced_constrs.append(self.model_split.addConstr(svar >= 0))

            elif type(layer) is Flatten:
                pass
            else:
                raise NotImplementedError
            self.model_split.update()
            rep_index += 1

        # compute optimum
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
        target_var = self.gurobi_vars[-1][0]
        target_svar = self.model_split.getVarByName(target_var.VarName)

        self.model_split.update()
        # self.model.reset()
        self.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
        # self.model.setObjective(0, grb.GRB.MINIMIZE)
        self.model_split.optimize()
        # assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success([introduced_constrs], model=self.model_split)

        glb = target_svar.X
        print(f"mip glb: {glb}")
        lower_bounds[-1] = torch.tensor([glb]).to(lower_bounds[0].device)

        # get input variable values at which minimum is achieved
        inp_size = lower_bounds[0].size()
        mini_inp = torch.zeros(inp_size).to(lower_bounds[0].device)
        if len(inp_size) == 1:
            # This is a linear input.
            for i in range(inp_size[0]):
                var = self.gurobi_vars[0][i]
                svar = self.model_split.getVarByName(var.VarName)
                mini_inp[i] = svar.x

        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        var = self.gurobi_vars[0][i][j][k]
                        svar = self.model_split.getVarByName(var.VarName)
                        mini_inp[i, j, k] = svar.x

        LOG = False

        if LOG:
            l = open("log_dense_mip", "w")

            lp_lbs, lp_ubs = [], []
            # print(len(self.gurobi_vars), len(upper_bounds), len(self.layers))
            for layer_idx, layer in enumerate(self.layers):
                rep_index = layer_idx + 1
                if type(layer) is nn.Linear:
                    # print(rep_index, len(self.gurobi_vars[rep_index]), upper_bounds[rep_index].reshape(-1).shape)
                    # print(rep_index+1, len(self.gurobi_vars[rep_index+1]), upper_bounds[rep_index+1].reshape(-1).shape)
                    # continue
                    dense_ub = upper_bounds[rep_index].reshape(-1)
                    dense_lb = lower_bounds[rep_index].reshape(-1)
                    new_lb_bounds, new_up_bounds = [], []
                    for idx, var in enumerate(self.gurobi_vars[rep_index]):
                        # if "lay8" not in var.VarName: continue
                        svar = self.model_split.getVarByName(var.VarName)

                        svar.ub = upper_bounds[rep_index][idx].item()
                        svar.lb = lower_bounds[rep_index][idx].item()

                        svar = self.model_split.getVarByName(var.VarName)
                        svar.ub = dense_ub[idx].item()
                        svar.lb = dense_lb[idx].item()
                        self.model_split.setObjective(svar, grb.GRB.MINIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
                        glb = svar.X

                        self.model_split.setObjective(svar, grb.GRB.MAXIMIZE)
                        self.model_split.update()
                        self.model_split.optimize()
                        gub = svar.X

                        new_lb_bounds.append(glb)
                        new_up_bounds.append(gub)

                        print(var, dense_lb[idx].item(), glb, gub, dense_ub[idx].item(), glb - dense_lb[idx].item(),
                              dense_ub[idx].item() - gub)
                        l.write("{}: {}, {}, {}, {}, {}, {}\n".format(var, dense_lb[idx].item(), glb, gub,
                                                                      dense_ub[idx].item(), glb - dense_lb[idx].item(),
                                                                      dense_ub[idx].item() - gub))
                        # l.write("{}: {}, {}\n".format(var, glb-dense_lb[idx].item(), dense_ub[idx].item()-gub))

                elif type(layer) is nn.Conv2d:
                    raise NotImplementedError
                else:
                    lp_lbs.append([])
                    lp_ubs.append([])

            exit()
        del self.model_split
        return


multiprocess_mip_model = None
stop_multiprocess = False
mip_refine_time_start = None
mip_refine_timeout = 230

def set_mip_refine_timeout(timeout):
    global mip_refine_timeout
    mip_refine_timeout = timeout
    # print(f"[reset mip_refine_timeout to be {mip_refine_timeout}]\n")

def mip_solver(candidate):

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


def mip_solver_lb(candidate):
    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    out_lb = v.LB
    global stop_multiprocess
    if stop_multiprocess: return out_lb
    refine_time = time.time()
    model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
    model.setObjective(v, grb.GRB.MINIMIZE)
    try:
        model.optimize()
    except grb.GurobiError as e: 
        handle_gurobi_error(e.message)
    if model.status == 9:
        # Timed out. Get current bound.
        vlb = max(model.objbound, out_lb)
    elif model.status == 2:
        vlb = model.objval
    elif model.status == 15:
        # We have find an lower bound >= 0, so this neuron becomes stable.
        vlb = 1e-5
    else:
        vlb = out_lb
    if vlb<0:
        stop_multiprocess = True

    assert model.status != 3, "should not be infeasible"
    print("solving MIP for {}, [{}]=>[{}], time: {}s".format(v.VarName, out_lb, vlb,
            time.time()-refine_time))
    sys.stdout.flush()
    return vlb, model.status
