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
from collections import defaultdict

import torch
from torch.nn import ZeroPad2d

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.utils import reduction_max, stop_criterion_max
from auto_LiRPA.perturbations import *
from model_defs import Flatten

try:
    import gurobipy as grb
except ModuleNotFoundError:
    pass

total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_finalize_time = 0.0


def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
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
            new_all_layers.append(joint_layer)
    return new_all_layers


def add_single_prop(layers, gt, cls):
    """
    gt: ground truth lablel
    cls: class we want to verify against
    """
    if gt is not None:
        additional_lin_layer = nn.Linear(layers[-1].out_features, 1, bias=True)
        lin_weights = additional_lin_layer.weight.data
        lin_weights.fill_(0)
        lin_bias = additional_lin_layer.bias.data
        lin_bias.fill_(0)
        lin_weights[0, cls] = -1
        lin_weights[0, gt] = 1

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
                 conv_mode='patches', c=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        assert type(list(model_ori.modules())[-1]) is torch.nn.Linear
        self.c = c
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        if simplify:
            added_prop_layers = add_single_prop(layers, pred, test)
            self.layers = added_prop_layers
            net._modules[list(model_ori._modules)[-1]] = self.layers[-1]  # replace the lase layer
        else:
            self.layers = layers
        self.solve_slope = solve_slope
        if solve_slope:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'adaptive', 'conv_mode': conv_mode},
                                     device=device)
        else:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'adaptive', 'conv_mode': conv_mode}, device=device)


        """
        l = torch.tensor([[ 0.6,         0.,          0.4375,      0.45,       -0.46249998]], device='cuda')
        u=torch.tensor([[ 0.6798578,  0.0078125,  0.46875,    0.5,       -0.45     ]], device='cuda')

        # old_l=torch.tensor([[ 0.6,     0.,      0.4375,  0.45,   -0.465 ]], device='cuda')
        old_l=torch.tensor([[ 0.6,     0.,      0.4375,  0.45,   -0.464 ]], device='cuda')
        old_u=torch.tensor([[ 0.6798578,  0.0078125,  0.46875,    0.5,       -0.45     ]], device='cuda')

        dm_l = old_l
        print(dm_l)
        dm_u = old_u
        C = self.c

        ptb = PerturbationLpNorm(norm=float("inf"), eps=None, x_L=dm_l, x_U=dm_u)
        new_x = BoundedTensor((dm_l + dm_u)/2., ptb)
        self.net(new_x)
        lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=C, method='CROWN-Optimized', new_interval=None, bound_upper=False, return_A=False)
        print(lb)
        lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=C, method='CROWN', new_interval=None, bound_upper=False, return_A=False)
        print(lb)

        import pdb; pdb.set_trace()
        """

        model_ori.to(device)
        self.net.eval()

    def get_lower_bound_naive(self, pre_lb_all=None, pre_ub_all=None, dm_l=None, dm_u=None, slopes=None,
                              shortcut=False, lr_alpha=0.01, iteration=10, branching_candidates=3):

        batch = len(dm_l)//2
        selected_dims = [None] * (batch * 2)

        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps, x_L=dm_l, x_U=dm_u)
        new_x = BoundedTensor(self.x.data.repeat(batch * 2, *([1] * (self.x.ndim - 1))), ptb)
        C = self.c.expand([batch*2, *self.c.shape[1:]]) if self.c is not None else None

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes)

        # new_candidate = {}
        # for i, (l, uc, lc, u) in enumerate(zip(lower_bounds, upper_bounds_cp, lower_bounds_cp, upper_bounds)):
        #     # we set lower = 0 in first half batch, and upper = 0 in second half batch
        #     new_candidate[self.name_dict[i]] = [torch.cat((l, lc), dim=0), torch.cat((uc, u), dim=0)]

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': False, 'ob_single_node_split': True,
                                                             'ob_update_by_layer': True}})
            with torch.no_grad():
                # FULL CROWN
                lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=C, method='backward',
                                                 new_interval=None, bound_upper=False, return_A=False)
            return lb

        if self.solve_slope:
            if self.input_grad:
                new_x.ptb.x_L.requires_grad_(True)
                new_x.ptb.x_U.requires_grad_(True)

            self.net.set_bound_opts({'optimize_bound_args':
                                     {'ob_beta': False, 'ob_single_node_split': True,
                                      'ob_update_by_layer': True, 'ob_iteration': iteration,
                                      'ob_lr': lr_alpha, 'ob_input_grad': self.input_grad}})
            lb, ub = self.net.compute_bounds(x=(new_x,), IBP=False, C=C, method='CROWN-Optimized', return_A=False,
                                             bound_upper=False)

            if self.input_grad:
                selected_dims = torch.cat([new_x.ptb.x_L.grad.flatten(1).argsort()[:, :branching_candidates],
                                           new_x.ptb.x_U.grad.flatten(1).argsort()[:, -branching_candidates:]], 1)

        else:
            # just use CROWN to calculate bound
            for m in self.net.relus:
                m.relu_options = "adaptive"

            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': False, 'ob_single_node_split': True,
                                                             'ob_update_by_layer': True}})

            with torch.no_grad():
                # FULL CROWN
                lb, _, = self.net.compute_bounds(x=(new_x,), IBP=False, C=C, method='backward',
                                                 new_interval=None, bound_upper=False, return_A=False)

        if len(slopes) > 0:
            ret_s = self.get_slope(self.net)
        else:
            ret_s = [None] * (batch * 2)

        # pre_lb_all = [torch.cat(2 * [i]) for i in pre_lb_all]
        # pre_ub_all = [torch.cat(2 * [i]) for i in pre_ub_all]

        # lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)
        # lower_bounds_new = [torch.max(lower_bounds_new[i], pre_lb_all[i]) for i in range(len(pre_lb_all))]
        # upper_bounds_new = [torch.min(upper_bounds_new[i], pre_ub_all[i]) for i in range(len(pre_ub_all))]

        # ret_l = [[] for _ in range(batch * 2)]
        # ret_u = [[] for _ in range(batch * 2)]
        # for i in range(batch):
        #     ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
        #     ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]
        #
        #     ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
        #     ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]

        lb = lb.max(1).values
        return lb, lb, ret_s, selected_dims

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
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch, -1).detach())
        upper_bounds.append(ub.view(batch, -1).detach())

        return lower_bounds, upper_bounds

    def get_mask_lA_parallel(self, model):
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons
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
            lA.append(this_relu.lA.detach().squeeze(0))

        ret_mask, ret_lA = [], []
        for i in range(mask[0].size(0)):
            ret_mask.append([j[i:i+1] for j in mask])
            ret_lA.append([j[i:i+1] for j in lA])
        return ret_mask, ret_lA

    def get_slope(self, model):
        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.
        batch_size = next(iter(model.relus[0].alpha.values())).size(2)
        ret = [defaultdict(dict) for i in range(batch_size)]
        for m in model.relus:
            for spec_name, alpha in m.alpha.items():
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:].clone().detach()
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

    def build_the_model(self, input_domain, x, decision_thresh=0, lr_init_alpha=0.5,
                        share_slopes=False, input_grad=False,  shape=None,):
        self.x = x
        self.input_domain = input_domain
        self.shape = shape
        self.input_grad = input_grad
        print('calculate grad on input:', self.input_grad)

        slope_opt, selected_dims = None, [None]

        # first get CROWN bounds
        if self.solve_slope:
            self.net.init_slope((self.x, ), share_slopes=share_slopes,c=self.c)
            if self.input_grad:
                x.ptb.x_L.requires_grad_(True)
                x.ptb.x_U.requires_grad_(True)
            self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': 10, 'ob_beta': False, 'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 'ob_opt_choice': "adam",
                                     'ob_early_stop': False, 'ob_verbose': 0,
                                     'ob_keep_best': True, 'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 'ob_init': False,
                                     'ob_loss_reduction_func': reduction_max,
                                     'ob_stop_criterion_func': stop_criterion_max(0),
                                     'ob_input_grad': self.input_grad}})
            lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
                                             bound_upper=False)
            slope_opt = self.get_slope(self.net)[0]  # initial with one node only
            if self.input_grad:
                selected_dims = torch.cat([x.ptb.x_L.grad.flatten(1).argsort()[:, :3], x.ptb.x_U.grad.flatten(1).argsort()[:, -3:]], 1)
        else:
            with torch.no_grad():
                lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='backward', return_A=False)

        # build a complete A_dict
        # self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
        # self.layer_names.sort()

        # update bounds
        print('initial CROWN bounds:', lb, ub)
        mini_inp = None
        # mini_inp, primals = self.get_primals(self.A_dict)
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        # mask, lA = self.get_mask_lA_parallel(self.net)

        return ub[-1], lb[-1], mini_inp, lb, ub, pre_relu_indices, slope_opt, x.ptb.x_L.detach(), x.ptb.x_U.detach(), selected_dims
