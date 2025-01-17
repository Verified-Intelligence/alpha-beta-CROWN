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
"""Interface of cutting plane methods, using GCP-CROWN as the solver."""
import itertools
import torch
from auto_LiRPA.bound_ops import *


class Cutter:
    def __init__(self, solver, A=None, x=None, number_cuts=50, fix_intermediate_bounds=False, device='cuda'):
        self.solver = solver
        self.update_net(solver.net)
        input_name = str(list(solver.net.input_name)[0])
        assert len([input_name]) == 1
        self.A = [a[input_name] for a in A.values()] if A is not None else None
        self.x = x
        self.number_cuts = number_cuts
        self.fix_intermediate_bounds = fix_intermediate_bounds
        self.device = device
        self.unstable_idx_list, self.lower, self.upper = [], [], []
        self.lAs, self.uAs, self.lbs, self.ubs = [], [], [], []
        self.log_interval = 10
        self.beta_init = 0
        # Cutter should keep track of cuts
        self.cuts = []
        self.cut_module = None
        self.cut_timestamp = -1
        self.num_relus = len(self.relus)

    def update_net(self, net):
        self.net = net
        self.relus = net.relus

    def construct_cut_module(self, start_nodes=None, use_patches_cut=False):
        num_cuts = len(self.cuts)

        self.net.cut_used = False
        for m in self.relus:
            m.masked_beta = None
            # m.split_beta_used: True if any of this layer node is used in the new split constraints
            m.split_beta_used = False
            # m.history_beta_used: True if any of this layer node is used in the history constraints
            m.history_beta_used = False
            # if any of the current/history split constraints are single node constraint,
            m.single_beta_used = False
            # There are nodes used in cuts in this layer if cut_used=True
            m.cut_used = False
            m.relu_coeffs = m.arelu_coeffs = m.pre_coeffs = None

        self.start_nodes = start_nodes = [
            relu_layer.inputs[0] for relu_layer in self.net.relus] if not self.fix_intermediate_bounds else []
        start_nodes.append(self.net[self.net.output_name[0]])

        # init cut_module
        self.cut_module = cut_module = CutModule(self.relus)
        # maximum layer of the neuron in each constr
        # -1 is the input layer, 0 means the layer corresponding to the first relu layer
        max_layer_idx_in_constr = []
        cut_used = False
        self.pre_layers, self.relu_layers, self.arelu_layers = [], [], []
        self.use_x_cuts = False
        for cut_idx, ci in enumerate(self.cuts):
            # c = ci["c"] # 1: >0; -1: <0
            all_decisions = (ci["x_decision"] + ci["relu_decision"] +
                ci["arelu_decision"] + ci["pre_decision"])
            if len(all_decisions) > 0:
                max_layer_idx_in_constr.append(max([item[0] for item in all_decisions]))
            else:
                max_layer_idx_in_constr.append(-1)
            if len(ci["x_decision"]) > 0:
                cut_used = self.use_x_cuts = True
            self.relu_layers.extend([ self.relus[item[0]] for item in ci["relu_decision"]])
            self.arelu_layers.extend([ self.relus[item[0]] for item in ci["arelu_decision"]])
            self.pre_layers.extend([ self.relus[item[0]] for item in ci["pre_decision"]])
        self.pre_layers = set(self.pre_layers)
        self.relu_layers = set(self.relu_layers)
        self.arelu_layers = set(self.arelu_layers)
        for node in itertools.chain(self.relu_layers, self.arelu_layers, self.pre_layers):
            cut_used = node.cut_used = True
        if cut_used:
            self.net.cut_used = True

        # config active cuts
        print("all start nodes to check full crown or not:", start_nodes)
        print("use patches cut: ", use_patches_cut)
        cut_module.active_cuts = {}
        for node_idx, start_node in enumerate(start_nodes):
            active_cuts = []
            # use_patches_cut: disable to optimize intermediate layer that is in patches mode
            if not use_patches_cut and hasattr(start_node, "mode") and start_node.mode=="patches":
                print("skip cut beta crown opt for patches layer:", start_node)
                cut_module.active_cuts[start_node.name] = torch.tensor(active_cuts, device=self.device).long()
                continue
            if node_idx + 1 == len(start_nodes):
                cut_module.active_cuts[start_node.name] = torch.arange(num_cuts, device=self.device)
            else:
                for cut_idx in range(num_cuts):
                    # the constraints are active only if they have all constraint neurons before start node
                    # i.e., the max layer node in the constraint should be smaller than the current start node
                    if (max_layer_idx_in_constr[cut_idx] == -1 or
                            start_nodes.index(self.relus[max_layer_idx_in_constr[cut_idx]].inputs[0]) < node_idx):
                        active_cuts.append(cut_idx)
                cut_module.active_cuts[start_node.name] = torch.tensor(active_cuts, device=self.device).long()

        if self.use_x_cuts:
            assert self.x.size(0) == 1  # do not have batch dimension.
            self.cut_module.x_coeffs = torch.zeros((num_cuts, self.x.numel()), device=self.device)
        #FIXME maybe just use self.relus
        for node in self.relu_layers:
            cut_module.relu_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)
        for node in self.arelu_layers:
            cut_module.arelu_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)
        for node in self.pre_layers:
            cut_module.pre_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)

        self.update_cut_module()

        return cut_module

    def construct_beta(self, shapes):
        general_betas = {}
        start_node_shape = {}
        for i, node in enumerate(self.net.relus):
            start_node_shape[node.inputs[0].name] = shapes[i]
        start_node_shape[self.net.final_name] = shapes[-1]
        # current all start nodes, only final_name if crown-ibp
        # all_start_nodes = self.net.backward_from[self.net.optimizable_activations[0].name]
        # manually get all start nodes, self.net.backward_from might not include first pre-relu layer
        # all_start_nodes = [relu_layer.inputs[0] for relu_layer in nodes] + [self.net[self.net.output_name[0]]]
        for start_node in self.start_nodes:
            # each general_beta: 2 (lA, uA), spec (out_c, out_h, out_w), batch, num_cuts
            general_betas[start_node.name] = self.beta_init * torch.ones(
                (2, *start_node_shape[start_node.name][1:], 1, len(self.cuts)), device=self.device)
            general_betas[start_node.name] = general_betas[start_node.name].detach()
            general_betas[start_node.name].requires_grad = True
        self.cut_module.general_beta = general_betas
        self.net.cut_beta_params = []
        for start_node in self.start_nodes:
            self.net.cut_beta_params.append(self.cut_module.general_beta[start_node.name])

    def init_cut(self, c=1):
        return {
            'x_decision': [], 'x_coeffs': [], 'arelu_decision': [], 'arelu_coeffs': [],
            'relu_decision': [], 'relu_coeffs': [], 'pre_decision': [], 'pre_coeffs': [],
            'c': c
        }

    def update_cut_module(self):
        # config the cut constraints coeffs in cut_module
        # add cut into each relu layer
        cut_bias = []
        for cut_idx, ci in enumerate(self.cuts):
            c = ci["c"] # 1: >0; -1: <0
            for node, coeff in zip(ci["x_decision"], ci["x_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.x_coeffs[cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["relu_decision"], ci["relu_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.relu_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["arelu_decision"], ci["arelu_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.arelu_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["pre_decision"], ci["pre_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.pre_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            ##### have to be careful, c * bias (c=1) if constraint > bias;
            cut_bias.append(c * ci["bias"])
            #FIXME bias should be on torch
        self.cut_module.cut_bias = torch.tensor(cut_bias, device=self.device)

    def update_cuts(self):
        pass

    def refine_cuts(self):
        pass
