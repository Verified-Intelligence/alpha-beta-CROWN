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
"""Helper for the BaBSR-like heuristic on general nonlinear functions."""
import numpy as np

import torch

from collections import deque
from auto_LiRPA.backward_bound import get_degrees, add_constant_node, add_bound
from auto_LiRPA.bound_ops import *


class BaBSRNonlinearBranching:
    def __init__(self, net, num_branches):
        self.net = net
        self.model = net.net
        self.num_branches = num_branches

    def compute_heuristic(self, node, points, domains, margin_before):
        name = node.name
        num_neurons = np.prod(points.shape[1:-1])
        lAs = domains['lAs']
        lb, ub = domains['lower_bounds'], domains['upper_bounds']
        lb_ori, ub_ori = lb[name], ub[name]
        batch_size = lb_ori.shape[0]
        start_nodes = [act[0] for act in self.net.split_activations[name]]

        bounds_before = self.get_partial_bounds_batch(lb, ub, lAs, start_nodes)
        bounds_before = bounds_before.reshape(batch_size, -1)

        branched_neurons = torch.arange(
            num_neurons, device=bounds_before.device
        ).expand(batch_size, num_neurons)

        def _repeat(v):
            return v.repeat(num_neurons, *([1]*(v.ndim-1)))

        lAs = {k: _repeat(v) for k, v in lAs.items()
                if self.net.net[k] in start_nodes}
        lb = {k: _repeat(v) for k, v in lb.items()
            if k != name and k in self.interm_bound_required}
        ub = {k: _repeat(v) for k, v in ub.items()
            if k != name and k in self.interm_bound_required}
        margin_after = None
        for i in range(self.num_branches):
            lb_branched = lb_ori if i == 0 else points[..., i - 1]
            ub_branched = ub_ori if i == self.num_branches - 1 else points[..., i]
            bounds_after = self._get_bounds_after_branching(
                branched_neurons, lAs, lb, ub, lb_ori, ub_ori,
                lb_branched, ub_branched, name, start_nodes)
            margin_after_ = (
                margin_before
                + (bounds_after - bounds_before.unsqueeze(1))).amax(dim=-1)
            if i == 0:
                margin_after = margin_after_
            else:
                margin_after = torch.min(margin_after, margin_after_)
        return {
            'scores': margin_after - margin_before.amax(dim=-1),
            'points': points
        }

    def _get_bounds_after_branching(
            self, branched_neurons, lAs, lb, ub,
            lb_ori, ub_ori, lb_branched, ub_branched,
            node_name, start_nodes):
        batch_size, num_selected_neurons = branched_neurons.shape
        bounds_after = []

        def _branch(ori, branched, index):
            num_selected_neurons = index.shape[1]
            repeated = ori.repeat(
                num_selected_neurons, *([1]*(ori.ndim - 1))
            ).reshape(num_selected_neurons, *ori.shape)
            repeated = repeated.transpose(0, 1).reshape(
                batch_size, num_selected_neurons, -1)
            branched_ = torch.gather(branched.view(batch_size, -1),
                                     index=index, dim=-1)
            repeated.scatter_(dim=2, index=index.unsqueeze(-1),
                              src=branched_.unsqueeze(-1))
            return repeated.transpose(0, 1).reshape(-1, *branched.shape[1:])

        lAs_ = {k: v for k, v in lAs.items()}
        lb_ = {k: v for k, v in lb.items()}
        ub_ = {k: v for k, v in ub.items()}
        lb_[node_name] = _branch(lb_ori, lb_branched, branched_neurons)
        ub_[node_name] = _branch(ub_ori, ub_branched, branched_neurons)

        bounds_after.append(self.get_partial_bounds_batch(
            lb_, ub_, lAs_, start_nodes))
        bounds_after = torch.concat(bounds_after, dim=0)
        bounds_after = bounds_after.view(num_selected_neurons, batch_size, -1)
        bounds_after = bounds_after.transpose(0, 1)

        return bounds_after

    @torch.no_grad()
    def get_partial_bounds_batch(self, lb, ub, lAs, start_nodes):
        self.interm_bound_required = set()
        batch_size = next(iter(lb.values())).shape[0]
        x = self.net.expand_x(batch_size, lb=lb, ub=ub)
        interm_bounds = {k: (lb[k], ub[k]) for k in lb}
        self.model.set_input(x, interm_bounds=interm_bounds)
        for node in self.model.nodes():
            if not node.perturbed:
                fv = self.model.get_forward_value(node)
                node.interval = node.lower, node.upper = fv, fv

        for node in self.model.nodes():
            node.lA = node.uA = None
        start_nodes = list(set(start_nodes))
        degree_out = get_degrees(start_nodes)
        for node in start_nodes:
            node.lA = lAs[node.name].transpose(0, 1)
        for k in lb:
            self.model[k].lower = lb[k]
            self.model[k].upper = ub[k]
        lb = ub = torch.tensor(0., device=x.device)

        queue = deque(list(set(start_nodes)))
        while len(queue) > 0:
            node = queue.popleft()  # backward from l
            if node.name in self.model.root_names: continue

            # if all the succeeds are done, then we can turn to this node in the
            # next iteration.
            for node_pre in node.inputs:
                degree_out[node_pre.name] -= 1
                if degree_out[node_pre.name] == 0:
                    queue.append(node_pre)

            # Initially, l.lA or l.uA will be set to C for this node.
            if node.lA is not None or node.uA is not None:
                if not node.perturbed:
                    if not hasattr(node, 'forward_value'):
                        self.model.get_forward_value(node)
                    lb, ub = add_constant_node(lb, ub, node)
                    continue
                if node.zero_uA_mtx and node.zero_lA_mtx:
                    # A matrices are all zero, no need to propagate.
                    continue
                lb = lb + self._backward_propagate(node, start_nodes)

        assert lb.ndim == 2
        lb = lb.transpose(0, 1)

        return lb

    def _backward_propagate(self, node, start_nodes):
        for i in node.requires_input_bounds:
            self.interm_bound_required.add(node.inputs[i].name)
        A, lower_b, _ = node.bound_backward(
            node.lA, None, *node.inputs,
            start_node=self.model[self.model.final_name], start_shape=1)
        lb = lower_b

        if isinstance(lower_b, torch.Tensor) and node not in start_nodes:
            # For the BaBSR-like heuristic, stop further back
            # propagating, once we get some bias term from the layer
            # before the activation (typically a linear layer).
            return lb
        else:
            for i, node_pre in enumerate(node.inputs):
                add_bound(node, node_pre, lA=A[i][0])

        return lb
