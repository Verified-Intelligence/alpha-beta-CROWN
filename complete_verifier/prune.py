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
"""Pruning: After CROWN pass, prune verified OR specs before starting the alpha-CROWN pass."""

import torch

import arguments

from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any, stop_criterion_general

class PruneAfterCROWN:
    def __init__(self, net, x, c, rhs, lb, aux_reference_bounds, stop_criterion, or_spec_size=None):
        self.net = net

        # Step 1: Set unverified indices based on the stop criterion function and lb.
        optimize_disjuncts_separately = arguments.Config["solver"]["optimize_disjuncts_separately"]

        if stop_criterion == stop_criterion_batch_any:
            assert optimize_disjuncts_separately or (rhs.shape[0] == lb.shape[0] == 1)
            unverified_or_mask = unverified_mask = ~stop_criterion(rhs)(lb).squeeze(1)
            prune_dim = 0
        elif stop_criterion == stop_criterion_all:
            assert (not optimize_disjuncts_separately) and (rhs.shape[0] == lb.shape[0] == 1)
            unverified_or_mask = unverified_mask = (lb <= rhs).squeeze(0)
            prune_dim = 1
        elif stop_criterion == stop_criterion_general:
            assert (not optimize_disjuncts_separately) and (rhs.shape[0] == lb.shape[0] == 1)
            assert or_spec_size is not None
            stop_criterion_per_or = stop_criterion_general(or_spec_size, rhs).__closure__[0].cell_contents(lb)
            unverified_or_mask = ~stop_criterion_per_or.squeeze(0)
            unverified_mask = torch.repeat_interleave(unverified_or_mask, or_spec_size)
            prune_dim = 1
        else:
            raise ValueError(f'Unknown stop criterion function.')

        self.optimize_disjuncts_separately = optimize_disjuncts_separately
        self.unverified_indices = unverified_mask.nonzero().view(-1)
        self.prune_dim = prune_dim
        self.prune_dim_alpha = 2 - prune_dim
        # alpha is (2, spec, batch, *shape), so we prune along the other dimension.
        # if prune_dim == 0, prune alpha along dim 2; if prune_dim == 1, prune alpha along dim 1.

        # Step 2: Prune everything based on the unverified indices.
        self.lb_ori = lb
        self.rhs_ori = rhs

        if optimize_disjuncts_separately:
            # optimize_disjuncts_separately
            # Prune input bounds.
            x.ptb.x_L = x.ptb.x_L.index_select(prune_dim, self.unverified_indices)
            x.ptb.x_U = x.ptb.x_U.index_select(prune_dim, self.unverified_indices)
            x.data = x.data.index_select(prune_dim, self.unverified_indices)
        
        self.x_pruned = x
        self.c_pruned = c.index_select(prune_dim, self.unverified_indices)
        self.rhs_pruned = self.rhs_ori.index_select(prune_dim, self.unverified_indices)
        self.or_spec_size_pruned = None
        if or_spec_size is not None:
            # or_spec_size: [num_or]. or_spec_size_pruned: [num_unverified_or].
            self.or_spec_size_pruned = or_spec_size[unverified_or_mask]
        print('prune_after_crown optimization in use')
        print(f'  original spec size = {self.rhs_ori.shape}')
        print(f'  pruned spec size = {self.rhs_pruned.shape[:2]}')

        # update bounds
        final_name = net.final_name
        if optimize_disjuncts_separately:
            # Prune intermediate bounds.
            for node, reference_bounds in aux_reference_bounds.items():
                aux_reference_bounds[node][0] = reference_bounds[0].index_select(
                    self.prune_dim, self.unverified_indices)
                aux_reference_bounds[node][1] = reference_bounds[1].index_select(
                    self.prune_dim, self.unverified_indices)

        optimizable_activations = net.get_enabled_opt_act()
        for m in optimizable_activations:
            # When the intermediate alphas are not shared,
            # Prune all the intermediate alphas.
            for spec_name, alpha in m.alpha.items():
                if optimize_disjuncts_separately or spec_name == final_name:
                    m.alpha[spec_name] = alpha.index_select(
                        self.prune_dim_alpha, self.unverified_indices).detach()

        # Step 3: Set the stop criterion function for the pruned indices.
        if stop_criterion == stop_criterion_batch_any:
            self.stop_criterion_func_pruned = stop_criterion_batch_any(self.rhs_pruned)
        elif stop_criterion == stop_criterion_all:
            self.stop_criterion_func_pruned = stop_criterion_all(self.rhs_pruned)
        elif stop_criterion == stop_criterion_general:
            self.stop_criterion_func_pruned = stop_criterion_general(self.or_spec_size_pruned, self.rhs_pruned)
        else:
            raise ValueError(f'Unknown stop criterion function.')

    def get_pruned_data(self):
        """Get pruned data."""
        return self.x_pruned, self.c_pruned, self.rhs_pruned, self.or_spec_size_pruned, self.stop_criterion_func_pruned

    def _recover_data(self, data, full_size, fill_value, recover_dim):
        """Recover full shape data."""
        full_shape = list(data.shape)
        full_shape[recover_dim] = full_size
        new_data = torch.full(full_shape, fill_value, device=data.device, dtype=data.dtype)
        new_data.index_copy_(recover_dim, self.unverified_indices, data)
        return new_data

    @torch.no_grad()
    def recover(self, lb, ub, lA, alphas, mask, input_split_idx, full_alpha_info):
        prune_dim = self.prune_dim
        final_name = self.net.final_name
        optimize_disjuncts_separately = self.optimize_disjuncts_separately
        batch_size_ori = self.lb_ori.shape[prune_dim]
        for node in lb.keys():
            if optimize_disjuncts_separately or node == final_name:
                lb[node] = self._recover_data(lb[node], batch_size_ori, float('inf'), prune_dim)
                ub[node] = self._recover_data(ub[node], batch_size_ori, float('inf'), prune_dim)

        # handle lA
        for node, Aitem in lA.items():
            lA[node] = self._recover_data(Aitem, batch_size_ori, 0, prune_dim)
        # handle alphas
        prune_dim_alpha = self.prune_dim_alpha
        if full_alpha_info:
            for node, v in alphas.items():
                for spec_name, alpha in v['alpha'].items():
                    if optimize_disjuncts_separately or spec_name == final_name:
                        alphas[node]['alpha'][spec_name] = self._recover_data(alpha, batch_size_ori, 0, prune_dim_alpha)
        else:
            for node, v in alphas.items():
                for spec_name, alpha in v.items():
                    if optimize_disjuncts_separately or spec_name == final_name:
                        alphas[node][spec_name] = self._recover_data(alpha, batch_size_ori, 0, prune_dim_alpha)

        for node, m_list in mask.items():
            if optimize_disjuncts_separately or node == final_name:
                for i, m in enumerate(m_list):
                    mask[node][i] = self._recover_data(m, batch_size_ori, False, prune_dim)

        for node, v in input_split_idx.items():
            if optimize_disjuncts_separately or node == final_name:
                input_split_idx[node] = self._recover_data(v, batch_size_ori, 0, prune_dim)


def prune_alphas(alpha, kept_names):
    if alpha is None:
        return None
    print(f'Keeping alphas for these layers: {kept_names}')
    new_alpha = {}
    for node, alphas in alpha.items():
        new_alpha[node] = {}
        for spec_name, v in alphas.items():
            if spec_name in kept_names:
                new_alpha[node][spec_name] = v
    return new_alpha
