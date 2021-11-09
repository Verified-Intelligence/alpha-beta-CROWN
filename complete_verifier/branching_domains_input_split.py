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
import bisect
import copy
import torch


class InputDomain:
    '''
    Object representing a domain as produced by the BranchAndBound algorithm.
    Comparison between its elements is based on the values of the lower bounds
    that are estimated for it.
    '''
    def __init__(self, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, slope=None, dm_l=None, dm_u=None,
                 selected_dims=None, device='cuda'):
        self.lower_bound = lb
        self.upper_bound = ub
        self.dm_l = dm_l
        self.dm_u = dm_u
        # self.lower_all = lb_all
        # self.upper_all = up_all
        self.slope = slope
        self.selected_dims = selected_dims
        self.device = device

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def __repr__(self):
        string = f"[LB: {self.lower_bound:.4e}\t" \
                 f" UB:  {self.upper_bound:.4e}\n"
        return string

    def to_cpu(self):
        if self.device == 'cuda':
            return self
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.dm_l = self.dm_l.to(device='cpu', non_blocking=True)
        self.dm_u = self.dm_u.to(device='cpu', non_blocking=True)
        self.lower_bound = self.lower_bound.to(device='cpu', non_blocking=True)
        self.upper_bound = self.upper_bound.to(device='cpu', non_blocking=True)
        # self.lower_all = [lbs.to(device='cpu', non_blocking=True) for lbs in self.lower_all]
        # self.upper_all = [ubs.to(device='cpu', non_blocking=True) for ubs in self.upper_all]
        if self.selected_dims is not None:
            self.selected_dims = self.selected_dims.to(device='cpu', non_blocking=True)

        if self.slope is None:
            return self
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)

        return self

    def to_device(self, device):
        if self.device == 'cuda':
            return self
        self.dm_l = self.dm_l.to(device, non_blocking=True)
        self.dm_u = self.dm_u.to(device, non_blocking=True)
        self.lower_bound = self.lower_bound.to(device, non_blocking=True)
        self.upper_bound = self.upper_bound.to(device, non_blocking=True)
        # self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
        # self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
        if self.selected_dims is not None:
            self.selected_dims = self.selected_dims.to(device, non_blocking=True)

        if self.slope is None:
            return self
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].to(device, non_blocking=True).float()
        return self


def pick_out_batch(domains, threshold, batch, device='cuda'):
    """
    Pick the first batch of domains in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    """
    if torch.cuda.is_available(): torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

    idx, idx2 = 0, 0
    batch = min(len(domains), batch)
    lower_all, upper_all, slopes_all, dm_l_all, dm_u_all, selected_candidates_all, selected_dims_all = [], [], [], [], [], [], []
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        if len(domains) == 0:
            print("No domain left to pick from. current batch: {}".format(idx))
            break
        selected_candidate = domains.pop(0)
        # idx2 -= 1
        if selected_candidate.lower_bound < threshold:
            # unique = [x for i, x in enumerate(selected_candidate.history) if i == selected_candidate.history.index(x)]
            # assert len(unique) == len(selected_candidate.history)
            selected_candidate.to_device(device)
            idx += 1
            dm_l_all.append(selected_candidate.dm_l)
            dm_u_all.append(selected_candidate.dm_u)
            # lower_all.append(selected_candidate.lower_all)
            # upper_all.append(selected_candidate.upper_all)
            slopes_all.append(selected_candidate.slope)
            selected_dims_all.append(selected_candidate.selected_dims)
            selected_candidates_all.append(selected_candidate)
            if idx == batch: break
        # else:
        #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate.lower_bound, selected_candidate.split))
    batch = idx

    if batch == 0:
        return None, None, None, None, None

    # lower_bounds = []
    # for j in range(len(lower_all[0])):
    #     lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
    #
    # upper_bounds = []
    # for j in range(len(upper_all[0])):
    #     upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))

    slopes = []
    if slopes_all[0] is not None:
        if isinstance(slopes_all[0], dict):
            # Per-neuron slope, each slope is a dictionary.
            slopes = slopes_all
        else:
            for j in range(len(slopes_all[0])):
                slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

    if selected_dims_all[0][0] is not None:
        selected_dims_all = torch.cat(selected_dims_all)
        print(selected_dims_all.shape)

    return slopes, torch.cat(dm_l_all), torch.cat(dm_u_all), selected_candidates_all, selected_dims_all


def add_domain_parallel(domains, lb, ub,  new_dm_l_all, new_dm_u_all, selected_domains, slope, selected_dims=None,
                        save_tree=False, decision_thresh=0,):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    add domains in two ways:
    1. add to a sorted list
    2. add to a binary tree
    """
    decision_thresh = torch.tensor(decision_thresh, device=lb[0].device)
    batch = len(selected_domains)
    for i in range(batch):
        infeasible = False
        if lb[i] < decision_thresh:
            # for ii, (l, u) in enumerate(zip(lb_all[i][1:-1], up_all[i][1:-1])):
            #     if (l-u).max() > 1e-6:
            #         infeasible = True
            #         print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
            #         break

            if not infeasible:
                left = InputDomain(lb=lb[i], ub=ub[i],  slope=slope[i], selected_dims=selected_dims[i:i+1],
                                   dm_l=new_dm_l_all[i:i+1], dm_u=new_dm_u_all[i:i+1])
                if save_tree:
                    selected_domains[i].left = left
                    left.parent = selected_domains[i]

                domains.add(left.to_cpu())

        infeasible = False
        if lb[i+batch] < decision_thresh:
            # for ii, (l, u) in enumerate(zip(lb_all[i+batch][1:-1], up_all[i+batch][1:-1])):
            #     if (l-u).max() > 1e-6:
            #         infeasible = True
            #         print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
            #         break

            if not infeasible:
                right = InputDomain(lb=lb[i + batch], ub=ub[i + batch], slope=slope[i + batch], selected_dims=selected_dims[i + batch:i+batch+1],
                                    dm_l=new_dm_l_all[i+batch:i+batch+1], dm_u=new_dm_u_all[i+batch:i+batch+1])

                if save_tree:
                    selected_domains[i].right = right
                    right.parent = selected_domains[i]

                domains.add(right.to_cpu())

    return


def input_split_batch(net, dm_l_all, dm_u_all, slopes, branching_method='input_grad', shape=None, selected_dims=None):
    batch = len(dm_l_all)
    # print(dm_l_all[0])
    # print(dm_u_all[0])

    dm_l_all = dm_l_all.flatten(1)
    dm_u_all = dm_u_all.flatten(1)

    if branching_method == 'naive':
        # we just select the longest edge
        i_idx = torch.max(dm_u_all - dm_l_all, 1).indices
    elif branching_method == 'input_grad':
        # search all input dims
        k_ret = torch.empty(size=(selected_dims.shape[-1], batch), device=dm_l_all[0].device, requires_grad=False)
        for d in range(selected_dims.shape[-1]):
            dim = selected_dims[:, d]
            dm_l_all_cp = dm_l_all.clone()
            dm_u_all_cp = dm_u_all.clone()

            for b in range(batch):
                dm_l_all_cp[b, dim[b]] = (dm_l_all[b, dim[b]] + dm_u_all[b, dim[b]]) / 2
                dm_u_all_cp[b, dim[b]] = (dm_l_all[b, dim[b]] + dm_u_all[b, dim[b]]) / 2

            new_dm_l = torch.cat([dm_l_all_cp, dm_l_all]).reshape(-1, *shape[1:])
            new_dm_u = torch.cat([dm_u_all, dm_u_all_cp]).reshape(-1, *shape[1:])

            k_ret_lbs = net.get_lower_bound_naive(pre_lb_all=None, pre_ub_all=None, dm_l=new_dm_l, dm_u=new_dm_u,
                                                  slopes=slopes, shortcut=True, lr_alpha=0.01)
            # we only consider the best lower bound across two splits by using min(0)
            k_ret[d] = k_ret_lbs.reshape(2, -1).sum(-1, keepdim=True).min(0).values

        i_idx = k_ret.max(0).indices  # compare across input dims
    else:
        # search all input dims
        k_ret = torch.empty(size=(dm_l_all.shape[-1], batch), device=dm_l_all[0].device, requires_grad=False)
        for dim in range(dm_l_all.shape[-1]):
            dm_l_all_cp = dm_l_all.clone()
            dm_u_all_cp = dm_u_all.clone()

            dm_l_all_cp[:, dim] = (dm_l_all[:, dim] + dm_u_all[:, dim]) / 2
            dm_u_all_cp[:, dim] = (dm_l_all[:, dim] + dm_u_all[:, dim]) / 2

            new_dm_l = torch.cat([dm_l_all_cp, dm_l_all]).reshape(-1, *shape[1:])
            new_dm_u = torch.cat([dm_u_all, dm_u_all_cp]).reshape(-1, *shape[1:])

            k_ret_lbs = net.get_lower_bound_naive(pre_lb_all=None, pre_ub_all=None, dm_l=new_dm_l, dm_u=new_dm_u, slopes=slopes,
                                                  shortcut=True, lr_alpha=0.01)
            # we only consider the best lower bound across two splits by using min(0)
            k_ret[dim] = k_ret_lbs.reshape(2, -1).sum(-1, keepdim=True).min(0).values

        i_idx = k_ret.max(0).indices  # compare across input dims

    dm_l_all_cp = dm_l_all.clone()
    dm_u_all_cp = dm_u_all.clone()

    for b in range(batch):
        dm_l_all_cp[b, i_idx[b]] = (dm_l_all[b, i_idx[b]] + dm_u_all[b, i_idx[b]]) / 2
        dm_u_all_cp[b, i_idx[b]] = (dm_l_all[b, i_idx[b]] + dm_u_all[b, i_idx[b]]) / 2

    new_dm_l_all = torch.cat([dm_l_all_cp, dm_l_all]).reshape(-1, *shape[1:])
    new_dm_u_all = torch.cat([dm_u_all, dm_u_all_cp]).reshape(-1, *shape[1:])

    return new_dm_l_all, new_dm_u_all
