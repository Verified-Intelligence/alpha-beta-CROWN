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
from collections import defaultdict
import torch
import numpy as np
from sortedcontainers import SortedList
import copy


class gconstr:
    """
    Object representing the general split constraint, assume this branch has m contraints and n nodes
    beta: the dual variable associated to this constraint, a 1*m vector
    coeff: coefficients assoicated to each node in this constraint, a list,
            each is a m*n constraints for each relu layer,
            [] if no constraint nodes used in this layer
    c: 1 for >=0 and 0 for <=0, a 1*m vector
    """
    def __init__(self, beta, var, coeff, c):
        assert self.beta.shape[1] == self.c.shape[1] and self.beta.shape[1] == self.coeffs[-1]
        self.beta = beta
        self.var = var
        self.coeffs = coeff
        self.c = c
        self.layer = []
        for i, coeff in enumerate(self.coeffs):
            if coeff:
                self.layer.append(i)

    def __str__(self):
        s = "layer{} m{}".format(self.layer, self.beta.shape[1])
        s = "beta{} ".format(self.beta.item())
        for v, a in zip(self.var, self.coeff):
            s += str(a)+str(v)+" "
        s += "c"+str(self.c)
        return s


class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.
    """

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, slope=None, beta=None,
                 depth=None, split_history=None, history=None, gnn_decision=None, intermediate_betas=None, primals=None, priority=0):
        if history is None:
            history = []
        if split_history is None:
            self.split_history = []

        self.lA = lA
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.history = history
        self.split_history = split_history
        self.intermediate_betas = intermediate_betas
        self.slope = slope
        self.beta = beta
        self.left = None
        self.right = None
        self.parent = None
        self.valid = True
        self.split = False
        self.depth = depth
        self.gnn_decision = gnn_decision
        # primals {"p": primal values for input, pre_relu, and obj output primals, 
        #   "z": integer values for each relu layer}
        # z: stable relus have -1, others all unstable neuron from 0 to 1
        self.primals = primals
        self.priority = priority  # Higher priority will be more likely to be selected.

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.lower_bound < other.lower_bound
        else:
            # higher priority should be in the front of the queue.
            return self.priority >= other.priority

    def __le__(self, other):
        if self.priority == other.priority:
            return self.lower_bound <= other.lower_bound
        else:
            return self.priority > other.priority

    def __eq__(self, other):
        if self.priority == other.priority:
            return self.lower_bound == other.lower_bound
        else:
            return self.priority == other.priority

    def del_node(self):
        if self.left is not None:
            self.left.del_node()
        if self.right is not None:
            self.right.del_node()

        self.valid = False

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.lA = [lA.to(device='cpu', non_blocking=True) for lA in self.lA]
        self.lower_all = [lbs.to(device='cpu', non_blocking=True) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device='cpu', non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)

        if self.split_history:
            for lidx in range(len(self.split_history["beta"])):
                if self.split_history["single_beta"][lidx] is not None:
                    self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                    self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device='cpu', non_blocking=True)
                    self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device='cpu', non_blocking=True)
                if self.split_history["beta"][lidx] is not None:
                    self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device='cpu', non_blocking=True)
                    self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device='cpu', non_blocking=True)
                    self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                    self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device='cpu', non_blocking=True)
                if self.split_history["bias"][lidx] is not None:
                    self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device='cpu', non_blocking=True)

        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device='cpu', non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device='cpu', non_blocking=True)

        if self.beta is not None:
            self.beta = [b.to(device='cpu', non_blocking=True) for b in self.beta]
        
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device='cpu', non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device='cpu', non_blocking=True)
        return self

    def to_device(self, device, partial=False):
        if not partial:
            self.lA = [lA.to(device, non_blocking=True) for lA in self.lA]
            self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
            self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].to(device, non_blocking=True, dtype=torch.get_default_dtype())
        if self.split_history:
            for lidx in range(len(self.split_history["beta"])):
                if self.split_history["single_beta"][lidx] is not None:
                    self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device=device, non_blocking=True)
                    self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device=device, non_blocking=True)
                    self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device=device, non_blocking=True)
                if self.split_history["beta"][lidx] is not None:
                    self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device=device, non_blocking=True)
                    self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device=device, non_blocking=True)
                    self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device=device, non_blocking=True)
                    self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device=device, non_blocking=True)
                if self.split_history["bias"][lidx] is not None:
                    self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device=device, non_blocking=True)
        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device, non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device, non_blocking=True)
        if self.beta is not None:
            self.beta = [b.to(device, non_blocking=True) for b in self.beta]
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device, non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device, non_blocking=True)
        return self

    def clone_to_dive(self, beam_search=False):
        
        if beam_search:
            lower_all = upper_all = None   # These should not be used in beam search.
            beta = None  # This should not be used.
            history = [[None, None] for i in range(len(self.history))]  # Create an empty history for each layer.
        else:
            history = copy.deepcopy(self.history)
            lower_all = [lb.clone().detach() for lb in self.lower_all]
            upper_all = [ub.clone().detach() for ub in self.upper_all]
            if self.beta is None:
                beta = None
            else:
                beta = []
                for b in self.beta:
                    beta.append(b.clone().detach())

        ####### Need to make sure we do not need to clone primals #######
        dive_d = ReLUDomain(lA=self.lA, lb=self.lower_bound, ub=self.upper_bound, 
                    lb_all=lower_all, up_all=upper_all, slope=self.slope, 
                    beta=beta, depth=self.depth,
                    split_history=[], history=history,
                    gnn_decision=self.gnn_decision, intermediate_betas=[],
                    primals=self.primals, priority=self.priority)
        
        return dive_d


class DFS_ReLUDomain:
    def __init__(self, domain):
        # refer DFS_ReLUDomain to the original ReLUDomain but rewrite the compare function by depth
        self.domain = domain

    def __lt__(self, other):
        if self.domain.depth != other.domain.depth:
            return self.domain.depth > other.domain.depth
        else:
            return self.domain.lower_bound < other.domain.lower_bound

    def __le__(self, other):
        if self.domain.depth != other.domain.depth:
            return self.domain.depth > other.domain.depth
        else:
            return self.domain.lower_bound <= other.domain.lower_bound

    def __eq__(self, other):
        return self.domain.lower_bound == other.domain.lower_bound and self.domain.depth == other.domain.depth


def add_domain(candidate, domains):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    """
    # bisect.insort_left(domains, candidate.to_cpu())
    domains.add(candidate.to_cpu())


def add_domain_parallel(lA, lb, ub, lb_all, up_all, domains, selected_domains, slope, beta, growth_rate=0,
                        split_history=None, branching_decision=None, save_tree=False, decision_thresh=0,
                        intermediate_betas=None, check_infeasibility=True, primals=None, priorities=None):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    add domains in two ways:
    1. add to a sorted list
    2. add to a binary tree
    # diving: we are adding diving domains if True
    """
    unsat_list = []
    batch = len(selected_domains)
    for i in range(batch):
        infeasible = False
        if lb[i] < decision_thresh:
            if check_infeasibility:
                for ii, (l, u) in enumerate(zip(lb_all[i][1:-1], up_all[i][1:-1])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

            if not infeasible:
                priority=0 if priorities is None else priorities[i].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                if branching_decision is not None:
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # first half batch: active neurons
                    new_history[branching_decision[i][0]][1].append(+1.0)  # first half batch: active neurons

                    # sanity check repeated split
                    if branching_decision[i][1] in selected_domains[i].history[branching_decision[i][0]][0]:
                        print('BUG!!! repeated split!')
                        print(selected_domains[i].history)
                        print(branching_decision[i])
                        raise RuntimeError

                left_primals = primals[i] if primals is not None else None
                left = ReLUDomain(lA[i], lb[i], ub[i], lb_all[i], up_all[i], slope[i], beta[i],
                                  selected_domains[i].depth+1, split_history=split_history[i],
                                  history=new_history,
                                  intermediate_betas=intermediate_betas[i],
                                  primals=left_primals, priority=priority)
                if save_tree:
                    selected_domains[i].left = left
                    left.parent = selected_domains[i]

                    # assert (m[mp == 0] == 0).all(), m[mp == 0].abs().sum()
                    # assert (m[mp == 1] == 1).all(), m[mp == 1].abs().sum()
                # bisect.insort_left(domains, left)
                domains.add(left)

        infeasible = False
        if lb[i+batch] < decision_thresh:
            # if growth_rate and (selected_domains[i].lower_bound - lb[i+batch]) > selected_domains[i].lower_bound * growth_rate and flag:
            #     selected_domains[i].split = True
            #     bisect.insort_left(domains, selected_domains[i])
            #     unsat_list.append(i+batch)
            #     # if len(unsat_list) == 1: choice = unsat_list[0]
            # else:

            if check_infeasibility:
                for ii, (l, u) in enumerate(zip(lb_all[i+batch][1:-1], up_all[i+batch][1:-1])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

            if not infeasible:
                priority=0 if priorities is None else priorities[i+batch].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                if branching_decision is not None:
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # second half batch: inactive neurons
                    new_history[branching_decision[i][0]][1].append(-1.0)  # second half batch: inactive neurons

                right_primals = primals[i + batch] if primals is not None else None
                right = ReLUDomain(lA[i+batch], lb[i+batch], ub[i+batch], lb_all[i+batch], up_all[i+batch],
                                   slope[i+batch],  beta[i+batch], selected_domains[i].depth+1, split_history=split_history[i+batch],
                                   history=new_history,
                                   intermediate_betas=intermediate_betas[i + batch],
                                   primals=right_primals, priority=priority)

                if save_tree:
                    selected_domains[i].right = right
                    right.parent = selected_domains[i]

                # for ii, (m, mp) in enumerate(zip(updated_mask[i+batch], selected_domains[i].mask)):
                #     if not ((m[mp == 0] == 0).all() and (m[mp == 1] == 1).all()):
                #         infeasible = True
                #         print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                #         break

                # assert (m[mp == 0] == 0).all(), m[mp == 0].abs().sum()
                # assert (m[mp == 1] == 1).all(), m[mp == 1].abs().sum()

                # bisect.insort_left(domains, right)
                domains.add(right)

    return unsat_list


def pick_out(domains, threshold):
    '''
    Pick the first domain in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    '''
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        assert len(domains) > 0, "No domain left to pick from."
        selected_candidate_domain = domains.pop(0)
        if selected_candidate_domain.lower_bound < threshold and selected_candidate_domain.valid is True:
            break
        else:
            print('select domain again', selected_candidate_domain.lower_bound, threshold)

    return selected_candidate_domain


def select_batch(domains, batch):
    '''
    Select a batch of domains. Ignore invalid ones.
    '''
    assert batch > 0
    idx = 0
    ret = []
    while batch > 0:
        if len(domains) <= idx:
            break
        if domains[idx].valid:
            batch -= 1
            ret.append(domains[idx])
        idx += 1
    return ret

def pick_out_batch(domains, threshold, batch, device='cuda', DFS_percent=0, diving=False):
    """
    Pick the first batch of domains in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    dive_rate: how many times of dive domains over selected domains picked out

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    """
    assert batch > 0
    DFS_batch = int(DFS_percent * min(batch, len(domains)))
    if DFS_batch > 0:
        tmp_d = []
        idx = 0
        while True:
            if idx == len(domains.sublist):
                break  # reach to end of the domains

            d = domains.sublist.pop(0)  # domains.sublist is already sorted by depth

            if d.domain.valid is True:
                tmp_d.append(d.domain)
                # will set d.domain.valid = False later in the recursive pick_out_batch()
                idx += 1

            if DFS_batch == idx:
                break  # we collected enough domains by DFS

        DFS_ret = pick_out_batch(tmp_d, threshold, idx, device, DFS_percent=0)
        batch -= idx

    if torch.cuda.is_available(): torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

    idx, idx2 = 0, 0
    batch = min(len(domains), batch)
    lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        if len(domains) == 0:
            print(f"No domain left to pick from. Batch limit {batch} current batch: {idx}")
            break
        # try:
        if idx2 == len(domains): break  # or len(domains)-1?
        if domains[idx2].split is True:
            idx2 += 1
            # print(idx2, len(domains))
            continue
        # except:
        #     import pdb; pdb.set_trace()
        selected_candidate_domain = domains.pop(idx2)
        # idx2 -= 1
        if selected_candidate_domain.lower_bound < threshold and selected_candidate_domain.valid is True:
            # unique = [x for i, x in enumerate(selected_candidate_domain.history) if i == selected_candidate_domain.history.index(x)]
            # assert len(unique) == len(selected_candidate_domain.history)
            # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
            selected_candidate_domain.to_device(device, partial=True)
            idx += 1
            lAs.append(selected_candidate_domain.lA)
            lower_all.append(selected_candidate_domain.lower_all)
            upper_all.append(selected_candidate_domain.upper_all)
            slopes_all.append(selected_candidate_domain.slope)
            betas_all.append(selected_candidate_domain.beta)
            intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
            selected_candidate_domains.append(selected_candidate_domain)
            selected_candidate_domain.valid = False  # set False to avoid another pop
            if idx == batch: break
        # else:
        #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate_domain.lower_bound, selected_candidate_domain.split))
        selected_candidate_domain.valid = False   # set False to avoid another pop

    batch = idx

    if batch == 0:
        return None, None, None, None, None

    lower_bounds = []
    for j in range(len(lower_all[0])):
        lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
    lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

    upper_bounds = []
    for j in range(len(upper_all[0])):
        upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
    upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

    # Reshape to batch first in each list.
    new_lAs = []
    for j in range(len(lAs[0])):
        new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
    # Transfer to GPU.
    new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

    slopes = []
    if slopes_all[0] is not None:
        if isinstance(slopes_all[0], dict):
            # Per-neuron slope, each slope is a dictionary.
            slopes = slopes_all
        else:
            for j in range(len(slopes_all[0])):
                slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

    # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
    lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
    upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]
    
    # Recompute the mask on GPU.
    new_masks = []
    for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
        new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())

    if DFS_batch > 0:
        DFS_masks = []
        for j in range(len(new_masks)):
            DFS_masks.append(torch.cat([DFS_ret[0][j], new_masks[j]]))

        DFS_lAs = []
        for j in range(len(new_lAs)):
            DFS_lAs.append(torch.cat([DFS_ret[1][j], new_lAs[j]]))

        DFS_lower_bounds = []
        for j in range(len(lower_bounds)):
            DFS_lower_bounds.append(torch.cat([DFS_ret[2][j], lower_bounds[j]]))

        DFS_upper_bounds = []
        for j in range(len(upper_bounds)):
            DFS_upper_bounds.append(torch.cat([DFS_ret[3][j], upper_bounds[j]]))

        DFS_slopes = DFS_ret[4] + slopes

        DFS_betas_all = DFS_ret[5] + betas_all

        DFS_intermediate_betas_all = DFS_ret[6] + intermediate_betas_all

        DFS_selected_candidate_domains = DFS_ret[7] + selected_candidate_domains

        return DFS_masks, DFS_lAs, DFS_lower_bounds, DFS_upper_bounds, DFS_slopes, DFS_betas_all, DFS_intermediate_betas_all, DFS_selected_candidate_domains
    return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains


def prune_domains(domains, threshold):
    '''
    Remove domain from `domains`
    that have a lower_bound greater than `threshold`
    '''
    # TODO: Could do this with binary search rather than iterating.
    # TODO: If this is not sorted according to lower bounds, this
    # implementation is incorrect because we can not reason about the lower
    # bounds of the domain that come after
    for i in range(len(domains)):
        if domains[i].lower_bound >= threshold:
            domains = domains[0:i]
            break
    return domains


class DFS_SortedList(SortedList):
    def __init__(self, iterable=None, key=None):
        super().__init__(iterable=iterable, key=key)
        self.sublist = SortedList()  # initial a SortedList to save domains sorted by depth

    def add(self, value):
        super().add(value=value)  # ReLUDomain wii be sorted by lowerbound
        self.sublist.add(DFS_ReLUDomain(value))  # DFS_ReLUDomain will be sorted by depth

    def pop(self, index=-1):
        return super().pop(index=index)


def merge_domains_params(domains_params, dive_domains_params):
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = domains_params
    dive_mask, dive_lAs, dive_orig_lbs, dive_orig_ubs, dive_slopes, dive_betas,\
            dive_intermediate_betas, dive_selected_domains = dive_domains_params
    merge_mask, merge_lAs, merge_orig_lbs, merge_orig_ubs, merge_slopes, merge_betas,\
            merge_intermediate_betas, merge_selected_domains = [], [], [], [], [], [], [], []
    for i in range(len(mask)):
        merge_mask.append(torch.cat([mask[i], dive_mask[i]], dim=0))
        merge_lAs.append(torch.cat([lAs[i], dive_lAs[i]], dim=0))

    for i in range(len(orig_lbs)):
        merge_orig_lbs.append(torch.cat([orig_lbs[i], dive_orig_lbs[i]]))
        merge_orig_ubs.append(torch.cat([orig_ubs[i], dive_orig_ubs[i]]))
    
    merge_slopes = slopes + dive_slopes
    merge_betas = betas + dive_betas
    merge_intermediate_betas = intermediate_betas + dive_intermediate_betas
    merge_selected_domains = selected_domains + dive_selected_domains

    return merge_mask, merge_lAs, merge_orig_lbs, merge_orig_ubs, merge_slopes, merge_betas,\
                    merge_intermediate_betas, merge_selected_domains


