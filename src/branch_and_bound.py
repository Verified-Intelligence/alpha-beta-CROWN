import bisect
import copy
import torch


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
                 depth=None, split_history=None, history=None, gnn_decision=None, split_hint=None, intermediate_betas=None):
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
        # Gradient based hints on which neuron to split next for this domain.
        self.split_hint = split_hint

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

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
        if isinstance(self.slope, dict):
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)
        else:
            self.slope = [s.half().to(device='cpu', non_blocking=True) for s in self.slope]

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
        return self

    def to_device(self, device, partial=False):
        if not partial:
            self.lA = [lA.to(device, non_blocking=True) for lA in self.lA]
            self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
            self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
        if isinstance(self.slope, dict):
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].to(device, non_blocking=True, dtype=torch.get_default_dtype())
        else:
            self.slope = [s.to(device, non_blocking=True, dtype=torch.get_default_dtype()) for s in self.slope]
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
        return self


def add_domain(candidate, domains):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    """
    # bisect.insort_left(domains, candidate.to_cpu())
    domains.add(candidate.to_cpu())


def add_domain_parallel(lA, lb, ub, lb_all, up_all, domains, selected_domains, slope, beta, growth_rate=0,
                        split_history=None, branching_decision=None, save_tree=False, decision_thresh=0,
                        next_split_hint=None, intermediate_betas=None, check_infeasibility=True):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    add domains in two ways:
    1. add to a sorted list
    2. add to a binary tree
    """
    unsat_list = []
    batch = len(selected_domains)
    for i in range(batch):
        if selected_domains[i].valid is True:
            infeasible = False
            if lb[i] < decision_thresh:
                if check_infeasibility:
                    for ii, (l, u) in enumerate(zip(lb_all[i][1:-1], up_all[i][1:-1])):
                        if (l-u).max() > 1e-6:
                            infeasible = True
                            print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                            break

                if not infeasible:
                    # only when two splits improved, we insert them to domains
                    split_hint = None if next_split_hint is None else next_split_hint[i]
                    """
                    new_hist = sorted(selected_domains[i].history+branching_decision[i], key=lambda x: x[0][1])
                    print('F adding', new_hist, 'split', split_hint)
                    """
                    new_history = copy.deepcopy(selected_domains[i].history)
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # first half batch: active neurons
                    new_history[branching_decision[i][0]][1].append(+1.0)  # first half batch: active neurons

                    # sanity check repeated split
                    if branching_decision[i][1] in selected_domains[i].history[branching_decision[i][0]][0]:
                        print('BUG!!! repeated split!')
                        print(selected_domains[i].history)
                        print(branching_decision[i])
                        raise RuntimeError

                    left = ReLUDomain(lA[i], lb[i], ub[i], lb_all[i], up_all[i], slope[i], beta[i],
                                      selected_domains[i].depth+1, split_history=split_history[i],
                                      history=new_history, split_hint=split_hint,
                                      intermediate_betas=intermediate_betas[i])
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
                    split_hint = None if next_split_hint is None else next_split_hint[i+batch]
                    """
                    new_hist = sorted(selected_domains[i].history+branching_decision[i+batch], key=lambda x: x[0][1])
                    print('B adding', new_hist, 'split', split_hint)
                    """
                    new_history = copy.deepcopy(selected_domains[i].history)
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # second half batch: inactive neurons
                    new_history[branching_decision[i][0]][1].append(-1.0)  # second half batch: inactive neurons

                    right = ReLUDomain(lA[i+batch], lb[i+batch], ub[i+batch], lb_all[i+batch], up_all[i+batch],
                                       slope[i+batch],  beta[i+batch], selected_domains[i].depth+1, split_history=split_history[i+batch],
                                       history=new_history, split_hint=split_hint,
                                       intermediate_betas=intermediate_betas[i + batch])

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
    lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        if len(domains) == 0:
            print("No domain left to pick from. current batch: {}".format(idx))
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
            if idx == batch: break
        # else:
        #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate_domain.lower_bound, selected_candidate_domain.split))
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
