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
"""Legacy SortedReLUDomainList for bab-attack."""
import torch
import numpy as np
import copy
from collections import defaultdict
from sortedcontainers import SortedList
from utils import fast_hist_copy, check_infeasible_bounds
from branching_domains import AbstractDomainList
import arguments


class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.
    """

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None,
                up_all=None, alpha=None, beta=None, depth=None, split_history=None,
                history=None, intermediate_betas=None, primals=None,
                priority=0, c=None, threshold=np.float64(0.)):
        if history is None:
            history = []
        if split_history is None:
            split_history = []

        self.lA = lA
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all  # TODO inherit root node
        self.upper_all = up_all
        self.history = history
        self.split_history = split_history
        self.intermediate_betas = intermediate_betas
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        # primals {"p": primal values for input, pre_relu, and obj output primals,
        #   "z": integer values for each relu layer}
        # z: stable relus have -1, others all unstable neuron from 0 to 1
        self.primals = primals
        self.priority = priority  # Higher priority will be more likely to be selected.
        # record c for each domain so that the domain list is capable of saving multiple c's
        # c shape here: (1,1,num_outputs)
        self.c = c
        self.threshold = np.float64(threshold)
        self.left = self.right = self.parent = None
        self.valid = True
        self.split = False

    def __lt__(self, other):
        if self.priority == other.priority:
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
                return (self.lower_bound - self.threshold).max() > (other.lower_bound - other.threshold).max()
            else:
                return (self.lower_bound - self.threshold).max() < (other.lower_bound - other.threshold).max()
        else:
            # higher priority should be in the front of the queue.
            return self.priority >= other.priority

    def __le__(self, other):
        if self.priority == other.priority:
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
                return (self.lower_bound - self.threshold).max() >= (other.lower_bound - other.threshold).max()
            else:
                return (self.lower_bound - self.threshold).max() <= (other.lower_bound - other.threshold).max()
        else:
            return self.priority > other.priority

    def __eq__(self, other):
        if self.priority == other.priority:
            return (self.lower_bound - self.threshold).max() == (other.lower_bound - other.threshold).max()
        else:
            return self.priority == other.priority

    def verify_criterion(self):
        return (self.lower_bound > self.threshold).any()

    def del_node(self):
        if self.left is not None:
            self.left.del_node()
        if self.right is not None:
            self.right.del_node()
        self.valid = False


# Since to_cpu and to_device are only used for bab attacks, they are put
# here to simplify ReLUDomain.

def to_cpu(self):
    # transfer the content of this domain to cpu memory (try to reduce memory consumption)
    self.lA = [lA.to(device='cpu', non_blocking=True) for lA in self.lA]
    self.lower_all = [lbs.to(device='cpu', non_blocking=True) for lbs in self.lower_all]
    self.upper_all = [ubs.to(device='cpu', non_blocking=True) for ubs in self.upper_all]
    for layer in self.alpha:
        for intermediate_layer in self.alpha[layer]:
            self.alpha[layer][intermediate_layer] = self.alpha[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)

    if self.split_history:
        if "beta" in self.split_history:
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
        if "general_beta" in self.split_history:
            self.split_history["general_beta"] = self.split_history["general_beta"].to(device="cpu", non_blocking=True)

    if self.intermediate_betas is not None:
        for split_layer in self.intermediate_betas:
            for intermediate_layer in self.intermediate_betas[split_layer]:
                self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device='cpu', non_blocking=True)
                self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device='cpu', non_blocking=True)

    if self.beta is not None:
        if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
            for i in range(len(self.beta)):
                self.beta[i] = [b.to(device='cpu', non_blocking=True) for b in self.beta[i]]
        else:
            self.beta = [b.to(device='cpu', non_blocking=True) for b in self.beta]

    if self.c is not None:
        self.c = self.c.to(device='cpu', non_blocking=True)
    return self


def to_device(self, device, partial=False):
    if not partial:
        self.lA = [lA.to(device, non_blocking=True) for lA in self.lA]
        self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
    for layer in self.alpha:
        for intermediate_layer in self.alpha[layer]:
            self.alpha[layer][intermediate_layer] = self.alpha[layer][intermediate_layer].to(device, non_blocking=True, dtype=torch.get_default_dtype())
    if self.split_history:
        if "beta" in self.split_history:
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
        if "general_beta" in self.split_history:
            self.split_history["general_beta"] = self.split_history["general_beta"].to(device=device, non_blocking=True)
    if self.intermediate_betas is not None:
        for split_layer in self.intermediate_betas:
            for intermediate_layer in self.intermediate_betas[split_layer]:
                self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device, non_blocking=True)
                self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device, non_blocking=True)
    if self.beta is not None:
        if arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']:
            for i in range(len(self.beta)):
                self.beta[i] = [b.to(device, non_blocking=True) for b in self.beta[i]]
        else:
            self.beta = [b.to(device, non_blocking=True) for b in self.beta]
    if self.c is not None:
        self.c = self.c.to(device, non_blocking=True)
    return self


class SortedReLUDomainList(AbstractDomainList):
    """
        Maintains a sorted list of domain list, but add and remove domains individually which is slow
    """

    def __init__(self, ret, lAs, global_lbs, global_ubs, alphas, history,
                 thresholds, net=None, branching_input_and_activation=False, **kwargs):
        super(SortedReLUDomainList, self).__init__()

        lb_alls, ub_alls = ret['lower_bounds'], ret['upper_bounds']
        beta = ret.get('betas', None)
        num = global_lbs.shape[0]

        self.net = net
        assert not branching_input_and_activation, "Branching on input space unsupported."

        instance_alphas = [defaultdict(dict) for _ in range(num)]
        for k in alphas:
            for item in instance_alphas: item[k] = {}
            for kk, v in alphas[k].items():
                for i, item in enumerate(instance_alphas):
                    item[k][kk] = v[:, :, i:(i+1)]

        instance_lAs = [[] for _ in range(num)]
        for i in range(num):
            instance_lAs[i] = [lA[i:(i+1)] for lA in lAs]
        candidate_domains = [to_cpu(ReLUDomain(
            instance_lAs[i],
            global_lbs[i],
            global_ubs[i],
            [lb[i:i+1] for lb in lb_alls],
            [ub[i:i+1] for ub in ub_alls],
            instance_alphas[i],
            history=copy.deepcopy(history),
            depth=0,
            c=net.c[i:i+1],
            threshold=thresholds[i] if thresholds.numel() > 1 else thresholds.view(1),
            beta=beta if beta is not None else None
        )) for i in range(num)]
        self.domains = SortedList()
        for domain in candidate_domains:
            self.domains.add(domain)

    def pick_out(self, batch, device='cpu'):

        assert batch > 0

        if torch.cuda.is_available(): torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        idx, idx2 = 0, 0
        batch = min(len(self), batch)
        lAs, lower_all, upper_all, alphas_all, betas_all, intermediate_betas_all, history_all, split_history_all, global_lbs_all, depths_all = [], [], [], [], [], [], [], [], [], []
        c_all, thresholds_all = [], []
        assert len(self) > 0, "The given domains list is empty."
        while True:
            # Pop out domains from the list one by one (SLOW).
            if len(self.domains) == 0:
                print(f"No domain left to pick from. Batch limit {batch} current batch: {idx}")
                break
            if idx2 == len(self.domains): break  # or len(domains)-1?
            if self.domains[idx2].split is True:
                idx2 += 1
                continue
            selected_candidate_domain = self.domains.pop(idx2)
            # idx2 -= 1
            if not selected_candidate_domain.verify_criterion() and selected_candidate_domain.valid is True:
                # unique = [x for i, x in enumerate(selected_candidate_domain.history) if i == selected_candidate_domain.history.index(x)]
                # assert len(unique) == len(selected_candidate_domain.history)
                # We transfer only some of the tensors directly to GPU. Other tensors will be transferred in batch later.
                to_device(selected_candidate_domain, device, partial=True)
                idx += 1
                lAs.append(selected_candidate_domain.lA)
                lower_all.append(selected_candidate_domain.lower_all)
                upper_all.append(selected_candidate_domain.upper_all)
                alphas_all.append(selected_candidate_domain.alpha)
                betas_all.append(selected_candidate_domain.beta)
                intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
                c_all.append(selected_candidate_domain.c)
                history_all.append(selected_candidate_domain.history)
                split_history_all.append(selected_candidate_domain.split_history)
                thresholds_all.append(selected_candidate_domain.threshold)
                global_lbs_all.append(selected_candidate_domain.lower_bound)
                depths_all.append(selected_candidate_domain.depth)
                selected_candidate_domain.valid = False  # set False to avoid another pop
                if idx == batch: break
            selected_candidate_domain.valid = False   # set False to avoid another pop

        batch = idx

        if batch == 0:
            return None, None, None, None, None, None, None, None, None, None, None,

        lower_bounds = []
        upper_bounds = []
        new_lAs = []
        new_masks = []
        # For ReLU split domains. Input split domains do not have these properties.
        for j in range(len(lower_all[0])):
            lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
        lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

        for j in range(len(upper_all[0])):
            upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
        upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

        # Reshape to batch first in each list.
        for j in range(len(lAs[0])):
            new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
        # Transfer to GPU.
        new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

        # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
        lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
        upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]

        # Recompute the mask on GPU.
        for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
            new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())

        thresholds = torch.stack(thresholds_all).to(device=device, non_blocking=True)
        global_lbs = torch.stack(global_lbs_all).to(device=device, non_blocking=True)

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).to(device=device, non_blocking=True)
        if not cs.is_contiguous():
            cs = cs.contiguous()

        alphas = defaultdict(dict)
        if alphas_all[0] is not None:
            if isinstance(alphas_all[0], dict):
                # Per-neuron alpha, each alpha is a dictionary.
                for k, v in alphas_all[0].items():
                    alphas[k] = {}
                    for kk in v:
                        alphas[k][kk] = torch.cat([alpha_item[k][kk] for alpha_item in alphas_all], dim=2)
            else:
                alphas = []
                for j in range(len(alphas_all[0])):
                    alphas.append(torch.cat([alphas_all[i][j] for i in range(batch)]))

        return {
            'mask': new_masks, 'lAs': new_lAs,
            'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds,
            'alphas': alphas, 'betas': betas_all,
            'intermediate_betas': intermediate_betas_all,
            'history': history_all, 'split_history': split_history_all,
            'global_lb': global_lbs, 'depths': depths_all, 'cs': cs,
            'thresholds': thresholds,
            'x_Ls': None, 'x_Us': None, 'input_split_idx': None,
        }

    def add(self, bounds, histories, depths,
            branching_decisions, decision_threshs, check_infeasibility):

        # TODO don't expand
        lAs = bounds['lAs']
        lbs, ubs = bounds['lower_bounds'][-1], bounds['upper_bounds'][-1]
        lb_alls, ub_alls = bounds['lower_bounds'], bounds['upper_bounds']
        split_histories = bounds['split_history']
        alphas, betas = bounds['alphas'], bounds['betas']
        intermediate_betas = bounds['intermediate_betas']
        Cs, x_Ls, x_Us = bounds['c'], bounds['x_Ls'], bounds['x_Us']
        input_split_idx = bounds['input_split_idx']
        num = len(depths) * 2

        lbs, ubs = lbs[:num], ubs[:num]
        split_histories = split_histories[:num]
        Cs = Cs[:num]
        lAs = [x[:num] for x in lAs]
        betas = betas[:num]
        intermediate_betas = intermediate_betas[:num]
        if x_Ls is not None or x_Us is not None or input_split_idx is not None:
            raise NotImplementedError("Branching on input space unsupported.")

        instance_alphas = [defaultdict(dict) for _ in range(len(lbs))]
        for k in alphas:
            for item in instance_alphas: item[k] = {}
            for kk, v in alphas[k].items():
                for i, item in enumerate(instance_alphas):
                    item[k][kk] = v[:, :, i:(i+1)]
        instance_alphas = instance_alphas[:num]
        instance_lAs = [[] for _ in range(len(lbs))]
        for item in lAs:
            for i in range(len(instance_lAs)):
                instance_lAs[i].append(item[i:i+1])
        instance_lb_alls = [[] for _ in range(len(lbs))]
        for item in lb_alls:
            if item is None:
                raise NotImplementedError("interm_transfer=false unimplemented for sorted domains.")
            for i in range(len(instance_lb_alls)):
                instance_lb_alls[i].append(item[i:i+1])
        instance_ub_alls = [[] for _ in range(len(lbs))]
        for item in ub_alls:
            for i in range(len(instance_ub_alls)):
                instance_ub_alls[i].append(item[i:i+1])

        batch = len(histories)
        decision_threshs = decision_threshs.to(lbs[0].device, non_blocking=True)
        for i in range(batch):
            infeasible = False
            if (lbs[i] <= decision_threshs[i]).all():
                if check_infeasibility:
                    if check_infeasible_bounds(
                        lb_alls[i], ub_alls[i], reduce=True):
                        infeasible = True

                if not infeasible:
                    priority=0
                    new_history = fast_hist_copy(histories[i])
                    if branching_decisions is not None:
                        new_history[branching_decisions[i][0]][0].append(branching_decisions[i][1])  # first half batch: active neurons
                        new_history[branching_decisions[i][0]][1].append(+1.0)  # first half batch: active neurons

                        # sanity check repeated split
                        if branching_decisions[i][1] in histories[i][branching_decisions[i][0]][0]:
                            print('BUG!!! repeated split!')
                            print(histories[i])
                            print(branching_decisions[i])
                            raise RuntimeError

                    left_primals = None
                    left = ReLUDomain(instance_lAs[i], lbs[i], ubs[i], instance_lb_alls[i], instance_ub_alls[i],
                                      instance_alphas[i],
                                      betas[i],
                                      depths[i]+1, split_history=split_histories[i],
                                      history=new_history,
                                      intermediate_betas=intermediate_betas[i],
                                      primals=left_primals, priority=priority,
                                      c=Cs[i:i+1], threshold=decision_threshs[i])

                    self.domains.add(left)

            infeasible = False
            if (lbs[i+batch] <= decision_threshs[i]).all():
                if check_infeasibility:
                    if check_infeasible_bounds(
                        lb_alls[i+batch], ub_alls[i+batch], reduce=True):
                        infeasible = True

                if not infeasible:
                    priority=0
                    new_history = fast_hist_copy(histories[i])
                    if branching_decisions is not None:
                        new_history[branching_decisions[i][0]][0].append(branching_decisions[i][1])  # second half batch: inactive neurons
                        new_history[branching_decisions[i][0]][1].append(-1.0)  # second half batch: inactive neurons

                    right_primals = None
                    right = ReLUDomain(instance_lAs[i+batch], lbs[i+batch], ubs[i+batch], instance_lb_alls[i+batch], instance_ub_alls[i+batch],
                                       instance_alphas[i+batch],  betas[i+batch], depths[i+batch]+1, split_history=split_histories[i+batch],
                                       history=new_history,
                                       intermediate_betas=intermediate_betas[i + batch],
                                       primals=right_primals, priority=priority,
                                       c=Cs[i+batch:i+batch+1], threshold=decision_threshs[i])
                    self.domains.add(right)

    def get_min_domain(self, num, rev_order=False):
        if not rev_order:
            return self.domains[:num]
        else:
            return self.domains[: -(num+1): -1]

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, index):
        return self.domains[index]

    def to_sortedList(self):
        """
            This function is only for supporting legacy code. It is slow. Avoid to use it frequently!
        :return:
        """
        return copy.copy(self.domains)

