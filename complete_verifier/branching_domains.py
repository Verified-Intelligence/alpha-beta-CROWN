#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch
from collections import defaultdict
from sortedcontainers import SortedList
import copy
import numpy as np
import arguments

from tensor_storage import TensorStorage


class AbstractReLUDomainList():
    """
        Abstract class that maintains the list of domains (variables on CPUs)
    """

    def __init__(self):
        pass

    def pick_out(self, batch, device):
        raise NotImplementedError

    def add(self, lAs, lbs, ubs, lb_alls, up_alls, histories, depths, slopes, beta,
            split_histories, branching_decisions, decision_threshs,
            intermediate_betas, check_infeasibility, Cs, num):
        raise NotImplementedError

    def get_min_domain(self, num, rev_order=False):
        # need to return a list of obj, each object has lower_bounds, upper_bounds, threshold, and depth
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def to_sortedList(self):
        """
            This function is only for supporting legacy code. It is slow. Avoid to use it frequently!
        :return:
        """
        now_len = len(self)
        ret = SortedList()
        for i in range(now_len):
            ret.add(self[i])
        return ret

class SortedReLUDomainList(AbstractReLUDomainList):
    """
        Maintains a sorted list of domain list, but add and remove domains individually which is slow
    """

    def __init__(self, lAs, global_lbs, global_ubs, lb_alls, ub_alls, slopes, history, depths, Cs,
                 thresholds, beta, num, interm_transfer=True):
        # interm_transfer is a dummy argument - has no effect for sorted domain list
        super(SortedReLUDomainList, self).__init__()

        instance_slopes = [defaultdict(dict) for _ in range(num)]
        for k in slopes:
            for item in instance_slopes: item[k] = {}
            for kk, v in slopes[k].items():
                for i, item in enumerate(instance_slopes):
                    item[k][kk] = v[:, :, i:(i+1)]

        instance_lAs = [[] for _ in range(num)]
        for i in range(num):
            instance_lAs[i] = [lA[i:(i+1)] for lA in lAs]
        candidate_domains = [ReLUDomain(
            instance_lAs[i],
            global_lbs[i],
            global_ubs[i],
            [lb[i:i+1] for lb in lb_alls],
            [ub[i:i+1] for ub in ub_alls],
            instance_slopes[i],
            history=copy.deepcopy(history),
            depth=depths[i],
            c=Cs[i:i+1],
            threshold=thresholds[i] if thresholds.numel() > 1 else thresholds,
            beta=beta if beta is not None else None
        ).to_cpu() for i in range(num)]
        self.domains = SortedList()
        for domain in candidate_domains:
            self.domains.add(domain)

    def pick_out(self, batch, device='cpu'):

        assert batch > 0

        if torch.cuda.is_available(): torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        idx, idx2 = 0, 0
        batch = min(len(self), batch)
        lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
        dm_l_all, dm_u_all = [], []
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
                # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
                selected_candidate_domain.to_device(device, partial=True)
                idx += 1
                lAs.append(selected_candidate_domain.lA)
                lower_all.append(selected_candidate_domain.lower_all)
                upper_all.append(selected_candidate_domain.upper_all)
                slopes_all.append(selected_candidate_domain.slope)
                betas_all.append(selected_candidate_domain.beta)
                intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
                dm_l_all.append(selected_candidate_domain.dm_l)
                dm_u_all.append(selected_candidate_domain.dm_u)
                c_all.append(selected_candidate_domain.c)
                thresholds_all.append(selected_candidate_domain.threshold)
                selected_candidate_domains.append(selected_candidate_domain)
                selected_candidate_domain.valid = False  # set False to avoid another pop
                if idx == batch: break
            # else:
            #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate_domain.lower_bound, selected_candidate_domain.split))
            selected_candidate_domain.valid = False   # set False to avoid another pop

        batch = idx

        if batch == 0:
            if isinstance(selected_candidate_domain, ReLUDomain):
                return None, None, None, None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None, None

        lower_bounds = []
        upper_bounds = []
        new_lAs = []
        new_masks = []
        if isinstance(selected_candidate_domain, ReLUDomain):
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

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).to(device=device, non_blocking=True)
        if not cs.is_contiguous():
            cs = cs.contiguous()

        slopes = defaultdict(dict)
        if slopes_all[0] is not None:
            if isinstance(slopes_all[0], dict):
                # Per-neuron slope, each slope is a dictionary.
                for k, v in slopes_all[0].items():
                    slopes[k] = {}
                    for kk in v:
                        slopes[k][kk] = torch.cat([slope_item[k][kk] for slope_item in slopes_all], dim=2)
            else:
                slopes = []
                for j in range(len(slopes_all[0])):
                    slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

        if isinstance(selected_candidate_domain, ReLUDomain):
            # Relu split domains.
            return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains, cs, thresholds
        else:
            # Input split domains.
            return slopes, torch.cat(dm_l_all).to(device=device, non_blocking=True), torch.cat(dm_u_all).to(device=device, non_blocking=True), selected_candidate_domains, cs, thresholds


    def add(self, lAs, lbs, ubs, lb_alls, up_alls, histories, depths, slopes, betas,
            split_histories, branching_decisions, decision_threshs,
            intermediate_betas, check_infeasibility, Cs, num):

        lbs, ubs = lbs[:num], ubs[:num]
        split_histories = split_histories[:num]
        Cs = Cs[:num]
        lAs = [x[:num] for x in lAs]
        betas = betas[:num]
        intermediate_betas = intermediate_betas[:num]

        instance_slopes = [defaultdict(dict) for _ in range(len(lbs))]
        for k in slopes:
            for item in instance_slopes: item[k] = {}
            for kk, v in slopes[k].items():
                for i, item in enumerate(instance_slopes):
                    item[k][kk] = v[:, :, i:(i+1)]
        instance_slopes = instance_slopes[:num]
        instance_lAs = [[] for _ in range(len(lbs))]
        for item in lAs:
            for i in range(len(instance_lAs)):
                instance_lAs[i].append(item[i:i+1])
        instance_lb_alls = [[] for _ in range(len(lbs))]
        for item in lb_alls:
            for i in range(len(instance_lb_alls)):
                instance_lb_alls[i].append(item[i:i+1])
        instance_up_alls = [[] for _ in range(len(lbs))]
        for item in up_alls:
            for i in range(len(instance_up_alls)):
                instance_up_alls[i].append(item[i:i+1])

        batch = len(histories)
        decision_threshs = decision_threshs.to(lbs[0].device, non_blocking=True)
        for i in range(batch):
            infeasible = False
            if (lbs[i] <= decision_threshs[i]).all():
                if check_infeasibility:
                    for ii, (l, u) in enumerate(zip(lb_alls[i][1:-1], up_alls[i][1:-1])):
                        if (l-u).max() > 1e-6:
                            infeasible = True
                            print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                            break

                if not infeasible:
                    priority=0
                    new_history = copy.deepcopy(histories[i])
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
                    left = ReLUDomain(instance_lAs[i], lbs[i], ubs[i], instance_lb_alls[i], instance_up_alls[i],
                                      instance_slopes[i],
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
                    for ii, (l, u) in enumerate(zip(lb_alls[i+batch][1:-1], up_alls[i+batch][1:-1])):
                        if (l-u).max() > 1e-6:
                            infeasible = True
                            print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                            break

                if not infeasible:
                    priority=0
                    new_history = copy.deepcopy(histories[i])
                    if branching_decisions is not None:
                        new_history[branching_decisions[i][0]][0].append(branching_decisions[i][1])  # second half batch: inactive neurons
                        new_history[branching_decisions[i][0]][1].append(-1.0)  # second half batch: inactive neurons

                    right_primals = None
                    right = ReLUDomain(instance_lAs[i+batch], lbs[i+batch], ubs[i+batch], instance_lb_alls[i+batch], instance_up_alls[i+batch],
                                       instance_slopes[i+batch],  betas[i+batch], depths[i+batch]+1, split_history=split_histories[i+batch],
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


class BatchedReLUDomainList(AbstractReLUDomainList):
    """
        Maintains an unsorted but batched list of domain list, which is more efficient than SortedReLUDomainList
    """

    def __init__(self, lAs, global_lbs, global_ubs, lb_alls, ub_alls, slopes, history, depths, Cs,
                 thresholds, beta, num, tensor_container=TensorStorage, interm_transfer=True):
        super(BatchedReLUDomainList, self).__init__()
        self.all_lAs = [tensor_container(item.cpu()) for item in lAs]
        self.all_global_lbs = tensor_container(global_lbs[:num].cpu())
        self.all_global_ubs = tensor_container(global_ubs[:num].cpu())
        self.all_lb_alls = [tensor_container(item[:num].cpu()) for item in lb_alls]
        self.all_ub_alls = [tensor_container(item[:num].cpu()) for item in ub_alls]
        self.all_slopes = defaultdict(dict)
        for k in slopes:
            self.all_slopes[k] = {}
            for kk, v in slopes[k].items():
                if kk not in self.all_slopes[k]:
                    self.all_slopes[k][kk] = tensor_container(v[:, :, :num].cpu(), concat_dim=2)
                else:
                    self.all_slopes[k][kk].append(v[:, :, :num].cpu())

        if thresholds.numel() > 1:
            self.all_thresholds = tensor_container(thresholds[:num].cpu())
        else:
            thresholds = thresholds.view(1, 1)
            self.all_thresholds = tensor_container(torch.cat([thresholds] * num).cpu())

        self.Cs = tensor_container(Cs[:num].cpu())

        # === seperator, things above are big tensors, things below are lists ===

        self.all_betas = [beta if beta is not None else None for i in range(num)]
        self.all_intermediate_betas = [None for _ in range(num)]

        self.histories = [copy.deepcopy(history) for _ in range(num)]
        self.split_histories = [[] for _ in range(num)]
        self.depths = depths.copy()

        # tracker of number of domains
        self.l = 0
        self.u = len(self.histories)

        # === save things for statical intermediate bound ===
        self.interm_transfer = interm_transfer
        if not interm_transfer:
            self.static_lb = [lb[0:1].to(device='cuda', non_blocking=True)
                              if torch.cuda.is_available() else lb[0:1] for lb in self.all_lb_alls[:-1]]
            self.static_ub = [ub[0:1].to(device='cuda', non_blocking=True)
                              if torch.cuda.is_available() else ub[0:1] for ub in self.all_ub_alls[:-1]]
        else:
            self.static_lb = self.static_ub = None

    def pick_out(self, batch, device='cpu'):
        assert batch > 0
        batch = int(batch)
        if not isinstance(batch, int):
            raise Exception('Strange batch')

        if torch.cuda.is_available(): torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self), batch)

        new_lAs = [lA.pop(batch).to(device=device, non_blocking=True) for lA in self.all_lAs]

        if not self.interm_transfer:
            # place dummy place to record interm bounds
            lower_bounds = [None] * len(self.static_lb) + [self.all_lb_alls[-1].pop(batch).to(device=device, non_blocking=True)]
            upper_bounds = [None] * len(self.static_ub) + [self.all_ub_alls[-1].pop(batch).to(device=device, non_blocking=True)]
            # repeat static_lb and static_ub when necessary
            for i in range(len(self.static_lb)):
                # enlarge the batch size in the static storage
                if batch > self.static_lb[i].shape[0]:
                    power = (batch + self.static_lb[i].shape[0] - 1) // self.static_lb[i].shape[0]
                    self.static_lb[i] = self.static_lb[i].repeat(power, *tuple([1] * (self.static_lb[i].dim() - 1)))
                if batch > self.static_ub[i].shape[0]:
                    power = (batch + self.static_ub[i].shape[0] - 1) // self.static_ub[i].shape[0]
                    self.static_ub[i] = self.static_ub[i].repeat(power, *tuple([1] * (self.static_ub[i].dim() - 1)))
        else:
            lower_bounds = [lower_bound.pop(batch) for lower_bound in self.all_lb_alls]
            upper_bounds = [upper_bound.pop(batch) for upper_bound in self.all_ub_alls]

            lower_bounds = [item.to(device=device, non_blocking=True) for item in lower_bounds]
            upper_bounds = [item.to(device=device, non_blocking=True) for item in upper_bounds]

        global_lb = self.all_global_lbs.pop(batch).to(device=device, non_blocking=True)
        global_ub = self.all_global_ubs.pop(batch).to(device=device, non_blocking=True)

        slopes = defaultdict(dict)
        for k, v_map in self.all_slopes.items():
            slopes[k] = {}
            for kk, vv in v_map.items():
                slopes[k][kk] = vv.pop(batch).to(device=device, non_blocking=True)

        betas_all = self.all_betas[self.u - batch: self.u]
        intermediate_betas_all = self.all_intermediate_betas[self.u - batch: self.u]

        cs = self.Cs.pop(batch).to(device=device, non_blocking=True)
        thresholds = self.all_thresholds.pop(batch).to(device=device, non_blocking=True)

        selected_candidate_domains = self._assemble_fake_domains(global_lb, global_ub,
                                                        self.histories[self.u - batch: self.u],
                                                        self.split_histories[self.u - batch: self.u],
                                                        self.depths[self.u - batch: self.u],
                                                        thresholds)

        if not self.interm_transfer:
            # need to fill in the slots
            local_histories = self.histories[self.u - batch: self.u]
            for i in range(len(self.static_lb)):
                # setting positive and negative neurons with lb 0 or ub 0
                lb_row_selector, lb_col_selector, ub_row_selector, ub_col_selector = [], [], [], []
                for j, hist in enumerate(local_histories):
                    for idx, direc in zip(hist[i][0], hist[i][1]):
                        if direc >= 0:
                            lb_row_selector.append(j)
                            lb_col_selector.append(idx)
                        else:
                            ub_row_selector.append(j)
                            ub_col_selector.append(idx)

                now_lb = self.static_lb[i][0: batch].clone()
                if len(lb_row_selector) > 0:
                    now_lb.view(batch, -1)[lb_row_selector, lb_col_selector] = 0.0
                now_ub = self.static_ub[i][0: batch].clone()
                if len(ub_row_selector) > 0:
                    now_ub.view(batch, -1)[ub_row_selector, ub_col_selector] = 0.0
                lower_bounds[i] = now_lb
                upper_bounds[i] = now_ub

        new_masks = [torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float().to(device=device, non_blocking=True)
                     for j in range(len(lower_bounds) - 1)]

        self.u -= batch

        self.all_betas = self.all_betas[:self.u]
        self.all_intermediate_betas = self.all_intermediate_betas[:self.u]
        self.histories = self.histories[:self.u]
        self.split_histories = self.split_histories[:self.u]
        self.depths = self.depths[:self.u]

        return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains, cs, thresholds


    def add(self, lAs, lbs, ubs, lb_alls, up_alls, histories, depths, slopes, betas,
             split_histories, branching_decisions, decision_threshs,
             intermediate_betas, check_infeasibility, Cs, num):

        batch = len(histories)
        decision_threshs = decision_threshs.to(lbs[0].device, non_blocking=True)

        left_indexer = torch.all(lbs[:batch] <= decision_threshs[:batch], dim=1).nonzero().view(-1)
        left_indexer_lst = left_indexer.tolist()
        if check_infeasibility:
            for ii, (l, u) in enumerate(zip(lb_alls[left_indexer][1:-1], up_alls[left_indexer][1:-1])):
                if l is not None and u is not None:
                    if (l-u).max() > 1e-6:
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        return
        for idx in left_indexer_lst:
            new_history = copy.deepcopy(histories[idx])
            if branching_decisions is not None:
                new_history[branching_decisions[idx][0]][0].append(branching_decisions[idx][1])
                new_history[branching_decisions[idx][0]][1].append(+1.0)

                if branching_decisions[idx][1] in histories[idx][branching_decisions[idx][0]][0]:
                    print('BUG!!! repeated split!')
                    print(histories[idx])
                    print(branching_decisions[idx])
                    raise RuntimeError

            self.histories.append(new_history)
            self.all_betas.append(betas[idx])
            self.all_intermediate_betas.append(intermediate_betas[idx])
            self.split_histories.append(split_histories[idx])
            self.depths.append(depths[idx] + 1)

        [lA.append(new_lA[left_indexer]) for lA, new_lA in zip(self.all_lAs, lAs)]
        self.all_global_lbs.append(lbs[left_indexer])
        self.all_global_ubs.append(ubs[left_indexer])
        [lb.append(new_lb[left_indexer]) if new_lb is not None else None for lb, new_lb in zip(self.all_lb_alls, lb_alls)]
        [up.append(new_up[left_indexer]) if new_up is not None else None for up, new_up in zip(self.all_ub_alls, up_alls)]
        for k, v in self.all_slopes.items():
            for kk, vv in v.items():
                vv.append(slopes[k][kk][:,:,left_indexer])
        self.all_thresholds.append(decision_threshs[left_indexer])
        self.Cs.append(Cs[left_indexer])

        # ============

        right_indexer = torch.all(lbs[batch: (2*batch)] <= decision_threshs[:batch], dim=1).nonzero().view(-1)
        right_indexer_lst = right_indexer.tolist()
        if check_infeasibility:
            for ii, (l, u) in enumerate(zip(lb_alls[right_indexer + batch][1:-1], up_alls[right_indexer + batch])):
                if (l-u).max() > 1e-6:
                    print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                    return
        for idx in right_indexer_lst:
            new_history = copy.deepcopy(histories[idx])
            if branching_decisions is not None:
                new_history[branching_decisions[idx][0]][0].append(branching_decisions[idx][1])
                new_history[branching_decisions[idx][0]][1].append(-1.0)

            self.histories.append(new_history)
            self.all_betas.append(betas[idx + batch])
            self.all_intermediate_betas.append(intermediate_betas[idx + batch])
            self.split_histories.append(split_histories[idx + batch])
            self.depths.append(depths[idx + batch] + 1)

        [lA.append(new_lA[right_indexer + batch]) for lA, new_lA in zip(self.all_lAs, lAs)]
        self.all_global_lbs.append(lbs[right_indexer + batch])
        self.all_global_ubs.append(ubs[right_indexer + batch])
        [lb.append(new_lb[right_indexer + batch]) if new_lb is not None else None for lb, new_lb in zip(self.all_lb_alls, lb_alls)]
        [up.append(new_up[right_indexer + batch]) if new_up is not None else None for up, new_up in zip(self.all_ub_alls, up_alls)]
        for k, v in self.all_slopes.items():
            for kk, vv in v.items():
                vv.append(slopes[k][kk][:,:,right_indexer + batch])
        self.all_thresholds.append(decision_threshs[right_indexer])
        self.Cs.append(Cs[right_indexer + batch])

        self.u = len(self.histories)

    def _assemble_fake_domains(self, global_lbs, global_ubs, history, split_history, depth, thresholds):
        ans = []
        for now_lb, now_ub, now_his, now_split_his, now_depth, now_threshold in zip(global_lbs, global_ubs, history, split_history, depth, thresholds):
            now_obj = SimplifiedReLUDomain()
            now_obj.history = now_his
            now_obj.split_history = now_split_his
            now_obj.depth = now_depth
            now_obj.lower_bound = now_lb
            now_obj.upper_bound = now_ub
            now_obj.threshold = now_threshold
            ans.append(now_obj)
        return ans

    def get_min_domain(self, num, rev_order=False):
        indices = (self.all_global_lbs - self.all_thresholds).max(dim=1)[0].argsort()[:num]
        indices_lst = indices.tolist()

        selected_candidate_domains = self._assemble_fake_domains(self.all_global_lbs[indices],
                                                                 self.all_global_ubs[indices],
                                                                 [self.histories[i] for i in indices_lst],
                                                                 [self.split_histories[i] for i in indices_lst],
                                                                 [self.depths[i] for i in indices_lst],
                                                                 self.all_thresholds[indices])
        return selected_candidate_domains


    def __len__(self):
        return self.u - self.l

    def __getitem__(self, index):
        selected_candidate_domains = self._assemble_fake_domains(self.all_global_lbs[index: index + 1],
                                                                 self.all_global_ubs[index: index + 1],
                                                                 [self.histories[index]],
                                                                 [self.split_histories[index]],
                                                                 [self.depths[index]],
                                                                 self.all_thresholds[index: index + 1])
        return selected_candidate_domains[0]

class SimplifiedReLUDomain:
    """
        This class is used by BatchedReLUDomainList to store list-indexed data that are queried by caller functions
    """
    def __init__(self):
        pass

class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.
    """

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None,
                up_all=None, slope=None, beta=None, depth=None, split_history=None,
                history=None, gnn_decision=None, intermediate_betas=None, primals=None,
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
        self.dm_l = None
        self.dm_u = None
        # record c for each domain so that the domain list is capable of saving multiple c's
        # c shape here: (1,1,num_outputs)
        self.c = c
        self.threshold = threshold
        if type(threshold) == int:
            self.threshold = np.float64(threshold)

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

    def attack_criterion(self):
        return (self.upper_bound <= self.threshold).all()

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
            # self.beta = [b.to(device='cpu', non_blocking=True) for b in self.beta]
        
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device='cpu', non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device='cpu', non_blocking=True)
        if self.c is not None:
            self.c = self.c.to(device='cpu', non_blocking=True)
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
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device, non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device, non_blocking=True)

        if self.c is not None:
            self.c = self.c.to(device, non_blocking=True)
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
                    primals=self.primals, priority=self.priority, c=self.c)
        
        return dive_d



def add_domain(candidate, domains):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    """
    domains.add(candidate.to_cpu())


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


def merge_domains_params(domains_params, dive_domains_params):
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs = domains_params
    dive_mask, dive_lAs, dive_orig_lbs, dive_orig_ubs, dive_slopes, dive_betas,\
            dive_intermediate_betas, dive_selected_domains, dive_cs = dive_domains_params
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

    merge_cs = torch.cat([cs, dive_cs], dim=0)

    return merge_mask, merge_lAs, merge_orig_lbs, merge_orig_ubs, merge_slopes, merge_betas,\
                    merge_intermediate_betas, merge_selected_domains, merge_cs
