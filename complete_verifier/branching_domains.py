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
import time
import operator
from types import SimpleNamespace
import arguments

from tensor_storage import TensorStorage
from utils import fast_hist_copy, check_infeasible_bounds
from cuts.implied_cuts import pick_out_implied


class AbstractDomainList():
    """Abstract class that maintains the list of domains (variables on CPUs)."""

    def __init__(self):
        pass

    def pick_out(self, batch, device):
        raise NotImplementedError

    def add(self, bounds, histories, depths, split,
            decision_threshs, check_infeasibility):
        raise NotImplementedError

    def get_min_domain(self, num, rev_order=False):
        # need to return a list of obj, each object has
        # lower_bounds, upper_bounds, threshold, and depth
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def print(self, n=20):
        bab_args = arguments.Config["bab"]
        get_upper_bound = bab_args["get_upper_bound"]
        if len(self) > 0:
            if get_upper_bound:
                print('Current worst splitting domains [lb, ub] (depth):')
            else:
                print('Current worst splitting domains lb-rhs (depth):')
            if (arguments.Config["bab"]["cut"]["enabled"]
                    and bab_args["cut"]["cplex_cuts"]
                    and bab_args["cut"]["cplex_cuts_revpickup"]):
                printed_d = self.get_min_domain(n, rev_order=True)
            else:
                printed_d = self.get_min_domain(n)
            for i in printed_d:
                if get_upper_bound:
                    print(f'[{(i.lower_bound - i.threshold).max():.5f}, '
                          f'{(i.upper_bound - i.threshold).min():5f}] ({i.depth})', end=', ')
                else:
                    print(f'{(i.lower_bound - i.threshold).max():.5f} ({i.depth})', end=', ')
            print()


class BatchedDomainList(AbstractDomainList):
    """An unsorted but batched list of domain list."""

    def __init__(self, ret, lAs, global_lbs, global_ubs,
                 alphas=None, history=None, thresholds=None,
                 net=None, branching_input_and_activation=False, x=None):
        super().__init__()

        lb_alls, ub_alls = ret['lower_bounds'], ret['upper_bounds']
        beta = ret.get('betas', None)
        num = global_lbs.shape[0]
        assert len(lb_alls) == len(ub_alls)

        self.net = net

        self.all_lAs = {k: TensorStorage(v.cpu()) for k, v in lAs.items()}
        self.all_global_lbs = TensorStorage(global_lbs.cpu())
        self.all_global_ubs = TensorStorage(global_ubs.cpu())
        self.all_lb_alls = {k: TensorStorage(v.cpu()) for k, v in lb_alls.items()}
        self.all_ub_alls = {k: TensorStorage(v.cpu()) for k, v in ub_alls.items()}

        self.all_alphas = defaultdict(dict)
        for k in alphas:
            self.all_alphas[k] = {}
            for kk, v in alphas[k].items():
                if kk not in self.all_alphas[k]:
                    self.all_alphas[k][kk] = TensorStorage(
                        v[:, :].cpu(), concat_dim=2)
                else:
                    self.all_alphas[k][kk].append(v[:, :].cpu())
        if thresholds.numel() > 1:
            self.all_thresholds = TensorStorage(thresholds.cpu())
        else:
            thresholds = thresholds.view(1, 1)
            self.all_thresholds = TensorStorage(torch.cat([thresholds] * num).cpu())
        self.Cs = TensorStorage(net.c.cpu())

        # === seperator, things above are big tensors, things below are lists ===
        self.all_betas = [beta for _ in range(num)]
        self.all_intermediate_betas = [None for _ in range(num)]
        self.histories = [fast_hist_copy(history) for _ in range(num)]
        self.split_histories = [[] for _ in range(num)]
        self.depths = [0] * num

        # tracker of number of domains
        self.l = 0
        self.u = len(self.histories)

        # === save things for statical intermediate bound ===
        self.interm_transfer = arguments.Config['bab']['interm_transfer']
        self.final_name = net.final_name
        if not self.interm_transfer:
            self.static_lb = {k: (lb[0:1].to(device='cuda', non_blocking=True)
                              if torch.cuda.is_available() else lb[0:1])
                              for k, lb in self.all_lb_alls.items()
                              if k != self.final_name}
            self.static_ub = {k: (ub[0:1].to(device='cuda', non_blocking=True)
                              if torch.cuda.is_available() else ub[0:1])
                              for k, ub in self.all_ub_alls.items()
                              if k != self.final_name}
        else:
            self.static_lb = self.static_ub = None

        self.branching_input_and_activation = branching_input_and_activation
        if branching_input_and_activation:
            input_split_idx = ret['input_split_idx']
            self.all_input_split_idx = TensorStorage(input_split_idx.cpu())
            self.all_x_Ls = TensorStorage(x.ptb.x_L.cpu())
            self.all_x_Us = TensorStorage(x.ptb.x_U.cpu())
        else:
            self.all_input_split_idx = self.all_x_Ls = self.all_x_Us = None

    def sort(self):
        """
        Sort all domains based the the margin between lower bounds and thresholds.
        """
        sort_time = time.time()
        N = self.u
        if N <= 0:
            return
        global_lb = self.all_global_lbs[:N]
        thresholds = self.all_thresholds[:N]
        indices = (global_lb - thresholds).max(dim=1).values.argsort(descending=True)
        indices_lst = indices.tolist()

        # Reorder all single tensor storage items or list of items.
        for item in [
            self.all_x_Ls, self.all_x_Us, self.all_input_split_idx,
            self.all_global_lbs, self.all_global_ubs, self.Cs,
            self.all_thresholds
        ] + (list(self.all_lAs.values()) + list(self.all_lb_alls.values())
             + list(self.all_ub_alls.values())):
            if item is not None:
                item._storage[:N] = item._storage[indices]

        # Reorder alphas (dictionary of dictionary).
        for v in self.all_alphas.values():
            for vv in v.values():
                # The batch dimension is at 2 for alphas.
                vv._storage[:,:,:N,:] = (
                    vv._storage[:,:,:N,:].index_select(index=indices, dim=2))

        # Reorder lists.
        selector = operator.itemgetter(*indices_lst) if len(
                indices_lst) > 1 else lambda _arr: (_arr[indices_lst[0]], )
        for name in ('histories', 'depths', 'split_histories',
                'all_betas', 'all_intermediate_betas'):
            item = getattr(self, name)
            if item is not None:
                setattr(self, name, list(selector(item)))
        sort_time = time.time() - sort_time
        print(f'Sorting batched domains takes {sort_time} seconds.')

    def pick_out(self, batch, device='cpu', impl_params=None):
        def _to(x, non_blocking=True):
            return x.to(device=device, non_blocking=non_blocking)

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        assert batch > 0
        batch = min(len(self), int(batch))

        new_lAs = {k: _to(lA.pop(batch)) for k, lA in self.all_lAs.items()}
        new_x_Ls = _to(self.all_x_Ls.pop(batch)) if self.all_x_Ls is not None else None
        new_x_Us = _to(self.all_x_Us.pop(batch)) if self.all_x_Us is not None else None
        new_input_split_idx = (_to(self.all_input_split_idx.pop(batch))
                               if self.all_input_split_idx is not None else None)
        global_lb = _to(self.all_global_lbs.pop(batch))
        alphas = defaultdict(dict)
        for k, v_map in self.all_alphas.items():
            alphas[k] = {kk: _to(vv.pop(batch)) for kk, vv in v_map.items()}
        betas_all = self.all_betas[self.u - batch: self.u]
        intermediate_betas_all = self.all_intermediate_betas[self.u - batch: self.u]
        cs = _to(self.Cs.pop(batch))
        thresholds = _to(self.all_thresholds.pop(batch))

        if self.interm_transfer:
            lower_bounds = {k: _to(v.pop(batch))
                            for k, v in self.all_lb_alls.items()}
            upper_bounds = {k: _to(v.pop(batch))
                            for k, v in self.all_ub_alls.items()}
        else:
            lower_bounds, upper_bounds = self._interm_bounds_wo_transfer(
                batch, device, impl_params)
        # TODO Move to beta_CROWN_solver. Duplicate code.
        new_masks = {}
        for k in lower_bounds:
            if k not in self.net.split_activations:
                continue
            mask = None
            for activation, index in self.net.split_activations[k]:
                mask_ = _to(
                    activation.get_split_mask(
                        lower_bounds[k], upper_bounds[k], index
                    ).flatten(1).float())
                mask = mask_ if mask is None else torch.logical_or(mask, mask_)
            if mask is None:
                mask = torch.ones_like(lower_bounds[k]).flatten(1)
            new_masks[k] = mask

        history = self.histories[self.u - batch: self.u]
        split_history = self.split_histories[self.u - batch: self.u]
        depths = self.depths[self.u - batch: self.u]
        self.u -= batch

        self.all_betas = self.all_betas[:self.u]
        self.all_intermediate_betas = self.all_intermediate_betas[:self.u]
        self.histories = self.histories[:self.u]
        self.split_histories = self.split_histories[:self.u]
        self.depths = self.depths[:self.u]

        return {
            'mask': new_masks, 'lAs': new_lAs,
            'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds,
            'alphas': alphas, 'betas': betas_all,
            'intermediate_betas': intermediate_betas_all,
            'history': history, 'split_history': split_history,
            'global_lb': global_lb, 'depths': depths, 'cs': cs,
            'thresholds': thresholds, 'x_Ls': new_x_Ls, 'x_Us': new_x_Us,
            'input_split_idx': new_input_split_idx,
        }

    def add(self, bounds, d, check_infeasibility):
        histories = d['history']
        decision_threshs = d['thresholds']
        final_lower = bounds['lower_bounds'][self.final_name]
        batch = final_lower.size(0)
        device = final_lower.device
        decision_threshs = decision_threshs.to(device, non_blocking=True)
        assert (self.all_x_Ls is None) == (bounds['x_Ls'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_x_Us is None) == (bounds['x_Us'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_input_split_idx is None) == (bounds['input_split_idx'] is None), "Inconsistent input_split_idx during construction and using {type(self)}"
        assert len(self.all_lAs) == len(bounds['lAs'])

        if check_infeasibility:
            infeasible = check_infeasible_bounds(
                bounds['lower_bounds'], bounds['upper_bounds'])
        else:
            infeasible = None

        # torch.all() is when taking max for multiple output specifications
        indexer = torch.all(
            bounds['lower_bounds'][self.final_name] <= decision_threshs[:batch],
            dim=1)
        if infeasible is not None:
            indexer = torch.logical_and(indexer, torch.logical_not(infeasible))
        indexer = indexer.nonzero().view(-1)
        if len(indexer) == 0:
            return
        # Add all list items in batch without for loop.
        if len(indexer) > 0:
            # itemgetter returns a value instead of tuple when length is 1, so need a special case.
            batch_indexer_lst = indexer.tolist()
            selector = (
                operator.itemgetter(*batch_indexer_lst)
                if len(batch_indexer_lst) > 1
                else lambda _arr: (_arr[batch_indexer_lst[0]], ))
            self.histories.extend(selector(histories))
            self.all_betas.extend(selector(bounds['betas']))
            self.all_intermediate_betas.extend(selector(bounds['intermediate_betas']))
            self.split_histories.extend(selector(bounds['split_history']))
            self.depths.extend(selector(d['depths']))
        for k in self.all_lAs:
            self.all_lAs[k].append(bounds['lAs'][k][indexer])
        if self.all_x_Ls is not None:
            self.all_x_Ls.append(bounds['x_Ls'][indexer])
        if self.all_x_Us is not None:
            self.all_x_Us.append(bounds['x_Us'][indexer])
        if self.all_input_split_idx is not None:
            self.all_input_split_idx.append(bounds['input_split_idx'][indexer])
        self.all_global_lbs.append(bounds['lower_bounds'][self.final_name][indexer])
        self.all_global_ubs.append(bounds['upper_bounds'][self.final_name][indexer])
        for k, v in bounds['lower_bounds'].items():
            self.all_lb_alls[k].append(v[indexer])
        for k, v in bounds['upper_bounds'].items():
            self.all_ub_alls[k].append(v[indexer])
        alpha_new = alpha_reuse = False
        for k, v in bounds['alphas'].items():
            if k not in self.all_alphas:
                self.all_alphas[k] = {}
            for kk, vv in v.items():
                if kk not in self.all_alphas[k]:
                    # This is the first time to create these alpha TensorStorage
                    self.all_alphas[k][kk] = TensorStorage(vv[:,:,indexer].cpu(), concat_dim=2)
                    alpha_new = True
                else:
                    # Reusing existing TensorStorage
                    self.all_alphas[k][kk].append(vv[:,:,indexer])
                    alpha_reuse = True
        assert not (alpha_new and alpha_reuse)
        self.all_thresholds.append(decision_threshs[indexer])
        self.Cs.append(bounds['c'][indexer])
        self.u = len(self.histories)

    def _interm_bounds_wo_transfer(self, batch, device, impl_params):
        def _to(x, non_blocking=True):
            return x.to(device=device, non_blocking=non_blocking)

        # place dummy place to record interm bounds
        lower_bounds, upper_bounds = {}, {}
        lower_bounds[self.final_name] = _to(
            self.all_lb_alls[self.final_name].pop(batch))
        upper_bounds[self.final_name] = _to(
            self.all_ub_alls[self.final_name].pop(batch))
        # FIXME This part of code looks unreliable. Need to be clear when
        # this situation might happen.
        # repeat static_lb and static_ub when necessary
        for k in self.static_lb:
            # enlarge the batch size in the static storage
            if batch > self.static_lb[k].shape[0]:
                power = (batch + self.static_lb[k].shape[0] - 1) // self.static_lb[k].shape[0]
                self.static_lb[k] = self.static_lb[k].repeat(
                    power, *tuple([1] * (self.static_lb[k].dim() - 1)))
            if batch > self.static_ub[k].shape[0]:
                power = (batch + self.static_ub[k].shape[0] - 1) // self.static_ub[k].shape[0]
                self.static_ub[k] = self.static_ub[k].repeat(
                    power, *tuple([1] * (self.static_ub[k].dim() - 1)))
        # need to fill in the slots
        histories = self.histories[self.u - batch: self.u]
        # setting positive and negative neurons with lb 0 or ub 0
        lb_row_selector = {k: [] for k in self.static_lb}
        lb_col_selector = {k: [] for k in self.static_lb}
        ub_row_selector = {k: [] for k in self.static_lb}
        ub_col_selector = {k: [] for k in self.static_lb}

        for k in self.static_lb:
            lower_bounds[k] = self.static_lb[k][:batch].clone()
            upper_bounds[k] = self.static_ub[k][:batch].clone()

        if impl_params is not None:
            pick_out_implied(impl_params, histories, lower_bounds, upper_bounds,
                             lb_row_selector, lb_col_selector,
                             ub_row_selector, ub_col_selector)
        else:
            # TODO merge the two loops
            for i in self.static_lb:
                for j, hist in enumerate(histories):
                    if isinstance(hist[i][0], torch.Tensor):
                        hist0 = hist[i][0].__array__()
                    else:
                        hist0 = hist[i][0]
                    if isinstance(hist[i][1], torch.Tensor):
                        hist1 = hist[i][1].__array__()
                    else:
                        hist1 = hist[i][1]
                    for k in range(len(hist0)):
                        idx = hist0.item(k)
                        direc = hist1.item(k)
                        if direc >= 0:
                            lb_row_selector[i].append(j)
                            lb_col_selector[i].append(idx)
                        else:
                            ub_row_selector[i].append(j)
                            ub_col_selector[i].append(idx)
                if len(lb_row_selector[i]) > 0:
                    lower_bounds[i].view(batch, -1)[
                        lb_row_selector[i], lb_col_selector[i]] = 0.0
                if len(ub_row_selector[i]) > 0:
                    upper_bounds[i].view(batch, -1)[
                        ub_row_selector[i], ub_col_selector[i]] = 0.0

        return lower_bounds, upper_bounds

    def _assemble_domains(self, global_lbs, global_ubs, history, split_history, depth, thresholds):
        ans = []
        for lb, ub, now_his, now_split_his, now_depth, now_threshold in zip(global_lbs, global_ubs, history, split_history, depth, thresholds):
            now_obj = SimpleNamespace()
            now_obj.history = now_his
            now_obj.split_history = now_split_his
            now_obj.depth = now_depth
            now_obj.lower_bound = lb
            now_obj.upper_bound = ub
            now_obj.threshold = now_threshold
            ans.append(now_obj)
        return ans

    def get_min_domain(self, num, rev_order=False):
        indices = (self.all_global_lbs - self.all_thresholds).max(dim=1)[0].argsort()[:num]
        indices_lst = indices.tolist()

        selected_candidate_domains = self._assemble_domains(
            self.all_global_lbs[indices],
            self.all_global_ubs[indices],
            [self.histories[i] for i in indices_lst],
            [self.split_histories[i] for i in indices_lst],
            [self.depths[i] for i in indices_lst],
            self.all_thresholds[indices])
        return selected_candidate_domains

    def __len__(self):
        return self.u - self.l

    def __getitem__(self, index):
        selected_candidate_domains = self._assemble_domains(
            self.all_global_lbs[index: index + 1],
            self.all_global_ubs[index: index + 1],
            [self.histories[index]],
            [self.split_histories[index]],
            [self.depths[index]],
            self.all_thresholds[index: index + 1])
        return selected_candidate_domains[0]


def check_worst_domain(d):
    use_cuts = (arguments.Config["bab"]["cut"]["enabled"]
                and arguments.Config["bab"]["cut"]["cplex_cuts"]
                and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"])
    if len(d) > 0:
        if use_cuts:
            worst_domain = d.get_min_domain(1, rev_order=True)
            global_lb = worst_domain[-1].lower_bound - worst_domain[-1].threshold
        else:
            worst_domain = d.get_min_domain(1, rev_order=False)
            global_lb = worst_domain[0].lower_bound - worst_domain[0].threshold
        return global_lb
    else:
        return torch.tensor(1e-7)
