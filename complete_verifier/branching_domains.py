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
import torch
import copy
from collections import defaultdict

import operator
from types import SimpleNamespace
import arguments
from itertools import islice, chain
from collections import deque
import math

from tensor_storage import get_tensor_storage
from domain_clipper import update_interm_bounds
from utils import fast_hist_copy, check_infeasible_bounds, convert_history_from_list


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

    def __init__(self, ret, c, lAs, global_lbs, global_ubs,
                 alphas=None, history=None, thresholds=None,
                 net=None, branching_input_and_activation=False, x=None,
                 enable_clip_domains=False):
        super().__init__()

        lb_alls, ub_alls = ret['lower_bounds'], ret['upper_bounds']
        beta = ret.get('betas', None)
        num = global_lbs.shape[0]
        assert len(lb_alls) == len(ub_alls)

        self.net = net

        self.all_lAs = {k: get_tensor_storage(v.cpu()) for k, v in lAs.items()}
        self.all_global_lbs = get_tensor_storage(global_lbs.cpu())
        self.all_global_ubs = get_tensor_storage(global_ubs.cpu())
        self.all_lb_alls = {k: get_tensor_storage(v.cpu()) for k, v in lb_alls.items()}
        self.all_ub_alls = {k: get_tensor_storage(v.cpu()) for k, v in ub_alls.items()}

        self.all_alphas = defaultdict(dict)
        for k in alphas:
            self.all_alphas[k] = {}
            for kk, v in alphas[k].items():
                if kk not in self.all_alphas[k]:
                    self.all_alphas[k][kk] = get_tensor_storage(
                        v[:, :].cpu(), concat_dim=2)
                else:
                    self.all_alphas[k][kk].append(v[:, :].cpu())
        if thresholds.numel() > 1:
            self.all_thresholds = get_tensor_storage(thresholds.cpu())
        else:
            thresholds = thresholds.view(1, 1)
            self.all_thresholds = get_tensor_storage(torch.cat([thresholds] * num).cpu())
        self.Cs = get_tensor_storage(c.cpu())

        # === seperator, things above are big tensors, things below are lists ===
        self.all_betas = [beta for _ in range(num)]
        self.all_intermediate_betas = [None for _ in range(num)]
        self.histories = [fast_hist_copy(history) for _ in range(num)]
        self.split_histories = [[] for _ in range(num)]
        self.depths = [0] * num
        if arguments.Config['bab']['tree_traversal'] == 'breadth_first':
            self.all_betas = deque(self.all_betas)
            self.all_intermediate_betas = deque(self.all_intermediate_betas)
            self.histories = deque(self.histories)
            self.split_histories = deque(self.split_histories)
            self.depths = deque(self.depths)

        # tracker of number of domains
        self.num_domains = len(self.histories)

        # === save things for statical intermediate bound ===
        self.interm_transfer = arguments.Config['bab']['interm_transfer']
        self.final_name = net.final_name
        self.static_lb = {k: (lb[0:1].to(device=net.device, non_blocking=True)
                              if torch.cuda.is_available() else lb[0:1])
                              for k, lb in self.all_lb_alls.items()
                              if k != self.final_name}
        self.static_ub = {k: (ub[0:1].to(device=net.device, non_blocking=True)
                              if torch.cuda.is_available() else ub[0:1])
                              for k, ub in self.all_ub_alls.items()
                              if k != self.final_name}
        self.unstable_mask = {}
        self.unstable_interm_bounds = None

        self.branching_input_and_activation = branching_input_and_activation
        self.enable_clip_domains = enable_clip_domains
        if branching_input_and_activation or enable_clip_domains:
            if branching_input_and_activation:
                input_split_idx = ret['input_split_idx']
                self.all_input_split_idx = get_tensor_storage(input_split_idx.cpu())
            else:
                self.all_input_split_idx = None
            self.all_x_Ls = get_tensor_storage(x.ptb.x_L.cpu())
            self.all_x_Us = get_tensor_storage(x.ptb.x_U.cpu())
        else:
            self.all_input_split_idx = self.all_x_Ls = self.all_x_Us = None

    def update_unstable_mask(self, updated_mask):
        """
        Update the unstable mask for each layer by mapping masks to the correct
        input nodes using the network graph.

        @params:    updated_mask (dict): A dictionary where keys are operation node names
                                    and values are lists of unstable masks.

        @init:      self.unstable_mask (dict): The dictionary to store the final mask
                                    for each corresponding input node.
        """
        # Iterate over each operation node and its list of calculated masks.
        for op_node_name, mask_list in updated_mask.items():
            # Get the actual operation node object from the network graph.
            node = self.net.net[op_node_name]

            # Get the names of all input nodes for this operation.
            input_node_names = [inp.name for inp in node.inputs]

            # A mask in mask_list corresponds to an input node in the same order.
            # For example, mask_list[0] is for input_node_names[0].
            # We zip them together to create the correct (name, mask) pairs.
            for input_name, mask in zip(input_node_names, mask_list):
                # mask can be None if the corresponding input node is not perturbed.
                if mask is not None:
                    # We only record the mask if the corresponding input node is perturbed.
                    self.unstable_mask[input_name] = mask if mask.any() else None

    def sort(self):
        """
        Sort all domains based on the margin between their global lower bounds and thresholds.
        
        The sorting is applied to all associated storage items so that the domains with the
        largest margin (i.e. the most promising domains) are processed first.
        """
        # Determine the number of active domains.
        # All storages share the same number of used domains, stored in self.num_domains.
        N = self.num_domains
        if N <= 0:
            return

        # Slice the arrays for active domains.
        global_lb = self.all_global_lbs[:N]
        thresholds = self.all_thresholds[:N]

        # Calculate the margin between each domain's global lower bound and threshold.
        # The margin is computed as the maximum difference along dimension 1.
        # Sort the domain indices in descending order based on these margins.
        indices = (global_lb - thresholds).max(dim=1).values.argsort(descending=True)
        indices_lst = indices.tolist()

        # === Reorder tensor-based storage items ===
        # Create an iterator over all items without extra list allocations.
        for item in chain(
            [
                self.all_x_Ls,
                self.all_x_Us,
                self.all_input_split_idx,
                self.all_global_lbs,
                self.all_global_ubs,
                self.Cs,
                self.all_thresholds
            ],
            self.all_lAs.values()
        ):
            if item is not None:
                item.reorder(N, indices)

        # If intermediate transfer is used, also reorder the lower and bounds.
        # If not, the static bounds are stored in the class, not need to be reordered.
        # This part costs time and memory.
        if self.interm_transfer and self.unstable_interm_bounds is not None:
            for key, bounds in self.unstable_interm_bounds.items():
                if bounds is None:
                    continue
                bounds[0].reorder(N, indices)   # Lower bounds
                bounds[1].reorder(N, indices)   # Upper bounds
        # Only sort the last layer's bounds.
        next(reversed(self.all_lb_alls.values())).reorder(N, indices)
        next(reversed(self.all_ub_alls.values())).reorder(N, indices)

        # Reorder the alphas, which are stored in a nested dictionary.
        # Note: For alphas, the batch dimension is at index 2.
        for v in self.all_alphas.values():
            for vv in v.values():
                # Reorder along dimension 2 (the batch dimension for alphas)
                vv.reorder(N, indices, reorder_dim=2)

        # === Reorder lists ===
        # Reorder additional lists (e.g., histories, depths, betas) based on the sorted order.
        # Create a selector function to pick elements in the order specified by indices_lst.
        if len(indices_lst) > 1:
            selector = operator.itemgetter(*indices_lst)
        else:
            # If there's only one element, return it as a single-element tuple.
            selector = lambda arr: (arr[indices_lst[0]],)

        # List of attribute names to reorder.
        for name in ('histories', 'depths', 'split_histories', 'all_betas', 'all_intermediate_betas'):
            item = getattr(self, name)
            if item is not None:
                # Apply the selector to reorder the list and update the attribute.
                sorted_item = list(selector(item))
                # When using BFS, the item should be a deque
                # Otherwise, the item should be a list
                if isinstance(item, deque):
                    sorted_item = deque(sorted_item)
                setattr(self, name, sorted_item)

        print(f'Sorting batched domains by lower bounds.')

    def pick_out(self, batch, device='cpu'):
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
        # Upper bounds are not used so far, but still need to pop it to avoid memory leak.
        self.all_global_ubs.pop(batch)
        alphas = defaultdict(dict)
        for k, v_map in self.all_alphas.items():
            alphas[k] = {kk: _to(vv.pop(batch)) for kk, vv in v_map.items()}

        cs = _to(self.Cs.pop(batch))
        thresholds = _to(self.all_thresholds.pop(batch))

        lower_bounds, upper_bounds = self._interm_bounds_wo_transfer(batch, device)

        if self.interm_transfer and self.unstable_interm_bounds is not None:
            interm_bounds = {k: [lower_bounds[k], upper_bounds[k]] for k in lower_bounds if k != self.final_name}
            unstable_bounds = {k: [_to(self.unstable_interm_bounds[k][0].pop(batch)),
                                    _to(self.unstable_interm_bounds[k][1].pop(batch))]
                                    for k in self.unstable_interm_bounds}
            interm_bounds = update_interm_bounds(interm_bounds, unstable_bounds, self.final_name, self.unstable_mask)
            for k, v in interm_bounds.items():
                lower_bounds[k] = v[0]
                upper_bounds[k] = v[1]

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
                    ).flatten(1))
                mask = mask_ if mask is None else torch.logical_or(mask, mask_)
            if mask is None:
                mask = torch.ones_like(lower_bounds[k], dtype=torch.bool).flatten(1)
            new_masks[k] = mask

        # Handle lists.
        if arguments.Config['bab']['tree_traversal'] == 'breadth_first':
            betas_all = [self.all_betas.popleft() for _ in range(batch)]
            intermediate_betas_all = [self.all_intermediate_betas.popleft() for _ in range(batch)]
            history = [self.histories.popleft() for _ in range(batch)]
            split_history = [self.split_histories.popleft() for _ in range(batch)]
            depths = [self.depths.popleft() for _ in range(batch)]
            self.num_domains -= batch
        else:
            betas_all = self.all_betas[self.num_domains - batch: self.num_domains]
            intermediate_betas_all = self.all_intermediate_betas[self.num_domains - batch: self.num_domains]
            history = self.histories[self.num_domains - batch: self.num_domains]
            split_history = self.split_histories[self.num_domains - batch: self.num_domains]
            depths = self.depths[self.num_domains - batch: self.num_domains]

            self.num_domains -= batch

            self.all_betas = self.all_betas[:self.num_domains]
            self.all_intermediate_betas = self.all_intermediate_betas[:self.num_domains]
            self.histories = self.histories[:self.num_domains]
            self.split_histories = self.split_histories[:self.num_domains]
            self.depths = self.depths[:self.num_domains]


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
        decision_threshs = decision_threshs.to(device)
        assert (self.all_x_Ls is None) == (bounds['x_Ls'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_x_Us is None) == (bounds['x_Us'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_input_split_idx is None) == (bounds['input_split_idx'] is None), "Inconsistent input_split_idx during construction and using {type(self)}"
        assert len(self.all_lAs) == len(bounds['lAs']), f"len(self.all_lAs)={len(self.all_lAs)} != len(bounds['lAs'])={len(bounds['lAs'])}"

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
            self.all_x_Ls.append(bounds['x_Ls'][indexer].type(self.all_x_Ls.dtype).to(self.all_x_Ls.device))
        if self.all_x_Us is not None:
            self.all_x_Us.append(bounds['x_Us'][indexer].type(self.all_x_Us.dtype).to(self.all_x_Us.device))
        if self.all_input_split_idx is not None:
            self.all_input_split_idx.append(bounds['input_split_idx'][indexer])
        self.all_global_lbs.append(bounds['lower_bounds'][self.final_name][indexer])
        self.all_global_ubs.append(bounds['upper_bounds'][self.final_name][indexer])
        for k, v in bounds['lower_bounds'].items():
            self.all_lb_alls[k].append(v[indexer])
        for k, v in bounds['upper_bounds'].items():
            self.all_ub_alls[k].append(v[indexer])
        if self.interm_transfer:
            # initial bounds are not used in the first round
            if self.unstable_interm_bounds is None and bounds['unstable_bounds'] is not None:
                self.unstable_interm_bounds = {k: [get_tensor_storage(v[0][indexer].cpu()), get_tensor_storage(v[1][indexer].cpu())] for k, v in bounds['unstable_bounds'].items()}
            # reusing existing TensorStorage
            else:
                for k, v in bounds['unstable_bounds'].items():
                    self.unstable_interm_bounds[k][0].append(v[0][indexer])
                    self.unstable_interm_bounds[k][1].append(v[1][indexer])
        alpha_new = alpha_reuse = False
        for k, v in bounds['alphas'].items():
            if k not in self.all_alphas:
                self.all_alphas[k] = {}
            for kk, vv in v.items():
                if kk not in self.all_alphas[k]:
                    # This is the first time to create these alpha TensorStorage
                    self.all_alphas[k][kk] = get_tensor_storage(vv[:,:,indexer].cpu(), concat_dim=2)
                    alpha_new = True
                else:
                    # Reusing existing TensorStorage
                    self.all_alphas[k][kk].append(vv[:,:,indexer])
                    alpha_reuse = True
        assert not (alpha_new and alpha_reuse)
        self.all_thresholds.append(decision_threshs[indexer])
        self.Cs.append(bounds['c'][indexer])
        self.num_domains = len(self.histories)

    def _interm_bounds_wo_transfer(self, batch, device):
        def _to(x, non_blocking=True):
            return x.to(device=device, non_blocking=non_blocking)

        # Initialize bounds dictionaries
        lower_bounds, upper_bounds = {}, {}
        lower_bounds[self.final_name] = _to(
            self.all_lb_alls[self.final_name].pop(batch))
        upper_bounds[self.final_name] = _to(
            self.all_ub_alls[self.final_name].pop(batch))

        # FIXME This part of code looks unreliable. Need to be clear when
        # this situation might happen.
        # Handle static bounds resizing
        for k in self.static_lb:
            # enlarge the batch size in the static storage
            if batch > self.static_lb[k].shape[0]:
                power = (batch + self.static_lb[k].shape[0] - 1) // self.static_lb[k].shape[0]
                self.static_lb[k] = self.static_lb[k].repeat(
                    power, *([1] * (self.static_lb[k].dim() - 1)))
            if batch > self.static_ub[k].shape[0]:
                power = (batch + self.static_ub[k].shape[0] - 1) // self.static_ub[k].shape[0]
                self.static_ub[k] = self.static_ub[k].repeat(
                    power, *([1] * (self.static_ub[k].dim() - 1)))

        # need to fill in the slots
        if arguments.Config['bab']['tree_traversal'] == 'breadth_first':
            # Get items from the deque without removing them.
            histories = list(islice(self.histories, 0, batch))
        else:
            histories = self.histories[self.num_domains - batch: self.num_domains]

        for i in self.static_lb:
            lower_bounds[i] = self.static_lb[i][:batch].clone()
            upper_bounds[i] = self.static_ub[i][:batch].clone()

            # Lists to collect data for vectorized update
            lb_batch_indices, lb_neuron_indices, lb_values = [], [], []
            ub_batch_indices, ub_neuron_indices, ub_values = [], [], []

            for j, hist in enumerate(histories):
                if i in hist:
                    hist[i] = convert_history_from_list(hist[i])
                    indices, directions, values = hist[i][0], hist[i][1], hist[i][2]
                    assert indices.shape[0] == directions.shape[0] == values.shape[0], \
                        f"Indices, directions, and values must have the same length. " \
                        f"Got {indices.shape[0]}, {directions.shape[0]}, {values.shape[0]}."

                    # Create masks for lower and upper bounds
                    lb_mask = directions > 0
                    ub_mask = ~lb_mask

                    # Append data for lower bound updates
                    lb_batch_indices.extend([j] * lb_mask.sum())
                    lb_neuron_indices.append(indices[lb_mask])
                    lb_values.append(values[lb_mask])

                    # Append data for upper bound updates
                    ub_batch_indices.extend([j] * ub_mask.sum())
                    ub_neuron_indices.append(indices[ub_mask])
                    ub_values.append(values[ub_mask])

            # Perform vectorized update for lower bounds
            if lb_batch_indices:
                neuron_indices = torch.cat(lb_neuron_indices)
                vals = torch.cat(lb_values)
                lower_bounds[i].view(batch, -1)[lb_batch_indices, neuron_indices] = _to(vals)

            # Perform vectorized update for upper bounds
            if ub_batch_indices:
                neuron_indices = torch.cat(ub_neuron_indices)
                vals = torch.cat(ub_values)
                upper_bounds[i].view(batch, -1)[ub_batch_indices, neuron_indices] = _to(vals)

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
        return self.num_domains

    def __getitem__(self, index):
        selected_candidate_domains = self._assemble_domains(
            self.all_global_lbs[index: index + 1],
            self.all_global_ubs[index: index + 1],
            [self.histories[index]],
            [self.split_histories[index]],
            [self.depths[index]],
            self.all_thresholds[index: index + 1])
        return selected_candidate_domains[0]

class ShallowFirstBatchedDomainList(BatchedDomainList):
    """First performs shallow branching, then switches to regular BaB"""

    def __init__(self, ret, c, lAs, global_lbs, global_ubs,
                 alphas=None, history=None, thresholds=None,
                 net=None, branching_input_and_activation=False, x=None):
        super().__init__(ret, c, lAs, global_lbs, global_ubs,
                         alphas, history, thresholds,
                         net, branching_input_and_activation, x)
        self.is_initial_setup = True
        self.use_bfs = True
        # For multi-tree search, we don't just track a single binary tree that defines all explored domains.
        # Instead, we track multiple trees, where each node may have multiple children.
        # This allows us to explore lots of possible splitting sequences in parallel.
        # After the multi-tree search terminates, we want to prune this forest to a single binary tree.
        # That's why we must track the relationship between all generated domains.
        self._multi_trees = {"children":dict()}
        self.mtb_backup = []

    class BaBTree:
        # This is a binary tree, which represents a sequence of splitting decisions.

        def __init__(self, layer_name, node, score):
            self.layer_name = layer_name
            self.node = node
            self.score = score
            self.pos_child = None
            self.neg_child = None

        def __str__(self):
            res = f"Layer: {self.layer_name}, Node: {self.node}, Score: {self.score} \n"
            res += "-1: \n"
            for l in str(self.neg_child).split("\n"):
                res += "  " + l + "\n"
            res += "1: \n"
            for l in str(self.pos_child).split("\n"):
                res += "  " + l + "\n"
            return res

    def _append_to_history(self, history_tuple, index, value):
        """
        Append a value to a history tuple element, supporting both list and Tensor.
        Args:
            history_tuple: tuple of (list or Tensor) elements
            index: which element to append to (0-4)
            value: value to append
        Returns:
            new tuple with appended value
        """
        result = list(history_tuple)
        if isinstance(result[index], torch.Tensor):
            # For Tensor, use torch.cat to concatenate
            new_value = torch.tensor([value], dtype=result[index].dtype, device=result[index].device)
            result[index] = torch.cat([result[index], new_value])
        else:
            # For list, use append
            result[index].append(value)
        return tuple(result)

    def restore_best_domains(self, bounds, d):
        best_tree = self._generate_tree(self._multi_trees)
        self._restore_domain_from_tree(best_tree, bounds, d, depth=0)

    def _restore_domain_from_tree(self, tree, bounds, d, depth):
        """
        Extract the best regular (= binary) tree from the set of branching decisions explored during multi-tree search.

        During multi-tree search, we explore a lot of different splitting sequences.
        These (not the complete domain objects, just the information which node was split)
        were stored in `._multi_tree`. `tree` was selected from this, and is a regular
        binary tree, describing a branching sequence we might encounter during regular Bab.
        We now need to reconstruct the domains that this tree defines.
        Some domains might be UNSAT, but they'll be pruned at a later step.

        Initially, `bounds` and `d` should define the root node. They'll be copied and modified
        to define all nodes in the tree.

        Args:
            tree: dict, encoding the multi-tree
            bounds: dict, as returned by `update_bounds()`
            d: dict, as returned by `pick_out()`
            depth: int, the current depth
        """

        assert len(d["history"]) == 1

        # Create deep copies of the current domain 'd' and its bounds so that modifications
        # for the negative branch do not affect the original domain and bounds.
        d_neg = copy.deepcopy(d)
        bounds_neg = copy.deepcopy(bounds)
        # Retrieve the current history from the negative domain copy.
        # The history is structured as a tuple whose first element is a dictionary mapping
        # each tree layer to a tuple of five lists:
        #   (nodes, signs, thresholds, scores, depths)
        d_neg_hist = d_neg['history'][0]

        # --- Update the history to reflect a negative decision at the current tree layer ---

        # 1. Record the current node index where the decision is made.
        d_neg_hist[tree.layer_name] = self._append_to_history(d_neg_hist[tree.layer_name], 0, tree.node)

        # 2. Record the decision sign.
        #    Here, -1.0 is used to indicate a negative branch decision.
        d_neg_hist[tree.layer_name] = self._append_to_history(d_neg_hist[tree.layer_name], 1, -1.0)

        # 3. Record a threshold value associated with this bias (usually not used).
        #    A value of 0 is appended, possibly representing a default or specific threshold action.
        d_neg_hist[tree.layer_name] = self._append_to_history(d_neg_hist[tree.layer_name], 2, 0)

        # 4. Record the score for the current decision.
        #    Ensure that the score exists before appending it.
        assert tree.score[0] is not None
        d_neg_hist[tree.layer_name] = self._append_to_history(d_neg_hist[tree.layer_name], 3, tree.score[0])

        # 5. Record the depth in the decision tree where this negative decision occurs.
        d_neg_hist[tree.layer_name] = self._append_to_history(d_neg_hist[tree.layer_name], 4, depth)

        # Update the domain copy's history with the modified history.
        d_neg['history'] = (d_neg_hist,)

        if tree.neg_child is None:
            # Base case: If there is no further negative child node,
            # convert the history lists at the current layer into torch tensors.
            # This conversion is typically needed for subsequent tensor-based computations.
            d_neg_hist[tree.layer_name] = (
                torch.tensor(d_neg_hist[tree.layer_name][0], dtype=torch.long),
                torch.tensor(d_neg_hist[tree.layer_name][1]),
                torch.tensor(d_neg_hist[tree.layer_name][2]),
                torch.tensor(d_neg_hist[tree.layer_name][3]),
                torch.tensor(d_neg_hist[tree.layer_name][4])
            )
            # Update the history in the domain copy with the tensor-converted data.
            d_neg['history'] = (d_neg_hist,)
            print("Restore", d_neg_hist)
            # Add the restored negative domain using the same method as its superclass.
            super().add(bounds_neg, d_neg, False)
        else:
            # Recursive case: If a negative child exists, continue restoring the domain recursively.
            # Increment the depth to reflect the deeper level in the decision tree.
            self._restore_domain_from_tree(tree.neg_child, bounds_neg, d_neg, depth+1)

        # --- Update the history to reflect a positive decision at the current tree layer ---
        # Similar to above, but for the positive branch.
        d_pos = copy.deepcopy(d)
        bounds_pos = copy.deepcopy(bounds)
        d_pos_hist = d_pos['history'][0]
        d_pos_hist[tree.layer_name] = self._append_to_history(d_pos_hist[tree.layer_name], 0, tree.node)
        d_pos_hist[tree.layer_name] = self._append_to_history(d_pos_hist[tree.layer_name], 1, 1.0)
        d_pos_hist[tree.layer_name] = self._append_to_history(d_pos_hist[tree.layer_name], 2, 0)
        assert tree.score[1] is not None
        d_pos_hist[tree.layer_name] = self._append_to_history(d_pos_hist[tree.layer_name], 3, tree.score[1])
        d_pos_hist[tree.layer_name] = self._append_to_history(d_pos_hist[tree.layer_name], 4, depth)
        d_pos['history'] = (d_pos_hist,)

        if tree.pos_child is None:
            d_pos_hist[tree.layer_name] = (
                torch.tensor(d_pos_hist[tree.layer_name][0], dtype=torch.long),
                torch.tensor(d_pos_hist[tree.layer_name][1]),
                torch.tensor(d_pos_hist[tree.layer_name][2]),
                torch.tensor(d_pos_hist[tree.layer_name][3]),
                torch.tensor(d_pos_hist[tree.layer_name][4])
            )
            d_pos['history'] = (d_pos_hist,)
            print("Restore", d_pos_hist)
            super().add(bounds_pos, d_pos, False)
        else:
            self._restore_domain_from_tree(tree.pos_child, bounds_pos, d_pos, depth+1)

    def _generate_tree(self, current):
        # Given the multi-tree in `current`, we select the best binary tree it contains.
        # Here "best" is the tree that was explored most (= where the bounds are the tightest)
        root_node, score = self._get_best_next_node(current)
        best_tree = ShallowFirstBatchedDomainList.BaBTree(root_node[0], root_node[1], score)
        assert (best_tree.layer_name, best_tree.node) == root_node
        if len(current["children"][(best_tree.layer_name, best_tree.node)][-1.0]['children']) > 0:
            best_tree.neg_child = self._generate_tree(current["children"][root_node][-1.0])
        if len(current["children"][(best_tree.layer_name, best_tree.node)][1.0]['children']) > 0:
            best_tree.pos_child = self._generate_tree(current["children"][root_node][1.0])
        return best_tree


    def _get_best_next_node(self, current):
        nodes = current["children"].keys()
        best_node = None
        best_layer_count = -1
        for layer_name in nodes:
            if current["children"][layer_name]["num_children"] > best_layer_count:
                best_layer_count = current["children"][layer_name]["num_children"]
                best_node = layer_name
        assert best_node is not None
        return best_node, current["children"][best_node]["score"]

    def add(self, bounds, d, check_infeasibility):
        """
        Add the top n UNSAT domains to the list of domains that still need to be explored.

        This is used during multi-tree search. We don't store all domains, but only the top N.
        These are then used in the next iteration to perform k splits each.
        In addition to selecting the top n nodes, we also keep track of all explored domains.
        """
        if not self.use_bfs:
            return super().add(bounds, d, check_infeasibility)
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
        verified_entries = (~indexer).nonzero().view(-1)

        duplicated_branches = set()
        if len(d['history']) > 1:
            # This block is about keeping track of all domains in this multi-tree. We need to make sure to add the correct
            # splits to self._multi_trees
            # Also, we check if a domain is known to be UNSAT due to some cuts infered from other domains in this batch
            # If so, we will not split this domain further, even if itself was not UNSAT yet
            for branch_id, branch in enumerate(d['history']):
                branched_nodes = dict()
                branched_nodes_as_set = set()
                for layer_name, (nodes, signs, threshs, scores, depths) in branch.items():
                    for node, sign, thresh, score, depth in zip(nodes, signs, threshs, scores, depths):
                        assert thresh == 0
                        # If these tensores are stored on the GPU, this will be slow.
                        # Therefore, we must ensure they're stored on the CPU instead.
                        assert "cuda" not in node.device.type
                        assert "cuda" not in sign.device.type
                        assert "cuda" not in depth.device.type
                        assert "cuda" not in score.device.type
                        node = node.item()
                        sign = sign.item()
                        depth = depth.item()
                        score = score.item()

                        assert depth not in branched_nodes
                        branched_nodes[depth] = {
                            "layer_name": layer_name,
                            "node": node,
                            "sign": sign,
                            "score": score,
                        }
                        branched_nodes_as_set.add((layer_name, node, sign))
                assert len(branched_nodes_as_set) == len(branched_nodes)
                sorted_domain = sorted(list(branched_nodes_as_set))
                arelu_decision = [[self.net.biccos.key_mapping[layer_name], node_id] for layer_name, node_id, _ in sorted_domain]
                arelu_coeffs = [coeff for _, _, coeff in sorted_domain]
                potential_cut = self.net.biccos.generate_cut(
                    relu_activation_decision=arelu_decision,
                    relu_activation_coeffs=arelu_coeffs,
                    b=999,  # bias does not matter, because this cut will not be added. It's only used to check for duplicates
                )
                drop_domain = False
                for potential_parent in self.net.biccos.biccos_cuts:
                    if self.net.biccos.is_cut_a_parent(potential_cut, potential_parent):
                        drop_domain = True
                        break
                if drop_domain:
                    if indexer[branch_id]:
                        duplicated_branches.add(branch_id)
                        indexer[branch_id] = False
                    else:
                        # This domain is duplicated, but it's also UNSAT. It won't be explored further, anyway.
                        # If we were to add it to the list of duplications, it would be counted twice in add_best_lower_k_domains:
                        # Once as a verified domain, and once as a domain that should be skipped. That will mess up the indices there.
                        pass
                num_children = 1
                if branch_id in verified_entries:
                    # Verified nodes will not have children. However, they're the best nodes to
                    # have in your tree. Therefore, we pretend that all their potential children
                    # would have been explored. This will prioritize them when we select a single binary tree for the
                    # regular BaB phase.
                    depth = len(branched_nodes)
                    target_batch_size = arguments.Config["bab"]["cut"]["biccos"]["multi_tree_branching"]["target_batch_size"]
                    keep_n_best_domains = arguments.Config["bab"]["cut"]["biccos"]["multi_tree_branching"]["keep_n_best_domains"]
                    if keep_n_best_domains <= 1:
                        raise ValueError("keep_n_best_domains must be greater than 1 for math.log calculation.")
                    if target_batch_size < keep_n_best_domains:
                        raise ValueError("target_batch_size must be greater than or equal to keep_n_best_domains for math.log calculation.")
                    splits_per_depth = math.floor(math.log(
                        target_batch_size,
                        keep_n_best_domains
                    ))
                    max_depth = arguments.Config["bab"]["cut"]["biccos"]["multi_tree_branching"]["iterations"]
                    num_children = splits_per_depth ** (max_depth - depth)

                current = self._multi_trees
                # Get sorted depth keys instead of assuming consecutive indices starting from 0
                sorted_depths = sorted(branched_nodes.keys())
                # Iterate over all but the last depth
                for depth in sorted_depths[:-1]:
                    next_node = branched_nodes[depth]
                    assert "children" in current.keys()
                    assert (next_node["layer_name"], next_node["node"]) in current["children"]
                    pos_neg = current["children"][(next_node["layer_name"], next_node["node"])]
                    pos_neg["num_children"] += num_children
                    assert next_node["sign"] in pos_neg, (next_node, pos_neg)
                    current = pos_neg[next_node["sign"]]
                next_node = branched_nodes[sorted_depths[-1]]
                next_key = (next_node["layer_name"], next_node["node"])
                if next_key not in current["children"].keys():
                    current["children"][next_key] = {
                        "num_children": num_children,
                        "score": [None, None],
                        -1.0: {
                            "children": dict(),
                        },
                        1.0: {
                            "children": dict(),
                        }
                    }
                else:
                    current["children"][next_key]["num_children"] += num_children
                assert next_node["sign"] in [-1, 1]
                assert current["children"][next_key]["score"][0 if next_node["sign"] == -1 else 1] is None
                current["children"][next_key]["score"][0 if next_node["sign"] == -1 else 1] = next_node["score"]

        if not torch.any(indexer):
            return
        
        if len(indexer) > 1:
            self.mtb_backup.append({
                'bounds': copy.deepcopy(bounds),
                'd': copy.deepcopy(d),
                'check_infeasibility': check_infeasibility,
                'duplicated_branches': copy.deepcopy(duplicated_branches),
                'verified_entries': copy.deepcopy(verified_entries),
                'skip': 0,
            })

        assert bounds['lower_bounds'][self.final_name].ndim == 2
        assert bounds['lower_bounds'][self.final_name].size(1) == 1
        self.add_best_k_lower_bounds(bounds, d, check_infeasibility, duplicated_branches, verified_entries)

    class EmptyKLower(Exception):
        """Used to indicate that the current set of domains has no unknown domains left (after skipping some of them)"""
        pass

    def add_best_k_lower_bounds(self, bounds, d, check_infeasibility, duplicated_branches, verified_entries, skip=0):
        lower_bounds = copy.deepcopy(bounds['lower_bounds'][self.final_name])
        for duplicated_branch in duplicated_branches:
            # We set the lower bound to a positive value, so it's grouped with all the verified domains.
            lower_bounds[duplicated_branch] = 42

        # Sort the lower bounds from best to worst and obtain the indices.
        indices_best_to_worst = lower_bounds.squeeze(1).sort(descending=True, dim=0)[1]

        # Calculate the number of domains that should be skipped,
        # which includes verified entries and duplicated branches.
        skip_first_n_domains = len(verified_entries) + len(duplicated_branches)

        # Retrieve the configuration parameter for how many of the best domains to keep.
        keep_best_n_domains = arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['keep_n_best_domains']

        # Select the best k lower bounds after skipping the appropriate number of domains.
        best_k_lower_bounds = indices_best_to_worst[
            skip_first_n_domains + skip * keep_best_n_domains : skip_first_n_domains + (skip + 1) * keep_best_n_domains
        ]

        # If no domains are available after filtering, raise an error with a human-friendly message.
        if best_k_lower_bounds.size(0) == 0:
            raise ShallowFirstBatchedDomainList.EmptyKLower(
                "No valid domains remain after excluding verified and duplicated branches. "
                "This indicates that all candidate domains have been processed or filtered out. "
                "Please review your configuration for 'keep_n_best_domains' and the criteria for verified or duplicated domains."
            )

        # dicts
        for key in ['lower_bounds', 'upper_bounds', 'lAs']:
            for layer_name, val in bounds[key].items():
                bounds[key][layer_name] = bounds[key][layer_name][best_k_lower_bounds]
        for key in ['lower_bounds']:
            for layer_name, val in d[key].items():
                d[key][layer_name] = d[key][layer_name][best_k_lower_bounds]

        # doubly nested dicts
        for key in ['alphas']:
            for layer_name, val in bounds[key].items():
                for layer_name2, val2 in bounds[key][layer_name].items():
                    bounds[key][layer_name][layer_name2] = bounds[key][layer_name][layer_name2][:, :, best_k_lower_bounds]

        # Lists
        batch_indexer_lst = best_k_lower_bounds.tolist()
        selector = (
            operator.itemgetter(*batch_indexer_lst)
            if len(batch_indexer_lst) > 1
            else lambda _arr: (_arr[batch_indexer_lst[0]], ))
        for key in ['betas', 'split_history', 'intermediate_betas']:
            bounds[key] = selector(bounds[key])
        d['history'] = selector(d['history'])
        d['depths'] = selector(d['depths'])

        # Tensors
        for key in ['c', 'x_Ls', 'x_Us', 'input_split_idx']:
            if bounds[key] is None:
                continue
            bounds[key] = bounds[key][best_k_lower_bounds]
        d['thresholds'] = d['thresholds'][best_k_lower_bounds]

        return super().add(bounds, d, check_infeasibility)

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
