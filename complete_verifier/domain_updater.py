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
"""Update domains after applying a split."""

from collections import defaultdict

import torch

from utils import fast_hist_copy
import arguments


def repeat(x, num_copy, unsqueeze=False):
    """Repeat a tensor by the first dimension."""
    if x is None:
        return None
    if isinstance(x, list):
        return x * num_copy
    if unsqueeze:
        return x.unsqueeze(0).repeat(num_copy, *[1]*x.ndim)
    else:
        return x.repeat(num_copy, *[1]*(x.ndim - 1))


class DomainUpdater:
    def __init__(self, root, final_name, split_nodes):
        self.root = root
        self.final_name = final_name
        self.split_nodes = split_nodes

        self.device = 'cpu'
        self.node_names = []
        cut_args = arguments.Config['bab']['cut']
        self.cut_usage = cut_args['enabled']
        self.biccos_usage = cut_args['enabled'] and cut_args['biccos']['enabled']
        self.multi_tree_searching = (self.biccos_usage and
                            cut_args['biccos']['multi_tree_branching']['enabled'])

    @staticmethod
    def get_num_domain_and_split(d, split, final_name):
        num_domain = d['lower_bounds'][final_name].shape[0]
        num_split = len(split['decision']) // num_domain
        return num_domain, num_split

    def as_tensor(self, t):
        return torch.as_tensor(t).to(device=self.device, non_blocking=True)

    def empty_dict(self):
        return {k: [] for k in self.node_names}

    def _convert_history_from_list(self, history):
        """Convert the history variables into tensors if they are lists.

        It is because some legacy code creates history as lists.
        """

        if isinstance(history[0], torch.Tensor):
            return history

        return (torch.tensor(history[0], dtype=torch.long),
                torch.tensor(history[1]),
                torch.tensor(history[2]),
                torch.tensor(history[3]),
                torch.tensor(history[4]))

    def set_branched_bounds(self, d, split, mode='depth'):
        """
        d: Domains
        split: Split decisions
        mode ('depth' or 'breadth'): For multiple candidate decisions, whether to
        stack them in the depth direction (to apply all the decisions) or
        breadth direction (to try different decisions separately).
        """
        self.num_domain, self.num_split = self.get_num_domain_and_split(
            d, split, self.final_name)
        if split.get('points', None) is not None and split['points'].ndim == 2:
            self.multi_branch = True
            self.num_branches = split['points'].shape[1] + 1
        else:
            self.multi_branch = False
            self.num_branches = 2
        if mode == 'depth':
            # TODO some branching points may be invalid and thus the actual
            # number of branches may be fewer (to allow some neurons to have
            # fewer branching points).
            self.num_copy = self.num_branches**self.num_split # TODO support multiple branches
        else:
            assert mode == 'breadth', f"Unsupported splitting mode {mode}"
            self.num_copy = self.num_branches * self.num_split

        self.device = d['lower_bounds'][self.final_name].device
        self.node_names = [k for k in d['lower_bounds'].keys() if k != self.final_name]

        d['lower_bounds'] = {
            k: repeat(v, self.num_copy, unsqueeze=True)
            for k, v in d['lower_bounds'].items()}
        d['upper_bounds'] = {
            k: repeat(v, self.num_copy, unsqueeze=True)
            for k, v in d['upper_bounds'].items()}
        self.history = d.get('history', None)
        self.new_history = []
        if self.history is not None:
            for _ in range(self.num_copy):
                for j in range(self.num_domain):
                    self.new_history.append(fast_hist_copy(self.history[j]))
        else:
            self.new_history = [None] * (self.num_copy * self.num_domain)
        self.upd_hist_l, self.upd_hist_u = self.empty_dict(), self.empty_dict()
        self.upd_domain_l, self.upd_domain_u = self.empty_dict(), self.empty_dict()
        self.upd_idx_l, self.upd_idx_u = self.empty_dict(), self.empty_dict()
        self.upd_val_l, self.upd_val_u = self.empty_dict(), self.empty_dict()

        self._set_history_and_bounds(d, split, mode)

        d['lower_bounds'] = {
            k: v.view(-1, *v.shape[2:]) for k, v in d['lower_bounds'].items()}
        d['upper_bounds'] = {
            k: v.view(-1, *v.shape[2:]) for k, v in d['upper_bounds'].items()}
        d['history'] = self.new_history

        if 'depths' in d:
            if mode == 'depth':
                d['depths'] = [depth + self.num_split for depth in d['depths']]
            else:
                d['depths'] = [depth + 1 for depth in d['depths']]
            d['depths'] = d['depths'] * self.num_copy
        if 'alphas' in d:
            new_alphas = defaultdict(dict)
            for k, v in d['alphas'].items():
                new_alphas[k] = {kk: torch.cat([vv] * self.num_copy, dim=2)
                    for kk, vv in v.items()}
            d['alphas'] = new_alphas
        if 'lAs' in d:
            d['lAs'] = {
                k: repeat(v, self.num_copy)
                for k, v in d['lAs'].items()
            }
        for k in ['split_history', 'cs', 'betas', 'intermediate_betas',
                'thresholds', 'x_Ls', 'x_Us']:
            if k in d:
                d[k] = repeat(d[k], self.num_copy)
        for k in split:
            if isinstance(split[k], list):
                split[k] = split[k][-self.num_domain:] * self.num_copy
            elif isinstance(split[k], torch.Tensor):
                split[k] = split[k][-self.num_domain:].repeat(
                    self.num_copy, *[1]*(split[k].ndim - 1))

    def _set_history_and_bounds(self, d, split, mode='depth'):
        if self.history is not None:
            history_new_len = [{} for _ in range(len(self.new_history))]
            for i in range(self.num_domain):
                cycle = 1
                for cur_split in range(self.num_split):
                    # FIXME Inconsistent node index for new_history (split_indices)
                    # and elsewhere.
                    node, idx = split['decision'][cur_split*self.num_domain+i]
                    node = self.split_nodes[node].name
                    # # TODO Allow some branching points to be invalid

                    if mode == 'depth':
                        j_iter = range(self.num_copy)
                    else:
                        j_iter = range(cur_split*self.num_branches,
                                       (cur_split+1)*self.num_branches)

                    branch_idx = 0
                    count = 0
                    for j in j_iter:
                        history_idx = (-self.num_copy * self.num_domain
                                        + j * self.num_domain + i)
                        if branch_idx + 1 < self.num_branches:
                            history_new_len[history_idx][node] = (
                                history_new_len[history_idx].get(node, 0) + 1)
                        if branch_idx > 0:
                            history_new_len[history_idx][node] = (
                                history_new_len[history_idx].get(node, 0) + 1)
                        if mode == 'depth':
                            count += 1
                            if count == cycle:
                                branch_idx = (branch_idx + 1) % self.num_branches
                                count = 0
                        else:
                            branch_idx += 1
                    if mode == 'depth':
                        cycle *= self.num_branches

            for i, lengths in enumerate(history_new_len):
                for node, l in lengths.items():
                    if len(self.new_history[i][node][0]) > 0:
                        self.new_history[i][node] = self._convert_history_from_list(
                            self.new_history[i][node])
                        shape_base = self.new_history[i][node][0].numel()
                    else:
                        shape_base = 0
                    loc = torch.empty(shape_base + l, dtype=torch.long)
                    sign = torch.empty(shape_base + l)
                    bias = torch.empty(shape_base + l)
                    score = torch.empty(shape_base + l) if self.biccos_usage else None
                    depth = torch.empty(shape_base + l) if self.multi_tree_searching else None
                    if len(self.new_history[i][node][0]) > 0:
                        loc[:shape_base] = self.new_history[i][node][0]
                        sign[:shape_base] = self.new_history[i][node][1]
                        if self.new_history[i][node][2] is not None and self.new_history[i][node][2].numel() > 0:
                            bias[:shape_base] = self.new_history[i][node][2]
                        if self.new_history[i][node][3] is not None and self.new_history[i][node][3].numel() > 0:
                            score[:shape_base] = self.new_history[i][node][3]
                        if self.new_history[i][node][4] is not None and self.new_history[i][node][4].numel() > 0:
                            depth[:shape_base] = self.new_history[i][node][4]
                    max_depth = -1

                    # x[4] is the depth of the split
                    # this block is to find the maximum depth of the split
                    # in the current domain
                    # ONLY used in multi-tree-shearching
                    if self.multi_tree_searching:
                        for x in self.new_history[i].values():
                            if isinstance(x[4], list):
                                if len(x[4]) == 0:
                                    continue
                                max_depth = max(max_depth, max(x[4]))
                            else:
                                assert x[4].ndim == 1
                                if x[4].numel() > 0:
                                    max_depth = max(max_depth, torch.max(x[4]))
                                else:
                                    # Handle the case where x[4] is empty
                                    # For example, it might just skip this step or assign a default value
                                    max_depth = max(max_depth, 0)
                        depth[shape_base:] = max_depth + 1

                    self.new_history[i][node] = (loc, sign, bias, score, depth)

        for i in range(self.num_domain):
            cycle = 1
            for cur_split in range(self.num_split):
                # FIXME Inconsistent node index for new_history (split_indices)
                # and elsewhere.
                node, idx = split['decision'][cur_split*self.num_domain+i]
                node = self.split_nodes[node].name
                if split.get('points', None) is not None:
                    points = split['points'][cur_split*self.num_domain+i]
                else:
                    points = 0.

                if mode == 'depth':
                    j_iter = range(self.num_copy)
                else:
                    j_iter = range(cur_split*self.num_branches,
                                   (cur_split+1)*self.num_branches)

                branch_idx = 0
                count = 0
                for j in j_iter:
                    history_idx = (-self.num_copy * self.num_domain
                                    + j * self.num_domain + i)
                    if branch_idx + 1 < self.num_branches:
                        val = points[branch_idx] if self.multi_branch else points
                        if self.history is not None:
                            new_item_idx = -history_new_len[history_idx][node]
                            self.new_history[history_idx][node][0][new_item_idx] = idx
                            self.new_history[history_idx][node][1][new_item_idx] = -1
                            self.new_history[history_idx][node][2][new_item_idx] = val
                            history_new_len[history_idx][node] -= 1
                        self.upd_hist_u[node].append(j)
                        self.upd_domain_u[node].append(i)
                        self.upd_idx_u[node].append(idx)
                        self.upd_val_u[node].append(val)
                    if branch_idx > 0:
                        val = points[branch_idx - 1] if self.multi_branch else points
                        if self.history is not None:
                            new_item_idx = -history_new_len[history_idx][node]
                            self.new_history[history_idx][node][0][new_item_idx] = idx
                            self.new_history[history_idx][node][1][new_item_idx] = 1
                            self.new_history[history_idx][node][2][new_item_idx] = val
                            history_new_len[history_idx][node] -= 1
                        self.upd_hist_l[node].append(j)
                        self.upd_domain_l[node].append(i)
                        self.upd_idx_l[node].append(idx)
                        self.upd_val_l[node].append(val)
                    if mode == 'depth':
                        count += 1
                        if count == cycle:
                            branch_idx = (branch_idx + 1) % self.num_branches
                            count = 0
                    else:
                        branch_idx += 1
                if mode == 'depth':
                    cycle *= self.num_branches

        upd = [self.upd_hist_l, self.upd_hist_u, self.upd_domain_l,
               self.upd_domain_u, self.upd_idx_l, self.upd_idx_u,
               self.upd_val_l, self.upd_val_u]
        for upd_list in upd:
            for k in upd_list:
                upd_list[k] = self.as_tensor(upd_list[k])
        for k in self.node_names:
            if len(self.upd_hist_u[k]):
                d['upper_bounds'][k].view(self.num_copy, self.num_domain, -1)[
                    self.upd_hist_u[k], self.upd_domain_u[k], self.upd_idx_u[k]
                ] = self.upd_val_u[k]
            if len(self.upd_hist_l[k]):
                d['lower_bounds'][k].view(self.num_copy, self.num_domain, -1)[
                    self.upd_hist_l[k], self.upd_domain_l[k], self.upd_idx_l[k]
                ] = self.upd_val_l[k]


class DomainUpdaterSimple(DomainUpdater):

    def _set_history_and_bounds(self, d, split, *args):
        assert self.num_copy == 2

        upd_domain, upd_idx= self.empty_dict(), self.empty_dict()
        upd = [upd_domain, upd_idx]

        branching_points = split.get('points', None) is not None

        if branching_points:
            upd_val = self.empty_dict()
            upd.append(upd_val)

        for i in range(self.num_domain):
            # FIXME Inconsistent node index for new_history (split_indices)
            # and elsewhere.
            node, idx = split['decision'][i]
            node = self.split_nodes[node].name
            points = split['points'][i] if branching_points else None
            for j in range(2):
                history_idx = (-self.num_copy * self.num_domain
                               + j * self.num_domain + i)
                upd_domain[node].append(i)
                upd_idx[node].append(idx)
                if branching_points:
                    upd_val[node].append(points)
                if self.history is not None:
                    self._append_history(
                        history_idx, node, idx, 1 - j * 2, points)

        for upd_list in upd:
            for k in upd_list:
                upd_list[k] = self.as_tensor(upd_list[k])
        for k in self.node_names:
            if len(upd_domain[k]):
                if branching_points:
                    d['lower_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = upd_val[k]
                    d['upper_bounds'][k][1].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = upd_val[k]
                else:
                    d['lower_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = 0.
                    d['upper_bounds'][k][1].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = 0.

    def _append_history(self, idx, node, this_loc, this_sign, this_points):
        # new_history stores the information about all performed splits in this domain.
        # loc/sign/bias define which node was split which way.
        # score is the score computed for BICCOS, which may be used to tighten inferred constraints.
        # depth encodes in which order the nodes were split.
        if self.new_history[idx] is None:
            return
        if this_points is None:
            this_points = 0.
        max_depth = -1
        if self.multi_tree_searching:
            for x in self.new_history[idx].values():
                if isinstance(x[4], list):
                    if len(x[4]) == 0:
                        continue
                    max_depth = max(max_depth, max(x[4]))
                else:
                    assert x[4].ndim == 1
                    if x[4].numel() > 0:
                        max_depth = max(max_depth, torch.max(x[4]))
                    else:
                        max_depth = max(max_depth, 0)

        if len(self.new_history[idx][node][0]) == 0:
            loc = torch.tensor([this_loc], dtype=torch.long)
            sign = torch.tensor([this_sign])
            if this_points is not None:
                bias = torch.tensor([this_points])
            else:
                bias = None
            score = torch.tensor([0.]) if self.biccos_usage else None
            depth = torch.tensor([max_depth + 1]) if self.multi_tree_searching else None
        else:
            self.new_history[idx][node] = self._convert_history_from_list(
                self.new_history[idx][node])
            shape = self.new_history[idx][node][0].numel()
            loc = torch.empty(shape + 1, dtype=torch.long)
            sign = torch.empty(shape + 1)
            bias = torch.empty(shape + 1)
            score = torch.empty(shape + 1) if self.biccos_usage else None
            depth = torch.empty(shape + 1) if self.multi_tree_searching else None
            loc[:shape] = self.new_history[idx][node][0]
            sign[:shape] = self.new_history[idx][node][1]

            if self.new_history[idx][node][2] is not None and self.new_history[idx][node][2].numel() > 0:
                bias[:shape] = self.new_history[idx][node][2]

            if score is not None and self.new_history[idx][node][3] is not None:
                if self.new_history[idx][node][3].numel() > 0:
                    score[:shape] = self.new_history[idx][node][3]

            # Ensure the source tensor has enough elements to be assigned to depth
            if depth is not None and self.new_history[idx][node][4] is not None:
                if self.new_history[idx][node][4].numel() >= shape:
                    depth[:shape] = self.new_history[idx][node][4][:shape]
                else:
                    depth[:self.new_history[idx][node][4].numel()] = self.new_history[idx][node][4]

            loc[shape] = this_loc
            sign[shape] = this_sign
            if this_points is not None:
                bias = torch.zeros(shape + 1)
                if self.new_history[idx][node][2].numel() > 0:
                    bias[:shape] = self.new_history[idx][node][2]
                bias[shape] = this_points
            else:
                bias = None
            if self.biccos_usage:
                score[shape] = 0.
            if self.multi_tree_searching:
                depth[shape] = max_depth + 1
        self.new_history[idx][node] = (loc, sign, bias, score, depth)
