"""Update domains after applying a split."""

from collections import defaultdict

import torch

from utils import fast_hist_copy


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


def retain_valid_domains(data, valid_domains, is_alpha=False):
    if data is None:
        return None
    elif isinstance(data, list):
        data = [item for i, item in enumerate(data) if valid_domains[i]]
    elif isinstance(data, torch.Tensor):
        if is_alpha:
            data = data[:, :, valid_domains]
        else:
            try:
                data = data[valid_domains]
            except:
                import pdb; pdb.set_trace()
    elif isinstance(data, dict):
        data = {k: retain_valid_domains(v, valid_domains, is_alpha=is_alpha)
                for k, v in data.items()}
    else:
        raise NotImplementedError(type(data))
    return data


class DomainUpdater:
    def __init__(self, root, final_name, split_nodes):
        self.root = root
        self.final_name = final_name
        self.split_nodes = split_nodes

        self.device = 'cpu'
        self.node_names = []

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
                torch.tensor(history[2]))

    def _append_history(self, idx, node, item):
        if self.new_history[idx] is None:
            return
        if len(self.new_history[idx][node][0]) == 0:
            loc = torch.tensor([item[0]], dtype=torch.long)
            sign = torch.tensor([item[1]])
            bias = torch.tensor([item[2]])
        else:
            self.new_history[idx][node] = self._convert_history_from_list(
                self.new_history[idx][node])
            shape = self.new_history[idx][node][0].numel() + 1
            loc = torch.empty(shape, dtype=torch.long)
            sign = torch.empty(shape)
            bias = torch.zeros(shape)
            loc[:shape-1] = self.new_history[idx][node][0]
            sign[:shape-1] = self.new_history[idx][node][1]
            if self.new_history[idx][node][2].numel() > 0:
                bias[:shape-1] = self.new_history[idx][node][2]
            loc[shape-1] = item[0]
            sign[shape-1] = item[1]
            bias[shape-1] = item[2]
        self.new_history[idx][node] = (loc, sign, bias)

    def set_branched_bounds(self, d, split, mode='depth'):
        """
        d: Domains
        split: Split decisions
        mode ('depth' or 'breath'): For multiple candidate decisions, whether to
        stack them in the depth direction (to apply all the decisions) or
        breath direction (to try different decisions separately).
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
            self.num_copy = self.num_branches * self.num_split

        self.device = d['lower_bounds'][self.final_name].device
        self.node_names = [k for k in d['lower_bounds'].keys() if k != self.final_name]

        self.valid_domains = torch.ones(
            self.num_copy, self.num_domain,
            device=self.device, dtype=torch.bool)

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

        self.valid_domains = self.valid_domains.view(-1)

        d['lower_bounds'] = retain_valid_domains(
            {k: v.view(-1, *v.shape[2:])
             for k, v in d['lower_bounds'].items()}, self.valid_domains)
        d['upper_bounds'] = retain_valid_domains(
            {k: v.view(-1, *v.shape[2:])
             for k, v in d['upper_bounds'].items()}, self.valid_domains)
        d['history'] = retain_valid_domains(self.new_history, self.valid_domains)

        if 'depths' in d:
            if mode == 'depth':
                d['depths'] = [depth + self.num_split for depth in d['depths']]
            else:
                d['depths'] = [depth + 1 for depth in d['depths']]
            d['depths'] = retain_valid_domains(
                d['depths'] * self.num_copy, self.valid_domains)
        if 'alphas' in d:
            new_alphas = defaultdict(dict)
            for k, v in d['alphas'].items():
                new_alphas[k] = {kk: torch.cat([vv] * self.num_copy, dim=2)
                    for kk, vv in v.items()}
            d['alphas'] = retain_valid_domains(
                new_alphas, self.valid_domains, is_alpha=True)

        for k in ['split_history', 'cs', 'betas', 'intermediate_betas',
                'thresholds', 'x_Ls', 'x_Us']:
            if k in d:
                d[k] = repeat(d[k], self.num_copy)
                d[k] = retain_valid_domains(d[k], self.valid_domains)
        for k in split:
            if isinstance(split[k], list):
                split[k] = split[k][-self.num_domain:] * self.num_copy
                split[k] = retain_valid_domains(split[k], self.valid_domains)
            elif isinstance(split[k], torch.Tensor):
                split[k] = split[k][-self.num_domain:].repeat(
                    self.num_copy, *[1]*(split[k].ndim - 1))
                split[k] = retain_valid_domains(split[k], self.valid_domains)

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
                    bias = torch.zeros(shape_base + l)
                    if len(self.new_history[i][node][0]) > 0:
                        loc[:shape_base] = self.new_history[i][node][0]
                        sign[:shape_base] = self.new_history[i][node][1]
                        if self.new_history[i][node][2].numel() > 0:
                            bias[:shape_base] = self.new_history[i][node][2]
                    self.new_history[i][node] = (loc, sign, bias)

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
                if split.get('points_mask', None) is not None:
                    points_mask = split['points_mask'][cur_split*self.num_domain+i]
                else:
                    points_mask = None
                # TODO Allow some branching points to be invalid

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
                    if (points_mask is not None
                            and branch_idx + 1 < self.num_branches
                            and not points_mask[branch_idx]):
                        self.valid_domains[j, i] = False
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

        upd_domain, upd_idx, upd_val = self.empty_dict(), self.empty_dict(), self.empty_dict()
        upd = [upd_domain, upd_idx, upd_val]

        for i in range(self.num_domain):
            # FIXME Inconsistent node index for new_history (split_indices)
            # and elsewhere.
            node, idx = split['decision'][i]
            node = self.split_nodes[node].name
            if split.get('points', None) is not None:
                points = split['points'][i]
            else:
                points = 0.
            for j in range(2):
                history_idx = (-self.num_copy * self.num_domain
                               + j * self.num_domain + i)
                upd_domain[node].append(i)
                upd_idx[node].append(idx)
                upd_val[node].append(points)
                if self.history is not None:
                    self._append_history(history_idx, node, (idx, 1 - j * 2, points))

        for upd_list in upd:
            for k in upd_list:
                upd_list[k] = self.as_tensor(upd_list[k])
        for k in self.node_names:
            if len(upd_domain[k]):
                d['lower_bounds'][k][0].view(self.num_domain, -1)[
                    upd_domain[k], upd_idx[k]] = upd_val[k]
                d['upper_bounds'][k][1].view(self.num_domain, -1)[
                    upd_domain[k], upd_idx[k]] = upd_val[k]


class DomainUpdaterInputSplit(DomainUpdater):

    def set_branched_bounds(self, d, split, mode='depth'):
        # Disable history during the input split
        # d['history'] = None
        # # Just need the input and the final output bounds
        d['lower_bounds'] = {k: v for k, v in d['lower_bounds'].items()
                                if k in [self.root.name, self.final_name]}
        d['upper_bounds'] = {k: v for k, v in d['upper_bounds'].items()
                                if k in [self.root.name, self.final_name]}
        super().set_branched_bounds(d, split, mode)

    def _set_history_and_bounds(self, d, split, mode='depth'):
        assert mode == 'depth'
        assert split.get('points', None) is not None
        assert self.num_split == 1
        assert self.num_copy == self.num_branches

        if self.num_branches == 2:
            # Shortcut
            assert split['points'].ndim == 1
            idx = self.as_tensor([split['decision'][i][1]
                                  for i in range(self.num_domain)])
            ub = d['upper_bounds'][self.root.name].view(
                self.num_copy, self.num_domain, -1)
            ub[0].scatter_(dim=-1, index=idx.unsqueeze(-1),
                           src=split['points'].unsqueeze(-1))
            lb = d['lower_bounds'][self.root.name].view(
                self.num_copy, self.num_domain, -1)
            lb[1].scatter_(dim=-1, index=idx.unsqueeze(-1),
                           src=split['points'].unsqueeze(-1))
            d['upper_bounds'][self.root.name] = ub.view(
                d['upper_bounds'][self.root.name].shape)
            d['lower_bounds'][self.root.name] = lb.view(
                d['lower_bounds'][self.root.name].shape)
        else:
            raise NotImplementedError

            # FIXME Some None issue in the history
            # super()._set_history_and_bounds(d, split, mode)
