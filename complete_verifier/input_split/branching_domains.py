#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch
from tensor_storage import TensorStorage


class InputDomainList:
    """Abstract class that maintains a list of domains for input split."""

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # get lb, dm_l, dm_u, cs, threshold for idx; for convenience, alpha and split_idx
        # are not returned for now
        raise NotImplementedError

    def add(self, lb, dm_l, dm_u, alpha, cs, threshold=0, split_idx=None,
            remaining_index=None):
        raise NotImplementedError

    def pick_out_batch(self, batch, device="cuda"):
        raise NotImplementedError

    def get_topk_indices(self, k=1, largest=False):
        # get the topk indices, by default worst k
        raise NotImplementedError


class UnsortedInputDomainList(InputDomainList):
    """Unsorted domain list for input split."""

    def __init__(self, storage_depth, use_alpha=False,
                 sort_index=None, sort_descending=True):
        super(UnsortedInputDomainList, self).__init__()
        self.lb = None
        self.dm_l = None
        self.dm_u = None
        self.alpha = {}
        self.use_alpha = use_alpha
        self.sort_index = sort_index
        self.cs = None
        self.threshold = None
        self.split_idx = None
        self.last_split_idx = None
        self.storage_depth = storage_depth
        self.sort_descending = sort_descending
        self.volume = self.all_volume = None

    def __len__(self):
        if self.dm_l is None:
            return 0
        return self.dm_l.num_used

    def __getitem__(self, idx):
        return (
            self.lb._storage[idx],
            self.dm_l._storage[idx],
            self.dm_u._storage[idx],
            self.cs._storage[idx],
            self.threshold._storage[idx],
        )

    def add(self, lb, dm_l, dm_u, alpha, cs, threshold=0, split_idx=None,
            remaining_index=None, last_split_idx=None):
        # check shape correctness
        batch = len(lb)
        if type(threshold) == int:
            threshold = torch.zeros(batch, 2)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_alpha:
            if alpha is None:
                raise ValueError("alpha should not be None in alpha-crown.")
        assert len(split_idx) == batch
        assert split_idx.shape[1] == self.storage_depth
        # initialize attributes using input shapes
        if self.lb is None:
            self.lb = TensorStorage(lb.shape)
        if self.dm_l is None:
            self.dm_l = TensorStorage(dm_l.shape)
        if self.dm_u is None:
            self.dm_u = TensorStorage(dm_u.shape)
        if self.use_alpha and not self.alpha:
            if type(alpha) == list:
                assert len(alpha) > 0
                for key0 in alpha[0].keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[0][key0].keys():
                        self.alpha[key0][key1] = TensorStorage(
                            alpha[0][key0][key1].shape, concat_dim=2
                        )
            else:
                for key0 in alpha.keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[key0].keys():
                        self.alpha[key0][key1] = TensorStorage(
                            alpha[key0][key1].shape, concat_dim=2
                        )
        if self.cs is None:
            self.cs = TensorStorage(cs.shape)
        if self.threshold is None:
            self.threshold = TensorStorage(threshold.shape)
        if self.split_idx is None:
            self.split_idx = TensorStorage([None, self.storage_depth])
        if self.last_split_idx is None:
            self.last_split_idx = TensorStorage(
                [None, self.storage_depth], dtype=torch.long)
        # compute unverified indices
        if remaining_index is None:
            remaining_index = torch.where(
                (lb.detach().cpu() <= threshold.detach().cpu()).all(1)
            )[0]
        # append the tensors
        self.lb.append(lb[remaining_index].type(self.lb.dtype).to(self.lb.device))

        dm_l = dm_l[remaining_index]
        dm_u = dm_u[remaining_index]
        self._add_volume(dm_l, dm_u)
        self.dm_l.append(dm_l.type(self.dm_l.dtype).to(self.dm_l.device))
        self.dm_u.append(dm_u.type(self.dm_u.dtype).to(self.dm_u.device))
        if self.use_alpha:
            if type(alpha) == list:
                for i in remaining_index:
                    for key0 in alpha[0].keys():
                        for key1 in alpha[0][key0].keys():
                            self.alpha[key0][key1].append(
                                alpha[i][key0][key1]
                                .type(self.alpha[key0][key1].dtype)
                                .to(self.alpha[key0][key1].device)
                            )
            else:
                for key0 in alpha.keys():
                    for key1 in alpha[key0].keys():
                        self.alpha[key0][key1].append(
                            alpha[key0][key1][:, :, remaining_index]
                            .type(self.alpha[key0][key1].dtype)
                            .to(self.alpha[key0][key1].device)
                        )
        self.cs.append(cs[remaining_index].type(self.cs.dtype).to(self.cs.device))
        self.threshold.append(
            threshold[remaining_index]
            .type(self.threshold.dtype)
            .to(self.threshold.device)
        )
        self.split_idx.append(
            split_idx[remaining_index]
            .type(self.split_idx.dtype)
            .to(self.split_idx.device)
        )
        if last_split_idx is None:
            self.last_split_idx.append(
                (torch.ones_like(split_idx[remaining_index])*(-1))
                .type(self.split_idx.dtype)
                .to(self.split_idx.device)
            )
        else:
            self.last_split_idx.append(
                last_split_idx[remaining_index]
                .type(self.split_idx.dtype)
                .to(self.split_idx.device)
            )

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = min(self.__len__(), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        lb = self.lb.pop(batch).to(device=device, non_blocking=True)
        dm_l = self.dm_l.pop(batch).to(device=device, non_blocking=True)
        dm_u = self.dm_u.pop(batch).to(device=device, non_blocking=True)
        alpha, val = [], []
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    val.append(self.alpha[key0][key1].pop(batch))
            for i in range(batch):
                val_idx, item = 0, {}
                for key0 in self.alpha.keys():
                    item[key0] = {}
                    for key1 in self.alpha[key0].keys():
                        item[key0][key1] = val[val_idx][:, :, i : i + 1].to(
                            device=device, non_blocking=True
                        )
                        val_idx += 1
                alpha.append(item)
        cs = self.cs.pop(batch).to(device=device, non_blocking=True)
        threshold = self.threshold.pop(batch).to(device=device, non_blocking=True)
        split_idx = self.split_idx.pop(batch).to(device=device, non_blocking=True)
        last_split_idx = self.last_split_idx.pop(batch).to(device=device, non_blocking=True)
        self._add_volume(dm_l, dm_u, sign=-1)
        return alpha, lb, dm_l, dm_u, cs, threshold, split_idx, last_split_idx

    def _add_volume(self, dm_l, dm_u, sign=1):
        volume = torch.prod(dm_u - dm_l, dim=-1).sum().item()
        if self.all_volume is None:
            self.all_volume = volume
            self.volume = 0
        self.volume = self.volume + sign * volume

    def get_progess(self):
        if self.all_volume is None or self.all_volume == 0:
            return 0.
        else:
            return 1 - self.volume / self.all_volume

    def _get_sort_margin(self, margin):
        if self.sort_index is not None:
            return margin[..., self.sort_index]
        else:
            return margin.max(dim=1).values

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        indices = self._get_sort_margin(lb - threshold).topk(k, largest=largest).indices
        return indices

    def sort(self):
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        indices = self._get_sort_margin(lb - threshold).argsort(
            descending=self.sort_descending)
        # sort the storage
        self.lb._storage[: self.lb.num_used] = self.lb._storage[indices]
        self.dm_l._storage[: self.dm_l.num_used] = self.dm_l._storage[indices]
        self.dm_u._storage[: self.dm_u.num_used] = self.dm_u._storage[indices]
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    self.alpha[key0][key1]._storage[
                        :, :, :self.alpha[key0][key1].num_used
                    ] = self.alpha[key0][key1]._storage[:, :, indices]
        self.cs._storage[: self.cs.num_used] = self.cs._storage[indices]
        self.threshold._storage[: self.threshold.num_used] = self.threshold._storage[indices]
        self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]
