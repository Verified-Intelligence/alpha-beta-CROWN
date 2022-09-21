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
import numpy as np
import math
from sortedcontainers import SortedList
from tensor_storage import TensorStorage
import arguments


class InputDomainList:
    """Abstract class that maintains a list of domains for input split."""

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # get lb, dm_l, dm_u, cs, threshold for idx; for convenience, slope and split_idx
        # are not returned for now
        raise NotImplementedError

    def add(
        self,
        lb,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None,
        remaining_index=None,
    ):
        raise NotImplementedError

    def pick_out_batch(self, batch, device="cuda"):
        raise NotImplementedError

    def get_topk_indices(self, k=1, largest=False):
        # get the topk indices, by default worst k
        raise NotImplementedError


class UnsortedInputDomainList(InputDomainList):
    """Unsorted domain list for input split."""

    def __init__(self, storage_depth, use_slope=False):
        super(UnsortedInputDomainList, self).__init__()
        self.lb = None
        self.dm_l = None
        self.dm_u = None
        self.slope = {}
        self.use_slope = use_slope
        self.cs = None
        self.threshold = 0
        self.split_idx = None
        self.storage_depth = storage_depth

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

    def add(
        self,
        lb,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None,
        remaining_index=None,
    ):
        # check shape correctness
        batch = len(lb)
        if type(threshold) == int:
            threshold = torch.zeros(batch, 2)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_slope:
            if slope is None:
                raise ValueError("Slope should not be None in alpha-crown.")
        assert len(split_idx) == batch
        assert split_idx.shape[1] == self.storage_depth
        # initialize attributes using input shapes
        if self.lb is None:
            self.lb = TensorStorage(lb.shape)
        if self.dm_l is None:
            self.dm_l = TensorStorage(dm_l.shape)
        if self.dm_u is None:
            self.dm_u = TensorStorage(dm_u.shape)
        if self.use_slope and self.slope == {}:
            if type(slope) == list:
                assert len(slope) > 0
                for key0 in slope[0].keys():
                    self.slope[key0] = {}
                    for key1 in slope[0][key0].keys():
                        self.slope[key0][key1] = TensorStorage(
                            slope[0][key0][key1].shape, concat_dim=2
                        )
            else:
                for key0 in slope.keys():
                    self.slope[key0] = {}
                    for key1 in slope[key0].keys():
                        self.slope[key0][key1] = TensorStorage(
                            slope[key0][key1].shape, concat_dim=2
                        )
        if self.cs is None:
            self.cs = TensorStorage(cs.shape)
        if self.threshold == 0:
            self.threshold = TensorStorage(threshold.shape)
        if self.split_idx is None:
            self.split_idx = TensorStorage([None, self.storage_depth])
        # compute unverified indices
        if remaining_index is None:
            remaining_index = torch.where(
                (lb.detach().cpu() <= threshold.detach().cpu()).all(1)
            )[0]
        # append the tensors
        self.lb.append(lb[remaining_index].type(self.lb.dtype).to(self.lb.device))
        self.dm_l.append(
            dm_l[remaining_index].type(self.dm_l.dtype).to(self.dm_l.device)
        )
        self.dm_u.append(
            dm_u[remaining_index].type(self.dm_u.dtype).to(self.dm_u.device)
        )
        if self.use_slope:
            if type(slope) == list:
                for i in remaining_index:
                    for key0 in slope[0].keys():
                        for key1 in slope[0][key0].keys():
                            self.slope[key0][key1].append(
                                slope[i][key0][key1]
                                .type(self.slope[key0][key1].dtype)
                                .to(self.slope[key0][key1].device)
                            )
            else:
                for key0 in slope.keys():
                    for key1 in slope[key0].keys():
                        self.slope[key0][key1].append(
                            slope[key0][key1][:, :, remaining_index]
                            .type(self.slope[key0][key1].dtype)
                            .to(self.slope[key0][key1].device)
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

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = min(self.__len__(), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        self.lb.pop(batch)
        dm_l = self.dm_l.pop(batch).to(device=device, non_blocking=True)
        dm_u = self.dm_u.pop(batch).to(device=device, non_blocking=True)
        slope, val = [], []
        if self.use_slope:
            for key0 in self.slope.keys():
                for key1 in self.slope[key0].keys():
                    val.append(self.slope[key0][key1].pop(batch))
            for i in range(batch):
                val_idx, item = 0, {}
                for key0 in self.slope.keys():
                    item[key0] = {}
                    for key1 in self.slope[key0].keys():
                        item[key0][key1] = val[val_idx][:, :, i : i + 1].to(
                            device=device, non_blocking=True
                        )
                        val_idx += 1
                slope.append(item)
        cs = self.cs.pop(batch).to(device=device, non_blocking=True)
        threshold = self.threshold.pop(batch).to(device=device, non_blocking=True)
        split_idx = None
        split_idx = self.split_idx.pop(batch).to(device=device, non_blocking=True)
        return slope, dm_l, dm_u, cs, threshold, split_idx

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        indices = (lb - threshold).max(dim=1).values.topk(k, largest=largest).indices
        return indices

    def sort(self):
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        indices = (lb - threshold).max(dim=1).values.argsort(descending=True)
        # sort the storage
        self.lb._storage[: self.lb.num_used] = self.lb._storage[indices]
        self.dm_l._storage[: self.dm_l.num_used] = self.dm_l._storage[indices]
        self.dm_u._storage[: self.dm_u.num_used] = self.dm_u._storage[indices]
        if self.use_slope:
            for key0 in self.slope.keys():
                for key1 in self.slope[key0].keys():
                    self.slope[key0][key1]._storage[: self.slope[key0][key1].num_used] = \
                            self.slope[key0][key1]._storage[indices]
        self.cs._storage[: self.cs.num_used] = self.cs._storage[indices]
        self.threshold._storage[: self.threshold.num_used] = self.threshold._storage[indices]
        self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]


class SortedInputDomainList(InputDomainList):
    """Sorted domain list for input split."""

    def __init__(self):
        super(SortedInputDomainList, self).__init__()
        self.domains = SortedList()
        min_batch_size = (
            arguments.Config["solver"]["beta-crown"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
        )
        self.max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]
        return domain.lower_bound, domain.dm_l, domain.dm_u, domain.c, domain.threshold

    def add(
        self,
        lb,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None,
        remaining_index=None,
    ):
        threshold = threshold.to(device=lb.device)
        dm_l = dm_l.to(device=lb.device)
        dm_u = dm_u.to(device=lb.device)
        cs = cs.to(device=lb.device)
        if remaining_index is None:
            # FIXME: this should check the criterion function.
            remaining_index = torch.where((lb <= threshold).all(1))[0]
        for i in remaining_index:
            if slope is not None and type(slope) != list:
                slope_dict = {}
                for key0 in slope.keys():
                    slope_dict[key0] = {}
                    for key1 in slope[key0].keys():
                        slope_dict[key0][key1] = slope[key0][key1][:, :, i : i + 1]
            dom = InputDomain(
                lb=lb[i],
                slope=slope_dict if slope is not None and type(slope) != list else None,
                dm_l=dm_l[i : i + 1],
                dm_u=dm_u[i : i + 1],
                threshold=threshold[i],
                c=cs[i : i + 1],
                split_idx=(split_idx[i : i + 1] if split_idx is not None else None),
            )
            self.domains.add(dom)

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self.domains), batch)
        # Lower and upper bounds of inputs.
        dm_l_all, dm_u_all = [], []
        # Specification matrices, and decision thresholds.
        c_all, thresholds_all = [], []
        slopes_all, split_idx = [], []
        assert len(self.domains) > 0, "The given domains list is empty."

        for i in range(batch):
            # Pop out domains from the list one by one (SLOW).
            selected_candidate_domain = self.domains.pop(0)
            # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
            selected_candidate_domain.to_device(device, partial=True)
            slopes_all.append(selected_candidate_domain.slope)
            dm_l_all.append(selected_candidate_domain.dm_l)
            dm_u_all.append(selected_candidate_domain.dm_u)
            c_all.append(selected_candidate_domain.c)
            thresholds_all.append(selected_candidate_domain.threshold)
            if (
                split_idx is not None
                and selected_candidate_domain.split_idx is not None
            ):
                split_idx.append(selected_candidate_domain.split_idx)
            else:
                split_idx = None

        thresholds = torch.stack(thresholds_all).to(device=device, non_blocking=True)
        split_idx = torch.cat(split_idx) if split_idx is not None else None

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).contiguous().to(device=device, non_blocking=True)

        # Input split domains.
        return (
            slopes_all,
            torch.cat(dm_l_all).to(device=device, non_blocking=True),
            torch.cat(dm_u_all).to(device=device, non_blocking=True),
            cs,
            thresholds,
            split_idx,
        )

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        return -torch.arange(k) - 1 if largest else torch.arange(k)


class InputDomain:
    """Singleton domain for input split, used by sorted domain list."""

    def __init__(
        self,
        lb=-float("inf"),
        ub=float("inf"),
        lb_all=None,
        up_all=None,
        slope=None,
        dm_l=None,
        dm_u=None,
        selected_dims=None,
        device="cpu",
        depth=None,
        c=None,
        threshold=0,
        split_idx=None,
    ):
        self.lower_bound = lb
        self.upper_bound = ub
        self.dm_l = dm_l
        self.dm_u = dm_u
        self.lower_all = lb_all
        self.upper_all = up_all
        self.slope = slope
        self.selected_dims = selected_dims
        self.device = device
        self.split = False
        self.valid = True
        self.depth = depth
        self.lA = None
        self.beta = None
        self.intermediate_betas = None
        self.c = c
        self.threshold = threshold
        self.split_idx = split_idx

    def __lt__(self, other):
        return (self.lower_bound - self.threshold).max() < (
            other.lower_bound - other.threshold
        ).max()

    def __le__(self, other):
        return (self.lower_bound - self.threshold).max() <= (
            other.lower_bound - other.threshold
        ).max()

    def __eq__(self, other):
        return (self.lower_bound - self.threshold).max() == (
            other.lower_bound - other.threshold
        ).max()

    def __repr__(self):
        string = (
            f"LB: {self.lower_bound - self.threshold}\n"
            if self.lower_bound is not None
            else ""
        )
        string += (
            f"UB: {self.upper_bound - self.threshold}\n"
            if self.upper_bound is not None
            else ""
        )
        return string

    def verify_criterion(self):
        return (self.lower_bound > self.threshold).any()

    def attack_criterion(self):
        return (self.upper_bound <= self.threshold).all()

    def to_cpu(self):
        if self.device == "cuda":
            return self
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.dm_l = self.dm_l.to(device="cpu", non_blocking=True)
        self.dm_u = self.dm_u.to(device="cpu", non_blocking=True)
        self.lower_bound = self.lower_bound.to(device="cpu", non_blocking=True)
        self.upper_bound = self.upper_bound.to(device="cpu", non_blocking=True)
        if self.selected_dims is not None:
            self.selected_dims = self.selected_dims.to(device="cpu", non_blocking=True)

        if self.c is not None:
            self.c = self.c.to(device="cpu", non_blocking=True)

        self.threshold = self.threshold.to(device="cpu", non_blocking=True)

        if self.slope is not None:
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = (
                        self.slope[layer][intermediate_layer]
                        .half()
                        .to(device="cpu", non_blocking=True)
                    )

        return self

    def to_device(self, device, partial=False):
        if self.device == "cuda":
            return self
        self.dm_l = self.dm_l.to(device, non_blocking=True)
        self.dm_u = self.dm_u.to(device, non_blocking=True)
        self.lower_bound = (
            self.lower_bound.to(device, non_blocking=True)
            if self.lower_bound is not None
            else None
        )

        if self.c is not None:
            self.c = self.c.to(device, non_blocking=True)

        self.threshold = self.threshold.to(device, non_blocking=True)

        if self.slope is not None:
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = (
                        self.slope[layer][intermediate_layer]
                        .to(device, non_blocking=True)
                        .to(torch.get_default_dtype())
                    )

        return self
