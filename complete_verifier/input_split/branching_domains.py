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
from typing import Union
import torch
from torch import Tensor
from typing import Union, Tuple
from tensor_storage import get_tensor_storage

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
                 sort_index=None, sort_descending=True, use_split_idx=True):
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
        self.storage_depth = storage_depth
        self.sort_descending = sort_descending
        self.volume = self.all_volume = None
        self.use_split_idx = use_split_idx

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

    def filter_verified_domains(
            self,
            batch: int,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: Union[dict, list, None],
            cs: Tensor,
            threshold: Union[int, Tensor] = 0,
            lA: Union[Tensor, None] = None,
            lbias: Union[Tensor, None] = None,
            check_thresholds=True,
            check_bounds=True
    ) -> Tuple[int, Tensor, Tensor, Tensor, Union[dict, list, None],
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """
        Filters out the domains that are verified and only returns unverified domains
        @param batch:                                   Batch size of domains
        @param lb: (batch, spec_dim)                    Domain lower bound output
        @param dm_l: (batch, dim_in)                    Input domain lower input bound
        @param dm_u: (batch, dim_in)                    Input domain upper input bound
        @param alpha:                                   CROWN alpha parameters for domains
        @param cs: (batch, spec_dim, lA_rows)           specification matrix
        @param threshold: (batch, spec_dim)             Threshold to verify specification with
        @param lA: (batch, lA_rows or spec_dim, dim_in) CROWN lA coefficient matrix
        @param lbias: (batch, spec_dim)                 CROWN lbias coefficient matrix
        @param check_thresholds:                        If true, filters out domains that have been verified
                                                        by lb > threshold
        @param check_bounds:                            If true, filters out domains that have been verified
                                                        by dm_l < dm_u
        @param double_alphas:                           If true, the alphas are repeated along the batch dimension by
                                                        split_partitions ** split_depth
        @param split_partitions:                        The number of partitions that domains are split into for BaB
        @return:
        """
        remaining_index = self.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_thresholds, check_bounds
        )
        lb_filt = lb[remaining_index]
        dm_l_filt = dm_l[remaining_index]
        dm_u_filt = dm_u[remaining_index]
        cs_filt = cs[remaining_index]
        batch_filt = len(dm_l_filt)
        alpha_filt = []
        if self.use_alpha and batch_filt > 0:
            alpha_filt = self.filter_alpha(alpha, remaining_index)
        threshold_filt = threshold[remaining_index]
        lA_filt = lA[remaining_index] if lA is not None else None
        lbias_filt = lbias[remaining_index] if lbias is not None else None

        return batch_filt, lb_filt, dm_l_filt, dm_u_filt, alpha_filt, cs_filt, threshold_filt, lA_filt, lbias_filt

    def filter_alpha(
            self,
            alpha: dict,
            remaining_index: Tensor
    ) -> dict:
        """
        Filters alphas w.r.t. remaining_index
        @param alpha:               Dictionary of alpha parameters
        @param remaining_index:     Batch indices to retain, typically the unverified indices
        @return:                    Filtered alpha dictionary
        """
        is_tensor = isinstance(remaining_index, Tensor)
        with torch.no_grad():
            alpha_filt = {}
            on_device = False # ensure remaining_index is on the correct device
            for key0 in alpha.keys():
                alpha_filt[key0] = {}
                for key1 in alpha[key0].keys():
                    if not on_device and is_tensor:
                        remaining_index = remaining_index.to(device=alpha[key0][key1].device)
                        on_device=True
                    # alpha[key0][key1] has shape (dim_in, spec_dim, batches, unstable size)
                    alpha_filt[key0][key1] = alpha[key0][key1][:, :, remaining_index]

        return alpha_filt

    def get_remaining_index(
            self,
            batch: int,
            lb: Tensor,
            threshold: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            check_thresholds=True,
            check_bounds=True
    ) -> Union[Tensor, Tuple]:
        """
        Gets the indices of the batch instances that are not verified. Verification conditions are specified by
        the check_thresholds and check_bounds flags. If both are None, all indicies are returned.
        @param batch:                       Batch size of domains
        @param lb: (batch, spec_dim)        Domain lower bound output
        @param threshold: (batch, spec_dim) Threshold to verify specification with
        @param dm_l: (batch, dim_in)        Input domain lower input bound
        @param dm_u: (batch, dim_in)        Input domain upper input bound
        @param check_thresholds:            If true, filters out domains that have been verified by lb > threshold
        @param check_bounds:                If true, filters out domains that have been verified by dm_l < dm_u
        @return:                            The indices of the batch instances that are left unverified
        """

        if check_thresholds and check_bounds:
            return torch.where(
                torch.logical_and(
                    (lb <= threshold).all(1),
                    (dm_l.view(batch, -1) <= dm_u.view(batch, -1)).all(1)
                )
            )[0]
        elif check_thresholds:
            return torch.where(
                    (lb <= threshold).all(1)
            )[0]
        elif check_bounds:
            return torch.where(
                    (dm_l.view(batch, -1) <= dm_u.view(batch, -1)).all(1)
            )[0]
        else:
            return slice(None)

    def add(
            self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: Tensor,
            cs: Tensor,
            threshold: Union[int, Tensor] = 0,
            split_idx: Union[Tensor, None] = None,
            remaining_index: Union[Tensor, None] = None,
            check_thresholds=True,
            check_bounds=True
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains
        @param lb: Shape (batch, input_dim)                 Lower bound on domain outputs
        @param dm_l: Shape (batch, num_spec)                Lower bound on domain inputs
        @param dm_u: Shape (batch, num_spec)                Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @return:                                            None
        """
        # check shape correctness
        batch = len(lb)
        if batch == 0:
            return
        if self.use_split_idx:
            assert split_idx is not None, "Cannot accept split_idx"
            assert len(split_idx) == batch
            assert split_idx.shape[1] == self.storage_depth
        else:
            assert split_idx is None, "Expected to receive split_idx"
        if type(threshold) == int:
            threshold = torch.zeros(batch, 2)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_alpha:
            if alpha is None:
                raise ValueError("alpha should not be None in alpha-crown.")
        # initialize attributes using input shapes
        if self.lb is None:
            self.lb = get_tensor_storage(lb.shape)
        if self.dm_l is None:
            self.dm_l = get_tensor_storage(dm_l.shape)
        if self.dm_u is None:
            self.dm_u = get_tensor_storage(dm_u.shape)
        if self.use_alpha and not self.alpha:
            if type(alpha) == list:
                assert len(alpha) > 0
                for key0 in alpha[0].keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[0][key0].keys():
                        self.alpha[key0][key1] = get_tensor_storage(
                            alpha[0][key0][key1].shape, concat_dim=2
                        )
            else:
                for key0 in alpha.keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[key0].keys():
                        self.alpha[key0][key1] = get_tensor_storage(
                            alpha[key0][key1].shape, concat_dim=2
                        )
        if self.cs is None:
            self.cs = get_tensor_storage(cs.shape)
        if self.threshold is None:
            self.threshold = get_tensor_storage(threshold.shape)
        if self.split_idx is None and self.use_split_idx:
            self.split_idx = get_tensor_storage([None, self.storage_depth])
        # compute unverified indices
        if remaining_index is None:
            remaining_index = self.get_remaining_index(
                batch, lb, threshold, dm_l, dm_u, check_thresholds, check_bounds
            )
            if isinstance(remaining_index, Tensor):
                remaining_index = remaining_index.detach().cpu()
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
        if self.use_split_idx:
            self.split_idx.append(
                split_idx[remaining_index]
                .type(self.split_idx.dtype)
                .to(self.split_idx.device)
            )

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = min(len(self), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        lb = self.lb.pop(batch).to(device=device, non_blocking=True)
        dm_l = self.dm_l.pop(batch).to(device=device, non_blocking=True)
        dm_u = self.dm_u.pop(batch).to(device=device, non_blocking=True)
        alpha, val = [], []
        if self.use_alpha:
            for key0, val0 in self.alpha.items():
                for key1, val1 in val0.items():
                    val.append(val1.pop(batch))
            for i in range(batch):
                val_idx, item = 0, {}
                for key0, val0 in self.alpha.items():
                    item[key0] = {}
                    for key1 in val0.keys():
                        item[key0][key1] = val[val_idx][:, :, i: i + 1].to(
                            device=device, non_blocking=True
                        )
                        val_idx += 1
                alpha.append(item)
        cs = self.cs.pop(batch).to(device=device, non_blocking=True)
        threshold = self.threshold.pop(batch).to(device=device, non_blocking=True)
        if self.use_split_idx:
            split_idx = self.split_idx.pop(batch).to(device=device, non_blocking=True)
        else:
            split_idx = None
        self._add_volume(dm_l, dm_u, sign=-1)
        return alpha, lb, dm_l, dm_u, cs, threshold, split_idx

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
        assert k <= len(self), print("Asked indices more than domain length.")
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
            for val0 in self.alpha.values():
                for val1 in val0.values():
                    val1._storage[
                    :, :, :val1.num_used] = val1._storage[:, :, indices]
        self.cs._storage[: self.cs.num_used] = self.cs._storage[indices]
        self.threshold._storage[: self.threshold.num_used] = self.threshold._storage[indices]
        if self.use_split_idx:
            self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]
