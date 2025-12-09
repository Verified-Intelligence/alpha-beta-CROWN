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
from torch import Tensor
from typing import Optional, Union, Tuple, List
from tensor_storage import get_tensor_storage
from abc import ABC, abstractmethod
import psutil, os

from utils import pad_list_of_input_to_tensor

class InputDomainList(ABC):
    """
    Abstract class that maintains a list of domains for input split.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        """Number of domains remaining in the list"""
        ...

    @abstractmethod
    def __getitem__(self, *args, **kwargs):
        """
        get lb, dm_l, dm_u, cs, threshold
        (and possibly more element) for idx
        """
        ...

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add domains to the list"""
        ...

    @abstractmethod
    def pick_out_batch(self, batch_size, device=None):
        """Pick out a batch of subdomains from the domain list."""
        ...

    @staticmethod
    def get_topk_indices(self, k=1, largest=False):
        """get the topk indices, by default worst k"""
        ...


class UnsortedInputDomainList(InputDomainList):
    """Unsorted domain list for input split."""

    def __init__(
        self,
        storage_depth: int,
        output_device: str,
        use_alpha: bool=False,
        sort_index: Optional[int]=None,
        sort_descending: bool=True,
        use_split_idx: bool=True,
    ):
        """
        The initialization method for the UnsortedInputDomainList class.
        
        :param storage_depth:       The maximum number of splits we could use in input BaB
        :param output_device:       The default device storing accessed data. Complete domain data is always stored on 'cpu'.
        :param use_alpha:           True if we must also store alpha parameters
        :param sort_index:          The index along which to sort the domains
        :param sort_descending:     If True, domains will get sorted in descending order w.r.t. their lower bounds
                                    whenever the 'sort' method is called.
        :param use_split_idx:       If True, we also store the split indices for each domain
        """
        super(UnsortedInputDomainList, self).__init__()
        self.output_device = output_device
        self.lb = None
        self.dm_l = None
        self.dm_u = None
        self.alpha = {}
        self.use_alpha = use_alpha
        self.sort_index = sort_index
        self.cs = None
        self.threshold = None
        self.constraint_A = None
        self.constraint_b = None
        self.split_idx = None
        self.storage_depth = storage_depth
        self.sort_descending = sort_descending
        self.volume = self.all_volume = None
        self.use_split_idx = use_split_idx
        self.spec_size = None

    def __len__(self):
        if self.dm_l is None:
            return 0
        return self.dm_l.num_used

    def __getitem__(self, idx):
        # convert idx to tensor on cpu for slicing.
        if isinstance(idx, slice):
            idx = torch.arange(len(self), device="cpu")[idx]
        else:
            idx = torch.as_tensor(idx, device="cpu").view(-1)
        assert idx.shape[0] > 0, "Empty index"
        output_device = self.output_device
        return (
            self.lb._storage[idx].to(output_device),
            self.dm_l._storage[idx].to(output_device),
            self.dm_u._storage[idx].to(output_device),
            self.cs._storage[idx].to(output_device),
            self.threshold._storage[idx].to(output_device),
            torch.tensor(self.spec_size, device=output_device, dtype=torch.int32).expand(idx.shape[0]),
        )

    @staticmethod
    def filter_verified_domains(
            batch: int,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: dict,
            cs: Tensor,
            threshold: Tensor,
            lA: Optional[Tensor] = None,
            lbias: Optional[Tensor] = None,
            constraints: Optional[tuple] = None,
            split_idx: Optional[Tensor] = None,
            spec_sizes: Optional[Tensor] = None,
            check_dm_lbs: bool = True,
            check_input_boxes: bool = True,
            remaining_index: Optional[Tensor] = None,
            use_alpha: bool = False
    ) -> Tuple[int, Tensor, Tensor, Tensor, dict,
    Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[tuple], Optional[Tensor], Optional[Tensor]]:
        """
        Filters out the domains that are verified and only returns unverified domains

        :param batch:                                   Batch size of domains.
        :param lb: (batch, spec_dim)                    Domain lower bound output
        :param dm_l: (batch, dim_in)                    Input domain lower input bound
        :param dm_u: (batch, dim_in)                    Input domain upper input bound
        :param alpha:                                   CROWN alpha parameters for domains
        :param cs: (batch, spec_dim, lA_rows)           specification matrix
        :param threshold: (batch, spec_dim)             Threshold to verify specification with
        :param lA: (batch, lA_rows or spec_dim, dim_in) CROWN lA coefficient matrix
        :param lbias: (batch, spec_dim)                 CROWN lbias coefficient matrix
        :param constraints:                             A tuple of linear constraints (constr_A, constr_b)
        :param split_idx: (batch, num of splits)        Specifies along which dimensions to split
        :param spec_sizes: (batch)                      The number of ANDs in the spec of each domain
        :param check_dm_lbs:                            If true, filters out domains that have been verified
                                                        by lb > threshold
        :param check_input_boxes:                       If true, filters out domains that have been verified
                                                        by dm_l > dm_u
        :param remaining_index:                         If not None, user is specifying which domains are unverified

        :return batch_filt:                             Number of filtered domains.
        :return lb_filt:                                Filtered domain lower bound output
        :return dm_l_filt:                              Filtered input domain lower input bound
        :return dm_u_filt:                              Filtered input domain upper input bound
        :return alpha_filt:                             Filtered CROWN alpha parameters for domains
        :return cs_filt:                                Filtered specification matrix
        :return threshold_filt:                         Filtered threshold to verify specification with
        :return lA_filt:                                Filtered CROWN lA coefficient matrix
        :return lbias_filt:                             Filtered CROWN lbias coefficient matrix
        :return constraints_filt:                       Filtered linear constraints tuple (constr_A, constr_b)
        :return split_idx_filt:                         Filtered specifies along which dimensions to split
        :return spec_size_filt:                         Filtered the number of ANDs in the spec of each domain
        """
        if remaining_index is None:
            remaining_index = UnsortedInputDomainList.get_remaining_index(
                batch, lb, threshold, dm_l, dm_u, check_dm_lbs, check_input_boxes
            )
        lb_filt = lb[remaining_index]
        dm_l_filt = dm_l[remaining_index]
        dm_u_filt = dm_u[remaining_index]
        cs_filt = cs[remaining_index]
        batch_filt = len(dm_l_filt)
        alpha_filt = {}
        if use_alpha and batch_filt > 0:
            with torch.no_grad():
                # alpha may have different decive from other tensors.
                # In get_lower_bound_naive() we transfer alpha to cpu and float16.
                alpha_device = next(iter(next(iter(alpha.values())).values())).device
                if isinstance(remaining_index, Tensor):
                    remaining_index_for_alpha = remaining_index.to(alpha_device)
                else:
                    remaining_index_for_alpha = remaining_index
                for key0 in alpha.keys():
                    alpha_filt[key0] = {}
                    for key1 in alpha[key0].keys():
                        # alpha[key0][key1] has shape (alpha_size, prod(start_node_shape), batch_size, node_size)
                        alpha_filt[key0][key1] = alpha[key0][key1][:, :, remaining_index_for_alpha]

        threshold_filt = threshold[remaining_index]
        lA_filt = lA[remaining_index] if lA is not None else None
        lbias_filt = lbias[remaining_index] if lbias is not None else None

        constraints_filt = None
        if constraints is not None:
            c_A, c_b = constraints
            c_A_filt = c_A[remaining_index]
            c_b_filt = c_b[remaining_index]
            constraints_filt = (c_A_filt, c_b_filt)

        split_idx_filt = split_idx[remaining_index] if split_idx is not None else None
        spec_sizes_filt = spec_sizes[remaining_index] if spec_sizes is not None else None

        return batch_filt, lb_filt, dm_l_filt, dm_u_filt, alpha_filt, cs_filt, threshold_filt, lA_filt, lbias_filt, constraints_filt, split_idx_filt, spec_sizes_filt

    @staticmethod
    def get_remaining_index(
            batch: int,
            lb: Tensor,
            threshold: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            check_dm_lbs=True,
            check_input_boxes=True
    ) -> Union[slice, Tensor]:
        """
        Gets the indices of the batch instances that are not verified. Verification conditions are specified by
        the check_dm_lbs and check_input_boxes flags. If both are None, all indicies are returned.

        @param batch:                       Batch size of domains
        @param lb: (batch, spec_dim)        Domain lower bound output
        @param threshold: (batch, spec_dim) Threshold to verify specification with
        @param dm_l: (batch, dim_in)        Input domain lower input bound
        @param dm_u: (batch, dim_in)        Input domain upper input bound
        @param check_dm_lbs:                If true, filters out domains that have been verified by lb > threshold
        @param check_input_boxes:           If true, filters out domains that have been verified by dm_l > dm_u

        @return:                            The indices of the batch instances that are left unverified
        """

        remaining_mask = torch.ones(batch, dtype=torch.bool, device=lb.device)
        if check_dm_lbs:
            remaining_mask = remaining_mask & (lb <= threshold).all(1)
        if check_input_boxes:
            remaining_mask = remaining_mask & (dm_l.view(batch, -1) <= dm_u.view(batch, -1)).all(1)
        if remaining_mask.all():
            return slice(None)
        return torch.where(remaining_mask)[0]

    def add(
            self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: dict,
            cs: Tensor,
            threshold: Tensor,
            constraints: tuple = None,
            split_idx: Optional[Tensor] = None,
            remaining_index: Optional[Tensor] = None,
            check_dm_lbs: bool=True,
            check_input_boxes: bool=True,
            **kwargs
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains

        @param lb: Shape (batch, num_spec)                  Lower bound on domain outputs
        @param dm_l: Shape (batch, *input_shape)               Lower bound on domain inputs
        @param dm_u: Shape (batch, *input_shape)               Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param constraints:                                 constraints parameters
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @param check_dm_lbs:                                If true, filters out domains that have been verified
                                                            by lb > threshold
        @param check_input_boxes:                           If true, filters out domains that have been verified
                                                            by dm_l > dm_u
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
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_alpha:
            if alpha is None:
                raise ValueError("alpha should not be None in alpha-crown.")
        # initialize attributes using input shapes and types
        if self.lb is None:
            self.lb = get_tensor_storage(lb.shape, dtype=lb.dtype, device="cpu")
        if self.dm_l is None:
            self.dm_l = get_tensor_storage(dm_l.shape, dtype=dm_l.dtype, device="cpu")
        if self.dm_u is None:
            self.dm_u = get_tensor_storage(dm_u.shape, dtype=dm_u.dtype, device="cpu")
        if self.use_alpha and not self.alpha:
            for key0 in alpha.keys():
                self.alpha[key0] = {}
                for key1 in alpha[key0].keys():
                    self.alpha[key0][key1] = get_tensor_storage(
                        alpha[key0][key1].shape, concat_dim=2,
                        dtype=alpha[key0][key1].dtype, device="cpu"
                    )
        if self.cs is None:
            self.cs = get_tensor_storage(cs.shape, dtype=cs.dtype, device="cpu")
        if self.threshold is None:
            self.threshold = get_tensor_storage(threshold.shape, dtype=threshold.dtype, device="cpu")
        if constraints is not None:
            constraint_A, constraint_b = constraints
            if self.constraint_A is None or self.constraint_b is None:
                self.constraint_A = get_tensor_storage(constraint_A.shape, dtype=constraint_A.dtype, device="cpu")
                self.constraint_b = get_tensor_storage(constraint_b.shape, dtype=constraint_b.dtype, device="cpu")
        if self.split_idx is None and self.use_split_idx:
            self.split_idx = get_tensor_storage(split_idx.shape, dtype=split_idx.dtype, device="cpu")
        if self.spec_size is None:
            self.spec_size = cs.shape[1]
        # compute unverified indices
        if remaining_index is None:
            remaining_index = UnsortedInputDomainList.get_remaining_index(
                batch, lb, threshold, dm_l, dm_u, check_dm_lbs, check_input_boxes
            )
        # append the tensors
        self.lb.append(lb[remaining_index].to(self.lb.device))

        dm_l = dm_l[remaining_index]
        dm_u = dm_u[remaining_index]
        self._add_volume(dm_l, dm_u)
        self.dm_l.append(dm_l.to(self.dm_l.device))
        self.dm_u.append(dm_u.to(self.dm_u.device))
        if self.use_alpha:
            for key0 in alpha.keys():
                for key1 in alpha[key0].keys():
                    self.alpha[key0][key1].append(
                        alpha[key0][key1][:, :, remaining_index]
                        .to(self.alpha[key0][key1].device)
                    )
        self.cs.append(cs[remaining_index].to(self.cs.device))
        self.threshold.append(
            threshold[remaining_index]
            .to(self.threshold.device)
        )

        if constraints is not None:
            self.constraint_A.append(
                constraint_A[remaining_index]
                .type(self.constraint_A.dtype)
                .to(self.constraint_A.device)
            )
            self.constraint_b.append(
                constraint_b[remaining_index]
                .type(self.constraint_b.dtype)
                .to(self.constraint_b.device)
            )
        if self.use_split_idx:
            self.split_idx.append(
                split_idx[remaining_index]
                .to(self.split_idx.device)
            )

    def pick_out_batch(self, batch_size: int, device=None
                       )->Tuple[dict, Tensor, Tensor, Tensor, Tensor, Tensor, ]:
        """
        Picks out a batch of subdomains from the domain list.

        :param batch_size:  The maximum number of domains we should pick out
        :param device:      The device all Tensors should be sent to

        :return alphas:         If supported, contains alpha parameters for the batch
        :return lb:             Output lower bounds
        :return dm_l:           Domain input lower bounds
        :return dm_u:           Domain input upper bounds
        :return cs:             Specification matrices
        :return threshold:      Thresholds
        :return constraints:    The constraints tuple.
        :return spec_sizes:     The number of ANDs in the spec of each domain
        :return split_idx:      If supported, the input dimensions we should split along
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_size = min(len(self), batch_size)
        assert batch_size > 0, "List of InputDomain is empty; pop failed."
        if device is None:
            device = self.output_device
        lb = self.lb.pop(batch_size).to(device=device, non_blocking=True)
        dm_l = self.dm_l.pop(batch_size).to(device=device, non_blocking=True)
        dm_u = self.dm_u.pop(batch_size).to(device=device, non_blocking=True)
        alpha = {}
        if self.use_alpha:
            for key0, val0 in self.alpha.items():
                alpha[key0] = {}
                for key1, val1 in val0.items():
                    alpha[key0][key1] = val1.pop(batch_size).to(device=device, dtype=lb.dtype, non_blocking=True)
        cs = self.cs.pop(batch_size).to(device=device, non_blocking=True)
        threshold = self.threshold.pop(batch_size).to(device=device, non_blocking=True)
        constraints = None
        if self.constraint_A is not None or self.constraint_b is not None:
            constraint_A = self.constraint_A.pop(batch_size).to(device, non_blocking=True)
            constraint_b = self.constraint_b.pop(batch_size).to(device, non_blocking=True)
            constraints = (constraint_A, constraint_b)
        spec_sizes = torch.tensor(cs.shape[1], device=device).repeat(batch_size)

        if self.use_split_idx:
            split_idx = self.split_idx.pop(batch_size).to(device=device, non_blocking=True)
        else:
            split_idx = None

        self._add_volume(dm_l, dm_u, sign=-1)
        return alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx

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

    def get_topk_indices(self, k=1, largest=False, return_margin=False):
        assert k <= len(self), print("Asked indices more than domain length.")
        lb = self.lb._storage[: self.lb.num_used]
        threshold = self.threshold._storage[: self.threshold.num_used]
        margins, indices = self._get_sort_margin(lb - threshold).topk(k, largest=largest)
        if return_margin:
            return indices, margins
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
        
        if self.constraint_A is not None:
            self.constraint_A._storage[: self.constraint_A.num_used] = self.constraint_A._storage[indices]
        if self.constraint_b is not None:
            self.constraint_b._storage[: self.constraint_b.num_used] = self.constraint_b._storage[indices]
        
        
        if self.use_split_idx:
            self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]

    def report_memory(self):
        def _report_memory(attr_name, allocated_in_MB, used_in_MB):
            print(f"[{attr_name}] allocated: {allocated_in_MB:.2f} MB, used: {used_in_MB:.2f} MB")

        _report_memory("lb", *self.lb.calculate_memory())
        _report_memory("dm_l", *self.dm_l.calculate_memory())
        _report_memory("dm_u", *self.dm_u.calculate_memory())
        if self.use_alpha:
            alpha_allocated_in_MB, alpha_used_in_MB = 0, 0
            for val0 in self.alpha.values():
                for val1 in val0.values():
                    curr_alpha_allocated, curr_alpha_used = val1.calculate_memory()
                    alpha_allocated_in_MB += curr_alpha_allocated
                    alpha_used_in_MB += curr_alpha_used
            _report_memory("alpha", alpha_allocated_in_MB, alpha_used_in_MB)
        _report_memory("cs", *self.cs.calculate_memory())
        _report_memory("threshold", *self.threshold.calculate_memory())
        if self.use_split_idx:
            _report_memory("split_idx", *self.split_idx.calculate_memory())

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_used_MB = memory_info.rss / 1024 / 1024
        print(f"Total memory used: {memory_used_MB:.2f} MB")

        return


class UnsortedMultiSpecInputDomainList(InputDomainList):
    """Unsorted domain list for input split with arbitrary specifications."""

    def __init__(
        self,
        or_spec_size: Tensor,
        input_shape: Tuple,
        output_dim: int,
        storage_depth: int,
        output_device: str,
        use_alpha: bool = False,
        sort_index: Optional[int] = None,
        sort_descending: bool = True,
        use_split_idx: bool = True,
        alpha_final_name: Optional[str] = None,
    ):
        """

        The initialization method for the UnsortedMultiSpecInputDomainList class.

        :param or_spec_size:        A tensor containing the unique specification dimensions we may support
        :param input_shape:         The shape of the input to the network
        :param output_dim:          The output dimension of the network before applying C
        :param storage_depth:       The maximum number of splits we could use in input BaB
        :param output_device:       The default device storing accessed data. Complete domain data is always stored on 'cpu'.
        :param use_alpha:           True if we must also store alpha parameters
        :param sort_descending:     If True, domains will get sorted in descending order w.r.t. their lower bounds
                                    whenever the 'sort' method is called.
        :param use_split_idx:       If True, we also store the split indices for each domain
        :param alpha_final_name:    The name of the final node in LiRPANet.
                                    alpha of the final node has the spec_size.
        """
        super().__init__()

        self.output_device = output_device
        or_spec_size = or_spec_size.unique()
        # we store the spec with largest number of ANDs first
        self.or_spec_size = or_spec_size.to(output_device).sort(descending=True).values

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.use_alpha = use_alpha
        self.use_split_idx = use_split_idx
        self.storage_depth = storage_depth
        if use_alpha:
            assert alpha_final_name is not None, "Expected to receive alpha_final_name"
            self.alpha_final_name = alpha_final_name

        # Initialize all input domain lists
        unsorted_input_domain_list_args = {
            'output_device': output_device,
            'storage_depth': storage_depth,
            'use_alpha': use_alpha,
            'sort_index': sort_index,
            'sort_descending': sort_descending,
            'use_split_idx': use_split_idx,
        }
        spec_domain_lists = []
        for _ in self.or_spec_size:
            spec_domain_lists.append(UnsortedInputDomainList(**unsorted_input_domain_list_args))

        self.spec_domain_lists: List[UnsortedInputDomainList] = spec_domain_lists

    def pick_out_batch(self, batch_size, device=None):
        """
        Pick out a batch of subdomains from the domain list. Zero-padding is applied when necessary.

        :param batch_size:      The maximum number of domains we should pick out
        :param device:          The device all Tensors should be sent to

        :return alphas:         If supported, contains alpha parameters for the batch
        :return lb:             Output lower bounds
        :return dm_l:           Domain input lower bounds
        :return dm_u:           Domain input upper bounds
        :return cs:             Specification matrices
        :return threshold:      Thresholds
        :return spec_sizes:     The number of ANDs in the spec of each domain
        :return split_idx:      If supported, the input dimensions we should split along
        """
        if device is None:
            device = self.output_device

        # determine which domain lists we should pick out from and how many domains we should pick out.
        pickout_dl_idx, pickout_num_per_dl = self._get_pickout_decision(batch_size)

        # if we only need to pick out from one domain list, we can directly return the pickout 
        if len(pickout_dl_idx) == 1:
            curr_dl = self.spec_domain_lists[pickout_dl_idx[0]]
            return curr_dl.pick_out_batch(pickout_num_per_dl[0], device=device)

        tmp_alpha, lb, dm_l, dm_u, cs, threshold, split_idx, spec_sizes = [], [], [], [], [], [], [], []
        if not self.use_split_idx:
            split_idx = None
        for i, dl_idx in enumerate(pickout_dl_idx):
            curr_dl = self.spec_domain_lists[dl_idx]
            curr_pickout_num = pickout_num_per_dl[i]
            curr_alpha, curr_lb, curr_dm_l, curr_dm_u, curr_cs, curr_threshold, curr_constraints, curr_spec_sizes, curr_split_idx = \
                curr_dl.pick_out_batch(curr_pickout_num, device=device)
            if self.use_alpha:
                tmp_alpha.append(curr_alpha)
            lb.append(curr_lb)
            dm_l.append(curr_dm_l)
            dm_u.append(curr_dm_u)
            cs.append(curr_cs)
            threshold.append(curr_threshold)
            assert curr_constraints is None, "Constraints are not supported in UnsortedMultiSpecInputDomainList"
            if self.use_split_idx:
                split_idx.append(curr_split_idx)

            spec_sizes.append(curr_spec_sizes)
        
        # concatenate the lists to tensors
        dm_l = torch.cat(dm_l, dim=0)
        dm_u = torch.cat(dm_u, dim=0)
        spec_sizes = torch.cat(spec_sizes, dim=0)
        if self.use_split_idx:
            split_idx = torch.cat(split_idx, dim=0)

        # zero-pad the lists to tensors
        lb = pad_list_of_input_to_tensor(
                lb, pad_value=float('-inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )
        cs = pad_list_of_input_to_tensor(
                cs, pad_value=0, pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )
        threshold = pad_list_of_input_to_tensor(
                threshold, pad_value=float('inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=device
            )

        constraints = None

        alpha = {}
        if self.use_alpha:
            for key0 in tmp_alpha[0].keys():
                alpha[key0] = {}
                for key1 in tmp_alpha[0][key0].keys():
                    if key1 == self.alpha_final_name:
                        alpha[key0][key1] = pad_list_of_input_to_tensor(
                            [a[key0][key1] for a in tmp_alpha], pad_value=0, pad_dim=1, batch_dim=2,
                            is_orginal_tensor=True, device=device
                        )
                    else:
                        alpha[key0][key1] = torch.cat([a[key0][key1] for a in tmp_alpha], dim=2)

        return alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx

    def _get_pickout_decision(self, batch_size: int) -> Tuple[List[int], List[int]]:
        """
        Returns the indices of the input domain lists that we should pick out and corresponding
        number of domains we should pick out from each input domain list.

        :param batch_size:   The number of domains we wish to pick out in this iteration.
        :return:        A list of indices of the input domain lists we should pick from.
                        A list of the respective numbers of batches we should pickout from each input domain list.
        """
        # get the number of domains per domain list
        num_domains_per_dl = self._get_num_domains_per_dl()
        # mask of non-emtpy domain lists
        non_empty_mask = num_domains_per_dl > 0

        # If we do not have enough total subdomains to fill out this pickout batch,
        # we pick out from all domain lists that are not empty
        if num_domains_per_dl.sum() < batch_size:
            # indices of non-empty domain lists
            non_empty_dl_idx = torch.where(non_empty_mask)[0].tolist()
            pickout_num_per_dl = num_domains_per_dl[non_empty_mask].tolist()
            return non_empty_dl_idx, pickout_num_per_dl

        # If we have at least one input domain list that can fulfill the pickout batch,
        # we return the first one.
        individual_list_mask = num_domains_per_dl >= batch_size
        if individual_list_mask.any():
            return [torch.where(individual_list_mask)[0][0].item()], [batch_size]

        # If there are enough domains in total, while none of the individual domain lists
        # can fulfill the pickout batch, we pick out from multiple domain lists.
        # In this case, we pick out domain lists in the descending order of the number of ANDs.
        # Since we have sorted the domain lists in this way, we can simply pick out from the front.
        global_offsets_per_dl = self._get_global_offsets_per_dl()
        pickout_dl_mask = (global_offsets_per_dl < batch_size) & non_empty_mask
        pickout_dl_idx = torch.where(pickout_dl_mask)[0].tolist()
        pickout_num_per_dl = num_domains_per_dl[pickout_dl_mask]
        pickout_num_per_dl[-1] = batch_size - global_offsets_per_dl[pickout_dl_mask][-1]
        return pickout_dl_idx, pickout_num_per_dl.tolist()

    def get_progress(self):
        # TODO: Test this method
        progress_list = [d.get_progess() for d in self.spec_domain_lists]
        return (torch.sum(torch.tensor(progress_list, self.output_device)) / len(progress_list)).item()

    def add(self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: dict,
            cs: Tensor,
            threshold: Tensor,
            spec_sizes: Tensor,
            split_idx: Optional[Tensor] = None,
            remaining_index: Optional[Tensor] = None,
            check_dm_lbs: bool=True,
            check_input_boxes: bool=True,
            **kwargs
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains

        @param lb: Shape (batch, num_spec)                  Lower bound on domain outputs
        @param dm_l: Shape (batch, *input_shape)            Lower bound on domain inputs
        @param dm_u: Shape (batch, *input_shape)            Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param spec_sizes: Shape (batch,)                   The number of ANDs in the spec of each domain
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @param check_dm_lbs:                                If true, filters out domains that have been verified by lb > threshold
        @param check_input_boxes:                           If true, filters out domains that have been verified by dm_l > dm_u
        """
        batch = len(lb)
        if batch == 0:
            return

        # filter out the verified domains
        batch, lb, dm_l, dm_u, alpha, cs, threshold, _, _, _, split_idx, spec_sizes = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold,
                None, None, None, split_idx, spec_sizes, check_dm_lbs, check_input_boxes, remaining_index, self.use_alpha
            )

        for i, this_spec_size in enumerate(self.or_spec_size):
            this_dl_mask = spec_sizes == this_spec_size
            if this_dl_mask.any():
                domain_idx_in_this_dl = torch.where(this_dl_mask)[0]
                # get the data for this domain list
                _, curr_lb, curr_dm_l, curr_dm_u, curr_alpha, curr_cs, curr_threshold, _, _, _, curr_split_idx, _ = \
                    UnsortedInputDomainList.filter_verified_domains(
                        batch, lb, dm_l, dm_u, alpha, cs, threshold,
                        None, None, None, split_idx, None, False, False, domain_idx_in_this_dl, self.use_alpha)
                # add the data to the domain list
                # since we have filtered out the verified domains, we do not need to check them again.
                # slice to remove the potential padding and align the added data with that in domain list
                curr_lb = curr_lb[:, :this_spec_size]
                curr_cs = curr_cs[:, :this_spec_size]
                curr_threshold = curr_threshold[:, :this_spec_size]
                for key0 in curr_alpha.keys():
                    for key1 in curr_alpha[key0].keys():
                        if key1 == self.alpha_final_name:
                            curr_alpha[key0][key1] = curr_alpha[key0][key1][:, :this_spec_size]
                self.spec_domain_lists[i].add(
                    curr_lb, curr_dm_l, curr_dm_u, curr_alpha, curr_cs, curr_threshold, None, curr_split_idx, None, False, False)


    def sort(self):
        """
        Sorts all non-empty input domain lists
        """
        for d in self.spec_domain_lists:
            if len(d) > 0:
                d.sort()

    def __len__(self) -> int:
        """
        :return: Total domains among all input domain lists.
        """
        return self._get_num_domains_per_dl().sum().item()

    def __getitem__(self, idx):
        output_device = self.output_device
        # convert idx to tensor on the specified device for slicing.
        # it is for efficient inner computation only, when calling domain list's __getitem__,
        # we transfer the idx to cpu (the device of the domain list).
        if isinstance(idx, slice):
            idx = torch.arange(len(self), device=output_device)[idx]
        else:
            idx = torch.as_tensor(idx, device=output_device).view(-1)
        assert idx.shape[0] > 0, "Empty index"
        global_offsets_per_dl = self._get_global_offsets_per_dl()
        spec_idx = torch.searchsorted(global_offsets_per_dl, idx, right=True) - 1
        inner_idx = idx - global_offsets_per_dl[spec_idx]

        # collect data from every domain list
        lb, dm_l, dm_u, cs, threshold = [], [], [], [], []
        # record the number of ANDs and number of domains for every domain list we visit
        spec_sizes, num_domains_per_dl = [], []
        # since we access the domain lists in order, instead of the order of idx,
        # we need to keep track of the order of elements we get in the orginal idx.
        order_in_original_idx = []
        for i, domain_list in enumerate(self.spec_domain_lists):
            mask = spec_idx == i
            if mask.any():
                curr_idx = mask.nonzero(as_tuple=True)[0]
                spec_sizes.append(self.or_spec_size[i].item())
                num_domains_per_dl.append(curr_idx.shape[0])
                order_in_original_idx.append(curr_idx)
                # TensorStorage is on CPU, so we need to move the idx to CPU if needed
                curr_lb, curr_dm_l, curr_dm_u, curr_cs, curr_threshold, _ = domain_list[
                    inner_idx[mask].cpu()
                ]
                lb.append(curr_lb)
                dm_l.append(curr_dm_l)
                dm_u.append(curr_dm_u)
                cs.append(curr_cs)
                threshold.append(curr_threshold)

        order_in_original_idx = torch.cat(order_in_original_idx, dim=0)
        inverted_order = torch.argsort(order_in_original_idx)

        # batch the tensors and adjust the order
        # [batch, *input_shape]
        dm_l = torch.cat(dm_l, dim=0)[inverted_order]
        dm_u = torch.cat(dm_u, dim=0)[inverted_order]

        # lb, cs, threshold have the dim of num_and
        # so we need to pad them if domains are from different domain lists with different num_and.
        if (spec_idx == spec_idx[0]).all():
            # if all domains are from the same domain list, just concatenate them
            lb = torch.cat(lb, dim=0)[inverted_order]
            cs = torch.cat(cs, dim=0)[inverted_order]
            threshold = torch.cat(threshold, dim=0)[inverted_order]
        else:
            lb = pad_list_of_input_to_tensor(
                lb, pad_value=float('-inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]
            cs = pad_list_of_input_to_tensor(
                cs, pad_value=0, pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]
            threshold = pad_list_of_input_to_tensor(
                threshold, pad_value=float('inf'), pad_dim=1, batch_dim=0, is_orginal_tensor=True, device=output_device
            )[inverted_order]

        # indicate the number of ANDs for every domain
        # it can be used to remove the padding if needed
        spec_sizes = torch.repeat_interleave(
            torch.tensor(spec_sizes, device=output_device),
            torch.tensor(num_domains_per_dl, device=output_device),
        )[inverted_order]

        return lb, dm_l, dm_u, cs, threshold, spec_sizes

    def get_topk_indices(
        self, k: int = 1, largest: bool = False, return_margin: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        This function returns the indices (w/ or w/o margins)of top k domains
        with worst(smallest) / best(largest) decision margins among all domain lists.

        :param k:                       The indices for the top k worst lower bounds
        :param largest:                 If False, returns the top k worst margins. Otherwise, returns the top k best margins.
        :param return_margin:           If True, returns the margins along with the indices.
        :return global_topk_indices:    The global top k indices and margins (if return_margin is True)
        """
        assert k <= len(self), print("Asked indices more than domain length.")
        output_device = self.output_device
        # get the top k margins and indices for each domain list
        local_topk_inner_indices, local_topk_margins, local_topk_num = [], [], []
        for i, domain_list in enumerate(self.spec_domain_lists):
            if len(domain_list) > 0:
                curr_indices, curr_margins = domain_list.get_topk_indices(
                    min(k, len(domain_list)), largest, return_margin=True
                )
                local_topk_inner_indices.append(curr_indices)
                local_topk_margins.append(curr_margins)
                local_topk_num.append(curr_indices.shape[0])

        # local_topk_inner_indices: inner indices of the top k margins for each domain list
        # local_topk_margins: top k margins for each domain list
        # local_topk_dl_indices: dl indices of the top k margins for each domain list
        local_topk_inner_indices = torch.cat(local_topk_inner_indices, dim=0).to(output_device)
        local_topk_margins = torch.cat(local_topk_margins, dim=0).to(output_device)
        local_topk_dl_indices = torch.repeat_interleave(
            torch.arange(len(local_topk_num), device=output_device),
            torch.tensor(local_topk_num, device=output_device),
        )

        # get top k margins and indices among top k margins of all domain lists
        sorted_topk_margins, sorted_topk_indices = local_topk_margins.topk(k, largest=largest)

        topk_indices_inner = local_topk_inner_indices[sorted_topk_indices]
        topk_indices_dl = local_topk_dl_indices[sorted_topk_indices]

        global_offsets_per_dl = self._get_global_offsets_per_dl()
        global_topk_indices = global_offsets_per_dl[topk_indices_dl] + topk_indices_inner

        if return_margin:
            return global_topk_indices, sorted_topk_margins
        return global_topk_indices

    def _get_num_domains_per_dl(self):
        """
        Returns the number of domains in each input domain list.
        """
        return torch.tensor([len(d) for d in self.spec_domain_lists], device=self.output_device)

    def _get_global_offsets_per_dl(self):
        """
        Returns the cumulative sum of the number of domains in each input domain list.
        """
        num_domains_per_dl = torch.cat([torch.tensor([0], device=self.output_device), self._get_num_domains_per_dl()[:-1]])
        return num_domains_per_dl.cumsum(0)
