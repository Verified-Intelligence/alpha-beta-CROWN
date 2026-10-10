#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""
BetaFullData: A batch of full beta information in SparseBeta formats that is assignable
    and extractable from BoundedModule without reorganization. Used to hold beta info for fast
    extraction and assign in update_bounds_pre and update_bounds_core.
NumsEffectiveBetasPerDomain: A batch of the numbers of effective beta values for each domain,
    since the dimension of each layer's beta is determined by the largest one among all domains.
BetaValues: A batch of beta values (only), used to store the beta values to DomainList.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Tuple
import torch
from state.traits_mixin import DictLikeMixIn, ListLikeMixIn
from auto_LiRPA.beta_crown import SparseBeta
import arguments

if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


@dataclass
class BetaFullData(DictLikeMixIn):
    """
    A batch of full beta information in SparseBeta formats that is assignable
    and extractable from BoundedModule without reorganization.

    For each layer, store its sparse beta object (layer.sparse_betas). Mainly used during BaB process.

    Format:
    {
        splitable_layer_name -> [SparseBeta,]
    }
    Though the inner list contains only one SparseBeta object currently.

    """

    _data: dict[str, list[SparseBeta]]

    @staticmethod
    def from_net(net: "LiRPANet", beta_layer_names: list) -> "BetaFullData":
        """
        Extract beta values from the network.

        Args:
            net: The network to extract from.
            beta_layer_names: The names of layers that have beta values.
        """
        if isinstance(beta_layer_names, ValueError):
            raise ValueError("beta_layer_names should be a list of layer names.")

        beta_data = {}
        for layer_name in beta_layer_names:
            layer = net.net[layer_name]
            beta_data[layer_name] = layer.sparse_betas

        return BetaFullData(beta_data)

    @staticmethod
    def from_domain_dict(
        d: dict, *, bias: bool, dtype_map: dict[str, torch.dtype], device="cpu"
    ) -> Tuple["BetaFullData", "NumsEffectiveBetasPerDomain"]:
        """
        Extract beta values from a domain's beta dict.

        Args:
            d: The domain's beta dict. Required keys: "history", "betas".
            bias: If true, turn on bias in SparseBeta.
            dtype_map: Computation dtype owned by each split node name.
            device: The device to put the tensors on.
        Returns:
            A BetaFullData instance containing the extracted beta values, and a NumsEffectiveBetasPerDomain instance.
        """

        assert "history" in d and "betas" in d

        batch_size = len(d["history"])
        nums_effective_beta_per_domain, max_splits_per_layer = NumsEffectiveBetasPerDomain.from_domain_dict(
            d
        )

        assert (
            arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
            is False
        )

        sparse_beta_by_layer = {}
        for layer_name, max_splits in max_splits_per_layer.items():
            try:
                dtype = dtype_map[layer_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Cannot restore beta for {layer_name} without its "
                    "computation dtype."
                ) from exc
            shape = (batch_size, max_splits)
            betas_ = [
                (d["betas"][bi][layer_name] if d["betas"][bi] is not None else None)
                for bi in range(batch_size)
            ]
            sparse_beta_by_layer[layer_name] = [
                SparseBeta(
                    shape=shape,
                    bias=bias,
                    betas=betas_,
                    device=device,
                    dtype=dtype,
                )
            ]
            for sparse_beta in sparse_beta_by_layer[layer_name]:
                sparse_beta.apply_splits(d["history"], layer_name)

        return BetaFullData(sparse_beta_by_layer), nums_effective_beta_per_domain

    def attach_to_net(self, net: "LiRPANet"):
        """Attach the beta values to the network."""
        net.net.accept_beta(self)

    def to(self, *, device=None, dtype=None, non_blocking=False) -> "BetaFullData":
        """Move all tensors in the data structure to the specified device and dtype."""
        new_data = {}
        for k, v in self._data.items():
            new_data[k] = [
                sb.to(device=device, dtype=dtype, non_blocking=non_blocking) for sb in v
            ]
        return BetaFullData(new_data)

    def mask_batch_dim_inplace(self, mask: torch.Tensor) -> "BetaFullData":
        """Mask the batch dimension of all SparseBeta objects in place."""
        for layer_name, sparse_betas in self._data.items():
            for sb in sparse_betas:
                sb.mask_batch_dim_inplace(mask)
        return self

    def to_domain_dict(
        self, nums_effective_beta_per_domain: NumsEffectiveBetasPerDomain, device
    ) -> "BetaValues":
        """Convert itself to the format d['betas'] requires.
        [0~batch_size] -> {splitable_layer -> tensor(shape=(effective splits))}
        """

        assert (
            arguments.Config["solver"]["beta-crown"]["enable_opt_interm_bounds"]
            is False
        ), "enable_opt_interm_bounds is expected to be False all the time"

        ret = []

        for i in range(len(nums_effective_beta_per_domain)):
            betas = {}
            for k in nums_effective_beta_per_domain[i]:
                betas[k] = (
                    self._data[k][0]
                    .val[i, : nums_effective_beta_per_domain[i][k]]
                    .detach()
                    .to(device=device)
                )
            ret.append(betas)

        return BetaValues(ret)


@dataclass
class NumsEffectiveBetasPerDomain(ListLikeMixIn):
    """
    A batch of the numbers of effective beta values for each domain, 
    since the dimension of each layer's beta is determined by the
    largest one among all domains.

    Format: [0~batch_size] -> {splitable_layer_name -> num of splits}
    """

    _data: list

    def mask_batch_dim(self, mask):
        new_res = []
        for i in range(len(self._data)):
            if mask[i]:
                new_res.append(self._data[i])
        return NumsEffectiveBetasPerDomain(new_res)

    @staticmethod
    def from_domain_dict(d: dict) -> tuple["NumsEffectiveBetasPerDomain", dict]:
        """
        Extract split per example info from a domain's beta dict.

        Args:
            d: The domain's beta dict. Required key: "history".

        Returns:
            NumsEffectiveBetasPerDomain: A batch numbers of effetive betas in each domain and each layer.
            max_splits_per_layer: The maximum number of effective betas in each layer among all domains.
        """

        assert "history" in d

        nums_effective_beta_per_domain = []
        max_splits_per_layer = {}
        batch = len(d["history"])
        for bi in range(batch):
            nums_effective_beta_per_domain.append({})
            for k, v in d["history"][bi].items():
                # First element of layer_splits is a list of split neuron IDs.
                nums_effective_beta_per_domain[bi][k] = len(v[0])
                max_splits_per_layer[k] = max(
                    max_splits_per_layer.get(k, 0), nums_effective_beta_per_domain[bi][k]
                )

        return NumsEffectiveBetasPerDomain(nums_effective_beta_per_domain), max_splits_per_layer


@dataclass
class BetaValues(ListLikeMixIn):
    """A batch of beta values (only) for each domain, used to store in DomainList and domain dict often named in `d`.

    Format: [0~batch_size] -> {layer -> tensor(shape=(effective depth))} (values of beta)
    """

    _data: list
