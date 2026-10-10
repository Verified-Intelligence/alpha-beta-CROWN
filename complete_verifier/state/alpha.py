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
AlphaValueData: A batch of Alpha **values**, mainly used in BaB interations.
AlphaFullInfoData: A batch of Alpha **values and related auxilaries**, used during .build() stage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from state.traits_mixin import DictLikeMixIn
from utils import transfer_obj

if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet

AlphaDataStartingNodeScope = Literal["part", "all"]
"""
* part: For each layer, starting_nodes=alpha_start_nodes (final node + --optimized_interm)
* all: For each layer, starting_nodes=all that present in the network
"""


def resolve_alpha_storage_dtype(
    configured_dtype: str,
    *,
    reproduce_alpha_bounds: bool,
) -> torch.dtype | None:
    """Resolve the dtype used for alpha tensors stored between BaB updates.

    Args:
        configured_dtype: Value of the ``alpha_dtype`` verifier option.
        reproduce_alpha_bounds: Whether stored alpha must reproduce the bound
            that produced it without a lossy dtype conversion.

    Returns:
        ``torch.float16`` when compact storage is enabled, or ``None`` to
        preserve the source tensor dtype.
    """
    if configured_dtype == "float16" and not reproduce_alpha_bounds:
        return torch.float16
    return None


@dataclass
class AlphaValueData(DictLikeMixIn):
    """
    A batch of Alpha **values**, mainly used in BaB interations.

    Format
    {activation_layer -> {
                final_layer -> tensor(2,1,batch,fixed_number),
                ... -> ...
            }
        ...
    }
    """

    _data: dict

    @staticmethod
    def from_net_part(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network with part scope (alpha_start_nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        net.alpha_drop_unused()
        alpha_start_nodes = net.alpha_start_nodes
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            raw_alpha_by_layer[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                if spec_name in alpha_start_nodes:
                    raw_alpha_by_layer[m.name][spec_name] = alpha.detach()
                    if move:
                        m.alpha[spec_name] = ValueError(
                            "Alpha has been popped and is no longer available "
                            "before assigning it again."
                        )

        return AlphaValueData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net_full(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network with full scope (all nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            raw_alpha_by_layer[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                raw_alpha_by_layer[m.name][spec_name] = alpha.detach()
                if move:
                    m.alpha[spec_name] = ValueError(
                        "Alpha has been popped and is no longer available "
                        "before assigning it again."
                    )

        return AlphaValueData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net(
        net: "LiRPANet",
        *,
        starting_node_scope: Literal["part", "all"],
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network.
        Args:
            net: The network to extract from.
            starting_node_scope: The scope of starting nodes to extract. Additionally, if "part", call net.alpha_drop_unused() first.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        if starting_node_scope == "part":
            return AlphaValueData.from_net_part(net, move=move)
        elif starting_node_scope == "all":
            return AlphaValueData.from_net_full(net, move=move)
        else:
            raise ValueError(f"Unknown starting_node_scope: {starting_node_scope}")

    @staticmethod
    def from_domain_dict(d: dict) -> "AlphaValueData":
        """Create AlphaValueData from a domain dictionary from pick_out."""
        assert "alphas" in d
        return AlphaValueData(_data=d["alphas"])

    def attach_to_net(self, net: "LiRPANet"):
        """
        Attach all stored alpha values to the network.

        Args:
            net: The network to attach to.
        """

        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            layer_alpha = self._data.get(m.name)
            if not layer_alpha:
                continue
            if m.output_dtype is None:
                raise RuntimeError(
                    f'Cannot restore alpha for {m.name} without its '
                    'graph output dtype.')
            for spec_name, alpha in layer_alpha.items():
                # Domain storage may reduce alpha precision; optimization
                # resumes with the dtype owned by the computation graph.
                m.alpha[spec_name] = alpha.to(
                    device=net.device, dtype=m.output_dtype
                ).detach().requires_grad_(True)

    def clone_tensor(self) -> "AlphaValueData":
        """Clone all tensors in the data structure."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha.clone() for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")

        return new_dict

    def mask_batch_dim_inplace(self, batch_mask) -> "AlphaValueData":
        """Inplace mask the batch dimension."""
        for _, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                for spec_name, alpha in layer_alpha.items():
                    layer_alpha[spec_name] = alpha[:, :, batch_mask, ...]
            else:
                raise ValueError("Unexpected alpha structure.")
        return self

    def mask_batch_dim(self, batch_mask) -> "AlphaValueData":
        """Return a new AlphaValueData with the batch dimension masked."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha[:, :, batch_mask, ...]
                    for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict

    def to(self, *, device=None, dtype=None, non_blocking=False) -> "AlphaValueData":
        """Return a new AlphaValueData with tensors moved to the specified device and/or dtype."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha.to(
                        device=device, dtype=dtype, non_blocking=non_blocking
                    )
                    for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict


@dataclass
class AlphaFullInfoData(DictLikeMixIn):
    """
    A batch of Alpha **values and related auxilaries**, used during .build() stage.

    For each layer, store its full information. Used during initialization stage.
    (e.g. layer.alpha/layer.alpha_lookup_idx/layer.alpha_indices/layer.init)

    Format:
    {activation_layer -> {
                final_layer -> {alpha-> tensor(2,1,batch,fixed_number), alpha_lookup_idx -> ..., ... -> ...}
                ... -> ...
            }
        ...
    }
    """

    _data: dict

    @staticmethod
    def from_net_part(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network with part scope (alpha_start_nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        net.alpha_drop_unused()
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            if not move:
                raw_alpha_by_layer[m.name] = m.dump_alpha()
            else:
                raw_alpha_by_layer[m.name] = m.pop_alpha()

        return AlphaFullInfoData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net_full(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network with full scope (all nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            if not move:
                raw_alpha_by_layer[m.name] = m.dump_alpha()
            else:
                raw_alpha_by_layer[m.name] = m.pop_alpha()

        return AlphaFullInfoData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net(
        net: "LiRPANet",
        *,
        starting_node_scope: Literal["part", "all"],
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network, including alpha tensor and other related info.
        Args:
            net: The network to extract from.
            starting_node_scope: The scope of starting nodes to extract. Additionally, if "part", call net.alpha_drop_unused() first.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        if starting_node_scope == "part":
            return AlphaFullInfoData.from_net_part(net, move=move)
        elif starting_node_scope == "all":
            return AlphaFullInfoData.from_net_full(net, move=move)
        else:
            raise ValueError(f"Unknown starting_node_scope: {starting_node_scope}")

    def to(self, *, device=None, dtype=None, non_blocking=False) -> "AlphaFullInfoData":
        """Return a new AlphaFullInfoData with tensors moved to the specified device and/or dtype."""
        new_dict = AlphaFullInfoData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    item_name: transfer_obj(
                        item,
                        device=device,
                        dtype=dtype if item_name == 'alpha' else None,
                        non_blocking=non_blocking,
                    )
                    for item_name, item in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict
