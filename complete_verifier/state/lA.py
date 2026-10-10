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
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
import torch
from torch import Tensor
from state.traits_mixin import DictLikeMixIn
from state.utils import EmptiedTensor
from auto_LiRPA.utils import transfer
import arguments

if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


@dataclass
class BatchedlA(DictLikeMixIn):
    """
    A batch of lA values.

    Format: {node_name -> tensor(batch, or_spec_size, *shape[1:])}
    """

    _data: dict[str, torch.Tensor]
    is_emptied: bool = False

    def inplace_mask_batch_dim(self, mask: torch.Tensor):
        """Inplace mask the batch dimension."""
        for layer_name in self:
            self[layer_name] = self[layer_name][mask, ...]
        return self

    @staticmethod
    def from_net(
        net: "LiRPANet",
        *,
        preserve_mask=None,
        tot_cells=None,
        device=None,
        move=False,
    ) -> "BatchedlA":
        """
        Extract lA values from the network in BatchedlA format.

        Args:
            net: The LiRPANet to extract from.
            preserve_mask: Optional mask for preserving certain cells.
            tot_cells: Total number of cells when using preserve_mask.
            device: Target device for the tensors.
        Returns:
            A BatchedlA instance.
        """
        lAs = {}

        if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
            # lA of the input layer is needed for input bab.
            nodes = [net.net[net.net.input_name[0]]]
        else:
            nodes = list(net.net.get_splittable_activations())

        for node in nodes:
            lA = getattr(node, "lA", None)
            if lA is None:
                continue
            if preserve_mask is not None:
                new_lA = torch.zeros(
                    [tot_cells, lA.shape[0]] + list(lA.shape[2:]),
                    dtype=lA.dtype,
                    device=lA.device,
                )
                new_lA[preserve_mask] = lA.transpose(0, 1)
                lA = new_lA
            else:
                # DomainlA format: (batch, 1, ...)
                lA = lA.transpose(0, 1)
            lAs[node.name] = transfer(lA, device)
            # if move:
            #     node.lA = EmptiedTensor("lA tensor has been moved to BatchedlA.")
        return BatchedlA(_data=lAs)

    @staticmethod
    def gc_lA_from_net(net: "LiRPANet"):
        """
        Garbage Collect lAs in the net.

        Args:
            net: The LiRPANet to extract from
        """
        nodes = list(net.net.nodes())
        for node in nodes:
            lA = getattr(node, "lA", None)
            if lA is None:
                continue
            node.lA = EmptiedTensor("lA tensor has been emptied.")

    def to_device(self, device):
        """Transfer all lA tensors to the specified device."""
        for layer_name in self:
            self[layer_name] = transfer(self[layer_name], device)
        return self
