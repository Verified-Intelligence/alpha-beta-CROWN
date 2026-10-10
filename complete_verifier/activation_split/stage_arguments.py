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
Arguments for branch_and_bound_preprocess/solve/postprocess.
"""
from dataclasses import dataclass

import torch
from torch import Tensor
from auto_LiRPA import BoundedTensor
from utils import Stats
from beta_CROWN_solver import LiRPANet
from domain_clipper import DomainClipScorer
from heuristics import BranchingHeuristicObj
from state import IntermBoundsFactory


@dataclass
class PreprocessConstArguments:
    """
    Constant arguments for the preprocess stage.
    """

    split_nodes_names: list[str]
    "A list mapping split_node index to split_node name. Used for logging."
    dtype_map: dict[str, torch.dtype]
    "The computation dtype owned by each split node."
    final_name: str
    "The name of the final layer in BoundedModule."
    net_x: Tensor | BoundedTensor
    "The original input x to the network."
    net_c: Tensor | None
    "The c tensor encoding the property."
    device: str
    "The device the preprocess is bound to."
    unstable_mask: dict[str, Tensor]
    "The mask for unstable activations in each layer"

    @staticmethod
    def from_net(net: LiRPANet) -> "PreprocessConstArguments":
        return PreprocessConstArguments(
            split_nodes_names=[n.name for n in net.net.split_nodes],
            dtype_map=net.get_split_node_dtype_map(),
            final_name=net.final_name,
            net_x=net.x,
            net_c=net.c,
            device=str(net.device),
            unstable_mask=net.unstable_mask
        )

@dataclass
class PreprocessMutableArguments:
    """
    Mutable arguments for the preprocess stage.
    """

    iter_idx: int
    "Number of iteration"
    device_batch_limit: int
    "The maximum batch size to use when running the network on the device."
    stats: Stats
    "Stats object to log time and other stats."
    domain_interm_factory: IntermBoundsFactory
    "The factory for batch interm bounds"


@dataclass
class SolveConstArguments:
    """
    Constants argument or objects for the solve stage.
    """

    domain_clip_scorer: None | DomainClipScorer
    branching_heuristic: BranchingHeuristicObj


@dataclass
class SolveMutableArguments:
    """
    Mutable arguments for the solve stage.
    """

    stats: Stats
    "Stats object to log time and other stats. "


@dataclass
class PostprocessConstArguments:
    """
    Constants argument or objects for the postprocess stage.
    """

    final_name: str
    split_nodes_names: list[str]
    layers_requiring_bounds_names: list[str]
    unstable_mask: dict
    interm_transfer: bool

    @staticmethod
    def from_net(net: LiRPANet) -> "PostprocessConstArguments":
        return PostprocessConstArguments(
            final_name=net.final_name,
            split_nodes_names=[n.name for n in net.net.split_nodes],
            layers_requiring_bounds_names=[n.name for n in net.net.layers_requiring_bounds],
            unstable_mask=net.unstable_mask,
            interm_transfer=net.interm_transfer,
        )


@dataclass
class PostprocessMutableArguments:
    """
    Mutable arguments for the postprocess stage.
    """

    iter_idx: int
    "Number of iteration"
    stats: Stats
    "Stats object to log time and other stats. "
