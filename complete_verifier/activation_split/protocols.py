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
Packets for the three stages of BaB:
branch_and_bound_preprocess/branch_and_bound_solve/branch_and_bound_postprocess.
Since stages are independent of each other,
any data that needs to pass between stages should EXPLICITLY live here.
"""
from dataclasses import dataclass
import torch
from typing import Literal
from activation_split.return_types import UpdateBoundPreReturn, UpdateBoundCoreReturn

PreprocessExitStatus = Literal[
    "normal", "error"
]

@dataclass
class PreprocessPacket:
    """All objects needed to pass from preprocess stage"""

    preResults: UpdateBoundPreReturn
    "Information regarding update_bounds"

    # book keeping infos
    iter_idx: int
    "Logical number of this iteration"
    num_visited_domains: int
    "The number of domains visited so far, in view of preprocess stage"
    device_batch_limit: int
    "Maximum batch size current device limit"
    exit_status: PreprocessExitStatus
    "Exit status"


CoreExitStatus = Literal[
    "all_node_split_LP_unsafe", "all_node_split_unknown", "normal_exit"
]


@dataclass
class SolvePacket:
    coreResults: UpdateBoundCoreReturn
    "All results from update_bounds_core needed for solve stage"

    # early exit info
    early_exit_status: CoreExitStatus

PostExitStatus = Literal["normal"]
@dataclass
class PostprocessPacket:
    exit_status: PostExitStatus
    "exit status of postprocess"
    global_lb: torch.Tensor
    "The global lower bound per or_spec"
