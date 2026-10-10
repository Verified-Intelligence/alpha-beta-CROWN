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
Definitions of return values for update_bounds_pre. 
"""
import torch
from typing import Any, List, Dict
from dataclasses import dataclass

from auto_LiRPA import BoundedTensor
from heuristics.decision_types import (
    BranchingDecisions,
    BatchFirstBranchingDecisions,
)
from domain_clipper import (
    ClipDecisions,
    SubDomainClipDecisions,
    DomainClipScorer,
)
from cuts.cplex_cut_recorder import CplexCutRecorder
from state.intermediate_bounds import WorkingIntermBoundsInfo
from state import (
    AlphaValueData,
    NumsEffectiveBetasPerDomain,
    BetaFullData,
    BetaValues,
    BatchedlA,
)


@dataclass
class UpdateBoundPreReturn:

    interm_bounds: dict[str, List[torch.Tensor]]
    "intermediate lower/upper bounds used for compute_bounds"
    lb_last: torch.Tensor
    "A batch of final layer (after c) lower bounds from last iteration, used to determine progress"
    ub_last: torch.Tensor
    "A batch of final layer (after c) upper bounds from last iteration, used to determine progress."
    c: torch.Tensor | None
    "A batch of c matrix to transform final layer bounds to the property bounds."
    new_x: torch.Tensor | BoundedTensor
    "A batch of the input x for this iteration"
    x_Ls: torch.Tensor | None
    "Currently discarded, always None. The input lower bounds of x."
    x_Us: torch.Tensor | None
    "Currently discarded, always None. The input upper bounds of x."
    new_x_Ls: None
    "Uesless, always None."
    new_x_Us: None
    "Uesless, always None."
    nums_effective_beta_per_domain: ValueError | NumsEffectiveBetasPerDomain
    "The numbers of effective betas for each domain."
    alphas_by_layer: ValueError | AlphaValueData
    "A batch of alpha values for each layer, where batch dimension inside tensor."
    betas_by_layer: ValueError | BetaFullData
    "A batch of beta values for each layer, where batch dimension inside tensor."
    beta_bias: bool
    "Whether there is a bias term in beta data."
    clip_decisions: ClipDecisions
    "A batch of top-k objectives that domain clip perform on for each domain"
    d_dict: dict
    "The original data from domainlist."
    batch_size: int
    "The number of domains in this batch"


@dataclass
class UpdateBoundCoreReturn:
    lb: torch.Tensor
    "A batch of final layer (after c) lower bounds for this iteration."
    ub: torch.Tensor
    "A batch of final layer (after c) upper bounds for this iteration."
    lb_last: torch.Tensor
    "A batch of final layer (after c) lower bounds from last iteration, used to determine progress"
    ub_last: torch.Tensor
    "A batch of final layer (after c) upper bounds from last iteration, used to determine progress."
    nums_effective_beta_per_domain: ValueError | NumsEffectiveBetasPerDomain
    "The numbers of effective betas for each domain."
    input_split_idx: None  # uesless
    "Currently discarded, always None."
    primal_x: None  # uesless
    "Currently discarded, always None."
    x_Ls: None  # uesless
    "Currently discarded, always None."
    x_Us: None  # uesless
    "Currently discarded, always None."
    new_x_Ls: None | torch.Tensor
    "Currently discarded, always None."
    new_x_Us: None | torch.Tensor
    "Currently discarded, always None."
    c: torch.Tensor
    "A batch of c matrix to transform final layer bounds to the property bounds."
    working_beta: BetaFullData | ValueError
    "Beta values extracted from net directly"
    working_alpha: AlphaValueData | ValueError
    "Alpha values extracted from net directly"
    working_interm_bounds: WorkingIntermBoundsInfo
    "Intermediate bounds extracted from net directly"
    batched_lA: BatchedlA
    "lA values extracted from net"
    branching_decision: BranchingDecisions
    """A batch of precomputed branching decisions in split-depth-first format, 
    extracted from branching_heuristic directly."""
    sub_domain_clip_decisions: SubDomainClipDecisions
    """A batch of precomputed top-k objectives for each domain"""
    decision_thresh: torch.Tensor
    """A batch of thresholds to determine whether a domain is verified"""
    lb_final_max: float
    """The maximum final layer lower bound among the domains in this batch. Used for logging"""
    lb_final_min: float
    """The minimum final layer lower bound among the domains in this batch. Used for logging"""
    n_verified: int
    """The number of verified domains in this batch. Used for logging"""
    n_splits: int
    """The number of splits performed in this batch. Used for logging"""
    new_split_history: List[Dict]
    """A batch of split history used when cuts are enabled."""
    history: list
    "A batch of (act split) history representing a batch of domains"
    depths: torch.Tensor
    "A batch of depths of each domain"
    thresholds: torch.Tensor
    "A batch of thresholds for each domain"


@dataclass
class UpdateBoundPostReturn:
    lower_bounds: Dict[str, torch.Tensor]
    "A batch of lower bounds for the final layer, currently only contains the final layer bounds."
    upper_bounds: Dict[str, torch.Tensor]
    "A batch of upper bounds for the final layer, currently only contains the final layer bounds."
    lAs: BatchedlA
    "A batch of lA values for each layer."
    alphas: AlphaValueData
    "A batch of alpha values for each layer."
    betas: BetaValues
    "A batch of beta values for each domain."
    split_history: List[Dict]
    "A batch of split history used when cuts are enabled."
    unstable_bounds: Dict[str, Any]
    "A batch of unstable bounds for each layer."
    primals: None  # useless
    "Currently discarded, always None."
    c: torch.Tensor
    "A batch of c matrix to transform final layer bounds to the property bounds."
    x_Ls: None  # uesless
    "Currently discarded, always None."
    x_Us: None  # useless
    "Currently discarded, always None."
    input_split_idx: None  # useless
    "Currently discarded, always None."
    decision_info: BatchFirstBranchingDecisions
    "A batch of split decisions for each domain, Domain-first memory"
    sub_domain_clip_decisions: SubDomainClipDecisions
    """A batch of precomputed top-k objectives for each domain"""


@dataclass(slots=True)
class UpdateBoundCoreContext:
    nums_effective_beta_per_domain: ValueError | NumsEffectiveBetasPerDomain
    "A batch of numbers of effective betas for each domain."
    temp_return_A: bool
    "Flag to return intermediate A matrices."
    lb: torch.Tensor
    "A batch of final layer (after c) lower bounds for this iteration."
    ub: torch.Tensor
    "A batch of final layer (after c) upper bounds for this iteration."
    lb_last: torch.Tensor
    "A batch of final layer (after c) lower bounds from the previous iteration."
    ub_last: torch.Tensor
    "A batch of final layer (after c) upper bounds from the previous iteration."
    primal_x: None  # uesless. branching_input_and_activation is always False
    "Currently discarded, always None."
    input_split_idx: None  # uesless branching_input_and_activation is always False
    "Currently discarded, always None."
    new_x_Ls: None | torch.Tensor
    "A batch of new lower bounds for the input."
    new_x_Us: None | torch.Tensor
    "A batch of new upper bounds for the input."
    c: torch.Tensor
    "A batch of c matrix to transform final layer bounds to the property bounds."
    original_size: int
    "The original size of this batch before splitting."
    x_Ls: None  # uesless
    "Currently discarded, always None."
    x_Us: None  # useless
    "Currently discarded, always None."
    decision_thresh: torch.Tensor
    "A batch of thresholds to determine whether a domain is verified"
    domain_clip_scorer: DomainClipScorer | None
    "A scorer for domain clipping."
    branching_heuristic: Any
    "The branching heuristic used for split decision finding"
    batch_device_limit: int
    "The memory limited batch size for branching decision precomputation."
    precompute_bfs_flag: bool
    "Flag to mark this iteration is multitree BFS iteration"
    is_multitree_bab: bool
    "Flag to mark this iteration is in the multitree Shallow BAB process"
    recorder: CplexCutRecorder | None
    "Recorder for recording cplex cuts."
