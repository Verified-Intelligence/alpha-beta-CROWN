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
Datastructures to hold a batch of split decisions.

- BranchingDecisions: SplitDepth-first memory layout for a batch of split decisions. 
- BatchFirstBranchingDecisions: Domain-first memory layout for a batch of split decisions.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class BranchingDecisions:
    """
    SplitDepth-first memory layout for a batch of split decisions.

        branching_decision[j + i * batch_size] = (layer, neuron) the j-th domain's decision at i-th split depth.
        branching_points[j + i * batch_size] = FP the j-th domain's branching point at i-th split depth.

    where j starts from 0 to (batch - 1), and i starts from 0 to split_depth - 1.

    It is the raw return format of branching_heuristics.
    """

    branching_decision: List
    # shape=(split_depth, batch_size, 2)
    branching_points: torch.Tensor | None
    split_depth: int
    batch_size: int

    @staticmethod
    def from_batch_first_decisions(
        domain_decision: "BatchFirstBranchingDecisions",
        strict: bool = True,
    ) -> "BranchingDecisions":
        """ 
        Convert a BatchFirstBranchingDecisions to a BranchingDecisions.
         If strict is True, raise an error when split depths in the batch are not the same.
         If strict is False, narrow to the minimum split depth when split depths in the batch are not the same.

        Split depths may vary due to `auto_batch_size` attempting to increase split depth when memory is not saturated.
        So DomainList end up holding different split depths for different domains.

        """
        split_depth = domain_decision.packed_branching_points_split_depth
        batch_size = domain_decision.batch_size

        split_depth_val: int = int(split_depth[0].item())
        if not (split_depth == split_depth_val).all():
            # get all distinct split depths
            distinct_split_depths = torch.unique(split_depth)
            histogram = torch.histc(
                split_depth.float(),
                bins=len(distinct_split_depths),
                min=distinct_split_depths.min().item(),
                max=distinct_split_depths.max().item(),
            )
            print(f"Distinct split depths: {distinct_split_depths}")
            print(f"Histogram of split depths: {histogram}")
            err_msg = (
                f"Split depths are not the same in BranchingDecisions."
                f"{distinct_split_depths=}. {histogram=}\n"
                "narrowing to the minimum split depth."
            )
            print(err_msg)
            if strict:
                raise ValueError(err_msg)
            split_depth_val = int(distinct_split_depths.min().item())
            assert split_depth_val > 0, (
                "split depth must be positive."
                "this case implies all nodes are split for some domain, "
                "which should be handled after postprocessing"
            )
            domain_decision.packed_branching_decision = [
                decision[0:split_depth_val]
                for decision in domain_decision.packed_branching_decision
            ]

        if not all([v is None for v in domain_decision.packed_branching_points]):
            branching_points = (
                torch.Tensor(
                    [
                        points[0:split_depth_val]
                        for points in domain_decision.packed_branching_points
                    ]
                )
                .reshape(batch_size, split_depth_val)
                .permute(1, 0)
                .reshape(-1)
            )
        else:
            branching_points = None

        # original shape=(batch_size, split_depth, 2)
        # target shape=(split_depth, batch_size, 2)
        packed_branching_decision = (
            torch.Tensor(domain_decision.packed_branching_decision)
            .reshape(batch_size, split_depth_val, 2)  # make split_depth=0 first citizen
            .permute(1, 0, 2)
            .to(torch.int32)
        )

        branching_decision = packed_branching_decision.reshape(
            (batch_size * split_depth_val, 2)
        ).tolist()

        return BranchingDecisions(
            branching_decision=branching_decision,
            branching_points=branching_points,
            split_depth=split_depth_val,
            batch_size=batch_size,
        )

    @staticmethod
    def reconstruct_from_masked_result(
        branching_decision: "BranchingDecisions",
        original_batch_size: int,
        used_batch_mask: torch.Tensor,
    ) -> "BranchingDecisions":
        """Reconstruct a BranchingDecisions from a masked result one"""

        new_branching_decision = []
        assert used_batch_mask.shape[0] == original_batch_size
        used_batch_idx = torch.nonzero(used_batch_mask).squeeze(1).tolist()

        magic_number = -999999999
        place_holder = [magic_number, magic_number]
        new_branching_decision = [place_holder] * (
            branching_decision.split_depth * original_batch_size
        )
        for d in range(branching_decision.split_depth):
            for rank, value in enumerate(used_batch_idx):
                new_branching_decision[d * original_batch_size + value] = (
                    branching_decision.branching_decision[
                        d * branching_decision.batch_size + rank
                    ]
                )

        if branching_decision.branching_points is not None:
            new_branching_points = torch.full(
                (branching_decision.split_depth, original_batch_size),
                magic_number,
                dtype=branching_decision.branching_points.dtype,
            )
            old_branching_points = branching_decision.branching_points.cpu().reshape(
                branching_decision.split_depth, branching_decision.batch_size
            )
            new_branching_points[:, used_batch_idx] = old_branching_points
            new_branching_points = new_branching_points.reshape(-1)
        else:
            new_branching_points = None

        return BranchingDecisions(
            branching_decision=new_branching_decision,
            branching_points=new_branching_points,
            split_depth=branching_decision.split_depth,
            batch_size=original_batch_size,
        )


@dataclass
class BatchFirstBranchingDecisions:
    """
    Domain-first memory layout for a batch of split decisions.

    Maps:
        [sample_index -> (branching_decision, branching_points, split_depth)]

    packed_branching_decision[sample_index] = list of decisions of the sample.
        shape=(batch_size, split_depth, 2)
    packed_branching_points[sample_index] = list of branching points of the sample.
        shape=(batch_size, split_depth)
    packed_branching_points_split_depth[sample_index] = split depth of the sample.
        shape=(batch_size)
    """

    packed_branching_decision: List
    # list of decisions. shape=(domain_size, split_depth, 2)
    packed_branching_points: List
    # list of branching points. shape=(domain_size, split_depth)
    packed_branching_points_split_depth: torch.Tensor
    batch_size: int

    @staticmethod
    def from_branching_decision(
        branching_decision: BranchingDecisions,
    ) -> "BatchFirstBranchingDecisions":

        split_depth = branching_decision.split_depth
        batch_size = branching_decision.batch_size

        working_branching_decision = torch.tensor(
            branching_decision.branching_decision, dtype=torch.int32
        ).reshape(split_depth, batch_size, 2)
        # shape=(split_depth, batch_size, 2)

        # original shape=(split_depth, batch_size, 2)
        # target shape=(batch_size, split_depth, 2)
        packed_branching_decision = working_branching_decision.permute(1, 0, 2)

        # make last two dimensions appear as object

        packed_branching_decision = [
            packed_branching_decision[i].tolist() for i in range(batch_size)
        ]

        # target shape=(batch_size)
        packed_split_domain = torch.zeros(batch_size, dtype=torch.int32) + split_depth

        if branching_decision.branching_points is not None:
            # original shape=(split_depth * batch_size)
            # target shape=(batch_size, split_depth)
            packed_branching_points = (
                branching_decision.branching_points.reshape(split_depth, batch_size)
                .permute(1, 0)
                .tolist()
            )
        else:
            packed_branching_points = [None] * batch_size

        return BatchFirstBranchingDecisions(
            packed_branching_decision=packed_branching_decision,
            packed_branching_points=packed_branching_points,
            packed_branching_points_split_depth=packed_split_domain,
            batch_size=batch_size,
        )

    def narrow_to(self, split_depth: int) -> "BatchFirstBranchingDecisions":
        """Narrow the BatchFirstBranchingDecisions to a smaller split_depth."""
        if split_depth > int(
            self.packed_branching_points_split_depth[0].item()
        ):
            raise ValueError(
                "Cannot narrow to a larger split_depth than current split_depth."
            )

        new_packed_branching_decision = [
            decision[0:split_depth] for decision in self.packed_branching_decision
        ]

        if not all([v is None for v in self.packed_branching_points]):
            new_packed_branching_points = [
                points[0:split_depth] for points in self.packed_branching_points
            ]
        else:
            new_packed_branching_points = self.packed_branching_points

        new_packed_branching_points_split_depth = torch.zeros(
            self.batch_size, dtype=torch.int32
        ) + split_depth

        return BatchFirstBranchingDecisions(
            packed_branching_decision=new_packed_branching_decision,
            packed_branching_points=new_packed_branching_points,
            packed_branching_points_split_depth=new_packed_branching_points_split_depth,
            batch_size=self.batch_size,
        )
