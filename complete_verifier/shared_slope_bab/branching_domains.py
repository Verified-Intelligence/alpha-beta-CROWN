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
"""Open domains at one depth in the shared-slope BaB."""
import torch


class SharedSlopeDomainList:
    """Open branch decisions and warm-start multipliers for one BaB round."""

    def __init__(self, sides, phase_dual, box_dual):
        self.sides = sides
        self.phase_dual = phase_dual
        self.box_dual = box_dual

    @classmethod
    def root(cls, device, dtype):
        return cls(torch.tensor([[0], [1]], device=device, dtype=torch.int8),
                   torch.zeros(2, 1, device=device, dtype=dtype),
                   torch.zeros(2, 1, device=device, dtype=dtype))

    def __len__(self):
        return self.sides.shape[0]

    def filter_verified_domains(self, unverified):
        return SharedSlopeDomainList(
            self.sides[unverified], self.phase_dual[unverified],
            self.box_dual[unverified])

    def split_domains(self):
        n_domains, depth = self.sides.shape
        sides = torch.empty(2 * n_domains, depth + 1,
                           device=self.sides.device, dtype=torch.int8)
        phase_dual = torch.empty(2 * n_domains, depth + 1,
                                 device=self.phase_dual.device,
                                 dtype=self.phase_dual.dtype)
        box_dual = torch.empty_like(phase_dual)

        sides[:n_domains, :depth] = self.sides
        sides[n_domains:, :depth] = self.sides
        sides[:n_domains, depth] = 0
        sides[n_domains:, depth] = 1

        phase_dual[:n_domains, :depth] = self.phase_dual
        phase_dual[n_domains:, :depth] = self.phase_dual
        phase_dual[:, depth] = 0

        box_dual[:n_domains, :depth] = self.box_dual
        box_dual[n_domains:, :depth] = self.box_dual
        box_dual[:, depth] = 0
        return SharedSlopeDomainList(sides, phase_dual, box_dual)
