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
import time

from dataclasses import dataclass

def initial_verify_criterion(lbs, rhs):
    """check whether verify successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    verified_idx = torch.any(
        (lbs - rhs) > 0, dim=-1
    )  # return bolling results in x's batch-wise
    if verified_idx.all():  # check whether all x verified
        print("Verified by initial bound!")
        return True, torch.where(verified_idx == 0)[0]
    else:
        return False, torch.where(verified_idx == 0)[0]

@dataclass
class Timer:
    total_func_time: float = 0.0
    total_prepare_time: float = 0.0
    total_bound_time: float = 0.0
    total_beta_bound_time: float = 0.0
    total_transfer_time: float = 0.0
    total_finalize_time: float = 0.0

    def __init__(self):
        self.time_start = {}
        self.time_last = {}
        self.time_sum = {}

    def start(self, name):
        self.time_start[name] = time.time()
        if name not in self.time_sum:
            self.time_sum[name] = 0

    def add(self, name):
        self.time_last[name] = time.time() - self.time_start[name]
        self.time_sum[name] += self.time_last[name]

    def print(self):
        print('Time: ', end='')
        for k, v in self.time_last.items():
            print(f'{k} {v:.4f}', end='    ')
        print()
        print('Accumulated time: ', end='')
        for k, v in self.time_sum.items():
            print(f'{k} {v:.4f}', end='    ')
        print()

class Stats:
    def __init__(self):
        self.visited = 0  # number of domains bounded in BaB
        self.storage_depth = 0  # maximum possible splits per BaB round
        self.timer = Timer()
