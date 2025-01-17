#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import matplotlib.pyplot as plt
from os import makedirs
from math import ceil
from numpy import ndarray
from typing import Optional
from warnings import warn

def save_sanity_check_graphs(
        global_lbs: ndarray,
        benchmark_name: Optional[str],
        vnnlib_id: int,
        property_idx: int,
        dir_timestamp: str
) -> None:
    """
    When called, creates a log scale convergence plot of (rhs - global_lb) where global_lb < rhs.
    Should global_lb >= rhs, an error will occur when plotting.
    @param global_lbs:      The global lower bound values after i iterations
    @param benchmark_name:  Name of the currently running benchmark
    @param vnnlib_id:       Current vnnlib_id in the benchmark
    @param property_idx:    Current property being verified for the current vnnlib file
    @param dir_timestamp:   The timestamp is used as the directory for which all graphs will be saved to
    @return:
    """

    if benchmark_name is None:
        warn("'save_sanity_check_graphs' was called but benchmark_name not given. Will skip creating graphs.")
        return

    iterations, features = global_lbs.shape
    grid_division = 2 if features > 1 else 1
    plt.figure(figsize=(16, 12))
    print(f"Saving graphs...")
    rows = ceil(features / grid_division)
    for i in range(features):
        plt.subplot(rows, grid_division, i + 1)
        best_lb = global_lbs[:, i].min()
        plt.plot(global_lbs[:, i])
        plt.grid()
        plt.title(f"Convergence for feature {i}\nBest (rhs - lb): {best_lb}")
        plt.xlabel("Iterations")
        plt.ylabel(f"rhs[{i}] - global_lb[{i}]")
        plt.yscale('log')
    save_dir = f"../sanity_check_outputs/{dir_timestamp}"
    makedirs(save_dir, exist_ok=True)
    plt.savefig(
        save_dir + f"/benchmark_{benchmark_name}_vnnlib_id_{vnnlib_id}_property_{property_idx}_sanity_check_graphs.png")