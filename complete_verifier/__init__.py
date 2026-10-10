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
import sys
import os

__version__ = '0.8.0'

print("Adding complete_verifier to sys.path")
sys.path = [os.path.dirname(__file__)] + sys.path


from api import (
    ABCrownSolver,
    IOConstraints,
    VerificationSpec,
    BoundsResult,
    ConfigBuilder,
    default_config,
    input_vars,
    output_vars,
    VNNCompInstance,
    VNNCompBenchmark,
    load_vnncomp_instance,
    run_all_instances,
    run_specific_instance,
)
from .abcrown import ABCROWN

__all__ = [
    "ABCrownSolver",
    "ABCROWN",
    "IOConstraints",
    "VerificationSpec",
    "BoundsResult",
    "ConfigBuilder",
    "default_config",
    "input_vars",
    "output_vars",
    "VNNCompInstance",
    "VNNCompBenchmark",
    "load_vnncomp_instance",
    "run_all_instances",
    "run_specific_instance",
]
