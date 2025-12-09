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
import sys
import os

__version__ = '0.7.0'

print("Adding complete_verifier to sys.path")
sys.path = [os.path.dirname(__file__)] + sys.path


from api import (
    ABCrownSolver,
    VerificationSpec,
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

__all__ = [
    "ABCrownSolver",
    "VerificationSpec",
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