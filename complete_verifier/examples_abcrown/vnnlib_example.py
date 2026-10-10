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

# An example of using the ABCrown API to verify a model on a traditional onnx + vnnlib instance with given config.

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    IOConstraints,
)

vnnlib_path = "vnnlib_example_dependency/cifar_base_kw-img876-eps0.024836601307189544.vnnlib"

constraints = IOConstraints(vnnlib_path=vnnlib_path)

config = ConfigBuilder.from_yaml(
    "../exp_configs/vnncomp22/oval22.yaml"
)

model_path = "vnnlib_example_dependency/cifar_base_kw.onnx"

solver = ABCrownSolver(model_path, config=config)
result = solver.verify(constraints=constraints)

print(f"status={result.status}, success={result.success}")
