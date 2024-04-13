#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import arguments

from heuristics.base import RandomNeuronBranching, InterceptBranching
from heuristics.babsr import BabsrBranching
from heuristics.fsb import FsbBranching
from heuristics.kfsb import KfsbBranching
from heuristics.nonlinear import NonlinearBranching


def get_branching_heuristic(net):
    branching_method = arguments.Config['bab']['branching']['method']
    branching_obj = None
    if branching_method == 'random':
        branching_obj = RandomNeuronBranching(net)
    elif branching_method == 'intercept':
        branching_obj = InterceptBranching(net)
    elif branching_method == 'nonlinear':
        branching_args = arguments.Config['bab']['branching']['nonlinear_split']
        branching_obj = NonlinearBranching(net, **branching_args)
    elif branching_method == 'babsr':
        branching_obj = BabsrBranching(net)
    elif branching_method == 'fsb':
        branching_obj = FsbBranching(net)
    elif branching_method.startswith('kfsb'):
        branching_obj = KfsbBranching(net)
    else:
        raise ValueError(f'Unsupported branching method "{branching_method}" '
                         'for activation splits.')
    return branching_obj
