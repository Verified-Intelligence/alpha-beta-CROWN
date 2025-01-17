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
import arguments

from heuristics.base import RandomNeuronBranching, InterceptBranching
from heuristics.babsr import BabsrBranching
from heuristics.fsb import FsbBranching
from heuristics.kfsb import KfsbBranching
from heuristics.nonlinear import NonlinearBranching


def get_branching_heuristic(net, method=None):
    if method is None:
        branching_method = arguments.Config['bab']['branching']['method']
    else:
        branching_method = method
    disable_genbab = arguments.Config['bab']['branching']['nonlinear_split']['disable']
    if branching_method != 'nonlinear' and net.nonlinear_split and not disable_genbab:
        branching_method = 'nonlinear'
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
