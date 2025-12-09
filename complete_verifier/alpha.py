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


"""
# This file contains functions to handle alphas in the LiRPANet.
# It includes functions to drop unused alphas, get alphas, and set alphas.
"""

import torch
from auto_LiRPA.utils import transfer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet

def drop_unused_alpha(self: 'LiRPANet'):
    optimizable_activations = self.net.get_enabled_opt_act()
    keep_nodes = self.alpha_start_nodes
    for m in optimizable_activations:
        m.drop_unused_alpha(keep_nodes)

def get_alpha(self: 'LiRPANet', get_all=True, half=False, device=None, full_info=False, drop_unused=True):
    """
    Saves alphas in a dictionary.

    :param get_all: if True, saves all alphas, otherwise only those which will be optimized.
    :param half: if True, saves alphas in half precision.
    :param device: device to transfer alphas to.
    :param full_info: if True, saves full alpha information as a dict, otherwise only the alpha values.
    :param drop_unused: if True, drops unused alphas before saving.
    :return: dictionary with alphas.
    :rtype: dict
    :note: the dictionary is structured as follows:
            When `full_info` is True:
                {node_name: {'alpha': {start_node_name: alpha_value},
                             'alpha_lookup_idx': {start_node_name: alpha_lookup_idx}, ...}}
            When `full_info` is False:
                {node_name: {start_node_name: alpha_value}}
    """
    if drop_unused and not get_all:
        self.drop_unused_alpha()

    # alpha has size (2, spec, batch, *shape). When we save it,
    # we make batch dimension the first.
    # spec is some intermediate layer neurons, or output spec size.
    ret = {}
    dtype = None if not half else torch.float16
    optimizable_activations = self.net.get_enabled_opt_act()
    for m in optimizable_activations:
        if full_info:
            ret[m.name] = m.dump_alpha(device=device, dtype=dtype)
        else:
            ret[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                if get_all or spec_name in self.alpha_start_nodes:
                    ret[m.name][spec_name] = transfer(alpha, device, dtype=dtype)
    return ret


def set_alpha(self: 'LiRPANet', alpha, set_all=False, full_info=False, device=None, dtype=None):
    """
    Sets alphas from a dictionary. It is a reverse operation of `get_alpha`.

    :param alpha: dictionary with alphas.
    :param set_all: if True, sets all alphas, otherwise only those which will be optimized.
    :param full_info: if True, alpha is a dict with full information, otherwise it is a tensor.
    :param device: device to transfer alphas to.
    :param dtype: dtype to transfer alphas to.
    """
    # If alpha is empty, we do nothing.
    if len(alpha) == 0:
        return

    # if set_all is True, we set all alphas from input alpha,
    # otherwise we only set alphas which will be further optimized.

    optimizable_activations = self.net.get_enabled_opt_act()
    keep_nodes = self.alpha_start_nodes
    for m in optimizable_activations:
        if full_info:
            m.restore_alpha(alpha[m.name], device=device, dtype=dtype)
        else:
            for spec_name in list(m.alpha.keys()):
                if (spec_name in alpha[m.name]) and (spec_name in keep_nodes or set_all):
                    m.alpha[spec_name] = alpha[m.name][spec_name]
                    # Duplicate for the second half of the batch.
                    m.alpha[spec_name] = m.alpha[spec_name].detach().requires_grad_()

    if not set_all:
        # If we are not setting all alphas, we need to drop unused alphas.
        # in most cases, set_all is False and dropping should be done before, here just in case.
        self.drop_unused_alpha()
