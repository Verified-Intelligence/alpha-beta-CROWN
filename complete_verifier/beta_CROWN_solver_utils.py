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
from domain_updater import DomainUpdater, DomainUpdaterSimple


def build_history_and_set_bounds_static(d_inout, split, mode, final_name, split_nodes_names):
    """
    Create new domains as the result of split decisions. Inplace d_inout.
    """
    _, num_split = DomainUpdater.get_num_domain_and_split(d_inout, split, final_name)
    args = (final_name, split_nodes_names)
    if num_split == 1 and (split.get("points", None) is None
                            or split["points"].ndim == 1):
        domain_updater = DomainUpdaterSimple(*args)
    else:
        domain_updater = DomainUpdater(*args)
    domain_updater.set_branched_bounds(d_inout, split, mode)
