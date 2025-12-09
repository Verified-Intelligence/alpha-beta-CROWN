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
import arguments
from . import attack_pgd, general_spec_attack

# They are shared with two versions of attack
from .attack_utils import (default_adv_saver, default_adv_verifier, PGDAttackResult,
                           reset_attack_stats, get_attack_stats, check_and_save_cex)
from .attack_interface import attack, attack_with_general_specs


def route(name):
    def wrapped(*args, **kwargs):
        if arguments.Config["attack"]["general_attack"]:
            return getattr(general_spec_attack, name)(*args, **kwargs)
        else:
            # For backward compatibility, we still support the old attack_pgd module.
            # In the future, we will remove this and only use general_spec_attack.
            return getattr(attack_pgd, name)(*args, **kwargs)
    return wrapped

# They are implemented differently in two versions of attack,
# so we need to call them dynamically depending on the attack version.
# Please refer to attack_pgd.py and general_spec_attack.py for specific implementations.
default_pgd_loss = route("default_pgd_loss")
default_early_stop_condition = route("default_early_stop_condition")
default_adv_example_finalizer = route("default_adv_example_finalizer")
pgd_attack_with_general_specs = route("pgd_attack_with_general_specs")
