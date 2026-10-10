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
from contextlib import contextmanager

import torch
from typing import Any

from state.traits_mixin import (
    DictLikeMixIn,
    ListLikeMixIn,
)


def mask_tensor_first_dim(obj, mask: torch.Tensor) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj[mask]

    if isinstance(obj, dict):
        return {k: mask_tensor_first_dim(v, mask) for k, v in obj.items()}

    if isinstance(obj, list):
        return [mask_tensor_first_dim(v, mask) for v in obj]

    raise TypeError(f"Unsupported type for masking: {type(obj)!r}")


def mask_tensor_first_dim_allow_none(obj, mask):
    if obj is None:
        return None
    return mask_tensor_first_dim(obj, mask)


def mask_list(obj, mask):
    if isinstance(obj, list):
        return [obj[i] for i in range(len(obj)) if mask[i]]
    else:
        raise ValueError("Unsupported type for masking.")


def brutely_gc_tensor(obj, visited_ids=None):
    if visited_ids is None:
        visited_ids = set()
    if isinstance(obj, torch.Tensor):
        if id(obj) in visited_ids:
            return
        visited_ids.add(id(obj))
        obj.resize_(0)
        return
    if isinstance(obj, (dict, DictLikeMixIn)):
        for v in obj.values():
            brutely_gc_tensor(v, visited_ids)
        return
    elif isinstance(obj, (list, ListLikeMixIn)):
        for v in obj:
            brutely_gc_tensor(v, visited_ids)
        return


@contextmanager
def tf32_ctx(enabled: bool):
    """Temporarily set TF32 policy for matmul and cuDNN."""
    old_mm  = torch.backends.cuda.matmul.allow_tf32
    old_dnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32       = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32       = old_dnn
