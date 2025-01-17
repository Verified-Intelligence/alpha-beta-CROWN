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
import torch
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any
from input_split.alpha import set_alpha_input_split

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def get_lower_bound_naive(
        self: 'LiRPANet', dm_lb=None, dm_l=None, dm_u=None, alphas=None,
        bounding_method='crown', branching_method='sb', C=None,
        stop_criterion_func=None, thresholds=None,
        reference_interm_bounds=None):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    enable_clip_domains = arguments.Config["bab"]["branching"]["input_split"]["enable_clip_domains"]
    split_partitions = input_split_args["split_partitions"]
    reorder_bab = input_split_args["reorder_bab"]

    if bounding_method == 'alpha-forward':
        raise ValueError("Should not use alpha-forward.")

    batch = len(dm_l) // split_partitions
    ptb = PerturbationLpNorm(
        norm=self.x.ptb.norm, eps=self.x.ptb.eps, x_L=dm_l, x_U=dm_u)
    new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
    # set alpha here again
    set_alpha_input_split(self, alphas, set_all=True, double=not reorder_bab, split_partitions=split_partitions)
    self.net.set_bound_opts({'optimize_bound_args': {
        'enable_beta_crown': False,
        'fix_interm_bounds': True,
        'iteration': arguments.Config["solver"]["beta-crown"]["iteration"],
        'lr_alpha': arguments.Config["solver"]["beta-crown"]['lr_alpha'],
        'stop_criterion_func': stop_criterion_func(thresholds),
    }})
    # need lA and lbias to shrink domains
    return_A = branching_method != 'naive' or enable_clip_domains
    # As of now, no method requires lbias so this is set to False. In the future, this can be set to
    # True based on any heuristic/method that requires it
    return_b = enable_clip_domains and reorder_bab
    lb_crown = None # CROWN lower bound from IBP enhancement
    if return_A:
        needed_A_dict = defaultdict(set)
        needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
    else:
        needed_A_dict = None

    def _get_lA(ret):
        if return_A:
            A_dict = ret[-1]
            lA = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']
        else:
            lA = None
        return lA

    def _get_lbias(ret):
        if return_b:
            A_dict = ret[-1]
            lbias = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lbias']
        else:
            lbias = None
        return lbias

    if bounding_method == "alpha-crown":
        ret = self.net.compute_bounds(
            x=(new_x,), C=C, method='CROWN-Optimized', bound_upper=False,
            return_A=return_A, needed_A_dict=needed_A_dict)
        lb = ret[0]
        lA = _get_lA(ret)
        lbias = _get_lbias(ret)
    elif bounding_method == 'ibp':
       lb = self.net.compute_bounds(x=(new_x,), C=C, method='ibp')[0]
       lA = None
       lbias = None
    else:
        if arguments.Config['bab']['branching']['input_split']['ibp_enhancement']:
            lb, lA, lbias, lb_crown = get_lower_bound_with_ibp_enhancement(
                self, new_x, C, bounding_method,
                thresholds, return_A=return_A, needed_A_dict=needed_A_dict,
                stop_criterion_func=stop_criterion_func,)
        else:
            with torch.no_grad():
                ret = self.net.compute_bounds(
                    x=(new_x,), C=C, method=bounding_method,
                    bound_upper=False, return_A=return_A, needed_A_dict=needed_A_dict,
                    reuse_alpha=bounding_method.lower() == 'crown' and len(alphas) > 0,
                    reference_bounds=reference_interm_bounds
                )
                lb = ret[0]
                lA = _get_lA(ret)
                lbias = _get_lbias(ret)

        if dm_lb is not None:
            lb = torch.max(lb, dm_lb)

        worst_idx = (lb - thresholds).amax(dim=-1).argmin()
        print('Worst bound:', (lb - thresholds)[worst_idx])

    with torch.no_grad():
        # Transfer everything to CPU.
        lb = lb.cpu()
        if bounding_method == "alpha-crown":
            ret_s = self.get_alpha(device='cpu', half=True)
        else:
            # There might be alphas in initial alpha-crown,
            # which will be used in all later bounding steps.
            # Here only the references in the list is copied.
            # The alphas will be actually duplicated in TensorStorage.
            ret_s = alphas * (batch * 2)

    # if IBP enhancement is not used, return lower bound
    if lb_crown is None:
        lb_crown = lb

    return lb, ret_s, lA, lbias, lb_crown


def get_lower_bound_with_ibp_enhancement(
        self, new_x, C, bounding_method, thresholds,
        return_A=False, needed_A_dict=None,
        stop_criterion_func=stop_criterion_batch_any):
    reference_interm_bounds = {}
    lb_ibp = self.net.compute_bounds(
        x=(new_x,), C=C, method='ibp', bound_upper=False, return_A=False)[0]
    lb = lb_ibp.clone()
    for node in self.net.nodes():
        if (node.perturbed
            and isinstance(node.lower, torch.Tensor)
            and isinstance(node.upper, torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    unverified = torch.logical_not(
        stop_criterion_func(thresholds)(lb_ibp).any(dim=-1))
    if not unverified.any():
        # The code below assumes that there is at least one unverified domain
        unverified[0] = True
    x_unverified = BoundedTensor(
        new_x[unverified],
        ptb=PerturbationLpNorm(
            norm=new_x.ptb.norm, eps=new_x.ptb.eps,
            x_L=new_x.ptb.x_L[unverified],
            x_U=new_x.ptb.x_U[unverified]))
    lb_crown, _, A_dict = self.net.compute_bounds(
        x=(x_unverified,), C=C[unverified], method=bounding_method,
        bound_upper=False, return_A=True, needed_A_dict=needed_A_dict,
        reference_bounds={
            k: (v[0][unverified], v[1][unverified])
            for k, v in reference_interm_bounds.items()}
    )
    lb[unverified] = torch.max(lb[unverified], lb_crown)
    full_lb_crown = torch.zeros_like(lb)
    full_lb_crown[unverified] = lb_crown

    if return_A:
        lA_ = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']
        lA = torch.empty(new_x.shape[0], *lA_.shape[1:], device=lA_.device)
        lA[unverified] = lA_
        lbias_ = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lbias']
        lbias = torch.empty(new_x.shape[0], lbias_.shape[1], device=lbias_.device)
        lbias[unverified] = lbias_
    else:
        lA = None
        lbias = None

    return lb, lA, lbias, full_lb_crown
