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
import torch
from torch import Tensor
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any

from typing import TYPE_CHECKING, Optional, Tuple, Callable
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def get_lower_bound_naive(
        self: 'LiRPANet',
        dm_lb: Optional[Tensor] = None,
        dm_l: Optional[Tensor] = None,
        dm_u: Optional[Tensor] = None,
        alphas: Optional[dict] = None,
        bounding_method: str = 'crown',
        C: Optional[Tensor] = None,
        stop_criterion=None,
        thresholds: Optional[Tensor] = None,
        reference_interm_bounds = None,
        return_A = False,
        return_b = False,
        constraints: Optional[tuple] = None,
        stats=None
) -> Tuple[Tensor, dict, Optional[Tensor], Optional[Tensor], Tensor]:
    """

    Bounds a batch of subdomains.

    :param self:                        Reference to the LiRPANet object
    :param dm_lb:                       The lower bound on the outputs of the domains
    :param dm_l:                        The lower bound on the inputs of the subdomains
    :param dm_u:                        The upper bound on the inputs of the subdomains
    :param alphas:                      If given, alpha parameters for each subdomain
    :param bounding_method:             crown, alpha-crown, etc.
    :param C:                           Output specification matrix
    :param stop_criterion:              Callable function which returns True for domains that have been verified
    :param thresholds:                  The specification threshold where dm_lb > thresholds implies the subdomain is verified
    :param reference_interm_bounds:     If given, these bounds are used as a reference, and the bounds
                                        for intermediate layers will still be computed (e.g., using CROWN,
                                        or other specified methods). The computed bounds will be
                                        compared to "reference_bounds" and the tighter one between the two
                                        will be used.
    :param return_A, return_b:          If true, the lA and lbias matrices of the final layer will be returned.
    :param constraints:                 A tuple of (constraints_A, constraints_b).
    :param stats:                       Statistics recorder.
    :return lb:                         The lower bound of the network output
    :return ret_s:                      Returns the updated alphas for each subdomain
    :return lA:                         If available, the bounding hyperplane coefficients
    :return lbias:                      If available, the bounding hyperplane offsets
    """
    clip_args = arguments.Config["bab"]["clip_n_verify"]
    enable_constrained_concretize = clip_args["clip_input_domain"]["enabled"] and (clip_args['clip_input_domain']['clip_type'] == "complete")
    rearrange_constraints = clip_args["rearrange_constraints"]

    if bounding_method == 'alpha-forward':
        raise ValueError("Should not use alpha-forward.")

    # Initialize the infeasible_bounds_constraints tensor
    if enable_constrained_concretize:
        batchsize = len(dm_l)
        device = dm_l.device
        self.net.init_infeasible_bounds_constraints(batchsize, device)

    ptb = PerturbationLpNorm(x_L=dm_l, x_U=dm_u,
        constraints=constraints, rearrange_constraints=rearrange_constraints, no_return_inf=enable_constrained_concretize,
        timer=stats.timer)
    new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
    # set alpha here again
    self.set_alpha(alphas, set_all=True)
    self.net.set_bound_opts({'optimize_bound_args': {
        'enable_beta_crown': False,
        'fix_interm_bounds': True,
        'iteration': arguments.Config['solver']['alpha-crown']['input_split_alpha_iteration'],
        'lr_alpha': arguments.Config['solver']['alpha-crown']['input_split_lr_alpha'],
        'stop_criterion_func': stop_criterion(thresholds),
    }})
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
            lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
                self, new_x, C, bounding_method,
                thresholds, return_A=return_A, needed_A_dict=needed_A_dict,
                stop_criterion=stop_criterion, constraints=constraints, timer=stats.timer,
                rearrange_constraints=rearrange_constraints, no_return_inf=enable_constrained_concretize)
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
                # For infeasible batches, CROWN will always return naive results out of computational stability.
                # But BoundedModule instance will keep track of these batches with "infeasible_bounds_constraints"
                infeasible_batch = self.net.infeasible_bounds_constraints
                if infeasible_batch is not None:
                    lb[infeasible_batch] = torch.inf

        # dm_lb is not None if compare_with_old_bounds is True, 
        if dm_lb is not None:
            lb = torch.max(lb, dm_lb)

        worst_idx = (lb - thresholds).amax(dim=-1).argmin()
        print('Worst bound:', (lb - thresholds)[worst_idx])

    with torch.no_grad():
        if bounding_method == "alpha-crown":
            ret_s = self.get_alpha(
                get_all=True,
                half=arguments.Config["solver"]["alpha-crown"]["alpha_dtype"] == "float16"
            )
        else:
            # There might be alphas in initial alpha-crown,
            # which will be used in all later bounding steps.
            ret_s = alphas

    return lb, ret_s, lA, lbias


def get_lower_bound_with_ibp_enhancement(
        self: 'LiRPANet', new_x: Tensor, C: Tensor, bounding_method: str, thresholds: Tensor,
        return_A:bool=False, needed_A_dict:Optional[dict]=None,
        stop_criterion: Callable=stop_criterion_batch_any,
        constraints:tuple=None, timer=None,
        rearrange_constraints:bool=False, no_return_inf:bool=False
)->Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
    """

    Bounds a batch of domains using interval bound propagation (IBP). If a subset of domains could not be verified
    using IBP, their lower bounds are refined using a specified bounding method e.g. crown.

    :param self:                Reference to the LiRPANet object
    :param new_x:               Updated input box for each subdomain
    :param C:                   Output specification matrix
    :param bounding_method:     crown, alpha-crown, etc.
    :param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param return_A:            If True, must return parameters specified in 'needed_A_dict'
    :param needed_A_dict:       Nodes whose CROWN lA and lbias parameters should be returned
    :param stop_criterion:      Callable function which returns True for domains that have been verified
    :param constraints:         A tuple of (constraints_A, constraints_b)
    :param timer:               A time recorder.
    :paran rearrange_constraints:
                                Whether to rearrange the constraints based on their distances to input region centroids.
    :param no_return_inf:       Whether inf value is allowed in intermediate bound.

    :return lb:                 The lower bound of the network output, a mixture of bounds from IBP and 'bounding_method'
    :return lA:                 CROWN hyperplane coefficients mapping the input to the network output
    :return lbias:              CROWN hyperplane offsets mapping the input to the network output
    :return lb_crown:           The lower bound of the network output, solely from 'bounding_method', not IBP
    """
    # Holds the intermediate bounds achieved using IBP to record as reference. When 'bounding_method' is subsequently
    # invoked on the remaining unverified domains, the tighter intermediate bounds between IBP and 'bounding_method' are used.
    reference_interm_bounds = {}
    # get the lower bound using interval bound propagation (IBP)
    lb_ibp = self.net.compute_bounds(
        x=(new_x,), C=C, method='ibp', bound_upper=False, return_A=False)[0]
    lb = lb_ibp.clone()
    for node in self.net.nodes():
        if (node.perturbed
            and isinstance(node.lower, torch.Tensor)
            and isinstance(node.upper, torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    unverified = torch.logical_not(stop_criterion(thresholds)(lb_ibp).any(dim=-1))
    if not unverified.any():
        # The code below assumes that there is at least one unverified domain
        unverified[0] = True
    unverified_constraints = None
    if constraints is not None:
        constraints_A, constraints_b = constraints
        unverified_A = constraints_A[unverified]
        unverified_b = constraints_b[unverified]
        unverified_constraints = (unverified_A, unverified_b)
    x_unverified = BoundedTensor(
        new_x[unverified],
        ptb=PerturbationLpNorm(
            x_L=new_x.ptb.x_L[unverified],
            x_U=new_x.ptb.x_U[unverified],
            constraints=unverified_constraints,
            timer=timer,
            rearrange_constraints=rearrange_constraints,
            no_return_inf=no_return_inf))
    # Initialize the infeasible_bounds_constraints tensor
    clip_args = arguments.Config["bab"]["clip_n_verify"]
    enable_constrained_concretize = clip_args["clip_input_domain"]["enabled"] and (clip_args['clip_input_domain']['clip_type'] == "complete")
    if enable_constrained_concretize:
        batchsize = unverified.sum()
        device = new_x.ptb.x_L.device
        self.net.init_infeasible_bounds_constraints(batchsize, device)

    ret = self.net.compute_bounds(
        x=(x_unverified,), C=C[unverified], method=bounding_method,
        bound_upper=False, return_A=return_A, needed_A_dict=needed_A_dict,
        reference_bounds={
            k: (v[0][unverified], v[1][unverified])
            for k, v in reference_interm_bounds.items()}
    )
    lb_crown = ret[0]
    A_dict = ret[-1] if return_A else None
    infeasible_batch = self.net.infeasible_bounds_constraints
    if infeasible_batch is not None:
        lb_crown[infeasible_batch] = torch.inf
    lb[unverified] = torch.max(lb[unverified], lb_crown)

    # format lA and lbias if required
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

    return lb, lA, lbias
