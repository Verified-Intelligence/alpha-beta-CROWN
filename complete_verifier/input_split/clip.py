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
from typing import Tuple, Optional, Union

log_underflow = torch.log(torch.tensor([torch.finfo(torch.float64).tiny], dtype=torch.float64)).item()

def _in_depth_volume_metrics(original_x_L: Tensor, original_x_U: Tensor, new_x_L: Tensor, new_x_U: Tensor,
                                  nv_mask: Tensor, lA:Tensor, thresholds: Tensor,
                                  dm_lb: Tensor, is_lower: int):
    """

    Performs in depth volume calculations in order to better gauge how effective clipping was on this batch of
    subdomains. In particular, the total volume of this batch of domains is calculated before and after clipping.
    The log-sum-exp trick is used in order to make the calculations more stable.

    When numerical underflow is encountered, each constraint is applied one at a time and evaluated in order to
    analyze if there is a particular constraint that is very valuable to the clipping procedure or if it is a
    combination of constraints that allow the input to be clipped a lot.

    These metrics are optional in the 'clip_domains' method because it faces two major limitations:

    1)  When the input space is high-dimensional and/or small, these calculations will consistently run into
        numerical underflow. Even the log-sum-exp trick cannot salvage this scenario.
    2)  This metric is not insightful in every instance, therefore it can add unnecessary computational overhead
        that can compound over many rounds of BaB.

    Despite these limitations, these metrics may be insightful in certain scenarios. For low dimensional inputs,
    it can consistently provide accurate volume calculations and help gauge which constraints are the most
    valuable.

    Note that the Tensors are convert to float64 for higher precision as to mitigate numerical underflow
    as much as possible.

    :param original_x_L:    The original lower bound on the inputs of the subdomains
    :param original_x_U:    The original upper bound on the inputs of the subdomains
    :param new_x_L:         The lower bound on the inputs of the subdomains after clipping
    :param new_x_U:         The upper bound on the inputs of the subdomains after clipping
    :param nv_mask:         Mask of unverified domains post-clipping
    :param lA:              CROWN lA for subdomains
    :param thresholds:      The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param dm_lb:           The lower bound on the outputs of the domains
    :param is_lower:        If set to true, we are currently lower bounding the network output, else we are upper
    """

    batches, num_spec, input_dim = lA.shape

    # convert tensors to float64 for better precision in order to try
    # and mitigate numerical underflow as much as possible
    lA = lA.to(dtype=torch.float64)
    thresholds = thresholds.to(dtype=torch.float64)
    dm_lb = dm_lb.to(dtype=torch.float64)
    nv_x_L = new_x_L[nv_mask].to(dtype=torch.float64)
    nv_x_U = new_x_U[nv_mask].to(dtype=torch.float64)
    nv_ori_x_L = original_x_L[nv_mask].to(dtype=torch.float64)
    nv_ori_x_U = original_x_U[nv_mask].to(dtype=torch.float64)

    # get the log sum of the areas of the original (unverified) domains before and after clipping
    log_original_areas = torch.log((nv_ori_x_U - nv_ori_x_L)).sum(1)
    log_new_areas = torch.log((nv_x_U - nv_x_L)).sum(1)

    # Uses log-sum-exp trick to calculate the ratio (total new area / total original area) in a numerically
    # stable manner; Smaller ratios are better as this indicates more shrinkage
    lse_new_total_area = torch.logsumexp(log_new_areas, dim=0)
    lse_original_total_area = torch.logsumexp(log_original_areas, dim=0)

    if lse_new_total_area > log_underflow:
        # lse_new_total_area is never greater than lse_original_total_area, only need to check underflow on this
        # variable.
        new_total_area = f"{torch.exp(lse_new_total_area).item():.2e}"
        original_total_area = f"{torch.exp(lse_original_total_area).item():.2e}"
    else:
        new_total_area = original_total_area = "numerical_underflow"

    if new_total_area == "numerical_underflow":
        # Look more closely, is a single constraint causing this numerical underflow? Or is it a combination
        # of constraints?
        print(f"Clipping got numerical underflow, looking closer at constraints one at a time...")
        for i in range(num_spec):

            c_lA = lA[:, i:i + 1, :]
            c_thresholds = thresholds[:, i:i + 1]
            c_dm_lb = dm_lb[:, i:i + 1]

            test_x_L, test_x_U = _clip_main_fn(original_x_L, original_x_U, c_lA, None, c_thresholds, c_dm_lb,
                                               1, is_lower)

            test_nv_x_U = test_x_U[nv_mask]
            test_nv_x_L = test_x_L[nv_mask]

            test_log_new_areas = torch.log((test_nv_x_U - test_nv_x_L)).sum(1)

            # Uses log-sum-exp trick to calculate the ratio (total new area / total original area) in a numerically
            # stable manner; Smaller ratios are better as this indicates more shrinkage
            test_lse_new_total_area = torch.logsumexp(test_log_new_areas, dim=0)
            test_shrunken_ratio = torch.exp(test_lse_new_total_area - lse_original_total_area).item()
            test_new_area = torch.exp(test_lse_new_total_area).item()
            print(
                f"\t{i}: area new/original {test_new_area}/{torch.exp(lse_original_total_area).item():.2e} "
                f"({100 * test_shrunken_ratio:.2f}%)")

    shrunken_ratio = torch.exp(lse_new_total_area - lse_original_total_area).item()

    area_str = (f"Domain clipping (input dim {input_dim}, spec dim {num_spec}): "
                f"area new/original {new_total_area}/{original_total_area} ({100 * shrunken_ratio:.2f}%), ")
    log_area_str = f"log area new: {lse_new_total_area:2f} log area original: {lse_original_total_area:2f}, "
    domain_str = area_str + log_area_str
    print(domain_str)

def _clip_main_fn(x_L: Tensor, x_U: Tensor, lA:Tensor, lbias: Union[Tensor, None], thresholds: Tensor, dm_lb: Tensor,
                  num_iters: int, is_lower: int)->Tuple[Tensor, Tensor]:
    """

    The main clipping algorithm. Applies constraints one at a time in order to use a closed-form solution that can
    potentially clip off axis-aligned regions of the input box.

    :param x_L:         The lower bound on the inputs of the subdomains
    :param x_U:         The upper bound on the inputs of the subdomains
    :param lA:          CROWN lA for subdomains
    :param lbias:       CROWN lbias for subdomains
    :param thresholds:  The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param dm_lb:       The lower bound on the outputs of the domains
    :param num_iters:   Number of times to perform clipping as this may potentially improve clipping. Requires
                        lbias to be given.
    :param is_lower:    If set to true, we are currently lower bounding the network output, else we are upper
    :return:            The new x_L, x_U
    """
    sign = 1 if is_lower else -1
    batches, num_spec, _ = lA.shape
    # adds singleton dimension so that we can later broadcast multiplication/addition/subtraction operations along
    # this dimension
    thresholds = thresholds.reshape(batches, num_spec, 1)
    dm_lb = dm_lb.reshape(batches, num_spec, 1)
    for i in range(num_iters):
        # The order in which the constraints are applied affect clipping. We can repeatedly perform clipping to
        # take advantage of this sequential dependency.
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        if i > 0:
            # after each clip, we need to update the domain lower bound
            dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower).reshape(batches, num_spec, 1)

        # Solve for x across all batches, all specifications, and all input dimensions; The following three lines greatly
        # simplify and parallelize an operation that naively require 3 nested for-loops.
        eop = 'bsn,bn->bsn'  # specify multiplication broadcasting
        concrete_minus_one = dm_lb - torch.einsum(eop, lA, xhat) + sign * torch.einsum(eop, lA.abs(), eps)
        curr_x = (thresholds - concrete_minus_one) / lA
        # concrete_minus_one and curr_x: (b=batch, s=num_spec, n=input_dim)

        # Sort solutions appropriately
        x_U_candidates = torch.where(lA > 0, curr_x, torch.inf)
        x_L_candidates = torch.where(lA < 0, curr_x, -torch.inf)

        # Update new_x_U(L)
        x_U = torch.min(x_U_candidates.amin(dim=1), x_U)
        x_L = torch.max(x_L_candidates.amax(dim=1), x_L)

    return x_L, x_U

def clip_domains(
        x_L: Tensor,
        x_U: Tensor,
        thresholds: Tensor,
        lA: Tensor,
        lbias: Tensor,
        dm_lb: Optional[Tensor] = None,
        is_lower: bool = True,
        num_iters: int = 1,
        calculate_volume: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Takes a batch of subdomains and shrinks along dimensions to remove verified portions of the input domain
    to remove redundancy and allow for more effective splits.
    @param x_L:                 The lower bound on the inputs of the subdomains
    @param x_U:                 The upper bound on the inputs of the subdomains
    @param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    @param lA:                  CROWN lA for subdomains
    @param dm_lb:               The lower bound on the outputs of the domains
    @param lbias:               CROWN lbias for subdomains. Needed to concretize dm_lb if dm_lb is not given/incorrect
    @param is_lower:            If set to true, we are currently lower bounding the network output, else we are upper
    @param num_iters:           Number of times to perform clipping as this may potentially improve clipping. Requires
                                lbias to be given
    @return:                    The new x_L, x_U
    """
    # save original shapes
    x_L_shape = x_L.shape
    x_U_shape = x_U.shape
    # Get initial variables and correct views
    lA = lA.flatten(2)                 # (batch, num_spec, input_dim)
    # lbias shape: (batch, num_spec)
    batches, num_spec, input_dim = lA.shape

    x_L = x_L.clone().view(batches, input_dim) # (batch, input_dim)
    x_U = x_U.clone().view(batches, input_dim) # (batch, input_dim)
    xhat = (x_U + x_L) / 2             # (batch, input_dim)
    eps = (x_U - x_L) / 2              # (batch, input_dim)
    original_x_L = x_L.clone()         # (batch, input_dim)
    original_x_U = x_U.clone()         # (batch, input_dim)

    if lbias is not None:
        dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower) # (batch, num_spec)
    elif dm_lb is None:
        # if lbias is not given, we cannot concretize dm_lb, so we assume it is already correct
        assert lbias is not None, "lbias must be given if dm_lb is not given"

    v_mask = (dm_lb > thresholds).any(1)
    num_verified = v_mask.sum().item()
    nv_mask = torch.logical_not(v_mask)
    num_unverified = nv_mask.sum().item()

    if num_unverified == 0:
        print(f"All {batches} domains are already verified, no clipping will be performed.")
        return x_L.view(x_L_shape), x_U.view(x_U_shape)

    nv_batches = nv_mask.sum().item()  # number of domains in batch that are not verified

    # perform the main clipping algorithm
    x_L, x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters, is_lower)

    # extract the domains that were not already verified pre-clipping
    nv_x_L = x_L[nv_mask]
    nv_x_U = x_U[nv_mask]
    nv_ori_x_L = original_x_L[nv_mask]
    nv_ori_x_U = original_x_U[nv_mask]

    # on these unverified domains, calculate the number of domains that became verified after clipping
    # (this scenario is only possible if there are at least 2 constraints, otherwise it will always be 0)
    if num_spec > 1:
        sav_mask = (nv_x_L > nv_x_U).any(1)
        shrunken_and_verified = sav_mask.sum().item()
        if calculate_volume:
            # nv_mask holds domains that were not verified before clipping, but should now be updated
            # to hold domains that were not verified after clipping. If not updated, NaN will occur when
            # performing volume calculations on this batch due to the presence of negative values
            # in the new input domain.
            nv_mask[nv_mask.clone()] = torch.logical_not(sav_mask)
    else:
        shrunken_and_verified = 0

    # on these unverified domains, calculate the number that were clipped along any dimension
    num_shrunken = torch.logical_or((nv_x_L > nv_ori_x_L).any(1), (nv_x_U < nv_ori_x_U).any(1)).sum(0).item()

    shrunken_str = (f"Domain clipping ({num_iters} iterations): shrunken {num_shrunken}/{nv_batches} "
                    f"({100 * (num_shrunken / nv_batches) if nv_batches > 0 else 0.:.2f}%), ")
    batch_str = (f"Of {batches} batches, {num_verified} were verified before clipping, {num_shrunken} domains "
                 f"were shrunken, and {shrunken_and_verified} were shrunken and additionally verified.")
    domain_str = shrunken_str + batch_str
    print(domain_str)

    if calculate_volume:
        if nv_mask.sum() > 0:
            # calculate the total volume of this batch of subdomains before and after clipping
            _in_depth_volume_metrics(original_x_L, original_x_U, x_L, x_U, nv_mask, lA, thresholds, dm_lb, is_lower)
        else:
            print(f"Domain clipping (input dim {input_dim}, spec dim {num_spec}): No volume calculations to perform; "
                  f"Some (potentially none) domains were verified before clipping, but all domains became verified "
                  f"after clipping.")

    # reshape x_L,x_U to originally given shape and discover how many batches were shrunken
    return x_L.view(x_L_shape), x_U.view(x_U_shape)

def concretize_bounds(xhat: Tensor, eps: Tensor, lA: Tensor, lbias: Tensor, is_lower: bool = True):
    """
    Given a batch of domains and hyperplanes, uses Hölder's inequality to concretize the lower bounds for L-inf norm.
    :param xhat:        Domain center
    :param eps:         Domain half-widths
    :param lA:          Hyperplane coefficients
    :param lbias:       Hyperplane offsets
    :param is_lower:    If true, these parameters are for the domain lower bound, else upper bound.
    :return:
    """
    sign = 1 if is_lower else -1  # sign based on if the network is being lower/upper bounded
    # use lbias to concretize dm_lb for the subdomains
    eop = 'bsn,bn->bs'  # specify batch matrix multiplication
    # b = batch, s = num_spec, n = input_dim
    dm_lb = torch.einsum(eop, lA, xhat) - sign * torch.einsum(eop, lA.abs(), eps) + lbias
    return dm_lb

def deconstruct_bias(x_L: Tensor, x_U: Tensor, A: Tensor, dm_ob: Tensor, is_lower: bool = True) -> Tensor:
    """
    Deconstructs the bias parameter given the A matrix and domain lower/upper bounds.
    :param x_L:         Domain input lower bound
    :param x_U:         Domain input upper bound
    :param A:           Hyperplane coefficients
    :param dm_ob:       Domain output lower/upper bounds
    :param is_lower:    If true, these parameters are for the domain lower bound, else upper bound.
    :return:
    """
    sign = 1 if is_lower else -1  # sign based on if the network was being lower/upper bounded
    A = A.flatten(2) # (batch, num_spec, in_dim)
    xhat_vect = ((x_U + x_L) / 2).flatten(1).unsqueeze(2) # (batch, in_dim, 1)
    eps_vect = ((x_U - x_L) / 2).flatten(1).unsqueeze(2) # (batch, in_dim, 1)
    dm_ob_vect = dm_ob.unsqueeze(2) # (batch, num_spec, 1)
    bias = dm_ob_vect - (A.bmm(xhat_vect) - sign*A.abs().bmm(eps_vect))
    return bias.squeeze(2) # (batch, num_spec)

def check_lbias(x_L: Tensor, x_U: Tensor, lA: Tensor, lbias: Union[Tensor, None], dm_lb: Tensor, thresholds: Tensor):
    """

    For development purposes. If it is imperative that the lA and lbias from CROWN correspond to the dm_lb,
    this function can check this (e.g. for clipping to be sound, if lA and/or lbias is incorrect, clipping will
    be unsound and remove valid regions, so this can be a valuable check).

    :param x_L:         Domain input lower bound
    :param x_U:         Domain input upper bound
    :param lA:          Hyperplane coefficients
    :param lbias:       Hyperplane offsets
    :param dm_lb:       Domain lower bound
    :param thresholds:  The specification threshold where dom_lb > thresholds implies the subdomain is verified
    """

    if lbias is None:
        # cannot check that lA and lbias is consistent with dm_lb if lbias is not given
        return

    # flatten shapes
    lbias = lbias.flatten(1)
    lA = lA.flatten(2)
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    dm_lb = dm_lb.flatten(1)
    thresholds = thresholds.flatten(1)

    # Only check lbias for unverified domains. lA and lbias are allowed to be incorrect for verified subdomains
    unverified_mask = (dm_lb <= thresholds).all(1)

    # reconstruct lbias to compare with the lbias returned from CROWN
    lbias_check = deconstruct_bias(x_L, x_U, lA, dm_lb)
    # from the unverified domains, get a mask of all the domains whose lbias do not match 'lbias_check'
    same = torch.isclose(lbias[unverified_mask], lbias_check[unverified_mask], atol=1e-4).all(1)
    check = same.all().item()

    if not check:
        # print some additional info before throwing an error

        # print the lbias and lbias_check for the first 10 subdomains that have a mismatch
        not_same = torch.logical_not(same)
        num_mismatch = not_same.sum().item()
        print(f"{num_mismatch} domains have a mismatch in lbias. First 10 domains with mismatches: "
              f"\nlbias:\n{lbias[unverified_mask][not_same][:10]}"
              f"\nlbias_check:\n{lbias_check[unverified_mask][not_same][:10]}")

        # if lbias from CROWN and deconstructed lbias do not match, check that the dm_lb can be recomputed using
        # lA and lbias from CROWN
        xl_nv = x_L[unverified_mask]
        xu_nv = x_U[unverified_mask]
        xhat = (xu_nv + xl_nv) / 2
        eps = (xu_nv - xl_nv) / 2
        lA_nv = lA[unverified_mask]
        lbias_nv = lbias[unverified_mask]
        # for the unverified domains, concretize the dm_lb using lbias from CROWN
        lb_with_lbias = concretize_bounds(xhat, eps, lA_nv, lbias_nv)
        dm_lb_nv = dm_lb[unverified_mask]
        # check if the given dm_lb matches the dm_lb we calculted
        same_with_lbias = torch.allclose(dm_lb_nv, lb_with_lbias, atol=1e-4)
        print(f"Can we recover dm_lb using lA and lbias from CROWN? {same_with_lbias}")

        assert check, "The lbias returned from bounding does not match the computed lbias"
