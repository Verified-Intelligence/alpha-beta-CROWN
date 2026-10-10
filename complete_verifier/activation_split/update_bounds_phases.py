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
"""
Implementation of computing alpha-beta-CROWN bounds given selected BaB domains.

* update_bounds_pre: Must be GPU-light. Pre-processing logic for update_bounds in beta-CROWN.
* update_bounds_core: Must be GPU-heavy. Core logic for update_bounds in beta-CROWN.
* update_bounds_post: Must be GPU-light. Post-processing logic for update_bounds in beta-CROWN.


Break LiRPANet.update_bounds into _pre, _core, _post functions to isolate CPU and GPU workloads.
The old LiRPANet.update_bounds is kept as a legacy with guarded calls.

Where GPU workload should only live in update_bounds_core.
"""

import torch
from heuristics.decision_types import BatchFirstBranchingDecisions

from typing import TYPE_CHECKING, Optional
import arguments

from input_split.input_split_on_relu_domains import input_branching_decisions
from domain_clipper import (
    ClipDecisions,
    DomainClipScorer,
)
from state import AlphaValueData, BetaFullData
from state.alpha import resolve_alpha_storage_dtype
from utils import expand_batch, Timer

from activation_split.return_types import (
    UpdateBoundCoreContext,
    UpdateBoundCoreReturn,
    UpdateBoundPreReturn,
    UpdateBoundPostReturn,
)
from activation_split.debug_utils import _check_mixed_precision_bound_soundness


if TYPE_CHECKING:
    from ..beta_CROWN_solver import LiRPANet


@torch.no_grad()
def _expand_tensors_impl(d, batch, final_name, net_c, net_x):
    lb, ub = d["lower_bounds"], d["upper_bounds"]
    cs, x_Ls, x_Us = d.get("cs", None), d.get("x_Ls", None), d.get("x_Us", None)
    # Only the last element is used later.
    lb_last, ub_last = lb[final_name], ub[final_name]
    interm_bounds = {k: [lb[k], ub[k]] for k in lb if k != final_name}
    # create new_x here since batch may change
    new_x = expand_batch(net_x, batch, x_L=x_Ls, x_U=x_Us)
    if cs is None:
        assert net_c is not None
        assert net_c.size(0) == 1
        cs = net_c.expand(new_x.shape[0], -1, -1)
    return interm_bounds, lb_last, ub_last, cs, new_x, x_Ls, x_Us


def update_bounds_pre(
    d,
    final_name: str,
    net_c,
    net_x,
    dtype_map: dict[str, torch.dtype],
    *,
    timer: Timer,
    device,
    beta_bias,
) -> UpdateBoundPreReturn:
    """
    Get all the needed tensor for update_bounds_core, re-organize and/or transfer to GPU.

    The needed Tensors are defined in UpdateBoundPreReturn.

    This function aims to do all pre-processing and preparation work,
    so that update_bounds_core can enter GPU-intense workload without delay.
    
    Arguments:
    - d: the dict of a batch of domains
    - final_name: the name of the final layer
    - net_c: the C matrix for the final layer. 
    - net_x: the input x of the 
    - dtype_map: computation dtype owned by each split node
    - timer: a Timer object
    - device: device that contruct alpha and betas to
    - beta_bias: mark if there is non-zero bias term in beta
    """

    # ====================== extract args ===================
    # alpha is always enabled for all current test suites.
    enable_alpha = True
    solver_args = arguments.Config["solver"]
    beta_args = solver_args["beta-crown"]
    enable_beta = beta_args["beta"]
    enable_opt_interm_bounds = beta_args["enable_opt_interm_bounds"]

    assert enable_opt_interm_bounds is False

    timer.start("func")
    timer.start("prepare")

    batch_size = d["lower_bounds"][final_name].shape[0]

    # ============= Extract needed object from d ====================

    ret = _expand_tensors_impl(d, batch_size, final_name, net_c, net_x)
    interm_bounds, lb_last, ub_last, c, new_x, x_Ls, x_Us = ret
    new_x_Ls, new_x_Us = None, None

    # ============= Preprocess alpha, beta, decision ================

    if enable_alpha:
        alphas_by_layer = AlphaValueData.from_domain_dict(d)
        # data within d is already transferred to device
    else:
        alphas_by_layer = ValueError("alphas_by_layer is not computed")

    if enable_beta:
        betas_by_layer, nums_effective_beta_per_domain = BetaFullData.from_domain_dict(
            d,
            bias=beta_bias,
            dtype_map=dtype_map,
            device=device
        )

    else:
        betas_by_layer = ValueError("betas_by_layer is not computed in pre")
        nums_effective_beta_per_domain = ValueError(
            "nums_effective_beta_per_domain is not computed in pre"
        )

    clip_decisions = (
        ClipDecisions.reconstruct_from_sub_domain_clip_decisions(
            d["sub_domain_clip_decisions"]
        )
    ).to_device(device=device)

    timer.add("prepare")

    return UpdateBoundPreReturn(
        interm_bounds=interm_bounds,
        lb_last=lb_last,
        ub_last=ub_last,
        c=c,
        new_x=new_x,
        x_Ls=x_Ls,
        x_Us=x_Us,
        new_x_Ls=new_x_Ls,
        new_x_Us=new_x_Us,
        nums_effective_beta_per_domain=nums_effective_beta_per_domain,
        alphas_by_layer=alphas_by_layer,
        betas_by_layer=betas_by_layer,
        beta_bias=beta_bias,
        clip_decisions=clip_decisions,
        d_dict=d,
        batch_size=batch_size,
    )


def update_bounds_core(
    net: "LiRPANet",
    pre_result: UpdateBoundPreReturn,
    fix_interm_bounds,
    stop_criterion_func,
    *,
    multi_spec_keep_func,
    branching_heuristic,
    precompute_bfs_flag,
    batch_device_limit,
    is_multitree_bab,
    domain_clip_scorer: Optional[DomainClipScorer],
    iter_idx: int,
    enable_clip_domains: bool,
    enable_decision_precompute: bool,
    visited_num: int,
) -> UpdateBoundCoreReturn:
    """
    Compute the alpha-beta-CROWN bounds for the given domains.

    This function aims to perform all GPU-intense workload,
    and it should offload any tensor transfer or re-organization work to
    update_bounds_pre or update_bounds_post.
    """

    net.timer.start("core")
    ###### 1. Extract options and data
    beta_args = arguments.Config["solver"]["beta-crown"]
    bab_args = arguments.Config["bab"]
    # The dtype spec ('fp16', 'bf16', 'tf32', 'fp16+tf32', 'bf16+tf32') is
    # parsed by auto_LiRPA (see parse_mixed_precision_dtype); 'none' disables
    # mixed precision.
    enable_bab_bound_mixed_precision = bab_args["mixed_precision"][
        "enable_bab_bound_mixed_precision"]
    dtype_spec = bab_args["mixed_precision"]["bab_bound_mixed_precision_dtype"]
    if not enable_bab_bound_mixed_precision or dtype_spec in ("none", ""):
        bab_bound_dtype = None
    else:
        bab_bound_dtype = dtype_spec
    prune_even_under_low_precision = bab_args["mixed_precision"][
        "prune_even_under_low_precision"]
    check_soundness = bab_args["mixed_precision"]["check_mixed_decision_bound_soundness"]
    # alpha is always enabled for all current test suites.
    enable_alpha = True
    enable_beta = beta_args["beta"]
    iteration = beta_args["iteration"]
    enable_opt_interm_bounds = beta_args["enable_opt_interm_bounds"]
    get_upper_bound = bab_args["get_upper_bound"]
    branching_input_and_activation = bab_args["branching"]["branching_input_and_activation"]
    enable_cut = bab_args["cut"]["enabled"]
    enable_biccos = bab_args["cut"]["biccos"]["enabled"]
    all_node_split_LP = beta_args["all_node_split_LP"]

    (
        interm_bounds,
        lb_last,
        ub_last,
        c,
        new_x,
        x_Ls,
        x_Us,
        new_x_Ls,
        new_x_Us,
        nums_effective_beta_per_domain,
        clip_decisions,
        d,
        beta_bias,
        betas_by_layer,
        alphas_by_layer,
        batch_size,
    ) = (
        pre_result.interm_bounds,
        pre_result.lb_last,
        pre_result.ub_last,
        pre_result.c,
        pre_result.new_x,
        pre_result.x_Ls,
        pre_result.x_Us,
        pre_result.new_x_Ls,
        pre_result.new_x_Us,
        pre_result.nums_effective_beta_per_domain,
        pre_result.clip_decisions,
        pre_result.d_dict,
        pre_result.beta_bias,
        pre_result.betas_by_layer,
        pre_result.alphas_by_layer,
        pre_result.batch_size,
    )
    decision_thresh = d["thresholds"]

    ######## 2. Configure BoundedModule Object, binding data to BoundedModule
    net.timer.start("accept_alpha_beta")
    if enable_beta:
        assert not isinstance(betas_by_layer, ValueError)
        betas_by_layer.attach_to_net(net)

        # iteration may vary due to set_cut_params. need to preprocess here.
        # even we need to use cut, maybe the cut is not fetched yet
        net.net.cut_used = (
            arguments.Config["bab"]["cut"]["enabled"]
            and arguments.Config["bab"]["cut"]["bab_cut"]
            and getattr(net.net, "cut_module", None) is not None
        )
        net.timer.start("set_cut_params")
        if net.net.cut_used:
            iteration = net.set_cut_params(
                batch_size, batch_size, d.get("split_history", None)
            )
        net.timer.add("set_cut_params")
        # here to handle the case where the split node happen to be in the cut constraint !!!

    if enable_alpha:
        assert not isinstance(alphas_by_layer, ValueError)
        alphas_by_layer.attach_to_net(net)
        # set_all option is ignored because alphas_by_layer contains data for
        # exactly the nodes that are needed.

    net.timer.add("accept_alpha_beta")

    net.timer.start("bound")

    # TODO: unify this with set_crown_bound_opts
    net.net.set_bound_opts(
        {
            "optimize_bound_args": {
                "enable_beta_crown": enable_beta,
                "fix_interm_bounds": fix_interm_bounds,
                "stop_criterion_func": stop_criterion_func,
                "multi_spec_keep_func": multi_spec_keep_func,
                "iteration": iteration,
                "mixed_precision_dtype": bab_bound_dtype,
                "low_precision_prune_in_iter": prune_even_under_low_precision,
                "low_precision_prune_margin_abs":
                    bab_args["mixed_precision"]["low_precision_prune_margin_abs"],
            },
            "enable_opt_interm_bounds": enable_opt_interm_bounds,
        }
    )
    net.set_crown_bound_opts("beta")

    # we need A matrix to construct adv example
    temp_return_A = get_upper_bound or branching_input_and_activation
    temp_needed_A_dict = net.needed_A_dict
    if enable_clip_domains and net.domain_clipper.using_final_layer:
        temp_return_A, temp_needed_A_dict = net._set_tmp_A(True, "alpha-crown")
    original_size = new_x.shape[0]

    if fix_interm_bounds:
        reference_bounds = {}
        for name in net.alpha_start_nodes:
            if name in interm_bounds:
                reference_bounds[name] = interm_bounds[name]
                interm_bounds.pop(name)
    else:
        reference_bounds = interm_bounds
        interm_bounds = {}
    if len(reference_bounds):
        print(
            "Recompute intermediate bounds for nodes:",
            ", ".join(list(reference_bounds.keys())),
        )

    ############# 3. Computing Bounds, including clip and compute_bounds

    ######### Clip and Verify Domains Start ########
    if enable_clip_domains and net.domain_clipper is not None:
        if net.domain_clipper.clip_input_domain:
            new_x, interm_bounds, d, batch_mask = (
                net.domain_clipper.domain_clip_ReLU(d, new_x, interm_bounds))
            if net.domain_clipper.prune and batch_mask is not None:
                ret_prune = net.prune_setting(
                    d,
                    enable_beta,
                    beta_bias,
                    lb_last,
                    ub_last,
                    batch_mask,
                    enable_opt_interm_bounds,
                )
                (c, decision_thresh, _, lb_last, ub_last, nums_effective_beta_per_domain) = (
                    ret_prune
                )

        if net.domain_clipper.clip_interm_domain:
            if enable_decision_precompute:
                interm_bounds = net.domain_clipper.optimize_interm_bounds(
                    d,
                    new_x.ptb.x_L,
                    new_x.ptb.x_U,
                    interm_bounds,
                    net.split_activations,
                    clip_decisions_ref=clip_decisions,
                )
            else:
                interm_bounds = net.domain_clipper.optimize_interm_bounds(
                    d,
                    new_x.ptb.x_L,
                    new_x.ptb.x_U,
                    interm_bounds,
                    net.split_activations,
                )
    ######### Clip and Verify Domains End ##########

    net.timer.start("update_bounds.compute_bounds")
    tmp_ret = net.net.compute_bounds(
        x=(new_x,),
        C=c,
        method="CROWN-optimized",
        interm_bounds=interm_bounds,
        reference_bounds=reference_bounds,
        return_A=temp_return_A,
        needed_A_dict=temp_needed_A_dict,
        cutter=net.cutter,
        bound_upper=False,
        decision_thresh=decision_thresh,
    )

    A = tmp_ret[2] if temp_return_A else None
    lb, _ = tmp_ret[0], tmp_ret[1]

    if check_soundness:
        _check_mixed_precision_bound_soundness(
            net, new_x, c, enable_opt_interm_bounds, lb, decision_thresh)


    # Using output constraints to clip input region.
    # TODO: clean up implementation, and make it more general.
    if enable_clip_domains:
        assert net.domain_clipper is not None
        if net.domain_clipper.using_final_layer:
            new_x_Ls, new_x_Us, interm_bounds = net.domain_clipper.domain_clip_outputs(
                A, new_x, interm_bounds
            )

    if get_upper_bound:
        primal_x, ub = net.get_primal_upper_bound(A)
    else:
        ub = torch.full_like(lb, fill_value=torch.inf)  # dummy upper bound
        primal_x = None
    # Use A matrix of the input, the find best neuron to branch in input space.
    input_split_idx = (
        input_branching_decisions(
            net.net,
            lb,
            A[net.net.output_name[0]][net.net.input_name[0]]["lA"],
            x_Ls,
            x_Us,
            decision_thresh,
        )
        if branching_input_and_activation
        else None
    )

    core_ctx = UpdateBoundCoreContext(
        temp_return_A=temp_return_A,
        lb=lb,
        ub=ub,
        lb_last=lb_last,
        ub_last=ub_last,
        primal_x=primal_x,
        nums_effective_beta_per_domain=nums_effective_beta_per_domain,
        input_split_idx=input_split_idx,
        new_x_Ls=new_x_Ls,
        new_x_Us=new_x_Us,
        c=c,
        original_size=original_size,
        x_Ls=x_Ls,
        x_Us=x_Us,
        decision_thresh=decision_thresh,
        domain_clip_scorer=domain_clip_scorer,
        branching_heuristic=branching_heuristic,
        batch_device_limit=batch_device_limit,
        precompute_bfs_flag=precompute_bfs_flag,
        is_multitree_bab=is_multitree_bab,
        recorder=net.recorder,
    )

    net.timer.add("update_bounds.compute_bounds")
    net.timer.add("bound")

    ############### 4. Extracting Results from BoundedModule
    net.timer.start("extract")

    if not net.solver_model_initialized:
        if all_node_split_LP and torch.any(
            torch.tensor(d["depths"]) == net.tot_ambi_nodes
        ):
            net.initialize_lp_solver_for_bab()
            net.solver_model_initialized = True

    interm_bounds.clear() # Save memory
    pre_result.interm_bounds.clear() # Save memory

    if enable_decision_precompute:
        # 1. Perform clip decision and split decision precompute.
        # 2. Perform BICCOS if enabled because BICCOS requires verified domains.
        # 3. Remove verified domains before return to save memory.

        if enable_cut and enable_biccos:
            ret = net.update_bounds_precompute_biccos_extract(
                d=d,
                alpha=enable_alpha,
                beta=enable_beta,
                enable_clip_domains=enable_clip_domains,
                enable_cut=enable_cut,
                enable_biccos=enable_biccos,
                iter_idx=iter_idx,
                visited_num=visited_num,
                core_ctx=core_ctx,
            )
        else:

            ret = net.update_bounds_precompute_extract(
                d=d,
                alpha=enable_alpha,
                beta=enable_beta,
                enable_clip_domains=enable_clip_domains,
                core_ctx=core_ctx,
            )
    else:
        # Prepare return value without masking or precomputing.
        ret = net.update_bounds_extract_no_mask_no_precompute(
            alpha=enable_alpha,
            beta=enable_beta,
            enable_opt_interm_bounds=enable_opt_interm_bounds,
            core_ctx=core_ctx,
            depths=d["depths"],
            history=d["history"],
        )

    net.timer.add("extract")

    net.timer.add("core")
    return ret


def update_bounds_post(
    core_result: UpdateBoundCoreReturn,
    timer,
    final_name: str,
    split_node_names: list[str],
    *,
    layers_requiring_bounds_names: list[str],
    unstable_mask: dict,
    interm_transfer: bool,
) -> UpdateBoundPostReturn:
    """
    Post process the result from update_bounds_core, translate them
    into the format for DomainList.add

    This function aims to do all the offloaded tensor work from update_bounds_core.
    """

    solver_args = arguments.Config["solver"]
    beta_args = solver_args["beta-crown"]
    bab_args = arguments.Config["bab"]
    # alpha is always enabled for all current test suites.
    enable_alpha = True
    enable_beta = beta_args["beta"]
    reproduce_alpha_bounds = solver_args["reproduce_alpha_bounds"]
    get_upper_bound = bab_args["get_upper_bound"]

    (
        lb,
        ub,
        lb_last,
        ub_last,
        nums_effective_beta_per_domain,
        input_split_idx,
        primal_x,
        x_Ls,
        x_Us,
        new_x_Ls,
        new_x_Us,
        c,
        working_alpha,
        working_beta,
        working_interm_bounds,
        batched_lA,
        new_split_history,
    ) = (
        core_result.lb,
        core_result.ub,
        core_result.lb_last,
        core_result.ub_last,
        core_result.nums_effective_beta_per_domain,
        core_result.input_split_idx,
        core_result.primal_x,
        core_result.x_Ls,
        core_result.x_Us,
        core_result.new_x_Ls,
        core_result.new_x_Us,
        core_result.c,
        core_result.working_alpha,
        core_result.working_beta,
        core_result.working_interm_bounds,
        core_result.batched_lA,
        core_result.new_split_history,
    )

    with torch.no_grad():
        # Move tensors to CPU for all elements in this batch.
        timer.start("transfer")
        lb, ub = lb.to(device="cpu"), ub.to(device="cpu")
        lAs = batched_lA.to_device(device="cpu")
        sub_domain_clip_decisions = (
            core_result.sub_domain_clip_decisions.to_device(device="cpu")
        )
        timer.add("transfer")
        timer.start("finalize")
        if enable_alpha:
            assert not isinstance(working_alpha, ValueError)
            alpha_storage_dtype = resolve_alpha_storage_dtype(
                arguments.Config["solver"]["alpha-crown"]["alpha_dtype"],
                reproduce_alpha_bounds=reproduce_alpha_bounds,
            )
            ret_alphas = working_alpha.to(
                device="cpu", dtype=alpha_storage_dtype
            )
        else:
            ret_alphas = AlphaValueData({})

        if enable_beta:
            assert not isinstance(nums_effective_beta_per_domain, ValueError)
            assert not isinstance(working_beta, ValueError)
            ret_betas = working_beta.to_domain_dict(
                nums_effective_beta_per_domain, device="cpu"
            )
        else:
            ret_betas = []
        # Reorganize tensors.
        if ub is None:
            ub = lb + torch.inf
        ret_l = {final_name: lb.detach().to("cpu")}
        ret_u = {final_name: ub.detach().to("cpu")}
        if interm_transfer:
            unstable_bounds = working_interm_bounds.to_unstable_bounds(
                unstable_mask=unstable_mask,
                split_nodes_names=split_node_names,
                layers_requiring_bounds_names=layers_requiring_bounds_names,
                device="cpu",
            )
        else:
            unstable_bounds = {}
        if not reproduce_alpha_bounds:
            ret_l[final_name] = torch.max(ret_l[final_name], lb_last.cpu())
            if not get_upper_bound:
                # Do not set to min so the primal is always corresponding
                # to the upper bound.
                ret_u[final_name] = torch.min(ret_u[final_name], ub_last.cpu())
        timer.add("finalize")

    batch_first_branching_decisions = BatchFirstBranchingDecisions.from_branching_decision(
        branching_decision=core_result.branching_decision,
    )

    timer.add("func")
    timer.print()

    print("max lb", core_result.lb_final_max, "min lb", core_result.lb_final_min)
    print(
        f"Number of Verified Splits: {core_result.n_verified} of {core_result.n_splits}"
    )

    return UpdateBoundPostReturn(
        lower_bounds=ret_l,  # dict of tensor
        upper_bounds=ret_u,  # dict of tensor
        lAs=lAs,  # dict of tensor
        alphas=ret_alphas,  # dict of tensor
        betas=ret_betas,  # dict of tensor
        split_history=new_split_history,  # list of dict
        unstable_bounds=unstable_bounds,  # dict of tensor
        primals=primal_x,  # always None
        c=c,  # tensor
        x_Ls=x_Ls if new_x_Ls is None else new_x_Ls,  # always None. useless
        x_Us=x_Us if new_x_Us is None else new_x_Us,  # always None. useless
        input_split_idx=input_split_idx,  # always None. useless
        decision_info=batch_first_branching_decisions,  # BatchFirstBranchingDecisions
        sub_domain_clip_decisions=sub_domain_clip_decisions,
        # SubDomainClipDecisions
    )
