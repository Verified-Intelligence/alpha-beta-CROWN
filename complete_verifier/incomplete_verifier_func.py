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
"""incomplete verifier main interface."""

import torch
from enum import Enum

import arguments
from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any, stop_criterion_general
from auto_LiRPA.operators.convolution import BoundConv
from beta_CROWN_solver import LiRPANet
from specifications import vnnlibHandler
from utils import print_model
from utils import pad_list_of_input_to_tensor, unpad_to_list_of_tensors, take_batch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abcrown import ABCROWN


def incomplete_verifier(
    self: "ABCROWN", model_ori, interm_bounds=None
):
    # TODO: clean up the implementation of invprop
    # -------------------- invprop start --------------------
    tighten_input_bounds = arguments.Config["solver"]["invprop"]["tighten_input_bounds"]
    apply_output_constraints_to = arguments.Config["solver"]["invprop"][
        "apply_output_constraints_to"
    ]
    # --------------------- invprop end ---------------------

    spec_handler = SpecHandler(self.vnnlib_handler)
    x, c, rhs = spec_handler.x, spec_handler.c, spec_handler.rhs
    or_spec_size = spec_handler.or_spec_size
    stop_criterion = spec_handler.stop_criterion

    model = LiRPANet(model_ori, in_size=[1, *x.shape[1:]])
    print_model(model.net)
    self.spec_handler_incomplete = spec_handler

    output = model.net(x[-1:].to(model.net.device))
    print("Original output in the first batch:", output)

    if arguments.Config["general"]["save_output"]:
        arguments.Globals["out"]["pred"] = output.cpu()

    # -------------------- invprop start --------------------
    if len(apply_output_constraints_to) > 0:
        assert arguments.Config["solver"]["bound_prop_method"] == "alpha-crown"
        assert spec_handler.spec_type == SpecType.SINGLE_OR or spec_handler.spec_type == SpecType.SINGLE_AND_IN_MULTI_ORS
        # TODO: support other spec types. rhs should not be flattened.
        model.net.constraints = c
        model.net.thresholds = rhs.flatten()

        # We need to use matrix mode for the layer that should utilize output constraints
        for node in model.net.nodes():
            if node.are_output_constraints_activated_for_layer(apply_output_constraints_to):
                if isinstance(node, BoundConv) and node.mode == "patches":
                    node.mode = "matrix"
    # --------------------- invprop end ---------------------

    # main function to build the model and perform incomplete verification
    global_lb, ret = model.build(
        x, c, rhs, stop_criterion, vnnlib_handler=self.vnnlib_handler,
        interm_bounds=interm_bounds, or_spec_size=or_spec_size,
        full_alpha_info=True)

    if arguments.Config["general"]["return_optimized_model"]:
        return model, {}

    # check if there is any counterexample. If yes, return immediately.
    # counterexample should have been saved, we just need to return the status.
    if arguments.Config["attack"]["pgd_order"] == "middle":
        if "attack_examples" in ret:
            return "unsafe-pgd", {}

    spec_handler.set_unverified_or_mask(global_lb)

    # check if all ORs are verified. If yes, return immediately.
    # Since all ORs are verified, we just need to return the status.
    if not spec_handler.unverified_or_mask.any():
        print("verified with init bound!")
        return "safe-incomplete", {}

    ret = spec_handler.post_process(model, ret)

    # -------------------- invprop start --------------------
    if tighten_input_bounds:
        perturbed_root = None
        for root in model.net.roots():
            if hasattr(root, "perturbation") and root.perturbation is not None:
                assert (
                    perturbed_root is None
                ), "BaB based on tightened bounds currently supports only one input layer"
                perturbed_root = root
        assert perturbed_root is not None
        self.vnnlib_handler.update_input_bounds(
            perturbed_root.perturbation.x_L.detach(),
            perturbed_root.perturbation.x_U.detach(),
        )
    # --------------------- invprop end ---------------------
    return "unknown", ret


class SpecType(Enum):
    # Case 1: single OR with single AND / multiple ANDs, e.g., A; A and B and C
    SINGLE_OR = 1
    # Case 2: multiple ORs with single AND, e.g., A or B or C
    SINGLE_AND_IN_MULTI_ORS = 2
    # Case 3: multiple ORs with same number (>1) of ANDs, e.g., (A and B) or (C and D)
    FIXED_NUM_ANDS_IN_MULTI_ORS = 3
    # Case 4: multiple ORs with different numbers of ANDs, e.g., (A) or (B and C)
    VARIABLE_NUM_ANDS_IN_EACH_OR = 4


class PostProcessingType(Enum):
    # No post-processing
    NONE = 1
    # Unflattening c and rhs
    RESHAPE = 2
    # Splitting c and rhs
    SPLIT_PAD = 3

# Specification Handler for incomplete verification
class SpecHandler:
    # Type of specification
    spec_type: SpecType
    # Whether the input range is the same for all disjuncts
    same_x_range: bool
    # Handling strategy: jointly (efficient but weak) or separately (strong but inefficient)
    optimize_disjuncts_separately: bool
    # Number of ANDs per OR
    or_spec_size: torch.Tensor
    # Stop criterion function
    stop_func: callable
    # Post-processing type
    post_processing_type: PostProcessingType

    def __init__(
        self,
        vnnlib_handler: vnnlibHandler,
    ):
        solver_args = arguments.Config["solver"]
        prune_after_crown = solver_args["prune_after_crown"]
        bounding_method = solver_args['bound_prop_method']
        prune_after_crown = prune_after_crown and bounding_method == "alpha-crown"
        apply_output_constraints_to = solver_args["invprop"]["apply_output_constraints_to"]
        if arguments.Config['general']['store_all_specs_on_cpu']:
            device = "cpu"
        else:
            device = arguments.Config['general']['device']
        (
            x,
            c,
            rhs,
            or_spec_size,
            same_x_range,
            same_or_spec_size,
        ) = vnnlib_handler.all_specs.get(device=device)

        # if the input range is not shared, then intermediate bounds cannot be shared.
        # so fix optimize_disjuncts_separately as True
        if not same_x_range:
            print("Input range is not shared, setting optimize_disjuncts_separately to True.")
            solver_args["optimize_disjuncts_separately"] = True
        # if the input range is shared, and we want to solve all disjuncts together,
        # we will share intermediate bounds by keeping batch size as 1
        self.optimize_disjuncts_separately = solver_args["optimize_disjuncts_separately"]

        num_or = or_spec_size.shape[0]

        if num_or == 1:
            # Case 1: single OR with single AND / multiple ANDs
            # there is no difference between share intermediate bounds or not.
            self.spec_type = SpecType.SINGLE_OR
            self.post_processing_type = PostProcessingType.NONE
            stop_criterion = stop_criterion_batch_any
        elif same_or_spec_size and or_spec_size[0] == 1:
            # Case 2: multiple ORs with single AND
            # NOTE: Case 2 can be merged in Case 3, but there are other components have not support Case 3 as Case 2,
            # so we keep it as a separate case for now.
            self.spec_type = SpecType.SINGLE_AND_IN_MULTI_ORS
            if self.optimize_disjuncts_separately:
                # c: [num_or, num_and (1), num_output]
                # rhs: [num_or, num_and (1)]
                self.post_processing_type = PostProcessingType.NONE
                # for every OR, verify any AND (specs along dim 1) to stop
                stop_criterion = stop_criterion_batch_any
            else:
                # x: [1, *input_shape]
                # c: [1, num_clause, num_output]
                # rhs: [1, num_clause]. num_clause = num_or * num_and (1).
                x = take_batch(x, 1, 0)
                c = c.view(1, -1, c.shape[-1])
                rhs = rhs.view(1, -1)
                self.post_processing_type = PostProcessingType.RESHAPE
                # verify all ORs (all specs along dim 1) to stop
                stop_criterion = stop_criterion_all
                # invprop code
                assert len(apply_output_constraints_to) == 0, (
                    "To apply output constraints, set --optimize_disjuncts_separately"
                )
        elif same_or_spec_size and or_spec_size[0] > 1:
            # Case 3: multiple ORs with same number (>1) of ANDs
            self.spec_type = SpecType.FIXED_NUM_ANDS_IN_MULTI_ORS
            if self.optimize_disjuncts_separately:
                # c: [num_or, num_and, num_output]
                # rhs: [num_or, num_and]
                self.post_processing_type = PostProcessingType.NONE
                # for every OR, verify any AND clause to stop
                stop_criterion = stop_criterion_batch_any
            else:
                # x: [1, *input_shape]
                # c: [1, num_clause, num_output]
                # rhs: [1, num_clause]. num_clause = num_or * num_and.
                x = take_batch(x, 1, 0)
                c = c.view(1, -1, c.shape[-1])
                rhs = rhs.view(1, -1)
                self.post_processing_type = PostProcessingType.RESHAPE
                # verify all ORs to stop; for every OR, verify any AND clause to stop
                # in this case, the structure of OR-AND is flattened in the dim 1,
                # we need a new general stop criterion function.
                stop_criterion = stop_criterion_general
            # invprop code
            assert len(apply_output_constraints_to) == 0, (
                "Applying output constraints is not tested for multiple ORs with same number (>1) of ANDs."
            )
        else:
            # Case 4: multiple ORs with different numbers of ANDs
            self.spec_type = SpecType.VARIABLE_NUM_ANDS_IN_EACH_OR
            if self.optimize_disjuncts_separately:
                # for every OR, verify any AND clause to stop
                # c and rhs are already padded with dummy unverifiable clauses.
                # c: [num_or, max_num_and, num_output]
                # rhs: [num_or, max_num_and]. max_num_and = max(num_and_1, num_and_2, ...)
                stop_criterion = stop_criterion_batch_any
                self.post_processing_type = PostProcessingType.NONE
            else:
                # x: [1, *input_shape]
                # c: [1, num_clause, num_output]
                # rhs: [1, num_clause]. num_clause = sum(num_and_1, num_and_2, ...)
                x = take_batch(x, 1, 0)
                c = torch.cat(unpad_to_list_of_tensors(c, 0, 1, or_spec_size, True), dim=1)
                rhs = torch.cat(unpad_to_list_of_tensors(rhs, 0, 1, or_spec_size, True), dim=1)
                self.post_processing_type = PostProcessingType.SPLIT_PAD
                # verify all ORs to stop; for every OR, verify any AND clause to stop
                # in this case, the structure of OR-AND is flattened in the dim 1,
                # we need a new general stop criterion function.
                stop_criterion = stop_criterion_general
            # invprop code
            assert len(apply_output_constraints_to) == 0, (
                "Applying output constraints is not tested for multiple ORs with different numbers of ANDs."
            )

        self.x = x
        self.c = c
        self.rhs = rhs
        self.or_spec_size = or_spec_size
        self.num_or = num_or
        self.stop_criterion = stop_criterion
        self.same_x_range = same_x_range
        self.vnnlib_handler = vnnlib_handler

    def set_unverified_or_mask(self, lb, stop_criterion=None, rhs=None, or_spec_size=None):
        if stop_criterion is None:
            stop_criterion = self.stop_criterion
        if rhs is None:
            rhs = self.rhs
        rhs = rhs.to(lb)

        if stop_criterion is stop_criterion_batch_any:
            unverified_or_mask = ~stop_criterion_batch_any(rhs)(lb).squeeze(1)
        elif stop_criterion is stop_criterion_all:
            # [1, num_or]
            unverified_or_mask = (lb <= rhs).squeeze(0)
        elif stop_criterion is stop_criterion_general:
            if or_spec_size is None:
                or_spec_size = self.or_spec_size.to(lb.device)
            stop_criterion_per_or = stop_criterion_general(or_spec_size, rhs).__closure__[0].cell_contents(lb)
            unverified_or_mask = ~stop_criterion_per_or.squeeze(0)
        else:
            raise ValueError(f"Unknown stop criterion function: {stop_criterion}")
        # [num_or]
        self.unverified_or_mask = unverified_or_mask
        self.unverified_or_indices = unverified_or_mask.nonzero().view(-1)

    def _prune(self, data, dim, prune_and_size=False):
        # Prune the data along the specified dimension using unverified OR indices.
        # dim should be 2 for alphas, 0 for others.
        data = data.index_select(dim, self.unverified_or_indices.to(data.device))

        if prune_and_size and self.spec_type == SpecType.VARIABLE_NUM_ANDS_IN_EACH_OR:
            # if the number of AND clauses in each OR is different,
            # we need to remove redundant padding along the AND dimension,
            # which is always dim 1 for the data.
            max_or_spec_size = self.or_spec_size.max()
            data = data[:, :max_or_spec_size, ...]
        return data

    def prune_verified_or_specs(self, ret):
        verified_or_mask = ~self.unverified_or_mask
        verified_or_indices = verified_or_mask.nonzero().view(-1)
        print(f"{verified_or_mask.sum().item()} / {self.num_or} OR specs are verified.")
        print(f"Verified OR indices (first 10 items): {verified_or_indices[:10].tolist()}")
        print(f"Verified OR lbs (first 10 items): {ret['global_lb'][verified_or_mask][:10].tolist()}")
        print(f"Unverified OR indices (first 10 items): {self.unverified_or_indices[:10].tolist()}")
        print(f"Unverified OR lbs (first 10 items): {ret['global_lb'][self.unverified_or_indices][:10].tolist()}")

        self.num_or = self.unverified_or_mask.sum().item()
        self.or_spec_size = self.or_spec_size[self.unverified_or_indices]

        # final bounds: [num_or, num_and] -> [num_unverified_or, num_and]
        ret["global_lb"] = self._prune(ret["global_lb"], 0, True)
        ret["global_ub"] = self._prune(ret["global_ub"], 0, True)
        final_name = ret["model"].final_name

        # ret["lower_bounds"] can be None after mip()
        if ret["lower_bounds"]:
            # intermediate bounds: [num_or, ...] -> [num_unverified_or, ...]
            # intermediate bounds can be [1, ...] if not optimize_disjuncts_separately, no need to prune.
            for k in ret["lower_bounds"]:
                if self.optimize_disjuncts_separately or k == final_name:
                    ret["lower_bounds"][k] = self._prune(ret["lower_bounds"][k], 0, k == final_name)
                    ret["upper_bounds"][k] = self._prune(ret["upper_bounds"][k], 0, k == final_name)

        if ret["lA"]:
            # lA: [num_or, num_and, num_output] -> [num_unverified_or, num_and, num_output]
            ret["lA"] = {
                k: self._prune(v, 0, True) for k, v in ret["lA"].items()
            }

        if ret["alphas"]:
            for k in ret["alphas"]:
                for spec_name, alpha in ret["alphas"][k]["alpha"].items():
                    if self.optimize_disjuncts_separately or spec_name == final_name:
                        # alphas: [alpha_size, spec, num_or, num_output]
                        # -> [alpha_size, spec, num_unverified_or, num_output]
                        ret["alphas"][k]["alpha"][spec_name] = self._prune(alpha, 2, spec_name == final_name)

        if ret["mask"] and self.optimize_disjuncts_separately:
            for k in ret["mask"]:
                for i, m in enumerate(ret["mask"][k]):
                    # mask can be None if the corresponding input node is not perturbed.
                    if m is not None:
                        # mask: [num_or, ...] -> [num_unverified_or, ...]
                        # mask can be [1, ...] if not optimize_disjuncts_separately, no need to prune.
                        ret["mask"][k][i] = self._prune(m, 0, False)

        if ret["input_split_idx"] and self.optimize_disjuncts_separately:
            # input_split_idx: [num_or, 3 (split_depth)] -> [num_unverified_or, 3]
            # input_split_idx can be [1, 3] if not optimize_disjuncts_separately, no need to prune.
            ret["input_split_idx"] = {
                k: self._prune(v, 0, False) for k, v in ret["input_split_idx"].items()
            }

        # update the vnnlib handler with the unverified OR indices
        self.vnnlib_handler.prune_verified_or_specs(self.unverified_or_mask)
        return

    def post_process(self, model, ret):
        ret.update({"model": model})
        final_name = model.final_name
        num_or = self.num_or

        # NOTE: in following every post-processing,
        # 1. ideally, we need to update c and rhs for consistency,
        #   but they are not used later, we do not update them.
        # 2. although we share intermediate bounds,
        #   we do not expand intermediate bounds here for memory efficiency.
        if self.post_processing_type == PostProcessingType.RESHAPE:
            num_and = self.or_spec_size[0].item()
            # if # of AND clauses in each OR are the same and > 1, and we optimize them together,
            # we need to reshape the output from [1, num_clause] to [batch (num_or), num_and].

            # final bounds: [1, num_clause] -> [num_or, num_and]
            ret["global_lb"] = ret["global_lb"].view(num_or, num_and)
            ret["global_ub"] = ret["global_ub"].view(num_or, num_and)
            ret["lower_bounds"][final_name] = ret["global_lb"]
            ret["upper_bounds"][final_name] = ret["global_ub"]

            if ret["lA"]:
                # lA: [1, num_clause, ...] -> [num_or, num_and, ...]
                # ... refers to the output shape of the layer.
                ret["lA"] = {k: v.view(num_or, num_and, *v.shape[2:]) for k, v in ret["lA"].items()}

            if ret["alphas"]:
                for m in ret["alphas"]:
                    if final_name not in ret["alphas"][m]["alpha"]:
                        continue
                    # final alphas: [alpha_size, num_clause, 1, ...]
                    # alpha_size is 2 for ReLU and 4 or more for complex nonlinear splits.
                    # -> [alpha_size, num_and, num_or, ...]
                    final_alpha = ret["alphas"][m]["alpha"][final_name].detach()
                    ret["alphas"][m]["alpha"][final_name] = (
                        final_alpha
                        .view(final_alpha.shape[0], num_or, num_and, *final_alpha.shape[3:])
                        .transpose(1, 2)
                    )

        elif self.post_processing_type == PostProcessingType.SPLIT_PAD:
            # if # of AND clauses in each OR are different, and we optimize them together,
            # we need to split the output from [1, num_clause] to list of [1, num_and],
            # and pad the output with dummy unverifiable clauses.

            split_sections = self.or_spec_size.tolist()
            # final bounds: [1, num_clause] -> list of [1, num_and]
            # -> [num_or, num_and_max].
            ret["global_lb"] = pad_list_of_input_to_tensor(
                torch.split(ret["global_lb"], split_sections, dim=1),
                pad_value=float("-inf"),
                pad_dim=1,
                batch_dim=0,
                is_orginal_tensor=True,
            )
            ret["global_ub"] = pad_list_of_input_to_tensor(
                torch.split(ret["global_ub"], split_sections, dim=1),
                pad_value=float("inf"),
                pad_dim=1,
                batch_dim=0,
                is_orginal_tensor=True,
            )
            ret["lower_bounds"][final_name] = ret["global_lb"]
            ret["upper_bounds"][final_name] = ret["global_ub"]

            if ret["lA"]:
                # lA: [1, num_clause, ...] -> list of [1, num_and, ...]
                # -> [num_or, num_and_max, ...].
                for k, v in ret["lA"].items():
                    ret["lA"][k] = pad_list_of_input_to_tensor(
                        torch.split(v, split_sections, dim=1),
                        pad_value=0,
                        pad_dim=1,
                        batch_dim=0,
                        is_orginal_tensor=True
                    )

            if ret["alphas"]:
                for m in ret["alphas"]:
                    if final_name not in ret["alphas"][m]["alpha"]:
                        continue
                    # final alphas: [alpha_size, num_clause, 1, ...]
                    # -> list of [alpha_size, num_and, 1, ...]
                    # -> [alpha_size, num_and_max, num_or, ...].
                    final_alpha = ret["alphas"][m]["alpha"][final_name].detach()
                    ret["alphas"][m]["alpha"][final_name] = pad_list_of_input_to_tensor(
                        torch.split(final_alpha, split_sections, dim=1),
                        pad_value=0,
                        pad_dim=1,
                        batch_dim=2,
                        is_orginal_tensor=True
                    )

        self.prune_verified_or_specs(ret)

        # they are not needed anymore
        del self.x, self.c, self.rhs

        return ret

    def expand_intermediate(self, ret):
        if self.optimize_disjuncts_separately:
            return

        num_or = self.num_or
        model = ret["model"]
        final_name = model.final_name
        for k in ret["lower_bounds"]:
            if k == final_name:
                continue
            # intermediate bounds: [1, ...] -> [num_or, ...]
            ret["lower_bounds"][k] = ret["lower_bounds"][k].expand(num_or, *ret["lower_bounds"][k].shape[1:])
            ret["upper_bounds"][k] = ret["upper_bounds"][k].expand(num_or, *ret["upper_bounds"][k].shape[1:])
        if ret["alphas"]:
            for m in ret["alphas"]:
                for k, v in ret["alphas"][m]["alpha"].items():
                    if k == final_name:
                        continue
                    # intermediate alphas: [alpha_size, num_inter_specs, 1, ...]
                    # -> [alpha_size, num_inter_specs, num_or, ...].
                    ret["alphas"][m]["alpha"][k] = v.expand(*v.shape[:2], num_or, *v.shape[3:])

        return

    def prune_attack_ret(self, attack_examples, attack_margins, all_adv_candidates):
        attack_examples = self._prune(attack_examples, 0, False) if attack_examples is not None else None
        attack_margins = self._prune(attack_margins, 0, False) if attack_margins is not None else None
        all_adv_candidates = self._prune(all_adv_candidates, 0, False) if all_adv_candidates is not None else None
        return attack_examples, attack_margins, all_adv_candidates

    def adhoc_process_for_mip(self, ret):
        num_or = self.num_or
        model: LiRPANet = ret["model"]
        final_name = model.final_name

        # mip() only supports cases satisfying the following conditions:
        # 1. the final specification is in the form of single OR / multiple ORs with single AND
        # 2. Multiple ORs with same input range
        # 3. Multiple ORs are optimized together when the LiRPANet is built.
        # 4. Single OR with multiple ANDs.
        assert self.same_x_range
        assert not self.optimize_disjuncts_separately

        # following data are needed for mip() and should be processed:
        # 1. model.c, model.x
        # 3. model.net[final_name].lower
        # TODO: Ideally, mip() should only get everything needed from ret instead of model.
        x, c, _, _, _, _ = self.vnnlib_handler.all_specs.get(device=model.device)

       # Case 1: OR/ORs with single AND
        if self.spec_type == SpecType.SINGLE_AND_IN_MULTI_ORS or (
            self.spec_type == SpecType.SINGLE_OR and ret["global_lb"].shape[1] == 1
        ):
            # mip() expects they have batch size 1,
            # we rewrite model.x and model.c to have batch size 1 with only unverified ORs.
            # the previous same_x_range assert has ensured that input is the same for all ORs.
            model.x = take_batch(x, 1, 0)
            model.c = c.view(1, -1, c.shape[-1])
            # ret["global_lb"] has be pruned, we reuse it with batch size 1.
            model.net[final_name].lower = ret["global_lb"].view(1, num_or)
        # Case 2: single OR with multiple ANDs (AND-style specification)
        elif self.spec_type == SpecType.SINGLE_OR:
            # For AND-style specifications, we don't need to reshape
            model.x = x
            model.c = c
            model.net[final_name].lower = ret["global_lb"]
        
        else:
            # TODO: support other spec types in mip()
            raise ValueError(f"Unsupported spec type for mip(): {self.spec_type}")

    def adhoc_post_process_for_mip(self, ret):
        num_or = self.num_or
        model: LiRPANet = ret["model"]
        final_name = model.final_name

        # Case 1: OR/ORs with single AND
        if self.spec_type == SpecType.SINGLE_AND_IN_MULTI_ORS or (
            self.spec_type == SpecType.SINGLE_OR and ret["global_lb"].shape[1] == 1
        ):
            # ret["global_lb"] should be in shape [num_or, 1].
            # model.net[final_name].lower should be never accessed after mip() call,
            # but we still set it for consistency
            model.net[final_name].lower = ret["global_lb"]
        # Case 2: single OR with multiple ANDs (AND-style specification)
        elif self.spec_type == SpecType.SINGLE_OR:
            # For AND-style specifications, model.net[final_name].lower should match ret["global_lb"]
            model.net[final_name].lower = ret["global_lb"]
        
        else:
            # TODO: support other spec types in mip()
            raise ValueError(f"Unsupported spec type for mip(): {self.spec_type}")

        # Prune OR specs that are verified by mip()
        # here rhs has the shape [num_or, num_and (1)], we should always use stop_criterion_batch_any
        self.set_unverified_or_mask(ret["global_lb"], stop_criterion_batch_any, self.vnnlib_handler.all_specs.rhs)
        self.prune_verified_or_specs(ret)