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
import torch
from typing import TYPE_CHECKING, Any, Mapping, Tuple, Optional
from torch import Tensor
from dataclasses import dataclass, field
from auto_LiRPA.utils import transfer
import arguments
from utils import convert_history_from_list
from tensor_storage import TensorStorage


if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


class EmptiedTensor(ValueError):
    """
    A special ValueError to indicate that the tensor has been emptied.
    Initially used to save lA transfer time thanks to decision precomputing
    """


@dataclass
class IntermBoundsInfo:
    """
    Format: [batch, *shape]
    """

    lower_bound: Optional[Tensor]
    upper_bound: Optional[Tensor]

    def __iter__(self):
        yield self.lower_bound
        yield self.upper_bound


@dataclass(slots = True)
class AggregateSplitList:
    """
    An efficient list of activation splits,
    each activation split defines the position, direction,
    and the value of the split.
    Used to hold all splits of a batch of domains for one splitable layer.

    Format:

    the t-th split is defined by the following:
    - batch_ids[t]: domain index this entry came from.
    - indexes[t]: 1-D long tensor of flat neuron indices split in that domain.
    - directions[t]: 1-D tensor aligned with indexes[t]; > 0 means the split
      keeps x >= value (raises the lower bound), otherwise x <= value
      (lowers the upper bound).
    - values[t]: 1-D tensor of split points aligned with indexes[t]
      (0 for ReLU splits).
    """

    batch_ids: list[int] = field(default_factory=list)
    values: list[Tensor] = field(default_factory=list)
    indexes: list[Tensor] = field(default_factory=list)
    directions: list[Tensor] = field(default_factory=list)

    def append(self, batch_id: int, values: Tensor, indexes: Tensor,
               directions: Tensor) -> None:
        self.batch_ids.append(batch_id)
        self.values.append(values)
        self.indexes.append(indexes)
        self.directions.append(directions)

    def __iter__(self):
        yield self.batch_ids
        yield self.values
        yield self.indexes
        yield self.directions


class WorkingIntermBoundsInfo(dict[str, IntermBoundsInfo]):
    """
    Intermediate bounds information directly from the working net.


    Format: {node_name -> IntermBoundsInfo}
    """

    def mask_batch_dim_inplace(self, batch_mask: torch.Tensor):
        """Inplace mask the batch dimension."""
        for layer_name, bounds_info in self.items():
            if bounds_info.lower_bound is not None and isinstance(
                bounds_info.lower_bound, torch.Tensor
            ):
                bounds_info.lower_bound = bounds_info.lower_bound[batch_mask]
            if bounds_info.upper_bound is not None and isinstance(
                bounds_info.upper_bound, torch.Tensor
            ):
                bounds_info.upper_bound = bounds_info.upper_bound[batch_mask]
        return self

    def to_lower_bounds_dict(
        self, append_dict: Optional[dict[str, Tensor | None]] = None
    ) -> dict[str, Tensor | None]:
        """Convert to a dict of lower bounds."""
        res = {
            layer_name: (
                bounds_info.lower_bound.detach()
                if isinstance(bounds_info.lower_bound, torch.Tensor)
                else bounds_info.lower_bound
            )
            for layer_name, bounds_info in self.items()
        }
        if append_dict is not None:
            res.update(append_dict)
        return res

    def to_upper_bounds_dict(
        self, append_dict: Optional[dict[str, Tensor | None]] = None
    ) -> dict[str, Tensor | None]:
        """Convert to a dict of upper bounds."""
        res = {
            layer_name: (
                bounds_info.upper_bound.detach()
                if isinstance(bounds_info.upper_bound, torch.Tensor)
                else bounds_info.upper_bound
            )
            for layer_name, bounds_info in self.items()
        }
        if append_dict is not None:
            res.update(append_dict)
        return res

    def to_unstable_bounds(
        self,
        unstable_mask: dict,
        layers_requiring_bounds_names: list[str],
        split_nodes_names: list[str],
        device=None,
    ) -> dict | None:
        """Extract masked unstable bounds from self.

        Returns None when input_split is enabled (unstable_bounds not used there).

        Args:
            net: LiRPANet instance for split_nodes, interm_transfer access.
            unstable_mask: {layer_name: mask} for unstable neurons.
            device: target device for transfer.

        Returns:
            {layer_name: [lb_masked, ub_masked]} or None
        """
        if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
            return None

        unstable_bounds = {}
        for layer_name in layers_requiring_bounds_names:
            if layer_name not in split_nodes_names:
                continue
            mask = unstable_mask[layer_name]
            if mask is None:
                continue
            lower = self[layer_name].lower_bound
            upper = self[layer_name].upper_bound
            assert (
                lower is not None and upper is not None
            ), f"Layer {layer_name} is in unstable_mask but has no bounds in working_interm_bounds."
            unstable_bounds[layer_name] = [
                transfer(lower[:, mask[0]], device),
                transfer(upper[:, mask[0]], device),
            ]
        return unstable_bounds

    def to_complete_init_bounds(
        self,
        net: "LiRPANet",
        lb,
        ub,
    ) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        """Build complete (lower_bounds, upper_bounds) dicts for all intermediate layers.

        By default, also appends the final layer bound (lb and ub after applying C).

        Args:
            net: LiRPANet instance for final_name and get_split_nodes access.
            lb: lower bound tensor for the final layer.
            ub: upper bound tensor for the final layer (or None, treated as +inf).
        """
        if ub is None:
            ub = lb + torch.inf

        # If input split is enabled, we do not need to get intermediate bounds;
        # just return the bounds for the final layer.
        if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
            return {net.final_name: lb.detach()}, {net.final_name: ub.detach()}

        net.get_split_nodes()
        lower_bounds: Any = self.to_lower_bounds_dict(
            append_dict={net.final_name: lb.detach()}
        )
        upper_bounds: Any = self.to_upper_bounds_dict(
            append_dict={net.final_name: ub.detach()}
        )

        return lower_bounds, upper_bounds 

    @staticmethod
    def from_net(net: "LiRPANet", move: bool = False) -> "WorkingIntermBoundsInfo":
        """Extract intermediate bounds from a BoundedModule.

        Args:
            net: BoundedModule whose layers_requiring_bounds will be read.
            move: If True (default), also set each layer's lower/upper to an
                  EmptiedTensor sentinel after copying, transferring ownership.
                  If False, only copy the bounds without modifying the layers.

        Returns:
            WorkingIntermBoundsInfo: {node_name: IntermBoundsInfo}
        """
        ret = WorkingIntermBoundsInfo()
        for layer in net.net.layers_requiring_bounds:
            if layer.lower is None and layer.upper is None:
                continue

            ret[layer.name] = IntermBoundsInfo(
                lower_bound=(
                    layer.lower.detach()
                    if isinstance(layer.lower, torch.Tensor)
                    else layer.lower  # might be ValueError | None
                ),
                upper_bound=(
                    layer.upper.detach()
                    if isinstance(layer.upper, torch.Tensor)
                    else layer.upper  # might be ValueError | None
                ),
            )

            if move:
                layer.lower = EmptiedTensor("Intermediate lower bound has been emptied.")
                layer.upper = EmptiedTensor("Intermediate upper bound has been emptied.")

        return ret

    @staticmethod
    def from_histories(
        histories: list,
        ref_static_lbs: dict,
        ref_static_ubs: dict,
        device,
    ):
        """
        Construct intermediate bounds from relu bab histories and reference static bounds.
        Set concrete values according to the direction of relu split or non-linear split.

        Args:
            histories: A batch of histories.
            ref_static_lbs: reference lower bounds,
                batch dimension should be longger than
                target batch size implicitly in histories.
            ref_static_ubs: reference upper bounds
            device: target device for the resulting bounds
        """

        # Initialize bounds dictionaries
        lower_bounds, upper_bounds = {}, {}
        batch = len(histories)
        assert ref_static_lbs.keys() == ref_static_ubs.keys()

        for i in ref_static_lbs:
            lower_bounds[i] = ref_static_lbs[i][:batch].clone()
            upper_bounds[i] = ref_static_ubs[i][:batch].clone()

        # Key idea: writing each domain's splits into the bound tensors one
        # domain at a time costs O(batch) tiny tensor ops per layer. Instead,
        # first bucket all histories by layer in plain Python (Pass 1), then
        # update each layer's bounds with one batched indexing op per bound
        # side (Pass 2), reducing tensor ops from O(batch * layers) to
        # O(layers).
        #
        # Pass 1: a single sweep over domains, aggregating each (domain,
        # layer) split entry into that layer's AggregateSplitList. No tensor
        # ops happen here. Layers not in ref_static_lbs are skipped, matching
        # the original loop which only visited layers in the template.
        aggregated = {i: AggregateSplitList() for i in ref_static_lbs}
        for j, hist in enumerate(histories):
            for i in hist:
                layer_agg = aggregated.get(i)
                if layer_agg is None:
                    continue
                hist[i] = convert_history_from_list(hist[i])
                indices, directions, values = hist[i][0], hist[i][1], hist[i][2]
                assert indices.shape[0] == directions.shape[0] == values.shape[0], (
                    f"Indices, directions, and values must have the same length. "
                    f"Got {indices.shape[0]}, {directions.shape[0]}, {values.shape[0]}."
                )
                if indices.shape[0] == 0:
                    continue
                layer_agg.append(
                    batch_id=j, values=values, indexes=indices,
                    directions=directions,
                )

        # Pass 2: per layer, concatenate the aggregated chunks and write all
        # split values of all domains into the flattened bound tensors with
        # one advanced-indexing assignment per bound side.
        for i, layer_agg in aggregated.items():
            if not layer_agg.batch_ids:
                continue
            all_indices = torch.cat(layer_agg.indexes)
            all_directions = torch.cat(layer_agg.directions)
            all_values = torch.cat(layer_agg.values)
            # The masks and batch ids below are built on CPU and mixed with
            # history tensors in one indexing expression, so histories must
            # live on CPU.
            assert all_indices.device.type == 'cpu', (
                'from_histories requires history tensors on CPU, '
                f'got {all_indices.device}.'
            )
            # all_batch[k] is the batch row (domain) that the k-th element of
            # the concatenated all_indices/all_values belongs to: chunk t
            # contributed indexes[t].shape[0] splits, so its batch id is
            # repeated that many times. (all_batch[k], all_indices[k]) is
            # then the (row, flat neuron) coordinate receiving all_values[k].
            all_batch = torch.repeat_interleave(
                torch.tensor(layer_agg.batch_ids, dtype=torch.long),
                torch.tensor(
                    [chunk.shape[0] for chunk in layer_agg.indexes],
                    dtype=torch.long,
                ),
            )
            lb_mask = all_directions > 0
            ub_mask = ~lb_mask

            # Only view() a layer's bounds when there is something to scatter:
            # bounds of untouched layers may be non-contiguous (e.g. channels
            # last in patches mode) and would fail the view.
            if lb_mask.any():
                lb_flat = lower_bounds[i].view(batch, -1)
                lb_flat[
                    all_batch[lb_mask].to(device, non_blocking=True),
                    all_indices[lb_mask].to(device, non_blocking=True),
                ] = all_values[lb_mask].to(device, non_blocking=True)

            if ub_mask.any():
                ub_flat = upper_bounds[i].view(batch, -1)
                ub_flat[
                    all_batch[ub_mask].to(device, non_blocking=True),
                    all_indices[ub_mask].to(device, non_blocking=True),
                ] = all_values[ub_mask].to(device, non_blocking=True)

        return WorkingIntermBoundsInfo(
            {
                layer_name: IntermBoundsInfo(
                    lower_bound=lower_bounds[layer_name],
                    upper_bound=upper_bounds[layer_name],
                )
                for layer_name in ref_static_lbs
            }
        )

    def update_by_unstable_bounds(
        self,
        unstable_bounds: dict,
        final_name: str,
        unstable_mask: dict,
    ) -> None:
        """Update self in-place using unstable (sparse) bounds.

        Packs self into the interm_bounds format expected by update_interm_bounds,
        applies the update, then writes results back into self.

        Args:
            unstable_bounds: {layer_name: [lb, ub]} with shape [batch, num_unstable].
                             Should already be popped and device-transferred by the caller.
            final_name: name of the final layer (excluded from update).
            unstable_mask: {layer_name: mask} indicating unstable neuron positions.
        """
        interm_bounds = {
            k: [v.lower_bound, v.upper_bound]
            for k, v in self.items()
            if k != final_name
        }
        updated = update_interm_bounds(
            interm_bounds, unstable_bounds, final_name, unstable_mask
        )
        for k, (lb, ub) in updated.items():
            self[k].lower_bound = lb
            self[k].upper_bound = ub

    @staticmethod
    def from_two_dicts(
        lower_bounds_dict: dict[str, Tensor | None],
        upper_bounds_dict: dict[str, Tensor | None],
    ) -> "WorkingIntermBoundsInfo":
        """Construct from two separate dicts of lower and upper bounds."""
        assert (
            lower_bounds_dict.keys() == upper_bounds_dict.keys()
        ), "Lower and upper bounds dicts must have the same keys."
        return WorkingIntermBoundsInfo(
            {
                layer_name: IntermBoundsInfo(
                    lower_bound=lower_bounds_dict[layer_name],
                    upper_bound=upper_bounds_dict[layer_name],
                )
                for layer_name in lower_bounds_dict
            }
        )

    def compute_unstable_mask(self, net: "LiRPANet") -> dict[str, torch.BoolTensor]:
        """Compute unstable masks for this batch"""
        new_masks = {}
        for k in self:
            if k not in net.split_activations:
                continue
            lb = self[k].lower_bound
            ub = self[k].upper_bound
            mask = None
            for activation, index in net.split_activations[k]:
                mask_ = activation.get_split_mask(lb, ub, index).flatten(1)
                mask = mask_ if mask is None else torch.logical_or(mask, mask_)
            if mask is None:
                mask = torch.ones_like(lb, dtype=torch.bool).flatten(1)
            new_masks[k] = mask
        return new_masks

    def to_dict_of_list(self):
        """Convert self to dict of list format for easier storage."""
        return {
            k: [
                v.lower_bound.detach() if isinstance(v.lower_bound, torch.Tensor) else v.lower_bound,
                v.upper_bound.detach() if isinstance(v.upper_bound, torch.Tensor) else v.upper_bound,
            ]
            for k, v in self.items()
        }

@dataclass
class IntermBoundsFactory:
    """
    A factory to construct full interm bounds
      for compute_bounds from history and
        optionally unstable_bounds. used in ReLU BaB.
    """

    static_lb: dict[str, Tensor]
    static_ub: dict[str, Tensor]
    batch_dim_size: int
    device: torch.device
    final_name: str

    @staticmethod
    def from_interm(interm: WorkingIntermBoundsInfo, final_name, device=None):

        if device is None:
            first = next(
                (lb for k, (lb, _) in interm.items() if k != final_name), None
            )
            assert first is not None, "interm has no intermediate layers."
            device = first.device

        static_lb = {}
        static_ub = {}

        for k, (lb, ub) in interm.items():
            assert (
                lb is not None and ub is not None
            ), f"Layer {k} has None bounds in interm."
            if k == final_name:
                continue
            static_lb[k] = lb[0:1].to(device=device, non_blocking=True)
            static_ub[k] = ub[0:1].to(device=device, non_blocking=True)

        return IntermBoundsFactory(
            static_lb=static_lb,
            static_ub=static_ub,
            batch_dim_size=1,
            device=device,
            final_name=final_name,
        )

    def enlarge_template_size(self, batch: int):
        for k in self.static_lb:
            # enlarge the batch size in the static storage
            if batch > self.static_lb[k].shape[0]:
                power = (batch + self.static_lb[k].shape[0] - 1) // self.static_lb[
                    k
                ].shape[0]
                self.static_lb[k] = self.static_lb[k].repeat(
                    power, *([1] * (self.static_lb[k].dim() - 1))
                )
            if batch > self.static_ub[k].shape[0]:
                power = (batch + self.static_ub[k].shape[0] - 1) // self.static_ub[
                    k
                ].shape[0]
                self.static_ub[k] = self.static_ub[k].repeat(
                    power, *([1] * (self.static_ub[k].dim() - 1))
                )
        self.batch_dim_size = next(iter(self.static_lb.values())).shape[0]

    def construct_interm_bounds(
        self,
        d_histories,
        d_unstable_bounds: Optional[dict],
        unstable_mask: Optional[dict],
        final_name: str,
    ) -> WorkingIntermBoundsInfo:
        """
        Construct intermediate bounds for compute_bounds for BaB without host-device transfer.
        Args:
            d_histories: list of histories for the current batch, used to set concrete values in the intermediate bounds.
            d_unstable_bounds: {layer_name: [lb, ub]} You want to use unstable_bounds to update the complete interm bounds.
            unstable_mask: {layer_name: mask} indicating unstable neuron positions, used to update the intermediate bounds.
        """
        if len(d_histories) > self.batch_dim_size:
            self.enlarge_template_size(len(d_histories))
        interm = WorkingIntermBoundsInfo.from_histories(
            histories=d_histories,
            ref_static_lbs=self.static_lb,
            ref_static_ubs=self.static_ub,
            device=self.device,
        )
        if d_unstable_bounds is not None:
            assert (
                unstable_mask is not None
            ), "unstable_mask is required when d_unstable_bounds is provided."
            interm.update_by_unstable_bounds(
                unstable_bounds=d_unstable_bounds,
                final_name=final_name,
                unstable_mask=unstable_mask,
            )
        return interm

    def construct_interm_bounds_in_d(self, d_inout, unstable_mask) -> None:
        """
        Construct intermediate bounds and write them into d_inout["lower_bounds"] and d_inout["upper_bounds"].
        Use static_lb and static_ub as templates, change values according to d_inout["history"] and d_inout["unstable_bounds"].
        """
        assert all(
            x in d_inout
            for x in [
                "final_lb",
                "final_ub",
                "history",
                "unstable_bounds",
            ]
        )
        interm = self.construct_interm_bounds(
            d_histories=d_inout["history"],
            d_unstable_bounds=(
                d_inout["unstable_bounds"]
                if isinstance(d_inout["unstable_bounds"], dict)
                else None
            ),
            unstable_mask=unstable_mask,
            final_name=self.final_name,
        )
        d_inout["lower_bounds"] = interm.to_lower_bounds_dict(
            append_dict=d_inout["final_lb"]
        )
        d_inout["upper_bounds"] = interm.to_upper_bounds_dict(
            append_dict=d_inout["final_ub"]
        )
        if not isinstance(d_inout["unstable_bounds"], ValueError):
            d_inout["unstable_bounds"] = ValueError(
                "unstable bounds are updated into "
                "lower/upper bounds. Delete for performance"
            )


def update_interm_bounds(
    interm_bounds,
    new_interm_bounds,
    final_name,
    unstable_mask,
    prune_mask=None,
    verbose=False,
):
    """
    Update each lb tensor in lb_dict based on some new_interm_bounds,
    optionally pruning along the batch dimension via prune_mask.

    @params:
        interm_bounds : dict
            key: Each layer's name
            value: a list
                v[0]: lb with shape [batch, input_dim]
                v[1]: ub with shape [batch, input_dim]

        new_interm_bounds : dict
            key: same layer names as above
            value: a list
                v[0]: lb with shape [batch, num_unstable_neurons]
                v[1]: ub with shape [batch, num_unstable_neurons]

        prune_mask : torch.BoolTensor or None
            A boolean mask of shape [batch]. If provided, we only keep
            the rows where prune_mask == True.

    @return:
        updated_interm_bounds: A dict with same structure as interm_bounds
                            but updated (and optionally pruned).
    """
    updated_interm_bounds = {}
    for key in interm_bounds.keys():
        # 1) If it's the final layer, or if new_interm_bounds doesn't have it, just copy over
        if key == final_name or key not in new_interm_bounds.keys():
            updated_interm_bounds[key] = interm_bounds[key]
            continue

        # 2) Extract lb/ub from both dictionaries
        lb, ub = interm_bounds[key]
        some_lb, some_ub = new_interm_bounds[key]

        # ---------------------------------------------------
        # 3) PRUNE BATCH if a prune_mask is provided
        # ---------------------------------------------------
        if prune_mask is not None:
            lb = lb[prune_mask]  # shape: [new_batch, input_dim]
            ub = ub[prune_mask]  # shape: [new_batch, input_dim]

        # 4) The "mask" in self.mask[key] is usually about neuron indices (columns), not the batch.
        #    e.g. mask might indicate which neurons are "unstable" to be updated.
        mask = unstable_mask[key]  # Typically a boolean or index list for columns

        # Safety check
        if mask.sum().item() != some_lb.size(1):
            print(
                f"Warning: Key {key} has mismatch: mask size={mask.sum().item()}, some_lb size={some_lb.size()}"
            )
            # If mismatch, just keep original (already pruned) lb/ub
            updated_interm_bounds[key] = [lb, ub]
            continue

        # 5) Slice the columns (neurons) out from lb and ub
        #    Here mask is presumably shape [num_neurons] or something like that
        lb_masked = lb[:, mask[0]]  # shape: [new_batch, N]
        ub_masked = ub[:, mask[0]]

        if isinstance(lb_masked, TensorStorage):
            lb_masked = lb_masked.tensor()
            ub_masked = ub_masked.tensor()

        if isinstance(some_lb, TensorStorage):
            some_lb = some_lb.tensor()
            some_ub = some_ub.tensor()

        some_lb = some_lb.to(lb_masked.device)
        some_ub = some_ub.to(ub_masked.device)

        # 6) Compute better bounds
        lb_best = torch.maximum(lb_masked, some_lb)  # shape: [new_batch, N]
        ub_best = torch.minimum(ub_masked, some_ub)

        # 7) Write back the improved lb/ub into the original (pruned) lb/ub
        lb[:, mask[0]] = lb_best
        ub[:, mask[0]] = ub_best

        if verbose:
            # 8) Collect some stats
            lb_diff = lb_best - lb_masked
            ub_diff = ub_best - ub_masked
            #  new batch size after pruning
            new_batch_size = lb.size(0)
            if new_batch_size > 0:

                lb_num_improved = (lb_diff > 0).sum().item() / new_batch_size
                ub_num_improved = (ub_diff < 0).sum().item() / new_batch_size

                lb_improved = lb_diff.sum().item() / new_batch_size
                ub_improved = ub_diff.sum().item() / new_batch_size

                print(f" layer: {key}")
                print(
                    f"    lower bounds improved: average # {lb_num_improved}, value {lb_improved}"
                )
                print(
                    f"    upper bounds improved: average # {ub_num_improved}, value {ub_improved}"
                )
        # 9) Save updated (and pruned) lb, ub
        updated_interm_bounds[key] = [lb, ub]

    return updated_interm_bounds
