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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet

class NeuronBranchingHeuristic():
    """Base class for branching heuristics."""

    def __init__(self, net: 'LiRPANet'):
        self.net = net
        self.batch_size = None
        self.device = None

    def layer_iterator(self, bounds):
        """
        Iterate over a list or a dictionary of bounds in a consistent order.
        """
        keys = sorted(bounds.keys())
        for key in keys:
            if key in self.net.split_activations:
                yield key, bounds[key]

    def update_batch_size_and_device(self, bounds):
        """Set self.batch_size and self.device based on intermediate bounds."""
        assert isinstance(bounds, dict)
        first_bounds = bounds[list(bounds.keys())[0]]
        self.batch_size = first_bounds.size(0)
        self.device = first_bounds.device

    def get_branching_decisions(self, domains, split_depth=1, **kwargs):
        r"""
        Get branching decisions given intermediate layer bounds and a mask
        indicating which neurons are "splittable".

        Args:
            domains: A dictionary containing the data of the picked out domains.
            Items may contain:
                - lower_bounds:  A dictionary or list of intermediate layer lower
                bounds in shape (batch, layer_size). A dictionary is the preferred
                format, with keys as the name of the intermediate layer.

                - upper_bounds: A dictionary or list of intermediate layer upper
                bounds  in shape (batch, layer_size). A dictionary is the preferred
                format, with keys as the name of the intermediate layer.

                - mask: A dictionary of list of masks tensors, one for each
                layer, and the size of the mask tensor is the same as the number
                of neurons in that layer. A value of 1 indicates that a neuron can
                be split, and 0 will exclude that neuron in branching heuristic.

                - lAs: A list or dictionary of lA matrices for each splittable layer.
                Note that only the lA matrices for splittable layers are saved to
                save space.

            split_depth: The number of neurons to choose for split.

        Returns:
            The return value depends on the format_decisions() method.
            Currently, the first value is a list of 2-tuples of (layer_idx,
            neuron_idx) and there are real_topk * batch_size number of tuples.
            The second return value is real_topk which can be smaller than
            topk argument.
            FIXME: the return must be changed for higher efficiency.
        """
        split_masks = domains['mask']
        self.update_batch_size_and_device(domains['lower_bounds'])
        # FIXME no need to provide self.net.split_indices
        layer_scores = self.compute_neuron_scores(domains, **kwargs)
        # Fixup the scores, adding a small epsilon to non masked neurons
        # to avoid we accidentally select a neuron that we should not split.
        for idx, score in self.layer_iterator(layer_scores):
            # split_mask = 1 => a neuron can be split.
            score += split_masks[idx] * 1e-10
        topk_neuron_layers, topk_neuron_indices = self.find_topk_scores(
            layer_scores, split_masks, split_depth)

        # TODO need to return the branching point
        return self.format_decisions(topk_neuron_layers, topk_neuron_indices)

    def find_topk_scores(self, layer_scores, split_masks, k, return_scores=False):
        """
        After scores for each neuron is computed, this function finds
        the top-k scores across all layers.
        """
        # Scores of topk neurons for all elements in this batch.
        topk_scores = torch.full(
            size=(self.batch_size, k),
            device=self.device, fill_value=float('-inf'))
        # Layer of topk neurons for all elements in this batch.
        topk_neuron_layers = torch.full(
            size=(self.batch_size, k),
            device=self.device, dtype=torch.int64, fill_value=-1)
        # Index in each layer of topk neurons for all elements in this batch.
        topk_neuron_indices = torch.full(
            size=(self.batch_size, k),
            device=self.device, dtype=torch.int64, fill_value=-1)
        # Maximum number of valid scores for each batch element.
        max_valid_scores = torch.zeros(
            size=(self.batch_size,), device=self.device)
        for idx, layer in enumerate(self.net.split_nodes):
            if layer.name not in layer_scores:
                continue
            scores = layer_scores[layer.name]
            mask = split_masks[layer.name]
            # Each score should have shape [batch, score] for layer name "idx".
            # Clamp k to the maximum number of neurons per layer.
            this_layer_k = min(k, scores.size(1))
            if this_layer_k > 0:
                # First find the topk score from this layer. Shape (batch, k).
                # FIXME duplicate split_masks for branching scores?
                layer_topk_scores, layer_topk_neuron_indices = (
                    scores * mask).topk(k=this_layer_k, dim=1)
                # Then compare the topk score from global topk scores so far.
                # The first half is the current global topk, and the second
                # half is the per-layer topk. So if the new_topk_indices has
                # values >= k then it is selected from this layer.
                topk_scores, new_topk_indices = torch.cat(
                    [topk_scores, layer_topk_scores], dim=1).topk(k=k, dim=1)
                # Update the layer indices and neuron indices for topk scores
                # selected for this layer.
                select_from_this_layer_mask = new_topk_indices >= k
                topk_neuron_layers = torch.where(
                    select_from_this_layer_mask,
                    # For neurons from previous layers, we only care about
                    # the indices >= k.
                    idx,
                    # For neurons from previous layers, we only care about
                    # the indices < k.
                    topk_neuron_layers.gather(
                        dim=1, index=new_topk_indices.clamp(max=k-1)))
                # Update the neuron indices, based on if this neuron if from
                # this layer or previous layers.
                topk_neuron_indices = torch.where(
                    select_from_this_layer_mask,
                    # For neurons from previous layers, we only care about
                    # the indices >= k.
                    layer_topk_neuron_indices.gather(
                        dim=1, index=(new_topk_indices - k).clamp(min=0)),
                    # For neurons from previous layers, we only care about
                    # the indices < k.
                    topk_neuron_indices.gather(
                        dim=1, index=new_topk_indices.clamp(max=k-1)))
                # Update the number of valid scores based on the mask.
                max_valid_scores += mask.sum(dim=1)
        # Since all elements in this batch must have the same split depth, we
        # take the minimum.
        max_k = int(max_valid_scores.min().item())
        ret = (topk_neuron_layers[:, :max_k], topk_neuron_indices[:, :max_k])
        if return_scores:
            ret = ret + (topk_scores[:, :max_k],)
        return ret

    def format_decisions(self, layers, indices, points=None):
        """
        Given the topk layer idx and neuron idx, return the branching
        decisions. layers and indices both have shape
        (batch, split_depth).
        """
        # TODO: return two tensors, not list. Avoid a lot of for loops later.
        # return layers, indices
        # Shape is (batch, split_depth, 2)
        decisions = (layers, indices)
        stacked_decisions = torch.stack(decisions, dim=-1)
        # FIXME: so far the returned shape is very bad, mixing shape and depth.
        # This is temporary and must be fixed.
        split_depth = layers.size(1)
        stacked_decisions = stacked_decisions.transpose(0, 1).reshape(-1, 2).tolist()
        if points is not None:
            # FIXME Possibly don't .tolist() which is required by copy_domains
            # right now.
            points = points.transpose(0, 1)
            # For compatibility
            if points.size(-1) == 1:
                points = points.reshape(-1)
            else:
                points = points.reshape(-1, points.size(-1))
        return stacked_decisions, points, split_depth

    def compute_neuron_scores(self, domains, **kwargs):
        """To-be implemented by instances of this abstract class."""
        raise NotImplementedError

    def filter_decisions(self, topk):
        """
        TODO: implement k-fsb like filtering, running top-k scored neurons
        using beta-CROWN and find the best ones.
        """
        raise NotImplementedError


class RandomNeuronBranching(NeuronBranchingHeuristic):
    """Randomly choose k neurons among all layers."""

    def compute_neuron_scores(self, domains, **kwargs):
        scores = {}
        for idx, lb in self.layer_iterator(domains['lower_bounds']):
            # Random score 0 - 1.
            scores[idx] = domains['mask'][idx] * torch.rand_like(lb).flatten(1)
        return scores


class InterceptBranching(NeuronBranchingHeuristic):
    """Branching using the intercept term in ReLU relaxation."""

    def compute_neuron_scores(self, domains, **kwargs):
        """
        Args:
            lAs: A list or dictionary of lA matrices for each splittable layer.
            Note that only the lA matrices for splittable layers are saved to
            save space.
            Other args of this function are the same as the parent class.
        """
        scores = {}
        for idx, lb in self.layer_iterator(domains['lower_bounds']):
            ub = domains['upper_bounds'][idx]
            ub_c, lb_c = ub.clamp(min=0.), lb.clamp(max=0.)
            lA_key = self.net.split_activations[idx][0][0].name
            A = domains['lAs'][lA_key]  # Size is (batch, spec, neurons)
            assert A.size(1) == 1
            scores[idx] = ((- ub_c * lb_c) / (ub - lb + 1e-8)
                           * A.squeeze(1).abs()).flatten(1)
        return scores
