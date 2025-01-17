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
"""Pool of adversarial examples. We use them to suggest subdomains for searching better adversarial examples."""

import math
import torch
import arguments
from auto_LiRPA import BoundedModule
from sortedcontainers import SortedList


class AdvExample:
    """A single adversarial example."""

    def __init__(self, x, obj, pattern):
        self.x = x
        self.obj = obj  # smaller obj is better, we want it to be negative.
        self.activation_pattern = pattern  # A list, each element is the activation pattern of one layer's neuron.

    def __lt__(self, other):
        return self.obj < other.obj

    def __le__(self, other):
        return self.obj <= other.obj

    def __eq__(self, other):
        return self.obj == other.obj


class AdvExamplePool:
    """Keeps a pool of best adversarial examples so far."""

    """We only keep the best `capacity` adversarial images."""
    def __init__(self, network, unstable_mask, capacity=100, C=None):
        assert isinstance(network, BoundedModule)
        self.net = network
        self.unstable_mask = [m.squeeze(0).int().cpu() for m in unstable_mask]  # Initial unstable neuron masks.
        self.total_unstable = sum([m.sum().item() for m in self.unstable_mask])
        self.capacity = 100
        self.adv_pool = SortedList()
        self.C = C  # Output specification in [1, num_of_label_verify, out_dim]
        self.nlayers = len(self.net.relus)
        self.threshold = arguments.Config["bab"]["attack"]["adv_pool_threshold"]
        if self.threshold is not None:
            print('Threshold of adding examples in adv_pool is set to: {} by config file'.format(self.threshold))

    """Add adversarial examples to the pool. Expect adv_images in (N, C, H, W) format."""
    def add_adv_images(self, adv_images, max_to_add=float("inf")):
        if adv_images.size(0) == 0:
            return
        with torch.no_grad():
            # reset_perturbed_nodes must be False otherwise the .perturbed property will be missing.
            pred = self.net(adv_images, reset_perturbed_nodes=False).cpu()  # Pred has size [batch, out_dim]
        adv_images = adv_images.cpu()
        if self.C is not None:
            # Compute the "margin", < 0 means attack successful.
            pred = pred.matmul(self.C.cpu()[0].transpose(-1, -2)).squeeze(-1)
        else:
            assert pred.size(1) == 1
            # Prediction is just one element.
            pred = pred.squeeze(1)
        print("AdvPool received image with prediction:", pred)
        # Get all activations and transfer them to CPU.
        activations = [None] * self.nlayers
        for layer_i, layer in enumerate(self.net.relus):
            # +1 => active, 0 => inactive
            activations[layer_i] = (layer.inputs[0].forward_value.flatten(1) > 0).int().cpu()
        # Set activations for each example.
        c_replaced = 0
        c_added = 0
        c_rejected = 0
        for adv_i in range(adv_images.size(0)):
            if len(self.adv_pool) >= self.capacity:
                if self.adv_pool[-1].obj > pred[adv_i].item():
                    self.adv_pool.pop(-1)  # Remove the worst adversarial example.
                else:
                    # This adversarial example is worse than the worst one in our list. Skip it.
                    c_rejected += 1
                    continue
            example_activations = [None] * self.nlayers
            for layer_i, layer in enumerate(self.net.relus):
                example_activations[layer_i] = activations[layer_i][adv_i]
            c_replaced, c_added, c_rejected = self.replace_adv_example(AdvExample(adv_images[adv_i], pred[adv_i].item(), example_activations), c_replaced, c_added, c_rejected)
        print('Number of adv examples replaced/added/rejected: {}/{}/{}'.format(c_replaced, c_added, c_rejected))

    def print_pool_status(self):
        print('Current adv_pool statistic: length {}, variance {}, Manhattan distance mean {}, min {}, max {}, std {:.4f}'.format(len(self.adv_pool), *self.get_var()))
        for t in [0.6, 0.8, 1.0]:
            act_length = len(self.get_activation_pattern_from_pool(prob_threshold=t)[0])
            print(f'{t*100}% threshold will select {act_length} neurons.')

    """replace the closest adv examples in adv_pool if the diff < threshold"""
    def replace_adv_example(self, adv_example, c_replaced, c_added, c_rejected):
        if len(self.adv_pool) > arguments.Config["attack"]["pgd_restarts"]:
            current_adv_patterns = torch.stack([torch.cat([ii.flatten() for ii in i.activation_pattern]) for i in self.adv_pool])
            this_pattern = torch.cat([ii.flatten() for ii in adv_example.activation_pattern]).view(1, -1)
            diff = torch.cdist(current_adv_patterns.float(), this_pattern.float(), p=0).flatten()
            min_idx = diff.argmin()
            if self.threshold is None:
                # we set the threshold as the lowest diff when the first time we need to filter
                if diff[min_idx] > 1.:
                    self.threshold = diff[min_idx]
                else:
                    self.threshold = 1.
                print('Threshold of adding examples in adv_pool is automatically set to: {}'.format(self.threshold))
            if diff[min_idx] < self.threshold and adv_example.obj < self.adv_pool[min_idx].obj:
                # print('Replace adv_example.')
                self.adv_pool.pop(min_idx)  # pop the closest adv examples in adv_pool
                self.adv_pool.add(adv_example)
                return c_replaced + 1, c_added, c_rejected
            elif adv_example.obj < self.adv_pool[min_idx].obj or diff[min_idx] >= self.threshold:
                # add to adv_pool either the example has better obj or different pattern
                # print('Add better adv_example without replacement.')
                self.adv_pool.add(adv_example)
                return c_replaced, c_added + 1, c_rejected
            else:
                return c_replaced, c_added, c_rejected + 1
        else:
            self.adv_pool.add(adv_example)
            return c_replaced, c_added + 1, c_rejected

    """get the variance of activation patterns in current adv_pool"""
    def get_var(self):
        if len(self.adv_pool) > 0:
            current_adv_patterns = torch.stack([torch.cat([ii.flatten() for ii in i.activation_pattern]) for i in self.adv_pool[1:]])
            best_adv_pattern = torch.cat([ii.flatten() for ii in self.adv_pool[0].activation_pattern]).view(1, -1)
            variance = current_adv_patterns.float().var(0).mean()  # var across length of adv_pool
            distance = torch.cdist(current_adv_patterns.float(), best_adv_pattern.float(), p=0).flatten()
            return variance, distance.mean(), distance.min(), distance.max(), distance.std()
        else:
            return None

    """
    Given an probability threshold, return the common activation shared by at least threshold% adversarial examples.
    If "find_uncommon" is set to True, we find the least common activations, and the coeffs returned are invalid (only decision matters).
    """
    def get_activation_pattern_from_pool(self, prob_threshold=1.0, find_uncommon=False, n_advs=-1):
        all_layer_decisions = []
        all_layer_coeffs = []
        if n_advs == -1:
            # Use entire pool.
            selected_advs = self.adv_pool
        else:
            # Choose the top-n adversarial examples.
            selected_advs = self.adv_pool[:n_advs]
        # Threshold of number of examples with active/inactive neurons.
        pos_threshold = min(int(math.ceil(prob_threshold * len(selected_advs))), len(selected_advs))
        neg_threshold = len(selected_advs) - pos_threshold
        if pos_threshold == neg_threshold:
            # Cannot be equal - otherwise one neuron will be counted in both category.
            neg_threshold -= 1
        # Collect all activation patterns.
        all_patterns = [[] for i in range(self.nlayers)]
        for adv in selected_advs:
            for layer_i in range(self.nlayers):
                all_patterns[layer_i].append(adv.activation_pattern[layer_i])
        # Concatenate activation patterns across examples.
        for layer_i in range(self.nlayers):
            layer_i_pattern = torch.stack(all_patterns[layer_i], dim=0)
            # Shape is (batch, neurons).
            acc_pattern = layer_i_pattern.sum(dim=0)
            # +1 for most neurons active, -1 for most neurons inactive. 0 means this neuron is not selected due to the threshold or unstable mask.
            remaining_acc_pattern = ((acc_pattern >= pos_threshold).int() - (acc_pattern <= neg_threshold).int()) * self.unstable_mask[layer_i]
            if find_uncommon:
                nnz = ((remaining_acc_pattern == 0) * self.unstable_mask[layer_i]).nonzero(as_tuple=True)[0]
            else:
                nnz = remaining_acc_pattern.nonzero(as_tuple=True)[0]
            # Decision is an array with (layer_i, neuron_ids).
            decisions = torch.stack([torch.empty_like(nnz).fill_(layer_i), nnz], dim=1)
            # Decision is an array with +1/-1. (0 when find_uncommon is True).
            coeffs = remaining_acc_pattern[nnz]
            all_layer_decisions.append(decisions)
            all_layer_coeffs.append(coeffs)

        decisions = torch.cat(all_layer_decisions, dim=0)
        coeffs = torch.cat(all_layer_coeffs, dim=0)

        return decisions, coeffs

    """Return a list of unstable neurons, ranked by how common they are in the pool."""
    def get_ranked_activation_pattern(self, n_activations=-1, blacklist=None, find_uncommon=False, n_advs=-1, random_keep=False, softmax_temperature=10.0):
        if n_advs == -1:
            # Use entire pool.
            selected_advs = self.adv_pool
        else:
            # Choose the top-n adversarial examples.
            selected_advs = self.adv_pool[:n_advs]
        if isinstance(blacklist, list):
            blacklist = torch.tensor(blacklist, device='cpu', dtype=torch.long)
        # Collect all activation patterns.
        all_patterns = [[] for i in range(self.nlayers)]
        all_counts = [None] * self.nlayers
        all_status = [None] * self.nlayers  # Dominating neuron activation status per layer.
        for adv in selected_advs:
            for layer_i in range(self.nlayers):
                all_patterns[layer_i].append(adv.activation_pattern[layer_i])
        total_unstable = 0
        # Concatenate activation patterns across examples and count activations.
        for layer_i in range(self.nlayers):
            layer_i_pattern = torch.stack(all_patterns[layer_i], dim=0)
            # Count number of active neurons.
            counts = layer_i_pattern.sum(dim=0)
            # Total number of adv examples.
            total = layer_i_pattern.size(0)
            # Move the counts to reflect both sides (active/inactive).
            # A larger value means more common activations. Stable neurons will have negative counts.
            all_counts[layer_i] = (counts.float() - total / 2).abs()
            enabled_mask = self.unstable_mask[layer_i].clone()
            # Find out the neuron idx that are masked in this layer.
            if blacklist is not None:
                disabled_idx = blacklist[blacklist[:, 0] == layer_i][:, 1].squeeze()
                enabled_mask[disabled_idx] = 0
            # Make disable neurons' counts negative
            all_counts[layer_i] -= 2 * total * (1 - enabled_mask)
            total_unstable += enabled_mask.sum().item()
            # Store the dominating activation status.
            all_status[layer_i] = counts > (total / 2)
        # Find the boundaries of counts.
        counts_length = torch.cumsum(torch.tensor([c.numel() for c in all_counts]), dim=0).tolist()
        counts_length = torch.tensor([0] + counts_length)
        # Flatten the counts tensor.
        flat_all_counts = torch.cat(all_counts, dim=0)
        flat_all_status = torch.cat(all_status, dim=0)
        # Sort the counts, largest ones at the beginning. (negative means disabled neurons; but they should be removed by [:total_untable]).
        sort_idx = flat_all_counts.argsort(descending=True)[:total_unstable]
        # Find out the original neuron layer/idx.
        sorted_layer_idx = torch.searchsorted(counts_length, sort_idx, right=True) - 1
        # Find out the location at each layer.
        sorted_neuron_idx = sort_idx - counts_length[sorted_layer_idx]
        decisions = torch.stack([sorted_layer_idx, sorted_neuron_idx], dim=1)
        # Find out the activation pattern.
        coeffs = (flat_all_status[sort_idx].float() - 0.5) * 2

        # Now we get decisions and coeffs ranked from most common to most uncommon.
        # If we want to keep the most uncommon ones, just reverse the list.
        if find_uncommon:
            decisions = reversed(decisions)
            coeffs = reversed(coeffs)

        if n_activations != -1:
            if random_keep:
                # Randomly keep n_activations, based on commonness.
                scores = flat_all_counts.sort(descending=True).values[:total_unstable]
                if find_uncommon:
                    # Negate the score, so the most uncommon ones become the first.
                    scores = -reversed(scores)
                # Convert scores to probabilities.
                probs = torch.nn.functional.softmax(scores / softmax_temperature, dim=0)
                # Sample based on the probabilities.
                selected_indices = probs.multinomial(n_activations, replacement=False)
                return decisions[selected_indices], coeffs[selected_indices]

            else:
                # Keep most common ones.
                return decisions[:n_activations], coeffs[:n_activations]
        else:
            return decisions, coeffs

    """
    Given a list of unstable neurons, find the top-1 probability configuration based on activation database.
    `decisions` is a list of neurons in [(layer, neuron_idx), (layer, neuron_idx), ... ].
    """
    def find_most_likely_activation(self, decisions, n_advs=-1):
        # This will get the pattern for all unstable neurons.
        adv_decisions, adv_coeffs = self.get_activation_pattern_from_pool(prob_threshold=0.5, n_advs=n_advs)
        adv_decisions = adv_decisions.tolist()
        adv_coeffs = adv_coeffs.tolist()
        coeffs = []
        for d in decisions:
            assert d in adv_decisions
            coeffs.append(adv_coeffs[adv_decisions.index(d)])
        return coeffs

    """Returns activation patterns as a (decisions, coeffs) format, and exclude neurons in blacklist in decisions format."""
    def get_activation_pattern(self, adv_example, blacklist=None):
        if isinstance(blacklist, list):
            blacklist = torch.tensor(blacklist, device='cpu', dtype=torch.long)
        all_layer_decisions = []
        all_layer_coeffs = []
        for layer_idx, layer_pattern in enumerate(adv_example.activation_pattern):
            # layer_pattern is 0 when inactive, 1 active.
            layer_size = len(layer_pattern)
            # Generate the list of all neurons in this layer.
            all_decisions_this_layer = torch.stack([torch.empty(layer_size, dtype=torch.int32).fill_(layer_idx),
                torch.arange(layer_size, dtype=torch.int32)], dim=1)
            enabled_mask = torch.ones(layer_size, dtype=torch.bool)
            enabled_mask = torch.logical_and(enabled_mask, self.unstable_mask[layer_idx])
            if blacklist is not None:
                # Find out the neuron idx that are masked in this layer.
                disabled_mask = blacklist[blacklist[:, 0] == layer_idx][:, 1].squeeze()
                enabled_mask[disabled_mask] = 0
            all_layer_decisions.append(all_decisions_this_layer[enabled_mask])
            # Converting 0, 1 mask to -1, +1 coeffs.
            all_layer_coeffs.append((layer_pattern[enabled_mask] - 0.5) * 2)

        # Concat all layer decisions and coeffs into 1.
        decisions = torch.cat(all_layer_decisions, dim=0)
        coeffs = torch.cat(all_layer_coeffs, dim=0)

        return decisions, coeffs

    """
    Cross-over activation patterns from different adversarial examples.
    """
    def generate_crossover(self):
        # TODO.
        pass

