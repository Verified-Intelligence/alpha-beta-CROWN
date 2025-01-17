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
import numpy as np

from heuristics.base import NeuronBranchingHeuristic
from heuristics.utils import compute_ratio, get_preact_params
from utils import get_reduce_op, get_batch_size_from_masks


class BabsrBranching(NeuronBranchingHeuristic):
    def __init__(self, net):
        super().__init__(net)
        self.icp_score_counter = 0

    def babsr_score(self, lower_bounds, upper_bounds, lAs,
                    mask, reduce_op, number_bounds, prioritize_alphas='none'):
        """Compute branching scores for kfsb.
        lower_bounds: [lower_bounds1, lower_bounds2, ...], lower bounds for different pre-activation layers.
        upper_bounds: [upper_bounds1, upper_bounds2, ...], upper bounds for different pre-activation layers.
        lAs: list, A matrix used in CROWN for all pre-activation layers.
        batch: int, batch size for current branching.
        mask: list, mask indicates whether the neuron in this layer is unstable or not, 1: unstable, 0: stable.
        reduce_op: min() or max(), consider min or max info for two branches, similar to BFS (min) or DFS (max).
        number_bounds: int, the number of bounds that will output for one property.
        prioritize_alphas: 'none', 'positive', 'negative',  Prioritize splits with only positive/negative lA or none.

        return
        score: list, same structure as lower_bounds indicates the score for all neurons.
        intercept_tb: list, same as score's structure, only contain the  intercept scores.
        """
        batch = get_batch_size_from_masks(mask)
        score = []
        intercept_tb = []
        relu_idx = -1
        small_score_threshold = 1e-4
        big_constant = 1e6

        def normalize_scores(scores, normal_score_idx, reduced_score_idx, larger_is_better=True):
            #  We want to reduce all scores in the reduced_score_idx set, so they are no better than the scores in the normal_score_idx set.
            if larger_is_better:
                thresh = small_score_threshold
                # idx is a mask, setting irrelevant scores to 0. Valid scores are positive.
                best_score_in_reduced_set = torch.max(
                    scores * reduced_score_idx, dim=1).values
                worst_score_in_normal_set = torch.clamp_min(( # Setting irrelevant scores to inf.
                    torch.min(
                        scores * normal_score_idx
                        + (1.0 - normal_score_idx) * big_constant,
                        dim=1).values), thresh)
            else:
                thresh = -small_score_threshold
                # idx is a mask, setting irrelevant scores to 0. Valid scores are negative.
                best_score_in_reduced_set = torch.min(
                    scores * reduced_score_idx, dim=1).values
                worst_score_in_normal_set = torch.clamp_max(torch.max(
                    scores * normal_score_idx
                    - (1.0 - normal_score_idx) * big_constant,
                    dim=1).values, thresh)
            # Sizes are (batch,).
            ratio = torch.clamp_max(
                worst_score_in_normal_set / (best_score_in_reduced_set + thresh),
                1.0)
            # Make the scores in the reduced_score_idx set smaller.
            adjusted_scores = (scores * normal_score_idx
                               + scores * reduced_score_idx * ratio.unsqueeze(1))
            return adjusted_scores

        # Compute BaBSR scores, starting from the last layer.
        for layer_i, layer in enumerate(reversed(self.net.split_nodes)):
            assert len(self.net.split_activations[layer.name]) == 1
            layer = self.net.split_activations[layer.name][0][0]
            key = layer.inputs[0].name
            lA_key = layer.name
            this_layer_mask = mask[key].unsqueeze(1)
            if prioritize_alphas == 'positive':
                # Prioritize splits with only positive lA.
                normal_score_mask = (lAs[lA_key] >= 0).view(batch, number_bounds, -1) * this_layer_mask
                reduced_score_mask = (lAs[lA_key] < 0).view(batch, number_bounds, -1) * this_layer_mask
            elif prioritize_alphas == 'negative':
                # Prioritize splits with only positive lA.
                normal_score_mask = (lAs[lA_key] <= 0).view(batch, number_bounds, -1) * this_layer_mask
                reduced_score_mask = (lAs[lA_key] > 0).view(batch, number_bounds, -1) * this_layer_mask
            elif prioritize_alphas != 'none':
                raise ValueError(f'Unknown prioritize_alphas parameter {prioritize_alphas}')

            ratio = lAs[lA_key]
            ratio_temp_0, ratio_temp_1 = compute_ratio(
                lower_bounds[key], upper_bounds[key])

            # Intercept score, used as a backup score in BaBSR. A lower (more negative) score is better.
            intercept_temp = torch.clamp(ratio, max=0)
            intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
            reshaped_intercept_candidate = intercept_candidate.view(
                batch, number_bounds, -1) * this_layer_mask
            # In case for AND clauses, there are multiple bounds outputs
            # we need to calculate mean over number_bounds dim to get a average score
            reshaped_intercept_candidate = reshaped_intercept_candidate.mean(1)
            if prioritize_alphas != 'none':
                adjusted_intercept_candidate = normalize_scores(
                    reshaped_intercept_candidate, normal_score_mask,
                    reduced_score_mask, larger_is_better=False)
            else:
                adjusted_intercept_candidate = reshaped_intercept_candidate
            # intercept_tb is a list of intercept scores, each with a array of (batch, neuron).
            intercept_tb.insert(0, adjusted_intercept_candidate)

            b_temp = get_preact_params(layer)
            # In some cases, bias=0, we can't treat it like tensors
            if not isinstance(b_temp, int):
                b_temp = b_temp.view(-1, *([1] * (ratio.ndim - 3)))
            b_temp = b_temp * ratio
            # Estimated bounds of the two sides of the bounds.
            ratio_temp_0 = ratio_temp_0.unsqueeze(1)
            bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
            bias_candidate_2 = b_temp * ratio_temp_0
            bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
            score_candidate = bias_candidate + intercept_candidate
            score_candidate = score_candidate.abs().view(batch, number_bounds, -1) * this_layer_mask
            # In case for AND clauses, there are multiple bounds outputs
            # we need to calculate mean over number_bounds dim to get a average score
            score_candidate = score_candidate.mean(1)
            if prioritize_alphas != 'none':
                adjusted_score_candidate = normalize_scores(
                    score_candidate, normal_score_mask, reduced_score_mask,
                    larger_is_better=True)
                remaining_branches = normal_score_mask.sum(dim=1, dtype=torch.int32)
                print(f'layer {len(self.net.split_nodes) - layer_i} '
                      'remaining preferred branching variables: '
                      f'{remaining_branches[:10].tolist()}, '
                      f'avg {remaining_branches.sum().item() / remaining_branches.numel()}')
            else:
                adjusted_score_candidate = score_candidate
            # alpha score, the main score in BaBSR. A higher (more positive) score is batter.
            score.insert(0, adjusted_score_candidate)

            relu_idx -= 1

        return score, intercept_tb

    @torch.no_grad()
    def get_branching_decisions(self, domains, split_depth,
                                branching_reduceop='min',
                                prioritize_alphas='none',
                                sparsest_layer=0, max_info_threshold=0.001,
                                **kwargs):
        """
        choose the dimension to split on
        based on each node's contribution to the cost function
        in the KW formulation.

        sparsest_layer: if all layers are dense, set it to -1
        max_info_threshold: if the maximum score is below the threshold,
                            we consider it to be non-informative
        """

        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']

        batch = get_batch_size_from_masks(orig_mask)
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = get_reduce_op(branching_reduceop, with_dim=False)

        number_bounds = 1 if cs is None else cs.shape[1]
        score, intercept_tb = self.babsr_score(
            lower_bounds, upper_bounds, lAs, mask, reduce_op,
            number_bounds, prioritize_alphas)

        decision = [[] for _ in range(batch)]

        random_dict = {}
        for b in range(batch):
            mask_item = [mask[node.name][b] for node in self.net.split_nodes]
            new_score = [score[j][b] for j in range(len(score))]
            split_depth = min(split_depth, new_score[0].shape[0])
            max_info = [torch.topk(i, split_depth, 0) for i in new_score]

            max_info_index = [a[1] for a in max_info]
            max_info = [a[0] for a in max_info]  # [num_layer, split_depth]

            _, max_info_top_k_index = torch.topk(torch.cat(max_info, dim=0), split_depth)

            for l in range(split_depth):
                decision_layer = max_info_top_k_index[l].item() // split_depth
                decision_index = max_info_index[decision_layer][max_info_top_k_index[l] % split_depth].item()
                if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > max_info_threshold:
                    decision[b].append((decision_layer, decision_index))
                    mask_item[decision_layer][decision_index] = 0
                else:
                    min_info = [[i, torch.min(intercept_tb[i][b], 0)] for i in range(len(intercept_tb)) if
                                torch.min(intercept_tb[i][b]) < -1e-4]

                    if len(min_info) != 0 and self.icp_score_counter < 2 and (
                    min_info[-1][0], min_info[-1][1][1].item()) not in decision[b]:
                        intercept_layer = min_info[-1][0]
                        intercept_index = min_info[-1][1][1].item()
                        self.icp_score_counter += 1
                        decision[b].append((intercept_layer, intercept_index))
                        mask_item[intercept_layer][intercept_index] = 0
                        if intercept_layer != 0:
                            self.icp_score_counter = 0
                    else:
                        random_dict[b] = random_dict.get(b, 0) + 1
                        for preferred_layer in np.random.choice(len(self.net.split_indices), len(self.net.split_indices), replace=False):
                            if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                                decision[b].append(
                                    (preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()))
                                mask_item[decision[b][-1][0]][decision[b][-1][1]] = 0
                                break
                        self.icp_score_counter = 0
        if random_dict:
            print(f'Random branching decision used for {{example_idx:n_random}}: {random_dict}')

        split_depth = min([len(d) for d in decision])

        decision = [[batch[i] for batch in decision] for i in
                    range(split_depth)]  # change the order of final decision to split_depth * batch
        decision = sum(decision, [])

        return decision, None, split_depth  # None for points
