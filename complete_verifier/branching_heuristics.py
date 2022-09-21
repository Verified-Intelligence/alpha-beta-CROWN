#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from typing_extensions import final
from collections import defaultdict
import math

from numpy.lib.twodim_base import mask_indices
import torch
import numpy as np
from itertools import groupby

from torch import nn
from torch.nn import functional as F

from model_defs import Flatten
from auto_LiRPA.bound_ops import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd
import arguments

Icp_score_counter = 0


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio, intercept


def get_branching_op(branching_reduceop):
    if branching_reduceop == 'min':
        reduce_op = torch.min
    elif branching_reduceop == 'max':
        reduce_op = torch.max
    elif branching_reduceop == 'mean':
        reduce_op = torch.mean
    else:
        reduce_op = None
    return reduce_op


@torch.no_grad()
def choose_node_conv(lower_bounds, upper_bounds, orig_mask, layers, pre_relu_indices, icp_score_counter, random_order,
                     sparsest_layer, decision_threshold=0.001, gt=False):
    """
    choose the dimension to split on
    based on each node's contribution to the cost function
    in the KW formulation.

    sparsest_layer: if all layers are dense, set it to -1
    decision_threshold: if the maximum score is below the threshold,
                        we consider it to be non-informative
    random_order: priority to each layer when making a random choice
                  with preferences. Increased preference for later elements in the list
    """

    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    score = []
    intercept_tb = []
    random_choice = random_order.copy()

    ratio = torch.ones(1).to(lower_bounds[0].device)
    # starting from 1, back-propogating: if the weight is negative
    # introduce bias; otherwise, intercept is 0
    # we are only interested in two terms for now: the slope x bias of the node
    # and bias x the amount of argumentation introduced by later layers.
    # From the last relu-containing layer to the first relu-containing layer

    # Record score in a dic
    # new_score = {}
    # new_intercept = {}
    relu_idx = -1

    for layer_idx, layer in reversed(list(enumerate(layers))):
        if type(layer) is nn.Linear:
            ratio = ratio.unsqueeze(-1)
            w_temp = layer.weight.detach()
            ratio = torch.t(w_temp) @ ratio
            ratio = ratio.view(-1)
            # import pdb; pdb.set_trace()

        elif type(layer) is nn.ReLU:
            # compute KW ratio
            ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                       upper_bounds[pre_relu_indices[relu_idx]])
            # Intercept
            intercept_temp = torch.clamp(ratio, max=0)
            intercept_candidate = intercept_temp * ratio_temp_1
            intercept_tb.insert(0, intercept_candidate.view(-1) * mask[relu_idx])

            # Bias
            b_temp = layers[layer_idx - 1].bias.detach()
            if type(layers[layer_idx - 1]) is nn.Conv2d:
                b_temp = b_temp.unsqueeze(-1).unsqueeze(-1)
            ratio_1 = ratio * (ratio_temp_0 - 1)
            bias_candidate_1 = b_temp * ratio_1
            ratio = ratio * ratio_temp_0
            bias_candidate_2 = b_temp * ratio
            bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
            # test = (intercept_candidate!=0).float()
            # ???if the intercept_candiate at a node is 0, we should skip this node
            #    (intuitively no relaxation triangle is introduced at this node)
            #    score_candidate = test*bias_candidate + intercept_candidate
            score_candidate = bias_candidate + intercept_candidate
            score.insert(0, (abs(score_candidate).view(-1) * mask[relu_idx]).cpu())

            relu_idx -= 1

        elif type(layer) is nn.Conv2d:
            # import pdb; pdb.set_trace()
            ratio = ratio.unsqueeze(0)
            ratio = F.conv_transpose2d(ratio, layer.weight, stride=layer.stride, padding=layer.padding)
            ratio = ratio.squeeze(0)

        elif type(layer) is Flatten:
            # import pdb; pdb.set_trace()
            ratio = ratio.reshape(lower_bounds[layer_idx].size())
        else:
            raise NotImplementedError

    max_info = [torch.max(i, 0) for i in score]
    decision_layer = max_info.index(max(max_info))
    decision_index = max_info[decision_layer][1].item()
    if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > decision_threshold:
        # temp = torch.zeros(score[decision_layer].size())
        # temp[decision_index]=1
        # decision_index = torch.nonzero(temp.reshape(mask[decision_layer].shape))[0].tolist()
        decision = [decision_layer, decision_index]

    else:
        min_info = [[i, torch.min(intercept_tb[i], 0)] for i in range(len(intercept_tb)) if
                    torch.min(intercept_tb[i]) < -1e-4]
        # import pdb; pdb.set_trace()
        if len(min_info) != 0 and icp_score_counter < 2:
            intercept_layer = min_info[-1][0]
            intercept_index = min_info[-1][1][1].item()
            icp_score_counter += 1
            # inter_temp = torch.zeros(intercept_tb[intercept_layer].size())
            # inter_temp[intercept_index]=1
            # intercept_index = torch.nonzero(inter_temp.reshape(mask[intercept_layer].shape))[0].tolist()
            decision = [intercept_layer, intercept_index]
            if intercept_layer != 0:
                icp_score_counter = 0
            print('\tusing intercept score')
        else:
            print('\t using a random choice')
            undecided = True
            while undecided:
                preferred_layer = random_choice.pop(-1)
                if len(mask[preferred_layer].nonzero()) != 0:
                    decision = [preferred_layer, mask[preferred_layer].nonzero()[0].item()]
                    undecided = False
                else:
                    pass
            icp_score_counter = 0
    if gt is False:
        return decision, icp_score_counter
    else:
        return decision, icp_score_counter, score


@torch.no_grad()
def choose_node_parallel_crown(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, sparsest_layer=0,
                               decision_threshold=0.001, batch=5, branching_reduceop='min', split_depth=1, cs=None, rhs=0):
    batch = min(batch, len(orig_mask[0]))
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    number_bounds = 1 if cs is None else cs.shape[1]

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # align with number of output dim
        ratio_temp_0, ratio_temp_1 = ratio_temp_0.unsqueeze(1), ratio_temp_1.unsqueeze(1)
        casted_mask = mask[relu_idx].unsqueeze(1)
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, (intercept_candidate.view(batch, number_bounds, -1) * casted_mask).mean(1))

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO: bias of BN need refine
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)

        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, (abs(score_candidate).view(batch, number_bounds, -1) * casted_mask).mean(1).cpu())

        relu_idx -= 1

    decision = [[] for _ in range(batch)]

    random_dict = {}
    for b in range(batch):
        mask_item = [m[b] for m in mask]
        new_score = [score[j][b] for j in range(len(score))]
        split_depth = min(split_depth, new_score[0].shape[0])
        max_info = [torch.topk(i, split_depth, 0) for i in new_score]

        max_info_index = [a[1] for a in max_info]
        max_info = [a[0] for a in max_info] # [num_layer, split_depth]

        max_info_top_k_value, max_info_top_k_index = torch.topk(torch.cat(max_info, dim=0), split_depth)

        for l in range(split_depth):
            decision_layer = max_info_top_k_index[l].item() // split_depth
            decision_index = max_info_index[decision_layer][max_info_top_k_index[l] % split_depth].item()

            if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > decision_threshold:
                decision[b].append((decision_layer, decision_index))
                mask_item[decision_layer][decision_index] = 0
            else:
                min_info = [[i, torch.min(intercept_tb[i][b], 0)] for i in range(len(intercept_tb)) if
                            torch.min(intercept_tb[i][b]) < -1e-4]

                global Icp_score_counter
                if len(min_info) != 0 and Icp_score_counter < 2 and (min_info[-1][0], min_info[-1][1][1].item()) not in decision[b]:
                    intercept_layer = min_info[-1][0]
                    intercept_index = min_info[-1][1][1].item()
                    Icp_score_counter += 1
                    decision[b].append((intercept_layer, intercept_index))
                    mask_item[intercept_layer][intercept_index] = 0
                    if intercept_layer != 0:
                        Icp_score_counter = 0
                else:
                    random_dict[b] = random_dict.get(b, 0) + 1
                    for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                        if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                            decision[b].append((preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()))
                            mask_item[decision[b][-1][0]][decision[b][-1][1]] = 0
                            break
                    Icp_score_counter = 0
    if random_dict:
            print(f'Random branching decision used for {{example_idx:n_random}}: {random_dict}')

    split_depth = min([len(d) for d in decision])

    decision = [[batch[i] for batch in decision] for i in range(split_depth)] # change the order of final decision to split_depth * batch
    decision = sum(decision, [])
    return decision, split_depth


@torch.no_grad()
def choose_node_parallel_FSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, branching_candidates=5, branching_reduceop='min', slopes=None,
                             betas=None, history=None, use_beta=False, split_depth=1, cs=None, rhs=0):

    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = min(branching_candidates, int(sum([i.sum() for i in mask]).item()))  # in case number of unstable neurons less than topk
    number_bounds = 1 if cs is None else cs.shape[1]

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        this_layer_mask = mask[relu_idx].unsqueeze(1)
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        ratio_temp_0 = ratio_temp_0.unsqueeze(1)
        ratio_temp_1 = ratio_temp_1.unsqueeze(1)
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, (intercept_candidate.view(batch, number_bounds, -1) * this_layer_mask).mean(1))

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, (abs(score_candidate).view(batch, number_bounds, -1) * this_layer_mask).mean(1))

        relu_idx -= 1

    final_decision = [[] for b in range(batch)]
    decision_tmp = {}
    tmp_ret = {}
    score_from_layer_idx = 1 if len(score) > 1 else 0
    skip_layers = list(range(score_from_layer_idx))

    # real batch = batch * 2, since we have two kinds of scores
    lbs = [torch.cat([i, i]) for i in lower_bounds]
    ups = [torch.cat([i, i]) for i in upper_bounds]
    if isinstance(slopes, dict):
        # per neuron slope.
        sps = defaultdict(dict)
        for k, vv in slopes.items():
            sps[k] = {}
            for kk, v in vv.items():
                sps[k][kk] = torch.cat([v, v], dim=2)
    else:
        sps = [torch.cat([i, i]) for i in slopes]
    if use_beta:
        bs = [torch.cat([i, i]) for i in betas]
        history += history
    rhs = torch.cat([rhs, rhs])
    if cs is not None:
        cs = torch.cat([cs, cs])

    set_slope = True  # We only set the slope once.
    for i in range(score_from_layer_idx, len(score)):
        if (score[i].max(1).values <= 1e-4).all() and (intercept_tb[i].min(1).values >= -1e-4).all():
            print('{}th layer has no valid scores'.format(i))
            skip_layers.append(i)
            continue

        score_idx = torch.topk(score[i], topk)
        score_idx_indices = score_idx.indices.cpu()
        itb_idx = torch.topk(intercept_tb[i], topk, largest=False)
        itb_idx_indices = itb_idx.indices.cpu()

        k_ret = torch.empty(size=(topk, batch*2), device=lower_bounds[0].device, requires_grad=False)
        k_decision = []
        for k in range(topk):
            decision_index = score_idx_indices[:, k]
            decision_max_ = [[i, j.item()] for j in decision_index]  # add decision_index with layer's idx
            decision_index = itb_idx_indices[:, k]
            decision_min_ = [[i, j.item()] for j in decision_index]

            k_decision.append(decision_max_ + decision_min_)

            # only save the best lower bounds of the two splits
            if use_beta:
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], betas=bs,
                                                       fix_intermediate_layer_bounds=True, shortcut=True, history=history, cs=cs)
            else:
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], beta=False,
                                                       fix_intermediate_layer_bounds=True, shortcut=True, cs=cs)
            # consider the max improvement among multi bounds in one C matrix
            k_ret_lbs = (k_ret_lbs - torch.cat([rhs, rhs])).max(-1).values
            # No need to set slope next time; we do not optimize the slopes.
            set_slope = False
            # print(f'layer {i} k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')

            mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
            mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
            # make the invalid lower bounds worse than normal lower bounds by minus 999999
            # we only consider the best lower bound across two splits by using min(0)
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

        split_depth = min(split_depth, k_ret.shape[0])
        i_idx = k_ret.topk(split_depth, dim=0)  # compare across topK
        tmp_ret[i] = i_idx.values  # [split_depth, batch*2]
        tmp_indice = i_idx.indices
        decision_tmp[i] = [k_decision[tmp_indice[ii//(2*batch)][ii%(2*batch)]][ii%(2*batch)] for ii in range(split_depth*(batch*2))]

    # shape of tmp_ret: [layer, num_split, batch*2]
    if len(tmp_ret) > 0:
        stacked_layers = torch.stack([i for i in tmp_ret.values()])  # [layer, split_depth, batch*2]
        max_ret = torch.topk(stacked_layers.view(-1, batch*2), split_depth, dim=0)  # compare across layers [split_depth, batch*2]
        # shape: [num_split*batch*2]
        rets, decision_layers = max_ret.values.view(-1).cpu().numpy(), max_ret.indices.view(-1).cpu().numpy()  # first batch: score; second batch: intercept_tb.
        decision_layers = decision_layers // split_depth

        # add index number for the skipped layers
        # for _, g in groupby(enumerate(skip_layers), lambda ix: ix[0] - ix[1]):
        #     decision_layers[decision_layers >= list(g)[-1][-1]] += 1
        for s in skip_layers:
            decision_layers[decision_layers >= s] += 1

        for l in range(split_depth):
            for b in range(batch):
                decision_layer_1, decision_index_1 = decision_tmp[decision_layers[2*l*batch+b].item()][l*2*batch+b]
                decision_layer_2, decision_index_2 = decision_tmp[decision_layers[2*l*batch+b+batch].item()][l*2*batch+b+batch]

                if max([s[b].max() for s in score]) > 1e-4 and min([s[b].min() for s in intercept_tb]) < -1e-4 \
                        and max(rets[2*l*batch+b], rets[2*l*batch+ b + batch]) > -10000 \
                            and (mask[decision_layer_1][b][decision_index_1] != 0 or mask[decision_layer_2][b][decision_index_2] != 0):  # make sure this potential split is valid
                    if rets[2*l*batch+b] > rets[2*l*batch+b+batch] and mask[decision_layer_1][b][decision_index_1] != 0:  # score > intercept_tb
                        final_decision[b].append(decision_tmp[decision_layers[2*l*batch+b].item()][l*2*batch+b])
                        mask[final_decision[b][-1][0]][b][final_decision[b][-1][1]] = 0
                    elif mask[decision_layer_2][b][decision_index_2] != 0:
                        final_decision[b].append(decision_tmp[decision_layers[2*l*batch+b+batch].item()][l*2*batch+b+batch])
                        mask[final_decision[b][-1][0]][b][final_decision[b][-1][1]] = 0
                    else:
                        mask_item = [m[b] for m in mask]
                        for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                            if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                                final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                                mask[final_decision[b][-1][0]][b][final_decision[b][-1][1]] = 0
                                break
                else:
                    # print('\t using a random choice')
                    mask_item = [m[b] for m in mask]
                    for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                        if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                            final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                            mask[final_decision[b][-1][0]][b][final_decision[b][-1][1]] = 0
                            break
    else:
        # all layers are split or has no improvement
        for b in range(split_depth*batch):
            # print('\t using a random choice')
            mask_item = [m[b] for m in mask]
            # for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):  # random
            for preferred_layer in reversed(range(len(pre_relu_indices))):  # from last layer to first layer
                if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                    final_decision[b].append(
                        [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                    mask[final_decision[b][-1][0]][b][final_decision[b][-1][1]] = 0
                    break

    split_depth = min([len(d) for d in final_decision])

    final_decision = [[batch[i] for batch in final_decision] for i in range(split_depth)] # change the order of final decision to split_depth * batch
    final_decision = sum(final_decision, [])

    return final_decision, split_depth #[split_depth*batch]


def branching_scores_kfsb_intercept_only(lower_bounds, upper_bounds, lAs, batch):
    """Compute branching scores for kfsb based on intercept only."""
    score = []
    for layer_idx, (lbs, ubs) in enumerate(zip(lower_bounds[:-1], upper_bounds[:-1])):
        ratio = ((- lbs).clamp(0, None) * ubs.clamp(0, None)) / (ubs - lbs)
        ratio *= (- lAs[layer_idx].mean(dim=1)).clamp(0, None)
        score.append(ratio.reshape(batch, -1))
    return score


def branching_scores_kfsb(lower_bounds, upper_bounds, net, pre_relu_indices, lAs, batch,
                          mask, reduce_op, number_bounds, prioritize_slopes='none'):
    """Compute branching scores for kfsb."""
    score = []
    intercept_tb = []
    relu_idx = -1
    small_score_threshold = 1e-4
    big_constant = 1e6

    def normalize_scores(scores, normal_score_idx, reduced_score_idx, larger_is_better=True):
        #  We want to reduce all scores in the reduced_score_idx set, so they are no better than the scores in the normal_score_idx set.
        if larger_is_better:
            thresh = small_score_threshold
            get_best_score = lambda candidate, idx: torch.max(candidate * idx, dim=1).values  # idx is a mask, setting irrelevant scores to 0. Valid scores are positive.
            get_worst_score = lambda candidate, idx: torch.min(candidate * idx + (1.0 - idx) * big_constant, dim=1).values  # Setting irrelevant scores to inf.
            clamp_score = lambda candidate: torch.clamp_min(candidate, thresh)
        else:
            thresh = -small_score_threshold
            get_best_score = lambda candidate, idx: torch.min(candidate * idx, dim=1).values  # idx is a mask, setting irrelevant scores to 0. Valid scores are negative.
            get_worst_score = lambda candidate, idx: torch.max(candidate * idx - (1.0 - idx) * big_constant, dim=1).values  # Setting irrelevant scores to -inf.
            clamp_score = lambda candidate: torch.clamp_max(candidate, thresh)
        # Sizes are (batch,).
        best_score_in_reduced_set = get_best_score(scores, reduced_score_idx)
        worst_score_in_normal_set = clamp_score(get_worst_score(scores, normal_score_idx))
        ratio = torch.clamp_max(worst_score_in_normal_set / (best_score_in_reduced_set + thresh), 1.0)
        # Make the scores in the reduced_score_idx set smaller.
        adjusted_scores = scores * normal_score_idx + scores * reduced_score_idx * ratio.unsqueeze(1)
        return adjusted_scores

    # Compute BaBSR scores, starting from the last layer.
    for layer_i, layer in enumerate(reversed(net.net.relus)):
        this_layer_mask = mask[pre_relu_indices[relu_idx]].unsqueeze(1)
        if prioritize_slopes == 'positive':
            # Prioritize splits with only positive lA.
            normal_score_mask = (lAs[pre_relu_indices[relu_idx]] >= 0).view(batch, number_bounds, -1) * this_layer_mask
            reduced_score_mask = (lAs[pre_relu_indices[relu_idx]] < 0).view(batch, number_bounds, -1) * this_layer_mask
        elif prioritize_slopes == 'negative':
            # Prioritize splits with only positive lA.
            normal_score_mask = (lAs[pre_relu_indices[relu_idx]] <= 0).view(batch, number_bounds, -1) * this_layer_mask
            reduced_score_mask = (lAs[pre_relu_indices[relu_idx]] > 0).view(batch, number_bounds, -1) * this_layer_mask
        elif prioritize_slopes != 'none':
            raise ValueError(f'Unknown prioritize_slopes parameter {prioritize_slopes}')

        ratio = lAs[pre_relu_indices[relu_idx]]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])

        # Intercept score, used as a backup score in BaBSR. A lower (more negative) score is better.
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
        reshaped_intercept_candidate = intercept_candidate.view(batch, number_bounds, -1) * this_layer_mask
        reshaped_intercept_candidate = reshaped_intercept_candidate.mean(1)   # mean over number_bounds dim
        if prioritize_slopes != 'none':
            adjusted_intercept_candidate = normalize_scores(reshaped_intercept_candidate, normal_score_mask, reduced_score_mask, larger_is_better=False)
        else:
            adjusted_intercept_candidate = reshaped_intercept_candidate
        # intercept_tb is a list of intercept scores, each with a array of (batch, neuron).
        intercept_tb.insert(0, adjusted_intercept_candidate)

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            # for BN, bias is the -3th inputs
            b_temp = input_node.inputs[-3].param.detach().view(-1, *([1] * (ratio.ndim - 3)))

        b_temp = b_temp * ratio
        # Estimated bounds of the two sides of the bounds.
        ratio_temp_0 = ratio_temp_0.unsqueeze(1)
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score_candidate = score_candidate.abs().view(batch, number_bounds, -1) * this_layer_mask
        score_candidate = score_candidate.mean(1)  # mean over number_bounds dim
        if prioritize_slopes != 'none':
            adjusted_score_candidate = normalize_scores(score_candidate, normal_score_mask, reduced_score_mask, larger_is_better=True)
            remaining_branches = normal_score_mask.sum(dim=1, dtype=torch.int32)
            print(f'layer {len(net.net.relus) - layer_i} remaining preferred branching variables: {remaining_branches[:10].tolist()}, avg {remaining_branches.sum().item() / remaining_branches.numel()}')
        else:
            adjusted_score_candidate = score_candidate
        # Slope score, the main score in BaBSR. A higher (more positive) score is batter.
        score.insert(0, adjusted_score_candidate)

        relu_idx -= 1

    return score, intercept_tb


@torch.no_grad()
def choose_node_parallel_kFSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs,
                              branching_candidates=5, branching_reduceop='min', slopes=None,
                              betas=None, history=None, use_beta=False, keep_all_decision=False, prioritize_slopes='none',
                              split_depth=1, cs=None, rhs=0, method='kfsb'):

    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = min(branching_candidates, int(sum([i.sum() for i in mask]).item()))
    # FIXME: it seems cs and should always be not None because they are used below.
    number_bounds = 1 if cs is None else cs.shape[1]

    if method == 'kfsb-intercept-only':
        score = branching_scores_kfsb_intercept_only(lower_bounds, upper_bounds, lAs, batch)
    elif method == 'kfsb':
        score, intercept_tb = branching_scores_kfsb(lower_bounds, upper_bounds, net, pre_relu_indices,
                                                    lAs, batch, mask, reduce_op, number_bounds, prioritize_slopes)
    else:
        raise ValueError(f'Unsupported branching method "{method}" for relu splits.')

    final_decision = []

    if method == 'kfsb-intercept-only':
        # We only have one kind of score, so batch does not change.
        lbs, ups, sps, css, rhs = lower_bounds, upper_bounds, slopes, cs, rhs
        if use_beta:
            bs = betas
    elif method == 'kfsb':
        # real batch = batch * 2, since we have two kinds of scores.
        lbs = [torch.cat([i, i]) for i in lower_bounds]
        ups = [torch.cat([i, i]) for i in upper_bounds]
        if isinstance(slopes, dict):
            # per neuron slope.
            sps = defaultdict(dict)
            for k, vv in slopes.items():
                sps[k] = {}
                for kk, v in vv.items():
                    sps[k][kk] = torch.cat([v, v], dim=2)
        else:
            sps = [torch.cat([i, i]) for i in slopes]
        if use_beta:
            bs = [torch.cat([i, i]) for i in betas]
            history += history
        css = torch.cat([cs, cs])
        rhs = torch.cat([rhs, rhs])

    # Use score_length to convert an index to its layer and offset.
    score_length = np.cumsum([len(score[i][0]) for i in range(len(score))])
    score_length = np.insert(score_length, 0, 0)

    # Flatten the scores vector.
    all_score = torch.cat(score, dim=1)
    # Select top-k candidates among all layers for two kinds of scores.
    score_idx = torch.topk(all_score, topk)
    # These indices are the indices for the top-k scores in flatten
    score_idx_indices = score_idx.indices.cpu()
    if method == 'kfsb':
        all_itb = torch.cat(intercept_tb, dim=1)
        itb_idx = torch.topk(all_itb, topk, largest=False)  # k-smallest elements.
        itb_idx_indices = itb_idx.indices.cpu()

    k_decision = []
    batch_size = batch * 2 if method == 'kfsb' else batch
    k_ret = torch.empty(size=(topk, batch_size), device=lower_bounds[0].device, requires_grad=False)
    set_slope = True  # We only set the slope once.
    for k in range(topk):
        # top-k candidates from the slope scores.
        decision_index = score_idx_indices[:, k]
        # Find which layer and neuron this topk gradient belongs to.
        decision_max_ = []
        for l in decision_index:
            # Go over each element in this batch.
            l = l.item()
            # Recover the (layer, idx) from the flattend array.
            layer = np.searchsorted(score_length, l, side='right') - 1
            idx = l - score_length[layer]
            decision_max_.append([layer, idx])

        if method == 'kfsb-intercept-only':
            k_decision.append(decision_max_)
        elif method == 'kfsb':
            # top-k candidates from the intercept (backup) scores.
            decision_index = itb_idx_indices[:, k]
            # Find which layer and neuron this topk gradient belongs to.
            decision_min_ = []
            for l in decision_index:
                # Go over each element in this batch.
                l = l.item()
                layer = np.searchsorted(score_length, l, side='right') - 1
                idx = l - score_length[layer]
                decision_min_.append([layer, idx])
            # Stores the top-k decisions, so after finding the max we can go back to see which decision it is.
            k_decision.append(decision_max_ + decision_min_)

        # only save the best lower bounds of the two splits
        if use_beta:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], betas=bs,
                                                   fix_intermediate_layer_bounds=True, shortcut=True, history=history, cs=css)
        else:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], beta=False,
                                                   fix_intermediate_layer_bounds=True, shortcut=True, cs=css)
        # print(f'k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')

        # consider the max improvement among multi bounds in one C matrix
        k_ret_lbs = (k_ret_lbs - torch.cat([rhs, rhs])).max(-1).values
        # No need to set slope next time; we do not optimize the slopes.
        set_slope = False

        mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
        if method == 'kfsb-intercept-only':
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - mask_score.repeat(2) * 999999).reshape(2, -1), dim=0).values
        elif method == 'kfsb':
            mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
            # We first make the invalid lower bounds worse than normal lower bounds by minus 999999.
            # Then we consider the best lower bound across two splits (in the first dimension after reshape) by using min(0) or max(0).
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

    split_depth = min(split_depth, k_ret.shape[0])
    if not keep_all_decision:
        # k_ret has shape (top-k, batch*2) and we take the score eveluated using bound propagation based on the top-k choice.
        i_idx = k_ret.topk(split_depth, 0)
        rets = i_idx.values.cpu().numpy()
        rets_indices = i_idx.indices.cpu().numpy()

        # Given the indices of the max score, find what is its corresponding decision.
        decision_tmp = [[k_decision[rets_indices[l][ii]][ii] for ii in range(batch_size)] for l in range(split_depth)]

        # regular kfsb, select the top {split_depth} decision from k
        final_decision = [[] for b in range(batch)]
        random_decision_dict = {}

        # customize kfsb but sometimes duplicate split for k_decision[ki][b] and k_decision[ki][b+batch]
        for l in range(split_depth):
            for b in range(batch):
                # use mask to check if a node is unstable when random selection
                # not detach for now, please check!!!
                mask_item = [m[b] for m in mask]
                thres = max(rets[l][b], rets[l][b + batch]) if method == 'kfsb' else rets[l][b]
                if thres > -10000:
                    if method == 'kfsb-intercept-only':
                        decision = decision_tmp[l][b]
                    elif method == 'kfsb':
                        if rets[l][b] > rets[l][b + batch]:
                            decision = decision_tmp[l][b]
                        else:
                            decision = decision_tmp[l][b + batch]
                    if mask_item[decision[0]][decision[1]] != 0:
                        final_decision[b].append(decision)
                        # print(decision, mask_item[decision[0]][decision[1]])
                        # assert mask_item[decision[0]][decision[1]] == 1, "selected decision node should be unstable!"
                        mask_item[decision[0]][decision[1]] = 0

        for b in range(batch):
            mask_item = [m[b] for m in mask]
            if len(final_decision[b]) < split_depth:
                # No valid scores, have to choose neurons randomly.
                random_decision_dict[b] = split_depth - len(final_decision[b])
                for i in range(split_depth - len(final_decision[b])):
                    for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                        if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                            final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                            mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                            break
            assert len(final_decision[b]) <= split_depth, f"{len(final_decision[b])} <= {split_depth}"

        split_depth = min([len(d) for d in final_decision])

        final_decision = [[batch[i] for batch in final_decision] for i in range(split_depth)] # change the order of final decision to split_depth * batch
        final_decision = sum(final_decision, [])
        return final_decision, split_depth
    else:
        # keep all the k decisions
        # final_decision: batch -> k splits
        final_decision = [[] for _ in range(batch)]
        random_decision_dict = {}
        # customize kfsb but sometimes duplicate split for k_decision[ki][b] and k_decision[ki][b+batch]
        for b in range(batch):
            # use mask to check if a node is unstable when random selection
            # not detach for now, please check!!!
            mask_item = [m[b] for m in mask]
            for ki in range(topk):
                thres = max(k_ret[ki][b], k_ret[ki][b + batch]) if method == 'kfsb' else k_ret[ki][b]
                if thres > -10000:
                    if method == 'kfsb-intercept-only':
                        decision = k_decision[ki][b]
                    elif method == 'kfsb':
                        if k_ret[ki][b] > k_ret[ki][b + batch] and mask_item[k_decision[ki][b][0]][k_decision[ki][b][1]] == 1:
                            decision = k_decision[ki][b]
                        else:
                            decision = k_decision[ki][b + batch]
                    if mask_item[decision[0]][decision[1]] == 1:
                        final_decision[b].append(decision)
                        # print(decision, mask_item[decision[0]][decision[1]])
                        # assert mask_item[decision[0]][decision[1]] == 1, "selected decision node should be unstable!"
                        mask_item[decision[0]][decision[1]] = 0
            if len(final_decision[b]) < topk:
                # No valid scores, have to choose a neuron randomly.
                random_decision_dict[b] = topk - len(final_decision[b])
                for i in range(topk - len(final_decision[b])):
                    for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                        if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                            final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                            mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                            break
            assert len(final_decision[b]) <= topk, f"{len(final_decision[b])} <= {topk}"
            # del mask_item
        if random_decision_dict:
            print(f'Random branching decision used for {{example_idx:n_random}}: {random_decision_dict}')
        return final_decision, split_depth


@torch.no_grad()
def input_split_branching(net, dom_lb, dm_l_all, dm_u_all, lA, thresholds, branching_method, selected_dims, shape, slopes, split_depth=1):
    """
    Produce input split according to branching methods.
    """
    dm_l_all = dm_l_all.flatten(1)
    dm_u_all = dm_u_all.flatten(1)

    batch = len(dm_l_all)
    if branching_method == 'naive':
        # we just select the longest edge
        i_idx = torch.topk(dm_u_all - dm_l_all, split_depth, -1).indices
    elif branching_method == 'sb':
        lA = lA.view(lA.shape[0], lA.shape[1], -1)
        # lA shape: (batch, spec, # inputs)
        perturb = (dm_u_all - dm_l_all).unsqueeze(-2)
        # perturb shape: (batch, 1, # inputs)
        # dom_lb shape: (batch, spec)
        # thresholds shape: (batch, spec)
        lA_clamping_thresh = arguments.Config["bab"]["branching"]["sb_coeff_thresh"]
        assert lA_clamping_thresh >= 0
        score = dom_lb.to(lA.device)[..., None] + lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2 - thresholds[..., None]
        # score shape: (batch, spec, # inputs)
        score = score.amax(dim=-2)
        # note: the k (split_depth) in topk <= # inputs, because split_depth is computed as
        # min(max split depth, # inputs).
        # 1) If max split depth <= # inputs, then split_depth <= # inputs.
        # 2) If max split depth > # inputs, then split_depth = # inputs.
        i_idx = torch.topk(score, split_depth, -1).indices
    else:
        raise ValueError(f'Unsupported branching method "{branching_method}" for input splits.')

    return i_idx

@torch.no_grad()
def input_split_parallel(dm_l_all, dm_u_all, shape=None,
                         cs=None, thresholds=None, split_depth=1, i_idx=None):
    """
    Split the dm_l_all and dm_u_all given split_idx and split_depth.
    """
    dm_l_all = dm_l_all.flatten(1)
    dm_u_all = dm_u_all.flatten(1)

    dm_l_all_cp = dm_l_all.clone()
    dm_u_all_cp = dm_u_all.clone()

    # since we store using storage depth, we don't need the judgments here
    # if i_idx.shape[1] < dm_l_all.shape[1]:
    #     split_depth = min(split_depth, i_idx.shape[1])

    # if i_idx.shape[1] < dm_l_all.shape[1], this means that we need more split_depth than the pre-computed
    # i_idx computed in the last branching
    # So we need to clamp the split_depth

    remaining_depth = split_depth
    input_dim = dm_l_all.shape[1]
    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(dm_l_all_cp.shape[0])
            copy_num = dm_l_all_cp.shape[0]//dm_l_all.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            dm_l_all_cp_tmp = dm_l_all_cp.clone()
            dm_u_all_cp_tmp = dm_u_all_cp.clone()

            mid = (dm_l_all_cp[indices, idx] + dm_u_all_cp[indices, idx]) / 2

            dm_l_all_cp[indices, idx] = mid
            dm_u_all_cp_tmp[indices, idx] = mid
            dm_l_all_cp = torch.cat([dm_l_all_cp, dm_l_all_cp_tmp])
            dm_u_all_cp = torch.cat([dm_u_all_cp, dm_u_all_cp_tmp])
        remaining_depth -= min(input_dim, remaining_depth)

    split_depth = split_depth - remaining_depth

    new_dm_l_all = dm_l_all_cp.reshape(-1, *shape[1:])
    new_dm_u_all = dm_u_all_cp.reshape(-1, *shape[1:])

    if cs is not None:
        cs_shape = [2 ** split_depth] + [1] * (len(cs.shape) - 1)
        cs = cs.repeat(*cs_shape)
    if thresholds is not None:
        thresholds = thresholds.repeat(2 ** split_depth, 1)
    return new_dm_l_all, new_dm_u_all, cs, thresholds, split_depth


def get_split_depth(dm_l_all):
    split_depth = 1
    if len(dm_l_all) < arguments.Config["solver"]["min_batch_size_ratio"] * arguments.Config["solver"]["batch_size"]:
        min_batch_size = arguments.Config["solver"]["min_batch_size_ratio"] * arguments.Config["solver"]["batch_size"]
        split_depth = int(math.log(min_batch_size//len(dm_l_all))//math.log(2))
        split_depth = max(split_depth, 1)
    return split_depth
