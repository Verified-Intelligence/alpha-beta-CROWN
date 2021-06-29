import torch
import numpy as np
from itertools import groupby

from torch import nn
from torch.nn import functional as F

from modules import Flatten
from auto_LiRPA.bound_ops import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd

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
                  with preferences. Increased preference for later elements                  in the list

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
                               decision_threshold=0.001, batch=5, branching_reduceop='min'):
    batch = min(batch, len(orig_mask[0]))
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

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
        score.insert(0, (abs(score_candidate).view(batch, -1) * mask[relu_idx]).cpu())

        relu_idx -= 1

    decision = []
    for b in range(batch):
        new_score = [score[j][b] for j in range(len(score))]
        max_info = [torch.max(i, 0) for i in new_score]
        decision_layer = max_info.index(max(max_info))
        decision_index = max_info[decision_layer][1].item()

        if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > decision_threshold:
            decision.append([decision_layer, decision_index])
        else:
            min_info = [[i, torch.min(intercept_tb[i][b], 0)] for i in range(len(intercept_tb)) if
                        torch.min(intercept_tb[i][b]) < -1e-4]
            # import pdb; pdb.set_trace()
            global Icp_score_counter
            if len(min_info) != 0 and Icp_score_counter < 2:
                intercept_layer = min_info[-1][0]
                intercept_index = min_info[-1][1][1].item()
                Icp_score_counter += 1
                decision.append([intercept_layer, intercept_index])
                if intercept_layer != 0:
                    Icp_score_counter = 0
                # else:
                #     print('using first layer split')
                # print('\tusing intercept score')
            else:
                print('\t using a random choice')
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        decision.append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        break
                Icp_score_counter = 0

    return decision


@torch.no_grad()
def choose_node_parallel_FSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, branching_candidates=5, branching_reduceop='min', slopes=None,
                             betas=None, history=None, use_beta=False):

    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = branching_candidates

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

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

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, abs(score_candidate).view(batch, -1) * mask[relu_idx])

        relu_idx -= 1

    final_decision = []
    decision_tmp = {}
    tmp_ret = {}
    skip_layers = [0]

    # real batch = batch * 2, since we have two kinds of scores
    lbs = [torch.cat([i, i]) for i in lower_bounds]
    ups = [torch.cat([i, i]) for i in upper_bounds]
    if isinstance(slopes[0], dict):
        # per neuron slope.
        sps = slopes + slopes
    else:
        sps = [torch.cat([i, i]) for i in slopes]
    if use_beta:
        bs = [torch.cat([i, i]) for i in betas]
        history += history

    set_slope = True  # We only set the slope once.
    for i in range(1, len(score)):
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
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, betas=bs,
                                                       layer_set_bound=True, shortcut=True, history=history)
            else:
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, beta=False,
                                                       layer_set_bound=True, shortcut=True)
            # No need to set slope next time; we do not optimize the slopes.
            set_slope = False
            # print(f'layer {i} k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')

            mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
            mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
            # make the invalid lower bounds worse than normal lower bounds by minus 999999
            # we only consider the best lower bound across two splits by using min(0)
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

        i_idx = k_ret.max(0)  # compare across topK
        tmp_ret[i] = i_idx.values
        decision_tmp[i] = [k_decision[i_idx.indices[ii]][ii] for ii in range(batch*2)]

    if len(tmp_ret) > 0:
        max_ret = torch.max(torch.stack([i for i in tmp_ret.values()]), dim=0)  # compare across layers
        rets, decision_layers = max_ret.values.cpu().numpy(), max_ret.indices.cpu().numpy()  # first batch: score; second batch: intercept_tb.

        # add index number for the skipped layers
        # for _, g in groupby(enumerate(skip_layers), lambda ix: ix[0] - ix[1]):
        #     decision_layers[decision_layers >= list(g)[-1][-1]] += 1
        for s in skip_layers: decision_layers[decision_layers >= s] += 1

        for b in range(batch):
            if max([i[b].max() for i in score]) > 1e-4 and min([i[b].min() for i in intercept_tb]) < -1e-4 \
                    and max(rets[b], rets[b + batch]) > -10000:  # make sure this potential split is valid
                if rets[b] > rets[b+batch]:  # score > intercept_tb
                    final_decision.append(decision_tmp[decision_layers[b].item()][b])
                else:
                    final_decision.append(decision_tmp[decision_layers[b+batch].item()][b+batch])
            else:
                # print('\t using a random choice')
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        final_decision.append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        break
    else:
        # all layers are split or has no improvement
        for b in range(batch):
            # print('\t using a random choice')
            mask_item = [m[b] for m in mask]
            # for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):  # random
            for preferred_layer in reversed(range(len(pre_relu_indices))):  # from last layer to first layer
                if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                    final_decision.append(
                        [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                    break

    return final_decision

@torch.no_grad()
def choose_node_parallel_kFSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, branching_candidates=5, branching_reduceop='min', slopes=None,
                              betas=None, history=None, use_beta=False):

    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = branching_candidates

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer_i, layer in enumerate(reversed(net.net.relus)):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

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

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, abs(score_candidate).view(batch, -1) * mask[relu_idx])

        relu_idx -= 1

    final_decision = []

    # real batch = batch * 2, since we have two kinds of scores
    lbs = [torch.cat([i, i]) for i in lower_bounds]
    ups = [torch.cat([i, i]) for i in upper_bounds]
    if isinstance(slopes[0], dict):
        # per neuron slope.
        sps = slopes + slopes
    else:
        sps = [torch.cat([i, i]) for i in slopes]
    if use_beta:
        bs = [torch.cat([i, i]) for i in betas]
        history += history

    # Use beta_length to convert an index to its layer and offset.
    score_length = np.cumsum([len(score[i][0]) for i in range(len(score))])
    score_length = np.insert(score_length, 0, 0)

    # Flatten the gradients vector.
    all_score = torch.cat(score, dim=1)
    all_itb = torch.cat(intercept_tb, dim=1)

    score_idx = torch.topk(all_score, topk)
    score_idx_indices = score_idx.indices.cpu()
    itb_idx = torch.topk(all_itb, topk, largest=False)
    itb_idx_indices = itb_idx.indices.cpu()

    k_decision = []
    k_ret = torch.empty(size=(topk, batch * 2), device=lower_bounds[0].device, requires_grad=False)
    set_slope = True  # We only set the slope once.
    for k in range(topk):
        decision_index = score_idx_indices[:, k]
        # Find which layer and neuron this topk gradient belongs to.
        decision_max_ = []
        for l in decision_index:
            # Go over each element in this batch.
            l = l.item()
            layer = np.searchsorted(score_length, l, side='right') - 1
            idx = l - score_length[layer]
            decision_max_.append([layer, idx])

        decision_index = itb_idx_indices[:, k]
        # Find which layer and neuron this topk gradient belongs to.
        decision_min_ = []
        for l in decision_index:
            # Go over each element in this batch.
            l = l.item()
            layer = np.searchsorted(score_length, l, side='right') - 1
            idx = l - score_length[layer]
            decision_min_.append([layer, idx])

        k_decision.append(decision_max_ + decision_min_)

        # only save the best lower bounds of the two splits
        if use_beta:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, betas=bs,
                                                   layer_set_bound=True, shortcut=True, history=history)
        else:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, beta=False,
                                                   layer_set_bound=True, shortcut=True)
        # print(f'k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')
        # No need to set slope next time; we do not optimize the slopes.
        set_slope = False

        mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
        mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
        # make the invalid lower bounds worse than normal lower bounds by minus 999999
        # we only consider the best lower bound across two splits by using min(0)
        k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

    i_idx = k_ret.max(0)  # compare across topK
    rets = i_idx.values.cpu().numpy()
    rets_indices = i_idx.indices.cpu().numpy()
    decision_tmp = [k_decision[rets_indices[ii]][ii] for ii in range(batch*2)]

    for b in range(batch):
        if max(rets[b], rets[b + batch]) > -10000:  # make sure this potential split is valid
            if rets[b] > rets[b + batch]:  # score > intercept_tb
                final_decision.append(decision_tmp[b])
            else:
                final_decision.append(decision_tmp[b + batch])
        else:
            print('\t using a random choice')
            mask_item = [m[b] for m in mask]
            for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                    final_decision.append(
                        [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                    break

    return final_decision
