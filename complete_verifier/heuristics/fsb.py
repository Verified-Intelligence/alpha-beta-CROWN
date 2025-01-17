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
from collections import defaultdict
import torch
import numpy as np
from heuristics.babsr import BabsrBranching
from utils import get_reduce_op


class FsbBranching(BabsrBranching):

    @torch.no_grad()
    def get_branching_decisions(
            self, domains, split_depth, branching_candidates=5,
            branching_reduceop='min', use_beta=False, prioritize_alphas='none',
            **kwargs):

        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']
        history = domains['history']
        alphas, betas = domains['alphas'], domains['betas']
        rhs = domains['thresholds']

        batch = len(next(iter(orig_mask.values())))
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = get_reduce_op(branching_reduceop)
        # In case number of unstable neurons less than topk
        topk = min(branching_candidates,
                   int(sum([item.sum() for item in mask.values()]).item()))
        number_bounds = 1 if cs is None else cs.shape[1]
        score, intercept_tb = self.babsr_score(
            lower_bounds, upper_bounds, lAs, mask, reduce_op,
            number_bounds, prioritize_alphas)

        final_decision = [[] for _ in range(batch)]
        decision_tmp = {}
        tmp_ret = {}
        score_from_layer_idx = 1 if len(score) > 1 else 0
        skip_layers = list(range(score_from_layer_idx))

        # real batch = batch * 2, since we have two kinds of scores
        lbs = {k: torch.concat([v, v]) for k, v in lower_bounds.items()}
        ubs = {k: torch.concat([v, v]) for k, v in upper_bounds.items()}
        # per neuron alpha.
        sps = defaultdict(dict)
        for k, vv in alphas.items():
            sps[k] = {}
            for kk, v in vv.items():
                sps[k][kk] = torch.cat([v, v], dim=2)
        if use_beta:
            bs = [torch.cat([i, i]) for i in betas]
            history += history
        rhs = torch.cat([rhs, rhs])
        if cs is not None:
            cs = torch.cat([cs, cs])
        set_alpha = True  # We only set the alpha once.

        for i in range(score_from_layer_idx, len(score)):
            if ((score[i].max(1).values <= 1e-4).all()
                    and (intercept_tb[i].min(1).values >= -1e-4).all()):
                print(f'{i}th layer has no valid scores')
                skip_layers.append(i)
                continue
            score_idx = torch.topk(score[i], topk)
            score_idx_indices = score_idx.indices.cpu()
            itb_idx = torch.topk(intercept_tb[i], topk, largest=False)
            itb_idx_indices = itb_idx.indices.cpu()
            k_ret = torch.empty(size=(topk, batch * 2), device=score[i].device)
            k_decision = []
            for k in range(topk):
                decision_index = score_idx_indices[:, k]
                # add decision_index with layer's idx
                decision_max_ = [[i, j.item()] for j in decision_index]
                decision_index = itb_idx_indices[:, k]
                decision_min_ = [[i, j.item()] for j in decision_index]
                k_decision.append(decision_max_ + decision_min_)
                # only save the best lower bounds of the two splits
                args_update_bounds = {
                    'lower_bounds': lbs, 'upper_bounds': ubs,
                    'alphas': sps if set_alpha else {}, 'cs': cs
                }
                split = {'decision': k_decision[-1]}
                self.net.build_history_and_set_bounds(args_update_bounds, split)
                if use_beta:
                    args_update_bounds.update({'betas': bs, 'history': history})
                    k_ret_lbs = self.net.update_bounds(
                        args_update_bounds,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                else:
                    k_ret_lbs = self.net.update_bounds(
                        args_update_bounds, beta=False,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                # consider the max improvement among multi bounds in one C matrix
                k_ret_lbs = (k_ret_lbs - torch.cat([rhs, rhs])).max(-1).values
                # No need to set alpha next time; we do not optimize the alphas.
                set_alpha = False
                # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
                mask_score = (score_idx.values[:, k] <= 1e-4).float()
                mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
                # make the invalid lower bounds worse than normal lower bounds by minus 999999
                # we only consider the best lower bound across two splits by using min(0)
                k_ret[k] = reduce_op((
                    k_ret_lbs.view(-1) - torch.cat(
                        [mask_score, mask_itb]).repeat(2) * 999999
                    ).reshape(2, -1),
                    dim=0).values
            split_depth = min(split_depth, k_ret.shape[0])
            i_idx = k_ret.topk(split_depth, dim=0)  # compare across topK
            tmp_ret[i] = i_idx.values  # [split_depth, batch*2]
            tmp_indice = i_idx.indices
            decision_tmp[i] = [
                k_decision[tmp_indice[ii // (2 * batch)][
                    ii % (2 * batch)]][ii % (2 * batch)]
                for ii in range(split_depth * (batch * 2))
            ]

        # shape of tmp_ret: [layer, num_split, batch*2]
        if len(tmp_ret):
            stacked_layers = torch.stack([i for i in tmp_ret.values()])  # [layer, split_depth, batch*2]
            max_ret = torch.topk(stacked_layers.view(-1, batch * 2), split_depth,
                                 dim=0)  # compare across layers [split_depth, batch*2]
            # shape: [num_split*batch*2]
            rets, decision_layers = max_ret.values.view(-1).cpu().numpy(), max_ret.indices.view(
                -1).cpu().numpy()  # first batch: score; second batch: intercept_tb.
            decision_layers = decision_layers // split_depth

            # add index number for the skipped layers
            # for _, g in groupby(enumerate(skip_layers), lambda ix: ix[0] - ix[1]):
            #     decision_layers[decision_layers >= list(g)[-1][-1]] += 1
            for s in skip_layers:
                decision_layers[decision_layers >= s] += 1

            for l in range(split_depth):
                for b in range(batch):
                    decision_layer_1, decision_index_1 = decision_tmp[
                        decision_layers[2 * l * batch + b].item()][
                        l * 2 * batch + b]
                    decision_layer_2, decision_index_2 = decision_tmp[
                        decision_layers[2 * l * batch + b + batch].item()
                    ][l * 2 * batch + b + batch]
                    decision_layer_1 = self.net.split_nodes[decision_layer_1].name
                    decision_layer_2 = self.net.split_nodes[decision_layer_2].name
                    len_final_decision = len(final_decision[b])
                    if (max([s[b].max() for s in score]) > 1e-4
                            and min([s[b].min() for s in intercept_tb]) < -1e-4
                            and max(rets[2 * l * batch + b], rets[2 * l * batch + b + batch]) > -10000
                            and (mask[decision_layer_1][b][decision_index_1] != 0
                                 or mask[decision_layer_2][b][decision_index_2] != 0)
                    ):  # make sure this potential split is valid
                        if (rets[2 * l * batch + b] > rets[2 * l * batch + b + batch]
                                and mask[decision_layer_1][b][decision_index_1] != 0):  # score > intercept_tb
                            final_decision[b].append(
                                decision_tmp[decision_layers[2 * l * batch + b].item()][l * 2 * batch + b])
                        elif mask[decision_layer_2][b][decision_index_2] != 0:
                            final_decision[b].append(decision_tmp[decision_layers[2 * l * batch + b + batch].item()][
                                                         l * 2 * batch + b + batch])
                        else:
                            mask_item = {k: m[b] for k, m in mask.items()}
                            for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                                preferred_layer_ = self.net.split_nodes[preferred_layer].name
                                if len(mask_item[preferred_layer_].nonzero(as_tuple=False)) != 0:
                                    final_decision[b].append(
                                        [preferred_layer, mask_item[preferred_layer_].nonzero(as_tuple=False)[0].item()])
                                    break
                    else:
                        # using a random choice
                        mask_item = {k: m[b] for k, m in mask.items()}
                        for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if len(mask_item[preferred_layer_].nonzero(as_tuple=False)) != 0:
                                final_decision[b].append(
                                    [preferred_layer, mask_item[preferred_layer_].nonzero(as_tuple=False)[0].item()])
                                break
                    if len(final_decision[b]) > len_final_decision:
                        final_decision_ = self.net.split_nodes[final_decision[b][-1][0]].name
                        mask[final_decision_][b][final_decision[b][-1][1]] = 0
        else:
            # all layers are split or has no improvement
            for l in range(split_depth):
                for b in range(batch):
                    mask_item = {k: m[b] for k, m in mask.items()}
                    len_final_decision = len(final_decision[b])
                    for preferred_layer in range(len(self.net.split_nodes)-1, -1, -1): # from last layer to first layer
                        preferred_layer_ = self.net.split_nodes[preferred_layer].name

                        if len(mask_item[preferred_layer_].nonzero(as_tuple=False)) != 0:
                            final_decision[b].append(
                                [preferred_layer, mask_item[preferred_layer_].nonzero(as_tuple=False)[0].item()])
                            final_decision_ = self.net.split_nodes[final_decision[b][-1][0]].name
                            mask[final_decision_][b][final_decision[b][-1][1]] = 0
                            break

        split_depth = min([len(d) for d in final_decision])
        final_decision = [[batch[i] for batch in final_decision] for i in
                          range(split_depth)]  # change the order of final decision to split_depth * batch
        final_decision = sum(final_decision, [])

        return final_decision, None, split_depth # None for points
