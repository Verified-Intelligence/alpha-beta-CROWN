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
from utils import get_reduce_op, get_batch_size_from_masks


class KfsbBranching(BabsrBranching):
    """
    Branching using the intercept term in ReLU relaxation.
    TODO: support general activation functions.
    """

    def babsr_score_intercept_only(self, lbs, ubs, lAs, batch):
        """Compute branching scores for kfsb based on intercept only."""
        score = []
        for k in lbs:
            if k == self.net.final_name:
                continue
            assert len(self.net.split_activations[k]) == 1
            A_key = self.net.split_activations[k][0][0].name
            ratio = ((-lbs[k]).clamp(0, None) * ubs[k].clamp(0, None)) / (ubs[k] - lbs[k])
            ratio *= (-lAs[A_key].mean(dim=1)).clamp(0, None)
            score.append(ratio.reshape(batch, -1))
        return score

    @torch.no_grad()
    def get_branching_decisions(self, domains, split_depth, branching_candidates=5,
                                branching_reduceop='min', use_beta=False, keep_all_decision=False,
                                prioritize_alphas='none',  method='kfsb', **kwargs):

        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']
        history = domains['history']
        alphas, betas = domains['alphas'], domains['betas']
        rhs = domains['thresholds']

        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        batch = get_batch_size_from_masks(mask)
        reduce_op = get_reduce_op(branching_reduceop, with_dim=False)
        topk = min(branching_candidates,
                   int(sum([i.sum() for i in mask.values()]).item()))
        # FIXME: it seems cs and should always be not None because they are used below.
        number_bounds = 1 if cs is None else cs.shape[1]

        if method == 'kfsb-intercept-only':
            score = self.babsr_score_intercept_only(lower_bounds, upper_bounds, lAs, batch)
        elif method == 'kfsb':
            score, intercept_tb = self.babsr_score(
                lower_bounds, upper_bounds, lAs, mask, reduce_op,
                number_bounds, prioritize_alphas)
        else:
            raise ValueError(f'Unsupported branching method "{method}" for relu splits.')

        final_decision = []
        if method == 'kfsb-intercept-only':
            # We only have one kind of score, so batch does not change.
            lbs, ubs, sps, css = lower_bounds, upper_bounds, alphas, cs
            if use_beta:
                bs = betas
        elif method == 'kfsb':
            # real batch = batch * 2, since we have two kinds of scores.
            lbs = {k: torch.cat([v, v]) for k, v in lower_bounds.items()}
            ubs = {k: torch.cat([v, v]) for k, v in upper_bounds.items()}
            # per neuron alpha.
            sps = defaultdict(dict)
            for k, vv in alphas.items():
                sps[k] = {}
                for kk, v in vv.items():
                    sps[k][kk] = torch.cat([v, v], dim=2)
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
        k_ret = torch.empty(size=(topk, batch_size),
                            device=all_score.device, requires_grad=False)
        set_alpha = True  # We only set the alpha once.

        reduce_op = get_reduce_op(branching_reduceop, with_dim=True)
        for k in range(topk):
            # top-k candidates from the alpha scores.
            decision_index = score_idx_indices[:, k]
            # Find which layer and neuron this topk gradient belongs to.
            decision_max_ = []
            for l in decision_index:
                # Go over each element in this batch.
                l = l.item()
                # Recover the (layer, idx) from the flattened array.
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
            args_update_bounds = {
                'lower_bounds': lbs, 'upper_bounds': ubs,
                'alphas': sps if set_alpha else {}, 'cs': css
            }
            split = {'decision': k_decision[-1]}
            self.net.build_history_and_set_bounds(args_update_bounds, split)

            if use_beta:
                args_update_bounds.update({'betas': bs, 'history': history})
                k_ret_lbs = self.net.update_bounds(
                    args_update_bounds,
                    fix_interm_bounds=True, shortcut=True,
                    beta_bias=False)
            else:
                k_ret_lbs = self.net.update_bounds(
                    args_update_bounds, beta=False,
                    fix_interm_bounds=True, shortcut=True,
                    beta_bias=False)

            # consider the max improvement among multi bounds in one C matrix
            k_ret_lbs = (k_ret_lbs - torch.cat([rhs, rhs])).max(-1).values
            # No need to set alpha next time; we do not optimize the alphas.
            set_alpha = False

            mask_score = (score_idx.values[:,
                          k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
            if method == 'kfsb-intercept-only':
                reduced_score = reduce_op((k_ret_lbs.view(-1) - mask_score.repeat(2) * 999999).reshape(2, -1), dim=0)
            elif method == 'kfsb':
                mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
                # We first make the invalid lower bounds worse than normal lower bounds by minus 999999.
                # Then we consider the best lower bound across two splits (in the first dimension after reshape) by using min(0) or max(0).
                reduced_score = reduce_op(
                    (k_ret_lbs.view(-1)
                     - torch.cat([mask_score, mask_itb]).repeat(2) * 999999
                    ).reshape(2, -1), dim=0)
            else:
                raise NotImplementedError(method)
            if isinstance(reduced_score, torch.Tensor):
                k_ret[k] = reduced_score
            else:
                k_ret[k] = reduced_score.values

        if method == 'kfsb':
            for v in sps.values():
                for kk, vv in v.items():
                    v[kk] = torch.split(vv, vv.shape[2] // 2, dim=2)[0]

        split_depth = min(split_depth, k_ret.shape[0])
        if not keep_all_decision:
            # k_ret has shape (top-k, batch*2) and we take the score evaluated using bound propagation based on the top-k choice.
            i_idx = k_ret.topk(split_depth, 0)
            rets = i_idx.values.cpu().numpy()
            rets_indices = i_idx.indices.cpu().numpy()

            # Given the indices of the max score, find what is its corresponding decision.
            decision_tmp = [[k_decision[rets_indices[l][ii]][ii] for ii in range(batch_size)]
                            for l in range(split_depth)]

            # regular kfsb, select the top {split_depth} decision from k
            final_decision = [[] for _ in range(batch)]
            random_decision_dict = {}

            # customize kfsb but sometimes duplicate split for k_decision[ki][b] and k_decision[ki][b+batch]
            for l in range(split_depth):
                for b in range(batch):
                    # use mask to check if a node is unstable when random selection
                    # not detach for now, please check!!!
                    thres = max(rets[l][b], rets[l][b + batch]) if method == 'kfsb' else rets[l][b]
                    if thres > -10000:
                        if method == 'kfsb-intercept-only':
                            decision = decision_tmp[l][b]
                        elif method == 'kfsb':
                            if rets[l][b] > rets[l][b + batch]:
                                decision = decision_tmp[l][b]
                            else:
                                decision = decision_tmp[l][b + batch]
                        decision_layer = self.net.split_nodes[decision[0]].name
                        if mask[decision_layer][b][decision[1]] != 0:
                            final_decision[b].append(decision)
                            mask[decision_layer][b][decision[1]] = 0

            for b in range(batch):
                if len(final_decision[b]) < split_depth:
                    # No valid scores, have to choose neurons randomly.
                    random_decision_dict[b] = split_depth - len(final_decision[b])
                    for i in range(split_depth - len(final_decision[b])):
                        for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if len(mask[preferred_layer_][b].nonzero(as_tuple=False)) != 0:
                                final_decision[b].append(
                                    [preferred_layer, mask[preferred_layer_][b].nonzero(as_tuple=False)[0].item()])
                                mask[preferred_layer_][b][final_decision[b][-1][1]] = 0
                                break
                assert len(final_decision[b]) <= split_depth, f'{len(final_decision[b])} <= {split_depth}'

            split_depth = min([len(d) for d in final_decision])

            final_decision = [[batch[i] for batch in final_decision] for i in
                              range(split_depth)]  # change the order of final decision to split_depth * batch
            final_decision = sum(final_decision, [])

            return final_decision, None, split_depth  # None for points
        else:
            # keep all the k decisions
            # final_decision: batch -> k splits
            final_decision = [[] for _ in range(batch)]
            random_decision_dict = {}
            # customize kfsb but sometimes duplicate split for k_decision[ki][b] and k_decision[ki][b+batch]
            for b in range(batch):
                # use mask to check if a node is unstable when random selection
                # not detach for now, please check!!!
                for ki in range(topk):
                    thres = max(k_ret[ki][b], k_ret[ki][b + batch]) if method == 'kfsb' else k_ret[ki][b]
                    if thres > -10000:
                        if method == 'kfsb-intercept-only':
                            decision = k_decision[ki][b]
                        elif method == 'kfsb':
                            decision_layer = self.net.split_nodes[k_decision[ki][b][0]].name
                            if (k_ret[ki][b] > k_ret[ki][b + batch]
                                    and mask[decision_layer][b][k_decision[ki][b][1]] == 1):
                                decision = k_decision[ki][b]
                            else:
                                decision = k_decision[ki][b + batch]
                        decision_layer = self.net.split_nodes[decision[0]].name
                        if mask[decision_layer][b][decision[1]] == 1:
                            final_decision[b].append(decision)
                            mask[decision_layer][b][decision[1]] = 0
                if len(final_decision[b]) < topk:
                    # No valid scores, have to choose a neuron randomly.
                    random_decision_dict[b] = topk - len(final_decision[b])
                    for _ in range(topk - len(final_decision[b])):
                        for preferred_layer in np.random.choice(len(self.net.split_nodes), len(self.net.split_nodes), replace=False):
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if len(mask[preferred_layer_][b].nonzero(as_tuple=False)) != 0:
                                final_decision[b].append(
                                    [preferred_layer, mask[preferred_layer_][b].nonzero(as_tuple=False)[0].item()])
                                mask[preferred_layer_][b][final_decision[b][-1][1]] = 0
                                break
                assert len(final_decision[b]) <= topk, f'{len(final_decision[b])} <= {topk}'
            if random_decision_dict:
                print(f'Random branching decision used for {{example_idx:n_random}}: {random_decision_dict}')

            return final_decision, None, split_depth  # None for points
