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
"""Pruning: After CROWN pass, prune verified labels before starting the alpha-CROWN pass."""

import time
import torch


class PruneAfterCROWN:
    def __init__(self, net, c, lb, decision_thresh):
        self.net = net
        if isinstance(decision_thresh, torch.Tensor):
            # Batch size must be 1.
            assert decision_thresh.size(0) == 1

        stime = time.time()
        self.final_layer_lb = lb[-1]
        self.unverified_label_mask = (
            self.final_layer_lb <= decision_thresh[0]).nonzero().view(-1)
        self.c = c[:, self.unverified_label_mask]
        # fix the alpha shape
        for relu in net.splittable_activations:
            if relu.alpha is not None and net.final_name in relu.alpha:
                relu.alpha[net.final_name] = relu.alpha[
                    net.final_name][:, self.unverified_label_mask].detach()
        print('prune_after_crown optimization in use: original label size =',
              self.final_layer_lb.shape[0],
              'pruned label size =', len(self.unverified_label_mask))
        self.overhead = time.time() - stime
        if isinstance(decision_thresh, torch.Tensor):
            assert decision_thresh.ndim == 2
            self.decision_thresh = decision_thresh[:, self.unverified_label_mask]
        else:
            self.decision_thresh = decision_thresh

    def recover_lb(self, lb):
        """Recover full shape lb."""
        if lb is not None:
            stime = time.time()
            new_final_layer_lb = torch.full_like(self.final_layer_lb, float("inf"))
            new_final_layer_lb = new_final_layer_lb.unsqueeze(0)
            new_final_layer_lb[:, self.unverified_label_mask] = lb
            lb = new_final_layer_lb
            self.overhead += time.time() - stime
        return lb

    def recover_lA_and_alpha(self, lA, alphas):
        stime = time.time()
        with torch.no_grad():
            # handle lA
            newlA = {}
            for k, Aitem in lA.items():
                newAshape = list(Aitem.shape)
                newAshape[1] = self.final_layer_lb.shape[0]
                newA = torch.zeros(
                    newAshape, device=Aitem.device, dtype=Aitem.dtype)
                newA[:, self.unverified_label_mask] = Aitem
                newlA[k] = newA
            # handle alphas
            for v in alphas.values():
                if self.net.final_name in v:
                    oldalpha = v[self.net.final_name]
                    alphashape = list(oldalpha.shape)
                    alphashape[1] = self.final_layer_lb.shape[0]
                    newalpha = torch.zeros(
                        alphashape, device=oldalpha.device, dtype=oldalpha.dtype)
                    newalpha[:, self.unverified_label_mask] = oldalpha
                    v[self.net.final_name] = newalpha
            # since we may reread the alpha from the network, we push the full shape alpha back to net
            for m in self.net.splittable_activations:
                if m.name in alphas:
                    m.alpha = alphas[m.name]
        self.overhead += time.time() - stime
        return newlA

    def print_overhead(self):
        print('  Prune after CROWN overhead: '
              f'{self.prune_after_crown_overhead}s')


@staticmethod
def prune_reference_alphas(reference_alphas, keep_condition, final_node_name):
    for spec_dict in reference_alphas.values():
        for spec in spec_dict:
            if spec == final_node_name:
                if spec_dict[spec].size()[1] > 1:
                    # correspond to multi-x case
                    spec_dict[spec] = spec_dict[spec][:, keep_condition]
                else:
                    spec_dict[spec] = spec_dict[spec][:, :, keep_condition]


@staticmethod
def prune_lA(lA, keep_condition):
    return {k: v[keep_condition] for k, v in lA.items()}


def prune_alphas(alpha, kept_names):
    if alpha is None:
        return None
    print(f'Keeping alphas for these layers: {kept_names}')
    new_alpha = {}
    for node, alphas in alpha.items():
        new_alpha[node] = {}
        for spec_name, v in alphas.items():
            if spec_name in kept_names:
                new_alpha[node][spec_name] = v
    return new_alpha
