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
import arguments

from prune import prune_alphas

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def copy_alpha(self: 'LiRPANet', reference_alphas, num_targets,
               target_batch_size=None, now_batch=None, interm_bounds=None,
               batch_size=None):
    # alpha manipulation, since after init_alpha all things are copied
    # from alpha-CROWN and these alphas may have wrong shape
    opt_interm_bounds = arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']
    final_name = self.net.final_node().name
    for m in self.net.optimizable_activations:
        keys = list(m.alpha.keys())
        # when fixed intermediate bounds are available, since intermediate betas
        # are not used anymore because we use fixed intermediate bounds later,
        # we can delete these intermediate betas to save space
        if interm_bounds is not None and not opt_interm_bounds:
            for k in keys:
                if k not in self.alpha_start_nodes:
                    del m.alpha[k]
        if final_name not in m.alpha:
            continue
        if (m.alpha[final_name].shape[1] != 1
                or m.alpha[final_name].shape[2] != batch_size):
            # shape mismatch detected
            # pick the first slice with shape [2, 1, 1, ...],
            # and repeat to [2, 1, batch_size, ...]
            repeat = [1 if i != 2 else batch_size
                    for i in range(m.alpha[final_name].dim())]
            m.alpha[final_name] = m.alpha[final_name][:, 0:1, 0:1].repeat(*repeat)

    if reference_alphas is None:
        return False

    # We already have alphas available
    all_alpha_initialized = True
    for m in self.net.optimizable_activations:
        for spec_name, alpha in m.alpha.items():
            # each alpha size is (2, spec, batch_size, *shape); batch size is 1.
            if not spec_name in reference_alphas[m.name]:
                continue
            reference_alpha = reference_alphas[m.name][spec_name]
            if spec_name == self.net.final_node().name:
                target_start = now_batch * target_batch_size
                target_end = min((now_batch + 1) * target_batch_size, num_targets)
                if alpha.size()[2] == target_end - target_start:
                    print(f'setting alpha for layer {m.name} '
                          f'start_node {spec_name} with alignment adjustment')
                    # The reference alpha has deleted the pred class itself,
                    # while our alpha keeps that
                    # now align these two
                    # note: this part actually implements the following
                    # TODO (extract alpha according to different label)
                    if reference_alpha.size()[1] > 1:
                        # didn't apply multiple x in incomplete_verifier
                        alpha.data = reference_alpha[:, target_start:target_end].reshape_as(alpha.data)
                    else:
                        # applied multiple x in incomplete_verifier
                        alpha.data = reference_alpha[:, :, target_start:target_end].reshape_as(alpha.data)
                else:
                    all_alpha_initialized = False
                    print(f'not setting layer {m.name} start_node {spec_name} '
                          'because shape mismatch '
                          f'({alpha.size()} != {reference_alpha.size()})')
            elif alpha.size() == reference_alpha.size():
                print(f"setting alpha for layer {m.name} start_node {spec_name}")
                alpha.data.copy_(reference_alpha)
            elif all([si == sj or ((d == 2) and sj == 1)
                      for d, (si, sj) in enumerate(
                          zip(alpha.size(), reference_alpha.size()))]):
                print(f'setting alpha for layer {m.name} start_node {spec_name} '
                      'with batch sample broadcasting')
                alpha.data.copy_(reference_alpha)
            else:
                # TODO extract alpha according to different label
                all_alpha_initialized = False
                print(f'not setting layer {m.name} start_node {spec_name} '
                      'because shape mismatch '
                      f'({alpha.size()} != {reference_alpha.size()})')

    return all_alpha_initialized


def get_alpha(self: 'LiRPANet', only_final=False, half=False, device=None):
    # alpha has size (2, spec, batch, *shape). When we save it,
    # we make batch dimension the first.
    # spec is some intermediate layer neurons, or output spec size.
    ret = {}
    for m in self.net.perturbed_optimizable_activations:
        ret[m.name] = {}
        for spec_name, alpha in m.alpha.items():
            if not only_final or spec_name in self.alpha_start_nodes:
                ret[m.name][spec_name] = self._transfer(alpha, device, half=half)
    return ret


def set_alpha(self: 'LiRPANet', d, set_all=False):
    alpha = d['alphas']
    if len(alpha) == 0:
        return

    for m in self.net.perturbed_optimizable_activations:
        for spec_name in list(m.alpha.keys()):
            if spec_name in alpha[m.name]:
                # Only setup the last layer alphas if no refinement is done.
                if spec_name in self.alpha_start_nodes or set_all:
                    m.alpha[spec_name] = alpha[m.name][spec_name]
                    # Duplicate for the second half of the batch.
                    m.alpha[spec_name] = m.alpha[spec_name].detach().requires_grad_()
            else:
                # This layer's alpha is not used.
                # For example, we can drop all intermediate layer alphas.
                del m.alpha[spec_name]


def add_batch_alpha(self, alphas, reference_alphas):
    batch_alphas = self.get_alpha()
    if arguments.Config['bab']['attack']['enabled']:
        # Save all alphas, which will be further refined in bab-attack.
        self.refined_alpha = reference_alphas
    # early alpha delete to save space
    # If we are not optimizing intermediate layer bounds, we do not need
    # to save all the intermediate alpha. We only keep the alpha for the
    # last layer.`
    new_batch_alphas = prune_alphas(batch_alphas, self.alpha_start_nodes)
    del batch_alphas
    batch_alphas = new_batch_alphas
    for k, v in batch_alphas.items():
        if k not in alphas:
            alphas[k] = {}
        for kk, vv in v.items():
            if kk not in alphas[k]:
                alphas[k][kk] = vv
            else:
                alphas[k][kk] = torch.cat([alphas[k][kk], vv], dim=2)
