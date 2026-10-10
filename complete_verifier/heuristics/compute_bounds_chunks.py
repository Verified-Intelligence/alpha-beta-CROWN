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
import arguments
from state import AlphaValueData, BetaFullData
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def _compute_bounds_chunks_pass(net: 'LiRPANet', d):
    """Run one bound pass on a single batch; called by compute_lower_bounds_chunks.

    Reuses alphas from d and skips the full update_bounds pipeline.

    Returns:
        torch.Tensor: Lower bounds, shape (batch, output_dim).
    """
    solver_args = arguments.Config['solver']
    beta_args = solver_args['beta-crown']

    iteration = beta_args['iteration']
    enable_opt_interm_bounds = beta_args['enable_opt_interm_bounds']

    decision_thresh = d.get('thresholds', None)

    if net.net.cut_used:
        net.disable_cut_for_branching()

    batch_size = d['upper_bounds'][net.final_name].shape[0]
    ret = net._expand_tensors(d, batch_size)
    interm_bounds, _, _, c, new_x, _, _ = ret

    # Set alpha.
    AlphaValueData.from_domain_dict(d).attach_to_net(net)

    net.net.set_bound_opts({
        'optimize_bound_args': {
            'enable_beta_crown': False,
            'fix_interm_bounds': True,
            'iteration': iteration,
        },
        'enable_opt_interm_bounds': enable_opt_interm_bounds,
    })
    net.set_crown_bound_opts('beta')

    args_compute_bounds = dict(
        x=(new_x,),
        C=c,
        reuse_alpha=True,
        interm_bounds=interm_bounds,
        bound_upper=False,
        decision_thresh=decision_thresh
    )
    with torch.no_grad():
        lb, _ = net.net.compute_bounds(
            method='backward',
            **args_compute_bounds
        )

    return lb


def compute_lower_bounds_chunks(net: 'LiRPANet', d, sub_batch_size=None):
    """Compute lower bounds for the given domains using CROWN only. Perform in chunks if sub_batch_size is set.

    Args:
        net: LiRPANet instance.
        d: Domain dict.  Must contain 'lower_bounds', 'upper_bounds',
           'cs' and optionally 'alphas'.
        sub_batch_size: Maximum number of domains per chunk.  None (default)
                        processes the whole batch in one pass.

    Returns:
        torch.Tensor: Lower bounds, shape (batch, or_spec_size).
    """

    batch = d['upper_bounds'][net.final_name].shape[0]

    if sub_batch_size is None or batch <= sub_batch_size:
        # Fast path: the whole batch fits in one pass. Skip sub-batch dict
        # construction and alpha cloning that the loop below always incurs.
        return _compute_bounds_chunks_pass(net, d)

    num_chunks = (batch + sub_batch_size - 1) // sub_batch_size
    results = []
    print(f"Chunked bound pass: {num_chunks} chunks of at most {sub_batch_size} domains")
    for chunk_idx in range(num_chunks):
        i = chunk_idx * sub_batch_size
        j = min((chunk_idx + 1) * sub_batch_size, batch)

        sub_d = {}
        sub_d['lower_bounds'] = {k: v[i:j] for k, v in d['lower_bounds'].items()}
        sub_d['upper_bounds'] = {k: v[i:j] for k, v in d['upper_bounds'].items()}
        sub_d['cs'] = d['cs'][i:j]
        if 'alphas' in d:
            # alphas: {activation_layer_name -> {starting_node_name -> tensor}}
            # tensor shape: (2, 1, batch, num_neurons) — batch at dim 2.
            sub_d['alphas'] = {
                activation_layer: {
                    starting_node: alpha[:, :, i:j, ...].detach()
                    for starting_node, alpha in per_layer_alphas.items()
                }
                for activation_layer, per_layer_alphas in d['alphas'].items()
            }
        if 'betas' in d:
            raise NotImplementedError("beta sub-batching is not supported")

        results.append(
            _compute_bounds_chunks_pass(net, sub_d)
        )

    return torch.cat(results, dim=0)
