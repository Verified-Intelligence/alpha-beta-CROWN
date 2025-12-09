#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import time
import torch
from tqdm import tqdm
import torch.nn as nn
import arguments
from load_model import Customized
import attack.attack_utils as attack_utils
from attack.attack_utils import (OSI_init_C, boundary_attack,
                                 PGDAttackResult, AdamClipping)

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


def default_pgd_loss(origin_out, output, C_mat, rhs_mat, or_spec_size,
                     gama_lambda=0, threshold=-1e-5, mode='hinge', model=None):
    '''
    output: [batch_size, num_restarts, num_or, num_output]
    C_mat: [batch_size, num_restarts, num_spec, num_output]
    rhs_mat: [batch_size, num_spec]
    or_spec_size: PLACEHOLDER
    gama_lambda: weight factor for gama loss. If true, sum the loss and return the sum of loss
    threshold: the threshold for hinge loss
    '''
    C_mat = C_mat.view(C_mat.shape[0], 1, output.shape[2], -1, C_mat.shape[-1])
    # [batch_size, 1, num_or, num_and_spec, num_output]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, output.shape[2], -1)
    loss = C_mat.matmul(output.unsqueeze(-1)).squeeze(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"]
    if arguments.Config["debug"]["sanity_check"] is None:
        loss = torch.clamp(loss, min=threshold)
    # [batch_size, num_restarts, num_or, num_and_spec]
    loss = -loss

    if origin_out is not None:
        loss_gamma = loss.sum() + (gama_lambda * (output - origin_out)**2).sum(dim=3).sum()
    else:
        loss_gamma = loss.sum()
    # [batch_size, num_restarts, num_or, num_and_spec]

    if mode == "sum":
        loss[loss >= 0] = 1.0
    # loss is returned for best loss selection, loss_gamma is for gradient descent.
    return loss, loss_gamma


def test_conditions(input, output, data_min, data_max, C_mat, rhs_mat, or_spec_size, return_best_idx=False):
    '''
    Whether the output satisfies the specifiction conditions.
    If the output satisfies the specification for adversarial examples, this function returns True, otherwise False.

    :param input: [batch_size, num_restarts, num_or/1, *input_shape]
    :param output: [batch_size, num_restarts, num_or/1, num_output]
    :param data_min: [batch_size, 1, num_or/1, *input_shape]
    :param data_max: [batch_size, 1, num_or/1, *input_shape]
    :param C_mat: [batch_size, num_spec, num_output]
    :param rhs_mat: [batch_size, num_spec]
    :param or_spec_size: [num_or]
    :param return_best_idx: whether to return the index of the OR spec with the smallest margin (largest violation if successful).
    '''
    # we assume input.shape[2] can be 1 or num_or.
    # when input.shape[2] == 1, it means the input contains the adv example of the best or spec
    # but here we check it with all or specs. Ideally, we can filter out C_mat, rhs_mat before this function.
    # when input.shape[2] == num_or, it means the input contains adv examples of all or specs.
    # we naturally check all or specs.
    num_or = or_spec_size.shape[0]
    assert input.shape[2] == num_or or input.shape[2] == 1

    C_mat = C_mat.view(C_mat.shape[0], 1, num_or, -1, C_mat.shape[-1])
    # [batch_size, restarts, num_or, num_and_spec, output_dim]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, num_or, -1)
    # [batch_size, restarts, num_or, num_and_spec]

    # apply a small tolerance to rhs so that we are more confident about the adv example
    cond_value = torch.matmul(C_mat, output.unsqueeze(-1)).squeeze(-1) - rhs_mat + arguments.Config["attack"]["attack_tolerance"]
    # [batch_size, restarts, num_or, num_and_spec]
    cond_value = cond_value.amax(dim=-1, keepdim=False)
    cond = cond_value < 0.0
    # [batch_size, restarts, num_or]

    valid = ((input <= data_max) & (input >= data_min))

    valid = valid.view(*valid.shape[:3], -1).all(-1)
    # [batch_size, restarts, num_or]

    res = (cond & valid).any(dim=-1).any(dim=-1)
    # [batch_size]

    if return_best_idx:
        # select the index of the OR spec with the largest violation
        vio_value = cond_value * valid
        # [batch_size, restarts, num_or]
        idx = torch.min(torch.min(vio_value, dim=1).values, dim=1).indices
        # [batch_size]
        return res, idx

    return res


def default_early_stop_condition(inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size,
        model, return_best_idx=False):

    return test_conditions(inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size,
        return_best_idx)


def default_adv_example_finalizer(
    model_ori: nn.Module,
    x: torch.Tensor,
    best_deltas: torch.Tensor,
    data_min: torch.Tensor,
    data_max: torch.Tensor,
    C_mat: torch.Tensor,
    rhs_mat: torch.Tensor,
    or_spec_size: torch.Tensor,
):
    # x, best_deltas, data_min and data_max have shape (batch, num_or/1, *input_shape).
    num_or = or_spec_size.shape[0]
    assert x.shape[1] == num_or or x.shape[1] == 1
    # x.shape[1] != num_or only for gtrsb benchmark.
    # refer to customized_gtrsb_adv_example_finalizer() for more details.

    adv_input = torch.clamp(x + best_deltas, min=data_min, max=data_max)
    # [batch_size, num_or, *input_shape]

    adv_output: torch.Tensor = model_ori(adv_input.view(-1, *x.shape[2:])).view(*adv_input.shape[:2], -1)
    # [batch_size, num_or, out_dim]

    C_mat = C_mat.view(C_mat.shape[0], num_or, -1, C_mat.shape[-1])
    # [batch_size, num_or, num_and_spec, output_dim]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], num_or, -1)
    # [batch_size, num_or, num_and_spec]

    adv_margin = torch.matmul(C_mat, adv_output.unsqueeze(-1)).squeeze(-1) - rhs_mat 
    # [batch_size, num_or, num_and_spec]
    adv_margin_per_or = adv_margin.max(dim=-1).values
    # [batch_size, num_or]
    adv_margin = adv_margin.view(adv_margin.shape[0], -1)
    # [batch_size, num_spec]

    print("PGD attack margin (first 2 examples and 10 specs):\n", adv_margin_per_or[:2, :10])
    print("number of violation: ", (adv_margin_per_or < 0).sum().item())
    # print the first 10 specifications for the first 2 examples

    return adv_input, adv_output, adv_margin, adv_margin_per_or


def pgd_attack_with_general_specs(model, X, data_min, data_max, C_mat, rhs_mat,
                                  or_spec_size, alpha,
                                  use_adam=True, normalize=lambda x: x,
                                  initialization='uniform', GAMA_loss=False,
                                  num_restarts=None, pgd_steps=None) -> PGDAttackResult:

    ''' the functional function for pgd attack

    :param model: PyTorch model under attack.
    :param x: Input (x_0). [batch_size, num_or, *input_shape]
    :param data_min: Lower bounds of data input. (e.g., 0 for mnist). [batch_size, num_or, *input_shape]
    :param data_max: Lower bounds of data input. (e.g., 1 for mnist). [batch_size, num_or, *input_shape]
    :param C_mat: [batch_size, num_spec, num_output]
    :param rhs_mat: [batch_size, num_spec]
    :param or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or.
    :param alpha: alpha for pgd attack
    :param use_adam: whether to use adam optimizer
    :param normalize: normalization function
    :param initialization: initialization method
    :param GAMA_loss: whether to use GAMA loss
    :param num_restarts: number of restarts
    :param pgd_steps: number of pgd steps
    '''

    device = X.device
    attack_iters = arguments.Config["attack"]["pgd_steps"] if pgd_steps is None else pgd_steps
    num_restarts = arguments.Config["attack"]["pgd_restarts"] if num_restarts is None else num_restarts

    lr_decay=arguments.Config["attack"]["pgd_lr_decay"]
    early_stop=arguments.Config["attack"]["pgd_early_stop"]
    restart_when_stuck = arguments.Config["attack"]["pgd_restart_when_stuck"]

    if restart_when_stuck:
        total_replaced_deltas = 0

    # [batch, num_or, *input_shape]
    original_X = X.clone().detach()
    X_shape = X.shape
    num_output = C_mat.shape[-1]

    extra_dim = (num_restarts,)

    batch_indices = torch.arange(X_shape[0], device=device)
    num_or = or_spec_size.shape[0]
    same_or_spec_size = (or_spec_size == or_spec_size[0]).all()
    assert same_or_spec_size, "The number of AND statements in each OR clause should be the same."
    num_and = or_spec_size[0].item()

    # we will return the best delta and the best loss per batch per OR spec.
    # best_delta: [batch, num_or, *input_shape]. best_loss: [batch, num_or]
    best_delta = torch.zeros(X_shape, device=device)
    best_loss = torch.empty(X_shape[:2], device=device).fill_(float("-inf"))

    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)
    # [batch, 1, num_or, *input_shape]

    X_ndim = X.ndim

    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X

    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim - 1))
    extra_dim = (X.shape[1], X.shape[2])

    if initialization == 'osi':
        osi_start_time = time.time()
        X_init = OSI_init_C(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
        osi_time = time.time() - osi_start_time
        print(f'diversed PGD initialization time: {osi_time:.4f}')
    if initialization == 'boundary':
        boundary_adv_inputs = boundary_attack(model, X[:,0,...].reshape(-1, *X_shape[1:]), data_min.view(*X_shape), data_max.view(*X_shape))
        if boundary_adv_inputs is not None:
            X_init = boundary_adv_inputs.view(X.shape[0], -1, *X.shape[2:])
            X = X[:,:X_init.shape[1],...]
            extra_dim = (X.shape[1], X.shape[2])
        else:
            initialization = 'uniform'

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    if initialization == 'osi' or initialization == 'boundary':
        delta = (X_init - X).detach().requires_grad_()
    elif initialization == 'uniform':
        delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
    elif initialization == 'none':
        delta = torch.zeros_like(X).requires_grad_()
    else:
        raise ValueError(f"Unknown initialization method {initialization}")

    if use_adam:
        opt = AdamClipping(params=[delta], lr=alpha)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
    else:
        # alpha: [num_or, *input_shape] -> [1, 1, num_or, *input_shape]
        # will boardcast to [batch, num_restarts, num_or, *input_shape] when used.
        alpha = alpha.unsqueeze(0).unsqueeze(0)

    for iteration in tqdm(range(attack_iters)):
        attack_utils.stats.accumulate(num_restarts, 1)
        # additional clamping to make sure inputs in the range
        # otherwise inputs may out of range due to numerical error when updating delta
        inputs = normalize(X + delta).clamp(min=data_min, max=data_max)
        # [batch_size, num_restarts, num_or, *input_shape]
        output = model(inputs.view(-1, *X_shape[2:])).view(
            X_shape[0], *extra_dim, num_output)

        if GAMA_loss:
            # Output on original model is needed if gama loss is used.
            origin_out = torch.softmax(model(normalize(X.reshape(-1, *X_shape[2:]))), 1)
            origin_out = origin_out.view(output.shape)
        else:
            origin_out = None

        loss, loss_gama = eval(arguments.Config["attack"]["pgd_loss"])(
            origin_out, output, C_mat, rhs_mat,
            None, gama_lambda if GAMA_loss else 0.0,
            mode=arguments.Config['attack']['pgd_loss_mode'], model=model)
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        # shape of loss: [batch_size, num_restarts, num_or]
        # or float when gama_lambda > 0

        loss_gama.sum().backward()

        with torch.no_grad():
            # Save the best loss so far.
            loss = loss.amin(-1)
            # loss has shape [batch_size, num_restarts, num_or].
            # best_loss_ is the best loss per batch per OR spec in the current iteration.
            # best_indices_ is corresponding restart indices of the best loss.
            # [batch, num_or] (reduced the num_restarts dimension by max)
            best_loss_, best_indices_ = loss.max(1)
            # best_delta_ is the best delta per batch per OR spec in the current iteration.
            # [batch, num_or, *input_shape]
            best_delta_ = delta.gather(
                dim=1,
                index=best_indices_.view(*best_indices_.shape, *([1] * (delta.ndim - 3)))
                .unsqueeze(1)
                .expand(-1, -1, -1, *delta.shape[3:]),
            ).squeeze(1)

            best_delta[best_loss_ >= best_loss] = best_delta_[best_loss_ >= best_loss]
            best_loss = torch.max(best_loss, best_loss_)

            if early_stop:
                # we select the best input and output for the best one OR spec for each batch
                # to check the early stop condition for efficiency.
                # indices: [batch]
                best_or_indices = best_loss_.max(dim=1).indices
                best_restart_indices_best_or = best_indices_.gather(dim=1, index=best_or_indices.unsqueeze(1)).squeeze(1)

                # test_inputs: [batch, 1, 1, *input_shape], test_outputs: [batch, 1, 1, output_dim]
                test_inputs = inputs[batch_indices, best_restart_indices_best_or, best_or_indices].unsqueeze(1).unsqueeze(1)
                test_data_max = data_max[batch_indices, :, best_or_indices].unsqueeze(2)
                test_data_min = data_min[batch_indices, :, best_or_indices].unsqueeze(2)
                test_outputs = output[batch_indices, best_restart_indices_best_or, best_or_indices].unsqueeze(1).unsqueeze(1)

                spec_mask = torch.zeros((X_shape[0], num_or), dtype=torch.bool, device=device)
                spec_mask[batch_indices, best_or_indices] = True
                # [batch_size, num_or]
                spec_mask = spec_mask.repeat_interleave(num_and, dim=1)
                # [batch_size, num_spec]
                test_C_mat = C_mat[spec_mask].view(X_shape[0], num_and, -1)
                # [batch_size, num_and, output_dim]
                test_rhs_mat = rhs_mat[spec_mask].view(X_shape[0], num_and)
                # [batch_size, num_and]
                # we assume the number of AND statements in each OR clause is the same.
                test_or_spec_size = or_spec_size[0:1]
                # [1]

                if eval(arguments.Config["attack"]["early_stop_condition"])(
                        test_inputs, test_outputs, test_data_min, test_data_max,
                        test_C_mat, test_rhs_mat,
                        test_or_spec_size, model).all():
                    print("pgd early stop")
                    break

        if restart_when_stuck:
            old_delta = delta.clone().detach()
        if use_adam:
            opt.step(clipping=True, lower_limit=delta_lower_limit,
                     upper_limit=delta_upper_limit, sign=1)
            opt.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            d = delta + alpha * torch.sign(delta.grad)
            d = torch.max(torch.min(d, delta_upper_limit), delta_lower_limit)
            delta = d.detach().requires_grad_()

        if restart_when_stuck:
            unchanged = ((delta - old_delta).abs().sum(list(range(2, delta.ndim)), keepdim=True) == 0).to(delta.dtype)
            total_replaced_deltas += int(unchanged.sum().item())
            new_init = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit)
            delta.data.copy_(delta * (1 - unchanged) + new_init * unchanged)

    if restart_when_stuck:
        total_num_deltas = X.size(0) * delta.size(1) * (iteration + 1)
        replaced_percentage = total_replaced_deltas / total_num_deltas * 100
        print(f'Attack batch size: {X.size(0)}, restarts: {delta.size(1)}, iterations: {iteration + 1} '
              f'replaced deltas {total_replaced_deltas} ({replaced_percentage}%)')

    adv_input_per_or, adv_output_per_or, adv_margin_per_spec, adv_margin_per_or = eval(
        arguments.Config["attack"]["adv_example_finalizer"]
    )(model, original_X, best_delta, data_min.squeeze(1), data_max.squeeze(1), C_mat, rhs_mat, or_spec_size)
    # [batch_size, num_or, *input_shape], [batch_size, num_or, num_output],
    # [batch_size, num_spec], [batch_size, num_or]

    attack_success, best_or_idx = test_conditions(
        adv_input_per_or.unsqueeze(1), adv_output_per_or.unsqueeze(1),
        data_min, data_max, C_mat, rhs_mat, or_spec_size, True)
    # [batch_size], [batch_size]

    adv_input_best = adv_input_per_or[batch_indices, best_or_idx]
    # [batch_size, *input_shape]
    adv_output_best = adv_output_per_or[batch_indices, best_or_idx]
    # [batch_size, num_output]
    adv_margin_best = adv_margin_per_or[batch_indices, best_or_idx]
    # [batch_size]

    # a sanity check. found a successful adv example <==> corrsponding margin <= 0
    assert (
        (attack_success == (adv_margin_best <= 0.0)).all()
    ), "result of test_conditions() and adv_example_finalizer() inconsistent."

    adv_input_all = None
    if arguments.Config['bab']['attack']['enabled']:
        adv_input_all = (delta + original_X.unsqueeze(1)).clamp(
            min=data_min.unsqueeze(1), max=data_max.unsqueeze(1))
        # [batch_size, num_restarts, num_or, *input_shape]

    return PGDAttackResult(
        attack_success=attack_success,
        best_or_idx=best_or_idx,
        adv_input_per_or=adv_input_per_or,
        adv_output_per_or=adv_output_per_or,
        adv_margin_per_or=adv_margin_per_or,
        adv_input_best=adv_input_best,
        adv_output_best=adv_output_best,
        adv_margin_best=adv_margin_best,
        adv_margin_per_spec=adv_margin_per_spec,
        adv_input_all=adv_input_all
    )
