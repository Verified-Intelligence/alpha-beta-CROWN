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
import arguments
from load_model import Customized
import attack.attack_utils as attack_utils
from attack.attack_utils import (OSI_init_C, boundary_attack, precompute_group_indices,
                                 PGDAttackResult, AdamClipping)
from utils import pad_list_of_input_to_tensor

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


def default_pgd_loss(origin_out, output, C_mat, rhs_mat, or_spec_size,
                     gama_lambda=0, threshold=-1e-5, mode='hinge', model=None):
    '''
    origin_out: output on original model if gama loss is used. [batch, num_or, num_restarts, num_output]
    output: [batch, num_or, num_restarts, num_output]
    C_mat: [batch, num_spec, num_output]
    rhs_mat: [batch, num_spec]
    or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or
    gama_lambda: weight factor for gama loss. If true, sum the loss and return the sum of loss
    threshold: the threshold for hinge loss
    '''
    # repeat output based on cond
    output = output.repeat_interleave(or_spec_size, dim=1)
    # [batch, num_spec, num_restarts, num_output]

    if origin_out is not None:
        origin_out = origin_out

    lhs_mat = torch.matmul(output, C_mat.unsqueeze(-1)).squeeze(-1)
    # [batch, num_spec, num_restarts, num_output] @ [batch, num_spec, num_output, 1]
    # -> [batch, num_spec, num_restarts, 1] -> [batch, num_spec, num_restarts]

    loss = lhs_mat - rhs_mat.unsqueeze(2) + arguments.Config["attack"]["attack_tolerance"]
    # [batch, num_spec, num_restarts]

    # apply threshold and negative sign to get the final loss value
    loss = torch.clamp(loss, min=threshold)
    loss = -loss
    # [batch, num_spec, num_restarts]

    if origin_out is not None:
        total_loss = loss.sum() + (gama_lambda * (output - origin_out)**2).sum()
    else:
        # sum across num_spec and num_restarts
        total_loss = loss.sum()

    if mode == "sum":
        loss[loss >= 0] = 1.0

    # loss is returned for best loss selection, total_loss is for gradient descent.
    return loss, total_loss


def test_conditions(input, output, data_min, data_max, C_mat, rhs_mat, or_spec_size, return_best_idx=False):
    '''
    Whether the output satisfies the specifiction conditions.
    If the output satisfies the specification for adversarial examples, this function returns True, otherwise False.

    input: [batch, num_or, num_restarts, *input_shape]
    output: [batch, num_or, num_restarts, num_output]
    C_mat: [batch, num_spec, num_output]
    rhs_mat: [batch, num_spec]
    or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or
    data_max & data_min: [batch, num_or, 1, *input_shape]
    '''
    input_within_range = ((input <= data_max) & (input >= data_min)).all(dim=tuple(range(3, input.ndim)))
    # [batch, num_or, num_restarts]
    
    # repeat output based on cond
    output = output.repeat_interleave(or_spec_size, dim=1)
    # [batch, num_spec, num_restarts, num_output]

    lhs_mat = torch.matmul(output, C_mat.unsqueeze(-1)).squeeze(-1)
    # [batch, num_spec, num_restarts, num_output] @ [batch, num_spec, num_output, 1]
    # -> [batch, num_spec, num_restarts, 1] -> [batch, num_spec, num_restarts]

    cond = lhs_mat - rhs_mat.unsqueeze(2) + arguments.Config["attack"]["attack_tolerance"]
    # [batch, num_spec, num_restarts]

    # get the indices of the belonging OR clauses for AND clauses.
    group_indices, num_or = precompute_group_indices(or_spec_size)
    group_indices = group_indices.view(1, -1, 1).expand(cond.shape)

    # Compute the worst margin (max) for each OR clause among ANDs
    worst_margin = torch.scatter_reduce(cond[:, :num_or, :], 1, group_indices, cond, 'amax', include_self=False)
    # [batch, num_or, num_restarts]

    # For cases inputs out of range, we set the margin to +inf.
    worst_margin[~input_within_range] = float('inf')

    # Find the best margin (min) for each OR clause among restarts
    best_margin_per_or = torch.min(worst_margin, dim=2)[0]
    # [batch, num_or]

    # get the best margin and its OR index for each batch
    best_margin, best_margin_indices = torch.min(best_margin_per_or, dim=1)
    # [batch]

    # check if the best OR clause is satisfied for each batch
    final_result = best_margin <= 0

    return (final_result, best_margin_indices) if return_best_idx else final_result


def default_early_stop_condition(inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size,
        model, return_best_idx=False):

    return test_conditions(inputs, output, data_min, data_max, C_mat, rhs_mat, or_spec_size,
        return_best_idx)


@torch.no_grad()
def default_adv_example_finalizer(model_ori, x: torch.Tensor, best_deltas: torch.Tensor, 
                                  data_min: torch.Tensor, data_max: torch.Tensor, 
                                  C_mat: torch.Tensor, rhs_mat: torch.Tensor, or_spec_size: torch.Tensor):
    '''
    x and best_deltas has shape [batch, num_or, *input_shape]
    data_min and data_max have shape [batch, num_or, *input_shape]
    C_mat: [batch, num_spec, num_output]
    rhs_mat: [batch, num_spec]
    or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or
    '''
    # x, best_deltas, data_min and data_max have shape (batch, num_or/1, *input_shape).
    num_or = or_spec_size.shape[0]
    assert x.shape[1] == num_or or x.shape[1] == 1
    # x.shape[1] != num_or only for gtrsb benchmark.
    # refer to customized_gtrsb_adv_example_finalizer() for more details.

    input_shape = x.size()
    adv_input = (x + best_deltas)
    adv_input.clamp_(data_min, data_max)
    # [batch, num_or, *input_shape]

    adv_output = model_ori(adv_input.view(-1, *input_shape[2:])).view(
            *input_shape[:2], C_mat.shape[-1])
    # [batch, num_or, num_output]

    adv_output_repeat = adv_output.repeat_interleave(or_spec_size, dim=1)
    # [batch, num_spec, num_output]
    
    adv_margin = (C_mat * adv_output_repeat).sum(-1) - rhs_mat
    # [batch, num_spec]

    group_indices, num_or = precompute_group_indices(or_spec_size)
    group_indices = group_indices.view(1, -1).expand_as(adv_margin)
    adv_margin_per_or = torch.scatter_reduce(adv_margin[:, :num_or], 1, group_indices, adv_margin, 'amax', include_self=False)
    # [batch, num_or]

    print("PGD attack margin (first 2 examples and 10 specs):\n", adv_margin_per_or[:2, :10])
    print("Total number of violation: ", (adv_margin_per_or < 0).sum().item())

    return adv_input, adv_output, adv_margin, adv_margin_per_or


@torch.no_grad()
def find_optimal_loss(loss: torch.Tensor, group_indices: torch.Tensor, num_or: int):
    '''
    For each batch element, this function selects from the best over
    restarts dimension.

    Args:
    loss: [batch, num_spec, num_restarts]
    group_indices: [num_spec], the indices of the belonging OR clauses for AND clauses.
    num_or: number of OR clauses.

    Return:
    curr_best_losses:[batch, num_or] 
    restart_indices: [batch, num_or]
    '''
    # Compute the worst loss (minimum) for each OR clause among ANDs
    group_indices = group_indices.view(1, -1, 1).expand(loss.shape)
    worst_losses = torch.scatter_reduce(loss[:, :num_or, :], 1, group_indices, loss, 'amin', include_self=False)

    # Find the best loss (maximum) for each OR clause among restarts
    best_losses_per_or, best_restart_indices_per_or = torch.max(worst_losses, dim=2)

    return best_losses_per_or, best_restart_indices_per_or


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

    # make sure X, data_min, data_max have the same shape to avoid unexpected broadcasting and repeating.
    assert X.shape == data_min.shape == data_max.shape, "X, data_min, data_max should have the same shape."

    original_X = X.clone().detach()
    X_shape = X.size()
    # [batch, num_or, *input_shape]
    num_output = C_mat.shape[-1]

    batch_size = X_shape[0]
    batch_indices = torch.arange(batch_size, device=device)

    best_losses = torch.full(X_shape[:2], float('-inf'), device=device)
    # [batch, num_or]
    best_delta = torch.zeros(X_shape, device=device)
    # [batch, num_or, *input_shape]

    # unsqueeze for num_restart dim
    data_min = data_min.unsqueeze(2)
    data_max = data_max.unsqueeze(2)
    X = X.unsqueeze(2)
    # [batch, num_or, 1, *input_shape]

    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X
    # [batch, num_or, 1, *input_shape]

    # add num_restart
    X = X.expand(X.shape[0], X.shape[1], num_restarts, *X.shape[3:])
    # [batch, num_or, num_restarts, *input_shape]

    if initialization == 'osi':
        osi_start_time = time.time()
        X_init = OSI_init_C(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
        osi_time = time.time() - osi_start_time
        print(f'diversed PGD initialization time: {osi_time:.4f}')
    if initialization == 'boundary':
        boundary_adv_inputs = boundary_attack(model, X[:,0,...].reshape(-1, *X_shape[2:]), data_min.view(*X_shape), data_max.view(*X_shape))
        if boundary_adv_inputs is not None:
            X_init = boundary_adv_inputs.view(X.shape[1], -1, *X.shape[3:])
            X = X[:,:X_init.shape[2],...]
        else:
            initialization = 'uniform'

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    if initialization == 'osi' or initialization == 'boundary':
        delta = (X_init - X).detach().requires_grad_()
    elif initialization == 'uniform':
        # the part below is used to reproduce the original implementation.
        X_ = X.transpose(1, 2)
        data_max_ = data_max.transpose(1, 2)
        data_min_ = data_min.transpose(1, 2)
        delta_upper_limit_ = data_max_ - X_
        delta_lower_limit_ = data_min_ - X_
        delta = (torch.empty_like(X_).uniform_() * (delta_upper_limit_ - delta_lower_limit_) + delta_lower_limit_)
        delta = delta.transpose(1, 2).contiguous().requires_grad_()
        del X_, data_max_, data_min_, delta_upper_limit_, delta_lower_limit_

        # # the part below is more natural and more efficient, but may not reproduce the original implementation.
        # delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()

        # [batch, num_or, num_restarts, *input_shape]
    elif initialization == 'none':
        delta = torch.zeros_like(X).requires_grad_()
        # [batch, num_or, num_restarts, *input_shape]
    else:
        raise ValueError(f"Unknown initialization method {initialization}")

    if use_adam:
        opt = AdamClipping(params=[delta], lr=alpha)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
    else:
        # alpha: [num_or, *input_shape] -> [1, num_or, 1, *input_shape]
        # will boardcast to [batch, num_or, num_restarts, *input_shape] when used.
        alpha = alpha.unsqueeze(1).unsqueeze(0)

    # precompute the group indices of the belonging OR clauses for AND clauses.
    group_indices, num_or = precompute_group_indices(or_spec_size)
    # [num_spec]

    for iteration in tqdm(range(attack_iters)):
        attack_utils.stats.accumulate(num_restarts, 1)
        # additional clamping to make sure inputs in the range
        # otherwise inputs may out of range due to numerical error when updating delta
        inputs = normalize(X + delta).clamp(min=data_min, max=data_max)
        # [batch, num_or, num_restarts, *input_shape]
        output = model(inputs.view(-1, *X_shape[2:])).view(batch_size, X_shape[1], num_restarts, num_output)
        # [batch, num_or, num_restarts, num_output]

        if GAMA_loss:
            # Output on original model is needed if gama loss is used.
            origin_out = torch.softmax(model(normalize(X.reshape(-1, *X_shape[2:]))), 1)
            origin_out = origin_out.view(output.shape)
            # [batch, num_or, num_restarts, num_output]
        else:
            origin_out = None

        loss_function = eval(arguments.Config["attack"]["pgd_loss"])
        loss, total_loss = loss_function(
            origin_out, output, C_mat, rhs_mat,
            or_spec_size,
            gama_lambda if GAMA_loss else 0.0,
            mode=arguments.Config['attack']['pgd_loss_mode'], model=model)
        # [batch, num_spec, num_restarts], [batch]
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        total_loss.backward()

        # we find the current best loss among 'or' conditions
        curr_best_losses, restart_indices = find_optimal_loss(loss, group_indices, num_or)
        # [batch, num_or], [batch, num_or]
        # we select the corresponding delta among all restarts
        # TODO: use torch.gather.
        batch_range = torch.arange(delta.size(0)).unsqueeze(1).expand(delta.size(0), delta.size(1))
        # [batch, num_or]
        num_x_range = torch.arange(delta.size(1)).expand_as(restart_indices)
        # [batch, num_or]
        delta_targeted = delta[batch_range, num_x_range, restart_indices]
        # [batch, num_or, *input_shape]
        best_delta[curr_best_losses >= best_losses] = delta_targeted[curr_best_losses >= best_losses]
        # [batch, num_or, *input_shape]
        # for each batch element, we select the best loss among all 'or' conditions
        best_losses = torch.max(best_losses, curr_best_losses)
        # [batch, num_or]
        if early_stop:
            # we select the best input and output for the best one OR spec for each batch
            # to check the early stop condition for efficiency.
            # indices: [batch]
            best_or_indices = curr_best_losses.max(dim=1).indices
            best_restart_indices_best_or = restart_indices.gather(dim=1, index=best_or_indices.unsqueeze(1)).squeeze(1)

            # test_inputs: [batch, 1, 1, *input_shape], test_outputs: [batch, 1, 1, output_dim]
            test_inputs = inputs[batch_indices, best_or_indices, best_restart_indices_best_or].unsqueeze(1).unsqueeze(1)
            test_data_max = data_max[batch_indices, best_or_indices].unsqueeze(1)
            test_data_min = data_min[batch_indices, best_or_indices].unsqueeze(1)
            test_outputs = output[batch_indices, best_or_indices, best_restart_indices_best_or].unsqueeze(1).unsqueeze(1)

            spec_mask = torch.zeros((batch_size, num_or), dtype=torch.bool, device=device)
            spec_mask[batch_indices, best_or_indices] = True
            # [batch_size, num_or]
            spec_mask = spec_mask.repeat_interleave(or_spec_size, dim=1)
            # [batch_size, num_spec]

            # test_or_spec_size is the max number of ANDs in the best OR specs among all batches. [1]
            test_or_spec_size = or_spec_size[best_or_indices].max(dim=0, keepdim=True).values

            if (or_spec_size[best_or_indices] == or_spec_size[best_or_indices][0]).all():
                # the best C_mat and rhs_mat for each batch have the same number of specs (ANDs),
                # slice the items for the best OR spec.
                num_and = test_or_spec_size.item()
                test_C_mat = C_mat[spec_mask].view(batch_size, num_and, -1)
                # [batch_size, num_and, output_dim]
                test_rhs_mat = rhs_mat[spec_mask].view(batch_size, num_and)
                # [batch_size, num_and]
            else:
                # since the best C_mat and rhs_mat for each batch may have different number of specs (ANDs),
                # we can only slice them in list and then pad into a tensor.
                # we pad C_mat with 0 and rhs_mat with inf, so that the padded specs are always satisfied.
                # list of tensors: [different_num_and, output_dim]
                test_C_mat = [C_mat[i][spec_mask[i]] for i in range(batch_size)]
                # [batch_size, max_num_and, output_dim]
                test_C_mat = pad_list_of_input_to_tensor(
                    test_C_mat, pad_value=0, pad_dim=0, batch_dim=None, is_orginal_tensor=True, device=device
                )
                # list of tensors: [different_num_and]
                test_rhs_mat = [rhs_mat[i][spec_mask[i]] for i in range(batch_size)]
                # [batch_size, max_num_and]
                test_rhs_mat = pad_list_of_input_to_tensor(
                    test_rhs_mat, pad_value=float("inf"), pad_dim=0, batch_dim=None, is_orginal_tensor=True, device=device
                )

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
            unchanged = ((delta - old_delta).abs().sum([1] + list(range(3, delta.ndim)), keepdim=True) == 0).to(delta.dtype)
            total_replaced_deltas += int(unchanged.sum().item())
            new_init = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit)
            delta.data.copy_(delta * (1 - unchanged) + new_init * unchanged)

    if restart_when_stuck:
        total_num_deltas = batch_size * num_restarts * (iteration + 1)
        replaced_percentage = total_replaced_deltas / total_num_deltas * 100
        print(f'Attack batch size: {batch_size}, restarts: {num_restarts}, iterations: {iteration + 1} '
              f'replaced deltas {total_replaced_deltas} ({replaced_percentage}%)')

    adv_input_per_or, adv_output_per_or, adv_margin_per_spec, adv_margin_per_or = eval(
        arguments.Config["attack"]["adv_example_finalizer"]
    )(model, original_X, best_delta, data_min.squeeze(2), data_max.squeeze(2), C_mat, rhs_mat, or_spec_size)
    # [batch_size, num_or, *input_shape], [batch_size, num_or, num_output],
    # [batch_size, num_spec], [batch_size, num_or]

    attack_success, best_or_idx = test_conditions(
        adv_input_per_or.unsqueeze(2), adv_output_per_or.unsqueeze(2),
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
        adv_input_all = (delta + original_X.unsqueeze(2)).clamp(
            min=data_min.unsqueeze(2), max=data_max.unsqueeze(2))
        # [batch_size, num_or, num_restarts, *input_shape]

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
