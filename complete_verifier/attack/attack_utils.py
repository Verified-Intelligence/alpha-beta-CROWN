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

import os
import sys
import subprocess
import math
import numpy as np
import torch
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import Optional
from string import Template

import arguments
from cex_validation import CEX_INPUT_TOLERANCE, CEX_OUTPUT_TOLERANCE
from load_model import inference_onnx, Customized
from auto_LiRPA import BoundedTensor
from utils import unpad_to_list_of_tensors, expand_batch, assert_allowed_caller

class Stats:
    def __init__(self):
        self.num_restarts = 0
        self.num_steps = 0

    def reset(self):
        """Reset statistics."""
        self.num_restarts = 0
        self.num_steps = 0

    def report(self):
        """Return current statistics as a dict."""
        return {
            'num_restarts': self.num_restarts,
            'num_steps': self.num_steps
        }

    def accumulate(self, restarts, steps):
        """Accumulate new restarts and steps."""
        self.num_restarts += restarts
        self.num_steps += steps

stats = Stats()


def reset_attack_stats():
    if arguments.Config['attack']['pgd_order'] != 'skip':
        stats.reset()

def get_attack_stats(logger, new_idx):
    if arguments.Config['attack']['pgd_order'] != 'skip':
        logger.record_pgd_stats(new_idx, stats.report())


def process_data_for_attack(x: BoundedTensor, c: torch.Tensor, rhs: torch.Tensor, or_spec_size: torch.Tensor):
    num_or = or_spec_size.numel()
    if x.shape[0] == 1 and num_or > 1:
        x = expand_batch(x, num_or)
    assert x.shape[0] == num_or
    data_min = x.ptb.x_L
    data_max = x.ptb.x_U
    x = x.data

    if c.shape[0] == 1 and num_or > 1:
        # c: [1, num_spec, num_output] -> [num_spec, num_output]
        # rhs: [1, num_spec] -> [num_spec]
        assert c.shape[1] == or_spec_size.sum().item()
        c = c.squeeze(0)
        rhs = rhs.squeeze(0)
    elif c.shape[0] == num_or:
        same_or_spec_size = (or_spec_size == or_spec_size[0]).all()
        if same_or_spec_size:
            # c: [num_or, num_and, num_output] -> [num_spec, num_output]
            # rhs: [num_or, num_and] -> [num_spec]
            # num_spec = num_or * num_and
            c = c.view(-1, c.shape[-1])
            rhs = rhs.view(-1)
        else:
            # c: [num_or, max_num_and, num_output] -> list of [1, num_and, num_output] -> [num_spec, num_output]
            # rhs: [num_or, num_and] -> list of [1, num_and] -> [num_spec]
            # num_spec = sum(or_spec_size)
            c = torch.cat(unpad_to_list_of_tensors(c, 0, 1, or_spec_size, True), dim=1).squeeze(0)
            rhs = torch.cat(unpad_to_list_of_tensors(rhs, 0, 1, or_spec_size, True), dim=1).squeeze(0)
    else:
        raise ValueError(f"Invalid shape of c: {c.shape}, or_spec_size: {or_spec_size}")

    return x, data_min, data_max, c, rhs


@dataclass
class PGDAttackResult:
    # [batch_size]. whether the attack is successful for each batch.
    attack_success: torch.Tensor
    # [batch_size]. the index of the OR spec with the smallest attack margin for each batch.
    best_or_idx: torch.Tensor
    # [batch_size, num_or, *input_shape]. the adv example of each OR spec for each batch.
    adv_input_per_or: torch.Tensor
    # [batch_size, num_or, num_output]. the output of each OR spec for each batch.
    adv_output_per_or: torch.Tensor
    # [batch_size, num_or]. the attack margin of each OR spec for each batch.
    # (max margin among all ANDs in the OR spec)
    adv_margin_per_or: torch.Tensor
    # [batch_size, *input_shape]. the adv example of the best OR spec for each batch.
    # the best OR spec is the one with the smallest attack margin.
    adv_input_best: torch.Tensor
    # [batch_size, num_output]. the output of the best OR spec for each batch.
    adv_output_best: torch.Tensor
    # [batch_size]. the attack margin of the best OR spec for each batch.
    adv_margin_best: torch.Tensor
    # [batch_size, num_spec]. the attack margin of each spec for each batch.
    adv_margin_per_spec: torch.Tensor
    # [batch_size, num_restarts, num_or, *input_shape] / None.
    # all adv examples of all OR specs for each batch if bab attack is enabled.
    adv_input_all: Optional[torch.Tensor] = None


def _has_open_constraints(strictness) -> bool:
    """Return whether any lowered entry has open input or spec constraints."""
    if strictness is None:
        return False
    for branch in strictness:
        if bool(branch.input_lower.any()) or bool(branch.input_upper.any()):
            return True
        if any(bool(spec.any()) for spec in branch.specs):
            return True
    return False


def _inside_open_aware_box(x, input_box, branch_strictness, tol):
    """Check a flattened input against closed and open box bounds."""
    box = torch.tensor(input_box, dtype=x.dtype, device=x.device)
    lower, upper = box[:, 0], box[:, 1]
    lower_strict = torch.as_tensor(
        branch_strictness.input_lower, dtype=torch.bool, device=x.device)
    upper_strict = torch.as_tensor(
        branch_strictness.input_upper, dtype=torch.bool, device=x.device)

    if x.numel() != lower.numel():
        return False
    closed_lower = x >= lower - tol
    closed_upper = x <= upper + tol
    strict_lower = x > lower
    strict_upper = x < upper
    # Strict masks select the bounds that exclude equality.
    lower_ok = torch.where(lower_strict, strict_lower, closed_lower).all()
    upper_ok = torch.where(upper_strict, strict_upper, closed_upper).all()
    return bool((lower_ok & upper_ok).item())


def _satisfies_open_aware_spec(y, spec, spec_strictness, output_tol):
    """Check one AND spec with per-row open-constraint metadata."""
    mat, rhs = spec
    mat = torch.tensor(mat, dtype=y.dtype, device=y.device)
    rhs = torch.tensor(rhs, dtype=y.dtype, device=y.device)
    strict = torch.as_tensor(
        spec_strictness, dtype=torch.bool, device=y.device)

    if mat.shape[-1] != y.numel() or rhs.numel() != mat.shape[0]:
        return False
    value = torch.matmul(mat, y)
    closed_rows = value <= rhs + output_tol
    strict_rows = value < rhs
    # Each row is an AND constraint. Strict rows reject equality-boundary
    # counterexamples, while closed rows use the configured output tolerance.
    return bool(torch.where(strict, strict_rows, closed_rows).all().item())


def satisfies_open_constraints(
        adv_input, adv_output, vnnlib,
        input_tol=CEX_INPUT_TOLERANCE,
        output_tol=CEX_OUTPUT_TOLERANCE):
    """Check open-bound metadata at the counterexample boundary.

    The lowered vnnlib keeps the normal verifier structure:

    * entries are OR branches over input boxes;
    * each entry contains OR alternatives in ``spec_list``;
    * rows within each spec matrix are AND constraints.

    Strictness metadata selects the rows and input bounds that must be checked
    as open inequalities.
    """
    metadata = getattr(vnnlib, "metadata", None)
    strictness = getattr(metadata, "strictness", None)
    if not _has_open_constraints(strictness):
        return True

    x = adv_input.view(-1)
    y = adv_output.view(-1)
    assert len(vnnlib) == len(strictness)

    for (input_box, spec_list), branch_strictness in zip(
            vnnlib, strictness):
        if not _inside_open_aware_box(
                x, input_box, branch_strictness, input_tol):
            continue
        assert len(spec_list) == len(branch_strictness.specs)
        for spec, spec_strictness in zip(spec_list, branch_strictness.specs):
            if _satisfies_open_aware_spec(
                    y, spec, spec_strictness, output_tol):
                return True
    return False


def check_and_save_cex(adv_input, adv_output, vnnlib, res_path, expected_verified_status):
    """
    adv_input: [batch_size, *input_shape]
    adv_output: [batch_size, num_output]
    expected_verified_status: <"unsafe-pgd", "unsafe-bab", "unsafe", ...>
    """
    print('\nChecking and Saving Counterexample in check_and_save_cex')
    assert adv_input.shape[0] == 1, f'The batch_size of adv_input should be 1.'
    assert adv_output.dim() == 2 and adv_output.shape[0] == 1, \
        f'The adv_output should be in the shape of (batch_size (1), num_output).'
    assert vnnlib is not None, f'Cached vnnlib should be used to enable check specs conditions.'

    verified_status = expected_verified_status
    verified_success = True
    saved_adv_example = False
    metadata = getattr(vnnlib, "metadata", None)

    if not satisfies_open_constraints(adv_input, adv_output, vnnlib):
        print('WARNING: Counterexample does not satisfy open constraints.')
        return 'unknown', False

    if arguments.Globals['example_idx'] >= 0:
        res_path = Template(res_path).substitute(IDX=arguments.Globals['example_idx'])

    if arguments.Config['general']['save_adv_example']:
        verify_with_onnxruntime = arguments.Config[
            'general']['verify_onnxruntime_output']
        if eval(arguments.Config['attack']['adv_verifier'])(adv_input, adv_output, vnnlib,
                                                            verify_with_onnxruntime):
            print('Saving counterexample to', os.path.abspath(res_path))
            adv_saver = eval(arguments.Config['attack']['adv_saver'])
            adv_saver(adv_input, adv_output, res_path, vnnlib=vnnlib)
            verified_status = expected_verified_status
            verified_success = True
            saved_adv_example = True
        else:
            verified_status = 'unknown'
            verified_success = False

    if arguments.Config['general']['eval_adv_example']:
        if saved_adv_example:
            _run_external_cex_validation(res_path, metadata)
        else:
            raise RuntimeError(
                "External counterexample validation requires a saved "
                "counterexample file.")

    if arguments.Config['general']['show_adv_example']:
        print('Adv example:')
        print(adv_input[0])
    print()
    return verified_status, verified_success


def validate_counterexample_with_onnxruntime(adv_input, vnnlib):
    """Check a candidate counterexample using ONNXRuntime output."""
    metadata = getattr(vnnlib, "metadata", None)
    if metadata is not None:
        return _vnnlib2_onnx_replay_counterexample_valid(adv_input, metadata)

    onnx_path = _onnx_path_for_validation(
        'ONNXRuntime counterexample validation', required=True)

    try:
        onnx_output = inference_onnx(
            onnx_path, adv_input.detach().cpu().numpy())
    except Exception as e:
        raise RuntimeError(
            f'ONNXRuntime counterexample validation failed: {e}') from e

    onnx_spec_output = torch.as_tensor(
        onnx_output, dtype=adv_input.dtype,
        device=adv_input.device).reshape(1, -1)

    return is_specification_vio(
        vnnlib, adv_input.view(-1), onnx_spec_output.view(-1),
        tol=CEX_INPUT_TOLERANCE, output_tol=CEX_OUTPUT_TOLERANCE)


def _vnnlib2_assertion_holds(expr, assignment):
    """Evaluate one original VNN-LIB 2.0 assertion under concrete values."""
    def arithmetic_value(node):
        node_type = type(node).__name__
        if node_type == "Var":
            return assignment[node.name][tuple(node.indices)]
        if node_type in ("Float", "Int", "IntExpr"):
            return node.value
        if node_type == "Literal":
            return float(node.lexeme)
        if node_type == "Negate":
            return -arithmetic_value(node.expr)
        if node_type == "Plus":
            return sum(arithmetic_value(arg) for arg in node.args)
        if node_type == "Minus":
            value = arithmetic_value(node.head)
            return value - sum(arithmetic_value(arg) for arg in node.rest)
        if node_type == "Multiply":
            return math.prod(arithmetic_value(arg) for arg in node.args)
        raise NotImplementedError(
            f'Unsupported VNN-LIB 2.0 expression {node_type}')

    def boolean_value(node):
        node_type = type(node).__name__
        if node_type == "And":
            return all(boolean_value(arg) for arg in node.args)
        if node_type == "Or":
            return any(boolean_value(arg) for arg in node.args)

        lhs = arithmetic_value(node.lhs)
        rhs = arithmetic_value(node.rhs)
        if node_type == "GreaterThan":
            return lhs > rhs
        if node_type == "LessThan":
            return lhs < rhs
        if node_type == "GreaterEqual":
            return lhs >= rhs
        if node_type == "LessEqual":
            return lhs <= rhs
        if node_type == "Equal":
            return lhs == rhs
        if node_type == "NotEqual":
            return lhs != rhs
        raise NotImplementedError(
            f'Unsupported VNN-LIB 2.0 expression {node_type}')

    return boolean_value(expr)


def _vnnlib2_onnx_replay_counterexample_valid(adv_input, metadata):
    """Validate VNN-LIB 2.0 cex candidates against original ONNX execution."""
    vnnlib_path = arguments.Config["specification"]["vnnlib_path"]
    if vnnlib_path is None:
        raise RuntimeError(
            'VNN-LIB 2.0 ONNX replay requires a VNNLIB path.')
    network_paths = getattr(metadata, "network_paths", None) or {}
    if not network_paths:
        onnx_path = arguments.Config["model"]["onnx_path"]
        if onnx_path is None:
            raise RuntimeError(
                'VNN-LIB 2.0 ONNX replay requires ONNX paths.')
        if len(metadata.networks) != 1:
            raise RuntimeError(
                'VNN-LIB 2.0 ONNX replay for multiple networks requires '
                'named ONNX paths.')
        network_paths = {metadata.networks[0].name: onnx_path}

    try:
        import vnnlib as vnnlib_parser  # pylint: disable=import-outside-toplevel

        flat_input = adv_input.detach().cpu().reshape(-1).numpy()
        input_index_map = np.asarray(metadata.input_index_map, dtype=np.int64)
        assignment = {}
        for network in metadata.networks:
            onnx_path = network_paths.get(network.name)
            if (onnx_path is None and network.relation
                    and network.relation[0] == 'equal-to'):
                onnx_path = network_paths.get(network.relation[1])
            if onnx_path is None:
                raise RuntimeError(
                    f'Missing ONNX path for VNN-LIB network {network.name}.')
            if len(network.outputs) != 1:
                raise RuntimeError(
                    'VNN-LIB 2.0 ONNX replay does not support multiple '
                    f'outputs for network {network.name}.')

            onnx_inputs = []
            for decl in network.inputs:
                indices = input_index_map[decl.offset:decl.offset + decl.size]
                value = flat_input[indices].reshape(decl.shape)
                assignment[decl.name] = value
                onnx_inputs.append(value)

            if len(onnx_inputs) == 1:
                onnx_inputs = onnx_inputs[0]
            onnx_output = inference_onnx(onnx_path, onnx_inputs).reshape(-1)
            output_decl = network.outputs[0]
            if onnx_output.size != output_decl.size:
                raise RuntimeError(
                    f'ONNX output for VNN-LIB network {network.name} has '
                    f'{onnx_output.size} values, expected {output_decl.size}.')
            assignment[output_decl.name] = onnx_output.reshape(
                output_decl.shape)

        query = vnnlib_parser.parse_query_file(vnnlib_path)
        for assertion_idx, assertion in enumerate(query.assertions):
            if not _vnnlib2_assertion_holds(assertion.expr, assignment):
                print('VNN-LIB 2.0 ONNX replay failed assertion '
                      f'{assertion_idx}.')
                return False
    except Exception as error:
        raise RuntimeError(
            f'VNN-LIB 2.0 ONNX replay failed: {error}') from error
    return True


def _run_external_cex_validation(res_path, metadata=None):
    if metadata is not None:
        print(
            'Skipping external counterexample validation: VNN-LIB 2.0 '
            'assignment files are not supported by check_counterexample.py.')
        return True

    onnx_path = _onnx_path_for_validation('external counterexample validation')
    vnnlib_path = arguments.Config['specification']['vnnlib_path']

    if onnx_path is None:
        return True

    onnx_path = os.path.join(os.getcwd(), onnx_path)
    vnnlib_path = os.path.join(os.getcwd(), vnnlib_path)
    current_dir = os.path.dirname(__file__)
    script_path = os.path.join(current_dir, '../', 'check_counterexample.py')

    subprocess.run(
        [sys.executable, script_path, onnx_path, vnnlib_path, res_path],
        check=True)


def is_valid_attack_counterexample(
        adv_input, adv_output, vnnlib, verify_with_onnxruntime=False):
    """Return whether one candidate passes concrete counterexample checks."""
    if vnnlib is not None and not satisfies_open_constraints(
            adv_input, adv_output, vnnlib):
        return False
    verify_with_onnxruntime = (
        verify_with_onnxruntime
        or arguments.Config['general']['verify_onnxruntime_output'])
    return eval(arguments.Config['attack']['adv_verifier'])(
        adv_input, adv_output, vnnlib, verify_with_onnxruntime)


def are_attack_counterexamples_valid(
        adv_input, adv_output, vnnlib, verify_with_onnxruntime=False):
    """Return whether every candidate in a batch is a valid counterexample."""
    if vnnlib is None:
        return True

    adv_input = adv_input.detach()
    adv_output = adv_output.detach()
    candidate_rank = adv_output.dim() - 1
    adv_input = adv_input.reshape(-1, *adv_input.shape[candidate_rank:])
    adv_output = adv_output.reshape(-1, adv_output.shape[-1])
    for batch_idx in range(adv_input.shape[0]):
        if not is_valid_attack_counterexample(
                adv_input[batch_idx:batch_idx + 1],
                adv_output[batch_idx:batch_idx + 1],
                vnnlib, verify_with_onnxruntime=verify_with_onnxruntime):
            return False
    return True


def invalid_attack_success_mask(
        adv_input_per_or, adv_output_per_or, adv_margin_per_or, vnnlib,
        attack_tolerance, verify_with_onnxruntime=False):
    """Return successful OR candidates that fail concrete validation."""
    success_mask = adv_margin_per_or <= attack_tolerance
    invalid_mask = torch.zeros_like(success_mask, dtype=torch.bool)
    if vnnlib is None or not success_mask.any():
        return invalid_mask

    for batch_idx, or_idx in success_mask.nonzero(as_tuple=False).tolist():
        if not is_valid_attack_counterexample(
                adv_input_per_or[batch_idx:batch_idx + 1, or_idx],
                adv_output_per_or[batch_idx:batch_idx + 1, or_idx],
                vnnlib, verify_with_onnxruntime=verify_with_onnxruntime):
            invalid_mask[batch_idx, or_idx] = True
    return invalid_mask


class AdamClipping(Optimizer):
    r"""Implements Adam algorithm, with per-parameter gradient clipping.
    The function is from PyTorch source code.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @staticmethod
    @torch.no_grad()
    # @torch.jit.script
    def _clip_update(exp_avg : torch.Tensor, denom : torch.Tensor, step_size : float, clipping_step_eps : float, lower_limit : torch.Tensor, upper_limit : torch.Tensor, p : torch.Tensor):
        # Compute the Adam update.
        update = exp_avg / denom * step_size
        # update = p.grad
        # Linf norm, scale according to sign.
        scaled_update = torch.sign(update) * clipping_step_eps
        # Apply the update.
        d = p.data + scaled_update
        # Avoid out-of-boundary updates.
        d = d.clamp(min=lower_limit, max=upper_limit)
        p.copy_(d)

    @torch.no_grad()
    def step(self, clipping=None, lower_limit=None, upper_limit=None, sign=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Currently we only deal with 1 parameter group.
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if clipping:
                    assert sign == 1  # gradient ascent for adversarial attacks.
                    self._clip_update(exp_avg, denom, step_size, step_size, lower_limit, upper_limit, p)
                else:
                    # No clipping. Original Adam update.
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def precompute_group_indices(or_spec_size: torch.Tensor):
    '''
    Precompute group indices of the belonging OR clauses for AND clauses.

    Args:
    or_spec_size: A 1D tensor defining the size of each 'AND' group. len(or_spec_size) = num_or

    Return:
    group_indices: [num_spec]
    num_or: number of OR clauses.
    '''
    num_or = or_spec_size.shape[0]
    group_indices = torch.repeat_interleave(torch.arange(num_or, device=or_spec_size.device), or_spec_size)
    return group_indices, num_or


def OSI_init_C(model, X, alpha, output_dim, iter_steps=50, lower_limit=0.0, upper_limit=1.0):
    # the general version of OSI initialization.
    input_shape = X.shape
    # [batch_size, num_restarts, num_or, *X_shape[1:]]
    X_init = X.clone().detach()
    # [batch_size, num_restarts, num_or, *X_shape[1:]]
    X_init = X_init.view(-1, *X_init.shape[3:])
    X = X.reshape(-1, *X.shape[3:])
    # [batch_size, * num_restarts * num_or, *X_shape[1:]]

    w_d = (torch.rand([X.shape[0], output_dim], device=X.device) - 0.5) * 2

    for i in range(iter_steps):
        X_init = X_init.detach().requires_grad_()
        output = model(X_init)

        # test whether we need to early stop here.

        dot = torch.einsum('...,...->', w_d, output)
        # dot = (w_d * output).sum()
        dot.backward()

        with torch.no_grad():
            X_init = X_init + alpha * torch.sign(X_init.grad)
            X_init = X_init.view(input_shape)
            X_init = torch.max(torch.min(X_init, upper_limit), lower_limit)
            X_init = X_init.view(-1, *X_init.shape[3:])

    X_init = X_init.view(input_shape)
    X = X.view(input_shape)

    assert (X_init <= upper_limit).all()
    assert (X_init >= lower_limit).all()

    return X_init


def boundary_attack(model, x, data_min, data_max):
    perturbation_index = ((data_max - data_min) != 0).view(data_max.shape[0], -1).nonzero()
    # index of the pixels perturbed
    if len(perturbation_index) > 5:
        print("Error: number of perturbed pixels is larger than 5, boundary attack is disabled.")
        return None

    data_max_flatten = data_max.view(data_max.shape[0], -1)
    data_min_flatten = data_min.view(data_min.shape[0], -1)

    adv_input = data_max_flatten
    for idx in perturbation_index:
        adv_input_neg = adv_input.clone()
        adv_input[:, idx] = data_max_flatten[0, idx]
        adv_input_neg[:, idx] = data_min_flatten[0, idx]
        adv_input = torch.cat([adv_input, adv_input_neg], dim=0)

    return adv_input.view(-1, *data_max.shape[1:])


def default_adv_saver(adv_input, adv_output, res_path, vnnlib=None):
    '''
    Saves an adversarial example and its corresponding outputs to a specified file in a
    format compatible with further analysis or validation processes.

    Parameters:
    - adv_input: [1, *input_shape]
    - adv_output: [1, num_output]
    - res_path: String specifying the path where the counter-example should be saved.
    '''
    if not assert_allowed_caller(["check_and_save_cex"]):
        raise RuntimeError(
            "Adversarial example savers must be called through "
            "check_and_save_cex().")
    metadata = getattr(vnnlib, "metadata", None)
    adv_input = adv_input.detach().cpu().numpy()
    adv_output = adv_output.view(-1).detach().cpu().numpy()
    if arguments.Config['general']['onnx_adv_example']:
        # Inference using ONNX Runtime to get the output
        onnx_path = _onnx_path_for_validation('ONNX adversarial output')
        if onnx_path is not None:
            adv_output = inference_onnx(onnx_path, adv_input).flatten()
    adv_input = adv_input.flatten()

    if metadata is not None:
        _save_vnnlib2_assignment(adv_input, adv_output, res_path, metadata)
        return
    _save_vnnlib1_assignment(adv_input, adv_output, res_path)


def _save_vnnlib1_assignment(flat_input, flat_output, res_path):
    if res_path.endswith(".pt"):
        torch.save((flat_input, flat_output), res_path)
    else:
        with open(res_path, 'w+') as f:
            f.write("(")
            for i, value in enumerate(flat_input):
                f.write("(X_{}  {})\n".format(i, value))
            f.write("(Y_{}  {})".format(0, flat_output[0]))
            for j, value in enumerate(flat_output[1:], start=1):
                f.write("\n(Y_{}  {})".format(j, value))
            f.write(")")


def _save_vnnlib2_assignment(flat_input, flat_output, res_path, metadata):
    def write_assignment_block(output_file, decl, values):
        shape = ", ".join(str(dim) for dim in decl.shape)
        output_file.write(f"{decl.name} real [{shape}]\n")
        output_file.write("".join(f"{value:.17g}\n" for value in values))

    input_index_map = np.asarray(metadata.input_index_map, dtype=np.int64)

    with open(res_path, 'w+', encoding='utf-8') as output_file:
        output_file.write("sat\n")
        for network in metadata.networks:
            for decl in network.inputs:
                indices = input_index_map[decl.offset:decl.offset + decl.size]
                write_assignment_block(output_file, decl, flat_input[indices])
            for decl in network.outputs:
                write_assignment_block(
                    output_file, decl,
                    flat_output[decl.offset:decl.offset + decl.size])


def default_adv_verifier(
        adv_input, adv_output, vnnlib,
        verify_with_onnxruntime=False):
    """
    Do two kinds of check on counterexample:

    1.check the counterexample with ONNXRuntime when
      verify_with_onnxruntime is True.
      VNN-LIB 2.0 has original assertions in addition to the verifier's
      lowered spec. Replay the original ONNX model and evaluate those original
      assertions, matching the scoring checker. For VNN-LIB 1.0, validate the
      verifier's lowered spec with ONNXRuntime outputs.
    2.check the verifier-computed output against the vnnlib spec. Saved
      counterexample paths are expected to provide vnnlib.

    Args:
        adv_output: [1, *intput_shape]
        adv_input: [1, *input_shape]
    """
    if vnnlib is None:
        raise ValueError(
            'default_adv_verifier requires a vnnlib specification. Use '
            'verify_onnxruntime_output_matches_pytorch() for legacy output '
            'consistency checks without a spec.')

    flatten_output = adv_output.view(-1)
    if verify_with_onnxruntime:
        return validate_counterexample_with_onnxruntime(adv_input, vnnlib)

    if not is_specification_vio(
            vnnlib, adv_input.view(-1), flatten_output,
            tol=CEX_INPUT_TOLERANCE, output_tol=CEX_OUTPUT_TOLERANCE):
        return False

    return True


def verify_onnxruntime_output_matches_pytorch(
        adv_input, adv_output, operation='ONNX Runtime output verification'):
    """Legacy check for callers without a specification to validate."""
    onnx_path = _onnx_path_for_validation(operation, required=True)
    return is_onnx_equal_to_pytorch_output(
        onnx_path, adv_input.cpu().numpy(),
        adv_output.view(-1).detach().cpu().numpy(), rel_tol=1e-3)


def _onnx_path_for_validation(operation, required=False):
    """Return the configured ONNX file for optional or required validation."""
    onnx_path = arguments.Config["model"]["onnx_path"]
    if onnx_path is None:
        message = (
            f'{operation} requires a single ONNX model path '
            '(for example, VNN-LIB 2.0 multi-network wrappers do not have one).')
        if required:
            raise RuntimeError(message)
        print(
            f'Skipping {operation}: no single ONNX model path is available '
            '(for example, VNN-LIB 2.0 multi-network wrappers do not have one).')
    return onnx_path


def is_onnx_equal_to_pytorch_output(onnx_path, onnx_adv_input, pytorch_y, rel_tol):
    """
    Compare the output of ONNX Runtime and PyTorch for given inputs.

    Args:
        onnx_path (str): Path to the ONNX model.
        onnx_adv_input (numpy.ndarray): Input for ONNX model.
        pytorch_y (numpy.ndarray): Expected output from PyTorch model.
        rel_tol (float): Relative tolerance for comparison.

    Returns:
        bool: True if the outputs are similar within the given tolerance, False otherwise.
    """
    print('Checking if onnxruntime output is equal to pytorch')
    onnx_output = inference_onnx(onnx_path, onnx_adv_input)
    onnxruntime_y = onnx_output.flatten("C")

    try:
        diff = np.linalg.norm(onnxruntime_y - pytorch_y, ord=np.inf)
        norm = np.linalg.norm(pytorch_y, ord=np.inf)
        if norm < 1e-6:  # don't divide by zero
            rel_error = 0
        else:
            rel_error = diff / norm
    except ValueError as e:
        diff = 9999
        rel_error = 9999

    print(f'L-inf norm difference between onnx execution and pytorch output: {diff} (rel error: {rel_error});' f'(rel_limit: {rel_tol})')
    if rel_error > rel_tol:
        print('WARNING: Failed in the comparison between ONNX Runtime and PyTorch results.')
        return False
    print('Succeed in ONNX Runtime comparison check.')
    return True


def is_specification_vio(
        box_spec_list, x_list, expected_y, tol,
        output_tol=CEX_OUTPUT_TOLERANCE):
    """Check that the spec file was obeyed.

    ``tol`` is only for input-box membership. Output/spec rows are checked with
    ``output_tol`` and default to exact validation for saved counterexamples.
    """
    rv = False

    for i, box_spec in enumerate(box_spec_list):
        input_box, spec_list = box_spec
        assert len(input_box) == len(x_list), f"input box len: {len(input_box)}, x_in len: {len(x_list)}"

        input_box_tensor = torch.tensor(input_box, dtype=x_list.dtype, device=x_list.device)
        lb_tensor, ub_tensor = input_box_tensor[:, 0], input_box_tensor[:, 1]

        # The candidate must belong to this input box before its output specs
        # can witness a violation.
        inside_input_box = torch.all((x_list >= lb_tensor - tol) & (x_list <= ub_tensor + tol))

        if inside_input_box:
            # Check spec
            violated = False

            for j, (prop_mat, prop_rhs) in enumerate(spec_list):
                prop_mat_tensor = torch.tensor(prop_mat, dtype=expected_y.dtype, device=expected_y.device)
                prop_rhs_tensor = torch.tensor(prop_rhs, dtype=expected_y.dtype, device=expected_y.device)

                vec = torch.matmul(prop_mat_tensor, expected_y)
                sat = torch.all(vec <= prop_rhs_tensor + output_tol)

                if sat:
                    violated = True
                    break

            if violated:
                rv = True
                break

    if rv:
        print('Succeed in specification conditions check.')
    else:
        print('WARNING: Failed in specs conditions check.')
    return rv
