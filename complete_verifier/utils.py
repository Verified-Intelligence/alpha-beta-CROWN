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

import copy
import os
import time
import pickle
import arguments
from dataclasses import dataclass
from collections import defaultdict
from string import Template
import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

@dataclass
class Timer:
    total_func_time: float = 0.0
    total_prepare_time: float = 0.0
    total_bound_time: float = 0.0
    total_beta_bound_time: float = 0.0
    total_transfer_time: float = 0.0
    total_finalize_time: float = 0.0

    def __init__(self):
        self.time_start = {}
        self.time_last = {}
        self.time_sum = {}

    def start(self, name):
        self.time_start[name] = time.time()
        if name not in self.time_sum:
            self.time_sum[name] = 0

    def add(self, name):
        self.time_last[name] = time.time() - self.time_start[name]
        self.time_sum[name] += self.time_last[name]

    def print(self):
        print('Time: ', end='')
        for k, v in self.time_last.items():
            print(f'{k} {v:.4f}', end='    ')
        print()
        print('Accumulated time: ', end='')
        for k, v in self.time_sum.items():
            print(f'{k} {v:.4f}', end='    ')
        print()


class Logger:
    def __init__(self, run_mode, save_path, timeout_threshold):
        self.run_mode = run_mode
        self.save_path = save_path
        self.timeout_threshold = timeout_threshold
        self.verification_summary = defaultdict(list)
        self.time_all_instances = []
        self.status_per_sample_list = []
        self.bab_ret = []
        self.count = 0
        self.pgd_stats = {}

    def update_timeout(self, timeout):
        self.timeout_threshold = timeout

    def record_start_time(self):
        self.start_time = time.time()

    def record_pgd_stats(self, idx, stats):
        self.pgd_stats[idx] = stats

    def summarize_results(self, verified_status, index):
        self.count += 1
        if self.run_mode == 'single_vnnlib':
            # run in run_instance.sh
            if ('unknown' in verified_status or 'timeout' in verified_status
                or 'timed out' in verified_status):
                verified_status = 'timeout'
            elif 'unsafe' in verified_status:
                verified_status = 'sat'
            elif 'safe' in verified_status:
                verified_status = 'unsat'
            else:
                raise ValueError(f'Unknown verified_status {verified_status}')

            print('Result:', verified_status)
            time_all = time.time() - self.start_time
            print('Time:', time_all)
            with open(self.save_path, 'w') as file:
                file.write(verified_status)
                if arguments.Config['general']['save_adv_example']:
                    if verified_status == 'sat':
                        file.write('\n')
                        cex_path = arguments.Config['attack']['cex_path']
                        with open(cex_path, 'r') as adv_example:
                            file.write(adv_example.read())
                file.flush()

            # For single_vnnlib, save the output for test here instead of in finish()
            if arguments.Config['general']['save_output']:
                # save output for test
                arguments.Globals['out']['results'] = verified_status
                arguments.Globals['out']['time'] = time_all
                if len(self.bab_ret) != 0:
                    arguments.Globals['out']['domains_visited'] = sum(
                        item[2] for item in self.bab_ret
                    )
                    # self.bab_ret: [[idx, l, nodes, time]]
                    # (see abcrown.complete_verifier)
                arguments.Globals['out']['pgd_stats'] = self.pgd_stats

                with open(arguments.Config['general']['output_file'], 'wb') as f:
                    pickle.dump(arguments.Globals['out'], f)
                print(f"Result dict saved to {arguments.Config['general']['output_file']}.")

        else:
            if time.time() - self.start_time > self.timeout_threshold:
                if 'unknown' not in verified_status:
                    verified_status += ' (timed out)'
            self.verification_summary[verified_status].append(index)
            self.status_per_sample_list.append(
                [verified_status, time.time() - self.start_time])
            self._save()
            print(f'Result: {verified_status} '
                  f'in {self.status_per_sample_list[-1][1]:.4f} seconds')

    def finish(self):
        if self.run_mode != 'single_vnnlib':
            # Finished all examples.
            time_timeout = [
                s[1] for s in self.status_per_sample_list if 'unknown' in s[0]]
            time_verified = [
                s[1] for s in self.status_per_sample_list
                if 'safe' in s[0] and 'unsafe' not in s[0]]
            time_unsafe = [
                s[1] for s in self.status_per_sample_list if 'unsafe' in s[0]]
            time_all_instances = [s[1] for s in self.status_per_sample_list]
            self._save()

            count_timeout = len(time_timeout)
            count_verified = len(time_verified)
            count_unsafe = len(time_unsafe)
            count_all_instances = len(time_all_instances)
            max_time_timeout = max(time_timeout) if count_timeout > 0 else 0
            max_time_verified = max(time_verified) if count_verified > 0 else 0
            max_time_unsafe = max(time_unsafe) if count_unsafe > 0 else 0
            max_time_all_instances = max(time_all_instances) if count_all_instances > 0 else 0
            sum_time_verified = sum(time_verified)
            sum_time_unsafe = sum(time_unsafe)
            sum_time_timeout = sum(time_timeout)
            sum_time_all_instances = sum(time_all_instances)

            print('############# Summary #############')
            acc = count_verified / self.count * 100.
            print(f'Final verified acc: {acc}% (total {self.count} examples)')
            print('Problem instances count:',
                  count_verified + count_unsafe + count_timeout,
                  ', total verified (safe/unsat):', count_verified,
                  ', total falsified (unsafe/sat):', count_unsafe,
                  ', timeout:', count_timeout)
            print('mean time for ALL instances '
                  f'(total {count_all_instances}):'
                  f'{sum_time_all_instances / (count_all_instances + 1e-5)},'
                  f' max time: {max_time_all_instances}')
            if count_verified > 0:
                print('mean time for verified SAFE instances'
                      f'(total {count_verified}): '
                      f'{sum_time_verified / count_verified}, '
                      f'max time: {max_time_verified}')
            if count_verified > 0 and count_unsafe > 0:
                mean_time = (sum_time_verified + sum_time_unsafe) / (
                    count_verified + count_unsafe)
                max_time = max(max_time_verified, max_time_unsafe)
                print('mean time for verified (SAFE + UNSAFE) instances '
                      f'(total {(count_verified + count_unsafe)}):'
                      f' {mean_time}, max time: {max_time}')
            if count_verified > 0 and count_timeout > 0:
                mean_time = (sum_time_verified + sum_time_timeout) / (
                    count_verified + count_timeout)
                max_time = max(max_time_verified, max_time_timeout)
                print('mean time for verified SAFE + TIMEOUT instances '
                      f'(total {(count_verified + count_timeout)}):'
                      f' {mean_time}, max time: {max_time} ')
            if count_unsafe > 0:
                print(f'mean time for verified UNSAFE instances '
                      f'(total {count_unsafe}): '
                      f'{sum_time_unsafe / count_unsafe}, '
                      f'max time: {max_time_unsafe}')

            for k, v in self.verification_summary.items():
                print(f'{k} (total {len(v)}), index:', v)

            if arguments.Config['general']['save_output']:
                # save output for test
                arguments.Globals['out']['results'] = self.status_per_sample_list[0][0]
                if arguments.Globals['out']['results'] == 'unknown':
                    arguments.Globals['out']['results'] = 'timeout'
                arguments.Globals['out']['time'] = time_all_instances[0]
                if len(self.bab_ret) != 0:
                    arguments.Globals['out']['domains_visited'] = sum(
                        item[2] for item in self.bab_ret
                    )
                    # self.bab_ret: [[idx, l, nodes, time]]
                    # (see abcrown.complete_verifier)
                arguments.Globals['out']['pgd_stats'] = self.pgd_stats

                with open(arguments.Config['general']['output_file'], 'wb') as f:
                    pickle.dump(arguments.Globals['out'], f)
                print(f"Result dict saved to {arguments.Config['general']['output_file']}.")

    def _save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'summary': self.verification_summary,
                'results': self.status_per_sample_list,
                'bab_ret': self.bab_ret
            }, f)


class Stats:
    def __init__(self):
        self.visited = 0
        self.timer = Timer()
        self.all_node_split = False


def get_reduce_op(op, with_dim=True):
    """Convert reduce op in str to the actual function."""
    if op is None:
        return op
    elif op in ['min', 'max']:
        return getattr(torch, op)
    elif op == 'mean':
        if with_dim:
            return torch.mean
        else:
            return lambda a, b: (a + b) / 2
    else:
        raise ValueError(op)


def fast_hist_copy(hists):
    """Copy the history for one element. Much faster than deepcopy()."""
    if hists is None:
        return None
    ret = {}
    for k, hist in hists.items():
        if isinstance(hist[0], torch.Tensor):
            ret[k] = hist
        elif isinstance(hist[0], list):
            ret[k] = tuple([item.clone() if isinstance(item, torch.Tensor)
                            else item.copy() for item in hist[:5]])
        else:
            ret[k] = tuple(copy.deepcopy(hist[i]) for i in range(5))
    return ret

def convert_history_from_list(history):
    '''
    Convert the history variables into tensors if they are lists.
    It is because some legacy code creates history as lists.

    return:
        history: a tuple of tensors
            history[0]: relu_idx
            history[1]: relu_status
            history[2]: relu_bias
            history[3]: relu_score
            history[4]: depths
    '''
    if isinstance(history[0], torch.Tensor):
        return history

    return (torch.tensor(history[0], dtype=torch.long),
            torch.tensor(history[1]),
            torch.tensor(history[2]),
            torch.tensor(history[3]),
            torch.tensor(history[4]))

def print_splitting_decisions(net, d, split_depth, split, verbose=False):
    """Print the first two split for first 10 domains."""
    print('splitting decisions: ')
    branching_decision = split['decision']
    batch = next(iter(d['lower_bounds'].values())).shape[0]
    for l in range(split_depth):
        print(f'split level {l}', end=': ')
        for b in range(min(10, batch)):
            decision = branching_decision[l*batch + b]
            print(f'[{net.split_nodes[decision[0]].name}, {decision[1]}]',
                  end=' ')
        print('')
        if verbose:
            if 'points' in split and split['points'] is not None:
                print('Branching points:')
                for b in range(min(50, batch)):
                    idx = l * batch + b
                    decision = branching_decision[l*batch + b]
                    node = net.split_nodes[decision[0]]
                    print('[{:.4f}, {:.4f}]'.format(
                        d['lower_bounds'][node.name][idx].view(-1)[decision[1]],
                        d['upper_bounds'][node.name][idx].view(-1)[decision[1]]),
                        end=' ')
                    print('branched at', end=' ')
                    if split['points'].ndim == 1:
                        print('{:.4f}'.format(split['points'][idx]))
                    else:
                        for i in range(split['points'].shape[-1]):
                            print('{:.4f}'.format(split['points'][idx][i]), end=' ')
                        print()


def check_infeasible_bounds(lower, upper, reduce=False):
    print('Checking infeasibility')
    infeasible = None
    for k in lower:
        infeasible_ = (lower[k] - upper[k]).view(
            lower[k].shape[0], -1).max(dim=-1).values > 1e-6
        # FIXME check_infeasible_bounds first before moving the bounds to CPU
        infeasible_ = infeasible_.cpu()
        infeasible = (infeasible_ if infeasible is None
                      else torch.logical_or(infeasible, infeasible_))
    any_infeasible = infeasible.any()
    if any_infeasible:
        print(f'Infeasiblity detected! {int(infeasible.sum())} domains')
    if reduce:
        return any_infeasible
    else:
        return infeasible


def get_save_path(csv):
    if csv:
        return 'a-b-crown_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}.npz'.format(  # pylint: disable=line-too-long,consider-using-f-string
            os.path.splitext(os.path.basename(arguments.Config['general']['csv_name']))[0],
            arguments.Config['data']['start'],
            arguments.Config['data']['end'], arguments.Config['solver']['beta-crown']['iteration'],
            arguments.Config['solver']['batch_size'],
            arguments.Config['bab']['timeout'], arguments.Config['bab']['branching']['method'],
            arguments.Config['bab']['branching']['reduceop'],
            arguments.Config['bab']['branching']['candidates'],
            arguments.Config['solver']['alpha-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_beta'],
            arguments.Config['attack']['pgd_order'],
            arguments.Config['bab']['cut']['cplex_cuts'])
    else:
        if arguments.Config['model']['name'] is None:
            # use onnx model prefix as model_name
            model_name = arguments.Config['model']['onnx_path'].split('.onnx')[-2].split('/')[-1]
        elif 'Customized' in arguments.Config['model']['name']:
            model_name = 'Customized_model'
        else:
            model_name = arguments.Config['model']['name']
        return 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}.npy'.format(  # pylint: disable=line-too-long,consider-using-f-string
            model_name, arguments.Config['data']['start'], arguments.Config['data']['end'],
            arguments.Config['solver']['beta-crown']['iteration'],
            arguments.Config['solver']['batch_size'],
            arguments.Config['bab']['timeout'], arguments.Config['bab']['branching']['method'],
            arguments.Config['bab']['branching']['reduceop'],
            arguments.Config['bab']['branching']['candidates'],
            arguments.Config['solver']['alpha-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_beta'],
            arguments.Config['attack']['pgd_order'], arguments.Config['bab']['cut']['cplex_cuts'])


def get_batch_size_from_masks(mask):
    return len(next(iter(mask.values())))


def get_unstable_neurons(updated_mask, net):
    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = {k: [item[0:1] if item is not None else None
                        for item in mask]
                    for k, mask in updated_mask.items()}
    for k, masks in updated_mask.items():
        if type(net.net[k]).__name__ == 'BoundRelu':
            # Initialize the ReLU indicators from the updated masks.
            # The set_gcp_relu_indicators() method updates the gcp_unstable_relu_indicators
            # for the specified ReLU layer, thereby restoring its unstable ReLUs information.
            assert len(masks) == 1, 'Expected single mask for ReLU layer'
            if masks[0] is not None:
                # If the mask is not None, it means the ReLU layer has been used and perturbed.
                # We can set the gcp_unstable_relu_indicators for this layer.
                net.net.set_gcp_relu_indicators(k, masks[0])
        for i, mask in enumerate(masks):
            if mask is None: # Not perturbed
                continue
            n_unstable = int(torch.sum(mask).item())
            print(f'Node {k} input {i}: size {mask.shape[1:]} unstable {n_unstable}')
            tot_ambi_nodes += n_unstable
    print(f'-----------------\n# of unstable neurons: {tot_ambi_nodes}\n-----------------\n')
    return updated_mask, tot_ambi_nodes


def expand_path(path):
    dirname = os.path.dirname(arguments.Config.file)
    config_path = '.' if dirname == '' else dirname
    return Template(path).substitute(CONFIG_PATH=config_path)


def print_model(model):
    print('Model:', model)
    if arguments.Config['debug']['view_model']:
        print('Perturbed nodes:')
        for node in model.nodes():
            if node.perturbed:
                print('  ', node)
        print('Nonlinearities:')
        for node in model.nodes():
            if node.perturbed and node.requires_input_bounds:
                print('  ', node)
        breakpoint()


def check_auto_enlarge_batch_size(auto_batch_size):
    ret = auto_batch_size.update()
    if ret is not None:
        current_vram = ret['current_vram']
        total_vram = ret['total_vram']
        print('current_vram/total_varm: '
            f'{current_vram/1e9:.1f}GB/{total_vram/1e9:.1f}GB, '
            f'batch_size increase to {auto_batch_size.batch_size}')
    return auto_batch_size.batch_size

def _take_batch_Tensor(data: torch.Tensor, batch_size, batch_idx, device=None, batch_dim=0):
    """
    Take a batch of data from a tensor. The returned object is a new tensor.
    """
    idx = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
    index = [slice(None)] * data.ndim
    index[batch_dim] = idx
    batch_data = data[tuple(index)]
    if device is not None:
        batch_data = batch_data.to(device=device)
    return batch_data

def _take_batch_Tensor_BoundedTensor(x: BoundedTensor, batch_size, batch_idx, device=None):
    """
    Take a batch of data from a BoundedTensor. The returned object is a new BoundedTensor.
    """
    batch_data = _take_batch_Tensor(x.data, batch_size, batch_idx)
    ptb = PerturbationLpNorm(
        norm=x.ptb.norm,
        eps=x.ptb.eps,
        x_L=_take_batch_Tensor(x.ptb.x_L, batch_size, batch_idx),
        x_U=_take_batch_Tensor(x.ptb.x_U, batch_size, batch_idx))
    new_x = BoundedTensor(batch_data, ptb)
    if device is not None:
        new_x = new_x.to(device=device)
    return new_x

def take_batch(data, batch_size, batch_idx, device=None, batch_dim=0):
    """
    Take a batch of data from a tensor or BoundedTensor.
    The returned object is a new tensor or BoundedTensor.
    """
    if isinstance(data, BoundedTensor):
        return _take_batch_Tensor_BoundedTensor(data, batch_size, batch_idx, device)
    elif isinstance(data, torch.Tensor):
        return _take_batch_Tensor(data, batch_size, batch_idx, device, batch_dim)
    else:
        raise TypeError(f'Unsupported data type: {type(data)}.')

def _expand_batch(data: torch.Tensor, batch_size, device=None, batch_dim=0):
    """
    Expand a tensor to a specified batch size by repeating the first element.
    """
    # Select first slice along batch_dim
    index = [slice(None)] * data.ndim
    index[batch_dim] = slice(0, 1)
    base = data[tuple(index)]

    # Validate all elements are the same along batch_dim
    if not (data == base).all():
        raise ValueError('All elements along batch_dim must be equal to the first slice.')

    # Expand along batch_dim
    shape = list(base.shape)
    shape[batch_dim] = batch_size
    expanded_data = base.expand(*shape)

    if device is not None:
        expanded_data = expanded_data.to(device)

    return expanded_data

def _expand_batch_BoundedTensor(x: BoundedTensor, batch_size, x_L=None, x_U=None, device=None):
    """
    Expand a BoundedTensor to a specified batch size.
    """
    if x_L is None and x_U is None:
        ptb = PerturbationLpNorm(
            norm=x.ptb.norm,
            eps=x.ptb.eps,
            x_L=_expand_batch(x.ptb.x_L, batch_size),
            x_U=_expand_batch(x.ptb.x_U, batch_size))
    else:
        ptb = PerturbationLpNorm(norm=x.ptb.norm, x_L=x_L, x_U=x_U)
    new_x = BoundedTensor(_expand_batch(x.data, batch_size), ptb)
    if device is not None:
        new_x = new_x.to(device=device)
    return new_x

def expand_batch(data, batch_size, device=None, batch_dim=0, x_L=None, x_U=None):
    """
    Expand a tensor or BoundedTensor to a specified batch size.
    The returned object is a new tensor or BoundedTensor.
    x_L and x_U are optional bounds for BoundedTensor.
    """
    if isinstance(data, BoundedTensor):
        assert batch_dim == 0
        return _expand_batch_BoundedTensor(data, batch_size, x_L, x_U, device)
    elif isinstance(data, torch.Tensor):
        return _expand_batch(data, batch_size, device, batch_dim)
    else:
        raise TypeError(f'Unsupported data type: {type(data)}.')

def unpad_to_list_of_tensors(padded_tensor, unbind_dim, unpad_dim, ori_lengths, keep_dim):
    """
    Unpad a tensor along a specified dimension into a list of tensors.

    Args:
        padded_tensor (torch.Tensor): The padded tensor.
        unbind_dim (int): The dimension to split tensors along.
        unpad_dim (int): The dimension to unpad along.
        ori_lengths (list): List of lengths of original tensors along the unpad_dim.
        keep_dim (bool): Whether to keep the unbind_dim in the output tensors.

    Returns:
        list[torch.Tensor]: List of unpadded tensors.
    """
    # Check input
    ndim = padded_tensor.ndim
    assert 0 <= unpad_dim < ndim, f'Invalid dim {unpad_dim} for tensor with {ndim} dims.'
    assert 0 <= unbind_dim < ndim, f'Invalid dim {unbind_dim} for tensor with {ndim} dims.'

    # Set up slice indices
    slice_indices = [slice(None)] * ndim
    if not keep_dim:
        slice_indices[unbind_dim] = 0

    # Unpad tensors
    unpadded_list = []
    for t, l in zip(padded_tensor.unbind(unbind_dim), ori_lengths):
        slice_indices[unpad_dim] = slice(l)
        unpadded_list.append(t.unsqueeze(unbind_dim)[tuple(slice_indices)])

    return unpadded_list


def pad_list_of_input_to_tensor(
    list_of_input, pad_value=0, pad_dim=0, batch_dim=None, is_orginal_tensor=False, device=None
):
    """
    Pad a list of input arrays/tensors and stack it to a tensor.
    If batch_dim is None, we create a new batch dimension at the front after padding.
    If batch_dim is not None, we concat arrays/tensors along the batch_dim after padding.

    Args:
        list_of_input (list[torch.Tensor]): List of input tensors to pad.
        pad_dim (int): The dimension to pad along.
        batch_dim (int): The dimension to stack tensors along.
        is_orginal_tensor (bool): Whether the input is a list of tensors or arrays.
        device (torch.device): The device to put the padded tensor on.

    Returns:
        torch.Tensor: The padded tensor.
    """
    if device is None:
        device = arguments.Config['general']['device']

    # Convert input to tensors on the correct device if they are not already
    if is_orginal_tensor:
        list_of_input = [item.to(device) for item in list_of_input]
    else:
        list_of_input = [
            torch.as_tensor(item, dtype=torch.get_default_dtype(), device=device)
            for item in list_of_input
        ]

    # Check input
    assert all(
        list_of_input[0].ndim == t.ndim for t in list_of_input
    ), 'All tensors must have the same number of dimensions.'
    assert (
        pad_dim >= 0 and pad_dim <= list_of_input[0].ndim
    ), f'Invalid pad_dim {pad_dim} for tensors with {list_of_input[0].ndim} dims.'
    if batch_dim is not None:
        assert (
            batch_dim >= 0 and batch_dim <= list_of_input[0].ndim
        ), f'Invalid batch_dim {batch_dim} for tensors with {list_of_input[0].ndim} dims.'
    shapes = [t.shape for t in list_of_input]
    for dim in range(list_of_input[0].ndim):
        if dim != pad_dim and dim != batch_dim:
            assert all(
                s[dim] == shapes[0][dim] for s in shapes
            ), 'All tensors must have the same shape except for the padding dimension.'

    # Pad tensors
    max_pad_size = max(t.shape[pad_dim] for t in list_of_input)
    padded_tensors = []
    for t in list_of_input:
        pad_size = max_pad_size - t.shape[pad_dim]
        pad_config = [0, 0] * t.ndim
        pad_config[-(2 * pad_dim + 1)] = pad_size
        padded_tensors.append(F.pad(t, pad_config, value=pad_value))

    # Stack tensors
    if batch_dim is None:
        # Stack in the front
        padded_tensor = torch.stack(padded_tensors, dim=0)
    else:
        # Concatenate in batch_dim
        padded_tensor = torch.cat(padded_tensors, dim=batch_dim)

    return padded_tensor


def transfer_obj(obj, device=None, dtype=None, inplace=False):
    """
    Move all tensors in the object to a specified dest
    (device or dtype). The inplace=True option is available for dict.
    """
    if obj is None:
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device=device if device is not None else obj.device,
                      dtype=dtype if dtype is not None else obj.dtype)
    elif isinstance(obj, tuple):
        return tuple([transfer_obj(item, device=device, dtype=dtype) for item in obj])
    elif isinstance(obj, list):
        return [transfer_obj(item, device=device, dtype=dtype) for item in obj]
    elif isinstance(obj, dict):
        if inplace:
            for k, v in obj.items():
                obj[k] = transfer_obj(v, device=device, dtype=dtype, inplace=True)
            return obj
        else:
            return {k: transfer_obj(v, device=device, dtype=dtype) for k, v in obj.items()}
    else:
        return obj
