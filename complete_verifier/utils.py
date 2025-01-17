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

import copy
import os
import time
import pickle
import arguments
from dataclasses import dataclass
from collections import defaultdict
from string import Template
import torch


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

    def update_timeout(self, timeout):
        self.timeout_threshold = timeout

    def record_start_time(self):
        self.start_time = time.time()

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
                    arguments.Globals['out']['neurons_visited'] = self.bab_ret[0][2]
                    # self.bab_ret: [[idx, l, nodes, time, init_failure_idx]]
                    # (see abcrown.complete_verifier line 380)

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

            print('############# Summary #############')
            acc = len(time_verified) / self.count * 100.
            print(f'Final verified acc: {acc}% (total {self.count} examples)')
            print('Problem instances count:',
                  len(time_verified) + len(time_unsafe) + len(time_timeout),
                  ', total verified (safe/unsat):', len(time_verified),
                  ', total falsified (unsafe/sat):', len(time_unsafe),
                  ', timeout:', len(time_timeout))
            print('mean time for ALL instances '
                  f'(total {len(time_all_instances)}):'
                  f'{sum(time_all_instances)/(len(time_all_instances) + 1e-5)},'
                  f' max time: {max(time_all_instances)}')
            if len(time_verified) > 0:
                print('mean time for verified SAFE instances'
                      f'(total {len(time_verified)}): '
                      f'{sum(time_verified) / len(time_verified)}, '
                      f'max time: {max(time_verified)}')
            if len(time_verified) > 0 and len(time_unsafe) > 0:
                mean_time = (sum(time_verified) + sum(time_unsafe)) / (
                    len(time_verified) + len(time_unsafe))
                max_time = max(time_verified, time_unsafe)
                print('mean time for verified (SAFE + UNSAFE) instances '
                      f'(total {(len(time_verified) + len(time_unsafe))}):'
                      f' {mean_time}, max time: {max_time}')
            if len(time_verified) > 0 and len(time_timeout) > 0:
                mean_time = (sum(time_verified) + sum(time_timeout)) / (
                    len(time_verified) + len(time_timeout))
                max_time = max(time_verified, time_timeout)
                print('mean time for verified SAFE + TIMEOUT instances '
                      f'(total {(len(time_verified) + len(time_timeout))}):'
                      f' {mean_time}, max time: {max_time} ')
            if len(time_unsafe) > 0:
                print(f'mean time for verified UNSAFE instances '
                      f'(total {len(time_unsafe)}): '
                      f'{sum(time_unsafe) / len(time_unsafe)}, '
                      f'max time: {max(time_unsafe)}')

            for k, v in self.verification_summary.items():
                print(f'{k} (total {len(v)}), index:', v)

            if arguments.Config['general']['save_output']:
                # save output for test
                arguments.Globals['out']['results'] = self.status_per_sample_list[0][0]
                if arguments.Globals['out']['results'] == 'unknown':
                    arguments.Globals['out']['results'] = 'timeout'
                arguments.Globals['out']['time'] = time_all_instances[0]
                if len(self.bab_ret) != 0:
                    arguments.Globals['out']['neurons_visited'] = self.bab_ret[0][2]
                    # self.bab_ret: [[idx, l, nodes, time, init_failure_idx]]
                    # (see abcrown.complete_verifier line 380)

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
        self.implied_cuts = {'statistics': [], 'average_branched_neurons': []}


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

def print_average_branching_neurons(branching_decision, impl_stats, impl_params=None):
    """Print and store the average branched neurons at each iteration."""
    total_branched_neurons = 0

    if impl_params:
        components = impl_params['dependency_components']
        idx_mapping = impl_params['index_mappings']
        for neurons in branching_decision:
            core_idx = idx_mapping[(neurons[0], neurons[1])]
            total_branched_neurons += len(components[core_idx][2])
        average_branched_neurons = total_branched_neurons / len(branching_decision)
    else:
        average_branched_neurons = 1.0

    impl_stats['average_branched_neurons'].append(average_branched_neurons)
    cur_step = len(impl_stats['average_branched_neurons'])
    if impl_params:
        print(f'Average branched neurons at iteration {cur_step}: '
              f'{average_branched_neurons: .4f}')


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
        return 'a-b-crown_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_initial_max_domains={}.npz'.format(  # pylint: disable=line-too-long,consider-using-f-string
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
            arguments.Config['bab']['cut']['cplex_cuts'],
            arguments.Config['bab']['initial_max_domains'])
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
            net.net.set_gcp_relu_indicators(k, torch.stack(masks))
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
