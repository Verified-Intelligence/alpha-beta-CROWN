"""New input split.

This module supports the new input split that can be integrated into the main
branch-and-bound code, compared to the legacy standalone code in `input_split/`.
The input split may be conducted before activation splits. For efficiency,
some alpha and beta features are turned off during the input split but need to
be reinitialized before entering the activation split phase.
"""

import copy

import torch

import arguments
from utils import fast_hist_copy
from branching_domains import BatchedDomainList, check_worst_domain


class NewInputSplit:
    def __init__(self, net, ret, lA, global_lb, global_ub, rhs, alpha):
        assert ret.get('betas', None) is None

        args = arguments.Config['bab']['branching']['new_input_split']
        self.input_split_rounds = args['rounds']
        self.input_split_batch_size = args['batch_size']
        self.init_alpha_batch_size = args['init_alpha_batch_size']
        self.full_alpha = args['full_alpha']
        self.filter = arguments.Config['bab']['branching']['nonlinear_split']['filter']
        self.device = global_lb.device
        self.net = net

        self.domains_input_split = BatchedDomainList(
            ret, lA, global_lb, global_ub,
            {}, None, # Do not set alpha and history
            rhs, net=net)
        # Only create the domain list and empty it
        self.domains_activation_split = BatchedDomainList(
            ret, lA, global_lb, global_ub,
            alpha, copy.deepcopy(ret['history']), rhs, net=net)
        self.domains_activation_split.pick_out(
            batch=global_lb.shape[0], device=net.x.device)

    def __len__(self):
        return len(self.domains_input_split) + len(self.domains_activation_split)

    def sort(self):
        self.domains_input_split.sort()
        self.domains_activation_split.sort()

    def __del__(self):
        del self.domains_input_split
        del self.domains_activation_split

    def get_global_lb(self):
        print('length of input split domains:', len(self.domains_input_split))
        if not self.net.new_input_split_now:
            print('length of activation split domains:',
                  len(self.domains_activation_split))
        global_lb_input = check_worst_domain(
            self.domains_input_split).to(self.device).amax(dim=-1).min()
        global_lb_act = check_worst_domain(
            self.domains_activation_split).to(self.device).amax(dim=-1).min()
        global_lb = torch.min(global_lb_input, global_lb_act)
        print(f'Current in input split (lb-rhs): {global_lb_input}')
        print(f'Current in activation split (lb-rhs): {global_lb_act}')
        print(f'Current (lb-rhs): {global_lb}')
        return global_lb

    def pickout(self, iter_idx, batch):
        if iter_idx <= self.input_split_rounds:
            # Input split phase
            self.net.new_input_split_now = iter_idx
            batch = self.input_split_batch_size
            domains_target = self.domains_input_split # domain list for new domains
            d = self.domains_input_split.pick_out(
                batch=batch, device=self.device)
            set_init_alpha = None
            source = 'input'
        else:
            if self.full_alpha:
                set_init_alpha = None
                if len(self.domains_input_split):
                    init_alpha = True
                    batch = self.init_alpha_batch_size
                    d = self.domains_input_split.pick_out(
                        batch=batch, device=self.device)
                    empty_history = self.net.empty_history()
                    batch = len(d['history'])
                    d['history'] = [fast_hist_copy(empty_history)
                                    for _ in range(batch)]
                    source = 'input'
                else:
                    init_alpha = False
                    d = self.domains_activation_split.pick_out(
                        batch=batch, device=self.device)
                    source = 'act'
            else:
                init_alpha = set_init_alpha = bool(self.net.new_input_split_now)
                if set_init_alpha:
                    batch = self.init_alpha_batch_size
                if len(self.domains_input_split):
                    d = self.domains_input_split.pick_out(
                        batch=batch, device=self.device)
                    source = 'input'
                else:
                    d = self.domains_activation_split.pick_out(
                        batch=batch, device=self.device)
                    source = 'act'
                    if self.filter:
                        self.net.new_input_split_filter = True

            self.net.net.set_bound_opts(
                {'optimize_bound_args': {'init_alpha': init_alpha}})
            self.net.new_input_split_now = False
            domains_target = self.domains_activation_split # domain list for new domains

        return d, domains_target, set_init_alpha, source
