import copy
import time
import tqdm
import math
import numpy as np

import torch
import torch.nn as nn

import arguments
from collections import deque
from heuristics.base import NeuronBranchingHeuristic
from auto_LiRPA.backward_bound import (
    get_degrees, add_constant_node, add_bound)
from auto_LiRPA import BoundedTensor
from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import stop_criterion_batch_any, multi_spec_keep_func_all
from auto_LiRPA.utils import prod


def precompute_A(net, A, x, interm_bounds):
    # TODO maybe some can be pruned
    need_A = set()
    for nodes in net.split_activations.values():
        for node in nodes:
            for inp in node[0].inputs:
                if inp.name not in net.root_names and inp.perturbed:
                    need_A.add(inp)
    for node in need_A:
        if node.name not in A:
            print(f'Missing A for {node}. Making an additional CROWN call.')
            dim_output = int(prod(node.output_shape[1:]))
            C = torch.eye(dim_output, device=net.device).unsqueeze(0)
            ret = net.compute_bounds(
                x=(x,), C=C, method='CROWN', final_node_name=node.name,
                return_A=True,
                needed_A_dict={node.name: [net.input_name[0]]},
                interm_bounds=interm_bounds)
            A.update(ret[-1])


class NonlinearBranching(NeuronBranchingHeuristic):
    """A general branching heuristic for nonlinear functions.

    TODO Scores are computed by calling the heuristic score function of the
    corresponding operator. We also want to implement optimizable branching
    point not limited to 0.
    """

    def __init__(self, net, **kwargs):
        super().__init__(net)
        self.input_split_method = 'sb'
        self.branching_point_method = kwargs.pop('branching_point_method')
        self.branching_point_refinement = kwargs.pop('branching_point_refinement')
        self.num_branches = kwargs.pop('num_branches')
        self.method = kwargs.pop('method')
        self.filter = kwargs.pop('filter')
        self.filter_beta = kwargs.pop('filter_beta')
        self.filter_batch_size = kwargs.pop('filter_batch_size')
        self.filter_iterations = kwargs.pop('filter_iterations')
        self.shortlist_size = kwargs.pop('shortlist_size')
        self.root_name = self.net.net.root_names[0]

        if self.branching_point_method.endswith('.pt'):
            self.branching_point_db = torch.load(self.branching_point_method)

    def _get_uniform_branching_points(self, lb, ub):
        ratio = torch.arange(0, 1, step=1./self.num_branches)[1:].to(lb)
        assert ratio.shape[-1] == self.num_branches - 1
        points = lb.unsqueeze(-1) * (1 - ratio) + ub.unsqueeze(-1) * ratio
        return points

    def _get_input_split_scores(self, domains):
        print('Prioritizing input split for this round.')
        lb = domains['lower_bounds'][self.root_name]
        ub = domains['upper_bounds'][self.root_name]
        lA = domains['lAs'][self.root_name]
        if self.input_split_method == 'naive':
            scores = (ub - lb).flatten(1)
        elif self.input_split_method == 'sb':
            # TODO clampping lA?
            scores = (lA.abs() * (ub - lb).unsqueeze(1)).amax(dim=1)
        else:
            raise ValueError(self.input_split_method)
        scores = {self.root_name: scores}
        points = {
            self.root_name: self._get_uniform_branching_points(
                lb, ub).flatten(1, -2)
        }
        return scores, points

    def get_branching_decisions(self, domains, split_depth=1,
                                branching_candidates=1, verbose=False,
                                **kwargs):
        lb, ub = domains['lower_bounds'], domains['upper_bounds']
        split_masks = domains['mask']
        self.update_batch_size_and_device(domains['lower_bounds'])

        if getattr(self.net, 'new_input_split_now', False):
            # Shortcut for the new input split
            scores, points = self._get_input_split_scores(domains)
        else:
            scores = {}
            points = {}
            points_mask = {}
            for node in self.net.split_nodes:
                name = node.name
                if verbose:
                    print(f'Computing branching score for {name}')
                ret = self.compute_branching_scores(
                    node, lb=lb, ub=ub, domains=domains)
                scores[name] = ret['scores'].flatten(1) * split_masks[node.name]
                scores[name] += split_masks[name] * 1e-10
                points[name] = ret['points'].flatten(1, -2)
                if ret['points_mask'] is not None:
                    points_mask[name] = ret['points_mask'].flatten(1, -2)
                    assert split_depth == 1
                else:
                    points_mask[name] = None

        if self.filter and split_depth == 1:
            if hasattr(self.net, 'new_input_split_now'):
                use_filter = (not self.net.new_input_split_now
                            and getattr(self.net, 'new_input_split_filter', False))
            else:
                use_filter = True
        else:
            use_filter = False

        if use_filter:
            topk = split_depth if split_depth != 1 else branching_candidates
        else:
            topk = split_depth
        layers, indices, scores = self.find_topk_scores(
            scores, split_masks, topk, return_scores=True)
        num_branching_points = self.num_branches - 1
        points_ret = torch.full(
            (layers.shape[0], layers.shape[1], num_branching_points), -np.inf,
            device=scores.device)
        points_mask_ret = torch.full(
            (layers.shape[0], layers.shape[1], num_branching_points), 0,
            dtype=torch.bool, device=scores.device)
        for idx, layer in enumerate(self.net.split_nodes):
            mask = layers.view(-1) == idx
            if mask.sum() == 0:
                continue
            name = layer.name
            indices_ = indices.clamp(max=points[name].shape[1]-1)
            points_ = torch.gather(
                points[name], dim=1,
                index=indices_.unsqueeze(-1).repeat(
                    1, 1, num_branching_points))
            points_ret.view(-1, num_branching_points)[
                mask, :] = points_.view(-1, num_branching_points)[mask, :]
            if points_mask[name] is None:
                points_mask_ret = None
            else:
                points_mask_ = torch.gather(
                    points_mask[name], dim=1,
                    index=indices_.unsqueeze(-1).repeat(
                        1, 1, num_branching_points))
                points_mask_ret.view(-1, num_branching_points)[
                    mask, :] = points_mask_.view(-1, num_branching_points)[mask, :]

        points = points_ret
        points_mask = points_mask_ret

        if use_filter:
            layers, indices, points = self._filter(
                domains, lb, ub, layers, indices, points, branching_candidates)

        return self.format_decisions(layers, indices, points, points_mask)

    def _filter(self, domains, lb, ub, layers, indices, points,
                branching_candidates):
        decisions = self.format_decisions(layers, indices, points)
        args_update_bounds = {
            'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': domains['alphas'], 'cs': domains['cs'],
            'thresholds': domains.get('thresholds', None)
        }
        if self.filter_beta:
            args_update_bounds.update({
                'betas': domains['betas'],
                'history': domains['history']
            })
        filter_start_time = time.time()
        print('Start filtering...')
        branching_decision, branching_points, branching_points_mask, _ = decisions
        if branching_points_mask is not None:
            raise NotImplementedEror
        split = {
            'decision': branching_decision,
            'points': branching_points
        }

        self.net.build_history_and_set_bounds(
            args_update_bounds, split, mode='breath')

        total_candidates = args_update_bounds['cs'].shape[0]
        num_batches = (
            total_candidates + self.filter_batch_size - 1
        ) // self.filter_batch_size
        ret_lbs = []
        iterations = arguments.Config['solver']['beta-crown']['iteration']
        arguments.Config['solver']['beta-crown']['iteration'] = self.filter_iterations
        for i in tqdm.tqdm(range(num_batches)):
            args_update_bounds_ = {
                'lower_bounds': {
                    k: v[i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                    for k, v in args_update_bounds['lower_bounds'].items()},
                'upper_bounds': {
                    k: v[i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                    for k, v in args_update_bounds['upper_bounds'].items()}
            }
            for k in ['cs', 'thresholds']:
                args_update_bounds_[k] = args_update_bounds[k][
                    i*self.filter_batch_size:(i+1)*self.filter_batch_size]
            if self.filter_beta:
                for k in ['betas', 'history']:
                    args_update_bounds_[k] = (args_update_bounds[k][ # copy.deepcopy
                        i*self.filter_batch_size:(i+1)*self.filter_batch_size])
            args_update_bounds_['alphas'] = {
                k: {kk: vv[:, :, i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                    for kk, vv in v.items()}
                for k, v in args_update_bounds['alphas'].items()
            }
            ret_lbs_ = self.net.update_bounds(
                copy.deepcopy(args_update_bounds_),
                shortcut=True, beta=self.filter_beta, beta_bias=True,
                stop_criterion_func=stop_criterion_batch_any(args_update_bounds_['thresholds']),
                multi_spec_keep_func=multi_spec_keep_func_all)
            ret_lbs.append(ret_lbs_.detach())
        arguments.Config['solver']['beta-crown']['iteration'] = iterations

        ret_lbs = torch.concat(ret_lbs, dim=0)
        ret_lbs = ret_lbs.reshape(
            branching_candidates, self.num_branches, -1, ret_lbs.shape[-1])

        kfsb_scores = (ret_lbs-domains['thresholds']).amax(dim=-1).min(dim=1).values
        kfsb_choice = kfsb_scores.argmax(dim=0)
        print('kfsb choice:', kfsb_choice)
        layers = torch.gather(layers, index=kfsb_choice.unsqueeze(-1),dim=1)
        indices = torch.gather(indices, index=kfsb_choice.unsqueeze(-1),dim=1)
        points = torch.gather(
            points, index=kfsb_choice.unsqueeze(-1).unsqueeze(-1),dim=1)
        print('Filtering time:', time.time() - filter_start_time)
        lb_old = domains['lower_bounds'][self.net.final_name]
        previous_best = (lb_old-domains['thresholds']).amax(dim=-1).max()
        previous_worst = (lb_old-domains['thresholds']).amax(dim=-1).min()
        ret_selected = (ret_lbs-domains['thresholds']).amax(dim=-1).min(dim=1).values.max(dim=0).values
        new_worst = ret_selected.min()
        print('Previous best:', previous_best)
        print('Previous worst:', previous_worst)
        print('New worst:', new_worst)
        return layers, indices, points

    def compute_branching_scores(self, node, lb, ub, domains):
        name = node.name
        lb_ori = lb[name]
        ub_ori = ub[name]
        lAs = domains['lAs']

        points_uniform = self._get_uniform_branching_points(lb_ori, ub_ori)
        if self.branching_point_method == 'uniform':
            points = points_uniform
            points_mask = None
        else:
            assert len(self.net.split_activations[name]) == 1
            points = self.branching_point_db['points'].detach()
            points_mask = self.branching_point_db['points_mask'].detach()
            range_l = self.branching_point_db['range_l']
            range_u = self.branching_point_db['range_u']
            step_size = self.branching_point_db['step_size']
            assert self.num_branches == points.shape[-1] + 1
            valid = torch.logical_and(lb_ori >= range_l, ub_ori <= range_u)
            index_l = torch.ceil((lb_ori - range_l) / step_size).to(torch.long)
            index_u = torch.floor((ub_ori - range_l) / step_size).to(torch.long)
            index = index_l * self.branching_point_db['num_samples'] + index_u
            index = torch.where(valid, index, 0)
            points = points[index]
            points_mask = points_mask[index]
            if not valid.all():
                invalid = valid.logical_not()
                points[invalid] = points_uniform[invalid]
                points_mask[invalid] = 1
            print(f'Using non-trivial branching points: valid {valid.sum()}/{valid.numel()}')
            for i in range(points_mask.shape[-1]):
                print(f'  {i+2} branches: {(points_mask.sum(dim=-1) == i+1).sum()}')

        for n in self.net.net.nodes():
            if hasattr(n, 'opt_stage'):
                n.opt_stage = None

        if (name == self.root_name
                and not getattr(self.net, 'new_input_split_now', False)):
            return {
                'scores': torch.zeros_like(lb_ori),
                'points': points,
                'points_mask': points_mask,
            }

        start_nodes = [act[0] for act in self.net.split_activations[name]]

        # Specicial cases for now
        if len(start_nodes) == 1:
            if isinstance(start_nodes[0], (BoundRelu, BoundSign, BoundSignMerge)):
                # For ReLU or LeakyReLU, always branch at 0.
                mask_unstable = torch.logical_and(lb_ori < 0, ub_ori > 0)
                points[mask_unstable, :] = 0
                # TODO set points_mask

        global_lb = domains['lower_bounds'][self.net.final_name]
        margin_before = global_lb - domains['thresholds']

        if self.method != 'babsr-like':
            margin_after = self._fast_heuristic(
                node, start_nodes, lAs, lb, ub, lb_ori, ub_ori,
                points, points_mask, domains)
            scores = (margin_after - margin_before.unsqueeze(1)).sum(dim=-1)
            return {'scores': scores, 'points': points, 'points_mask': points_mask}

        assert points_mask is None

        margin_before = margin_before.amax(dim=-1)

        batch_size = lb_ori.shape[0]
        num_neurons = np.prod(points.shape[1:-1])

        bounds_before = self.get_partial_bounds_batch(lb, ub, lAs, start_nodes)
        bounds_before = bounds_before.reshape(batch_size, -1)

        if num_neurons > self.shortlist_size:
            # TODO try multiplying with lA
            branched_neurons = torch.argsort(
                (ub[name] - lb[name]).view(batch_size, num_neurons),
                dim=-1, descending=True)[:, :self.shortlist_size]
        else:
            branched_neurons = torch.arange(
                num_neurons, device=self.device).expand(batch_size, num_neurons)

        def _repeat(v):
            return v.repeat(num_neurons, *([1]*(v.ndim-1)))

        lAs = {k: _repeat(v) for k, v in lAs.items()
                if self.net.net[k] in start_nodes}
        lb = {k: _repeat(v) for k, v in lb.items()
            if k != name and k in self.interm_bound_required}
        ub = {k: _repeat(v) for k, v in ub.items()
            if k != name and k in self.interm_bound_required}
        margin_after = None
        for i in range(self.num_branches):
            lb_branched = lb_ori if i == 0 else points[..., i - 1]
            ub_branched = ub_ori if i == self.num_branches - 1 else points[..., i]
            bounds_after = self._get_bounds_after_branching(
                branched_neurons, lAs, lb, ub, lb_ori, ub_ori,
                lb_branched, ub_branched, name, start_nodes)
            margin_after_ = (
                (global_lb - domains['thresholds']).unsqueeze(1)
                + (bounds_after - bounds_before.unsqueeze(1))).amax(dim=-1)
            if i == 0:
                margin_after = margin_after_
            else:
                margin_after = torch.min(margin_after, margin_after_)

        # bounds_after: [batch_size, num_neurons, -1]
        # margin_after is only for the shortlisted neurons.
        # We then construct margin_after_all for all the neurons.
        margin_after_all = torch.empty(
            batch_size, num_neurons, device=margin_after.device)
        if margin_after_all.shape == margin_after.shape:
            margin_after_all = margin_after
        else:
            margin_after_all.scatter_(
                src=margin_after, dim=-1, index=branched_neurons)

        return {
            'scores': margin_after_all - margin_before.unsqueeze(1),
            'points': points
        }

    def _fast_heuristic(self, node, start_nodes, lAs, lb, ub, lb_ori, ub_ori,
                        points, points_mask, domains):
        model = self.net.net
        name = node.name
        global_lb = domains['lower_bounds'][self.net.final_name]

        A_before, bound_before, unstable_idx = self._fast_backward_propagation(
            lAs, lb, ub, node, start_nodes)
        dim_output, batch_size, num_neurons = bound_before.shape
        roots = model.roots()
        assert isinstance(roots[0], BoundInput)
        assert isinstance(roots[0].value, BoundedTensor)
        x_new = self.net.expand_x(batch_size)
        roots[0].center = x_new
        roots[0].perturbation = x_new.ptb
        roots[0].aux = None
        roots[0].uA = None
        for r in roots[1:]:
            assert isinstance(r, (BoundConstant,
                                    BoundParams, BoundBuffers))
            if isinstance(r, BoundParams):
                assert isinstance(r.param, nn.Parameter)
                r.center = r.param
            r.perturbation = None
            r.lA = r.uA = None
        # (dim_output, batch_size, dim_input)
        roots[0].lA = A_before.sum(dim=2)
        # (batch_size, dim_output)
        bound_from_A = model.concretize(
            batch_size, dim_output,
            torch.zeros((batch_size, dim_output), device=bound_before.device),
            bound_upper=False)[0]
        # (batch_size, num_neurons, dim_output)
        bound_before = bound_from_A.unsqueeze(1) + bound_before.permute(1, 2, 0)

        margin_afters = []
        if points_mask is not None:
            num_branches = points_mask.sum(dim=-1) + 1
        for i in range(self.num_branches):
            lb_branched = lb_ori if i == 0 else points[..., i - 1]
            if i + 1 == self.num_branches:
                ub_branched = ub_ori
            elif points_mask is None:
                ub_branched = points[..., i]
            else:
                ub_branched = torch.where(num_branches - 1 == i,
                                          ub_ori, points[..., i])

            lb_ = {k: lb_branched if k == name else v
                   for k, v in lb.items()}
            ub_ = {k: ub_branched if k == name else v
                   for k, v in ub.items()}

            A_after, bound_after, _ = self._fast_backward_propagation(
                lAs, lb_, ub_, node, start_nodes)
            # A_before: (dim_output, batch_size, num_neurons, dim_input)
            diff_A = A_after - A_before
            A_ = A_before.sum(dim=2, keepdim=True) + diff_A
            # (dim_output * num_neurons, batch_size, dim_input)
            roots[0].lA = A_.transpose(1, 2).reshape(
                -1, batch_size, A_after.shape[-1])
            # (batch_size, dim_output * num_neurons)
            bound_after = bound_after.transpose(
                1, 2).reshape(-1, batch_size).transpose(0, 1)
            # (batch_size, dim_output, num_neurons)
            bound_after = model.concretize(
                bound_after.shape[0], bound_after.shape[1],
                bound_after, bound_upper=False)[0]
            # (batch_size, num_neurons, dim_output)
            bound_after = bound_after.reshape(
                batch_size, dim_output, num_neurons).transpose(1, 2)
            # (batch_size, dim_output, num_neurons)
            bound_delta = bound_after - bound_before
            margin_after_ = (
                (global_lb - domains['thresholds']).unsqueeze(1)
                + bound_delta)
            margin_afters.append(margin_after_)

        margin_after = torch.concat(margin_afters).reshape(
            -1, *margin_afters[0].shape)
        if points_mask is not None:
            margin_after[1:][
                points_mask.permute(2, 0, 1).logical_not()] = torch.inf
        margin_after = margin_after.min(dim=0).values

        if unstable_idx is not None:
            margin_full = torch.zeros(
                margin_after.shape[0], *node.output_shape[1:],
                margin_after.shape[-1], device=margin_after.device)
            if isinstance(unstable_idx, torch.Tensor):
                margin_full[:, unstable_idx, :] = margin_after
            else:
                margin_full[:, unstable_idx[0],
                            unstable_idx[1], unstable_idx[2], :] = margin_after
            return margin_full.reshape(
                margin_full.shape[0], -1,  margin_full.shape[-1])

        return margin_after

    def _fast_backward_propagation(self, lAs, lb, ub,
                                   branched_node, start_nodes):
        model = self.net.net
        A_root = None
        bound = None
        unstable_idx = None
        for node in start_nodes:
            lA = lAs[node.name].transpose(0, 1)
            for i in node.requires_input_bounds:
                inp = node.inputs[i]
                inp.lower, inp.upper = lb[inp.name], ub[inp.name]
            if not isinstance(node, (BoundActivation,
                                     BoundOptimizableActivation)):
                raise NotImplementedError
            A, lower_b, _ = node.bound_backward(
                lA, None, *node.inputs,
                start_node=model[model.final_name], reduce_bias=False)
            for i, node_pre in enumerate(node.inputs):
                if node_pre != branched_node:
                    continue
                bound_ = lower_b[i] if isinstance(lower_b, tuple) else lower_b
                if bound_.shape[2:] != branched_node.output_shape[1:]:
                    print('Error: incorrect shapes in the branching heuristic.')
                    print('It may be because that _fast_backward_propagation has '
                          f'not been supported for {node} yet')
                    print('Please debug:')
                    import pdb; pdb.set_trace()

                def maybe_convert_A(A):
                    if isinstance(A, torch.Tensor):
                        return A
                    else:
                        return A.to_matrix(model.roots()[0].output_shape)

                A_saved = self.net.A_saved[node_pre.name][model.input_name[0]]
                lA_next = maybe_convert_A(A_saved['lA'])
                assert lA_next.shape[0] == 1
                lA_next = lA_next.reshape(lA_next.shape[1], -1)
                uA_next = maybe_convert_A(A_saved['uA'])
                assert uA_next.shape[0] == 1
                uA_next = uA_next.reshape(uA_next.shape[1], -1)
                lbias = A_saved['lbias'].flatten()
                ubias = A_saved['ubias'].flatten()
                A_ = A[i][0]

                if A_saved['unstable_idx'] is not None:
                    assert unstable_idx is None
                    unstable_idx = A_saved['unstable_idx']
                    if isinstance(A_saved['unstable_idx'], torch.Tensor):
                        A_ = A_[:, :, A_saved['unstable_idx']]
                        assert A_saved['unstable_idx'].max()<bound_.shape[2]
                        bound_ = bound_[:, :, A_saved['unstable_idx']]
                    else:
                        A_ = A_[:, :,
                                A_saved['unstable_idx'][0],
                                A_saved['unstable_idx'][1],
                                A_saved['unstable_idx'][2]]
                        assert A_saved['unstable_idx'][0].max()<bound_.shape[2]
                        assert A_saved['unstable_idx'][1].max()<bound_.shape[3]
                        assert A_saved['unstable_idx'][2].max()<bound_.shape[4]
                        bound_ = bound_[:, :, A_saved['unstable_idx'][0],
                                        A_saved['unstable_idx'][1],
                                        A_saved['unstable_idx'][2]]

                A_ = A_.reshape(A_.shape[0], A_.shape[1], -1)
                bound_ = bound_.reshape(bound_.shape[0], bound_.shape[1], -1)
                if lbias is not None:
                    bound_  = bound_ + (A_.clamp(min=0) * lbias
                                        + A_.clamp(max=0) * ubias)

                A_ = A_.unsqueeze(-1)
                A_root_ = (A_.clamp(min=0) * lA_next + A_.clamp(max=0) * uA_next)
                if A_root is None:
                    A_root = A_root_
                else:
                    assert A_root.shape == A_root_.shape
                    A_root += A_root_

                if bound is None:
                    bound = bound_
                else:
                    bound += bound_

        if bound.ndim < 3:
            import pdb; pdb.set_trace()
        return A_root, bound, unstable_idx

    def _get_bounds_after_branching(self, branched_neurons, lAs, lb, ub,
                                    lb_ori, ub_ori, lb_branched, ub_branched,
                                    node_name, start_nodes):
        batch_size, num_selected_neurons = branched_neurons.shape
        bounds_after = []

        def _branch(ori, branched, index):
            num_selected_neurons = index.shape[1]
            repeated = ori.repeat(
                num_selected_neurons, *([1]*(ori.ndim - 1))
            ).reshape(num_selected_neurons, *ori.shape)
            repeated = repeated.transpose(0, 1).reshape(
                batch_size, num_selected_neurons, -1)
            branched_ = torch.gather(branched.view(batch_size, -1),
                                     index=index, dim=-1)
            repeated.scatter_(dim=2, index=index.unsqueeze(-1),
                              src=branched_.unsqueeze(-1))
            return repeated.transpose(0, 1).reshape(-1, *branched.shape[1:])

        branched_neurons_batch = branched_neurons
        this_batch_size = branched_neurons_batch.numel()
        lAs_ = {k: v for k, v in lAs.items()}
        lb_ = {k: v for k, v in lb.items()}
        ub_ = {k: v for k, v in ub.items()}
        lb_[node_name] = _branch(lb_ori, lb_branched, branched_neurons_batch)
        ub_[node_name] = _branch(ub_ori, ub_branched, branched_neurons_batch)

        bounds_after.append(self.get_partial_bounds_batch(
            lb_, ub_, lAs_, start_nodes))
        bounds_after = torch.concat(bounds_after, dim=0)
        bounds_after = bounds_after.view(num_selected_neurons, batch_size, -1)
        bounds_after = bounds_after.transpose(0, 1)

        return bounds_after

    @torch.no_grad()
    def get_partial_bounds_batch(self, lb, ub, lAs, start_nodes):
        self.interm_bound_required = set()
        model = self.net.net
        roots = model.roots()
        batch_size = next(iter(lb.values())).shape[0]
        x = self.net.expand_x(batch_size, lb=lb, ub=ub)
        interm_bounds = {k: (lb[k], ub[k]) for k in lb}
        model.set_input(x, interm_bounds=interm_bounds)
        for node in model.nodes():
            if not node.perturbed:
                fv = model.get_forward_value(node)
                node.interval = node.lower, node.upper = fv, fv

        for node in model.nodes():
            node.lA = node.uA = None
        start_nodes = list(set(start_nodes))
        degree_out = get_degrees(start_nodes)
        for node in start_nodes:
            node.lA = lAs[node.name].transpose(0, 1)
        for k in lb:
            model[k].lower = lb[k]
            model[k].upper = ub[k]
        lb = ub = torch.tensor(0., device=self.device)

        queue = deque(list(set(start_nodes)))
        while len(queue) > 0:
            node = queue.popleft()  # backward from l
            if node.name in model.root_names: continue

            # if all the succeeds are done, then we can turn to this node in the
            # next iteration.
            for node_pre in node.inputs:
                degree_out[node_pre.name] -= 1
                if degree_out[node_pre.name] == 0:
                    queue.append(node_pre)

            # Initially, l.lA or l.uA will be set to C for this node.
            if node.lA is not None or node.uA is not None:
                if not node.perturbed:
                    if not hasattr(node, 'forward_value'):
                        model.get_forward_value(node)
                    lb, ub = add_constant_node(lb, ub, node)
                    continue
                if node.zero_uA_mtx and node.zero_lA_mtx:
                    # A matrices are all zero, no need to propagate.
                    continue
                lb = lb + self._backward_propagate(node, start_nodes)

        assert lb.ndim == 2
        lb = lb.transpose(0, 1)

        if self.method != 'babsr-like':
            # Take bounds from the concretization which is important.
            assert isinstance(roots[0], BoundInput)
            assert isinstance(roots[0].value, BoundedTensor)
            roots[0].center = roots[0].value
            roots[0].perturbation = roots[0].value.ptb
            roots[0].aux = None
            for r in roots[1:]:
                assert isinstance(r, (BoundConstant,
                                      BoundParams, BoundBuffers))
                if isinstance(r, BoundParams):
                    assert isinstance(r.param, nn.Parameter)
                    r.center = r.param
                r.perturbation = None
            tmp = model.concretize(
                lb.shape[0], lb.shape[1],
                lb, None, bound_lower=True, bound_upper=False)
            assert tmp[0].shape == lb.shape
            lb = tmp[0]

        return lb

    def _backward_propagate(self, node, start_nodes):
        model = self.net.net
        roots = model.roots()

        for i in node.requires_input_bounds:
            self.interm_bound_required.add(node.inputs[i].name)
        A, lower_b, _ = node.bound_backward(
            node.lA, None, *node.inputs,
            start_node=model[model.final_name], start_shape=1)
        lb = lower_b

        if self.method == 'babsr-like':
            if isinstance(lower_b, torch.Tensor) and node not in start_nodes:
                # For the BaBSR-like heuristic, stop further back
                # propagating, once we get some bias term from the layer
                # before the activation (typically a linear layer).
                return lb
            else:
                for i, node_pre in enumerate(node.inputs):
                    add_bound(node, node_pre, lA=A[i][0])
        else:
            for i, node_pre in enumerate(node.inputs):
                if (self.method == 'shortcut'
                        and node_pre.name in self.net.A_saved):
                    # Take the shortcut
                    lA_, lb_ = self._shortcut_A(
                        A[i][0],
                        self.net.A_saved[node_pre.name][model.input_name[0]])
                    lA_ = lA_.view(lA_.shape[0], lA_.shape[1],
                                    *roots[0].value.shape[1:])
                    add_bound(node, roots[0], lA=lA_)
                    lb = lb + lb_
                else:
                    if (node_pre.name not in model.root_names
                            and node_pre.perturbed):
                        print(f'Warning: Missing A for {node_pre}.')
                    add_bound(node, node_pre, lA=A[i][0])

        return lb

    def _shortcut_A(self, A, A_saved):
        A = A.reshape(A.shape[0], A.shape[1], -1)
        assert A_saved['lA'].shape[0] == 1
        if A_saved['lbias'] is None:
            assert A_saved['ubias'] is None
            bound = 0.
        else:
            bound = (A.clamp(min=0) * A_saved['lbias']
                    + A.clamp(max=0) * A_saved['ubias']).sum(dim=2)
        lA_next = A_saved['lA'].reshape(A.shape[-1], -1)
        uA_next = A_saved['uA'].reshape(A.shape[-1], -1)
        A_next = (A.clamp(min=0).matmul(lA_next)
                    + A.clamp(max=0).matmul(uA_next))
        return A_next, bound
