"""Optimized branching points."""

import time
import os
import torch
from auto_LiRPA.bound_ops import *


class BranchingPointOpt:
    version = 'v1'

    def __init__(self, net, **kwargs):
        self.net = net
        self.db_path = kwargs.pop('db_path')
        self.num_iterations = kwargs.pop('num_iterations')
        self.range_l = kwargs.pop('range_l')
        self.range_u = kwargs.pop('range_u')
        self.step_size_1d = kwargs.pop('step_size_1d')
        self.step_size = kwargs.pop('step_size')
        self.batch_size = kwargs.pop('batch_size')
        self.log_interval = kwargs.pop('log_interval')
        self.device = net.net.device
        if os.path.exists(self.db_path):
            self.db = torch.load(self.db_path, map_location='cpu')
            assert isinstance(self.db, dict)
            if self.db['version'] != self.version:
                print(f'Warning: Version of the lookup table at {self.db_path} '
                      f'does not match the expected version {self.version}.')
                self.db['tables'] = []
        else:
            self.db = {'version': self.version, 'tables': []}

    def _get_architecture(self, node):
        """Get the local architecture around node."""
        arch = []
        output_nodes = []
        for output_name in node.output_name:
            node_output = self.net.net[output_name]
            if not isinstance(node_output, BoundActivation):
                continue
            output_nodes.append(node_output)
            index = None
            for i, inp in enumerate(node_output.inputs):
                if inp == node:
                    index = i
                    break
            assert index is not None
            arch.append({
                'op': type(node_output).__name__,
                'num_inputs': len(node_output.inputs),
                'index': index,
                'range_l': node_output.range_l,
                'range_u': node_output.range_u,
            })
        return arch, output_nodes

    def _get_lookup_table(self, arch):
        for db_item in self.db['tables']:
            if db_item['arch'] == arch:
                return db_item
        # The architecture does not exist in the lookup table,
        # and thus we need an optimization now.
        lookup_table = self._create_lookup_table(arch)
        # Save the new lookup table to the database.
        self.db['tables'].append(lookup_table)
        torch.save(self.db, self.db_path)
        return lookup_table

    def _create_lookup_table(self, arch):
        print('Creating a new lookup table for the architecture:')
        print(arch)
        start_time = time.time()
        num_inputs = 1
        for item in arch:
            num_inputs += item['num_inputs'] - 1
        step_size = self.step_size_1d if num_inputs == 1 else self.step_size

        # It's assuming that all the inputs have the same range_l/range_u for now
        range_l = self.range_l
        range_u = self.range_u
        for item in arch:
            range_l = max(range_l, item['range_l'])
            range_u = min(range_u, item['range_u'])

        sample = torch.arange(
            range_l, range_u, step_size,
            device=self.device)
        bounds = torch.meshgrid(*([sample] * 2 * num_inputs))
        lower, upper = [], []
        for i in range(num_inputs):
            lower.append(bounds[i * 2].reshape(-1))
            upper.append(bounds[i * 2 + 1].reshape(-1))
        mask = lower[0] <= upper[0]
        for i in range(1, len(lower)):
            mask = mask & (lower[i] <= upper[i])
        results = {
            'arch': arch,
            'range_l': range_l,
            'range_u': range_u,
            'step_size': step_size,
            'lower': lower,
            'upper': upper,
            'num_samples': len(sample),
        }
        results['opt'] = {}
        ret = []
        num_batches = (lower[0].shape[0] + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            print(f'Batch {i+1}/{num_batches}')
            ret.append(self._optimize_points(
                arch,
                [l[i*self.batch_size:(i+1)*self.batch_size] for l in lower],
                [u[i*self.batch_size:(i+1)*self.batch_size] for u in upper],
                mask[i*self.batch_size:(i+1)*self.batch_size]))
        results['points'] = torch.concat([item['points'] for item in ret], dim=0)
        results['loss'] = torch.concat([item['loss'] for item in ret], dim=0)
        print(f'Time spent on creating the lookup table: {time.time() - start_time}s')
        return results

    def _optimize_points(self, arch, lower, upper, mask):
        nodes = [eval(item['op'])(attr={'device': self.device}).to(self.device)
                 for item in arch]

        loss_best = None
        points_best = None
        loss_best_all, points_best_all = [], []

        for t in range(self.num_iterations):
            ratio = t / self.num_iterations
            points = (upper[0] * ratio + lower[0] * (1 - ratio)).unsqueeze(-1)

            lower_branched = [
                torch.concat([lb.unsqueeze(-1), points], dim=-1).view(-1)
                if i == 0 else lb.repeat(2)
                for i, lb in enumerate(lower)
            ]
            upper_branched = [
                torch.concat([points, ub.unsqueeze(-1)], dim=-1).view(-1)
                if i == 0 else ub.repeat(2)
                for i, ub in enumerate(upper)
            ]

            loss = self._get_loss(arch, nodes, lower_branched, upper_branched, mask)

            if loss_best is None:
                loss_best = loss.detach().clone()
                points_best = points.detach().clone()
            else:
                mask_improved = loss < loss_best
                points_best[mask_improved] = points[mask_improved].detach()
                loss_best[mask_improved] = loss[mask_improved].detach()

            if (t + 1) % self.log_interval == 0:
                print(f'Iteration {t + 1}: '
                    f'loss {(loss[mask].sum() / mask.int().sum()).item()}, '
                    f'loss_best {(loss_best[mask].sum() / mask.int().sum()).item()}')

        loss_best_all.append(loss_best.unsqueeze(-1))
        points_best_all.append(points_best)

        loss_best_all = torch.concat(loss_best_all, dim=1)
        points_best_all = torch.concat(points_best_all, dim=1)

        return {
            'loss': loss_best_all,
            'points': points_best_all,
        }

    def _get_loss(self, arch, nodes, lower_branched, upper_branched, mask):
        inputs_all = []
        for i in range(len(lower_branched)):
            inp = SimpleNamespace()
            inp.lower = lower_branched[i]
            inp.upper = upper_branched[i]
            inputs_all.append(inp)

        loss = 0
        idx_others = 1
        for t, node in enumerate(nodes):
            num_inputs = arch[t]['num_inputs']
            inputs = []
            for i in range(num_inputs):
                if i == arch[t]['index']:
                    inputs.append(inputs_all[0])
                else:
                    inputs.append(inputs_all[idx_others])
                    idx_others += 1

            node.bound_relax(*inputs, init=True)

            for i in range(num_inputs):
                if isinstance(node.lw, list) and isinstance(node.uw, list):
                    lw, uw = node.lw[i], node.uw[i]
                elif isinstance(node.lw, torch.Tensor) and isinstance(node.uw, torch.Tensor):
                    assert num_inputs == 1
                    lw, uw = node.lw, node.uw
                else:
                    raise NotImplementedError
                loss_ = (uw - lw) * (inputs[i].upper**2 - inputs[i].lower**2) / 2
                for j in range(num_inputs):
                    if j != i:
                        loss_ *= (inputs[j].upper - inputs[j].lower)
                loss += loss_

            loss_ = node.ub - node.lb
            for j in range(num_inputs):
                loss_ *= inputs[j].upper - inputs[j].lower
            loss += loss_

        loss = loss.view(mask.shape[0], -1)
        loss = loss.sum(dim=-1) * mask

        return loss

    def get_branching_points(self, node, lower_bounds, upper_bounds):
        for out in node.output_name:
            if isinstance(self.net.net[out], (BoundMatMul, BoundRelu)):
                # MatMul is not elementwise and is skipped for now
                # ReLU does not need optimized branching points
                return None
        arch, output_nodes = self._get_architecture(node)
        lookup_table = self._get_lookup_table(arch)
        if lookup_table['points'].device.type != self.device:
            lookup_table['points'] = lookup_table['points'].to(self.device)

        # Lower and upper bounds for the input nodes of the nonlinearities
        lbs, ubs = [lower_bounds[node.name]], [upper_bounds[node.name]]
        for i, item in enumerate(arch):
            for j, node_input in enumerate(output_nodes[i].inputs):
                if j != item['index']:
                    lbs.append(lower_bounds[node_input.name])
                    ubs.append(upper_bounds[node_input.name])

        valid = None
        index = 0
        for i in range(len(lbs)):
            if valid is not None and valid.shape != lbs[i].shape:
                assert valid.ndim == lbs[i].ndim
                broadcast = True
                for j in range(valid.ndim):
                    broadcast &= lbs[i].shape[j] == 1 or lbs[i].shape[j] == valid.shape[j]
                if not broadcast:
                    # Don't use optimized branching points if the shapes are not compatible
                    # (e.g., when the node being branched has a smaller shape than other input nodes)
                    return None
                lbs[i] = lbs[i].expand(valid.shape)
                ubs[i] = ubs[i].expand(valid.shape)
            index_l = torch.ceil(
                (lbs[i] - lookup_table['range_l']) / lookup_table['step_size']
            ).to(torch.long)
            index_u = torch.floor(
                (ubs[i] - lookup_table['range_l']) / lookup_table['step_size']
            ).to(torch.long)
            valid_ = ((index_l < index_u)
                      & (lbs[i] >= lookup_table['range_l'])
                      & (ubs[i] <= lookup_table['range_u']))
            if valid is None:
                valid = valid_
            else:
                valid = valid & valid_
            index = index * lookup_table['num_samples'] + index_l
            index = index * lookup_table['num_samples'] + index_u
        index = torch.where(valid, index, 0)
        valid = valid.view(-1)
        index = index.view(-1)
        points = lookup_table['points'][index]
        points_middle = (lower_bounds[node.name] + upper_bounds[node.name]) / 2
        points = torch.where(valid.unsqueeze(-1), points, points_middle.view(points.shape))
        points = points.view(*lower_bounds[node.name].shape, 1)
        return points
