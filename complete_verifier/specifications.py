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
"""Various kinds of specifications for verification."""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from utils import pad_list_of_input_to_tensor, take_batch

class Specification:
    def __init__(self):
        self.num_outputs = arguments.Config['data']['num_outputs']
        # FIXME Do not use numpy. Use torch instead.
        self.rhs = np.array([arguments.Config['bab']['decision_thresh']])

    def construct_vnnlib(self):
        raise NotImplementedError


class SpecificationVerifiedAcc(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            c = (torch.eye(self.num_outputs)[label].unsqueeze(1)
                - torch.eye(self.num_outputs).unsqueeze(0))
            I = (~(label.unsqueeze(1) == torch.arange(
                    self.num_outputs).type_as(label.data).unsqueeze(0)))
            c = c[I].view(1, self.num_outputs - 1, self.num_outputs)
            new_c = []
            for ii in range(self.num_outputs - 1):
                new_c.append((c[:, ii], self.rhs))
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationTarget(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            target_label = dataset['target_label'][example_idx_list[i]]
            c = torch.zeros([1, self.num_outputs])
            c[0, label] = 1
            c[0, target_label] = -1
            new_c = [(c, self.rhs)]
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationRunnerup(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            label = dataset['labels'][example_idx_list[i]].view(1, 1)
            this_x_range = x_range[i]
            runnerup = dataset['runnerup'][example_idx_list[i]]
            c = torch.zeros([1, self.num_outputs])
            c[0, label] = 1
            c[0, runnerup] = -1
            new_c = [(c, self.rhs)]
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


class SpecificationAllPositive(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []
        for i in range(len(example_idx_list)):
            this_x_range = x_range[i]
            c = torch.eye(self.num_outputs).unsqueeze(0)
            new_c = []
            for ii in range(self.num_outputs):
                new_c.append((c[:, ii], self.rhs))
            vnnlib.append([(this_x_range, new_c)])
        return vnnlib


def construct_vnnlib(dataset, example_idx_list):
    X = dataset['X']
    x_lower = x_upper = None
    if arguments.Config['specification']['type'] == 'lp':
        perturb_epsilon = dataset['eps']
        if isinstance(perturb_epsilon, list):
            # Each example has different perturbations.
            perturb_epsilon = torch.cat(perturb_epsilon)
            perturb_epsilon = perturb_epsilon[example_idx_list]
        assert perturb_epsilon is not None
        # FIXME why flatten?
        if arguments.Config['specification']['norm'] == float('inf'):
            if dataset.get('data_max', None) is None:
                # perturb_eps is already normalized.
                x_lower = (X[example_idx_list] - perturb_epsilon).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).flatten(1)
            else:
                x_lower = (X[example_idx_list] - perturb_epsilon).clamp(
                    min=dataset['data_min']).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).clamp(
                    max=dataset['data_max']).flatten(1)
            x_range = torch.stack([x_lower, x_upper], -1).numpy()
        else:
            # TODO create classes to handle it generally
            x_range = []
            for idx in example_idx_list:
                x_item = {
                    # Shape is (batch, ...), where batch = 1.
                    'X': X[idx : idx+1],
                    # eps should be Tensor with 1 element, or float.
                    # If dataset['eps'] is a list, each example has different eps.
                    'eps': (
                        dataset['eps'][idx]
                        if isinstance(dataset['eps'], list)
                        else dataset['eps']
                    ),
                    'norm': dataset['norm'],
                }
                if not isinstance(x_item['X'], torch.Tensor):
                    x_item['X'] = torch.tensor(x_item['X'])
                # data_min, data_max in shape (batch, ...) where batch = 1.
                x_item['data_min'] = x_item['X'] - x_item['eps']
                x_item['data_max'] = x_item['X'] + x_item['eps']
                if dataset.get('data_min', None) is not None:
                    x_item['data_min'] = x_item['data_min'].clamp(
                        min=dataset['data_min'])
                if dataset.get('data_max', None) is not None:
                    x_item['data_max'] = x_item['data_max'].clamp(
                        min=dataset['data_max'])
                x_range.append(x_item)
    elif (arguments.Config['specification']['type'] == 'box' or
            # Some old config files use "bound"; keep for compatibility.
            arguments.Config['specification']['type'] == 'bound'):
        x_lower = dataset['data_min'].flatten(1)
        x_upper = dataset['data_max'].flatten(1)
        x_range = torch.stack([x_lower, x_upper], -1).numpy()
    else:
        raise ValueError('Unsupported perturbation type ' +
                         arguments.Config['specification']['type'])

    # TODO rename "robustness_type", since the verification objective may
    # not be related to robustness.
    robustness_type = arguments.Config['specification']['robustness_type']
    if robustness_type == 'verified-acc':
        specification = SpecificationVerifiedAcc()
    elif robustness_type == 'specify-target':
        specification = SpecificationTarget()
    elif robustness_type == 'runnerup':
        specification = SpecificationRunnerup()
    elif robustness_type == 'all-positive':
        specification = SpecificationAllPositive()
    else:
        raise ValueError(robustness_type)

    return specification.construct_vnnlib(
        dataset, x_range, example_idx_list)

# TODO: This function below should be removed when every component is updated to use vnnlibHandler.
def add_rhs_offset_legacy(
        vnnlib: list,
        rhs_offset: torch.Tensor = None
) -> list:
    """
    Updates the second operand's offset value where rhs_offset is either a scalar that may be
    broadcast to ALL clauses, or rhs_offset is an array of offset values applied to each clause.
    @param vnnlib:      The vnnlib file formatted as a list object. Structure can be found in the
                        read_vnnlib.md.
    @param rhs_offset:  Scalar, array, or None. If array, it modifies the offsets in the clauses 
                        of the vnnlib file accordingly. If scalar, it is broadcast to all clauses.
    @return:            The modified vnnlib object
    """
    # If rhs_offset is None, return the original vnnlib
    if rhs_offset is None:
        return vnnlib

    # FIXME After PR #536, the code was broken when --rhs_offset is provided.
    # Also, the hard-coded delta = 1e-3 should probably be avoided.
    if isinstance(rhs_offset, float):
        is_scalar = True
    else:
        # Determine if rhs_offset is a scalar or array
        is_scalar = isinstance(rhs_offset, float) or rhs_offset.numel() == 1

        delta = 1e-3  # Small value to avoid numerical issues
        if is_scalar:
            rhs_offset = rhs_offset.item() + delta
        else:
            rhs_offset = rhs_offset.view(-1).cpu().numpy() + delta

    updated_vnnlib = []
    k = 0  # Index counter if rhs_offset is an array

    for v in vnnlib:
        result = []
        for i in range(len(v[1])):
            if is_scalar:
                # If scalar, broadcast the same rhs_offset to all clauses
                item = (v[1][i][0], v[1][i][1] + rhs_offset)
            else:
                # If rhs_offset is an array, apply the offset to each clause
                item = (v[1][i][0], v[1][i][1] + rhs_offset[k:k + len(v[1][i][1])])
                k += len(v[1][i][1])
            result.append(item)
        updated_vnnlib.append((v[0], result))
    
    return updated_vnnlib


@dataclass
class BatchedSpecs:
    # batch_size: number of specs in this batch
    batch_size: int
    # x: clean input, [batch, *input_shape[1:]]
    # it also includes the perturbation information, i.e., x_L, x_U, eps...
    x: BoundedTensor
    # c: c matrix after output, [batch, num_and, num_output]
    c: torch.Tensor
    # rhs: values of the right-hand side of the specification, [batch, num_and]
    rhs: torch.Tensor
    # or_spec_size: number of AND clauses in each OR spec, [batch]
    or_spec_size: torch.Tensor
    # same_x_range: whether the input range is the same in this batch
    # * if same_x_range is True, x, data_min and data_max: [1, *input_shape[1:]]
    # * this setting is used to align with the current implementation of complete verifier
    same_x_range: bool
    # same_or_spec_size: whether the number of AND clauses is the same in this batch
    # * if same_or_spec_size is False, c and rhs are zero-padded to the same size
    # * but can be recovered by or_spec_size
    same_or_spec_size: bool

    def get(self, device, single_x_range=False):
        """Move the batched specs to the device and return."""
        
        x = self.x.to(device)
        c = self.c.to(device)
        rhs = self.rhs.to(device)
        or_spec_size = self.or_spec_size.to(device)
        same_x_range = self.same_x_range
        same_or_spec_size = self.same_or_spec_size

        if same_x_range and single_x_range:
            # When the input range is the same, we can get a single copy of it
            # if single_x_range is True.
            # [1, *input_shape[1:]]
            x = take_batch(x, batch_size=1, batch_dim=0)

        return (
            x,
            c,
            rhs,
            or_spec_size,
            same_x_range,
            same_or_spec_size,
        )

    def print_stats(self):
        """Print the statistics of the batched specs."""
        print(f"Batch size: {self.batch_size}")
        print(f"First 10 spec matrices")
        print(f"C: {self.c[:10]}")
        print(f"RHS: {self.rhs[:10]}")
        print(f"OR spec sizes: {self.or_spec_size[:10]}")

class vnnlibHandler:
    def __init__(self, vnnlib, vnnlib_shape):
        """Initialize vnnlibHandler by parsing and storing data."""
        shrink_eps = arguments.Config['specification']['shrink_eps']
        if shrink_eps is not None:
            vnnlib = shrink_vnnlib(vnnlib, float(shrink_eps))
        if arguments.Config['general']['store_all_specs_on_cpu']:
            self.device = "cpu"
        else:
            self.device = arguments.Config['general']['device']
        self.vnnlib = vnnlib
        # here we define the input shape with -1 as the batch size.
        # it is helpful to reshape the input data later.
        # in all comments, we specify the specific batch sizes for clarity.
        input_shape = [-1, *vnnlib_shape[1:]]
        x_range_list, x_list, data_min_list, data_max_list = [], [], [], []
        eps_list = []
        c_list, or_spec_size_list, rhs_list = [], [], []
        num_or_per_x_list = []

        # Determine input format
        # input range can be: 1. a dict for general Lp norm and 2. np array or list of tuples for Linf norm
        all_dict = all(isinstance(vnn[0], dict) for vnn in vnnlib)
        all_list = all(isinstance(vnn[0], list) for vnn in vnnlib)
        all_array = all(isinstance(vnn[0], np.ndarray) for vnn in vnnlib)
        assert (
            all_dict or all_list or all_array
        ), "Input range should be either all dict for Lp norm or all list/array for Linf norm"

        is_linf_norm = not all_dict

        # Parse input data
        # iterate through different inputs
        for vnn in vnnlib:
            if is_linf_norm:
                # vnn[0] is a list of tuples
                x_range_list.append(vnn[0])
            else:
                # vnn[0]["X"] is a tensor
                x_list.append(vnn[0]["X"])
                data_min_list.append(vnn[0]["data_min"])
                data_max_list.append(vnn[0]["data_max"])
                # vnn[0]["eps"] can be float or tensor
                eps_list.append(torch.as_tensor(vnn[0]["eps"]).view(-1))
            num_or_per_x_list.append(len(vnn[1]))
            # iterate through different OR specs
            for c, rhs in vnn[1]:
                # c shape: [num_and, num_output], rhs shape: [num_and]
                # num_and can be different for different OR specs.
                c_list.append(c)
                rhs_list.append(rhs)
                # record the number of ANDs in every OR spec
                or_spec_size_list.append(c.shape[0])

        # Convert lists to tensors with desired dtype, device, and shape.
        # We empirically find that using torch.Tensor is efficient.
        device = self.device
        dtype = torch.get_default_dtype()
        self.or_spec_size = torch.Tensor(or_spec_size_list).to(dtype=torch.int, device=device)
        if is_linf_norm:
            # [num_different_X, *input_shape[1:], 2]
            x_range_list = torch.Tensor(x_range_list).to(dtype=dtype, device=device).view(input_shape + [2])
            # [num_different_X, *input_shape[1:]]
            self.data_min = x_range_list.select(dim=-1, index=0)
            self.data_max = x_range_list.select(dim=-1, index=1)
            self.x = (self.data_min + self.data_max) / 2
        else:
            # [num_different_X, *input_shape[1:]]
            self.x = torch.cat(x_list).to(dtype=dtype, device=device).view(input_shape)
            self.data_min = torch.cat(data_min_list).to(dtype=dtype, device=device).view(input_shape)
            self.data_max = torch.cat(data_max_list).to(dtype=dtype, device=device).view(input_shape)
            self.eps = torch.cat(eps_list).to(dtype=dtype, device=device)

        # Repeat the data_min, data_max, and x for each OR spec
        # [num_different_X]
        num_or_per_x_list = torch.Tensor(num_or_per_x_list).to(dtype=torch.int, device=device)
        # [num_or, *input_shape[1:]]
        self.data_min = self.data_min.repeat_interleave(num_or_per_x_list, dim=0)
        self.data_max = self.data_max.repeat_interleave(num_or_per_x_list, dim=0)
        self.x = self.x.repeat_interleave(num_or_per_x_list, dim=0)
        
        if not is_linf_norm:
            # [num_or]
            self.eps = self.eps.repeat_interleave(num_or_per_x_list, dim=0)
        
        # Initialize other attributes
        # c's in c_list and rhs's in rhs_list may have ununiformed shapes,
        # to minimize padding, we store the flattened c's and rhs's

        # [num_spec, num_output], num_spec = sum(or_spec_size)
        self.c = torch.Tensor(np.concatenate(c_list)).to(device=device, dtype=dtype)
        # [num_spec]
        self.rhs = torch.Tensor(np.concatenate(rhs_list)).to(device=device, dtype=dtype)

        # [num_or]. start / end index of each OR spec in c and rhs
        self.end_idx_per_or = torch.cumsum(self.or_spec_size, dim=0, dtype=torch.int64)
        self.start_idx_per_or = torch.roll(self.end_idx_per_or, 1)
        self.start_idx_per_or[0] = 0

        self.is_linf_norm = is_linf_norm
        self.total_num_or = self.end_idx_per_or.shape[0]
        self.input_shape = input_shape
        self.num_output = self.c.shape[-1]

        self.current_index = 0
        self._set_all_specs()

    def _pop(self, batch_size):
        """Pop a batch of specs from the VNNLibBatcher."""
        if self.current_index >= self.total_num_or:
            return None  # No more data

        # Update index
        start_idx = self.current_index
        end_idx = min(self.current_index + batch_size, self.total_num_or)
        batch_size = end_idx - start_idx
        self.current_index = end_idx

        # for general Lp norm, eps should be the same for all specs in this batch
        if not self.is_linf_norm:
            assert torch.all(self.eps[start_idx:end_idx] == self.eps[start_idx])

        # slice input
        # [batch_size, *input_shape[1:]]
        batch_x = self.x[start_idx:end_idx].view(self.input_shape)
        batch_data_min = self.data_min[start_idx:end_idx].view(self.input_shape)
        batch_data_max = self.data_max[start_idx:end_idx].view(self.input_shape)

        # Check if data_min and data_max are identical in this batch
        same_x_range = ((batch_data_min == batch_data_min[0:1]).all() and 
                        (batch_data_max == batch_data_max[0:1]).all())
        # If the input range is the same, we only need one copy of it.

        # slice c and rhs
        # [batch_size]
        batch_or_spec_size = self.or_spec_size[start_idx:end_idx]
        # [num_spec_in_batch, num_output], [num_spec_in_batch]
        # num_spec_in_batch = sum(batch_num_and)
        batch_c = self.c[self.start_idx_per_or[start_idx]: self.end_idx_per_or[end_idx-1]]
        batch_rhs = self.rhs[self.start_idx_per_or[start_idx]: self.end_idx_per_or[end_idx-1]]

        same_or_spec_size = (batch_or_spec_size == batch_or_spec_size[0]).all()
        if same_or_spec_size:
            # [batch_size, num_and, num_output], [batch_size, num_and]
            batch_c = batch_c.view(batch_size, -1, self.num_output)
            batch_rhs = batch_rhs.view(batch_size, -1)
        else:
            # split c and rhs by or_spec_size to tuples of tensors with different sizes
            batch_c = torch.split(batch_c, batch_or_spec_size.tolist(), dim=0)
            batch_rhs = torch.split(batch_rhs, batch_or_spec_size.tolist(), dim=0)

            # pad c and rhs and convert to tensors
            # we pad c with 0s and rhs with infs to
            # make the padded specs always unverifiable
            # and have no effect on the original specs
            # [batch_size, max_num_and, num_output], [batch_size, max_num_and]
            batch_c = pad_list_of_input_to_tensor(
                batch_c, pad_value=0, pad_dim=0, batch_dim=None, is_orginal_tensor=True, device=self.device
            )
            batch_rhs = pad_list_of_input_to_tensor(
                batch_rhs, pad_value=float("inf"), pad_dim=0, batch_dim=None, is_orginal_tensor=True, device=self.device
            )

        norm = arguments.Config['specification']['norm']
        if self.is_linf_norm:
            ptb = PerturbationLpNorm(norm=norm, x_L=batch_data_min, x_U=batch_data_max)
        else:
            ptb = PerturbationLpNorm(norm=norm, eps=self.eps[start_idx].item(), x_L=batch_data_min, x_U=batch_data_max)
        batch_x = BoundedTensor(batch_x, ptb)

        batched_specs = BatchedSpecs(
            batch_size=batch_c.shape[0],
            x=batch_x,
            c=batch_c,
            rhs=batch_rhs,
            or_spec_size=batch_or_spec_size,
            same_x_range=same_x_range,
            same_or_spec_size=same_or_spec_size,
        )
        if not self.is_linf_norm:
            batched_specs.eps = self.eps[start_idx]

        return batched_specs

    def _set_all_specs(self):
        """Get a BatchedSpecs object containing all specifications."""
        # c and rhs are in the regular form
        # c: [num_or, num_and, num_output], rhs: [num_or, num_and]
        current_index = self.current_index
        self.current_index = 0
        all_specs = self._pop(self.total_num_or)
        self.current_index = current_index
        self.all_specs = all_specs

    def add_rhs_offset(self, rhs_offset: Optional[torch.Tensor] = None):
        """Add offset to the right-hand side of the specifications."""
        if rhs_offset is None:
            return

        self.vnnlib = add_rhs_offset_legacy(self.vnnlib, rhs_offset)

        # For debugging, add a print statement if sanity check is enabled
        print('Add an offset to RHS for debugging:', rhs_offset)

        # FIXME After PR #536, the code was broken when --rhs_offset is provided.
        # Also, the hard-coded delta = 1e-3 should probably be avoided.
        if isinstance(rhs_offset, float):
            self.rhs += rhs_offset
        else:
            delta = 1e-3  # Small value to avoid numerical issues
            rhs_offset = (rhs_offset + delta).to(self.device)
            assert (
                rhs_offset.numel() == 1 or rhs_offset.numel() == self.rhs.numel()
            ), "rhs_offset should be either a scalar or an array with the same size as rhs"

            self.rhs += rhs_offset.view(-1).expand_as(self.rhs)

        return

    def prune_verified_or_specs(self, unverified_or_mask):
        unverified_or_mask = unverified_or_mask.to(self.device)
        unverified_spec_mask = unverified_or_mask.repeat_interleave(
            self.or_spec_size, dim=0
        )
        # Prune the specifications based on the unverified mask
        self.x = self.x[unverified_or_mask]
        self.data_min = self.data_min[unverified_or_mask]
        self.data_max = self.data_max[unverified_or_mask]
        if not self.is_linf_norm:
            self.eps = self.eps[unverified_or_mask]
        self.c = self.c[unverified_spec_mask]
        self.rhs = self.rhs[unverified_spec_mask]
        self.or_spec_size = self.or_spec_size[unverified_or_mask]

        # Update total number of OR specs and reset current index
        self.total_num_or = self.x.shape[0]
        self.current_index = 0

        # Update the end and start indices for OR specs
        self.end_idx_per_or = torch.cumsum(self.or_spec_size, dim=0, dtype=torch.int64)
        self.start_idx_per_or = torch.roll(self.end_idx_per_or, 1)
        if self.total_num_or > 0:
            self.start_idx_per_or[0] = 0

        # Update all_specs
        self._set_all_specs()

    # NOTE: this function is only used by invprop.
    def update_input_bounds(self, data_min, data_max):
        self.data_min = data_min.to(self.device)
        self.data_max = data_max.to(self.device)

        self._set_all_specs()

def shrink_vnnlib(vnnlib, shrink_eps):
    assert shrink_eps > 0, "shrink_eps should be positive"
    for vnn in vnnlib:
        for i in range(len(vnn[0])):
            vnn[0][i] = (vnn[0][i][0] + shrink_eps, vnn[0][i][1] - shrink_eps)
    return vnnlib
