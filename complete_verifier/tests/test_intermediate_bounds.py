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
"""
Unit tests for state/intermediate_bounds.py
Currently covers: WorkingIntermBoundsInfo.from_histories.
"""

import os
import sys
import unittest
import torch

# Add complete_verifier (parent dir) to path for imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.intermediate_bounds import (
    WorkingIntermBoundsInfo,
    IntermBoundsInfo,
)

# CI runs on CPU; keep everything on CPU/float32 so expected values are exact.
device = 'cpu'
dtype = torch.float32
def _tensor(x):
    return torch.tensor(x, device=device, dtype=dtype)


def make_split(indices, directions, values):
    """Build one layer's history entry from plain Python lists.

    Returns the 5-tuple (indices, directions, values, scores, depths)
    used by BaB histories; scores and depths are unused by from_histories
    and filled with zeros.
    """
    n = len(indices)
    return (
        torch.tensor(indices, dtype=torch.long),
        _tensor(directions),
        _tensor(values),
        torch.zeros(n, dtype=dtype),
        torch.zeros(n, dtype=torch.long),
    )


class TestFromHistories(unittest.TestCase):

    def test_single_domain_relu_splits(self):
        """One domain, one layer: a positive split sets lb := 0, a negative
        split sets ub := 0, all other neurons keep the template bounds."""
        ref_lb = {'/relu1': _tensor([[-1., -2., -3., -4.]])}
        ref_ub = {'/relu1': _tensor([[1., 2., 3., 4.]])}

        # Neuron 1 split positive (lb -> 0), neuron 3 split negative (ub -> 0).
        histories = [
            {'/relu1': make_split(indices=[1, 3], directions=[+1., -1.],
                                values=[0., 0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert isinstance(interm, WorkingIntermBoundsInfo)
        assert set(interm.keys()) == {'/relu1'}
        assert isinstance(interm['/relu1'], IntermBoundsInfo)
        assert torch.equal(interm['/relu1'].lower_bound,
                        _tensor([[-1., 0., -3., -4.]]))
        assert torch.equal(interm['/relu1'].upper_bound,
                        _tensor([[1., 2., 3., 0.]]))


    def test_batch_rows_are_independent(self):
        """Two domains splitting the SAME neuron must each land on their own batch
        row only.
        """
        ref_lb = {'/relu1': _tensor([[-1., -1., -1.],
                                    [-1., -1., -1.]])}
        ref_ub = {'/relu1': _tensor([[1., 1., 1.],
                                    [1., 1., 1.]])}

        histories = [
            # Domain 0 -> batch row 0: neuron 1 positive split -> lb[0, 1] = 0.
            {'/relu1': make_split(indices=[1], directions=[+1.], values=[0.])},
            # Domain 1 -> batch row 1: neuron 1 negative split -> ub[1, 1] = 0.
            {'/relu1': make_split(indices=[1], directions=[-1.], values=[0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        # Only row 0's lower bound changes (neuron 1); row 1 keeps the template lb.
        assert torch.equal(interm['/relu1'].lower_bound,
                        _tensor([[-1., 0., -1.],
                                    [-1., -1., -1.]]))
        # Only row 1's upper bound changes (neuron 1); row 0 keeps the template ub.
        assert torch.equal(interm['/relu1'].upper_bound,
                        _tensor([[1., 1., 1.],
                                    [1., 0., 1.]]))


    def test_nonzero_split_values(self):
        """General (non-ReLU) splits carry arbitrary split points: the stored
        value, not 0, must be written into the bound."""
        ref_lb = {'/sin1': _tensor([[-2., -2.]])}
        ref_ub = {'/sin1': _tensor([[2., 2.]])}

        # Neuron 0: take x >= 0.5 (lb -> 0.5); neuron 1: take x <= -0.25
        # (ub -> -0.25).
        histories = [
            {'/sin1': make_split(indices=[0, 1], directions=[+1., -1.],
                                values=[0.5, -0.25])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert torch.equal(interm['/sin1'].lower_bound, _tensor([[0.5, -2.]]))
        assert torch.equal(interm['/sin1'].upper_bound, _tensor([[2., -0.25]]))


    def test_multiple_layers_and_untouched_layer(self):
        """Several layers in the templates; a layer never mentioned in any
        history must come back exactly equal to its template."""
        ref_lb = {
            '/relu1': _tensor([[-1., -2.]]),
            '/relu2': _tensor([[-3., -4.]]),
        }
        ref_ub = {
            '/relu1': _tensor([[1., 2.]]),
            '/relu2': _tensor([[3., 4.]]),
        }

        histories = [
            {'/relu1': make_split(indices=[0], directions=[+1.], values=[0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert set(interm.keys()) == {'/relu1', '/relu2'}
        assert torch.equal(interm['/relu1'].lower_bound, _tensor([[0., -2.]]))
        assert torch.equal(interm['/relu1'].upper_bound, _tensor([[1., 2.]]))
        # /relu2 untouched.
        assert torch.equal(interm['/relu2'].lower_bound, ref_lb['/relu2'])
        assert torch.equal(interm['/relu2'].upper_bound, ref_ub['/relu2'])


    def test_history_layer_missing_from_templates_is_skipped(self):
        """A history entry whose layer is not in the templates is ignored,
        matching the behavior of only visiting layers in the template."""
        ref_lb = {'/relu1': _tensor([[-1., -1.]])}
        ref_ub = {'/relu1': _tensor([[1., 1.]])}

        histories = [
            {
                '/relu1': make_split(indices=[0], directions=[-1.], values=[0.]),
                '/not_in_template': make_split(
                    indices=[1], directions=[+1.], values=[0.]),
            },
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert set(interm.keys()) == {'/relu1'}
        assert torch.equal(interm['/relu1'].lower_bound, _tensor([[-1., -1.]]))
        assert torch.equal(interm['/relu1'].upper_bound, _tensor([[0., 1.]]))


    def test_multi_dimensional_layer_uses_flat_indices(self):
        """For a layer of shape [batch, 2, 3], history indices address the
        flattened per-sample tensor: flat index 4 is position (1, 1)."""
        ref_lb = {'/conv1': torch.full((1, 2, 3), -1., dtype=dtype,
                                    device=device)}
        ref_ub = {'/conv1': torch.full((1, 2, 3), 1., dtype=dtype,
                                    device=device)}

        histories = [
            # Flat index 0 -> (0, 0) positive split; flat index 4 -> (1, 1)
            # negative split.
            {'/conv1': make_split(indices=[0, 4], directions=[+1., -1.],
                                values=[0., 0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        expected_lb = _tensor([[[0., -1., -1.],
                                [-1., -1., -1.]]])
        expected_ub = _tensor([[[1., 1., 1.],
                                [1., 0., 1.]]])
        assert interm['/conv1'].lower_bound.shape == (1, 2, 3)
        assert torch.equal(interm['/conv1'].lower_bound, expected_lb)
        assert torch.equal(interm['/conv1'].upper_bound, expected_ub)


    def test_legacy_list_history_is_converted(self):
        """Histories may store plain Python lists (legacy format); they must be
        converted via convert_history_from_list and yield identical results."""
        ref_lb = {'/relu1': _tensor([[-1., -2., -3.]])}
        ref_ub = {'/relu1': _tensor([[1., 2., 3.]])}

        # 5-element list-of-lists entry: indices, directions, values, scores,
        # depths.
        histories = [
            {'/relu1': [[2], [+1.], [0.], [0.], [0]]},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert torch.equal(interm['/relu1'].lower_bound,
                        _tensor([[-1., -2., 0.]]))
        assert torch.equal(interm['/relu1'].upper_bound,
                        _tensor([[1., 2., 3.]]))


    def test_template_longer_than_batch_and_not_modified(self):
        """Templates may have a larger batch dimension than len(histories):
        only the first rows are used, and the templates themselves must stay
        untouched (from_histories clones before writing).

        Templates are reused, pre-grown buffers (IntermBoundsFactory grows the
        static bounds by repeating a single reference row), so in production every
        row is identical. Here we deliberately give each row a DISTINCT value
        (-1/-2/-3) purely so the asserts can tell which rows the [:batch] slice
        actually kept -- with identical rows that slice would be unobservable."""
        ref_lb = {'/relu1': _tensor([[-1., -1.],
                                    [-2., -2.],
                                    [-3., -3.]])}
        ref_ub = {'/relu1': _tensor([[1., 1.],
                                    [2., 2.],
                                    [3., 3.]])}
        ref_lb_copy = {k: v.clone() for k, v in ref_lb.items()}
        ref_ub_copy = {k: v.clone() for k, v in ref_ub.items()}

        histories = [
            {'/relu1': make_split(indices=[0], directions=[+1.], values=[0.])},
            {'/relu1': make_split(indices=[1], directions=[-1.], values=[0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        # Only the first len(histories) = 2 template rows are used.
        assert interm['/relu1'].lower_bound.shape == (2, 2)
        assert torch.equal(interm['/relu1'].lower_bound,
                        _tensor([[0., -1.],
                                    [-2., -2.]]))
        assert torch.equal(interm['/relu1'].upper_bound,
                        _tensor([[1., 1.],
                                    [2., 0.]]))

        # The reference templates must not be modified in place.
        assert torch.equal(ref_lb['/relu1'], ref_lb_copy['/relu1'])
        assert torch.equal(ref_ub['/relu1'], ref_ub_copy['/relu1'])


    def test_empty_histories_keep_templates(self):
        """Domains with no splits at all (empty dicts) just clone the first
        len(histories) rows of the templates."""
        ref_lb = {'/relu1': _tensor([[-1., -2.],
                                    [-3., -4.]])}
        ref_ub = {'/relu1': _tensor([[1., 2.],
                                    [3., 4.]])}

        histories = [{}, {}]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert torch.equal(interm['/relu1'].lower_bound, ref_lb['/relu1'])
        assert torch.equal(interm['/relu1'].upper_bound, ref_ub['/relu1'])


    def test_mixed_batch_some_domains_without_splits(self):
        """A batch mixing domains with and without splits on a layer: rows of
        domains without splits stay equal to the template rows."""
        ref_lb = {'/relu1': _tensor([[-5., -5.],
                                    [-5., -5.],
                                    [-5., -5.]])}
        ref_ub = {'/relu1': _tensor([[5., 5.],
                                    [5., 5.],
                                    [5., 5.]])}

        histories = [
            {'/relu1': make_split(indices=[0, 1], directions=[+1., +1.],
                                values=[0., 0.])},
            {},  # No splits in domain 1.
            {'/relu1': make_split(indices=[1], directions=[-1.], values=[0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert torch.equal(interm['/relu1'].lower_bound,
                        _tensor([[0., 0.],
                                    [-5., -5.],
                                    [-5., -5.]]))
        assert torch.equal(interm['/relu1'].upper_bound,
                        _tensor([[5., 5.],
                                    [5., 5.],
                                    [5., 0.]]))


    def test_empty_history_entry_on_noncontiguous_template(self):
        """A history entry with zero splits must leave the layer untouched
        without ever calling view() on its bounds."""
        ref_lb = {'/conv1': torch.full((1, 4, 3, 3), -1., dtype=dtype,
                                    device=device)
                  .to(memory_format=torch.channels_last)}
        ref_ub = {'/conv1': torch.full((1, 4, 3, 3), 1., dtype=dtype,
                                    device=device)
                  .to(memory_format=torch.channels_last)}
        assert not ref_lb['/conv1'].is_contiguous()

        # The layer IS mentioned in the history, but with zero splits, as in
        # the first BaB iteration.
        histories = [
            {'/conv1': make_split(indices=[], directions=[], values=[])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        assert torch.equal(interm['/conv1'].lower_bound, ref_lb['/conv1'])
        assert torch.equal(interm['/conv1'].upper_bound, ref_ub['/conv1'])


    def test_one_sided_splits_on_noncontiguous_template(self):
        """Splits in only one direction must only view() the bound tensor of
        that direction.

        NOTE: the touched side's view() still requires a contiguous
        template; this test only pins the untouched side."""
        ref_lb = {'/conv1': torch.full((1, 4, 3, 3), -1., dtype=dtype,
                                    device=device)}  # contiguous, gets written
        ref_ub = {'/conv1': torch.full((1, 4, 3, 3), 1., dtype=dtype,
                                    device=device)
                  .to(memory_format=torch.channels_last)}  # untouched side

        # Positive split only: lb[flat 5] -> 0, ub must never be viewed.
        histories = [
            {'/conv1': make_split(indices=[5], directions=[+1.], values=[0.])},
        ]

        interm = WorkingIntermBoundsInfo.from_histories(
            histories, ref_lb, ref_ub, device=device)

        expected_lb = ref_lb['/conv1'].clone()
        expected_lb.view(1, -1)[0, 5] = 0.
        assert torch.equal(interm['/conv1'].lower_bound, expected_lb)
        assert torch.equal(interm['/conv1'].upper_bound, ref_ub['/conv1'])


    def test_ill_formed_history_raises(self):
        """Mismatched indices/directions/values lengths must raise loudly
        instead of silently writing garbage into the bounds."""
        ref_lb = {'/relu1': _tensor([[-1., -1.]])}
        ref_ub = {'/relu1': _tensor([[1., 1.]])}

        # 2 indices but only 1 direction.
        bad_directions = (
            torch.tensor([0, 1], dtype=torch.long),
            _tensor([+1.]),
            _tensor([0., 0.]),
            torch.zeros(2, dtype=dtype),
            torch.zeros(2, dtype=torch.long),
        )
        with self.assertRaises(AssertionError):
            WorkingIntermBoundsInfo.from_histories(
                [{'/relu1': bad_directions}], ref_lb, ref_ub, device=device)

        # 2 indices but only 1 value.
        bad_values = (
            torch.tensor([0, 1], dtype=torch.long),
            _tensor([+1., -1.]),
            _tensor([0.]),
            torch.zeros(2, dtype=dtype),
            torch.zeros(2, dtype=torch.long),
        )
        with self.assertRaises(AssertionError):
            WorkingIntermBoundsInfo.from_histories(
                [{'/relu1': bad_values}], ref_lb, ref_ub, device=device)


if __name__ == '__main__':
    unittest.main()
