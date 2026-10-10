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
"""Unit tests for beta_CROWN_solver.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict

import torch
import torch.nn as nn
import pytest

from .conftest import requires_cuda

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize arguments.Config before importing beta_CROWN_solver
import arguments
arguments.Config.parse_config(args=[], verbose=False)

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any

from beta_CROWN_solver import LiRPANet, BatchHandler
from state import AlphaFullInfoData, AlphaValueData, BetaFullData
from utils import Timer


class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """A simple convolutional model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def create_bounded_tensor(shape, eps=0.1):
    """Helper to create a BoundedTensor for testing."""
    data = torch.randn(shape)
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - eps, x_U=data + eps)
    return BoundedTensor(data, ptb)


class TestLiRPANetInit(unittest.TestCase):
    """Tests for LiRPANet initialization."""

    @requires_cuda
    def test_init_simple_model(self):
        """Test initialization with a simple linear model."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNotNone(net.net)
        self.assertEqual(net.input_shape, in_size)
        self.assertIsNotNone(net.final_name)
        self.assertIsInstance(net.timer, Timer)

    @requires_cuda
    def test_init_stores_model_ori(self):
        """Test that original model is stored."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIs(net.model_ori, model)

    @requires_cuda
    def test_init_default_device(self):
        """Test default device assignment."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # Device should be set from arguments.Config
        self.assertIsNotNone(net.device)

    def test_init_custom_device(self):
        """Test custom device assignment."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size, device='cpu')

        self.assertEqual(net.device, 'cpu')

    @requires_cuda
    def test_init_with_c_matrix(self):
        """Test initialization with specification matrix C."""
        model = SimpleModel(output_size=5)
        in_size = (1, 10)
        c = torch.eye(5).unsqueeze(0)
        net = LiRPANet(model, in_size, c=c)

        self.assertIsNotNone(net.c)
        self.assertEqual(net.c.shape, (1, 5, 5))

    @requires_cuda
    def test_init_creates_bounded_module(self):
        """Test that BoundedModule is created."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        from auto_LiRPA import BoundedModule
        self.assertIsInstance(net.net, BoundedModule)

    @requires_cuda
    def test_init_return_A_default_false(self):
        """Test that return_A defaults to False."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertFalse(net.return_A)

    @requires_cuda
    def test_init_needed_A_dict_default_none(self):
        """Test that needed_A_dict defaults to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.needed_A_dict)

    @requires_cuda
    def test_init_cutter_default_none(self):
        """Test that cutter defaults to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.cutter)

    @requires_cuda
    def test_init_interm_transfer_default_true(self):
        """Test that interm_transfer defaults to True."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertTrue(net.interm_transfer)


class TestLiRPANetSplitNodes(unittest.TestCase):
    """Tests for LiRPANet.split_nodes property."""

    @requires_cuda
    def test_split_nodes_property(self):
        """Test split_nodes property returns net.split_nodes after get_split_nodes called."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # Need to call get_split_nodes first to populate split_nodes
        net.get_split_nodes()

        # Access split_nodes property
        split_nodes = net.split_nodes
        # Should be the same as net.net.split_nodes
        self.assertIs(split_nodes, net.net.split_nodes)


class TestLiRPANetEmptyHistory(unittest.TestCase):
    """Tests for LiRPANet.empty_history method."""

    @requires_cuda
    def test_empty_history_returns_dict(self):
        """Test that empty_history returns a dictionary."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)
        net.get_split_nodes()

        history = net.empty_history()

        self.assertIsInstance(history, dict)

    @requires_cuda
    def test_empty_history_structure(self):
        """Test the structure of empty history."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)
        net.get_split_nodes()

        history = net.empty_history()

        # Each entry should be a tuple of 5 empty lists
        for layer_name, hist_tuple in history.items():
            self.assertIsInstance(hist_tuple, tuple)
            self.assertEqual(len(hist_tuple), 5)
            for lst in hist_tuple:
                self.assertIsInstance(lst, list)
                self.assertEqual(len(lst), 0)

    @requires_cuda
    def test_empty_history_input_split_returns_none(self):
        """Test empty_history returns None when input split is enabled."""
        model = SimpleModel()
        in_size = (1, 10)

        # Enable input split
        original_value = arguments.Config['bab']['branching']['input_split']['enable']
        arguments.Config['bab']['branching']['input_split']['enable'] = True

        net = LiRPANet(model, in_size)
        history = net.empty_history()
        self.assertIsNone(history)

        arguments.Config['bab']['branching']['input_split']['enable'] = original_value


class TestLiRPANetGetMask(unittest.TestCase):
    """Tests for LiRPANet.get_mask method."""

    @requires_cuda
    def test_get_mask_returns_dict(self):
        """Test that get_mask returns a dictionary."""
        model = SimpleModel()
        in_size = (1, 10)
        # Use CPU device explicitly to avoid device mismatch
        net = LiRPANet(model, in_size, device='cpu')

        masks = net.get_mask()

        self.assertIsInstance(masks, dict)

    @requires_cuda
    def test_get_mask_input_split_returns_empty(self):
        """Test get_mask returns empty dict when input split is enabled."""
        model = SimpleModel()
        in_size = (1, 10)

        # Enable input split
        original_value = arguments.Config['bab']['branching']['input_split']['enable']
        arguments.Config['bab']['branching']['input_split']['enable'] = True

        net = LiRPANet(model, in_size)
        masks = net.get_mask()
        self.assertEqual(masks, {})

        arguments.Config['bab']['branching']['input_split']['enable'] = original_value


class TestLiRPANetSetAOptions(unittest.TestCase):
    """Tests for LiRPANet._set_A_options method."""

    @requires_cuda
    def test_set_A_options_default(self):
        """Test _set_A_options with default parameters."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        net._set_A_options()

        # By default, return_A should be False
        self.assertFalse(net.return_A)

    @requires_cuda
    def test_set_A_options_with_return_A_true(self):
        """Test _set_A_options with return_A=True."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        net._set_A_options(return_A=True)

        self.assertTrue(net.return_A)
        self.assertIsNotNone(net.needed_A_dict)

    @requires_cuda
    def test_set_A_options_with_bab_get_upper_bound(self):
        """Test _set_A_options when get_upper_bound is enabled."""
        model = SimpleModel()
        in_size = (1, 10)

        original_value = arguments.Config['bab']['get_upper_bound']
        arguments.Config['bab']['get_upper_bound'] = True

        net = LiRPANet(model, in_size)
        net._set_A_options(bab=True)

        self.assertTrue(net.return_A)
        self.assertIsNotNone(net.needed_A_dict)

        arguments.Config['bab']['get_upper_bound'] = original_value


class TestLiRPANetSetTmpA(unittest.TestCase):
    """Tests for LiRPANet._set_tmp_A method."""

    @requires_cuda
    def test_set_tmp_A_clip_disabled(self):
        """Test _set_tmp_A with clip domains disabled."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        temp_return_A, temp_needed_A_dict = net._set_tmp_A(False, 'alpha-crown')

        self.assertFalse(temp_return_A)
        self.assertEqual(len(temp_needed_A_dict), 0)

    @requires_cuda
    def test_set_tmp_A_clip_enabled_no_existing_A(self):
        """Test _set_tmp_A with clip enabled but no existing return_A."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)
        net.return_A = False

        temp_return_A, temp_needed_A_dict = net._set_tmp_A(True, 'alpha-crown')

        self.assertTrue(temp_return_A)
        self.assertIn(net.net.output_name[0], temp_needed_A_dict)

    @requires_cuda
    def test_set_tmp_A_alpha_forward_method(self):
        """Test _set_tmp_A with alpha-forward method returns unchanged."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)
        net.return_A = False

        temp_return_A, temp_needed_A_dict = net._set_tmp_A(True, 'alpha-forward')

        self.assertFalse(temp_return_A)


class TestLiRPANetSetCrownBoundOpts(unittest.TestCase):
    """Tests for LiRPANet.set_crown_bound_opts method."""

    @requires_cuda
    def test_set_crown_bound_opts_alpha(self):
        """Test set_crown_bound_opts with 'alpha' crown name."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # Should not raise
        net.set_crown_bound_opts('alpha')

    @requires_cuda
    def test_set_crown_bound_opts_beta(self):
        """Test set_crown_bound_opts with 'beta' crown name."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # Should not raise
        net.set_crown_bound_opts('beta')


class TestLiRPANetGetSplitNodes(unittest.TestCase):
    """Tests for LiRPANet.get_split_nodes method."""

    @requires_cuda
    def test_get_split_nodes_sets_split_activations(self):
        """Test that get_split_nodes sets split_activations."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        net.get_split_nodes()

        self.assertIsNotNone(net.split_activations)


@pytest.fixture
def build_reference_config(monkeypatch):
    """Use an isolated CPU configuration and reference output for real builds."""
    config = arguments.ConfigHandler()
    config.parse_config(args=[], verbose=False)
    config['general']['device'] = 'cpu'
    config['general']['save_output'] = True
    config['solver']['alpha-crown']['iteration'] = 2
    config['solver']['prune_after_crown'] = True
    config['solver']['optimize_disjuncts_separately'] = True
    monkeypatch.setattr(arguments, 'Config', config)
    out = {'init_crown_bounds': None, 'init_alpha_crown': None}
    monkeypatch.setitem(arguments.Globals, 'out', out)
    return config, out


def reference_build_inputs():
    """Outputs are ReLU(x) + 1 and ReLU(-x) - 2 for x in [-1, 1]."""
    model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model[0].bias.zero_()
        model[2].weight.copy_(torch.eye(2))
        model[2].bias.copy_(torch.tensor([1.0, -2.0]))
    x = BoundedTensor(torch.zeros(5, 1), PerturbationLpNorm(
        norm=float('inf'), x_L=-torch.ones(5, 1), x_U=torch.ones(5, 1)))
    # Each row is an independently optimized property; signs and scales give
    # distinct bounds and interleave CROWN-verified and unverified properties.
    c = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 2.0]],
                      [[0.0, -1.0]], [[2.0, 0.0]]])
    return LiRPANet(model, (1, 1), device='cpu'), x, c


@pytest.mark.parametrize('rhs_value, solving_mode, nonlinear_method, needs_A', [
    pytest.param(0.9, False, 'bbps', True, id='unverified-positive-bound'),
    pytest.param(0.1, False, 'bbps', False, id='verified'),
    pytest.param(0.1, True, 'bbps', True, id='solving-objective'),
    pytest.param(0.9, False, 'babsr-like', False, id='babsr-like'),
])
def test_build_completes_saved_A_for_bbps(
        build_reference_config, rhs_value, solving_mode,
        nonlinear_method, needs_A):
    """BBPS receives intermediate affine bounds when its search needs them."""
    config, _ = build_reference_config
    config['solver']['bound_prop_method'] = 'crown'
    config['solving']['solving_mode'] = solving_mode
    config['bab']['branching']['nonlinear_split']['method'] = nonlinear_method
    model = nn.Sequential(nn.Sigmoid(), nn.Sigmoid())
    net = LiRPANet(model, (1, 1), device='cpu')
    x = BoundedTensor(torch.zeros(1, 1), PerturbationLpNorm(
        norm=float('inf'), x_L=-torch.ones(1, 1), x_U=torch.ones(1, 1)))
    c = torch.ones(1, 1, 1)
    rhs = torch.full((1, 1), rhs_value)

    bounds, _ = net.build(x, c, rhs, stop_criterion_batch_any)
    assert bounds.item() > 0
    assert bool(stop_criterion_batch_any(rhs)(bounds).all()) == (rhs_value == 0.1)

    # The first Sigmoid is bounded by IBP. Its output is the second
    # Sigmoid's split input, so BBPS needs an additional input-space A entry.
    target = net.net.final_node().inputs[0]
    if not needs_A:
        assert target.name not in net.A_saved
        return

    entry = net.A_saved[target.name][net.net.input_name[0]]
    assert entry['lA'].shape == entry['uA'].shape == (1, 1, 1)
    assert entry['lbias'].shape == entry['ubias'].shape == (1, 1)
    points = torch.tensor([[-1.0], [0.0], [1.0]])
    exact = torch.sigmoid(points)
    lower = points @ entry['lA'][0].T + entry['lbias']
    upper = points @ entry['uA'][0].T + entry['ubias']
    assert (lower <= exact + 1e-6).all()
    assert (upper >= exact - 1e-6).all()


@pytest.mark.parametrize('batch_size, optimized_bounds', [
    (1, [1.0, -2.0, -4.0, 1.0, 2.0]),
    (2, [float('inf'), -2.0, -4.0, float('inf'), 2.0]),
    (1000000, [float('inf'), -2.0, -4.0, float('inf'), float('inf')]),
])
def test_build_reference_preserves_all_properties(
        build_reference_config, batch_size, optimized_bounds):
    """References preserve property order, including pruned and tail batches."""
    config, out = build_reference_config
    config['solver']['build_batch_size'] = batch_size
    net, x, c = reference_build_inputs()
    rhs = torch.zeros(c.shape[:2])
    bounds, _ = net.build(x, c, rhs, stop_criterion_batch_any)
    # Pruned properties are restored as +inf, except in CROWN-only batches
    # where the finite bounds already prove every property in that batch.
    expected = torch.tensor(optimized_bounds).unsqueeze(1)
    torch.testing.assert_close(bounds, expected)
    torch.testing.assert_close(out['init_crown_bounds'],
                               torch.tensor([[1.0], [-2.0], [-4.0], [1.0], [2.0]]))
    torch.testing.assert_close(out['init_alpha_crown'], expected)
    assert out['init_crown_bounds'].device.type == 'cpu'
    assert not out['init_alpha_crown'].requires_grad
    # Recording must not change bounds compared with the production path.
    config['general']['save_output'] = False
    net, x, c = reference_build_inputs()
    unsaved_bounds, _ = net.build(x, c, rhs, stop_criterion_batch_any)
    torch.testing.assert_close(unsaved_bounds, bounds)


def test_build_reference_initial_crown_fast_path(build_reference_config):
    """A CROWN-only build does not retain optimized bounds from an earlier build."""
    config, out = build_reference_config
    config['solver']['build_batch_size'] = 1000000
    net, x, c = reference_build_inputs()
    net.build(x, c, torch.zeros(c.shape[:2]), stop_criterion_batch_any)
    net.build(x[:1], c[:1], torch.zeros(1, 1), stop_criterion_batch_any)
    torch.testing.assert_close(out['init_crown_bounds'], torch.tensor([[1.0]]))
    assert out['init_alpha_crown'] is None


def test_build_reference_middle_attack_keeps_computed_prefix(
        build_reference_config, monkeypatch):
    """A counterexample leaves only bounds actually computed before the return."""
    config, out = build_reference_config
    config['solver']['build_batch_size'] = 1
    config['attack']['pgd_order'] = 'middle'
    net, x, c = reference_build_inputs()
    witness = torch.zeros(1, 1)
    monkeypatch.setattr('beta_CROWN_solver.attack',
                        lambda *args, **kwargs: (None, True, witness, None, None))
    handler = MagicMock()
    handler.vnnlib = []
    bounds, result = net.build(
        x, c, torch.zeros(c.shape[:2]), stop_criterion_batch_any,
        vnnlib_handler=handler)
    assert bounds is None
    torch.testing.assert_close(result['attack_examples'], witness)
    torch.testing.assert_close(out['init_crown_bounds'], torch.tensor([[1.0], [-2.0]]))
    torch.testing.assert_close(out['init_alpha_crown'], torch.tensor([[1.0]]))


class TestBatchHandlerInit(unittest.TestCase):
    """Tests for BatchHandler initialization."""

    def test_init_basic(self):
        """Test basic BatchHandler initialization."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(handler.batch_size_ori, 4)
        self.assertEqual(handler.batch_size_target, 2)
        self.assertEqual(handler.total_batches, 2)

    def test_init_single_batch(self):
        """Test BatchHandler with single batch."""
        x = create_bounded_tensor((2, 10))
        c = torch.randn(2, 3, 5)
        rhs = torch.randn(2, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=10,  # Larger than batch_size_ori
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(handler.total_batches, 1)

    def test_init_stores_attributes(self):
        """Test that BatchHandler stores all attributes."""
        # Use batch_size_target that equals batch_size_ori to avoid assertion
        # when need_alphas=True (alphas require same x range or single batch)
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=4,  # Single batch to allow need_alphas=True
            final_name='/output',
            need_alphas=True,
            full_alpha_info=True
        )

        self.assertTrue(handler.need_alphas)
        self.assertTrue(handler.full_alpha_info)
        self.assertEqual(handler.final_name, '/output')

    def test_init_empty_result_lists(self):
        """Test that result lists are initialized empty."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(len(handler.batch_lb), 0)
        self.assertEqual(len(handler.batch_ub), 0)
        self.assertEqual(len(handler.batch_lA), 0)
        self.assertEqual(len(handler.batch_alpha), 0)
        self.assertEqual(len(handler.batch_mask), 0)
        self.assertEqual(len(handler.batch_A), 0)


class TestBatchHandlerTakeBatch(unittest.TestCase):
    """Tests for BatchHandler.take_batch method."""

    def test_take_batch_first(self):
        """Test taking first batch."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        data = torch.arange(20).reshape(4, 5).float()
        batch = handler.take_batch(data, 0)

        self.assertEqual(batch.shape, (2, 5))
        self.assertTrue(torch.equal(batch, data[:2]))

    def test_take_batch_second(self):
        """Test taking second batch."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        data = torch.arange(20).reshape(4, 5).float()
        batch = handler.take_batch(data, 1)

        self.assertEqual(batch.shape, (2, 5))
        self.assertTrue(torch.equal(batch, data[2:4]))


class TestBatchHandlerAddBatchResult(unittest.TestCase):
    """Tests for BatchHandler.add_batch_result method."""

    @requires_cuda
    def test_add_batch_result_stores_results(self):
        """Test that add_batch_result stores results."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        lb = {'/output': torch.randn(2, 5)}
        ub = {'/output': torch.randn(2, 5)}
        lA = {'/relu': torch.randn(2, 5, 10)}
        alphas = None
        mask = {'/relu': [torch.ones(2, 10, dtype=torch.bool)]}
        input_split_idx = {}

        handler.add_batch_result(lb, ub, lA, alphas, mask, input_split_idx, None)

        self.assertEqual(len(handler.batch_lb), 1)
        self.assertEqual(len(handler.batch_ub), 1)
        self.assertEqual(len(handler.batch_lA), 1)

    @requires_cuda
    def test_add_batch_result_with_A(self):
        """Test add_batch_result with A matrix."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        lb = {'/output': torch.randn(2, 5)}
        ub = {'/output': torch.randn(2, 5)}
        lA = None
        alphas = None
        mask = {'/relu': [torch.ones(2, 10, dtype=torch.bool)]}
        input_split_idx = {}
        A = {'key': torch.randn(2, 5, 10)}

        handler.add_batch_result(lb, ub, lA, alphas, mask, input_split_idx, A)

        self.assertEqual(len(handler.batch_A), 1)


class TestBatchHandlerGetResults(unittest.TestCase):
    """Tests for BatchHandler.get_results method."""

    @requires_cuda
    def test_get_results_concatenates_batches(self):
        """Test that get_results concatenates multiple batches."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        # Add two batches
        for _ in range(2):
            lb = {'/output': torch.randn(2, 5)}
            ub = {'/output': torch.randn(2, 5)}
            lA = {'/relu': torch.randn(2, 5, 10)}
            mask = {'/relu': [torch.ones(2, 10, dtype=torch.bool)]}
            input_split_idx = {'/input': torch.zeros(2, dtype=torch.long)}
            handler.add_batch_result(lb, ub, lA, None, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertIn('lower_bounds', results)
        self.assertIn('upper_bounds', results)
        self.assertEqual(results['lower_bounds']['/output'].shape[0], 4)
        self.assertEqual(results['upper_bounds']['/output'].shape[0], 4)

    @requires_cuda
    def test_get_results_returns_expected_keys(self):
        """Test that get_results returns all expected keys."""
        x = create_bounded_tensor((2, 10))
        c = torch.randn(2, 3, 5)
        rhs = torch.randn(2, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=10,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        lb = {'/output': torch.randn(2, 5)}
        ub = {'/output': torch.randn(2, 5)}
        lA = {'/relu': torch.randn(2, 5, 10)}
        mask = {'/relu': [torch.ones(2, 10, dtype=torch.bool)]}
        input_split_idx = {'/input': torch.zeros(2, dtype=torch.long)}
        handler.add_batch_result(lb, ub, lA, None, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        expected_keys = ['mask', 'lA', 'lower_bounds', 'upper_bounds',
                         'alphas', 'history', 'input_split_idx',
                         'global_lb', 'global_ub', 'betas']
        for key in expected_keys:
            self.assertIn(key, results)


class TestBatchHandlerGetBatchInput(unittest.TestCase):
    """Tests for BatchHandler.get_batch_input method."""

    def test_get_batch_input_returns_tuple(self):
        """Test that get_batch_input returns a tuple of 7 elements."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        result = handler.get_batch_input(0, 'cpu')

        self.assertEqual(len(result), 7)

    def test_get_batch_input_returns_correct_batch_size(self):
        """Test that get_batch_input returns correct batch sizes."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        batch_x, batch_c, batch_rhs, _, _, _, _ = handler.get_batch_input(0, 'cpu')

        self.assertEqual(batch_x.shape[0], 2)
        self.assertEqual(batch_c.shape[0], 2)
        self.assertEqual(batch_rhs.shape[0], 2)

    def test_get_batch_input_with_interm_bounds(self):
        """Test get_batch_input with intermediate bounds."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        interm_bounds = {
            '/relu': [torch.randn(4, 20), torch.randn(4, 20)]
        }

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=interm_bounds,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        _, _, _, _, batch_interm_bounds, _, _ = handler.get_batch_input(0, 'cpu')

        self.assertIsNotNone(batch_interm_bounds)
        self.assertIn('/relu', batch_interm_bounds)
        self.assertEqual(batch_interm_bounds['/relu'][0].shape[0], 2)

    def test_get_batch_input_with_shared_interm_bounds(self):
        """Test get_batch_input expands shared intermediate bounds."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        # Shared bounds (batch size 1)
        interm_bounds = {
            '/relu': [torch.randn(1, 20), torch.randn(1, 20)]
        }

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=interm_bounds,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        _, _, _, _, batch_interm_bounds, _, _ = handler.get_batch_input(0, 'cpu')

        # Should be expanded to batch size 2
        self.assertEqual(batch_interm_bounds['/relu'][0].shape[0], 2)


class TestBatchHandlerWithOrSpecSize(unittest.TestCase):
    """Tests for BatchHandler with or_spec_size."""

    def test_init_with_or_spec_size(self):
        """Test BatchHandler initialization with or_spec_size."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        or_spec_size = torch.tensor([2, 2, 2, 2])

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=or_spec_size,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertIsNotNone(handler.or_spec_size)


class TestLiRPANetPrintIntermBounds(unittest.TestCase):
    """Tests for LiRPANet._print_interm_bounds method."""

    @requires_cuda
    def test_print_interm_bounds_no_args(self):
        """Test _print_interm_bounds with no arguments."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # Should not raise
        net._print_interm_bounds()

    @requires_cuda
    def test_print_interm_bounds_with_bounds(self):
        """Test _print_interm_bounds with bounds provided."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        lb = {net.final_name: torch.randn(1, 5)}
        ub = {net.final_name: torch.randn(1, 5) + 1}

        # Should not raise
        net._print_interm_bounds(lb, ub)


class TestLiRPANetExpandTensors(unittest.TestCase):
    """Tests for LiRPANet._expand_tensors method."""

    def test_expand_tensors_returns_tuple(self):
        """Test that _expand_tensors returns expected tuple."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size, device='cpu')

        # Create a BoundedTensor with uniform bounds (same across batch)
        data = torch.randn(1, 10)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)
        net.x = x
        c = torch.randn(1, 3, 5)
        net.c = c

        d = {
            'lower_bounds': {net.final_name: torch.randn(2, 5)},
            'upper_bounds': {net.final_name: torch.randn(2, 5)},
        }

        result = net._expand_tensors(d, batch=2)

        self.assertEqual(len(result), 7)
        interm_bounds, lb_last, ub_last, c_out, new_x, x_Ls, x_Us = result
        self.assertIsInstance(interm_bounds, dict)
        self.assertEqual(lb_last.shape[0], 2)


class TestLiRPANetConvModel(unittest.TestCase):
    """Tests for LiRPANet with convolutional model."""

    @requires_cuda
    def test_init_conv_model(self):
        """Test initialization with convolutional model."""
        model = ConvModel()
        in_size = (1, 1, 8, 8)
        net = LiRPANet(model, in_size)

        self.assertIsNotNone(net.net)
        self.assertEqual(net.input_shape, in_size)


class TestLiRPANetTimer(unittest.TestCase):
    """Tests for LiRPANet timer functionality."""

    @requires_cuda
    def test_timer_initialized(self):
        """Test that timer is properly initialized."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsInstance(net.timer, Timer)

    @requires_cuda
    def test_timer_can_be_used(self):
        """Test that timer can be started and stopped."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        net.timer.start('test')
        net.timer.add('test')

        self.assertIn('test', net.timer.time_sum)


class TestLiRPANetNonlinearSplit(unittest.TestCase):
    """Tests for LiRPANet nonlinear_split property."""

    @requires_cuda
    def test_nonlinear_split_default(self):
        """Test nonlinear_split default value."""
        model = SimpleModel()
        in_size = (1, 10)

        # Ensure input split is disabled and method is not nonlinear
        orig_input_split = arguments.Config['bab']['branching']['input_split']['enable']
        orig_method = arguments.Config['bab']['branching']['method']

        arguments.Config['bab']['branching']['input_split']['enable'] = False
        arguments.Config['bab']['branching']['method'] = 'kfsb'

        net = LiRPANet(model, in_size)
        # nonlinear_split should be False when method != 'nonlinear'
        self.assertFalse(net.nonlinear_split)

        arguments.Config['bab']['branching']['input_split']['enable'] = orig_input_split
        arguments.Config['bab']['branching']['method'] = orig_method


class TestLiRPANetAlphaStartNodes(unittest.TestCase):
    """Tests for LiRPANet alpha_start_nodes."""

    @requires_cuda
    def test_alpha_start_nodes_includes_final(self):
        """Test that alpha_start_nodes includes final_name."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIn(net.final_name, net.alpha_start_nodes)


class TestAlphaValueData(unittest.TestCase):
    """Tests for restoring stored alpha values to a LiRPANet."""

    def test_attach_restores_per_node_dtype_and_optimizer_parameters(self):
        """Storage precision must not become optimization precision."""
        activations = []
        for name, dtype in [('relu32', torch.float32),
                            ('relu64', torch.float64)]:
            activation = MagicMock()
            activation.name = name
            activation.alpha = {
                'output': torch.ones(2, 1, 1, 3, dtype=dtype)
            }
            activation.output_dtype = dtype
            activations.append(activation)
        bounded_module = MagicMock()
        bounded_module.get_enabled_opt_act.return_value = activations
        net = MagicMock()
        net.net = bounded_module
        net.device = torch.device('cpu')

        stored = AlphaValueData.from_net(
            net, starting_node_scope='all', move=True
        ).to(device='cpu', dtype=torch.float16)
        stored.attach_to_net(net)

        restored = [activation.alpha['output'] for activation in activations]
        self.assertEqual(restored[0].dtype, torch.float32)
        self.assertEqual(restored[1].dtype, torch.float64)
        for activation, alpha in zip(activations, restored):
            self.assertEqual(alpha.device, net.device)
            self.assertTrue(alpha.requires_grad)
            self.assertTrue(alpha.is_leaf)
            self.assertEqual(stored[activation.name]['output'].dtype, torch.float16)

        optimizer = torch.optim.Adam(restored, lr=0.1)
        before = [alpha.detach().clone() for alpha in restored]
        sum(alpha.sum() for alpha in restored).backward()
        optimizer.step()
        for old, alpha in zip(before, restored):
            self.assertFalse(torch.equal(old, alpha))

    def test_full_info_compacts_alpha_and_restores_graph_dtype(self):
        """Full-info export compacts alpha but not its integer metadata."""
        activation = MagicMock()
        activation.name = 'relu'
        original = torch.ones(2, 1, 1, 3, dtype=torch.float64)
        activation.alpha = {'output': original}
        activation.output_dtype = torch.float64
        activation.dump_alpha.return_value = {
            'alpha': {'output': original.detach()},
            'alpha_lookup_idx': {
                'output': torch.tensor([0, 1], dtype=torch.int64)
            },
        }
        bounded_module = MagicMock()
        bounded_module.get_enabled_opt_act.return_value = [activation]
        net = MagicMock()
        net.net = bounded_module
        net.device = torch.device('cpu')

        full_info = AlphaFullInfoData.from_net(
            net, starting_node_scope='all', move=False
        ).to(device='cpu', dtype=torch.float16)
        stored = AlphaValueData(
            _data={activation.name: full_info[activation.name]['alpha']}
        )
        activation.alpha = {}
        stored.attach_to_net(net)

        restored = activation.alpha['output']
        self.assertEqual(stored[activation.name]['output'].dtype, torch.float16)
        self.assertEqual(
            full_info[activation.name]['alpha_lookup_idx']['output'].dtype,
            torch.int64)
        self.assertEqual(restored.dtype, torch.float64)
        self.assertEqual(restored.device, net.device)
        self.assertTrue(restored.requires_grad)
        self.assertTrue(restored.is_leaf)

    def test_attach_requires_graph_output_dtype(self):
        """Alpha restoration must not fall back to a global default dtype."""
        activation = MagicMock()
        activation.name = 'relu'
        activation.output_dtype = None
        activation.alpha = {}
        bounded_module = MagicMock()
        bounded_module.get_enabled_opt_act.return_value = [activation]
        net = MagicMock()
        net.net = bounded_module
        net.device = torch.device('cpu')
        stored = AlphaValueData(
            _data={'relu': {'output': torch.ones(1, dtype=torch.float16)}}
        )

        with self.assertRaisesRegex(
                RuntimeError, 'without its graph output dtype'):
            stored.attach_to_net(net)


class TestBetaFullData(unittest.TestCase):
    """Tests for restoring stored sparse beta values to a LiRPANet."""

    def test_restore_uses_split_node_dtype_and_optimizer_parameters(self):
        """Domain storage must not replace each beta's graph dtype."""
        nodes = {}
        for name, dtype in [('layer32', torch.float32),
                            ('layer64', torch.float64)]:
            node = MagicMock()
            node.name = name
            node.output_dtype = dtype
            nodes[name] = node

        bounded_module = MagicMock()
        bounded_module.__getitem__.side_effect = nodes.__getitem__
        bounded_module.get_output_dtype.side_effect = (
            lambda node: node.output_dtype)

        def accept_beta(beta_data):
            for name, sparse_betas in beta_data.items():
                nodes[name].sparse_betas = sparse_betas

        bounded_module.accept_beta.side_effect = accept_beta
        net = MagicMock()
        net.net = bounded_module

        history = [{
            'layer32': ([0], [1.0]),
            'layer64': ([1], [-1.0]),
        }]
        domain = {
            'history': history,
            'betas': [{
                'layer32': torch.tensor([0.25], dtype=torch.float32),
                'layer64': torch.tensor([0.75], dtype=torch.float64),
            }],
        }

        restored, counts = BetaFullData.from_domain_dict(
            domain, bias=False,
            dtype_map={'layer32': torch.float32, 'layer64': torch.float64},
            device='cpu')
        restored.attach_to_net(net)

        parameters = []
        for name, dtype in [('layer32', torch.float32),
                            ('layer64', torch.float64)]:
            sparse_beta = nodes[name].sparse_betas[0]
            self.assertEqual(sparse_beta.val.dtype, dtype)
            self.assertEqual(sparse_beta.sign.dtype, dtype)
            self.assertEqual(sparse_beta.loc.dtype, torch.int64)
            self.assertTrue(sparse_beta.val.requires_grad)
            self.assertTrue(sparse_beta.val.is_leaf)
            parameters.append(sparse_beta.val)

        optimizer = torch.optim.Adam(parameters, lr=0.1)
        before = [parameter.detach().clone() for parameter in parameters]
        sum(parameter.sum() for parameter in parameters).backward()
        optimizer.step()
        for old, parameter in zip(before, parameters):
            self.assertFalse(torch.equal(old, parameter))

        stored = restored.to_domain_dict(counts, device='cpu')
        self.assertEqual(stored[0]['layer32'].dtype, torch.float32)
        self.assertEqual(stored[0]['layer64'].dtype, torch.float64)

        restored_again, _ = BetaFullData.from_domain_dict(
            {'history': history, 'betas': stored},
            bias=False,
            dtype_map={'layer32': torch.float32, 'layer64': torch.float64},
            device='cpu',
        )
        self.assertEqual(
            restored_again['layer64'][0].val.dtype, torch.float64)
        self.assertTrue(restored_again['layer64'][0].val.is_leaf)

    def test_restore_requires_every_split_node_dtype(self):
        domain = {
            'history': [{'layer': ([0], [1.0])}],
            'betas': [{'layer': torch.tensor([0.25])}],
        }

        with self.assertRaisesRegex(
                RuntimeError, 'Cannot restore beta for layer'):
            BetaFullData.from_domain_dict(
                domain, bias=False, dtype_map={}, device='cpu')


class TestPreprocessConstArguments(unittest.TestCase):
    """Tests for data captured before activation-split preprocessing."""

    def test_capture_split_node_dtypes(self):
        from activation_split.stage_arguments import PreprocessConstArguments

        nodes = []
        for name, dtype in [('layer32', torch.float32),
                            ('layer64', torch.float64)]:
            node = MagicMock()
            node.name = name
            node.output_dtype = dtype
            nodes.append(node)

        net = object.__new__(LiRPANet)
        net.net = MagicMock()
        net.net.split_nodes = nodes
        net.net.get_output_dtype.side_effect = lambda node: node.output_dtype
        net.final_name = 'output'
        net.x = torch.zeros(1)
        net.c = torch.ones(1)
        net.device = torch.device('cpu')
        net.unstable_mask = {}

        const_args = PreprocessConstArguments.from_net(net)

        self.assertEqual(const_args.dtype_map, {
            'layer32': torch.float32,
            'layer64': torch.float64,
        })


class TestLiRPANetBuildHistoryAndSetBounds(unittest.TestCase):
    """Tests for LiRPANet.build_history_and_set_bounds method."""

    @requires_cuda
    def test_build_history_and_set_bounds_basic(self):
        """Test basic build_history_and_set_bounds functionality."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)
        net.get_split_nodes()

        # Create domain dict
        d = {
            'lower_bounds': {net.final_name: torch.randn(2, 5)},
            'upper_bounds': {net.final_name: torch.randn(2, 5) + 1},
        }

        # Create split dict
        split = {
            'decision': torch.tensor([[0, 1]]),
        }

        # Should not raise
        net.build_history_and_set_bounds(d, split)


class TestBatchHandlerWithReferenceAlphas(unittest.TestCase):
    """Tests for BatchHandler with reference_alphas."""

    def test_init_with_reference_alphas(self):
        """Test BatchHandler with reference_alphas."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        reference_alphas = {
            '/relu': {
                'alpha': {'/output': torch.randn(2, 3, 4, 20)},
                'alpha_lookup_idx': torch.tensor([0, 1])
            }
        }

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=4,  # Single batch for alpha
            final_name='/output',
            need_alphas=True,
            full_alpha_info=True,
            reference_alphas=reference_alphas
        )

        self.assertIsNotNone(handler.reference_alphas)

    def test_get_batch_input_with_reference_alphas(self):
        """Test get_batch_input processes reference_alphas correctly."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        reference_alphas = {
            '/relu': {
                'alpha': {'/output': torch.randn(2, 3, 4, 20)},
                'alpha_lookup_idx': torch.tensor([0, 1])
            }
        }

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=4,
            final_name='/output',
            need_alphas=True,
            full_alpha_info=True,
            reference_alphas=reference_alphas
        )

        _, _, _, _, _, batch_alphas, _ = handler.get_batch_input(0, 'cpu')

        self.assertIsNotNone(batch_alphas)
        self.assertIn('/relu', batch_alphas)


class TestBatchHandlerWithReferenceLa(unittest.TestCase):
    """Tests for BatchHandler with reference_lA."""

    @requires_cuda
    def test_init_with_reference_la(self):
        """Test BatchHandler with reference_lA."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        reference_lA = {'/relu': torch.randn(4, 5, 20)}

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False,
            reference_lA=reference_lA
        )

        self.assertIsNotNone(handler.reference_lA)

    @requires_cuda
    def test_get_results_uses_reference_la_when_la_none(self):
        """Test get_results uses reference_lA when batch_lA is None."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any
        reference_lA = {'/relu': torch.randn(4, 5, 20)}

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=4,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False,
            reference_lA=reference_lA
        )

        lb = {'/output': torch.randn(4, 5)}
        ub = {'/output': torch.randn(4, 5)}
        mask = {'/relu': [torch.ones(4, 20, dtype=torch.bool)]}
        input_split_idx = {}
        handler.add_batch_result(lb, ub, None, None, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertIs(results['lA'], reference_lA)


class TestBatchHandlerGetResultsWithAlphas(unittest.TestCase):
    """Tests for BatchHandler.get_results with alpha handling."""

    @requires_cuda
    def test_get_results_with_alphas_full_info_false(self):
        """Test get_results alpha concatenation when full_alpha_info=False."""
        # Create x with uniform bounds (same x range) to allow multiple batches with alphas
        data = torch.zeros(4, 10)  # Uniform data
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=True,
            full_alpha_info=False
        )

        # Add two batches with alphas
        for i in range(2):
            lb = {'/output': torch.randn(2, 5)}
            ub = {'/output': torch.randn(2, 5)}
            alphas = {'/relu': {'/output': torch.randn(2, 3, 2, 20)}}
            mask = {'/relu': [torch.ones(2, 20, dtype=torch.bool)]}
            input_split_idx = {}
            handler.add_batch_result(lb, ub, None, alphas, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertIsNotNone(results['alphas'])
        # Alpha should be concatenated along dim 2
        self.assertEqual(results['alphas']['/relu']['/output'].shape[2], 4)

    @requires_cuda
    def test_get_results_with_alphas_full_info_true(self):
        """Test get_results alpha handling when full_alpha_info=True."""
        # Create x with uniform bounds (same x range) to allow multiple batches with alphas
        data = torch.zeros(4, 10)  # Uniform data
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=True,
            full_alpha_info=True
        )

        for i in range(2):
            lb = {'/output': torch.randn(2, 5)}
            ub = {'/output': torch.randn(2, 5)}
            alphas = {
                '/relu': {
                    'alpha': {'/output': torch.randn(2, 3, 2, 20)},
                    'alpha_lookup_idx': torch.tensor([0, 1])
                }
            }
            mask = {'/relu': [torch.ones(2, 20, dtype=torch.bool)]}
            input_split_idx = {}
            handler.add_batch_result(lb, ub, None, alphas, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertIsNotNone(results['alphas'])
        self.assertIn('alpha', results['alphas']['/relu'])
        # Alpha should be concatenated along dim 2
        self.assertEqual(results['alphas']['/relu']['alpha']['/output'].shape[2], 4)


class TestBatchHandlerMaskWithNone(unittest.TestCase):
    """Tests for BatchHandler mask handling with None elements."""

    @requires_cuda
    def test_get_results_with_none_mask_elements(self):
        """Test get_results properly handles None elements in mask."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        for i in range(2):
            lb = {'/output': torch.randn(2, 5)}
            ub = {'/output': torch.randn(2, 5)}
            mask = {'/relu': [None]}  # None mask element
            input_split_idx = {}
            handler.add_batch_result(lb, ub, None, None, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertEqual(results['mask']['/relu'], [None])


class TestLiRPANetWithMIP(unittest.TestCase):
    """Tests for LiRPANet MIP-related attributes."""

    @requires_cuda
    def test_mip_building_proc_initialized_none(self):
        """Test mip_building_proc is initialized to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.mip_building_proc)

    @requires_cuda
    def test_processes_initialized_none(self):
        """Test processes is initialized to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.processes)

    @requires_cuda
    def test_pool_attributes_initialized_none(self):
        """Test pool attributes are initialized to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.pool)
        self.assertIsNone(net.pool_result)
        self.assertIsNone(net.pool_termination_flag)


class TestLiRPANetDomainClipper(unittest.TestCase):
    """Tests for LiRPANet domain_clipper attribute."""

    @requires_cuda
    def test_domain_clipper_initialized_none(self):
        """Test domain_clipper is initialized to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.domain_clipper)


class TestLiRPANetBiccos(unittest.TestCase):
    """Tests for LiRPANet BICCOS attribute."""

    @requires_cuda
    def test_biccos_initialized_none(self):
        """Test biccos is initialized to None."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNone(net.biccos)


class TestLiRPANetRoot(unittest.TestCase):
    """Tests for LiRPANet root attribute."""

    @requires_cuda
    def test_root_is_set(self):
        """Test that root is properly set from net."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNotNone(net.root)


class TestBatchHandlerDeviceHandling(unittest.TestCase):
    """Tests for BatchHandler device handling."""

    def test_device_from_config(self):
        """Test device is taken from config when not storing on CPU."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        # Device should be set from config
        self.assertIsNotNone(handler.device)

    def test_device_cpu_when_store_on_cpu(self):
        """Test device is CPU when store_all_specs_on_cpu is True."""
        orig_value = arguments.Config['general']['store_all_specs_on_cpu']
        arguments.Config['general']['store_all_specs_on_cpu'] = True

        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(handler.device, 'cpu')

        arguments.Config['general']['store_all_specs_on_cpu'] = orig_value


class TestBatchHandlerTotalBatches(unittest.TestCase):
    """Tests for BatchHandler total_batches calculation."""

    def test_total_batches_exact_division(self):
        """Test total_batches when batch_size_ori divides evenly."""
        x = create_bounded_tensor((6, 10))
        c = torch.randn(6, 3, 5)
        rhs = torch.randn(6, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(handler.total_batches, 3)

    def test_total_batches_with_remainder(self):
        """Test total_batches when there's a remainder."""
        x = create_bounded_tensor((5, 10))
        c = torch.randn(5, 3, 5)
        rhs = torch.randn(5, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertEqual(handler.total_batches, 3)  # ceil(5/2) = 3


class TestLiRPANetInputShape(unittest.TestCase):
    """Tests for LiRPANet input_shape attribute."""

    @requires_cuda
    def test_input_shape_stored_correctly(self):
        """Test that input_shape is stored correctly."""
        model = SimpleModel()
        in_size = (2, 10)
        net = LiRPANet(model, in_size)

        self.assertEqual(net.input_shape, (2, 10))

    @requires_cuda
    def test_input_shape_4d(self):
        """Test input_shape for 4D input (CNN)."""
        model = ConvModel()
        in_size = (1, 1, 8, 8)
        net = LiRPANet(model, in_size)

        self.assertEqual(net.input_shape, (1, 1, 8, 8))


class TestLiRPANetFinalName(unittest.TestCase):
    """Tests for LiRPANet final_name attribute."""

    @requires_cuda
    def test_final_name_is_set(self):
        """Test that final_name is properly set."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        self.assertIsNotNone(net.final_name)
        self.assertIsInstance(net.final_name, str)


class TestBatchHandlerOptimizeDisjunctsSeparately(unittest.TestCase):
    """Tests for BatchHandler optimize_disjuncts_separately handling."""

    def test_optimize_disjuncts_separately_batch_any(self):
        """Test optimize_disjuncts_separately with stop_criterion_batch_any."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion_batch_any,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        self.assertTrue(handler.optimize_disjuncts_separately)


class TestBatchHandlerInputSplitIdx(unittest.TestCase):
    """Tests for BatchHandler input_split_idx handling."""

    @requires_cuda
    def test_input_split_idx_concatenated(self):
        """Test that input_split_idx is properly concatenated."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        for i in range(2):
            lb = {'/output': torch.randn(2, 5)}
            ub = {'/output': torch.randn(2, 5)}
            mask = {'/relu': [torch.ones(2, 20, dtype=torch.bool)]}
            input_split_idx = {'/input': torch.tensor([i, i])}
            handler.add_batch_result(lb, ub, None, None, mask, input_split_idx, None)

        history = {'/relu': ([], [], [], [], [])}
        results = handler.get_results(history)

        self.assertEqual(results['input_split_idx']['/input'].shape[0], 4)


class TestLiRPANetNetAttribute(unittest.TestCase):
    """Tests for LiRPANet.net attribute (BoundedModule)."""

    @requires_cuda
    def test_net_is_in_eval_mode(self):
        """Test that net is in evaluation mode."""
        model = SimpleModel()
        in_size = (1, 10)
        lirpa_net = LiRPANet(model, in_size)

        # BoundedModule should be in eval mode
        self.assertFalse(lirpa_net.net.training)


class TestBatchHandlerCurrRhs(unittest.TestCase):
    """Tests for BatchHandler.curr_rhs attribute."""

    def test_curr_rhs_set_on_get_batch_input(self):
        """Test that curr_rhs is set when get_batch_input is called."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False
        )

        _, _, batch_rhs, _, _, _, _ = handler.get_batch_input(0, 'cpu')

        self.assertIsNotNone(handler.curr_rhs)
        self.assertTrue(torch.equal(handler.curr_rhs, batch_rhs))


class TestLiRPANetGetPrimalUpperBound(unittest.TestCase):
    """Tests for LiRPANet.get_primal_upper_bound method."""

    def test_get_primal_upper_bound_linf(self):
        """Test get_primal_upper_bound with Linf perturbation."""
        model = SimpleModel(output_size=2)
        in_size = (1, 10)
        net = LiRPANet(model, in_size, device='cpu')

        # Setup x with Linf perturbation
        data = torch.randn(2, 10)
        x_L = data - 0.1
        x_U = data + 0.1
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
        net.x = BoundedTensor(data, ptb)

        # Setup C matrix
        net.c = torch.tensor([[[1, -1]]]).float()  # Shape: (1, 1, 2)

        # Create A matrix structure
        A = {
            net.net.output_name[0]: {
                net.net.input_name[0]: {
                    'lA': torch.randn(2, 1, 10)  # (batch, spec, input_dim)
                }
            }
        }

        primal_x, ub = net.get_primal_upper_bound(A)

        self.assertEqual(primal_x.shape[0], 2)
        self.assertEqual(primal_x.shape[1], 10)

    def test_get_primal_upper_bound_non_linf_raises(self):
        """Test that get_primal_upper_bound raises for non-Linf norm."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size, device='cpu')

        # Setup x with L2 perturbation
        data = torch.randn(2, 10)
        ptb = PerturbationLpNorm(norm=2, eps=0.1)
        net.x = BoundedTensor(data, ptb)

        A = {
            net.net.output_name[0]: {
                net.net.input_name[0]: {'lA': torch.randn(2, 1, 10)}
            }
        }

        with self.assertRaises(AssertionError):
            net.get_primal_upper_bound(A)


class TestLiRPANetASaved(unittest.TestCase):
    """Tests for LiRPANet.A_saved attribute."""

    @requires_cuda
    def test_A_saved_not_set_initially(self):
        """Test that A_saved is not set after initialization."""
        model = SimpleModel()
        in_size = (1, 10)
        net = LiRPANet(model, in_size)

        # A_saved should not exist or be None before build()
        self.assertFalse(hasattr(net, 'A_saved') and net.A_saved is not None)


class TestBatchHandlerClipInAlphaCrown(unittest.TestCase):
    """Tests for BatchHandler clip_in_alpha_crown handling."""

    def test_clip_in_alpha_crown_sets_optimize_separately(self):
        """Test that clip_in_alpha_crown=True sets optimize_disjuncts_separately."""
        x = create_bounded_tensor((4, 10))
        c = torch.randn(4, 3, 5)
        rhs = torch.randn(4, 3)
        stop_criterion = stop_criterion_batch_any

        handler = BatchHandler(
            x=x, c=c, rhs=rhs,
            stop_criterion=stop_criterion,
            or_spec_size=None,
            interm_bounds=None,
            batch_size_target=2,
            final_name='/output',
            need_alphas=False,
            full_alpha_info=False,
            clip_in_alpha_crown=True
        )

        self.assertTrue(handler.optimize_disjuncts_separately)


if __name__ == '__main__':
    unittest.main()
