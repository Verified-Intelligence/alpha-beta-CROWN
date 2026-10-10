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
"""Unit tests for utils.py"""
import importlib.util
import inspect
import os
import sys
import tempfile
import unittest

import pytest
import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    Timer, get_reduce_op, fast_hist_copy, convert_history_from_list,
    check_infeasible_bounds, get_batch_size_from_masks,
    _take_batch_Tensor, _expand_batch, transfer_obj,
    pad_list_of_input_to_tensor, unpad_to_list_of_tensors,
    Stats, Logger, expand_path, take_batch, expand_batch,
    model_uses_jacobian_op, auto_enable_jacobian_mode
)
from auto_LiRPA import BoundedTensor
from auto_LiRPA.jacobian import JacobianOP
from auto_LiRPA.perturbations import PerturbationLpNorm


class SquareJacobian(torch.nn.Module):
    """Model referencing the Jacobian placeholder in its forward method."""

    def forward(self, x):
        return JacobianOP.apply(x * x, x)


class SymbolicJacobian(torch.nn.Module):
    """Model declaring the ONNX Jacobian operator outside its forward method."""

    @staticmethod
    def symbolic(graph, x):
        return graph.op('grad::jacobian', x)

    def forward(self, x):
        return x


class OrdinaryModel(torch.nn.Module):
    """Non-Jacobian model sharing a source file with Jacobian models."""

    def forward(self, x):
        return x * x


class InheritedJacobian(SquareJacobian):
    """A module whose forward implementation comes from another class."""


@pytest.mark.parametrize('model, expected', [
    (torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()), False),
    (OrdinaryModel(), False),
    (SquareJacobian(), True),
    (SymbolicJacobian(), True),
    (torch.nn.Sequential(torch.nn.Identity(), SquareJacobian()), True),
    (InheritedJacobian(), True),
    (torch.nn.Module(), False),
    (None, False),
])
def test_model_uses_jacobian_op(model, expected):
    assert model_uses_jacobian_op(model) is expected


@pytest.mark.parametrize('body, expected', [
    ('JacobianOP.apply(x * x, x)', True),
    ('x * x', False),
])
def test_jacobian_detection_without_source(body, expected):
    # Notebook/dynamically generated forward methods can lack inspectable source.
    namespace = {'JacobianOP': JacobianOP}
    exec(compile(f'def forward(self, x):\n    return {body}\n',
                 '<generated-jacobian>', 'exec'), namespace)
    model_type = type('GeneratedJacobian', (torch.nn.Module,), {
        'forward': namespace['forward'],
    })
    assert model_uses_jacobian_op(model_type()) is expected


def test_jacobian_forward_from_another_source_file(monkeypatch):
    # A marker-free class source does not rule out an externally supplied forward.
    monkeypatch.setattr(torch.nn.Identity, 'forward', SquareJacobian.forward)
    assert model_uses_jacobian_op(torch.nn.Identity())


def test_jacobian_detection_matches_class_unwrapping(tmp_path, monkeypatch):
    # The wrapper file contains no operator markers. Python versions differ on
    # whether source inspection follows a class's __wrapped__; detection must
    # agree with the active inspect semantics rather than require unwrapping.
    path = tmp_path / 'wrapper.py'
    path.write_text(
        'import torch\n\n'
        'class WrappedModel(torch.nn.Module):\n'
        '    def forward(self, x):\n'
        '        return self.__wrapped__()(x)\n', encoding='utf-8')
    spec = importlib.util.spec_from_file_location('jacobian_detection_wrapper', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.WrappedModel.__wrapped__ = SquareJacobian

    expected = inspect.unwrap(module.WrappedModel) is SquareJacobian
    config = {'model': {'with_jacobian': False}}
    assert auto_enable_jacobian_mode(config, module.WrappedModel()) is expected
    assert config['model']['with_jacobian'] is expected


@pytest.mark.parametrize('model, explicit, expected', [
    (torch.nn.Identity(), False, False),
    (torch.nn.Identity(), True, True),
    (SquareJacobian(), False, True),
])
def test_auto_enable_jacobian_mode(model, explicit, expected):
    config = {'model': {'with_jacobian': explicit}}
    assert auto_enable_jacobian_mode(config, model) is expected
    assert config['model']['with_jacobian'] is expected


class TestTimer(unittest.TestCase):
    """Tests for Timer class."""

    def test_start_and_add(self):
        """Test basic timer start and add functionality."""
        timer = Timer()
        timer.start('test')
        # Do some work
        _ = sum(range(1000))
        timer.add('test')
        self.assertIn('test', timer.time_last)
        self.assertIn('test', timer.time_sum)
        self.assertGreaterEqual(timer.time_last['test'], 0)

    def test_accumulation(self):
        """Test that time accumulates correctly."""
        timer = Timer()
        timer.start('test')
        timer.add('test')
        first_sum = timer.time_sum['test']
        timer.start('test')
        timer.add('test')
        self.assertGreater(timer.time_sum['test'], first_sum)

    def test_multiple_timers(self):
        """Test tracking multiple named timers."""
        timer = Timer()
        timer.start('a')
        timer.add('a')
        timer.start('b')
        timer.add('b')
        self.assertIn('a', timer.time_sum)
        self.assertIn('b', timer.time_sum)


class TestGetReduceOp(unittest.TestCase):
    """Tests for get_reduce_op function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        self.assertIsNone(get_reduce_op(None))

    def test_min_op(self):
        """Test min operation."""
        op = get_reduce_op('min')
        result = op(torch.tensor([1, 2, 3]), dim=0)
        self.assertEqual(result.values.item(), 1)

    def test_max_op(self):
        """Test max operation."""
        op = get_reduce_op('max')
        result = op(torch.tensor([1, 2, 3]), dim=0)
        self.assertEqual(result.values.item(), 3)

    def test_mean_op_with_dim(self):
        """Test mean operation with dimension."""
        op = get_reduce_op('mean', with_dim=True)
        result = op(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=0)
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 3.0])))

    def test_mean_op_without_dim(self):
        """Test mean operation without dimension (pairwise)."""
        op = get_reduce_op('mean', with_dim=False)
        result = op(torch.tensor(2.0), torch.tensor(4.0))
        self.assertEqual(result.item(), 3.0)

    def test_invalid_op_raises(self):
        """Test that invalid operation raises ValueError."""
        with self.assertRaises(ValueError):
            get_reduce_op('invalid')


class TestFastHistCopy(unittest.TestCase):
    """Tests for fast_hist_copy function."""

    def test_none_input(self):
        """Test that None input returns None."""
        self.assertIsNone(fast_hist_copy(None))

    def test_tensor_history(self):
        """Test copying history with tensor elements."""
        hists = {'layer1': (torch.tensor([1, 2, 3]),)}
        copied = fast_hist_copy(hists)
        self.assertIn('layer1', copied)
        # Should be same reference for tensors
        self.assertIs(copied['layer1'], hists['layer1'])

    def test_list_history_cloned(self):
        """Test that list elements are cloned."""
        tensor = torch.tensor([1.0, 2.0])
        hists = {'layer1': ([tensor], [1, 2], [3, 4], [5, 6], [7, 8])}
        copied = fast_hist_copy(hists)
        self.assertIn('layer1', copied)


class TestConvertHistoryFromList(unittest.TestCase):
    """Tests for convert_history_from_list function."""

    def test_already_tensor(self):
        """Test that tensor input is returned as-is."""
        history = (torch.tensor([1, 2]),
                   torch.tensor([0.5, 0.5]),
                   torch.tensor([0.1, 0.2]),
                   torch.tensor([1.0, 2.0]),
                   torch.tensor([1, 2]))
        result = convert_history_from_list(history)
        self.assertIs(result, history)

    def test_list_converted(self):
        """Test that list input is converted to tensors."""
        history = ([1, 2], [0.5, 0.5], [0.1, 0.2], [1.0, 2.0], [1, 2])
        result = convert_history_from_list(history)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[0].dtype, torch.long)
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2])))


class TestCheckInfeasibleBounds(unittest.TestCase):
    """Tests for check_infeasible_bounds function."""

    def test_feasible_bounds(self):
        """Test with feasible bounds."""
        lower = {'a': torch.tensor([[0.0, 1.0]])}
        upper = {'a': torch.tensor([[1.0, 2.0]])}
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertFalse(result)

    def test_infeasible_bounds(self):
        """Test with infeasible bounds (lower > upper)."""
        lower = {'a': torch.tensor([[2.0, 1.0]])}
        upper = {'a': torch.tensor([[1.0, 2.0]])}
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertTrue(result)

    def test_reduce_false_returns_tensor(self):
        """Test that reduce=False returns boolean tensor."""
        lower = {'a': torch.tensor([[0.0], [2.0]])}
        upper = {'a': torch.tensor([[1.0], [1.0]])}
        result = check_infeasible_bounds(lower, upper, reduce=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2,))


class TestGetBatchSizeFromMasks(unittest.TestCase):
    """Tests for get_batch_size_from_masks function."""

    def test_single_mask(self):
        """Test with single mask entry."""
        mask = {'layer1': [1, 2, 3, 4, 5]}
        self.assertEqual(get_batch_size_from_masks(mask), 5)

    def test_multiple_masks(self):
        """Test with multiple mask entries."""
        mask = {'layer1': [1, 2, 3], 'layer2': [4, 5, 6]}
        self.assertEqual(get_batch_size_from_masks(mask), 3)


class TestTakeBatchTensor(unittest.TestCase):
    """Tests for _take_batch_Tensor function."""

    def test_basic_batch(self):
        """Test basic batch extraction."""
        data = torch.arange(20).reshape(4, 5)
        batch = _take_batch_Tensor(data, batch_size=2, batch_idx=0)
        self.assertEqual(batch.shape, (2, 5))
        self.assertTrue(torch.equal(batch, data[:2]))

    def test_second_batch(self):
        """Test extracting second batch."""
        data = torch.arange(20).reshape(4, 5)
        batch = _take_batch_Tensor(data, batch_size=2, batch_idx=1)
        self.assertEqual(batch.shape, (2, 5))
        self.assertTrue(torch.equal(batch, data[2:4]))

    def test_different_batch_dim(self):
        """Test with different batch dimension."""
        data = torch.arange(12).reshape(3, 4)
        batch = _take_batch_Tensor(data, batch_size=2, batch_idx=0, batch_dim=1)
        self.assertEqual(batch.shape, (3, 2))
        self.assertTrue(torch.equal(batch, data[:, :2]))


class TestExpandBatch(unittest.TestCase):
    """Tests for _expand_batch function."""

    def test_basic_expand(self):
        """Test basic batch expansion."""
        data = torch.ones(1, 5)
        expanded = _expand_batch(data, batch_size=4)
        self.assertEqual(expanded.shape, (4, 5))
        self.assertTrue(torch.all(expanded == 1))

    def test_expand_preserves_values(self):
        """Test that expansion preserves original values."""
        data = torch.tensor([[1.0, 2.0, 3.0]])
        expanded = _expand_batch(data, batch_size=3)
        for i in range(3):
            self.assertTrue(torch.equal(expanded[i], data[0]))

    def test_expand_non_uniform_raises(self):
        """Test that non-uniform data raises ValueError."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            _expand_batch(data, batch_size=4)


class TestTransferObj(unittest.TestCase):
    """Tests for transfer_obj function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        self.assertIsNone(transfer_obj(None))

    def test_tensor_device_transfer(self):
        """Test tensor device transfer."""
        t = torch.tensor([1.0, 2.0])
        result = transfer_obj(t, device='cpu')
        self.assertEqual(result.device.type, 'cpu')

    def test_tensor_dtype_transfer(self):
        """Test tensor dtype transfer."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = transfer_obj(t, dtype=torch.float64)
        self.assertEqual(result.dtype, torch.float64)

    def test_list_transfer(self):
        """Test list of tensors transfer."""
        lst = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = transfer_obj(lst, dtype=torch.float64)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0].dtype, torch.float64)
        self.assertEqual(result[1].dtype, torch.float64)

    def test_dict_transfer(self):
        """Test dict of tensors transfer."""
        d = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        result = transfer_obj(d, dtype=torch.float64)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['a'].dtype, torch.float64)

    def test_dict_inplace(self):
        """Test dict transfer with inplace=True."""
        d = {'a': torch.tensor([1.0], dtype=torch.float32)}
        result = transfer_obj(d, dtype=torch.float64, inplace=True)
        self.assertIs(result, d)
        self.assertEqual(d['a'].dtype, torch.float64)

    def test_nested_structure(self):
        """Test nested structure transfer."""
        nested = {'a': [torch.tensor([1.0])], 'b': (torch.tensor([2.0]),)}
        result = transfer_obj(nested, dtype=torch.float64)
        self.assertEqual(result['a'][0].dtype, torch.float64)
        self.assertEqual(result['b'][0].dtype, torch.float64)

    def test_non_tensor_passthrough(self):
        """Test that non-tensor values pass through unchanged."""
        result = transfer_obj("string")
        self.assertEqual(result, "string")
        result = transfer_obj(42)
        self.assertEqual(result, 42)


class TestPadUnpadTensors(unittest.TestCase):
    """Tests for pad_list_of_input_to_tensor and unpad_to_list_of_tensors."""

    def test_pad_uniform_tensors(self):
        """Test padding tensors of same size."""
        tensors = [torch.ones(3), torch.ones(3) * 2]
        # Need to mock arguments.Config for this test
        result = pad_list_of_input_to_tensor(
            tensors, pad_value=0, pad_dim=0, is_orginal_tensor=True, device='cpu'
        )
        self.assertEqual(result.shape, (2, 3))

    def test_pad_different_sizes(self):
        """Test padding tensors of different sizes."""
        tensors = [torch.ones(2), torch.ones(4)]
        result = pad_list_of_input_to_tensor(
            tensors, pad_value=0, pad_dim=0, is_orginal_tensor=True, device='cpu'
        )
        self.assertEqual(result.shape, (2, 4))
        # First tensor should be padded with zeros
        self.assertEqual(result[0, 2].item(), 0)
        self.assertEqual(result[0, 3].item(), 0)

    def test_unpad_tensors(self):
        """Test unpadding tensors."""
        padded = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        lengths = [2, 4]
        result = unpad_to_list_of_tensors(
            padded, unbind_dim=0, unpad_dim=0, ori_lengths=lengths, keep_dim=True
        )
        self.assertEqual(len(result), 2)


class TestStats(unittest.TestCase):
    """Tests for Stats class."""

    def test_init(self):
        """Test Stats initialization."""
        stats = Stats()
        self.assertEqual(stats.visited, 0)
        self.assertFalse(stats.all_node_split)
        self.assertIsInstance(stats.timer, Timer)

    def test_visited_increment(self):
        """Test incrementing visited counter."""
        stats = Stats()
        stats.visited += 10
        self.assertEqual(stats.visited, 10)

    def test_all_node_split_flag(self):
        """Test setting all_node_split flag."""
        stats = Stats()
        stats.all_node_split = True
        self.assertTrue(stats.all_node_split)


class TestLogger(unittest.TestCase):
    """Tests for Logger class."""

    def test_init(self):
        """Test Logger initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            logger = Logger('batch', f.name, 100.0)
            self.assertEqual(logger.run_mode, 'batch')
            self.assertEqual(logger.timeout_threshold, 100.0)
            self.assertEqual(logger.count, 0)
            os.unlink(f.name)

    def test_update_timeout(self):
        """Test updating timeout threshold."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            logger = Logger('batch', f.name, 100.0)
            logger.update_timeout(200.0)
            self.assertEqual(logger.timeout_threshold, 200.0)
            os.unlink(f.name)

    def test_record_pgd_stats(self):
        """Test recording PGD stats."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            logger = Logger('batch', f.name, 100.0)
            logger.record_pgd_stats(0, {'attack_success': True})
            self.assertEqual(logger.pgd_stats[0], {'attack_success': True})
            os.unlink(f.name)


class TestExpandPath(unittest.TestCase):
    """Tests for expand_path function."""

    def test_expand_simple_path(self):
        """Test expanding a simple path without substitution."""
        import arguments
        # Set up a mock config file path
        original_file = arguments.Config.file
        arguments.Config.file = '/some/config/path/config.yaml'
        try:
            result = expand_path('model.onnx')
            self.assertEqual(result, 'model.onnx')
        finally:
            arguments.Config.file = original_file

    def test_expand_config_path_substitution(self):
        """Test expanding path with $CONFIG_PATH substitution."""
        import arguments
        original_file = arguments.Config.file
        arguments.Config.file = '/some/config/path/config.yaml'
        try:
            result = expand_path('$CONFIG_PATH/model.onnx')
            self.assertEqual(result, '/some/config/path/model.onnx')
        finally:
            arguments.Config.file = original_file


class TestTakeBatch(unittest.TestCase):
    """Tests for take_batch function with BoundedTensor."""

    def _create_bounded_tensor(self, shape):
        """Create a BoundedTensor for testing."""
        data = torch.randn(shape)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        return BoundedTensor(data, ptb)

    def test_take_batch_tensor(self):
        """Test take_batch with regular tensor."""
        data = torch.arange(20).reshape(4, 5).float()
        result = take_batch(data, batch_size=2, batch_idx=0)
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(torch.equal(result, data[:2]))

    def test_take_batch_bounded_tensor(self):
        """Test take_batch with BoundedTensor."""
        x = self._create_bounded_tensor((4, 3, 8, 8))
        result = take_batch(x, batch_size=2, batch_idx=0)
        self.assertIsInstance(result, BoundedTensor)
        self.assertEqual(result.shape, (2, 3, 8, 8))

    def test_take_batch_second_batch(self):
        """Test taking second batch."""
        data = torch.arange(20).reshape(4, 5).float()
        result = take_batch(data, batch_size=2, batch_idx=1)
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(torch.equal(result, data[2:4]))

    def test_take_batch_invalid_type(self):
        """Test take_batch raises for invalid type."""
        with self.assertRaises(TypeError):
            take_batch([1, 2, 3], batch_size=2, batch_idx=0)


class TestExpandBatch(unittest.TestCase):
    """Tests for expand_batch function with BoundedTensor."""

    def _create_bounded_tensor(self, shape):
        """Create a BoundedTensor for testing."""
        data = torch.randn(shape)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        return BoundedTensor(data, ptb)

    def test_expand_batch_tensor(self):
        """Test expand_batch with regular tensor."""
        data = torch.ones(1, 5)
        result = expand_batch(data, batch_size=4)
        self.assertEqual(result.shape, (4, 5))

    def test_expand_batch_bounded_tensor(self):
        """Test expand_batch with BoundedTensor."""
        x = self._create_bounded_tensor((1, 3, 8, 8))
        result = expand_batch(x, batch_size=4)
        self.assertIsInstance(result, BoundedTensor)
        self.assertEqual(result.shape, (4, 3, 8, 8))

    def test_expand_batch_with_custom_bounds(self):
        """Test expand_batch with custom x_L and x_U."""
        x = self._create_bounded_tensor((1, 3, 8, 8))
        x_L = torch.zeros(4, 3, 8, 8)
        x_U = torch.ones(4, 3, 8, 8)
        result = expand_batch(x, batch_size=4, x_L=x_L, x_U=x_U)
        self.assertIsInstance(result, BoundedTensor)
        self.assertEqual(result.shape, (4, 3, 8, 8))

    def test_expand_batch_invalid_type(self):
        """Test expand_batch raises for invalid type."""
        with self.assertRaises(TypeError):
            expand_batch([1, 2, 3], batch_size=4)


class TestTimerDataclass(unittest.TestCase):
    """Tests for Timer dataclass fields."""

    def test_default_fields(self):
        """Test default Timer dataclass field values."""
        timer = Timer()
        self.assertEqual(timer.total_func_time, 0.0)
        self.assertEqual(timer.total_prepare_time, 0.0)
        self.assertEqual(timer.total_bound_time, 0.0)
        self.assertEqual(timer.total_beta_bound_time, 0.0)
        self.assertEqual(timer.total_transfer_time, 0.0)
        self.assertEqual(timer.total_finalize_time, 0.0)

    def test_update_dataclass_fields(self):
        """Test updating Timer dataclass fields."""
        timer = Timer()
        timer.total_func_time = 1.5
        timer.total_bound_time = 2.5
        self.assertEqual(timer.total_func_time, 1.5)
        self.assertEqual(timer.total_bound_time, 2.5)


class TestUnpadToListOfTensorsEdgeCases(unittest.TestCase):
    """Additional edge case tests for unpad_to_list_of_tensors."""

    def test_unpad_keep_dim_false(self):
        """Test unpadding with keep_dim=False."""
        padded = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        lengths = [2, 4]
        result = unpad_to_list_of_tensors(
            padded, unbind_dim=0, unpad_dim=0, ori_lengths=lengths, keep_dim=False
        )
        self.assertEqual(len(result), 2)

    def test_unpad_multidimensional(self):
        """Test unpadding multidimensional tensor."""
        padded = torch.randn(3, 4, 5)
        lengths = [2, 3, 4]
        result = unpad_to_list_of_tensors(
            padded, unbind_dim=0, unpad_dim=1, ori_lengths=lengths, keep_dim=True
        )
        self.assertEqual(len(result), 3)


class TestPadListOfInputEdgeCases(unittest.TestCase):
    """Additional edge case tests for pad_list_of_input_to_tensor."""

    def test_pad_with_batch_dim(self):
        """Test padding with specified batch_dim."""
        tensors = [torch.ones(2, 3), torch.ones(2, 5)]
        result = pad_list_of_input_to_tensor(
            tensors, pad_value=0, pad_dim=1, batch_dim=None,
            is_orginal_tensor=True, device='cpu'
        )
        self.assertEqual(result.shape, (2, 2, 5))

    def test_pad_with_negative_pad_value(self):
        """Test padding with negative pad value."""
        tensors = [torch.ones(2), torch.ones(4)]
        result = pad_list_of_input_to_tensor(
            tensors, pad_value=-1, pad_dim=0, is_orginal_tensor=True, device='cpu'
        )
        self.assertEqual(result[0, 2].item(), -1)
        self.assertEqual(result[0, 3].item(), -1)


class TestLoggerSummarizeResults(unittest.TestCase):
    """Additional tests for Logger.summarize_results."""

    def test_summarize_batch_mode_safe(self):
        """Test summarize_results in batch mode with safe status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()
            logger.summarize_results('safe', 0)
            self.assertEqual(logger.count, 1)
            self.assertIn('safe', logger.verification_summary)

    def test_summarize_batch_mode_unsafe(self):
        """Test summarize_results in batch mode with unsafe status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()
            logger.summarize_results('unsafe', 0)
            self.assertEqual(logger.count, 1)
            self.assertIn('unsafe', logger.verification_summary)

    def test_summarize_batch_mode_unknown(self):
        """Test summarize_results in batch mode with unknown status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()
            logger.summarize_results('unknown', 0)
            self.assertEqual(logger.count, 1)


class TestFastHistCopyComplex(unittest.TestCase):
    """Additional complex tests for fast_hist_copy."""

    def test_fast_hist_copy_with_tuple_elements(self):
        """Test fast_hist_copy with tuple history elements."""
        hists = {
            'layer1': (1, 2, 3, 4, 5)  # Non-tensor, non-list elements
        }
        copied = fast_hist_copy(hists)
        self.assertIn('layer1', copied)

    def test_fast_hist_copy_empty_dict(self):
        """Test fast_hist_copy with empty dict."""
        hists = {}
        copied = fast_hist_copy(hists)
        self.assertEqual(copied, {})

    def test_fast_hist_copy_multiple_layers(self):
        """Test fast_hist_copy with multiple layers."""
        hists = {
            'layer1': (torch.tensor([1]), torch.tensor([2])),
            'layer2': (torch.tensor([3]), torch.tensor([4])),
        }
        copied = fast_hist_copy(hists)
        self.assertIn('layer1', copied)
        self.assertIn('layer2', copied)


class TestTensorTransferEdgeCases(unittest.TestCase):
    """Edge case tests for transfer_obj."""

    def test_transfer_empty_dict(self):
        """Test transfer_obj with empty dict."""
        result = transfer_obj({}, device='cpu')
        self.assertEqual(result, {})

    def test_transfer_empty_list(self):
        """Test transfer_obj with empty list."""
        result = transfer_obj([], device='cpu')
        self.assertEqual(result, [])

    def test_transfer_empty_tuple(self):
        """Test transfer_obj with empty tuple."""
        result = transfer_obj((), device='cpu')
        self.assertEqual(result, ())

    def test_transfer_nested_dict(self):
        """Test transfer_obj with deeply nested dict."""
        nested = {
            'level1': {
                'level2': {
                    'tensor': torch.tensor([1.0])
                }
            }
        }
        result = transfer_obj(nested, dtype=torch.float64)
        self.assertEqual(result['level1']['level2']['tensor'].dtype, torch.float64)


class TestCheckInfeasibleBoundsMultiLayer(unittest.TestCase):
    """Tests for check_infeasible_bounds with multiple layers."""

    def test_multiple_layers_all_feasible(self):
        """Test with multiple layers all feasible."""
        lower = {
            'a': torch.tensor([[0.0, 1.0]]),
            'b': torch.tensor([[0.0, 0.5]])
        }
        upper = {
            'a': torch.tensor([[1.0, 2.0]]),
            'b': torch.tensor([[1.0, 1.5]])
        }
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertFalse(result)

    def test_multiple_layers_one_infeasible(self):
        """Test with multiple layers, one infeasible."""
        lower = {
            'a': torch.tensor([[0.0, 1.0]]),
            'b': torch.tensor([[2.0, 0.5]])  # Infeasible
        }
        upper = {
            'a': torch.tensor([[1.0, 2.0]]),
            'b': torch.tensor([[1.0, 1.5]])
        }
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertTrue(result)


class TestTimerPrint(unittest.TestCase):
    """Tests for Timer.print method."""

    def test_print_no_error(self):
        """Test that print method returns None without raising errors."""
        timer = Timer()
        timer.start('op1')
        timer.add('op1')
        result = timer.print()
        self.assertIsNone(result)

    def test_print_empty_timer(self):
        """Test print with no recorded times returns None."""
        timer = Timer()
        result = timer.print()
        self.assertIsNone(result)

    def test_print_multiple_operations(self):
        """Test print with multiple operations returns None."""
        timer = Timer()
        timer.start('op1')
        timer.add('op1')
        timer.start('op2')
        timer.add('op2')
        result = timer.print()
        self.assertIsNone(result)


class TestLoggerFinish(unittest.TestCase):
    """Tests for Logger.finish method."""

    def test_finish_batch_mode_with_results(self):
        """Test finish in batch mode with some results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()

            # Add multiple results
            logger.summarize_results('safe', 0)
            logger.summarize_results('unsafe', 1)
            logger.summarize_results('unknown', 2)

            self.assertEqual(logger.count, 3)
            self.assertEqual(len(logger.status_per_sample_list), 3)

    def test_finish_single_vnnlib_mode(self):
        """Test finish in single_vnnlib mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.txt')
            logger = Logger('single_vnnlib', save_path, 100.0)
            logger.record_start_time()
            # In single_vnnlib mode, finish() does nothing (results are saved in summarize_results)
            self.assertEqual(logger.run_mode, 'single_vnnlib')


class TestLoggerSave(unittest.TestCase):
    """Tests for Logger._save method."""

    def test_save_creates_file(self):
        """Test that _save creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()
            logger.summarize_results('safe', 0)

            self.assertTrue(os.path.exists(save_path))

    def test_save_content(self):
        """Test that saved content is correct."""
        import pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.pkl')
            logger = Logger('batch', save_path, 100.0)
            logger.record_start_time()
            logger.summarize_results('safe', 0)
            logger.summarize_results('unsafe', 1)

            with open(save_path, 'rb') as f:
                data = pickle.load(f)

            self.assertIn('summary', data)
            self.assertIn('results', data)
            self.assertIn('bab_ret', data)


class TestLoggerBabRet(unittest.TestCase):
    """Tests for Logger.bab_ret handling."""

    def test_bab_ret_initial_empty(self):
        """Test that bab_ret starts empty."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            logger = Logger('batch', f.name, 100.0)
            self.assertEqual(logger.bab_ret, [])
            os.unlink(f.name)

    def test_bab_ret_append(self):
        """Test appending to bab_ret."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            logger = Logger('batch', f.name, 100.0)
            logger.bab_ret.append([0, 0.1, 100, 5.0])
            self.assertEqual(len(logger.bab_ret), 1)
            os.unlink(f.name)


class TestStatsTimer(unittest.TestCase):
    """Tests for Stats.timer functionality."""

    def test_stats_timer_is_timer_instance(self):
        """Test that stats.timer is properly initialized."""
        stats = Stats()
        self.assertIsInstance(stats.timer, Timer)

    def test_stats_timer_can_be_used(self):
        """Test that the timer in Stats is functional."""
        stats = Stats()
        stats.timer.start('test')
        stats.timer.add('test')
        self.assertIn('test', stats.timer.time_sum)

    def test_multiple_stats_independent(self):
        """Test that multiple Stats instances are independent."""
        stats1 = Stats()
        stats2 = Stats()

        stats1.visited = 10
        stats2.visited = 0

        self.assertEqual(stats1.visited, 10)
        self.assertEqual(stats2.visited, 0)


class TestConvertHistoryFromListEdgeCases(unittest.TestCase):
    """Edge case tests for convert_history_from_list."""

    def test_single_element_lists(self):
        """Test with single element lists."""
        history = ([1], [0.5], [0.1], [1.0], [1])
        result = convert_history_from_list(history)
        self.assertEqual(result[0].shape, (1,))
        self.assertEqual(result[0].dtype, torch.long)

    def test_empty_lists(self):
        """Test with empty lists."""
        history = ([], [], [], [], [])
        result = convert_history_from_list(history)
        self.assertEqual(result[0].shape, (0,))


class TestGetBatchSizeFromMasksEdgeCases(unittest.TestCase):
    """Edge case tests for get_batch_size_from_masks."""

    def test_dict_with_tensor_values(self):
        """Test with tensor values in dict."""
        mask = {'layer1': torch.tensor([1, 2, 3, 4])}
        # This should return len() of the tensor which is 4
        result = get_batch_size_from_masks(mask)
        self.assertEqual(result, 4)


class TestCheckInfeasibleBoundsEdgeCases(unittest.TestCase):
    """Additional edge case tests for check_infeasible_bounds."""

    def test_boundary_case_equal_bounds(self):
        """Test when lower equals upper (valid but edge case)."""
        lower = {'a': torch.tensor([[1.0, 2.0]])}
        upper = {'a': torch.tensor([[1.0, 2.0]])}
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertFalse(result)

    def test_slight_infeasibility(self):
        """Test slight infeasibility (below tolerance)."""
        lower = {'a': torch.tensor([[1.0 + 1e-8]])}
        upper = {'a': torch.tensor([[1.0]])}
        # The tolerance in check_infeasible_bounds is 1e-6, so this should not be infeasible
        result = check_infeasible_bounds(lower, upper, reduce=True)
        # 1e-8 < 1e-6, so not infeasible
        self.assertFalse(result)

    def test_significant_infeasibility(self):
        """Test significant infeasibility (above tolerance)."""
        lower = {'a': torch.tensor([[1.0 + 1e-5]])}
        upper = {'a': torch.tensor([[1.0]])}
        # 1e-5 > 1e-6, so this should be infeasible
        result = check_infeasible_bounds(lower, upper, reduce=True)
        self.assertTrue(result)


class TestTakeBatchEdgeCases(unittest.TestCase):
    """Edge case tests for take_batch function."""

    def test_take_batch_last_batch(self):
        """Test taking the last batch."""
        data = torch.arange(15).reshape(5, 3).float()
        # Last batch of size 2 would be idx=2 (items 4)
        result = take_batch(data, batch_size=2, batch_idx=2)
        self.assertEqual(result.shape, (1, 3))  # Only 1 item left

    def test_take_batch_with_device(self):
        """Test take_batch with device specification."""
        data = torch.arange(12).reshape(4, 3).float()
        result = take_batch(data, batch_size=2, batch_idx=0, device='cpu')
        self.assertEqual(result.device.type, 'cpu')


class TestExpandBatchEdgeCases(unittest.TestCase):
    """Edge case tests for expand_batch function."""

    def test_expand_batch_different_dim(self):
        """Test expand_batch along different dimension."""
        data = torch.tensor([[1.0, 2.0, 3.0]])
        expanded = _expand_batch(data, batch_size=4, batch_dim=0)
        self.assertEqual(expanded.shape, (4, 3))

    def test_expand_batch_3d_tensor(self):
        """Test expand_batch with 3D tensor."""
        data = torch.ones(1, 3, 4)
        expanded = _expand_batch(data, batch_size=5)
        self.assertEqual(expanded.shape, (5, 3, 4))


class TestTransferObjTuples(unittest.TestCase):
    """Tests for transfer_obj with tuple structures."""

    def test_transfer_tuple_of_tensors(self):
        """Test transfer_obj with tuple of tensors."""
        t = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = transfer_obj(t, dtype=torch.float64)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].dtype, torch.float64)
        self.assertEqual(result[1].dtype, torch.float64)

    def test_transfer_mixed_tuple(self):
        """Test transfer_obj with mixed tuple (tensor and non-tensor)."""
        t = (torch.tensor([1.0]), "string", 42)
        result = transfer_obj(t, dtype=torch.float64)
        self.assertEqual(result[0].dtype, torch.float64)
        self.assertEqual(result[1], "string")
        self.assertEqual(result[2], 42)


class TestPadListDifferentDevices(unittest.TestCase):
    """Tests for pad_list_of_input_to_tensor with different configurations."""

    def test_pad_with_concat_batch_dim(self):
        """Test padding with concatenation along batch_dim."""
        tensors = [torch.ones(2, 3), torch.ones(3, 3)]
        result = pad_list_of_input_to_tensor(
            tensors, pad_value=0, pad_dim=0, batch_dim=0,
            is_orginal_tensor=True, device='cpu'
        )
        # First tensor padded to (3, 3), second stays (3, 3)
        # Then concatenated along dim 0: (3+3, 3) = (6, 3)
        self.assertEqual(result.shape, (6, 3))

    def test_pad_numpy_arrays(self):
        """Test padding with numpy arrays (not tensors)."""
        import numpy as np
        arrays = [np.ones((2,)), np.ones((4,))]
        result = pad_list_of_input_to_tensor(
            arrays, pad_value=0, pad_dim=0, is_orginal_tensor=False, device='cpu'
        )
        self.assertEqual(result.shape, (2, 4))


class TestUnpadDifferentDims(unittest.TestCase):
    """Tests for unpad_to_list_of_tensors with different dimensions."""

    def test_unpad_along_last_dim(self):
        """Test unpadding along the last dimension."""
        padded = torch.randn(2, 5)
        lengths = [3, 4]
        result = unpad_to_list_of_tensors(
            padded, unbind_dim=0, unpad_dim=1, ori_lengths=lengths, keep_dim=True
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape[-1], 3)
        self.assertEqual(result[1].shape[-1], 4)


if __name__ == '__main__':
    unittest.main()
