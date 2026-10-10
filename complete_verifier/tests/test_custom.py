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
"""Unit tests for custom package modules."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom import custom_model_loader
from custom import custom_adhoc_tuning
from custom import custom_soundness_attack
from custom.custom_model_loader import conv_output_shape, transpose_linear_layers
from custom.custom_op import (
    LinearMaskedReluOp, LinearMaskedRelu
)


class TestNN4SysTuning(unittest.TestCase):
    """Large NN4Sys models cap the batch without increasing a user's limit."""

    def test_build_batch_upper_limit(self):
        model = mock.Mock()
        model.parameters.return_value = [SimpleNamespace(
            requires_grad=True, numel=lambda: 50_000_000)]
        handler = SimpleNamespace(all_specs=SimpleNamespace(batch_size=4150))
        for requested, expected in [(128, 128), (1000, 1000), (1000000, 1000)]:
            with self.subTest(requested=requested):
                config = {'solver': {'build_batch_size': requested}}
                with mock.patch.object(custom_adhoc_tuning.arguments, 'Config', config):
                    custom_adhoc_tuning.customized_nn4sys_tuning(model, handler)
                self.assertEqual(config['solver']['build_batch_size'], expected)


class TestSoundnessAttackBudget(unittest.TestCase):
    """Slow refinement leaves time for backstop without bypassing validation."""

    def test_refinement_respects_budget_and_returns_validated_success(self):
        for can_succeed in [False, True]:
            with self.subTest(can_succeed=can_succeed):
                clock = SimpleNamespace(now=0.)

                class SlowSurrogate(nn.Module):
                    def forward(self, value):
                        clock.now += 0.05
                        return value * 0 + 1e-7

                def replay(value):
                    clock.now += 0.05
                    # ORT becomes valid only in the reserved final portion;
                    # the differentiable surrogate never claims success.
                    output = -2e-7 if can_succeed and clock.now >= 9.5 else 0.
                    return np.full((len(value), 1), output, dtype=np.float32)

                config = {'attack': {'pgd_steps': 1, 'pgd_restarts': 2,
                                     'pgd_lr_decay': 0.99},
                          'general': {'seed': 100},
                          'specification': {'vnnlib_path': None}}
                module = custom_soundness_attack
                with mock.patch.object(module.arguments, 'Config', config), \
                     mock.patch.object(module, 'time', SimpleNamespace(time=lambda: clock.now)), \
                     mock.patch.object(module, '_ort_out', side_effect=replay):
                    ret = module.soundness_attack(
                        SlowSurrogate(), torch.zeros(1, 1), -torch.ones(1, 1),
                        torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1),
                        torch.ones(1, dtype=torch.long), time_budget=10.,
                        ils_hops=0, verbose=False)
                self.assertEqual(bool(ret[0].item()), can_succeed)
                self.assertLessEqual(clock.now, 10.5)
                self.assertTrue(torch.isfinite(ret[4]).all())
                if can_succeed:
                    self.assertTrue(torch.all(ret[4] <= -5e-8))

    def test_round_cutoff_preserves_candidates_within_total_budget(self):
        """Validate existing PGD candidates after the round, not total, cutoff."""
        class SlowLinearModel(nn.Module):
            def __init__(self, clock, pgd_seconds):
                super().__init__()
                self.clock = clock
                self.pgd_seconds = pgd_seconds

            def forward(self, value):
                # The PGD step and final candidate evaluation exhaust
                # the round budget; subsequent evaluations are cheap.
                self.clock.forwards += 1
                self.clock.now += (self.pgd_seconds / 2
                                   if self.clock.forwards <= 2 else 0.01)
                return 0.25 - value

        for pgd_seconds, can_succeed in [(9.1, True), (10.1, False)]:
            with self.subTest(pgd_seconds=pgd_seconds):
                clock = SimpleNamespace(now=0., forwards=0)

                def replay(value, current_clock=clock):
                    current_clock.now += 0.01
                    return (np.float32(0.25) - value).astype(np.float32)

                config = {'attack': {'pgd_steps': 1, 'pgd_restarts': 2,
                                     'pgd_lr_decay': 0.99},
                          'general': {'seed': 100},
                          'specification': {'vnnlib_path': None}}
                module = custom_soundness_attack
                with mock.patch.object(module.arguments, 'Config', config), \
                     mock.patch.object(module, 'time', SimpleNamespace(
                         time=lambda current_clock=clock: current_clock.now)), \
                     mock.patch.object(module, '_ort_out', side_effect=replay):
                    ret = module.soundness_attack(
                        SlowLinearModel(clock, pgd_seconds), torch.zeros(1, 1), -torch.ones(1, 1),
                        torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1),
                        torch.ones(1, dtype=torch.long), time_budget=10.,
                        ils_hops=0, drill_steps=0, verbose=False)
                self.assertEqual(bool(ret[0].item()), can_succeed)
                self.assertLessEqual(clock.now, pgd_seconds + 0.1)
                if can_succeed:
                    self.assertTrue(torch.all(ret[2] > 0.25))
                    self.assertTrue(torch.all(ret[4] <= -5e-8))


class TestConvOutputShapeCustom(unittest.TestCase):
    """Tests for conv_output_shape function in custom_model_loader."""

    def test_basic_conv(self):
        """Test basic convolution output shape."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 30)

    def test_same_padding(self):
        """Test convolution with same padding."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=1)
        self.assertEqual(h, 32)
        self.assertEqual(w, 32)

    def test_stride_2(self):
        """Test convolution with stride 2."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=2, pad=1)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_int_input(self):
        """Test with integer input instead of tuple."""
        h, w = conv_output_shape(h_w=32, kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 30)

    def test_dilation(self):
        """Test convolution with dilation."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=0, dilation=2)
        self.assertEqual(h, 28)
        self.assertEqual(w, 28)

    def test_asymmetric_input(self):
        """Test convolution with asymmetric input."""
        h, w = conv_output_shape(h_w=(28, 32), kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 26)
        self.assertEqual(w, 30)

    def test_1x1_conv(self):
        """Test 1x1 convolution."""
        h, w = conv_output_shape(h_w=(16, 16), kernel_size=1, stride=1, pad=0)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)


class TestTransposeLinearLayers(unittest.TestCase):
    """Tests for transpose_linear_layers function."""

    def test_transpose_single_linear(self):
        """Test transposing a single linear layer."""
        model = nn.Sequential(
            nn.Linear(10, 20)
        )
        model[0].weight.data = torch.randn(20, 10)
        model[0].bias.data = torch.randn(20)

        result = transpose_linear_layers(model)

        self.assertIsInstance(result, nn.Sequential)
        self.assertEqual(result[0].in_features, 10)
        self.assertEqual(result[0].out_features, 20)

    def test_transpose_multiple_linear(self):
        """Test transposing multiple linear layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        result = transpose_linear_layers(model)

        self.assertIsInstance(result, nn.Sequential)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], nn.Linear)
        self.assertIsInstance(result[1], nn.ReLU)
        self.assertIsInstance(result[2], nn.Linear)

    def test_transpose_preserves_activation(self):
        """Test that activation layers are preserved."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Sigmoid()
        )

        result = transpose_linear_layers(model)

        self.assertIsInstance(result[1], nn.ReLU)
        self.assertIsInstance(result[3], nn.Sigmoid)

    def test_transpose_with_no_bias(self):
        """Test transposing linear layer without bias."""
        model = nn.Sequential(
            nn.Linear(10, 20, bias=False)
        )
        model[0].bias = None

        result = transpose_linear_layers(model)

        self.assertIsInstance(result[0], nn.Linear)


class TestCustomizedTestLoader(unittest.TestCase):
    """Tests for the VNN-COMP test benchmark loader."""

    def test_vnnlib2_specs_use_vnnlib2_loader(self):
        """VNN-LIB 2.0 specs should not be sent to the legacy parser."""
        calls = {}
        original_model = nn.Identity()
        transformed_model = nn.Sequential(original_model, nn.ReLU())

        class DummyWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.models = nn.ModuleDict({"f": original_model})

        def fake_vnnlib2_loader(file_root, onnx_path, vnnlib_path):
            calls["args"] = (file_root, onnx_path, vnnlib_path)
            return DummyWrapper(), (-1, 1), ["vnnlib"]

        def fake_transpose(model):
            calls["transposed_model"] = model
            return transformed_model

        with mock.patch.object(
                custom_model_loader, "is_vnnlib2_file",
                return_value=True) as detect_vnnlib2, mock.patch.object(
                    custom_model_loader, "vnnlib2_onnx_and_vnnlib_loader",
                    side_effect=fake_vnnlib2_loader), mock.patch.object(
                        custom_model_loader, "transpose_linear_layers",
                        side_effect=fake_transpose), mock.patch.object(
                        custom_model_loader, "read_vnnlib",
                        side_effect=AssertionError(
                            "legacy parser should not be used")):
            model, shape, vnnlib = custom_model_loader.customized_TEST_loader(
                "/bench", "model.onnx", "spec.vnnlib")

        self.assertEqual(shape, (-1, 1))
        self.assertEqual(vnnlib, ["vnnlib"])
        detect_vnnlib2.assert_called_once_with(
            os.path.join("/bench", "spec.vnnlib"))
        self.assertEqual(calls["args"], ("/bench", "model.onnx", "spec.vnnlib"))
        self.assertIs(calls["transposed_model"], original_model)
        self.assertIs(model.models["f"], transformed_model)


class TestLinearMaskedReluOp(unittest.TestCase):
    """Tests for LinearMaskedReluOp custom autograd function."""

    def test_forward_all_relu(self):
        """Test forward pass with all ReLU (mask=0)."""
        input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        mask = torch.zeros(4)
        slope = torch.ones(4)
        bias = torch.zeros(4)

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        expected = torch.tensor([0.0, 0.0, 1.0, 2.0])  # ReLU output
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_all_linear(self):
        """Test forward pass with all linear (mask=1)."""
        input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        mask = torch.ones(4)
        slope = torch.tensor([2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        expected = torch.tensor([-1.0, 1.0, 3.0, 5.0])  # 2*x + 1
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_mixed(self):
        """Test forward pass with mixed mask."""
        input_tensor = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
        slope = torch.tensor([2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([0.5, 0.5, 0.5, 0.5])

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        # mask=0: ReLU(-1)=0, ReLU(1)=1
        # mask=1: 2*(-1)+0.5=-1.5, 2*1+0.5=2.5
        expected = torch.tensor([0.0, -1.5, 1.0, 2.5])
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_gradient_enabled(self):
        """Test that gradients can be computed."""
        input_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        mask = torch.zeros(2)
        slope = torch.ones(2)
        bias = torch.zeros(2)

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)


class TestLinearMaskedRelu(unittest.TestCase):
    """Tests for LinearMaskedRelu module."""

    def test_init_int_size(self):
        """Test initialization with integer size."""
        module = LinearMaskedRelu(10)

        self.assertEqual(module.mask.shape, (10,))
        self.assertEqual(module.slope.shape, (10,))
        self.assertEqual(module.bias.shape, (10,))

    def test_init_tuple_size(self):
        """Test initialization with tuple size."""
        module = LinearMaskedRelu((3, 4, 5))

        self.assertEqual(module.mask.shape, (3, 4, 5))
        self.assertEqual(module.slope.shape, (3, 4, 5))
        self.assertEqual(module.bias.shape, (3, 4, 5))

    def test_forward_shape(self):
        """Test that forward preserves shape."""
        module = LinearMaskedRelu(10)
        input_tensor = torch.randn(5, 10)

        result = module(input_tensor)

        self.assertEqual(result.shape, (5, 10))

    def test_forward_deterministic(self):
        """Test that forward is deterministic."""
        module = LinearMaskedRelu(10)
        input_tensor = torch.randn(5, 10)

        result1 = module(input_tensor)
        result2 = module(input_tensor)

        self.assertTrue(torch.allclose(result1, result2))

    def test_buffers_registered(self):
        """Test that mask, slope, bias are registered as buffers."""
        module = LinearMaskedRelu(10)

        buffer_names = [name for name, _ in module.named_buffers()]
        self.assertIn('mask', buffer_names)
        self.assertIn('slope', buffer_names)
        self.assertIn('bias', buffer_names)

    def test_no_parameters(self):
        """Test that module has no learnable parameters."""
        module = LinearMaskedRelu(10)

        params = list(module.parameters())
        self.assertEqual(len(params), 0)


class TestCustomOpIntegration(unittest.TestCase):
    """Integration tests for custom operations."""

    def test_linear_masked_relu_in_sequential(self):
        """Test LinearMaskedRelu in a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            LinearMaskedRelu(20),
            nn.Linear(20, 5)
        )

        input_tensor = torch.randn(3, 10)
        result = model(input_tensor)

        self.assertEqual(result.shape, (3, 5))

    def test_linear_masked_relu_conv_shapes(self):
        """Test LinearMaskedRelu with conv layer output shapes."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            LinearMaskedRelu((16, 32, 32)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )

        input_tensor = torch.randn(2, 3, 32, 32)
        result = model(input_tensor)

        self.assertEqual(result.shape, (2, 32, 32, 32))

    def test_backward_pass(self):
        """Test that backward pass works through LinearMaskedRelu."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            LinearMaskedRelu(20),
            nn.Linear(20, 5)
        )

        input_tensor = torch.randn(3, 10, requires_grad=True)
        result = model(input_tensor)
        loss = result.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(model[0].weight.grad)
        self.assertIsNotNone(model[2].weight.grad)


class TestConvOutputShapeEdgeCases(unittest.TestCase):
    """Edge case tests for conv_output_shape."""

    def test_large_kernel(self):
        """Test with kernel larger than input."""
        h, w = conv_output_shape(h_w=(5, 5), kernel_size=7, stride=1, pad=3)
        self.assertEqual(h, 5)
        self.assertEqual(w, 5)

    def test_large_stride(self):
        """Test with large stride."""
        h, w = conv_output_shape(h_w=(64, 64), kernel_size=7, stride=4, pad=3)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_asymmetric_kernel_tuple(self):
        """Test with asymmetric kernel as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=(3, 5), stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 28)

    def test_asymmetric_stride_tuple(self):
        """Test with asymmetric stride as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=(2, 1), pad=1)
        self.assertEqual(h, 16)
        self.assertEqual(w, 32)

    def test_asymmetric_pad_tuple(self):
        """Test with asymmetric padding as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=(0, 1))
        self.assertEqual(h, 30)
        self.assertEqual(w, 32)


if __name__ == '__main__':
    unittest.main()
