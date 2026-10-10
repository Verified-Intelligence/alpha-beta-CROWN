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
"""Unit tests for attack package."""
import os
import sys
import unittest
from unittest.mock import patch
import tempfile

import torch
import torch.nn as nn
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize arguments.Config before importing attack modules
import arguments
arguments.Config.parse_config(args=[], verbose=False)
from cex_validation import CEX_INPUT_TOLERANCE

from auto_LiRPA import BoundedTensor, BoundedModule
from auto_LiRPA.perturbations import PerturbationLpNorm

import attack.attack_utils as attack_utils_module
from attack.attack_utils import (
    AdamClipping, precompute_group_indices, process_data_for_attack,
    boundary_attack, default_adv_verifier, is_specification_vio,
    check_and_save_cex, default_adv_saver, satisfies_open_constraints,
    PGDAttackResult, verify_onnxruntime_output_matches_pytorch
)
from attack.adv_domains import AdvExample, AdvExamplePool


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


class TestAdamClipping(unittest.TestCase):
    """Tests for AdamClipping optimizer."""

    def test_init_valid_params(self):
        """Test AdamClipping initialization with valid parameters."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_init_rejects_invalid_parameters(self):
        """Test validation of the public optimizer parameters."""
        param = torch.randn(10, requires_grad=True)
        invalid_parameters = [
            {'lr': -0.01},
            {'eps': -1e-8},
            {'betas': (1.5, 0.999)},
            {'betas': (0.9, 1.5)},
            {'weight_decay': -0.1},
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    AdamClipping([param], **parameters)

    def test_step_without_clipping(self):
        """Test step without clipping."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)

        # Simulate a backward pass
        loss = param.sum()
        loss.backward()

        # Take a step
        optimizer.step()

        # Parameter should have been updated
        self.assertTrue(param.grad is not None)

    def test_step_with_clipping(self):
        """Test step with clipping."""
        param = torch.full((10,), 0.5, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.1)

        lower_limit = torch.zeros(10)
        upper_limit = torch.ones(10)

        # Simulate a backward pass
        loss = param.sum()
        loss.backward()

        # Take a step with clipping
        optimizer.step(clipping=True, lower_limit=lower_limit,
                       upper_limit=upper_limit, sign=1)

        # Parameter should be within bounds
        self.assertTrue((param >= lower_limit).all())
        self.assertTrue((param <= upper_limit).all())

    def test_setstate(self):
        """Test __setstate__ sets amsgrad default."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)

        # Get state and restore it
        state = optimizer.__getstate__()
        optimizer.__setstate__(state)

        # Check amsgrad is set
        self.assertIn('amsgrad', optimizer.param_groups[0])

    def test_step_with_amsgrad(self):
        """Test step with AMSGrad variant."""
        param = torch.randn(10, requires_grad=True)
        param_before = param.clone().detach()
        optimizer = AdamClipping([param], lr=0.01, amsgrad=True)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Verify parameter was updated (not equal to original)
        self.assertFalse(torch.equal(param.data, param_before),
            "Parameter should be updated after optimizer step with amsgrad")

        # Verify amsgrad state is maintained
        self.assertTrue(optimizer.param_groups[0]['amsgrad'],
            "amsgrad should be enabled in optimizer")

    def test_step_with_weight_decay(self):
        """Test step with weight decay."""
        param = torch.randn(10, requires_grad=True)
        param_before = param.clone().detach()
        optimizer = AdamClipping([param], lr=0.01, weight_decay=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Verify parameter was updated (not equal to original)
        self.assertFalse(torch.equal(param.data, param_before),
            "Parameter should be updated after optimizer step with weight_decay")

        # Verify weight_decay is set in optimizer
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01,
            "weight_decay should be set to 0.01")


class TestPrecomputeGroupIndices(unittest.TestCase):
    """Tests for precompute_group_indices function."""

    def test_uniform_groups(self):
        """Test with uniform group sizes."""
        or_spec_size = torch.tensor([3, 3, 3])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 3)
        self.assertEqual(group_indices.shape, (9,))
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(torch.equal(group_indices, expected))

    def test_varying_groups(self):
        """Test with varying group sizes."""
        or_spec_size = torch.tensor([2, 3, 1])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 3)
        self.assertEqual(group_indices.shape, (6,))
        expected = torch.tensor([0, 0, 1, 1, 1, 2])
        self.assertTrue(torch.equal(group_indices, expected))

    def test_single_group(self):
        """Test with single group."""
        or_spec_size = torch.tensor([5])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 1)
        self.assertEqual(group_indices.shape, (5,))
        expected = torch.tensor([0, 0, 0, 0, 0])
        self.assertTrue(torch.equal(group_indices, expected))


class TestProcessDataForAttack(unittest.TestCase):
    """Tests for process_data_for_attack function."""

    def test_expand_single_input(self):
        """Test expanding single input to multiple OR specs."""
        # Create bounded tensor with batch size 1
        data = torch.randn(1, 10)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        c = torch.randn(1, 6, 5)  # 6 specs total
        rhs = torch.randn(1, 6)
        or_spec_size = torch.tensor([2, 2, 2])  # 3 OR clauses, 2 AND each

        x_out, data_min, data_max, c_out, rhs_out = process_data_for_attack(
            x, c, rhs, or_spec_size
        )

        self.assertEqual(x_out.shape[0], 3)  # Expanded to num_or
        self.assertEqual(c_out.shape, (6, 5))  # Flattened specs
        self.assertEqual(rhs_out.shape, (6,))

    def test_matching_batch_and_or_size(self):
        """Test when batch size matches num_or."""
        num_or = 3
        data = torch.randn(num_or, 10)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        c = torch.randn(num_or, 2, 5)  # 2 AND per OR
        rhs = torch.randn(num_or, 2)
        or_spec_size = torch.tensor([2, 2, 2])

        x_out, data_min, data_max, c_out, rhs_out = process_data_for_attack(
            x, c, rhs, or_spec_size
        )

        self.assertEqual(x_out.shape[0], num_or)
        self.assertEqual(c_out.shape, (6, 5))


class TestVNNLib2StrictCounterexample:
    """Tests for strict VNN-LIB 2.0 checks at the cex save boundary."""

    def _read(self, tmp_path, text):
        from read_vnnlib2 import read_vnnlib2
        from specifications import VNNLibEntries

        path = tmp_path / "strict.vnnlib"
        path.write_text(text)
        entries, metadata = read_vnnlib2(str(path))
        return VNNLibEntries(entries, metadata)

    def _check_without_saving(self, adv_input, adv_output, vnnlib):
        with patch.dict(arguments.Config['general'], {
            'save_adv_example': False,
            'eval_adv_example': False,
        }):
            return check_and_save_cex(
                adv_input, adv_output, vnnlib, "unused.txt", "unsafe")

    def test_rejects_strict_input_boundary(self, tmp_path):
        vnnlib = self._read(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (> X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (<= Y[0] 0.0))
""")

        adv_input = torch.tensor([[0.0]])
        adv_output = torch.tensor([[-1.0]])

        assert not satisfies_open_constraints(adv_input, adv_output, vnnlib)
        status, success = self._check_without_saving(
            adv_input, adv_output, vnnlib)
        assert status == "unknown"
        assert not success

    def test_rejects_strict_output_boundary(self, tmp_path):
        vnnlib = self._read(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (> Y[0] 0.0))
""")

        adv_input = torch.tensor([[0.5]])
        adv_output = torch.tensor([[0.0]])

        status, success = self._check_without_saving(
            adv_input, adv_output, vnnlib)
        assert status == "unknown"
        assert not success

    def test_accepts_strict_counterexample_inside_region(self, tmp_path):
        vnnlib = self._read(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (> X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (> Y[0] 0.0))
""")

        adv_input = torch.tensor([[0.5]])
        adv_output = torch.tensor([[0.1]])

        status, success = self._check_without_saving(
            adv_input, adv_output, vnnlib)
        assert status == "unsafe"
        assert success


class TestCounterexampleSaving(unittest.TestCase):
    """Tests for the controlled counterexample saving path."""

    @staticmethod
    def _vnnlib():
        return [(
            [[0.0, 1.0]],
            [(np.array([[1.0]]), np.array([0.0]))],
        )]

    @staticmethod
    def _vnnlib2(temp_dir, text):
        from read_vnnlib2 import read_vnnlib2
        from specifications import VNNLibEntries

        path = os.path.join(temp_dir, 'property.vnnlib')
        with open(path, 'w', encoding='utf-8') as vnnlib_file:
            vnnlib_file.write(text)
        entries, metadata = read_vnnlib2(path)
        return VNNLibEntries(entries, metadata)

    def _config(self, cex_path, eval_adv_example=False,
                verify_onnxruntime_output=False,
                onnx_path='model.onnx', vnnlib_path='property.vnnlib'):
        return (
            patch.dict(arguments.Config['general'], {
                'save_adv_example': True,
                'eval_adv_example': eval_adv_example,
                'onnx_adv_example': False,
                'verify_onnxruntime_output': verify_onnxruntime_output,
            }),
            patch.dict(arguments.Config['attack'], {
                'adv_saver': 'default_adv_saver',
                'adv_verifier': 'default_adv_verifier',
                'cex_path': cex_path,
            }),
            patch.dict(arguments.Config['model'], {'onnx_path': onnx_path}),
            patch.dict(
                arguments.Config['specification'],
                {'vnnlib_path': vnnlib_path}),
        )

    def test_adv_savers_reject_direct_calls(self):
        from custom.custom_adv_saver import customized_gtrsb_saver

        cases = [
            (
                default_adv_saver,
                torch.tensor([[0.5]]),
                torch.tensor([[0.0]]),
            ),
            (
                customized_gtrsb_saver,
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 1),
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            for saver, x, y in cases:
                with self.subTest(saver=saver.__name__):
                    with self.assertRaisesRegex(
                            RuntimeError, 'check_and_save_cex'):
                        saver(x, y, path)

    def test_check_and_save_cex_allows_configured_saver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            patches = self._config(path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, '_save_vnnlib1_assignment',
                        wraps=attack_utils_module._save_vnnlib1_assignment
                    ) as mock_assignment:
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                    self._vnnlib(), path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            mock_assignment.assert_called_once()
            with open(path, encoding='utf-8') as cex_file:
                content = cex_file.read()
            self.assertIn('(X_0', content)
            self.assertIn('(Y_0', content)

    def test_configured_saver_can_receive_vnnlib(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            vnnlib = self._vnnlib()
            seen = {}

            def vnnlib_aware_saver(adv_input, adv_output, res_path, vnnlib=None):
                seen['vnnlib'] = vnnlib
                with open(res_path, 'w', encoding='utf-8') as cex_file:
                    cex_file.write('saved')

            patches = self._config(path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.dict(
                        arguments.Config['attack'],
                        {'adv_saver': 'vnnlib_aware_saver'}), \
                    patch.object(
                        attack_utils_module,
                        'vnnlib_aware_saver',
                        vnnlib_aware_saver,
                        create=True):
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            self.assertIs(seen['vnnlib'], vnnlib)

    def test_check_and_save_cex_writes_vnnlib2_assignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X1 Real [1, 2])
  (declare-input X2 Real [1, 1])
  (declare-output Y Real [1, 2])
)
(assert (<= Y[0,0] 10.0))
""")
            patches = self._config(path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'default_adv_verifier',
                        return_value=True), \
                    patch.object(
                        attack_utils_module, '_save_vnnlib2_assignment',
                        wraps=attack_utils_module._save_vnnlib2_assignment
                    ) as mock_assignment:
                status, success = check_and_save_cex(
                    torch.tensor([[1.0, 2.0, 3.0]]),
                    torch.tensor([[4.0, 5.0]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            mock_assignment.assert_called_once()
            with open(path, encoding='utf-8') as cex_file:
                lines = cex_file.read().splitlines()
            self.assertEqual(lines, [
                'sat',
                'X1 real [1, 2]',
                '1',
                '2',
                'X2 real [1, 1]',
                '3',
                'Y real [1, 2]',
                '4',
                '5',
            ])

    def test_vnnlib2_assignment_reconstructs_shared_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (equal-to f)
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (and (>= X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (== X_f[0] X_g[0]))
(assert (<= Y_f[0] Y_g[0]))
""")
            self.assertEqual(vnnlib.metadata.input_index_map, [0, 0])
            patches = self._config(path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'default_adv_verifier',
                        return_value=True):
                status, success = check_and_save_cex(
                    torch.tensor([[7.0]]),
                    torch.tensor([[11.0, 13.0]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            with open(path, encoding='utf-8') as cex_file:
                lines = cex_file.read().splitlines()
            self.assertEqual(lines, [
                'sat',
                'X_f real [1]',
                '7',
                'Y_f real [1]',
                '11',
                'X_g real [1]',
                '7',
                'Y_g real [1]',
                '13',
            ])

    def test_vnnlib2_eval_adv_example_skips_external_checker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (<= Y[0] 0.0))
""")
            patches = self._config(path, eval_adv_example=True)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'default_adv_verifier',
                        return_value=True), \
                    patch.object(
                        attack_utils_module.subprocess, 'run') as mock_run:
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]),
                    torch.tensor([[-1.0]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            mock_run.assert_not_called()

    def test_onnxruntime_spec_check_rejects_nonviolating_cex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            patches = self._config(path, verify_onnxruntime_output=True)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'inference_onnx',
                        return_value=np.array([[1.0]], dtype=np.float32)):
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                    self._vnnlib(), path, 'unsafe')

            self.assertEqual(status, 'unknown')
            self.assertFalse(success)
            self.assertFalse(os.path.exists(path))

    def test_onnxruntime_spec_check_accepts_violating_cex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            patches = self._config(path, verify_onnxruntime_output=True)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'inference_onnx',
                        return_value=np.array([[-0.5]], dtype=np.float32)):
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                    self._vnnlib(), path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(path))

    def test_vnnlib2_rejects_onnx_replay_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            property_path = os.path.join(temp_dir, 'property.vnnlib')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (> Y[0] 0.0))
""")
            patches = self._config(
                path, verify_onnxruntime_output=True,
                vnnlib_path=property_path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'inference_onnx',
                        return_value=np.array([[-1.0]], dtype=np.float32)):
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]),
                    torch.tensor([[1.0]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unknown')
            self.assertFalse(success)
            self.assertFalse(os.path.exists(path))

    def test_vnnlib2_onnx_replay_supports_multiple_networks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            property_path = os.path.join(temp_dir, 'property.vnnlib')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (== X_f[0] X_g[0]))
(assert (> Y_g[0] (+ Y_f[0] 0.05)))
""")
            vnnlib.metadata.network_paths = {
                'f': 'f.onnx',
                'g': 'g.onnx',
            }

            def inference(path, inputs):
                np.testing.assert_allclose(inputs, np.array([0.5]))
                return np.array([0.0 if path == 'f.onnx' else 0.1])

            patches = self._config(
                path, verify_onnxruntime_output=True,
                onnx_path=None, vnnlib_path=property_path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'inference_onnx',
                        side_effect=inference) as replay:
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]),
                    torch.tensor([[0.0, 0.1]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unsafe')
            self.assertTrue(success)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(
                [call.args[0] for call in replay.call_args_list],
                ['f.onnx', 'g.onnx'])

    def test_vnnlib2_onnx_replay_rejects_invalid_multiple_network_cex(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            property_path = os.path.join(temp_dir, 'property.vnnlib')
            vnnlib = self._vnnlib2(temp_dir, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (== X_f[0] X_g[0]))
(assert (> Y_g[0] (+ Y_f[0] 0.05)))
""")
            vnnlib.metadata.network_paths = {
                'f': 'f.onnx',
                'g': 'g.onnx',
            }

            def inference(path, inputs):
                del inputs
                return np.array([0.0 if path == 'f.onnx' else 0.01])

            patches = self._config(
                path, verify_onnxruntime_output=True,
                onnx_path=None, vnnlib_path=property_path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module, 'inference_onnx',
                        side_effect=inference):
                status, success = check_and_save_cex(
                    torch.tensor([[0.5]]),
                    torch.tensor([[0.0, 0.1]]),
                    vnnlib, path, 'unsafe')

            self.assertEqual(status, 'unknown')
            self.assertFalse(success)
            self.assertFalse(os.path.exists(path))

    def test_external_validation_failure_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            patches = self._config(path, eval_adv_example=True)
            error = attack_utils_module.subprocess.CalledProcessError(
                1, ['check_counterexample.py'])
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(
                        attack_utils_module.subprocess, 'run',
                        side_effect=error) as mock_run:
                with self.assertRaises(
                        attack_utils_module.subprocess.CalledProcessError):
                    check_and_save_cex(
                        torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                        self._vnnlib(), path, 'unsafe')

            self.assertEqual(mock_run.call_count, 1)

    def test_saver_failure_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'cex.txt')
            patches = self._config(path)
            with patches[0], patches[1], patches[2], patches[3], \
                    patch(
                        'attack.attack_utils.default_adv_saver',
                        side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    check_and_save_cex(
                        torch.tensor([[0.5]]), torch.tensor([[0.0]]),
                        self._vnnlib(), path, 'unsafe')


class TestBoundaryAttack(unittest.TestCase):
    """Tests for boundary_attack function."""

    def test_small_perturbation(self):
        """Test boundary attack with small number of perturbed pixels."""
        model = SimpleModel(input_size=10)
        x = torch.randn(1, 10)
        data_min = x.clone()
        data_max = x.clone()
        # Perturb only 3 pixels
        data_min[0, :3] -= 0.1
        data_max[0, :3] += 0.1

        result = boundary_attack(model, x, data_min, data_max)

        self.assertIsNotNone(result)
        # Should generate 2^3 = 8 adversarial inputs
        self.assertEqual(result.shape[0], 8)

    def test_large_perturbation_returns_none(self):
        """Test boundary attack returns None for large perturbation."""
        model = SimpleModel(input_size=10)
        x = torch.randn(1, 10)
        data_min = x - 0.1  # All pixels perturbed
        data_max = x + 0.1

        result = boundary_attack(model, x, data_min, data_max)

        self.assertIsNone(result)


class TestAttackWithGeneralSpecs(unittest.TestCase):
    """Tests for the top-level general-spec attack interface."""

    def test_clean_input_invalid_candidate_runs_pgd(self):
        from attack.attack_interface import attack_with_general_specs
        import attack.general_spec_attack as general_spec_attack

        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(0.5)

        config = {
            'specification': {'norm': float('inf')},
            'attack': {
                'general_attack': True,
                'pgd_alpha': 0.01,
                'pgd_alpha_scale': False,
                'pgd_restarts': 1,
                'pgd_batch_size': 1,
                'pgd_steps': 1,
                'attack_tolerance': 0.0,
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'bab': {'attack': {'enabled': False}},
        }
        pgd_result = PGDAttackResult(
            attack_success=torch.tensor([False]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.tensor([[[0.25]]]),
            adv_output_per_or=torch.tensor([[[0.4]]]),
            adv_margin_per_or=torch.tensor([[0.1]]),
            adv_input_best=torch.tensor([[0.25]]),
            adv_output_best=torch.tensor([[0.4]]),
            adv_margin_best=torch.tensor([0.1]),
            adv_margin_per_spec=torch.tensor([[0.1]]),
            adv_input_all=None,
        )
        vnnlib = [([[0.0, 1.0]], [])]

        with patch('arguments.Config', config), \
                patch.object(
                    general_spec_attack, 'pgd_attack_with_general_specs',
                    return_value=pgd_result) as pgd_attack, \
                patch(
                    'attack.attack_utils.is_valid_attack_counterexample',
                    return_value=False) as validator:
            result = attack_with_general_specs(
                model,
                torch.tensor([[0.25]]),
                torch.tensor([[0.0]]),
                torch.tensor([[1.0]]),
                torch.tensor([[-1.0]]),
                torch.tensor([-0.5]),
                torch.tensor([1]),
                vnnlib=vnnlib,
            )

        self.assertFalse(result[0].all())
        self.assertEqual(validator.call_count, 1)
        pgd_attack.assert_called_once()
        self.assertIs(pgd_attack.call_args.kwargs['vnnlib'], vnnlib)

    def test_clean_input_uses_other_valid_or_candidate(self):
        from attack.attack_interface import attack_with_general_specs
        import attack.general_spec_attack as general_spec_attack

        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(0.5)

        config = {
            'specification': {'norm': float('inf')},
            'attack': {
                'general_attack': True,
                'pgd_alpha': 0.01,
                'pgd_alpha_scale': False,
                'pgd_restarts': 1,
                'pgd_batch_size': 1,
                'pgd_steps': 1,
                'attack_tolerance': 0.0,
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'bab': {'attack': {'enabled': False}},
        }
        vnnlib = [([[0.0, 1.0]], [])]

        with patch('arguments.Config', config), \
                patch.object(
                    general_spec_attack, 'pgd_attack_with_general_specs'
                ) as pgd_attack, \
                patch(
                    'attack.attack_utils.is_valid_attack_counterexample',
                    side_effect=[False, True]) as validator:
            result = attack_with_general_specs(
                model,
                torch.tensor([[0.25], [0.75]]),
                torch.tensor([[0.0], [0.0]]),
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([[-1.0], [-1.0]]),
                torch.tensor([-0.3, -0.4]),
                torch.tensor([1, 1]),
                vnnlib=vnnlib,
            )

        self.assertTrue(result[0].all())
        self.assertEqual(result[1].item(), 1)
        self.assertEqual(validator.call_count, 2)
        pgd_attack.assert_not_called()

    def test_invalid_pgd_batch_candidate_continues_restarts(self):
        from attack.attack_interface import attack_with_general_specs
        import attack.general_spec_attack as general_spec_attack

        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(0.5)

        config = {
            'specification': {'norm': float('inf')},
            'attack': {
                'general_attack': True,
                'pgd_alpha': 0.01,
                'pgd_alpha_scale': False,
                'pgd_restarts': 2,
                'pgd_batch_size': 1,
                'pgd_steps': 1,
                'attack_tolerance': 0.0,
                'adv_example_finalizer': 'default_adv_example_finalizer',
            },
            'bab': {'attack': {'enabled': False}},
        }
        invalid_result = PGDAttackResult(
            attack_success=torch.tensor([True]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.tensor([[[0.0]]]),
            adv_output_per_or=torch.tensor([[[-1.0]]]),
            adv_margin_per_or=torch.tensor([[-1.0]]),
            adv_input_best=torch.tensor([[0.0]]),
            adv_output_best=torch.tensor([[-1.0]]),
            adv_margin_best=torch.tensor([-1.0]),
            adv_margin_per_spec=torch.tensor([[-1.0]]),
            adv_input_all=None,
        )
        valid_result = PGDAttackResult(
            attack_success=torch.tensor([True]),
            best_or_idx=torch.tensor([0]),
            adv_input_per_or=torch.tensor([[[0.25]]]),
            adv_output_per_or=torch.tensor([[[-0.5]]]),
            adv_margin_per_or=torch.tensor([[-0.5]]),
            adv_input_best=torch.tensor([[0.25]]),
            adv_output_best=torch.tensor([[-0.5]]),
            adv_margin_best=torch.tensor([-0.5]),
            adv_margin_per_spec=torch.tensor([[-0.5]]),
            adv_input_all=None,
        )
        vnnlib = [([[0.0, 1.0]], [])]

        with patch('arguments.Config', config), \
                patch.object(
                    general_spec_attack, 'pgd_attack_with_general_specs',
                    side_effect=[invalid_result, valid_result]) as pgd_attack, \
                patch(
                    'attack.attack_utils.is_valid_attack_counterexample',
                    side_effect=[False, True]) as validator:
            result = attack_with_general_specs(
                model,
                torch.tensor([[0.25]]),
                torch.tensor([[0.0]]),
                torch.tensor([[1.0]]),
                torch.tensor([[-1.0]]),
                torch.tensor([-0.6]),
                torch.tensor([1]),
                vnnlib=vnnlib,
            )

        self.assertTrue(result[0].all())
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(pgd_attack.call_count, 2)


class TestDefaultAdvVerifier(unittest.TestCase):
    """Tests for default_adv_verifier function."""

    def test_gtrsb_onnxruntime_counterexample_conversion(self):
        """GTSRB ONNX checks use NHWC inputs and Softmax outputs."""
        from custom.custom_adv_verifier import (
            prepare_gtrsb_onnxruntime_counterexample)

        adv_input = torch.arange(
            2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        logits = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)

        onnx_input, onnx_output = prepare_gtrsb_onnxruntime_counterexample(
            adv_input, logits)

        self.assertEqual(tuple(onnx_input.shape), (2, 4, 5, 3))
        self.assertTrue(onnx_input.is_contiguous())
        self.assertTrue(torch.equal(onnx_input[0, :, :, 0], adv_input[0, 0]))
        self.assertTrue(torch.allclose(
            onnx_output, torch.nn.functional.softmax(logits, dim=1)))

    def test_requires_vnnlib(self):
        """default_adv_verifier validates specs and requires a vnnlib."""
        adv_input = torch.randn(1, 10)
        adv_output = torch.randn(1, 5)

        with self.assertRaisesRegex(ValueError, 'requires a vnnlib'):
            default_adv_verifier(adv_input, adv_output, vnnlib=None)

    def test_output_consistency_helper_requires_onnx_path(self):
        """Legacy output comparison raises when no ONNX path exists."""
        adv_input = torch.randn(1, 10)
        adv_output = torch.randn(1, 5)
        old_path = arguments.Config['model']['onnx_path']
        arguments.Config['model']['onnx_path'] = None
        try:
            with patch('attack.attack_utils.inference_onnx') as check:
                with self.assertRaisesRegex(
                        RuntimeError, 'requires a single ONNX'):
                    verify_onnxruntime_output_matches_pytorch(
                        adv_input, adv_output)
        finally:
            arguments.Config['model']['onnx_path'] = old_path

        check.assert_not_called()

    def test_uses_onnx_output_for_vnnlib1_spec_check(self):
        """ONNX output, not PyTorch output closeness, decides VNN-LIB 1 specs."""
        adv_input = torch.tensor([[0.5]])
        adv_output = torch.tensor([[1.0]])
        vnnlib = [(
            np.array([[0.0, 1.0]]),
            [(np.array([[1.0]]), np.array([0.0]))],
        )]
        old_path = arguments.Config['model']['onnx_path']
        arguments.Config['model']['onnx_path'] = 'model.onnx'
        try:
            with patch(
                    'attack.attack_utils.inference_onnx',
                    return_value=np.array([[-1.0]], dtype=np.float32)):
                result = default_adv_verifier(
                    adv_input, adv_output, vnnlib=vnnlib,
                    verify_with_onnxruntime=True)
        finally:
            arguments.Config['model']['onnx_path'] = old_path

        self.assertTrue(result)

    def test_rejects_vnnlib1_when_onnx_output_does_not_violate_spec(self):
        """A PyTorch-only violation is not enough when ONNX checking is on."""
        adv_input = torch.tensor([[0.5]])
        adv_output = torch.tensor([[-1.0]])
        vnnlib = [(
            np.array([[0.0, 1.0]]),
            [(np.array([[1.0]]), np.array([0.0]))],
        )]
        old_path = arguments.Config['model']['onnx_path']
        arguments.Config['model']['onnx_path'] = 'model.onnx'
        try:
            with patch(
                    'attack.attack_utils.inference_onnx',
                    return_value=np.array([[1.0]], dtype=np.float32)):
                result = default_adv_verifier(
                    adv_input, adv_output, vnnlib=vnnlib,
                    verify_with_onnxruntime=True)
        finally:
            arguments.Config['model']['onnx_path'] = old_path

        self.assertFalse(result)

    def test_onnxruntime_batch_validation_preserves_input_shape(self):
        """Batched attack candidate checks keep model input shape for ONNX."""
        adv_input = torch.full((1, 1, 1, 3, 4, 5), 0.5)
        adv_output = torch.tensor([[[[-1.0]]]])
        input_box = np.array([[0.0, 1.0]] * 60)
        vnnlib = [(input_box, [(np.array([[1.0]]), np.array([0.0]))])]

        old_path = arguments.Config['model']['onnx_path']
        arguments.Config['model']['onnx_path'] = 'model.onnx'
        try:
            def fake_inference_onnx(path, onnx_input):
                self.assertEqual(path, 'model.onnx')
                self.assertEqual(onnx_input.shape, (1, 3, 4, 5))
                return np.array([[-1.0]], dtype=np.float32)

            with patch(
                    'attack.attack_utils.inference_onnx',
                    side_effect=fake_inference_onnx):
                result = attack_utils_module.are_attack_counterexamples_valid(
                    adv_input, adv_output, vnnlib,
                    verify_with_onnxruntime=True)
        finally:
            arguments.Config['model']['onnx_path'] = old_path

        self.assertTrue(result)

    def test_onnxruntime_spec_check_requires_onnx_path(self):
        """ONNX validation must not fall back when no ONNX path is available."""
        adv_input = torch.tensor([[0.5]])
        adv_output = torch.tensor([[-1.0]])
        vnnlib = [(
            np.array([[0.0, 1.0]]),
            [(np.array([[1.0]]), np.array([0.0]))],
        )]
        old_path = arguments.Config['model']['onnx_path']
        arguments.Config['model']['onnx_path'] = None
        try:
            with self.assertRaisesRegex(RuntimeError, 'requires a single ONNX'):
                default_adv_verifier(
                    adv_input, adv_output, vnnlib=vnnlib,
                    verify_with_onnxruntime=True)
        finally:
            arguments.Config['model']['onnx_path'] = old_path


class TestIsSpecificationVio(unittest.TestCase):
    """Tests for is_specification_vio function."""

    def test_specification_violated(self):
        """Test when specification is violated."""
        # Create a simple spec: y[0] <= 0
        input_box = [(0.0, 1.0)] * 5
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [0.0]
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)
        expected_y = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0])  # y[0] = -0.5 < 0, violates y[0] <= 0? No, satisfies.

        # Actually, is_specification_vio checks if ANY spec in spec_list is SAT (violated from verifier perspective)
        # If vec = mat @ y <= rhs + output_tol, then sat=True,
        # meaning the spec is violated (adversarial found).
        result = is_specification_vio(
            box_spec_list, x_list, expected_y, tol=CEX_INPUT_TOLERANCE)

        self.assertTrue(result)  # Spec is satisfied (attacked successfully)

    def test_specification_not_violated(self):
        """Test when specification is not violated."""
        input_box = [(0.0, 1.0)] * 5
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [-1.0]  # y[0] <= -1.0
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)
        expected_y = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0])  # y[0] = 0.5 > -1.0, does not satisfy

        result = is_specification_vio(
            box_spec_list, x_list, expected_y, tol=CEX_INPUT_TOLERANCE)

        self.assertFalse(result)

    def test_input_outside_box(self):
        """Test when input is outside the box."""
        input_box = [(0.0, 0.1)] * 5  # Very small box
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [0.0]
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)  # Outside the box
        expected_y = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0])

        result = is_specification_vio(
            box_spec_list, x_list, expected_y, tol=CEX_INPUT_TOLERANCE)

        self.assertFalse(result)  # Input outside box, so no violation found


class TestAdvExample(unittest.TestCase):
    """Tests for AdvExample class."""

    def test_init(self):
        """Test AdvExample initialization."""
        x = torch.randn(1, 10)
        obj = -0.5
        pattern = [torch.randint(0, 2, (20,))]

        adv = AdvExample(x, obj, pattern)

        self.assertTrue(torch.equal(adv.x, x))
        self.assertEqual(adv.obj, obj)
        self.assertEqual(len(adv.activation_pattern), 1)

    def test_lt(self):
        """Test __lt__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -0.5, [])

        # adv1 has smaller obj (-1.0 < -0.5)
        self.assertTrue(adv1 < adv2)

    def test_le(self):
        """Test __le__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -1.0, [])

        self.assertTrue(adv1 <= adv2)

    def test_eq(self):
        """Test __eq__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -1.0, [])

        self.assertTrue(adv1 == adv2)


class TestAdvExamplePool(unittest.TestCase):
    """Tests for AdvExamplePool class."""

    def _create_bounded_model(self):
        """Create a bounded model for testing."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        return bounded_model

    def test_init(self):
        """Test AdvExamplePool initialization."""
        model = self._create_bounded_model()
        # Create unstable masks for each ReLU layer
        unstable_mask = [torch.ones(1, 20)]  # One ReLU layer with 20 neurons

        pool = AdvExamplePool(model, unstable_mask, capacity=100)

        self.assertEqual(pool.capacity, 100)
        self.assertEqual(len(pool.adv_pool), 0)

    def test_get_var_empty_pool(self):
        """Test get_var with empty pool."""
        model = self._create_bounded_model()
        unstable_mask = [torch.ones(1, 20)]

        pool = AdvExamplePool(model, unstable_mask)
        result = pool.get_var()

        self.assertIsNone(result)

    def test_replace_adv_example_add(self):
        """Test replace_adv_example adds new example."""
        model = self._create_bounded_model()
        unstable_mask = [torch.ones(1, 20)]

        pool = AdvExamplePool(model, unstable_mask)

        adv = AdvExample(torch.randn(10), -0.5, [torch.randint(0, 2, (20,))])
        c_replaced, c_added, c_rejected = pool.replace_adv_example(adv, 0, 0, 0)

        self.assertEqual(c_added, 1)
        self.assertEqual(len(pool.adv_pool), 1)


class TestAdvExamplePoolActivationPattern(unittest.TestCase):
    """Tests for AdvExamplePool activation pattern methods."""

    def _create_pool_with_examples(self):
        """Create a pool with some adversarial examples."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        unstable_mask = [torch.ones(20)]  # One ReLU layer

        pool = AdvExamplePool(bounded_model, unstable_mask, capacity=100)

        # Add some examples manually
        for i in range(5):
            pattern = [torch.randint(0, 2, (20,))]
            adv = AdvExample(torch.randn(10), -float(i) / 10, pattern)
            pool.adv_pool.add(adv)

        return pool

    def test_get_activation_pattern_from_pool(self):
        """Test getting common activation patterns."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_activation_pattern_from_pool(prob_threshold=0.6)

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_ranked_activation_pattern(self):
        """Test getting ranked activation patterns."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_ranked_activation_pattern()

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_ranked_activation_pattern_with_limit(self):
        """Test getting ranked activation patterns with n_activations limit."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_ranked_activation_pattern(n_activations=5)

        self.assertEqual(len(decisions), 5)
        self.assertEqual(len(coeffs), 5)

    def test_get_activation_pattern(self):
        """Test getting activation pattern from single example."""
        pool = self._create_pool_with_examples()
        adv_example = pool.adv_pool[0]

        decisions, coeffs = pool.get_activation_pattern(adv_example)

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_activation_pattern_with_blacklist(self):
        """Test getting activation pattern with blacklist."""
        pool = self._create_pool_with_examples()
        adv_example = pool.adv_pool[0]

        # Blacklist some neurons
        blacklist = [[0, 0], [0, 1], [0, 2]]

        decisions, coeffs = pool.get_activation_pattern(adv_example, blacklist=blacklist)

        # Blacklisted neurons should not appear in decisions
        for d in decisions.tolist():
            self.assertNotIn(d, blacklist)


class TestAdvExamplePoolMostLikelyActivation(unittest.TestCase):
    """Tests for find_most_likely_activation method."""

    def test_find_most_likely_activation(self):
        """Test finding most likely activation for given decisions."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        unstable_mask = [torch.ones(20)]

        pool = AdvExamplePool(bounded_model, unstable_mask)

        # Add examples with consistent patterns
        for i in range(5):
            pattern = [torch.ones(20)]  # All active
            adv = AdvExample(torch.randn(10), -float(i) / 10, pattern)
            pool.adv_pool.add(adv)

        # Get decisions that exist
        decisions, _ = pool.get_activation_pattern_from_pool(prob_threshold=0.5)

        if len(decisions) > 0:
            test_decisions = decisions[:2].tolist()
            coeffs = pool.find_most_likely_activation(test_decisions)
            self.assertEqual(len(coeffs), len(test_decisions))


if __name__ == '__main__':
    unittest.main()
