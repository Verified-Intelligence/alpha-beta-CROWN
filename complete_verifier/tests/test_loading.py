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
"""Unit tests for loading.py module."""

import os
import sys
import pytest
import tempfile
import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# joint_optimization_with_onnx_and_vnnlib Tests
# ============================================================================

class TestJointOptimizationWithOnnxAndVnnlib:
    """Tests for joint_optimization_with_onnx_and_vnnlib function."""

    @pytest.mark.parametrize("flags", [[], ['none'], ['none', 'none', 'none']])
    def test_no_op_flags(self, flags):
        """Test flags that should not modify anything."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        # Create a mock model
        class MockModel:
            output_names = ['output']
            onnx_model = None

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        assert result_model is model
        assert result_vnnlib == vnnlib

    def test_peel_off_sigmoid_flag_no_sigmoid(self):
        """Test peel_off_last_softmax_layer flag when no sigmoid layer exists."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        class MockOnnxGraph:
            node = []  # No nodes

        class MockOnnxModel:
            graph = MockOnnxGraph()

        class MockModel:
            output_names = ['output']
            onnx_model = MockOnnxModel()

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]
        flags = ['peel_off_last_softmax_layer']

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        # No sigmoid found, so model and vnnlib should be unchanged
        assert result_model is model
        assert result_vnnlib == vnnlib


# ============================================================================
# load_verification_dataset Tests
# ============================================================================

class TestLoadVerificationDataset:
    """Tests for load_verification_dataset function."""

    def test_unsupported_dataset_raises(self):
        """Test that unsupported dataset raises NotImplementedError."""
        import arguments
        from loading import load_verification_dataset

        arguments.Config['data']['dataset'] = 'UNSUPPORTED_DATASET'
        arguments.Config['data']['pkl_path'] = None

        with pytest.raises(NotImplementedError, match='Dataset not supported'):
            load_verification_dataset()

# ============================================================================
# vnnlib2_onnx_and_vnnlib_loader Tests
# ============================================================================

class TestVNNLib2OnnxAndVnnlibLoader:
    """Tests for vnnlib2_onnx_and_vnnlib_loader."""

    def test_uses_csv_network_mapping(self, monkeypatch, tmp_path):
        import loading
        from loading import vnnlib2_onnx_and_vnnlib_loader
        from vnnlib2_wrappers import VNNLib2Wrapper

        vnnlib_dir = tmp_path / "vnnlib"
        vnnlib_dir.mkdir()
        spec_path = vnnlib_dir / "case.vnnlib"
        spec_path.write_text("""
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (and (>= X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (== X_f[0] X_g[0]))
(assert (<= Y_f[0] Y_g[0]))
""")

        loaded_paths = []

        def fake_load_model_onnx(path):
            loaded_paths.append(path)
            return torch.nn.Identity(), [(1,)]

        monkeypatch.setattr(loading, "load_model_onnx", fake_load_model_onnx)

        model, shape, vnnlib = vnnlib2_onnx_and_vnnlib_loader(
            str(tmp_path),
            "[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx')]",
            "vnnlib/case.vnnlib")

        assert isinstance(model, VNNLib2Wrapper)
        assert shape == (-1, 1)
        assert len(vnnlib) == 1
        assert set(model.models.keys()) == {"f", "g"}
        assert loaded_paths == [
            os.path.join(str(tmp_path), "onnx/f.onnx"),
            os.path.join(str(tmp_path), "onnx/g.onnx"),
        ]

    def test_single_onnx_path(self):
        from loading import _single_onnx_path

        assert _single_onnx_path("onnx/model.onnx") == "onnx/model.onnx"
        assert _single_onnx_path(
            "[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx')]") is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
