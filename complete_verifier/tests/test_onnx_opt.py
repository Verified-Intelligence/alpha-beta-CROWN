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
"""Unit tests for onnx_opt.py"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import onnx

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnx_opt import (
    create_initializer_tensor, conv_output_shape, fuse_conv_and_bn
)


class TestCreateInitializerTensor(unittest.TestCase):
    """Tests for create_initializer_tensor function."""

    def test_float32_array(self):
        """Test creating initializer from float32 array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = create_initializer_tensor('test', arr)
        self.assertEqual(tensor.name, 'test')
        self.assertEqual(tensor.data_type, onnx.TensorProto.FLOAT)
        self.assertEqual(list(tensor.dims), [3])

    def test_float64_array(self):
        """Test creating initializer from float64 array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor = create_initializer_tensor('test', arr)
        self.assertEqual(tensor.data_type, onnx.TensorProto.FLOAT)

    def test_int64_array(self):
        """Test creating initializer from int64 array."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        tensor = create_initializer_tensor('test', arr)
        self.assertEqual(tensor.data_type, onnx.TensorProto.INT64)

    def test_multidimensional_array(self):
        """Test creating initializer from multidimensional array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = create_initializer_tensor('test_2d', arr)
        self.assertEqual(list(tensor.dims), [2, 2])

    def test_explicit_data_type(self):
        """Test creating initializer with explicit data type."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        tensor = create_initializer_tensor(
            'test', arr, data_type=onnx.TensorProto.DOUBLE)
        self.assertEqual(tensor.data_type, onnx.TensorProto.DOUBLE)

    def test_empty_array(self):
        """Test creating initializer from empty array."""
        arr = np.array([], dtype=np.float32)
        tensor = create_initializer_tensor('empty', arr)
        self.assertEqual(list(tensor.dims), [0])


class TestConvOutputShape(unittest.TestCase):
    """Tests for conv_output_shape function."""

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

    def test_kernel_5x5(self):
        """Test convolution with 5x5 kernel."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=5, stride=1, pad=2)
        self.assertEqual(h, 32)
        self.assertEqual(w, 32)

    def test_asymmetric_input(self):
        """Test convolution with asymmetric input."""
        h, w = conv_output_shape(h_w=(28, 32), kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 26)
        self.assertEqual(w, 30)

    def test_asymmetric_kernel(self):
        """Test convolution with asymmetric kernel."""
        h, w = conv_output_shape(
            h_w=(32, 32), kernel_size=(3, 5), stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 28)

    def test_asymmetric_stride(self):
        """Test convolution with asymmetric stride."""
        h, w = conv_output_shape(
            h_w=(32, 32), kernel_size=3, stride=(2, 1), pad=1)
        self.assertEqual(h, 16)
        self.assertEqual(w, 32)

    def test_asymmetric_padding(self):
        """Test convolution with asymmetric padding."""
        h, w = conv_output_shape(
            h_w=(32, 32), kernel_size=3, stride=1, pad=(0, 1))
        self.assertEqual(h, 30)
        self.assertEqual(w, 32)

    def test_int_input(self):
        """Test with integer input instead of tuple."""
        h, w = conv_output_shape(h_w=32, kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 30)

    def test_dilation(self):
        """Test convolution with dilation."""
        h, w = conv_output_shape(
            h_w=(32, 32), kernel_size=3, stride=1, pad=0, dilation=2)
        self.assertEqual(h, 28)
        self.assertEqual(w, 28)

    def test_1x1_conv(self):
        """Test 1x1 convolution."""
        h, w = conv_output_shape(h_w=(16, 16), kernel_size=1, stride=1, pad=0)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_large_stride(self):
        """Test convolution with large stride."""
        h, w = conv_output_shape(h_w=(64, 64), kernel_size=7, stride=4, pad=3)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)


class TestFuseConvAndBn(unittest.TestCase):
    """Tests for fuse_conv_and_bn function."""

    def test_basic_fusion(self):
        """Test basic conv-bn fusion."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
            bn = nn.BatchNorm2d(16)
            # Set BatchNorm to eval mode to use running stats
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            self.assertIsInstance(fused, nn.Conv2d)
            self.assertEqual(fused.in_channels, 3)
            self.assertEqual(fused.out_channels, 16)
            self.assertTrue(fused.bias is not None)

    def test_fusion_with_conv_bias(self):
        """Test fusion when conv has bias."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
            bn = nn.BatchNorm2d(16)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            self.assertIsInstance(fused, nn.Conv2d)
            self.assertTrue(fused.bias is not None)

    def test_fusion_preserves_shape(self):
        """Test that fused layer produces same shape output."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
            bn = nn.BatchNorm2d(16)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            x = torch.randn(1, 3, 32, 32)
            orig_out = bn(conv(x))
            fused_out = fused(x)

            self.assertEqual(orig_out.shape, fused_out.shape)

    def test_fusion_numerical_accuracy(self):
        """Test that fused layer produces numerically similar output."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
            bn = nn.BatchNorm2d(8)

            # Initialize with known values
            nn.init.ones_(conv.weight)
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            bn.running_mean.fill_(0.5)
            bn.running_var.fill_(1.0)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            x = torch.randn(2, 3, 16, 16)
            orig_out = bn(conv(x))
            fused_out = fused(x)

            self.assertTrue(torch.allclose(orig_out, fused_out, atol=1e-5))

    def test_fusion_different_kernel_sizes(self):
        """Test fusion with different kernel sizes."""
        with torch.no_grad():
            for kernel_size in [1, 3, 5, 7]:
                conv = nn.Conv2d(3, 16, kernel_size=kernel_size, padding=kernel_size//2)
                bn = nn.BatchNorm2d(16)
                bn.eval()

                fused = fuse_conv_and_bn(conv, bn)

                self.assertEqual(fused.kernel_size, (kernel_size, kernel_size))

    def test_fusion_with_stride(self):
        """Test fusion with strided convolution."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            bn = nn.BatchNorm2d(16)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            self.assertEqual(fused.stride, (2, 2))


class TestOnnxOptEdgeCases(unittest.TestCase):
    """Edge case tests for onnx_opt functions."""

    def test_conv_output_shape_zero_padding(self):
        """Test conv output shape with zero padding."""
        h, w = conv_output_shape(h_w=(10, 10), kernel_size=5, stride=1, pad=0)
        self.assertEqual(h, 6)
        self.assertEqual(w, 6)

    def test_create_initializer_single_value(self):
        """Test creating initializer from single value array."""
        arr = np.array([42.0], dtype=np.float32)
        tensor = create_initializer_tensor('single', arr)
        self.assertEqual(list(tensor.dims), [1])

    def test_create_initializer_high_dimensional(self):
        """Test creating initializer from high dimensional array."""
        arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        tensor = create_initializer_tensor('high_dim', arr)
        self.assertEqual(list(tensor.dims), [2, 3, 4, 5])


class TestFuseConvAndBnVariations(unittest.TestCase):
    """Additional tests for fuse_conv_and_bn function."""

    def test_fusion_preserves_numerical_accuracy(self):
        """Test that fusion preserves numerical accuracy on various inputs."""
        with torch.no_grad():
            for _ in range(5):
                conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                bn = nn.BatchNorm2d(16)
                bn.eval()

                fused = fuse_conv_and_bn(conv, bn)

                x = torch.randn(2, 3, 16, 16)
                orig_out = bn(conv(x))
                fused_out = fused(x)

                self.assertTrue(torch.allclose(orig_out, fused_out, atol=1e-5))

    def test_fusion_with_different_channels(self):
        """Test fusion with different channel configurations."""
        with torch.no_grad():
            for in_ch, out_ch in [(1, 8), (3, 16), (16, 32), (32, 64)]:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                bn = nn.BatchNorm2d(out_ch)
                bn.eval()

                fused = fuse_conv_and_bn(conv, bn)

                self.assertEqual(fused.in_channels, in_ch)
                self.assertEqual(fused.out_channels, out_ch)


class TestConvOutputShapeEdgeCases(unittest.TestCase):
    """Edge case tests for conv_output_shape."""

    def test_7x7_kernel_stride_2(self):
        """Test 7x7 kernel with stride 2 (typical ResNet first layer)."""
        h, w = conv_output_shape(h_w=(224, 224), kernel_size=7, stride=2, pad=3)
        self.assertEqual(h, 112)
        self.assertEqual(w, 112)

    def test_very_small_input(self):
        """Test with very small input."""
        h, w = conv_output_shape(h_w=(3, 3), kernel_size=1, stride=1, pad=0)
        self.assertEqual(h, 3)
        self.assertEqual(w, 3)

    def test_large_dilation(self):
        """Test with large dilation."""
        # With dilation=4, effective kernel size = 1 + 4*(3-1) = 9
        # output = (32 + 2*4 - 9) / 1 + 1 = 32
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=4, dilation=4)
        self.assertEqual(h, 32)
        self.assertEqual(w, 32)

    def test_asymmetric_all_params(self):
        """Test with all asymmetric parameters."""
        h, w = conv_output_shape(
            h_w=(28, 32),
            kernel_size=(3, 5),
            stride=(1, 2),
            pad=(1, 2)
        )
        self.assertEqual(h, 28)
        self.assertEqual(w, 16)


class TestCreateInitializerTensorEdgeCases(unittest.TestCase):
    """Edge case tests for create_initializer_tensor."""

    def test_bool_array(self):
        """Test with bool array (should raise NotImplementedError)."""
        arr = np.array([True, False, True], dtype=np.bool_)
        with self.assertRaises(NotImplementedError):
            create_initializer_tensor('test_bool', arr)

    def test_large_array(self):
        """Test with large array."""
        arr = np.random.randn(100, 100, 100).astype(np.float32)
        tensor = create_initializer_tensor('large', arr)
        self.assertEqual(list(tensor.dims), [100, 100, 100])

    def test_scalar_array(self):
        """Test with scalar (0-d) array."""
        arr = np.float32(3.14)
        # Scalars have empty shape
        tensor = create_initializer_tensor('scalar', np.array([arr]))
        self.assertEqual(list(tensor.dims), [1])


class TestConvOutputShapePooling(unittest.TestCase):
    """Tests for conv_output_shape with pooling-like configurations."""

    def test_maxpool_like(self):
        """Test max pool like configuration (2x2 kernel, stride 2)."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=2, stride=2, pad=0)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_global_avg_pool_like(self):
        """Test global average pool like configuration."""
        h, w = conv_output_shape(h_w=(7, 7), kernel_size=7, stride=1, pad=0)
        self.assertEqual(h, 1)
        self.assertEqual(w, 1)


class TestCreateNewInitializers(unittest.TestCase):
    """Tests for create_new_initializers function."""

    def test_create_new_initializers_with_matching_inputs(self):
        """Test creating new initializers when inputs match."""
        from onnx_opt import create_new_initializers

        # Create a mock node with input names
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['input', 'weight', 'bias'],
            outputs=['output']
        )

        initializers = {
            'weight': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            'bias': np.array([0.1, 0.2], dtype=np.float32),
        }

        new_inits = create_new_initializers(node, initializers)

        self.assertEqual(len(new_inits), 2)
        names = [init.name for init in new_inits]
        self.assertIn('weight', names)
        self.assertIn('bias', names)

    def test_create_new_initializers_no_matching(self):
        """Test creating new initializers when no inputs match."""
        from onnx_opt import create_new_initializers

        node = onnx.helper.make_node(
            'Relu',
            inputs=['input'],
            outputs=['output']
        )

        initializers = {
            'weight': np.array([1.0, 2.0], dtype=np.float32),
        }

        new_inits = create_new_initializers(node, initializers)
        self.assertEqual(len(new_inits), 0)

    def test_create_new_initializers_partial_match(self):
        """Test creating new initializers with partial matches."""
        from onnx_opt import create_new_initializers

        node = onnx.helper.make_node(
            'Gemm',
            inputs=['input', 'weight', 'missing_bias'],
            outputs=['output']
        )

        initializers = {
            'weight': np.array([[1.0, 2.0]], dtype=np.float32),
        }

        new_inits = create_new_initializers(node, initializers)
        self.assertEqual(len(new_inits), 1)
        self.assertEqual(new_inits[0].name, 'weight')


class TestStridedConversion(unittest.TestCase):
    """Tests for strided_conversion function."""

    def test_strided_conversion_basic(self):
        """Test basic strided conversion."""
        from onnx_opt import strided_conversion

        # Small input for testing
        input_feature = torch.randn(1, 1, 4, 4).double()
        kernel = torch.randn(2, 1, 3, 3).double()

        result = strided_conversion(input_feature, kernel, padding=1, stride=1)

        # Output should be (out_channels * out_h * out_w, in_channels * in_h * in_w)
        # With padding=1, stride=1, kernel=3x3, output is 4x4
        # So shape should be (2 * 4 * 4, 1 * 4 * 4) = (32, 16)
        self.assertEqual(result.shape[0], 2 * 4 * 4)
        self.assertEqual(result.shape[1], 1 * 4 * 4)

    def test_strided_conversion_stride_2(self):
        """Test strided conversion with stride 2."""
        from onnx_opt import strided_conversion

        input_feature = torch.randn(1, 1, 8, 8).double()
        kernel = torch.randn(4, 1, 3, 3).double()

        result = strided_conversion(input_feature, kernel, padding=1, stride=2)

        # With padding=1, stride=2, kernel=3x3, input 8x8 -> output 4x4
        # Shape: (4 * 4 * 4, 1 * 8 * 8) = (64, 64)
        self.assertEqual(result.shape[0], 4 * 4 * 4)
        self.assertEqual(result.shape[1], 1 * 8 * 8)

    def test_strided_conversion_no_padding(self):
        """Test strided conversion without padding."""
        from onnx_opt import strided_conversion

        input_feature = torch.randn(1, 1, 6, 6).double()
        kernel = torch.randn(2, 1, 3, 3).double()

        result = strided_conversion(input_feature, kernel, padding=0, stride=1)

        # Without padding, 6x6 with 3x3 kernel -> 4x4 output
        # Shape: (2 * 4 * 4, 1 * 6 * 6) = (32, 36)
        self.assertEqual(result.shape[0], 2 * 4 * 4)
        self.assertEqual(result.shape[1], 1 * 6 * 6)

    def test_strided_conversion_multi_channel(self):
        """Test strided conversion with multiple input channels."""
        from onnx_opt import strided_conversion

        input_feature = torch.randn(1, 3, 4, 4).double()
        kernel = torch.randn(8, 3, 3, 3).double()

        result = strided_conversion(input_feature, kernel, padding=1, stride=1)

        # Shape: (8 * 4 * 4, 3 * 4 * 4) = (128, 48)
        self.assertEqual(result.shape[0], 8 * 4 * 4)
        self.assertEqual(result.shape[1], 3 * 4 * 4)


class TestConvertOnnxToDouble(unittest.TestCase):
    """Tests for convert_onnx_to_double function."""

    def _create_simple_onnx_model(self):
        """Create a simple ONNX model for testing."""
        # Create a simple model: input -> MatMul -> output
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 2])

        weight = np.random.randn(4, 2).astype(np.float32)
        weight_init = onnx.numpy_helper.from_array(weight, name='weight')

        matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs=['input', 'weight'],
            outputs=['output']
        )

        graph = onnx.helper.make_graph(
            [matmul_node],
            'test_graph',
            [input_tensor],
            [output_tensor],
            [weight_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_convert_to_double_file(self):
        """Test converting an ONNX file to double precision."""
        from onnx_opt import convert_onnx_to_double

        # Create and save a simple model
        model = self._create_simple_onnx_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            result = convert_onnx_to_double(temp_path)

            # Check that inputs are converted to double
            for inp in result.graph.input:
                self.assertEqual(
                    inp.type.tensor_type.elem_type,
                    onnx.TensorProto.DOUBLE
                )

            # Check that outputs are converted to double
            for out in result.graph.output:
                self.assertEqual(
                    out.type.tensor_type.elem_type,
                    onnx.TensorProto.DOUBLE
                )

            # Check that initializers are converted to double
            for init in result.graph.initializer:
                self.assertEqual(init.data_type, onnx.TensorProto.DOUBLE)
        finally:
            os.unlink(temp_path)


class TestCompressOnnxBasic(unittest.TestCase):
    """Tests for compress_onnx function with basic flags."""

    def _create_mlp_onnx_model(self):
        """Create a simple MLP ONNX model for testing."""
        # input -> MatMul -> Add -> Relu -> MatMul -> Add -> output
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 2])

        w1 = np.random.randn(4, 8).astype(np.float32)
        b1 = np.random.randn(8).astype(np.float32)
        w2 = np.random.randn(8, 2).astype(np.float32)
        b2 = np.random.randn(2).astype(np.float32)

        w1_init = onnx.numpy_helper.from_array(w1, name='w1')
        b1_init = onnx.numpy_helper.from_array(b1, name='b1')
        w2_init = onnx.numpy_helper.from_array(w2, name='w2')
        b2_init = onnx.numpy_helper.from_array(b2, name='b2')

        nodes = [
            onnx.helper.make_node('MatMul', ['input', 'w1'], ['mm1']),
            onnx.helper.make_node('Add', ['mm1', 'b1'], ['add1']),
            onnx.helper.make_node('Relu', ['add1'], ['relu1']),
            onnx.helper.make_node('MatMul', ['relu1', 'w2'], ['mm2']),
            onnx.helper.make_node('Add', ['mm2', 'b2'], ['output']),
        ]

        graph = onnx.helper.make_graph(
            nodes,
            'mlp_graph',
            [input_tensor],
            [output_tensor],
            [w1_init, b1_init, w2_init, b2_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_compress_onnx_none_flag(self):
        """Test compress_onnx with 'none' flag (passthrough)."""
        from onnx_opt import compress_onnx

        model = self._create_mlp_onnx_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(model, old_path, save_path, ['none'])

            self.assertIsNotNone(result)
            # Model should be valid
            onnx.checker.check_model(result)
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_compress_onnx_merge_linear(self):
        """Test compress_onnx with merge_linear flag."""
        from onnx_opt import compress_onnx

        # Create a model with consecutive linear operations
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 4])

        w1 = np.eye(4).astype(np.float32)
        b1 = np.zeros(4).astype(np.float32)
        w2 = np.eye(4).astype(np.float32)
        b2 = np.ones(4).astype(np.float32)

        w1_init = onnx.numpy_helper.from_array(w1, name='w1')
        b1_init = onnx.numpy_helper.from_array(b1, name='b1')
        w2_init = onnx.numpy_helper.from_array(w2, name='w2')
        b2_init = onnx.numpy_helper.from_array(b2, name='b2')

        nodes = [
            onnx.helper.make_node('MatMul', ['input', 'w1'], ['mm1']),
            onnx.helper.make_node('Add', ['mm1', 'b1'], ['add1']),
            onnx.helper.make_node('MatMul', ['add1', 'w2'], ['mm2']),
            onnx.helper.make_node('Add', ['mm2', 'b2'], ['output']),
        ]

        graph = onnx.helper.make_graph(
            nodes,
            'linear_graph',
            [input_tensor],
            [output_tensor],
            [w1_init, b1_init, w2_init, b2_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(model, old_path, save_path, ['merge_linear'])

            self.assertIsNotNone(result)
            onnx.checker.check_model(result)
            # After merging, should have fewer nodes
            self.assertLessEqual(len(result.graph.node), len(model.graph.node))
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestCompressOnnxWithBN(unittest.TestCase):
    """Tests for compress_onnx with BatchNormalization merging."""

    def _create_conv_bn_model(self):
        """Create a Conv + BatchNorm model for testing.

        Note: The merge_bn optimization requires Conv nodes with specific
        attribute ordering: [dilations, group, kernel_shape, pads, strides].
        """
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 3, 8, 8])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 16, 8, 8])

        # Conv weights: (out_channels, in_channels, kH, kW)
        conv_w = np.random.randn(16, 3, 3, 3).astype(np.float32)
        conv_w_init = onnx.numpy_helper.from_array(conv_w, name='conv_w')

        # BatchNorm parameters
        bn_scale = np.ones(16).astype(np.float32)
        bn_bias = np.zeros(16).astype(np.float32)
        bn_mean = np.zeros(16).astype(np.float32)
        bn_var = np.ones(16).astype(np.float32)

        bn_scale_init = onnx.numpy_helper.from_array(bn_scale, name='bn_scale')
        bn_bias_init = onnx.numpy_helper.from_array(bn_bias, name='bn_bias')
        bn_mean_init = onnx.numpy_helper.from_array(bn_mean, name='bn_mean')
        bn_var_init = onnx.numpy_helper.from_array(bn_var, name='bn_var')

        # Create Conv node with attributes in expected order for merge_bn
        # The code accesses: attribute[0]=dilations, [2]=kernel_shape, [3]=pads, [4]=strides
        conv_node = onnx.helper.make_node(
            'Conv',
            inputs=['input', 'conv_w'],
            outputs=['conv_out'],
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )

        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
            outputs=['output'],
            epsilon=1e-5
        )

        graph = onnx.helper.make_graph(
            [conv_node, bn_node],
            'conv_bn_graph',
            [input_tensor],
            [output_tensor],
            [conv_w_init, bn_scale_init, bn_bias_init, bn_mean_init, bn_var_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_compress_onnx_merge_bn(self):
        """Test compress_onnx with merge_bn flag."""
        from onnx_opt import compress_onnx

        model = self._create_conv_bn_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(model, old_path, save_path, ['merge_bn'])

            self.assertIsNotNone(result)
            onnx.checker.check_model(result)
            # After merging Conv+BN, we should have 1 Conv node instead of 2 nodes
            self.assertEqual(len(result.graph.node), 1)
            self.assertEqual(result.graph.node[0].op_type, 'Conv')
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestCompressOnnxRemoveIneffective(unittest.TestCase):
    """Tests for compress_onnx with remove_ineffective_layers flag.

    Note: The remove_ineffective_layers optimization is designed to work
    within larger graphs where removing Sub(x, 0) or Div(x, 1) operations
    doesn't break output connections. Testing it in isolation requires
    careful graph construction to maintain valid output connections.
    """

    def _create_model_with_ineffective_sub_in_chain(self):
        """Create a model with Sub(x, 0) in a processing chain."""
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 4])

        w = np.eye(4).astype(np.float32)
        w_init = onnx.numpy_helper.from_array(w, name='w')

        # Use scalar zero constant
        const_zero = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['zero'],
            value=onnx.helper.make_tensor(
                name='zero_val',
                data_type=onnx.TensorProto.FLOAT,
                dims=[],  # scalar
                vals=[0.0]
            )
        )

        matmul_node = onnx.helper.make_node(
            'MatMul', ['input', 'w'], ['mm_out'])
        # Sub(mm_out, 0) - ineffective
        sub_node = onnx.helper.make_node('Sub', ['mm_out', 'zero'], ['sub_out'])
        relu_node = onnx.helper.make_node('Relu', ['sub_out'], ['output'])

        graph = onnx.helper.make_graph(
            [const_zero, matmul_node, sub_node, relu_node],
            'ineffective_chain_graph',
            [input_tensor],
            [output_tensor],
            [w_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_compress_onnx_remove_ineffective_flag_import(self):
        """Test that compress_onnx accepts remove_ineffective_layers flag."""
        from onnx_opt import compress_onnx

        # Just test that the flag is recognized without errors
        # The actual optimization behavior depends on graph structure
        model = self._create_model_with_ineffective_sub_in_chain()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            # The flag should be accepted without raising an exception
            result = compress_onnx(
                model, old_path, save_path, ['remove_ineffective_layers'])
            self.assertIsNotNone(result)
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestCompressOnnxRemoveLastLayer(unittest.TestCase):
    """Tests for compress_onnx with last layer removal flags."""

    def _create_model_with_relu_last(self):
        """Create a model with ReLU as the last layer."""
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 4])

        w = np.random.randn(4, 4).astype(np.float32)
        w_init = onnx.numpy_helper.from_array(w, name='w')

        nodes = [
            onnx.helper.make_node('MatMul', ['input', 'w'], ['mm_out']),
            onnx.helper.make_node('Relu', ['mm_out'], ['output']),
        ]

        graph = onnx.helper.make_graph(
            nodes,
            'relu_last_graph',
            [input_tensor],
            [output_tensor],
            [w_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_compress_onnx_remove_relu_in_last_layer(self):
        """Test compress_onnx with remove_relu_in_last_layer flag."""
        from onnx_opt import compress_onnx

        model = self._create_model_with_relu_last()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(
                model, old_path, save_path, ['remove_relu_in_last_layer'])

            self.assertIsNotNone(result)
            onnx.checker.check_model(result)
            # ReLU should be removed
            self.assertEqual(len(result.graph.node), 1)
            self.assertEqual(result.graph.node[0].op_type, 'MatMul')
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)

    def _create_model_with_squeeze_last(self):
        """Create a model with Squeeze as the last layer after some ops."""
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 4])

        w = np.random.randn(4, 4).astype(np.float32)
        w_init = onnx.numpy_helper.from_array(w, name='w')

        matmul_node = onnx.helper.make_node(
            'MatMul', ['input', 'w'], ['mm_out'])

        # Squeeze node without axes attribute (old style)
        squeeze_node = onnx.helper.make_node(
            'Squeeze',
            inputs=['mm_out'],
            outputs=['output']
        )

        graph = onnx.helper.make_graph(
            [matmul_node, squeeze_node],
            'squeeze_last_graph',
            [input_tensor],
            [output_tensor],
            [w_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 11  # Squeeze without axes attr
        return model

    def test_compress_onnx_remove_squeeze_in_last_layer(self):
        """Test compress_onnx with remove_squeeze_in_last_layer flag."""
        from onnx_opt import compress_onnx

        model = self._create_model_with_squeeze_last()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(
                model, old_path, save_path, ['remove_squeeze_in_last_layer'])

            self.assertIsNotNone(result)
            # Squeeze should be removed, leaving only MatMul
            self.assertEqual(len(result.graph.node), 1)
            self.assertEqual(result.graph.node[0].op_type, 'MatMul')
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_remove_final_squeeze_reshape_output_wrapper(self):
        """Remove the final Squeeze->Reshape([1, 1]) wrapper."""
        from onnx_opt import compress_onnx

        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 1])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 1])
        w_init = onnx.numpy_helper.from_array(
            np.ones((1, 1), dtype=np.float32), name='w')
        shape_init = onnx.numpy_helper.from_array(
            np.array([1, 1], dtype=np.int64), name='output_shape')
        nodes = [
            onnx.helper.make_node(
                'MatMul', ['input', 'w'], ['mm_out']),
            onnx.helper.make_node(
                'Squeeze', ['mm_out'], ['scalar_or_vector']),
            onnx.helper.make_node(
                'Reshape', ['scalar_or_vector', 'output_shape'], ['output']),
        ]
        graph = onnx.helper.make_graph(
            nodes, 'squeeze_reshape_output_wrapper',
            [input_tensor], [output_tensor], [w_init, shape_init])
        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 11

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            onnx.save(model, old_path)
            result = compress_onnx(
                model, old_path, save_path,
                ['remove_squeeze_in_last_layer'])

            onnx.checker.check_model(result)
            self.assertEqual(len(result.graph.node), 1)
            self.assertEqual(result.graph.node[0].op_type, 'MatMul')
            self.assertEqual(result.graph.output[0].name, 'mm_out')
            self.assertEqual(
                [dim.dim_value for dim in
                 result.graph.output[0].type.tensor_type.shape.dim],
                [1, 1])
            self.assertEqual(
                [initializer.name for initializer in result.graph.initializer],
                ['w'])
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestCompressOnnxMultipleFlags(unittest.TestCase):
    """Tests for compress_onnx with multiple optimization flags."""

    def _create_complex_model(self):
        """Create a more complex model for testing multiple flags."""
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 4])

        w1 = np.eye(4).astype(np.float32)
        b1 = np.zeros(4).astype(np.float32)
        w2 = np.eye(4).astype(np.float32)
        b2 = np.zeros(4).astype(np.float32)

        w1_init = onnx.numpy_helper.from_array(w1, name='w1')
        b1_init = onnx.numpy_helper.from_array(b1, name='b1')
        w2_init = onnx.numpy_helper.from_array(w2, name='w2')
        b2_init = onnx.numpy_helper.from_array(b2, name='b2')

        nodes = [
            onnx.helper.make_node('MatMul', ['input', 'w1'], ['mm1']),
            onnx.helper.make_node('Add', ['mm1', 'b1'], ['add1']),
            onnx.helper.make_node('Relu', ['add1'], ['relu1']),
            onnx.helper.make_node('MatMul', ['relu1', 'w2'], ['mm2']),
            onnx.helper.make_node('Add', ['mm2', 'b2'], ['add2']),
            onnx.helper.make_node('Relu', ['add2'], ['output']),
        ]

        graph = onnx.helper.make_graph(
            nodes,
            'complex_graph',
            [input_tensor],
            [output_tensor],
            [w1_init, b1_init, w2_init, b2_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 13
        return model

    def test_compress_onnx_multiple_flags(self):
        """Test compress_onnx with multiple flags."""
        from onnx_opt import compress_onnx

        model = self._create_complex_model()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(
                model, old_path, save_path,
                ['merge_linear', 'remove_relu_in_last_layer'])

            self.assertIsNotNone(result)
            onnx.checker.check_model(result)
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestCompressOnnxDuplicateUpsample(unittest.TestCase):
    """Tests for compress_onnx with upsample deduplication."""

    def _create_model_with_duplicate_upsample(self):
        """Create a model with duplicate Upsample initializers."""
        input_tensor = onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, 3, 4, 4])
        output_tensor = onnx.helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, [1, 3, 8, 8])

        # Scales for upsample (2x upsampling)
        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        scales_init = onnx.numpy_helper.from_array(scales, name='scales')

        # Two upsample nodes sharing the same scales initializer
        upsample1 = onnx.helper.make_node(
            'Upsample',
            name='Upsample_1',
            inputs=['input', 'scales'],
            outputs=['up1'],
            mode='nearest'
        )

        upsample2 = onnx.helper.make_node(
            'Upsample',
            name='Upsample_2',
            inputs=['up1', 'scales'],
            outputs=['output'],
            mode='nearest'
        )

        graph = onnx.helper.make_graph(
            [upsample1, upsample2],
            'upsample_graph',
            [input_tensor],
            [output_tensor],
            [scales_init]
        )

        model = onnx.helper.make_model(graph, producer_name='test')
        model.opset_import[0].version = 9  # Upsample op version
        return model

    def test_compress_onnx_check_duplicate_upsample(self):
        """Test compress_onnx with check_duplicate_upsample_initializers flag."""
        from onnx_opt import compress_onnx

        model = self._create_model_with_duplicate_upsample()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            old_path = f.name
            onnx.save(model, old_path)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            result = compress_onnx(
                model, old_path, save_path,
                ['check_duplicate_upsample_initializers'])

            self.assertIsNotNone(result)
            onnx.checker.check_model(result)
        finally:
            if os.path.exists(old_path):
                os.unlink(old_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestConvOutputShapeComplex(unittest.TestCase):
    """More complex tests for conv_output_shape."""

    def test_mobilenet_depthwise(self):
        """Test MobileNet-style depthwise convolution."""
        # MobileNet depthwise: 3x3 kernel, stride 1 or 2
        h, w = conv_output_shape(h_w=(56, 56), kernel_size=3, stride=1, pad=1)
        self.assertEqual(h, 56)
        self.assertEqual(w, 56)

        h, w = conv_output_shape(h_w=(56, 56), kernel_size=3, stride=2, pad=1)
        self.assertEqual(h, 28)
        self.assertEqual(w, 28)

    def test_inception_various_kernels(self):
        """Test Inception-style various kernel sizes."""
        # 1x1
        h, w = conv_output_shape(h_w=(35, 35), kernel_size=1, stride=1, pad=0)
        self.assertEqual((h, w), (35, 35))

        # 3x3
        h, w = conv_output_shape(h_w=(35, 35), kernel_size=3, stride=1, pad=1)
        self.assertEqual((h, w), (35, 35))

        # 5x5
        h, w = conv_output_shape(h_w=(35, 35), kernel_size=5, stride=1, pad=2)
        self.assertEqual((h, w), (35, 35))

    def test_vgg_style(self):
        """Test VGG-style convolutions."""
        # VGG: 3x3 kernels, padding 1
        sizes = [224, 112, 56, 28, 14]
        for size in sizes:
            h, w = conv_output_shape(h_w=(size, size), kernel_size=3, stride=1, pad=1)
            self.assertEqual((h, w), (size, size))


class TestFuseConvAndBnEdgeCases(unittest.TestCase):
    """Edge case tests for fuse_conv_and_bn."""

    def test_fusion_with_small_eps(self):
        """Test fusion with small epsilon."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(8, eps=1e-10)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            x = torch.randn(1, 3, 8, 8)
            orig_out = bn(conv(x))
            fused_out = fused(x)

            self.assertTrue(torch.allclose(orig_out, fused_out, atol=1e-4))

    def test_fusion_with_momentum(self):
        """Test fusion with different momentum values."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(8, momentum=0.5)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            x = torch.randn(1, 3, 8, 8)
            orig_out = bn(conv(x))
            fused_out = fused(x)

            self.assertTrue(torch.allclose(orig_out, fused_out, atol=1e-5))

    def test_fusion_large_batch_norm_values(self):
        """Test fusion with large batch norm values."""
        with torch.no_grad():
            conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(8)
            # Set large values
            bn.weight.data.fill_(10.0)
            bn.bias.data.fill_(5.0)
            bn.running_mean.fill_(2.0)
            bn.running_var.fill_(4.0)
            bn.eval()

            fused = fuse_conv_and_bn(conv, bn)

            x = torch.randn(1, 3, 8, 8)
            orig_out = bn(conv(x))
            fused_out = fused(x)

            self.assertTrue(torch.allclose(orig_out, fused_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
