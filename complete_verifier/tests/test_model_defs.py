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
"""Unit tests for model_defs.py module."""

import pytest
import torch
import torch.nn as nn


# ============================================================================
# BasicBlock Tests
# ============================================================================

class TestBasicBlock:
    """Tests for the BasicBlock class."""

    def test_basic_block_default_params(self):
        """Test BasicBlock with default parameters."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=1)
        assert block is not None
        assert block.bn is True
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'bn1')
        assert hasattr(block, 'bn2')

    def test_basic_block_without_bn(self):
        """Test BasicBlock without batch normalization."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=1, bn=False)
        assert block.bn is False
        assert not hasattr(block, 'bn1')
        assert not hasattr(block, 'bn2')

    def test_basic_block_stride_2(self):
        """Test BasicBlock with stride=2 (downsampling)."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=2)
        assert len(block.shortcut) > 0  # Should have shortcut connection

    def test_basic_block_kernel_2(self):
        """Test BasicBlock with kernel size 2."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=1, kernel=2)
        assert block.conv1.kernel_size == (2, 2)

    def test_basic_block_kernel_1(self):
        """Test BasicBlock with kernel size 1."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=1, kernel=1)
        assert block.conv1.kernel_size == (1, 1)

    def test_basic_block_forward_with_bn(self):
        """Test forward pass with batch normalization."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=16, stride=1, bn=True)
        block.eval()
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert out.shape == (1, 16, 8, 8)

    def test_basic_block_forward_without_bn(self):
        """Test forward pass without batch normalization."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=16, stride=1, bn=False)
        block.eval()
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert out.shape == (1, 16, 8, 8)

    def test_basic_block_expansion(self):
        """Test that expansion attribute is 1."""
        from model_defs import BasicBlock

        assert BasicBlock.expansion == 1


# ============================================================================
# ResNet Tests
# ============================================================================

class TestResNet:
    """Tests for the ResNet class."""

    def test_resnet_creation(self):
        """Test creating a ResNet model."""
        from model_defs import ResNet, BasicBlock

        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, in_planes=64)
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        assert hasattr(model, 'layer4')

    def test_resnet18_factory(self):
        """Test ResNet18 factory function."""
        from model_defs import ResNet18

        model = ResNet18(in_planes=2)
        assert model is not None
        assert model.in_planes == 16  # After making layers

    def test_resnet_forward(self):
        """Test ResNet forward pass."""
        from model_defs import ResNet, BasicBlock

        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10, in_planes=8)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# CResNet5 Tests
# ============================================================================

class TestCResNet5:
    """Tests for the CResNet5 class."""

    def test_cresnet5_avg_with_bn(self):
        """Test CResNet5 with avg pooling and batch norm."""
        from model_defs import CResNet5, BasicBlock

        model = CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")
        assert model is not None
        assert model.bn is True
        assert model.last_layer == "avg"

    def test_cresnet5_dense_without_bn(self):
        """Test CResNet5 with dense layer without batch norm."""
        from model_defs import CResNet5, BasicBlock

        model = CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
        assert model is not None
        assert model.bn is False
        assert model.last_layer == "dense"

    def test_cresnet5_forward_avg(self):
        """Test CResNet5 forward pass with avg pooling."""
        from model_defs import cresnet5_8_avg_bn

        model = cresnet5_8_avg_bn()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_cresnet5_forward_dense(self):
        """Test CResNet5 forward pass with dense layer."""
        from model_defs import cresnet5_8_dense_bn

        model = cresnet5_8_dense_bn()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# CResNet7 Tests
# ============================================================================

class TestCResNet7:
    """Tests for the CResNet7 class."""

    def test_cresnet7_avg_with_bn(self):
        """Test CResNet7 with avg pooling and batch norm."""
        from model_defs import CResNet7, BasicBlock

        model = CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")
        assert model is not None
        assert model.bn is True

    def test_cresnet7_forward_avg(self):
        """Test CResNet7 forward pass with avg pooling."""
        from model_defs import cresnet7_8_avg_bn

        model = cresnet7_8_avg_bn()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_cresnet7_forward_dense(self):
        """Test CResNet7 forward pass with dense layer."""
        from model_defs import cresnet7_8_dense_bn

        model = cresnet7_8_dense_bn()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestModelFactoryFunctions:
    """Tests for various model factory functions."""

    def test_resnet4b(self):
        """Test resnet4b factory."""
        from model_defs import resnet4b, CResNet7

        model = resnet4b()
        assert isinstance(model, CResNet7)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_resnet2b(self):
        """Test resnet2b factory."""
        from model_defs import resnet2b, CResNet5

        model = resnet2b()
        assert isinstance(model, CResNet5)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_cresnet5_16_dense_bn(self):
        """Test cresnet5_16_dense_bn factory."""
        from model_defs import cresnet5_16_dense_bn, CResNet5

        model = cresnet5_16_dense_bn()
        assert isinstance(model, CResNet5)
        assert model.bn is True

    def test_cresnet5_16_avg_bn(self):
        """Test cresnet5_16_avg_bn factory."""
        from model_defs import cresnet5_16_avg_bn, CResNet5

        model = cresnet5_16_avg_bn()
        assert isinstance(model, CResNet5)
        assert model.bn is True

    def test_cresnet5_4_dense_bn(self):
        """Test cresnet5_4_dense_bn factory."""
        from model_defs import cresnet5_4_dense_bn, CResNet5

        model = cresnet5_4_dense_bn()
        assert isinstance(model, CResNet5)

    def test_cresnet5_4_avg_bn(self):
        """Test cresnet5_4_avg_bn factory."""
        from model_defs import cresnet5_4_avg_bn, CResNet5

        model = cresnet5_4_avg_bn()
        assert isinstance(model, CResNet5)

    def test_cresnet7_4_dense_bn(self):
        """Test cresnet7_4_dense_bn factory."""
        from model_defs import cresnet7_4_dense_bn, CResNet7

        model = cresnet7_4_dense_bn()
        assert isinstance(model, CResNet7)

    def test_cresnet7_4_avg_bn(self):
        """Test cresnet7_4_avg_bn factory."""
        from model_defs import cresnet7_4_avg_bn, CResNet7

        model = cresnet7_4_avg_bn()
        assert isinstance(model, CResNet7)

    def test_cresnet5_16_dense(self):
        """Test cresnet5_16_dense factory (no bn)."""
        from model_defs import cresnet5_16_dense, CResNet5

        model = cresnet5_16_dense()
        assert isinstance(model, CResNet5)
        assert model.bn is False

    def test_cresnet5_16_avg(self):
        """Test cresnet5_16_avg factory (no bn)."""
        from model_defs import cresnet5_16_avg, CResNet5

        model = cresnet5_16_avg()
        assert isinstance(model, CResNet5)
        assert model.bn is False

    def test_cresnet5_4_dense(self):
        """Test cresnet5_4_dense factory (no bn)."""
        from model_defs import cresnet5_4_dense, CResNet5

        model = cresnet5_4_dense()
        assert isinstance(model, CResNet5)
        assert model.bn is False

    def test_cresnet5_4_avg(self):
        """Test cresnet5_4_avg factory (no bn)."""
        from model_defs import cresnet5_4_avg, CResNet5

        model = cresnet5_4_avg()
        assert isinstance(model, CResNet5)
        assert model.bn is False

    def test_cresnet7_4_dense(self):
        """Test cresnet7_4_dense factory (no bn)."""
        from model_defs import cresnet7_4_dense, CResNet7

        model = cresnet7_4_dense()
        assert isinstance(model, CResNet7)
        assert model.bn is False

    def test_cresnet7_4_avg(self):
        """Test cresnet7_4_avg factory (no bn)."""
        from model_defs import cresnet7_4_avg, CResNet7

        model = cresnet7_4_avg()
        assert isinstance(model, CResNet7)
        assert model.bn is False


# ============================================================================
# Dense and DenseSequential Tests
# ============================================================================

class TestDense:
    """Tests for the Dense class."""

    def test_dense_creation(self):
        """Test creating Dense module."""
        from model_defs import Dense

        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(10, 5)
        dense = Dense(linear1, linear2)

        assert dense is not None
        assert len(dense.Ws) == 2

    def test_dense_forward(self):
        """Test Dense forward pass."""
        from model_defs import Dense

        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(10, 5)
        dense = Dense(linear1, linear2)

        x1 = torch.randn(2, 10)
        x2 = torch.randn(2, 10)
        out = dense(x1, x2)

        assert out.shape == (2, 5)

    def test_dense_out_features(self):
        """Test Dense out_features attribute."""
        from model_defs import Dense

        linear = nn.Linear(10, 5)
        dense = Dense(linear)

        assert dense.out_features == 5

    def test_dense_empty(self):
        """Test Dense with no modules."""
        from model_defs import Dense

        dense = Dense()
        assert len(dense.Ws) == 0


class TestDenseSequential:
    """Tests for the DenseSequential class."""

    def test_dense_sequential_forward(self):
        """Test DenseSequential forward pass."""
        from model_defs import DenseSequential, Dense

        seq = DenseSequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(2, 10)
        out = seq(x)
        assert out.shape == (2, 5)


# ============================================================================
# model_resnet Tests
# ============================================================================

class TestModelResnet:
    """Tests for model_resnet function."""

    def test_model_resnet_forward(self):
        """Test model_resnet forward pass."""
        from model_defs import model_resnet, DenseSequential

        model = model_resnet(in_ch=3, in_dim=32, width=1, mult=8, N=1)
        assert isinstance(model, DenseSequential)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# MNIST FC Model Tests
# ============================================================================

class TestMNISTFC:
    """Tests for MNIST fully-connected models."""

    def test_mnist_6_100(self):
        """Test mnist_6_100 factory."""
        from model_defs import mnist_6_100

        model = mnist_6_100()
        assert isinstance(model, nn.Sequential)
        model.eval()
        x = torch.randn(1, 784)
        out = model(x)
        assert out.shape == (1, 10)

    def test_mnist_9_200(self):
        """Test mnist_9_200 factory."""
        from model_defs import mnist_9_200

        model = mnist_9_200()
        assert isinstance(model, nn.Sequential)
        model.eval()
        x = torch.randn(1, 784)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# CIFAR Model Tests
# ============================================================================

class TestCIFARModels:
    """Tests for CIFAR models."""

    def test_cifar_model_base(self):
        """Test cifar_model_base factory."""
        from model_defs import cifar_model_base

        model = cifar_model_base()
        assert isinstance(model, nn.Sequential)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_cifar_model_wide(self):
        """Test cifar_model_wide factory."""
        from model_defs import cifar_model_wide

        model = cifar_model_wide()
        assert isinstance(model, nn.Sequential)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_cifar_model_deep(self):
        """Test cifar_model_deep factory."""
        from model_defs import cifar_model_deep

        model = cifar_model_deep()
        assert isinstance(model, nn.Sequential)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# getShapeConv Tests
# ============================================================================

class TestGetShapeConv:
    """Tests for getShapeConv function."""

    def test_basic_convolution_shape(self):
        """Test basic convolution output shape calculation."""
        from model_defs import getShapeConv

        in_shape = (3, 32, 32)  # 3 channels, 32x32
        conv_shape = (64, 3, 3)  # 64 output channels, 3x3 kernel
        result = getShapeConv(in_shape, conv_shape, stride=1, padding=1)

        assert result == (64, 32, 32)  # Same spatial size with padding=1

    def test_convolution_with_stride(self):
        """Test convolution with stride > 1."""
        from model_defs import getShapeConv

        in_shape = (3, 32, 32)
        conv_shape = (64, 3, 3)
        result = getShapeConv(in_shape, conv_shape, stride=2, padding=1)

        assert result == (64, 16, 16)  # Halved spatial size

    def test_convolution_no_padding(self):
        """Test convolution without padding."""
        from model_defs import getShapeConv

        in_shape = (3, 32, 32)
        conv_shape = (64, 3, 3)
        result = getShapeConv(in_shape, conv_shape, stride=1, padding=0)

        assert result == (64, 30, 30)


# ============================================================================
# BasicBlock_eth Tests
# ============================================================================

class TestBasicBlockEth:
    """Tests for the BasicBlock_eth class."""

    def test_basic_block_eth_creation(self):
        """Test BasicBlock_eth creation stores correct parameters."""
        from model_defs import BasicBlock_eth

        block = BasicBlock_eth(
            in_planes=16, planes=32, stride=1, bn=True, kernel=3, in_dim=32
        )
        assert block.in_planes == 16
        assert block.planes == 32
        assert block.stride == 1
        assert block.bn is True
        assert block.kernel == 3
        assert hasattr(block, 'path_a')
        assert hasattr(block, 'path_b')

    def test_basic_block_eth_without_bn(self):
        """Test BasicBlock_eth without batch normalization has no BatchNorm layers."""
        from model_defs import BasicBlock_eth

        block = BasicBlock_eth(
            in_planes=16, planes=32, stride=1, bn=False, kernel=3, in_dim=32
        )
        assert block.bn is False
        # Verify path_b has no BatchNorm layers when bn=False
        bn_layers = [m for m in block.path_b.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 0

    def test_basic_block_eth_with_bn(self):
        """Test BasicBlock_eth with batch normalization has BatchNorm layers."""
        from model_defs import BasicBlock_eth

        block = BasicBlock_eth(
            in_planes=16, planes=32, stride=1, bn=True, kernel=3, in_dim=32
        )
        assert block.bn is True
        # Verify path_b has BatchNorm layers when bn=True
        bn_layers = [m for m in block.path_b.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) > 0

    def test_basic_block_eth_stride_variation(self):
        """Test BasicBlock_eth with different stride values."""
        from model_defs import BasicBlock_eth

        block_stride1 = BasicBlock_eth(
            in_planes=16, planes=16, stride=1, bn=True, kernel=3, in_dim=32
        )
        block_stride2 = BasicBlock_eth(
            in_planes=16, planes=16, stride=2, bn=True, kernel=3, in_dim=32
        )
        assert block_stride1.stride == 1
        assert block_stride2.stride == 2
        # Stride 2 should produce smaller output dimension
        assert block_stride2.out_dim < block_stride1.out_dim

    def test_basic_block_eth_kernel_variation(self):
        """Test BasicBlock_eth with different kernel sizes."""
        from model_defs import BasicBlock_eth

        block_k1 = BasicBlock_eth(
            in_planes=16, planes=32, stride=1, bn=True, kernel=1, in_dim=32
        )
        block_k3 = BasicBlock_eth(
            in_planes=16, planes=32, stride=1, bn=True, kernel=3, in_dim=32
        )
        assert block_k1.kernel == 1
        assert block_k3.kernel == 3

    def test_basic_block_eth_forward(self):
        """Test BasicBlock_eth forward pass produces correct output shape."""
        from model_defs import BasicBlock_eth

        block = BasicBlock_eth(
            in_planes=16, planes=16, stride=1, bn=True, kernel=3, in_dim=32
        )
        block.eval()
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 16  # planes * expansion

    def test_basic_block_eth_expansion(self):
        """Test BasicBlock_eth expansion attribute."""
        from model_defs import BasicBlock_eth

        assert BasicBlock_eth.expansion == 1


# ============================================================================
# ResNet_eth Tests
# ============================================================================

class TestResNetEth:
    """Tests for ResNet_eth class."""

    def test_resnet_eth_creation(self):
        """Test ResNet_eth creation stores correct in_planes."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=True,
            last_layer="avg"
        )
        assert isinstance(model, nn.Sequential)
        # First layer should be Conv2d with in_ch input channels
        assert isinstance(model[0], nn.Conv2d)
        assert model[0].in_channels == 3

    def test_resnet_eth_with_bn(self):
        """Test ResNet_eth with batch normalization has BatchNorm layers."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=True,
            last_layer="avg"
        )
        # With bn=True, should have BatchNorm2d layer after first Conv2d
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) > 0

    def test_resnet_eth_without_bn(self):
        """Test ResNet_eth without batch normalization has no BatchNorm layers."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=False,
            last_layer="avg"
        )
        # With bn=False, should have no BatchNorm layers
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 0

    def test_resnet_eth_avg_last_layer(self):
        """Test ResNet_eth with avg last layer has AvgPool2d."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=True,
            last_layer="avg"
        )
        # With last_layer="avg", should have AvgPool2d
        avgpool_layers = [m for m in model.modules() if isinstance(m, nn.AvgPool2d)]
        assert len(avgpool_layers) == 1

    def test_resnet_eth_dense_last_layer(self):
        """Test ResNet_eth with dense last layer has two Linear layers at end."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=True,
            last_layer="dense"
        )
        # With last_layer="dense", should have no AvgPool2d
        avgpool_layers = [m for m in model.modules() if isinstance(m, nn.AvgPool2d)]
        assert len(avgpool_layers) == 0
        # Should end with Linear layer outputting num_classes
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) >= 2
        # Last Linear should output num_classes (10)
        assert linear_layers[-1].out_features == 10

    def test_resnet_eth_num_classes(self):
        """Test ResNet_eth respects num_classes parameter."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model_10 = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=True,
            last_layer="avg"
        )
        model_100 = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=100,
            in_planes=8,
            bn=True,
            last_layer="avg"
        )
        # Final Linear layers should have different out_features
        linear_10 = [m for m in model_10.modules() if isinstance(m, nn.Linear)][-1]
        linear_100 = [m for m in model_100.modules() if isinstance(m, nn.Linear)][-1]
        assert linear_10.out_features == 10
        assert linear_100.out_features == 100

    def test_resnet_eth_in_planes_variation(self):
        """Test ResNet_eth with different in_planes values."""
        from model_defs import ResNet_eth, BasicBlock_eth

        model_8 = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=8,
            bn=False,
            last_layer="avg"
        )
        model_16 = ResNet_eth(
            BasicBlock_eth,
            in_ch=3,
            num_stages=1,
            num_blocks=2,
            num_classes=10,
            in_planes=16,
            bn=False,
            last_layer="avg"
        )
        # First conv should have different out_channels
        assert model_8[0].out_channels == 8
        assert model_16[0].out_channels == 16


# ============================================================================
# ResNet_eth Factory Functions Tests
# ============================================================================

class TestResNetEthFactoryFunctions:
    """Tests for ResNet_eth factory functions."""

    def test_resnet2b_eth(self):
        """Test resnet2b_eth factory."""
        from model_defs import resnet2b_eth, ResNet_eth

        model = resnet2b_eth(bn=False)
        assert isinstance(model, ResNet_eth)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_resnet2b2_eth(self):
        """Test resnet2b2_eth factory."""
        from model_defs import resnet2b2_eth, ResNet_eth

        model = resnet2b2_eth(bn=True)
        assert isinstance(model, ResNet_eth)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_resnet4b1(self):
        """Test resnet4b1 factory."""
        from model_defs import resnet4b1, ResNet_eth

        model = resnet4b1(bn=True)
        assert isinstance(model, ResNet_eth)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_resnet4b2(self):
        """Test resnet4b2 factory."""
        from model_defs import resnet4b2, ResNet_eth

        model = resnet4b2(bn=True)
        assert isinstance(model, ResNet_eth)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)


# ============================================================================
# Edge Cases and Error Tests
# ============================================================================

class TestModelDefsEdgeCases:
    """Edge case tests for model_defs."""

    def test_basic_block_different_in_out_planes(self):
        """Test BasicBlock with different in/out planes."""
        from model_defs import BasicBlock

        block = BasicBlock(in_planes=16, planes=32, stride=2, bn=True)
        block.eval()
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert out.shape[1] == 32

    def test_dense_single_layer(self):
        """Test Dense with single layer."""
        from model_defs import Dense

        linear = nn.Linear(10, 5)
        dense = Dense(linear)
        x = torch.randn(2, 10)
        out = dense(x)
        assert out.shape == (2, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
