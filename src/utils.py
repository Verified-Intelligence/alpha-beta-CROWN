from collections import OrderedDict

from modules import Flatten
import math
import gzip
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
import pandas as pd
import onnx2pytorch
import onnx   
import onnxruntime as ort

########################################
# Defined the model architectures
########################################


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            # can do planes 32
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        # print("residual relu:", out.shape, out[0].view(-1).shape)
        return out



class CResNet5(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet5, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out


class CResNet7(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet7, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out



def cresnet5_16_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="dense")

def cresnet5_16_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="avg")


def cresnet5_8_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet5_8_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet5_4_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet5_4_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")


def cresnet7_8_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet7_8_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet7_4_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet7_4_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")




def cresnet5_16_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")

def cresnet5_16_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="avg")


def cresnet5_8_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet5_8_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet5_4_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet5_4_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


def cresnet7_8_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet7_8_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet7_4_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet7_4_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x, W in zip(xs, self.Ws) if W is not None)
        return out


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]


def model_resnet(in_ch=3, in_dim=32, width=1, mult=16, N=1):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)),
            nn.ReLU(),
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0),
                  None,
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)),
            nn.ReLU()
        ]

    conv1 = [nn.Conv2d(in_ch, mult, 3, stride=1, padding=3 if in_dim == 28 else 1), nn.ReLU()]
    conv2 = block(mult, mult * width, 3, False)
    for _ in range(N):
        conv2.extend(block(mult * width, mult * width, 3, False))
    conv3 = block(mult * width, mult * 2 * width, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(mult * 2 * width, mult * 2 * width, 3, False))
    conv4 = block(mult * 2 * width, mult * 4 * width, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(mult * 4 * width, mult * 4 * width, 3, False))
    layers = (
            conv1 +
            conv2 +
            conv3 +
            conv4 +
            [Flatten(),
             nn.Linear(mult * 4 * width * 8 * 8, 1000),
             nn.ReLU(),
             nn.Linear(1000, 10)]
    )
    model = DenseSequential(
        *layers
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def mnist_fc():
    # cifar base
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    )
    return model


def cifar_model():
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    # cifar deep
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cnn_4layer():
    # cifar_cnn_a
    return cifar_model_wide()


def cnn_4layer_adv():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_adv4():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_mix4():
    # cifar_cnn_a_mix4
    return cifar_model_wide()


def cnn_4layer_b():
    # cifar_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cnn_4layer_b4():
    # cifar_cnn_b4
    return cnn_4layer_b()

def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def cifar_conv_small():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv_big():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_conv_small():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*5*5,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_big():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_6_100():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100, 10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_9_100():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_6_200():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_9_200():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_fc1():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    return model


def mnist_fc2():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fc3():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model




def mnist_fc_3_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_4_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_5_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_6_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_7_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv1():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv3():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2048, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv4():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv5():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv6():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w

def get_test_acc(model, input_shape, is_channel_last=False):
    import torchvision
    import torchvision.transforms as transforms
    if input_shape == (3, 32, 32):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif input_shape == (1, 28, 28):
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        raise RuntimeError("Unable to determine dataset for test accuracy.")
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    total = 0
    correct = 0
    for data in testloader:
        images, labels = data
        if is_channel_last:
            images = images.permute(0,2,3,1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=  (predicted == labels).sum().item()
    print(f'correct {correct} of {total}')


def load_onnx(path):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model


def inference_onnx(path, *inputs):
    print(inputs)
    sess = ort.InferenceSession(load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res


def load_model_onnx(path, input_shape, compute_test_acc=False, force_convert=False):
    # pip install onnx2pytorch
    onnx_model = load_onnx(path)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    input_shape = tuple(input_shape)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    if force_convert:
        new_modules = []
        modules = list(pytorch_model.modules())[1:]
        for mi, m in enumerate(modules):
            if isinstance(m, torch.nn.Linear):
                new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
                new_m.weight.data.copy_(m.weight.data)
                new_m.bias.data.copy_(m.bias)
                new_modules.append(new_m)
            elif isinstance(m, torch.nn.ReLU):
                new_modules.append(torch.nn.ReLU())
            elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
                new_modules.append(torch.nn.Flatten())
            else:
                raise NotImplementedError

        seq_model = nn.Sequential(*new_modules)
        return seq_model

    if len(input_shape) <= 2:
        return pytorch_model

    # Check model input shape.
    is_channel_last = False
    if onnx_shape != input_shape:
        # Change channel location.
        onnx_shape = onnx_shape[2:] + onnx_shape[:2]
        if onnx_shape == input_shape:
            is_channel_last = True
        else:
            print(f"Unexpected input shape in onnx: {onnx_shape}, given {input_shape}")

    # Fixup converted ONNX model. For ResNet we directly return; for other models, we convert them to a Sequential model.
    # We also need to handle NCHW and NHWC formats here.
    conv_c, conv_h, conv_w = input_shape
    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    need_permute = False
    for mi, m in enumerate(modules):
        if isinstance(m, onnx2pytorch.operations.add.Add):
            # ResNet model. No need to convert to sequential.
            return pytorch_model, is_channel_last
        if isinstance(m, torch.nn.Conv2d):
            # Infer the output size of conv.
            conv_h, conv_w = conv_output_shape((conv_h, conv_w), m.kernel_size, m.stride, m.padding)
            conv_c = m.weight.size(0)
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
            # May need to permute the next linear layer if the model was in NHWC format.
            need_permute = True and is_channel_last
        elif isinstance(m, torch.nn.Linear) and need_permute:
            # The original model is in NHWC format and we now have NCHW format, so the dense layer's weight must be adjusted.
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.view(m.weight.size(0), conv_h, conv_w, conv_c).permute(0, 3, 1, 2).contiguous().view(m.weight.size(0), -1))
            new_m.bias.data.copy_(m.bias)
            need_permute = False
            new_modules.append(new_m)
        elif isinstance(m, torch.nn.ReLU) and mi == (len(modules)-1):
            # not add relu if last layer is relu
            pass
        else:
            new_modules.append(m)

    seq_model = nn.Sequential(*new_modules)

    if compute_test_acc:
        get_test_acc(seq_model, input_shape)

    return seq_model, is_channel_last


def load_model(args, weights_loaded=True):
    """
    Load the model architectures and weights
    """
    model_ori = eval(args.model)()
    print(model_ori)

    if not weights_loaded:
        return model_ori

    if 'cnn_4layer' not in args.model:
        sd = torch.load(args.load, map_location=torch.device('cpu'))['state_dict']
        if type(sd) == list:
            sd = sd[0]
        elif type(sd) == OrderedDict:
            pass
        else:
            raise NotImplementedError
        model_ori.load_state_dict(sd)
    else:
        model_ori.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
        if args.model == "cnn_4layer_b":
            args.MODEL = "b_adv"
            print(args.data)
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "cnn_4layer_b4":
            args.MODEL = "b_adv4"
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "cnn_4layer":
            args.MODEL = "a_mix"
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "cnn_4layer_mix4":
            args.MODEL = "a_mix4"
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "cnn_4layer_adv":
            args.MODEL = "a_adv"
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "cnn_4layer_adv4":
            args.MODEL = "a_adv4"
            assert args.data == "CIFAR_SAMPLE" or args.data =="CIFAR_SDP"
        elif args.model == "mnist_cnn_4layer":
            args.MODEL = "mnist_a_adv"
            assert args.data == "MNIST_SAMPLE" or args.data =="MNIST_SDP"
        else:
            exit("model not supported!")

    return model_ori


########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Proprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    upper_limit, lower_limit = 1., 0.
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    X = np.load("../data/sample100_unnormalized/"+MODEL+"/X.npy")
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0,3,1,2))
    y = np.load("../data/sample100_unnormalized/"+MODEL+"/y.npy")
    runnerup = np.load("../data/sample100_unnormalized/"+MODEL+"/runnerup.npy")
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int))
    runnerup = torch.from_numpy(runnerup.astype(np.int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    X = np.load("../data/sample100_unnormalized/"+MODEL+"/X.npy")
    X = np.transpose(X, (0,3,1,2))
    y = np.load("../data/sample100_unnormalized/"+MODEL+"/y.npy")
    runnerup = np.load("../data/sample100_unnormalized/"+MODEL+"/runnerup.npy")
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int))
    runnerup = torch.from_numpy(runnerup.astype(np.int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset(args):
    """
    Load regular data; Robustness region defined in results pickle 
    """
    if args.data == 'MNIST':
        # dummy_input = torch.randn(1, 1, 28, 28)
        test_data = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
        test_data.mean = torch.tensor([0.0])
        test_data.std = torch.tensor([1.0])
        data_max = torch.tensor(1.)
        data_min = torch.tensor(0.)
    elif args.data == 'CIFAR':
        # dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
        test_data = datasets.CIFAR10("../data", train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]))
        test_data.mean = torch.tensor([0.485, 0.456, 0.406])
        test_data.std = torch.tensor([0.225, 0.225, 0.225])
        # set data_max and data_min to be None if no clip
        data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    elif args.data == 'CIFAR_norm':
        # dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        test_data = datasets.CIFAR10("../data", train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]))
        test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        test_data.std = torch.tensor([0.2471, 0.2435, 0.2616])
        # set data_max and data_min to be None if no clip
        data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset(args):
    """
    Load sampled data and define the robustness region
    """
    if args.data == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=args.MODEL)
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif args.data == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=args.MODEL)
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, runnerup, data_max, data_min, eps_temp


def load_sdp_dataset(args, eps_temp=None):
    if args.data == "CIFAR_SDP":
        X = np.load("../data/cifar_sdp/X_sdp.npy")
        X = preprocess_cifar(X)
        X = np.transpose(X, (0,3,1,2))
        y = np.load("../data/cifar_sdp/y_sdp.npy")
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int))
        runnerup = torch.from_numpy(runnerup.astype(np.int))

        if eps_temp is None: eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)

        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    elif args.data == "MNIST_SDP":
        X = np.load("../data/mnist_sdp/X_sdp.npy")
        X = np.transpose(X, (0,3,1,2))
        y = np.load("../data/mnist_sdp/y_sdp.npy")
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.int))
        runnerup = torch.from_numpy(runnerup.astype(np.int))

        if eps_temp is None: eps_temp = 0.3

        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    else:
        exit("sdp dataset not supported!")

    return X, y, runnerup, data_max, data_min, eps_temp

def load_eran_dataset(args, eps_temp=None):
    """
    Load sampled data and define the robustness region
    """
    if args.data == "CIFAR_ERAN":
        X = np.load("../data/cifar_eran/X_eran.npy")
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1,-1,1,1).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.201]).reshape(1,-1,1,1).astype(np.float32)
        X = (X-mean)/std

        labels = np.load("../data/cifar_eran/y_eran.npy")
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int))
        runnerup = torch.from_numpy(runnerup.astype(np.int))
        if eps_temp is None: eps_temp = 2./255.

        eps_temp = torch.tensor(eps_temp/std).reshape(1,-1,1,1)
        data_max = torch.tensor((1.-mean)/std).reshape(1,-1,1,1)
        data_min = torch.tensor((0.-mean)/std).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif args.data == "MNIST_ERAN":
        X = np.load("../data/mnist_eran/X_eran.npy")
        mean = 0.1307
        std = 0.3081
        X = (X-mean)/std

        labels = np.load("../data/mnist_eran/y_eran.npy")
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int))
        runnerup = torch.from_numpy(runnerup.astype(np.int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp/std).reshape(1,-1,1,1)
        data_max = torch.tensor((1.-mean)/std).reshape(1,-1,1,1)
        data_min = torch.tensor((0.-mean)/std).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif args.data == "MNIST_ERAN_UN":
        X = np.load("../data/mnist_eran/X_eran.npy")
        
        labels = np.load("../data/mnist_eran/y_eran.npy")
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int))
        runnerup = torch.from_numpy(runnerup.astype(np.int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    else:
        exit("dataset not supported!")

    return X, labels, runnerup, data_max, data_min, eps_temp


########################################
# Load results stored in the pickle files
########################################
def load_pickle_results(args):
    """
    Load results stored in the pickle files
    """
    if 'deep' in args.model:
        gt_results = pd.read_pickle('../data/deep_100.pkl')
        # js = json.load(open('compare_results/sdp_results_cnn-deep.json',))
    elif 'wide' in args.model:
        gt_results = pd.read_pickle('../data/wide_100.pkl')
    elif args.model == 'cnn_4layer_b':
        gt_results = pd.read_pickle('compare_results/cnn-b-adv.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == 'cnn_4layer_b4':
        gt_results = pd.read_pickle('compare_results/cnn-b-adv4.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == 'cnn_4layer':
        gt_results = pd.read_pickle('compare_results/cnn-a-mix.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == 'cnn_4layer_mix4':
        gt_results = pd.read_pickle('compare_results/cnn_a_mix4.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == 'cnn_4layer_adv':
        gt_results = pd.read_pickle('compare_results/cnn_a_adv.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == "mnist_cnn_4layer":
        gt_results = pd.read_pickle('compare_results/mnist-cnn-a-adv.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == "mnist_fc2":
        gt_results = pd.read_pickle('../data/our_models/mnist_fc2_adv8.pkl')
    elif args.model == "mnist_fc_5_512":
        gt_results = pd.read_pickle('../data/our_models/mnist_fc_5_512_adv0.3_eps_0.2.pkl')
    elif args.model == "mnist_fc_7_512":
        gt_results = pd.read_pickle('../data/our_models/mnist_fc_7_512_adv0.3_eps_0.15.pkl')
    elif args.model == "mnist_fc_6_512":
        gt_results = pd.read_pickle('../data/our_models/mnist_fc_6_512_adv0.3_eps_0.15.pkl')
    elif args.model == "cifar_conv1":
        gt_results = pd.read_pickle('../data/our_models/cifar_conv1_pgd8_eps_8.pkl')
    elif args.model == "cifar_conv2":
        gt_results = pd.read_pickle('../data/our_models/cifar_conv2_pgd8_eps_8.pkl')
    elif args.model == "cifar_conv4":
        gt_results = pd.read_pickle('../data/our_models/cifar_conv4_pgd8_eps_8.pkl')
    elif args.model == "cifar_conv6":
        gt_results = pd.read_pickle('../data/our_models/cifar_conv6_pgd8_eps_8.pkl')
    else:
        print('loading base_100!')
        gt_results = pd.read_pickle('../data/base_100.pkl')
    return gt_results


def convert_test_model(model_ori):
    # NOTE: It looks like `in_features` and `out_features` are in the wrong order
    # after converting the onnx model to pytorch model.
    # Swap them below.
    modules = []
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            layer = nn.Linear(m.out_features, m.in_features)
            layer.weight.data = m.weight.data.t().to(torch.float)
            layer.bias.data = m.bias.data.to(torch.float)
            modules.append(layer)
            # pdb.set_trace()
        else:
            modules.append(m)
    model_ori = nn.Sequential(*modules)

    return model_ori


def convert_nn4sys_model(model_ori):
    model_ori = nn.Sequential(*list(model_ori._modules.values()))
    # Split the model into v1 and v2 models to resolve numerical issues
    modules_v1 = []
    modules_v2 = []
    stage = 1
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            if m.weight.abs().max() > 1e9:
                stage = 2 if len(modules_v2) == 0 else 3
                continue
        else:
            continue
        if stage == 1:
            modules_v1 += [m, nn.ReLU(inplace=True)]
        elif stage == 2:
            dim = modules_v1[-2].out_features - 1
            lin = nn.Linear(m.in_features - dim, m.out_features - dim)
            lin.weight.data = m.weight[:lin.out_features, :lin.in_features]
            lin.weight = lin.weight.to(dtype=torch.float64)
            lin.bias.data = m.bias[:lin.out_features]
            lin.bias = lin.bias.to(dtype=torch.float64)
            modules_v2 += [lin, nn.ReLU(inplace=True)]
    x = torch.tensor([[119740.8]], dtype=torch.float64)                
    modules_v1 = modules_v1[:-1]
    model_v1 = nn.Sequential(*modules_v1)
    y = model_v1(x)
    dim = y.size(-1) - 1  
    modules_v2 = modules_v2[:-1]
    linear_ident = nn.Linear(1, dim, bias=False)
    linear_ident.weight.data = torch.ones_like(linear_ident.weight, dtype=torch.float64)
    modules_v2.insert(0, linear_ident)
    model_v2 = nn.Sequential(*modules_v2)
    y[:, :-2] *= (y[:, 1:-1] <= 0).int()
    select = (y[:, :-1] > 0).int()
    y2 = model_v2(x)
    y2 = y2[:] * select
    res = y2.sum(dim=-1, keepdim=True)
    res_ref = model_ori(x)
    print(res.item(), res_ref.item())
    # import pdb; pdb.set_trace()
    model_ori = (model_v1, model_v2, model_ori)   
    return model_ori
