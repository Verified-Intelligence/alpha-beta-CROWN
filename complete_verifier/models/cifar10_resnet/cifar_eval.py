#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from resnet import resnet2b, resnet4b

cifar10_mean = (0.4914, 0.4822, 0.4465)  # np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def load_data(data_dir: str = "../../data", num_imgs: int = 25, random: bool = False) -> tuple:

    """
    Loads the cifar10 data.

    Args:
        data_dir:
            The directory to store the full CIFAR10 dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))

def clean_acc(model_name, images, labels):
    
    """
    Load the resnet models and measure the clean accuracy
    Args:
        model_name:
            Either resnet2b or resnet4b
        images:
            The test images with pixel value 0-1
        labels:
            True labels of testing images
    """

    model = eval(model_name)()
    model.load_state_dict(torch.load(model_name + '.pth')["state_dict"])
    model, images, labels = model.cuda(), images.cuda(), labels.cuda()
    output = model(normalize(images))
    acc = (output.max(1)[1]==labels).sum().item()
    return acc


def main():
    num_imgs = 10000
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    images, labels = load_data(num_imgs=num_imgs, random=False)
    print("ResNet-2B clean acc: {} out of {}".format(clean_acc("resnet2b", images, labels), num_imgs))
    print("ResNet-4B clean acc: {} out of {}".format(clean_acc("resnet4b", images, labels), num_imgs))

if __name__ == "__main__":
    main()
