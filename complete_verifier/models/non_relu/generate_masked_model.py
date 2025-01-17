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
import torch
from collections import OrderedDict

s = torch.load('cifar_conv_small_sigmoid.pth')['state_dict']

n = OrderedDict()

names_table = ['conv1', '', 'conv2', '', '', 'linear1', '', 'linear2']

sizes = [(16,15,15), (32,6,6), (100,)]
mask_counter = 0

for k in s:
    layer, name = k.split('.')
    new_name = names_table[int(layer)] + '.' + name
    print(f'{k} -> {new_name}')
    n[new_name] = s[k]
    if name == 'bias' and mask_counter < len(sizes):
        shape = sizes[mask_counter]
        n[f'linear_masked_sigmoid{mask_counter+1}.mask'] = torch.zeros(size=shape)
        n[f'linear_masked_sigmoid{mask_counter+1}.slope'] = torch.ones(size=shape)
        n[f'linear_masked_sigmoid{mask_counter+1}.bias'] = torch.zeros(size=shape)
        mask_counter += 1

print(n.keys())
torch.save(n, 'cifar_conv_small_sigmoid_masked.pth')
