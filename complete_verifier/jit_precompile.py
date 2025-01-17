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
import time
import torch.nn as nn
import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from beta_CROWN_solver import LiRPANet
import numpy as np


class SimpleModelForJIT(nn.Module):
    def __init__(self):
        super(SimpleModelForJIT, self).__init__()
        self.fc1 = torch.nn.Linear(3, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 3)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


def precompile_jit_kernels():
    # TODO: also add attack?
    print('Pre-compile jit kernels on a toy network...')
    start_time = time.time()
    device = arguments.Config["general"]["device"]
    model_ori = SimpleModelForJIT()
    data = torch.randn(1, 3, device=device)
    num_outputs = 3
    y = 0
    labels = torch.tensor([y]).long()
    c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
    I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
    c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))
    # TODO We should use a BoundedModule instead. See https://github.com/Verified-Intelligence/Verifier_Development/pull/248#discussion_r1376529149
    model = LiRPANet(model_ori, in_size=data.shape, c=c)

    data_lb = torch.tensor([[-2.5, -2.5, 0.]], device=device)
    data_ub = torch.tensor([[2.5, 2.5, 5.0]], device=device)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.02, x_L = data_lb, x_U = data_ub)
    x = BoundedTensor(data, ptb).to(device)

    lb, ub, aux = model.net.init_alpha((x, ), share_alphas=True, c=c, bound_upper=False)
    model.net.set_bound_opts({'optimize_bound_args': {'iteration': 2, 'use_float64_in_last_iteration': False}})
    # Output constraints might be activated for use on the real model. But for this toy model it must not be used, as the settings won't fit
    model.net.set_bound_opts({'optimize_bound_args': {'apply_output_constraints_to': []}})
    ret = model.net.compute_bounds(x=(x,), method='CROWN-Optimized', C=c, bound_upper=False)

    del data, c, data_lb, data_ub, ptb, x, lb, ub, aux
    del model_ori, model, labels, ret, I

    if device == "cuda":
        torch.cuda.empty_cache()
    print(f'JIT kernels compiled in {time.time() - start_time:.4f}s.')
