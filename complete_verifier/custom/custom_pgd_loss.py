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
import math
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import arguments
import os
import subprocess
from load_model import Customized
from attack.attack_pgd import default_pgd_loss
import sys

def customized_gtrsb_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const,
               gama_lambda=0, threshold=-1e-5, mode='hinge', model=None):
    # first return the original loss
    loss, loss_gama = default_pgd_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const,
               gama_lambda, threshold, mode)
    signmerge_loss = torch.zeros_like(loss, device=loss.device)
    signmerge_losses = []

    _, num_restarts, num_specs, _ = output.shape

    # pick up all SignMerge layers from the model
    model_layers = model._modules.keys()
    signmerge_layers = []
    for layer_id in model_layers:
        if "/merge" in layer_id:
            signmerge_layers.append(layer_id)

    threshold = 1e-4
    scaler = 10

    for layer_id in signmerge_layers[1:]:   # ignore the first SignMerge layer since there isn't any error
        input_signmerge = model.get_forward_value(model[layer_id].inputs[0])    # the input of this SignMerge layer
        elementwise_loss = torch.clamp(threshold - torch.abs(input_signmerge), min=0)
        elementwise_loss = elementwise_loss.view(num_restarts * num_specs, -1)
        layer_loss = torch.mean(elementwise_loss, dim=1)
        signmerge_losses.append(-layer_loss.view(loss.shape))

    signmerge_loss = torch.mean(torch.stack(signmerge_losses), dim=0) / (scaler * threshold)

    loss_gama += torch.sum(signmerge_loss)

    return loss, loss_gama

