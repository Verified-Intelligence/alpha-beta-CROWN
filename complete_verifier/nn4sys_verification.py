#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""A specialized verifier for the NN4sys benchmark."""
import math
import time
import pdb
import arguments

from utils import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
from collections import deque
import numpy as np

@torch.no_grad()
def compute_bounds(model_v1, model_v2, input_l, input_u, x=None, eps=None):
    if x is None or eps is None:
        x = (input_l + input_u) / 2
        eps = (input_u - input_l) / 2
    ptb = PerturbationLpNorm(norm=np.inf, x_L=input_l, x_U=input_u)
    x = BoundedTensor(x, ptb).to(x.device)
    l, u = model_v1.compute_bounds(x=(x,), method='forward')
    # check that there won't be two positions with positive values after the first 1e10 layer    
    margin = ( (u<=0).int() * 1e20 + (u>0).int() * l )[:, :-1].min(dim=-1).values
    safe_numerical_1 = (u[:, :-1].max(dim=-1).values - margin * 1e10 < 0)
    # check that there will be at least one positive with a positive value after the first 1e10 layer
    safe_numerical_2 = (l[:, 0] > 1e-8) 

    select = (u[:, :-1] > 0).int()
    select[:, :-1] *= ( u[:,:-2] - l[:, 1:-1]*1e10 >= 1e-8).int() # 1e-8 is a tolerance term
    l, u = model_v2.compute_bounds(x=(x,), method='forward')
    l_final = torch.min(select * l + (1 - select) * (1e10), dim=-1).values
    u_final = torch.max(select * u + (1 - select) * (-1e10), dim=-1).values  

    safe_numerical = (safe_numerical_1 * safe_numerical_2).int()
    l_final = l_final * safe_numerical + l_final.clamp(max=0) * (1 - safe_numerical)

    return l_final, u_final

@torch.no_grad()
def verify(model_ori, model_v1, model_v2, data_ub=None, data_lb=None, spec=None):
    device = arguments.Config["general"]["device"]

    data_lb, data_ub = data_lb.to(device), data_ub.to(device)
    data = (data_lb + data_ub) / 2
    # use forward mode
    # y = model(x) # NOTE currently must call a clean forward before computing forward bounds
    queue = deque() # a queue of input segments to be verified
    queue.append((data_lb, data_ub))
    verified_regions = []
    base = None
    res_l, res_u = None, None
    max_queue_size = int(1e6) # TODO confirm it
    input_length = float(data_ub - data_lb)
    first = True
    batch_size = 32
    split_batch = 16
    while len(queue) > 0:
        print('\rqueue size', len(queue), 'input length', input_length, end='')                
        # TODO timeout
        # TODO batching, we can draw len(args.batch_size, len(queue)) examples from the queue directly.
        # seg_l, seg_u: input segment
        seg_l, seg_u = [], []
        while len(queue) > 0 and len(seg_l) < batch_size:
            seg_l_, seg_u_ = queue.pop()   
            seg_l.append(seg_l_)
            seg_u.append(seg_u_)
        seg_l = torch.cat(seg_l).view(len(seg_l), -1)
        seg_u = torch.cat(seg_u).view(len(seg_u), -1)
        input_length -= (seg_u - seg_l).sum()
        eps = (seg_u - seg_l) / 2      
        x = (seg_u + seg_l) / 2
        l_final, u_final = compute_bounds(model_v1, model_v2, seg_l, seg_u, x=x, eps=eps)
        for k in range(seg_l.size(0)):
            l, u = l_final[k].item(), u_final[k].item()
            # if l == 0 or u == 0:
            #     print(f'\nWARNING: numerical error detected, seg_l={seg_l[k].item()}, seg_u={seg_u[k].item()}, l={l}, u={u}')  
            #     pdb.set_trace()
            if base is None:
                base = (l, u)
            verified = l >= spec[2] if spec[3] == 'lower' else u <= spec[2]
            if verified:
                if res_l is None:
                    res_l, res_u = l, u
                else:
                    res_l = min(res_l, l)
                    res_u = max(res_u, u)
            else:
                has_unstable = False
                for r in model_v1.relus + model_v2.relus:
                    has_unstable = (has_unstable or 
                        torch.logical_and(r.inputs[0].lower[k] < 0, r.inputs[0].upper[k] > 0).any())
                if has_unstable:
                    eps_ = eps[k].item()
                    # first time to split
                    if first:
                        for i in range(split_batch):
                            queue.append((x[k:(k+1)] - (eps_/split_batch) * (i+1), 
                                x[k:(k+1)] - (eps_/split_batch) * i))
                            queue.append((x[k:(k+1)] + (eps_/split_batch) * i, 
                                x[k:(k+1)] + (eps_/split_batch) * (i+1)))
                        first = False
                    else:
                        # NOTE: simply using (seg_l, x) and (x, seg_u) would have 
                        # numerical issues and endless branching
                        queue.append((x[k:(k+1)]-eps_, x[k:(k+1)]))
                        queue.append((x[k:(k+1)], x[k:(k+1)]+eps_))
                    input_length += eps_ * 2
                    if len(queue) > max_queue_size:
                        # Too many. Give up
                        return base
                else:
                    print(f'Adversarial example exists: l={l}, u={u}')
                    pred_l, pred_u = model_ori[-1](seg_l[k:(k+1)]), model_ori[-1](seg_u[k:(k+1)])
                    print(f'Model pred: {seg_l[k].item():.10g}->{pred_l.item()}, {seg_u[k].item():.10g}->{pred_u.item()}')
                    unsafe = False
                    if spec[3] == 'lower':
                        if pred_l < spec[2] or pred_u < spec[2]: unsafe = True
                    else:
                        if pred_l > spec[2] or pred_u > spec[2]: unsafe = True
                    if unsafe:
                        return base[0], base[1], 'unsafe'
                    else:
                        return base[0], base[1]
    print()
    print('bab_single_input output', res_l, res_u)
    return res_l, res_u

""" Attack nn4sys:
Let y be the output of model_v1.
If there exists y[i], y[i+1], y[i+2], such that 
y[i]-y[i+1]*1e10>0, y[i]-y[i+1]*2e10<0, y[i+2]=0, 
the output of the original model will be 0 due to the numerical issue.
"""
@torch.no_grad()
def attack_nn4sys(model_ori, model_v1, spec, onnx_path, device):
    def query(x):
        x = torch.tensor(x, dtype=torch.float64, device=device).view(1, 1)
        return F.relu(model_v1(x)[0])
    out_l = query(spec[0])
    out_u = query(spec[1])
    idx = -1
    for i in range(1, out_l.numel() - 2):
        if out_u[i + 1] == 0 and out_l[i] == 0 and out_u[i] > 0:
            idx = i
            break
    if idx == -1:
        return None
    l = spec[0]
    r = spec[1]
    while l + 1e-20 < r:
        m = (l + r) / 2
        out = query(m)
        if out[idx].item() == 0:
            l = m
        elif (out[idx-1]-1e10*out[idx]) < 0:
            r = m
        elif (out[idx-1]-2e10*out[idx]) > 0:
            l = m
        else:
            # TODO confirm that onnxruntime gives same results
            x = torch.tensor(m, dtype=torch.float64, device=device).view(1, 1)
            out_model_ori = model_ori[-1](x)
            print('attack result', m)
            print('model_ori output', out_model_ori)
            out_onnx = inference_onnx(onnx_path, x[0].cpu().numpy())
            print('onnx output', out_onnx)
            if out_model_ori == 0:
                return m
            else:
                return None
    return None

@torch.no_grad()
def nn4sys_verification(model_ori, vnnlib, onnx_path):
    start_time = time.time()
    all_verified = True
    unsafe = False

    device = arguments.Config["general"]["device"]
    for i in range(3):
        model_ori[i].to(device=device)    
    model_v1 = BoundedModule(
        model_ori[0], 
        torch.empty((2, 1), dtype=torch.float64, device=device), 
        device=device)
    model_v2 = BoundedModule(
        model_ori[1], 
        torch.empty((2, 1), dtype=torch.float64, device=device), 
        device=device)

    # When there are more than 1 properties, we batch all properties to be verified, and check all of them once in a big batch. Many of them can be verified directly without splitting.
    # Then, we only split input those properties that cannot be verified directly. This allows us to solve many instances instantly.
    if len(vnnlib) > 2:
        data_min = torch.tensor([spec[0] for spec in vnnlib], dtype=torch.float64).view(-1, 1).to(device)
        data_max = torch.tensor([spec[1] for spec in vnnlib], dtype=torch.float64).view(-1, 1).to(device)
        l_pre, u_pre = compute_bounds(model_v1, model_v2, data_min, data_max)
    
    for k, spec in enumerate(vnnlib):
        print(f'Verifying spec with {spec[0]} <= x <= {spec[1]} with output {spec[3]} {spec[2]}')
        if spec[-1] == 'lower' and spec[2] > 0:
            attack_x = attack_nn4sys(model_ori, model_v1, spec, onnx_path, device)
            print('attack', attack_x)
            if attack_x is not None:
                unsafe = True
                break

        if len(vnnlib) > 2:
            if spec[-1] == 'upper':
                verified = u_pre[k] <= spec[2]
            else:
                verified = l_pre[k] >= spec[2]
            if verified:
                print(f'Verified from precomputed bounds {l_pre[k].item():.6f} {u_pre[k].item():.6f}')
                continue

        data_min = torch.tensor(spec[0], dtype=torch.float64).view(1, 1)
        data_max = torch.tensor(spec[1], dtype=torch.float64).view(1, 1)
        res = verify(model_ori, model_v1, model_v2, data_ub=data_max, data_lb=data_min, spec=spec)
        if len(res) == 3:
            assert res[2] == 'unsafe'
            unsafe = True
            break
        else:
            l, u = res
            if False:
                # search for check
                from tqdm import tqdm
                step = 0.1
                batch_size = 1024
                x = torch.arange(spec[0], spec[1], step, dtype=torch.float64, device=device)
                print(f'Brute force check, {spec[0]}~{spec[1]}, {x.numel()} points')
                bl, bu = 1e10, -1e10
                for i in tqdm(range((x.numel() + batch_size - 1) // batch_size)):
                    x_ = x[i*batch_size:(i+1)*batch_size].view(-1, 1)
                    out = model_ori[-1](x_)
                    try:
                        assert l <= out.min()
                        assert u >= out.max()
                    except AssertionError:
                        pdb.set_trace()
                    bl = min(bl, out.min().item())
                    bu = max(bu, out.max().item())
                print(f'Verification {l} {u}')
                print(f'Brute force {bl} {bu}')
                # pdb.set_trace()

        print('spec', spec)
        print('verified lower', l)
        print('verified upper', u)
        if spec[3] == 'lower':
            verified = l >= spec[2]
        else:
            verified = u <= spec[2]
        if not verified:
            all_verified = False
            break

    if unsafe:
        res = 'unsafe'
    elif all_verified and time.time() - start_time <= arguments.Config["bab"]["timeout"] - 5:
        res = 'verified'
    else:
        res = 'timeout'

    print('Result', res)
    print('Time', time.time() - start_time)

    return res
