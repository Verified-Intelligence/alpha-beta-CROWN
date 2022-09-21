import logging
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import appdirs
from collections import defaultdict, namedtuple
from collections.abc import Sequence
from functools import reduce
import operator
import math
import warnings
from typing import Tuple
from .patches import Patches, insert_zeros

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('AUTOLIRPA_DEBUG', 0) else logging.INFO)

warnings.simplefilter("once")

# Special identity matrix. Avoid extra computation of identity matrix multiplication in various places.
eyeC = namedtuple('eyeC', 'shape device')
OneHotC = namedtuple('OneHotC', 'shape device index coeffs')

# Benchmarking mode disable some expensive assertions.
Benchmarking = True

reduction_sum = lambda x: x.sum(1, keepdim=True)
reduction_mean = lambda x: x.mean(1, keepdim=True)
reduction_max = lambda x: x.max(1, keepdim=True).values
reduction_min = lambda x: x.min(1, keepdim=True).values

MIN_HALF_FP = 5e-8  # 2**-24, which is the smallest value that float16 can be represented


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func

def stop_criterion_sum(threshold=0):
    return lambda x: (x.sum(1, keepdim=True) > threshold)

def stop_criterion_mean(threshold=0):
    return lambda x: (x.mean(1, keepdim=True) > threshold)

def stop_criterion_min(threshold=0):
    return lambda x: (x.min(1, keepdim=True).values > threshold)

def stop_criterion_max(threshold=0):
    return lambda x: (x.max(1, keepdim=True).values > threshold)

def stop_criterion_batch(threshold=0):
    # may unexpected broadcast, pay attention to the shape of threshold
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    # print('threshold', threshold.shape)
    return lambda x: (x > threshold)

def stop_criterion_batch_any(threshold=0):
    # may unexpected broadcast, pay attention to the shape of threshold
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    # print('threshold', threshold.shape)
    return lambda x: (x > threshold).any(dim=1)

def stop_criterion_batch_topk(threshold=0, k=1314):
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    # print('threshold', threshold.shape)
    return lambda x: (torch.kthvalue(x, k, dim=-1, keepdim=True).values > threshold).any(dim=1)

user_data_dir = appdirs.user_data_dir('auto_LiRPA')
if not os.path.exists(user_data_dir):
    try:
        os.makedirs(user_data_dir)
    except:
        logger.error('Failed to create directory {}'.format(user_data_dir))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)
    def update(self, key, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.lasts[key] = val
        self.sum_meter[key] += val * n
        self.counts_meter[key] += n
    def last(self, key):
        return self.lasts[key]
    def avg(self, key):
        if self.counts_meter[key] == 0:
            return 0.0
        else:
            return self.sum_meter[key] / self.counts_meter[key]
    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()

class MultiTimer(object):
    """Count the time for each part of training."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.timer_starts = defaultdict(float)
        self.timer_total = defaultdict(float)
    def start(self, key):
        if self.timer_starts[key] != 0:
            raise RuntimeError("start() is called more than once")
        self.timer_starts[key] = time.time()
    def stop(self, key):
        if key not in self.timer_starts:
            raise RuntimeError("Key does not exist; please call start() before stop()")
        self.timer_total[key] += time.time() - self.timer_starts[key]
        self.timer_starts[key] = 0
    def total(self, key):
        return self.timer_total[key]
    def __repr__(self):
        s = ""
        for k in self.timer_total:
            s += "{}_time={:.3f} ".format(k, self.timer_total[k])
        return s.strip()

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)        

class Unflatten(nn.Module):
    def __init__(self, wh):
        super().__init__()
        self.wh = wh # width and height of the feature maps
    def forward(self, x):
        return x.view(x.size(0), -1, self.wh, self.wh)              

def scale_gradients(optimizer, gradient_accumulation_steps, grad_clip=None):    
    parameters = []
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            parameters.append(param)
            if param.grad is not None:
                param.grad.data /= gradient_accumulation_steps
    if grad_clip is not None:
        return torch.nn.utils.clip_grad_norm_(parameters, grad_clip)                

def recursive_map (seq, func):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)

# unpack tuple, dict, list into one single list
# TODO: not sure if the order matches graph.inputs()
def unpack_inputs(inputs, device=None):
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        res = []
        for item in inputs: 
            res += unpack_inputs(item, device=device)
        return res
    else:
        if device is not None:
            inputs = inputs.to(device)
        return [inputs]

def isnan(x):
    if isinstance(x, Patches):
        return False
    return torch.isnan(x).any()
    
def prod(x):
    return reduce(operator.mul, x, 1)

def batched_index_select(input, dim, index):
    # Assuming the input has a batch dimension.
    # index has dimensin [spec, batch].
    if input.ndim == 4:
        # Alphas for fully connected layers, shape [2, spec, batch, neurons]
        index = index.unsqueeze(-1).unsqueeze(0).expand(input.size(0), -1, -1, input.size(3))
    elif input.ndim == 6:
        # Alphas for fully connected layers, shape [2, spec, batch, c, h, w].
        index = index.view(1, index.size(0), index.size(1), *([1] * (input.ndim - 3))).expand(input.size(0), -1, -1, *input.shape[3:])
    elif input.ndim == 3:
        # Weights.
        input = input.expand(index.size(0), -1, -1)
        index = index.unsqueeze(-1).expand(-1, -1, input.size(2))
    elif input.ndim == 2:
        # Bias.
        input = input.expand(index.size(0), -1)
    else:
        raise ValueError
    return torch.gather(input, dim, index)


def check_padding(x, padding):
    if isinstance(padding, int):
        return x, (padding, padding)
    if len(padding) == 2:
        return x, padding
    if (padding[0] == padding[1]) and (padding[2] == padding[3]):
        return x, (padding[0], padding[2])
    return F.pad(x, padding), (0, 0)


def get_spec_matrix(X, y, num_classes):
    with torch.no_grad():
        c = (torch.eye(num_classes).type_as(X)[y].unsqueeze(1) 
            - torch.eye(num_classes).type_as(X).unsqueeze(0))
        I = (~(y.unsqueeze(1) == torch.arange(num_classes).type_as(y).unsqueeze(0)))
        c = (c[I].view(X.size(0), num_classes - 1, num_classes))  
    return c

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, a list with tensors in shape (N, D).

    Code borrowed from:
        https://github.com/pytorch/pytorch/issues/35674
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='trunc')

    return list(reversed(coord))

def get_A_shape(A):
    if A is None:
        return 'None'
    if isinstance(A, Patches):
        if A.patches is not None:
            return A.patches.shape
        else:
            return A.shape
    if isinstance(A, torch.Tensor):
        return A.shape
    return 'Unknown'
