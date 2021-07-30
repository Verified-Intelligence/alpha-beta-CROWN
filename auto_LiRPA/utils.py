import logging
import pickle
import time
import torch
import torch.nn as nn
import os
import sys
import appdirs
from collections import defaultdict, Sequence, namedtuple
from functools import reduce
import operator
import math
import torch.nn.functional as F

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Special identity matrix. Avoid extra computation of identity matrix multiplication in various places.
eyeC = namedtuple('eyeC', 'shape device')
OneHotC = namedtuple('OneHotC', 'shape device index coeffs')

# Benchmarking mode disable some expensive assertions.
Benchmarking = True

reduction_sum = lambda x: x.sum(1, keepdim=True)
reduction_mean = lambda x: x.mean(1, keepdim=True)
reduction_max = lambda x: x.max(1, keepdim=True).values
reduction_min = lambda x: x.min(1, keepdim=True).values

def stop_criterion_sum(threshold=0):
    return lambda x: (x.sum(1, keepdim=True) > threshold)

def stop_criterion_mean(threshold=0):
    return lambda x: (x.mean(1, keepdim=True) > threshold)

def stop_criterion_min(threshold=0):
    return lambda x: (x.min(1, keepdim=True).values > threshold)

def stop_criterion_max(threshold=0):
    return lambda x: (x.max(1, keepdim=True).values > threshold)

# Create a namedtuple with defaults
def namedtuple_with_defaults(name, attr, defaults):
    assert sys.version_info.major == 3
    if sys.version_info.major >= 7:
        return namedtuple(name, attr, defaults=defaults)
    else:
        # The defaults argument is not available in Python < 3.7
        t = namedtuple(name, attr)
        t.__new__.__defaults__ = defaults
        return t

# A special class which denotes a convoluntional operator as a group of patches
# the shape of Patches.patches is [batch_size, num_of_patches, out_channel, in_channel, M, M]
# M is the size of a single patch
# Assume that we have a conv2D layer with w.weight(out_channel, in_channel, M, M), stride and padding applied on an image (N * N)
# num_of_patches = ((N + padding * 2 - M)//stride + 1) ** 2
# Here we only consider kernels with the same H and W
Patches = namedtuple_with_defaults('Patches', ('patches', 'stride', 'padding', 'shape', 'identity'), defaults=(None, 1, 0, None, 0))
BoundList = namedtuple_with_defaults('BoundList', ('bound_list'), defaults=([],))
# Linear bounds with coefficients. Used for forward bound propagation.
LinearBound = namedtuple_with_defaults('LinearBound', ('lw', 'lb', 'uw', 'ub', 'lower', 'upper', 'from_input', 'nominal', 'lower_offset', 'upper_offset'), defaults=(None,) * 10)

# for debugging
if False:
    file_handler = logging.FileHandler('debug.log')
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

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
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)                

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
    if input.ndim == 4:
        # Alphas.
        index = index.unsqueeze(-1).unsqueeze(0).expand(input.size(0), -1, -1, input.size(3))
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


def patchesToMatrix(pieces, input_shape, stride, padding):
    if type(padding) == int:
        padding = (padding, padding, padding, padding)
        
    batch_size, total_patches, output_channel, input_channel, kernel_x, kernel_y = pieces.shape[0], pieces.shape[1], pieces.shape[2], pieces.shape[3], pieces.shape[4], pieces.shape[5]
    input_x, input_y = input_shape[-2:]
    output_x = output_y = int(math.sqrt(total_patches))
    A_matrix = torch.zeros(batch_size, output_channel, total_patches, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device)
    # Save its orignal stride.
    orig_stride = A_matrix.stride()
    # This is the main trick - we create a *view* of the original matrix, and it contains all sliding windows for the convolution.
    # Since we only created a view (in fact, only metadata of the matrix changed), it should be very efficient.
    matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, total_patches, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[3], input_y + padding[0] + padding[1], 1])
    # TODO: support stride. Hint: you will need to change the stride in the above line, to something like --------------------------------------> [orig_stride[0], orig_stride[1], orig_stride[2], input_y * STRIDE_Y, STRIED_X, orig_stride[3], input_y, 1]).
    # TODO: support padding. Hint: you can use F.pad2d to create a padded matrix and use it instead, and then remove the unsed rows and columns afterwards.

    # Now we need to fill the conv kernel parameters into the last three dimensions of matrix_strided.
    # The first index: we fill each patch location with a conv kernel.
    first_indices = torch.arange(total_patches, device=pieces.device)
    second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
    third_indices = torch.fmod(first_indices, output_y)
    pieces = pieces.transpose(1,2)

    matrix_strided[:,:,first_indices,second_indices,third_indices,:,:,:] = pieces.view(batch_size, output_channel, total_patches, input_channel,kernel_y,kernel_x)
    A_matrix = A_matrix.view(batch_size, output_channel*total_patches, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
    A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]

    return A_matrix


def check_padding(x, padding):
    if isinstance(padding, int):
        return x, (padding, padding)
    if len(padding) == 2:
        return x, padding
    if (padding[0] == padding[1]) and (padding[2] == padding[3]):
        return x, (padding[0], padding[2])
    return F.pad(x, padding), (0, 0)