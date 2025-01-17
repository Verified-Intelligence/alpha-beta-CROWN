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
from abc import ABC, abstractmethod
import torch

import arguments

class TensorStorage(ABC):
    """
    Fast managed dynamic sized tensor storage.
    """
    def __init__(self, full_shape, initial_size=1024, switching_size=65536,
                 dtype=None, device='cpu', concat_dim=0):
        """
        full_shape is the tensor shape you want to store using this object, including the "batch" dimension.
        dtype is tensor type (default float32).
        initial_size is the initial size of the storage. It will go up exponentially until reaching a batch size of "switching_size".
        switching_size is the point where exponential growth changes to linear growth.
        device is storage device, and for CPU memory it will be pinned.
        concat_dim is the axis of batch dimension (default is 0).
        """
        if isinstance(full_shape, torch.Tensor):
            data = full_shape
            full_shape = data.shape
        else:
            data = None
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.shape = list(full_shape)  # Full shape, with batch size that will become dynamic.
        self.dtype = dtype
        self.device = device
        self.concat_dim = concat_dim
        self.num_used = 0
        self.switching_size = switching_size
        self._storage = self._allocate(initial_size)

        if data is not None:
            self.append(data)

    def _allocate(self, new_size):
        allocate_shape = self.shape.copy()
        allocate_shape[self.concat_dim] = new_size
        if self.device == 'cpu' and torch.cuda.is_available():
            # Pin CPU memory if cuda is available.
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device, pin_memory=True)
        else:
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device)

    def _get_new_size(self, request_size):
        """Compute new size of storage given the current request."""
        if self._storage.size(self.concat_dim) < self.switching_size:
            # Tensor is small. Exponential growth.
            return max(self._storage.size(self.concat_dim) * 2, self.num_used + request_size)
        else:
            # Tensor is big. Linear growth.
            return self._storage.size(self.concat_dim) + request_size * 32

    @abstractmethod
    def append(self, appended_tensor):
        pass

    @abstractmethod
    def pop(self, size):
        pass

    @abstractmethod
    def tensor(self):
        pass

    def __getattr__(self, attr):
        """Proxy all tensor attributes."""
        return getattr(self.tensor(), attr)

    def __getitem__(self, idx):
        return self.tensor()[idx]

    def __len__(self):
        return self.num_used

    def __sub__(self, o):
        return self.tensor() - o.tensor()

class StackTensorStorage(TensorStorage):
    @torch.no_grad()
    def append(self, appended_tensor):
        """
        Append a new tensor to the storage object. This invalidates all previously returned tensors.

        If you need to reuse the previously returned tensors, you should copy them before calling this function.
        """
        if self.num_used + appended_tensor.size(self.concat_dim) > self._storage.size(self.concat_dim):
            # Reallocate a new tensor, copying the existing contents over.
            new_size = self._get_new_size(appended_tensor.size(self.concat_dim))
            new_tensor = self._allocate(new_size)
            new_tensor.narrow(dim=self.concat_dim, start=0, length=self.num_used).copy_(
                self._storage.narrow(dim=self.concat_dim, start=0, length=self.num_used))
            # And then remove the old storage object.
            del self._storage
            self._storage = new_tensor
        self._storage.narrow(self.concat_dim, self.num_used, appended_tensor.size(self.concat_dim)).copy_(appended_tensor)
        self.num_used += appended_tensor.size(self.concat_dim)
        return self

    @torch.no_grad()
    def pop(self, size):
        """Remove tensors with 'size' at the end of the storage."""
        size = max(min(size, self.num_used), 0)
        ret = self._storage.narrow(self.concat_dim, self.num_used - size, size)
        self.num_used -= size
        return ret

    def tensor(self):
        return self._storage.narrow(self.concat_dim, 0, self.num_used)

class QueueTensorStorage(TensorStorage):
    def __init__(self, full_shape, initial_size=1024, switching_size=65536,
                 dtype=None, device='cpu', concat_dim=0):
        self._usage_start = 0
        super().__init__(full_shape, initial_size, switching_size, dtype, device, concat_dim)

    def _move_to_new_tensor(self, new_tensor):
        current_size = self._storage.size(self.concat_dim)
        entries_to_end_of_buffer_or_tail = min(self.num_used, current_size - self._usage_start)
        new_tensor.narrow(dim=self.concat_dim, start=0, length=entries_to_end_of_buffer_or_tail).copy_(
            self._storage.narrow(dim=self.concat_dim, start=self._usage_start, length=entries_to_end_of_buffer_or_tail))
        if entries_to_end_of_buffer_or_tail < self.num_used:
            entries_at_start_of_buffer = self.num_used - entries_to_end_of_buffer_or_tail
            assert entries_at_start_of_buffer > 0
            new_tensor.narrow(dim=self.concat_dim, start=entries_to_end_of_buffer_or_tail, length=entries_at_start_of_buffer).copy_(
                self._storage.narrow(dim=self.concat_dim, start=0, length=entries_at_start_of_buffer))
        self._usage_start = 0
        # And then remove the old storage object.
        del self._storage
        self._storage = new_tensor

    @torch.no_grad()
    def append(self, appended_tensor):
        """
        Append a new tensor to the storage object. This invalidates all previously returned tensors.

        If you need to reuse the previously returned tensors, you should copy them before calling this function.
        """
        current_size = self._storage.size(self.concat_dim)
        appended_size = appended_tensor.size(self.concat_dim)
        if self.num_used + appended_size > current_size:
            # Reallocate a new tensor, copying the existing contents over.
            new_size = self._get_new_size(appended_size)
            new_tensor = self._allocate(new_size)
            self._move_to_new_tensor(new_tensor)
            current_size = self._storage.size(self.concat_dim)

        first_free_index = (self._usage_start + self.num_used) % current_size
        entries_at_buffer_tail = current_size - first_free_index
        # We can be sure that this never overwrites any existing entries, because if it would, we'd
        # have extended the storage above.
        entries_copied_to_tail = min(entries_at_buffer_tail, appended_size)
        self._storage.narrow(dim=self.concat_dim, start=first_free_index, length=entries_copied_to_tail).copy_(
            appended_tensor.narrow(dim=self.concat_dim, start=0, length=entries_copied_to_tail)
        )
        if entries_copied_to_tail < appended_size:
            entries_copied_to_start = appended_size - entries_copied_to_tail
            self._storage.narrow(dim=self.concat_dim, start=0, length=entries_copied_to_start).copy_(
                appended_tensor.narrow(dim=self.concat_dim, start=entries_copied_to_tail, length=entries_copied_to_start)
            )
        self.num_used += appended_size
        return self

    @torch.no_grad()
    def pop(self, size):
        """Remove tensors with 'size' from the start of the storage."""
        size = max(min(size, self.num_used), 0)
        if size == 0:
            return self._storage.narrow(self.concat_dim, 0, 0)
        current_size = self._storage.size(self.concat_dim)
        entries_to_buffer_end = min(size, current_size - self._usage_start)
        assert entries_to_buffer_end > 0
        if entries_to_buffer_end == size:
            ret = self._storage.narrow(self.concat_dim, self._usage_start, size)
        else:
            ret1 = self._storage.narrow(self.concat_dim, self._usage_start, entries_to_buffer_end)
            ret2 = self._storage.narrow(self.concat_dim, 0, size - entries_to_buffer_end)
            ret = torch.cat([ret1, ret2], dim=self.concat_dim)
        self.num_used -= size
        self._usage_start = (self._usage_start + size) % current_size
        return ret

    def tensor(self):
        current_size = self._storage.size(self.concat_dim)
        if self._usage_start + self.num_used > current_size:
            # We'll have to move the data anyway to return a single consequtive tensor
            # Instead of just returning torch.cat([elements_at_buffer_end, elements_at_buffer_start])
            # we make the buffer itself consequtive. This way, the next call to .tensor() will be
            # faster.
            new_storage = self._allocate(current_size)
            self._move_to_new_tensor(new_storage)

        return self._storage.narrow(self.concat_dim, self._usage_start, self.num_used)

def get_tensor_storage(full_shape, initial_size=1024, switching_size=65536,
        dtype=None, device='cpu', concat_dim=0):
    tree_traversal = arguments.Config['bab']['tree_traversal']
    if tree_traversal == 'depth_first':
        return StackTensorStorage(full_shape, initial_size, switching_size, dtype, device, concat_dim)
    elif tree_traversal == 'breadth_first':
        return QueueTensorStorage(full_shape, initial_size, switching_size, dtype, device, concat_dim)
    else:
        raise ValueError(f"Unknown tree traversal mode: {tree_traversal}")