#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from abc import ABC, abstractmethod
import math
import torch
import ctypes
import platform
import warnings
import traceback
import logging

import arguments

class HugeTensorAllocator:
    """
    Cross-platform hugepage tensor allocation utility.
    Provides memory-aligned tensor allocation with hugepage support where available.
    """
    # Constants
    MADV_HUGEPAGE = 14  # Linux-specific huge page hint
    ALIGN_SIZE = 2 * 1024 * 1024  # 2MB alignment (common hugepage size)

    # Class variables to cache the results of is_supported() and get_libc()
    _hugepages_supported = None
    _libc = None

    @classmethod
    def _check_support(cls):
        """Internal method to check if hugepages are supported on the current system."""
        # Check if system is Linux
        system = platform.system()
        if system != 'Linux':
            return False
        # Check if madvise is available and /proc/meminfo contains HugePages info
        try:
            # Try to get libc
            try:
                libc = ctypes.CDLL("libc.so.6")
            except OSError:
                libc = ctypes.CDLL("libc.so")
            # Check if madvise exists
            if not hasattr(libc, 'madvise'):
                return False           
            # That step checks if the Linux system has HugePages support enabled.
            # In Linux, `/proc/meminfo` is a virtual file that contains information
            # about the system's memory usage and settings. 
            # When HugePages are configured in the Linux kernel, this file will 
            # contain lines with the prefix `HugePages_`.
            # The code is opening this file, reading its contents, and then 
            # searching for the string "HugePages_" in the content. If found,
            # it indicates that:
            # 1. The Linux kernel has HugePages support compiled in
            # 2. The system has HugePages configured and available for use
            # If "HugePages_" is not found in `/proc/meminfo`, attempting to use HugePages would fail,
            # so the allocator should fall back to regular memory allocation.
            with open('/proc/meminfo', 'r') as f:
                content = f.read()
                logging.info('Hugepage enabled for branch-and-bound domains.')
                return 'HugePages_' in content
        except:
            traceback.print_exc()
            return False

    @classmethod
    def _load_libc(cls):
        """Internal method to get the appropriate C library based on the platform."""
        try:
            return ctypes.CDLL("libc.so.6")
        except OSError:
            return ctypes.CDLL("libc.so")

    @classmethod
    def is_supported(cls):
        """Check if hugepages are supported (uses cached result)."""
        try:
            if not arguments.Config['bab']['hugetensor_allocator']:
                return False
            return cls._hugepages_supported
        except Exception as e:
            traceback.print_exc()
            # Fall back to False if we encounter any errors checking support
            return False

    @classmethod
    def get_libc(cls):
        """Get the libc library (uses cached result)."""
        return cls._libc

    @classmethod
    def allocate(cls, shape, dtype, pin_memory=False):
        """
        Allocate memory-aligned tensor with hugepage support if available.

        Args:
            shape: The tensor shape
            dtype: The data type
            pin_memory: Whether to lock the memory in RAM

        Returns:
            A torch.Tensor using the allocated memory
        """
        # If hugepages are not supported or disabled, fall back to regular PyTorch allocation
        if not cls.is_supported():
            return torch.empty(shape, dtype=dtype, pin_memory=pin_memory)

        total_size = math.prod(shape)
        size_bytes = total_size * dtype.itemsize
        # Handle empty tensor case
        if size_bytes == 0:
            return torch.empty(shape, dtype=dtype)
        # Calculate aligned size
        aligned_size = ((size_bytes + cls.ALIGN_SIZE - 1) // 
                       cls.ALIGN_SIZE) * cls.ALIGN_SIZE
        # Allocate aligned memory
        try:
            libc = cls._libc
            # Setup calling arguments
            libc.aligned_alloc.restype = ctypes.c_void_p
            libc.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            # Allocate aligned memory
            buffer_ptr = libc.aligned_alloc(cls.ALIGN_SIZE, aligned_size)
            if not buffer_ptr:
                raise MemoryError("Failed to allocate aligned memory")
            # Try to use hugepages
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise(buffer_ptr, aligned_size, ctypes.c_int(cls.MADV_HUGEPAGE))
            # Pin memory if requested
            if pin_memory and hasattr(libc, 'mlock'):
                libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                libc.mlock(buffer_ptr, aligned_size)
            # Create a Python buffer with this memory address
            wrapped_buffer_ptr = (ctypes.c_byte * size_bytes).from_address(buffer_ptr)
            # Create a tensor with this buffer
            tensor = torch.frombuffer(wrapped_buffer_ptr, dtype=dtype).view(shape)
            # Store the pointer for later freeing
            tensor._buffer_ptr = buffer_ptr

            return tensor

        except Exception as e:
            # Print the full traceback
            traceback.print_exc()
            # Fall back to regular PyTorch allocation
            warnings.warn(f"HugeTensor allocation failed ({str(e)}). Falling back to regular allocation.")
            return torch.empty(shape, dtype=dtype, pin_memory=pin_memory)

    @classmethod
    def free(cls, tensor_or_ptr):
        """
        Free memory allocated by allocate.

        Args:
            tensor_or_ptr: Either a tensor created by allocate or a memory address
        """
        # Get the pointer from tensor if needed
        if isinstance(tensor_or_ptr, torch.Tensor):
            if not hasattr(tensor_or_ptr, '_buffer_ptr'):
                return  # Not a hugetensor, nothing to free
            ptr = tensor_or_ptr._buffer_ptr
        else:
            ptr = tensor_or_ptr
        if ptr is None:
            return
        try:
            libc = cls._libc
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free(ptr)

        except Exception as e:
            warnings.warn(f"Error freeing hugetensor memory: {str(e)}")

# Initialize the class variables when the module is imported
HugeTensorAllocator._hugepages_supported = HugeTensorAllocator._check_support()
if HugeTensorAllocator._hugepages_supported:
    HugeTensorAllocator._libc = HugeTensorAllocator._load_libc()

class TensorStorage(ABC):
    """
    Fast managed dynamic sized tensor storage.
    """
    def __init__(self, full_shape, initial_size=1024, switching_size=65536,
                 dtype=None, device='cpu', concat_dim=0, use_hugepage='auto'):
        """
        full_shape is the tensor shape you want to store using this object, including the "batch" dimension.
        dtype is tensor type (default float32).
        initial_size is the initial size of the storage. It will go up exponentially until reaching a batch size of "switching_size".
        switching_size is the point where exponential growth changes to linear growth.
        device is storage device, and for CPU memory it will be pinned.
        concat_dim is the axis of batch dimension (default is 0).
        use_hugepage sets allocated storage to 2MB aligned memory address.
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
        if use_hugepage == 'auto':
            self.use_hugepage = HugeTensorAllocator.is_supported()
        else:
            self.use_hugepage = use_hugepage
        self._storage = self._allocate(initial_size)
        self._buffer_ptr = 0

        if data is not None:
            self.append(data)

    def _allocate(self, new_size):
        allocate_shape = self.shape.copy()
        allocate_shape[self.concat_dim] = new_size
        total_size = math.prod(allocate_shape)
        if self.device == 'cpu' and torch.cuda.is_available():
            if not self.use_hugepage:
                # Pin CPU memory if cuda is available.
                return torch.empty(allocate_shape, dtype=self.dtype, device=self.device, pin_memory=False)
            if total_size != 0:
                # Allocate storage tensor with hugepages.
                tensor = HugeTensorAllocator.allocate(allocate_shape, self.dtype, pin_memory=False)
                self._buffer_ptr = tensor.data_ptr()
            else:
                # Create a zero-sized tensor.
                tensor = torch.empty(allocate_shape, dtype=self.dtype, device=self.device)
            return tensor
        else:
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device)

    def _deallocate(self):
        # For zero-sized tensor, aligned memory was not allocated even if self.use_hugepage is True.
        if self.use_hugepage and self._storage.numel() != 0:
            HugeTensorAllocator.free(self._storage)
        del self._storage
        self._storage = None

    def _get_new_size(self, request_size):
        """Compute new size of storage given the current request."""
        if self._storage.size(self.concat_dim) < self.switching_size:
            # Tensor is small. Exponential growth.
            return max(self._storage.size(self.concat_dim) * 2, self.num_used + request_size)
        else:
            # Tensor is big. Linear growth.
            return self._storage.size(self.concat_dim) + request_size * 32

    def calculate_memory(self):
        """
        Report memory allocated and used by the tensor storage, in megabytes (MB).
        """
        elem_size = self.dtype.itemsize

        allocated_elems = self._storage.numel()
        used_shape = self.shape.copy()
        used_shape[self.concat_dim] = self.num_used
        used_elems = torch.Size(used_shape).numel()

        allocated_bytes = allocated_elems * elem_size
        used_bytes = used_elems * elem_size

        return allocated_bytes / (1024**2), used_bytes / (1024**2)

    @abstractmethod
    def append(self, appended_tensor):
        pass

    @abstractmethod
    def pop(self, size):
        pass

    @abstractmethod
    def tensor(self):
        pass

    @abstractmethod
    def reorder(self):
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
            self._deallocate()
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

    @torch.no_grad()
    def reorder(self, num_domains, indices, reorder_dim=None):
        """
        Reorder the 'num_domains' entries of the valid data based on indices.
        
        Args:
            num_domains (int): Number of domains to reorder.
            indices (torch.Tensor): Indices to use for reordering.
            reorder_dim (int, optional): Dimension along which to reorder. 
                                        Defaults to self.concat_dim.
        """
        # Use concat_dim if no specific reorder dimension is provided
        if reorder_dim is None:
            reorder_dim = self.concat_dim

        # Slice the storage to select the relevant portion
        storage_slice = [slice(None)] * self._storage.ndim
        storage_slice[reorder_dim] = slice(0, num_domains)

        # Reorder along the specified dimension
        self._storage[tuple(storage_slice)] = self._storage[tuple(storage_slice)].index_select(
            index=indices, 
            dim=reorder_dim
        )

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

    @torch.no_grad()
    def reorder(self, num_domains, indices, reorder_dim=None):
        """
        Reorder the 'num_domains' entries of the valid data based on indices.
        
        Handles reordering for circular buffer storage, ensuring data is contiguous 
        before reordering.
        
        Args:
            num_domains (int): Number of domains to reorder.
            indices (torch.Tensor): Indices to use for reordering.
            reorder_dim (int, optional): Dimension along which to reorder. 
                                        Defaults to self.concat_dim.
        
        Raises:
            ValueError: If num_domains is larger than the number of stored entries.
        """
        # Check that we have at least num_domains entries
        if self.num_used < num_domains:
            raise ValueError("num_domains is larger than the number of stored entries.")
        
        # Ensure that the valid data is contiguous (i.e. _usage_start==0)
        if self._usage_start != 0:
            current_size = self._storage.size(self.concat_dim)
            new_storage = self._allocate(current_size)
            self._move_to_new_tensor(new_storage)
        
        # Use concat_dim if no specific reorder dimension is provided
        if reorder_dim is None:
            reorder_dim = self.concat_dim
        
        # Prepare slices for reordering
        storage_slice = [slice(None)] * self._storage.ndim
        storage_slice[reorder_dim] = slice(0, num_domains)
        
        # Reorder along the specified dimension
        self._storage[tuple(storage_slice)] = self._storage[tuple(storage_slice)].index_select(
            index=indices, 
            dim=reorder_dim
        )

    def tensor(self):
        """
        Return a contiguous tensor of the used storage.
        
        If the used storage spans the end of the buffer, reorganize to make it contiguous.
        
        Returns:
            torch.Tensor: A contiguous tensor of the used storage.
        """
        current_size = self._storage.size(self.concat_dim)
        if self._usage_start + self.num_used > current_size:
            # Move data to make it contiguous for faster subsequent .tensor() calls
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
