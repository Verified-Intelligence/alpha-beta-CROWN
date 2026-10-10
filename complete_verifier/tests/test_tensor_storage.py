#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Unit tests for tensor_storage.py"""
import os
import sys
import unittest

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_storage import (
    TensorStorage, StackTensorStorage, QueueTensorStorage,
    HugeTensorAllocator
)


class TestTensorStorageDtype(unittest.TestCase):
    """Tests for the dtype contract shared by all tensor storages."""

    def test_tensor_input_owns_storage_dtype(self):
        """Tensor initialization preserves its dtype through storage use."""
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for storage_cls in (StackTensorStorage, QueueTensorStorage):
                for input_dtype in (torch.float16, torch.int64):
                    with self.subTest(
                            storage=storage_cls.__name__, dtype=input_dtype):
                        data = torch.arange(
                            6, dtype=input_dtype).reshape(2, 3)
                        storage = storage_cls(
                            data, initial_size=2, switching_size=4)
                        self.assertEqual(storage.dtype, input_dtype)
                        self.assertEqual(storage.tensor().dtype, input_dtype)

                        storage.append(torch.ones(
                            3, 3, dtype=input_dtype))
                        self.assertEqual(storage.tensor().dtype, input_dtype)
                        self.assertEqual(storage.pop(1).dtype, input_dtype)
        finally:
            torch.set_default_dtype(original_default)

    def test_explicit_dtype_overrides_tensor_dtype(self):
        """An explicit storage dtype takes precedence over input dtype."""
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        for storage_cls in (StackTensorStorage, QueueTensorStorage):
            with self.subTest(storage=storage_cls.__name__):
                storage = storage_cls(data, dtype=torch.float32)
                self.assertEqual(storage.dtype, torch.float32)
                self.assertEqual(storage.tensor().dtype, torch.float32)
                self.assertTrue(torch.equal(storage.tensor(), data.float()))

    def test_shape_input_uses_default_dtype(self):
        """Shape-only initialization retains the default dtype contract."""
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for storage_cls in (StackTensorStorage, QueueTensorStorage):
                with self.subTest(storage=storage_cls.__name__):
                    storage = storage_cls((0, 3), initial_size=2)
                    self.assertEqual(storage.dtype, torch.float64)
                    self.assertEqual(storage.tensor().dtype, torch.float64)
        finally:
            torch.set_default_dtype(original_default)


class TestStackTensorStorage(unittest.TestCase):
    """Tests for StackTensorStorage class."""

    def test_init_empty(self):
        """Test initialization with empty storage."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        self.assertEqual(len(storage), 0)
        self.assertEqual(storage.num_used, 0)

    def test_init_with_data(self):
        """Test initialization with initial data tensor."""
        data = torch.randn(5, 10)
        storage = StackTensorStorage(data)
        self.assertEqual(len(storage), 5)
        self.assertTrue(torch.allclose(storage.tensor(), data))

    def test_append_single(self):
        """Test appending a single tensor."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(4, 10)
        storage.append(tensor)
        self.assertEqual(len(storage), 4)
        self.assertTrue(torch.allclose(storage.tensor(), tensor))

    def test_append_multiple(self):
        """Test appending multiple tensors."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        t1 = torch.randn(3, 10)
        t2 = torch.randn(5, 10)
        storage.append(t1)
        storage.append(t2)
        self.assertEqual(len(storage), 8)
        expected = torch.cat([t1, t2], dim=0)
        self.assertTrue(torch.allclose(storage.tensor(), expected))

    def test_append_triggers_reallocation(self):
        """Test that appending beyond initial size triggers reallocation."""
        storage = StackTensorStorage((0, 10), initial_size=4)
        tensor = torch.randn(10, 10)
        storage.append(tensor)
        self.assertEqual(len(storage), 10)
        self.assertTrue(torch.allclose(storage.tensor(), tensor))

    def test_pop_single(self):
        """Test popping elements from storage."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(8, 10)
        storage.append(tensor)
        popped = storage.pop(3)
        self.assertEqual(popped.shape, (3, 10))
        self.assertEqual(len(storage), 5)
        self.assertTrue(torch.allclose(popped, tensor[5:8]))

    def test_pop_more_than_available(self):
        """Test popping more elements than available."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(5, 10)
        storage.append(tensor)
        popped = storage.pop(10)  # Try to pop more than available
        self.assertEqual(popped.shape, (5, 10))
        self.assertEqual(len(storage), 0)

    def test_pop_zero(self):
        """Test popping zero elements."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(5, 10)
        storage.append(tensor)
        popped = storage.pop(0)
        self.assertEqual(popped.shape, (0, 10))
        self.assertEqual(len(storage), 5)

    def test_pop_negative(self):
        """Test popping negative number of elements (should be treated as 0)."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(5, 10)
        storage.append(tensor)
        popped = storage.pop(-5)
        self.assertEqual(popped.shape, (0, 10))
        self.assertEqual(len(storage), 5)

    def test_tensor_view(self):
        """Test that tensor() returns correct view."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        t1 = torch.randn(3, 10)
        t2 = torch.randn(2, 10)
        storage.append(t1)
        storage.append(t2)
        result = storage.tensor()
        self.assertEqual(result.shape, (5, 10))

    def test_reorder(self):
        """Test reordering elements in storage."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.arange(40).reshape(4, 10).float()
        storage.append(tensor)
        indices = torch.tensor([3, 1, 0, 2])
        storage.reorder(4, indices)
        expected = tensor[indices]
        self.assertTrue(torch.allclose(storage.tensor(), expected))

    def test_reorder_partial(self):
        """Test reordering only a portion of elements."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.arange(50).reshape(5, 10).float()
        storage.append(tensor)
        indices = torch.tensor([2, 0, 1])
        storage.reorder(3, indices)
        # First 3 elements should be reordered
        self.assertTrue(torch.allclose(storage.tensor()[:3], tensor[:3][indices]))
        # Last 2 elements should remain unchanged
        self.assertTrue(torch.allclose(storage.tensor()[3:], tensor[3:]))

    def test_getitem(self):
        """Test indexing into storage."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(5, 10)
        storage.append(tensor)
        self.assertTrue(torch.allclose(storage[0], tensor[0]))
        self.assertTrue(torch.allclose(storage[2:4], tensor[2:4]))

    def test_sub(self):
        """Test subtraction between storages."""
        s1 = StackTensorStorage((0, 10), initial_size=16)
        s2 = StackTensorStorage((0, 10), initial_size=16)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10)
        s1.append(t1)
        s2.append(t2)
        result = s1 - s2
        expected = t1 - t2
        self.assertTrue(torch.allclose(result, expected))

    def test_calculate_memory(self):
        """Test memory calculation."""
        storage = StackTensorStorage((0, 10), initial_size=100, dtype=torch.float32)
        tensor = torch.randn(50, 10)
        storage.append(tensor)
        allocated_mb, used_mb = storage.calculate_memory()
        # 100 * 10 * 4 bytes = 4000 bytes allocated
        self.assertGreater(allocated_mb, 0)
        # 50 * 10 * 4 bytes = 2000 bytes used
        self.assertGreater(used_mb, 0)
        self.assertGreater(allocated_mb, used_mb)

    def test_different_concat_dim(self):
        """Test storage with different concatenation dimension."""
        storage = StackTensorStorage((10, 0, 5), initial_size=16, concat_dim=1)
        tensor = torch.randn(10, 4, 5)
        storage.append(tensor)
        self.assertEqual(len(storage), 4)
        self.assertEqual(storage.tensor().shape, (10, 4, 5))

    def test_different_dtype(self):
        """Test storage with different data type."""
        storage = StackTensorStorage((0, 10), initial_size=16, dtype=torch.float64)
        tensor = torch.randn(5, 10, dtype=torch.float64)
        storage.append(tensor)
        self.assertEqual(storage.tensor().dtype, torch.float64)

    def test_exponential_growth(self):
        """Test exponential growth for small tensors."""
        storage = StackTensorStorage((0, 10), initial_size=4, switching_size=1000)
        # Append enough to trigger multiple reallocations
        for _ in range(10):
            storage.append(torch.randn(3, 10))
        self.assertEqual(len(storage), 30)

    def test_linear_growth(self):
        """Test linear growth for large tensors."""
        storage = StackTensorStorage((0, 10), initial_size=100, switching_size=50)
        # Append to trigger linear growth mode
        storage.append(torch.randn(60, 10))
        storage.append(torch.randn(20, 10))
        self.assertEqual(len(storage), 80)


class TestQueueTensorStorage(unittest.TestCase):
    """Tests for QueueTensorStorage class."""

    def test_init_empty(self):
        """Test initialization with empty storage."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        self.assertEqual(len(storage), 0)
        self.assertEqual(storage.num_used, 0)

    def test_append_single(self):
        """Test appending a single tensor."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(4, 10)
        storage.append(tensor)
        self.assertEqual(len(storage), 4)
        self.assertTrue(torch.allclose(storage.tensor(), tensor))

    def test_append_multiple(self):
        """Test appending multiple tensors."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        t1 = torch.randn(3, 10)
        t2 = torch.randn(5, 10)
        storage.append(t1)
        storage.append(t2)
        self.assertEqual(len(storage), 8)
        expected = torch.cat([t1, t2], dim=0)
        self.assertTrue(torch.allclose(storage.tensor(), expected))

    def test_pop_fifo(self):
        """Test that pop removes from the front (FIFO behavior)."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        t1 = torch.arange(30).reshape(3, 10).float()
        t2 = torch.arange(30, 50).reshape(2, 10).float()
        storage.append(t1)
        storage.append(t2)
        popped = storage.pop(2)
        self.assertEqual(popped.shape, (2, 10))
        # Should pop first 2 elements (from t1)
        self.assertTrue(torch.allclose(popped, t1[:2]))
        self.assertEqual(len(storage), 3)

    def test_pop_zero(self):
        """Test popping zero elements."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        tensor = torch.randn(5, 10)
        storage.append(tensor)
        popped = storage.pop(0)
        self.assertEqual(popped.shape, (0, 10))
        self.assertEqual(len(storage), 5)

    def test_circular_buffer_wrap(self):
        """Test circular buffer behavior when wrapping around."""
        storage = QueueTensorStorage((0, 10), initial_size=8)
        # Fill partially
        t1 = torch.randn(5, 10)
        storage.append(t1)
        # Pop some from front
        storage.pop(3)
        self.assertEqual(len(storage), 2)
        # Append more - should wrap around
        t2 = torch.randn(4, 10)
        storage.append(t2)
        self.assertEqual(len(storage), 6)
        # Verify data integrity
        result = storage.tensor()
        expected = torch.cat([t1[3:], t2], dim=0)
        self.assertTrue(torch.allclose(result, expected))

    def test_pop_across_wrap(self):
        """Test popping when data spans the buffer wrap point."""
        storage = QueueTensorStorage((0, 10), initial_size=8)
        # Setup: fill, pop, append to create wrap
        storage.append(torch.randn(6, 10))
        storage.pop(4)  # usage_start = 4
        storage.append(torch.randn(5, 10))  # wraps around
        self.assertEqual(len(storage), 7)
        # Pop across wrap boundary
        popped = storage.pop(5)
        self.assertEqual(popped.shape, (5, 10))
        self.assertEqual(len(storage), 2)

    def test_reorder(self):
        """Test reordering elements in queue storage."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        tensor = torch.arange(40).reshape(4, 10).float()
        storage.append(tensor)
        indices = torch.tensor([3, 1, 0, 2])
        storage.reorder(4, indices)
        expected = tensor[indices]
        self.assertTrue(torch.allclose(storage.tensor(), expected))

    def test_reorder_with_wrap(self):
        """Test reordering when buffer has wrapped."""
        storage = QueueTensorStorage((0, 10), initial_size=8)
        # Create wrapped state
        storage.append(torch.randn(5, 10))
        storage.pop(3)
        tensor = torch.arange(40).reshape(4, 10).float()
        storage.append(tensor)
        # Clear and add fresh data for reorder test
        storage.pop(len(storage))
        tensor = torch.arange(40).reshape(4, 10).float()
        storage.append(tensor)
        indices = torch.tensor([2, 0, 1, 3])
        storage.reorder(4, indices)
        result = storage.tensor()
        expected = tensor[indices]
        self.assertTrue(torch.allclose(result, expected))

    def test_reorder_raises_on_invalid_num_domains(self):
        """Test that reorder raises ValueError for invalid num_domains."""
        storage = QueueTensorStorage((0, 10), initial_size=16)
        storage.append(torch.randn(3, 10))
        with self.assertRaises(ValueError):
            storage.reorder(5, torch.tensor([0, 1, 2, 3, 4]))

    def test_tensor_makes_contiguous(self):
        """Test that tensor() returns contiguous data."""
        storage = QueueTensorStorage((0, 10), initial_size=8)
        # Create wrapped state
        storage.append(torch.randn(6, 10))
        storage.pop(4)
        storage.append(torch.randn(4, 10))
        # tensor() should reorganize to make data contiguous
        result = storage.tensor()
        self.assertEqual(result.shape[0], 6)
        self.assertTrue(result.is_contiguous())

    def test_calculate_memory(self):
        """Test memory calculation for queue storage."""
        storage = QueueTensorStorage((0, 10), initial_size=100, dtype=torch.float32)
        storage.append(torch.randn(50, 10))
        allocated_mb, used_mb = storage.calculate_memory()
        self.assertGreater(allocated_mb, 0)
        self.assertGreater(used_mb, 0)


class TestHugeTensorAllocator(unittest.TestCase):
    """Tests for HugeTensorAllocator class."""

    def test_is_supported_returns_bool(self):
        """Test that is_supported returns a boolean."""
        result = HugeTensorAllocator.is_supported()
        self.assertIsInstance(result, bool)

    def test_allocate_fallback(self):
        """Test allocation fallback when hugepages not supported."""
        # This should work regardless of hugepage support
        tensor = HugeTensorAllocator.allocate((10, 10), torch.float32)
        self.assertEqual(tensor.shape, (10, 10))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_allocate_empty_tensor(self):
        """Test allocation of empty tensor."""
        tensor = HugeTensorAllocator.allocate((0, 10), torch.float32)
        self.assertEqual(tensor.shape, (0, 10))

    def test_free_regular_tensor(self):
        """Test freeing a regular tensor returns None (no-op)."""
        tensor = torch.randn(10, 10)
        result = HugeTensorAllocator.free(tensor)
        self.assertIsNone(result)

    def test_free_none(self):
        """Test freeing None pointer returns None."""
        result = HugeTensorAllocator.free(None)
        self.assertIsNone(result)


class TestTensorStorageAttributes(unittest.TestCase):
    """Tests for TensorStorage attribute proxying."""

    def test_shape_attribute(self):
        """Test accessing shape through proxy."""
        storage = StackTensorStorage((0, 10), initial_size=16)
        storage.append(torch.randn(5, 10))
        # Access shape through __getattr__
        self.assertEqual(storage.size(), (5, 10))

    def test_device_attribute(self):
        """Test accessing device through proxy."""
        storage = StackTensorStorage((0, 10), initial_size=16, device='cpu')
        storage.append(torch.randn(5, 10))
        self.assertEqual(storage.tensor().device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
