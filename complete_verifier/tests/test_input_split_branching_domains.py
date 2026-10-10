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
"""Unit tests for input_split/branching_domains.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_split.branching_domains import (
    InputDomainList,
    UnsortedInputDomainList,
    UnsortedMultiSpecInputDomainList,
)


# ============================================================================
# UnsortedInputDomainList Initialization Tests
# ============================================================================

class TestUnsortedInputDomainListInit(unittest.TestCase):
    """Tests for UnsortedInputDomainList initialization."""

    def test_basic_init(self):
        """Test basic initialization with default parameters."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        self.assertEqual(domain_list.storage_depth, 10)
        self.assertEqual(domain_list.output_device, 'cpu')
        self.assertFalse(domain_list.use_alpha)
        self.assertIsNone(domain_list.sort_index)
        self.assertTrue(domain_list.sort_descending)
        self.assertTrue(domain_list.use_split_idx)

    def test_init_with_use_alpha(self):
        """Test initialization with use_alpha=True."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_alpha=True
        )
        self.assertTrue(domain_list.use_alpha)
        self.assertEqual(domain_list.alpha, {})

    def test_init_with_sort_index(self):
        """Test initialization with custom sort_index."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_index=0
        )
        self.assertEqual(domain_list.sort_index, 0)

    def test_init_with_sort_descending_false(self):
        """Test initialization with sort_descending=False."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_descending=False
        )
        self.assertFalse(domain_list.sort_descending)

    def test_init_with_use_split_idx_false(self):
        """Test initialization with use_split_idx=False."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_split_idx=False
        )
        self.assertFalse(domain_list.use_split_idx)

    def test_initial_storage_is_none(self):
        """Test that storage attributes are initially None."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        self.assertIsNone(domain_list.lb)
        self.assertIsNone(domain_list.dm_l)
        self.assertIsNone(domain_list.dm_u)
        self.assertIsNone(domain_list.cs)
        self.assertIsNone(domain_list.threshold)
        self.assertIsNone(domain_list.split_idx)


class TestUnsortedInputDomainListLen(unittest.TestCase):
    """Tests for UnsortedInputDomainList __len__ method."""

    def test_len_empty(self):
        """Test length when domain list is empty."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        self.assertEqual(len(domain_list), 0)

    def test_len_after_add(self):
        """Test length after adding domains."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        # Add some domains
        lb = torch.tensor([[-1.0], [-0.5]])  # Unverified
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)
        self.assertEqual(len(domain_list), 2)


# ============================================================================
# get_remaining_index Tests
# ============================================================================

class TestGetRemainingIndex(unittest.TestCase):
    """Tests for UnsortedInputDomainList.get_remaining_index static method."""

    def test_all_verified_by_lb(self):
        """Test when all domains are verified by lb > threshold."""
        batch = 3
        lb = torch.tensor([[1.0], [0.5], [0.1]])
        threshold = torch.zeros(3, 1)
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)

        remaining = UnsortedInputDomainList.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_dm_lbs=True, check_input_boxes=True
        )
        # All verified, remaining should be tensor with 0 elements
        if isinstance(remaining, torch.Tensor):
            self.assertEqual(remaining.shape[0], 0)
        else:
            self.fail(f"Expected empty tensor when all verified, got {type(remaining)}")

    def test_none_verified(self):
        """Test when no domains are verified."""
        batch = 3
        lb = torch.tensor([[-1.0], [-0.5], [-0.1]])
        threshold = torch.zeros(3, 1)
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)

        remaining = UnsortedInputDomainList.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_dm_lbs=True, check_input_boxes=True
        )
        # All unverified, should return slice(None)
        if isinstance(remaining, slice):
            self.assertEqual(remaining, slice(None))
        else:
            self.assertEqual(len(remaining), 3)

    def test_partial_verified(self):
        """Test when some domains are verified."""
        batch = 4
        lb = torch.tensor([[1.0], [-0.5], [0.5], [-0.1]])  # 0,2 verified
        threshold = torch.zeros(4, 1)
        dm_l = torch.zeros(4, 4)
        dm_u = torch.ones(4, 4)

        remaining = UnsortedInputDomainList.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_dm_lbs=True, check_input_boxes=True
        )
        self.assertIsInstance(remaining, torch.Tensor)
        self.assertEqual(len(remaining), 2)
        # Indices 1 and 3 should remain (lb <= threshold means unverified)
        self.assertTrue(torch.equal(remaining, torch.tensor([1, 3])))

    def test_verified_by_input_boxes(self):
        """Test verification by dm_l > dm_u (impossible box)."""
        batch = 3
        lb = torch.tensor([[-1.0], [-0.5], [-0.1]])
        threshold = torch.zeros(3, 1)
        dm_l = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                             [0.5, 0.5, 0.5, 0.5],  # Will be > dm_u
                             [0.0, 0.0, 0.0, 0.0]])
        dm_u = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                             [0.4, 0.4, 0.4, 0.4],  # < dm_l, invalid box
                             [1.0, 1.0, 1.0, 1.0]])

        remaining = UnsortedInputDomainList.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_dm_lbs=True, check_input_boxes=True
        )
        self.assertIsInstance(remaining, torch.Tensor)
        self.assertEqual(len(remaining), 2)  # Index 1 filtered out

    def test_no_checks(self):
        """Test with both checks disabled."""
        batch = 3
        lb = torch.tensor([[1.0], [0.5], [0.1]])  # All would be verified
        threshold = torch.zeros(3, 1)
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)

        remaining = UnsortedInputDomainList.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_dm_lbs=False, check_input_boxes=False
        )
        # No checks, all remain
        self.assertEqual(remaining, slice(None))


# ============================================================================
# filter_verified_domains Tests
# ============================================================================

class TestFilterVerifiedDomains(unittest.TestCase):
    """Tests for UnsortedInputDomainList.filter_verified_domains static method."""

    def test_filters_verified_domains(self):
        """Test that verified domains are filtered out."""
        batch = 4
        lb = torch.tensor([[1.0], [-0.5], [0.5], [-0.1]])
        dm_l = torch.zeros(4, 4)
        dm_u = torch.ones(4, 4)
        alpha = {}
        cs = torch.eye(1).unsqueeze(0).expand(4, 1, 1)
        threshold = torch.zeros(4, 1)

        batch_filt, lb_filt, dm_l_filt, dm_u_filt, alpha_filt, cs_filt, threshold_filt, *rest = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold
            )

        self.assertEqual(batch_filt, 2)  # 2 unverified domains
        self.assertEqual(lb_filt.shape[0], 2)

    def test_no_filtering_when_none_verified(self):
        """Test no filtering when all domains are unverified."""
        batch = 3
        lb = torch.tensor([[-1.0], [-0.5], [-0.1]])
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)
        alpha = {}
        cs = torch.eye(1).unsqueeze(0).expand(3, 1, 1)
        threshold = torch.zeros(3, 1)

        batch_filt, lb_filt, *rest = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold
            )

        self.assertEqual(batch_filt, 3)

    def test_with_constraints(self):
        """Test filtering with constraints tuple."""
        batch = 2
        lb = torch.tensor([[-1.0], [-0.5]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        alpha = {}
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        constr_A = torch.randn(2, 3, 4)
        constr_b = torch.randn(2, 3)
        constraints = (constr_A, constr_b)

        batch_filt, lb_filt, dm_l_filt, dm_u_filt, alpha_filt, cs_filt, threshold_filt, \
            lA_filt, lbias_filt, constraints_filt, split_idx_filt, spec_sizes_filt = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold,
                constraints=constraints
            )

        self.assertIsNotNone(constraints_filt)
        self.assertEqual(constraints_filt[0].shape[0], 2)

    def test_with_split_idx(self):
        """Test filtering with split_idx."""
        batch = 2
        lb = torch.tensor([[-1.0], [-0.5]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        alpha = {}
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        *rest, split_idx_filt, _ = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold,
                split_idx=split_idx
            )

        self.assertIsNotNone(split_idx_filt)
        self.assertEqual(split_idx_filt.shape[0], 2)

    def test_with_remaining_index_provided(self):
        """Test with explicitly provided remaining_index."""
        batch = 4
        lb = torch.tensor([[-1.0], [-0.5], [-0.3], [-0.1]])
        dm_l = torch.zeros(4, 4)
        dm_u = torch.ones(4, 4)
        alpha = {}
        cs = torch.eye(1).unsqueeze(0).expand(4, 1, 1)
        threshold = torch.zeros(4, 1)
        remaining_index = torch.tensor([1, 3])  # Only keep indices 1 and 3

        batch_filt, lb_filt, *rest = \
            UnsortedInputDomainList.filter_verified_domains(
                batch, lb, dm_l, dm_u, alpha, cs, threshold,
                remaining_index=remaining_index
            )

        self.assertEqual(batch_filt, 2)
        self.assertTrue(torch.allclose(lb_filt, lb[[1, 3]]))


# ============================================================================
# UnsortedInputDomainList.add Tests
# ============================================================================

class TestUnsortedInputDomainListAdd(unittest.TestCase):
    """Tests for UnsortedInputDomainList.add method."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )

    def test_add_single_batch(self):
        """Test adding a single batch of domains."""
        lb = torch.tensor([[-1.0], [-0.5]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        self.assertEqual(len(self.domain_list), 2)

    def test_add_empty_batch(self):
        """Test adding an empty batch."""
        lb = torch.tensor([]).reshape(0, 1)
        dm_l = torch.tensor([]).reshape(0, 4)
        dm_u = torch.tensor([]).reshape(0, 4)
        cs = torch.tensor([]).reshape(0, 1, 1)
        threshold = torch.tensor([]).reshape(0, 1)
        split_idx = torch.tensor([], dtype=torch.long).reshape(0, 10)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        self.assertEqual(len(self.domain_list), 0)

    def test_add_filters_verified(self):
        """Test that add filters out verified domains."""
        lb = torch.tensor([[1.0], [-0.5]])  # First verified, second not
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        self.assertEqual(len(self.domain_list), 1)  # Only unverified added

    def test_add_multiple_batches(self):
        """Test adding multiple batches."""
        for i in range(3):
            lb = torch.tensor([[-float(i+1)]])
            dm_l = torch.zeros(1, 4)
            dm_u = torch.ones(1, 4)
            cs = torch.eye(1).unsqueeze(0)
            threshold = torch.zeros(1, 1)
            split_idx = torch.zeros(1, 10, dtype=torch.long)

            self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        self.assertEqual(len(self.domain_list), 3)

    def test_add_without_split_idx(self):
        """Test adding when use_split_idx=False."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_split_idx=False
        )
        lb = torch.tensor([[-1.0]])
        dm_l = torch.zeros(1, 4)
        dm_u = torch.ones(1, 4)
        cs = torch.eye(1).unsqueeze(0)
        threshold = torch.zeros(1, 1)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold)

        self.assertEqual(len(domain_list), 1)


# ============================================================================
# UnsortedInputDomainList.pick_out_batch Tests
# ============================================================================

class TestUnsortedInputDomainListPickOutBatch(unittest.TestCase):
    """Tests for UnsortedInputDomainList.pick_out_batch method."""

    def setUp(self):
        """Set up test fixtures with domains added."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        # Add 5 domains
        lb = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]])
        dm_l = torch.zeros(5, 4)
        dm_u = torch.ones(5, 4)
        cs = torch.eye(1).unsqueeze(0).expand(5, 1, 1)
        threshold = torch.zeros(5, 1)
        split_idx = torch.zeros(5, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

    def test_pick_out_full_batch(self):
        """Test picking out a full batch."""
        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=5)

        self.assertEqual(lb.shape[0], 5)
        self.assertEqual(len(self.domain_list), 0)  # All picked out

    def test_pick_out_partial_batch(self):
        """Test picking out a partial batch."""
        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=3)

        self.assertEqual(lb.shape[0], 3)
        self.assertEqual(len(self.domain_list), 2)  # 2 remaining

    def test_pick_out_more_than_available(self):
        """Test picking out more than available (should pick all)."""
        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=10)

        self.assertEqual(lb.shape[0], 5)  # Only 5 available
        self.assertEqual(len(self.domain_list), 0)

    def test_pick_out_returns_correct_shapes(self):
        """Test that pick_out_batch returns tensors with correct shapes."""
        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=2)

        self.assertEqual(lb.shape, (2, 1))
        self.assertEqual(dm_l.shape, (2, 4))
        self.assertEqual(dm_u.shape, (2, 4))
        self.assertEqual(cs.shape, (2, 1, 1))
        self.assertEqual(threshold.shape, (2, 1))
        self.assertEqual(split_idx.shape, (2, 10))
        self.assertEqual(spec_sizes.shape, (2,))

    def test_pick_out_to_specific_device(self):
        """Test picking out to a specific device."""
        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=2, device='cpu')

        self.assertEqual(lb.device.type, 'cpu')
        self.assertEqual(dm_l.device.type, 'cpu')


# ============================================================================
# UnsortedInputDomainList.__getitem__ Tests
# ============================================================================

class TestUnsortedInputDomainListGetItem(unittest.TestCase):
    """Tests for UnsortedInputDomainList.__getitem__ method."""

    def setUp(self):
        """Set up test fixtures with domains added."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        # Add 5 domains with distinct values
        self.lb = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]])
        dm_l = torch.arange(5).unsqueeze(1).expand(5, 4).float()
        dm_u = torch.arange(5).unsqueeze(1).expand(5, 4).float() + 1
        cs = torch.eye(1).unsqueeze(0).expand(5, 1, 1)
        threshold = torch.zeros(5, 1)
        split_idx = torch.zeros(5, 10, dtype=torch.long)

        self.domain_list.add(self.lb, dm_l, dm_u, {}, cs, threshold,
                            split_idx=split_idx, check_dm_lbs=False)

    def test_getitem_single_index(self):
        """Test getting a single domain by index."""
        lb, dm_l, dm_u, cs, threshold, spec_size = self.domain_list[[0]]

        self.assertEqual(lb.shape[0], 1)

    def test_getitem_multiple_indices(self):
        """Test getting multiple domains by indices."""
        lb, dm_l, dm_u, cs, threshold, spec_size = self.domain_list[[0, 2, 4]]

        self.assertEqual(lb.shape[0], 3)

    def test_getitem_slice(self):
        """Test getting domains by slice."""
        lb, dm_l, dm_u, cs, threshold, spec_size = self.domain_list[:3]

        self.assertEqual(lb.shape[0], 3)

    def test_getitem_tensor_index(self):
        """Test getting domains by tensor index."""
        idx = torch.tensor([1, 3])
        lb, dm_l, dm_u, cs, threshold, spec_size = self.domain_list[idx]

        self.assertEqual(lb.shape[0], 2)


# ============================================================================
# UnsortedInputDomainList Volume Tracking Tests
# ============================================================================

class TestUnsortedInputDomainListVolume(unittest.TestCase):
    """Tests for volume tracking in UnsortedInputDomainList."""

    def setUp(self):
        """Set up test fixture."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )

    def test_initial_volume_none(self):
        """Test initial volume is None."""
        self.assertIsNone(self.domain_list.volume)
        self.assertIsNone(self.domain_list.all_volume)

    def test_add_volume_tracking(self):
        """Test volume tracking after add."""
        lb = torch.tensor([[-1.0]])
        dm_l = torch.zeros(1, 4)
        dm_u = torch.ones(1, 4)  # Volume = 1
        cs = torch.eye(1).unsqueeze(0)
        threshold = torch.zeros(1, 1)
        split_idx = torch.zeros(1, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        self.assertIsNotNone(self.domain_list.all_volume)

    def test_get_progress_empty(self):
        """Test get_progess returns 0 when empty."""
        progress = self.domain_list.get_progess()
        self.assertEqual(progress, 0.0)

    def test_get_progress_after_add_and_pick(self):
        """Test get_progess after adding and picking out."""
        lb = torch.tensor([[-1.0], [-2.0]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)
        self.domain_list.pick_out_batch(1)

        progress = self.domain_list.get_progess()
        self.assertGreater(progress, 0.0)
        self.assertLess(progress, 1.0)


# ============================================================================
# UnsortedInputDomainList Sorting Tests
# ============================================================================

class TestUnsortedInputDomainListSort(unittest.TestCase):
    """Tests for UnsortedInputDomainList sorting functionality."""

    def setUp(self):
        """Set up test fixture with unsorted domains."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_descending=True
        )
        # Add domains with different lb values
        lb = torch.tensor([[-1.0], [-5.0], [-3.0], [-2.0], [-4.0]])
        dm_l = torch.zeros(5, 4)
        dm_u = torch.ones(5, 4)
        cs = torch.eye(1).unsqueeze(0).expand(5, 1, 1)
        threshold = torch.zeros(5, 1)
        split_idx = torch.arange(5).unsqueeze(1).expand(5, 10)  # Distinct values to track order

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

    def test_sort_descending(self):
        """Test sorting in descending order (worst first)."""
        self.domain_list.sort()

        # Get all domains
        lb, *_ = self.domain_list[:5]

        # Should be sorted in descending order of lb - threshold
        for i in range(len(lb) - 1):
            self.assertGreaterEqual(lb[i].item(), lb[i+1].item())

    def test_sort_ascending(self):
        """Test sorting in ascending order."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_descending=False
        )
        lb = torch.tensor([[-1.0], [-5.0], [-3.0]])
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)
        cs = torch.eye(1).unsqueeze(0).expand(3, 1, 1)
        threshold = torch.zeros(3, 1)
        split_idx = torch.zeros(3, 10, dtype=torch.long)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)
        domain_list.sort()

        lb_sorted, *_ = domain_list[:3]
        for i in range(len(lb_sorted) - 1):
            self.assertLessEqual(lb_sorted[i].item(), lb_sorted[i+1].item())


# ============================================================================
# UnsortedInputDomainList.get_topk_indices Tests
# ============================================================================

class TestUnsortedInputDomainListTopK(unittest.TestCase):
    """Tests for UnsortedInputDomainList.get_topk_indices method."""

    def setUp(self):
        """Set up test fixture with domains."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        lb = torch.tensor([[-1.0], [-5.0], [-3.0], [-2.0], [-4.0]])
        dm_l = torch.zeros(5, 4)
        dm_u = torch.ones(5, 4)
        cs = torch.eye(1).unsqueeze(0).expand(5, 1, 1)
        threshold = torch.zeros(5, 1)
        split_idx = torch.zeros(5, 10, dtype=torch.long)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

    def test_topk_worst(self):
        """Test getting top k worst (smallest) margins."""
        # lb values: [-1.0, -5.0, -3.0, -2.0, -4.0] at indices 0,1,2,3,4
        # Smallest (worst) are: -5.0 (idx 1), -4.0 (idx 4)
        indices = self.domain_list.get_topk_indices(k=2, largest=False)

        self.assertEqual(len(indices), 2)
        # Both indices 1 (-5.0) and 4 (-4.0) should be in top 2 worst
        self.assertIn(1, indices.tolist())  # -5.0 is at index 1
        self.assertIn(4, indices.tolist())  # -4.0 is at index 4

    def test_topk_best(self):
        """Test getting top k best (largest) margins."""
        # lb values: [-1.0, -5.0, -3.0, -2.0, -4.0] at indices 0,1,2,3,4
        # Largest (best) are: -1.0 (idx 0), -2.0 (idx 3)
        indices = self.domain_list.get_topk_indices(k=2, largest=True)

        self.assertEqual(len(indices), 2)
        # Both indices 0 (-1.0) and 3 (-2.0) should be in top 2 best
        self.assertIn(0, indices.tolist())  # -1.0 is at index 0
        self.assertIn(3, indices.tolist())  # -2.0 is at index 3

    def test_topk_with_margins(self):
        """Test getting top k with margins returned."""
        # lb values: [-1.0, -5.0, -3.0, -2.0, -4.0] at indices 0,1,2,3,4
        # Top 3 smallest (worst): -5.0, -4.0, -3.0 at indices 1, 4, 2
        indices, margins = self.domain_list.get_topk_indices(k=3, largest=False, return_margin=True)

        self.assertEqual(len(indices), 3)
        self.assertEqual(len(margins), 3)
        # Verify the indices returned are the correct 3 worst
        self.assertIn(1, indices.tolist())  # -5.0
        self.assertIn(4, indices.tolist())  # -4.0
        self.assertIn(2, indices.tolist())  # -3.0
        # Verify margins are sorted (smallest first for largest=False)
        expected_margins = torch.tensor([-5.0, -4.0, -3.0])
        self.assertTrue(torch.allclose(margins, expected_margins))


# ============================================================================
# UnsortedInputDomainList._get_sort_margin Tests
# ============================================================================

class TestGetSortMargin(unittest.TestCase):
    """Tests for UnsortedInputDomainList._get_sort_margin method."""

    def test_sort_margin_without_sort_index(self):
        """Test sort margin without sort_index uses max."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_index=None
        )
        margin = torch.tensor([[-1.0, -2.0, -3.0], [0.5, 0.1, 0.3]])

        result = domain_list._get_sort_margin(margin)

        expected = torch.tensor([-1.0, 0.5])  # max per row
        self.assertTrue(torch.allclose(result, expected))

    def test_sort_margin_with_sort_index(self):
        """Test sort margin with specific sort_index."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            sort_index=1
        )
        margin = torch.tensor([[-1.0, -2.0, -3.0], [0.5, 0.1, 0.3]])

        result = domain_list._get_sort_margin(margin)

        expected = torch.tensor([-2.0, 0.1])  # column at index 1
        self.assertTrue(torch.allclose(result, expected))


# ============================================================================
# UnsortedMultiSpecInputDomainList Initialization Tests
# ============================================================================

class TestUnsortedMultiSpecInputDomainListInit(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        or_spec_size = torch.tensor([1, 2, 3])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        self.assertEqual(len(domain_list.spec_domain_lists), 3)
        self.assertEqual(domain_list.output_device, 'cpu')
        # or_spec_size should be sorted in descending order
        self.assertEqual(domain_list.or_spec_size.tolist(), [3, 2, 1])

    def test_init_with_alpha(self):
        """Test initialization with use_alpha=True."""
        or_spec_size = torch.tensor([1, 2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu',
            use_alpha=True,
            alpha_final_name='final'
        )

        self.assertTrue(domain_list.use_alpha)
        self.assertEqual(domain_list.alpha_final_name, 'final')

    def test_duplicate_spec_sizes_handled(self):
        """Test that duplicate spec sizes are deduplicated."""
        or_spec_size = torch.tensor([1, 2, 2, 3, 1])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        # Should only have 3 unique sizes
        self.assertEqual(len(domain_list.spec_domain_lists), 3)


class TestUnsortedMultiSpecInputDomainListLen(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList __len__ method."""

    def test_len_empty(self):
        """Test length when all domain lists are empty."""
        or_spec_size = torch.tensor([1, 2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        self.assertEqual(len(domain_list), 0)


# ============================================================================
# UnsortedMultiSpecInputDomainList._get_pickout_decision Tests
# ============================================================================

class TestGetPickoutDecision(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList._get_pickout_decision method."""

    def test_single_list_sufficient(self):
        """Test when single domain list can fulfill batch."""
        or_spec_size = torch.tensor([1, 2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        # Add 10 domains to first spec list (spec_size=2)
        lb = torch.randn(10, 2) - 5  # All unverified
        dm_l = torch.zeros(10, 4)
        dm_u = torch.ones(10, 4)
        cs = torch.randn(10, 2, 10)
        threshold = torch.zeros(10, 2)
        spec_sizes = torch.full((10,), 2)
        split_idx = torch.zeros(10, 10, dtype=torch.long)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, spec_sizes,
                       split_idx=split_idx, check_dm_lbs=False)

        dl_idx, num_per_dl = domain_list._get_pickout_decision(5)

        self.assertEqual(len(dl_idx), 1)
        self.assertEqual(num_per_dl, [5])

    def test_multiple_lists_needed(self):
        """Test when multiple domain lists are needed."""
        or_spec_size = torch.tensor([1, 2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        # Add 3 domains to spec_size=2 list
        lb1 = torch.randn(3, 2) - 5
        dm_l1 = torch.zeros(3, 4)
        dm_u1 = torch.ones(3, 4)
        cs1 = torch.randn(3, 2, 10)
        threshold1 = torch.zeros(3, 2)
        spec_sizes1 = torch.full((3,), 2)
        split_idx1 = torch.zeros(3, 10, dtype=torch.long)

        domain_list.add(lb1, dm_l1, dm_u1, {}, cs1, threshold1, spec_sizes1,
                       split_idx=split_idx1, check_dm_lbs=False)

        # Add 3 domains to spec_size=1 list
        lb2 = torch.randn(3, 1) - 5
        dm_l2 = torch.zeros(3, 4)
        dm_u2 = torch.ones(3, 4)
        cs2 = torch.randn(3, 1, 10)
        threshold2 = torch.zeros(3, 1)
        spec_sizes2 = torch.full((3,), 1)
        split_idx2 = torch.zeros(3, 10, dtype=torch.long)

        domain_list.add(lb2, dm_l2, dm_u2, {}, cs2, threshold2, spec_sizes2,
                       split_idx=split_idx2, check_dm_lbs=False)

        dl_idx, num_per_dl = domain_list._get_pickout_decision(5)

        self.assertEqual(len(dl_idx), 2)
        self.assertEqual(sum(num_per_dl), 5)


# ============================================================================
# UnsortedMultiSpecInputDomainList._get_num_domains_per_dl Tests
# ============================================================================

class TestGetNumDomainsPerDl(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList._get_num_domains_per_dl method."""

    def test_empty_lists(self):
        """Test with all empty domain lists."""
        or_spec_size = torch.tensor([1, 2, 3])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        num_per_dl = domain_list._get_num_domains_per_dl()

        self.assertEqual(num_per_dl.tolist(), [0, 0, 0])


# ============================================================================
# UnsortedMultiSpecInputDomainList._get_global_offsets_per_dl Tests
# ============================================================================

class TestGetGlobalOffsetsPerDl(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList._get_global_offsets_per_dl method."""

    def test_empty_lists(self):
        """Test with all empty domain lists."""
        or_spec_size = torch.tensor([1, 2, 3])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        offsets = domain_list._get_global_offsets_per_dl()

        self.assertEqual(offsets.tolist(), [0, 0, 0])


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases in branching_domains."""

    def test_add_requires_split_idx_when_enabled(self):
        """Test that add raises error when split_idx expected but not provided."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_split_idx=True
        )
        lb = torch.tensor([[-1.0]])
        dm_l = torch.zeros(1, 4)
        dm_u = torch.ones(1, 4)
        cs = torch.eye(1).unsqueeze(0)
        threshold = torch.zeros(1, 1)

        with self.assertRaises(AssertionError):
            domain_list.add(lb, dm_l, dm_u, {}, cs, threshold)  # No split_idx

    def test_add_rejects_split_idx_when_disabled(self):
        """Test that add raises error when split_idx not expected but provided."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_split_idx=False
        )
        lb = torch.tensor([[-1.0]])
        dm_l = torch.zeros(1, 4)
        dm_u = torch.ones(1, 4)
        cs = torch.eye(1).unsqueeze(0)
        threshold = torch.zeros(1, 1)
        split_idx = torch.zeros(1, 10, dtype=torch.long)

        with self.assertRaises(AssertionError):
            domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

class TestWithConstraints(unittest.TestCase):
    """Tests for domain list operations with constraints."""

    def setUp(self):
        """Set up domain list with constraints."""
        self.domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )

    def test_add_with_constraints(self):
        """Test adding domains with constraints."""
        lb = torch.tensor([[-1.0], [-2.0]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)
        constr_A = torch.randn(2, 3, 4)
        constr_b = torch.randn(2, 3)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold,
                            constraints=(constr_A, constr_b),
                            split_idx=split_idx)

        self.assertEqual(len(self.domain_list), 2)
        self.assertIsNotNone(self.domain_list.constraint_A)
        self.assertIsNotNone(self.domain_list.constraint_b)

    def test_pick_out_with_constraints(self):
        """Test picking out domains with constraints."""
        lb = torch.tensor([[-1.0], [-2.0]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)
        constr_A = torch.randn(2, 3, 4)
        constr_b = torch.randn(2, 3)

        self.domain_list.add(lb, dm_l, dm_u, {}, cs, threshold,
                            constraints=(constr_A, constr_b),
                            split_idx=split_idx)

        alpha, lb, dm_l, dm_u, cs, threshold, constraints, spec_sizes, split_idx = \
            self.domain_list.pick_out_batch(batch_size=2)

        self.assertIsNotNone(constraints)
        self.assertEqual(len(constraints), 2)
        self.assertEqual(constraints[0].shape[0], 2)
        self.assertEqual(constraints[1].shape[0], 2)


class TestMultiSpecDomainOperations(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList operations."""

    def setUp(self):
        """Set up multi-spec domain list."""
        self.or_spec_size = torch.tensor([1, 2])
        self.domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=self.or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

    def test_add_domains_different_specs(self):
        """Test adding domains with different spec sizes."""
        # Add domains with spec_size=2
        lb1 = torch.randn(3, 2) - 5
        dm_l1 = torch.zeros(3, 4)
        dm_u1 = torch.ones(3, 4)
        cs1 = torch.randn(3, 2, 10)
        threshold1 = torch.zeros(3, 2)
        spec_sizes1 = torch.full((3,), 2)
        split_idx1 = torch.zeros(3, 10, dtype=torch.long)

        self.domain_list.add(lb1, dm_l1, dm_u1, {}, cs1, threshold1, spec_sizes1,
                            split_idx=split_idx1, check_dm_lbs=False)

        # Add domains with spec_size=1
        lb2 = torch.randn(2, 1) - 5
        dm_l2 = torch.zeros(2, 4)
        dm_u2 = torch.ones(2, 4)
        cs2 = torch.randn(2, 1, 10)
        threshold2 = torch.zeros(2, 1)
        spec_sizes2 = torch.full((2,), 1)
        split_idx2 = torch.zeros(2, 10, dtype=torch.long)

        self.domain_list.add(lb2, dm_l2, dm_u2, {}, cs2, threshold2, spec_sizes2,
                            split_idx=split_idx2, check_dm_lbs=False)

        self.assertEqual(len(self.domain_list), 5)

    def test_sort_all_lists(self):
        """Test that sort sorts all non-empty domain lists."""
        # Add domains to both spec lists
        lb1 = torch.tensor([[-3.0, -1.0], [-1.0, -2.0]])
        dm_l1 = torch.zeros(2, 4)
        dm_u1 = torch.ones(2, 4)
        cs1 = torch.randn(2, 2, 10)
        threshold1 = torch.zeros(2, 2)
        spec_sizes1 = torch.full((2,), 2)
        split_idx1 = torch.zeros(2, 10, dtype=torch.long)

        self.domain_list.add(lb1, dm_l1, dm_u1, {}, cs1, threshold1, spec_sizes1,
                            split_idx=split_idx1, check_dm_lbs=False)

        # Sort should not raise
        self.domain_list.sort()


class TestReportMemory(unittest.TestCase):
    """Tests for report_memory method."""

    def test_report_memory_no_error(self):
        """Test that report_memory doesn't raise errors."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu'
        )
        lb = torch.tensor([[-1.0]])
        dm_l = torch.zeros(1, 4)
        dm_u = torch.ones(1, 4)
        cs = torch.eye(1).unsqueeze(0)
        threshold = torch.zeros(1, 1)
        split_idx = torch.zeros(1, 10, dtype=torch.long)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, split_idx=split_idx)

        # Should not raise
        domain_list.report_memory()


class TestMultiSpecGetItem(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList.__getitem__ method."""

    def test_getitem_single_spec(self):
        """Test __getitem__ with single spec size."""
        or_spec_size = torch.tensor([2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        # Add domains
        lb = torch.randn(3, 2) - 5
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)
        cs = torch.randn(3, 2, 10)
        threshold = torch.zeros(3, 2)
        spec_sizes = torch.full((3,), 2)
        split_idx = torch.zeros(3, 10, dtype=torch.long)

        domain_list.add(lb, dm_l, dm_u, {}, cs, threshold, spec_sizes,
                       split_idx=split_idx, check_dm_lbs=False)

        lb_out, dm_l_out, dm_u_out, cs_out, threshold_out, spec_sizes_out = domain_list[:2]

        self.assertEqual(lb_out.shape[0], 2)
        self.assertEqual(dm_l_out.shape[0], 2)


class TestMultiSpecTopK(unittest.TestCase):
    """Tests for UnsortedMultiSpecInputDomainList.get_topk_indices method."""

    def test_topk_across_specs(self):
        """Test get_topk_indices across multiple spec sizes."""
        or_spec_size = torch.tensor([1, 2])
        domain_list = UnsortedMultiSpecInputDomainList(
            or_spec_size=or_spec_size,
            input_shape=(4,),
            output_dim=10,
            storage_depth=10,
            output_device='cpu'
        )

        # Add domains with spec_size=2 (lb around -5)
        lb1 = torch.tensor([[-5.0, -4.0], [-6.0, -5.0]])
        dm_l1 = torch.zeros(2, 4)
        dm_u1 = torch.ones(2, 4)
        cs1 = torch.randn(2, 2, 10)
        threshold1 = torch.zeros(2, 2)
        spec_sizes1 = torch.full((2,), 2)
        split_idx1 = torch.zeros(2, 10, dtype=torch.long)

        domain_list.add(lb1, dm_l1, dm_u1, {}, cs1, threshold1, spec_sizes1,
                       split_idx=split_idx1, check_dm_lbs=False)

        # Add domains with spec_size=1 (lb around -1)
        lb2 = torch.tensor([[-1.0], [-2.0]])
        dm_l2 = torch.zeros(2, 4)
        dm_u2 = torch.ones(2, 4)
        cs2 = torch.randn(2, 1, 10)
        threshold2 = torch.zeros(2, 1)
        spec_sizes2 = torch.full((2,), 1)
        split_idx2 = torch.zeros(2, 10, dtype=torch.long)

        domain_list.add(lb2, dm_l2, dm_u2, {}, cs2, threshold2, spec_sizes2,
                       split_idx=split_idx2, check_dm_lbs=False)

        # Get top 2 worst (smallest margins)
        indices = domain_list.get_topk_indices(k=2, largest=False)

        self.assertEqual(len(indices), 2)


class TestAlphaHandling(unittest.TestCase):
    """Tests for alpha parameter handling."""

    def test_add_with_alpha(self):
        """Test adding domains with alpha parameters."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_alpha=True
        )

        lb = torch.tensor([[-1.0], [-2.0]])
        dm_l = torch.zeros(2, 4)
        dm_u = torch.ones(2, 4)
        cs = torch.eye(1).unsqueeze(0).expand(2, 1, 1)
        threshold = torch.zeros(2, 1)
        split_idx = torch.zeros(2, 10, dtype=torch.long)

        # Create alpha structure
        alpha = {
            'layer1': {
                'spec': torch.randn(2, 5, 2, 10)  # (alpha_size, spec_size, batch, node_size)
            }
        }

        domain_list.add(lb, dm_l, dm_u, alpha, cs, threshold, split_idx=split_idx)

        self.assertEqual(len(domain_list), 2)
        self.assertIn('layer1', domain_list.alpha)

    def test_pick_out_with_alpha(self):
        """Test picking out domains with alpha parameters."""
        domain_list = UnsortedInputDomainList(
            storage_depth=10,
            output_device='cpu',
            use_alpha=True
        )

        lb = torch.tensor([[-1.0], [-2.0], [-3.0]])
        dm_l = torch.zeros(3, 4)
        dm_u = torch.ones(3, 4)
        cs = torch.eye(1).unsqueeze(0).expand(3, 1, 1)
        threshold = torch.zeros(3, 1)
        split_idx = torch.zeros(3, 10, dtype=torch.long)

        alpha = {
            'layer1': {
                'spec': torch.randn(2, 5, 3, 10, dtype=torch.float64)
            }
        }

        domain_list.add(lb, dm_l, dm_u, alpha, cs, threshold, split_idx=split_idx)

        alpha_out, lb_out, *_ = domain_list.pick_out_batch(batch_size=2)

        self.assertIn('layer1', alpha_out)
        self.assertEqual(alpha_out['layer1']['spec'].shape[2], 2)  # batch dim
        self.assertEqual(alpha_out['layer1']['spec'].dtype, torch.float64)

if __name__ == '__main__':
    unittest.main()
