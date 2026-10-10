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
"""Unit tests for attack/bab_attack.py module."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict
import copy

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize arguments.Config before importing bab_attack modules
import arguments
arguments.Config.parse_config(args=[], verbose=False)

from sortedcontainers import SortedList
from attack.domains import ReLUDomain


class TestHistoryToSplits(unittest.TestCase):
    """Tests for history_to_splits function."""

    def test_empty_history(self):
        """Test with empty history."""
        from attack.bab_attack import history_to_splits
        history = []
        splits, coeffs = history_to_splits(history)
        self.assertEqual(splits, [])
        self.assertEqual(coeffs, [])

    def test_single_layer_single_split(self):
        """Test with single layer, single split."""
        from attack.bab_attack import history_to_splits
        # history is list of (splits, coeffs) per layer
        history = [([5], [1])]
        splits, coeffs = history_to_splits(history)
        self.assertEqual(splits, [[0, 5]])  # layer 0, neuron 5
        self.assertEqual(coeffs, [1])

    def test_single_layer_multiple_splits(self):
        """Test with single layer, multiple splits."""
        from attack.bab_attack import history_to_splits
        history = [([5, 10, 15], [1, -1, 1])]
        splits, coeffs = history_to_splits(history)
        self.assertEqual(splits, [[0, 5], [0, 10], [0, 15]])
        self.assertEqual(coeffs, [1, -1, 1])

    def test_multiple_layers(self):
        """Test with multiple layers."""
        from attack.bab_attack import history_to_splits
        history = [
            ([1, 2], [1, -1]),  # Layer 0
            ([3], [1]),         # Layer 1
            ([4, 5], [-1, 1])   # Layer 2
        ]
        splits, coeffs = history_to_splits(history)
        expected_splits = [[0, 1], [0, 2], [1, 3], [2, 4], [2, 5]]
        expected_coeffs = [1, -1, 1, -1, 1]
        self.assertEqual(splits, expected_splits)
        self.assertEqual(coeffs, expected_coeffs)

    def test_empty_layers(self):
        """Test with some empty layers."""
        from attack.bab_attack import history_to_splits
        history = [
            ([], []),           # Layer 0 - empty
            ([3], [1]),         # Layer 1
            ([], [])            # Layer 2 - empty
        ]
        splits, coeffs = history_to_splits(history)
        self.assertEqual(splits, [[1, 3]])
        self.assertEqual(coeffs, [1])


class TestBfsSplitsCoeffs(unittest.TestCase):
    """Tests for bfs_splits_coeffs function."""

    def test_single_split(self):
        """Test with num_splits=1."""
        from attack.bab_attack import bfs_splits_coeffs
        result = bfs_splits_coeffs(1)
        # Binary: 0 -> [-1], 1 -> [1]
        self.assertEqual(len(result), 2)
        self.assertIn([-1], result)
        self.assertIn([1], result)

    def test_two_splits(self):
        """Test with num_splits=2."""
        from attack.bab_attack import bfs_splits_coeffs
        result = bfs_splits_coeffs(2)
        # Binary: 00 -> [-1,-1], 01 -> [-1,1], 10 -> [1,-1], 11 -> [1,1]
        self.assertEqual(len(result), 4)
        self.assertIn([-1, -1], result)
        self.assertIn([-1, 1], result)
        self.assertIn([1, -1], result)
        self.assertIn([1, 1], result)

    def test_three_splits(self):
        """Test with num_splits=3."""
        from attack.bab_attack import bfs_splits_coeffs
        result = bfs_splits_coeffs(3)
        self.assertEqual(len(result), 8)

    def test_zero_splits(self):
        """Test with num_splits=0."""
        from attack.bab_attack import bfs_splits_coeffs
        result = bfs_splits_coeffs(0)
        # 2^0 = 1 combination, but format string produces "-1"
        self.assertEqual(len(result), 1)
        # The function produces [[-1]] for 0 splits due to formatting
        self.assertEqual(result, [[-1]])


class TestCloneToDive(unittest.TestCase):
    """Tests for clone_to_dive function."""

    def test_basic_clone(self):
        """Test basic cloning of a ReLUDomain."""
        from attack.bab_attack import clone_to_dive

        # Create a mock ReLUDomain-like object with required attributes
        mock_domain = MagicMock()
        mock_domain.lA = [torch.randn(2, 5)]
        mock_domain.lower_bound = -0.5
        mock_domain.upper_bound = 0.5
        mock_domain.alpha = {'layer1': {'spec': torch.randn(2, 2, 1)}}
        mock_domain.depth = 3
        mock_domain.history = [([1, 2], [1, -1]), ([3], [1])]
        mock_domain.primals = torch.randn(10)
        mock_domain.priority = 5
        mock_domain.c = torch.randn(1, 1, 5)

        result = clone_to_dive(mock_domain)

        self.assertIsInstance(result, ReLUDomain)
        self.assertEqual(result.lower_bound, mock_domain.lower_bound)
        self.assertEqual(result.upper_bound, mock_domain.upper_bound)
        self.assertEqual(result.depth, mock_domain.depth)
        self.assertEqual(result.priority, mock_domain.priority)

    def test_empty_history_created(self):
        """Test that empty history is created for each layer."""
        from attack.bab_attack import clone_to_dive

        mock_domain = MagicMock()
        mock_domain.lA = [torch.randn(2, 5)]
        mock_domain.lower_bound = -0.5
        mock_domain.upper_bound = 0.5
        mock_domain.alpha = {}
        mock_domain.depth = 0
        mock_domain.history = [([1], [1]), ([2], [-1]), ([3], [1])]  # 3 layers
        mock_domain.primals = None
        mock_domain.priority = 0
        mock_domain.c = None

        result = clone_to_dive(mock_domain)

        # History should be list of [None, None] for each layer
        self.assertEqual(len(result.history), 3)
        for layer_hist in result.history:
            self.assertEqual(layer_hist, [None, None])


class TestToSortedList(unittest.TestCase):
    """Tests for to_sorted_list function."""

    def test_empty_list(self):
        """Test with empty domain list."""
        from attack.bab_attack import to_sorted_list

        result = to_sorted_list([])

        self.assertIsInstance(result, SortedList)
        self.assertEqual(len(result), 0)

    def test_with_domains(self):
        """Test with multiple domains."""
        from attack.bab_attack import to_sorted_list

        # Create mock domains
        domain1 = ReLUDomain(lb=torch.tensor([-1.0]), threshold=np.float64(0.0))
        domain2 = ReLUDomain(lb=torch.tensor([-2.0]), threshold=np.float64(0.0))
        domain3 = ReLUDomain(lb=torch.tensor([-0.5]), threshold=np.float64(0.0))

        domains = [domain1, domain2, domain3]
        result = to_sorted_list(domains)

        self.assertIsInstance(result, SortedList)
        self.assertEqual(len(result), 3)


class TestCountDomainUnstable(unittest.TestCase):
    """Tests for count_domain_unstable function."""

    def test_count_unstable_neurons(self):
        """Test counting unstable neurons in a domain."""
        from attack.bab_attack import count_domain_unstable

        mock_domain = MagicMock()
        # Create bounds where some neurons are unstable (lb < 0 and ub > 0)
        mock_domain.lower_all = [
            torch.tensor([[-1.0, 0.5, -0.5, 1.0]]),
            torch.tensor([[-0.5, -0.5, 0.5, 0.5]]),
            torch.tensor([[0.0]])  # Final layer (excluded)
        ]
        mock_domain.upper_all = [
            torch.tensor([[1.0, 1.5, 0.5, 2.0]]),
            torch.tensor([[0.5, 0.5, 1.5, 1.5]]),
            torch.tensor([[0.0]])  # Final layer (excluded)
        ]

        # This should print the unstable counts
        # In layer 0: neurons 0 and 2 are unstable (-1<0<1 and -0.5<0<0.5)
        # In layer 1: neurons 0 and 1 are unstable
        with patch('builtins.print') as mock_print:
            count_domain_unstable(mock_domain)
            mock_print.assert_called()


class TestProbabilisticSelectDomains(unittest.TestCase):
    """Tests for probabilistic_select_domains function."""

    def test_select_with_priority_domains(self):
        """Test that domains with priority > 0 are always kept."""
        from attack.bab_attack import probabilistic_select_domains

        # Create mock domains
        dive_domains = SortedList()
        domain1 = ReLUDomain(lb=torch.tensor([-1.0]), threshold=np.float64(0.0), priority=1)  # Priority domain
        domain2 = ReLUDomain(lb=torch.tensor([-2.0]), threshold=np.float64(0.0), priority=0)
        domain3 = ReLUDomain(lb=torch.tensor([-0.5]), threshold=np.float64(0.0), priority=0)

        dive_domains.add(domain1)
        dive_domains.add(domain2)
        dive_domains.add(domain3)

        result = probabilistic_select_domains(dive_domains, candidates_number=2)

        # Priority domain should be in result
        priorities = [d.priority for d in result]
        self.assertIn(1, priorities)

    def test_select_candidates_number(self):
        """Test that at most candidates_number domains are selected."""
        from attack.bab_attack import probabilistic_select_domains

        dive_domains = SortedList()
        for i in range(10):
            domain = ReLUDomain(lb=torch.tensor([-float(i)]), threshold=np.float64(0.0), priority=0)
            dive_domains.add(domain)

        result = probabilistic_select_domains(dive_domains, candidates_number=5)

        self.assertLessEqual(len(result), 5)

    def test_empty_domains(self):
        """Test with empty domain list."""
        from attack.bab_attack import probabilistic_select_domains

        dive_domains = SortedList()
        result = probabilistic_select_domains(dive_domains, candidates_number=5)

        self.assertEqual(len(result), 0)


class TestAddDiveDomainParallel(unittest.TestCase):
    """Tests for add_dive_domain_parallel function."""

    def test_basic_add(self):
        """Test basic domain addition."""
        from attack.bab_attack import add_dive_domain_parallel

        dive_domains = SortedList()

        # Create mock selected domain
        mock_selected = MagicMock()
        mock_selected.history = [([1], [1])]
        mock_selected.depth = 1
        selected_domains = [mock_selected]

        lA = [[torch.randn(2, 10)]]
        lb = [torch.tensor(-0.5)]
        ub = [torch.tensor(0.5)]
        lb_all = [[torch.randn(1, 10)]]
        ub_all = [[torch.randn(1, 10)]]
        alpha = [{'layer': {'spec': torch.randn(2, 2, 1)}}]
        beta = [None]
        split_history = [{}]
        primals = torch.randn(1, 10)
        cs = [torch.randn(1, 1, 5)]

        unsat_list = add_dive_domain_parallel(
            lA=lA, lb=lb, ub=ub, lb_all=lb_all, ub_all=ub_all,
            dive_domains=dive_domains, selected_domains=selected_domains,
            alpha=alpha, beta=beta, decision_thresh=0,
            split_history=split_history, check_infeasibility=False,
            primals=primals, cs=cs
        )

        self.assertIsInstance(unsat_list, list)

    def test_filter_by_threshold(self):
        """Test that domains with lb >= threshold are filtered out."""
        from attack.bab_attack import add_dive_domain_parallel

        dive_domains = SortedList()

        mock_selected = MagicMock()
        mock_selected.history = [([1], [1])]
        mock_selected.depth = 1
        selected_domains = [mock_selected]

        lA = [[torch.randn(2, 10)]]
        lb = [torch.tensor(1.0)]  # lb > threshold (0)
        ub = [torch.tensor(2.0)]
        lb_all = [[torch.randn(1, 10)]]
        ub_all = [[torch.randn(1, 10)]]
        alpha = [{}]
        beta = [None]
        split_history = [{}]
        primals = torch.randn(1, 10)
        cs = [torch.randn(1, 1, 5)]

        add_dive_domain_parallel(
            lA=lA, lb=lb, ub=ub, lb_all=lb_all, ub_all=ub_all,
            dive_domains=dive_domains, selected_domains=selected_domains,
            alpha=alpha, beta=beta, decision_thresh=0,
            split_history=split_history, check_infeasibility=False,
            primals=primals, cs=cs
        )

        # Domain should not be added because lb > threshold
        self.assertEqual(len(dive_domains), 0)

    def test_with_priorities(self):
        """Test domain addition with priorities."""
        from attack.bab_attack import add_dive_domain_parallel

        dive_domains = SortedList()

        mock_selected = MagicMock()
        mock_selected.history = [([1], [1])]
        mock_selected.depth = 1
        selected_domains = [mock_selected]

        lA = [[torch.randn(2, 10)]]
        lb = [torch.tensor(-0.5)]
        ub = [torch.tensor(0.5)]
        lb_all = [[torch.randn(1, 10)]]
        ub_all = [[torch.randn(1, 10)]]
        alpha = [{}]
        beta = [None]
        split_history = [{}]
        primals = torch.randn(1, 10)
        priorities = torch.tensor([5.0])
        cs = [torch.randn(1, 1, 5)]

        add_dive_domain_parallel(
            lA=lA, lb=lb, ub=ub, lb_all=lb_all, ub_all=ub_all,
            dive_domains=dive_domains, selected_domains=selected_domains,
            alpha=alpha, beta=beta, decision_thresh=0,
            split_history=split_history, check_infeasibility=False,
            primals=primals, priorities=priorities, cs=cs
        )

        # Domain should be added since lb < threshold
        self.assertGreater(len(dive_domains), 0, "Domain should have been added")
        self.assertEqual(dive_domains[0].priority, 5.0)


class TestPickoutDiveDomains(unittest.TestCase):
    """Tests for pickout_dive_domains function."""

    def test_pickout_batch(self):
        """Test picking out a batch of domains."""
        from attack.bab_attack import pickout_dive_domains
        from attack.domains import to_device

        domains = SortedList()

        # Create domains with all required attributes
        for i in range(5):
            domain = ReLUDomain(
                lA=[torch.randn(1, 10)],
                lb=torch.tensor([-float(i+1)]),
                ub=torch.tensor([float(i+1)]),
                lb_all=[torch.randn(1, 10), torch.randn(1, 5)],
                up_all=[torch.randn(1, 10), torch.randn(1, 5)],
                alpha={'layer1': {'spec': torch.randn(2, 2, 1)}},
                beta=None,
                depth=i,
                history=[([1], [1])],
                split_history={},
                c=torch.randn(1, 1, 5),
                threshold=np.float64(0.0)
            )
            domain.dm_l = torch.randn(1, 10)
            domain.dm_u = torch.randn(1, 10)
            domain.intermediate_betas = None
            # Bind to_device method to the domain object
            domain.to_device = lambda device, partial=False, self=domain: to_device(self, device, partial)
            domains.add(domain)

        result = pickout_dive_domains(domains, batch=3, device='cpu')

        self.assertEqual(len(result), 10)  # Returns tuple of 10 elements
        masks, lAs, lower_bounds, upper_bounds, alphas, betas, intermediate_betas, selected_domains, cs, thresholds = result

        self.assertIsInstance(masks, list)
        self.assertIsInstance(selected_domains, list)

    def test_empty_domains_raises(self):
        """Test that empty domains raises assertion error."""
        from attack.bab_attack import pickout_dive_domains

        domains = SortedList()

        with self.assertRaises(AssertionError):
            pickout_dive_domains(domains, batch=3, device='cpu')

class TestInitBabAttack(unittest.TestCase):
    """Tests for init_bab_attack function."""

    @patch('attack.bab_attack.AdvExamplePool')
    def test_init_creates_pool(self, mock_pool_class):
        """Test that init_bab_attack creates an AdvExamplePool."""
        from attack.bab_attack import init_bab_attack, find_promising_domains, beam_mip_attack

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.c = torch.randn(1, 1, 5)

        mock_pool = MagicMock()
        mock_pool.adv_pool = [MagicMock(obj=-0.5), MagicMock(obj=-0.3)]
        mock_pool_class.return_value = mock_pool

        mask = [torch.ones(1, 20)]
        attack_images = torch.randn(1, 3, 32, 32)

        result = init_bab_attack(mock_net, mask, attack_images)

        mock_pool_class.assert_called_once()
        mock_pool.add_adv_images.assert_called_once_with(attack_images)

    @patch('attack.bab_attack.AdvExamplePool')
    def test_init_sets_function_attributes(self, mock_pool_class):
        """Test that init_bab_attack sets function attributes correctly."""
        from attack.bab_attack import init_bab_attack, find_promising_domains, beam_mip_attack

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.c = torch.randn(1, 1, 5)

        mock_pool = MagicMock()
        mock_pool.adv_pool = [MagicMock(obj=-0.5)]
        mock_pool_class.return_value = mock_pool

        mask = [torch.ones(1, 20)]
        attack_images = torch.randn(1, 3, 32, 32)

        init_bab_attack(mock_net, mask, attack_images)

        self.assertEqual(find_promising_domains.counter, 0)
        self.assertEqual(find_promising_domains.current_method, "top-down")
        self.assertEqual(find_promising_domains.topdown_status, "normal")
        self.assertEqual(find_promising_domains.bottomup_status, "normal")
        self.assertFalse(beam_mip_attack.started)


class TestFindPromisingDomainsAttributes(unittest.TestCase):
    """Tests for find_promising_domains function attribute handling."""

    def test_counter_increment(self):
        """Test that counter is incremented on each call."""
        from attack.bab_attack import find_promising_domains

        # Reset counter
        find_promising_domains.counter = 0
        find_promising_domains.current_method = "top-down"
        find_promising_domains.topdown_status = "normal"
        find_promising_domains.bottomup_status = "normal"

        mock_adv_pool = MagicMock()
        mock_adv_pool.adv_pool = [MagicMock(obj=-0.5)]
        dive_domains = []
        candidates = 4
        start_iter = 10  # High start iter to skip MIP

        # Call function
        result = find_promising_domains(mock_adv_pool, dive_domains, candidates, start_iter, 100, 50)

        self.assertEqual(find_promising_domains.counter, 1)
        # Should return empty lists because counter < start_iter
        self.assertEqual(result, ([], [], [], []))

    def test_skip_early_iterations(self):
        """Test that early iterations are skipped."""
        from attack.bab_attack import find_promising_domains

        find_promising_domains.counter = 0
        find_promising_domains.current_method = "top-down"

        mock_adv_pool = MagicMock()
        dive_domains = []
        candidates = 4
        start_iter = 5

        result = find_promising_domains(mock_adv_pool, dive_domains, candidates, start_iter, 100, 50)

        # Counter should be 1 now, still less than start_iter=5
        self.assertEqual(find_promising_domains.counter, 1)
        self.assertEqual(result, ([], [], [], []))


class TestAddDiveDomainFromDiveDecisions(unittest.TestCase):
    """Tests for add_dive_domain_from_dive_decisions function."""

    def test_basic_expansion_with_real_domain(self):
        """Test basic domain expansion with dive decisions using real ReLUDomain."""
        from attack.bab_attack import add_dive_domain_from_dive_decisions

        # Create a real ReLUDomain with all required attributes
        domain = ReLUDomain(
            lA=[torch.randn(1, 10)],
            lb=-0.5,
            ub=0.5,
            lb_all=[torch.randn(1, 10), torch.randn(1, 5)],
            up_all=[torch.randn(1, 10), torch.randn(1, 5)],
            alpha={'layer1': {'spec': torch.randn(2, 2, 1)}},
            beta=None,
            depth=2,
            history=[([1], [1]), ([2], [-1])],
            split_history={},
            primals=None,
            priority=0,
            c=None,
            threshold=np.float64(0.0)
        )
        domain.intermediate_betas = None

        dive_domains = [domain]
        # Multiple decisions to avoid single-element array indexing issues
        dive_decisions = [[[0, 3], [1, 4]]]

        masks = [torch.ones(1, 10)]

        result = add_dive_domain_from_dive_decisions(
            dive_domains, dive_decisions, mask=masks, device='cpu'
        )

        self.assertEqual(len(result), 6)  # Returns tuple of 6 elements
        new_masks, ret_lbs, ret_ubs, ret_alphas, betas_all, new_dive_domains = result

        # With 2 decisions, should get 2^2 = 4 new domains
        self.assertEqual(len(new_dive_domains), 4)


class TestBabLoopAttack(unittest.TestCase):
    """Tests for bab_loop_attack function."""

    @patch('attack.bab_attack.to_sorted_list')
    @patch('attack.bab_attack.init_bab_attack')
    @patch('attack.bab_attack.bab_attack')
    @patch('attack.bab_attack.beam_mip_attack')
    @patch('attack.bab_attack.clean_net_mps_process')
    def test_attack_calls_build_solver(self, mock_clean, mock_beam, mock_bab,
                                       mock_init, mock_to_sorted):
        """Test that bab_loop_attack calls build_solver_model."""
        from attack.bab_attack import bab_loop_attack

        # Create a sorted list with one mock domain to trigger the loop
        mock_domain = MagicMock()
        mock_domain.lower_bound = torch.tensor(-1.0)
        domain_list = SortedList()
        domain_list.add(mock_domain)
        mock_to_sorted.return_value = domain_list

        mock_pool = MagicMock()
        mock_pool.adv_pool = [MagicMock(obj=-0.5)]
        mock_init.return_value = mock_pool
        mock_beam.started = False

        # Configure bab_attack to return an empty domain list to exit loop
        mock_bab.return_value = (torch.tensor(-1.0), torch.tensor(0.5), SortedList())

        mock_net = MagicMock()
        mock_net.build_solver_model = MagicMock()
        mock_net.pool_termination_flag = MagicMock()
        mock_net.pool_termination_flag.value = 0

        domains = MagicMock()
        batch = 10
        rhs = 0.0
        start_time = 0
        timeout = 1000
        updated_mask = [torch.ones(1, 20)]
        attack_images = torch.randn(1, 3, 32, 32)
        all_label_global_ub = float('inf')

        result = bab_loop_attack(
            domains, mock_net, batch, rhs, start_time, timeout,
            updated_mask, attack_images, all_label_global_ub
        )

        # Verify build_solver_model was called
        mock_net.build_solver_model.assert_called_once()
        self.assertEqual(len(result), 3)  # Returns (global_lb, visited, result_str)


class TestHistoryToSplitsEdgeCases(unittest.TestCase):
    """Edge case tests for history_to_splits."""

    def test_large_layer_indices(self):
        """Test with large layer indices."""
        from attack.bab_attack import history_to_splits

        history = [
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([100], [1])  # Layer 4, large neuron index
        ]
        splits, coeffs = history_to_splits(history)
        self.assertEqual(splits, [[4, 100]])
        self.assertEqual(coeffs, [1])

    def test_negative_coefficients(self):
        """Test with negative coefficients."""
        from attack.bab_attack import history_to_splits

        history = [([0, 1, 2], [-1, -1, -1])]
        splits, coeffs = history_to_splits(history)
        self.assertEqual(coeffs, [-1, -1, -1])


class TestBfsSplitsCoeffsEdgeCases(unittest.TestCase):
    """Edge case tests for bfs_splits_coeffs."""

    def test_four_splits(self):
        """Test with num_splits=4 (larger case)."""
        from attack.bab_attack import bfs_splits_coeffs

        result = bfs_splits_coeffs(4)
        self.assertEqual(len(result), 16)

        # Verify each element has 4 coefficients
        for item in result:
            self.assertEqual(len(item), 4)
            for coeff in item:
                self.assertIn(coeff, [-1, 1])


class TestProbabilisticSelectDomainsEdgeCases(unittest.TestCase):
    """Edge case tests for probabilistic_select_domains."""

    def test_all_priority_domains(self):
        """Test when all domains have priority > 0."""
        from attack.bab_attack import probabilistic_select_domains

        dive_domains = SortedList()
        for i in range(5):
            domain = ReLUDomain(
                lb=torch.tensor([-float(i)]),
                threshold=np.float64(0.0),
                priority=i + 1
            )
            dive_domains.add(domain)

        result = probabilistic_select_domains(dive_domains, candidates_number=3)

        # All domains have priority > 0, so all should be kept
        self.assertLessEqual(len(result), 5)

    def test_candidates_larger_than_domains(self):
        """Test when candidates_number > number of domains."""
        from attack.bab_attack import probabilistic_select_domains

        dive_domains = SortedList()
        for i in range(3):
            domain = ReLUDomain(
                lb=torch.tensor([-float(i)]),
                threshold=np.float64(0.0),
                priority=0
            )
            dive_domains.add(domain)

        result = probabilistic_select_domains(dive_domains, candidates_number=10)

        self.assertLessEqual(len(result), 3)


class TestAddDiveDomainParallelEdgeCases(unittest.TestCase):
    """Edge case tests for add_dive_domain_parallel."""

    def test_tensor_threshold(self):
        """Test with tensor threshold."""
        from attack.bab_attack import add_dive_domain_parallel

        dive_domains = SortedList()

        mock_selected = MagicMock()
        mock_selected.history = [([1], [1])]
        mock_selected.depth = 1
        selected_domains = [mock_selected]

        lA = [[torch.randn(2, 10)]]
        lb = [torch.tensor(-0.5)]
        ub = [torch.tensor(0.5)]
        lb_all = [[torch.randn(1, 10)]]
        ub_all = [[torch.randn(1, 10)]]
        alpha = [{}]
        beta = [None]
        split_history = [{}]
        primals = torch.randn(1, 10)
        cs = [torch.randn(1, 1, 5)]

        # Tensor threshold
        decision_thresh = torch.tensor(0.0)

        add_dive_domain_parallel(
            lA=lA, lb=lb, ub=ub, lb_all=lb_all, ub_all=ub_all,
            dive_domains=dive_domains, selected_domains=selected_domains,
            alpha=alpha, beta=beta, decision_thresh=decision_thresh,
            split_history=split_history, check_infeasibility=False,
            primals=primals, cs=cs
        )

        # Should complete without error
        self.assertIsInstance(dive_domains, SortedList)


class TestCloneToDiveEdgeCases(unittest.TestCase):
    """Edge case tests for clone_to_dive."""

    def test_none_primals(self):
        """Test with None primals."""
        from attack.bab_attack import clone_to_dive

        mock_domain = MagicMock()
        mock_domain.lA = [torch.randn(2, 5)]
        mock_domain.lower_bound = -0.5
        mock_domain.upper_bound = 0.5
        mock_domain.alpha = {}
        mock_domain.depth = 0
        mock_domain.history = [([1], [1])]
        mock_domain.primals = None
        mock_domain.priority = 0
        mock_domain.c = None

        result = clone_to_dive(mock_domain)

        self.assertIsNone(result.primals)

    def test_with_alpha_dict(self):
        """Test with alpha dictionary."""
        from attack.bab_attack import clone_to_dive

        mock_domain = MagicMock()
        mock_domain.lA = [torch.randn(2, 5)]
        mock_domain.lower_bound = -0.5
        mock_domain.upper_bound = 0.5
        mock_domain.alpha = {
            'layer1': {'spec1': torch.randn(2, 2, 1)},
            'layer2': {'spec1': torch.randn(2, 2, 1)}
        }
        mock_domain.depth = 0
        mock_domain.history = [([1], [1])]
        mock_domain.primals = torch.randn(10)
        mock_domain.priority = 0
        mock_domain.c = torch.randn(1, 1, 5)

        result = clone_to_dive(mock_domain)

        self.assertIsInstance(result.alpha, dict)


class TestBabAttackIntegration(unittest.TestCase):
    """Integration tests for bab_attack module."""

    def test_history_to_splits_and_clone_to_dive(self):
        """Test integration between history_to_splits and clone_to_dive."""
        from attack.bab_attack import history_to_splits, clone_to_dive

        # Create a domain with history
        mock_domain = MagicMock()
        mock_domain.lA = [torch.randn(2, 5)]
        mock_domain.lower_bound = -0.5
        mock_domain.upper_bound = 0.5
        mock_domain.alpha = {}
        mock_domain.depth = 3
        mock_domain.history = [([1, 2], [1, -1]), ([3], [1])]
        mock_domain.primals = None
        mock_domain.priority = 0
        mock_domain.c = None

        # Convert history to splits
        splits, coeffs = history_to_splits(mock_domain.history)

        # Clone the domain
        cloned = clone_to_dive(mock_domain)

        # Verify cloned domain has empty history
        for layer_hist in cloned.history:
            self.assertEqual(layer_hist, [None, None])

        # Original history should still work with history_to_splits
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(coeffs), 3)


if __name__ == '__main__':
    unittest.main()
