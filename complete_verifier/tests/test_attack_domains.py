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
"""Unit tests for attack/domains.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict
import copy

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_module():
    """Setup arguments.Config for all tests."""
    import arguments
    global original_config
    original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    new_config['bab']['cut']['enabled'] = False
    new_config['bab']['cut']['cplex_cuts'] = False
    new_config['bab']['cut']['cplex_cuts_revpickup'] = False
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestReLUDomain(unittest.TestCase):
    """Tests for ReLUDomain class."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def test_init_defaults(self):
        """Test ReLUDomain initialization with defaults."""
        from attack.domains import ReLUDomain
        domain = ReLUDomain()
        self.assertEqual(domain.lower_bound, -float('inf'))
        self.assertEqual(domain.upper_bound, float('inf'))
        self.assertEqual(domain.history, [])
        self.assertEqual(domain.split_history, [])
        self.assertEqual(domain.priority, 0)
        self.assertTrue(domain.valid)
        self.assertFalse(domain.split)

    def test_init_with_values(self):
        """Test ReLUDomain initialization with values."""
        from attack.domains import ReLUDomain
        lb = torch.tensor([0.1, 0.2])
        ub = torch.tensor([0.9, 0.8])
        domain = ReLUDomain(lb=lb, ub=ub, depth=3, priority=5)
        self.assertTrue(torch.equal(domain.lower_bound, lb))
        self.assertTrue(torch.equal(domain.upper_bound, ub))
        self.assertEqual(domain.depth, 3)
        self.assertEqual(domain.priority, 5)

    def test_comparison_lt_same_priority(self):
        """Test less than comparison with same priority."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        d2 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        # With default config (cplex_cuts_revpickup=False), lower lb is "less"
        self.assertTrue(d1 < d2)
        self.assertFalse(d2 < d1)

    def test_comparison_lt_different_priority(self):
        """Test less than comparison with different priority."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.5]), priority=1)
        d2 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        # Higher priority should be "less" (comes first in queue)
        self.assertTrue(d1 < d2)

    def test_comparison_le(self):
        """Test less than or equal comparison."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        d2 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        self.assertTrue(d1 <= d2)
        self.assertTrue(d2 <= d1)

    def test_comparison_eq(self):
        """Test equality comparison."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        d2 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        self.assertTrue(d1 == d2)

    def test_comparison_eq_different_priority(self):
        """Test equality with different priority."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.5]), priority=1)
        d2 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        self.assertFalse(d1 == d2)

    def test_verify_criterion_satisfied(self):
        """Test verify_criterion when satisfied."""
        from attack.domains import ReLUDomain
        domain = ReLUDomain(
            lb=torch.tensor([0.5, 0.3]),
            threshold=np.float64(0.0)
        )
        self.assertTrue(domain.verify_criterion())

    def test_verify_criterion_not_satisfied(self):
        """Test verify_criterion when not satisfied."""
        from attack.domains import ReLUDomain
        domain = ReLUDomain(
            lb=torch.tensor([-0.5, -0.3]),
            threshold=np.float64(0.0)
        )
        self.assertFalse(domain.verify_criterion())

    def test_del_node_leaf(self):
        """Test del_node on a leaf node."""
        from attack.domains import ReLUDomain
        domain = ReLUDomain()
        self.assertTrue(domain.valid)
        domain.del_node()
        self.assertFalse(domain.valid)

    def test_del_node_with_children(self):
        """Test del_node with children."""
        from attack.domains import ReLUDomain
        parent = ReLUDomain()
        left = ReLUDomain()
        right = ReLUDomain()
        parent.left = left
        parent.right = right

        parent.del_node()

        self.assertFalse(parent.valid)
        self.assertFalse(left.valid)
        self.assertFalse(right.valid)

    def test_threshold_attribute(self):
        """Test threshold attribute handling."""
        from attack.domains import ReLUDomain
        domain = ReLUDomain(threshold=0.5)
        self.assertEqual(domain.threshold, np.float64(0.5))


class TestReLUDomainWithCuts(unittest.TestCase):
    """Tests for ReLUDomain with cuts enabled."""

    def setUp(self):
        """Enable cuts for these tests."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = True
        arguments.Config['bab']['cut']['cplex_cuts'] = True
        arguments.Config['bab']['cut']['cplex_cuts_revpickup'] = True

    def tearDown(self):
        """Disable cuts after tests."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False
        arguments.Config['bab']['cut']['cplex_cuts'] = False
        arguments.Config['bab']['cut']['cplex_cuts_revpickup'] = False

    def test_comparison_lt_with_revpickup(self):
        """Test less than comparison with cplex_cuts_revpickup enabled."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        d2 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        # With revpickup, higher lb is "less" (reversed order)
        self.assertFalse(d1 < d2)
        self.assertTrue(d2 < d1)

    def test_comparison_le_with_revpickup(self):
        """Test less than or equal with cplex_cuts_revpickup enabled."""
        from attack.domains import ReLUDomain
        d1 = ReLUDomain(lb=torch.tensor([0.5]), priority=0)
        d2 = ReLUDomain(lb=torch.tensor([0.1]), priority=0)
        self.assertTrue(d1 <= d2)


class TestReLUDomainEdgeCases(unittest.TestCase):
    """Edge case tests for ReLUDomain."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def test_multidimensional_bounds(self):
        """Test with multidimensional bounds."""
        from attack.domains import ReLUDomain
        lb = torch.randn(2, 3, 4)
        ub = torch.randn(2, 3, 4)
        domain = ReLUDomain(lb=lb, ub=ub)
        self.assertEqual(domain.lower_bound.shape, (2, 3, 4))

    def test_history_list(self):
        """Test with history list."""
        from attack.domains import ReLUDomain
        history = [{'layer1': ([1, 2], [0.5, 0.6])}]
        domain = ReLUDomain(history=history)
        self.assertEqual(domain.history, history)

    def test_alpha_beta_storage(self):
        """Test alpha and beta storage."""
        from attack.domains import ReLUDomain
        alpha = {'layer1': torch.randn(10)}
        beta = {'layer1': torch.randn(5)}
        domain = ReLUDomain(alpha=alpha, beta=beta)
        self.assertIsNotNone(domain.alpha)
        self.assertIsNotNone(domain.beta)

    def test_primals_storage(self):
        """Test primals storage."""
        from attack.domains import ReLUDomain
        primals = {'p': torch.randn(10), 'z': torch.randn(5)}
        domain = ReLUDomain(primals=primals)
        self.assertEqual(domain.primals, primals)

    def test_c_storage(self):
        """Test c matrix storage."""
        from attack.domains import ReLUDomain
        c = torch.randn(1, 1, 10)
        domain = ReLUDomain(c=c)
        self.assertTrue(torch.equal(domain.c, c))

    def test_parent_child_references(self):
        """Test parent and child references."""
        from attack.domains import ReLUDomain
        parent = ReLUDomain()
        child1 = ReLUDomain()
        child2 = ReLUDomain()

        parent.left = child1
        parent.right = child2
        child1.parent = parent
        child2.parent = parent

        self.assertEqual(child1.parent, parent)
        self.assertEqual(child2.parent, parent)
        self.assertEqual(parent.left, child1)
        self.assertEqual(parent.right, child2)


class TestToCpuFunction(unittest.TestCase):
    """Tests for to_cpu function."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_cpu_basic_tensors(self):
        """Test to_cpu transfers basic tensors to CPU."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3), torch.randn(2, 4)],
            lb_all=[torch.randn(2, 5), torch.randn(2, 6)],
            up_all=[torch.randn(2, 7), torch.randn(2, 8)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
        )

        result = to_cpu(domain)

        # Check all tensors are on CPU
        for lA in result.lA:
            self.assertEqual(lA.device.type, 'cpu')
        for lb in result.lower_all:
            self.assertEqual(lb.device.type, 'cpu')
        for ub in result.upper_all:
            self.assertEqual(ub.device.type, 'cpu')
        for layer in result.alpha:
            for inter in result.alpha[layer]:
                self.assertEqual(result.alpha[layer][inter].device.type, 'cpu')

    def test_to_cpu_returns_self(self):
        """Test to_cpu returns the domain itself."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
        )

        result = to_cpu(domain)
        self.assertIs(result, domain)

    def test_to_cpu_with_split_history_beta(self):
        """Test to_cpu handles split_history with beta."""
        from attack.domains import ReLUDomain, to_cpu

        split_history = {
            "beta": [torch.randn(3, 4), None],
            "single_beta": [
                {"nonzero": torch.randn(2), "value": torch.randn(2), "c": torch.randn(2)},
                None
            ],
            "c": [torch.randn(3, 4), None],
            "coeffs": [
                {"nonzero": torch.randn(2), "coeffs": torch.randn(2)},
                None
            ],
            "bias": [torch.randn(3), None],
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=split_history,
        )

        result = to_cpu(domain)

        # Check split_history tensors are on CPU
        self.assertEqual(result.split_history["beta"][0].device.type, 'cpu')
        self.assertEqual(result.split_history["single_beta"][0]["nonzero"].device.type, 'cpu')
        self.assertEqual(result.split_history["c"][0].device.type, 'cpu')
        self.assertEqual(result.split_history["coeffs"][0]["nonzero"].device.type, 'cpu')
        self.assertEqual(result.split_history["bias"][0].device.type, 'cpu')

    def test_to_cpu_with_general_beta(self):
        """Test to_cpu handles split_history with general_beta."""
        from attack.domains import ReLUDomain, to_cpu

        split_history = {
            "general_beta": torch.randn(5, 6),
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=split_history,
        )

        result = to_cpu(domain)
        self.assertEqual(result.split_history["general_beta"].device.type, 'cpu')

    def test_to_cpu_with_intermediate_betas(self):
        """Test to_cpu handles intermediate_betas."""
        from attack.domains import ReLUDomain, to_cpu

        intermediate_betas = {
            'split_layer1': {
                'inter1': {'lb': torch.randn(3, 4), 'ub': torch.randn(3, 4)}
            }
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            intermediate_betas=intermediate_betas,
        )

        result = to_cpu(domain)
        self.assertEqual(
            result.intermediate_betas['split_layer1']['inter1']['lb'].device.type, 'cpu'
        )
        self.assertEqual(
            result.intermediate_betas['split_layer1']['inter1']['ub'].device.type, 'cpu'
        )

    def test_to_cpu_with_beta_list(self):
        """Test to_cpu handles beta as list."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=[torch.randn(3), torch.randn(4)],
        )

        result = to_cpu(domain)
        for b in result.beta:
            self.assertEqual(b.device.type, 'cpu')

    def test_to_cpu_with_beta_nested_list(self):
        """Test to_cpu handles beta as nested list when opt_interm_bounds enabled."""
        import arguments
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = True

        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=[[torch.randn(3), torch.randn(4)], [torch.randn(2)]],
        )

        result = to_cpu(domain)
        for beta_list in result.beta:
            for b in beta_list:
                self.assertEqual(b.device.type, 'cpu')

        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_cpu_with_c_tensor(self):
        """Test to_cpu handles c tensor."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            c=torch.randn(1, 1, 10),
        )

        result = to_cpu(domain)
        self.assertEqual(result.c.device.type, 'cpu')

    def test_to_cpu_alpha_converts_to_half(self):
        """Test to_cpu converts alpha to half precision."""
        from attack.domains import ReLUDomain, to_cpu

        alpha_tensor = torch.randn(3, 4, dtype=torch.float32)
        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': alpha_tensor}},
        )

        result = to_cpu(domain)
        self.assertEqual(result.alpha['layer1']['inter1'].dtype, torch.float16)


class TestToDeviceFunction(unittest.TestCase):
    """Tests for to_device function."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_device_basic_tensors(self):
        """Test to_device transfers tensors to specified device."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
        )

        result = to_device(domain, 'cpu')

        for lA in result.lA:
            self.assertEqual(lA.device.type, 'cpu')
        for lb in result.lower_all:
            self.assertEqual(lb.device.type, 'cpu')
        for ub in result.upper_all:
            self.assertEqual(ub.device.type, 'cpu')

    def test_to_device_partial_mode(self):
        """Test to_device with partial=True skips lA, lower_all, upper_all."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
        )

        # Store original references
        original_lA = domain.lA[0]

        result = to_device(domain, 'cpu', partial=True)

        # lA should not be modified (still the same tensor)
        self.assertIs(result.lA[0], original_lA)

    def test_to_device_returns_self(self):
        """Test to_device returns the domain itself."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
        )

        result = to_device(domain, 'cpu')
        self.assertIs(result, domain)

    def test_to_device_converts_alpha_dtype(self):
        """Test to_device converts alpha to default dtype."""
        from attack.domains import ReLUDomain, to_device

        alpha_tensor = torch.randn(3, 4, dtype=torch.float16)
        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': alpha_tensor}},
        )

        result = to_device(domain, 'cpu')
        self.assertEqual(result.alpha['layer1']['inter1'].dtype, torch.get_default_dtype())

    def test_to_device_with_split_history_beta(self):
        """Test to_device handles split_history with beta."""
        from attack.domains import ReLUDomain, to_device

        split_history = {
            "beta": [torch.randn(3, 4), None],
            "single_beta": [
                {"nonzero": torch.randn(2), "value": torch.randn(2), "c": torch.randn(2)},
                None
            ],
            "c": [torch.randn(3, 4), None],
            "coeffs": [
                {"nonzero": torch.randn(2), "coeffs": torch.randn(2)},
                None
            ],
            "bias": [torch.randn(3), None],
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=split_history,
        )

        result = to_device(domain, 'cpu')

        self.assertEqual(result.split_history["beta"][0].device.type, 'cpu')
        self.assertEqual(result.split_history["single_beta"][0]["nonzero"].device.type, 'cpu')

    def test_to_device_with_general_beta(self):
        """Test to_device handles split_history with general_beta."""
        from attack.domains import ReLUDomain, to_device

        split_history = {
            "general_beta": torch.randn(5, 6),
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=split_history,
        )

        result = to_device(domain, 'cpu')
        self.assertEqual(result.split_history["general_beta"].device.type, 'cpu')

    def test_to_device_with_intermediate_betas(self):
        """Test to_device handles intermediate_betas."""
        from attack.domains import ReLUDomain, to_device

        intermediate_betas = {
            'split_layer1': {
                'inter1': {'lb': torch.randn(3, 4), 'ub': torch.randn(3, 4)}
            }
        }

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            intermediate_betas=intermediate_betas,
        )

        result = to_device(domain, 'cpu')
        self.assertEqual(
            result.intermediate_betas['split_layer1']['inter1']['lb'].device.type, 'cpu'
        )

    def test_to_device_with_beta_list(self):
        """Test to_device handles beta as list."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=[torch.randn(3), torch.randn(4)],
        )

        result = to_device(domain, 'cpu')
        for b in result.beta:
            self.assertEqual(b.device.type, 'cpu')

    def test_to_device_with_beta_nested_list(self):
        """Test to_device handles beta as nested list when opt_interm_bounds enabled."""
        import arguments
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = True

        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=[[torch.randn(3), torch.randn(4)], [torch.randn(2)]],
        )

        result = to_device(domain, 'cpu')
        for beta_list in result.beta:
            for b in beta_list:
                self.assertEqual(b.device.type, 'cpu')

        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_device_with_c_tensor(self):
        """Test to_device handles c tensor."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            c=torch.randn(1, 1, 10),
        )

        result = to_device(domain, 'cpu')
        self.assertEqual(result.c.device.type, 'cpu')


class TestSortedReLUDomainListInit(unittest.TestCase):
    """Tests for SortedReLUDomainList initialization."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_mock_ret(self, num=2):
        """Create mock return dictionary."""
        return {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }

    def test_init_basic(self):
        """Test basic SortedReLUDomainList initialization."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = self._create_mock_ret(num)
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []], [[], []]]
        thresholds = torch.zeros(num)

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        self.assertEqual(len(domain_list), num)

    def test_init_with_betas(self):
        """Test SortedReLUDomainList initialization with betas."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = self._create_mock_ret(num)
        ret['betas'] = [torch.randn(3), torch.randn(3)]
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []], [[], []]]
        thresholds = torch.zeros(num)

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        self.assertEqual(len(domain_list), num)

    def test_init_single_threshold(self):
        """Test initialization with single threshold value."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = self._create_mock_ret(num)
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []], [[], []]]
        thresholds = torch.tensor([0.0])  # Single threshold

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        self.assertEqual(len(domain_list), num)

    def test_init_raises_on_input_branching(self):
        """Test initialization raises on input-space branching."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = self._create_mock_ret(num)
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []], [[], []]]
        thresholds = torch.zeros(num)

        with self.assertRaises(AssertionError):
            SortedReLUDomainList(
                ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds,
                branching_input_and_activation=True
            )


class TestSortedReLUDomainListLen(unittest.TestCase):
    """Tests for SortedReLUDomainList __len__ method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_domain_list(self, num=2):
        """Helper to create a domain list."""
        from attack.domains import SortedReLUDomainList

        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []]] * num
        thresholds = torch.zeros(num)

        return SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

    def test_len_returns_correct_count(self):
        """Test __len__ returns correct domain count."""
        domain_list = self._create_domain_list(3)
        self.assertEqual(len(domain_list), 3)

    def test_len_single_domain(self):
        """Test __len__ with single domain."""
        domain_list = self._create_domain_list(1)
        self.assertEqual(len(domain_list), 1)


class TestSortedReLUDomainListGetItem(unittest.TestCase):
    """Tests for SortedReLUDomainList __getitem__ method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_domain_list(self, num=3):
        """Helper to create a domain list."""
        from attack.domains import SortedReLUDomainList

        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        # Create predictable global_lbs for sorting
        global_lbs = torch.tensor([[float(i)] * 5 for i in range(num)])
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []]] * num
        thresholds = torch.zeros(num)

        return SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

    def test_getitem_first_element(self):
        """Test __getitem__ returns first element."""
        from attack.domains import ReLUDomain
        domain_list = self._create_domain_list(3)
        first = domain_list[0]
        self.assertIsInstance(first, ReLUDomain)

    def test_getitem_last_element(self):
        """Test __getitem__ returns last element."""
        from attack.domains import ReLUDomain
        domain_list = self._create_domain_list(3)
        last = domain_list[-1]
        self.assertIsInstance(last, ReLUDomain)

    def test_getitem_middle_element(self):
        """Test __getitem__ returns middle element."""
        from attack.domains import ReLUDomain
        domain_list = self._create_domain_list(3)
        middle = domain_list[1]
        self.assertIsInstance(middle, ReLUDomain)


class TestSortedReLUDomainListGetMinDomain(unittest.TestCase):
    """Tests for SortedReLUDomainList get_min_domain method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_domain_list(self, num=5):
        """Helper to create a domain list with predictable ordering."""
        from attack.domains import SortedReLUDomainList

        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.tensor([[float(i)] * 5 for i in range(num)])
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []]] * num
        thresholds = torch.zeros(num)

        return SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

    def test_get_min_domain_returns_list(self):
        """Test get_min_domain returns a list."""
        domain_list = self._create_domain_list(5)
        result = domain_list.get_min_domain(3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_get_min_domain_single(self):
        """Test get_min_domain with single domain request."""
        domain_list = self._create_domain_list(5)
        result = domain_list.get_min_domain(1)
        self.assertEqual(len(result), 1)

    def test_get_min_domain_reverse_order(self):
        """Test get_min_domain with reverse order."""
        domain_list = self._create_domain_list(5)
        result = domain_list.get_min_domain(3, rev_order=True)
        self.assertEqual(len(result), 3)

    def test_get_min_domain_all(self):
        """Test get_min_domain requesting all domains."""
        domain_list = self._create_domain_list(5)
        result = domain_list.get_min_domain(5)
        self.assertEqual(len(result), 5)


class TestSortedReLUDomainListToSortedList(unittest.TestCase):
    """Tests for SortedReLUDomainList to_sortedList method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_domain_list(self, num=3):
        """Helper to create a domain list."""
        from attack.domains import SortedReLUDomainList

        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = [[[], []]] * num
        thresholds = torch.zeros(num)

        return SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

    def test_to_sorted_list_returns_copy(self):
        """Test to_sortedList returns a copy."""
        from sortedcontainers import SortedList
        domain_list = self._create_domain_list(3)
        result = domain_list.to_sortedList()
        self.assertIsInstance(result, SortedList)
        self.assertEqual(len(result), 3)

    def test_to_sorted_list_is_shallow_copy(self):
        """Test to_sortedList returns a shallow copy."""
        domain_list = self._create_domain_list(3)
        result = domain_list.to_sortedList()
        # Modifying result should not affect original internal domains list
        self.assertIsNot(result, domain_list.domains)


class TestSortedReLUDomainListPickOutBasic(unittest.TestCase):
    """Basic tests for SortedReLUDomainList pick_out method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def test_pick_out_asserts_positive_batch(self):
        """Test pick_out asserts batch > 0."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = {0: [[], []], 1: [[], []]}
        thresholds = torch.zeros(num)

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        # batch > 0 is asserted
        with self.assertRaises(AssertionError):
            domain_list.pick_out(0)

    def test_pick_out_empty_list_returns_none(self):
        """Test pick_out on effectively empty list returns None values."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        # Create positive global_lbs so verify_criterion is True (verified)
        global_lbs = torch.ones(num, 5) + 1.0
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        history = {0: [[], []], 1: [[], []]}
        thresholds = torch.zeros(num)

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        # All domains should verify (lb > threshold), so pick_out should skip them
        result = domain_list.pick_out(2)
        # Result should be tuple of Nones when no valid domains
        if isinstance(result, dict):
            self.assertIsNone(result.get('mask'))
        else:
            self.assertIsNone(result[0])


class TestSortedReLUDomainListAddBasic(unittest.TestCase):
    """Basic tests for SortedReLUDomainList add method."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def _create_domain_list(self, num=2):
        """Helper to create a domain list."""
        from attack.domains import SortedReLUDomainList

        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10)]
        global_lbs = torch.zeros(num, 5) - 1.0
        global_ubs = torch.randn(num, 5)
        alphas = {'layer1': {'inter1': torch.randn(2, 3, num, 5)}}
        # history should be dict format: {layer_idx: [[neuron_indices], [signs]]}
        history = {0: [[], []], 1: [[], []]}
        thresholds = torch.zeros(num)

        return SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

    def test_add_rejects_input_split_state(self):
        """Activation-domain storage requires activation-split child bounds."""
        batch = 1
        num_add = batch * 2
        cases = {
            'x_Ls': torch.randn(num_add, 10),
            'x_Us': torch.randn(num_add, 10),
            'input_split_idx': [0, 1],
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                bounds = {
                    'lAs': [torch.randn(num_add, 3, 10)],
                    'lower_bounds': [
                        torch.randn(num_add, 10),
                        torch.zeros(num_add, 5) - 1.0],
                    'upper_bounds': [
                        torch.randn(num_add, 10), torch.randn(num_add, 5)],
                    'split_history': [[] for _ in range(num_add)],
                    'alphas': {'layer1': {
                        'inter1': torch.randn(2, 3, num_add, 5)}},
                    'betas': [None] * num_add,
                    'intermediate_betas': [None] * num_add,
                    'c': torch.randn(num_add, 1, 5),
                    'x_Ls': None,
                    'x_Us': None,
                    'input_split_idx': None,
                }
                bounds[key] = value
                with self.assertRaises(NotImplementedError):
                    self._create_domain_list(2).add(
                        bounds,
                        [{0: [[], []], 1: [[], []]}],
                        [1], [(0, 0)], torch.zeros(batch),
                        check_infeasibility=False)


class TestReLUDomainComparisonWithThreshold(unittest.TestCase):
    """Tests for ReLUDomain comparison methods with non-zero threshold."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def test_lt_with_threshold(self):
        """Test less than with non-zero threshold."""
        from attack.domains import ReLUDomain

        d1 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.2))
        d2 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.1))

        # d1.lb - d1.threshold = 0.3
        # d2.lb - d2.threshold = 0.4
        # With cuts disabled, lower value is "less"
        self.assertTrue(d1 < d2)

    def test_le_with_threshold(self):
        """Test less than or equal with non-zero threshold."""
        from attack.domains import ReLUDomain

        d1 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.2))
        d2 = ReLUDomain(lb=torch.tensor([0.6]), threshold=np.float64(0.3))

        # Both have same (lb - threshold) = 0.3
        self.assertTrue(d1 <= d2)
        self.assertTrue(d2 <= d1)

    def test_eq_with_threshold(self):
        """Test equality with non-zero threshold."""
        from attack.domains import ReLUDomain

        # Use same lb-threshold value: 0.5 - 0.1 = 0.4 and 0.5 - 0.1 = 0.4
        d1 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.1))
        d2 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.1))

        # Both have (lb - threshold) = 0.4
        self.assertTrue(d1 == d2)

    def test_verify_criterion_with_threshold(self):
        """Test verify_criterion with non-zero threshold."""
        from attack.domains import ReLUDomain

        # lb = 0.5, threshold = 0.3 -> lb > threshold -> verified
        d1 = ReLUDomain(lb=torch.tensor([0.5]), threshold=np.float64(0.3))
        self.assertTrue(d1.verify_criterion())

        # lb = 0.2, threshold = 0.3 -> lb < threshold -> not verified
        d2 = ReLUDomain(lb=torch.tensor([0.2]), threshold=np.float64(0.3))
        self.assertFalse(d2.verify_criterion())


class TestReLUDomainDelNodeRecursive(unittest.TestCase):
    """Tests for ReLUDomain del_node with deep tree structures."""

    def test_del_node_deep_tree(self):
        """Test del_node on a deep tree."""
        from attack.domains import ReLUDomain

        # Create a 3-level tree
        root = ReLUDomain()
        left1 = ReLUDomain()
        right1 = ReLUDomain()
        left2 = ReLUDomain()
        right2 = ReLUDomain()

        root.left = left1
        root.right = right1
        left1.left = left2
        left1.right = right2

        root.del_node()

        self.assertFalse(root.valid)
        self.assertFalse(left1.valid)
        self.assertFalse(right1.valid)
        self.assertFalse(left2.valid)
        self.assertFalse(right2.valid)

    def test_del_node_partial_tree(self):
        """Test del_node on tree with only left children."""
        from attack.domains import ReLUDomain

        root = ReLUDomain()
        left1 = ReLUDomain()
        left2 = ReLUDomain()

        root.left = left1
        left1.left = left2

        root.del_node()

        self.assertFalse(root.valid)
        self.assertFalse(left1.valid)
        self.assertFalse(left2.valid)


class TestToCpuEmptyStructures(unittest.TestCase):
    """Tests for to_cpu with empty or None structures."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_cpu_empty_split_history(self):
        """Test to_cpu with empty split_history."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=[],  # Empty list
        )

        result = to_cpu(domain)
        self.assertEqual(result.split_history, [])

    def test_to_cpu_none_beta(self):
        """Test to_cpu with None beta."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=None,
        )

        result = to_cpu(domain)
        self.assertIsNone(result.beta)

    def test_to_cpu_none_c(self):
        """Test to_cpu with None c."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            c=None,
        )

        result = to_cpu(domain)
        self.assertIsNone(result.c)

    def test_to_cpu_none_intermediate_betas(self):
        """Test to_cpu with None intermediate_betas."""
        from attack.domains import ReLUDomain, to_cpu

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            intermediate_betas=None,
        )

        result = to_cpu(domain)
        self.assertIsNone(result.intermediate_betas)


class TestToDeviceEmptyStructures(unittest.TestCase):
    """Tests for to_device with empty or None structures."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False

    def test_to_device_empty_split_history(self):
        """Test to_device with empty split_history."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            split_history=[],
        )

        result = to_device(domain, 'cpu')
        self.assertEqual(result.split_history, [])

    def test_to_device_none_beta(self):
        """Test to_device with None beta."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            beta=None,
        )

        result = to_device(domain, 'cpu')
        self.assertIsNone(result.beta)

    def test_to_device_none_c(self):
        """Test to_device with None c."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            c=None,
        )

        result = to_device(domain, 'cpu')
        self.assertIsNone(result.c)

    def test_to_device_none_intermediate_betas(self):
        """Test to_device with None intermediate_betas."""
        from attack.domains import ReLUDomain, to_device

        domain = ReLUDomain(
            lA=[torch.randn(2, 3)],
            lb_all=[torch.randn(2, 5)],
            up_all=[torch.randn(2, 7)],
            alpha={'layer1': {'inter1': torch.randn(3, 4)}},
            intermediate_betas=None,
        )

        result = to_device(domain, 'cpu')
        self.assertIsNone(result.intermediate_betas)


class TestSortedReLUDomainListMultipleAlphas(unittest.TestCase):
    """Tests for SortedReLUDomainList with multiple alpha layers."""

    def setUp(self):
        """Setup for each test."""
        import arguments
        arguments.Config['bab']['cut']['enabled'] = False

    def test_init_with_multiple_alpha_layers(self):
        """Test initialization with multiple alpha layers."""
        from attack.domains import SortedReLUDomainList

        num = 2
        ret = {
            'lower_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'upper_bounds': [torch.randn(num, 10), torch.randn(num, 5)],
            'betas': None,
        }
        c = torch.randn(num, 1, 5)
        lAs = [torch.randn(num, 3, 10), torch.randn(num, 3, 5)]
        global_lbs = torch.randn(num, 5)
        global_ubs = torch.randn(num, 5)
        alphas = {
            'layer1': {'inter1': torch.randn(2, 3, num, 5), 'inter2': torch.randn(2, 3, num, 10)},
            'layer2': {'inter1': torch.randn(2, 4, num, 5)},
        }
        history = [[[], []]] * num
        thresholds = torch.zeros(num)

        domain_list = SortedReLUDomainList(
            ret, c, lAs, global_lbs, global_ubs, alphas, history, thresholds
        )

        self.assertEqual(len(domain_list), num)

        # Check alphas were properly split
        first_domain = domain_list[0]
        self.assertIn('layer1', first_domain.alpha)
        self.assertIn('layer2', first_domain.alpha)
        self.assertEqual(first_domain.alpha['layer1']['inter1'].shape[2], 1)


class TestReLUDomainLowerUpperAll(unittest.TestCase):
    """Tests for ReLUDomain lower_all and upper_all attributes."""

    def test_lower_upper_all_initialization(self):
        """Test lower_all and upper_all initialization."""
        from attack.domains import ReLUDomain

        lb_all = [torch.randn(1, 10), torch.randn(1, 5)]
        up_all = [torch.randn(1, 10), torch.randn(1, 5)]

        domain = ReLUDomain(lb_all=lb_all, up_all=up_all)

        self.assertEqual(len(domain.lower_all), 2)
        self.assertEqual(len(domain.upper_all), 2)
        self.assertEqual(domain.lower_all[0].shape, (1, 10))
        self.assertEqual(domain.upper_all[1].shape, (1, 5))

    def test_lower_upper_all_none(self):
        """Test lower_all and upper_all can be None."""
        from attack.domains import ReLUDomain

        domain = ReLUDomain()

        self.assertIsNone(domain.lower_all)
        self.assertIsNone(domain.upper_all)


class TestReLUDomainIntermediateBetas(unittest.TestCase):
    """Tests for ReLUDomain intermediate_betas attribute."""

    def test_intermediate_betas_initialization(self):
        """Test intermediate_betas initialization."""
        from attack.domains import ReLUDomain

        intermediate_betas = {
            'layer1': {
                'inter1': {'lb': torch.randn(3, 4), 'ub': torch.randn(3, 4)},
                'inter2': {'lb': torch.randn(3, 5), 'ub': torch.randn(3, 5)},
            }
        }

        domain = ReLUDomain(intermediate_betas=intermediate_betas)

        self.assertIn('layer1', domain.intermediate_betas)
        self.assertIn('inter1', domain.intermediate_betas['layer1'])
        self.assertIn('lb', domain.intermediate_betas['layer1']['inter1'])

    def test_intermediate_betas_none(self):
        """Test intermediate_betas can be None."""
        from attack.domains import ReLUDomain

        domain = ReLUDomain()

        self.assertIsNone(domain.intermediate_betas)


class TestReLUDomainSplitHistory(unittest.TestCase):
    """Tests for ReLUDomain split_history attribute."""

    def test_split_history_default_empty_list(self):
        """Test split_history defaults to empty list."""
        from attack.domains import ReLUDomain

        domain = ReLUDomain()

        self.assertEqual(domain.split_history, [])

    def test_split_history_with_data(self):
        """Test split_history with data."""
        from attack.domains import ReLUDomain

        split_history = {
            'layer': 0,
            'neuron': 5,
            'decision': 1,
        }

        domain = ReLUDomain(split_history=split_history)

        self.assertEqual(domain.split_history['layer'], 0)
        self.assertEqual(domain.split_history['neuron'], 5)
        self.assertEqual(domain.split_history['decision'], 1)


class TestReLUDomainlA(unittest.TestCase):
    """Tests for ReLUDomain lA attribute."""

    def test_lA_initialization(self):
        """Test lA initialization."""
        from attack.domains import ReLUDomain

        lA = [torch.randn(1, 3, 10), torch.randn(1, 3, 5)]

        domain = ReLUDomain(lA=lA)

        self.assertEqual(len(domain.lA), 2)
        self.assertEqual(domain.lA[0].shape, (1, 3, 10))
        self.assertEqual(domain.lA[1].shape, (1, 3, 5))

    def test_lA_none(self):
        """Test lA can be None."""
        from attack.domains import ReLUDomain

        domain = ReLUDomain()

        self.assertIsNone(domain.lA)


if __name__ == '__main__':
    unittest.main()
