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
"""Unit tests for heuristics/branching_heuristics.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

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
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestGetBranchingHeuristic(unittest.TestCase):
    """Tests for get_branching_heuristic function."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_get_heuristic_returns_object(self):
        """Test get_branching_heuristic returns a heuristic object."""
        import arguments
        from heuristics.branching_heuristics import get_branching_heuristic

        mock_net = self._create_mock_net()
        heuristic = get_branching_heuristic(mock_net)

        # Should return some heuristic object with a net attribute
        self.assertIsNotNone(heuristic)
        self.assertEqual(heuristic.net, mock_net)


class TestReadsSavedA(unittest.TestCase):
    """Only BBPS reads the A matrices of the split-node inputs from A_saved."""

    def _reads_saved_A(self, nonlinear_split, method='kfsb', disable=False,
                       nonlinear_method='bbps'):
        import arguments
        from heuristics.branching_heuristics import reads_saved_A

        branching = arguments.Config['bab']['branching']
        saved = (branching['method'], branching['nonlinear_split']['disable'],
                 branching['nonlinear_split']['method'])
        branching['method'] = method
        branching['nonlinear_split']['disable'] = disable
        branching['nonlinear_split']['method'] = nonlinear_method
        try:
            return reads_saved_A(MagicMock(nonlinear_split=nonlinear_split))
        finally:
            (branching['method'], branching['nonlinear_split']['disable'],
             branching['nonlinear_split']['method']) = saved

    def test_saved_A_requirement_follows_branching_selection(self):
        cases = (
            ({'nonlinear_split': True}, True),
            ({'nonlinear_split': False, 'method': 'nonlinear'}, True),
            ({'nonlinear_split': False}, False),
            ({'nonlinear_split': True, 'disable': True}, False),
            ({'nonlinear_split': True,
              'nonlinear_method': 'babsr-like'}, False),
        )
        for options, expected in cases:
            with self.subTest(**options):
                self.assertEqual(self._reads_saved_A(**options), expected)


class TestFsbBranching(unittest.TestCase):
    """Tests for FsbBranching class."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_fsb_branching_init(self):
        """Test FsbBranching initialization."""
        from heuristics.fsb import FsbBranching

        mock_net = self._create_mock_net()
        heuristic = FsbBranching(mock_net)

        self.assertEqual(heuristic.net, mock_net)

    def test_fsb_branching_inheritance(self):
        """Test FsbBranching inherits from NeuronBranchingHeuristic."""
        from heuristics.fsb import FsbBranching
        from heuristics.base import NeuronBranchingHeuristic

        self.assertTrue(issubclass(FsbBranching, NeuronBranchingHeuristic))


class TestBabsrBranching(unittest.TestCase):
    """Tests for BabsrBranching class."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_babsr_branching_init(self):
        """Test BabsrBranching initialization."""
        from heuristics.babsr import BabsrBranching

        mock_net = self._create_mock_net()
        heuristic = BabsrBranching(mock_net)

        self.assertEqual(heuristic.net, mock_net)

    def test_babsr_branching_inheritance(self):
        """Test BabsrBranching inherits from NeuronBranchingHeuristic."""
        from heuristics.babsr import BabsrBranching
        from heuristics.base import NeuronBranchingHeuristic

        self.assertTrue(issubclass(BabsrBranching, NeuronBranchingHeuristic))


class TestKfsbBranching(unittest.TestCase):
    """Tests for KfsbBranching class."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_kfsb_branching_init(self):
        """Test KfsbBranching initialization."""
        from heuristics.kfsb import KfsbBranching

        mock_net = self._create_mock_net()
        heuristic = KfsbBranching(mock_net)

        self.assertEqual(heuristic.net, mock_net)

    def test_kfsb_branching_inheritance(self):
        """Test KfsbBranching inherits from NeuronBranchingHeuristic."""
        from heuristics.kfsb import KfsbBranching
        from heuristics.base import NeuronBranchingHeuristic

        self.assertTrue(issubclass(KfsbBranching, NeuronBranchingHeuristic))


class TestComputeRatioFunction(unittest.TestCase):
    """Tests for compute_ratio function."""

    def test_compute_ratio_basic(self):
        """Test basic compute_ratio functionality."""
        from heuristics.utils import compute_ratio

        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        slope_ratio, intercept = compute_ratio(lower, upper)

        # For [-1, 1]:
        # lower_temp = -1 (clamped max 0)
        # upper_temp = relu(1) = 1
        # slope = 1 / (1 - (-1)) = 0.5
        # intercept = -1 * (-1) * 0.5 = 0.5
        self.assertAlmostEqual(slope_ratio.item(), 0.5, places=5)
        self.assertAlmostEqual(intercept.item(), 0.5, places=5)

    def test_compute_ratio_asymmetric(self):
        """Test compute_ratio with asymmetric bounds."""
        from heuristics.utils import compute_ratio

        lower = torch.tensor([-2.0])
        upper = torch.tensor([1.0])

        slope_ratio, intercept = compute_ratio(lower, upper)

        # lower_temp = -2
        # upper_temp = 1
        # slope = 1 / (1 - (-2)) = 1/3
        expected_slope = 1.0 / 3.0
        self.assertAlmostEqual(slope_ratio.item(), expected_slope, places=5)

    def test_compute_ratio_batch(self):
        """Test compute_ratio with batch input."""
        from heuristics.utils import compute_ratio

        lower = torch.tensor([-1.0, -2.0, -3.0])
        upper = torch.tensor([1.0, 1.0, 1.0])

        slope_ratio, intercept = compute_ratio(lower, upper)

        self.assertEqual(slope_ratio.shape, (3,))
        self.assertEqual(intercept.shape, (3,))

    def test_compute_ratio_positive_lower(self):
        """Test compute_ratio when lower bound is positive."""
        from heuristics.utils import compute_ratio

        lower = torch.tensor([0.5])
        upper = torch.tensor([1.0])

        slope_ratio, intercept = compute_ratio(lower, upper)

        # lower_temp = 0 (clamped max 0)
        # upper_temp = 1
        # slope = 1 / (1 - 0) = 1.0
        self.assertAlmostEqual(slope_ratio.item(), 1.0, places=5)

    def test_compute_ratio_negative_upper(self):
        """Test compute_ratio when upper bound is negative."""
        from heuristics.utils import compute_ratio

        lower = torch.tensor([-2.0])
        upper = torch.tensor([-1.0])

        slope_ratio, intercept = compute_ratio(lower, upper)

        # lower_temp = -2
        # upper_temp = 0 (relu)
        # slope = 0 / (0 - (-2)) = 0
        self.assertAlmostEqual(slope_ratio.item(), 0.0, places=5)


class TestRandomNeuronBranching(unittest.TestCase):
    """Tests for RandomNeuronBranching class."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_random_branching_init(self):
        """Test RandomNeuronBranching initialization."""
        from heuristics.base import RandomNeuronBranching

        mock_net = self._create_mock_net()
        heuristic = RandomNeuronBranching(mock_net)

        self.assertEqual(heuristic.net, mock_net)


class TestInterceptBranching(unittest.TestCase):
    """Tests for InterceptBranching class."""

    def _create_mock_net(self):
        """Create a mock network."""
        mock_net = MagicMock()
        mock_node = MagicMock()
        mock_node.name = 'layer1'
        mock_net.split_nodes = [mock_node]
        mock_net.split_activations = {'layer1': MagicMock()}
        return mock_net

    def test_intercept_branching_init(self):
        """Test InterceptBranching initialization."""
        from heuristics.base import InterceptBranching

        mock_net = self._create_mock_net()
        heuristic = InterceptBranching(mock_net)

        self.assertEqual(heuristic.net, mock_net)


if __name__ == '__main__':
    unittest.main()
