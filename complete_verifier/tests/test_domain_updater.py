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
"""Unit tests for domain_updater.py module."""

import os
import sys
import pytest
import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def setup_arguments():
    """Setup arguments.Config for testing."""
    import arguments
    original_config = arguments.Config
    try:
        new_config = arguments.ConfigHandler()
        new_config.construct_config_dict(new_config.default_args)
        new_config.file = None
        new_config['bab']['cut']['enabled'] = False
        new_config['bab']['cut']['biccos']['enabled'] = False
        new_config['bab']['cut']['biccos']['multi_tree_branching']['enabled'] = False
        arguments.Config = new_config
        yield
    finally:
        arguments.Config = original_config


# ============================================================================
# repeat Function Tests
# ============================================================================

class TestRepeatFunction:
    """Tests for the repeat function."""

    def test_repeat_none(self):
        """Test that repeat returns None for None input."""
        from domain_updater import repeat
        assert repeat(None, 3) is None

    def test_repeat_list(self):
        """Test repeat with list input."""
        from domain_updater import repeat
        lst = [1, 2, 3]
        result = repeat(lst, 2)
        assert result == [1, 2, 3, 1, 2, 3]

    def test_repeat_tensor_without_unsqueeze(self):
        """Test repeat tensor without unsqueeze."""
        from domain_updater import repeat
        t = torch.tensor([[1, 2], [3, 4]])
        result = repeat(t, 2, unsqueeze=False)
        assert result.shape == (4, 2)
        assert torch.equal(result[:2], t)
        assert torch.equal(result[2:], t)

    def test_repeat_tensor_with_unsqueeze(self):
        """Test repeat tensor with unsqueeze."""
        from domain_updater import repeat
        t = torch.tensor([[1, 2], [3, 4]])
        result = repeat(t, 2, unsqueeze=True)
        assert result.shape == (2, 2, 2)

    def test_repeat_1d_tensor(self):
        """Test repeat with 1D tensor."""
        from domain_updater import repeat
        t = torch.tensor([1, 2, 3])
        result = repeat(t, 3, unsqueeze=False)
        assert result.shape == (9,)

    def test_repeat_empty_list(self):
        """Test repeat with empty list."""
        from domain_updater import repeat
        result = repeat([], 3)
        assert result == []


# ============================================================================
# DomainUpdater Class Tests
# ============================================================================

class TestDomainUpdater:
    """Tests for the DomainUpdater class."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_domain_updater_init(self):
        """Test DomainUpdater initialization."""
        from domain_updater import DomainUpdater

        root = 'root'
        final_name = 'final'
        split_nodes = [self._create_mock_split_node('layer1')]

        updater = DomainUpdater(final_name, [n.name for n in split_nodes])

        assert updater.final_name == final_name
        assert updater.split_nodes_names == [n.name for n in split_nodes]
        assert updater.device == 'cpu'
        assert updater.node_names == []
        assert updater.cut_usage is False

    def test_domain_updater_with_cut_enabled(self):
        """Test DomainUpdater with cut enabled."""
        import arguments
        from domain_updater import DomainUpdater

        arguments.Config['bab']['cut']['enabled'] = True

        root = 'root'
        final_name = 'final'
        split_nodes = []

        updater = DomainUpdater(final_name, [n.name for n in split_nodes])

        assert updater.cut_usage is True

    def test_get_num_domain_and_split(self):
        """Test get_num_domain_and_split static method."""
        from domain_updater import DomainUpdater

        d = {
            'lower_bounds': {'final': torch.zeros(4, 10)},
        }
        split = {
            'decision': [(0, 1), (0, 2), (0, 3), (0, 4),
                        (0, 5), (0, 6), (0, 7), (0, 8)],
        }

        num_domain, num_split = DomainUpdater.get_num_domain_and_split(
            d, split, 'final')

        assert num_domain == 4
        assert num_split == 2

    def test_as_tensor(self):
        """Test as_tensor method."""
        from domain_updater import DomainUpdater

        updater = DomainUpdater('final', [])

        lst = [1, 2, 3]
        result = updater.as_tensor(lst)

        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cpu'

    def test_empty_dict(self):
        """Test empty_dict method."""
        from domain_updater import DomainUpdater

        updater = DomainUpdater('final', [])
        updater.node_names = ['layer1', 'layer2']

        result = updater.empty_dict()

        assert result == {'layer1': [], 'layer2': []}


# ============================================================================
# DomainUpdaterSimple Class Tests
# ============================================================================

class TestDomainUpdaterSimple:
    """Tests for the DomainUpdaterSimple class."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_domain_updater_simple_init(self):
        """Test DomainUpdaterSimple initialization."""
        from domain_updater import DomainUpdaterSimple

        root = 'root'
        final_name = 'final'
        split_nodes = [self._create_mock_split_node('layer1')]

        updater = DomainUpdaterSimple(final_name, [n.name for n in split_nodes])

        assert updater.final_name == final_name
        assert updater.split_nodes_names == [n.name for n in split_nodes]

    def test_domain_updater_simple_inherits(self):
        """Test that DomainUpdaterSimple inherits from DomainUpdater."""
        from domain_updater import DomainUpdater, DomainUpdaterSimple

        assert issubclass(DomainUpdaterSimple, DomainUpdater)


# ============================================================================
# Integration Tests
# ============================================================================

class TestDomainUpdaterSetBranchedBounds:
    """Tests for DomainUpdater.set_branched_bounds method."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_set_branched_bounds_basic(self):
        """Test basic set_branched_bounds functionality."""
        from domain_updater import DomainUpdater

        root = 'root'
        final_name = 'final'
        split_nodes = [self._create_mock_split_node('layer1')]

        updater = DomainUpdater(final_name, [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 10),
                'layer1': torch.zeros(2, 20)
            },
            'upper_bounds': {
                'final': torch.ones(2, 10),
                'layer1': torch.ones(2, 20)
            },
        }
        split = {
            'decision': [(0, 1), (0, 2), (0, 3), (0, 4)],
        }

        updater.set_branched_bounds(d, split)

        assert updater.num_domain == 2
        assert updater.num_split == 2
        assert updater.num_copy == 4  # 2^2 for depth mode

    def test_set_branched_bounds_breadth_mode(self):
        """Test set_branched_bounds in breadth mode - basic attributes."""
        from domain_updater import DomainUpdater

        # Create mock split nodes matching decision indices
        split_nodes = [self._create_mock_split_node('layer0')]

        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(1, 5),
                'layer0': torch.zeros(1, 10),
            },
            'upper_bounds': {
                'final': torch.ones(1, 5),
                'layer0': torch.ones(1, 10),
            },
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # node_idx 0, neuron_idx 1 and 2
        }

        updater.set_branched_bounds(d, split, mode='breadth')

        assert updater.num_domain == 1
        assert updater.num_split == 2
        assert updater.num_copy == 4  # 2 * 2 for breadth mode


class TestDomainUpdaterWithHistory:
    """Tests for DomainUpdater with history."""

    def test_empty_dict_method(self):
        """Test empty_dict method."""
        from domain_updater import DomainUpdater

        updater = DomainUpdater('final', [])
        updater.node_names = ['layer1', 'layer2', 'layer3']

        result = updater.empty_dict()

        assert len(result) == 3
        assert 'layer1' in result
        assert 'layer2' in result
        assert 'layer3' in result
        assert result['layer1'] == []


class TestRepeatEdgeCases:
    """Additional edge case tests for repeat function."""

    def test_repeat_3d_tensor(self):
        """Test repeat with 3D tensor."""
        from domain_updater import repeat
        t = torch.randn(2, 3, 4)
        result = repeat(t, 3, unsqueeze=False)
        assert result.shape == (6, 3, 4)

    def test_repeat_4d_tensor(self):
        """Test repeat with 4D tensor."""
        from domain_updater import repeat
        t = torch.randn(2, 3, 4, 5)
        result = repeat(t, 2, unsqueeze=False)
        assert result.shape == (4, 3, 4, 5)

    def test_repeat_with_unsqueeze_3d(self):
        """Test repeat with unsqueeze on 3D tensor."""
        from domain_updater import repeat
        t = torch.randn(2, 3, 4)
        result = repeat(t, 2, unsqueeze=True)
        assert result.shape == (2, 2, 3, 4)


# ============================================================================
# DomainUpdater with History Tests
# ============================================================================

class TestDomainUpdaterWithHistoryProcessing:
    """Tests for DomainUpdater with history processing."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def _create_empty_history_entry(self, node_name):
        """Create an empty history entry for a node."""
        # History format: (loc, sign, bias, score, depth)
        # where each can be a list or tensor
        return {node_name: ([], [], [], None, None)}

    def test_set_branched_bounds_with_history(self):
        """Test set_branched_bounds with history present."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Create history with tensor entries for the node (proper format)
        # History format: (loc, sign, bias, score, depth)
        history = [
            {'layer0': (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )},
            {'layer0': (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )},
        ]

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'history': history,
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # 2 domains, 1 split each
        }

        updater.set_branched_bounds(d, split)

        # Check history was created
        assert d['history'] is not None
        assert len(d['history']) == 4  # 2 domains * 2 copies

    def test_set_branched_bounds_with_depths(self):
        """Test set_branched_bounds with depths field."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'depths': [1, 2],  # Initial depths for 2 domains
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # 2 domains, 1 split each
        }

        updater.set_branched_bounds(d, split, mode='depth')

        # Check depths were updated: original + num_split for depth mode
        assert 'depths' in d
        assert len(d['depths']) == 4  # 2 domains * 2 copies
        # Each depth should be incremented by num_split (1)
        assert d['depths'][0] == 2  # 1 + 1
        assert d['depths'][1] == 3  # 2 + 1

    def test_set_branched_bounds_with_depths_breadth_mode(self):
        """Test set_branched_bounds with depths in breadth mode."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(1, 5),
                'layer0': torch.zeros(1, 10),
            },
            'upper_bounds': {
                'final': torch.ones(1, 5),
                'layer0': torch.ones(1, 10),
            },
            'depths': [5],  # Initial depth
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # 1 domain, 2 splits
        }

        updater.set_branched_bounds(d, split, mode='breadth')

        # In breadth mode, depth should be incremented by 1
        assert d['depths'][0] == 6  # 5 + 1

    def test_set_branched_bounds_with_alphas(self):
        """Test set_branched_bounds with alphas field."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Create alphas structure
        alphas = {
            'layer0': {
                'sub_layer': torch.randn(3, 4, 2, 5),  # dim 2 is batch
            }
        }

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'alphas': alphas,
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # 2 domains, 1 split each
        }

        updater.set_branched_bounds(d, split)

        # Check alphas were repeated
        assert 'alphas' in d
        # Batch dimension (dim 2) should be doubled: 2 * num_copy
        assert d['alphas']['layer0']['sub_layer'].shape[2] == 4  # 2 * 2

    def test_set_branched_bounds_with_lAs(self):
        """Test set_branched_bounds with lAs field."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'lAs': {
                'layer0': torch.randn(2, 10, 5),
            },
        }
        split = {
            'decision': [(0, 1), (0, 2)],  # 2 domains, 1 split each
        }

        updater.set_branched_bounds(d, split)

        # Check lAs were repeated
        assert 'lAs' in d
        assert d['lAs']['layer0'].shape[0] == 4  # 2 * 2

    def test_set_branched_bounds_with_split_history(self):
        """Test set_branched_bounds with split_history field."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'split_history': [[], []],  # list type
        }
        split = {
            'decision': [(0, 1), (0, 2)],
        }

        updater.set_branched_bounds(d, split)

        # split_history should be repeated
        assert len(d['split_history']) == 4

    def test_set_branched_bounds_with_thresholds(self):
        """Test set_branched_bounds with thresholds tensor."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'thresholds': torch.tensor([0.5, 0.6]),
        }
        split = {
            'decision': [(0, 1), (0, 2)],
        }

        updater.set_branched_bounds(d, split)

        # thresholds should be repeated
        assert d['thresholds'].shape[0] == 4

    def test_set_branched_bounds_updates_split_list(self):
        """Test that split dict lists are updated."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
        }
        split = {
            'decision': [(0, 1), (0, 2)],
            'custom_list': ['a', 'b'],  # This should be repeated
        }

        updater.set_branched_bounds(d, split)

        # custom_list should be repeated
        assert len(split['custom_list']) == 4

    def test_set_branched_bounds_updates_split_tensor(self):
        """Test that split dict tensors are updated."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
        }
        split = {
            'decision': [(0, 1), (0, 2)],
            'scores': torch.tensor([1.0, 2.0]),  # Tensor should be repeated
        }

        updater.set_branched_bounds(d, split)

        # scores tensor should be repeated
        assert split['scores'].shape[0] == 4


# ============================================================================
# DomainUpdater Multi-Branch Tests
# ============================================================================

class TestDomainUpdaterMultiBranch:
    """Tests for DomainUpdater with multi-branch splits."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_multi_branch_with_2d_points(self):
        """Test multi-branch splitting with 2D points tensor."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
        }
        # 2D points: (num_decisions, num_branch_points)
        # e.g., 2 branch points means 3 branches
        split = {
            'decision': [(0, 1), (0, 2)],
            'points': torch.tensor([[0.3, 0.7], [0.4, 0.6]]),  # 2 decisions, 2 branch points each
        }

        updater.set_branched_bounds(d, split)

        assert updater.multi_branch is True
        assert updater.num_branches == 3  # points.shape[1] + 1 = 2 + 1

    def test_single_branch_with_1d_points(self):
        """Test single branch splitting with 1D points tensor."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
        }
        split = {
            'decision': [(0, 1), (0, 2)],
            'points': torch.tensor([0.5, 0.5]),  # 1D points - standard binary split
        }

        updater.set_branched_bounds(d, split)

        assert updater.multi_branch is False
        assert updater.num_branches == 2


# ============================================================================
# DomainUpdaterSimple Tests
# ============================================================================

class TestDomainUpdaterSimpleSetHistoryAndBounds:
    """Tests for DomainUpdaterSimple._set_history_and_bounds method."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_simple_updater_basic_bounds_update(self):
        """Test DomainUpdaterSimple basic bounds update."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.full((2, 10), -1.0),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.full((2, 10), 1.0),
            },
        }
        split = {
            'decision': [(0, 3), (0, 5)],  # 2 domains, split neurons 3 and 5
        }

        updater.set_branched_bounds(d, split)

        # After branching, domain should be doubled
        assert d['lower_bounds']['layer0'].shape[0] == 4
        # The split neuron should have bounds updated to 0
        # First copy: upper bound at split point = 0
        # Second copy: lower bound at split point = 0

    def test_simple_updater_with_branching_points(self):
        """Test DomainUpdaterSimple with custom branching points."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.full((2, 10), -1.0),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.full((2, 10), 1.0),
            },
        }
        split = {
            'decision': [(0, 3), (0, 5)],
            'points': torch.tensor([0.25, -0.25]),  # Custom split points
        }

        updater.set_branched_bounds(d, split)

        # Check shape
        assert d['lower_bounds']['layer0'].shape[0] == 4

    def test_simple_updater_with_history(self):
        """Test DomainUpdaterSimple with history tracking."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        # Create history with empty tensor entries (proper format)
        history = [
            {'layer0': (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )},
            {'layer0': (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )},
        ]

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.full((2, 10), -1.0),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.full((2, 10), 1.0),
            },
            'history': history,
        }
        split = {
            'decision': [(0, 3), (0, 5)],
        }

        updater.set_branched_bounds(d, split)

        # History should be updated
        assert d['history'] is not None
        assert len(d['history']) == 4

    def test_simple_updater_with_history_and_points(self):
        """Test DomainUpdaterSimple with history and custom points."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        # Create history with empty tensor entries (proper format)
        history = [
            {'layer0': (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )},
        ]

        d = {
            'lower_bounds': {
                'final': torch.zeros(1, 5),
                'layer0': torch.full((1, 10), -1.0),
            },
            'upper_bounds': {
                'final': torch.ones(1, 5),
                'layer0': torch.full((1, 10), 1.0),
            },
            'history': history,
        }
        split = {
            'decision': [(0, 3)],
            'points': torch.tensor([0.5]),
        }

        updater.set_branched_bounds(d, split)

        # History should be updated with the split point
        assert d['history'] is not None
        assert len(d['history']) == 2


# ============================================================================
# DomainUpdaterSimple _append_history Tests
# ============================================================================

class TestDomainUpdaterSimpleAppendHistory:
    """Tests for DomainUpdaterSimple._append_history method."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_append_history_empty_initial(self):
        """Test _append_history with empty initial history."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        # Initialize new_history with empty entry
        updater.new_history = [{'layer0': ([], [], [], None, None)}]

        updater._append_history(0, 'layer0', 5, 1, 0.5)

        # Check that history was appended
        loc, sign, bias, score, depth = updater.new_history[0]['layer0']
        assert loc[0] == 5
        assert sign[0] == 1
        assert bias[0] == 0.5

    def test_append_history_existing_entries(self):
        """Test _append_history with existing history entries."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        # Initialize new_history with existing entries
        updater.new_history = [{
            'layer0': (
                torch.tensor([1, 2]),
                torch.tensor([1.0, -1.0]),
                torch.tensor([0.0, 0.0]),
                None,
                None
            )
        }]

        updater._append_history(0, 'layer0', 5, 1, 0.5)

        # Check that history was appended
        loc, sign, bias, score, depth = updater.new_history[0]['layer0']
        assert len(loc) == 3
        assert loc[-1] == 5
        assert sign[-1] == 1
        assert bias[-1] == 0.5

    def test_append_history_none_points(self):
        """Test _append_history with None points (defaults to 0)."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        updater.new_history = [{'layer0': ([], [], [], None, None)}]

        updater._append_history(0, 'layer0', 5, -1, None)

        loc, sign, bias, score, depth = updater.new_history[0]['layer0']
        assert loc[0] == 5
        assert sign[0] == -1
        # bias should be 0 when points is None
        assert bias[0] == 0.0

    def test_append_history_none_new_history(self):
        """Test _append_history when new_history entry is None."""
        from domain_updater import DomainUpdaterSimple

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        updater.new_history = [None]

        # Should return early without error
        updater._append_history(0, 'layer0', 5, 1, 0.5)

        # new_history should still be None
        assert updater.new_history[0] is None


# ============================================================================
# DomainUpdater with BICCOS Tests
# ============================================================================

class TestDomainUpdaterWithBiccos:
    """Tests for DomainUpdater with BICCOS enabled."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_simple_updater_with_biccos(self):
        """Test DomainUpdaterSimple with BICCOS enabled."""
        import arguments
        from domain_updater import DomainUpdaterSimple

        # Enable BICCOS
        arguments.Config['bab']['cut']['enabled'] = True
        arguments.Config['bab']['cut']['biccos']['enabled'] = True

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        assert updater.biccos_usage is True

        # Initialize new_history with empty entry
        updater.new_history = [{'layer0': ([], [], [], None, None)}]

        updater._append_history(0, 'layer0', 5, 1, 0.5)

        # Check that score was added
        loc, sign, bias, score, depth = updater.new_history[0]['layer0']
        assert score is not None
        assert score[0] == 0.0

    def test_simple_updater_with_multi_tree(self):
        """Test DomainUpdaterSimple with multi-tree searching enabled."""
        import arguments
        from domain_updater import DomainUpdaterSimple

        # Enable multi-tree searching
        arguments.Config['bab']['cut']['enabled'] = True
        arguments.Config['bab']['cut']['biccos']['enabled'] = True
        arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled'] = True

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdaterSimple('final', [n.name for n in split_nodes])

        assert updater.multi_tree_searching is True

        # Initialize new_history with empty tensor entries (proper format)
        updater.new_history = [{'layer0': (
            torch.tensor([], dtype=torch.long),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([])
        )}]

        updater._append_history(0, 'layer0', 5, 1, 0.5)

        # Check that depth was added
        loc, sign, bias, score, depth = updater.new_history[0]['layer0']
        assert depth is not None
        # When depth tensor is empty, max_depth defaults to max(-1, 0) = 0, then depth = 0 + 1 = 1
        assert depth[0] == 1


# ============================================================================
# DomainUpdater Bound Update Tests
# ============================================================================

class TestDomainUpdaterBoundUpdates:
    """Tests for DomainUpdater bound update functionality."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_bound_updates_applied_correctly(self):
        """Test that bounds are correctly updated after split."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Create domain with known bounds
        d = {
            'lower_bounds': {
                'final': torch.zeros(1, 5),
                'layer0': torch.full((1, 10), -1.0),  # All -1
            },
            'upper_bounds': {
                'final': torch.ones(1, 5),
                'layer0': torch.full((1, 10), 1.0),   # All 1
            },
        }
        split = {
            'decision': [(0, 3)],  # Split neuron 3 in layer0
        }

        updater.set_branched_bounds(d, split)

        # After split, we should have 2 copies (batch size 2)
        assert d['lower_bounds']['layer0'].shape[0] == 2
        assert d['upper_bounds']['layer0'].shape[0] == 2

        # First copy: upper bound at index 3 should be 0 (neuron forced <= 0, i.e., inactive)
        # Second copy: lower bound at index 3 should be 0 (neuron forced >= 0, i.e., active)
        assert d['upper_bounds']['layer0'][0, 3].item() == 0.0
        assert d['lower_bounds']['layer0'][1, 3].item() == 0.0

        # Other bounds should remain unchanged
        assert d['lower_bounds']['layer0'][0, 3].item() == -1.0  # First copy lower unchanged
        assert d['upper_bounds']['layer0'][1, 3].item() == 1.0   # Second copy upper unchanged

    def test_bound_updates_with_custom_points(self):
        """Test bound updates with custom split points."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        d = {
            'lower_bounds': {
                'final': torch.zeros(1, 5),
                'layer0': torch.full((1, 10), -1.0),
            },
            'upper_bounds': {
                'final': torch.ones(1, 5),
                'layer0': torch.full((1, 10), 1.0),
            },
        }
        split = {
            'decision': [(0, 3)],
            'points': torch.tensor([0.25]),  # Split at 0.25 instead of 0
        }

        updater.set_branched_bounds(d, split)

        # Bounds should be updated with the custom split point
        assert d['lower_bounds']['layer0'].shape[0] == 2


# ============================================================================
# DomainUpdater with x_Ls and x_Us Tests
# ============================================================================

class TestDomainUpdaterWithInputBounds:
    """Tests for DomainUpdater with input bounds (x_Ls, x_Us)."""

    def _create_mock_split_node(self, name):
        """Create a mock split node."""
        class MockNode:
            def __init__(self, name):
                self.name = name
        return MockNode(name)

    def test_set_branched_bounds_with_x_bounds(self):
        """Test set_branched_bounds with x_Ls and x_Us."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Use distinct values so we can verify repetition
        x_Ls_original = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        x_Us_original = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'x_Ls': x_Ls_original.clone(),
            'x_Us': x_Us_original.clone(),
        }
        split = {
            'decision': [(0, 1), (0, 2)],
        }

        updater.set_branched_bounds(d, split)

        # x_Ls and x_Us should be repeated (2 domains * 2 copies = 4)
        assert d['x_Ls'].shape[0] == 4
        assert d['x_Us'].shape[0] == 4

        # Verify values are actually repeated: the entire tensor is repeated
        # Pattern is: [original[0], original[1], original[0], original[1]]
        assert torch.allclose(d['x_Ls'][0], x_Ls_original[0])
        assert torch.allclose(d['x_Ls'][1], x_Ls_original[1])
        assert torch.allclose(d['x_Ls'][2], x_Ls_original[0])
        assert torch.allclose(d['x_Ls'][3], x_Ls_original[1])

        assert torch.allclose(d['x_Us'][0], x_Us_original[0])
        assert torch.allclose(d['x_Us'][1], x_Us_original[1])
        assert torch.allclose(d['x_Us'][2], x_Us_original[0])
        assert torch.allclose(d['x_Us'][3], x_Us_original[1])

    def test_set_branched_bounds_with_cs(self):
        """Test set_branched_bounds with cs (specification matrix)."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Use distinct values so we can verify repetition
        cs_original = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]],
                                    [[6.0, 7.0, 8.0, 9.0, 10.0]]])

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'cs': cs_original.clone(),
        }
        split = {
            'decision': [(0, 1), (0, 2)],
        }

        updater.set_branched_bounds(d, split)

        # cs should be repeated (2 domains * 2 copies = 4)
        assert d['cs'].shape[0] == 4

        # Verify values are actually repeated: the entire tensor is repeated
        # Pattern is: [original[0], original[1], original[0], original[1]]
        assert torch.allclose(d['cs'][0], cs_original[0])
        assert torch.allclose(d['cs'][1], cs_original[1])
        assert torch.allclose(d['cs'][2], cs_original[0])
        assert torch.allclose(d['cs'][3], cs_original[1])

    def test_set_branched_bounds_with_betas(self):
        """Test set_branched_bounds with betas."""
        from domain_updater import DomainUpdater

        split_nodes = [self._create_mock_split_node('layer0')]
        updater = DomainUpdater('final', [n.name for n in split_nodes])

        # Use distinct values so we can verify repetition
        betas_original = [['beta_a'], ['beta_b']]

        d = {
            'lower_bounds': {
                'final': torch.zeros(2, 5),
                'layer0': torch.zeros(2, 10),
            },
            'upper_bounds': {
                'final': torch.ones(2, 5),
                'layer0': torch.ones(2, 10),
            },
            'betas': [lst.copy() for lst in betas_original],
        }
        split = {
            'decision': [(0, 1), (0, 2)],
        }

        updater.set_branched_bounds(d, split)

        # betas should be repeated (2 domains * 2 copies = 4)
        assert len(d['betas']) == 4

        # Verify values are actually repeated: the entire list is repeated
        # Pattern is: [original[0], original[1], original[0], original[1]]
        assert d['betas'][0] == betas_original[0]
        assert d['betas'][1] == betas_original[1]
        assert d['betas'][2] == betas_original[0]
        assert d['betas'][3] == betas_original[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
