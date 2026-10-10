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
"""Unit tests for branching_domains.py module."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from collections import deque


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def setup_arguments():
    """Setup arguments.Config for each test."""
    import arguments

    # Store original config - this is a module-level global
    original_config = arguments.Config

    try:
        # Create a fresh ConfigHandler and initialize it with defaults
        new_config = arguments.ConfigHandler()
        # Call construct_config_dict with default_args to populate all_args
        new_config.construct_config_dict(new_config.default_args)
        new_config.file = None

        # Override with test-specific values
        new_config['bab']['get_upper_bound'] = False
        new_config['bab']['cut']['enabled'] = False
        new_config['bab']['cut']['cplex_cuts'] = False
        new_config['bab']['cut']['cplex_cuts_revpickup'] = False

        arguments.Config = new_config

        yield
    finally:
        # Always restore original Config, even if test fails
        arguments.Config = original_config


@pytest.fixture
def mock_tensor_storage():
    """Create a mock TensorStorage that acts like a list with tensor operations."""
    return MockTensorStorage


class MockTensorStorage:
    """Mock TensorStorage class for testing. Supports all operations used in tests."""

    def __init__(self, tensor, dtype=None, device=None, concat_dim=0):
        self.data = tensor
        self.concat_dim = concat_dim
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(tensor, 'dtype'):
            self.dtype = tensor.dtype
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(tensor, 'device'):
            self.device = tensor.device

    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def __sub__(self, other):
        if isinstance(other, MockTensorStorage):
            return self.data - other.data
        return self.data - other

    def append(self, tensor):
        if hasattr(tensor, 'to') and hasattr(self, 'device'):
            tensor = tensor.to(self.device)
        self.data = torch.cat([self.data, tensor], dim=self.concat_dim)

    def pop(self, n):
        result = self.data[-n:]
        self.data = self.data[:-n]
        return result

    def reorder(self, n, indices, reorder_dim=0):
        self.data = self.data.index_select(reorder_dim, indices)

    def cpu(self):
        return self.data.cpu()

    def items(self):
        """Return empty items for dict-like behavior."""
        return []


@pytest.fixture
def mock_net():
    """Create a mock network with required attributes."""
    net = MagicMock()
    net.relus = []
    net.final_name = 'output'
    net.device = 'cpu'
    net.split_activations = {}
    net.net = {}
    return net


@pytest.fixture
def create_batched_domain_list(mock_net):
    """Factory fixture to create BatchedDomainList instances with proper initialization."""
    from branching_domains import BatchedDomainList
    import arguments

    def _create(num_domains=2, output_dim=10, tree_traversal='depth_first',
                with_alphas=False, with_input_bounds=False):
        """
        Create a properly initialized BatchedDomainList.

        Args:
            num_domains: Number of domains to create
            output_dim: Dimension of output layer
            tree_traversal: 'depth_first' or 'breadth_first'
            with_alphas: Whether to include alpha values
            with_input_bounds: Whether to include input bounds (for input splitting)
        """
        arguments.Config['bab']['tree_traversal'] = tree_traversal
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(num_domains, output_dim)},
                'upper_bounds': {'output': torch.ones(num_domains, output_dim)},
                'betas': None
            }

            if with_input_bounds:
                ret['input_split_idx'] = torch.arange(num_domains).unsqueeze(1)

            c = torch.ones(num_domains, 1, output_dim)
            lAs = {'layer1': torch.randn(num_domains, output_dim, 5)}
            global_lbs = torch.linspace(0.0, 0.5, num_domains).unsqueeze(1)
            global_ubs = torch.linspace(1.0, 0.6, num_domains).unsqueeze(1)

            alphas = {}
            if with_alphas:
                alphas = {'layer1': {'spec': torch.randn(2, 5, num_domains)}}

            history = {}
            thresholds = torch.tensor([[0.0]])

            mock_net.final_name = 'output'

            x = None
            if with_input_bounds:
                x = MagicMock()
                x.ptb = MagicMock()
                x.ptb.x_L = torch.zeros(num_domains, 5)
                x.ptb.x_U = torch.ones(num_domains, 5)

            domain_list = BatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=with_input_bounds,
                x=x,
                enable_clip_domains=False
            )

            return domain_list

    return _create


@pytest.fixture
def create_shallow_first_domain_list(mock_net):
    """Factory fixture to create ShallowFirstBatchedDomainList instances."""
    from branching_domains import ShallowFirstBatchedDomainList
    import arguments

    def _create(num_domains=2, output_dim=10, tree_traversal='depth_first',
                with_alphas=False, with_input_bounds=False):
        """Create a properly initialized ShallowFirstBatchedDomainList."""
        arguments.Config['bab']['tree_traversal'] = tree_traversal
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(num_domains, output_dim)},
                'upper_bounds': {'output': torch.ones(num_domains, output_dim)},
                'betas': None
            }

            if with_input_bounds:
                ret['input_split_idx'] = torch.arange(num_domains).unsqueeze(1)

            c = torch.ones(num_domains, 1, output_dim)
            lAs = {'layer1': torch.randn(num_domains, output_dim, 5)}
            global_lbs = torch.linspace(0.0, 0.5, num_domains).unsqueeze(1)
            global_ubs = torch.linspace(1.0, 0.6, num_domains).unsqueeze(1)

            alphas = {}
            if with_alphas:
                alphas = {'layer1': {'spec': torch.randn(2, 5, num_domains)}}

            history = {}
            thresholds = torch.tensor([[0.0]])

            mock_net.final_name = 'output'

            x = None
            if with_input_bounds:
                x = MagicMock()
                x.ptb = MagicMock()
                x.ptb.x_L = torch.zeros(num_domains, 5)
                x.ptb.x_U = torch.ones(num_domains, 5)

            domain_list = ShallowFirstBatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=with_input_bounds,
                x=x
            )

            return domain_list

    return _create


# ============================================================================
# AbstractDomainList print tests
# ============================================================================


class TestAbstractDomainListPrint:
    """Tests for the AbstractDomainList.print() method."""

    def test_print_with_empty_list(self, capsys):
        """Test print when list is empty."""
        from branching_domains import AbstractDomainList

        class TestDomainList(AbstractDomainList):
            def __len__(self):
                return 0

        domain_list = TestDomainList()
        domain_list.print()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_with_domains_no_upper_bound(self, capsys):
        """Test print with domains when get_upper_bound is False."""
        import arguments
        from branching_domains import AbstractDomainList

        arguments.Config['bab']['get_upper_bound'] = False

        class TestDomainList(AbstractDomainList):
            def __len__(self):
                return 1

            def get_min_domain(self, n, rev_order=False):
                domain = SimpleNamespace()
                domain.lower_bound = torch.tensor([0.5])
                domain.upper_bound = torch.tensor([1.0])
                domain.threshold = torch.tensor([0.0])
                domain.depth = 2
                return [domain]

        domain_list = TestDomainList()
        domain_list.print()
        captured = capsys.readouterr()
        assert 'Current worst splitting domains lb-rhs (depth):' in captured.out
        assert '(2)' in captured.out

    def test_print_with_domains_with_upper_bound(self, capsys):
        """Test print with domains when get_upper_bound is True."""
        import arguments
        from branching_domains import AbstractDomainList

        arguments.Config['bab']['get_upper_bound'] = True

        class TestDomainList(AbstractDomainList):
            def __len__(self):
                return 1

            def get_min_domain(self, n, rev_order=False):
                domain = SimpleNamespace()
                domain.lower_bound = torch.tensor([0.5])
                domain.upper_bound = torch.tensor([1.0])
                domain.threshold = torch.tensor([0.0])
                domain.depth = 3
                return [domain]

        domain_list = TestDomainList()
        domain_list.print()
        captured = capsys.readouterr()
        assert 'Current worst splitting domains [lb, ub] (depth):' in captured.out


# ============================================================================
# check_worst_domain Tests
# ============================================================================

class TestCheckWorstDomain:
    """Tests for the check_worst_domain function."""

    def test_empty_domain_list(self):
        """Test with empty domain list."""
        from branching_domains import check_worst_domain

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=0)

        result = check_worst_domain(mock_domain_list)
        assert torch.isclose(result, torch.tensor(1e-7))

    def test_non_empty_domain_list_no_cuts(self):
        """Test with non-empty domain list without cuts enabled."""
        import arguments
        from branching_domains import check_worst_domain

        arguments.Config['bab']['cut']['enabled'] = False

        domain = SimpleNamespace()
        domain.lower_bound = torch.tensor([0.5])
        domain.threshold = torch.tensor([0.1])

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=1)
        mock_domain_list.get_min_domain = MagicMock(return_value=[domain])

        result = check_worst_domain(mock_domain_list)
        expected = torch.tensor([0.4])  # 0.5 - 0.1
        assert torch.allclose(result, expected)

    def test_non_empty_domain_list_with_cuts(self):
        """Test with non-empty domain list with cuts enabled."""
        import arguments
        from branching_domains import check_worst_domain

        arguments.Config['bab']['cut']['enabled'] = True
        arguments.Config['bab']['cut']['cplex_cuts'] = True
        arguments.Config['bab']['cut']['cplex_cuts_revpickup'] = True

        domain = SimpleNamespace()
        domain.lower_bound = torch.tensor([0.8])
        domain.threshold = torch.tensor([0.2])

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=2)
        mock_domain_list.get_min_domain = MagicMock(return_value=[None, domain])

        result = check_worst_domain(mock_domain_list)
        expected = torch.tensor([0.6])  # 0.8 - 0.2
        assert torch.allclose(result, expected)

    def test_get_min_domain_called_correctly(self):
        """Test that get_min_domain is called with correct arguments."""
        import arguments
        from branching_domains import check_worst_domain

        arguments.Config['bab']['cut']['enabled'] = False

        domain = SimpleNamespace()
        domain.lower_bound = torch.tensor([0.0])
        domain.threshold = torch.tensor([0.0])

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=1)
        mock_domain_list.get_min_domain = MagicMock(return_value=[domain])

        check_worst_domain(mock_domain_list)

        mock_domain_list.get_min_domain.assert_called_once_with(1, rev_order=False)


# ============================================================================
# ShallowFirstBatchedDomainList.BaBTree Tests
# ============================================================================

class TestBaBTree:
    """Tests for the BaBTree inner class."""

    def test_babtree_init(self):
        """Test BaBTree initialization."""
        from branching_domains import ShallowFirstBatchedDomainList

        tree = ShallowFirstBatchedDomainList.BaBTree(
            layer_name='layer1',
            node=5,
            score=[0.1, 0.2]
        )

        assert tree.layer_name == 'layer1'
        assert tree.node == 5
        assert tree.score == [0.1, 0.2]
        assert tree.pos_child is None
        assert tree.neg_child is None

    def test_babtree_str_leaf(self):
        """Test BaBTree string representation for leaf node."""
        from branching_domains import ShallowFirstBatchedDomainList

        tree = ShallowFirstBatchedDomainList.BaBTree(
            layer_name='layer1',
            node=5,
            score=[0.1, 0.2]
        )

        str_repr = str(tree)
        assert 'layer1' in str_repr
        assert '5' in str_repr

    def test_babtree_str_with_children(self):
        """Test BaBTree string representation with children."""
        from branching_domains import ShallowFirstBatchedDomainList

        parent = ShallowFirstBatchedDomainList.BaBTree('parent', 1, [0.1, 0.2])
        neg_child = ShallowFirstBatchedDomainList.BaBTree('neg', 2, [0.3, 0.4])
        pos_child = ShallowFirstBatchedDomainList.BaBTree('pos', 3, [0.5, 0.6])

        parent.neg_child = neg_child
        parent.pos_child = pos_child

        str_repr = str(parent)
        assert 'parent' in str_repr
        assert 'neg' in str_repr
        assert 'pos' in str_repr
        assert '-1:' in str_repr
        assert '1:' in str_repr


# ============================================================================
# ShallowFirstBatchedDomainList._append_to_history Tests
# ============================================================================

class TestAppendToHistory:
    """Tests for _append_to_history method."""

    def test_append_to_list(self, mock_net, mock_tensor_storage):
        """Test appending to a list element in history."""
        from branching_domains import ShallowFirstBatchedDomainList

        with patch('branching_domains.get_tensor_storage', mock_tensor_storage):
            # Create minimal valid inputs
            ret = {
                'lower_bounds': {'output': torch.zeros(1, 10)},
                'upper_bounds': {'output': torch.ones(1, 10)},
            }
            lAs = {'layer1': torch.randn(1, 10, 5)}
            global_lbs = torch.tensor([[0.0]])
            global_ubs = torch.tensor([[1.0]])
            alphas = {}
            history = {}
            thresholds = torch.tensor([[0.0]])

            # This will fail during init due to complex dependencies
            # Instead, test the method directly by creating a partial object
            class TestClass:
                _append_to_history = ShallowFirstBatchedDomainList._append_to_history

            obj = TestClass()
            history_tuple = ([1, 2, 3], [4, 5, 6])
            result = obj._append_to_history(history_tuple, 0, 10)

            assert result[0] == [1, 2, 3, 10]
            assert result[1] == [4, 5, 6]

    def test_append_to_tensor(self):
        """Test appending to a tensor element in history."""
        from branching_domains import ShallowFirstBatchedDomainList

        class TestClass:
            _append_to_history = ShallowFirstBatchedDomainList._append_to_history

        obj = TestClass()
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0])
        history_tuple = (tensor1, tensor2)

        result = obj._append_to_history(history_tuple, 0, 10.0)

        expected = torch.tensor([1.0, 2.0, 3.0, 10.0])
        assert torch.allclose(result[0], expected)
        assert torch.equal(result[1], tensor2)


# ============================================================================
# BatchedDomainList._assemble_domains Tests
# ============================================================================

class TestAssembleDomains:
    """Tests for _assemble_domains method."""

    def test_assemble_single_domain(self):
        """Test assembling a single domain."""
        from branching_domains import BatchedDomainList

        class TestClass:
            _assemble_domains = BatchedDomainList._assemble_domains

        obj = TestClass()

        global_lbs = torch.tensor([[0.1]])
        global_ubs = torch.tensor([[0.9]])
        histories = [{'layer1': []}]
        split_histories = [[]]
        depths = [2]
        thresholds = torch.tensor([[0.0]])

        result = obj._assemble_domains(
            global_lbs, global_ubs, histories, split_histories, depths, thresholds
        )

        assert len(result) == 1
        assert torch.equal(result[0].lower_bound, torch.tensor([0.1]))
        assert torch.equal(result[0].upper_bound, torch.tensor([0.9]))
        assert result[0].depth == 2
        assert result[0].history == {'layer1': []}

    def test_assemble_multiple_domains(self):
        """Test assembling multiple domains."""
        from branching_domains import BatchedDomainList

        class TestClass:
            _assemble_domains = BatchedDomainList._assemble_domains

        obj = TestClass()

        global_lbs = torch.tensor([[0.1], [0.2], [0.3]])
        global_ubs = torch.tensor([[0.9], [0.8], [0.7]])
        histories = [{'l1': []}, {'l2': []}, {'l3': []}]
        split_histories = [[], [], []]
        depths = [1, 2, 3]
        thresholds = torch.tensor([[0.0], [0.1], [0.2]])

        result = obj._assemble_domains(
            global_lbs, global_ubs, histories, split_histories, depths, thresholds
        )

        assert len(result) == 3
        assert result[0].depth == 1
        assert result[1].depth == 2
        assert result[2].depth == 3


# ============================================================================
# Integration-style tests with mocked dependencies
# ============================================================================

class TestBatchedDomainListBasics:
    """Basic tests for BatchedDomainList that don't require full initialization."""

    def test_len_returns_num_domains(self, create_batched_domain_list):
        """Test that __len__ returns num_domains."""
        domain_list = create_batched_domain_list(num_domains=42)
        assert len(domain_list) == 42

    def test_getitem_calls_assemble_domains(self, create_batched_domain_list):
        """Test that __getitem__ uses _assemble_domains."""
        domain_list = create_batched_domain_list(num_domains=5)

        result = domain_list[2]

        assert result.depth == 0  # All depths are initialized to 0
        assert result.lower_bound is not None


class TestBatchedDomainListUpdateUnstableMask:
    """Tests for update_unstable_mask method."""

    def test_update_unstable_mask_empty(self):
        """Test update_unstable_mask with empty mask."""
        from branching_domains import BatchedDomainList

        instance = object.__new__(BatchedDomainList)
        instance.unstable_mask = {}
        instance.net = MagicMock()
        instance.net.net = {}

        instance.update_unstable_mask({})

        assert instance.unstable_mask == {}

    def test_update_unstable_mask_with_data(self):
        """Test update_unstable_mask with actual mask data."""
        from branching_domains import BatchedDomainList

        instance = object.__new__(BatchedDomainList)
        instance.unstable_mask = {}

        # Create mock network nodes
        mock_input1 = MagicMock()
        mock_input1.name = 'input1'
        mock_input2 = MagicMock()
        mock_input2.name = 'input2'

        mock_node = MagicMock()
        mock_node.inputs = [mock_input1, mock_input2]

        instance.net = MagicMock()
        instance.net.net = {'op_node': mock_node}

        # Create masks
        mask1 = torch.tensor([True, False, True])
        mask2 = None  # Some masks can be None

        instance.update_unstable_mask({'op_node': [mask1, mask2]})

        assert 'input1' in instance.unstable_mask
        assert torch.equal(instance.unstable_mask['input1'], mask1)
        assert 'input2' not in instance.unstable_mask  # None mask should not be stored

    def test_update_unstable_mask_all_false(self):
        """Test update_unstable_mask when mask has no True values."""
        from branching_domains import BatchedDomainList

        instance = object.__new__(BatchedDomainList)
        instance.unstable_mask = {}

        mock_input = MagicMock()
        mock_input.name = 'input1'

        mock_node = MagicMock()
        mock_node.inputs = [mock_input]

        instance.net = MagicMock()
        instance.net.net = {'op_node': mock_node}

        # Mask with all False - should be stored as None
        mask = torch.tensor([False, False, False])

        instance.update_unstable_mask({'op_node': [mask]})

        assert instance.unstable_mask['input1'] is None


class TestBatchedDomainListSort:
    """Tests for sort method."""

    def test_sort_empty_list(self, capsys, create_batched_domain_list):
        """Test sort with empty domain list."""
        # Note: Cannot create with num_domains=0 due to initialization constraints,
        # so we create and then clear
        domain_list = create_batched_domain_list(num_domains=1)
        domain_list.num_domains = 0

        # Should not raise any errors
        domain_list.sort()

        # Should not print anything
        captured = capsys.readouterr()
        assert 'Sorting' not in captured.out


class TestBatchedDomainListGetMinDomain:
    """Tests for get_min_domain method."""

    def test_get_min_domain_single(self, create_batched_domain_list):
        """Test get_min_domain with single domain request."""
        domain_list = create_batched_domain_list(num_domains=3)

        result = domain_list.get_min_domain(1)

        # Should return a single domain
        assert len(result) == 1

    def test_get_min_domain_multiple(self, create_batched_domain_list):
        """Test get_min_domain with multiple domains request."""
        domain_list = create_batched_domain_list(num_domains=5)

        result = domain_list.get_min_domain(3)

        assert len(result) == 3


# ============================================================================
# Deque-related tests for BFS traversal
# ============================================================================

class TestBFSTraversal:
    """Tests related to breadth-first traversal using deques."""

    def test_deque_operations(self):
        """Test that deque operations work as expected for BFS."""
        from collections import deque

        # Simulate the deque-based storage used in BFS mode
        histories = deque([{'l': []} for _ in range(5)])
        depths = deque([1, 2, 3, 4, 5])

        # Pop from left (BFS behavior)
        batch = 2
        popped_histories = [histories.popleft() for _ in range(batch)]
        popped_depths = [depths.popleft() for _ in range(batch)]

        assert len(popped_histories) == 2
        assert len(popped_depths) == 2
        assert popped_depths == [1, 2]
        assert len(depths) == 3

    def test_deque_extend(self):
        """Test deque extend operations."""
        from collections import deque

        histories = deque([{'l': [1]}])
        new_items = ({'l': [2]}, {'l': [3]})

        histories.extend(new_items)

        assert len(histories) == 3
        assert histories[0] == {'l': [1]}
        assert histories[1] == {'l': [2]}
        assert histories[2] == {'l': [3]}


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for branching_domains module."""

    def test_check_worst_domain_with_multidim_bounds(self):
        """Test check_worst_domain with multi-dimensional bounds."""
        import arguments
        from branching_domains import check_worst_domain

        arguments.Config['bab']['cut']['enabled'] = False

        domain = SimpleNamespace()
        domain.lower_bound = torch.tensor([0.1, 0.2, 0.3])
        domain.threshold = torch.tensor([0.0, 0.0, 0.0])

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=1)
        mock_domain_list.get_min_domain = MagicMock(return_value=[domain])

        result = check_worst_domain(mock_domain_list)
        expected = torch.tensor([0.1, 0.2, 0.3])  # lb - threshold
        assert torch.allclose(result, expected)

    def test_check_worst_domain_negative_values(self):
        """Test check_worst_domain with negative bound values."""
        import arguments
        from branching_domains import check_worst_domain

        arguments.Config['bab']['cut']['enabled'] = False

        domain = SimpleNamespace()
        domain.lower_bound = torch.tensor([-0.5])
        domain.threshold = torch.tensor([0.1])

        mock_domain_list = MagicMock()
        mock_domain_list.__len__ = MagicMock(return_value=1)
        mock_domain_list.get_min_domain = MagicMock(return_value=[domain])

        result = check_worst_domain(mock_domain_list)
        expected = torch.tensor([-0.6])  # -0.5 - 0.1
        assert torch.allclose(result, expected)

    def test_assemble_domains_with_tensor_thresholds(self):
        """Test _assemble_domains with 2D threshold tensor."""
        from branching_domains import BatchedDomainList

        class TestClass:
            _assemble_domains = BatchedDomainList._assemble_domains

        obj = TestClass()

        global_lbs = torch.tensor([[0.1, 0.2]])
        global_ubs = torch.tensor([[0.9, 0.8]])
        histories = [{}]
        split_histories = [[]]
        depths = [1]
        thresholds = torch.tensor([[0.05, 0.05]])

        result = obj._assemble_domains(
            global_lbs, global_ubs, histories, split_histories, depths, thresholds
        )

        assert len(result) == 1
        assert torch.equal(result[0].threshold, torch.tensor([0.05, 0.05]))


# ============================================================================
# SimpleNamespace domain tests
# ============================================================================

class TestSimpleNamespaceDomain:
    """Tests for SimpleNamespace domain objects returned by various methods."""

    def test_domain_has_required_attributes(self):
        """Test that assembled domains have all required attributes."""
        from branching_domains import BatchedDomainList

        class TestClass:
            _assemble_domains = BatchedDomainList._assemble_domains

        obj = TestClass()

        result = obj._assemble_domains(
            torch.tensor([[0.1]]),
            torch.tensor([[0.9]]),
            [{'layer': []}],
            [[]],
            [5],
            torch.tensor([[0.0]])
        )

        domain = result[0]

        # Check all required attributes exist
        assert hasattr(domain, 'history')
        assert hasattr(domain, 'split_history')
        assert hasattr(domain, 'depth')
        assert hasattr(domain, 'lower_bound')
        assert hasattr(domain, 'upper_bound')
        assert hasattr(domain, 'threshold')


# ============================================================================
# BatchedDomainList Initialization Tests with Mocked Dependencies
# ============================================================================

class TestBatchedDomainListInit:
    """Tests for BatchedDomainList initialization."""

    def test_init_with_minimal_inputs(self, mock_net):
        """Test BatchedDomainList initialization with minimal inputs."""
        from branching_domains import BatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'depth_first'
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(2, 10)},
                'upper_bounds': {'output': torch.ones(2, 10)},
                'betas': None
            }
            c = torch.ones(2, 1, 10)
            lAs = {'layer1': torch.randn(2, 10, 5)}
            global_lbs = torch.tensor([[0.0], [0.1]])
            global_ubs = torch.tensor([[1.0], [0.9]])
            alphas = {'layer1': {'spec': torch.randn(2, 5, 2)}}
            history = {}
            thresholds = torch.tensor([[0.0]])

            mock_net.final_name = 'output'

            domain_list = BatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=False,
                x=None,
                enable_clip_domains=False
            )

            assert domain_list.num_domains == 2
            assert len(domain_list.histories) == 2
            assert len(domain_list.depths) == 2
            assert all(d == 0 for d in domain_list.depths)

    def test_alpha_storage_preserves_reduced_precision(self, mock_net):
        """Stored alpha values keep their configured domain precision."""
        from branching_domains import BatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'depth_first'
        arguments.Config['bab']['interm_transfer'] = False

        num_domains = 2
        alpha = torch.randn(2, 5, num_domains, dtype=torch.float16)
        domain_list = BatchedDomainList(
            ret={
                'lower_bounds': {'output': torch.zeros(num_domains, 10)},
                'upper_bounds': {'output': torch.ones(num_domains, 10)},
                'betas': None,
            },
            c=torch.ones(num_domains, 1, 10),
            lAs={'layer1': torch.randn(num_domains, 10, 5)},
            global_lbs=torch.tensor([[0.0], [0.1]]),
            global_ubs=torch.tensor([[1.0], [0.9]]),
            alphas={'layer1': {'spec': alpha}},
            history={},
            thresholds=torch.tensor([[0.0]]),
            net=mock_net,
            branching_input_and_activation=False,
            x=None,
            enable_clip_domains=False,
        )

        stored = domain_list.all_alphas['layer1']['spec']
        assert stored.dtype == torch.float16
        picked = domain_list.pick_out(batch=num_domains, device='cpu')
        assert picked['alphas']['layer1']['spec'].dtype == torch.float16

    def test_init_with_bfs_traversal(self, mock_net):
        """Test BatchedDomainList initialization with BFS traversal (uses deques)."""
        from branching_domains import BatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'breadth_first'
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(2, 10)},
                'upper_bounds': {'output': torch.ones(2, 10)},
                'betas': None
            }
            c = torch.ones(2, 1, 10)
            lAs = {'layer1': torch.randn(2, 10, 5)}
            global_lbs = torch.tensor([[0.0], [0.1]])
            global_ubs = torch.tensor([[1.0], [0.9]])
            alphas = {'layer1': {'spec': torch.randn(2, 5, 2)}}
            history = {}
            thresholds = torch.tensor([[0.0]])

            mock_net.final_name = 'output'

            domain_list = BatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=False,
                x=None,
                enable_clip_domains=False
            )

            # Check that BFS mode uses deques
            assert isinstance(domain_list.histories, deque)
            assert isinstance(domain_list.depths, deque)
            assert isinstance(domain_list.all_betas, deque)

    def test_init_with_input_branching(self, mock_net):
        """Test initialization with input branching enabled."""
        from branching_domains import BatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'depth_first'
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(2, 10)},
                'upper_bounds': {'output': torch.ones(2, 10)},
                'betas': None,
                'input_split_idx': torch.tensor([[0], [1]])
            }
            c = torch.ones(2, 1, 10)
            lAs = {'layer1': torch.randn(2, 10, 5)}
            global_lbs = torch.tensor([[0.0], [0.1]])
            global_ubs = torch.tensor([[1.0], [0.9]])
            alphas = {}
            history = {}
            thresholds = torch.tensor([[0.0]])

            # Mock input x with perturbation bounds
            mock_x = MagicMock()
            mock_x.ptb = MagicMock()
            mock_x.ptb.x_L = torch.zeros(2, 5)
            mock_x.ptb.x_U = torch.ones(2, 5)

            mock_net.final_name = 'output'

            domain_list = BatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=True,
                x=mock_x,
                enable_clip_domains=False
            )

            assert domain_list.all_x_Ls is not None
            assert domain_list.all_x_Us is not None
            assert domain_list.all_input_split_idx is not None

    def test_init_with_multi_threshold(self, mock_net):
        """Test initialization with multiple thresholds."""
        from branching_domains import BatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'depth_first'
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(2, 10)},
                'upper_bounds': {'output': torch.ones(2, 10)},
                'betas': None
            }
            c = torch.ones(2, 1, 10)
            lAs = {'layer1': torch.randn(2, 10, 5)}
            global_lbs = torch.tensor([[0.0], [0.1]])
            global_ubs = torch.tensor([[1.0], [0.9]])
            alphas = {}
            history = {}
            # Multiple thresholds (one per batch item)
            thresholds = torch.tensor([[0.1], [0.2]])

            mock_net.final_name = 'output'

            domain_list = BatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=False,
                x=None,
                enable_clip_domains=False
            )

            assert domain_list.all_thresholds is not None


# ============================================================================
# BatchedDomainList.sort Tests with Data
# ============================================================================

class TestBatchedDomainListSortWithData:
    """Tests for BatchedDomainList.sort method with actual domain data."""

    def test_sort_reorders_by_lower_bound(self, capsys, create_batched_domain_list):
        """Test that sort reorders domains by global lower bounds."""
        domain_list = create_batched_domain_list(num_domains=3)

        # Set specific lower bounds to test sorting: [0.5, 0.1, 0.3] -> sorted descending: [0.5, 0.3, 0.1]
        domain_list.all_global_lbs.data = torch.tensor([[0.5], [0.1], [0.3]])
        domain_list.all_global_ubs.data = torch.tensor([[0.9], [0.8], [0.7]])

        # Set specific depths to track reordering
        domain_list.depths = [1, 2, 3]

        domain_list.sort()

        captured = capsys.readouterr()
        assert 'Sorting batched domains' in captured.out

        # After sorting by descending margin: [0.5, 0.3, 0.1]
        # Original indices: 0 (0.5), 2 (0.3), 1 (0.1)
        assert domain_list.depths == [1, 3, 2]

    def test_sort_single_domain(self, capsys, create_batched_domain_list):
        """Test sort with a single domain."""
        domain_list = create_batched_domain_list(num_domains=1)

        # Set specific depth to verify it's unchanged
        domain_list.depths = [1]

        domain_list.sort()

        captured = capsys.readouterr()
        assert 'Sorting batched domains' in captured.out
        assert domain_list.depths == [1]

    def test_sort_with_bfs_deques(self, capsys, create_batched_domain_list):
        """Test sort with BFS mode using deques."""
        domain_list = create_batched_domain_list(num_domains=3, tree_traversal='breadth_first')

        # BFS mode should use deques
        assert isinstance(domain_list.histories, deque)
        assert isinstance(domain_list.depths, deque)

        domain_list.sort()

        # Check that result is still a deque after sorting
        assert isinstance(domain_list.histories, deque)
        assert isinstance(domain_list.depths, deque)


# ============================================================================
# BatchedDomainList.pick_out Tests
# ============================================================================

class TestBatchedDomainListPickOut:
    """Tests for pick_out method."""

    def test_pick_out_basic(self, create_batched_domain_list):
        """Test basic pick_out functionality."""
        domain_list = create_batched_domain_list(num_domains=3, with_alphas=True)

        # Set specific depths
        domain_list.depths = [1, 2, 3]

        result = domain_list.pick_out(batch=2, device='cpu')

        assert 'lAs' in result
        assert 'lower_bounds' in result
        assert 'upper_bounds' in result
        assert 'alphas' in result
        assert 'history' in result
        assert 'depths' in result
        assert len(result['history']) == 2
        assert domain_list.num_domains == 1

    def test_pick_out_with_bfs(self, create_batched_domain_list):
        """Test pick_out with BFS traversal using deques."""
        domain_list = create_batched_domain_list(num_domains=3, tree_traversal='breadth_first')

        # Set specific depths in deque
        domain_list.depths = deque([1, 2, 3])

        result = domain_list.pick_out(batch=2, device='cpu')

        # BFS pops from left
        assert result['depths'] == [1, 2]
        assert domain_list.num_domains == 1

    def test_pick_out_with_input_bounds(self, create_batched_domain_list):
        """Test pick_out with input bounds (x_Ls, x_Us)."""
        domain_list = create_batched_domain_list(num_domains=2, with_input_bounds=True)

        # Set specific depths
        domain_list.depths = [1, 2]

        result = domain_list.pick_out(batch=1, device='cpu')

        assert result['x_Ls'] is not None
        assert result['x_Us'] is not None
        assert result['input_split_idx'] is not None


# ============================================================================
# ShallowFirstBatchedDomainList Tests
# ============================================================================

class TestShallowFirstBatchedDomainListInit:
    """Tests for ShallowFirstBatchedDomainList initialization."""

    def test_init_sets_multi_tree_attributes(self, mock_net):
        """Test that initialization sets multi-tree attributes."""
        from branching_domains import ShallowFirstBatchedDomainList
        import arguments

        arguments.Config['bab']['tree_traversal'] = 'depth_first'
        arguments.Config['bab']['interm_transfer'] = False

        with patch('branching_domains.get_tensor_storage') as mock_storage:
            mock_storage.side_effect = MockTensorStorage

            ret = {
                'lower_bounds': {'output': torch.zeros(2, 10)},
                'upper_bounds': {'output': torch.ones(2, 10)},
                'betas': None
            }
            c = torch.ones(2, 1, 10)
            lAs = {'layer1': torch.randn(2, 10, 5)}
            global_lbs = torch.tensor([[0.0], [0.1]])
            global_ubs = torch.tensor([[1.0], [0.9]])
            alphas = {}
            history = {}
            thresholds = torch.tensor([[0.0]])

            mock_net.final_name = 'output'

            domain_list = ShallowFirstBatchedDomainList(
                ret=ret,
                c=c,
                lAs=lAs,
                global_lbs=global_lbs,
                global_ubs=global_ubs,
                alphas=alphas,
                history=history,
                thresholds=thresholds,
                net=mock_net,
                branching_input_and_activation=False,
                x=None
            )

            assert domain_list.is_initial_setup == True
            assert domain_list.use_bfs == True
            assert domain_list._multi_trees == {"children": dict()}
            assert domain_list.mtb_backup == []


class TestShallowFirstGenerateTree:
    """Tests for _generate_tree and _get_best_next_node methods."""

    def test_get_best_next_node(self):
        """Test _get_best_next_node returns node with most children."""
        from branching_domains import ShallowFirstBatchedDomainList

        class TestClass:
            _get_best_next_node = ShallowFirstBatchedDomainList._get_best_next_node

        obj = TestClass()

        multi_tree = {
            "children": {
                ('layer1', 0): {
                    "num_children": 5,
                    "score": [0.1, 0.2],
                    -1.0: {"children": {}},
                    1.0: {"children": {}},
                },
                ('layer1', 1): {
                    "num_children": 10,
                    "score": [0.3, 0.4],
                    -1.0: {"children": {}},
                    1.0: {"children": {}},
                },
            }
        }

        best_node, score = obj._get_best_next_node(multi_tree)

        assert best_node == ('layer1', 1)
        assert score == [0.3, 0.4]

    def test_generate_tree_leaf(self):
        """Test _generate_tree with leaf nodes only."""
        from branching_domains import ShallowFirstBatchedDomainList

        class TestClass:
            _generate_tree = ShallowFirstBatchedDomainList._generate_tree
            _get_best_next_node = ShallowFirstBatchedDomainList._get_best_next_node

        obj = TestClass()

        multi_tree = {
            "children": {
                ('layer1', 0): {
                    "num_children": 2,
                    "score": [0.1, 0.2],
                    -1.0: {"children": {}},
                    1.0: {"children": {}},
                },
            }
        }

        tree = obj._generate_tree(multi_tree)

        assert tree.layer_name == 'layer1'
        assert tree.node == 0
        assert tree.score == [0.1, 0.2]
        assert tree.neg_child is None
        assert tree.pos_child is None


# ============================================================================
# EmptyKLower and add_best_k_lower_bounds Tests
# ============================================================================

class TestAddBestKLowerBounds:
    """Tests for add_best_k_lower_bounds method."""

    def test_raises_empty_k_lower_when_no_domains(self):
        """Test that EmptyKLower is raised when no valid domains remain."""
        from branching_domains import ShallowFirstBatchedDomainList
        import arguments

        arguments.Config['bab']['cut']['biccos']['multi_tree_branching'] = {
            'keep_n_best_domains': 5,
            'target_batch_size': 10,
            'iterations': 3
        }

        class TestClass:
            add_best_k_lower_bounds = ShallowFirstBatchedDomainList.add_best_k_lower_bounds
            final_name = 'output'  # Need to add this attribute

        obj = TestClass()

        bounds = {
            'lower_bounds': {'output': torch.tensor([[0.5]])},
            'upper_bounds': {'output': torch.tensor([[0.9]])},
            'lAs': {},
            'alphas': {},
            'betas': [None],
            'intermediate_betas': [None],
            'split_history': [[]],
            'c': torch.ones(1, 1, 10),
            'x_Ls': None,
            'x_Us': None,
            'input_split_idx': None,
        }

        d = {
            'lower_bounds': {'output': torch.tensor([[0.5]])},
            'history': [{}],
            'thresholds': torch.tensor([[0.0]]),
            'depths': [1],
        }

        # All domains are verified (in verified_entries) and duplicated
        duplicated_branches = {0}
        verified_entries = torch.tensor([0])

        with pytest.raises(ShallowFirstBatchedDomainList.EmptyKLower) as exc_info:
            obj.add_best_k_lower_bounds(bounds, d, False, duplicated_branches, verified_entries, skip=0)

        assert "No valid domains remain" in str(exc_info.value)


# ============================================================================
# Print method with different configurations
# ============================================================================

class TestAbstractDomainListPrintConfigurations:
    """Additional tests for print method with various configurations."""

    def test_print_with_cplex_cuts_revpickup(self, capsys):
        """Test print when cplex_cuts_revpickup is enabled."""
        import arguments
        from branching_domains import AbstractDomainList

        arguments.Config['bab']['get_upper_bound'] = False
        arguments.Config['bab']['cut']['enabled'] = True
        arguments.Config['bab']['cut']['cplex_cuts'] = True
        arguments.Config['bab']['cut']['cplex_cuts_revpickup'] = True

        class TestDomainList(AbstractDomainList):
            def __len__(self):
                return 1

            def get_min_domain(self, n, rev_order=False):
                domain = SimpleNamespace()
                domain.lower_bound = torch.tensor([0.5])
                domain.upper_bound = torch.tensor([1.0])
                domain.threshold = torch.tensor([0.0])
                domain.depth = 2
                return [domain]

        domain_list = TestDomainList()
        domain_list.print()
        captured = capsys.readouterr()
        assert 'Current worst splitting domains lb-rhs (depth):' in captured.out

    def test_print_multiple_domains(self, capsys):
        """Test print with multiple domains."""
        import arguments
        from branching_domains import AbstractDomainList

        arguments.Config['bab']['get_upper_bound'] = True

        class TestDomainList(AbstractDomainList):
            def __len__(self):
                return 3

            def get_min_domain(self, n, rev_order=False):
                domains = []
                for i in range(min(n, 3)):
                    domain = SimpleNamespace()
                    domain.lower_bound = torch.tensor([0.1 * (i + 1)])
                    domain.upper_bound = torch.tensor([0.9 - 0.1 * i])
                    domain.threshold = torch.tensor([0.0])
                    domain.depth = i + 1
                    domains.append(domain)
                return domains

        domain_list = TestDomainList()
        domain_list.print(n=3)
        captured = capsys.readouterr()
        assert '(1)' in captured.out
        assert '(2)' in captured.out
        assert '(3)' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
