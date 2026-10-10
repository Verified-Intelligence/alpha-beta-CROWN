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
"""Unit tests for cuts/cutter.py - Cutter class for cutting plane methods."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch


class TestCutterInit(unittest.TestCase):
    """Test Cutter.__init__ method."""

    def _create_mock_solver(self, num_relus=2, input_name='input'):
        """Create a mock solver with network for testing."""
        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {input_name}
        solver.net.relus = [MagicMock(name=f'relu_{i}') for i in range(num_relus)]
        return solver

    def test_init_basic(self):
        """Test basic Cutter initialization with minimal arguments."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver)

        self.assertEqual(cutter.solver, solver)
        self.assertEqual(cutter.net, solver.net)
        self.assertEqual(cutter.relus, solver.net.relus)
        self.assertIsNone(cutter.A)
        self.assertIsNone(cutter.x)
        self.assertEqual(cutter.number_cuts, 50)
        self.assertFalse(cutter.fix_intermediate_bounds)
        self.assertEqual(cutter.device, 'cuda')
        self.assertEqual(cutter.cuts, [])
        self.assertIsNone(cutter.cut_module)
        self.assertEqual(cutter.cut_timestamp, -1)
        self.assertEqual(cutter.num_relus, 2)

    def test_init_with_A_parameter(self):
        """Test Cutter initialization with A parameter."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver(input_name='x_input')

        # A is a dict of dicts where inner dict has input_name as key
        A = {
            'output1': {'x_input': torch.randn(3, 4)},
            'output2': {'x_input': torch.randn(5, 4)}
        }

        cutter = Cutter(solver, A=A)

        self.assertIsNotNone(cutter.A)
        self.assertEqual(len(cutter.A), 2)

    def test_init_with_x_parameter(self):
        """Test Cutter initialization with x (input tensor) parameter."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()
        x = torch.randn(1, 3, 32, 32)

        cutter = Cutter(solver, x=x)

        self.assertIsNotNone(cutter.x)
        self.assertEqual(cutter.x.shape, (1, 3, 32, 32))

    def test_init_with_number_cuts(self):
        """Test Cutter initialization with custom number_cuts."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver, number_cuts=100)

        self.assertEqual(cutter.number_cuts, 100)

    def test_init_with_fix_intermediate_bounds(self):
        """Test Cutter initialization with fix_intermediate_bounds=True."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver, fix_intermediate_bounds=True)

        self.assertTrue(cutter.fix_intermediate_bounds)

    def test_init_with_cpu_device(self):
        """Test Cutter initialization with CPU device."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver, device='cpu')

        self.assertEqual(cutter.device, 'cpu')

    def test_init_initializes_empty_lists(self):
        """Test that Cutter initializes all list attributes as empty."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver)

        self.assertEqual(cutter.unstable_idx_list, [])
        self.assertEqual(cutter.lower, [])
        self.assertEqual(cutter.upper, [])
        self.assertEqual(cutter.lAs, [])
        self.assertEqual(cutter.uAs, [])
        self.assertEqual(cutter.lbs, [])
        self.assertEqual(cutter.ubs, [])

    def test_init_default_attributes(self):
        """Test that default attributes are set correctly."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver)

        self.assertEqual(cutter.log_interval, 10)
        self.assertEqual(cutter.beta_init, 0)


class TestCutterUpdateNet(unittest.TestCase):
    """Test Cutter.update_net method."""

    def _create_mock_solver(self, num_relus=2):
        """Create a mock solver with network."""
        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = [MagicMock(name=f'relu_{i}') for i in range(num_relus)]
        return solver

    def test_update_net_updates_references(self):
        """Test that update_net updates net and relus references."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()
        cutter = Cutter(solver)

        # Create new net
        new_net = MagicMock()
        new_net.relus = [MagicMock(name='new_relu')]

        cutter.update_net(new_net)

        self.assertEqual(cutter.net, new_net)
        self.assertEqual(cutter.relus, new_net.relus)

    def test_update_net_with_different_relu_count(self):
        """Test update_net with a net having different number of relus."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver(num_relus=2)
        cutter = Cutter(solver)

        self.assertEqual(len(cutter.relus), 2)

        # Create new net with 5 relus
        new_net = MagicMock()
        new_net.relus = [MagicMock() for _ in range(5)]

        cutter.update_net(new_net)

        self.assertEqual(len(cutter.relus), 5)


class TestCutterInitCut(unittest.TestCase):
    """Test Cutter.init_cut method."""

    def _create_cutter(self):
        """Create a Cutter instance for testing."""
        from cuts.cutter import Cutter

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = []
        return Cutter(solver)

    def test_init_cut_default_c(self):
        """Test init_cut with default c=1."""
        cutter = self._create_cutter()

        cut = cutter.init_cut()

        self.assertEqual(cut['c'], 1)
        self.assertEqual(cut['x_decision'], [])
        self.assertEqual(cut['x_coeffs'], [])
        self.assertEqual(cut['arelu_decision'], [])
        self.assertEqual(cut['arelu_coeffs'], [])
        self.assertEqual(cut['relu_decision'], [])
        self.assertEqual(cut['relu_coeffs'], [])
        self.assertEqual(cut['pre_decision'], [])
        self.assertEqual(cut['pre_coeffs'], [])

    def test_init_cut_with_negative_c(self):
        """Test init_cut with c=-1."""
        cutter = self._create_cutter()

        cut = cutter.init_cut(c=-1)

        self.assertEqual(cut['c'], -1)

    def test_init_cut_returns_mutable_lists(self):
        """Test that init_cut returns independent mutable lists."""
        cutter = self._create_cutter()

        cut1 = cutter.init_cut()
        cut2 = cutter.init_cut()

        # Modify cut1
        cut1['x_decision'].append([0, 1])

        # cut2 should not be affected
        self.assertEqual(len(cut2['x_decision']), 0)


class TestCutterConstructCutModule(unittest.TestCase):
    """Test Cutter.construct_cut_module method."""

    def _create_mock_relu(self, name, flattened_nodes=10):
        """Create a mock ReLU layer."""
        relu = MagicMock()
        relu.name = name
        relu.flattened_nodes = flattened_nodes
        relu.inputs = [MagicMock(name=f'pre_{name}')]
        relu.inputs[0].name = f'pre_{name}'
        return relu

    def _create_cutter_with_relus(self, num_relus=2, device='cpu'):
        """Create a Cutter with mock ReLU layers."""
        from cuts.cutter import Cutter

        relus = [self._create_mock_relu(f'relu_{i}') for i in range(num_relus)]

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = relus
        solver.net.output_name = ['output']
        solver.net.__getitem__ = MagicMock(return_value=MagicMock(name='output_node'))
        solver.net.__getitem__.return_value.name = 'output_node'

        return Cutter(solver, device=device)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_no_cuts(self, mock_print, mock_cut_module):
        """Test construct_cut_module with no cuts."""
        cutter = self._create_cutter_with_relus()
        cutter.cuts = []

        mock_cut_module_instance = MagicMock()
        mock_cut_module.return_value = mock_cut_module_instance

        result = cutter.construct_cut_module()

        self.assertEqual(result, mock_cut_module_instance)
        mock_cut_module.assert_called_once_with(cutter.relus)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_resets_relu_attributes(self, mock_print, mock_cut_module):
        """Test that construct_cut_module resets relu layer attributes."""
        cutter = self._create_cutter_with_relus()
        cutter.cuts = []

        # Set some initial values
        for relu in cutter.relus:
            relu.masked_beta = torch.tensor([1.0])
            relu.split_beta_used = True
            relu.history_beta_used = True
            relu.single_beta_used = True
            relu.cut_used = True
            relu.relu_coeffs = torch.tensor([1.0])

        mock_cut_module.return_value = MagicMock()

        cutter.construct_cut_module()

        for relu in cutter.relus:
            self.assertIsNone(relu.masked_beta)
            self.assertFalse(relu.split_beta_used)
            self.assertFalse(relu.history_beta_used)
            self.assertFalse(relu.single_beta_used)
            self.assertFalse(relu.cut_used)
            self.assertIsNone(relu.relu_coeffs)
            self.assertIsNone(relu.arelu_coeffs)
            self.assertIsNone(relu.pre_coeffs)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_fix_intermediate_bounds(self, mock_print, mock_cut_module):
        """Test construct_cut_module with fix_intermediate_bounds=True."""
        cutter = self._create_cutter_with_relus()
        cutter.fix_intermediate_bounds = True
        cutter.cuts = []

        mock_cut_module.return_value = MagicMock()

        cutter.construct_cut_module()

        # When fix_intermediate_bounds=True, start_nodes should only contain output node
        # start_nodes is modified in place, so check self.start_nodes
        self.assertEqual(len(cutter.start_nodes), 1)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_relu_cuts(self, mock_print, mock_cut_module):
        """Test construct_cut_module with relu decision cuts."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [[0, 5]],  # layer 0, neuron 5
            'relu_coeffs': [1.0],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.5
        }]

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.relu_coeffs = {}
        mock_cut_module_instance.arelu_coeffs = {}
        mock_cut_module_instance.pre_coeffs = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module()

        # Should have relu layer marked as cut_used
        self.assertTrue(cutter.relus[0].cut_used)
        self.assertTrue(cutter.net.cut_used)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_x_cuts(self, mock_print, mock_cut_module):
        """Test construct_cut_module with x (input) decision cuts."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.x = torch.randn(1, 10)  # batch=1, 10 features
        cutter.cuts = [{
            'x_decision': [[0, 3]],  # input layer, neuron 3
            'x_coeffs': [2.0],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.0
        }]

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.relu_coeffs = {}
        mock_cut_module_instance.arelu_coeffs = {}
        mock_cut_module_instance.pre_coeffs = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module()

        self.assertTrue(cutter.use_x_cuts)
        self.assertTrue(cutter.net.cut_used)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_arelu_cuts(self, mock_print, mock_cut_module):
        """Test construct_cut_module with arelu decision cuts."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [[1, 7]],  # layer 1, neuron 7
            'arelu_coeffs': [0.5],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 1.0
        }]

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.relu_coeffs = {}
        mock_cut_module_instance.arelu_coeffs = {}
        mock_cut_module_instance.pre_coeffs = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module()

        self.assertTrue(cutter.relus[1].cut_used)
        self.assertIn(cutter.relus[1], cutter.arelu_layers)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_pre_cuts(self, mock_print, mock_cut_module):
        """Test construct_cut_module with pre-activation decision cuts."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [[0, 2]],  # layer 0, neuron 2
            'pre_coeffs': [3.0],
            'c': -1,
            'bias': -2.0
        }]

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.relu_coeffs = {}
        mock_cut_module_instance.arelu_coeffs = {}
        mock_cut_module_instance.pre_coeffs = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module()

        self.assertTrue(cutter.relus[0].cut_used)
        self.assertIn(cutter.relus[0], cutter.pre_layers)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_patches_mode_skip(self, mock_print, mock_cut_module):
        """Test that patches mode layers are skipped without use_patches_cut."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = []

        # Make the pre-relu node have patches mode
        cutter.relus[0].inputs[0].mode = 'patches'

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.active_cuts = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module(use_patches_cut=False)

        # The patches layer should have been skipped (empty active cuts)
        # Check that print was called with skip message
        skip_calls = [call for call in mock_print.call_args_list
                      if 'skip cut beta crown opt for patches layer' in str(call)]
        self.assertTrue(len(skip_calls) > 0)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_patches_mode_enabled(self, mock_print, mock_cut_module):
        """Test that patches mode layers are NOT skipped with use_patches_cut=True."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = []

        # Make the pre-relu node have patches mode
        cutter.relus[0].inputs[0].mode = 'patches'

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.active_cuts = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module(use_patches_cut=True)

        # The patches layer should NOT have been skipped
        skip_calls = [call for call in mock_print.call_args_list
                      if 'skip cut beta crown opt for patches layer' in str(call)]
        self.assertEqual(len(skip_calls), 0)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_empty_decisions(self, mock_print, mock_cut_module):
        """Test construct_cut_module with a cut that has no decisions."""
        cutter = self._create_cutter_with_relus(device='cpu')
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.0
        }]

        mock_cut_module_instance = MagicMock()
        mock_cut_module_instance.relu_coeffs = {}
        mock_cut_module_instance.arelu_coeffs = {}
        mock_cut_module_instance.pre_coeffs = {}
        mock_cut_module.return_value = mock_cut_module_instance

        cutter.construct_cut_module()

        # Empty decision cut should have max_layer_idx = -1
        self.assertFalse(cutter.net.cut_used)


class TestCutterUpdateCutModule(unittest.TestCase):
    """Test Cutter.update_cut_module method."""

    def _create_cutter_with_cut_module(self, device='cpu'):
        """Create a Cutter with initialized cut module."""
        from cuts.cutter import Cutter

        relu0 = MagicMock()
        relu0.name = 'relu_0'
        relu0.flattened_nodes = 10

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = [relu0]

        cutter = Cutter(solver, device=device)

        # Set up cut_module manually
        cutter.cut_module = MagicMock()
        cutter.cut_module.x_coeffs = torch.zeros((2, 5), device=device)
        cutter.cut_module.relu_coeffs = {'relu_0': torch.zeros((2, 10), device=device)}
        cutter.cut_module.arelu_coeffs = {'relu_0': torch.zeros((2, 10), device=device)}
        cutter.cut_module.pre_coeffs = {'relu_0': torch.zeros((2, 10), device=device)}

        return cutter

    def test_update_cut_module_x_coeffs(self):
        """Test update_cut_module updates x_coeffs correctly."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [[0, 2]],
            'x_coeffs': [1.5],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.5
        }]

        cutter.update_cut_module()

        # Check x_coeffs was updated
        self.assertEqual(cutter.cut_module.x_coeffs[0, 2].item(), 1.5)

    def test_update_cut_module_relu_coeffs(self):
        """Test update_cut_module updates relu_coeffs correctly."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [[0, 5]],
            'relu_coeffs': [2.0],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 1.0
        }]

        cutter.update_cut_module()

        self.assertEqual(cutter.cut_module.relu_coeffs['relu_0'][0, 5].item(), 2.0)

    def test_update_cut_module_arelu_coeffs(self):
        """Test update_cut_module updates arelu_coeffs correctly."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [[0, 3]],
            'arelu_coeffs': [0.75],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.0
        }]

        cutter.update_cut_module()

        self.assertEqual(cutter.cut_module.arelu_coeffs['relu_0'][0, 3].item(), 0.75)

    def test_update_cut_module_pre_coeffs(self):
        """Test update_cut_module updates pre_coeffs correctly."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [[0, 8]],
            'pre_coeffs': [3.5],
            'c': 1,
            'bias': 2.0
        }]

        cutter.update_cut_module()

        self.assertEqual(cutter.cut_module.pre_coeffs['relu_0'][0, 8].item(), 3.5)

    def test_update_cut_module_negative_c(self):
        """Test update_cut_module with c=-1 negates coefficients."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [[0, 1]],
            'x_coeffs': [2.0],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': -1,
            'bias': 1.0
        }]

        cutter.update_cut_module()

        # c=-1 should negate the coefficient
        self.assertEqual(cutter.cut_module.x_coeffs[0, 1].item(), -2.0)

    def test_update_cut_module_bias(self):
        """Test update_cut_module sets cut_bias correctly."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [
            {
                'x_decision': [],
                'x_coeffs': [],
                'relu_decision': [],
                'relu_coeffs': [],
                'arelu_decision': [],
                'arelu_coeffs': [],
                'pre_decision': [],
                'pre_coeffs': [],
                'c': 1,
                'bias': 0.5
            },
            {
                'x_decision': [],
                'x_coeffs': [],
                'relu_decision': [],
                'relu_coeffs': [],
                'arelu_decision': [],
                'arelu_coeffs': [],
                'pre_decision': [],
                'pre_coeffs': [],
                'c': -1,
                'bias': 2.0
            }
        ]

        cutter.update_cut_module()

        self.assertEqual(cutter.cut_module.cut_bias[0].item(), 0.5)
        self.assertEqual(cutter.cut_module.cut_bias[1].item(), -2.0)

    def test_update_cut_module_accumulates_coeffs(self):
        """Test that coefficients accumulate when same neuron appears multiple times."""
        cutter = self._create_cutter_with_cut_module()
        cutter.cuts = [{
            'x_decision': [[0, 1], [0, 1]],  # Same neuron twice
            'x_coeffs': [1.0, 0.5],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.0
        }]

        cutter.update_cut_module()

        # Should accumulate: 1.0 + 0.5 = 1.5
        self.assertEqual(cutter.cut_module.x_coeffs[0, 1].item(), 1.5)


class TestCutterConstructBeta(unittest.TestCase):
    """Test Cutter.construct_beta method."""

    def _create_cutter_for_beta(self, num_relus=2, device='cpu'):
        """Create a Cutter ready for construct_beta testing."""
        from cuts.cutter import Cutter

        relus = []
        for i in range(num_relus):
            relu = MagicMock()
            relu.name = f'relu_{i}'
            relu.inputs = [MagicMock()]
            relu.inputs[0].name = f'pre_relu_{i}'
            relu.inputs[0].output_dtype = (
                torch.float64 if i == 0 else torch.float32)
            relus.append(relu)

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = relus
        solver.net.final_name = 'output'
        solver.net.get_output_dtype.side_effect = (
            lambda node: node.output_dtype)

        cutter = Cutter(solver, device=device)
        cutter.cuts = [{'dummy': True}]  # One cut
        cutter.cut_module = MagicMock()

        # Set up start_nodes
        output_node = MagicMock()
        output_node.name = 'output'
        output_node.output_dtype = torch.float64
        cutter.start_nodes = [relus[0].inputs[0], relus[1].inputs[0], output_node]

        return cutter

    def test_construct_beta_creates_general_betas(self):
        """Test construct_beta creates general_betas dictionary."""
        cutter = self._create_cutter_for_beta()

        # Shapes: [relu0_shape, relu1_shape, output_shape]
        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        # Check general_betas were created
        self.assertIsNotNone(cutter.cut_module.general_beta)
        self.assertIn('pre_relu_0', cutter.cut_module.general_beta)
        self.assertIn('pre_relu_1', cutter.cut_module.general_beta)
        self.assertIn('output', cutter.cut_module.general_beta)

    def test_construct_beta_shape(self):
        """Test construct_beta creates tensors with correct shape."""
        cutter = self._create_cutter_for_beta()

        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        # Shape should be (2, *shape[1:], 1, num_cuts)
        # For pre_relu_0: (2, 10, 1, 1)
        self.assertEqual(cutter.cut_module.general_beta['pre_relu_0'].shape, (2, 10, 1, 1))
        self.assertEqual(cutter.cut_module.general_beta['pre_relu_1'].shape, (2, 20, 1, 1))
        self.assertEqual(cutter.cut_module.general_beta['output'].shape, (2, 5, 1, 1))

    def test_construct_beta_requires_grad(self):
        """Test construct_beta creates tensors with requires_grad=True."""
        cutter = self._create_cutter_for_beta()

        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        for name, beta in cutter.cut_module.general_beta.items():
            self.assertTrue(beta.requires_grad, f"Beta for {name} should require grad")

    def test_construct_beta_uses_each_start_node_dtype(self):
        """Cut beta parameters use the dtype of their propagation start."""
        cutter = self._create_cutter_for_beta()

        cutter.construct_beta([(1, 10), (1, 20), (1, 5)])

        betas = cutter.cut_module.general_beta
        self.assertEqual(betas['pre_relu_0'].dtype, torch.float64)
        self.assertEqual(betas['pre_relu_1'].dtype, torch.float32)
        self.assertEqual(betas['output'].dtype, torch.float64)
        for beta in betas.values():
            self.assertTrue(beta.requires_grad)
            self.assertTrue(beta.is_leaf)

        optimizer = torch.optim.Adam(list(betas.values()), lr=0.1)
        sum(beta.sum() for beta in betas.values()).backward()
        optimizer.step()

    def test_construct_beta_with_beta_init(self):
        """Test construct_beta initializes with beta_init value."""
        cutter = self._create_cutter_for_beta()
        cutter.beta_init = 0.5

        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        # All values should be 0.5 initially
        for beta in cutter.cut_module.general_beta.values():
            # Check before requires_grad effect
            self.assertTrue(torch.allclose(beta, torch.full_like(beta, 0.5)))

    def test_construct_beta_sets_cut_beta_params(self):
        """Test construct_beta sets net.cut_beta_params."""
        cutter = self._create_cutter_for_beta()

        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        # cut_beta_params should contain all general_betas
        self.assertEqual(len(cutter.net.cut_beta_params), 3)

    def test_construct_beta_multiple_cuts(self):
        """Test construct_beta with multiple cuts."""
        cutter = self._create_cutter_for_beta()
        cutter.cuts = [{'dummy': i} for i in range(5)]  # 5 cuts

        shapes = [(1, 10), (1, 20), (1, 5)]

        cutter.construct_beta(shapes)

        # Last dimension should be num_cuts=5
        self.assertEqual(cutter.cut_module.general_beta['pre_relu_0'].shape[-1], 5)


class TestCutterIntegration(unittest.TestCase):
    """Integration tests for Cutter class."""

    def _create_full_mock_network(self, num_relus=2, device='cpu'):
        """Create a more complete mock network for integration tests."""
        from cuts.cutter import Cutter

        relus = []
        for i in range(num_relus):
            relu = MagicMock()
            relu.name = f'relu_{i}'
            relu.flattened_nodes = 10 + i * 5
            relu.inputs = [MagicMock()]
            relu.inputs[0].name = f'pre_relu_{i}'
            relu.masked_beta = None
            relu.split_beta_used = False
            relu.history_beta_used = False
            relu.single_beta_used = False
            relu.cut_used = False
            relu.relu_coeffs = None
            relu.arelu_coeffs = None
            relu.pre_coeffs = None
            relus.append(relu)

        output_node = MagicMock()
        output_node.name = 'output'

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = relus
        solver.net.output_name = ['output']
        solver.net.final_name = 'output'
        solver.net.__getitem__ = MagicMock(return_value=output_node)
        solver.net.cut_used = False

        return Cutter(solver, device=device)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_full_workflow_with_multiple_cut_types(self, mock_print, mock_cut_module):
        """Test full workflow with cuts containing multiple decision types."""
        cutter = self._create_full_mock_network(device='cpu')
        cutter.x = torch.randn(1, 20)

        # Create cuts with all decision types
        cutter.cuts = [
            {
                'x_decision': [[0, 1]],
                'x_coeffs': [1.0],
                'relu_decision': [[0, 2]],
                'relu_coeffs': [2.0],
                'arelu_decision': [[1, 3]],
                'arelu_coeffs': [1.5],
                'pre_decision': [[0, 4]],
                'pre_coeffs': [0.5],
                'c': 1,
                'bias': 0.5
            },
            {
                'x_decision': [],
                'x_coeffs': [],
                'relu_decision': [[1, 5]],
                'relu_coeffs': [3.0],
                'arelu_decision': [],
                'arelu_coeffs': [],
                'pre_decision': [],
                'pre_coeffs': [],
                'c': -1,
                'bias': -1.0
            }
        ]

        # Set up mock cut module
        mock_cm = MagicMock()
        mock_cm.relu_coeffs = {}
        mock_cm.arelu_coeffs = {}
        mock_cm.pre_coeffs = {}
        mock_cut_module.return_value = mock_cm

        # Run construct_cut_module
        result = cutter.construct_cut_module()

        # Verify net.cut_used is True
        self.assertTrue(cutter.net.cut_used)

        # Verify use_x_cuts is True (we have x_decision)
        self.assertTrue(cutter.use_x_cuts)

        # Verify layers are tracked
        self.assertEqual(len(cutter.relu_layers), 2)  # Both relus used
        self.assertEqual(len(cutter.arelu_layers), 1)  # Only relu_1
        self.assertEqual(len(cutter.pre_layers), 1)  # Only relu_0

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_active_cuts_for_output_node(self, mock_print, mock_cut_module):
        """Test that output node gets all cuts as active."""
        cutter = self._create_full_mock_network(device='cpu')
        cutter.cuts = [
            {'x_decision': [], 'x_coeffs': [], 'relu_decision': [], 'relu_coeffs': [],
             'arelu_decision': [], 'arelu_coeffs': [], 'pre_decision': [], 'pre_coeffs': [],
             'c': 1, 'bias': 0.0}
            for _ in range(5)
        ]

        mock_cm = MagicMock()
        mock_cm.relu_coeffs = {}
        mock_cm.arelu_coeffs = {}
        mock_cm.pre_coeffs = {}
        mock_cm.active_cuts = {}
        mock_cut_module.return_value = mock_cm

        cutter.construct_cut_module()

        # Output node should have all 5 cuts active
        self.assertEqual(mock_cm.active_cuts['output'].numel(), 5)


class TestCutterEdgeCases(unittest.TestCase):
    """Test edge cases for Cutter class."""

    def _create_mock_solver(self, num_relus=1):
        """Create a mock solver."""
        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = [MagicMock() for _ in range(num_relus)]
        return solver

    def test_init_with_empty_A(self):
        """Test Cutter with empty A dictionary."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()
        A = {}  # Empty dict

        cutter = Cutter(solver, A=A)

        self.assertEqual(cutter.A, [])

    def test_init_with_zero_relus(self):
        """Test Cutter with network having no relus."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver(num_relus=0)

        cutter = Cutter(solver)

        self.assertEqual(cutter.num_relus, 0)
        self.assertEqual(cutter.relus, [])

    def test_init_cut_with_zero_c(self):
        """Test init_cut with c=0."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()
        cutter = Cutter(solver)

        cut = cutter.init_cut(c=0)

        self.assertEqual(cut['c'], 0)

    @patch('cuts.cutter.CutModule')
    @patch('builtins.print')
    def test_construct_cut_module_with_zero_bias(self, mock_print, mock_cut_module):
        """Test construct_cut_module with zero bias cut."""
        from cuts.cutter import Cutter

        relu = MagicMock()
        relu.name = 'relu_0'
        relu.flattened_nodes = 10
        relu.inputs = [MagicMock()]
        relu.inputs[0].name = 'pre_relu_0'

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = [relu]
        solver.net.output_name = ['output']
        solver.net.__getitem__ = MagicMock(return_value=MagicMock(name='output'))
        solver.net.__getitem__.return_value.name = 'output'

        cutter = Cutter(solver, device='cpu')
        cutter.cuts = [{
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [[0, 0]],
            'relu_coeffs': [1.0],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'c': 1,
            'bias': 0.0
        }]

        mock_cm = MagicMock()
        mock_cm.relu_coeffs = {}
        mock_cm.arelu_coeffs = {}
        mock_cm.pre_coeffs = {}
        mock_cut_module.return_value = mock_cm

        result = cutter.construct_cut_module()

        self.assertIsNotNone(result)

    def test_update_cut_module_empty_cuts(self):
        """Test update_cut_module with empty cuts list."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()
        cutter = Cutter(solver, device='cpu')
        cutter.cuts = []
        cutter.cut_module = MagicMock()

        # Should not raise with empty cuts
        cutter.update_cut_module()

        # cut_bias should be an empty tensor
        self.assertEqual(cutter.cut_module.cut_bias.numel(), 0)


class TestCutterDeviceHandling(unittest.TestCase):
    """Test Cutter device handling."""

    def _create_mock_solver(self):
        """Create a mock solver."""
        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = []
        return solver

    def test_default_device_is_cuda(self):
        """Test that default device is cuda."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver)

        self.assertEqual(cutter.device, 'cuda')

    def test_cpu_device(self):
        """Test explicit CPU device."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver, device='cpu')

        self.assertEqual(cutter.device, 'cpu')

    def test_specific_cuda_device(self):
        """Test specific CUDA device (e.g., cuda:1)."""
        from cuts.cutter import Cutter

        solver = self._create_mock_solver()

        cutter = Cutter(solver, device='cuda:1')

        self.assertEqual(cutter.device, 'cuda:1')


class TestCutterAttributeAccess(unittest.TestCase):
    """Test Cutter attribute access patterns."""

    def _create_cutter(self):
        """Create a Cutter instance."""
        from cuts.cutter import Cutter

        solver = MagicMock()
        solver.net = MagicMock()
        solver.net.input_name = {'input'}
        solver.net.relus = [MagicMock(), MagicMock()]
        return Cutter(solver)

    def test_solver_reference_retained(self):
        """Test that solver reference is retained."""
        cutter = self._create_cutter()

        self.assertIsNotNone(cutter.solver)
        self.assertEqual(cutter.solver.net, cutter.net)

    def test_relus_reference_synced(self):
        """Test that relus reference stays synced with net.relus."""
        cutter = self._create_cutter()

        self.assertEqual(cutter.relus, cutter.net.relus)

    def test_cuts_list_mutable(self):
        """Test that cuts list is mutable."""
        cutter = self._create_cutter()

        initial_count = len(cutter.cuts)
        cutter.cuts.append({'new': 'cut'})

        self.assertEqual(len(cutter.cuts), initial_count + 1)


if __name__ == '__main__':
    unittest.main()
