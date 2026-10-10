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
"""Unit tests for cuts/cut_verification.py"""
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch, mock_open

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCutVerification(unittest.TestCase):
    """Tests for the cut_verification function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_net = MagicMock()
        self.mock_net.cutter = MagicMock()
        self.mock_net.cutter.cuts = None
        self.mock_net.cutter.opt = False
        self.mock_net.net = MagicMock()
        self.mock_net.net.relus = []

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_no_cuts_no_domains(self, mock_print, mock_time):
        """Test cut_verification with no cuts enabled and empty domains."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        domains = []

        cut_verification(self.mock_net, domains)

        # Should not call build_the_model_lp or generate_cplex_cuts
        self.mock_net.build_the_model_lp.assert_not_called()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': True, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_lp_cut_enabled(self, mock_print, mock_time):
        """Test cut_verification with LP cut enabled."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        domains = []

        cut_verification(self.mock_net, domains)

        # Should call build_the_model_lp
        self.mock_net.build_the_model_lp.assert_called_once()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': True, 'cplex_cuts_wait': 0.1}}
    })
    @patch('cuts.cut_verification.generate_cplex_cuts')
    @patch('cuts.cut_verification.time.sleep')
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_cplex_cuts_enabled(self, mock_print, mock_time, mock_sleep, mock_generate):
        """Test cut_verification with CPLEX cuts enabled."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        domains = []

        cut_verification(self.mock_net, domains)

        # Should wait and generate cplex cuts
        mock_sleep.assert_called_once_with(0.1)
        mock_generate.assert_called_once_with(self.mock_net, recorder=None)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_with_cutter_opt(self, mock_print, mock_time):
        """Test cut_verification with cutter optimization enabled."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1

        # Set up cutter with opt=True
        self.mock_net.cutter.opt = True

        # Create domain with split_history
        mock_domain = MagicMock()
        mock_domain.split_history = {
            'general_betas': torch.tensor([1.0, 2.0])
        }
        domains = [mock_domain]

        cut_verification(self.mock_net, domains)

        # Should call refine_cuts
        self.mock_net.cutter.refine_cuts.assert_called_once_with(
            split_history=mock_domain.split_history
        )

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_with_cuts(self, mock_print, mock_time):
        """Test cut_verification when cuts exist."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1

        # Set up cutter with cuts
        self.mock_net.cutter.cuts = [{'some': 'cut'}]
        mock_cut_module = MagicMock()
        self.mock_net.cutter.construct_cut_module.return_value = mock_cut_module

        # Add relus to net
        mock_relu = MagicMock()
        self.mock_net.net.relus = [mock_relu]

        domains = []

        cut_verification(self.mock_net, domains)

        # Should construct cut module and set it
        self.mock_net.cutter.construct_cut_module.assert_called_once()
        self.assertEqual(self.mock_net.net.cut_module, mock_cut_module)
        self.assertEqual(mock_relu.cut_module, mock_cut_module)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_multiple_domains_assertion(self, mock_print, mock_time):
        """Test cut_verification asserts single domain when cutter has opt."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        self.mock_net.cutter.opt = True

        # Multiple domains should fail assertion
        mock_domain1 = MagicMock()
        mock_domain1.split_history = {'general_betas': torch.tensor([1.0])}
        mock_domain2 = MagicMock()
        mock_domain2.split_history = {'general_betas': torch.tensor([2.0])}
        domains = [mock_domain1, mock_domain2]

        with self.assertRaises(AssertionError):
            cut_verification(self.mock_net, domains)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': True, 'cplex_cuts': True, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.generate_cplex_cuts')
    @patch('cuts.cut_verification.time.sleep')
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_both_lp_and_cplex(self, mock_print, mock_time, mock_sleep, mock_generate):
        """Test cut_verification with both LP and CPLEX cuts enabled."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        domains = []

        cut_verification(self.mock_net, domains)

        # Should call both
        self.mock_net.build_the_model_lp.assert_called_once()
        mock_generate.assert_called_once_with(self.mock_net, recorder=None)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_empty_domains_no_opt(self, mock_print, mock_time):
        """Test cut_verification with empty domains doesn't call refine_cuts."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        self.mock_net.cutter.opt = True
        domains = []

        cut_verification(self.mock_net, domains)

        # Should not call refine_cuts because len(domains) < 1
        self.mock_net.cutter.refine_cuts.assert_not_called()


class TestSetCuts(unittest.TestCase):
    """Tests for the set_cuts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.cutter = MagicMock()
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.relus = []

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': None,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('builtins.print')
    @patch('builtins.exit')
    def test_set_cuts_no_manual_no_auto_exits(self, mock_exit, mock_print):
        """Test set_cuts exits when no cuts are configured."""
        from cuts.cut_verification import set_cuts

        set_cuts(self.mock_lirpa_net)

        mock_exit.assert_called_once()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': None,
            'cplex_cuts': True,
            'biccos': {'enabled': False}
        }}
    })
    @patch('builtins.print')
    def test_set_cuts_cplex_enabled_no_exit(self, mock_print):
        """Test set_cuts doesn't exit when cplex_cuts is enabled."""
        from cuts.cut_verification import set_cuts

        # Should not exit
        set_cuts(self.mock_lirpa_net)
        # No assertion needed - just checking no exception

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': None,
            'cplex_cuts': False,
            'biccos': {'enabled': True}
        }}
    })
    @patch('builtins.print')
    def test_set_cuts_biccos_enabled_no_exit(self, mock_print):
        """Test set_cuts doesn't exit when biccos is enabled."""
        from cuts.cut_verification import set_cuts

        # Should not exit
        set_cuts(self.mock_lirpa_net)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/test_cuts',
            'number_cuts': 10,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('builtins.print')
    def test_set_cuts_manual_cuts_text_file(self, mock_print, mock_read_cut):
        """Test set_cuts with manual cuts from text file."""
        from cuts.cut_verification import set_cuts

        mock_cuts = [{'cut': i} for i in range(15)]
        mock_read_cut.return_value = mock_cuts

        mock_cut_module = MagicMock()
        self.mock_lirpa_net.cutter.construct_cut_module.return_value = mock_cut_module

        set_cuts(self.mock_lirpa_net)

        mock_read_cut.assert_called_once_with('/tmp/test_cuts.cuts')
        # Should use only first 10 cuts (number_cuts=10)
        self.assertEqual(len(self.mock_lirpa_net.cutter.cuts), 10)
        self.mock_lirpa_net.cutter.construct_cut_module.assert_called_once()
        self.assertEqual(self.mock_lirpa_net.net.cut_module, mock_cut_module)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/test_cuts',
            'number_cuts': 100,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('cuts.cut_utils.parse_cplex_indx')
    @patch('cuts.cut_utils.parse_cplex_cuts')
    @patch('builtins.print')
    def test_set_cuts_binary_file_fallback(self, mock_print, mock_parse_cuts,
                                           mock_parse_indx, mock_read_cut):
        """Test set_cuts falls back to binary parsing on UnicodeDecodeError."""
        from cuts.cut_verification import set_cuts

        mock_read_cut.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'test')
        mock_parse_indx.return_value = ['var_0', 'var_1']
        mock_parse_cuts.return_value = ([{'cut': 0}], None)

        # Set up relus
        mock_relu = MagicMock()
        mock_relu.name = 'relu_0'
        mock_relu.inputs = [MagicMock(name='pre_relu_0')]
        self.mock_lirpa_net.net.relus = [mock_relu]

        mock_cut_module = MagicMock()
        self.mock_lirpa_net.cutter.construct_cut_module.return_value = mock_cut_module

        set_cuts(self.mock_lirpa_net)

        mock_parse_indx.assert_called_once_with('/tmp/test_cuts.indx')
        mock_parse_cuts.assert_called_once()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/nonexistent',
            'number_cuts': 100,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('cuts.cut_utils.parse_cplex_indx')
    @patch('builtins.print')
    def test_set_cuts_file_not_found_raises(self, mock_print, mock_parse_indx, mock_read_cut):
        """Test set_cuts raises exception when cut file not found."""
        from cuts.cut_verification import set_cuts

        mock_read_cut.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'test')
        mock_parse_indx.side_effect = FileNotFoundError()

        with self.assertRaises(Exception) as context:
            set_cuts(self.mock_lirpa_net)

        self.assertIn('Cannot read cut file', str(context.exception))

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/test_cuts',
            'number_cuts': 5,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('builtins.print')
    def test_set_cuts_respects_number_cuts_limit(self, mock_print, mock_read_cut):
        """Test set_cuts respects number_cuts configuration limit."""
        from cuts.cut_verification import set_cuts

        mock_cuts = [{'cut': i} for i in range(20)]
        mock_read_cut.return_value = mock_cuts

        mock_cut_module = MagicMock()
        self.mock_lirpa_net.cutter.construct_cut_module.return_value = mock_cut_module

        set_cuts(self.mock_lirpa_net)

        # Should use only first 5 cuts
        self.assertEqual(len(self.mock_lirpa_net.cutter.cuts), 5)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/test_cuts',
            'number_cuts': 100,
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('builtins.print')
    def test_set_cuts_sets_cut_module_on_relus(self, mock_print, mock_read_cut):
        """Test set_cuts sets cut_module on all relu layers."""
        from cuts.cut_verification import set_cuts

        mock_cuts = [{'cut': 0}]
        mock_read_cut.return_value = mock_cuts

        mock_cut_module = MagicMock()
        self.mock_lirpa_net.cutter.construct_cut_module.return_value = mock_cut_module

        # Set up multiple relus
        mock_relu1 = MagicMock()
        mock_relu2 = MagicMock()
        self.mock_lirpa_net.net.relus = [mock_relu1, mock_relu2]

        set_cuts(self.mock_lirpa_net)

        self.assertEqual(mock_relu1.cut_module, mock_cut_module)
        self.assertEqual(mock_relu2.cut_module, mock_cut_module)


class TestCreateMipBuildingProc(unittest.TestCase):
    """Tests for the create_mip_building_proc function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.model_ori = MagicMock()
        self.mock_lirpa_net.input_shape = (1, 3, 32, 32)
        self.mock_lirpa_net.c = torch.randn(1, 10, 10)
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net._modules = {}

    @patch('cuts.cut_verification.CPLEX_FOLDER', '/tmp/cplex')
    @patch('cuts.cut_verification.os.path.isfile')
    @patch('cuts.cut_verification.os.access')
    def test_create_mip_raises_if_no_executable(self, mock_access, mock_isfile):
        """Test create_mip_building_proc raises if get_cuts executable missing."""
        from cuts.cut_verification import create_mip_building_proc

        mock_isfile.return_value = False
        x = torch.randn(1, 3, 32, 32)

        with self.assertRaises(Exception) as context:
            create_mip_building_proc(self.mock_lirpa_net, x)

        self.assertIn('CPLEX cutting planes are needed', str(context.exception))

    @patch('cuts.cut_verification.CPLEX_FOLDER', '/tmp/cplex')
    @patch('cuts.cut_verification.os.path.isfile')
    @patch('cuts.cut_verification.os.access')
    @patch('cuts.cut_verification.multiprocessing.Manager')
    @patch('cuts.cut_verification.multiprocessing.Process')
    @patch('cuts.cut_verification.construct_mip_with_model')
    @patch('cuts.cut_verification.copy.deepcopy')
    def test_create_mip_starts_process(self, mock_deepcopy, mock_construct, mock_process,
                                        mock_manager, mock_access, mock_isfile):
        """Test create_mip_building_proc starts multiprocessing process."""
        from cuts.cut_verification import create_mip_building_proc

        mock_isfile.return_value = True
        mock_access.return_value = True

        mock_manager_instance = MagicMock()
        mock_manager_instance.dict.return_value = {}
        mock_manager.return_value = mock_manager_instance

        mock_mip_proc = MagicMock()
        mock_watchdog_proc = MagicMock()
        mock_process.side_effect = [mock_mip_proc, mock_watchdog_proc]

        mock_deepcopy.return_value = MagicMock()

        x = torch.randn(1, 3, 32, 32)

        create_mip_building_proc(self.mock_lirpa_net, x)

        # Process is called twice: once for mip_building_proc, once for get_cuts_watch_dog
        assert mock_process.call_count == 2
        mock_mip_proc.start.assert_called_once()
        mock_watchdog_proc.start.assert_called_once()
        self.assertEqual(self.mock_lirpa_net.mip_building_proc, mock_mip_proc)
        self.assertEqual(self.mock_lirpa_net.get_cuts_watch_dog_proc, mock_watchdog_proc)

    @patch('cuts.cut_verification.CPLEX_FOLDER', '/tmp/cplex')
    @patch('cuts.cut_verification.os.path.isfile')
    @patch('cuts.cut_verification.os.access')
    @patch('cuts.cut_verification.multiprocessing.Manager')
    @patch('cuts.cut_verification.multiprocessing.Process')
    @patch('cuts.cut_verification.construct_mip_with_model')
    @patch('cuts.cut_verification.copy.deepcopy')
    def test_create_mip_collects_intermediate_bounds(self, mock_deepcopy, mock_construct,
                                                      mock_process, mock_manager,
                                                      mock_access, mock_isfile):
        """Test create_mip_building_proc collects intermediate layer bounds."""
        from cuts.cut_verification import create_mip_building_proc

        mock_isfile.return_value = True
        mock_access.return_value = True

        mock_manager_instance = MagicMock()
        mock_manager_instance.dict.return_value = {}
        mock_manager.return_value = mock_manager_instance

        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        mock_deepcopy.return_value = MagicMock()

        # Set up layer with bounds
        mock_layer = MagicMock()
        mock_layer.lower = torch.tensor([1.0, 2.0])
        mock_layer.upper = torch.tensor([3.0, 4.0])
        mock_layer.is_lower_bound_current.return_value = True
        mock_layer.is_upper_bound_current.return_value = True
        self.mock_lirpa_net.net._modules = {'layer1': mock_layer}

        x = torch.randn(1, 3, 32, 32)

        create_mip_building_proc(self.mock_lirpa_net, x)

        # Process should be called with intermediate_bounds containing layer1
        call_args = mock_process.call_args
        self.assertIsNotNone(call_args)

    @patch('cuts.cut_verification.CPLEX_FOLDER', '/tmp/cplex')
    @patch('cuts.cut_verification.os.path.isfile')
    @patch('cuts.cut_verification.os.access')
    @patch('cuts.cut_verification.multiprocessing.Manager')
    @patch('cuts.cut_verification.multiprocessing.Process')
    @patch('cuts.cut_verification.construct_mip_with_model')
    @patch('cuts.cut_verification.copy.deepcopy')
    def test_create_mip_skips_layers_without_bounds(self, mock_deepcopy, mock_construct,
                                                     mock_process, mock_manager,
                                                     mock_access, mock_isfile):
        """Test create_mip_building_proc skips layers without current bounds."""
        from cuts.cut_verification import create_mip_building_proc

        mock_isfile.return_value = True
        mock_access.return_value = True

        mock_manager_instance = MagicMock()
        mock_manager_instance.dict.return_value = {}
        mock_manager.return_value = mock_manager_instance

        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        mock_deepcopy.return_value = MagicMock()

        # Set up layer without current bounds
        mock_layer = MagicMock()
        mock_layer.is_lower_bound_current.return_value = False
        mock_layer.is_upper_bound_current.return_value = False
        self.mock_lirpa_net.net._modules = {'layer1': mock_layer}

        x = torch.randn(1, 3, 32, 32)

        create_mip_building_proc(self.mock_lirpa_net, x)

        # Should start both mip_building_proc and get_cuts_watch_dog_proc
        assert mock_process_instance.start.call_count == 2


class TestEnableCuts(unittest.TestCase):
    """Tests for the enable_cuts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.return_A = False
        self.mock_lirpa_net.needed_A_dict = None
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.output_name = ['output_0']
        self.mock_lirpa_net.net.input_name = ['input_0']
        self.mock_lirpa_net.net.get_splittable_activations.return_value = []

    def test_enable_cuts_sets_return_A(self):
        """Test enable_cuts sets return_A to True."""
        from cuts.cut_verification import enable_cuts

        enable_cuts(self.mock_lirpa_net)

        self.assertTrue(self.mock_lirpa_net.return_A)

    def test_enable_cuts_creates_needed_A_dict(self):
        """Test enable_cuts creates needed_A_dict if None."""
        from cuts.cut_verification import enable_cuts

        enable_cuts(self.mock_lirpa_net)

        self.assertIsNotNone(self.mock_lirpa_net.needed_A_dict)
        self.assertIsInstance(self.mock_lirpa_net.needed_A_dict, defaultdict)

    def test_enable_cuts_adds_output_to_input_mapping(self):
        """Test enable_cuts adds output to input mapping."""
        from cuts.cut_verification import enable_cuts

        enable_cuts(self.mock_lirpa_net)

        self.assertIn('input_0', self.mock_lirpa_net.needed_A_dict['output_0'])

    def test_enable_cuts_adds_splittable_activations(self):
        """Test enable_cuts adds splittable activations to needed_A_dict."""
        from cuts.cut_verification import enable_cuts

        # Set up splittable activation
        mock_activation = MagicMock()
        mock_activation.inputs = [MagicMock(name='pre_relu_0')]
        mock_activation.inputs[0].name = 'pre_relu_0'
        self.mock_lirpa_net.net.get_splittable_activations.return_value = [
            mock_activation]

        enable_cuts(self.mock_lirpa_net)

        self.assertIn('input_0', self.mock_lirpa_net.needed_A_dict['pre_relu_0'])

    def test_enable_cuts_preserves_existing_needed_A_dict(self):
        """Test enable_cuts preserves existing needed_A_dict entries."""
        from cuts.cut_verification import enable_cuts

        existing_dict = defaultdict(set)
        existing_dict['existing_key'].add('existing_value')
        self.mock_lirpa_net.needed_A_dict = existing_dict

        enable_cuts(self.mock_lirpa_net)

        self.assertIn('existing_value', self.mock_lirpa_net.needed_A_dict['existing_key'])


class TestCreateCutter(unittest.TestCase):
    """Tests for the create_cutter function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.mip_building_proc = None
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.device = 'cpu'

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'number_cuts': 50, 'cplex_cuts': False}}
    })
    @patch('cuts.cut_verification.Cutter')
    def test_create_cutter_instantiates_cutter(self, mock_cutter_class):
        """Test create_cutter instantiates Cutter object."""
        from cuts.cut_verification import create_cutter

        A = {'layer': torch.randn(10, 10)}
        x = torch.randn(1, 3, 32, 32)

        create_cutter(self.mock_lirpa_net, A, x)

        mock_cutter_class.assert_called_once()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'number_cuts': 100, 'cplex_cuts': False}}
    })
    @patch('cuts.cut_verification.Cutter')
    def test_create_cutter_passes_number_cuts(self, mock_cutter_class):
        """Test create_cutter passes number_cuts config."""
        from cuts.cut_verification import create_cutter

        A = {'layer': torch.randn(10, 10)}
        x = torch.randn(1, 3, 32, 32)

        create_cutter(self.mock_lirpa_net, A, x)

        call_kwargs = mock_cutter_class.call_args[1]
        self.assertEqual(call_kwargs['number_cuts'], 100)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'number_cuts': 50, 'cplex_cuts': True}}
    })
    @patch('cuts.cut_verification.Cutter')
    def test_create_cutter_creates_mip_proc_if_cplex(self, mock_cutter_class):
        """Test create_cutter creates MIP building proc if cplex_cuts enabled."""
        from cuts.cut_verification import create_cutter

        A = {'layer': torch.randn(10, 10)}
        x = torch.randn(1, 3, 32, 32)

        create_cutter(self.mock_lirpa_net, A, x)

        self.mock_lirpa_net.create_mip_building_proc.assert_called_once_with(x)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'number_cuts': 50, 'cplex_cuts': True}}
    })
    @patch('cuts.cut_verification.Cutter')
    def test_create_cutter_skips_mip_if_already_exists(self, mock_cutter_class):
        """Test create_cutter skips MIP building proc if already exists."""
        from cuts.cut_verification import create_cutter

        self.mock_lirpa_net.mip_building_proc = MagicMock()  # Already exists

        A = {'layer': torch.randn(10, 10)}
        x = torch.randn(1, 3, 32, 32)

        create_cutter(self.mock_lirpa_net, A, x)

        self.mock_lirpa_net.create_mip_building_proc.assert_not_called()


class TestSetCutParams(unittest.TestCase):
    """Tests for the set_cut_params function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.cutter = MagicMock()
        self.mock_lirpa_net.cutter.beta_init = 0.5
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.device = 'cpu'
        self.mock_lirpa_net.net.final_name = 'output'
        output_node = MagicMock()
        output_node.name = 'output'
        output_node.output_dtype = torch.float64
        self.mock_lirpa_net.net.__getitem__.side_effect = {
            'output': output_node
        }.__getitem__
        self.mock_lirpa_net.net.get_output_dtype.side_effect = (
            lambda node: node.output_dtype)
        self.mock_lirpa_net.net.cut_module = MagicMock()
        self.mock_lirpa_net.net.cut_module.cut_bias = torch.zeros(10)
        self.mock_lirpa_net.net.get_splittable_activations.return_value = []
        self.mock_lirpa_net.net.cut_timestamp = 1

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_uses_solver_iteration(self, mock_print):
        """Test set_cut_params uses solver iteration when bab_iteration is 0."""
        from cuts.cut_verification import set_cut_params

        batch_size = 4
        batch_base = 2
        split_history = None

        iteration = set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        self.assertEqual(iteration, 100)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 50}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_uses_bab_iteration(self, mock_print):
        """Test set_cut_params uses bab_iteration when > 0."""
        from cuts.cut_verification import set_cut_params

        batch_size = 4
        batch_base = 2
        split_history = None

        iteration = set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        self.assertEqual(iteration, 50)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_creates_general_beta(self, mock_print):
        """Test set_cut_params creates general_beta tensor."""
        from cuts.cut_verification import set_cut_params

        batch_size = 4
        batch_base = 2
        split_history = None

        set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        # Check general_beta was set on cut_module
        general_beta = self.mock_lirpa_net.net.cut_module.general_beta
        self.assertIn('output', general_beta)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_with_split_history_warmup(self, mock_print):
        """Test set_cut_params warms up general_beta from split_history."""
        from cuts.cut_verification import set_cut_params

        batch_size = 2
        batch_base = 2
        num_constrs = 10

        # Create split_history with general_betas
        split_history = [
            {
                'general_betas': torch.randn(2, 1, 1, num_constrs),
                'cut_timestamp': 1
            },
            {
                'general_betas': torch.randn(2, 1, 1, num_constrs),
                'cut_timestamp': 1
            }
        ]

        set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        # Check that cut_beta_params was set
        self.assertIsNotNone(self.mock_lirpa_net.net.cut_beta_params)

        general_beta = self.mock_lirpa_net.net.cut_beta_params[0]
        self.assertEqual(general_beta.dtype, torch.float64)
        self.assertTrue(general_beta.requires_grad)
        self.assertTrue(general_beta.is_leaf)
        optimizer = torch.optim.Adam([general_beta], lr=0.1)
        general_beta.sum().backward()
        optimizer.step()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_skips_mismatched_timestamp(self, mock_print):
        """Test set_cut_params skips warmup when timestamp doesn't match."""
        from cuts.cut_verification import set_cut_params

        batch_size = 2
        batch_base = 2
        num_constrs = 10

        # Create split_history with wrong timestamp
        split_history = [
            {
                'general_betas': torch.randn(2, 1, 1, num_constrs),
                'cut_timestamp': 999  # Different from net.cut_timestamp (1)
            }
        ]

        set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        # Should still succeed without errors

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_sets_cut_used_on_activations(self, mock_print):
        """Test set_cut_params sets cut_used on splittable activations."""
        from cuts.cut_verification import set_cut_params

        mock_activation = MagicMock()
        self.mock_lirpa_net.net.get_splittable_activations.return_value = [
            mock_activation]

        batch_size = 4
        batch_base = 2
        split_history = None

        set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        self.assertEqual(mock_activation.cut_module, self.mock_lirpa_net.net.cut_module)
        self.assertTrue(mock_activation.cut_used)

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 0}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_sets_cut_timestamps(self, mock_print):
        """Test set_cut_params sets cut_timestamps list."""
        from cuts.cut_verification import set_cut_params

        batch_size = 4
        batch_base = 2
        split_history = None

        set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)

        timestamps = self.mock_lirpa_net.net.cut_module.cut_timestamps
        self.assertEqual(len(timestamps), batch_size)
        for ts in timestamps:
            self.assertEqual(ts, self.mock_lirpa_net.net.cut_timestamp)


class TestSetCutNewSplitHistory(unittest.TestCase):
    """Tests for the get_cut_new_split_history function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.final_name = 'output'
        self.mock_lirpa_net.net.cut_module = MagicMock()
        # Create a tensor with shape [2, 1, batch_size, num_constrs]
        self.mock_lirpa_net.net.cut_module.general_beta = {
            'output': torch.randn(2, 1, 4, 10)
        }
        self.mock_lirpa_net.net.cut_module.cut_timestamps = [1, 2, 3, 4]

    def test_get_cut_new_split_history_updates_all_batches(self):
        """Test get_cut_new_split_history updates all batch entries."""
        from cuts.cut_verification import get_cut_new_split_history

        batch_size = 4

        new_split_history = get_cut_new_split_history(self.mock_lirpa_net, batch_size)

        for i in range(batch_size):
            self.assertIn('general_betas', new_split_history[i])
            self.assertIn('cut_timestamp', new_split_history[i])

    def test_get_cut_new_split_history_correct_timestamps(self):
        """Test get_cut_new_split_history sets correct timestamps."""
        from cuts.cut_verification import get_cut_new_split_history

        batch_size = 4

        new_split_history = get_cut_new_split_history(self.mock_lirpa_net, batch_size)

        for i in range(batch_size):
            self.assertEqual(new_split_history[i]['cut_timestamp'], i + 1)

    def test_get_cut_new_split_history_detaches_tensors(self):
        """Test get_cut_new_split_history detaches general_betas tensors."""
        from cuts.cut_verification import get_cut_new_split_history

        # Create tensor that requires grad
        beta_tensor = torch.randn(2, 1, 4, 10, requires_grad=True)
        self.mock_lirpa_net.net.cut_module.general_beta = {'output': beta_tensor}

        batch_size = 4

        new_split_history = get_cut_new_split_history(self.mock_lirpa_net, batch_size)

        for i in range(batch_size):
            self.assertFalse(new_split_history[i]['general_betas'].requires_grad)

    def test_get_cut_new_split_history_correct_shape(self):
        """Test get_cut_new_split_history creates correct tensor shape."""
        from cuts.cut_verification import get_cut_new_split_history

        batch_size = 4

        new_split_history = get_cut_new_split_history(self.mock_lirpa_net, batch_size)

        for i in range(batch_size):
            # Each general_betas should be [2, 1, 1, 10] (single batch slice)
            shape = new_split_history[i]['general_betas'].shape
            self.assertEqual(shape[2], 1)  # Single batch dimension


class TestDisableCutForBranching(unittest.TestCase):
    """Tests for the disable_cut_for_branching function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.cut_used = True
        self.mock_lirpa_net.net.cut_beta_params = [torch.randn(10)]

        # Set up splittable activations
        mock_activation1 = MagicMock()
        mock_activation1.cut_used = True
        mock_activation2 = MagicMock()
        mock_activation2.cut_used = True
        self.mock_lirpa_net.net.get_splittable_activations.return_value = [
            mock_activation1, mock_activation2]

    @patch('builtins.print')
    def test_disable_cut_sets_net_cut_used_false(self, mock_print):
        """Test disable_cut_for_branching sets net.cut_used to False."""
        from cuts.cut_verification import disable_cut_for_branching

        disable_cut_for_branching(self.mock_lirpa_net)

        self.assertFalse(self.mock_lirpa_net.net.cut_used)

    @patch('builtins.print')
    def test_disable_cut_clears_cut_beta_params(self, mock_print):
        """Test disable_cut_for_branching clears cut_beta_params."""
        from cuts.cut_verification import disable_cut_for_branching

        disable_cut_for_branching(self.mock_lirpa_net)

        self.assertEqual(self.mock_lirpa_net.net.cut_beta_params, [])

    @patch('builtins.print')
    def test_disable_cut_sets_activation_cut_used_false(self, mock_print):
        """Test disable_cut_for_branching sets cut_used=False on activations."""
        from cuts.cut_verification import disable_cut_for_branching

        disable_cut_for_branching(self.mock_lirpa_net)

        for activation in (
                self.mock_lirpa_net.net.get_splittable_activations()):
            self.assertFalse(activation.cut_used)

    @patch('builtins.print')
    def test_disable_cut_prints_message(self, mock_print):
        """Test disable_cut_for_branching prints status message."""
        from cuts.cut_verification import disable_cut_for_branching

        disable_cut_for_branching(self.mock_lirpa_net)

        mock_print.assert_called_with('cut disabled for branching node selection')

    @patch('builtins.print')
    def test_disable_cut_handles_empty_activations(self, mock_print):
        """Test disable_cut_for_branching handles empty splittable_activations."""
        from cuts.cut_verification import disable_cut_for_branching

        self.mock_lirpa_net.net.get_splittable_activations.return_value = []

        # Should not raise
        disable_cut_for_branching(self.mock_lirpa_net)

        self.assertFalse(self.mock_lirpa_net.net.cut_used)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.net = MagicMock()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'lp_cut': False, 'cplex_cuts': False, 'cplex_cuts_wait': 0}}
    })
    @patch('cuts.cut_verification.time.time')
    @patch('builtins.print')
    def test_cut_verification_cutter_without_opt(self, mock_print, mock_time):
        """Test cut_verification when cutter has opt=False with domains."""
        from cuts.cut_verification import cut_verification

        mock_time.return_value = 1
        self.mock_lirpa_net.cutter = MagicMock()
        self.mock_lirpa_net.cutter.cuts = None
        self.mock_lirpa_net.cutter.opt = False

        mock_domain = MagicMock()
        domains = [mock_domain]

        # Should not call refine_cuts because opt=False
        cut_verification(self.mock_lirpa_net, domains)
        self.mock_lirpa_net.cutter.refine_cuts.assert_not_called()

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {
            'manual_cuts': '/tmp/test',
            'number_cuts': 0,  # Zero cuts requested
            'cplex_cuts': False,
            'biccos': {'enabled': False}
        }}
    })
    @patch('cuts.cut_utils.read_cut')
    @patch('builtins.print')
    def test_set_cuts_zero_number_cuts(self, mock_print, mock_read_cut):
        """Test set_cuts with number_cuts=0."""
        from cuts.cut_verification import set_cuts

        mock_cuts = [{'cut': i} for i in range(10)]
        mock_read_cut.return_value = mock_cuts

        mock_cut_module = MagicMock()
        self.mock_lirpa_net.cutter = MagicMock()
        self.mock_lirpa_net.cutter.construct_cut_module.return_value = mock_cut_module
        self.mock_lirpa_net.net.relus = []

        set_cuts(self.mock_lirpa_net)

        # Should use empty list when number_cuts=0
        self.assertEqual(len(self.mock_lirpa_net.cutter.cuts), 0)

    def test_set_cuts_requires_cutter(self):
        """Test set_cuts asserts cutter exists."""
        from cuts.cut_verification import set_cuts

        self.mock_lirpa_net.cutter = None

        with self.assertRaises(AssertionError):
            set_cuts(self.mock_lirpa_net)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration-style tests for common scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_lirpa_net = MagicMock()
        self.mock_lirpa_net.net = MagicMock()
        self.mock_lirpa_net.net.output_name = ['output_0']
        self.mock_lirpa_net.net.input_name = ['input_0']
        self.mock_lirpa_net.net.get_splittable_activations.return_value = []
        self.mock_lirpa_net.needed_A_dict = None
        self.mock_lirpa_net.return_A = False

    def test_enable_cuts_then_check_mapping(self):
        """Test enable_cuts creates expected A matrix mapping."""
        from cuts.cut_verification import enable_cuts

        # Add splittable activation
        mock_activation = MagicMock()
        mock_activation.inputs = [MagicMock()]
        mock_activation.inputs[0].name = 'relu_input_0'
        self.mock_lirpa_net.net.get_splittable_activations.return_value = [
            mock_activation]

        enable_cuts(self.mock_lirpa_net)

        # Check both output and activation inputs map to input
        self.assertIn('input_0', self.mock_lirpa_net.needed_A_dict['output_0'])
        self.assertIn('input_0', self.mock_lirpa_net.needed_A_dict['relu_input_0'])

    @patch('cuts.cut_verification.arguments.Config', {
        'bab': {'cut': {'bab_iteration': 25}},
        'solver': {'beta-crown': {'iteration': 100}}
    })
    @patch('builtins.print')
    def test_set_cut_params_split_history_without_general_betas(self, mock_print):
        """Test set_cut_params handles split_history without general_betas key."""
        from cuts.cut_verification import set_cut_params

        self.mock_lirpa_net.cutter = MagicMock()
        self.mock_lirpa_net.cutter.beta_init = 0
        self.mock_lirpa_net.net.device = 'cpu'
        self.mock_lirpa_net.net.final_name = 'output'
        output_node = MagicMock()
        output_node.name = 'output'
        output_node.output_dtype = torch.float64
        self.mock_lirpa_net.net.__getitem__.side_effect = {
            'output': output_node
        }.__getitem__
        self.mock_lirpa_net.net.get_output_dtype.side_effect = (
            lambda node: node.output_dtype)
        self.mock_lirpa_net.net.cut_module = MagicMock()
        self.mock_lirpa_net.net.cut_module.cut_bias = torch.zeros(5)
        self.mock_lirpa_net.net.cut_timestamp = 1

        batch_size = 2
        batch_base = 2
        # Split history without general_betas key
        split_history = [
            {'some_other_key': 'value'},
            {'cut_timestamp': 1}  # Has timestamp but no general_betas
        ]

        # Should not raise
        iteration = set_cut_params(self.mock_lirpa_net, batch_size, batch_base, split_history)
        self.assertEqual(iteration, 25)


if __name__ == '__main__':
    unittest.main()
