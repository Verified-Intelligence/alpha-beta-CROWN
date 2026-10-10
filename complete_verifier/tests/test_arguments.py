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
"""Unit tests for arguments.py"""
import os
import sys
import tempfile
import unittest

import torch
import yaml

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ConfigHandler


class TestConfigHandlerInit(unittest.TestCase):
    """Tests for ConfigHandler initialization."""

    def test_init_creates_parsers(self):
        """Test that initialization creates both parsers."""
        handler = ConfigHandler()
        self.assertIsNotNone(handler.defaults_parser)
        self.assertIsNotNone(handler.no_defaults_parser)

    def test_init_creates_hierarchies(self):
        """Test that initialization creates config hierarchies."""
        handler = ConfigHandler()
        self.assertIsInstance(handler.config_file_hierarchies, dict)
        self.assertGreater(len(handler.config_file_hierarchies), 0)

    def test_init_creates_default_args(self):
        """Test that initialization parses default args."""
        handler = ConfigHandler()
        self.assertIsInstance(handler.default_args, dict)
        self.assertIn('device', handler.default_args)
        self.assertIn('seed', handler.default_args)


class TestAddArgument(unittest.TestCase):
    """Tests for ConfigHandler.add_argument method."""

    def test_add_argument_requires_hierarchy(self):
        """Test that add_argument requires hierarchy parameter."""
        handler = ConfigHandler()
        with self.assertRaises(ValueError):
            handler.add_argument('--test_arg', type=int, default=5,
                                 help='Test argument for testing.')

    def test_add_argument_with_hierarchy(self):
        """Test adding argument with hierarchy."""
        handler = ConfigHandler()
        handler.add_argument('--test_param', type=int, default=10,
                             help='A test parameter for unit testing.',
                             hierarchy=['test', 'param'])
        self.assertIn('test_param', handler.config_file_hierarchies)
        self.assertEqual(handler.config_file_hierarchies['test_param'],
                         ['test', 'param'])

    def test_add_argument_requires_proper_help(self):
        """Test that add_argument requires proper help message."""
        handler = ConfigHandler()
        # Help too short
        with self.assertRaises(ValueError):
            handler.add_argument('--bad_arg', type=int, default=5,
                                 help='Short.', hierarchy=['test', 'arg'])
        # Help doesn't start with uppercase
        with self.assertRaises(ValueError):
            handler.add_argument('--bad_arg', type=int, default=5,
                                 help='this is a lowercase start.',
                                 hierarchy=['test', 'arg'])
        # Help doesn't end with period
        with self.assertRaises(ValueError):
            handler.add_argument('--bad_arg', type=int, default=5,
                                 help='This help has no period',
                                 hierarchy=['test', 'arg'])

    def test_add_private_argument_skips_help_check(self):
        """Test that private arguments skip help validation."""
        handler = ConfigHandler()
        # Should not raise even with short help
        handler.add_argument('--private_arg', type=int, default=5,
                             help='x', hierarchy=['test', 'private'],
                             private=True)


class TestSetDictByHierarchy(unittest.TestCase):
    """Tests for ConfigHandler.set_dict_by_hierarchy method."""

    def test_set_single_level(self):
        """Test setting value at single level."""
        handler = ConfigHandler()
        handler.all_args = {}
        handler.set_dict_by_hierarchy({}, ['level1'], 'value')
        self.assertEqual(handler.all_args['level1'], 'value')

    def test_set_multi_level(self):
        """Test setting value at multiple levels."""
        handler = ConfigHandler()
        handler.all_args = {}
        handler.set_dict_by_hierarchy({}, ['l1', 'l2', 'l3'], 'value')
        self.assertEqual(handler.all_args['l1']['l2']['l3'], 'value')

    def test_set_nonexist_ok_false(self):
        """Test that nonexist_ok=False raises on missing key."""
        handler = ConfigHandler()
        handler.all_args = {}
        with self.assertRaises(ValueError):
            handler.set_dict_by_hierarchy({}, ['nonexistent', 'key'], 'value',
                                          nonexist_ok=False)


class TestConstructConfigDict(unittest.TestCase):
    """Tests for ConfigHandler.construct_config_dict method."""

    def test_construct_from_defaults(self):
        """Test constructing config from defaults."""
        handler = ConfigHandler()
        handler.all_args = {}
        handler.construct_config_dict(handler.default_args)
        # Should have hierarchical structure
        self.assertIn('general', handler.all_args)
        self.assertIn('device', handler.all_args['general'])


class TestUpdateConfigDict(unittest.TestCase):
    """Tests for ConfigHandler.update_config_dict method."""

    def test_update_existing_value(self):
        """Test updating an existing value."""
        handler = ConfigHandler()
        handler.all_args = {'general': {'device': 'cuda'}}
        handler.update_config_dict(handler.all_args,
                                   {'general': {'device': 'cpu'}})
        self.assertEqual(handler.all_args['general']['device'], 'cpu')

    def test_update_nonexist_raises(self):
        """Test that updating non-existent key raises."""
        handler = ConfigHandler()
        handler.all_args = {'general': {'device': 'cuda'}}
        with self.assertRaises(ValueError):
            handler.update_config_dict(handler.all_args,
                                       {'nonexistent': {'key': 'value'}})


class TestDumpConfig(unittest.TestCase):
    """Tests for ConfigHandler.dump_config method."""

    def test_dump_nested(self):
        """Test dump with nested structure."""
        handler = ConfigHandler()
        args_dict = {'level1': {'level2': {'key': 'value'}}}
        result = handler.dump_config(args_dict)
        self.assertIn('level1:', result)
        self.assertIn('level2:', result)
        self.assertIn('key', result)


class TestParseConfig(unittest.TestCase):
    """Tests for ConfigHandler.parse_config method."""

    def test_parse_empty_args(self):
        """Test parsing with empty arguments."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        self.assertIn('general', handler.all_args)
        self.assertIn('device', handler.all_args['general'])

    def test_parse_commandline_args(self):
        """Test parsing commandline arguments."""
        handler = ConfigHandler()
        handler.parse_config(args=['--seed', '42'], verbose=False)
        self.assertEqual(handler.all_args['general']['seed'], 42)

    def test_parse_config_file(self):
        """Test parsing from config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            yaml.dump({'general': {'seed': 999}}, f)
            config_path = f.name
        try:
            handler = ConfigHandler()
            handler.parse_config(args=['--config', config_path], verbose=False)
            self.assertEqual(handler.all_args['general']['seed'], 999)
        finally:
            os.unlink(config_path)

    def test_commandline_overrides_config(self):
        """Test that commandline args override config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            yaml.dump({'general': {'seed': 999}}, f)
            config_path = f.name
        try:
            handler = ConfigHandler()
            handler.parse_config(
                args=['--config', config_path, '--seed', '123'],
                verbose=False
            )
            self.assertEqual(handler.all_args['general']['seed'], 123)
        finally:
            os.unlink(config_path)


class TestConfigHandlerAccessors(unittest.TestCase):
    """Tests for ConfigHandler accessor methods."""

    def test_getitem(self):
        """Test dictionary-style access."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        device = handler['general']['device']
        self.assertIn(device, ['cpu', 'cuda'])

    def test_keys(self):
        """Test getting keys."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        keys = handler.keys()
        self.assertIn('general', keys)


class TestDefaultValues(unittest.TestCase):
    """Tests for default parameter values."""

    def test_default_device(self):
        """Test default device is cuda."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        self.assertEqual(handler['general']['device'], 'cuda')

    def test_default_seed(self):
        """Test default seed value."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        self.assertEqual(handler['general']['seed'], 100)

    def test_default_complete_verifier(self):
        """Test default complete verifier."""
        handler = ConfigHandler()
        handler.parse_config(args=[], verbose=False)
        self.assertEqual(handler['general']['complete_verifier'], 'bab')


if __name__ == '__main__':
    unittest.main()
