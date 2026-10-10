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
"""Unit tests for complete_verifier/abcrown.py - the main ABCROWN verifier interface."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import importlib

import numpy as np
import torch

# Add complete_verifier to path as the FIRST entry to ensure proper module resolution
COMPLETE_VERIFIER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMPLETE_VERIFIER_DIR not in sys.path:
    sys.path.insert(0, COMPLETE_VERIFIER_DIR)

# Remove the root Verifier_Development from path temporarily to avoid abcrown package conflict
ROOT_DIR = os.path.dirname(COMPLETE_VERIFIER_DIR)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)


def get_abcrown_class():
    """Import and return the ABCROWN class from complete_verifier/abcrown.py.

    This function creates an isolated environment to test the ABCROWN class
    without polluting sys.modules for other tests.
    """
    # Save original modules to restore later
    modules_to_mock = [
        'abcrown', 'jit_precompile', 'beta_CROWN_solver', 'lp_mip_solver',
        'lp_mip_solver.mip', 'attack', 'utils', 'specifications',
        'incomplete_verifier_func', 'loading', 'read_vnnlib', 'cuts',
        'cuts.cut_utils', 'lp_test', 'complete_verifier_func', 'arguments'
    ]

    saved_modules = {}
    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]

    try:
        import importlib.util
        abcrown_path = os.path.join(COMPLETE_VERIFIER_DIR, 'abcrown.py')
        spec = importlib.util.spec_from_file_location("abcrown_test_module", abcrown_path)
        abcrown_module = importlib.util.module_from_spec(spec)

        # Mock all the imports abcrown.py needs
        mock_modules = {
            'jit_precompile': MagicMock(),
            'beta_CROWN_solver': MagicMock(),
            'lp_mip_solver': MagicMock(),
            'lp_mip_solver.mip': MagicMock(),
            'attack': MagicMock(),
            'utils': MagicMock(),
            'specifications': MagicMock(),
            'incomplete_verifier_func': MagicMock(),
            'loading': MagicMock(),
            'read_vnnlib': MagicMock(),
            'cuts': MagicMock(),
            'cuts.cut_utils': MagicMock(),
            'lp_test': MagicMock(),
            'complete_verifier_func': MagicMock(),
        }

        # Setup arguments mock
        args_mock = MagicMock()
        args_mock.Config = MagicMock()
        args_mock.Config.parse_config = MagicMock()
        mock_modules['arguments'] = args_mock

        for mod_name, mock_obj in mock_modules.items():
            sys.modules[mod_name] = mock_obj

        spec.loader.exec_module(abcrown_module)
        ABCROWN_class = abcrown_module.ABCROWN

        return ABCROWN_class, args_mock
    finally:
        # Clean up - remove mock modules and restore originals
        for mod_name in modules_to_mock:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Restore original modules
        for mod_name, mod_obj in saved_modules.items():
            sys.modules[mod_name] = mod_obj


class TestABCROWNInit(unittest.TestCase):
    """Tests for ABCROWN.__init__ method."""

    def test_init_empty_args(self):
        """Test initialization with empty args list."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[])

        args_mock.Config.parse_config.assert_called_once_with([])

    def test_init_with_positional_args(self):
        """Test initialization passes args to parse_config."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=['--config', 'test.yaml'])

        args_mock.Config.parse_config.assert_called_once_with(['--config', 'test.yaml'])

    def test_init_with_list_kwarg(self):
        """Test initialization with list-type kwargs."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], some_list=['a', 'b', 'c'])

        # List kwargs should expand to: --some_list a b c
        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--some_list', call_args)
        self.assertIn('a', call_args)
        self.assertIn('b', call_args)
        self.assertIn('c', call_args)

    def test_init_with_bool_true_kwarg(self):
        """Test initialization with True boolean kwargs."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], enable_feature=True)

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--enable_feature', call_args)

    def test_init_with_bool_false_kwarg(self):
        """Test initialization with False boolean kwargs."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], disable_feature=False)

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--no_disable_feature', call_args)

    def test_init_with_non_list_non_bool_kwarg(self):
        """Test initialization with scalar kwargs (string, int, float)."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], timeout=60, device='cpu')

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--timeout=60', call_args)
        self.assertIn('--device=cpu', call_args)

    def test_init_with_mixed_kwargs(self):
        """Test initialization with mixed kwarg types."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(
            args=['--base_arg'],
            seed=42,
            verbose=True,
            skip_attack=False,
            methods=['ibp', 'crown']
        )

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--base_arg', call_args)
        self.assertIn('--seed=42', call_args)
        self.assertIn('--verbose', call_args)
        self.assertIn('--no_skip_attack', call_args)
        self.assertIn('--methods', call_args)


class TestABCROWNKwargsProcessing(unittest.TestCase):
    """Tests for kwargs processing in ABCROWN.__init__."""

    def test_empty_list_kwarg(self):
        """Test that empty list kwargs are handled."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], empty_list=[])

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--empty_list', call_args)

    def test_numeric_list_kwarg(self):
        """Test that numeric list kwargs are converted to strings."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], numbers=[1, 2, 3])

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--numbers', call_args)
        self.assertIn('1', call_args)
        self.assertIn('2', call_args)
        self.assertIn('3', call_args)

    def test_float_kwarg(self):
        """Test that float kwargs are formatted correctly."""
        ABCROWN, args_mock = get_abcrown_class()

        abcrown_instance = ABCROWN(args=[], epsilon=0.001)

        call_args = args_mock.Config.parse_config.call_args[0][0]
        self.assertIn('--epsilon=0.001', call_args)


if __name__ == '__main__':
    unittest.main()
