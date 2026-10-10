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
"""Unit tests for heuristics/nonlinear/bp_opt.py - Optimized branching points."""

import unittest
import sys
import os
import tempfile
import builtins
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

import torch
import numpy as np


class TestBranchingPointOptInit(unittest.TestCase):
    """Test cases for BranchingPointOpt.__init__"""

    def setUp(self):
        """Set up common mocks for initialization tests."""
        # Mock the auto_LiRPA imports
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = MagicMock
        self.mock_bound_ops.BoundMatMul = MagicMock
        self.mock_bound_ops.BoundRelu = MagicMock

        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _create_mock_net(self, device='cpu'):
        """Helper to create a mock network."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = device
        return mock_net

    def test_init_creates_new_db_when_no_file_exists(self):
        """Test that __init__ creates a new database when no file exists."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = self._create_mock_net()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'nonexistent.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            self.assertEqual(bp_opt.db, {'version': 'v1', 'tables': []})
            self.assertEqual(bp_opt.db_path, db_path)

    def test_init_loads_existing_db(self):
        """Test that __init__ loads an existing database."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = self._create_mock_net()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'existing.pt')
            existing_db = {
                'version': 'v1',
                'tables': [{'arch': 'test', 'data': 'test_data'}]
            }
            torch.save(existing_db, db_path)

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            self.assertEqual(bp_opt.db['version'], 'v1')
            self.assertEqual(len(bp_opt.db['tables']), 1)
            self.assertEqual(bp_opt.db['tables'][0]['arch'], 'test')

    def test_init_clears_tables_on_version_mismatch(self):
        """Test that __init__ clears tables when version doesn't match."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = self._create_mock_net()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'old_version.pt')
            old_db = {
                'version': 'v0',  # Old version
                'tables': [{'arch': 'old', 'data': 'old_data'}]
            }
            torch.save(old_db, db_path)

            with patch('builtins.print'):  # Suppress warning
                bp_opt = BranchingPointOpt(
                    mock_net,
                    db_path=db_path,
                    num_iterations=10,
                    range_l=-1.0,
                    range_u=1.0,
                    step_size_1d=0.1,
                    step_size=0.2,
                    batch_size=100,
                    log_interval=5
                )

            self.assertEqual(bp_opt.db['tables'], [])

    def test_init_stores_parameters(self):
        """Test that __init__ stores all parameters correctly."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = self._create_mock_net('cuda:0')

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=100,
                range_l=-2.0,
                range_u=2.0,
                step_size_1d=0.05,
                step_size=0.1,
                batch_size=256,
                log_interval=10
            )

            self.assertEqual(bp_opt.num_iterations, 100)
            self.assertEqual(bp_opt.range_l, -2.0)
            self.assertEqual(bp_opt.range_u, 2.0)
            self.assertEqual(bp_opt.step_size_1d, 0.05)
            self.assertEqual(bp_opt.step_size, 0.1)
            self.assertEqual(bp_opt.batch_size, 256)
            self.assertEqual(bp_opt.log_interval, 10)
            self.assertEqual(bp_opt.device, 'cuda:0')

    def test_init_uses_network_device(self):
        """Test that __init__ uses the network's device."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        for device in ['cpu', 'cuda:0', 'cuda:1']:
            mock_net = self._create_mock_net(device)

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, 'test.pt')

                bp_opt = BranchingPointOpt(
                    mock_net,
                    db_path=db_path,
                    num_iterations=10,
                    range_l=-1.0,
                    range_u=1.0,
                    step_size_1d=0.1,
                    step_size=0.2,
                    batch_size=100,
                    log_interval=5
                )

                self.assertEqual(bp_opt.device, device)


class TestBranchingPointOptVersion(unittest.TestCase):
    """Test cases for BranchingPointOpt version attribute."""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = MagicMock
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_version_is_v1(self):
        """Test that version is 'v1'."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt
        self.assertEqual(BranchingPointOpt.version, 'v1')


class TestGetArchitecture(unittest.TestCase):
    """Test cases for BranchingPointOpt._get_architecture"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = MagicMock
        self.mock_bound_ops.BoundMatMul = type('BoundMatMul', (), {})
        self.mock_bound_ops.BoundRelu = type('BoundRelu', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_get_architecture_single_output(self):
        """Test _get_architecture with a single output node."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt
        from heuristics.nonlinear import bp_opt as bp_opt_module

        # Create a mock node
        mock_node = MagicMock()
        mock_node.output_name = ['output1']

        # Create a mock output node
        mock_output_node = MagicMock()
        mock_output_node.inputs = [mock_node]
        mock_output_node.range_l = -0.5
        mock_output_node.range_u = 0.5

        # Create mock network
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'
        mock_net.net.__getitem__ = MagicMock(return_value=mock_output_node)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')
            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Patch isinstance to return True for BoundActivation check
            original_isinstance = builtins.isinstance

            def mock_isinstance(obj, classinfo):
                # Check if checking against BoundActivation
                if hasattr(classinfo, '__name__') and classinfo.__name__ == 'BoundActivation':
                    return obj is mock_output_node
                return original_isinstance(obj, classinfo)

            with patch('builtins.isinstance', mock_isinstance):
                arch, output_nodes = bp_opt._get_architecture(mock_node)

            self.assertEqual(len(arch), 1)
            self.assertEqual(arch[0]['num_inputs'], 1)
            self.assertEqual(arch[0]['index'], 0)
            self.assertEqual(arch[0]['range_l'], -0.5)
            self.assertEqual(arch[0]['range_u'], 0.5)
            self.assertEqual(len(output_nodes), 1)

    def test_get_architecture_skips_non_activation(self):
        """Test that _get_architecture skips non-BoundActivation nodes."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        # Create a mock node
        mock_node = MagicMock()
        mock_node.output_name = ['output1']

        # Create a mock output node that is NOT a BoundActivation
        mock_output_node = MagicMock()

        # Create mock network
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'
        mock_net.net.__getitem__ = MagicMock(return_value=mock_output_node)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')
            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Don't patch isinstance - let it fail naturally for non-BoundActivation
            arch, output_nodes = bp_opt._get_architecture(mock_node)

            self.assertEqual(len(arch), 0)
            self.assertEqual(len(output_nodes), 0)


class TestGetLookupTable(unittest.TestCase):
    """Test cases for BranchingPointOpt._get_lookup_table"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_get_lookup_table_returns_existing(self):
        """Test that _get_lookup_table returns existing table if arch matches."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                      'range_l': -5.0, 'range_u': 5.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')
            existing_db = {
                'version': 'v1',
                'tables': [{'arch': test_arch, 'points': torch.zeros(10)}]
            }
            torch.save(existing_db, db_path)

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            result = bp_opt._get_lookup_table(test_arch)

            self.assertEqual(result['arch'], test_arch)

    def test_get_lookup_table_creates_new_when_not_found(self):
        """Test that _get_lookup_table creates new table when arch not found."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                      'range_l': -5.0, 'range_u': 5.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Mock _create_lookup_table to avoid complex computation
            mock_table = {
                'arch': test_arch,
                'points': torch.zeros(10),
                'loss': torch.zeros(10)
            }
            bp_opt._create_lookup_table = MagicMock(return_value=mock_table)

            result = bp_opt._get_lookup_table(test_arch)

            bp_opt._create_lookup_table.assert_called_once_with(test_arch)
            self.assertEqual(result['arch'], test_arch)
            # Check that the table was added to db
            self.assertEqual(len(bp_opt.db['tables']), 1)


class TestCreateLookupTable(unittest.TestCase):
    """Test cases for BranchingPointOpt._create_lookup_table"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_create_lookup_table_single_input(self):
        """Test _create_lookup_table with single input architecture."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,  # Few iterations for testing
                range_l=-0.5,
                range_u=0.5,
                step_size_1d=0.5,  # Large step to reduce computation
                step_size=0.5,
                batch_size=1000,
                log_interval=1
            )

            # Mock _optimize_points to avoid complex computation
            mock_result = {
                'points': torch.zeros(100, 1),
                'loss': torch.zeros(100, 1)
            }
            bp_opt._optimize_points = MagicMock(return_value=mock_result)

            with patch('builtins.print'):
                result = bp_opt._create_lookup_table(test_arch)

            self.assertEqual(result['arch'], test_arch)
            self.assertIn('points', result)
            self.assertIn('loss', result)
            self.assertEqual(result['step_size'], 0.5)  # step_size_1d for single input

    def test_create_lookup_table_uses_step_size_1d_for_single_input(self):
        """Test that step_size_1d is used for single input architectures."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        # Single input architecture (num_inputs = 1)
        test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=1,
                range_l=-0.5,
                range_u=0.5,
                step_size_1d=0.1,  # Different from step_size
                step_size=0.5,
                batch_size=10000,
                log_interval=1
            )

            mock_result = {'points': torch.zeros(100, 1), 'loss': torch.zeros(100, 1)}
            bp_opt._optimize_points = MagicMock(return_value=mock_result)

            with patch('builtins.print'):
                result = bp_opt._create_lookup_table(test_arch)

            self.assertEqual(result['step_size'], 0.1)  # step_size_1d

    def test_create_lookup_table_range_clamping(self):
        """Test that range is clamped based on architecture ranges."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        # Architecture with tighter ranges than global ranges
        test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                      'range_l': -0.3, 'range_u': 0.3}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=1,
                range_l=-1.0,  # Global range larger
                range_u=1.0,   # Global range larger
                step_size_1d=0.5,
                step_size=0.5,
                batch_size=10000,
                log_interval=1
            )

            mock_result = {'points': torch.zeros(100, 1), 'loss': torch.zeros(100, 1)}
            bp_opt._optimize_points = MagicMock(return_value=mock_result)

            with patch('builtins.print'):
                result = bp_opt._create_lookup_table(test_arch)

            # Range should be clamped to architecture's range
            self.assertEqual(result['range_l'], -0.3)
            self.assertEqual(result['range_u'], 0.3)


class TestOptimizePoints(unittest.TestCase):
    """Test cases for BranchingPointOpt._optimize_points"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_optimize_points_returns_correct_structure(self):
        """Test that _optimize_points returns correct structure."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            # Mock _get_loss to avoid complex computation
            bp_opt._get_loss = MagicMock(return_value=torch.rand(10))

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            lower = [torch.zeros(10)]
            upper = [torch.ones(10)]
            mask = torch.ones(10, dtype=torch.bool)

            with patch('builtins.print'):
                result = bp_opt._optimize_points(test_arch, lower, upper, mask)

            self.assertIn('points', result)
            self.assertIn('loss', result)
            self.assertEqual(result['points'].shape[0], 10)
            self.assertEqual(result['loss'].shape[0], 10)

    def test_optimize_points_iterations(self):
        """Test that _optimize_points runs correct number of iterations."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        num_iterations = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=num_iterations,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            call_count = 0

            def mock_get_loss(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return torch.rand(10)

            bp_opt._get_loss = mock_get_loss

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            lower = [torch.zeros(10)]
            upper = [torch.ones(10)]
            mask = torch.ones(10, dtype=torch.bool)

            with patch('builtins.print'):
                bp_opt._optimize_points(test_arch, lower, upper, mask)

            self.assertEqual(call_count, num_iterations)


class TestGetLoss(unittest.TestCase):
    """Test cases for BranchingPointOpt._get_loss"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_get_loss_returns_correct_shape(self):
        """Test that _get_loss returns tensor with correct shape."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            # Create mock node
            mock_node = MagicMock()
            mock_node.lw = torch.ones(10)
            mock_node.uw = torch.ones(10) * 2
            mock_node.lb = torch.zeros(10)
            mock_node.ub = torch.ones(10)
            mock_node.bound_relax = MagicMock()

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            nodes = [mock_node]
            lower_branched = [torch.zeros(10)]
            upper_branched = [torch.ones(10)]
            mask = torch.ones(5, dtype=torch.bool)

            result = bp_opt._get_loss(test_arch, nodes, lower_branched, upper_branched, mask)

            self.assertEqual(result.shape[0], mask.shape[0])

    def test_get_loss_applies_mask(self):
        """Test that _get_loss applies mask correctly."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            mock_node = MagicMock()
            mock_node.lw = torch.ones(10)
            mock_node.uw = torch.ones(10) * 2
            mock_node.lb = torch.zeros(10)
            mock_node.ub = torch.ones(10)
            mock_node.bound_relax = MagicMock()

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            nodes = [mock_node]
            lower_branched = [torch.zeros(10)]
            upper_branched = [torch.ones(10)]

            # Create mask with some False values
            mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)

            result = bp_opt._get_loss(test_arch, nodes, lower_branched, upper_branched, mask)

            # Check that masked values are zero
            self.assertEqual(result[1].item(), 0.0)
            self.assertEqual(result[3].item(), 0.0)


class TestGetBranchingPoints(unittest.TestCase):
    """Test cases for BranchingPointOpt.get_branching_points"""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = MagicMock
        self.mock_bound_ops.BoundMatMul = MagicMock
        self.mock_bound_ops.BoundRelu = MagicMock
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_get_branching_points_calls_get_architecture(self):
        """Test that get_branching_points calls _get_architecture."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_node = MagicMock()
        mock_node.name = 'test_node'
        mock_node.output_name = []  # Empty to skip the MatMul/ReLU check

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Mock _get_architecture to track calls
            bp_opt._get_architecture = MagicMock(return_value=([], []))
            # Mock _get_lookup_table to avoid actual lookup table creation
            # Index calculation: index = index * num_samples + index_l, then * num_samples + index_u
            # For bounds 0 to 1 with step 0.1, range_l=-1: index_l = ceil((0-(-1))/0.1) = 10
            # index_u = floor((1-(-1))/0.1) = 20, so max index = 10*20 + 20 = 220
            mock_table = {
                'points': torch.zeros(500, 1, device='cpu'),  # Large enough
                'range_l': -1.0,
                'range_u': 1.0,
                'step_size': 0.1,
                'num_samples': 20
            }
            bp_opt._get_lookup_table = MagicMock(return_value=mock_table)

            lower_bounds = {'test_node': torch.zeros(10)}
            upper_bounds = {'test_node': torch.ones(10)}

            bp_opt.get_branching_points(mock_node, lower_bounds, upper_bounds)

            bp_opt._get_architecture.assert_called_once_with(mock_node)

    def test_get_branching_points_uses_lookup_table(self):
        """Test that get_branching_points uses the lookup table."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_node = MagicMock()
        mock_node.name = 'test_node'
        mock_node.output_name = []

        mock_output = MagicMock()
        mock_output.inputs = [mock_node]

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        test_arch = [{'op': 'TestOp', 'num_inputs': 1, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Mock _get_architecture to return non-empty arch
            bp_opt._get_architecture = MagicMock(return_value=(test_arch, [mock_output]))

            # Mock _get_lookup_table with proper points tensor
            mock_table = {
                'points': torch.zeros(10000, 1, device='cpu'),  # Large enough for indexing
                'range_l': -1.0,
                'range_u': 1.0,
                'step_size': 0.1,
                'num_samples': 20
            }
            bp_opt._get_lookup_table = MagicMock(return_value=mock_table)

            lower_bounds = {'test_node': torch.zeros(10)}
            upper_bounds = {'test_node': torch.ones(10)}

            bp_opt.get_branching_points(mock_node, lower_bounds, upper_bounds)

            bp_opt._get_lookup_table.assert_called_once_with(test_arch)

    def test_get_branching_points_returns_points_tensor(self):
        """Test that get_branching_points returns a points tensor."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_node = MagicMock()
        mock_node.name = 'test_node'
        mock_node.output_name = []

        mock_output = MagicMock()
        mock_output.inputs = [mock_node]

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        test_arch = [{'op': 'TestOp', 'num_inputs': 1, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Mock _get_architecture to return non-empty arch
            bp_opt._get_architecture = MagicMock(return_value=(test_arch, [mock_output]))

            # Mock _get_lookup_table with proper points tensor
            mock_table = {
                'points': torch.zeros(10000, 1, device='cpu'),
                'range_l': -1.0,
                'range_u': 1.0,
                'step_size': 0.1,
                'num_samples': 20
            }
            bp_opt._get_lookup_table = MagicMock(return_value=mock_table)

            lower_bounds = {'test_node': torch.zeros(10)}
            upper_bounds = {'test_node': torch.ones(10)}

            result = bp_opt.get_branching_points(mock_node, lower_bounds, upper_bounds)

            self.assertIsInstance(result, torch.Tensor)
            # The result should have the same shape as lower_bounds plus one extra dim
            self.assertEqual(result.shape[0], 10)

    def test_get_branching_points_handles_shape_incompatibility(self):
        """Test that get_branching_points returns None for incompatible shapes."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_node = MagicMock()
        mock_node.name = 'test_node'
        mock_node.output_name = []

        mock_output = MagicMock()
        other_input = MagicMock()
        other_input.name = 'other_node'
        mock_output.inputs = [mock_node, other_input]

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        # Architecture with 2 inputs where index=0 means the node is at position 0
        test_arch = [{'op': 'TestOp', 'num_inputs': 2, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Mock _get_architecture
            bp_opt._get_architecture = MagicMock(return_value=(test_arch, [mock_output]))

            # Mock _get_lookup_table
            mock_table = {
                'points': torch.zeros(10000, 1, device='cpu'),
                'range_l': -1.0,
                'range_u': 1.0,
                'step_size': 0.1,
                'num_samples': 20
            }
            bp_opt._get_lookup_table = MagicMock(return_value=mock_table)

            # Test with incompatible shapes (same ndim but different shape, not broadcastable)
            # Shape (10,) and shape (5,) with neither being 1 is not broadcastable
            lower_bounds = {
                'test_node': torch.zeros(10),  # 1D, shape (10,)
                'other_node': torch.zeros(5)   # 1D, shape (5,) - incompatible
            }
            upper_bounds = {
                'test_node': torch.ones(10),
                'other_node': torch.ones(5)
            }

            result = bp_opt.get_branching_points(mock_node, lower_bounds, upper_bounds)

            # Should return None due to shape incompatibility
            self.assertIsNone(result)


class TestBranchingPointOptIntegration(unittest.TestCase):
    """Integration tests for BranchingPointOpt."""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.mock_bound_ops.BoundMatMul = type('BoundMatMul', (), {})
        self.mock_bound_ops.BoundRelu = type('BoundRelu', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_db_persistence(self):
        """Test that database is persisted across instances."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            # First instance creates empty db
            bp_opt1 = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Manually add a table
            test_table = {'arch': 'test', 'data': 'test_data'}
            bp_opt1.db['tables'].append(test_table)
            torch.save(bp_opt1.db, db_path)

            # Second instance should load the saved db
            bp_opt2 = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            self.assertEqual(len(bp_opt2.db['tables']), 1)
            self.assertEqual(bp_opt2.db['tables'][0]['arch'], 'test')

    def test_multiple_architectures_in_db(self):
        """Test handling multiple architectures in database."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        arch1 = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                  'range_l': -1.0, 'range_u': 1.0}]
        arch2 = [{'op': 'BoundTanh', 'num_inputs': 1, 'index': 0,
                  'range_l': -2.0, 'range_u': 2.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')
            existing_db = {
                'version': 'v1',
                'tables': [
                    {'arch': arch1, 'points': torch.zeros(10)},
                    {'arch': arch2, 'points': torch.ones(10)},
                ]
            }
            torch.save(existing_db, db_path)

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=10,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=5
            )

            # Should find arch1
            result1 = bp_opt._get_lookup_table(arch1)
            self.assertEqual(result1['arch'], arch1)
            self.assertTrue(torch.all(result1['points'] == 0))

            # Should find arch2
            result2 = bp_opt._get_lookup_table(arch2)
            self.assertEqual(result2['arch'], arch2)
            self.assertTrue(torch.all(result2['points'] == 1))


class TestLookupTableCreationEdgeCases(unittest.TestCase):
    """Test edge cases in lookup table creation."""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_create_lookup_table_multiple_inputs(self):
        """Test lookup table creation with multiple inputs."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        # Architecture with 2 inputs
        test_arch = [{'op': 'BoundMul', 'num_inputs': 2, 'index': 0,
                      'range_l': -1.0, 'range_u': 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=1,
                range_l=-0.5,
                range_u=0.5,
                step_size_1d=0.1,
                step_size=0.5,  # Should use this, not step_size_1d
                batch_size=10000,
                log_interval=1
            )

            mock_result = {'points': torch.zeros(100, 1), 'loss': torch.zeros(100, 1)}
            bp_opt._optimize_points = MagicMock(return_value=mock_result)

            with patch('builtins.print'):
                result = bp_opt._create_lookup_table(test_arch)

            # Should use step_size (0.5) not step_size_1d (0.1)
            self.assertEqual(result['step_size'], 0.5)

    def test_num_inputs_calculation(self):
        """Test that num_inputs is calculated correctly for multiple ops."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        # Architecture with multiple ops
        # First op: 2 inputs, second op: 3 inputs
        # Total inputs: 1 + (2-1) + (3-1) = 4
        test_arch = [
            {'op': 'BoundMul', 'num_inputs': 2, 'index': 0, 'range_l': -1.0, 'range_u': 1.0},
            {'op': 'BoundAdd', 'num_inputs': 3, 'index': 0, 'range_l': -1.0, 'range_u': 1.0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=1,
                range_l=-0.5,
                range_u=0.5,
                step_size_1d=0.1,
                step_size=0.5,
                batch_size=10000,
                log_interval=1
            )

            mock_result = {'points': torch.zeros(100, 1), 'loss': torch.zeros(100, 1)}
            bp_opt._optimize_points = MagicMock(return_value=mock_result)

            with patch('builtins.print'):
                result = bp_opt._create_lookup_table(test_arch)

            # With num_inputs > 1, should use step_size
            self.assertEqual(result['step_size'], 0.5)


class TestGetLossEdgeCases(unittest.TestCase):
    """Test edge cases in _get_loss method."""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_get_loss_with_list_lw_uw(self):
        """Test _get_loss when lw and uw are lists."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            # Create mock node with list lw/uw
            mock_node = MagicMock()
            mock_node.lw = [torch.ones(10), torch.ones(10) * 0.5]
            mock_node.uw = [torch.ones(10) * 2, torch.ones(10)]
            mock_node.lb = torch.zeros(10)
            mock_node.ub = torch.ones(10)
            mock_node.bound_relax = MagicMock()

            test_arch = [{'op': 'BoundMul', 'num_inputs': 2, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            nodes = [mock_node]
            lower_branched = [torch.zeros(10), torch.zeros(10)]
            upper_branched = [torch.ones(10), torch.ones(10)]
            mask = torch.ones(5, dtype=torch.bool)

            result = bp_opt._get_loss(test_arch, nodes, lower_branched, upper_branched, mask)

            self.assertEqual(result.shape[0], mask.shape[0])

    def test_get_loss_calls_bound_relax(self):
        """Test that _get_loss calls bound_relax on nodes."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=2,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            mock_node = MagicMock()
            mock_node.lw = torch.ones(10)
            mock_node.uw = torch.ones(10) * 2
            mock_node.lb = torch.zeros(10)
            mock_node.ub = torch.ones(10)
            mock_node.bound_relax = MagicMock()

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            nodes = [mock_node]
            lower_branched = [torch.zeros(10)]
            upper_branched = [torch.ones(10)]
            mask = torch.ones(5, dtype=torch.bool)

            bp_opt._get_loss(test_arch, nodes, lower_branched, upper_branched, mask)

            # Verify bound_relax was called
            mock_node.bound_relax.assert_called_once()
            # Verify init=True was passed
            _, kwargs = mock_node.bound_relax.call_args
            self.assertTrue(kwargs.get('init', False))


class TestOptimizePointsImprovement(unittest.TestCase):
    """Test improvement tracking in _optimize_points."""

    def setUp(self):
        """Set up common mocks."""
        self.mock_bound_ops = MagicMock()
        self.mock_bound_ops.BoundActivation = type('BoundActivation', (), {})
        self.patches = [
            patch.dict('sys.modules', {'auto_LiRPA.bound_ops': self.mock_bound_ops}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_optimize_points_tracks_best_loss(self):
        """Test that _optimize_points tracks best loss."""
        from heuristics.nonlinear.bp_opt import BranchingPointOpt

        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.device = 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.pt')

            bp_opt = BranchingPointOpt(
                mock_net,
                db_path=db_path,
                num_iterations=3,
                range_l=-1.0,
                range_u=1.0,
                step_size_1d=0.1,
                step_size=0.2,
                batch_size=100,
                log_interval=1
            )

            # Return decreasing losses
            loss_values = [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([0.5, 1.5, 2.5]),  # All better
                torch.tensor([0.3, 1.6, 2.4]),  # Mixed
            ]
            call_idx = [0]

            def mock_get_loss(*args, **kwargs):
                result = loss_values[call_idx[0]]
                call_idx[0] += 1
                return result

            bp_opt._get_loss = mock_get_loss

            test_arch = [{'op': 'BoundSigmoid', 'num_inputs': 1, 'index': 0,
                          'range_l': -1.0, 'range_u': 1.0}]
            lower = [torch.zeros(3)]
            upper = [torch.ones(3)]
            mask = torch.ones(3, dtype=torch.bool)

            with patch('builtins.print'):
                result = bp_opt._optimize_points(test_arch, lower, upper, mask)

            # Best loss should be the minimum seen
            expected_best = torch.tensor([0.3, 1.5, 2.4])
            self.assertTrue(torch.allclose(result['loss'].squeeze(), expected_best))


if __name__ == '__main__':
    unittest.main()
