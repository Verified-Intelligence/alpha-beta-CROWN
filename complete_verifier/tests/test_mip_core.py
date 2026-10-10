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
"""Unit tests for lp_mip_solver/mip_core.py module."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import sys
import os

# Add the complete_verifier directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSolverResultToLegacyFormat(unittest.TestCase):
    """Tests for SolverResult.to_legacy_format method."""

    def test_solver_result_to_legacy_format_with_values(self):
        """Test to_legacy_format with global_lb present."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        global_lb = torch.tensor([[0.5]])
        lower_bounds = {'output': torch.tensor([0.5])}
        upper_bounds = {'output': torch.tensor([1.0])}

        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            global_lb=global_lb,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            refined_betas={'beta': 0.2}
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(status_str, "safe-mip")
        self.assertTrue(torch.equal(result_dict['global_lb'], global_lb))
        self.assertEqual(result_dict['lower_bounds'], lower_bounds)
        self.assertEqual(result_dict['upper_bounds'], upper_bounds)
        self.assertEqual(result_dict['refined_betas'], {'beta': 0.2})

    def test_solver_result_to_legacy_format_without_global_lb(self):
        """Test to_legacy_format when global_lb is None."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        result = SolverResult(status=VerificationResult.UNKNOWN_MIP)

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(status_str, "unknown-mip")
        # Should create a default global_lb of -inf
        self.assertEqual(result_dict['global_lb'].item(), float('-inf'))
        self.assertEqual(result_dict['lower_bounds'], {})
        self.assertEqual(result_dict['upper_bounds'], {})

    def test_solver_result_to_legacy_format_empty_bounds(self):
        """Test to_legacy_format with None bounds returns empty dicts."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        result = SolverResult(
            status=VerificationResult.SAFE_LP,
            global_lb=torch.tensor([[1.0]]),
            lower_bounds=None,
            upper_bounds=None
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(result_dict['lower_bounds'], {})
        self.assertEqual(result_dict['upper_bounds'], {})


class TestMIPSolverInit(unittest.TestCase):
    """Tests for MIPSolver initialization."""

    def test_mip_solver_init(self):
        """Test MIPSolver initialization."""
        from lp_mip_solver.mip_core import MIPSolver

        config = {'timeout': 100, 'threads': 4}
        solver = MIPSolver(config)

        self.assertEqual(solver.config, config)
        self.assertIsNone(solver._start_time)

    def test_mip_solver_init_empty_config(self):
        """Test MIPSolver initialization with empty config."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        self.assertEqual(solver.config, {})


class TestMIPSolverTimer(unittest.TestCase):
    """Tests for MIPSolver timer functionality."""

    def test_start_timer(self):
        """Test _start_timer sets start time."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})
        self.assertIsNone(solver._start_time)

        solver._start_timer()

        self.assertIsNotNone(solver._start_time)

    def test_get_elapsed_time_without_start(self):
        """Test _get_elapsed_time returns 0 if timer not started."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})
        elapsed = solver._get_elapsed_time()

        self.assertEqual(elapsed, 0.0)

    def test_get_elapsed_time_with_start(self):
        """Test _get_elapsed_time returns positive value after start."""
        from lp_mip_solver.mip_core import MIPSolver
        import time

        solver = MIPSolver({})
        solver._start_timer()
        time.sleep(0.01)  # Small delay
        elapsed = solver._get_elapsed_time()

        self.assertGreater(elapsed, 0.0)


class TestMIPSolverContextManager(unittest.TestCase):
    """Tests for MIPSolver context manager functionality."""

    def test_context_manager_enter(self):
        """Test __enter__ returns self."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        with solver as s:
            self.assertIs(s, solver)

    def test_context_manager_exit(self):
        """Test __exit__ returns False and calls cleanup."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})
        solver.cleanup = MagicMock()

        result = solver.__exit__(None, None, None)

        self.assertFalse(result)
        solver.cleanup.assert_called_once()

    def test_context_manager_exit_with_exception(self):
        """Test __exit__ returns False even with exception."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})
        solver.cleanup = MagicMock()

        result = solver.__exit__(ValueError, ValueError("test"), None)

        self.assertFalse(result)
        solver.cleanup.assert_called_once()


class TestMIPSolverCleanup(unittest.TestCase):
    """Tests for MIPSolver cleanup method."""

    def test_cleanup_does_not_raise(self):
        """Test cleanup method doesn't raise exceptions."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        # Should not raise
        solver.cleanup()


class TestMIPSolverDetermineMode(unittest.TestCase):
    """Tests for MIPSolver._determine_solver_mode method."""

    def test_determine_mode_or_single_spec(self):
        """Test _determine_solver_mode returns 'or' for single OR specs."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        mock_spec_handler = MagicMock()
        mock_all_specs = MagicMock()
        # or_spec_size all equal to 1 -> OR mode
        mock_all_specs.get.return_value = (
            None, None, None,
            torch.tensor([1, 1, 1]),  # or_spec_size - all 1s
            None, None
        )
        mock_spec_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'

        mode = solver._determine_solver_mode(mock_spec_handler, mock_model)

        self.assertEqual(mode, "or")

    def test_determine_mode_and_single_or(self):
        """Test _determine_solver_mode returns 'and' for single OR with multiple ANDs."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        mock_spec_handler = MagicMock()
        mock_all_specs = MagicMock()
        # Single OR with multiple ANDs -> AND mode
        mock_all_specs.get.return_value = (
            None, None, None,
            torch.tensor([3]),  # Single OR with 3 ANDs
            None, None
        )
        mock_spec_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'

        mode = solver._determine_solver_mode(mock_spec_handler, mock_model)

        self.assertEqual(mode, "and")

    def test_determine_mode_complex_spec(self):
        """Test _determine_solver_mode returns 'or' for complex specs."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        mock_spec_handler = MagicMock()
        mock_all_specs = MagicMock()
        # Multiple ORs with different AND sizes
        mock_all_specs.get.return_value = (
            None, None, None,
            torch.tensor([2, 3]),  # Different AND sizes
            None, None
        )
        mock_spec_handler.all_specs = mock_all_specs

        mock_model = MagicMock()
        mock_model.device = 'cpu'

        mode = solver._determine_solver_mode(mock_spec_handler, mock_model)

        self.assertEqual(mode, "or")

    def test_determine_mode_exception_fallback(self):
        """Test _determine_solver_mode falls back to 'or' on exception."""
        from lp_mip_solver.mip_core import MIPSolver

        solver = MIPSolver({})

        mock_spec_handler = MagicMock()
        mock_all_specs = MagicMock()
        mock_all_specs.get.side_effect = Exception("Test error")
        mock_spec_handler.all_specs = mock_all_specs

        mock_model = MagicMock()

        with patch('builtins.print'):
            mode = solver._determine_solver_mode(mock_spec_handler, mock_model)

        self.assertEqual(mode, "or")


class TestMIPSolverBuildSolverModel(unittest.TestCase):
    """Tests for MIPSolver.build_solver_model method."""

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_build_solver_model_basic(self, mock_grb, mock_arguments):
        """Test build_solver_model basic flow."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'gurobi'}}
        }

        solver = MIPSolver({})

        # Setup mock model
        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        # Setup mock gurobi model
        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        with patch('builtins.print'):
            result = solver.build_solver_model(
                mock_m, timeout=100, mip_multi_proc=4, mip_threads=2
            )

        # Should call build_solver_module
        mock_m.net.build_solver_module.assert_called_once()
        mock_gurobi_model.update.assert_called_once()

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_build_solver_model_closes_existing_pool(self, mock_grb, mock_arguments):
        """Test build_solver_model closes existing pool."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'gurobi'}}
        }

        solver = MIPSolver({})

        # Setup mock model with existing pool
        mock_pool = MagicMock()
        mock_m = MagicMock()
        mock_m.pool = mock_pool
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        with patch('builtins.print'):
            solver.build_solver_model(mock_m, timeout=100)

        # Should close, terminate, and kill pool
        mock_pool.close.assert_called_once()
        mock_pool.terminate.assert_called_once()
        mock_pool.kill.assert_called_once()
        self.assertIsNone(mock_m.pool)

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_build_solver_model_without_C(self, mock_grb, mock_arguments):
        """Test build_solver_model with include_C=False."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'gurobi'}}
        }

        solver = MIPSolver({})

        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        with patch('builtins.print'):
            solver.build_solver_model(mock_m, timeout=100, include_C=False)

        # C should be None in call
        call_args = mock_m.net.build_solver_module.call_args
        self.assertIsNone(call_args.kwargs.get('C'))

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_build_solver_model_with_model_modifier_callback(self, mock_grb, mock_arguments):
        """Test build_solver_model calls model_modifier_callback."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'gurobi'}}
        }

        solver = MIPSolver({})

        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_callback = MagicMock()

        with patch('builtins.print'):
            solver.build_solver_model(
                mock_m, timeout=100,
                model_modifier_callback=mock_callback
            )

        mock_callback.assert_called_once_with(mock_gurobi_model)

    @patch('lp_mip_solver.mip_core.arguments')
    def test_build_solver_model_unsupported_solver(self, mock_arguments):
        """Test build_solver_model raises for unsupported solver."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'unsupported'}}
        }

        solver = MIPSolver({})

        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'

        with patch('builtins.print'):
            with self.assertRaises(NotImplementedError):
                solver.build_solver_model(mock_m, timeout=100)


class TestMIPSolverSolve(unittest.TestCase):
    """Tests for MIPSolver.solve method."""

    def test_solve_calls_determine_mode(self):
        """Test solve calls _determine_solver_mode."""
        from lp_mip_solver.mip_core import MIPSolver, SolverResult, VerificationResult

        solver = MIPSolver({})
        solver._determine_solver_mode = MagicMock(return_value="or")
        solver._solve_or_mode = MagicMock(return_value=SolverResult(
            status=VerificationResult.SAFE_MIP
        ))

        mock_model = MagicMock()
        mock_spec = MagicMock()

        with patch('builtins.print'):
            solver.solve(mock_model, {}, mock_spec)

        solver._determine_solver_mode.assert_called_once()

    def test_solve_and_mode(self):
        """Test solve routes to _solve_and_mode correctly."""
        from lp_mip_solver.mip_core import MIPSolver, SolverResult, VerificationResult

        solver = MIPSolver({})
        solver._determine_solver_mode = MagicMock(return_value="and")
        solver._solve_and_mode = MagicMock(return_value=SolverResult(
            status=VerificationResult.SAFE_MIP
        ))

        mock_model = MagicMock()
        mock_spec = MagicMock()

        with patch('builtins.print'):
            result = solver.solve(mock_model, {}, mock_spec)

        solver._solve_and_mode.assert_called_once()
        self.assertEqual(result.status, VerificationResult.SAFE_MIP)

    def test_solve_or_mode(self):
        """Test solve routes to _solve_or_mode correctly."""
        from lp_mip_solver.mip_core import MIPSolver, SolverResult, VerificationResult

        solver = MIPSolver({})
        solver._determine_solver_mode = MagicMock(return_value="or")
        solver._solve_or_mode = MagicMock(return_value=SolverResult(
            status=VerificationResult.UNSAFE_MIP
        ))

        mock_model = MagicMock()
        mock_spec = MagicMock()

        with patch('builtins.print'):
            result = solver.solve(mock_model, {}, mock_spec)

        solver._solve_or_mode.assert_called_once()
        self.assertEqual(result.status, VerificationResult.UNSAFE_MIP)


class TestMIPSolverSolveAndMode(unittest.TestCase):
    """Tests for MIPSolver._solve_and_mode method."""

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_and_mode_optimal_unsafe(self, mock_grb, mock_arguments):
        """Test _solve_and_mode returns UNSAFE_MIP for optimal result."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        # Setup mock model
        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        mock_final_node.lower = torch.tensor([[0.5, -0.5]])
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        # Setup spec_handler
        mock_spec_handler = MagicMock()
        mock_spec_handler.vnnlib = [([None, None], [([None], [0.0])])]

        # Mock mip_solver_lb_ub_and in utils module (where it's imported from)
        with patch('lp_mip_solver.utils.mip_solver_lb_ub_and') as mock_mip_solver:
            mock_mip_solver.return_value = (None, None, 2, [0.5, 0.5])  # status 2 = optimal

            with patch('builtins.print'):
                result = solver._solve_and_mode(mock_model, mock_spec_handler, False, None)

        self.assertEqual(result.status, VerificationResult.UNSAFE_MIP)
        self.assertIsNotNone(result.adversarial_example)

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_and_mode_infeasible_safe(self, mock_grb, mock_arguments):
        """Test _solve_and_mode returns SAFE_MIP for infeasible result."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        mock_final_node.lower = torch.tensor([[0.5, 0.5]])
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_spec_handler = MagicMock()
        mock_spec_handler.vnnlib = [([None, None], [([None], [0.0])])]

        with patch('lp_mip_solver.utils.mip_solver_lb_ub_and') as mock_mip_solver:
            mock_mip_solver.return_value = (None, None, 3, None)  # status 3 = infeasible

            with patch('builtins.print'):
                result = solver._solve_and_mode(mock_model, mock_spec_handler, False, None)

        self.assertEqual(result.status, VerificationResult.SAFE_MIP)

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_and_mode_timeout_unknown(self, mock_grb, mock_arguments):
        """Test _solve_and_mode returns UNKNOWN_MIP for timeout."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        mock_final_node.lower = torch.tensor([[0.5, -0.5]])
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_spec_handler = MagicMock()
        mock_spec_handler.vnnlib = [([None, None], [([None], [0.0])])]

        with patch('lp_mip_solver.utils.mip_solver_lb_ub_and') as mock_mip_solver:
            mock_mip_solver.return_value = (None, None, 9, None)  # status 9 = time limit

            with patch('builtins.print'):
                result = solver._solve_and_mode(mock_model, mock_spec_handler, False, None)

        self.assertEqual(result.status, VerificationResult.UNKNOWN_MIP)


class TestMIPSolverSolveOrMode(unittest.TestCase):
    """Tests for MIPSolver._solve_or_mode method."""

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_or_mode_all_verified(self, mock_grb, mock_arguments):
        """Test _solve_or_mode returns SAFE_MIP when all verified."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        # All bounds positive - already verified
        mock_final_node.lower = torch.tensor([[0.5, 0.5]])
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_spec_handler = MagicMock()

        with patch('builtins.print'):
            result = solver._solve_or_mode(
                mock_model, mock_spec_handler, None, False, None
            )

        # No candidates to solve, but lb >= 0 for all
        self.assertEqual(result.status, VerificationResult.SAFE_MIP)

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_or_mode_finds_adversarial(self, mock_grb, mock_arguments):
        """Test _solve_or_mode returns UNSAFE_MIP when adversarial found."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult
        import lp_mip_solver.utils as utils

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        mock_final_node.lower = torch.tensor([[-0.5, -0.5]])  # Need solving
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_spec_handler = MagicMock()

        # Mock mip_solver_lb_ub to return adversarial
        def mock_mip_solver_lb_ub(*args, **kwargs):
            return (-1.0, 0.0, 2, [0.5])  # Found adversarial

        with patch.object(utils, 'mip_solver_lb_ub', mock_mip_solver_lb_ub):
            with patch('builtins.print'):
                result = solver._solve_or_mode(
                    mock_model, mock_spec_handler, None, False, None
                )

        self.assertEqual(result.status, VerificationResult.UNSAFE_MIP)

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    def test_solve_or_mode_unknown(self, mock_grb, mock_arguments):
        """Test _solve_or_mode returns UNKNOWN_MIP when inconclusive."""
        from lp_mip_solver.mip_core import MIPSolver, VerificationResult
        import lp_mip_solver.utils as utils

        mock_arguments.Config = {
            'bab': {'timeout': 100},
            'solver': {
                'mip': {
                    'parallel_solvers': 1,
                    'solver_threads': 1,
                    'mip_solver': 'gurobi',
                    'adv_warmup': False,
                    'lp_solver': False,
                    'early_stop': False,
                }
            },
            'attack': {'pgd_order': 'skip'}
        }

        solver = MIPSolver({})
        solver._start_timer()

        mock_model = MagicMock()
        mock_model.net.final_name = 'output'
        mock_model.c = torch.tensor([[1, 0]])

        mock_final_node = MagicMock()
        mock_final_node.lower = torch.tensor([[-0.5, -0.5]])
        mock_final_node.solver_vars = [MagicMock(VarName='v0'), MagicMock(VarName='v1')]
        mock_model.net.__getitem__ = MagicMock(return_value=mock_final_node)
        mock_model.net.final_node.return_value = mock_final_node
        mock_model.net.input_vars = [[MagicMock(VarName='x0')]]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        mock_spec_handler = MagicMock()

        def mock_mip_solver_lb_ub(*args, **kwargs):
            return (-0.1, 0.0, 9, None)  # Timeout, still negative lb

        with patch.object(utils, 'mip_solver_lb_ub', mock_mip_solver_lb_ub):
            with patch('builtins.print'):
                result = solver._solve_or_mode(
                    mock_model, mock_spec_handler, None, False, None
                )

        self.assertEqual(result.status, VerificationResult.UNKNOWN_MIP)


class TestSolverResultWithTensors(unittest.TestCase):
    """Tests for SolverResult with various tensor configurations."""

    def test_solver_result_with_empty_bounds(self):
        """Test SolverResult with empty bounds dictionaries."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            lower_bounds={},
            upper_bounds={}
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(result_dict['lower_bounds'], {})
        self.assertEqual(result_dict['upper_bounds'], {})

    def test_solver_result_with_multi_layer_bounds(self):
        """Test SolverResult with bounds from multiple layers."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        lower_bounds = {
            'input': torch.tensor([0.0, 0.0]),
            'hidden1': torch.tensor([-1.0, -1.0, -1.0]),
            'hidden2': torch.tensor([-0.5, -0.5]),
            'output': torch.tensor([0.1])
        }
        upper_bounds = {
            'input': torch.tensor([1.0, 1.0]),
            'hidden1': torch.tensor([1.0, 1.0, 1.0]),
            'hidden2': torch.tensor([0.5, 0.5]),
            'output': torch.tensor([0.5])
        }

        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            global_lb=torch.tensor([[0.1]]),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(len(result_dict['lower_bounds']), 4)
        self.assertEqual(len(result_dict['upper_bounds']), 4)

    def test_solver_result_with_batched_tensors(self):
        """Test SolverResult with batched tensors."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        batch_size = 4
        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            global_lb=torch.randn(batch_size, 1),
            global_ub=torch.randn(batch_size, 1),
            lower_bounds={'output': torch.randn(batch_size, 10)},
            upper_bounds={'output': torch.randn(batch_size, 10)}
        )

        self.assertEqual(result.global_lb.shape[0], batch_size)
        self.assertEqual(result.lower_bounds['output'].shape[0], batch_size)


class TestMIPSolverWithSCIP(unittest.TestCase):
    """Tests for MIPSolver with SCIP solver."""

    @patch('lp_mip_solver.mip_core.arguments')
    def test_build_solver_model_scip(self, mock_arguments):
        """Test build_solver_model with SCIP solver."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'scip'}}
        }

        solver = MIPSolver({})

        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        mock_scip_model = MagicMock()

        # SCIPModel is imported inside a try/except, so we need to patch it via sys.modules
        with patch.dict('sys.modules', {'scip_model': MagicMock(SCIPModel=MagicMock(return_value=mock_scip_model))}):
            # Need to reload the module to pick up the patched SCIPModel
            import importlib
            import lp_mip_solver.mip_core as mip_core_module

            # Patch at the module level after checking it might not exist
            with patch.object(mip_core_module, 'SCIPModel', MagicMock(return_value=mock_scip_model), create=True):
                with patch('builtins.print'):
                    solver.build_solver_model(mock_m, timeout=100)

        # Check that solver was configured
        mock_m.net.build_solver_module.assert_called_once()


class TestMIPSolverCutOptions(unittest.TestCase):
    """Tests for MIPSolver cut options handling."""

    @patch('lp_mip_solver.mip_core.arguments')
    @patch('lp_mip_solver.mip_core.grb')
    @patch.dict(os.environ, {'ALPHA_BETA_CROWN_MIP_CUT_DEBUG': 'Gomory=1,Clique=2'})
    def test_build_solver_model_with_cut_options(self, mock_grb, mock_arguments):
        """Test build_solver_model with cut options environment variable."""
        from lp_mip_solver.mip_core import MIPSolver

        mock_arguments.Config = {
            'solver': {'mip': {'mip_solver': 'gurobi'}}
        }

        solver = MIPSolver({})

        mock_m = MagicMock()
        mock_m.pool = None
        mock_m.c = torch.tensor([[1, 0]])
        mock_m.net.final_name = 'output'
        mock_m.net.build_solver_module.return_value = [MagicMock()]

        mock_gurobi_model = MagicMock()
        mock_grb.Model.return_value = mock_gurobi_model

        with patch('builtins.print'):
            solver.build_solver_model(mock_m, timeout=100)

        # Should set Cuts parameter
        calls = mock_gurobi_model.setParam.call_args_list
        param_names = [call[0][0] for call in calls]
        self.assertIn('Cuts', param_names)


class TestSolverResultRefinedBetasInLegacy(unittest.TestCase):
    """Tests for SolverResult refined_betas in legacy format."""

    def test_refined_betas_in_legacy_format(self):
        """Test refined_betas is included in legacy format."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        refined_betas = {'layer1': torch.tensor([0.1, 0.2])}

        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            global_lb=torch.tensor([[1.0]]),
            refined_betas=refined_betas
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertEqual(result_dict['refined_betas'], refined_betas)

    def test_refined_betas_none_in_legacy_format(self):
        """Test refined_betas None is preserved in legacy format."""
        from lp_mip_solver.mip_core import SolverResult, VerificationResult

        result = SolverResult(
            status=VerificationResult.SAFE_MIP,
            global_lb=torch.tensor([[1.0]])
        )

        status_str, result_dict = result.to_legacy_format()

        self.assertIsNone(result_dict['refined_betas'])


if __name__ == '__main__':
    unittest.main()
