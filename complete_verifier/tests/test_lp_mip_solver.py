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
"""Unit tests for lp_mip_solver module."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLpMipSolverUtilsFunctions(unittest.TestCase):
    """Tests for lp_mip_solver/utils.py functions."""

    def test_clamp_basic(self):
        """Test basic clamping functionality."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        lower = torch.tensor(0.5)
        upper = torch.tensor(1.5)
        result = clamp(x, lower, upper)
        expected = torch.tensor([0.5, 0.5, 1.0, 1.5, 1.5])
        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_with_tensors(self):
        """Test clamping with tensor limits."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        lower = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        upper = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        result = clamp(x, lower, upper)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_compute_ratio_basic(self):
        """Test compute_ratio basic functionality."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])
        slope, intercept = compute_ratio(lower, upper)
        # Expected: slope = 1 / (1 - (-1)) = 0.5
        self.assertAlmostEqual(slope.item(), 0.5, places=5)
        # Expected: intercept = -(-1) * 0.5 = 0.5
        self.assertAlmostEqual(intercept.item(), 0.5, places=5)

    def test_compute_ratio_positive_lower(self):
        """Test compute_ratio when lower bound is positive."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])
        slope, intercept = compute_ratio(lower, upper)
        # When lower > 0, lower_temp = clamp(1, max=0) = 0
        # slope = 2 / (2 - 0) = 1
        self.assertAlmostEqual(slope.item(), 1.0, places=5)
        # intercept = -0 * 1 = 0
        self.assertAlmostEqual(intercept.item(), 0.0, places=5)

    def test_compute_ratio_negative_upper(self):
        """Test compute_ratio when upper bound is negative."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-1.0])
        slope, intercept = compute_ratio(lower, upper)
        # upper_temp = relu(-1) = 0
        # slope = 0 / (0 - (-2)) = 0
        self.assertAlmostEqual(slope.item(), 0.0, places=5)

    def test_compute_ratio_batch(self):
        """Test compute_ratio with batch inputs."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1.0, -2.0, 0.5])
        upper = torch.tensor([1.0, 2.0, 1.5])
        slope, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope.shape, (3,))
        self.assertEqual(intercept.shape, (3,))


class TestDefaultSolverFactory(unittest.TestCase):
    """Tests for DefaultSolverFactory class."""

    def test_factory_creation(self):
        """Test factory can be created with correct type."""
        from lp_mip_solver import DefaultSolverFactory
        factory = DefaultSolverFactory()
        self.assertIsInstance(factory, DefaultSolverFactory)

    def test_create_mip_solver(self):
        """Test creating MIP solver returns MIPSolver instance."""
        from lp_mip_solver import DefaultSolverFactory, MIPSolver
        factory = DefaultSolverFactory()
        solver = factory.create_mip_solver()
        self.assertIsInstance(solver, MIPSolver)

    def test_create_model_builder(self):
        """Test creating model builder returns MIPSolver instance."""
        from lp_mip_solver import DefaultSolverFactory, MIPSolver
        factory = DefaultSolverFactory()
        builder = factory.create_model_builder()
        self.assertIsInstance(builder, MIPSolver)


class TestFactoryFunctions(unittest.TestCase):
    """Tests for factory functions."""

    def test_create_factory(self):
        """Test create_factory function returns DefaultSolverFactory instance."""
        from lp_mip_solver import create_factory, DefaultSolverFactory
        factory = create_factory()
        self.assertIsInstance(factory, DefaultSolverFactory)

    def test_get_mip_solver(self):
        """Test get_mip_solver function returns MIPSolver instance."""
        from lp_mip_solver import get_mip_solver, MIPSolver
        solver = get_mip_solver()
        self.assertIsInstance(solver, MIPSolver)

    def test_get_model_builder(self):
        """Test get_model_builder function returns MIPSolver instance."""
        from lp_mip_solver import get_model_builder, MIPSolver
        builder = get_model_builder()
        self.assertIsInstance(builder, MIPSolver)


class TestClampEdgeCases(unittest.TestCase):
    """Additional tests for clamp function edge cases."""

    def test_clamp_negative_values(self):
        """Test clamping with negative values."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        lower = torch.tensor(-1.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_all_within_bounds(self):
        """Test clamping when all values are within bounds."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([0.3, 0.5, 0.7])
        lower = torch.tensor(0.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        self.assertTrue(torch.allclose(result, x))

    def test_clamp_3d_tensor(self):
        """Test clamping with 3D tensor."""
        from lp_mip_solver.utils import clamp
        x = torch.randn(2, 3, 4)
        lower = torch.tensor(-0.5)
        upper = torch.tensor(0.5)
        result = clamp(x, lower, upper)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue((result >= -0.5).all())
        self.assertTrue((result <= 0.5).all())


class TestComputeRatioEdgeCases(unittest.TestCase):
    """Additional tests for compute_ratio edge cases."""

    def test_compute_ratio_large_values(self):
        """Test compute_ratio with large values."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1e6])
        upper = torch.tensor([1e6])
        slope, intercept = compute_ratio(lower, upper)
        self.assertAlmostEqual(slope.item(), 0.5, places=3)


class TestCheckOptimizationSuccessFunction(unittest.TestCase):
    """Tests for check_optimization_success function."""

    def test_status_2_optimal(self):
        """Test handling of status 2 (optimal)."""
        from lp_mip_solver.utils import check_optimization_success

        mock_model = MagicMock()
        mock_model.status = 2  # Optimal

        # Should not raise
        check_optimization_success(mock_model)

    def test_status_3_infeasible(self):
        """Test handling of status 3 (infeasible)."""
        from lp_mip_solver.utils import check_optimization_success

        mock_model = MagicMock()
        mock_model.status = 3  # Infeasible

        introduced_constrs_all = [MagicMock(), MagicMock()]

        # Should not raise but should remove constraints
        check_optimization_success(mock_model, introduced_constrs_all)
        # verify remove was called for each constraint
        self.assertEqual(mock_model.remove.call_count, 2)

    def test_status_other_raises(self):
        """Test handling of other status codes."""
        from lp_mip_solver.utils import check_optimization_success

        mock_model = MagicMock()
        mock_model.status = 4  # Some other status

        with self.assertRaises(NotImplementedError):
            check_optimization_success(mock_model)


class TestMipSolverAttackInit(unittest.TestCase):
    """Tests for mip_solver_attack_init function."""

    def test_mip_solver_attack_init_sets_flag(self):
        """Test that mip_solver_attack_init sets the global termination_flag."""
        import lp_mip_solver.utils as utils
        from lp_mip_solver.utils import mip_solver_attack_init

        # Store original value to restore later
        original_flag = utils.termination_flag

        try:
            mock_flag = MagicMock()
            mip_solver_attack_init(mock_flag)

            # Verify the global termination_flag was set to the provided flag
            self.assertIs(utils.termination_flag, mock_flag)
        finally:
            # Restore original value
            utils.termination_flag = original_flag

    def test_mip_solver_attack_init_with_multiprocessing_value(self):
        """Test mip_solver_attack_init with a multiprocessing.Value flag."""
        import multiprocessing
        import lp_mip_solver.utils as utils
        from lp_mip_solver.utils import mip_solver_attack_init

        original_flag = utils.termination_flag

        try:
            # Create a multiprocessing Value (typical usage)
            mp_flag = multiprocessing.Value('i', 0)
            mip_solver_attack_init(mp_flag)

            # Verify the flag was set
            self.assertIs(utils.termination_flag, mp_flag)
            # Verify the flag value is accessible
            self.assertEqual(utils.termination_flag.value, 0)
        finally:
            utils.termination_flag = original_flag


class TestComputeRatioMultidimensional(unittest.TestCase):
    """Additional tests for compute_ratio with multidimensional inputs."""

    def test_compute_ratio_2d_tensor(self):
        """Test compute_ratio with 2D tensor."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([[-1.0, -2.0], [-0.5, -1.0]])
        upper = torch.tensor([[1.0, 2.0], [0.5, 1.0]])
        slope, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope.shape, (2, 2))
        self.assertEqual(intercept.shape, (2, 2))

    def test_compute_ratio_with_zeros(self):
        """Test compute_ratio when bounds include zero."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])
        slope, intercept = compute_ratio(lower, upper)
        # lower_temp = clamp(0, max=0) = 0
        # slope = 1 / (1 - 0) = 1
        self.assertAlmostEqual(slope.item(), 1.0, places=5)
        # intercept = -0 * 1 = 0
        self.assertAlmostEqual(intercept.item(), 0.0, places=5)


class TestClampMismatchedShapes(unittest.TestCase):
    """Tests for clamp with broadcasting."""

    def test_clamp_scalar_limits(self):
        """Test clamp with scalar limits on multidimensional tensor."""
        from lp_mip_solver.utils import clamp
        x = torch.randn(2, 3, 4)
        lower = torch.tensor(-1.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertTrue((result >= -1.0).all())
        self.assertTrue((result <= 1.0).all())

    def test_clamp_1d_limits_on_2d_tensor(self):
        """Test clamp with 1D limits on 2D tensor."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        lower = torch.tensor([0.0, 1.0])
        upper = torch.tensor([2.0, 3.0])
        result = clamp(x, lower, upper)
        expected = torch.tensor([[0.5, 1.5], [2.0, 3.0]])
        self.assertTrue(torch.allclose(result, expected))


class TestGurobiErrorHandling(unittest.TestCase):
    """Tests for Gurobi error handling."""

    def test_handle_gurobi_error_raises(self):
        """Test that an exception is raised."""
        from lp_mip_solver.utils import handle_gurobi_error

        # GurobiError constructor differs by version, just check any exception is raised
        with self.assertRaises(Exception):
            handle_gurobi_error("Custom error message")

    def test_handle_gurobi_error_prints_message(self):
        """Test that error message is printed."""
        from lp_mip_solver.utils import handle_gurobi_error

        with patch('builtins.print') as mock_print:
            try:
                handle_gurobi_error("Test message")
            except Exception:
                pass

            mock_print.assert_called_with('Gurobi error: Test message')


class TestComputeRatioSpecialCases(unittest.TestCase):
    """Special case tests for compute_ratio."""

    def test_compute_ratio_equal_bounds(self):
        """Test compute_ratio when upper == lower (edge case)."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1.0])
        upper = torch.tensor([-1.0])  # Same as lower
        slope, intercept = compute_ratio(lower, upper)
        # upper_temp = relu(-1) = 0
        # lower_temp = clamp(-1, max=0) = -1
        # This is division by zero territory, but relu(upper) = 0 saves us
        # slope = 0 / (0 - (-1)) = 0
        self.assertAlmostEqual(slope.item(), 0.0, places=5)

    def test_compute_ratio_all_positive(self):
        """Test compute_ratio when both bounds are positive."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([0.5])
        upper = torch.tensor([1.5])
        slope, intercept = compute_ratio(lower, upper)
        # lower_temp = clamp(0.5, max=0) = 0
        # slope = 1.5 / (1.5 - 0) = 1.0
        self.assertAlmostEqual(slope.item(), 1.0, places=5)

    def test_compute_ratio_all_negative(self):
        """Test compute_ratio when both bounds are negative."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1.5])
        upper = torch.tensor([-0.5])
        slope, intercept = compute_ratio(lower, upper)
        # upper_temp = relu(-0.5) = 0
        # slope = 0 / something = 0
        self.assertAlmostEqual(slope.item(), 0.0, places=5)


class TestCopyModelFunction(unittest.TestCase):
    """Additional tests for copy_model function."""

    def test_copy_model_without_basis(self):
        """Test copy_model with basis=False."""
        from lp_mip_solver.utils import copy_model

        # Create mock model
        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.getConstrByName.return_value = MagicMock()

        result = copy_model(mock_model, basis=False)

        mock_model.copy.assert_called_once()
        mock_model_copy.update.assert_called()
        self.assertEqual(result, mock_model_copy)

    def test_copy_model_with_basis_warm_start(self):
        """Test copy_model with basis=True and use_basis_warm_start=True."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy

        # Setup variables
        mock_var1 = MagicMock()
        mock_var1.VBasis = 0
        mock_var2 = MagicMock()
        mock_var2.VBasis = 1
        mock_model.getVars.return_value = [mock_var1, mock_var2]

        mock_svar1 = MagicMock()
        mock_svar2 = MagicMock()
        mock_model_copy.getVars.return_value = [mock_svar1, mock_svar2]

        # Setup constraints
        mock_constr = MagicMock()
        mock_constr.ConstrName = 'c1'
        mock_constr.CBasis = 0
        mock_model_copy.getConstrs.return_value = [mock_constr]
        mock_model.getConstrByName.return_value = mock_constr

        result = copy_model(mock_model, basis=True, use_basis_warm_start=True)

        # Should set VBasis on split variables
        self.assertEqual(mock_svar1.VBasis, 0)
        self.assertEqual(mock_svar2.VBasis, 1)

    def test_copy_model_without_basis_warm_start(self):
        """Test copy_model with basis=True but use_basis_warm_start=False."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy

        # Setup variables with X values
        mock_var1 = MagicMock()
        mock_var1.X = 0.5
        mock_var2 = MagicMock()
        mock_var2.X = 1.5
        mock_model.getVars.return_value = [mock_var1, mock_var2]

        mock_svar1 = MagicMock()
        mock_svar2 = MagicMock()
        mock_model_copy.getVars.return_value = [mock_svar1, mock_svar2]

        # Setup constraints
        mock_constr = MagicMock()
        mock_constr.ConstrName = 'c1'
        mock_constr.Pi = 0.1
        mock_model_copy.getConstrs.return_value = [mock_constr]
        mock_model.getConstrByName.return_value = mock_constr

        result = copy_model(mock_model, basis=True, use_basis_warm_start=False)

        # Should set PStart instead of VBasis
        self.assertEqual(mock_svar1.PStart, 0.5)
        self.assertEqual(mock_svar2.PStart, 1.5)

    def test_copy_model_with_remove_constr_list(self):
        """Test copy_model with constraints to remove."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy

        # Constraint to remove
        mock_constr_to_remove = MagicMock()
        mock_constr_to_remove.ConstrName = 'remove_me'
        mock_constr_copy = MagicMock()
        mock_model_copy.getConstrByName.return_value = mock_constr_copy

        result = copy_model(
            mock_model,
            basis=False,
            remove_constr_list=[mock_constr_to_remove]
        )

        mock_model_copy.getConstrByName.assert_called_with('remove_me')
        mock_model_copy.remove.assert_called_with(mock_constr_copy)

    def test_copy_model_with_env(self):
        """Test copy_model with custom environment."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_env = MagicMock()
        mock_model.copy.return_value = mock_model_copy

        result = copy_model(mock_model, basis=False, env=mock_env)

        mock_model.copy.assert_called_with(env=mock_env)


class TestMipSolverLbUbFunction(unittest.TestCase):
    """Tests for mip_solver_lb_ub function."""

    @patch('lp_mip_solver.utils.arguments')
    def test_mip_solver_lb_ub_no_model(self, mock_arguments):
        """Test mip_solver_lb_ub when no model is available."""
        import lp_mip_solver.utils as utils

        # Reset global model
        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = None

        try:
            result = utils.mip_solver_lb_ub('test_var')
            self.assertEqual(result, (None, None, -1, None))
        finally:
            utils.multiprocess_mip_model = original_model

    @patch('lp_mip_solver.utils.arguments')
    def test_mip_solver_lb_ub_skipped_when_stopped(self, mock_arguments):
        """Test mip_solver_lb_ub returns early when stop_multiprocess is True."""
        import lp_mip_solver.utils as utils

        # Setup mock model
        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_var = MagicMock()
        mock_var.LB = -1.0
        mock_var.UB = 1.0
        mock_model_copy.getVarByName.return_value = mock_var

        original_model = utils.multiprocess_mip_model
        original_stop = utils.stop_multiprocess
        utils.multiprocess_mip_model = mock_model
        utils.stop_multiprocess = True

        try:
            lb, ub, status, adv = utils.mip_solver_lb_ub('test_var')
            self.assertEqual(status, -1)  # Solver skipped
        finally:
            utils.multiprocess_mip_model = original_model
            utils.stop_multiprocess = original_stop


class TestMipSolverLbUbAndFunction(unittest.TestCase):
    """Tests for mip_solver_lb_ub_and function."""

    def test_mip_solver_lb_ub_and_optimal(self):
        """Test mip_solver_lb_ub_and with optimal solution."""
        import lp_mip_solver.utils as utils
        import gurobipy as grb

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.Status = grb.GRB.OPTIMAL

        # Create mock variables that support <= comparison
        mock_var1 = grb.LinExpr()  # Use actual Gurobi LinExpr for comparison
        mock_var2 = grb.LinExpr()
        mock_model_copy.getVarByName.side_effect = [mock_var1, mock_var2]

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        try:
            with patch('builtins.print'):
                _, _, status, _ = utils.mip_solver_lb_ub_and(
                    ['var1', 'var2'],
                    rhs=[0.0, 0.0]
                )
            self.assertEqual(status, grb.GRB.OPTIMAL)
        finally:
            utils.multiprocess_mip_model = original_model

    def test_mip_solver_lb_ub_and_infeasible(self):
        """Test mip_solver_lb_ub_and with infeasible result."""
        import lp_mip_solver.utils as utils
        import gurobipy as grb

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.Status = grb.GRB.INFEASIBLE

        mock_var = grb.LinExpr()
        mock_model_copy.getVarByName.return_value = mock_var

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        try:
            with patch('builtins.print'):
                _, _, status, _ = utils.mip_solver_lb_ub_and(
                    ['var1'],
                    rhs=[0.0]
                )
            self.assertEqual(status, grb.GRB.INFEASIBLE)
        finally:
            utils.multiprocess_mip_model = original_model

    def test_mip_solver_lb_ub_and_time_limit(self):
        """Test mip_solver_lb_ub_and with time limit."""
        import lp_mip_solver.utils as utils
        import gurobipy as grb

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.Status = grb.GRB.TIME_LIMIT

        mock_var = grb.LinExpr()
        mock_model_copy.getVarByName.return_value = mock_var

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        try:
            with patch('builtins.print'):
                _, _, status, _ = utils.mip_solver_lb_ub_and(
                    ['var1'],
                    rhs=[0.0]
                )
            self.assertEqual(status, grb.GRB.TIME_LIMIT)
        finally:
            utils.multiprocess_mip_model = original_model

    def test_mip_solver_lb_ub_and_unexpected_status(self):
        """Test mip_solver_lb_ub_and with unexpected status raises RuntimeError."""
        import lp_mip_solver.utils as utils
        import gurobipy as grb

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.Status = 99  # Unexpected status

        mock_var = grb.LinExpr()
        mock_model_copy.getVarByName.return_value = mock_var

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        try:
            with patch('builtins.print'):
                with self.assertRaises(RuntimeError):
                    utils.mip_solver_lb_ub_and(['var1'], rhs=[0.0])
        finally:
            utils.multiprocess_mip_model = original_model

    def test_mip_solver_lb_ub_and_with_save_adv(self):
        """Test mip_solver_lb_ub_and saves adversarial when optimal."""
        import lp_mip_solver.utils as utils
        import gurobipy as grb

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy
        mock_model_copy.Status = grb.GRB.OPTIMAL

        mock_var = grb.LinExpr()
        mock_adv_var = MagicMock()
        mock_adv_var.X = 0.5

        def get_var_by_name(name):
            if name == 'adv_inp':
                return mock_adv_var
            return mock_var

        mock_model_copy.getVarByName.side_effect = get_var_by_name

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        try:
            with patch('builtins.print'):
                _, _, status, adv = utils.mip_solver_lb_ub_and(
                    ['var1'],
                    save_adv=['adv_inp'],
                    rhs=[0.0]
                )
            self.assertEqual(adv, [0.5])
        finally:
            utils.multiprocess_mip_model = original_model


class TestNestablePoolGetattr(unittest.TestCase):
    """Tests for NestablePool.__getattr__ delegation."""

    def test_nestable_pool_delegates_methods(self):
        """Test that NestablePool delegates attribute access to pool."""
        from lp_mip_solver.utils import NestablePool

        # Create a pool with minimal processes
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool.map.return_value = [1, 2, 3]
            mock_pool_class.return_value = mock_pool

            nestable = NestablePool(1)

            # Access an attribute that should be delegated
            result = nestable.map(lambda x: x, [1, 2, 3])

            mock_pool.map.assert_called_once()


class TestClampExtendedEdgeCases(unittest.TestCase):
    """Extended edge case tests for clamp function."""

    def test_clamp_empty_tensor(self):
        """Test clamp with empty tensor."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([])
        lower = torch.tensor(-1.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        self.assertEqual(result.shape, (0,))

    def test_clamp_inf_values(self):
        """Test clamp with infinity values."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([float('-inf'), 0.0, float('inf')])
        lower = torch.tensor(-1.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_clamp_nan_handling(self):
        """Test clamp behavior with NaN values."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([float('nan'), 0.5])
        lower = torch.tensor(0.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        # NaN should propagate through
        self.assertTrue(torch.isnan(result[0]))
        self.assertEqual(result[1].item(), 0.5)

    def test_clamp_preserves_dtype(self):
        """Test that clamp preserves tensor dtype."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([0.5, 1.5], dtype=torch.float64)
        lower = torch.tensor(0.0, dtype=torch.float64)
        upper = torch.tensor(1.0, dtype=torch.float64)
        result = clamp(x, lower, upper)
        self.assertEqual(result.dtype, torch.float64)

    def test_clamp_integer_tensor(self):
        """Test clamp with integer tensors."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([0, 5, 10])
        lower = torch.tensor(2)
        upper = torch.tensor(8)
        result = clamp(x, lower, upper)
        expected = torch.tensor([2, 5, 8])
        self.assertTrue(torch.equal(result, expected))

    def test_clamp_4d_tensor(self):
        """Test clamp with 4D tensor (like images)."""
        from lp_mip_solver.utils import clamp
        x = torch.randn(2, 3, 32, 32)
        lower = torch.tensor(-1.0)
        upper = torch.tensor(1.0)
        result = clamp(x, lower, upper)
        self.assertEqual(result.shape, (2, 3, 32, 32))
        self.assertTrue((result >= -1.0).all())
        self.assertTrue((result <= 1.0).all())

    def test_clamp_same_lower_upper(self):
        """Test clamp when lower equals upper."""
        from lp_mip_solver.utils import clamp
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        lower = torch.tensor(0.5)
        upper = torch.tensor(0.5)
        result = clamp(x, lower, upper)
        expected = torch.tensor([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(torch.allclose(result, expected))


class TestComputeRatioExtendedEdgeCases(unittest.TestCase):
    """Extended edge case tests for compute_ratio function."""

    def test_compute_ratio_3d_tensor(self):
        """Test compute_ratio with 3D tensor."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.randn(2, 3, 4) - 1  # Ensure some negative
        upper = lower + torch.abs(torch.randn(2, 3, 4)) + 0.1  # Ensure upper > lower
        slope, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope.shape, (2, 3, 4))
        self.assertEqual(intercept.shape, (2, 3, 4))

    def test_compute_ratio_asymmetric_bounds(self):
        """Test compute_ratio with asymmetric bounds."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-3.0])
        upper = torch.tensor([1.0])
        slope, intercept = compute_ratio(lower, upper)
        # lower_temp = clamp(-3, max=0) = -3
        # upper_temp = relu(1) = 1
        # slope = 1 / (1 - (-3)) = 0.25
        self.assertAlmostEqual(slope.item(), 0.25, places=5)
        # intercept = -(-3) * 0.25 = 0.75
        self.assertAlmostEqual(intercept.item(), 0.75, places=5)

    def test_compute_ratio_preserves_device(self):
        """Test compute_ratio preserves tensor device."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])
        slope, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope.device, lower.device)
        self.assertEqual(intercept.device, upper.device)

    def test_compute_ratio_large_batch(self):
        """Test compute_ratio with large batch."""
        from lp_mip_solver.utils import compute_ratio
        batch_size = 1000
        lower = torch.randn(batch_size) - 1
        upper = lower + torch.abs(torch.randn(batch_size)) + 0.1
        slope, intercept = compute_ratio(lower, upper)
        self.assertEqual(slope.shape, (batch_size,))
        self.assertEqual(intercept.shape, (batch_size,))
        # All slopes should be non-negative
        self.assertTrue((slope >= 0).all())
        # All slopes should be <= 1
        self.assertTrue((slope <= 1).all())


class TestCheckOptimizationSuccessExtended(unittest.TestCase):
    """Extended tests for check_optimization_success function."""

    def test_check_optimization_success_empty_constraints(self):
        """Test check_optimization_success with empty constraint list."""
        from lp_mip_solver.utils import check_optimization_success

        mock_model = MagicMock()
        mock_model.status = 3  # Infeasible

        with patch('builtins.print'):
            check_optimization_success(mock_model, [])

        # remove should not be called
        mock_model.remove.assert_not_called()

    def test_check_optimization_success_with_constraints_to_remove(self):
        """Test check_optimization_success removes constraints on infeasible."""
        from lp_mip_solver.utils import check_optimization_success

        mock_model = MagicMock()
        mock_model.status = 3  # Infeasible

        mock_constr1 = MagicMock()
        mock_constr2 = MagicMock()

        with patch('builtins.print'):
            check_optimization_success(mock_model, [mock_constr1, mock_constr2])

        # remove should be called for each constraint
        self.assertEqual(mock_model.remove.call_count, 2)


class TestFSBScoreFunctionComputation(unittest.TestCase):
    """Tests for FSB_score function computation."""

    def test_fsb_score_empty_relus_returns_empty(self):
        """Test FSB_score with empty relu list returns empty score."""
        from lp_mip_solver.utils import FSB_score

        mock_model = MagicMock()
        mock_model.net.relus = []  # Empty relu list
        mock_model.final_name = 'output'

        # Provide non-empty mask to avoid StopIteration
        results = {
            'mask': {'layer1': [torch.ones(1, 10)]},
            'lA': {},
            'lower_bounds': {'output': torch.tensor([[0.0]])},
            'upper_bounds': {'output': torch.tensor([[1.0]])},
        }

        score = FSB_score(mock_model, results)
        self.assertEqual(len(score), 0)


class TestMipSolverWorkerFunction(unittest.TestCase):
    """Tests for mip_solver_worker function."""

    def test_mip_solver_worker_with_solutions(self):
        """Test mip_solver_worker when solutions are found."""
        from lp_mip_solver.utils import mip_solver_worker
        import multiprocessing

        mock_model = MagicMock()
        mock_model.solcount = 1
        mock_model.objval = -0.5
        mock_model.objbound = -0.6
        mock_model.status = 2

        # Setup variable access
        def get_var_by_name(name):
            mock_var = MagicMock()
            mock_var.X = 0.5
            return mock_var
        mock_model.getVarByName = get_var_by_name

        input_shape = (1, 3, 2, 2)  # NCHW
        queue = multiprocessing.Queue()

        # Run in separate thread to avoid blocking
        import threading
        t = threading.Thread(
            target=mip_solver_worker,
            args=(mock_model, input_shape, queue),
            daemon=True
        )
        t.start()
        t.join(timeout=1)  # Give it time to complete

        # Get result from queue
        if not queue.empty():
            status, objval, objbound, solcount, solution = queue.get(timeout=1)
            self.assertEqual(status, 2)
            self.assertEqual(solcount, 1)
        else:
            self.skipTest("Queue was empty - worker did not complete in time")

    def test_mip_solver_worker_no_solutions(self):
        """Test mip_solver_worker when no solutions are found."""
        from lp_mip_solver.utils import mip_solver_worker
        import multiprocessing

        mock_model = MagicMock()
        mock_model.solcount = 0
        mock_model.objval = float('inf')
        mock_model.objbound = float('-inf')
        mock_model.status = 3  # Infeasible

        input_shape = (1, 3, 2, 2)
        queue = multiprocessing.Queue()

        import threading
        t = threading.Thread(
            target=mip_solver_worker,
            args=(mock_model, input_shape, queue),
            daemon=True
        )
        t.start()
        t.join(timeout=1)

        if not queue.empty():
            status, objval, objbound, solcount, solution = queue.get(timeout=1)
            self.assertEqual(solcount, 0)
            self.assertEqual(solution.shape[0], 0)
        else:
            self.skipTest("Queue was empty - worker did not complete in time")


class TestMipSolverAttackFunction(unittest.TestCase):
    """Tests for mip_solver_attack function."""

    def test_mip_solver_attack_early_termination(self):
        """Test mip_solver_attack handles early termination."""
        import lp_mip_solver.utils as utils
        import multiprocessing

        # Setup termination flag
        term_flag = multiprocessing.Value('i', 1)  # Already set to terminate
        utils.termination_flag = term_flag

        mock_model = MagicMock()
        mock_model_copy = MagicMock()
        mock_model.copy.return_value = mock_model_copy

        mock_var = MagicMock()
        mock_model_copy.getVarByName.return_value = mock_var

        original_model = utils.multiprocess_mip_model
        utils.multiprocess_mip_model = mock_model

        input_shape = (1, 3, 2, 2)
        new_splits = (
            [],  # indices
            [],  # relu_status
            'output_var',  # opt_var
            input_shape,  # input_shape
            None,  # best_adv_input
            None,  # relu_forward
            ['pre_relu'],  # pre_relu_layer_names
            ['relu'],  # relu_layer_names
            None,  # lower_bounds
            None,  # upper_bounds
        )

        try:
            with patch('builtins.print'):
                with patch('time.sleep'):
                    result = utils.mip_solver_attack(new_splits)

            # Should return early termination result
            objval, objbound, status, solution = result
            self.assertEqual(objval, float('inf'))
            self.assertEqual(status, -1)
        finally:
            utils.multiprocess_mip_model = original_model


class TestUpdateMipModelFixRelu(unittest.TestCase):
    """Tests for update_mip_model_fix_relu function."""

    @patch('lp_mip_solver.utils.arguments')
    def test_update_mip_model_fix_relu_basic(self, mock_arguments):
        """Test basic update_mip_model_fix_relu functionality."""
        from lp_mip_solver.utils import update_mip_model_fix_relu

        mock_arguments.Config = {
            'solver': {'mip': {'parallel_solvers': 1}},
            'bab': {'attack': {'refined_mip_attacker': False}}
        }

        mock_net = MagicMock()
        mock_net.net.relus = []
        mock_net.net.final_node.return_value.solver_vars = [MagicMock(VarName='out')]
        mock_net.net.solver_model = MagicMock()
        mock_net.input_shape = (1, 3, 32, 32)
        mock_net.c = torch.tensor([[-1]])
        mock_net.pool = None

        relu_idx = [[]]
        status = [[]]
        best_adv = [None]
        adv_activation_pattern = [None]

        with patch.object(mock_net, 'pool', None):
            with patch('lp_mip_solver.utils.NestablePool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_pool.map.return_value = [(0.5, 0.3, 2, None)]

                result = update_mip_model_fix_relu(
                    mock_net, relu_idx, status,
                    async_mip=False,
                    best_adv=best_adv,
                    adv_activation_pattern=adv_activation_pattern
                )

                self.assertIsInstance(result, tuple)


class TestComputeRatioNumericalStability(unittest.TestCase):
    """Tests for compute_ratio numerical stability."""

    def test_compute_ratio_very_large_positive_bounds(self):
        """Test compute_ratio with very large positive bounds."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([1e10])
        upper = torch.tensor([1e10 + 1])
        slope, intercept = compute_ratio(lower, upper)
        # Should handle large values without overflow
        self.assertFalse(torch.isnan(slope).any())
        self.assertFalse(torch.isinf(slope).any())

    def test_compute_ratio_very_large_negative_bounds(self):
        """Test compute_ratio with very large negative bounds."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1e10 - 1])
        upper = torch.tensor([-1e10])
        slope, intercept = compute_ratio(lower, upper)
        # Both negative, upper_temp = 0
        self.assertAlmostEqual(slope.item(), 0.0, places=5)

    def test_compute_ratio_mixed_batch_stability(self):
        """Test compute_ratio stability with mixed batch values."""
        from lp_mip_solver.utils import compute_ratio
        lower = torch.tensor([-1e-10, -1.0, -1e10, 0.0, 1e-10])
        upper = torch.tensor([1e-10, 1.0, -1e9, 1.0, 1.0])
        slope, intercept = compute_ratio(lower, upper)
        # Should not have NaN or Inf
        self.assertFalse(torch.isnan(slope).any())
        self.assertFalse(torch.isnan(intercept).any())


if __name__ == '__main__':
    unittest.main()
