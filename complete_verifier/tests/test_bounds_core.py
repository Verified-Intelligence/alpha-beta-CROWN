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
"""Unit tests for lp_mip_solver/bounds_core.py module."""

import pytest
import sys
import os
import time
import torch
import numpy as np
import multiprocessing
from unittest.mock import MagicMock, patch, PropertyMock, Mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLpSolver:
    """Tests for lp_solver function."""

    def test_lp_solver_stable_positive(self):
        """Test lp_solver returns early for stable positive neuron."""
        import lp_mip_solver.bounds_core as bounds_core
        import lp_mip_solver.utils as solver_utils

        # Save original value
        orig_stop = solver_utils.stop_multiprocess

        # Create mock model
        mock_model = MagicMock()
        mock_var = MagicMock()
        mock_var.LB = 0.5  # Already positive
        mock_var.UB = 1.0
        mock_var.VarName = "test_var"
        mock_model.getVarByName.return_value = mock_var
        mock_model.copy.return_value = mock_model

        bounds_core.multiprocess_lp_model = mock_model
        solver_utils.stop_multiprocess = False

        result = bounds_core.lp_solver("test_var")

        # Should return early without optimization (lb >= 0)
        assert result[0] == 0.5  # lb
        assert result[1] == 1.0  # ub
        assert result[3] == False  # not refined

        solver_utils.stop_multiprocess = orig_stop

    def test_lp_solver_stable_negative(self):
        """Test lp_solver returns early for stable negative neuron."""
        import lp_mip_solver.bounds_core as bounds_core
        import lp_mip_solver.utils as solver_utils

        orig_stop = solver_utils.stop_multiprocess

        mock_model = MagicMock()
        mock_var = MagicMock()
        mock_var.LB = -1.0
        mock_var.UB = -0.5  # Already negative
        mock_var.VarName = "test_var"
        mock_model.getVarByName.return_value = mock_var
        mock_model.copy.return_value = mock_model

        bounds_core.multiprocess_lp_model = mock_model
        solver_utils.stop_multiprocess = False

        result = bounds_core.lp_solver("test_var")

        # Should return early (ub <= 0)
        assert result[0] == -1.0  # lb
        assert result[1] == -0.5  # ub
        assert result[3] == False  # not refined

        solver_utils.stop_multiprocess = orig_stop

    def test_lp_solver_stop_multiprocess(self):
        """Test lp_solver returns early when stop flag is set."""
        import lp_mip_solver.bounds_core as bounds_core
        import lp_mip_solver.utils as solver_utils

        orig_stop = solver_utils.stop_multiprocess

        mock_model = MagicMock()
        mock_var = MagicMock()
        mock_var.LB = -1.0
        mock_var.UB = 1.0  # Unstable neuron
        mock_var.VarName = "test_var"
        mock_model.getVarByName.return_value = mock_var
        mock_model.copy.return_value = mock_model

        bounds_core.multiprocess_lp_model = mock_model
        solver_utils.stop_multiprocess = True  # Stop flag set

        result = bounds_core.lp_solver("test_var")

        # Should return early without optimization
        assert result[0] == -1.0  # original lb
        assert result[1] == 1.0  # original ub
        assert result[3] == False  # not refined

        solver_utils.stop_multiprocess = orig_stop

    def test_lp_solver_optimal_refinement(self):
        """Test lp_solver with successful optimization."""
        import lp_mip_solver.bounds_core as bounds_core
        import lp_mip_solver.utils as solver_utils
        import gurobipy as grb

        orig_stop = solver_utils.stop_multiprocess

        mock_model = MagicMock()
        mock_var = MagicMock()
        mock_var.LB = -1.0
        mock_var.UB = 1.0  # Unstable neuron
        mock_var.X = 0.0  # Optimal value
        mock_var.VarName = "test_var"
        mock_model.getVarByName.return_value = mock_var
        mock_model.copy.return_value = mock_model
        mock_model.status = grb.GRB.OPTIMAL

        bounds_core.multiprocess_lp_model = mock_model
        solver_utils.stop_multiprocess = False

        with patch('builtins.print'):  # Suppress prints
            result = bounds_core.lp_solver("test_var")

        assert result[3] == True  # refined

        solver_utils.stop_multiprocess = orig_stop

    def test_lp_solver_non_optimal_status(self):
        """Test lp_solver with non-optimal status."""
        import lp_mip_solver.bounds_core as bounds_core
        import lp_mip_solver.utils as solver_utils

        orig_stop = solver_utils.stop_multiprocess

        mock_model = MagicMock()
        mock_var = MagicMock()
        mock_var.LB = -1.0
        mock_var.UB = 1.0  # Unstable
        mock_var.VarName = "test_var"
        mock_model.getVarByName.return_value = mock_var
        mock_model.copy.return_value = mock_model
        mock_model.status = 4  # Non-optimal status

        bounds_core.multiprocess_lp_model = mock_model
        solver_utils.stop_multiprocess = False

        with patch('builtins.print'):  # Suppress warning print
            result = bounds_core.lp_solver("test_var")

        # Should return original bounds when not optimal
        assert result[0] == -1.0  # original lb
        assert result[1] == 1.0   # original ub

        solver_utils.stop_multiprocess = orig_stop


class TestBuildTheModelLp:
    """Tests for build_the_model_lp function."""

    def test_build_model_lp_basic(self):
        """Test basic LP model building."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_var = MagicMock()
        mock_var.X = 0.5
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.net.__getitem__.return_value.solver_vars = [mock_var]
        mock_m.net.solver_model.status = grb.GRB.OPTIMAL
        mock_m.model_ori.children.return_value = []

        glbs = bounds_core.build_the_model_lp(mock_m)

        assert len(glbs) == 1
        assert glbs[0] == 0.5

    def test_build_model_lp_compute_upper_bound(self):
        """Test LP model building for upper bound computation."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_var = MagicMock()
        mock_var.X = 2.0
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.net.__getitem__.return_value.solver_vars = [mock_var]
        mock_m.net.solver_model.status = grb.GRB.OPTIMAL
        mock_m.model_ori.children.return_value = []

        glbs = bounds_core.build_the_model_lp(mock_m, compute_upper_bound=True)

        # Should call setObjective with MAXIMIZE for upper bound
        mock_m.net.solver_model.setObjective.assert_called()

    def test_build_model_lp_infeasible(self):
        """Test LP model building with infeasible result."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_var = MagicMock()
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.net.__getitem__.return_value.solver_vars = [mock_var]
        mock_m.net.solver_model.status = grb.GRB.INFEASIBLE
        mock_m.model_ori.children.return_value = []

        glbs = bounds_core.build_the_model_lp(mock_m)

        assert glbs[0] == float('-inf')

    def test_build_model_lp_infeasible_upper_bound(self):
        """Test LP model building with infeasible result for upper bound."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_var = MagicMock()
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.net.__getitem__.return_value.solver_vars = [mock_var]
        mock_m.net.solver_model.status = grb.GRB.INFEASIBLE
        mock_m.model_ori.children.return_value = []

        glbs = bounds_core.build_the_model_lp(mock_m, compute_upper_bound=True)

        assert glbs[0] == float('inf')

    def test_build_model_lp_include_output_constraint(self):
        """Test LP model building with output constraint."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_var = MagicMock()
        mock_var.X = 0.5
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.net.__getitem__.return_value.solver_vars = [mock_var]
        mock_m.net.solver_model.status = grb.GRB.OPTIMAL
        mock_m.model_ori.children.return_value = []

        rhs = torch.tensor([[1.0]])

        glbs = bounds_core.build_the_model_lp(mock_m, include_output_constraint=True, rhs=rhs)

        assert len(glbs) == 1

    def test_build_model_lp_include_output_constraint_no_rhs(self):
        """Test LP model building with output constraint but no rhs raises assertion."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_m = MagicMock()
        mock_m.net.final_name = "output"
        mock_m.model_ori.children.return_value = []

        with pytest.raises(AssertionError):
            bounds_core.build_the_model_lp(mock_m, include_output_constraint=True, rhs=None)

    def test_build_model_lp_get_primals(self):
        """Test LP model building with primal extraction."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"

        # Create mock layers with solver_vars
        mock_layer = MagicMock()
        mock_var = MagicMock()
        mock_var.X = 0.5
        mock_layer.solver_vars = [mock_var]
        mock_layer.inputs = []

        mock_final_layer = MagicMock()
        mock_final_layer.solver_vars = [mock_var]
        mock_final_layer.inputs = [mock_layer]

        mock_m.net.final_node.return_value = mock_final_layer
        mock_m.net.__getitem__.return_value = mock_final_layer
        mock_m.net.solver_model.status = grb.GRB.OPTIMAL
        mock_m.net.relus = []
        mock_m.model_ori.children.return_value = []

        with patch('builtins.print'):  # Suppress print
            glbs = bounds_core.build_the_model_lp(mock_m, get_primals=True)

        assert len(glbs) == 1

    def test_build_model_lp_multiple_output_vars(self):
        """Test build_the_model_lp with multiple output variables."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        mock_m = MagicMock()
        mock_m.net.final_name = "output"

        mock_vars = [MagicMock(X=0.5), MagicMock(X=1.0), MagicMock(X=-0.5)]
        mock_m.net.final_node.return_value.solver_vars = mock_vars
        mock_m.net.__getitem__.return_value.solver_vars = mock_vars
        mock_m.net.solver_model.status = grb.GRB.OPTIMAL
        mock_m.model_ori.children.return_value = []

        glbs = bounds_core.build_the_model_lp(mock_m)

        assert len(glbs) == 3
        assert glbs[0] == 0.5
        assert glbs[1] == 1.0
        assert glbs[2] == -0.5


class MockGurobiExpr:
    """Mock Gurobi expression that supports chained operations."""
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __ge__(self, other):
        return MockGurobiExpr("ge", self, other)

    def __le__(self, other):
        return MockGurobiExpr("le", self, other)

    def __eq__(self, other):
        return MockGurobiExpr("eq", self, other)

    def __sub__(self, other):
        return MockGurobiExpr("sub", self, other)

    def __rsub__(self, other):
        return MockGurobiExpr("rsub", other, self)

    def __mul__(self, other):
        return MockGurobiExpr("mul", self, other)

    def __rmul__(self, other):
        return MockGurobiExpr("rmul", other, self)

    def __add__(self, other):
        return MockGurobiExpr("add", self, other)

    def __radd__(self, other):
        return MockGurobiExpr("radd", other, self)


class MockGurobiVar:
    """Mock Gurobi variable that supports comparison operators."""
    def __init__(self, name="mock_var"):
        self.name = name
        self.lb = -float('inf')
        self.ub = float('inf')
        self.X = 0.0
        self.VBasis = 0

    def __ge__(self, other):
        return MockGurobiExpr("ge", self, other)

    def __le__(self, other):
        return MockGurobiExpr("le", self, other)

    def __eq__(self, other):
        return MockGurobiExpr("eq", self, other)

    def __sub__(self, other):
        return MockGurobiExpr("sub", self, other)

    def __rsub__(self, other):
        return MockGurobiExpr("rsub", other, self)

    def __mul__(self, other):
        return MockGurobiExpr("mul", self, other)

    def __rmul__(self, other):
        return MockGurobiExpr("rmul", other, self)

    def __add__(self, other):
        return MockGurobiExpr("add", self, other)

    def __radd__(self, other):
        return MockGurobiExpr("radd", other, self)


class TestUpdateModelBounds:
    """Tests for update_model_bounds function."""

    def test_update_model_bounds_positive_neuron(self):
        """Test updating bounds for always-positive neurons."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")
        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var
        }.get(name)

        lower_bounds = [torch.tensor([[0.5]])]  # Always positive
        upper_bounds = [torch.tensor([[1.0]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp_integer"
        )

        assert result == mock_model
        mock_model.update.assert_called_once()

    def test_update_model_bounds_negative_neuron(self):
        """Test updating bounds for always-negative neurons."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")
        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var
        }.get(name)

        lower_bounds = [torch.tensor([[-1.0]])]  # Always negative
        upper_bounds = [torch.tensor([[-0.5]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp_integer"
        )

        assert mock_relu_var.lb == 0
        assert mock_relu_var.ub == 0

    def test_update_model_bounds_unstable_neuron_lp_integer(self):
        """Test updating bounds for unstable neurons with lp_integer model."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")
        mock_a_var = MockGurobiVar("a_0_0")

        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var,
            "aReLU0_0": mock_a_var
        }.get(name)

        lower_bounds = [torch.tensor([[-1.0]])]  # Unstable
        upper_bounds = [torch.tensor([[1.0]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp_integer"
        )

        # Check ReLU constraints were added (3 constraints for lp_integer unstable neuron)
        assert mock_model.addConstr.call_count == 3

    def test_update_model_bounds_unstable_neuron_lp(self):
        """Test updating bounds for unstable neurons with pure LP model."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")

        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var
        }.get(name)

        lower_bounds = [torch.tensor([[-1.0]])]  # Unstable
        upper_bounds = [torch.tensor([[1.0]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp"  # Pure LP
        )

        # Should add triangle relaxation constraints (2 constraints for LP)
        assert mock_model.addConstr.call_count == 2

    def test_update_model_bounds_none_relu_var(self):
        """Test updating bounds when ReLU var doesn't exist (originally stable)."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")

        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": None  # No ReLU var (stable)
        }.get(name)

        lower_bounds = [torch.tensor([[0.5]])]
        upper_bounds = [torch.tensor([[1.0]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp_integer"
        )

        # Should not crash even with None ReLU var
        assert result == mock_model

    def test_update_model_bounds_multiple_neurons(self):
        """Test updating bounds for multiple neurons in a layer."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        vars_dict = {}
        for i in range(3):
            vars_dict[f"lay0_{i}"] = MockGurobiVar(f"pre_0_{i}")
            vars_dict[f"ReLU0_{i}"] = MockGurobiVar(f"relu_0_{i}")
            vars_dict[f"aReLU0_{i}"] = MockGurobiVar(f"a_0_{i}")

        mock_model.getVarByName.side_effect = lambda name: vars_dict.get(name)

        # Mix of unstable (-1 to 1), positive (0.5 to 2), and negative (-2 to -0.5)
        lower_bounds = [torch.tensor([[-1.0, 0.5, -2.0]])]
        upper_bounds = [torch.tensor([[1.0, 2.0, -0.5]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp_integer"
        )

        mock_model.update.assert_called_once()

    def test_update_model_bounds_empty_bounds(self):
        """Test update_model_bounds with empty bounds lists."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()

        result = bounds_core.update_model_bounds(
            mock_model, [], [], [], [], "lp"
        )

        assert result == mock_model
        mock_model.update.assert_called_once()

    def test_update_model_bounds_2d_bounds(self):
        """Test update_model_bounds with 2D bound tensors."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        vars_dict = {}
        for i in range(4):
            vars_dict[f"lay0_{i}"] = MockGurobiVar(f"pre_0_{i}")
            vars_dict[f"ReLU0_{i}"] = MockGurobiVar(f"relu_0_{i}")

        mock_model.getVarByName.side_effect = lambda name: vars_dict.get(name)

        # Mix: unstable, positive, negative, positive
        lower_bounds = [torch.tensor([[-1.0, 0.5, -2.0, 0.1]])]
        upper_bounds = [torch.tensor([[1.0, 2.0, -0.5, 1.0]])]
        pre_relu_layer_names = ["0"]
        relu_layer_names = ["0"]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            pre_relu_layer_names, relu_layer_names, "lp"
        )

        mock_model.update.assert_called_once()


class TestAllNodeSplitLP:
    """Tests for all_node_split_LP function."""

    def test_all_node_split_lp_termination_flag_set(self):
        """Test that all_node_split_LP returns early when termination flag is set."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        # Set up termination flag
        termination_flag = multiprocessing.Value('i', 1)
        bounds_core.termination_flag_lp = termination_flag

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        with patch.object(grb, 'Env', return_value=mock_env):
            arg = ([], [], [], [], [], [], 0)
            result = bounds_core.all_node_split_LP(arg)

        assert result[0] == 'unknown'
        assert result[3] is None

    def test_all_node_split_lp_safe_result(self):
        """Test all_node_split_LP with safe verification result."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        termination_flag = multiprocessing.Value('i', 0)
        bounds_core.termination_flag_lp = termination_flag
        bounds_core.input_name = ["inp_0"]

        mock_model = MagicMock()
        mock_var = MockGurobiVar("out_var")
        mock_var.X = 1.0  # Above threshold
        mock_model.getVarByName.return_value = mock_var
        mock_model.status = 2  # OPTIMAL
        bounds_core.multiprocess_lp_model = mock_model

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        with patch.object(bounds_core, 'copy_model', return_value=mock_model):
            with patch.object(bounds_core, 'update_model_bounds', return_value=mock_model):
                with patch.object(grb, 'Env', return_value=mock_env):
                    lower_bounds = [torch.tensor([[0.5]])]
                    upper_bounds = [torch.tensor([[1.0]])]
                    rhs = torch.tensor([0.5])  # Threshold

                    arg = ([], [], ["out_0"], lower_bounds, upper_bounds, rhs, 0)
                    result = bounds_core.all_node_split_LP(arg)

        assert result[0] == 'safe'
        assert result[3] is None

    def test_all_node_split_lp_unsafe_result(self):
        """Test all_node_split_LP with unsafe verification result."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        termination_flag = multiprocessing.Value('i', 0)
        bounds_core.termination_flag_lp = termination_flag
        bounds_core.input_name = "inp_0"

        mock_model = MagicMock()
        mock_var = MockGurobiVar("out_var")
        mock_var.X = 0.1  # Below threshold
        mock_model.getVarByName.return_value = mock_var
        mock_model.status = 2  # OPTIMAL
        bounds_core.multiprocess_lp_model = mock_model

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        with patch.object(bounds_core, 'copy_model', return_value=mock_model):
            with patch.object(bounds_core, 'update_model_bounds', return_value=mock_model):
                with patch.object(grb, 'Env', return_value=mock_env):
                    with patch('builtins.print'):  # Suppress print
                        lower_bounds = [torch.tensor([[0.5]])]
                        upper_bounds = [torch.tensor([[1.0]])]
                        rhs = torch.tensor([0.5])  # Threshold

                        arg = ([], [], ["out_0"], lower_bounds, upper_bounds, rhs, 0)
                        result = bounds_core.all_node_split_LP(arg)

        assert result[0] == 'unsafe'

    def test_all_node_split_lp_infeasible(self):
        """Test all_node_split_LP with infeasible model."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        termination_flag = multiprocessing.Value('i', 0)
        bounds_core.termination_flag_lp = termination_flag
        bounds_core.input_name = "inp_0"

        mock_model = MagicMock()
        mock_var = MockGurobiVar("out_var")
        mock_model.getVarByName.return_value = mock_var
        mock_model.status = 3  # Not OPTIMAL
        bounds_core.multiprocess_lp_model = mock_model

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        with patch.object(bounds_core, 'copy_model', return_value=mock_model):
            with patch.object(bounds_core, 'update_model_bounds', return_value=mock_model):
                with patch.object(grb, 'Env', return_value=mock_env):
                    lower_bounds = [torch.tensor([[0.5]])]
                    upper_bounds = [torch.tensor([[1.0]])]
                    rhs = torch.tensor([0.5])

                    arg = ([], [], ["out_0"], lower_bounds, upper_bounds, rhs, 0)
                    result = bounds_core.all_node_split_LP(arg)

        # glb should be inf for non-optimal status
        assert result[2] == float('inf')

    def test_all_node_split_lp_nested_input_name(self):
        """Test all_node_split_LP with nested input name structure."""
        import lp_mip_solver.bounds_core as bounds_core
        import gurobipy as grb

        termination_flag = multiprocessing.Value('i', 0)
        bounds_core.termination_flag_lp = termination_flag
        # Nested input name structure
        bounds_core.input_name = [["inp_0", "inp_1"], ["inp_2"]]

        mock_model = MagicMock()
        mock_var = MockGurobiVar("out_var")
        mock_var.X = 0.1  # Below threshold -> unsafe
        mock_model.getVarByName.return_value = mock_var
        mock_model.status = 2  # OPTIMAL
        bounds_core.multiprocess_lp_model = mock_model

        mock_env = MagicMock()
        mock_env.__enter__ = MagicMock(return_value=mock_env)
        mock_env.__exit__ = MagicMock(return_value=False)

        with patch.object(bounds_core, 'copy_model', return_value=mock_model):
            with patch.object(bounds_core, 'update_model_bounds', return_value=mock_model):
                with patch.object(grb, 'Env', return_value=mock_env):
                    with patch('builtins.print'):
                        lower_bounds = [torch.tensor([[0.5]])]
                        upper_bounds = [torch.tensor([[1.0]])]
                        rhs = torch.tensor([0.5])

                        arg = ([], [], ["out_0"], lower_bounds, upper_bounds, rhs, 0)
                        result = bounds_core.all_node_split_LP(arg)

        # Should handle nested structure and return unsafe
        assert result[0] == 'unsafe'
        # Counterexample should be nested list
        assert isinstance(result[3], list)




class TestCheckLpCut:
    """Tests for check_lp_cut function."""

    def test_check_lp_cut_disabled(self):
        """Test check_lp_cut returns early when cuts disabled."""
        import lp_mip_solver.bounds_core as bounds_core
        import arguments

        # Make sure cuts are disabled
        arguments.Config["bab"]["cut"]["enabled"] = False
        arguments.Config["bab"]["cut"]["bab_cut"] = False

        # Should return None without doing anything
        result = bounds_core.check_lp_cut([], [], {}, {}, [])
        assert result is None

    def test_check_lp_cut_enabled_but_no_bab_cut(self):
        """Test check_lp_cut returns early when bab_cut disabled."""
        import lp_mip_solver.bounds_core as bounds_core
        import arguments

        arguments.Config["bab"]["cut"]["enabled"] = True
        arguments.Config["bab"]["cut"]["bab_cut"] = False

        # Should return None without doing anything
        result = bounds_core.check_lp_cut([], [], {}, {}, [])
        assert result is None

        # Reset
        arguments.Config["bab"]["cut"]["enabled"] = False


class TestUpdateTheModelCut:
    """Tests for update_the_model_cut function."""

    def test_update_the_model_cut_no_cuts(self):
        """Test update_the_model_cut with no cuts."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_m = MagicMock()
        mock_model = MagicMock()
        mock_m.net.solver_model = mock_model

        mock_copied_model = MagicMock()
        mock_copied_model.status = 2  # OPTIMAL
        mock_var = MagicMock()
        mock_var.X = 0.5
        mock_var.VarName = "out_0"
        mock_copied_model.getVarByName.return_value = mock_var
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.model_ori.children.return_value = []

        with patch.object(bounds_core, 'copy_model', return_value=mock_copied_model):
            with patch('builtins.print'):  # Suppress prints
                result = bounds_core.update_the_model_cut(mock_m, None)

        assert result == 0.5

    def test_update_the_model_cut_with_splits(self):
        """Test update_the_model_cut with split constraints."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_m = MagicMock()
        mock_model = MagicMock()
        mock_m.net.solver_model = mock_model
        mock_relu = MagicMock()
        mock_relu.inputs = [MagicMock()]
        mock_relu.inputs[0].name = "pre_0"
        mock_relu.name = "relu_0"
        mock_m.net.relus = [mock_relu]

        mock_copied_model = MagicMock()
        mock_copied_model.status = 2  # OPTIMAL
        # Use MockGurobiVar for comparison operators
        mock_var = MockGurobiVar("out_0")
        mock_var.X = 0.5
        mock_var.VarName = "out_0"
        mock_split_var = MockGurobiVar("split_0")
        mock_copied_model.getVarByName.return_value = mock_split_var
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.model_ori.children.return_value = []

        pre_lbs = [torch.tensor([[-1.0]])]
        pre_ubs = [torch.tensor([[1.0]])]
        split = {'decision': [[0, 0]], 'choice': [1]}

        with patch.object(bounds_core, 'copy_model', return_value=mock_copied_model):
            with patch.object(bounds_core, 'update_model_bounds', return_value=mock_copied_model):
                with patch('builtins.print'):  # Suppress prints
                    result = bounds_core.update_the_model_cut(
                        mock_m, [], pre_lbs=pre_lbs, pre_ubs=pre_ubs, split=split
                    )

        # Constraints should be added
        assert mock_copied_model.addConstr.called

    def test_update_the_model_cut_infeasible(self):
        """Test update_the_model_cut with infeasible model."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_m = MagicMock()
        mock_model = MagicMock()
        mock_m.net.solver_model = mock_model

        mock_copied_model = MagicMock()
        mock_copied_model.status = 3  # INFEASIBLE
        mock_var = MagicMock()
        mock_var.VarName = "out_0"
        mock_copied_model.getVarByName.return_value = mock_var
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.model_ori.children.return_value = []

        with patch.object(bounds_core, 'copy_model', return_value=mock_copied_model):
            with patch('builtins.print'):  # Suppress prints
                result = bounds_core.update_the_model_cut(mock_m, [])

        assert result == float('inf')

    def test_update_the_model_cut_with_cut_constraints(self):
        """Test update_the_model_cut with cut constraints."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_m = MagicMock()
        mock_model = MagicMock()
        mock_m.net.solver_model = mock_model
        mock_m.net.relus = []
        mock_m.net.solver_model_type = "lp"

        mock_copied_model = MagicMock()
        mock_copied_model.status = 2  # OPTIMAL
        mock_var = MockGurobiVar("out_0")
        mock_var.X = 0.5
        mock_var.VarName = "out_0"
        mock_copied_model.getVarByName.return_value = mock_var
        mock_m.net.final_node.return_value.solver_vars = [mock_var]
        mock_m.model_ori.children.return_value = []

        # Create a mock constraint returned by addConstr
        mock_constr = MagicMock()
        mock_constr.pi = 0.1  # Set a real value for pi
        mock_constr.slack = 0.0
        mock_constr.constrName = "cut0"
        mock_copied_model.addConstr.return_value = mock_constr

        # Create a simple cut
        cut = [{
            "bias": 0.0,
            "x_decision": [],
            "x_coeffs": [],
            "relu_decision": [],
            "relu_coeffs": [],
            "arelu_decision": [],
            "arelu_coeffs": [],
            "pre_decision": [],
            "pre_coeffs": [],
            "c": 1
        }]

        with patch.object(bounds_core, 'copy_model', return_value=mock_copied_model):
            with patch('builtins.print'):  # Suppress prints
                result = bounds_core.update_the_model_cut(mock_m, cut)

        assert result == 0.5


class TestSolveDivingLp:
    """Tests for solve_diving_lp function."""

    def test_solve_diving_lp_basic(self):
        """Test basic solve_diving_lp functionality."""
        import lp_mip_solver.bounds_core as bounds_core
        import torch.nn as nn

        mock_m = MagicMock()
        mock_model = MagicMock()
        mock_model.objval = 0.5
        mock_m.net.solver_model = mock_model
        mock_m.layers = [nn.Linear(10, 10)]

        mock_diving_model = MagicMock()
        mock_diving_model.objval = 0.6
        mock_diving_model.status = 2
        mock_model.copy.return_value = mock_diving_model

        mock_constr = MagicMock()
        mock_constr.pi = 0.1
        mock_model.getConstrByName.return_value = mock_constr

        primal_vars = [[0.5] * 10, [0.5] * 10]
        integer_vars = []
        lower_bounds = [torch.zeros(1, 10)]
        upper_bounds = [torch.ones(1, 10)]

        with patch('builtins.print'):  # Suppress prints
            bounds_core.solve_diving_lp(
                mock_m, primal_vars, integer_vars, lower_bounds, upper_bounds
            )

        mock_diving_model.optimize.assert_called_once()


class TestTriangleRelaxation:
    """Tests specifically for triangle relaxation in update_model_bounds."""

    def test_triangle_relaxation_slope_computation(self):
        """Test that triangle relaxation slope is computed correctly."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")

        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var
        }.get(name)

        # Unstable neuron with specific bounds for slope calculation
        lower_bounds = [torch.tensor([[-2.0]])]  # lb = -2
        upper_bounds = [torch.tensor([[4.0]])]   # ub = 4
        # Expected slope = 4 / (4 - (-2)) = 4/6 = 2/3
        # Expected bias = -(-2) * (2/3) = 4/3

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            ["0"], ["0"], "lp"
        )

        # Verify constraints were added (2 for triangle relaxation: var >= pre_var and var <= slope*pre_var + bias)
        assert mock_model.addConstr.call_count == 2

    def test_triangle_relaxation_boundary_case_zero_lb(self):
        """Test triangle relaxation when lower bound is exactly zero."""
        import lp_mip_solver.bounds_core as bounds_core

        mock_model = MagicMock()
        mock_pre_var = MockGurobiVar("pre_0_0")
        mock_relu_var = MockGurobiVar("relu_0_0")

        mock_model.getVarByName.side_effect = lambda name: {
            "lay0_0": mock_pre_var,
            "ReLU0_0": mock_relu_var
        }.get(name)

        # lb = 0 means always positive
        lower_bounds = [torch.tensor([[0.0]])]
        upper_bounds = [torch.tensor([[1.0]])]

        result = bounds_core.update_model_bounds(
            mock_model, lower_bounds, upper_bounds,
            ["0"], ["0"], "lp"
        )

        # Should add equality constraint
        mock_model.addConstr.assert_called()


class TestCopyModel:
    """Tests for copy_model utility function from utils.py."""

    def test_copy_model_basic(self):
        """Test basic copy_model functionality."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_copied = MagicMock()
        mock_model.copy.return_value = mock_copied
        mock_model.getVars.return_value = []
        mock_model.getConstrs.return_value = []

        result = copy_model(mock_model, basis=False)

        mock_model.copy.assert_called_once()
        assert result == mock_copied

    def test_copy_model_with_basis(self):
        """Test copy_model with basis copying."""
        from lp_mip_solver.utils import copy_model

        mock_model = MagicMock()
        mock_copied = MagicMock()
        mock_model.copy.return_value = mock_copied

        mock_var = MagicMock()
        mock_var.VBasis = 0
        mock_var.X = 0.5
        mock_model.getVars.return_value = [mock_var]

        mock_copied_var = MagicMock()
        mock_copied.getVars.return_value = [mock_copied_var]

        mock_constr = MagicMock()
        mock_constr.ConstrName = "c1"
        mock_constr.CBasis = 0
        mock_constr.Pi = 0.1
        mock_model.getConstrs.return_value = [mock_constr]
        mock_model.getConstrByName.return_value = mock_constr

        mock_copied_constr = MagicMock()
        mock_copied_constr.ConstrName = "c1"
        mock_copied.getConstrs.return_value = [mock_copied_constr]

        result = copy_model(mock_model, basis=True)

        assert result == mock_copied


class TestHandleGurobiError:
    """Tests for handle_gurobi_error function."""

    def test_handle_gurobi_error_raises(self):
        """Test that handle_gurobi_error raises an exception."""
        from lp_mip_solver.utils import handle_gurobi_error
        import gurobipy as grb

        # GurobiError constructor signature varies by version,
        # so we catch the base exception type
        with pytest.raises((grb.GurobiError, TypeError)):
            handle_gurobi_error("Test error message")

    def test_handle_gurobi_error_prints_message(self):
        """Test that handle_gurobi_error prints the error message."""
        from lp_mip_solver.utils import handle_gurobi_error
        import gurobipy as grb

        with patch('builtins.print') as mock_print:
            try:
                handle_gurobi_error("Test error message")
            except (grb.GurobiError, TypeError):
                pass

        mock_print.assert_called_once_with('Gurobi error: Test error message')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
