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
"""Unit tests for scip_model.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import io

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_mocked_imports():
    """Setup mocked pyscipopt and gurobipy imports."""
    # Create mock for pyscipopt
    mock_pyscipopt = MagicMock()

    # Mock Eventhdlr as a proper class for inheritance
    class MockEventhdlr:
        pass

    mock_pyscipopt.Eventhdlr = MockEventhdlr
    mock_pyscipopt.SCIP_RESULT = MagicMock()
    mock_pyscipopt.SCIP_EVENTTYPE = MagicMock()
    mock_pyscipopt.SCIP_EVENTTYPE.ROWADDEDSEPA = 'ROWADDEDSEPA'
    mock_pyscipopt.SCIP_EVENTTYPE.NODESOLVED = 'NODESOLVED'
    mock_pyscipopt.SCIP_PARAMSETTING = MagicMock()
    mock_pyscipopt.SCIP_PARAMSETTING.OFF = 'OFF'
    mock_pyscipopt.scip = MagicMock()

    # Create a real base class for Model that can be inherited from
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        def setParam(self, name, value):
            pass

        def setObjective(self, expr, sense):
            pass

        def addVar(self, *args, **kwargs):
            pass

        def getVars(self):
            return []

        def addCons(self, cons, name=''):
            pass

        def getConss(self):
            return []

        def getStatus(self):
            return "unknown"

        def getSols(self):
            return []

        def getDualbound(self):
            return 0.0

        def getPrimalbound(self):
            return 0.0

    mock_pyscipopt.Model = MockModel

    # Create mock for gurobipy
    mock_grb = MagicMock()
    mock_grb.GRB.CONTINUOUS = 'CONT'
    mock_grb.GRB.BINARY = 'BIN'
    mock_grb.GRB.INTEGER = 'INT'
    mock_grb.GRB.MAXIMIZE = 1
    mock_grb.GRB.MINIMIZE = -1

    return mock_pyscipopt, mock_grb, MockEventhdlr


class TestSCIPVariableProperties(unittest.TestCase):
    """Tests for SCIPVariable class properties."""

    def test_lb_property(self):
        """Test lb property returns getLbLocal()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getLbLocal.return_value = -1.5

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.lb, -1.5)
            mock_var.getLbLocal.assert_called_once()

    def test_ub_property(self):
        """Test ub property returns getUbLocal()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getUbLocal.return_value = 2.5

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.ub, 2.5)
            mock_var.getUbLocal.assert_called_once()

    def test_LB_property(self):
        """Test LB property (uppercase) returns getLbLocal()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getLbLocal.return_value = -3.0

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.LB, -3.0)

    def test_UB_property(self):
        """Test UB property (uppercase) returns getUbLocal()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getUbLocal.return_value = 4.0

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.UB, 4.0)

    def test_VarName_property(self):
        """Test VarName property returns var.name."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.name = "test_variable"

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.VarName, "test_variable")

    def test_var_attribute_stored(self):
        """Test that var attribute is stored correctly."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            scip_var = SCIPVariable(mock_var)

            self.assertIs(scip_var.var, mock_var)

    def test_lb_and_LB_same_value(self):
        """Test lb and LB return the same value."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getLbLocal.return_value = -2.0

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.lb, scip_var.LB)

    def test_ub_and_UB_same_value(self):
        """Test ub and UB return the same value."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPVariable

            mock_var = MagicMock()
            mock_var.getUbLocal.return_value = 3.0

            scip_var = SCIPVariable(mock_var)

            self.assertEqual(scip_var.ub, scip_var.UB)


class TestSCIPModelStatusMapping(unittest.TestCase):
    """Tests for SCIPModel status property mapping SCIP -> Gurobi codes."""

    def _test_status_mapping(self, scip_status, expected_gurobi_code):
        """Helper to test status mapping."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        # Override getStatus in MockModel
        mock_pyscipopt.Model.getStatus = MagicMock(return_value=scip_status)

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            self.assertEqual(model.status, expected_gurobi_code)

    def test_status_optimal_returns_2(self):
        """Test 'optimal' status maps to Gurobi code 2."""
        self._test_status_mapping("optimal", 2)

    def test_status_timelimit_returns_9(self):
        """Test 'timelimit' status maps to Gurobi code 9."""
        self._test_status_mapping("timelimit", 9)

    def test_status_infeasible_returns_3(self):
        """Test 'infeasible' status maps to Gurobi code 3."""
        self._test_status_mapping("infeasible", 3)

    def test_status_unbounded_returns_5(self):
        """Test 'unbounded' status maps to Gurobi code 5."""
        self._test_status_mapping("unbounded", 5)

    def test_status_userinterrupt_returns_11(self):
        """Test 'userinterrupt' status maps to Gurobi code 11."""
        self._test_status_mapping("userinterrupt", 11)

    def test_status_inforunbd_returns_4(self):
        """Test 'inforunbd' status maps to Gurobi code 4."""
        self._test_status_mapping("inforunbd", 4)

    def test_status_nodelimit_returns_9(self):
        """Test 'nodelimit' status maps to Gurobi code 9."""
        self._test_status_mapping("nodelimit", 9)

    def test_status_totalnodelimit_returns_9(self):
        """Test 'totalnodelimit' status maps to Gurobi code 9."""
        self._test_status_mapping("totalnodelimit", 9)

    def test_status_stallnodelimit_returns_9(self):
        """Test 'stallnodelimit' status maps to Gurobi code 9."""
        self._test_status_mapping("stallnodelimit", 9)

    def test_status_gaplimit_returns_9(self):
        """Test 'gaplimit' status maps to Gurobi code 9."""
        self._test_status_mapping("gaplimit", 9)

    def test_status_memlimit_returns_9(self):
        """Test 'memlimit' status maps to Gurobi code 9."""
        self._test_status_mapping("memlimit", 9)

    def test_status_sollimit_returns_9(self):
        """Test 'sollimit' status maps to Gurobi code 9."""
        self._test_status_mapping("sollimit", 9)

    def test_status_bestsollimit_returns_15(self):
        """Test 'bestsollimit' status maps to Gurobi code 15."""
        self._test_status_mapping("bestsollimit", 15)

    def test_status_restartlimit_returns_9(self):
        """Test 'restartlimit' status maps to Gurobi code 9."""
        self._test_status_mapping("restartlimit", 9)

    def test_status_unknown_returns_unknown_string(self):
        """Test unknown status returns 'unknown' string."""
        self._test_status_mapping("some_unknown_status", "unknown")


class TestSCIPModelSetObjective(unittest.TestCase):
    """Tests for SCIPModel setObjective method."""

    def test_setObjective_maximize(self):
        """Test setObjective with MAXIMIZE sense calls parent with 'maximize'."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []
        original_setObjective = mock_pyscipopt.Model.setObjective

        def capture_setObjective(self, expr, sense):
            calls.append((expr, sense))

        mock_pyscipopt.Model.setObjective = capture_setObjective

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            mock_expr = MagicMock()

            model.setObjective(mock_expr, mock_grb.GRB.MAXIMIZE)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 'maximize')

    def test_setObjective_minimize(self):
        """Test setObjective with MINIMIZE sense calls parent with 'minimize'."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setObjective(self, expr, sense):
            calls.append((expr, sense))

        mock_pyscipopt.Model.setObjective = capture_setObjective

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            mock_expr = MagicMock()

            model.setObjective(mock_expr, mock_grb.GRB.MINIMIZE)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 'minimize')

    def test_setObjective_invalid_sense_raises(self):
        """Test setObjective with invalid sense raises NotImplementedError."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            mock_expr = MagicMock()

            with self.assertRaises(NotImplementedError):
                model.setObjective(mock_expr, "invalid_sense")

    def test_setObjective_with_scip_variable(self):
        """Test setObjective unwraps SCIPVariable."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setObjective(self, expr, sense):
            calls.append((expr, sense))

        mock_pyscipopt.Model.setObjective = capture_setObjective

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel, SCIPVariable

            model = SCIPModel()
            mock_var = MagicMock()
            scip_var = SCIPVariable(mock_var)

            model.setObjective(scip_var, mock_grb.GRB.MINIMIZE)

            # Should pass the unwrapped var
            self.assertEqual(calls[0][0], mock_var)


class TestSCIPModelAddVar(unittest.TestCase):
    """Tests for SCIPModel addVar method."""

    def test_addVar_continuous(self):
        """Test addVar with CONTINUOUS type maps to 'C'."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_addVar(self, **kwargs):
            calls.append(kwargs)

        mock_pyscipopt.Model.addVar = capture_addVar

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            model.addVar(lb=0.0, ub=1.0, vtype=mock_grb.GRB.CONTINUOUS, name='x')

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['vtype'], 'C')
            self.assertEqual(calls[0]['name'], 'x')

    def test_addVar_binary(self):
        """Test addVar with BINARY type maps to 'B'."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_addVar(self, **kwargs):
            calls.append(kwargs)

        mock_pyscipopt.Model.addVar = capture_addVar

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            model.addVar(lb=0.0, ub=1.0, vtype=mock_grb.GRB.BINARY, name='b')

            self.assertEqual(calls[0]['vtype'], 'B')

    def test_addVar_integer(self):
        """Test addVar with INTEGER type maps to 'I'."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_addVar(self, **kwargs):
            calls.append(kwargs)

        mock_pyscipopt.Model.addVar = capture_addVar

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            model.addVar(lb=0.0, ub=10.0, vtype=mock_grb.GRB.INTEGER, name='i')

            self.assertEqual(calls[0]['vtype'], 'I')


class TestSCIPModelGetVarByName(unittest.TestCase):
    """Tests for SCIPModel getVarByName method."""

    def test_getVarByName_found(self):
        """Test getVarByName returns SCIPVariable when found."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        mock_var1 = MagicMock()
        mock_var1.name = "x"
        mock_var2 = MagicMock()
        mock_var2.name = "y"

        def getVars(self):
            return [mock_var1, mock_var2]

        mock_pyscipopt.Model.getVars = getVars

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel, SCIPVariable

            model = SCIPModel()
            result = model.getVarByName("x")

            self.assertIsInstance(result, SCIPVariable)
            self.assertEqual(result.var, mock_var1)

    def test_getVarByName_not_found(self):
        """Test getVarByName returns None when not found."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        mock_var1 = MagicMock()
        mock_var1.name = "x"

        def getVars(self):
            return [mock_var1]

        mock_pyscipopt.Model.getVars = getVars

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            result = model.getVarByName("z")

            self.assertIsNone(result)

    def test_getVarByName_partial_match(self):
        """Test getVarByName matches partial names (uses 'in')."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        mock_var1 = MagicMock()
        mock_var1.name = "x_layer1_neuron0"

        def getVars(self):
            return [mock_var1]

        mock_pyscipopt.Model.getVars = getVars

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel, SCIPVariable

            model = SCIPModel()
            result = model.getVarByName("layer1")

            self.assertIsInstance(result, SCIPVariable)


class TestSCIPModelCompatibilityNoOps(unittest.TestCase):
    """Tests for no-op methods used by shared solver code."""

    def test_reset_and_update_are_supported_no_ops(self):
        """The Gurobi-compatible reset and update calls should be harmless."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            for method_name in ('reset', 'update'):
                with self.subTest(method_name=method_name):
                    self.assertIsNone(getattr(model, method_name)())


class TestSCIPModelSetParamBasic(unittest.TestCase):
    """Tests for SCIPModel setParam method - basic parameter mappings."""

    def _test_param_mapping(self, gurobi_name, gurobi_value, expected_scip_name, expected_scip_value):
        """Helper to test parameter mapping."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            model.setParam(gurobi_name, gurobi_value)

            # Find the call with expected_scip_name and expected_scip_value
            # Use last matching call to handle cases where __init__ sets default values
            matching_calls = [c for c in calls if c[0] == expected_scip_name and c[1] == expected_scip_value]
            self.assertTrue(len(matching_calls) > 0, f"Expected {expected_scip_name} with value {expected_scip_value} to be called")

    def test_setParam_OutputFlag_true(self):
        """Test setParam OutputFlag with True value maps to verblevel 4."""
        self._test_param_mapping('OutputFlag', True, 'display/verblevel', 4)

    def test_setParam_OutputFlag_false(self):
        """Test setParam OutputFlag with False value maps to verblevel 1."""
        self._test_param_mapping('OutputFlag', False, 'display/verblevel', 1)

    def test_setParam_Threads(self):
        """Test setParam Threads maps to parallel/maxnthreads."""
        self._test_param_mapping('Threads', 4, 'parallel/maxnthreads', 4)

    def test_setParam_FeasibilityTol(self):
        """Test setParam FeasibilityTol maps to numerics/feastol."""
        self._test_param_mapping('FeasibilityTol', 1e-6, 'numerics/feastol', 1e-6)

    def test_setParam_TimeLimit(self):
        """Test setParam TimeLimit maps to limits/time."""
        self._test_param_mapping('TimeLimit', 100, 'limits/time', 100)

    def test_setParam_MIPGapAbs(self):
        """Test setParam MIPGapAbs maps to limits/absgap."""
        self._test_param_mapping('MIPGapAbs', 1e-8, 'limits/absgap', 1e-8)

    def test_setParam_MIPGap(self):
        """Test setParam MIPGap maps to limits/gap."""
        self._test_param_mapping('MIPGap', 0.01, 'limits/gap', 0.01)

    def test_setParam_unsupported_raises(self):
        """Test setParam with unsupported parameter raises NotImplementedError."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with self.assertRaises(NotImplementedError):
                model.setParam('UnsupportedParam', 1)


class TestSCIPModelSetParamCuts(unittest.TestCase):
    """Tests for SCIPModel setParam method - cut parameter handling."""

    def test_setParam_Cuts_disable(self):
        """Test setParam Cuts with value <= 0 disables all cuts."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('Cuts', 0)

            # Should have set many separating/*/freq params to -1
            freq_calls = [c for c in calls if '/freq' in c[0]]
            self.assertTrue(len(freq_calls) > 10)
            for name, value in freq_calls:
                self.assertEqual(value, -1)

    def test_setParam_Cuts_enable(self):
        """Test setParam Cuts with value > 0 enables cuts with defaults."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            model.setParam('Cuts', 1)

            # Should have set some separating/*/freq params to positive values
            gomory_calls = [c for c in calls if 'gomory/freq' in c[0]]
            self.assertTrue(len(gomory_calls) > 0)
            self.assertEqual(gomory_calls[0][1], 10)

    def test_setParam_GomoryPasses(self):
        """Test setParam GomoryPasses sets gomory frequencies."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('GomoryPasses', 1)

            call_names = [c[0] for c in calls]
            self.assertTrue(any('gomory' in n for n in call_names))

    def test_setParam_RLTCuts(self):
        """Test setParam RLTCuts sets rlt frequency."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('RLTCuts', 1)

            call_names = [c[0] for c in calls]
            self.assertTrue(any('rlt' in n for n in call_names))

    def test_setParam_FlowCoverCuts(self):
        """Test setParam FlowCoverCuts sets flowcover frequencies."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('FlowCoverCuts', 1)

            call_names = [c[0] for c in calls]
            self.assertTrue(any('flowcover' in n for n in call_names))

    def test_setParam_MIRCuts(self):
        """Test setParam MIRCuts sets cmir frequencies."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('MIRCuts', 1)

            call_names = [c[0] for c in calls]
            self.assertTrue(any('cmir' in n for n in call_names))

    def test_setParam_ImpliedCuts(self):
        """Test setParam ImpliedCuts sets impliedbounds frequencies."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_setParam(self, name, value):
            calls.append((name, value))

        mock_pyscipopt.Model.setParam = capture_setParam

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            with patch('builtins.print'):
                model.setParam('ImpliedCuts', 1)

            call_names = [c[0] for c in calls]
            self.assertTrue(any('impliedbounds' in n for n in call_names))


class TestSCIPModelAddConstr(unittest.TestCase):
    """Tests for SCIPModel addConstr method."""

    def test_addConstr_calls_addCons(self):
        """Test addConstr calls parent addCons with cons and name."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_addCons(self, cons, name=''):
            calls.append((cons, name))

        mock_pyscipopt.Model.addCons = capture_addCons

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            mock_cons = MagicMock()

            model.addConstr(mock_cons, name='c1')

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], mock_cons)
            self.assertEqual(calls[0][1], 'c1')

    def test_addConstr_default_name(self):
        """Test addConstr with default empty name."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        calls = []

        def capture_addCons(self, cons, name=''):
            calls.append((cons, name))

        mock_pyscipopt.Model.addCons = capture_addCons

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            mock_cons = MagicMock()

            model.addConstr(mock_cons)

            self.assertEqual(calls[0][1], '')


class TestSCIPModelGetConstrs(unittest.TestCase):
    """Tests for SCIPModel getConstrs method."""

    def test_getConstrs_calls_getConss(self):
        """Test getConstrs calls parent getConss."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        mock_conss = [MagicMock(), MagicMock()]

        def getConss(self):
            return mock_conss

        mock_pyscipopt.Model.getConss = getConss

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            result = model.getConstrs()

            self.assertEqual(result, mock_conss)


class TestSCIPModelCopy(unittest.TestCase):
    """Tests for SCIPModel copy method."""

    def test_copy_returns_new_model(self):
        """Test copy returns a new SCIPModel."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            copied = model.copy()

            self.assertIsInstance(copied, SCIPModel)
            self.assertIsNot(copied, model)


class TestSCIPModelProperties(unittest.TestCase):
    """Tests for SCIPModel solcount, objbound, objval properties."""

    def test_solcount_returns_length_of_sols(self):
        """Test solcount returns len(getSols())."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        mock_sols = [MagicMock(), MagicMock(), MagicMock()]

        def getSols(self):
            return mock_sols

        mock_pyscipopt.Model.getSols = getSols

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            self.assertEqual(model.solcount, 3)

    def test_solcount_empty(self):
        """Test solcount returns 0 when no solutions."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        def getSols(self):
            return []

        mock_pyscipopt.Model.getSols = getSols

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            self.assertEqual(model.solcount, 0)

    def test_objbound_returns_dual_bound(self):
        """Test objbound returns getDualbound()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        def getDualbound(self):
            return -1.5

        mock_pyscipopt.Model.getDualbound = getDualbound

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            self.assertEqual(model.objbound, -1.5)

    def test_objval_returns_primal_bound(self):
        """Test objval returns getPrimalbound()."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        def getPrimalbound(self):
            return 2.5

        mock_pyscipopt.Model.getPrimalbound = getPrimalbound

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()

            self.assertEqual(model.objval, 2.5)


class TestEarlyStopException(unittest.TestCase):
    """Tests for EarlyStop exception."""

    def test_early_stop_prints_message(self):
        """Test EarlyStop prints its message."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStop

            with patch('builtins.print') as mock_print:
                exc = EarlyStop("Test early stop message")
                mock_print.assert_called_once_with("Test early stop message")

    def test_early_stop_is_exception(self):
        """Test EarlyStop is an Exception."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStop

            self.assertTrue(issubclass(EarlyStop, Exception))


class TestGenerateCutsEvent(unittest.TestCase):
    """Tests for GenerateCutsEvent event handler."""

    def test_eventinit_sets_call_count(self):
        """Test eventinit sets _call_count to 0."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import GenerateCutsEvent

            event = GenerateCutsEvent()
            event.model = MagicMock()

            event.eventinit()

            self.assertEqual(event._call_count, 0)

    def test_eventinit_catches_event(self):
        """Test eventinit catches ROWADDEDSEPA event."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import GenerateCutsEvent

            event = GenerateCutsEvent()
            event.model = MagicMock()

            event.eventinit()

            event.model.catchEvent.assert_called_once()

    def test_eventexit_drops_event(self):
        """Test eventexit drops ROWADDEDSEPA event."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import GenerateCutsEvent

            event = GenerateCutsEvent()
            event.model = MagicMock()

            event.eventexit()

            event.model.dropEvent.assert_called_once()

    def test_eventexec_no_cuts_no_increment(self):
        """Test eventexec does nothing when no cuts."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import GenerateCutsEvent

            event = GenerateCutsEvent()
            event.model = MagicMock()
            event.model.getCuts.return_value = []
            event._call_count = 0

            event.eventexec(MagicMock())

            self.assertEqual(event._call_count, 0)

    def test_eventexec_with_cuts_increments_count(self):
        """Test eventexec increments call count when cuts exist."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import GenerateCutsEvent

            event = GenerateCutsEvent()
            event.model = MagicMock()
            event.model.getCuts.return_value = [MagicMock()]
            event._call_count = 0

            with patch('builtins.open', MagicMock()):
                with patch('builtins.print'):
                    event.eventexec(MagicMock())

            self.assertEqual(event._call_count, 1)


class TestEarlyStopEvent(unittest.TestCase):
    """Tests for EarlyStopEvent event handler."""

    def test_eventinit_catches_nodesolved(self):
        """Test eventinit catches NODESOLVED event."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStopEvent

            event = EarlyStopEvent()
            event.model = MagicMock()

            event.eventinit()

            event.model.catchEvent.assert_called_once()

    def test_eventexit_drops_nodesolved(self):
        """Test eventexit drops NODESOLVED event."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStopEvent

            event = EarlyStopEvent()
            event.model = MagicMock()

            event.eventexit()

            event.model.dropEvent.assert_called_once()

    def test_eventexec_stops_when_dual_bound_positive(self):
        """Test eventexec interrupts when dual bound > 1e-5."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStopEvent

            event = EarlyStopEvent()
            event.model = MagicMock()
            event.model.getDualbound.return_value = 0.1  # > 1e-5
            event.model.getPrimalbound.return_value = 1.0

            with patch('builtins.print'):
                with patch('time.sleep'):
                    event.eventexec(MagicMock())

            event.model.interruptSolve.assert_called_once()

    def test_eventexec_stops_when_primal_bound_negative(self):
        """Test eventexec interrupts when primal bound < -1e-5."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStopEvent

            event = EarlyStopEvent()
            event.model = MagicMock()
            event.model.getDualbound.return_value = 0.0
            event.model.getPrimalbound.return_value = -0.1  # < -1e-5

            with patch('builtins.print'):
                with patch('time.sleep'):
                    event.eventexec(MagicMock())

            event.model.interruptSolve.assert_called_once()

    def test_eventexec_no_stop_normal_bounds(self):
        """Test eventexec does not interrupt with normal bounds."""
        mock_pyscipopt, mock_grb, MockEventhdlr = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import EarlyStopEvent

            event = EarlyStopEvent()
            event.model = MagicMock()
            event.model.getDualbound.return_value = 0.0
            event.model.getPrimalbound.return_value = 0.0

            event.eventexec(MagicMock())

            event.model.interruptSolve.assert_not_called()


class TestSCIPModelInit(unittest.TestCase):
    """Tests for SCIPModel __init__ method."""

    def test_init_creates_model(self):
        """Test __init__ creates a model instance."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel()
            self.assertIsNotNone(model)

    def test_init_with_name(self):
        """Test __init__ with model name."""
        mock_pyscipopt, mock_grb, _ = setup_mocked_imports()

        with patch.dict('sys.modules', {
            'pyscipopt': mock_pyscipopt,
            'gurobipy': mock_grb,
        }):
            if 'scip_model' in sys.modules:
                del sys.modules['scip_model']
            from scip_model import SCIPModel

            model = SCIPModel("TestModel")
            self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
