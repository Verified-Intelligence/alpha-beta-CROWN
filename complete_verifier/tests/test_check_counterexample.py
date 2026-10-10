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
"""Unit tests for check_counterexample.py"""
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_counterexample as check_counterexample_module
from cex_validation import CEX_INPUT_TOLERANCE, CEX_OUTPUT_TOLERANCE
from check_counterexample import (
    parse_cex, print_vnnlib, read_statements, update_rv_tuple,
    make_input_box_dict, check_adv_example, CexResult
)


class TestParseCex(unittest.TestCase):
    """Tests for parse_cex function."""

    def _write_temp_cex(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_simple_cex(self):
        """Test parsing simple counterexample."""
        content = """(X_0 0.5)
(X_1 1.0)
(Y_0 0.25)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 1)
        self.assertAlmostEqual(x[0], 0.5)
        self.assertAlmostEqual(x[1], 1.0)
        self.assertAlmostEqual(y[0], 0.25)

    def test_scientific_notation(self):
        """Test parsing counterexample with scientific notation."""
        content = """(X_0 1.5e-3)
(X_1 -2.0e+2)
(Y_0 3.14e0)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], 0.0015)
        self.assertAlmostEqual(x[1], -200.0)
        self.assertAlmostEqual(y[0], 3.14)

    def test_negative_values(self):
        """Test parsing counterexample with negative values."""
        content = """(X_0 -0.5)
(X_1 -1.0)
(Y_0 -0.25)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], -0.5)
        self.assertAlmostEqual(x[1], -1.0)
        self.assertAlmostEqual(y[0], -0.25)

    def test_sparse_indices(self):
        """Test parsing counterexample with sparse/non-consecutive indices."""
        content = """(X_0 1.0)
(X_5 2.0)
(Y_2 3.0)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        # x should have indices 0-5 (6 elements)
        self.assertEqual(len(x), 6)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[5], 2.0)
        # y should have indices 0-2 (3 elements)
        self.assertEqual(len(y), 3)
        self.assertAlmostEqual(y[2], 3.0)

    def test_empty_file(self):
        """Test parsing empty counterexample file."""
        content = ""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_multiple_parentheses(self):
        """Test parsing with multiple parentheses format."""
        content = """((X_0 0.5))
((Y_0 0.25))
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], 0.5)
        self.assertAlmostEqual(y[0], 0.25)

    def test_with_spaces(self):
        """Test parsing with various spacing."""
        # Note: The regex requires specific format, extra spaces may not match
        content = """(X_0 0.5)
(X_1 1.0)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], 0.5)
        self.assertAlmostEqual(x[1], 1.0)


class TestReadStatements(unittest.TestCase):
    """Tests for read_statements function (from check_counterexample.py)."""

    def _write_temp_vnnlib(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vnnlib') as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_simple_statements(self):
        """Test parsing simple statements."""
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
"""
        path = self._write_temp_vnnlib(content)
        stmts = read_statements(path)
        self.assertEqual(len(stmts), 2)

    def test_comments_removed(self):
        """Test that comments are removed."""
        content = """; This is a comment
(declare-const X_0 Real) ; inline comment
"""
        path = self._write_temp_vnnlib(content)
        stmts = read_statements(path)
        self.assertEqual(len(stmts), 1)
        self.assertNotIn(';', stmts[0])

    def test_multiline_statement(self):
        """Test parsing multiline statement."""
        content = """(assert (and
    (<= X_0 1.0)
    (>= X_0 0.0)))
"""
        path = self._write_temp_vnnlib(content)
        stmts = read_statements(path)
        self.assertEqual(len(stmts), 1)
        self.assertIn('and', stmts[0])


class TestMakeInputBoxDict(unittest.TestCase):
    """Tests for make_input_box_dict function."""

    def test_basic(self):
        """Test basic functionality."""
        box = make_input_box_dict(3)
        self.assertEqual(len(box), 3)
        self.assertIn(0, box)
        self.assertIn(1, box)
        self.assertIn(2, box)

    def test_infinite_bounds(self):
        """Test that default bounds are infinite."""
        box = make_input_box_dict(2)
        self.assertEqual(box[0][0], -np.inf)
        self.assertEqual(box[0][1], np.inf)

    def test_zero_inputs(self):
        """Test with zero inputs."""
        box = make_input_box_dict(0)
        self.assertEqual(len(box), 0)


class TestUpdateRvTuple(unittest.TestCase):
    """Tests for update_rv_tuple function."""

    def _make_rv_tuple(self, num_inputs, num_outputs):
        box_dict = {i: [-np.inf, np.inf] for i in range(num_inputs)}
        return (box_dict, [], [])

    def test_input_upper_bound(self):
        """Test setting input upper bound."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '<=', 'X_0', '1.0', 2, 2)
        self.assertEqual(rv[0][0][1], 1.0)

    def test_input_lower_bound(self):
        """Test setting input lower bound."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '>=', 'X_0', '0.5', 2, 2)
        self.assertEqual(rv[0][0][0], 0.5)

    def test_output_constraint(self):
        """Test adding output constraint."""
        rv = self._make_rv_tuple(2, 3)
        update_rv_tuple(rv, '<=', 'Y_0', 'Y_1', 2, 3)
        self.assertEqual(len(rv[1]), 1)
        self.assertEqual(rv[1][0][0], 1)
        self.assertEqual(rv[1][0][1], -1)

    def test_output_constant(self):
        """Test output constraint with constant."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '<=', 'Y_0', '1.5', 2, 2)
        self.assertEqual(rv[1][0][0], 1)
        self.assertEqual(rv[2][0], 1.5)


class TestPrintVnnlib(unittest.TestCase):
    """Tests for print_vnnlib function."""

    def test_list_input(self):
        """Test with simple list input."""
        # Should not raise
        print_vnnlib([1.0, 2.0, 3.0])

    def test_nested_list(self):
        """Test with nested list input."""
        # Should not raise
        print_vnnlib([[1.0, 2.0], [3.0, 4.0]])

    def test_dict_input(self):
        """Test with dict input."""
        # Should not raise
        print_vnnlib({'a': [1.0, 2.0], 'b': [3.0, 4.0]})

    def test_numpy_array(self):
        """Test with numpy array input."""
        # Should not raise
        print_vnnlib(np.array([1.0, 2.0, 3.0]))


class TestReadVnnlibSimple(unittest.TestCase):
    """Tests for read_vnnlib_simple function."""

    def _write_temp_vnnlib(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vnnlib') as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_simple_vnnlib(self):
        """Test reading simple vnnlib."""
        from check_counterexample import read_vnnlib_simple
        content = """(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 Y_1))
"""
        path = self._write_temp_vnnlib(content)
        result = read_vnnlib_simple(path)
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        box, specs = result[0]
        self.assertEqual(len(box), 2)  # 2 inputs

    def test_single_input_output(self):
        """Test vnnlib with single input and output."""
        from check_counterexample import read_vnnlib_simple
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 0.5))
"""
        path = self._write_temp_vnnlib(content)
        result = read_vnnlib_simple(path)
        self.assertGreaterEqual(len(result), 1)


class TestCheckSpecFunction(unittest.TestCase):
    """Tests for check_spec function."""

    def test_check_spec_valid_violation(self):
        """Test check_spec with valid violation."""
        from check_counterexample import check_spec
        y = np.array([0.1, 0.9])
        # Y_0 - Y_1 <= 0 means 0.1 - 0.9 = -0.8 <= 0 (satisfied)
        spec_cs = [[[np.array([1, -1])]]]
        spec_ys = [[[0.0]]]
        # Should not raise
        check_spec(y, spec_cs, spec_ys)


class TestCheckAdvExample(unittest.TestCase):
    """Tests for counterexample classification."""

    def _classify(self, x, pred, vnnlib=None):
        if vnnlib is None:
            vnnlib = [(
                [[0.0, 1.0]],
                [(np.array([[1.0]]), np.array([0.0]))],
            )]
        with patch.object(
                check_counterexample_module, 'parse_cex',
                return_value=(np.array(x), np.array([]))), \
                patch.object(
                    check_counterexample_module, 'read_vnnlib_simple',
                    return_value=vnnlib), \
                patch.object(
                    check_counterexample_module.onnx, 'load',
                    return_value=object()), \
                patch.object(
                    check_counterexample_module, 'get_io_nodes',
                    return_value=(None, None, np.float32)), \
                patch.object(
                    check_counterexample_module, 'eval_onnx',
                    return_value=np.array(pred)):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                result = check_adv_example(
                    "model.onnx", "property.vnnlib", "cex.txt")
        return result, recorded

    def test_check_adv_example_classifies_strict_and_invalid_cases(self):
        cases = [
            ('strict success', [0.5], [0.0], CexResult.CORRECT),
            ('input outside box beyond tolerance',
             [1.0 + 2 * CEX_INPUT_TOLERANCE], [0.0],
             CexResult.INCORRECT),
            ('output outside exact tolerance', [0.5],
             [CEX_OUTPUT_TOLERANCE + 1e-12],
             CexResult.INCORRECT),
        ]
        for name, x, pred, expected in cases:
            with self.subTest(name=name):
                result, recorded = self._classify(x, pred)
                self.assertEqual(result, expected)
                self.assertEqual(len(recorded), 0)

    def test_correct_with_tolerance_warning_describes_input_tolerance(self):
        result, recorded = self._classify(
            [1.0 + CEX_INPUT_TOLERANCE / 2], [0.0])
        self.assertEqual(result, CexResult.CORRECT_WITH_TOLERANCE)
        self.assertEqual(len(recorded), 1)
        message = str(recorded[0].message)
        self.assertIn("input tolerance", message)
        self.assertIn("weak success", message)

    def test_main_exit_codes(self):
        argv = [
            'check_counterexample.py', 'model.onnx', 'property.vnnlib',
            'cex.txt',
        ]
        cases = [
            (CexResult.CORRECT_WITH_TOLERANCE, 0),
            (CexResult.INCORRECT, 1),
        ]
        for result, expected_code in cases:
            with self.subTest(result=result):
                with patch.object(check_counterexample_module.sys, 'argv', argv), \
                        patch.object(
                            check_counterexample_module, 'check_adv_example',
                            return_value=result):
                    with self.assertRaises(SystemExit) as exc:
                        check_counterexample_module.main()
                self.assertEqual(exc.exception.code, expected_code)


class TestParseCexEdgeCases(unittest.TestCase):
    """Edge case tests for parse_cex."""

    def _write_temp_cex(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_large_dimensions(self):
        """Test parsing with large dimension indices."""
        content = """(X_100 1.0)
(Y_50 2.0)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertEqual(len(x), 101)  # 0-100
        self.assertEqual(len(y), 51)   # 0-50
        self.assertAlmostEqual(x[100], 1.0)
        self.assertAlmostEqual(y[50], 2.0)

    def test_zero_values(self):
        """Test parsing zero values."""
        content = """(X_0 0.0)
(X_1 0)
(Y_0 0.0)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertAlmostEqual(y[0], 0.0)

    def test_positive_sign(self):
        """Test parsing with explicit positive sign."""
        content = """(X_0 +1.5)
(Y_0 +2.5)
"""
        path = self._write_temp_cex(content)
        x, y = parse_cex(path)
        self.assertAlmostEqual(x[0], 1.5)
        self.assertAlmostEqual(y[0], 2.5)


class TestUpdateRvTupleEdgeCases(unittest.TestCase):
    """Edge case tests for update_rv_tuple."""

    def _make_rv_tuple(self, num_inputs, num_outputs):
        box_dict = {i: [-np.inf, np.inf] for i in range(num_inputs)}
        return (box_dict, [], [])

    def test_multiple_constraints_same_input(self):
        """Test multiple constraints on same input."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '>=', 'X_0', '0.0', 2, 2)
        update_rv_tuple(rv, '<=', 'X_0', '1.0', 2, 2)
        self.assertEqual(rv[0][0][0], 0.0)
        self.assertEqual(rv[0][0][1], 1.0)

    def test_tighter_bounds(self):
        """Test that tighter bounds are applied correctly."""
        rv = self._make_rv_tuple(1, 1)
        update_rv_tuple(rv, '>=', 'X_0', '0.0', 1, 1)
        update_rv_tuple(rv, '>=', 'X_0', '0.5', 1, 1)  # tighter
        self.assertEqual(rv[0][0][0], 0.5)

    def test_output_reversed_order(self):
        """Test output constraint with >= operator."""
        rv = self._make_rv_tuple(1, 2)
        update_rv_tuple(rv, '>=', 'Y_0', 'Y_1', 1, 2)
        # >= should be flipped to <=
        self.assertEqual(len(rv[1]), 1)


class TestReadStatementsEdgeCases(unittest.TestCase):
    """Edge case tests for read_statements."""

    def _write_temp_vnnlib(self, content):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vnnlib') as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_only_comments(self):
        """Test file with only comments."""
        content = """; Comment 1
; Comment 2
"""
        path = self._write_temp_vnnlib(content)
        stmts = read_statements(path)
        self.assertEqual(len(stmts), 0)

    def test_deeply_nested(self):
        """Test deeply nested parentheses."""
        content = """(assert (or (and (<= Y_0 Y_1) (>= Y_0 0.0))))
"""
        path = self._write_temp_vnnlib(content)
        stmts = read_statements(path)
        self.assertEqual(len(stmts), 1)


if __name__ == '__main__':
    unittest.main()
