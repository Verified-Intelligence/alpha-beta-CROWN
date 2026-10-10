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
"""Unit tests for read_vnnlib.py"""
import hashlib
import os
import pickle
import sys
import tempfile
import unittest
import pytest
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from read_vnnlib import read_statements, update_rv_tuple, make_input_box_dict, read_vnnlib


@pytest.fixture(autouse=True)
def setup_arguments():
    """Setup arguments.Config for testing."""
    import arguments
    original_config = arguments.Config
    try:
        new_config = arguments.ConfigHandler()
        new_config.construct_config_dict(new_config.default_args)
        new_config.file = None
        new_config['debug']['rescale_vnnlib_ptb'] = None
        arguments.Config = new_config
        yield
    finally:
        arguments.Config = original_config


class TestReadStatements(unittest.TestCase):
    """Tests for read_statements function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def _write_temp_vnnlib(self, content):
        path = os.path.join(self.temp_dir, 'test.vnnlib')
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_simple_statements(self):
        """Test parsing simple single-line statements."""
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        self.assertEqual(len(statements), 3)
        self.assertEqual(statements[0], '(declare-const X_0 Real)')
        self.assertEqual(statements[1], '(declare-const Y_0 Real)')
        self.assertEqual(statements[2], '(assert (<= X_0 1.0))')

    def test_multiline_statement(self):
        """Test parsing statements spanning multiple lines."""
        content = """(assert (and
    (<= X_0 1.0)
    (>= X_0 0.0)))"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        self.assertEqual(len(statements), 1)
        self.assertIn('and', statements[0])
        self.assertIn('<= X_0 1.0', statements[0])

    def test_comments_removed(self):
        """Test that comments are properly removed."""
        content = """; This is a comment
(declare-const X_0 Real) ; inline comment
; Another comment
(declare-const Y_0 Real)"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        self.assertEqual(len(statements), 2)
        self.assertNotIn(';', statements[0])
        self.assertNotIn('comment', statements[0])

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        content = """(declare-const    X_0    Real)
(assert  (<=   X_0   1.0))"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        # Multiple spaces should be collapsed to single space
        self.assertEqual(statements[0], '(declare-const X_0 Real)')
        self.assertEqual(statements[1], '(assert (<= X_0 1.0))')

    def test_parenthesis_space_removal(self):
        """Test that spaces after parentheses are removed."""
        content = """( declare-const X_0 Real)
(assert ( <= X_0 1.0) )"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        self.assertTrue(statements[0].startswith('(declare'))
        self.assertNotIn('( ', statements[0])

    def test_empty_lines_ignored(self):
        """Test that empty lines are ignored."""
        content = """(declare-const X_0 Real)

(declare-const Y_0 Real)

"""
        path = self._write_temp_vnnlib(content)
        statements = read_statements(path)
        self.assertEqual(len(statements), 2)


class TestMakeInputBoxDict(unittest.TestCase):
    """Tests for make_input_box_dict function."""

    def test_creates_correct_size(self):
        """Test that dict has correct number of entries."""
        box = make_input_box_dict(5)
        self.assertEqual(len(box), 5)

    def test_indices_correct(self):
        """Test that indices 0 to num_inputs-1 exist."""
        box = make_input_box_dict(3)
        self.assertIn(0, box)
        self.assertIn(1, box)
        self.assertIn(2, box)
        self.assertNotIn(3, box)

    def test_default_bounds_infinite(self):
        """Test that default bounds are -inf, inf."""
        import numpy as np
        box = make_input_box_dict(2)
        self.assertEqual(box[0][0], -np.inf)
        self.assertEqual(box[0][1], np.inf)
        self.assertEqual(box[1][0], -np.inf)
        self.assertEqual(box[1][1], np.inf)

    def test_zero_inputs(self):
        """Test with zero inputs."""
        box = make_input_box_dict(0)
        self.assertEqual(len(box), 0)


class TestUpdateRvTuple(unittest.TestCase):
    """Tests for update_rv_tuple function."""

    def _make_rv_tuple(self, num_inputs, num_outputs):
        import numpy as np
        box_dict = {i: [-np.inf, np.inf] for i in range(num_inputs)}
        return (box_dict, [], [])

    def test_input_upper_bound(self):
        """Test setting input upper bound with <=."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '<=', 'X_0', '1.5', 2, 2)
        self.assertEqual(rv[0][0][1], 1.5)

    def test_input_lower_bound(self):
        """Test setting input lower bound with >=."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '>=', 'X_1', '0.5', 2, 2)
        self.assertEqual(rv[0][1][0], 0.5)

    def test_input_bound_min_max(self):
        """Test that multiple bounds take min/max correctly."""
        rv = self._make_rv_tuple(2, 2)
        # Set upper bound twice, should keep minimum
        update_rv_tuple(rv, '<=', 'X_0', '2.0', 2, 2)
        update_rv_tuple(rv, '<=', 'X_0', '1.0', 2, 2)
        self.assertEqual(rv[0][0][1], 1.0)
        # Set lower bound twice, should keep maximum
        update_rv_tuple(rv, '>=', 'X_0', '0.1', 2, 2)
        update_rv_tuple(rv, '>=', 'X_0', '0.5', 2, 2)
        self.assertEqual(rv[0][0][0], 0.5)

    def test_output_constraint_single_var(self):
        """Test output constraint with single variable."""
        rv = self._make_rv_tuple(2, 3)
        update_rv_tuple(rv, '<=', 'Y_0', '1.0', 2, 3)
        mat, rhs_list = rv[1], rv[2]
        self.assertEqual(len(mat), 1)
        self.assertEqual(mat[0][0], 1)
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[0][2], 0)
        self.assertEqual(rhs_list[0], 1.0)

    def test_output_constraint_two_vars(self):
        """Test output constraint comparing two variables (Y_0 <= Y_1)."""
        rv = self._make_rv_tuple(2, 3)
        update_rv_tuple(rv, '<=', 'Y_0', 'Y_1', 2, 3)
        mat, rhs_list = rv[1], rv[2]
        self.assertEqual(len(mat), 1)
        self.assertEqual(mat[0][0], 1)   # Y_0 coefficient
        self.assertEqual(mat[0][1], -1)  # Y_1 coefficient
        self.assertEqual(mat[0][2], 0)
        self.assertEqual(rhs_list[0], 0.0)

    def test_output_constraint_ge_swaps(self):
        """Test that >= swaps operands correctly."""
        rv = self._make_rv_tuple(2, 2)
        # Y_0 >= Y_1 should become Y_1 <= Y_0, i.e., Y_1 - Y_0 <= 0
        update_rv_tuple(rv, '>=', 'Y_0', 'Y_1', 2, 2)
        mat = rv[1]
        self.assertEqual(mat[0][0], -1)  # Y_0 coefficient (swapped)
        self.assertEqual(mat[0][1], 1)   # Y_1 coefficient (swapped)

    def test_output_constraint_constant_on_left(self):
        """Test output constraint with constant on left side (1.0 <= Y_0)."""
        rv = self._make_rv_tuple(2, 2)
        update_rv_tuple(rv, '<=', '1.0', 'Y_0', 2, 2)
        mat, rhs_list = rv[1], rv[2]
        self.assertEqual(mat[0][0], -1)
        self.assertEqual(rhs_list[0], -1.0)


# ============================================================================
# read_vnnlib Function Tests (pytest)
# ============================================================================

class TestReadVnnlibPytest:
    """Tests for read_vnnlib function."""

    def _write_temp_vnnlib(self, content, tmp_path):
        path = tmp_path / 'test.vnnlib'
        path.write_text(content)
        return str(path)

    def test_simple_vnnlib(self, tmp_path):
        """Test reading a simple vnnlib file."""
        content = """; Simple vnnlib file
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))

(assert (<= Y_0 Y_1))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1
        # Check input bounds
        box = result[0][0]
        assert len(box) == 2
        assert box[0] == (0.0, 1.0)
        assert box[1] == (0.0, 1.0)

        # Check output spec
        specs = result[0][1]
        assert len(specs) >= 1

    def test_vnnlib_with_disjunction(self, tmp_path):
        """Test reading vnnlib with disjunction (or clause).

        A disjunction (or clause) with N conjuncts should result in the
        specification list containing N elements, one for each disjunct.
        The structure is: [(box, [spec1, spec2, ...]), ...]
        """
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

(assert (or (and (<= Y_0 Y_1))
            (and (<= Y_0 Y_2))))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1
        # The disjunction has 2 disjuncts: (<= Y_0 Y_1) and (<= Y_0 Y_2)
        # These should appear as separate specs in the specification list
        specs = result[0][1]
        assert len(specs) == 2, (
            f"Disjunction with 2 disjuncts should produce 2 specs, got {len(specs)}"
        )

    def test_vnnlib_caching(self, tmp_path):
        """Test that compiled vnnlib cache is created and used.

        The read_vnnlib function creates a .compiled cache file on first read.
        On subsequent reads with the same file (matching sha256), it should
        use the cache and print a message indicating cache use.
        """
        from unittest.mock import patch

        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

(assert (<= Y_0 0.5))
"""
        path = self._write_temp_vnnlib(content, tmp_path)

        # First read - creates the cache
        result1 = read_vnnlib(path)

        # Check compiled file was created
        compiled_path = path + '.compiled'
        assert os.path.exists(compiled_path), "Compiled cache file should be created"

        # Second read - should use cache (prints "Precompiled vnnlib file found")
        with patch('builtins.print') as mock_print:
            result2 = read_vnnlib(path)

            # Verify the cache was used by checking for the cache hit message
            print_calls = [str(call) for call in mock_print.call_args_list]
            cache_hit_message_found = any(
                'Precompiled vnnlib file found' in call for call in print_calls
            )
            assert cache_hit_message_found, (
                "Second read should use cache and print 'Precompiled vnnlib file found'"
            )

        # Results should be equivalent
        assert len(result1) == len(result2)

    def test_vnnlib_wrong_brackets_fixed(self, tmp_path):
        """Test that < and > are converted to <= and >=.

        The read_vnnlib function has a workaround that converts strict
        inequalities (< and >) to non-strict ones (<= and >=) since some
        vnnlib files incorrectly use strict inequalities.

        This test uses:
        - (> X_0 0.0) which should be treated as (>= X_0 0.0) -> lower bound 0.0
        - (< X_0 1.0) which should be treated as (<= X_0 1.0) -> upper bound 1.0
        - (< Y_0 0.5) which should be treated as (<= Y_0 0.5) -> output constraint
        """
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (> X_0 0.0))
(assert (< X_0 1.0))

(assert (< Y_0 0.5))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1

        # Verify that the bounds were parsed correctly after conversion
        # (> X_0 0.0) -> (>= X_0 0.0) -> lower bound = 0.0
        # (< X_0 1.0) -> (<= X_0 1.0) -> upper bound = 1.0
        box = result[0][0]
        assert box[0] == (0.0, 1.0), (
            f"Expected input bounds (0.0, 1.0) after < and > conversion, got {box[0]}"
        )

        # Verify output constraint was parsed
        # (< Y_0 0.5) -> (<= Y_0 0.5) -> Y_0 <= 0.5
        specs = result[0][1]
        assert len(specs) >= 1
        mat, rhs = specs[0]
        assert rhs[0] == 0.5, (
            f"Expected output constraint rhs=0.5 after < conversion, got {rhs[0]}"
        )

    def test_vnnlib_multiple_inputs(self, tmp_path):
        """Test vnnlib with many input variables."""
        content = """; Multiple inputs
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (>= X_2 -1.0))
(assert (<= X_2 1.0))
(assert (>= X_3 -1.0))
(assert (<= X_3 1.0))

(assert (<= Y_0 0.0))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1
        box = result[0][0]
        assert len(box) == 4
        for i in range(4):
            assert box[i] == (-1.0, 1.0)

    def test_vnnlib_multiple_outputs(self, tmp_path):
        """Test vnnlib with multiple output constraints.

        Multiple output constraints (conjunctions via separate assert statements)
        should be combined into a single spec with a matrix having multiple rows.
        Each row represents one constraint: Y_0 - Y_1 <= 0 and Y_0 - Y_2 <= 0.
        """
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

(assert (<= Y_0 Y_1))
(assert (<= Y_0 Y_2))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1
        specs = result[0][1]
        # Should have one spec (conjunction of constraints)
        assert len(specs) == 1, (
            f"Multiple conjunctive constraints should produce 1 spec, got {len(specs)}"
        )

        # The spec should have 2 rows (one per constraint)
        mat, rhs = specs[0]
        assert mat.shape[0] == 2, (
            f"Expected 2 constraint rows in matrix, got {mat.shape[0]}"
        )
        assert len(rhs) == 2, (
            f"Expected 2 rhs values, got {len(rhs)}"
        )

    def test_vnnlib_constant_output_constraint(self, tmp_path):
        """Test vnnlib with constant in output constraint."""
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

(assert (<= Y_0 0.5))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        assert len(result) >= 1
        # Check the mat and rhs
        mat, rhs = result[0][1][0]
        assert mat.shape == (1, 1)
        assert rhs.shape == (1,)

    def test_vnnlib_without_output_constraints_keeps_output_dimension(self, tmp_path):
        """Test that empty output specs still preserve declared output dimension."""
        content = """(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        result = read_vnnlib(path)

        mat, rhs = result[0][1][0]
        assert mat.shape == (0, 1)
        assert rhs.shape == (0,)

    def test_vnnlib_invalid_compiled_cache_regenerates(self, tmp_path):
        """Test that stale caches with 1-D empty matrices are regenerated."""
        content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
"""
        path = self._write_temp_vnnlib(content, tmp_path)
        compiled_path = path + '.compiled'
        with open(path, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        bad_cache = [([(0.0, 1.0)], [(np.array([], dtype=float), np.array([], dtype=float))])]
        with open(compiled_path, 'wb') as f:
            pickle.dump((bad_cache, sha), f, protocol=pickle.HIGHEST_PROTOCOL)

        result = read_vnnlib(path)

        mat, rhs = result[0][1][0]
        assert mat.shape == (0, 1)
        assert rhs.shape == (0,)


if __name__ == '__main__':
    unittest.main()
