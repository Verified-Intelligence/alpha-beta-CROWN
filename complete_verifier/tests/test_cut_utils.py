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
"""Unit tests for cuts/cut_utils.py"""
import os
import sys
import tempfile
import struct
import unittest
import time
from unittest.mock import MagicMock, patch, PropertyMock

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuts.cut_utils import (
    read_cut, get_cplex_cut_timestamp, parse_cplex_indx,
    cut_analysis, close_cut_log, remove_cut_files,
    read_cut_efficient, parse_cplex_cuts, fetch_cut_from_cplex,
    terminate_mip_processes, terminate_mip_processes_by_c_matching,
    generate_cplex_cuts, clean_net_mps_process,
    cplex_update_general_beta, biccos_update_general_beta
)


class TempFileTestCase(unittest.TestCase):
    """Base test class with helper methods for creating temporary files."""

    def create_temp_file(self, content, suffix='', mode='wb'):
        """Create a temporary file with the given content and register cleanup.

        Args:
            content: File content (bytes for 'wb' mode, str for 'w' mode)
            suffix: File suffix (e.g., '.cut', '.cuts', '.indx')
            mode: File mode ('wb' for binary, 'w' for text)

        Returns:
            Path to the created temporary file
        """
        with tempfile.NamedTemporaryFile(mode=mode, delete=False, suffix=suffix) as f:
            f.write(content)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def create_temp_binary_file(self, content, suffix=''):
        """Create a temporary binary file with the given content."""
        return self.create_temp_file(content, suffix=suffix, mode='wb')

    def create_temp_text_file(self, content, suffix=''):
        """Create a temporary text file with the given content."""
        return self.create_temp_file(content, suffix=suffix, mode='w')


class TestReadCut(TempFileTestCase):
    """Tests for read_cut function."""

    def test_simple_relu_constraint(self):
        """Test parsing a simple ReLU constraint."""
        path = self.create_temp_text_file("1.0*relu_0_5 >= 0.5\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_decision'], [[0, 5]])
        self.assertEqual(cuts[0]['relu_coeffs'], [1.0])
        self.assertEqual(cuts[0]['bias'], 0.5)
        self.assertEqual(cuts[0]['c'], 1)

    def test_simple_pre_constraint(self):
        """Test parsing a pre-activation constraint."""
        path = self.create_temp_text_file("2.0*pre_1_10 <= 1.0\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['pre_decision'], [[1, 10]])
        self.assertEqual(cuts[0]['pre_coeffs'], [2.0])
        self.assertEqual(cuts[0]['c'], -1)

    def test_input_constraint(self):
        """Test parsing an input variable constraint."""
        path = self.create_temp_text_file("0.5*x_3 >= 0.1\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['x_decision'], [[-1, 3]])
        self.assertEqual(cuts[0]['x_coeffs'], [0.5])

    def test_arelu_constraint(self):
        """Test parsing an arelu (integer) constraint."""
        path = self.create_temp_text_file("1.0*arelu_2_7 >= 0.0\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['arelu_decision'], [[2, 7]])
        self.assertEqual(cuts[0]['arelu_coeffs'], [1.0])

    def test_multiple_constraints(self):
        """Test parsing multiple constraints."""
        path = self.create_temp_text_file(
            "1.0*relu_0_1 >= 0.5\n2.0*pre_1_2 <= 1.0\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 2)

    def test_mixed_constraint(self):
        """Test parsing a constraint with multiple variable types."""
        path = self.create_temp_text_file(
            "1.0*relu_0_1+2.0*pre_0_2 >= 0.5\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_decision'], [[0, 1]])
        self.assertEqual(cuts[0]['pre_decision'], [[0, 2]])

    def test_negative_coefficient(self):
        """Test parsing constraint with negative coefficient."""
        path = self.create_temp_text_file("-1.5*relu_0_3 >= -0.25\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_coeffs'], [-1.5])
        self.assertEqual(cuts[0]['bias'], -0.25)

    def test_constraint_with_spaces(self):
        """Test parsing constraint with spaces (should be removed)."""
        path = self.create_temp_text_file("1.0 * relu_0_5 >= 0.5\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_decision'], [[0, 5]])

    def test_constraint_with_multiple_terms(self):
        """Test parsing constraint with multiple terms of same type."""
        path = self.create_temp_text_file(
            "1.0*relu_0_1+2.0*relu_0_2+3.0*relu_0_3 >= 1.0\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(len(cuts[0]['relu_decision']), 3)
        self.assertEqual(cuts[0]['relu_coeffs'], [1.0, 2.0, 3.0])

    def test_constraint_with_subtraction(self):
        """Test parsing constraint with subtraction term."""
        path = self.create_temp_text_file(
            "1.0*relu_0_1-2.0*relu_0_2 >= 0.5\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_coeffs'], [1.0, -2.0])

    def test_zero_bias(self):
        """Test parsing constraint with zero bias."""
        path = self.create_temp_text_file("1.0*relu_0_1 >= 0\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['bias'], 0.0)

    def test_large_layer_index(self):
        """Test parsing constraint with large layer index."""
        path = self.create_temp_text_file("1.0*relu_99_1000 >= 0.5\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_decision'], [[99, 1000]])

    def test_empty_file(self):
        """Test parsing empty file."""
        path = self.create_temp_text_file("", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 0)

    def test_x_variable_single_index(self):
        """Test parsing x variable with single index (no layer)."""
        path = self.create_temp_text_file("1.0*x_5 >= 0.1\n", suffix='.cut')
        cuts = read_cut(path)
        self.assertEqual(len(cuts), 1)
        # x variables have layer=-1 by convention
        self.assertEqual(cuts[0]['x_decision'], [[-1, 5]])


class TestGetCplexCutTimestamp(TempFileTestCase):
    """Tests for get_cplex_cut_timestamp function."""

    def test_nonexistent_file(self):
        """Test with non-existent file."""
        result = get_cplex_cut_timestamp('/nonexistent/path/file.cuts')
        self.assertIsNone(result)

    def test_existing_file(self):
        """Test with existing file."""
        temp_path = self.create_temp_binary_file(b'test')
        result = get_cplex_cut_timestamp(temp_path)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, int)
        # Timestamp should be positive
        self.assertGreaterEqual(result, 0)

    def test_timestamp_changes(self):
        """Test that timestamp changes when file is modified."""
        temp_path = self.create_temp_binary_file(b'test1')
        ts1 = get_cplex_cut_timestamp(temp_path)
        time.sleep(0.02)  # Wait a bit
        with open(temp_path, 'w') as f:
            f.write('test2')
        ts2 = get_cplex_cut_timestamp(temp_path)
        # Timestamps may or may not differ depending on filesystem granularity
        self.assertIsNotNone(ts1)
        self.assertIsNotNone(ts2)

    def test_timestamp_modulo_operation(self):
        """Test that timestamp is within expected range due to modulo."""
        temp_path = self.create_temp_binary_file(b'test')
        result = get_cplex_cut_timestamp(temp_path)
        # Timestamp should be < 100000000 due to modulo
        self.assertLess(result, 100000000)


class TestParseCplexIndx(TempFileTestCase):
    """Tests for parse_cplex_indx function."""

    def _create_indx_file(self, names):
        """Create a valid INDX binary file."""
        # Build names string
        names_bytes = b'\x00'.join(n.encode('ascii') for n in names) + b'\x00'

        # Header: signature + first_col_num + num_cols + names_offset
        header = b'INDX' + struct.pack('iii', 0, len(names), 16)

        return self.create_temp_binary_file(header + names_bytes, suffix='.indx')

    def test_valid_indx_file(self):
        """Test parsing a valid INDX file."""
        names = ['var_0', 'var_1', 'var_2']
        path = self._create_indx_file(names)
        result = parse_cplex_indx(path)
        self.assertEqual(result, names)

    def test_single_name(self):
        """Test parsing INDX file with single name."""
        names = ['var_0']
        path = self._create_indx_file(names)
        result = parse_cplex_indx(path)
        self.assertEqual(result, names)

    def test_invalid_file(self):
        """Test parsing invalid file."""
        temp_path = self.create_temp_binary_file(b'invalid data')
        result = parse_cplex_indx(temp_path)
        self.assertIsNone(result)

    def test_nonexistent_file(self):
        """Test parsing non-existent file."""
        result = parse_cplex_indx('/nonexistent/path/file.indx')
        self.assertIsNone(result)

    def test_many_names(self):
        """Test parsing INDX file with many names."""
        names = [f'var_{i}' for i in range(100)]
        path = self._create_indx_file(names)
        result = parse_cplex_indx(path)
        self.assertEqual(result, names)

    def test_special_characters_in_names(self):
        """Test parsing INDX file with special characters in names."""
        names = ['ReLU/layer_0', 'aReLU/layer_1', 'inp_0']
        path = self._create_indx_file(names)
        result = parse_cplex_indx(path)
        self.assertEqual(result, names)

    def test_wrong_signature(self):
        """Test parsing file with wrong signature."""
        content = b'XXXX' + struct.pack('iii', 0, 1, 16) + b'var_0\x00'
        temp_path = self.create_temp_binary_file(content)
        result = parse_cplex_indx(temp_path)
        self.assertIsNone(result)


class TestParseCplexCuts(TempFileTestCase):
    """Tests for parse_cplex_cuts function."""

    def _create_cuts_file(self, cuts_data):
        """Create a valid CUTS binary file.

        cuts_data: list of tuples (indices, values, rhs)
        """
        num_rows = len(cuts_data)
        num_elements = sum(len(c[0]) for c in cuts_data)

        # Calculate offsets
        header_size = 28  # 4 + 6*4 bytes
        row_begin_size = num_rows * 8  # uint64
        rhs_size = num_rows * 8  # double
        indices_size = num_elements * 4  # int32

        row_begin_idx_offset = header_size
        rhs_values_offset = row_begin_idx_offset + row_begin_size
        row_indices_offset = rhs_values_offset + rhs_size
        row_values_offset = row_indices_offset + indices_size

        # Build file content
        header = b'CUTS' + struct.pack('6i',
            num_rows, num_elements,
            row_begin_idx_offset, rhs_values_offset,
            row_indices_offset, row_values_offset)

        # Row begin indices
        row_begin = []
        current_idx = 0
        for indices, values, rhs in cuts_data:
            row_begin.append(current_idx)
            current_idx += len(indices)
        row_begin_bytes = struct.pack(f'{num_rows}Q', *row_begin)

        # RHS values
        rhs_values = [c[2] for c in cuts_data]
        rhs_bytes = struct.pack(f'{num_rows}d', *rhs_values)

        # Flatten indices and values
        all_indices = []
        all_values = []
        for indices, values, rhs in cuts_data:
            all_indices.extend(indices)
            all_values.extend(values)

        indices_bytes = struct.pack(f'{num_elements}i', *all_indices) if all_indices else b''
        values_bytes = struct.pack(f'{num_elements}d', *all_values) if all_values else b''

        content = header + row_begin_bytes + rhs_bytes + indices_bytes + values_bytes
        return self.create_temp_binary_file(content, suffix='.cuts')

    def test_parse_empty_cuts(self):
        """Test parsing cuts file with no cuts."""
        # Create header only
        header = b'CUTS' + struct.pack('6i', 0, 0, 28, 28, 28, 28)
        temp_path = self.create_temp_binary_file(header, suffix='.cuts')
        cuts, timestamp = parse_cplex_cuts(temp_path, [], [], [])
        self.assertEqual(len(cuts), 0)

    def test_parse_single_input_cut(self):
        """Test parsing cuts with input variable."""
        var_names = ['inp_0', 'inp_1']
        relu_layer_names = []
        pre_relu_layer_names = []

        # Cut: inp_0 with coeff 1.0, rhs 0.5
        path = self._create_cuts_file([([0], [1.0], 0.5)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['x_decision'], [[-1, 0]])
        self.assertEqual(cuts[0]['x_coeffs'], [1.0])
        self.assertEqual(cuts[0]['bias'], 0.5)
        self.assertEqual(cuts[0]['c'], -1)

    def test_parse_relu_cut(self):
        """Test parsing cuts with ReLU variable."""
        var_names = ['ReLUlayer1_5']
        relu_layer_names = ['layer1']
        pre_relu_layer_names = ['pre_layer1']

        path = self._create_cuts_file([([0], [2.0], 1.0)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['relu_decision'], [[0, 5]])
        self.assertEqual(cuts[0]['relu_coeffs'], [2.0])

    def test_parse_arelu_cut(self):
        """Test parsing cuts with aReLU (integer) variable."""
        var_names = ['aReLUlayer1_3']
        relu_layer_names = ['layer1']
        pre_relu_layer_names = ['pre_layer1']

        path = self._create_cuts_file([([0], [1.0], 0.0)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['arelu_decision'], [[0, 3]])
        self.assertEqual(cuts[0]['arelu_coeffs'], [1.0])

    def test_parse_pre_relu_cut(self):
        """Test parsing cuts with pre-activation (lay) variable.

        The format is lay{layer_name}_{neuron_idx} where layer_name should
        match an entry in pre_relu_layer_names after removing 'lay' prefix.
        Note: layer_name should NOT contain 'lay' substring as it gets replaced.
        """
        # Variable format: lay{layer_name}_{neuron_idx}
        # After removing 'lay': 'laymynode_2' -> 'mynode_2' -> split -> ['mynode', '2']
        var_names = ['laymynode_2']
        relu_layer_names = ['relu1']
        pre_relu_layer_names = ['mynode']

        path = self._create_cuts_file([([0], [3.0], 1.5)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['pre_decision'], [[0, 2]])
        self.assertEqual(cuts[0]['pre_coeffs'], [3.0])

    def test_parse_skip_invalid_var(self):
        """Test that cuts with unknown variable types are skipped."""
        var_names = ['unknown_var_0']
        relu_layer_names = ['layer1']
        pre_relu_layer_names = ['pre_layer1']

        path = self._create_cuts_file([([0], [1.0], 0.5)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 0)  # Cut should be skipped

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent cuts file."""
        cuts, timestamp = parse_cplex_cuts('/nonexistent/file.cuts', [], [], [])
        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    def test_parse_invalid_signature(self):
        """Test parsing file with invalid signature."""
        temp_path = self.create_temp_binary_file(b'XXXX' + b'\x00' * 24, suffix='.cuts')
        cuts, timestamp = parse_cplex_cuts(temp_path, [], [], [])
        self.assertIsNone(cuts)

    def test_parse_multiple_cuts(self):
        """Test parsing multiple cuts."""
        var_names = ['inp_0', 'inp_1', 'ReLUlayer1_0']
        relu_layer_names = ['layer1']
        pre_relu_layer_names = ['pre_layer1']

        path = self._create_cuts_file([
            ([0], [1.0], 0.5),
            ([1, 2], [2.0, 3.0], 1.0),
        ])
        cuts, timestamp = parse_cplex_cuts(path, var_names, relu_layer_names, pre_relu_layer_names)
        self.assertEqual(len(cuts), 2)

    def test_parse_returns_timestamp(self):
        """Test that parse_cplex_cuts returns valid timestamp."""
        var_names = ['inp_0']
        path = self._create_cuts_file([([0], [1.0], 0.5)])
        cuts, timestamp = parse_cplex_cuts(path, var_names, [], [])
        self.assertIsNotNone(timestamp)
        self.assertIsInstance(timestamp, int)
        self.assertGreaterEqual(timestamp, 0)


class TestCutAnalysis(unittest.TestCase):
    """Tests for cut_analysis function."""

    def test_empty_cuts(self):
        """Test analysis with empty cuts list."""
        # Should not raise
        cut_analysis([])

    def test_single_cut(self):
        """Test analysis with single cut."""
        cuts = [{
            'arelu_coeffs': [1.0],
            'pre_coeffs': [2.0],
            'x_coeffs': [],
            'relu_coeffs': [3.0]
        }]
        # Should not raise, just prints
        cut_analysis(cuts)

    def test_multiple_cuts_different_lengths(self):
        """Test analysis with cuts of different lengths."""
        cuts = [
            {'arelu_coeffs': [1.0], 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0, 2.0], 'pre_coeffs': [3.0], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': list(range(25)), 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
        ]
        # Should not raise
        cut_analysis(cuts)

    def test_custom_parameters(self):
        """Test analysis with custom max_length and cluster_size."""
        cuts = [
            {'arelu_coeffs': [1.0] * 5, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0] * 10, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
        ]
        # Should not raise
        cut_analysis(cuts, max_length=10, cluster_size=2)

    def test_all_cuts_above_max_length(self):
        """Test analysis with all cuts above max_length."""
        cuts = [
            {'arelu_coeffs': [1.0] * 25, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0] * 30, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
        ]
        cut_analysis(cuts, max_length=20)

    def test_cuts_at_cluster_boundaries(self):
        """Test analysis with cuts at cluster boundaries."""
        cuts = [
            {'arelu_coeffs': [1.0] * 3, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0] * 6, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0] * 9, 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
        ]
        cut_analysis(cuts, max_length=20, cluster_size=3)

    def test_single_element_clusters(self):
        """Test analysis with cluster_size=1."""
        cuts = [
            {'arelu_coeffs': [1.0], 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
            {'arelu_coeffs': [1.0, 2.0], 'pre_coeffs': [], 'x_coeffs': [], 'relu_coeffs': []},
        ]
        cut_analysis(cuts, max_length=5, cluster_size=1)


class TestCloseRemoveFunctions(TempFileTestCase):
    """Tests for close_cut_log and remove_cut_files functions."""

    def test_close_cut_log_no_logfile(self):
        """Test close_cut_log when no logfile exists."""
        processes = {0: {}}
        # Should not raise
        close_cut_log(processes, 0)

    def test_close_cut_log_none_logfile(self):
        """Test close_cut_log when logfile is None."""
        processes = {0: {'_logfile': None}}
        # Should not raise
        close_cut_log(processes, 0)

    def test_close_cut_log_valid_fd(self):
        """Test close_cut_log with a valid file descriptor."""
        temp_path = self.create_temp_binary_file(b'')
        fd = os.open(temp_path, os.O_RDONLY)
        processes = {0: {'_logfile': fd}}
        close_cut_log(processes, 0)
        # File descriptor should be closed, attempting to close again should fail
        with self.assertRaises(OSError):
            os.close(fd)

    def test_close_cut_log_invalid_fd(self):
        """Test close_cut_log with an invalid file descriptor."""
        processes = {0: {'_logfile': 99999}}  # Invalid fd
        # Should not raise (exception is caught)
        close_cut_log(processes, 0)

    def test_remove_cut_files_no_fname(self):
        """Test remove_cut_files when fname is not set."""
        processes = {0: {}}
        # Should not raise, just prints warning
        remove_cut_files(processes, 0)

    def test_remove_cut_files_nonexistent(self):
        """Test remove_cut_files with non-existent files."""
        processes = {0: {'_fname_stamped': '/nonexistent/path/model'}}
        # Should not raise
        remove_cut_files(processes, 0)

    def test_remove_cut_files_existing(self):
        """Test remove_cut_files with existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, 'model')
            # Create test files
            for ext in ['.mps', '.cuts', '.indx', '.log']:
                with open(base_path + ext, 'w') as f:
                    f.write('test')

            processes = {0: {'_fname_stamped': base_path}}
            remove_cut_files(processes, 0)

            # Files should be removed
            for ext in ['.mps', '.cuts', '.indx', '.log']:
                self.assertFalse(os.path.exists(base_path + ext))

    def test_remove_cut_files_partial(self):
        """Test remove_cut_files with only some files existing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, 'model')
            # Create only some files
            for ext in ['.mps', '.cuts']:
                with open(base_path + ext, 'w') as f:
                    f.write('test')

            processes = {0: {'_fname_stamped': base_path}}
            remove_cut_files(processes, 0)

            # Existing files should be removed
            self.assertFalse(os.path.exists(base_path + '.mps'))
            self.assertFalse(os.path.exists(base_path + '.cuts'))


class TestReadCutEfficient(unittest.TestCase):
    """Tests for read_cut_efficient function."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.var_names = None
        mock_net.net.cut_timestamp = -1
        mock_net.net.relus = []
        return mock_net

    @patch('cuts.cut_utils.get_cplex_cut_timestamp')
    def test_nonexistent_cut_file(self, mock_timestamp):
        """Test read_cut_efficient when cut file doesn't exist."""
        mock_timestamp.return_value = None
        mock_net = self._create_mock_net()

        cuts, timestamp = read_cut_efficient(mock_net, '/fake/path.cuts', '/fake/path.indx')

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    @patch('cuts.cut_utils.get_cplex_cut_timestamp')
    def test_same_timestamp_no_update(self, mock_timestamp):
        """Test read_cut_efficient when timestamp hasn't changed."""
        mock_timestamp.return_value = 12345
        mock_net = self._create_mock_net()
        mock_net.net.cut_timestamp = 12345  # Same as returned timestamp
        mock_net.net.var_names = ['var1']  # Already parsed

        cuts, timestamp = read_cut_efficient(mock_net, '/fake/path.cuts', '/fake/path.indx')

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    @patch('cuts.cut_utils.parse_cplex_cuts')
    @patch('cuts.cut_utils.parse_cplex_indx')
    @patch('cuts.cut_utils.get_cplex_cut_timestamp')
    def test_new_cuts_parsed(self, mock_timestamp, mock_indx, mock_cuts):
        """Test read_cut_efficient parses new cuts when timestamp differs."""
        mock_timestamp.return_value = 12345
        mock_indx.return_value = ['var1', 'var2']
        mock_cuts.return_value = ([{'cut': 1}], 12345)

        mock_net = self._create_mock_net()
        mock_net.net.cut_timestamp = -1  # No previous timestamp

        # Create mock relus
        mock_relu = MagicMock()
        mock_relu.name = 'relu1'
        mock_relu.inputs = [MagicMock()]
        mock_relu.inputs[0].name = 'pre_relu1'
        mock_net.net.relus = [mock_relu]

        cuts, timestamp = read_cut_efficient(mock_net, '/fake/path.cuts', '/fake/path.indx')

        self.assertEqual(cuts, [{'cut': 1}])
        self.assertEqual(timestamp, 12345)
        mock_indx.assert_called_once()

    @patch('cuts.cut_utils.parse_cplex_cuts')
    @patch('cuts.cut_utils.parse_cplex_indx')
    @patch('cuts.cut_utils.get_cplex_cut_timestamp')
    def test_var_names_already_parsed(self, mock_timestamp, mock_indx, mock_cuts):
        """Test that var_names is not re-parsed if already set."""
        mock_timestamp.return_value = 12346  # Different timestamp
        mock_cuts.return_value = ([{'cut': 2}], 12346)

        mock_net = self._create_mock_net()
        mock_net.net.cut_timestamp = 12345  # Previous timestamp
        mock_net.net.var_names = ['var1']  # Already parsed
        mock_net.net.relus = []

        cuts, timestamp = read_cut_efficient(mock_net, '/fake/path.cuts', '/fake/path.indx')

        mock_indx.assert_not_called()  # Should not parse indx again


class TestFetchCutFromCplex(unittest.TestCase):
    """Tests for fetch_cut_from_cplex function."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.mip_building_proc = None
        mock_net.processes = None
        mock_net.c = torch.tensor([[1, 0]])
        mock_net.cutter = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.relus = []
        return mock_net

    def test_mip_process_error(self):
        """Test fetch_cut_from_cplex when MIP process failed."""
        mock_net = self._create_mock_net()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.exitcode = 1  # Error exit code

        cuts, timestamp = fetch_cut_from_cplex(mock_net)

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    def test_no_processes_yet(self):
        """Test fetch_cut_from_cplex when processes dict not ready."""
        mock_net = self._create_mock_net()
        mock_net.processes = None

        cuts, timestamp = fetch_cut_from_cplex(mock_net)

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    @patch('cuts.cut_utils.read_cut_efficient')
    def test_matching_process_found(self, mock_read):
        """Test fetch_cut_from_cplex when matching process is found."""
        mock_read.return_value = ([{'cut': 1}], 12345)

        mock_net = self._create_mock_net()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.exitcode = None  # Still running
        mock_net.processes = {
            0: {'c': torch.tensor([[1, 0]]), '_fname_stamped': '/tmp/model'}
        }

        cuts, timestamp = fetch_cut_from_cplex(mock_net, sync_to_net=False)

        self.assertEqual(cuts, [{'cut': 1}])
        self.assertEqual(timestamp, 12345)

    @patch('cuts.cut_utils.read_cut_efficient')
    def test_sync_to_net_true(self, mock_read):
        """Test fetch_cut_from_cplex with sync_to_net=True."""
        mock_read.return_value = ([{'cut': 1}], 12345)

        mock_net = self._create_mock_net()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.exitcode = None
        mock_net.processes = {
            0: {'c': torch.tensor([[1, 0]]), '_fname_stamped': '/tmp/model'}
        }
        mock_net.cutter.construct_cut_module = MagicMock(return_value=MagicMock())

        cuts, timestamp = fetch_cut_from_cplex(mock_net, sync_to_net=True)

        # Verify net was updated
        mock_net.cutter.construct_cut_module.assert_called_once()
        self.assertEqual(mock_net.net.cut_timestamp, 12345)

    @patch('cuts.cut_utils.read_cut_efficient')
    def test_no_matching_process(self, mock_read):
        """Test fetch_cut_from_cplex when no matching process found."""
        mock_net = self._create_mock_net()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.exitcode = None
        mock_net.processes = {
            0: {'c': torch.tensor([[0, 1]]), '_fname_stamped': '/tmp/model'}  # Different c
        }

        cuts, timestamp = fetch_cut_from_cplex(mock_net)

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)

    @patch('cuts.cut_utils.read_cut_efficient')
    def test_cuts_none_returns_none(self, mock_read):
        """Test fetch_cut_from_cplex when read_cut_efficient returns None."""
        mock_read.return_value = (None, -1)

        mock_net = self._create_mock_net()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.exitcode = None
        mock_net.processes = {
            0: {'c': torch.tensor([[1, 0]]), '_fname_stamped': '/tmp/model'}
        }

        cuts, timestamp = fetch_cut_from_cplex(mock_net)

        self.assertIsNone(cuts)
        self.assertEqual(timestamp, -1)


class TestTerminateMipProcesses(unittest.TestCase):
    """Tests for terminate_mip_processes function."""

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    def test_terminate_single_process(self, mock_sleep, mock_pid_exists,
                                       mock_process, mock_close, mock_remove):
        """Test terminating a single MIP process."""
        mock_mip_proc = MagicMock()
        mock_mip_proc.is_alive.side_effect = [True, False]  # First alive, then terminated

        mock_pid_exists.side_effect = [True, False]  # Process exists, then terminated
        mock_process.return_value.kill = MagicMock()

        mock_watch_dog = MagicMock()
        mock_watch_dog.is_alive.return_value = False

        processes = {0: {'pid': 12345}}

        terminate_mip_processes(mock_mip_proc, processes, mock_watch_dog)

        mock_mip_proc.terminate.assert_called()
        mock_close.assert_called_once_with(processes, 0)
        mock_remove.assert_called_once_with(processes, 0)

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    def test_terminate_multiple_processes(self, mock_sleep, mock_pid_exists,
                                          mock_process, mock_close, mock_remove):
        """Test terminating multiple MIP processes."""
        mock_mip_proc = MagicMock()
        mock_mip_proc.is_alive.return_value = False

        mock_pid_exists.return_value = False  # Processes already terminated

        mock_watch_dog = MagicMock()
        mock_watch_dog.is_alive.return_value = False

        processes = {
            0: {'pid': 12345},
            1: {'pid': 12346},
        }

        terminate_mip_processes(mock_mip_proc, processes, mock_watch_dog)

        self.assertEqual(mock_close.call_count, 2)
        self.assertEqual(mock_remove.call_count, 2)

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    def test_handle_no_such_process(self, mock_sleep, mock_pid_exists,
                                     mock_process, mock_close, mock_remove):
        """Test handling NoSuchProcess exception."""
        from psutil import NoSuchProcess

        mock_mip_proc = MagicMock()
        mock_mip_proc.is_alive.return_value = False

        mock_pid_exists.side_effect = [True, False]
        mock_process.return_value.kill.side_effect = NoSuchProcess(12345)

        mock_watch_dog = MagicMock()
        mock_watch_dog.is_alive.return_value = False

        processes = {0: {'pid': 12345}}

        # Should not raise
        terminate_mip_processes(mock_mip_proc, processes, mock_watch_dog)


class TestTerminateMipProcessesByCMatching(unittest.TestCase):
    """Tests for terminate_mip_processes_by_c_matching function."""

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    def test_matching_c_found(self, mock_pid_exists, mock_process,
                               mock_close, mock_remove):
        """Test terminating process with matching c tensor."""
        mock_pid_exists.return_value = True
        mock_process.return_value.kill = MagicMock()

        c_list = [torch.tensor([[1, 0]])]
        processes = {
            0: {'c': torch.tensor([[1, 0]]), 'pid': 12345}
        }

        terminate_mip_processes_by_c_matching(processes, c_list)

        mock_process.return_value.kill.assert_called()
        mock_close.assert_called_once()
        mock_remove.assert_called_once()

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    def test_no_matching_c(self, mock_pid_exists, mock_process,
                           mock_close, mock_remove):
        """Test when no process matches c tensor."""
        c_list = [torch.tensor([[1, 0]])]
        processes = {
            0: {'c': torch.tensor([[0, 1]]), 'pid': 12345}  # Different c
        }

        terminate_mip_processes_by_c_matching(processes, c_list)

        mock_process.return_value.kill.assert_not_called()
        mock_close.assert_not_called()
        mock_remove.assert_not_called()

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    def test_multiple_c_values(self, mock_pid_exists, mock_process,
                                mock_close, mock_remove):
        """Test with multiple c values to match."""
        mock_pid_exists.return_value = True
        mock_process.return_value.kill = MagicMock()

        c_list = [torch.tensor([[1, 0]]), torch.tensor([[0, 1]])]
        processes = {
            0: {'c': torch.tensor([[1, 0]]), 'pid': 12345},
            1: {'c': torch.tensor([[0, 1]]), 'pid': 12346},
            2: {'c': torch.tensor([[1, 1]]), 'pid': 12347},  # No match
        }

        terminate_mip_processes_by_c_matching(processes, c_list)

        self.assertEqual(mock_process.return_value.kill.call_count, 2)
        self.assertEqual(mock_close.call_count, 2)
        self.assertEqual(mock_remove.call_count, 2)

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    def test_process_already_terminated(self, mock_pid_exists, mock_process,
                                         mock_close, mock_remove):
        """Test handling when process already terminated."""
        mock_pid_exists.return_value = False

        c_list = [torch.tensor([[1, 0]])]
        processes = {
            0: {'c': torch.tensor([[1, 0]]), 'pid': 12345}
        }

        terminate_mip_processes_by_c_matching(processes, c_list)

        mock_process.return_value.kill.assert_not_called()
        mock_close.assert_called_once()  # Still called even if process gone

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.close_cut_log')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    def test_handle_no_such_process_exception(self, mock_pid_exists, mock_process,
                                               mock_close, mock_remove):
        """Test handling NoSuchProcess exception during kill."""
        from psutil import NoSuchProcess

        mock_pid_exists.return_value = True
        mock_process.return_value.kill.side_effect = NoSuchProcess(12345)

        c_list = [torch.tensor([[1, 0]])]
        processes = {
            0: {'c': torch.tensor([[1, 0]]), 'pid': 12345}
        }

        # Should not raise
        terminate_mip_processes_by_c_matching(processes, c_list)


class TestGenerateCplexCuts(unittest.TestCase):
    """Tests for generate_cplex_cuts function."""

    @patch('cuts.cut_utils.fetch_cut_from_cplex')
    def test_generate_cuts(self, mock_fetch):
        """Test generate_cplex_cuts assigns cuts to net."""
        mock_fetch.return_value = ([{'cut': 1}], 12345)

        mock_net = MagicMock()
        mock_net.cutter = MagicMock()

        generate_cplex_cuts(mock_net)

        mock_fetch.assert_called_once_with(mock_net, sync_to_net=False, recorder=None)
        self.assertEqual(mock_net.cutter.cuts, [{'cut': 1}])
        self.assertEqual(mock_net.cutter.cut_timestamp, 12345)

    @patch('cuts.cut_utils.fetch_cut_from_cplex')
    def test_generate_cuts_none(self, mock_fetch):
        """Test generate_cplex_cuts when no cuts available."""
        mock_fetch.return_value = (None, -1)

        mock_net = MagicMock()
        mock_net.cutter = MagicMock()

        generate_cplex_cuts(mock_net)

        self.assertIsNone(mock_net.cutter.cuts)
        self.assertEqual(mock_net.cutter.cut_timestamp, -1)


class TestCleanNetMpsProcess(unittest.TestCase):
    """Tests for clean_net_mps_process function."""

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    def test_clean_with_processes(self, mock_sleep, mock_pid_exists,
                                   mock_process, mock_remove):
        """Test clean_net_mps_process with active processes."""
        mock_pid_exists.return_value = False  # Processes already terminated

        mock_net = MagicMock()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.is_alive.side_effect = [True, False]
        mock_net.processes = {
            0: {'pid': 12345, '_logfile': None}
        }

        clean_net_mps_process(mock_net)

        mock_net.mip_building_proc.kill.assert_called()
        mock_remove.assert_called_once()

    def test_clean_no_processes(self):
        """Test clean_net_mps_process when processes is None."""
        mock_net = MagicMock()
        mock_net.processes = None

        # Should not raise
        clean_net_mps_process(mock_net)

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    @patch('os.close')
    def test_clean_closes_logfiles(self, mock_os_close, mock_sleep,
                                    mock_pid_exists, mock_process, mock_remove):
        """Test clean_net_mps_process closes log files."""
        mock_pid_exists.return_value = False

        mock_net = MagicMock()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.is_alive.return_value = False
        mock_net.processes = {
            0: {'pid': 12345, '_logfile': 99}
        }

        clean_net_mps_process(mock_net)

        mock_os_close.assert_called_once_with(99)

    @patch('cuts.cut_utils.remove_cut_files')
    @patch('cuts.cut_utils.psutil.Process')
    @patch('cuts.cut_utils.psutil.pid_exists')
    @patch('cuts.cut_utils.time.sleep')
    def test_clean_handles_no_such_process(self, mock_sleep, mock_pid_exists,
                                            mock_process, mock_remove):
        """Test clean_net_mps_process handles NoSuchProcess exception."""
        from psutil import NoSuchProcess

        mock_pid_exists.side_effect = [True, False]
        mock_process.return_value.kill.side_effect = NoSuchProcess(12345)

        mock_net = MagicMock()
        mock_net.mip_building_proc = MagicMock()
        mock_net.mip_building_proc.is_alive.return_value = False
        mock_net.processes = {
            0: {'pid': 12345}
        }

        # Should not raise
        clean_net_mps_process(mock_net)


class TestCplexUpdateGeneralBeta(unittest.TestCase):
    """Tests for cplex_update_general_beta function."""

    def _create_mock_net(self):
        """Create a mock network with required attributes."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.cut_timestamp = 12345
        mock_net.net.final_name = 'final'
        mock_net.net.cut_module = MagicMock()
        mock_net.net.cut_module.general_beta = {
            'final': torch.ones(2, 1, 4, 10)  # Shape: (2, spec, batch, num_cuts)
        }
        mock_net.cutter = MagicMock()
        mock_net.cutter.beta_init = 0.5
        return mock_net

    def test_update_with_different_timestamp(self):
        """Test updating general_betas when timestamp differs."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [
                {'general_betas': torch.zeros(2, 1, 1, 10), 'cut_timestamp': 0}
            ]
        }

        cplex_update_general_beta(mock_net, selected_domains)

        # Check that general_betas was updated
        self.assertEqual(
            selected_domains['split_history'][0]['cut_timestamp'], 12345
        )

    def test_update_without_general_betas(self):
        """Test updating when general_betas not in split_history."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [{}]  # No general_betas key
        }

        cplex_update_general_beta(mock_net, selected_domains)

        # Should add general_betas
        self.assertIn('general_betas', selected_domains['split_history'][0])
        self.assertEqual(
            selected_domains['split_history'][0]['cut_timestamp'], 12345
        )

    def test_no_update_with_same_timestamp(self):
        """Test no update when timestamp is the same."""
        mock_net = self._create_mock_net()

        original_betas = torch.zeros(2, 1, 1, 10)
        selected_domains = {
            'split_history': [
                {'general_betas': original_betas.clone(), 'cut_timestamp': 12345}  # Same timestamp
            ]
        }

        cplex_update_general_beta(mock_net, selected_domains)

        # general_betas should not be changed
        self.assertTrue(torch.equal(
            selected_domains['split_history'][0]['general_betas'], original_betas
        ))

    def test_no_cut_module(self):
        """Test when cut_module.general_beta is None."""
        mock_net = self._create_mock_net()
        mock_net.net.cut_module.general_beta = None

        original_betas = torch.zeros(2, 1, 1, 10)
        selected_domains = {
            'split_history': [
                {'general_betas': original_betas.clone(), 'cut_timestamp': 0}
            ]
        }

        # Should not raise and not update
        cplex_update_general_beta(mock_net, selected_domains)

        # Verify general_betas and cut_timestamp were not updated
        self.assertTrue(torch.equal(
            selected_domains['split_history'][0]['general_betas'], original_betas
        ))
        self.assertEqual(selected_domains['split_history'][0]['cut_timestamp'], 0)

    def test_multiple_split_histories(self):
        """Test updating multiple split histories."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [
                {'general_betas': torch.zeros(2, 1, 1, 10), 'cut_timestamp': 0},
                {},  # No general_betas
                {'general_betas': torch.zeros(2, 1, 1, 10), 'cut_timestamp': 12345},  # Same timestamp
            ]
        }

        cplex_update_general_beta(mock_net, selected_domains)

        # First should be updated
        self.assertEqual(selected_domains['split_history'][0]['cut_timestamp'], 12345)
        # Second should be added
        self.assertIn('general_betas', selected_domains['split_history'][1])
        # Third should remain unchanged (same timestamp)


class TestBiccosUpdateGeneralBeta(unittest.TestCase):
    """Tests for biccos_update_general_beta function."""

    def _create_mock_net(self):
        """Create a mock network with required attributes."""
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.final_name = 'final'
        mock_net.net.cut_module = MagicMock()
        mock_net.net.cut_module.general_beta = {
            'final': torch.ones(2, 1, 4, 10)
        }
        mock_net.cutter = MagicMock()
        mock_net.cutter.beta_init = 0.5
        return mock_net

    def test_update_with_existing_general_betas(self):
        """Test updating existing general_betas."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [
                {'general_betas': torch.zeros(2, 1, 1, 10)}
            ]
        }

        biccos_update_general_beta(mock_net, selected_domains)

        # Check that general_betas was updated with beta_init value
        expected_val = mock_net.cutter.beta_init
        updated_betas = selected_domains['split_history'][0]['general_betas']
        self.assertTrue(torch.allclose(updated_betas, expected_val * torch.ones_like(updated_betas)))

    def test_update_without_general_betas(self):
        """Test updating when general_betas not in split_history."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [{}]
        }

        biccos_update_general_beta(mock_net, selected_domains)

        self.assertIn('general_betas', selected_domains['split_history'][0])

    def test_no_cut_module(self):
        """Test when cut_module.general_beta is None."""
        mock_net = self._create_mock_net()
        mock_net.net.cut_module.general_beta = None

        selected_domains = {
            'split_history': [
                {'general_betas': torch.zeros(2, 1, 1, 10)}
            ]
        }

        # Should not raise
        biccos_update_general_beta(mock_net, selected_domains)

    def test_no_cut_module_attr(self):
        """Test when cut_module doesn't exist."""
        mock_net = self._create_mock_net()
        mock_net.net.cut_module = None

        selected_domains = {
            'split_history': [{}]
        }

        # Should not update (condition not met)
        biccos_update_general_beta(mock_net, selected_domains)
        self.assertNotIn('general_betas', selected_domains['split_history'][0])

    def test_multiple_split_histories(self):
        """Test updating multiple split histories."""
        mock_net = self._create_mock_net()

        selected_domains = {
            'split_history': [
                {'general_betas': torch.zeros(2, 1, 1, 10)},
                {},
                {'general_betas': torch.ones(2, 1, 1, 10)},
            ]
        }

        biccos_update_general_beta(mock_net, selected_domains)

        # All should be updated
        for sh in selected_domains['split_history']:
            self.assertIn('general_betas', sh)


class TestEdgeCases(TempFileTestCase):
    """Tests for edge cases and error handling."""

    def test_read_cut_with_decimal_coeffs(self):
        """Test read_cut with decimal coefficients."""
        temp_path = self.create_temp_text_file(
            "0.123456*relu_0_1 >= 0.654321\n", suffix='.cut')
        cuts = read_cut(temp_path)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0]['relu_coeffs'][0], 0.123456, places=6)
        self.assertAlmostEqual(cuts[0]['bias'], 0.654321, places=6)

    def test_read_cut_with_scientific_notation(self):
        """Test read_cut with scientific notation coefficients."""
        temp_path = self.create_temp_text_file(
            "1e-5*relu_0_1 >= 1e5\n", suffix='.cut')
        cuts = read_cut(temp_path)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0]['relu_coeffs'][0], 1e-5, places=10)
        self.assertAlmostEqual(cuts[0]['bias'], 1e5, places=0)

    def test_close_cut_log_with_multiple_processes(self):
        """Test close_cut_log with multiple process entries."""
        temp_path1 = self.create_temp_binary_file(b'')
        temp_path2 = self.create_temp_binary_file(b'')

        fd1 = os.open(temp_path1, os.O_RDONLY)
        fd2 = os.open(temp_path2, os.O_RDONLY)

        processes = {
            0: {'_logfile': fd1},
            1: {'_logfile': fd2},
        }

        close_cut_log(processes, 0)
        close_cut_log(processes, 1)

        # Both should be closed
        with self.assertRaises(OSError):
            os.close(fd1)
        with self.assertRaises(OSError):
            os.close(fd2)

    def test_cut_analysis_with_zero_length_cut(self):
        """Test cut_analysis with a cut that has zero total length."""
        cuts = [{
            'arelu_coeffs': [],
            'pre_coeffs': [],
            'x_coeffs': [],
            'relu_coeffs': []
        }]
        # Should not raise
        cut_analysis(cuts)

    def test_parse_cplex_cuts_skip_last_layer_var(self):
        """Test parse_cplex_cuts skips constraints with last layer vars."""
        # Create a cuts file with a variable not in pre_relu_layer_names
        var_names = ['layoutput_0']  # This is output layer, not pre-relu
        relu_layer_names = ['relu1']
        pre_relu_layer_names = ['pre_relu1']  # output is not here

        # Create minimal cuts file
        header = b'CUTS' + struct.pack('6i', 1, 1, 28, 36, 44, 48)
        row_begin = struct.pack('Q', 0)
        rhs = struct.pack('d', 0.5)
        indices = struct.pack('i', 0)
        values = struct.pack('d', 1.0)

        content = header + row_begin + rhs + indices + values
        temp_path = self.create_temp_binary_file(content, suffix='.cuts')

        cuts, timestamp = parse_cplex_cuts(temp_path, var_names,
                                           relu_layer_names, pre_relu_layer_names)
        # Cut should be skipped because layoutput is not in pre_relu_layer_names
        self.assertEqual(len(cuts), 0)


if __name__ == '__main__':
    unittest.main()
