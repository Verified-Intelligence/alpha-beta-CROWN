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
"""Tests for the VNN-LIB 2.0 compatibility lowerer."""

import importlib
import os
import sys

import numpy as np
import pytest
import vnnlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from read_vnnlib2 import is_vnnlib2_file, read_vnnlib2

read_vnnlib2_module = importlib.import_module("read_vnnlib2")


def _write(tmp_path, text):
    path = tmp_path / "case.vnnlib"
    path.write_text(text)
    return str(path)


def _single_input_output_property(lower_bound=0.0, upper_bound=1.0):
    return f"""
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] {lower_bound}) (<= X[0] {upper_bound})))
(assert (<= Y[0] 0.5))
"""


def test_read_vnnlib2_uses_compiled_cache(tmp_path, monkeypatch):
    path = _write(tmp_path, _single_input_output_property())

    lowered_vnnlib, metadata = read_vnnlib2(path)
    assert os.path.exists(path + ".compiled")

    def fail_parse(_path):
        raise AssertionError("read_vnnlib2 should have used the cache")

    monkeypatch.setattr(
        read_vnnlib2_module.vnnlib, "parse_query_file", fail_parse)

    cached_vnnlib, cached_metadata = read_vnnlib2(path)
    assert cached_metadata.total_input_size == metadata.total_input_size
    assert cached_metadata.total_output_size == metadata.total_output_size
    assert cached_vnnlib[0][0] == lowered_vnnlib[0][0]
    np.testing.assert_array_equal(
        cached_vnnlib[0][1][0][0], lowered_vnnlib[0][1][0][0])
    np.testing.assert_array_equal(
        cached_vnnlib[0][1][0][1], lowered_vnnlib[0][1][0][1])
    assert cached_metadata.strictness[0].input_lower.tolist() == (
        metadata.strictness[0].input_lower.tolist())
    assert cached_metadata.strictness[0].input_upper.tolist() == (
        metadata.strictness[0].input_upper.tolist())
    assert cached_metadata.strictness[0].specs[0].tolist() == (
        metadata.strictness[0].specs[0].tolist())


def test_read_vnnlib2_regenerates_stale_compiled_cache(tmp_path, monkeypatch):
    path = _write(tmp_path, _single_input_output_property())
    read_vnnlib2(path)

    os.remove(path)
    path = _write(tmp_path, _single_input_output_property(
        lower_bound=0.25, upper_bound=0.75))
    parse_calls = 0
    original_parse = read_vnnlib2_module.vnnlib.parse_query_file

    def counting_parse(parse_path):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(parse_path)

    monkeypatch.setattr(
        read_vnnlib2_module.vnnlib, "parse_query_file", counting_parse)

    lowered_vnnlib, _ = read_vnnlib2(path)
    assert parse_calls == 1
    assert lowered_vnnlib[0][0] == [[0.25, 0.75]]


def get_metadata(lowered):
    return lowered[1]


def get_branch_strictness(lowered, idx=0):
    return get_metadata(lowered).strictness[idx]


def test_is_vnnlib2_file_reads_declared_version(tmp_path):
    assert not is_vnnlib2_file(_write(tmp_path, "(declare-const X_0 Real)\n"))
    assert not is_vnnlib2_file(
        _write(tmp_path, "(vnnlib-version <1.0>)\n")
    )
    assert is_vnnlib2_file(
        _write(tmp_path, "; comment\n(vnnlib-version <2.0>)\n")
    )
    assert is_vnnlib2_file(
        _write(tmp_path, "(vnnlib-version\n  <2.0>)\n")
    )


def test_is_vnnlib2_file_rejects_malformed_declaration(tmp_path):
    path = _write(tmp_path, "(vnnlib-version <two>)\n")
    with pytest.raises(ValueError, match="Malformed VNN-LIB version"):
        is_vnnlib2_file(path)


def test_is_vnnlib2_file_stops_after_first_vnnlib1_statement(tmp_path):
    path = _write(tmp_path, "(declare-const X_0 Real)\n)\n")
    assert not is_vnnlib2_file(path)


def test_multi_network_equal_inputs_are_shared(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [2])
  (declare-output Y_f Real [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [2])
  (declare-output Y_g Real [1])
)
(assert (and (>= X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (and (>= X_f[1] 2.0) (<= X_f[1] 3.0)))
(assert (== X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (<= (- Y_g[0] Y_f[0]) 0.05))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [2.0, 3.0]]
    assert metadata.total_input_size == 2
    assert metadata.input_index_map == [0, 1, 0, 1]
    assert not metadata.include_input_passthrough
    mat, rhs = specs[0]
    np.testing.assert_array_equal(mat, np.array([[-1.0, 1.0]]))
    np.testing.assert_allclose(rhs, np.array([0.05]))


def test_multi_network_input_order_relation_uses_passthrough(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (and (>= X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (and (>= X_g[0] 0.0) (<= X_g[0] 2.0)))
(assert (<= X_f[0] X_g[0]))
(assert (<= Y_f[0] Y_g[0]))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [0.0, 2.0]]
    assert metadata.total_input_size == 2
    assert metadata.input_index_map == [0, 1]
    assert metadata.include_input_passthrough
    mat, rhs = specs[0]
    assert mat.shape == (2, 4)  # Y_f, Y_g, then two input passthrough values.
    np.testing.assert_array_equal(mat[0], np.array([0.0, 0.0, 1.0, -1.0]))
    np.testing.assert_allclose(rhs[0], 0.0)
    np.testing.assert_array_equal(mat[1], np.array([1.0, -1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rhs[1], 0.0)


def test_branch_local_equal_inputs_are_not_shared(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (and (>= X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (and (>= X_g[0] 0.0) (<= X_g[0] 1.0)))
(assert (or (== X_f[0] X_g[0]) (<= Y_f[0] Y_g[0])))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    assert metadata.total_input_size == 2
    assert metadata.input_index_map == [0, 1]
    assert metadata.include_input_passthrough
    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [0.0, 1.0]]
    assert [mat.shape for mat, _ in specs] == [(2, 4), (1, 4)]


def test_multiple_inputs_lower_to_one_flat_box(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X1 Real [1, 2])
  (declare-input X2 Real [1, 1, 2])
  (declare-output Y Real [1, 1])
)
(assert (and (>= X1[0,0] -1.0) (<= X1[0,0] 1.0)))
(assert (and (>= X1[0,1] 2.0) (<= X1[0,1] 3.0)))
(assert (and (>= X2[0,0,0] 4.0) (<= X2[0,0,0] 5.0)))
(assert (and (>= X2[0,0,1] 6.0) (<= X2[0,0,1] 7.0)))
(assert (> Y[0,0] 0.5))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    box, specs = lowered_vnnlib[0]
    assert box == [[-1.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert not metadata.include_input_passthrough
    mat, rhs = specs[0]
    np.testing.assert_array_equal(mat, np.array([[-1.0]]))
    np.testing.assert_allclose(rhs, np.array([-0.5]))
    assert metadata.strictness[0].specs[0][0]


def test_strict_output_bound_records_open_row(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (> Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    strictness = get_branch_strictness(lowered)
    assert strictness.specs[0].tolist() == [True]


def test_closed_output_bound_records_closed_row(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (<= Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    strictness = get_branch_strictness(lowered)
    assert strictness.specs[0].tolist() == [False]


def test_strict_input_bound_records_open_box(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (> X[0] 0.0) (< X[0] 1.0)))
(assert (<= Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    strictness = get_branch_strictness(lowered)
    assert strictness.input_lower.tolist() == [True]
    assert strictness.input_upper.tolist() == [True]


def test_tighter_input_bound_updates_strictness(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (> X[0] 0.0))
(assert (>= X[0] 0.1))
(assert (<= X[0] 1.0))
(assert (<= Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    box, _ = lowered[0][0]
    strictness = get_branch_strictness(lowered)
    assert box == [[0.1, 1.0]]
    assert strictness.input_lower.tolist() == [False]


def test_equal_input_bound_ors_strictness(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (>= X[0] 0.1))
(assert (> X[0] 0.1))
(assert (<= X[0] 1.0))
(assert (<= Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    strictness = get_branch_strictness(lowered)
    assert strictness.input_lower.tolist() == [True]


def test_shared_input_compression_preserves_strict_bound(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1])
  (declare-output Y_f Real [1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [1])
  (declare-output Y_g Real [1])
)
(assert (and (> X_f[0] 0.0) (<= X_f[0] 1.0)))
(assert (and (>= X_g[0] 0.0) (<= X_g[0] 1.0)))
(assert (== X_f[0] X_g[0]))
(assert (<= Y_f[0] Y_g[0]))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    box, _ = lowered_vnnlib[0]
    assert box == [[0.0, 1.0]]
    assert metadata.input_index_map == [0, 0]
    strictness = get_branch_strictness((lowered_vnnlib, metadata))
    assert strictness.input_lower.tolist() == [True]


def test_strict_input_relation_is_not_shared_equality(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-input Z Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (and (>= Z[0] 0.0) (<= Z[0] 1.0)))
(assert (< X[0] Z[0]))
(assert (<= Z[0] X[0]))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [0.0, 1.0]]
    assert metadata.input_index_map == [0, 1]

    mat, rhs = specs[0]
    assert mat.shape == (2, metadata.total_output_size)
    np.testing.assert_allclose(rhs, np.array([0.0, 0.0]))
    strictness = get_branch_strictness((lowered_vnnlib, metadata))
    assert strictness.specs[0].tolist() == [True, False]


def test_simple_input_bounds_bypass_dnf_lowering(tmp_path, monkeypatch):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [2])
  (declare-output Y Real [1])
)
(assert (>= X[0] 0.0))
(assert (and (<= X[0] 1.0) (>= X[1] -1.0) (<= X[1] 1.0)))
""")

    def fail_dnf(*_args, **_kwargs):
        raise AssertionError("simple input bounds should not use DNF lowering")

    monkeypatch.setattr(read_vnnlib2_module, "_assertion_to_dnf", fail_dnf)

    lowered_vnnlib, _ = read_vnnlib2(path)
    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [-1.0, 1.0]]
    assert specs[0][0].shape == (0, 1)


def test_or_input_bounds_use_generic_dnf_lowering(tmp_path, monkeypatch):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (or (<= X[0] -1.0) (>= X[0] 1.0)))
""")
    original_dnf = read_vnnlib2_module._assertion_to_dnf
    dnf_calls = 0

    def counting_dnf(*args, **kwargs):
        nonlocal dnf_calls
        dnf_calls += 1
        return original_dnf(*args, **kwargs)

    monkeypatch.setattr(
        read_vnnlib2_module, "_assertion_to_dnf", counting_dnf)

    lowered_vnnlib, _ = read_vnnlib2(path)
    boxes = sorted(box[0] for box, _ in lowered_vnnlib)
    assert dnf_calls == 1
    assert boxes == [[-np.inf, -1.0], [1.0, np.inf]]


def test_nonlinear_constraints_are_lowered_to_extra_outputs(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1, 2])
  (declare-output Y Real [1, 1])
)
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))
(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))
(assert (<= (* X[0,0] X[0,1]) Y[0,0]))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    assert len(metadata.nonlinear_outputs) == 1
    left, right = metadata.nonlinear_outputs[0]
    assert isinstance(left, vnnlib.Multiply)
    assert right.name == "Y"
    assert metadata.total_output_size == 2
    assert not metadata.include_input_passthrough
    mat, rhs = lowered_vnnlib[0][1][0]
    np.testing.assert_array_equal(mat, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(rhs, np.array([0.0]))


def test_not_equal_lowers_to_two_or_branches(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (!= Y[0] 0.0))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    assert len(lowered_vnnlib) == 1
    _, specs = lowered_vnnlib[0]
    assert len(specs) == 2
    np.testing.assert_array_equal(specs[0][0], np.array([[1.0]]))
    np.testing.assert_allclose(specs[0][1], np.array([0.0]))
    np.testing.assert_array_equal(specs[1][0], np.array([[-1.0]]))
    np.testing.assert_allclose(specs[1][1], np.array([0.0]))
    assert metadata.strictness[0].specs[0][0]
    assert metadata.strictness[0].specs[1][0]


def test_equal_lowers_to_closed_rows(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (== Y[0] 0.5))
""")
    lowered = read_vnnlib2(path)

    strictness = get_branch_strictness(lowered)
    assert strictness.specs[0].tolist() == [False, False]


def test_infeasible_or_branches_are_dropped(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] -2.0) (<= X[0] -1.0)))
(assert (or (>= X[0] 0.0) (<= Y[0] 1.0)))
""")
    lowered_vnnlib, _ = read_vnnlib2(path)

    assert len(lowered_vnnlib) == 1
    box, specs = lowered_vnnlib[0]
    assert box == [[-2.0, -1.0]]
    mat, rhs = specs[0]
    np.testing.assert_array_equal(mat, np.array([[1.0]]))
    np.testing.assert_allclose(rhs, np.array([1.0]))


def test_strict_output_contradiction_branch_is_dropped(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (or (and (< Y[0] 0.0) (>= Y[0] 0.0)) (<= Y[0] -1.0)))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    assert len(lowered_vnnlib) == 1
    _, specs = lowered_vnnlib[0]
    assert len(specs) == 1
    mat, rhs = specs[0]
    np.testing.assert_array_equal(mat, np.array([[1.0]]))
    np.testing.assert_allclose(rhs, np.array([-1.0]))
    assert metadata.strictness[0].specs[0].tolist() == [False]


def test_narrow_strict_output_interval_is_not_dropped(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1])
  (declare-output Y Real [1])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (and (>= Y[0] 0.0) (< Y[0] 0.0000000000001)))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    assert len(lowered_vnnlib) == 1
    _, specs = lowered_vnnlib[0]
    assert len(specs) == 1
    mat, rhs = specs[0]
    np.testing.assert_array_equal(mat, np.array([[-1.0], [1.0]]))
    np.testing.assert_allclose(rhs, np.array([-0.0, 1e-13]))
    assert metadata.strictness[0].specs[0].tolist() == [False, True]


def test_without_output_constraints_keeps_output_dimension(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [2])
  (declare-output Y Real [3])
)
(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))
(assert (and (>= X[1] 2.0) (<= X[1] 3.0)))
""")
    lowered_vnnlib, _ = read_vnnlib2(path)

    box, specs = lowered_vnnlib[0]
    assert box == [[0.0, 1.0], [2.0, 3.0]]
    mat, rhs = specs[0]
    assert mat.shape == (0, 3)
    assert rhs.shape == (0,)
