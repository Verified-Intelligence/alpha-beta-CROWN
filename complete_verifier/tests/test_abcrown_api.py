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
"""Unit tests for abcrown_api/api.py"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
import arguments

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import (
    _deep_update,
    _clone_config,
    _config_context,
    _shift_other_by_eps,
    _assign_path,
    _ensure_linear_expr,
    _ensure_predicate,
    _predicate_to_dnf,
    _parse_input_bounds,
    _negate_comparison,
    _STRICT_INEQUALITY_EPS,
    default_config,
    ConfigBuilder,
    VariableVector,
    LinearExpr,
    Predicate,
    ComparisonPredicate,
    AndPredicate,
    OrPredicate,
    VerificationSpec,
    SolveResult,
    ABCrownSolver,
    VNNCompInstance,
    input_vars,
    output_vars,
)


class TestDeepUpdate(unittest.TestCase):
    """Tests for _deep_update function."""

    def test_update_flat(self):
        """Test updating a flat dictionary."""
        base = {'a': 1, 'b': 2}
        updates = {'b': 3, 'c': 4}
        _deep_update(base, updates)
        self.assertEqual(base, {'a': 1, 'b': 3, 'c': 4})

    def test_update_nested(self):
        """Test updating nested dictionaries."""
        base = {'level1': {'a': 1, 'b': 2}}
        updates = {'level1': {'b': 3, 'c': 4}}
        _deep_update(base, updates)
        self.assertEqual(base['level1'], {'a': 1, 'b': 3, 'c': 4})

    def test_update_deep_nested(self):
        """Test updating deeply nested dictionaries."""
        base = {'l1': {'l2': {'l3': 1}}}
        updates = {'l1': {'l2': {'l3': 2, 'l3b': 3}}}
        _deep_update(base, updates)
        self.assertEqual(base['l1']['l2'], {'l3': 2, 'l3b': 3})

    def test_update_creates_new_keys(self):
        """Test that update creates new nested keys."""
        base = {}
        updates = {'new': {'key': 'value'}}
        _deep_update(base, updates)
        self.assertEqual(base, {'new': {'key': 'value'}})

    def test_update_non_mapping_raises(self):
        """Test that updating non-mapping with mapping raises TypeError."""
        base = {'key': 'string_value'}
        updates = {'key': {'nested': 'value'}}
        with self.assertRaises(TypeError):
            _deep_update(base, updates)


class TestCloneConfig(unittest.TestCase):
    """Tests for _clone_config function."""

    def test_clone_creates_deep_copy(self):
        """Test that clone creates a deep copy."""
        original = {'a': {'b': [1, 2, 3]}}
        cloned = _clone_config(original)
        cloned['a']['b'].append(4)
        self.assertEqual(original['a']['b'], [1, 2, 3])
        self.assertEqual(cloned['a']['b'], [1, 2, 3, 4])

    def test_clone_preserves_structure(self):
        """Test that clone preserves dictionary structure."""
        original = {'x': 1, 'y': {'z': 2}}
        cloned = _clone_config(original)
        self.assertEqual(original, cloned)


class TestShiftOtherByEps(unittest.TestCase):
    """Tests for _shift_other_by_eps function."""

    def test_shift_float(self):
        """Test shifting a float value."""
        result = _shift_other_by_eps(1.0, 0.5)
        self.assertEqual(result, 1.5)

    def test_shift_int(self):
        """Test shifting an integer value."""
        result = _shift_other_by_eps(10, -2.0)
        self.assertEqual(result, 8.0)

    def test_shift_tensor(self):
        """Test shifting a torch tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _shift_other_by_eps(tensor, 0.1)
        expected = torch.tensor([1.1, 2.1, 3.1])
        self.assertTrue(torch.allclose(result, expected))

    def test_shift_numpy_array(self):
        """Test shifting a numpy array."""
        arr = np.array([1.0, 2.0])
        result = _shift_other_by_eps(arr, 0.5)
        expected = np.array([1.5, 2.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shift_list(self):
        """Test shifting a list."""
        lst = [1.0, 2.0, 3.0]
        result = _shift_other_by_eps(lst, 1.0)
        self.assertEqual(list(result), [2.0, 3.0, 4.0])

    def test_shift_variable_vector_passthrough(self):
        """Test that VariableVector is passed through unchanged."""
        vv = VariableVector("input", 10)
        result = _shift_other_by_eps(vv, 1.0)
        self.assertIs(result, vv)

    def test_shift_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError."""
        with self.assertRaises(TypeError):
            _shift_other_by_eps(object(), 1.0)


class TestAssignPath(unittest.TestCase):
    """Tests for _assign_path function."""

    def test_assign_single_level(self):
        """Test assigning at single level."""
        target = {}
        _assign_path(target, ['key'], 'value')
        self.assertEqual(target, {'key': 'value'})

    def test_assign_multi_level(self):
        """Test assigning at multiple levels."""
        target = {}
        _assign_path(target, ['l1', 'l2', 'l3'], 'value')
        self.assertEqual(target, {'l1': {'l2': {'l3': 'value'}}})

    def test_assign_creates_intermediate_dicts(self):
        """Test that intermediate dictionaries are created."""
        target = {'existing': 'value'}
        _assign_path(target, ['new', 'nested', 'path'], 42)
        self.assertEqual(target['new']['nested']['path'], 42)
        self.assertEqual(target['existing'], 'value')

    def test_assign_overwrites_existing(self):
        """Test that existing values are overwritten."""
        target = {'l1': {'l2': 'old'}}
        _assign_path(target, ['l1', 'l2'], 'new')
        self.assertEqual(target['l1']['l2'], 'new')

    def test_assign_into_non_mapping_raises(self):
        """Test that assigning into non-mapping raises TypeError."""
        target = {'key': 'string'}
        with self.assertRaises(TypeError):
            _assign_path(target, ['key', 'nested'], 'value')


class TestDefaultConfig(unittest.TestCase):
    """Tests for default_config function."""

    def test_returns_dict(self):
        """Test that default_config returns a dictionary."""
        config = default_config()
        self.assertIsInstance(config, dict)

    def test_returns_clone(self):
        """Test that default_config returns a clone each time."""
        config1 = default_config()
        config2 = default_config()
        config1['test_key'] = 'test_value'
        self.assertNotIn('test_key', config2)

    def test_has_general_section(self):
        """Test that config has general section."""
        config = default_config()
        self.assertIn('general', config)

    def test_complete_verifier_is_auto(self):
        """Test that complete_verifier defaults to 'auto' in API."""
        config = default_config()
        self.assertEqual(config['general']['complete_verifier'], 'auto')


class TestConfigBuilder(unittest.TestCase):
    """Tests for ConfigBuilder class."""

    def test_init_from_defaults(self):
        """Test initialization from defaults."""
        builder = ConfigBuilder.from_defaults()
        config = builder.to_dict()
        self.assertIn('general', config)

    def test_init_from_config(self):
        """Test initialization from existing config."""
        base = {'custom': {'key': 'value'}}
        builder = ConfigBuilder.from_config(base)
        config = builder.to_dict()
        self.assertEqual(config['custom']['key'], 'value')

    def test_update_with_mapping(self):
        """Test update with a mapping."""
        builder = ConfigBuilder()
        builder.update({'general': {'seed': 42}})
        config = builder.to_dict()
        self.assertEqual(config['general']['seed'], 42)

    def test_update_with_callable(self):
        """Test update with a callable."""
        def modifier(cfg):
            cfg['general']['seed'] = 999
            return cfg
        builder = ConfigBuilder()
        builder.update(modifier)
        config = builder.to_dict()
        self.assertEqual(config['general']['seed'], 999)

    def test_update_with_kwargs(self):
        """Test update with keyword arguments using __ separator."""
        builder = ConfigBuilder()
        builder.update(general__seed=123)
        config = builder.to_dict()
        self.assertEqual(config['general']['seed'], 123)

    def test_update_chaining(self):
        """Test that update returns self for chaining."""
        builder = ConfigBuilder()
        result = builder.update({'general': {'seed': 1}})
        self.assertIs(result, builder)

    def test_set_is_alias_for_update(self):
        """Test that set works like update."""
        builder = ConfigBuilder()
        builder.set(general__seed=777)
        config = builder.to_dict()
        self.assertEqual(config['general']['seed'], 777)

    def test_replace_overwrites_config(self):
        """Test that replace overwrites entire config."""
        builder = ConfigBuilder()
        builder.replace({'new': 'config'})
        config = builder.to_dict()
        self.assertEqual(config, {'new': 'config'})

    def test_copy_creates_independent_builder(self):
        """Test that copy creates an independent builder."""
        builder1 = ConfigBuilder()
        builder1.update({'general': {'seed': 100}})
        builder2 = builder1.copy()
        builder2.update({'general': {'seed': 200}})
        self.assertEqual(builder1.to_dict()['general']['seed'], 100)
        self.assertEqual(builder2.to_dict()['general']['seed'], 200)

    def test_call_returns_dict(self):
        """Test that calling builder returns dict."""
        builder = ConfigBuilder()
        config = builder()
        self.assertIsInstance(config, dict)

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            yaml.dump({'general': {'seed': 555}}, f)
            config_path = f.name
        try:
            builder = ConfigBuilder.from_yaml(config_path)
            config = builder.to_dict()
            self.assertEqual(config['general']['seed'], 555)
        finally:
            os.unlink(config_path)

    def test_update_unsupported_modifier_raises(self):
        """Test that unsupported modifier type raises TypeError."""
        builder = ConfigBuilder()
        with self.assertRaises(TypeError):
            builder.update(12345)  # Not a mapping or callable


class TestVariableVector(unittest.TestCase):
    """Tests for VariableVector class."""

    def test_init_with_int_shape(self):
        """Test initialization with integer shape."""
        vv = VariableVector("input", 10)
        self.assertEqual(vv.shape, (10,))
        self.assertEqual(vv.size, 10)
        self.assertEqual(vv.kind, "input")

    def test_init_with_tuple_shape(self):
        """Test initialization with tuple shape."""
        vv = VariableVector("output", (3, 4, 5))
        self.assertEqual(vv.shape, (3, 4, 5))
        self.assertEqual(vv.size, 60)

    def test_init_with_torch_size(self):
        """Test initialization with torch.Size."""
        vv = VariableVector("input", torch.Size([2, 3]))
        self.assertEqual(vv.shape, (2, 3))
        self.assertEqual(vv.size, 6)

    def test_init_empty_shape_raises(self):
        """Test that empty shape raises ValueError."""
        with self.assertRaises(ValueError):
            VariableVector("input", ())

    def test_getitem_returns_linear_expr(self):
        """Test that indexing returns LinearExpr."""
        vv = VariableVector("input", 5)
        expr = vv[2]
        self.assertIsInstance(expr, LinearExpr)
        self.assertEqual(expr.coeffs, {("input", 2): 1.0})
        self.assertEqual(expr.constant, 0.0)

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        vv = VariableVector("output", 5)
        expr = vv[-1]
        self.assertEqual(expr.coeffs, {("output", 4): 1.0})

    def test_getitem_multi_dim(self):
        """Test multi-dimensional indexing."""
        vv = VariableVector("input", (3, 4))
        expr = vv[1, 2]
        # 1*4 + 2 = 6
        self.assertEqual(expr.coeffs, {("input", 6): 1.0})

    def test_getitem_out_of_range_raises(self):
        """Test that out-of-range index raises IndexError."""
        vv = VariableVector("input", 5)
        with self.assertRaises(IndexError):
            _ = vv[10]

    def test_getitem_single_index_on_multidim(self):
        """Test single index on multi-dim uses flat indexing."""
        vv = VariableVector("input", (3, 4))
        # Single index is interpreted as flat index
        expr = vv[5]
        self.assertEqual(expr.coeffs, {("input", 5): 1.0})

    def test_ge_comparison_with_float(self):
        """Test >= comparison with float creates Predicate."""
        vv = VariableVector("input", 3)
        pred = vv >= 0.5
        self.assertIsInstance(pred, Predicate)

    def test_le_comparison_with_float(self):
        """Test <= comparison with float creates Predicate."""
        vv = VariableVector("input", 3)
        pred = vv <= 1.0
        self.assertIsInstance(pred, Predicate)

    def test_gt_comparison_with_float(self):
        """Test > comparison creates Predicate with epsilon shift."""
        vv = VariableVector("output", 2)
        pred = vv > 0.0
        self.assertIsInstance(pred, Predicate)

    def test_lt_comparison_with_float(self):
        """Test < comparison creates Predicate with epsilon shift."""
        vv = VariableVector("output", 2)
        pred = vv < 1.0
        self.assertIsInstance(pred, Predicate)

    def test_comparison_with_tensor(self):
        """Test comparison with tensor."""
        vv = VariableVector("input", 3)
        bounds = torch.tensor([0.0, 0.1, 0.2])
        pred = vv >= bounds
        self.assertIsInstance(pred, Predicate)

    def test_comparison_with_numpy(self):
        """Test comparison with numpy array."""
        vv = VariableVector("input", 3)
        bounds = np.array([0.0, 0.1, 0.2])
        pred = vv <= bounds
        self.assertIsInstance(pred, Predicate)

    def test_comparison_with_list(self):
        """Test comparison with list."""
        vv = VariableVector("input", 2)
        pred = vv >= [0.0, 0.5]
        self.assertIsInstance(pred, Predicate)

    def test_comparison_size_mismatch_raises(self):
        """Test that size mismatch in comparison raises ValueError."""
        vv = VariableVector("input", 3)
        with self.assertRaises(ValueError):
            _ = vv >= [0.0, 0.5]  # Only 2 elements for 3-element vector


class TestInputOutputVars(unittest.TestCase):
    """Tests for input_vars and output_vars factory functions."""

    def test_input_vars_creates_input_vector(self):
        """Test input_vars creates vector with 'input' kind."""
        x = input_vars(10)
        self.assertEqual(x.kind, "input")
        self.assertEqual(x.size, 10)

    def test_output_vars_creates_output_vector(self):
        """Test output_vars creates vector with 'output' kind."""
        y = output_vars(5)
        self.assertEqual(y.kind, "output")
        self.assertEqual(y.size, 5)

    def test_input_vars_with_shape(self):
        """Test input_vars with multi-dimensional shape."""
        x = input_vars((3, 32, 32))
        self.assertEqual(x.shape, (3, 32, 32))
        self.assertEqual(x.size, 3072)


class TestLinearExpr(unittest.TestCase):
    """Tests for LinearExpr class."""

    def test_init_empty(self):
        """Test empty LinearExpr initialization."""
        expr = LinearExpr()
        self.assertEqual(expr.coeffs, {})
        self.assertEqual(expr.constant, 0.0)

    def test_init_with_coeffs(self):
        """Test initialization with coefficients."""
        expr = LinearExpr({("input", 0): 2.0, ("input", 1): 3.0}, 1.0)
        self.assertEqual(expr.coeffs[("input", 0)], 2.0)
        self.assertEqual(expr.coeffs[("input", 1)], 3.0)
        self.assertEqual(expr.constant, 1.0)

    def test_init_filters_zero_coeffs(self):
        """Test that zero coefficients are filtered out."""
        expr = LinearExpr({("input", 0): 0.0, ("input", 1): 1.0}, 0.0)
        self.assertNotIn(("input", 0), expr.coeffs)
        self.assertIn(("input", 1), expr.coeffs)

    def test_add_linear_exprs(self):
        """Test adding two LinearExprs."""
        expr1 = LinearExpr({("x", 0): 1.0}, 2.0)
        expr2 = LinearExpr({("x", 0): 2.0, ("x", 1): 1.0}, 3.0)
        result = expr1 + expr2
        self.assertEqual(result.coeffs[("x", 0)], 3.0)
        self.assertEqual(result.coeffs[("x", 1)], 1.0)
        self.assertEqual(result.constant, 5.0)

    def test_add_scalar(self):
        """Test adding scalar to LinearExpr."""
        expr = LinearExpr({("x", 0): 1.0}, 2.0)
        result = expr + 3.0
        self.assertEqual(result.constant, 5.0)
        self.assertEqual(result.coeffs[("x", 0)], 1.0)

    def test_radd_scalar(self):
        """Test reverse add with scalar."""
        expr = LinearExpr({("x", 0): 1.0}, 2.0)
        result = 3.0 + expr
        self.assertEqual(result.constant, 5.0)

    def test_sub_linear_exprs(self):
        """Test subtracting LinearExprs."""
        expr1 = LinearExpr({("x", 0): 5.0}, 10.0)
        expr2 = LinearExpr({("x", 0): 2.0}, 3.0)
        result = expr1 - expr2
        self.assertEqual(result.coeffs[("x", 0)], 3.0)
        self.assertEqual(result.constant, 7.0)

    def test_sub_scalar(self):
        """Test subtracting scalar from LinearExpr."""
        expr = LinearExpr({}, 10.0)
        result = expr - 3.0
        self.assertEqual(result.constant, 7.0)

    def test_rsub_scalar(self):
        """Test reverse subtraction with scalar."""
        expr = LinearExpr({("x", 0): 2.0}, 3.0)
        result = 10.0 - expr
        self.assertEqual(result.coeffs[("x", 0)], -2.0)
        self.assertEqual(result.constant, 7.0)

    def test_mul_scalar(self):
        """Test multiplying LinearExpr by scalar."""
        expr = LinearExpr({("x", 0): 2.0, ("x", 1): 3.0}, 4.0)
        result = expr * 2.0
        self.assertEqual(result.coeffs[("x", 0)], 4.0)
        self.assertEqual(result.coeffs[("x", 1)], 6.0)
        self.assertEqual(result.constant, 8.0)

    def test_rmul_scalar(self):
        """Test reverse multiplication with scalar."""
        expr = LinearExpr({("x", 0): 2.0}, 3.0)
        result = 3.0 * expr
        self.assertEqual(result.coeffs[("x", 0)], 6.0)
        self.assertEqual(result.constant, 9.0)

    def test_truediv_scalar(self):
        """Test dividing LinearExpr by scalar."""
        expr = LinearExpr({("x", 0): 4.0}, 8.0)
        result = expr / 2.0
        self.assertEqual(result.coeffs[("x", 0)], 2.0)
        self.assertEqual(result.constant, 4.0)

    def test_truediv_zero_raises(self):
        """Test that division by zero raises ZeroDivisionError."""
        expr = LinearExpr({("x", 0): 1.0}, 1.0)
        with self.assertRaises(ZeroDivisionError):
            _ = expr / 0.0

    def test_neg(self):
        """Test negation of LinearExpr."""
        expr = LinearExpr({("x", 0): 2.0, ("x", 1): -3.0}, 4.0)
        result = -expr
        self.assertEqual(result.coeffs[("x", 0)], -2.0)
        self.assertEqual(result.coeffs[("x", 1)], 3.0)
        self.assertEqual(result.constant, -4.0)

    def test_le_creates_comparison(self):
        """Test <= creates ComparisonPredicate."""
        expr = LinearExpr({("x", 0): 1.0}, 0.0)
        pred = expr <= 5.0
        self.assertIsInstance(pred, ComparisonPredicate)
        self.assertEqual(pred.op, "<=")

    def test_ge_creates_comparison(self):
        """Test >= creates ComparisonPredicate."""
        expr = LinearExpr({("x", 0): 1.0}, 0.0)
        pred = expr >= 0.0
        self.assertIsInstance(pred, ComparisonPredicate)
        self.assertEqual(pred.op, ">=")

    def test_lt_creates_comparison(self):
        """Test < creates ComparisonPredicate."""
        expr = LinearExpr({("y", 0): 1.0}, 0.0)
        pred = expr < 1.0
        self.assertIsInstance(pred, ComparisonPredicate)
        self.assertEqual(pred.op, "<")

    def test_gt_creates_comparison(self):
        """Test > creates ComparisonPredicate."""
        expr = LinearExpr({("y", 0): 1.0}, 0.0)
        pred = expr > 0.0
        self.assertIsInstance(pred, ComparisonPredicate)
        self.assertEqual(pred.op, ">")

    def test_eq_creates_and_predicate(self):
        """Test == creates conjunction of <= and >=."""
        expr = LinearExpr({("x", 0): 1.0}, 0.0)
        pred = expr == 5.0
        self.assertIsInstance(pred, AndPredicate)


class TestEnsureLinearExpr(unittest.TestCase):
    """Tests for _ensure_linear_expr helper."""

    def test_passthrough_linear_expr(self):
        """Test that LinearExpr is passed through."""
        expr = LinearExpr({("x", 0): 1.0}, 2.0)
        result = _ensure_linear_expr(expr)
        self.assertIs(result, expr)

    def test_convert_float(self):
        """Test converting float to LinearExpr."""
        result = _ensure_linear_expr(3.5)
        self.assertEqual(result.coeffs, {})
        self.assertEqual(result.constant, 3.5)

    def test_convert_int(self):
        """Test converting int to LinearExpr."""
        result = _ensure_linear_expr(7)
        self.assertEqual(result.coeffs, {})
        self.assertEqual(result.constant, 7.0)

    def test_unsupported_type_raises(self):
        """Test that unsupported type raises TypeError."""
        with self.assertRaises(TypeError):
            _ensure_linear_expr("string")


class TestComparisonPredicate(unittest.TestCase):
    """Tests for ComparisonPredicate class."""

    def test_init(self):
        """Test ComparisonPredicate initialization."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred = ComparisonPredicate(lhs, rhs, "<=")
        self.assertEqual(pred.op, "<=")

    def test_invalid_op_raises(self):
        """Test that invalid operator raises ValueError."""
        lhs = LinearExpr()
        rhs = LinearExpr()
        with self.assertRaises(ValueError):
            ComparisonPredicate(lhs, rhs, "==")

    def test_normalized_expr_le(self):
        """Test normalized_expr for <= operator."""
        # x[0] <= 5 means x[0] - 5 <= 0
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred = ComparisonPredicate(lhs, rhs, "<=")
        norm = pred.normalized_expr()
        self.assertEqual(norm.coeffs[("x", 0)], 1.0)
        self.assertEqual(norm.constant, -5.0)

    def test_normalized_expr_ge(self):
        """Test normalized_expr for >= operator."""
        # x[0] >= 0 means 0 - x[0] <= 0, i.e., -x[0] <= 0
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 0.0)
        pred = ComparisonPredicate(lhs, rhs, ">=")
        norm = pred.normalized_expr()
        self.assertEqual(norm.coeffs[("x", 0)], -1.0)
        self.assertEqual(norm.constant, 0.0)

    def test_normalized_expr_lt(self):
        """Test normalized_expr for < operator."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred = ComparisonPredicate(lhs, rhs, "<")
        norm = pred.normalized_expr()
        # Should have epsilon added
        self.assertAlmostEqual(norm.constant, -5.0 + _STRICT_INEQUALITY_EPS)

    def test_normalized_expr_gt(self):
        """Test normalized_expr for > operator."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 0.0)
        pred = ComparisonPredicate(lhs, rhs, ">")
        norm = pred.normalized_expr()
        self.assertAlmostEqual(norm.constant, _STRICT_INEQUALITY_EPS)


class TestNegateComparison(unittest.TestCase):
    """Tests for _negate_comparison function."""

    def test_negate_le(self):
        """Test negating <= gives >."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred = ComparisonPredicate(lhs, rhs, "<=")
        negated = _negate_comparison(pred)
        self.assertEqual(negated.op, ">")

    def test_negate_lt(self):
        """Test negating < gives >=."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred = ComparisonPredicate(lhs, rhs, "<")
        negated = _negate_comparison(pred)
        self.assertEqual(negated.op, ">=")

    def test_negate_ge(self):
        """Test negating >= gives <."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 0.0)
        pred = ComparisonPredicate(lhs, rhs, ">=")
        negated = _negate_comparison(pred)
        self.assertEqual(negated.op, "<")

    def test_negate_gt(self):
        """Test negating > gives <=."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 0.0)
        pred = ComparisonPredicate(lhs, rhs, ">")
        negated = _negate_comparison(pred)
        self.assertEqual(negated.op, "<=")


class TestAndPredicate(unittest.TestCase):
    """Tests for AndPredicate class."""

    def test_init(self):
        """Test AndPredicate initialization."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        rhs = LinearExpr({}, 5.0)
        pred1 = ComparisonPredicate(lhs, rhs, "<=")
        pred2 = ComparisonPredicate(lhs, LinearExpr({}, 0.0), ">=")
        and_pred = AndPredicate(pred1, pred2)
        self.assertIs(and_pred.left, pred1)
        self.assertIs(and_pred.right, pred2)

    def test_and_operator(self):
        """Test & operator creates AndPredicate."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        pred1 = ComparisonPredicate(lhs, LinearExpr({}, 5.0), "<=")
        pred2 = ComparisonPredicate(lhs, LinearExpr({}, 0.0), ">=")
        result = pred1 & pred2
        self.assertIsInstance(result, AndPredicate)


class TestOrPredicate(unittest.TestCase):
    """Tests for OrPredicate class."""

    def test_init(self):
        """Test OrPredicate initialization."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        pred1 = ComparisonPredicate(lhs, LinearExpr({}, 5.0), "<=")
        pred2 = ComparisonPredicate(lhs, LinearExpr({}, 10.0), ">=")
        or_pred = OrPredicate(pred1, pred2)
        self.assertIs(or_pred.left, pred1)
        self.assertIs(or_pred.right, pred2)

    def test_or_operator(self):
        """Test | operator creates OrPredicate."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        pred1 = ComparisonPredicate(lhs, LinearExpr({}, 5.0), "<=")
        pred2 = ComparisonPredicate(lhs, LinearExpr({}, 10.0), ">=")
        result = pred1 | pred2
        self.assertIsInstance(result, OrPredicate)


class TestEnsurePredicate(unittest.TestCase):
    """Tests for _ensure_predicate helper."""

    def test_passthrough_predicate(self):
        """Test that Predicate is passed through."""
        lhs = LinearExpr({("x", 0): 1.0}, 0.0)
        pred = ComparisonPredicate(lhs, LinearExpr({}, 5.0), "<=")
        result = _ensure_predicate(pred)
        self.assertIs(result, pred)

    def test_non_predicate_raises(self):
        """Test that non-Predicate raises TypeError."""
        with self.assertRaises(TypeError):
            _ensure_predicate(True)


class TestPredicateToDnf(unittest.TestCase):
    """Tests for _predicate_to_dnf function."""

    def test_single_comparison(self):
        """Test DNF of single comparison."""
        x = input_vars(2)
        pred = x[0] >= 0.0
        clauses = _predicate_to_dnf(pred)
        self.assertEqual(len(clauses), 1)
        self.assertEqual(len(clauses[0]), 1)

    def test_conjunction(self):
        """Test DNF of conjunction (AND)."""
        x = input_vars(2)
        pred = (x[0] >= 0.0) & (x[1] <= 1.0)
        clauses = _predicate_to_dnf(pred)
        self.assertEqual(len(clauses), 1)
        self.assertEqual(len(clauses[0]), 2)

    def test_disjunction(self):
        """Test DNF of disjunction (OR)."""
        x = input_vars(2)
        pred = (x[0] >= 0.0) | (x[1] >= 0.0)
        clauses = _predicate_to_dnf(pred)
        self.assertEqual(len(clauses), 2)

    def test_negation(self):
        """Test DNF with negation."""
        x = input_vars(2)
        pred = x[0] <= 1.0
        clauses = _predicate_to_dnf(pred, negate=True)
        # Negation of x[0] <= 1 is x[0] > 1
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0][0].op, ">")


class TestParseInputBounds(unittest.TestCase):
    """Tests for _parse_input_bounds function."""

    def test_simple_bounds(self):
        """Test parsing simple input bounds."""
        x = input_vars(2)
        pred = (x[0] >= 0.0) & (x[0] <= 1.0) & (x[1] >= 0.0) & (x[1] <= 1.0)
        lower, upper = _parse_input_bounds(pred, x)
        self.assertEqual(lower.shape, (1, 2))
        self.assertEqual(upper.shape, (1, 2))
        self.assertTrue(torch.allclose(lower, torch.tensor([[0.0, 0.0]])))
        self.assertTrue(torch.allclose(upper, torch.tensor([[1.0, 1.0]])))

    def test_missing_bounds_raises(self):
        """Test that missing bounds raises ValueError."""
        x = input_vars(2)
        pred = (x[0] >= 0.0) & (x[0] <= 1.0)  # Missing bounds for x[1]
        with self.assertRaises(ValueError):
            _parse_input_bounds(pred, x)

    def test_or_in_input_raises(self):
        """Test that OR in input constraint raises ValueError."""
        x = input_vars(2)
        pred = (x[0] >= 0.0) | (x[0] >= 0.5)
        with self.assertRaises(ValueError):
            _parse_input_bounds(pred, x)


class TestVerificationSpecInputSpec(unittest.TestCase):
    """Tests for VerificationSpec.InputSpec class."""

    def test_init_basic(self):
        """Test basic InputSpec initialization."""
        lower = torch.zeros(1, 10)
        upper = torch.ones(1, 10)
        spec = VerificationSpec.InputSpec(lower, upper)
        self.assertEqual(spec.num_inputs, 1)
        self.assertEqual(spec.data_shape, (10,))

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        lower = torch.zeros(1, 10)
        upper = torch.ones(1, 5)
        with self.assertRaises(ValueError):
            VerificationSpec.InputSpec(lower, upper)

    def test_missing_batch_dim_raises(self):
        """Test that missing batch dimension raises ValueError."""
        lower = torch.zeros(10)
        upper = torch.ones(10)
        with self.assertRaises(ValueError):
            VerificationSpec.InputSpec(lower, upper)

    def test_reshape(self):
        """Test reshaping input bounds."""
        lower = torch.zeros(1, 12)
        upper = torch.ones(1, 12)
        spec = VerificationSpec.InputSpec(lower, upper)
        spec.reshape((3, 4))
        self.assertEqual(spec.data_shape, (3, 4))
        self.assertEqual(spec.lower.shape, (1, 3, 4))

    def test_reshape_size_mismatch_raises(self):
        """Test that reshape with size mismatch raises ValueError."""
        lower = torch.zeros(1, 10)
        upper = torch.ones(1, 10)
        spec = VerificationSpec.InputSpec(lower, upper)
        with self.assertRaises(ValueError):
            spec.reshape((3, 4))  # 3*4=12 != 10


class TestVerificationSpecOutputSpec(unittest.TestCase):
    """Tests for VerificationSpec.OutputSpec class."""

    def test_init_basic(self):
        """Test basic OutputSpec initialization."""
        clauses = [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.OutputSpec(clauses)
        self.assertEqual(len(spec.clauses), 1)

    def test_empty_clauses_raises(self):
        """Test that empty clauses raises ValueError."""
        with self.assertRaises(ValueError):
            VerificationSpec.OutputSpec([])

    def test_normalize_broadcasts_shared_clauses(self):
        """Test that normalize broadcasts shared clauses."""
        clauses = [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.OutputSpec(clauses)
        spec.normalize(3)
        self.assertEqual(len(spec.clauses), 3)


class TestVerificationSpec(unittest.TestCase):
    """Tests for VerificationSpec class."""

    def test_build_from_input_bounds(self):
        """Test building spec from input bounds."""
        lower = torch.zeros(1, 5)
        upper = torch.ones(1, 5)
        clauses = [[(torch.tensor([[1.0, 0, 0, 0, -1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        self.assertEqual(spec.num_inputs, 1)
        self.assertEqual(spec.input_shape, (-1, 5))

    def test_build_from_center(self):
        """Test building spec from center and epsilon."""
        center = torch.tensor([[0.5, 0.5, 0.5]])
        epsilon = 0.1
        clauses = [[(torch.tensor([[1.0, -1.0, 0.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_from_center(center, epsilon, clauses)
        self.assertTrue(torch.allclose(spec.lower, torch.tensor([[0.4, 0.4, 0.4]])))
        self.assertTrue(torch.allclose(spec.upper, torch.tensor([[0.6, 0.6, 0.6]])))

    def test_build_from_expressions(self):
        """Test building spec from symbolic expressions."""
        x = input_vars(3)
        y = output_vars(2)
        input_constraint = (x >= 0.0) & (x <= 1.0)
        output_constraint = y[0] > y[1]
        spec = VerificationSpec.build_from_expressions(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        self.assertEqual(spec.input_spec.data_shape, (3,))

    def test_build_spec_bounds_mode(self):
        """Test build_spec with bounds mode."""
        lower = torch.zeros(1, 4)
        upper = torch.ones(1, 4)
        clauses = [[(torch.tensor([[1, -1, 0, 0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_spec(lower=lower, upper=upper, clauses=clauses)
        self.assertIsInstance(spec, VerificationSpec)

    def test_build_spec_center_mode(self):
        """Test build_spec with center/epsilon mode."""
        center = torch.zeros(1, 4)
        epsilon = 0.5
        clauses = [[(torch.tensor([[1, -1, 0, 0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_spec(center=center, epsilon=epsilon, clauses=clauses)
        self.assertIsInstance(spec, VerificationSpec)

    def test_build_spec_expression_mode(self):
        """Test build_spec with expression mode."""
        x = input_vars(3)
        y = output_vars(2)
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= 0) & (x <= 1),
            output_constraint=y[0] > y[1],
        )
        self.assertIsInstance(spec, VerificationSpec)

    def test_build_spec_mixed_mode_raises(self):
        """Test that mixing modes raises ValueError."""
        lower = torch.zeros(1, 4)
        upper = torch.ones(1, 4)
        center = torch.zeros(1, 4)
        clauses = [[(torch.tensor([[1, -1, 0, 0]]), torch.tensor([0.0]))]]
        with self.assertRaises(ValueError):
            VerificationSpec.build_spec(
                lower=lower, upper=upper, center=center, clauses=clauses
            )

    def test_build_spec_incomplete_bounds_raises(self):
        """Test that incomplete bounds raises ValueError."""
        lower = torch.zeros(1, 4)
        # Missing upper
        with self.assertRaises(ValueError):
            VerificationSpec.build_spec(lower=lower, clauses=[])

    def test_build_spec_clauses_without_bounds_raises(self):
        """Test that clauses without bounds raises ValueError."""
        clauses = [[(torch.tensor([[1, -1]]), torch.tensor([0.0]))]]
        with self.assertRaises(ValueError):
            VerificationSpec.build_spec(clauses=clauses)

    def test_to_vnnlib(self):
        """Test converting spec to vnnlib format."""
        lower = torch.zeros(2, 3)
        upper = torch.ones(2, 3)
        clauses = [
            [(torch.tensor([[1.0, -1.0, 0.0]]), torch.tensor([0.0]))],
            [(torch.tensor([[0.0, 1.0, -1.0]]), torch.tensor([0.0]))],
        ]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        vnnlib = spec.to_vnnlib()
        self.assertEqual(len(vnnlib), 2)
        self.assertEqual(len(vnnlib[0][0]), 3)  # 3 input bounds

    def test_build_from_vnnlib_preserves_empty_output_clause(self):
        """Test round-tripping vnnlib with no explicit output constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty_output.vnnlib")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "(declare-const X_0 Real)\n"
                    "(declare-const X_1 Real)\n"
                    "(declare-const Y_0 Real)\n"
                    "(assert (>= X_0 0.0))\n"
                    "(assert (<= X_0 1.0))\n"
                    "(assert (>= X_1 0.0))\n"
                    "(assert (<= X_1 1.0))\n"
                )

            spec = VerificationSpec.build_spec(vnnlib_path=path)
            vnnlib = spec.to_vnnlib()

            self.assertEqual(len(vnnlib), 1)
            self.assertEqual(len(vnnlib[0][1]), 1)
            mat, rhs = vnnlib[0][1][0]
            self.assertEqual(mat.shape, (0, 1))
            self.assertEqual(rhs.shape, (0,))

    def test_reshape_input(self):
        """Test reshaping input in spec."""
        lower = torch.zeros(1, 12)
        upper = torch.ones(1, 12)
        clauses = [[(torch.tensor([[1.0] * 12]), torch.tensor([6.0]))]]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        spec.reshape_input((3, 4))
        self.assertEqual(spec.input_shape, (-1, 3, 4))


class TestSolveResult(unittest.TestCase):
    """Tests for SolveResult dataclass."""

    def test_init_basic(self):
        """Test basic SolveResult initialization."""
        result = SolveResult(status="verified", success=True)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.success)
        self.assertEqual(result.reference, {})
        self.assertEqual(result.stats, {})

    def test_init_with_reference(self):
        """Test SolveResult with reference dict."""
        ref = {"model": "test"}
        result = SolveResult(status="unsafe-pgd", success=False, reference=ref)
        self.assertEqual(result.reference, {"model": "test"})

    def test_init_with_stats(self):
        """Test SolveResult with stats dict."""
        stats = {"elapsed": 1.5, "iterations": 100}
        result = SolveResult(status="timeout", success=False, stats=stats)
        self.assertEqual(result.stats["elapsed"], 1.5)

    def test_as_dict(self):
        """Test as_dict method."""
        result = SolveResult(
            status="verified",
            success=True,
            reference={"key": "value"},
            stats={"time": 1.0},
        )
        d = result.as_dict()
        self.assertEqual(d["status"], "verified")
        self.assertTrue(d["success"])
        self.assertEqual(d["reference"]["key"], "value")
        self.assertEqual(d["stats"]["time"], 1.0)


class TestVNNCompInstance(unittest.TestCase):
    """Tests for VNNCompInstance dataclass."""

    def test_init(self):
        """Test VNNCompInstance initialization."""
        instance = VNNCompInstance(
            index=0,
            onnx_path="/path/to/model.onnx",
            vnnlib_path="/path/to/spec.vnnlib",
            csv_row=("model.onnx", "spec.vnnlib", "120"),
        )
        self.assertEqual(instance.index, 0)
        self.assertEqual(instance.onnx_path, "/path/to/model.onnx")
        self.assertEqual(instance.vnnlib_path, "/path/to/spec.vnnlib")
        self.assertEqual(len(instance.csv_row), 3)

    def test_frozen(self):
        """Test that VNNCompInstance is frozen (immutable)."""
        instance = VNNCompInstance(
            index=0,
            onnx_path="/path/to/model.onnx",
            vnnlib_path="/path/to/spec.vnnlib",
            csv_row=("model.onnx", "spec.vnnlib"),
        )
        with self.assertRaises(Exception):  # FrozenInstanceError
            instance.index = 1


class TestABCrownSolverInit(unittest.TestCase):
    """Tests for ABCrownSolver initialization."""

    def _create_simple_model(self):
        """Create a simple linear model for testing."""
        return torch.nn.Linear(4, 2)

    def _create_simple_spec(self):
        """Create a simple verification spec for testing."""
        lower = torch.zeros(1, 4)
        upper = torch.ones(1, 4)
        clauses = [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]]
        return VerificationSpec.build_from_input_bounds(lower, upper, clauses)

    def test_init_basic(self):
        """Test basic ABCrownSolver initialization."""
        model = self._create_simple_model()
        spec = self._create_simple_spec()
        solver = ABCrownSolver(spec, model)
        self.assertIsNotNone(solver.config)
        self.assertIsNotNone(solver.spec)
        self.assertEqual(solver.name, "instance")

    def test_init_with_name(self):
        """Test ABCrownSolver initialization with custom name."""
        model = self._create_simple_model()
        spec = self._create_simple_spec()
        solver = ABCrownSolver(spec, model, name="test_instance")
        self.assertEqual(solver.name, "test_instance")

    def test_init_with_config(self):
        """Test ABCrownSolver initialization with custom config."""
        model = self._create_simple_model()
        spec = self._create_simple_spec()
        config = {"general": {"seed": 42}}
        solver = ABCrownSolver(spec, model, config=config)
        self.assertEqual(solver.config["general"]["seed"], 42)

    def test_init_with_config_builder(self):
        """Test ABCrownSolver initialization with ConfigBuilder."""
        model = self._create_simple_model()
        spec = self._create_simple_spec()
        builder = ConfigBuilder().set(general__seed=999)
        solver = ABCrownSolver(spec, model, config=builder)
        self.assertEqual(solver.config["general"]["seed"], 999)

    def test_init_none_spec_raises(self):
        """Test that None spec raises ValueError."""
        model = self._create_simple_model()
        with self.assertRaises(ValueError):
            ABCrownSolver(None, model)

    def test_init_none_model_raises(self):
        """Test that None computing_graph raises ValueError."""
        spec = self._create_simple_spec()
        with self.assertRaises(ValueError):
            ABCrownSolver(spec, None)

    def test_normalize_spec_from_dict_bounds(self):
        """Test _normalize_spec with dict containing bounds."""
        model = self._create_simple_model()
        spec_dict = {
            "lower": torch.zeros(1, 4),
            "upper": torch.ones(1, 4),
            "clauses": [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]],
        }
        solver = ABCrownSolver(spec_dict, model)
        self.assertIsInstance(solver.spec, VerificationSpec)

    def test_normalize_spec_from_dict_center(self):
        """Test _normalize_spec with dict containing center/epsilon."""
        model = self._create_simple_model()
        spec_dict = {
            "center": torch.full((1, 4), 0.5),
            "epsilon": 0.1,
            "clauses": [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]],
        }
        solver = ABCrownSolver(spec_dict, model)
        self.assertIsInstance(solver.spec, VerificationSpec)

    def test_normalize_spec_unsupported_raises(self):
        """Test that unsupported spec format raises TypeError."""
        model = self._create_simple_model()
        with self.assertRaises(TypeError):
            ABCrownSolver({"invalid": "format"}, model)


class TestABCrownSolverConfigContext(unittest.TestCase):
    """Tests for ABCrownSolver config context management."""

    def test_config_isolation(self):
        """Test that solver config is isolated during solve."""
        import arguments

        # Store original seed
        original_seed = arguments.Config["general"]["seed"]

        model = torch.nn.Linear(4, 2)
        lower = torch.zeros(1, 4)
        upper = torch.ones(1, 4)
        clauses = [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)

        solver = ABCrownSolver(spec, model, config={"general": {"seed": 12345}})

        # The global config should not be modified by creating the solver
        self.assertEqual(arguments.Config["general"]["seed"], original_seed)


class TestABCrownSolverPrepareModelDtype(unittest.TestCase):
    """Tests for dtype alignment between config and prepared models."""

    @staticmethod
    def _build_solver() -> ABCrownSolver:
        model = torch.nn.Linear(4, 2).double()
        lower = torch.zeros(1, 4)
        upper = torch.ones(1, 4)
        clauses = [[(torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        return ABCrownSolver(spec, model, config={"general": {"device": "cpu"}})

    def test_prepare_model_casts_to_float32_when_double_fp_disabled(self):
        solver = self._build_solver()
        original_flag = arguments.Config["general"]["double_fp"]
        try:
            arguments.Config["general"]["double_fp"] = False
            model = solver._prepare_model("cpu")
            self.assertEqual({param.dtype for param in model.parameters()}, {torch.float32})
        finally:
            arguments.Config["general"]["double_fp"] = original_flag

    def test_prepare_model_casts_to_float64_when_double_fp_enabled(self):
        solver = self._build_solver()
        original_flag = arguments.Config["general"]["double_fp"]
        try:
            arguments.Config["general"]["double_fp"] = True
            model = solver._prepare_model("cpu")
            self.assertEqual({param.dtype for param in model.parameters()}, {torch.float64})
        finally:
            arguments.Config["general"]["double_fp"] = original_flag


class TestABCrownSolverPrepareEnvironmentDtype(unittest.TestCase):
    """Tests for default-dtype setup in the API environment preparation."""

    def test_prepare_environment_resets_default_dtype_for_float32_runs(self):
        solver = ABCrownSolver(
            VerificationSpec.build_from_input_bounds(
                torch.zeros(1, 1),
                torch.ones(1, 1),
                [[(torch.tensor([[1.0]]), torch.tensor([0.0]))]],
            ),
            torch.nn.Linear(1, 1),
            config={"general": {"device": "cpu"}},
        )
        original_flag = arguments.Config["general"]["double_fp"]
        original_dtype = torch.get_default_dtype()
        try:
            arguments.Config["general"]["double_fp"] = True
            solver._prepare_environment("cpu")
            self.assertEqual(torch.get_default_dtype(), torch.float64)

            arguments.Config["general"]["double_fp"] = False
            solver._prepare_environment("cpu")
            self.assertEqual(torch.get_default_dtype(), torch.float32)
        finally:
            arguments.Config["general"]["double_fp"] = original_flag
            torch.set_default_dtype(original_dtype)


class TestABCrownSolverSolvingModeAttackSemantics(unittest.TestCase):
    """Regression tests for solving-mode top-level PGD behavior."""

    @staticmethod
    def _build_solver(*, pgd_order: str, solving_mode: bool) -> ABCrownSolver:
        model = torch.nn.Linear(2, 1)
        lower = torch.zeros(1, 2)
        upper = torch.ones(1, 2)
        clauses = [[(torch.tensor([[1.0]]), torch.tensor([0.0]))]]
        spec = VerificationSpec.build_from_input_bounds(lower, upper, clauses)
        config = default_config()
        config["general"]["device"] = "cpu"
        config["general"]["complete_verifier"] = "skip"
        config["attack"]["pgd_order"] = pgd_order
        config["solving"]["solving_mode"] = solving_mode
        return ABCrownSolver(spec, model, config=config)

    @staticmethod
    def _mock_handler() -> types.SimpleNamespace:
        return types.SimpleNamespace(
            x=torch.zeros(1, 2),
            data_min=torch.zeros(1, 2),
            data_max=torch.ones(1, 2),
            input_shape=(1, 2),
            vnnlib=[],
            add_rhs_offset=MagicMock(),
        )

    def test_solving_mode_skips_top_level_pgd_after(self):
        """Solving mode should not early-return from top-level PGD-after."""
        solver = self._build_solver(pgd_order="after", solving_mode=True)
        with (
            patch.object(solver, "_prepare_environment"),
            patch.object(solver, "_prepare_model", return_value=solver.computing_graph),
            patch.object(solver, "_build_vnnlib_handler", return_value=self._mock_handler()),
            patch("api.incomplete_verifier_core", return_value=("unknown", {})),
            patch.object(solver, "_attack", return_value=("unsafe-pgd", True, None, None, None)) as mock_attack,
        ):
            result = solver.solve(return_reference=False)

        mock_attack.assert_not_called()
        self.assertEqual(result.status, "unknown")

    def test_non_solving_mode_runs_top_level_pgd_after(self):
        """Legacy PGD-after behavior should still run outside solving mode."""
        solver = self._build_solver(pgd_order="after", solving_mode=False)
        with (
            patch.object(solver, "_prepare_environment"),
            patch.object(solver, "_prepare_model", return_value=solver.computing_graph),
            patch.object(solver, "_build_vnnlib_handler", return_value=self._mock_handler()),
            patch("api.incomplete_verifier_core", return_value=("unknown", {})),
            patch.object(solver, "_attack", return_value=("unsafe-pgd", True, None, None, None)) as mock_attack,
        ):
            result = solver.solve(return_reference=False)

        mock_attack.assert_called_once()
        # Newer status normalization may map PGD-found counterexamples to
        # "falsified" instead of the legacy "unsafe-pgd" label.
        self.assertIn(result.status, ["unsafe-pgd", "falsified"])


if __name__ == "__main__":
    unittest.main()
