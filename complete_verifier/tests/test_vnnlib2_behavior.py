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
"""Behavior tests for lowered VNN-LIB 2.0 queries and wrapper outputs."""

import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from read_vnnlib2 import read_vnnlib2
from vnnlib2_wrappers import VNNLib2Wrapper


_TOL = 1e-6


def _write(tmp_path, text):
    path = tmp_path / "case.vnnlib"
    path.write_text(text)
    return str(path)


def _tokenize(expr):
    return expr.replace("(", " ( ").replace(")", " ) ").split()


def _parse_tokens(tokens):
    token = tokens.pop(0)
    if token == "(":
        out = []
        while tokens[0] != ")":
            out.append(_parse_tokens(tokens))
        tokens.pop(0)
        return out
    return token


def _statements(text):
    open_parentheses = 0
    current = ""
    out = []
    for raw_line in text.splitlines():
        line = raw_line.split(";", maxsplit=1)[0].strip()
        if not line:
            continue
        open_parentheses += line.count("(") - line.count(")")
        current += " " if current else ""
        current += line
        if open_parentheses == 0:
            out.append(" ".join(current.split()))
            current = ""
    return out


def _assertion_expr(statement):
    parsed = _parse_tokens(_tokenize(statement))
    assert parsed[0] == "assert"
    return parsed[1]


def _value(expr, env):
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        return float(expr)
    op = expr[0]
    if op == "+":
        return sum(_value(child, env) for child in expr[1:])
    if op == "-":
        if len(expr) == 2:
            return -_value(expr[1], env)
        return _value(expr[1], env) - sum(
            _value(child, env) for child in expr[2:])
    if op == "*":
        result = 1.0
        for child in expr[1:]:
            result *= _value(child, env)
        return result
    raise AssertionError(f"unsupported operator in semantic test: {op}")


def _semantics(expr, env):
    op = expr[0]
    if op == "and":
        return all(_semantics(child, env) for child in expr[1:])
    if op == "or":
        return any(_semantics(child, env) for child in expr[1:])
    if op == "<=":
        return _value(expr[1], env) <= _value(expr[2], env) + _TOL
    if op == "<":
        return _value(expr[1], env) < _value(expr[2], env) + _TOL
    if op == ">=":
        return _value(expr[1], env) + _TOL >= _value(expr[2], env)
    if op == ">":
        return _value(expr[1], env) + _TOL > _value(expr[2], env)
    if op == "==":
        return abs(_value(expr[1], env) - _value(expr[2], env)) <= _TOL
    if op == "!=":
        return abs(_value(expr[1], env) - _value(expr[2], env)) > _TOL
    raise AssertionError(f"unsupported boolean operator in semantic test: {op}")


def _lowered_holds(lowered_vnnlib, wrapper_output):
    y = wrapper_output.detach().numpy().reshape(-1)
    for _, specs in lowered_vnnlib:
        if any(np.all(mat @ y <= rhs + _TOL) for mat, rhs in specs):
            return True
    return False


def test_wrapper_excludes_declared_batch_dim_when_calling_model(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1, 3, 4])
  (declare-output Y Real [1, 1])
)
(assert (<= Y[0,0] 10.0))
""")
    _, metadata = read_vnnlib2(path)

    class ShapeCheckingModel(nn.Module):
        def forward(self, x):
            assert tuple(x.shape) == (2, 3, 4)
            return x.flatten(1).sum(dim=1, keepdim=True)

    wrapper = VNNLib2Wrapper({"f": ShapeCheckingModel()}, metadata)
    out = wrapper(torch.arange(24.0).reshape(2, 12))

    torch.testing.assert_close(
        out,
        torch.tensor([[66.0], [210.0]]))


def test_wrapper_evaluates_nonlinear_query_outputs(tmp_path):
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
    _, metadata = read_vnnlib2(path)

    class SumModel(nn.Module):
        def forward(self, x):
            return x.flatten(1).sum(dim=1, keepdim=True).reshape(x.size(0), 1, 1)

    wrapper = VNNLib2Wrapper({"f": SumModel()}, metadata)
    out = wrapper(torch.tensor([[2.0, 3.0]]))

    # Original model output is 5. The added query output is X0 * X1 - Y = 1.
    torch.testing.assert_close(out, torch.tensor([[5.0, 1.0]]))


def test_lowered_specs_match_direct_semantics_on_samples(tmp_path):
    text = """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X Real [1, 2])
  (declare-output Y Real [1, 1])
)
(assert (and (>= X[0,0] -2.0) (<= X[0,0] 2.0)))
(assert (and (>= X[0,1] -1.0) (<= X[0,1] 3.0)))
(assert (or (<= (* X[0,0] X[0,1]) Y[0,0])
            (>= (+ X[0,0] X[0,1]) Y[0,0])))
(assert (<= (- Y[0,0] X[0,0]) 3.0))
"""
    path = _write(tmp_path, text)
    lowered_vnnlib, metadata = read_vnnlib2(path)
    assertions = [
        _assertion_expr(statement)
        for statement in _statements(text)
        if statement.startswith("(assert ")
    ]

    class SampleModel(nn.Module):
        def forward(self, x):
            flat = x.flatten(1)
            y = 0.5 * flat[:, 0:1] - flat[:, 1:2] + 1.0
            return y.reshape(x.size(0), 1, 1)

    wrapper = VNNLib2Wrapper({"f": SampleModel()}, metadata)
    samples = torch.tensor([
        [-2.0, -1.0],
        [-1.0, 0.5],
        [0.0, 0.0],
        [1.5, 2.0],
        [2.0, 3.0],
    ])

    for sample in samples:
        output = wrapper(sample.view(1, -1))
        y0 = output[0, 0].item()
        env = {
            "X[0,0]": sample[0].item(),
            "X[0,1]": sample[1].item(),
            "Y[0,0]": y0,
        }
        direct_result = all(_semantics(expr, env) for expr in assertions)
        lowered_result = _lowered_holds(lowered_vnnlib, output)
        assert lowered_result == direct_result


def test_wrapper_feeds_shared_input_to_both_models(tmp_path):
    path = _write(tmp_path, """
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X_f Real [1, 2])
  (declare-output Y_f Real [1, 1])
)
(declare-network g
  (isomorphic-to f)
  (declare-input X_g Real [1, 2])
  (declare-output Y_g Real [1, 1])
)
(assert (and (>= X_f[0,0] 0.0) (<= X_f[0,0] 1.0)))
(assert (and (>= X_f[0,1] 2.0) (<= X_f[0,1] 3.0)))
(assert (== X_f[0,0] X_g[0,0]))
(assert (== X_f[0,1] X_g[0,1]))
(assert (<= Y_f[0,0] Y_g[0,0]))
""")
    lowered_vnnlib, metadata = read_vnnlib2(path)

    class ScaleModel(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            out = self.scale * x.flatten(1).sum(dim=1, keepdim=True)
            return out.reshape(x.size(0), 1, 1)

    wrapper = VNNLib2Wrapper(
        {"f": ScaleModel(1.0), "g": ScaleModel(2.0)}, metadata)

    assert len(lowered_vnnlib[0][0]) == 2
    torch.testing.assert_close(
        wrapper(torch.tensor([[1.0, 3.0]])),
        torch.tensor([[4.0, 8.0]]))
