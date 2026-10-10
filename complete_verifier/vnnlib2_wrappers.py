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
"""PyTorch wrappers used for lowered VNN-LIB 2.0 queries."""

from __future__ import annotations

import torch
import vnnlib
from torch import nn


class VNNLib2Wrapper(nn.Module):
    """Expose a VNN-LIB 2.0 query through the verifier model interface.

    The verifier expects one PyTorch model with one flat input tensor and one
    flat output tensor, plus linear specs of the form ``mat @ output <= rhs``.
    VNN-LIB 2.0 can declare multiple networks, multiple named inputs/outputs,
    and constraints that mention input variables or nonlinear expressions. This
    wrapper bridges that gap:

    1. split the verifier's flat input into declared VNN-LIB inputs;
    2. call each declared network with the input tensors it owns, excluding
       the fixed leading batch dimension from the declared VNN-LIB shapes;
    3. concatenate the declared network outputs;
    4. optionally append the original inputs, so linear input constraints can be
       encoded as ordinary linear rows over the wrapper output;
    5. append synthetic nonlinear outputs, where each synthetic column is
       ``left_expr - right_expr`` for one nonlinear VNN-LIB comparison.

    After this wrapping, verification specs use ``mat @ wrapped_output <= rhs``.
    """

    def __init__(self, models, metadata):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.metadata = metadata

    def forward(self, z):
        # The verifier gives us one flattened input. Recover the named VNN-LIB
        # input tensors. VNN-LIB 2 declarations include a fixed leading batch
        # dimension of 1; the verifier runtime batch is z.size(0), so model
        # inputs use the remaining per-example dimensions.
        z = z.flatten(1)
        ordered_inputs = sorted(self.metadata.inputs.values(), key=lambda decl: decl.offset)
        input_chunks = {}
        chunks_by_indices = {}
        for decl in ordered_inputs:
            # A declared input keeps its VNN-LIB offset, but shared inputs may
            # map multiple declared positions to the same verifier input column.
            # `indices` is the verifier-column layout for this declared tensor.
            indices = tuple(
                self.metadata.input_index_map[decl.offset + index]
                for index in range(decl.size)
            )
            chunk = chunks_by_indices.get(indices)
            if chunk is None:
                # Reuse a slice when the declared tensor is backed by a
                # contiguous verifier input range. This is the common case for
                # shared inputs such as X_f[i] == X_g[i].
                if indices == tuple(range(indices[0], indices[0] + len(indices))):
                    chunk = z[:, indices[0]:indices[0] + len(indices)]
                else:
                    # Non-contiguous layouts still represent a fixed linear
                    # gather from the verifier input for this query.
                    index = torch.tensor(indices, device=z.device)
                    chunk = torch.index_select(z, 1, index)
                chunks_by_indices[indices] = chunk
            input_chunks[decl.name] = chunk

        outputs = []
        output_chunks = {}
        for network in self.metadata.networks:
            # A VNN-LIB 2.0 query may wire different declared inputs into
            # different networks. Build the argument list for this network in
            # declaration order.
            inputs = []
            for decl in network.inputs:
                assert decl.shape[0] == 1, (
                    f"VNN-LIB 2 input {decl.name} must have batch dimension 1, "
                    f"but got shape {decl.shape}")
                inputs.append(
                    input_chunks[decl.name].reshape(z.size(0), *decl.shape[1:]))

            if len(network.outputs) != 1:
                raise RuntimeError(
                    f"Cannot map one model output to "
                    f"{len(network.outputs)} VNN-LIB outputs for {network.name}")

            model = self.models[network.name]
            out = model(*inputs)

            # Keep a name -> tensor map for later expression evaluation, while
            # also building the flat output vector used by verifier specs.
            flat_output = out.flatten(1)
            output_chunks[network.outputs[0].name] = flat_output
            outputs.append(flat_output)

        joint_output = torch.cat(outputs, dim=1)
        if self.metadata.include_input_passthrough:
            # Linear constraints involving inputs, such as X[0] - X[1] <= 0,
            # are represented as rows over these appended input columns.
            joint_output = torch.cat([joint_output, z], dim=1)
        nonlinear_outputs = [
            # Nonlinear comparisons are lowered to synthetic outputs. A row
            # selecting this column with rhs=0 represents left_expr - right_expr
            # <= 0, and PyTorch/autograd/bound propagation see the expression as
            # part of this wrapper graph.
            self._eval_expr(left, z, input_chunks, output_chunks)
            - self._eval_expr(right, z, input_chunks, output_chunks)
            for left, right in self.metadata.nonlinear_outputs
        ]
        if nonlinear_outputs:
            joint_output = torch.cat([joint_output, *nonlinear_outputs], dim=1)
        return joint_output

    def _eval_expr(self, expr, z, input_chunks, output_chunks):
        """Evaluate one VNNLIB-Python arithmetic AST for the current batch.

        ``expr`` is one side of a nonlinear VNN-LIB comparison, such as
        ``(* X[0,0] X[0,1])`` or ``Y[0,0]``. ``z`` is the verifier's flat input
        tensor. ``input_chunks`` and ``output_chunks`` map declared VNN-LIB
        tensor names to their batched PyTorch values.

        Returns a tensor of shape ``(batch, 1)`` containing the expression value
        for each example in the batch.
        """
        if isinstance(expr, vnnlib.Var):
            return self._eval_var(expr, input_chunks, output_chunks)

        if isinstance(expr, (vnnlib.Float, vnnlib.Int)):
            return z.new_full((z.size(0), 1), float(expr.value))

        if isinstance(expr, vnnlib.Plus):
            result = z.new_zeros((z.size(0), 1))
            for child in expr.args:
                result = result + self._eval_expr(child, z, input_chunks, output_chunks)
            return result
        if isinstance(expr, vnnlib.Multiply):
            result = z.new_ones((z.size(0), 1))
            for child in expr.args:
                result = result * self._eval_expr(child, z, input_chunks, output_chunks)
            return result
        if isinstance(expr, vnnlib.Minus):
            result = self._eval_expr(expr.head, z, input_chunks, output_chunks)
            for child in expr.rest:
                result = result - self._eval_expr(child, z, input_chunks, output_chunks)
            return result
        if isinstance(expr, vnnlib.Negate):
            return -self._eval_expr(expr.expr, z, input_chunks, output_chunks)

        raise RuntimeError(
            f"Unsupported VNN-LIB nonlinear expression node: {type(expr).__name__}")

    def _eval_var(self, expr, input_chunks, output_chunks):
        """Return one flattened declared input/output column for a VNN-LIB var."""
        if expr.name in self.metadata.inputs:
            shape = self.metadata.inputs[expr.name].shape
        else:
            shape = self.metadata.outputs[expr.name].shape

        index = 0
        for idx, dim in zip(expr.indices, shape):
            index = index * dim + int(idx)

        values = input_chunks.get(expr.name)
        if values is None:
            values = output_chunks[expr.name]
        return values[:, index:index + 1]
