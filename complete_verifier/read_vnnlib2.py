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
"""Lower VNN-LIB 2.0 queries to the verifier specification format.

The VNN-LIB language work is delegated to VNNLIB-Python: parsing, type
checking, DNF conversion, and arithmetic linearisation.  This module keeps the
verifier-specific lowering: flattening declared tensors, extracting one flat
input box, and encoding the remaining constraints as ``row @ output <= rhs``
clauses over wrapper outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import vnnlib


@dataclass(frozen=True)
class TensorDecl:
    """One declared tensor's shape and offset in a flattened verifier vector."""

    name: str
    shape: Tuple[int, ...]
    # Starting index in the global input vector or normal-output vector.
    offset: int

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass
class NetworkDecl:
    """Verifier-facing network declaration derived from the parsed query."""

    name: str
    # (relation kind, referenced network name), for model sharing in loading.py.
    relation: Optional[Tuple[str, str]] = None
    inputs: List[TensorDecl] = field(default_factory=list)
    outputs: List[TensorDecl] = field(default_factory=list)


@dataclass
class VNNLib2Metadata:
    """Tensor layout and network information needed after AST parsing.

    ``inputs`` and ``outputs`` provide name-based lookup for expression
    lowering. Their offsets address two separate flattened spaces: all input
    tensors concatenated in declaration order, and all ordinary network output
    tensors concatenated in declaration order.

    Index terminology:

    * Declared input index: row-major position from the VNN-LIB tensor
      declarations. These indices remain stable while constraints are lowered.
    * Verifier input index: column in the flat tensor passed to
      ``VNNLib2Wrapper.forward``. Shared-input compression can map multiple
      declared input indices to one verifier input index.
    """

    networks: List[NetworkDecl]
    inputs: Dict[str, TensorDecl]
    outputs: Dict[str, TensorDecl]
    # Size of the ordinary network-output prefix, excluding synthetic outputs.
    normal_output_size: int
    total_input_size: int
    # Map each declared flattened input index to the verifier input index.
    input_index_map: List[int] = field(default_factory=list)
    # VNNLIB-Python AST operand pairs evaluated by the wrapper as left - right.
    nonlinear_outputs: List["NonlinearOutput"] = field(default_factory=list)
    # Whether wrapper outputs must also include the flattened input vector.
    include_input_passthrough: bool = False
    # Strict/open-bound metadata aligned with the lowered vnnlib entries.
    strictness: Optional[List["VNNLib2BranchStrictness"]] = None
    # Resolved ONNX paths by declared network name, filled by the loader.
    network_paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input_index_map:
            # Without shared inputs, declared and verifier input indices match.
            self.input_index_map = list(range(self.total_input_size))

    @property
    def total_output_size(self) -> int:
        total = self.normal_output_size + len(self.nonlinear_outputs)
        if self.include_input_passthrough:
            total += self.total_input_size
        return total

    @property
    def input_passthrough_offset(self) -> int:
        return self.normal_output_size

    @property
    def nonlinear_output_offset(self) -> int:
        offset = self.normal_output_size
        if self.include_input_passthrough:
            offset += self.total_input_size
        return offset


@dataclass
class VNNLib2BranchStrictness:
    """Strict/open-bound metadata for one lowered branch.

    ``input_lower[i]`` marks whether ``box[i][0]`` is open. ``input_upper[i]``
    marks whether ``box[i][1]`` is open. ``specs[j][k]`` marks whether row
    ``k`` of lowered output spec ``j`` is strict.
    """

    input_lower: np.ndarray
    input_upper: np.ndarray
    specs: List[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class Var:
    """A tensor element addressed in a global flattened variable space."""

    name: str
    index: int


@dataclass
class LinearExpr:
    """Sparse affine expression ``sum(terms[var] * var) + const``."""

    terms: Dict[Var, float] = field(default_factory=dict)
    const: float = 0.0

    def copy(self) -> "LinearExpr":
        return LinearExpr(dict(self.terms), self.const)

    def add(self, other: "LinearExpr", scale: float = 1.0) -> "LinearExpr":
        out = self.copy()
        out.const += scale * other.const
        for var, coeff in other.terms.items():
            new_coeff = out.terms.get(var, 0.0) + scale * coeff
            if abs(new_coeff) < 1e-15:
                out.terms.pop(var, None)
            else:
                out.terms[var] = new_coeff
        return out

    def scale(self, value: float) -> "LinearExpr":
        return LinearExpr(
            {var: coeff * value for var, coeff in self.terms.items()},
            self.const * value,
        )


@dataclass
class LinearConstraint:
    """A normalized affine inequality ``expr <= 0``."""

    expr: LinearExpr
    strict: bool = False


NonlinearOutput = Tuple[object, object]


@dataclass(frozen=True)
class NonlinearConstraint:
    """Constraint requiring one synthetic nonlinear output to be nonpositive.

    ``output_index`` refers to ``metadata.nonlinear_outputs[output_index]``,
    a pair of VNNLIB-Python AST operands evaluated by the wrapper as
    ``left - right``. The expression cannot be encoded directly as constant
    matrix coefficients, so the wrapper appends it as an extra output column.
    """

    # Index into metadata.nonlinear_outputs, not the full wrapper output vector.
    output_index: int
    strict: bool = False


Constraint = LinearConstraint | NonlinearConstraint
# A conjunction is one verifier branch: every constraint in it must hold.
ConstraintConjunction = List[Constraint]
# A disjunction contains alternative branches: at least one must hold.
ConstraintDisjunction = List[ConstraintConjunction]


@dataclass
class BranchState:
    """Partially lowered state for one disjunctive property branch.

    ``box`` contains input bounds extracted so far. ``lower_strict`` and
    ``upper_strict`` preserve whether each final input bound came from a strict
    VNN-LIB comparison. ``deferred`` contains constraints that must later become
    specification rows or be used to infer additional bounds from input-input
    relations.
    """

    box: List[List[float]]
    lower_strict: np.ndarray
    upper_strict: np.ndarray
    deferred: List[Constraint] = field(default_factory=list)

    def copy(self) -> "BranchState":
        """Return an independent state for expanding another OR branch."""
        return BranchState(
            [bounds[:] for bounds in self.box],
            self.lower_strict.copy(),
            self.upper_strict.copy(),
            list(self.deferred),
        )


_VNNLIB_VERSION_RE = re.compile(
    r"^\(\s*vnnlib-version\s+<\s*(\d+)\.(\d+)\s*>\s*\)$"
)
_VNNLIB_VERSION_PREFIX_RE = re.compile(r"^\(\s*vnnlib-version\b")


def is_vnnlib2_file(path: str) -> bool:
    """Return whether a file's leading declaration specifies VNN-LIB 2.x."""
    declaration = ""
    open_parentheses = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            comment_index = line.find(";")
            if comment_index != -1:
                line = line[:comment_index].rstrip()
            if not line:
                continue

            if not declaration and _VNNLIB_VERSION_PREFIX_RE.match(line) is None:
                # A missing version declaration means VNN-LIB 1.0.
                return False

            open_parentheses += line.count("(") - line.count(")")
            assert open_parentheses >= 0, "mismatched parenthesis in vnnlib file"
            declaration += " " if declaration else ""
            declaration += line
            if open_parentheses == 0:
                break

    if not declaration:
        return False

    match = _VNNLIB_VERSION_RE.fullmatch(declaration)
    if match is not None:
        return int(match.group(1)) == 2

    raise ValueError(f"Malformed VNN-LIB version declaration: {declaration}")


def _read_cached_vnnlib2(
        path: str, source_hash: str) -> Optional[Tuple[list, VNNLib2Metadata]]:
    """Load a valid parser cache or return ``None``.

    The ``.compiled`` file is only a speed optimization. Missing, stale, or
    malformed caches fall back to parsing the source VNN-LIB file.
    """
    cache_path = path + ".compiled"
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as cache_file:
            payload = pickle.load(cache_file)
    except (
            OSError,
            EOFError,
            AttributeError,
            ImportError,
            pickle.PickleError,
            TypeError,
            ValueError,
    ):
        return None

    try:
        lowered_vnnlib, metadata, cached_hash = payload
    except (TypeError, ValueError):
        return None

    if cached_hash != source_hash:
        return None
    return lowered_vnnlib, metadata


def _write_cached_vnnlib2(
        path: str, lowered_vnnlib: list, metadata: VNNLib2Metadata,
        source_hash: str) -> None:
    """Write a sidecar parser cache, ignoring failures after successful parse."""
    payload = (lowered_vnnlib, metadata, source_hash)
    try:
        payload_bytes = pickle.dumps(
            payload, protocol=pickle.HIGHEST_PROTOCOL)
    except (
            AttributeError,
            pickle.PickleError,
            TypeError,
            ValueError,
    ) as exc:
        # Do not raise here: nonlinear VNN-LIB2 specs may keep VNNLIB-Python
        # AST objects in metadata.nonlinear_outputs, and those objects are not
        # always pickleable. The parsed result is still valid, so these files
        # should continue without a sidecar cache instead of failing verifier
        # loading.
        print(
            "Warning: skipping VNN-LIB2 parser cache for "
            f"{path}: parsed metadata is not pickleable "
            f"({type(exc).__name__}: {exc})")
        return

    try:
        with open(path + ".compiled", "wb") as cache_file:
            cache_file.write(payload_bytes)
    except OSError:
        # Cache write failures should not affect verifier behavior.
        return


def read_vnnlib2(path: str) -> Tuple[list, VNNLib2Metadata]:
    """Parse and lower a VNN-LIB 2.0 file using VNNLIB-Python's typed AST."""
    with open(path, "rb") as query_file:
        source_hash = hashlib.sha256(query_file.read()).hexdigest()
    cached = _read_cached_vnnlib2(path, source_hash)
    if cached is not None:
        return cached

    # VNNLIB-Python owns parsing, type checking, and AST construction.
    query = vnnlib.parse_query_file(path)

    # Flatten declared input/output tensors into ABCROWN vector offsets.
    metadata = _metadata_from_query(query)
    branches = [_new_branch_state(metadata.total_input_size)]

    # All top-level assertions must hold, so assertions are combined with AND.
    # Each assertion is independently lowered to DNF:
    #
    #   branch_constraints = [
    #       [constraint, ...],  # one conjunction
    #       [constraint, ...],  # an alternative conjunction
    #   ]
    #
    # The outer list is OR and each inner list is AND. ``branches`` contains
    # the DNF accumulated from assertions processed so far.
    for assertion in query.assertions:
        # Large generated specs may contain many top-level assertions that are
        # just one input bound each. Avoid DNF conversion and BranchState.copy()
        # for this common non-branching case.
        if len(branches) == 1 and _try_apply_simple_box_assertion(
                branches[0], assertion, metadata):
            continue

        branch_constraints = _assertion_to_dnf(assertion, metadata)
        if len(branch_constraints) == 1:
            for branch in branches:
                _apply_constraints_to_branch(
                    branch, branch_constraints[0], metadata)
            continue

        updated_branches = []

        # Compute the Cartesian product between existing branches and the
        # current assertion's alternatives. For example,
        # ``(A or B) and (C or D)`` becomes AC, AD, BC, and BD.
        for branch in branches:
            for constraints in branch_constraints:
                # Each combination needs an independent box and deferred list.
                branch_copy = branch.copy()
                _apply_constraints_to_branch(
                    branch_copy, constraints, metadata)
                updated_branches.append(branch_copy)

        # The accumulated DNF now includes this assertion.
        branches = updated_branches

    # Sharing inputs changes the verifier input domain for every branch, so
    # only equalities required by all DNF branches can be compressed.
    _compress_equivalent_inputs(branches, metadata)

    # Mixed input/output constraints need the wrapper to append the flat input
    # vector to its outputs so they can be represented as spec rows.
    # All single-variable input bounds have already been consumed into boxes,
    # so any deferred linear constraint containing an input needs passthrough.
    metadata.include_input_passthrough = any(
        isinstance(constraint, LinearConstraint)
        and any(
            _is_input_var(var, metadata)
            for var in constraint.expr.terms
        )
        for branch in branches
        for constraint in branch.deferred
    )

    rv = []
    for branch in branches:
        box = branch.box
        deferred = branch.deferred
        # Deferred input-input relations such as ``X_f == X_g`` can imply
        # missing or tighter box bounds before empty-branch pruning.
        _infer_input_bounds_from_relations(box, deferred, metadata)

        # Drop infeasible DNF branches; remaining OR alternatives can still be
        # valid verifier specs.
        if any(bounds[0] > bounds[1] for bounds in box):
            continue

        # Deferred constraints are emitted as rows over normal model outputs,
        # optional input passthrough outputs, and optional nonlinear query outputs.
        # spec_strict[row_idx] stores whether that row should be checked as an
        # open inequality during counterexample validation.
        mat = []
        rhs = []
        spec_strict = np.zeros(len(deferred), dtype=bool)
        for row_idx, constraint in enumerate(deferred):
            row, row_rhs = _constraint_to_output_row(constraint, metadata)
            mat.append(row)
            rhs.append(row_rhs)
            spec_strict[row_idx] = constraint.strict

        # Input-only branches are represented with zero output-constraint rows.
        if mat:
            mat = np.array(mat, dtype=float)
            rhs = np.array(rhs, dtype=float)
        else:
            mat = np.zeros((0, metadata.total_output_size), dtype=float)
            rhs = np.zeros((0,), dtype=float)

        if _spec_has_static_contradiction(mat, rhs, spec_strict, metadata):
            continue

        rv.append((
            box,
            [(mat, rhs)],
            VNNLib2BranchStrictness(
                branch.lower_strict.copy(),
                branch.upper_strict.copy(),
                [spec_strict],
            ),
        ))

    if not rv:
        raise NotImplementedError(
            "All VNN-LIB 2.0 disjunctive branches are infeasible.")

    vnnlib_entries, strictness = _merge_same_input_boxes(rv)
    metadata.strictness = strictness
    _write_cached_vnnlib2(path, vnnlib_entries, metadata, source_hash)
    return vnnlib_entries, metadata


def _new_branch_state(total_input_size: int) -> BranchState:
    """Create an empty branch over the current flat verifier input space."""
    return BranchState(
        [[-np.inf, np.inf] for _ in range(total_input_size)],
        # lower_strict[i] says whether box[i][0] is an open lower bound.
        np.zeros(total_input_size, dtype=bool),
        # upper_strict[i] says whether box[i][1] is an open upper bound.
        np.zeros(total_input_size, dtype=bool),
    )


def _apply_constraints_to_branch(
        branch: BranchState, constraints: ConstraintConjunction,
        metadata: VNNLib2Metadata) -> None:
    """Apply one conjunction to a branch, mutating it in place."""
    for constraint in constraints:
        # Simple one-variable input bounds are represented directly in the box.
        # Output, relational, and nonlinear constraints are kept for later
        # propagation or conversion to C rows.
        if not _try_apply_input_bound(branch, constraint, metadata):
            branch.deferred.append(constraint)


def _metadata_from_query(query) -> VNNLib2Metadata:
    """Convert a VNNLIB-Python ``Query`` into verifier tensor metadata.

    ``query`` is the typed AST returned by ``vnnlib.parse_query_file``. Its
    ``networks`` contain names, network relations, and typed input/output
    definitions; its ``assertions`` contain the property expressions and are
    handled separately by ``_assertion_to_dnf``.

    The returned metadata discards no tensor layout information needed by the
    verifier, but deliberately does not copy assertions. It assigns each input
    and output tensor a row-major offset in a global flat vector so boxes,
    specification matrices, and model wrappers agree on one ordering.
    """
    networks: List[NetworkDecl] = []
    # These maps resolve AST variable names to shapes and flattened offsets.
    inputs: Dict[str, TensorDecl] = {}
    outputs: Dict[str, TensorDecl] = {}
    input_offset = 0
    output_offset = 0

    for query_network in query.networks:
        network = NetworkDecl(name=query_network.name)

        # Network relations let loading.py reuse another declaration's model.
        # The VNN-LIB keyword and parser node are ``isomorphic-to`` and
        # IsomorphicTo, but VNNLIB-CPP stores that same relation in the typed
        # AST field ``isometricTo`` (exposed by Python as ``isometric_to``).
        # Accept both API spellings while normalizing to the VNN-LIB keyword.
        equal_to = getattr(query_network, "equal_to", None)
        if equal_to:
            network.relation = ("equal-to", equal_to)
        else:
            isomorphic_to = (
                getattr(query_network, "isomorphic_to", None)
                or getattr(query_network, "isometric_to", None)
            )
            if isomorphic_to:
                network.relation = ("isomorphic-to", isomorphic_to)

        # Offsets are global across networks. For example, tensors with sizes
        # 2 and 3 occupy [0, 2) and [2, 5) in the flattened input vector.
        for item in query_network.inputs:
            decl = TensorDecl(
                item.name, tuple(int(x) for x in item.shape), input_offset)
            input_offset += decl.size
            network.inputs.append(decl)
            inputs[decl.name] = decl

        # Outputs use an independent flat space because they become columns in
        # the specification matrix before passthrough and nonlinear outputs.
        for item in query_network.outputs:
            decl = TensorDecl(
                item.name, tuple(int(x) for x in item.shape), output_offset)
            output_offset += decl.size
            network.outputs.append(decl)
            outputs[decl.name] = decl

        networks.append(network)

    if not networks:
        raise NotImplementedError("No declare-network block found.")
    return VNNLib2Metadata(networks, inputs, outputs, output_offset, input_offset)


def _assertion_to_dnf(
        assertion, metadata: VNNLib2Metadata) -> ConstraintDisjunction:
    """Lower one assertion to OR branches of ANDed verifier constraints.

    VNNLIB-Python returns DNF as ``list[list[Comparison]]``:

    * the outer list is a disjunction of clauses;
    * each inner list is a conjunction of comparisons.

    For example, ``(or (and A B) C)`` becomes ``[[A, B], [C]]``.

    Lowering one comparison may introduce another disjunction. In particular,
    ``A != B`` has two options: ``A < B`` or ``A > B``. Each option is itself
    a list because one comparison can also require multiple ANDed constraints;
    equality, for example, lowers to both ``A - B <= 0`` and ``B - A <= 0``.

    The result has the same outer-OR/inner-AND shape, but contains verifier
    ``Constraint`` objects instead of VNNLIB-Python comparison nodes.
    """
    dnf = assertion.expr.to_dnf()

    disjunction: ConstraintDisjunction = []
    # Each clause is one OR branch whose comparisons must all hold.
    for clause in dnf:
        clause_branches: ConstraintDisjunction = [[]]
        for comparison in clause:
            comparison_disjunction = _comparison_to_clause_options(
                comparison, metadata)
            # Take the Cartesian product with choices introduced by earlier
            # comparisons. For ``D and (A != B)``, this produces the branches
            # ``[D, A < B]`` and ``[D, A > B]``.
            clause_branches = [
                prefix + conjunction
                for prefix in clause_branches
                for conjunction in comparison_disjunction
            ]
        disjunction.extend(clause_branches)
    return disjunction


def _comparison_to_clause_options(
        comparison, metadata: VNNLib2Metadata) -> ConstraintDisjunction:
    """Lower one comparison into OR options of ANDed constraints.

    Most comparisons have one option. ``NotEqual`` has two: ``left < right``
    or ``right < left``. Greater-than comparisons are normalized by swapping
    operands, so downstream code only handles ``left <= right``. Equality is
    represented by applying that normalized inequality in both directions.
    """
    if isinstance(comparison, vnnlib.NotEqual):
        return [
            _comparison_parts_to_constraints(
                comparison.lhs, comparison.rhs, metadata, strict=True),
            _comparison_parts_to_constraints(
                comparison.rhs, comparison.lhs, metadata, strict=True),
        ]

    if isinstance(comparison, (vnnlib.LessEqual, vnnlib.LessThan)):
        left, right = comparison.lhs, comparison.rhs
        equality = False
        strict = isinstance(comparison, vnnlib.LessThan)
    elif isinstance(comparison, (vnnlib.GreaterEqual, vnnlib.GreaterThan)):
        # Normalize greater-than comparisons by swapping their operands.
        left, right = comparison.rhs, comparison.lhs
        equality = False
        strict = isinstance(comparison, vnnlib.GreaterThan)
    elif isinstance(comparison, vnnlib.Equal):
        left, right = comparison.lhs, comparison.rhs
        equality = True
        strict = False
    else:
        raise NotImplementedError(
            f"Unsupported comparison operator: {type(comparison).__name__}")

    return [_comparison_parts_to_constraints(
        left, right, metadata, equality=equality, strict=strict)]


def _comparison_parts_to_constraints(
        left, right, metadata: VNNLib2Metadata,
        equality: bool = False,
        strict: bool = False) -> ConstraintConjunction:
    """Lower normalized ``left <= right`` into verifier constraints.

    Linear AST expressions become sparse ``LinearExpr`` objects and then
    ``left - right <= 0``. If either side is nonlinear, the expression is
    registered as a synthetic wrapper output instead. Equality requests both
    directions and therefore returns two ANDed constraints.
    """
    try:
        left_expr = _linear_expr_from_ast(left, metadata)
        right_expr = _linear_expr_from_ast(right, metadata)
    except ValueError:
        # Nonlinear comparisons are delegated to the wrapper by adding a
        # synthetic output whose value is constrained to be <= 0.
        return _nonlinear_comparison_to_constraints(
            left, right, metadata, equality, strict=strict)

    # Every comparison is normalized to left - right <= 0.
    diff = left_expr.add(right_expr, scale=-1.0)
    constraints = [LinearConstraint(diff, strict=strict)]
    if equality:
        # Equality also requires right - left <= 0.
        constraints.append(LinearConstraint(diff.scale(-1.0), strict=False))
    return constraints


def _nonlinear_comparison_to_constraints(
        left, right, metadata: VNNLib2Metadata,
        equality: bool,
        strict: bool = False) -> ConstraintConjunction:
    """Represent a nonlinear comparison using synthetic wrapper outputs.

    Each registered expression is later evaluated by ``VNNLib2Wrapper``.
    The returned constraint requires ``left - right <= 0``. Equality registers
    the reverse expression as a second, simultaneously required output.
    """
    constraints = [
        _make_nonlinear_constraint(left, right, metadata, strict=strict)
    ]
    if equality:
        constraints.append(
            _make_nonlinear_constraint(right, left, metadata, strict=False))
    return constraints


def _make_nonlinear_constraint(
        left, right,
        metadata: VNNLib2Metadata,
        strict: bool = False) -> NonlinearConstraint:
    """Register a synthetic expression and return a constraint referencing it."""
    metadata.nonlinear_outputs.append((left, right))
    return NonlinearConstraint(len(metadata.nonlinear_outputs) - 1, strict=strict)


def _linear_expr_from_ast(expr, metadata: VNNLib2Metadata) -> LinearExpr:
    """Convert a VNNLIB-Python arithmetic AST into a sparse affine expression.

    VNNLIB-Python performs the arithmetic linearization. This function maps
    its already-combined tensor terms into verifier-wide flattened indices.
    A ``ValueError`` signals that nonlinear lowering is required.
    """
    try:
        linear = expr.to_linear_expr()
    except Exception as exc:
        raise ValueError(expr) from exc

    out = LinearExpr(const=float(linear.constant))
    for term in linear.terms:
        var = _flat_var_from_ast(term.var, metadata)
        out.terms[var] = float(term.coeff)
    return out


def _flat_var_from_ast(var, metadata: VNNLib2Metadata) -> Var:
    """Map an indexed AST variable to its global flattened input/output index."""
    name = var.name
    if name in metadata.inputs:
        decl = metadata.inputs[name]
    elif name in metadata.outputs:
        decl = metadata.outputs[name]
    else:
        raise NotImplementedError(f"Unknown VNN-LIB variable: {name}")
    return Var(name, decl.offset + _flatten_index(tuple(var.indices), decl.shape))


def _flatten_index(indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Convert a tensor index tuple to a row-major offset within its tensor."""
    if len(indices) != len(shape):
        raise NotImplementedError(
            f"Index rank {len(indices)} does not match shape rank {len(shape)}"
        )
    index = 0
    for idx, dim in zip(indices, shape):
        if idx < 0 or idx >= dim:
            raise NotImplementedError(f"Index {indices} is out of bounds for {shape}")
        index = index * dim + idx
    return index


def _is_input_var(var: Var, metadata: VNNLib2Metadata) -> bool:
    """Return whether ``var`` belongs to the flattened input space."""
    return var.name in metadata.inputs


def _is_output_var(var: Var, metadata: VNNLib2Metadata) -> bool:
    """Return whether ``var`` belongs to the normal model-output space."""
    return var.name in metadata.outputs


def _is_single_input_bound(
        constraint: LinearConstraint, metadata: VNNLib2Metadata) -> bool:
    """Return whether ``expr <= 0`` bounds exactly one input variable."""
    terms = constraint.expr.terms
    return (
        len(terms) == 1
        and next(iter(terms))
        and _is_input_var(next(iter(terms)), metadata)
    )


def _try_apply_simple_box_assertion(
        branch: BranchState, assertion, metadata: VNNLib2Metadata) -> bool:
    """Apply an assertion made only of input bounds.

    This accepts only a single inequality or an ``and`` of inequalities. The
    updates are collected before mutating ``branch``, so mixed assertions leave
    the branch unchanged and use the generic DNF lowering path.
    """
    expr = assertion.expr
    if isinstance(expr, (vnnlib.LessEqual, vnnlib.LessThan,
                         vnnlib.GreaterEqual, vnnlib.GreaterThan)):
        comparisons = [expr]
    elif isinstance(expr, vnnlib.And):
        comparisons = list(expr.args)
    else:
        return False

    updates = []
    for comparison in comparisons:
        update = _simple_input_bound_from_comparison(comparison, metadata)
        if update is None:
            return False
        updates.append(update)

    for index, kind, value, strict in updates:
        _update_input_bound(branch, index, kind, value, strict)
    return True


def _simple_input_bound_from_comparison(
        comparison, metadata: VNNLib2Metadata
) -> Optional[Tuple[int, str, float, bool]]:
    """Return ``(input_index, bound_kind, value, strict)`` for simple bounds."""
    if isinstance(comparison, (vnnlib.LessEqual, vnnlib.LessThan)):
        left, right = comparison.lhs, comparison.rhs
        strict = isinstance(comparison, vnnlib.LessThan)
        if _is_ast_input_var(left, metadata) and _is_ast_scalar_constant(right):
            return (
                _ast_input_index(left, metadata), "upper",
                _ast_scalar_constant_value(right), strict)
        if _is_ast_scalar_constant(left) and _is_ast_input_var(right, metadata):
            return (
                _ast_input_index(right, metadata), "lower",
                _ast_scalar_constant_value(left), strict)
    elif isinstance(comparison, (vnnlib.GreaterEqual, vnnlib.GreaterThan)):
        left, right = comparison.lhs, comparison.rhs
        strict = isinstance(comparison, vnnlib.GreaterThan)
        if _is_ast_input_var(left, metadata) and _is_ast_scalar_constant(right):
            return (
                _ast_input_index(left, metadata), "lower",
                _ast_scalar_constant_value(right), strict)
        if _is_ast_scalar_constant(left) and _is_ast_input_var(right, metadata):
            return (
                _ast_input_index(right, metadata), "upper",
                _ast_scalar_constant_value(left), strict)
    return None


def _is_ast_input_var(expr, metadata: VNNLib2Metadata) -> bool:
    return isinstance(expr, vnnlib.Var) and expr.name in metadata.inputs


def _is_ast_scalar_constant(expr) -> bool:
    return hasattr(expr, "value")


def _ast_scalar_constant_value(expr) -> float:
    return float(expr.value)


def _ast_input_index(expr, metadata: VNNLib2Metadata) -> int:
    decl = metadata.inputs[expr.name]
    return decl.offset + _flatten_index(tuple(expr.indices), decl.shape)


def _try_apply_input_bound(
        branch: BranchState, constraint: Constraint,
        metadata: VNNLib2Metadata) -> bool:
    """Apply a one-variable affine input bound directly to ``box``.

    For ``coeff * x + const <= 0``, the coefficient sign determines whether
    ``-const / coeff`` is an upper or lower bound. Returns ``False`` when the
    constraint must remain deferred.
    """
    if not isinstance(constraint, LinearConstraint):
        return False
    if not _is_single_input_bound(constraint, metadata):
        return False
    var, coeff = next(iter(constraint.expr.terms.items()))
    bound = -constraint.expr.const / coeff
    if coeff > 0:
        _update_input_bound(branch, var.index, "upper", bound, constraint.strict)
    else:
        _update_input_bound(branch, var.index, "lower", bound, constraint.strict)
    return True


def _update_input_bound(
        branch: BranchState, index: int, kind: str,
        value: float, strict: bool) -> None:
    """Update an input bound and its strictness flag."""
    if kind == "lower":
        old = branch.box[index][0]
        if value > old:
            branch.box[index][0] = value
            branch.lower_strict[index] = strict
        elif value == old:
            branch.lower_strict[index] = branch.lower_strict[index] or strict
    else:
        old = branch.box[index][1]
        if value < old:
            branch.box[index][1] = value
            branch.upper_strict[index] = strict
        elif value == old:
            branch.upper_strict[index] = branch.upper_strict[index] or strict


def _infer_input_bounds_from_relations(
        box: List[List[float]], constraints: List[Constraint],
        metadata: VNNLib2Metadata) -> None:
    """Infer box bounds from deferred input-input relations ``x <= y``.

    This handles equality and ordering relations between inputs after direct
    single-variable bounds have been placed in ``box``. Iteration continues to
    a fixed point so chains such as ``x <= y <= z`` can infer all reachable
    interval tightenings.
    """
    relations = []
    for constraint in constraints:
        relation = _verifier_input_leq_indices(constraint, metadata)
        if relation is not None:
            relations.append(relation)
    if not relations:
        return

    # A bound can move across at most one input-input edge per pass. A chain of
    # N input variables needs at most N passes, plus one pass to detect that no
    # interval changed and stop.
    for _ in range(metadata.total_input_size + 1):
        changed = False
        for left, right in relations:
            changed |= _tighten_leq(box, left, right)
        if not changed:
            break


def _branch_input_equalities(
        branch: BranchState,
        metadata: VNNLib2Metadata) -> set[Tuple[int, int]]:
    """Return input-index pairs constrained equal in one branch."""
    inequalities = set()
    for constraint in branch.deferred:
        if isinstance(constraint, LinearConstraint) and constraint.strict:
            # Strict input relations do not imply equality when both directions
            # appear, e.g. x < y and y <= x is infeasible, not x == y.
            continue
        indices = _declared_input_leq_indices(constraint, metadata)
        if indices is None:
            continue
        inequalities.add(indices)
    return {
        tuple(sorted((left, right)))
        for left, right in inequalities
        if (right, left) in inequalities
    }


def _compress_equivalent_inputs(
        branches: List[BranchState], metadata: VNNLib2Metadata) -> None:
    """Merge input variables constrained equal in every branch."""
    if not branches:
        return

    # 1. Find input equalities that can be shared for the whole query. Each
    # branch is one OR alternative after all assertions are lowered, so an input
    # pair is shareable only if every alternative requires both equality
    # directions.
    common_equalities = _branch_input_equalities(branches[0], metadata)
    for branch in branches[1:]:
        common_equalities &= _branch_input_equalities(branch, metadata)
    if not common_equalities:
        return

    # 2. Merge declared inputs connected by those common equalities, e.g.
    # X_f[i] == X_g[i] == X_h[i].
    original_size = metadata.total_input_size
    parent = list(range(original_size))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in common_equalities:
        union(left, right)

    index_roots = [find(index) for index in range(original_size)]
    roots = sorted(set(index_roots))
    if len(roots) == original_size:
        return

    root_to_index = {root: index for index, root in enumerate(roots)}
    input_index_map = [root_to_index[root] for root in index_roots]

    # 3. Rebuild boxes in the compressed verifier input space.
    for branch in branches:
        compressed_branch = _new_branch_state(len(roots))
        for original_index, bounds in enumerate(branch.box):
            index = input_index_map[original_index]
            _update_input_bound(
                compressed_branch, index, "lower", bounds[0],
                bool(branch.lower_strict[original_index]))
            _update_input_bound(
                compressed_branch, index, "upper", bounds[1],
                bool(branch.upper_strict[original_index]))
        branch.box = compressed_branch.box
        branch.lower_strict = compressed_branch.lower_strict
        branch.upper_strict = compressed_branch.upper_strict
        # Equalities inside one merged component are enforced by construction.
        # Other input relations remain deferred as passthrough constraints.
        branch.deferred = [
            constraint for constraint in branch.deferred
            if not (
                (indices := _declared_input_leq_indices(constraint, metadata))
                is not None
                and input_index_map[indices[0]] == input_index_map[indices[1]])
        ]

    metadata.input_index_map = input_index_map
    metadata.total_input_size = len(roots)


def _declared_input_leq_indices(
        constraint: Constraint,
        metadata: VNNLib2Metadata) -> Optional[Tuple[int, int]]:
    """Return declared input indices for ``left <= right``.

    The returned indices use the original VNN-LIB declaration layout. This is
    the coordinate system used to decide which declared inputs can share one
    verifier input.
    """
    if not isinstance(constraint, LinearConstraint):
        return None
    if constraint.expr.const != 0.0:
        return None

    terms = constraint.expr.terms
    if len(terms) != 2:
        return None

    (v1, c1), (v2, c2) = list(terms.items())
    if not (_is_input_var(v1, metadata) and _is_input_var(v2, metadata)):
        return None

    if abs(c1 - 1.0) < 1e-12 and abs(c2 + 1.0) < 1e-12:
        return v1.index, v2.index
    if abs(c1 + 1.0) < 1e-12 and abs(c2 - 1.0) < 1e-12:
        return v2.index, v1.index
    return None


def _verifier_input_leq_indices(
        constraint: Constraint,
        metadata: VNNLib2Metadata) -> Optional[Tuple[int, int]]:
    """Return verifier input indices for ``left <= right``."""
    indices = _declared_input_leq_indices(constraint, metadata)
    if indices is None:
        return None
    return (
        metadata.input_index_map[indices[0]],
        metadata.input_index_map[indices[1]],
    )


def _tighten_leq(box: List[List[float]], left: int, right: int) -> bool:
    """Tighten two input intervals using ``box[left] <= box[right]``."""
    changed = False
    # left <= right implies upper(left) <= upper(right).
    if box[left][1] > box[right][1]:
        box[left][1] = box[right][1]
        changed = True
    # It also implies lower(right) >= lower(left).
    if box[right][0] < box[left][0]:
        box[right][0] = box[left][0]
        changed = True
    return changed


def _constraint_to_output_row(
        constraint: Constraint, metadata: VNNLib2Metadata) -> Tuple[List[float], float]:
    """Encode one normalized constraint as ``row @ wrapper_output <= rhs``.

    Ordinary output variables occupy the first columns. Optional input
    passthrough values follow them, then synthetic nonlinear outputs. For a
    linear ``sum(coeff * var) + const <= 0``, ``rhs`` is ``-const``. A
    nonlinear constraint selects its synthetic output with coefficient one.
    """
    row = [0.0] * metadata.total_output_size
    if isinstance(constraint, NonlinearConstraint):
        row[metadata.nonlinear_output_offset + constraint.output_index] = 1.0
        return row, 0.0
    for var, coeff in constraint.expr.terms.items():
        if _is_output_var(var, metadata):
            row[var.index] += coeff
        elif _is_input_var(var, metadata):
            if not metadata.include_input_passthrough:
                raise NotImplementedError(
                    "Input relation found, but input passthrough is disabled."
                )
            row[
                metadata.input_passthrough_offset
                + metadata.input_index_map[var.index]
            ] += coeff
        else:
            raise NotImplementedError(f"Unknown variable in constraint: {var}")
    return row, -constraint.expr.const


def _spec_has_static_contradiction(
        mat: np.ndarray, rhs: np.ndarray, strict: np.ndarray,
        metadata: VNNLib2Metadata) -> bool:
    """Return whether one lowered conjunction is algebraically empty.

    The check uses exact zero rows and exact opposite output-only rows. Rows
    with input passthrough columns are ignored here.
    """
    bounds_by_row = {}
    for row, row_rhs, row_strict in zip(mat, rhs, strict):
        # A zero row is a constant constraint: 0 <= rhs, or 0 < rhs when
        # the original comparison was strict.
        if not np.any(row != 0.0):
            row_is_satisfied = (
                0.0 < row_rhs if row_strict else 0.0 <= row_rhs
            )
            if not row_is_satisfied:
                return True
            continue

        # Normalize row and -row to the same key so their lower and upper
        # bounds can be compared.
        bound = _canonical_output_bound(row, row_rhs, metadata)
        if bound is None:
            continue
        key, bound_kind, value = bound

        # Track the tightest known lower and upper bound for this expression.
        lower, lower_strict, upper, upper_strict = bounds_by_row.get(
            key, (-np.inf, False, np.inf, False))
        if bound_kind == "upper":
            if value < upper or (value == upper and row_strict):
                upper = value
                upper_strict = bool(row_strict)
        else:
            if value > lower or (value == lower and row_strict):
                lower = value
                lower_strict = bool(row_strict)

        # Empty interval, or a single boundary point excluded by a strict side.
        if lower > upper:
            return True
        if lower == upper and (lower_strict or upper_strict):
            return True
        bounds_by_row[key] = (lower, lower_strict, upper, upper_strict)
    return False


def _canonical_output_bound(
        row: np.ndarray, rhs: float,
        metadata: VNNLib2Metadata,
) -> Optional[Tuple[Tuple[float, ...], str, float]]:
    """Return the canonical bound form for one output-only row."""
    if metadata.include_input_passthrough:
        start = metadata.input_passthrough_offset
        end = start + metadata.total_input_size
        # Rows with input passthrough columns depend on the current input
        # domain, so this output-only contradiction check does not classify them.
        if np.any(row[start:end] != 0.0):
            return None

    nonzero = np.flatnonzero(row != 0.0)
    assert nonzero.size > 0
    # Use the first nonzero coefficient to choose a stable orientation, so
    # ``row`` and ``-row`` map to the same expression key.
    if row[nonzero[0]] > 0.0:
        # row @ output <= rhs.
        return tuple(row.tolist()), "upper", float(rhs)
    # (-row) @ output >= -rhs.
    return tuple((-row).tolist()), "lower", float(-rhs)


def _merge_same_input_boxes(
        rv: list) -> Tuple[list, List[VNNLib2BranchStrictness]]:
    """Merge specs with the same flat input box and input strictness."""
    merged = {}
    for box, specs, strictness in rv:
        # Branches with the same numeric box but different open/closed bounds
        # cannot share one merged entry; their validation semantics differ.
        key = (
            tuple(tuple(bounds) for bounds in box),
            strictness.input_lower.tobytes(),
            strictness.input_upper.tobytes(),
        )
        if key not in merged:
            merged[key] = (
                box,
                [],
                VNNLib2BranchStrictness(
                    strictness.input_lower.copy(),
                    strictness.input_upper.copy(),
                    [],
                ),
            )
        merged[key][1].extend(specs)
        merged[key][2].specs.extend(mask.copy() for mask in strictness.specs)
    entries = list(merged.values())
    return (
        [(box, specs) for box, specs, _ in entries],
        [strictness for _, _, strictness in entries],
    )
