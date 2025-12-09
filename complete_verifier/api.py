#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from __future__ import annotations

import copy
import contextlib
import csv
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Set, cast

import numpy as np
import torch
import sympy
import yaml

import arguments
from attack import attack, get_attack_stats, reset_attack_stats
from beta_CROWN_solver import LiRPANet
from complete_verifier_func import bab as bab_core
from complete_verifier_func import complete_verifier as complete_verifier_core
from cuts.cut_utils import terminate_mip_processes
from incomplete_verifier_func import SpecHandler, incomplete_verifier as incomplete_verifier_core
from jit_precompile import precompile_jit_kernels
from lp_mip_solver import mip
from lp_test import compare_optimized_bounds_against_lp_bounds
from specifications import vnnlibHandler
from read_vnnlib import read_vnnlib
from load_model import load_model_onnx

__all__ = [
    "ABCrownSolver",
    "VerificationSpec",
    "default_config",
    "ConfigBuilder",
    "VNNCompInstance",
    "VNNCompBenchmark",
    "load_vnncomp_instance",
    "run_all_instances",
    "run_specific_instance",
    "input_vars",
    "output_vars",
]

# Epsilon used to turn strict inequalities into relaxed non-strict constraints.
_STRICT_INEQUALITY_EPS = 1e-8


def _ensure_config_defaults() -> None:
    """Load default values into arguments.Config if needed."""
    if not getattr(arguments.Config, "all_args", None):
        arguments.Config.all_args = {}
    if len(arguments.Config.all_args) == 0:
        arguments.Config.construct_config_dict(arguments.Config.default_args)
        arguments.Config.update_arguments()


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if not isinstance(node, MutableMapping):
                raise TypeError(f"Cannot merge mapping into non-mapping at {key}.")
            _deep_update(node, value)
        else:
            base[key] = value


def _clone_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(config)


def _shift_other_by_eps(
    other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int],
    delta: float,
) -> Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float]:
    """Shift comparison bounds by delta; passthrough VariableVector unchanged."""
    if isinstance(other, VariableVector):
        return other
    if torch.is_tensor(other):
        return other + delta
    if isinstance(other, np.ndarray):
        return other + delta
    if isinstance(other, (float, int)):
        return float(other) + delta
    if isinstance(other, Sequence):
        return type(other)(item + delta for item in other)
    raise TypeError(f"Unsupported comparison with type {type(other).__name__}")


def _assign_path(target: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    node: MutableMapping[str, Any] = target
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, MutableMapping):
            raise TypeError(f"Cannot assign into non-mapping at {'.'.join(path)}.")
        node = child
    node[path[-1]] = value


_ensure_config_defaults()
_DEFAULT_CONFIG = _clone_config(arguments.Config.all_args)
# Align new API defaults with the legacy front-end: prefer automatic verifier selection.
_DEFAULT_CONFIG.setdefault("general", {})["complete_verifier"] = "auto"


def default_config() -> Dict[str, Any]:
    """Clone the project-wide default configuration."""
    return _clone_config(_DEFAULT_CONFIG)


class ConfigBuilder:
    """Chainable helper for building verification configs."""

    def __init__(self, base: Optional[Mapping[str, Any]] = None):
        _ensure_config_defaults()
        if base is None:
            base = _DEFAULT_CONFIG
        self._cfg = _clone_config(base)

    @classmethod
    def from_defaults(cls) -> "ConfigBuilder":
        return cls(_DEFAULT_CONFIG)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ConfigBuilder":
        return cls(config)

    def update(self,
               *modifiers: Union[Mapping[str, Any], Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
               **overrides: Any) -> "ConfigBuilder":
        nested_overrides: Dict[str, Any] = {}
        for key, value in overrides.items():
            if isinstance(key, str) and "__" in key:
                _assign_path(nested_overrides, key.split("__"), value)
            else:
                nested_overrides[key] = value
        cfg = _clone_config(self._cfg)
        for modifier in modifiers:
            if callable(modifier):
                updated = modifier(_clone_config(cfg))
                if updated is not None:
                    cfg = _clone_config(updated)
            elif isinstance(modifier, Mapping):
                _deep_update(cfg, modifier)
            else:
                raise TypeError(f"Unsupported config modifier type: {type(modifier).__name__}")
        if nested_overrides:
            _deep_update(cfg, nested_overrides)
        self._cfg = cfg
        return self

    def set(self, **overrides: Any) -> "ConfigBuilder":
        return self.update(**overrides)

    def replace(self, config: Mapping[str, Any]) -> "ConfigBuilder":
        self._cfg = _clone_config(config)
        return self

    def copy(self) -> "ConfigBuilder":
        return ConfigBuilder(self._cfg)

    def to_dict(self) -> Dict[str, Any]:
        return _clone_config(self._cfg)

    def __call__(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_yaml(cls, path: str) -> "ConfigBuilder":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls().update(data)


@dataclass(frozen=True)
class VNNCompInstance:
    index: int
    onnx_path: str
    vnnlib_path: str
    csv_row: Tuple[str, ...]


class VNNCompBenchmark:
    """
    Convenience wrapper for running α,β-CROWN on VNN-COMP benchmark configs.

    Instantiate with the YAML config path, then call `run_all_instances()` to
    process every row in the benchmark CSV or `run_specific_instance(idx)` to
    execute a single row. Use `load_instance(idx)` if you want the prepared
    `(spec, computing_graph, config, metadata)` tuple for custom control.
    Optionally pass `root=...` to override the benchmark root directory
    (otherwise the loader searches relative to the config and the workspace).
    """

    def __init__(self, config_path: str, *, root: Optional[str] = None):
        self.config_path = os.path.abspath(config_path)
        self._root_override = root
        self._config: Optional[Dict[str, Any]] = None
        self._entries: Optional[Tuple[VNNCompInstance, ...]] = None

    def _ensure_loaded(self) -> None:
        if self._config is None or self._entries is None:
            config, entries = _load_vnncomp_instances(self.config_path, self._root_override)
            self._config = config
            self._entries = entries

    @property
    def config(self) -> Dict[str, Any]:
        """Return a deep copy of the parsed configuration."""
        self._ensure_loaded()
        return _clone_config(self._config)  # type: ignore[arg-type]

    @property
    def entries(self) -> Tuple[VNNCompInstance, ...]:
        """Return the list of benchmark instances (cached)."""
        self._ensure_loaded()
        return tuple(self._entries)  # type: ignore[arg-type]

    def load_instance(
        self,
        instance_id: int,
    ) -> Tuple[VerificationSpec, Union[str, torch.nn.Module], Dict[str, Any], VNNCompInstance]:
        """
        Prepare a single instance, returning `(spec, onnx_path, config, metadata)`.
        """
        self._ensure_loaded()
        entries = self._entries  # type: ignore[assignment]
        if instance_id < 0 or instance_id >= len(entries):
            raise IndexError(f"instance_id {instance_id} out of range (0..{len(entries) - 1}).")
        entry = entries[instance_id]
        if not os.path.exists(entry.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {entry.onnx_path}")
        if not os.path.exists(entry.vnnlib_path):
            raise FileNotFoundError(f"vnnlib spec not found: {entry.vnnlib_path}")
        spec = VerificationSpec.build_from_vnnlib(entry.vnnlib_path)
        return spec, entry.onnx_path, self.config, entry

    def run_all_instances(self) -> Tuple[Tuple[int, VNNCompInstance, SolveResult], ...]:
        """Execute every benchmark entry declared in the YAML."""
        self._ensure_loaded()
        return _execute_vnncomp_entries(self.config, self.entries)

    def run_specific_instance(
        self,
        instance_id: int,
    ) -> Tuple[int, VNNCompInstance, SolveResult]:
        """Execute exactly one benchmark entry by index."""
        self._ensure_loaded()
        entries = self.entries
        if instance_id < 0 or instance_id >= len(entries):
            raise IndexError(f"instance_id {instance_id} out of range (0..{len(entries) - 1}).")
        return _execute_vnncomp_entries(self.config, (entries[instance_id],))[0]


def _resolve_vnncomp_root(
    config_path: str,
    config: Mapping[str, Any],
    root_override: Optional[str] = None,
) -> str:
    general = config.get("general", {})
    raw_root = general.get("root_path")
    if not raw_root:
        raise KeyError("VNN-COMP config must define general.root_path.")
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = []
    if root_override:
        candidates.append(os.path.abspath(root_override))
    if os.path.isabs(raw_root):
        candidates.append(raw_root)
    else:
        candidates.append(os.path.normpath(os.path.join(cfg_dir, raw_root)))
        sanitized = raw_root.lstrip("./")
        if sanitized:
            cfg_path = Path(cfg_dir).resolve()
            for ancestor in cfg_path.parents:
                candidate = ancestor / sanitized
                candidates.append(str(candidate))
    env_root = os.environ.get("ABCROWN_VNNCOMP_ROOT")
    if env_root:
        candidates.append(os.path.normpath(os.path.join(env_root, os.path.basename(raw_root))))
    seen = set()
    ordered = [path for path in candidates if path not in seen and not seen.add(path)]
    for path in ordered:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Unable to resolve benchmark root directory. Tried:\n"
        + "\n".join(f"  - {p}" for p in ordered)
    )


def _load_vnncomp_instances(
    config_path: str,
    root_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], Tuple[VNNCompInstance, ...]]:
    builder = ConfigBuilder.from_yaml(config_path)
    config = builder()
    root = _resolve_vnncomp_root(config_path, config, root_override)
    config = _clone_config(config)
    _assign_path(config, ("general", "root_path"), root)

    csv_name = config.get("general", {}).get("csv_name")
    if not csv_name:
        raise KeyError("VNN-COMP config must define general.csv_name.")
    csv_path = os.path.join(root, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Instances CSV not found: {csv_path}")

    entries: list[VNNCompInstance] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            head = row[0].strip()
            if not head or head.startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"CSV row must contain at least model and spec columns: {row}")
            onnx_rel, vnnlib_rel = row[0].strip(), row[1].strip()
            onnx_path = os.path.normpath(os.path.join(root, onnx_rel))
            vnnlib_path = os.path.normpath(os.path.join(root, vnnlib_rel))
            entries.append(
                VNNCompInstance(
                    index=len(entries),
                    onnx_path=onnx_path,
                    vnnlib_path=vnnlib_path,
                    csv_row=tuple(row),
                )
            )
    if not entries:
        raise ValueError(f"No instances found in {csv_path}.")
    return config, tuple(entries)


def load_vnncomp_instance(
    config_path: str,
    *,
    instance_id: int,
    root: Optional[str] = None,
) -> Tuple[VerificationSpec, Union[str, torch.nn.Module], Dict[str, Any], VNNCompInstance]:
    """
    Prepare a single VNN-COMP instance.

    Returns the specification, computing graph (ONNX path), cloned config dict,
    and metadata describing the instance. Set `root=...` to override the
    benchmark root directory if needed.
    """
    runner = VNNCompBenchmark(config_path, root=root)
    return runner.load_instance(instance_id)


def _execute_vnncomp_entries(
    config: Mapping[str, Any],
    entries: Sequence[VNNCompInstance],
) -> Tuple[Tuple[int, VNNCompInstance, SolveResult], ...]:
    results: list[Tuple[int, VNNCompInstance, SolveResult]] = []
    for entry in entries:
        if not os.path.exists(entry.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {entry.onnx_path}")
        if not os.path.exists(entry.vnnlib_path):
            raise FileNotFoundError(f"vnnlib spec not found: {entry.vnnlib_path}")
        spec = VerificationSpec.build_from_vnnlib(entry.vnnlib_path)
        solver = ABCrownSolver(
            spec,
            entry.onnx_path,
            config=_clone_config(config),
            name=f"vnncomp/{entry.index}",
        )
        result = solver.solve()
        results.append((entry.index, entry, result))
    return tuple(results)


def run_all_instances(
    config_path: str,
    *,
    root: Optional[str] = None,
) -> Tuple[Tuple[int, VNNCompInstance, SolveResult], ...]:
    """
    Run every instance declared in a VNN-COMP YAML config.
    Returns a tuple of (index, instance_metadata, solve_result) entries.
    Set `root=...` to override the benchmark root directory if needed.
    """
    runner = VNNCompBenchmark(config_path, root=root)
    return runner.run_all_instances()


def run_specific_instance(
    config_path: str,
    instance_id: int,
    *,
    root: Optional[str] = None,
) -> Tuple[int, VNNCompInstance, SolveResult]:
    """
    Run a single instance (by index) defined in a VNN-COMP YAML config.
    Returns (index, instance_metadata, solve_result).
    Set `root=...` to override the benchmark root directory if needed.
    """
    runner = VNNCompBenchmark(config_path, root=root)
    return runner.run_specific_instance(instance_id)


class VariableVector:
    """Symbolic vector used when writing spec expressions."""

    def __init__(self, kind: str, shape: Union[int, Sequence[int], torch.Size]):
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, torch.Size):
            shape = tuple(shape)
        else:
            shape = tuple(shape)
        if len(shape) == 0:
            raise ValueError("Shape must contain at least one dimension.")
        self.kind = kind
        self.shape = shape
        self.size = int(np.prod(shape))

    def _flatten_index(self, key: Union[int, Tuple[int, ...]]) -> int:
        if isinstance(key, tuple):
            if len(key) != len(self.shape):
                raise IndexError(
                    f"Expected {len(self.shape)} indices, got {len(key)}.")
            return int(np.ravel_multi_index(key, self.shape))
        idx = int(key)
        if idx < 0:
            idx += self.size
        if not 0 <= idx < self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}.")
        return idx

    def __getitem__(self, key: Union[int, Tuple[int, ...]]):
        flat_idx = self._flatten_index(key)
        return LinearExpr({(self.kind, flat_idx): 1.0}, 0.0)

    def _iter_other(self, other: Union[Sequence[float], np.ndarray, torch.Tensor, float, int]) -> Sequence[float]:
        if isinstance(other, torch.Tensor):
            flat = other.detach().view(-1).tolist()
        elif isinstance(other, np.ndarray):
            flat = np.asarray(other).reshape(-1).tolist()
        elif isinstance(other, (list, tuple)):
            flat = list(other)
        elif isinstance(other, (int, float)):
            flat = [float(other)] * self.size
        else:
            raise TypeError(f"Unsupported comparison with type {type(other).__name__}")
        if len(flat) != self.size:
            raise ValueError(f"Comparison value size {len(flat)} does not match VariableVector size {self.size}.")
        return flat

    def _compare(self, other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int], op: str) -> "Predicate":
        if op not in (">=", "<=", ">", "<"):
            raise ValueError(f"Unsupported comparison operator {op}.")
        if op in (">", "<"):
            # Map strict inequalities to non-strict ones with an epsilon shift on the bound.
            delta = _STRICT_INEQUALITY_EPS if op == ">" else -_STRICT_INEQUALITY_EPS
            adjusted = _shift_other_by_eps(other, delta)
            mapped_op = ">=" if op == ">" else "<="
            return self._compare(adjusted, mapped_op)
        bounds = self._iter_other(other)
        pred: Optional[Predicate] = None
        for idx, bound in enumerate(bounds):
            atom = (self[idx] >= bound) if op == ">=" else (self[idx] <= bound)
            pred = atom if pred is None else (pred & atom)
        if pred is None:
            raise ValueError("Empty comparison on VariableVector.")
        return pred

    def __ge__(self, other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int]) -> "Predicate":
        return self._compare(other, ">=")

    def __le__(self, other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int]) -> "Predicate":
        return self._compare(other, "<=")

    def __gt__(self, other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int]) -> "Predicate":
        return self._compare(other, ">")

    def __lt__(self, other: Union["VariableVector", Sequence[float], np.ndarray, torch.Tensor, float, int]) -> "Predicate":
        return self._compare(other, "<")


def input_vars(shape: Union[int, Sequence[int], torch.Size]) -> VariableVector:
    """Create symbolic input variables (x)."""

    return VariableVector("input", shape)


def output_vars(num_outputs: int) -> VariableVector:
    """Create symbolic output variables (y)."""

    return VariableVector("output", num_outputs)


def _ensure_linear_expr(value: Union["LinearExpr", int, float]) -> "LinearExpr":
    if isinstance(value, LinearExpr):
        return value
    if isinstance(value, (int, float)):
        return LinearExpr({}, float(value))
    raise TypeError(f"Unsupported operand type {type(value).__name__} in linear expression.")


class LinearExpr:
    __slots__ = ("coeffs", "constant")

    def __init__(self,
                 coeffs: Optional[Dict[Tuple[str, int], float]] = None,
                 constant: float = 0.0) -> None:
        self.coeffs: Dict[Tuple[str, int], float] = {}
        if coeffs:
            for key, value in coeffs.items():
                if value != 0:
                    self.coeffs[key] = float(value)
        self.constant = float(constant)

    def _combine(self, other: "LinearExpr", scale: float) -> "LinearExpr":
        new_coeffs = self.coeffs.copy()
        for key, value in other.coeffs.items():
            new_coeffs[key] = new_coeffs.get(key, 0.0) + scale * value
            if new_coeffs[key] == 0:
                del new_coeffs[key]
        return LinearExpr(new_coeffs, self.constant + scale * other.constant)

    def __add__(self, other: Union["LinearExpr", int, float]) -> "LinearExpr":
        return self._combine(_ensure_linear_expr(other), 1.0)

    def __radd__(self, other: Union["LinearExpr", int, float]) -> "LinearExpr":
        return _ensure_linear_expr(other)._combine(self, 1.0)

    def __sub__(self, other: Union["LinearExpr", int, float]) -> "LinearExpr":
        return self._combine(_ensure_linear_expr(other), -1.0)

    def __rsub__(self, other: Union["LinearExpr", int, float]) -> "LinearExpr":
        return _ensure_linear_expr(other)._combine(self, -1.0)

    def __mul__(self, scalar: Union[int, float]) -> "LinearExpr":
        scalar = float(scalar)
        coeffs = {k: scalar * v for k, v in self.coeffs.items() if scalar * v != 0}
        return LinearExpr(coeffs, scalar * self.constant)

    def __rmul__(self, scalar: Union[int, float]) -> "LinearExpr":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> "LinearExpr":
        scalar = float(scalar)
        if scalar == 0:
            raise ZeroDivisionError("Division by zero in linear expression.")
        coeffs = {k: v / scalar for k, v in self.coeffs.items() if v / scalar != 0}
        return LinearExpr(coeffs, self.constant / scalar)

    def __neg__(self) -> "LinearExpr":
        coeffs = {k: -v for k, v in self.coeffs.items()}
        return LinearExpr(coeffs, -self.constant)

    def __le__(self, other: Union["LinearExpr", int, float]) -> "ComparisonPredicate":
        return ComparisonPredicate(self, _ensure_linear_expr(other), "<=")

    def __ge__(self, other: Union["LinearExpr", int, float]) -> "ComparisonPredicate":
        return ComparisonPredicate(self, _ensure_linear_expr(other), ">=")

    def __lt__(self, other: Union["LinearExpr", int, float]) -> "ComparisonPredicate":
        return ComparisonPredicate(self, _ensure_linear_expr(other), "<")

    def __gt__(self, other: Union["LinearExpr", int, float]) -> "ComparisonPredicate":
        return ComparisonPredicate(self, _ensure_linear_expr(other), ">")

    def __eq__(self, other: object) -> "Predicate":  # type: ignore[override]
        other_expr = _ensure_linear_expr(other)  # type: ignore[arg-type]
        return (self <= other_expr) & (self >= other_expr)


class Predicate:
    def __and__(self, other: "Predicate") -> "Predicate":
        return AndPredicate(self, _ensure_predicate(other))

    def __rand__(self, other: "Predicate") -> "Predicate":
        return AndPredicate(_ensure_predicate(other), self)

    def __or__(self, other: "Predicate") -> "Predicate":
        return OrPredicate(self, _ensure_predicate(other))

    def __ror__(self, other: "Predicate") -> "Predicate":
        return OrPredicate(_ensure_predicate(other), self)


def _ensure_predicate(value: Union["Predicate", bool]) -> "Predicate":
    if isinstance(value, Predicate):
        return value
    raise TypeError(f"Unsupported boolean expression operand {type(value).__name__}")


class ComparisonPredicate(Predicate):
    def __init__(self, lhs: LinearExpr, rhs: LinearExpr, op: str) -> None:
        if op not in ("<=", ">=", "<", ">"):
            raise ValueError(f"Unsupported comparison operator {op}.")
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def normalized_expr(self) -> LinearExpr:
        """Return lhs - rhs <= 0 in linear form."""

        if self.op == "<=":
            return self.lhs - self.rhs
        if self.op == "<":
            return self.lhs - self.rhs + _STRICT_INEQUALITY_EPS
        if self.op == ">=":
            return self.rhs - self.lhs
        return self.rhs - self.lhs + _STRICT_INEQUALITY_EPS
def _negate_comparison(pred: ComparisonPredicate) -> ComparisonPredicate:
    """Return the logical negation of a comparison predicate."""
    if pred.op == "<=":
        new_op = ">"
    elif pred.op == "<":
        new_op = ">="
    elif pred.op == ">=":
        new_op = "<"
    else:
        new_op = "<="
    return ComparisonPredicate(pred.lhs, pred.rhs, new_op)



class AndPredicate(Predicate):
    def __init__(self, left: Predicate, right: Predicate) -> None:
        self.left = left
        self.right = right


class OrPredicate(Predicate):
    def __init__(self, left: Predicate, right: Predicate) -> None:
        self.left = left
        self.right = right


def _assert_strict_output(predicate: Predicate) -> None:
    """Ensure output constraints use strict inequalities only."""

    def _walk(node: Predicate) -> None:
        if isinstance(node, ComparisonPredicate):
            if node.op in ("<=", ">="):
                raise ValueError("Output constraints must use strict inequalities (< or >).")
            return
        if isinstance(node, AndPredicate):
            _walk(node.left)
            _walk(node.right)
            return
        if isinstance(node, OrPredicate):
            _walk(node.left)
            _walk(node.right)
            return
        raise TypeError(f"Unsupported predicate type {type(node).__name__}")

    _walk(predicate)


def _predicate_to_dnf(pred: Predicate, *, negate: bool = False) -> Sequence[Sequence[ComparisonPredicate]]:
    symbol_map: Dict[sympy.Symbol, ComparisonPredicate] = {}

    def to_sympy(node: Predicate) -> sympy.Expr:
        if isinstance(node, ComparisonPredicate):
            symbol = sympy.Symbol(f"p{len(symbol_map)}", boolean=True)
            symbol_map[symbol] = node
            return symbol
        if isinstance(node, AndPredicate):
            return sympy.And(to_sympy(node.left), to_sympy(node.right))
        if isinstance(node, OrPredicate):
            return sympy.Or(to_sympy(node.left), to_sympy(node.right))
        raise TypeError(f"Unsupported predicate type {type(node).__name__}")

    sympy_expr = to_sympy(pred)
    if negate:
        sympy_expr = sympy.Not(sympy_expr)
    dnf_expr = sympy.to_dnf(sympy_expr, simplify=True)

    def extract(expr: sympy.Expr) -> Sequence[Sequence[ComparisonPredicate]]:
        if expr is sympy.true:
            return [[]]
        if expr is sympy.false:
            return []
        if expr.is_Symbol:
            return [[symbol_map[expr]]]
        if expr.func is sympy.Not:
            inner = expr.args[0]
            if inner.is_Symbol:
                return [[_negate_comparison(symbol_map[inner])]]
            raise TypeError("Unexpected negation structure in DNF expression.")
        if expr.func is sympy.Or:
            clauses: list[list[ComparisonPredicate]] = []
            for arg in expr.args:
                clauses.extend(extract(arg))
            return clauses
        if expr.func is sympy.And:
            clause: list[ComparisonPredicate] = []
            for arg in expr.args:
                if arg is sympy.true:
                    continue
                if arg is sympy.false:
                    return []
                if arg.is_Symbol:
                    clause.append(symbol_map[arg])
                elif arg.func is sympy.Not and arg.args[0].is_Symbol:
                    clause.append(_negate_comparison(symbol_map[arg.args[0]]))
                else:
                    raise TypeError("Nested boolean expressions are not supported in comparisons.")
            return [clause]
        raise TypeError("Unexpected expression returned from sympy.to_dnf().")

    clauses = extract(dnf_expr)
    return clauses


def _aggregate_rows(
    clause: Sequence[ComparisonPredicate],
    *,
    expected_kind: str,
    vector: VariableVector,
) -> Tuple[torch.Tensor, torch.Tensor, Sequence[ComparisonPredicate]]:
    rows = []
    rhs_values = []
    input_preds: list[ComparisonPredicate] = []
    for atom in clause:
        expr = atom.normalized_expr()
        coeff_row = [0.0] * vector.size
        kinds_in_expr: Set[str] = set(kind for kind, _ in expr.coeffs.keys())
        if not expr.coeffs:
            raise ValueError("Constraints must involve at least one variable.")
        if kinds_in_expr == {expected_kind}:
            for (kind, idx), coeff in expr.coeffs.items():
                coeff_row[idx] += coeff
            rhs_values.append(-expr.constant)
            rows.append(coeff_row)
        elif kinds_in_expr == {"input"}:
            input_preds.append(atom)
        else:
            raise ValueError("Mixed input/output constraints are not supported.")
    if rows:
        C = torch.tensor(rows, dtype=torch.float32)
        rhs_tensor = torch.tensor(rhs_values, dtype=torch.float32)
    else:
        C = torch.empty((0, vector.size), dtype=torch.float32)
        rhs_tensor = torch.empty((0,), dtype=torch.float32)
    return C, rhs_tensor, input_preds


def _parse_input_bounds(predicate: Predicate, inputs: VariableVector) -> Tuple[torch.Tensor, torch.Tensor]:
    def _flatten_conjunction(node: Predicate) -> Sequence[ComparisonPredicate]:
        if isinstance(node, ComparisonPredicate):
            return [node]
        if isinstance(node, AndPredicate):
            return [atom for child in (node.left, node.right) for atom in _flatten_conjunction(child)]
        if isinstance(node, OrPredicate):
            raise ValueError("Input specification must be a conjunction without OR operators.")
        raise TypeError(f"Unsupported predicate type {type(node).__name__}")

    try:
        clauses = [_flatten_conjunction(predicate)]
    except ValueError as exc:
        # Fallback to sympy DNF for expressions that may include ORs.
        clauses = _predicate_to_dnf(predicate)
        if len(clauses) != 1:
            raise ValueError("Input specification must be a conjunction without OR operators.") from exc

    lower = [-float("inf")] * inputs.size
    upper = [float("inf")] * inputs.size
    for atom in clauses[0]:
        expr = atom.normalized_expr()
        if not expr.coeffs:
            raise ValueError("Input constraints must involve at least one variable.")
        if len(expr.coeffs) != 1:
            raise ValueError("Input constraints must refer to a single variable.")
        (kind, idx), coeff = next(iter(expr.coeffs.items()))
        if kind != "input":
            raise ValueError("Input constraints can only reference input variables.")
        bound = -expr.constant / coeff
        if coeff > 0:
            upper[idx] = min(upper[idx], bound)
        else:
            lower[idx] = max(lower[idx], bound)
    if any(np.isinf(value) for value in lower) or any(np.isinf(value) for value in upper):
        raise ValueError("Each input dimension must have both lower and upper bounds.")
    lower_tensor = torch.tensor(lower, dtype=torch.float32).view((1, *inputs.shape))
    upper_tensor = torch.tensor(upper, dtype=torch.float32).view((1, *inputs.shape))
    return lower_tensor, upper_tensor


@dataclass
class VerificationSpec:
    input_spec: "VerificationSpec.InputSpec"
    output_spec: "VerificationSpec.OutputSpec"

    @dataclass
    class InputSpec:
        lower: torch.Tensor
        upper: torch.Tensor

        def __post_init__(self) -> None:
            self.lower = torch.as_tensor(self.lower).detach().clone().float()
            self.upper = torch.as_tensor(self.upper).detach().clone().float()
            if self.lower.shape != self.upper.shape:
                raise ValueError("Lower and upper bounds must share the same shape.")
            if self.lower.ndim < 2:
                raise ValueError("Input bounds must include batch and data dimensions.")

        @property
        def num_inputs(self) -> int:
            return self.lower.shape[0]

        @property
        def data_shape(self) -> Tuple[int, ...]:
            return tuple(self.lower.shape[1:])

        def reshape(self, target_shape: Sequence[int]) -> None:
            target = tuple(int(dim) for dim in target_shape)
            current = self.data_shape
            if current == target:
                return
            flat_current = int(np.prod(current)) if current else 1
            flat_target = int(np.prod(target)) if target else 1
            if flat_current != flat_target:
                raise ValueError(
                    f"Cannot reshape input from {current} to {target}: "
                    f"flattened size mismatch ({flat_current} vs {flat_target})."
                )
            self.lower = self.lower.reshape(self.num_inputs, *target)
            self.upper = self.upper.reshape(self.num_inputs, *target)

    @dataclass
    class OutputSpec:
        clauses: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]

        def __post_init__(self) -> None:
            if not isinstance(self.clauses, Sequence) or len(self.clauses) == 0:
                raise ValueError("At least one specification clause is required.")
            self.clauses = list(self.clauses)

        @staticmethod
        def _wrap_clause(clause: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            if not isinstance(clause, Sequence) or len(clause) != 2:
                raise ValueError("Each clause must be a (C, rhs) pair.")
            c, rhs = clause
            c_tensor = torch.as_tensor(c).detach().clone().float()
            rhs_tensor = torch.as_tensor(rhs).detach().clone().float()
            if c_tensor.ndim == 1:
                c_tensor = c_tensor.unsqueeze(0)
            if c_tensor.ndim != 2:
                raise ValueError("Clause matrix must be 2-D (num_and, num_outputs).")
            if rhs_tensor.ndim == 0:
                rhs_tensor = rhs_tensor.unsqueeze(0)
            if rhs_tensor.ndim != 1:
                raise ValueError("Clause rhs must be 1-D (num_and,).")
            if c_tensor.shape[0] != rhs_tensor.shape[0]:
                raise ValueError("Clause matrix and rhs must share the same number of rows.")
            return c_tensor, rhs_tensor

        def normalize(self, num_inputs: int) -> None:
            first_clause = self.clauses[0]
            is_or_list = (
                isinstance(first_clause, Sequence)
                and len(first_clause) == 2
                and isinstance(first_clause[0], torch.Tensor)
            )
            if is_or_list:
                wrapped = [self._wrap_clause(c) for c in self.clauses]
                count = max(1, num_inputs)
                self.clauses = [wrapped for _ in range(count)]
            else:
                if len(self.clauses) not in {1, num_inputs}:
                    raise ValueError("Clauses must be provided for each input or shared as a single list.")
                if len(self.clauses) == 1 and num_inputs > 1:
                    shared = [self._wrap_clause(c) for c in self.clauses[0]]
                    self.clauses = [shared for _ in range(num_inputs)]
                else:
                    self.clauses = [
                        [self._wrap_clause(c) for c in per_input] for per_input in self.clauses
                    ]

    @property
    def num_inputs(self) -> int:
        return self.input_spec.num_inputs

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (-1, *self.input_spec.data_shape)

    def __post_init__(self) -> None:
        if not isinstance(self.input_spec, VerificationSpec.InputSpec):
            raise TypeError("input_spec must be an instance of VerificationSpec.InputSpec.")
        if not isinstance(self.output_spec, VerificationSpec.OutputSpec):
            raise TypeError("output_spec must be an instance of VerificationSpec.OutputSpec.")
        self.output_spec.normalize(self.input_spec.num_inputs)

    @property
    def lower(self) -> torch.Tensor:
        return self.input_spec.lower

    @property
    def upper(self) -> torch.Tensor:
        return self.input_spec.upper

    @property
    def clauses(self) -> Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        return self.output_spec.clauses

    def reshape_input(self, target_shape: Sequence[int]) -> None:
        self.input_spec.reshape(target_shape)

    def to_vnnlib(self) -> Sequence[Tuple[Sequence[Tuple[float, float]], Sequence[Tuple[np.ndarray, np.ndarray]]]]:
        vnn_entries = []
        for idx in range(self.num_inputs):
            lb = self.input_spec.lower[idx].view(-1).cpu().numpy()
            ub = self.input_spec.upper[idx].view(-1).cpu().numpy()
            input_box = list(zip(lb.tolist(), ub.tolist()))
            or_clauses = []
            for c_tensor, rhs_tensor in self.output_spec.clauses[idx]:
                or_clauses.append((c_tensor.cpu().numpy(), rhs_tensor.cpu().numpy()))
            vnn_entries.append((input_box, or_clauses))
        return vnn_entries

    @classmethod
    def build_from_center(cls,
                          center: torch.Tensor,
                          epsilon: Union[float, torch.Tensor],
                          clauses: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]) -> "VerificationSpec":
        center_t = torch.as_tensor(center).float()
        eps_t = torch.as_tensor(epsilon).float()
        if eps_t.ndim == 0:
            eps_t = torch.full_like(center_t, float(eps_t))
        lower = center_t - eps_t
        upper = center_t + eps_t
        lower_batched = lower.unsqueeze(0) if lower.ndim == center_t.ndim else lower
        upper_batched = upper.unsqueeze(0) if upper.ndim == center_t.ndim else upper
        return cls.build_from_input_bounds(lower_batched, upper_batched, clauses)

    @classmethod
    def build_from_input_bounds(
        cls,
        lower: torch.Tensor,
        upper: torch.Tensor,
        clauses: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> "VerificationSpec":
        input_spec = cls.InputSpec(lower, upper)
        output_spec = cls.OutputSpec(clauses)
        return cls(input_spec=input_spec, output_spec=output_spec)

    @classmethod
    def build_from_expressions(
        cls,
        *,
        input_vars: VariableVector,
        output_vars: VariableVector,
        input_constraint: Predicate,
        output_constraint: Predicate,
    ) -> "VerificationSpec":
        """Build a specification from symbolic expressions."""
        lower, upper = _parse_input_bounds(input_constraint, input_vars)
        output_clauses = _predicate_to_dnf(
            output_constraint,
            negate=True,
        )
        clauses = []
        lower_flat = lower.view(1, -1)[0]
        upper_flat = upper.view(1, -1)[0]
        for clause in output_clauses:
            C, rhs, input_preds = _aggregate_rows(clause, expected_kind="output", vector=output_vars)
            for pred in input_preds:
                expr = pred.normalized_expr()
                if len(expr.coeffs) != 1:
                    raise ValueError(
                        "Input-side constraints inside output clause must reference a single input variable."
                    )
                (kind, idx), coeff = next(iter(expr.coeffs.items()))
                if kind != "input":
                    raise ValueError("Mixed input/output constraints are not supported.")
                const = float(expr.constant)
                bound_val = upper_flat[idx] if coeff >= 0 else lower_flat[idx]
                if coeff * bound_val + const > 1e-8:
                    raise ValueError("Input-side constraint inside output clause is not satisfied by input bounds.")
            clause_entries: list[Tuple[torch.Tensor, torch.Tensor]] = []
            if C.numel() > 0:
                clause_entries.append((C, rhs))
            clauses.append(clause_entries)

        repeat_shape = (max(1, len(clauses)),) + (1,) * len(input_vars.shape)
        lower_batched = lower.repeat(repeat_shape)
        upper_batched = upper.repeat(repeat_shape)

        input_spec = cls.InputSpec(lower_batched, upper_batched)
        output_spec = cls.OutputSpec(clauses)
        return cls(input_spec=input_spec, output_spec=output_spec)

    @classmethod
    def build_from_vnnlib(
        cls,
        path: str,
        input_shape: Optional[Sequence[int]] = None,
    ) -> "VerificationSpec":
        vnnlib = read_vnnlib(path)
        if input_shape is None:
            if not vnnlib:
                raise ValueError("Empty vnnlib or input_shape missing.")
            num_inputs = len(vnnlib[0][0])
            shape = [-1, num_inputs]
        else:
            if len(input_shape) == 0:
                raise ValueError("input_shape must describe the input dimensions.")
            shape = list(input_shape)
            if shape[0] != -1:
                shape = [-1, *shape]
        handler = vnnlibHandler(vnnlib, shape)
        specs = handler.all_specs
        x, c, rhs, or_spec_size, _, _ = specs.get("cpu")
        lower = x.ptb.x_L.detach().cpu()
        upper = x.ptb.x_U.detach().cpu()
        if or_spec_size.dim() == 0:
            or_spec_size = or_spec_size.unsqueeze(0)
        c_cpu = c.detach().cpu()
        rhs_cpu = rhs.detach().cpu()
        clauses: list[list[Tuple[torch.Tensor, torch.Tensor]]] = []
        for idx, size in enumerate(or_spec_size.tolist()):
            size = int(size)
            if size <= 0:
                clauses.append([])
                continue
            C = c_cpu[idx, :size].clone()
            rhs_vec = rhs_cpu[idx, :size].clone()
            clauses.append([(C, rhs_vec)])
        return cls.build_from_input_bounds(lower, upper, clauses)

    @classmethod
    def build_spec(
        cls,
        *,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
        clauses: Optional[Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        center: Optional[torch.Tensor] = None,
        epsilon: Optional[Union[float, torch.Tensor]] = None,
        input_vars: Optional["VariableVector"] = None,
        output_vars: Optional["VariableVector"] = None,
        input_constraint: Optional["Predicate"] = None,
        output_constraint: Optional["Predicate"] = None,
        vnnlib_path: Optional[str] = None,
        input_shape: Optional[Sequence[int]] = None,
    ) -> "VerificationSpec":
        """
        Unified builder for common spec constructions.

        Supported modes (mutually exclusive):
        - bounds: provide lower/upper/clauses
        - center box: provide center/epsilon/clauses
        - expression DSL: provide input_vars/output_vars/input_constraint/output_constraint
        - vnnlib: provide vnnlib_path (and optional input_shape)

        """
        if vnnlib_path is not None:
            return cls.build_from_vnnlib(vnnlib_path, input_shape=input_shape)

        # Basic consistency checks to catch mixing modes.
        if (lower is not None) ^ (upper is not None):
            raise ValueError("lower and upper must be provided together.")
        if (center is not None) ^ (epsilon is not None):
            raise ValueError("center and epsilon must be provided together.")
        if clauses is not None and not (
            (lower is not None and upper is not None) or (center is not None and epsilon is not None)
        ):
            raise ValueError("clauses must pair with either (lower, upper) or (center, epsilon).")
        if (lower is not None or upper is not None) and clauses is None:
            raise ValueError("clauses are required when lower/upper are provided.")

        has_bounds = lower is not None and upper is not None and clauses is not None
        has_center = center is not None and epsilon is not None and clauses is not None
        has_expr = (
            input_vars is not None
            and output_vars is not None
            and input_constraint is not None
            and output_constraint is not None
        )
        modes = [has_bounds, has_center, has_expr]
        if sum(modes) != 1:
            raise ValueError(
                "Specify exactly one mode: "
                "(lower, upper, clauses) or (center, epsilon, clauses) or "
                "(input_vars, output_vars, input_constraint, output_constraint) "
                "or vnnlib_path."
            )

        if has_bounds:
            return cls.build_from_input_bounds(lower, upper, clauses)  # type: ignore[arg-type]
        if has_center:
            return cls.build_from_center(center, epsilon, clauses)  # type: ignore[arg-type]
        _assert_strict_output(output_constraint)  # type: ignore[arg-type]
        return cls.build_from_expressions(  # type: ignore[arg-type]
            input_vars=input_vars,
            output_vars=output_vars,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )

@dataclass
class SolveResult:
    status: str
    success: bool
    reference: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "reference": self.reference,
            "stats": self.stats,
        }


@dataclass
class CounterexampleCheck:
    valid: bool
    inside_bounds: bool
    specification_violated: bool
    margin: Optional[float] = None


class _ApiLogger:
    def __init__(self, timeout: float) -> None:
        self.timeout_threshold = timeout
        self.start_time: Optional[float] = None
        self.bab_ret: list = []
        self.pgd_stats: Dict[int, Dict[str, Any]] = {}
        self.summary: Optional[Tuple[str, float]] = None

    def update_timeout(self, timeout: float) -> None:
        self.timeout_threshold = timeout

    def record_start_time(self) -> None:
        self.start_time = time.time()

    def record_pgd_stats(self, idx: int, stats: Dict[str, Any]) -> None:
        self.pgd_stats[idx] = stats

    def summarize_results(self, status: str, idx: int) -> None:
        if self.start_time is None:
            raise RuntimeError("Logger start time not recorded before summarizing results.")
        elapsed = time.time() - self.start_time
        self.summary = (status, elapsed)

    def finish(self) -> None:
        return


@contextlib.contextmanager
def _config_context(config: Mapping[str, Any]):
    _ensure_config_defaults()
    backup = _clone_config(arguments.Config.all_args)
    backup_file = arguments.Config.file
    new_cfg = _clone_config(arguments.Config.all_args)
    _deep_update(new_cfg, config)
    arguments.Config.all_args = new_cfg
    arguments.Config.update_arguments()
    try:
        yield
    finally:
        arguments.Config.all_args = backup
        arguments.Config.file = backup_file
        arguments.Config.update_arguments()


class ABCrownSolver:
    def __init__(
        self,
        spec: Union[VerificationSpec, Mapping[str, Any]],
        computing_graph: torch.nn.Module,
        *,
        config: Optional[Mapping[str, Any]] = None,
        name: str = "instance",
    ) -> None:
        if spec is None:
            raise ValueError("spec must be provided.")
        if computing_graph is None:
            raise ValueError("computing_graph must be provided.")
        if config is None:
            cfg_source = _DEFAULT_CONFIG
        elif isinstance(config, ConfigBuilder) or hasattr(config, "to_dict"):
            cfg_source = cast(Any, config).to_dict()
        else:
            cfg_source = config
        self.config = _clone_config(cfg_source)
        self.spec = self._normalize_spec(spec)
        self.computing_graph = computing_graph
        self.name = name
        self.logger: Optional[_ApiLogger] = None
        self.vnnlib_handler: Optional[vnnlibHandler] = None
        self.spec_handler_incomplete: Optional[SpecHandler] = None
        self._model: Optional[torch.nn.Module] = None
        self._last_result: Optional[SolveResult] = None

    def solve(self, interm_bounds: Optional[Dict[str, Any]] = None,
              return_reference: bool = True) -> SolveResult:
        with _config_context(self.config):
            result = self._solve_impl(interm_bounds=interm_bounds, return_reference=return_reference)
        self._last_result = result
        return result

    def _solve_impl(self, interm_bounds: Optional[Dict[str, Any]], return_reference: bool) -> SolveResult:
        general_args = arguments.Config['general']
        bab_args = arguments.Config['bab']
        cut_enabled = bab_args['cut']['enabled']
        debug_args = arguments.Config['debug']

        timeout_threshold = float(bab_args['timeout'])
        if bab_args['timeout_scale'] != 1:
            timeout_threshold *= bab_args['timeout_scale']
        if bab_args['override_timeout'] is not None:
            timeout_threshold = float(bab_args['override_timeout'])

        self.logger = _ApiLogger(timeout=timeout_threshold)
        self.logger.record_start_time()

        device = general_args['device']
        self._prepare_environment(device)

        model_ori = self._prepare_model(device)

        self.vnnlib_handler = self._build_vnnlib_handler(self.spec)
        x = self.vnnlib_handler.x[0:1].to(device)
        data_min = self.vnnlib_handler.data_min[0:1].to(device)
        data_max = self.vnnlib_handler.data_max[0:1].to(device)
        if general_args['adhoc_tuning']:
            eval(general_args['adhoc_tuning'])(model_ori, self.vnnlib_handler)

        complete_verifier = general_args['complete_verifier']
        enable_incomplete = general_args['enable_incomplete_verification']
        bab_attack_enabled = arguments.Config['bab']['attack']['enabled']
        if bab_attack_enabled:
            raise AssertionError('BaB attack is not yet supported in the new API.')

        if general_args['complete_verifier'] == 'auto':
            use_input_split = (np.prod(np.array(self.vnnlib_handler.input_shape[1:]))
                               <= bab_args['branching']['input_split']['input_dim_threshold'])
            if use_input_split:
                complete_verifier = 'input_bab'
            else:
                conv_keywords = ['Conv1d', 'Conv2d', 'ConvTranspose2d']
                model_has_conv = any(type(m).__name__ in conv_keywords for m in model_ori.modules())
                complete_verifier = 'bab-refine' if not model_has_conv else 'bab'

            bab_args['branching']['input_split']['enable'] = use_input_split
            bab_args['branching']['method'] = 'sb' if use_input_split else 'kfsb'
            arguments.Config['solver']['bound_prop_method'] = 'crown' if use_input_split else 'alpha-crown'
            bab_args['cut']['enabled'] = cut_enabled and complete_verifier == 'bab'
            arguments.Config['general']['conv_mode'] = 'matrix' if bab_args['cut']['enabled'] else 'patches'
            if complete_verifier == 'bab-refine':
                clip_cfg = arguments.Config['bab']['clip_n_verify']
                clip_cfg['clip_input_domain']['enabled'] = False
                clip_cfg['clip_interm_domain']['enabled'] = False

        use_temp_cuts_path = (arguments.Config['bab']['cut']['cplex_cuts']
                              and bab_args['cut']['cuts_path'] is None)
        temp_cuts_folder = None
        if use_temp_cuts_path:
            temp_cuts_folder = tempfile.TemporaryDirectory(prefix='abcrown_cuts_', dir='/tmp')
            bab_args['cut']['cuts_path'] = temp_cuts_folder.name

        rhs_offset_init = arguments.Config['specification']['rhs_offset']
        if rhs_offset_init is not None and not debug_args['sanity_check']:
            self.vnnlib_handler.add_rhs_offset(rhs_offset_init)

        verified_status, verified_success = 'unknown', False
        attack_examples = attack_margins = all_adv_candidates = None

        if arguments.Config['attack']['pgd_order'] != 'skip':
            reset_attack_stats()

        if arguments.Config['attack']['pgd_order'] == 'before':
            verified_status, verified_success, attack_examples, attack_margins, all_adv_candidates = (
                self._attack(model_ori, verified_status, verified_success)
            )
            get_attack_stats(self.logger, 0)
            if debug_args['sanity_check']:
                rhs_offset = attack_margins if debug_args['sanity_check'] == 'Full' else attack_margins.min()
                self.vnnlib_handler.add_rhs_offset(rhs_offset)
                arguments.Config['attack']['pgd_order'] = 'skip'
                verified_status, verified_success = 'unknown', False

        model_incomplete = None
        reference = {}

        if debug_args['test_optimized_bounds']:
            compare_optimized_bounds_against_lp_bounds(
                model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=self.vnnlib_handler.vnnlib
            )

        if not verified_success and enable_incomplete:
            verified_status, reference = incomplete_verifier_core(self, model_ori, interm_bounds)
            if self.spec_handler_incomplete is not None:
                attack_examples, attack_margins, all_adv_candidates = (
                    self.spec_handler_incomplete.prune_attack_ret(
                        attack_examples, attack_margins, all_adv_candidates
                    )
                )
            if general_args['return_optimized_model']:
                return SolveResult(status=verified_status, success=verified_status != 'unknown')
            verified_success = verified_status != 'unknown'
            model_incomplete = reference.get('model', None)

        if not verified_success and arguments.Config['attack']['pgd_order'] == 'after':
            verified_status, verified_success, attack_examples, attack_margins, all_adv_candidates = (
                self._attack(model_ori, verified_status, verified_success)
            )
            get_attack_stats(self.logger, 0)

        if not verified_success and complete_verifier in ['bab-refine', 'mip']:
            mip_skip_unsafe = arguments.Config['solver']['mip']['skip_unsafe']
            if self.spec_handler_incomplete is not None:
                self.spec_handler_incomplete.adhoc_process_for_mip(reference)
            verified_status, ret_mip = mip(
                model_incomplete, reference, self.vnnlib_handler,
                mip_skip_unsafe=mip_skip_unsafe,
                pgd_attack_example=[attack_examples, attack_margins],
                verifier=complete_verifier
            )
            verified_success = verified_status != 'unknown'
            reference.update(ret_mip)
            if self.spec_handler_incomplete is not None:
                self.spec_handler_incomplete.adhoc_post_process_for_mip(reference)

        if (not verified_success
                and complete_verifier != 'skip'
                and verified_status != 'unknown-mip'):
            if enable_incomplete and self.spec_handler_incomplete is not None:
                self.spec_handler_incomplete.expand_intermediate(reference)
            if arguments.Config['bab']['attack']['enabled']:
                reference['attack_examples'] = all_adv_candidates
                reference['attack_margins'] = attack_margins

            remaining_timeout = timeout_threshold - (time.time() - self.logger.start_time)
            verified_status = complete_verifier_core(
                self,
                model_ori,
                0,
                timeout_threshold=remaining_timeout,
                bab_ret=self.logger.bab_ret,
                reference_dict=reference,
            )

        if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                and model_incomplete is not None):
            terminate_mip_processes(
                model_incomplete.mip_building_proc,
                getattr(model_incomplete, 'processes', None)
            )
            if hasattr(model_incomplete, 'processes'):
                del model_incomplete.processes

        if temp_cuts_folder is not None:
            temp_cuts_folder.cleanup()
            bab_args['cut']['cuts_path'] = None

        if debug_args['sanity_check']:
            if 'unknown' not in verified_status:
                raise AssertionError('Sanity check failed: status should remain unknown.')

        self.logger.summarize_results(verified_status, 0)
        self.logger.finish()

        stats = {
            'elapsed': None if self.logger.summary is None else self.logger.summary[1],
            'pgd': self.logger.pgd_stats.get(0),
            'bab': self.logger.bab_ret,
            'attack_examples': attack_examples,
            'attack_margins': attack_margins,
            'all_adv_candidates': all_adv_candidates,
        }

        return SolveResult(status=verified_status, success=verified_status != 'unknown',
                           reference=reference if return_reference else {}, stats=stats)

    def bab(self, *args: Any, **kwargs: Any) -> Any:
        return bab_core(self, *args, **kwargs)

    def _prepare_environment(self, device: str) -> None:
        general_args = arguments.Config['general']
        seed = general_args['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_printoptions(precision=8)
        has_cudnn = hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available()
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            if has_cudnn:
                torch.backends.cudnn.allow_tf32 = False
        if general_args['deterministic']:
            torch.use_deterministic_algorithms(True)
            if has_cudnn:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.allow_tf32 = False
        if general_args['double_fp']:
            torch.set_default_dtype(torch.float64)
        if general_args['precompile_jit']:
            precompile_jit_kernels()
        if general_args['reset_seed_after_precompile']:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _build_vnnlib_handler(self, spec: VerificationSpec) -> vnnlibHandler:
        vnnlib = spec.to_vnnlib()
        return vnnlibHandler(vnnlib, spec.input_shape)

    def _prepare_model(self, device: str) -> torch.nn.Module:
        graph = self.computing_graph
        model: Optional[torch.nn.Module] = None

        if isinstance(graph, LiRPANet):
            model = graph.model_ori
        elif isinstance(graph, torch.nn.Module):
            model = graph
        elif isinstance(graph, Mapping):
            maybe_model = graph.get("model")
            state_dict = graph.get("state_dict")
            if isinstance(maybe_model, torch.nn.Module) and state_dict is not None:
                maybe_model.load_state_dict(state_dict)
                model = maybe_model
            else:
                raise TypeError(
                    "Unsupported mapping for computing_graph; expected {'model': nn.Module, 'state_dict': ...}."
                )
        elif isinstance(graph, str) and graph.lower().endswith((".onnx", ".onnx.gz")):
            model, onnx_shape = load_model_onnx(graph)
            arguments.Config['model']['input_shape'] = onnx_shape
            self.config.setdefault('model', {})['input_shape'] = onnx_shape
            graph = model
            try:
                self.spec.reshape_input(onnx_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Specification input shape {self.spec.input_shape[1:]} is incompatible with "
                    f"ONNX model expected shape {onnx_shape}."
                ) from exc
        else:
            raise TypeError(f"Unsupported computing graph type: {type(graph).__name__}")

        assert model is not None
        model = model.to(device)
        model.eval()
        if isinstance(self.computing_graph, str) and self.computing_graph.lower().endswith((".onnx", ".onnx.gz")):
            self.computing_graph = model
        self._model = model
        return model

    def _normalize_spec(self, spec: Union[VerificationSpec, Mapping[str, Any]]) -> VerificationSpec:
        if isinstance(spec, VerificationSpec):
            return spec
        if isinstance(spec, Mapping):
            if {'lower', 'upper', 'clauses'}.issubset(spec.keys()):
                return VerificationSpec.build_from_input_bounds(spec['lower'], spec['upper'], spec['clauses'])
            if {'center', 'epsilon', 'clauses'}.issubset(spec.keys()):
                center = spec['center']
                epsilon = spec['epsilon']
                clauses = spec['clauses']
                center_t = torch.as_tensor(center).float()
                eps_t = torch.as_tensor(epsilon).float()
                if eps_t.ndim == 0:
                    eps_t = torch.full_like(center_t, float(eps_t))
                lower = center_t - eps_t
                upper = center_t + eps_t
                return VerificationSpec.build_from_input_bounds(lower.unsqueeze(0), upper.unsqueeze(0), clauses)
        raise TypeError('Unsupported specification format.')

    def _attack(self,
                model_ori: torch.nn.Module,
                verified_status: str,
                verified_success: bool):
        if arguments.Config['model']['with_jacobian']:
            model = LiRPANet(model_ori, in_size=[1, *self.vnnlib_handler.input_shape[1:]]).net
        else:
            model = model_ori
        device = arguments.Config['general']['device']
        x, c, rhs, or_spec_size, _, _ = self.vnnlib_handler.all_specs.get(device)
        try:
            return attack(model, x, c, rhs, or_spec_size, self.vnnlib_handler.vnnlib,
                          verified_status, verified_success)
        except NotImplementedError:
            # Some models (e.g., with Jacobian operations) may include custom autograd
            # Functions without backward; skip PGD instead of crashing.
            print("[warn] PGD attack skipped due to missing autograd backward; continuing without attack.")
            return verified_status, verified_success, None, None, None

    def _spec_violation(self, flat_input: torch.Tensor, output: torch.Tensor) -> Tuple[bool, Optional[float]]:
        best_margin: Optional[float] = None
        for input_box, spec_list in self.vnnlib_handler.vnnlib:
            lb = torch.tensor([item[0] for item in input_box], dtype=flat_input.dtype, device=flat_input.device)
            ub = torch.tensor([item[1] for item in input_box], dtype=flat_input.dtype, device=flat_input.device)
            if not torch.all(flat_input >= lb) or not torch.all(flat_input <= ub):
                continue
            for c_np, rhs_np in spec_list:
                c = torch.tensor(c_np, dtype=output.dtype, device=output.device)
                rhs = torch.tensor(rhs_np, dtype=output.dtype, device=output.device)
                values = c.matmul(output)
                if torch.all(values <= rhs):
                    margin = float((values - rhs).max().item())
                    if best_margin is None or margin > best_margin:
                        best_margin = margin
        return (best_margin is not None), best_margin
