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
from utils import auto_enable_jacobian_mode

__all__ = [
    "ABCrownSolver",
    "IOConstraints",
    "VerificationSpec",
    "BoundsResult",
    "OptimizeResult",
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
# Default per-subproblem BaB timeout used by compute_bounds when the user keeps
# the project-wide 360s default.
_COMPUTE_BOUNDS_DEFAULT_TIMEOUT = 30.0


def _ensure_config_defaults() -> None:
    """Load default values into arguments.Config if needed."""
    if not getattr(arguments.Config, "all_args", None):
        arguments.Config.all_args = {}
    if len(arguments.Config.all_args) == 0:
        arguments.Config.construct_config_dict(arguments.Config.default_args)
        arguments.Config.update_arguments()


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    """Recursively merge a nested mapping into another mapping."""
    for key, value in updates.items():
        if isinstance(value, Mapping):
            node = base.setdefault(key, {})
            if not isinstance(node, MutableMapping):
                raise TypeError(f"Cannot merge mapping into non-mapping at {key}.")
            _deep_update(node, value)
        else:
            base[key] = value


def _clone_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Make a deep copy of a configuration dictionary."""
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
    """Create nested dict keys along a path and set the final value."""
    node: MutableMapping[str, Any] = target
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, MutableMapping):
            raise TypeError(f"Cannot assign into non-mapping at {'.'.join(path)}.")
        node = child
    node[path[-1]] = value


_UNSET = object()


def _split_config_path(key: str) -> Sequence[str]:
    """Split config key path supporting '/' (preferred) and '__' (legacy)."""
    if "/" in key:
        path = [part for part in key.split("/") if part]
    elif "__" in key:
        path = [part for part in key.split("__") if part]
    else:
        path = [key]
    if not path:
        raise ValueError("Config path cannot be empty.")
    return path


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
        """Initialize builder from defaults or a provided config snapshot."""
        _ensure_config_defaults()
        if base is None:
            base = _DEFAULT_CONFIG
        self._cfg = _clone_config(base)

    @classmethod
    def from_defaults(cls) -> "ConfigBuilder":
        """Create a builder starting from global defaults."""
        return cls(_DEFAULT_CONFIG)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ConfigBuilder":
        """Create a builder from an existing config mapping."""
        return cls(config)

    def update(self,
               *modifiers: Union[Mapping[str, Any], Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
               **overrides: Any) -> "ConfigBuilder":
        """Apply modifier callables or dict merges plus keyword overrides."""
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

    def set(self, key: Any = _UNSET, value: Any = _UNSET, **overrides: Any) -> "ConfigBuilder":
        """
        Set config entries with path-style keys.

        Preferred usage:
            .set("bab/timeout", 30)

        Also supports legacy keyword style for compatibility:
            .set(bab__timeout=30)
        """
        cfg = _clone_config(self._cfg)

        if key is not _UNSET:
            if isinstance(key, Mapping):
                if value is not _UNSET:
                    raise TypeError("When key is a mapping, do not provide a separate value.")
                _deep_update(cfg, cast(Mapping[str, Any], key))
            elif isinstance(key, str):
                if value is _UNSET:
                    raise TypeError("set(path, value) requires both path and value.")
                _assign_path(cfg, _split_config_path(key), value)
            else:
                raise TypeError("set() expects a path string or mapping as its first argument.")
            if overrides:
                raise TypeError("Do not mix positional set(path, value) with keyword overrides in one call.")

        if overrides:
            nested_overrides: Dict[str, Any] = {}
            for override_key, override_value in overrides.items():
                _assign_path(nested_overrides, _split_config_path(str(override_key)), override_value)
            _deep_update(cfg, nested_overrides)

        if key is _UNSET and not overrides:
            raise TypeError("set() expects either set(path, value), set(mapping), or keyword overrides.")

        self._cfg = cfg
        return self

    def replace(self, config: Mapping[str, Any]) -> "ConfigBuilder":
        """Replace stored config with a deep copy of another mapping."""
        self._cfg = _clone_config(config)
        return self

    def copy(self) -> "ConfigBuilder":
        """Return a new builder that holds a copied config."""
        return ConfigBuilder(self._cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copied config dict."""
        return _clone_config(self._cfg)

    def __call__(self) -> Dict[str, Any]:
        """Alias for to_dict()."""
        return self.to_dict()

    @classmethod
    def from_yaml(cls, path: str) -> "ConfigBuilder":
        """Load a YAML config file into a builder."""
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
        """Store config path and optional root override; entries load lazily."""
        self.config_path = os.path.abspath(config_path)
        self._root_override = root
        self._config: Optional[Dict[str, Any]] = None
        self._entries: Optional[Tuple[VNNCompInstance, ...]] = None

    def _ensure_loaded(self) -> None:
        """Parse the YAML and CSV once and cache config and entries."""
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
    ) -> Tuple["IOConstraints", Union[str, torch.nn.Module], Dict[str, Any], VNNCompInstance]:
        """
        Prepare a single instance, returning `(constraints, onnx_path, config, metadata)`.
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
        constraints = IOConstraints(vnnlib_path=entry.vnnlib_path)
        return constraints, entry.onnx_path, self.config, entry

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
    """Resolve the benchmark root directory from config, override, or env hints."""
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
    """Read config and CSV to produce per-instance metadata and a resolved config."""
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
) -> Tuple["IOConstraints", Union[str, torch.nn.Module], Dict[str, Any], VNNCompInstance]:
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
    """Run solver over a list of VNN-COMP entries and collect results."""
    results: list[Tuple[int, VNNCompInstance, SolveResult]] = []
    for entry in entries:
        if not os.path.exists(entry.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {entry.onnx_path}")
        if not os.path.exists(entry.vnnlib_path):
            raise FileNotFoundError(f"vnnlib spec not found: {entry.vnnlib_path}")
        constraints = IOConstraints(vnnlib_path=entry.vnnlib_path)
        solver = ABCrownSolver(
            entry.onnx_path,
            constraint=constraints,
            config=_clone_config(config),
            name=f"vnncomp/{entry.index}",
        )
        result = solver.verify()
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
        """Record the variable kind (input/output) and flattened size."""
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
        """Convert multi-dimensional index into a flat position."""
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
        """Return a one-hot LinearExpr for a given index."""
        flat_idx = self._flatten_index(key)
        return LinearExpr({(self.kind, flat_idx): 1.0}, 0.0)

    def _iter_other(self, other: Union[Sequence[float], np.ndarray, torch.Tensor, float, int]) -> Sequence[float]:
        """Flatten and size-check comparison bounds."""
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
        """Build a predicate comparing each element against a bound."""
        if op not in (">=", "<=", ">", "<"):
            raise ValueError(f"Unsupported comparison operator {op}.")
        if op in (">", "<"):
            # Map strict inequalities to non-strict ones with an epsilon shift on the bound.
            delta = _STRICT_INEQUALITY_EPS if op == ">" else -_STRICT_INEQUALITY_EPS
            adjusted = _shift_other_by_eps(other, delta)
            mapped_op = ">=" if op == ">" else "<="
            return self._compare(adjusted, mapped_op)
        bounds = self._iter_other(other)
        atoms: list[Predicate] = []
        for idx, bound in enumerate(bounds):
            atom = (self[idx] >= bound) if op == ">=" else (self[idx] <= bound)
            atoms.append(atom)
        if not atoms:
            raise ValueError("Empty comparison on VariableVector.")
        # Build a balanced conjunction tree to avoid deep recursion.
        while len(atoms) > 1:
            next_level: list[Predicate] = []
            it = iter(atoms)
            for left in it:
                right = next(it, None)
                if right is None:
                    next_level.append(left)
                else:
                    next_level.append(left & right)
            atoms = next_level
        return atoms[0]

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
    """Coerce scalars into LinearExpr for arithmetic helpers."""
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
    """Verify a value is a Predicate instance."""
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
    """Flip a comparison predicate to its logical negation."""
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


def _predicate_to_dnf(
    pred: Predicate,
    *,
    negate: bool = False,
    force_simplify: Optional[bool] = None,
    print_original: bool = False,
) -> Sequence[Sequence[ComparisonPredicate]]:
    """Convert a predicate tree into disjunctive normal form using sympy."""
    symbol_map: Dict[sympy.Symbol, ComparisonPredicate] = {}
    predicate_map: Dict[Tuple[Tuple[Tuple[str, int], float], float], sympy.Symbol] = {}

    def _format_linear_expr(expr: LinearExpr) -> str:
        """Render a LinearExpr into a readable string for printing specs."""
        if not expr.coeffs:
            return f"{expr.constant:.6g}"
        items = []
        for (kind, idx), coeff in sorted(expr.coeffs.items()):
            var = "x" if kind == "input" else "y"
            items.append((coeff, f"{var}[{idx}]"))
        out = ""
        for i, (coeff, label) in enumerate(items):
            sign = "-" if coeff < 0 else "+"
            abs_coeff = abs(coeff)
            if abs_coeff == 1:
                term = label
            else:
                term = f"{abs_coeff:.6g}*{label}"
            if i == 0:
                out = term if coeff >= 0 else f"-{term}"
            else:
                out = f"{out} {sign} {term}"
        if expr.constant != 0:
            const_sign = "-" if expr.constant < 0 else "+"
            out = f"{out} {const_sign} {abs(expr.constant):.6g}"
        return out

    def _format_predicate(pred: ComparisonPredicate) -> str:
        return f"{_format_linear_expr(pred.lhs)} {pred.op} {_format_linear_expr(pred.rhs)}"

    def _predicate_key(pred: ComparisonPredicate) -> Tuple[Tuple[Tuple[str, int], float], float]:
        expr = pred.normalized_expr()
        coeff_items = tuple(sorted(((kind, idx), float(coeff)) for (kind, idx), coeff in expr.coeffs.items()))
        constant = 0.0 if expr.constant == 0 else float(expr.constant)
        return coeff_items, constant

    def to_sympy(node: Predicate) -> sympy.Expr:
        if isinstance(node, ComparisonPredicate):
            key = _predicate_key(node)
            symbol = predicate_map.get(key)
            if symbol is None:
                symbol = sympy.Symbol(f"p{len(symbol_map)}", boolean=True)
                predicate_map[key] = symbol
                symbol_map[symbol] = node
            return symbol
        if isinstance(node, AndPredicate):
            return sympy.And(to_sympy(node.left), to_sympy(node.right))
        if isinstance(node, OrPredicate):
            return sympy.Or(to_sympy(node.left), to_sympy(node.right))
        raise TypeError(f"Unsupported predicate type {type(node).__name__}")

    sympy_expr_original = to_sympy(pred)
    sympy_expr = sympy_expr_original
    if negate:
        sympy_expr = sympy.Not(sympy_expr)
    num_symbols = len(sympy_expr.free_symbols)
    if negate:
        print_original = True
    if force_simplify is None:
        use_simplify = num_symbols <= 10
        use_force = use_simplify
    else:
        use_simplify = force_simplify
        use_force = force_simplify
    if use_simplify:
        dnf_expr = sympy.to_dnf(sympy_expr, simplify=True, force=use_force)
    else:
        dnf_expr = sympy.to_dnf(sympy_expr, simplify=False)
    replacements = {
        str(symbol): f"({_format_predicate(symbol_map[symbol])})"
        for symbol in symbol_map
    }

    def _render_expr(expr: sympy.Expr) -> str:
        text = str(expr)
        for key in sorted(replacements, key=len, reverse=True):
            text = text.replace(key, replacements[key])
        return text

    def _dnf_signature(expr: sympy.Expr) -> Tuple[Tuple[Tuple[str, bool], ...], ...]:
        """
        SymPy to_dnf(simplify=True) canonicalizes commutative args via
        default_sort_key (e.g., a & b vs b & a), so normalize clauses to
        avoid treating pure reordering as a simplification change.
        https://docs.sympy.org/latest/modules/core.html#sympy.core.sorting.default_sort_key
        """
        def _collect(node: sympy.Expr) -> list[list[Tuple[str, bool]]]:
            if node is sympy.true:
                return [[]]
            if node is sympy.false:
                return []
            if node.is_Symbol:
                return [[(str(node), False)]]
            if node.func is sympy.Not and node.args[0].is_Symbol:
                return [[(str(node.args[0]), True)]]
            if node.func is sympy.Or:
                clauses: list[list[Tuple[str, bool]]] = []
                for arg in node.args:
                    clauses.extend(_collect(arg))
                return clauses
            if node.func is sympy.And:
                clause: list[Tuple[str, bool]] = []
                for arg in node.args:
                    if arg is sympy.true:
                        continue
                    if arg is sympy.false:
                        return []
                    if arg.is_Symbol:
                        clause.append((str(arg), False))
                    elif arg.func is sympy.Not and arg.args[0].is_Symbol:
                        clause.append((str(arg.args[0]), True))
                    else:
                        clause.append((str(arg), False))
                return [clause]
            return [[(str(node), False)]]

        clauses = _collect(expr)
        normalized = [tuple(sorted(clause)) for clause in clauses]
        normalized.sort()
        return tuple(normalized)

    if print_original:
        spec_expr = sympy_expr_original if negate else sympy_expr
        original_dnf = sympy.to_dnf(spec_expr, simplify=False)
        if use_simplify:
            simplified_dnf = sympy.to_dnf(spec_expr, simplify=True, force=True)
            if _dnf_signature(simplified_dnf) == _dnf_signature(original_dnf):
                print(f"Specification DNF: {_render_expr(original_dnf)}")
            else:
                print(f"Specification DNF (original): {_render_expr(original_dnf)}")
                print(f"Specification DNF (simplified): {_render_expr(simplified_dnf)}")
        else:
            print(f"Specification DNF: {_render_expr(original_dnf)}")

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
    """Split a clause into matrix rows for one variable kind and leftover predicates."""
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


def _parse_input_bounds(
    predicate: Predicate,
    inputs: VariableVector,
    *,
    force_simplify: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a conjunction of input predicates into lower/upper bound tensors."""
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
        clauses = _predicate_to_dnf(predicate, force_simplify=force_simplify)
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
    # Use the default dtype from env variables. instead of torch.float32.
    # This is because solving mode uses float64.
    dtype = torch.get_default_dtype()
    lower_tensor = torch.tensor(lower, dtype=dtype).view((1, *inputs.shape))
    upper_tensor = torch.tensor(upper, dtype=dtype).view((1, *inputs.shape))
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
            """Standardize tensors and validate shape for bounds."""
            # Use the default dtype from env variables. instead of torch.float32.
            # This is because solving mode uses float64.
            dtype = torch.get_default_dtype()
            self.lower = torch.as_tensor(self.lower).detach().clone().to(dtype=dtype)
            self.upper = torch.as_tensor(self.upper).detach().clone().to(dtype=dtype)
            if self.lower.shape != self.upper.shape:
                raise ValueError("Lower and upper bounds must share the same shape.")
            if self.lower.ndim < 2:
                raise ValueError("Input bounds must include batch and data dimensions.")

        @property
        def num_inputs(self) -> int:
            """Number of input samples in the batch."""
            return self.lower.shape[0]

        @property
        def data_shape(self) -> Tuple[int, ...]:
            """Shape of a single input example."""
            return tuple(self.lower.shape[1:])

        def reshape(self, target_shape: Sequence[int]) -> None:
            """Reshape bounds while preserving flattened size."""
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
        num_outputs: Optional[int] = None

        def __post_init__(self) -> None:
            """Validate and normalize clause storage."""
            if not isinstance(self.clauses, Sequence) or len(self.clauses) == 0:
                raise ValueError("At least one specification clause is required.")
            self.clauses = list(self.clauses)
            inferred = self._infer_num_outputs(self.clauses)
            if self.num_outputs is None:
                self.num_outputs = inferred
            elif inferred is not None and self.num_outputs != inferred:
                raise ValueError(
                    f"Conflicting output dimensions: expected {self.num_outputs}, got {inferred}."
                )

        @classmethod
        def _infer_num_outputs(cls, clauses: Any) -> Optional[int]:
            """Infer output dimension from the first non-empty clause."""
            if isinstance(clauses, tuple) and len(clauses) == 2:
                c_tensor = torch.as_tensor(clauses[0])
                if c_tensor.ndim == 1:
                    return int(c_tensor.shape[0])
                if c_tensor.ndim >= 2:
                    return int(c_tensor.shape[-1])
                return None
            if not isinstance(clauses, Sequence):
                return None
            for item in clauses:
                inferred = cls._infer_num_outputs(item)
                if inferred is not None:
                    return inferred
            return None

        @staticmethod
        def _wrap_clause(clause: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            """Convert a raw clause into float tensors with expected dims."""
            if not isinstance(clause, Sequence) or len(clause) != 2:
                raise ValueError("Each clause must be a (C, rhs) pair.")
            c, rhs = clause
            # Use the default dtype from env variables. instead of torch.float32.
            # This is because solving mode uses float64.
            dtype = torch.get_default_dtype()
            c_tensor = torch.as_tensor(c).detach().clone().to(dtype=dtype)
            rhs_tensor = torch.as_tensor(rhs).detach().clone().to(dtype=dtype)
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
            """Expand or wrap clauses so each input has an OR-of-AND list."""
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
        """Number of inputs represented in this spec."""
        return self.input_spec.num_inputs

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape tuple with batch placeholder for model input."""
        return (-1, *self.input_spec.data_shape)

    def __post_init__(self) -> None:
        """Ensure input/output specs are the right types and normalized."""
        if not isinstance(self.input_spec, VerificationSpec.InputSpec):
            raise TypeError("input_spec must be an instance of VerificationSpec.InputSpec.")
        if not isinstance(self.output_spec, VerificationSpec.OutputSpec):
            raise TypeError("output_spec must be an instance of VerificationSpec.OutputSpec.")
        self.output_spec.normalize(self.input_spec.num_inputs)

    @property
    def lower(self) -> torch.Tensor:
        """Lower bounds tensor for inputs."""
        return self.input_spec.lower

    @property
    def upper(self) -> torch.Tensor:
        """Upper bounds tensor for inputs."""
        return self.input_spec.upper

    @property
    def clauses(self) -> Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        return self.output_spec.clauses

    def reshape_input(self, target_shape: Sequence[int]) -> None:
        """Reshape input bounds to the desired data shape."""
        self.input_spec.reshape(target_shape)

    def to_vnnlib(self) -> Sequence[Tuple[Sequence[Tuple[float, float]], Sequence[Tuple[np.ndarray, np.ndarray]]]]:
        """Convert this spec into VNNLIB-compatible tuples."""
        vnn_entries = []
        for idx in range(self.num_inputs):
            lb = self.input_spec.lower[idx].view(-1).cpu().numpy()
            ub = self.input_spec.upper[idx].view(-1).cpu().numpy()
            input_box = list(zip(lb.tolist(), ub.tolist()))
            or_clauses = []
            if len(self.output_spec.clauses[idx]) == 0 and self.output_spec.num_outputs is not None:
                or_clauses.append((
                    np.zeros((0, self.output_spec.num_outputs), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                ))
            else:
                for c_tensor, rhs_tensor in self.output_spec.clauses[idx]:
                    or_clauses.append((c_tensor.cpu().numpy(), rhs_tensor.cpu().numpy()))
            vnn_entries.append((input_box, or_clauses))
        return vnn_entries

    @classmethod
    def build_from_center(cls,
                          center: torch.Tensor,
                          epsilon: Union[float, torch.Tensor],
                          clauses: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
                          ) -> "VerificationSpec":
        """Build a box specification from a center point and epsilon radius."""
        center_t = torch.as_tensor(center).float()
        eps_t = torch.as_tensor(epsilon).float()
        if eps_t.ndim == 0:
            eps_t = torch.full_like(center_t, float(eps_t))
        lower = center_t - eps_t
        upper = center_t + eps_t
        lower_batched = lower.unsqueeze(0) if lower.ndim == center_t.ndim else lower
        upper_batched = upper.unsqueeze(0) if upper.ndim == center_t.ndim else upper
        return cls.build_from_input_bounds(
            lower_batched,
            upper_batched,
            clauses,
        )

    @classmethod
    def build_from_input_bounds(
        cls,
        lower: torch.Tensor,
        upper: torch.Tensor,
        clauses: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> "VerificationSpec":
        """Build a specification from explicit lower/upper bounds and clauses."""
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
        output_constraint: Optional[Predicate] = None,
        force_simplify: Optional[bool] = None,
    ) -> "VerificationSpec":
        """Build a specification from symbolic expressions."""
        input_force_simplify = force_simplify
        output_force_simplify = force_simplify
        if force_simplify is None:
            output_force_simplify = output_vars.size <= 10
        lower, upper = _parse_input_bounds(
            input_constraint,
            input_vars,
            force_simplify=input_force_simplify,
        )
        clauses: list[list[Tuple[torch.Tensor, torch.Tensor]]] = []
        if output_constraint is None:
            clauses.append([])
        else:
            output_clauses = _predicate_to_dnf(
                output_constraint,
                negate=True,
                force_simplify=output_force_simplify,
                print_original=True,
            )
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

        input_spec = cls.InputSpec(
            lower_batched,
            upper_batched,
        )
        output_spec = cls.OutputSpec(clauses)
        return cls(input_spec=input_spec, output_spec=output_spec)

    @classmethod
    def build_from_vnnlib(
        cls,
        path: str,
        input_shape: Optional[Sequence[int]] = None,
    ) -> "VerificationSpec":
        """Load a VNNLIB file and wrap its content as a VerificationSpec."""
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
        with _config_context({"general": {"store_all_specs_on_cpu": True}}):
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
        input_spec = cls.InputSpec(lower, upper)
        output_spec = cls.OutputSpec(clauses, num_outputs=handler.num_output)
        return cls(input_spec=input_spec, output_spec=output_spec)

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
        force_simplify: Optional[bool] = None,
        vnnlib_path: Optional[str] = None,
        input_shape: Optional[Sequence[int]] = None,
    ) -> "VerificationSpec":
        """
        Unified builder for common spec constructions.

        Supported modes (mutually exclusive):
        - bounds: provide lower/upper/clauses
        - center box: provide center/epsilon/clauses
        - expression DSL: provide input_vars/output_vars/input_constraint and optional output_constraint
        - vnnlib: provide vnnlib_path (and optional input_shape)

        Parameters:
        - lower/upper: input bounds (batched tensors) for the bounds mode.
        - clauses: OR-of-AND list of (C, rhs) tuples for output constraints.
        - center/epsilon: center point and L-infinity radius for the center box mode.
        - input_vars/output_vars: symbolic variables for the DSL mode.
        - input_constraint/output_constraint: DSL predicates for input/output constraints (output optional).
        - force_simplify: override DNF simplification (True forces, False skips; None uses auto threshold).
        - vnnlib_path/input_shape: load a VNNLIB property (input_shape required if absent in file).

        """
        if vnnlib_path is not None:
            return cls.build_from_vnnlib(
                vnnlib_path,
                input_shape=input_shape,
            )

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
        )
        modes = [has_bounds, has_center, has_expr]
        if sum(modes) != 1:
            raise ValueError(
                "Specify exactly one mode: "
                "(lower, upper, clauses) or (center, epsilon, clauses) or "
                "(input_vars, output_vars, input_constraint[, output_constraint]) "
                "or vnnlib_path."
            )

        if has_bounds:
            return cls.build_from_input_bounds(  # type: ignore[arg-type]
                lower,
                upper,
                clauses,
            )
        if has_center:
            return cls.build_from_center(  # type: ignore[arg-type]
                center,
                epsilon,
                clauses,
            )
        if output_vars is not None and output_vars.size > 10 and force_simplify is not True:
            print(
                "Simplification skipped: more than 10 output variables detected. "
                "Set force_simplify=True to force simplification."
            )
        if output_constraint is not None:
            _assert_strict_output(output_constraint)
        return cls.build_from_expressions(  # type: ignore[arg-type]
            input_vars=input_vars,
            output_vars=output_vars,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
            force_simplify=force_simplify,
        )


class IOConstraints(VerificationSpec):
    """High-level constraint constructor for verification and bound workflows."""

    def __init__(
        self,
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
        force_simplify: Optional[bool] = None,
        vnnlib_path: Optional[str] = None,
        input_shape: Optional[Sequence[int]] = None,
        input_spec: Optional["VerificationSpec.InputSpec"] = None,
        output_spec: Optional["VerificationSpec.OutputSpec"] = None,
    ) -> None:
        has_raw_specs = input_spec is not None or output_spec is not None
        if has_raw_specs:
            if input_spec is None or output_spec is None:
                raise ValueError("input_spec and output_spec must be provided together.")
            built = VerificationSpec(input_spec=input_spec, output_spec=output_spec)
        elif (
            input_vars is not None
            and input_constraint is not None
            and output_vars is None
            and output_constraint is None
            and lower is None
            and upper is None
            and clauses is None
            and center is None
            and epsilon is None
            and vnnlib_path is None
            and input_shape is None
        ):
            lower_t, upper_t = _parse_input_bounds(
                input_constraint,
                input_vars,
                force_simplify=force_simplify,
            )
            built = VerificationSpec(
                input_spec=VerificationSpec.InputSpec(lower_t, upper_t),
                output_spec=VerificationSpec.OutputSpec([[]]),
            )
        else:
            built = VerificationSpec.build_spec(
                lower=lower,
                upper=upper,
                clauses=clauses,
                center=center,
                epsilon=epsilon,
                input_vars=input_vars,
                output_vars=output_vars,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
                force_simplify=force_simplify,
                vnnlib_path=vnnlib_path,
                input_shape=input_shape,
            )
        self.input_spec = built.input_spec
        self.output_spec = built.output_spec


@dataclass
class SolveResult:
    status: str
    success: bool
    reference: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "status": self.status,
            "success": self.success,
            "reference": self.reference,
            "stats": self.stats,
        }


@dataclass
class BoundsResult:
    lower: torch.Tensor
    upper: torch.Tensor
    success: bool
    stats: Dict[str, Any] = field(default_factory=dict)
    linear_bounds: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "success": self.success,
            "stats": self.stats,
            "linear_bounds": self.linear_bounds,
        }


@dataclass
class OptimizeResult:
    status: str
    success: bool
    primal_value: Optional[float] = None
    x_best: Optional[torch.Tensor] = None
    solver: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    bound_history: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "primal_value": self.primal_value,
            "x_best": self.x_best,
            "solver": self.solver,
            "stats": self.stats,
            "bound_history": self.bound_history,
        }


@dataclass
class CounterexampleCheck:
    valid: bool
    inside_bounds: bool
    specification_violated: bool
    margin: Optional[float] = None


class _ApiLogger:
    def __init__(self, timeout: float) -> None:
        """Track timing and per-phase stats during a verify run."""
        self.timeout_threshold = timeout
        self.start_time: Optional[float] = None
        self.bab_ret: list = []
        self.pgd_stats: Dict[int, Dict[str, Any]] = {}
        self.summary: Optional[Tuple[str, float]] = None

    def update_timeout(self, timeout: float) -> None:
        """Change the timeout budget."""
        self.timeout_threshold = timeout

    def record_start_time(self) -> None:
        """Mark the start timestamp."""
        self.start_time = time.time()

    def record_pgd_stats(self, idx: int, stats: Dict[str, Any]) -> None:
        """Save PGD statistics for a run index."""
        self.pgd_stats[idx] = stats

    def summarize_results(self, status: str, idx: int) -> None:
        """Compute elapsed time and store summary for an index."""
        if self.start_time is None:
            raise RuntimeError("Logger start time not recorded before summarizing results.")
        elapsed = time.time() - self.start_time
        self.summary = (status, elapsed)

    def finish(self) -> None:
        """Placeholder for cleanup hook."""
        return


class _ObjectiveOutputWrapper(torch.nn.Module):
    """Prepend a scalar linear objective as output dimension 0."""

    def __init__(self, base_model: torch.nn.Module, objective_row: torch.Tensor) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("objective_row", objective_row.detach().clone().reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_model(x)
        if base_output.ndim != 2:
            raise ValueError("Objective augmentation expects a 2-D model output tensor.")
        if base_output.shape[1] != int(self.objective_row.numel()):
            raise ValueError(
                "Objective row dimension does not match model output dimension: "
                f"{base_output.shape[1]} vs {int(self.objective_row.numel())}."
            )
        objective_value = (base_output * self.objective_row.unsqueeze(0)).sum(dim=1, keepdim=True)
        return torch.cat([objective_value, base_output], dim=1)

    def get_output_dimension(self) -> int:
        if hasattr(self.base_model, "get_output_dimension"):
            base_dim = int(self.base_model.get_output_dimension())
        elif hasattr(self.base_model, "out_features"):
            base_dim = int(self.base_model.out_features)
        else:
            raise AttributeError(
                "Base model does not expose output dimension; implement get_output_dimension()."
            )
        return base_dim + 1


@contextlib.contextmanager
def _config_context(config: Mapping[str, Any]):
    """Temporarily apply a config to the global arguments.Config."""
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
        *args: Any,
        constraint: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        spec: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        computing_graph: Optional[Union[torch.nn.Module, Mapping[str, Any], str]] = None,
        input_vars: Optional[VariableVector] = None,
        output_vars: Optional[VariableVector] = None,
        config: Optional[Mapping[str, Any]] = None,
        name: str = "instance",
    ) -> None:
        """Store constraints/model graph/config and accept both legacy and new constructor forms."""
        if spec is not None:
            if constraint is not None:
                raise ValueError("Provide only one of constraint or spec.")
            constraint = spec

        if len(args) > 3:
            raise TypeError("ABCrownSolver accepts at most 3 positional arguments.")
        if len(args) == 3:
            if computing_graph is not None or input_vars is not None or output_vars is not None:
                raise TypeError("Duplicate constructor arguments for computing_graph/input_vars/output_vars.")
            computing_graph = cast(Union[torch.nn.Module, Mapping[str, Any], str], args[0])
            input_vars = cast(VariableVector, args[1])
            output_vars = cast(VariableVector, args[2])
        elif len(args) == 2:
            first, second = args
            if self._looks_like_constraint(first):
                if constraint is not None:
                    raise TypeError("Constraint provided both positionally and by keyword.")
                if computing_graph is not None:
                    raise TypeError("computing_graph provided both positionally and by keyword.")
                constraint = cast(Union[VerificationSpec, Mapping[str, Any]], first)
                computing_graph = cast(Union[torch.nn.Module, Mapping[str, Any], str], second)
            else:
                if computing_graph is not None or input_vars is not None:
                    raise TypeError("Duplicate constructor arguments for computing_graph/input_vars.")
                computing_graph = cast(Union[torch.nn.Module, Mapping[str, Any], str], first)
                input_vars = cast(VariableVector, second)
        elif len(args) == 1:
            if computing_graph is not None:
                raise TypeError("computing_graph provided both positionally and by keyword.")
            computing_graph = cast(Union[torch.nn.Module, Mapping[str, Any], str], args[0])

        if computing_graph is None:
            raise ValueError("computing_graph must be provided.")
        if config is None:
            cfg_source = _DEFAULT_CONFIG
        elif isinstance(config, ConfigBuilder) or hasattr(config, "to_dict"):
            cfg_source = cast(Any, config).to_dict()
        else:
            cfg_source = config
        self.config = _clone_config(cfg_source)
        self.constraint = None if constraint is None else self._normalize_constraint(constraint)
        # Backward-compatible alias.
        self.spec = self.constraint
        self.computing_graph = computing_graph
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.name = name
        self.logger: Optional[_ApiLogger] = None
        self.vnnlib_handler: Optional[vnnlibHandler] = None
        self.spec_handler_incomplete: Optional[SpecHandler] = None
        self._model: Optional[torch.nn.Module] = None
        self._last_result: Optional[SolveResult] = None
        self._runtime_spec: Optional[VerificationSpec] = None
        self._return_linear_bounds = False
        self._enable_jacobian_mode_if_needed()

    def _enable_jacobian_mode_if_needed(self) -> None:
        """Automatically enable Jacobian handling when the model uses JacobianOP."""
        auto_enable_jacobian_mode(self.config, self.computing_graph)

    @staticmethod
    def _looks_like_constraint(candidate: Any) -> bool:
        if isinstance(candidate, VerificationSpec):
            return True
        # Any Mapping in the constraint position is treated as a constraint
        # candidate; _normalize_constraint validates the format and raises
        # TypeError for unsupported schemas.
        return isinstance(candidate, Mapping)

    @staticmethod
    def _has_output_constraints(constraints: VerificationSpec) -> bool:
        return any(len(per_input) > 0 for per_input in constraints.clauses)

    @staticmethod
    def _normalize_verify_status(status: str) -> str:
        """Collapse verifier-specific labels into API-facing verification outcomes."""
        if status in {"verified", "falsified", "unknown"}:
            return status
        if status.startswith("safe"):
            return "verified"
        if status.startswith("unsafe"):
            return "falsified"
        if status.startswith("unknown"):
            return "unknown"
        return status

    @staticmethod
    def _extract_lbs_from_verify_result(
        result: SolveResult,
        *,
        expected_rows: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        def _normalize_candidate(value: Any) -> Optional[torch.Tensor]:
            if value is None:
                return None
            tensor = torch.as_tensor(value).detach().clone().float().view(-1)
            if tensor.numel() == 0:
                return None
            if not torch.isfinite(tensor).all():
                return None
            return tensor.cpu()

        candidates: list[torch.Tensor] = []
        # Preferred: per-spec BaB margins passed through complete_verifier.
        bab_global_lb = _normalize_candidate(result.reference.get("bab_global_lb"))
        if bab_global_lb is not None:
            candidates.append(bab_global_lb)
        # Fallback: incomplete/global reference bounds.
        ref_global_lb = _normalize_candidate(result.reference.get("global_lb"))
        if ref_global_lb is not None:
            candidates.append(ref_global_lb)
        # Last fallback: scalar BaB summary from logger stats.
        bab_stats = result.stats.get("bab")
        if isinstance(bab_stats, Sequence) and len(bab_stats) > 0:
            last = bab_stats[-1]
            if isinstance(last, Sequence) and len(last) >= 2:
                try:
                    stat_lb = float(last[1])
                    if np.isfinite(stat_lb):
                        candidates.append(torch.tensor([stat_lb], dtype=torch.float32))
                except (TypeError, ValueError):
                    pass
        if expected_rows is not None:
            candidates = [c for c in candidates if c.numel() == expected_rows]
        if len(candidates) == 0:
            return None
        merged = candidates[0]
        for cand in candidates[1:]:
            merged = torch.maximum(merged, cand)
        return merged

    @staticmethod
    def _extract_lb_from_verify_result(result: SolveResult) -> Optional[float]:
        lbs = ABCrownSolver._extract_lbs_from_verify_result(result, expected_rows=1)
        if lbs is None:
            return None
        return float(lbs[0].item())

    @staticmethod
    def _extract_reference_lbs_from_verify_result(
        result: SolveResult,
        *,
        expected_rows: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        value = result.reference.get("global_lb")
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach().clone().float().view(-1)
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            return None
        tensor = tensor.cpu()
        if expected_rows is not None and tensor.numel() != expected_rows:
            return None
        return tensor

    @staticmethod
    def _extract_linear_bound_matrix(
        result: SolveResult,
        *,
        expected_rows: int,
    ) -> Optional[torch.Tensor]:
        """Extract the returned input-side lA matrix as [spec, flattened_input]."""
        def _find_lA_tensor(value: Any) -> Optional[torch.Tensor]:
            if torch.is_tensor(value):
                return value
            if isinstance(value, Mapping):
                direct = value.get("lA")
                if torch.is_tensor(direct):
                    return direct
                for item in value.values():
                    found = _find_lA_tensor(item)
                    if found is not None:
                        return found
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    found = _find_lA_tensor(item)
                    if found is not None:
                        return found
            return None

        lA_dict = result.reference.get("lA")
        matrix_source = None
        if isinstance(lA_dict, Mapping) and len(lA_dict) > 0:
            matrix_source = _find_lA_tensor(lA_dict)
        if matrix_source is None:
            matrix_source = _find_lA_tensor(result.reference.get("A"))
        if matrix_source is None:
            return None
        matrix = matrix_source.detach().cpu().float()
        if matrix.ndim >= 2 and matrix.shape[0] == 1 and matrix.shape[1] == expected_rows:
            matrix = matrix.squeeze(0)
        elif matrix.ndim >= 2 and matrix.shape[0] != expected_rows and matrix.shape[1] == expected_rows:
            matrix = matrix.transpose(0, 1)
        if matrix.shape[0] != expected_rows:
            return None
        return matrix.reshape(expected_rows, -1).clone()

    @staticmethod
    def _concretize_affine_over_box(
        matrix: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        *,
        use_upper: bool,
    ) -> torch.Tensor:
        """Concretize a flattened affine form over a box input domain."""
        matrix = matrix.detach().cpu().float()
        lower = lower.detach().cpu().float().reshape(-1)
        upper = upper.detach().cpu().float().reshape(-1)
        matrix_pos = matrix.clamp(min=0)
        matrix_neg = matrix.clamp(max=0)
        if use_upper:
            return matrix_pos @ upper + matrix_neg @ lower
        return matrix_pos @ lower + matrix_neg @ upper

    @staticmethod
    def _capture_per_c_subdomains(
        ordered_cs: torch.Tensor,
        cs: Optional[torch.Tensor],
        dm_l: Optional[torch.Tensor],
        dm_u: Optional[torch.Tensor],
        lb: Optional[torch.Tensor],
        threshold: Optional[torch.Tensor],
        lA: Optional[torch.Tensor],
        lbias: Optional[torch.Tensor],
        hash_vec_cache: Dict[str, torch.Tensor],
    ) -> Optional[list[Dict[str, Any]]]:
        try:
            if any(item is None for item in (cs, dm_l, dm_u, lb, threshold, lA, lbias)):
                return None
            cs_flat = cs.reshape(cs.shape[0], -1)
            target_cs = ordered_cs.reshape(ordered_cs.shape[0], -1).to(
                device=cs_flat.device, dtype=cs_flat.dtype
            )
            if target_cs.shape[1] != cs_flat.shape[1]:
                return None
            hash_vec = hash_vec_cache.get("vec")
            target_hash = hash_vec_cache.get("target")
            if hash_vec is None or target_hash is None:
                rng = torch.Generator(device=cs_flat.device).manual_seed(0)
                hash_vec = torch.randn(
                    cs_flat.shape[1], dtype=cs_flat.dtype, device=cs_flat.device, generator=rng,
                )
                target_hash = target_cs @ hash_vec
                hash_vec_cache["vec"] = hash_vec
                hash_vec_cache["target"] = target_hash
            domain_hash = cs_flat @ hash_vec
            diff = (domain_hash.unsqueeze(-1) - target_hash.unsqueeze(0)).abs()
            objective_indices = diff.argmin(dim=-1)
            matched = diff.gather(-1, objective_indices.unsqueeze(-1)).squeeze(-1) < 1e-5
            if not bool(matched.any().item()):
                return None

            records: list[Dict[str, Any]] = []
            matched_indices = matched.nonzero(as_tuple=False).view(-1)
            for domain_idx in matched_indices.tolist():
                objective_idx = int(objective_indices[domain_idx].item())
                records.append(
                    {
                        "objective_index": objective_idx,
                        "x_L": dm_l[domain_idx].detach().cpu().clone(),
                        "x_U": dm_u[domain_idx].detach().cpu().clone(),
                        "A": lA[domain_idx].detach().cpu().reshape(lA.shape[1], -1).clone(),
                        "bias": lbias[domain_idx].detach().cpu().reshape(-1).clone(),
                        "bound": lb[domain_idx].detach().cpu().reshape(-1).clone(),
                        "threshold": threshold[domain_idx].detach().cpu().reshape(-1).clone(),
                    }
                )
            return records
        except Exception:
            return None

    @staticmethod
    def _unique_input_indices_for_bounds(constraints: VerificationSpec) -> list[int]:
        flat_lower = constraints.lower.view(constraints.num_inputs, -1)
        flat_upper = constraints.upper.view(constraints.num_inputs, -1)
        selected: list[int] = []
        for idx in range(constraints.num_inputs):
            is_dup = any(
                torch.equal(flat_lower[idx], flat_lower[prev]) and torch.equal(flat_upper[idx], flat_upper[prev])
                for prev in selected
            )
            if not is_dup:
                selected.append(idx)
        return selected

    @staticmethod
    def _infer_output_dim_from_exprs(exprs: Sequence[LinearExpr]) -> int:
        max_idx = -1
        for expr in exprs:
            for (kind, idx), _ in expr.coeffs.items():
                if kind == "input":
                    raise ValueError("objective cannot contain input variables.")
                if kind != "output":
                    raise ValueError(f"Unsupported objective variable kind: {kind}")
                max_idx = max(max_idx, int(idx))
        if max_idx < 0:
            raise ValueError("objective must involve at least one output variable.")
        return max_idx + 1

    def _objective_to_rows(
        self,
        objective: Union[VariableVector, LinearExpr, Sequence[LinearExpr]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(objective, VariableVector):
            if objective.kind != "output":
                raise ValueError("objective VariableVector must be output variables.")
            objective_exprs: list[LinearExpr] = [cast(LinearExpr, objective[i]) for i in range(objective.size)]
            output_vector = objective
        elif isinstance(objective, LinearExpr):
            objective_exprs = [objective]
            if isinstance(self.output_vars, VariableVector):
                output_vector = self.output_vars
            else:
                output_vector = output_vars(self._infer_output_dim_from_exprs(objective_exprs))
        elif isinstance(objective, Sequence) and not isinstance(objective, (str, bytes)):
            if len(objective) == 0:
                raise ValueError("objective sequence cannot be empty.")
            if not all(isinstance(item, LinearExpr) for item in objective):
                raise TypeError("objective sequence must contain LinearExpr elements only.")
            objective_exprs = list(cast(Sequence[LinearExpr], objective))
            if isinstance(self.output_vars, VariableVector):
                output_vector = self.output_vars
            else:
                output_vector = output_vars(self._infer_output_dim_from_exprs(objective_exprs))
        else:
            raise TypeError("objective must be output_vars, LinearExpr, or a sequence of LinearExpr.")

        rows: list[torch.Tensor] = []
        offsets: list[float] = []
        for expr in objective_exprs:
            pseudo_pred = ComparisonPredicate(expr, LinearExpr({}, 0.0), "<=")
            C, rhs, input_preds = _aggregate_rows([pseudo_pred], expected_kind="output", vector=output_vector)
            if input_preds:
                raise ValueError("objective cannot contain input-side predicates.")
            if C.shape[0] != 1 or rhs.shape[0] != 1:
                raise RuntimeError("Internal objective parsing error: expected one row per expression.")
            rows.append(C[0].detach().clone())
            offsets.append(float(-rhs[0].item()))

        return torch.stack(rows, dim=0), torch.tensor(offsets, dtype=torch.float32)

    def _build_objective_constraints(
        self,
        base_constraints: VerificationSpec,
        coeff_row: torch.Tensor,
        input_index: int = 0,
    ) -> IOConstraints:
        row = torch.as_tensor(coeff_row).float()
        if row.ndim == 1:
            row = row.view(1, -1)
        elif row.ndim != 2:
            raise ValueError("Objective coefficients must be a 1-D row or a 2-D matrix.")
        rhs = torch.zeros((row.shape[0],), dtype=torch.float32)
        lower = base_constraints.lower[input_index:input_index + 1].detach().clone()
        upper = base_constraints.upper[input_index:input_index + 1].detach().clone()
        if row.shape[0] > 1:
            repeat_shape = (row.shape[0],) + (1,) * (lower.ndim - 1)
            lower = lower.repeat(repeat_shape)
            upper = upper.repeat(repeat_shape)
        clauses = [
            [(row[idx:idx + 1].detach().clone(), rhs[idx:idx + 1].detach().clone())]
            for idx in range(row.shape[0])
        ]
        return IOConstraints(
            lower=lower,
            upper=upper,
            clauses=clauses,
        )

    @staticmethod
    def _build_solving_mode_optimization_spec(
        base_constraints: VerificationSpec,
        *,
        objective_threshold: float,
        augmented_output_dim: int,
    ) -> VerificationSpec:
        """Prepend the objective row and shift existing constraints right by one column."""
        dtype = base_constraints.lower.dtype
        clauses: list[list[Tuple[torch.Tensor, torch.Tensor]]] = []
        for per_input in base_constraints.clauses:
            if len(per_input) == 0:
                clauses.append([
                    (
                        torch.cat(
                            [
                                torch.tensor([[1.0]], dtype=dtype),
                                torch.zeros((1, augmented_output_dim - 1), dtype=dtype),
                            ],
                            dim=1,
                        ),
                        torch.tensor([objective_threshold], dtype=dtype),
                    )
                ])
                continue
            augmented_per_input: list[Tuple[torch.Tensor, torch.Tensor]] = []
            for C_constr, rhs_constr in per_input:
                C_constr_t = torch.as_tensor(C_constr).detach().clone().to(dtype=dtype)
                rhs_constr_t = torch.as_tensor(rhs_constr).detach().clone().to(dtype=dtype)
                if C_constr_t.ndim == 1:
                    C_constr_t = C_constr_t.view(1, -1)
                if rhs_constr_t.ndim == 0:
                    rhs_constr_t = rhs_constr_t.view(1)
                C_aug = torch.zeros(
                    (C_constr_t.shape[0], augmented_output_dim),
                    dtype=C_constr_t.dtype,
                    device=C_constr_t.device,
                )
                if C_constr_t.numel() > 0:
                    C_aug[:, 1:] = C_constr_t
                objective_row = torch.zeros(
                    (1, augmented_output_dim), dtype=C_constr_t.dtype, device=C_constr_t.device
                )
                objective_row[0, 0] = 1.0
                augmented_per_input.append((
                    torch.cat([objective_row, C_aug], dim=0),
                    torch.cat([
                        torch.tensor([objective_threshold], dtype=rhs_constr_t.dtype, device=rhs_constr_t.device),
                        rhs_constr_t,
                    ], dim=0),
                ))
            clauses.append(augmented_per_input)
        return VerificationSpec.build_from_input_bounds(
            base_constraints.lower.detach().clone(),
            base_constraints.upper.detach().clone(),
            clauses,
        )

    def _run_bound_subproblem(
        self,
        *,
        constraints: VerificationSpec,
        interm_bounds: Optional[Dict[str, Any]],
        bound_config: Mapping[str, Any],
        return_linear_bounds: bool = False,
    ) -> SolveResult:
        prev_constraint = self.constraint
        prev_spec = self.spec
        prev_return_linear_bounds = self._return_linear_bounds
        try:
            self.constraint = constraints
            self.spec = constraints
            self._return_linear_bounds = return_linear_bounds
            with _config_context(bound_config):
                result = self._verify_impl(interm_bounds=interm_bounds, return_reference=True)
            self._last_result = result
            return result
        finally:
            self._return_linear_bounds = prev_return_linear_bounds
            self.constraint = prev_constraint
            self.spec = prev_spec

    @staticmethod
    def _compute_per_c_worst_lb_rhs(
        d: Any,
        ordered_cs: torch.Tensor,
        hash_vec_cache: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Group remaining BaB domains by their c row and return the worst
        ``lb - threshold`` per group, in the order of ``ordered_cs``.

        Used by ``_capture_per_objective_lb`` to expose a per-objective
        tightened lower bound from a single parallel BaB run, without
        touching ``batch_branch_and_bound.py``.
        """
        try:
            if d is None or d.cs is None or d.cs.num_used == 0:
                return None
            num_used = int(d.cs.num_used)
            cs_storage = d.cs._storage[:num_used].reshape(num_used, -1)
            target_cs = ordered_cs.reshape(ordered_cs.shape[0], -1).to(
                device=cs_storage.device, dtype=cs_storage.dtype
            )
            if target_cs.shape[1] != cs_storage.shape[1]:
                return None
            K = int(target_cs.shape[0])
            if K < 1:
                return None
            hash_vec = hash_vec_cache.get("vec")
            target_hash = hash_vec_cache.get("target")
            if hash_vec is None or target_hash is None:
                rng = torch.Generator(device=cs_storage.device).manual_seed(0)
                hash_vec = torch.randn(
                    cs_storage.shape[1], dtype=cs_storage.dtype, device=cs_storage.device, generator=rng,
                )
                target_hash = target_cs @ hash_vec
                hash_vec_cache["vec"] = hash_vec
                hash_vec_cache["target"] = target_hash
            domain_hash = cs_storage @ hash_vec
            # Match each domain hash to the nearest target via abs-diff.
            diff = (domain_hash.unsqueeze(-1) - target_hash.unsqueeze(0)).abs()
            inverse_idx = diff.argmin(dim=-1)
            matched_diff = diff.gather(-1, inverse_idx.unsqueeze(-1)).squeeze(-1)
            matched = matched_diff < 1e-5
            if not bool(matched.any().item()):
                return None
            lb_min = d.lb._storage[:num_used].reshape(num_used, -1).min(dim=-1).values
            thr_min = d.threshold._storage[:num_used].reshape(num_used, -1).min(dim=-1).values
            margins = lb_min - thr_min
            inverse_idx = inverse_idx[matched]
            margins = margins[matched]
            group_worst = torch.full(
                (K,), float("inf"), dtype=margins.dtype, device=margins.device,
            )
            group_worst.scatter_reduce_(0, inverse_idx, margins, reduce="amin", include_self=True)
            return group_worst.detach().cpu()
        except Exception:
            return None

    @staticmethod
    @contextlib.contextmanager
    def _capture_per_objective_lb(ordered_cs: torch.Tensor):
        """Temporarily monkey-patch the input-BaB entry points so that the
        single parallel BaB run records per-objective tightened ``lb - rhs``
        and writes it into ``lirpa_model.last_bab_global_lb``.

        ``ordered_cs`` defines the canonical objective order: each row is the
        ``c`` of one objective the API submitted as a separate input batch.
        After exiting the context, ``state['latest']`` contains the captured
        tensor (shape ``(K,)``) or ``None`` if BaB did not run / did not match.

        We patch:
        * ``input_split.batch_branch_and_bound.batch_verification_input_split``
          (module-level lookup) to capture the per-iteration state from the
          live domain list ``d``.
        * ``complete_verifier_func.input_bab_parallel`` (the imported alias
          actually called during BaB) to override the scalar
          ``net.last_bab_global_lb`` with our per-objective tensor before
          the API extraction reads it.

        Neither ``batch_branch_and_bound.py`` nor ``complete_verifier_func.py``
        is edited; this is a process-local monkey patch installed only while
        the context is active.
        """
        import input_split.batch_branch_and_bound as bb_mod
        import input_split.branching_domains as bd_mod
        import complete_verifier_func as cv_mod

        state: Dict[str, Any] = {"latest": None, "subdomains": None}
        hash_cache: Dict[str, torch.Tensor] = {}

        original_bv = bb_mod.batch_verification_input_split
        original_filter = bd_mod.UnsortedInputDomainList.filter_verified_domains
        original_ibp = cv_mod.input_bab_parallel

        def patched_filter_verified_domains(*args, **kwargs):
            ret = original_filter(*args, **kwargs)
            try:
                if isinstance(ret, Sequence) and len(ret) >= 8:
                    if len(ret) >= 9 and isinstance(ret[4], dict):
                        _, lb_filt, dm_l_filt, dm_u_filt, _, cs_filt, thr_filt, lA_filt, lbias_filt = ret[:9]
                    else:
                        _, lb_filt, dm_l_filt, dm_u_filt, cs_filt, thr_filt, lA_filt, lbias_filt = ret[:8]
                    subdomains = ABCrownSolver._capture_per_c_subdomains(
                        ordered_cs,
                        cs_filt,
                        dm_l_filt,
                        dm_u_filt,
                        lb_filt,
                        thr_filt,
                        lA_filt,
                        lbias_filt,
                        hash_cache,
                    )
                    if subdomains is not None:
                        state["subdomains"] = subdomains
            except Exception:
                pass
            return ret

        def patched_bv(d, net_inner, batch, *args, **kwargs):
            result = original_bv(d, net_inner, batch, *args, **kwargs)
            per_obj = ABCrownSolver._compute_per_c_worst_lb_rhs(d, ordered_cs, hash_cache)
            if per_obj is not None:
                state["latest"] = per_obj
            return result

        def patched_ibp(net, x, c, rhs, *args, **kwargs):
            ret = original_ibp(net, x, c, rhs, *args, **kwargs)
            if state["latest"] is not None:
                # Replace the scalar worst-domain summary with the tight
                # per-objective tensor so the API extraction (which filters
                # by expected_rows) accepts it.
                net.last_bab_global_lb = state["latest"].detach().clone().cpu()
            return ret

        bb_mod.batch_verification_input_split = patched_bv
        bd_mod.UnsortedInputDomainList.filter_verified_domains = staticmethod(patched_filter_verified_domains)
        cv_mod.input_bab_parallel = patched_ibp
        try:
            yield state
        finally:
            bb_mod.batch_verification_input_split = original_bv
            bd_mod.UnsortedInputDomainList.filter_verified_domains = staticmethod(original_filter)
            cv_mod.input_bab_parallel = original_ibp

    @staticmethod
    @contextlib.contextmanager
    def _capture_optimization_history(objective_index: int = 0):
        """Capture primal/dual sequences from solving-mode input BaB without editing runtime files."""
        import input_split.attack_in_input_split as ais_mod
        import input_split.batch_branch_and_bound as bb_mod

        state: Dict[str, Any] = {
            "primal_signed": [],
            "dual_signed": [],
            "best_signed_primal": None,
            "best_x": None,
        }

        original_update = ais_mod.update_rhs_with_attack
        original_update_alias = bb_mod.update_rhs_with_attack
        original_bv = bb_mod.batch_verification_input_split

        def patched_update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model_ori):
            device = x_L.device
            max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
            num_domains = min(max_num_domains, x_L.shape[0])
            print(f'Running PGD attack on {num_domains} domains')
            x_L = x_L[:num_domains]
            x_U = x_U[:num_domains]
            cs = cs[:num_domains]
            rhs = thresholds[:num_domains]
            dm_lb = dm_lb[:num_domains]

            data_max = x_U.unsqueeze(0)
            data_min = x_L.unsqueeze(0)
            x = (data_min + data_max) / 2
            spec_size = torch.full([cs.shape[0]], cs.shape[1], dtype=torch.int64, device=device)
            C_mat = cs.view(1, -1, cs.shape[-1])
            rhs_mat = rhs.view(1, -1)
            alpha = (data_max - data_min).max() / 8
            pgd_steps = arguments.Config['attack']['input_split_check_adv']['pgd_steps']
            ret = ais_mod.pgd_attack_with_general_specs(
                model_ori, x, data_min, data_max, C_mat, rhs_mat,
                spec_size, alpha=alpha, pgd_steps=pgd_steps
            )

            adv_input, adv_output = ret.adv_input_per_or, ret.adv_output_per_or
            adv_output = adv_output.view(cs.shape[0], -1, 1)
            upper_bound = cs.matmul(adv_output).squeeze(-1)

            print('Trying to update RHS with attack')
            print(f'  Current RHS: mean {rhs.mean().item()}')
            print(f'  New upper bound: mean {upper_bound.mean().item()}')
            print(f'  Number of updated RHS: {(upper_bound < rhs).sum()}/{rhs.numel()}')
            rhs = torch.min(rhs, upper_bound)
            thresholds[:num_domains] = rhs
            gap = rhs - dm_lb
            min_gap = gap.min()
            print('  Gap between lower/upper bounds: '
                  f'mean {gap.mean().item()}, min {min_gap.item()}')
            assert min_gap >= -1e-3, 'Gap between lower and upper bounds is negative'

            if upper_bound.numel() > 0:
                objective_values = upper_bound[:, objective_index].reshape(-1)
                best_idx = int(objective_values.argmin().item())
                best_signed = float(objective_values[best_idx].item())
                current_best = state['best_signed_primal']
                if current_best is None or best_signed < current_best:
                    state['best_signed_primal'] = best_signed
                    state['best_x'] = adv_input[0, best_idx].detach().clone().cpu()

            return thresholds

        def patched_bv(d, net_inner, batch, *args, **kwargs):
            result = original_bv(d, net_inner, batch, *args, **kwargs)
            try:
                if len(d) == 0:
                    if state['primal_signed']:
                        final_signed = float(state['primal_signed'][-1])
                        state['primal_signed'].append(final_signed)
                        state['dual_signed'].append(final_signed)
                    return result

                num_used = int(getattr(d.cs, 'num_used', 0))
                if num_used <= 0:
                    return result
                lb_storage = d.lb._storage[:num_used].reshape(num_used, -1)
                threshold_storage = d.threshold._storage[:num_used].reshape(num_used, -1)
                signed_dual = float(lb_storage[:, objective_index].min().item())
                signed_primal = float(threshold_storage[:, objective_index].min().item())

                if state['primal_signed']:
                    signed_primal = min(float(state['primal_signed'][-1]), signed_primal)
                if state['dual_signed']:
                    signed_dual = max(float(state['dual_signed'][-1]), signed_dual)

                state['primal_signed'].append(signed_primal)
                state['dual_signed'].append(signed_dual)
            except Exception:
                pass
            return result

        ais_mod.update_rhs_with_attack = patched_update_rhs_with_attack
        bb_mod.update_rhs_with_attack = patched_update_rhs_with_attack
        bb_mod.batch_verification_input_split = patched_bv
        try:
            yield state
        finally:
            ais_mod.update_rhs_with_attack = original_update
            bb_mod.update_rhs_with_attack = original_update_alias
            bb_mod.batch_verification_input_split = original_bv

    def verify(
        self,
        *,
        constraints: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        interm_bounds: Optional[Dict[str, Any]] = None,
        return_reference: bool = True,
    ) -> SolveResult:
        """Run verification with current solver/model/config setup."""
        if constraints is not None:
            self.constraint = self._normalize_constraint(constraints)
            self.spec = self.constraint
        if self.constraint is None:
            raise ValueError("constraints must be provided in constructor or verify().")
        if not self._has_output_constraints(self.constraint):
            raise ValueError(
                "output_constraint is required for verify(). "
                "You can omit output_constraint for future compute-bounds workflows."
            )
        with _config_context(self.config):
            result = self._verify_impl(interm_bounds=interm_bounds, return_reference=return_reference)
        self._last_result = result
        return result

    def solve(
        self,
        interm_bounds: Optional[Dict[str, Any]] = None,
        return_reference: bool = True,
    ) -> SolveResult:
        """Solve a verification instance with the configured setup.

        Backward-compatible wrapper used by tests that predate the rename to
        ``verify()``. Unlike ``verify()``, this does not require an output
        constraint to be present.
        """
        if self.constraint is None:
            raise ValueError("constraints must be provided in constructor or solve().")
        with _config_context(self.config):
            result = self._verify_impl(interm_bounds=interm_bounds, return_reference=return_reference)
        self._last_result = result
        return result

    def compute_bounds(
        self,
        *,
        constraints: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        objective: Optional[Union[VariableVector, LinearExpr, Sequence[LinearExpr]]] = None,
        interm_bounds: Optional[Dict[str, Any]] = None,
        return_linear_bounds: bool = False,
    ) -> BoundsResult:
        """
        Compute lower/upper bounds for objective expression(s) over the given constraints.

        This method internally reuses the complete_verifier (including BaB when enabled)
        by turning each objective expression into an auxiliary bound subproblem.
        PGD attack is always skipped in this API to avoid returning adversarial-status
        shortcuts instead of optimization bounds. If return_linear_bounds is true,
        the result also includes a structured affine relaxation object inside
        ``result.linear_bounds`` containing global ``(A, b)`` terms and, when
        input BaB is active, the current per-subdomain affine relaxations.
        """
        if constraints is not None:
            self.constraint = self._normalize_constraint(constraints)
            self.spec = self.constraint
        if self.constraint is None:
            raise ValueError("constraints must be provided in constructor or compute_bounds().")
        base_constraints = self.constraint

        if objective is None:
            if self.output_vars is None:
                raise ValueError("objective must be provided when solver has no output_vars.")
            objective = self.output_vars

        rows, offsets = self._objective_to_rows(objective)
        if rows.numel() == 0:
            raise ValueError("objective produced no rows.")

        original_constraint = self.constraint
        original_spec = self.spec

        bound_config = _clone_config(self.config)
        _deep_update(bound_config, {"attack": {"pgd_order": "skip"}, "bab": {"attack": {"enabled": False}}})
        bab_config = bound_config.setdefault("bab", {})
        if not isinstance(bab_config, MutableMapping):
            raise TypeError("Config key 'bab' must be a mapping.")
        try:
            sort_interval = int(bab_config.get("sort_domain_interval", -1))
        except (TypeError, ValueError):
            sort_interval = -1
        if sort_interval <= 0:
            bab_config["sort_domain_interval"] = 1
        try:
            timeout = float(bab_config.get("timeout", 360.0))
        except (TypeError, ValueError):
            timeout = 360.0
        if bab_config.get("override_timeout") is None and timeout == 360.0:
            bab_config["timeout"] = _COMPUTE_BOUNDS_DEFAULT_TIMEOUT

        input_indices = self._unique_input_indices_for_bounds(base_constraints)
        num_inputs = len(input_indices)
        num_objectives = int(rows.shape[0])
        lower_vals = torch.empty((num_inputs, num_objectives), dtype=torch.float32)
        upper_vals = torch.empty((num_inputs, num_objectives), dtype=torch.float32)
        statuses: list[list[Dict[str, str]]] = []
        linear_bound_matrices: list[torch.Tensor] = []
        linear_bound_biases: list[torch.Tensor] = []
        linear_bound_subdomains: list[Dict[str, list[Dict[str, Any]]]] = []
        success = True
        def _fmt(t: Optional[torch.Tensor]) -> str:
            if t is None:
                return "None"
            return "[" + ", ".join(f"{float(v):+.6f}" for v in t.view(-1).tolist()) + "]"

        try:
            for compact_input_idx, source_input_idx in enumerate(input_indices):
                print(f"[compute_bounds] input_batch={source_input_idx} "
                      f"objectives={num_objectives} -- running LOWER subproblem (1 parallel BaB)")
                lower_constraint = self._build_objective_constraints(
                    base_constraints, rows, input_index=source_input_idx
                )
                with self._capture_per_objective_lb(rows.detach().clone()) as lower_capture:
                    lower_result = self._run_bound_subproblem(
                        constraints=lower_constraint,
                        interm_bounds=interm_bounds,
                        bound_config=bound_config,
                        return_linear_bounds=return_linear_bounds,
                    )
                lbs = self._extract_lbs_from_verify_result(lower_result, expected_rows=num_objectives)
                print(f"[compute_bounds] LOWER per-objective lb after BaB: {_fmt(lbs)} "
                      f"(status={lower_result.status})")

                print(f"[compute_bounds] input_batch={source_input_idx} "
                      f"objectives={num_objectives} -- running UPPER subproblem (1 parallel BaB)")
                upper_constraint = self._build_objective_constraints(
                    base_constraints, -rows, input_index=source_input_idx
                )
                with self._capture_per_objective_lb((-rows).detach().clone()) as upper_capture:
                    upper_result = self._run_bound_subproblem(
                        constraints=upper_constraint,
                        interm_bounds=interm_bounds,
                        bound_config=bound_config,
                        return_linear_bounds=return_linear_bounds,
                    )
                neg_lbs = self._extract_lbs_from_verify_result(upper_result, expected_rows=num_objectives)
                print(f"[compute_bounds] UPPER per-objective lb(-obj) after BaB: {_fmt(neg_lbs)} "
                      f"(status={upper_result.status})")

                if lbs is None:
                    lower_vals[compact_input_idx, :] = float("-inf")
                    success = False
                else:
                    lower_vals[compact_input_idx, :] = lbs + offsets
                if neg_lbs is None:
                    upper_vals[compact_input_idx, :] = float("inf")
                    success = False
                else:
                    upper_vals[compact_input_idx, :] = -neg_lbs + offsets

                if lower_result.status == "unknown" or upper_result.status == "unknown":
                    success = False
                input_statuses: list[Dict[str, str]] = []
                for objective_idx in range(num_objectives):
                    input_statuses.append(
                        {
                            "source_input_index": str(source_input_idx),
                            "objective_index": str(objective_idx),
                            "lower_status": lower_result.status,
                            "upper_status": upper_result.status,
                        }
                    )
                statuses.append(input_statuses)
                if return_linear_bounds:
                    lower_A = self._extract_linear_bound_matrix(
                        lower_result, expected_rows=num_objectives
                    )
                    upper_neg_A = self._extract_linear_bound_matrix(
                        upper_result, expected_rows=num_objectives
                    )
                    if lower_A is None or upper_neg_A is None:
                        raise RuntimeError("return_linear_bounds=True but verifier did not return lA matrices.")
                    flat_input_dim = int(base_constraints.lower[source_input_idx].numel())
                    if lower_A.shape[1] != flat_input_dim or upper_neg_A.shape[1] != flat_input_dim:
                        raise RuntimeError(
                            "return_linear_bounds=True currently requires input-side A matrices. "
                            f"Received shapes {tuple(lower_A.shape)} and {tuple(upper_neg_A.shape)} for "
                            f"an input box with flattened dimension {flat_input_dim}."
                        )
                    lower_ref_lb = self._extract_reference_lbs_from_verify_result(
                        lower_result, expected_rows=num_objectives
                    )
                    upper_neg_ref_lb = self._extract_reference_lbs_from_verify_result(
                        upper_result, expected_rows=num_objectives
                    )
                    if lower_ref_lb is None or upper_neg_ref_lb is None:
                        raise RuntimeError(
                            "return_linear_bounds=True but verifier did not return reference global_lb values."
                        )
                    # The upper run lower-bounds -objective, so negate its A matrix.
                    upper_A = -upper_neg_A
                    linear_bound_matrices.append(torch.stack([lower_A, upper_A], dim=0))

                    input_lower = base_constraints.lower[source_input_idx]
                    input_upper = base_constraints.upper[source_input_idx]
                    lower_affine_min = self._concretize_affine_over_box(
                        lower_A,
                        input_lower,
                        input_upper,
                        use_upper=False,
                    )
                    upper_affine_max = self._concretize_affine_over_box(
                        upper_A,
                        input_lower,
                        input_upper,
                        use_upper=True,
                    )
                    lower_bias = lower_ref_lb - lower_affine_min
                    upper_bias = -upper_neg_ref_lb - upper_affine_max
                    linear_bound_biases.append(torch.stack([lower_bias, upper_bias], dim=0))
                    lower_subdomains = [] if lower_capture.get("subdomains") is None else lower_capture["subdomains"]
                    upper_subdomains = []
                    if upper_capture.get("subdomains") is not None:
                        for record in upper_capture["subdomains"]:
                            upper_subdomains.append(
                                {
                                    **record,
                                    "A": -record["A"],
                                    "bias": -record["bias"],
                                    "bound": -record["bound"],
                                }
                            )
                    for record in lower_subdomains:
                        record["source_input_index"] = source_input_idx
                    for record in upper_subdomains:
                        record["source_input_index"] = source_input_idx
                    linear_bound_subdomains.append({
                        "lower": lower_subdomains,
                        "upper": upper_subdomains,
                    })
        finally:
            self.constraint = base_constraints if constraints is not None else original_constraint
            self.spec = self.constraint if constraints is not None else original_spec

        lower_tensor = lower_vals[0] if num_inputs == 1 else lower_vals
        upper_tensor = upper_vals[0] if num_inputs == 1 else upper_vals
        linear_bounds: Optional[Dict[str, Any]] = None
        if return_linear_bounds:
            stacked_linear_bounds = torch.stack(linear_bound_matrices, dim=0)
            stacked_linear_biases = torch.stack(linear_bound_biases, dim=0)
            lower_A = stacked_linear_bounds[:, 0]
            upper_A = stacked_linear_bounds[:, 1]
            lower_bias = stacked_linear_biases[:, 0]
            upper_bias = stacked_linear_biases[:, 1]
            linear_bounds = {
                "lower_A": lower_A[0] if num_inputs == 1 else lower_A,
                "lower_bias": lower_bias[0] if num_inputs == 1 else lower_bias,
                "upper_A": upper_A[0] if num_inputs == 1 else upper_A,
                "upper_bias": upper_bias[0] if num_inputs == 1 else upper_bias,
                "subdomains": linear_bound_subdomains[0] if num_inputs == 1 else linear_bound_subdomains,
            }
        effective_timeout = bab_config.get("override_timeout")
        if effective_timeout is None:
            effective_timeout = bab_config.get("timeout")
        return BoundsResult(
            lower=lower_tensor,
            upper=upper_tensor,
            success=success,
            stats={
                "statuses": statuses,
                "num_objectives": num_objectives,
                "num_inputs": num_inputs,
                "source_input_indices": input_indices,
                "bab_timeout": effective_timeout,
                "sort_domain_interval": bab_config.get("sort_domain_interval"),
            },
            linear_bounds=linear_bounds,
        )

    @staticmethod
    def _scalar_or_none(value: Any) -> Optional[float]:
        """Convert scalars/tensors to a finite float."""
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach().reshape(-1)
            if tensor.numel() == 0:
                return None
            value = tensor[0].item()
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scalar):
            return None
        return scalar

    @staticmethod
    def _as_feasible_candidate(
        *,
        objective_value: Any,
        constraint_violation: Any,
        x_solution: Optional[torch.Tensor],
        feas_tol: float,
    ) -> Optional[Tuple[float, float, torch.Tensor]]:
        """Normalize solver output into (objective, violation, x) if feasible."""
        obj = ABCrownSolver._scalar_or_none(objective_value)
        viol = ABCrownSolver._scalar_or_none(constraint_violation)
        if obj is None or viol is None or x_solution is None:
            return None
        if viol > feas_tol:
            return None
        return obj, viol, x_solution.detach().clone()

    @staticmethod
    def _run_pgd_primal(
        model: torch.nn.Module,
        dm_l: torch.Tensor,
        dm_u: torch.Tensor,
        obj_row: torch.Tensor,
        C_constr: torch.Tensor,
        rhs_constr: torch.Tensor,
        input_constr_A: Optional[torch.Tensor],
        input_constr_b: Optional[torch.Tensor],
        num_restarts: int = 8,
        num_steps: int = 300,
        step_size: float = 0.02,
        penalty: float = 100.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Minimum projected-gradient-descent primal solver.

        Minimizes ``obj_row . model(x)`` over the box ``[dm_l, dm_u]`` with a
        soft quadratic penalty on output clause constraints
        ``C_constr . model(x) <= rhs_constr`` and input linear constraints
        ``input_constr_A . x <= input_constr_b``. No external solvers.
        """
        dev = device if device is not None else dm_l.device
        dt = dtype if dtype is not None else dm_l.dtype
        dm_l = dm_l.detach().to(dev, dt)
        dm_u = dm_u.detach().to(dev, dt)
        obj_row = obj_row.detach().to(dev, dt).view(-1)
        if C_constr.numel() > 0:
            C_constr = C_constr.detach().to(dev, dt)
            rhs_constr = rhs_constr.detach().to(dev, dt).view(-1)
        if input_constr_A is not None and input_constr_A.numel() > 0:
            input_constr_A = input_constr_A.detach().to(dev, dt)
            input_constr_b = input_constr_b.detach().to(dev, dt).view(-1)
        else:
            input_constr_A = None

        model = model.to(dev)
        was_training = model.training
        model.eval()

        best_obj: Optional[float] = None
        best_viol = float("inf")
        best_x: Optional[torch.Tensor] = None

        try:
            for _ in range(num_restarts):
                x = dm_l + torch.rand_like(dm_l) * (dm_u - dm_l)
                x = x.detach().requires_grad_(True)
                for _ in range(num_steps):
                    y = model(x.unsqueeze(0)).reshape(-1)
                    obj = (obj_row * y).sum()
                    penalty_loss = torch.zeros((), device=dev, dtype=dt)
                    if C_constr.numel() > 0:
                        penalty_loss = penalty_loss + (torch.relu(C_constr @ y - rhs_constr) ** 2).sum()
                    if input_constr_A is not None:
                        penalty_loss = penalty_loss + (torch.relu(input_constr_A @ x.view(-1) - input_constr_b) ** 2).sum()
                    loss = obj + penalty * penalty_loss
                    grad = torch.autograd.grad(loss, x)[0]
                    with torch.no_grad():
                        x = (x - step_size * grad).clamp(dm_l, dm_u)
                    x = x.detach().requires_grad_(True)
                with torch.no_grad():
                    y = model(x.unsqueeze(0)).reshape(-1)
                    obj_val = float((obj_row * y).sum().item())
                    viol = 0.0
                    if C_constr.numel() > 0:
                        viol += float(torch.relu(C_constr @ y - rhs_constr).sum().item())
                    if input_constr_A is not None:
                        viol += float(torch.relu(input_constr_A @ x.view(-1) - input_constr_b).sum().item())
                better = (
                    best_x is None
                    or viol < best_viol - 1e-9
                    or (abs(viol - best_viol) <= 1e-9 and obj_val < (best_obj if best_obj is not None else float("inf")))
                )
                if better:
                    best_obj = obj_val
                    best_viol = viol
                    best_x = x.detach().clone()
        finally:
            if was_training:
                model.train()

        return {
            "solver": "pgd",
            "objective": best_obj,
            "violation": best_viol if best_x is not None else None,
            "x": best_x,
            "status": "feasible" if (best_x is not None and best_viol < 1e-4) else "infeasible_or_no_solution",
            "time": 0.0,
        }

    def _compute_optimization_history(
        self,
        *,
        objective: Union[VariableVector, LinearExpr, Sequence[LinearExpr]],
        constraints: VerificationSpec,
        direction: str,
        warm_start: Optional[OptimizeResult] = None,
    ) -> Dict[str, Any]:
        if constraints.num_inputs != 1:
            raise ValueError("return_bound_history=True currently supports exactly one input region.")

        rows, offsets = self._objective_to_rows(objective)
        if rows.shape[0] != 1:
            raise ValueError("return_bound_history=True requires a scalar objective expression.")

        sign = 1.0 if direction == 'minimize' else -1.0
        signed_obj_row = rows[0].detach().clone().float() * sign
        obj_offset = float(offsets[0].item())
        initial_primal_value = None if warm_start is None else warm_start.primal_value
        if initial_primal_value is None:
            raise ValueError(
                "return_bound_history=True requires a finite primal warm start; "
                "the local PGD optimizer did not produce one."
            )
        initial_signed_threshold = sign * (float(initial_primal_value) - obj_offset)

        history_config = _clone_config(self.config)
        _deep_update(
            history_config,
            {
                'general': {'complete_verifier': 'input_bab'},
                'solver': {'bound_prop_method': 'crown'},
                'attack': {'pgd_order': 'skip'},
                'bab': {
                    'attack': {'enabled': False},
                    'branching': {
                        'method': 'naive',
                        'input_split': {
                            'enable': True,
                            'update_rhs_with_attack': True,
                        },
                    },
                },
            },
        )
        bab_config = history_config.setdefault('bab', {})
        if not isinstance(bab_config, MutableMapping):
            raise TypeError("Config key 'bab' must be a mapping.")
        try:
            sort_interval = int(bab_config.get('sort_domain_interval', -1))
        except (TypeError, ValueError):
            sort_interval = -1
        if sort_interval <= 0:
            bab_config['sort_domain_interval'] = 1

        base_model = self.computing_graph
        augmented_model = _ObjectiveOutputWrapper(cast(torch.nn.Module, base_model), signed_obj_row)
        solving_constraints = self._build_solving_mode_optimization_spec(
            constraints,
            objective_threshold=initial_signed_threshold,
            augmented_output_dim=int(signed_obj_row.numel()) + 1,
        )

        prev_model_graph = self.computing_graph
        prev_model = self._model
        try:
            self.computing_graph = augmented_model
            self._model = None
            with self._capture_optimization_history(objective_index=0) as history_state:
                self._run_bound_subproblem(
                    constraints=solving_constraints,
                    interm_bounds=None,
                    bound_config=history_config,
                    return_linear_bounds=False,
                )
        finally:
            self.computing_graph = prev_model_graph
            self._model = prev_model

        primal_signed = [float(v) for v in history_state['primal_signed']]
        dual_signed = [float(v) for v in history_state['dual_signed']]
        primal_values = [sign * value + obj_offset for value in primal_signed]
        dual_bounds = [sign * value + obj_offset for value in dual_signed]
        best_signed = history_state['best_signed_primal']
        primal_value = None if best_signed is None else sign * float(best_signed) + obj_offset
        if primal_value is None and warm_start is not None:
            primal_value = warm_start.primal_value

        return {
            'direction': direction,
            'primal_values': primal_values,
            'dual_bounds': dual_bounds,
            'dual_bound_type': 'lower' if direction == 'minimize' else 'upper',
            'primal_value': primal_value,
            'x_best': history_state['best_x'] if history_state['best_x'] is not None else None if warm_start is None else warm_start.x_best,
        }

    def _optimize_with_nlp(
        self,
        *,
        objective: Optional[Union[VariableVector, LinearExpr, Sequence[LinearExpr]]],
        constraints: Optional[Union[VerificationSpec, Mapping[str, Any]]],
        direction: str,
    ) -> OptimizeResult:
        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if constraints is not None:
            self.constraint = self._normalize_constraint(constraints)
            self.spec = self.constraint
        if self.constraint is None:
            raise ValueError("constraints must be provided in constructor or minimize()/maximize().")
        base_constraints = self.constraint

        if objective is None:
            if self.output_vars is None:
                raise ValueError("objective must be provided when solver has no output_vars.")
            if self.output_vars.size != 1:
                raise ValueError(
                    "objective is ambiguous when output_vars has more than one element; "
                    "please provide a scalar objective expression."
                )
            objective = cast(LinearExpr, self.output_vars[0])

        rows, offsets = self._objective_to_rows(objective)
        if rows.shape[0] != 1:
            raise ValueError("minimize()/maximize() require a scalar objective expression.")

        sign = 1.0 if direction == "minimize" else -1.0
        obj_row = rows[0].detach().clone().float() * sign
        obj_offset = float(offsets[0].item())

        with _config_context(self.config):
            device = arguments.Config["general"]["device"]
            model_ori = self._prepare_model(device)
            target_device = torch.device(device) if isinstance(device, str) else device
            feas_tol = 1e-4

            solver_runs = 0
            best_obj: Optional[float] = None
            best_viol: Optional[float] = None
            best_x: Optional[torch.Tensor] = None
            best_input_idx: Optional[int] = None
            best_clause_idx: Optional[int] = None

            for input_idx in range(base_constraints.num_inputs):
                dm_l = base_constraints.lower[input_idx].detach().clone()
                dm_u = base_constraints.upper[input_idx].detach().clone()

                per_input_clauses = list(base_constraints.clauses[input_idx])
                if len(per_input_clauses) == 0:
                    per_input_clauses = [
                        (
                            torch.empty((0, obj_row.numel()), dtype=torch.float32),
                            torch.empty((0,), dtype=torch.float32),
                        )
                    ]

                input_constr_A = None
                input_constr_b = None

                for clause_idx, (C_constr, rhs_constr) in enumerate(per_input_clauses):
                    C_constr_t = torch.as_tensor(C_constr).detach().clone().float()
                    rhs_constr_t = torch.as_tensor(rhs_constr).detach().clone().float()
                    if C_constr_t.ndim == 1:
                        C_constr_t = C_constr_t.view(1, -1)
                    if rhs_constr_t.ndim == 0:
                        rhs_constr_t = rhs_constr_t.view(1)
                    if C_constr_t.numel() > 0 and C_constr_t.shape[1] != obj_row.numel():
                        raise ValueError(
                            f"Objective output dimension ({obj_row.numel()}) does not match clause output "
                            f"dimension ({C_constr_t.shape[1]}) at input {input_idx}, clause {clause_idx}."
                        )

                    run = self._run_pgd_primal(
                        model=model_ori,
                        dm_l=dm_l,
                        dm_u=dm_u,
                        obj_row=obj_row,
                        C_constr=C_constr_t,
                        rhs_constr=rhs_constr_t,
                        input_constr_A=input_constr_A,
                        input_constr_b=input_constr_b,
                        device=target_device,
                    )
                    solver_runs += 1
                    candidate = self._as_feasible_candidate(
                        objective_value=run["objective"],
                        constraint_violation=run["violation"],
                        x_solution=cast(Optional[torch.Tensor], run["x"]),
                        feas_tol=feas_tol,
                    )
                    if candidate is None:
                        continue
                    cand_obj, cand_viol, cand_x = candidate
                    if best_obj is None or cand_obj < best_obj:
                        best_obj = cand_obj
                        best_viol = cand_viol
                        best_x = cand_x
                        best_input_idx = input_idx
                        best_clause_idx = clause_idx

            if best_obj is None or best_x is None:
                return OptimizeResult(
                    status="infeasible_or_no_solution",
                    success=False,
                    primal_value=None,
                    x_best=None,
                    solver=None,
                    stats={
                        "direction": direction,
                        "solver_runs": solver_runs,
                        "num_inputs": base_constraints.num_inputs,
                    },
                )

            primal_value = sign * best_obj + obj_offset
            return OptimizeResult(
                status="feasible",
                success=True,
                primal_value=primal_value,
                x_best=best_x.detach().clone().cpu(),
                solver="pgd",
                stats={
                    "direction": direction,
                    "solver_runs": solver_runs,
                    "best_violation": best_viol,
                    "source_input_index": best_input_idx,
                    "source_clause_index": best_clause_idx,
                },
            )

    def minimize(
        self,
        objective: Optional[Union[VariableVector, LinearExpr, Sequence[LinearExpr]]] = None,
        constraints: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        return_bound_history: bool = False,
    ) -> OptimizeResult:
        """Find a feasible point minimizing a scalar objective under solver constraints."""
        result = self._optimize_with_nlp(
            objective=objective,
            constraints=constraints,
            direction="minimize",
        )
        if return_bound_history:
            if objective is None:
                if self.output_vars is None:
                    raise ValueError("objective must be provided when solver has no output_vars.")
                if self.output_vars.size != 1:
                    raise ValueError(
                        "objective is ambiguous when output_vars has more than one element; "
                        "please provide a scalar objective expression."
                    )
                objective = cast(LinearExpr, self.output_vars[0])
            if constraints is None:
                if self.constraint is None:
                    raise ValueError("constraints must be provided in constructor or minimize().")
                norm_constraints = self.constraint
            else:
                norm_constraints = self._normalize_constraint(constraints)
            history = self._compute_optimization_history(
                objective=objective,
                constraints=norm_constraints,
                direction='minimize',
                warm_start=result,
            )
            result.bound_history = history
            if history['primal_value'] is not None:
                result.primal_value = history['primal_value']
            if history['x_best'] is not None:
                result.x_best = cast(torch.Tensor, history['x_best']).detach().clone().cpu()
            result.solver = 'input_bab+pgd'
        return result

    def maximize(
        self,
        objective: Optional[Union[VariableVector, LinearExpr, Sequence[LinearExpr]]] = None,
        constraints: Optional[Union[VerificationSpec, Mapping[str, Any]]] = None,
        return_bound_history: bool = False,
    ) -> OptimizeResult:
        """Find a feasible point maximizing a scalar objective under solver constraints."""
        result = self._optimize_with_nlp(
            objective=objective,
            constraints=constraints,
            direction="maximize",
        )
        if return_bound_history:
            if objective is None:
                if self.output_vars is None:
                    raise ValueError("objective must be provided when solver has no output_vars.")
                if self.output_vars.size != 1:
                    raise ValueError(
                        "objective is ambiguous when output_vars has more than one element; "
                        "please provide a scalar objective expression."
                    )
                objective = cast(LinearExpr, self.output_vars[0])
            if constraints is None:
                if self.constraint is None:
                    raise ValueError("constraints must be provided in constructor or maximize().")
                norm_constraints = self.constraint
            else:
                norm_constraints = self._normalize_constraint(constraints)
            history = self._compute_optimization_history(
                objective=objective,
                constraints=norm_constraints,
                direction='maximize',
                warm_start=result,
            )
            result.bound_history = history
            if history['primal_value'] is not None:
                result.primal_value = history['primal_value']
            if history['x_best'] is not None:
                result.x_best = cast(torch.Tensor, history['x_best']).detach().clone().cpu()
            result.solver = 'input_bab+pgd'
        return result

    def _verify_impl(self, interm_bounds: Optional[Dict[str, Any]], return_reference: bool) -> SolveResult:
        """Core verify routine that orchestrates attacks, incomplete, and complete verification."""
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
        runtime_spec = self._prepare_runtime_spec()

        if self.constraint is None:
            raise RuntimeError("constraints must be set before verification.")
        self.vnnlib_handler = self._build_vnnlib_handler(runtime_spec)
        x = self.vnnlib_handler.x[0:1].to(device)
        data_min = self.vnnlib_handler.data_min[0:1].to(device)
        data_max = self.vnnlib_handler.data_max[0:1].to(device)
        if general_args['adhoc_tuning']:
            eval(general_args['adhoc_tuning'])(model_ori, self.vnnlib_handler)

        complete_verifier = general_args['complete_verifier']
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
        solving_mode = arguments.Config['solving']['solving_mode']

        if arguments.Config['attack']['pgd_order'] != 'skip':
            reset_attack_stats()

        if arguments.Config['attack']['pgd_order'] == 'before' and not solving_mode:
            verified_status, verified_success, attack_examples, attack_margins, all_adv_candidates = (
                self._attack(model_ori, verified_status, verified_success)
            )
            get_attack_stats(self.logger, 0)
            if debug_args['sanity_check']:
                rhs_offset = attack_margins if debug_args['sanity_check'] == 'Full' else attack_margins.min()
                self.vnnlib_handler.add_rhs_offset(rhs_offset)
                arguments.Config['attack']['pgd_order'] = 'skip'
                verified_status, verified_success = 'unknown', False
        elif arguments.Config['attack']['pgd_order'] == 'before':
            print('Skipping top-level PGD before in solving_mode; search is handled inside the solver.')

        model_incomplete = None
        reference = {}

        if debug_args['test_optimized_bounds']:
            compare_optimized_bounds_against_lp_bounds(
                model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=self.vnnlib_handler.vnnlib
            )

        if not verified_success:
            verified_status, reference = incomplete_verifier_core(self, model_ori, interm_bounds)
            if self.spec_handler_incomplete is not None:
                attack_examples, attack_margins, all_adv_candidates = (
                    self.spec_handler_incomplete.prune_attack_ret(
                        attack_examples, attack_margins, all_adv_candidates
                    )
                )
            if general_args['return_optimized_model']:
                normalized_status = self._normalize_verify_status(verified_status)
                return SolveResult(status=normalized_status, success=normalized_status != 'unknown')
            verified_success = verified_status != 'unknown'
            model_incomplete = reference.get('model', None)

        if not verified_success and arguments.Config['attack']['pgd_order'] == 'after' and not solving_mode:
            verified_status, verified_success, attack_examples, attack_margins, all_adv_candidates = (
                self._attack(model_ori, verified_status, verified_success)
            )
            get_attack_stats(self.logger, 0)
        elif not verified_success and arguments.Config['attack']['pgd_order'] == 'after':
            print('Skipping top-level PGD after in solving_mode; search is handled inside the solver.')

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
            if self.spec_handler_incomplete is not None:
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
                getattr(model_incomplete, 'processes', None),
                watch_dog_proc=getattr(model_incomplete, 'get_cuts_watch_dog_proc', None),
            )
            if hasattr(model_incomplete, 'processes'):
                del model_incomplete.processes

        if temp_cuts_folder is not None:
            temp_cuts_folder.cleanup()
            bab_args['cut']['cuts_path'] = None

        if debug_args['sanity_check']:
            if 'unknown' not in verified_status:
                raise AssertionError('Sanity check failed: status should remain unknown.')

        verified_status = self._normalize_verify_status(verified_status)
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
        """Expose legacy BaB entry point."""
        return bab_core(self, *args, **kwargs)

    def _prepare_environment(self, device: str) -> None:
        """Set seeds, torch settings, and optional precompile based on config."""
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
        torch.set_default_dtype(torch.float64 if general_args['double_fp'] else torch.float32)
        if general_args['precompile_jit']:
            precompile_jit_kernels()
        if general_args['reset_seed_after_precompile']:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _build_vnnlib_handler(self, constraints: VerificationSpec) -> vnnlibHandler:
        """Convert internal constraints into a handler used by verification routines."""
        vnnlib = constraints.to_vnnlib()
        return vnnlibHandler(vnnlib, constraints.input_shape)

    def _prepare_model(self, device: str) -> torch.nn.Module:
        """Normalize the provided computing graph into a Torch model on the right device."""
        graph = self.computing_graph
        model: Optional[torch.nn.Module] = None
        target_dtype = torch.float64 if arguments.Config['general']['double_fp'] else torch.float32

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
                if self.constraint is not None:
                    self.constraint.reshape_input(onnx_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Constraint input shape {self.constraint.input_shape[1:] if self.constraint is not None else None} is incompatible with "
                    f"ONNX model expected shape {onnx_shape}."
                ) from exc
        else:
            raise TypeError(f"Unsupported computing graph type: {type(graph).__name__}")

        assert model is not None
        model = model.to(device=device, dtype=target_dtype)
        model.eval()
        if isinstance(self.computing_graph, str) and self.computing_graph.lower().endswith((".onnx", ".onnx.gz")):
            self.computing_graph = model
        self._model = model
        return model

    def _prepare_runtime_spec(self) -> VerificationSpec:
        """Build the spec object used by the current solve call."""
        if self.constraint is None:
            raise RuntimeError("constraints must be set before verification.")
        runtime_spec = copy.deepcopy(self.constraint)
        self._runtime_spec = runtime_spec
        return runtime_spec

    def _normalize_constraint(self, constraint: Union[VerificationSpec, Mapping[str, Any]]) -> VerificationSpec:
        """Accept various constraint formats and convert to IOConstraints."""
        if isinstance(constraint, VerificationSpec):
            return constraint
        if isinstance(constraint, Mapping):
            if {'lower', 'upper', 'clauses'}.issubset(constraint.keys()):
                return VerificationSpec.build_from_input_bounds(
                    constraint['lower'],
                    constraint['upper'],
                    constraint['clauses'],
                )
            if {'center', 'epsilon', 'clauses'}.issubset(constraint.keys()):
                center = constraint['center']
                epsilon = constraint['epsilon']
                clauses = constraint['clauses']
                center_t = torch.as_tensor(center).float()
                eps_t = torch.as_tensor(epsilon).float()
                if eps_t.ndim == 0:
                    eps_t = torch.full_like(center_t, float(eps_t))
                lower = center_t - eps_t
                upper = center_t + eps_t
                return VerificationSpec.build_from_input_bounds(
                    lower.unsqueeze(0),
                    upper.unsqueeze(0),
                    clauses,
                )
            if 'vnnlib_path' in constraint:
                input_shape = constraint.get('input_shape')
                return IOConstraints(vnnlib_path=constraint['vnnlib_path'], input_shape=input_shape)
            if {'input_vars', 'input_constraint'}.issubset(constraint.keys()):
                return IOConstraints(
                    input_vars=constraint.get('input_vars'),
                    output_vars=constraint.get('output_vars'),
                    input_constraint=constraint.get('input_constraint'),
                    output_constraint=constraint.get('output_constraint'),
                    force_simplify=constraint.get('force_simplify'),
                )
        raise TypeError('Unsupported constraint format.')

    def _attack(self,
                model_ori: torch.nn.Module,
                verified_status: str,
                verified_success: bool):
        """Run PGD attack if enabled and supported."""
        if arguments.Config['model']['with_jacobian']:
            model = LiRPANet(model_ori, in_size=[1, *self.vnnlib_handler.input_shape[1:]]).net
        else:
            model = model_ori
        device = arguments.Config['general']['device']
        x, c, rhs, or_spec_size, _, _ = self.vnnlib_handler.all_specs.get(device)
        attack_cfg = arguments.Config['attack']
        general_attack_enabled = bool(attack_cfg.get('general_attack', False))
        original_pgd_loss = attack_cfg.get('pgd_loss')
        original_restarts = attack_cfg.get('pgd_restarts')
        original_batch_size = attack_cfg.get('pgd_batch_size')

        def _run_attack_once():
            return attack(model, x, c, rhs, or_spec_size, self.vnnlib_handler.vnnlib,
                          verified_status, verified_success)

        try:
            return _run_attack_once()
        except ValueError as exc:
            # Defensive fallback for potential return-arity mismatch in custom configs.
            if general_attack_enabled and "too many values to unpack" in str(exc):
                print("[warn] PGD attack failed due to incompatible PGD loss return format under general attack.")
            raise
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try one conservative retry in general-attack mode before skipping.
            if general_attack_enabled and isinstance(attack_cfg.get('pgd_restarts'), int):
                curr_restarts = int(attack_cfg['pgd_restarts'])
                if curr_restarts > 1:
                    reduced_restarts = max(1, curr_restarts // 2)
                    attack_cfg['pgd_restarts'] = reduced_restarts
                    if isinstance(attack_cfg.get('pgd_batch_size'), int):
                        attack_cfg['pgd_batch_size'] = min(int(attack_cfg['pgd_batch_size']), reduced_restarts)
                    print(
                        f"[warn] PGD OOM, retrying with pgd_restarts={attack_cfg['pgd_restarts']}, "
                        f"pgd_batch_size={attack_cfg.get('pgd_batch_size')}."
                    )
                    try:
                        return _run_attack_once()
                    except torch.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            print("[warn] PGD attack skipped due to CUDA OOM; continuing without attack.")
            return verified_status, verified_success, None, None, None
        finally:
            # Keep global config stable across repeated solves.
            attack_cfg['pgd_loss'] = original_pgd_loss
            attack_cfg['pgd_restarts'] = original_restarts
            attack_cfg['pgd_batch_size'] = original_batch_size

    def _spec_violation(self, flat_input: torch.Tensor, output: torch.Tensor) -> Tuple[bool, Optional[float]]:
        """Check if a given input/output pair violates any loaded spec."""
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
