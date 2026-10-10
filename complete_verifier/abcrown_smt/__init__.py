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

import builtins as _builtins
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import os
import io
import sys
import contextlib
import warnings
import math
import numbers
import tempfile
import yaml

from auto_LiRPA.bound_ops import JacobianOP

_torch_jacobian_original = getattr(torch, 'jacobian', None)


def _auto_lirpa_jacobian(*args, **kwargs):
    if len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
        output, input_tensor = args
        jac = JacobianOP.apply(output, input_tensor)
        return jac.view(jac.shape[0], -1)
    if _torch_jacobian_original is not None:
        return _torch_jacobian_original(*args, **kwargs)
    raise NotImplementedError("torch.jacobian does not support provided arguments.")


torch.jacobian = _auto_lirpa_jacobian  # type: ignore[attr-defined]
try:
    import sympy
    from sympy.logic.boolalg import to_dnf as sympy_to_dnf
except ImportError:  # pragma: no cover - optional dependency
    sympy = None
    sympy_to_dnf = None


# Defaults for the internal ABCROWN run.
DEFAULT_DNF_TIMEOUT: float = 20.0
DEFAULT_DNF_BATCH_SIZE: int = 32
_DEFAULT_NEGATION_EPS: float = 1e-6
_DEFAULT_DISCRETE_MAX_NODES: int = 4096
# Match dReal's default integer domain: [-INT_MAX, INT_MAX].
_DREAL_INT_BOUND: float = float((1 << 31) - 1)
_VAR_TYPES: Dict[str, str] = {}


def _register_var_type(name: str, var_type: str) -> None:
    if not name:
        return
    _VAR_TYPES[name] = str(var_type).lower()

def set_dnf_defaults(timeout: Optional[float] = None, batch_size: Optional[int] = None):
    """Set default timeout and batch size used by the solver."""
    global DEFAULT_DNF_TIMEOUT, DEFAULT_DNF_BATCH_SIZE
    if timeout is not None:
        DEFAULT_DNF_TIMEOUT = float(timeout)
    if batch_size is not None:
        DEFAULT_DNF_BATCH_SIZE = int(batch_size)


class Config:
    """dReal-compatible configuration holder for the ABCROWN SMT interface."""

    kDefaultPrecision: float = 1e-3
    kDefaultNloptFtolRel: float = 1e-6
    kDefaultNloptFtolAbs: float = 1e-6
    kDefaultNloptMaxEval: int = 100
    kDefaultNloptMaxTime: float = 0.01

    class SatDefaultPhase(Enum):
        FALSE = 0
        TRUE = 1
        JEROSLOW_WANG = 2
        RANDOM_INITIAL_PHASE = 3

        def __str__(self) -> str:  # pragma: no cover - stringify helper
            names = {
                Config.SatDefaultPhase.FALSE: "False",
                Config.SatDefaultPhase.TRUE: "True",
                Config.SatDefaultPhase.JEROSLOW_WANG: "Jeroslow-Wang",
                Config.SatDefaultPhase.RANDOM_INITIAL_PHASE: "Random Initial Phase",
            }
            return names[self]

    Brancher = Callable[[Any, Any, Any, Any], int]

    def __init__(self, **kwargs: Any):
        self._precision: float = float(self.kDefaultPrecision)
        self._produce_models: bool = False
        self._use_polytope: bool = False
        self._use_polytope_in_forall: bool = False
        self._use_worklist_fixpoint: bool = False
        self._use_local_optimization: bool = False
        self._dump_theory_literals: bool = False
        self._number_of_jobs: int = 1
        self._stack_left_box_first: bool = False
        self._smtlib2_compliant: bool = False
        self._nlopt_ftol_rel: float = float(self.kDefaultNloptFtolRel)
        self._nlopt_ftol_abs: float = float(self.kDefaultNloptFtolAbs)
        self._nlopt_maxeval: int = int(self.kDefaultNloptMaxEval)
        self._nlopt_maxtime: float = float(self.kDefaultNloptMaxTime)
        self._sat_default_phase: Config.SatDefaultPhase = (
            Config.SatDefaultPhase.JEROSLOW_WANG
        )
        self._random_seed: int = 0
        self._brancher: Optional[Config.Brancher] = None

        for name, value in kwargs.items():
            if not hasattr(self.__class__, name):
                raise TypeError(f"Unknown Config option '{name}'")
            setattr(self, name, value)

    # ------------------------------------------------------------------
    # Helper conversion utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _as_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral):
            return bool(value)
        raise TypeError(f"{name} expects a boolean value")

    @staticmethod
    def _as_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} expects an integer, got bool")
        if isinstance(value, numbers.Integral):
            return int(value)
        raise TypeError(f"{name} expects an integer value")

    @staticmethod
    def _as_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} expects a floating-point value, got bool")
        if isinstance(value, numbers.Real):
            return float(value)
        raise TypeError(f"{name} expects a floating-point value")

    @classmethod
    def _coerce_sat_default_phase(cls, value: Any) -> "Config.SatDefaultPhase":
        if isinstance(value, cls.SatDefaultPhase):
            return value
        if isinstance(value, numbers.Integral):
            return cls.SatDefaultPhase(int(value))
        if isinstance(value, str):
            normalized = value.replace("_", "").replace("-", "").replace(" ", "").lower()
            mapping = {
                "false": cls.SatDefaultPhase.FALSE,
                "true": cls.SatDefaultPhase.TRUE,
                "jeroslowwang": cls.SatDefaultPhase.JEROSLOW_WANG,
                "randominitialphase": cls.SatDefaultPhase.RANDOM_INITIAL_PHASE,
            }
            if normalized in mapping:
                return mapping[normalized]
        raise TypeError("sat_default_phase expects an enum, int, or known string")

    # ------------------------------------------------------------------
    # Properties mirroring dReal's Config API
    # ------------------------------------------------------------------
    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, value: Any) -> None:
        val = self._as_float(value, "precision")
        if val <= 0:
            raise ValueError("precision must be positive")
        self._precision = val

    @property
    def produce_models(self) -> bool:
        return self._produce_models

    @produce_models.setter
    def produce_models(self, value: Any) -> None:
        self._produce_models = self._as_bool(value, "produce_models")

    @property
    def use_polytope(self) -> bool:
        return self._use_polytope

    @use_polytope.setter
    def use_polytope(self, value: Any) -> None:
        self._use_polytope = self._as_bool(value, "use_polytope")

    @property
    def use_polytope_in_forall(self) -> bool:
        return self._use_polytope_in_forall

    @use_polytope_in_forall.setter
    def use_polytope_in_forall(self, value: Any) -> None:
        self._use_polytope_in_forall = self._as_bool(value, "use_polytope_in_forall")

    @property
    def use_worklist_fixpoint(self) -> bool:
        return self._use_worklist_fixpoint

    @use_worklist_fixpoint.setter
    def use_worklist_fixpoint(self, value: Any) -> None:
        self._use_worklist_fixpoint = self._as_bool(value, "use_worklist_fixpoint")

    @property
    def use_local_optimization(self) -> bool:
        return self._use_local_optimization

    @use_local_optimization.setter
    def use_local_optimization(self, value: Any) -> None:
        self._use_local_optimization = self._as_bool(value, "use_local_optimization")

    @property
    def dump_theory_literals(self) -> bool:
        return self._dump_theory_literals

    @dump_theory_literals.setter
    def dump_theory_literals(self, value: Any) -> None:
        self._dump_theory_literals = self._as_bool(value, "dump_theory_literals")

    @property
    def number_of_jobs(self) -> int:
        return self._number_of_jobs

    @number_of_jobs.setter
    def number_of_jobs(self, value: Any) -> None:
        val = self._as_int(value, "number_of_jobs")
        if val <= 0:
            raise ValueError("number_of_jobs must be positive")
        self._number_of_jobs = val

    @property
    def stack_left_box_first(self) -> bool:
        return self._stack_left_box_first

    @stack_left_box_first.setter
    def stack_left_box_first(self, value: Any) -> None:
        self._stack_left_box_first = self._as_bool(value, "stack_left_box_first")

    @property
    def smtlib2_compliant(self) -> bool:
        return self._smtlib2_compliant

    @smtlib2_compliant.setter
    def smtlib2_compliant(self, value: Any) -> None:
        self._smtlib2_compliant = self._as_bool(value, "smtlib2_compliant")

    @property
    def brancher(self) -> Optional[Brancher]:
        return self._brancher

    @brancher.setter
    def brancher(self, value: Optional[Brancher]) -> None:
        if value is not None and not callable(value):
            raise TypeError("brancher expects a callable or None")
        self._brancher = value

    @property
    def nlopt_ftol_rel(self) -> float:
        return self._nlopt_ftol_rel

    @nlopt_ftol_rel.setter
    def nlopt_ftol_rel(self, value: Any) -> None:
        self._nlopt_ftol_rel = self._as_float(value, "nlopt_ftol_rel")

    @property
    def nlopt_ftol_abs(self) -> float:
        return self._nlopt_ftol_abs

    @nlopt_ftol_abs.setter
    def nlopt_ftol_abs(self, value: Any) -> None:
        self._nlopt_ftol_abs = self._as_float(value, "nlopt_ftol_abs")

    @property
    def nlopt_maxeval(self) -> int:
        return self._nlopt_maxeval

    @nlopt_maxeval.setter
    def nlopt_maxeval(self, value: Any) -> None:
        self._nlopt_maxeval = self._as_int(value, "nlopt_maxeval")

    @property
    def nlopt_maxtime(self) -> float:
        return self._nlopt_maxtime

    @nlopt_maxtime.setter
    def nlopt_maxtime(self, value: Any) -> None:
        self._nlopt_maxtime = self._as_float(value, "nlopt_maxtime")

    @property
    def sat_default_phase(self) -> "Config.SatDefaultPhase":
        return self._sat_default_phase

    @sat_default_phase.setter
    def sat_default_phase(self, value: Any) -> None:
        self._sat_default_phase = self._coerce_sat_default_phase(value)

    @property
    def random_seed(self) -> int:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Any) -> None:
        val = self._as_int(value, "random_seed")
        if val < 0:
            raise ValueError("random_seed must be non-negative")
        self._random_seed = val

    def copy(self) -> "Config":
        return Config(
            precision=self.precision,
            produce_models=self.produce_models,
            use_polytope=self.use_polytope,
            use_polytope_in_forall=self.use_polytope_in_forall,
            use_worklist_fixpoint=self.use_worklist_fixpoint,
            use_local_optimization=self.use_local_optimization,
            dump_theory_literals=self.dump_theory_literals,
            number_of_jobs=self.number_of_jobs,
            stack_left_box_first=self.stack_left_box_first,
            smtlib2_compliant=self.smtlib2_compliant,
            nlopt_ftol_rel=self.nlopt_ftol_rel,
            nlopt_ftol_abs=self.nlopt_ftol_abs,
            nlopt_maxeval=self.nlopt_maxeval,
            nlopt_maxtime=self.nlopt_maxtime,
            sat_default_phase=self.sat_default_phase,
            random_seed=self.random_seed,
            brancher=self.brancher,
        )

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return (
            "Config("
            f"precision = {self.precision}, "
            f"produce_model = {self.produce_models}, "
            f"use_polytope = {self.use_polytope}, "
            f"use_polytope_in_forall = {self.use_polytope_in_forall}, "
            f"use_worklist_fixpoint = {self.use_worklist_fixpoint}, "
            f"use_local_optimization = {self.use_local_optimization}, "
            f"dump_theory_literals = {self.dump_theory_literals}, "
            f"number_of_jobs = {self.number_of_jobs}, "
            f"nlopt_ftol_rel = {self.nlopt_ftol_rel}, "
            f"nlopt_ftol_abs = {self.nlopt_ftol_abs}, "
            f"nlopt_maxeval = {self.nlopt_maxeval}, "
            f"nlopt_maxtime = {self.nlopt_maxtime}, "
            f"sat_default_phase = {self.sat_default_phase}, "
            f"random_seed = {self.random_seed}"
            ")"
        )

    __repr__ = __str__

class ConstraintType(Enum):
    """Supported constraint types."""
    LE = "<="  # Less than or equal
    GE = ">="  # Greater than or equal
    EQ = "=="  # Equal


class Expr:
    """Expression node with operator overloading (builds an AST)."""
    def __init__(self, kind: str, args=None, name: Optional[str] = None, value: Optional[float] = None):
        self.kind = kind  # 'var', 'const', 'add', 'sub', 'mul', 'div', 'pow', 'neg', 'sin', 'cos', ...
        self.args = [] if args is None else args
        self.name = name
        self.value = value

    # Coerce inputs into Expr
    @staticmethod
    def to_expr(v):
        if isinstance(v, Expr):
            return v
        if isinstance(v, (int, float)):
            return Expr('const', value=float(v))
        return Variable(str(v))

    # Pretty printing
    def __str__(self):
        if self.kind == 'var':
            return self.name or 'var'
        if self.kind == 'const':
            return str(self.value)
        if self.kind == 'neg':
            return f"(-{self.args[0]})"
        if self.kind == 'abs':
            return f"abs({self.args[0]})"
        if self.kind == 'tan':
            return f"tan({self.args[0]})"
        if self.kind == 'atan':
            return f"atan({self.args[0]})"
        if self.kind in {'min', 'max'}:
            return f"{self.kind}({self.args[0]}, {self.args[1]})"
        if self.kind in {'add','sub','mul','div','pow'}:
            op = {'add':'+','sub':'-','mul':'*','div':'/','pow':'**'}[self.kind]
            return f"({self.args[0]} {op} {self.args[1]})"
        if self.kind == 'differentiate':
            return f"d({self.args[0]})/d({self.args[1]})"
        return f"{self.kind}({self.args[0]})"

    __repr__ = __str__

    # Arithmetic overloads
    def __add__(self, other):
        return Expr('add', [self, Expr.to_expr(other)])
    def __radd__(self, other):
        return Expr('add', [Expr.to_expr(other), self])
    def __sub__(self, other):
        return Expr('sub', [self, Expr.to_expr(other)])
    def __rsub__(self, other):
        return Expr('sub', [Expr.to_expr(other), self])
    def __mul__(self, other):
        return Expr('mul', [self, Expr.to_expr(other)])
    def __rmul__(self, other):
        return Expr('mul', [Expr.to_expr(other), self])
    def __truediv__(self, other):
        return Expr('div', [self, Expr.to_expr(other)])
    def __rtruediv__(self, other):
        return Expr('div', [Expr.to_expr(other), self])
    def __pow__(self, other):
        return Expr('pow', [self, Expr.to_expr(other)])
    def __neg__(self):
        return Expr('neg', [self])

    # Comparison overloads return Constraint
    def __le__(self, other):
        return Constraint(left=self, operator=ConstraintType.LE, right=Expr.to_expr(other))
    def __ge__(self, other):
        return Constraint(left=self, operator=ConstraintType.GE, right=Expr.to_expr(other))
    def __eq__(self, other):  # type: ignore[override]
        return Constraint(left=self, operator=ConstraintType.EQ, right=Expr.to_expr(other))

    def Differentiate(self, var: 'Expr') -> 'Expr':
        return differentiate(self, var)


@dataclass
class Constraint:
    """Single constraint: left <op> right."""
    left: Expr
    operator: ConstraintType
    right: Expr
    def __str__(self):
        return f"{self.left} {self.operator.value} {self.right}"


class Interval:
    """Closed interval with helpers mimicking dReal's Interval API."""

    def __init__(self, lower: float, upper: float):
        self._lower = float(lower)
        self._upper = float(upper)

    def lb(self) -> float:
        return self._lower

    def ub(self) -> float:
        return self._upper

    def mid(self) -> float:
        return 0.5 * (self._lower + self._upper)

    def diam(self) -> float:
        return self._upper - self._lower

    def __iter__(self):
        yield self._lower
        yield self._upper

    def __repr__(self) -> str:
        return f"[{self._lower}, {self._upper}]"


class Box:
    """Minimal Box container compatible with dReal's pretty-printing."""

    def __init__(self, intervals: Dict[str, List[float]], order: Optional[List[str]] = None):
        self._intervals: Dict[str, Interval] = {}
        for name, bounds in intervals.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Interval for {name} must be a pair of (lower, upper)")
            self._intervals[name] = Interval(bounds[0], bounds[1])
        self._order: List[str] = []
        if order is not None:
            for name in order:
                if name in self._intervals and name not in self._order:
                    self._order.append(name)
        for name in self._intervals.keys():
            if name not in self._order:
                self._order.append(name)

    def _resolve_key(self, key: Union[int, str, Any]) -> str:
        if isinstance(key, int):
            if key < 0 or key >= len(self._order):
                raise IndexError("Box index out of range")
            return self._order[key]
        if hasattr(key, "name"):
            return getattr(key, "name")
        return str(key)

    def __getitem__(self, key: Union[int, str, Any]) -> Interval:
        name = self._resolve_key(key)
        if name not in self._intervals:
            raise KeyError(name)
        return self._intervals[name]

    def keys(self):
        return self._intervals.keys()

    def items(self):
        return self._intervals.items()

    def values(self):
        return self._intervals.values()

    def __str__(self) -> str:
        def _fmt(val: Any) -> str:
            try:
                return f"{float(val):.17g}"
            except Exception:
                return str(val)

        lines: List[str] = []
        for name in self._order:
            if name.startswith("Y_"):
                continue
            interval = self._intervals.get(name)
            if interval is None:
                continue
            lines.append(f"{name} : [{_fmt(interval.lb())}, {_fmt(interval.ub())}]")
        return "\n".join(lines)

    __repr__ = __str__

@dataclass(frozen=True)
class Formula:
    """Minimal boolean formula wrapper mimicking dReal formulas."""

    op: str
    args: Tuple[Any, ...] = ()

    @staticmethod
    def ensure(value: Any) -> "Formula":
        if isinstance(value, Formula):
            return value
        if isinstance(value, Constraint):
            return Formula('atom', (value,))
        if isinstance(value, bool):
            return Formula('true') if value else Formula('false')
        if value is None:
            raise TypeError("Unsupported operand None for logical formula")
        raise TypeError(f"Unsupported operand type {type(value)!r} for logical formula")


def _dnf_cross_product(a: List[DNFClause], b: List[DNFClause]) -> List[DNFClause]:
    if not a or not b:
        return []
    result: List[DNFClause] = []
    for ca in a:
        for cb in b:
            result.append(DNFClause(constraints=list(ca.constraints) + list(cb.constraints)))
    return result


def _negate_constraint_to_dnf(constraint: Constraint) -> List[DNFClause]:
    left_expr = Expr.to_expr(constraint.left)
    right_expr = Expr.to_expr(constraint.right)
    if constraint.operator == ConstraintType.LE:
        # not (L <= R)  =>  R + eps <= L
        new_left = right_expr + _DEFAULT_NEGATION_EPS
        new_constraint = Constraint(left=Expr.to_expr(new_left), operator=ConstraintType.LE, right=left_expr)
        return [DNFClause([new_constraint])]
    if constraint.operator == ConstraintType.GE:
        # not (L >= R)  =>  L + eps <= R
        new_left = left_expr + _DEFAULT_NEGATION_EPS
        new_constraint = Constraint(left=Expr.to_expr(new_left), operator=ConstraintType.LE, right=right_expr)
        return [DNFClause([new_constraint])]
    if constraint.operator == ConstraintType.EQ:
        # not (L == R)  =>  (L + eps <= R) OR (R + eps <= L)
        left_greater = Constraint(left=Expr.to_expr(left_expr + _DEFAULT_NEGATION_EPS),
                                  operator=ConstraintType.LE,
                                  right=right_expr)
        right_greater = Constraint(left=Expr.to_expr(right_expr + _DEFAULT_NEGATION_EPS),
                                   operator=ConstraintType.LE,
                                   right=left_expr)
        return [DNFClause([left_greater]), DNFClause([right_greater])]
    raise NotImplementedError(f"Unsupported constraint operator {constraint.operator}")


def _formula_to_dnf(formula: Formula, negated: bool = False) -> List[DNFClause]:
    f = Formula.ensure(formula)
    if f.op == 'atom':
        constraint = f.args[0]
        return _negate_constraint_to_dnf(constraint) if negated else [DNFClause([constraint])]
    if f.op == 'true':
        return [] if negated else [DNFClause([])]
    if f.op == 'false':
        return [DNFClause([])] if negated else []
    if f.op == 'not':
        return _formula_to_dnf(Formula.ensure(f.args[0]), not negated)

    if f.op == 'and':
        if not negated:
            dnf: List[DNFClause] = [DNFClause([])]
            for arg in f.args:
                dnf = _dnf_cross_product(dnf, _formula_to_dnf(Formula.ensure(arg), False))
            return dnf
        result: List[DNFClause] = []
        for arg in f.args:
            result.extend(_formula_to_dnf(Formula.ensure(arg), True))
        return result

    if f.op == 'or':
        if not negated:
            result: List[DNFClause] = []
            for arg in f.args:
                result.extend(_formula_to_dnf(Formula.ensure(arg), False))
            return result
        dnf: List[DNFClause] = [DNFClause([])]
        for arg in f.args:
            dnf = _dnf_cross_product(dnf, _formula_to_dnf(Formula.ensure(arg), True))
        return dnf

    raise NotImplementedError(f"Unsupported formula operator '{f.op}'")


def _dnf_clauses_to_specification(clauses: List[DNFClause]) -> Optional[DNFSpecification]:
    if not clauses:
        return None

    def _const_value(expr: Expr) -> Optional[float]:
        node = Expr.to_expr(expr)
        if node.kind == 'const':
            return float(node.value)
        if node.kind == 'neg':
            val = _const_value(node.args[0])
            return None if val is None else -val
        if node.kind in {'add', 'sub', 'mul', 'div', 'pow'}:
            a = _const_value(node.args[0])
            b = _const_value(node.args[1])
            if a is None or b is None:
                return None
            if node.kind == 'add':
                return a + b
            if node.kind == 'sub':
                return a - b
            if node.kind == 'mul':
                return a * b
            if node.kind == 'div':
                return a / b
            if node.kind == 'pow':
                return a ** b
        return None

    def _var_with_offset(expr: Expr) -> Optional[Tuple[str, float]]:
        node = Expr.to_expr(expr)
        if node.kind == 'var':
            return node.name, 0.0
        if node.kind in {'add', 'sub'}:
            left = Expr.to_expr(node.args[0])
            right = Expr.to_expr(node.args[1])
            if left.kind == 'var':
                const = _const_value(right)
                if const is None:
                    return None
                offset = const if node.kind == 'add' else -const
                return left.name, float(offset)
            if node.kind == 'add' and right.kind == 'var':
                const = _const_value(left)
                if const is None:
                    return None
                return right.name, float(const)
        return None

    def _flatten_add_terms(expr: Expr, sign: float = 1.0) -> List[Tuple[Expr, float]]:
        node = Expr.to_expr(expr)
        if node.kind == 'add':
            terms: List[Tuple[Expr, float]] = []
            for arg in node.args:
                terms.extend(_flatten_add_terms(arg, sign))
            return terms
        if node.kind == 'sub':
            terms = _flatten_add_terms(node.args[0], sign)
            terms.extend(_flatten_add_terms(node.args[1], -sign))
            return terms
        if node.kind == 'neg':
            return _flatten_add_terms(node.args[0], -sign)
        return [(node, sign)]

    def _match_var_square(expr: Expr) -> Optional[str]:
        node = Expr.to_expr(expr)
        if node.kind == 'mul':
            left = Expr.to_expr(node.args[0])
            right = Expr.to_expr(node.args[1])
            if left.kind == 'var' and right.kind == 'var' and left.name == right.name:
                return left.name
        if node.kind == 'pow':
            base = Expr.to_expr(node.args[0])
            exp = Expr.to_expr(node.args[1])
            if base.kind == 'var':
                exp_val = _const_value(exp)
                if exp_val is not None and _builtins.abs(exp_val - 2.0) <= 1e-9:
                    return base.name
        return None

    def _match_coeff_square(expr: Expr) -> Optional[Tuple[str, float]]:
        node = Expr.to_expr(expr)
        var_name = _match_var_square(node)
        if var_name is not None:
            return var_name, 1.0
        if node.kind == 'mul':
            left = Expr.to_expr(node.args[0])
            right = Expr.to_expr(node.args[1])
            left_const = _const_value(left)
            if left_const is not None:
                var_name = _match_var_square(right)
                if var_name is not None:
                    return var_name, float(left_const)
            right_const = _const_value(right)
            if right_const is not None:
                var_name = _match_var_square(left)
                if var_name is not None:
                    return var_name, float(right_const)
        return None

    def _sum_squares_coeffs(expr: Expr) -> Optional[Tuple[Dict[str, float], float]]:
        terms = _flatten_add_terms(expr)
        coeffs: Dict[str, float] = {}
        const_term = 0.0
        for term, sign in terms:
            const_val = _const_value(term)
            if const_val is not None:
                const_term += sign * float(const_val)
                continue
            matched = _match_coeff_square(term)
            if matched is None or sign < 0:
                return None
            name, coeff = matched
            if coeff <= 0:
                return None
            coeffs[name] = coeffs.get(name, 0.0) + sign * coeff
        if not coeffs:
            return None
        return coeffs, const_term

    def _tighten_from_sum_squares(expr: Expr, bound: Optional[float]) -> None:
        if bound is None or not math.isfinite(bound):
            return
        parsed = _sum_squares_coeffs(expr)
        if parsed is None:
            return
        coeffs, const_term = parsed
        remaining = bound - const_term
        if not math.isfinite(remaining) or remaining < 0.0:
            return
        for name, coeff in coeffs.items():
            if coeff <= 0:
                continue
            limit = remaining / coeff
            if limit < 0.0:
                continue
            bound_abs = math.sqrt(limit)
            if not math.isfinite(bound_abs):
                continue
            consider_lb(name, -bound_abs)
            consider_ub(name, bound_abs)

    # Collect variable names in deterministic encounter order
    seen: set[str] = set()
    ordered_vars: List[str] = []

    def collect_vars(expr: Expr) -> None:
        node = Expr.to_expr(expr)
        if node.kind == 'var':
            if node.name not in seen:
                seen.add(node.name)
                ordered_vars.append(node.name)
            return
        for sub in node.args:
            collect_vars(sub)

    for cl in clauses:
        for ct in cl.constraints:
            collect_vars(ct.left)
            collect_vars(ct.right)

    variables: Dict[str, List[float]] = {name: [-float('inf'), float('inf')] for name in ordered_vars}

    lb_seen: Dict[str, bool] = {name: False for name in ordered_vars}
    ub_seen: Dict[str, bool] = {name: False for name in ordered_vars}
    lb_min: Dict[str, float] = {name: float('inf') for name in ordered_vars}
    ub_max: Dict[str, float] = {name: -float('inf') for name in ordered_vars}

    def consider_lb(var: str, value: float) -> None:
        lb_seen[var] = True
        if value < lb_min[var]:
            lb_min[var] = value

    def consider_ub(var: str, value: float) -> None:
        ub_seen[var] = True
        if value > ub_max[var]:
            ub_max[var] = value

    for cl in clauses:
        for ct in cl.constraints:
            L, R, op = ct.left, ct.right, ct.operator
            left_var = _var_with_offset(L)
            right_var = _var_with_offset(R)
            left_const = _const_value(L)
            right_const = _const_value(R)

            if op == ConstraintType.LE:
                # (var + k) <= c  ==> var <= c - k
                if left_var is not None and right_const is not None:
                    name, offset = left_var
                    consider_ub(name, right_const - offset)
                # c <= (var + k)  ==> var >= c - k
                elif left_const is not None and right_var is not None:
                    name, offset = right_var
                    consider_lb(name, left_const - offset)
                _tighten_from_sum_squares(L, right_const)
            elif op == ConstraintType.GE:
                # (var + k) >= c  ==> var >= c - k
                if left_var is not None and right_const is not None:
                    name, offset = left_var
                    consider_lb(name, right_const - offset)
                # c >= (var + k)  ==> var <= c - k
                elif left_const is not None and right_var is not None:
                    name, offset = right_var
                    consider_ub(name, left_const - offset)
                _tighten_from_sum_squares(R, left_const)
            else:  # EQ
                if left_var is not None and right_const is not None:
                    name, offset = left_var
                    val = right_const - offset
                    consider_lb(name, val)
                    consider_ub(name, val)
                elif left_const is not None and right_var is not None:
                    name, offset = right_var
                    val = left_const - offset
                    consider_lb(name, val)
                    consider_ub(name, val)
                _tighten_from_sum_squares(L, right_const)
                _tighten_from_sum_squares(R, left_const)

    for name in ordered_vars:
        lo, hi = variables[name]
        if lb_seen[name]:
            lo = lb_min[name]
        if ub_seen[name]:
            hi = ub_max[name]
        variables[name] = [lo, hi]
        if math.isfinite(lo) and math.isfinite(hi) and lo > hi:
            return None

    discrete_types: Dict[str, str] = {}
    for name in ordered_vars:
        vtype = _VAR_TYPES.get(name)
        if vtype in {"int", "bool", "binary"}:
            discrete_types[name] = vtype
    return DNFSpecification(variables=variables, clauses=clauses, discrete_types=discrete_types)


@dataclass
class DNFClause:
    """Conjunction of constraints (a single DNF clause)."""
    constraints: List[Constraint]
    
    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)


@dataclass
class DNFSpecification:
    """DNF specification: OR over clauses with variable bounds."""
    clauses: List[DNFClause]
    variables: Dict[str, List[float]]  # var_name -> [lower, upper]
    discrete_types: Dict[str, str] = field(default_factory=dict)
    
    def add_clause(self, clause: DNFClause):
        self.clauses.append(clause)
    
    def add_variable(self, name: str, lower: float, upper: float):
        self.variables[name] = (lower, upper)


@dataclass
class CNFClause:
    """Disjunction of constraints (a single CNF clause)."""
    constraints: List[Constraint]

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)


@dataclass
class CNFSpecification:
    """CNF specification: AND over CNF clauses with variable bounds."""
    clauses: List[CNFClause]
    variables: Dict[str, List[float]]

    def add_clause(self, clause: CNFClause):
        self.clauses.append(clause)

    def add_variable(self, name: str, lower: float, upper: float):
        self.variables[name] = (lower, upper)


def _constraint_signature(constraint: Constraint) -> str:
    return f"{constraint}"


def _sympy_expr_to_symbol_conjunctions(expr) -> List[List[Any]]:
    if sympy is None:
        raise ImportError("Sympy is required to convert CNF inputs.")
    if expr is sympy.false:
        return []
    if expr is sympy.true:
        return [[]]
    if isinstance(expr, sympy.Symbol):
        return [[expr]]
    if isinstance(expr, sympy.And):
        clause: List[Any] = []
        for arg in expr.args:
            sub = _sympy_expr_to_symbol_conjunctions(arg)
            if not sub:
                return []
            if len(sub) == 1:
                clause.extend(sub[0])
            else:
                # to_dnf should avoid this path, but flatten best-effort
                for item in sub:
                    clause.extend(item)
        return [clause]
    if isinstance(expr, sympy.Or):
        clauses: List[List[Any]] = []
        for arg in expr.args:
            clauses.extend(_sympy_expr_to_symbol_conjunctions(arg))
        return clauses
    return [[expr]]


def _cnf_clauses_to_dnf_clauses(cnf_clauses: List[CNFClause]) -> List[DNFClause]:
    if sympy_to_dnf is None:
        raise ImportError("Sympy is required to convert CNF inputs to DNF. Please install sympy.")

    if not cnf_clauses:
        return [DNFClause(constraints=[])]

    literal_to_symbol: Dict[str, Any] = {}
    symbol_to_constraint: Dict[Any, Constraint] = {}
    sympy_clause_literals: List[List[Any]] = []

    for clause in cnf_clauses:
        if not clause.constraints:
            return []
        literals: List[Any] = []
        for constraint in clause.constraints:
            key = _constraint_signature(constraint)
            symbol = literal_to_symbol.get(key)
            if symbol is None:
                symbol = sympy.Symbol(f'c{len(literal_to_symbol)}')
                literal_to_symbol[key] = symbol
                symbol_to_constraint[symbol] = constraint
            literals.append(symbol)
        sympy_clause_literals.append(literals)

    expr = sympy.true
    for literals in sympy_clause_literals:
        clause_expr = sympy.false
        for symbol in literals:
            clause_expr = sympy.Or(clause_expr, symbol)
        expr = sympy.And(expr, clause_expr)

    dnf_expr = sympy_to_dnf(expr, simplify=True)
    symbol_clauses = _sympy_expr_to_symbol_conjunctions(dnf_expr)
    if not symbol_clauses:
        return []

    dnf_clauses: List[DNFClause] = []
    for symbol_clause in symbol_clauses:
        constraints = [symbol_to_constraint[sym] for sym in symbol_clause if sym is not sympy.true]
        dnf_clauses.append(DNFClause(constraints=constraints))

    return dnf_clauses


class Variable(Expr):
    """Variable node."""
    def __init__(self, name: str, var_type: str = "Real"):
        super().__init__('var', name=name)
        self.var_type = var_type
        _register_var_type(name, var_type)
    # Comparison overloads returning Constraint
    def __le__(self, other):
        return Constraint(left=self, operator=ConstraintType.LE, right=Expr.to_expr(other))
    def __ge__(self, other):
        return Constraint(left=self, operator=ConstraintType.GE, right=Expr.to_expr(other))
    def __eq__(self, other):  # type: ignore[override]
        return Constraint(left=self, operator=ConstraintType.EQ, right=Expr.to_expr(other))


def Expression(value: Any = 0.0) -> Expr:
    """Factory mirroring dReal's Expression(), returning an Expr tree."""
    return Expr.to_expr(value)


def logical_and(*formulas: Any) -> Formula:
    """Associative logical AND producing a Formula tree."""
    filtered = [Formula.ensure(f) for f in formulas if f is not None]
    if not filtered:
        raise ValueError("logical_and expects at least one operand")
    if len(filtered) == 1:
        return filtered[0]
    return Formula('and', tuple(filtered))


def logical_or(*formulas: Any) -> Formula:
    """Associative logical OR producing a Formula tree."""
    filtered = [Formula.ensure(f) for f in formulas if f is not None]
    if not filtered:
        raise ValueError("logical_or expects at least one operand")
    if len(filtered) == 1:
        return filtered[0]
    return Formula('or', tuple(filtered))


def logical_not(formula: Any) -> Formula:
    return Formula('not', (Formula.ensure(formula),))


def logical_imply(lhs: Any, rhs: Any) -> Formula:
    return logical_or(logical_not(lhs), rhs)


def sin(v: Expr) -> Expr:     return Expr('sin', [Expr.to_expr(v)])
def cos(v: Expr) -> Expr:     return Expr('cos', [Expr.to_expr(v)])
def tan(v: Expr) -> Expr:     return Expr('tan', [Expr.to_expr(v)])
def atan(v: Expr) -> Expr:    return Expr('atan', [Expr.to_expr(v)])
def abs(v: Expr) -> Expr:     return Expr('abs', [Expr.to_expr(v)])
def min(a: Expr, b: Expr) -> Expr: return Expr('min', [Expr.to_expr(a), Expr.to_expr(b)])
def max(a: Expr, b: Expr) -> Expr: return Expr('max', [Expr.to_expr(a), Expr.to_expr(b)])
def exp(v: Expr) -> Expr:     return Expr('exp', [Expr.to_expr(v)])
def log(v: Expr) -> Expr:     return Expr('log', [Expr.to_expr(v)])
def tanh(v: Expr) -> Expr:    return Expr('tanh', [Expr.to_expr(v)])
def sigmoid(v: Expr) -> Expr: return Expr('sigmoid', [Expr.to_expr(v)])
def sqrt(v: Expr) -> Expr:    return Expr('sqrt', [Expr.to_expr(v)])
def differentiate(expr: Expr, var: Union[Expr, str]) -> Expr:
    expr_node = Expr.to_expr(expr)
    var_node = Expr.to_expr(var)
    if var_node.kind != 'var':
        raise TypeError("differentiate expects the second argument to be a variable.")
    try:
        return _differentiate_expr_tree(expr_node, var_node.name)
    except NotImplementedError:
        return Expr('differentiate', [expr_node, var_node])


def _intern_expr_tree(expr: Expr, pool: Dict[Tuple[Any, ...], Expr]) -> Expr:
    expr = Expr.to_expr(expr)
    if expr.kind == 'var':
        key = ('var', expr.name)
        node = pool.get(key)
        if node is None:
            pool[key] = expr
            node = expr
        return node
    if expr.kind == 'const':
        value = float(expr.value)
        key = ('const', repr(value))
        node = pool.get(key)
        if node is None:
            pool[key] = expr
            node = expr
        return node
    interned_args = []
    arg_keys = []
    for arg in expr.args:
        interned_arg = _intern_expr_tree(arg, pool)
        interned_args.append(interned_arg)
        arg_keys.append((interned_arg.kind, id(interned_arg)))
    value_key = None if expr.value is None else repr(float(expr.value))
    key = (expr.kind, expr.name, value_key, tuple(arg_keys))
    node = pool.get(key)
    if node is None:
        node = Expr(expr.kind, args=interned_args, name=expr.name, value=expr.value)
        pool[key] = node
    return node


def _intern_clause_exprs(clauses: List[DNFClause]) -> List[DNFClause]:
    pool: Dict[Tuple[Any, ...], Expr] = {}
    interned_clauses: List[DNFClause] = []
    for clause in clauses:
        new_constraints: List[Constraint] = []
        for constraint in clause.constraints:
            left = _intern_expr_tree(constraint.left, pool)
            right = _intern_expr_tree(constraint.right, pool)
            new_constraints.append(Constraint(left=left, operator=constraint.operator, right=right))
        interned_clauses.append(DNFClause(constraints=new_constraints))
    return interned_clauses


def _const_expr(value: float) -> Expr:
    return Expr('const', value=float(value))


def _differentiate_expr_tree(expr: Expr, var_name: str) -> Expr:
    expr = Expr.to_expr(expr)
    kind = expr.kind
    if kind == 'const':
        return _const_expr(0.0)
    if kind == 'var':
        return _const_expr(1.0) if expr.name == var_name else _const_expr(0.0)
    if kind == 'neg':
        return -_differentiate_expr_tree(expr.args[0], var_name)
    if kind == 'add':
        return (_differentiate_expr_tree(expr.args[0], var_name) +
                _differentiate_expr_tree(expr.args[1], var_name))
    if kind == 'sub':
        return (_differentiate_expr_tree(expr.args[0], var_name) -
                _differentiate_expr_tree(expr.args[1], var_name))
    if kind == 'mul':
        u, v = expr.args
        du = _differentiate_expr_tree(u, var_name)
        dv = _differentiate_expr_tree(v, var_name)
        return du * Expr.to_expr(v) + Expr.to_expr(u) * dv
    if kind == 'div':
        u, v = expr.args
        du = _differentiate_expr_tree(u, var_name)
        dv = _differentiate_expr_tree(v, var_name)
        numerator = du * Expr.to_expr(v) - Expr.to_expr(u) * dv
        denominator = Expr.to_expr(v) * Expr.to_expr(v)
        return numerator / denominator
    if kind == 'pow':
        base, exponent = expr.args
        exponent = Expr.to_expr(exponent)
        base_expr = Expr.to_expr(base)
        if exponent.kind == 'const':
            power = float(exponent.value)
            if power == 0.0:
                return _const_expr(0.0)
            new_exponent = _const_expr(power - 1.0)
            return (_const_expr(power) * base_expr ** new_exponent *
                    _differentiate_expr_tree(base, var_name))
        if base_expr.kind == 'const':
            return _const_expr(0.0)
        raise NotImplementedError("Symbolic differentiation for general power expressions is not supported.")
    if kind == 'exp':
        inner = expr.args[0]
        return exp(Expr.to_expr(inner)) * _differentiate_expr_tree(inner, var_name)
    if kind == 'log':
        inner = expr.args[0]
        return _differentiate_expr_tree(inner, var_name) / Expr.to_expr(inner)
    if kind == 'sin':
        inner = expr.args[0]
        return cos(Expr.to_expr(inner)) * _differentiate_expr_tree(inner, var_name)
    if kind == 'cos':
        inner = expr.args[0]
        return -sin(Expr.to_expr(inner)) * _differentiate_expr_tree(inner, var_name)
    if kind == 'tanh':
        inner = expr.args[0]
        tanh_expr = tanh(Expr.to_expr(inner))
        return (_const_expr(1.0) - tanh_expr * tanh_expr) * _differentiate_expr_tree(inner, var_name)
    if kind == 'sigmoid':
        inner = expr.args[0]
        sig_expr = sigmoid(Expr.to_expr(inner))
        return sig_expr * (_const_expr(1.0) - sig_expr) * _differentiate_expr_tree(inner, var_name)
    if kind == 'sqrt':
        inner = expr.args[0]
        return _differentiate_expr_tree(inner, var_name) / (_const_expr(2.0) * sqrt(Expr.to_expr(inner)))
    raise NotImplementedError(f"Symbolic differentiation not implemented for kind {kind}")


class AbcrownDNFSolver:
    """DNF solver using ABCROWN's BaB over an exported ONNX graph."""
    
    def __init__(self, spec: DNFSpecification, verbose: bool = False):
        self.spec = spec
        self.var_names = list(spec.variables.keys())
        self.verbose = verbose

    @contextlib.contextmanager
    def _suppress_output(self):
        """Silence noisy stdout/stderr and warnings from dependencies."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            saved_out, saved_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            try:
                yield
            finally:
                sys.stdout, sys.stderr = saved_out, saved_err

    class _DNFExportNet(nn.Module):
        """Compute Y = min_k max_i r_{k,i} from DNF constraints."""
        def __init__(self, var_names: List[str], clauses: List[DNFClause]):
            super().__init__()
            self.var_names = var_names
            self.clauses = _intern_clause_exprs(clauses)
            self.indices = {n: i for i, n in enumerate(var_names)}
            self._var_cache: Dict[str, torch.Tensor] = {}
            self._var_index_cache: Dict[str, torch.Tensor] = {}
            self._expr_cache: Dict[int, torch.Tensor] = {}
            self._var_cols: Optional[Tuple[torch.Tensor, ...]] = None

        def _prepare_inputs(self, x: torch.Tensor) -> None:
            self._var_cols = tuple(torch.split(x, 1, dim=1))

        def _var_tensor(self, name: str, x: torch.Tensor) -> torch.Tensor:
            cached = self._var_cache.get(name)
            if cached is not None:
                return cached
            idx = self.indices[name]
            if self._var_cols is not None and idx < len(self._var_cols):
                tensor = self._var_cols[idx]
                self._var_cache[name] = tensor
                return tensor
            index = self._var_index_cache.get(name)
            if index is None or index.device != x.device:
                index = torch.tensor([idx], dtype=torch.long, device=x.device)
                self._var_index_cache[name] = index
            tensor = torch.index_select(x, dim=1, index=index)
            self._var_cache[name] = tensor
            return tensor

        def _const_tensor(self, value: float, x: torch.Tensor) -> torch.Tensor:
            if self.var_names:
                ref = self._var_tensor(self.var_names[0], x)
                return ref - ref + float(value)
            return torch.full((x.shape[0], 1), float(value), dtype=x.dtype, device=x.device)

        def _eval_expr_jacobian(self, base_expr: Expr, var_expr: Expr, x: torch.Tensor) -> torch.Tensor:
            sub = self._eval_expr(base_expr, x)
            if var_expr.kind != 'var':
                raise NotImplementedError("Jacobian fallback expects variable second argument.")
            if base_expr.kind == 'const':
                return torch.zeros_like(sub)
            if base_expr.kind == 'var':
                if base_expr.name == var_expr.name:
                    return torch.ones_like(sub)
                return torch.zeros_like(sub)
            if var_expr.name not in self.indices:
                raise KeyError(f"Variable {var_expr.name} not found in specification.")
            var_tensor = self._var_tensor(var_expr.name, x)
            if var_tensor.numel() == 0:
                return torch.zeros_like(sub)
            print('[debug jacobian input] base', base_expr.kind, 'sub shape', tuple(sub.shape), 'var shape', tuple(var_tensor.shape))
            jac = JacobianOP.apply(sub, var_tensor)
            batch = jac.shape[0]
            sub_flat = sub.reshape(batch, -1)
            var_flat = var_tensor.reshape(batch, -1)
            if var_flat.shape[1] == 0:
                return torch.zeros_like(sub)
            jac = jac.reshape(batch, sub_flat.shape[1], var_flat.shape[1])
            grad = jac[:, :, 0]
            print("[debug jacobian]", base_expr.kind, "sub", tuple(sub.shape), "var", tuple(var_tensor.shape), "jac", tuple(jac.shape), "grad", tuple(grad.shape))
            return grad.reshape_as(sub)

        def _eval_expr(self, e: Expr, x: torch.Tensor) -> torch.Tensor:
            cache_key = id(e)
            cached = self._expr_cache.get(cache_key)
            if cached is not None:
                return cached
            if e.kind == 'var':
                result = self._var_tensor(e.name, x)
            elif e.kind == 'const':
                result = self._const_tensor(float(e.value), x)
            elif e.kind == 'neg':
                result = -self._eval_expr(e.args[0], x)
            elif e.kind == 'abs':
                result = torch.abs(self._eval_expr(e.args[0], x))
            elif e.kind == 'tan':
                inner = self._eval_expr(e.args[0], x)
                result = torch.sin(inner) / torch.cos(inner)
            elif e.kind == 'min':
                result = torch.minimum(self._eval_expr(e.args[0], x), self._eval_expr(e.args[1], x))
            elif e.kind == 'max':
                result = torch.maximum(self._eval_expr(e.args[0], x), self._eval_expr(e.args[1], x))
            elif e.kind == 'add':
                result = self._eval_expr(e.args[0], x) + self._eval_expr(e.args[1], x)
            elif e.kind == 'sub':
                result = self._eval_expr(e.args[0], x) - self._eval_expr(e.args[1], x)
            elif e.kind == 'mul':
                result = self._eval_expr(e.args[0], x) * self._eval_expr(e.args[1], x)
            elif e.kind == 'div':
                result = self._eval_expr(e.args[0], x) / self._eval_expr(e.args[1], x)
            elif e.kind == 'pow':
                base = self._eval_expr(e.args[0], x)
                exp_arg = e.args[1]
                if isinstance(exp_arg, Expr) and exp_arg.kind == 'const':
                    result = base ** float(exp_arg.value)
                else:
                    result = base ** self._eval_expr(exp_arg, x)
            elif e.kind == 'differentiate':
                base_expr = Expr.to_expr(e.args[0])
                var_expr = Expr.to_expr(e.args[1])
                if var_expr.kind != 'var':
                    raise NotImplementedError("differentiate() expects a variable as the second argument.")
                result = self._eval_expr_jacobian(base_expr, var_expr, x)
            elif e.kind == 'sin':
                result = torch.sin(self._eval_expr(e.args[0], x))
            elif e.kind == 'cos':
                result = torch.cos(self._eval_expr(e.args[0], x))
            elif e.kind == 'atan':
                result = torch.atan(self._eval_expr(e.args[0], x))
            elif e.kind == 'tanh':
                result = torch.tanh(self._eval_expr(e.args[0], x))
            elif e.kind == 'sigmoid':
                result = torch.sigmoid(self._eval_expr(e.args[0], x))
            elif e.kind == 'exp':
                result = torch.exp(self._eval_expr(e.args[0], x))
            elif e.kind == 'log':
                result = torch.log(self._eval_expr(e.args[0], x))
            elif e.kind == 'sqrt':
                result = torch.sqrt(self._eval_expr(e.args[0], x))
            else:
                raise NotImplementedError(f"Unsupported expr kind: {e.kind}")
            self._expr_cache[cache_key] = result
            return result

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._var_cache = {}
            self._expr_cache = {}
            self._prepare_inputs(x)
            clause_vals = []
            for cl in self.clauses:
                res_list = []
                for c in cl.constraints:
                    vl = self._eval_expr(c.left, x)
                    vr = self._eval_expr(c.right, x)
                    delta = vl - vr
                    if c.operator == ConstraintType.LE:
                        res_list.append(delta)
                    elif c.operator == ConstraintType.GE:
                        res_list.append(-delta)
                    else:  # EQ as two-sided inequality
                        res_list.append(torch.abs(delta))
                if res_list:
                    res_tensor = torch.cat(res_list, dim=1)
                    yk, _ = torch.max(res_tensor, dim=1, keepdim=True)
                else:
                    yk = torch.full((x.shape[0], 1), float('-inf'), dtype=x.dtype, device=x.device)
                clause_vals.append(yk)
            if clause_vals:
                clause_tensor = torch.cat(clause_vals, dim=1)
                Y, _ = torch.min(clause_tensor, dim=1, keepdim=True)
                return Y
            return torch.full((x.shape[0], 1), float('+inf'), dtype=x.dtype, device=x.device)

    def _export_onnx_and_vnnlib(self, tmp_dir: str, y_slack: float = 0.0,
                                export_residual_vnnlib: bool = True) -> Tuple[str, str, List[str]]:
        os.makedirs(tmp_dir, exist_ok=True)
        onnx_path = os.path.join(tmp_dir, 'dnf_model.onnx')
        vnnlib_path = os.path.join(tmp_dir, 'dnf_spec.vnnlib')

        # Drop pure input-box constraints (x <= c, c <= x, ...): VNNLIB already enforces them.
        def is_box_constraint(ct: Constraint) -> bool:
            if ct.operator not in (ConstraintType.LE, ConstraintType.GE):
                return False
            L, R = ct.left, ct.right
            if L.kind == 'var' and R.kind == 'const':
                return True
            if L.kind == 'const' and R.kind == 'var':
                return True
            return False

        filtered_clauses: List[DNFClause] = []
        for cl in self.spec.clauses:
            kept = [ct for ct in cl.constraints if not is_box_constraint(ct)]
            filtered_clauses.append(DNFClause(kept))

        # If all clauses are empty after removing box constraints, short-circuit:
        # the problem reduces to checking the input box only, which is trivially SAT
        # with any point inside the box. We return a dummy ONNX/VNNLIB ensuring Y_0 <= y_slack.
        all_empty = all(len(cl.constraints) == 0 for cl in filtered_clauses)
        if all_empty:
            # Build a trivial 1-output network outputting constant 0.
            class _ZeroNet(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

            net = _ZeroNet().eval()
            dummy = torch.zeros(1, len(self.var_names), dtype=torch.float32)
            with self._suppress_output():
                torch.onnx.export(
                    net, dummy, onnx_path,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                    opset_version=18
                )

            with open(vnnlib_path, 'w') as f:
                n = len(self.var_names)
                for i in range(n):
                    f.write(f"(declare-const X_{i} Real)\n")
                f.write("(declare-const Y_0 Real)\n")
                # Input box constraints as-is (with sanitization below handled in normal path)
                for i, name in enumerate(self.var_names):
                    lo, hi = self.spec.variables[name]
                    safe_lo = lo if math.isfinite(lo) else -1e6
                    safe_hi = hi if math.isfinite(hi) else 1e6
                    f.write(f"(assert (>= X_{i} {safe_lo}))\n")
                    f.write(f"(assert (<= X_{i} {safe_hi}))\n")
                f.write(f"(assert (<= Y_0 {float(y_slack)}))\n")
            return onnx_path, vnnlib_path, self.var_names

        # Either export per-residual outputs (many Y_j) or an aggregated single output Y.
        residual_indices_per_clause: List[List[int]] = []
        if export_residual_vnnlib:
            class _DNFResidualNet(nn.Module):
                def __init__(self, var_names: List[str], clauses: List[DNFClause]):
                    super().__init__()
                    self.var_names = var_names
                    self.clauses = _intern_clause_exprs(clauses)
                    self.indices = {n: i for i, n in enumerate(var_names)}
                    self._var_cache: Dict[str, torch.Tensor] = {}
                    self._var_index_cache: Dict[str, torch.Tensor] = {}
                    self._expr_cache: Dict[int, torch.Tensor] = {}
                    self._var_cols: Optional[Tuple[torch.Tensor, ...]] = None

                def _prepare_inputs(self, x: torch.Tensor) -> None:
                    self._var_cols = tuple(torch.split(x, 1, dim=1))

                def _var_tensor(self, name: str, x: torch.Tensor) -> torch.Tensor:
                    cached = self._var_cache.get(name)
                    if cached is not None:
                        return cached
                    idx = self.indices[name]
                    if self._var_cols is not None and idx < len(self._var_cols):
                        tensor = self._var_cols[idx]
                        self._var_cache[name] = tensor
                        return tensor
                    index = self._var_index_cache.get(name)
                    if index is None or index.device != x.device:
                        index = torch.tensor([idx], dtype=torch.long, device=x.device)
                        self._var_index_cache[name] = index
                    tensor = torch.index_select(x, dim=1, index=index)
                    self._var_cache[name] = tensor
                    return tensor

                def _const_tensor(self, value: float, x: torch.Tensor) -> torch.Tensor:
                    if self.var_names:
                        ref = self._var_tensor(self.var_names[0], x)
                        return ref - ref + float(value)
                    return torch.full((x.shape[0], 1), float(value), dtype=x.dtype, device=x.device)

                def _eval_expr_jacobian(self, base_expr: Expr, var_expr: Expr, x: torch.Tensor) -> torch.Tensor:
                    sub = self._eval_expr(base_expr, x)
                    if var_expr.kind != 'var':
                        raise NotImplementedError("differentiate() expects a variable as the second argument.")
                    if base_expr.kind == 'const':
                        return torch.zeros_like(sub)
                    if base_expr.kind == 'var':
                        if base_expr.name == var_expr.name:
                            return torch.ones_like(sub)
                        return torch.zeros_like(sub)
                    if var_expr.name not in self.indices:
                        raise KeyError(f"Variable {var_expr.name} not found in specification.")
                    var_tensor = self._var_tensor(var_expr.name, x)
                    if var_tensor.numel() == 0:
                        return torch.zeros_like(sub)
                    print('[debug jacobian input] base', base_expr.kind, 'sub shape', tuple(sub.shape), 'var shape', tuple(var_tensor.shape))
                    jac = JacobianOP.apply(sub, var_tensor)
                    batch = jac.shape[0]
                    sub_flat = sub.reshape(batch, -1)
                    var_flat = var_tensor.reshape(batch, -1)
                    if var_flat.shape[1] == 0:
                        return torch.zeros_like(sub)
                    jac = jac.reshape(batch, sub_flat.shape[1], var_flat.shape[1])
                    grad = jac[:, :, 0]
                    print("[debug jacobian]", base_expr.kind, "sub", tuple(sub.shape), "var", tuple(var_tensor.shape), "jac", tuple(jac.shape), "grad", tuple(grad.shape))
                    return grad.reshape_as(sub)

                def _eval_expr(self, e: Expr, x: torch.Tensor) -> torch.Tensor:
                    cache_key = id(e)
                    cached = self._expr_cache.get(cache_key)
                    if cached is not None:
                        return cached
                    if e.kind == 'var':
                        result = self._var_tensor(e.name, x)
                    elif e.kind == 'const':
                        result = self._const_tensor(float(e.value), x)
                    elif e.kind == 'neg':
                        result = -self._eval_expr(e.args[0], x)
                    elif e.kind == 'abs':
                        result = torch.abs(self._eval_expr(e.args[0], x))
                    elif e.kind == 'tan':
                        inner = self._eval_expr(e.args[0], x)
                        result = torch.sin(inner) / torch.cos(inner)
                    elif e.kind == 'min':
                        result = torch.minimum(self._eval_expr(e.args[0], x), self._eval_expr(e.args[1], x))
                    elif e.kind == 'max':
                        result = torch.maximum(self._eval_expr(e.args[0], x), self._eval_expr(e.args[1], x))
                    elif e.kind == 'add':
                        result = self._eval_expr(e.args[0], x) + self._eval_expr(e.args[1], x)
                    elif e.kind == 'sub':
                        result = self._eval_expr(e.args[0], x) - self._eval_expr(e.args[1], x)
                    elif e.kind == 'mul':
                        result = self._eval_expr(e.args[0], x) * self._eval_expr(e.args[1], x)
                    elif e.kind == 'div':
                        result = self._eval_expr(e.args[0], x) / self._eval_expr(e.args[1], x)
                    elif e.kind == 'pow':
                        base = self._eval_expr(e.args[0], x)
                        exp_arg = e.args[1]
                        if isinstance(exp_arg, Expr) and exp_arg.kind == 'const':
                            result = base ** float(exp_arg.value)
                        else:
                            result = base ** self._eval_expr(exp_arg, x)
                    elif e.kind == 'differentiate':
                        base_expr = Expr.to_expr(e.args[0])
                        var_expr = Expr.to_expr(e.args[1])
                        if var_expr.kind != 'var':
                            raise NotImplementedError("differentiate() expects a variable as the second argument.")
                        result = self._eval_expr_jacobian(base_expr, var_expr, x)
                    elif e.kind == 'sin':
                        result = torch.sin(self._eval_expr(e.args[0], x))
                    elif e.kind == 'cos':
                        result = torch.cos(self._eval_expr(e.args[0], x))
                    elif e.kind == 'atan':
                        result = torch.atan(self._eval_expr(e.args[0], x))
                    elif e.kind == 'tanh':
                        result = torch.tanh(self._eval_expr(e.args[0], x))
                    elif e.kind == 'sigmoid':
                        result = torch.sigmoid(self._eval_expr(e.args[0], x))
                    elif e.kind == 'exp':
                        result = torch.exp(self._eval_expr(e.args[0], x))
                    elif e.kind == 'log':
                        result = torch.log(self._eval_expr(e.args[0], x))
                    elif e.kind == 'sqrt':
                        result = torch.sqrt(self._eval_expr(e.args[0], x))
                    else:
                        raise NotImplementedError(f"Unsupported expr kind: {e.kind}")
                    self._expr_cache[cache_key] = result
                    return result

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self._var_cache = {}
                    self._expr_cache = {}
                    self._prepare_inputs(x)
                    outs = []
                    for cl in self.clauses:
                        for c in cl.constraints:
                            vl = self._eval_expr(c.left, x)
                            vr = self._eval_expr(c.right, x)
                            delta = vl - vr
                            if c.operator == ConstraintType.LE:
                                outs.append(delta)
                            elif c.operator == ConstraintType.GE:
                                outs.append(-delta)
                            else:
                                abs_delta = torch.abs(delta)
                                outs.append(abs_delta)
                                outs.append(abs_delta)
                    if len(outs) == 0:
                        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
                    return torch.cat(outs, dim=1)

            # Map residual indices per clause for vnnlib.
            idx = 0
            for cl in filtered_clauses:
                ids = []
                for c in cl.constraints:
                    if c.operator in (ConstraintType.LE, ConstraintType.GE):
                        ids.append(idx); idx += 1
                    else:
                        ids.append(idx); idx += 1
                        ids.append(idx); idx += 1
                residual_indices_per_clause.append(ids)

            net = _DNFResidualNet(self.var_names, filtered_clauses).eval()
        else:
            net = self._DNFExportNet(self.var_names, filtered_clauses).eval()
        dummy = torch.zeros(1, len(self.var_names), dtype=torch.float32)
        with self._suppress_output():
            torch.onnx.export(
                net, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=18
            )

        # Enforce domains for restricted ops (log, sqrt).
        restricted_lb: Dict[str, float] = {name: self.spec.variables[name][0] for name in self.var_names}
        def tighten_from_expr(e: Expr):
            if e.kind in {'log','sqrt'}:
                arg = e.args[0]
                if arg.kind == 'var':
                    name = arg.name
                    if e.kind == 'log':
                        restricted_lb[name] = _builtins.max(restricted_lb.get(name, -float('inf')), 1e-6)
                    else:  # sqrt
                        restricted_lb[name] = _builtins.max(restricted_lb.get(name, -float('inf')), 0.0)
            for a in e.args:
                tighten_from_expr(a)
        for cl in self.spec.clauses:
            for c in cl.constraints:
                tighten_from_expr(c.left)
                tighten_from_expr(c.right)

        with open(vnnlib_path, 'w') as f:
            n = len(self.var_names)
            for i in range(n):
                f.write(f"(declare-const X_{i} Real)\n")
            if export_residual_vnnlib:
                # Declare outputs for each residual
                total_res = sum(len(ids) for ids in residual_indices_per_clause)
                if total_res == 0:
                    total_res = 1
                for j in range(total_res):
                    f.write(f"(declare-const Y_{j} Real)\n")
            else:
                f.write("(declare-const Y_0 Real)\n")
            # Input box constraints (sanitize infinities for VNNLIB)
            for i, name in enumerate(self.var_names):
                lo, hi = self.spec.variables[name]
                # Respect domains from restricted ops (log/sqrt): may raise the lower bound
                safe_lo = restricted_lb.get(name, lo)
                # Sanitize infinities: VNNLIB requires finite numbers
                if not math.isfinite(safe_lo):
                    safe_lo = -1e6
                safe_hi = hi
                if not math.isfinite(safe_hi):
                    safe_hi = 1e6
                f.write(f"(assert (>= X_{i} {safe_lo}))\n")
                f.write(f"(assert (<= X_{i} {safe_hi}))\n")
            if export_residual_vnnlib:
                # Constrain residual outputs. If any clause has no residuals, it's trivially true.
                has_empty_clause = any(len(ids) == 0 for ids in residual_indices_per_clause)
                if has_empty_clause:
                    f.write(f"(assert (<= Y_0 {float(y_slack)}))\n")
                else:
                    f.write("(assert (or ")
                    for ids in residual_indices_per_clause:
                        f.write("(and ")
                        for local_idx in ids:
                            f.write(f"(<= Y_{local_idx} {float(y_slack)})")
                        f.write(") ")
                    f.write("))\n")
            else:
                # Aggregated single-output mode
                f.write(f"(assert (<= Y_0 {float(y_slack)}))\n")

        return onnx_path, vnnlib_path, self.var_names

    def ExportOnnxAndVnnlib(
        self,
        out_dir: Optional[str] = None,
        y_slack: float = 0.0,
        export_residual_vnnlib: bool = False,
    ) -> Tuple[str, str, List[str]]:
        """Export ONNX/VNNLIB to a persistent directory."""
        if out_dir is None:
            script_dir: Optional[str] = None
            try:
                if sys.argv and sys.argv[0]:
                    script_path = os.path.abspath(sys.argv[0])
                    if os.path.isfile(script_path):
                        script_dir = os.path.dirname(script_path)
            except Exception:
                script_dir = None
            base_dir = script_dir or os.getcwd()
            out_dir = os.path.join(base_dir, 'out_onnx_vnnlib_dir')
        else:
            out_dir = os.path.abspath(out_dir)
        return self._export_onnx_and_vnnlib(
            out_dir, y_slack=y_slack, export_residual_vnnlib=export_residual_vnnlib
        )

    def _write_config_yaml(self, out_dir: str, onnx_path: str, vnnlib_path: str,
                           device: Optional[str] = None, timeout: float = 10.0,
                           complete_verifier: str = 'bab', solver_batch_size: Optional[int] = None) -> str:
        """Write a minimal ABCROWN config.yaml for the exported ONNX/VNNLIB."""
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, 'config.yaml')
        if device is None:
            device = 'cuda'
        results_file = os.path.join(out_dir, 'result.txt')
        cex_path = os.path.join(out_dir, 'cex.txt')
        cfg = {
            'general': {
                'device': device,
                'complete_verifier': complete_verifier,
                'save_adv_example': True,
                'results_file': results_file,
                'root_path': '.',
                'double_fp': True,
            },
            'model': {
                'onnx_path': onnx_path,
                'with_jacobian': True,
            },
            'specification': {
                'norm': float('inf'),
                'vnnlib_path': vnnlib_path,
            },
            'bab': {
                # 'timeout': float(timeout),
                'clip_n_verify': {
                    'clip_input_domain': {
                        'enabled': True,
                    },
                    'clip_interm_domain': {
                        'enabled': True,
                    },
                },
                'branching': {
                    'method': 'sb',
                    'input_split': {
                        'enable': True,
                    },
                },
                'cut': {
                    'enabled': False,
                },
            },
            'attack': {
                'general_attack': True,
                # 'pgd_order': 'before',
                # 'pgd_steps': 100,
                # 'pgd_restarts': 100,
                # 'pgd_batch_size': 100000000,
                'pgd_early_stop': True,
                # 'pgd_lr_decay': 0.99,
                'cex_path': cex_path,
            },
            'debug': {
                'save_minimal_config': os.path.join(out_dir, 'dnf_effective_config.yaml'),
            },
        }
        solver_cfg = {}
        solver_cfg['auto_enlarge_batch_size'] = True
        # if solver_batch_size is not None:
        #     solver_cfg['batch_size'] = int(solver_batch_size)
        # else:
        #     solver_cfg['batch_size'] = 80000
        # solver_cfg['bound_prop_method'] = 'crown'
        # solver_cfg['init_bound_prop_method'] = 'crown'
        cfg['solver'] = solver_cfg

        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return cfg_path

    def _run_abcrown(self, onnx_path: str, vnnlib_path: str, timeout: float = DEFAULT_DNF_TIMEOUT,
                     results_file: Optional[str] = None, cex_path: Optional[str] = None,
                     solver_batch_size: Optional[int] = None) -> Tuple[str, Optional[Dict[int, float]]]:
        try:
            from ..abcrown import ABCROWN  # type: ignore
        except ImportError:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            from abcrown import ABCROWN  # type: ignore
        tmp_dir = os.path.dirname(onnx_path)
        # Always emit a ready-to-run config.yaml so users can call abcrown like tests.
        cfg_path = self._write_config_yaml(
            out_dir=tmp_dir, onnx_path=onnx_path, vnnlib_path=vnnlib_path,
            timeout=timeout, solver_batch_size=solver_batch_size)

        # Run via config
        if results_file is None:
            results_file = os.path.join(os.path.dirname(vnnlib_path), 'result.txt')
        if cex_path is None:
            cex_path = os.path.join(os.path.dirname(vnnlib_path), 'cex.txt')
        args = [f"--config={cfg_path}"]
        if solver_batch_size is not None:
            args.append(f"--batch_size={int(solver_batch_size)}")
        # Allow overriding results/cex paths if needed.
        args += [f"--cex_path={cex_path}", f"--results_file={results_file}"]
        args = [a for a in args if a is not None]

        if os.environ.get('DNF_LOG', '').lower() in ('1','true','yes','on'):
            abc = ABCROWN(args=args)
            _ = abc.main()
        else:
            with self._suppress_output():
                abc = ABCROWN(args=args)
                _ = abc.main()

        status = 'unknown'
        try:
            with open(results_file, 'r') as rf:
                status = rf.readline().strip()
        except Exception:
            pass

        adv = None
        if status == 'sat':
            try:
                with open(cex_path, 'r') as cf:
                    content = cf.read()
                xs: Dict[int, float] = {}
                for m in re.finditer(r"\(X_(\d+)\s+([\-\+eE0-9\.]+)\)", content):
                    xs[int(m.group(1))] = float(m.group(2))
                adv = xs if xs else None
            except Exception:
                adv = None
        return status, adv

    def CheckSatisfiability(
        self,
        delta: float = 0.001,
        pretty: bool = True,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        if config is not None:
            delta = float(config.precision)
            if config.random_seed != 0:
                torch.manual_seed(int(config.random_seed))
        solver_batch_size = DEFAULT_DNF_BATCH_SIZE if config is None else _builtins.max(1, int(config.number_of_jobs))

        if len(self.spec.clauses) == 0:
            return {"verification_result": "UNSAT", "reason": "No constraints"}

        # Tiny slack helps numerics on equalities.
        y_slack = _builtins.max(delta, 1e-6)

        def _sanity_for_point(point: Dict[str, float]) -> Optional[Dict[str, Any]]:
            if os.environ.get("ABCROWN_SANITY") in {"0", "false", "False"}:
                return None
            try:
                y_val = self._compute_y_value(point)
            except Exception:
                y_val = float('inf')
            try:
                y_val_torch = self._compute_y_value_torch(point)
            except Exception:
                y_val_torch = float('inf')
            math_ok = math.isfinite(y_val) and y_val <= y_slack
            torch_ok = math.isfinite(y_val_torch) and y_val_torch <= y_slack
            return {
                "math_residual": float(y_val),
                "torch_residual": float(y_val_torch),
                "tolerance": float(y_slack),
                "passed": bool(math_ok and torch_ok),
            }

        # Early short-circuit: if there exists at least one clause containing only
        # input-box constraints (var <= const / const <= var / var >= const / const >= var),
        # construct a concrete witness inside that clause and return SAT directly.
        def _is_box_constraint(ct: Constraint) -> bool:
            if ct.operator not in (ConstraintType.LE, ConstraintType.GE):
                return False
            L, R = ct.left, ct.right
            return ((L.kind == 'var' and R.kind == 'const') or (L.kind == 'const' and R.kind == 'var'))

        def _clause_box_bounds(clause: DNFClause) -> Optional[Dict[str, Tuple[float, float]]]:
            bounds: Dict[str, List[float]] = {}
            for ct in clause.constraints:
                if not _is_box_constraint(ct):
                    return None
                L, R, op = ct.left, ct.right, ct.operator
                if op == ConstraintType.LE:
                    if L.kind == 'var' and R.kind == 'const':
                        b = bounds.setdefault(L.name, [-float('inf'), float('inf')])
                        b[1] = _builtins.min(b[1], float(R.value))
                    elif L.kind == 'const' and R.kind == 'var':
                        b = bounds.setdefault(R.name, [-float('inf'), float('inf')])
                        b[0] = _builtins.max(b[0], float(L.value))
                else:  # GE
                    if L.kind == 'var' and R.kind == 'const':
                        b = bounds.setdefault(L.name, [-float('inf'), float('inf')])
                        b[0] = _builtins.max(b[0], float(R.value))
                    elif L.kind == 'const' and R.kind == 'var':
                        b = bounds.setdefault(R.name, [-float('inf'), float('inf')])
                        b[1] = _builtins.min(b[1], float(L.value))
            # Validate bounds (no empty interval)
            for name, (lo, hi) in bounds.items():
                if lo > hi:
                    return None
            return {k: (v[0], v[1]) for k, v in bounds.items()}

        for clause in self.spec.clauses:
            clause_bounds = _clause_box_bounds(clause)
            if clause_bounds is not None:
                # Build a witness inside this clause.
                point: Dict[str, float] = {}
                # Use clause-specific bounds where available; fall back to global variable hulls.
                for name in self.var_names:
                    if name in clause_bounds:
                        lo, hi = clause_bounds[name]
                    else:
                        lo, hi = self.spec.variables.get(name, [-1.0, 1.0])
                    # Sanitize infinities
                    if not math.isfinite(lo):
                        lo = -1.0
                    if not math.isfinite(hi):
                        hi = 1.0
                    val = 0.5 * (lo + hi)
                    point[name] = float(val)
                # Return SAT with zero-width intervals as witness
                intervals = {k: [v, v] for k, v in point.items()}
                sanity_details = _sanity_for_point(point)
                res = {"verification_result": "SAT", "intervals": intervals, "sanity_check": sanity_details}
                if pretty:
                    pretty_result: Dict[str, Any] = {
                        "result": "SAT",
                        "intervals": res["intervals"],
                        "constraints": [],
                    }
                    if sanity_details is not None:
                        pretty_result["sanity_check"] = sanity_details
                    return pretty_result
                return res

        # Heuristic: try midpoint of current bounds as a quick witness.
        point: Dict[str, float] = {}
        point_lo: Dict[str, float] = {}
        point_hi: Dict[str, float] = {}
        for name in self.var_names:
            lo, hi = self.spec.variables.get(name, [-1.0, 1.0])
            if not math.isfinite(lo):
                lo = -1.0
            if not math.isfinite(hi):
                hi = 1.0
            point[name] = 0.5 * (float(lo) + float(hi))
            point_lo[name] = float(lo)
            point_hi[name] = float(hi)

        def _point_satisfies(point: Dict[str, float], tol: float) -> bool:
            for cl in self.spec.clauses:
                clause_ok = True
                for c in cl.constraints:
                    try:
                        vl = self._eval_expr_math(c.left, point)
                        vr = self._eval_expr_math(c.right, point)
                    except Exception:
                        clause_ok = False
                        break
                    if c.operator == ConstraintType.LE:
                        if (vl - vr) > tol:
                            clause_ok = False
                            break
                    elif c.operator == ConstraintType.GE:
                        if (vr - vl) > tol:
                            clause_ok = False
                            break
                    else:
                        if _builtins.abs(vl - vr) > tol:
                            clause_ok = False
                            break
                if clause_ok:
                    return True
            return False

        # Try a few cheap corner points before running ABCROWN.
        for candidate in (point, point_lo, point_hi):
            if _point_satisfies(candidate, y_slack):
                intervals = {k: [float(v), float(v)] for k, v in candidate.items()}
                sanity_details = _sanity_for_point(candidate)
                res = {"verification_result": "SAT", "intervals": intervals, "sanity_check": sanity_details}
                if pretty:
                    pretty_result: Dict[str, Any] = {
                        "result": "SAT",
                        "intervals": res["intervals"],
                        "constraints": [],
                    }
                    if sanity_details is not None:
                        pretty_result["sanity_check"] = sanity_details
                    return pretty_result
                return res
        try:
            y_val = self._compute_y_value(point)
        except Exception:
            y_val = float('inf')
        if math.isfinite(y_val) and y_val <= y_slack:
            intervals = {k: [float(v), float(v)] for k, v in point.items()}
            sanity_details = _sanity_for_point(point)
            res = {"verification_result": "SAT", "intervals": intervals, "sanity_check": sanity_details}
            if pretty:
                pretty_result: Dict[str, Any] = {
                    "result": "SAT",
                    "intervals": res["intervals"],
                    "constraints": [],
                }
                if sanity_details is not None:
                    pretty_result["sanity_check"] = sanity_details
                return pretty_result
            return res
        sanity_details: Optional[Dict[str, Any]] = None
        with tempfile.TemporaryDirectory(prefix='abcrown_dnf_') as tmp_dir:
            onnx_path, vnnlib_path, var_order = self._export_onnx_and_vnnlib(tmp_dir, y_slack=y_slack)
            status, adv = self._run_abcrown(
                onnx_path,
                vnnlib_path,
                timeout=DEFAULT_DNF_TIMEOUT,
                solver_batch_size=solver_batch_size,
            )

            if status == 'sat' and adv is not None:
                # Validate witness numerically against the DNF constraints.
                point = {var_order[i]: adv[i] for i in adv if i < len(var_order)}
                for idx, name in enumerate(var_order):
                    if name not in point:
                        lo, hi = self.spec.variables.get(name, [-1.0, 1.0])
                        if not math.isfinite(lo):
                            lo = -1.0
                        if not math.isfinite(hi):
                            hi = 1.0
                        point[name] = 0.5 * (lo + hi)
                tolerance = y_slack
                if os.environ.get("ABCROWN_SANITY") not in {"0", "false", "False"}:
                    try:
                        y_val = self._compute_y_value(point)
                    except Exception:
                        y_val = float('inf')
                    try:
                        y_val_torch = self._compute_y_value_torch(point)
                    except Exception:
                        y_val_torch = float('inf')

                    math_ok = math.isfinite(y_val) and y_val <= tolerance
                    torch_ok = math.isfinite(y_val_torch) and y_val_torch <= tolerance
                    sanity_details = {
                        "math_residual": float(y_val),
                        "torch_residual": float(y_val_torch),
                        "tolerance": float(tolerance),
                        "passed": bool(math_ok and torch_ok),
                    }
                else:
                    sanity_details = None

                if sanity_details and sanity_details.get("passed", False):
                    # Output witness values as zero-width intervals [v, v].
                    intervals = {k: [float(v), float(v)] for k, v in point.items()}
                    result = {
                        "verification_result": "SAT",
                        "intervals": intervals,
                        "sanity_check": sanity_details,
                        "var_order": list(var_order),
                        "witness": point,
                    }
                else:
                    result = {
                        "verification_result": "UNKNOWN",
                        "intervals": {},
                        "sanity_check": sanity_details,
                        "var_order": list(var_order),
                        "witness": point,
                    }
            elif status == 'unsat':
                result = {
                    "verification_result": "UNSAT",
                    "intervals": {},
                    "sanity_check": sanity_details,
                    "var_order": list(var_order),
                    "witness": None,
                }
            else:
                result = {
                    "verification_result": "UNKNOWN",
                    "intervals": {},
                    "sanity_check": sanity_details,
                    "var_order": list(var_order),
                    "witness": None,
                }

        if pretty:
            pretty_result: Dict[str, Any] = {
                "result": result["verification_result"],
                "intervals": result.get("intervals", {}),
                "constraints": [],
                "var_order": result.get("var_order", list(var_order)),
            }
            if result.get("sanity_check") is not None:
                pretty_result["sanity_check"] = result["sanity_check"]
            return pretty_result

        return result
    

    def _autograd_inputs(self, point: Dict[str, float], dtype: torch.dtype = torch.float64
                        ) -> Tuple[Dict[str, torch.Tensor], torch.dtype, torch.device]:
        device = torch.device('cpu')
        inputs: Dict[str, torch.Tensor] = {}
        for name in self.var_names:
            if name in point:
                val = float(point[name])
            else:
                bounds = self.spec.variables.get(name, [-1.0, 1.0])
                lo, hi = float(bounds[0]), float(bounds[1])
                if not math.isfinite(lo):
                    lo = -1.0
                if not math.isfinite(hi):
                    hi = 1.0
                val = 0.5 * (lo + hi)
            inputs[name] = torch.tensor(val, dtype=dtype, device=device, requires_grad=True)
        return inputs, dtype, device

    def _eval_expr_autograd_recursive(
        self, e: Expr, inputs: Dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if e.kind == 'var':
            return inputs[e.name]
        if e.kind == 'const':
            return torch.tensor(float(e.value), dtype=dtype, device=device)
        if e.kind == 'neg':
            return -self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device)
        if e.kind == 'abs':
            return torch.abs(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'tan':
            inner = self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device)
            return torch.sin(inner) / torch.cos(inner)
        if e.kind == 'min':
            return torch.minimum(
                self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device),
                self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device),
            )
        if e.kind == 'max':
            return torch.maximum(
                self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device),
                self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device),
            )
        if e.kind == 'add':
            return self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device) + \
                   self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device)
        if e.kind == 'sub':
            return self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device) - \
                   self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device)
        if e.kind == 'mul':
            return self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device) * \
                   self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device)
        if e.kind == 'div':
            return self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device) / \
                   self._eval_expr_autograd_recursive(e.args[1], inputs, dtype, device)
        if e.kind == 'pow':
            base = self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device)
            exp_arg = e.args[1]
            if isinstance(exp_arg, Expr) and exp_arg.kind == 'const':
                return base ** float(exp_arg.value)
            return base ** self._eval_expr_autograd_recursive(exp_arg, inputs, dtype, device)
        if e.kind == 'differentiate':
            var_expr = e.args[1]
            if not isinstance(var_expr, Expr) or var_expr.kind != 'var':
                raise TypeError("differentiate expects a variable as its second argument.")
            if var_expr.name not in inputs:
                raise KeyError(f"Variable {var_expr.name} not found for differentiation.")
            sub = self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device)
            sub_scalar = sub
            if sub_scalar.ndim > 0:
                sub_scalar = sub_scalar.sum()
            grad = torch.autograd.grad(
                sub_scalar,
                inputs[var_expr.name],
                grad_outputs=torch.ones_like(sub_scalar),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if grad is None:
                return torch.zeros((), dtype=dtype, device=device)
            return grad
        if e.kind == 'sin':
            return torch.sin(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'cos':
            return torch.cos(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'atan':
            return torch.atan(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'tanh':
            return torch.tanh(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'sigmoid':
            return torch.sigmoid(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'exp':
            return torch.exp(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'log':
            return torch.log(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        if e.kind == 'sqrt':
            return torch.sqrt(self._eval_expr_autograd_recursive(e.args[0], inputs, dtype, device))
        raise NotImplementedError(f"Unsupported expression kind for autograd evaluation: {e.kind}")

    def _eval_expr_math_derivative(
        self, expr_node: Expr, var_node: Expr, point: Dict[str, float]
    ) -> float:
        if var_node.kind != 'var':
            raise TypeError("differentiate expects a variable as its second argument.")
        inputs, dtype, device = self._autograd_inputs(point)
        if var_node.name not in inputs:
            raise KeyError(f"Variable {var_node.name} not available for differentiation.")
        value = self._eval_expr_autograd_recursive(expr_node, inputs, dtype, device)
        value_scalar = value
        if value_scalar.ndim > 0:
            value_scalar = value_scalar.sum()
        grad = torch.autograd.grad(
            value_scalar,
            inputs[var_node.name],
            grad_outputs=torch.ones_like(value_scalar),
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )[0]
        if grad is None:
            return 0.0
        return float(grad.detach().cpu().item())

    def _eval_expr_math(self, e: Expr, point: Dict[str, float]) -> float:
        if e.kind == 'var':
            return float(point[e.name])
        if e.kind == 'const':
            return float(e.value)
        if e.kind == 'neg':
            return -self._eval_expr_math(e.args[0], point)
        if e.kind == 'abs':
            return _builtins.abs(self._eval_expr_math(e.args[0], point))
        if e.kind == 'tan':
            inner = self._eval_expr_math(e.args[0], point)
            return math.sin(inner) / math.cos(inner)
        if e.kind == 'min':
            return _builtins.min(self._eval_expr_math(e.args[0], point),
                                 self._eval_expr_math(e.args[1], point))
        if e.kind == 'max':
            return _builtins.max(self._eval_expr_math(e.args[0], point),
                                 self._eval_expr_math(e.args[1], point))
        if e.kind in {'add','sub','mul','div','pow'}:
            a = self._eval_expr_math(e.args[0], point)
            b = self._eval_expr_math(e.args[1], point)
            if e.kind == 'add': return a + b
            if e.kind == 'sub': return a - b
            if e.kind == 'mul': return a * b
            if e.kind == 'div': return a / b
            if e.kind == 'pow': return a ** b
        if e.kind == 'differentiate':
            var_expr = Expr.to_expr(e.args[1])
            if var_expr.kind != 'var':
                raise NotImplementedError("differentiate() expects a variable as the second argument.")
            return self._eval_expr_math_derivative(Expr.to_expr(e.args[0]), var_expr, point)
        a = self._eval_expr_math(e.args[0], point)
        if e.kind == 'sin': return math.sin(a)
        if e.kind == 'cos': return math.cos(a)
        if e.kind == 'atan': return math.atan(a)
        if e.kind == 'tanh': return math.tanh(a)
        if e.kind == 'sigmoid': return 1.0/(1.0+math.exp(-a))
        if e.kind == 'exp': return math.exp(a)
        if e.kind == 'log': return math.log(a)
        if e.kind == 'sqrt': return math.sqrt(a)
        raise NotImplementedError(e.kind)

    def _compute_y_value(self, point: Dict[str, float]) -> float:
        if not self.spec.clauses:
            return float('inf')
        clause_vals: List[float] = []
        for cl in self.spec.clauses:
            res_list: List[float] = []
            for c in cl.constraints:
                vl = self._eval_expr_math(c.left, point)
                vr = self._eval_expr_math(c.right, point)
                if c.operator == ConstraintType.LE:
                    res_list.append(vl - vr)
                elif c.operator == ConstraintType.GE:
                    res_list.append(vr - vl)
                else:  # EQ two-sided
                    res_list.append(vl - vr)
                    res_list.append(vr - vl)
            yk = _builtins.max(res_list) if res_list else float('-inf')
            clause_vals.append(yk)
        return _builtins.min(clause_vals) if clause_vals else float('inf')

    def _compute_y_value_torch(self, point: Dict[str, float]) -> float:
        """Evaluate the clause residual aggregator using torch operators."""
        if not self.spec.clauses:
            return float('inf')
        inputs, dtype, device = self._autograd_inputs(point)
        clause_vals: List[torch.Tensor] = []
        for cl in self.spec.clauses:
            residuals: List[torch.Tensor] = []
            for c in cl.constraints:
                vl = self._eval_expr_autograd_recursive(c.left, inputs, dtype, device)
                vr = self._eval_expr_autograd_recursive(c.right, inputs, dtype, device)
                if c.operator == ConstraintType.LE:
                    residuals.append(vl - vr)
                elif c.operator == ConstraintType.GE:
                    residuals.append(vr - vl)
                else:
                    residuals.append(vl - vr)
                    residuals.append(vr - vl)
            if residuals:
                clause_val = residuals[0]
                for res in residuals[1:]:
                    clause_val = torch.maximum(clause_val, res)
            else:
                clause_val = torch.tensor(float('-inf'), dtype=dtype, device=device)
            clause_vals.append(clause_val)
        if not clause_vals:
            return float('inf')
        agg = clause_vals[0]
        for val in clause_vals[1:]:
            agg = torch.minimum(agg, val)
        return float(agg.detach().cpu().item())


# Convenience functions to mimic dReal's interface
def create_variable(name: str, var_type: str = "Real") -> Variable:
    """Create a variable, mimicking dreal's Variable constructor."""
    return Variable(name, var_type)


def And(*args) -> Union[DNFClause, List[CNFClause]]:
    """Conjunction helper compatible with both DNF and CNF styles."""
    constraints: List[Constraint] = []
    cnf_clauses: List[CNFClause] = []
    for item in args:
        if isinstance(item, Constraint):
            constraints.append(item)
        elif isinstance(item, CNFClause):
            cnf_clauses.append(item)
        else:
            # Ignore legacy/None inputs for backward compatibility.
            continue

    if cnf_clauses:
        if constraints:
            raise TypeError("Cannot mix raw constraints with CNFClause in And().")
        return cnf_clauses

    return DNFClause(constraints)


def Or(*clauses: Union[DNFClause, Constraint]) -> Union[List[DNFClause], CNFClause]:
    """Disjunction helper supporting both DNF and CNF construction."""
    if all(isinstance(cl, DNFClause) for cl in clauses):
        return list(clauses)  # DNF: list of clauses

    constraints: List[Constraint] = []
    for item in clauses:
        if isinstance(item, Constraint):
            constraints.append(item)
        elif item is None:
            continue
        else:
            raise TypeError("Or() expects DNFClause or Constraint inputs.")
    return CNFClause(constraints)


def CNFOr(*args) -> CNFClause:
    """Backward-compatible alias for Or()."""
    warnings.warn("CNFOr is deprecated; use Or instead.", DeprecationWarning, stacklevel=2)
    return Or(*args)  # type: ignore[return-value]


def CNFAnd(*clauses, variables: Optional[Dict[str, Tuple[float, float]]] = None) -> CNFSpecification:
    """Conjunction helper to create a CNFSpecification from CNFClause objects."""
    cnf_clauses: List[CNFClause] = [cl for cl in clauses if isinstance(cl, CNFClause)]
    if variables is None:
        var_bounds: Dict[str, List[float]] = {}
    else:
        var_bounds = {name: [float(bounds[0]), float(bounds[1])] for name, bounds in variables.items()}
    return CNFSpecification(clauses=cnf_clauses, variables=var_bounds)


class Result:
    """dReal-compatible solver result with interval-style access."""

    def __init__(
        self,
        result: str,
        intervals: Dict[str, List[float]],
        constraints: List[Dict[str, Any]],
        sanity_check: Optional[Dict[str, Any]] = None,
        var_order: Optional[List[str]] = None,
        witness: Optional[Dict[str, float]] = None,
    ):
        self.result = result
        self.sanity_check = sanity_check
        self.constraints = constraints
        self.intervals = intervals
        self.witness = witness
        self._intervals: Dict[str, Interval] = {}
        for name, bounds in intervals.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Interval for {name} must be a pair of (lower, upper)")
            self._intervals[name] = Interval(bounds[0], bounds[1])
        self._order: List[str] = []
        if var_order is not None:
            for name in var_order:
                if name in self._intervals and name not in self._order:
                    self._order.append(name)
        for name in self._intervals.keys():
            if name not in self._order:
                self._order.append(name)

    def __str__(self) -> str:
        lines: List[str] = [self.result]
        if self.result.upper() == "SAT":
            EPS = 1e-12
            formatted: List[str] = []
            for name in self._order:
                interval = self._intervals.get(name)
                if interval is None or name.startswith("Y_"):
                    continue
                lo, hi = interval.lb(), interval.ub()
                if _builtins.abs(hi - lo) <= EPS:
                    formatted.append(f"{name}={interval.mid():.6f}")
                else:
                    formatted.append(f"{name}=[{lo:.6f}, {hi:.6f}]")
            if formatted:
                lines.append("; ".join(formatted))

        if self.sanity_check is not None:
            status = "PASS" if self.sanity_check.get("passed") else "FAIL"
            math_res = self.sanity_check.get("math_residual")
            torch_res = self.sanity_check.get("torch_residual")
            tol = self.sanity_check.get("tolerance")

            def _fmt(val: Any) -> str:
                try:
                    return f"{float(val):.6g}"
                except Exception:
                    return str(val)

            lines.append(f"[sanity:{status}] math={_fmt(math_res)} torch={_fmt(torch_res)} tol={_fmt(tol)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self.result.upper() == "SAT"

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return len(self._order)

    def _resolve_key(self, key: Union[int, str, Any]) -> str:
        if isinstance(key, int):
            if key < 0 or key >= len(self._order):
                raise IndexError("Result index out of range")
            return self._order[key]
        if hasattr(key, "name"):
            return getattr(key, "name")
        return str(key)

    def __getitem__(self, key: Union[int, str, Any]) -> Interval:
        name = self._resolve_key(key)
        if name not in self._intervals:
            raise KeyError(name)
        return self._intervals[name]

    def keys(self):
        return self._intervals.keys()

    def items(self):
        return self._intervals.items()

    def values(self):
        return self._intervals.values()


@dataclass
class OptimizationResult:
    """Best-effort optimization result using repeated satisfiability checks."""
    status: str
    objective: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    iterations: int
    witness: Optional[Dict[str, float]]
    result: Optional[Result] = None

    def __str__(self) -> str:
        if self.status == "UNSAT":
            return "UNSAT"
        if self.objective is None:
            return f"{self.status}"
        return f"{self.status} obj={self.objective:.6g} bounds=[{self.lower_bound:.6g}, {self.upper_bound:.6g}]"


_DEFAULT_OBJ_RANGE: float = 1e6


def _collect_expr_vars(expr: Expr, names: Optional[set] = None) -> set:
    if names is None:
        names = set()
    node = Expr.to_expr(expr)
    if node.kind == 'var':
        names.add(node.name)
        return names
    for sub in node.args:
        if isinstance(sub, Expr):
            _collect_expr_vars(sub, names)
        else:
            _collect_expr_vars(Expr.to_expr(sub), names)
    return names


def _interval_eval_expr(expr: Expr, bounds: Dict[str, List[float]]) -> Tuple[float, float]:
    node = Expr.to_expr(expr)
    exp_cap = math.log(_DEFAULT_OBJ_RANGE) if _DEFAULT_OBJ_RANGE > 0 else 0.0

    def _safe_exp(v: float) -> float:
        if not math.isfinite(v):
            return _DEFAULT_OBJ_RANGE
        if v >= exp_cap:
            return _DEFAULT_OBJ_RANGE
        if v <= -exp_cap:
            return 0.0
        return math.exp(v)

    def _safe_pow(base_val: float, exp_val: float) -> float:
        try:
            val = base_val ** exp_val
        except OverflowError:
            if exp_val % 2 == 0:
                return _DEFAULT_OBJ_RANGE
            return _DEFAULT_OBJ_RANGE if base_val >= 0 else -_DEFAULT_OBJ_RANGE
        if not math.isfinite(val):
            return _DEFAULT_OBJ_RANGE if val > 0 else -_DEFAULT_OBJ_RANGE
        if val > _DEFAULT_OBJ_RANGE:
            return _DEFAULT_OBJ_RANGE
        if val < -_DEFAULT_OBJ_RANGE:
            return -_DEFAULT_OBJ_RANGE
        return val
    if node.kind == 'const':
        val = float(node.value)
        return val, val
    if node.kind == 'var':
        lo, hi = bounds.get(node.name, [-_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE])
        return float(lo), float(hi)
    if node.kind == 'neg':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return -hi, -lo
    if node.kind == 'abs':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return (0.0, _builtins.max(abs(lo), abs(hi)))
    if node.kind == 'min':
        lo1, hi1 = _interval_eval_expr(node.args[0], bounds)
        lo2, hi2 = _interval_eval_expr(node.args[1], bounds)
        return (_builtins.min(lo1, lo2), _builtins.min(hi1, hi2))
    if node.kind == 'max':
        lo1, hi1 = _interval_eval_expr(node.args[0], bounds)
        lo2, hi2 = _interval_eval_expr(node.args[1], bounds)
        return (_builtins.max(lo1, lo2), _builtins.max(hi1, hi2))
    if node.kind in {'add', 'sub', 'mul', 'div'}:
        lo1, hi1 = _interval_eval_expr(node.args[0], bounds)
        lo2, hi2 = _interval_eval_expr(node.args[1], bounds)
        if node.kind == 'add':
            return lo1 + lo2, hi1 + hi2
        if node.kind == 'sub':
            return lo1 - hi2, hi1 - lo2
        if node.kind == 'mul':
            candidates = [lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2]
            return _builtins.min(candidates), _builtins.max(candidates)
        if node.kind == 'div':
            if lo2 <= 0 <= hi2:
                return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
            candidates = [lo1 / lo2, lo1 / hi2, hi1 / lo2, hi1 / hi2]
            return _builtins.min(candidates), _builtins.max(candidates)
    if node.kind == 'pow':
        base = node.args[0]
        exponent = node.args[1]
        if isinstance(exponent, Expr) and exponent.kind == 'const':
            exp_val = float(exponent.value)
            lo, hi = _interval_eval_expr(base, bounds)
            if exp_val.is_integer():
                exp_int = int(exp_val)
                vals = [_safe_pow(lo, exp_int), _safe_pow(hi, exp_int)]
                if exp_int % 2 == 0 and lo < 0 < hi:
                    vals.append(0.0)
                return _builtins.min(vals), _builtins.max(vals)
            if lo < 0:
                return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
            return _safe_pow(lo, exp_val), _safe_pow(hi, exp_val)
        return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
    if node.kind == 'sin' or node.kind == 'cos' or node.kind == 'tan':
        return -1.0, 1.0
    if node.kind == 'atan':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return math.atan(lo), math.atan(hi)
    if node.kind == 'tanh':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return math.tanh(lo), math.tanh(hi)
    if node.kind == 'sigmoid':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return 1.0 / (1.0 + _safe_exp(-lo)), 1.0 / (1.0 + _safe_exp(-hi))
    if node.kind == 'exp':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        return _safe_exp(lo), _safe_exp(hi)
    if node.kind == 'log':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        if lo <= 0:
            return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
        return math.log(lo), math.log(hi)
    if node.kind == 'sqrt':
        lo, hi = _interval_eval_expr(node.args[0], bounds)
        if hi < 0:
            return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
        safe_lo = _builtins.max(lo, 0.0)
        return math.sqrt(safe_lo), math.sqrt(_builtins.max(hi, 0.0))
    return -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE


def _expr_const_value(expr: Expr) -> Optional[float]:
    node = Expr.to_expr(expr)
    if node.kind == 'const':
        return float(node.value)
    if node.kind == 'neg':
        val = _expr_const_value(node.args[0])
        return None if val is None else -val
    if node.kind in {'add', 'sub', 'mul', 'div', 'pow'}:
        a = _expr_const_value(node.args[0])
        b = _expr_const_value(node.args[1])
        if a is None or b is None:
            return None
        if node.kind == 'add':
            return a + b
        if node.kind == 'sub':
            return a - b
        if node.kind == 'mul':
            return a * b
        if node.kind == 'div':
            if b == 0:
                return None
            return a / b
        if node.kind == 'pow':
            return a ** b
    if node.kind == 'abs':
        val = _expr_const_value(node.args[0])
        return None if val is None else _builtins.abs(val)
    if node.kind == 'min':
        a = _expr_const_value(node.args[0])
        b = _expr_const_value(node.args[1])
        return None if a is None or b is None else _builtins.min(a, b)
    if node.kind == 'max':
        a = _expr_const_value(node.args[0])
        b = _expr_const_value(node.args[1])
        return None if a is None or b is None else _builtins.max(a, b)
    if node.kind == 'sin':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.sin(val)
    if node.kind == 'cos':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.cos(val)
    if node.kind == 'tan':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.tan(val)
    if node.kind == 'atan':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.atan(val)
    if node.kind == 'tanh':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.tanh(val)
    if node.kind == 'sigmoid':
        val = _expr_const_value(node.args[0])
        return None if val is None else 1.0 / (1.0 + math.exp(-val))
    if node.kind == 'exp':
        val = _expr_const_value(node.args[0])
        return None if val is None else math.exp(val)
    if node.kind == 'log':
        val = _expr_const_value(node.args[0])
        if val is None or val <= 0:
            return None
        return math.log(val)
    if node.kind == 'sqrt':
        val = _expr_const_value(node.args[0])
        if val is None or val < 0:
            return None
        return math.sqrt(val)
    return None


def _rewrite_atan2_expr(y: Expr, x: Expr, bounds: Dict[str, List[float]]) -> Expr:
    # Branch-free identity using only atan/sqrt/div, compatible with AutoLiRPA.
    # atan2(y, x) = 2 * atan( y / (sqrt(x^2 + y^2) + x) )
    # This is valid for all (x, y) except when x < 0 and y = 0 (denominator is 0).
    # We add a tiny epsilon to avoid division by zero in that degenerate case.
    eps = float(_DEFAULT_NEGATION_EPS)
    denom = sqrt(x * x + y * y) + x + eps
    return 2.0 * atan(y / denom)


def _rewrite_expr_for_autolirpa(
    expr: Expr,
    bounds: Dict[str, List[float]],
    inv_counter: List[int],
) -> Tuple[Expr, List[Constraint]]:
    node = Expr.to_expr(expr)
    if node.kind in {'var', 'const'}:
        return node, []

    if node.kind == 'atan2':
        rewritten = _rewrite_atan2_expr(Expr.to_expr(node.args[0]), Expr.to_expr(node.args[1]), bounds)
        return _rewrite_expr_for_autolirpa(rewritten, bounds, inv_counter)

    if node.kind == 'div':
        left, left_extra = _rewrite_expr_for_autolirpa(node.args[0], bounds, inv_counter)
        right, right_extra = _rewrite_expr_for_autolirpa(node.args[1], bounds, inv_counter)
        denom_const = _expr_const_value(right)
        if denom_const is not None:
            if denom_const == 0:
                raise Smt2ParseError("Division by zero is not supported.")
            return left * (1.0 / denom_const), left_extra + right_extra

        inv_name = f"__inv_{inv_counter[0]}"
        inv_counter[0] += 1
        inv_var = Variable(inv_name, "Real")
        denom_lo, denom_hi = _interval_eval_expr(right, bounds)

        if math.isfinite(denom_lo) and math.isfinite(denom_hi) and denom_lo > 0:
            inv_lo, inv_hi = 1.0 / denom_hi, 1.0 / denom_lo
        elif math.isfinite(denom_lo) and math.isfinite(denom_hi) and denom_hi < 0:
            inv_lo, inv_hi = 1.0 / denom_hi, 1.0 / denom_lo
        else:
            inv_lo, inv_hi = -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE

        bounds[inv_name] = [float(inv_lo), float(inv_hi)]
        inv_constraints = [
            Constraint(left=inv_var, operator=ConstraintType.GE, right=Expr('const', value=float(inv_lo))),
            Constraint(left=inv_var, operator=ConstraintType.LE, right=Expr('const', value=float(inv_hi))),
            Constraint(left=inv_var * right, operator=ConstraintType.EQ, right=Expr('const', value=1.0)),
        ]
        return left * inv_var, left_extra + right_extra + inv_constraints

    # Recursive rewrite for other expressions.
    if node.kind in {'neg', 'abs', 'sin', 'cos', 'tan', 'atan', 'tanh', 'sigmoid', 'exp', 'log', 'sqrt'}:
        arg, extra = _rewrite_expr_for_autolirpa(node.args[0], bounds, inv_counter)
        return Expr(node.kind, [arg]), extra
    if node.kind in {'min', 'max', 'add', 'sub', 'mul', 'pow'}:
        left, left_extra = _rewrite_expr_for_autolirpa(node.args[0], bounds, inv_counter)
        right, right_extra = _rewrite_expr_for_autolirpa(node.args[1], bounds, inv_counter)
        return Expr(node.kind, [left, right]), left_extra + right_extra

    return node, []


def _append_constraint_to_spec(spec: DNFSpecification, constraint: Constraint) -> DNFSpecification:
    new_clauses = [DNFClause(constraints=list(cl.constraints) + [constraint]) for cl in spec.clauses]
    return DNFSpecification(clauses=new_clauses,
                            variables=dict(spec.variables),
                            discrete_types=dict(spec.discrete_types))


def _spec_point_satisfies(spec: DNFSpecification, point: Dict[str, float], tol: float) -> bool:
    for cl in spec.clauses:
        clause_ok = True
        for c in cl.constraints:
            try:
                vl = _eval_expr_value(c.left, point)
                vr = _eval_expr_value(c.right, point)
            except Exception:
                clause_ok = False
                break
            if c.operator == ConstraintType.LE:
                if (vl - vr) > tol:
                    clause_ok = False
                    break
            elif c.operator == ConstraintType.GE:
                if (vr - vl) > tol:
                    clause_ok = False
                    break
            else:
                if _builtins.abs(vl - vr) > tol:
                    clause_ok = False
                    break
        if clause_ok:
            return True
    return False


def _candidate_points_from_bounds(bounds: Dict[str, List[float]],
                                  discrete_types: Optional[Dict[str, str]] = None,
                                  max_corners: int = 256) -> List[Dict[str, float]]:
    names = list(bounds.keys())
    if not names:
        return [{}]

    def _sanitize(val: float, default: float) -> float:
        return default if not math.isfinite(val) else float(val)

    lows: Dict[str, float] = {}
    highs: Dict[str, float] = {}
    mids: Dict[str, float] = {}
    for name in names:
        lo, hi = bounds.get(name, [-1.0, 1.0])
        lo = _sanitize(float(lo), -1.0)
        hi = _sanitize(float(hi), 1.0)
        if discrete_types and name in discrete_types:
            dtype = discrete_types[name]
            if dtype in {"bool", "binary"}:
                lo = _builtins.max(lo, 0.0)
                hi = _builtins.min(hi, 1.0)
                if not math.isfinite(lo):
                    lo = 0.0
                if not math.isfinite(hi):
                    hi = 1.0
            else:
                if not math.isfinite(lo):
                    lo = -_DREAL_INT_BOUND
                if not math.isfinite(hi):
                    hi = _DREAL_INT_BOUND
                # Clamp to the dReal integer domain.
                lo = _builtins.max(lo, -_DREAL_INT_BOUND)
                hi = _builtins.min(hi, _DREAL_INT_BOUND)
            lo = float(math.ceil(lo))
            hi = float(math.floor(hi))
            if lo > hi:
                return []
            mid = float(int((lo + hi) // 2))
        else:
            mid = 0.5 * (lo + hi)
        lows[name] = lo
        highs[name] = hi
        mids[name] = mid

    points: List[Dict[str, float]] = [mids, lows, highs]

    # If dimensionality is small, try all box corners.
    if len(names) <= 8 and (2 ** len(names)) <= max_corners:
        for mask in range(2 ** len(names)):
            corner: Dict[str, float] = {}
            for i, name in enumerate(names):
                corner[name] = highs[name] if ((mask >> i) & 1) else lows[name]
            points.append(corner)

    # Deduplicate by value tuples.
    seen = set()
    unique: List[Dict[str, float]] = []
    for p in points:
        key = tuple((name, p[name]) for name in names)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _best_feasible_candidate(spec: DNFSpecification,
                             obj_expr: Expr,
                             sense: str,
                             tol: float) -> Optional[Tuple[Dict[str, float], float]]:
    best_point: Optional[Dict[str, float]] = None
    best_val: Optional[float] = None
    discrete = _collect_discrete_types(spec)
    for point in _candidate_points_from_bounds(spec.variables, discrete_types=discrete):
        if not _spec_point_satisfies(spec, point, tol):
            continue
        val = _eval_expr_value(obj_expr, point)
        if best_val is None:
            best_val = val
            best_point = point
        elif sense == "min" and val < best_val:
            best_val = val
            best_point = point
        elif sense == "max" and val > best_val:
            best_val = val
            best_point = point
    if best_point is None or best_val is None:
        return None
    return best_point, best_val


def _result_from_point(point: Dict[str, float]) -> Result:
    intervals = {k: [float(v), float(v)] for k, v in point.items()}
    return Result(result="SAT", intervals=intervals, constraints=[], sanity_check=None,
                  var_order=list(point.keys()), witness=dict(point))


def _collect_discrete_types(spec: DNFSpecification) -> Dict[str, str]:
    discrete = dict(spec.discrete_types)
    if not discrete:
        for name in spec.variables:
            vtype = _VAR_TYPES.get(name)
            if vtype in {"int", "bool", "binary"}:
                discrete[name] = vtype
    return discrete


def _tighten_discrete_bounds(spec: DNFSpecification,
                             discrete: Dict[str, str]) -> Tuple[Optional[DNFSpecification], bool]:
    if not discrete:
        return spec, True
    variables = dict(spec.variables)
    all_fixed = True
    for name, dtype in discrete.items():
        lo, hi = variables.get(name, [-float('inf'), float('inf')])
        if dtype in {"bool", "binary"}:
            lo = _builtins.max(lo, 0.0)
            hi = _builtins.min(hi, 1.0)
            if not math.isfinite(lo):
                lo = 0.0
            if not math.isfinite(hi):
                hi = 1.0
        else:
            if not math.isfinite(lo):
                lo = -_DREAL_INT_BOUND
            if not math.isfinite(hi):
                hi = _DREAL_INT_BOUND
            # Clamp to the dReal integer domain.
            lo = _builtins.max(lo, -_DREAL_INT_BOUND)
            hi = _builtins.min(hi, _DREAL_INT_BOUND)
        lo_i = float(math.ceil(lo))
        hi_i = float(math.floor(hi))
        if lo_i > hi_i:
            return None, False
        variables[name] = [lo_i, hi_i]
        if lo_i != hi_i:
            all_fixed = False
    return DNFSpecification(clauses=spec.clauses,
                            variables=variables,
                            discrete_types=dict(discrete)), all_fixed


def _split_discrete_spec(spec: DNFSpecification, name: str) -> List[DNFSpecification]:
    lo, hi = spec.variables[name]
    lo_i = int(lo)
    hi_i = int(hi)
    if lo_i >= hi_i:
        return []
    # If only two integers remain, split into two singletons.
    if hi_i - lo_i == 1:
        left = DNFSpecification(
            clauses=spec.clauses,
            variables={**spec.variables, name: [float(lo_i), float(lo_i)]},
            discrete_types=dict(spec.discrete_types),
        )
        right = DNFSpecification(
            clauses=spec.clauses,
            variables={**spec.variables, name: [float(hi_i), float(hi_i)]},
            discrete_types=dict(spec.discrete_types),
        )
        return [left, right]
    mid = (lo_i + hi_i + 1) // 2
    left_hi = mid - 1
    right_lo = mid
    left = DNFSpecification(
        clauses=spec.clauses,
        variables={**spec.variables, name: [float(lo_i), float(left_hi)]},
        discrete_types=dict(spec.discrete_types),
    )
    right = DNFSpecification(
        clauses=spec.clauses,
        variables={**spec.variables, name: [float(right_lo), float(hi_i)]},
        discrete_types=dict(spec.discrete_types),
    )
    return [left, right]


def _rewrite_expr(node: Expr, fixed: Optional[Dict[str, float]] = None) -> Expr:
    if node.kind == 'var' and fixed and node.name in fixed:
        return Expr('const', value=float(fixed[node.name]))
    if node.kind in {'var', 'const'}:
        return node
    args = [_rewrite_expr(arg, fixed) for arg in node.args]
    if node.kind == 'pow':
        base = args[0]
        exp = args[1]
        if exp.kind == 'const':
            exp_val = float(exp.value)
            if _builtins.abs(exp_val - round(exp_val)) <= 1e-9:
                exp_int = int(round(exp_val))
                if exp_int == 0:
                    return Expr('const', value=1.0)
                if exp_int == 1:
                    return base
                if exp_int < 0:
                    pos = _rewrite_expr(Expr('pow', [base, Expr('const', value=float(-exp_int))]))
                    return Expr('div', [Expr('const', value=1.0), pos])
                return Expr('pow', [base, Expr('const', value=float(exp_int))])
        return Expr('pow', args=args)
    return Expr(node.kind, args=args, name=node.name, value=node.value)


def _expr_sort_key(expr: Expr) -> Tuple[Any, ...]:
    node = Expr.to_expr(expr)
    if node.kind == 'var':
        return ('var', node.name)
    if node.kind == 'const':
        return ('const', repr(float(node.value)))
    value_key = None if node.value is None else repr(float(node.value))
    return (node.kind, node.name, value_key,
            tuple(_expr_sort_key(arg) for arg in node.args))


def _fold_binary(kind: str, args: List[Expr]) -> Expr:
    result = args[0]
    for arg in args[1:]:
        result = Expr(kind, [result, arg])
    return result


def _canonicalize_expr(node: Expr) -> Expr:
    node = Expr.to_expr(node)
    if node.kind in {'var', 'const'}:
        return node
    args = [_canonicalize_expr(arg) for arg in node.args]
    if node.kind == 'neg':
        arg = args[0]
        if arg.kind == 'const':
            return Expr('const', value=-float(arg.value))
        if arg.kind == 'neg':
            return Expr.to_expr(arg.args[0])
        return Expr('neg', [arg])
    if node.kind in {'add', 'mul', 'min', 'max'}:
        flat: List[Expr] = []
        for arg in args:
            if arg.kind == node.kind:
                flat.extend(arg.args)
            else:
                flat.append(arg)
        if node.kind == 'add':
            const_sum = 0.0
            new_args: List[Expr] = []
            for arg in flat:
                if arg.kind == 'const':
                    const_sum += float(arg.value)
                else:
                    new_args.append(arg)
            if const_sum != 0.0:
                new_args.append(Expr('const', value=const_sum))
            if not new_args:
                return Expr('const', value=0.0)
            new_args = sorted(new_args, key=_expr_sort_key)
            return new_args[0] if len(new_args) == 1 else _fold_binary('add', new_args)
        if node.kind == 'mul':
            const_prod = 1.0
            new_args = []
            for arg in flat:
                if arg.kind == 'const':
                    const_prod *= float(arg.value)
                else:
                    new_args.append(arg)
            if const_prod == 0.0:
                return Expr('const', value=0.0)
            if const_prod != 1.0:
                new_args.append(Expr('const', value=const_prod))
            if not new_args:
                return Expr('const', value=1.0)
            new_args = sorted(new_args, key=_expr_sort_key)
            return new_args[0] if len(new_args) == 1 else _fold_binary('mul', new_args)
        const_val: Optional[float] = None
        new_args = []
        for arg in flat:
            if arg.kind == 'const':
                val = float(arg.value)
                if const_val is None:
                    const_val = val
                else:
                    const_val = min(const_val, val) if node.kind == 'min' else max(const_val, val)
            else:
                new_args.append(arg)
        if const_val is not None:
            new_args.append(Expr('const', value=const_val))
        if not new_args:
            return Expr('const', value=const_val if const_val is not None else 0.0)
        new_args = sorted(new_args, key=_expr_sort_key)
        return new_args[0] if len(new_args) == 1 else _fold_binary(node.kind, new_args)
    return Expr(node.kind, args=args, name=node.name, value=node.value)


def _simplify_pow_constants(spec: DNFSpecification) -> DNFSpecification:
    new_clauses: List[DNFClause] = []
    for cl in spec.clauses:
        new_constraints: List[Constraint] = []
        for ct in cl.constraints:
            left = _canonicalize_expr(_rewrite_expr(ct.left, None))
            right = _canonicalize_expr(_rewrite_expr(ct.right, None))
            new_constraints.append(Constraint(left=left, operator=ct.operator, right=right))
        new_clauses.append(DNFClause(new_constraints))
    return DNFSpecification(
        clauses=new_clauses,
        variables=dict(spec.variables),
        discrete_types=dict(spec.discrete_types),
    )


def _substitute_fixed_vars(spec: DNFSpecification,
                           fixed: Dict[str, float]) -> Optional[DNFSpecification]:
    if not fixed:
        return spec
    tol = 1e-9
    new_vars = {name: bounds for name, bounds in spec.variables.items()
                if name not in fixed}
    new_discrete = {name: vtype for name, vtype in spec.discrete_types.items()
                    if name in new_vars}

    new_clauses: List[DNFClause] = []
    for cl in spec.clauses:
        keep: List[Constraint] = []
        clause_unsat = False
        for ct in cl.constraints:
            left = _rewrite_expr(ct.left, fixed)
            right = _rewrite_expr(ct.right, fixed)
            if left.kind == 'const' and right.kind == 'const':
                lv = float(left.value)
                rv = float(right.value)
                if ct.operator == ConstraintType.LE:
                    if lv <= rv + tol:
                        continue
                elif ct.operator == ConstraintType.GE:
                    if lv + tol >= rv:
                        continue
                else:  # EQ
                    if _builtins.abs(lv - rv) <= tol:
                        continue
                clause_unsat = True
                break
            keep.append(Constraint(left=left, operator=ct.operator, right=right))
        if clause_unsat:
            continue
        new_clauses.append(DNFClause(keep))
    if not new_clauses:
        return None
    return DNFSpecification(clauses=new_clauses, variables=new_vars, discrete_types=new_discrete)


def _merge_fixed_result(res: Result, fixed: Dict[str, float]) -> Result:
    if not fixed:
        return res
    intervals = dict(res.intervals)
    for name, value in fixed.items():
        intervals[name] = [float(value), float(value)]
    witness: Optional[Dict[str, float]] = None
    if res.witness is not None or fixed:
        witness = dict(res.witness or {})
        for name, value in fixed.items():
            witness[name] = float(value)
    var_order = list(res._order) if hasattr(res, "_order") else list(intervals.keys())
    for name in fixed:
        if name not in var_order:
            var_order.append(name)
    return Result(
        result=res.result,
        intervals=intervals,
        constraints=res.constraints,
        sanity_check=res.sanity_check,
        var_order=var_order,
        witness=witness,
    )


def _point_from_result(res: Result) -> Optional[Dict[str, float]]:
    if res.witness:
        return dict(res.witness)
    if not res.intervals:
        return None
    point: Dict[str, float] = {}
    for name, bounds in res.intervals.items():
        if name.startswith("Y_"):
            continue
        lo, hi = bounds
        point[name] = 0.5 * (float(lo) + float(hi))
    return point


def _eval_expr_value(expr: Expr, point: Dict[str, float]) -> float:
    node = Expr.to_expr(expr)
    if node.kind == 'var':
        return float(point[node.name])
    if node.kind == 'const':
        return float(node.value)
    if node.kind == 'neg':
        return -_eval_expr_value(node.args[0], point)
    if node.kind == 'abs':
        return _builtins.abs(_eval_expr_value(node.args[0], point))
    if node.kind == 'tan':
        inner = _eval_expr_value(node.args[0], point)
        return math.sin(inner) / math.cos(inner)
    if node.kind == 'min':
        return _builtins.min(_eval_expr_value(node.args[0], point),
                             _eval_expr_value(node.args[1], point))
    if node.kind == 'max':
        return _builtins.max(_eval_expr_value(node.args[0], point),
                             _eval_expr_value(node.args[1], point))
    if node.kind in {'add', 'sub', 'mul', 'div', 'pow'}:
        a = _eval_expr_value(node.args[0], point)
        b = _eval_expr_value(node.args[1], point)
        if node.kind == 'add':
            return a + b
        if node.kind == 'sub':
            return a - b
        if node.kind == 'mul':
            return a * b
        if node.kind == 'div':
            return a / b
        if node.kind == 'pow':
            return a ** b
    a = _eval_expr_value(node.args[0], point)
    if node.kind == 'sin':
        return math.sin(a)
    if node.kind == 'cos':
        return math.cos(a)
    if node.kind == 'atan':
        return math.atan(a)
    if node.kind == 'tanh':
        return math.tanh(a)
    if node.kind == 'sigmoid':
        return 1.0 / (1.0 + math.exp(-a))
    if node.kind == 'exp':
        return math.exp(a)
    if node.kind == 'log':
        return math.log(a)
    if node.kind == 'sqrt':
        return math.sqrt(a)
    raise NotImplementedError(f"Unsupported expression kind: {node.kind}")


def _normalize_opt_args(objective: Any, spec_or_clause: Any) -> Tuple[Expr, Any]:
    spec_like = (Formula, DNFSpecification, DNFClause, CNFSpecification, CNFClause, list)
    if isinstance(objective, spec_like) and isinstance(spec_or_clause, Expr):
        return Expr.to_expr(spec_or_clause), objective
    return Expr.to_expr(objective), spec_or_clause


def Optimize(objective: Expr,
             spec_or_clause,
             delta: Union[float, Config] = 0.001,
             sense: str = "min",
             objective_bounds: Optional[Tuple[float, float]] = None,
             max_iter: int = 40,
             tol: Optional[float] = None,
             reject_ode: bool = True) -> OptimizationResult:
    obj_expr, spec_or_clause = _normalize_opt_args(objective, spec_or_clause)
    config: Optional[Config] = None
    if isinstance(delta, Config):
        config = delta
        delta = float(config.precision)

    if isinstance(spec_or_clause, str):
        if not os.path.isfile(spec_or_clause):
            raise Smt2ParseError(f"SMT2 file not found: {spec_or_clause}")
        from .smt2_adapter import parse_smt2_formula
        spec_or_clause = parse_smt2_formula(spec_or_clause, reject_ode=reject_ode)

    spec = _normalize_to_dnf_spec(spec_or_clause)
    if spec is None:
        return OptimizationResult(status="UNSAT", objective=None,
                                  lower_bound=None, upper_bound=None,
                                  iterations=0, witness=None, result=None)

    for name in _collect_expr_vars(obj_expr):
        if name not in spec.variables:
            spec.variables[name] = [-1.0, 1.0]

    if objective_bounds is None:
        lo, hi = _interval_eval_expr(obj_expr, spec.variables)
    else:
        lo, hi = objective_bounds

    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = -_DEFAULT_OBJ_RANGE, _DEFAULT_OBJ_RANGE
    if lo > hi:
        lo, hi = hi, lo

    if tol is None:
        tol = _builtins.max(float(delta), 1e-6)

    sense = sense.lower()
    if sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")

    best_result: Optional[Result] = None
    best_obj: Optional[float] = None
    best_witness: Optional[Dict[str, float]] = None
    iterations = 0

    # Seed with a feasibility check on the base constraints (no objective).
    base_res = CheckSatisfiability(spec, delta=config or delta)
    if base_res.result == "UNSAT":
        return OptimizationResult(status="UNSAT", objective=None,
                                  lower_bound=None, upper_bound=None,
                                  iterations=0, witness=None, result=None)
    if base_res.result == "SAT":
        base_point = _point_from_result(base_res)
        if base_point is not None:
            base_obj = _eval_expr_value(obj_expr, base_point)
            best_result = base_res
            best_obj = base_obj
            best_witness = base_point
            if sense == "min":
                hi = _builtins.min(hi, base_obj)
            else:
                lo = _builtins.max(lo, base_obj)

    # Try cheap feasible candidates to tighten the initial objective bound.
    candidate = _best_feasible_candidate(spec, obj_expr, sense, tol)
    if candidate is not None:
        point, val = candidate
        best_result = _result_from_point(point)
        best_obj = val
        best_witness = point
        if sense == "min":
            hi = _builtins.min(hi, val)
        else:
            lo = _builtins.max(lo, val)

    for _ in range(max_iter):
        iterations += 1
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        constraint = (obj_expr <= mid) if sense == "min" else (obj_expr >= mid)
        spec_mid = _append_constraint_to_spec(spec, constraint)
        res = CheckSatisfiability(spec_mid, delta=config or delta)
        if res.result == "SAT":
            point = _point_from_result(res)
            if point is not None:
                best_obj = _eval_expr_value(obj_expr, point)
                best_witness = point
            else:
                best_obj = mid
                best_witness = None
            best_result = res
            if sense == "min":
                hi = mid
            else:
                lo = mid
            continue
        if res.result == "UNSAT":
            if sense == "min":
                lo = mid
            else:
                hi = mid
            continue

        # UNKNOWN: try cheap candidate search before giving up.
        candidate = _best_feasible_candidate(spec_mid, obj_expr, sense, tol)
        if candidate is not None:
            point, val = candidate
            best_result = _result_from_point(point)
            best_obj = val
            best_witness = point
            if sense == "min":
                hi = _builtins.min(hi, val)
            else:
                lo = _builtins.max(lo, val)
        else:
            # keep best if any witness is available
            point = _point_from_result(res)
            if point is not None:
                best_obj = _eval_expr_value(obj_expr, point)
                best_witness = point
                best_result = res
                if sense == "min":
                    hi = _builtins.min(hi, best_obj)
                else:
                    lo = _builtins.max(lo, best_obj)
            else:
                break

    status = "UNKNOWN"
    if best_result is not None:
        status = "SAT"

    return OptimizationResult(status=status,
                              objective=best_obj,
                              lower_bound=lo,
                              upper_bound=hi,
                              iterations=iterations,
                              witness=best_witness,
                              result=best_result)


def _optimization_result_to_box(opt: OptimizationResult) -> Optional[Box]:
    if opt.status != "SAT":
        return None
    if opt.result is not None and opt.result.intervals:
        order = getattr(opt.result, "_order", None)
        return Box(opt.result.intervals, order=order)
    if opt.witness:
        intervals = {k: [float(v), float(v)] for k, v in opt.witness.items()}
        return Box(intervals)
    return None


def Minimize(objective: Expr,
             spec_or_clause,
             delta: Union[float, Config] = 0.001,
             objective_bounds: Optional[Tuple[float, float]] = None,
             max_iter: int = 40,
             tol: Optional[float] = None,
             reject_ode: bool = True) -> Optional[Box]:
    obj_expr, spec_or_clause = _normalize_opt_args(objective, spec_or_clause)
    opt = Optimize(obj_expr, spec_or_clause, delta=delta, sense="min",
                   objective_bounds=objective_bounds, max_iter=max_iter, tol=tol,
                   reject_ode=reject_ode)
    return _optimization_result_to_box(opt)


def Maximize(objective: Expr,
             spec_or_clause,
             delta: Union[float, Config] = 0.001,
             objective_bounds: Optional[Tuple[float, float]] = None,
             max_iter: int = 40,
             tol: Optional[float] = None,
             reject_ode: bool = True) -> Optional[Box]:
    obj_expr, spec_or_clause = _normalize_opt_args(objective, spec_or_clause)
    opt = Optimize(obj_expr, spec_or_clause, delta=delta, sense="max",
                   objective_bounds=objective_bounds, max_iter=max_iter, tol=tol,
                   reject_ode=reject_ode)
    return _optimization_result_to_box(opt)


def OptimizeFromSMT2(path: str,
                     delta: Union[float, Config] = 0.001,
                     objective_bounds: Optional[Tuple[float, float]] = None,
                     max_iter: int = 40,
                     tol: Optional[float] = None,
                     reject_ode: bool = True) -> OptimizationResult:
    from .smt2_adapter import parse_smt2_problem
    formula, objective, sense = parse_smt2_problem(path, reject_ode=reject_ode)
    if objective is None or sense is None:
        raise Smt2ParseError("No optimization objective found in SMT2 input.")
    return Optimize(objective, formula, delta=delta, sense=sense,
                    objective_bounds=objective_bounds, max_iter=max_iter, tol=tol,
                    reject_ode=reject_ode)

def CheckSatisfiability(spec_or_clause,
                        delta: Union[float, Config] = 0.001,
                        reject_ode: bool = True) -> Result:
    """Solve δ-SAT using ABCROWN (Y <= delta or config.precision).

    Accepts CNF or DNF style specifications and normalizes them internally.
    """
    config: Optional[Config] = None
    if isinstance(delta, Config):
        config = delta
        delta = float(config.precision)

    if isinstance(spec_or_clause, str):
        if not os.path.isfile(spec_or_clause):
            raise Smt2ParseError(f"SMT2 file not found: {spec_or_clause}")
        from .smt2_adapter import parse_smt2_formula
        spec_or_clause = parse_smt2_formula(spec_or_clause, reject_ode=reject_ode)

    spec = _normalize_to_dnf_spec(spec_or_clause)
    if spec is None:
        return Result(result="UNSAT", intervals={}, constraints=[], sanity_check=None)

    def _check_continuous(cur_spec: DNFSpecification) -> Result:
        if not cur_spec.variables:
            tol = delta if config is None else float(config.precision)
            if _spec_point_satisfies(cur_spec, {}, tol):
                return Result(result="SAT", intervals={}, constraints=[], sanity_check=None)
            return Result(result="UNSAT", intervals={}, constraints=[], sanity_check=None)
        cur_spec = _simplify_pow_constants(cur_spec)
        analyzer = AbcrownDNFSolver(cur_spec, verbose=False)
        result_dict = analyzer.CheckSatisfiability(delta, pretty=True, config=config)
        return Result(
            result=result_dict.get("result", "UNSAT"),
            intervals=result_dict.get("intervals", {}),
            constraints=result_dict.get("constraints", []),
            sanity_check=result_dict.get("sanity_check"),
            var_order=result_dict.get("var_order"),
            witness=result_dict.get("witness"),
        )

    discrete = _collect_discrete_types(spec)
    if not discrete:
        return _check_continuous(spec)

    max_nodes = _DEFAULT_DISCRETE_MAX_NODES
    env_limit = os.environ.get("ABCROWN_DISCRETE_MAX_NODES")
    if env_limit is not None:
        try:
            max_nodes = int(env_limit)
        except ValueError:
            pass

    stack: List[DNFSpecification] = [spec]
    saw_unknown = False
    nodes = 0
    while stack:
        nodes += 1
        if nodes > max_nodes:
            return Result(result="UNKNOWN", intervals={}, constraints=[], sanity_check=None)
        cur = stack.pop()
        tightened, all_fixed = _tighten_discrete_bounds(cur, discrete)
        if tightened is None:
            continue
        if all_fixed:
            fixed_vals = {name: bounds[0] for name, bounds in tightened.variables.items()
                          if isinstance(bounds, (list, tuple)) and len(bounds) == 2
                          and float(bounds[0]) == float(bounds[1])}
            if len(fixed_vals) == len(tightened.variables):
                tol = delta if config is None else float(config.precision)
                if _spec_point_satisfies(tightened, fixed_vals, tol):
                    return _result_from_point(fixed_vals)
                continue
            reduced = _substitute_fixed_vars(tightened, fixed_vals)
            if reduced is None:
                continue
            res = _check_continuous(reduced)
            if res.result == "SAT" and fixed_vals:
                res = _merge_fixed_result(res, fixed_vals)
            if res.result == "SAT":
                return res
            if res.result == "UNKNOWN":
                saw_unknown = True
            continue
        # Pick the widest discrete variable to branch.
        branch_var = None
        branch_width = -1.0
        for name in discrete:
            lo, hi = tightened.variables.get(name, [0.0, -1.0])
            width = float(hi) - float(lo)
            if width > branch_width:
                branch_width = width
                branch_var = name
        if branch_var is None:
            continue
        stack.extend(_split_discrete_spec(tightened, branch_var))

    if saw_unknown:
        return Result(result="UNKNOWN", intervals={}, constraints=[], sanity_check=None)
    return Result(result="UNSAT", intervals={}, constraints=[], sanity_check=None)


def ExportOnnxAndVnnlib(spec_or_clause,
                        delta: Union[float, Config] = 0.001,
                        out_dir: Optional[str] = None,
                        export_residual_vnnlib: bool = False,
                        reject_ode: bool = True) -> Tuple[str, str, List[str]]:
    """Export ONNX/VNNLIB for a given spec or SMT2 input.

    Returns (onnx_path, vnnlib_path, var_order).
    """
    config: Optional[Config] = None
    if isinstance(delta, Config):
        config = delta
        delta = float(config.precision)

    if isinstance(spec_or_clause, str):
        if not os.path.isfile(spec_or_clause):
            raise Smt2ParseError(f"SMT2 file not found: {spec_or_clause}")
        from .smt2_adapter import parse_smt2_formula
        spec_or_clause = parse_smt2_formula(spec_or_clause, reject_ode=reject_ode)

    spec = _normalize_to_dnf_spec(spec_or_clause)
    if spec is None:
        raise ValueError("Cannot export ONNX/VNNLIB for an empty specification.")
    spec = _simplify_pow_constants(spec)

    analyzer = AbcrownDNFSolver(spec, verbose=False)
    y_slack = _builtins.max(float(delta), 1e-6)
    return analyzer.ExportOnnxAndVnnlib(
        out_dir=out_dir,
        y_slack=y_slack,
        export_residual_vnnlib=export_residual_vnnlib,
    )


def ExportOnnxAndVnnlibFromSMT2(path: str,
                               delta: Union[float, Config] = 0.001,
                               out_dir: Optional[str] = None,
                               export_residual_vnnlib: bool = False,
                               reject_ode: bool = True) -> Tuple[str, str, List[str]]:
    """Export ONNX/VNNLIB for an SMT2 file. Returns (onnx_path, vnnlib_path, var_order)."""
    return ExportOnnxAndVnnlib(
        path,
        delta=delta,
        out_dir=out_dir,
        export_residual_vnnlib=export_residual_vnnlib,
        reject_ode=reject_ode,
    )


class Smt2ParseError(RuntimeError):
    pass


def _tokenize_smt2(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "(" or ch == ")":
            tokens.append(ch)
            i += 1
            continue
        if ch == "|":
            i += 1
            start = i
            while i < n and text[i] != "|":
                i += 1
            if i >= n:
                raise Smt2ParseError("Unterminated |symbol| in SMT2 input.")
            tokens.append(text[start:i])
            i += 1
            continue
        if ch == "\"":
            i += 1
            start = i
            while i < n and text[i] != "\"":
                i += 1
            if i >= n:
                raise Smt2ParseError("Unterminated string literal in SMT2 input.")
            tokens.append(text[start:i])
            i += 1
            continue
        start = i
        while i < n and text[i] not in " \t\r\n();":
            i += 1
        tokens.append(text[start:i])
    return tokens


def _parse_smt2_sexps(tokens: List[str]) -> List[Any]:
    pos = 0

    def parse_one() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise Smt2ParseError("Unexpected end of SMT2 input.")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            items: List[Any] = []
            while True:
                if pos >= len(tokens):
                    raise Smt2ParseError("Unterminated list in SMT2 input.")
                if tokens[pos] == ")":
                    pos += 1
                    break
                items.append(parse_one())
            return items
        if tok == ")":
            raise Smt2ParseError("Unexpected ')'.")
        pos += 1
        return tok

    sexps: List[Any] = []
    while pos < len(tokens):
        sexps.append(parse_one())
    return sexps


class _Smt2Parser:
    def __init__(self):
        self.vars: Dict[str, Variable] = {}
        self.var_sorts: Dict[str, str] = {}
        self.var_bounds: Dict[str, Tuple[float, float]] = {}
        self._assert_stack: List[List[Formula]] = [[]]
        self.logic: Optional[str] = None
        self.objective: Optional[Expr] = None
        self.objective_sense: Optional[str] = None

    def _current_asserts(self) -> List[Formula]:
        return self._assert_stack[-1]

    def _declare_var(self, name: str, sort: str) -> None:
        if sort not in {"Real", "Int", "Bool"}:
            raise Smt2ParseError(f"Unsupported sort {sort} for {name}.")
        if name not in self.vars:
            self.vars[name] = Variable(name, sort)
            self.var_sorts[name] = sort

    def _parse_bounds(self, tokens: List[Any]) -> Tuple[float, float]:
        if len(tokens) == 1 and isinstance(tokens[0], list):
            tokens = tokens[0]
        if len(tokens) == 2 and all(isinstance(t, (int, float, str)) for t in tokens):
            try:
                return float(tokens[0]), float(tokens[1])
            except ValueError:
                pass
        text = " ".join(str(t) for t in tokens)
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        text = text.replace(",", " ")
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if len(nums) != 2:
            raise Smt2ParseError(f"Invalid bounds specification: {tokens!r}")
        return float(nums[0]), float(nums[1])

    @staticmethod
    def _is_number(token: str) -> bool:
        try:
            float(token)
        except ValueError:
            return False
        return True

    def _parse_expr(self, term: Any, env: Dict[str, Any]) -> Expr:
        if isinstance(term, (int, float)):
            return Expr('const', value=float(term))
        parsed = self._parse_term(term, env)
        if isinstance(parsed, Expr):
            return parsed
        raise Smt2ParseError(f"Expected expression, got {parsed!r}.")

    def _as_formula(self, value: Any) -> Formula:
        if isinstance(value, Formula):
            return value
        if isinstance(value, Constraint):
            return Formula.ensure(value)
        raise Smt2ParseError(f"Expected boolean term, got {value!r}.")

    def _chain_constraints(self, constraints: List[Constraint]) -> Formula:
        if not constraints:
            return Formula('true')
        if len(constraints) == 1:
            return Formula.ensure(constraints[0])
        return logical_and(*[Formula.ensure(c) for c in constraints])

    def _parse_relation(self, op: str, args: List[Any], env: Dict[str, Any]) -> Formula:
        if len(args) < 2:
            raise Smt2ParseError(f"Relation {op} expects at least 2 operands.")
        def _is_bool_term(term: Any) -> bool:
            if isinstance(term, str):
                if term in {"true", "false"}:
                    return True
                if term in self.var_sorts and self.var_sorts[term] == "Bool":
                    return True
            return False

        if op == "=" and any(_is_bool_term(a) for a in args):
            bool_terms = [self._as_formula(self._parse_term(a, env)) for a in args]
            if len(bool_terms) == 1:
                return bool_terms[0]
            eqs: List[Formula] = []
            for left, right in zip(bool_terms[:-1], bool_terms[1:]):
                eqs.append(logical_imply(left, right))
                eqs.append(logical_imply(right, left))
            return logical_and(*eqs)

        exprs = [self._parse_expr(a, env) for a in args]
        constraints: List[Constraint] = []
        eps = float(_DEFAULT_NEGATION_EPS)
        for left, right in zip(exprs[:-1], exprs[1:]):
            if op == "<=":
                constraints.append(left <= right)
            elif op == ">=":
                constraints.append(left >= right)
            elif op == "=":
                constraints.append(left == right)
            elif op == "<":
                constraints.append((left + eps) <= right)
            elif op == ">":
                constraints.append(left >= (right + eps))
            else:
                raise Smt2ParseError(f"Unsupported relation {op}.")
        return self._chain_constraints(constraints)

    def _parse_let(self, bindings: List[Any], body: Any, env: Dict[str, Any]) -> Any:
        if not isinstance(bindings, list):
            raise Smt2ParseError("let expects a list of bindings.")
        local_bindings: Dict[str, Any] = {}
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2:
                raise Smt2ParseError("let binding must be a pair (name value).")
            name, value_term = binding
            if not isinstance(name, str):
                raise Smt2ParseError("let binding name must be a symbol.")
            local_bindings[name] = self._parse_term(value_term, env)
        new_env = dict(env)
        new_env.update(local_bindings)
        return self._parse_term(body, new_env)

    def _parse_term(self, term: Any, env: Dict[str, Any]) -> Any:
        if isinstance(term, list):
            if not term:
                raise Smt2ParseError("Empty SMT2 list.")
            head = term[0]
            if head == "let":
                if len(term) != 3:
                    raise Smt2ParseError("let expects bindings and a body.")
                return self._parse_let(term[1], term[2], env)
            if head in ("and", "or"):
                if len(term) == 1:
                    return Formula('true') if head == "and" else Formula('false')
                args = [self._as_formula(self._parse_term(arg, env)) for arg in term[1:]]
                return logical_and(*args) if head == "and" else logical_or(*args)
            if head == "not":
                if len(term) != 2:
                    raise Smt2ParseError("not expects one operand.")
                return logical_not(self._as_formula(self._parse_term(term[1], env)))
            if head in ("=>", "implies"):
                if len(term) != 3:
                    raise Smt2ParseError("=> expects two operands.")
                lhs = self._as_formula(self._parse_term(term[1], env))
                rhs = self._as_formula(self._parse_term(term[2], env))
                return logical_imply(lhs, rhs)
            if head == "xor":
                if len(term) < 3:
                    raise Smt2ParseError("xor expects at least two operands.")
                acc = self._as_formula(self._parse_term(term[1], env))
                for arg in term[2:]:
                    nxt = self._as_formula(self._parse_term(arg, env))
                    acc = logical_or(logical_and(acc, logical_not(nxt)),
                                     logical_and(logical_not(acc), nxt))
                return acc
            if head in ("<", "<=", ">", ">=", "="):
                return self._parse_relation(head, term[1:], env)
            if head == "ite":
                if len(term) != 4:
                    raise Smt2ParseError("ite expects 3 operands.")
                cond = self._as_formula(self._parse_term(term[1], env))
                then_term = self._parse_term(term[2], env)
                else_term = self._parse_term(term[3], env)
                if isinstance(then_term, Formula) and isinstance(else_term, Formula):
                    return logical_and(
                        logical_imply(cond, then_term),
                        logical_imply(logical_not(cond), else_term),
                    )
                raise Smt2ParseError("ite over expressions is not supported.")
            if head in ("+", "*", "-", "/", "^", "pow"):
                if len(term) < 2:
                    raise Smt2ParseError(f"{head} expects operands.")
                if head == "+":
                    expr = self._parse_expr(term[1], env)
                    for arg in term[2:]:
                        expr = expr + self._parse_expr(arg, env)
                    return expr
                if head == "*":
                    expr = self._parse_expr(term[1], env)
                    for arg in term[2:]:
                        expr = expr * self._parse_expr(arg, env)
                    return expr
                if head == "-":
                    if len(term) == 2:
                        return -self._parse_expr(term[1], env)
                    expr = self._parse_expr(term[1], env)
                    for arg in term[2:]:
                        expr = expr - self._parse_expr(arg, env)
                    return expr
                if head == "/":
                    if len(term) == 2:
                        return self._parse_expr(1.0, env) / self._parse_expr(term[1], env)
                    expr = self._parse_expr(term[1], env)
                    for arg in term[2:]:
                        expr = expr / self._parse_expr(arg, env)
                    return expr
                if head in ("^", "pow"):
                    if len(term) != 3:
                        raise Smt2ParseError("pow/^ expects 2 operands.")
                base = self._parse_expr(term[1], env)
                exp_expr = self._parse_expr(term[2], env)
                if isinstance(exp_expr, Expr) and exp_expr.kind == 'const':
                    exp_val = float(exp_expr.value)
                    if math.isfinite(exp_val) and _builtins.abs(exp_val - round(exp_val)) <= 1e-9:
                        return base ** exp_expr
                    # Rewrite non-integer power as exp(exp_val * log(base)).
                    return exp(exp_val * log(base))
                return base ** exp_expr
            if head in ("atan2", "arctan2"):
                if len(term) != 3:
                    raise Smt2ParseError(f"{head} expects 2 operands.")
                a = self._parse_expr(term[1], env)
                b = self._parse_expr(term[2], env)
                return Expr('atan2', [a, b])
            if head in ("sin", "cos", "tan", "atan", "arctan", "exp", "log", "sqrt",
                        "tanh", "sigmoid", "abs", "sinh", "cosh", "arccos", "acos"):
                if len(term) != 2:
                    raise Smt2ParseError(f"{head} expects 1 operand.")
                arg = self._parse_expr(term[1], env)
                if head == "sin": return sin(arg)
                if head == "cos": return cos(arg)
                if head == "tan": return tan(arg)
                if head in ("atan", "arctan"): return atan(arg)
                if head == "exp": return exp(arg)
                if head == "log": return log(arg)
                if head == "sqrt": return sqrt(arg)
                if head == "tanh": return tanh(arg)
                if head == "sigmoid": return sigmoid(arg)
                if head == "abs": return abs(arg)
                if head == "sinh": return 0.5 * (exp(arg) - exp(-arg))
                if head == "cosh": return 0.5 * (exp(arg) + exp(-arg))
                if head in ("arccos", "acos"):
                    return 2.0 * atan(sqrt(1.0 - arg) / sqrt(1.0 + arg))
            if head in ("min", "max"):
                if len(term) < 3:
                    raise Smt2ParseError(f"{head} expects at least 2 operands.")
                expr = self._parse_expr(term[1], env)
                for arg in term[2:]:
                    expr = min(expr, self._parse_expr(arg, env)) if head == "min" else max(expr, self._parse_expr(arg, env))
                return expr
            raise Smt2ParseError(f"Unsupported operator {head}.")
        if isinstance(term, str):
            if term == "true":
                return Formula('true')
            if term == "false":
                return Formula('false')
            if term in env:
                return env[term]
            if term in self.vars:
                sort = self.var_sorts.get(term, "Real")
                if sort == "Bool":
                    return Formula.ensure(self.vars[term] >= 0.5)
                return self.vars[term]
            if self._is_number(term):
                return Expr('const', value=float(term))
            raise Smt2ParseError(f"Unknown symbol {term}.")
        raise Smt2ParseError(f"Unsupported SMT2 term {term!r}.")

    def parse(self, sexps: List[Any]) -> Formula:
        for cmd in sexps:
            if not isinstance(cmd, list) or not cmd:
                continue
            head = cmd[0]
            if head == "set-logic":
                if len(cmd) < 2:
                    raise Smt2ParseError("set-logic expects a logic name.")
                self.logic = cmd[1]
                if isinstance(self.logic, str) and "ODE" in self.logic:
                    raise Smt2ParseError("ODE logic is not supported in this interface.")
                continue
            if head in ("set-info", "set-option"):
                continue
            if head == "declare-fun":
                if len(cmd) < 4:
                    raise Smt2ParseError("declare-fun expects name, arg list, and sort.")
                name, args, sort = cmd[1], cmd[2], cmd[3]
                if args != []:
                    raise Smt2ParseError("Only nullary functions are supported.")
                if not isinstance(name, str) or not isinstance(sort, str):
                    raise Smt2ParseError("declare-fun expects symbol name and sort.")
                self._declare_var(name, sort)
                if len(cmd) > 4:
                    bounds = self._parse_bounds(cmd[4:])
                    self.var_bounds[name] = bounds
                continue
            if head == "declare-const":
                if len(cmd) < 3:
                    raise Smt2ParseError("declare-const expects name and sort.")
                name, sort = cmd[1], cmd[2]
                if not isinstance(name, str) or not isinstance(sort, str):
                    raise Smt2ParseError("declare-const expects symbol name and sort.")
                self._declare_var(name, sort)
                if len(cmd) > 3:
                    bounds = self._parse_bounds(cmd[3:])
                    self.var_bounds[name] = bounds
                continue
            if head in ("define-fun", "define-fun-rec"):
                raise Smt2ParseError("define-fun is not supported in this SMT2 reader.")
            if head == "assert":
                if len(cmd) != 2:
                    raise Smt2ParseError("assert expects a single term.")
                formula = self._as_formula(self._parse_term(cmd[1], {}))
                self._current_asserts().append(formula)
                continue
            if head in ("minimize", "maximize"):
                if len(cmd) != 2:
                    raise Smt2ParseError(f"{head} expects a single objective term.")
                if self.objective is not None:
                    raise Smt2ParseError("Multiple objectives are not supported.")
                obj_expr = self._parse_expr(cmd[1], {})
                self.objective = obj_expr
                self.objective_sense = "min" if head == "minimize" else "max"
                continue
            if head == "check-sat":
                continue
            if head == "push":
                n = int(cmd[1]) if len(cmd) > 1 else 1
                for _ in range(n):
                    self._assert_stack.append(list(self._assert_stack[-1]))
                continue
            if head == "pop":
                n = int(cmd[1]) if len(cmd) > 1 else 1
                for _ in range(n):
                    if len(self._assert_stack) <= 1:
                        raise Smt2ParseError("pop underflow.")
                    self._assert_stack.pop()
                continue
            if head == "exit":
                break
            raise Smt2ParseError(f"Unsupported SMT2 command {head}.")

        asserts = self._current_asserts()
        if self.var_bounds:
            for name, (lo, hi) in self.var_bounds.items():
                var = self.vars.get(name)
                if var is None:
                    continue
                asserts.append(Formula.ensure(var >= float(lo)))
                asserts.append(Formula.ensure(var <= float(hi)))
        if not asserts:
            return Formula('true')
        return logical_and(*asserts)


def parse_smt2_string(text: str) -> Formula:
    tokens = _tokenize_smt2(text)
    sexps = _parse_smt2_sexps(tokens)
    parser = _Smt2Parser()
    return parser.parse(sexps)


def parse_smt2_file(path: str) -> Formula:
    with open(path, 'r') as f:
        text = f.read()
    return parse_smt2_string(text)


def parse_smt2_string_with_objective(text: str) -> Tuple[Formula, Optional[Expr], Optional[str]]:
    tokens = _tokenize_smt2(text)
    sexps = _parse_smt2_sexps(tokens)
    parser = _Smt2Parser()
    formula = parser.parse(sexps)
    return formula, parser.objective, parser.objective_sense


def parse_smt2_file_with_objective(path: str) -> Tuple[Formula, Optional[Expr], Optional[str]]:
    with open(path, 'r') as f:
        text = f.read()
    return parse_smt2_string_with_objective(text)


def CheckSatisfiabilityFromSMT2(path: str,
                                delta: Union[float, Config] = 0.001,
                                reject_ode: bool = True) -> Result:
    from .smt2_adapter import parse_smt2_formula
    formula = parse_smt2_formula(path, reject_ode=reject_ode)
    return CheckSatisfiability(formula, delta=delta)


def ParseSMT2String(text: str) -> Formula:
    """dReal-compatible wrapper for SMT2 string parsing."""
    return parse_smt2_string(text)


def ParseSMT2File(path: str, reject_ode: bool = True) -> Formula:
    """dReal-compatible wrapper for SMT2 file parsing."""
    from .smt2_adapter import parse_smt2_formula
    return parse_smt2_formula(path, reject_ode=reject_ode)


def _expr_to_smt2(expr: Expr) -> str:
    node = Expr.to_expr(expr)

    def _fmt_num(val: float) -> str:
        if math.isfinite(val) and _builtins.abs(val - round(val)) <= 1e-9:
            return str(int(round(val)))
        return f"{val:.15g}"

    if node.kind == 'var':
        return node.name or 'var'
    if node.kind == 'const':
        v = float(node.value)
        if v < 0:
            return f"(- {_fmt_num(-v)})"
        return _fmt_num(v)
    if node.kind == 'neg':
        return f"(- {_expr_to_smt2(node.args[0])})"
    if node.kind in {'add', 'sub', 'mul', 'div', 'pow'}:
        op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': 'pow'}[node.kind]
        return f"({op} {_expr_to_smt2(node.args[0])} {_expr_to_smt2(node.args[1])})"
    if node.kind in {'sin', 'cos', 'tan', 'atan', 'tanh', 'sigmoid', 'exp', 'log', 'sqrt', 'abs', 'min', 'max'}:
        op = node.kind
        if node.kind in {'min', 'max'}:
            return f"({op} {_expr_to_smt2(node.args[0])} {_expr_to_smt2(node.args[1])})"
        return f"({op} {_expr_to_smt2(node.args[0])})"
    if node.kind == 'differentiate':
        raise Smt2ParseError("differentiate() cannot be exported to SMT2.")
    raise Smt2ParseError(f"Unsupported expression kind for SMT2 export: {node.kind}")


def _constraint_to_smt2(constraint: Constraint) -> str:
    left = _expr_to_smt2(constraint.left)
    right = _expr_to_smt2(constraint.right)
    if constraint.operator == ConstraintType.LE:
        return f"(<= {left} {right})"
    if constraint.operator == ConstraintType.GE:
        return f"(>= {left} {right})"
    if constraint.operator == ConstraintType.EQ:
        return f"(= {left} {right})"
    raise Smt2ParseError(f"Unsupported constraint operator {constraint.operator}")


def _formula_to_smt2(formula: Formula) -> str:
    f = Formula.ensure(formula)
    if f.op == 'true':
        return "true"
    if f.op == 'false':
        return "false"
    if f.op == 'atom':
        return _constraint_to_smt2(f.args[0])
    if f.op == 'not':
        return f"(not {_formula_to_smt2(Formula.ensure(f.args[0]))})"
    if f.op in {'and', 'or'}:
        if not f.args:
            return "true" if f.op == 'and' else "false"
        inner = " ".join(_formula_to_smt2(Formula.ensure(arg)) for arg in f.args)
        return f"({f.op} {inner})"
    raise Smt2ParseError(f"Unsupported formula operator {f.op}")


def _dnf_clause_to_formula(clause: DNFClause) -> Formula:
    if not clause.constraints:
        return Formula('true')
    return logical_and(*[Formula.ensure(ct) for ct in clause.constraints])


def _cnf_clause_to_formula(clause: CNFClause) -> Formula:
    if not clause.constraints:
        return Formula('false')
    return logical_or(*[Formula.ensure(ct) for ct in clause.constraints])


def _collect_vars_from_expr(expr: Expr, seen: set, ordered: List[str]) -> None:
    node = Expr.to_expr(expr)
    if node.kind == 'var':
        if node.name not in seen:
            seen.add(node.name)
            ordered.append(node.name)
        return
    for arg in node.args:
        _collect_vars_from_expr(arg, seen, ordered)


def _collect_vars_from_formula(formula: Formula) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []

    def walk(f: Formula) -> None:
        node = Formula.ensure(f)
        if node.op == 'atom':
            ct = node.args[0]
            _collect_vars_from_expr(ct.left, seen, ordered)
            _collect_vars_from_expr(ct.right, seen, ordered)
            return
        if node.op in {'and', 'or'}:
            for arg in node.args:
                walk(Formula.ensure(arg))
            return
        if node.op == 'not':
            walk(Formula.ensure(node.args[0]))
            return

    walk(formula)
    return ordered


def _spec_to_formula_and_bounds(spec_or_clause) -> Tuple[Formula, Dict[str, List[float]], Dict[str, str]]:
    var_bounds: Dict[str, List[float]] = {}
    discrete_types: Dict[str, str] = {}

    if isinstance(spec_or_clause, Formula):
        formula = spec_or_clause
        return formula, var_bounds, discrete_types
    if isinstance(spec_or_clause, DNFSpecification):
        formula = Formula('false') if not spec_or_clause.clauses else logical_or(
            *[_dnf_clause_to_formula(cl) for cl in spec_or_clause.clauses]
        )
        var_bounds = {name: list(bounds) for name, bounds in spec_or_clause.variables.items()}
        discrete_types = dict(spec_or_clause.discrete_types)
        return formula, var_bounds, discrete_types
    if isinstance(spec_or_clause, CNFSpecification):
        formula = Formula('true') if not spec_or_clause.clauses else logical_and(
            *[_cnf_clause_to_formula(cl) for cl in spec_or_clause.clauses]
        )
        var_bounds = {name: list(bounds) for name, bounds in spec_or_clause.variables.items()}
        return formula, var_bounds, discrete_types
    if isinstance(spec_or_clause, DNFClause):
        formula = _dnf_clause_to_formula(spec_or_clause)
        return formula, var_bounds, discrete_types
    if isinstance(spec_or_clause, CNFClause):
        formula = _cnf_clause_to_formula(spec_or_clause)
        return formula, var_bounds, discrete_types
    if isinstance(spec_or_clause, list):
        if not spec_or_clause:
            return Formula('false'), var_bounds, discrete_types
        if all(isinstance(cl, DNFClause) for cl in spec_or_clause):
            formula = logical_or(*[_dnf_clause_to_formula(cl) for cl in spec_or_clause])
            return formula, var_bounds, discrete_types
        if all(isinstance(cl, CNFClause) for cl in spec_or_clause):
            formula = logical_and(*[_cnf_clause_to_formula(cl) for cl in spec_or_clause])
            return formula, var_bounds, discrete_types
    raise TypeError("Unsupported input type for SMT2 export.")


def ToSMT2(spec_or_clause,
           logic: str = "QF_NRA",
           include_bounds: bool = True,
           include_bool_domain: bool = True) -> str:
    if isinstance(spec_or_clause, str) and os.path.isfile(spec_or_clause):
        from .smt2_adapter import parse_smt2_formula
        spec_or_clause = parse_smt2_formula(spec_or_clause)

    formula, var_bounds, discrete_types = _spec_to_formula_and_bounds(spec_or_clause)
    ordered_vars = _collect_vars_from_formula(formula)
    for name in var_bounds:
        if name not in ordered_vars:
            ordered_vars.append(name)

    lines: List[str] = []
    lines.append(f"(set-logic {logic})")

    for name in ordered_vars:
        vtype = discrete_types.get(name) or _VAR_TYPES.get(name, "Real")
        if vtype in {"bool", "binary"}:
            sort = "Int"
        elif vtype in {"int", "integer"}:
            sort = "Int"
        else:
            sort = "Real"
        lines.append(f"(declare-const {name} {sort})")

    if include_bounds:
        for name in ordered_vars:
            bounds = var_bounds.get(name)
            if bounds is None or len(bounds) != 2:
                continue
            lo, hi = bounds
            if math.isfinite(lo):
                lines.append(f"(assert (>= {name} {_expr_to_smt2(Expr('const', value=float(lo))) }))")
            if math.isfinite(hi):
                lines.append(f"(assert (<= {name} {_expr_to_smt2(Expr('const', value=float(hi))) }))")

    if include_bool_domain:
        for name in ordered_vars:
            vtype = discrete_types.get(name) or _VAR_TYPES.get(name)
            if vtype in {"bool", "binary"}:
                lines.append(f"(assert (or (= {name} 0) (= {name} 1)))")

    lines.append(f"(assert {_formula_to_smt2(formula)})")
    lines.append("(check-sat)")
    lines.append("(exit)")
    return "\n".join(lines) + "\n"


def ExportSMT2(spec_or_clause,
               path: str,
               logic: str = "QF_NRA",
               include_bounds: bool = True,
               include_bool_domain: bool = True) -> None:
    text = ToSMT2(spec_or_clause,
                  logic=logic,
                  include_bounds=include_bounds,
                  include_bool_domain=include_bool_domain)
    with open(path, "w") as f:
        f.write(text)


def _normalize_to_dnf_spec(spec_or_clause: Union[DNFSpecification, DNFClause, List[DNFClause], CNFSpecification, CNFClause, List[CNFClause]]) -> Optional[DNFSpecification]:
    """Convert supported inputs (CNF/DNF) into a DNFSpecification."""
    if isinstance(spec_or_clause, Formula):
        clauses = _formula_to_dnf(spec_or_clause)
        if not clauses:
            return None
        spec_or_clause = clauses
    if isinstance(spec_or_clause, CNFSpecification):
        dnf_clauses = _cnf_clauses_to_dnf_clauses(spec_or_clause.clauses)
        if not dnf_clauses:
            return None
        var_bounds: Dict[str, List[float]] = {}
        for name, bounds in spec_or_clause.variables.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                var_bounds[name] = [float(bounds[0]), float(bounds[1])]
            else:
                raise TypeError(
                    "CNFSpecification variables should map to (lower, upper) pairs."
                )
        discrete_types: Dict[str, str] = {}
        for name in var_bounds:
            vtype = _VAR_TYPES.get(name)
            if vtype in {"int", "bool", "binary"}:
                discrete_types[name] = vtype
        spec_or_clause = DNFSpecification(variables=var_bounds,
                                          clauses=dnf_clauses,
                                          discrete_types=discrete_types)
    elif isinstance(spec_or_clause, CNFClause):
        dnf_clauses = _cnf_clauses_to_dnf_clauses([spec_or_clause])
        if not dnf_clauses:
            return None
        spec_or_clause = dnf_clauses
    elif isinstance(spec_or_clause, list) and any(isinstance(cl, CNFClause) for cl in spec_or_clause):
        if not all(isinstance(cl, CNFClause) for cl in spec_or_clause):
            raise TypeError("Mixed CNFClause and DNFClause inputs are not supported.")
        dnf_clauses = _cnf_clauses_to_dnf_clauses(spec_or_clause)
        if not dnf_clauses:
            return None
        spec_or_clause = dnf_clauses

    if isinstance(spec_or_clause, DNFClause):
        spec_or_clause = _dnf_clauses_to_specification([spec_or_clause])
        if spec_or_clause is None:
            return None

    elif isinstance(spec_or_clause, list):
        if len(spec_or_clause) == 0:
            raise TypeError("Or() expects at least one argument.")
        if not all(isinstance(cl, DNFClause) for cl in spec_or_clause):
            raise TypeError("Or() expects DNFClause inputs when used for DNF disjunction.")
        spec_or_clause = _dnf_clauses_to_specification(list(spec_or_clause))
        if spec_or_clause is None:
            return None

    elif isinstance(spec_or_clause, DNFSpecification):
        spec_or_clause = spec_or_clause
    else:
        raise TypeError("Unsupported specification type provided to CheckSatisfiability.")

    if not isinstance(spec_or_clause, DNFSpecification):
        raise TypeError("Internal error: expected DNFSpecification after normalization.")

    # Rewrite unsupported ops (e.g., atan2/div) into AutoLiRPA-friendly form.
    bounds = dict(spec_or_clause.variables)
    inv_counter: List[int] = [0]
    rewritten_clauses: List[DNFClause] = []
    for clause in spec_or_clause.clauses:
        new_constraints: List[Constraint] = []
        extra_constraints: List[Constraint] = []
        for c in clause.constraints:
            left, extra_left = _rewrite_expr_for_autolirpa(c.left, bounds, inv_counter)
            right, extra_right = _rewrite_expr_for_autolirpa(c.right, bounds, inv_counter)
            new_constraints.append(Constraint(left=left, operator=c.operator, right=right))
            extra_constraints.extend(extra_left)
            extra_constraints.extend(extra_right)
        if extra_constraints:
            new_constraints.extend(extra_constraints)
        rewritten_clauses.append(DNFClause(constraints=new_constraints))

    return _dnf_clauses_to_specification(rewritten_clauses)
