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
"""SMT2 helpers to load non-ODE benchmarks into the ABCROWN SMT interface."""

from __future__ import annotations

from typing import Optional, Union


_ODE_TOKENS = ("(define-ode", "(integral", "d/dt[", "forall_t")


def _detect_ode_syntax(text: str) -> Optional[str]:
    for token in _ODE_TOKENS:
        if token in text:
            return token
    return None


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def parse_smt2_formula(path: str, reject_ode: bool = True):
    """Parse SMT2 file into a Formula that the ABCROWN SMT layer can consume."""
    text = _read_text(path)
    if reject_ode:
        token = _detect_ode_syntax(text)
        if token is not None:
            from . import Smt2ParseError
            raise Smt2ParseError(f"ODE syntax '{token}' is not supported in this SMT2 reader.")
    from . import parse_smt2_string
    return parse_smt2_string(text)


def parse_smt2_spec(path: str, reject_ode: bool = True):
    """Parse SMT2 file into a DNFSpecification with variable bounds."""
    from . import Smt2ParseError, _normalize_to_dnf_spec
    formula = parse_smt2_formula(path, reject_ode=reject_ode)
    spec = _normalize_to_dnf_spec(formula)
    if spec is None:
        raise Smt2ParseError("SMT2 input produced no satisfiable clauses.")
    return spec


def parse_smt2_problem(path: str, reject_ode: bool = True):
    """Parse SMT2 file into (Formula, objective, sense)."""
    text = _read_text(path)
    if reject_ode:
        token = _detect_ode_syntax(text)
        if token is not None:
            from . import Smt2ParseError
            raise Smt2ParseError(f"ODE syntax '{token}' is not supported in this SMT2 reader.")
    from . import parse_smt2_string_with_objective
    return parse_smt2_string_with_objective(text)


def check_smt2_file(path: str, delta: Union[float, "Config"] = 0.001, reject_ode: bool = True):
    """Solve an SMT2 file by parsing it into ABCROWN SMT constraints."""
    from . import CheckSatisfiability
    formula = parse_smt2_formula(path, reject_ode=reject_ode)
    return CheckSatisfiability(formula, delta=delta)
