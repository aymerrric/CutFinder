"""
Problem-independent model transformations, used to build formulations for the Solver.
They are top-level functions: combined with functools.partial they stay picklable, so the
Solver can send them to its child process, e.g.
    partial(fixed, build=partial(with_cut, base=create_model, code=code), solution=tour)
"""
import itertools
import math
from typing import Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from solvers.solver import Formulation


def with_cut(base: Formulation, code: str, **args: object) -> gp.Model:
    """
    The base model plus a cut written by the LLM. `code` runs right after base(**args) with
    these names defined (as promised in prompts/common.yaml): m, gp, GRB, itertools, math and
    every argument of the base formulation (n, c for the TSP).
    """
    m: gp.Model = base(**args)
    names: Dict[str, object] = {"m": m, "gp": gp, "GRB": GRB, "itertools": itertools, "math": math}
    exec(code, {**names, **args})  # noqa: S102 (LLM code, run in the Solver's child process)
    m.update()
    return m


def relaxed(build: Formulation, **args: object) -> gp.Model:
    """LP relaxation of build(**args): same variables (same names), integrality dropped."""
    m: gp.Model = build(**args)
    m.update()
    lp: gp.Model = m.relax()
    m.dispose()
    return lp


def fixed(build: Formulation, solution: Dict[str, float], **args: object) -> gp.Model:
    """
    build(**args) with every variable of `solution` fixed to its value (by variable name).
    Variables the model does not have are skipped, and variables the solution does not have
    (e.g. auxiliary variables of a cut) stay free. Integer values are rounded (0.9999999 -> 1)
    and every value is clipped to the variable's bounds.
    """
    m: gp.Model = build(**args)
    m.update()
    for name, value in solution.items():
        var: Optional[gp.Var] = m.getVarByName(name)
        if var is None:
            continue
        if var.VType != GRB.CONTINUOUS:
            value = round(value)
        value = min(max(value, var.LB), var.UB)
        var.LB = var.UB = value
    m.update()
    return m
