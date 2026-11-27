"""
Problem-independent model transformations, used to build formulations for the Solver.
They are top-level functions: combined with functools.partial they stay picklable, so the
Solver can send them to its child process, e.g.
    partial(fixed, build=partial(with_cut, base=create_model, code=code), solution=tour)
"""
import itertools
import math
from typing import Dict, List, Optional

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


def with_stops(build: Formulation, obj_stop: float, bound_stop: float, **args: object) -> gp.Model:
    """
    build(**args) (a minimization) that stops as soon as the answer to "is there a solution of
    cost <= obj_stop?" is known: a solution that good found (BestObjStop: Gurobi status
    USER_OBJ_LIMIT) or a bound >= bound_stop proved (BestBdStop). MIPGap = 0 so that Gurobi does
    not stop at its default 0.01% gap before either one is decided. Cutoff = bound_stop: Gurobi
    discards every node whose own bound reaches bound_stop, and every solution worse than it, so a
    cut whose optimum is above bound_stop ends with status CUTOFF without proving that optimum
    (BestBdStop alone waits for the GLOBAL bound, the smallest over all open nodes, which rises
    late: learnt_results.md 12.4). Used by the safe check of the evaluator, where both are the
    known optimum (plus a tolerance).
    """
    m: gp.Model = build(**args)
    m.Params.BestObjStop = obj_stop
    m.Params.BestBdStop = bound_stop
    m.Params.Cutoff = bound_stop
    m.Params.MIPGap = 0.0
    m.update()
    return m


CUT_ROW_PREFIX: str = "cut_row_"  # name given to the cut's constraints the LLM left unnamed
RELAX_PREFIXES: tuple = ("ArtP_", "ArtN_")  # Gurobi's feasRelax variables: ArtP_<row>, ArtN_<row>


def cut_violations(build: Formulation, base: Formulation, solution: Dict[str, float], **args: object) -> gp.Model:
    """
    The diagnosis of a cut that removes a known optimal solution: build(**args) (the base model +
    the cut) with the variables fixed to `solution`, where only the CUT's linear constraints (the
    rows after the base model's) may be violated, each at a cost of 1 per unit (Gurobi feasRelax).
    Solved, the values of the variables ArtP_<row> / ArtN_<row> are how much each of the cut's
    constraints is violated at that solution. Unnamed cut constraints are renamed cut_row_<k>
    (the k-th constraint the cut adds). Runs in the Solver's child process like any formulation.
    """
    reference: gp.Model = base(**args)
    reference.update()
    n_base: int = reference.NumConstrs
    reference.dispose()
    m: gp.Model = fixed(build, solution, **args)
    rows: List[gp.Constr] = m.getConstrs()[n_base:]
    for k, row in enumerate(rows, 1):
        if row.ConstrName == f"R{row.index}" or not row.ConstrName:  # Gurobi's default name
            row.ConstrName = f"{CUT_ROW_PREFIX}{k}"
    m.update()
    if rows:
        m.feasRelax(0, False, None, None, None, rows, [1.0] * len(rows))
    m.update()
    return m


def read_violations(solution: Dict[str, float], tolerance: float = 1e-6) -> Dict[str, float]:
    """{cut constraint name: violation} from the solution of cut_violations (largest first)."""
    violated: Dict[str, float] = {}
    for name, value in solution.items():
        for prefix in RELAX_PREFIXES:
            if name.startswith(prefix) and value > tolerance:
                row: str = name[len(prefix):]
                violated[row] = violated.get(row, 0.0) + value
    return dict(sorted(violated.items(), key=lambda kv: -kv[1]))
