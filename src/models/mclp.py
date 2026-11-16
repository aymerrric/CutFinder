from typing import Dict, List

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

# Maximal Covering Location Problem, the reference's formulation (configs/prompts/mclp.yaml). It
# maximizes the covered population; our base models minimize (problems.py), so the objective is
# its negation. Not in the paper: the reference code only.

# Baseline cut (seed of the first generation, format example for the initializer): chosen after
# measurement (runs/analysis/mclp_sizes_20261113.txt), see the BASELINE_* comment there.
BASELINE_IDEA: str = ""
BASELINE_CODE: str = ""


def create_model(I: List[int], J: List[int], Ni: Dict[int, List[int]], a: Dict[int, int], P: int) -> gp.Model:
    """
    Open exactly P of the candidate sites J; a demand node i of I is covered when at least one open
    site is in Ni[i] (the sites within reach of i). Maximize the covered population sum a[i] over
    covered nodes; the model MINIMIZES its negation.

    Variables (binary): x[j] = 1 if site j is open, y[i] = 1 if node i is counted as covered.
    Objective: minimize -sum_i a[i] * y[i].
    Constraints: y[i] <= sum_{j in Ni[i]} x[j] for every node (cover), sum_j x[j] = P (budget).

    The variables and data are also stored on the model (m._x, m._y, m._I, m._J, m._Ni, m._a, m._P)
    so that code receiving only the model (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("mclp")
    x: gp.tupledict = m.addVars(J, vtype=GRB.BINARY, name="x")
    y: gp.tupledict = m.addVars(I, vtype=GRB.BINARY, name="y")
    m.setObjective(-gp.quicksum(a[i] * y[i] for i in I), GRB.MINIMIZE)
    m.addConstrs((y[i] <= gp.quicksum(x[j] for j in Ni[i]) for i in I), name="cover")
    m.addConstr(x.sum() == P, name="budget")
    m._x, m._y, m._I, m._J, m._Ni, m._a, m._P = x, y, I, J, Ni, a, P
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
