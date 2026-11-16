from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

# Generalized Independent Set Problem, the formulation of Distributional MIPLIB's generator
# (GISP/gisp.py; Hochbaum and Pathria 1997, Colombi, Mansini and Savelsbergh 2017). It maximizes
# the weight of the chosen nodes minus the cost of the removed edges; our base models minimize
# (problems.py), so the objective is negated, as the generator itself writes it. Not in the paper.
Edge = Tuple[int, int]

# Baseline cut (seed of the first generation, format example for the initializer): chosen weak,
# so that the textbook clique family is left to find. runs/analysis/gisp_sizes_20261113.txt: it
# closes 4.5-7.7% of the root gap (n = 75, 150); x_i + x_j + x_k <= 1 on the permanent triangles
# closes 45-61%, so it is NOT given.
BASELINE_IDEA: str = (
    "Triangles with one removable edge: if (i, j) and (i, k) are permanent edges and (j, k) is "
    "removable, i excludes j and k, and j and k can both be chosen only if (j, k) is removed: "
    "x[i] + x[j] + x[k] <= 1 + y[j, k]."
)
BASELINE_CODE: str = (
    "x, y = m._x, m._y\n"
    "neighbours = {i: set() for i in m._nodes}\n"
    "for (i, j) in m._permanent:\n"
    "    neighbours[i].add(j)\n"
    "    neighbours[j].add(i)\n"
    "m.addConstrs((x[i] + x[j] + x[k] <= 1 + y[j, k] for (j, k) in m._removable for i in neighbours[j] & neighbours[k]),\n"
    "             name=\"mixed_triangle\")"
)


def create_model(nodes: List[int], permanent: List[Edge], removable: List[Edge], weight: int, cost: int) -> gp.Model:
    """
    Choose a set of nodes of a graph. Two chosen nodes joined by a permanent edge are forbidden;
    two chosen nodes joined by a removable edge are allowed if the edge is removed, at a cost.
    Maximize weight * (number of chosen nodes) - cost * (number of removed edges); the model
    MINIMIZES its negation.

    Variables (binary): x[i] = 1 if node i is chosen; y[i, j] = 1 if the removable edge (i, j) is
    removed (edges are pairs i < j).
    Objective: minimize -weight * sum_i x[i] + cost * sum_(i, j) y[i, j].
    Constraints: x[i] + x[j] <= 1 for every permanent edge (permanent);
                 x[i] + x[j] - y[i, j] <= 1 for every removable edge (removable).

    The variables and data are also stored on the model (m._x, m._y, m._nodes, m._permanent,
    m._removable, m._weight, m._cost) so that code receiving only the model (e.g. a cut proposed by
    the LLM) can reach them.
    """
    m: gp.Model = gp.Model("gisp")
    x: gp.tupledict = m.addVars(nodes, vtype=GRB.BINARY, name="x")
    y: gp.tupledict = m.addVars(removable, vtype=GRB.BINARY, name="y")
    m.setObjective(-weight * x.sum() + cost * y.sum(), GRB.MINIMIZE)
    m.addConstrs((x[i] + x[j] <= 1 for (i, j) in permanent), name="permanent")
    m.addConstrs((x[i] + x[j] - y[i, j] <= 1 for (i, j) in removable), name="removable")
    m._x, m._y = x, y
    m._nodes, m._permanent, m._removable, m._weight, m._cost = nodes, permanent, removable, weight, cost
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
