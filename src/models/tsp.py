from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from loaders.tsp import Costs
from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

Arc = Tuple[int, int]  # (i, j): travel from city i to city j


def create_model(n: int, c: Costs) -> gp.Model:
    """
    TSP with the MTZ formulation, city 1 as depot (same as the reference tsp.yaml).

    Variables:
        x[i, j] binary : 1 if the tour goes directly from i to j
        u[i]   integer : position of city i in the tour (u[1] = 1, u[i] in [2, n] otherwise)

    The variables are also stored on the model (m._x, m._u, m._n) so that code receiving only
    the model (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("tsp")
    cities: range = range(1, n + 1)
    arcs: List[Arc] = list(c.keys())  # every ordered pair (i, j) with i != j

    x: gp.tupledict = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    # bounds put directly on the variables (the reference uses separate constraints instead)
    u: gp.tupledict = m.addVars(cities, lb=2, ub=n, vtype=GRB.INTEGER, name="u")
    u[1].LB = u[1].UB = 1  # depot is first

    # objective: total travel cost
    m.setObjective(x.prod(c), GRB.MINIMIZE)

    # each city is left exactly once and entered exactly once
    m.addConstrs((x.sum(i, "*") == 1 for i in cities), name="out")
    m.addConstrs((x.sum("*", j) == 1 for j in cities), name="in")

    # MTZ subtour elimination: if x[i, j] = 1 then u[j] >= u[i] + 1 (depot excluded)
    m.addConstrs(
        (u[i] - u[j] + n * x[i, j] <= n - 1 for (i, j) in arcs if i != 1 and j != 1),
        name="mtz",
    )

    m._x, m._u, m._n = x, u, n
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
