from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

Pair = Tuple[int, int]  # (i, j): customer i served by warehouse j

# Baseline cut (seed of the first generation, format example for the initializer): a weak known cut,
# so that the strong one (the disaggregated linking x[i, j] <= y[j]) is left to rediscover.
# runs/analysis/cwlp_sizes_20261102.txt: it closes 1.5-10.8% of the root gap, the linking 57-84%.
BASELINE_IDEA: str = (
    "Minimum number of open warehouses: the open warehouses must hold the total demand, so at least "
    "k_min of them are open, k_min being the smallest k whose k largest capacities sum to the total "
    "demand: sum_j y[j] >= k_min. Valid for every feasible solution; the LP can open fractions of "
    "warehouses, which this integer count forbids."
)
BASELINE_CODE: str = (
    "y, d, u = m._y, m._d, m._u\n"
    "total = sum(d.values())\n"
    "caps = sorted(u.values(), reverse=True)\n"
    "k_min = next(k for k in range(1, len(caps) + 1) if sum(caps[:k]) >= total)\n"
    "m.addConstr(y.sum() >= k_min, name=\"min_open\")"
)


def create_model(
    I: List[int], J: List[int], d: Dict[int, int], u: Dict[int, int], f: Dict[int, int],
    c: Dict[Pair, float],
) -> gp.Model:
    """
    Capacitated Warehouse Location Problem with unsplittable demands (same as the reference
    cwlp.yaml): open warehouses and assign every customer to exactly one open warehouse, within
    its capacity, at minimum opening + assignment cost.

    Arguments: I customers, J warehouses, d[i] demand, u[j] capacity, f[j] opening cost,
    c[i, j] cost of serving all of customer i's demand from warehouse j.

    Variables:
        x[i, j] binary : 1 if customer i is served by warehouse j
        y[j]    binary : 1 if warehouse j is open

    The variables and data are also stored on the model (m._x, m._y, m._I, m._J, m._d, m._u) so
    that code receiving only the model (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("cwlp")
    x: gp.tupledict = m.addVars(c.keys(), vtype=GRB.BINARY, name="x")
    y: gp.tupledict = m.addVars(J, vtype=GRB.BINARY, name="y")

    # objective: opening costs + assignment costs
    m.setObjective(y.prod(f) + x.prod(c), GRB.MINIMIZE)

    # every customer is served by exactly one warehouse
    m.addConstrs((x.sum(i, "*") == 1 for i in I), name="assign")
    # a warehouse serves at most its capacity, and only if it is open (aggregated: the textbook
    # disaggregated linking x[i, j] <= y[j] is NOT in the model)
    m.addConstrs((gp.quicksum(d[i] * x[i, j] for i in I) <= u[j] * y[j] for j in J), name="capacity")

    m._x, m._y, m._I, m._J, m._d, m._u = x, y, I, J, d, u
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
