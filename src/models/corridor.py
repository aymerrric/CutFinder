from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

# Wildlife corridor design (the connected subgraph problem with node costs and utilities), the
# single-commodity flow formulation of the corridor generator's MIP (Corridor/lib/corMIPGen.py,
# Distributional MIPLIB generators; Conrad et al. 2012, Dilkina and Gomes 2010). It maximizes the
# utility of the bought cells; our base models minimize (problems.py), so the objective is negated,
# as the generator writes it. Their extra variable x_0 (x_0 + y = n, x_0 >= 0) is left out: it only
# says y <= n, implied by y = sum x. Not in the paper.
Edge = Tuple[int, int]

# Baseline cut (seed of the first generation, format example for the initializer): chosen weak.
# runs/analysis/corridor_sizes_20261113.txt: it closes 0.4-1.2% of the root gap (orders 12, 20);
# "a bought cell has a bought neighbour" closes 25-39% alone, so it is NOT given.
BASELINE_IDEA: str = (
    "Flow only leaves bought cells: a cell that is not bought keeps nothing and passes nothing on, "
    "so f[i, j] <= n * x[i] for every arc (i, j), like the given f[i, j] <= n * x[j]."
)
BASELINE_CODE: str = (
    "x, f, n = m._x, m._f, len(m._nodes)\n"
    "m.addConstrs((f[i, j] <= n * x[i] for (i, j) in m._arcs), name=\"flow_out\")"
)


def create_model(nodes: List[int], edges: List[Edge], reserves: List[int], cost: Dict[int, int],
                 utility: Dict[int, int], budget: float) -> gp.Model:
    """
    Buy cells (nodes of a graph) within a budget so that the bought cells form a connected
    subgraph containing every reserve; maximize the total utility of the bought cells. The model
    MINIMIZES its negation.

    Connectivity is a single-commodity flow: a source sends y = (number of bought cells) units into
    the first reserve r; every bought cell keeps one unit; flow uses arcs (both directions of every
    edge) only into bought cells.

    Variables: x[i] binary, 1 if cell i is bought; f[i, j] >= 0 the flow on arc (i, j); y >= 0
    the flow from the source into r. n = number of cells.
    Objective: minimize -sum_i utility[i] * x[i].
    Constraints: sum_i cost[i] * x[i] <= budget (budget); x[t] = 1 for every reserve t (reserve);
    y = sum_i x[i] (source); f[i, j] <= n * x[j] for every arc (arc); for every cell j:
    inflow(j) (+ y if j = r) = x[j] + outflow(j) (balance).

    The variables and data are also stored on the model (m._x, m._f, m._y, m._nodes, m._edges,
    m._arcs, m._reserves, m._cost, m._utility, m._budget) so that code receiving only the model
    (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("corridor")
    n: int = len(nodes)
    arcs: List[Edge] = [(i, j) for (i, j) in edges] + [(j, i) for (i, j) in edges]
    root: int = reserves[0]
    x: gp.tupledict = m.addVars(nodes, vtype=GRB.BINARY, name="x")
    f: gp.tupledict = m.addVars(arcs, lb=0.0, name="f")
    y: gp.Var = m.addVar(lb=0.0, name="y")
    m.setObjective(-gp.quicksum(utility[i] * x[i] for i in nodes), GRB.MINIMIZE)
    m.addConstr(gp.quicksum(cost[i] * x[i] for i in nodes) <= budget, name="budget")
    m.addConstrs((x[t] == 1 for t in reserves), name="reserve")
    m.addConstr(y == x.sum(), name="source")
    m.addConstrs((f[i, j] <= n * x[j] for (i, j) in arcs), name="arc")
    m.addConstrs((f.sum("*", j) + (y if j == root else 0) == x[j] + f.sum(j, "*") for j in nodes), name="balance")
    m._x, m._f, m._y = x, f, y
    m._nodes, m._edges, m._arcs, m._reserves = nodes, edges, arcs, reserves
    m._cost, m._utility, m._budget = cost, utility, budget
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
