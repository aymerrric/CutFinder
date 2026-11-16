from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

Arc = Tuple[int, int]  # (i, j): directed arc from node i to node j

# Baseline cut (seed of the first generation, format example for the initializer): a weak known
# cut, so that the strong linking x[i, j, k] <= min(u[i, j], d[k]) * y[i, j] is left to
# rediscover. runs/analysis/mcnd_sizes_20261113.txt: it closes 0-31% of the root gap, the strong
# linking 90-100%.
BASELINE_IDEA: str = (
    "Enough open arcs out of every origin: the flow leaving node s is at least the total demand of "
    "the commodities that start at s, and only open arcs carry flow, so at least q_s arcs leaving s "
    "are open, q_s being the smallest q whose q largest capacities out of s sum to that supply: "
    "sum_j y[s, j] >= q_s. Valid for every feasible solution; the LP opens fractions of arcs, which "
    "this integer count forbids."
)
BASELINE_CODE: str = (
    "y, u, d, O = m._y, m._u, m._d, m._O\n"
    "for s in set(O.values()):\n"
    "    supply = sum(d[k] for k in m._K if O[k] == s)\n"
    "    caps = sorted((u[i, j] for (i, j) in m._A if i == s), reverse=True)\n"
    "    q = next((n for n in range(1, len(caps) + 1) if sum(caps[:n]) >= supply), len(caps))\n"
    "    m.addConstr(gp.quicksum(y[i, j] for (i, j) in m._A if i == s) >= q, name=f\"origin_arcs[{s}]\")"
)


def create_model(
    N: List[int], A: List[Arc], K: List[int], c: Dict[Arc, float], f: Dict[Arc, float],
    u: Dict[Arc, float], d: Dict[int, float], O: Dict[int, int], D: Dict[int, int],
) -> gp.Model:
    """
    Multi-Commodity capacitated fixed-charge Network Design, the compact arc-based formulation of
    the paper (Appendix F.2, Eq. MCND; same as the reference mcnd.yaml): open arcs and route every
    commodity from its origin to its destination within the capacities of the open arcs, at
    minimum opening + routing cost.

    Arguments: N nodes, A directed arcs, K commodities, c[i, j] cost per unit of flow on arc
    (i, j), f[i, j] cost of opening it, u[i, j] its capacity, d[k] demand of commodity k, O[k] /
    D[k] its origin / destination.

    Variables:
        x[i, j, k] >= 0 : flow of commodity k on arc (i, j)
        y[i, j] binary  : 1 if arc (i, j) is open

    Flow conservation as in the paper: the flow leaving the origin equals d[k], the flow entering
    the destination equals d[k], and every other node is balanced. (The textbook form writes
    out - in = d[k] at the origin; with positive unit costs both have the same optimal solutions.)
    Capacity: the total flow on an arc is at most u[i, j] if it is open, 0 otherwise (aggregated:
    the strong linking x[i, j, k] <= min(u[i, j], d[k]) * y[i, j] is NOT in the model).

    The variables and data are also stored on the model (m._x, m._y, m._N, m._A, m._K, m._c,
    m._f, m._u, m._d, m._O, m._D) so that code receiving only the model (e.g. a cut proposed by
    the LLM) can reach them.
    """
    m: gp.Model = gp.Model("mcnd")
    x: gp.tupledict = m.addVars([(i, j, k) for (i, j) in A for k in K], lb=0.0, name="x")
    y: gp.tupledict = m.addVars(A, vtype=GRB.BINARY, name="y")

    # objective: routing costs + opening costs
    m.setObjective(gp.quicksum(c[i, j] * x[i, j, k] for (i, j) in A for k in K) + y.prod(f), GRB.MINIMIZE)

    successors: DefaultDict[int, List[int]] = defaultdict(list)
    predecessors: DefaultDict[int, List[int]] = defaultdict(list)
    for i, j in A:
        successors[i].append(j)
        predecessors[j].append(i)

    def flow(i: int, k: int) -> Tuple[gp.LinExpr, float]:
        """(left-hand side, right-hand side) of the flow conservation of commodity k at node i."""
        out: gp.LinExpr = gp.quicksum(x[i, j, k] for j in successors[i])
        into: gp.LinExpr = gp.quicksum(x[j, i, k] for j in predecessors[i])
        if i == O[k]:
            return out, d[k]
        if i == D[k]:
            return into, d[k]
        return out - into, 0.0

    for i in N:
        for k in K:
            lhs, rhs = flow(i, k)
            m.addConstr(lhs == rhs, name=f"flow[{i},{k}]")
    # an arc carries at most its capacity, and only if it is open
    m.addConstrs((x.sum(i, j, "*") <= u[i, j] * y[i, j] for (i, j) in A), name="capacity")

    m._x, m._y, m._N, m._A, m._K = x, y, N, A, K
    m._c, m._f, m._u, m._d, m._O, m._D = c, f, u, d, O, D
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
