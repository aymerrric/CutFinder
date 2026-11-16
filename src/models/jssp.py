from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

Operation = Tuple[int, int]  # (j, k): the k-th operation of job j
Pair = Tuple[int, int, int, int]  # (j1, k1, j2, k2): two operations on the same machine, (j1, k1) < (j2, k2)

# Baseline cut (seed of the first generation, format example for the initializer): the classic
# one-machine bound. Every machine processes all its operations one at a time, so the makespan is
# at least the total processing time of each machine. Valid for every feasible schedule.
BASELINE_IDEA: str = (
    "Machine load bound: a machine processes its operations one at a time, so the schedule lasts at "
    "least the sum of the processing times of the operations on that machine: for every machine "
    "r, Cmax >= sum of p[j, k] over the operations (j, k) with machines[j][k] == r. Valid for every "
    "feasible schedule; the LP relaxation, whose big-M disjunctions are almost free, ignores it."
)
BASELINE_CODE: str = (
    "Cmax, p = m._Cmax, m._p\n"
    "for r in range(n_machines):\n"
    "    load = sum(p[j, k] for (j, k) in p if machines[j][k] == r)\n"
    "    m.addConstr(Cmax >= load, name=f\"machine_load[{r}]\")"
)


def create_model(n_jobs: int, n_machines: int, times: List[List[int]], machines: List[List[int]]) -> gp.Model:
    """
    Job Shop Scheduling Problem, disjunctive big-M formulation (same as the reference jssp.yaml):
    every job j runs its operations k = 0..n_machines-1 in order, operation (j, k) uses machine
    machines[j][k] for times[j][k], a machine runs one operation at a time; minimize the makespan.

    Variables:
        S[j, k]              continuous >= 0 : start time of operation (j, k)
        Cmax                 continuous >= 0 : makespan
        y[j1, k1, j2, k2]    binary : for two operations on the same machine, with
                             (j1, k1) < (j2, k2): 1 if (j1, k1) runs before (j2, k2)

    The variables and data are also stored on the model (m._S, m._Cmax, m._y, m._p, m._bigM) so
    that code receiving only the model (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("jssp")
    operations: List[Operation] = [(j, k) for j in range(n_jobs) for k in range(n_machines)]
    p: Dict[Operation, int] = {(j, k): times[j][k] for (j, k) in operations}
    pairs: List[Pair] = [
        (j1, k1, j2, k2) for (j1, k1) in operations for (j2, k2) in operations
        if (j1, k1) < (j2, k2) and machines[j1][k1] == machines[j2][k2]
    ]
    big_m: int = sum(sum(row) for row in times)  # the reference's simple safe big-M

    S: gp.tupledict = m.addVars(operations, lb=0.0, name="S")
    Cmax: gp.Var = m.addVar(lb=0.0, name="Cmax")
    y: gp.tupledict = m.addVars(pairs, vtype=GRB.BINARY, name="y")

    m.setObjective(Cmax, GRB.MINIMIZE)

    # (1) the operations of a job run in order
    m.addConstrs((S[j, k + 1] >= S[j, k] + p[j, k] for j in range(n_jobs) for k in range(n_machines - 1)),
                 name="precedence")
    # (2) two operations on the same machine do not overlap: one of them runs first
    m.addConstrs((S[j1, k1] + p[j1, k1] <= S[j2, k2] + big_m * (1 - y[j1, k1, j2, k2])
                  for (j1, k1, j2, k2) in pairs), name="disj1")
    m.addConstrs((S[j2, k2] + p[j2, k2] <= S[j1, k1] + big_m * y[j1, k1, j2, k2]
                  for (j1, k1, j2, k2) in pairs), name="disj2")
    # (3) the makespan is after the end of every job
    m.addConstrs((Cmax >= S[j, n_machines - 1] + p[j, n_machines - 1] for j in range(n_jobs)), name="makespan")

    m._S, m._Cmax, m._y, m._p, m._bigM = S, Cmax, y, p, big_m
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
