"""
Counterexample search for a cut (paper plan, step 1): a free filter before any proof.

On tiny instances of each registered MILP, run two tests. TSP, CWLP and JSSP also satisfy the
assumptions of their Lean models.
  - SAFETY (every cut): the optimum with the cut must equal the optimum without it. A cut whose
    model is infeasible, or whose optimum is worse, is REFUTED: the base model's optimal point is
    the counterexample.
  - VALIDITY (cuts that add no variable): no feasible point of the base model may violate a row the
    cut adds. Each row is first evaluated at the feasible points of Gurobi's solution pool, then,
    when none violates it, Gurobi looks for the feasible point that violates it most (a small MIP:
    maximize lhs - rhs over the base model's feasible set). A tightened variable bound counts as a
    row. A cut that only removes non-optimal points is SAFE_NOT_VALID.
Verdicts: REFUTED, SAFE_NOT_VALID, NO_COUNTEREXAMPLE (nothing found: not a proof), ERROR (the code
fails on an instance). The counterexample (instance, point, row, violation) is kept for the file.
"""
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from loaders.base import Instance
from loaders.corridor import lattice
from loaders.gisp import Edge, erdos_renyi, partition_edges
from loaders.imo6 import instance as imo6_instance
from loaders.mcnd import Arc, LoaderMCND
from loaders.validity import mcnd_instance
from models.transforms import with_cut
from problems import PROBLEMS, Problem

REFUTED: str = "REFUTED"
SAFE_NOT_VALID: str = "SAFE_NOT_VALID"
NO_COUNTEREXAMPLE: str = "NO_COUNTEREXAMPLE"
ERROR: str = "ERROR"

N_INSTANCES: int = 200  # tiny instances per problem
POOL_SOLUTIONS: int = 200  # feasible points of the base model kept per instance (solution pool)
ROW_TIME_LIMIT: float = 2.0  # seconds of each "most violated point" MIP
MAX_ROW_SOLVES: int = 300  # "most violated point" MIPs per instance at most (the rows past it are not checked)
TOLERANCE: float = 1e-6  # relative: a violation or an objective increase above it counts


@dataclass
class Counterexample:
    instance: str  # name of the tiny instance
    args: Dict[str, Any]  # its create_model arguments (tuple keys as strings, for the file)
    kind: str  # REFUTED (the optimum changes) or SAFE_NOT_VALID (a non-optimal feasible point is removed)
    row: str  # the cut's row (or bound) the point violates, "lhs <= rhs" in the variables' names
    violation: float  # lhs - rhs at the point (> 0)
    point: Dict[str, float]  # the point: every variable's value (zeros left out)
    base_optimum: Optional[float] = None
    cut_optimum: Optional[float] = None  # None = the model with the cut is infeasible


@dataclass
class Verdict:
    verdict: str
    instances: int = 0  # instances tried (infeasible base models skipped)
    rows: int = 0  # rows the cut adds on the last instance tried (bounds included)
    rows_unchecked: int = 0  # rows left without a "most violated point" MIP (MAX_ROW_SOLVES), summed
    adds_variables: bool = False  # only the safety test then
    nonlinear: bool = False  # the cut adds quadratic or general constraints: only its linear rows are checked
    seconds: float = 0.0
    error: Optional[str] = None
    counterexample: Optional[Counterexample] = None
    not_valid_example: Optional[Counterexample] = None  # kept even when a REFUTED one is found later

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# tiny instances for counterexample search
# ---------------------------------------------------------------------------------------------

def tiny_tsp(rng: random.Random, k: int) -> Instance:
    """4-6 cities; costs with many ties: random 0-3 (symmetric), all equal, cities on a line,
    asymmetric 0-5, all zero."""
    n: int = rng.randint(4, 6)
    family: str = ["ties", "equal", "line", "asymmetric", "zero"][k % 5]
    pairs = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
    c: Dict[Tuple[int, int], float] = {}
    if family == "line":
        pos: Dict[int, int] = {i: rng.randint(0, 3) for i in range(1, n + 1)}
        c = {(i, j): float(abs(pos[i] - pos[j])) for i, j in pairs}
    elif family == "asymmetric":
        c = {(i, j): float(rng.randint(0, 5)) for i, j in pairs}
    else:
        for i, j in pairs:
            if i < j:
                c[i, j] = c[j, i] = {"ties": float(rng.randint(0, 3)), "equal": 1.0, "zero": 0.0}[family]
    return {"name": f"tiny_tsp_{family}_{k}", "args": {"n": n, "c": c}}


def tiny_cwlp(rng: random.Random, k: int) -> Instance:
    """1-3 customers and warehouses, demands 1-5; capacities exact (a subset's total demand),
    loose (the total demand) or equal to the largest demand; opening costs zero or 0-5. Every
    fourth instance is "spread": 3-5 customers on 2-4 warehouses, small demands (1-2) beside one or
    two large ones (10-20), roomy capacities below the total demand, positive opening costs (cuts
    that open more warehouses when demands vary, e.g. learnt_results.md 5.12)."""
    family: str = ["exact", "loose", "largest", "spread"][k % 4]
    if family == "spread":
        customers: List[int] = list(range(1, rng.randint(3, 5) + 1))
        warehouses: List[int] = list(range(1, rng.randint(2, 4) + 1))
        large: List[int] = rng.sample(customers, rng.randint(1, 2))
        d: Dict[int, int] = {i: rng.randint(10, 20) if i in large else rng.randint(1, 2) for i in customers}
        total: int = sum(d.values())
        u: Dict[int, int] = {j: rng.randint(max(d.values()), max(max(d.values()), total - 1)) for j in warehouses}
        f: Dict[int, int] = {j: rng.randint(1, 5) for j in warehouses}
        c: Dict[Tuple[int, int], float] = {(i, j): float(rng.randint(0, 5)) for i in customers for j in warehouses}
        return {"name": f"tiny_cwlp_spread_{k}", "args": {"I": customers, "J": warehouses, "d": d, "u": u, "f": f, "c": c}}
    customers = list(range(1, rng.randint(1, 3) + 1))
    warehouses = list(range(1, rng.randint(1, 3) + 1))
    d = {i: rng.randint(1, 5) for i in customers}
    u = {}
    for j in warehouses:
        if family == "exact":
            subset = [i for i in customers if rng.random() < 0.6] or [rng.choice(customers)]
            u[j] = sum(d[i] for i in subset)
        elif family == "loose":
            u[j] = sum(d.values())
        else:
            u[j] = max(d.values())
    zero_fixed: bool = k % 2 == 0
    f = {j: 0 if zero_fixed else rng.randint(0, 5) for j in warehouses}
    c = {(i, j): float(rng.randint(0, 5)) for i in customers for j in warehouses}
    return {"name": f"tiny_cwlp_{family}_{'f0' if zero_fixed else 'f'}_{k}",
            "args": {"I": customers, "J": warehouses, "d": d, "u": u, "f": f, "c": c}}


def tiny_jssp(rng: random.Random, k: int) -> Instance:
    """1-3 jobs on 1-2 machines, processing times 1-4 (equal times every third instance)."""
    n_jobs: int = rng.randint(1, 3)
    n_machines: int = rng.randint(1, 2)
    equal: bool = k % 3 == 0
    times: List[List[int]] = [[1 if equal else rng.randint(1, 4) for _ in range(n_machines)] for _ in range(n_jobs)]
    machines: List[List[int]] = [rng.sample(range(n_machines), n_machines) for _ in range(n_jobs)]
    return {"name": f"tiny_jssp_{n_jobs}x{n_machines}{'_equal' if equal else ''}_{k}",
            "args": {"n_jobs": n_jobs, "n_machines": n_machines, "times": times, "machines": machines}}


def tiny_mcnd(rng: random.Random, k: int) -> Instance:
    """A forced flow split, a feasible circulation above demand, and shared-capacity networks."""
    demand: int = rng.randint(2, 8)
    if k % 3 == 0:
        arcs: Dict[Arc, Tuple[int, int, int]] = {
            a: (rng.randint(0, 3), demand - 1, rng.randint(0, 4))
            for a in [(1, 2), (2, 4), (1, 3), (3, 4)]}
        return mcnd_instance(f"tiny_mcnd_split_{k}", "neither path can carry the whole demand",
                             arcs, [(1, 4, demand)])
    if k % 3 == 1:
        spare: int = rng.randint(1, demand)
        arcs = {(1, 2): (rng.randint(1, 4), demand + spare, rng.randint(0, 4)),
                (2, 1): (rng.randint(1, 4), spare, rng.randint(0, 4))}
        return mcnd_instance(f"tiny_mcnd_cycle_{k}", "flow can circulate above the net demand",
                             arcs, [(1, 2, demand)])
    inst: Instance = LoaderMCND().generate(1, seed=rng.getrandbits(32), min_nodes=3, max_nodes=5,
                                           min_arcs=4, max_arcs=8, min_commodities=2, max_commodities=3)[0]
    inst["name"] = f"tiny_mcnd_shared_{k}"
    return inst


def tiny_gisp(rng: random.Random, k: int) -> Instance:
    """Empty, permanent-clique, removable-clique, mixed-triangle and random graphs."""
    nodes: List[int] = list(range(rng.randint(3, 5)))
    pairs: List[Edge] = [(i, j) for i in nodes for j in nodes if i < j]
    family: str = ("empty", "permanent", "removable", "triangle", "random")[k % 5]
    permanent: List[Edge] = []
    removable: List[Edge] = []
    if family == "permanent":
        permanent = pairs
    elif family == "removable":
        removable = pairs
    elif family == "triangle":
        i, j, t = sorted(rng.sample(nodes, 3))
        permanent, removable = [(i, j), (i, t)], [(j, t)]
    elif family == "random":
        permanent, removable = partition_edges(erdos_renyi(len(nodes), 0.5, rng), 0.5, rng)
    return {"name": f"tiny_gisp_{family}_{k}", "args": {"nodes": nodes, "permanent": permanent,
            "removable": removable, "weight": rng.randint(1, 5), "cost": rng.randint(1, 5)}}


def tiny_corridor(rng: random.Random, k: int) -> Instance:
    """A forced path, a branching tree, or a square lattice with routes of different cost."""
    family: str = ("path", "star", "grid")[k % 3]
    n: int = rng.randint(4, 6)
    if family == "grid":
        order: int = rng.choice([2, 3])
        n = order * order
        edges: List[Edge] = lattice(order)
        # Buy the top row and the right column to connect opposite corners.
        route: List[int] = list(range(order)) + list(range(2 * order - 1, n, order))
        reserves: List[int] = [0, n - 1, rng.choice(route[1:-1])]
    elif family == "star":
        edges = [(0, i) for i in range(1, n)]
        reserves = rng.sample(range(1, n), 2)
        route = [0, *reserves]
    else:
        edges = [(i, i + 1) for i in range(n - 1)]
        reserves = [0, n - 1]
        route = list(range(n))
    cost: Dict[int, int] = {i: 0 if i in reserves else rng.randint(1, 5) for i in range(n)}
    utility: Dict[int, int] = {i: rng.randint(1, 5) for i in range(n)}
    budget: int = sum(cost[i] for i in route) if k % 2 == 0 else sum(cost.values())
    return {"name": f"tiny_corridor_{family}_{k}", "args": {"nodes": list(range(n)), "edges": edges,
            "reserves": reserves, "cost": cost, "utility": utility, "budget": budget}}


# N is the only input. Bound both the solution-pool search and the per-row violation MIPs.
TINY_IMO6_SIZES: Tuple[int, ...] = (2, 3, 4, 5)


def tiny_imo6(_rng: random.Random, k: int) -> Instance:
    return imo6_instance(TINY_IMO6_SIZES[k])


TINY: Dict[str, Callable[[random.Random, int], Instance]] = {
    "tsp": tiny_tsp, "cwlp": tiny_cwlp, "jssp": tiny_jssp, "mcnd": tiny_mcnd,
    "imo6": tiny_imo6, "gisp": tiny_gisp, "corridor": tiny_corridor,
}


def tiny_instances(problem: str, count: int = N_INSTANCES, seed: int = 0) -> List[Instance]:
    """Up to count cases. IMO6 exhausts its distinct tiny grids in increasing size."""
    if problem not in TINY:
        raise ValueError(f"no tiny instances for {problem!r} (supported: {', '.join(TINY)})")
    if problem == "imo6":
        count = min(count, len(TINY_IMO6_SIZES))
    rng: random.Random = random.Random(seed)
    return [TINY[problem](rng, k) for k in range(count)]


# ---------------------------------------------------------------------------------------------
# the two tests
# ---------------------------------------------------------------------------------------------

Row = Tuple[List[int], List[float], str, float, str]  # var indices, coefficients, sense, rhs, text


def _quiet(m: gp.Model, threads: int) -> None:
    m.Params.OutputFlag = 0
    m.Params.Threads = threads


def _text(idx: List[int], coef: List[float], sense: str, rhs: float, names: List[str]) -> str:
    terms: str = " + ".join(f"{a:g} {names[i]}" for i, a in zip(idx, coef)) or "0"
    return f"{terms} {'<=' if sense == '<' else '>=' if sense == '>' else '=='} {rhs:g}"


def cut_rows(base_model: gp.Model, cut_model: gp.Model) -> List[Row]:
    """The linear rows the cut adds (after the base model's rows) and the variable bounds it
    tightens, as rows over the base model's variables (same indices: the cut adds no variable)."""
    names: List[str] = [v.VarName for v in base_model.getVars()]
    rows: List[Row] = []
    for constr in cut_model.getConstrs()[base_model.NumConstrs:]:
        expr: gp.LinExpr = cut_model.getRow(constr)
        idx: List[int] = [expr.getVar(t).index for t in range(expr.size())]
        coef: List[float] = [expr.getCoeff(t) for t in range(expr.size())]
        rows.append((idx, coef, constr.Sense, constr.RHS, _text(idx, coef, constr.Sense, constr.RHS, names)))
    for b, v in zip(base_model.getVars(), cut_model.getVars()):
        if v.UB < b.UB:
            rows.append(([b.index], [1.0], "<", v.UB, f"{b.VarName} <= {v.UB:g} (bound)"))
        if v.LB > b.LB:
            rows.append(([b.index], [1.0], ">", v.LB, f"{b.VarName} >= {v.LB:g} (bound)"))
    return rows


def violation(row: Row, x: List[float]) -> float:
    """How much the point x violates the row (> 0 = violated), the larger side of an equality."""
    idx, coef, sense, rhs, _ = row
    lhs: float = sum(a * x[i] for i, a in zip(idx, coef))
    if sense == "<":
        return lhs - rhs
    if sense == ">":
        return rhs - lhs
    return abs(lhs - rhs)


def _violated(row: Row, x: List[float]) -> bool:
    return violation(row, x) > TOLERANCE * max(1.0, abs(row[3]))


def _point(m: gp.Model, x: List[float]) -> Dict[str, float]:
    return {v.VarName: round(val, 9) for v, val in zip(m.getVars(), x) if abs(val) > 1e-9}


def _pool(m: gp.Model) -> List[List[float]]:
    points: List[List[float]] = []
    variables: List[gp.Var] = m.getVars()
    if not variables:
        return points
    if not m.IsMIP:  # an LP (e.g. one job: no disjunction): no pool, its optimal point only
        return [m.getAttr("X", variables)]
    for k in range(m.SolCount):
        m.Params.SolutionNumber = k
        points.append(m.getAttr("Xn", variables))
    return points


def most_violated(base_model: gp.Model, row: Row, time_limit: float) -> Optional[List[float]]:
    """A feasible point of the base model that violates `row`, or None (none found within the
    time limit): maximize the violation, stopping as soon as one is positive."""
    idx, coef, sense, rhs, _ = row
    variables: List[gp.Var] = base_model.getVars()
    expr: gp.LinExpr = gp.LinExpr(coef, [variables[i] for i in idx])
    found: Optional[List[float]] = None
    for direction in ([1.0] if sense == "<" else [-1.0] if sense == ">" else [1.0, -1.0]):
        base_model.setObjective(direction * (expr - rhs), GRB.MAXIMIZE)
        base_model.Params.TimeLimit = time_limit
        base_model.Params.BestObjStop = TOLERANCE * max(1.0, abs(rhs)) * 2
        base_model.Params.PoolSolutions = 1
        base_model.optimize()
        if base_model.SolCount > 0 and base_model.ObjVal > TOLERANCE * max(1.0, abs(rhs)):
            found = base_model.getAttr("X", variables)
            break
    return found


def _serializable(args: Dict[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return convert(args)


def falsify(problem_name: str, code: str, instances: Optional[List[Instance]] = None, threads: int = 1,
            row_time_limit: float = ROW_TIME_LIMIT, max_row_solves: int = MAX_ROW_SOLVES,
            check_validity: bool = True) -> Verdict:
    """Search a counterexample to `code` on `instances` (default: the problem's tiny ones). Stops
    at the first REFUTED instance; a SAFE_NOT_VALID point found earlier is kept beside it."""
    problem: Problem = PROBLEMS[problem_name]
    instances = tiny_instances(problem_name) if instances is None else instances
    verdict: Verdict = Verdict(NO_COUNTEREXAMPLE)
    start: float = time.time()
    try:
        for inst in instances:
            args: Dict[str, Any] = inst["args"]
            base_model: gp.Model = problem.base(**args)
            cut_model: Optional[gp.Model] = None
            try:
                _quiet(base_model, threads)
                base_model.Params.MIPGap = 0.0
                base_model.Params.PoolSearchMode = 2
                base_model.Params.PoolSolutions = POOL_SOLUTIONS
                base_model.optimize()
                if base_model.Status != GRB.OPTIMAL:  # infeasible tiny instance (e.g. capacities): skipped
                    continue
                verdict.instances += 1
                base_value: float = base_model.ObjVal
                optimum: List[float] = base_model.getAttr("X", base_model.getVars()) if base_model.NumVars else []
                pool: List[List[float]] = _pool(base_model)
                try:
                    cut_model = with_cut(problem.base, code, **args)
                except Exception as e:  # the code fails on this instance
                    verdict.verdict, verdict.error = ERROR, f"{inst['name']}: {type(e).__name__}: {e}"
                    return verdict
                _quiet(cut_model, threads)
                verdict.adds_variables = verdict.adds_variables or cut_model.NumVars != base_model.NumVars
                verdict.nonlinear = verdict.nonlinear or (
                    cut_model.NumQConstrs > base_model.NumQConstrs or cut_model.NumGenConstrs > base_model.NumGenConstrs
                    or cut_model.NumSOS > base_model.NumSOS)
                rows: List[Row] = [] if verdict.adds_variables else cut_rows(base_model, cut_model)
                verdict.rows = len(rows)

                # safety: the optimum must not change
                cut_model.Params.MIPGap = 0.0
                cut_model.optimize()
                cut_value: Optional[float] = cut_model.ObjVal if cut_model.Status == GRB.OPTIMAL else None
                if cut_model.Status not in (GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                    continue  # not decided (time limit): no verdict from this instance
                if cut_value is None or cut_value > base_value + TOLERANCE * max(1.0, abs(base_value)):
                    broken: List[Row] = [r for r in rows if _violated(r, optimum)]
                    worst: Optional[Row] = max(broken, key=lambda r: violation(r, optimum)) if broken else None
                    verdict.verdict = REFUTED
                    verdict.counterexample = Counterexample(
                        inst["name"], _serializable(args), REFUTED, worst[4] if worst else "(not a linear row of the cut)",
                        violation(worst, optimum) if worst else 0.0, _point(base_model, optimum), base_value, cut_value)
                    return verdict

                # validity: no feasible point may violate a row of the cut
                if not check_validity or verdict.not_valid_example is not None or not rows:
                    continue
                example: Optional[Tuple[Row, List[float]]] = next(
                    ((r, x) for r in rows for x in pool if _violated(r, x)), None)
                if example is None:
                    base_model.Params.PoolSearchMode = 0
                    unchecked: List[Row] = []
                    for n_solved, r in enumerate(rows):
                        if n_solved >= max_row_solves:
                            unchecked = rows[n_solved:]
                            break
                        x = most_violated(base_model, r, row_time_limit)
                        if x is not None and _violated(r, x):
                            example = (r, x)
                            break
                    verdict.rows_unchecked += len(unchecked)
                if example is not None:
                    r, x = example
                    verdict.verdict = SAFE_NOT_VALID
                    verdict.not_valid_example = Counterexample(
                        inst["name"], _serializable(args), SAFE_NOT_VALID, r[4], violation(r, x), _point(base_model, x),
                        base_value, cut_value)
                    verdict.counterexample = verdict.not_valid_example
            finally:
                base_model.dispose()
                if cut_model is not None:
                    cut_model.dispose()
        return verdict
    finally:
        verdict.seconds = round(time.time() - start, 2)
