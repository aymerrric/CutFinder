import itertools
from typing import Dict, List, Tuple

import gurobipy as gp
import pytest
from gurobipy import GRB

from loaders.base import Instance
from loaders.mcnd import LoaderMCND
from models.mcnd import BASELINE_CODE, create_model, create_model_paper
from models.transforms import with_cut

Arc = Tuple[int, int]


def tiny(seed: int) -> Instance:
    return LoaderMCND(data_dir="/tmp").generate(1, seed=seed, min_nodes=4, max_nodes=4, min_arcs=7, max_arcs=7,
                                                min_commodities=3, max_commodities=3)[0]


def routing_cost(open_arcs: List[Arc], N: List[int], K: List[int], c: Dict[Arc, float], u: Dict[Arc, float],
                 d: Dict[int, float], O: Dict[int, int], D: Dict[int, int]) -> float:
    """Cheapest routing on the open arcs only, written independently of create_model, in the textbook
    form (out - in = d at the origin, -d at the destination): inf if the demands do not fit."""
    m: gp.Model = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars([(i, j, k) for (i, j) in open_arcs for k in K], lb=0.0)
    for i in N:
        for k in K:
            balance = (gp.quicksum(x[i, j, k] for (a, j) in open_arcs if a == i)
                       - gp.quicksum(x[j, i, k] for (j, b) in open_arcs if b == i))
            m.addConstr(balance == (d[k] if i == O[k] else -d[k] if i == D[k] else 0))
    for arc in open_arcs:
        m.addConstr(gp.quicksum(x[arc[0], arc[1], k] for k in K) <= u[arc])
    m.setObjective(gp.quicksum(c[i, j] * x[i, j, k] for (i, j) in open_arcs for k in K))
    m.optimize()
    value: float = m.ObjVal if m.Status == GRB.OPTIMAL else float("inf")
    m.dispose()
    return value


def brute_force(N: List[int], A: List[Arc], K: List[int], c: Dict[Arc, float], f: Dict[Arc, float],
                u: Dict[Arc, float], d: Dict[int, float], O: Dict[int, int], D: Dict[int, int]) -> float:
    """Every set of open arcs, each with its cheapest routing."""
    best: float = float("inf")
    for size in range(len(A) + 1):
        for open_arcs in itertools.combinations(A, size):
            best = min(best, sum(f[a] for a in open_arcs) + routing_cost(list(open_arcs), N, K, c, u, d, O, D))
    return best


def solve(model: gp.Model) -> float:
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    return model.ObjVal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimum_equals_brute_force(seed: int) -> None:
    inst: Instance = tiny(seed)
    assert solve(create_model(**inst["args"])) == pytest.approx(brute_force(**inst["args"]))


def test_model_size_and_attributes() -> None:
    inst: Instance = LoaderMCND(data_dir="/tmp").generate(1, seed=5, min_nodes=8, max_nodes=8, min_arcs=20, max_arcs=20,
                                                          min_commodities=6, max_commodities=6)[0]
    a = inst["args"]
    m: gp.Model = create_model(**a)
    # the paper's Table 20 counts: |A| |K| flows + |A| designs; |N| |K| conservation + |A| capacity rows
    assert m.NumVars == 20 * 6 + 20 and m.NumConstrs == 8 * 6 + 20
    assert m.NumBinVars == 20 and m.ModelSense == GRB.MINIMIZE
    assert set(m._x.keys()) == {(i, j, k) for (i, j) in a["A"] for k in a["K"]} and set(m._y.keys()) == set(a["A"])
    assert m._d is a["d"] and m._O is a["O"] and m._D is a["D"] and m._u is a["u"] and m._N is a["N"]
    assert m.getConstrByName("capacity[%d,%d]" % a["A"][0]) is not None
    assert m.getConstrByName(f"flow[{a['O'][1]},1]").RHS == a["d"][1]


def test_the_strong_linking_is_valid_and_tightens_the_lp() -> None:
    """x[i, j, k] <= min(u, d[k]) y[i, j] keeps the optimum and raises the LP bound (the known strong
    inequality the seed cut must leave to the LLM)."""
    inst: Instance = LoaderMCND(data_dir="/tmp").generate(1, seed=2, min_nodes=8, max_nodes=8, min_arcs=24,
                                                          max_arcs=24, min_commodities=10, max_commodities=10)[0]
    base: gp.Model = create_model(**inst["args"])
    strong: gp.Model = create_model(**inst["args"])
    x, y, u, d = strong._x, strong._y, strong._u, strong._d
    strong.addConstrs(x[i, j, k] <= min(u[i, j], d[k]) * y[i, j] for (i, j, k) in x.keys())
    assert solve(strong) == pytest.approx(solve(base))
    assert solve(strong.relax()) > solve(base.relax()) + 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_seed_cut_keeps_the_optimum(seed: int) -> None:
    inst: Instance = tiny(seed)
    assert solve(with_cut(create_model, BASELINE_CODE, **inst["args"])) == pytest.approx(brute_force(**inst["args"]))


def test_the_seed_cut_tightens_the_lp_of_a_validation_size_instance() -> None:
    """It must be useful (cut off the base LP optimum) on the validation sizes of problems.py."""
    from problems import PROBLEMS
    inst: Instance = LoaderMCND(data_dir="/tmp").generate(1, seed=1, **PROBLEMS["mcnd"].validation_sizes)[0]
    base: float = solve(create_model(**inst["args"]).relax())
    assert solve(with_cut(create_model, BASELINE_CODE, **inst["args"]).relax()) > base + 1e-6


def loops() -> Dict[str, object]:
    """One commodity 1 -> 4 on an expensive path 1 -> 2 -> 3 -> 4, with cheap loops 1 -> 5 -> 1
    and 4 -> 6 -> 4 at its two ends."""
    A: List[Arc] = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 1), (4, 6), (6, 4)]
    c: Dict[Arc, float] = {a: (100.0 if a in [(1, 2), (2, 3), (3, 4)] else 1.0) for a in A}
    return dict(N=list(range(1, 7)), A=A, K=[1], c=c, f={a: 1.0 for a in A}, u={a: 100.0 for a in A},
                d={1: 10.0}, O={1: 1}, D={1: 4})


def test_the_net_flow_routes_the_commodity_and_the_paper_model_loops_instead() -> None:
    """The paper's model only fixes the gross flow out of O and into D: the two loops satisfy it
    (4 arcs x 10 units x 1 + 4 openings = 44) without any flow 1 -> 4. The net flow forces the path."""
    args = loops()
    assert solve(create_model(**args)) == pytest.approx(3 * 100 * 10 + 3)
    assert solve(create_model(**args)) == pytest.approx(brute_force(**args))
    paper: gp.Model = create_model_paper(**args)
    assert solve(paper) == pytest.approx(44)
    assert paper._x[1, 2, 1].X == pytest.approx(0) and paper._x[5, 1, 1].X == pytest.approx(10)


def test_the_flow_rows_at_the_origin_and_destination() -> None:
    args = loops()
    for build, into_origin, out_of_destination in [(create_model, -1.0, -1.0), (create_model_paper, 0.0, 0.0)]:
        m: gp.Model = build(**args)
        origin, destination = m.getConstrByName("flow[1,1]"), m.getConstrByName("flow[4,1]")
        assert m.getCoeff(origin, m._x[1, 5, 1]) == 1.0 and m.getCoeff(origin, m._x[5, 1, 1]) == into_origin
        assert m.getCoeff(destination, m._x[3, 4, 1]) == 1.0
        assert m.getCoeff(destination, m._x[4, 6, 1]) == out_of_destination
        assert origin.RHS == destination.RHS == 10.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_paper_model_is_a_relaxation(seed: int) -> None:
    inst: Instance = tiny(seed)
    assert solve(create_model_paper(**inst["args"])) <= solve(create_model(**inst["args"])) + 1e-6


def test_no_flow_into_the_origin_is_safe_but_not_valid() -> None:
    """With the net flow a commodity may still pass again through its origin (a costly cycle):
    forbidding it removes feasible points but never the optimum (unit costs are positive)."""
    args = loops()
    no_return: gp.Model = create_model(**args)
    no_return.addConstr(gp.quicksum(no_return._x[i, j, 1] for (i, j) in args["A"] if j == 1) == 0)
    assert solve(no_return) == pytest.approx(solve(create_model(**args)))
    detour: gp.Model = create_model(**args)  # route the path plus one loop 1 -> 5 -> 1
    detour.addConstr(detour._x[5, 1, 1] >= 1)
    assert solve(detour) > solve(create_model(**args))  # a feasible point the cut removes
    for seed in [0, 1, 2]:
        inst: Instance = tiny(seed)
        m: gp.Model = create_model(**inst["args"])
        m.addConstrs(gp.quicksum(m._x[i, j, k] for (i, j) in m._A if j == m._O[k]) == 0 for k in m._K)
        assert solve(m) == pytest.approx(brute_force(**inst["args"]))
