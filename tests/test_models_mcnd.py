import itertools
from typing import Dict, List, Tuple

import gurobipy as gp
import pytest
from gurobipy import GRB

from loaders.base import Instance
from loaders.mcnd import LoaderMCND
from models.mcnd import BASELINE_CODE, create_model
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
