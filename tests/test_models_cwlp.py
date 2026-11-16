import functools
import itertools
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import pytest
from gurobipy import GRB

from evolution.evaluator import CutEvaluator
from loaders.base import Instance
from loaders.cwlp import LoaderCWLP
from loaders.sets import BASE, BASE_LP
from models.cwlp import BASELINE_CODE, create_model
from models.transforms import relaxed, with_cut
from problems import PROBLEMS
from solvers.solver import SolveResult, Solver
from utils import parameters

LINKING: str = "x, y = m._x, m._y\nm.addConstrs(x[i, j] <= y[j] for (i, j) in x.keys())"


def generate(seed: int, customers: int, warehouses: int) -> Instance:
    return LoaderCWLP(data_dir="/tmp").generate(1, seed=seed, min_customers=customers, max_customers=customers,
                                                min_warehouses=warehouses, max_warehouses=warehouses)[0]


def tiny(k: int) -> Instance:
    """The k-th feasible instance of 5 customers x 3 warehouses: with unsplittable demands, some
    tiny instances have no solution (the demands do not fit in the capacities)."""
    feasible: List[Instance] = []
    seed: int = 0
    while len(feasible) <= k:
        inst: Instance = generate(seed, 5, 3)
        if brute_force(**inst["args"]) < float("inf"):
            feasible.append(inst)
        seed += 1
    return feasible[k]


def brute_force(I: List[int], J: List[int], d: Dict[int, int], u: Dict[int, int], f: Dict[int, int],
                c: Dict[Tuple[int, int], float]) -> float:
    """Every assignment of the customers (each to one warehouse) that fits the capacities."""
    best: float = float("inf")
    for choice in itertools.product(J, repeat=len(I)):
        load: Dict[int, int] = {j: 0 for j in J}
        for i, j in zip(I, choice):
            load[j] += d[i]
        if all(load[j] <= u[j] for j in J):
            best = min(best, sum(f[j] for j in set(choice)) + sum(c[i, j] for i, j in zip(I, choice)))
    return best


def solve(model: gp.Model) -> float:
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    return model.ObjVal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimum_matches_brute_force(seed: int) -> None:
    inst: Instance = tiny(seed)
    assert solve(create_model(**inst["args"])) == pytest.approx(brute_force(**inst["args"]))


def test_model_size_and_attributes() -> None:
    inst: Instance = tiny(0)
    m: gp.Model = create_model(**inst["args"])
    assert m.NumVars == 5 * 3 + 3 and m.NumConstrs == 5 + 3  # x, y; assignment + capacity
    assert m.ModelSense == GRB.MINIMIZE
    assert set(m._x) == set(inst["args"]["c"]) and set(m._y) == {1, 2, 3} and m._d == inst["args"]["d"]


@pytest.mark.parametrize("code", [BASELINE_CODE, LINKING])
def test_seed_and_linking_cuts_keep_the_optimum(code: str) -> None:
    for seed in range(3):
        inst: Instance = tiny(seed)
        assert solve(with_cut(create_model, code, **inst["args"])) == pytest.approx(brute_force(**inst["args"]))


@pytest.fixture(scope="module")
def validation() -> List[Instance]:
    """The real CWLP validation set (sizes of problems.py, seed of parameters.py), as prepared."""
    problem = PROBLEMS["cwlp"]
    solver: Solver = Solver(log_dir=None, isolated=False, keep_solution=True)
    solver.add_formulation(BASE, create_model)
    solver.add_formulation(BASE_LP, functools.partial(relaxed, build=create_model))
    instances: List[Instance] = LoaderCWLP(data_dir="/tmp").generate(
        parameters.N_VALIDATION, parameters.VALIDATION_SEED, **problem.validation_sizes)
    for inst in instances:
        for name in (BASE, BASE_LP):
            result: SolveResult = solver.solve(inst, name)
            assert result.status == "OPTIMAL"
            inst[name] = result
    return instances


def test_seed_cut_passes_the_checks_on_the_real_validation_set(validation: List[Instance]) -> None:
    """Otherwise every CWLP run would stop at generation 0 ("the baseline cut is rejected")."""
    gap_instance: Instance = {**validation[0], BASE: SolveResult(
        "e", BASE, 42, "WORK_LIMIT", 1.0, 10.0, 1.0, 1, validation[0][BASE].obj_val, 0.0, 0.01)}
    evaluator: CutEvaluator = CutEvaluator(create_model, validation, [gap_instance],
                                           Solver(log_dir=None, isolated=False))
    evaluator.check(BASELINE_CODE)  # valid and useful: no CutRejected
    evaluator.check(LINKING)


def test_seed_cut_is_weaker_than_the_linking_cut() -> None:
    """The seed leaves the textbook strong cut to rediscover (runs/analysis/cwlp_sizes_20261102.txt:
    on 100x20, 1.5% of the root gap closed vs 57%). Not true on every small instance: on 30x8 the
    integer count can beat the linking."""
    def lp(code: Optional[str], inst: Instance) -> float:
        return solve((with_cut(create_model, code, **inst["args"]) if code else create_model(**inst["args"])).relax())

    inst: Instance = generate(7, 100, 20)
    base, seed, linking = lp(None, inst), lp(BASELINE_CODE, inst), lp(LINKING, inst)
    assert base <= seed < linking
