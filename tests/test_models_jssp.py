import functools
import itertools
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import pytest
from gurobipy import GRB

from evolution.evaluator import CutEvaluator
from loaders.base import Instance
from loaders.jssp import LoaderJSSP
from loaders.sets import BASE, BASE_LP
from models.jssp import BASELINE_CODE, create_model
from models.transforms import with_cut
from models.transforms import relaxed
from problems import PROBLEMS
from solvers.solver import SolveResult, Solver
from utils import parameters


def generate(seed: int, jobs: int, machines: int) -> Instance:
    return LoaderJSSP(data_dir="/tmp").generate(1, seed=seed, min_jobs=jobs, max_jobs=jobs,
                                                min_machines=machines, max_machines=machines)[0]


def brute_force(n_jobs: int, n_machines: int, times: List[List[int]], machines: List[List[int]]) -> float:
    """Every order of the jobs on every machine: the semi-active schedule of each feasible one."""
    best: float = float("inf")
    for orders in itertools.product(*[list(itertools.permutations(range(n_jobs)))] * n_machines):
        position: Dict[Tuple[int, int], int] = {(r, j): orders[r].index(j) for r in range(n_machines) for j in range(n_jobs)}
        start: Dict[Tuple[int, int], int] = {}
        done: bool = True
        for _ in range(n_jobs * n_machines):  # schedule the operations whose predecessors are scheduled
            done = True
            for j in range(n_jobs):
                for k in range(n_machines):
                    if (j, k) in start:
                        continue
                    r: int = machines[j][k]
                    before: List[Tuple[int, int]] = [(j2, machines[j2].index(r)) for j2 in range(n_jobs)
                                                     if position[r, j2] == position[r, j] - 1]
                    preds: List[Tuple[int, int]] = ([(j, k - 1)] if k > 0 else []) + before
                    if all(o in start for o in preds):
                        start[j, k] = max([start[o] + times[o[0]][o[1]] for o in preds], default=0)
                    else:
                        done = False
            if done:
                break
        if done:  # else the orders contradict the job sequences (a cycle): infeasible
            best = min(best, max(start[j, n_machines - 1] + times[j][n_machines - 1] for j in range(n_jobs)))
    return best


def solve(model: gp.Model) -> float:
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    return model.ObjVal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimum_matches_brute_force(seed: int) -> None:
    inst: Instance = generate(seed, 3, 2)
    assert solve(create_model(**inst["args"])) == pytest.approx(brute_force(**inst["args"]))


def test_model_size_and_attributes() -> None:
    inst: Instance = generate(0, 4, 3)
    m: gp.Model = create_model(**inst["args"])
    pairs: int = 3 * (4 * 3 // 2)  # per machine, every pair of the 4 jobs
    assert m.NumVars == 4 * 3 + 1 + pairs and m.NumBinVars == pairs
    assert m.NumConstrs == 4 * 2 + 2 * pairs + 4  # precedences, 2 disjunctions per pair, makespan
    assert m.ModelSense == GRB.MINIMIZE and m._bigM == sum(map(sum, inst["args"]["times"]))


def test_seed_cut_keeps_the_optimum() -> None:
    for seed in range(3):
        inst: Instance = generate(seed, 3, 3)
        assert solve(with_cut(create_model, BASELINE_CODE, **inst["args"])) == pytest.approx(brute_force(**inst["args"]))


def test_seed_cut_passes_the_checks_on_the_real_validation_set() -> None:
    """Otherwise every JSSP run would stop at generation 0 ("the baseline cut is rejected")."""
    solver: Solver = Solver(log_dir=None, isolated=False, keep_solution=True)
    solver.add_formulation(BASE, create_model)
    solver.add_formulation(BASE_LP, functools.partial(relaxed, build=create_model))
    validation: List[Instance] = LoaderJSSP(data_dir="/tmp").generate(
        parameters.N_VALIDATION, parameters.VALIDATION_SEED, **PROBLEMS["jssp"].validation_sizes)
    for inst in validation:
        for name in (BASE, BASE_LP):
            result: SolveResult = solver.solve(inst, name)
            assert result.status == "OPTIMAL"
            inst[name] = result
    gap_instance: Instance = {**validation[0], BASE: SolveResult(
        "e", BASE, 42, "WORK_LIMIT", 1.0, 10.0, 1.0, 1, validation[0][BASE].obj_val, 0.0, 0.01)}
    CutEvaluator(create_model, validation, [gap_instance], Solver(log_dir=None, isolated=False)).check(BASELINE_CODE)


def test_seed_cut_tightens_the_weak_big_m_relaxation() -> None:
    """runs/analysis/jssp_sizes_20261102.txt: it closes 33-100% of the root gap."""
    def lp(code: Optional[str], inst: Instance) -> float:
        return solve((with_cut(create_model, code, **inst["args"]) if code else create_model(**inst["args"])).relax())

    inst: Instance = generate(7, 10, 5)
    assert lp(BASELINE_CODE, inst) > 1.5 * lp(None, inst)
