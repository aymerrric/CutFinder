import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from evolution.review import BASE_NAME, NO_SOLUTION_GAP, CutReview, format_review, review_cuts, score
from loaders.base import Instance
from loaders.sets import BASE, build_set
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from solvers.solver import Solver

TWO_CYCLE: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"
WORK_LIMIT: float = 0.05  # tiny: 40-city instances stop with a gap, fast


def make_solver(seed: int) -> Solver:
    return Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT, seed=seed)


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory: pytest.TempPathFactory) -> List[Instance]:
    folder: Path = tmp_path_factory.mktemp("data")
    os.mkdir(os.path.join(folder, "tsp"))
    solver: Solver = make_solver(42)
    solver.add_formulation(BASE, create_model)
    return build_set(LoaderTSP(data_dir=str(folder)), "eval.pkl", 2, 2, solver, require_optimal=False,
                     accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=40)


def test_review_under_several_seeds(evaluation: List[Instance]) -> None:
    reviews: List[CutReview] = review_cuts(
        create_model, {"seed": TWO_CYCLE, "best": TWO_CYCLE + "\nm.addConstr(m._u[2] >= 2)"},
        evaluation, [0, 1, 2], make_solver, run_fitness={"best": 12.5},
    )
    assert [r.name for r in reviews] == [BASE_NAME, "seed", "best"]
    base, seed, best = reviews
    assert base.fitnesses == [10.0, 10.0, 10.0]  # compared with itself at every seed
    assert base.std == 0.0 and best.run_fitness == 12.5 and seed.run_fitness is None
    for r in (seed, best):
        assert [s.seed for s in r.scores] == [0, 1, 2] and all(math.isfinite(f) for f in r.fitnesses)
        assert r.std == pytest.approx(__import__("statistics").stdev(r.fitnesses))
        assert all(len(s.gaps) == len(evaluation) for s in r.scores)
    json.dumps([r.to_dict() for r in reviews])  # saved as JSON at the end of a run
    table: str = format_review(reviews)
    assert "under 3 Gurobi seeds (0, 1, 2)" in table and "beats the seed cut on" in table


def gap(value: object, status: str = "WORK_LIMIT") -> SimpleNamespace:
    return SimpleNamespace(mip_gap=value, status=status)


def test_score_formula_and_edge_cases() -> None:
    insts: List[Instance] = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    s = score(0, insts, [gap(0.1), gap(0.2), gap(0.0)], [gap(0.05), gap(None, "KILLED"), gap(0.0)])
    assert s.skipped == ["c"]  # solved by the base at this seed: no ratio
    assert s.gaps == [0.05, NO_SOLUTION_GAP, 0.0] and s.statuses[1] == "KILLED"
    assert s.fitness == pytest.approx(10 * math.exp(-((0.05 - 0.1) / 0.1 + (1.0 - 0.2) / 0.2) / 2))
