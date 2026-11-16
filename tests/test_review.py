import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from evolution.review import BASE_NAME, NO_SOLUTION_GAP, CutReview, format_review, paired_difference, review_cuts, score
from loaders.base import Instance
from loaders.sets import BASE, build_set
from loaders.tsp import LoaderTSP
from models.transforms import with_cut
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
    assert "± = sample std over these 3 seeds" in table and "(std over the seeds of the per-seed difference)" in table


def test_review_with_optimality_instances_measures_the_work_to_prove_optimality(
    evaluation: List[Instance], tmp_path: Path,
) -> None:
    os.mkdir(tmp_path / "tsp")
    optimality: List[Instance] = LoaderTSP(data_dir=str(tmp_path)).generate(2, seed=4000, min_cities=10, max_cities=11)
    reviews: List[CutReview] = review_cuts(create_model, {"seed": TWO_CYCLE}, evaluation, [0, 1], make_solver,
                                           optimality=optimality)
    base, seed = reviews
    assert base.work_changes == [0.0, 0.0]  # the base against itself
    assert len(seed.work_changes) == 2 and all(w is not None and w > -1 for w in seed.work_changes)
    entry = seed.to_dict()
    assert entry["work_change_mean"] == pytest.approx(sum(seed.work_changes) / 2)
    assert entry["work_change_std"] == pytest.approx(__import__("statistics").stdev(seed.work_changes))
    assert "work to prove optimality" in format_review(reviews) and "seed " in format_review(reviews)
    without: List[CutReview] = review_cuts(create_model, {"seed": TWO_CYCLE}, evaluation, [0], make_solver)
    assert without[1].work_changes == [] and without[1].to_dict()["work_change_mean"] is None
    assert "work to prove optimality" not in format_review(without)


def work(status: str, value: object) -> SimpleNamespace:
    return SimpleNamespace(status=status, work=value)


def test_work_ratios_to_prove_optimality() -> None:
    from evolution.review import work_ratios
    insts: List[Instance] = [{"name": n} for n in "abcd"]
    base = [work("OPTIMAL", 2.0), work("OPTIMAL", 4.0), work("WORK_LIMIT", 10.0), work("OPTIMAL", 5.0)]
    cut = [work("OPTIMAL", 1.0), work("WORK_LIMIT", 10.02), work("OPTIMAL", 1.0), work("OPTIMAL", 5.0)]
    ratios, skipped = work_ratios(insts, base, cut, 10.0)
    # a: twice as fast; b: not proved = the work limit (10 - 4) / 4; c: the base did not prove it; d: same
    assert ratios == [pytest.approx(-0.5), pytest.approx(1.5), 0.0] and skipped == ["c"]
    pooled = score(0, [{"name": "e"}], [gap(0.2)], [gap(0.1)], ratios)  # gap ratio -0.5, pooled with the work
    assert pooled.fitness == pytest.approx(10 * math.exp(-(-0.5 - 0.5 + 1.5 + 0.0) / 4))


def test_paired_difference_of_two_cuts_under_the_same_seeds() -> None:
    mean, std = paired_difference([12.0, 13.0, 17.0], [9.0, 11.0, 14.0])  # differences 3, 2, 3
    assert mean == pytest.approx(8 / 3) and std == pytest.approx(__import__("statistics").stdev([3, 2, 3]))
    assert paired_difference([12.0], [9.0]) == (3.0, 0.0)  # one seed: no spread


def gap(value: object, status: str = "WORK_LIMIT") -> SimpleNamespace:
    return SimpleNamespace(mip_gap=value, status=status)


def test_score_formula_and_edge_cases() -> None:
    insts: List[Instance] = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    s = score(0, insts, [gap(0.1), gap(0.2), gap(0.0)], [gap(0.05), gap(None, "KILLED"), gap(0.0)])
    assert s.skipped == ["c"]  # solved by the base at this seed: no ratio
    assert s.gaps == [0.05, NO_SOLUTION_GAP, 0.0] and s.statuses[1] == "KILLED"
    assert s.fitness == pytest.approx(10 * math.exp(-((0.05 - 0.1) / 0.1 + (1.0 - 0.2) / 0.2) / 2))


def direct_lp(code: str, inst: Instance) -> float:
    model = with_cut(create_model, code, **inst["args"]) if code else create_model(**inst["args"])
    lp = model.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    return lp.ObjVal


def test_review_also_measures_the_lp_relaxation(evaluation: List[Instance]) -> None:
    solvers: List[Solver] = []

    def recording(seed: int) -> Solver:
        solvers.append(make_solver(seed))
        return solvers[-1]

    reviews: List[CutReview] = review_cuts(create_model, {"seed": TWO_CYCLE}, evaluation, [0, 1], recording)
    base, seed = reviews
    for i, inst in enumerate(evaluation):
        assert base.lp_values[i] == pytest.approx(direct_lp("", inst))
        assert seed.lp_values[i] == pytest.approx(direct_lp(TWO_CYCLE, inst))
        # UB = the cheapest tour of the base model under the review seeds
        known: float = min(s.results["review_base"][inst["name"]][0].obj_val for s in solvers[:2])
        assert seed.lp_closed[i] == pytest.approx((seed.lp_values[i] - base.lp_values[i]) / (known - base.lp_values[i]))
    assert base.lp_fitness == 0.0 and 0 < seed.lp_fitness < 100
    assert seed.to_dict()["lp_fitness"] == seed.lp_fitness
    assert "LP gap closed" in format_review(reviews)
