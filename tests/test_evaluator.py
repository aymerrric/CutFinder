import functools
import math
import os
from pathlib import Path
from typing import List, Tuple

import pytest

from evolution.evaluator import BASE_SEED, CutEvaluator
from evolution.genetic import CutRejected
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP, build_set
from loaders.tsp import LoaderTSP
from models.transforms import relaxed
from models.tsp import create_model
from evolution.review import score
from solvers.solver import SolveResult, Solver

TWO_CYCLE: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"
WORK_LIMIT: float = 0.05  # tiny, so that 40-city evaluation instances stop with a gap, fast


def make_solver(**kwargs: object) -> Solver:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, **kwargs)
    solver.add_formulation(BASE, create_model)
    solver.add_formulation(BASE_LP, functools.partial(relaxed, build=create_model))
    return solver


@pytest.fixture(scope="module")
def sets(tmp_path_factory: pytest.TempPathFactory) -> Tuple[List[Instance], List[Instance]]:
    folder: Path = tmp_path_factory.mktemp("data")
    os.mkdir(os.path.join(folder, "tsp"))
    loader: LoaderTSP = LoaderTSP(data_dir=str(folder))
    validation: List[Instance] = build_set(
        loader, "val.pkl", 2, 1, make_solver(keep_solution=True), require_optimal=True,
        formulations=[BASE, BASE_LP], min_cities=7, max_cities=7,
    )
    evaluation: List[Instance] = build_set(
        loader, "eval.pkl", 3, 2, make_solver(work_limit=WORK_LIMIT), require_optimal=False,
        accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=40,
    )
    return validation, evaluation


def make_evaluator(sets: Tuple[List[Instance], List[Instance]]) -> CutEvaluator:
    return CutEvaluator(create_model, sets[0], sets[1], make_solver(work_limit=WORK_LIMIT))


def used_arc(inst: Instance) -> Tuple[int, int]:
    """An arc of the optimal tour of a validation instance."""
    name: str = next(k for k, v in inst[BASE].solution.items() if k.startswith("x") and v > 0.5)
    i, j = name[2:-1].split(",")  # "x[3,5]" -> 3, 5
    return int(i), int(j)


def test_valid_useful_cut_gets_a_fitness(sets: Tuple[List[Instance], List[Instance]]) -> None:
    fitness: float = make_evaluator(sets)(TWO_CYCLE)
    assert math.isfinite(fitness) and fitness > 0


def test_cut_removing_an_optimal_tour_is_invalid(sets: Tuple[List[Instance], List[Instance]]) -> None:
    i, j = used_arc(sets[0][0])
    with pytest.raises(CutRejected, match="removes the optimal solution"):
        make_evaluator(sets)(f"m.addConstr(m._x[{i}, {j}] == 0)\n" + TWO_CYCLE)


def test_implied_constraint_is_useless(sets: Tuple[List[Instance], List[Instance]]) -> None:
    with pytest.raises(CutRejected, match="useless"):
        make_evaluator(sets)("m.addConstr(m._x.sum() <= n)")  # already implied by the degrees


def test_bound_above_a_known_solution_is_invalid(
    sets: Tuple[List[Instance], List[Instance]],
) -> None:
    inst: Instance = sets[1][0]
    c12: float = inst["args"]["c"][1, 2]
    too_high: float = 1.01 * inst[BASE].obj_val  # only tours worse than the known one are left
    code: str = TWO_CYCLE + (
        f"\nif abs(c[1, 2] - {c12!r}) < 1e-12:\n    m.addConstr(m.getObjective() >= {too_high!r})"
    )
    # more work than WORK_LIMIT: Gurobi must at least solve the root LP to have a real bound
    evaluator: CutEvaluator = CutEvaluator(create_model, sets[0], sets[1], make_solver(work_limit=1))
    with pytest.raises(CutRejected, match="lower bound"):
        evaluator(code)


def test_fitness_formula(sets: Tuple[List[Instance], List[Instance]]) -> None:
    evaluator: CutEvaluator = make_evaluator(sets)
    fitness: float = evaluator(TWO_CYCLE)
    gaps: List[float] = [
        r.mip_gap for r in evaluator.solver.results["cut1_eval"].values() for r in r[:1]
    ]
    bases: List[float] = [inst[BASE].mip_gap for inst in sets[1]]
    mean: float = sum((g - b) / b for g, b in zip(gaps, bases)) / len(bases)
    assert fitness == pytest.approx(10 * math.exp(-mean))


def test_broken_code_is_rejected_with_its_error(sets: Tuple[List[Instance], List[Instance]]) -> None:
    with pytest.raises(CutRejected, match="ERROR.*KeyError"):
        make_evaluator(sets)("m.addConstr(m._x[1, 1] == 0)")


def test_validation_without_solutions_is_refused(sets: Tuple[List[Instance], List[Instance]]) -> None:
    no_solution: List[Instance] = [{**sets[1][0]}]  # evaluation instances keep no solution
    with pytest.raises(ValueError):
        CutEvaluator(create_model, no_solution, sets[1], make_solver())


def test_solved_evaluation_instance_is_refused(sets: Tuple[List[Instance], List[Instance]]) -> None:
    with pytest.raises(ValueError, match="gap"):  # validation instances are solved: gap 0
        CutEvaluator(create_model, sets[0], sets[0], make_solver())


def test_each_cut_has_its_own_formulations(sets: Tuple[List[Instance], List[Instance]]) -> None:
    evaluator: CutEvaluator = make_evaluator(sets)
    evaluator(TWO_CYCLE)
    evaluator(TWO_CYCLE)
    assert {"cut1_valid", "cut1_useful", "cut1_eval", "cut2_eval"} <= set(evaluator.solver.formulations)


def test_cut_hitting_the_time_limit_is_rejected_as_too_slow(
    sets: Tuple[List[Instance], List[Instance]]
) -> None:
    """Scored at the time limit, the fitness would depend on the machine's speed."""
    solver: Solver = make_solver(work_limit=1e6, time_limit=0.001)
    with pytest.raises(CutRejected, match="too slow"):
        CutEvaluator(create_model, sets[0], sets[1], solver)(TWO_CYCLE)


def test_check_runs_only_the_cheap_checks(sets: Tuple[List[Instance], List[Instance]]) -> None:
    evaluator: CutEvaluator = make_evaluator(sets)
    assert evaluator.check(TWO_CYCLE) is None  # valid and useful
    i, j = used_arc(sets[0][0])
    with pytest.raises(CutRejected, match="invalid"):
        evaluator.check(f"m.addConstr(m._x[{i}, {j}] == 0)\nm.addConstr(m._x[{j}, {i}] == 0)")
    with pytest.raises(CutRejected, match="useless"):
        evaluator.check("m.addConstr(m._u[2] >= 2)")


def test_fitness_is_the_mean_over_the_scoring_seeds(sets: Tuple[List[Instance], List[Instance]]) -> None:
    base_seed: int = sets[1][0][BASE].seed  # the evaluation set was solved under this seed
    solvers: List[Solver] = [make_solver(work_limit=WORK_LIMIT, seed=s) for s in (base_seed, 7)]
    evaluator: CutEvaluator = CutEvaluator(create_model, sets[0], sets[1], solvers[0], solvers)
    # the stored base results are reused for their seed; seed 7 solves the base model once
    assert evaluator.base_results[base_seed] == [inst[BASE] for inst in sets[1]]
    assert BASE_SEED not in solvers[0].formulations and len(solvers[1].results[BASE_SEED]) == len(sets[1])
    fitness: float = evaluator(TWO_CYCLE)
    per_seed: List[float] = []
    for solver in solvers:
        cut: List[SolveResult] = [solver.results["cut1_eval"][inst["name"]][0] for inst in sets[1]]
        per_seed.append(score(solver.seed, sets[1], evaluator.base_results[solver.seed], cut).fitness)
    assert fitness == pytest.approx(sum(per_seed) / 2)
    assert per_seed[0] != pytest.approx(per_seed[1])  # another seed, another search path
