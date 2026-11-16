import functools
import math
import os
from pathlib import Path
from typing import List, Tuple

import pytest

from evolution.evaluator import BASE_LP_EVAL, BASE_SEED, CutEvaluator, FitnessMode
from evolution.genetic import CutRejected
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP, build_set
from loaders.tsp import LoaderTSP
from models.transforms import relaxed, with_cut
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
    with pytest.raises(CutRejected, match="LP relaxation value .* above the cost") as rejected:
        evaluator(code)  # caught by the LP check on the evaluation instances, before any MIP solve
    assert rejected.value.invalid


def test_validity_instances_are_checked_and_quoted(sets: Tuple[List[Instance], List[Instance]]) -> None:
    tight: Instance = {**sets[0][1], "name": "tight_one", "note": "a tight test instance"}
    evaluator: CutEvaluator = CutEvaluator(create_model, sets[0][:1], sets[1], make_solver(work_limit=WORK_LIMIT),
                                           validity=[tight])
    def arcs(inst: Instance) -> set:
        return {k for k, v in inst[BASE].solution.items() if k.startswith("x") and v > 0.5}
    only: str = sorted(arcs(tight) - arcs(sets[0][0]))[0]  # in the tight instance's optimal tour only
    i, j = (int(t) for t in only[2:-1].split(","))
    with pytest.raises(CutRejected, match=r"tight_one \(a tight test instance\)") as rejected:
        evaluator(f"m.addConstr(m._x[{i}, {j}] == 0, name='ban')\n" + TWO_CYCLE)
    assert rejected.value.invalid
    message: str = str(rejected.value)
    assert "these constraints of your cut are violated: ban (by 1)" in message  # the faulty line
    assert "The data of tight_one" in message and "  n = 7" in message  # small: its data is shown
    assert math.isfinite(evaluator(TWO_CYCLE))  # a valid cut passes the validity instances too
    with pytest.raises(ValueError, match="no base solution"):
        CutEvaluator(create_model, sets[0], sets[1], make_solver(), validity=[{"name": "v", "args": {}}])


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


def lp_value(code: str, inst: Instance) -> float:
    """LP relaxation value of the base model + `code`, solved directly (no evaluator)."""
    model = with_cut(create_model, code, **inst["args"]) if code else create_model(**inst["args"])
    lp = model.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    return lp.ObjVal


def make_lp_evaluator(sets: Tuple[List[Instance], List[Instance]], *solvers: Solver) -> CutEvaluator:
    # no tiny work limit here: the LP relaxations must reach OPTIMAL
    solver: Solver = make_solver()
    return CutEvaluator(create_model, sets[0], sets[1], solver, list(solvers) or None, FitnessMode.LP)


def test_lp_fitness_is_the_share_of_the_root_gap_closed(sets: Tuple[List[Instance], List[Instance]]) -> None:
    evaluator: CutEvaluator = make_lp_evaluator(sets)
    fitness: float = evaluator.evaluate(TWO_CYCLE)
    closed: List[float] = []
    for inst in sets[1]:
        base, cut, known = lp_value("", inst), lp_value(TWO_CYCLE, inst), inst[BASE].obj_val
        assert evaluator.base_lp[inst["name"]] == pytest.approx(base)
        closed.append((cut - base) / (known - base))
    assert fitness == pytest.approx(100 * sum(closed) / len(closed))
    assert 0 < fitness < 100  # the 2-cycle cut tightens the MTZ relaxation, but not to the optimum
    assert {"cut1_lp"} <= set(evaluator.solver.formulations) and "cut1_eval" not in evaluator.solver.formulations


def test_lp_mode_uses_one_solver_and_no_extra_seed(sets: Tuple[List[Instance], List[Instance]]) -> None:
    other: Solver = make_solver(seed=7)
    evaluator: CutEvaluator = make_lp_evaluator(sets, other)
    assert evaluator.scoring_solvers == [evaluator.solver]  # the seeds do not matter for an LP
    assert BASE_SEED not in other.formulations and BASE_LP_EVAL in evaluator.solver.formulations


def test_lp_value_above_a_known_solution_is_invalid(sets: Tuple[List[Instance], List[Instance]]) -> None:
    inst: Instance = sets[1][0]
    c12: float = inst["args"]["c"][1, 2]
    too_high: float = 1.01 * inst[BASE].obj_val
    code: str = TWO_CYCLE + (
        f"\nif abs(c[1, 2] - {c12!r}) < 1e-12:\n    m.addConstr(m.getObjective() >= {too_high!r})"
    )
    with pytest.raises(CutRejected, match="LP relaxation value"):
        make_lp_evaluator(sets).evaluate(code)


def test_gap_mode_is_the_default_and_call_is_evaluate(sets: Tuple[List[Instance], List[Instance]]) -> None:
    first, second = make_evaluator(sets), make_evaluator(sets)
    assert first.mode is FitnessMode.GAP
    assert first(TWO_CYCLE) == second.evaluate(TWO_CYCLE)  # deterministic: same fitness


def test_unnamed_cut_rows_are_numbered_and_big_data_is_not_shown(sets: Tuple[List[Instance], List[Instance]]) -> None:
    from evolution.evaluator import data_size, describe_data
    inst: Instance = sets[0][0]
    evaluator: CutEvaluator = CutEvaluator(create_model, [], [], make_solver(), validity=[inst])
    i, j = used_arc(inst)
    with pytest.raises(CutRejected, match=r"cut_row_2 \(by 1\)"):  # its 2nd constraint, left unnamed
        evaluator._check_validity("t", functools.partial(
            with_cut, base=create_model, code=f"m.addConstr(m._x.sum() >= 0)\nm.addConstr(m._x[{i}, {j}] == 0)"))
    assert data_size(inst["args"]) == 1 + 7 * 6  # n, and the 42 costs
    assert describe_data(inst, max_numbers=10) == ""


@pytest.fixture(scope="module")
def optimality(tmp_path_factory: pytest.TempPathFactory) -> List[Instance]:
    """Instances the base model solves to optimality under WORK_LIMIT (prepare_instances.py --optimality)."""
    from prepare_instances import proved_with_work
    folder: Path = tmp_path_factory.mktemp("optimality")
    os.mkdir(os.path.join(folder, "tsp"))
    return build_set(LoaderTSP(data_dir=str(folder)), "optimality.pkl", 2, 4000, make_solver(work_limit=WORK_LIMIT),
                     require_optimal=False, accept=proved_with_work((0.005, WORK_LIMIT)), min_cities=10, max_cities=12)


def make_work_evaluator(sets: Tuple[List[Instance], List[Instance]], optimality: List[Instance]) -> CutEvaluator:
    return CutEvaluator(create_model, sets[0], sets[1], make_solver(work_limit=WORK_LIMIT),
                        mode=FitnessMode.GAP_WORK, optimality=optimality)


def test_gap_work_pools_the_work_to_prove_optimality(
    sets: Tuple[List[Instance], List[Instance]], optimality: List[Instance],
) -> None:
    from evolution.review import work_ratios
    assert all(inst[BASE].status == "OPTIMAL" and inst[BASE].work >= 0.005 for inst in optimality)
    evaluator: CutEvaluator = make_work_evaluator(sets, optimality)
    fitness: float = evaluator.evaluate(TWO_CYCLE)
    solver: Solver = evaluator.solver
    cut: List[SolveResult] = [solver.results["cut1_eval"][inst["name"]][0] for inst in sets[1]]
    proofs: List[SolveResult] = [solver.results["cut1_opt"][inst["name"]][0] for inst in optimality]
    work, skipped = work_ratios(optimality, evaluator.base_optimality[solver.seed], proofs, WORK_LIMIT)
    assert len(work) == len(optimality) and skipped == []  # the base proves optimality on each at this seed
    expected: float = score(solver.seed, sets[1], evaluator.base_results[solver.seed], cut, work).fitness
    assert fitness == pytest.approx(expected)
    gap_only: float = score(solver.seed, sets[1], evaluator.base_results[solver.seed], cut).fitness
    assert fitness != pytest.approx(gap_only)  # the work term changes the fitness
    # the optimum of each optimality instance is a known solution: the LP check covers them too
    assert set(solver.results["cut1_lpcheck"]) == {inst["name"] for inst in sets[1] + optimality}


def test_gap_work_needs_instances_solved_by_the_base(
    sets: Tuple[List[Instance], List[Instance]], optimality: List[Instance],
) -> None:
    with pytest.raises(ValueError, match="optimality instances"):
        make_work_evaluator(sets, [])
    with pytest.raises(ValueError, match="not solved to optimality"):
        make_work_evaluator(sets, sets[1])  # the evaluation instances stop with a gap
    assert make_evaluator(sets).optimality == []  # the gap mode ignores them


def test_a_cut_above_the_optimum_of_an_optimality_instance_is_invalid(
    sets: Tuple[List[Instance], List[Instance]], optimality: List[Instance],
) -> None:
    inst: Instance = optimality[0]
    c12: float = inst["args"]["c"][1, 2]
    too_high: float = 1.01 * inst[BASE].obj_val  # above the proven optimum of this instance only
    code: str = TWO_CYCLE + f"\nif abs(c[1, 2] - {c12!r}) < 1e-12:\n    m.addConstr(m.getObjective() >= {too_high!r})"
    with pytest.raises(CutRejected, match="invalid cut"):
        make_work_evaluator(sets, optimality).evaluate(code)
    make_evaluator(sets).evaluate(code)  # the gap mode never solves that instance: accepted there
