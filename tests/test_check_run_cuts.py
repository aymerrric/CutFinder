import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from check_run_cuts import (
    VALIDITY_RESULT, check_run, describe, is_valid, known_solutions, main, public_violations, ranked_cuts,
    removed_optima,
)
from evolution.individual import ELITE, INIT, SEED, Individual
from evolution.population import Population
from loaders.base import Instance
from loaders.sets import BASE, build_set
from loaders.tsp import LoaderTSP
from models.tsp import BASELINE_CODE, create_model
from problems import PROBLEMS
from solvers.solver import Solver
from utils import parameters
from utils.run_files import POPULATION, STATUS

VALID: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"  # no 2-cycles
INVALID: str = "m.addConstr(m.getObjective() >= 1e9)"  # removes every tour, the optimum included


@pytest.fixture(scope="module")
def validity_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Two small TSP instances solved to optimality, saved like prepare_validity.py does."""
    folder: Path = tmp_path_factory.mktemp("validity")
    loader: LoaderTSP = LoaderTSP(data_dir=str(folder))
    (folder / "tsp").mkdir()
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, keep_solution=True)
    solver.add_formulation(BASE, create_model)
    build_set(loader, parameters.VALIDITY_FILE, 2, 5, solver, require_optimal=True, min_cities=6, max_cities=7)
    return str(folder)


def validity(validity_dir: str) -> List[Instance]:
    return LoaderTSP(data_dir=validity_dir).load(parameters.VALIDITY_FILE)


def solver() -> Solver:
    return Solver(log_dir=None, threads=1, isolated=False)


def make_run(folder: Path, best_is_invalid: bool = True) -> str:
    """Generation 0: the seed cut and a cut; generation 1: a better cut and an elite copy."""
    folder.mkdir()
    seed: Individual = Individual(idea="seed", code=BASELINE_CODE, fitness=10.0, generator=SEED)
    top: Individual = Individual(idea="top", code=INVALID if best_is_invalid else VALID, fitness=20.0, generator=INIT)
    second: Individual = Individual(idea="second", code=VALID + "\n# a comment", fitness=15.0, generator="Lifted")
    pop: Population = Population(2)
    pop.set_generation([seed, top])
    pop.set_generation([second, top.copy(ELITE)])
    pop.save(str(folder / POPULATION))
    (folder / STATUS).write_text(json.dumps({"arguments": {"problem_name": "tsp"}}))
    return str(folder)


def test_removed_optima(validity_dir: str) -> None:
    instances: List[Instance] = validity(validity_dir)
    assert removed_optima(PROBLEMS["tsp"], VALID, instances, solver(), "a") == []
    removed: List[str] = removed_optima(PROBLEMS["tsp"], INVALID, instances, solver(), "b")
    assert len(removed) == 2 and all("INFEASIBLE" in r for r in removed)


def test_ranked_cuts_fittest_first_each_code_once(tmp_path: Path) -> None:
    pop: Population = Population.load(str(Path(make_run(tmp_path / "run")) / POPULATION))
    ranked = ranked_cuts(pop)
    # the elite copy repeats "top"; "second" differs from VALID only by a comment, but VALID is not in this run
    assert [(g, ind.idea) for g, ind in ranked] == [(0, "top"), (1, "second"), (0, "seed")]


def test_an_invalid_best_cut_is_flagged_with_the_best_valid_one(validity_dir: str, tmp_path: Path) -> None:
    run_dir: str = make_run(tmp_path / "run")
    result: Dict[str, Any] = check_run(run_dir, validity(validity_dir), solver(), "validity.pkl")
    assert not result["best"]["valid"] and result["best"]["fitness"] == 20.0 and len(result["best"]["removed"]) == 2
    assert result["best_valid"]["idea"] == "second" and result["best_valid"]["generation"] == 1
    assert result["cuts_checked"] == 2 and result["n_instances"] == 2
    assert json.loads((tmp_path / "run" / VALIDITY_RESULT).read_text()) == result  # read by the report
    line: str = describe("run", result)
    assert "INVALID, removes the optimum of 2/2" in line and "best valid cut: fitness 15.000 (generation 1, Lifted)" in line


def test_a_valid_best_cut_needs_one_check(validity_dir: str, tmp_path: Path) -> None:
    result: Dict[str, Any] = check_run(make_run(tmp_path / "run", best_is_invalid=False), validity(validity_dir),
                                       solver(), "validity.pkl")
    assert result["best"]["valid"] and result["best_valid"]["idea"] == "top" and result["cuts_checked"] == 1
    assert "valid on 2 instances" in describe("run", result)


def test_main_checks_every_run_of_a_folder(validity_dir: str, tmp_path: Path) -> None:
    runs: Path = tmp_path / "runs"
    runs.mkdir()
    make_run(runs / "tsp_a")
    make_run(runs / "tsp_b", best_is_invalid=False)
    (runs / "not_a_run").mkdir()
    lines: List[str] = main(["--runs_dir", str(runs), "--validity_dir", validity_dir, "--threads", "1"])
    assert [line.split(":")[0] for line in lines] == ["tsp_a", "tsp_b"]
    assert main(["--runs_dir", str(runs), "--validity_dir", validity_dir, "--problem_name", "cwlp"]) == []


def test_known_solutions_are_the_best_trusted_benchmark_results(tmp_path: Path) -> None:
    for label, cpu, value in [("base", "cpu1", 120.0), ("seed", "cpu2", 110.0), ("paper_sage", "cpu1", 130.0),
                              ("some_run", "cpu1", 50.0)]:  # a run's cut may be invalid: not trusted
        folder: Path = tmp_path / "tsp" / "tl2000" / label / cpu
        folder.mkdir(parents=True)
        (folder / "a.json").write_text(json.dumps({"obj_val": value}))
    (tmp_path / "tsp" / "tl2000" / "base" / "cpu1" / "b.json").write_text(json.dumps({"obj_val": None}))
    assert known_solutions("tsp", str(tmp_path)) == {"a": 110.0}


def test_a_bound_above_a_known_solution_is_invalid(validity_dir: str) -> None:
    instances: List[Instance] = validity(validity_dir)
    known: Dict[str, float] = {inst["name"]: inst[BASE].obj_val for inst in instances}
    target: Instance = instances[0]
    above: str = (f"if abs(c[1, 2] - {target['args']['c'][1, 2]!r}) < 1e-12:\n"
                  f"    m.addConstr(m.getObjective() >= {1.01 * known[target['name']]!r})")
    result: Dict[str, Any] = public_violations(PROBLEMS["tsp"], above, instances, known, solver(), "p")
    assert result["checked"] == 2 and len(result["invalid"]) == 1 and target["name"] in result["invalid"][0]
    assert public_violations(PROBLEMS["tsp"], VALID, instances, known, solver(), "q")["invalid"] == []
    assert public_violations(PROBLEMS["tsp"], VALID, instances, {}, solver(), "r")["checked"] == 0  # nothing known


def test_check_run_records_the_public_layer(validity_dir: str, tmp_path: Path) -> None:
    instances: List[Instance] = validity(validity_dir)
    known: Dict[str, float] = {inst["name"]: inst[BASE].obj_val for inst in instances}
    run_dir: str = make_run(tmp_path / "run", best_is_invalid=False)
    result: Dict[str, Any] = check_run(run_dir, instances, solver(), "validity.pkl",
                                       public=(instances, known, solver(), 5.0))
    assert result["public"] == {"seconds": 5.0, "checked": 2, "invalid": [], "no_bound": []} and is_valid(result)
    assert "public bounds (5 s): valid on 2" in describe("run", result)
    result["public"]["invalid"] = ["x (bound 2 > known 1, +100.00%)"]
    assert not is_valid(result) and "public bounds (5 s): INVALID on 1/2: x" in describe("run", result)
    assert "public" not in check_run(run_dir, instances, solver(), "validity.pkl")  # off by default
