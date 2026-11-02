import functools
import json
import os
from pathlib import Path
from typing import List

import pytest

from evolution.individual import SEED, Individual
from evolution.population import Population
from evolution.prompts import Prompts
from llm.fake import FakeLLM
from loaders.sets import BASE, BASE_LP, build_set
from loaders.tsp import LoaderTSP
from models.transforms import relaxed
from models.tsp import create_model
from prompt_bench import (ACCEPTED, CODE_ERROR, CROSSOVER, FORMAT, INIT, INVALID, KINDS, MUTATION,
                          TOO_SLOW, USELESS, Result, classify, make_tasks, normalize, parse_args,
                          mcnemar, merge, pick_parents, run_bench, wilson)
from solvers.solver import Solver
from utils import parameters

TWO_CYCLE: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"
BETTER: str = TWO_CYCLE + "\nm.addConstr(m._u[2] >= 2)"
WORK_LIMIT: float = 0.05


def reply(code: str) -> str:
    return f"IDEA: an idea\nCODE:\n```python\n{code}\n```"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    folder: Path = tmp_path_factory.mktemp("data")
    os.mkdir(os.path.join(folder, "tsp"))
    loader: LoaderTSP = LoaderTSP(data_dir=str(folder))
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, keep_solution=True)
    solver.add_formulation(BASE, create_model)
    solver.add_formulation(BASE_LP, functools.partial(relaxed, build=create_model))
    build_set(loader, parameters.VALIDATION_FILE, 2, 1, solver, require_optimal=True,
              formulations=[BASE, BASE_LP], min_cities=7, max_cities=7)
    evaluation_solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT)
    evaluation_solver.add_formulation(BASE, create_model)
    build_set(loader, parameters.EVALUATION_FILE, 2, 2, evaluation_solver, require_optimal=False,
              accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=40)
    return str(folder)


def population_file(tmp_path: Path) -> str:
    pop: Population = Population(2)
    pop.set_generation([Individual(idea="2-cycle", code=TWO_CYCLE, fitness=10.0, generator=SEED),
                        Individual(idea="better", code=BETTER, fitness=12.0, generator="init")])
    path: str = str(tmp_path / "population.json")
    pop.save(path)
    return path


def test_benchmark_end_to_end(data_dir: str, tmp_path: Path) -> None:
    args = parse_args([
        "--model_name", "fake", "--data_dir", data_dir, "--runs_dir", str(tmp_path), "--threads", "1",
        "--work_limit", str(WORK_LIMIT), "--variants", "default", "focused_ops", "--samples", "1",
        "--population_file", population_file(tmp_path), "--score_instances", "1", "--name", "t",
    ])
    # default: init + 3 mutations + 4 crossovers = 8 answers; focused_ops: 1 + 4 + 2 = 7
    replies: List[str] = ["no format at all", reply(TWO_CYCLE)] + [reply(BETTER)] * 13
    results: List[Result] = run_bench(args, llm=FakeLLM(replies))
    assert [r.variant for r in results] == ["default"] * 8 + ["focused_ops"] * 7
    assert results[0].kind == INIT and results[0].outcome == FORMAT
    assert results[1].kind == MUTATION and results[1].outcome == ACCEPTED and results[1].copy_of_parent
    assert all(r.outcome == ACCEPTED and r.fitness is not None for r in results[2:])
    folder: str = os.path.join(tmp_path, "prompt_bench", "t")
    assert len(open(os.path.join(folder, "results.jsonl")).read().splitlines()) == 15
    assert json.load(open(os.path.join(folder, "config.json")))["variants"] == ["default", "focused_ops"]
    report: str = open(os.path.join(folder, "report.md")).read()
    # BETTER is also a parent: answering it to a request that has it as parent is a copy
    assert "## Verdict" in report and "**Most new cuts accepted:** `focused_ops`, 43%" in report
    assert "| 1 | `focused_ops` | 3/7 (43%) |" in report and "| 7 (4) |" in report  # 7 accepted, 4 copies
    assert "| 2 | `default` | 1/8 (12%) |" in report and "| 7 (6) |" in report
    assert "| 2 / 0, p=0.500 |" in report  # paired with default: the format error and the copy
    assert "## Quality of the accepted cuts" in report and "## Why answers were refused" in report

    # the same benchmark split in two parts (other rounds), then merged into one report
    args2 = parse_args(["--model_name", "fake", "--data_dir", data_dir, "--runs_dir", str(tmp_path),
                        "--threads", "1", "--work_limit", str(WORK_LIMIT), "--variants", "default",
                        "--samples", "1", "--sample_offset", "1", "--tasks", "init",
                        "--population_file", population_file(tmp_path), "--name", "t2"])
    part2: List[Result] = run_bench(args2, llm=FakeLLM([reply(BETTER)]))
    assert [r.sample for r in part2] == [1]
    path: str = merge([folder, os.path.join(tmp_path, "prompt_bench", "t2")], os.path.join(tmp_path, "prompt_bench", "all"))
    merged: str = open(path).read()
    assert "16 answers from `fake`: 2 prompt variants × 2 rounds" in merged
    assert "| `default` | 2/9 (22%) |" in merged.replace("| 2 | ", "| ")  # + 1 new cut from part 2


def test_tasks_are_the_same_requests_in_every_variant() -> None:
    parents: List[Individual] = [Individual(idea=str(k), code=f"c{k}", fitness=k) for k in range(3)]
    default: Prompts = Prompts("def create_model(n, c): ...")
    tasks = make_tasks(default, KINDS, 2, parents)
    assert len(tasks) == 2 * (1 + len(default.mutations) + len(default.crossovers))
    other = make_tasks(Prompts("def create_model(n, c): ...", variant="focused_ops"), KINDS, 2, parents)
    seeds = {(t.kind, t.slot, t.sample): (t.llm_seed, [p.id for p in t.parents]) for t in tasks}
    for t in other:  # same slot and sample -> same LLM seed and same parents
        if (t.kind, t.slot, t.sample) in seeds:
            assert seeds[(t.kind, t.slot, t.sample)] == (t.llm_seed, [p.id for p in t.parents])
    assert {t.kind for t in tasks if len(t.parents) == 2} == {CROSSOVER}


def test_pick_parents_seed_first_then_best_distinct() -> None:
    seed: Individual = Individual(idea="s", code="a", fitness=10.0, generator=SEED)
    pop: Population = Population(3)
    pop.set_generation([seed, Individual(idea="b", code="b  # comment", fitness=14.0),
                        Individual(idea="b2", code="b", fitness=13.0)])
    assert [p.idea for p in pick_parents(pop, seed, 3)] == ["s", "b"]  # "b" twice: kept once
    assert pick_parents(None, seed, 3) == [seed]


def test_classify_normalize_wilson() -> None:
    assert classify("invalid cut: ...") == INVALID and classify("useless cut: ...") == USELESS
    assert classify("too slow: ...") == TOO_SLOW and classify("KILLED on instance x") == TOO_SLOW
    assert classify("ERROR on instance x: KeyError") == CODE_ERROR
    assert normalize("x = 1  # a\n\n y=2") == normalize("x=1\ny = 2")
    low, high = wilson(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8 and wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_and_round_offset() -> None:
    assert mcnemar(0, 0) == 1.0 and mcnemar(3, 3) == 1.0
    assert mcnemar(10, 0) == pytest.approx(2 / 2 ** 10)  # 10 discordant requests all one way
    assert mcnemar(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 2 ** 10)
    parents: List[Individual] = [Individual(idea="a", code="a", fitness=1.0), Individual(idea="b", code="b", fitness=2.0)]
    tasks = make_tasks(Prompts("def create_model(n, c): ..."), [INIT], 2, parents, offset=5)
    assert [t.sample for t in tasks] == [5, 6] and [t.llm_seed for t in tasks] == [5000, 6000]
