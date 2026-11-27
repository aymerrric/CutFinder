import json
import os
from pathlib import Path

import pytest

import select_cuts
from evolution.individual import INIT, SEED, Individual
from evolution.population import Population
from loaders.sets import BASE, build_set
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from problems import PROBLEMS, instance_set
from solvers.solver import Solver

TWO_CYCLE: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"
NOTHING: str = "pass"
WORK_LIMIT: float = 0.05  # tiny: 40-city instances stop with a gap, fast


def population(seed_fitness: float = 11.0) -> Population:
    pop: Population = Population(4)
    pop.set_generation([Individual("seed", TWO_CYCLE, seed_fitness, SEED),
                        Individual("a", NOTHING, 12.0, INIT),
                        Individual("a again", NOTHING + "\n", 12.5, INIT),  # same code: one cut
                        Individual("b", TWO_CYCLE + "\nm.update()", 15.0, INIT)])
    return pop


def test_top_cuts_are_distinct_best_first_with_the_seed_cut() -> None:
    cuts = select_cuts.top_cuts(population(), 2)
    assert [c.idea for c in cuts] == ["b", "a again", "seed"]  # the seed cut is added after the top 2
    assert [c.idea for c in select_cuts.top_cuts(population(seed_fitness=20.0), 2)] == ["seed", "b"]


def test_compare_picks_by_run_fitness_by_selection_and_the_oracle() -> None:
    res = {"run": "r", "cuts": [
        {"name": "top1", "run_fitness": 16.0, "selection": 11.0, "test": 10.0},
        {"name": "top2", "run_fitness": 15.0, "selection": 14.0, "test": 13.0},
        {"name": "top3", "run_fitness": 14.0, "selection": 12.0, "test": 14.0},
        {"name": "seed", "run_fitness": 12.0, "selection": 20.0, "test": 12.0}]}  # never picked
    table: str = select_cuts.compare([res])
    assert "| r | 10.00 (top1) | 13.00 (top2) | 14.00 (top3) | 12.00 |" in table
    assert "**10.00** | **13.00** | **14.00** | 12.00" in table


def test_select_cuts_scores_a_run_on_the_selection_and_test_sets(tmp_path: Path) -> None:
    data: Path = tmp_path / "data"
    os.makedirs(data / "tsp")
    loader = LoaderTSP(data_dir=str(data))
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT, seed=42)
    solver.add_formulation(BASE, create_model)
    for kind, seed in (("test", 1000),):  # the selection set: built by select_cuts.py itself
        build_set(loader, instance_set(PROBLEMS["tsp"], kind).filename, 2, seed, solver, require_optimal=False,
                  accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=40)
    run: Path = tmp_path / "runs" / "tsp_fake_host_20261120_000000"
    os.makedirs(run)
    population().save(str(run / "population.json"))
    with open(run / "status.json", "w") as f:
        json.dump({"arguments": {"problem_name": "tsp", "fitness_seeds": [42]}}, f)
    args = select_cuts.parse_args(["--population_file", str(run / "population.json"), "--data_dir", str(data),
                                   "--top", "2", "--review_seeds", "1", "--n_selection", "2", "--work_limit", str(WORK_LIMIT),
                                   "--threads", "1", "--log_dir", str(tmp_path / "logs")])
    table: str = select_cuts.main(args)
    with open(run / "selection.json") as f:
        saved = json.load(f)
    assert [c["name"] for c in saved["cuts"]] == ["top1", "top2", "seed"]
    assert all(c["selection"] > 0 and c["test"] > 0 for c in saved["cuts"])
    assert saved["selection_seeds"] == [42] and saved["test_seeds"] == [0]
    assert "tsp_fake_host_20261120_000000" in table
    assert len(LoaderTSP(data_dir=str(data)).load("selection.pkl")) == 2 and not os.path.exists(data / "tsp" / "selection.pkl.lock")
