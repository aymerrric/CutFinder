import argparse
import functools
import os
from pathlib import Path
from typing import List

import pytest

from evolution.individual import INIT, SEED
from evolution.population import Population
from llm.fake import FakeLLM
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP, build_set
from loaders.tsp import LoaderTSP
from main import check_same_machine, run
from models.transforms import relaxed
from models.tsp import create_model
from solvers.solver import Solver
from utils import parameters
from utils.parser import arg_parser

TWO_CYCLE_REPLY: str = (
    "IDEA: no 2-cycles\nCODE:\n```python\nx = m._x\n"
    "m.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)\n```"
)
WORK_LIMIT: float = 0.05  # tiny: 40-city evaluation instances stop with a gap, fast


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Small validation and evaluation sets, built like prepare_instances.py (threads=1)."""
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


def make_args(data_dir: str, runs_dir: Path, *extra: str) -> argparse.Namespace:
    args: argparse.Namespace = arg_parser([
        "--data_dir", data_dir, "--runs_dir", str(runs_dir), "--work_limit", str(WORK_LIMIT),
        "--threads", "1", "--population_size", "2", "--num_generations", "1", "--seed", "0", *extra,
    ])
    args.run_name = "test_run"
    return args


def test_a_small_run_saves_every_generation(data_dir: str, tmp_path: Path) -> None:
    pop: Population = run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    assert pop.generation == 1 and len(pop.current) == 2
    assert [ind.generator for ind in pop.history[0]] == [SEED, INIT]
    saved: Population = Population.load(os.path.join(tmp_path, "test_run.json"))
    assert saved.generation == 1 and saved.best == pop.best


def test_resume_continues_without_a_new_first_generation(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    path: str = os.path.join(tmp_path, "test_run.json")
    llm: FakeLLM = FakeLLM([TWO_CYCLE_REPLY] * 5)
    pop: Population = run(
        make_args(data_dir, tmp_path, "--population_file", path, "--num_generations", "2"), llm=llm
    )
    assert pop.generation == 2 and Population.load(path).generation == 2
    assert len(llm.received) == 2  # one generation step: 2 children (0 elite), no initializer


def test_missing_sets_are_reported(tmp_path: Path) -> None:
    os.mkdir(os.path.join(tmp_path, "tsp"))
    with pytest.raises(FileNotFoundError, match="prepare_instances"):
        run(make_args(str(tmp_path), tmp_path), llm=FakeLLM([]))


def test_model_name_is_required_without_an_llm(data_dir: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_name"):
        run(make_args(data_dir, tmp_path))


def test_sets_from_another_machine_type_are_refused(data_dir: str) -> None:
    evaluation: List[Instance] = LoaderTSP(data_dir=data_dir).load(parameters.EVALUATION_FILE)
    check_same_machine(evaluation, threads=1)  # built here, with 1 thread: fine
    with pytest.raises(RuntimeError, match="threads"):
        check_same_machine(evaluation, threads=8)
    evaluation[0][BASE].cpu = "another CPU"
    with pytest.raises(RuntimeError, match="another CPU"):
        check_same_machine(evaluation, threads=1)
