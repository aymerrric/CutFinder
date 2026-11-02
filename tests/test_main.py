import argparse
import functools
import json
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
from llm.base import LLMClient, Message
from main import check_same_machine, make_run_name, run
from review_run import review_run
from models.transforms import relaxed
from models.tsp import create_model
from solvers.solver import Solver
from utils import parameters
from utils.parser import arg_parser
from utils.status import FINISHED, STOPPED, Status, read_status

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
        "--data_dir", data_dir, "--runs_dir", str(runs_dir), "--log_dir", str(runs_dir / "logs"), "--work_limit", str(WORK_LIMIT),
        "--threads", "1", "--population_size", "2", "--num_generations", "1", "--seed", "0",
        "--review_seeds", "0", "--fitness_seeds", "42", *extra,
    ])
    args.run_name = "test_run"
    return args


def test_a_small_run_saves_every_generation(data_dir: str, tmp_path: Path) -> None:
    pop: Population = run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    assert pop.generation == 1 and len(pop.current) == 2
    assert [ind.generator for ind in pop.history[0]] == [SEED, INIT]
    saved: Population = Population.load(os.path.join(tmp_path, "test_run", "population.json"))
    assert saved.generation == 1 and saved.best == pop.best


def test_resume_continues_without_a_new_first_generation(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    path: str = os.path.join(tmp_path, "test_run", "population.json")
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


class InterruptedLLM(LLMClient):
    """Behaves like a Ctrl-C (or a SIGTERM, which main turns into one) during the first request."""

    def _send(self, messages: List[Message]) -> str:
        raise KeyboardInterrupt


def test_status_file_follows_the_run(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    status: Status = read_status(os.path.join(tmp_path, "test_run", "status.json"))
    assert status["state"] == FINISHED and status["generation"] == 1 and status["child"] is None
    assert status["best"] > 0 and status["size"] == 2 and status["pid"] == os.getpid()
    assert status["population_file"] == os.path.join(tmp_path, "test_run", "population.json")
    assert os.listdir(os.path.join(tmp_path, "logs", "test_run"))  # the run's own Gurobi logs
    proposals: str = os.path.join(tmp_path, "test_run", "llm", "gen_000")  # every LLM proposal
    assert os.listdir(proposals) == ["child_02_attempt_1.json"]


def test_interrupted_run_is_marked_stopped(data_dir: str, tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        run(make_args(data_dir, tmp_path), llm=InterruptedLLM())
    status: Status = read_status(os.path.join(tmp_path, "test_run", "status.json"))
    assert status["state"] == STOPPED and status["building"] == 0 and status["child"] == 2


def test_run_name_has_the_host_or_the_resumed_file() -> None:
    assert make_run_name(arg_parser(["--model_name", "qwen3-coder:30b"]), "saumon").startswith(
        "tsp_qwen3-coder-30b_saumon_"
    )
    resumed: argparse.Namespace = arg_parser(["--population_file", "../runs/tsp_x_saumon_1.json"])
    assert make_run_name(resumed, "thon") == "tsp_x_saumon_1"  # old layout still understood


def test_end_of_run_review_and_review_of_a_saved_run(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path, "--review_seeds", "2"),
        llm=FakeLLM([TWO_CYCLE_REPLY.replace("i < j)", "i < j)\nm.addConstr(m._u[2] >= 2)")] * 5))
    path: str = os.path.join(tmp_path, "test_run", "review.json")
    review = json.load(open(path))
    pop: Population = Population.load(os.path.join(tmp_path, "test_run", "population.json"))
    seed_code: str = pop.history[0][0].code  # generation 0 (history: the generations before the current one)
    names: List[str] = ["base", "seed"] + (["best"] if pop.best.code != seed_code else [])  # same code: once
    assert review["seeds"] == [0, 1] and [m["name"] for m in review["models"]] == names
    assert all(len(m["scores"]) == 2 for m in review["models"])
    status: Status = read_status(os.path.join(tmp_path, "test_run", "status.json"))
    assert status["state"] == FINISHED and status["phase"] is None

    os.remove(path)  # the standalone script does the same review on the saved population
    args: argparse.Namespace = make_args(data_dir, tmp_path, "--review_seeds", "2", "--population_file",
                                         os.path.join(tmp_path, "test_run", "population.json"))
    assert [r.name for r in review_run(args)] == names and os.path.exists(path)


def test_everything_in_the_run_folder_with_a_summary(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    folder: str = os.path.join(tmp_path, "test_run")
    assert sorted(os.listdir(folder)) == ["llm", "population.json", "status.json", "summary.md"]
    summary: str = open(os.path.join(folder, "summary.md")).read()
    assert summary.startswith("# Run test_run") and "## Best cut" in summary
    assert "## Fitness per generation" in summary and "| 1 |" in summary  # generations 0 and 1
    assert "## LLM proposals" in summary and "| init |" in summary
    assert "population_size = 2" in summary  # the settings


def test_resume_from_the_run_folder_keeps_the_name() -> None:
    args: argparse.Namespace = arg_parser(["--population_file", "../runs/tsp_x_saumon_1/population.json"])
    assert make_run_name(args, "thon") == "tsp_x_saumon_1"


def test_review_of_a_run_with_only_generation_0(data_dir: str, tmp_path: Path) -> None:
    run(make_args(data_dir, tmp_path, "--num_generations", "0", "--review_seeds", "1"),
        llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    assert os.path.exists(os.path.join(tmp_path, "test_run", "review.json"))


def test_fitness_seeds_overlapping_the_review_seeds_are_reported(
    data_dir: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run(make_args(data_dir, tmp_path, "--num_generations", "0", "--fitness_seeds", "42", "1",
                  "--review_seeds", "2"), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    assert "fitness seeds [1] are also review seeds" in caplog.text
