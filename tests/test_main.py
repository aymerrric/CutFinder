import argparse
import functools
import json
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

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
from problems import PROBLEMS
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
    """Small validation, evaluation and test sets, built like prepare_instances.py (threads=1)."""
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
    build_set(loader, parameters.TEST_FILE, 2, 1000, evaluation_solver, require_optimal=False,
              accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=35, max_cities=45)
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
    # each part of the run packed its solve logs into one archive, the folder is gone
    assert sorted(os.listdir(tmp_path / "logs")) == ["test_run.tar.gz", "test_run_2.tar.gz"]


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
    with tarfile.open(os.path.join(tmp_path, "logs", "test_run.tar.gz")) as tar:  # the run's own Gurobi logs, packed
        assert any(name.endswith(".log") for name in tar.getnames())
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
    test: List[str] = [inst["name"] for inst in LoaderTSP(data_dir).load(parameters.TEST_FILE)]
    evaluation: List[str] = [inst["name"] for inst in LoaderTSP(data_dir).load(parameters.EVALUATION_FILE)]
    assert review["fitness_mode"] == "gap" and all("lp_fitness" in m for m in review["models"])
    assert review["models"][0]["lp_fitness"] == 0.0  # the base model closes nothing of its own gap
    assert review["instance_set"] == "test.pkl" and review["instances"] == test  # held out:
    assert not set(test) & set(evaluation)  # never the instances the GA was scored on
    assert all(len(s["gaps"]) == len(test) for m in review["models"] for s in m["scores"])
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


def test_a_run_with_the_lp_fitness(data_dir: str, tmp_path: Path) -> None:
    llm: FakeLLM = FakeLLM([TWO_CYCLE_REPLY] * 5)
    pop: Population = run(make_args(data_dir, tmp_path, "--fitness_mode", "lp", "--fitness_seeds", "42", "43"), llm=llm)
    assert pop.generation == 1 and 0 < pop.best.fitness <= 100  # % of the root gap closed
    assert llm.received and all("root gap" in messages[0]["content"] for messages in llm.received)  # told to the LLM


def test_a_run_without_the_test_set_fails_before_the_ga(data_dir: str, tmp_path: Path) -> None:
    import shutil
    folder: Path = tmp_path / "data"
    shutil.copytree(data_dir, folder)
    os.remove(folder / "tsp" / parameters.TEST_FILE)
    llm: FakeLLM = FakeLLM([TWO_CYCLE_REPLY] * 5)
    with pytest.raises(FileNotFoundError, match="test.pkl"):
        run(make_args(str(folder), tmp_path, "--review_seeds", "2"), llm=llm)
    assert llm.received == []  # no LLM call, no solve: it failed at startup
    run(make_args(str(folder), tmp_path), llm=llm)  # without a review, test.pkl is not needed


# small sets per problem: (validation sizes, evaluation and test sizes, a valid and useful cut, words of the prompt)
SMALL: Dict[str, Tuple[Dict[str, int], Dict[str, int], str, str]] = {
    "cwlp": ({"min_customers": 30, "max_customers": 30, "min_warehouses": 8, "max_warehouses": 8},
             {"min_customers": 100, "max_customers": 100, "min_warehouses": 20, "max_warehouses": 20},
             "x, y = m._x, m._y\nm.addConstrs(x[i, j] <= y[j] for (i, j) in x.keys())",
             "Capacitated Warehouse Location Problem"),
    "jssp": ({"min_jobs": 6, "max_jobs": 6, "min_machines": 4, "max_machines": 4},
             {"min_jobs": 10, "max_jobs": 10, "min_machines": 5, "max_machines": 5},
             "Cmax, p = m._Cmax, m._p\nfor r in range(n_machines):\n"
             "    ops = [(j, k) for (j, k) in p if machines[j][k] == r]\n"
             "    m.addConstr(Cmax >= min(sum(times[j][:k]) for (j, k) in ops) + sum(p[o] for o in ops))",
             "Job Shop Scheduling Problem"),
    "mcnd": ({"min_nodes": 8, "max_nodes": 8, "min_arcs": 25, "max_arcs": 25, "min_commodities": 10, "max_commodities": 10},
             {"min_nodes": 15, "max_nodes": 15, "min_arcs": 80, "max_arcs": 80, "min_commodities": 40, "max_commodities": 40},
             "x, y, u, d = m._x, m._y, m._u, m._d\n"
             "m.addConstrs(x[i, j, k] <= min(u[i, j], d[k]) * y[i, j] for (i, j, k) in x.keys())",
             "Multi-Commodity capacitated fixed-charge Network Design"),
    "imo6": ({"min_n": 4, "max_n": 5}, {"min_n": 10, "max_n": 12},
             "s, N, I = m._s, m._N, m._I\n"
             "m.addConstrs(gp.quicksum(s[a, b, i] for (a, b) in I) >= 1 for i in range(2, N + 1))",
             "grid tiling problem"),
    "gisp": ({"min_nodes": 30, "max_nodes": 30}, {"min_nodes": 75, "max_nodes": 75},
             "x, P = m._x, m._permanent\n"
             "near = {i: set() for i in m._nodes}\n"
             "for (i, j) in P:\n    near[i].add(j)\n    near[j].add(i)\n"
             "m.addConstrs(x[i] + x[j] + x[k] <= 1 for (i, j) in P for k in near[i] & near[j] if k > j)",
             "Generalized Independent Set Problem"),
    "corridor": ({"min_order": 6, "max_order": 7}, {"min_order": 9, "max_order": 10},
                 "x, near = m._x, {i: [] for i in m._nodes}\n"
                 "for (i, j) in m._edges:\n    near[i].append(j)\n    near[j].append(i)\n"
                 "m.addConstrs(x[j] <= gp.quicksum(x[i] for i in near[j]) for j in m._nodes)",
                 "wildlife corridor design problem"),
}


def small_sets(problem: str, folder: Path) -> str:
    """Small sets of `problem`, built like prepare_instances.py --problem_name <problem> (threads=1)."""
    spec = PROBLEMS[problem]
    validation_sizes, evaluation_sizes, _, _ = SMALL[problem]
    loader = spec.loader(str(folder))
    os.makedirs(loader.folder)
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, keep_solution=True)
    solver.add_formulation(BASE, spec.base)
    solver.add_formulation(BASE_LP, functools.partial(relaxed, build=spec.base))
    build_set(loader, parameters.VALIDATION_FILE, 2, 1, solver, require_optimal=True,
              formulations=[BASE, BASE_LP], **validation_sizes)
    evaluation_solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT)
    evaluation_solver.add_formulation(BASE, spec.base)
    for filename, seed in ((parameters.EVALUATION_FILE, 2), (parameters.TEST_FILE, 1000)):
        build_set(loader, filename, 2, seed, evaluation_solver, require_optimal=False,
                  accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, **evaluation_sizes)
    return str(folder)


@pytest.mark.parametrize("problem", sorted(SMALL))
def test_a_run_end_to_end_on_another_problem(problem: str, tmp_path: Path) -> None:
    data_dir: str = small_sets(problem, tmp_path / "data")
    _, _, cut, words = SMALL[problem]
    llm: FakeLLM = FakeLLM([f"IDEA: a cut\nCODE:\n```python\n{cut}\n```"] * 5)
    pop: Population = run(make_args(data_dir, tmp_path, "--problem_name", problem, "--review_seeds", "1"), llm=llm)
    assert pop.generation == 1 and pop.history[0][0].code == PROBLEMS[problem].baseline_code  # its seed cut
    assert all(words in m[0]["content"] for m in llm.received)  # its prompts
    assert any(r["outcome"] == "accepted" for r in map(json.loads, (
        open(os.path.join(dp, f)).read() for dp, _, fs in os.walk(os.path.join(tmp_path, "test_run", "llm")) for f in fs)))
    review = json.load(open(os.path.join(tmp_path, "test_run", "review.json")))
    test_names = [inst["name"] for inst in PROBLEMS[problem].loader(data_dir).load(parameters.TEST_FILE)]
    assert review["instances"] == test_names  # reviewed on its own test set (IMO6 names have no seed)


def test_validity_set_is_loaded_or_skipped_with_a_warning(tmp_path: Path, caplog) -> None:
    from main import load_validity
    from problems import PROBLEMS
    assert load_validity(PROBLEMS["tsp"], str(tmp_path)) == []
    assert "prepare_validity" in caplog.text
    loader = LoaderTSP(data_dir=str(tmp_path))
    os.makedirs(loader.folder)
    loader.save([{"name": "v", "args": {}}], parameters.VALIDITY_FILE)
    assert [i["name"] for i in load_validity(PROBLEMS["tsp"], str(tmp_path))] == ["v"]


def test_a_run_on_the_hard_instance_level(data_dir: str, tmp_path: Path) -> None:
    import shutil
    folder: Path = tmp_path / "data"
    shutil.copytree(data_dir, folder)
    llm: FakeLLM = FakeLLM([TWO_CYCLE_REPLY] * 5)
    with pytest.raises(FileNotFoundError, match="--instance_level hard"):  # its sets are not built yet
        run(make_args(str(folder), tmp_path, "--instance_level", "hard"), llm=llm)
    loader: LoaderTSP = LoaderTSP(data_dir=str(folder))
    evaluation_solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT)
    evaluation_solver.add_formulation(BASE, create_model)
    for filename, seed in (("evaluation_hard.pkl", 2000), ("test_hard.pkl", 3000)):  # problems.instance_set
        build_set(loader, filename, 2, seed, evaluation_solver, require_optimal=False,
                  accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=45)
    run(make_args(str(folder), tmp_path, "--instance_level", "hard", "--review_seeds", "1"), llm=llm)
    review = json.load(open(os.path.join(tmp_path, "test_run", "review.json")))
    assert review["instance_set"] == "test_hard.pkl" and review["instance_level"] == "hard"
    assert all(name.startswith("rand_s3000_") for name in review["instances"])  # not test.pkl's
    assert llm.received and all("60 to 200 cities" in m[0]["content"] for m in llm.received)  # the sizes it is told
    status: Status = read_status(os.path.join(tmp_path, "test_run", "status.json"))
    assert status["arguments"]["instance_level"] == "hard"  # kept by a resume (argv) and read by the report
    # the review of a saved run must use the run's own level
    args: argparse.Namespace = make_args(str(folder), tmp_path, "--review_seeds", "1", "--population_file",
                                         os.path.join(tmp_path, "test_run", "population.json"))
    with pytest.raises(ValueError, match="pass --instance_level hard"):
        review_run(args)


def test_a_default_run_reviewed_on_the_hard_test_set(data_dir: str, tmp_path: Path) -> None:
    import shutil
    folder: Path = tmp_path / "data"
    shutil.copytree(data_dir, folder)
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT)
    solver.add_formulation(BASE, create_model)
    build_set(LoaderTSP(data_dir=str(folder)), "test_hard.pkl", 2, 3000, solver, require_optimal=False,
              accept=lambda inst: (inst[BASE].mip_gap or 0) > 0.002, min_cities=40, max_cities=45)
    run(make_args(str(folder), tmp_path, "--review_seeds", "1"), llm=FakeLLM([TWO_CYCLE_REPLY] * 5))
    own: str = (tmp_path / "test_run" / "review.json").read_text()
    args: argparse.Namespace = make_args(str(folder), tmp_path, "--review_seeds", "1", "--review_level", "hard",
                                         "--population_file", os.path.join(tmp_path, "test_run", "population.json"))
    review_run(args)
    hard = json.load(open(tmp_path / "test_run" / "review_hard.json"))
    assert hard["instance_set"] == "test_hard.pkl" and hard["instance_level"] == "hard"
    assert all(name.startswith("rand_s3000_") for name in hard["instances"])
    assert (tmp_path / "test_run" / "review.json").read_text() == own  # the run's own review is kept


def test_a_run_with_the_gap_work_fitness(data_dir: str, tmp_path: Path) -> None:
    import shutil
    from prepare_instances import proved_with_work
    folder: Path = tmp_path / "data"
    shutil.copytree(data_dir, folder)
    llm: FakeLLM = FakeLLM([TWO_CYCLE_REPLY] * 5)
    with pytest.raises(FileNotFoundError, match="--optimality"):  # its sets are not built yet
        run(make_args(str(folder), tmp_path, "--fitness_mode", "gap_work"), llm=llm)
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, work_limit=WORK_LIMIT)
    solver.add_formulation(BASE, create_model)
    for filename, seed in ((parameters.OPTIMALITY_FILE, 4000), (parameters.OPTIMALITY_TEST_FILE, 5000)):
        build_set(LoaderTSP(data_dir=str(folder)), filename, 2, seed, solver, require_optimal=False,
                  accept=proved_with_work((0.002, WORK_LIMIT)), min_cities=10, max_cities=12)
    pop: Population = run(make_args(str(folder), tmp_path, "--fitness_mode", "gap_work", "--review_seeds", "1"), llm=llm)
    assert pop.generation == 1 and pop.best.fitness > 0
    assert llm.received and all("PROVE optimality" in m[0]["content"] for m in llm.received)  # told to the LLM
    review = json.load(open(os.path.join(tmp_path, "test_run", "review.json")))
    held_out: List[str] = [inst["name"] for inst in LoaderTSP(str(folder)).load(parameters.OPTIMALITY_TEST_FILE)]
    assert review["fitness_mode"] == "gap_work" and review["optimality_instances"] == held_out
    assert all(m["work_change_mean"] is not None for m in review["models"])
    args: argparse.Namespace = make_args(str(folder), tmp_path, "--review_seeds", "1", "--population_file",
                                         os.path.join(tmp_path, "test_run", "population.json"))
    with pytest.raises(ValueError, match="pass --fitness_mode gap_work"):  # reviewed like it ran
        review_run(args)
