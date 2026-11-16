import argparse
import os
from typing import List

import pytest

import benchmark_cut
from benchmark_cut import (
    BASE_LABEL, CODE_FILE, cut_sources, load_results, pair_by_cpu, parse_args, report, run, select, slug,
)
from evolution.individual import Individual
from evolution.population import Population
from loaders.base import Instance
from loaders.tsp import LoaderTSP
from problems import PROBLEMS
from solvers.solver import Solver
from utils import parameters


@pytest.fixture
def bench(tmp_path, monkeypatch) -> argparse.Namespace:
    """A 2-instance TSP benchmark set in tmp_path, and arguments that use it (2 s, 1 thread)."""
    loader = LoaderTSP(str(tmp_path / "data"))
    os.makedirs(loader.folder)
    loader.save(LoaderTSP().generate(2, seed=0, min_cities=8, max_cities=8), parameters.BENCHMARK_FILE)
    monkeypatch.setattr(parameters, "LOG_DIR", str(tmp_path / "logs"))
    return parse_args(["--problem_name", "tsp", "--cut", "seed", "--time_limit", "2", "--threads", "1",
                       "--data_dir", str(tmp_path / "data"), "--out_dir", str(tmp_path / "out"), "--any_cpu"])


def test_run_saves_base_and_cut_then_resumes_without_solving(bench, monkeypatch, capsys) -> None:
    run(bench)
    root: str = os.path.join(bench.out_dir, "tsp", "tl2")
    cpu: str = slug(benchmark_cut.machine_info()[1])
    for label in (BASE_LABEL, "seed"):
        assert sorted(os.listdir(os.path.join(root, label, cpu))) == ["rand_s0_0_8.json", "rand_s0_1_8.json"]
    with open(os.path.join(root, "seed", CODE_FILE)) as f:
        assert f.read() == PROBLEMS["tsp"].baseline_code
    assert "[2/2]" in capsys.readouterr().out
    monkeypatch.setattr(Solver, "solve", lambda *a, **k: pytest.fail("already solved"))
    run(bench)  # every result is on disk: nothing is solved again
    text: str = report(bench)
    assert "| Baseline (seed cut) | 2 |" in text and "| 2 s | Average |" in text  # 2 s run: one checkpoint, the limit
    assert os.path.exists(os.path.join(root, "report_any_cpu.md"))  # the fixture solves with --any_cpu


def test_another_cpu_is_refused_and_left_out_of_the_report(bench, monkeypatch) -> None:
    bench.any_cpu = False
    monkeypatch.setattr(benchmark_cut, "machine_info", lambda: ("host", "Other CPU"))
    with pytest.raises(RuntimeError, match="BENCHMARK_CPU"):
        run(bench)
    bench.any_cpu = True
    run(bench)  # solved on "Other CPU"
    bench.any_cpu = False
    assert "No result yet." in report(bench)  # not the benchmark's CPU: not in the table
    monkeypatch.setattr(parameters, "BENCHMARK_CPU", "Other CPU")
    text: str = report(bench)
    assert "| Baseline (seed cut) | 2 |" in text and "Solved on: Other CPU" in text


def test_base_only_and_shard(bench) -> None:
    bench.base_only, bench.shard = True, "2/2"
    run(bench)
    results = load_results(os.path.join(bench.out_dir, "tsp", "tl2", BASE_LABEL))
    assert [sorted(r) for r in results.values()] == [["rand_s0_1_8"]]
    assert not os.path.exists(os.path.join(bench.out_dir, "tsp", "tl2", "seed"))


def population_file(folder, name: str, code: str) -> str:
    run_dir = folder / name
    run_dir.mkdir()
    population = Population(size=1)
    population.set_generation([Individual(idea="i", code=code, fitness=12.0)])
    population.save(str(run_dir / "population.json"))
    return str(run_dir / "population.json")


def test_cut_sources(tmp_path) -> None:
    problem = PROBLEMS["tsp"]
    assert cut_sources(parse_args(["--cut", "seed"]), problem) == [("seed", problem.baseline_code)]
    file = tmp_path / "my_cut.py"
    file.write_text("x = m._x")
    assert cut_sources(parse_args(["--cut_file", str(file)]), problem) == [("my_cut", "x = m._x")]
    assert cut_sources(parse_args(["--cut_file", str(file), "--label", "other"]), problem) == [("other", "x = m._x")]
    run_file: str = population_file(tmp_path, "tsp_model_host_20261107_000000", "pass")
    best = parse_args(["--cut", "best", "--population_file", run_file])
    assert cut_sources(best, problem) == [("tsp_model_host_20261107_000000", "pass")]
    with pytest.raises(FileNotFoundError):
        cut_sources(parse_args(["--cut", "best"]), problem)
    with pytest.raises(ValueError, match="name a cut"):
        cut_sources(parse_args([]), problem)


def test_cut_sources_several_cuts_in_order(tmp_path) -> None:
    problem = PROBLEMS["tsp"]
    file = tmp_path / "paper_sage.py"
    file.write_text("pass  # paper")
    run_file: str = population_file(tmp_path, "tsp_model_host_20261107_000000", "pass  # best")
    args = parse_args(["--cut", "seed", "--population_file", run_file, "--cut_file", str(file)])
    assert [label for label, _ in cut_sources(args, problem)] == [
        "seed", "tsp_model_host_20261107_000000", "paper_sage"]
    args.label = "one"
    with pytest.raises(ValueError, match="single cut"):
        cut_sources(args, problem)
    with pytest.raises(ValueError, match="same label"):
        cut_sources(parse_args(["--cut_file", str(file), str(file)]), problem)


def test_run_with_several_cuts_solves_the_base_once(bench, monkeypatch) -> None:
    second = os.path.join(os.path.dirname(bench.out_dir), "other_cut.py")
    with open(second, "w") as f:
        f.write(PROBLEMS["tsp"].baseline_code)
    bench.cut_file = [second]
    solved: List[str] = []
    original = Solver.solve
    monkeypatch.setattr(Solver, "solve", lambda self, inst, form: solved.append(form) or original(self, inst, form))
    run(bench)
    assert solved == ["base", "seed", "other_cut"] * 2  # per instance: base, then every cut
    assert "| other_cut | 2 |" in report(bench) and "| Baseline (seed cut) | 2 |" in report(bench)


def test_base_label_is_reserved(bench) -> None:
    bench.label = "base"
    with pytest.raises(ValueError, match="reference"):
        run(bench)


def test_select() -> None:
    instances: List[Instance] = [{"name": f"i{k}", "args": {}} for k in range(5)]
    assert [i["name"] for i in select(instances, None, "1/2")] == ["i0", "i2", "i4"]
    assert [i["name"] for i in select(instances, ["i3", "i1"], None)] == ["i1", "i3"]
    with pytest.raises(ValueError, match="unknown"):
        select(instances, ["nope"], None)


def test_pair_by_cpu_never_mixes_cpus() -> None:
    a, b = object(), object()
    base = {"cpuA": {"i1": a}, "cpuB": {"i2": b}}
    cut = {"cpuA": {"i1": b, "i2": a}}  # i2 has a base only on cpuB: not compared
    base_pairs, cut_pairs = pair_by_cpu(base, cut)  # type: ignore[arg-type]
    assert base_pairs == {"i1": a} and cut_pairs == {"i1": b}


def test_slug() -> None:
    assert slug("Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz") == "Intel-R-Xeon-R-W-2245-CPU-3.90GHz"
    assert slug("") == "unknown"


def test_display_names_and_order() -> None:
    from benchmark_cut import display_name, row_order
    assert display_name("seed", "jssp") == "Baseline (seed cut)"
    assert display_name("jssp_qwen3-coder-30b_harpie_20261104_101456", "jssp") == "Our best cut (harpie_20261104)"
    assert display_name("paper_sage", "jssp") == "Paper cut (on our CPU)"
    assert display_name("other_cut", "jssp") == "other_cut"
    labels = ["paper_sage", "other_cut", "jssp_m_h_20261104_1", "seed"]
    assert sorted(labels, key=lambda l: row_order(l, "jssp")) == ["seed", "jssp_m_h_20261104_1", "paper_sage", "other_cut"]


def test_lp_relaxations_give_the_lp_column(bench) -> None:
    run(bench)
    bench.lp = True
    benchmark_cut.solve_lps(bench)
    root: str = os.path.join(bench.out_dir, "tsp", "tl2")
    assert sorted(os.listdir(os.path.join(root, "lp"))) == ["base", "seed"]
    text: str = report(bench)
    assert "LP gap closed" in text
    line: str = next(l for l in text.splitlines() if l.startswith("| Baseline (seed cut) |"))
    closed: float = float(line.split("|")[-2].split("(")[0])
    assert 0.0 <= closed <= 100.0 and "(n=2)" in line  # the 2-cycle cut: valid, never above UB


def test_report_has_a_table_in_work_units(bench) -> None:
    run(bench)  # new solves: the work is recorded along each solve
    text: str = report(bench)
    work_part: str = text.split("## In Gurobi work units", 1)[1]
    assert "| 5 wu |" in work_part and "| Baseline (seed cut) | 2 |" in work_part
    assert benchmark_cut.STD_NOTE in text and "spread between instances" in benchmark_cut.STD_NOTE


def test_redo_without_work_solves_old_results_again_missing_first(bench, monkeypatch, capsys) -> None:
    run(bench)
    root: str = os.path.join(bench.out_dir, "tsp", "tl2", "base", slug(benchmark_cut.machine_info()[1]))
    old: str = os.path.join(root, "rand_s0_1_8.json")
    import json
    with open(old) as f:
        data = json.load(f)
    data["trajectory"] = [p[:3] for p in data["trajectory"]] or [[0.0, 1e100, 0.0]]  # as saved before 9 Nov
    with open(old, "w") as f:
        json.dump(data, f)
    os.remove(os.path.join(bench.out_dir, "tsp", "tl2", "seed", slug(benchmark_cut.machine_info()[1]), "rand_s0_0_8.json"))
    assert benchmark_cut.without_work(old) and not benchmark_cut.without_work(old + ".missing")
    solved: List[str] = []
    original = Solver.solve
    monkeypatch.setattr(Solver, "solve", lambda self, inst, form: solved.append(f"{inst['name']}:{form}") or original(self, inst, form))
    bench.redo_without_work = True
    capsys.readouterr()
    run(bench)
    # rand_s0_0_8 has a missing result: first; rand_s0_1_8 has a solve without work: base and seed again
    assert solved == ["rand_s0_0_8:seed", "rand_s0_1_8:base", "rand_s0_1_8:seed"]
    assert "solved again" in capsys.readouterr().out and not benchmark_cut.without_work(old)


def test_a_cut_with_a_bound_above_a_known_solution_is_flagged(bench) -> None:
    import json
    run(bench)
    cpu: str = slug(benchmark_cut.machine_info()[1])
    path: str = os.path.join(bench.out_dir, "tsp", "tl2", "seed", cpu, "rand_s0_0_8.json")
    with open(path) as f:
        data = json.load(f)
    data["obj_bound"] = data["obj_val"] * 1.5  # as if the cut had removed the optimum
    with open(path, "w") as f:
        json.dump(data, f)
    text: str = report(bench)
    assert "| Baseline (seed cut) (INVALID on 1/2) | 2 |" in text
    assert "**Invalid cuts**" in text and "on rand_s0_0_8" in text
    assert benchmark_cut.violations(os.path.join(bench.out_dir, "tsp", "tl2"), "seed", {"rand_s0_0_8": 1e9}) == ([], 1)
