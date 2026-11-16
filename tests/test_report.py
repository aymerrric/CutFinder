import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from evolution.individual import SEED, Individual
from evolution.population import Population
from evolution.report import STALLED_AFTER, RunRow, best_per_host, group_runs, read_run, report_markdown, state_of
from utils.run_files import POPULATION, REVIEW, STATUS


def make_run(folder: Path, host: str, model: str = "qwen3-coder:30b", mode: str = "gap", problem: str = "tsp",
             run_fitness: float = 14.0, review_gap: Optional[float] = 12.0, review_lp: Optional[float] = 5.0,
             state: str = "finished", held_out: bool = True, level: Optional[str] = None) -> str:
    """A run folder with a status, a population (best cut = 'cut of <host>') and maybe a review."""
    folder.mkdir(parents=True)
    best: Individual = Individual(idea=f"idea of {host}", code=f"cut of {host}", fitness=run_fitness, generator=SEED)
    pop: Population = Population(1)
    pop.set_generation([best])
    pop.save(str(folder / POPULATION))
    (folder / STATUS).write_text(json.dumps({
        "state": state, "host": host, "model": model, "generation": 20, "num_generations": 20,
        "best": run_fitness, "updated": time.time(), "arguments": {"fitness_mode": mode, "prompt_variant": "rules", "problem_name": problem,
                                                                   **({"instance_level": level} if level else {})},
    }))
    if review_gap is not None:
        review: Dict[str, Any] = {"seeds": [0, 1], "models": [
            {"name": "base", "fitness_mean": 10.0, "fitness_std": 0.0, "lp_fitness": 0.0},
            {"name": "best", "fitness_mean": review_gap, "fitness_std": 1.5, "lp_fitness": review_lp},
        ]}
        if held_out:
            review.update(instance_set="test.pkl", instances=["a", "b"])
        (folder / REVIEW).write_text(json.dumps(review))
    return str(folder)


def test_read_run(tmp_path: Path) -> None:
    row: Optional[RunRow] = read_run(make_run(tmp_path / "r1", "saumon"))
    assert row is not None and row.host == "saumon" and row.model == "qwen3-coder:30b"
    assert (row.run_fitness, row.review_gap, row.review_gap_std, row.review_lp) == (14.0, 12.0, 1.5, 5.0)
    assert row.held_out and row.best_code == "cut of saumon" and row.prompt_variant == "rules"
    assert row.score == 12.0  # gap mode: ranked by the review gap fitness, never the run fitness
    lp: Optional[RunRow] = read_run(make_run(tmp_path / "r2", "thon", mode="lp"))
    assert lp is not None and lp.score == 5.0  # lp mode: ranked by the review's LP gap closed
    assert read_run(str(tmp_path)) is None  # no status.json: not a run


def test_old_run_without_fitness_mode_nor_review(tmp_path: Path) -> None:
    folder: str = make_run(tmp_path / "old", "allemagne", review_gap=None)
    status: Dict[str, Any] = json.loads((tmp_path / "old" / STATUS).read_text())
    status["arguments"] = {}
    (tmp_path / "old" / STATUS).write_text(json.dumps(status))
    row: Optional[RunRow] = read_run(folder)
    assert row is not None and row.fitness_mode == "gap" and row.prompt_variant == "default"
    assert row.score is None and not row.held_out and row.problem == "tsp"  # before other problems


def test_a_killed_run_is_shown_as_stalled() -> None:
    now: float = 1e9
    assert state_of({"state": "running", "updated": now - STALLED_AFTER - 1}, now) == "stalled"
    assert state_of({"state": "running", "updated": now - 10}, now) == "running"
    assert state_of({"state": "finished", "updated": 0.0}, now) == "finished"


def test_groups_by_llm_and_fitness_mode(tmp_path: Path) -> None:
    rows: List[RunRow] = [read_run(make_run(tmp_path / name, host, model, mode)) for name, host, model, mode in [
        ("a", "saumon", "qwen3-coder:30b", "gap"), ("b", "thon", "qwen3-coder:30b", "gap"),
        ("c", "sole", "qwen3-coder:30b", "lp"), ("d", "lieu", "devstral:24b", "gap")]]
    groups = group_runs(rows)
    assert list(groups) == [("tsp", "devstral:24b", "gap", "default"), ("tsp", "qwen3-coder:30b", "gap", "default"),
                            ("tsp", "qwen3-coder:30b", "lp", "default")]
    assert [r.host for r in groups[("tsp", "qwen3-coder:30b", "gap", "default")]] == ["saumon", "thon"]


def test_instance_levels_are_never_mixed(tmp_path: Path) -> None:
    rows: List[RunRow] = [read_run(make_run(tmp_path / "a", "saumon")),  # older run: no instance_level
                          read_run(make_run(tmp_path / "b", "thon", level="hard"))]
    assert rows[0].instance_level == "default" and rows[1].instance_level == "hard"
    assert list(group_runs(rows)) == [("tsp", "qwen3-coder:30b", "gap", "default"), ("tsp", "qwen3-coder:30b", "gap", "hard")]
    assert "## TSP, qwen3-coder:30b, fitness `gap`, instances `hard`" in report_markdown(rows)


def test_best_per_host_keeps_each_computers_best_reviewed_run(tmp_path: Path) -> None:
    rows: List[RunRow] = [
        read_run(make_run(tmp_path / "a", "saumon", review_gap=11.0)),
        read_run(make_run(tmp_path / "b", "saumon", review_gap=15.0)),
        read_run(make_run(tmp_path / "c", "thon", review_gap=13.0)),
        read_run(make_run(tmp_path / "d", "sole", review_gap=None)),  # not reviewed: left out
    ]
    assert [(r.host, r.review_gap) for r in best_per_host(rows)] == [("saumon", 15.0), ("thon", 13.0)]


def test_report_has_best_of_all_and_mean_of_each_computers_best(tmp_path: Path) -> None:
    rows: List[RunRow] = [
        read_run(make_run(tmp_path / "a", "saumon", run_fitness=19.0, review_gap=11.0, review_lp=2.0)),
        read_run(make_run(tmp_path / "b", "saumon", run_fitness=14.0, review_gap=15.0, review_lp=6.0)),
        read_run(make_run(tmp_path / "c", "thon", run_fitness=16.0, review_gap=13.0, review_lp=4.0)),
        read_run(make_run(tmp_path / "d", "sole", review_gap=None, state="running")),
        read_run(make_run(tmp_path / "e", "lieu", model="devstral:24b", review_gap=12.0, held_out=False)),
    ]
    text: str = report_markdown(rows)
    qwen: str = text.split("## TSP, qwen3-coder:30b, fitness `gap`")[1]
    devstral: str = text.split("## TSP, devstral:24b, fitness `gap`")[1].split("## ")[0]
    assert "4 runs on 3 computers, 3 finished, 2 computers with a review" in qwen
    # best of all by the review (15.0), not by the run fitness (19.0)
    assert ("**Best of all:** `b` (saumon): review gap fitness 15.000 ± 1.500 (std over the review seeds), "
            "LP gap closed 6.00%") in qwen
    assert "> idea of saumon" in qwen and "cut of saumon" in qwen
    # mean over the computers of their best run: saumon 15 (b), thon 13
    assert "review gap fitness: 14.000 ± 1.414 (n=2, std between the 2 computers)" in qwen
    assert "review LP gap closed (%): 5.000 ± 1.414 (n=2, std between the 2 computers)" in qwen
    assert "run fitness (gap scale, biased upwards): 15.000 ± 1.414 (n=2, std between the 2 computers)" in qwen
    assert "Where each ± comes from" in text
    assert qwen.index("| `b` |") < qwen.index("| `c` |") < qwen.index("| `a` |") < qwen.index("| `d` |")
    assert "review on the evaluation set, not held out" in devstral


def test_between_llms_table_gives_both_stds(tmp_path: Path) -> None:
    rows: List[RunRow] = [
        read_run(make_run(tmp_path / "a", "saumon", review_gap=15.0)),
        read_run(make_run(tmp_path / "b", "thon", review_gap=13.0)),
        read_run(make_run(tmp_path / "c", "lieu", model="muse", review_gap=24.0)),
        read_run(make_run(tmp_path / "d", "sole", model="old", review_gap=30.0, held_out=False)),  # not held out: left out
        read_run(make_run(tmp_path / "e", "bar", problem="jssp", review_gap=16.0)),  # one LLM only: no table
    ]
    text: str = report_markdown(rows).split("## Between LLMs: TSP, fitness `gap`, instances `default`")[1]
    assert "| muse | 1 | 24.000 (n=1: no std) | 24.000 (`c`) |" in text  # one run: no spread to show
    assert "| qwen3-coder:30b | 2 | 14.000 ± 1.414 (n=2) | 15.000 (`a`) |" in text  # std between its runs
    # mean of the LLM means (24, 14) and the std between them
    assert "| **between the 2 LLMs** | | 19.000 ± 7.071 (n=2, std between the 2 LLM means) | |" in text
    assert text.index("| muse |") < text.index("| qwen3-coder:30b |") and "| old |" not in text
    assert "Between LLMs: JSSP" not in report_markdown(rows)


def test_a_run_whose_best_cut_is_invalid_is_left_out(tmp_path: Path) -> None:
    from evolution.report import VALIDITY
    rows_dirs: List[str] = [make_run(tmp_path / "a", "saumon", review_gap=25.0), make_run(tmp_path / "b", "thon", review_gap=13.0)]
    (tmp_path / "a" / VALIDITY).write_text(json.dumps({"n_instances": 9, "best": {"valid": False, "removed": ["x (INFEASIBLE)", "y (INFEASIBLE)"]}}))
    (tmp_path / "b" / VALIDITY).write_text(json.dumps({"n_instances": 9, "best": {"valid": True, "removed": []}}))
    rows: List[RunRow] = [read_run(d) for d in rows_dirs]
    assert rows[0].best_valid is False and rows[0].score is None  # 25.0 is not a result
    assert rows[1].best_valid is True and rows[1].score == 13.0
    text: str = report_markdown(rows)
    assert "**Best of all:** `b` (thon)" in text and "1 runs left out: their best cut is invalid" in text
    assert "review gap fitness: 13.000 (n=1: no std)" in text
    assert "| INVALID (2/9) |" in text and "| valid |" in text
    assert read_run(make_run(tmp_path / "c", "sole")).validity_note == "-"  # not checked
    # the public-bound layer of check_run_cuts.py --public_seconds: invalid there = invalid
    (tmp_path / "b" / VALIDITY).write_text(json.dumps({"n_instances": 9, "best": {"valid": True, "removed": []},
                                                       "public": {"checked": 36, "invalid": ["capa1 (...)"]}}))
    row: RunRow = read_run(rows_dirs[1])
    assert row.best_valid is False and row.score is None and row.validity_note == "INVALID on public (1/36)"
    (tmp_path / "b" / VALIDITY).write_text(json.dumps({"n_instances": 9, "best": {"valid": True, "removed": []},
                                                       "public": {"checked": 36, "invalid": []}}))
    assert read_run(rows_dirs[1]).validity_note == "valid, public bounds too"


def test_report_without_runs() -> None:
    assert "No run found." in report_markdown([])


def test_problems_are_never_mixed(tmp_path: Path) -> None:
    rows: List[RunRow] = [read_run(make_run(tmp_path / "a", "saumon", problem="tsp")),
                          read_run(make_run(tmp_path / "b", "thon", problem="jssp"))]
    assert list(group_runs(rows)) == [("jssp", "qwen3-coder:30b", "gap", "default"), ("tsp", "qwen3-coder:30b", "gap", "default")]
    assert "## JSSP, qwen3-coder:30b" in report_markdown(rows) and "## TSP, qwen3-coder:30b" in report_markdown(rows)
