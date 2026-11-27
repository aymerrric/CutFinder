import json
import os
from pathlib import Path

import pytest

import falsify_cut
from problems import PROBLEMS
from proving.falsify import ERROR, NO_COUNTEREXAMPLE, REFUTED, SAFE_NOT_VALID, TINY, falsify, tiny_instances
from proving.problems import LEAN_PROBLEMS

CUTS: str = os.path.join(os.path.dirname(__file__), "..", "cuts")


def cut(problem: str, name: str) -> str:
    with open(os.path.join(CUTS, problem, f"{name}.py")) as f:
        return f.read()


@pytest.mark.parametrize("problem", ["tsp", "cwlp", "jssp"])
def test_tiny_instances_meet_the_lean_assumptions(problem: str) -> None:
    instances = tiny_instances(problem, 60)
    assert len({inst["name"] for inst in instances}) == 60
    assert all(LEAN_PROBLEMS[problem].assumptions(inst["args"]) == [] for inst in instances)


def test_every_registered_problem_has_a_tiny_generator() -> None:
    assert TINY.keys() == PROBLEMS.keys()


def test_imo6_checks_distinct_grid_sizes_only() -> None:
    assert [case["args"]["N"] for case in tiny_instances("imo6")] == [2, 3, 4, 5]
    assert tiny_instances("imo6", 2, seed=8) == tiny_instances("imo6")[:2]


@pytest.mark.parametrize("problem,count", [("mcnd", 6), ("gisp", 10), ("corridor", 6), ("imo6", 4)])
def test_new_generators_are_reproducible_and_seed_cuts_keep_feasible_points(problem: str, count: int) -> None:
    cases = tiny_instances(problem, count, seed=1)
    assert cases == tiny_instances(problem, count, seed=1)
    assert len({case["name"] for case in cases}) == count
    if problem != "imo6":
        assert [case["args"] for case in cases] != [case["args"] for case in tiny_instances(problem, count, seed=2)]
    verdict = falsify(problem, PROBLEMS[problem].baseline_code, cases)
    assert verdict.verdict == NO_COUNTEREXAMPLE and verdict.instances == count


def test_mcnd_strong_linking_excludes_circulations_and_preserves_the_optimum() -> None:
    verdict = falsify("mcnd", cut("mcnd", "textbook"), tiny_instances("mcnd", 2))
    assert verdict.verdict == SAFE_NOT_VALID
    assert verdict.counterexample.instance == "tiny_mcnd_cycle_1"
    assert verdict.counterexample.cut_optimum == verdict.counterexample.base_optimum


@pytest.mark.parametrize("problem,code", [
    ("mcnd", "m.addConstr(m._y[1,2] == 0)"),
    ("imo6", "m.addConstr(m._h.sum(1, '*') == 0)"),
    ("gisp", "m.addConstr(m._x.sum() <= 1)"),
    ("corridor", "m.addConstr(m._x[1] == 0)"),
])
def test_generated_cases_refute_an_unsafe_cut(problem: str, code: str) -> None:
    verdict = falsify(problem, code, tiny_instances(problem, 1))
    assert verdict.verdict == REFUTED
    assert verdict.counterexample.violation > 0


def test_a_cut_that_removes_every_tour_is_refuted() -> None:
    verdict = falsify("tsp", cut("tsp", "invalid_position_bound"), tiny_instances("tsp", 5))
    assert verdict.verdict == REFUTED and verdict.counterexample is not None
    assert verdict.counterexample.cut_optimum is None  # no tour left
    assert "position_bound" in verdict.counterexample.row or "u[" in verdict.counterexample.row
    assert verdict.counterexample.point  # the base model's optimal tour


def test_a_bound_above_the_optimum_is_refuted() -> None:
    # learnt_results.md 5.16: "at least 2 open warehouses", false when one warehouse suffices
    verdict = falsify("cwlp", "m.addConstr(m._y.sum() >= 2)", tiny_instances("cwlp", 20))
    assert verdict.verdict == REFUTED
    assert verdict.counterexample.cut_optimum is None or verdict.counterexample.cut_optimum > verdict.counterexample.base_optimum


def test_a_cut_that_removes_only_non_optimal_points_is_safe_not_valid() -> None:
    # the makespan never needs more than the total processing time, but the formulation allows it
    verdict = falsify("jssp", "m.addConstr(m._Cmax <= sum(map(sum, times)))", tiny_instances("jssp", 10))
    assert verdict.verdict == SAFE_NOT_VALID
    assert "Cmax" in verdict.counterexample.row and verdict.counterexample.violation > 0


@pytest.mark.parametrize("problem", ["cwlp", "jssp"])
def test_the_textbook_cuts_have_no_counterexample(problem: str) -> None:
    verdict = falsify(problem, cut(problem, "textbook"), tiny_instances(problem, 40))
    assert verdict.verdict == NO_COUNTEREXAMPLE and verdict.instances > 0


def test_code_that_fails_is_an_error() -> None:
    assert falsify("tsp", "m._nothing.sum()", tiny_instances("tsp", 2)).verdict == ERROR


def test_the_command_writes_one_line_per_cut(tmp_path: Path) -> None:
    out: Path = tmp_path / "f.jsonl"
    args = falsify_cut.parse_args(["--problem_name", "tsp", "--cut_file", os.path.join(CUTS, "tsp", "invalid_position_bound.py"),
                                   "--instances", "3", "--out", str(out)])
    verdicts = falsify_cut.main(args)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [v.verdict for v in verdicts] == [REFUTED] and lines[0]["verdict"] == REFUTED
    assert lines[0]["counterexample"]["args"]["n"] >= 4
