import math
from typing import Dict, List, Optional

import pytest

from evolution.benchmark import (
    MAX_GAP, PAPER_CHECKPOINTS, PAPER_TABLE2, compare, format_table, gap, gap_at, improvement, lp_gap_closed, gap_at_work,
    result_from_json, result_to_json, split_shard,
)
from solvers.solver import ERROR, SolveResult


def result(name: str, runtime: Optional[float], obj: Optional[float], bound: Optional[float],
           trajectory: Optional[List] = None) -> SolveResult:
    status: str = "TIME_LIMIT" if runtime is not None else ERROR
    return SolveResult(name, "f", 0, status, runtime, 1.0, 1.0, 1, obj, bound,
                       None if obj is None else gap(obj, bound), trajectory=trajectory)


def test_gap() -> None:
    assert gap(100.0, 90.0) == pytest.approx(0.1)
    assert gap(1e100, 90.0) == MAX_GAP  # no solution yet
    assert gap(None, None) == MAX_GAP
    assert gap(100.0, -500.0) == MAX_GAP  # capped
    assert gap(0.0, 0.0) == 0.0 and gap(0.0, -1.0) == MAX_GAP


def test_gap_at_reads_the_trajectory() -> None:
    r = result("a", 20.0, 100.0, 95.0, [(1.0, 1e100, 50.0), (3.0, 120.0, 60.0), (8.0, 100.0, 90.0)])
    assert gap_at(r, 0.5) == MAX_GAP  # nothing recorded yet
    assert gap_at(r, 2.0) == MAX_GAP  # a bound but no solution
    assert gap_at(r, 5.0) == pytest.approx(60 / 120)
    assert gap_at(r, 10.0) == pytest.approx(0.1)
    assert gap_at(r, 20.0) == pytest.approx(0.05)  # end of the solve: its final values
    assert gap_at(r, 2000.0) == pytest.approx(0.05)  # stopped earlier: the gap stays final
    assert gap_at(result("b", None, None, None), 5.0) == MAX_GAP  # failed solve
    with pytest.raises(ValueError, match="trajectory"):
        gap_at(result("c", 20.0, 100.0, 95.0), 5.0)


def test_improvement_is_the_papers_delta_g() -> None:
    assert improvement(0.4, 0.2, 1e-5) == pytest.approx(0.5)
    assert improvement(0.2, 0.4, 1e-5) == pytest.approx(-1.0)
    assert improvement(0.0, 0.0, 1e-5) == 0.0  # both solved
    assert improvement(0.0, 1e-4, 1e-5) == pytest.approx(-10.0)  # stabilized by epsilon


def test_improvement_tolerance_and_clip() -> None:
    assert improvement(9.9e-6, 0.0, 1e-5) == pytest.approx(0.99)  # the paper's raw measure: noise
    assert improvement(9.9e-6, 0.0, 1e-5, tolerance=1e-5) == 0.0  # both optimal for Gurobi
    assert improvement(0.0, 0.02, 1e-5) == pytest.approx(-2000.0)
    assert improvement(0.0, 0.02, 1e-5, tolerance=1e-5, clip=1.0) == -1.0  # at most 100% worse
    assert improvement(0.4, 0.0, 1e-5, clip=1.0) == 1.0
    assert improvement(0.4, 0.2, 1e-5, 1e-5, 1.0) == pytest.approx(0.5)  # usual values unchanged


def test_compare_and_table() -> None:
    traj = [(0.0, 100.0, 50.0)]
    base: Dict[str, SolveResult] = {
        "i1": result("i1", 10.0, 100.0, 80.0, traj),  # gap 0.5 at 5 s, 0.2 at 10 s
        "i2": result("i2", 10.0, 100.0, 80.0, traj),
        "i3": result("i3", None, None, None),  # failed reference: left out
        "i4": result("i4", 10.0, 100.0, 80.0, traj),  # no cut result yet: left out
    }
    cut: Dict[str, SolveResult] = {
        "i1": result("i1", 10.0, 100.0, 90.0, [(0.0, 100.0, 75.0)]),  # 0.25 then 0.1
        "i2": result("i2", None, None, None),  # failed with the cut: gap 1
        "i3": result("i3", 10.0, 100.0, 90.0, traj),
    }
    row = compare("c", base, cut, [5, 10], 1e-5)
    assert row.n_instances == 2 and row.failed == ["i2"] and row.missing == ["i3", "i4"]
    assert row.means[0] == pytest.approx(((0.5 - 0.25) / 0.5 + (0.5 - 1.0) / 0.5) / 2)
    assert row.means[1] == pytest.approx(((0.2 - 0.1) / 0.2 + (0.2 - 1.0) / 0.2) / 2)
    assert row.average == pytest.approx(sum(row.means) / 2)
    table: str = format_table("tsp", [row])
    assert "| c | 2 |" in table and "i2" in table and "paper" not in table  # other checkpoints
    clipped = compare("c", base, cut, [5, 10], 1e-5, tolerance=1e-5, clip=1.0)
    assert clipped.means[1] == pytest.approx(((0.2 - 0.1) / 0.2 - 1.0) / 2)  # i2's -4 clipped to -1
    full = compare("c", base, cut, PAPER_CHECKPOINTS, 1e-5)
    assert "paper, Gemini 3 Pro" in format_table("tsp", [full])
    assert format_table("tsp", []) == "No result yet."
    one: str = format_table("tsp", [full], paper_rows={"Paper cut (announced)": "Gemini 3 Pro"})
    assert "| Paper cut (announced) | | 13.1 |" in one and "DeepSeek" not in one


def test_paper_table_rows_have_seven_checkpoints_and_their_average() -> None:
    for rows in PAPER_TABLE2.values():
        for values in rows.values():
            assert len(values) == 8
            assert values[-1] == pytest.approx(sum(values[:-1]) / 7, abs=0.06)  # rounding of the paper


def test_json_round_trip() -> None:
    r = result("a", 2.0, 10.0, 9.0, [(0.1, 1e100, 1.0), (1.0, 10.0, 9.0)])
    assert result_from_json(result_to_json(r)) == r


def test_split_shard() -> None:
    assert split_shard("2/5") == (2, 5)
    for bad in ("0/5", "6/5"):
        with pytest.raises(ValueError):
            split_shard(bad)


def test_lp_gap_closed() -> None:
    assert lp_gap_closed(80.0, 90.0, 100.0) == pytest.approx(0.5)
    assert lp_gap_closed(80.0, 80.0, 100.0) == 0.0  # the relaxation is not tightened
    assert lp_gap_closed(80.0, 100.0, 100.0) == 1.0  # the LP reaches the best known solution
    assert lp_gap_closed(80.0, 105.0, 100.0) > 1  # cuts off the best known solution: invalid
    assert lp_gap_closed(100.0, 100.0, 100.0) is None  # no root gap
    assert lp_gap_closed(None, 90.0, 100.0) is None and lp_gap_closed(80.0, 90.0, None) is None


def test_table_with_the_lp_column() -> None:
    traj = [(0.0, 100.0, 50.0)]
    row = compare("c", {"i": result("i", 10.0, 100.0, 80.0, traj)}, {"i": result("i", 10.0, 100.0, 90.0, traj)},
                  PAPER_CHECKPOINTS, 1e-5)
    assert "LP gap closed" not in format_table("tsp", [row])  # no LP solved: no column
    row.lp_closed, row.lp_n = 0.25, 1
    table: str = format_table("tsp", [row], paper_rows={"announced": "Gemini 3 Pro"})
    assert "| Average | LP gap closed |" in table and "| 25.0 (n=1) |" in table
    assert table.splitlines()[-1].endswith("| 25.1 | - |")  # the paper does not report it


def test_gap_at_work_reads_the_work_in_the_trajectory() -> None:
    traj = [(0.5, 100.0, 50.0, 1.0), (3.0, 90.0, 60.0, 4.0)]  # (time, best, bound, work)
    done = result("d", 6.0, 90.0, 90.0, traj)
    done.work, done.status = 8.0, "OPTIMAL"
    assert gap_at_work(done, 0.5) == MAX_GAP  # nothing yet
    assert gap_at_work(done, 2.0) == pytest.approx(0.5)
    assert gap_at_work(done, 5.0) == pytest.approx(1 / 3)
    assert gap_at_work(done, 100.0) == 0.0  # finished (optimal) before 100 work: its final gap
    stopped = result("s", 2000.0, 90.0, 60.0, traj)
    stopped.work, stopped.status = 8.0, "TIME_LIMIT"
    assert gap_at_work(stopped, 100.0) is None  # stopped before reaching 100 work: unknown
    old = result("o", 6.0, 90.0, 80.0, [(0.5, 100.0, 50.0)])
    old.work = 8.0
    assert gap_at_work(old, 2.0) is None  # saved before the work was recorded


def test_compare_in_work_units_counts_the_known_instances() -> None:
    traj = [(0.0, 100.0, 50.0, 0.0)]
    def solve(name: str, best: float, bound: float, work: float, status: str) -> SolveResult:
        r = result(name, 10.0, best, bound, traj)
        r.work, r.status = work, status
        return r
    base = {"a": solve("a", 100, 80, 3000, "TIME_LIMIT"), "b": solve("b", 100, 80, 900, "TIME_LIMIT")}
    cut = {"a": solve("a", 100, 90, 2500, "TIME_LIMIT"), "b": solve("b", 100, 90, 3000, "TIME_LIMIT")}
    row = compare("c", base, cut, [10, 2000], 1e-5, measure="work")
    assert row.counts == [2, 1]  # at 2000 wu: b's base stopped at 900, a's cut at 2500 (known)
    table: str = format_table("tsp", [row], unit="wu")
    assert "| 10 wu | 2000 wu |" in table and "[n=1]" in table and "paper" not in table
