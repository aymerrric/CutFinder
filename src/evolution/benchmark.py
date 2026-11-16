"""
The paper's test measure (papers/sage_evocut_tmlr2026.pdf, Section 5.2): every test instance i is
solved without the cut (reference) and with it, and at each checkpoint t (5 s ... 2000 s) the
optimality gap g of both solves gives the stabilized relative gap improvement

    Delta g(i, t) = (g_ref(i, t) - g_aug(i, t)) / max(g_ref(i, t), epsilon),   epsilon = 1e-5

averaged over the instances (mean +- standard deviation), and the row average of the
checkpoint means. Delta g > 0 iff the cut has the smaller gap at t.

The gap at t is read from the trajectory the solver records during the solve (SolveResult.trajectory).
Not specified by the paper, chosen here:
  - no solution yet at t (or a failed solve with the cut): gap = 1 (100%), the worst value;
  - gaps are capped at 1, so a bad solve counts at most like no solution;
  - an instance whose reference solve failed is left out (it has no reference gap);
  - a gap at most `tolerance` (Gurobi's MIPGap, 1e-5) counts as 0: Gurobi calls both solves
    optimal, so 9.9e-6 against 0 is not a +-99% difference;
  - Delta g is clipped to [-clip, clip] (clip = 1): when the reference is optimal (gap 0) and the
    cut is not, (0 - g) / epsilon would be e.g. -2000 for a 2% gap, and a few such instances
    would decide the mean (learnt_results.md 5.3). Clipped, a cut counts at most 100% worse,
    as it counts at most 100% better; the paper's standard deviations (< 40%) imply as much.
Pure functions: no solver here (benchmark_cut.py runs the solves).
"""
import dataclasses
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from solvers.solver import SolveResult, TrajectoryPoint

NO_SOLUTION: float = 1e99  # Gurobi reports GRB.INFINITY = 1e100 as the best objective without solution
MAX_GAP: float = 1.0

# Paper, Table 2: mean Delta g (%) at the checkpoints 5, 10, 50, 150, 300, 1000, 2000 s, then the row average
PAPER_TABLE2: Dict[str, Dict[str, List[float]]] = {
    "tsp": {
        "DeepSeek-R1": [8.2, 9.1, 11.9, 18.1, 18.5, 12.9, 20.7, 14.2],
        "Gemini 3 Pro": [13.1, 15.0, 22.6, 34.2, 27.6, 26.4, 37.0, 25.1],
        "GPT-5.1": [12.1, 8.8, 15.6, 17.4, 11.3, 16.7, 18.2, 14.3],
    },
    "cwlp": {
        "DeepSeek-R1": [17.8, 18.6, 25.2, 36.3, 34.1, 33.8, 32.5, 28.3],
        "Gemini 3 Pro": [43.8, 49.4, 68.8, 76.3, 64.8, 35.2, 30.1, 52.6],
        "GPT-5.1": [35.9, 41.3, 64.7, 73.7, 57.5, 31.9, 18.3, 46.2],
    },
    "jssp": {
        "DeepSeek-R1": [41.7, 38.2, 27.5, 37.5, 28.9, 39.8, 41.5, 36.4],
        "Gemini 3 Pro": [64.1, 63.7, 63.3, 61.8, 63.2, 59.4, 59.2, 62.1],
        "GPT-5.1": [49.0, 46.5, 39.2, 38.0, 37.4, 42.8, 56.7, 44.2],
    },
}
PAPER_CHECKPOINTS: List[float] = [5, 10, 50, 150, 300, 1000, 2000]


def gap(best: Optional[float], bound: Optional[float]) -> float:
    """Gurobi's relative gap |best - bound| / |best|, in [0, MAX_GAP]; MAX_GAP without a solution."""
    if best is None or bound is None or abs(best) >= NO_SOLUTION:
        return MAX_GAP
    if best == 0:
        return 0.0 if bound == 0 else MAX_GAP
    return min(abs(best - bound) / abs(best), MAX_GAP)


def solved(result: SolveResult) -> bool:
    """The solve reached the end of optimize() (not ERROR, KILLED, TOO_LARGE or CRASHED)."""
    return result.runtime is not None


def gap_at(result: SolveResult, t: float) -> float:
    """
    The gap of the solve after t seconds. After the end of the solve (it finished or hit the time
    limit before t), its final gap; before, the last recorded point at or before t.
    """
    if not solved(result):
        return MAX_GAP
    if t >= result.runtime:  # type: ignore[operator]
        return gap(result.obj_val, result.obj_bound)
    if result.trajectory is None:
        raise ValueError(f"{result.instance}: no trajectory recorded (Solver(record_trajectory=True))")
    last: Optional[TrajectoryPoint] = None
    for point in result.trajectory:
        if point[0] > t:
            break
        last = point
    return MAX_GAP if last is None else gap(last[1], last[2])


def gap_at_work(result: SolveResult, w: float) -> Optional[float]:
    """
    The gap of the solve after `w` units of Gurobi work (CPU-independent, unlike seconds), or None
    when unknown: the solve stopped at its time limit before reaching w work, or its trajectory
    has no work (saved before 2026-11-09). After a solve that finished (optimal, infeasible...)
    with less work than w, its final gap.
    """
    if not solved(result):
        return MAX_GAP
    if result.work is None:
        return None
    if w >= result.work:
        return None if result.status == "TIME_LIMIT" else gap(result.obj_val, result.obj_bound)
    if not result.trajectory or len(result.trajectory[0]) < 4:
        return None
    last: Optional[TrajectoryPoint] = None
    for point in result.trajectory:
        if point[3] > w:
            break
        last = point
    return MAX_GAP if last is None else gap(last[1], last[2])


def improvement(g_ref: float, g_aug: float, epsilon: float, tolerance: float = 0.0,
                clip: Optional[float] = None) -> float:
    """The paper's stabilized relative gap improvement Delta g (a fraction: 0.25 = 25%), with the
    gaps at most `tolerance` counted as 0 and the result clipped to [-clip, clip] (None: not)."""
    g_ref = 0.0 if g_ref <= tolerance else g_ref
    g_aug = 0.0 if g_aug <= tolerance else g_aug
    delta: float = (g_ref - g_aug) / max(g_ref, epsilon)
    return delta if clip is None else max(-clip, min(clip, delta))


@dataclass
class BenchmarkRow:
    """One line of the table: a cut on the benchmark set, against the base model."""

    label: str
    checkpoints: List[float]
    means: List[float]  # mean Delta g (fraction) at each checkpoint
    stds: List[float]  # sample standard deviation at each checkpoint
    n_instances: int  # instances compared (both solves present, the reference one solved)
    failed: List[str]  # instances where the solve with the cut failed (counted with gap 1)
    missing: List[str]  # instances left out: no result yet with the cut, or a failed reference
    counts: Optional[List[int]] = None  # work measure: instances compared at each checkpoint
    lp_closed: Optional[float] = None  # mean share of the root gap closed by the cut's LP (lp_gap_closed)
    lp_n: int = 0  # instances in that mean

    @property
    def average(self) -> float:
        """Row average of the checkpoint means (the paper's "Average" column); checkpoints without
        any instance (nan) are left out."""
        known: List[float] = [m for m in self.means if m == m]
        return statistics.mean(known) if known else float("nan")


def compare(
    label: str,
    base: Dict[str, SolveResult],
    cut: Dict[str, SolveResult],
    checkpoints: List[float],
    epsilon: float,
    tolerance: float = 0.0,
    clip: Optional[float] = None,
    measure: str = "time",
) -> BenchmarkRow:
    """Delta g of `cut` against `base` (both by instance name) at every checkpoint (improvement()).
    measure "time": checkpoints in seconds (the paper's); "work": in Gurobi work units, where an
    instance counts at a checkpoint only if both gaps are known there (gap_at_work)."""
    deltas: List[List[float]] = [[] for _ in checkpoints]
    failed: List[str] = []
    missing: List[str] = []
    n: int = 0
    for name in sorted(base):
        if name not in cut or not solved(base[name]):
            missing.append(name)
            continue
        n += 1
        if not solved(cut[name]):
            failed.append(name)
        for k, t in enumerate(checkpoints):
            if measure == "work":
                g_ref, g_cut = gap_at_work(base[name], t), gap_at_work(cut[name], t)
                if g_ref is None or g_cut is None:
                    continue
            else:
                g_ref, g_cut = gap_at(base[name], t), gap_at(cut[name], t)
            deltas[k].append(improvement(g_ref, g_cut, epsilon, tolerance, clip))
    missing += sorted(set(cut) - set(base))
    means: List[float] = [statistics.mean(d) if d else float("nan") for d in deltas]
    stds: List[float] = [statistics.stdev(d) if len(d) > 1 else 0.0 for d in deltas]
    counts: Optional[List[int]] = [len(d) for d in deltas] if measure == "work" else None
    return BenchmarkRow(label, list(checkpoints), means, stds, n, failed, missing, counts)


def lp_gap_closed(lp_base: Optional[float], lp_cut: Optional[float], upper: Optional[float]) -> Optional[float]:
    """Share of the root gap closed by the cut's LP relaxation (minimization):
    (LP_cut - LP_base) / (UB - LP_base), UB = best known solution. 0 = the relaxation is not
    tightened, 1 = it reaches UB; above 1 = the cut removes the best known solution (invalid).
    None when a value is missing or UB <= LP_base (no root gap)."""
    if lp_base is None or lp_cut is None or upper is None or upper <= lp_base:
        return None
    return (lp_cut - lp_base) / (upper - lp_base)


def format_table(problem: str, rows: List[BenchmarkRow], with_paper: bool = True,
                 paper_rows: Optional[Dict[str, str]] = None, unit: str = "s") -> str:
    """Markdown table like the paper's Table 2 (percentages), with the paper's rows below ours.
    `paper_rows`: {name in the table: LLM of PAPER_TABLE2} (default: every LLM, "paper, <LLM>")."""
    if not rows:
        return "No result yet."
    checkpoints: List[float] = rows[0].checkpoints
    head: str = " | ".join(f"{t:g} {unit}" for t in checkpoints)
    with_lp: bool = any(row.lp_closed is not None for row in rows)
    lp_head: str = " | LP gap closed |" if with_lp else " |"
    lines: List[str] = [f"| cut | n | {head} | Average{lp_head}", "|---" * (len(checkpoints) + 3 + with_lp) + "|"]
    for row in rows:
        counts: List[Optional[int]] = list(row.counts) if row.counts else [None] * len(row.means)
        cells: str = " | ".join(
            ("-" if c == 0 else f"{100 * m:.1f} (± {100 * sd:.1f})" + (f" [n={c}]" if c is not None and c < row.n_instances else ""))
            for m, sd, c in zip(row.means, row.stds, counts))
        lp: str = ""
        if with_lp:
            lp = f" {100 * row.lp_closed:.1f} (n={row.lp_n}) |" if row.lp_closed is not None else " - |"
        lines.append(f"| {row.label} | {row.n_instances} | {cells} | {100 * row.average:.1f} |{lp}")
    if with_paper and unit == "s" and problem in PAPER_TABLE2 and checkpoints == PAPER_CHECKPOINTS:
        shown: Dict[str, str] = paper_rows if paper_rows is not None else {
            f"paper, {llm}": llm for llm in PAPER_TABLE2[problem]}
        for name, llm in shown.items():
            values: List[float] = PAPER_TABLE2[problem][llm]
            cells = " | ".join(f"{v:.1f}" for v in values[:-1])
            lines.append(f"| {name} | | {cells} | {values[-1]:.1f} |" + (" - |" if with_lp else ""))
    for row in rows:
        if row.failed:
            lines.append(f"\n{row.label}: the solve with the cut failed on {', '.join(row.failed)} (gap 1).")
        if row.missing:
            lines.append(f"\n{row.label}: left out (no result or failed reference): {', '.join(row.missing)}.")
    return "\n".join(lines)


def result_to_json(result: SolveResult) -> Dict[str, Any]:
    return dataclasses.asdict(result)


def result_from_json(data: Dict[str, Any]) -> SolveResult:
    values: Dict[str, Any] = dict(data)
    if values.get("trajectory") is not None:  # JSON has lists, not tuples
        values["trajectory"] = [tuple(point) for point in values["trajectory"]]
    return SolveResult(**values)


def split_shard(spec: str) -> Tuple[int, int]:
    """'2/5' -> (2, 5): the 2nd of 5 shards (numbered from 1)."""
    index, count = (int(x) for x in spec.split("/"))
    if not 1 <= index <= count:
        raise ValueError(f"shard {spec}: expected i/n with 1 <= i <= n")
    return index, count
