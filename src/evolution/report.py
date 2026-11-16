"""
Report over several runs (e.g. one run per lab machine): for each problem and LLM, the best cut of
all its runs and the mean of the best run of each computer.

Runs are grouped by (problem, LLM, fitness mode, instance level): different problems are never
compared, a gap fitness (10 = no gain) and an LP fitness (% of the root gap closed) are on
different scales, and runs scored and reviewed on the sets of different instance levels
(--instance_level) are on different instances, so none of them are averaged together. Inside a group,
runs are ranked by their review score in the metric of their fitness mode: the gap fitness under
the review seeds, or the LP gap closed. The run fitness is biased upwards (the best of many noisy
scores: learnt_results.md 1.4), so it is shown but never used to rank. Runs without a review are
listed, not ranked, and so are the runs whose best cut fails the validity check of
check_run_cuts.py (runs/<run>/validity.json): their review score is not a result.

Only reads the run folders' JSON files (status.json, population.json, review.json): no solver.
"""
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from evolution.population import Population
from utils.run_files import POPULATION, REVIEW, STATUS

VALIDITY: str = "validity.json"  # written by check_run_cuts.py in runs/<run>/

GroupKey = Tuple[str, str, str, str]  # (problem, LLM, fitness mode, instance level)
STALLED_AFTER: float = 3600  # seconds: a "running" run without any update for longer is shown as stalled
#                              (a run killed without warning keeps "running" in its status.json)


@dataclass
class RunRow:
    """What the report keeps from one run folder."""

    run: str
    host: str
    problem: str  # runs before 2026-11-02: "tsp"
    model: str
    fitness_mode: str  # "gap" or "lp" (runs before 2026-11-02: "gap")
    prompt_variant: str
    instance_level: str  # sets the run was scored and reviewed on (runs before 2026-11-12: "default")
    state: str
    generation: Optional[int]
    num_generations: Optional[int]
    run_fitness: Optional[float]  # best fitness during the run, on the run's own scale
    review_gap: Optional[float]  # review: gap fitness, mean over the review seeds
    review_gap_std: Optional[float]
    review_lp: Optional[float]  # review: % of the root gap closed (reviews before 2026-11-02: none)
    held_out: bool  # the review ran on the held-out test set (else on the evaluation set)
    best_idea: Optional[str]
    best_code: Optional[str]
    best_valid: Optional[bool] = None  # check_run_cuts.py: the best cut keeps every optimum (None = not checked)
    validity_note: str = ""  # e.g. "INVALID (2/9)"

    @property
    def score(self) -> Optional[float]:
        """The review score in the metric of the run's fitness mode (None = not reviewed, or a best
        cut found invalid: its score is not a result)."""
        if self.best_valid is False:
            return None
        return self.review_lp if self.fitness_mode == "lp" else self.review_gap


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def state_of(status: Dict[str, Any], now: Optional[float] = None) -> str:
    """The status file's state, or "stalled" for a "running" run that stopped updating it."""
    state: str = status.get("state", "?")
    updated: Optional[float] = status.get("updated")
    if state == "running" and updated is not None and (now or time.time()) - updated > STALLED_AFTER:
        return "stalled"
    return state


def read_run(run_dir: str) -> Optional[RunRow]:
    """One row per run folder (None if it has no status.json: not a run)."""
    status: Optional[Dict[str, Any]] = _read_json(os.path.join(run_dir, STATUS))
    if status is None:
        return None
    arguments: Dict[str, Any] = status.get("arguments") or {}
    review: Optional[Dict[str, Any]] = _read_json(os.path.join(run_dir, REVIEW))
    reviewed: Optional[Dict[str, Any]] = None  # the best cut's review entry
    if review is not None:
        models: Dict[str, Dict[str, Any]] = {m["name"]: m for m in review["models"]}
        reviewed = models.get("best") or models.get("seed")  # no "best": the seed cut stayed the best
    validity: Optional[Dict[str, Any]] = _read_json(os.path.join(run_dir, VALIDITY))
    valid: Optional[bool] = None
    note: str = "-"
    if validity is not None:  # check_run_cuts.py: validity instances, and maybe the public bounds
        public: Dict[str, Any] = validity.get("public") or {}
        valid = bool(validity["best"]["valid"]) and not public.get("invalid")
        notes: List[str] = []
        if not validity["best"]["valid"]:
            notes.append(f"INVALID ({len(validity['best']['removed'])}/{validity['n_instances']})")
        if public.get("invalid"):
            notes.append(f"INVALID on public ({len(public['invalid'])}/{public['checked']})")
        note = ", ".join(notes) or ("valid" + (", public bounds too" if public else ""))
    idea: Optional[str] = None
    code: Optional[str] = None
    if os.path.exists(os.path.join(run_dir, POPULATION)):
        population: Population = Population.load(os.path.join(run_dir, POPULATION))
        if population.best is not None:
            idea, code = population.best.idea, population.best.code
    return RunRow(
        run=os.path.basename(os.path.normpath(run_dir)),
        host=status.get("host", "?"),
        problem=arguments.get("problem_name") or "tsp",
        model=status.get("model") or arguments.get("model_name") or "?",
        fitness_mode=arguments.get("fitness_mode", "gap"),
        prompt_variant=arguments.get("prompt_variant") or "default",
        instance_level=arguments.get("instance_level") or "default",
        state=state_of(status),
        generation=status.get("generation"),
        num_generations=status.get("num_generations"),
        run_fitness=status.get("best"),
        review_gap=reviewed["fitness_mean"] if reviewed else None,
        review_gap_std=reviewed["fitness_std"] if reviewed else None,
        review_lp=reviewed.get("lp_fitness") if reviewed else None,
        held_out=review is not None and "instances" in review,
        best_idea=idea,
        best_code=code,
        best_valid=valid,
        validity_note=note,
    )


def group_runs(rows: List[RunRow]) -> Dict[GroupKey, List[RunRow]]:
    groups: Dict[GroupKey, List[RunRow]] = {}
    for row in rows:
        groups.setdefault((row.problem, row.model, row.fitness_mode, row.instance_level), []).append(row)
    return dict(sorted(groups.items()))


def best_per_host(rows: List[RunRow]) -> List[RunRow]:
    """For each computer, its best reviewed run (computers without a reviewed run are left out)."""
    best: Dict[str, RunRow] = {}
    for row in rows:
        if row.score is not None and (row.host not in best or row.score > best[row.host].score):
            best[row.host] = row
    return sorted(best.values(), key=lambda r: r.host)


def _mean_std(values: List[Optional[float]], source: str = "") -> str:
    """'mean ± std (n=k)': sample std of the known values (none for one value). `source` says what the
    values are (e.g. "computers"): the std is the spread between them, so it is written next to it."""
    known: List[float] = [v for v in values if v is not None]
    if not known:
        return "-"
    if len(known) == 1:  # a std needs two values: "± 0" would read as "no spread"
        return f"{known[0]:.3f} (n=1: no std)"
    std: float = statistics.stdev(known)
    between: str = f", std between the {len(known)} {source}" if source else ""
    return f"{statistics.mean(known):.3f} ± {std:.3f} (n={len(known)}{between})"


def _number(value: Optional[float], fmt: str = ".3f", suffix: str = "") -> str:
    return "-" if value is None else f"{value:{fmt}}{suffix}"


def group_section(key: GroupKey, rows: List[RunRow]) -> List[str]:
    problem, model, mode, level = key
    metric: str = "LP gap closed (%)" if mode == "lp" else "gap fitness (10 = no cut)"
    hosts: List[RunRow] = best_per_host(rows)
    lines: List[str] = [f"## {problem.upper()}, {model}, fitness `{mode}`, instances `{level}`", ""]
    invalid: int = sum(r.best_valid is False for r in rows)
    lines.append(f"{len(rows)} runs on {len({r.host for r in rows})} computers, "
                 f"{sum(r.state == 'finished' for r in rows)} finished, {len(hosts)} computers with a review. "
                 f"Ranked by the review's {metric}."
                 + (f" {invalid} runs left out: their best cut is invalid (check_run_cuts.py)." if invalid else ""))
    lines.append("")
    if hosts:
        top: RunRow = max(hosts, key=lambda r: r.score)
        lines += [
            f"- **Best of all:** `{top.run}` ({top.host}): review gap fitness "
            f"{_number(top.review_gap)} ± {_number(top.review_gap_std)} (std over the review seeds), LP gap closed "
            f"{_number(top.review_lp, '.2f', '%')}, run fitness {_number(top.run_fitness)}"
            + ("" if top.held_out else " (review on the evaluation set, not held out)"),
            f"- **Mean of the best run of each computer** ({len(hosts)} computers; ± = std between "
            "these runs, i.e. between runs of this LLM):",
            f"  - review gap fitness: {_mean_std([r.review_gap for r in hosts], 'computers')}",
            f"  - review LP gap closed (%): {_mean_std([r.review_lp for r in hosts], 'computers')}",
            f"  - run fitness ({mode} scale, biased upwards): {_mean_std([r.run_fitness for r in hosts], 'computers')}",
            "",
            "Best cut of all:", "", f"> {top.best_idea}", "",
            "```python", top.best_code or "", "```", "",
        ]
    else:
        lines += ["No reviewed run yet.", ""]
    lines += [
        "| run | computer | state | generation | prompts | run fitness | review gap fitness (± std over the review seeds) "
        "| LP gap closed | held out | best cut valid |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.score is None, -(r.score or 0.0))):
        generation: str = f"{_number(r.generation, 'd')}/{_number(r.num_generations, 'd')}"
        gap: str = "-" if r.review_gap is None else f"{r.review_gap:.3f} ± {_number(r.review_gap_std)}"
        lines.append(f"| `{r.run}` | {r.host} | {r.state} | {generation} | {r.prompt_variant} | "
                     f"{_number(r.run_fitness)} | {gap} | {_number(r.review_lp, '.2f', '%')} | "
                     f"{'yes' if r.held_out else 'no' if r.review_gap is not None else '-'} | {r.validity_note} |")
    return lines + [""]


def between_llms_section(groups: Dict[GroupKey, List[RunRow]]) -> List[str]:
    """
    One table per (problem, fitness mode, instance level) with at least two LLMs: each LLM's mean
    review score over the best run of each computer (± std between those runs), then the mean of
    the LLM means ± the std between the LLMs. The two stds answer different questions: how much a
    run of one LLM varies, and how much the LLM changes the result.
    """
    by_setting: Dict[Tuple[str, str, str], Dict[str, List[RunRow]]] = {}
    for (problem, model, mode, level), rows in groups.items():
        hosts: List[RunRow] = best_per_host([r for r in rows if r.held_out])  # same test set for every LLM
        if hosts:
            by_setting.setdefault((problem, mode, level), {})[model] = hosts
    lines: List[str] = []
    for (problem, mode, level), models in sorted(by_setting.items()):
        if len(models) < 2:
            continue
        metric: str = "LP gap closed (%)" if mode == "lp" else "gap fitness"
        means: Dict[str, float] = {m: statistics.mean(r.score for r in hosts) for m, hosts in models.items()}
        lines += [f"## Between LLMs: {problem.upper()}, fitness `{mode}`, instances `{level}`", "",
                  f"Review {metric} of the best run of each computer (held-out reviews only). Per LLM: "
                  "mean ± std between its runs (one per computer). Last line: mean of the LLM means ± "
                  "std between the LLMs.", "",
                  f"| LLM | runs | review {metric}: mean ± std between runs | best run |", "|---|---|---|---|"]
        for model, hosts in sorted(models.items(), key=lambda item: -means[item[0]]):
            top: RunRow = max(hosts, key=lambda r: r.score)
            lines.append(f"| {model} | {len(hosts)} | {_mean_std([r.score for r in hosts])} | "
                         f"{top.score:.3f} (`{top.run}`) |")
        lines += [f"| **between the {len(models)} LLMs** | | {_mean_std(list(means.values()), 'LLM means')} | |", ""]
    return lines


def report_markdown(rows: List[RunRow], title: str = "Report over the runs") -> str:
    lines: List[str] = [f"# {title}", "",
                        "One section per problem, LLM, fitness mode and instance level (problems, "
                        "the two fitness scales and the instance sets are never mixed). "
                        "\"Best\" = the review score (several Gurobi seeds; held-out instances since "
                        "2026-11-02) in the metric of the run's fitness mode, not the run fitness.", "",
                        "Where each ± comes from: next to one run's review score, the std over the "
                        "Gurobi seeds of its review (same cut, same instances: solver noise); next to a "
                        "mean over computers, the std between the best runs of these computers (between "
                        "runs of one LLM); in the \"Between LLMs\" tables, the std between the LLM means.", ""]
    if not rows:
        return "\n".join(lines + ["No run found."]) + "\n"
    groups: Dict[GroupKey, List[RunRow]] = group_runs(rows)
    for key, group in groups.items():
        lines += group_section(key, group)
    return "\n".join(lines + between_llms_section(groups))


def write_report(run_dirs: List[str], path: str, title: str = "Report over the runs") -> str:
    rows: List[RunRow] = [row for row in (read_run(d) for d in run_dirs) if row is not None]
    with open(path, "w") as f:
        f.write(report_markdown(rows, title))
    return path
