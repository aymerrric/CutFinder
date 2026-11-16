"""
summary.md of a run folder: what to read first. Built only from the folder's files (population,
status, review, LLM proposals), so it can be rewritten at any time, also for an old run.
"""
import glob
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

from evolution.individual import Individual
from evolution.population import Population
from evolution.recorder import ACCEPTED
from evolution.review import paired_difference
from utils.run_files import LLM_DIR, POPULATION, REVIEW, STATUS, SUMMARY

SETTINGS: List[str] = [  # main.py arguments shown in the summary, when known
    "problem_name", "population_size", "num_generations", "elitism_ratio", "crossover_rate",
    "max_llm_attempts", "temperature", "seed", "work_limit", "threads", "fitness_mode", "fitness_seeds",
    "review_seeds",
]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _when(timestamp: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp)) if timestamp else "-"


def generations(population: Population) -> List[List[Individual]]:
    """Every generation, the current one included (history holds only the previous ones)."""
    return population.history + ([population.current] if population.current else [])


def found_in(population: Population, ind: Individual) -> Optional[int]:
    """The generation where `ind` appeared."""
    for generation, individuals in enumerate(generations(population)):
        if any(other.id == ind.id for other in individuals):
            return generation
    return None


def reason_kind(reason: Optional[str]) -> str:
    """Short family of a rejection reason: 'invalid cut', 'useless cut', 'too slow', 'KILLED on ...'."""
    if not reason:
        return "-"
    head: str = reason.split(":")[0]
    return " ".join(head.split()[:3])


def attempts_table(run_dir: str) -> List[str]:
    """Per operator: proposals, accepted, acceptance rate, and the main rejection reasons."""
    paths: List[str] = sorted(glob.glob(os.path.join(run_dir, LLM_DIR, "gen_*", "*.json")))
    if not paths:
        return ["No LLM proposal recorded."]
    outcomes: DefaultDict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for path in paths:
        with open(path) as f:
            attempt: Dict[str, Any] = json.load(f)
        outcomes[attempt["operator"]][attempt["outcome"]] += 1
        if attempt["outcome"] != ACCEPTED:
            reasons[reason_kind(attempt.get("reason"))] += 1
    lines: List[str] = [
        f"{len(paths)} proposals in `{LLM_DIR}/` (one folder per generation).", "",
        "| operator | proposals | accepted | rate | format errors | rejected | LLM errors |",
        "|---|---|---|---|---|---|---|",
    ]
    total: Counter = Counter()
    for operator, counts in sorted(outcomes.items()):
        total += counts
        lines.append(_attempt_row(operator, counts))
    lines.append(_attempt_row("**all**", total))
    if reasons:
        lines += ["", "Most frequent rejection reasons: " + ", ".join(
            f"{kind} ({n})" for kind, n in reasons.most_common(5)
        ) + "."]
    return lines


def _attempt_row(operator: str, counts: Counter) -> str:
    n: int = sum(counts.values())
    rate: str = f"{100 * counts[ACCEPTED] / n:.0f}%" if n else "-"
    return (f"| {operator} | {n} | {counts[ACCEPTED]} | {rate} | {counts['format_error']} | "
            f"{counts['rejected']} | {counts['llm_error']} |")


def generations_table(population: Population) -> List[str]:
    lines: List[str] = ["| generation | best ever | max | mean | individuals |", "|---|---|---|---|---|"]
    best: float = float("-inf")
    for generation, individuals in enumerate(generations(population)):
        fitness: List[float] = [ind.fitness for ind in individuals]
        best = max(best, max(fitness))
        origins: str = ", ".join(f"{g} {n}" for g, n in Counter(ind.generator for ind in individuals).items())
        lines.append(f"| {generation} | {best:.3f} | {max(fitness):.3f} | {statistics.mean(fitness):.3f} | {origins} |")
    return lines


def review_section(review: Dict[str, Any]) -> List[str]:
    models: Dict[str, Dict[str, Any]] = {m["name"]: m for m in review["models"]}
    instances: str = (  # reviews before 2026-11-02 reused the evaluation set (no "instances" key)
        f"on the {len(review['instances'])} held-out instances of `{review['instance_set']}` (never seen by the GA)"
        if "instances" in review else "on the evaluation set (the instances the GA was scored on)"
    )
    mode: str = review.get("fitness_mode", "gap")  # reviews before 2026-11-02: gap runs only
    lines: List[str] = [
        f"The base model, the seed cut and the best cut {instances}, under Gurobi seeds {review['seeds']} "
        "(fitness against the base model solved with the same seed), and the % of the root gap "
        f"closed by each model's LP relaxation. ± = sample std over the {len(review['seeds'])} review seeds "
        "(same cut, same instances: solver noise, not the spread between runs). From `review.json`.", "",
        f"| model | fitness mean ± std over the seeds | min | max | mean gap | LP gap closed | fitness during the run ({mode}) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        fitness: List[float] = [s["fitness"] for s in m["scores"]]
        run: str = f"{m['run_fitness']:.3f}" if m.get("run_fitness") is not None else "-"
        lp: str = f"{m['lp_fitness']:.2f}%" if m.get("lp_fitness") is not None else "-"  # older reviews: none
        lines.append(f"| {name} | {m['fitness_mean']:.3f} ± {m['fitness_std']:.3f} | {min(fitness):.3f} | "
                     f"{max(fitness):.3f} | {100 * m['mean_gap']:.2f}% | {lp} | {run} |")
    if "best" in models and "seed" in models:
        best: List[float] = [s["fitness"] for s in models["best"]["scores"]]
        seed: List[float] = [s["fitness"] for s in models["seed"]["scores"]]
        wins: int = sum(b > s for b, s in zip(best, seed))
        mean, std = paired_difference(best, seed)
        lines += ["", f"The best cut beats the seed cut on **{wins}/{len(best)} seeds** "
                      f"({mean:+.3f} on average, ± {std:.3f}: std over the seeds of the per-seed difference)."]
        if models["best"].get("lp_fitness") is not None and models["seed"].get("lp_fitness") is not None:
            lines[-1] += (f" LP relaxation: {models['best']['lp_fitness']:.2f}% of the root gap closed vs "
                          f"{models['seed']['lp_fitness']:.2f}% for the seed cut.")
    elif "seed" in models:
        lines += ["", "The best cut is the seed cut itself: no cut better than the seed was found."]
    worked: List[str] = [f"{name} {100 * m['work_change_mean']:+.1f}% ± {100 * m['work_change_std']:.1f}%"
                         for name, m in models.items() if name != "base" and m.get("work_change_mean") is not None]
    if worked:  # gap_work runs
        lines += ["", f"Work to prove optimality on the {len(review.get('optimality_instances', []))} held-out "
                      "instances the base model solves (mean change vs the base model, ± std over the seeds; "
                      "< 0 = faster): " + ", ".join(worked) + "."]
    return lines


def summary_markdown(run_dir: str, population: Population) -> str:
    status: Dict[str, Any] = _read_json(os.path.join(run_dir, STATUS)) or {}
    review: Optional[Dict[str, Any]] = _read_json(os.path.join(run_dir, REVIEW))
    name: str = status.get("run") or os.path.basename(os.path.normpath(run_dir))
    arguments: Dict[str, Any] = status.get("arguments") or {}
    duration: str = "-"
    if status.get("started") and status.get("updated"):
        duration = f"{(status['updated'] - status['started']) / 60:.0f} min"
    lines: List[str] = [
        f"# Run {name}", "",
        f"- **State:** {status.get('state', '?')}"
        + (f" ({status['error']})" if status.get("error") else "")
        + f", generation {population.generation if population.current else '-'}"
        + f"/{status.get('num_generations', '?')}",
        f"- **Machine:** {status.get('host', '?')} ({status.get('cpu', '?')})",
        f"- **LLM:** {status.get('model', '?')}",
        f"- **Gurobi license:** `{status.get('license_file') or '?'}`",
        f"- **Started:** {_when(status.get('started'))}, last update {_when(status.get('updated'))} "
        f"({duration})",
    ]
    settings: List[str] = [f"{key} = {arguments[key]}" for key in SETTINGS if key in arguments]
    if settings:
        lines.append(f"- **Settings:** {', '.join(settings)}")
    best: Optional[Individual] = population.best
    lines += ["", "## Best cut", ""]
    if best is None:
        lines.append("No generation finished yet.")
    else:
        found: Optional[int] = found_in(population, best)
        lines += [
            f"- **Fitness during the run:** {best.fitness:.3f} (10 = same gap as the base model, more = better)",
            f"- **Found in generation** {found if found is not None else '?'} by `{best.generator}`"
            f" (id {best.id}, parents {best.parents or '-'})",
            f"- **Idea:** {best.idea}", "", "```python", best.code, "```",
        ]
    if review is not None:
        lines += ["", "## Review under several seeds", ""] + review_section(review)
    if population.current:
        lines += ["", "## Fitness per generation", ""] + generations_table(population)
    lines += ["", "## LLM proposals", ""] + attempts_table(run_dir)
    lines += [
        "", "## Files", "",
        f"`{POPULATION}` (population, to resume), `{STATUS}`, `run.log`, `{REVIEW}`, "
        f"`{LLM_DIR}/gen_<k>/child_<c>_attempt_<a>.json` (every LLM proposal with its conversation, "
        "reply, outcome and reason). Gurobi logs: `logs/<run>/`.",
    ]
    return "\n".join(lines) + "\n"


def write_summary(run_dir: str, population: Population) -> str:
    """Write <run_dir>/summary.md; returns its path."""
    path: str = os.path.join(run_dir, SUMMARY)
    with open(path, "w") as f:
        f.write(summary_markdown(run_dir, population))
    return path
