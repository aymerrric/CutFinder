"""
Final review of cuts under several Gurobi seeds.

During a run, a cut's fitness comes from one deterministic solve per evaluation instance (one
Gurobi seed). But any change to a model changes Gurobi's search path, so a cut can score well by
luck (performance variability). The review solves the evaluation instances again under several
seeds, with the base model and with each cut. Per seed, the fitness uses the run's formula
against the base model solved with the same seed:
    fitness_s = 10 * exp(-mean_i (gap_cut_i,s - gap_base_i,s) / gap_base_i,s)
It then reports the mean and standard deviation over the seeds.
"""
import functools
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from loaders.base import Instance
from models.transforms import with_cut
from solvers.solver import Formulation, SolveResult, Solver

BASE_NAME: str = "base"  # the model without any cut (fitness 10 at every seed, by definition)
NO_SOLUTION_GAP: float = 1.0  # gap counted when a solve ends without a solution (as in the run)

SolverFactory = Callable[[int], Solver]  # Gurobi seed -> solver with the run's settings


@dataclass
class SeedScore:
    """One model under one seed, over the evaluation instances."""

    seed: int
    fitness: float
    mean_gap: float
    gaps: List[float]  # per evaluation instance
    statuses: List[str]  # per evaluation instance (WORK_LIMIT normally)
    skipped: List[str] = field(default_factory=list)  # instances the base solved at this seed


@dataclass
class CutReview:
    name: str
    code: Optional[str]  # None for the base model
    run_fitness: Optional[float]  # the fitness the run gave it (one seed)
    scores: List[SeedScore] = field(default_factory=list)

    @property
    def fitnesses(self) -> List[float]:
        return [s.fitness for s in self.scores]

    @property
    def mean(self) -> float:
        return statistics.mean(self.fitnesses)

    @property
    def std(self) -> float:
        return statistics.stdev(self.fitnesses) if len(self.scores) > 1 else 0.0

    @property
    def mean_gap(self) -> float:
        return statistics.mean(s.mean_gap for s in self.scores)

    def to_dict(self) -> Dict[str, object]:
        return {**asdict(self), "fitness_mean": self.mean, "fitness_std": self.std,
                "mean_gap": self.mean_gap}


def review_cuts(
    base: Formulation,
    cuts: Dict[str, str],
    evaluation: List[Instance],
    seeds: List[int],
    make_solver: SolverFactory,
    run_fitness: Optional[Dict[str, float]] = None,
) -> List[CutReview]:
    """
    Solve every evaluation instance with `base` and with each cut (`name -> code`) under each
    seed. Returns one CutReview per model, the base model first.
    """
    run_fitness = run_fitness or {}
    reviews: List[CutReview] = [CutReview(BASE_NAME, None, None)]
    reviews += [CutReview(name, code, run_fitness.get(name)) for name, code in cuts.items()]
    for seed in seeds:
        solver: Solver = make_solver(seed)
        solver.add_formulation(f"review_{BASE_NAME}", base)
        for name, code in cuts.items():
            solver.add_formulation(f"review_{name}", functools.partial(with_cut, base=base, code=code))
        base_results: List[SolveResult] = [solver.solve(inst, f"review_{BASE_NAME}") for inst in evaluation]
        for review in reviews:
            results: List[SolveResult] = (
                base_results if review.code is None
                else [solver.solve(inst, f"review_{review.name}") for inst in evaluation]
            )
            review.scores.append(score(seed, evaluation, base_results, results))
    return reviews


def score(seed: int, evaluation: List[Instance], base: List[SolveResult], cut: List[SolveResult]) -> SeedScore:
    """The run's fitness formula, against the base results of the same seed."""
    gaps: List[float] = [r.mip_gap if r.mip_gap is not None else NO_SOLUTION_GAP for r in cut]
    ratios: List[float] = []
    skipped: List[str] = []
    for inst, b, gap in zip(evaluation, base, gaps):
        if b.mip_gap is None or b.mip_gap <= 0:  # solved by the base at this seed: no ratio
            skipped.append(inst["name"])
            continue
        ratios.append((gap - b.mip_gap) / b.mip_gap)
    fitness: float = 10 * math.exp(-statistics.mean(ratios)) if ratios else float("nan")
    return SeedScore(seed, fitness, statistics.mean(gaps), gaps, [r.status for r in cut], skipped)


def format_review(reviews: List[CutReview]) -> str:
    """A table: fitness mean ± std over the seeds, min-max, mean gap, the run's fitness."""
    n: int = len(reviews[0].scores)
    lines: List[str] = [
        f"review of the cuts under {n} Gurobi seeds ({', '.join(str(s.seed) for s in reviews[0].scores)}):",
        f"  {'model':<8} {'fitness mean ± std':>20} {'min':>7} {'max':>7} {'mean gap':>9} {'run fitness':>12}",
    ]
    for r in reviews:
        run: str = f"{r.run_fitness:.3f}" if r.run_fitness is not None else "-"
        lines.append(
            f"  {r.name:<8} {r.mean:>11.3f} ± {r.std:<6.3f} {min(r.fitnesses):>7.3f} {max(r.fitnesses):>7.3f} "
            f"{100 * r.mean_gap:>8.2f}% {run:>12}"
        )
    by_name: Dict[str, CutReview] = {r.name: r for r in reviews}
    if "best" in by_name and "seed" in by_name:
        best, seed = by_name["best"], by_name["seed"]
        wins: int = sum(b > s for b, s in zip(best.fitnesses, seed.fitnesses))
        lines.append(
            f"  best - seed: {best.mean - seed.mean:+.3f} on average; the best cut beats the seed cut "
            f"on {wins}/{n} seeds"
        )
    statuses: set = {st for r in reviews for s in r.scores for st in s.statuses} - {"WORK_LIMIT", "OPTIMAL"}
    if statuses:
        lines.append(f"  note: some solves ended with {', '.join(sorted(statuses))}")
    return "\n".join(lines)
