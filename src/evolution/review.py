"""
Final review of cuts under several Gurobi seeds.

During a run, a cut's fitness comes from one deterministic solve per evaluation instance (one
Gurobi seed). But any change to a model changes Gurobi's search path, so a cut can score well by
luck (performance variability). The review solves the evaluation instances again under several
seeds, with the base model and with each cut. Per seed, the fitness uses the run's formula
against the base model solved with the same seed:
    fitness_s = 10 * exp(-mean_i (gap_cut_i,s - gap_base_i,s) / gap_base_i,s)
It then reports the mean and standard deviation over the seeds.

It also measures every model's LP relaxation, whatever fitness the run used (one solve per
instance: an LP is deterministic, so no seed), as the share of the root gap it closes:
    lp_fitness = 100 * mean_i (LP_model_i - LP_base_i) / (UB_i - LP_base_i)
with UB_i the cheapest tour the base model found on instance i under the review seeds (the
evaluator's FitnessMode.LP formula). The gap says whether a cut makes Gurobi faster, the LP
whether it is a stronger formulation (learnt_results.md 1.7: the two can disagree).

For a run of the "gap_work" fitness mode, it also solves a few held-out instances that the base
model PROVES optimal under the work limit (optimality_test.pkl), and reports, per seed, the mean
relative change of the work needed to prove optimality (work_ratios: < 0 = the cut proves it
faster, > 0 = slower).
"""
import functools
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loaders.base import Instance
from models.transforms import relaxed, with_cut
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
    run_fitness: Optional[float]  # the fitness the run gave it (on the run's own scale: gap or lp)
    scores: List[SeedScore] = field(default_factory=list)
    lp_values: List[Optional[float]] = field(default_factory=list)  # LP relaxation value per instance
    lp_closed: List[Optional[float]] = field(default_factory=list)  # share of the root gap closed per instance
    # per seed: mean of work_ratios on the optimality instances (None = none solved by the base at
    # that seed); empty when the review has no optimality instances
    work_changes: List[Optional[float]] = field(default_factory=list)

    @property
    def work_change(self) -> Optional[Tuple[float, float]]:
        """(mean, sample std) over the seeds of the work change (None without a value)."""
        known: List[float] = [w for w in self.work_changes if w is not None]
        if not known:
            return None
        return statistics.mean(known), statistics.stdev(known) if len(known) > 1 else 0.0

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

    @property
    def lp_fitness(self) -> float:
        """% of the root gap closed, mean over the instances whose LP was solved (nan if none)."""
        closed: List[float] = [c for c in self.lp_closed if c is not None]
        return 100 * statistics.mean(closed) if closed else float("nan")

    def to_dict(self) -> Dict[str, object]:
        work: Optional[Tuple[float, float]] = self.work_change
        return {**asdict(self), "fitness_mean": self.mean, "fitness_std": self.std,
                "mean_gap": self.mean_gap, "lp_fitness": self.lp_fitness,
                "work_change_mean": work[0] if work else None, "work_change_std": work[1] if work else None}


def review_cuts(
    base: Formulation,
    cuts: Dict[str, str],
    evaluation: List[Instance],
    seeds: List[int],
    make_solver: SolverFactory,
    run_fitness: Optional[Dict[str, float]] = None,
    optimality: Optional[List[Instance]] = None,
) -> List[CutReview]:
    """
    Solve every evaluation instance with `base` and with each cut (`name -> code`) under each
    seed, then the LP relaxation of each model once (solver of the first seed). Returns one
    CutReview per model, the base model first. `optimality`: instances the base model solves to
    optimality under the work limit; each model is also solved on them under each seed, for
    CutReview.work_changes.
    """
    run_fitness = run_fitness or {}
    reviews: List[CutReview] = [CutReview(BASE_NAME, None, None)]
    reviews += [CutReview(name, code, run_fitness.get(name)) for name, code in cuts.items()]
    best_known: List[Optional[float]] = [None] * len(evaluation)  # cheapest base tour per instance
    for seed in seeds:
        solver: Solver = make_solver(seed)
        solver.add_formulation(f"review_{BASE_NAME}", base)
        for name, code in cuts.items():
            solver.add_formulation(f"review_{name}", functools.partial(with_cut, base=base, code=code))
        base_results: List[SolveResult] = [solver.solve(inst, f"review_{BASE_NAME}") for inst in evaluation]
        for i, r in enumerate(base_results):
            if r.obj_val is not None and (best_known[i] is None or r.obj_val < best_known[i]):
                best_known[i] = r.obj_val
        for review in reviews:
            results: List[SolveResult] = (
                base_results if review.code is None
                else [solver.solve(inst, f"review_{review.name}") for inst in evaluation]
            )
            review.scores.append(score(seed, evaluation, base_results, results))
        if optimality:
            base_opt: List[SolveResult] = [solver.solve(inst, f"review_{BASE_NAME}") for inst in optimality]
            for review in reviews:
                opt: List[SolveResult] = (
                    base_opt if review.code is None
                    else [solver.solve(inst, f"review_{review.name}") for inst in optimality]
                )
                ratios: List[float] = work_ratios(optimality, base_opt, opt, solver.work_limit)[0]
                review.work_changes.append(statistics.mean(ratios) if ratios else None)
    lp_review(base, reviews, evaluation, make_solver(seeds[0]), best_known)
    return reviews


def lp_review(
    base: Formulation, reviews: List[CutReview], evaluation: List[Instance], solver: Solver,
    best_known: List[Optional[float]],
) -> None:
    """Fill lp_values and lp_closed of every review (reviews[0] = the base model)."""
    solver.work_limit = None  # an LP must reach OPTIMAL to give its bound (the time limit still applies)
    for review in reviews:
        build: Formulation = base if review.code is None else functools.partial(with_cut, base=base, code=review.code)
        solver.add_formulation(f"review_lp_{review.name}", functools.partial(relaxed, build=build))
        for inst in evaluation:
            result: SolveResult = solver.solve(inst, f"review_lp_{review.name}")
            review.lp_values.append(result.obj_val if result.status == "OPTIMAL" else None)
    for review in reviews:
        for lp, lp_base, known in zip(review.lp_values, reviews[0].lp_values, best_known):
            ok: bool = lp is not None and lp_base is not None and known is not None and known > lp_base
            review.lp_closed.append((lp - lp_base) / (known - lp_base) if ok else None)


def score(seed: int, evaluation: List[Instance], base: List[SolveResult], cut: List[SolveResult],
          work: Optional[List[float]] = None) -> SeedScore:
    """The run's fitness formula, against the base results of the same seed. `work`: the
    work_ratios of the optimality instances (fitness mode gap_work), pooled with the gap ratios
    (each instance counts the same): fitness = 10 * exp(-mean of all the ratios)."""
    gaps: List[float] = [r.mip_gap if r.mip_gap is not None else NO_SOLUTION_GAP for r in cut]
    ratios: List[float] = []
    skipped: List[str] = []
    for inst, b, gap in zip(evaluation, base, gaps):
        if b.mip_gap is None or b.mip_gap <= 0:  # solved by the base at this seed: no ratio
            skipped.append(inst["name"])
            continue
        ratios.append((gap - b.mip_gap) / b.mip_gap)
    ratios += work or []
    fitness: float = 10 * math.exp(-statistics.mean(ratios)) if ratios else float("nan")
    return SeedScore(seed, fitness, statistics.mean(gaps), gaps, [r.status for r in cut], skipped)


def work_ratios(optimality: List[Instance], base: List[SolveResult], cut: List[SolveResult],
                work_limit: Optional[float]) -> Tuple[List[float], List[str]]:
    """
    (ratios, skipped): the relative change of the work needed to PROVE optimality,
        (work_cut - work_base) / work_base,
    on each instance the base model solves to optimality under this seed (the others are
    skipped). A cut that does not prove optimality within the work limit counts
    work_cut = work_limit: its true work is higher, so a slowdown is underestimated, never
    overestimated. < 0 = the cut proves optimality faster.
    """
    ratios: List[float] = []
    skipped: List[str] = []
    for inst, b, c in zip(optimality, base, cut):
        if b.status != "OPTIMAL" or not b.work:
            skipped.append(inst["name"])
            continue
        proved: bool = c.status == "OPTIMAL" and c.work is not None
        work: float = c.work if proved else (work_limit if work_limit is not None else c.work or 0.0)
        ratios.append((work - b.work) / b.work)
    return ratios, skipped


def paired_difference(best: List[float], seed: List[float]) -> Tuple[float, float]:
    """Mean and sample std (0 for one seed) over the seeds of best - seed, seed by seed: both cuts
    are solved under the same Gurobi seeds, so the per-seed difference removes the noise they share."""
    differences: List[float] = [b - s for b, s in zip(best, seed)]
    return statistics.mean(differences), statistics.stdev(differences) if len(differences) > 1 else 0.0


def format_review(reviews: List[CutReview]) -> str:
    """A table: fitness mean ± std over the seeds, min-max, mean gap, the run's fitness."""
    n: int = len(reviews[0].scores)
    lines: List[str] = [
        f"review of the cuts under {n} Gurobi seeds ({', '.join(str(s.seed) for s in reviews[0].scores)}); "
        f"± = sample std over these {n} seeds (same cut, same instances: solver noise):",
        f"  {'model':<8} {'fitness mean ± std':>20} {'min':>7} {'max':>7} {'mean gap':>9} "
        f"{'LP gap closed':>14} {'run fitness':>12}",
    ]
    for r in reviews:
        run: str = f"{r.run_fitness:.3f}" if r.run_fitness is not None else "-"
        lines.append(
            f"  {r.name:<8} {r.mean:>11.3f} ± {r.std:<6.3f} {min(r.fitnesses):>7.3f} {max(r.fitnesses):>7.3f} "
            f"{100 * r.mean_gap:>8.2f}% {r.lp_fitness:>13.2f}% {run:>12}"
        )
    by_name: Dict[str, CutReview] = {r.name: r for r in reviews}
    if "best" in by_name and "seed" in by_name:
        best, seed = by_name["best"], by_name["seed"]
        wins: int = sum(b > s for b, s in zip(best.fitnesses, seed.fitnesses))
        mean, std = paired_difference(best.fitnesses, seed.fitnesses)
        lines.append(
            f"  best - seed: {mean:+.3f} ± {std:.3f} (std over the seeds of the per-seed difference); "
            f"the best cut beats the seed cut on {wins}/{n} seeds; LP gap closed {best.lp_fitness:.2f}% "
            f"vs {seed.lp_fitness:.2f}%"
        )
    worked: List[str] = [f"{r.name} {100 * r.work_change[0]:+.1f}% ± {100 * r.work_change[1]:.1f}%"
                         for r in reviews if r.code is not None and r.work_change is not None]
    if worked:
        lines.append("  work to prove optimality on the optimality instances (mean change vs the base, "
                     "± std over the seeds; < 0 = faster): " + ", ".join(worked))
    statuses: set = {st for r in reviews for s in r.scores for st in s.statuses} - {"WORK_LIMIT", "OPTIMAL"}
    if statuses:
        lines.append(f"  note: some solves ended with {', '.join(sorted(statuses))}")
    return "\n".join(lines)
