import functools
import logging
import statistics
from typing import Dict, List, Optional, Tuple

from evolution.genetic import CutRejected
from evolution.review import SeedScore, score
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP
from models.transforms import fixed, relaxed, with_cut
from solvers.solver import CRASHED, ERROR, KILLED, TOO_LARGE, Formulation, SolveResult, Solver

FAILED: List[str] = [ERROR, TOO_LARGE, KILLED, CRASHED]  # the solve did not reach the end
NO_SOLUTION: List[str] = ["INFEASIBLE", "INF_OR_UNBD"]
TIME_LIMIT: str = "TIME_LIMIT"  # stopped by the time limit before the work limit
BASE_SEED: str = "base_seed"  # formulation of the base model solved under an extra scoring seed

logger: logging.Logger = logging.getLogger(__name__)


class CutEvaluator:
    """
    The `evaluate(code) -> fitness` of the genetic algorithm. Follows the reference
    (CutVerifier.verify_score_cut and cal_fitness), with a work limit instead of a time limit:

    1. validity: on every validation instance, the model with the cut, with its variables
       fixed to the base model's optimal solution, must stay feasible (the cut keeps the
       optimal solution);
    2. usefulness: on at least one validation instance, the LP relaxation with the cut, with
       its variables fixed to the base LP optimum, must be infeasible (the cut removes the
       LP optimum, so it tightens the relaxation);
    3. score: under each Gurobi seed s of the scoring solvers, solve every evaluation instance
       with the cut and with the base model, under the same work limit, and compare the gaps:
           fitness_s = 10 * exp(-mean_i (gap_cut_i,s - gap_base_i,s) / gap_base_i,s)
           fitness   = mean_s fitness_s
       (review.score: the end-of-run review uses the same formula). 10 = same gap as without the
       cut, more = smaller gap (all solved: 10 * e = 27.2). No solution found with the cut counts
       as a gap of review.NO_SOLUTION_GAP (fitness near 0). Instances the base model solves at
       seed s are left out of fitness_s. Several seeds average out the luck of one search path
       (performance variability: learnt_results.md 1.2).
       A lower bound above the cost of a tour the base model found means an invalid cut
       (it removed every tour at least that good, the optimum included). Minimization assumed.

    Any failure raises CutRejected with a reason for the LLM. Every solve goes through the
    Solver: the LLM code runs in its child process, with its timeout and size cap.
    """

    def __init__(
        self,
        base: Formulation,
        validation: List[Instance],
        evaluation: List[Instance],
        solver: Solver,
        scoring_solvers: Optional[List[Solver]] = None,
    ) -> None:
        """
        validation: instances with instance[BASE] (optimal, with solution) and instance[BASE_LP].
        evaluation: instances with instance[BASE] stopped with a gap > 0 (not solved), under the
                    same settings as `solver` (work limit, threads), so that gaps compare.
        solver: runs the validity and usefulness checks.
        scoring_solvers: one solver per Gurobi seed (SeedMode.SET), same settings as `solver`
                         except the seed; the fitness is the mean over them. None = [solver].
                         The base model is solved here under every seed that instance[BASE]
                         was not solved with (8 solves per new seed).
        """
        for inst in validation:
            for key in (BASE, BASE_LP):
                if key not in inst or inst[key].solution is None:
                    raise ValueError(f"validation instance {inst['name']} has no {key} solution")
        for inst in evaluation:
            gap: object = inst[BASE].mip_gap if BASE in inst else None
            if not (isinstance(gap, float) and gap > 0):
                raise ValueError(
                    f"evaluation instance {inst['name']} needs a base gap > 0, got {gap}: build the "
                    "set with prepare_instances.py (it keeps only instances the base cannot finish)"
                )
        self.base: Formulation = base
        self.validation: List[Instance] = validation
        self.evaluation: List[Instance] = evaluation
        self.solver: Solver = solver
        self.scoring_solvers: List[Solver] = scoring_solvers or [solver]
        self.n_evaluated: int = 0  # also names the formulations of each cut: cut1_valid, ...
        # Gurobi seed -> the base model's result on each evaluation instance under that seed
        self.base_results: Dict[int, List[SolveResult]] = {
            s.seed: self._base_results(s) for s in self.scoring_solvers
        }
        # cheapest tour known for each evaluation instance: a cut's lower bound must stay below it
        self.best_known: Dict[str, float] = {
            inst["name"]: min(
                [inst[BASE].obj_val] + [r[i].obj_val for r in self.base_results.values() if r[i].obj_val is not None]
            )
            for i, inst in enumerate(evaluation)
        }

    def __call__(self, code: str) -> float:
        tag, build = self._new_cut(code)
        self._check_validity(tag, build)
        self._check_usefulness(tag, build)
        return self._score(tag, build)

    def check(self, code: str) -> None:
        """Only the cheap checks (validity, usefulness: small instances, ~0.4 s): raises
        CutRejected or returns. Used by prompt_bench.py to compare prompts without scoring."""
        tag, build = self._new_cut(code)
        self._check_validity(tag, build)
        self._check_usefulness(tag, build)

    def _new_cut(self, code: str) -> Tuple[str, Formulation]:
        self.n_evaluated += 1
        return f"cut{self.n_evaluated}", functools.partial(with_cut, base=self.base, code=code)

    def _check_validity(self, tag: str, build: Formulation) -> None:
        for inst in self.validation:
            result: SolveResult = self._solve(
                inst, f"{tag}_valid", functools.partial(fixed, build=build, solution=inst[BASE].solution)
            )
            if result.status in NO_SOLUTION:
                raise CutRejected(
                    f"invalid cut: it removes the optimal solution of instance {inst['name']}"
                )
            if result.status != "OPTIMAL":
                raise CutRejected(f"could not check validity on {inst['name']}: {result.status}")

    def _check_usefulness(self, tag: str, build: Formulation) -> None:
        lp: Formulation = functools.partial(relaxed, build=build)
        for inst in self.validation:
            result: SolveResult = self._solve(
                inst, f"{tag}_useful", functools.partial(fixed, build=lp, solution=inst[BASE_LP].solution)
            )
            if result.status in NO_SOLUTION:
                return  # the cut removes this LP optimum: useful
        raise CutRejected(
            "useless cut: the LP relaxation optimum of every validation instance still satisfies "
            "it, so it does not tighten the relaxation"
        )

    def _base_results(self, solver: Solver) -> List[SolveResult]:
        """The base model under `solver`'s seed: instance[BASE] if it was solved with it, else solved now."""
        if all(inst[BASE].seed == solver.seed for inst in self.evaluation):
            return [inst[BASE] for inst in self.evaluation]
        logger.info(f"solving the base model under the scoring seed {solver.seed}")
        solver.add_formulation(BASE_SEED, self.base)
        results: List[SolveResult] = [solver.solve(inst, BASE_SEED) for inst in self.evaluation]
        for inst, result in zip(self.evaluation, results):
            if result.status in FAILED or result.status == TIME_LIMIT:
                raise RuntimeError(f"base model on {inst['name']} under seed {solver.seed}: {result.status}")
        if not any(r.mip_gap is not None and r.mip_gap > 0 for r in results):
            raise ValueError(f"the base model solves every evaluation instance under seed {solver.seed}")
        return results

    def _score(self, tag: str, build: Formulation) -> float:
        scores: List[SeedScore] = []
        runtime: float = 0.0
        for solver in self.scoring_solvers:
            results: List[SolveResult] = []
            for inst in self.evaluation:
                result: SolveResult = self._solve(inst, f"{tag}_eval", build, solver)
                if result.status in NO_SOLUTION:
                    raise CutRejected(f"invalid cut: instance {inst['name']} becomes infeasible")
                known: float = self.best_known[inst["name"]]
                if result.obj_bound is not None and result.obj_bound > known + 1e-6 * max(1.0, abs(known)):
                    raise CutRejected(
                        f"invalid cut: on instance {inst['name']} its lower bound {result.obj_bound:.6f} "
                        f"is above the cost {known:.6f} of a known solution, so it removes every "
                        "solution at least that good, the optimum included"
                    )
                results.append(result)
                runtime += result.runtime or 0.0
            scores.append(score(solver.seed, self.evaluation, self.base_results[solver.seed], results))
        fitness: float = statistics.mean(s.fitness for s in scores)
        per_seed: str = ", ".join(f"{s.fitness:.3f} (seed {s.seed})" for s in scores)
        logger.info(
            f"{tag}: fitness {fitness:.3f} = mean of {per_seed}, "
            f"{runtime:.1f}s of solving on {result.machine} ({result.cpu}, {result.threads} threads)"
        )
        return fitness

    def _solve(
        self, inst: Instance, name: str, build: Formulation, solver: Optional[Solver] = None
    ) -> SolveResult:
        """Solve `inst` with the formulation `build` (registered as `name`) on `solver` (default:
        the checks' solver); failures -> CutRejected."""
        solver = solver or self.solver
        solver.add_formulation(name, build)
        result: SolveResult = solver.solve(inst, name)
        if result.status in FAILED:
            raise CutRejected(f"{result.status} on instance {inst['name']}: {result.error}")
        if result.status == TIME_LIMIT:  # how far a solve gets in some seconds depends on the machine
            raise CutRejected(
                f"too slow: on instance {inst['name']} the solve hit the {solver.time_limit:g} s "
                "time limit before the work limit (the model with the cut is too large or too slow)"
            )
        return result
