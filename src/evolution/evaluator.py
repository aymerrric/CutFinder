import functools
import logging
import math
from typing import List

from evolution.genetic import CutRejected
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP
from models.transforms import fixed, relaxed, with_cut
from solvers.solver import CRASHED, ERROR, KILLED, TOO_LARGE, Formulation, SolveResult, Solver

FAILED: List[str] = [ERROR, TOO_LARGE, KILLED, CRASHED]  # the solve did not reach the end
NO_SOLUTION: List[str] = ["INFEASIBLE", "INF_OR_UNBD"]
NO_SOLUTION_GAP: float = 1.0  # gap counted when the cut model finds no solution within the limit

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
    3. score: solve every evaluation instance with the cut, under the same work limit as its
       base result (the base model does not finish: its gap is > 0), and compare the gaps:
           fitness = 10 * exp(-mean_i (gap_cut_i - gap_base_i) / gap_base_i)
       10 = same gap as without the cut, more = smaller gap (all solved: 10 * e = 27.2).
       No solution found with the cut counts as a gap of NO_SOLUTION_GAP (fitness near 0).
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
    ) -> None:
        """
        validation: instances with instance[BASE] (optimal, with solution) and instance[BASE_LP].
        evaluation: instances with instance[BASE] stopped with a gap > 0 (not solved), under the
                    same settings as `solver` (work limit, threads), so that gaps compare.
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
        self.n_evaluated: int = 0  # also names the formulations of each cut: cut1_valid, ...

    def __call__(self, code: str) -> float:
        self.n_evaluated += 1
        tag: str = f"cut{self.n_evaluated}"
        build: Formulation = functools.partial(with_cut, base=self.base, code=code)
        self._check_validity(tag, build)
        self._check_usefulness(tag, build)
        return self._score(tag, build)

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

    def _score(self, tag: str, build: Formulation) -> float:
        ratios: List[float] = []
        runtime: float = 0.0
        for inst in self.evaluation:
            result: SolveResult = self._solve(inst, f"{tag}_eval", build)
            base: SolveResult = inst[BASE]
            if result.status in NO_SOLUTION:
                raise CutRejected(f"invalid cut: instance {inst['name']} becomes infeasible")
            if (
                result.obj_bound is not None
                and base.obj_val is not None
                and result.obj_bound > base.obj_val + 1e-6 * max(1.0, abs(base.obj_val))
            ):
                raise CutRejected(
                    f"invalid cut: on instance {inst['name']} its lower bound {result.obj_bound:.6f} "
                    f"is above the cost {base.obj_val:.6f} of a known solution, so it removes every "
                    "solution at least that good, the optimum included"
                )
            gap: float = result.mip_gap if result.mip_gap is not None else NO_SOLUTION_GAP
            ratios.append((gap - base.mip_gap) / base.mip_gap)
            runtime += result.runtime or 0.0
        mean_ratio: float = sum(ratios) / len(ratios)
        fitness: float = 10 * math.exp(-mean_ratio)
        logger.info(
            f"{tag}: fitness {fitness:.3f}, gap {100 * mean_ratio:+.1f}% vs base on average, "
            f"{runtime:.1f}s of solving on {result.machine} ({result.cpu}, {result.threads} threads)"
        )
        return fitness

    def _solve(self, inst: Instance, name: str, build: Formulation) -> SolveResult:
        """Solve `inst` with the formulation `build` (registered as `name`); failures -> CutRejected."""
        self.solver.add_formulation(name, build)
        result: SolveResult = self.solver.solve(inst, name)
        if result.status in FAILED:
            raise CutRejected(f"{result.status} on instance {inst['name']}: {result.error}")
        return result
