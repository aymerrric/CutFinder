import functools
import logging
import statistics
from enum import Enum
from typing import Dict, List, Optional, Tuple

from evolution.genetic import CutRejected
from evolution.review import SeedScore, score, work_ratios
from loaders.base import Instance
from loaders.sets import BASE, BASE_LP
from models.transforms import cut_violations, fixed, read_violations, relaxed, with_cut
from solvers.solver import CRASHED, ERROR, KILLED, TOO_LARGE, Formulation, SolveResult, Solver

FAILED: List[str] = [ERROR, TOO_LARGE, KILLED, CRASHED]  # the solve did not reach the end
NO_SOLUTION: List[str] = ["INFEASIBLE", "INF_OR_UNBD"]
TIME_LIMIT: str = "TIME_LIMIT"  # stopped by the time limit before the work limit
BASE_SEED: str = "base_seed"  # formulation of the base model solved under an extra scoring seed
BASE_LP_EVAL: str = "base_lp_eval"  # formulation of the base model's LP relaxation (LP mode)

logger: logging.Logger = logging.getLogger(__name__)


class FitnessMode(Enum):
    """How a valid and useful cut is scored (step 3 of CutEvaluator)."""

    GAP = "gap"  # MIP gap reached under the work limit, averaged over the scoring seeds (slow)
    LP = "lp"  # share of the root gap closed by the cut's LP relaxation (deterministic, fast)
    # GAP, plus the work needed to PROVE optimality on instances the base model solves (slower)
    GAP_WORK = "gap_work"

    @property
    def solves_mip(self) -> bool:
        """The modes that solve the MIP under the scoring seeds (GAP and GAP_WORK)."""
        return self is not FitnessMode.LP


MAX_SHOWN_NUMBERS: int = 250  # the data of a refused instance is shown to the LLM up to this many numbers
MAX_SHOWN_VIOLATIONS: int = 5  # violated constraints listed in the refusal message
DIAGNOSIS_TIME_LIMIT: float = 60  # seconds for the diagnosis of a refused cut (an LP-sized solve)


def data_size(value: object) -> int:
    """How many numbers an instance's data holds (dict keys that are tuples of indices excluded)."""
    if isinstance(value, dict):
        return sum(data_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(data_size(v) for v in value)
    return 1


def _short(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_short(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    return repr(value)


def describe_data(inst: Instance, max_numbers: int = MAX_SHOWN_NUMBERS) -> str:
    """The instance's data as the cut code sees it (the arguments of the base model), when small."""
    size: int = data_size(inst["args"])
    if size > max_numbers:
        return ""
    lines: str = "\n".join(f"  {name} = {_short(value)}" for name, value in inst["args"].items())
    return f"The data of {inst['name']} (the names your code receives):\n{lines}"


class CutEvaluator:
    """
    The `evaluate(code) -> fitness` of the genetic algorithm. Follows the reference
    (CutVerifier.verify_score_cut and cal_fitness), with a work limit instead of a time limit:

    1. validity: on every validation instance and every validity instance (prepare_validity.py:
       small random, TIGHT and public instances), the model with the cut, with its variables
       fixed to the base model's optimal solution, must stay feasible (the cut keeps the optimal
       solution). In GAP mode, also: the LP relaxation with the cut of every (large) evaluation
       instance must stay at or below a known solution's cost. Refusals for invalidity raise
       CutRejected(invalid=True): the LLM then gets the invalid-cut retry message;
    2. usefulness: on at least one validation instance, the LP relaxation with the cut, with
       its variables fixed to the base LP optimum, must be infeasible (the cut removes the
       LP optimum, so it tightens the relaxation);
    3. score, depending on `mode`:
       FitnessMode.GAP: under each Gurobi seed s of the scoring solvers, solve every evaluation instance
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
       FitnessMode.GAP_WORK: GAP, plus a few `optimality` instances that the base model PROVES
       optimal under the work limit (none of them in the evaluation set). On each, under each
       seed, the ratio is the relative change of the work needed to prove optimality,
       (work_cut - work_base) / work_base (review.work_ratios; not proved = the work limit), and
       it is pooled with the gap ratios of the evaluation instances:
           fitness_s = 10 * exp(-mean over the evaluation AND optimality instances of their ratio)
       A cut that closes gaps on the large instances but slows the proof down on the smaller
       ones is penalized (on ta01-ta04 our JSSP cut is still at a 0.3-9% gap at 2000 s while
       the base model proves the optimum in 7-12 min: TODO.md P1.3). Their optimum is known,
       so a cut whose bound (or LP value) goes above it is refused as invalid.
       FitnessMode.LP: solve the LP relaxation of every evaluation instance with the cut, and
       measure how much of the root gap it closes:
           fitness = 100 * mean_i (LP_cut_i - LP_base_i) / (UB_i - LP_base_i)
       where LP_base_i is the base model's LP value and UB_i the cost of the best tour known
       (found by the base model). 0 = the relaxation is not tightened, 100 = the LP reaches the
       best known tour. An LP value above UB_i means an invalid cut. The LP is deterministic: one
       solve per instance, no seed, much cheaper than GAP. But a tighter LP does not always mean a
       faster solve (Gurobi adds its own cuts), and conversely (learnt_results.md 1.7).

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
        mode: FitnessMode = FitnessMode.GAP,
        validity: Optional[List[Instance]] = None,
        optimality: Optional[List[Instance]] = None,
    ) -> None:
        """
        validation: instances with instance[BASE] (optimal, with solution) and instance[BASE_LP].
        evaluation: instances with instance[BASE] stopped with a gap > 0 (not solved), under the
                    same settings as `solver` (work limit, threads), so that gaps compare.
        solver: runs the validity and usefulness checks.
        scoring_solvers: one solver per Gurobi seed (SeedMode.SET), same settings as `solver`
                         except the seed; the fitness is the mean over them. None = [solver].
                         The base model is solved here under every seed that instance[BASE]
                         was not solved with (8 solves per new seed). Ignored in LP mode.
        mode: how a cut is scored (GAP or LP, see the class docstring).
        validity: more instances solved to optimality (prepare_validity.py: tight and public ones,
                  each with a "note"): a cut must keep each optimal solution too.
        optimality: GAP_WORK mode only (required there): instances with instance[BASE] OPTIMAL
                    under the same settings as `solver` (prepare_instances.py --optimality).
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
        for inst in validity or []:
            if BASE not in inst or inst[BASE].solution is None:
                raise ValueError(f"validity instance {inst['name']} has no {BASE} solution")
        if mode is FitnessMode.GAP_WORK and not optimality:
            raise ValueError("the gap_work fitness needs optimality instances (prepare_instances.py --optimality)")
        for inst in optimality or []:
            if BASE not in inst or inst[BASE].status != "OPTIMAL":
                raise ValueError(f"optimality instance {inst['name']} is not solved to optimality by the base model")
        self.base: Formulation = base
        self.validation: List[Instance] = validation
        # smallest first: a refusal then names the simplest counterexample, small enough to be shown
        self.validity: List[Instance] = sorted(validity or [], key=lambda inst: data_size(inst["args"]))
        # diagnoses a refused cut: which of its constraints the optimal solution violates
        self.diagnose_solver: Solver = Solver(
            time_limit=DIAGNOSIS_TIME_LIMIT, threads=solver.threads, log_dir=solver.log_dir,
            isolated=solver.isolated, build_timeout=solver.build_timeout, keep_solution=True)
        self.evaluation: List[Instance] = evaluation
        self.optimality: List[Instance] = optimality if mode is FitnessMode.GAP_WORK and optimality else []
        self.solver: Solver = solver
        self.mode: FitnessMode = mode
        # the LP relaxation needs no seed: only the checks' solver
        self.scoring_solvers: List[Solver] = [solver] if mode is FitnessMode.LP else scoring_solvers or [solver]
        self.n_evaluated: int = 0  # also names the formulations of each cut: cut1_valid, ...
        # Gurobi seed -> the base model's result on each evaluation instance under that seed
        self.base_results: Dict[int, List[SolveResult]] = {
            s.seed: self._base_results(s) for s in self.scoring_solvers
        }
        # Gurobi seed -> the base model's result on each optimality instance (GAP_WORK mode)
        self.base_optimality: Dict[int, List[SolveResult]] = {
            s.seed: self._base_results(s, self.optimality) for s in self.scoring_solvers
        } if self.optimality else {}
        # cheapest tour known for each evaluation instance: a cut's lower bound must stay below it
        self.best_known: Dict[str, float] = {
            inst["name"]: min(
                [inst[BASE].obj_val] + [r[i].obj_val for r in self.base_results.values() if r[i].obj_val is not None]
            )
            for i, inst in enumerate(evaluation)
        }
        self.best_known.update({inst["name"]: inst[BASE].obj_val for inst in self.optimality})  # the optimum
        # evaluation instance -> LP value of the base model (LP mode only)
        self.base_lp: Dict[str, float] = self._base_lp() if mode is FitnessMode.LP else {}

    def __call__(self, code: str) -> float:
        """The GA's Evaluator: same as evaluate."""
        return self.evaluate(code)

    def evaluate(self, code: str) -> float:
        """Check the cut (validity, usefulness), then score it according to `mode`."""
        tag, build = self._new_cut(code)
        self._check_validity(tag, build)
        self._check_usefulness(tag, build)
        if self.mode.solves_mip:
            self._check_lp_bound(tag, build)  # LP mode makes the same check while scoring
            return self._score_gap(tag, build)
        elif self.mode is FitnessMode.LP:
            return self._score_lp(tag, build)
        raise ValueError(f"unknown fitness mode {self.mode}")

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
        """Every optimal solution of the validation and validity instances must stay feasible."""
        for inst in self.validation + self.validity:
            result: SolveResult = self._solve(
                inst, f"{tag}_valid", functools.partial(fixed, build=build, solution=inst[BASE].solution)
            )
            if result.status in NO_SOLUTION:
                note: str = f" ({inst['note']})" if inst.get("note") else ""
                details: List[str] = [part for part in (self._diagnose(tag, inst, build), describe_data(inst)) if part]
                raise CutRejected(
                    f"invalid cut: it removes the optimal solution (cost {inst[BASE].obj_val:g}) of instance "
                    f"{inst['name']}{note}." + "".join("\n" + part for part in details), invalid=True)
            if result.status != "OPTIMAL":
                raise CutRejected(f"could not check validity on {inst['name']}: {result.status}")

    def _diagnose(self, tag: str, inst: Instance, build: Formulation) -> str:
        """Which of the cut's constraints the optimal solution of `inst` violates, and by how much
        ("" if the diagnosis cannot tell, e.g. the cut removes it through a non-linear constraint)."""
        name: str = f"{tag}_why"
        self.diagnose_solver.add_formulation(name, functools.partial(
            cut_violations, build=build, base=self.base, solution=inst[BASE].solution))
        result: SolveResult = self.diagnose_solver.solve(inst, name)
        if result.status != "OPTIMAL" or not result.solution:
            return ""
        violated: Dict[str, float] = read_violations(result.solution)
        if not violated:
            return ""
        shown: List[str] = [f"{row} (by {amount:.6g})" for row, amount in list(violated.items())[:MAX_SHOWN_VIOLATIONS]]
        more: int = len(violated) - len(shown)
        return ("At that optimal solution, these constraints of your cut are violated: " + ", ".join(shown)
                + (f", and {more} more" if more > 0 else "") + ".")

    def _check_lp_bound(self, tag: str, build: Formulation) -> None:
        """On every (large) evaluation instance, the LP relaxation with the cut must stay at or
        below the cost of a known solution: one LP each, cheap, and it catches at full size the
        cuts that only removed nothing on the small instances (learnt_results.md 5.6)."""
        lp: Formulation = functools.partial(relaxed, build=build)
        for inst in self.evaluation + self.optimality:
            result: SolveResult = self._solve(inst, f"{tag}_lpcheck", lp)
            if result.status in NO_SOLUTION:
                raise CutRejected(f"invalid cut: the LP relaxation of instance {inst['name']} becomes infeasible",
                                  invalid=True)
            known: float = self.best_known[inst["name"]]
            if result.status == "OPTIMAL" and result.obj_val > known + 1e-6 * max(1.0, abs(known)):
                raise CutRejected(
                    f"invalid cut: on instance {inst['name']} its LP relaxation value {result.obj_val:.6f} "
                    f"is above the cost {known:.6f} of a known solution ({100 * (result.obj_val - known) / abs(known):.2f}% "
                    "above), so it removes every solution at least that good, the optimum included", invalid=True)

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

    def _base_results(self, solver: Solver, instances: Optional[List[Instance]] = None) -> List[SolveResult]:
        """The base model under `solver`'s seed on `instances` (default: the evaluation set):
        instance[BASE] if it was solved with it, else solved now."""
        optimality: bool = instances is not None
        instances = self.evaluation if instances is None else instances
        if all(inst[BASE].seed == solver.seed for inst in instances):
            return [inst[BASE] for inst in instances]
        logger.info(f"solving the base model under the scoring seed {solver.seed}"
                    + (" (optimality instances)" if optimality else ""))
        solver.add_formulation(BASE_SEED, self.base)
        results: List[SolveResult] = [solver.solve(inst, BASE_SEED) for inst in instances]
        for inst, result in zip(instances, results):
            if result.status in FAILED or result.status == TIME_LIMIT:
                raise RuntimeError(f"base model on {inst['name']} under seed {solver.seed}: {result.status}")
        if not optimality and not any(r.mip_gap is not None and r.mip_gap > 0 for r in results):
            raise ValueError(f"the base model solves every evaluation instance under seed {solver.seed}")
        return results

    def _base_lp(self) -> Dict[str, float]:
        """LP value of the base model on each evaluation instance (8 LP solves, a few seconds)."""
        self.solver.add_formulation(BASE_LP_EVAL, functools.partial(relaxed, build=self.base))
        values: Dict[str, float] = {}
        for inst in self.evaluation:
            result: SolveResult = self.solver.solve(inst, BASE_LP_EVAL)
            if result.status != "OPTIMAL":
                raise RuntimeError(f"base LP relaxation of {inst['name']}: {result.status} {result.error or ''}")
            values[inst["name"]] = result.obj_val
        return values

    def _score_lp(self, tag: str, build: Formulation) -> float:
        lp: Formulation = functools.partial(relaxed, build=build)
        closed: List[float] = []  # per evaluation instance: share of the root gap closed
        for inst in self.evaluation:
            result: SolveResult = self._solve(inst, f"{tag}_lp", lp)
            if result.status in NO_SOLUTION:
                raise CutRejected(f"invalid cut: the LP relaxation of instance {inst['name']} becomes infeasible",
                                  invalid=True)
            if result.status != "OPTIMAL":
                raise CutRejected(f"the LP relaxation of instance {inst['name']} did not finish: {result.status}")
            known: float = self.best_known[inst["name"]]
            if result.obj_val > known + 1e-6 * max(1.0, abs(known)):
                raise CutRejected(
                    f"invalid cut: on instance {inst['name']} its LP relaxation value {result.obj_val:.6f} "
                    f"is above the cost {known:.6f} of a known solution, so it removes every "
                    "solution at least that good, the optimum included", invalid=True
                )
            lp_base: float = self.base_lp[inst["name"]]
            closed.append((result.obj_val - lp_base) / (known - lp_base))
        fitness: float = 100 * statistics.mean(closed)
        logger.info(
            f"{tag}: fitness {fitness:.3f} (% of the root gap closed; per instance "
            f"{', '.join(f'{100 * c:.1f}' for c in closed)})"
        )
        return fitness

    def _score_gap(self, tag: str, build: Formulation) -> float:
        scores: List[SeedScore] = []
        runtime: float = 0.0
        for solver in self.scoring_solvers:
            results: List[SolveResult] = []
            for inst in self.evaluation:
                result: SolveResult = self._solve(inst, f"{tag}_eval", build, solver)
                if result.status in NO_SOLUTION:
                    raise CutRejected(f"invalid cut: instance {inst['name']} becomes infeasible", invalid=True)
                known: float = self.best_known[inst["name"]]
                if result.obj_bound is not None and result.obj_bound > known + 1e-6 * max(1.0, abs(known)):
                    raise CutRejected(
                        f"invalid cut: on instance {inst['name']} its lower bound {result.obj_bound:.6f} "
                        f"is above the cost {known:.6f} of a known solution, so it removes every "
                        "solution at least that good, the optimum included", invalid=True
                    )
                results.append(result)
                runtime += result.runtime or 0.0
            work: Optional[List[float]] = None
            if self.optimality:
                proofs: List[SolveResult] = [self._solve_optimality(tag, build, solver, inst) for inst in self.optimality]
                runtime += sum(r.runtime or 0.0 for r in proofs)
                work = work_ratios(self.optimality, self.base_optimality[solver.seed], proofs, solver.work_limit)[0]
            scores.append(score(solver.seed, self.evaluation, self.base_results[solver.seed], results, work))
        fitness: float = statistics.mean(s.fitness for s in scores)
        per_seed: str = ", ".join(f"{s.fitness:.3f} (seed {s.seed})" for s in scores)
        logger.info(
            f"{tag}: fitness {fitness:.3f} = mean of {per_seed}, "
            f"{runtime:.1f}s of solving on {result.machine} ({result.cpu}, {result.threads} threads)"
        )
        return fitness

    def _solve_optimality(self, tag: str, build: Formulation, solver: Solver, inst: Instance) -> SolveResult:
        """The cut on an optimality instance: its optimum is known, so a bound (or a proven value)
        above it means the cut removed it."""
        result: SolveResult = self._solve(inst, f"{tag}_opt", build, solver)
        optimum: float = inst[BASE].obj_val
        tolerance: float = 1e-6 * max(1.0, abs(optimum))
        if result.status in NO_SOLUTION:
            raise CutRejected(f"invalid cut: instance {inst['name']} becomes infeasible", invalid=True)
        if result.obj_bound is not None and result.obj_bound > optimum + tolerance:
            raise CutRejected(
                f"invalid cut: on instance {inst['name']} its lower bound {result.obj_bound:.6f} is above "
                f"the optimal cost {optimum:.6f}, so it removes the optimal solution", invalid=True)
        return result

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
