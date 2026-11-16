"""
Build the validation, evaluation and test sets of a problem and solve them once with its base model.
Run from src/:  python prepare_instances.py [--problem_name tsp --n_validation 2 --n_evaluation 8 ...]
The instance sizes of each set come from the problem (problems.py).
Validation: small instances solved to optimality. Evaluation: large instances that the base
model cannot finish within WORK_LIMIT (base gap > MIN_BASE_GAP), so cuts are scored by the gap.
Test: held-out instances built the same way from other seeds, for the end-of-run review only.
Existing files are loaded instead of rebuilt (so running it again only builds a missing set):
delete data/<problem>/<set>.pkl to rebuild it. --data_dir: another folder (e.g. one per CPU type:
Gurobi work, so the evaluation set, depends on the CPU). --instance_level hard: the evaluation and
test sets of larger instances (Problem.hard_sizes), in evaluation_hard.pkl and test_hard.pkl.
--optimality: also the optimality sets of the gap_work fitness (optimality.pkl, optimality_test.pkl):
instances the base model proves optimal with a work in parameters.OPTIMALITY_WORK.
"""
import argparse
import functools
import logging
import os
from typing import Callable, List, Optional, Tuple

from loaders.base import Instance
from loaders.sets import BASE, BASE_LP, build_set
from loaders.base import Loader
from models.transforms import relaxed
from problems import PROBLEMS, Problem, SetSpec, check_minimization, instance_set
from solvers.solver import SolveResult, Solver, machine_info
from utils import parameters


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and solve the validation / evaluation sets")
    parser.add_argument("--problem_name", type=str, choices=sorted(PROBLEMS), default=parameters.PROBLEM_NAME)
    parser.add_argument("--data_dir", type=str, default=parameters.DATA_DIR)
    parser.add_argument("--instance_level", type=str, choices=parameters.INSTANCE_LEVELS,
                        default=parameters.INSTANCE_LEVEL, help="sizes of the evaluation and test sets")
    parser.add_argument("--optimality", action="store_true",
                        help="also build the optimality sets of the gap_work fitness")
    parser.add_argument("--n_optimality", type=int, default=parameters.N_OPTIMALITY)
    parser.add_argument("--n_validation", type=int, default=parameters.N_VALIDATION)
    parser.add_argument("--n_evaluation", type=int, default=parameters.N_EVALUATION)
    parser.add_argument("--n_test", type=int, default=parameters.N_TEST)
    parser.add_argument("--min_base_gap", type=float, default=parameters.MIN_BASE_GAP)
    parser.add_argument("--validation_time_limit", type=float, default=parameters.VALIDATION_TIME_LIMIT)
    parser.add_argument("--time_limit", type=float, default=parameters.TIME_LIMIT)
    parser.add_argument("--work_limit", type=float, default=parameters.WORK_LIMIT)
    return parser.parse_args(argv)


def has_gap(min_gap: float) -> Callable[[Instance], bool]:
    """Accept an instance whose base solve stopped with a gap above `min_gap` (not solved)."""

    def accept(inst: Instance) -> bool:
        gap: Optional[float] = inst[BASE].mip_gap
        return gap is not None and gap > min_gap

    return accept


def proved_with_work(work_range: Tuple[float, float]) -> Callable[[Instance], bool]:
    """Accept an instance the base model solved to optimality with a work in `work_range`
    (an optimality instance of the gap_work fitness)."""

    def accept(inst: Instance) -> bool:
        result: SolveResult = inst[BASE]
        return result.status == "OPTIMAL" and result.work is not None and work_range[0] <= result.work <= work_range[1]

    return accept


def summary(title: str, instances: List[Instance]) -> str:
    lines: List[str] = [
        f"{title}:",
        f"  {'name':<17} {'status':<10} {'obj':>8} {'gap':>7} {'work':>8} {'time':>7} {'nodes':>6}",
    ]
    for inst in instances:
        r: SolveResult = inst[BASE]
        nan: float = float("nan")  # printed for values a failed solve does not have
        obj: float = r.obj_val if r.obj_val is not None else nan
        runtime: float = r.runtime if r.runtime is not None else nan
        nodes: float = r.node_count if r.node_count is not None else nan
        work: float = r.work if r.work is not None else nan
        gap: float = 100 * r.mip_gap if r.mip_gap is not None else nan
        lines.append(
            f"  {inst['name']:<17} {r.status:<10} {obj:8.4f} {gap:6.2f}% {work:8.4f} {runtime:6.3f}s "
            f"{nodes:6.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # required: the Solver's child processes import this file
    args: argparse.Namespace = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    problem: Problem = PROBLEMS[args.problem_name]
    loader: Loader = problem.loader(args.data_dir)
    os.makedirs(loader.folder, exist_ok=True)

    # validation: optimal tour (validity check) and LP-relaxation optimum (usefulness check)
    validation_solver: Solver = Solver(time_limit=args.validation_time_limit, keep_solution=True)
    validation_solver.add_formulation(BASE, problem.base)
    validation_solver.add_formulation(BASE_LP, functools.partial(relaxed, build=problem.base))
    validation: List[Instance] = build_set(
        loader, parameters.VALIDATION_FILE, args.n_validation, parameters.VALIDATION_SEED,
        validation_solver, require_optimal=True, formulations=[BASE, BASE_LP],
        **problem.validation_sizes,
    )
    check_minimization(problem, validation[0])

    # evaluation: same settings as the scoring of the cuts (work limit, threads); only
    # instances the base model cannot finish are kept
    evaluation_solver: Solver = Solver(time_limit=args.time_limit, work_limit=args.work_limit)
    evaluation_solver.add_formulation(BASE, problem.base)
    spec: SetSpec = instance_set(problem, "evaluation", args.instance_level)
    evaluation: List[Instance] = build_set(
        loader, spec.filename, args.n_evaluation, spec.seed,
        evaluation_solver, require_optimal=False, accept=has_gap(args.min_base_gap), **spec.sizes,
    )
    # test: held-out instances for the end-of-run review, never seen by the GA (other seeds)
    spec = instance_set(problem, "test", args.instance_level)
    test: List[Instance] = build_set(
        loader, spec.filename, args.n_test, spec.seed,
        evaluation_solver, require_optimal=False, accept=has_gap(args.min_base_gap), **spec.sizes,
    )

    optimality: List[List[Instance]] = []
    if args.optimality:
        if problem.optimality_sizes is None:
            raise ValueError(f"{problem.name} has no optimality sizes (Problem.optimality_sizes in problems.py)")
        for filename, seed in ((parameters.OPTIMALITY_FILE, parameters.OPTIMALITY_SEED),
                               (parameters.OPTIMALITY_TEST_FILE, parameters.OPTIMALITY_TEST_SEED)):
            optimality.append(build_set(
                loader, filename, args.n_optimality, seed, evaluation_solver, require_optimal=False,
                accept=proved_with_work(parameters.OPTIMALITY_WORK), **problem.optimality_sizes,
            ))

    host, cpu = machine_info()
    print(f"machine: {host} ({cpu}), {parameters.THREADS} threads")
    print(summary("validation", validation))
    print(summary(f"evaluation ({args.instance_level} level)", evaluation))
    print(summary(f"test ({args.instance_level} level)", test))
    for title, instances in zip(("optimality", "optimality test"), optimality):
        print(summary(title, instances))
