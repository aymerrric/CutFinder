"""
Build the validation and evaluation sets of the TSP and solve them once with the base model.
Run from src/:  python prepare_instances.py [--n_validation 2 --n_evaluation 8 ...]
Validation: small instances solved to optimality. Evaluation: large instances that the base
model cannot finish within WORK_LIMIT (base gap > MIN_BASE_GAP), so cuts are scored by the gap.
Existing files are loaded instead of rebuilt: delete data/tsp/validation.pkl or
data/tsp/evaluation.pkl to rebuild them. --data_dir: another folder (e.g. one per CPU type:
Gurobi work, so the evaluation set, depends on the CPU).
"""
import argparse
import functools
import logging
import os
from typing import Callable, List, Optional

from loaders.base import Instance
from loaders.sets import BASE, BASE_LP, build_set
from loaders.tsp import LoaderTSP
from models.transforms import relaxed
from models.tsp import create_model
from solvers.solver import SolveResult, Solver, machine_info
from utils import parameters


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and solve the validation / evaluation sets")
    parser.add_argument("--data_dir", type=str, default=parameters.DATA_DIR)
    parser.add_argument("--n_validation", type=int, default=parameters.N_VALIDATION)
    parser.add_argument("--validation_cities", type=int, default=parameters.VALIDATION_CITIES)
    parser.add_argument("--n_evaluation", type=int, default=parameters.N_EVALUATION)
    parser.add_argument("--evaluation_min_cities", type=int, default=parameters.EVALUATION_MIN_CITIES)
    parser.add_argument("--evaluation_max_cities", type=int, default=parameters.EVALUATION_MAX_CITIES)
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
    os.makedirs(os.path.join(args.data_dir, "tsp"), exist_ok=True)
    loader: LoaderTSP = LoaderTSP(data_dir=args.data_dir)

    # validation: optimal tour (validity check) and LP-relaxation optimum (usefulness check)
    validation_solver: Solver = Solver(time_limit=args.validation_time_limit, keep_solution=True)
    validation_solver.add_formulation(BASE, create_model)
    validation_solver.add_formulation(BASE_LP, functools.partial(relaxed, build=create_model))
    validation: List[Instance] = build_set(
        loader, parameters.VALIDATION_FILE, args.n_validation, parameters.VALIDATION_SEED,
        validation_solver, require_optimal=True, formulations=[BASE, BASE_LP],
        min_cities=args.validation_cities, max_cities=args.validation_cities,
    )

    # evaluation: same settings as the scoring of the cuts (work limit, threads); only
    # instances the base model cannot finish are kept
    evaluation_solver: Solver = Solver(time_limit=args.time_limit, work_limit=args.work_limit)
    evaluation_solver.add_formulation(BASE, create_model)
    evaluation: List[Instance] = build_set(
        loader, parameters.EVALUATION_FILE, args.n_evaluation, parameters.EVALUATION_SEED,
        evaluation_solver, require_optimal=False, accept=has_gap(args.min_base_gap),
        min_cities=args.evaluation_min_cities, max_cities=args.evaluation_max_cities,
    )

    host, cpu = machine_info()
    print(f"machine: {host} ({cpu}), {parameters.THREADS} threads")
    print(summary("validation", validation))
    print(summary("evaluation", evaluation))
