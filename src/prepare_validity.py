"""
Build the validity instances of a problem (loaders/validity.py) and solve each one to optimality
with the base model, keeping its optimal solution: a valid cut must keep it (evolution/evaluator.py).
Optimality does not depend on the CPU, so this runs once, anywhere (e.g. this Mac), and the file
is copied to the lab (fleet.py copies data/validity/ with the benchmark sets).
Run from src/:  python prepare_validity.py [--problem_name jssp | all]
Writes data/validity/<problem>/validity.pkl; instances not solved to optimality within
VALIDITY_TIME_LIMIT, or infeasible (possible for the tight CWLP candidates), are left out; a public
instance whose optimum differs from the published one stops the script (a loader error).
"""
import argparse
import logging
import os
from typing import List, Optional

from loaders.base import Instance, Loader
from loaders.sets import BASE
from problems import PROBLEMS, Problem
from solvers.solver import SolveResult, Solver
from utils import parameters


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and solve the validity instances")
    parser.add_argument("--problem_name", type=str, choices=sorted(PROBLEMS) + ["all"], default="all")
    parser.add_argument("--data_dir", type=str, default=parameters.VALIDITY_DIR)
    parser.add_argument("--time_limit", type=float, default=parameters.VALIDITY_TIME_LIMIT)
    return parser.parse_args(argv)


def prepare(problem_name: str, data_dir: str, time_limit: float) -> List[Instance]:
    problem: Problem = PROBLEMS[problem_name]
    loader: Loader = problem.loader(data_dir)
    os.makedirs(loader.folder, exist_ok=True)
    solver: Solver = Solver(time_limit=time_limit, keep_solution=True)
    solver.add_formulation(BASE, problem.base)
    kept: List[Instance] = []
    for inst in problem.validity_instances(os.path.join(data_dir, "raw", problem_name)):
        result: SolveResult = solver.solve(inst, BASE)
        line: str = f"  {inst['name']:<22} {result.status:<11} obj {result.obj_val} {result.runtime or 0:7.1f}s  ({inst['note']})"
        if result.status != "OPTIMAL" or result.solution is None:
            print(line + "  -> left out", flush=True)
            continue
        if inst.get("optimum") is not None and abs(result.obj_val - inst["optimum"]) > 1e-6 * max(1.0, abs(inst["optimum"])):
            raise RuntimeError(f"{inst['name']}: optimum {result.obj_val}, published {inst['optimum']}: loader error?")
        inst[BASE] = result
        kept.append(inst)
        print(line, flush=True)
    loader.save(kept, parameters.VALIDITY_FILE)
    print(f"{problem_name}: {len(kept)} validity instances saved to {os.path.join(loader.folder, parameters.VALIDITY_FILE)}")
    return kept


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.WARNING)
    arguments: argparse.Namespace = parse_args()
    for name in sorted(PROBLEMS) if arguments.problem_name == "all" else [arguments.problem_name]:
        print(f"== {name}", flush=True)
        prepare(name, arguments.data_dir, arguments.time_limit)
