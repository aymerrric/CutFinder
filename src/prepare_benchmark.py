"""
Build the public benchmark set of a problem: the paper's held-out test set (TSPLIB n < 250,
OR-Library CWLP, Taillard JSSP; loaders/benchmarks.py), in our instance format.
Run from src/:  python prepare_benchmark.py --problem_name tsp   (or --problem_name all)
Downloads once into data/benchmark/raw/<problem>/, writes data/benchmark/<problem>/benchmark.pkl
(an existing file is rebuilt: the downloads are kept), and prints each instance's base model size.
The mean model size is compared with the paper's Table 20: equal means same instances and same
formulation. No solve: the set does not depend on the CPU (benchmark_cut.py solves it).
"""
import argparse
import logging
import os
import statistics
from typing import List, Optional, Tuple

import gurobipy as gp

from loaders.base import Instance, Loader
from loaders.benchmarks import BENCHMARKS, Benchmark
from problems import PROBLEMS, Problem
from utils import parameters

RELATIVE_TOLERANCE: float = 0.005  # mean sizes within 0.5% of the paper's count as equal


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert the paper's public benchmark sets")
    parser.add_argument("--problem_name", type=str, choices=sorted(BENCHMARKS) + ["all"], default="all")
    parser.add_argument("--data_dir", type=str, default=parameters.BENCHMARK_DIR)
    parser.add_argument("--no_sizes", action="store_true",
                        help="skip building the base models (sizes); no Gurobi needed")
    return parser.parse_args(argv)


def model_size(problem: Problem, instance: Instance) -> Tuple[int, int]:
    """(variables, constraints) of the base model, before presolve."""
    model: gp.Model = problem.base(**instance["args"])
    try:
        model.update()
        return model.NumVars, model.NumConstrs
    finally:
        model.dispose()


def size_check(name: str, ours: float, paper: float) -> str:
    verdict: str = "same" if abs(ours - paper) <= RELATIVE_TOLERANCE * paper else "DIFFERENT"
    return f"mean {name}: {ours:.2f} (paper {paper:.2f}: {verdict})"


def prepare(problem_name: str, data_dir: str, sizes: bool = True) -> List[Instance]:
    benchmark: Benchmark = BENCHMARKS[problem_name]
    problem: Problem = PROBLEMS[problem_name]
    loader: Loader = problem.loader(data_dir)
    raw_dir: str = os.path.join(data_dir, "raw", problem_name)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(loader.folder, exist_ok=True)
    instances: List[Instance] = benchmark.fetch(raw_dir)
    loader.save(instances, parameters.BENCHMARK_FILE)
    print(f"{problem_name}: {len(instances)} instances ({benchmark.description}; paper: "
          f"{benchmark.paper_count}) -> {os.path.join(loader.folder, parameters.BENCHMARK_FILE)}")
    if len(instances) != benchmark.paper_count:
        print(f"  WARNING: the paper has {benchmark.paper_count} instances")
    if sizes:
        counts: List[Tuple[int, int]] = []
        for instance in instances:
            counts.append(model_size(problem, instance))
            optimum: str = f", optimum {instance['optimum']:g}" if "optimum" in instance else ""
            print(f"  {instance['name']:<12} {counts[-1][0]:>7} variables {counts[-1][1]:>7} constraints{optimum}")
        print("  " + size_check("variables", statistics.mean(v for v, _ in counts), benchmark.paper_variables))
        print("  " + size_check("constraints", statistics.mean(c for _, c in counts), benchmark.paper_constraints))
    return instances


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    names: List[str] = sorted(BENCHMARKS) if args.problem_name == "all" else [args.problem_name]
    for name in names:
        prepare(name, args.data_dir, sizes=not args.no_sizes)
