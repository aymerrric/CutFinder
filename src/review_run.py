"""
Review the best cut of a saved run on the held-out instances (test.pkl, or test_<level>.pkl for a
run of another --instance_level: give the run's level, it is checked) under several Gurobi seeds,
against the seed cut and the base model (the same review main.py does at the end of a run; no LLM
needed). The review solves the base model itself, so test.pkl needs no matching CPU.
Run from src/, on a machine that can use the run's instance sets:
    python review_run.py --population_file ../runs/<run>/population.json --data_dir ../data/a4000
Writes runs/<run>/review.json, updates runs/<run>/summary.md, and prints the table.
--review_level hard: the same review on the test set of another instance level (e.g. a cut of a
default-level run on test_hard.pkl, to compare it with the cuts of hard-level runs), written to
runs/<run>/review_hard.json; review.json and summary.md are kept.
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from evolution.population import Population
from evolution.review import CutReview
from evolution.summary import write_summary
from loaders.base import Instance, Loader
from main import final_review, load_optimality_set, load_test_set
from problems import PROBLEMS, Problem
from utils.parser import arg_parser
from utils import parameters
from utils.run_files import REVIEW, STATUS, run_name_of


def check_level(run_dir: str, level: str, fitness_mode: str = parameters.FITNESS_MODE) -> None:
    """Refuse to review a run on the test set of another instance level than the one it ran on,
    or with another fitness mode (a gap_work run is also reviewed on optimality_test.pkl): its
    status.json arguments (runs older than these options: default level, gap fitness)."""
    path: str = os.path.join(run_dir, STATUS)
    arguments: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as f:
            arguments = json.load(f).get("arguments") or {}
    recorded: str = arguments.get("instance_level") or parameters.INSTANCE_LEVEL
    if recorded != level:
        raise ValueError(f"the run was scored on the {recorded!r} instance level: pass --instance_level {recorded}")
    mode: str = arguments.get("fitness_mode") or parameters.FITNESS_MODE
    if mode != fitness_mode:
        raise ValueError(f"the run used the {mode!r} fitness: pass --fitness_mode {mode}")


def review_run(args: argparse.Namespace) -> List[CutReview]:
    if not args.population_file or not os.path.exists(args.population_file):
        raise FileNotFoundError(f"--population_file: {args.population_file} does not exist")
    problem: Problem = PROBLEMS[args.problem_name]
    run_dir: str = os.path.dirname(os.path.abspath(args.population_file))
    check_level(run_dir, args.instance_level, args.fitness_mode)
    other: bool = args.review_level is not None and args.review_level != args.instance_level
    if other:  # reviewed on another level's test set: final_review records that level
        args = argparse.Namespace(**{**vars(args), "instance_level": args.review_level})
    loader: Loader = problem.loader(args.data_dir)
    test: List[Instance] = load_test_set(loader, problem, args.instance_level)
    optimality: List[Instance] = (load_optimality_set(loader, parameters.OPTIMALITY_TEST_FILE)
                                  if args.fitness_mode == "gap_work" else [])
    population: Population = Population.load(args.population_file)
    args.run_name = run_name_of(args.population_file)  # also its Gurobi log folder
    filename: str = f"review_{args.instance_level}.json" if other else REVIEW
    reviews: List[CutReview] = final_review(args, problem, test, population, run_dir, optimality, filename)
    if not other:
        write_summary(run_dir, population)
    return reviews


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("solvers.solver").setLevel(logging.WARNING)  # one line per solve otherwise
    review_run(arg_parser())
