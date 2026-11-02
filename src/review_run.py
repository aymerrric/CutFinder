"""
Review the best cut of a saved run under several Gurobi seeds, against the seed cut and the base
model (the same review main.py does at the end of a run; no LLM needed).
Run from src/, on a machine that can use the run's instance sets:
    python review_run.py --population_file ../runs/<run>/population.json --data_dir ../data/a4000
Writes runs/<run>/review.json, updates runs/<run>/summary.md, and prints the table.
"""
import argparse
import logging
import os
import sys
from typing import List

from evolution.population import Population
from evolution.review import CutReview
from evolution.summary import write_summary
from loaders.base import Instance, Loader
from main import PROBLEMS, Problem, check_same_machine, final_review
from utils import parameters
from utils.parser import arg_parser
from utils.run_files import run_name_of


def review_run(args: argparse.Namespace) -> List[CutReview]:
    if not args.population_file or not os.path.exists(args.population_file):
        raise FileNotFoundError(f"--population_file: {args.population_file} does not exist")
    problem: Problem = PROBLEMS[args.problem_name]
    loader: Loader = problem.loader(args.data_dir)
    evaluation: List[Instance] = loader.load(parameters.EVALUATION_FILE)
    check_same_machine(evaluation, args.threads)
    population: Population = Population.load(args.population_file)
    args.run_name = run_name_of(args.population_file)  # also its Gurobi log folder
    run_dir: str = os.path.dirname(os.path.abspath(args.population_file))
    reviews: List[CutReview] = final_review(args, problem, evaluation, population, run_dir)
    write_summary(run_dir, population)
    return reviews


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("solvers.solver").setLevel(logging.WARNING)  # one line per solve otherwise
    review_run(arg_parser())
