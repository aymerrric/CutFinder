"""
Does choosing a run's final cut on held-out instances beat taking its highest run fitness?
(learnt_results.md 12.14: every best cut scores ~16 during the run and ~13 on the test set, the
seed cut's level: the GA overfits its 8 evaluation instances.)

For each saved run: its `--top` distinct cuts by run fitness (the seed cut added), each scored
- on the selection set (selection.pkl: test sizes, other seeds; built here when missing, or by
  prepare_instances.py --selection) under the run's fitness seeds: the score a run could use to choose its final cut;
- on the test set (test.pkl) under the review seeds 0..review_seeds-1: the held-out truth.
No LLM needed. Writes runs/<run>/selection.json (one entry per cut: run fitness, selection and
test scores), then compares, over all the runs given, the test score of the cut each rule picks:
highest run fitness (what runs do now), highest selection score, and the best possible (oracle).
Run from src/, on a machine of the runs' instance group (the sets are per CPU type):
    python select_cuts.py --data_dir ../data/a5000 --population_file ../runs/<run>/population.json ...
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

from evolution.individual import SEED, Individual
from evolution.population import Population
from evolution.review import CutReview, review_cuts
from loaders.base import Instance, Loader
from loaders.sets import BASE, build_set
from main import load_test_set, make_solver
from prepare_instances import has_gap
from problems import PROBLEMS, Problem, SetSpec, instance_set
from solvers.solver import Solver
from utils import parameters
from utils.run_files import STATUS, run_name_of

SELECTION_JSON: str = "selection.json"
TOP: int = 5
LOCK_WAIT: float = 3 * 3600  # seconds a machine waits for another one to build the selection set


def top_cuts(population: Population, k: int) -> List[Individual]:
    """The `k` distinct cuts (by code) of highest run fitness, best first, then the seed cut if it
    is not among them. Ties keep the oldest, like Population.best."""
    seen: Dict[str, Individual] = {}
    for gen in population.history + [population.current]:
        for ind in gen:
            if ind.fitness is None:
                continue
            key: str = ind.code.strip()
            if key not in seen or ind.fitness > seen[key].fitness:
                seen[key] = ind
    ranked: List[Individual] = sorted(seen.values(), key=lambda ind: ind.fitness, reverse=True)[:k]
    first: List[Individual] = population.history[0] if population.history else population.current
    seed: Optional[Individual] = next((ind for ind in first if ind.generator == SEED), None)
    if seed is not None and all(ind.code.strip() != seed.code.strip() for ind in ranked):
        ranked.append(seed)
    return ranked


def run_arguments(run_dir: str) -> Dict[str, Any]:
    path: str = os.path.join(run_dir, STATUS)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("arguments") or {}


def selection_set(args: argparse.Namespace, problem: Problem, loader: Loader, level: str) -> List[Instance]:
    """The selection set, built here when missing (like prepare_instances.py --selection: same
    work limit, base gap > MIN_BASE_GAP). Several machines share the home: the first one takes a
    lock file and builds it, the others wait for the file (at most LOCK_WAIT seconds)."""
    spec: SetSpec = instance_set(problem, "selection", level)
    path: str = os.path.join(loader.folder, spec.filename)
    lock: str = path + ".lock"
    os.makedirs(loader.folder, exist_ok=True)
    waited: float = 0.0
    while not os.path.exists(path):
        try:
            fd: int = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:  # another machine is building it
            if waited >= LOCK_WAIT:
                raise TimeoutError(f"{lock} is still there after {LOCK_WAIT:.0f} s: delete it if no build runs")
            time.sleep(30)
            waited += 30
            continue
        try:
            os.close(fd)
            solver: Solver = Solver(time_limit=parameters.TIME_LIMIT, work_limit=args.work_limit, threads=args.threads)
            solver.add_formulation(BASE, problem.base)
            build_set(loader, spec.filename, args.n_selection, spec.seed, solver, require_optimal=False,
                      accept=has_gap(parameters.MIN_BASE_GAP), **spec.sizes)
        finally:
            os.remove(lock)
    return loader.load(spec.filename)


def name_of(rank: int, ind: Individual) -> str:
    return "seed" if ind.generator == SEED else f"top{rank + 1}"


def score_run(args: argparse.Namespace, population_file: str) -> Dict[str, Any]:
    """Score one run's top cuts on the selection and test sets; writes <run>/selection.json."""
    run_dir: str = os.path.dirname(os.path.abspath(population_file))
    recorded: Dict[str, Any] = run_arguments(run_dir)
    problem: Problem = PROBLEMS[recorded.get("problem_name") or args.problem_name]
    level: str = recorded.get("instance_level") or parameters.INSTANCE_LEVEL
    fitness_seeds: List[int] = recorded.get("fitness_seeds") or [parameters.SEED]
    loader: Loader = problem.loader(args.data_dir)
    selection_file: str = instance_set(problem, "selection", level).filename
    selection: List[Instance] = selection_set(args, problem, loader, level)
    test: List[Instance] = load_test_set(loader, problem, level)
    population: Population = Population.load(population_file)
    cuts: List[Individual] = top_cuts(population, args.top)
    named: Dict[str, Individual] = {name_of(i, ind): ind for i, ind in enumerate(cuts)}
    codes: Dict[str, str] = {name: ind.code for name, ind in named.items()}
    fitness: Dict[str, float] = {name: ind.fitness for name, ind in named.items()}
    solver_args = argparse.Namespace(**{**vars(args), "run_name": run_name_of(population_file)})
    factory = lambda s: make_solver(solver_args, seed=s)
    logging.getLogger(__name__).info(f"{run_dir}: {len(codes)} cuts on {len(selection)} selection and "
                                     f"{len(test)} test instances")
    on_selection: List[CutReview] = review_cuts(problem.base, codes, selection, fitness_seeds, factory, fitness)
    on_test: List[CutReview] = review_cuts(problem.base, codes, test, list(range(args.review_seeds)), factory, fitness)
    test_by_name: Dict[str, CutReview] = {r.name: r for r in on_test}
    result: Dict[str, Any] = {
        "run": os.path.basename(run_dir), "problem": problem.name, "instance_level": level,
        "selection_set": selection_file, "selection_seeds": fitness_seeds,
        "test_seeds": list(range(args.review_seeds)),
        "cuts": [{"name": r.name, "id": named[r.name].id, "generator": named[r.name].generator,
                  "run_fitness": fitness[r.name], "selection": r.mean, "selection_std": r.std,
                  "test": test_by_name[r.name].mean, "test_std": test_by_name[r.name].std}
                 for r in on_selection if r.name in named],
    }
    with open(os.path.join(run_dir, SELECTION_JSON), "w") as f:
        json.dump(result, f, indent=1)
    return result


def compare(results: List[Dict[str, Any]]) -> str:
    """The test score of the cut each rule picks, run by run, and the means."""
    lines: List[str] = ["| run | by run fitness | by selection set | oracle | seed cut |", "|---|---|---|---|---|"]
    picks: Dict[str, List[float]] = {"fitness": [], "selection": [], "oracle": [], "seed": []}
    for res in results:
        evolved: List[Dict[str, Any]] = [c for c in res["cuts"] if c["name"] != "seed"] or res["cuts"]
        by_fitness = max(evolved, key=lambda c: c["run_fitness"])
        by_selection = max(evolved, key=lambda c: c["selection"])
        oracle = max(evolved, key=lambda c: c["test"])
        seed = next((c for c in res["cuts"] if c["name"] == "seed"), None)
        for key, cut in (("fitness", by_fitness), ("selection", by_selection), ("oracle", oracle)):
            picks[key].append(cut["test"])
        if seed is not None:
            picks["seed"].append(seed["test"])
        seed_text: str = f"{seed['test']:.2f}" if seed is not None else "-"
        lines.append(f"| {res['run']} | {by_fitness['test']:.2f} ({by_fitness['name']}) | "
                     f"{by_selection['test']:.2f} ({by_selection['name']}) | {oracle['test']:.2f} ({oracle['name']}) | "
                     f"{seed_text} |")
    means = {k: (f"{statistics.mean(v):.2f}" if v else "-") for k, v in picks.items()}
    lines.append(f"| **mean** | **{means['fitness']}** | **{means['selection']}** | **{means['oracle']}** | {means['seed']} |")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--population_file", nargs="+", required=True)
    parser.add_argument("--problem_name", default=parameters.PROBLEM_NAME, help="for runs without status.json")
    parser.add_argument("--data_dir", default=parameters.DATA_DIR)
    parser.add_argument("--top", type=int, default=TOP)
    parser.add_argument("--n_selection", type=int, default=parameters.N_SELECTION, help="when the set is built here")
    parser.add_argument("--review_seeds", type=int, default=parameters.REVIEW_SEEDS)
    parser.add_argument("--work_limit", type=float, default=parameters.WORK_LIMIT)
    parser.add_argument("--threads", type=int, default=parameters.THREADS)
    parser.add_argument("--cut_time_limit", type=float, default=parameters.CUT_TIME_LIMIT)
    parser.add_argument("--cut_build_timeout", type=float, default=parameters.CUT_BUILD_TIMEOUT)
    parser.add_argument("--log_dir", default=parameters.LOG_DIR)
    parser.add_argument("--report", default=None, help="write the comparison table to this file too")
    parser.add_argument("--rescore", action="store_true", help="score runs that already have selection.json")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> str:
    results: List[Dict[str, Any]] = []
    for path in args.population_file:
        done: str = os.path.join(os.path.dirname(os.path.abspath(path)), SELECTION_JSON)
        if os.path.exists(done) and not args.rescore:
            with open(done) as f:
                results.append(json.load(f))
            continue
        results.append(score_run(args, path))
    table: str = compare(results)
    if args.report:
        with open(args.report, "w") as f:
            f.write(table + "\n")
    return table


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("solvers.solver").setLevel(logging.WARNING)
    print(main(parse_args()))
