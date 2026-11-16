"""
Re-check saved runs with the current validity check: the best cut of each run must keep the stored
optimal solution of every validity instance (data/validity/<problem>/validity.pkl: tight and public
instances, the layer that refuses the invalid cuts of early November, learnt_results.md 5.6-5.8).
When the best cut fails, the run's other cuts are tried from the fittest down, and the first one
that passes is recorded as the run's best valid cut. Runs started before the check existed could
select an invalid cut; their review score is then not a result.

--public_seconds S adds a second layer to the best cut (learnt_results.md 5.12, 5.18): on every
public benchmark instance (data/benchmark/<problem>/benchmark.pkl) the model with the cut is solved
for S seconds; the cut is invalid there when the model is infeasible or when its proven bound is
above a solution already known (the best objective among the saved benchmark results of the base
model, the seed cut and the paper's cut: runs/benchmark/<problem>/tl2000/). Passing it is not a
proof: a longer solve could still push the bound above the known solution.

Writes runs/<run>/validity.json, read by evolution/report.py (a run whose best cut is invalid is
flagged and left out of the means). Needs no instance set of the lab: runs on the Mac.
Run from src/:
    python check_run_cuts.py                      # every run of ../runs with a population.json
    python check_run_cuts.py ../runs/<run> ...    # these runs only
    python check_run_cuts.py --problem_name cwlp  # the runs of one problem
    python check_run_cuts.py ../runs/<run> --public_seconds 20   # and the public-bound layer
"""
import argparse
import functools
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from evolution.individual import Individual, normalized_code
from evolution.population import Population
from loaders.base import Instance
from loaders.sets import BASE
from models.transforms import fixed, with_cut
from problems import PROBLEMS, Problem
from solvers.solver import SeedMode, SolveResult, Solver
from utils import parameters
from utils.run_files import POPULATION, STATUS

VALIDITY_RESULT: str = "validity.json"  # in runs/<run>/
MAX_CANDIDATES: int = 30  # cuts tried per run to find its best valid one
TRUSTED: List[str] = ["base", "seed", "paper_sage"]  # benchmark labels whose solutions are valid (5.6)
BOUND_TOLERANCE: float = 1e-6  # relative: a bound above known * (1 + tolerance) removes the known solution


def known_solutions(problem: str, benchmark_runs_dir: str = parameters.BENCHMARK_RUNS_DIR) -> Dict[str, float]:
    """Public instance -> the lowest objective among the saved benchmark solves of the TRUSTED
    labels (any CPU, any time limit): a solution every valid cut keeps."""
    known: Dict[str, float] = {}
    for label in TRUSTED:
        for path in glob.glob(os.path.join(benchmark_runs_dir, problem, "tl*", label, "*", "*.json")):
            with open(path) as f:
                value: Optional[float] = json.load(f).get("obj_val")
            name: str = os.path.basename(path)[:-len(".json")]
            if value is not None:
                known[name] = min(value, known.get(name, float("inf")))
    return known


def public_violations(problem: Problem, code: str, instances: List[Instance], known: Dict[str, float],
                      solver: Solver, tag: str) -> Dict[str, Any]:
    """The public-bound layer: {"checked", "invalid": ["<name> (<why>)"], "no_bound": [...]}. Only
    the instances with a known solution are solved."""
    solver.add_formulation(tag, functools.partial(with_cut, base=problem.base, code=code))
    invalid: List[str] = []
    no_bound: List[str] = []
    checked: int = 0
    for inst in instances:
        if inst["name"] not in known:
            continue
        result: SolveResult = solver.solve(inst, tag)
        best: float = known[inst["name"]]
        if result.status == "INFEASIBLE":
            invalid.append(f"{inst['name']} (infeasible)")
        elif result.obj_bound is None:
            no_bound.append(f"{inst['name']} ({result.status})")
            continue
        elif result.obj_bound > best * (1 + BOUND_TOLERANCE) + BOUND_TOLERANCE:
            invalid.append(f"{inst['name']} (bound {result.obj_bound:.6g} > known {best:.6g}, "
                           f"+{100 * (result.obj_bound / best - 1):.2f}%)")
        checked += 1
    return {"checked": checked, "invalid": invalid, "no_bound": no_bound}


def removed_optima(problem: Problem, code: str, validity: List[Instance], solver: Solver, tag: str) -> List[str]:
    """The validity instances whose stored optimal solution the cut removes ("<name> (<status>)");
    [] = the cut keeps every optimum (accepted by the check)."""
    build = functools.partial(with_cut, base=problem.base, code=code)
    removed: List[str] = []
    for inst in validity:
        name: str = f"{tag}_{inst['name']}"
        solver.add_formulation(name, functools.partial(fixed, build=build, solution=inst[BASE].solution))
        result: SolveResult = solver.solve(inst, name)
        if result.status != "OPTIMAL":
            removed.append(f"{inst['name']} ({result.status})")
    return removed


def ranked_cuts(population: Population) -> List[Tuple[int, Individual]]:
    """(generation, individual) of every evaluated cut of the run, fittest first, each code once
    (the first generation where it appears)."""
    seen: Dict[str, Tuple[int, Individual]] = {}
    for generation, individuals in enumerate(population.history + [population.current]):
        for ind in individuals:
            key: str = normalized_code(ind.code)
            if ind.evaluated and key not in seen:
                seen[key] = (generation, ind)
    return sorted(seen.values(), key=lambda item: -item[1].fitness)


def problem_of(run_dir: str) -> str:
    """The run's problem (status.json arguments; runs before 2026-11-02 are TSP runs)."""
    path: str = os.path.join(run_dir, STATUS)
    if not os.path.exists(path):
        return "tsp"
    with open(path) as f:
        return (json.load(f).get("arguments") or {}).get("problem_name") or "tsp"


def _entry(generation: int, ind: Individual, removed: List[str]) -> Dict[str, Any]:
    return {"id": ind.id, "generation": generation, "generator": ind.generator, "fitness": ind.fitness,
            "idea": ind.idea, "valid": not removed, "removed": removed}


def check_run(run_dir: str, validity: List[Instance], solver: Solver, validity_file: str,
              max_candidates: int = MAX_CANDIDATES, public: Optional[Tuple[List[Instance], Dict[str, float],
                                                                             Solver, float]] = None) -> Dict[str, Any]:
    """Check the run's best cut, then (if it fails) its next cuts until one passes; write and
    return runs/<run>/validity.json. `public`: (benchmark instances, known solutions, solver,
    seconds): also the public-bound layer on the best cut (result["public"])."""
    problem: Problem = PROBLEMS[problem_of(run_dir)]
    population: Population = Population.load(os.path.join(run_dir, POPULATION))
    best: Individual = population.best
    cuts: List[Tuple[int, Individual]] = ranked_cuts(population)
    generation: int = next((g for g, ind in cuts if normalized_code(ind.code) == normalized_code(best.code)), 0)
    removed: List[str] = removed_optima(problem, best.code, validity, solver, "best")
    best_valid: Optional[Dict[str, Any]] = _entry(generation, best, removed) if not removed else None
    checked: int = 1
    for k, (g, ind) in enumerate(cuts):
        if best_valid is not None or checked >= max_candidates:
            break
        if normalized_code(ind.code) == normalized_code(best.code):
            continue
        checked += 1
        other: List[str] = removed_optima(problem, ind.code, validity, solver, f"cut{k}")
        if not other:
            best_valid = _entry(g, ind, other)
    result: Dict[str, Any] = {
        "checked": time.strftime("%Y-%m-%d %H:%M"), "problem": problem.name,
        "validity_file": validity_file, "n_instances": len(validity),
        "best": _entry(generation, best, removed), "best_valid": best_valid, "cuts_checked": checked,
    }
    if public is not None:
        instances, known, public_solver, seconds = public
        result["public"] = {"seconds": seconds,
                            **public_violations(problem, best.code, instances, known, public_solver, "best_public")}
    with open(os.path.join(run_dir, VALIDITY_RESULT), "w") as f:
        json.dump(result, f, indent=1)
    return result


def is_valid(result: Dict[str, Any]) -> bool:
    """The best cut passed every layer recorded in a validity.json (validity instances, public bounds)."""
    return bool(result["best"]["valid"]) and not (result.get("public") or {}).get("invalid")


def describe(run: str, result: Dict[str, Any]) -> str:
    """One line per run: valid, or invalid with what it removes and the best valid cut found."""
    best: Dict[str, Any] = result["best"]
    public: Optional[Dict[str, Any]] = result.get("public")
    bounds: str = ""
    if public is not None:
        bounds = (f"; public bounds ({public['seconds']:g} s): INVALID on {len(public['invalid'])}/{public['checked']}: "
                  + ", ".join(public["invalid"]) if public["invalid"] else
                  f"; public bounds ({public['seconds']:g} s): valid on {public['checked']}")
    if best["valid"]:
        return f"{run}: best cut (fitness {best['fitness']:.3f}) valid on {result['n_instances']} instances" + bounds
    found: Optional[Dict[str, Any]] = result["best_valid"]
    valid: str = (f"best valid cut: fitness {found['fitness']:.3f} (generation {found['generation']}, {found['generator']})"
                  if found else f"no valid cut among the {result['cuts_checked']} fittest")
    return (f"{run}: best cut (fitness {best['fitness']:.3f}) INVALID, removes the optimum of "
            f"{len(best['removed'])}/{result['n_instances']}: {', '.join(best['removed'])}; {valid}" + bounds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-check the best cut of saved runs with the validity check")
    parser.add_argument("runs", nargs="*", help="run folders (default: every run of --runs_dir)")
    parser.add_argument("--runs_dir", type=str, default=parameters.RUNS_DIR)
    parser.add_argument("--validity_dir", type=str, default=parameters.VALIDITY_DIR)
    parser.add_argument("--problem_name", type=str, choices=sorted(PROBLEMS), default=None,
                        help="only the runs of this problem")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--public_seconds", type=float, default=0,
                        help="seconds per public benchmark instance for the public-bound layer (0 = off)")
    parser.add_argument("--benchmark_dir", type=str, default=parameters.BENCHMARK_DIR)
    parser.add_argument("--benchmark_runs_dir", type=str, default=parameters.BENCHMARK_RUNS_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> List[str]:
    args: argparse.Namespace = parse_args(argv)
    runs: List[str] = args.runs or sorted(
        os.path.join(args.runs_dir, name) for name in os.listdir(args.runs_dir)
        if os.path.exists(os.path.join(args.runs_dir, name, POPULATION)))
    validity_sets: Dict[str, List[Instance]] = {}
    lines: List[str] = []
    solver: Solver = Solver(time_limit=60, threads=args.threads, log_dir=None)
    for run_dir in runs:
        name: str = problem_of(run_dir)
        if args.problem_name and name != args.problem_name:
            continue
        if name not in validity_sets:
            validity_sets[name] = PROBLEMS[name].loader(args.validity_dir).load(parameters.VALIDITY_FILE)
        path: str = os.path.join(PROBLEMS[name].loader(args.validity_dir).folder, parameters.VALIDITY_FILE)
        public: Optional[Tuple[List[Instance], Dict[str, float], Solver, float]] = None
        if args.public_seconds > 0:
            public_solver: Solver = Solver(time_limit=args.public_seconds, threads=8, seed=0, seed_mode=SeedMode.SET,
                                           log_dir=None, build_timeout=parameters.BENCHMARK_BUILD_TIMEOUT)
            public = (PROBLEMS[name].loader(args.benchmark_dir).load(parameters.BENCHMARK_FILE),
                      known_solutions(name, args.benchmark_runs_dir), public_solver, args.public_seconds)
        line: str = describe(os.path.basename(os.path.normpath(run_dir)),
                             check_run(run_dir, validity_sets[name], solver, path, public=public))
        print(line, flush=True)
        lines.append(line)
    return lines


if __name__ == "__main__":  # required: the Solver's child processes import this file
    main()
