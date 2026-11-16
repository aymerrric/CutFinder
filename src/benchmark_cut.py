"""
Score a cut on the paper's public benchmark set with the paper's protocol (evolution/benchmark.py):
every instance solved without the cut and with it, TimeLimit 2000 s, 8 threads, MIPGap 1e-5,
Method 4, the gap read at 5, 10, 50, 150, 300, 1000 and 2000 s.
Run from src/ (the set comes from prepare_benchmark.py):
    python benchmark_cut.py --problem_name cwlp --cut best --population_file ../runs/<run>/population.json
    python benchmark_cut.py --problem_name tsp --cut seed --shard 3/20     # this machine's 1/20
    python benchmark_cut.py --problem_name tsp --cut_file ../cuts/tsp/paper_sage.py
    python benchmark_cut.py --problem_name jssp --cut seed --cut_file ../cuts/jssp/paper_sage.py \
        --population_file ../runs/<run>/population.json                    # several cuts at once
    python benchmark_cut.py --problem_name tsp --report                    # table only, no solve
Several cuts in one run: on each instance the base is solved once, then every cut, one after the
other (the seed cut, the best cut of each --population_file, each --cut_file).
Results: runs/benchmark/<problem>/tl<T>/<label>/<cpu>/<instance>.json, one file per solve, written
as soon as it ends: a stopped run resumes where it was (existing files are not solved again).
The reference is <label> "base". The time limit is wall clock, so a cut is only compared with
the base solved on the same CPU model (<cpu> folder); on each instance the base is solved first,
right before the cut. All the benchmark is solved on ONE CPU model, parameters.BENCHMARK_CPU (the
lab's a5000 machines): run() refuses another CPU, and --report reads only that CPU's results, so
every instance of a table is solved on the same hardware (--any_cpu lifts both, for tests or a
second hardware study). --report writes runs/benchmark/<problem>/tl<T>/report.md.
"""
import argparse
import functools
import json
import logging
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

from evolution.benchmark import (
    NO_SOLUTION, BenchmarkRow, compare, format_table, lp_gap_closed, result_from_json, result_to_json, split_shard,
)
from evolution.population import Population
from loaders.base import Instance, Loader
from models.transforms import relaxed, with_cut
from problems import PROBLEMS, Problem
from solvers.solver import Formulation, SeedMode, SolveResult, Solver, machine_info
from utils import parameters
from utils.run_files import run_name_of

BASE_LABEL: str = "base"
CODE_FILE: str = "cut.py"  # the cut's code, next to its results
LP_DIR: str = "lp"  # <root>/lp/<label>/<instance>.json: LP relaxation (no CPU folder: the value does not depend on it)
# What the "(± x)" of a benchmark cell is: written above the benchmark tables of the report.
STD_NOTE: str = ("Each cell: mean Delta g (± sample std of Delta g across the n instances at that checkpoint). "
                 "Each instance is solved once per cut (Gurobi seed 0), so the ± is the spread between "
                 "instances, not solver noise and not the spread between runs or LLMs.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a cut on the paper's benchmark set")
    parser.add_argument("--problem_name", type=str, choices=sorted(PROBLEMS), default=parameters.PROBLEM_NAME)
    parser.add_argument("--cut", type=str, choices=["seed", "best"], default=None,
                        help="seed: the problem's seed cut; best: the best cut of each --population_file")
    parser.add_argument("--population_file", type=str, nargs="+", default=[],
                        help="runs whose best cut to score (label: the run's name)")
    parser.add_argument("--cut_file", type=str, nargs="+", default=[],
                        help="Python files holding a cut's code (label: the file's name)")
    parser.add_argument("--label", type=str, default=None, help="name of the cut in the results (one cut only)")
    parser.add_argument("--base_only", action="store_true", help="solve only the base model")
    parser.add_argument("--report", action="store_true", help="only build the table from the saved results")
    parser.add_argument("--instances", type=str, nargs="*", default=None, help="only these instances")
    parser.add_argument("--shard", type=str, default=None, help="i/n: every n-th instance, from the i-th")
    parser.add_argument("--time_limit", type=float, default=parameters.BENCHMARK_TIME_LIMIT)
    parser.add_argument("--threads", type=int, default=parameters.THREADS)
    parser.add_argument("--data_dir", type=str, default=parameters.BENCHMARK_DIR)
    parser.add_argument("--out_dir", type=str, default=parameters.BENCHMARK_RUNS_DIR)
    parser.add_argument("--lp", action="store_true",
                        help="solve the LP relaxations of the base and of every cut already in the results "
                             "(the report's LP gap closed column); any CPU, e.g. this Mac")
    parser.add_argument("--redo_without_work", action="store_true",
                        help="also solve again (base + every cut) the instances whose saved solves have no work "
                             "along the solve (before 2026-11-09); instances with a missing result come first")
    parser.add_argument("--any_cpu", action="store_true",
                        help="solve / report on any CPU (default: parameters.BENCHMARK_CPU only)")
    return parser.parse_args(argv)


def best_cut(population_file: str) -> str:
    """The code of the best cut of a run's population file."""
    if not os.path.exists(population_file):
        raise FileNotFoundError(f"no population file {population_file}")
    population: Population = Population.load(population_file)
    if population.best is None or population.best.code is None:
        raise ValueError(f"{population_file} has no best cut")
    return population.best.code


def cut_sources(args: argparse.Namespace, problem: Problem) -> List[Tuple[str, str]]:
    """(label, code) of every cut the arguments name: the seed cut, the best cut of each
    population file, then each cut file. Labels are folder names: unique, never "base"."""
    if args.cut == "best" and not args.population_file:
        raise FileNotFoundError("--cut best needs --population_file")
    cuts: List[Tuple[str, str]] = []
    if args.cut == "seed":
        cuts.append(("seed", problem.baseline_code))
    for path in args.population_file:
        cuts.append((run_name_of(path), best_cut(path)))
    for path in args.cut_file:
        with open(path) as f:
            cuts.append((os.path.splitext(os.path.basename(path))[0], f.read()))
    if not cuts:
        raise ValueError("name a cut: --cut seed, --population_file ..., --cut_file ..., or --base_only")
    if args.label:
        if len(cuts) > 1:
            raise ValueError("--label names one cut: give a single cut")
        cuts = [(args.label, cuts[0][1])]
    cuts = [(slug(label), code) for label, code in cuts]
    labels: List[str] = [label for label, _ in cuts]
    if BASE_LABEL in labels:
        raise ValueError(f"the label {BASE_LABEL!r} is the reference: choose another one")
    if len(set(labels)) < len(labels):
        raise ValueError(f"two cuts have the same label: {labels}")
    return cuts


def slug(text: str) -> str:
    """A folder name for a label or a CPU model ("Intel(R) Xeon(R) W-2245" -> "Intel-R-Xeon-R-W-2245")."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unknown"


def results_dir(out_dir: str, problem: str, time_limit: float) -> str:
    return os.path.join(out_dir, problem, f"tl{time_limit:g}")


def select(instances: List[Instance], names: Optional[List[str]], shard: Optional[str]) -> List[Instance]:
    if names:
        unknown: List[str] = sorted(set(names) - {inst["name"] for inst in instances})
        if unknown:
            raise ValueError(f"unknown instances: {unknown}")
        instances = [inst for inst in instances if inst["name"] in names]
    if shard:
        index, count = split_shard(shard)
        instances = instances[index - 1::count]  # the set is sorted by size: shards get a mix
    return instances


def make_solver(time_limit: float, threads: int, log_dir: Optional[str]) -> Solver:
    return Solver(
        time_limit=time_limit, threads=threads, seed=parameters.BENCHMARK_SEED, seed_mode=SeedMode.SET,
        log_dir=log_dir, build_timeout=parameters.BENCHMARK_BUILD_TIMEOUT,
        params=parameters.BENCHMARK_GUROBI_PARAMS, record_trajectory=True,
    )


def solve_once(solver: Solver, instance: Instance, formulation: str, path: str) -> SolveResult:
    """The saved result at `path`, else solve and save it (then the run can stop at any time)."""
    if os.path.exists(path):
        with open(path) as f:
            return result_from_json(json.load(f))
    result: SolveResult = solver.solve(instance, formulation)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w") as f:
        json.dump(result_to_json(result), f)
    os.replace(path + ".part", path)
    return result


def check_cpu(cpu: str, any_cpu: bool) -> None:
    """Refuse to solve on a CPU other than the benchmark's (gaps at a wall-clock time depend on it)."""
    if not any_cpu and cpu != parameters.BENCHMARK_CPU:
        raise RuntimeError(f"this machine's CPU is {cpu!r}, the benchmark is solved on {parameters.BENCHMARK_CPU!r} "
                           "(parameters.BENCHMARK_CPU): run it on that machine type, or --any_cpu")


def without_work(path: str) -> bool:
    """A saved solve whose trajectory has no work (recorded before 2026-11-09): its gap in work
    units is unknown. A solve without any trajectory point has nothing to redo."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        trajectory: Optional[List[List[float]]] = json.load(f).get("trajectory")
    return bool(trajectory) and len(trajectory[0]) < 4


def run(args: argparse.Namespace) -> None:
    problem: Problem = PROBLEMS[args.problem_name]
    loader: Loader = problem.loader(args.data_dir)
    instances: List[Instance] = select(loader.load(parameters.BENCHMARK_FILE), args.instances, args.shard)
    cuts: List[Tuple[str, str]] = [] if args.base_only else cut_sources(args, problem)
    root: str = results_dir(args.out_dir, problem.name, args.time_limit)
    check_cpu(machine_info()[1], args.any_cpu)
    cpu: str = slug(machine_info()[1])
    log_dir: str = os.path.join(parameters.LOG_DIR, "benchmark", problem.name)
    os.makedirs(log_dir, exist_ok=True)
    solver: Solver = make_solver(args.time_limit, args.threads, log_dir)
    solver.add_formulation(BASE_LABEL, problem.base)
    for label, code in cuts:
        os.makedirs(os.path.join(root, label), exist_ok=True)
        with open(os.path.join(root, label, CODE_FILE), "w") as f:
            f.write(code)
        solver.add_formulation(label, functools.partial(with_cut, base=problem.base, code=code))
    labels: List[str] = [BASE_LABEL, *(label for label, _ in cuts)]

    def path_of(label: str, name: str) -> str:
        return os.path.join(root, label, cpu, f"{name}.json")

    if args.redo_without_work:  # the instances with a missing result first, then the ones to redo
        instances = sorted(instances, key=lambda i: all(os.path.exists(path_of(l, i["name"])) for l in labels))
    print(f"{problem.name}: {len(instances)} instances, cuts {', '.join(l for l, _ in cuts) or '(none)'}, "
          f"{args.time_limit:g} s, {args.threads} threads, CPU {cpu}", flush=True)
    for k, instance in enumerate(instances, 1):
        name: str = instance["name"]
        if args.redo_without_work and any(without_work(path_of(l, name)) for l in labels):
            for label in labels:  # the base and every cut again, together: they stay comparable
                if os.path.exists(path_of(label, name)):
                    os.remove(path_of(label, name))
            print(f"[{k}/{len(instances)}] {name}: solved again (no work recorded along the solves)", flush=True)
        result: SolveResult = solve_once(solver, instance, BASE_LABEL, os.path.join(root, BASE_LABEL, cpu, f"{name}.json"))
        line: str = f"[{k}/{len(instances)}] {name}: base {result.status} gap {result.mip_gap}"
        for label, _ in cuts:
            result = solve_once(solver, instance, label, os.path.join(root, label, cpu, f"{name}.json"))
            line += f", {label} {result.status} gap {result.mip_gap}"
        print(line, flush=True)


def cut_codes(root: str) -> Dict[str, str]:
    """{label: code} of every cut of the results (<root>/<label>/cut.py)."""
    codes: Dict[str, str] = {}
    if os.path.isdir(root):
        for label in sorted(os.listdir(root)):
            path: str = os.path.join(root, label, CODE_FILE)
            if os.path.isfile(path):
                with open(path) as f:
                    codes[label] = f.read()
    return codes


def solve_lps(args: argparse.Namespace) -> None:
    """The LP relaxation of the base model and of base + each cut, on every instance (saved as soon
    as solved, never solved twice). Deterministic and CPU-independent: run anywhere."""
    problem: Problem = PROBLEMS[args.problem_name]
    instances: List[Instance] = select(problem.loader(args.data_dir).load(parameters.BENCHMARK_FILE),
                                       args.instances, args.shard)
    root: str = results_dir(args.out_dir, problem.name, args.time_limit)
    codes: Dict[str, str] = cut_codes(root)
    log_dir: str = os.path.join(parameters.LOG_DIR, "benchmark", problem.name, LP_DIR)
    os.makedirs(log_dir, exist_ok=True)
    solver: Solver = Solver(time_limit=parameters.BENCHMARK_LP_TIME_LIMIT, threads=args.threads,
                            seed=parameters.BENCHMARK_SEED, seed_mode=SeedMode.SET, log_dir=log_dir,
                            build_timeout=parameters.BENCHMARK_BUILD_TIMEOUT)
    solver.add_formulation(BASE_LABEL, functools.partial(relaxed, build=problem.base))
    for label, code in codes.items():
        solver.add_formulation(label, functools.partial(
            relaxed, build=functools.partial(with_cut, base=problem.base, code=code)))
    print(f"{problem.name}: LP relaxations of {len(instances)} instances, base + {', '.join(codes) or 'no cut'}",
          flush=True)
    for k, instance in enumerate(instances, 1):
        values: List[str] = []
        for label in [BASE_LABEL, *codes]:
            path: str = os.path.join(root, LP_DIR, label, f"{instance['name']}.json")
            result: SolveResult = solve_once(solver, instance, label, path)
            values.append(f"{label} {result.obj_val if result.status == 'OPTIMAL' else result.status}")
        print(f"[{k}/{len(instances)}] {instance['name']}: " + ", ".join(values), flush=True)


def lp_values(folder: str) -> Dict[str, Optional[float]]:
    """<root>/lp/<label>/ -> {instance: LP value, None if the LP was not solved to optimality}."""
    values: Dict[str, Optional[float]] = {}
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            if file.endswith(".json"):
                with open(os.path.join(folder, file)) as f:
                    result: SolveResult = result_from_json(json.load(f))
                values[file[:-len(".json")]] = result.obj_val if result.status == "OPTIMAL" else None
    return values


def upper_bounds(instances: List[Instance], base: Dict[str, Dict[str, SolveResult]]) -> Dict[str, float]:
    """Best known solution of each instance: the best one the base model found (any CPU: a solution
    is a solution), or the published optimum ("optimum", TSPLIB) if smaller."""
    upper: Dict[str, float] = {}
    for inst in instances:
        known: List[float] = [r.obj_val for results in base.values() for name, r in results.items()
                              if name == inst["name"] and r.obj_val is not None and abs(r.obj_val) < NO_SOLUTION]
        if inst.get("optimum") is not None:
            known.append(float(inst["optimum"]))
        if known:
            upper[inst["name"]] = min(known)
    return upper


def violations(root: str, label: str, upper: Dict[str, float]) -> Tuple[List[str], int]:
    """(instances where the cut is proven invalid, instances checked): its MIP bound (any CPU) or
    its LP relaxation is above a known solution, so it removed every solution at least that good."""
    bad: set = set()
    checked: set = set()
    for results in load_results(os.path.join(root, label)).values():
        for name, r in results.items():
            if name in upper and r.obj_bound is not None:
                checked.add(name)
                if r.obj_bound > upper[name] + 1e-6 * max(1.0, abs(upper[name])):
                    bad.add(name)
    for name, value in lp_values(os.path.join(root, LP_DIR, label)).items():
        if name in upper and value is not None:
            checked.add(name)
            if value > upper[name] + 1e-6 * max(1.0, abs(upper[name])):
                bad.add(name)
    return sorted(bad), len(checked)


def mean_lp_closed(root: str, label: str, upper: Dict[str, float]) -> Tuple[Optional[float], int]:
    """(mean LP gap closed by the cut, instances), over the instances with both LPs and a UB."""
    base: Dict[str, Optional[float]] = lp_values(os.path.join(root, LP_DIR, BASE_LABEL))
    cut: Dict[str, Optional[float]] = lp_values(os.path.join(root, LP_DIR, label))
    closed: List[float] = [c for c in (lp_gap_closed(base.get(name), value, upper.get(name))
                                       for name, value in cut.items()) if c is not None]
    return (sum(closed) / len(closed) if closed else None), len(closed)


def load_results(folder: str) -> Dict[str, Dict[str, SolveResult]]:
    """<label>/<cpu>/<instance>.json -> {cpu: {instance: result}} (empty if the folder does not exist)."""
    results: Dict[str, Dict[str, SolveResult]] = {}
    if not os.path.isdir(folder):
        return results
    for cpu in sorted(os.listdir(folder)):
        path: str = os.path.join(folder, cpu)
        if not os.path.isdir(path):
            continue
        for file in sorted(os.listdir(path)):
            if file.endswith(".json"):
                with open(os.path.join(path, file)) as f:
                    results.setdefault(cpu, {})[file[:-len(".json")]] = result_from_json(json.load(f))
    return results


def pair_by_cpu(
    base: Dict[str, Dict[str, SolveResult]], cut: Dict[str, Dict[str, SolveResult]]
) -> Tuple[Dict[str, SolveResult], Dict[str, SolveResult]]:
    """One (base, cut) pair per instance, both from the same CPU model (the first CPU that has both)."""
    base_out: Dict[str, SolveResult] = {}
    cut_out: Dict[str, SolveResult] = {}
    for cpu in sorted(cut):
        for name, result in cut[cpu].items():
            if name not in cut_out and name in base.get(cpu, {}):
                base_out[name], cut_out[name] = base[cpu][name], result
    return base_out, cut_out


SEED_LABEL: str = "seed"
PAPER_CUT_LABEL: str = "paper_sage"  # cuts/<problem>/paper_sage.py
PAPER_LLM: str = "Gemini 3 Pro"  # the paper's tables 5-6 name the selected family "Gemini 3 Pro selected"


def display_name(label: str, problem: str) -> str:
    """The row name in the report: Baseline (seed cut), Our best cut (<host>_<date>), Paper cut."""
    if label == SEED_LABEL:
        return "Baseline (seed cut)"
    if label == PAPER_CUT_LABEL:
        return "Paper cut (on our CPU)"
    if label.startswith(f"{problem}_"):  # a run's name: <problem>_<model>_<host>_<date>_<time>
        parts: List[str] = label.split("_")
        return f"Our best cut ({'_'.join(parts[-3:-1])})" if len(parts) >= 5 else f"Our best cut ({label})"
    return label


def row_order(label: str, problem: str) -> Tuple[int, str]:
    """Baseline, our best cuts, the paper's cut, then any other cut."""
    if label == SEED_LABEL:
        return 0, label
    if label == PAPER_CUT_LABEL:
        return 2, label
    return (1 if label.startswith(f"{problem}_") else 3), label


def report(args: argparse.Namespace) -> str:
    problem: Problem = PROBLEMS[args.problem_name]
    root: str = results_dir(args.out_dir, problem.name, args.time_limit)
    instances: List[Instance] = problem.loader(args.data_dir).load(parameters.BENCHMARK_FILE)
    n_set: int = len(instances)
    # the paper's checkpoints within the time limit, and the time limit itself (2000 s: the paper's 7)
    checkpoints: List[float] = [t for t in parameters.BENCHMARK_CHECKPOINTS if t < args.time_limit] + [args.time_limit]
    # {cpu: {instance: result}}, kept to the benchmark's CPU (other CPUs' results stay on disk, unused)
    keep: Callable[[Dict[str, Dict[str, SolveResult]]], Dict[str, Dict[str, SolveResult]]] = (
        (lambda r: r) if args.any_cpu else (lambda r: {c: v for c, v in r.items() if c == slug(parameters.BENCHMARK_CPU)}))
    base: Dict[str, Dict[str, SolveResult]] = keep(load_results(os.path.join(root, BASE_LABEL)))
    rows: List[BenchmarkRow] = []
    work_rows: List[BenchmarkRow] = []  # the same, at checkpoints in Gurobi work units
    invalid_notes: List[str] = []  # cuts proven invalid on some instance (violations)
    upper: Dict[str, float] = upper_bounds(instances, load_results(os.path.join(root, BASE_LABEL)))
    labels: List[str] = sorted(d for d in os.listdir(root) if d not in (BASE_LABEL, LP_DIR)
                               and os.path.isdir(os.path.join(root, d))) \
        if os.path.isdir(root) else []
    for label in sorted(labels, key=lambda l: row_order(l, problem.name)):
        base_pairs, cut_pairs = pair_by_cpu(base, keep(load_results(os.path.join(root, label))))
        row: BenchmarkRow = compare(label, base_pairs, cut_pairs, checkpoints, parameters.BENCHMARK_EPSILON,
                                    parameters.BENCHMARK_GAP_TOLERANCE, parameters.BENCHMARK_CLIP)
        if row.n_instances > 0:  # a cut with no result on this CPU yet: no row
            row.lp_closed, row.lp_n = mean_lp_closed(root, label, upper)
            row.label = display_name(label, problem.name)
            bad, checked = violations(root, label, upper)
            if bad:  # the table keeps it, flagged: its numbers are not results
                row.label += f" (INVALID on {len(bad)}/{checked})"
                invalid_notes.append(f"- {row.label.split(' (INVALID')[0]}: bound or LP relaxation above a known "
                                     f"solution on {', '.join(bad)}")
            rows.append(row)
        work_row: BenchmarkRow = compare(label, base_pairs, cut_pairs, parameters.BENCHMARK_WORK_CHECKPOINTS,
                                         parameters.BENCHMARK_EPSILON, parameters.BENCHMARK_GAP_TOLERANCE,
                                         parameters.BENCHMARK_CLIP, measure="work")
        if work_row.counts and any(work_row.counts):
            work_row.label = next((r.label for r in rows if r.label.startswith(display_name(label, problem.name))),
                                  display_name(label, problem.name))
            work_rows.append(work_row)
    hardware: str = "any CPU (each cut paired with the base on its CPU)" if args.any_cpu else parameters.BENCHMARK_CPU
    text: str = (f"# {problem.name.upper()}: benchmark set ({n_set} instances), TimeLimit {args.time_limit:g} s\n\n"
                 f"Solved on: {hardware}, {args.threads} threads.\n\n"
                 "Mean stabilized relative gap improvement (%) of each cut over the base model, as in the "
                 "paper's Table 2 (evolution/benchmark.py); n = instances compared so far. Gaps <= "
                 f"{parameters.BENCHMARK_GAP_TOLERANCE:g} count as 0, Delta g clipped to "
                 f"[-{100 * parameters.BENCHMARK_CLIP:g}%, {100 * parameters.BENCHMARK_CLIP:g}%].\n\n"
                 "Baseline = the problem's seed cut; Our best cut = the best cut of a run of ours; Paper cut "
                 "(on our CPU) = the paper's selected cut (Appendix F, cuts/<problem>/paper_sage.py) solved "
                 f"here; Paper cut (announced) = the paper's Table 2, {PAPER_LLM} row. LP gap closed = mean "
                 "(LP_cut - LP_base) / (UB - LP_base) of the root LP relaxation, UB = best known solution "
                 "(the base model's best, or the published optimum); `benchmark_cut.py --lp` solves the LPs.\n\n"
                 + f"{STD_NOTE}\n\n"
                 + format_table(problem.name, rows, paper_rows={f"Paper cut (announced, {PAPER_LLM})": PAPER_LLM})
                 + ("\n\n**Invalid cuts** (they remove optimal solutions: their gains are not results):\n"
                    + "\n".join(invalid_notes) if invalid_notes else "")
                 + "\n\n## In Gurobi work units\n\nThe same Delta g at checkpoints in work units (deterministic, "
                 "the same on every CPU; the paper reports seconds only). An instance counts at a checkpoint only "
                 "if both solves are known there: a solve stopped by the 2000 s limit before reaching that work "
                 "is left out ([n=...]: instances counted, when fewer than n). Only solves recorded since "
                 "2026-11-09 have the work along the solve.\n\n"
                 + (format_table(problem.name, work_rows, unit="wu") if work_rows else "No solve with the work recorded yet.")
                 + "\n")
    if os.path.isdir(root):
        with open(os.path.join(root, "report_any_cpu.md" if args.any_cpu else "report.md"), "w") as f:
            f.write(text)
    return text


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("solvers.solver").setLevel(logging.WARNING)
    arguments: argparse.Namespace = parse_args()
    if arguments.report:
        print(report(arguments))
    elif arguments.lp:
        solve_lps(arguments)
    else:
        run(arguments)
