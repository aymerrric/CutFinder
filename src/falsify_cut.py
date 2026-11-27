"""
Search a counterexample to cuts on tiny instances (proving/falsify.py): REFUTED (the optimum
changes), SAFE_NOT_VALID (a non-optimal feasible point is removed), NO_COUNTEREXAMPLE (nothing
found: not a proof), ERROR. A free filter before any proof (paper plan, step 1).
Run from src/:
    python falsify_cut.py --problem_name tsp --cut_file ../cuts/tsp/paper_sage.py
    python falsify_cut.py --population_file ../runs/<run>/population.json            # its best cut
    python falsify_cut.py --population_file ../runs/<run>/population.json --all      # every distinct cut
The problem of a run comes from its folder name. Writes one JSON line per cut to --out (default
runs/falsify/falsify_<date>.jsonl): the verdict and its counterexample (instance, point, row).
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from proving.falsify import N_INSTANCES, TINY, TINY_IMO6_SIZES, Verdict, falsify, tiny_instances
from utils import parameters


def cuts_of(args: argparse.Namespace) -> Iterator[Tuple[str, str, str]]:
    """(problem, label, code) of every cut to check."""
    for path in args.cut_file or []:
        if not args.problem_name:
            raise ValueError("--cut_file needs --problem_name")
        with open(path) as f:
            yield args.problem_name, path, f.read()
    for path in args.population_file or []:
        run: str = os.path.basename(os.path.dirname(os.path.abspath(path)))
        problem: str = args.problem_name or run.split("_")[0]
        with open(path) as f:
            data: Dict[str, Any] = json.load(f)
        if not args.all:
            yield problem, f"{run}:best", data["best"]["code"]
            continue
        seen: set = set()
        for gen in data["history"] + [data["current"]]:
            for ind in gen:
                code: str = ind["code"].strip()
                if ind.get("fitness") is not None and code not in seen:
                    seen.add(code)
                    yield problem, f"{run}:{ind['id']}", ind["code"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problem_name", choices=sorted(TINY), help="default: from the run's folder")
    parser.add_argument("--cut_file", nargs="*")
    parser.add_argument("--population_file", nargs="*")
    parser.add_argument("--all", action="store_true", help="every distinct cut of the populations, not only the best")
    parser.add_argument("--instances", type=int, default=N_INSTANCES,
                        help=f"maximum cases per cut (IMO6 checks each N in {TINY_IMO6_SIZES} once)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the tiny instances")
    parser.add_argument("--out", default=os.path.join(parameters.RUNS_DIR, "falsify",
                                                       f"falsify_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> List[Verdict]:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    verdicts: List[Verdict] = []
    instances: Dict[str, list] = {}
    with open(args.out, "a") as out:
        for problem, label, code in cuts_of(args):
            if problem not in instances:
                instances[problem] = tiny_instances(problem, args.instances, args.seed)
            verdict: Verdict = falsify(problem, code, instances[problem])
            verdicts.append(verdict)
            ce = verdict.counterexample
            print(f"{label}: {verdict.verdict} ({verdict.instances} instances, {verdict.seconds} s)"
                  + (f" - {ce.instance}: {ce.row[:120]}" if ce else "") + (f" - {verdict.error}" if verdict.error else ""),
                  flush=True)
            out.write(json.dumps({"problem": problem, "cut": label, "code": code, "seed": args.seed,
                                  **verdict.to_dict()}) + "\n")
    print(f"written to {args.out}")
    return verdicts


if __name__ == "__main__":
    main(parse_args())
    sys.exit(0)
