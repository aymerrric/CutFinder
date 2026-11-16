"""
Everything problem-specific, in one table: PROBLEMS[<name>] gives the loader, the base model, the
seed cut and the sizes of the three instance sets. main.py, prepare_instances.py, review_run.py and
prompt_bench.py only go through it. The text the LLM reads about a problem is in
prompts/problems/<name>.yaml (evolution/prompts.py).

Convention: every base model MINIMIZES. A maximization problem minimizes the negated objective in
its create_model (the relative MIP gap and the share of the root gap closed do not change when the
objective is negated, so the evaluator and the review need no special case). check_minimization
enforces it.
"""
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from loaders.base import Instance, Loader
from loaders.corridor import LoaderCorridor
from loaders.cwlp import LoaderCWLP
from loaders.gisp import LoaderGISP
from loaders.imo6 import LoaderIMO6
from loaders.jssp import LoaderJSSP
from loaders.mcnd import LoaderMCND
from loaders.tsp import LoaderTSP
from loaders import validity
from models import corridor, cwlp, gisp, imo6, jssp, mcnd, tsp
from solvers.solver import Formulation
from utils import parameters

Sizes = Dict[str, Any]  # keyword arguments of the loader's generate (instance sizes)


@dataclass
class Problem:
    """Everything problem-specific that a run needs."""

    name: str  # also the folder of its data (data/<group>/<name>/) and of its prompt file
    loader: Callable[[str], Loader]  # data_dir -> loader
    base: Formulation  # the model without any cut (it must minimize)
    baseline_idea: str  # the seed of the first generation
    baseline_code: str
    validation_sizes: Sizes  # small: solved to optimality (validity and usefulness checks)
    evaluation_sizes: Sizes  # large: the base model cannot finish them (scoring during the run)
    test_sizes: Sizes  # held-out instances of the end-of-run review
    describe_sizes: Callable[[Sizes], str]  # e.g. {"min_cities": 60, "max_cities": 100} -> "60 to 100 cities"
    # raw download folder -> the (unsolved) validity instances (loaders/validity.py, prepare_validity.py)
    validity_instances: Callable[[str], List[Instance]] = lambda raw_dir: []
    # evaluation and test sets of the "hard" instance level (parameters.INSTANCE_LEVEL); None = the
    # problem has none
    hard_sizes: Optional[Sizes] = None
    # optimality sets of the gap_work fitness (parameters.OPTIMALITY_FILE): sizes the base model
    # proves optimal under the work limit; None = the problem has none
    optimality_sizes: Optional[Sizes] = None


@dataclass
class SetSpec:
    """How one evaluation or test set is built: its file in data/<group>/<problem>/, its seed, its sizes."""

    filename: str
    seed: int
    sizes: Sizes


def instance_set(problem: Problem, kind: str, level: str = parameters.INSTANCE_LEVEL) -> SetSpec:
    """The `kind` set ("evaluation" or "test") of `problem` at instance `level` (parameters.INSTANCE_LEVELS).
    The default level keeps the files evaluation.pkl / test.pkl; another level adds its name
    (evaluation_hard.pkl), so the sets of every level live side by side."""
    if kind not in ("evaluation", "test"):
        raise ValueError(f"unknown set {kind!r}: 'evaluation' or 'test'")
    if level not in parameters.INSTANCE_LEVELS:
        raise ValueError(f"unknown instance level {level!r} ({', '.join(parameters.INSTANCE_LEVELS)})")
    default_file: str = parameters.EVALUATION_FILE if kind == "evaluation" else parameters.TEST_FILE
    if level == "default":
        sizes: Sizes = problem.evaluation_sizes if kind == "evaluation" else problem.test_sizes
        return SetSpec(default_file, parameters.SET_SEEDS[(kind, level)], sizes)
    if problem.hard_sizes is None:
        raise ValueError(f"{problem.name} has no {level!r} instance level (Problem.hard_sizes in problems.py)")
    stem, extension = os.path.splitext(default_file)
    return SetSpec(f"{stem}_{level}{extension}", parameters.SET_SEEDS[(kind, level)], problem.hard_sizes)


def _range(low: int, high: int, unit: str) -> str:
    return f"{low} {unit}" if low == high else f"{low} to {high} {unit}"


PROBLEMS: Dict[str, Problem] = {
    "tsp": Problem(
        name="tsp",
        loader=LoaderTSP,
        base=tsp.create_model,
        baseline_idea=tsp.BASELINE_IDEA,
        baseline_code=tsp.BASELINE_CODE,
        validation_sizes={"min_cities": 15, "max_cities": 15},
        # the base model must NOT finish within WORK_LIMIT, so that a cut is scored by the gap it closes
        evaluation_sizes={"min_cities": 60, "max_cities": 100},
        test_sizes={"min_cities": 50, "max_cities": 100},  # under ~60 cities the base solves them
        describe_sizes=lambda s: _range(s["min_cities"], s["max_cities"], "cities"),
        validity_instances=lambda raw_dir: validity.tsp_instances(),
        # runs/analysis/tsp_sizes_20261112.txt (WorkLimit 10): the base model still finds a tour up
        # to 250 cities (gaps 51-67% at 200), and the best Muse cut, which solves most 60-100-city
        # instances, ends at a 53-100% gap from 125 cities on: larger instances rank such cuts again
        hard_sizes={"min_cities": 60, "max_cities": 200},
        # runs/analysis/tsp_optimality_sizes_20261112.txt: 40-60 cities solved with 0.3-8.7 work
        optimality_sizes={"min_cities": 40, "max_cities": 60},
    ),
    # sizes measured in runs/analysis/cwlp_sizes_20261102.txt: up to 200x40 the base model solves to
    # optimality within WORK_LIMIT, 300x50 stops at 0.06-0.17% (< MIN_BASE_GAP), 500x80 to 1000x100
    # at 0.43-0.77% in 5-6 s
    "cwlp": Problem(
        name="cwlp",
        loader=LoaderCWLP,
        base=cwlp.create_model,
        baseline_idea=cwlp.BASELINE_IDEA,
        baseline_code=cwlp.BASELINE_CODE,
        validation_sizes={"min_customers": 30, "max_customers": 30, "min_warehouses": 8, "max_warehouses": 8},
        evaluation_sizes={"min_customers": 500, "max_customers": 1000, "min_warehouses": 80, "max_warehouses": 150},
        test_sizes={"min_customers": 400, "max_customers": 1000, "min_warehouses": 60, "max_warehouses": 150},
        # cwlp_sizes_20261102.txt: 150x30 to 200x40 solved to optimality with 0.4-8.0 work
        optimality_sizes={"min_customers": 150, "max_customers": 200, "min_warehouses": 30, "max_warehouses": 40},
        describe_sizes=lambda s: (f"{_range(s['min_customers'], s['max_customers'], 'customers')} and "
                                  f"{_range(s['min_warehouses'], s['max_warehouses'], 'warehouses')}"),
        validity_instances=lambda raw_dir: validity.cwlp_instances(),
    ),
    # sizes measured in runs/analysis/jssp_sizes_20261102.txt: up to 10x10 the base model solves to
    # optimality within WORK_LIMIT (< 2 s); 15x5 to 20x15 stop at 11.7-36.0% gap in 4.5-6.8 s
    "jssp": Problem(
        name="jssp",
        loader=LoaderJSSP,
        base=jssp.create_model,
        baseline_idea=jssp.BASELINE_IDEA,
        baseline_code=jssp.BASELINE_CODE,
        validation_sizes={"min_jobs": 6, "max_jobs": 6, "min_machines": 4, "max_machines": 4},
        evaluation_sizes={"min_jobs": 15, "max_jobs": 20, "min_machines": 5, "max_machines": 15},
        test_sizes={"min_jobs": 12, "max_jobs": 20, "min_machines": 5, "max_machines": 15},
        # jssp_sizes_20261102.txt: 10x5 and 10x10 solved to optimality with 0.6-3.4 work
        optimality_sizes={"min_jobs": 10, "max_jobs": 10, "min_machines": 5, "max_machines": 10},
        describe_sizes=lambda s: (f"{_range(s['min_jobs'], s['max_jobs'], 'jobs')} on "
                                  f"{_range(s['min_machines'], s['max_machines'], 'machines')}"),
        validity_instances=lambda raw_dir: validity.jssp_generated() + validity.jsplib_instances(raw_dir),
    ),
    # sizes measured in runs/analysis/mcnd_sizes_20261113.txt: up to 20 nodes x 120 arcs x 100
    # commodities the base model mostly solves to optimality within WORK_LIMIT; 20 x 220-300 x
    # 100-200 stop at 0.75-5.44% in 7-9 s. The paper's test set: 20 nodes, 120-318 arcs, 40-200
    # commodities (loaders/benchmarks.py, CommaLab R)
    "mcnd": Problem(
        name="mcnd",
        loader=LoaderMCND,
        base=mcnd.create_model,
        baseline_idea=mcnd.BASELINE_IDEA,
        baseline_code=mcnd.BASELINE_CODE,
        validation_sizes={"min_nodes": 8, "max_nodes": 8, "min_arcs": 25, "max_arcs": 25,
                          "min_commodities": 10, "max_commodities": 10},
        evaluation_sizes={"min_nodes": 20, "max_nodes": 20, "min_arcs": 220, "max_arcs": 300,
                          "min_commodities": 100, "max_commodities": 200},
        test_sizes={"min_nodes": 20, "max_nodes": 20, "min_arcs": 200, "max_arcs": 300,
                    "min_commodities": 80, "max_commodities": 200},
        describe_sizes=lambda s: (f"{_range(s['min_nodes'], s['max_nodes'], 'nodes')}, "
                                  f"{_range(s['min_arcs'], s['max_arcs'], 'arcs')} and "
                                  f"{_range(s['min_commodities'], s['max_commodities'], 'commodities')}"),
        validity_instances=lambda raw_dir: validity.mcnd_instances(raw_dir),
        # 20 x 120 x 40-100: solved to optimality with work 1.65-9.36
        optimality_sizes={"min_nodes": 20, "max_nodes": 20, "min_arcs": 120, "max_arcs": 120,
                          "min_commodities": 40, "max_commodities": 100},
    ),
    # IMO 2025 Problem 6 (paper Appendix F.6): an instance is only its grid size N, so a set draws
    # distinct N (loaders/imo6.py). runs/analysis/imo6_sizes_20261113.txt: the base model proves
    # optimality up to N = 8 (14.6 s) and stops at a 50-98% gap under WORK_LIMIT from N = 8 to 46.
    # As in the paper: development (our evaluation) N = 37..46, test N = 3..36 (the benchmark);
    # our review set draws N = 20..36 from it, where the base model does not finish. No optimality
    # sets: only N = 7 is proved with a work of 1-5, and the gap_work fitness needs 8 distinct N
    "imo6": Problem(
        name="imo6",
        loader=LoaderIMO6,
        base=imo6.create_model,
        baseline_idea=imo6.BASELINE_IDEA,
        baseline_code=imo6.BASELINE_CODE,
        validation_sizes={"min_n": 5, "max_n": 6},
        evaluation_sizes={"min_n": 37, "max_n": 46},
        test_sizes={"min_n": 20, "max_n": 36},
        describe_sizes=lambda s: ("N x N grids with N = " + (str(s["min_n"]) if s["min_n"] == s["max_n"]
                                                             else f"{s['min_n']} to {s['max_n']}")),
        validity_instances=lambda raw_dir: validity.imo6_instances(),
    ),
    # Generalized Independent Set (not in the paper; Distributional MIPLIB's generator: G(n, 0.3),
    # 25% removable edges). runs/analysis/gisp_sizes_20261113.txt: up to 100 nodes the base model
    # proves optimality within WORK_LIMIT (work 1.6-1.8 at 100); from 125 nodes it stops at a
    # 16-114% gap in about 3 s (125: 2 of 3 unfinished). The benchmark (loaders/benchmarks.py) is
    # D-MIPLIB's "hard" test set, 175 nodes
    "gisp": Problem(
        name="gisp",
        loader=LoaderGISP,
        base=gisp.create_model,
        baseline_idea=gisp.BASELINE_IDEA,
        baseline_code=gisp.BASELINE_CODE,
        validation_sizes={"min_nodes": 40, "max_nodes": 40},
        evaluation_sizes={"min_nodes": 150, "max_nodes": 175},
        test_sizes={"min_nodes": 140, "max_nodes": 200},
        describe_sizes=lambda s: _range(s["min_nodes"], s["max_nodes"], "nodes"),
        validity_instances=lambda raw_dir: validity.gisp_instances(),
        # 100 nodes: proved optimal with a work of 1.6-1.8 (75: 0.25-0.31, too easy)
        optimality_sizes={"min_nodes": 90, "max_nodes": 110},
    ),
    # Wildlife corridor design (not in the paper; the corridor generator of Dilkina, Sabharwal and
    # Gomes: square lattices, 3 reserves, budget 20-30% of the total cost).
    # runs/analysis/corridor_sizes_20261113.txt: up to order 12 the base model proves optimality
    # within WORK_LIMIT (work 0.15-6.1); from order 15 it stops at 1.3-60% gaps (order 20: 1.3-5.9%,
    # 25: 8.8-27%, 30: 17-60%). Some instances are infeasible (the budget cannot connect the
    # reserves): build_set refuses them. The benchmark is generated (loaders/benchmarks.py)
    "corridor": Problem(
        name="corridor",
        loader=LoaderCorridor,
        base=corridor.create_model,
        baseline_idea=corridor.BASELINE_IDEA,
        baseline_code=corridor.BASELINE_CODE,
        validation_sizes={"min_order": 7, "max_order": 7},
        evaluation_sizes={"min_order": 20, "max_order": 25},
        test_sizes={"min_order": 18, "max_order": 30},
        describe_sizes=lambda s: (f"square lattices of {s['min_order']} x {s['min_order']}"
                                  + (f" to {s['max_order']} x {s['max_order']}" if s["max_order"] != s["min_order"] else "")
                                  + " cells"),
        validity_instances=lambda raw_dir: validity.corridor_instances(),
        # order 12: proved optimal with a work of 1.2-6.1 (10: 0.15-0.60, too easy)
        optimality_sizes={"min_order": 12, "max_order": 13},
    ),
}


def check_minimization(problem: Problem, instance: Instance) -> None:
    """Refuse a base model that does not minimize (see the module docstring)."""
    model: gp.Model = problem.base(**instance["args"])
    try:
        model.update()
        if model.ModelSense != GRB.MINIMIZE:
            raise ValueError(
                f"the base model of {problem.name} maximizes: minimize the negated objective instead "
                "(the evaluator and the review assume minimization)"
            )
    finally:
        model.dispose()