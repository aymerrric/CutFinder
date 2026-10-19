import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from loaders.base import Instance
from utils import parameters

# A formulation builds a Gurobi model from the instance arguments: model = build(**instance["args"])
# e.g. models.tsp.create_model(n, c). The solver knows nothing else about the problem.
Formulation = Callable[..., gp.Model]

# Gurobi status code -> readable name (2 -> "OPTIMAL", 9 -> "TIME_LIMIT", ...)
STATUS_NAMES: Dict[int, str] = {
    getattr(GRB.Status, name): name for name in dir(GRB.Status) if name.isupper()
}

# Gurobi accepts seeds in [0, 2_000_000_000]
MAX_SEED: int = 2_000_000_000

logger: logging.Logger = logging.getLogger(__name__)


class SeedMode(Enum):
    SET = "set"  # every solve uses the same seed (`seed` argument)
    RANDOM = "random"  # every solve draws a new random seed (recorded in its SolveResult)


@dataclass
class SolveResult:
    """What we keep from one solve (the model itself is discarded)."""

    instance: str
    formulation: str
    seed: int  # Gurobi Seed used: rerun with SeedMode.SET and this seed to reproduce
    status: str  # e.g. "OPTIMAL", "TIME_LIMIT", "INFEASIBLE"
    runtime: float  # seconds
    node_count: float  # branch-and-bound nodes explored
    sol_count: int  # feasible solutions found
    obj_val: Optional[float]  # best solution value, None if no solution found
    obj_bound: Optional[float]  # best proven bound
    mip_gap: Optional[float]  # |obj_val - obj_bound| / |obj_val|, None if no solution found


class Solver:
    """
    Problem-independent MILP solver: registers named formulations, solves instances with them
    under the same settings, and keeps every result in `self.results[formulation][instance]`
    (a list: one entry per solve, e.g. one per seed).
    """

    def __init__(
        self,
        time_limit: float = parameters.TIME_LIMIT,
        seed: int = parameters.SEED,
        seed_mode: SeedMode = SeedMode(parameters.SEED_MODE),
        threads: int = parameters.THREADS,
        log_dir: Optional[str] = parameters.LOG_DIR,
        verbose: bool = False,
    ) -> None:
        """
        seed_mode: SET = always use `seed`; RANDOM = draw a new seed for every solve
                   (`seed` is then ignored).
        log_dir: folder receiving one Gurobi log file per solve (None = no log files).
                 It must already exist: the solver does not create it.
        verbose: also print the Gurobi log to the console.
        """
        if log_dir is not None and not os.path.isdir(log_dir):
            raise FileNotFoundError(f"{log_dir} does not exist: create it or pass log_dir=None")
        self.time_limit: float = time_limit
        self.seed: int = seed
        self.seed_mode: SeedMode = seed_mode
        self._rng: random.Random = random.Random()  # unseeded: different seeds on every run
        self.threads: int = threads
        self.log_dir: Optional[str] = log_dir
        self.verbose: bool = verbose
        # gurobipy also forwards its whole log to the Python logger "gurobipy": keep it quiet
        # unless verbose (the full log still goes to the log files)
        logging.getLogger("gurobipy").setLevel(logging.INFO if verbose else logging.WARNING)
        # set once on the default environment: models created afterwards inherit it
        gp.setParam("LogToConsole", int(verbose))
        self.formulations: Dict[str, Formulation] = {}
        # formulation -> instance -> results, in solve order
        self.results: Dict[str, Dict[str, List[SolveResult]]] = {}

    def add_formulation(self, name: str, build: Formulation) -> None:
        """Register `build` under `name` (e.g. "base", or "base+cut_3")."""
        self.formulations[name] = build
        self.results.setdefault(name, {})

    def solve(self, instance: Instance, formulation: str) -> SolveResult:
        """Build the model of `instance` with `formulation`, solve it, store and return the result."""
        seed: int = self._next_seed()
        model: gp.Model = self.formulations[formulation](**instance["args"])
        self._configure(model, seed, f"{formulation}_{instance['name']}_seed{seed}.log")
        model.optimize()
        result: SolveResult = self._read_result(model, instance["name"], formulation, seed)
        model.dispose()  # free Gurobi memory right away
        self.results[formulation].setdefault(instance["name"], []).append(result)
        logger.info(
            f"[{formulation}] {instance['name']} seed={seed}: {result.status}, "
            f"obj={result.obj_val}, gap={result.mip_gap}, {result.runtime:.2f}s"
        )
        return result

    def solve_all(
        self, instances: List[Instance], formulations: Optional[List[str]] = None
    ) -> None:
        """Solve every instance with every formulation (all registered ones by default)."""
        names: List[str] = formulations if formulations is not None else list(self.formulations)
        for name in names:
            for instance in instances:
                self.solve(instance, name)

    def _next_seed(self) -> int:
        """Seed for the next solve, according to `seed_mode`."""
        if self.seed_mode is SeedMode.RANDOM:
            return self._rng.randint(0, MAX_SEED)
        return self.seed

    def _configure(self, model: gp.Model, seed: int, log_name: str) -> None:
        """Apply the same settings to every model, so that formulations are compared fairly."""
        model.Params.LogToConsole = int(self.verbose)  # first: otherwise the next lines are echoed
        model.Params.TimeLimit = self.time_limit
        model.Params.Seed = seed  # also written in the log file header ("Non-default parameters")
        model.Params.Threads = self.threads
        if self.log_dir is not None:
            model.Params.LogFile = os.path.join(self.log_dir, log_name)

    @staticmethod
    def _read_result(model: gp.Model, instance: str, formulation: str, seed: int) -> SolveResult:
        has_solution: bool = model.SolCount > 0
        return SolveResult(
            instance=instance,
            formulation=formulation,
            seed=seed,
            status=STATUS_NAMES.get(model.Status, str(model.Status)),
            runtime=model.Runtime,
            node_count=model.NodeCount,
            sol_count=model.SolCount,
            obj_val=model.ObjVal if has_solution else None,
            obj_bound=model.ObjBound if model.Status != GRB.INFEASIBLE else None,
            mip_gap=model.MIPGap if has_solution else None,
        )
