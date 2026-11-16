import functools
import logging
import multiprocessing as mp
import os
import pickle
import platform
import random
import socket
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Our own statuses, for solves that did not reach the end of optimize()
ERROR: str = "ERROR"  # the formulation (or Gurobi) raised an exception
TOO_LARGE: str = "TOO_LARGE"  # more constraints than max_constraints: refused before solving
KILLED: str = "KILLED"  # isolated child still running after time_limit + build_timeout
CRASHED: str = "CRASHED"  # isolated child died without sending a result (e.g. out of memory)

# One point of a solve's progress: (seconds since the start of optimize(), best objective found,
# best bound, Gurobi work so far). The best objective is GRB.INFINITY (1e100) while no solution is
# known. Results saved before 2026-11-09 have 3-element points (no work).
TrajectoryPoint = Tuple[float, ...]

# Gurobi accepts seeds in [0, 2_000_000_000]
MAX_SEED: int = 2_000_000_000

# "spawn": a fresh Python process, the same on every OS. ("fork" would copy the parent, which
# is unsafe once Gurobi has started threads in it.)
_MP_CONTEXT: Any = mp.get_context("spawn")

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
    status: str  # Gurobi's ("OPTIMAL", "TIME_LIMIT", ...) or ours (ERROR, TOO_LARGE, ...)
    runtime: Optional[float]  # seconds inside optimize() (Gurobi's Runtime); None if not solved
    work: Optional[float]  # Gurobi's Work: deterministic effort (same model + threads -> same value)
    node_count: Optional[float]  # branch-and-bound nodes explored
    sol_count: int  # feasible solutions found
    obj_val: Optional[float]  # best solution value, None if no solution found
    obj_bound: Optional[float]  # best proven bound, None if Gurobi has none
    mip_gap: Optional[float]  # |obj_val - obj_bound| / |obj_val|, None if no solution found
    error: Optional[str] = None  # what went wrong, for our own statuses
    # best solution found, by variable name ("x[1,2]" -> 1.0); only with keep_solution=True
    solution: Optional[Dict[str, float]] = None
    # where it ran: Work depends on the thread count, the runtime also on the CPU
    threads: int = 0
    machine: str = ""
    cpu: str = ""
    # progress during the solve, in time order (only with record_trajectory=True): the gap at
    # any time t can be read from it (evolution/benchmark.py), e.g. the paper's checkpoints
    trajectory: Optional[List[TrajectoryPoint]] = None

    @classmethod
    def failed(
        cls, instance: str, formulation: str, seed: int, status: str, error: str
    ) -> "SolveResult":
        """A solve that did not reach the end of optimize(): no statistics."""
        return cls(instance, formulation, seed, status, None, None, None, 0, None, None, None, error)


@dataclass
class SolveSettings:
    """Everything one solve needs besides the model (picklable: sent to the child process)."""

    time_limit: float
    work_limit: Optional[float]  # None = no work limit
    threads: int
    seed: int
    log_file: Optional[str]  # None = no log file
    verbose: bool
    max_constraints: Optional[int]  # None = no cap
    keep_solution: bool = False  # store the variable values in SolveResult.solution
    params: Dict[str, Any] = field(default_factory=dict)  # other Gurobi parameters, e.g. {"MIPGap": 1e-5}
    record_trajectory: bool = False  # record (time, best objective, bound) during the solve


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
        isolated: bool = parameters.ISOLATED,
        build_timeout: float = parameters.BUILD_TIMEOUT,
        max_constraints: Optional[int] = parameters.MAX_CONSTRAINTS,
        keep_solution: bool = False,
        work_limit: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        record_trajectory: bool = False,
    ) -> None:
        """
        seed_mode: SET = always use `seed`; RANDOM = draw a new seed for every solve
                   (`seed` is then ignored).
        log_dir: folder receiving one Gurobi log file per solve (None = no log files).
                 It must already exist: the solver does not create it.
        verbose: also print the Gurobi log to the console.
        isolated: build and solve every model in its own child process, killed if it is still
                  running after `time_limit + build_timeout` seconds. Formulations must then be
                  picklable (top-level functions, not lambdas).
        max_constraints: models with more constraints are refused (status TOO_LARGE).
        keep_solution: store the best solution's variable values in SolveResult.solution
                       (e.g. the optimal tours used to check that a cut is valid).
        work_limit: stop after this much Gurobi work (about 1 s of work on Gurobi's reference
                    machine). Unlike `time_limit`, it gives the same result on every run and
                    machine (for the same `threads`). `time_limit` stays as a safety net.
        params: other Gurobi parameters set on every model, by name (e.g. the paper's test
                protocol: {"MIPGap": 1e-5, "Method": 4}). Applied last: they override the others.
        record_trajectory: record the best objective and bound during the solve
                           (SolveResult.trajectory), to read the gap at given times afterwards.
        """
        if log_dir is not None and not os.path.isdir(log_dir):
            raise FileNotFoundError(f"{log_dir} does not exist: create it or pass log_dir=None")
        self.time_limit: float = time_limit
        self.work_limit: Optional[float] = work_limit
        self.seed: int = seed
        self.seed_mode: SeedMode = seed_mode
        self._rng: random.Random = random.Random()  # unseeded: different seeds on every run
        self.threads: int = threads
        self.log_dir: Optional[str] = log_dir
        self.verbose: bool = verbose
        self.isolated: bool = isolated
        self.build_timeout: float = build_timeout
        self.max_constraints: Optional[int] = max_constraints
        self.keep_solution: bool = keep_solution
        self.params: Dict[str, Any] = dict(params or {})
        self.record_trajectory: bool = record_trajectory
        self.formulations: Dict[str, Formulation] = {}
        # formulation -> instance -> results, in solve order
        self.results: Dict[str, Dict[str, List[SolveResult]]] = {}

    def add_formulation(self, name: str, build: Formulation) -> None:
        """Register `build` under `name` (e.g. "base", or "base+cut_3")."""
        if self.isolated:
            try:  # fail now rather than at the first solve
                pickle.dumps(build)
            except (pickle.PicklingError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"formulation {name!r} cannot be sent to a child process ({e}): "
                    "use a top-level function, or isolated=False"
                ) from e
        self.formulations[name] = build
        self.results.setdefault(name, {})

    def solve(self, instance: Instance, formulation: str) -> SolveResult:
        """Build the model of `instance` with `formulation`, solve it, store and return the result."""
        seed: int = self._next_seed()
        log_file: Optional[str] = None
        if self.log_dir is not None:
            log_file = os.path.join(self.log_dir, f"{formulation}_{instance['name']}_seed{seed}.log")
        settings: SolveSettings = SolveSettings(
            self.time_limit, self.work_limit, self.threads, seed, log_file, self.verbose,
            self.max_constraints, self.keep_solution, self.params, self.record_trajectory,
        )
        build: Formulation = self.formulations[formulation]
        result: SolveResult
        if self.isolated:
            result = solve_isolated(
                build, instance, formulation, settings, self.time_limit + self.build_timeout
            )
        else:
            result = build_and_solve(build, instance, formulation, settings)
        result.threads = self.threads
        result.machine, result.cpu = machine_info()
        self.results[formulation].setdefault(instance["name"], []).append(result)
        if result.error is not None:
            logger.warning(f"[{formulation}] {instance['name']} seed={seed}: {result.status}, {result.error}")
        else:
            logger.info(
                f"[{formulation}] {instance['name']} seed={seed}: {result.status}, "
                f"obj={result.obj_val}, gap={result.mip_gap}, work={result.work:.4f}, "
                f"{result.runtime:.2f}s on {result.machine}"
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


@functools.lru_cache(maxsize=None)
def machine_info() -> Tuple[str, str]:
    """(short host name, CPU model) of this computer, e.g. ("saumon", "Intel(R) Xeon(R) ...")."""
    host: str = socket.gethostname().split(".")[0]
    cpu: str = platform.processor() or platform.machine()  # fallback: "x86_64", "arm"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
        elif platform.system() == "Darwin":  # gethostname() there is often the network's name
            host = _run(["scutil", "--get", "LocalHostName"]) or host
            cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    except (OSError, StopIteration):
        pass
    return host, cpu


def _run(command: List[str]) -> str:
    """Output of a small system command, or "" if it fails."""
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# The functions below are top-level (not methods) so that a child process can run them.


def build_and_solve(
    build: Formulation, instance: Instance, formulation: str, settings: SolveSettings
) -> SolveResult:
    """
    Build the model, check its size, solve it and read the result. Never raises: an exception
    in the formulation (e.g. in code written by the LLM) gives an ERROR result.
    Identical in both modes, so isolated and in-process solves are timed the same way:
    `runtime` is Gurobi's Runtime, which covers optimize() only (not building the model,
    nor starting the child process).
    """
    name: str = instance["name"]
    _apply_console_settings(settings.verbose)
    try:
        model: gp.Model = build(**instance["args"])
    except Exception as e:  # noqa: BLE001 (any error in the formulation is reported, not raised)
        return SolveResult.failed(name, formulation, settings.seed, ERROR, f"{type(e).__name__}: {e}")
    if not isinstance(model, gp.Model):
        return SolveResult.failed(
            name, formulation, settings.seed, ERROR, f"formulation returned {type(model).__name__}, not a gp.Model"
        )
    try:
        model.update()  # apply pending changes so NumConstrs is up to date
        if settings.max_constraints is not None and model.NumConstrs > settings.max_constraints:
            return SolveResult.failed(
                name, formulation, settings.seed, TOO_LARGE,
                f"{model.NumConstrs} constraints > max_constraints={settings.max_constraints}",
            )
        _configure(model, settings)
        if settings.record_trajectory:
            model._trajectory = []  # filled by _record_progress
            model.optimize(_record_progress)
        else:
            model.optimize()
        result: SolveResult = _read_result(model, name, formulation, settings.seed, settings.keep_solution)
        if settings.record_trajectory:
            result.trajectory = model._trajectory
        return result
    except Exception as e:  # noqa: BLE001
        return SolveResult.failed(name, formulation, settings.seed, ERROR, f"{type(e).__name__}: {e}")
    finally:
        model.dispose()  # free Gurobi memory right away


def solve_isolated(
    build: Formulation,
    instance: Instance,
    formulation: str,
    settings: SolveSettings,
    timeout: float,
) -> SolveResult:
    """Run `build_and_solve` in a child process; kill it after `timeout` seconds."""
    receiver: Connection
    sender: Connection
    receiver, sender = _MP_CONTEXT.Pipe(duplex=False)
    process: Any = _MP_CONTEXT.Process(
        target=_child_main, args=(sender, build, instance, formulation, settings), daemon=True
    )
    process.start()
    sender.close()  # the parent keeps only its end: recv() then sees EOF if the child dies
    try:
        if receiver.poll(timeout):
            return receiver.recv()
        return SolveResult.failed(
            instance["name"], formulation, settings.seed, KILLED,
            f"no result after {timeout:.0f}s (building + solving)",
        )
    except EOFError:
        process.join()
        return SolveResult.failed(
            instance["name"], formulation, settings.seed, CRASHED,
            f"child process exited with code {process.exitcode} without a result",
        )
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        receiver.close()


def _child_main(
    sender: Connection,
    build: Formulation,
    instance: Instance,
    formulation: str,
    settings: SolveSettings,
) -> None:
    """Entry point of the child process."""
    if not settings.verbose:
        # Gurobi prints its license banner when it starts, in every new process: hide stdout
        devnull: int = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
    sender.send(build_and_solve(build, instance, formulation, settings))
    sender.close()


def _apply_console_settings(verbose: bool) -> None:
    """
    Gurobi console output, set at every solve (not once per Solver) so that each solve follows
    the `verbose` of its own Solver. In isolated mode this runs in the child process only.
    """
    # gurobipy also forwards its whole log to the Python logger "gurobipy"
    logging.getLogger("gurobipy").setLevel(logging.INFO if verbose else logging.WARNING)
    gp.setParam("LogToConsole", int(verbose))  # default environment: inherited by new models


def _configure(model: gp.Model, settings: SolveSettings) -> None:
    """Apply the same settings to every model, so that formulations are compared fairly."""
    model.Params.LogToConsole = int(settings.verbose)  # first: otherwise the next lines are echoed
    model.Params.TimeLimit = settings.time_limit
    if settings.work_limit is not None:
        model.Params.WorkLimit = settings.work_limit
    model.Params.Seed = settings.seed  # also written in the log file header ("Non-default parameters")
    model.Params.Threads = settings.threads
    if settings.log_file is not None:
        model.Params.LogFile = settings.log_file
    for name, value in settings.params.items():
        model.setParam(name, value)


# what each callback location calls the best objective and the best bound
_PROGRESS_CODES: Dict[int, Tuple[int, int]] = {
    GRB.Callback.MIP: (GRB.Callback.MIP_OBJBST, GRB.Callback.MIP_OBJBND),
    GRB.Callback.MIPSOL: (GRB.Callback.MIPSOL_OBJBST, GRB.Callback.MIPSOL_OBJBND),
    GRB.Callback.MIPNODE: (GRB.Callback.MIPNODE_OBJBST, GRB.Callback.MIPNODE_OBJBND),
}


def _record_progress(model: gp.Model, where: int) -> None:
    """Gurobi callback: append (time, best objective, bound, work) to model._trajectory when the
    best objective or the bound changes."""
    codes: Optional[Tuple[int, int]] = _PROGRESS_CODES.get(where)
    if codes is None:
        return
    best: float = model.cbGet(codes[0])
    bound: float = model.cbGet(codes[1])
    points: List[TrajectoryPoint] = model._trajectory
    if not points or points[-1][1:3] != (best, bound):
        points.append((model.cbGet(GRB.Callback.RUNTIME), best, bound, model.cbGet(GRB.Callback.WORK)))


def _safe_attr(model: gp.Model, name: str) -> Optional[float]:
    """Model attribute, or None when Gurobi has no value for it (e.g. ObjBound when UNBOUNDED)."""
    try:
        return getattr(model, name)
    except (AttributeError, gp.GurobiError):
        return None


def _read_result(
    model: gp.Model, instance: str, formulation: str, seed: int, keep_solution: bool = False
) -> SolveResult:
    has_solution: bool = model.SolCount > 0
    solution: Optional[Dict[str, float]] = None
    if keep_solution and has_solution:
        solution = {var.VarName: var.X for var in model.getVars()}
    return SolveResult(
        instance=instance,
        formulation=formulation,
        seed=seed,
        status=STATUS_NAMES.get(model.Status, str(model.Status)),
        runtime=model.Runtime,
        work=_safe_attr(model, "Work"),
        node_count=_safe_attr(model, "NodeCount"),
        sol_count=model.SolCount,
        obj_val=model.ObjVal if has_solution else None,
        # INFEASIBLE: Gurobi reports ObjBound = inf, which is no real bound
        obj_bound=_safe_attr(model, "ObjBound") if model.Status != GRB.INFEASIBLE else None,
        mip_gap=_safe_attr(model, "MIPGap") if has_solution else None,
        solution=solution,
    )
