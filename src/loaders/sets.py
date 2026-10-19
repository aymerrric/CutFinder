import logging
import os
from typing import Any, Callable, List, Optional

from loaders.base import Instance, Loader
from solvers.solver import SolveResult, Solver

# Names of formulations in the Solver, also used as keys of their results in an instance:
# instance[BASE] is the SolveResult of the base model (no cut) on that instance.
BASE: str = "base"
BASE_LP: str = "base_lp"  # LP relaxation of the base model (its optimum checks cut usefulness)

MAX_BATCHES: int = 20  # build_set gives up after this many generation rounds

logger: logging.Logger = logging.getLogger(__name__)


def build_set(
    loader: Loader,
    filename: str,
    n_instances: int,
    seed: int,
    solver: Solver,
    require_optimal: bool,
    formulations: Optional[List[str]] = None,
    accept: Optional[Callable[[Instance], bool]] = None,
    **gen_kwargs: Any,
) -> List[Instance]:
    """
    Instances solved once, saved in `<loader.folder>/<filename>`.
    If the file exists it is loaded as is (delete it to rebuild the set).
    Otherwise: generate `n_instances` (with `seed` and `gen_kwargs`), solve each with every
    formulation of `formulations` (default [BASE]), store each result in instance[<name>],
    then save.

    accept: keep only the solved instances for which accept(instance) is True (e.g. a base gap
        large enough). Refused ones are replaced by new instances, generated with the next
        seeds (seed + 1, seed + 2, ...), until `n_instances` are kept.

    require_optimal: every instance must be solved to optimality (validation set: its
        optimal solutions are the reference that a valid cut must keep). Needs a solver
        with keep_solution=True, so that the solutions are stored.
    """
    path: str = os.path.join(loader.folder, filename)
    if os.path.exists(path):
        logger.info(f"loading {path}")
        return loader.load(filename)
    names: List[str] = formulations if formulations is not None else [BASE]
    missing: List[str] = [name for name in names if name not in solver.formulations]
    if missing:
        raise ValueError(f"the solver has no formulation {missing}")
    if require_optimal and not solver.keep_solution:
        raise ValueError("require_optimal needs a solver with keep_solution=True")
    instances: List[Instance] = []
    for batch in range(MAX_BATCHES):
        for inst in loader.generate(n_instances - len(instances), seed + batch, **gen_kwargs):
            for name in names:
                result: SolveResult = solver.solve(inst, name)
                if require_optimal and result.status != "OPTIMAL":
                    raise RuntimeError(
                        f"{inst['name']} ({name}): {result.status}, but a validation instance must "
                        "be solved to optimality: give the solver more time or use smaller instances"
                    )
                inst[name] = result
            if accept is None or accept(inst):
                instances.append(inst)
            else:
                logger.info(f"{inst['name']} refused, replaced by a new instance")
        if len(instances) == n_instances:
            break
    else:
        raise RuntimeError(
            f"only {len(instances)}/{n_instances} instances accepted after {MAX_BATCHES} rounds"
        )
    loader.save(instances, filename)
    logger.info(f"saved {len(instances)} instances to {path}")
    return instances
