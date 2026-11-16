import logging
import os
import time
from pathlib import Path
from typing import List

import gurobipy as gp
import pytest

from loaders.base import Instance
from conftest import brute_force_tsp
from loaders.tsp import Costs, LoaderTSP
from models.tsp import create_model
from solvers.solver import (
    CRASHED,
    ERROR,
    KILLED,
    TOO_LARGE,
    Formulation,
    SeedMode,
    SolveResult,
    Solver,
    machine_info,
)

# Formulations are top-level functions: the isolated mode sends them to a child process.


def with_2cycle_cut(n: int, c: Costs) -> gp.Model:
    """A valid cut: never use both i -> j and j -> i."""
    m: gp.Model = create_model(n, c)
    m.addConstrs((m._x[i, j] + m._x[j, i] <= 1 for (i, j) in m._x.keys() if i < j), name="cut")
    return m


def unbounded() -> gp.Model:
    m: gp.Model = gp.Model()
    y: gp.Var = m.addVar(lb=-gp.GRB.INFINITY)
    m.setObjective(y, gp.GRB.MINIMIZE)
    return m


def raises(n: int, c: Costs) -> gp.Model:
    raise KeyError("bad index")  # like a buggy cut written by the LLM


def returns_none(n: int, c: Costs) -> gp.Model:
    return None


def hangs(n: int, c: Costs) -> gp.Model:
    while True:  # like a cut enumerating 2^n subsets
        time.sleep(0.1)


def dies(n: int, c: Costs) -> gp.Model:
    os._exit(3)  # the process disappears without a Python exception (like being out of memory)


def slow_build(n: int, c: Costs) -> gp.Model:
    time.sleep(1)  # building takes 1 s: must not be counted in the runtime
    return create_model(n, c)


def infeasible() -> gp.Model:
    m: gp.Model = gp.Model()
    x: gp.Var = m.addVar(vtype=gp.GRB.BINARY)
    m.addConstr(x >= 2)
    return m


def test_results_are_stored_per_formulation_and_instance(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("base", create_model)
    solver.add_formulation("base+2cycle", with_2cycle_cut)
    solver.solve_all(tsp_instances)
    for inst in tsp_instances:
        base: SolveResult = solver.results["base"][inst["name"]][0]
        cut: SolveResult = solver.results["base+2cycle"][inst["name"]][0]
        assert base.status == cut.status == "OPTIMAL"
        assert base.obj_val == pytest.approx(cut.obj_val)  # a valid cut keeps the optimum


def test_set_mode_always_uses_the_given_seed(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(seed=123, seed_mode=SeedMode.SET, log_dir=None, threads=1)
    solver.add_formulation("base", create_model)
    for _ in range(3):
        solver.solve(tsp_instances[0], "base")
    runs: List[SolveResult] = solver.results["base"][tsp_instances[0]["name"]]
    assert [r.seed for r in runs] == [123, 123, 123]


def test_random_mode_draws_and_records_new_seeds(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(seed_mode=SeedMode.RANDOM, log_dir=None, threads=1)
    solver.add_formulation("base", create_model)
    for _ in range(5):
        solver.solve(tsp_instances[0], "base")
    runs: List[SolveResult] = solver.results["base"][tsp_instances[0]["name"]]
    assert len(runs) == 5  # every run is kept, not overwritten
    assert len({r.seed for r in runs}) > 1


def test_log_file_per_solve_with_seed_in_name(tsp_instances: List[Instance], tmp_path: Path) -> None:
    solver: Solver = Solver(seed=7, log_dir=str(tmp_path), threads=1)
    solver.add_formulation("base", create_model)
    solver.solve(tsp_instances[0], "base")
    log_file: str = os.path.join(tmp_path, f"base_{tsp_instances[0]['name']}_seed7.log")
    assert os.path.exists(log_file)


def test_missing_log_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Solver(log_dir="/this/folder/does/not/exist")


def test_infeasible_model_gives_empty_values() -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("bad", infeasible)
    result: SolveResult = solver.solve({"name": "bad", "args": {}}, "bad")
    assert result.status == "INFEASIBLE"
    assert result.obj_val is None and result.mip_gap is None and result.obj_bound is None


def test_unbounded_model_does_not_crash() -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("unb", unbounded)
    result: SolveResult = solver.solve({"name": "unb", "args": {}}, "unb")
    assert result.status in ("UNBOUNDED", "INF_OR_UNBD")
    assert result.obj_val is None and result.error is None


@pytest.mark.parametrize("isolated", [True, False])
def test_isolated_and_in_process_give_the_same_optimum(
    tsp_instances: List[Instance], isolated: bool
) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=isolated)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(tsp_instances[0], "base")
    assert result.status == "OPTIMAL"
    assert result.obj_val == pytest.approx(brute_force_tsp(**tsp_instances[0]["args"]))


@pytest.mark.parametrize("isolated", [True, False])
@pytest.mark.parametrize("build", [raises, returns_none])
def test_broken_formulation_gives_error(
    tsp_instances: List[Instance], isolated: bool, build: Formulation
) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=isolated)
    solver.add_formulation("broken", build)
    result: SolveResult = solver.solve(tsp_instances[0], "broken")
    assert result.status == ERROR and result.runtime is None
    assert result.error


def test_error_message_is_kept() -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("broken", raises)
    result: SolveResult = solver.solve({"name": "x", "args": {"n": 3, "c": {}}}, "broken")
    assert result.error == "KeyError: 'bad index'"  # to be shown to the LLM later


@pytest.mark.parametrize("isolated", [True, False])
def test_too_many_constraints_is_refused(tsp_instances: List[Instance], isolated: bool) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=isolated, max_constraints=5)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(tsp_instances[0], "base")
    assert result.status == TOO_LARGE and result.runtime is None


def test_hanging_formulation_is_killed(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, time_limit=0.5, build_timeout=1)
    solver.add_formulation("hangs", hangs)
    start: float = time.time()
    result: SolveResult = solver.solve(tsp_instances[0], "hangs")
    assert result.status == KILLED
    assert time.time() - start < 10


def test_dead_child_is_reported(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("dies", dies)
    result: SolveResult = solver.solve(tsp_instances[0], "dies")
    assert result.status == CRASHED and "code 3" in result.error


@pytest.mark.parametrize("isolated", [True, False])
def test_runtime_does_not_count_building(tsp_instances: List[Instance], isolated: bool) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=isolated)
    solver.add_formulation("slow", slow_build)
    result: SolveResult = solver.solve(tsp_instances[0], "slow")
    assert result.status == "OPTIMAL"
    assert result.runtime < 0.5  # the 1 s of building is not in Gurobi's Runtime


def test_lambda_is_refused_only_when_isolated() -> None:
    with pytest.raises(ValueError):
        Solver(log_dir=None).add_formulation("lam", lambda n, c: create_model(n, c))
    Solver(log_dir=None, isolated=False).add_formulation("lam", lambda n, c: create_model(n, c))


def test_each_solve_follows_its_own_verbose(tsp_instances: List[Instance]) -> None:
    loud: Solver = Solver(log_dir=None, threads=1, isolated=False, verbose=True)
    quiet: Solver = Solver(log_dir=None, threads=1, isolated=False, verbose=False)  # created after
    loud.add_formulation("base", create_model)
    quiet.add_formulation("base", create_model)
    loud.solve(tsp_instances[0], "base")
    assert logging.getLogger("gurobipy").level == logging.INFO  # not silenced by `quiet`
    quiet.solve(tsp_instances[0], "base")
    assert logging.getLogger("gurobipy").level == logging.WARNING


@pytest.mark.parametrize("isolated", [True, False])
def test_keep_solution_stores_the_tour(tsp_instances: List[Instance], isolated: bool) -> None:
    inst: Instance = tsp_instances[0]
    solver: Solver = Solver(log_dir=None, threads=1, isolated=isolated, keep_solution=True)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(inst, "base")
    n: int = inst["args"]["n"]
    assert len(result.solution) == n * (n - 1) + n  # every x[i,j] and u[i]
    used: List[str] = [name for name, value in result.solution.items() if name.startswith("x") and value > 0.5]
    assert len(used) == n  # a tour uses n arcs


def test_solution_not_kept_by_default(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False)
    solver.add_formulation("base", create_model)
    assert solver.solve(tsp_instances[0], "base").solution is None


def test_work_is_recorded_and_reproducible(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=2, isolated=False)
    solver.add_formulation("base", create_model)
    works: List[float] = [solver.solve(tsp_instances[0], "base").work for _ in range(3)]
    assert works[0] > 0 and works == [works[0]] * 3  # same model + threads -> same work


def test_work_limit_stops_the_solve() -> None:
    big: Instance = LoaderTSP().generate(1, seed=0, min_cities=60, max_cities=60)[0]
    solver: Solver = Solver(log_dir=None, threads=2, isolated=False, work_limit=0.05)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(big, "base")
    assert result.status == "WORK_LIMIT"
    assert result.work == pytest.approx(0.05, rel=0.5)


@pytest.mark.parametrize("isolated", [True, False])
def test_machine_and_threads_are_recorded(tsp_instances: List[Instance], isolated: bool) -> None:
    solver: Solver = Solver(log_dir=None, threads=3, isolated=isolated)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(tsp_instances[0], "base")
    assert result.threads == 3
    assert (result.machine, result.cpu) == machine_info() and result.machine and result.cpu


def test_machine_is_recorded_for_failed_solves(tsp_instances: List[Instance]) -> None:
    solver: Solver = Solver(log_dir=None, threads=1)
    solver.add_formulation("broken", raises)
    assert solver.solve(tsp_instances[0], "broken").machine == machine_info()[0]


@pytest.mark.parametrize("isolated", [True, False])
def test_extra_gurobi_params_are_applied(isolated: bool) -> None:
    big: Instance = LoaderTSP().generate(1, seed=0, min_cities=40, max_cities=40)[0]
    solver: Solver = Solver(log_dir=None, threads=2, isolated=isolated, params={"SolutionLimit": 1})
    solver.add_formulation("base", create_model)
    assert solver.solve(big, "base").status == "SOLUTION_LIMIT"  # stopped at the first solution


@pytest.mark.parametrize("isolated", [True, False])
def test_trajectory_is_recorded_only_when_asked(tsp_instances: List[Instance], isolated: bool) -> None:
    big: Instance = LoaderTSP().generate(1, seed=0, min_cities=40, max_cities=40)[0]
    solver: Solver = Solver(log_dir=None, threads=2, isolated=isolated, record_trajectory=True, time_limit=5)
    solver.add_formulation("base", create_model)
    result: SolveResult = solver.solve(big, "base")
    points = result.trajectory
    assert points, "no progress recorded"
    times = [p[0] for p in points]
    assert times == sorted(times) and times[-1] <= result.runtime
    works = [p[3] for p in points]  # Gurobi work, alongside the time
    assert works == sorted(works) and works[-1] <= result.work + 1e-9
    bounds = [p[2] for p in points]
    assert bounds == sorted(bounds)  # minimization: the bound only rises
    found = [p[1] for p in points if p[1] < 1e99]
    assert found == sorted(found, reverse=True) and found[-1] == pytest.approx(result.obj_val)
    plain: Solver = Solver(log_dir=None, threads=2, isolated=isolated)
    plain.add_formulation("base", create_model)
    assert plain.solve(tsp_instances[0], "base").trajectory is None
