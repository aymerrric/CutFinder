import os
from pathlib import Path
from typing import List

import gurobipy as gp
import pytest

from loaders.base import Instance
from loaders.tsp import Costs
from models.tsp import create_model
from solvers.solver import SeedMode, SolveResult, Solver


def with_2cycle_cut(n: int, c: Costs) -> gp.Model:
    """A valid cut: never use both i -> j and j -> i."""
    m: gp.Model = create_model(n, c)
    m.addConstrs((m._x[i, j] + m._x[j, i] <= 1 for (i, j) in m._x.keys() if i < j), name="cut")
    return m


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
