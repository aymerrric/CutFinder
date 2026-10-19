import os
from pathlib import Path
from typing import List

import pytest

from loaders.base import Instance
from loaders.sets import BASE, build_set
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from solvers.solver import Solver


@pytest.fixture
def loader(tmp_path: Path) -> LoaderTSP:
    os.mkdir(os.path.join(tmp_path, "tsp"))  # the code never creates folders
    return LoaderTSP(data_dir=str(tmp_path))


def make_solver(keep_solution: bool, time_limit: float = 10) -> Solver:
    solver: Solver = Solver(log_dir=None, threads=1, isolated=False, keep_solution=keep_solution,
                            time_limit=time_limit)
    solver.add_formulation(BASE, create_model)
    return solver


def test_validation_set_is_solved_to_optimality_with_solutions(loader: LoaderTSP) -> None:
    instances: List[Instance] = build_set(
        loader, "val.pkl", 2, 1, make_solver(True), require_optimal=True, min_cities=6, max_cities=6
    )
    assert len(instances) == 2
    for inst in instances:
        assert inst[BASE].status == "OPTIMAL" and inst[BASE].solution is not None


def test_saved_then_loaded_without_solving_again(loader: LoaderTSP) -> None:
    first: List[Instance] = build_set(
        loader, "val.pkl", 2, 1, make_solver(True), require_optimal=True, min_cities=6, max_cities=6
    )
    no_base: Solver = Solver(log_dir=None, isolated=False)  # would fail if it had to solve
    again: List[Instance] = build_set(loader, "val.pkl", 2, 1, no_base, require_optimal=True)
    assert [i["name"] for i in again] == [i["name"] for i in first]
    assert again[0][BASE].obj_val == first[0][BASE].obj_val


def test_evaluation_set_keeps_no_solution(loader: LoaderTSP) -> None:
    instances: List[Instance] = build_set(
        loader, "eval.pkl", 2, 2, make_solver(False), require_optimal=False, min_cities=6, max_cities=6
    )
    assert all(inst[BASE].solution is None for inst in instances)


def test_not_optimal_validation_instance_is_refused(loader: LoaderTSP) -> None:
    with pytest.raises(RuntimeError, match="optimality"):
        build_set(loader, "val.pkl", 1, 1, make_solver(True, time_limit=0), require_optimal=True,
                  min_cities=12, max_cities=12)
    assert not os.path.exists(os.path.join(loader.folder, "val.pkl"))  # nothing saved


def test_missing_base_formulation_or_solution_is_refused(loader: LoaderTSP) -> None:
    with pytest.raises(ValueError):
        build_set(loader, "a.pkl", 1, 1, Solver(log_dir=None, isolated=False), require_optimal=False,
                  min_cities=5, max_cities=5)
    with pytest.raises(ValueError):
        build_set(loader, "b.pkl", 1, 1, make_solver(False), require_optimal=True,
                  min_cities=5, max_cities=5)


def test_refused_instances_are_replaced(loader: LoaderTSP) -> None:
    sizes: List[int] = []

    def big_enough(inst: Instance) -> bool:
        sizes.append(inst["args"]["n"])
        return inst["args"]["n"] >= 6

    instances: List[Instance] = build_set(
        loader, "eval.pkl", 3, 10, make_solver(False), require_optimal=False, accept=big_enough,
        min_cities=4, max_cities=7,
    )
    assert len(instances) == 3 and all(inst["args"]["n"] >= 6 for inst in instances)
    assert len(sizes) > 3  # some were refused and replaced
    assert len({inst["name"] for inst in instances}) == 3  # replacements have new names


def test_gives_up_when_nothing_is_accepted(loader: LoaderTSP) -> None:
    with pytest.raises(RuntimeError, match="accepted"):
        build_set(loader, "eval.pkl", 1, 10, make_solver(False), require_optimal=False,
                  accept=lambda inst: False, min_cities=4, max_cities=4)
