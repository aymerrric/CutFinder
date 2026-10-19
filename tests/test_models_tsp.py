from typing import Dict, List

import gurobipy as gp
import pytest
from conftest import brute_force_tsp

from loaders.base import Instance
from models.tsp import create_model


def solve_quietly(model: gp.Model) -> None:
    model.Params.LogToConsole = 0
    model.optimize()


def test_model_size(tsp_instances: List[Instance]) -> None:
    n: int = tsp_instances[0]["args"]["n"]
    m: gp.Model = create_model(**tsp_instances[0]["args"])
    assert m.NumVars == n * (n - 1) + n  # x[i, j] for i != j, plus u[i]
    assert m.NumConstrs == 2 * n + (n - 1) * (n - 2)  # out + in + MTZ (depot excluded)


def test_optimum_matches_brute_force(tsp_instances: List[Instance]) -> None:
    for inst in tsp_instances:
        m: gp.Model = create_model(**inst["args"])
        solve_quietly(m)
        assert m.ObjVal == pytest.approx(brute_force_tsp(**inst["args"]))


def test_solution_is_a_single_tour(tsp_instances: List[Instance]) -> None:
    for inst in tsp_instances:
        n: int = inst["args"]["n"]
        m: gp.Model = create_model(**inst["args"])
        solve_quietly(m)
        succ: Dict[int, int] = {i: j for (i, j), var in m._x.items() if var.X > 0.5}
        tour: List[int] = [1]
        while succ[tour[-1]] != 1:
            tour.append(succ[tour[-1]])
        assert sorted(tour) == list(range(1, n + 1))  # visits every city once, no subtour
