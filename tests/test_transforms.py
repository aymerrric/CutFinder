import functools
from typing import Dict, List

import gurobipy as gp
import pytest
from conftest import brute_force_tsp

from loaders.base import Instance
from models.transforms import fixed, relaxed, with_cut
from models.tsp import create_model

TWO_CYCLE: str = "x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)"


def solve(m: gp.Model) -> gp.Model:
    m.Params.OutputFlag = 0
    m.optimize()
    return m


def test_with_cut_runs_the_code_with_the_promised_names(tsp_instances: List[Instance]) -> None:
    args = tsp_instances[0]["args"]
    code: str = (
        "assert isinstance(m, gp.Model) and GRB.BINARY and n > 0 and c\n"
        "assert list(itertools.combinations([1, 2], 2)) and math.sqrt(4) == 2\n" + TWO_CYCLE
    )
    m: gp.Model = with_cut(create_model, code, **args)
    base: gp.Model = create_model(**args)
    n: int = args["n"]
    assert m.NumConstrs == base.NumConstrs + n * (n - 1) // 2  # one 2-cycle row per pair
    assert solve(m).ObjVal == pytest.approx(brute_force_tsp(**args))  # valid cut: same optimum


def test_with_cut_raises_the_code_errors(tsp_instances: List[Instance]) -> None:
    with pytest.raises(KeyError):
        with_cut(create_model, "m._x[1, 1]", **tsp_instances[0]["args"])


def test_relaxed_drops_integrality_and_keeps_names(tsp_instances: List[Instance]) -> None:
    args = tsp_instances[0]["args"]
    lp: gp.Model = relaxed(create_model, **args)
    assert lp.NumIntVars == 0 and lp.getVarByName("x[1,2]") is not None
    assert solve(lp).ObjVal <= brute_force_tsp(**args) + 1e-9  # a relaxation is a lower bound


def test_fixed_to_the_optimum_is_feasible_and_to_a_bad_point_infeasible(
    tsp_instances: List[Instance],
) -> None:
    args = tsp_instances[0]["args"]
    opt: gp.Model = solve(create_model(**args))
    tour: Dict[str, float] = {v.VarName: v.X for v in opt.getVars()}
    assert solve(fixed(create_model, tour, **args)).Status == gp.GRB.OPTIMAL
    bad: Dict[str, float] = {name: 1.0 for name in tour if name.startswith("x")}  # every arc used
    assert solve(fixed(create_model, bad, **args)).Status == gp.GRB.INFEASIBLE


def test_fixed_rounds_integers_and_skips_unknown_names(tsp_instances: List[Instance]) -> None:
    args = tsp_instances[0]["args"]
    m: gp.Model = fixed(create_model, {"x[1,2]": 0.9999999, "nope": 3.0}, **args)
    var: gp.Var = m.getVarByName("x[1,2]")
    assert var.LB == var.UB == 1.0


def test_partials_compose(tsp_instances: List[Instance]) -> None:
    build = functools.partial(relaxed, build=functools.partial(with_cut, base=create_model, code=TWO_CYCLE))
    assert solve(build(**tsp_instances[0]["args"])).Status == gp.GRB.OPTIMAL
