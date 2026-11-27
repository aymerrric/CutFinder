import functools
from typing import Dict, List

import gurobipy as gp
import pytest
from conftest import brute_force_tsp

from loaders.base import Instance
from models.transforms import fixed, relaxed, with_cut, with_stops
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


def test_cut_violations_measures_each_violated_cut_row() -> None:
    from models.transforms import cut_violations, read_violations
    import gurobipy as gp

    def base(**args: object) -> gp.Model:
        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVar(name="x")
        m.addConstr(x <= 10, name="cap")
        m._x = x
        m.update()
        return m

    def with_two_rows(**args: object) -> gp.Model:
        m = base()
        m.addConstr(m._x >= 7, name="too_strong")  # violated by 2 at x = 5
        m.addConstr(m._x <= 9)  # unnamed, satisfied
        return m

    model = cut_violations(with_two_rows, base, {"x": 5.0})
    model.optimize()
    values = {v.VarName: v.X for v in model.getVars()}
    assert read_violations(values) == {"too_strong": pytest.approx(2.0)}
    assert "cut_row_2" in [c.ConstrName for c in model.getConstrs()]  # the unnamed row, numbered


def test_with_stops_stops_at_a_good_enough_solution_or_a_high_enough_bound(tsp_instances: List[Instance]) -> None:
    args = tsp_instances[0]["args"]
    optimum: float = solve(create_model(**args)).ObjVal
    found: gp.Model = solve(with_stops(create_model, obj_stop=optimum + 1e-6, bound_stop=optimum + 1e-6, **args))
    assert found.ObjVal <= optimum + 1e-6 and found.Params.MIPGap == 0
    # forbid every optimal tour: the bound must pass the optimum, then the solve stops
    worse: gp.Model = with_stops(create_model, obj_stop=optimum + 1e-6, bound_stop=optimum + 1e-6, **args)
    worse.addConstr(worse.getObjective() >= optimum + 1)
    solve(worse)
    assert worse.Status == gp.GRB.CUTOFF and worse.SolCount == 0  # Cutoff: no worse optimum is proved
