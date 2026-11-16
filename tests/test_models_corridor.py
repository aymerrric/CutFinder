import itertools
from typing import Dict, List, Set

from gurobipy import GRB

from loaders import validity
from loaders.corridor import LoaderCorridor, connectable
from models.corridor import BASELINE_CODE, create_model
from models.transforms import with_cut


def connected(chosen: Set[int], edges) -> bool:
    if not chosen:
        return False
    start = next(iter(chosen))
    seen, todo = {start}, [start]
    while todo:
        i = todo.pop()
        for (a, b) in edges:
            for u, v in ((a, b), (b, a)):
                if u == i and v in chosen and v not in seen:
                    seen.add(v)
                    todo.append(v)
    return seen == chosen


def brute_force(args: Dict) -> float:
    """The best utility over every connected cell set with every reserve, within the budget."""
    others: List[int] = [i for i in args["nodes"] if i not in args["reserves"]]
    best: float = float("-inf")
    for size in range(len(others) + 1):
        for extra in itertools.combinations(others, size):
            chosen = set(args["reserves"]) | set(extra)
            if sum(args["cost"][i] for i in chosen) <= args["budget"] + 1e-9 and connected(chosen, args["edges"]):
                best = max(best, sum(args["utility"][i] for i in chosen))
    return best


def optimum(model) -> float:
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    value: float = model.ObjVal
    model.dispose()
    return value


def test_the_model_finds_the_best_corridor(tmp_path) -> None:
    loader = LoaderCorridor(data_dir=str(tmp_path))
    checked: int = 0
    for seed in range(20):
        args = loader.generate(1, seed, min_order=3, max_order=4)[0]["args"]
        if not connectable(**args):
            continue
        m = create_model(**args)
        m.Params.OutputFlag = 0
        m.optimize()
        assert m.Status == GRB.OPTIMAL and m.ModelSense == GRB.MINIMIZE
        assert abs(-m.ObjVal - brute_force(args)) < 1e-6
        chosen = {i for i in args["nodes"] if m._x[i].X > 0.5}
        assert set(args["reserves"]) <= chosen and connected(chosen, args["edges"])
        m.dispose()
        checked += 1
    assert checked >= 6


def test_the_data_is_on_the_model(tmp_path) -> None:
    args = LoaderCorridor(data_dir=str(tmp_path)).generate(1, 0, min_order=5, max_order=5)[0]["args"]
    m = create_model(**args)
    assert (m._nodes, m._edges, m._reserves, m._cost, m._utility, m._budget) == (
        args["nodes"], args["edges"], args["reserves"], args["cost"], args["utility"], args["budget"])
    assert len(m._arcs) == 2 * len(args["edges"]) and len(m._x) == 25 and len(m._f) == 80
    assert m.NumVars == 25 + 80 + 1 and m.NumConstrs == 1 + 3 + 1 + 80 + 25
    m.dispose()


def test_the_validity_instances_have_the_expected_optima() -> None:
    expected: Dict[str, float] = {
        "corridor_path_6": -14,  # every cell
        "corridor_cheapest_3x3": -5,  # the cheapest corridor, nothing else affordable
        "corridor_everything_3x3": -9,
        "corridor_cycle_8": -62,  # the useful side
        "corridor_star_6": -21,  # the centre and two useful leaves
    }
    instances = {i["name"]: i for i in validity.corridor_instances()}
    for name, value in expected.items():
        assert optimum(create_model(**instances[name]["args"])) == value, name
        assert brute_force(instances[name]["args"]) == -value


def test_the_seed_cut_keeps_the_optimum_of_the_validity_instances() -> None:
    for inst in validity.corridor_instances():
        base: float = optimum(create_model(**inst["args"]))
        assert abs(optimum(with_cut(code=BASELINE_CODE, base=create_model, **inst["args"])) - base) < 1e-6, inst["name"]


def test_the_seed_cut_lifts_the_lp_bound(tmp_path) -> None:
    args = LoaderCorridor(data_dir=str(tmp_path)).generate(1, 2012, min_order=12, max_order=12)[0]["args"]
    base = optimum(create_model(**args).relax())
    seeded = optimum(with_cut(code=BASELINE_CODE, base=create_model, **args).relax())
    assert seeded > base + 1e-3
