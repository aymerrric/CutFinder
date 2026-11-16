import itertools
from typing import Dict

from gurobipy import GRB

from loaders import validity
from loaders.gisp import LoaderGISP
from models.gisp import BASELINE_CODE, create_model
from models.transforms import with_cut


def brute_force(args: Dict) -> float:
    """The best weight * |S| - cost * (removable edges inside S), over every node set S without a
    permanent edge inside (the model's objective is its negation)."""
    nodes, permanent, removable = args["nodes"], args["permanent"], args["removable"]
    best: float = 0
    for size in range(len(nodes) + 1):
        for chosen in itertools.combinations(nodes, size):
            inside = set(chosen)
            if any(i in inside and j in inside for (i, j) in permanent):
                continue
            removed: int = sum(i in inside and j in inside for (i, j) in removable)
            best = max(best, args["weight"] * size - args["cost"] * removed)
    return best


def optimum(model) -> float:
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    value: float = model.ObjVal
    model.dispose()
    return value


def test_the_model_finds_the_best_node_set(tmp_path) -> None:
    loader = LoaderGISP(data_dir=str(tmp_path))
    for seed in range(4):
        args = loader.generate(1, seed, min_nodes=10, max_nodes=12, edge_probability=0.4, alpha=0.5)[0]["args"]
        args["cost"] = 60  # removing an edge costs more than half a node: the trade-off matters
        m = create_model(**args)
        m.Params.OutputFlag = 0
        m.optimize()
        assert m.Status == GRB.OPTIMAL and m.ModelSense == GRB.MINIMIZE
        assert -m.ObjVal == brute_force(args)
        chosen = {i for i in args["nodes"] if m._x[i].X > 0.5}
        assert not any(i in chosen and j in chosen for (i, j) in args["permanent"])
        assert all(m._y[i, j].X > 0.5 for (i, j) in args["removable"] if i in chosen and j in chosen)
        m.dispose()


def test_the_data_is_on_the_model(tmp_path) -> None:
    args = LoaderGISP(data_dir=str(tmp_path)).generate(1, 0, min_nodes=20, max_nodes=20)[0]["args"]
    m = create_model(**args)
    assert (m._nodes, m._permanent, m._removable, m._weight, m._cost) == (
        args["nodes"], args["permanent"], args["removable"], 100, 1)
    assert len(m._x) == 20 and len(m._y) == len(args["removable"])
    assert m.NumConstrs == len(args["permanent"]) + len(args["removable"])
    m.dispose()


def test_the_seed_cut_keeps_the_optimum_of_the_validity_instances() -> None:
    for inst in validity.gisp_instances():
        base: float = optimum(create_model(**inst["args"]))
        assert optimum(with_cut(code=BASELINE_CODE, base=create_model, **inst["args"])) == base, inst["name"]


def test_the_validity_instances_have_the_expected_optima() -> None:
    expected: Dict[str, float] = {
        "gisp_empty_6": -600,  # every node
        "gisp_permanent_clique_6": -100,  # one node
        "gisp_removable_clique_6": -(600 - 15),  # every node, the 15 edges removed
        "gisp_permanent_cycle_5": -200,  # an odd cycle: 2 nodes
        "gisp_mixed_cycle_5": -299,  # 3 nodes, the removable edge removed
        "gisp_break_even_6": -5,  # 1 or 2 nodes: 5 = 10 - 5
        "gisp_bipartite_8": -400,  # one side
    }
    instances = {i["name"]: i for i in validity.gisp_instances()}
    for name, value in expected.items():
        assert optimum(create_model(**instances[name]["args"])) == value, name
        assert -value == brute_force(instances[name]["args"])


def test_the_seed_cut_lifts_the_lp_bound(tmp_path) -> None:
    args = LoaderGISP(data_dir=str(tmp_path)).generate(1, 4, min_nodes=60, max_nodes=60)[0]["args"]
    lp = [create_model(**args).relax(), with_cut(code=BASELINE_CODE, base=create_model, **args).relax()]
    base, seeded = (optimum(r) for r in lp)
    assert base == -100 * 60 / 2  # x = 1/2 everywhere, y = 0
    assert seeded > base + 1
