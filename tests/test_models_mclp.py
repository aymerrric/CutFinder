import itertools

from gurobipy import GRB

from loaders.mclp import LoaderMCLP
from models.mclp import create_model


def brute_force(args) -> float:
    """The largest covered population, over every choice of P sites."""
    best: float = 0
    for opened in itertools.combinations(args["J"], args["P"]):
        open_set = set(opened)
        best = max(best, sum(args["a"][i] for i in args["I"] if open_set & set(args["Ni"][i])))
    return best


def test_the_model_finds_the_best_cover(tmp_path) -> None:
    loader = LoaderMCLP(data_dir=str(tmp_path))
    for seed in range(4):
        args = loader.generate(1, seed, min_demands=30, max_demands=40, min_sites=12, max_sites=12,
                               min_radius=0.2, max_radius=0.35)[0]["args"]
        args["P"] = 3  # the reference's 1% would open one site only
        m = create_model(**args)
        m.Params.OutputFlag = 0
        m.optimize()
        assert m.Status == GRB.OPTIMAL and m.ModelSense == GRB.MINIMIZE
        assert -m.ObjVal == brute_force(args)
        opened = [j for j in args["J"] if m._x[j].X > 0.5]
        assert len(opened) == 3
        covered = {i for i in args["I"] if m._y[i].X > 0.5}
        assert all(set(args["Ni"][i]) & set(opened) for i in covered)
        m.dispose()


def test_the_data_is_on_the_model(tmp_path) -> None:
    args = LoaderMCLP(data_dir=str(tmp_path)).generate(1, 0, min_demands=10, max_demands=10, min_sites=5,
                                                       max_sites=5, min_radius=0.3, max_radius=0.3)[0]["args"]
    m = create_model(**args)
    assert (m._I, m._J, m._Ni, m._a, m._P) == (args["I"], args["J"], args["Ni"], args["a"], args["P"])
    assert len(m._x) == 5 and len(m._y) == 10 and m.NumConstrs == 11
    m.dispose()
