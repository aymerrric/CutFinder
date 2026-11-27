"""The paper's selected cuts (cuts/<problem>/paper_sage.py) are valid: on small real instances,
the model with the cut has the same optimum as without it."""
import functools
import os

import pytest

from loaders import benchmarks
from loaders.base import Instance
from models.transforms import with_cut
from problems import PROBLEMS
from solvers.solver import SeedMode, SolveResult, Solver

ROOT: str = os.path.dirname(os.path.dirname(__file__))
DATA: str = os.path.join(ROOT, "tests", "data")


def paper_cut(problem: str) -> str:
    with open(os.path.join(ROOT, "cuts", problem, "paper_sage.py")) as f:
        return f.read()


def small_cwlp() -> Instance:
    """cap61 cut to its first 12 customers and 5 warehouses (optimum found in a second)."""
    with open(os.path.join(DATA, "beasley", "cap61")) as f:
        a = benchmarks.parse_beasley(f.read(), "cap61")["args"]
    I, J = a["I"][:12], a["J"][:5]
    d = {i: a["d"][i] for i in I}
    cap: float = round(max(sum(d.values()) / 2.5, 1.1 * max(d.values())))  # tight: 3 of 5 open
    u = {j: cap for j in J}
    return {"name": "cap61_small", "args": {"I": I, "J": J, "d": d, "u": u,
                                            "f": {j: a["f"][j] for j in J},
                                            "c": {(i, j): a["c"][i, j] for i in I for j in J}}}


def small_jssp() -> Instance:
    """ta01 cut to its first 4 jobs and 4 operations each (machines renumbered 0..3)."""
    with open(os.path.join(DATA, "taillard", "ta01")) as f:
        a = benchmarks.parse_taillard(f.read(), "ta01")["args"]
    jobs = 4
    machines = [[k for k in range(4)] for _ in range(jobs)]
    machines = [row[j % 4:] + row[:j % 4] for j, row in enumerate(machines)]
    return {"name": "ta01_small", "args": {"n_jobs": jobs, "n_machines": 4, "machines": machines,
                                           "times": [a["times"][j][:4] for j in range(jobs)]}}


def small_mcnd() -> Instance:
    """CommaLab r01.1 (10 nodes, 35 arcs, 10 commodities), outside the paper's test set."""
    with open(os.path.join(DATA, "commalab", "r01.1.dow")) as f:
        return benchmarks.parse_dow(f.read(), "r01.1")


def small_tsp() -> Instance:
    import gzip
    with gzip.open(os.path.join(DATA, "tsplib", "ulysses16.tsp.gz"), "rt") as f:
        return benchmarks.parse_tsplib(f.read())


@pytest.mark.parametrize("problem,make,optimum", [
    ("tsp", small_tsp, 6859.0),  # TSPLIB's published optimum of ulysses16
    ("cwlp", small_cwlp, None),
    ("jssp", small_jssp, None),
    ("mcnd", small_mcnd, 74079.0),  # its optimum in runs/analysis/mcnd_validity_set_20261113.txt
    ("imo6", lambda: {"name": "imo6_5", "args": {"N": 5}}, 7.0),  # 5 + ceil(2 sqrt 5) - 3
])
def test_paper_cut_keeps_the_optimum(problem: str, make, optimum, tmp_path) -> None:
    spec = PROBLEMS[problem]
    solver = Solver(time_limit=120, threads=2, seed=0, seed_mode=SeedMode.SET, log_dir=str(tmp_path))
    solver.add_formulation("base", spec.base)
    solver.add_formulation("paper", functools.partial(with_cut, base=spec.base, code=paper_cut(problem)))
    instance: Instance = make()
    base: SolveResult = solver.solve(instance, "base")
    cut: SolveResult = solver.solve(instance, "paper")
    assert base.status == "OPTIMAL" and cut.status == "OPTIMAL"
    assert cut.obj_val == pytest.approx(base.obj_val)
    if optimum is not None:
        assert base.obj_val == pytest.approx(optimum)


def test_the_mcnd_paper_cut_tightens_the_lp() -> None:
    from models.mcnd import create_model
    inst: Instance = small_mcnd()
    lp = lambda model: (setattr(model.Params, "OutputFlag", 0), model.optimize(), model.ObjVal)[2]
    base: float = lp(create_model(**inst["args"]).relax())
    cut = with_cut(create_model, paper_cut("mcnd"), **inst["args"])
    assert any(c.ConstrName.startswith("sage_cover") for c in cut.getConstrs())
    assert lp(cut.relax()) > base + 1e-6


def test_the_imo6_paper_cut_lifts_the_lp_to_n() -> None:
    """The paper (Appendix F.6): with its family, the LP relaxation proves sum s >= N."""
    import gurobipy as gp
    from models.imo6 import create_model
    N: int = 6
    values = []
    for model in (create_model(N), with_cut(create_model, paper_cut("imo6"), N=N)):
        relaxed: gp.Model = model.relax()
        relaxed.Params.OutputFlag = 0
        relaxed.optimize()
        values.append(relaxed.ObjVal)
    assert values[0] < 1.5 and values[1] >= N - 1e-6


def textbook_cut(problem: str) -> str:
    with open(os.path.join(ROOT, "cuts", problem, "textbook.py")) as f:
        return f.read()


def small_gisp() -> Instance:
    """A random graph of 14 nodes, half of its edges permanent (triangles: clique rows)."""
    import random
    rng = random.Random(3)
    edges = [(i, j) for i in range(14) for j in range(i + 1, 14) if rng.random() < 0.45]
    permanent = [e for e in edges if rng.random() < 0.5]
    removable = [e for e in edges if e not in permanent]
    return {"name": "gisp_small", "args": {"nodes": list(range(14)), "permanent": permanent,
                                           "removable": removable, "weight": 100, "cost": 1}}


@pytest.mark.parametrize("problem,make,optimum", [
    ("tsp", small_tsp, 6859.0),
    ("cwlp", small_cwlp, None),
    ("jssp", small_jssp, None),
    ("mcnd", small_mcnd, 74079.0),
    ("gisp", small_gisp, None),
])
def test_textbook_cut_keeps_the_optimum(problem: str, make, optimum, tmp_path) -> None:
    """cuts/<problem>/textbook.py (the literature's cuts, the benchmark's human reference)."""
    spec = PROBLEMS[problem]
    solver = Solver(time_limit=120, threads=2, seed=0, seed_mode=SeedMode.SET, log_dir=str(tmp_path))
    solver.add_formulation("base", spec.base)
    solver.add_formulation("textbook", functools.partial(with_cut, base=spec.base, code=textbook_cut(problem)))
    instance: Instance = make()
    base: SolveResult = solver.solve(instance, "base")
    cut: SolveResult = solver.solve(instance, "textbook")
    assert base.status == "OPTIMAL" and cut.status == "OPTIMAL"
    assert cut.obj_val == pytest.approx(base.obj_val)
    if optimum is not None:
        assert base.obj_val == pytest.approx(optimum)
