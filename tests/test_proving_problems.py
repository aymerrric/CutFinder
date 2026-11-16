import os

from loaders import validity
from loaders.cwlp import LoaderCWLP
from loaders.jssp import LoaderJSSP
from loaders.tsp import LoaderTSP
from problems import PROBLEMS
from proving.problems import LEAN_PROBLEMS, broken_assumptions
from utils import parameters


def test_every_lean_problem_is_a_problem_with_its_lean_file() -> None:
    for name, problem in LEAN_PROBLEMS.items():
        assert name in PROBLEMS and problem.name == name
        source: str = problem.source()
        assert f"namespace {problem.namespace}" in source and "def formulation : MILPFormulation" in source
        assert os.path.exists(os.path.join(parameters.LEAN_DIR, *problem.module.split(".")) + ".lean")


def test_generated_and_validity_instances_satisfy_the_assumptions(tmp_path) -> None:
    sets = {
        "tsp": LoaderTSP(str(tmp_path)).generate(3, 0, min_cities=5, max_cities=9) + validity.tsp_instances(),
        "cwlp": LoaderCWLP(str(tmp_path)).generate(3, 0, min_customers=10, max_customers=20, min_warehouses=3,
                                                   max_warehouses=5) + validity.cwlp_instances(),
        "jssp": LoaderJSSP(str(tmp_path)).generate(3, 0, min_jobs=3, max_jobs=5, min_machines=2,
                                                   max_machines=4) + validity.jssp_generated(),
    }
    for name, instances in sets.items():
        assert broken_assumptions(LEAN_PROBLEMS[name], instances) == {}, name


def test_broken_assumptions_are_named() -> None:
    tsp = LEAN_PROBLEMS["tsp"]
    assert tsp.assumptions({"n": 2, "c": {(1, 2): 1.0, (2, 1): -1.0}}) == [
        "hn: 2 cities, at least 4 are assumed", "hc_nn: a travel cost is negative"]
    cwlp = LEAN_PROBLEMS["cwlp"]
    args = {"I": [1], "d": {1: 0}, "u": {1: -1}, "f": {1: -2}, "c": {(1, 1): -3.0}}
    assert [b.split(":")[0] for b in cwlp.assumptions(args)] == ["hd_pos", "hu_nn", "hf_nn", "hc_nn"]
    assert cwlp.assumptions({"I": [], "d": {}, "u": {1: 5}, "f": {1: 1}, "c": {}}) == ["hnI: no customer"]
    jssp = LEAN_PROBLEMS["jssp"]
    args = {"n_jobs": 1, "n_machines": 2, "times": [[3, 0]], "machines": [[0, 0]]}
    assert [b.split(":")[0] for b in jssp.assumptions(args)] == ["htimes_pos", "hmachines_inj"]
    report = broken_assumptions(tsp, [{"name": "small", "args": {"n": 2, "c": {}}},
                                      {"name": "fine", "args": {"n": 4, "c": {}}}])
    assert list(report) == ["small"]
