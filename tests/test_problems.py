import dataclasses
from typing import List

import gurobipy as gp
import pytest
from gurobipy import GRB

from loaders.base import Instance
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from problems import PROBLEMS, Problem, SetSpec, check_minimization, instance_set
from utils import parameters


def maximizing(**args: object) -> gp.Model:
    model: gp.Model = create_model(**args)
    model.ModelSense = GRB.MAXIMIZE
    return model


def small_instance() -> Instance:
    return LoaderTSP(data_dir="/tmp").generate(1, seed=1, min_cities=6, max_cities=6)[0]


def test_tsp_entry() -> None:
    tsp: Problem = PROBLEMS["tsp"]
    assert tsp.name == "tsp" and tsp.base is create_model and tsp.loader is LoaderTSP
    assert tsp.validation_sizes == {"min_cities": 15, "max_cities": 15}
    assert tsp.evaluation_sizes == {"min_cities": 60, "max_cities": 100}
    assert tsp.test_sizes == {"min_cities": 50, "max_cities": 100}


def test_sizes_are_described_for_the_llm() -> None:
    tsp: Problem = PROBLEMS["tsp"]
    assert tsp.describe_sizes(tsp.evaluation_sizes) == "60 to 100 cities"
    assert tsp.describe_sizes(tsp.validation_sizes) == "15 cities"


def test_every_problem_generates_with_its_sizes() -> None:
    for problem in PROBLEMS.values():
        for sizes in (problem.validation_sizes, problem.evaluation_sizes, problem.test_sizes, problem.hard_sizes,
                      problem.optimality_sizes):
            if sizes is not None:
                assert len(problem.loader("/tmp").generate(1, seed=0, **sizes)) == 1


def test_default_level_keeps_the_old_files_and_seeds() -> None:
    tsp: Problem = PROBLEMS["tsp"]
    assert instance_set(tsp, "evaluation") == SetSpec("evaluation.pkl", parameters.EVALUATION_SEED, tsp.evaluation_sizes)
    assert instance_set(tsp, "test", "default") == SetSpec("test.pkl", parameters.TEST_SEED, tsp.test_sizes)


def test_hard_level_has_its_own_files_seeds_and_larger_instances() -> None:
    tsp: Problem = PROBLEMS["tsp"]
    evaluation: SetSpec = instance_set(tsp, "evaluation", "hard")
    test: SetSpec = instance_set(tsp, "test", "hard")
    assert (evaluation.filename, test.filename) == ("evaluation_hard.pkl", "test_hard.pkl")
    assert evaluation.sizes == test.sizes == {"min_cities": 60, "max_cities": 200}
    # the default sizes are still drawn (a mix), and larger ones are added
    assert evaluation.sizes["min_cities"] <= tsp.evaluation_sizes["min_cities"]
    assert evaluation.sizes["max_cities"] > tsp.evaluation_sizes["max_cities"]
    # names hold the seed: no seed (nor its 19 replacements) is shared by two sets
    seeds: List[int] = sorted(parameters.SET_SEEDS.values()) + [parameters.VALIDATION_SEED, parameters.OPTIMALITY_SEED,
                                                               parameters.OPTIMALITY_TEST_SEED]
    assert all(b - a >= 20 for a, b in zip(sorted(seeds), sorted(seeds)[1:]))
    assert tsp.describe_sizes(evaluation.sizes) == "60 to 200 cities"


def test_instance_set_refuses_unknown_levels_and_problems_without_hard_sizes() -> None:
    with pytest.raises(ValueError, match="unknown instance level"):
        instance_set(PROBLEMS["tsp"], "evaluation", "huge")
    with pytest.raises(ValueError, match="unknown set"):
        instance_set(PROBLEMS["tsp"], "validation")
    with pytest.raises(ValueError, match="has no 'hard' instance level"):
        instance_set(PROBLEMS["jssp"], "test", "hard")


def test_only_minimization_is_accepted() -> None:
    check_minimization(PROBLEMS["tsp"], small_instance())  # the TSP minimizes: no error
    flipped: Problem = dataclasses.replace(PROBLEMS["tsp"], base=maximizing)
    with pytest.raises(ValueError, match="minimize the negated objective"):
        check_minimization(flipped, small_instance())


def test_every_problem_has_optimality_sizes_smaller_than_its_evaluation_sizes() -> None:
    assert PROBLEMS["imo6"].optimality_sizes is None  # only N = 7 fits the work range (problems.py)
    for problem in (p for p in PROBLEMS.values() if p.name != "imo6"):
        assert problem.optimality_sizes is not None and set(problem.optimality_sizes) == set(problem.evaluation_sizes)
        # no size larger, at least one smaller (min_cities, min_customers, min_jobs, min_arcs, ...)
        assert all(problem.optimality_sizes[k] <= problem.evaluation_sizes[k] for k in problem.evaluation_sizes)
        assert any(problem.optimality_sizes[k] < problem.evaluation_sizes[k] for k in problem.evaluation_sizes)


def test_imo6_entry() -> None:
    from loaders.imo6 import LoaderIMO6
    from models import imo6
    spec: Problem = PROBLEMS["imo6"]
    assert spec.loader is LoaderIMO6 and spec.base is imo6.create_model and spec.baseline_code == imo6.BASELINE_CODE
    assert spec.describe_sizes(spec.evaluation_sizes) == "N x N grids with N = 37 to 46"
    assert spec.describe_sizes({"min_n": 5, "max_n": 5}) == "N x N grids with N = 5"
    # the paper's split: development N = 37..46, test N = 3..36 (our review draws from 20..36)
    assert spec.evaluation_sizes["min_n"] > spec.test_sizes["max_n"] and spec.test_sizes["max_n"] <= 36
    assert [inst["args"]["N"] for inst in spec.validity_instances("unused")] == [2, 3, 4, 5, 6, 7, 8]


def test_gisp_entry() -> None:
    from loaders.gisp import LoaderGISP
    from models import gisp
    spec: Problem = PROBLEMS["gisp"]
    assert spec.loader is LoaderGISP and spec.base is gisp.create_model and spec.baseline_code == gisp.BASELINE_CODE
    assert spec.describe_sizes(spec.evaluation_sizes) == "150 to 175 nodes"
    assert spec.describe_sizes(spec.validation_sizes) == "40 nodes"
    assert len(spec.validity_instances("unused")) == 14


def test_corridor_entry() -> None:
    from loaders.corridor import LoaderCorridor, connectable
    from models import corridor
    spec: Problem = PROBLEMS["corridor"]
    assert spec.loader is LoaderCorridor and spec.base is corridor.create_model
    assert spec.baseline_code == corridor.BASELINE_CODE
    assert spec.describe_sizes(spec.evaluation_sizes) == "square lattices of 20 x 20 to 25 x 25 cells"
    assert spec.describe_sizes(spec.validation_sizes) == "square lattices of 7 x 7 cells"
    instances = spec.validity_instances("unused")
    assert len(instances) == 11 and all(connectable(**i["args"]) for i in instances)


def test_mcnd_entry() -> None:
    from loaders.mcnd import LoaderMCND
    from models import mcnd
    spec: Problem = PROBLEMS["mcnd"]
    assert spec.loader is LoaderMCND and spec.base is mcnd.create_model and spec.baseline_code == mcnd.BASELINE_CODE
    assert spec.describe_sizes(spec.evaluation_sizes) == "20 nodes, 220 to 300 arcs and 100 to 200 commodities"
    assert spec.describe_sizes(spec.validation_sizes) == "8 nodes, 25 arcs and 10 commodities"
