from typing import List

from loaders.base import Instance
from loaders.jssp import TIME_RANGE, LoaderJSSP

SIZES = {"min_jobs": 3, "max_jobs": 8, "min_machines": 2, "max_machines": 5}


def test_generate_respects_bounds_and_format() -> None:
    instances: List[Instance] = LoaderJSSP(data_dir="/tmp").generate(5, seed=3, **SIZES)
    for idx, inst in enumerate(instances):
        a = inst["args"]
        n_jobs, n_machines = a["n_jobs"], a["n_machines"]
        assert 3 <= n_jobs <= 8 and 2 <= n_machines <= 5
        assert len(a["times"]) == len(a["machines"]) == n_jobs
        for times, machines in zip(a["times"], a["machines"]):
            assert len(times) == n_machines and all(TIME_RANGE[0] <= t <= TIME_RANGE[1] for t in times)
            assert sorted(machines) == list(range(n_machines))  # every machine once, in a random order
        assert inst["name"] == f"jssp_s3_{idx}_{n_jobs}x{n_machines}"


def test_generate_is_reproducible_with_seed() -> None:
    loader: LoaderJSSP = LoaderJSSP(data_dir="/tmp")
    assert loader.generate(2, seed=8, **SIZES) == loader.generate(2, seed=8, **SIZES)
    assert loader.generate(1, seed=8, **SIZES) != loader.generate(1, seed=9, **SIZES)


def test_loader_folder_is_the_problem_name() -> None:
    assert LoaderJSSP(data_dir="/data").folder == "/data/jssp"
