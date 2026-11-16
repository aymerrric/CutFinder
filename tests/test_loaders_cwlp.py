from typing import List

from loaders.base import Instance
from loaders.cwlp import CAPACITY_BUFFER_RANGE, DEMAND_RANGE, FIXED_COST_RANGE, UNIT_COST_RANGE, LoaderCWLP

SIZES = {"min_customers": 10, "max_customers": 20, "min_warehouses": 3, "max_warehouses": 6}


def test_generate_respects_bounds_and_format() -> None:
    instances: List[Instance] = LoaderCWLP(data_dir="/tmp").generate(5, seed=3, **SIZES)
    assert len(instances) == 5
    for inst in instances:
        a = inst["args"]
        assert set(a) == {"I", "J", "d", "u", "f", "c"}
        assert 10 <= len(a["I"]) <= 20 and 3 <= len(a["J"]) <= 6
        assert a["I"] == list(range(1, len(a["I"]) + 1)) and a["J"] == list(range(1, len(a["J"]) + 1))
        assert all(DEMAND_RANGE[0] <= a["d"][i] <= DEMAND_RANGE[1] for i in a["I"])
        assert all(FIXED_COST_RANGE[0] <= a["f"][j] <= FIXED_COST_RANGE[1] for j in a["J"])
        assert set(a["c"]) == {(i, j) for i in a["I"] for j in a["J"]}
        assert all(UNIT_COST_RANGE[0] * a["d"][i] <= a["c"][i, j] <= UNIT_COST_RANGE[1] * a["d"][i]
                   for (i, j) in a["c"])
        # total capacity = 1.1-1.3 x total demand (capacities rounded down, spread +-10%)
        ratio: float = sum(a["u"].values()) / sum(a["d"].values())
        assert CAPACITY_BUFFER_RANGE[0] * 0.9 <= ratio <= CAPACITY_BUFFER_RANGE[1] * 1.1
        assert inst["name"] == f"cwlp_s3_{instances.index(inst)}_{len(a['I'])}x{len(a['J'])}"


def test_generate_is_reproducible_with_seed() -> None:
    loader: LoaderCWLP = LoaderCWLP(data_dir="/tmp")
    assert loader.generate(2, seed=8, **SIZES) == loader.generate(2, seed=8, **SIZES)
    assert loader.generate(1, seed=8, **SIZES) != loader.generate(1, seed=9, **SIZES)


def test_loader_folder_is_the_problem_name() -> None:
    assert LoaderCWLP(data_dir="/data").folder == "/data/cwlp"
