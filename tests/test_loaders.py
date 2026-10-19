import math
import os
from pathlib import Path
from typing import List

import pytest

from loaders.base import Instance
from loaders.tsp import LoaderTSP


def test_generate_respects_bounds_and_format() -> None:
    instances: List[Instance] = LoaderTSP().generate(10, seed=1, min_cities=4, max_cities=9)
    assert len(instances) == 10
    for inst in instances:
        n: int = inst["args"]["n"]
        assert 4 <= n <= 9
        assert set(inst["coords"]) == set(range(1, n + 1))  # cities numbered 1..n
        assert len(inst["args"]["c"]) == n * (n - 1)  # every ordered pair i != j


def test_generate_is_reproducible_with_seed() -> None:
    a: List[Instance] = LoaderTSP().generate(3, seed=7, min_cities=5, max_cities=8)
    b: List[Instance] = LoaderTSP().generate(3, seed=7, min_cities=5, max_cities=8)
    assert a == b


def test_costs_are_symmetric_euclidean() -> None:
    inst: Instance = LoaderTSP().generate(1, seed=2, min_cities=6, max_cities=6)[0]
    c, coords = inst["args"]["c"], inst["coords"]
    for (i, j), cost in c.items():
        assert cost == c[j, i]
        assert cost == pytest.approx(math.dist(coords[i], coords[j]))


def test_get_generates_then_loads(tmp_path: Path) -> None:
    os.mkdir(os.path.join(tmp_path, "tsp"))  # the code never creates folders
    loader: LoaderTSP = LoaderTSP(data_dir=str(tmp_path))
    generated: List[Instance] = loader.get("x.pkl", n_instances=3, seed=0, min_cities=5, max_cities=6)
    assert os.path.exists(os.path.join(tmp_path, "tsp", "x.pkl"))
    loaded: List[Instance] = loader.get("x.pkl")  # file exists: no generation arguments needed
    assert loaded == generated


def test_get_without_file_nor_n_instances_raises(tmp_path: Path) -> None:
    os.mkdir(os.path.join(tmp_path, "tsp"))
    with pytest.raises(ValueError):
        LoaderTSP(data_dir=str(tmp_path)).get("missing.pkl")


def test_split_is_disjoint_and_complete() -> None:
    instances: List[Instance] = LoaderTSP().generate(10, seed=0, min_cities=4, max_cities=5)
    names_before: List[str] = [inst["name"] for inst in instances]
    first, second = LoaderTSP().split(instances, ratio=0.7, seed=0)
    assert len(first) == 7 and len(second) == 3
    assert sorted(inst["name"] for inst in first + second) == sorted(names_before)
    assert [inst["name"] for inst in instances] == names_before  # input list left untouched
