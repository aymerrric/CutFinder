"""Fixtures shared by all test files (pytest loads this file automatically)."""
import itertools
from typing import List

import pytest

from loaders.base import Instance
from loaders.tsp import Costs, LoaderTSP


@pytest.fixture
def tsp_instances() -> List[Instance]:
    """A few small TSP instances: small enough to be checked by brute force."""
    return LoaderTSP().generate(4, seed=0, min_cities=5, max_cities=7)


def brute_force_tsp(n: int, c: Costs) -> float:
    """Optimal tour cost by trying every tour starting at city 1 ((n-1)! tours)."""
    best: float = float("inf")
    for perm in itertools.permutations(range(2, n + 1)):
        tour = (1,) + perm + (1,)
        best = min(best, sum(c[a, b] for a, b in zip(tour, tour[1:])))
    return best
