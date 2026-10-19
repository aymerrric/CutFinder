import math
import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

Coords = Dict[int, Tuple[float, float]]  # city -> (x, y)
Costs = Dict[Tuple[int, int], float]  # (i, j) -> travel cost from i to j

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)


class LoaderTSP(Loader):
    """Loader for the Traveling Salesman Problem."""

    problem: Problem = Problem.TSP

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_cities: int,
        max_cities: int,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances.
        For each one: draw n uniformly in [min_cities, max_cities], then draw the (x, y)
        coordinates of the n cities uniformly in the unit square (cities are numbered 1..n),
        then derive the Euclidean cost matrix.

        Names are "rand_s<seed>_<idx>_<n>": the seed makes them unique across instance sets
        (results, logs and caches are keyed by name). Without a seed, one is drawn at random,
        so the name still identifies the instance and it can be regenerated.
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n: int = rng.randint(min_cities, max_cities)  # both bounds included
            coords: Coords = {
                city: (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)) for city in range(1, n + 1)
            }
            instances.append(
                {
                    "name": f"rand_s{seed}_{idx}_{n}",
                    "coords": coords,
                    "args": {"n": n, "c": self._cost_matrix(coords)},  # create_model(n, c)
                }
            )
        return instances

    @staticmethod
    def _cost_matrix(coords: Coords) -> Costs:
        """Euclidean distance between every ordered pair of distinct cities (symmetric)."""
        costs: Costs = {}
        for i in coords:
            for j in coords:
                if i != j:
                    costs[i, j] = math.dist(coords[i], coords[j])
        return costs
