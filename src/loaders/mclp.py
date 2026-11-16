import math
import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)

# Generator settings of the reference (data/data_prepare_mclp_rand.py)
WEIGHT_RANGE: Tuple[int, int] = (50, 500)  # population a[i] of a demand node
SITE_FRACTION: float = 0.01  # P = ceil(1% of the candidate sites)

Point = Tuple[float, float]


def coverage_sets(demands: List[Point], sites: List[Point], radius: float, rng: random.Random) -> Dict[int, List[int]]:
    """
    Ni[i] = the sites (numbered from 1) within `radius` of demand node i (numbered from 1), in
    increasing order. A node with no site in range gets one random site, as in the reference (every
    node can then be covered). Sites are bucketed in a grid of cells of side `radius`, so only the
    3 x 3 cells around a node are searched (the reference computes every distance).
    """
    cell: float = radius
    grid: Dict[Tuple[int, int], List[int]] = {}
    for j, (sx, sy) in enumerate(sites, start=1):
        grid.setdefault((int(sx // cell), int(sy // cell)), []).append(j)
    r2: float = radius ** 2
    Ni: Dict[int, List[int]] = {}
    for i, (dx, dy) in enumerate(demands, start=1):
        cx, cy = int(dx // cell), int(dy // cell)
        near: List[int] = [j for gx in (cx - 1, cx, cx + 1) for gy in (cy - 1, cy, cy + 1)
                           for j in grid.get((gx, gy), [])
                           if (sites[j - 1][0] - dx) ** 2 + (sites[j - 1][1] - dy) ** 2 <= r2]
        Ni[i] = sorted(near) if near else [rng.randint(1, len(sites))]
    return Ni


class LoaderMCLP(Loader):
    """Loader for the Maximal Covering Location Problem (open P sites, maximize the covered population)."""

    problem: Problem = Problem.MCLP

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_demands: int,
        max_demands: int,
        min_sites: int,
        max_sites: int,
        min_radius: float,
        max_radius: float,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances, as the reference does. For each one: draw the
        number of demand nodes and of candidate sites uniformly in their ranges, the coverage
        radius uniformly in [min_radius, max_radius], P = ceil(SITE_FRACTION * sites), then:
          - demand nodes and sites uniform in the unit square;
          - populations a[i] uniform in WEIGHT_RANGE;
          - Ni[i] = the sites within the radius of node i (coverage_sets).
        Names are "mclp_s<seed>_<idx>_<I>x<J>_P<P>_r<radius>".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n_demands: int = rng.randint(min_demands, max_demands)
            n_sites: int = rng.randint(min_sites, max_sites)
            radius: float = rng.uniform(min_radius, max_radius)
            P: int = max(1, math.ceil(SITE_FRACTION * n_sites))
            demands: List[Point] = [(rng.random(), rng.random()) for _ in range(n_demands)]
            sites: List[Point] = [(rng.random(), rng.random()) for _ in range(n_sites)]
            a: Dict[int, int] = {i: rng.randint(*WEIGHT_RANGE) for i in range(1, n_demands + 1)}
            instances.append({
                "name": f"mclp_s{seed}_{idx}_{n_demands}x{n_sites}_P{P}_r{radius:.3f}",
                "args": {"I": list(range(1, n_demands + 1)), "J": list(range(1, n_sites + 1)),
                         "Ni": coverage_sets(demands, sites, radius, rng), "a": a, "P": P},
            })
        return instances
