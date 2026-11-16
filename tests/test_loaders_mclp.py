import math
import random
from typing import List, Tuple

from loaders.mclp import LoaderMCLP, SITE_FRACTION, WEIGHT_RANGE, coverage_sets

SIZES = dict(min_demands=50, max_demands=80, min_sites=100, max_sites=300, min_radius=0.1, max_radius=0.2)


def test_generate_follows_the_reference_settings(tmp_path) -> None:
    loader: LoaderMCLP = LoaderMCLP(data_dir=str(tmp_path))
    instances = loader.generate(4, seed=5, **SIZES)
    assert loader.generate(4, seed=5, **SIZES) == instances  # reproducible
    assert len({inst["name"] for inst in instances}) == 4
    for inst in instances:
        args = inst["args"]
        I, J = args["I"], args["J"]
        assert 50 <= len(I) <= 80 and 100 <= len(J) <= 300
        assert I == list(range(1, len(I) + 1)) and J == list(range(1, len(J) + 1))
        assert args["P"] == math.ceil(SITE_FRACTION * len(J))
        assert all(WEIGHT_RANGE[0] <= args["a"][i] <= WEIGHT_RANGE[1] for i in I)
        assert set(args["Ni"]) == set(I) and all(args["Ni"][i] and set(args["Ni"][i]) <= set(J) for i in I)
        assert inst["name"].startswith(f"mclp_s5_") and f"_{len(I)}x{len(J)}_P{args['P']}_r" in inst["name"]


def test_coverage_sets_match_all_distances() -> None:
    rng = random.Random(1)
    demands: List[Tuple[float, float]] = [(rng.random(), rng.random()) for _ in range(200)]
    sites: List[Tuple[float, float]] = [(rng.random(), rng.random()) for _ in range(150)]
    radius: float = 0.07
    Ni = coverage_sets(demands, sites, radius, random.Random(0))
    for i, (dx, dy) in enumerate(demands, start=1):
        near = [j for j, (sx, sy) in enumerate(sites, start=1) if (sx - dx) ** 2 + (sy - dy) ** 2 <= radius ** 2]
        if near:
            assert Ni[i] == near
        else:
            assert len(Ni[i]) == 1  # one random site, as in the reference
