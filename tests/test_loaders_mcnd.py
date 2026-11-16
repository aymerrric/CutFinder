from typing import Dict, List, Set, Tuple

from loaders.base import Instance
from loaders.mcnd import (
    CAPACITY_DEMANDS_RANGE, DEMAND_RANGE, FIXED_COST_LEVELS, FIXED_RATIO_PER_COMMODITY_RANGE, TIGHTNESS_LEVELS,
    UNIT_COST_RANGE, LoaderMCND,
)

SIZES = {"min_nodes": 6, "max_nodes": 10, "min_arcs": 15, "max_arcs": 30, "min_commodities": 4, "max_commodities": 12}


def reachable(arcs: List[Tuple[int, int]], start: int, u: Dict[Tuple[int, int], int], at_least: int) -> Set[int]:
    """Nodes reachable from `start` through arcs of capacity >= at_least."""
    seen: Set[int] = {start}
    stack: List[int] = [start]
    while stack:
        i = stack.pop()
        for a, b in arcs:
            if a == i and b not in seen and u[a, b] >= at_least:
                seen.add(b)
                stack.append(b)
    return seen


def test_generate_respects_bounds_and_format() -> None:
    instances: List[Instance] = LoaderMCND(data_dir="/tmp").generate(5, seed=3, **SIZES)
    assert len(instances) == 5
    for idx, inst in enumerate(instances):
        a = inst["args"]
        assert set(a) == {"N", "A", "K", "c", "f", "u", "d", "O", "D"}
        n, arcs, commodities = len(a["N"]), a["A"], a["K"]
        assert 6 <= n <= 10 and 15 <= len(arcs) <= 30 and 4 <= len(commodities) <= 12
        assert a["N"] == list(range(1, n + 1)) and commodities == list(range(1, len(commodities) + 1))
        assert len(set(arcs)) == len(arcs) and all(i != j for i, j in arcs) and arcs == sorted(arcs)
        assert set(a["c"]) == set(a["f"]) == set(a["u"]) == set(arcs)
        assert all(UNIT_COST_RANGE[0] <= a["c"][arc] <= UNIT_COST_RANGE[1] for arc in arcs)
        assert all(DEMAND_RANGE[0] <= a["d"][k] <= DEMAND_RANGE[1] for k in commodities)
        od = [(a["O"][k], a["D"][k]) for k in commodities]
        assert len(set(od)) == len(od) and all(o != d for o, d in od)
        total: int = sum(a["d"].values())
        mean: float = total / len(commodities)
        uncapacitated = [arc for arc in arcs if a["u"][arc] == total]
        assert len(uncapacitated) >= n  # the cycle (an off-cycle arc never gets the whole demand here)
        others = [arc for arc in arcs if a["u"][arc] != total]
        assert all(1 <= a["u"][arc] <= round(inst["tightness"] * mean * CAPACITY_DEMANDS_RANGE[1]) for arc in others)
        assert inst["tightness"] in TIGHTNESS_LEVELS and inst["fixed_level"] in FIXED_COST_LEVELS
        low, high = FIXED_RATIO_PER_COMMODITY_RANGE
        assert all(round(a["c"][arc] * len(commodities) * low * inst["fixed_level"]) <= a["f"][arc]
                   <= round(a["c"][arc] * len(commodities) * high * inst["fixed_level"]) for arc in arcs)
        assert inst["name"] == f"mcnd_s3_{idx}_{n}n{len(arcs)}a{len(commodities)}k"


def test_every_instance_is_feasible_through_its_uncapacitated_cycle() -> None:
    for inst in LoaderMCND(data_dir="/tmp").generate(10, seed=11, **SIZES):
        a = inst["args"]
        total: int = sum(a["d"].values())
        for k in a["K"]:  # the whole demand fits on the uncapacitated arcs, from every origin
            assert a["D"][k] in reachable(a["A"], a["O"][k], a["u"], total)


def test_generate_is_reproducible_with_seed() -> None:
    loader: LoaderMCND = LoaderMCND(data_dir="/tmp")
    assert loader.generate(2, seed=8, **SIZES) == loader.generate(2, seed=8, **SIZES)
    assert loader.generate(1, seed=8, **SIZES) != loader.generate(1, seed=9, **SIZES)


def test_sizes_are_clipped_to_the_graph() -> None:
    inst: Instance = LoaderMCND(data_dir="/tmp").generate(
        1, seed=0, min_nodes=4, max_nodes=4, min_arcs=2, max_arcs=100, min_commodities=50, max_commodities=50)[0]
    assert 4 <= len(inst["args"]["A"]) <= 12 and len(inst["args"]["K"]) == 12  # n (n - 1) = 12 pairs


def test_loader_folder_is_the_problem_name() -> None:
    assert LoaderMCND(data_dir="/data").folder == "/data/mcnd"
