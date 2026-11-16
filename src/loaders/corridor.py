import heapq
import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)

# The settings of the corridor generator (Dilkina, Sabharwal and Gomes, corGenerator.c, as
# Distributional MIPLIB runs it in Corridor/generate.py): a square lattice, "2f+random" reserves
# (the upper left and lower right corners plus random ones, 3 in all), "uncorrelated" costs and
# utilities uniform on 1..L and 1..D, free reserves, a budget of 20-30% of the total cost.
N_RESERVES: int = 3
COST_MAX: int = 100  # L
UTILITY_MAX: int = 100  # D
BUDGET_FRACTIONS: Tuple[float, ...] = (0.2, 0.225, 0.25, 0.275, 0.3)

Edge = Tuple[int, int]  # (i, j) with i < j, an undirected lattice edge


def lattice(order: int) -> List[Edge]:
    """The edges of the order x order grid; node i is the cell (i // order, i % order)."""
    edges: List[Edge] = []
    for i in range(order * order):
        row, column = divmod(i, order)
        if column + 1 < order:
            edges.append((i, i + 1))
        if row + 1 < order:
            edges.append((i, i + order))
    return edges


def connectable(nodes: List[int], edges: List[Edge], reserves: List[int], cost: Dict[int, int], budget: float,
                **_: object) -> bool:
    """A sufficient condition for feasibility: the cheapest paths (node costs) from the first
    reserve to each other reserve fit in the budget together. (Their union connects every reserve;
    it may cost less than the sum, so a False does not prove the instance infeasible.)"""
    near: Dict[int, List[int]] = {i: [] for i in nodes}
    for (i, j) in edges:
        near[i].append(j)
        near[j].append(i)
    best: Dict[int, float] = {reserves[0]: cost[reserves[0]]}
    queue: List[Tuple[float, int]] = [(cost[reserves[0]], reserves[0])]
    while queue:
        d, i = heapq.heappop(queue)
        if d > best[i]:
            continue
        for j in near[i]:
            if d + cost[j] < best.get(j, float("inf")):
                best[j] = d + cost[j]
                heapq.heappush(queue, (best[j], j))
    if any(t not in best for t in reserves):
        return False
    return sum(best[t] for t in reserves[1:]) <= budget


class LoaderCorridor(Loader):
    """Loader for the wildlife corridor design problem (a connected subgraph with node costs and utilities)."""

    problem: Problem = Problem.CORRIDOR

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_order: int,
        max_order: int,
        n_reserves: int = N_RESERVES,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances as the corridor generator does: for each one, the
        lattice order uniform in [min_order, max_order]; reserves at the two opposite corners and
        n_reserves - 2 other random cells; every other cell with a cost uniform on 1..COST_MAX
        (reserves cost 0) and every cell with a utility uniform on 1..UTILITY_MAX; the budget a
        fraction of the total cost drawn from BUDGET_FRACTIONS. Our own random generator: the same
        distribution, not the same lattices. Names are "corridor_s<seed>_<idx>_o<order>_b<budget %>".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            order: int = rng.randint(min_order, max_order)
            n: int = order * order
            reserves: List[int] = [0, n - 1] + rng.sample(range(1, n - 1), n_reserves - 2)
            cost: Dict[int, int] = {i: 0 if i in reserves else rng.randint(1, COST_MAX) for i in range(n)}
            utility: Dict[int, int] = {i: rng.randint(1, UTILITY_MAX) for i in range(n)}
            fraction: float = rng.choice(BUDGET_FRACTIONS)
            instances.append({
                "name": f"corridor_s{seed}_{idx}_o{order}_b{round(100 * fraction, 1):g}",
                "args": {"nodes": list(range(n)), "edges": lattice(order), "reserves": sorted(reserves),
                         "cost": cost, "utility": utility, "budget": fraction * sum(cost.values())},
            })
        return instances
