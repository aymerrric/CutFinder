import random
from typing import List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)

# Generator settings of Distributional MIPLIB (github.com/amf272/Distributional-MIPLIB-Generators,
# GISP/generate_erdos_renyi.py, after Ferber et al. 2022): an Erdos-Renyi graph G(n, p), each edge
# removable with probability ALPHA, the same weight for every node and the same cost for every
# removable edge. Their "easy" distribution: n = 150, p = 0.3; "hard": n = 175.
EDGE_PROBABILITY: float = 0.3
ALPHA: float = 0.25  # share of removable edges
NODE_WEIGHT: int = 100
EDGE_COST: int = 1

Edge = Tuple[int, int]  # (i, j) with i < j, nodes numbered from 0 as in the generator


def erdos_renyi(n: int, p: float, rng: random.Random) -> List[Edge]:
    """Every pair i < j of the n nodes is an edge with probability p (G(n, p))."""
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def partition_edges(edges: List[Edge], alpha: float, rng: random.Random) -> Tuple[List[Edge], List[Edge]]:
    """(permanent, removable): each edge is removable with probability alpha, as in the generator."""
    permanent: List[Edge] = []
    removable: List[Edge] = []
    for e in edges:
        (removable if rng.random() <= alpha else permanent).append(e)
    return permanent, removable


class LoaderGISP(Loader):
    """Loader for the Generalized Independent Set Problem (forest harvesting, Hochbaum and Pathria 1997)."""

    problem: Problem = Problem.GISP

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_nodes: int,
        max_nodes: int,
        edge_probability: float = EDGE_PROBABILITY,
        alpha: float = ALPHA,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances, as Distributional MIPLIB does: for each one, the
        number of nodes uniform in [min_nodes, max_nodes], a G(n, edge_probability) graph, each edge
        removable with probability alpha, weight NODE_WEIGHT per node and cost EDGE_COST per
        removable edge. Our own random generator (no numpy / networkx): same distribution, not the
        same graphs. Names are "gisp_s<seed>_<idx>_n<n>".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n: int = rng.randint(min_nodes, max_nodes)
            permanent, removable = partition_edges(erdos_renyi(n, edge_probability, rng), alpha, rng)
            instances.append({
                "name": f"gisp_s{seed}_{idx}_n{n}",
                "args": {"nodes": list(range(n)), "permanent": permanent, "removable": removable,
                         "weight": NODE_WEIGHT, "cost": EDGE_COST},
            })
        return instances
