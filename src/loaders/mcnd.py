import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)

Arc = Tuple[int, int]

# Generator settings, measured on the paper's test set (CommaLab group R, r10-r18: 20 nodes,
# 120-318 arcs, 40-200 commodities; runs/analysis/mcnd_r_set_20261113.txt, learnt_results.md 6.1):
UNIT_COST_RANGE: Tuple[int, int] = (10, 100)  # c[i, j] of every R instance
DEMAND_RANGE: Tuple[int, int] = (10, 100)  # R: 13-96 (mean about 54)
# capacity of an arc off the cycle, in mean demands, times the instance's tightness. R: the loose
# median is 290-780 units (5-15 mean demands, more with more arcs), not a share of the total
# demand; the medium level halves it, the tight one divides it by 7-8
CAPACITY_DEMANDS_RANGE: Tuple[float, float] = (2.5, 15.0)
TIGHTNESS_LEVELS: Tuple[float, ...] = (1.0, 0.5, 0.125)
# fixed cost / unit cost of an arc, per commodity, times the instance's fixed-cost level. R: the
# median ratio is 0.53 x the number of commodities at the first level (21, 51, 104 for 40, 100,
# 200 commodities), x5 and x10 at the other two
FIXED_RATIO_PER_COMMODITY_RANGE: Tuple[float, float] = (0.35, 0.75)
FIXED_COST_LEVELS: Tuple[int, ...] = (1, 5, 10)


class LoaderMCND(Loader):
    """Loader for the Multi-Commodity capacitated fixed-charge Network Design problem."""

    problem: Problem = Problem.MCND

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_nodes: int,
        max_nodes: int,
        min_arcs: int,
        max_arcs: int,
        min_commodities: int,
        max_commodities: int,
        tightness: Optional[float] = None,
        fixed_level: Optional[int] = None,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances shaped like the paper's test set (CommaLab R, made
        by Crainic's MULTIGEN). For each one: draw the numbers of nodes, arcs and commodities in
        their ranges (arcs between n and n (n - 1), commodities at most n (n - 1)), a capacity
        tightness in TIGHTNESS_LEVELS and a fixed-cost level in FIXED_COST_LEVELS, then:
          - commodities: distinct (origin, destination) pairs, demand uniform in DEMAND_RANGE;
          - a cycle through every node in a random order, whose arcs are uncapacitated (capacity =
            total demand): every commodity can follow it, so every instance is feasible. R has
            exactly n such arcs too, except its tight 120-arc instances (r10-r12 .7-.9), whose
            arcs are all capacitated;
          - the other arcs: distinct random ordered pairs, capacity = tightness * mean demand *
            uniform(CAPACITY_DEMANDS_RANGE) (at least 1);
          - every arc: unit cost uniform in UNIT_COST_RANGE, fixed cost = unit cost * number of
            commodities * uniform(FIXED_RATIO_PER_COMMODITY_RANGE) * fixed-cost level.
        `tightness` / `fixed_level`: the same level for every instance instead of a random one
        (validity instances). Nodes and commodities are numbered from 1. Names are
        "mcnd_s<seed>_<idx>_<n>n<arcs>a<k>k".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n: int = rng.randint(min_nodes, max_nodes)
            pairs: List[Arc] = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
            n_arcs: int = min(max(rng.randint(min_arcs, max_arcs), n), len(pairs))
            n_commodities: int = min(rng.randint(min_commodities, max_commodities), len(pairs))
            tight: float = rng.choice(TIGHTNESS_LEVELS) if tightness is None else tightness
            fixed: int = rng.choice(FIXED_COST_LEVELS) if fixed_level is None else fixed_level

            order: List[int] = rng.sample(range(1, n + 1), n)
            cycle: List[Arc] = [(order[p], order[(p + 1) % n]) for p in range(n)]
            others: List[Arc] = rng.sample(sorted(set(pairs) - set(cycle)), n_arcs - n)
            arcs: List[Arc] = sorted(cycle + others)
            commodities: List[int] = list(range(1, n_commodities + 1))
            od: List[Arc] = rng.sample(pairs, n_commodities)
            d: Dict[int, int] = {k: rng.randint(*DEMAND_RANGE) for k in commodities}
            total: int = sum(d.values())

            mean_demand: float = total / n_commodities
            on_cycle: set = set(cycle)
            u: Dict[Arc, int] = {
                a: total if a in on_cycle
                else max(1, round(tight * mean_demand * rng.uniform(*CAPACITY_DEMANDS_RANGE)))
                for a in arcs
            }
            c: Dict[Arc, int] = {a: rng.randint(*UNIT_COST_RANGE) for a in arcs}
            f: Dict[Arc, int] = {
                a: round(c[a] * n_commodities * rng.uniform(*FIXED_RATIO_PER_COMMODITY_RANGE) * fixed)
                for a in arcs
            }
            instances.append({
                "name": f"mcnd_s{seed}_{idx}_{n}n{n_arcs}a{n_commodities}k",
                "args": {"N": list(range(1, n + 1)), "A": arcs, "K": commodities, "c": c, "f": f, "u": u,
                         "d": d, "O": {k: od[k - 1][0] for k in commodities},
                         "D": {k: od[k - 1][1] for k in commodities}},
                "tightness": tight, "fixed_level": fixed,
            })
        return instances
