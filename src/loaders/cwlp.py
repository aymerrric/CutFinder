import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)

# Generator settings of the reference (data/data_prepare_cwlp_rand.py)
DEMAND_RANGE: Tuple[int, int] = (50, 1_500)  # demand of a customer
FIXED_COST_RANGE: Tuple[int, int] = (7_000, 15_000)  # cost of opening a warehouse
UNIT_COST_RANGE: Tuple[float, float] = (5.0, 75.0)  # cost per unit of demand served by a warehouse
CAPACITY_BUFFER_RANGE: Tuple[float, float] = (1.1, 1.3)  # total capacity / total demand
CAPACITY_SPREAD: float = 0.1  # each capacity is the average one * uniform(0.9, 1.1)


class LoaderCWLP(Loader):
    """Loader for the Capacitated Warehouse Location Problem (unsplittable demands)."""

    problem: Problem = Problem.CWLP

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_customers: int,
        max_customers: int,
        min_warehouses: int,
        max_warehouses: int,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances, as the reference does. For each one: draw the
        number of customers and of warehouses uniformly in their ranges, a capacity buffer in
        CAPACITY_BUFFER_RANGE, then:
          - demands d[i] uniform in DEMAND_RANGE;
          - capacities u[j] = buffer * total demand / n_warehouses * uniform(0.9, 1.1);
          - opening costs f[j] uniform in FIXED_COST_RANGE;
          - c[i, j] = d[i] * (unit cost uniform in UNIT_COST_RANGE): the cost of serving all of
            customer i from warehouse j.
        Customers and warehouses are numbered from 1. Names are "cwlp_s<seed>_<idx>_<I>x<J>".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n_customers: int = rng.randint(min_customers, max_customers)
            n_warehouses: int = rng.randint(min_warehouses, max_warehouses)
            buffer: float = rng.uniform(*CAPACITY_BUFFER_RANGE)
            customers: List[int] = list(range(1, n_customers + 1))
            warehouses: List[int] = list(range(1, n_warehouses + 1))
            d: Dict[int, int] = {i: rng.randint(*DEMAND_RANGE) for i in customers}
            average: float = buffer * sum(d.values()) / n_warehouses
            u: Dict[int, int] = {
                j: int(average * rng.uniform(1 - CAPACITY_SPREAD, 1 + CAPACITY_SPREAD)) for j in warehouses
            }
            f: Dict[int, int] = {j: rng.randint(*FIXED_COST_RANGE) for j in warehouses}
            c: Dict[Tuple[int, int], float] = {
                (i, j): d[i] * rng.uniform(*UNIT_COST_RANGE) for i in customers for j in warehouses
            }
            instances.append({
                "name": f"cwlp_s{seed}_{idx}_{n_customers}x{n_warehouses}",
                "args": {"I": customers, "J": warehouses, "d": d, "u": u, "f": f, "c": c},
            })
        return instances
