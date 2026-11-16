import random
from typing import List, Optional

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)


def instance(N: int) -> Instance:
    """The IMO6 instance of an N x N grid (the only data of the problem is N)."""
    return {"name": f"imo6_{N}", "args": {"N": N}}


class LoaderIMO6(Loader):
    """Loader for IMO 2025 Problem 6 (tiles on an N x N grid, one hole per row and per column)."""

    problem: Problem = Problem.IMO6

    def generate(self, n_instances: int, seed: Optional[int] = None, *, min_n: int, max_n: int) -> List[Instance]:
        """
        `n_instances` grid sizes drawn WITHOUT replacement in [min_n, max_n] (an instance is only
        its N: two draws of the same N would be the same instance), in increasing order. Raises
        when the range holds fewer than `n_instances` sizes. Names are "imo6_<N>".
        Note: build_set replaces a refused instance by a new draw with the next seed, which can
        repeat an N already kept; choose ranges where the base model never finishes (all N >= 8
        under WORK_LIMIT, runs/analysis/imo6_sizes_20261113.txt), so that nothing is refused.
        """
        sizes: List[int] = list(range(min_n, max_n + 1))
        if n_instances > len(sizes):
            raise ValueError(f"{n_instances} IMO6 instances asked, but only {len(sizes)} sizes in [{min_n}, {max_n}]")
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        return [instance(N) for N in sorted(random.Random(seed).sample(sizes, n_instances))]
