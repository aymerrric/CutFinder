import random
from typing import List, Optional, Tuple

from loaders.base import Instance, Loader, Problem

MAX_INSTANCE_SEED: int = 2**31  # seeds drawn when none is given are in [0, MAX_INSTANCE_SEED)
TIME_RANGE: Tuple[int, int] = (5, 20)  # processing time of an operation (reference ptime_range)


class LoaderJSSP(Loader):
    """Loader for the Job Shop Scheduling Problem (minimize the makespan)."""

    problem: Problem = Problem.JSSP

    def generate(
        self,
        n_instances: int,
        seed: Optional[int] = None,
        *,
        min_jobs: int,
        max_jobs: int,
        min_machines: int,
        max_machines: int,
    ) -> List[Instance]:
        """
        Generate `n_instances` random instances, as the reference does (uni_instance_gen): draw the
        number of jobs and of machines uniformly in their ranges; every job has one operation on
        every machine, in a random order (machines[j] is a random permutation of 0..n_machines-1),
        and each operation takes an integer time uniform in TIME_RANGE (times[j][k]: the k-th
        operation of job j). Jobs, positions and machines are numbered from 0.
        Names are "jssp_s<seed>_<idx>_<jobs>x<machines>".
        """
        if seed is None:
            seed = random.randrange(MAX_INSTANCE_SEED)
        rng: random.Random = random.Random(seed)  # local RNG: does not touch the global one
        instances: List[Instance] = []
        for idx in range(n_instances):
            n_jobs: int = rng.randint(min_jobs, max_jobs)
            n_machines: int = rng.randint(min_machines, max_machines)
            times: List[List[int]] = [[rng.randint(*TIME_RANGE) for _ in range(n_machines)] for _ in range(n_jobs)]
            machines: List[List[int]] = [rng.sample(range(n_machines), n_machines) for _ in range(n_jobs)]
            instances.append({
                "name": f"jssp_s{seed}_{idx}_{n_jobs}x{n_machines}",
                "args": {"n_jobs": n_jobs, "n_machines": n_machines, "times": times, "machines": machines},
            })
        return instances
