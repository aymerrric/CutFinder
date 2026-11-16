import itertools
import math
from typing import List, Set, Tuple

import gurobipy as gp
import pytest
from gurobipy import GRB

from models.imo6 import BASELINE_CODE, create_model, strips
from models.transforms import with_cut


def brute_force(N: int) -> int:
    """The fewest tiles, by enumeration: every permutation of holes, then the fewest rectangles that
    exactly cover the other cells (greedy-free search over rectangles containing the first free cell)."""
    best: int = N * N
    for perm in itertools.permutations(range(N)):
        free: Set[Tuple[int, int]] = {(i, j) for i in range(N) for j in range(N) if perm[i] != j}
        best = min(best, cover(free, best))
    return best


def cover(free: Set[Tuple[int, int]], bound: int) -> int:
    if not free:
        return 0
    if bound <= 0:
        return bound + 1
    i, j = min(free)  # the first free cell must be the top-left corner of its tile
    best: int = bound + 1
    for i2 in range(i, max(r for r, _ in free) + 1):
        if (i2, j) not in free:
            break
        for j2 in range(j, max(c for _, c in free) + 1):
            cells = {(r, c) for r in range(i, i2 + 1) for c in range(j, j2 + 1)}
            if not cells <= free:
                break
            best = min(best, 1 + cover(free - cells, min(bound, best) - 1))
    return best


def solve(m: gp.Model, relax: bool = False) -> float:
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    model: gp.Model = m.relax() if relax else m
    model.Params.OutputFlag = 0
    model.optimize()
    assert model.Status == GRB.OPTIMAL
    return model.ObjVal


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_the_model_finds_the_fewest_tiles(N: int) -> None:
    optimum: float = solve(create_model(N))
    assert optimum == pytest.approx(brute_force(N))
    assert optimum == N + math.ceil(2 * math.sqrt(N)) - 3  # the olympiad's answer


def test_the_solution_is_a_tiling() -> None:
    N: int = 5
    m: gp.Model = create_model(N)
    solve(m)
    holes: List[Tuple[int, int]] = [(i, j) for i in range(1, N + 1) for j in range(1, N + 1) if m._h[i, j].X > 0.5]
    assert sorted(i for i, _ in holes) == list(range(1, N + 1)) and sorted(j for _, j in holes) == list(range(1, N + 1))
    covered = [(i, j) for (a, b, i) in m._x if m._x[a, b, i].X > 0.5 for j in range(a, b + 1)]
    assert len(covered) == len(set(covered)) == N * N - N and not set(covered) & set(holes)
    starts: float = sum(v.X for v in m._s.values())
    assert starts == pytest.approx(m.ObjVal)


def test_sizes_match_the_paper_formulation() -> None:
    N: int = 6
    m: gp.Model = create_model(N)
    n_strips: int = len(strips(N))
    assert n_strips == N * (N + 1) // 2
    assert m.NumVars == N * N + 3 * N * n_strips
    assert m.NumConstrs == 2 * N + N * N + (N + 1) * n_strips
    assert (m._N, len(m._I)) == (N, n_strips)


def test_the_seed_cut_keeps_the_optimum_and_lifts_the_lp() -> None:
    N: int = 5
    assert solve(with_cut(create_model, BASELINE_CODE, N=N)) == solve(create_model(N))
    assert solve(with_cut(create_model, BASELINE_CODE, N=N), relax=True) > solve(create_model(N), relax=True) + 0.5
