from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

# IMO 2025 Problem 6 as a MILP: the paper's formulation RT-2DFlow (Appendix F.6, the same as its
# Pyomo code). The olympiad's answer is N + ceil(2 sqrt N) - 3 tiles (2112 for N = 2025). The LLM
# reads the source of create_model: its docstring must not name the olympiad nor give the answer
# (prompts/problems/imo6.yaml; tests/test_prompts.py checks it).
Strip = Tuple[int, int]  # (a, b): the contiguous columns a, a + 1, ..., b of one row (a <= b)

# Baseline cut (seed of the first generation, format example for the initializer): chosen weak,
# so that the strong families of the paper (Appendix F.6, SAGE-RT-Disrupt) are left to find.
# runs/analysis/imo6_sizes_20261113.txt: it closes 2.6-6.9% of the root gap (N = 10-30); "a tile
# starts in every row after the first" alone closes 69-78%, so it is NOT given.
BASELINE_IDEA: str = (
    "Count the strips of a row: the row has one hole, so its covered cells form two separate "
    "intervals when the hole is inside the row and one interval when it is in the first or last "
    "column, and an interval needs at least one strip: sum_(a, b) x[a, b, i] >= 2 - h[i, 1] - h[i, N]."
)
BASELINE_CODE: str = (
    "x, h, N, I = m._x, m._h, m._N, m._I\n"
    "m.addConstrs((gp.quicksum(x[a, b, i] for (a, b) in I) >= 2 - h[i, 1] - h[i, N] for i in range(1, N + 1)),\n"
    "             name=\"row_components\")"
)


def strips(N: int) -> List[Strip]:
    """Every contiguous column interval (a, b), 1 <= a <= b <= N."""
    return [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]


def create_model(N: int) -> gp.Model:
    """
    On an N x N grid, place axis-aligned rectangular tiles without overlap so that every row and
    every column has exactly one uncovered cell (a hole); minimize the number of tiles.

    A tile is a column interval (a, b) used on consecutive rows: a vertical strip. Rows and columns
    are numbered from 1.

    Variables (all binary; i a row, j a column, (a, b) in I = strips(N)):
        h[i, j]    : 1 if cell (i, j) is the hole of row i (and of column j)
        x[a, b, i] : 1 if on row i the columns a..b are covered by one tile (strip (a, b) active)
        s[a, b, i] : 1 if a tile of columns a..b starts at row i
        t[a, b, i] : 1 if a tile of columns a..b ends at row i
    Objective: the number of starts, i.e. of tiles.
    Constraints: one hole per row and per column; every cell is covered by exactly one active
    strip or is the hole; flow along each column interval: x[a, b, 1] = s[a, b, 1],
    x[a, b, i] - x[a, b, i - 1] = s[a, b, i] - t[a, b, i - 1] (i >= 2), x[a, b, N] = t[a, b, N].

    The variables and data are also stored on the model (m._h, m._x, m._s, m._t, m._N, m._I) so
    that code receiving only the model (e.g. a cut proposed by the LLM) can reach them.
    """
    m: gp.Model = gp.Model("imo6")
    rows: List[int] = list(range(1, N + 1))
    I: List[Strip] = strips(N)
    h: gp.tupledict = m.addVars(rows, rows, vtype=GRB.BINARY, name="h")
    keys: List[Tuple[int, int, int]] = [(a, b, i) for (a, b) in I for i in rows]
    x: gp.tupledict = m.addVars(keys, vtype=GRB.BINARY, name="x")
    s: gp.tupledict = m.addVars(keys, vtype=GRB.BINARY, name="s")
    t: gp.tupledict = m.addVars(keys, vtype=GRB.BINARY, name="t")

    m.setObjective(s.sum(), GRB.MINIMIZE)  # one start per tile

    m.addConstrs((h.sum(i, "*") == 1 for i in rows), name="row_hole")
    m.addConstrs((h.sum("*", j) == 1 for j in rows), name="column_hole")
    # cell (i, j): covered by exactly one active strip containing column j, or it is the hole
    m.addConstrs((gp.quicksum(x[a, b, i] for (a, b) in I if a <= j <= b) + h[i, j] == 1
                  for i in rows for j in rows), name="cover")
    m.addConstrs((x[a, b, 1] - s[a, b, 1] == 0 for (a, b) in I), name="top_flow")
    m.addConstrs((x[a, b, i] - x[a, b, i - 1] - s[a, b, i] + t[a, b, i - 1] == 0
                  for (a, b) in I for i in rows[1:]), name="mid_flow")
    m.addConstrs((x[a, b, N] - t[a, b, N] == 0 for (a, b) in I), name="bottom_flow")

    m._h, m._x, m._s, m._t, m._N, m._I = h, x, s, t, N, I
    m.update()  # apply the pending changes so the model can be inspected right away
    return m
