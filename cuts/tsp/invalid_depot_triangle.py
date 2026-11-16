# A cut known to be INVALID, kept as a control for prove_cut.py: no proof must be found.
# FLARE (Robbins et al. 2026, Proposition F.3, EvoCut's v2-EC2): no triangle through the depot.
# With 3 cities the only tours are triangles through the depot: 1 -> 2 -> 3 -> 1 gives 3 <= 2.
x, n = m._x, m._n
m.addConstrs((x[1, i] + x[i, 1] + x[1, j] + x[j, 1] + x[i, j] + x[j, i] <= 2
              for i in range(2, n + 1) for j in range(i + 1, n + 1)), name="depot_triangle")
