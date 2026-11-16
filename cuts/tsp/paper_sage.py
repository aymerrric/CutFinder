# SAGE-TSP: the cut family the paper selected for the TSP (papers/sage_evocut_tmlr2026.pdf,
# Appendix F.1, Eq. SAGE-TSP), written for our model (models/tsp.py: x[i, j], u[i], city 1 the
# depot, u[1] = 1 and u[i] in [2, n]: the paper's numbering). Run by models/transforms.with_cut.
x, u, n = m._x, m._u, m._n
V = range(1, n + 1)
# short cycles: no 2-cycle, and no 3-cycle through the depot
m.addConstrs((x[i, j] + x[j, i] <= 1 for i in V for j in V if i < j), name="sage_two_cycle")
m.addConstrs((x[1, i] + x[i, 1] + x[1, j] + x[j, 1] + x[i, j] + x[j, i] <= 2
              for i in V for j in V if 1 < i < j), name="sage_depot_triangle")
# lifted MTZ (Desrochers-Laporte)
m.addConstrs((u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2
              for i in V for j in V if i != 1 and j != 1 and i != j), name="sage_lifted_mtz")
# position bounds anchored at the depot: 2 right after it, n right before it
m.addConstrs((u[i] >= 3 - x[1, i] + (n - 3) * x[i, 1] for i in V if i != 1), name="sage_u_low")
m.addConstrs((u[i] <= (n - 1) + x[i, 1] - (n - 3) * x[1, i] for i in V if i != 1), name="sage_u_up")
