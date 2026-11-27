# Textbook strengthening of the MTZ formulation (no LLM): the lifted MTZ constraints and the lifted
# position bounds of Desrochers and Laporte (Oper. Res. Letters 10, 1991), plus the 2-cycle
# elimination. Written for our model (models/tsp.py: x[i, j], u[i], city 1 the depot, u[1] = 1,
# u[i] in [2, n]). Valid for n >= 3. The paper's SAGE-TSP cut contains all of it, plus a depot
# triangle row (cuts/tsp/paper_sage.py). Run by models/transforms.with_cut.
x, u, n = m._x, m._u, m._n
V = range(1, n + 1)
if n >= 3:
    m.addConstrs((x[i, j] + x[j, i] <= 1 for i in V for j in V if i < j), name="tb_two_cycle")
    m.addConstrs((u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2
                  for i in V for j in V if i != 1 and j != 1 and i != j), name="tb_lifted_mtz")
    m.addConstrs((u[i] >= 3 - x[1, i] + (n - 3) * x[i, 1] for i in V if i != 1), name="tb_u_low")
    m.addConstrs((u[i] <= (n - 1) + x[i, 1] - (n - 3) * x[1, i] for i in V if i != 1), name="tb_u_up")
