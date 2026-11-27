# Textbook strengthening of the arc-based network design model (no LLM): the strong linking
# x[i, j, k] <= min(u[i, j], d[k]) y[i, j] (Gendron, Crainic, Frangioni 1999; the "strong
# relaxation"). Written for our model (models/mcnd.py, net flow conservation). Safe: removing
# the flow cycles of an optimal solution keeps its cost (c >= 0) and satisfies the rows.
# Run by models/transforms.with_cut.
x, y, u, d, A, K = m._x, m._y, m._u, m._d, m._A, m._K
m.addConstrs((x[i, j, k] <= min(u[i, j], d[k]) * y[i, j] for (i, j) in A for k in K), name="tb_strong_link")
