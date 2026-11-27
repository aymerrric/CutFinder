# Textbook strengthening of the capacitated facility location model (no LLM): the disaggregated
# ("strong") linking x[i, j] <= y[j], the total-capacity cover sum_j u[j] y[j] >= sum_i d[i], and
# x[i, j] = 0 when the demand d[i] exceeds the capacity u[j] (Cornuejols, Sridharan, Thizy, EJOR
# 1991). Written for our model (models/cwlp.py). Run by models/transforms.with_cut.
x, y, d, u, I, J = m._x, m._y, m._d, m._u, m._I, m._J
m.addConstrs((x[i, j] <= y[j] for (i, j) in x.keys()), name="tb_link")
m.addConstr(gp.quicksum(u[j] * y[j] for j in J) >= sum(d[i] for i in I), name="tb_capacity")
for (i, j) in x.keys():
    if d[i] > u[j]:
        x[i, j].UB = 0
