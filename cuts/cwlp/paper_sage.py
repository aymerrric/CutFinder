# SAGE-CWLP: the cut family the paper selected for the CWLP (papers/sage_evocut_tmlr2026.pdf,
# Appendix F.3, Eq. SAGE-CWLP), written for our model (models/cwlp.py: x[i, j] customer i served
# by warehouse j, y[j] open, d demands, u capacities). Run by models/transforms.with_cut.
x, y, d, u, I, J = m._x, m._y, m._d, m._u, m._I, m._J
# global bounds: total capacity, and "slots" for the customers of demand >= v (the 8 largest v)
m.addConstr(gp.quicksum(u[j] * y[j] for j in J) >= sum(d[i] for i in I), name="sage_capacity")
for v in sorted({d[i] for i in I}, reverse=True)[:8]:
    m.addConstr(gp.quicksum(math.floor(u[j] / v) * y[j] for j in J) >= sum(1 for i in I if d[i] >= v),
                name=f"sage_slots[{v:g}]")
m.addConstrs((x[i, j] <= y[j] for i in I for j in J), name="sage_link")
m.addConstrs((x[i, j] <= 0 for i in I for j in J if d[i] > u[j]), name="sage_too_big")
by_demand = sorted(I, key=lambda i: -d[i])
for j in J:
    # conflict set: customers that pairwise cannot share warehouse j (starts with d > u/2, then
    # adds smaller customers that conflict with every customer already in it)
    clique = [i for i in by_demand if d[i] > u[j] / 2]
    members = set(clique)
    for i in by_demand:
        if i not in members and all(d[i] + d[k] > u[j] for k in clique):
            clique.append(i)
            members.add(i)
    if len(clique) >= 2:
        m.addConstr(gp.quicksum(x[i, j] for i in clique) <= y[j], name=f"sage_clique[{j}]")
    # at most 2 customers of demand > u/3 on warehouse j
    big = [i for i in I if d[i] > u[j] / 3]
    if len(big) >= 3:
        m.addConstr(gp.quicksum(x[i, j] for i in big) <= 2 * y[j], name=f"sage_third[{j}]")
