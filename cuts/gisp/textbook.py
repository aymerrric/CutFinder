# Textbook clique inequalities for the stable-set part of the GISP (no LLM; Padberg 1973): on
# the graph of the PERMANENT edges, at most one node of a clique can be chosen. Each permanent
# edge is extended greedily into a maximal clique (highest degree first); every distinct clique
# of 3 or more nodes gives sum_{i in C} x[i] <= 1. Valid for every feasible point.
# Written for our model (models/gisp.py). Run by models/transforms.with_cut.
x, permanent = m._x, m._permanent
adj = {}
for (i, j) in permanent:
    adj.setdefault(i, set()).add(j)
    adj.setdefault(j, set()).add(i)
cliques = set()
for (i, j) in permanent:
    clique = {i, j}
    candidates = sorted(adj[i] & adj[j], key=lambda v: (-len(adj[v]), v))
    for v in candidates:
        if all(v in adj[w] for w in clique):
            clique.add(v)
    if len(clique) >= 3:
        cliques.add(frozenset(clique))
for r, clique in enumerate(sorted(cliques, key=sorted)):
    m.addConstr(gp.quicksum(x[v] for v in clique) <= 1, name=f"tb_clique[{r}]")
