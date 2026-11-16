# SAGE-MCND: the cut family the paper selected for the MCND (papers/sage_evocut_tmlr2026.pdf,
# Appendix F.2, Eq. SAGE-MCND), written for our model (models/mcnd.py: x[i, j, k] flows, y[i, j]
# open arcs, u capacities, d demands, O / D origins and destinations). Run by models/transforms.with_cut.
# RECONSTRUCTION: the paper takes node sets S from "concentric breadth-first search zones expanded
# from commodity origins" and "sampled small bundles" B of the commodities crossing the cut, without
# giving the sampling. Here: every BFS zone (1, 2, ... hops) around every origin that is not the
# whole network, and as bundles all the crossing commodities plus each of the 3 largest alone.
# For each (S, B), with delta+(S) the arcs leaving S and D_B the total demand of B:
#   sum_{delta+(S)} min(u[i, j], D_B) y[i, j] >= D_B      (knapsack-cover cut-set)
#   sum_{delta+(S)} y[i, j] >= q_{S,B}                     (q: fewest largest capacities covering D_B)
y, u, d, O, D, N, A, K = m._y, m._u, m._d, m._O, m._D, m._N, m._A, m._K
successors = {i: [] for i in N}
for (i, j) in A:
    successors[i].append(j)
zones = set()
for origin in sorted(set(O.values())):
    hops = {origin: 0}
    frontier = [origin]
    while frontier:
        nxt = []
        for i in frontier:
            for j in successors[i]:
                if j not in hops:
                    hops[j] = hops[i] + 1
                    nxt.append(j)
        frontier = nxt
    for r in range(max(hops.values()) + 1):
        zone = frozenset(v for v, h in hops.items() if h <= r)
        if len(zone) < len(N):
            zones.add(zone)
row = 0
for zone in sorted(zones, key=lambda z: (len(z), sorted(z))):
    leaving = [(i, j) for (i, j) in A if i in zone and j not in zone]
    crossing = [k for k in K if O[k] in zone and D[k] not in zone]
    if not leaving or not crossing:
        continue
    bundles = [crossing] + [[k] for k in sorted(crossing, key=lambda k: -d[k])[:3]]
    for bundle in bundles:
        demand = sum(d[k] for k in bundle)
        m.addConstr(gp.quicksum(min(u[a], demand) * y[a] for a in leaving) >= demand, name=f"sage_cover[{row}]")
        capacities = sorted((u[a] for a in leaving), reverse=True)
        q = next((n for n in range(1, len(capacities) + 1) if sum(capacities[:n]) >= demand), None)
        if q is not None:
            m.addConstr(gp.quicksum(y[a] for a in leaving) >= q, name=f"sage_count[{row}]")
        row += 1
