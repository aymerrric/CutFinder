# A cut known to be INVALID, kept as the control of prove_cut.py: no proof must be found.
# "No city is at position n." The n - 1 cities other than the depot need n - 1 increasing
# positions in 2..n along the tour (MTZ), so the last one is at position n: with this cut no tour
# is left, for every n. (The first control, "no triangle through the depot", FLARE Prop. F.3, is
# invalid only with 3 cities, which the Lean model excludes since 14 Nov: hn : 4 <= n.)
u, n = m._u, m._n
m.addConstrs((u[i] <= n - 1 for i in range(2, n + 1)), name="position_bound")
