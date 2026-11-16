# SAGE-RT-Disrupt: the cut family the paper selected for IMO6 (papers/sage_evocut_tmlr2026.pdf,
# Appendix F.6), written for our model (models/imo6.py: holes h[i, j], active strips x[a, b, i],
# starts s[a, b, i]; rows and columns 1..N, I = the intervals (a, b)). Run by models/transforms.with_cut.
# The paper gives the whole family, so this is a transcription, not a reconstruction:
#   (1) x[a, b, i] + sum_{l = a..b} h[i, l] <= 1                  an active strip avoids the row's hole
#   (2) sum_I x[., ., i] >= 2 - h[i, 1] - h[i, N]                 covered components of a row
#   (3) sum_I s[., ., i] >= 1                       (i >= 2)       a tile starts in every later row
#   (4) sum_{a <= j <= b} s[a, b, i] >= h[i - 1, j]  (i >= 2)       the vacated column is covered by a start
#   (5) j < k: sum_{a <= j <= b < k} s[a, b, i] >= sum_{a <= j, b >= k} x[a, b, i - 1] + h[i, k] - 1
#   (6) j > k: sum_{k < a <= j <= b} s[a, b, i] >= sum_{a <= k, b >= j} x[a, b, i - 1] + h[i, k] - 1
# (5) and (6): a strip of row i - 1 covering j and k must stop when k becomes the hole of row i, so
# cell (i, j) is covered by a new tile on j's side of k. Together they give the LP bound sum s >= N.
h, x, s, N, I = m._h, m._x, m._s, m._N, m._I
rows = range(1, N + 1)
m.addConstrs((x[a, b, i] + gp.quicksum(h[i, l] for l in range(a, b + 1)) <= 1 for (a, b) in I for i in rows),
             name="sage_strip_hole")
m.addConstrs((gp.quicksum(x[a, b, i] for (a, b) in I) >= 2 - h[i, 1] - h[i, N] for i in rows),
             name="sage_row_components")
m.addConstrs((gp.quicksum(s[a, b, i] for (a, b) in I) >= 1 for i in rows if i >= 2), name="sage_row_start")
covering = {j: [(a, b) for (a, b) in I if a <= j <= b] for j in rows}
m.addConstrs((gp.quicksum(s[a, b, i] for (a, b) in covering[j]) >= h[i - 1, j] for i in rows if i >= 2 for j in rows),
             name="sage_vacated")
for j in rows:
    for k in rows:
        if j == k:
            continue
        if j < k:
            new = [(a, b) for (a, b) in covering[j] if b < k]  # starts covering j, stopping before k
            old = [(a, b) for (a, b) in covering[j] if b >= k]  # strips covering j and k
        else:
            new = [(a, b) for (a, b) in covering[j] if a > k]  # starts covering j, beginning after k
            old = [(a, b) for (a, b) in covering[j] if a <= k]  # strips covering k and j
        for i in rows:
            if i >= 2:
                m.addConstr(gp.quicksum(s[a, b, i] for (a, b) in new)
                            >= gp.quicksum(x[a, b, i - 1] for (a, b) in old) + h[i, k] - 1,
                            name=f"sage_disrupt[{i},{j},{k}]")
