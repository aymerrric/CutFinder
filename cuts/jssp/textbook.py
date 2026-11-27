# Textbook bounds of the job shop (no LLM), for the disjunctive big-M model (models/jssp.py):
# - heads and tails: operation (j, k) starts after the work of its job before it, and the
#   makespan is at least its end plus the work of its job after it;
# - longest job: Cmax >= the total time of each job;
# - one-machine (Jackson / Carlier) bound: Cmax >= the load of each machine + the smallest head
#   + the smallest tail among its operations.
# Run by models/transforms.with_cut.
S, Cmax, p = m._S, m._Cmax, m._p
head = {(j, k): sum(p[j, t] for t in range(k)) for (j, k) in p}
tail = {(j, k): sum(p[j, t] for t in range(k + 1, n_machines)) for (j, k) in p}
for o in p:
    S[o].LB = max(S[o].LB, head[o])
m.addConstrs((Cmax >= S[o] + p[o] + tail[o] for o in p), name="tb_tail")
m.addConstrs((Cmax >= sum(p[j, k] for k in range(n_machines)) for j in range(n_jobs)), name="tb_job")
for r in range(n_machines):
    ops = [o for o in p if machines[o[0]][o[1]] == r]
    if ops:
        m.addConstr(Cmax >= sum(p[o] for o in ops) + min(head[o] for o in ops) + min(tail[o] for o in ops),
                    name=f"tb_machine[{r}]")
