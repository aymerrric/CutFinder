# SAGE-JSSP: the cut family the paper selected for the JSSP (papers/sage_evocut_tmlr2026.pdf,
# Appendix F.4, Eq. SAGE-JSSP), written for our model (models/jssp.py: operations (j, k) from 0,
# machines[j][k] its machine, p[j, k] its time, Cmax). Run by models/transforms.with_cut.
# Only the published family: the paper's instantiated form also adds first/last-on-machine
# indicator variables on the most loaded machines, described too briefly to rebuild here.
Cmax, p = m._Cmax, m._p
# average machine load
m.addConstr(Cmax >= sum(p.values()) / n_machines, name="sage_average_load")
# machine critical path: load + smallest head + smallest tail of the machine's operations
head = {(j, k): sum(p[j, t] for t in range(k)) for (j, k) in p}
tail = {(j, k): sum(p[j, t] for t in range(k + 1, n_machines)) for (j, k) in p}
for r in range(n_machines):
    ops = [o for o in p if machines[o[0]][o[1]] == r]
    m.addConstr(Cmax >= sum(p[o] for o in ops) + min(head[o] for o in ops) + min(tail[o] for o in ops),
                name=f"sage_machine_path[{r}]")
# longest job
m.addConstrs((Cmax >= sum(p[j, k] for k in range(n_machines)) for j in range(n_jobs)), name="sage_job")
