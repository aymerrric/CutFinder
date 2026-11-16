# The textbook disaggregated linking of the CWLP: a customer is served only by an open warehouse.
# Valid because demands are positive (lean/EvoCut/Problems/Cwlp.lean, hd_pos); the model has only
# the aggregated capacity row. Used to try prove_cut.py; the seed cut is weaker on purpose.
x, y = m._x, m._y
m.addConstrs((x[i, j] <= y[j] for (i, j) in x.keys()), name="linking")
