import EvoCut.Common
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic

/-!
# CWLP with unsplittable demands: `src/models/cwlp.py`, `create_model(I, J, d, u, f, c)`

Written by hand from the Python model, constraint by constraint. Python → Lean:

| Python                           | Lean                                              |
|----------------------------------|---------------------------------------------------|
| customers `I = 1..len(I)`        | `Fin p.nI`: customer `k` is the index `k - 1`     |
| warehouses `J = 1..len(J)`       | `Fin p.nJ`: warehouse `k` is the index `k - 1`    |
| `d[i]`, `u[j]`, `f[j]`, `c[i, j]`| `p.d i`, `p.u j`, `p.f j`, `p.c i j` (all `ℝ`)    |
| `x[i, j]` binary                 | `v.x i j : ℤ`, with `hx_bin`                      |
| `y[j]` binary                    | `v.y j : ℤ`, with `hy_bin`                        |
| constraints `assign`, `capacity` | `hassign`, `hcap`                                 |

Assumptions on the data (`proving/problems.py` checks them on our instances): at least one
customer, demands are positive, capacities and costs are not negative. With a zero demand, a
customer could be assigned to a closed warehouse, and the textbook `x[i, j] <= y[j]` would not
be valid. Without any customer no warehouse has to be open, and the seed cut (at least `k_min`
open warehouses, `k_min >= 1`) is not valid: found by the LLM on 14 Nov 2026
(`learnt_results.md` 11.1).
-/

open BigOperators Finset

namespace EvoCut.Cwlp

structure Params where
  nI : ℕ  -- number of customers
  nJ : ℕ  -- number of warehouses
  d : Fin nI → ℝ  -- demand of customer i
  u : Fin nJ → ℝ  -- capacity of warehouse j
  f : Fin nJ → ℝ  -- cost of opening warehouse j
  c : Fin nI → Fin nJ → ℝ  -- cost of serving all of customer i from warehouse j
  -- Assumptions
  hnI : 0 < nI
  hd_pos : ∀ i, 0 < d i
  hu_nn : ∀ j, 0 ≤ u j
  hf_nn : ∀ j, 0 ≤ f j
  hc_nn : ∀ i j, 0 ≤ c i j

structure Vars (p : Params) where
  x : Fin p.nI → Fin p.nJ → ℤ  -- 1 if customer i is served by warehouse j
  y : Fin p.nJ → ℤ  -- 1 if warehouse j is open

structure Feasible (p : Params) (v : Vars p) : Prop where
  hx_bin : ∀ (i : Fin p.nI) (j : Fin p.nJ), v.x i j = 0 ∨ v.x i j = 1
  hy_bin : ∀ j : Fin p.nJ, v.y j = 0 ∨ v.y j = 1
  -- every customer is served by exactly one warehouse
  hassign : ∀ i : Fin p.nI, ∑ j : Fin p.nJ, v.x i j = 1
  -- a warehouse serves at most its capacity, and only if it is open
  hcap : ∀ j : Fin p.nJ, ∑ i : Fin p.nI, p.d i * (v.x i j : ℝ) ≤ p.u j * (v.y j : ℝ)

-- Minimize the opening costs plus the assignment costs
def obj (p : Params) (v : Vars p) : ℝ :=
  ∑ j : Fin p.nJ, p.f j * (v.y j : ℝ) + ∑ i : Fin p.nI, ∑ j : Fin p.nJ, p.c i j * (v.x i j : ℝ)

def formulation : MILPFormulation where
  Params   := Params
  Vars     := Vars
  feasible := Feasible
  obj      := obj

end EvoCut.Cwlp
