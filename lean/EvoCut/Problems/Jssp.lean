import EvoCut.Common
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic

/-!
# JSSP, disjunctive big-M formulation: `src/models/jssp.py`,
`create_model(n_jobs, n_machines, times, machines)`

Written by hand from the Python model, constraint by constraint. The big-M is kept as it is in
the Python model (FLARE's own formalizations replace big-M rows by disjunctions; keeping them
makes this file a line-by-line copy, and cuts that use `y` or the big-M can be stated).

| Python                                      | Lean                                               |
|---------------------------------------------|----------------------------------------------------|
| jobs `0..n_jobs-1`                          | `Fin p.nJ` (same index)                            |
| operations `k = 0..n_machines-1` of a job   | `Fin p.nM` (same index)                            |
| machines `0..n_machines-1`                  | `Fin p.nM` (same index)                            |
| `times[j][k]`, also `p[j, k]`               | `p.times j k : ℝ`                                  |
| `machines[j][k]`                            | `p.machines j k : Fin p.nM`                        |
| `big_m`, `m._bigM` = sum of all the times   | `bigM p`                                           |
| `S[j, k]`, `Cmax` continuous `>= 0`         | `v.S j k`, `v.Cmax : ℝ`, with `hS_nn`, `hCmax_nn`  |
| `y[j1, k1, j2, k2]` binary, on `pairs`      | `v.y j1 k1 j2 k2 : ℤ`, with `hy_bin`               |
| `pairs`: same machine, `(j1, k1) < (j2, k2)`| `Pair p j1 k1 j2 k2`                               |
| the last operation `n_machines - 1`         | `last p`                                           |
| `precedence`, `disj1`, `disj2`, `makespan`  | `hprec`, `hdisj1`, `hdisj2`, `hmakespan`           |

`y` exists only on `pairs` in Python. Here it is a function of four indices; outside `Pair`
its value is free and appears in no constraint, so the feasible sets correspond.

Assumptions on the data (`proving/problems.py` checks them on our instances): at least one
machine, positive processing times, and every job visits every machine exactly once.
-/

open BigOperators Finset

namespace EvoCut.Jssp

structure Params where
  nJ : ℕ  -- number of jobs
  nM : ℕ  -- number of machines = number of operations of a job
  times : Fin nJ → Fin nM → ℝ  -- processing time of the k-th operation of job j
  machines : Fin nJ → Fin nM → Fin nM  -- machine of the k-th operation of job j
  -- Assumptions
  hnM : 0 < nM
  htimes_pos : ∀ j k, 0 < times j k
  hmachines_inj : ∀ j, Function.Injective (machines j)

/-- The big-M of the Python model: the sum of all the processing times. -/
def bigM (p : Params) : ℝ := ∑ j : Fin p.nJ, ∑ k : Fin p.nM, p.times j k

/-- The last operation of a job (`n_machines - 1` in Python). -/
def last (p : Params) : Fin p.nM := ⟨p.nM - 1, by have := p.hnM; omega⟩

/-- The pairs of the Python model: two operations on the same machine, in lexicographic order. -/
def Pair (p : Params) (j1 : Fin p.nJ) (k1 : Fin p.nM) (j2 : Fin p.nJ) (k2 : Fin p.nM) : Prop :=
  (j1 < j2 ∨ (j1 = j2 ∧ k1 < k2)) ∧ p.machines j1 k1 = p.machines j2 k2

structure Vars (p : Params) where
  S : Fin p.nJ → Fin p.nM → ℝ  -- start time of operation (j, k)
  Cmax : ℝ  -- makespan
  y : Fin p.nJ → Fin p.nM → Fin p.nJ → Fin p.nM → ℤ  -- 1 if (j1, k1) runs before (j2, k2)

structure Feasible (p : Params) (v : Vars p) : Prop where
  hS_nn : ∀ (j : Fin p.nJ) (k : Fin p.nM), 0 ≤ v.S j k
  hCmax_nn : 0 ≤ v.Cmax
  hy_bin : ∀ (j1 : Fin p.nJ) (k1 : Fin p.nM) (j2 : Fin p.nJ) (k2 : Fin p.nM),
    Pair p j1 k1 j2 k2 → v.y j1 k1 j2 k2 = 0 ∨ v.y j1 k1 j2 k2 = 1
  -- the operations of a job run in order
  hprec : ∀ (j : Fin p.nJ) (k : Fin p.nM) (hk : k.val + 1 < p.nM),
    v.S j k + p.times j k ≤ v.S j ⟨k.val + 1, hk⟩
  -- two operations on the same machine do not overlap: one of them runs first
  hdisj1 : ∀ (j1 : Fin p.nJ) (k1 : Fin p.nM) (j2 : Fin p.nJ) (k2 : Fin p.nM),
    Pair p j1 k1 j2 k2 →
    v.S j1 k1 + p.times j1 k1 ≤ v.S j2 k2 + bigM p * (1 - (v.y j1 k1 j2 k2 : ℝ))
  hdisj2 : ∀ (j1 : Fin p.nJ) (k1 : Fin p.nM) (j2 : Fin p.nJ) (k2 : Fin p.nM),
    Pair p j1 k1 j2 k2 →
    v.S j2 k2 + p.times j2 k2 ≤ v.S j1 k1 + bigM p * (v.y j1 k1 j2 k2 : ℝ)
  -- the makespan is after the end of every job
  hmakespan : ∀ j : Fin p.nJ, v.S j (last p) + p.times j (last p) ≤ v.Cmax

-- Minimize the makespan
def obj (p : Params) (v : Vars p) : ℝ := v.Cmax

def formulation : MILPFormulation where
  Params   := Params
  Vars     := Vars
  feasible := Feasible
  obj      := obj

end EvoCut.Jssp
