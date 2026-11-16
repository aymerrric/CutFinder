import EvoCut.Common
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic

/-!
# TSP, MTZ formulation: `src/models/tsp.py`, `create_model(n, c)`

Written by hand from the Python model, constraint by constraint. Python → Lean:

| Python                                  | Lean                                                    |
|-----------------------------------------|---------------------------------------------------------|
| cities `1..n`                           | `Fin p.n`: city `k` is the index `k - 1`                |
| the depot, city `1`                     | `depot p` (the index 0)                                 |
| `c[i, j]`, `i != j`                     | `p.c i j : ℝ` (`p.c i i` is never used: `x i i = 0`)    |
| `x[i, j]` binary, `i != j`              | `v.x i j : ℤ`, with `hx_bin` and `hx_self`              |
| `u[i]` integer                          | `v.u i : ℤ`                                             |
| constraints `out`, `in`, `mtz`          | `hout`, `hin`, `hmtz`                                   |
| `u[1] = 1`, `2 <= u[i] <= n` otherwise  | `hu_depot`, `hu_lo`, `hu_hi`                            |

Assumptions on the data (`proving/problems.py` checks them on our instances): at least 4 cities
and costs that are not negative. With 2 cities the only tour uses both arcs 1 → 2 → 1, and with
3 every tour is a triangle through the depot: cuts that are valid from 3 or 4 cities on fail
there (FLARE, Propositions F.2 and F.3; the paper's own TSP cut has such a family, found again
by the agent on 14 Nov 2026, `learnt_results.md` 11.2). Our smallest instance has 5 cities.
-/

open BigOperators Finset

namespace EvoCut.Tsp

structure Params where
  n : ℕ  -- number of cities
  c : Fin n → Fin n → ℝ  -- travel cost from i to j
  -- Assumptions
  hn : 4 ≤ n
  hc_nn : ∀ i j, 0 ≤ c i j

/-- The depot: city 1 of the Python model. -/
def depot (p : Params) : Fin p.n := ⟨0, by have := p.hn; omega⟩

structure Vars (p : Params) where
  x : Fin p.n → Fin p.n → ℤ  -- 1 if the tour goes directly from i to j
  u : Fin p.n → ℤ  -- position of city i in the tour

structure Feasible (p : Params) (v : Vars p) : Prop where
  hx_bin : ∀ i j : Fin p.n, v.x i j = 0 ∨ v.x i j = 1
  -- the model has no arc from a city to itself
  hx_self : ∀ i : Fin p.n, v.x i i = 0
  -- each city is left exactly once
  hout : ∀ i : Fin p.n, ∑ j : Fin p.n, v.x i j = 1
  -- each city is entered exactly once
  hin : ∀ j : Fin p.n, ∑ i : Fin p.n, v.x i j = 1
  -- MTZ subtour elimination, depot excluded
  hmtz : ∀ i j : Fin p.n, i ≠ depot p → j ≠ depot p → i ≠ j →
    v.u i - v.u j + (p.n : ℤ) * v.x i j ≤ (p.n : ℤ) - 1
  -- the depot is first
  hu_depot : v.u (depot p) = 1
  hu_lo : ∀ i : Fin p.n, i ≠ depot p → 2 ≤ v.u i
  hu_hi : ∀ i : Fin p.n, i ≠ depot p → v.u i ≤ (p.n : ℤ)

-- Minimize the total travel cost
def obj (p : Params) (v : Vars p) : ℝ :=
  ∑ i : Fin p.n, ∑ j : Fin p.n, p.c i j * (v.x i j : ℝ)

def formulation : MILPFormulation where
  Params   := Params
  Vars     := Vars
  feasible := Feasible
  obj      := obj

end EvoCut.Tsp
