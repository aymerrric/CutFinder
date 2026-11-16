import Mathlib.Tactic
import EvoCut.Problems.Tsp

/-!
# The TSP seed cut is valid (written by hand: the reference certificate)

`models/tsp.py` `BASELINE_CODE`, 2-cycle elimination:
`m.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)`.

Proof. Suppose both arcs between `i < j` are used. If neither city is the depot, the two MTZ
rows give `u i < u j` and `u j < u i`. Otherwise `i` is the depot: the tour goes depot → j →
depot. As `n ≥ 3` there are other cities; take one, `k`, with the largest position `u k`. Its
successor `l` is neither the depot nor `j` (each is already entered from the other one), so the
MTZ row of the arc `k → l` gives `u l > u k`, against the choice of `k`.
-/

open BigOperators Finset

namespace EvoCut.Tsp.Seed

-- § Statement

/-- 2-cycle elimination: at most one of the two arcs between two cities is used. -/
def Cut (p : Params) (v : Vars p) : Prop :=
  ∀ i j : Fin p.n, i < j → v.x i j + v.x j i ≤ 1

-- § Proof

/-- In a 0/1 family that sums to 1, at most one entry is 1. -/
private lemma eq_zero_of_sum_one {n : ℕ} {f : Fin n → ℤ} (hbin : ∀ k, f k = 0 ∨ f k = 1)
    (hsum : ∑ k, f k = 1) {a b : Fin n} (hab : a ≠ b) (ha : f a = 1) : f b = 0 := by
  rcases hbin b with hb | hb
  · exact hb
  · exfalso
    have hnn : ∀ k : Fin n, 0 ≤ f k := fun k => by rcases hbin k with h | h <;> omega
    have hle : ∑ k ∈ ({a, b} : Finset (Fin n)), f k ≤ ∑ k, f k :=
      Finset.sum_le_sum_of_subset_of_nonneg (subset_univ _) (fun k _ _ => hnn k)
    rw [Finset.sum_pair hab] at hle
    omega

/-- In a 0/1 family that sums to 1, some entry is 1. -/
private lemma exists_one_of_sum_one {n : ℕ} {f : Fin n → ℤ} (hbin : ∀ k, f k = 0 ∨ f k = 1)
    (hsum : ∑ k, f k = 1) : ∃ k, f k = 1 := by
  by_contra hne
  push_neg at hne
  have hzero : ∑ k, f k = 0 := Finset.sum_eq_zero (fun k _ => (hbin k).resolve_right (hne k))
  omega

theorem cut_valid : formulation.ValidCut Cut := by
  intro p v h i j hij
  by_contra hcon
  have hxij : v.x i j = 1 := by
    rcases h.hx_bin i j with a | a <;> rcases h.hx_bin j i with b | b <;> omega
  have hxji : v.x j i = 1 := by
    rcases h.hx_bin i j with a | a <;> rcases h.hx_bin j i with b | b <;> omega
  have hne : i ≠ j := ne_of_lt hij
  -- j comes after i, so it is not the depot (index 0)
  have hj : j ≠ depot p := by
    intro e
    have h0 : j.val = 0 := by rw [e]; rfl
    have hlt : i.val < j.val := hij
    omega
  by_cases hi : i = depot p
  · -- depot → j → depot: the other cities form a set closed under "successor"
    let S : Finset (Fin p.n) := univ.filter (fun k => k ≠ i ∧ k ≠ j)
    have hS : S.Nonempty := by
      by_contra hempty
      rw [Finset.not_nonempty_iff_eq_empty] at hempty
      have hsub : (univ : Finset (Fin p.n)) ⊆ {i, j} := by
        intro k _
        by_contra hk
        have hkS : k ∈ S := by
          simp only [mem_insert, mem_singleton, not_or] at hk
          simp only [S, mem_filter, mem_univ, true_and]
          exact hk
        rw [hempty] at hkS
        exact absurd hkS (Finset.notMem_empty k)
      have hcard := Finset.card_le_card hsub
      have htwo : ({i, j} : Finset (Fin p.n)).card ≤ 2 := Finset.card_le_two
      simp only [Finset.card_univ, Fintype.card_fin] at hcard
      have := p.hn
      omega
    obtain ⟨k, hkS, hkmax⟩ := Finset.exists_max_image S v.u hS
    have hk : k ≠ i ∧ k ≠ j := by
      simp only [S, mem_filter, mem_univ, true_and] at hkS
      exact hkS
    obtain ⟨l, hl⟩ := exists_one_of_sum_one (h.hx_bin k) (h.hout k)
    have hlk : l ≠ k := by
      intro e
      rw [e] at hl
      have := h.hx_self k
      omega
    -- the depot is entered from j, and j from the depot: k enters neither
    have hki : v.x k i = 0 :=
      eq_zero_of_sum_one (f := fun a => v.x a i) (fun a => h.hx_bin a i) (h.hin i)
        (a := j) (b := k) (Ne.symm hk.2) hxji
    have hkj : v.x k j = 0 :=
      eq_zero_of_sum_one (f := fun a => v.x a j) (fun a => h.hx_bin a j) (h.hin j)
        (a := i) (b := k) (Ne.symm hk.1) hxij
    have hli : l ≠ i := by
      intro e
      rw [e] at hl
      omega
    have hlj : l ≠ j := by
      intro e
      rw [e] at hl
      omega
    have hlS : l ∈ S := by
      simp only [S, mem_filter, mem_univ, true_and]
      exact ⟨hli, hlj⟩
    have hmax : v.u l ≤ v.u k := hkmax l hlS
    have hkd : k ≠ depot p := by rw [← hi]; exact hk.1
    have hld : l ≠ depot p := by rw [← hi]; exact hli
    have hm := h.hmtz k l hkd hld (Ne.symm hlk)
    rw [hl] at hm
    omega
  · -- neither is the depot: the two MTZ rows contradict each other
    have h1 := h.hmtz i j hi hj hne
    have h2 := h.hmtz j i hj hi hne.symm
    rw [hxij] at h1
    rw [hxji] at h2
    omega

end EvoCut.Tsp.Seed

/-- The certificate, stated without any open namespace. -/
theorem EvoCut.Certificate.tsp_seed :
    MILPFormulation.ValidCut EvoCut.Tsp.formulation EvoCut.Tsp.Seed.Cut :=
  EvoCut.Tsp.Seed.cut_valid

#print axioms EvoCut.Certificate.tsp_seed
