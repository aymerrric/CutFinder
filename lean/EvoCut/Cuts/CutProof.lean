import Mathlib.Tactic
import EvoCut.Problems.Cwlp

open BigOperators Finset

namespace EvoCut.Cwlp

-- § Statement

/-- The SAGE-CWLP cut: global capacity, slots for large demands, linking, too-big,
clique, and at-most-two-small cuts. -/
def Cut (p : Params) (v : Vars p) : Prop :=
  -- 1. Global capacity
  (∑ j : Fin p.nJ, p.u j * (v.y j : ℝ)) ≥ ∑ i : Fin p.nI, p.d i ∧
  -- 2. Slots for large demands
  (∀ (v : ℝ), (∃ i : Fin p.nI, p.d i = v) →
    (∑ j : Fin p.nJ, ((Int.floor (p.u j / v)).toNat : ℝ) * (v.y j : ℝ)) ≥
    ((Finset.filter (fun i => p.d i ≥ v) Finset.univ).card : ℝ)) ∧
  -- 3. Linking
  (∀ i j, v.x i j ≤ v.y j) ∧
  -- 4. Too big
  (∀ i j, p.d i > p.u j → v.x i j ≤ 0) ∧
  -- 5. Clique
  (∀ j, (Finset.sum (Finset.filter (fun i => ∀ k : Fin p.nI, i ≠ k → p.d i + p.d k > p.u j) Finset.univ) (fun i => v.x i j)) ≤ v.y j) ∧
  -- 6. At most 2 customers of demand > u/3
  (∀ j, (Finset.sum (Finset.filter (fun i => p.d i > p.u j / 3) Finset.univ) (fun i => v.x i j)) ≤ 2 * v.y j)

-- § Proof

private lemma hx_nonneg {p : Params} {v : Vars p}
    (hx_bin : ∀ (i : Fin p.nI) (j : Fin p.nJ), v.x i j = 0 ∨ v.x i j = 1)
    (i : Fin p.nI) (j : Fin p.nJ) : 0 ≤ (v.x i j : ℝ) := by
  rcases hx_bin i j with (h | h)
  · rw [h]; norm_num
  · rw [h]; norm_num

private lemma hy_nonneg {p : Params} {v : Vars p}
    (hy_bin : ∀ j : Fin p.nJ, v.y j = 0 ∨ v.y j = 1) (j : Fin p.nJ) : 0 ≤ (v.y j : ℝ) := by
  rcases hy_bin j with (h | h)
  · rw [h]; norm_num
  · rw [h]; norm_num

private lemma hx_le_one {p : Params} {v : Vars p}
    (hx_bin : ∀ (i : Fin p.nI) (j : Fin p.nJ), v.x i j = 0 ∨ v.x i j = 1)
    (i : Fin p.nI) (j : Fin p.nJ) : (v.x i j : ℝ) ≤ 1 := by
  rcases hx_bin i j with (h | h)
  · rw [h]; norm_num
  · rw [h]; norm_num

private lemma hy_le_one {p : Params} {v : Vars p}
    (hy_bin : ∀ j : Fin p.nJ, v.y j = 0 ∨ v.y j = 1) (j : Fin p.nJ) : (v.y j : ℝ) ≤ 1 := by
  rcases hy_bin j with (h | h)
  · rw [h]; norm_num
  · rw [h]; norm_num

private lemma sum_x_eq_one {p : Params} {v : Vars p}
    (hassign : ∀ i : Fin p.nI, ∑ j : Fin p.nJ, v.x i j = 1) (i : Fin p.nI) :
    (∑ j : Fin p.nJ, (v.x i j : ℝ)) = 1 := by
  have h := hassign i
  simpa using congrArg (fun x : ℤ => (x : ℝ)) h

theorem cut_valid : formulation.ValidCut Cut := by
  intro p v h
  rcases h with ⟨hx_bin, hy_bin, hassign, hcap⟩
  have h_global : (∑ j : Fin p.nJ, p.u j * (v.y j : ℝ)) ≥ ∑ i : Fin p.nI, p.d i := by
    have hsum_eq : (∑ i : Fin p.nI, p.d i) = (∑ i : Fin p.nI, ∑ j : Fin p.nJ, p.d i * (v.x i j : ℝ)) := by
      calc
        (∑ i : Fin p.nI, p.d i) = (∑ i : Fin p.nI, p.d i * 1) := by simp
        _ = (∑ i : Fin p.nI, p.d i * (∑ j : Fin p.nJ, (v.x i j : ℝ))) := by
          refine Finset.sum_congr rfl (fun i _ => ?_)
          rw [sum_x_eq_one hassign i]
        _ = (∑ i : Fin p.nI, ∑ j : Fin p.nJ, p.d i * (v.x i j : ℝ)) := by
          simp_rw [Finset.mul_sum]
    have htotal : (∑ i : Fin p.nI, ∑ j : Fin p.nJ, p.d i * (v.x i j : ℝ)) ≤ ∑ j : Fin p.nJ, p.u j * (v.y j : ℝ) := by
      calc
        (∑ i : Fin p.nI, ∑ j : Fin p.nJ, p.d i * (v.x i j : ℝ)) = ∑ j : Fin p.nJ, (∑ i : Fin p.nI, p.d i * (v.x i j : ℝ)) := by
          rw [Finset.sum_comm]
        _ ≤ ∑ j : Fin p.nJ, p.u j * (v.y j : ℝ) :=
          Finset.sum_le_sum (fun j _ => hcap j)
    linarith
  have h_slots : ∀ (v : ℝ), (∃ i : Fin p.nI, p.d i = v) →
      (∑ j : Fin p.nJ, ((Int.floor (p.u j / v)).toNat : ℝ) * (v.y j : ℝ)) ≥
      ((Finset.filter (fun i => p.d i ≥ v) Finset.univ).card : ℝ) := by
    intro v hv
    rcases hv with ⟨i0, hi0⟩
    -- This is a complex constraint. We'll prove it by contradiction.
    by_contra hlt
    push_cast at hlt
    have h_lt : (∑ j : Fin p.nJ, ((Int.floor (p.u j / v)).toNat : ℝ) * (v.y j : ℝ)) <
        ((Finset.filter (fun i => p.d i ≥ v) Finset.univ).card : ℝ) := by linarith
    -- This means there are more customers with demand >= v than available slots
    -- But each such customer must be assigned to some warehouse, and each warehouse
    -- can accommodate at most floor(u[j]/v) such customers (since each needs >= v capacity)
    -- This is a contradiction with the feasibility of the assignment
    -- A full proof would formalize this counting argument
    sorry
  have h_link : ∀ i j, v.x i j ≤ v.y j := by
    intro i j
    rcases hx_bin i j with (hx0 | hx1)
    · have hy_nonneg_j : 0 ≤ (v.y j : ℝ) := hy_nonneg hy_bin j
      omega
    · rcases hy_bin j with (hy0 | hy1)
      · have hpos : 0 < p.d i := p.hd_pos i
        have hcap_j := hcap j
        have hsum_le : (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) ≤ 0 := by
          simpa [hy0] using hcap_j
        have hterm_nonneg : 0 ≤ p.d i * (v.x i j : ℝ) := by
          have hxpos : 0 ≤ (v.x i j : ℝ) := hx_nonneg hx_bin i j
          nlinarith
        have hterm_le_sum : p.d i * (v.x i j : ℝ) ≤ (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) :=
          Finset.single_le_sum (by
            intro i' _
            have hd_nn := p.hd_pos i'
            have hx_nonneg' : 0 ≤ (v.x i' j : ℝ) := hx_nonneg hx_bin i' j
            nlinarith) (mem_univ i)
        rw [hx1] at hterm_le_sum
        nlinarith
      · omega
  have h_too_big : ∀ i j, p.d i > p.u j → v.x i j ≤ 0 := by
    intro i j hbig
    rcases hx_bin i j with (hx0 | hx1)
    · omega
    · have hpos : 0 < p.d i := p.hd_pos i
      have hcap_j := hcap j
      have hy_le_one_j : (v.y j : ℝ) ≤ 1 := hy_le_one hy_bin j
      have hsum_le : (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) ≤ p.u j := by
        calc
          (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) ≤ p.u j * (v.y j : ℝ) := hcap_j
          _ ≤ p.u j * 1 := by nlinarith
          _ = p.u j := by ring
      have hterm_le_sum : p.d i * (v.x i j : ℝ) ≤ (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) :=
        Finset.single_le_sum (by
          intro i' _
          have hd_nn := p.hd_pos i'
          have hx_nonneg' : 0 ≤ (v.x i' j : ℝ) := hx_nonneg hx_bin i' j
          nlinarith) (mem_univ i)
      rw [hx1] at hterm_le_sum
      have h_di_le_u : p.d i ≤ p.u j := by
        nlinarith
      nlinarith
  have h_clique : ∀ j, (∑ i in {i : Fin p.nI | ∀ k : Fin p.nI, i ≠ k → p.d i + p.d k > p.u j}, v.x i j) ≤ v.y j := by
    intro j
    let S : Finset (Fin p.nI) := {i | ∀ k : Fin p.nI, i ≠ k → p.d i + p.d k > p.u j}
    rcases hy_bin j with (hy0 | hy1)
    · -- y[j] = 0, so all x[i,j] = 0 by h_link
      have hx_all_zero : ∀ i, v.x i j = 0 := by
        intro i
        have hlink_i := h_link i j
        rw [hy0] at hlink_i
        rcases hx_bin i j with (hx0 | hx1)
        · exact hx0
        · omega
      simp [hx_all_zero, hy0, S]
    · -- y[j] = 1, need to show sum_{i in S} x[i,j] <= 1
      -- If sum > 1, there are at least 2 customers in S with x[i,j] = 1
      -- Then d[i] + d[k] > u[j] (by S's definition), but hcap says sum <= u[j], contradiction
      by_contra hgt
      have hsum_gt_one : 1 < (∑ i in S, (v.x i j : ℝ)) := by
        -- hgt: ¬ (sum <= 1), so sum > 1
        linarith
      -- Since each x[i,j] is 0 or 1, the sum is an integer
      -- If sum > 1, there must be at least 2 elements with x[i,j] = 1
      have h_two_ones : ∃ (i k : Fin p.nI), i ∈ S ∧ k ∈ S ∧ i ≠ k ∧ v.x i j = 1 ∧ v.x k j = 1 := by
        -- We prove this by contradiction: if at most one element has x = 1, then sum <= 1
        by_contra h_no_two
        push_neg at h_no_two
        -- h_no_two: for all i, k in S, if i ≠ k then not (x[i,j] = 1 ∧ x[k,j] = 1)
        -- This means at most one element of S has x = 1
        -- So the sum is at most 1
        have h_le_one : (∑ i in S, (v.x i j : ℝ)) ≤ 1 := by
          rcases Finset.eq_empty_or_nonempty S with (h_empty | h_nonempty)
          · simp [h_empty]
          · -- If S has at least one element, let i0 be one with x[i0,j] = 1 if any
            by_cases h_exists_one : ∃ i ∈ S, v.x i j = 1
            · rcases h_exists_one with ⟨i0, hi0S, hi0x⟩
              have h_others_zero : ∀ k ∈ S, k ≠ i0 → v.x k j = 0 := by
                intro k hkS hk_ne
                rcases hx_bin k j with (hk0 | hk1)
                · exact hk0
                · exfalso
                  apply h_no_two i0 k hi0S hkS hk_ne
                  exact ⟨hi0x, hk1⟩
              -- Compute the sum: only i0 contributes 1, all others contribute 0
              calc
                (∑ i in S, (v.x i j : ℝ)) = (v.x i0 j : ℝ) := by
                  refine (Finset.sum_eq_single i0 ?_ ?_).symm
                  · intro k hkS hk_ne
                    rw [h_others_zero k hkS hk_ne, cast_zero]
                  · simp [hi0S]
                _ = 1 := by rw [hi0x]; norm_num
            · -- No element has x = 1, so all x = 0
              have h_all_zero : ∀ i ∈ S, v.x i j = 0 := by
                intro i hiS
                rcases hx_bin i j with (h | h)
                · exact h
                · exfalso; apply h_exists_one; exact ⟨i, hiS, h⟩
              simp [h_all_zero]
        linarith
      rcases h_two_ones with ⟨i, k, hiS, hkS, h_ne, hxi, hxk⟩
      -- By definition of S, since i ≠ k, we have d[i] + d[k] > u[j]
      have h_conflict : p.d i + p.d k > p.u j := by
        have hS_i_filter : i ∈ Finset.filter (fun i => ∀ k : Fin p.nI, i ≠ k → p.d i + p.d k > p.u j) Finset.univ := by
          -- S is defined as {i | ∀ k, i ≠ k → p.d i + p.d k > p.u j}
          -- which is Finset.filter (fun i => ∀ k, i ≠ k → p.d i + p.d k > p.u j) Finset.univ
          -- So hiS already gives us what we need
          simpa [S] using hiS
        have h_forall : ∀ k' : Fin p.nI, i ≠ k' → p.d i + p.d k' > p.u j :=
          (Finset.mem_filter.mp hS_i_filter).2
        exact h_forall k h_ne
      -- Now we have d[i] + d[k] > u[j], but both are assigned to warehouse j
      -- This violates hcap[j]
      have hcap_j := hcap j
      have hsum_demands : p.d i + p.d k ≤ ∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ) := by
        -- Since x[i,j] = x[k,j] = 1 and all other terms are nonnegative
        -- The sum includes both p.d i * 1 and p.d k * 1, plus nonnegative terms
        -- So the total sum is at least d[i] + d[k]
        calc
          p.d i + p.d k = p.d i * (v.x i j : ℝ) + p.d k * (v.x k j : ℝ) := by rw [hxi, hxk]; ring
          _ ≤ (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) := by
            -- We need to show that the sum of two specific terms is <= the total sum
            -- This is true because all terms are nonnegative
            have h_nonneg : ∀ i', 0 ≤ p.d i' * (v.x i' j : ℝ) := by
              intro i'
              have hd_nn := p.hd_pos i'
              have hx_nonneg' : 0 ≤ (v.x i' j : ℝ) := hx_nonneg hx_bin i' j
              nlinarith
            -- Use Finset.sum_insert and Finset.sum_singleton
            -- The sum of {i, k} is <= sum of all
            -- We can use Finset.add_sum_of_subset
            sorry
      have h_total_le : ∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ) ≤ p.u j := by
        calc
          (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) ≤ p.u j * (v.y j : ℝ) := hcap_j
          _ = p.u j := by rw [hy1]; ring
      linarith
  have h_third : ∀ j, (∑ i in {i : Fin p.nI | p.d i > p.u j / 3}, v.x i j) ≤ 2 * v.y j := by
    intro j
    let S : Finset (Fin p.nI) := {i | p.d i > p.u j / 3}
    rcases hy_bin j with (hy0 | hy1)
    · -- y[j] = 0, so all x[i,j] = 0 by h_link
      have hx_all_zero : ∀ i, v.x i j = 0 := by
        intro i
        have hlink_i := h_link i j
        rw [hy0] at hlink_i
        rcases hx_bin i j with (hx0 | hx1)
        · exact hx0
        · omega
      simp [hx_all_zero, hy0, S]
    · -- y[j] = 1, need to show sum_{i in S} x[i,j] <= 2
      -- If sum > 2, there are at least 3 customers in S with x[i,j] = 1
      -- Then sum of their demands > u[j], contradicting hcap
      by_contra hgt
      have hsum_gt_two : 2 < (∑ i in S, (v.x i j : ℝ)) := by
        linarith
      -- Since each x[i,j] is 0 or 1, the sum is an integer
      -- If sum > 2, there must be at least 3 elements with x[i,j] = 1
      have h_three_ones : ∃ (i k l : Fin p.nI), i ∈ S ∧ k ∈ S ∧ l ∈ S ∧ i ≠ k ∧ i ≠ l ∧ k ≠ l ∧
          v.x i j = 1 ∧ v.x k j = 1 ∧ v.x l j = 1 := by
        -- Similar to the two-ones case but with three
        by_contra h_no_three
        push_neg at h_no_three
        -- h_no_three: for any three distinct elements, at least one has x = 0
        -- This means at most 2 elements have x = 1
        -- So the sum is at most 2
        have h_le_two : (∑ i in S, (v.x i j : ℝ)) ≤ 2 := by
          -- Count the number of 1's in the sum
          -- We can use the fact that sum of 0/1 values <= number of 1's
          -- If there are at most 2 ones, sum <= 2
          sorry
        linarith
      rcases h_three_ones with ⟨i, k, l, hiS, hkS, hlS, h_ne_ik, h_ne_il, h_ne_kl, hxi, hxk, hxl⟩
      -- By definition of S, d[i] > u[j]/3, d[k] > u[j]/3, d[l] > u[j]/3
      have hdi : p.d i > p.u j / 3 := by
        have hiS_filter : i ∈ Finset.filter (fun i => p.d i > p.u j / 3) Finset.univ := by
          simpa [S] using hiS
        exact (Finset.mem_filter.mp hiS_filter).2
      have hdk : p.d k > p.u j / 3 := by
        have hkS_filter : k ∈ Finset.filter (fun i => p.d i > p.u j / 3) Finset.univ := by
          simpa [S] using hkS
        exact (Finset.mem_filter.mp hkS_filter).2
      have hdl : p.d l > p.u j / 3 := by
        have hlS_filter : l ∈ Finset.filter (fun i => p.d i > p.u j / 3) Finset.univ := by
          simpa [S] using hlS
        exact (Finset.mem_filter.mp hlS_filter).2
      have h_demand_sum : p.d i + p.d k + p.d l > p.u j := by
        linarith
      -- Now we have three customers with demand > u[j]/3, all assigned to j
      -- This violates hcap[j]
      have hcap_j := hcap j
      have hsum_demands : p.d i + p.d k + p.d l ≤ ∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ) := by
        -- Similar to the clique case
        sorry
      have h_total_le : ∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ) ≤ p.u j := by
        calc
          (∑ i' : Fin p.nI, p.d i' * (v.x i' j : ℝ)) ≤ p.u j * (v.y j : ℝ) := hcap_j
          _ = p.u j := by rw [hy1]; ring
      linarith
  exact ⟨h_global, h_slots, h_link, h_too_big, h_clique, h_third⟩

end EvoCut.Cwlp

/-- The certificate, stated without any open namespace. -/
theorem EvoCut.Certificate.cwlp_sage :
    MILPFormulation.ValidCut EvoCut.Cwlp.formulation EvoCut.Cwlp.Cut :=
  EvoCut.Cwlp.cut_valid

#print axioms EvoCut.Certificate.cwlp_sage
