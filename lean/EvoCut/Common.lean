import Mathlib.Tactic
import Mathlib.Data.Real.Basic
import Mathlib.Order.Basic

/-!
# MILP formulations, reformulations and cuts

`MILPFormulation` and `MILPReformulation` are FLARE's definitions, verbatim (Robbins et al. 2026,
Listing 1; `repos/flare/.../assets/lean/Common.lean`). What follows them is ours: what it means
for a cut to be valid, and why a proof of it is a certificate.

A formulation minimizes `obj` over the points `x` with `feasible p x`, for every instance `p`.
`feasible` is the INTEGER feasible set (integrality is part of it), not the LP relaxation.
-/

structure MILPFormulation where
  Params   : Type
  Vars     : Params → Type
  feasible : (p : Params) → Vars p → Prop
  obj      : (p : Params) → Vars p → ℝ

structure MILPReformulation (F G : MILPFormulation) where
  paramMap    : F.Params → G.Params
  fwd         : (p : F.Params) → F.Vars p → G.Vars (paramMap p)
  bwd         : (p : F.Params) → G.Vars (paramMap p) → F.Vars p
  fwd_feas    : ∀ p x, F.feasible p x → G.feasible (paramMap p) (fwd p x)
  bwd_feas    : ∀ p x', G.feasible (paramMap p) x' → F.feasible p (bwd p x')
  objMap      : ℝ → ℝ
  objMap_mono : StrictMono objMap
  fwd_obj     : ∀ p x, F.feasible p x →
                  G.obj (paramMap p) (fwd p x) = objMap (F.obj p x)
  bwd_obj     : ∀ p x', G.feasible (paramMap p) x' →
                  G.obj (paramMap p) x' = objMap (F.obj p (bwd p x'))

namespace MILPFormulation

/-- A cut on `F`: a property of the variables, stated for every instance. A cut that adds
auxiliary variables is the statement that values of them exist (`∃ a, ...`): adding its
constraints to the model keeps exactly the points for which they do. -/
abbrev Cut (F : MILPFormulation) : Type := (p : F.Params) → F.Vars p → Prop

/-- `F` with the cut added: same instances, same variables, same objective. -/
def withCut (F : MILPFormulation) (C : F.Cut) : MILPFormulation where
  Params   := F.Params
  Vars     := F.Vars
  feasible := fun p x => F.feasible p x ∧ C p x
  obj      := F.obj

/-- `x` is an optimal solution of the instance `p`. -/
def IsOptimal (F : MILPFormulation) (p : F.Params) (x : F.Vars p) : Prop :=
  F.feasible p x ∧ ∀ y, F.feasible p y → F.obj p x ≤ F.obj p y

/-- A VALID cut (a valid inequality): every feasible point of every instance satisfies it. -/
def ValidCut (F : MILPFormulation) (C : F.Cut) : Prop :=
  ∀ p x, F.feasible p x → C p x

/-- A SAFE cut: it may remove feasible points, but for each feasible point it keeps one that
is at least as good. Weaker than `ValidCut`; it covers symmetry breaking and dominance. -/
def SafeCut (F : MILPFormulation) (C : F.Cut) : Prop :=
  ∀ p x, F.feasible p x → ∃ x', F.feasible p x' ∧ C p x' ∧ F.obj p x' ≤ F.obj p x

variable {F : MILPFormulation} {C : F.Cut}

/-- A valid cut is safe. -/
theorem ValidCut.safe (h : F.ValidCut C) : F.SafeCut C :=
  fun p x hx => ⟨x, hx, h p x hx, le_refl _⟩

/-- A valid cut gives FLARE's certificate: `F` with the cut is a constructive reformulation of
`F` (Definition 4.4 of the paper), with identity maps. -/
def ValidCut.reformulation (h : F.ValidCut C) : MILPReformulation F (F.withCut C) where
  paramMap    := id
  fwd         := fun _ x => x
  bwd         := fun _ x => x
  fwd_feas    := fun p x hx => ⟨hx, h p x hx⟩
  bwd_feas    := fun _ _ hx => hx.1
  objMap      := id
  objMap_mono := strictMono_id
  fwd_obj     := fun _ _ _ => rfl
  bwd_obj     := fun _ _ _ => rfl

/-- With a safe cut, an instance that has an optimal solution keeps one, of the same value. -/
theorem SafeCut.exists_optimal (h : F.SafeCut C) {p : F.Params} {x : F.Vars p}
    (hx : F.IsOptimal p x) :
    ∃ x' : F.Vars p, (F.withCut C).IsOptimal p x' ∧ F.obj p x' = F.obj p x := by
  obtain ⟨x', hx', hC, hle⟩ := h p x hx.1
  exact ⟨x', ⟨⟨hx', hC⟩, fun y hy => le_trans hle (hx.2 y hy.1)⟩, le_antisymm hle (hx.2 x' hx')⟩

/-- With a safe cut, every optimal solution of the model with the cut is an optimal solution of
the model without it (when that one has an optimal solution). With `exists_optimal`, this is
Audet's notion of reformulation (Definition 4.1 of the paper), with identity maps. -/
theorem SafeCut.optimal_of_withCut (h : F.SafeCut C) {p : F.Params} {x x' : F.Vars p}
    (hx : F.IsOptimal p x) (hx' : (F.withCut C).IsOptimal p x') : F.IsOptimal p x' := by
  obtain ⟨x'', hx'', hC, hle⟩ := h p x hx.1
  have h1 : F.obj p x' ≤ F.obj p x'' := hx'.2 x'' ⟨hx'', hC⟩
  exact ⟨hx'.1.1, fun y hy => le_trans h1 (le_trans hle (hx.2 y hy))⟩

/-! ## Combining cuts

A cut made of several families can be proved family by family, with care:
- valid families combine freely (`ValidCut.and`);
- a valid family can be added to a safe one (`SafeCut.and_valid`);
- two SAFE families do NOT combine in general: each may keep a different optimal solution, and
  together they may keep none. The second must be proved safe for the model that already has
  the first (`SafeCut.and_then`). -/

/-- Two valid cuts together are a valid cut. -/
theorem ValidCut.and {C₁ C₂ : F.Cut} (h₁ : F.ValidCut C₁) (h₂ : F.ValidCut C₂) :
    F.ValidCut (fun p x => C₁ p x ∧ C₂ p x) :=
  fun p x hx => ⟨h₁ p x hx, h₂ p x hx⟩

/-- A safe cut and a valid cut together are a safe cut. -/
theorem SafeCut.and_valid {C₁ C₂ : F.Cut} (h₁ : F.SafeCut C₁) (h₂ : F.ValidCut C₂) :
    F.SafeCut (fun p x => C₁ p x ∧ C₂ p x) := by
  intro p x hx
  obtain ⟨x', hx', hC, hle⟩ := h₁ p x hx
  exact ⟨x', hx', ⟨hC, h₂ p x' hx'⟩, hle⟩

/-- A cut that is safe for the model, then a cut that is safe for the model WITH the first one:
together they are safe for the model. -/
theorem SafeCut.and_then {C₁ C₂ : F.Cut} (h₁ : F.SafeCut C₁) (h₂ : (F.withCut C₁).SafeCut C₂) :
    F.SafeCut (fun p x => C₁ p x ∧ C₂ p x) := by
  intro p x hx
  obtain ⟨x', hx', hC₁, hle⟩ := h₁ p x hx
  obtain ⟨x'', hx'', hC₂, hle'⟩ := h₂ p x' ⟨hx', hC₁⟩
  exact ⟨x'', hx''.1, ⟨hx''.2, hC₂⟩, le_trans hle' hle⟩

end MILPFormulation
