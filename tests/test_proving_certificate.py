from pathlib import Path

import pytest

from proving import certificate, lean
from proving.certificate import Verdict, verify
from proving.lean import LeanMessage
from proving.problems import LEAN_PROBLEMS

needs_lean = pytest.mark.skipif(lean.unavailable() is not None, reason=str(lean.unavailable()))
TSP = LEAN_PROBLEMS["tsp"]

STATEMENT: str = "/-- an arc is used at most once -/\ndef Cut (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j ≤ 1"
PROOF: str = ("theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n"
              "  rcases h.hx_bin i j with a | a <;> omega")
FALSE_STATEMENT: str = "def Cut (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j = 0"


def test_the_last_lean_block_of_a_reply() -> None:
    reply: str = "draft:\n```lean\nold\n```\nfinal:\n```lean4\ndef Cut := 1\n```\nDone."
    assert certificate.lean_block(reply) == "def Cut := 1"
    assert certificate.lean_block("no code") is None
    assert certificate.lean_block("```python\nx = 1\n```") is None


def test_imports_are_moved_to_the_header() -> None:
    modules, rest = certificate.split_imports("import Mathlib.Data.Finset.Sort\n\nlemma a : True := trivial")
    assert modules == ["Mathlib.Data.Finset.Sort"] and rest == "lemma a : True := trivial"
    text, offset = certificate.proof_file(TSP, STATEMENT, "import Mathlib.Data.Finset.Sort\n" + PROOF)
    lines = text.splitlines()
    assert lines[:3] == ["import Mathlib.Tactic", "import EvoCut.Problems.Tsp", "import Mathlib.Data.Finset.Sort"]
    assert "namespace EvoCut.Tsp.CutProof" in lines and lines[offset] == PROOF.splitlines()[0]
    assert certificate.refused_imports(["Mathlib.Tactic", "EvoCut.Cuts.TspSeed", "Aesop"]) == ["EvoCut.Cuts.TspSeed", "Aesop"]


def test_the_claim_is_read_from_the_theorem_name() -> None:
    assert certificate.claim_of(PROOF) == "valid"
    assert certificate.claim_of("theorem cut_safe : formulation.SafeCut Cut := by simp") == "safe"
    assert certificate.claim_of("-- theorem cut_valid\nlemma helper : True := trivial") is None


def test_the_check_file_states_the_claim_with_full_names() -> None:
    text: str = certificate.check_file(TSP, "safe")
    assert text.startswith("import CutProof\n")
    assert "example : MILPFormulation.SafeCut EvoCut.Tsp.formulation EvoCut.Tsp.CutProof.Cut :=" in text
    assert text.rstrip().endswith("#print axioms EvoCut.Tsp.CutProof.cut_safe")


def test_refused_before_any_compile(tmp_path: Path) -> None:
    sorry: Verdict = verify(TSP, FALSE_STATEMENT, "theorem cut_valid : formulation.ValidCut Cut := by sorry", str(tmp_path))
    assert not sorry.proved and "sorry" in sorry.reason and sorry.compilation is None
    no_theorem: Verdict = verify(TSP, STATEMENT, "lemma helper : True := trivial", str(tmp_path))
    assert not no_theorem.proved and "cut_valid" in no_theorem.reason
    imported: Verdict = verify(TSP, STATEMENT, "import EvoCut.Cuts.TspSeed\n" + PROOF, str(tmp_path))
    assert not imported.proved and "only Mathlib" in imported.reason
    assert list(tmp_path.iterdir()) == []


def test_messages_are_described_for_the_llm() -> None:
    source: str = "line one\nline two\n  linarith\nlast"
    messages = [LeanMessage("warning", 4, 0, "unused variable"), LeanMessage("error", 3, 2, "linarith failed\n"),
                LeanMessage("information", 1, 0, "Try this")]
    text: str = certificate.describe(messages, source, offset=1)
    assert text.startswith("error, line 2: `linarith`\nlinarith failed")
    assert "warning, line 3: `last`" in text and "Try this" not in text
    assert certificate.describe(messages, source, limit=20).endswith("[... cut]")


@needs_lean
def test_a_true_cut_is_proved_and_a_false_one_is_not(tmp_path: Path) -> None:
    good: Verdict = verify(TSP, STATEMENT, PROOF, str(tmp_path / "good"))
    assert good.proved and good.claim == "valid" and set(good.axioms) <= lean.STANDARD_AXIOMS
    assert (tmp_path / "good" / "CutProof.lean").exists() and (tmp_path / "good" / "Check.lean").exists()
    # the same proof text does not prove another statement
    bad: Verdict = verify(TSP, FALSE_STATEMENT, PROOF, str(tmp_path / "bad"))
    assert not bad.proved and bad.compilation is not None and bad.compilation.errors
    assert "does not compile" in bad.reason
    # a statement that is not a cut of the formulation does not compile
    result = lean.compile_file(_write(tmp_path / "S.lean", certificate.statement_file(TSP, "def Cut : Prop := True")))
    assert not result.ok
    assert lean.compile_file(_write(tmp_path / "S2.lean", certificate.statement_file(TSP, STATEMENT))).ok


@needs_lean
def test_a_hypothesis_leaked_into_the_theorem_is_caught_by_the_clean_file(tmp_path: Path) -> None:
    # `include` adds the false hypothesis h to cut_valid: CutProof.lean compiles, but cut_valid
    # no longer proves the claim, and Check.lean (which sees none of this) refuses it
    cheat: str = ("variable {h : False}\ninclude h\n"
                  "theorem cut_valid : formulation.ValidCut Cut := h.elim")
    verdict: Verdict = verify(TSP, FALSE_STATEMENT, cheat, str(tmp_path))
    assert verdict.compilation is not None and verdict.compilation.ok
    assert not verdict.proved and "does not prove the claim as stated" in verdict.reason


@needs_lean
def test_a_safe_claim_and_the_hand_written_seed_proof(tmp_path: Path) -> None:
    safe_proof: str = PROOF.replace("cut_valid", "helper") + "\n\ntheorem cut_safe : formulation.SafeCut Cut := helper.safe"
    safe: Verdict = verify(TSP, STATEMENT, safe_proof, str(tmp_path / "safe"))
    assert safe.proved and safe.claim == "safe" and "SafeCut proved" in safe.reason
    # lean/EvoCut/Cuts/TspSeed.lean, cut into its statement and its proof
    import os
    from utils import parameters
    text: str = open(os.path.join(parameters.LEAN_DIR, "EvoCut", "Cuts", "TspSeed.lean")).read()
    statement: str = text.split("-- § Statement\n", 1)[1].split("-- § Proof\n", 1)[0]
    proof: str = text.split("-- § Proof\n", 1)[1].split("end EvoCut.Tsp.Seed", 1)[0]
    seed: Verdict = verify(TSP, statement.strip(), proof.strip(), str(tmp_path / "seed"))
    assert seed.proved and seed.claim == "valid", seed.reason


def _write(path: Path, text: str) -> str:
    path.write_text(text)
    return str(path)


def test_a_repeated_header_is_removed_from_an_answer() -> None:
    answer: str = ("import Mathlib.Tactic\nimport EvoCut.Problems.Tsp\nimport Mathlib.Data.Real.Basic\n\n"
                   "open BigOperators Finset\n\nnamespace EvoCut.Tsp.CutProof\n\n" + STATEMENT
                   + "\n\nend EvoCut.Tsp.CutProof\n")
    cleaned: str = certificate.unwrap(TSP, answer)
    assert cleaned.startswith("import Mathlib.Tactic\nimport Mathlib.Data.Real.Basic") and cleaned.endswith(STATEMENT)
    assert "namespace" not in cleaned and "EvoCut.Problems.Tsp" not in cleaned and "open BigOperators" not in cleaned
    # other namespaces, sections and imports are kept (and refused later if not allowed)
    other: str = "import EvoCut.Cuts.TspSeed\nnamespace Foo\n" + STATEMENT + "\nend Foo"
    assert certificate.unwrap(TSP, other) == other
    text: str = certificate.statement_file(TSP, cleaned)
    assert text.count("namespace EvoCut.Tsp.CutProof") == 1 and text.count("import EvoCut.Problems.Tsp") == 1


FAMILIES: str = ("def Family_upper (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j ≤ 1\n\n"
                 "-- def Family_commented : Prop := True\n"
                 "def Family_lower (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, 0 ≤ v.x i j")


def test_families_of_a_statement_and_the_cut_added() -> None:
    assert certificate.families(FAMILIES) == ["Family_upper", "Family_lower"] and certificate.families(STATEMENT) == []
    assert not certificate.defines_cut(FAMILIES) and certificate.defines_cut(STATEMENT)
    whole: str = certificate.with_cut(FAMILIES, ["Family_upper", "Family_lower"])
    assert whole.endswith("def Cut (p : Params) (v : Vars p) : Prop :=\n  Family_upper p v ∧ Family_lower p v")
    assert certificate.without_cut(whole) == FAMILIES and certificate.without_cut(STATEMENT) == STATEMENT
    assert certificate.with_cut(FAMILIES, ["Family_lower"]).endswith(":=\n  Family_lower p v")


@needs_lean
def test_proofs_of_the_families_combine_into_a_proof_of_the_cut(tmp_path: Path) -> None:
    names = ["Family_upper", "Family_lower"]
    lower: str = ("import Mathlib.Data.Int.Basic\nprivate lemma helper : (0 : ℤ) ≤ 1 := by omega\n"
                  "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n"
                  "  rcases h.hx_bin i j with a | a <;> omega")
    proof: str = certificate.combined_proof(names, [PROOF, lower])
    assert proof.startswith("import Mathlib.Data.Int.Basic\n") and proof.count("namespace Part") == 2
    verdict: Verdict = verify(TSP, certificate.with_cut(FAMILIES, names), proof, str(tmp_path / "both"))
    assert verdict.proved and verdict.claim == "valid", verdict.reason
    # each part proves its own family only: the upper proof does not prove the lower family
    wrong: str = certificate.combined_proof(names, [PROOF, "theorem cut_valid : formulation.ValidCut Cut := by\n"
                                                           "  intro p v h i j\n  exact (h.hx_bin i j).elim (by omega) (by omega)"])
    assert verify(TSP, certificate.with_cut(FAMILIES, names), wrong, str(tmp_path / "both2")).proved
    swapped: Verdict = verify(TSP, certificate.with_cut(FAMILIES, names),
                              certificate.combined_proof(names, [lower.replace("import Mathlib.Data.Int.Basic\n", ""),
                                                                 "theorem cut_valid : formulation.ValidCut Cut := fun p v h i j => le_refl _"]),
                              str(tmp_path / "swapped"))
    assert not swapped.proved
