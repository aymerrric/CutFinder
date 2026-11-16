import inspect
from pathlib import Path
from typing import List

import pytest

from llm.fake import FakeLLM
from models import tsp
from proving import certificate, lean, prover as prover_module
from proving.certificate import Verdict
from proving.lean import Compilation, LeanMessage
from proving.problems import LEAN_PROBLEMS
from proving.prover import CLAIMED_INVALID, NO_STATEMENT, NOT_PROVED, PROVED, Prover, fill, reference_proof

STATEMENT: str = "def Cut (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j ≤ 1"
PROOF: str = "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n  rcases h.hx_bin i j with a | a <;> omega"
FAITHFUL: str = "VERDICT: FAITHFUL\nREASON: same ranges."


def block(text: str) -> str:
    return f"Here it is.\n```lean\n{text}\n```"


def make(llm: FakeLLM, tmp_path: Path, **options) -> Prover:
    attempts: List[dict] = []
    prover = Prover(llm, LEAN_PROBLEMS["tsp"], inspect.getsource(tsp.create_model), str(tmp_path),
                    record=attempts.append, **options)
    prover.attempts = attempts
    return prover


@pytest.fixture
def no_lean(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Lean replaced: a text containing BAD does not compile, anything else is fine / proved."""
    compiled: List[str] = []

    def compile_file(path: str, **kwargs) -> Compilation:
        text: str = open(path).read()
        compiled.append(text)
        if "BAD" in text:
            line: int = next(k for k, row in enumerate(text.splitlines(), 1) if "BAD" in row)
            return Compilation(1, [LeanMessage("error", line, 0, "unknown identifier 'BAD'")], seconds=1.0)
        return Compilation(0, seconds=1.0)

    def verify(problem, statement, proof, folder, timeout=0, lean_dir="") -> Verdict:
        text, offset = certificate.proof_file(problem, statement, proof)
        if lean.forbidden(proof):
            return Verdict(False, "the proof uses refused words: " + ", ".join(lean.forbidden(proof)))
        compilation: Compilation = compile_file_text(text)
        if not compilation.ok:
            return Verdict(False, "the proof does not compile (1 errors)", claim="valid", compilation=compilation, offset=offset)
        return Verdict(True, "ValidCut proved, axioms: propext", claim="valid", axioms=["propext"], compilation=compilation)

    def compile_file_text(text: str) -> Compilation:
        compiled.append(text)
        if "BAD" in text:
            line: int = next(k for k, row in enumerate(text.splitlines(), 1) if "BAD" in row)
            return Compilation(1, [LeanMessage("error", line, 0, "unknown identifier 'BAD'")], seconds=1.0)
        return Compilation(0, seconds=1.0)

    monkeypatch.setattr(lean, "compile_file", compile_file)
    monkeypatch.setattr(certificate, "verify", verify)
    return compiled


def test_fill_replaces_placeholders_once() -> None:
    assert fill("a <<x>> b <<y>>", x="<<y>>", y="{z}") == "a <<y>> b {z}"
    with pytest.raises(KeyError, match="idea"):
        fill("<<idea>>", code="c")


def test_every_template_fills_and_shows_the_model(tmp_path: Path) -> None:
    prover = make(FakeLLM([]), tmp_path)
    values = dict(prover.values, idea="IDEA", code="CODE", statement="STATEMENT", feedback="FEEDBACK",
                  marker="MARKER", common_path="C", lean_model_path="L", example_path="E")
    for name, template in prover.templates.items():
        assert "<<" not in fill(template, **values), name
    system: str = fill(prover.templates["proof_system"], **values)
    assert "structure MILPFormulation" in system and "namespace EvoCut.Tsp" in system
    assert "def create_model" in system and "private lemma eq_zero_of_sum_one" in system
    assert "namespace EvoCut.Tsp.CutProof" in system
    assert reference_proof().endswith("omega") and "def Cut" not in reference_proof()


def test_statement_then_judge_then_proof(tmp_path: Path, no_lean: List[str]) -> None:
    llm = FakeLLM([block("def Cut := BAD"), block(STATEMENT), "VERDICT: UNFAITHFUL\nREASON: i < j is missing.",
                   block(STATEMENT + "\n-- fixed"), FAITHFUL,
                   block("theorem cut_valid : formulation.ValidCut Cut := by BAD"), block(PROOF)])
    prover = make(llm, tmp_path)
    result = prover.run("the idea", "the code")
    assert result.proved and result.status == PROVED and result.claim == "valid" and result.axioms == ["propext"]
    assert result.statement == STATEMENT + "\n-- fixed" and result.proof == PROOF
    assert (result.statement_rounds, result.proof_rounds, result.llm_calls) == (3, 2, 7)
    assert [(a["phase"], a["round"], a["outcome"]) for a in prover.attempts] == [
        ("statement", 1, "refused"), ("statement", 2, "refused"), ("statement", 3, "accepted"),
        ("proof", 1, "refused"), ("proof", 2, PROVED)]
    # the compiler's error and the judge's reason were sent back
    second, third = llm.received[1], llm.received[3]
    assert "unknown identifier 'BAD'" in second[-1]["content"] and "line 1" in second[-1]["content"]
    assert "i < j is missing" in third[-1]["content"]
    # the judge sees the code and the statement, in its own conversation
    judge = llm.received[2]
    assert len(judge) == 2 and "the code" in judge[1]["content"] and STATEMENT in judge[1]["content"]
    # the proof conversation holds the frozen statement, and the retry the error with its line
    retry = llm.received[6]
    assert STATEMENT + "\n-- fixed" in retry[1]["content"]
    assert "error, line 1: `theorem cut_valid : formulation.ValidCut Cut := by BAD`" in retry[-1]["content"]
    assert result.lean_seconds > 0 and result.to_dict()["status"] == PROVED


def test_no_statement_after_the_rounds(tmp_path: Path, no_lean: List[str]) -> None:
    llm = FakeLLM(["no code at all", block("def Cut := BAD")])
    result = make(llm, tmp_path, statement_rounds=2).run("idea", "code")
    assert result.status == NO_STATEMENT and not result.proved and result.statement is None
    assert "statement round 2" in result.reason and result.proof_rounds == 0
    assert "no ```lean block" in llm.received[1][-1]["content"]


def test_a_statement_with_a_theorem_or_a_refused_word_is_not_compiled(tmp_path: Path, no_lean: List[str]) -> None:
    prover = make(FakeLLM([]), tmp_path)
    assert "no theorem" in prover.check_statement(STATEMENT + "\ntheorem t : True := trivial")
    assert "refused words: axiom" in prover.check_statement("axiom a : False\n" + STATEMENT)
    assert "only Mathlib" in prover.check_statement("import Foo.Bar\n" + STATEMENT)
    assert no_lean == []
    assert prover.check_statement(STATEMENT) is None and len(no_lean) == 1
    assert no_lean[0].rstrip().endswith("example : formulation.Cut := Cut")


def test_not_proved_keeps_the_last_proof_and_only_recent_history(tmp_path: Path, no_lean: List[str]) -> None:
    bad = [block(f"theorem cut_valid : formulation.ValidCut Cut := by BAD -- try {k}") for k in range(1, 5)]
    llm = FakeLLM([block(STATEMENT), *bad])
    result = make(llm, tmp_path, judge=False, proof_rounds=4, history=2).run("idea", "code")
    assert result.status == NOT_PROVED and not result.proved and result.judge is None
    assert "try 4" in result.proof and "proof round 4" in result.reason
    last = llm.received[-1]  # system, task, 2 exchanges (tries 2 and 3)
    assert len(last) == 2 + 2 * 2 and "try 2" in last[2]["content"] and "try 3" in last[4]["content"]
    assert all("try 1" not in m["content"] for m in last)


def test_sorry_is_compiled_for_feedback_but_never_accepted(tmp_path: Path, no_lean: List[str]) -> None:
    sorry: str = "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h\n  sorry"
    llm = FakeLLM([block(sorry), block(PROOF)])
    prover = make(llm, tmp_path, judge=False)
    result = prover.run("idea", "code", statement=STATEMENT)
    assert result.proved and result.proof_rounds == 2 and result.statement_rounds == 0
    assert prover.attempts[0]["outcome"] == "refused" and "still contains `sorry`" in prover.attempts[0]["feedback"]


def test_the_llm_may_answer_that_the_cut_is_invalid(tmp_path: Path, no_lean: List[str]) -> None:
    llm = FakeLLM(["INVALID: with 3 cities the tour 1 -> 2 -> 3 -> 1 violates it."])
    result = make(llm, tmp_path, judge=False).run("idea", "code", statement=STATEMENT)
    assert result.status == CLAIMED_INVALID and not result.proved and "3 cities" in result.reason


def test_a_given_statement_that_does_not_compile_stops_before_any_call(tmp_path: Path, no_lean: List[str]) -> None:
    llm = FakeLLM([])
    result = make(llm, tmp_path).run("idea", "code", statement="def Cut := BAD")
    assert result.status == NO_STATEMENT and llm.received == [] and "does not compile" in result.reason


def test_a_statement_wrapped_in_the_header_is_unwrapped(tmp_path: Path, no_lean: List[str]) -> None:
    wrapped: str = ("import EvoCut.Problems.Tsp\nopen BigOperators Finset\nnamespace EvoCut.Tsp.CutProof\n"
                    + STATEMENT + "\nend EvoCut.Tsp.CutProof")
    llm = FakeLLM([block(wrapped), block(PROOF + "\nend EvoCut.Tsp.CutProof")])
    result = make(llm, tmp_path, judge=False).run("idea", "code")
    assert result.proved and result.statement == STATEMENT and result.proof == PROOF
    assert result.statement_rounds == 1


FAMILIES: str = ("/-- an arc is used at most once -/\ndef Family_upper (p : Params) (v : Vars p) : Prop :=\n"
                 "  ∀ i j : Fin p.n, v.x i j ≤ 1\n\n/-- an arc is not negative -/\n"
                 "def Family_lower (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, 0 ≤ v.x i j")


def test_a_cut_stated_and_proved_family_by_family(tmp_path: Path, no_lean: List[str]) -> None:
    llm = FakeLLM([block(FAMILIES + "\n\ndef Cut (p : Params) (v : Vars p) : Prop := True"), block(FAMILIES),
                   block(PROOF), block("theorem cut_valid : formulation.ValidCut Cut := by BAD"), block(PROOF)])
    prover = make(llm, tmp_path, judge=False, by_families=True)
    result = prover.run("idea", "code")
    assert result.proved and result.claim == "valid" and result.statement_rounds == 2 and result.proof_rounds == 3
    assert "do not define `Cut`" in llm.received[1][-1]["content"] and "BY FAMILIES" in llm.received[0][1]["content"]
    assert result.statement.endswith("def Cut (p : Params) (v : Vars p) : Prop :=\n  Family_upper p v ∧ Family_lower p v")
    assert [(f["name"], f["status"], f["proof_rounds"]) for f in result.families] == [
        ("Family_upper", PROVED, 1), ("Family_lower", PROVED, 2)]
    # each family is proved on a statement whose Cut is that family alone
    assert llm.received[2][1]["content"].count("Family_upper p v") == 1 and "Family_upper p v ∧" not in llm.received[2][1]["content"]
    assert "the family `Family_lower`" in llm.received[3][1]["content"]
    assert "namespace Part1" in result.proof and "⟨Part1.cut_valid p v h, Part2.cut_valid p v h⟩" in result.proof
    assert [a["phase"] for a in prover.attempts] == ["statement", "statement", "proof_Family_upper",
                                                    "proof_Family_lower", "proof_Family_lower"]


def test_families_not_all_proved_are_named(tmp_path: Path, no_lean: List[str]) -> None:
    bad: str = block("theorem cut_valid : formulation.ValidCut Cut := by BAD")
    llm = FakeLLM([block(PROOF), bad, bad])
    statement: str = certificate.with_cut(FAMILIES, ["Family_upper", "Family_lower"])
    result = make(llm, tmp_path, judge=False, by_families=True, proof_rounds=2).run("idea", "code", statement=statement)
    assert not result.proved and result.reason == "1 of 2 families proved valid; not proved: Family_lower"
    assert [f["status"] for f in result.families] == [PROVED, NOT_PROVED] and result.proof is None


def test_a_statement_without_families_is_proved_whole(tmp_path: Path, no_lean: List[str]) -> None:
    result = make(FakeLLM([block(PROOF)]), tmp_path, judge=False, by_families=True).run("idea", "code", statement=STATEMENT)
    assert result.proved and result.families == []
