import json
import os
from pathlib import Path

import pytest

from proving import lean

needs_lean = pytest.mark.skipif(lean.unavailable() is not None, reason=str(lean.unavailable()))


def test_messages_are_read_from_the_json_lines() -> None:
    output: str = "\n".join([
        json.dumps({"severity": "error", "pos": {"line": 7, "column": 2}, "data": "linarith failed"}),
        "some text of lake",
        json.dumps({"severity": "warning", "pos": {"line": 3, "column": 8}, "data": "declaration uses `sorry`"}),
    ])
    result = lean.parse_messages(output)
    assert [(m.severity, m.line, m.column) for m in result.messages] == [("error", 7, 2), ("warning", 3, 8)]
    assert result.other_output == "some text of lake" and not result.ok and result.uses_sorry
    assert [m.text for m in result.errors] == ["linarith failed"]


def test_axioms_of_a_print_axioms_message() -> None:
    assert lean.axioms_in("'a.b' depends on axioms: [propext, Classical.choice, Quot.sound]") == [
        "propext", "Classical.choice", "Quot.sound"]
    assert lean.axioms_in("'a' depends on axioms: [sorryAx]") == ["sorryAx"]
    assert lean.axioms_in("'a' does not depend on any axioms") == []
    assert lean.axioms_in("unknown constant") is None
    assert set(lean.axioms_in("'a' depends on axioms: [propext, Quot.sound]")) <= lean.STANDARD_AXIOMS


def test_comments_are_removed() -> None:
    text: str = "a -- sorry\n/- axiom /- nested -/ still -/ b\nc"
    assert lean.strip_comments(text).split() == ["a", "b", "c"]


def test_refused_words() -> None:
    assert lean.forbidden("theorem t : True := by\n  sorry") == ["sorry"]
    assert lean.forbidden("theorem t : True := by\n  sorry", allow_sorry=True) == []
    assert lean.forbidden("-- no sorry here\ntheorem t : True := trivial") == []
    assert lean.forbidden("axiom cheat : False") == ["axiom"]
    assert lean.forbidden("local notation \"x\" => 1\n#eval 1") == ["notation", "#eval"]
    assert lean.forbidden("lemma no_axiom_here (h : my_sorry_free) : True := trivial") == []
    assert lean.forbidden("set_option maxHeartbeats 400000 in\nset_option linter.unusedVariables false") == []
    assert lean.forbidden("set_option debug.skipKernelTC true") == ["set_option debug.skipKernelTC"]
    assert lean.forbidden("example : 2 + 2 = 4 := by native_decide") == ["native_decide"]


def test_unavailable_says_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lean, "lake", lambda: None)
    assert "not installed" in lean.unavailable(str(tmp_path))
    monkeypatch.setattr(lean, "lake", lambda: "/bin/lake")
    assert "Mathlib is not downloaded" in lean.unavailable(str(tmp_path))
    os.makedirs(tmp_path / ".lake" / "packages" / "mathlib")
    assert "lake build" in lean.unavailable(str(tmp_path))


@needs_lean
def test_a_file_is_compiled_and_its_errors_reported(tmp_path: Path) -> None:
    path: Path = tmp_path / "T.lean"
    path.write_text("import Mathlib.Tactic\n\nexample (a b : ℝ) (h : a ≤ b) : a - b ≤ 0 := by linarith\n"
                    "example (a : ℝ) : a ≤ 0 := by linarith\n")
    result = lean.compile_file(str(path))
    assert not result.ok and not result.timed_out and result.seconds > 0
    assert [m.line for m in result.errors] == [4] and "linarith failed" in result.errors[0].text


@needs_lean
def test_a_compile_is_killed_at_the_timeout(tmp_path: Path) -> None:
    path: Path = tmp_path / "T.lean"
    path.write_text("import Mathlib.Tactic\nexample : True := trivial\n")
    result = lean.compile_file(str(path), timeout=0.2)
    assert result.timed_out and not result.ok
