import json
import os
from pathlib import Path

import pytest

from evolution.individual import Individual
from evolution.population import Population
from llm.fake import FakeLLM
from problems import PROBLEMS
from prove_cut import checked_sets, cut_source, parse_args, run
from proving import lean
from proving.problems import LEAN_PROBLEMS

needs_lean = pytest.mark.skipif(lean.unavailable() is not None, reason=str(lean.unavailable()))

STATEMENT: str = "/-- an arc is used at most once -/\ndef Cut (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j ≤ 1"
WRONG: str = "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n  linarith"
PROOF: str = "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n  rcases h.hx_bin i j with a | a <;> omega"


def block(text: str) -> str:
    return f"```lean\n{text}\n```"


def test_the_cut_named_by_the_arguments(tmp_path: Path) -> None:
    tsp = PROBLEMS["tsp"]
    assert cut_source(parse_args(["--cut", "seed"]), tsp) == ("seed", tsp.baseline_idea, tsp.baseline_code)
    cut_file: Path = tmp_path / "my cut.py"
    cut_file.write_text("m.addConstr(x >= 0)")
    assert cut_source(parse_args(["--cut_file", str(cut_file), "--idea", "why"]), tsp) == (
        "my-cut", "why", "m.addConstr(x >= 0)")
    assert cut_source(parse_args(["--cut_file", str(cut_file), "--label", "paper"]), tsp)[0] == "paper"
    run_dir: Path = tmp_path / "tsp_qwen_host_20261101_000000"
    run_dir.mkdir()
    population = Population(size=1)
    population.set_generation([Individual(code="best code", idea="best idea", fitness=12.0, generator="seed")])
    population.save(str(run_dir / "population.json"))
    label, idea, code = cut_source(parse_args(["--population_file", str(run_dir / "population.json")]), tsp)
    assert (label, idea, code) == ("tsp_qwen_host_20261101_000000", "best idea", "best code")
    for argv in ([], ["--cut", "seed", "--cut_file", str(cut_file)]):
        with pytest.raises(ValueError, match="name one cut"):
            cut_source(parse_args(argv), tsp)
    with pytest.raises(FileNotFoundError):
        cut_source(parse_args(["--population_file", str(tmp_path / "none.json")]), tsp)


def test_data_assumptions_are_checked_on_the_sets_found(tmp_path: Path) -> None:
    assert checked_sets(LEAN_PROBLEMS["tsp"], str(tmp_path)) == {}
    loader = PROBLEMS["tsp"].loader(str(tmp_path / "validity"))
    os.makedirs(loader.folder)
    loader.save([{"name": "ok", "args": {"n": 4, "c": {(1, 2): 1.0}}}, {"name": "two", "args": {"n": 2, "c": {}}}],
                "validity.pkl")
    report = checked_sets(LEAN_PROBLEMS["tsp"], str(tmp_path))
    assert report == {"validity/tsp/validity.pkl": {"instances": 2, "broken": {"two": ["hn: 2 cities, at least 4 are assumed"]}}}


def test_without_lean_the_script_says_what_to_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lean, "unavailable", lambda: "Lean is not installed")
    with pytest.raises(RuntimeError, match="not installed"):
        run(parse_args(["--cut", "seed", "--out_dir", str(tmp_path)]), llm=FakeLLM([]))


@needs_lean
def test_a_cut_is_stated_and_proved_end_to_end(tmp_path: Path) -> None:
    llm = FakeLLM([block(STATEMENT), "VERDICT: FAITHFUL\nREASON: ok.", block(WRONG), block(PROOF)])
    cut_file: Path = tmp_path / "arc_bound.py"
    cut_file.write_text("m.addConstrs(m._x[i, j] <= 1 for (i, j) in m._x.keys())")
    args = parse_args(["--problem_name", "tsp", "--cut_file", str(cut_file), "--idea", "x is binary",
                       "--out_dir", str(tmp_path / "proofs"), "--data_dir", str(tmp_path / "data")])
    result = run(args, llm=llm)
    assert result.proved and result.claim == "valid" and set(result.axioms) <= lean.STANDARD_AXIOMS
    assert (result.statement_rounds, result.proof_rounds, result.llm_calls) == (1, 2, 4)
    folder: Path = tmp_path / "proofs" / "tsp" / "arc_bound"
    saved = json.load(open(folder / "result.json"))
    assert saved["status"] == "proved" and saved["label"] == "arc_bound" and saved["idea"] == "x is binary"
    assert saved["judge_asked"] and saved["data_assumptions"] == {} and saved["judge"].startswith("VERDICT: FAITHFUL")
    assert (folder / "statement.lean").read_text().strip() == STATEMENT
    assert (folder / "proof.lean").read_text().strip() == PROOF
    assert sorted(os.listdir(folder / "attempts")) == ["proof_01.json", "proof_02.json", "statement_01.json"]
    first = json.load(open(folder / "attempts" / "proof_01.json"))
    assert first["outcome"] == "refused" and "linarith failed" in first["feedback"] and "error, line 3" in first["feedback"]
    assert (folder / "proof_02" / "Check.lean").exists()
    # the errors of the first proof reached the LLM
    assert "linarith failed" in llm.received[3][-1]["content"]
