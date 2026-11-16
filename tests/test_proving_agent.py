import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import pytest

from models import tsp
from proving import agent as agent_module
from proving import certificate
from proving.agent import MARKER, AgentProver, agent_file, command, last_result, proof_of
from proving.certificate import Verdict
from proving.problems import LEAN_PROBLEMS

TSP = LEAN_PROBLEMS["tsp"]
STATEMENT: str = "def Cut (p : Params) (v : Vars p) : Prop :=\n  ∀ i j : Fin p.n, v.x i j ≤ 1"
PROOF: str = "theorem cut_valid : formulation.ValidCut Cut := by\n  intro p v h i j\n  rcases h.hx_bin i j with a | a <;> omega"


def test_the_agents_file_and_the_proof_read_back() -> None:
    text: str = agent_file(TSP, STATEMENT)
    assert text.startswith("import Mathlib.Tactic\nimport EvoCut.Problems.Tsp\n")
    assert STATEMENT in text and text.index(STATEMENT) < text.index(MARKER) < text.index("sorry")
    assert proof_of(text).endswith("sorry")
    edited: str = text.split(MARKER)[0].replace("≤ 1", "≤ 2") + MARKER + "\n\n" + PROOF + "\n"
    assert proof_of(edited) == PROOF  # what is above the marker is never read back
    assert proof_of("the agent deleted everything") is None and proof_of(text.split(MARKER)[0] + MARKER + "\n") is None


def test_the_agent_command_allows_only_its_tools() -> None:
    line: List[str] = command("opus", "medium", "/wd/.mcp.json", 2.5)
    assert line[1:4] == ["-p", "--model", "opus"] and line[line.index("--permission-mode") + 1] == "acceptEdits"
    allowed: List[str] = line[line.index("--allowedTools") + 1:line.index("--disallowedTools")]
    assert "Bash(./check.sh)" in allowed and "Bash(/wd/check.sh)" in allowed and "mcp__lean-lsp" in allowed
    assert "Bash" not in allowed
    assert "--strict-mcp-config" in line and line[line.index("--max-budget-usd") + 1] == "2.5"
    assert "--dangerously-skip-permissions" not in line and "bypassPermissions" not in line
    assert "--max-budget-usd" not in command("opus", None, "x", None)


def test_the_last_result_event(tmp_path: Path) -> None:
    path: Path = tmp_path / "out.jsonl"
    path.write_text('{"type": "assistant"}\nnot json\n{"type": "result", "total_cost_usd": 1.2, "num_turns": 40}\n')
    assert last_result(str(path)) == {"type": "result", "total_cost_usd": 1.2, "num_turns": 40}
    assert last_result(str(tmp_path / "none")) == {}


class FakeAgent:
    """Stands for `claude -p`: writes `proof` under the marker of CutProof.lean in its folder."""

    def __init__(self, proof: str) -> None:
        self.proof: str = proof
        self.task: str = ""
        self.returncode: int = 0
        self.pid: int = 0

    def __call__(self, line: List[str], cwd: str, **kwargs: Any) -> "FakeAgent":
        self.cwd, self.line = cwd, line
        return self

    def communicate(self, task: Optional[str] = "", timeout: float = 0) -> None:
        self.task = task
        path: str = os.path.join(self.cwd, "CutProof.lean")
        text: str = open(path).read()
        with open(path, "w") as f:
            f.write(text.split(MARKER)[0] + MARKER + "\n\n" + self.proof + "\n")


def make(tmp_path: Path) -> AgentProver:
    lean_dir: Path = tmp_path / "lean"
    for name in ("Common.lean", "Problems/Tsp.lean", "Cuts/TspSeed.lean"):
        (lean_dir / "EvoCut" / name).parent.mkdir(parents=True, exist_ok=True)
        (lean_dir / "EvoCut" / name).write_text("-- stub")
    return AgentProver(TSP, inspect.getsource(tsp.create_model), "arc", lean_dir=str(lean_dir), minutes=1)


def test_the_agents_proof_goes_through_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent(PROOF)
    monkeypatch.setattr(subprocess, "Popen", fake)
    verified: List[Any] = []

    def verify(problem, statement, proof, folder, timeout=0, lean_dir="") -> Verdict:
        verified.append((statement, proof, folder))
        return Verdict(True, "ValidCut proved, axioms: propext", claim="valid", axioms=["propext"])

    monkeypatch.setattr(certificate, "verify", verify)
    prover = make(tmp_path)
    result = prover.run("the idea", "the code", STATEMENT, str(tmp_path / "out"))
    assert result.proved and result.claim == "valid" and result.proof == PROOF and result.statement == STATEMENT
    assert verified == [(STATEMENT, PROOF, str(tmp_path / "out" / "agent_proof"))]
    folder: Path = tmp_path / "lean" / "work" / "tsp_arc"
    assert fake.cwd == str(folder) and os.access(folder / "check.sh", os.X_OK)
    assert f"lake env lean '{folder}/CutProof.lean'" in (folder / "check.sh").read_text()
    mcp = json.load(open(folder / ".mcp.json"))
    assert mcp["mcpServers"]["lean-lsp"]["env"]["LEAN_PROJECT_PATH"] == str(tmp_path / "lean")
    assert "the idea" in fake.task and "the code" in fake.task and MARKER in fake.task
    assert "def create_model" in fake.task and str(tmp_path / "lean" / "EvoCut" / "Problems" / "Tsp.lean") in fake.task
    assert (folder / "task.md").read_text() == fake.task and "<<" not in fake.task


def test_an_agent_that_leaves_no_proof_or_a_wrong_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", FakeAgent(""))
    monkeypatch.setattr(certificate, "verify", lambda *a, **k: pytest.fail("nothing to verify"))
    result = make(tmp_path).run("idea", "code", STATEMENT, str(tmp_path / "out"))
    assert not result.proved and "no proof under the marker" in result.reason
    monkeypatch.setattr(subprocess, "Popen", FakeAgent(PROOF))
    monkeypatch.setattr(certificate, "verify", lambda *a, **k: Verdict(False, "the proof does not compile (2 errors)"))
    result = make(tmp_path).run("idea", "code", STATEMENT, str(tmp_path / "out"))
    assert not result.proved and result.status == "not_proved" and "does not compile" in result.reason


def test_an_agent_over_its_time_is_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow(FakeAgent):
        def communicate(self, task: str = "", timeout: float = 0) -> None:
            if timeout:
                raise subprocess.TimeoutExpired("claude", timeout)

    killed: List[int] = []
    monkeypatch.setattr(subprocess, "Popen", Slow(""))
    monkeypatch.setattr(agent_module.os, "killpg", lambda pid, sig: killed.append(pid))
    result = make(tmp_path).run("idea", "code", STATEMENT, str(tmp_path / "out"))
    assert killed == [0] and "stopped after 1 minutes" in result.reason and not result.proved


def test_an_agent_that_answers_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", FakeAgent("-- INVALID: n = 3, the tour 1 -> 2 -> 3 -> 1"))
    monkeypatch.setattr(certificate, "verify", lambda *a, **k: pytest.fail("a claim is not verified"))
    result = make(tmp_path).run("idea", "code", STATEMENT, str(tmp_path / "out"))
    assert result.status == "claimed_invalid" and result.reason == "n = 3, the tour 1 -> 2 -> 3 -> 1"


def test_vibe_settings_and_command(tmp_path: Path) -> None:
    home: str = agent_module.vibe_setup(str(tmp_path))
    config: str = open(os.path.join(home, "config.toml")).read()
    assert home == str(tmp_path / "work" / "vibe_home")
    assert 'command = "uvx"' in config and 'args = ["lean-lsp-mcp", "--disable-tools", "lean_loogle"]' in config and f'LEAN_PROJECT_PATH = "{tmp_path}"' in config
    assert 'api_key_env_var = "REQUESTY_API_KEY"' in config and 'name = "mistral/leanstral-1-5"' in config
    agent: str = open(os.path.join(home, "agents", "lean-requesty.toml")).read()
    assert 'system_prompt_id = "lean"' in agent and '"bash"' in agent and '"web_fetch"' in agent
    line: List[str] = agent_module.vibe_command(40)
    assert line[1:3] == ["--agent", "lean-requesty"] and "--trust" in line and line[-1] == "-p"
    assert line[line.index("--max-turns") + 1] == "40"


def test_a_vibe_agent_gets_the_task_as_an_argument_and_no_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent(PROOF)
    seen: dict = {}

    def popen(line: List[str], cwd: str, **kwargs: Any) -> FakeAgent:
        seen.update(kwargs)
        seen["line"] = line
        return fake(line, cwd)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(certificate, "verify", lambda *a, **k: Verdict(True, "ValidCut proved", claim="valid"))
    lean_dir: Path = tmp_path / "lean"
    for name in ("Common.lean", "Problems/Tsp.lean", "Cuts/TspSeed.lean"):
        (lean_dir / "EvoCut" / name).parent.mkdir(parents=True, exist_ok=True)
        (lean_dir / "EvoCut" / name).write_text("-- stub")
    prover = AgentProver(TSP, inspect.getsource(tsp.create_model), "arc", lean_dir=str(lean_dir), minutes=1,
                         cli="vibe", api="requesty", env={"REQUESTY_API_KEY": "k"})
    result = prover.run("the idea", "the code", STATEMENT, str(tmp_path / "out"))
    assert result.proved and seen["stdin"] == subprocess.DEVNULL
    assert "the idea" in seen["line"][-1] and seen["env"]["REQUESTY_API_KEY"] == "k"
    assert seen["env"]["VIBE_HOME"].endswith("work/vibe_home") and fake.task is None


def test_leanstral_on_mistrals_api_by_default(tmp_path: Path) -> None:
    base, model, variable = agent_module.LEANSTRAL_APIS["mistral"]
    assert base == "https://api.mistral.ai/v1" and model == "labs-leanstral-1-5" and variable == "MISTRAL_API_KEY"
    config: str = open(os.path.join(agent_module.vibe_setup(str(tmp_path), model=model, key_variable=variable),
                                    "config.toml")).read()
    assert 'name = "labs-leanstral-1-5"' in config and 'api_key_env_var = "MISTRAL_API_KEY"' in config
    assert AgentProver.__init__.__defaults__ is not None
