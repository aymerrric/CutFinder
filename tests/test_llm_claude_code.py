import json
import subprocess
from typing import Any, Dict, List

import pytest

from llm import claude_code
from llm.base import LLMError, QuotaExhausted, UsageLimit
from llm.claude_code import ClaudeCodeClient, as_prompt


class Done:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_a_conversation_as_one_prompt() -> None:
    assert as_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]) == "Q"
    text: str = as_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                           {"role": "assistant", "content": "A"}, {"role": "user", "content": "errors"}])
    assert text == "=== USER ===\nQ\n\n=== YOUR EARLIER ANSWER ===\nA\n\n=== USER ===\nerrors"


def test_one_question_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def run(command: List[str], **kwargs: Any) -> Done:
        system: str = (open(command[command.index("--system-prompt-file") + 1]).read()
                       if "--system-prompt-file" in command else "")
        calls.append({"command": command, "input": kwargs["input"], "system": system,
                      "cwd": kwargs["cwd"]})
        return Done(json.dumps({"result": "the answer", "total_cost_usd": 0.25, "duration_ms": 1200}))

    monkeypatch.setattr(claude_code, "claude", lambda: "/bin/claude")
    monkeypatch.setattr(subprocess, "run", run)
    client = ClaudeCodeClient("sonnet", effort="high")
    assert client.send([{"role": "system", "content": "be exact"}, {"role": "user", "content": "prove it"}]) == "the answer"
    command: List[str] = calls[0]["command"]
    assert command[:4] == ["/bin/claude", "-p", "--model", "sonnet"] and calls[0]["input"] == "prove it"
    assert command[command.index("--tools") + 1] == "" and "--strict-mcp-config" in command
    assert command[command.index("--effort") + 1] == "high" and "--no-session-persistence" in command
    assert calls[0]["system"] == "be exact" and client.cost_usd == 0.25
    client.send([{"role": "user", "content": "again"}])
    assert client.cost_usd == 0.5 and calls[1]["system"] == "" and "--system-prompt-file" not in calls[1]["command"]


def test_a_failed_call_is_an_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_code, "claude", lambda: "/bin/claude")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Done("", 1, "segmentation fault"))
    with pytest.raises(LLMError, match="segmentation fault"):
        ClaudeCodeClient("opus", max_retries=1).send([{"role": "user", "content": "q"}])
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Done(json.dumps({"is_error": True, "result": "overloaded"})))
    with pytest.raises(LLMError, match="overloaded"):
        ClaudeCodeClient("opus", max_retries=1).send([{"role": "user", "content": "q"}])


@pytest.mark.parametrize("answer, resumed", [
    (Done(json.dumps({"is_error": True, "result": "Claude AI usage limit reached|1763740800"})), True),
    (Done("", 1, "You've hit your limit · resets 3pm (Europe/Paris)"), True),
    (Done("", 1, "Invalid API key · Please run /login"), False),
])
def test_a_subscription_limit_stops_the_run(monkeypatch: pytest.MonkeyPatch, answer: Done, resumed: bool) -> None:
    monkeypatch.setattr(claude_code, "claude", lambda: "/bin/claude")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: answer)
    with pytest.raises(QuotaExhausted) as stop:  # not retried, not a copy: the run stops
        ClaudeCodeClient("sonnet", max_retries=3, retry_delay=0).send([{"role": "user", "content": "q"}])
    assert isinstance(stop.value, UsageLimit) == resumed  # a limit is resumed by the night job, a login error not


def test_the_token_of_the_keys_file_logs_claude_in(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    keys = tmp_path / "llm_keys.env"
    keys.write_text("MUSE_API_KEY=x\nCLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat-123\n")
    monkeypatch.delenv(claude_code.TOKEN_VARIABLE, raising=False)
    monkeypatch.setattr(claude_code, "KEYS_FILE", str(keys))
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs["env"])
        return Done(json.dumps({"result": "ok"}))

    monkeypatch.setattr(claude_code, "claude", lambda: "/bin/claude")
    monkeypatch.setattr(subprocess, "run", run)
    ClaudeCodeClient("sonnet").send([{"role": "user", "content": "q"}])
    assert seen[claude_code.TOKEN_VARIABLE] == "sk-ant-oat-123"
    keys.write_text("MUSE_API_KEY=x\n")  # no token: Claude Code's own login
    seen.clear()
    ClaudeCodeClient("sonnet").send([{"role": "user", "content": "q"}])
    assert claude_code.TOKEN_VARIABLE not in seen


# An answer of `claude -p --model sonnet --output-format json` recorded on the lab (23 Nov 2026,
# Claude Code 2.1.330), cut to the fields the client reads: the alias "sonnet" is answered by
# the model named in modelUsage.
RECORDED_ANSWER: Dict[str, Any] = {
    "type": "result", "subtype": "success", "is_error": False, "duration_ms": 2412, "num_turns": 1,
    "result": "ok", "total_cost_usd": 0.0003302,
    "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 2, "outputTokens": 4, "costUSD": 0.0003302,
                                          "canonicalModel": "claude-sonnet-5-5", "provider": "firstParty"}},
}


def test_the_exact_model_and_the_effort_are_recorded(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(claude_code, "claude", lambda: "/bin/claude")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Done(json.dumps(RECORDED_ANSWER)))
    client = ClaudeCodeClient("sonnet")
    assert client.identity() == {"llm_model_id": None, "llm_effort": "default"}  # nothing answered yet
    with caplog.at_level("INFO", logger="llm.claude_code"):
        assert client.send([{"role": "user", "content": "Say ok"}]) == "ok"
    assert client.identity() == {"llm_model_id": "claude-sonnet-5-5", "llm_effort": "default"}
    assert "claude sonnet (claude-sonnet-5-5, effort default)" in caplog.text  # run.log
    client.send([{"role": "user", "content": "again"}])
    assert client.model_ids == ["claude-sonnet-5-5"]  # once, however many calls
    assert ClaudeCodeClient("opus", effort="high").identity()["llm_effort"] == "high"
