import json
import subprocess
from typing import Any, Dict, List

import pytest

from llm import claude_code
from llm.base import LLMError
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
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Done("", 1, "not logged in"))
    with pytest.raises(LLMError, match="not logged in"):
        ClaudeCodeClient("opus", max_retries=1).send([{"role": "user", "content": "q"}])
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: Done(json.dumps({"is_error": True, "result": "limit reached"})))
    with pytest.raises(LLMError, match="limit reached"):
        ClaudeCodeClient("opus", max_retries=1).send([{"role": "user", "content": "q"}])
