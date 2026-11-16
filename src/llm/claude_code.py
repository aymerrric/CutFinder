"""
An LLMClient that asks Claude through the Claude Code command line (`claude -p`), with no tool:
one question, one answer. It uses the account Claude Code is logged in with (a subscription), so
it needs no API key. Used by prove_cut.py (--llm_backend claude_code), not by the genetic algorithm.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, Message
from utils import parameters

logger: logging.Logger = logging.getLogger(__name__)


def claude() -> Optional[str]:
    """The path of the `claude` command, or None when Claude Code is not installed."""
    return shutil.which("claude") or shutil.which(os.path.expanduser("~/.local/bin/claude"))


def as_prompt(messages: List[Message]) -> str:
    """The conversation after the system message, as one text: a single question is sent as it
    is, a longer conversation with a line naming who speaks before each message."""
    turns: List[Message] = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return turns[0]["content"]
    names: Dict[str, str] = {"user": "USER", "assistant": "YOUR EARLIER ANSWER"}
    return "\n\n".join(f"=== {names.get(m['role'], m['role'].upper())} ===\n{m['content']}" for m in turns)


class ClaudeCodeClient(LLMClient):
    """`model`: an alias ("opus", "sonnet", "fable", "haiku") or a full model name."""

    def __init__(self, model: str, effort: Optional[str] = None, timeout: float = parameters.LLM_TIMEOUT,
                 max_retries: int = 2, retry_delay: float = parameters.LLM_RETRY_DELAY) -> None:
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model: str = model
        self.effort: Optional[str] = effort
        self.timeout: float = timeout
        self.cost_usd: float = 0.0  # what the calls would cost at API prices (reported by the command)

    def command(self, system_file: Optional[str]) -> List[str]:
        """No tool, no MCP server, no skill, nothing saved: a plain question."""
        command: List[str] = [claude() or "claude", "-p", "--model", self.model, "--output-format", "json",
                              "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
                              "--no-session-persistence"]
        if self.effort:
            command += ["--effort", self.effort]
        if system_file:
            command += ["--system-prompt-file", system_file]
        return command

    def _send(self, messages: List[Message]) -> str:
        system: str = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        with tempfile.TemporaryDirectory() as folder:  # an empty folder: no CLAUDE.md is picked up
            system_file: Optional[str] = None
            if system:
                system_file = os.path.join(folder, "system.txt")
                with open(system_file, "w") as f:
                    f.write(system)
            done: subprocess.CompletedProcess = subprocess.run(
                self.command(system_file), input=as_prompt(messages), capture_output=True, text=True,
                timeout=self.timeout, cwd=folder)
        if done.returncode != 0:
            raise RuntimeError(f"claude exited with code {done.returncode}: {(done.stderr or done.stdout)[-500:]}")
        answer: Dict[str, Any] = json.loads(done.stdout)
        if answer.get("is_error"):
            raise RuntimeError(f"claude answered an error: {str(answer.get('result'))[:500]}")
        self.cost_usd += float(answer.get("total_cost_usd") or 0.0)
        logger.info(f"claude {self.model}: {answer.get('duration_ms', 0) / 1000:.1f}s, "
                    f"${float(answer.get('total_cost_usd') or 0):.3f} at API prices")
        return str(answer.get("result", ""))
