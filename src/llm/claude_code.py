"""
An LLMClient that asks Claude through the Claude Code command line (`claude -p`), with no tool:
one question, one answer. It uses the account Claude Code is logged in with (a subscription), so
it needs no API key. Used by prove_cut.py and main.py (--llm_backend claude_code).
On a machine where Claude Code is not logged in (the lab), the token of `claude setup-token`,
put on a CLAUDE_CODE_OAUTH_TOKEN=... line of llm_keys.env (or in the environment), logs it in.
A usage limit of the subscription stops the run (QuotaExhausted): the night plan resumes it.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, Message, QuotaExhausted, UsageLimit
from llm.openai_compatible import MissingAPIKey, api_key
from utils import parameters

logger: logging.Logger = logging.getLogger(__name__)

TOKEN_VARIABLE: str = "CLAUDE_CODE_OAUTH_TOKEN"  # what `claude setup-token` prints
KEYS_FILE: str = parameters.LLM_KEYS_FILE
# how Claude Code words a reached subscription limit ("Claude AI usage limit reached|<time>",
# "You've hit your limit · resets 3pm", "5-hour limit reached", ...)
LIMIT_RE: "re.Pattern[str]" = re.compile(r"usage limit|hit your limit|limit reached|limit will reset|rate.?limit", re.I)
# not logged in: no call can succeed either (the GA would otherwise fill generations with copies)
LOGIN_RE: "re.Pattern[str]" = re.compile(r"not logged in|please run /login|invalid api key|oauth token|authentication", re.I)


def environment() -> Dict[str, str]:
    """The environment of the `claude` command: ours, plus the login token of llm_keys.env if any."""
    env: Dict[str, str] = dict(os.environ)
    try:
        env[TOKEN_VARIABLE] = api_key(TOKEN_VARIABLE, KEYS_FILE)
    except MissingAPIKey:
        pass  # Claude Code uses its own login (the Mac)
    return env


def limit_reached(text: str) -> bool:
    return bool(LIMIT_RE.search(text or ""))


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
        # the exact models that answered ("claude-sonnet-5-5"): the keys of the answer's modelUsage,
        # since the alias given ("sonnet") moves to each new model
        self.model_ids: List[str] = []

    def identity(self) -> Dict[str, Optional[str]]:
        """For status.json: the exact model ids seen so far (None before the first answer) and the
        effort passed to claude -p ("default" = none passed: Claude Code's own default for that model;
        the answer does not report it)."""
        return {"llm_model_id": ", ".join(self.model_ids) or None, "llm_effort": self.effort or "default"}

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
                timeout=self.timeout, cwd=folder, env=environment())
        result: str = done.stdout + done.stderr
        if limit_reached(result[-2000:]) and (done.returncode != 0 or '"is_error":true' in result.replace(" ", "")):
            raise UsageLimit(f"Claude subscription limit: {result[-300:].strip()}")
        if done.returncode != 0 and LOGIN_RE.search(result[-2000:]):
            raise QuotaExhausted(f"claude is not logged in (claude setup-token, then {TOKEN_VARIABLE}= in llm_keys.env): "
                                 f"{result[-300:].strip()}")
        if done.returncode != 0:
            raise RuntimeError(f"claude exited with code {done.returncode}: {(done.stderr or done.stdout)[-500:]}")
        answer: Dict[str, Any] = json.loads(done.stdout)
        if answer.get("is_error"):
            raise RuntimeError(f"claude answered an error: {str(answer.get('result'))[:500]}")
        self.cost_usd += float(answer.get("total_cost_usd") or 0.0)
        answered: List[str] = sorted(answer.get("modelUsage") or {})
        self.model_ids += [m for m in answered if m not in self.model_ids]
        logger.info(f"claude {self.model} ({', '.join(answered) or 'model not reported'}, "
                    f"effort {self.effort or 'default'}): {answer.get('duration_ms', 0) / 1000:.1f}s, "
                    f"${float(answer.get('total_cost_usd') or 0):.3f} at API prices")
        return str(answer.get("result", ""))
