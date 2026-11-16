"""
The proof phase done by a coding agent, as in FLARE: Claude Code, started without a terminal
(`claude -p`), works in a folder that holds the frozen statement, edits one file, and compiles as
often as it wants (./check.sh, and the lean-lsp MCP server, which shows the goal at any line).

Differences with FLARE's harness:
  - no Docker: the agent may only edit files of its folder (permission mode acceptEdits, a list
    of allowed tools) and its only shell command is ./check.sh;
  - the agent does not write the statement (it is frozen before, proving/prover.py) and its
    file is not trusted: when it stops, the text under the marker line is taken as the proof and
    goes through the same certificate.verify as every other proof.
The agent uses the account Claude Code is logged in with. A run can take 5-30 minutes.
"""
import json
import os
import shutil
import signal
import stat
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llm.claude_code import claude
from proving import certificate
from proving.certificate import Verdict
from proving.problems import LeanProblem
from proving.retry_proxy import RetryProxy
from proving.prover import CLAIMED_INVALID, NOT_PROVED, PROVED, PROMPTS_FILE, ProofResult, fill
from utils import parameters

MARKER: str = "-- ===== PROOF: write below this line, change nothing above it ====="
AGENT_FILE: str = f"{certificate.PROOF_MODULE}.lean"
OUTPUT_FILE: str = "agent_output.jsonl"  # the agent's events, one JSON object per line
WORK_DIR: str = "work"  # in lean/: the agent's folders (inside the Lake project, for the Lean server)
AGENT_MINUTES: float = 30  # FLARE's limit
ALLOWED_TOOLS: List[str] = ["Read", "Edit", "Write", "Glob", "Grep", "Bash(./check.sh)", "mcp__lean-lsp"]
INVALID_PREFIX: str = "-- INVALID:"  # the agent's answer when it finds the cut invalid (agent_task)
DENIED_TOOLS: List[str] = ["Read(**/*.env)", "Read(**/*.lic)", "Read(**/license_keys.txt)", "WebFetch", "WebSearch"]


def agent_file(problem: LeanProblem, statement: str) -> str:
    """The file the agent edits: header, frozen statement, marker, a theorem to complete."""
    imports, body = certificate.split_imports(statement)
    return (certificate.header(problem, imports) + body + f"\n\n{MARKER}\n\n"
            "theorem cut_valid : formulation.ValidCut Cut := by\n  sorry\n")


def proof_of(text: str) -> Optional[str]:
    """The proof the agent left: what is under the marker (None if the marker is gone)."""
    if MARKER not in text:
        return None
    proof: str = text.split(MARKER, 1)[1].strip()
    end: str = "end " + ".".join(["EvoCut"])  # a closing `end EvoCut.<Problem>.CutProof` is harmless
    return proof if proof and not proof.startswith(end) else None


VIBE_AGENT: str = "lean-requesty"  # the Vibe agent written by vibe_setup
# where Leanstral is served: API -> (base url, model name, environment variable of the key)
LEANSTRAL_APIS: Dict[str, Tuple[str, str, str]] = {
    "mistral": ("https://api.mistral.ai/v1", "labs-leanstral-1-5", "MISTRAL_API_KEY"),  # Labs models enabled
    "requesty": ("https://router.requesty.ai/v1", "mistral/leanstral-1-5", "REQUESTY_API_KEY"),  # free, rate limited
}
VIBE_HOME_DIR: str = "vibe_home"  # in lean/work/: Vibe's own settings, apart from the user's ~/.vibe


def vibe_setup(lean_dir: str, model: str = "mistral/leanstral-1-5", api_base: str = "https://router.requesty.ai/v1",
               key_variable: str = "REQUESTY_API_KEY") -> str:
    """Write Vibe's settings for the Lean agent (Mistral's Lean system prompt, the model through an
    OpenAI-style router, the lean-lsp MCP server; no shell, no web) under lean/work/vibe_home, and
    return that folder. Leanstral's page strongly recommends the lean-lsp MCP server. Its online
    search lean_loogle is turned off: 3 requests per 30 s, and on 15 Nov Leanstral spent 57 of its
    80 turns on refused calls to it; lean_local_search (ripgrep over Mathlib) and lean_leansearch
    (90 per 30 s) stay."""
    home: str = os.path.join(lean_dir, WORK_DIR, VIBE_HOME_DIR)
    os.makedirs(os.path.join(home, "agents"), exist_ok=True)
    with open(os.path.join(home, "config.toml"), "w") as f:
        f.write(f"""active_model = "leanstral"
default_agent = "{VIBE_AGENT}"
enable_telemetry = false
enable_update_checks = false
enable_connectors = false
enable_notifications = false

[[mcp_servers]]
name = "lean-lsp"
transport = "stdio"
command = "uvx"
args = ["lean-lsp-mcp", "--disable-tools", "lean_loogle"]
tool_timeout_sec = 600

[mcp_servers.env]
LEAN_PROJECT_PATH = "{lean_dir}"

[[providers]]
name = "router"
api_base = "{api_base}"
api_key_env_var = "{key_variable}"
api_style = "openai"
backend = "generic"

[[models]]
name = "{model}"
provider = "router"
alias = "leanstral"
temperature = 1.0
""")
    with open(os.path.join(home, "agents", f"{VIBE_AGENT}.toml"), "w") as f:
        f.write('display_name = "Lean (router)"\nsafety = "neutral"\nsystem_prompt_id = "lean"\n'
                'active_model = "leanstral"\ndisabled_tools = ["exit_plan_mode", "web_fetch", "web_search", '
                '"bash", "experimental_bash", "task"]\n')
    return home


def vibe_command(max_turns: int) -> List[str]:
    """Vibe without a terminal: the task is given on the command line (programmatic mode), tools
    approved (the agent has no shell and no web, set in its settings)."""
    return [shutil.which("vibe") or os.path.expanduser("~/.local/bin/vibe"), "--agent", VIBE_AGENT,
            "--auto-approve", "--trust", "--max-turns", str(max_turns), "--output", "streaming", "-p"]


def check_script(folder: str, lean_dir: str) -> str:
    """check.sh: compile the agent's file from the project root (lake needs it) and print the messages."""
    return ("#!/bin/sh\n# Compiles CutProof.lean and prints the compiler's messages (nothing = no error, no warning).\n"
            f"cd '{lean_dir}' && lake env lean '{os.path.join(folder, AGENT_FILE)}'\n"
            "echo \"exit code: $?\"\n")


def command(model: str, effort: Optional[str], mcp_config: str, budget_usd: Optional[float]) -> List[str]:
    """The agent's command line; the task is given on its standard input. check.sh is allowed
    under both spellings: an agent called it by its full path and was refused (14 Nov)."""
    script: str = os.path.join(os.path.dirname(mcp_config), "check.sh")
    line: List[str] = [claude() or "claude", "-p", "--model", model, "--output-format", "stream-json", "--verbose",
                       "--permission-mode", "acceptEdits", "--allowedTools", *ALLOWED_TOOLS, f"Bash({script})",
                       "--disallowedTools", *DENIED_TOOLS, "--mcp-config", mcp_config, "--strict-mcp-config",
                       "--no-session-persistence"]
    if effort:
        line += ["--effort", effort]
    if budget_usd is not None:
        line += ["--max-budget-usd", str(budget_usd)]
    return line


def last_result(output_path: str) -> Dict[str, Any]:
    """The agent's final event ({"type": "result", ...}: cost, turns, duration), or {}."""
    found: Dict[str, Any] = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    event: Dict[str, Any] = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    found = event
    return found


class AgentProver:
    """Proves a frozen statement with a Claude Code agent. `run` returns a ProofResult like Prover.prove."""

    def __init__(
        self, problem: LeanProblem, model_code: str, label: str, model: str = "opus",
        effort: Optional[str] = "medium", minutes: float = AGENT_MINUTES, budget_usd: Optional[float] = None,
        prompts_dir: str = parameters.PROMPTS_DIR, lean_dir: str = parameters.LEAN_DIR,
        timeout: float = parameters.LEAN_TIMEOUT,
        cli: str = "claude",  # "claude" (Claude Code) or "vibe" (Mistral Vibe, e.g. Leanstral)
        api: str = "mistral",  # vibe: a key of LEANSTRAL_APIS
        env: Optional[Dict[str, str]] = None,  # extra environment (vibe: the router's key)
        max_turns: int = 200,  # 80 were used up on 15 Nov, most by a rate-limited search tool
    ) -> None:
        self.cli: str = cli
        self.api: str = api
        self.env: Dict[str, str] = dict(env or {})
        self.max_turns: int = max_turns
        self.problem: LeanProblem = problem
        self.model_code: str = model_code
        self.model: str = model
        self.effort: Optional[str] = effort
        self.minutes: float = minutes
        self.budget_usd: Optional[float] = budget_usd
        self.lean_dir: str = lean_dir
        self.timeout: float = timeout
        self.folder: str = os.path.join(lean_dir, WORK_DIR, f"{problem.name}_{label}")
        with open(os.path.join(prompts_dir, PROMPTS_FILE)) as f:
            self.task_template: str = yaml.safe_load(f)["agent_task"]

    def setup(self, idea: str, code: str, statement: str) -> str:
        """Write the agent's folder; returns the task text."""
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, AGENT_FILE), "w") as f:
            f.write(agent_file(self.problem, statement))
        script: str = os.path.join(self.folder, "check.sh")
        with open(script, "w") as f:
            f.write(check_script(self.folder, self.lean_dir))
        os.chmod(script, os.stat(script).st_mode | stat.S_IXUSR)
        with open(os.path.join(self.folder, ".mcp.json"), "w") as f:
            json.dump({"mcpServers": {"lean-lsp": {
                "type": "stdio", "command": "uvx", "args": ["lean-lsp-mcp"],
                "env": {"LEAN_PROJECT_PATH": self.lean_dir}}}}, f, indent=1)
        task: str = fill(
            self.task_template, idea=idea, code=code, marker=MARKER, model_code=self.model_code.strip(),
            lean_model_path=os.path.join(self.lean_dir, *self.problem.module.split(".")) + ".lean",
            common_path=os.path.join(self.lean_dir, "EvoCut", "Common.lean"),
            example_path=os.path.join(self.lean_dir, "EvoCut", "Cuts", "TspSeed.lean"))
        with open(os.path.join(self.folder, "task.md"), "w") as f:
            f.write(task)
        return task

    def run(self, idea: str, code: str, statement: str, out_folder: str) -> ProofResult:
        """Start the agent, wait for it (at most `minutes`), then verify what it left.
        out_folder: where the verified Lean files and the agent's events are kept."""
        task: str = self.setup(idea, code, statement)
        os.makedirs(out_folder, exist_ok=True)
        output_path: str = os.path.join(out_folder, OUTPUT_FILE)
        start: float = time.time()
        with open(output_path, "w") as output:
            relay: Optional[RetryProxy] = None
            if self.cli == "vibe":  # the task as an argument, no input (Vibe waits on an open stdin)
                base, model, variable = LEANSTRAL_APIS[self.api]
                relay = RetryProxy(base).__enter__()  # waits out the API's rate limits
                env: Dict[str, str] = {**os.environ, **self.env, "VIBE_HOME": vibe_setup(
                    self.lean_dir, model=model, api_base=relay.base_url, key_variable=variable)}
                process: subprocess.Popen = subprocess.Popen(
                    vibe_command(self.max_turns) + [task], cwd=self.folder, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
                stdin_text: Optional[str] = None
            else:
                process = subprocess.Popen(
                    command(self.model, self.effort, os.path.join(self.folder, ".mcp.json"), self.budget_usd),
                    cwd=self.folder, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, text=True,
                    start_new_session=True)
                stdin_text = task
            try:
                process.communicate(stdin_text, timeout=self.minutes * 60)
                stopped: str = f"the agent ended with code {process.returncode}"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                stopped = f"the agent was stopped after {self.minutes:g} minutes"
            finally:
                if relay is not None:
                    relay.__exit__()
                    if relay.retries:
                        stopped += f"; {relay.retries} requests tried again after a rate limit"
        agent_seconds: float = time.time() - start
        result: ProofResult = ProofResult(self.problem.name, NOT_PROVED, stopped, statement=statement,
                                          llm_calls=1, llm_seconds=agent_seconds, proof_rounds=1)
        with open(os.path.join(self.folder, AGENT_FILE)) as f:
            proof: Optional[str] = proof_of(f.read())
        if proof is None:
            result.reason = f"{stopped}; no proof under the marker of {AGENT_FILE}"
            return result
        result.proof = proof
        if proof.startswith(INVALID_PREFIX):  # the agent's claim, with its counterexample: not checked
            result.status, result.reason = CLAIMED_INVALID, proof[len(INVALID_PREFIX):].strip()[:2000]
            return result
        verdict: Verdict = certificate.verify(self.problem, statement, proof, os.path.join(out_folder, "agent_proof"),
                                              self.timeout, self.lean_dir)
        result.lean_seconds = sum(c.seconds for c in (verdict.compilation, verdict.check) if c is not None)
        result.reason = f"{verdict.reason} ({stopped})"
        if verdict.proved:
            result.status, result.claim, result.axioms = PROVED, verdict.claim, verdict.axioms
        return result
