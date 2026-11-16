"""
Running Lean: compile one file of the Lake project lean/ and read the compiler's messages.

A proof is trusted only through `verify` (proving/certificate.py), which compiles the proof as a
module and then checks it from a second, clean file. This module only runs the compiler.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from utils import parameters

# the three axioms all of Mathlib is built on; a proof that depends on anything else is refused
# (`sorry` shows up as sorryAx, `native_decide` as Lean.ofReduceBool, an `axiom` under its name)
STANDARD_AXIOMS: Set[str] = {"propext", "Classical.choice", "Quot.sound"}

# Words refused in the Lean text written by the LLM. `sorry`, `admit`, `axiom` and `native_decide`
# would be caught by the axiom check anyway (a faster answer here). The others change the syntax
# or run code, which could make the checked statement mean something else, or stop the file early.
FORBIDDEN: List[str] = [
    "sorry", "admit", "axiom", "native_decide", "unsafe", "macro", "macro_rules", "syntax",
    "notation", "infix", "infixl", "infixr", "prefix", "postfix", "elab", "elab_rules",
    "declare_syntax_cat", "initialize", "run_cmd", "run_elab", "run_meta", "implemented_by",
    "extern", "#eval", "#exit",
]
# `set_option` is refused except for these (debug.skipKernelTC would skip the kernel's check)
ALLOWED_OPTIONS: List[str] = ["maxHeartbeats", "maxRecDepth", "linter."]

_AXIOMS_RE: "re.Pattern[str]" = re.compile(r"depends on axioms: \[([^\]]*)\]")
_OPTION_RE: "re.Pattern[str]" = re.compile(r"\bset_option\s+([A-Za-z0-9_.]+)")


@dataclass
class LeanMessage:
    """One message of the compiler. `line` counts from 1 in the compiled file."""

    severity: str  # "error", "warning" or "information"
    line: int
    column: int
    text: str


@dataclass
class Compilation:
    """The result of compiling one file."""

    exit_code: Optional[int]  # None = killed at the timeout
    messages: List[LeanMessage] = field(default_factory=list)
    other_output: str = ""  # what the compiler printed outside its JSON messages
    seconds: float = 0.0

    @property
    def timed_out(self) -> bool:
        return self.exit_code is None

    @property
    def errors(self) -> List[LeanMessage]:
        return [m for m in self.messages if m.severity == "error"]

    @property
    def ok(self) -> bool:
        """Compiled: exit code 0 and no error message (warnings are allowed)."""
        return self.exit_code == 0 and not self.errors

    @property
    def uses_sorry(self) -> bool:
        return any("sorry" in m.text for m in self.messages if m.severity == "warning")


def lake() -> Optional[str]:
    """The path of the `lake` command, or None when Lean is not installed."""
    for candidate in ("lake", os.path.expanduser("~/.elan/bin/lake"), "/opt/homebrew/bin/lake"):
        found: Optional[str] = shutil.which(candidate)
        if found:
            return found
    return None


def unavailable(lean_dir: str = parameters.LEAN_DIR) -> Optional[str]:
    """Why proofs cannot be checked on this machine, or None when they can."""
    if lake() is None:
        return "Lean is not installed (`brew install elan-init`, then `lake exe cache get` in lean/)"
    if not os.path.isdir(os.path.join(lean_dir, ".lake", "packages", "mathlib")):
        return f"Mathlib is not downloaded: run `lake exe cache get` in {lean_dir}"
    if not os.path.isdir(os.path.join(lean_dir, ".lake", "build", "lib", "lean", "EvoCut")):
        return f"the base formulations are not built: run `lake build` in {lean_dir}"
    return None


def parse_messages(output: str) -> Compilation:
    """The messages of `lean --json` (one JSON object per line); other lines are kept as text."""
    result: Compilation = Compilation(exit_code=0)
    other: List[str] = []
    for line in output.splitlines():
        try:
            data: Dict = json.loads(line)
            position: Dict = data.get("pos") or {}
            result.messages.append(LeanMessage(data["severity"], int(position.get("line", 0)),
                                               int(position.get("column", 0)), str(data.get("data", ""))))
        except (ValueError, KeyError, TypeError):
            if line.strip():
                other.append(line)
    result.other_output = "\n".join(other)
    return result


def compile_file(
    path: str,
    olean: Optional[str] = None,
    search_path: Optional[str] = None,
    timeout: float = parameters.LEAN_TIMEOUT,
    lean_dir: str = parameters.LEAN_DIR,
) -> Compilation:
    """
    Compile `path` with the project's Lean and Mathlib (`lake env lean --json`).
    olean: also write the compiled module there (the file's folder becomes its module root, so
           `CutProof.lean` is the module `CutProof`); nothing is written when there is an error.
    search_path: a folder added to the module search path (to import a module written by `olean`).
    The compiler and its children are killed at `timeout`.
    """
    command: List[str] = [lake() or "lake", "env", "lean", "--json"]
    if olean:
        command += [f"--root={os.path.dirname(os.path.abspath(path))}", "-o", olean]
    command.append(os.path.abspath(path))
    env: Dict[str, str] = dict(os.environ)
    if search_path:
        env["LEAN_PATH"] = search_path  # `lake env` puts the project's own paths before it
    start: float = time.time()
    process: subprocess.Popen = subprocess.Popen(
        command, cwd=lean_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,  # its own process group: lake starts lean as a child
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        result: Compilation = parse_messages(output)
        result.exit_code = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        result = parse_messages(output or "")
        result.exit_code = None
    result.seconds = time.time() - start
    return result


def axioms_in(text: str) -> Optional[List[str]]:
    """The axioms listed by a `#print axioms` message; [] for "does not depend on any axioms";
    None when the text is not such a message."""
    match: Optional[re.Match] = _AXIOMS_RE.search(text)
    if match:
        return [a.strip() for a in match.group(1).split(",") if a.strip()]
    return [] if "does not depend on any axioms" in text else None


def strip_comments(text: str) -> str:
    """Lean text without its comments (`-- ...` and `/- ... -/`, which nest)."""
    out: List[str] = []
    depth: int = 0
    i: int = 0
    while i < len(text):
        if text.startswith("/-", i):
            depth += 1
            i += 2
        elif depth and text.startswith("-/", i):
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif text.startswith("--", i):
            end: int = text.find("\n", i)
            i = len(text) if end < 0 else end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def forbidden(text: str, allow_sorry: bool = False) -> List[str]:
    """The refused words of FORBIDDEN and the refused options used in `text` (comments ignored)."""
    code: str = strip_comments(text)
    found: List[str] = []
    for word in FORBIDDEN:
        if allow_sorry and word == "sorry":
            continue
        if re.search(r"(?<![A-Za-z0-9_.'])" + re.escape(word) + r"(?![A-Za-z0-9_'])", code):
            found.append(word)
    for option in _OPTION_RE.findall(code):
        if not any(option == ok or (ok.endswith(".") and option.startswith(ok)) for ok in ALLOWED_OPTIONS):
            found.append(f"set_option {option}")
    return found
