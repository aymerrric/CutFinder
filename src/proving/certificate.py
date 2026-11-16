"""
The Lean files of a cut, and the check that decides whether a proof is a certificate.

Three pieces of text:
  statement  the cut in Lean: `def Cut (p : Params) (v : Vars p) : Prop := ...` (written by the
             LLM, then FROZEN: this is what a human reviews against the Python code);
  proof      helper lemmas and `theorem cut_valid : formulation.ValidCut Cut` (or `cut_safe :
             formulation.SafeCut Cut`), written by the LLM;
  check      written here, never by the LLM.

`verify` trusts nothing in the proof text:
  1. refused words (proving/lean.FORBIDDEN) in the statement or the proof: not proved;
  2. CutProof.lean = header + statement + proof is compiled into a module;
  3. Check.lean, a separate file that only imports that module, states the claim with full names
     and prints the axioms the theorem depends on. Nothing the proof text opened or declared
     locally (sections, variables, options) reaches that file.
The cut is PROVED when both files compile and the axioms are among Lean's three standard ones.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from proving import lean
from proving.lean import Compilation, LeanMessage
from proving.problems import LeanProblem
from utils import parameters

PROOF_MODULE: str = "CutProof"  # CutProof.lean in the attempt's folder
CHECK_FILE: str = "Check.lean"
# claim -> (the theorem the proof must contain, the definition of EvoCut/Common.lean it proves)
CLAIMS: Dict[str, Tuple[str, str]] = {"valid": ("cut_valid", "ValidCut"), "safe": ("cut_safe", "SafeCut")}
BASE_IMPORTS: List[str] = ["Mathlib.Tactic"]  # a full `import Mathlib` takes 55 s per compile, this 8-12 s

_LEAN_BLOCK_RE: "re.Pattern[str]" = re.compile(r"```(?:lean4?|Lean)?[ \t]*\n(.*?)```", re.DOTALL)
_IMPORT_RE: "re.Pattern[str]" = re.compile(r"^\s*import\s+(\S+)\s*$")


def lean_block(reply: str) -> Optional[str]:
    """The last ```lean block of an LLM answer (a reasoning model may quote drafts before it)."""
    blocks: List[str] = _LEAN_BLOCK_RE.findall(reply)
    return blocks[-1].strip("\n") if blocks else None


def split_imports(text: str) -> Tuple[List[str], str]:
    """(modules imported by the `import` lines of `text`, the text without them). Imports must be
    at the top of a Lean file: the LLM may write them in its block, they are moved to the header."""
    modules: List[str] = []
    rest: List[str] = []
    for line in text.splitlines():
        match: Optional[re.Match] = _IMPORT_RE.match(line)
        if match:
            modules.append(match.group(1))
        else:
            rest.append(line)
    return modules, "\n".join(rest).strip("\n")


def unwrap(problem: LeanProblem, text: str) -> str:
    """An LLM answer without the lines of the header it repeated: the import of the problem's own
    module, `open BigOperators Finset`, and the `namespace ... CutProof` / `end ... CutProof` pair
    (small models copy the wrapper they are shown; left in, their `Cut` ends up in another
    namespace or the file ends before the check). Nothing else is removed."""
    namespace: str = f"{problem.namespace}.{PROOF_MODULE}"
    dropped: List[str] = [f"import {problem.module}", "open BigOperators Finset", "open Finset BigOperators",
                          f"namespace {namespace}", f"end {namespace}"]
    return "\n".join(line for line in text.splitlines() if line.strip() not in dropped).strip("\n")


_FAMILY_RE: "re.Pattern[str]" = re.compile(r"^\s*(?:noncomputable\s+)?def\s+(Family_\w+)\b", re.MULTILINE)
_CUT_RE: "re.Pattern[str]" = re.compile(r"^\s*(?:noncomputable\s+)?def\s+Cut\b", re.MULTILINE)
CUT_HEAD: str = "def Cut (p : Params) (v : Vars p) : Prop :="


def families(statement: str) -> List[str]:
    """The constraint families of a statement written by families: the names of its
    `def Family_<name>` definitions, in order ([] for a statement written as one `Cut`)."""
    return _FAMILY_RE.findall(lean.strip_comments(statement))


def with_cut(statement: str, names: List[str]) -> str:
    """`statement` (family definitions, no `Cut`) followed by `Cut` = the conjunction of the
    families `names`. Written here, not by the LLM: the cut is exactly its families."""
    body: str = " ∧ ".join(f"{name} p v" for name in names)
    return (statement.rstrip() + "\n\n/-- The cut: every family holds. -/\n" if len(names) > 1
            else statement.rstrip() + "\n\n/-- The cut restricted to one family. -/\n") + f"{CUT_HEAD}\n  {body}"


def defines_cut(statement: str) -> bool:
    return bool(_CUT_RE.search(lean.strip_comments(statement)))


def without_cut(statement: str) -> str:
    """A family statement without the `Cut` that `with_cut` appended (the text before it)."""
    marker: str = "\n\n/-- The cut: every family holds. -/\n"
    return statement.split(marker, 1)[0] if marker in statement else statement


def combined_proof(names: List[str], proofs: List[str]) -> str:
    """One proof of the whole cut from one proof per family. Each family's proof was written
    for a file where `Cut` is that family alone: it is put in its own namespace with that local
    `Cut`, then `cut_valid` of the whole is the tuple of the parts (valid families combine:
    MILPFormulation.ValidCut.and)."""
    imports: List[str] = []
    parts: List[str] = []
    for k, (name, proof) in enumerate(zip(names, proofs), 1):
        modules, body = split_imports(proof)
        imports += [m for m in modules if m not in imports]
        parts.append(f"namespace Part{k}\n\n{CUT_HEAD}\n  {name} p v\n\n{body}\n\nend Part{k}")
    terms: str = ", ".join(f"Part{k}.cut_valid p v h" for k in range(1, len(names) + 1))
    final: str = ("theorem cut_valid : formulation.ValidCut Cut :=\n  fun p v h => "
                  + (f"⟨{terms}⟩" if len(names) > 1 else terms))
    return "\n".join(f"import {m}" for m in imports) + ("\n\n" if imports else "") + "\n\n".join(parts) + "\n\n" + final


def refused_imports(modules: List[str]) -> List[str]:
    """Only Mathlib (and Lean's own libraries) may be imported."""
    return [m for m in modules if m.split(".")[0] not in ("Mathlib", "Batteries", "Std", "Init", "Lean")]


def header(problem: LeanProblem, imports: List[str]) -> str:
    """The first lines of every file: imports, then the namespace the statement and proof live in
    (inside the problem's namespace, so `Params`, `Vars`, `Feasible`, `formulation` are in scope)."""
    modules: List[str] = BASE_IMPORTS + [problem.module] + [m for m in imports if m not in BASE_IMPORTS]
    return ("\n".join(f"import {m}" for m in modules) + "\n\nopen BigOperators Finset\n\n"
            f"namespace {problem.namespace}.{PROOF_MODULE}\n\n")


def statement_file(problem: LeanProblem, statement: str) -> str:
    """The statement alone, with a last line that checks `Cut` is a cut of the formulation."""
    imports, body = split_imports(statement)
    return header(problem, imports) + body + "\n\nexample : formulation.Cut := Cut\n"


def proof_file(problem: LeanProblem, statement: str, proof: str) -> Tuple[str, int]:
    """(CutProof.lean, the number of lines before the proof text: a message at line L of the file
    is at line L - offset of the proof)."""
    statement_imports, statement_body = split_imports(statement)
    proof_imports, proof_body = split_imports(proof)
    top: str = header(problem, statement_imports + proof_imports) + statement_body + "\n\n"
    return top + proof_body + "\n", top.count("\n")


def claim_of(proof: str) -> Optional[str]:
    """The claim a proof text makes: "valid" if it has `theorem cut_valid`, else "safe" if it has
    `theorem cut_safe`, else None."""
    code: str = lean.strip_comments(proof)
    for claim, (theorem, _) in CLAIMS.items():
        if re.search(r"\b(theorem|lemma)\s+" + theorem + r"\b", code):
            return claim
    return None


def check_file(problem: LeanProblem, claim: str) -> str:
    """Check.lean: the claim with full names, proved by the LLM's theorem, and its axioms."""
    theorem, definition = CLAIMS[claim]
    name: str = f"{problem.namespace}.{PROOF_MODULE}"
    return (f"import {PROOF_MODULE}\n\n"
            f"example : MILPFormulation.{definition} {problem.namespace}.formulation {name}.Cut :=\n"
            f"  {name}.{theorem}\n\n"
            f"#print axioms {name}.{theorem}\n")


@dataclass
class Verdict:
    """What `verify` found. `proved` is the only field that makes a certificate."""

    proved: bool
    reason: str  # one sentence: why it is proved or not
    claim: Optional[str] = None  # "valid" or "safe"
    axioms: List[str] = field(default_factory=list)
    compilation: Optional[Compilation] = None  # of CutProof.lean
    check: Optional[Compilation] = None  # of Check.lean
    offset: int = 0  # lines of CutProof.lean before the proof text


def verify(
    problem: LeanProblem, statement: str, proof: str, folder: str,
    timeout: float = parameters.LEAN_TIMEOUT, lean_dir: str = parameters.LEAN_DIR,
) -> Verdict:
    """Is `proof` a proof that the cut `statement` is valid (or safe) for `problem`? Writes
    CutProof.lean and Check.lean in `folder` (see the module docstring)."""
    for name, text in (("statement", statement), ("proof", proof)):
        words: List[str] = lean.forbidden(text)
        if words:
            return Verdict(False, f"the {name} uses refused words: {', '.join(words)}")
        modules: List[str] = refused_imports(split_imports(text)[0])
        if modules:
            return Verdict(False, f"the {name} imports {', '.join(modules)}: only Mathlib modules are allowed")
    claim: Optional[str] = claim_of(proof)
    if claim is None:
        return Verdict(False, "the proof has no `theorem cut_valid` and no `theorem cut_safe`")
    os.makedirs(folder, exist_ok=True)
    text, offset = proof_file(problem, statement, proof)
    path: str = os.path.join(folder, f"{PROOF_MODULE}.lean")
    olean: str = os.path.join(folder, f"{PROOF_MODULE}.olean")
    if os.path.exists(olean):
        os.remove(olean)  # never check an older proof's module
    with open(path, "w") as f:
        f.write(text)
    compilation: Compilation = lean.compile_file(path, olean=olean, timeout=timeout, lean_dir=lean_dir)
    verdict: Verdict = Verdict(False, "", claim=claim, compilation=compilation, offset=offset)
    if compilation.timed_out:
        verdict.reason = f"the proof did not compile within {timeout:.0f} s"
        return verdict
    if not compilation.ok or not os.path.exists(olean):
        verdict.reason = f"the proof does not compile ({len(compilation.errors)} errors)"
        return verdict
    check_path: str = os.path.join(folder, CHECK_FILE)
    check_text: str = check_file(problem, claim)
    with open(check_path, "w") as f:
        f.write(check_text)
    check: Compilation = lean.compile_file(check_path, search_path=folder, timeout=timeout, lean_dir=lean_dir)
    verdict.check = check
    if not check.ok:
        first: str = check.errors[0].text if check.errors else check.other_output
        verdict.reason = f"the theorem does not prove the claim as stated: {first[:300]}"
        return verdict
    axioms_line: int = check_text.count("\n")  # `#print axioms` is the last line
    found: Optional[List[str]] = None
    for message in check.messages:
        if message.line == axioms_line and message.severity == "information":
            found = lean.axioms_in(message.text)
    if found is None:
        verdict.reason = "the axioms of the theorem could not be read"
        return verdict
    verdict.axioms = found
    extra: List[str] = [a for a in found if a not in lean.STANDARD_AXIOMS]
    if extra:
        verdict.reason = f"the proof depends on {', '.join(extra)}" + (
            " (it uses `sorry`)" if "sorryAx" in extra else "")
        return verdict
    verdict.proved = True
    verdict.reason = f"{CLAIMS[claim][1]} proved, axioms: {', '.join(found) or 'none'}"
    return verdict


def describe(messages: List[LeanMessage], source: str, offset: int = 0, limit: int = parameters.PROOF_FEEDBACK_CHARS) -> str:
    """The compiler's messages as text for the LLM: errors first, each with its line number in the
    LLM's own text (file line - offset) and the source line; cut at `limit` characters."""
    lines: List[str] = source.splitlines()
    parts: List[str] = []
    for message in sorted(messages, key=lambda m: (m.severity != "error", m.line)):
        if message.severity == "information":
            continue
        where: int = message.line - offset
        quoted: str = lines[message.line - 1].strip() if 0 < message.line <= len(lines) else ""
        parts.append(f"{message.severity}, line {where}: `{quoted}`\n{message.text.strip()}")
    text: str = "\n\n".join(parts)
    return text if len(text) <= limit else text[:limit] + "\n[... cut]"
