"""
The proof loop: an LLM states a cut in Lean, a second call judges the statement, then the LLM
proves it, each answer being compiled and its errors sent back (prompts/proving.yaml).

    statement  up to `statement_rounds` answers until the Lean definition of the cut compiles and
               the judge (a separate conversation) finds it faithful to the Python code;
    proof      up to `proof_rounds` answers until certificate.verify accepts the proof.

The statement is frozen before the proof starts: a proof that does not go through cannot lead the
LLM to weaken what is proved. Every answer is recorded (`record`), so a run can be read afterwards.
Works with any LLMClient (Ollama, an API): it only needs `send`.
"""
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import yaml

from llm.base import LLMClient, Message
from proving import certificate, lean
from proving.certificate import Verdict
from proving.lean import Compilation
from proving.problems import LeanProblem
from utils import parameters

PROMPTS_FILE: str = "proving.yaml"
REFERENCE: str = os.path.join("EvoCut", "Cuts", "TspSeed.lean")  # the proof shown as an example
_PLACEHOLDER_RE: "re.Pattern[str]" = re.compile(r"<<([a-z_]+)>>")
_VERDICT_RE: "re.Pattern[str]" = re.compile(r"VERDICT:\**\s*(UNFAITHFUL|FAITHFUL)", re.IGNORECASE)
_INVALID_RE: "re.Pattern[str]" = re.compile(r"^\W*INVALID:\s*(.*)", re.DOTALL)

# status of a ProofResult
PROVED: str = "proved"  # a certificate: the only status that says something about the cut
NOT_PROVED: str = "not_proved"  # no proof found within the rounds: says nothing about the cut
NO_STATEMENT: str = "no_statement"  # no Lean statement that compiles and that the judge accepts
CLAIMED_INVALID: str = "claimed_invalid"  # the LLM answered that the cut is invalid (not checked)

Attempt = Dict[str, Any]  # one LLM answer and what became of it (JSON)


def fill(template: str, **values: str) -> str:
    """`template` with every <<name>> replaced by values[name]; the values are not re-read."""
    missing: List[str] = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(values))
    if missing:
        raise KeyError(f"no value for the placeholders {missing}")
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def reference_proof(lean_dir: str = parameters.LEAN_DIR) -> str:
    """The proof part of the hand-written certificate (between `-- § Proof` and the namespace's end)."""
    with open(os.path.join(lean_dir, REFERENCE)) as f:
        text: str = f.read()
    return text.split("-- § Proof\n", 1)[1].split("\nend EvoCut.", 1)[0].strip()


@dataclass
class ProofResult:
    """What the loop found for one cut."""

    problem: str
    status: str  # PROVED, NOT_PROVED, NO_STATEMENT or CLAIMED_INVALID
    reason: str
    claim: Optional[str] = None  # "valid" or "safe" (PROVED only)
    axioms: List[str] = field(default_factory=list)
    statement: Optional[str] = None  # the cut in Lean, as frozen
    proof: Optional[str] = None  # the accepted proof (PROVED), else the last one tried
    judge: Optional[str] = None  # the judge's last answer on the statement (None = not asked)
    statement_rounds: int = 0
    proof_rounds: int = 0
    families: List[Dict[str, Any]] = field(default_factory=list)  # by families: name, status, rounds of each
    llm_calls: int = 0
    llm_seconds: float = 0.0
    lean_seconds: float = 0.0

    @property
    def proved(self) -> bool:
        return self.status == PROVED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Prover:
    """States and proves cuts of one problem with one LLM."""

    def __init__(
        self,
        llm: LLMClient,
        problem: LeanProblem,
        model_code: str,  # the source of the problem's create_model
        folder: str,  # where the Lean files of the attempts are written
        statement_rounds: int = parameters.PROOF_STATEMENT_ROUNDS,
        proof_rounds: int = parameters.PROOF_ROUNDS,
        judge: bool = True,
        by_families: bool = False,  # the statement as one definition per family, each proved alone
        judge_llm: Optional[LLMClient] = None,  # None = the same LLM, in a separate conversation
        history: int = parameters.PROOF_HISTORY,
        prompts_dir: str = parameters.PROMPTS_DIR,
        lean_dir: str = parameters.LEAN_DIR,
        timeout: float = parameters.LEAN_TIMEOUT,
        record: Optional[Callable[[Attempt], None]] = None,
    ) -> None:
        self.llm: LLMClient = llm
        self.judge_llm: Optional[LLMClient] = (judge_llm or llm) if judge else None
        self.problem: LeanProblem = problem
        self.folder: str = folder
        self.statement_rounds: int = statement_rounds
        self.proof_rounds: int = proof_rounds
        self.history: int = history
        self.by_families: bool = by_families
        self.lean_dir: str = lean_dir
        self.timeout: float = timeout
        self.record: Callable[[Attempt], None] = record or (lambda attempt: None)
        with open(os.path.join(prompts_dir, PROMPTS_FILE)) as f:
            self.templates: Dict[str, str] = yaml.safe_load(f)
        with open(os.path.join(lean_dir, "EvoCut", "Common.lean")) as f:
            common: str = f.read()
        self.values: Dict[str, str] = {
            "common": common.strip(), "lean_model": problem.source(lean_dir).strip(),
            "model_code": model_code.strip(), "example": reference_proof(lean_dir),
            "header": certificate.header(problem, []).strip(),
        }
        self.result: ProofResult = ProofResult(problem.name, NOT_PROVED, "")

    # ------------------------------------------------------------------ conversations

    def _ask(self, llm: LLMClient, messages: List[Message]) -> str:
        start: float = time.time()
        reply: str = llm.send(messages)
        self.result.llm_calls += 1
        self.result.llm_seconds += time.time() - start
        return reply

    def _conversation(self, system: str, user: str, exchanges: List[List[Message]]) -> List[Message]:
        """system + task + the last `history` (answer, feedback) exchanges."""
        kept: List[List[Message]] = exchanges[-self.history:] if self.history > 0 else []
        return [{"role": "system", "content": system}, {"role": "user", "content": user},
                *[message for exchange in kept for message in exchange]]

    def _compiled(self, compilation: Compilation) -> None:
        self.result.lean_seconds += compilation.seconds

    # ------------------------------------------------------------------ statement

    def check_statement(self, statement: str) -> Optional[str]:
        """Why a statement is refused (text for the LLM), or None when it compiles as a cut."""
        words: List[str] = lean.forbidden(statement)
        if words:
            return f"refused words: {', '.join(words)}"
        if re.search(r"\b(theorem|lemma)\b", lean.strip_comments(statement)):
            return "the statement must only contain definitions: no theorem, no lemma"
        modules: List[str] = certificate.refused_imports(certificate.split_imports(statement)[0])
        if modules:
            return f"only Mathlib modules may be imported, not {', '.join(modules)}"
        os.makedirs(self.folder, exist_ok=True)
        path: str = os.path.join(self.folder, "Statement.lean")
        text: str = certificate.statement_file(self.problem, statement)
        with open(path, "w") as f:
            f.write(text)
        compilation: Compilation = lean.compile_file(path, timeout=self.timeout, lean_dir=self.lean_dir)
        self._compiled(compilation)
        if compilation.timed_out:
            return f"the statement did not compile within {self.timeout:.0f} s"
        if compilation.ok:
            return None
        offset: int = certificate.header(self.problem, certificate.split_imports(statement)[0]).count("\n")
        return ("the statement does not compile (the last line of the file checks that `Cut` has "
                "the type `(p : Params) → Vars p → Prop`):\n"
                + (certificate.describe(compilation.messages, text, offset) or compilation.other_output))

    def judge(self, idea: str, code: str, statement: str) -> Optional[str]:
        """Why the judge finds the statement unfaithful to the code, or None when it does not
        (or when there is no judge). An answer without a verdict counts as unfaithful."""
        if self.judge_llm is None:
            return None
        reply: str = self._ask(self.judge_llm, [
            {"role": "system", "content": fill(self.templates["judge_system"], **self.values)},
            {"role": "user", "content": fill(self.templates["judge_user"], idea=idea, code=code,
                                             statement=statement, **self.values)}])
        self.result.judge = reply.strip()
        match: Optional[re.Match] = _VERDICT_RE.search(reply)
        if match and match.group(1).upper() == "FAITHFUL":
            return None
        return "a reviewer compared the statement with the Python code and found it unfaithful:\n" + reply.strip()

    def formalize(self, idea: str, code: str) -> Optional[str]:
        """The cut as a Lean statement that compiles and passes the judge, or None."""
        system: str = fill(self.templates["statement_system"], **self.values)
        user: str = fill(self.templates["statement_user"], idea=idea, code=code, **self.values)
        if self.by_families:
            user += "\n" + self.templates["statement_families"]
        exchanges: List[List[Message]] = []
        for round_number in range(1, self.statement_rounds + 1):
            self.result.statement_rounds = round_number
            reply: str = self._ask(self.llm, self._conversation(system, user, exchanges))
            statement: Optional[str] = certificate.lean_block(reply)
            if statement is not None:
                statement = certificate.unwrap(self.problem, statement)
            problem: Optional[str] = "no ```lean block in the answer" if statement is None else None
            if statement is not None and self.by_families:
                names: List[str] = certificate.families(statement)
                if not names:
                    problem = "no `def Family_<name>` definition: write one per constraint family"
                elif certificate.defines_cut(statement):
                    problem = "do not define `Cut`: it is added as the conjunction of your families"
                else:
                    statement = certificate.with_cut(statement, names)
            if statement is not None and problem is None:
                problem = self.check_statement(statement)
            if problem is None:
                problem = self.judge(idea, code, statement or "")
            self.record({"phase": "statement", "round": round_number, "reply": reply, "lean": statement,
                         "outcome": "accepted" if problem is None else "refused", "feedback": problem,
                         "judge": self.result.judge})
            if problem is None:
                return statement
            self.result.reason = f"statement round {round_number}: {problem.splitlines()[0]}"
            exchanges.append([{"role": "assistant", "content": reply},
                              {"role": "user", "content": fill(self.templates["statement_retry"], feedback=problem)}])
        return None

    # ------------------------------------------------------------------ proof

    def check_proof(self, statement: str, proof: str, round_number: int) -> Verdict:
        """Compile a proof text. With `sorry` left it is compiled (to show its other errors) but
        never accepted."""
        folder: str = os.path.join(self.folder, f"proof_{round_number:02d}")
        if "sorry" in lean.forbidden(proof) and lean.forbidden(proof) == ["sorry"]:
            text, offset = certificate.proof_file(self.problem, statement, proof)
            os.makedirs(folder, exist_ok=True)
            path: str = os.path.join(folder, f"{certificate.PROOF_MODULE}.lean")
            with open(path, "w") as f:
                f.write(text)
            compilation: Compilation = lean.compile_file(path, timeout=self.timeout, lean_dir=self.lean_dir)
            self._compiled(compilation)
            reason: str = ("the proof still contains `sorry`" if compilation.ok
                           else f"the proof does not compile ({len(compilation.errors)} errors), and contains `sorry`")
            return Verdict(False, reason, compilation=compilation, offset=offset)
        verdict: Verdict = certificate.verify(self.problem, statement, proof, folder, self.timeout, self.lean_dir)
        for compilation in (verdict.compilation, verdict.check):
            if compilation is not None:
                self._compiled(compilation)
        return verdict

    def feedback(self, verdict: Verdict, statement: str, proof: str) -> str:
        """What the LLM is told about a refused proof."""
        text: str = verdict.reason
        if verdict.compilation is not None and (verdict.compilation.errors or verdict.compilation.uses_sorry):
            source, _ = certificate.proof_file(self.problem, statement, proof)
            text += "\n\n" + certificate.describe(verdict.compilation.messages, source, verdict.offset)
        elif verdict.compilation is not None and verdict.compilation.other_output and not verdict.compilation.ok:
            text += "\n\n" + verdict.compilation.other_output[:parameters.PROOF_FEEDBACK_CHARS]
        return text

    def prove(self, idea: str, code: str, statement: str) -> ProofResult:
        """Look for a proof of the frozen `statement`; fills and returns self.result."""
        result: ProofResult = self.result
        result.statement = statement
        system: str = fill(self.templates["proof_system"], **self.values)
        user: str = fill(self.templates["proof_user"], idea=idea, code=code, statement=statement, **self.values)
        exchanges: List[List[Message]] = []
        for round_number in range(1, self.proof_rounds + 1):
            result.proof_rounds = round_number
            reply: str = self._ask(self.llm, self._conversation(system, user, exchanges))
            invalid: Optional[re.Match] = _INVALID_RE.match(reply.strip())
            proof: Optional[str] = certificate.lean_block(reply)
            if proof is not None:
                proof = certificate.unwrap(self.problem, proof)
            if invalid and proof is None:
                result.status, result.reason = CLAIMED_INVALID, invalid.group(1).strip()[:2000]
                self.record({"phase": "proof", "round": round_number, "reply": reply, "lean": None,
                             "outcome": CLAIMED_INVALID, "feedback": None})
                return result
            if proof is None:
                verdict: Verdict = Verdict(False, "no ```lean block in the answer")
            else:
                result.proof = proof
                verdict = self.check_proof(statement, proof, round_number)
            text: str = verdict.reason if proof is None else self.feedback(verdict, statement, proof)
            self.record({"phase": "proof", "round": round_number, "reply": reply, "lean": proof,
                         "outcome": PROVED if verdict.proved else "refused", "feedback": text,
                         "axioms": verdict.axioms, "claim": verdict.claim})
            if verdict.proved:
                result.status, result.reason = PROVED, verdict.reason
                result.claim, result.axioms = verdict.claim, verdict.axioms
                return result
            result.reason = f"proof round {round_number}: {verdict.reason}"
            exchanges.append([{"role": "assistant", "content": reply},
                              {"role": "user", "content": fill(self.templates["proof_retry"], feedback=text)}])
        result.status = NOT_PROVED
        return result

    def prove_by_families(self, idea: str, code: str, statement: str) -> ProofResult:
        """Prove each family of `statement` alone (the same loop, on a statement whose `Cut` is
        that family), then check the combined proof of the whole cut. A family that is not proved
        does not stop the others: the result says which families have a proof."""
        names: List[str] = certificate.families(statement)
        base: str = certificate.without_cut(statement)
        result: ProofResult = self.result
        result.statement = statement
        folder: str = self.folder
        proofs: List[str] = []
        total_rounds: int = 0
        for k, name in enumerate(names, 1):
            self.folder = os.path.join(folder, f"family_{k:02d}_{name}")
            record: Callable[[Attempt], None] = self.record
            self.record = lambda attempt, name=name, record=record: record({**attempt, "phase": f"proof_{name}"})
            part: ProofResult = self.prove(f"{idea}\n(Only the family `{name}` of the cut is to be proved here.)",
                                           code, certificate.with_cut(base, [name]))
            self.record = record
            result.families.append({"name": name, "status": part.status, "proof_rounds": part.proof_rounds,
                                    "claim": part.claim, "reason": part.reason})
            total_rounds += part.proof_rounds
            if part.status == PROVED and part.claim == "valid" and part.proof:
                proofs.append(part.proof)
        self.folder = folder
        result.statement, result.proof_rounds, result.claim, result.axioms = statement, total_rounds, None, []
        proved: List[str] = [f["name"] for f in result.families if f["status"] == PROVED and f["claim"] == "valid"]
        if len(proved) < len(names):
            missing: List[str] = [n for n in names if n not in proved]
            result.status, result.proof = NOT_PROVED, None
            result.reason = f"{len(proved)} of {len(names)} families proved valid; not proved: {', '.join(missing)}"
            return result
        result.proof = certificate.combined_proof(names, proofs)
        verdict: Verdict = certificate.verify(self.problem, statement, result.proof,
                                              os.path.join(folder, "combined"), self.timeout, self.lean_dir)
        for compilation in (verdict.compilation, verdict.check):
            if compilation is not None:
                self._compiled(compilation)
        if verdict.proved:
            result.status, result.claim, result.axioms = PROVED, verdict.claim, verdict.axioms
            result.reason = f"{verdict.reason} ({len(names)} families, each proved alone)"
        else:
            result.status = NOT_PROVED
            result.reason = f"every family is proved, but the combined proof is refused: {self.feedback(verdict, statement, result.proof)[:600]}"
        return result

    def run(self, idea: str, code: str, statement: Optional[str] = None) -> ProofResult:
        """State the cut (unless a reviewed `statement` is given), then prove it."""
        if statement is None:
            statement = self.formalize(idea, code)
            if statement is None:
                self.result.status = NO_STATEMENT
                return self.result
        else:
            problem: Optional[str] = self.check_statement(statement)
            if problem is not None:
                self.result.status, self.result.reason = NO_STATEMENT, problem
                return self.result
        if self.by_families and len(certificate.families(statement)) > 1:
            return self.prove_by_families(idea, code, statement)
        return self.prove(idea, code, statement)
