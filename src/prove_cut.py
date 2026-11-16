"""
Prove that a cut is valid, in Lean 4: the second pipeline, after the genetic algorithm. An LLM
states the cut in Lean (lean/EvoCut/Problems/<Name>.lean is the model, written by hand), a second
call judges that statement against the Python code, then the LLM writes a proof; every answer is
compiled and its errors are sent back (proving/prover.py). A proof that passes the final check
(proving/certificate.py: clean file, no `sorry`, only Lean's three standard axioms) is a
certificate for EVERY instance that satisfies the model's data assumptions, not only the tested
ones (the method of FLARE, Robbins et al. 2026; papers/flare_robbins2026.txt).

What a result means:
    proved           the Lean statement is a theorem. It is about the cut as far as the statement
                     says what the code does: read <folder>/statement.lean against the code.
    not_proved       no proof found in the rounds given: says NOTHING about the cut.
    no_statement     no Lean statement that compiles and that the judge finds faithful.
    claimed_invalid  the LLM answered that the cut is invalid, with its reason (not checked).

Needs Lean on this machine (lean/README or README Step 41): runs on the Mac, not on the lab.
Run from src/:
    python prove_cut.py --problem_name tsp --cut seed --llm_backend muse --model_name muse-spark-1.3-contributor
    python prove_cut.py --problem_name cwlp --population_file ../runs/<run>/population.json ...   # its best cut
    python prove_cut.py --problem_name jssp --cut_file ../cuts/jssp/paper_sage.py --idea "..." ...
    python prove_cut.py ... --statement_file ../runs/proofs/tsp/seed/statement.lean   # a reviewed statement
    python prove_cut.py ... --llm_backend claude_code --model_name sonnet      # Claude, through the `claude` command
    python prove_cut.py ... --harness agent --agent_model opus                 # the proof by a Claude Code agent
Writes ../runs/proofs/<problem>/<label>/: statement.lean, proof.lean, result.json, attempts/*.json
and the Lean files of every round.
"""
import argparse
import inspect
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from evolution.population import Population
from llm.base import LLMClient
from llm.claude_code import ClaudeCodeClient
from llm.factory import BACKENDS, make_llm
from llm.openai_compatible import api_key
from problems import PROBLEMS, Problem
from proving import lean
from proving.problems import LEAN_PROBLEMS, LeanProblem, broken_assumptions
from proving.agent import AGENT_MINUTES, LEANSTRAL_APIS, AgentProver
from proving.prover import NO_STATEMENT, Attempt, ProofResult, Prover
from utils import parameters
from utils.run_files import run_name_of

logger: logging.Logger = logging.getLogger(__name__)
RESULT_FILE: str = "result.json"
CLAUDE_CODE: str = "claude_code"  # --llm_backend: Claude through the `claude` command (llm/claude_code.py)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Prove a cut valid in Lean 4 with an LLM")
    parser.add_argument("--problem_name", type=str, choices=sorted(LEAN_PROBLEMS), default=parameters.PROBLEM_NAME)
    parser.add_argument("--cut", type=str, choices=["seed"], default=None, help="seed = the problem's seed cut")
    parser.add_argument("--population_file", type=str, default=None, help="prove the best cut of this run")
    parser.add_argument("--cut_file", type=str, default=None, help="prove the cut of this Python file")
    parser.add_argument("--idea", type=str, default=None, help="the idea of a --cut_file cut (one paragraph)")
    parser.add_argument("--label", type=str, default=None, help="folder name of the result (default: from the cut)")
    parser.add_argument("--statement_file", type=str, default=None,
                        help="a Lean statement already written (and reviewed): skip the statement phase")
    parser.add_argument("--llm_backend", type=str, choices=[*BACKENDS, CLAUDE_CODE], default=parameters.LLM_BACKEND,
                        help="claude_code = Claude through the `claude` command (its account, no API key)")
    parser.add_argument("--model_name", type=str, default=parameters.MODEL_NAME,
                        help="for claude_code: an alias (opus, sonnet, fable, haiku) or a full model name")
    parser.add_argument("--harness", type=str, choices=["loop", "agent"], default="loop",
                        help="who writes the proof: loop = --llm_backend, one answer per round, each compiled; "
                             "agent = a Claude Code agent that edits and compiles by itself (FLARE's way)")
    parser.add_argument("--agent_model", type=str, default="opus", help="--harness agent: the agent's model")
    parser.add_argument("--agent_cli", type=str, choices=["claude", "vibe"], default="claude",
                        help="--harness agent: Claude Code, or Mistral Vibe with Leanstral and the lean-lsp MCP server")
    parser.add_argument("--leanstral_api", type=str, choices=["mistral", "requesty"], default="mistral",
                        help="--agent_cli vibe: Mistral's API (MISTRAL_API_KEY) or Requesty's free access")
    parser.add_argument("--agent_effort", type=str, default="medium", help="--harness agent: low, medium, high, ...")
    parser.add_argument("--agent_minutes", type=float, default=AGENT_MINUTES, help="--harness agent: time limit")
    parser.add_argument("--agent_budget", type=float, default=None,
                        help="--harness agent: stop the agent at this cost in USD (at API prices)")
    parser.add_argument("--ollama_host", type=str, default=parameters.OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=parameters.TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="ollama: the most tokens of one answer (num_predict); without it Ollama drops the "
                             "oldest context when full and a model that writes long analyses never stops")
    parser.add_argument("--llm_timeout", type=float, default=parameters.LLM_TIMEOUT, help="seconds for one answer")
    parser.add_argument("--statement_rounds", type=int, default=parameters.PROOF_STATEMENT_ROUNDS)
    parser.add_argument("--proof_rounds", type=int, default=parameters.PROOF_ROUNDS)
    parser.add_argument("--families", action="store_true",
                        help="state the cut as one definition per constraint family and prove each family alone "
                             "(loop harness); a given --statement_file written by families is proved that way too")
    parser.add_argument("--statement_only", action="store_true",
                        help="stop after the statement (statement.lean): to review it, or to give the same "
                             "statement to several provers with --statement_file")
    parser.add_argument("--no_judge", action="store_true", help="do not ask an LLM whether the statement is faithful")
    parser.add_argument("--lean_timeout", type=float, default=parameters.LEAN_TIMEOUT)
    parser.add_argument("--out_dir", type=str, default=parameters.PROOFS_DIR)
    parser.add_argument("--data_dir", type=str, default=parameters.DATA_DIR,
                        help="where the validity and benchmark sets are (their data assumptions are checked)")
    return parser.parse_args(argv)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "cut"


def cut_source(args: argparse.Namespace, problem: Problem) -> Tuple[str, str, str]:
    """(label, idea, code) of the cut the arguments name: exactly one of --cut seed,
    --population_file, --cut_file."""
    named: int = sum(x is not None for x in (args.cut, args.population_file, args.cut_file))
    if named != 1:
        raise ValueError("name one cut: --cut seed, --population_file <file> or --cut_file <file>")
    if args.cut == "seed":
        label, idea, code = "seed", problem.baseline_idea, problem.baseline_code
    elif args.population_file:
        if not os.path.exists(args.population_file):
            raise FileNotFoundError(f"no population file {args.population_file}")
        best = Population.load(args.population_file).best
        if best is None or best.code is None:
            raise ValueError(f"{args.population_file} has no best cut")
        label, idea, code = run_name_of(args.population_file), best.idea or "", best.code
    else:
        with open(args.cut_file) as f:
            code = f.read()
        label, idea = os.path.splitext(os.path.basename(args.cut_file))[0], args.idea or "(see the comments of the code)"
    return slug(args.label or label), idea, code


def checked_sets(problem: LeanProblem, data_dir: str) -> Dict[str, Any]:
    """The data assumptions of the Lean model, checked on the instance sets found on this machine
    (the validity set and the public benchmark): set -> {"instances": n, "broken": {name: [...]}}."""
    report: Dict[str, Any] = {}
    for folder, filename in (("validity", parameters.VALIDITY_FILE), ("benchmark", parameters.BENCHMARK_FILE)):
        loader = PROBLEMS[problem.name].loader(os.path.join(data_dir, folder))
        if os.path.exists(os.path.join(loader.folder, filename)):
            instances: List[Dict[str, Any]] = loader.load(filename)
            report[f"{folder}/{problem.name}/{filename}"] = {
                "instances": len(instances), "broken": broken_assumptions(problem, instances)}
    return report


class AttemptWriter:
    """Writes each attempt of the loop to <folder>/attempts/<phase>_<round>.json."""

    def __init__(self, folder: str) -> None:
        self.folder: str = os.path.join(folder, "attempts")
        os.makedirs(self.folder, exist_ok=True)

    def __call__(self, attempt: Attempt) -> None:
        path: str = os.path.join(self.folder, f"{attempt['phase']}_{attempt['round']:02d}.json")
        with open(path, "w") as f:
            json.dump({**attempt, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1, ensure_ascii=False)
        logger.info(f"{attempt['phase']} round {attempt['round']}: {attempt['outcome']}"
                    + (f" ({str(attempt['feedback']).splitlines()[0][:150]})" if attempt.get("feedback") else ""))


def run(args: argparse.Namespace, llm: Optional[LLMClient] = None) -> ProofResult:
    """Prove the cut the arguments name; `llm` replaces the one built from the arguments (tests)."""
    reason: Optional[str] = lean.unavailable()
    if reason:
        raise RuntimeError(reason)
    problem: Problem = PROBLEMS[args.problem_name]
    lean_problem: LeanProblem = LEAN_PROBLEMS[args.problem_name]
    label, idea, code = cut_source(args, problem)
    if llm is None and not (args.harness == "agent" and args.statement_file):  # that case asks no LLM here
        if not args.model_name:
            raise ValueError("--model_name is required")
        llm = (ClaudeCodeClient(args.model_name) if args.llm_backend == CLAUDE_CODE
               else make_llm(args.llm_backend, args.model_name, args.temperature, args.ollama_host))
        if hasattr(llm, "timeout"):
            llm.timeout = args.llm_timeout  # type: ignore[attr-defined]
        if args.max_tokens and hasattr(llm, "options"):
            llm.options["num_predict"] = args.max_tokens  # type: ignore[attr-defined]
    folder: str = os.path.join(args.out_dir, args.problem_name, label)
    os.makedirs(folder, exist_ok=True)
    statement: Optional[str] = None
    if args.statement_file:
        with open(args.statement_file) as f:
            statement = f.read().strip()
    model_code: str = inspect.getsource(problem.base)
    prover: Prover = Prover(
        llm, lean_problem, model_code, folder,  # type: ignore[arg-type]  (llm is None only when it is not asked)
        statement_rounds=args.statement_rounds, proof_rounds=args.proof_rounds, judge=not args.no_judge,
        by_families=args.families, timeout=args.lean_timeout, record=AttemptWriter(folder))
    start: float = time.time()
    if args.statement_only:
        result: ProofResult = prover.result
        result.statement = statement or prover.formalize(idea, code)
        result.status = NO_STATEMENT if result.statement is None else "statement_only"
        result.reason = result.reason if result.statement is None else "statement written, no proof asked"
    elif args.harness == "loop":
        result = prover.run(idea, code, statement)
    else:  # the statement as in the loop (or given), then the agent proves it
        result = prover.result
        problem_found: Optional[str] = prover.check_statement(statement) if statement else None
        if statement is None:
            statement = prover.formalize(idea, code)
        if statement is None or problem_found:
            result.status, result.reason = NO_STATEMENT, problem_found or result.reason
        else:
            agent: AgentProver = AgentProver(
                lean_problem, model_code, label, model=args.agent_model, effort=args.agent_effort,
                minutes=args.agent_minutes, budget_usd=args.agent_budget, timeout=args.lean_timeout,
                cli=args.agent_cli, api=args.leanstral_api,
                env=({LEANSTRAL_APIS[args.leanstral_api][2]: api_key(LEANSTRAL_APIS[args.leanstral_api][2])}
                     if args.agent_cli == "vibe" else None))
            proved: ProofResult = agent.run(idea, code, statement, folder)
            proved.statement_rounds, proved.judge = result.statement_rounds, result.judge
            proved.llm_calls += result.llm_calls
            proved.llm_seconds += result.llm_seconds
            proved.lean_seconds += result.lean_seconds
            result = proved
    for name, text in (("statement.lean", result.statement), ("proof.lean", result.proof)):
        if text is not None:
            with open(os.path.join(folder, name), "w") as f:
                f.write(text + "\n")
    with open(os.path.join(folder, RESULT_FILE), "w") as f:
        json.dump({
            **result.to_dict(), "label": label, "idea": idea, "code": code,
            "llm_backend": args.llm_backend, "model_name": args.model_name, "harness": args.harness,
            "by_families": args.families,
            "agent_model": args.agent_model if args.harness == "agent" else None,
            "agent_cli": args.agent_cli if args.harness == "agent" else None,
            "statement_given": args.statement_file, "judge_asked": not args.no_judge,
            "seconds": round(time.time() - start, 1),
            "data_assumptions": checked_sets(lean_problem, args.data_dir),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=1, ensure_ascii=False)
    print(f"{args.problem_name} / {label}: {result.status.upper()} ({result.reason})")
    print(f"  {result.statement_rounds} statement rounds, {result.proof_rounds} proof rounds, {result.llm_calls} LLM "
          f"calls ({result.llm_seconds:.0f} s), Lean {result.lean_seconds:.0f} s -> {folder}")
    if result.proved:
        print(f"  the certificate is about {os.path.join(folder, 'statement.lean')}: read it against the code")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())
