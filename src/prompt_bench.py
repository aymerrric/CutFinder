"""
Compare prompt variants WITHOUT running the genetic algorithm.

A full run per prompt variant takes hours, and its result depends on luck (one run, noisy
fitness). Here every variant gets exactly the same requests: the same parents (taken from a saved
population), the same operators, the same LLM seed per request. Each answer goes through the
cheap checks of the run (format, code, validity, usefulness: 15-city instances, ~0.4 s), and
optionally a quick score on a few evaluation instances (--score_instances). So the comparison
measures the prompts, not the luck of a run, and costs mostly LLM time.

Run from src/, on a lab machine with Ollama (or with fleet.py bench, see computer_connection/):
    python prompt_bench.py --model_name qwen3-coder:30b --data_dir ../data/rtx3090 \\
        --population_file ../runs/<run>/population.json --variants default rules concise \\
        --samples 5 --score_instances 2

Outputs runs/prompt_bench/<name>/: report.md (read this), results.jsonl (one line per answer,
with the reply, the code and the outcome), config.json. The report is rewritten after every
round, so a stopped benchmark still has one.

Split over several machines (one variant each, and/or --sample_offset for other rounds), then
one report for all:  python prompt_bench.py --merge ../runs/prompt_bench/<out> <part folders...>
"""
import argparse
import inspect
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from evolution.evaluator import CutEvaluator
from evolution.genetic import CutRejected
from evolution.individual import SEED, Individual, normalized_code
from evolution.population import Population
from evolution.prompts import Prompts, ReplyFormatError, parse_reply, prompt_variants
from llm.base import LLMClient, LLMError, Message
from llm.ollama import OllamaClient
from loaders.base import Instance, Loader
from main import PROBLEMS, Problem, check_same_machine
from solvers.solver import Solver, machine_info
from utils import parameters

logger: logging.Logger = logging.getLogger("prompt_bench")

# task kinds
INIT: str = "init"
MUTATION: str = "mutation"
CROSSOVER: str = "crossover"
KINDS: List[str] = [INIT, MUTATION, CROSSOVER]

# outcomes of an answer (the first five are refusals)
FORMAT: str = "format_error"
CODE_ERROR: str = "code_error"  # the code raises (ERROR: e.g. KeyError, NameError)
TOO_SLOW: str = "too_slow"  # too slow to build or solve, or too many constraints
INVALID: str = "invalid"
USELESS: str = "useless"
LLM_FAILED: str = "llm_error"
ACCEPTED: str = "accepted"
OUTCOMES: List[str] = [ACCEPTED, FORMAT, CODE_ERROR, TOO_SLOW, INVALID, USELESS, LLM_FAILED]


@dataclass
class Task:
    """One request, identical for every variant (except the operator texts of the variant)."""

    kind: str
    operator: str  # operator name of the variant ("init" for the initializer)
    slot: int  # position of the operator in its kind: pairs the requests of two variants
    sample: int
    parents: List[Individual]
    llm_seed: int  # same seed for the same request in every variant: paired comparison


@dataclass
class Result:
    variant: str
    kind: str
    operator: str
    slot: int
    sample: int
    outcome: str
    attempts: int  # answers asked (> 1 when --attempts > 1 and the first ones were refused)
    reason: Optional[str] = None
    idea: Optional[str] = None
    code: Optional[str] = None
    fitness: Optional[float] = None
    copy_of_parent: bool = False
    llm_seconds: float = 0.0
    eval_seconds: float = 0.0
    parents: List[str] = field(default_factory=list)
    replies: List[str] = field(default_factory=list)


def classify(reason: str) -> str:
    """Outcome of a CutRejected message (see CutEvaluator and Solver statuses)."""
    if reason.startswith("invalid"):
        return INVALID
    if reason.startswith("useless"):
        return USELESS
    if reason.startswith(("too slow", "KILLED", "TOO_LARGE")):
        return TOO_SLOW
    return CODE_ERROR


normalize = normalized_code  # code without comments and spaces: to detect copies and duplicates


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% confidence interval of a proportion (Wilson): honest even for small n."""
    if n == 0:
        return 0.0, 0.0
    p: float = successes / n
    centre: float = (p + z * z / (2 * n)) / (1 + z * z / n)
    half: float = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pick_parents(population: Optional[Population], seed: Individual, k: int) -> List[Individual]:
    """The seed cut and the k best distinct cuts of the population (all generations)."""
    parents: List[Individual] = [seed]
    if population is not None:
        everyone: List[Individual] = [ind for gen in population.history + [population.current] for ind in gen]
        seen: set = {normalize(seed.code)}
        for ind in sorted(everyone, key=lambda i: i.fitness, reverse=True):
            if normalize(ind.code) not in seen and len(parents) < k + 1:
                parents.append(ind)
                seen.add(normalize(ind.code))
    return parents


def make_tasks(
    prompts: Prompts, kinds: List[str], samples: int, parents: List[Individual], offset: int = 0
) -> List[Task]:
    """The requests of one variant: per sample, one initializer request, one per mutation
    operator (parent cycling over `parents`), one per crossover operator (pairs cycling)."""
    pairs: List[Tuple[Individual, Individual]] = [
        (a, b) for i, a in enumerate(parents) for b in parents[i + 1:]
    ]
    tasks: List[Task] = []
    for s in range(offset, offset + samples):  # offset: rounds split over several machines
        if INIT in kinds:
            tasks.append(Task(INIT, INIT, 0, s, [parents[0]], llm_seed=1000 * s))
        if MUTATION in kinds:
            for j, operator in enumerate(prompts.mutations):
                tasks.append(Task(MUTATION, operator, j, s, [parents[(s + j) % len(parents)]], 1000 * s + 100 + j))
        if CROSSOVER in kinds and pairs:
            for j, operator in enumerate(prompts.crossovers):
                tasks.append(Task(CROSSOVER, operator, j, s, list(pairs[(s + j) % len(pairs)]), 1000 * s + 200 + j))
    return tasks


def messages_for(prompts: Prompts, task: Task, best_fitness: float) -> List[Message]:
    if task.kind == INIT:
        seed: Individual = task.parents[0]
        return prompts.initializer_messages(seed, [seed.idea], best_fitness)
    if task.kind == MUTATION:
        return prompts.mutation_messages(task.operator, task.parents[0], best_fitness)
    return prompts.crossover_messages(task.operator, task.parents[0], task.parents[1], best_fitness)


def ask(
    variant: str, prompts: Prompts, task: Task, llm: LLMClient, checker: CutEvaluator,
    scorer: Optional[CutEvaluator], attempts: int, best_fitness: float,
) -> Result:
    """Ask the LLM (up to `attempts` answers, with the run's feedback), check the answer."""
    result: Result = Result(variant, task.kind, task.operator, task.slot, task.sample, LLM_FAILED, 0,
                            parents=[p.id for p in task.parents])
    if isinstance(llm, OllamaClient):
        llm.options["seed"] = task.llm_seed
    messages: List[Message] = messages_for(prompts, task, best_fitness)
    for attempt in range(1, attempts + 1):
        result.attempts = attempt
        start: float = time.time()
        try:
            reply: str = llm.send(messages)
        except LLMError as e:
            result.outcome, result.reason = LLM_FAILED, str(e)
            return result
        finally:
            result.llm_seconds += time.time() - start
        result.replies.append(reply)
        messages.append({"role": "assistant", "content": reply})
        try:
            result.idea, result.code = parse_reply(reply)
            start = time.time()
            try:
                checker.check(result.code)
            finally:
                result.eval_seconds += time.time() - start
        except ReplyFormatError as e:
            result.outcome, result.reason = FORMAT, str(e)
        except CutRejected as e:
            result.outcome, result.reason = classify(str(e)), str(e)
        else:
            result.outcome, result.reason = ACCEPTED, None
            result.copy_of_parent = normalize(result.code) in {normalize(p.code) for p in task.parents}
            if scorer is not None:
                start = time.time()
                try:
                    result.fitness = scorer(result.code)
                except CutRejected as e:  # e.g. the bound check on the evaluation instances
                    result.outcome, result.reason = classify(str(e)), str(e)
                result.eval_seconds += time.time() - start
            if result.outcome == ACCEPTED:
                return result
        messages.append(prompts.retry_message(result.reason or ""))
    return result


# ---------------------------------------------------------------- report

def mcnemar(better: int, worse: int) -> float:
    """Exact two-sided p-value of the paired comparison (sign test on the discordant requests):
    the probability of a split at least this uneven if both variants were equally good."""
    n: int = better + worse
    if n == 0:
        return 1.0
    tail: float = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _bar(rate: float, width: int = 20) -> str:
    return "█" * round(rate * width) + "·" * (width - round(rate * width))


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "-"


@dataclass
class VariantStats:
    """Ranked by `new`: accepted answers that are not a copy of a parent (a copy is accepted by
    the checks, since the parent is valid, but brings nothing to a run)."""

    variant: str
    answers: int
    accepted: int  # every accepted answer, copies of a parent included
    new: int  # accepted and not a copy of a parent
    low: float  # 95% interval of new / answers
    high: float
    distinct: int  # distinct accepted codes that are not a copy of a parent
    copies: int
    fitness: List[float]
    seconds: float  # LLM seconds per answer
    better: int = 0  # paired with the reference: accepted where the reference is not
    worse: int = 0
    p_value: float = 1.0

    @property
    def rate(self) -> float:
        return self.new / self.answers if self.answers else 0.0


def variant_stats(results: List[Result], variants: List[str]) -> List[VariantStats]:
    by_variant: DefaultDict[str, List[Result]] = defaultdict(list)
    for r in results:
        by_variant[r.variant].append(r)
    accepted: Dict[Tuple[str, str, int, int], bool] = {  # a new cut was obtained for this request
        (r.variant, r.kind, r.slot, r.sample): r.outcome == ACCEPTED and not r.copy_of_parent for r in results
    }
    ref: str = variants[0]
    stats: List[VariantStats] = []
    for v in variants:
        rs: List[Result] = by_variant[v]
        if not rs:
            continue
        acc: List[Result] = [r for r in rs if r.outcome == ACCEPTED]
        new: int = sum(not r.copy_of_parent for r in acc)
        low, high = wilson(new, len(rs))
        vs: VariantStats = VariantStats(
            v, len(rs), len(acc), new, low, high,
            distinct=len({normalize(r.code) for r in acc if r.code and not r.copy_of_parent}),
            copies=sum(r.copy_of_parent for r in acc),
            fitness=[r.fitness for r in acc if r.fitness is not None],
            seconds=statistics.mean(r.llm_seconds / max(r.attempts, 1) for r in rs),
        )
        if v != ref:
            for (variant, kind, slot, sample), ok in accepted.items():
                other: Optional[bool] = accepted.get((ref, kind, slot, sample))
                if variant == v and other is not None:
                    vs.better += ok and not other
                    vs.worse += other and not ok
            vs.p_value = mcnemar(vs.better, vs.worse)
        stats.append(vs)
    return stats


def _fitness(values: List[float]) -> str:
    if not values:
        return "-"
    std: float = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {std:.2f} (n={len(values)})"


def verdict(stats: List[VariantStats], ref: str) -> List[str]:
    ranked: List[VariantStats] = sorted(stats, key=lambda s: s.rate, reverse=True)
    reference: Optional[VariantStats] = next((s for s in stats if s.variant == ref), None)
    best: VariantStats = ranked[0]
    lines: List[str] = [f"- **Most new cuts accepted:** `{best.variant}`, {100 * best.rate:.0f}% of its answers "
                        f"(95% interval {100 * best.low:.0f}–{100 * best.high:.0f}%)."]
    if reference is not None:
        lines.append(f"- **Reference** `{ref}`: {100 * reference.rate:.0f}% "
                     f"({100 * reference.low:.0f}–{100 * reference.high:.0f}%).")
        significant: List[VariantStats] = [s for s in stats if s.variant != ref and s.p_value < 0.05]
        up: List[str] = [f"`{s.variant}` (+{100 * (s.rate - reference.rate):.0f} pts, p={s.p_value:.3f})"
                         for s in significant if s.rate > reference.rate]
        down: List[str] = [f"`{s.variant}` ({100 * (s.rate - reference.rate):.0f} pts, p={s.p_value:.3f})"
                           for s in significant if s.rate < reference.rate]
        lines.append("- **Significantly better than the reference** (paired test, p < 0.05): "
                     + (", ".join(up) if up else "none."))
        lines.append("- **Significantly worse:** " + (", ".join(down) if down else "none."))
    scored: List[VariantStats] = [s for s in stats if len(s.fitness) >= 5]
    if scored:
        top: VariantStats = max(scored, key=lambda s: statistics.mean(s.fitness))
        lines.append(f"- **Best fitness of the accepted cuts** (quick score, indicative): `{top.variant}`, "
                     f"{_fitness(top.fitness)}.")
    most: VariantStats = max(stats, key=lambda s: s.distinct / s.answers if s.answers else 0)
    lines.append(f"- **Most distinct new cuts per answer:** `{most.variant}` ({most.distinct} distinct "
                 f"out of {most.answers} answers).")
    return lines


def report(results: List[Result], config: Dict[str, Any]) -> str:
    """The benchmark report (markdown): verdict, ranking, quality, tasks, refusals, setup."""
    variants: List[str] = [v for v in config["variants"] if any(r.variant == v for r in results)]
    if not variants:
        return f"# Prompt benchmark {config['name']}\n\nNo answer yet.\n"
    ref: str = variants[0]
    stats: List[VariantStats] = variant_stats(results, variants)
    rounds: int = len({r.sample for r in results})
    lines: List[str] = [
        f"# Prompt benchmark {config['name']}", "",
        f"{len(results)} answers from `{config['model_name']}`: {len(variants)} prompt variants × "
        f"{rounds} rounds of the same requests. Reference: `{ref}`.", "",
        "## Verdict", "",
    ] + verdict(stats, ref) + [
        "", "## Ranking (new cuts accepted)", "",
        "An answer is *accepted* when it passes every check of a run (format, code, validity, "
        "usefulness). The ranking counts only **new** accepted cuts: an answer that repeats a parent "
        "passes the checks (the parent is valid) but brings nothing to a run. *vs ref*: on the same "
        "requests, a new cut where the reference has none / the reverse, and the p-value of that split "
        "(exact McNemar test; below 0.05 = a real difference). *Distinct*: different codes among the "
        "new cuts.", "",
        "| # | variant | new cuts accepted | | 95% interval | vs ref (better / worse, p) | all accepted (copies) | distinct | s / answer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, st in enumerate(sorted(stats, key=lambda x: x.rate, reverse=True), 1):
        vs: str = "reference" if st.variant == ref else f"{st.better} / {st.worse}, p={st.p_value:.3f}"
        lines.append(
            f"| {rank} | `{st.variant}` | {st.new}/{st.answers} ({100 * st.rate:.0f}%) | `{_bar(st.rate)}` "
            f"| {100 * st.low:.0f}–{100 * st.high:.0f}% | {vs} | {st.accepted} ({st.copies}) | {st.distinct} "
            f"| {st.seconds:.1f} |"
        )
    if any(st.fitness for st in stats):
        lines += ["", "## Quality of the accepted cuts", "",
                  f"Quick score on {config['score_instances']} evaluation instance(s), one Gurobi seed: "
                  "10 = same gap as the model without cut, higher is better. Indicative only (Step 15: "
                  "single-seed scores are noisy).", "",
                  "| variant | fitness mean ± std | best | share above 10 |", "|---|---|---|---|"]
        for st in sorted(stats, key=lambda x: statistics.mean(x.fitness) if x.fitness else 0, reverse=True):
            if st.fitness:
                above: int = sum(f > 10 for f in st.fitness)
                lines.append(f"| `{st.variant}` | {_fitness(st.fitness)} | {max(st.fitness):.2f} "
                             f"| {_pct(above, len(st.fitness))} |")
    lines += ["", "## Accepted answers per task", "", "| variant | " + " | ".join(KINDS) + " |",
              "|---|" + "---|" * len(KINDS)]
    for st in stats:
        cells: List[str] = []
        for kind in KINDS:
            rs: List[Result] = [r for r in results if r.variant == st.variant and r.kind == kind]
            ok: int = sum(r.outcome == ACCEPTED for r in rs)
            cells.append(f"{_pct(ok, len(rs))} ({ok}/{len(rs)})" if rs else "-")
        lines.append(f"| `{st.variant}` | " + " | ".join(cells) + " |")
    lines += ["", "## Why answers were refused (share of all answers)", "",
              "| variant | " + " | ".join(OUTCOMES[1:]) + " |", "|---|" + "---|" * (len(OUTCOMES) - 1)]
    for st in stats:
        counts: Counter = Counter(r.outcome for r in results if r.variant == st.variant)
        lines.append(f"| `{st.variant}` | " + " | ".join(_pct(counts[o], st.answers) for o in OUTCOMES[1:]) + " |")
    lines += [
        "", "## Setup", "",
        f"- **LLM:** {config['model_name']} (temperature {config['temperature'] or 'default'}), "
        f"{config['attempts']} answer(s) per request",
        f"- **Requests:** per round and per variant, one initializer request, one per mutation operator, one "
        f"per crossover operator; the same parents and the same LLM seed for the same request in every variant",
        f"- **Parents:** {len(config['parents'])} cuts: the seed cut and the best distinct cuts of "
        f"`{config['population_file'] or 'no population: seed only'}`",
        f"- **Checks:** format, code, validity, usefulness, as in a run"
        + (f"; accepted cuts scored on {config['score_instances']} evaluation instance(s)" if config["score_instances"] else ""),
        f"- **Machines:** {config['machine']}; started {config['started']}",
        "", "Every answer (reply, code, outcome, reason) is in `results.jsonl`. Next step: a full run with the "
        "best variant (`main.py --prompt_variant <name>`), reviewed under several seeds.",
    ]
    return "\n".join(lines) + "\n"


def merge(folders: List[str], out: str) -> str:
    """One report for benchmarks run in parallel (same requests, split by variant or by rounds):
    reads their config.json and results.jsonl, writes out/report.md and out/results.jsonl."""
    results: List[Result] = []
    configs: List[Dict[str, Any]] = []
    for folder in folders:
        with open(os.path.join(folder, "config.json")) as f:
            configs.append(json.load(f))
        with open(os.path.join(folder, "results.jsonl")) as f:
            results += [Result(**json.loads(line)) for line in f if line.strip()]
    variants: List[str] = []
    for c in configs:  # keep the first config's order: its first variant is the reference
        variants += [v for v in c["variants"] if v not in variants]
    if "default" in variants:
        variants.remove("default")
        variants.insert(0, "default")
    config: Dict[str, Any] = {
        **configs[0], "name": os.path.basename(os.path.normpath(out)), "variants": variants,
        "machine": ", ".join(sorted({c["machine"].split(" ")[0] for c in configs})),
        "started": min(c["started"] for c in configs), "parts": folders,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        f.writelines(json.dumps(asdict(r)) + "\n" for r in results)
    path: str = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write(report(results, config))
    return path


# ---------------------------------------------------------------- main

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt variants without running the GA")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="variants to compare (default: every one); the first is the reference")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--population_file", type=str, default=None,
                        help="a saved population: its best distinct cuts become the parents")
    parser.add_argument("--parents", type=int, default=3, help="best distinct cuts used as parents")
    parser.add_argument("--samples", type=int, default=5, help="rounds of requests per variant")
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="first round number (to split the rounds over several machines)")
    parser.add_argument("--tasks", nargs="+", default=KINDS, choices=KINDS)
    parser.add_argument("--attempts", type=int, default=1,
                        help="answers per request (> 1 also tests the feedback on refusals)")
    parser.add_argument("--score_instances", type=int, default=0,
                        help="score accepted cuts on the first k evaluation instances (0 = no score)")
    parser.add_argument("--temperature", type=float, default=parameters.TEMPERATURE)
    parser.add_argument("--name", type=str, default=None, help="folder name (default: date)")
    parser.add_argument("--problem_name", type=str, default=parameters.PROBLEM_NAME)
    parser.add_argument("--ollama_host", type=str, default=parameters.OLLAMA_HOST)
    parser.add_argument("--data_dir", type=str, default=parameters.DATA_DIR)
    parser.add_argument("--runs_dir", type=str, default=parameters.RUNS_DIR)
    parser.add_argument("--work_limit", type=float, default=parameters.WORK_LIMIT)
    parser.add_argument("--threads", type=int, default=parameters.THREADS)
    parser.add_argument("--cut_time_limit", type=float, default=parameters.CUT_TIME_LIMIT)
    parser.add_argument("--cut_build_timeout", type=float, default=parameters.CUT_BUILD_TIMEOUT)
    return parser.parse_args(argv)


def run_bench(args: argparse.Namespace, llm: Optional[LLMClient] = None) -> List[Result]:
    """Every variant answers the same requests; results.jsonl and report.md are written as it goes."""
    problem: Problem = PROBLEMS[args.problem_name]
    loader: Loader = problem.loader(args.data_dir)
    validation: List[Instance] = loader.load(parameters.VALIDATION_FILE)
    evaluation: List[Instance] = loader.load(parameters.EVALUATION_FILE)
    check_same_machine(evaluation, args.threads)
    variants: List[str] = args.variants or prompt_variants()
    model_code: str = inspect.getsource(problem.base)
    prompts: Dict[str, Prompts] = {v: Prompts(model_code, variant=v) for v in variants}  # fails early if one is broken

    name: str = args.name or time.strftime("%Y%m%d_%H%M%S")
    folder: str = os.path.join(args.runs_dir, "prompt_bench", name)
    os.makedirs(folder, exist_ok=True)
    solver: Solver = Solver(work_limit=args.work_limit, threads=args.threads, log_dir=None,
                            time_limit=args.cut_time_limit, build_timeout=args.cut_build_timeout)
    checker: CutEvaluator = CutEvaluator(problem.base, validation, evaluation, solver)
    scorer: Optional[CutEvaluator] = (
        CutEvaluator(problem.base, validation, evaluation[:args.score_instances], solver)
        if args.score_instances > 0 else None
    )

    population: Optional[Population] = Population.load(args.population_file) if args.population_file else None
    seed: Individual
    if population is not None:
        first: List[Individual] = population.history[0] if population.history else population.current
        seed = next(ind for ind in first if ind.generator == SEED)
    else:  # no population: the seed cut, scored once
        seed = Individual(idea=problem.baseline_idea, code=problem.baseline_code, generator=SEED,
                          fitness=checker(problem.baseline_code))
    parents: List[Individual] = pick_parents(population, seed, args.parents)
    best_fitness: float = max(p.fitness for p in parents)

    if llm is None:
        options: Dict[str, Any] = {} if args.temperature is None else {"temperature": args.temperature}
        llm = OllamaClient(args.model_name, host=args.ollama_host, options=options)
    host, cpu = machine_info()
    config: Dict[str, Any] = {
        "name": name, "variants": variants, "model_name": args.model_name, "temperature": args.temperature,
        "samples": args.samples, "tasks": args.tasks, "attempts": args.attempts,
        "score_instances": args.score_instances, "population_file": args.population_file,
        "parents": [asdict(p) for p in parents], "machine": f"{host} ({cpu})",
        "started": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    tasks: Dict[str, List[Task]] = {
        v: make_tasks(prompts[v], args.tasks, args.samples, parents, args.sample_offset) for v in variants
    }
    results: List[Result] = []
    first: int = args.sample_offset
    for sample in range(first, first + args.samples):  # round by round: a stopped benchmark stays balanced
        for v in variants:
            for task in (t for t in tasks[v] if t.sample == sample):
                result: Result = ask(v, prompts[v], task, llm, checker, scorer, args.attempts, best_fitness)
                results.append(result)
                logger.info(f"{v} {task.kind}/{task.operator} sample {sample}: {result.outcome}"
                            + (f" ({result.reason[:80]})" if result.reason else "")
                            + (f", fitness {result.fitness:.2f}" if result.fitness is not None else ""))
                with open(os.path.join(folder, "results.jsonl"), "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
        with open(os.path.join(folder, "report.md"), "w") as f:
            f.write(report(results, config))
        logger.info(f"round {sample - first + 1}/{args.samples} done: {os.path.join(folder, 'report.md')}")
    return results


if __name__ == "__main__":  # required: the Solver's child processes import this file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("solvers.solver").setLevel(logging.WARNING)
    logging.getLogger("llm.ollama").setLevel(logging.WARNING)
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":  # python prompt_bench.py --merge OUT PART...
        print(merge(sys.argv[3:], sys.argv[2]))
    else:
        run_bench(parse_args())
