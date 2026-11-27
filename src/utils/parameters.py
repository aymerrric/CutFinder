import os
from typing import Dict, List, Optional, Tuple

# Default values of every setting. The command-line flags of parser.py use them as defaults.

# Paths: absolute, computed from this file, so they do not depend on the working directory.
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")  # datasets, one subfolder per problem
PROMPTS_DIR: str = os.path.join(PROJECT_ROOT, "prompts")  # LLM prompts (YAML), versioned with the code
CACHE_DIR: str = os.path.join(PROJECT_ROOT, "cache")  # where solver results are cached
LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")  # Gurobi log files, one per solve
RUNS_DIR: str = os.path.join(PROJECT_ROOT, "runs")  # one <run>.json (population) + <run>.log per run

# Gurobi license (secret, git-ignored). Gurobi reads GRB_LICENSE_FILE when it first starts,
# so this must run before any gurobipy model is created.
GUROBI_LICENSE_FILE: str = os.path.join(PROJECT_ROOT, "gurobi.lic")


def use_license_file(path: str) -> None:
    """
    Point Gurobi to `path`, unless GRB_LICENSE_FILE is already exported (an explicit choice wins)
    or `path` does not exist. Without a file, Gurobi falls back to its other licenses (e.g. the
    size-limited one shipped with pip), instead of failing on a missing file.
    """
    if os.path.exists(path):
        os.environ.setdefault("GRB_LICENSE_FILE", path)


use_license_file(GUROBI_LICENSE_FILE)

# LLM
# None = to be chosen with --model_name (a local LLM; several can be compared run by run).
MODEL_NAME: Optional[str] = None
LLM_MAX_RETRIES: int = 3  # tries per request when the call itself fails (server down, timeout)
LLM_RETRY_DELAY: float = 1.0  # seconds before the 2nd try, doubled before each next one
OLLAMA_HOST: str = "http://localhost:11434"  # Ollama server (an SSH tunnel to a lab machine)
LLM_TIMEOUT: float = 600  # seconds to wait for one answer (a long answer can take minutes)
# Where the LLM runs: "ollama" (local server, OLLAMA_HOST) or one of the OpenAI-compatible APIs
# below (llm/openai_compatible.py). Provider -> (base URL, environment variable holding its key).
LLM_BACKEND: str = "ollama"
API_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "muse": ("https://api.meta.ai/v1", "MUSE_API_KEY"),  # Meta Model API (Muse models)
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),  # Leanstral (prove_cut.py)
    "requesty": ("https://router.requesty.ai/v1", "REQUESTY_API_KEY"),  # a router: mistral/leanstral-1-5, ...
}
# API keys (secret, git-ignored): one KEY=value line per provider, e.g. GROQ_API_KEY=gsk_...
# An environment variable of the same name wins. In the project root, so the lab machines see it
# through the shared home (fleet.py sync copies it, never git).
LLM_KEYS_FILE: str = os.path.join(PROJECT_ROOT, "llm_keys.env")
API_MAX_WAIT: float = 120  # seconds: longest wait asked by a rate limit (HTTP 429) that is honoured
API_BUSY_WAIT: float = 60  # seconds waited after a 429 / 500-504 that gives no Retry-After (overloads last minutes)
API_MAX_RETRIES: int = 6  # tries per request for an API backend (Gemini's "high demand" 503s come and go)

# Problem
PROBLEM_NAME: str = "tsp"

# Instance sets, each solved once with the base formulation (reference: 2 verification, 8 evaluation).
# Their sizes depend on the problem: problems.py (Problem.validation_sizes, ...).
N_VALIDATION: int = 2  # solved to optimality: their optimal solutions check that a cut is valid
VALIDATION_SEED: int = 1
VALIDATION_FILE: str = "validation.pkl"  # in data/<problem>/
VALIDATION_TIME_LIMIT: float = 600  # seconds: validation instances must reach OPTIMAL
# Validity instances (prepare_validity.py, loaders/validity.py): many small instances, tight and
# public ones included, solved to optimality; a cut must keep every optimal solution. Optimality
# does not depend on the CPU: one folder for every machine, built once.
VALIDITY_DIR: str = os.path.join(DATA_DIR, "validity")  # validity/<problem>/validity.pkl
VALIDITY_FILE: str = "validity.pkl"
VALIDITY_TIME_LIMIT: float = 600  # seconds per instance; an instance not solved to optimality is left out
# seconds per instance for the safe check (evolution/evaluator.py): when a cut removes the stored
# optimum, the instance is re-solved with the cut until it finds a solution of the optimal cost or
# proves there is none. Undecided at this limit = refused. The base model's slowest validity
# instance is JSSP orb01, 122 s on the Mac M4 (learnt_results.md 12.3).
SAFE_CHECK_TIME_LIMIT: float = 300
N_EVALUATION: int = 8  # solved under WORK_LIMIT: used to score the cuts
MIN_BASE_GAP: float = 0.002  # evaluation instances with a smaller base gap are replaced
#                              (reference GAP_LOWERBOUND = 0.2, in percent)
EVALUATION_SEED: int = 100  # replacements use the next seeds (101, 102, ...): no clash with 1
EVALUATION_FILE: str = "evaluation.pkl"
# Held-out test set: the end-of-run review scores the best cut on these instances, never seen by
# the GA (a cut selected on the evaluation set is biased upwards there: learnt_results.md 1.4).
# Built like the evaluation set (same work limit, base gap > MIN_BASE_GAP), from other seeds.
N_TEST: int = 8
TEST_SEED: int = 1000  # replacements use 1001, 1002, ...: no clash with 1 or 100-119
TEST_FILE: str = "test.pkl"
# Instance level of the evaluation and test sets (--instance_level of main.py and
# prepare_instances.py). "default": Problem.evaluation_sizes / test_sizes, in evaluation.pkl and
# test.pkl. "hard": Problem.hard_sizes (larger instances mixed with the default sizes), in
# evaluation_hard.pkl and test_hard.pkl, from other seeds: for when the best cuts solve the default
# instances to optimality and stop being ranked (learnt_results.md 5.15). The validation set is the
# same at every level. A run keeps its level when resumed (it is in its argv), and runs of
# different levels are never averaged together (evolution/report.py).
INSTANCE_LEVEL: str = "default"
INSTANCE_LEVELS: List[str] = ["default", "hard"]
# Seed of each (set, level); replacements of refused instances use the next 19 seeds at most.
SET_SEEDS: Dict[Tuple[str, str], int] = {
    ("evaluation", "default"): EVALUATION_SEED, ("test", "default"): TEST_SEED,
    ("evaluation", "hard"): 2000, ("test", "hard"): 3000,
    ("selection", "default"): 6000, ("selection", "hard"): 7000,
}
# Selection set: a second held-out set, built like the test set (same sizes) from other seeds, to
# choose a run's final cut among its best ones instead of taking the highest run fitness (the GA
# overfits its evaluation instances: learnt_results.md 12.14). Built by prepare_instances.py
# --selection; used by select_cuts.py.
SELECTION_FILE: str = "selection.pkl"
N_SELECTION: int = 16
# Optimality sets of the "gap_work" fitness (evolution.evaluator.FitnessMode.GAP_WORK): instances
# the base model PROVES optimal under WORK_LIMIT, with a work between OPTIMALITY_WORK (enough to
# measure a change, and room under the limit for a cut that slows the proof down; measured in
# runs/analysis/tsp_optimality_sizes_20261112.txt). One set scored during the run, one held out for
# the review. Built by prepare_instances.py --optimality (sizes: Problem.optimality_sizes).
N_OPTIMALITY: int = 4  # each adds one solve per fitness seed to the scoring of every cut
OPTIMALITY_WORK: Tuple[float, float] = (1.0, 5.0)
OPTIMALITY_FILE: str = "optimality.pkl"
OPTIMALITY_TEST_FILE: str = "optimality_test.pkl"
OPTIMALITY_SEED: int = 4000
OPTIMALITY_TEST_SEED: int = 5000

# Public benchmark sets of the paper (its test set D_t: TSPLIB, OR-Library CWLP, Taillard JSSP),
# built by prepare_benchmark.py. They need no solve, so they do not depend on the CPU: one folder
# for every machine. Raw downloads go to BENCHMARK_DIR/raw/<problem>/.
BENCHMARK_DIR: str = os.path.join(DATA_DIR, "benchmark")
BENCHMARK_FILE: str = "benchmark.pkl"  # in BENCHMARK_DIR/<problem>/
# Test protocol of the paper (Section 5.1-5.2), used by benchmark_cut.py: every instance solved
# once without and once with the cut, for BENCHMARK_TIME_LIMIT seconds of wall clock; the gap is
# read at each checkpoint. Gurobi defaults except MIPGap and Method (8 threads: THREADS).
BENCHMARK_TIME_LIMIT: float = 2000
BENCHMARK_CHECKPOINTS: List[float] = [5, 10, 50, 150, 300, 1000, 2000]
BENCHMARK_GUROBI_PARAMS: Dict[str, float] = {"MIPGap": 1e-5, "Method": 4}
BENCHMARK_SEED: int = 0  # Gurobi's default seed (the paper solves each instance once)
BENCHMARK_EPSILON: float = 1e-5  # Delta g = (g_ref - g_aug) / max(g_ref, epsilon)
BENCHMARK_GAP_TOLERANCE: float = BENCHMARK_GUROBI_PARAMS["MIPGap"]  # a gap <= this counts as 0 (optimal)
BENCHMARK_CLIP: float = 1.0  # Delta g clipped to [-1, 1] (evolution/benchmark.py)
BENCHMARK_BUILD_TIMEOUT: float = 600  # seconds to build a model (100 x 20 JSSP + cut) before the kill
# The one CPU model the whole benchmark is solved on (the lab's a5000 machines, 14 with their own
# Gurobi license): a gap at a wall-clock time depends on the CPU, so a table never mixes CPUs.
BENCHMARK_CPU: str = "Intel(R) Xeon(R) W-1270P CPU @ 3.80GHz"
# The same checkpoints in Gurobi work units (CPU-independent; the paper gives seconds only): a
# second table of the report. A work unit is very roughly a second (here 1.4-1.6 work per second).
BENCHMARK_WORK_CHECKPOINTS: List[float] = [5, 10, 50, 150, 300, 1000, 2000]
BENCHMARK_LP_TIME_LIMIT: float = 600  # seconds for one LP relaxation (benchmark_cut.py --lp)
BENCHMARK_RUNS_DIR: str = os.path.join(RUNS_DIR, "benchmark")  # results: <problem>/tl<T>/<label>/<cpu>/

# Genetic algorithm (values from the reference configs/general_configs.yaml)
POPULATION_FILE: Optional[str] = None  # None = start from scratch, else a saved population .json
POPULATION_SIZE: int = 8  # individuals per generation
NUM_GENERATIONS: int = 20  # generations after the first one (reference: num_steps_ga)
ELITISM_RATIO: float = 0.2  # share of the best individuals copied unchanged: int(0.2 * 8) = 1
CROSSOVER_RATE: float = 0.7  # probability that a child comes from a crossover, else a mutation
MAX_LLM_ATTEMPTS: int = 3  # LLM tries per child before falling back to a copy of a parent
# First generation: more tries per slot. Prompt benchmark of 2026-10-24: only 3-17% of the
# initializer's answers are accepted, so 3 tries often end with a copy of the seed.
MAX_INIT_ATTEMPTS: int = 15
GA_SEED: Optional[int] = None  # seed of the GA's random choices (None = different every run)
TEMPERATURE: Optional[float] = None  # LLM sampling temperature (None = the model's default)
# Gurobi seeds of the fitness during a run: a cut's fitness is the mean over them. One seed is too
# noisy (learnt_results.md 1.2: the seed cut scores 2.6-14.6 depending on the seed). 42 = the seed
# of the evaluation set's base results; disjoint from the review seeds 0..REVIEW_SEEDS-1, so that
# the review stays an independent check of the selected cut. Scoring time grows with their number.
FITNESS_SEEDS: List[int] = [42, 43, 44]
# How a cut is scored during the run (evolution.evaluator.FitnessMode): "gap" = MIP gap under the
# work limit, averaged over FITNESS_SEEDS; "lp" = % of the root gap closed by the LP relaxation
# (one deterministic LP per instance, FITNESS_SEEDS unused); "gap_work" = "gap" plus the work to
# prove optimality on the optimality set (OPTIMALITY_FILE).
FITNESS_MODE: str = "gap"
REVIEW_SEEDS: int = 5  # end of a run: best cut vs seed cut vs base under Gurobi seeds 0..4 (0 = off)
TOURNAMENT_SIZE: int = 2  # individuals compared to pick one parent (reference: 2)

# Solver (values from the reference configs/general_configs.yaml)
TIME_LIMIT: float = 60  # seconds per solve: a safety net, scoring is limited by WORK_LIMIT
WORK_LIMIT: float = 10  # Gurobi work units per scoring solve (~10 s of work on Gurobi's reference
#                         machine): same result on every run and machine, unlike a time limit
SEED: int = 42  # Gurobi Seed (used when SEED_MODE = "set")
SEED_MODE: str = "set"  # "set" = always SEED, "random" = new random seed for every solve
THREADS: int = 8  # paper: eight threads

# Solver safety (a formulation may contain code written by the LLM)
ISOLATED: bool = True  # build and solve every model in its own child process
BUILD_TIMEOUT: float = 60  # seconds allowed on top of TIME_LIMIT before the child is killed
# Cut solves (validity, usefulness, scoring) during a run: tighter than the safety values above.
# A normal scoring solve takes 4-11 s at WORK_LIMIT = 10 on the lab machines and builds in under
# 1 s. 20 s was too tight: on a loaded machine (dindon, 2026-10-27) the seed cut itself reached
# only 6.5 work units in 20 s and was rejected. 60 s keeps a 6x margin; a model that never
# finishes building is still stopped by CUT_BUILD_TIMEOUT.
CUT_TIME_LIMIT: float = 60  # seconds: hitting it rejects the cut ("too slow"), never scores it
CUT_BUILD_TIMEOUT: float = 10  # seconds to build the model on top of CUT_TIME_LIMIT, then killed
MAX_CONSTRAINTS: Optional[int] = 1_000_000  # bigger models are refused before solving (None = no cap)

# Proofs in Lean 4 (prove_cut.py, src/proving/): an LLM states a cut in Lean and proves it valid
LEAN_DIR: str = os.path.join(PROJECT_ROOT, "lean")  # the Lake project (Mathlib in lean/.lake)
PROOFS_DIR: str = os.path.join(RUNS_DIR, "proofs")  # one folder per proof of a cut
LEAN_TIMEOUT: float = 300  # seconds for one compile (a normal one takes 8-30 s on the Mac)
PROOF_STATEMENT_ROUNDS: int = 4  # LLM tries to write the cut in Lean so that it compiles
PROOF_ROUNDS: int = 8  # LLM tries to write a proof that compiles (each one sees the last errors)
PROOF_FEEDBACK_CHARS: int = 6000  # compiler messages shown to the LLM after a failed try
PROOF_HISTORY: int = 2  # earlier (try, errors) exchanges kept in the conversation of a phase
