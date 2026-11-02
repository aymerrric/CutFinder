import os
from typing import List, Optional

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

# Problem
PROBLEM_NAME: str = "tsp"

# Instance sets, each solved once with the base formulation (reference: 2 verification, 8 evaluation)
N_VALIDATION: int = 2  # solved to optimality: their optimal solutions check that a cut is valid
VALIDATION_CITIES: int = 15  # small: must be solved to optimality
VALIDATION_SEED: int = 1
VALIDATION_FILE: str = "validation.pkl"  # in data/<problem>/
VALIDATION_TIME_LIMIT: float = 600  # seconds: validation instances must reach OPTIMAL
N_EVALUATION: int = 8  # solved under WORK_LIMIT: used to score the cuts
EVALUATION_MIN_CITIES: int = 60  # large: the base model must NOT finish within WORK_LIMIT,
EVALUATION_MAX_CITIES: int = 100  # so that a cut can be scored by the gap it closes
MIN_BASE_GAP: float = 0.002  # evaluation instances with a smaller base gap are replaced
#                              (reference GAP_LOWERBOUND = 0.2, in percent)
EVALUATION_SEED: int = 100  # replacements use the next seeds (101, 102, ...): no clash with 1
EVALUATION_FILE: str = "evaluation.pkl"

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
