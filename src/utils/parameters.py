import os

# Default values of the command-line parameters (see parser.py).
# Numeric values are left as None on purpose: to be decided later.

# Paths: absolute, computed from this file, so they do not depend on the working directory.
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")  # datasets, one subfolder per problem

# Gurobi license (secret, git-ignored). Gurobi reads GRB_LICENSE_FILE when it first starts,
# so this must run before any gurobipy model is created. setdefault: an already exported value wins.
GUROBI_LICENSE_FILE: str = os.path.join(PROJECT_ROOT, "gurobi.lic")
os.environ.setdefault("GRB_LICENSE_FILE", GUROBI_LICENSE_FILE)

MODEL_NAME = "deepseek-reasoner"  # LLM used to generate the cuts
POPULATION_FILE = None  # None = start from scratch, otherwise path to a saved population .json
PROBLEM_NAME = "tsp"  # problem to solve
CACHE_DIR: str = os.path.join(PROJECT_ROOT, "cache")  # where solver results are cached
TOP_SELECTED = 2  # number of individuals kept (elitism)
POPULATION_SIZE = 8  # number of individuals in the population

# Solver (values from the reference configs/general_configs.yaml)
TIME_LIMIT: float = 10  # seconds per solve
SEED: int = 42  # Gurobi Seed (used when SEED_MODE = "set")
SEED_MODE: str = "set"  # "set" = always SEED, "random" = new random seed for every solve
THREADS: int = 8  # paper: eight threads
LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")  # Gurobi log files, one per solve
