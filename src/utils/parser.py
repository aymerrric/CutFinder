import argparse
from typing import List, Optional

from utils import parameters


def arg_parser(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (`argv=None` reads sys.argv; tests pass their own list)."""
    parser = argparse.ArgumentParser(description="My EvoCut Argument Parser")
    parser.add_argument(
        "--model_name",
        type=str,
        default=parameters.MODEL_NAME,
        help="Name of the local LLM used to generate the cuts",
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default=parameters.PROBLEM_NAME,
        help="The name of the problem to solve (e.g. tsp)",
    )
    parser.add_argument(
        "--population_file",
        type=str,
        default=parameters.POPULATION_FILE,
        help="Path to a saved population (.json) to resume from; if omitted, start from scratch",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=parameters.POPULATION_SIZE,
        help="Number of individuals in each generation",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=parameters.NUM_GENERATIONS,
        help="Number of generations after the first one",
    )
    parser.add_argument(
        "--elitism_ratio",
        type=float,
        default=parameters.ELITISM_RATIO,
        help="Share of the best individuals copied unchanged into the next generation",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=parameters.CROSSOVER_RATE,
        help="Probability that a child comes from a crossover (otherwise from a mutation)",
    )
    parser.add_argument(
        "--max_llm_attempts",
        type=int,
        default=parameters.MAX_LLM_ATTEMPTS,
        help="LLM tries per child before copying a parent instead",
    )
    parser.add_argument(
        "--max_init_attempts",
        type=int,
        default=parameters.MAX_INIT_ATTEMPTS,
        help="LLM tries per slot of the first generation before copying the seed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=parameters.GA_SEED,
        help="Seed of the genetic algorithm's random choices (default: different every run)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=parameters.TEMPERATURE,
        help="LLM sampling temperature (default: the model's own)",
    )
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=["ollama", *parameters.API_PROVIDERS, "claude_code"],
        default=parameters.LLM_BACKEND,
        help="Where the LLM runs: the local Ollama server, an OpenAI-compatible API (key in llm_keys.env), "
             "or claude_code: Claude through `claude -p` (the Claude subscription; --model_name sonnet, opus, ...)",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=parameters.OLLAMA_HOST,
        help="Ollama server URL",
    )
    parser.add_argument(
        "--work_limit",
        type=float,
        default=parameters.WORK_LIMIT,
        help="Gurobi work limit per scoring solve (must match the evaluation set's)",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=["ga", "sample", "recall"],
        default="ga",
        help="ga: the genetic algorithm; sample: fresh conversations asking for the best cut (no parent, "
             "no fitness shown); recall: fresh conversations asking for the best cuts known in the "
             "literature (evolution/sampling.py). sample and recall need --llm_calls",
    )
    parser.add_argument(
        "--llm_calls",
        type=int,
        default=None,
        help="--search sample | recall: LLM calls in all (e.g. the number of calls of the GA run compared)",
    )
    parser.add_argument(
        "--hide_problem",
        action="store_true",
        help="Do not tell the LLM which problem it is (prompts/anonymous/<problem>.yaml)",
    )
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default=None,
        help="Folder of prompts/variants/ overriding part of the prompts (default: prompts/ only)",
    )
    parser.add_argument(
        "--fitness_seeds",
        type=int,
        nargs="+",
        default=parameters.FITNESS_SEEDS,
        help="Gurobi seeds of the fitness during the run (mean over them), e.g. --fitness_seeds 42 43 44",
    )
    parser.add_argument(
        "--fitness_mode",
        type=str,
        choices=["gap", "lp", "gap_work"],
        default=parameters.FITNESS_MODE,
        help="Fitness of the GA: 'gap' (MIP gap, mean over --fitness_seeds), 'lp' (%% of the root gap "
             "closed) or 'gap_work' (gap, plus the work to prove optimality on optimality.pkl)",
    )
    parser.add_argument(
        "--review_seeds",
        type=int,
        default=parameters.REVIEW_SEEDS,
        help="End of the run: review the best cut under Gurobi seeds 0..N-1 (0 = no review)",
    )
    parser.add_argument(
        "--cut_time_limit",
        type=float,
        default=parameters.CUT_TIME_LIMIT,
        help="Seconds per cut solve; a cut that hits it is rejected as too slow",
    )
    parser.add_argument(
        "--cut_build_timeout",
        type=float,
        default=parameters.CUT_BUILD_TIMEOUT,
        help="Seconds to build a cut's model on top of --cut_time_limit before it is killed",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=parameters.THREADS,
        help="Gurobi threads (must match the evaluation set's: work depends on it)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=parameters.DATA_DIR,
        help="Folder holding <problem>/validation.pkl and <problem>/evaluation.pkl",
    )
    parser.add_argument(
        "--instance_level",
        type=str,
        choices=parameters.INSTANCE_LEVELS,
        default=parameters.INSTANCE_LEVEL,
        help="Evaluation and test sets: 'default' (evaluation.pkl, test.pkl) or 'hard' (larger "
             "instances: evaluation_hard.pkl, test_hard.pkl; prepare_instances.py --instance_level hard)",
    )
    parser.add_argument(
        "--review_level",
        type=str,
        choices=parameters.INSTANCE_LEVELS,
        default=None,
        help="review_run.py only: review the run's best cut on the test set of this instance level "
             "instead of its own (written to review_<level>.json; review.json is kept)",
    )
    parser.add_argument(
        "--validity_dir",
        type=str,
        default=parameters.VALIDITY_DIR,
        help="Folder holding <problem>/validity.pkl (prepare_validity.py; the same for every CPU)",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=parameters.RUNS_DIR,
        help="Folder receiving <run>.json (population) and <run>.log (must exist)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=parameters.LOG_DIR,
        help="Folder receiving logs/<run>/, the Gurobi log of every solve of the run",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=parameters.CACHE_DIR,
        help="Directory where solver results are cached",
    )
    return parser.parse_args(argv)
