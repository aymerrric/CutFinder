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
