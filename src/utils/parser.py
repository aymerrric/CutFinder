import argparse

from utils import parameters


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="My EvoCut Argument Parser")
    parser.add_argument(
        "--model_name",
        type=str,
        default=parameters.MODEL_NAME,
        help="Name of the LLM used to generate the cuts",
    )
    parser.add_argument(
        "--population_file",
        type=str,
        default=parameters.POPULATION_FILE,
        help="Path to a saved population (.json) to resume from; if omitted, start from scratch",
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default=parameters.PROBLEM_NAME,
        help="The name of the problem to solve (e.g. tsp)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=parameters.CACHE_DIR,
        help="Directory where solver results are cached",
    )
    parser.add_argument(
        "--top_selected",
        type=int,
        default=parameters.TOP_SELECTED,
        help="Number of top individuals kept from one generation to the next",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=parameters.POPULATION_SIZE,
        help="Number of individuals in the population",
    )
    return parser.parse_args()
