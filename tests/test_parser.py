import argparse

import pytest

from utils import parameters
from utils.parser import arg_parser


def test_defaults_come_from_parameters() -> None:
    args: argparse.Namespace = arg_parser([])
    assert args.model_name == parameters.MODEL_NAME
    assert args.problem_name == parameters.PROBLEM_NAME
    assert args.population_file == parameters.POPULATION_FILE
    assert args.population_size == parameters.POPULATION_SIZE
    assert args.num_generations == parameters.NUM_GENERATIONS
    assert args.elitism_ratio == parameters.ELITISM_RATIO
    assert args.crossover_rate == parameters.CROSSOVER_RATE
    assert args.max_llm_attempts == parameters.MAX_LLM_ATTEMPTS
    assert args.cache_dir == parameters.CACHE_DIR
    assert args.seed == parameters.GA_SEED and args.temperature == parameters.TEMPERATURE
    assert args.ollama_host == parameters.OLLAMA_HOST
    assert args.work_limit == parameters.WORK_LIMIT and args.threads == parameters.THREADS
    assert args.cut_time_limit == parameters.CUT_TIME_LIMIT == 60
    assert args.max_init_attempts == parameters.MAX_INIT_ATTEMPTS == 15
    assert args.cut_build_timeout == parameters.CUT_BUILD_TIMEOUT == 10
    assert args.data_dir == parameters.DATA_DIR and args.runs_dir == parameters.RUNS_DIR
    assert args.fitness_seeds == parameters.FITNESS_SEEDS == [42, 43, 44]
    assert args.fitness_mode == parameters.FITNESS_MODE == "gap"
    assert args.llm_backend == parameters.LLM_BACKEND == "ollama"
    assert not set(args.fitness_seeds) & set(range(args.review_seeds))  # the review stays independent


def test_fitness_seeds_take_a_list() -> None:
    assert arg_parser(["--fitness_seeds", "7", "8"]).fitness_seeds == [7, 8]


def test_fitness_mode_is_gap_or_lp() -> None:
    assert arg_parser(["--fitness_mode", "lp"]).fitness_mode == "lp"
    with pytest.raises(SystemExit):
        arg_parser(["--fitness_mode", "hybrid"])


def test_flags_override_and_are_typed() -> None:
    args: argparse.Namespace = arg_parser(
        ["--model_name", "qwen", "--population_size", "4", "--elitism_ratio", "0.5"]
    )
    assert args.model_name == "qwen"
    assert args.population_size == 4
    assert args.elitism_ratio == 0.5


def test_bad_type_is_refused() -> None:
    with pytest.raises(SystemExit):  # argparse prints the error and exits
        arg_parser(["--population_size", "four"])

