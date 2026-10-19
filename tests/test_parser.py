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
    assert args.data_dir == parameters.DATA_DIR and args.runs_dir == parameters.RUNS_DIR


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

