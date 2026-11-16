import argparse

from typing import Optional

from loaders.sets import BASE
from prepare_instances import has_gap, parse_args, proved_with_work
from solvers.solver import SolveResult
from utils import parameters


def result(gap: Optional[float]) -> SolveResult:
    return SolveResult("i", BASE, 0, "WORK_LIMIT", 1.0, 10.0, 5.0, 1, 3.0, 2.9, gap)


def test_defaults_come_from_parameters() -> None:
    args: argparse.Namespace = parse_args([])
    assert args.n_validation == parameters.N_VALIDATION
    assert args.problem_name == parameters.PROBLEM_NAME == "tsp"
    assert args.n_evaluation == parameters.N_EVALUATION
    assert args.min_base_gap == parameters.MIN_BASE_GAP
    assert args.n_test == parameters.N_TEST == 8
    # held out: generated from seeds the validation (1) and evaluation (100 + replacements) never use
    assert parameters.TEST_SEED > parameters.EVALUATION_SEED + 20
    assert args.data_dir == parameters.DATA_DIR


def test_flags_override() -> None:
    assert parse_args(["--n_test", "4"]).n_test == 4
    assert parse_args(["--data_dir", "../data/rtx3090"]).data_dir == "../data/rtx3090"


def test_has_gap_accepts_only_unsolved_instances() -> None:
    accept = has_gap(0.002)
    assert accept({BASE: result(0.05)}) and not accept({BASE: result(0.0)})
    assert not accept({BASE: result(None)})  # no solution found: no gap


def test_instance_level_flag() -> None:
    assert parse_args([]).instance_level == parameters.INSTANCE_LEVEL == "default"
    assert parse_args(["--instance_level", "hard"]).instance_level == "hard"


def test_optimality_instances_are_proved_with_a_measurable_work() -> None:
    accept = proved_with_work(parameters.OPTIMALITY_WORK)
    assert parameters.OPTIMALITY_WORK == (1.0, 5.0)
    solved = lambda work: SolveResult("i", BASE, 0, "OPTIMAL", 1.0, work, 5.0, 1, 3.0, 3.0, 0.0)
    assert accept({BASE: solved(2.5)})
    assert not accept({BASE: solved(0.2)})  # too little work: the change would be noise
    assert not accept({BASE: solved(8.0)})  # no room left under the work limit for a slower cut
    assert not accept({BASE: result(0.05)})  # stopped with a gap
    args: argparse.Namespace = parse_args(["--optimality"])
    assert args.optimality and args.n_optimality == parameters.N_OPTIMALITY and not parse_args([]).optimality
