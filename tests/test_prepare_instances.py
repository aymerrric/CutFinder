import argparse

from typing import Optional

from loaders.sets import BASE
from prepare_instances import has_gap, parse_args
from solvers.solver import SolveResult
from utils import parameters


def result(gap: Optional[float]) -> SolveResult:
    return SolveResult("i", BASE, 0, "WORK_LIMIT", 1.0, 10.0, 5.0, 1, 3.0, 2.9, gap)


def test_defaults_come_from_parameters() -> None:
    args: argparse.Namespace = parse_args([])
    assert args.n_validation == parameters.N_VALIDATION
    assert args.validation_cities == parameters.VALIDATION_CITIES
    assert args.n_evaluation == parameters.N_EVALUATION
    assert args.evaluation_min_cities == parameters.EVALUATION_MIN_CITIES
    assert args.evaluation_max_cities == parameters.EVALUATION_MAX_CITIES
    assert args.min_base_gap == parameters.MIN_BASE_GAP
    assert args.data_dir == parameters.DATA_DIR


def test_flags_override() -> None:
    assert parse_args(["--validation_cities", "20"]).validation_cities == 20
    assert parse_args(["--data_dir", "../data/rtx3090"]).data_dir == "../data/rtx3090"


def test_has_gap_accepts_only_unsolved_instances() -> None:
    accept = has_gap(0.002)
    assert accept({BASE: result(0.05)}) and not accept({BASE: result(0.0)})
    assert not accept({BASE: result(None)})  # no solution found: no gap
