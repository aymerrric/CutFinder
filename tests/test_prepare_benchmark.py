import os
import shutil
from typing import List

from loaders import benchmarks
from loaders.base import Instance
from prepare_benchmark import model_size, parse_args, prepare, size_check
from problems import PROBLEMS

DATA: str = os.path.join(os.path.dirname(__file__), "data")


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.problem_name == "all" and not args.no_sizes


def test_size_check() -> None:
    assert size_check("variables", 100.2, 100.0).endswith("same)")
    assert size_check("variables", 110.0, 100.0).endswith("DIFFERENT)")


def test_prepare_saves_the_set(tmp_path, monkeypatch, capsys) -> None:
    raw = tmp_path / "raw" / "jssp"
    raw.mkdir(parents=True)
    shutil.copy(os.path.join(DATA, "taillard", "ta01"), raw / "ta01")  # no download
    monkeypatch.setattr(benchmarks, "TAILLARD_NAMES", ["ta01"])
    instances: List[Instance] = prepare("jssp", str(tmp_path))
    assert os.path.exists(tmp_path / "jssp" / "benchmark.pkl")
    assert PROBLEMS["jssp"].loader(str(tmp_path)).load("benchmark.pkl") == instances
    out: str = capsys.readouterr().out
    assert "WARNING: the paper has 80 instances" in out  # only 1 here
    # ta01 (15 x 15): 225 starts + Cmax + 15 * C(15, 2) order variables
    assert model_size(PROBLEMS["jssp"], instances[0])[0] == 225 + 1 + 15 * 105
