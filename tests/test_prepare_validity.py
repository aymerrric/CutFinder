import os
import shutil

import pytest

import prepare_validity
from loaders import validity
from loaders.sets import BASE
from problems import PROBLEMS
from utils import parameters

DATA: str = os.path.join(os.path.dirname(__file__), "data", "jsplib")


def test_prepare_solves_and_keeps_the_optimal_ones(tmp_path, monkeypatch, capsys) -> None:
    raw = tmp_path / "raw" / "jssp"
    raw.mkdir(parents=True)
    shutil.copy(os.path.join(DATA, "ft06"), raw / "ft06")
    monkeypatch.setattr(validity, "JSPLIB_OPTIMA", {"ft06": 55})
    first_two = validity.jssp_generated()[:2]
    monkeypatch.setattr(validity, "jssp_generated", lambda seed=0: first_two)
    kept = prepare_validity.prepare("jssp", str(tmp_path), time_limit=60)
    assert [i["name"] for i in kept] == ["tiny_3x3", "tiny_4x3", "ft06"]
    assert kept[-1][BASE].obj_val == pytest.approx(55) and kept[-1][BASE].solution
    saved = PROBLEMS["jssp"].loader(str(tmp_path)).load(parameters.VALIDITY_FILE)
    assert [i["name"] for i in saved] == ["tiny_3x3", "tiny_4x3", "ft06"]
    assert "3 validity instances saved" in capsys.readouterr().out


def test_a_wrong_published_optimum_stops(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "raw" / "jssp"
    raw.mkdir(parents=True)
    shutil.copy(os.path.join(DATA, "ft06"), raw / "ft06")
    monkeypatch.setattr(validity, "JSPLIB_OPTIMA", {"ft06": 54})
    monkeypatch.setattr(validity, "jssp_generated", lambda seed=0: [])
    with pytest.raises(RuntimeError, match="loader error"):
        prepare_validity.prepare("jssp", str(tmp_path), time_limit=60)


def test_infeasible_or_unsolved_candidates_are_left_out(tmp_path, monkeypatch, capsys) -> None:
    candidates = validity.cwlp_instances()[:1]  # tight_cap_0_15x5: total capacity 0.99 x demand
    monkeypatch.setattr(PROBLEMS["cwlp"], "validity_instances", lambda raw_dir: candidates)
    assert prepare_validity.prepare("cwlp", str(tmp_path), time_limit=60) == []
    assert "left out" in capsys.readouterr().out
