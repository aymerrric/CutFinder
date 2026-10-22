import os
from pathlib import Path

from evolution.recorder import ACCEPTED, AttemptRecorder, load_attempt


def test_one_file_per_attempt_in_its_generation_folder(tmp_path: Path) -> None:
    record: AttemptRecorder = AttemptRecorder(str(tmp_path / "run"))
    path: str = record({"generation": 3, "child": 2, "attempt": 1, "outcome": ACCEPTED})
    assert path == str(tmp_path / "run" / "gen_003" / "child_02_attempt_1.json")
    assert load_attempt(path)["outcome"] == ACCEPTED


def test_a_resumed_generation_does_not_overwrite(tmp_path: Path) -> None:
    record: AttemptRecorder = AttemptRecorder(str(tmp_path))
    first: str = record({"generation": 0, "child": 2, "attempt": 1, "reply": "a"})
    second: str = record({"generation": 0, "child": 2, "attempt": 1, "reply": "b"})
    assert first != second and second.endswith("child_02_attempt_1_resumed2.json")
    assert load_attempt(first)["reply"] == "a" and len(os.listdir(tmp_path / "gen_000")) == 2
