import os
from pathlib import Path

from utils.status import FINISHED, RUNNING, RunStatus, Status, read_status


def test_created_running_then_updated(tmp_path: Path) -> None:
    path: str = os.path.join(tmp_path, "run.status.json")
    status: RunStatus = RunStatus(path, run="r", host="h")
    first: Status = read_status(path)
    assert first["state"] == RUNNING and first["run"] == "r" and first["host"] == "h"
    assert first["updated"] >= first["started"]

    status.update(state=FINISHED, generation=3)
    second: Status = read_status(path)
    assert second["state"] == FINISHED and second["generation"] == 3 and second["run"] == "r"
    assert os.listdir(tmp_path) == ["run.status.json"]  # no temporary file left
