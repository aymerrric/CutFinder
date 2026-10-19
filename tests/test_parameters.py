import os
from pathlib import Path

import pytest

from utils import parameters


def test_paths_are_absolute() -> None:
    for path in [parameters.PROJECT_ROOT, parameters.DATA_DIR, parameters.PROMPTS_DIR, parameters.LOG_DIR]:
        assert os.path.isabs(path)


def test_license_is_set_only_if_the_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
    missing: str = os.path.join(tmp_path, "missing.lic")
    parameters.use_license_file(missing)
    assert "GRB_LICENSE_FILE" not in os.environ
    present: str = os.path.join(tmp_path, "gurobi.lic")
    Path(present).touch()
    parameters.use_license_file(present)
    assert os.environ["GRB_LICENSE_FILE"] == present


def test_exported_license_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRB_LICENSE_FILE", "/my/own.lic")
    present: str = os.path.join(tmp_path, "gurobi.lic")
    Path(present).touch()
    parameters.use_license_file(present)
    assert os.environ["GRB_LICENSE_FILE"] == "/my/own.lic"
