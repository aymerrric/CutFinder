from pathlib import Path

from report_runs import find_runs, main, run_date
from test_report import make_run


def test_run_date() -> None:
    assert run_date("../runs/tsp_qwen3-coder-30b_saumon_20261027_104904") == "20261027"


def test_find_runs_keeps_run_folders_since_a_day(tmp_path: Path) -> None:
    make_run(tmp_path / "tsp_m_saumon_20261027_104904", "saumon")
    make_run(tmp_path / "tsp_m_thon_20261102_090000", "thon")
    (tmp_path / "analysis").mkdir()  # not a run: no status.json
    assert [Path(d).name for d in find_runs(str(tmp_path))] == [
        "tsp_m_saumon_20261027_104904", "tsp_m_thon_20261102_090000"]
    assert [Path(d).name for d in find_runs(str(tmp_path), "20261102")] == ["tsp_m_thon_20261102_090000"]


def test_main_writes_the_report(tmp_path: Path) -> None:
    make_run(tmp_path / "tsp_m_thon_20261102_090000", "thon")
    path: str = main(["--runs_dir", str(tmp_path), "--since", "20261101"])
    assert path == str(tmp_path / "report.md")
    text: str = Path(path).read_text()
    assert text.startswith("# Report over the runs since 20261101") and "**Best of all:**" in text
