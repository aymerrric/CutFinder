"""
Write a report over several runs (evolution/report.py): per LLM and fitness mode, the best cut of
all and the mean of the best run of each computer. fleet.py watch rewrites it when runs end.
Run from src/:
    python report_runs.py                          # every run of ../runs -> ../runs/report.md
    python report_runs.py --since 20261102         # runs started on or after that day
    python report_runs.py ../runs/<run> ../runs/<other run> --output ../runs/report_x.md
"""
import argparse
import os
from typing import List, Optional

from evolution.report import write_report
from utils import parameters
from utils.run_files import STATUS

REPORT: str = "report.md"


def run_date(run_dir: str) -> str:
    """YYYYMMDD from the run name <problem>_<model>_<host>_<YYYYMMDD>_<HHMMSS>."""
    parts: List[str] = os.path.basename(os.path.normpath(run_dir)).split("_")
    return parts[-2] if len(parts) >= 2 else ""


def find_runs(runs_dir: str, since: Optional[str] = None) -> List[str]:
    """The run folders of `runs_dir` (those with a status.json), started on or after `since`."""
    folders: List[str] = sorted(
        os.path.join(runs_dir, name) for name in os.listdir(runs_dir)
        if os.path.exists(os.path.join(runs_dir, name, STATUS))
    )
    return [d for d in folders if since is None or run_date(d) >= since]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report over several runs, grouped by LLM")
    parser.add_argument("runs", nargs="*", help="run folders (default: every run of --runs_dir)")
    parser.add_argument("--runs_dir", type=str, default=parameters.RUNS_DIR)
    parser.add_argument("--since", type=str, default=None, help="only runs started on or after YYYYMMDD")
    parser.add_argument("--output", type=str, default=None, help=f"default: <runs_dir>/{REPORT}")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> str:
    args: argparse.Namespace = parse_args(argv)
    run_dirs: List[str] = args.runs or find_runs(args.runs_dir, args.since)
    title: str = "Report over the runs" + (f" since {args.since}" if args.since else "")
    return write_report(run_dirs, args.output or os.path.join(args.runs_dir, REPORT), title)


if __name__ == "__main__":
    print(f"wrote {main()}")
