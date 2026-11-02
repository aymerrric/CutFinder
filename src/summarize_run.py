"""
(Re)write runs/<run>/summary.md from the run folder's files (main.py does it after every
generation; this is for old runs, or to refresh one).
Run from src/:  python summarize_run.py ../runs/<run> [../runs/<other run> ...]
"""
import os
import sys
from typing import List

from evolution.population import Population
from evolution.summary import write_summary
from utils.run_files import POPULATION


def summarize_runs(run_dirs: List[str]) -> List[str]:
    """The summary path of each run folder."""
    return [write_summary(d, Population.load(os.path.join(d, POPULATION))) for d in run_dirs]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in summarize_runs(sys.argv[1:]):
        print(f"wrote {path}")
