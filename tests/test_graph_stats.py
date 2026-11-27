"""The statistics of graphs/make_graphs.py (Table 3 of test_results.tex): Welch's t-test and the
bootstrap interval of a difference of means, checked against scipy and a textbook example."""
import os
import sys

import pytest

stats = pytest.importorskip("scipy.stats")  # graphs/ runs on the Mac: the lab's .venv has neither
pytest.importorskip("matplotlib")           # scipy nor matplotlib, so these tests are skipped there

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graphs"))
from make_graphs import bootstrap_interval, welch  # noqa: E402

# The first example of the Wikipedia article "Welch's t-test" (15 values per sample): t = -2.46,
# about 25 degrees of freedom, p = 0.021.
A1 = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4]
A2 = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4]


def test_welch_matches_scipy_and_the_textbook_example() -> None:
    t, df, p = welch(A1, A2)
    reference = stats.ttest_ind(A1, A2, equal_var=False)
    assert t == pytest.approx(reference.statistic) and p == pytest.approx(reference.pvalue)
    assert df == pytest.approx(reference.df)
    assert round(t, 2) == -2.46 and round(df) == 25 and round(p, 3) == 0.021
    # our TSP case: Muse GA against Muse without the GA (3 runs each)
    ga, no_ga = [26.93, 26.48, 23.88], [22.27, 18.07, 15.49]
    assert welch(ga, no_ga)[2] == pytest.approx(stats.ttest_ind(ga, no_ga, equal_var=False).pvalue)


def test_welch_is_undefined_with_one_run_or_no_spread() -> None:
    assert welch([25.0, 26.0, 27.0], [15.0]) is None  # one run in an arm
    assert welch([10.1, 10.1], [10.1, 10.1]) is None  # no spread at all: t would be 0/0


def test_bootstrap_interval_of_a_difference_of_means() -> None:
    assert bootstrap_interval([3.0, 3.0], [1.0, 1.0, 1.0]) == (2.0, 2.0)  # nothing to resample
    low, high = bootstrap_interval(A1, A2)
    difference = sum(A1) / len(A1) - sum(A2) / len(A2)
    assert low < difference < high and high < 0  # Welch's p < 0.05: the interval excludes 0
    reference = stats.bootstrap((A1, A2), lambda a, b: a.mean() - b.mean(), method="percentile",
                                n_resamples=10_000, random_state=0, vectorized=False).confidence_interval
    assert low == pytest.approx(reference.low, abs=0.15) and high == pytest.approx(reference.high, abs=0.15)
    assert bootstrap_interval(A1, A2) == (low, high)  # fixed seed: the same table every time
