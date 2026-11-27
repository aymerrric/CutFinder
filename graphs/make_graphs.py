"""Three overviews (one panel per problem, one colour per LLM) from the GA runs in ../runs:
overview_mean_fitness.png (average fitness of the population, mean over runs),
overview_best_fitness.png (fitness of the population's best cut, mean over runs) and
overview_best_runs.png (for each LLM, the run with the best final result).
And, for GA against no GA: ga_vs_no_ga_calls.png (best training fitness found after X LLM calls,
mean over runs) and test_results.tex (the test scores as three booktabs tables for the paper: GA against the controls,
the GA per problem, and the GA-minus-control differences with a bootstrap interval and Welch's
t-test) and test_results.pdf (its preview).

    cd graphs && ../.venv/bin/python make_graphs.py

Only comparable GA runs are kept: 8 cuts x 20 generations, fitness "gap" on the training seeds
42 43 44, work limit 10, the default instances and prompts (runs started since the
instance_level option, plus INCLUDE), and runs whose children are not all copies (the Muse runs of
22 Nov that hit the spend limit). Runs still going count up to their last generation.
Also writes runs.csv (the runs used).
"""
import csv
import glob
import json
import os
import math
import random
import statistics
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402  (the t distribution of Welch's test)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

# Fixed colour per LLM (categorical order of the reference palette), the same in every graph.
LLMS = {
    "muse-spark-1.3-contributor": ("Muse", "#2a78d6"),
    "sonnet": ("Sonnet 5.5", "#eb6834"),
    "qwen3-coder:30b": ("qwen3-coder 30B", "#1baf7a"),
    "qwen2.5-coder:14b": ("qwen2.5-coder 14B", "#eda100"),
    "gemini-3.6-flash": ("Gemini 3.6 Flash", "#e87ba4"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
# Runs from before the instance_level option (11 Nov) with the same settings otherwise.
INCLUDE = {"tsp_muse-spark-1.3-contributor_bengali_20261111_110328"}  # the first Muse run, part of the replicates
PROBLEMS = {"tsp": "TSP", "cwlp": "CWLP", "jssp": "JSSP", "mcnd": "MCND", "corridor": "Corridor",
            "gisp": "GISP", "imo6": "IMO6"}


def comparable(a: Dict, run: str) -> bool:
    if run in INCLUDE:
        a = dict(a, instance_level="default")
    return (a.get("search") in (None, "ga") and a.get("fitness_mode") == "gap"
            and a.get("fitness_seeds") == [42, 43, 44] and a.get("population_size") == 8
            and a.get("num_generations") == 20 and a.get("work_limit") == 10
            and a.get("instance_level") == "default" and not a.get("prompt_variant")
            and not a.get("hide_problem") and a.get("model_name") in LLMS)


CONDITIONS = {  # search mode -> label and colour, the same in both GA-vs-no-GA outputs
    "ga": ("GA", "#2a78d6"),
    "sample": ("no GA (problem named)", "#eb6834"),
    "hidden": ("no GA (problem hidden)", "#1baf7a"),
    "recall": ("recall (textbook cuts asked)", "#eda100"),
}


SHORT = {"sample": "No GA", "hidden": "No GA, name hidden", "recall": "Recall"}  # table headers


def condition(a: Dict) -> Optional[str]:
    search = a.get("search") or "ga"
    if search == "sample":
        return "hidden" if a.get("hide_problem") else "sample"
    if search in ("ga", "recall") and not (search == "ga" and a.get("hide_problem")):
        return search
    return None


def comparable_any_search(a: Dict, run: str) -> bool:
    """The same settings as comparable(), with any search mode."""
    mode = condition(a)
    return mode is not None and comparable(dict(a, search="ga", hide_problem=False), run)


def fitness(individual: Dict) -> Optional[float]:
    value = individual.get("fitness")
    return None if value is None else float(value)


def test_score(run: str) -> Optional[float]:
    """Mean test score (5 seeds) of the run's winner, from review.json."""
    try:
        review = json.load(open(os.path.join(run, "review.json")))
    except (OSError, ValueError):
        return None
    for model in review.get("models", []):
        if model.get("name") == "best" and model.get("scores"):
            return statistics.mean(s["fitness"] for s in model["scores"])
    for model in review.get("models", []):  # no cut beat the seed: the seed is the winner
        if model.get("name") == "seed" and model.get("scores"):
            return statistics.mean(s["fitness"] for s in model["scores"])
    return None


def generations(run: str) -> List[List[Dict]]:
    """Every generation of a run: population.json keeps the last one in "current", not in "history"."""
    population = json.load(open(os.path.join(run, "population.json")))
    history = list(population["history"])
    current = population.get("current") or []
    if current and (not history or [i["id"] for i in history[-1]] != [i["id"] for i in current]):
        history.append(current)
    return history


def call_trace(run: str, seed: float) -> List[float]:
    """Best fitness found after each LLM call, in time order: one record per call in llm/ (rejected
    answers and retries included), the same for the GA and the searches without it."""
    calls = []
    for path in glob.glob(os.path.join(run, "llm", "*", "*.json")):
        try:
            record = json.load(open(path))
        except (OSError, ValueError):
            continue
        value = record.get("fitness")
        accepted = record.get("outcome") == "accepted" and value not in (None, "None")
        calls.append((float(record.get("time") or 0), float(value) if accepted else None))
    trace, best = [], seed
    for _, value in sorted(calls):
        if value is not None:
            best = max(best, value)
        trace.append(best)
    return trace


def load_conditions() -> List[Dict]:
    """Finished runs of every search mode, with the best fitness found after each LLM call."""
    runs = []
    for status_file in sorted(glob.glob(os.path.join(RUNS, "*", "status.json"))):
        run = os.path.dirname(status_file)
        if "invalid" in run:
            continue
        try:
            status = json.load(open(status_file))
            history = generations(run)
        except (OSError, ValueError, KeyError):
            continue
        arguments = status.get("arguments", {})
        if status.get("state") != "finished" or not history or not comparable_any_search(arguments, os.path.basename(run)):
            continue
        cuts = [i for generation in history for i in generation]
        children = [i for i in cuts if i.get("generator") not in ("seed", "elite")]
        if os.path.isdir(run + "_invalid_spend_limit"):
            continue  # a stale copy of a run that hit the Muse spend limit (renamed on the lab)
        seed = max(v for v in (fitness(i) for i in cuts if i.get("generator") == "seed") if v is not None)
        trace = call_trace(run, seed)
        if not trace:
            continue
        runs.append({"run": os.path.basename(run), "problem": arguments["problem_name"],
                     "llm": arguments["model_name"], "condition": condition(arguments),
                     "seed": seed, "trace": trace, "test": test_score(run), "seed_test": named_score(run, "seed")})
    return runs


def named_score(run: str, name: str) -> Optional[float]:
    try:
        review = json.load(open(os.path.join(run, "review.json")))
    except (OSError, ValueError):
        return None
    for model in review.get("models", []):
        if model.get("name") == name and model.get("scores"):
            return statistics.mean(s["fitness"] for s in model["scores"])
    return None


def at_calls(r: Dict, c: int) -> Optional[float]:
    """Best fitness found once c LLM calls are made (the seed cut at 0, None after the run's last call)."""
    if c > len(r["trace"]):
        return None
    return r["seed"] if c == 0 else r["trace"][c - 1]


def ga_vs_no_ga(runs: List[Dict]) -> None:
    with_ga = {(r["problem"], r["llm"]) for r in runs if r["condition"] == "ga"}
    panels = sorted({(r["problem"], r["llm"]) for r in runs if r["condition"] != "ga"} & with_ga,
                    key=lambda p: (list(PROBLEMS).index(p[0]), list(LLMS).index(p[1])))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False)
    for axis, (problem, llm) in zip(axes.flat, panels):
        labels = []
        for mode, (label, colour) in CONDITIONS.items():
            group = [r for r in runs if (r["problem"], r["llm"], r["condition"]) == (problem, llm, mode)]
            if not group:
                continue
            end = min(len(r["trace"]) for r in group)  # every run reaches it: the mean never mixes run sets
            xs, ys = [], []
            for c in range(0, end + 1):
                values = [at_calls(r, c) for r in group]
                if all(v is not None for v in values):
                    xs.append(c)
                    ys.append(statistics.mean(values))
            axis.plot(xs, ys, color=colour, linewidth=2, drawstyle="steps-post",
                      label=f"{label} ({len(group)} run{'s' * (len(group) > 1)})")
            labels.append((xs[-1] + end * 0.02 + 2, ys[-1], f"{ys[-1]:.1f}"))
        style(axis, f"{PROBLEMS[problem]}, {LLMS[llm][0]}", "best training fitness")
        axis.set_xlabel("LLM calls", color=MUTED, fontsize=9)
        axis.set_xticks(range(0, 301, 50))
        axis.set_xlim(0, max(max(len(r["trace"]) for r in runs if (r["problem"], r["llm"]) == (problem, llm)) * 1.15, 60))
        place_labels(axis, labels)
        axis.legend(frameon=False, fontsize=7, loc="best", labelcolor=INK)
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    figure.suptitle("Best training fitness found after X LLM calls, GA against no GA (mean over finished runs)",
                    x=0.01, ha="left", fontsize=12, color=INK)
    save(figure, "ga_vs_no_ga_calls.png")


BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 0  # fixed: the same tables at every run of the script


def welch(a: List[float], b: List[float]) -> Optional[Tuple[float, float, float]]:
    """Welch's t-test of mean(a) = mean(b), variances not assumed equal (Welch 1947): t, the
    Welch-Satterthwaite degrees of freedom, and the two-sided p-value. None when it is not
    defined: fewer than 2 runs in an arm, or no spread at all in either arm."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    if va + vb == 0:
        return None
    t = (statistics.mean(a) - statistics.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def bootstrap_interval(a: List[float], b: List[float], level: float = 0.95,
                       resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Percentile bootstrap interval of mean(a) - mean(b): the runs of each arm resampled with
    replacement, independently (Efron and Tibshirani 1993, chapter 13). With 2 to 5 runs per arm
    it is narrow: it cannot show more spread than the runs observed."""
    rng = random.Random(seed)
    differences = sorted(statistics.mean(rng.choices(a, k=len(a))) - statistics.mean(rng.choices(b, k=len(b)))
                         for _ in range(resamples))
    tail = (1 - level) / 2
    return differences[int(tail * resamples)], differences[min(resamples - 1, int((1 - tail) * resamples))]


def p_text(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def cell(scores: List[float], bold: bool = False) -> str:
    """mean +- standard deviation over runs, the number of runs in small type."""
    if not scores:
        return "--"
    mean = f"{statistics.mean(scores):.1f}"
    if bold:
        mean = f"\\mathbf{{{mean}}}"
    spread = f" \\pm {statistics.stdev(scores):.1f}" if len(scores) > 1 else ""
    return f"${mean}{spread}$\\,{{\\scriptsize({len(scores)})}}"


def test_table(runs: List[Dict]) -> None:
    """Three booktabs tables for the paper (test_results.tex) and a compiled preview (test_results.pdf)."""
    def scores(problem: str, llm: str, mode: str) -> List[float]:
        return [r["test"] for r in runs if (r["problem"], r["llm"], r["condition"]) == (problem, llm, mode)
                and r["test"] is not None]

    def seed(problem: str) -> str:
        values = [r["seed_test"] for r in runs if r["problem"] == problem and r["seed_test"] is not None]
        return f"{statistics.mean(values):.1f}" if values else "--"

    modes = list(CONDITIONS)
    # Table 1: GA against the controls, for the problems that have controls.
    rows, previous = [], None
    def has(problem: str, ga: bool) -> bool:
        return any(r["problem"] == problem and (r["condition"] == "ga") == ga and r["test"] is not None for r in runs)
    controlled = [p for p in PROBLEMS if has(p, True) and has(p, False)]  # GA and a control to compare
    for problem in controlled:
        llms = [l for l in LLMS if any(r["problem"] == problem and r["llm"] == l and r["test"] is not None for r in runs)]
        for k, llm in enumerate(llms):
            values = {m: scores(problem, llm, m) for m in modes}
            best = max((statistics.mean(v) for v in values.values() if v), default=None)
            cells = [cell(values[m], bool(values[m]) and statistics.mean(values[m]) == best) for m in modes]
            name = f"\\multirow{{{len(llms)}}}{{*}}{{{PROBLEMS[problem]}}}" if k == 0 else ""
            first = f"\\multirow{{{len(llms)}}}{{*}}{{{seed(problem)}}}" if k == 0 else ""
            rows.append(f"{name} & {first} & {LLMS[llm][0]} & " + " & ".join(cells) + " \\\\")
        if problem != controlled[-1]:
            rows.append("\\midrule")
    table1 = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Test score of the best cut found by each search, with the same LLM and the same number "
        "of LLM calls. GA: the genetic algorithm. No GA: independent requests for one good cut, with the "
        "problem named or hidden. Recall: requests for the best cuts known in the literature (24 calls). "
        "Seed cut: the hand-written cut every search starts from. "
        "Score on 5 unseen test seeds, higher is better; default Gurobi scores 10. "
        "Mean $\\pm$ standard deviation over runs, number of runs in brackets; best search in bold.}",
        "\\label{tab:ga-vs-controls}",
        "\\begin{tabular}{@{}l c l cccc@{}}", "\\toprule",
        "Problem & Seed cut & LLM & GA & No GA & No GA, name hidden & Recall \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])
    # Table 2: the GA on every problem, per LLM.
    llms = [l for l in LLMS if any(r["llm"] == l and r["condition"] == "ga" and r["test"] is not None for r in runs)]
    rows = []
    for problem in PROBLEMS:
        values = {l: scores(problem, l, "ga") for l in llms}
        if not any(values.values()):
            continue
        best = max(statistics.mean(v) for v in values.values() if v)
        rows.append(f"{PROBLEMS[problem]} & {seed(problem)} & " + " & ".join(
            cell(values[l], bool(values[l]) and statistics.mean(values[l]) == best) for l in llms) + " \\\\")
    table2 = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Test score of the best cut found by the GA, per problem and LLM (20 generations of 8 cuts). "
        "Seed cut: the hand-written cut the GA starts from. "
        "Score on 5 unseen test seeds, higher is better; default Gurobi scores 10. "
        "Mean $\\pm$ standard deviation over runs, number of runs in brackets; best LLM in bold.}",
        "\\label{tab:ga-per-problem}",
        "\\begin{tabular}{@{}l c " + "c" * len(llms) + "@{}}", "\\toprule",
        "Problem & Seed cut & " + " & ".join(LLMS[l][0] for l in llms) + " \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])
    # Table 3: GA minus each control, with its uncertainty.
    rows = []
    for problem in controlled:
        block = []
        for llm in LLMS:
            ga = scores(problem, llm, "ga")
            shown = False  # the LLM's name on its first row only
            for mode in modes[1:]:
                control = scores(problem, llm, mode)
                if not ga or not control:
                    continue
                low, high = bootstrap_interval(ga, control)
                test = welch(ga, control)
                block.append(f" & {'' if shown else LLMS[llm][0]} & {SHORT[mode]} & "
                             f"${statistics.mean(ga) - statistics.mean(control):+.1f}$ & "
                             f"$[{low:+.1f}, {high:+.1f}]$ & "
                             f"{p_text(test[2]) if test else '--'} & {len(ga)} / {len(control)} \\\\")
                shown = True
        if block:
            block[0] = f"\\multirow{{{len(block)}}}{{*}}{{{PROBLEMS[problem]}}}" + block[0]
            rows += block + ["\\midrule"]
    table3 = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Difference of mean test score, GA minus each control, for the same LLM "
        "(Table~\\ref{tab:ga-vs-controls}). Interval: 95\\% percentile bootstrap, the runs of each arm "
        f"resampled independently ({BOOTSTRAP_RESAMPLES:,} resamples); with 2 to 5 runs per arm it is "
        "optimistic. $p$: two-sided Welch's $t$-test (unequal variances); -- when an arm has fewer "
        "than 2 runs or no spread. Runs: GA / control.}",
        "\\label{tab:ga-minus-controls}",
        "\\begin{tabular}{@{}l l l c c c c@{}}", "\\toprule",
        "Problem & LLM & Control & GA $-$ control & 95\\% interval & $p$ & Runs \\\\", "\\midrule",
        *rows[:-1], "\\bottomrule", "\\end{tabular}", "\\end{table}"])
    body = ("% Written by graphs/make_graphs.py; needs \\usepackage{booktabs,multirow}.\n"
            + table1 + "\n\n" + table2 + "\n\n" + table3 + "\n")
    with open(os.path.join(HERE, "test_results.tex"), "w") as f:
        f.write(body)
    preview = ("\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n"
               "\\usepackage{booktabs,multirow}\n\\pagestyle{empty}\n\\begin{document}\n\\input{test_results.tex}\n\\end{document}\n")
    with open(os.path.join(HERE, "test_results_preview.tex"), "w") as f:
        f.write(preview)
    import subprocess
    for _ in range(2):  # twice: the second pass resolves \\ref
        done = subprocess.run(["pdflatex", "-interaction=nonstopmode", "test_results_preview.tex"], cwd=HERE,
                              capture_output=True, text=True)
    for ending in (".aux", ".log"):
        path = os.path.join(HERE, "test_results_preview" + ending)
        if os.path.exists(path):
            os.remove(path)
    if done.returncode == 0:
        os.replace(os.path.join(HERE, "test_results_preview.pdf"), os.path.join(HERE, "test_results.pdf"))
    else:
        print("pdflatex failed:", done.stdout[-1500:])
    os.remove(os.path.join(HERE, "test_results_preview.tex"))

def load() -> List[Dict]:
    runs = []
    for status_file in sorted(glob.glob(os.path.join(RUNS, "*", "status.json"))):
        run = os.path.dirname(status_file)
        if "invalid" in run:
            continue
        try:
            status = json.load(open(status_file))
            history = json.load(open(os.path.join(run, "population.json")))["history"]
        except (OSError, ValueError, KeyError):
            continue
        history = generations(run)
        arguments = status.get("arguments", {})
        if not comparable(arguments, os.path.basename(run)) or not history:
            continue
        children = [i for generation in history for i in generation if i.get("generator") != "seed"]
        if os.path.isdir(run + "_invalid_spend_limit"):
            continue  # a stale copy of a run that hit the Muse spend limit (renamed on the lab): not a real run
        mean, best = [], []
        for generation in history:
            values = [v for v in map(fitness, generation) if v is not None]
            mean.append(statistics.mean(values) if values else float("nan"))
            best.append(max(values) if values else float("nan"))
        runs.append({"run": os.path.basename(run), "problem": arguments["problem_name"],
                     "llm": arguments["model_name"], "state": status.get("state"),
                     "mean": mean, "best": best, "test": test_score(run)})
    return runs


def style(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    axis.set_xlabel("generation", color=MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.set_xticks(range(0, 20, 5))
    axis.set_xlim(0, 21.5)


def place_labels(axis, labels: List) -> None:
    """End-of-line values, pushed apart so that they never overlap."""
    low, high = axis.get_ylim()
    gap = (high - low) * 0.05
    placed: List[float] = []
    for x, y, text in sorted(labels, key=lambda label: label[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        axis.text(x, y, text, color=INK, fontsize=8, va="center")


def averaged(axis, runs: List[Dict], key: str) -> None:
    """One line per LLM: mean over its runs at each generation, band from the lowest to the highest run."""
    labels = []
    for llm, (label, colour) in LLMS.items():
        group = [r[key] for r in runs if r["llm"] == llm]
        if not group:
            continue
        length = max(len(g) for g in group)
        xs, ys, lows, highs = [], [], [], []
        for g in range(length):
            values = [s[g] for s in group if g < len(s) and s[g] == s[g]]
            if values:
                xs.append(g)
                ys.append(statistics.mean(values))
                lows.append(min(values))
                highs.append(max(values))
        if len(group) > 1:
            axis.fill_between(xs, lows, highs, color=colour, alpha=0.12, linewidth=0)
        axis.plot(xs, ys, color=colour, linewidth=2, label=f"{label} ({len(group)} run{'s' * (len(group) > 1)})")
        labels.append((xs[-1] + 0.3, ys[-1], f"{ys[-1]:.1f}"))
    place_labels(axis, labels)


def legend(axis) -> None:
    axis.legend(frameon=False, fontsize=8, loc="best", labelcolor=INK)


def save(figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(os.path.join(HERE, name), dpi=160, facecolor="white")
    plt.close(figure)


def overview(runs: List[Dict], problems: List[str]) -> None:
    for key, what in (("mean", "Average fitness of the population"), ("best", "Fitness of the best cut of the population")):
        columns = 3
        rows = (len(problems) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False)
        for axis, problem in zip(axes.flat, problems):
            averaged(axis, [r for r in runs if r["problem"] == problem], key)
            style(axis, PROBLEMS[problem], "fitness")
            legend(axis)
        for axis in list(axes.flat)[len(problems):]:
            axis.axis("off")
        figure.suptitle(f"{what} per generation, per LLM (mean over runs, band = lowest to highest run)",
                        x=0.01, ha="left", fontsize=12, color=INK)
        save(figure, f"overview_{key}_fitness.png")


def best_runs(runs: List[Dict], problems: List[str]) -> None:
    """For each problem and LLM, the run with the best final result (test score, else training)."""
    columns = 3
    rows = (len(problems) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False)
    for axis, problem in zip(axes.flat, problems):
        tests = []
        for llm, (label, colour) in LLMS.items():
            group = [r for r in runs if r["problem"] == problem and r["llm"] == llm]
            if not group:
                continue
            tested = [r for r in group if r["test"] is not None]
            top = max(tested, key=lambda r: r["test"]) if tested else max(group, key=lambda r: r["best"][-1])
            xs = list(range(len(top["best"])))
            axis.plot(xs, top["best"], color=colour, linewidth=2, label=f"{label}: best cut")
            axis.plot(xs, top["mean"], color=colour, linewidth=1, linestyle=(0, (3, 2)), alpha=0.8,
                      label=f"{label}: population average")
            if top["test"] is not None:
                axis.scatter([20.5], [top["test"]], s=40, color=colour, edgecolor="white", linewidth=2, zorder=3)
                tests.append((20.9, top["test"], f"test {top['test']:.1f}"))
        style(axis, PROBLEMS[problem], "fitness")
        axis.set_xlim(0, 24)
        place_labels(axis, tests)
        axis.legend(frameon=False, fontsize=6.5, loc="best", labelcolor=INK)
    for axis in list(axes.flat)[len(problems):]:
        axis.axis("off")
    figure.suptitle("The best run of each LLM (picked by its winner's test score, dot = test score on 5 seeds; "
                    "untested runs by training)", x=0.01, ha="left", fontsize=12, color=INK)
    save(figure, "overview_best_runs.png")


def main() -> None:
    runs = load()
    problems = [p for p in PROBLEMS if any(r["problem"] == p for r in runs)]
    overview(runs, problems)
    best_runs(runs, problems)
    everything = load_conditions()
    ga_vs_no_ga(everything)
    test_table(everything)
    with open(os.path.join(HERE, "runs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "llm", "run", "state", "generations", "best_training", "test"])
        for r in runs:
            writer.writerow([r["problem"], r["llm"], r["run"], r["state"], len(r["best"]),
                             round(r["best"][-1], 2), "" if r["test"] is None else round(r["test"], 2)])
    for r in runs:
        print(r["problem"], r["llm"], r["run"], len(r["best"]), round(r["best"][-1], 2), r["test"])


if __name__ == "__main__":
    main()
