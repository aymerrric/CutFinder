import json
import os
from pathlib import Path

from evolution.individual import ELITE, SEED, Individual
from evolution.population import Population
from evolution.recorder import ACCEPTED, REJECTED, AttemptRecorder
from evolution.summary import reason_kind, summary_markdown, write_summary
from summarize_run import summarize_runs
from utils.run_files import POPULATION, REVIEW, STATUS


def make_run(folder: Path) -> Population:
    """A run folder: 2 generations, a status, 3 LLM proposals and a review."""
    seed: Individual = Individual(idea="2-cycle", code="m.addConstr(a)", fitness=10.0, generator=SEED)
    child: Individual = Individual(idea="3-cycle", code="m.addConstr(b)", fitness=14.0, generator="Hybrid")
    pop: Population = Population(2)
    pop.set_generation([seed, seed.copy("copy")])
    pop.set_generation([pop.top(1)[0].copy(ELITE), child])
    folder.mkdir()
    pop.save(str(folder / POPULATION))
    (folder / STATUS).write_text(json.dumps({
        "run": "tsp_m_h_1", "state": "finished", "host": "saumon", "cpu": "Xeon", "model": "qwen",
        "num_generations": 1, "started": 1000.0, "updated": 1600.0,
        "arguments": {"population_size": 2, "seed": None},
    }))
    record: AttemptRecorder = AttemptRecorder(str(folder / "llm"))
    record({"generation": 1, "child": 2, "attempt": 1, "operator": "Hybrid", "outcome": REJECTED,
            "reason": "invalid cut: instance rand_s1_0_15 becomes infeasible"})
    record({"generation": 1, "child": 2, "attempt": 2, "operator": "Hybrid", "outcome": ACCEPTED})
    record({"generation": 1, "child": 3, "attempt": 1, "operator": "Lifted", "outcome": "format_error",
            "reason": "no CODE: section"})
    (folder / REVIEW).write_text(json.dumps({"seeds": [0, 1], "models": [
        {"name": "base", "scores": [{"fitness": 10.0}, {"fitness": 10.0}], "fitness_mean": 10.0,
         "fitness_std": 0.0, "mean_gap": 0.1, "run_fitness": None},
        {"name": "seed", "scores": [{"fitness": 9.0}, {"fitness": 11.0}], "fitness_mean": 10.0,
         "fitness_std": 1.4, "mean_gap": 0.09, "run_fitness": 10.0},
        {"name": "best", "scores": [{"fitness": 12.0}, {"fitness": 13.0}], "fitness_mean": 12.5,
         "fitness_std": 0.7, "mean_gap": 0.06, "run_fitness": 14.0},
    ]}))
    return pop


def test_summary_has_every_section(tmp_path: Path) -> None:
    pop: Population = make_run(tmp_path / "run")
    text: str = summary_markdown(str(tmp_path / "run"), pop)
    assert text.startswith("# Run tsp_m_h_1")
    assert "**State:** finished, generation 1/1" in text and "(10 min)" in text
    assert "population_size = 2, seed = None" in text
    assert "**Fitness during the run:** 14.000" in text and "Found in generation** 1 by `Hybrid`" in text
    assert "```python\nm.addConstr(b)\n```" in text
    assert "beats the seed cut on **2/2 seeds** (+2.500 on average)" in text
    assert "| 1 | 14.000 | 14.000 | 12.000 |" in text  # generation 1: best ever, max, mean
    assert "| Hybrid | 2 | 1 | 50% | 0 | 1 | 0 |" in text and "| **all** | 3 | 1 | 33% |" in text
    assert "invalid cut (1)" in text


def test_summary_of_a_run_without_review_or_proposals(tmp_path: Path) -> None:
    folder: Path = tmp_path / "run"
    folder.mkdir()
    text: str = summary_markdown(str(folder), Population(2))
    assert "No generation finished yet." in text and "No LLM proposal recorded." in text


def test_reason_kind() -> None:
    assert reason_kind("invalid cut: instance x becomes infeasible") == "invalid cut"
    assert reason_kind("too slow: on instance x the solve hit") == "too slow"
    assert reason_kind(None) == "-"


def test_write_summary_and_the_script(tmp_path: Path) -> None:
    pop: Population = make_run(tmp_path / "run")
    path: str = write_summary(str(tmp_path / "run"), pop)
    assert path == str(tmp_path / "run" / "summary.md") and os.path.exists(path)
    os.remove(path)
    assert summarize_runs([str(tmp_path / "run")]) == [path] and os.path.exists(path)
