import re
from typing import Any, List

import pytest

from evolution.genetic import CutRejected, GeneticAlgorithm
from evolution.individual import COPY, ELITE, INIT, SEED, Individual
from evolution.population import Population
from evolution.prompts import Prompts
from llm.base import Message
from llm.fake import FakeLLM

PROMPTS: Prompts = Prompts(model_code="def create_model(n, c): ...")


def reply(k: int) -> str:
    """A well-formed LLM answer whose cut has fitness k (see `evaluate`)."""
    return f"IDEA: idea {k}\nCODE:\n```python\nm.addConstr(cut_{k})\n```"


BAD_CUT: str = "IDEA: wrong\nCODE:\n```python\nm.addConstr(bad)\n```"


def evaluate(code: str) -> float:
    """Dummy evaluator: 'cut_7' has fitness 7.0, code containing 'bad' is rejected."""
    if "bad" in code:
        raise CutRejected("the cut removes the optimal solution")
    return float(re.search(r"cut_(\d+)", code).group(1))


def make_population(fitnesses: List[float]) -> Population:
    pop: Population = Population(len(fitnesses))
    pop.set_generation([Individual(idea=f"i{f}", code=f"cut_{f}", fitness=f) for f in fitnesses])
    return pop


def make_ga(pop: Population, replies: List[str], **kwargs: Any) -> GeneticAlgorithm:
    return GeneticAlgorithm(pop, FakeLLM(replies), PROMPTS, evaluate, seed=0, **kwargs)


def test_generation_step_keeps_the_size_and_copies_the_elite() -> None:
    pop: Population = make_population([1.0, 4.0, 2.0, 3.0])
    best: Individual = pop.best
    make_ga(pop, [reply(k) for k in range(10, 20)], elitism_ratio=0.25).generation_step()
    assert pop.generation == 1 and len(pop.current) == 4
    elite: Individual = pop.current[0]
    assert elite.generator == ELITE and elite.parents == [best.id] and elite.fitness == 4.0
    for child in pop.current[1:]:
        assert child.generator in list(PROMPTS.mutations) + list(PROMPTS.crossovers)
        assert child.fitness >= 10 and child.parents


@pytest.mark.parametrize("rate, n_parents", [(1.0, 2), (0.0, 1)])
def test_crossover_rate_chooses_the_operator(rate: float, n_parents: int) -> None:
    pop: Population = make_population([1.0, 2.0, 3.0, 4.0])
    make_ga(pop, [reply(k) for k in range(10, 20)], elitism_ratio=0, crossover_rate=rate).generation_step()
    operators: List[str] = list(PROMPTS.crossovers if n_parents == 2 else PROMPTS.mutations)
    for child in pop.current:
        assert child.generator in operators and len(child.parents) == n_parents


def test_bad_format_is_sent_back_and_retried() -> None:
    pop: Population = make_population([1.0, 2.0])
    ga: GeneticAlgorithm = make_ga(pop, ["Here is a cut: m.addConstr(x)", reply(9)])
    child: Individual = ga.mutation(pop.current[0], "General")
    assert child.fitness == 9.0 and child.idea == "idea 9" and child.generator == "General"
    second_try: List[Message] = ga.llm.received[1]
    assert [m["role"] for m in second_try] == ["system", "user", "assistant", "user"]
    assert "IDEA:" in second_try[-1]["content"]  # the format error is explained


def test_rejected_cut_reason_is_sent_back() -> None:
    pop: Population = make_population([1.0, 2.0])
    ga: GeneticAlgorithm = make_ga(pop, [BAD_CUT, reply(5)])
    child: Individual = ga.crossover(pop.current[0], pop.current[1], "Hybrid")
    assert child.fitness == 5.0 and child.parents == [pop.current[0].id, pop.current[1].id]
    assert "removes the optimal solution" in ga.llm.received[1][-1]["content"]


def test_parent_is_copied_when_every_attempt_fails() -> None:
    pop: Population = make_population([1.0, 2.0])
    ga: GeneticAlgorithm = make_ga(pop, [BAD_CUT] * 3, crossover_rate=0, max_llm_attempts=3)
    child: Individual = ga.make_child()
    assert len(ga.llm.received) == 3
    assert child.generator == COPY and len(child.parents) == 1
    parent: Individual = next(ind for ind in pop.current if ind.id == child.parents[0])
    assert child.fitness == parent.fitness and child.code == parent.code


def test_unreachable_llm_gives_up_at_once() -> None:
    pop: Population = make_population([1.0, 2.0])
    ga: GeneticAlgorithm = make_ga(pop, [], max_llm_attempts=3)  # FakeLLM raises: no reply
    assert ga.mutation(pop.current[0], "General") is None
    assert len(ga.llm.received) == 1


def test_best_fitness_is_shown_to_the_llm() -> None:
    pop: Population = make_population([1.0, 7.0])
    ga: GeneticAlgorithm = make_ga(pop, [reply(3)])
    ga.mutation(pop.current[0], "General")
    assert "Best fitness found so far: 7.0" in ga.llm.received[0][1]["content"]


def test_tournament_parents_are_distinct_and_the_fittest_win() -> None:
    pop: Population = make_population([1.0, 4.0, 2.0, 3.0])
    ga: GeneticAlgorithm = make_ga(pop, [], tournament_size=4)  # everyone competes
    assert [p.fitness for p in ga.select_parents(2)] == [4.0, 3.0]


def test_crossover_needs_two_individuals() -> None:
    pop: Population = make_population([1.0])
    ga: GeneticAlgorithm = make_ga(pop, [reply(2)], elitism_ratio=0, crossover_rate=1.0)
    ga.generation_step()  # falls back to a mutation instead of crashing
    assert pop.current[0].generator in PROMPTS.mutations


def test_same_seed_same_run() -> None:
    runs: List[List[str]] = []
    for _ in range(2):
        pop: Population = make_population([1.0, 2.0, 3.0, 4.0])
        make_ga(pop, [reply(k) for k in range(10, 20)]).generation_step()
        runs.append([ind.generator for ind in pop.current])
    assert runs[0] == runs[1]


def test_empty_population_raises() -> None:
    with pytest.raises(RuntimeError):
        make_ga(Population(2), []).generation_step()


@pytest.mark.parametrize("kwargs", [{"elitism_ratio": 1.5}, {"crossover_rate": -0.1}])
def test_ratios_must_be_in_0_1(kwargs: Any) -> None:
    with pytest.raises(ValueError):
        make_ga(make_population([1.0]), [], **kwargs)


# --- initialize ---

BASELINE: Individual = Individual(idea="2-cycle", code="m.addConstr(cut_6)")


def test_initialize_puts_the_evaluated_seed_first_then_new_cuts() -> None:
    pop: Population = Population(3)
    ga: GeneticAlgorithm = make_ga(pop, [reply(10), reply(11)])
    ga.initialize(BASELINE)
    seed: Individual = pop.current[0]
    assert seed.generator == SEED and seed.fitness == 6.0 and seed.code == BASELINE.code
    assert [ind.generator for ind in pop.current[1:]] == [INIT, INIT]
    assert [ind.fitness for ind in pop.current] == [6.0, 10.0, 11.0]
    assert all(ind.parents == [] for ind in pop.current)
    assert pop.generation == 0 and pop.best.fitness == 11.0


def test_initializer_sees_the_seed_and_the_growing_list_of_ideas() -> None:
    pop: Population = Population(3)
    ga: GeneticAlgorithm = make_ga(pop, [reply(10), reply(11)])
    ga.initialize(BASELINE)
    first, second = (conv[1]["content"] for conv in ga.llm.received)
    assert "m.addConstr(cut_6)" in first  # the seed's code, as a format example
    assert "- 2-cycle" in first and "- idea 10" not in first
    assert "- 2-cycle\n- idea 10" in second  # the previous answer is now an idea to avoid
    assert "Best fitness found so far: 10.0" in second


def test_failed_slot_gets_a_copy_of_the_seed() -> None:
    pop: Population = Population(2)
    ga: GeneticAlgorithm = make_ga(pop, [BAD_CUT] * 3, max_llm_attempts=3)
    ga.initialize(BASELINE)
    assert pop.current[1].generator == COPY and pop.current[1].parents == [pop.current[0].id]


def test_rejected_baseline_raises() -> None:
    with pytest.raises(ValueError, match="baseline"):
        make_ga(Population(2), []).initialize(Individual(idea="bad", code="m.addConstr(bad)"))


def test_initialize_twice_raises() -> None:
    pop: Population = make_population([1.0, 2.0])
    with pytest.raises(RuntimeError):
        make_ga(pop, []).initialize(BASELINE)


def test_progress_is_reported_before_each_child() -> None:
    calls: List[Any] = []
    pop: Population = Population(3)
    ga: GeneticAlgorithm = make_ga(
        pop, [reply(k) for k in range(10, 20)], elitism_ratio=0.34,
        progress=lambda generation, child: calls.append((generation, child)),
    )
    ga.initialize(BASELINE)
    ga.generation_step()
    assert calls == [(0, 2), (0, 3), (1, 2), (1, 3)]  # the seed and the elite are not children


def test_every_attempt_is_recorded() -> None:
    records: List[Any] = []
    pop: Population = make_population([1.0, 2.0])
    ga: GeneticAlgorithm = make_ga(pop, ["no format", BAD_CUT, reply(10), reply(11)], elitism_ratio=0,
                                   crossover_rate=0, max_llm_attempts=3, record=records.append)
    ga.generation_step()
    assert [(r["child"], r["attempt"], r["outcome"]) for r in records] == [
        (1, 1, "format_error"), (1, 2, "rejected"), (1, 3, "accepted"), (2, 1, "accepted"),
    ]
    first, rejected, accepted = records[0], records[1], records[2]
    assert first["generation"] == 1 and first["reply"] == "no format" and first["code"] is None
    assert rejected["code"] == "m.addConstr(bad)" and "optimal solution" in rejected["reason"]
    assert accepted["fitness"] == 10.0 and accepted["individual"] == pop.current[0].id
    assert len(accepted["messages"]) > len(first["messages"])  # the retries are in the conversation
    assert first["llm_seconds"] is not None and accepted["eval_seconds"] is not None
