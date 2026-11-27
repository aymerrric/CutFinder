import re
from typing import List

import pytest

from evolution.genetic import CutRejected
from evolution.individual import COPY, SEED, Individual
from evolution.population import Population
from evolution.prompts import Prompts
from evolution.sampling import RECALL, SAMPLE, Sampler
from llm.fake import FakeLLM

PROMPTS: Prompts = Prompts(model_code="def create_model(n, c): ...")
SEED_CUT: Individual = Individual(idea="seed idea", code="m.addConstr(cut_50)")


def reply(k: int) -> str:
    return f"IDEA: idea {k}\nCODE:\n```python\nm.addConstr(cut_{k})\n```"


def evaluate(code: str) -> float:
    if "bad" in code:
        raise CutRejected("the cut removes the optimal solution")
    return float(re.search(r"cut_(\d+)", code).group(1))


def make(replies: List[str], mode: str = SAMPLE, budget: int = 5, size: int = 3, **kwargs) -> Sampler:
    return Sampler(Population(size), FakeLLM(replies), PROMPTS, evaluate, mode=mode, budget=budget, seed=0, **kwargs)


def test_the_budget_of_llm_calls_is_spent_and_the_cuts_saved_in_batches() -> None:
    saved: List[int] = []
    sampler: Sampler = make([reply(k) for k in range(1, 6)], budget=5, size=3)
    sampler.run(SEED_CUT, lambda: saved.append(sampler.population.generation))
    pop: Population = sampler.population
    assert sampler.llm_calls == 5 and saved == [0, 1]  # seed + 2, then 3
    assert [ind.generator for ind in pop.history[0]] == [SEED, SAMPLE, SAMPLE]
    assert len(pop.all_individuals()) == 6 and pop.best.fitness == 50.0


def test_the_last_batch_is_completed_with_copies() -> None:
    sampler: Sampler = make([reply(k) for k in range(1, 4)], budget=3, size=3)
    sampler.run(SEED_CUT, lambda: None)
    assert [ind.generator for ind in sampler.population.current] == [SAMPLE, COPY, COPY]
    assert sampler.population.current[1].fitness == 3.0


def test_a_rejected_answer_is_retried_in_its_conversation_and_counted() -> None:
    llm: FakeLLM = FakeLLM(["no format", "IDEA: x\nCODE:\n```python\nm.addConstr(bad)\n```", reply(7), reply(8)])
    sampler: Sampler = Sampler(Population(2), llm, PROMPTS, evaluate, mode=SAMPLE, budget=4, seed=0)
    sampler.run(SEED_CUT, lambda: None)
    assert sampler.llm_calls == 4
    assert len(llm.received[2]) == 6  # system, user, then two refused answers and their reasons
    assert len(llm.received[3]) == 2  # a new conversation: nothing of the earlier ones
    assert sorted(ind.fitness for ind in sampler.population.all_individuals()) == [7.0, 8.0, 50.0]


def test_sample_prompts_show_no_parent_and_no_fitness() -> None:
    llm: FakeLLM = FakeLLM([reply(7), reply(8)])
    Sampler(Population(3), llm, PROMPTS, evaluate, mode=SAMPLE, budget=2).run(SEED_CUT, lambda: None)
    for conversation in llm.received:
        text: str = "\n".join(message["content"] for message in conversation)
        assert "seed idea" not in text and "cut_50" not in text and "idea 7" not in text
        assert "Best fitness" not in text and "best fitness found so far" not in text
        assert "genetic algorithm" not in text


def test_recall_lists_the_known_cuts_already_given() -> None:
    llm: FakeLLM = FakeLLM([reply(7), reply(8), reply(9)])
    Sampler(Population(4), llm, PROMPTS, evaluate, mode=RECALL, budget=3).run(SEED_CUT, lambda: None)
    assert "(none yet)" in llm.received[0][1]["content"] and "seed idea" not in llm.received[0][1]["content"]
    assert "- idea 7\n- idea 8" in llm.received[2][1]["content"]
    assert "KNOWN in the literature" in llm.received[0][0]["content"]


def test_a_copy_of_a_found_cut_is_refused() -> None:
    llm: FakeLLM = FakeLLM([reply(7), reply(7), reply(8)])
    sampler: Sampler = Sampler(Population(3), llm, PROMPTS, evaluate, mode=SAMPLE, budget=3)
    sampler.run(SEED_CUT, lambda: None)
    assert "duplicate" in llm.received[2][-1]["content"]
    assert sorted(ind.fitness for ind in sampler.population.all_individuals()) == [7.0, 8.0, 50.0]


def test_a_resumed_run_spends_only_what_is_left() -> None:
    first: Sampler = make([reply(k) for k in range(1, 3)], budget=2, size=3)
    first.run(SEED_CUT, lambda: None)
    second: Sampler = Sampler(first.population, FakeLLM([reply(k) for k in range(3, 9)]), PROMPTS, evaluate,
                              mode=SAMPLE, budget=5, calls_done=2)
    second.run(SEED_CUT, lambda: None)
    assert second.llm_calls == 5 and first.population.generation == 1  # no new seed: one more batch
    assert [ind.generator for ind in first.population.history[0]] == [SEED, SAMPLE, SAMPLE]


def test_a_budget_is_required() -> None:
    with pytest.raises(ValueError, match="budget"):
        make([], budget=0)
