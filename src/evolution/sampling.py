"""
Search without a genetic algorithm, the control of the GA (main.py --search sample | recall):
the same LLM, the same checks and the same fitness, but no parent, no selection and no fitness
shown to the LLM. It answers fresh conversations until `budget` LLM calls are spent (the number
of calls of the GA run it is compared with):
    sample: "propose the one cut you expect to make the solve fastest" (prompts/sampling.yaml),
            each conversation alone; with --hide_problem the problem is not named;
    recall: "give the best cut KNOWN in the literature, other than these", with the ideas already
            given, so the answers walk through what the LLM remembers.
As in the GA, a refused answer (format, code error, invalid, useless, slow, or a copy of a cut
already found) gets the reason and up to `max_llm_attempts` tries in the same conversation.
The accepted cuts are stored in batches of `population.size` as the generations of a Population
(the seed cut first, as in the GA), so population.json, the review and the benchmark work as for
a GA run; the last batch is completed with copies (is_copy, skipped by every analysis).
"""
import logging
from typing import Callable, List, Optional

from evolution.genetic import CutRejected, GeneticAlgorithm
from evolution.individual import COPY, SEED, Individual
from llm.base import Message

SAMPLE: str = "sample"  # --search values, also the generator of the cuts they produce
RECALL: str = "recall"
MODES: List[str] = [SAMPLE, RECALL]

logger: logging.Logger = logging.getLogger(__name__)


class Sampler(GeneticAlgorithm):
    """`mode`: SAMPLE or RECALL; `budget`: LLM calls in all; `calls_done`: calls of the batches
    already saved (a resumed run; main.py counts them in the run's llm/ records)."""

    def __init__(self, *args, mode: str = SAMPLE, budget: int = 0, calls_done: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mode not in MODES:
            raise ValueError(f"unknown search mode {mode!r} ({', '.join(MODES)})")
        if budget < 1:
            raise ValueError(f"the LLM call budget must be >= 1, got {budget}")
        self.mode: str = mode
        self.budget: int = budget
        self.llm_calls = calls_done

    def run(self, baseline: Individual, batch_done: Callable[[], None]) -> None:
        """Ask until the budget is spent; `batch_done` is called after each saved batch."""
        batch: List[Individual] = []
        if not self.population.current:
            try:
                fitness: float = self.evaluate(baseline.code)
            except CutRejected as e:
                raise ValueError(f"the baseline cut is rejected: {e}") from e
            batch.append(Individual(idea=baseline.idea, code=baseline.code, fitness=fitness, generator=SEED,
                                    claim=self._claim(baseline.code)))
        known: List[Individual] = self.population.all_individuals() + batch
        while self.llm_calls < self.budget:
            self._report(self.population.generation + 1 if self.population.current else 0, len(batch) + 1)
            child: Optional[Individual] = self._ask_llm(
                self.messages(known), self.mode, [],
                attempts=min(self.max_llm_attempts, self.budget - self.llm_calls), known=known)
            if child is not None:
                batch.append(child)
                known.append(child)
            if len(batch) == self.population.size:
                self._save(batch, batch_done)
                batch = []
        if batch:  # the budget ended inside a batch: complete it with copies
            best: Individual = max(batch, key=lambda ind: ind.fitness)
            self._save(batch + [best.copy(COPY) for _ in range(self.population.size - len(batch))], batch_done)
        logger.info(f"{self.mode}: {self.llm_calls} LLM calls spent, "
                    f"{len(self.population.all_individuals())} cuts (seed included)")

    def messages(self, known: List[Individual]) -> List[Message]:
        if self.mode == RECALL:
            return self.prompts.recall_messages([ind.idea for ind in known if ind.generator == RECALL])
        return self.prompts.sample_messages()

    def _save(self, batch: List[Individual], batch_done: Callable[[], None]) -> None:
        self.population.set_generation(batch)
        batch_done()
