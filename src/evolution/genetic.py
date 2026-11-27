import logging
import random
import time
from typing import Callable, Dict, List, Optional

from evolution.individual import COPY, ELITE, INIT, SEED, Individual, normalized_code
from evolution.population import Population
from evolution.prompts import Prompts, ReplyFormatError, parse_reply
from evolution.recorder import ACCEPTED, FORMAT_ERROR, LLM_ERROR, REJECTED, Attempt
from llm.base import LLMClient, LLMError, Message
from utils import parameters

# Fitness of a cut, from its code. Raises CutRejected when the cut cannot be used.
# (Implemented by the cut-evaluation step: run the code, check validity, solve with Solver.)
Evaluator = Callable[[str], float]
ClaimOf = Callable[[str], Optional[str]]  # code of an accepted cut -> its claim (CutEvaluator.claim_of)
# Called before each child with (generation being built, child number from 1): progress reports.
Progress = Callable[[int, int], None]
# Called with every LLM attempt, accepted or not (see evolution/recorder.py).
Recorder = Callable[[Attempt], None]

logger: logging.Logger = logging.getLogger(__name__)


class CutRejected(Exception):
    """The cut cannot be used (code error, invalid cut, ...). The message is shown to the LLM.
    invalid: the cut removes optimal (or all) solutions; the LLM then gets the invalid-cut retry
    message (prompts/common.yaml retry_invalid_user) instead of the generic one."""

    def __init__(self, message: str, invalid: bool = False) -> None:
        super().__init__(message)
        self.invalid: bool = invalid


class GeneticAlgorithm:
    """
    Evolves the population one generation at a time. Each generation:
    1. elitism: the best `int(elitism_ratio * size)` individuals are copied unchanged;
    2. every other slot gets exactly one child: a crossover (probability `crossover_rate`) or a
       mutation, written by the LLM; if all `max_llm_attempts` fail, a copy of the parent.
    So a generation always has exactly `size` individuals.
    """

    def __init__(
        self,
        population: Population,
        llm: LLMClient,
        prompts: Prompts,
        evaluate: Evaluator,
        elitism_ratio: float = parameters.ELITISM_RATIO,
        crossover_rate: float = parameters.CROSSOVER_RATE,
        max_llm_attempts: int = parameters.MAX_LLM_ATTEMPTS,
        max_init_attempts: int = parameters.MAX_INIT_ATTEMPTS,
        tournament_size: int = parameters.TOURNAMENT_SIZE,
        seed: Optional[int] = None,
        progress: Optional[Progress] = None,
        record: Optional[Recorder] = None,
        claim_of: Optional[ClaimOf] = None,
    ) -> None:
        """claim_of: gives the claim of each accepted cut (CutEvaluator.claim_of), stored on the
        individual and in its attempt record; None = not recorded."""
        for name, value in [("elitism_ratio", elitism_ratio), ("crossover_rate", crossover_rate)]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.population: Population = population
        self.llm: LLMClient = llm
        self.prompts: Prompts = prompts
        self.evaluate: Evaluator = evaluate
        self.n_elites: int = int(elitism_ratio * population.size)
        self.crossover_rate: float = crossover_rate
        self.max_llm_attempts: int = max_llm_attempts
        self.max_init_attempts: int = max_init_attempts  # first generation: more tries per slot
        self.tournament_size: int = tournament_size
        self.rng: random.Random = random.Random(seed)  # local RNG: reproducible with `seed`
        self.progress: Optional[Progress] = progress
        self.record: Optional[Recorder] = record
        self.claim_of: Optional[ClaimOf] = claim_of
        self.llm_calls: int = 0  # LLM requests sent (each attempt is one)
        self.building: int = 0  # generation and child being built (for the records)
        self.child: int = 0

    def initialize(self, baseline: Individual) -> None:
        """
        First generation: the baseline cut, evaluated (generator SEED), then `size - 1` cuts from
        scratch by the initializer prompt (generator INIT). Each request shows the baseline as a
        format example and lists the ideas already in the generation, which the new cut must be
        far from. Each slot gets up to `max_init_attempts` answers (more than the other children:
        few initializer answers pass the checks), and an answer repeating a cut already in the
        generation is refused. A slot where every attempt fails gets a copy of the seed (COPY), so
        the generation is always full (the reference would loop until the LLM succeeds).
        """
        if self.population.current:
            raise RuntimeError("the population already has a generation")
        try:
            fitness: float = self.evaluate(baseline.code)
        except CutRejected as e:
            raise ValueError(f"the baseline cut is rejected: {e}") from e
        seed: Individual = Individual(
            idea=baseline.idea, code=baseline.code, fitness=fitness, generator=SEED,
            claim=self._claim(baseline.code),
        )
        generation: List[Individual] = [seed]
        while len(generation) < self.population.size:
            self._report(0, len(generation) + 1)
            ideas: List[str] = [ind.idea for ind in generation if not ind.is_copy]
            best: float = max(ind.fitness for ind in generation)
            messages: List[Message] = self.prompts.initializer_messages(seed, ideas, best)
            child: Optional[Individual] = self._ask_llm(
                messages, INIT, [], attempts=self.max_init_attempts, known=generation
            )
            if child is None:
                logger.warning("every LLM attempt failed: copying the seed")
                child = seed.copy(COPY)
            generation.append(child)
        self.population.set_generation(generation)

    def generation_step(self) -> None:
        """Build the next generation and make it the current one."""
        if not self.population.current:
            raise RuntimeError("the population is empty: set the first generation first")
        new_generation: List[Individual] = self.elitism()
        while len(new_generation) < self.population.size:
            self._report(self.population.generation + 1, len(new_generation) + 1)
            new_generation.append(self.make_child())
        self.population.set_generation(new_generation)

    def _report(self, generation: int, child: int) -> None:
        logger.info(f"generation {generation}: child {child}/{self.population.size}")
        self.building, self.child = generation, child
        if self.progress is not None:
            self.progress(generation, child)

    def elitism(self) -> List[Individual]:
        """Copies (new individuals, same fitness) of the best individuals of the generation."""
        return [ind.copy(ELITE) for ind in self.population.top(self.n_elites)]

    def make_child(self) -> Individual:
        """One child crossover or mutation, or a copy of the parent if the LLM fails."""
        parent: Individual
        child: Optional[Individual]
        if self.population.size >= 2 and self.rng.random() < self.crossover_rate:
            parent, other = self.select_parents(2)
            child = self.crossover(parent, other, self.rng.choice(list(self.prompts.crossovers)))
        else:
            (parent,) = self.select_parents(1)
            child = self.mutation(parent, self.rng.choice(list(self.prompts.mutations)))
        if child is None:
            logger.warning(f"every LLM attempt failed: copying parent {parent.id}")
            return parent.copy(COPY)
        return child

    def select_parents(self, k: int) -> List[Individual]:
        """
        Tournament selection, `k` distinct parents: each one is the fittest of
        `tournament_size` individuals drawn at random among those not yet selected.
        """
        available: List[Individual] = list(self.population.current)
        parents: List[Individual] = []
        for _ in range(k):
            n_contestants: int = min(self.tournament_size, len(available))
            contestants: List[Individual] = self.rng.sample(available, n_contestants)
            winner: Individual = max(contestants, key=lambda ind: ind.fitness)
            parents.append(winner)
            available.remove(winner)
        return parents

    def mutation(self, parent: Individual, operator: str) -> Optional[Individual]:
        """New cut from one parent with a mutation prompt (e.g. "Lifted"); None if the LLM fails."""
        messages: List[Message] = self.prompts.mutation_messages(
            operator, parent, self._best_fitness()
        )
        return self._ask_llm(messages, operator, [parent])

    def crossover(
        self, parent1: Individual, parent2: Individual, operator: str
    ) -> Optional[Individual]:
        """New cut from two parents with a crossover prompt (e.g. "Hybrid"); None if the LLM fails."""
        messages: List[Message] = self.prompts.crossover_messages(
            operator, parent1, parent2, self._best_fitness()
        )
        return self._ask_llm(messages, operator, [parent1, parent2])

    def _ask_llm(
        self, messages: List[Message], operator: str, parents: List[Individual],
        attempts: Optional[int] = None, known: Optional[List[Individual]] = None,
    ) -> Optional[Individual]:
        """
        Up to `attempts` answers (default `max_llm_attempts`) in one conversation: after a
        rejected answer (wrong format, a copy of a cut of `known`, or CutRejected from
        `evaluate`), the reason is sent back and the LLM tries again.
        """
        attempts = attempts or self.max_llm_attempts
        known_codes: Dict[str, Individual] = {normalized_code(ind.code): ind for ind in known or []}
        for attempt in range(1, attempts + 1):
            record: Attempt = {
                "generation": self.building, "child": self.child, "attempt": attempt,
                "operator": operator, "parents": [p.id for p in parents],
                "messages": [dict(message) for message in messages],  # the conversation sent
                "reply": None, "idea": None, "code": None, "outcome": None, "reason": None,
                "fitness": None, "individual": None, "claim": None, "llm_seconds": None, "eval_seconds": None,
                "time": time.time(),
            }
            start: float = time.time()
            self.llm_calls += 1
            try:
                reply: str = self.llm.send(messages)
            except LLMError as e:  # the LLM cannot be reached: asking again would not help
                logger.error(f"{operator}: {e}")
                self._record(record, outcome=LLM_ERROR, reason=str(e), llm_seconds=time.time() - start)
                return None
            record.update(reply=reply, llm_seconds=time.time() - start)
            messages.append({"role": "assistant", "content": reply})
            try:
                idea, code = parse_reply(reply)
            except ReplyFormatError as e:
                self._reject(record, attempt, operator, messages, FORMAT_ERROR, e, attempts)
                continue
            record.update(idea=idea, code=code)
            same: Optional[Individual] = known_codes.get(normalized_code(code))
            if same is not None:  # checked before evaluating: a copy costs no solve
                duplicate: CutRejected = CutRejected(
                    f"duplicate: this is the same cut as one already in the generation ({same.idea[:80]}); "
                    "propose a different cut"
                )
                self._reject(record, attempt, operator, messages, REJECTED, duplicate, attempts)
                continue
            start = time.time()
            try:
                fitness: float = self.evaluate(code)
            except CutRejected as e:
                record["eval_seconds"] = time.time() - start
                self._reject(record, attempt, operator, messages, REJECTED, e, attempts)
                continue
            child: Individual = Individual(
                idea=idea,
                code=code,
                fitness=fitness,
                generator=operator,
                parents=[p.id for p in parents],
                claim=self._claim(code),
            )
            self._record(record, outcome=ACCEPTED, fitness=fitness, individual=child.id,
                         eval_seconds=time.time() - start, claim=child.claim)
            return child
        return None

    def _reject(self, record: Attempt, attempt: int, operator: str, messages: List[Message],
                outcome: str, error: Exception, attempts: int) -> None:
        """Log and record a refused answer, and send the reason back to the LLM."""
        logger.info(f"{operator}: attempt {attempt}/{attempts} rejected: {error}")
        self._record(record, outcome=outcome, reason=str(error))
        messages.append(self.prompts.retry_message(str(error), invalid=getattr(error, "invalid", False)))

    def _record(self, record: Attempt, **fields: object) -> None:
        if self.record is not None:
            record.update(fields)
            self.record(record)

    def _claim(self, code: str) -> Optional[str]:
        """The claim of the accepted cut `code`, None if no claim_of was given."""
        return self.claim_of(code) if self.claim_of is not None else None

    def _best_fitness(self) -> float:
        assert self.population.best is not None  # set with the first generation
        return self.population.best.fitness
