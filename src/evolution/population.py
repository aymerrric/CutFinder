import json
from typing import Any, Dict, List, Optional

from evolution.individual import Individual


class Population:
    """
    The current generation of a fixed number of individuals, the previous generations and the
    best individual ever seen. Pure data: it never calls the LLM or the solver.

    Invariants, checked by `set_generation`:
    - every generation has exactly `size` individuals;
    - every individual is evaluated (fitness is not None), so they can be compared;
    - no id appears twice: elites and fallback copies are new objects (Individual.copy).
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size: int = size
        self.current: List[Individual] = []  # empty until the first generation is set
        self.history: List[List[Individual]] = []  # previous generations, oldest first
        self.best: Optional[Individual] = None  # best known, may come from an old generation

    @property
    def generation(self) -> int:
        """Index of the current generation (0 = the first one)."""
        return len(self.history)

    def set_generation(self, individuals: List[Individual]) -> None:
        """Make `individuals` the current generation; the previous one goes to `history`."""
        if len(individuals) != self.size:
            raise ValueError(f"a generation needs {self.size} individuals, got {len(individuals)}")
        not_evaluated: List[str] = [ind.id for ind in individuals if not ind.evaluated]
        if not_evaluated:
            raise ValueError(f"individuals without fitness: {not_evaluated}")
        ids: List[str] = [ind.id for ind in individuals]
        if len(set(ids)) != len(ids):
            raise ValueError("the same individual appears twice: use Individual.copy for elites")
        if self.current:  # nothing to archive before the first generation
            self.history.append(self.current)
        self.current = list(individuals)  # copy: later changes to the caller's list don't leak in
        for ind in self.current:
            if self.best is None or ind.fitness > self.best.fitness:  # strict: ties keep the old best
                self.best = ind

    def top(self, k: int) -> List[Individual]:
        """The `k` fittest individuals of the current generation, best first (used for elitism)."""
        return sorted(self.current, key=lambda ind: ind.fitness, reverse=True)[:k]

    def all_individuals(self) -> List[Individual]:
        """
        Every original individual of every generation, oldest first: copies (ELITE, COPY) are
        skipped, since they repeat a cut already listed. Later used to tell the LLM which
        ideas were already tried.
        """
        return [ind for gen in self.history + [self.current] for ind in gen if not ind.is_copy]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "current": [ind.to_dict() for ind in self.current],
            "history": [[ind.to_dict() for ind in gen] for gen in self.history],
            "best": self.best.to_dict() if self.best is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Population":
        pop: Population = cls(data["size"])
        pop.current = [Individual.from_dict(d) for d in data["current"]]
        pop.history = [[Individual.from_dict(d) for d in gen] for gen in data["history"]]
        pop.best = Individual.from_dict(data["best"]) if data["best"] is not None else None
        return pop

    def save(self, path: str) -> None:
        """Write the whole population (with history and best) to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Population":
        """Read back a population written by `save` (e.g. to resume an interrupted run)."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
