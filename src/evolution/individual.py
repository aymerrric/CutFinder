import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Generator name of the individuals of the first generation (made from scratch by the LLM)
INIT: str = "init"
# Generator name of the baseline cut put in the first generation (written by hand, evaluated)
SEED: str = "seed"
# Generator name of a parent copied unchanged because every LLM attempt failed for its slot
COPY: str = "copy"
# Generator name of an elite carried over unchanged to the next generation
ELITE: str = "elite"
# Copies bring no new idea: they are skipped when listing the ideas already tried
COPY_GENERATORS: Tuple[str, ...] = (COPY, ELITE)


@dataclass
class Individual:
    """
    One candidate cut of the population: what it means (`idea`), how to add it (`code`)
    and how much it helps the solver (`fitness`, higher is better).
    """

    idea: str  # natural-language description of the cut, written by the LLM
    code: str  # gurobipy code adding the cut to a model
    fitness: Optional[float] = None  # None = not evaluated yet
    generator: str = INIT  # what created it: INIT, SEED, ELITE, COPY, or a mutation / crossover
    parents: List[str] = field(default_factory=list)  # ids of the parents (empty for INIT)
    id: str = field(default_factory=lambda: uuid4().hex)  # unique, kept across save / load
    # what the validity check showed (evolution/evaluator.py: "keeps_optima" or "safe"); None =
    # not recorded (runs before 16 Nov 2026, or an evaluator that does not report it)
    claim: Optional[str] = None

    @property
    def evaluated(self) -> bool:
        return self.fitness is not None

    @property
    def is_copy(self) -> bool:
        return self.generator in COPY_GENERATORS

    def copy(self, generator: str) -> "Individual":
        """
        A new individual with the same cut and fitness (not re-evaluated), a new id, and this
        one as its only parent. Used for ELITE and COPY: a generation never holds the same
        object twice, and the history shows why the cut was carried over.
        """
        if generator not in COPY_GENERATORS:
            raise ValueError(f"generator must be one of {COPY_GENERATORS}, got {generator!r}")
        return Individual(
            idea=self.idea, code=self.code, fitness=self.fitness, generator=generator, parents=[self.id],
            claim=self.claim,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict (JSON-serializable) with every field."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        """Inverse of `to_dict`."""
        return cls(**data)


def normalized_code(code: str) -> str:
    """The code without comments and whitespace: two cuts with the same normalized code are the
    same cut (used to refuse copies)."""
    lines: List[str] = [re.sub(r"#.*", "", line) for line in code.splitlines()]
    return "".join("".join(lines).split())
