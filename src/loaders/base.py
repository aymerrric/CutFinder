import logging
import os
import pickle
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from utils import parameters

# One problem instance: {"name": str, "args": {...}, ...}
Instance = Dict[str, Any]


class Problem(Enum):
    TSP = "tsp"


class Loader(ABC):
    """Common interface to load or generate the instances of one problem."""

    problem: ClassVar[Problem]

    def __init__(self, data_dir: str = parameters.DATA_DIR) -> None:
        # every loader works in its own subfolder: data/<problem>/ (e.g. data/tsp/)
        self.folder: str = os.path.join(data_dir, self.problem.value)

    @abstractmethod
    def generate(
        self, n_instances: int, seed: Optional[int] = None, **kwargs: Any
    ) -> List[Instance]: ...

    def save(self, instances: List[Instance], filename: str) -> None:
        """Serialize the list of instances to `<folder>/<filename>` (pickle)."""
        with open(os.path.join(self.folder, filename), "wb") as f:
            pickle.dump(instances, f)

    def load(self, filename: str) -> List[Instance]:
        """Read back the list of instances saved by `save` in `<folder>/<filename>`."""
        with open(os.path.join(self.folder, filename), "rb") as f:
            instances: List[Instance] = pickle.load(f)
        return instances

    def get(
        self,
        filename: str,
        n_instances: Optional[int] = None,
        seed: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> List[Instance]:
        """
        Return the instances stored in `<folder>/<filename>` if the file exists.
        Otherwise generate `n_instances` (with `seed` and `gen_kwargs`, forwarded to `generate`),
        save them to that file, and return them.
        `n_instances`, `seed` and `gen_kwargs` are ignored when the file already exists.
        """
        path: str = os.path.join(self.folder, filename)
        if os.path.exists(path):
            logging.info(f"Loading instances from {path}")
            return self.load(filename)
        if n_instances is None:
            raise ValueError(f"{path} does not exist: n_instances is required to generate it")
        logging.info(f"{path} not found, generating {n_instances} instances")
        instances: List[Instance] = self.generate(n_instances, seed, **gen_kwargs)
        self.save(instances, filename)
        return instances

    def split(
        self, instances: List[Instance], ratio: float, seed: Optional[int] = None
    ) -> Tuple[List[Instance], List[Instance]]:
        """
        Shuffle a copy of `instances` and cut it in two disjoint parts.
        Returns (first, second) where `first` holds int(ratio * len(instances)) instances.
        """
        shuffled: List[Instance] = instances[:]  # copy: the input list is left untouched
        random.Random(seed).shuffle(shuffled)
        cut: int = int(ratio * len(shuffled))
        return shuffled[:cut], shuffled[cut:]
