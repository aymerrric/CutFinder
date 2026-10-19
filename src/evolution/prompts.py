import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import yaml

from evolution.individual import Individual
from llm.base import Message
from utils import parameters

# Reply format asked in common.yaml:  IDEA: <text>  CODE: ```python <code> ```
# Small models often write the labels in bold (**IDEA:**), hence the optional "*".
_IDEA_RE: "re.Pattern[str]" = re.compile(r"IDEA:\**\s*(.*?)\s*\**CODE:", re.DOTALL)
_CODE_BLOCK_RE: "re.Pattern[str]" = re.compile(
    r"CODE:\**\s*```[A-Za-z]*[ \t]*\n(.*?)```", re.DOTALL
)
_CODE_BARE_RE: "re.Pattern[str]" = re.compile(r"CODE:\**\s*(.+)", re.DOTALL)  # no ``` block


class ReplyFormatError(ValueError):
    """The LLM answer does not follow the IDEA: / CODE: format."""


def parse_reply(text: str) -> Tuple[str, str]:
    """(idea, code) from an LLM answer. Raises ReplyFormatError, with a message for the LLM."""
    idea_match: Optional[re.Match] = _IDEA_RE.search(text)
    if idea_match is None:
        raise ReplyFormatError("no 'IDEA:' section followed by a 'CODE:' section")
    code_match: Optional[re.Match] = _CODE_BLOCK_RE.search(
        text, idea_match.start()
    ) or _CODE_BARE_RE.search(text, idea_match.start())
    idea: str = idea_match.group(1).strip(" *\n")
    code: str = textwrap.dedent(code_match.group(1)).strip() if code_match else ""
    if not idea:
        raise ReplyFormatError("the 'IDEA:' section is empty")
    if not code:
        raise ReplyFormatError("the 'CODE:' section is empty")
    return idea, code


class Prompts:
    """
    The prompt templates of prompts/ (YAML), and the conversations built from them.
    `model_code` is the source of the problem's create_model, shown to the LLM.
    """

    def __init__(self, model_code: str, prompts_dir: str = parameters.PROMPTS_DIR) -> None:
        self.model_code: str = model_code
        self.common: Dict[str, str] = _load(prompts_dir, "common.yaml")
        self.mutations: Dict[str, str] = _load(prompts_dir, "mutations.yaml")  # operator -> task
        self.crossovers: Dict[str, str] = _load(prompts_dir, "crossovers.yaml")  # operator -> task
        self.initializer: Dict[str, str] = _load(prompts_dir, "initializer.yaml")  # task, user

    def initializer_messages(
        self, reference: Individual, ideas: List[str], best_fitness: float
    ) -> List[Message]:
        """
        A new cut from scratch: `reference` is shown as a format example only, and the new cut
        must be far from every idea of `ideas` (the ideas already in the generation).
        """
        user: str = self.initializer["user"].format(
            best_fitness=best_fitness,
            idea=reference.idea,
            fitness=reference.fitness,
            code=reference.code,
            ideas="\n".join(f"- {idea}" for idea in ideas),
        )
        return [self._system(self.initializer["task"]), {"role": "user", "content": user}]

    def mutation_messages(
        self, operator: str, parent: Individual, best_fitness: float
    ) -> List[Message]:
        user: str = self.common["mutation_user"].format(
            best_fitness=best_fitness, idea=parent.idea, fitness=parent.fitness, code=parent.code
        )
        return [self._system(self.mutations[operator]), {"role": "user", "content": user}]

    def crossover_messages(
        self, operator: str, parent1: Individual, parent2: Individual, best_fitness: float
    ) -> List[Message]:
        user: str = self.common["crossover_user"].format(
            best_fitness=best_fitness,
            idea1=parent1.idea, fitness1=parent1.fitness, code1=parent1.code,
            idea2=parent2.idea, fitness2=parent2.fitness, code2=parent2.code,
        )
        return [self._system(self.crossovers[operator]), {"role": "user", "content": user}]

    def retry_message(self, error: str) -> Message:
        """User message asking to fix a rejected answer (appended to the same conversation)."""
        return {"role": "user", "content": self.common["retry_user"].format(error=error)}

    def _system(self, task: str) -> Message:
        content: str = self.common["system"].format(task=task, model_code=self.model_code)
        return {"role": "system", "content": content}


def _load(prompts_dir: str, filename: str) -> Dict[str, str]:
    with open(os.path.join(prompts_dir, filename)) as f:
        return yaml.safe_load(f)
