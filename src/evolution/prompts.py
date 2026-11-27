import ast
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import yaml

from evolution.individual import Individual
from llm.base import Message
from problems import PROBLEMS, Problem, instance_set
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


DEFAULT_VARIANT: str = "default"  # the prompts of prompts/ themselves
DEFAULT_FITNESS_MODE: str = "gap"  # a key of prompts/fitness.yaml (= evaluator.FitnessMode values)


def strip_comments(source: str) -> str:
    """`source` without its comments and docstrings (the code the LLM sees when the problem is hidden)."""
    tree: ast.Module = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


class Hider:
    """Replaces the words that name or describe the problem (prompts/anonymous/<problem>.yaml
    `words`): whole words, any case, longest first."""

    def __init__(self, words: Dict[str, str]) -> None:
        self.words: Dict[str, str] = {w.lower(): r for w, r in words.items()}
        names: List[str] = sorted(words, key=len, reverse=True)
        self.pattern: "re.Pattern[str]" = re.compile(
            r"(?<![A-Za-z])(" + "|".join(re.escape(w) for w in names) + r")(?![A-Za-z])", re.I)

    def __call__(self, text: str) -> str:
        return self.pattern.sub(lambda match: self.words[match.group(1).lower()], text)

    def left(self, text: str) -> List[str]:
        """The words of the list still in `text` (none after __call__, unless a replacement brings one)."""
        return sorted({match.group(1) for match in self.pattern.finditer(text)})


class Prompts:
    """
    The prompt templates of prompts/ (YAML), and the conversations built from them.
    `model_code` is the source of the problem's create_model, shown to the LLM.

    `variant`: a folder of prompts/variants/ that overrides part of the prompts, to compare
    prompt versions (prompt_bench.py) or to run with one (main.py --prompt_variant):
      - common.yaml, initializer.yaml, fitness.yaml: only the keys it contains are replaced
        (e.g. `system`);
      - mutations.yaml, crossovers.yaml: the whole file is replaced (its operators replace the
        default ones).
    None or "default" = the prompts of prompts/.

    `fitness_mode`: a key of fitness.yaml ("gap" or "lp", the values of evaluator.FitnessMode).
    Its text, inserted at {scoring} in the system prompt, tells the LLM exactly how its cuts are
    scored, so that it aims at the fitness the run uses.

    `problem`: a key of problems.PROBLEMS. problems/<problem>.yaml (overridden key by key by the
    variant's problems/<problem>.yaml) fills the problem's placeholders ({problem}, {description},
    {solution}, ...) of the system prompt and of the fitness text; {evaluation_size} comes from the
    problem's evaluation set (problems.py). A placeholder left without value is refused here.

    `hide_problem`: the LLM is not told which problem it is: prompts/anonymous/<problem>.yaml
    replaces {problem}, {solution} and {description}, the model is shown without comments or
    docstrings, and its `words` are replaced in every text (a prompt still holding one is refused).
    """

    def __init__(
        self,
        model_code: str,
        prompts_dir: str = parameters.PROMPTS_DIR,
        variant: Optional[str] = None,
        fitness_mode: str = DEFAULT_FITNESS_MODE,
        problem: str = parameters.PROBLEM_NAME,
        instance_level: str = parameters.INSTANCE_LEVEL,  # the sizes of the evaluation set the LLM is told
        hide_problem: bool = False,
    ) -> None:
        self.model_code: str = model_code
        self.variant: str = variant or DEFAULT_VARIANT
        overlay: Optional[str] = None
        if self.variant != DEFAULT_VARIANT:
            overlay = os.path.join(prompts_dir, "variants", self.variant)
            if not os.path.isdir(overlay):
                raise FileNotFoundError(f"prompt variant {self.variant!r}: {overlay} does not exist")
        self.common: Dict[str, str] = _load(prompts_dir, "common.yaml", overlay, merge=True)
        self.mutations: Dict[str, str] = _load(prompts_dir, "mutations.yaml", overlay)  # operator -> task
        self.crossovers: Dict[str, str] = _load(prompts_dir, "crossovers.yaml", overlay)  # operator -> task
        self.initializer: Dict[str, str] = _load(prompts_dir, "initializer.yaml", overlay, merge=True)
        self.sampling: Dict[str, str] = _load(prompts_dir, "sampling.yaml", overlay, merge=True)  # no-GA search
        scorings: Dict[str, str] = _load(prompts_dir, "fitness.yaml", overlay, merge=True)  # mode -> text
        if fitness_mode not in scorings:
            raise ValueError(f"fitness mode {fitness_mode!r} has no text in fitness.yaml ({', '.join(scorings)})")
        self.fitness_mode: str = fitness_mode
        if problem not in PROBLEMS:
            raise ValueError(f"unknown problem {problem!r} ({', '.join(PROBLEMS)})")
        spec: Problem = PROBLEMS[problem]
        self.problem: str = problem
        # placeholder -> text: the problem file, then the sizes of the instances the cuts are scored on
        self.problem_texts: Dict[str, str] = {
            **_load(prompts_dir, os.path.join("problems", f"{problem}.yaml"), overlay, merge=True),
            "evaluation_size": spec.describe_sizes(instance_set(spec, "evaluation", instance_level).sizes),
        }
        self.hider: Optional[Hider] = None
        if hide_problem:
            anonymous_file: str = os.path.join(prompts_dir, "anonymous", f"{problem}.yaml")
            if not os.path.exists(anonymous_file):
                raise FileNotFoundError(f"--hide_problem: {anonymous_file} does not exist (write it first)")
            with open(anonymous_file) as f:
                anonymous: Dict = yaml.safe_load(f)
            self.hider = Hider(anonymous.pop("words"))
            self.problem_texts = {key: self.hider(text) for key, text in {**self.problem_texts, **anonymous}.items()}
            self.model_code = self.hider(strip_comments(model_code))
        try:
            self.scoring: str = self.hide(scorings[fitness_mode].format(**self.problem_texts))
            # no GA: no best fitness is shown, so the bullet that says it is goes
            self.scoring_alone: str = re.sub(r"\n[ \t]*- Judge a cut.*?(?=\n[ \t]*- |\Z)", "", self.scoring, flags=re.S)
            self._system("task")  # every placeholder of the system prompt has a value
        except KeyError as e:
            raise ValueError(
                f"placeholder {e} has no value for problem {problem!r} with prompt variant {self.variant!r}: "
                f"add it to prompts/problems/{problem}.yaml (or to the variant's problems/{problem}.yaml)"
            ) from e

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

    def sample_messages(self) -> List[Message]:
        """A cut from scratch, without the GA: no parent, no fitness, no earlier cut (sampling.yaml)."""
        return [self._system(self.sampling["sample_task"], self.sampling["role"], self.scoring_alone),
                {"role": "user", "content": self.hide(self.sampling["sample_user"])}]

    def recall_messages(self, ideas: List[str]) -> List[Message]:
        """A cut known in the literature, other than `ideas` (the ones already given)."""
        user: str = self.sampling["recall_user"].format(ideas="\n".join(f"- {idea}" for idea in ideas) or "(none yet)")
        return [self._system(self.sampling["recall_task"], self.sampling["role"], self.scoring_alone),
                {"role": "user", "content": self.hide(user)}]

    def hide(self, text: str) -> str:
        """`text` with the problem's revealing words replaced (hide_problem), else unchanged."""
        return self.hider(text) if self.hider is not None else text

    def retry_message(self, error: str, invalid: bool = False) -> Message:
        """User message asking to fix a rejected answer (appended to the same conversation). For
        an invalid cut, retry_invalid_user: why it was refused and how such cuts usually go wrong."""
        key: str = "retry_invalid_user" if invalid and "retry_invalid_user" in self.common else "retry_user"
        return {"role": "user", "content": self.hide(self.common[key].format(error=error))}

    def _system(self, task: str, role: Optional[str] = None, scoring: Optional[str] = None) -> Message:
        """The system prompt of `task`; `role`: replaces common.yaml's role (who the LLM is),
        `scoring`: the scoring text (default: self.scoring)."""
        role_text: str = (role or self.common.get("role", "")).format(**self.problem_texts)
        content: str = self.hide(self.common["system"].format(
            task=task, role=role_text, model_code=self.model_code, scoring=scoring or self.scoring, **self.problem_texts
        ))
        if self.hider is not None and self.hider.left(content):
            raise ValueError(f"--hide_problem: the prompt still names the problem ({', '.join(self.hider.left(content))})")
        return {"role": "system", "content": content}


def _load(prompts_dir: str, filename: str, overlay: Optional[str] = None, merge: bool = False) -> Dict[str, str]:
    """prompts_dir/filename, overridden by overlay/filename if it exists (key by key if `merge`)."""
    with open(os.path.join(prompts_dir, filename)) as f:
        texts: Dict[str, str] = yaml.safe_load(f)
    if overlay is not None and os.path.exists(os.path.join(overlay, filename)):
        with open(os.path.join(overlay, filename)) as f:
            override: Dict[str, str] = yaml.safe_load(f) or {}
        texts = {**texts, **override} if merge else override
    return texts


def prompt_variants(prompts_dir: str = parameters.PROMPTS_DIR) -> List[str]:
    """"default" and the folders of prompts/variants/."""
    folder: str = os.path.join(prompts_dir, "variants")
    names: List[str] = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))) \
        if os.path.isdir(folder) else []
    return [DEFAULT_VARIANT] + names
