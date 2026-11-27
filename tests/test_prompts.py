"""
Checks the YAML files of prompts/ (they load, every template fills without error) and
evolution/prompts.py (conversations built from them, parsing of the LLM reply).
"""
import inspect
import os
import re
from typing import Dict, List

import pytest
import yaml

from evolution.evaluator import FitnessMode
from evolution.individual import Individual
from evolution.prompts import DEFAULT_VARIANT, Prompts, ReplyFormatError, parse_reply, prompt_variants
from llm.base import Message
from models.tsp import create_model
from utils import parameters


def load(filename: str) -> Dict[str, str]:
    with open(os.path.join(parameters.PROMPTS_DIR, filename)) as f:
        return yaml.safe_load(f)


COMMON: Dict[str, str] = load("common.yaml")
MUTATIONS: Dict[str, str] = load("mutations.yaml")
CROSSOVERS: Dict[str, str] = load("crossovers.yaml")
FITNESS: Dict[str, str] = load("fitness.yaml")
TSP: Dict[str, str] = {**load(os.path.join("problems", "tsp.yaml")), "evaluation_size": "60 to 100 cities"}
MODEL_CODE: str = inspect.getsource(create_model)


def test_common_has_every_template() -> None:
    assert set(COMMON) == {"role", "system", "mutation_user", "crossover_user", "retry_user", "retry_invalid_user"}


def test_operators_match_the_reference() -> None:
    assert set(MUTATIONS) == {"General", "Lifted", "Exploratory"}
    assert set(CROSSOVERS) == {"Intersection", "Complement", "Hybrid", "Min_Violation"}


@pytest.mark.parametrize("task", list(MUTATIONS.values()) + list(CROSSOVERS.values()))
def test_system_prompt_fills_for_every_operator(task: str) -> None:
    prompt: str = COMMON["system"].format(task=task, role=COMMON["role"].format(**TSP), model_code=MODEL_CODE,
                                          scoring=FITNESS["gap"].format(**TSP), **TSP)
    assert task in prompt and MODEL_CODE in prompt
    assert "{" not in prompt.replace(MODEL_CODE, "")  # no placeholder left unfilled
    assert "Pyomo" not in prompt


def test_operator_tasks_have_no_placeholder() -> None:
    # tasks are inserted as values, so a brace there would be shown to the LLM as is
    for task in list(MUTATIONS.values()) + list(CROSSOVERS.values()):
        assert "{" not in task and "}" not in task


def test_user_messages_fill() -> None:
    mutation: str = COMMON["mutation_user"].format(
        best_fitness=0.9, idea="I", fitness=0.5, code="m.addConstr(x)"
    )
    for text in ["0.9", "I", "0.5", "m.addConstr(x)"]:
        assert text in mutation
    crossover: str = COMMON["crossover_user"].format(
        best_fitness=0.9, idea1="I1", fitness1=1, code1="C1", idea2="I2", fitness2=2, code2="C2"
    )
    for text in ["0.9", "I1", "C1", "I2", "C2", "Parent 1", "Parent 2"]:
        assert text in crossover


def test_best_fitness_is_shown_in_every_user_message() -> None:
    assert "{best_fitness}" in COMMON["mutation_user"]
    assert "{best_fitness}" in COMMON["crossover_user"]


# --- evolution/prompts.py: conversations ---

PARENT1: Individual = Individual(idea="forbid 2-cycles", code="m.addConstr(A)", fitness=0.25)
PARENT2: Individual = Individual(idea="degree bound", code="m.addConstr(B)", fitness=0.5)


def test_mutation_messages() -> None:
    messages: List[Message] = Prompts(MODEL_CODE).mutation_messages("Lifted", PARENT1, 0.75)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert MUTATIONS["Lifted"] in messages[0]["content"]
    assert MODEL_CODE in messages[0]["content"]
    for text in ["forbid 2-cycles", "m.addConstr(A)", "0.25", "0.75"]:
        assert text in messages[1]["content"]


def test_crossover_messages() -> None:
    messages: List[Message] = Prompts(MODEL_CODE).crossover_messages("Hybrid", PARENT1, PARENT2, 0.75)
    assert CROSSOVERS["Hybrid"] in messages[0]["content"]
    for text in ["m.addConstr(A)", "m.addConstr(B)", "0.25", "0.5", "0.75"]:
        assert text in messages[1]["content"]


def test_retry_message_contains_the_error() -> None:
    message: Message = Prompts(MODEL_CODE).retry_message("KeyError: (1, 1)")
    assert message["role"] == "user" and "KeyError: (1, 1)" in message["content"]


def test_unknown_operator_raises() -> None:
    with pytest.raises(KeyError):
        Prompts(MODEL_CODE).mutation_messages("Nope", PARENT1, 1.0)


# --- evolution/prompts.py: parsing the reply ---

def test_parse_the_requested_format() -> None:
    reply: str = "IDEA: forbid 2-cycles.\nCODE:\n```python\nx = m._x\nm.addConstr(x[1, 2] <= 1)\n```\n"
    assert parse_reply(reply) == ("forbid 2-cycles.", "x = m._x\nm.addConstr(x[1, 2] <= 1)")


def test_parse_tolerates_bold_labels_text_around_and_indentation() -> None:
    reply: str = (
        "Sure! Here is my cut.\n**IDEA:** a multi-line\nidea\n\n**CODE:**\n"
        "```\n    m.addConstr(A)\n    m.addConstr(B)\n```\nHope it helps."
    )
    assert parse_reply(reply) == ("a multi-line\nidea", "m.addConstr(A)\nm.addConstr(B)")


def test_parse_code_without_backticks() -> None:
    assert parse_reply("IDEA: i\nCODE: m.addConstr(A)") == ("i", "m.addConstr(A)")


@pytest.mark.parametrize(
    "reply",
    [
        "here is a cut: m.addConstr(A)",  # no labels
        "CODE:\n```python\nm.addConstr(A)\n```",  # no IDEA
        "IDEA: i",  # no CODE
        "IDEA:\nCODE:\n```python\nm.addConstr(A)\n```",  # empty IDEA
        "IDEA: i\nCODE:\n```python\n```",  # empty CODE
    ],
)
def test_parse_refuses_bad_formats(reply: str) -> None:
    with pytest.raises(ReplyFormatError):
        parse_reply(reply)


# --- initializer ---

INITIALIZER: Dict[str, str] = load("initializer.yaml")


def test_initializer_file() -> None:
    assert set(INITIALIZER) == {"task", "user"}
    assert "{" not in INITIALIZER["task"] and "}" not in INITIALIZER["task"]  # inserted as a value
    assert "Pyomo" not in INITIALIZER["task"]


def test_initializer_messages() -> None:
    messages: List[Message] = Prompts(MODEL_CODE).initializer_messages(
        PARENT1, ["forbid 2-cycles", "degree bound"], 0.75
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert INITIALIZER["task"] in messages[0]["content"] and MODEL_CODE in messages[0]["content"]
    user: str = messages[1]["content"]
    for text in ["m.addConstr(A)", "0.25", "0.75", "- forbid 2-cycles\n- degree bound", "do NOT reuse"]:
        assert text in user


def test_fitness_file_has_one_text_per_fitness_mode() -> None:
    assert set(FITNESS) == {mode.value for mode in FitnessMode}  # every mode the GA can use
    assert "LP RELAXATION" in FITNESS["lp"] and "root gap" in FITNESS["lp"]
    assert "optimality gap" in FITNESS["gap"] and "seeds" in FITNESS["gap"]
    assert "PROVE optimality" in FITNESS["gap_work"] and "optimality gap" in FITNESS["gap_work"]


@pytest.mark.parametrize("mode", [m.value for m in FitnessMode])
@pytest.mark.parametrize("variant", prompt_variants())
def test_every_variant_tells_the_llm_its_fitness(variant: str, mode: str) -> None:
    prompts: Prompts = Prompts(MODEL_CODE, variant=variant, fitness_mode=mode)
    other: str = "lp" if mode != "lp" else "gap"
    for messages in [prompts.mutation_messages(op, PARENT1, 1.0) for op in prompts.mutations]:
        system: str = messages[0]["content"]
        assert prompts.scoring in system and "{scoring}" not in system
        assert Prompts(MODEL_CODE, variant=variant, fitness_mode=other).scoring not in system
    assert ("root gap" in prompts.scoring) == (mode == "lp")
    assert ("PROVE optimality" in prompts.scoring) == (mode == "gap_work")


def test_unknown_fitness_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="hybrid"):
        Prompts(MODEL_CODE, fitness_mode="hybrid")


@pytest.mark.parametrize("variant", prompt_variants())
def test_every_variant_fills_without_placeholder(variant: str) -> None:
    prompts: Prompts = Prompts(MODEL_CODE, variant=variant)
    ind: Individual = Individual(idea="an idea", code="m.addConstr(c)", fitness=3.0)
    conversations: List[List[Message]] = [prompts.initializer_messages(ind, ["- a"], 4.0)]
    conversations += [prompts.mutation_messages(op, ind, 4.0) for op in prompts.mutations]
    conversations += [prompts.crossover_messages(op, ind, ind, 4.0) for op in prompts.crossovers]
    for messages in conversations:
        system: str = messages[0]["content"]
        assert MODEL_CODE in system and "{task}" not in system and "{model_code}" not in system
        assert "{scoring}" not in system and prompts.scoring in system
        assert "IDEA:" in system and "CODE:" in system  # parse_reply's format is still asked
        assert "Best fitness found so far: 4.0" in messages[1]["content"]
    assert "the error" in prompts.retry_message("the error")["content"]


def test_variants_override_keys_or_whole_files(tmp_path: object) -> None:
    import shutil
    folder: str = os.path.join(str(tmp_path), "prompts")
    shutil.copytree(parameters.PROMPTS_DIR, folder, ignore=shutil.ignore_patterns("variants"))
    os.makedirs(os.path.join(folder, "variants", "v"))
    with open(os.path.join(folder, "variants", "v", "common.yaml"), "w") as f:
        f.write("retry_user: 'NEW {error}'\n")
    with open(os.path.join(folder, "variants", "v", "mutations.yaml"), "w") as f:
        f.write("Only: 'Task (only)'\n")
    default: Prompts = Prompts(MODEL_CODE, prompts_dir=folder)
    variant: Prompts = Prompts(MODEL_CODE, prompts_dir=folder, variant="v")
    assert variant.retry_message("e")["content"] == "NEW e"
    assert variant.common["system"] == default.common["system"]  # other keys kept (merge)
    assert list(variant.mutations) == ["Only"]  # whole file replaced
    assert variant.crossovers == default.crossovers  # file absent from the variant: default
    assert prompt_variants(folder) == [DEFAULT_VARIANT, "v"]
    with pytest.raises(FileNotFoundError, match="nope"):
        Prompts(MODEL_CODE, prompts_dir=folder, variant="nope")


def test_the_problem_is_described_from_its_own_file() -> None:
    system: str = Prompts(MODEL_CODE).mutation_messages("Lifted", PARENT1, 1.0)[0]["content"]
    assert "a MILP model of the Traveling Salesman Problem (TSP), given at the end" in system
    assert TSP["description"] in system and "60 to 100 cities" in system
    lp: Prompts = Prompts(MODEL_CODE, fitness_mode="lp")
    assert "best_known_tour" in lp.scoring and "LARGER instances (60 to 100 cities)" in lp.scoring
    assert "{" not in system.replace(MODEL_CODE, "")


def test_the_llm_is_told_the_sizes_of_its_instance_level() -> None:
    hard: str = Prompts(MODEL_CODE, instance_level="hard").mutation_messages("Lifted", PARENT1, 1.0)[0]["content"]
    assert "60 to 200 cities" in hard and "60 to 100 cities" not in hard


def test_templates_say_nothing_about_the_tsp() -> None:
    """Everything problem-specific is in prompts/problems/ (or a variant's problems/)."""
    for root, _, files in os.walk(parameters.PROMPTS_DIR):
        if os.path.basename(root) in ("problems", "anonymous"):  # anonymous/: the words to hide
            continue
        for name in files:
            if name.endswith(".yaml"):
                text: str = "\n".join(line for line in open(os.path.join(root, name)) if not line.lstrip().startswith("#"))
                for word in ("cities", "city", "tour", "TSP", "MTZ", "salesman"):
                    assert word not in text.replace("tourna", ""), f"{word!r} in {os.path.join(root, name)}"


def test_few_shot_takes_its_examples_from_the_problem() -> None:
    system: str = Prompts(MODEL_CODE, variant="few_shot").mutation_messages("Lifted", PARENT1, 1.0)[0]["content"]
    assert "Examples of answers that were REFUSED" in system and "two_cycle" in system


def test_a_placeholder_without_value_is_refused(tmp_path: object) -> None:
    import shutil
    folder: str = os.path.join(str(tmp_path), "prompts")
    shutil.copytree(parameters.PROMPTS_DIR, folder)
    with open(os.path.join(folder, "problems", "tsp.yaml"), "w") as f:
        f.write("problem: a problem\nsolution: tour\n")  # no description
    with pytest.raises(ValueError, match="'description' has no value for problem 'tsp'"):
        Prompts(MODEL_CODE, prompts_dir=folder)


def test_unknown_problem_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown problem 'nope'"):
        Prompts(MODEL_CODE, problem="nope")


@pytest.mark.parametrize("problem, name, size, solution", [
    ("cwlp", "Capacitated Warehouse Location Problem", "500 to 1000 customers and 80 to 150 warehouses", "solution"),
    ("jssp", "Job Shop Scheduling Problem", "15 to 20 jobs on 5 to 15 machines", "schedule"),
    ("mcnd", "Multi-Commodity capacitated fixed-charge Network Design problem",
     "20 nodes, 220 to 300 arcs and 100 to 200 commodities", "network design"),
    ("imo6", "a grid tiling problem", "N x N grids with N = 37 to 46", "tiling"),
    ("gisp", "Generalized Independent Set Problem", "150 to 175 nodes", "node set"),
    ("corridor", "wildlife corridor design problem", "square lattices of 20 x 20 to 25 x 25 cells", "corridor"),
])
def test_prompts_of_the_other_problems(problem: str, name: str, size: str, solution: str) -> None:
    from problems import PROBLEMS
    code: str = inspect.getsource(PROBLEMS[problem].base)
    for variant in prompt_variants():
        prompts: Prompts = Prompts(code, variant=variant, problem=problem)
        system: str = prompts.mutation_messages(next(iter(prompts.mutations)), PARENT1, 1.0)[0]["content"]
        assert name in system and size in system
        assert not re.search(r"\{(task|model_code|scoring|problem|description|solution|evaluation_size|examples)\}", system)
    assert f"best known {solution}" in Prompts(code, problem=problem, fitness_mode="lp").scoring


def test_the_imo6_prompts_do_not_give_the_olympiad_away() -> None:
    from problems import PROBLEMS
    code: str = inspect.getsource(PROBLEMS["imo6"].base)
    for variant in prompt_variants():
        prompts: Prompts = Prompts(code, variant=variant, problem="imo6")
        system: str = prompts.mutation_messages(next(iter(prompts.mutations)), PARENT1, 1.0)[0]["content"]
        assert "olymp" not in system.lower() and "IMO" not in system and "sqrt" not in system


def test_an_invalid_cut_gets_its_own_retry_message() -> None:
    prompts: Prompts = Prompts(MODEL_CODE)
    invalid: str = prompts.retry_message("removes the optimum of la05", invalid=True)["content"]
    generic: str = prompts.retry_message("removes the optimum of la05")["content"]
    assert "removes the optimum of la05" in invalid and "NOT VALID" in invalid and "prove" in invalid
    assert "NOT VALID" not in generic  # other refusals (code error, too slow...) keep the usual text


def test_a_hidden_problem_is_never_named() -> None:
    from evolution.prompts import strip_comments
    import inspect
    from models.tsp import create_model
    prompts: Prompts = Prompts(inspect.getsource(create_model), problem="tsp", hide_problem=True)
    conversations = [prompts.sample_messages(), prompts.recall_messages(["a tour idea"]),
                     prompts.mutation_messages("Lifted", PARENT1, 1.0)]
    for conversation in conversations:
        text: str = "\n".join(message["content"] for message in conversation).lower()
        for word in ("cities", "city", "tour", "tsp", "mtz", "salesman", "depot", "traveling"):
            assert re.search(rf"(?<![a-z]){word}(?![a-z])", text) is None, word
    assert "def create_model(n: int, c: Costs)" in prompts.model_code and '"""' not in prompts.model_code
    assert "nodes" in prompts.retry_message("removes every tour of 5 cities")["content"]
    assert "#" not in strip_comments("def f():\n    # a comment\n    return 1  # another\n")


def test_hiding_a_problem_needs_its_anonymous_file() -> None:
    with pytest.raises(FileNotFoundError, match="anonymous"):
        Prompts(MODEL_CODE, problem="jssp", hide_problem=True)
