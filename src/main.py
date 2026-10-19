"""
Run EvoCut: evolve cuts for a problem with a local LLM.
Run from src/ (on the machine that built the instance sets, with Ollama running):
    python main.py --model_name qwen3-coder:30b --population_size 8 --num_generations 20
Resume a run:  python main.py --model_name ... --population_file ../runs/<run>.json

Writes runs/<run>.json (the population, saved after every generation) and runs/<run>.log.
"""
import argparse
import inspect
import logging
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evolution.evaluator import CutEvaluator
from evolution.genetic import GeneticAlgorithm
from evolution.individual import Individual
from evolution.population import Population
from evolution.prompts import Prompts
from llm.base import LLMClient
from llm.ollama import OllamaClient
from loaders.base import Instance, Loader
from loaders.sets import BASE
from loaders.tsp import LoaderTSP
from models import tsp
from solvers.solver import Formulation, SolveResult, Solver, machine_info
from utils import parameters
from utils.parser import arg_parser

logger: logging.Logger = logging.getLogger("main")


@dataclass
class Problem:
    """Everything problem-specific that a run needs."""

    loader: Callable[[str], Loader]  # data_dir -> loader
    base: Formulation  # the model without any cut
    baseline_idea: str  # the seed of the first generation
    baseline_code: str


PROBLEMS: Dict[str, Problem] = {
    "tsp": Problem(LoaderTSP, tsp.create_model, tsp.BASELINE_IDEA, tsp.BASELINE_CODE),
}


def check_same_machine(evaluation: List[Instance], threads: int) -> None:
    """
    The evaluation set's base gaps are compared with the cuts' gaps at the same work limit. Work
    depends on the CPU type and the thread count, so both must match the current run.
    """
    host, cpu = machine_info()
    for inst in evaluation:
        base: SolveResult = inst[BASE]
        if base.cpu != cpu or base.threads != threads:
            raise RuntimeError(
                f"{inst['name']} was solved on {base.machine} ({base.cpu}, {base.threads} threads), "
                f"but this run is on {host} ({cpu}, {threads} threads): Gurobi work differs, so "
                "the gaps cannot be compared. Rebuild the sets here (prepare_instances.py) or run "
                "on the same machine type."
            )


def summarize(pop: Population, elapsed: float) -> str:
    """One line per generation: best (ever) and current fitness, where the children come from."""
    fitness: List[float] = [ind.fitness for ind in pop.current]
    generators: Counter = Counter(ind.generator for ind in pop.current)
    return (
        f"generation {pop.generation}: best ever {pop.best.fitness:.3f} | current max "
        f"{max(fitness):.3f}, mean {statistics.mean(fitness):.3f} | "
        f"{dict(generators)} | {elapsed / 60:.1f} min"
    )


def run(args: argparse.Namespace, llm: Optional[LLMClient] = None) -> Population:
    """
    Load the sets, build the GA, then: first generation (if the population is empty) and
    generation steps until `args.num_generations`, saving after each generation.
    `llm`: replaces the Ollama client (tests pass a FakeLLM).
    """
    problem: Problem = PROBLEMS[args.problem_name]
    loader: Loader = problem.loader(args.data_dir)
    for filename in (parameters.VALIDATION_FILE, parameters.EVALUATION_FILE):
        if not os.path.exists(os.path.join(loader.folder, filename)):
            raise FileNotFoundError(
                f"{os.path.join(loader.folder, filename)} is missing: run prepare_instances.py first"
            )
    validation: List[Instance] = loader.load(parameters.VALIDATION_FILE)
    evaluation: List[Instance] = loader.load(parameters.EVALUATION_FILE)
    check_same_machine(evaluation, args.threads)

    if llm is None:
        if args.model_name is None:
            raise ValueError("choose the LLM with --model_name (e.g. qwen3-coder:30b)")
        options: Dict[str, Any] = {} if args.temperature is None else {"temperature": args.temperature}
        ollama: OllamaClient = OllamaClient(args.model_name, host=args.ollama_host, options=options)
        if args.model_name not in ollama.available_models():
            raise ValueError(f"{args.model_name} is not on the Ollama server {args.ollama_host}")
        llm = ollama

    solver: Solver = Solver(work_limit=args.work_limit, threads=args.threads)
    evaluator: CutEvaluator = CutEvaluator(problem.base, validation, evaluation, solver)
    prompts: Prompts = Prompts(inspect.getsource(problem.base))

    path: str = args.population_file or os.path.join(args.runs_dir, f"{args.run_name}.json")
    population: Population
    if args.population_file and os.path.exists(args.population_file):
        population = Population.load(args.population_file)
        logger.info(f"resumed {path}: generation {population.generation}")
    else:
        population = Population(args.population_size)

    ga: GeneticAlgorithm = GeneticAlgorithm(
        population, llm, prompts, evaluator,
        elitism_ratio=args.elitism_ratio,
        crossover_rate=args.crossover_rate,
        max_llm_attempts=args.max_llm_attempts,
        seed=args.seed,
    )
    start: float = time.time()
    if not population.current:
        ga.initialize(Individual(idea=problem.baseline_idea, code=problem.baseline_code))
        population.save(path)
        logger.info(summarize(population, time.time() - start))
    while population.generation < args.num_generations:
        ga.generation_step()
        population.save(path)
        logger.info(summarize(population, time.time() - start))

    best: Individual = population.best
    logger.info(f"best cut ({best.generator}, fitness {best.fitness:.3f}): {best.idea}\n{best.code}")
    logger.info(f"population saved to {path}")
    return population


def setup_logging(log_file: str) -> None:
    """Console + log file. The solver's per-solve lines go to the file only."""
    formatter: logging.Formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console: logging.Handler = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(lambda record: record.name != "solvers.solver" or record.levelno >= logging.WARNING)
    file: logging.Handler = logging.FileHandler(log_file)
    file.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, file])


if __name__ == "__main__":  # required: the Solver's child processes import this file
    arguments: argparse.Namespace = arg_parser()
    if not os.path.isdir(arguments.runs_dir):
        raise FileNotFoundError(f"{arguments.runs_dir} does not exist: create it (mkdir runs)")
    model: str = (arguments.model_name or "none").replace(":", "-").replace("/", "-")
    arguments.run_name = f"{arguments.problem_name}_{model}_{time.strftime('%Y%m%d_%H%M%S')}"
    setup_logging(os.path.join(arguments.runs_dir, f"{arguments.run_name}.log"))
    host, cpu = machine_info()
    logger.info(f"run {arguments.run_name} on {host} ({cpu}), arguments: {vars(arguments)}")
    run(arguments)
