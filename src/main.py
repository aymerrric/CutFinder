"""
Run EvoCut: evolve cuts for a problem with a local LLM.
Run from src/ (on the machine that built the instance sets, with Ollama running):
    python main.py --model_name qwen3-coder:30b --population_size 8 --num_generations 20
Resume a run:  python main.py --model_name ... --population_file ../runs/<run>/population.json

Everything about a run goes in its own folder runs/<run>/ (layout: utils/run_files.py):
summary.md (read this first), population.json, status.json, run.log, review.json, and every LLM
proposal in llm/gen_<k>/. The Gurobi logs go in logs/<run>/.
<run> = <problem>_<model>_<host>_<date>; a resumed run keeps its folder.
"""
import argparse
import inspect
import json
import logging
import os
import signal
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from evolution.evaluator import CutEvaluator, FitnessMode
from evolution.genetic import GeneticAlgorithm
from evolution.individual import SEED, Individual
from evolution.population import Population
from evolution.prompts import Prompts
from evolution.recorder import AttemptRecorder
from evolution.sampling import Sampler
from evolution.review import CutReview, format_review, review_cuts
from evolution.summary import write_summary
from llm.base import LLMClient
from llm.factory import make_llm
from loaders.base import Instance, Loader
from loaders.sets import BASE
from problems import PROBLEMS, Problem, check_minimization, instance_set
from solvers.solver import SolveResult, Solver, machine_info
from utils import parameters
from utils.parser import arg_parser
from utils.log_archive import pack_logs
from utils.run_files import LLM_DIR, LOG, POPULATION, REVIEW, STATUS, run_name_of
from utils.status import FAILED, FINISHED, STOPPED, RunStatus

logger: logging.Logger = logging.getLogger("main")


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


def load_validity(problem: Problem, validity_dir: str = parameters.VALIDITY_DIR) -> List[Instance]:
    """The problem's validity instances (prepare_validity.py), or [] with a warning if not built."""
    loader: Loader = problem.loader(validity_dir)
    if not os.path.exists(os.path.join(loader.folder, parameters.VALIDITY_FILE)):
        logger.warning(f"no {os.path.join(loader.folder, parameters.VALIDITY_FILE)}: the validity check uses only "
                       "the validation instances (run prepare_validity.py)")
        return []
    return loader.load(parameters.VALIDITY_FILE)


def make_solver(args: argparse.Namespace, seed: int = parameters.SEED) -> Solver:
    """The solver of the cut evaluations: work limit, threads, tight cut limits, the run's logs."""
    log_dir: str = os.path.join(args.log_dir, args.run_name)  # logs/ is shared by every machine
    os.makedirs(log_dir, exist_ok=True)
    return Solver(
        work_limit=args.work_limit, threads=args.threads, log_dir=log_dir, seed=seed,
        time_limit=args.cut_time_limit, build_timeout=args.cut_build_timeout,
    )


def load_test_set(loader: Loader, problem: Problem, level: str = parameters.INSTANCE_LEVEL) -> List[Instance]:
    """The held-out instances of the end-of-run review at instance `level` (test.pkl or
    test_<level>.pkl, built by prepare_instances.py)."""
    filename: str = instance_set(problem, "test", level).filename
    if not os.path.exists(os.path.join(loader.folder, filename)):
        raise FileNotFoundError(
            f"{os.path.join(loader.folder, filename)} is missing (held-out instances of the final review): "
            f"run prepare_instances.py --instance_level {level}, it only builds the missing sets"
        )
    return loader.load(filename)


def load_optimality_set(loader: Loader, filename: str) -> List[Instance]:
    """An optimality set of the gap_work fitness (prepare_instances.py --optimality)."""
    if not os.path.exists(os.path.join(loader.folder, filename)):
        raise FileNotFoundError(
            f"{os.path.join(loader.folder, filename)} is missing (instances the base model solves, for the "
            "gap_work fitness): run prepare_instances.py --optimality"
        )
    return loader.load(filename)


def final_review(
    args: argparse.Namespace, problem: Problem, test: List[Instance], population: Population,
    run_dir: str, optimality: Optional[List[Instance]] = None, filename: str = REVIEW,
) -> List[CutReview]:
    """
    The best cut, the seed cut and the base model on the held-out `test` instances (never seen by
    the GA), under Gurobi seeds 0..review_seeds-1: does the best cut generalize, and is its
    fitness more than luck? `optimality`: held-out instances the base model solves (gap_work
    runs), for the work to prove optimality. Logged, and saved as <run_dir>/<filename>
    (review.json; review_run.py --review_level writes review_<level>.json).
    """
    best: Individual = population.best
    first: List[Individual] = population.history[0] if population.history else population.current
    seed_cut: Individual = next(ind for ind in first if ind.generator == SEED)
    cuts: Dict[str, str] = {"seed": seed_cut.code}
    fitness: Dict[str, float] = {"seed": seed_cut.fitness}
    if best.code != seed_cut.code:
        cuts["best"], fitness["best"] = best.code, best.fitness
    seeds: List[int] = list(range(args.review_seeds))
    test_file: str = instance_set(problem, "test", args.instance_level).filename
    logger.info(f"final review: {', '.join(cuts)} and the base model on the {len(test)} held-out "
                f"instances of {test_file}, under Gurobi seeds {seeds}")
    reviews: List[CutReview] = review_cuts(
        problem.base, cuts, test, seeds, lambda s: make_solver(args, seed=s), fitness, optimality
    )
    logger.info(format_review(reviews))
    path: str = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump({"seeds": seeds, "work_limit": args.work_limit, "threads": args.threads,
                   "instance_set": test_file, "instance_level": args.instance_level,
                   "instances": [inst["name"] for inst in test],
                   "optimality_instances": [inst["name"] for inst in optimality or []],
                   "fitness_mode": args.fitness_mode,  # the scale of the models' run_fitness
                   "models": [r.to_dict() for r in reviews]}, f, indent=1)
    logger.info(f"review saved to {path}")
    return reviews


def pack_run_logs(args: argparse.Namespace) -> None:
    """logs/<run>/ (one Gurobi log per solve) -> logs/<run>.tar.gz (utils/log_archive.py)."""
    try:
        archive: Optional[str] = pack_logs(os.path.join(args.log_dir, args.run_name))
        if archive:
            logger.info(f"solve logs packed into {archive}")
    except Exception as e:  # never hide the run's own end
        logger.warning(f"could not pack the solve logs: {type(e).__name__}: {e}")


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
    if getattr(args, "search", "ga") != "ga" and not args.llm_calls:
        raise ValueError(f"--search {args.search} needs --llm_calls (the LLM call budget)")
    loader: Loader = problem.loader(args.data_dir)
    evaluation_file: str = instance_set(problem, "evaluation", args.instance_level).filename
    for filename in (parameters.VALIDATION_FILE, evaluation_file):
        if not os.path.exists(os.path.join(loader.folder, filename)):
            raise FileNotFoundError(
                f"{os.path.join(loader.folder, filename)} is missing: run prepare_instances.py "
                f"--instance_level {args.instance_level} first"
            )
    validation: List[Instance] = loader.load(parameters.VALIDATION_FILE)
    evaluation: List[Instance] = loader.load(evaluation_file)
    check_same_machine(evaluation, args.threads)
    check_minimization(problem, validation[0])
    test: List[Instance] = (load_test_set(loader, problem, args.instance_level)  # fail now, not after the GA
                            if args.review_seeds > 0 else [])
    mode: FitnessMode = FitnessMode(args.fitness_mode)
    optimality: List[Instance] = []
    optimality_test: List[Instance] = []
    if mode is FitnessMode.GAP_WORK:
        optimality = load_optimality_set(loader, parameters.OPTIMALITY_FILE)
        check_same_machine(optimality, args.threads)  # its base work is compared with the cuts' work
        if args.review_seeds > 0:
            optimality_test = load_optimality_set(loader, parameters.OPTIMALITY_TEST_FILE)

    if llm is None:
        if args.model_name is None:
            raise ValueError("choose the LLM with --model_name (e.g. qwen3-coder:30b)")
        llm = make_llm(args.llm_backend, args.model_name, args.temperature, args.ollama_host)

    overlap: List[int] = sorted(set(args.fitness_seeds) & set(range(args.review_seeds)))
    if overlap and mode.solves_mip:
        logger.warning(f"fitness seeds {overlap} are also review seeds: the review is no longer independent")
    scoring: List[Solver] = [make_solver(args, seed=s) for s in args.fitness_seeds]  # one per seed
    if mode is FitnessMode.LP:
        scoring = scoring[:1]  # deterministic LP: one solver, no extra base solves
    validity: List[Instance] = load_validity(problem, args.validity_dir)
    logger.info(f"validity check: {len(validation)} validation + {len(validity)} validity instances")
    evaluator: CutEvaluator = CutEvaluator(problem.base, validation, evaluation, scoring[0], scoring, mode,
                                           validity=validity, optimality=optimality)
    prompts: Prompts = Prompts(inspect.getsource(problem.base), variant=args.prompt_variant,
                               fitness_mode=args.fitness_mode,  # the LLM is told how it is scored
                               problem=problem.name,  # and what the problem is
                               instance_level=args.instance_level,  # and the sizes it is scored on
                               hide_problem=args.hide_problem)  # or not even which problem it is

    run_dir: str = os.path.join(args.runs_dir, args.run_name)  # everything about this run
    os.makedirs(run_dir, exist_ok=True)
    path: str = args.population_file or os.path.join(run_dir, POPULATION)
    population: Population
    if args.population_file and os.path.exists(args.population_file):
        population = Population.load(args.population_file)
        logger.info(f"resumed {path}: generation {population.generation}")
    else:
        population = Population(args.population_size)

    options: Dict[str, Any] = dict(
        elitism_ratio=args.elitism_ratio,
        crossover_rate=args.crossover_rate,
        max_llm_attempts=args.max_llm_attempts,
        max_init_attempts=args.max_init_attempts,
        seed=args.seed,
        claim_of=evaluator.claim_of,
    )
    ga: GeneticAlgorithm
    if args.search == "ga":
        ga = GeneticAlgorithm(population, llm, prompts, evaluator, **options)
    else:  # no GA: the control (evolution/sampling.py)
        ga = Sampler(population, llm, prompts, evaluator, mode=args.search, budget=args.llm_calls,
                     calls_done=calls_spent(os.path.join(run_dir, LLM_DIR), population), **options)
    host, cpu = machine_info()
    status: RunStatus = RunStatus(
        os.path.join(run_dir, STATUS),
        run=args.run_name, host=host, cpu=cpu, pid=os.getpid(), model=args.model_name,
        arguments={k: v for k, v in vars(args).items() if k != "argv"},
        license_file=os.environ.get("GRB_LICENSE_FILE"),  # the machine's own license or the WLS one
        argv=getattr(args, "argv", []), population_file=os.path.abspath(path),
        size=population.size, num_generations=args.num_generations,
        generation=population.generation if population.current else None, building=None,
        child=None, best=population.best.fitness if population.best else None,
    )
    ga.progress = lambda generation, child: status.update(building=generation, child=child)
    ga.record = AttemptRecorder(os.path.join(run_dir, LLM_DIR))  # every LLM proposal
    start: float = time.time()

    def generation_done() -> None:
        population.save(path)
        logger.info(summarize(population, time.time() - start))
        fitness: List[float] = [ind.fitness for ind in population.current]
        status.update(generation=population.generation, building=None, child=None,
                      best=population.best.fitness, mean=statistics.mean(fitness),
                      **getattr(llm, "identity", dict)())  # Claude: the exact model id and the effort
        write_summary(run_dir, population)

    try:
        if isinstance(ga, Sampler):
            ga.run(Individual(idea=problem.baseline_idea, code=problem.baseline_code), generation_done)
        elif not population.current:
            ga.initialize(Individual(idea=problem.baseline_idea, code=problem.baseline_code))
            generation_done()
        while not isinstance(ga, Sampler) and population.generation < args.num_generations:
            ga.generation_step()
            generation_done()
        if args.review_seeds > 0:
            status.update(phase="review")
            final_review(args, problem, test, population, run_dir, optimality_test)
    except KeyboardInterrupt:  # Ctrl-C, or SIGTERM (see __main__)
        status.update(state=STOPPED)
        write_summary(run_dir, population)
        raise
    except Exception as e:
        status.update(state=FAILED, error=f"{type(e).__name__}: {e}")
        write_summary(run_dir, population)
        raise
    finally:  # finished, stopped or failed: the run's solve logs become one file
        pack_run_logs(args)
    status.update(state=FINISHED, phase=None)
    logger.info(f"summary: {write_summary(run_dir, population)}")

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


def stop_on_sigterm(signum: int, frame: Any) -> None:
    """SIGTERM (our stop command, an admin, a shutdown) -> KeyboardInterrupt: state "stopped"."""
    raise KeyboardInterrupt(f"signal {signum}")


def calls_spent(llm_dir: str, population: Population) -> int:
    """LLM calls of the generations already saved (--search sample | recall, resumed): the
    attempt records of gen_000 .. gen_<last saved>; a batch cut before it was saved is redone."""
    if not population.current:
        return 0
    return sum(len([f for f in os.listdir(os.path.join(llm_dir, folder)) if f.endswith(".json")])
               for folder in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, folder))
               and folder.startswith("gen_") and int(folder[4:]) <= population.generation) \
        if os.path.isdir(llm_dir) else 0


def make_run_name(args: argparse.Namespace, host: str) -> str:
    """A resumed run keeps its name (its folder); a new one gets problem, model, host, date (the
    model part also names a search without GA and a hidden problem: <model>-sample-hidden)."""
    if args.population_file:
        return run_name_of(args.population_file)
    model: str = (args.model_name or "none").replace(":", "-").replace("/", "-").replace("_", "-")
    if args.search != "ga":
        model += f"-{args.search}"
    if args.hide_problem:
        model += "-hidden"
    return f"{args.problem_name}_{model}_{host}_{time.strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":  # required: the Solver's child processes import this file
    arguments: argparse.Namespace = arg_parser()
    if not os.path.isdir(arguments.runs_dir):
        raise FileNotFoundError(f"{arguments.runs_dir} does not exist: create it (mkdir runs)")
    host, cpu = machine_info()
    arguments.run_name = make_run_name(arguments, host)
    arguments.argv = sys.argv[1:]
    os.makedirs(os.path.join(arguments.runs_dir, arguments.run_name), exist_ok=True)
    setup_logging(os.path.join(arguments.runs_dir, arguments.run_name, LOG))
    signal.signal(signal.SIGTERM, stop_on_sigterm)
    logger.info(f"run {arguments.run_name} on {host} ({cpu}), arguments: {vars(arguments)}")
    run(arguments)
