"""
Layout of a run's folder (everything about one run in one place):

    runs/<run>/
        summary.md        readable summary: settings, best cut, fitness per generation, LLM
                          acceptance per operator, final review (rewritten after each generation)
        population.json   the population (every generation, best ever), to resume the run
        status.json       where the run is (read by computer_connection/fleet.py)
        run.log           the full log
        review.json       final review of the best cut under several Gurobi seeds
        llm/gen_<k>/      every LLM proposal of generation k, one JSON file per attempt

<run> = <problem>_<model>_<host>_<date>. The Gurobi log of every solve is in logs/<run>/.
"""
import os

SUMMARY: str = "summary.md"
POPULATION: str = "population.json"
STATUS: str = "status.json"
LOG: str = "run.log"
REVIEW: str = "review.json"
LLM_DIR: str = "llm"


def run_name_of(population_file: str) -> str:
    """The run a population file belongs to: runs/<run>/population.json -> <run>."""
    folder, filename = os.path.split(os.path.abspath(population_file))
    if filename == POPULATION:
        return os.path.basename(folder)
    return os.path.splitext(filename)[0]  # old layout: runs/<run>.json
