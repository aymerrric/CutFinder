"""
The problems that have a Lean formulation (lean/EvoCut/Problems/<Name>.lean, written by hand from
the problem's create_model), and the assumptions each one makes on the data.

A proof holds for every instance that satisfies the assumptions of the Lean `Params` structure.
`assumptions` is the same list in Python: it says which ones an instance breaks, so that the
instance sets can be checked (prove_cut.py does it on the sets it finds on disk).
"""
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from utils import parameters

Args = Dict[str, Any]  # the arguments of a problem's create_model


@dataclass
class LeanProblem:
    name: str  # a key of problems.PROBLEMS
    module: str  # the Lean module of its formulation, e.g. "EvoCut.Problems.Tsp"
    namespace: str  # the namespace of its Params, Vars, Feasible, formulation
    assumptions: Callable[[Args], List[str]]  # instance -> the data assumptions it breaks

    def source(self, lean_dir: str = parameters.LEAN_DIR) -> str:
        """The text of the Lean formulation (shown to the LLM)."""
        with open(os.path.join(lean_dir, *self.module.split(".")) + ".lean") as f:
            return f.read()


def tsp_assumptions(args: Args) -> List[str]:
    broken: List[str] = []
    if args["n"] < 4:
        broken.append(f"hn: {args['n']} cities, at least 4 are assumed")
    if any(cost < 0 for cost in args["c"].values()):
        broken.append("hc_nn: a travel cost is negative")
    return broken


def cwlp_assumptions(args: Args) -> List[str]:
    broken: List[str] = []
    if not args["I"]:
        broken.append("hnI: no customer")
    if any(d <= 0 for d in args["d"].values()):
        broken.append("hd_pos: a demand is not positive")
    if any(u < 0 for u in args["u"].values()):
        broken.append("hu_nn: a capacity is negative")
    if any(f < 0 for f in args["f"].values()):
        broken.append("hf_nn: an opening cost is negative")
    if any(c < 0 for c in args["c"].values()):
        broken.append("hc_nn: an assignment cost is negative")
    return broken


def jssp_assumptions(args: Args) -> List[str]:
    broken: List[str] = []
    n_machines: int = args["n_machines"]
    if n_machines < 1:
        broken.append("hnM: no machine")
    if any(t <= 0 for row in args["times"] for t in row):
        broken.append("htimes_pos: a processing time is not positive")
    if any(sorted(row) != list(range(n_machines)) for row in args["machines"]):
        broken.append("hmachines_inj: a job does not visit every machine exactly once")
    return broken


LEAN_PROBLEMS: Dict[str, LeanProblem] = {
    "tsp": LeanProblem("tsp", "EvoCut.Problems.Tsp", "EvoCut.Tsp", tsp_assumptions),
    "cwlp": LeanProblem("cwlp", "EvoCut.Problems.Cwlp", "EvoCut.Cwlp", cwlp_assumptions),
    "jssp": LeanProblem("jssp", "EvoCut.Problems.Jssp", "EvoCut.Jssp", jssp_assumptions),
}


def broken_assumptions(problem: LeanProblem, instances: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Instance name -> the assumptions it breaks (only the instances that break one)."""
    report: Dict[str, List[str]] = {}
    for instance in instances:
        broken: List[str] = problem.assumptions(instance["args"])
        if broken:
            report[instance["name"]] = broken
    return report
