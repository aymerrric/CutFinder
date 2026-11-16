"""
The validity instances of each problem: small instances, solved to optimality once
(prepare_validity.py), whose optimal solutions a cut must keep. They make the validity check much
harder to pass with an invalid cut than the 2 random validation instances (learnt_results.md 5.6):
  - random instances of several sizes, down to tiny ones (where "n - 3" style coefficients break);
  - TIGHT instances, built so that the classical bounds are tight: a cut that adds a margin or a
    constant to such a bound removes their optimum (the invalid CWLP and JSSP cuts of 2026-11 did);
  - public instances with a published optimum that are NOT in the paper's test sets (JSSP: JSPLIB,
    Lawrence, Fisher-Thompson, Adams-Balas-Zawack, Applegate-Cook). There is none for the CWLP:
    the 13 OR-Library instances outside the test set are infeasible with single sourcing.
Every instance has a "note" saying what it is, quoted to the LLM when a cut removes its optimum.
"""
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from loaders.base import Instance
from loaders.benchmarks import commalab_r_instances, download, parse_taillard
from loaders.corridor import LoaderCorridor, connectable, lattice
from loaders.cwlp import FIXED_COST_RANGE, UNIT_COST_RANGE
from loaders.gisp import EDGE_COST, NODE_WEIGHT, LoaderGISP
from loaders.imo6 import instance as imo6_instance
from loaders.jssp import TIME_RANGE
from loaders.mcnd import LoaderMCND
from loaders.tsp import LoaderTSP

VALIDITY_SEED: int = 7
JSPLIB_URL: str = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances/{name}"
# public JSSP instances with a published optimum, small enough to be solved here (JSPLIB's
# instances.json); none of them is in the paper's test set (Taillard ta01-ta80)
JSPLIB_OPTIMA: Dict[str, int] = {
    "ft06": 55, "la01": 666, "la02": 655, "la03": 597, "la04": 590, "la05": 593,
    "la16": 945, "la17": 784, "la18": 848, "la19": 842, "la20": 902,
    "abz5": 1234, "abz6": 943, "ft10": 930, "orb01": 1059,
}


# ---------------------------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------------------------

TSP_SIZES: List[int] = [5, 6, 7, 8, 10, 12, 15, 20]


def tsp_instances(seed: int = VALIDITY_SEED) -> List[Instance]:
    """Random instances of 5 to 20 cities, and the vertices of a regular polygon (its optimal tour
    is the perimeter: every city's two nearest neighbours are tied)."""
    instances: List[Instance] = []
    for k, n in enumerate(TSP_SIZES):
        inst: Instance = LoaderTSP().generate(1, seed=seed * 100 + k, min_cities=n, max_cities=n)[0]
        inst["note"] = f"random, {n} cities"
        instances.append(inst)
    n = 12
    coords: Dict[int, Tuple[float, float]] = {
        i: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(1, n + 1)}
    c: Dict[Tuple[int, int], float] = {
        (i, j): math.dist(coords[i], coords[j]) for i in coords for j in coords if i != j}
    instances.append({"name": f"polygon_{n}", "note": f"the {n} vertices of a regular polygon",
                      "args": {"n": n, "c": c}})
    return instances


# ---------------------------------------------------------------------------------------------
# CWLP
# ---------------------------------------------------------------------------------------------

def _cwlp(name: str, note: str, d: Dict[int, float], u: Dict[int, float], rng: random.Random,
          fixed_scale: float = 1.0) -> Instance:
    customers: List[int] = sorted(d)
    warehouses: List[int] = sorted(u)
    f: Dict[int, float] = {j: fixed_scale * rng.randint(*FIXED_COST_RANGE) for j in warehouses}
    c: Dict[Tuple[int, int], float] = {
        (i, j): d[i] * rng.uniform(*UNIT_COST_RANGE) for i in customers for j in warehouses}
    return {"name": name, "note": note, "args": {"I": customers, "J": warehouses, "d": d, "u": u, "f": f, "c": c}}


def cwlp_instances(seed: int = VALIDITY_SEED) -> List[Instance]:
    """
    Candidates (some may be infeasible with single sourcing: prepare_validity.py keeps the ones it
    solves to optimality):
      - tight capacity: total capacity only 1.02-1.10 x the total demand;
      - expensive warehouses (opening costs x 30): the optimum opens just enough capacity;
      - big customers (demands between u/3 and u/2 of a warehouse): few fit together;
      - equal capacities, and a single warehouse that must be opened.
    """
    rng: random.Random = random.Random(seed)
    out: List[Instance] = []
    for k, (n_c, n_w, buffer) in enumerate([(15, 5, 1.02), (20, 6, 1.05), (25, 8, 1.08), (30, 8, 1.10)]):
        d = {i: float(rng.randint(50, 1500)) for i in range(1, n_c + 1)}
        avg: float = buffer * sum(d.values()) / n_w
        u = {j: float(int(avg * rng.uniform(0.9, 1.1))) for j in range(1, n_w + 1)}
        out.append(_cwlp(f"tight_cap_{k}_{n_c}x{n_w}", f"tight capacity: total capacity {sum(u.values()) / sum(d.values()):.2f} "
                         "x the total demand", d, u, rng))
    for k, (n_c, n_w) in enumerate([(20, 6), (30, 8)]):
        d = {i: float(rng.randint(50, 1500)) for i in range(1, n_c + 1)}
        avg = 1.3 * sum(d.values()) / n_w * 2  # roomy warehouses, but each costs a lot to open
        u = {j: float(int(avg * rng.uniform(0.9, 1.1))) for j in range(1, n_w + 1)}
        out.append(_cwlp(f"expensive_{k}_{n_c}x{n_w}", "expensive warehouses: the optimum opens as few as possible",
                         d, u, rng, fixed_scale=30.0))
    for k, (n_c, n_w) in enumerate([(12, 6), (16, 8)]):
        cap: float = 1000.0
        u = {j: cap for j in range(1, n_w + 1)}
        d = {i: float(rng.randint(int(cap / 3) + 1, int(cap / 2))) for i in range(1, n_c + 1)}
        out.append(_cwlp(f"big_customers_{k}_{n_c}x{n_w}", "big customers: every demand between 1/3 and 1/2 "
                         "of a warehouse's capacity, so at most 2 per warehouse", d, u, rng))
    d = {i: float(rng.randint(50, 1500)) for i in range(1, 11)}
    out.append(_cwlp("one_warehouse_10x1", "a single warehouse, which must be opened and holds every customer",
                     d, {1: sum(d.values()) * 1.01}, rng))
    d = {i: float(rng.randint(50, 1500)) for i in range(1, 21)}
    u = {j: float(int(1.25 * sum(d.values()) / 5)) for j in range(1, 6)}
    out.append(_cwlp("equal_cap_20x5", "5 warehouses of equal capacity", d, u, rng))
    return out


# ---------------------------------------------------------------------------------------------
# JSSP
# ---------------------------------------------------------------------------------------------

def _jssp(name: str, note: str, times: List[List[int]], machines: List[List[int]]) -> Instance:
    return {"name": name, "note": note,
            "args": {"n_jobs": len(times), "n_machines": len(times[0]), "times": times, "machines": machines}}


def jssp_generated(seed: int = VALIDITY_SEED) -> List[Instance]:
    """
    - tiny random instances (3x3, 4x3, 4x4);
    - a bottleneck machine (its operations 5x longer): the makespan is close to its load;
    - a dominant job (its operations n_jobs x longer): the makespan is its total time;
    - flow shops (every job visits the machines in the same order).
    """
    rng: random.Random = random.Random(seed)

    def random_times(n_jobs: int, n_machines: int) -> List[List[int]]:
        return [[rng.randint(*TIME_RANGE) for _ in range(n_machines)] for _ in range(n_jobs)]

    def random_orders(n_jobs: int, n_machines: int) -> List[List[int]]:
        return [rng.sample(range(n_machines), n_machines) for _ in range(n_jobs)]

    out: List[Instance] = []
    for n_j, n_m in [(3, 3), (4, 3), (4, 4)]:
        out.append(_jssp(f"tiny_{n_j}x{n_m}", f"random, {n_j} jobs x {n_m} machines",
                         random_times(n_j, n_m), random_orders(n_j, n_m)))
    for n_j, n_m in [(5, 4), (6, 4), (8, 5)]:
        machines = random_orders(n_j, n_m)
        times = random_times(n_j, n_m)
        times = [[t * 5 if machines[j][k] == 0 else t for k, t in enumerate(row)] for j, row in enumerate(times)]
        out.append(_jssp(f"bottleneck_{n_j}x{n_m}", "a bottleneck machine (machine 0, operations 5x longer): "
                         "the makespan is close to its load", times, machines))
    for n_j, n_m in [(6, 4), (8, 5)]:
        machines = random_orders(n_j, n_m)
        times = random_times(n_j, n_m)
        times[0] = [t * n_j for t in times[0]]
        out.append(_jssp(f"dominant_job_{n_j}x{n_m}", "a dominant job (job 0, operations much longer): the makespan "
                         "is its total processing time", times, machines))
    for n_j, n_m in [(5, 4), (6, 5)]:
        out.append(_jssp(f"flow_shop_{n_j}x{n_m}", "a flow shop: every job visits the machines in the same order",
                         random_times(n_j, n_m), [list(range(n_m)) for _ in range(n_j)]))
    return out


def jsplib_instances(raw_dir: str, names: Optional[List[str]] = None) -> List[Instance]:
    """Public JSSP instances of JSPLIB (downloaded once to raw_dir), with "optimum" = the published value."""
    os.makedirs(raw_dir, exist_ok=True)
    out: List[Instance] = []
    for name in names or list(JSPLIB_OPTIMA):
        path: str = download(JSPLIB_URL.format(name=name), os.path.join(raw_dir, name))
        with open(path) as f:
            text: str = "\n".join(line for line in f if not line.lstrip().startswith("#"))
        inst: Instance = parse_taillard(text, name)
        inst["source"] = "JSPLIB"
        inst["optimum"] = JSPLIB_OPTIMA[name]
        inst["note"] = f"public instance {name}, published optimum {JSPLIB_OPTIMA[name]}"
        out.append(inst)
    return out


# ---------------------------------------------------------------------------------------------
# MCND
# ---------------------------------------------------------------------------------------------

# public R instances outside the paper's test set (r10-r18): 10 nodes, 35-83 arcs, 10-50 commodities;
# variants .1 / .5 / .9 = loose capacity and cheap arcs / medium / tight capacity and expensive arcs
MCND_PUBLIC: List[str] = [f"r{g:02d}.{v}" for g in range(1, 10) for v in (1, 5, 9)]


def _mcnd(name: str, note: str, arcs: Dict[Tuple[int, int], Tuple[int, int, int]],
          commodities: List[Tuple[int, int, int]]) -> Instance:
    """An instance written by hand: arcs (i, j) -> (unit cost, capacity, fixed cost), commodities
    (origin, destination, demand)."""
    nodes: List[int] = sorted({i for arc in arcs for i in arc})
    return {"name": name, "note": note, "args": {
        "N": nodes, "A": sorted(arcs), "K": list(range(1, len(commodities) + 1)),
        "c": {a: v[0] for a, v in arcs.items()}, "u": {a: v[1] for a, v in arcs.items()},
        "f": {a: v[2] for a, v in arcs.items()},
        "d": {k: q for k, (_, _, q) in enumerate(commodities, 1)},
        "O": {k: o for k, (o, _, _) in enumerate(commodities, 1)},
        "D": {k: t for k, (_, t, _) in enumerate(commodities, 1)}}}


def mcnd_instances(raw_dir: str, seed: int = VALIDITY_SEED) -> List[Instance]:
    """
    - random instances of 5 to 10 nodes (loaders/mcnd.py), each size with loose capacities and
      cheap arcs, and with tight capacities and expensive arcs;
    - a demand larger than any single path's capacity: the optimum splits it over two paths;
    - a network made of one cycle only: every arc on a used path must open;
    - expensive arcs and roomy capacities: the optimum opens one path per commodity;
    - public instances r01-r09 of CommaLab (variants .1, .5, .9), outside the paper's test set.
    """
    out: List[Instance] = []
    loader: LoaderMCND = LoaderMCND(data_dir="/tmp")
    for k, (n, arcs, commodities) in enumerate([(5, 10, 3), (6, 14, 5), (8, 20, 8), (10, 30, 12)]):
        for tight, fixed, note in [(1.0, 1, "loose capacities, cheap arcs"), (0.125, 10, "tight capacities, expensive arcs")]:
            inst: Instance = loader.generate(1, seed=seed + k, min_nodes=n, max_nodes=n, min_arcs=arcs, max_arcs=arcs,
                                             min_commodities=commodities, max_commodities=commodities,
                                             tightness=tight, fixed_level=fixed)[0]
            name: str = f"random_{'loose' if fixed == 1 else 'tight'}_{n}n{arcs}a{commodities}k"
            out.append({"name": name, "note": f"random, {note}", "args": inst["args"]})
    out.append(_mcnd("split_demand", "a demand of 10 on two parallel paths of capacity 6 each: the optimum "
                     "splits it", {(1, 2): (1, 6, 10), (2, 4): (1, 6, 10), (1, 3): (1, 6, 10), (3, 4): (1, 6, 10)},
                     [(1, 4, 10)]))
    cycle: Dict[Tuple[int, int], Tuple[int, int, int]] = {(i, i % 6 + 1): (10 + i, 100, 50 * i) for i in range(1, 7)}
    out.append(_mcnd("cycle_only", "a single cycle of 6 arcs: every arc on a used path must open",
                     cycle, [(1, 4, 20), (3, 6, 15), (5, 2, 30), (6, 1, 10)]))
    expensive: Dict[Tuple[int, int], Tuple[int, int, int]] = {
        (i, j): (5 + (3 * i + j) % 7, 200, 1000) for i in range(1, 7) for j in range(1, 7) if i != j}
    out.append(_mcnd("expensive_arcs", "expensive arcs, roomy capacities: few arcs open, shared by the commodities",
                     expensive, [(1, 6, 30), (2, 5, 20), (3, 4, 25), (6, 1, 15)]))
    for inst in commalab_r_instances(raw_dir, MCND_PUBLIC):
        inst["note"] = f"public instance {inst['name']} (CommaLab R, outside the paper's test set r10-r18)"
        out.append(inst)
    return out


# ---------------------------------------------------------------------------------------------
# IMO6
# ---------------------------------------------------------------------------------------------

# every N solved to optimality in under 15 s on the Mac (runs/analysis/imo6_sizes_20261113.txt:
# N = 8 takes 14.6 s, N = 9 much more). An IMO6 instance is only its N, so these small grids are
# also in the paper's test set (N = 3..36): unavoidable, and harmless for the benchmark (the base
# model solves them at once, their gap is 0 with or without a cut)
IMO6_SIZES: List[int] = [2, 3, 4, 5, 6, 7, 8]


def imo6_instances() -> List[Instance]:
    """The grids N = 2..8. The note is quoted to the LLM when a cut removes an optimum: it gives
    neither the minimum nor the olympiad's formula (prompts/problems/imo6.yaml)."""
    out: List[Instance] = []
    for N in IMO6_SIZES:
        inst: Instance = imo6_instance(N)
        inst["note"] = f"the {N} x {N} grid"
        out.append(inst)
    return out


# ---------------------------------------------------------------------------------------------
# GISP
# ---------------------------------------------------------------------------------------------

# random graphs of the generator's distribution (G(n, 0.3), 25% removable), solved at the root
# in under 0.1 s up to 50 nodes (runs/analysis/gisp_sizes_20261113.txt)
GISP_SIZES: List[int] = [6, 8, 10, 15, 20, 30, 40]


def _gisp(name: str, note: str, n: int, permanent: List[Tuple[int, int]], removable: List[Tuple[int, int]],
          weight: int = NODE_WEIGHT, cost: int = EDGE_COST) -> Instance:
    return {"name": name, "note": note, "args": {"nodes": list(range(n)), "permanent": permanent,
                                                 "removable": removable, "weight": weight, "cost": cost}}


def gisp_instances(seed: int = VALIDITY_SEED) -> List[Instance]:
    """Random graphs of 6 to 40 nodes, and TIGHT graphs where the classical bounds hold with
    equality or where removable edges change the answer: a cut that treats every edge as
    permanent, or adds a margin to a clique or odd-cycle bound, removes their optimum."""
    instances: List[Instance] = []
    for k, n in enumerate(GISP_SIZES):
        inst: Instance = LoaderGISP().generate(1, seed=seed * 100 + k, min_nodes=n, max_nodes=n)[0]
        inst["note"] = f"random, {n} nodes"
        instances.append(inst)
    cycle: List[Tuple[int, int]] = [(i, i + 1) for i in range(4)] + [(0, 4)]
    complete: List[Tuple[int, int]] = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    instances += [
        _gisp("gisp_empty_6", "6 nodes, no edge", 6, [], []),
        _gisp("gisp_permanent_clique_6", "6 nodes, every pair joined by a permanent edge", 6, complete, []),
        _gisp("gisp_removable_clique_6", "6 nodes, every pair joined by a removable edge", 6, [], complete),
        _gisp("gisp_permanent_cycle_5", "a cycle of 5 permanent edges", 5, cycle, []),
        _gisp("gisp_mixed_cycle_5", "a cycle of 5 edges, one of them removable", 5, cycle[1:], cycle[:1]),
        _gisp("gisp_break_even_6", "6 nodes, every pair removable, removing an edge costs a node's weight",
              6, [], complete, weight=5, cost=5),
        _gisp("gisp_bipartite_8", "a complete bipartite graph 4 x 4 of permanent edges", 8,
              [(i, j) for i in range(4) for j in range(4, 8)], []),
    ]
    return instances


# ---------------------------------------------------------------------------------------------
# Corridor
# ---------------------------------------------------------------------------------------------

# random lattices of the generator's distribution, solved at the root in under 0.3 s up to order 8
# (runs/analysis/corridor_sizes_20261113.txt); only those the budget surely connects (connectable)
CORRIDOR_ORDERS: List[int] = [3, 4, 5, 6, 7, 8]


def _corridor(name: str, note: str, n: int, edges: List[Tuple[int, int]], reserves: List[int],
              cost: List[int], utility: List[int], budget: float) -> Instance:
    return {"name": name, "note": note, "args": {
        "nodes": list(range(n)), "edges": edges, "reserves": reserves, "budget": budget,
        "cost": {i: cost[i] for i in range(n)}, "utility": {i: utility[i] for i in range(n)}}}


def corridor_instances(seed: int = VALIDITY_SEED) -> List[Instance]:
    """Random lattices of order 3 to 8, and TIGHT graphs: a budget that buys exactly the cheapest
    corridor, or everything, a path where every cell must be bought, and graphs that are not
    lattices (a cycle, a star): a cut that assumes a grid, or drops a cell the budget can just
    afford, removes their optimum."""
    instances: List[Instance] = []
    for k, order in enumerate(CORRIDOR_ORDERS):
        draw: int = 0
        while True:  # the first seed whose lattice the budget surely connects
            inst: Instance = LoaderCorridor().generate(1, seed=seed * 100 + 10 * k + draw, min_order=order,
                                                       max_order=order)[0]
            if connectable(**inst["args"]):
                break
            draw += 1
        inst["note"] = f"random, {order} x {order} lattice"
        instances.append(inst)
    path: List[Tuple[int, int]] = [(i, i + 1) for i in range(5)]
    grid3: List[Tuple[int, int]] = lattice(3)
    instances += [
        _corridor("corridor_path_6", "a path of 6 cells, reserves at both ends, the budget buys exactly the 4 cells between",
                  6, path, [0, 5], [0, 10, 20, 30, 40, 0], [5, 1, 1, 1, 1, 5], 100),
        _corridor("corridor_cheapest_3x3", "3 x 3 lattice, reserves at opposite corners, the budget buys exactly the cheapest corridor",
                  9, grid3, [0, 8], [0, 1, 9, 9, 1, 9, 9, 1, 0], [1, 1, 50, 50, 1, 50, 50, 1, 1], 3),
        _corridor("corridor_everything_3x3", "3 x 3 lattice, the budget buys every cell",
                  9, grid3, [0, 8], [0, 5, 5, 5, 5, 5, 5, 5, 0], [1] * 9, 35),
        _corridor("corridor_cycle_8", "a cycle of 8 cells, reserves opposite, one side cheap and one side useful",
                  8, [(i, (i + 1) % 8) if i < 7 else (0, 7) for i in range(8)], [0, 4],
                  [0, 1, 1, 1, 0, 5, 5, 5], [1, 1, 1, 1, 1, 20, 20, 20], 15),
        _corridor("corridor_star_6", "a star: a centre and 5 leaves, 2 reserves among the leaves",
                  6, [(0, i) for i in range(1, 6)], [1, 2], [7, 0, 0, 3, 3, 3], [1, 1, 1, 9, 9, 9], 13),
    ]
    return instances
