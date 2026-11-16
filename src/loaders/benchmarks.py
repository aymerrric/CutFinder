"""
The public benchmark instances on which the EvoCut paper reports its results (its held-out test
set D_t; papers/sage_evocut_tmlr2026.pdf, Section 5.1 and Appendix F):
    tsp   the 54 TSPLIB instances with n < 250 cities (Reinelt 1991)
    cwlp  the 36 Beasley / OR-Library instances feasible with single sourcing: cap61-cap134 and
          capa1-capc4 (16 to 100 warehouses, 50 to 1000 customers), as CommaLab distributes them
    jssp  Taillard ta01-ta80 (15 jobs x 15 machines to 100 x 20; Taillard 1993)
    mcnd  CommaLab group R, r10-r18 (9 variants each: 81 instances; 20 nodes, 120-318 arcs, 40-200
          commodities; Crainic, Frangioni & Gendron 2001)
    imo6  the grids N = 3..36 (34 instances, generated: nothing to download; Appendix F.6)
    gisp  not in the paper: Distributional MIPLIB's GISP "hard" test set (100 graphs of 175 nodes,
          Huang et al. 2024; over 1000 s for Gurobi 10 in their measurements)
    corridor  not in the paper and no public set: 50 lattices of order 20 to 30 generated once with
          a fixed seed (the corridor generator's distribution), nothing to download
Each one is downloaded once into a raw folder, then parsed into our instance format
({"name", "args"}: the arguments of the problem's create_model), so the base models and the cuts
run on them unchanged. prepare_benchmark.py is the script that does it.
"""
import ast
import csv
import gzip
import logging
import math
import os
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import gurobipy as gp

from loaders.base import Instance
from loaders.corridor import LoaderCorridor, connectable
from loaders.imo6 import instance as imo6_instance
from utils import parameters  # noqa: F401  (sets GRB_LICENSE_FILE before Gurobi starts)

logger: logging.Logger = logging.getLogger(__name__)

USER_AGENT: str = "my-evo-cut/1.0"
TSPLIB_URL: str = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
TSPLIB_MAX_CITIES: int = 250  # paper: "54 TSPLIB instances with n < 250 cities"
BEASLEY_URL: str = "https://commalab.di.unipi.it/files/Data/mexch/BeasleyData.zip"
TAILLARD_URL: str = (  # the repository the reference code names (data/data_prepare_jssp.py)
    "https://raw.githubusercontent.com/ai-for-decision-making-tue/"
    "Job_Shop_Scheduling_Benchmark_Environments_and_Instances/main/data/jsp/taillard/{name}"
)
TAILLARD_NAMES: List[str] = [f"ta{k:02d}" for k in range(1, 81)]
COMMALAB_R_URL: str = "https://commalab.di.unipi.it/files/Data/MMCF/R.tgz"  # the reference's source
COMMALAB_R_NAMES: List[str] = [f"r{g}.{v}" for g in range(10, 19) for v in range(1, 10)]  # the paper's 81
# Distributional MIPLIB (Hugging Face): one CSV per domain, level and split, one model per row as
# LP or MPS text (column "MILP": the repr of the bytes; column "format": "mps", else LP)
DMIPLIB_URL: str = "https://huggingface.co/datasets/weiminhu/D-MIPLIB/resolve/main/{domain}/{level}/{split}.csv"
GISP_LEVEL: str = "hard"

Costs = Dict[Tuple[int, int], float]


# ---------------------------------------------------------------------------------------------
# TSPLIB (format: Reinelt, "TSPLIB 95", the documentation of the library)
# ---------------------------------------------------------------------------------------------

def _nint(x: float) -> int:
    """TSPLIB's nint: nearest integer (x >= 0 here)."""
    return int(x + 0.5)


def _euc_2d(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    return _nint(math.dist(a, b))


def _att(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """Pseudo-Euclidean distance of att48 / att532."""
    r: float = math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10.0)
    t: int = _nint(r)
    return t + 1 if t < r else t


def _geo_radians(x: float) -> float:
    """TSPLIB GEO coordinate DDD.MM (degrees.minutes) -> radians, with TSPLIB's PI = 3.141592."""
    degrees: int = int(x)  # truncation, as every TSPLIB implementation (and its optimal values) does
    minutes: float = x - degrees
    return 3.141592 * (degrees + 5.0 * minutes / 3.0) / 180.0


def _geo(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """Geographical distance in km on an idealized sphere (RRR = 6378.388)."""
    lat_a, lon_a = _geo_radians(a[0]), _geo_radians(a[1])
    lat_b, lon_b = _geo_radians(b[0]), _geo_radians(b[1])
    q1: float = math.cos(lon_a - lon_b)
    q2: float = math.cos(lat_a - lat_b)
    q3: float = math.cos(lat_a + lat_b)
    return int(6378.388 * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


DISTANCES: Dict[str, Callable[[Tuple[float, float], Tuple[float, float]], int]] = {
    "EUC_2D": _euc_2d, "ATT": _att, "GEO": _geo,
}
_SECTIONS: Tuple[str, ...] = ("NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION",
                              "TOUR_SECTION", "EOF")


def _read_tsplib(text: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """(header "KEY: value" fields, section name -> its whitespace-separated tokens)."""
    header: Dict[str, str] = {}
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        stripped: str = line.strip()
        if not stripped:
            continue
        keyword: str = stripped.split(":")[0].strip().upper()
        if keyword in _SECTIONS:
            current = None if keyword == "EOF" else keyword
            if current is not None:
                sections[current] = []
            continue
        if current is not None:
            sections[current] += stripped.split()
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
            header[key.strip().upper()] = value.strip()
    return header, sections


def _explicit_matrix(n: int, fmt: str, tokens: List[str]) -> Costs:
    """The symmetric costs written as a matrix in EDGE_WEIGHT_SECTION (formats of the 54 instances)."""
    values: List[float] = [float(t) for t in tokens]
    pairs: List[Tuple[int, int]]
    if fmt == "FULL_MATRIX":
        pairs = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)]
    elif fmt == "UPPER_ROW":
        pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    elif fmt == "UPPER_DIAG_ROW":
        pairs = [(i, j) for i in range(1, n + 1) for j in range(i, n + 1)]
    elif fmt == "LOWER_DIAG_ROW":
        pairs = [(i, j) for i in range(1, n + 1) for j in range(1, i + 1)]
    else:
        raise ValueError(f"EDGE_WEIGHT_FORMAT {fmt} is not supported")
    if len(values) != len(pairs):
        raise ValueError(f"{fmt}: {len(values)} weights for {len(pairs)} entries (n={n})")
    costs: Costs = {}
    for (i, j), value in zip(pairs, values):
        if i != j:
            costs[i, j] = costs[j, i] = value
    return costs


def parse_tsplib(text: str) -> Instance:
    """A symmetric TSPLIB instance (.tsp text) -> {"name", "args": {"n", "c"}}, cities numbered 1..n."""
    header, sections = _read_tsplib(text)
    if not header.get("TYPE", "").startswith("TSP"):
        raise ValueError(f"{header.get('NAME')}: TYPE {header.get('TYPE')} is not a symmetric TSP")
    name: str = header["NAME"].removesuffix(".tsp")  # ulysses16/22 write "ulysses16.tsp"
    n: int = int(header["DIMENSION"])
    kind: str = header["EDGE_WEIGHT_TYPE"]
    costs: Costs
    if kind == "EXPLICIT":
        costs = _explicit_matrix(n, header["EDGE_WEIGHT_FORMAT"], sections["EDGE_WEIGHT_SECTION"])
    elif kind in DISTANCES:
        tokens: List[str] = sections["NODE_COORD_SECTION"]
        coords: Dict[int, Tuple[float, float]] = {
            k + 1: (float(tokens[3 * k + 1]), float(tokens[3 * k + 2])) for k in range(n)
        }  # the node ids of the 54 instances are 1..n in order
        distance = DISTANCES[kind]
        costs = {(i, j): float(distance(coords[i], coords[j])) for i in coords for j in coords if i != j}
    else:
        raise ValueError(f"{name}: EDGE_WEIGHT_TYPE {kind} is not supported")
    return {"name": name, "source": "TSPLIB", "args": {"n": n, "c": costs}}


def parse_tsplib_tour(text: str) -> List[int]:
    """The cities of a TSPLIB .opt.tour file, in tour order."""
    _, sections = _read_tsplib(text)
    tour: List[int] = []
    for token in sections["TOUR_SECTION"]:
        if token == "-1":
            break
        tour.append(int(token))
    return tour


def tour_length(costs: Costs, tour: List[int]) -> float:
    return sum(costs[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour)))


# ---------------------------------------------------------------------------------------------
# CWLP: Beasley's capacitated warehouse location instances, CommaLab's files
# ---------------------------------------------------------------------------------------------

def parse_beasley(text: str, name: str) -> Instance:
    """
    One file of CommaLab's BeasleyData.zip -> {"name", "args": {"I", "J", "d", "u", "f", "c"}}.
    Layout (also what the reference parser reads): "m n"; m lines "capacity fixed_cost"; one
    line with the n demands; then m lines (one per warehouse) of the n costs of serving each
    customer's whole demand from that warehouse. Customers and warehouses are numbered from 1.
    """
    values: List[float] = [float(t) for t in text.split()]
    m, n = int(values[0]), int(values[1])
    expected: int = 2 + 2 * m + n + m * n
    if len(values) != expected:
        raise ValueError(f"{name}: {len(values)} numbers, expected {expected} for {m} warehouses x {n} customers")
    pos: int = 2
    u: Dict[int, float] = {}
    f: Dict[int, float] = {}
    for j in range(1, m + 1):
        u[j], f[j] = values[pos], values[pos + 1]
        pos += 2
    d: Dict[int, float] = {i: values[pos + i - 1] for i in range(1, n + 1)}
    pos += n
    c: Costs = {}
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c[i, j] = values[pos]
            pos += 1
    return {"name": name, "source": "OR-Library (CommaLab)",
            "args": {"I": list(range(1, n + 1)), "J": list(range(1, m + 1)), "d": d, "u": u, "f": f, "c": c}}


# ---------------------------------------------------------------------------------------------
# JSSP: Taillard
# ---------------------------------------------------------------------------------------------

def parse_taillard(text: str, name: str) -> Instance:
    """
    One Taillard file -> {"name", "args": {"n_jobs", "n_machines", "times", "machines"}}.
    Layout: "n_jobs n_machines", then one line per job of "machine time" pairs in processing
    order, machines numbered from 0 (as in our models).
    """
    values: List[int] = [int(t) for t in text.split()]
    n_jobs, n_machines = values[0], values[1]
    if len(values) != 2 + 2 * n_jobs * n_machines:
        raise ValueError(f"{name}: {len(values)} numbers for {n_jobs} jobs x {n_machines} machines")
    pairs: List[int] = values[2:]
    machines: List[List[int]] = []
    times: List[List[int]] = []
    for j in range(n_jobs):
        row: List[int] = pairs[2 * n_machines * j: 2 * n_machines * (j + 1)]
        machines.append(row[0::2])
        times.append(row[1::2])
        if sorted(machines[-1]) != list(range(n_machines)):
            raise ValueError(f"{name}: job {j} does not visit every machine once")
    return {"name": name, "source": "Taillard",
            "args": {"n_jobs": n_jobs, "n_machines": n_machines, "times": times, "machines": machines}}


# ---------------------------------------------------------------------------------------------
# MCND: CommaLab (MULTIGEN .dow files)
# ---------------------------------------------------------------------------------------------

def parse_dow(text: str, name: str) -> Instance:
    """
    One .dow file -> {"name", "args": {"N", "A", "K", "c", "f", "u", "d", "O", "D"}}.
    Layout (also what the reference parser reads): a title line ("MULTIGEN.DAT:"); "n_nodes
    n_arcs n_commodities"; one line per arc "i j unit_cost capacity fixed_cost" (two more numbers
    follow, unused); one line per commodity "origin destination demand". Nodes are numbered from
    1 in the file; commodities are numbered 1, 2, ... in file order.
    """
    lines: List[str] = [line for line in text.splitlines() if line.strip()]
    n_nodes, n_arcs, n_commodities = (int(t) for t in lines[1].split()[:3])
    if len(lines) < 2 + n_arcs + n_commodities:
        raise ValueError(f"{name}: {len(lines)} lines for {n_arcs} arcs and {n_commodities} commodities")
    c: Costs = {}
    f: Costs = {}
    u: Costs = {}
    for line in lines[2:2 + n_arcs]:
        i, j, unit, capacity, fixed = (int(t) for t in line.split()[:5])
        if (i, j) in c:
            raise ValueError(f"{name}: arc {(i, j)} twice")
        c[i, j], u[i, j], f[i, j] = unit, capacity, fixed
    origin: Dict[int, int] = {}
    destination: Dict[int, int] = {}
    demand: Dict[int, int] = {}
    for k, line in enumerate(lines[2 + n_arcs:2 + n_arcs + n_commodities], 1):
        origin[k], destination[k], demand[k] = (int(t) for t in line.split()[:3])
    nodes: List[int] = sorted({i for arc in c for i in arc} | set(origin.values()) | set(destination.values()))
    if nodes != list(range(1, n_nodes + 1)):
        raise ValueError(f"{name}: nodes {nodes[:5]}... are not 1..{n_nodes}")
    return {"name": name, "source": "CommaLab (MULTIGEN, group R)",
            "args": {"N": nodes, "A": sorted(c), "K": list(range(1, n_commodities + 1)), "c": c, "f": f,
                     "u": u, "d": demand, "O": origin, "D": destination}}


# ---------------------------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------------------------

def download(url: str, path: str) -> str:
    """Download `url` to `path` unless the file is already there; return the path."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # e.g. data/validity/raw/<problem>/ the first time
    logger.info(f"downloading {url}")
    request: urllib.request.Request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data: bytes = response.read()
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)  # never leaves a half-written file behind
    return path


def tsplib_instances(raw_dir: str, max_cities: int = TSPLIB_MAX_CITIES) -> List[Instance]:
    """The symmetric TSPLIB instances with fewer than `max_cities` cities, smallest first."""
    archive: str = download(TSPLIB_URL, os.path.join(raw_dir, "ALL_tsp.tar.gz"))
    instances: List[Instance] = []
    with tarfile.open(archive, "r:gz") as tar:
        members: Dict[str, tarfile.TarInfo] = {m.name.split("/")[-1]: m for m in tar.getmembers() if m.isfile()}

        def read(member: str) -> str:
            data: bytes = tar.extractfile(members[member]).read()  # type: ignore[union-attr]
            return gzip.decompress(data).decode()

        for member in sorted(members):
            if not member.endswith(".tsp.gz"):
                continue
            text: str = read(member)
            dimension: Optional[re.Match] = re.search(r"^\s*DIMENSION\s*:\s*(\d+)", text, re.MULTILINE)
            if dimension is None or int(dimension.group(1)) >= max_cities:
                continue
            instance: Instance = parse_tsplib(text)
            tour_file: str = member.replace(".tsp.gz", ".opt.tour.gz")
            if tour_file in members:  # optimal tour shipped with the library: its length is the optimum
                tour: List[int] = parse_tsplib_tour(read(tour_file))
                instance["optimum"] = tour_length(instance["args"]["c"], tour)
            instances.append(instance)
    return sorted(instances, key=lambda inst: (inst["args"]["n"], inst["name"]))


def beasley_instances(raw_dir: str) -> List[Instance]:
    archive: str = download(BEASLEY_URL, os.path.join(raw_dir, "BeasleyData.zip"))
    instances: List[Instance] = []
    with zipfile.ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            base: str = member.split("/")[-1]
            if re.fullmatch(r"cap\w+", base):
                instances.append(parse_beasley(z.read(member).decode(), base))
    return sorted(instances, key=lambda inst: (len(inst["args"]["J"]), len(inst["args"]["I"]), inst["name"]))


def taillard_instances(raw_dir: str) -> List[Instance]:
    instances: List[Instance] = []
    for name in TAILLARD_NAMES:
        path: str = download(TAILLARD_URL.format(name=name), os.path.join(raw_dir, name))
        with open(path) as f:
            instances.append(parse_taillard(f.read(), name))
    return instances


def commalab_r_instances(raw_dir: str, names: Optional[List[str]] = None) -> List[Instance]:
    """The paper's MCND test set: r10.1-r18.9 of CommaLab's R.tgz, in that order."""
    archive: str = download(COMMALAB_R_URL, os.path.join(raw_dir, "R.tgz"))
    wanted: List[str] = names if names is not None else COMMALAB_R_NAMES
    instances: Dict[str, Instance] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            base: str = member.name.split("/")[-1]
            if member.isfile() and base.endswith(".dow") and base[:-4] in wanted:
                instances[base[:-4]] = parse_dow(tar.extractfile(member).read().decode(), base[:-4])  # type: ignore[union-attr]
    missing: List[str] = [name for name in wanted if name not in instances]
    if missing:
        raise ValueError(f"{archive} lacks {', '.join(missing)}")
    return [instances[name] for name in wanted]


# ---------------------------------------------------------------------------------------------
# IMO6: generated, one instance per grid size (the paper's test set: N = 3..36)
# ---------------------------------------------------------------------------------------------

IMO6_TEST_SIZES: List[int] = list(range(3, 37))


def imo6_benchmark() -> List[Instance]:
    """The paper's IMO6 test set: the grids N = 3..36, in that order."""
    return [imo6_instance(N) for N in IMO6_TEST_SIZES]


# ---------------------------------------------------------------------------------------------
# Distributional MIPLIB (Huang, Huang, Ferber, Dilkina 2024): models as LP / MPS text
# ---------------------------------------------------------------------------------------------

def read_dmiplib(path: str) -> List[gp.Model]:
    """The models of one Distributional MIPLIB CSV file, in the order of its rows. Gurobi reads
    each one from a temporary file (it needs the extension to know the format)."""
    csv.field_size_limit(1 << 30)
    env: gp.Env = gp.Env(params={"OutputFlag": 0})
    models: List[gp.Model] = []
    with open(path, newline="") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    for k, row in enumerate(rows):
        text: str = ast.literal_eval(row["MILP"]).decode()  # the cell holds b'...'
        extension: str = "mps" if row.get("format") == "mps" else "lp"
        temporary: str = f"{path}.{k}.{extension}"
        with open(temporary, "w") as out:
            out.write(text)
        try:
            models.append(gp.read(temporary, env))
        finally:
            os.remove(temporary)
    return models


def gisp_from_model(m: gp.Model, name: str) -> Instance:
    """Our GISP instance from the generator's model (GISP/gisp.py: variables node_<i> and
    edge_<i>_<j>, objective -weight * node + cost * edge, rows node_i + node_j (- edge_i_j) <= 1).
    Raises ValueError when the model is not of that form."""
    node_var: Dict[int, gp.Var] = {}
    edge_var: Dict[Tuple[int, int], gp.Var] = {}
    for v in m.getVars():
        match = re.fullmatch(r"node_(\d+)|edge_(\d+)_(\d+)", v.VarName)
        if match is None:
            raise ValueError(f"{name}: unexpected variable {v.VarName}")
        if match.group(1) is not None:
            node_var[int(match.group(1))] = v
        else:
            i, j = int(match.group(2)), int(match.group(3))
            edge_var[(min(i, j), max(i, j))] = v
    weights = {-v.Obj for v in node_var.values()}
    costs = {v.Obj for v in edge_var.values()}
    if len(weights) != 1 or len(costs) > 1 or m.ModelSense != gp.GRB.MINIMIZE:
        raise ValueError(f"{name}: node weights {weights}, edge costs {costs}: not one weight and one cost")
    permanent: List[Tuple[int, int]] = []
    removable: List[Tuple[int, int]] = []
    for c in m.getConstrs():
        row: gp.LinExpr = m.getRow(c)
        terms: Dict[str, float] = {row.getVar(k).VarName: row.getCoeff(k) for k in range(row.size())}
        ends: List[int] = sorted(int(n[5:]) for n in terms if n.startswith("node_"))
        edges: List[str] = [n for n in terms if n.startswith("edge_")]
        if c.Sense != "<" or c.RHS != 1 or len(ends) != 2 or any(terms[f"node_{e}"] != 1 for e in ends):
            raise ValueError(f"{name}: unexpected row {c.ConstrName}")
        if edges:
            if (edges not in ([f"edge_{ends[0]}_{ends[1]}"], [f"edge_{ends[1]}_{ends[0]}"])
                    or terms[edges[0]] != -1):
                raise ValueError(f"{name}: unexpected row {c.ConstrName}")
            removable.append((ends[0], ends[1]))
        else:
            permanent.append((ends[0], ends[1]))
    if set(removable) != set(edge_var):
        raise ValueError(f"{name}: the edge variables and the removable rows differ")
    return {"name": name, "args": {"nodes": sorted(node_var), "permanent": sorted(permanent),
                                   "removable": sorted(removable), "weight": weights.pop(),
                                   "cost": costs.pop() if costs else 1}}


def gisp_benchmark(raw_dir: str, level: str = GISP_LEVEL) -> List[Instance]:
    """Distributional MIPLIB's GISP test set of `level` (100 instances), downloaded once."""
    path: str = download(DMIPLIB_URL.format(domain="GISP", level=level, split="test"),
                         os.path.join(raw_dir, f"gisp_{level}_test.csv"))
    instances: List[Instance] = []
    for k, m in enumerate(read_dmiplib(path)):
        instances.append(gisp_from_model(m, f"gisp_dmiplib_{level}_{k:02d}"))
        m.dispose()
    return instances


CORRIDOR_BENCHMARK_SEED: int = 2026
CORRIDOR_BENCHMARK_COUNT: int = 50


def corridor_benchmark() -> List[Instance]:
    """Our corridor test set: the first CORRIDOR_BENCHMARK_COUNT lattices of order 20 to 30 drawn
    from CORRIDOR_BENCHMARK_SEED that the budget surely connects (loaders/corridor.connectable)."""
    drawn: List[Instance] = LoaderCorridor().generate(4 * CORRIDOR_BENCHMARK_COUNT, CORRIDOR_BENCHMARK_SEED,
                                                      min_order=20, max_order=30)
    return [inst for inst in drawn if connectable(**inst["args"])][:CORRIDOR_BENCHMARK_COUNT]


@dataclass
class Benchmark:
    """A public test set of the paper: where it comes from, and how many instances it has there."""

    description: str
    fetch: Callable[[str], List[Instance]]  # raw folder -> instances
    paper_count: int  # number of test instances in the paper (Section 5.1)
    # mean number of variables and constraints of the base models over the set (paper, Table 20):
    # matching them checks that we have the same instances and formulation
    paper_variables: float
    paper_constraints: float


BENCHMARKS: Dict[str, Benchmark] = {
    "tsp": Benchmark("TSPLIB, symmetric instances with n < 250", tsplib_instances, 54, 15821.65, 15931.00),
    "cwlp": Benchmark("OR-Library CWLP feasible with single sourcing (CommaLab cap61-capc4)",
                      beasley_instances, 36, 34398.00, 420.22),
    "jssp": Benchmark("Taillard ta01-ta80", taillard_instances, 80, 21382.25, 42046.88),
    "mcnd": Benchmark("CommaLab group R, r10.1-r18.9", commalab_r_instances, 81, 24991.89, 2485.22),
    "imo6": Benchmark("generated grids N = 3..36", lambda raw_dir: imo6_benchmark(), 34, 20759.50, 7524.50),
    # not in the paper: the count and the means are those of the Distributional MIPLIB files
    # themselves (read by Gurobi, 2026-11-13), so the check compares our create_model with theirs
    "gisp": Benchmark("Distributional MIPLIB GISP hard, test split", gisp_benchmark, 100, 1313.49, 4560.65),
    # generated: the means are those of our own model on it (13 Nov), a check that it does not change
    "corridor": Benchmark("generated lattices of order 20 to 30 (seed 2026)", lambda raw_dir: corridor_benchmark(),
                          50, 2878.02, 2882.02),
}

