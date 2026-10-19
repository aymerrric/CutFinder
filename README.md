# My_evo_cut

A step-by-step reimplementation of **EvoCut**, which speeds up Mixed-Integer Linear
Programming (MILP) solving. An LLM proposes problem-specific cutting planes (extra
constraints that tighten the LP relaxation), and an evolutionary algorithm keeps the cuts
that help the solver most.

The reference implementation is in `../repos/EvoCut`. This repository rebuilds it one
small piece at a time, with the goal of understanding every step. The
[Progress](#progress) section below records what has been built so far.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install gurobipy pyyaml tqdm pytest
```

- The code is written for **Python 3.9+**; the current `.venv` runs Python 3.11.
- You need a licensed MILP solver ([Gurobi](https://www.gurobi.com/)). Academic licenses
  are free. Without a license, the pip version of gurobipy is capped at 2000 variables and
  2000 constraints.

### Gurobi license
Put your license file at the project root as `gurobi.lic`. It is git-ignored: **never
commit it**. When `utils/parameters.py` is imported, it sets `GRB_LICENSE_FILE` to that path,
unless you have already exported that variable yourself. When the license loads, Gurobi
prints `Academic license ... registered to ...`.

### Folders
The code never creates folders. After cloning, create the git-ignored output folders
yourself:

```bash
mkdir -p data/tsp cache logs
```

Run every script **from inside `src/`**, because imports are written relative to it
(`from utils import parameters`):

```bash
cd src
python <script>.py --help
```

---

## Project structure

```
My_evo_cut/
├── data/                  # datasets, one subfolder per problem (data/tsp/, ...)
├── cache/                 # cached solver results
├── logs/                  # one Gurobi log file per solve
├── gurobi.lic             # Gurobi license (secret, git-ignored)
├── .gitignore             # hides secrets, .venv, data/, cache/, logs/
├── pytest.ini             # test config: run `pytest` from the project root
├── tests/                 # one test file per module, updated at every step
└── src/
    ├── utils/
    │   ├── parameters.py  # default values for every setting + project paths
    │   └── parser.py      # command-line arguments (defaults come from parameters.py)
    ├── loaders/
    │   ├── base.py        # Loader: abstract interface to generate / save / load / split instances
    │   └── tsp.py         # LoaderTSP: random Euclidean TSP instances
    ├── models/
    │   └── tsp.py         # create_model(n, c): TSP MILP (MTZ) in gurobipy
    └── solvers/
        └── solver.py      # Solver: problem-independent, compares formulations, logs, stores results
```

`data/`, `cache/` and `logs/` are resolved as absolute paths from `parameters.PROJECT_ROOT`, so they
work no matter which directory you run from.

---

## Progress

### Step 1: Configuration and command-line arguments
- [src/utils/parameters.py](src/utils/parameters.py) stores the default value of every
  setting: LLM name, problem name, population size, elitism count, and the cache and
  data paths.
- [src/utils/parser.py](src/utils/parser.py) exposes these settings as CLI flags:
  `--model_name`, `--population_file`, `--problem_name`, `--cache_dir`,
  `--top_selected` and `--population_size`.

### Step 2: Instance loaders
An **instance** is a plain dict, `Instance = Dict[str, Any]`:

```python
{"name": "rand_0_12", "args": {...}, ...}   # "args" are passed to create_model(**args)
```

- [src/loaders/base.py](src/loaders/base.py) defines the abstract class `Loader`. Each
  loader works in its own folder, `data/<problem>/`, and provides these methods:
  - `generate(n_instances, seed, **kwargs)`: abstract; each problem implements it.
  - `save` / `load`: pickle a list of instances to and from a file.
  - `get(filename, n_instances, seed, **kwargs)`: loads the file if it exists; otherwise
    it generates the instances, saves them and returns them.
  - `split(instances, ratio, seed)`: a reproducible shuffle that cuts the list into two
    disjoint parts (for example, train and test).
- [src/loaders/tsp.py](src/loaders/tsp.py) defines `LoaderTSP`. For each instance it
  draws `n ∈ [min_cities, max_cities]` cities uniformly in the unit square, numbered
  `1..n`, and builds the Euclidean cost dict `c[(i, j)]`. Instance format:
  `{"name", "coords", "args": {"n", "c"}}`.

```python
from loaders.tsp import LoaderTSP

loader = LoaderTSP()
instances = loader.get("train.pkl", n_instances=20, seed=0, min_cities=10, max_cities=15)
train, test = loader.split(instances, ratio=0.8, seed=0)
```

### Step 3: TSP MILP model (gurobipy)
We use **gurobipy directly** instead of Pyomo, the modelling layer used by the reference.
The code is simpler, the model builds faster, and we get direct access to the solver
statistics EvoCut scores cuts with (`Runtime`, `MIPGap`, `NodeCount`, ...). The trade-offs:
the project needs a Gurobi license, and the reference prompts (written in Pyomo) must be
rewritten in gurobipy.

[src/models/tsp.py](src/models/tsp.py) defines `create_model(n, c) -> gp.Model`, using the
MTZ formulation with city 1 as the depot:

$$
\begin{aligned}
\min\ & \textstyle\sum_{(i,j)} c_{ij} x_{ij} \\
& \textstyle\sum_j x_{ij} = 1,\ \sum_i x_{ij} = 1 && \forall i, j \quad \text{(leave and enter each city once)} \\
& u_i - u_j + n\,x_{ij} \le n - 1 && \forall i, j \ne 1 \quad \text{(MTZ: no subtours)} \\
& x_{ij} \in \{0,1\},\ u_1 = 1,\ u_i \in \{2,\dots,n\}
\end{aligned}
$$

- The variables are `tupledict`s keyed by the cities and arcs, so `x[i, j]` and `u[i]`
  read like the maths.
- The bounds on `u` are set directly on the variables (`lb`, `ub`). The reference uses
  separate constraints for this; both give the same model.
- The variables are also stored on the model as `m._x`, `m._u` and `m._n`. Later code that
  only receives the model, such as a cut, can still reach them.
- The model has `n²` variables. Gurobi's free size-limited license allows 2000, which is
  why we use the academic license (see Setup).

```python
from loaders.tsp import LoaderTSP
from models.tsp import create_model

inst = LoaderTSP().generate(1, seed=0, min_cities=8, max_cities=8)[0]
m = create_model(**inst["args"])
m.Params.OutputFlag = 0
m.optimize()
print(m.ObjVal, [a for a, v in m._x.items() if v.X > 0.5])
```

Checked on random instances with 5 to 8 cities: each solution is a single tour, and its
cost equals the brute-force optimum. A 60-city instance solves to optimality in about 9 s.

### Step 4: Problem-independent solver
[src/solvers/solver.py](src/solvers/solver.py) defines `Solver`. It knows nothing about the
TSP. It works with **formulations**: named functions that build a Gurobi model from an
instance's `args` (`Formulation = Callable[..., gp.Model]`). A formulation with a cut is
the base formulation plus extra constraints, so EvoCut can compare `"base"` against
`"base+cut"` under exactly the same settings.

- `add_formulation(name, build)` registers a formulation.
- `solve(instance, formulation)` builds the model, applies the settings, solves it and
  returns a `SolveResult`. The result is also appended to
  `solver.results[formulation][instance_name]`, a **list** with one entry per solve. Solving
  the same instance again, for example with another seed, adds a result instead of
  overwriting the previous one.
- `solve_all(instances, formulations=None)` runs every instance with every formulation.
- `SolveResult` holds the `seed` used, `status` (`"OPTIMAL"`, `"TIME_LIMIT"`, `"INFEASIBLE"`, ...),
  `runtime`, `node_count`, `sol_count`, `obj_val`, `obj_bound` and `mip_gap`. The
  objective and gap are `None` when no solution was found.
- Every model gets the same `TimeLimit` and `Threads`. The defaults in `parameters.py`
  (10 s and 8 threads) are taken from the reference config.
- **Seeds:** Gurobi's `Seed` changes its internal random choices, so the runtime of the same
  model can vary a lot from one seed to another. `seed_mode` controls it:
  - `SeedMode.SET` (the default) uses `seed` (42) for every solve, so runs are reproducible.
  - `SeedMode.RANDOM` draws a new seed in `[0, 2·10⁹]` for every solve. This is for
    experiments over many seeds. Each seed is recorded in `SolveResult.seed`, in the
    console log line, in the log file name and in the log file itself. To rerun one solve
    exactly, use `SeedMode.SET` with that seed.
- **Logging:** our own messages go through Python `logging` (one line per solve). Gurobi's
  full log goes to `logs/<formulation>_<instance>_seed<seed>.log`, and to the console only with
  `verbose=True`. Note that gurobipy also forwards its log to a Python logger named
  `gurobipy`, so the Solver quiets that logger unless `verbose` is on.
- The `logs/` folder must already exist; otherwise the Solver raises a clear error. Pass
  `log_dir=None` to write no log files.

```python
import logging
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from solvers.solver import SeedMode, Solver

logging.basicConfig(level=logging.INFO)

def with_2cycle_cut(n, c):  # a valid cut: never use both i->j and j->i
    m = create_model(n, c)
    m.addConstrs((m._x[i, j] + m._x[j, i] <= 1 for (i, j) in m._x.keys() if i < j), name="cut")
    return m

solver = Solver()
solver.add_formulation("base", create_model)
solver.add_formulation("base+2cycle", with_2cycle_cut)
instances = LoaderTSP().get("demo.pkl", n_instances=3, seed=0, min_cities=10, max_cities=20)
solver.solve_all(instances)
solver.results["base+2cycle"]["rand_0_16"][0].runtime

# many seeds: 5 solves per instance, all kept
multi = Solver(seed_mode=SeedMode.RANDOM)
multi.add_formulation("base", create_model)
for _ in range(5):
    multi.solve_all(instances)
[r.seed for r in multi.results["base"]["rand_0_16"]]
```

Checked: on 3 instances both formulations reach the same optimum. An 80-city instance with
`time_limit=1` returns `TIME_LIMIT` with an 11.9% gap. An infeasible model returns
`INFEASIBLE` with `None` values instead of crashing.

### Tests
The `tests/` folder holds one file per module. Every new step adds or updates its tests.
Run them from the project root:

```bash
pytest        # or: pytest -v for one line per test
```

[pytest.ini](pytest.ini) puts `src/` on the import path, so tests import modules the same
way scripts do (`from loaders.tsp import LoaderTSP`). Shared fixtures live in
[tests/conftest.py](tests/conftest.py): small TSP instances and a brute-force TSP solver.

| File | What it checks |
|---|---|
| [test_loaders.py](tests/test_loaders.py) | instance format and bounds, reproducibility with a seed, symmetric Euclidean costs, `get` saves then reloads, `split` is disjoint and complete |
| [test_models_tsp.py](tests/test_models_tsp.py) | model size (`n²` variables), optimum equals brute force, solution is a single tour |
| [test_solver.py](tests/test_solver.py) | results stored per formulation, a valid cut keeps the optimum, `SET` and `RANDOM` seed modes, every run kept, log file named with the seed, missing `log_dir` error, infeasible model |

Tests that need a folder create it in pytest's temporary directory (`tmp_path`), so they
never write to `data/` or `logs/`.

### Next steps
- [ ] `requirements.txt` listing the dependencies
- [x] MILP model for the TSP (`create_model(n, c)`) in gurobipy
- [x] Solver wrapper (`Solver`, Step 4)
- [ ] Cache solver results on disk (`cache/`)
- [ ] Representation of a cut / an individual of the population
- [ ] LLM calls to generate cuts (prompts rewritten for gurobipy)
- [ ] Evolutionary loop (selection, elitism with `top_selected`, mutation and crossover)

---

## Code conventions
- Type hints everywhere: parameters, return types (`-> None` included), class attributes
  and non-obvious local variables.
- Use Python 3.9-compatible `typing` generics: `List`, `Dict`, `Optional`, `Tuple`
  (not `list[...]` or `X | None`).
- Give recurring structures a type alias (`Instance`, `Coords`, `Costs`, ...).
