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
unless you have already exported that variable yourself, or the file does not exist. Without
the file, Gurobi falls back to its other licenses (for example the size-limited pip one)
instead of failing. When the license loads, Gurobi prints `Academic license ... registered to ...`.

### Remote machines
`computer_connection/` (git-ignored, local only) lists the polytechnique lab machines and holds
`check_gpus.sh`, which reports each machine's GPUs, RAM and logged-in users over SSH. It needs
SSH key login: see `computer_connection/README.md`.

Runs are launched **from the Mac** and execute **on the lab machines**. To drive all the machines
at once (start, stop, resume, watch), use `fleet.py` or its web dashboard (Step 13):

```bash
python3 computer_connection/fleet.py serve        # then open http://localhost:8765
python3 computer_connection/fleet.py status       # same information in the terminal
```

For a single machine there is also the older launcher:

```bash
bash computer_connection/run_remote.sh --model_name qwen3-coder:30b --population_size 5 --num_generations 2
```

It syncs the code (refused while another run is in progress there, because the solver's child
processes re-import the code at every solve), starts `main.py` detached on the machine, prints
each generation summary, and fetches `runs/` back to the Mac when the run ends. The results are
pulled by the Mac, not pushed by the lab machine, which cannot reach a laptop on eduroam.
`fetch_runs.sh` fetches them by hand at any time.

### Running on a lab machine
Experiments run entirely on a lab machine (currently **saumon**: RTX 3090, Intel Xeon w3-2435): the
LLM, Gurobi and the genetic algorithm. The setup there, done once:
- the code is copied to `~/My_evo_cut` with `bash computer_connection/sync.sh` (code only: no
  `.venv`, data, logs or license). The home is shared, so this copy serves every lab machine;
- the lab machines only have Python 3.9, and gurobipy 13 needs 3.10 or newer. So
  [uv](https://docs.astral.sh/uv/) (already in `~/.local/bin`) installed Python 3.11 in the home,
  and `.venv` uses it with the **same versions as locally**: Python 3.11, gurobipy 13.0.3,
  PyYAML 6.0.3, pytest 9.1.1:
  `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python gurobipy==13.0.3 PyYAML==6.0.3 pytest==9.1.1`;
- `gurobi.lic` was copied to `~/My_evo_cut/` (`chmod 600`: readable by me only);
- `mkdir -p data/tsp cache logs`. All tests pass there.

### Local LLM (Ollama)
The LLM runs locally with [Ollama](https://ollama.com) on a lab machine with a 24 GB GPU (RTX 3090),
without admin rights. The model is `qwen3-coder:30b` (Qwen3-Coder-30B-A3B-Instruct): code-specialised,
about 45 tokens/s, and it writes no `<think>` block. An SSH tunnel makes it reachable from this
computer at `http://localhost:11434`. The model files stay on the machine's local `/Data` disk, so
they survive reboots. Install details and the start / stop / tunnel scripts are in
`computer_connection/README.md`.

### Folders
The code never creates folders. After cloning, create the git-ignored output folders
yourself:

```bash
mkdir -p data/tsp cache logs runs
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
├── data/                  # datasets, one subfolder per problem (data/tsp/, ...); on the lab
│                          # machines one folder per CPU group: data/rtx3090/tsp/, data/a5000/tsp/, ...
├── cache/                 # cached solver results
├── logs/                  # Gurobi log files, one per solve, in logs/<run>/
├── runs/                  # per EvoCut run: <run>.json (population), .log, .status.json, and
│                          # <run>/gen_<k>/: every LLM proposal of generation k, and
│                          # <run>.review.json: the final review of the best cut (git-ignored)
├── prompts/               # LLM prompts (YAML, versioned)
│   ├── common.yaml        # shared system template + user message templates
│   ├── mutations.yaml     # General, Lifted, Exploratory
│   ├── crossovers.yaml    # Intersection, Complement, Hybrid, Min_Violation
│   └── initializer.yaml   # first generation: new cuts from scratch, baseline as format example
├── gurobi.lic             # Gurobi license (secret, git-ignored)
├── computer_connection/   # remote machines (git-ignored): machines.txt, fleet.py + dashboard.html, scripts
├── .gitignore             # hides secrets, .venv, data/, cache/, logs/, runs/
├── pytest.ini             # test config: run `pytest` from the project root
├── TODO.md                # ideas and known limits to come back to (e.g. relaxing the validity check)
├── tests/                 # one test file per module, updated at every step
└── src/
    ├── utils/
    │   ├── parameters.py  # default values for every setting + project paths
    │   ├── parser.py      # command-line arguments (defaults come from parameters.py)
    │   └── status.py      # RunStatus: runs/<run>.status.json, where a run is (read by fleet.py)
    ├── prepare_instances.py  # script: build + solve the validation and evaluation sets
    ├── main.py            # script: run EvoCut (first generation + generation steps), save each generation
    ├── review_run.py      # script: the final review for a saved run (no LLM)
    ├── loaders/
    │   ├── base.py        # Loader: abstract interface to generate / save / load / split instances
    │   ├── tsp.py         # LoaderTSP: random Euclidean TSP instances
    │   └── sets.py        # build_set: instances solved once with the base model, saved
    ├── models/
    │   ├── tsp.py         # create_model(n, c): TSP MILP (MTZ) in gurobipy + BASELINE cut (2-cycle)
    │   └── transforms.py  # with_cut / relaxed / fixed: problem-independent model transformations
    ├── solvers/
    │   └── solver.py      # Solver: problem-independent, compares formulations, logs, stores results
    ├── llm/
    │   ├── base.py        # LLMClient: abstract, send(messages) -> str with retries
    │   ├── fake.py        # FakeLLM: fixed replies, for tests
    │   └── ollama.py      # OllamaClient: any model served by Ollama (POST /api/chat)
    └── evolution/
        ├── individual.py  # Individual: one cut (idea, code, fitness, generator, parents, id)
        ├── population.py  # Population: fixed-size current generation, history, best known, save/load
        ├── prompts.py     # Prompts: conversations from prompts/*.yaml; parse_reply: IDEA/CODE
        ├── evaluator.py   # CutEvaluator: validity, usefulness, work-based fitness of a cut
        ├── recorder.py    # AttemptRecorder: every LLM proposal saved in runs/<run>/gen_<k>/
        ├── review.py      # review_cuts: best cut vs seed cut vs base under several Gurobi seeds
        └── genetic.py     # GeneticAlgorithm: elitism, tournament, mutation, crossover, generation_step
```

`data/`, `cache/` and `logs/` are resolved as absolute paths from `parameters.PROJECT_ROOT`, so they
work no matter which directory you run from.

---

## Progress

### Step 1: Configuration and command-line arguments
- [src/utils/parameters.py](src/utils/parameters.py) stores the default value of every
  setting: paths, LLM, problem, genetic algorithm and solver. The genetic-algorithm values
  come from the reference config: `POPULATION_SIZE = 8`, `NUM_GENERATIONS = 20`,
  `ELITISM_RATIO = 0.2` (int(0.2 × 8) = 1 elite), `CROSSOVER_RATE = 0.7` (otherwise the child
  comes from a mutation) and `MAX_LLM_ATTEMPTS = 3`.
- `MODEL_NAME` is `None`: the local LLM is chosen with `--model_name`, so several models can
  be compared run by run.
- [src/utils/parser.py](src/utils/parser.py) exposes the settings as CLI flags:
  `--model_name`, `--problem_name`, `--population_file`, `--population_size`,
  `--num_generations`, `--elitism_ratio`, `--crossover_rate`, `--max_llm_attempts` and
  `--cache_dir`. `arg_parser(argv)` reads `sys.argv` by default; tests pass their own list.

### Step 2: Instance loaders
An **instance** is a plain dict, `Instance = Dict[str, Any]`:

```python
{"name": "rand_s0_3_12", "args": {...}, ...}   # "args" are passed to create_model(**args)
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
- Names are `rand_s<seed>_<idx>_<n>`. The seed makes a name unique across instance sets:
  results, log files and caches are keyed by name, so `train.pkl` (seed 0) and `test.pkl`
  (seed 1) must not both contain a `rand_0_10`. Without a seed, one is drawn at random and put
  in the name, so the instance can still be regenerated.

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
  `runtime`, `node_count`, `sol_count`, `obj_val`, `obj_bound`, `mip_gap` and `error`. A value
  Gurobi cannot give is `None` instead of crashing: the objective and gap when no solution
  was found, and the bound of an infeasible or unbounded model.
- Every model gets the same `TimeLimit` and `Threads`. The defaults in `parameters.py` are
  8 threads (from the reference config) and a 60 s safety time limit. Cut scoring is limited by
  work instead (`WORK_LIMIT`, Step 9).
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
  `gurobipy`, so the Solver quiets that logger unless `verbose` is on. These console settings
  are applied at **every solve**, not once when the Solver is created. A second Solver
  therefore cannot change the verbosity of the first one.
- The `logs/` folder must already exist; otherwise the Solver raises a clear error. Pass
  `log_dir=None` to write no log files.

**Safety: every solve runs in its own process.** A formulation will soon contain code written
by the LLM, which can be wrong in many ways. With `isolated=True` (the default, `parameters.ISOLATED`),
the Solver builds and solves each model in a fresh child process (`multiprocessing`,
"spawn" start method), and the result comes back through a pipe. Each failure becomes a
status, and the run continues:

| Status | When | Example |
|---|---|---|
| `ERROR` | the formulation raises, or does not return a model | the cut uses a wrong index: `KeyError` |
| `TOO_LARGE` | more than `max_constraints` (default 10⁶) after building, checked before `optimize()` | a cut enumerating 2ⁿ subsets |
| `KILLED` | the child is still running after `time_limit + build_timeout` (10 + 60 s) | the building code loops forever |
| `CRASHED` | the child dies without sending a result | out of memory |

`SolveResult.error` holds the message (for example `"KeyError: 'bad index'"`). Later, it can
be shown to the LLM so that it fixes its code. Gurobi's `TimeLimit` alone could not catch
these cases, because the model is built *before* `optimize()` starts.

**Timing is unchanged.** `runtime` is Gurobi's own `Runtime`, which measures `optimize()`
only. Starting the process (about 0.3 s, mostly loading Python and gurobipy and fetching the
license) and building the model are not counted. `build_and_solve` is the same function in
both modes, and every formulation of a Solver uses the same mode, so `"base"` and
`"base+cut"` are always timed the same way. A test checks that a formulation taking 1 s to
build still reports a runtime under 0.5 s.

Two consequences of "spawn":
- Formulations must be **picklable**: top-level functions or a `functools.partial` of one,
  not lambdas or functions defined inside other functions. `add_formulation` checks this
  right away.
- A script that uses the Solver needs an `if __name__ == "__main__":` guard, because the
  child process imports the script.

`isolated=False` runs everything in the current process. It is faster for quick
experiments, but has no timeout and no protection against crashes; the `TOO_LARGE` check
still applies. The child process does **not** sandbox the code: the LLM code could still
read or write files. Restricting what the code can do belongs to the cut-evaluation step.

```python
import logging
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from solvers.solver import SeedMode, Solver


def with_2cycle_cut(n, c):  # a valid cut: never use both i->j and j->i (top-level: picklable)
    m = create_model(n, c)
    m.addConstrs((m._x[i, j] + m._x[j, i] <= 1 for (i, j) in m._x.keys() if i < j), name="cut")
    return m


if __name__ == "__main__":  # required: child processes import this file
    logging.basicConfig(level=logging.INFO)
    solver = Solver()
    solver.add_formulation("base", create_model)
    solver.add_formulation("base+2cycle", with_2cycle_cut)
    instances = LoaderTSP().get("demo.pkl", n_instances=3, seed=0, min_cities=10, max_cities=20)
    solver.solve_all(instances)
    print(solver.results["base+2cycle"]["rand_s0_0_16"][0].runtime)

    # many seeds: 5 solves per instance, all kept
    multi = Solver(seed_mode=SeedMode.RANDOM)
    multi.add_formulation("base", create_model)
    for _ in range(5):
        multi.solve_all(instances)
    print([r.seed for r in multi.results["base"]["rand_s0_0_16"]])
```

Checked: on 3 instances both formulations reach the same optimum. An 80-city instance with
`time_limit=1` returns `TIME_LIMIT` with an 11.9% gap. An infeasible model returns
`INFEASIBLE` with `None` values instead of crashing, and an unbounded model returns
`UNBOUNDED` / `INF_OR_UNBD` instead of the former `AttributeError` on `ObjBound`.

### Step 5: Individuals and population
The evolutionary part is split into three classes, each with a single job. This step builds
the first two. Neither one calls the LLM or the solver, so both can be tested in milliseconds.

- [src/evolution/individual.py](src/evolution/individual.py) defines `Individual`, a
  dataclass for one candidate cut:
  - `idea` (natural-language description) and `code` (gurobipy code adding the cut);
  - `fitness`: higher is better, `None` until the cut is evaluated (`ind.evaluated`);
  - `generator`: what created it. `INIT` for the first generation, `ELITE` for an elite
    carried over, `COPY` for a parent copied because every LLM attempt failed, otherwise the
    name of the mutation or crossover;
  - `parents` (their ids) and a unique `id`;
  - `copy(ELITE)` / `copy(COPY)`: a **new** individual with the same idea, code and fitness,
    a new `id` and `parents=[original.id]`. Elites and fallback copies are always made this
    way, never by reusing the object. Otherwise the same cut would appear twice in a
    generation, tournament selection would favour it, and the history would not say why it
    was carried over. The fitness is copied, **not re-evaluated**: this is simpler, and the
    solver is not run again on a cut it already measured. The catch: with a noisy fitness, an
    elite keeps a lucky value for as long as it stays elite;
  - `to_dict` / `from_dict` for JSON.
- [src/evolution/population.py](src/evolution/population.py) defines `Population(size)`:
  - `current`: the current generation. `history`: the previous ones. `generation`: index of
    the current one (0 = first).
  - `set_generation(individuals)` is **the only way to change the population**. It refuses
    a list that does not have exactly `size` individuals, that contains an unevaluated
    individual, or that holds the same `id` twice. The generation size can therefore never
    drift. The previous generation moves to
    `history`.
  - `best`: the best individual ever seen, possibly from an old generation. On a tie, the
    older best is kept.
  - `top(k)`: the `k` fittest individuals of the current generation (for elitism).
  - `all_individuals()`: every individual tried so far, skipping copies (`ELITE`, `COPY`)
    because they repeat a cut already listed. Later, this will tell the LLM which ideas
    were already tried.
  - `save(path)` / `Population.load(path)` write and read back the whole state as JSON,
    so an interrupted run can be resumed.

The third class, `GeneticAlgorithm`, will hold the settings (size, elitism, mutation and
crossover rates) and the logic. It fills each of the `size − elites` slots with exactly one
child, and copies a parent if the LLM keeps failing.

```python
from evolution.individual import ELITE, Individual
from evolution.population import Population

pop = Population(size=2)
pop.set_generation([Individual("2-cycle", "...", fitness=0.3), Individual("degree", "...", fitness=0.1)])
pop.set_generation([pop.top(1)[0].copy(ELITE), Individual("comb", "...", fitness=0.2)])  # 1 elite + 1 child
pop.best.idea, pop.generation          # ('2-cycle', 1)
pop.save("../cache/pop.json")
```

### Step 6: Mutation and crossover prompts
The prompts live in [prompts/](prompts/) as YAML files, outside `src/`: they are text, not code.
`parameters.PROMPTS_DIR` points to this folder. They are translated from the reference
(`configs/prompts/mutations.yaml` and `cross_overs.yaml`) and rewritten for gurobipy.

**Shared text in one place.** In the reference, all 7 prompts repeat the same role, rules
and response format. Here that text is written once, in
[prompts/common.yaml](prompts/common.yaml), and each operator file only holds its own task:

| File | Keys | Used for |
|---|---|---|
| [common.yaml](prompts/common.yaml) | `system`, `mutation_user`, `crossover_user` | templates shared by every operator |
| [mutations.yaml](prompts/mutations.yaml) | `General`, `Lifted`, `Exploratory` | one individual → one new cut |
| [crossovers.yaml](prompts/crossovers.yaml) | `Intersection`, `Complement`, `Hybrid`, `Min_Violation` | two parents → one new cut |

A conversation with the LLM is built like this (the `Prompts` class of Step 7 does it). The
templates use Python `str.format`, so a literal brace is written `{{` or `}}`:

```python
system = common["system"].format(task=mutations["Lifted"], model_code=inspect.getsource(create_model))
user = common["mutation_user"].format(best_fitness=pop.best.fitness, idea=ind.idea, fitness=ind.fitness, code=ind.code)
user = common["crossover_user"].format(best_fitness=pop.best.fitness, idea1=..., fitness1=..., code1=..., idea2=..., fitness2=..., code2=...)
```

`model_code` is the real source of `create_model`, read with `inspect.getsource`. The LLM
therefore always sees the model we actually solve, and no copy of it can drift out of date
(the reference pastes a copy of the model into `tsp.yaml`).

**How the cut code is used (gurobipy).** The reference pastes the Pyomo lines inside
`create_model`. Here the code runs **after** `create_model` has returned. These names are
already defined: `m` (the model), `gp` and `GRB`, and every argument of `create_model` under the
same name (`n`, `c` for the TSP). Variables are reached through the model attributes (`m._x`,
`m._u`). This rule is the same for every problem. The cut-evaluation step will implement
exactly this.

**Other differences from the reference:**
- **Response format:** an `IDEA:` paragraph followed by a ```` ```python ```` block, instead of a JSON
  object whose `"code"` string itself contains a code block. Code inside a JSON string needs
  every newline and quote escaped, and small local LLMs often get that wrong.
- **Fitness in context:** a fitness value means nothing on its own (its scale depends on
  the instances and on how fitness is defined). Every user message therefore starts with the
  **best fitness found so far** (`Population.best.fitness`), and the system prompt tells the
  LLM to judge a cut against it. The mutation prompt also shows the individual's own fitness,
  which the reference shows only for crossovers.
- **`Min_Violation`:** the reference asks to minimize "the joint violation of both parents
  in prior runs", but never passes any violation data. Here it asks the LLM to keep the
  safest parts of each parent and drop the risky ones.

### Step 7: LLM client and genetic algorithm
The design separates three jobs, so that each can change without touching the others:

| Piece | Job | Knows about |
|---|---|---|
| `LLMClient` ([src/llm/base.py](src/llm/base.py)) | send a conversation, return the text | nothing about cuts or prompts |
| `Prompts`, `parse_reply` ([src/evolution/prompts.py](src/evolution/prompts.py)) | build conversations from `prompts/*.yaml`, read the reply | our `IDEA:`/`CODE:` format |
| `GeneticAlgorithm` ([src/evolution/genetic.py](src/evolution/genetic.py)) | evolve the population | parents, operators, fitness |

**LLM client.** `LLMClient` is an abstract class. A backend only implements
`_send(messages) -> str`, which makes one call. `send(messages)` adds what every backend
needs:
- **Retries** for failed *calls* (server down, timeout): `LLM_MAX_RETRIES = 3`, waiting 1 s,
  then 2 s, then 4 s. After the last try it raises `LLMError`.
- **A `clean(text)` hook** for model quirks. It does nothing by default; a reasoning model's
  subclass will remove its `<think>` block there.

A message is `Message = Dict[str, str]`, for example `{"role": "user", "content": "..."}`. No real
backend exists yet, because the local LLM is still to be chosen.
[src/llm/fake.py](src/llm/fake.py) defines `FakeLLM(replies)`. It returns the given replies in
order and records every conversation in `received`, so the tests run without an LLM.

**Prompts and parsing.** `Prompts(model_code)` loads the three YAML files and builds:
- `mutation_messages(operator, parent, best_fitness)` → `[system, user]`;
- `crossover_messages(operator, parent1, parent2, best_fitness)` → `[system, user]`;
- `retry_message(error)`: a user message from the new `retry_user` template, saying why the
  previous answer was rejected.

`parse_reply(text) -> (idea, code)` reads the `IDEA:` / `CODE:` answer. It accepts the
variations small models often produce: bold labels (`**IDEA:**`), text before and after, a
code block without `python` or without backticks, and indented code. A missing or empty
section raises `ReplyFormatError`, whose message is meant to be sent back to the LLM.

**Genetic algorithm.** `GeneticAlgorithm(population, llm, prompts, evaluate, ...)`, with settings
`elitism_ratio`, `crossover_rate`, `max_llm_attempts`, `tournament_size` and `seed`:
- `generation_step()` builds the next generation: `elitism()`, then `make_child()` until the
  generation is full, then `population.set_generation(...)`.
- `elitism()` returns copies (`ELITE`) of the `int(elitism_ratio × size)` best individuals.
- `make_child()` fills one slot. With probability `crossover_rate` it picks 2 parents and a
  random crossover operator, otherwise 1 parent and a random mutation operator. If the LLM
  fails, it returns a copy (`COPY`) of the first parent. With a population of 1 it always
  mutates.
- `select_parents(k)` is tournament selection: each parent is the fittest of
  `tournament_size` (2) individuals drawn at random, and the `k` parents are distinct.
- `mutation(parent, operator)` and `crossover(parent1, parent2, operator)` return the new
  `Individual`, or `None` if the LLM failed. `operator` is a key of the YAML files
  (`"Lifted"`, `"Hybrid"`, ...), so a new prompt variant needs no new code. It is stored as
  the child's `generator`.
- **Attempts:** each child gets up to `max_llm_attempts` (3) answers, **in one
  conversation**. When an answer is rejected (wrong format, or `CutRejected` raised by
  `evaluate`), the reason is appended with `retry_message` and the LLM tries again with its
  previous answer in view. If the LLM cannot be reached (`LLMError`), the GA stops at once,
  because asking again would not help.
- `evaluate: Evaluator = Callable[[str], float]` turns a cut's code into its fitness, and
  raises `CutRejected(reason)` when the cut cannot be used. The cut-evaluation step will
  implement it (run the code, check validity, solve with `Solver`). For now the tests pass a
  dummy one.
- Every random choice uses `random.Random(seed)`, so a run is reproducible.

The first generation is not built here yet: it needs the initializer prompt.
`generation_step()` raises `RuntimeError` on an empty population.

```python
import inspect
from evolution.genetic import CutRejected, GeneticAlgorithm
from evolution.individual import Individual
from evolution.population import Population
from evolution.prompts import Prompts
from llm.fake import FakeLLM
from models.tsp import create_model

pop = Population(size=2)
pop.set_generation([Individual("2-cycle", "m.addConstr(...)", fitness=0.3),
                    Individual("degree", "m.addConstr(...)", fitness=0.1)])

def evaluate(code: str) -> float:  # placeholder until the cut-evaluation step
    if "bad" in code:
        raise CutRejected("the cut removes the optimal solution")
    return 0.5

llm = FakeLLM(["IDEA: lifted 2-cycle\nCODE:\n```python\nm.addConstr(...)\n```"])
ga = GeneticAlgorithm(pop, llm, Prompts(inspect.getsource(create_model)), evaluate,
                      elitism_ratio=0.5, seed=0)
ga.generation_step()
[(ind.generator, ind.fitness) for ind in pop.current]   # [('elite', 0.3), ('Lifted', 0.5)]
```

### Step 8: Ollama client and the two instance sets
**Ollama client.** [src/llm/ollama.py](src/llm/ollama.py) defines `OllamaClient(model, host, options,
timeout)`, a subclass of `LLMClient`. Its `_send` posts the conversation to `<host>/api/chat`
(no streaming) and returns the answer text. It uses Python's built-in `urllib`, so there is no
new dependency.
- **Model-agnostic, at two levels.** Ollama serves many open models through one API, so changing
  model only means changing `model` (`"qwen3-coder:30b"`, `"qwen3:32b"`, ...). And the GA only
  calls `LLMClient.send`, so a backend other than Ollama would only need another small subclass.
- `options` are passed to Ollama unchanged, for example `{"temperature": 0.7, "seed": 0}`.
- Any HTTP or network error (unknown model → 404, server down) raises an exception, so
  `LLMClient` retries it and then raises `LLMError`. A cut-off answer (`done_reason == "length"`)
  is logged as a warning.
- New settings: `OLLAMA_HOST = "http://localhost:11434"` (the SSH tunnel) and `LLM_TIMEOUT = 600` s.

```python
from llm.ollama import OllamaClient
llm = OllamaClient("qwen3-coder:30b", options={"temperature": 0.7})
llm.send([{"role": "user", "content": "Write a gurobipy constraint x + y <= 1."}])
```

**Saving solutions.** The validity check needs the optimal solutions, so the `Solver` has a new
option, `keep_solution=True`. It stores the best solution in `SolveResult.solution` as
`{variable name: value}`, for example `{"x[1,2]": 1.0, "u[2]": 5.0, ...}`. This works for any
problem, and in both isolated and in-process mode.

**The two instance sets.** As in the reference (2 verification + 8 evaluation instances):

| Set | File | Size | Solved | Used for |
|---|---|---|---|---|
| validation | `data/tsp/validation.pkl` | `N_VALIDATION = 2` | to **optimality** (`VALIDATION_TIME_LIMIT = 600` s), solution kept | a valid cut must keep these optimal solutions |
| evaluation | `data/tsp/evaluation.pkl` | `N_EVALUATION = 8` | under `TIME_LIMIT` (10 s) | scoring the cuts against the base model |

- [src/loaders/sets.py](src/loaders/sets.py) defines `build_set(loader, filename, n, seed, solver,
  require_optimal, **gen_kwargs)`. It generates the instances, solves each one with the solver's
  `"base"` formulation, stores the `SolveResult` in `instance["base"]`, and saves. If the file
  already exists, it is loaded instead (delete it to rebuild). With `require_optimal=True`, an
  instance not solved to optimality raises an error and nothing is saved.
- The two sets use different seeds (`VALIDATION_SEED = 1`, `EVALUATION_SEED = 2`), so they never
  share an instance; the names show it (`rand_s1_...`, `rand_s2_...`).
- [src/prepare_instances.py](src/prepare_instances.py) is the script. It builds both sets and
  prints a summary. Flags: `--n_cities` (default `N_CITIES = 15`), `--n_validation`,
  `--n_evaluation`, `--validation_time_limit`, `--time_limit`.

```bash
cd src && python prepare_instances.py
```

*(Step 10 replaced the evaluation set with larger instances.)* Result with 15 cities: all 10
instances are `OPTIMAL` in 0.01–0.1 s, and **5 of the 8 evaluation instances are solved at the
root node** (1 node). So the base gap is always 0, and the reference
score (relative gap) cannot be used on these instances. Scoring needs either larger evaluation
instances or another measure (for example Gurobi's deterministic work units). This is decided in
the evaluation step.

### Step 9: Cut evaluation (work-based fitness)
Three decisions:
1. **Everything runs on the lab machine** (see Setup), from the first run on.
2. **Fitness uses Gurobi work units, not time.** `Work` measures the solver's effort and is
   **exactly reproducible**: the same model with the same `threads` gives the same value on every
   run, while the runtime changes from run to run. The runtime, the machine and the CPU are still
   recorded with every solve.
3. **Validity is checked as in the reference** (`CutVerifier.verify_score_cut`).

**Solver additions** ([src/solvers/solver.py](src/solvers/solver.py)):
- `work_limit`: stop after this much work (`Params.WorkLimit`). It gives the same result on every
  run, unlike `time_limit`, which stays as a safety net (`TIME_LIMIT = 60` s, `WORK_LIMIT = 10`).
- `SolveResult.work` (Gurobi's `Work`), `threads`, `machine` and `cpu`. `machine_info()` returns
  `("saumon", "Intel(R) Xeon(R) w3-2435")`; on a Mac it asks macOS for the computer name,
  because the network name is returned otherwise.
- **Measured: work depends on the threads and on the CPU type.** On the same instance, 8 threads
  give 0.107 work units on the Mac (Apple M4) but 0.191 on saumon (Xeon). The optimal values are
  the same, but the search differs. So the base results and the cut results must come from the
  **same machine type and the same `threads`**. This is why everything is recorded, and why the
  instance sets are built on saumon.

**Model transformations** ([src/models/transforms.py](src/models/transforms.py)). These are
problem-independent and combine with `functools.partial` into picklable formulations:
- `with_cut(base, code, **args)`: `base(**args)`, then `exec(code)` with `m`, `gp`, `GRB` and the
  arguments (`n`, `c`) defined, exactly as promised in the prompt;
- `relaxed(build, **args)`: the LP relaxation (`Model.relax()`, same variable names);
- `fixed(build, solution, **args)`: every variable of `solution` fixed by name (integer values
  rounded, values clipped to the bounds); variables the solution doesn't have (a cut's auxiliary
  variables) stay free.

**The evaluator** ([src/evolution/evaluator.py](src/evolution/evaluator.py)).
`CutEvaluator(base, validation, evaluation, solver)` is the `evaluate(code) -> fitness` of the GA:
1. **Validity:** on every validation instance, the model with the cut, with its variables fixed to
   the base optimum, must stay feasible. Otherwise: `CutRejected("invalid cut: it removes the
   optimal solution of instance ...")`.
2. **Usefulness:** on at least one validation instance, the LP relaxation with the cut, fixed to
   the base LP optimum, must be infeasible (the cut removes the LP optimum). Otherwise:
   `CutRejected("useless cut: ...")`. This is why the validation set now also stores
   `instance["base_lp"]`, the LP optimum, solved once in `prepare_instances.py`.
3. **Score** *(replaced in Step 10 by the gap at the work limit, on larger instances)*: every
   evaluation instance is solved with the cut under the same settings as its base result, then
   $$\text{fitness} = 10 \cdot \exp\Big(-\tfrac{1}{N}\sum_i \tfrac{\text{work}_{\text{cut},i} - \text{work}_{\text{base},i}}{\text{work}_{\text{base},i}}\Big)$$
   This is the reference formula with work in place of the gap: 10 = no change, above 10 = less
   work (−20% on average → 12.2). **Added check:** if an evaluation instance's optimal value
   changes, the cut is invalid. This is free (the solve is done anyway) and catches invalid cuts
   that the 2 validation instances miss.
- Every solve goes through the `Solver`, so the LLM code runs in the child process, with the
  timeout and the size cap. A solver failure (`ERROR`, `KILLED`, ...) becomes a `CutRejected`
  carrying its message.
- `build_set(..., formulations=[...])` can now solve several formulations per instance.

On saumon (15 cities, `WORK_LIMIT = 10`, 8 threads): the 2-cycle cut gets **fitness 12.8**
(**−24.7%** work on average), evaluated in 1.2 s (12 solves, each in its own process). The
constraint `sum(x) <= n` is rejected as **useless**, since the degree constraints already imply it.

**Limits to keep in mind:**
- The reference validity check fixes the **exact** optimal tour, including its direction. A cut
  that forbids only one direction of the optimal tour is rejected, even though the reversed tour
  (same cost, symmetric TSP) is still feasible. Symmetry-breaking cuts are therefore rejected
  too. (The tests ran into this: forbidding one arc of an optimal tour does **not** change the
  optimal value, because the reversed tour takes over.)
- If an evaluation instance hits the work limit, its work equals the limit with or without the
  cut, so it tells nothing. The reference uses the gap in that case. With 15 cities every
  instance is solved well below the limit.

```python
from evolution.evaluator import CutEvaluator
from loaders.tsp import LoaderTSP
from models.tsp import create_model
from solvers.solver import Solver
from utils import parameters

if __name__ == "__main__":   # a real .py file: the Solver's child processes import it
    loader = LoaderTSP()
    evaluate = CutEvaluator(create_model, loader.load("validation.pkl"), loader.load("evaluation.pkl"),
                            Solver(work_limit=parameters.WORK_LIMIT))
    evaluate("x = m._x\nm.addConstrs(x[i, j] + x[j, i] <= 1 for (i, j) in x.keys() if i < j)")  # 12.797
```

### Step 10: Larger evaluation instances, gap-based score
With 15 cities the base model solves every evaluation instance almost instantly, so there is
almost nothing for a cut to improve. The evaluation set now uses **60–100 cities**, and keeps only
instances that the base model **cannot finish** within the work limit. Cuts are then scored by
the **gap** they close at that same work limit. This is the reference's score (`cal_fitness`),
with a work limit instead of a time limit, so it stays reproducible.

Measured on saumon (`WORK_LIMIT = 10`, 8 threads): 60 cities still solve (about 2 work units),
while 80–100 cities stop at the limit with 3–12% gap left.

- **Settings:** `VALIDATION_CITIES = 15` (validation instances must still be solved to
  optimality), `EVALUATION_MIN_CITIES = 60`, `EVALUATION_MAX_CITIES = 100`,
  `MIN_BASE_GAP = 0.002`, `EVALUATION_SEED = 100`. The gap threshold is the reference's
  `GAP_LOWERBOUND = 0.2`, which is in percent because the reference reads the gap from the
  Gurobi log.
- **`build_set(..., accept=...)`:** solved instances failing `accept` are replaced by new ones,
  generated with the next seeds (`seed + 1`, ...), until `n` are kept (at most `MAX_BATCHES = 20`
  rounds). `prepare_instances.py` passes `has_gap(MIN_BASE_GAP)`. New flags:
  `--validation_cities`, `--evaluation_min_cities`, `--evaluation_max_cities`, `--min_base_gap`.
- **Score:**
  $$\text{fitness} = 10 \cdot \exp\Big(-\tfrac{1}{N}\sum_i \tfrac{\text{gap}_{\text{cut},i} - \text{gap}_{\text{base},i}}{\text{gap}_{\text{base},i}}\Big)$$
  10 = same gap, more = smaller gap, and every instance solved gives 10·e ≈ 27.2. If the cut
  model finds no solution within the limit, its gap counts as 100% (`NO_SOLUTION_GAP`), which
  gives a fitness near 0.
- **New invalidity check** (it replaces "the optimal value changed", since the base no longer
  finishes): if the cut's **lower bound** goes above the cost of a solution the base model
  found, the cut removed every solution at least that good, the optimum included. This assumes
  minimization.
- `CutEvaluator` refuses an evaluation instance whose base gap is not > 0.

Result on saumon: the 8 evaluation instances have 65–97 cities and base gaps of 1–26%, built in
59 s (2 instances got solved and were replaced). The 2-cycle cut gets **fitness 11.9** (the gap
shrinks by 17% on average). The effect differs a lot between instances: 74 cities, 13.5% → 2.5%;
65 cities, fully solved; but 92 cities, 21.8% → 33.9%. **One cut evaluation takes about 53 s**
(8 solves of about 6.5 s).

### Step 11: First generation (seed + initializer)
The reference builds the first generation entirely from scratch (`initialize_population`). For
each slot, it sends the `initializer` prompt plus the ideas accepted so far, and loops until the
LLM succeeds. Here the first generation is a **hybrid**:
- **slot 1, the seed:** a hand-written baseline cut, evaluated like any other (generator `SEED`);
- **every other slot:** a cut from scratch by the initializer prompt (generator `INIT`). The seed
  is shown **only as a format example**, and the prompt asks for **maximum diversity**.

This gives the LLM a working example of the code style (`m._x`, `addConstrs`), which matters for
a 30B local model, without making the whole generation variants of one idea.

- **Baseline** ([src/models/tsp.py](src/models/tsp.py)): `BASELINE_IDEA` / `BASELINE_CODE`, the
  2-cycle cut `x[i, j] + x[j, i] <= 1`. It lives next to the model because it is problem
  knowledge. Fitness on saumon: 11.9.
- **Prompt** ([prompts/initializer.yaml](prompts/initializer.yaml)): `task` (the `{task}` of the
  shared system template) and `user` (the reference cut, the best fitness so far, and the ideas
  already in the generation). It insists on diversity: a "fundamentally different" cut, "another
  family of inequalities, other variables, another argument"; variants of a listed idea are
  "worthless"; the reference cut must not be reused, rephrased or modified.
  `Prompts.initializer_messages(reference, ideas, best_fitness)` builds the conversation.
- **`GeneticAlgorithm.initialize(baseline)`** evaluates the baseline (a rejected baseline raises
  `ValueError`) and makes it the seed. It then asks for `size - 1` new cuts, with the same
  attempt and feedback loop as mutations. Each request lists the ideas already in the generation,
  so each new cut has to differ from all the earlier ones. A slot where every attempt fails gets
  a copy of the seed (`COPY`): the generation is always full, whereas the reference loops
  forever if the LLM never succeeds. `initialize` on a non-empty population raises.
- **Names available to the cut code:** `itertools` and `math` were added (in `with_cut` and in the
  prompt). The prompt also says that the code runs on **large** instances (about 100 cities), so
  it must add a polynomial number of constraints and never enumerate all subsets.

A test with `qwen3-coder:30b` produced two different directions, which shows why the checks are
needed:
- "at least one arc between every pair" is **invalid** (no tour does that), so the validity check
  rejects it and tells the LLM why;
- full subtour elimination is a classic, strong idea, but it **enumerated all subsets** and used
  `itertools` without importing it. That prompted the two changes above.

```python
from evolution.individual import Individual
from models.tsp import BASELINE_CODE, BASELINE_IDEA

ga.initialize(Individual(BASELINE_IDEA, BASELINE_CODE))   # generation 0: seed + size-1 new cuts
ga.generation_step()                                      # generation 1, ...
```

### Step 12: The main script
[src/main.py](src/main.py) runs EvoCut from start to end. Run it on the machine that built the
instance sets, with Ollama running there:

```bash
cd src
python main.py --model_name qwen3-coder:30b --population_size 8 --num_generations 20
python main.py --model_name qwen3-coder:30b --population_file ../runs/<run>.json   # resume
```

`run(args, llm=None)` does the following:
1. Load `validation.pkl` and `evaluation.pkl`. If they are missing, it says to run
   `prepare_instances.py` first.
2. **Check the machine:** the evaluation set's base results must come from the same CPU and the
   same `--threads` as this run. Gurobi work depends on both, so otherwise the gaps cannot be
   compared and the run refuses to start (`check_same_machine`).
3. Build the `OllamaClient`. It checks that `--model_name` is on the server
   (`available_models()`, i.e. `GET /api/tags`) and passes `--temperature` if given.
4. Build the `Solver` (`--work_limit`, `--threads`), the `CutEvaluator`, the `Prompts` (from the
   problem's `create_model` source) and the `GeneticAlgorithm` (`--elitism_ratio`,
   `--crossover_rate`, `--max_llm_attempts`, `--seed`).
5. Load the population from `--population_file` if it exists; otherwise start a new one. If it
   is empty, `initialize(baseline)`. Then `generation_step()` until generation
   `--num_generations`. **The population is saved after every generation**, so an interrupted run
   can be resumed.
6. Log a summary line per generation (best fitness ever, current max and mean, where the
   individuals come from, elapsed time), then the best cut.

- **Problems:** `PROBLEMS = {"tsp": Problem(LoaderTSP, create_model, BASELINE_IDEA, BASELINE_CODE)}`
  holds everything problem-specific. A new problem only needs one more entry.
- **Outputs:** `runs/<problem>_<model>_<host>_<date>_<time>.json` (the population) and `.log`
  (the full log); since Step 13 also `.status.json` and the Gurobi logs in `logs/<run>/`.
  The console shows everything except the solver's per-solve lines, which go to the log
  file only. The code never creates folders: `mkdir runs` once.
- **New settings / flags:** `--seed` (`GA_SEED`), `--temperature` (`TEMPERATURE`), `--ollama_host`,
  `--work_limit`, `--threads`, `--data_dir`, `--runs_dir` (`RUNS_DIR`). The last four exist mostly
  for the tests.

**First real run on saumon** (first generation only, 2 individuals, `qwen3-coder:30b`), 1 min 16 s:
the seed was evaluated (fitness 11.9, 52 s of solving). For the second slot the LLM made 3
attempts in one conversation: twice *useless*, once *invalid* (each check about 0.4 s, since the
validity and usefulness checks run on the 15-city instances). The slot then got a copy of the
seed. The mechanics work; **the LLM's success rate is the thing to watch**. Since a rejected
attempt is cheap (about 7 s of LLM plus 0.4 s of checks), a larger `--max_llm_attempts` costs
little.

### Step 13: Runs on every lab machine (fleet)
Lab machines can be cleaned up by the administrators at any time, e.g. before classes. So runs
are started **by hand when the rooms are free**, report where they are, and can be **resumed**.
Everything is driven from the Mac by `computer_connection/fleet.py` (git-ignored with its folder,
standard library only), in the terminal or through a small web dashboard:

```bash
python3 computer_connection/fleet.py serve                 # dashboard on http://localhost:8765
python3 computer_connection/fleet.py status                # table of every machine
python3 computer_connection/fleet.py start all -- --population_size 8 --num_generations 20
python3 computer_connection/fleet.py stop all              # e.g. before classes
python3 computer_connection/fleet.py resume all            # continue the stopped / killed runs
python3 computer_connection/fleet.py prepare a5000         # build a group's instance sets
python3 computer_connection/fleet.py sync                  # code -> shared home (refused if a run is alive)
python3 computer_connection/fleet.py fetch                 # runs/ -> ./runs on the Mac
python3 computer_connection/fleet.py watch                 # fetch the logs when a run ends
```

**Logs come back by themselves:** while the dashboard (`serve`) or `watch` is running, a run
that goes from running to finished / stopped / failed / dead triggers a fetch of `runs/` to
`./runs` on the Mac (its `.log`, `.json`, `.status.json` and launch `.out`) and a macOS
notification ("allemagne: tsp_… finished, gen 20/20, best 14.948"). A dead run is fetched too,
to keep whatever it saved. A machine that becomes unreachable is not treated as ended. The Mac
must be on (and the dashboard or `watch` running) to catch the end. Otherwise the next `fetch`
gets everything, since nothing is deleted on the machines.

Targets are `all`, a group name or host names. The dashboard shows the same table (probed every
minute, page refreshed every 3 s) with:
- a **Run settings** form (problem, population size, generations, elitism ratio with the number
  of elites it gives, crossover rate, LLM attempts, temperature, seed), filled with
  `parameters.py`'s defaults. The model and the instance sets come from the machine;
- **one action bar** for the selected machines (Start, Resume, Stop, Prepare data) plus Sync code,
  Fetch runs and Probe now. Machines are selected by clicking rows, a whole group, or the quick
  selections *all / free / resumable / running / none*;
- what each action is doing **right now** in the machine's row (e.g. "starting Ollama +
  qwen3-coder:30b"), and the result of every action at the bottom.

A machine accepts **one action at a time**: a second Start while the first one is still starting
is refused (10 clicks once launched 10 runs on allemagne). Starts in progress count against the
license limit like the runs already alive. The dashboard listens on `localhost` only, and its
buttons need a random token that only the served page knows.

**Machines, groups and models** (`computer_connection/machines.txt`): each machine gets the best
LLM for its GPU memory, and each **CPU group** has its own instance sets in `data/<group>/`,
because Gurobi work depends on the CPU (`check_same_machine` refuses other sets).

| group | machines | GPU | CPU | model |
|---|---|---|---|---|
| `rtx3090` | 13 | RTX 3090, 24 GB | Xeon w3-2435 | `qwen3-coder:30b` |
| `a5000` | 14 | RTX A5000, 24 GB | Xeon W-1270P | `qwen3-coder:30b` |
| `ada4000` | 8 | RTX 4000 Ada, 20 GB | i7-14700K | `devstral:24b` |
| `a4000` | 2 | RTX A4000, 16 GB | i7-13700K | `qwen2.5-coder:14b` |

**How it works.**
- **Progress reports.** The lab machines cannot reach the Mac (NAT), so the Mac asks them. Every
  run writes `runs/<run>.status.json` ([src/utils/status.py](src/utils/status.py)), rewritten
  atomically before each child and after each generation: state, generation done, generation
  being built and child number, best and mean fitness, pid, arguments, population file.
  `GeneticAlgorithm(progress=...)` is the hook: it is called before each child. One SSH call
  per machine (all in parallel, about 2 s for 37 machines) reads it, and checks that the process
  is alive, our Ollama server, GPU memory (ours / other users'), and who is logged in.
- **States:** `running`, `preparing`, `finished`, `stopped` (SIGTERM or Ctrl-C: `main.py` turns
  SIGTERM into a clean stop and records it), `failed` (exception, with its message), `dead` (the
  status says running but the process is gone, e.g. killed with SIGKILL), `idle`, `unreachable`.
- **Gurobi license: at most 2 runs at once.** The WLS academic license allows 2 concurrent
  sessions (tested: of 10 machines opening one together, 3 succeeded), and every solve opens
  one. start / resume therefore refuse to go beyond `--max_runs 2` runs alive in total. The Mac
  shares the same 2 sessions: don't run the tests while 2 runs are going. Using all 37 machines
  needs more licenses (see TODO.md).
- **Busy machines** (someone at the screen, i.e. a `seat0` session in `loginctl`, or more than
  1 GB of GPU used by another user) are skipped by start / resume unless `--force`.
- **start** starts Ollama with the machine's model (downloaded into `/Data` the first time), then
  `main.py --model_name <model> --data_dir ../data/<group> <your arguments>`, detached with
  `setsid` (own session: it survives the SSH logout). When `main.py` ends, Ollama is stopped
  to free the shared GPU.
- **stop** sends SIGTERM to the run's whole process group (the solver's children too), waits up
  to 30 s, then SIGKILL, and stops Ollama.
- **resume** relaunches the run's own arguments with `--population_file <its population>`. A
  resumed run keeps the name of its population file, so it continues the same `.json`, `.log`
  and `.status.json`. Only the generation in progress is lost.
- **Shared home.** The code, venv, `data/`, `runs/` and `logs/` are the same files on every
  machine, which is why run names now contain the host, and each run has its own Gurobi log
  folder `logs/<run>/`. It is also why `sync` is refused while any run is alive.

Tested on ablette: start, a status every 45 s (`gen 0/1 (building 1: child 2/3)`), stop during
generation 0 (state `stopped`, no process left), resume, finished, Ollama freed. The seed's
fitness was 11.899, as on saumon: same CPU group, same work, same result.

**Work is the same on all four CPU types.** The sets built by `prepare` on albatros, ain and
allemagne are identical to saumon's: same gaps, same work, same node counts (Gurobi 13.0.3,
8 threads). So fitness values from different groups can be compared. `check_same_machine`
(same CPU name) is therefore stricter than needed. It is kept for now (see TODO.md).

### Step 14: Every LLM proposal recorded, tighter limits for cut solves
**Every proposal is kept.** The population file keeps only the accepted cuts. Now every
attempt of the LLM, accepted or not, is saved by
[src/evolution/recorder.py](src/evolution/recorder.py) in its own file, one folder per run and
per generation:

```
runs/<run>/gen_000/child_02_attempt_1.json
runs/<run>/gen_000/child_02_attempt_2.json
runs/<run>/gen_001/child_01_attempt_1.json
...
```

Each file holds: generation, child, attempt number, operator (`init`, `Lifted`, `Hybrid`, ...),
parent ids, **the full conversation sent** (system prompt, task, previous attempts and the
feedback on them), the reply, the parsed idea and code, the **outcome** (`accepted`,
`format_error`, `rejected`, `llm_error`) with its reason (e.g. "invalid cut: instance
rand_s1_0_15 becomes infeasible"), the fitness and the id of the new individual if accepted,
and the LLM and evaluation times. `GeneticAlgorithm(record=...)` is the hook, and `main.py`
plugs in an `AttemptRecorder`. A resumed run redoes the generation it was building: the new
attempts get a `_resumed2` suffix instead of overwriting the old ones. `fleet.py fetch` (and the
automatic fetch at the end of a run) copies these folders too.

**Tighter limits for cut solves.** A broken cut (e.g. a model that never finishes building)
used to be killed after `TIME_LIMIT + BUILD_TIMEOUT` = 60 + 60 = 120 s, while a normal scoring
solve takes 4–9 s at `WORK_LIMIT = 10`. The run's solver now uses `CUT_TIME_LIMIT = 20` s and
`CUT_BUILD_TIMEOUT = 10` s (flags `--cut_time_limit`, `--cut_build_timeout`): at most 30 s lost.
A solve that hits the time limit is **rejected as too slow**, never scored: how far a solve gets
in 20 s depends on the machine, while the work limit gives the same result everywhere.
`prepare_instances.py` keeps the old safety values.

### Step 15: Final review of the best cut under several seeds
During a run, each fitness comes from one deterministic solve per evaluation instance, with a
single Gurobi seed. Any change to a model changes Gurobi's search path, so a cut can score well
by luck. At the end of a run, `main.py` now **reviews the best cut**
([src/evolution/review.py](src/evolution/review.py)): the base model, the seed cut and the best
cut are solved on every evaluation instance under Gurobi seeds 0..4 (`--review_seeds 5`, 0 = no
review). For each seed, the fitness uses the run's formula against the base model **solved with
the same seed** (so the base scores 10 at every seed). Output, in the log and in
`runs/<run>.review.json`:

```
review of the cuts under 5 Gurobi seeds (0, 1, 2, 3, 4):
  model      fitness mean ± std     min     max  mean gap  run fitness
  base          10.000 ± 0.000   10.000  10.000     9.95%            -
  seed          11.2   ± 1.1     ...
  best          ...
  best - seed: +x.xxx on average; the best cut beats the seed cut on k/5 seeds
```

It costs (1 + 2) models × 5 seeds × 8 instances ≈ 120 solves, about 10–15 min at
`WORK_LIMIT = 10`. Meanwhile the status file says `phase: review`, and the dashboard shows
"reviewing the best cut". The automatic fetch happens when the run is finished, so the review
comes back too. For runs that already ended, use the script (no LLM needed):

```bash
cd src
python review_run.py --population_file ../runs/<run>.json --data_dir ../data/a4000
```

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
| [test_parameters.py](tests/test_parameters.py) | paths are absolute, license file used only if it exists, an exported `GRB_LICENSE_FILE` wins |
| [test_parser.py](tests/test_parser.py) | defaults come from `parameters.py`, flags override and are typed, a bad type is refused |
| [test_loaders.py](tests/test_loaders.py) | instance format and bounds, names unique across seeds (and regenerable without one), reproducibility with a seed, symmetric Euclidean costs, `get` saves then reloads, `split` is disjoint and complete |
| [test_models_tsp.py](tests/test_models_tsp.py) | model size (`n²` variables), optimum equals brute force, solution is a single tour, the baseline cut keeps the optimum |
| [test_individual.py](tests/test_individual.py) | defaults, unique ids, `parents` lists not shared, dict round trip, `copy` makes a new individual (new id, same cut and fitness, parent recorded), other generators refused |
| [test_population.py](tests/test_population.py) | wrong size and unevaluated individuals refused, history, best kept across worse generations and on ties, `top`, same id twice refused (an elite must be copied), `all_individuals` skips copies, an equal elite copy keeps the original as best, save/load |
| [test_prompts.py](tests/test_prompts.py) | YAML files load, expected operators present, every system prompt fills with the real `create_model` source and leaves no placeholder, user templates fill; `Prompts` builds mutation / crossover / retry messages; `parse_reply` reads the format (bold labels, extra text, no backticks, indentation) and refuses bad ones; `initializer.yaml` keys, initializer messages (reference code, ideas list, best fitness) |
| [test_llm_base.py](tests/test_llm_base.py) | `LLMClient` is abstract, failed calls retried, `LLMError` after the last try, `clean` hook applied |
| [test_llm_fake.py](tests/test_llm_fake.py) | replies in order, each conversation recorded as sent, no reply left → `LLMError` |
| [test_genetic.py](tests/test_genetic.py) | `initialize`: evaluated seed first then new cuts, the initializer sees the seed and the growing list of ideas, failed slot → copy of the seed, rejected baseline and second `initialize` raise; generation size kept and elite copied, `crossover_rate` picks the operator, bad format / rejected cut sent back and retried, parent copied when every attempt fails, unreachable LLM gives up at once, best fitness shown to the LLM, tournament, population of 1, same seed → same run, `progress` called before each child (not for the seed or the elites), every attempt recorded (format error, rejected with its reason, accepted with fitness and individual id, conversation and timings) |
| [test_llm_ollama.py](tests/test_llm_ollama.py) | against a fake Ollama HTTP server: conversation sent and answer returned, `available_models`, options forwarded, unknown model retried then `LLMError`, unreachable server; plus a live test, run only with `OLLAMA_LIVE=1` (needs the tunnel) |
| [test_sets.py](tests/test_sets.py) | validation set solved to optimality with solutions, saved then reloaded without solving, evaluation set keeps no solution, non-optimal validation instance refused (nothing saved), missing `"base"` formulation refused, refused instances replaced (new names), gives up when nothing is accepted |
| [test_prepare_instances.py](tests/test_prepare_instances.py) | script flags default to `parameters.py` and can be overridden (`--data_dir` too), `has_gap` accepts only unsolved instances |
| [test_review.py](tests/test_review.py) | under several seeds: the base model scores 10 at every seed, one score per seed and per model with its gaps, mean and std, JSON-serializable, table with "beats the seed cut on k/n seeds"; the score formula, instances solved by the base skipped, failed solves counted as gap 1 |
| [test_recorder.py](tests/test_recorder.py) | one file per attempt in its generation folder, a resumed generation does not overwrite the earlier attempts |
| [test_status.py](tests/test_status.py) | the status file is created "running", updates keep the other fields, no temporary file left |
| [test_transforms.py](tests/test_transforms.py) | `with_cut` runs the code with `m, gp, GRB, n, c` and keeps the optimum for a valid cut, code errors raise, `relaxed` drops integrality and keeps names, `fixed` to the optimum is feasible and to a bad point infeasible, rounding, partials compose |
| [test_evaluator.py](tests/test_evaluator.py) | valid and useful cut gets a fitness, the fitness follows the gap formula, removing an optimal tour → invalid, implied constraint → useless, lower bound above a known solution → invalid, broken code → rejected with its error, validation set without solutions or solved evaluation instance refused, a cut hitting the time limit rejected as too slow |
| [test_main.py](tests/test_main.py) | a small run on tiny sets with `FakeLLM` (seed + initializer, 1 generation, saved after each generation), resuming continues without a new first generation, missing sets reported, `--model_name` required, sets from another CPU or thread count refused, the status file follows the run (finished, generation, best, the run's own Gurobi log folder), an interrupted run is marked "stopped", run name with the host / the resumed file's name, end-of-run review saved (`<run>.review.json`) and the same review by `review_run.py` on the saved population |
| [test_solver.py](tests/test_solver.py) | results stored per formulation, a valid cut keeps the optimum, `SET` and `RANDOM` seed modes, every run kept, log file named with the seed, missing `log_dir` error, infeasible model, unbounded model, isolated and in-process give the same optimum, broken formulation → `ERROR` with its message, `TOO_LARGE`, hanging build → `KILLED`, dead child → `CRASHED`, build time not in `runtime`, lambdas refused when isolated, each solve follows its own `verbose`, `keep_solution` stores the tour (both modes) and is off by default, work recorded and reproducible, `work_limit` stops the solve, machine / CPU / threads recorded (also for failed solves) |

Tests that need a folder create it in pytest's temporary directory (`tmp_path`), so they
never write to `data/` or `logs/`.

`computer_connection/test_fleet.py` (git-ignored, like `fleet.py`) checks the fleet tool without
any machine: `python3 -m pytest computer_connection/test_fleet.py`.

### Next steps
The ordered plan for what comes next (sound fitness → first real result → scale up → later)
is in [TODO.md](TODO.md) → "Priorities".

- [ ] `requirements.txt` listing the dependencies
- [x] MILP model for the TSP (`create_model(n, c)`) in gurobipy
- [x] Solver wrapper (`Solver`, Step 4)
- [ ] Cache solver results on disk (`cache/`)
- [x] Representation of a cut / an individual of the population (`Individual`, `Population`, Step 5)
- [x] `LLMClient` abstract base (retry built in) + `FakeLLM` for tests (Step 7)
- [x] Local LLM served by Ollama on a lab GPU: `qwen3-coder:30b`, reached through an SSH tunnel
- [x] `OllamaClient(LLMClient)`: `_send` posts the conversation to `http://localhost:11434/api/chat` (Step 8)
- [x] Validation and evaluation sets, solved once with the base model (Step 8); evaluation
  instances of 60–100 cities that the base model cannot finish (Step 10)
- [x] Cut evaluation: run the LLM code, check validity and usefulness, work-based fitness (Step 9)
- [x] Mutation and crossover prompts, rewritten for gurobipy (Step 6)
- [x] First generation: baseline seed + initializer prompt asking for diversity (Step 11)
- [x] Build the messages from the prompts + parse the `IDEA:` / `CODE:` reply (Step 7).
  **To do when choosing the local LLM:** reasoning models (DeepSeek-R1, Qwen3, ...) write a
  `<think>...</think>` block before their answer. Remove it in the client's `clean` hook (the
  reference `extract_json` does the same).
- [x] Fitness measure: Gurobi work units, reproducible, so elites need no re-evaluation (Step 9)
- [x] Elites and fallback copies are new individuals (`Individual.copy`), and
  `set_generation` refuses duplicate ids
- [x] `GeneticAlgorithm`: tournament selection, elitism, one child per slot, mutation and crossover (Step 7)
- [x] `src/main.py`: load the sets, build the GA, `initialize()` then N × `generation_step()`,
  save the population after each generation, resume with `--population_file` (Step 12)
- [x] Runs on every lab machine from the Mac: `fleet.py` (start / stop / resume / status) and its
  web dashboard, per-run status file, one instance-set folder per CPU group (Step 13)
- [ ] Optional: stop the runs automatically before class hours (a `--until` time, or a cron job
  on the Mac calling `fleet.py stop all`)
- [ ] First full runs (8 individuals, a few generations), then look at the success rate
  of the LLM and the fitness over the generations

---

## Code conventions
- Type hints everywhere: parameters, return types (`-> None` included), class attributes
  and non-obvious local variables.
- Use Python 3.9-compatible `typing` generics: `List`, `Dict`, `Optional`, `Tuple`
  (not `list[...]` or `X | None`).
- Give recurring structures a type alias (`Instance`, `Coords`, `Costs`, ...).
