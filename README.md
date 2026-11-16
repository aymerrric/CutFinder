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
│                          # <problem>/: validation, evaluation, test .pkl (+ evaluation_hard, test_hard:
│                          # --instance_level hard; optimality, optimality_test: --fitness_mode gap_work)
├── cache/                 # cached solver results
├── logs/                  # Gurobi log files, one per solve, in logs/<run>/, packed into logs/<run>.tar.gz when the run ends (stopped, failed or finished)
├── runs/                  # one folder per EvoCut run (git-ignored):
│   ├── <run>/             #   summary.md (read first), population.json, status.json, run.log,
│   │                      #   review.json, llm/gen_<k>/ (every LLM proposal of generation k),
│   │                      #   validity.json (check_run_cuts.py: is the best cut valid)
│   ├── prompt_bench/<name>/ # prompt benchmarks: report.md, results.jsonl, config.json
│   └── launches/          #   console output of the launches and set preparations (fleet.py)
├── improve_license.md     # how to get more Gurobi capacity for free (campus server, named-user licenses, ...)
├── learnt_results.md     # what the runs taught us, each finding with its data (files and numbers)
├── problems.md           # short explanation of each problem studied (model, seed cut, sizes, results so far)
├── paper_approaches_20261115.md  # (in French) the papers this project could become, and the one recommended (certified cuts); what FLARE already did
├── paper_plan_20261115.md        # (in French) step-by-step plan of that paper: what to build, the runs, benchmarks and proofs needed, compute and money
├── papers/                # the EvoCut/SAGE paper (TMLR 10/2026): PDF + extracted text; the FLARE paper as text
├── lean/                  # the Lean 4 project of the proofs (Lake; Mathlib in lean/.lake, 6.9 GB, git-ignored)
│   ├── lakefile.toml, lean-toolchain, lake-manifest.json  # Lean 4.28.0 and Mathlib v4.28.0, as FLARE pins them
│   ├── EvoCut/Common.lean # MILPFormulation, MILPReformulation (FLARE's, verbatim); ValidCut, SafeCut and why they are certificates
│   ├── EvoCut/Problems/   # Tsp.lean, Cwlp.lean, Jssp.lean: each create_model written by hand in Lean, with its data assumptions
│   ├── EvoCut/Cuts/TspSeed.lean  # the TSP seed cut proved by hand: the reference certificate, shown to the LLM
│   └── work/              # folders of the Claude Code agent (git-ignored)
├── cuts/<problem>/paper_sage.py  # the paper's selected cut family (Appendix F) as cut code
├── cuts/tsp/invalid_position_bound.py  # a cut invalid for every n: the control of prove_cut.py
├── prompts/               # LLM prompts (YAML, versioned)
│   ├── common.yaml        # shared system template + user message templates
│   ├── mutations.yaml     # General, Lifted, Exploratory
│   ├── crossovers.yaml    # Intersection, Complement, Hybrid, Min_Violation
│   ├── initializer.yaml   # first generation: new cuts from scratch, baseline as format example
│   ├── fitness.yaml       # how cuts are scored, one text per fitness mode (gap, lp, gap_work), told to the LLM
│   ├── problems/          # what the LLM is told about each problem (tsp.yaml, cwlp.yaml, jssp.yaml, mcnd.yaml, imo6.yaml, gisp.yaml, corridor.yaml: name, description, solution)
│   ├── proving.yaml       # prompts of the proof pipeline: state a cut in Lean, judge the statement, prove it; the agent's task
│   └── variants/          # prompt variants to compare (rules, catalog, reasoning, concise, few_shot, focused_ops)
├── gurobi.lic             # Gurobi license (secret, git-ignored)
├── llm_keys.env           # API keys of the LLM providers (secret, git-ignored; template: llm_keys.env.example)
├── computer_connection/   # remote machines (git-ignored): machines.txt, fleet.py + dashboard.html, scripts
├── .gitignore             # hides secrets, .venv, data/, cache/, logs/, runs/
├── pytest.ini             # test config: run `pytest` from the project root
├── TODO.md                # ideas and known limits to come back to (e.g. relaxing the validity check)
├── tests/                 # one test file per module, updated at every step
│   └── data/              #   small real benchmark files (6 TSPLIB + optimal tours, cap61, ta01, r01.1)
└── src/
    ├── utils/
    │   ├── parameters.py  # default values for every setting + project paths
    │   ├── parser.py      # command-line arguments (defaults come from parameters.py)
    │   ├── status.py      # RunStatus: runs/<run>/status.json, where a run is (read by fleet.py)
    │   ├── run_files.py   # layout of a run folder (summary.md, population.json, llm/, ...)
    │   └── log_archive.py # pack logs/<run>/ (one Gurobi log per solve) into logs/<run>.tar.gz at the run's end
    ├── prepare_instances.py  # script: build + solve the validation and evaluation sets
    ├── prepare_benchmark.py  # script: download the paper's public test sets (TSPLIB, OR-Library, Taillard)
    ├── prepare_validity.py   # script: build + solve to optimality the validity instances (data/validity/<problem>/)
    ├── benchmark_cut.py   # script: score a cut on them with the paper's protocol (2000 s, Delta g)
    ├── main.py            # script: run EvoCut (first generation + generation steps), save each generation
    ├── problems.py        # PROBLEMS: per problem, loader, base model, seed cut, instance sizes
    ├── review_run.py      # script: the final review for a saved run (no LLM)
    ├── check_run_cuts.py  # script: re-check the best cut of saved runs with the validity instances
    ├── prove_cut.py       # script: prove a cut valid in Lean 4 with an LLM (runs/proofs/<problem>/<label>/)
    ├── proving/
    │   ├── lean.py        # run the Lean compiler on a file, read its messages, axioms, refused words
    │   ├── problems.py    # LEAN_PROBLEMS: the Lean formulation of each problem + its data assumptions, checked in Python
    │   ├── certificate.py # the Lean files of a cut and verify: when a proof is a certificate
    │   ├── prover.py      # Prover: the loop (statement, judge, proof), each answer compiled and its errors sent back
    │   └── agent.py       # AgentProver: the proof by a Claude Code agent with check.sh and the lean-lsp MCP server
    ├── summarize_run.py   # script: (re)write the summary.md of run folders
    ├── list_models.py     # script: the models a backend offers (checks the API key)
    ├── report_runs.py     # script: runs/report.md, best per LLM and mean of each computer's best
    ├── prompt_bench.py    # script: compare prompt variants without running the GA
    ├── loaders/
    │   ├── base.py        # Loader: abstract interface to generate / save / load / split instances
    │   ├── tsp.py         # LoaderTSP: random Euclidean TSP instances
    │   ├── cwlp.py        # LoaderCWLP: random CWLP instances (reference generator)
    │   ├── jssp.py        # LoaderJSSP: random job shops (reference generator)
    │   ├── mcnd.py        # LoaderMCND: random network design instances shaped like CommaLab R
    │   ├── imo6.py        # LoaderIMO6: grid sizes N drawn without replacement (an instance is only N)
    │   ├── corridor.py    # LoaderCorridor: square lattices, 3 reserves, budget 20-30% (the corridor generator); connectable
    │   ├── gisp.py        # LoaderGISP: random G(n, 0.3) graphs, 25% removable edges (Distributional MIPLIB's generator)
    │   ├── mclp.py        # LoaderMCLP: the reference's random MCLP (points in the unit square, coverage disks, P = 1% of the sites); in progress
    │   ├── benchmarks.py  # the paper's public test sets: TSPLIB / Beasley / Taillard / CommaLab R parsers + downloads; Distributional MIPLIB reader (GISP); generated corridor set
    │   ├── validity.py    # validity instances: small random, TIGHT (bounds = optimum) and public JSPLIB ones
    │   └── sets.py        # build_set: instances solved once with the base model, saved
    ├── models/
    │   ├── tsp.py         # create_model(n, c): TSP MILP (MTZ) in gurobipy + BASELINE cut (2-cycle)
    │   ├── cwlp.py        # create_model(I, J, d, u, f, c): CWLP MILP + BASELINE cut (minimum number of open warehouses)
    │   ├── jssp.py        # create_model(n_jobs, n_machines, times, machines): JSSP big-M MILP + BASELINE cut (machine load)
    │   ├── mcnd.py        # create_model(N, A, K, c, f, u, d, O, D): MCND arc-based MILP + BASELINE cut (open arcs out of each origin)
    │   ├── imo6.py        # create_model(N): IMO 2025 Problem 6 tiling MILP (strips with vertical flow) + BASELINE cut (row components)
    │   ├── corridor.py    # create_model(nodes, edges, reserves, cost, utility, budget): wildlife corridor, single-commodity flow (negated objective) + BASELINE cut (flow leaves only bought cells)
    │   ├── gisp.py        # create_model(nodes, permanent, removable, weight, cost): GISP (negated objective) + BASELINE cut (triangles with one removable edge)
    │   ├── mclp.py        # create_model(I, J, Ni, a, P): Maximal Covering Location (negated objective); seed cut not chosen yet
    │   └── transforms.py  # with_cut / relaxed / fixed: problem-independent model transformations
    ├── solvers/
    │   └── solver.py      # Solver: problem-independent, compares formulations, logs, stores results
    ├── llm/
    │   ├── base.py        # LLMClient: abstract, send(messages) -> str with retries
    │   ├── fake.py        # FakeLLM: fixed replies, for tests
    │   ├── ollama.py      # OllamaClient: any model served by Ollama (POST /api/chat)
    │   ├── openai_compatible.py  # OpenAICompatibleClient: Groq, Gemini, OpenRouter (POST /chat/completions)
    │   ├── claude_code.py # ClaudeCodeClient: Claude through the `claude` command, no tool, no API key (prove_cut.py)
    │   └── factory.py     # make_llm(--llm_backend, --model_name): the run's LLM client
    └── evolution/
        ├── individual.py  # Individual: one cut (idea, code, fitness, generator, parents, id)
        ├── population.py  # Population: fixed-size current generation, history, best known, save/load
        ├── prompts.py     # Prompts: conversations from prompts/*.yaml; parse_reply: IDEA/CODE
        ├── evaluator.py   # CutEvaluator: validity, usefulness, work-based fitness of a cut (gap, lp, gap_work)
        ├── recorder.py    # AttemptRecorder: every LLM proposal saved in runs/<run>/llm/gen_<k>/
        ├── summary.py     # write_summary: runs/<run>/summary.md, built from the run folder
        ├── report.py      # report over several runs, grouped by LLM, fitness mode and instance level; between-LLM tables
        ├── review.py      # review_cuts: best cut vs seed cut vs base under several Gurobi seeds
        ├── benchmark.py   # the paper's measure: gap at checkpoints, Delta g, Table 2 (with the paper's rows)
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
| [fitness.yaml](prompts/fitness.yaml) | `gap`, `lp` | how cuts are scored, inserted at `{scoring}` in `system` (Step 21) |

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
| test (Step 22) | `data/tsp/test.pkl` | `N_TEST = 8`, 50–100 cities | under `WORK_LIMIT`, base gap > 0 | end-of-run review only: held out, never seen by the GA |

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
  rounds). `prepare_instances.py` passes `has_gap(MIN_BASE_GAP)`. New flag: `--min_base_gap`.
  (Since Step 25 the instance sizes are per problem, in `src/problems.py`; the TSP keeps 15 /
  60–100 / 50–100 cities.)
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
python main.py --model_name qwen3-coder:30b --population_file ../runs/<run>/population.json   # resume
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
`./runs` on the Mac (the run's folder: summary, population, log, review, LLM proposals) and a macOS
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
  run writes `runs/<run>/status.json` ([src/utils/status.py](src/utils/status.py)), rewritten
  atomically before each child and after each generation: state, generation done, generation
  being built and child number, best and mean fitness, pid, arguments, population file.
  `GeneticAlgorithm(progress=...)` is the hook: it is called before each child. One SSH call
  per machine (all in parallel, about 2 s for 37 machines) reads it, and checks that the process
  is alive, our Ollama server, GPU memory (ours / other users'), and who is logged in.
- **States:** `running`, `preparing`, `finished`, `stopped` (SIGTERM or Ctrl-C: `main.py` turns
  SIGTERM into a clean stop and records it), `failed` (exception, with its message), `dead` (the
  status says running but the process is gone, e.g. killed with SIGKILL), `idle`, `unreachable`.
- **Machines with their own Gurobi license** (named-user academic license, activated with
  `fleet.py license <host>=<key>`, see [improve_license.md](improve_license.md)) use it
  automatically and are not counted in the limit below.
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
runs/<run>/llm/gen_000/child_02_attempt_1.json
runs/<run>/llm/gen_000/child_02_attempt_2.json
runs/<run>/llm/gen_001/child_01_attempt_1.json
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
*Update 2026-10-27:* `CUT_TIME_LIMIT` raised to **60 s**. With 20 s, a loaded machine (dindon)
rejected the seed cut itself: 6.5 work units in 20 s (`learnt_results.md` 1.6).

### Step 15: Final review of the best cut under several seeds
During a run, each fitness comes from one deterministic solve per evaluation instance, with a
single Gurobi seed. Any change to a model changes Gurobi's search path, so a cut can score well
by luck. At the end of a run, `main.py` now **reviews the best cut**
([src/evolution/review.py](src/evolution/review.py)): the base model, the seed cut and the best
cut are solved on every evaluation instance under Gurobi seeds 0..4 (`--review_seeds 5`, 0 = no
review). For each seed, the fitness uses the run's formula against the base model **solved with
the same seed** (so the base scores 10 at every seed). Output, in the log and in
`runs/<run>/review.json` (and in the run's `summary.md`):

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
python review_run.py --population_file ../runs/<run>/population.json --data_dir ../data/a4000
```

### Step 16: One folder per run, with a summary
Everything about a run is now in **its own folder** (layout in
[src/utils/run_files.py](src/utils/run_files.py)):

```
runs/<run>/
    summary.md        read this first
    population.json   every generation and the best ever (to resume: --population_file)
    status.json       where the run is (fleet.py, dashboard)
    run.log           the full log
    review.json       final review of the best cut under several seeds
    llm/gen_000/child_02_attempt_1.json ...   every LLM proposal, one folder per generation
```

`summary.md` ([src/evolution/summary.py](src/evolution/summary.py)) is rewritten after every
generation, at the end (after the review), and when the run stops or fails. It has:
- state, machine, LLM, start, duration and settings;
- **the best cut**: fitness, the generation where it was found, the operator, the parents, the
  idea and the code;
- **the review** under several seeds (mean ± std, on how many seeds it beats the seed cut);
- **the fitness per generation** (best ever, max, mean, where the individuals come from);
- **the LLM proposals per operator**: count, accepted, acceptance rate, format errors,
  rejections, and the most frequent rejection reasons.

It is built only from the folder's files, so `python summarize_run.py ../runs/<run>` rewrites
it at any time, also for old runs. A resumed run keeps its folder (`--population_file
../runs/<run>/population.json`). fleet.py writes its launch outputs in `runs/launches/`.

Clean-up of 2026-10-22: the test runs of 19 October and the old flat Gurobi logs were deleted,
and the two real runs (allemagne and finlande, 20 October) were moved to this layout with their
summaries. Their LLM proposals were not recorded yet at the time.

### Step 17: Comparing prompts without running the GA
A full run per prompt version takes hours, and its result is noisy (Step 15). Instead,
[src/prompt_bench.py](src/prompt_bench.py) gives every prompt variant **exactly the same
requests** and measures what happens to the answers:
- **Same requests:** the parents are the seed cut and the best distinct cuts of a saved run
  (`--population_file`). Per round (`--samples`), each variant gets one initializer request,
  one per mutation operator, and one per crossover operator, with the **same LLM seed** for the
  same request in every variant (a paired comparison).
- **Same checks as a run:** format, code, validity, usefulness (15-city instances, ~0.4 s,
  `CutEvaluator.check`). Optionally a quick score of the accepted cuts on the first k
  evaluation instances (`--score_instances k`). `--attempts 2` also tests the feedback after a
  refusal.
- **Report** `runs/prompt_bench/<name>/report.md`, rewritten after each round:
  - per variant: accepted answers with a 95% confidence interval, distinct accepted cuts,
    copies of a parent, fitness of the accepted cuts, LLM time;
  - acceptance per task (initializer, mutation, crossover);
  - why answers were refused (format, code error, too slow, invalid, useless);
  - paired comparison with the first variant (accepted where the reference is not, and the
    reverse).
  Every answer is in `results.jsonl`.

**Prompt variants** live in `prompts/variants/<name>/`. A variant overrides only what it
changes: keys of `common.yaml` / `initializer.yaml`, or the whole operator file
(`mutations.yaml`, `crossovers.yaml`). The six written so far, each testing one hypothesis
(details in [prompts/variants/README.md](prompts/variants/README.md)):

| variant | hypothesis |
|---|---|
| `rules` | answers fail because the LLM does not know how they are checked (symmetry, copies, cost) |
| `catalog` | a list of strengthening techniques gives better ideas |
| `reasoning` | a validity argument and a removed LP point before the code make answers more often valid |
| `concise` | a shorter prompt suits a small model better |
| `few_shot` | examples of refused and accepted answers teach the checks faster than rules |
| `focused_ops` | concrete operators (Tighten, Extend, Prune, Explore; Merge, Graft) give more useful children |

Run it from the Mac with fleet.py (Ollama, model and instance sets of the machine are set
automatically, the results are fetched when it ends):

```bash
python3 computer_connection/fleet.py bench saumon -- --name first --samples 5 \
    --population_file ../runs/tsp_qwen2.5-coder-14b_finlande_20261020_180716/population.json
```

or on a machine: `python prompt_bench.py --model_name ... --data_dir ../data/<group> ...`.
All variants are compared unless `--variants` lists some (the first one is the reference of
the paired comparison). **Parallel benchmarks:** split the work over several machines (one
variant per machine, and `--sample_offset` for other rounds of the same requests), then build
one report for all the parts:

```bash
python prompt_bench.py --merge ../runs/prompt_bench/<out> ../runs/prompt_bench/<part1> ...
```

**The report** starts with a **verdict**:
- the variant with the most accepted answers, with its 95% interval;
- which variants are **significantly** better or worse than `default`, using an exact McNemar
  test on the paired requests (p < 0.05);
- the best fitness of the accepted cuts;
- the most distinct new cuts.

Then a **ranking** (accepted answers with a text bar, interval, better / worse / p against the
reference, distinct new cuts, copies of a parent, LLM seconds), the quality of the accepted
cuts, the acceptance per task, the refusal reasons, and the setup. Cost: 7 requests per variant and per round (6 for `focused_ops`), so 5
rounds × 7 variants ≈ 245 LLM answers, about 30–60 min. The winner then gets a full run with
`main.py --prompt_variant <name>`.

**First benchmark (2026-10-24)**, `runs/prompt_bench/prompts_20261024/report.md`: 1,650 answers of
`qwen3-coder:30b` on 14 A5000 machines, 7 variants × 30 rounds. The ranking counts only *new*
accepted cuts (answers that repeat a parent pass the checks but bring nothing).

| variant | new cuts accepted | vs `default` (paired) |
|---|---|---|
| `rules` | 55% (49–61%) | +7 pts, p = 0.09 |
| `focused_ops` | 51% | +3 pts, p = 0.09 (but 17% of its accepted answers are copies) |
| `concise` | 50% | p = 0.66 (most distinct cuts) |
| `default` | 48% (42–55%) | reference |
| `few_shot` | 42% | p = 0.17 |
| `reasoning` | 39% | **worse**, p = 0.02 (many copies) |
| `catalog` | 25% | **worse**, p < 0.001 (many too-slow cuts) |

The initializer prompt fails for every variant (1–5 accepted out of 30), and invalid answers are
the main refusal: these are the two next things to improve.

### Step 18: More attempts for the first generation, no duplicates
The prompt benchmark showed that only 1–5 of 30 initializer answers pass the checks, and the
runs of 2026-10-20 had 4 copies of the seed out of 8 in generation 0 (`learnt_results.md` 2.2).
So the first generation now has **its own number of LLM attempts per slot**: `--max_init_attempts`
(`MAX_INIT_ATTEMPTS = 15`, versus `--max_llm_attempts` = 3 for the other children). And an
initial cut whose code (comments and spaces removed: `normalized_code` in
[src/evolution/individual.py](src/evolution/individual.py)) is already in the generation is
**refused before any solve**, with the reason sent back ("duplicate: this is the same cut as one
already in the generation ..."), so the extra attempts bring real variety.

### Step 19: Fitness averaged over several Gurobi seeds
With one Gurobi seed the fitness is mostly luck: the seed cut alone scores from 2.6 to 14.6
depending on the seed (`learnt_results.md` 1.2), and a cut that adds nothing won +26% (1.3).
So each cut is now scored under **several seeds during the run**, `--fitness_seeds`
(`FITNESS_SEEDS = [42, 43, 44]`):
- `CutEvaluator` gets one scoring solver per seed (`scoring_solvers`). The validity and
  usefulness checks still run once, on the first one.
- At startup the evaluator solves the base model under every seed that `evaluation.pkl` was not
  solved with: seed 42 reuses the stored results, while 43 and 44 cost 8 solves each (about 1.5 min).
- Per seed it applies the review's formula (`review.score`), and the fitness is the mean:
  `fitness = mean_s 10 * exp(-mean_i (gap_cut_i,s - gap_base_i,s) / gap_base_i,s)`. With one
  seed, this is exactly the old fitness.
- The bound check ("lower bound above a known tour") now uses the cheapest tour found by the base
  model under any of the seeds.
- The seeds are disjoint from the review seeds 0–4, so the end-of-run review stays an independent
  check of the selected cut. `main.py` warns if they overlap.

Cost: scoring is about 3x longer, and it already took 132 of the 142 min of a run on 2026-10-27
(saumon, `llm/gen_*/*.json`: `eval_seconds` vs `llm_seconds`). A 20-generation run will take
about 6 h, so start it in the evening (`learnt_results.md` 4.2).
```bash
python main.py --model_name qwen3-coder:30b --fitness_seeds 42 43 44   # the default
python main.py --model_name qwen3-coder:30b --fitness_seeds 42         # old behaviour, 1 seed
```
Caveat: resuming a run started with one seed mixes old 1-seed fitness values with new 3-seed
ones in the same population.

### Step 20: A second fitness, the LP relaxation (`--fitness_mode lp`)
The best cuts of 2026-10-27 win on the gap without tightening the LP relaxation at all
(`learnt_results.md` 1.7). So the GA can now score cuts by **how much they tighten the LP**.
`CutEvaluator` has a `mode` field, a `FitnessMode` enum (`GAP`, `LP`; a hybrid mode may come later).
Its public method `evaluate(code)` runs the shared checks (validity, usefulness) and then
dispatches on the mode (`__call__` = `evaluate`, so the GA is unchanged):
- `GAP` → `_score_gap`: the 3-seed MIP gap fitness of Step 19;
- `LP` → `_score_lp`: the LP relaxation of every evaluation instance with the cut, one solve per
  instance, no seed (an LP is deterministic):
  `fitness = 100 * mean_i (LP_cut_i - LP_base_i) / (UB_i - LP_base_i)`, the **% of the root gap
  closed**. UB_i is the cheapest tour the base model found, and the base LP values are solved once
  at startup. 0 = no tightening, 100 = the LP reaches the best known tour. An LP value above UB_i
  rejects the cut as invalid. In LP mode `--fitness_seeds` is ignored: only one solver, no
  extra base solves.

The two fitness scales differ (gap: 10 = no gain; LP: 0 = no gain), so do not mix modes in
one run or resume a run with the other mode.
```bash
python main.py --model_name qwen3-coder:30b --fitness_mode lp
```
Still to do: report both metrics in the end-of-run review.

### Step 21: The LLM is told exactly how its cuts are scored
A cut that wins on the LP fitness and one that wins on the gap are not the same: on the LP, only
the strength of the relaxation counts and the number of constraints is free, while on the gap a
heavy family of constraints slows down every node. So the prompt now states **the fitness of the
run, with its formula**, and what makes it rise:
- [prompts/fitness.yaml](prompts/fitness.yaml) holds one text per fitness mode (`gap`, `lp`, the
  values of `FitnessMode`). `gap` explains the gap under a fixed work budget and several seeds
  (10 = no gain, keep the model light). `lp` explains the % of the root gap closed (0 = no
  tightening, 100 = the best known tour; rewriting or repeating constraints brings nothing).
- The fitness paragraph of the `system` prompt (in `common.yaml` and in the 5 variants that
  replace `system`) is now the placeholder `{scoring}`. `Prompts(..., fitness_mode=...)` fills it,
  and `main.py` passes `--fitness_mode`, so the prompt and the evaluator always match.
- A variant may override the texts with its own `fitness.yaml` (key by key): `concise` does, to
  stay short.
- `prompt_bench.py` still uses the `gap` text, since it scores with the gap.
```bash
python main.py --model_name qwen3-coder:30b --fitness_mode lp   # evaluator AND prompt use the LP fitness
```

### Step 22: The final review runs on held-out instances
Until now the end-of-run review re-solved the **evaluation set**, the very instances the GA
selected the best cut on. Several seeds remove the luck of one seed, but not the luck of these
8 instances (winner's curse, `learnt_results.md` 1.4). So the review now uses a **held-out test
set** that the GA never sees:
- `prepare_instances.py` builds `data/<group>/tsp/test.pkl`: `N_TEST = 8` instances of
  50 to 100 cities (`test_sizes` in `src/problems.py` since Step 25), generated from `TEST_SEED = 1000`.
  That seed is disjoint from validation (1) and evaluation (100 + up to 19 replacements). As for
  the evaluation set, only instances the base model cannot finish under `WORK_LIMIT` are kept
  (base gap > `MIN_BASE_GAP`); otherwise the gap fitness has nothing to measure.
- The review (`main.py`'s `final_review` and `review_run.py`) scores the base model, the seed cut
  and the best cut on `test.pkl` under Gurobi seeds 0–4 (`--review_seeds 5`). It solves the base
  model itself, so it does not depend on the CPU that built `test.pkl`.
- `main.py` loads `test.pkl` at startup when a review is asked (`--review_seeds > 0`), so a
  missing file fails **before** the GA, not after hours of it.
- `review.json` now records `instance_set` and `instances` (their names), and `summary.md` says
  which set was used. Older reviews, without these keys, are shown as "on the evaluation set".
- `fleet.py` considers a group ready only if `test.pkl` also exists. On a group prepared before,
  `fleet.py prepare <group>` builds just the missing `test.pkl` (existing sets are loaded, not
  rebuilt).
```bash
python prepare_instances.py --data_dir ../data/a5000            # adds test.pkl to an existing group
python review_run.py --population_file ../runs/<run>/population.json --data_dir ../data/a5000
```

### Step 23: The review reports both metrics, whatever the fitness mode
Every review now gives, for the base model, the seed cut and the best cut on `test.pkl`:
- the **gap fitness** under Gurobi seeds 0–4 (as before): does the cut make Gurobi faster?
- the **LP gap closed**: one LP relaxation per instance (no seed, run to the end, without the work
  limit), `100 * mean_i (LP_i - LP_base_i) / (UB_i - LP_base_i)`, with UB_i the cheapest tour of
  the base model under the review seeds. Is it a stronger formulation? (0 for the base model.)

A cut trained on one metric is then always checked on the other (`learnt_results.md` 1.7: on
2026-10-27 they disagreed). `review.json` stores `lp_values`, `lp_closed` and `lp_fitness` per
model, and `fitness_mode`, which says on which scale `run_fitness` is. The table in `run.log` and
`summary.md` gets an "LP gap closed" column, and the "fitness during the run" column names its
mode. `review_run.py` gives both metrics for old runs too.

### Step 24: A report over all the runs, per LLM
When one run is launched per lab machine, each `summary.md` only describes its own run.
`runs/report.md` ([src/evolution/report.py](src/evolution/report.py), written by
[src/report_runs.py](src/report_runs.py)) sums them up, with **one section per problem, LLM and
fitness mode** (problem added on 2026-11-04). Gap and LP fitness have different scales, so they are never averaged together. Each section has:
- **Best of all:** the run whose best cut has the highest **review** score, in the metric of its
  fitness mode (gap fitness under seeds 0–4, or LP gap closed). The run fitness is shown but never
  used to rank, since it is biased upwards (`learnt_results.md` 1.4). It also gives the cut's idea and code.
- **Mean of the best run of each computer:** for each computer, its best reviewed run; then mean
  ± std over the computers of the review gap fitness, the review LP gap closed and the run fitness.
- A table of every run: computer, state, generation, prompt variant, run fitness, review gap
  fitness, LP gap closed, and whether the review was on the held-out set. A "running" run whose
  status has not been updated for over an hour is shown as **stalled** (killed without warning,
  `learnt_results.md` 4.2).

`fleet.py watch` (and the dashboard) rewrite it after fetching the logs of the runs that ended.
`--since` keeps only the runs of a campaign:
```bash
python3 computer_connection/fleet.py watch --since 20261102   # report rewritten when runs end
python3 computer_connection/fleet.py report --since 20261102  # fetch + report now
python report_runs.py --since 20261102                        # from src/, on the fetched runs
```

### Step 25: Nothing TSP-specific outside the problem's own files
Preparing a second problem (CWLP first). Three things assumed the TSP:
- **The problem table** moved from `main.py` to [src/problems.py](src/problems.py): `PROBLEMS[<name>]`
  gives the loader, the base model, the seed cut and now the **sizes of the three sets**
  (`validation_sizes`, `evaluation_sizes`, `test_sizes`: keyword arguments of the loader's
  `generate`), plus `describe_sizes` ("60 to 100 cities") for the prompts. `main.py`,
  `review_run.py`, `prompt_bench.py` and **`prepare_instances.py --problem_name <name>`** go
  through it. The TSP-only flags and parameters (`--validation_cities`, `EVALUATION_MIN_CITIES`, ...)
  are gone.
- **Objective direction:** every base model **minimizes**. A maximization problem minimizes the
  negated objective in its `create_model`: the relative MIP gap and the share of the root gap
  closed do not change when the objective is negated, so the evaluator and the review need no
  special case. `check_minimization` refuses a model that maximizes (`prepare_instances.py` and
  every run check it).
- **The prompts:** the system prompt has a "problem" part, made of placeholders filled by
  [prompts/problems/<name>.yaml](prompts/problems/tsp.yaml):
  - `{problem}`: its name, in "a MILP model of {problem}";
  - `{description}`: what the problem is and what the model's variables mean (no known strong cuts
    there, so that rediscovering them stays a real test);
  - `{solution}`: what a solution is called ("best known {solution}" in the fitness text);
  - `{evaluation_size}`: from `problems.py`, so it always matches the scoring instances.

  Every TSP phrase of `common.yaml`, `fitness.yaml` and the variants became a placeholder ("about
  100 cities for a TSP" → "{evaluation_size}"). The TSP examples of `few_shot` moved to
  `prompts/variants/few_shot/problems/tsp.yaml` (`{examples}`): a variant can add or override keys
  per problem. `Prompts(..., problem=...)` refuses a placeholder without a value, and a test checks
  that no template outside `problems/` mentions cities, tours, the TSP or MTZ.

Adding a problem now means: a loader, a `create_model` (minimizing), a seed cut, an entry in
`PROBLEMS`, and a `prompts/problems/<name>.yaml`. Like Step 21, the system prompt changed (it now
describes the problem), so new runs are not directly comparable with the old ones.

### Step 26: Second problem, the CWLP
The **Capacitated Warehouse Location Problem** with unsplittable demands, as in the reference
(`configs/prompts/cwlp.yaml`, `data/data_prepare_cwlp_rand.py`): open warehouses (cost `f[j]`)
and assign each customer to exactly one open warehouse (cost `c[i, j]`), within its capacity
`u[j]`, at minimum total cost.
- **Loader** ([src/loaders/cwlp.py](src/loaders/cwlp.py)): the reference generator. Demands
  50–1500, opening costs 7000–15000, `c[i, j] = d[i] * uniform(5, 75)`, total capacity 1.1–1.3 x
  total demand, capacities within ±10% of their average. Names `cwlp_s<seed>_<idx>_<I>x<J>`.
- **Model** ([src/models/cwlp.py](src/models/cwlp.py)): `x[i, j]`, `y[j]` binary, assignment
  `sum_j x[i, j] = 1`, aggregated capacity `sum_i d[i] x[i, j] <= u[j] y[j]`. The textbook strong
  linking `x[i, j] <= y[j]` is **not** in the model: it is the cut to rediscover.
- **Seed cut:** at least `k_min` warehouses are open, `k_min` being the smallest number of
  warehouses whose largest capacities cover the total demand. It is valid and useful on the
  validation instances, and weak: it closes 10.8% of the root gap on 30x8, 1.5% on 100x20 and
  0% on 500x80 and 1000x150, while the linking closes 84%, 57%, 33% and 26%
  (`runs/analysis/cwlp_sizes_20261102.txt`).
- **Sizes** (same file, base model under `WORK_LIMIT = 10`, 8 threads): up to 200x40 the base
  model solves to optimality, 300x50 stops at 0.06–0.17% (below `MIN_BASE_GAP`), and 500x80 to
  1000x100 stop at 0.43–0.77% in 5–6 s. Validation 30 customers x 8 warehouses, evaluation
  500–1000 x 80–150, test 400–1000 x 60–150 (too easy ones are replaced).
- **Prompts:** [prompts/problems/cwlp.yaml](prompts/problems/cwlp.yaml), and the `few_shot`
  examples in `prompts/variants/few_shot/problems/cwlp.yaml`.
```bash
python prepare_instances.py --problem_name cwlp --data_dir ../data/a5000
python main.py --problem_name cwlp --model_name qwen3-coder:30b --data_dir ../data/a5000
```
On the lab machines, `fleet.py` works per problem. Its probe lists the problems whose sets are
ready in `data/<group>/` (`evaluation.pkl` and `test.pkl`), and `status` shows them when there is
more than the TSP. `start` and `bench` refuse a machine whose group has no sets for the run's
`--problem_name`, and `prepare --problem` builds them (the dashboard's prepare uses its
`problem_name` parameter):
```bash
python3 computer_connection/fleet.py prepare a4000 a5000 rtx3090 --problem cwlp
python3 computer_connection/fleet.py start a5000 -- --problem_name cwlp --fitness_mode lp
```

### Step 27: Third problem, the JSSP
The **Job Shop Scheduling Problem**, as in the reference (`configs/prompts/jssp.yaml`,
`data/data_prepare_jssp_rand.py`): every job runs one operation on every machine in its own
order, a machine runs one operation at a time, minimize the makespan.
- **Loader** ([src/loaders/jssp.py](src/loaders/jssp.py)): processing times 5–20, the machine
  order of each job a random permutation; jobs, positions and machines numbered from 0.
- **Model** ([src/models/jssp.py](src/models/jssp.py)): start times `S[j, k]`, makespan `Cmax`,
  an ordering binary `y` per pair of operations on the same machine, big-M disjunctions with
  `M` = the sum of all processing times. A brute force over the machine orders checks the optimum
  on tiny instances.
- **Seed cut:** the machine load bound, `Cmax >= total processing time of each machine`. Valid for
  every schedule, and **strong at the root**: it closes 75–100% of the root gap on 6x4, 10x5 and
  15x10, and 33% on 20x15. The "head + load + tail" refinement adds nothing on the first three and
  reaches 39% on 20x15 (`runs/analysis/jssp_sizes_20261102.txt`).
- **Sizes** (same file): up to 10x10 the base model solves to optimality in under 2 s; 15x5 to
  20x15 stop at **11.7–36.0% gap** (root LP gap 36–64%: the big-M relaxation is very weak).
  Validation 6 jobs x 4 machines, evaluation 15–20 jobs x 5–15 machines, test 12–20 x 5–15.
- **Prompts:** [prompts/problems/jssp.yaml](prompts/problems/jssp.yaml) and the `few_shot`
  examples in `prompts/variants/few_shot/problems/jssp.yaml`.

**Caveat, the validity check:** it keeps ONE stored optimal schedule, and a job shop has many
(operations with slack can move, non-critical machines can be ordered differently). A valid cut
that removes the stored schedule but keeps another optimal one is refused. This is the "relax the
validity check" item of TODO, which matters more here than for the TSP.

### Step 28: LLMs through an API (Groq, Gemini, OpenRouter)
To try stronger LLMs than the local ones, `--llm_backend` chooses where the LLM runs: `ollama`
(default, unchanged) or an OpenAI-compatible API. [src/llm/openai_compatible.py](src/llm/openai_compatible.py)
defines `OpenAICompatibleClient(model, base_url, key)`, which sends `POST <base_url>/chat/completions`
with the key as a Bearer token. One class covers every provider of `API_PROVIDERS`
(`src/utils/parameters.py`): `groq`, `gemini`, `openrouter`, `muse` (Meta's Model API: `muse-spark-1.x`,
`muse-wonder-1.0`) (base URL + name of the key's variable). A 429 (rate limit) or a 500–504
(server overloaded) is waited out (`Retry-After`, else `API_BUSY_WAIT` = 60 s, capped by
`API_MAX_WAIT`), up to `API_MAX_RETRIES` = 6 tries; a 429 naming a **daily** quota raises
`QuotaExhausted`, which is not retried and not counted as a failed attempt: the run stops (state
"failed", after its last saved generation). The night plan does NOT resume it (it would die
again within minutes, the quota resets once a day): resume it by hand, `fleet.py resume <host>`.
- **Rate limits** (HTTP 429, frequent on free tiers): the call waits as long as the server asks
  (`Retry-After`, at most `API_MAX_WAIT = 120` s), then `LLMClient` retries it (3 tries). If every
  try fails, the child is a copy of its parent, as with an unreachable Ollama.
- A `<think>...</think>` block of a reasoning model is removed from the answer.
- Every request sends `User-Agent: my-evo-cut/1.0`: Groq's Cloudflare refuses urllib's default
  `Python-urllib/3.x` (HTTP 403, "error code: 1010"), while the key itself was fine (checked with curl).
- **Keys:** in the environment (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `MUSE_API_KEY`), or one
  `NAME=key` line per provider in `llm_keys.env` at the project root (git-ignored; template:
  `llm_keys.env.example`). `fleet.py sync` copies it to the shared lab home, so the lab machines
  see it too.
- `make_llm` ([src/llm/factory.py](src/llm/factory.py)) builds the client for `main.py` and
  `prompt_bench.py`, and refuses a model the provider does not offer before the run starts.
  [src/list_models.py](src/list_models.py) lists a backend's models (and so checks the key).
- `fleet.py start/bench` do not start Ollama for a run with an API backend.

```bash
python list_models.py groq                                   # the model names, and the key works
python main.py --llm_backend groq --model_name <name> --problem_name tsp
python3 computer_connection/fleet.py start saumon -- --llm_backend gemini --model_name <name>
```
First result (one run of `muse-spark-1.3-contributor` per problem, same GA and validity check as
the local runs, `learnt_results.md` 5.11): on the held-out review the best cut scores 23.9 on TSP
(seed cut 13.15; `qwen3-coder:30b` 12.97 on average over 11 runs) and 18.6 on JSSP (15.95; 16.09
over 10 runs), 94–96% of its proposals are accepted, and a run costs $0.39–0.48. Table of every
run and per-LLM summary: `runs/analysis/runs_overview_20261112.txt` (script
`runs/analysis/runs_overview.py <first run date>`, from the project root),
`runs/analysis/llm_comparison_20261112.txt`, `runs/analysis/acceptance_rates_20261112.txt`.
The model name goes into the run's name and `runs/report.md` groups runs by it, as for local models.

### Step 29: Automatic nights (plan, resume, fetch)
Runs on the lab machines are cut at 06:33 on weekdays (and sometimes in the day: `learnt_results.md`
4.2), and a TSP run takes more than one night. So the nights are planned and resumed automatically
from the Mac (`fleet.py`, standard library only):
- **The plan:** `computer_connection/night_plan.json` (git-ignored, like the folder; template
  `night_plan.example.json`): a time `window` (18:30–08:00), `resume_days`, `force_below_mib`,
  the campaign's first day `since`, and `runs`: per problem, how many runs and extra `main.py`
  arguments. Edit it when the agenda changes.
- **`fleet.py night`:** probes every machine, then
  1. **resumes** each run cut before its end (stopped, dead, failed) that started less than
     `resume_days` ago, on its own machine (from its last saved generation), and not before `resume_since`
     (`YYYYMMDD_HHMM`, optional: e.g. runs made under an older validity check are never resumed),
     and not if it failed on a used-up API quota (`NOT_RESUMED_ERRORS`);
  2. **starts** the plan's runs not started yet, the problems alternating (tsp, cwlp, jssp, tsp, ...),
     each on a free machine whose group has the sets its arguments need (`needed_data`: the
     problem's sets at its `--instance_level`, plus the optimality sets of a `gap_work` run),
     machines with their own Gurobi license first (the WLS one still allows 2). An entry may name
     `groups` (e.g. `["rtx3090"]`): its runs go only there (same CPU type and local LLM for runs
     that are compared). When no free machine has the sets, the night job **builds them** on a
     free machine of a group that lacks them (`prepare_instances.py`, one build per group and set,
     none where one is already running), and a later tick starts the run. `count` is the number of runs of a kind IN ALL
     since `resume_since`, not per tick: the night job records each run it starts in
     `computer_connection/logs/night_state.json` ("started"; resumes and runs started by hand do not
     count). Before this fix (11 Nov), each 30-min tick started the plan again: 14 Gemini runs for 3;
  3. starts and resumes **nothing in the last 30 min** before the window ends (`LAST_LAUNCH_MARGIN_MIN`):
     the morning job would stop them before their first generation.

  A machine is free if nobody is at its screen, none of our runs is on it, and other users take
  at most `force_below_mib` (6000) of its GPU. Outside the window it starts nothing. `--dry_run`
  shows the decisions, and `--now` ignores the window.
- **`fleet.py report`** fetches `runs/` and rewrites `runs/report.md`, by default from the plan's
  `since`. The final review is written by the run itself at the end of the GA (on the shared lab
  disk), so fetching is enough to bring it back.
- **`fleet.py morning`** stops all our runs (SIGTERM: their state becomes "stopped", so the next
  night resumes them), so that nobody finds our runs on their machine in the day. It does nothing
  inside the night window: a 08:01 job missed while the Mac slept runs when it wakes up, and must
  not stop the night that has started. Nor on a weekend day (`LAUNCH_WEEKENDS`, since 13 Nov) or a
  day of `LAUNCH_ALL_DAY` (days the user allowed runs all day, e.g. the 11 Nov holiday, when it had
  stopped 13 runs at 08:05): those days the night job also launches in the daytime.
- **Code sync:** at a tick when none of our runs is alive (`sync` refuses otherwise: a run's solver
  re-imports the code at every solve), the night job copies the code and the validity sets first,
  so that planned runs of a new problem find it. On weekdays that is the first evening tick, after
  the morning job stopped everything.
- **Benchmarks of the plan:** an entry that finds no free a5000 machine is tried again at the next
  ticks of the night (13 Nov: the JSSP benchmark found none at 19:17 and was lost for the night);
  an entry launched on fewer machines than asked is not topped up (its shards are fixed at launch).
- **launchd** (not cron: a job missed while the Mac slept runs at wake-up; the window and the
  guard of `morning` keep that safe): `bash computer_connection/schedule.sh install | remove | status`
  installs `com.myevocut.night` (every 30 min: whenever the Mac wakes up in the night, and as soon
  as a machine frees up or a run is cut, e.g. by the lab's `tclean`), `com.myevocut.morning` (08:01)
  and `com.myevocut.fetch` (every 30 min). The logs are in
  `computer_connection/logs/`. **macOS must allow it** (System Settings → Privacy & Security →
  Full Disk Access → **+**, ⌘⇧G, paste the `python:` path that `schedule.sh install` prints),
  otherwise it fails with "Operation not permitted": the project is in `~/Documents`. The jobs run
  the real program behind `/usr/bin/python3` (Apple's, here
  `.../Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9`):
  macOS checks the permission of the program launchd starts, so neither the `/usr/bin/python3`
  shim nor a symlink works. Another Python: `PYTHON=/path/to/python3 bash computer_connection/schedule.sh install`
  (avoid Homebrew's: its path changes at every upgrade, which drops the permission). After an
  Xcode or Python update, rerun `install` and grant the new path printed.
```bash
python3 computer_connection/fleet.py night --dry_run --now     # what tonight would do
launchctl kickstart gui/$(id -u)/com.myevocut.fetch             # run the fetch job now (test)
```

### Step 30: The paper's test sets and protocol (`prepare_benchmark.py`, `benchmark_cut.py`)
To compare with the paper (`papers/sage_evocut_tmlr2026.pdf`), our cuts must be measured on its
held-out test sets, with its protocol (Section 5.1–5.2, Appendix F). The GA is unchanged: the
paper's development sets are random too; only this final test differs.
- **The sets** (`loaders/benchmarks.py`): TSP = the 54 TSPLIB instances with n < 250 (distances
  EUC_2D, ATT, GEO and explicit matrices, as the TSPLIB documentation defines them; the optimal
  tours shipped with the library give back the published optimal lengths); CWLP = the 36
  OR-Library instances feasible with single sourcing (CommaLab's cap61–capc4); JSSP = Taillard
  ta01–ta80. `prepare_benchmark.py` downloads them once into `data/benchmark/raw/<problem>/`,
  writes `data/benchmark/<problem>/benchmark.pkl` (no solve: the same folder for every CPU), and
  builds every base model: the mean number of variables and constraints equals the paper's
  Table 20 on all three (TSP constraints: +2n in the paper, which writes the bounds of u as
  constraints; `learnt_results.md` 5.1).
- **The protocol** (`benchmark_cut.py`, measure in `evolution/benchmark.py`): every instance is
  solved by the base model, then right after by the base model + cut, on the same machine,
  TimeLimit 2000 s, 8 threads, MIPGap 1e-5, Method 4, seed 0 (`parameters.BENCHMARK_*`). The Solver
  records the progress of each solve (`record_trajectory`: time, best objective, bound), so the gap
  at 5, 10, 50, 150, 300, 1000 and 2000 s comes from one solve. Measure: the paper's
  Δg = (g_ref − g_aug) / max(g_ref, 1e-5), mean ± std over the instances, and the row average;
  no solution yet = gap 100% (the paper does not say), a failed solve with the cut too. Two
  choices of ours, as the raw formula made the means meaningless (`learnt_results.md` 5.3): a
  gap ≤ 1e-5 (Gurobi's MIPGap) counts as 0, and Δg is clipped to [−100%, +100%]
  (`parameters.BENCHMARK_GAP_TOLERANCE`, `BENCHMARK_CLIP`).
- **Results**: one JSON per solve, `runs/benchmark/<problem>/tl<T>/<label>/<cpu>/<instance>.json`,
  written as soon as the solve ends (an interrupted run resumes; nothing is solved twice). The
  time limit is wall clock, so a cut is only compared with the base solved on the same CPU model.
  **One CPU model for the whole benchmark** (`parameters.BENCHMARK_CPU`, the a5000 machines' Xeon
  W-1270P): `benchmark_cut.py` refuses to solve elsewhere and `--report` reads only that CPU's
  results, so every instance of a table, and every cut, is solved on the same hardware
  (`--any_cpu` lifts both); `fleet.py benchmark` uses only the a5000 machines.
  `--shard i/n` takes every n-th instance (one share per machine); `--report` builds the table
  (`report.md` next to the results). Rows: Baseline (seed cut), Our best cut (`<host>_<date>`
  of the run), Paper cut (on our CPU) = `paper_sage`, Paper cut (announced) = the paper's Table 2,
  Gemini 3 Pro row. Last column, **LP gap closed**: how much the cut tightens the root LP
  relaxation, mean of (LP_cut − LP_base) / (UB − LP_base), UB = the best known solution (the base
  model's best, or the published TSPLIB optimum). `--lp` solves these LPs: deterministic and
  CPU-independent, so it runs on the Mac (`runs/benchmark/<problem>/tl<T>/lp/<label>/`).
  **Work units:** the GA scores cuts in Gurobi work (`WorkLimit 10`, as the paper's development
  phase); the paper's test is in seconds only, so the report keeps its table in seconds and adds a
  second one at the same checkpoints in work units (deterministic, CPU-independent): every
  trajectory point now also holds the work (`(time, best, bound, work)`, since 2026-11-09), and a
  solve stopped by the time limit before a work checkpoint is left out there (`[n=...]`).
- **Several cuts in one run**: `--cut seed`, `--population_file A B ...` (the best cut of each
  run, labelled with the run's name) and `--cut_file X.py ...` (labelled with the file's name) can
  be combined; on each instance the base is solved once, then every cut, one after the other.
- **The paper's own cuts** (`cuts/<problem>/paper_sage.py`): SAGE-TSP, SAGE-CWLP and SAGE-JSSP of
  Appendix F written for our models' variables, to calibrate: if they give back the paper's
  Table 2 on our machines and Gurobi 13, our protocol reproduces the paper's. SAGE-JSSP is the
  published family only (the paper's instantiated version adds first/last indicator variables,
  described too briefly to rebuild). SAGE-MCND is a reconstruction (Step 37: the paper's zone and
  bundle sampling is not given). `tests/test_paper_cuts.py` checks that each keeps the optimum.
- **Cost**: 2000 s × 2 solves per instance (the base once per CPU model, reused by every cut):
  170 instances ≈ 94 h for the base, the same per cut, so it is spread over the fleet:
  `fleet.py benchmark TARGETS -- <benchmark_cut.py arguments>` gives each free machine its
  `--shard k/n` (computer_connection/README.md).
```bash
python prepare_benchmark.py                          # all three sets (or --problem_name tsp)
python benchmark_cut.py --problem_name cwlp --cut best --population_file ../runs/<run>/population.json --shard 1/10
python benchmark_cut.py --problem_name jssp --cut seed --cut_file ../cuts/jssp/paper_sage.py \
    --population_file ../runs/<run>/population.json   # 3 cuts against one base
python benchmark_cut.py --problem_name tsp --cut seed --time_limit 300   # shorter: 5 checkpoints + 300 s
python benchmark_cut.py --problem_name cwlp --lp       # LP relaxations of the base and every cut (the LP column)
python benchmark_cut.py --problem_name cwlp --report
# from the project root, on the fleet (results fetched with fleet.py fetch):
python3 computer_connection/fleet.py benchmark dindon,harpie -- --problem_name jssp --cut seed
```

### Step 31: A validity check that invalid cuts cannot pass (`prepare_validity.py`)
Our best CWLP and JSSP cuts were invalid on the public sets (`learnt_results.md` 5.6): the check
used only the optimal solutions of 2 small random instances, which these cuts happened to keep.
Five layers now:
1. **Validity instances** (`loaders/validity.py`): small random instances of several sizes, down
   to tiny ones, and TIGHT instances built so that the classical bounds are the optimum (CWLP:
   capacity 1.02–1.09 × demand, expensive warehouses, big customers, one warehouse; JSSP: a
   bottleneck machine, a dominant job, flow shops). A cut must keep every optimal solution.
2. **Public instances outside the test sets**: 15 JSPLIB instances (ft06, la01–la05, la16–la20,
   abz5, abz6, ft10, orb01; our optima equal the published ones). None for the CWLP: the 13
   OR-Library instances outside the test set are infeasible with single sourcing.
3. **LP check at full size** (GAP mode): the LP relaxation with the cut of every evaluation
   instance must stay at or below a known solution's cost.
4. **Flag in the benchmark tables**: a cut whose bound or LP is above a known solution gets
   "(INVALID on k/n)" and the instances are listed.
5. **Invalid-cut retry message** (`prompts/common.yaml` `retry_invalid_user`): when a cut is
   refused as invalid (`CutRejected(invalid=True)`), the LLM is told why: the instance, its note,
   the cost removed, **which of its constraints the optimal solution violates and by how much**
   (`models/transforms.cut_violations`: the model with the cut, variables fixed to the optimum,
   only the cut's rows relaxed by Gurobi's `feasRelax`; unnamed rows are called `cut_row_<k>`),
   and **the instance's data** when it has at most 250 numbers (validity instances are checked
   smallest first, so the counterexample shown is the simplest). It is also told how such cuts
   usually go wrong, and asked for a proof in the IDEA.
Optimality does not depend on the CPU: the sets are built once on the Mac and `fleet.py sync`
copies `data/validity/` to the lab. `main.py` loads them (`--validity_dir`), with a warning if absent.
```bash
python prepare_validity.py                  # all three problems (or --problem_name jssp)
```
Result: the new check refuses our two invalid cuts and accepts every valid one (seed cuts, the
paper's cuts, our TSP cut): `runs/analysis/validity_check_of_benchmark_cuts_20261109.txt`. The
messages the LLM gets for them: `runs/analysis/refusal_messages_20261109.txt`.

**Not enough yet.** One more invalid CWLP cut passed the five layers (`learnt_results.md` 5.12).
It was found by a sixth test, for now a script run by hand on the Mac: solve the model with the
run's best cut for a few seconds on every public benchmark instance; the cut is invalid where the
model is infeasible or its proven bound is above a solution already known (the best saved result
of the base model, the seed cut and the paper's cut). Passing is not a proof of validity.
```bash
cd src    # <problem> <seconds per instance> <population.json of a run> ...
PYTHONPATH=. ../.venv/bin/python ../runs/analysis/public_bound_check.py tsp 20 ../runs/<run>/population.json
cd ..     # the short solves next to the stored 2000 s results of the benchmark (indicative)
python3 runs/analysis/short_check_vs_benchmark.py tsp runs/analysis/public_bound_check_tsp_20261112.txt
```
Results of 12 Nov: `runs/analysis/public_bound_check_{cwlp,tsp,jssp}_20261112.txt` (the three Muse
cuts pass; a qwen3-coder CWLP cut is invalid on 6/36) and
`runs/analysis/muse_tsp_20s_vs_benchmark_20261112.txt`.

### Step 32: Harder evaluation and test sets (`--instance_level hard`)
The best Muse TSP cut reaches the fitness ceiling (27.18: every evaluation instance solved) and
proves optimality on 5–6 of the 8 held-out instances, so a better cut cannot score higher. On
larger instances it is the other way round: from 125 cities it ends at a 53–100% gap under the
work limit, worse than the base model (`learnt_results.md` 5.15). So a second **instance level**
of the evaluation and test sets, opt-in:
- `problems.py`: `Problem.hard_sizes` (TSP: 60–200 cities, the default sizes and larger ones
  mixed; none yet for the CWLP and the JSSP) and `instance_set(problem, kind, level)`, which gives
  the file, seed and sizes of a set: `evaluation.pkl` / `test.pkl` at the default level,
  `evaluation_hard.pkl` / `test_hard.pkl` (seeds 2000 / 3000) at the hard one. The validation set
  is the same.
- `--instance_level` of `prepare_instances.py` and `main.py`: the run is scored and reviewed on
  the sets of its level, and the LLM is told their sizes ("60 to 200 cities"). It is one of the
  run's arguments, so a resumed run keeps its level and `review.json` records `instance_set` and
  `instance_level`. `review_run.py` refuses another level than the run's.
- `evolution/report.py` groups the runs by instance level too: two levels are never averaged.
- `fleet.py`: the probe lists `tsp/hard` when a group has the hard sets, `start` refuses a hard
  run without them, `prepare --level hard` builds them.
```bash
python prepare_instances.py --instance_level hard          # in src/, builds only the missing hard sets
python3 computer_connection/fleet.py prepare a5000 --problem tsp --level hard   # on the lab (night only)
python main.py --model_name ... --instance_level hard
```

### Step 33: Where each ± comes from
A mean is only useful with its spread, and the spread depends on what varies. Each report now
says which one it shows:
- **Review** (`run.log`, `summary.md`): ± = sample std over the 5 review seeds (same cut, same
  instances: the solver's noise). The best − seed difference is computed seed by seed, with the
  std of that difference (`review.paired_difference`).
- **Report over the runs** (`runs/report.md`): next to one run's score, the std over its review
  seeds; next to a mean over computers, "std between the k computers" (between runs of one LLM);
  a single value is written "(n=1: no std)" instead of "± 0". New **"Between LLMs" tables** per
  problem, fitness mode and instance level: each LLM's mean ± std between its runs, then the mean
  of the LLM means ± the std between the LLMs (held-out reviews only).
- **Benchmark** (`benchmark_cut.py --report`): a line above the tables says that "(± x)" is the
  std of Δg across the instances (one solve each), not solver noise nor the spread between runs.
- **Prompt benchmark**: ± over the accepted cuts of a variant.
Not done: the run fitness (mean over the 3 fitness seeds) is stored without its std in
`population.json`; the per-seed values are in `run.log`.

### Step 34: Re-check the best cut of past runs (`check_run_cuts.py`)
Runs started before the validity check of Step 31 could select an invalid cut. The script puts
the best cut of each saved run through the validity instances (`data/validity/<problem>/`, no lab
set needed: on the Mac). When it fails, the run's other cuts are tried from the fittest down (at
most 30) and the first one that passes is recorded. Result per run in `runs/<run>/validity.json`;
`evolution/report.py` reads it: a run whose best cut is invalid is marked "INVALID (k/n)" and
left out of the best of all, the means and the between-LLM tables.
```bash
python check_run_cuts.py                       # in src/: every run of ../runs (about 20 min for 112 runs)
python check_run_cuts.py --problem_name cwlp   # or ../runs/<run> ...
```
Result of 12 Nov: 16 of 112 runs have an invalid best cut (CWLP 15/31, JSSP 1/26, TSP 0/55), all
started before 9 Nov evening (`learnt_results.md` 5.16).

**Second layer, the public bounds** (`--public_seconds S`, the script of Step 31 moved here): on
every public benchmark instance the model with the run's best cut is solved for S seconds; the cut
is invalid there when the model is infeasible or its proven bound is above a solution already
known (the best saved benchmark result of the base model, the seed cut and the paper's cut). The
result goes into `validity.json` ("public"), and the report leaves such a run out too ("INVALID on
public (k/n)"). It caught ablette and hibou (CWLP), which the validity instances accept
(`learnt_results.md` 5.12, 5.18). It takes S seconds × the public instances per run (CWLP 36, TSP
54, JSSP 80), so it is run on chosen runs:
```bash
python check_run_cuts.py ../runs/<run> --public_seconds 20
```

### Step 35: A time-to-optimality term in the fitness (`--fitness_mode gap_work`)
The gap fitness only uses instances that the base model cannot finish, so it cannot see a cut
that slows down the *proof* of optimality (on ta01–ta04, Gurobi alone proves the optimum in
7–12 min, while with our JSSP cut it is still at a 0.3–9% gap at 2000 s). A third fitness mode,
opt-in so that the scale of the other runs does not change:
- **Optimality sets** (`prepare_instances.py --optimality`): 4 instances that the base model
  proves optimal under the work limit with a work of 1–5 (enough to measure, and room under the
  limit for a slower cut): `optimality.pkl` (seed 4000) for the run, `optimality_test.pkl` (seed
  5000) held out for the review. Sizes in `Problem.optimality_sizes` (TSP 40–60 cities, CWLP
  150–200 × 30–40, JSSP 10 × 5–10; measured in `runs/analysis/tsp_optimality_sizes_20261112.txt`,
  `cwlp_sizes_20261102.txt`, `jssp_sizes_20261102.txt`).
- **Score** (`evaluator.FitnessMode.GAP_WORK`, `review.work_ratios`): on each optimality instance,
  under each seed, r = (work_cut − work_base) / work_base, with work_cut = the work limit when the
  cut does not prove optimality (a slowdown is underestimated, never overestimated); the r are
  pooled with the gap ratios of the evaluation instances, each instance counting the same:
  fitness_s = 10 · exp(−mean of all r). 10 = no change.
- **Validity**: the optimum of these instances is known, so a bound or an LP value above it
  refuses the cut as invalid.
- **Review**: the held-out gap fitness as before (comparable with the other runs), plus the work
  change on `optimality_test.pkl` (mean ± std over the seeds) in `review.json`
  (`work_change_mean`, `work_change_std`) and `summary.md`.
- The LLM is told the formula (`prompts/fitness.yaml` `gap_work`); `fleet.py` refuses a gap_work
  run on a group without the optimality sets, `prepare --optimality` builds them.
```bash
python prepare_instances.py --optimality       # in src/: optimality.pkl and optimality_test.pkl
python main.py --model_name ... --fitness_mode gap_work
```
Each cut costs 4 more solves per fitness seed, next to the 8 of the evaluation set (not timed yet). No
run has used it yet.

### Step 36: A run reviewed on another level's test set (`review_run.py --review_level`)
To know whether the hard sets of Step 32 help, the best cut of a default-level run must also be
measured on the hard test set. `review_run.py --review_level hard` does the review of Step 22 on
`test_hard.pkl` and writes `runs/<run>/review_hard.json`; the run's own `review.json` and
`summary.md` are kept.
```bash
python review_run.py --population_file ../runs/<run>/population.json --data_dir ../data/rtx3090 --review_level hard
```

### Step 37: Fourth problem, the MCND (multi-commodity capacitated fixed-charge network design)
The paper's Table 2 has it (Gemini 3 Pro 14.9, GPT-5.1 17.9), with a public test set: group R of
CommaLab, r10.1–r18.9 (81 instances: 20 nodes, 120–318 arcs, 40–200 commodities).
- **Model** (`models/mcnd.py`): the paper's compact arc-based formulation (Appendix F.2): flows
  x[i, j, k] ≥ 0, arc openings y[i, j]; the flow leaving the origin and entering the destination
  equal d[k], other nodes balanced; total flow ≤ u[i, j] y[i, j]. On the 81 test instances it has
  24991.89 ± 17962.83 variables and 2485.22 ± 1330.64 constraints, the paper's Table 20 exactly.
- **Random instances** (`loaders/mcnd.py`), shaped like R (`runs/analysis/mcnd_r_set_20261113.txt`):
  a cycle through every node made of uncapacitated arcs (R has exactly n such arcs: every
  instance is feasible), random other arcs whose capacity is a few mean demands × a tightness
  level (1, 1/2, 1/8), unit costs 10–100, demands 10–100, fixed cost ≈ unit cost × 0.35–0.75 ×
  number of commodities × a level (1, 5, 10).
- **Public test set** (`loaders/benchmarks.py` `parse_dow`, `commalab_r_instances`):
  `prepare_benchmark.py --problem_name mcnd` downloads R.tgz once and keeps r10–r18.
- **Sizes** (`problems.py`, measured in `runs/analysis/mcnd_sizes_20261113.txt`): validation 8
  nodes × 25 arcs × 10 commodities; evaluation 20 × 220–300 × 100–200 (the base model stops at
  0.75–5.44% under the work limit); test 20 × 200–300 × 80–200; optimality sets (gap_work) 20 ×
  120 × 40–100.
- **Seed cut**: enough open arcs out of every origin, sum_j y[s, j] ≥ q_s (fewest largest
  capacities covering the supply of s); it closes 0–31% of the root gap, the strong linking
  x[i, j, k] ≤ min(u, d[k]) y[i, j], left for the LLM to find, 90–100%.
- **Validity instances** (`loaders/validity.py` `mcnd_instances`): small random ones (loose and
  tight), a demand that must be split over two paths, a network that is one cycle, expensive arcs,
  and 27 public R instances outside the test set (r01–r09, variants .1, .5, .9).
- **Prompts**: `prompts/problems/mcnd.yaml`, and the few-shot examples
  `prompts/variants/few_shot/problems/mcnd.yaml`.
```bash
python prepare_benchmark.py --problem_name mcnd     # in src/: the 81 test instances
python prepare_validity.py --problem_name mcnd      # the validity instances (Mac, then fleet.py sync)
python prepare_instances.py --problem_name mcnd     # on each lab group: fleet.py prepare <group> --problem mcnd
python main.py --model_name ... --problem_name mcnd
```
- **The paper's cut** (`cuts/mcnd/paper_sage.py`), a RECONSTRUCTION: the paper's rows
  (knapsack-cover cut-set and cardinality, Eq. SAGE-MCND) over breadth-first zones around the
  origins, but its sampling of zones and bundles is not given; here every zone and, as bundles,
  all the crossing commodities plus the 3 largest alone. It keeps r01.1's optimum and tightens its LP.

Not done yet: the sets on the lab, the 2000 s benchmark of the base model and the paper's cut, a run.

### Step 38: Fifth problem, IMO6 (IMO 2025 Problem 6 as a MILP)
The paper's Table 2 has it (GPT-5.1 31.1, Gemini 3 Pro 21.7). On an N x N grid, place rectangular
tiles so that every row and every column has exactly one uncovered cell; minimize the number of
tiles (the olympiad's answer: N + ceil(2 sqrt N) - 3). It is new as a MILP, so the LLM cannot
take its cuts from the optimization literature.
- **Model** (`models/imo6.py`): the paper's RT-2DFlow (Appendix F.6): holes h[i, j], strips
  x[a, b, i] (columns a..b covered by one tile on row i), starts s and ends t of the tiles along
  each column interval; the objective counts the starts. On the paper's test set (N = 3..36) it
  has 20759.50 variables and 7524.50 constraints on average, Table 20 exactly. Its optimum equals
  the olympiad's answer for N = 2..8 (`runs/analysis/imo6_sizes_20261113.txt`).
- **Instances** (`loaders/imo6.py`): an instance is only N, so a set draws distinct N. As in the
  paper: evaluation (its development pool) N = 37..46; test N = 3..36, the benchmark
  (`loaders/benchmarks.py` `imo6_benchmark`, nothing to download); our review set draws 8 sizes
  from 20..36, where the base model does not finish; validation N = 5..6. No optimality sets: only
  N = 7 is proved with a work of 1–5.
- **Base model** under the work limit: proves optimality up to N = 8 (14.6 s), stops at a 50–98%
  gap from N = 8 to 46; its root LP bound is below 1 (the optimum: 11 to 57).
- **Seed cut**: covered components of a row, sum x[., ., i] ≥ 2 − h[i, 1] − h[i, N]; it closes
  2.6–6.9% of the root gap. "A tile starts in every row after the first" alone closes 69–78%, so
  it is not given.
- **Validity instances** (`loaders/validity.py` `imo6_instances`): N = 2..8, solved to optimality.
- **Prompts** (`prompts/problems/imo6.yaml`, few-shot examples in
  `prompts/variants/few_shot/problems/imo6.yaml`): the olympiad is NOT named and its answer is not
  given, nor in the model's docstring (which the LLM reads) nor in the validity notes (quoted to
  it): "objective ≥ N + ceil(2 sqrt N) - 3" is a valid cut that closes the whole gap and would
  measure what the LLM remembers, not the cuts it designs. `tests/test_prompts.py` checks it.
- **The paper's cut** (`cuts/imo6/paper_sage.py`, SAGE-RT-Disrupt): a transcription, the paper
  gives the whole family. It keeps the optimum and lifts the LP bound to N; it builds in 3.3 s at
  N = 36 (11.5 M nonzeros), 10.6 s at N = 46.
```bash
python prepare_benchmark.py --problem_name imo6     # in src/: the 34 test grids (no download)
python prepare_validity.py --problem_name imo6      # N = 2..8 (Mac, then fleet.py sync)
python prepare_instances.py --problem_name imo6     # on each lab group: fleet.py prepare <group> --problem imo6
python main.py --model_name ... --problem_name imo6
```
Not done yet: the sets on the lab, the benchmark of the base model and the paper's cut, a run.

### Step 39: A less-known problem, GISP (generalized independent set)
Not in the paper. The first of the less-known problems (TODO P3.11): few published cuts (one
polyhedral paper, Colombi, Mansini and Savelsbergh 2017), so a gain is less likely to come from
what the LLM remembers. Choose nodes of a graph: two chosen nodes must not share a permanent edge,
and may share a removable edge only if it is removed, at a cost; maximize weight x chosen nodes
minus cost x removed edges (Hochbaum and Pathria 1997, forest harvesting).
- **Model** (`models/gisp.py`): the formulation of Distributional MIPLIB's generator (GISP/gisp.py):
  x[i] chosen, y[i, j] removed; x[i] + x[j] <= 1 on permanent edges, x[i] + x[j] - y[i, j] <= 1 on
  removable ones; the objective negated (our models minimize), as their files write it.
- **Instances** (`loaders/gisp.py`): their distribution, G(n, 0.3) with each edge removable with
  probability 0.25, weight 100, cost 1, drawn with our own generator (no numpy / networkx): the
  same distribution, not the same graphs. Validation 40 nodes, evaluation 150–175, test 140–200,
  optimality 90–110.
- **Base model** under the work limit (`runs/analysis/gisp_sizes_20261113.txt`): proves optimality
  up to 100 nodes (work 1.6–1.8), stops at a 16–114% gap from 125 nodes on, in about 3 s.
- **Seed cut**: triangles with two permanent edges and one removable, x[i] + x[j] + x[k] <= 1 +
  y[j, k]; it closes 4.5–7.7% of the root gap. The textbook clique cut on permanent triangles
  closes 45–61% alone, so it is left for the LLM to find.
- **Validity instances** (`loaders/validity.py` `gisp_instances`): 7 random graphs of 6 to 40
  nodes and 7 tight ones (no edge, permanent and removable cliques, an odd cycle with and without a
  removable edge, a break-even cost, a complete bipartite graph).
- **Benchmark** (`loaders/benchmarks.py` `gisp_benchmark`): Distributional MIPLIB's GISP "hard"
  test set, 100 graphs of 175 nodes (over 1000 s for Gurobi in their measurements), downloaded from
  Hugging Face (95 MB) and read back from their LP / MPS text into our instances: the same mean
  size as their files (1313.49 variables, 4560.65 constraints). Their "easy" level has 115 nodes,
  "medium" 150; the "very-hard" files have anonymous variable names.
- **Not done:** no "paper's cut" (not in the paper); the sets on the lab, the base model on the
  benchmark, a run.
```bash
python prepare_benchmark.py --problem_name gisp     # in src/: downloads the D-MIPLIB test set
python prepare_validity.py --problem_name gisp      # the 14 validity graphs (Mac, then fleet.py sync)
python prepare_instances.py --problem_name gisp     # on each lab group: fleet.py prepare <group> --problem gisp
python main.py --model_name ... --problem_name gisp
```

### Step 40: A second less-known problem, wildlife corridor design
Not in the paper. Buy land cells within a budget so that the bought cells form a connected
subgraph containing every reserve; maximize their utility (the connected subgraph problem with node
costs and profits: Conrad et al. 2012, Dilkina and Gomes 2010). Less studied than TSP or CWLP, but
connectivity cuts are textbook for Steiner-type problems, so "less known", not unknown. SRPN and
MMCN were not possible: their public files are anonymous and they have no public generator.
- **Model** (`models/corridor.py`): the single-commodity flow MIP of the corridor generator
  (Corridor/lib/corMIPGen.py of the Distributional MIPLIB generators): x[i] bought; the source sends
  y = sum x into the first reserve, every bought cell keeps one unit, f[i, j] <= n x[j]; budget row;
  reserves fixed to 1. Their slack x_0 (x_0 + y = n) is left out (implied).
- **Instances** (`loaders/corridor.py`): the generator's settings: order x order lattice, reserves at
  two opposite corners plus one random cell (free), costs and utilities uniform on 1..100, budget
  20/22.5/25/27.5/30% of the total cost. Some draws are infeasible (the budget cannot connect the
  reserves): `build_set` refuses them; `connectable` is a sufficient check (cheapest paths).
  Validation order 7, evaluation 20–25, test 18–30, optimality 12–13.
- **Base model** under the work limit (`runs/analysis/corridor_sizes_20261113.txt`): proves
  optimality up to order 12 (work 0.15–6.1), stops at 1.3–60% gaps from order 15 on.
- **Seed cut**: flow only leaves bought cells, f[i, j] <= n x[i]: it closes 0.4–1.2% of the root
  gap. "A bought cell has a bought neighbour" closes 25–39% alone, so it is left for the LLM.
- **Validity instances** (`loaders/validity.py` `corridor_instances`): 6 random lattices of order
  3 to 8 and 5 tight graphs (a path where every cell must be bought, a budget that buys exactly the
  cheapest corridor or everything, a cycle, a star: not lattices).
- **Benchmark** (`loaders/benchmarks.py` `corridor_benchmark`): no public set, so 50 lattices of
  order 20 to 30 generated once (seed 2026, connectable ones only); mean size 2878.02 variables,
  2882.02 constraints.
```bash
python prepare_benchmark.py --problem_name corridor  # in src/: the 50 generated lattices
python prepare_validity.py --problem_name corridor   # 11 validity graphs (Mac, then fleet.py sync)
python prepare_instances.py --problem_name corridor  # on each lab group: fleet.py prepare <group> --problem corridor
python main.py --model_name ... --problem_name corridor
```

### Step 41: The second pipeline, proofs in Lean 4 (`prove_cut.py`)
The tests of a run can only say that a cut is wrong; three invalid cuts passed them
(`learnt_results.md` 5.6, 5.16, 5.18). This pipeline looks for a **proof**: a Lean 4 theorem saying
that the cut is valid for every instance, checked by Lean, not by us. It is the method of FLARE
(Robbins et al. 2026, `papers/flare_robbins2026.txt`), adapted to cuts.

**What is proved** (`lean/EvoCut/Common.lean`). A formulation is FLARE's `MILPFormulation`
(instances, variables, the integer feasible set, the objective). For a cut `C` on it:
- `ValidCut`: every feasible point of every instance satisfies `C` (a valid inequality). Proved
  once in Lean: it gives FLARE's own certificate, a `MILPReformulation` between the model and the
  model with the cut.
- `SafeCut`: `C` may remove feasible points, but for each one it keeps a point at least as good
  (symmetry breaking, dominance). FLARE's definition refuses these. Proved once in Lean: the model
  with the cut keeps an optimal solution of the same value, and each of its optimal solutions is
  optimal for the model without it (Audet's reformulation, Definition 4.1 of the paper).

**What is trusted, what is not.**
- Trusted, written by hand, to review once: `Common.lean` and one file per problem
  (`lean/EvoCut/Problems/Tsp.lean`, `Cwlp.lean`, `Jssp.lean`), each a line-by-line copy of its
  `create_model`, with a table Python name → Lean name. FLARE lets the agent write both models at
  every run; here the base model is fixed, only the cut is translated.
- Each Lean model states its **data assumptions** (TSP: at least 4 cities, costs ≥ 0; CWLP:
  demands > 0, capacities and costs ≥ 0; JSSP: times > 0, every job visits every machine once). A
  proof holds for the instances that satisfy them; `proving/problems.py` has the same list in
  Python and `prove_cut.py` checks it on the validity and benchmark sets of the Mac.
- The cut's **statement** in Lean is written by the LLM, so it is the weak point: a proof is
  about the statement, not about the Python code. It is frozen before the proof starts, a second
  LLM call judges it against the code (`--no_judge` skips it), and it is saved alone
  (`statement.lean`, a few lines) to be read by a human.
- The **proof** is not trusted at all (`proving/certificate.py` `verify`): refused words
  (`sorry`, `axiom`, `native_decide`, `macro`, `notation`, `set_option` other than two...), then
  `CutProof.lean` is compiled into a module, then `Check.lean`, a separate file that only imports
  it, states the claim with full names and prints the axioms of the theorem. Proved = both
  compile and the axioms are among `propext`, `Classical.choice`, `Quot.sound`. The separate
  file matters: a proof text that slips a false hypothesis into its theorem (`variable {h :
  False}` + `include h`) compiles, and is refused there (tested).

**Who writes the proof** (`--harness`).
- `loop` (default, `proving/prover.py`): any LLM of the project (`--llm_backend ollama | muse |
  gemini | ...`, or `claude_code`: Claude through the `claude` command, no API key). One answer
  per round; each is compiled and the errors, with the goal at the failing step, are sent back
  (4 statement rounds, 8 proof rounds by default). An intermediate answer may contain `sorry`.
- `agent` (`proving/agent.py`): FLARE's way. A Claude Code agent (`claude -p`) works in
  `lean/work/<problem>_<label>/`, edits `CutProof.lean` under a marker line, compiles with
  `./check.sh` and queries the goal at any line through the lean-lsp MCP server. No Docker: its
  only shell command is `./check.sh` and it may only edit its folder. When it stops, what is
  under the marker goes through the same `verify`.

**Results** (`result.json`): `proved` (a certificate), `not_proved` (nothing found in the rounds:
says nothing about the cut), `no_statement`, `claimed_invalid` (the LLM's claim, not checked).
```bash
brew install elan-init && cd lean && lake exe cache get && lake build   # once: Lean, Mathlib (9.3 GB with the toolchain)
cd src
python prove_cut.py --problem_name tsp --cut seed --llm_backend muse --model_name muse-spark-1.3-contributor
python prove_cut.py --problem_name cwlp --population_file ../runs/<run>/population.json --llm_backend ... --model_name ...
python prove_cut.py --problem_name tsp --cut_file ../cuts/tsp/invalid_position_bound.py --idea "..." ...   # the control
python prove_cut.py ... --statement_file ../runs/proofs/tsp/seed/statement.lean      # a reviewed statement
python prove_cut.py ... --harness agent --agent_model opus --agent_minutes 30         # the proof by a Claude Code agent
```
A compile takes 8–12 s (`import Mathlib.Tactic`; a full `import Mathlib` takes 55 s, so files
import only what they need, and an `import Mathlib.X` line of the LLM is moved to the top).
Lean runs on the Mac only: nothing of this is on the lab machines.

**First runs** (14 Nov, `learnt_results.md` 11.1, `runs/analysis/proofs_first_runs_20261114.txt`),
Muse Spark 1.3 in the loop, each at its first proof round, $0.03 in all:
- JSSP seed cut (machine load): **proved**, 203 lines, 7.9 min.
- CWLP seed cut (at least `k_min` open warehouses): first answer `INVALID`, with a true
  counterexample: with no customer no warehouse has to be open. The Lean model had no "at least
  one customer" assumption; with it added, **proved**, 147 lines, 5.8 min.
- The first control, "no triangle through the depot" (invalid with 3 cities, FLARE Prop. F.3; the
  model then assumed 3 cities or more; it now assumes 4 and the control is
  `cuts/tsp/invalid_position_bound.py`): no proof, and the
  right counterexample (3 cities).
- TSP seed cut: proved, but its proof is the example shown in the prompt (99.3% identical): it
  only shows that the pipeline runs.
- The Claude Code agent (Sonnet) proved the textbook CWLP linking `x[i, j] <= y[j]` in 45 s
  (7 turns, $0.22 at API prices), from a statement written by hand.

**Later the same day** (`learnt_results.md` 11.2, 11.3; the full report: `lean_report_20261114.md`):
- Opus 5.5 as the agent proved the paper's three cuts (3–5 min and about $1 each; the TSP one
  needs at least 4 cities, now an assumption of the model) and the best cut of the Muse TSP run
  (147 lines of code: a 1,038-line proof, 17 min, $3.77).
- Muse in the loop proved the paper's CWLP and JSSP cuts (round 3) and failed its TSP cut.
- qwen3-coder:30b in the loop proved none of the three seed cuts.

Not done: MCND, IMO6, GISP and corridor have no Lean model yet; the proofs are not joined to the
end of a run; a failed proof is not yet fed back to the GA; nothing checks the statement against
the code on concrete instances (it would need the Lean models over a computable field).

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
| [test_individual.py](tests/test_individual.py) | defaults, unique ids, `parents` lists not shared, dict round trip, `copy` makes a new individual (new id, same cut and fitness, parent recorded), other generators refused, `normalized_code` ignores comments and spaces |
| [test_population.py](tests/test_population.py) | wrong size and unevaluated individuals refused, history, best kept across worse generations and on ties, `top`, same id twice refused (an elite must be copied), `all_individuals` skips copies, an equal elite copy keeps the original as best, save/load |
| [test_prompts.py](tests/test_prompts.py) | YAML files load, expected operators present, every system prompt fills with the real `create_model` source and leaves no placeholder, user templates fill; `Prompts` builds mutation / crossover / retry messages; `parse_reply` reads the format (bold labels, extra text, no backticks, indentation) and refuses bad ones; `initializer.yaml` keys, initializer messages (reference code, ideas list, best fitness), every prompt variant fills without placeholder and keeps the IDEA/CODE format, the problem is described from `problems/<name>.yaml` (name, description, sizes, solution in the fitness text), no template outside `problems/` mentions the TSP, `few_shot` takes its examples from the problem, a placeholder without value and an unknown problem are refused, `fitness.yaml` has one text per `FitnessMode` and every variant shows the text of the chosen mode (and not the other), unknown fitness mode refused, variants override keys (common, initializer) or whole files (operators), unknown variant refused |
| [test_llm_base.py](tests/test_llm_base.py) | `LLMClient` is abstract, failed calls retried, `LLMError` after the last try, `clean` hook applied |
| [test_llm_fake.py](tests/test_llm_fake.py) | replies in order, each conversation recorded as sent, no reply left → `LLMError` |
| [test_genetic.py](tests/test_genetic.py) | `initialize`: evaluated seed first then new cuts, the initializer sees the seed and the growing list of ideas, failed slot → copy of the seed, rejected baseline and second `initialize` raise; generation size kept and elite copied, `crossover_rate` picks the operator, bad format / rejected cut sent back and retried, parent copied when every attempt fails, unreachable LLM gives up at once, best fitness shown to the LLM, tournament, population of 1, same seed → same run, `progress` called before each child (not for the seed or the elites), every attempt recorded (format error, rejected with its reason, accepted with fitness and individual id, conversation and timings), the first generation has its own number of attempts, an initial cut repeating the generation is refused without solving |
| [test_llm_openai_compatible.py](tests/test_llm_openai_compatible.py) | against a fake OpenAI-style server: conversation and options sent with the key, answer returned, models listed (Gemini's `models/` names too), wrong key or model retried then `LLMError`, a rate limit waited out (`Retry-After`, capped), a `<think>` block removed, key from the environment or `llm_keys.env` |
| [test_llm_factory.py](tests/test_llm_factory.py) | the backends, Ollama and API clients built with the provider's URL and key, missing key and unknown backend refused |
| [test_list_models.py](tests/test_list_models.py) | the script prints a backend's models, usage on an unknown backend |
| [test_llm_ollama.py](tests/test_llm_ollama.py) | against a fake Ollama HTTP server: conversation sent and answer returned, `available_models`, options forwarded, unknown model retried then `LLMError`, unreachable server; plus a live test, run only with `OLLAMA_LIVE=1` (needs the tunnel) |
| [test_sets.py](tests/test_sets.py) | validation set solved to optimality with solutions, saved then reloaded without solving, evaluation set keeps no solution, non-optimal validation instance refused (nothing saved), missing `"base"` formulation refused, refused instances replaced (new names), gives up when nothing is accepted |
| [test_loaders_cwlp.py](tests/test_loaders_cwlp.py) | generated CWLP instances respect the reference's ranges (demands, costs, capacity buffer) and names, reproducible with a seed, folder `data/cwlp` |
| [test_models_cwlp.py](tests/test_models_cwlp.py) | the CWLP optimum matches a brute force on tiny feasible instances, model size and attributes (minimizes), the seed and linking cuts keep the optimum, the seed cut passes the validity and usefulness checks on the real validation set, the seed is weaker than the linking on 100x20 |
| [test_loaders_validity.py](tests/test_loaders_validity.py) | every validity instance has a name and a note, reproducible, the tight ones are tight (capacity, big customers, dominant job, flow shop), the polygon's tour is its perimeter, JSPLIB files read without their comments (ft06, la01 in `tests/data/jsplib/`); the GISP graphs (cliques of 15 edges, the break-even cost) |
| [test_prepare_validity.py](tests/test_prepare_validity.py) | the instances are solved and the optimal ones saved with their solution, a wrong published optimum stops, infeasible candidates are left out |
| [test_loaders_benchmarks.py](tests/test_loaders_benchmarks.py) | TSPLIB: each distance format (GEO, ATT, EUC_2D, explicit FULL_MATRIX / UPPER_ROW / LOWER_DIAG_ROW, UPPER_DIAG_ROW) gives the published optimum on a real instance, symmetric matrices, rounding, unsupported files refused, size filter + optimum from the shipped tour; Beasley: cap61 and the per-warehouse cost rows, size check, zip reading; Taillard: ta01, checks, downloaded files used; no download when the file exists; GISP: Distributional MIPLIB rows in LP and MPS read back into the same instance, another model refused; the generated corridor set is fixed, connectable, of the recorded size |
| [test_prepare_benchmark.py](tests/test_prepare_benchmark.py) | defaults, size comparison with the paper, the set is saved and reloads, warning when the count differs from the paper, model size of ta01 |
| [test_benchmark.py](tests/test_benchmark.py) | gap (no solution, cap, zero), gap at a time from the trajectory (before, between, after the end, failed solve), the paper's Δg with epsilon, the gap tolerance (two optimal solves differ by 0) and the clip to ±100%, the LP gap closed and its column (absent without LP, `-` for the paper), comparison (failed and missing instances), table with the paper's rows only at its checkpoints, paper rows consistent with their averages, JSON round trip, shards |
| [test_benchmark_cut.py](tests/test_benchmark_cut.py) | base + cut saved per CPU with the cut's code, a second run solves nothing, report; `--base_only` and `--shard`; cut from the seed, a file or a run's best; several cuts in order, the base solved once per instance; another CPU refused and left out of the report (`--any_cpu` allows it); `--label` with one cut only, duplicate labels and label `base` refused; instance selection; never pairs two CPUs ; `--lp` solves the LP relaxations and the report gets the LP gap closed column; the note on what the ± of a cell is |
| [test_paper_cuts.py](tests/test_paper_cuts.py) | the paper's SAGE cuts (`cuts/<problem>/paper_sage.py`, MCND's a reconstruction that also tightens r01.1's LP) keep the optimum: ulysses16 (TSPLIB optimum 6859), a reduced cap61, a reduced ta01, r01.1, the 5 x 5 grid (7 tiles); the IMO6 family lifts the LP to N, as the paper says |
| [test_loaders_jssp.py](tests/test_loaders_jssp.py) | generated job shops: sizes, times 5–20, every machine once per job in a random order, names, reproducible, folder `data/jssp` |
| [test_models_jssp.py](tests/test_models_jssp.py) | the JSSP optimum matches a brute force over the machine orders, model size and attributes, the seed cut keeps the optimum and passes the checks on the real validation set, it tightens the big-M relaxation |
| [test_loaders_imo6.py](tests/test_loaders_imo6.py) | grid sizes drawn without replacement, in order, reproducible; more instances than sizes refused; an instance is only its N |
| [test_models_imo6.py](tests/test_models_imo6.py) | the optimum equals a brute force (every hole permutation, fewest rectangles) and the olympiad's N + ceil(2 sqrt N) - 3 for N = 2..5, the solution is a tiling, model size as the paper's formulation, the seed cut keeps the optimum and lifts the LP |
| [test_loaders_corridor.py](tests/test_loaders_corridor.py) | reproducible, lattice sizes and edges, 3 reserves with the two corners, free reserves, costs and utilities in range, the budget one of the generator's fractions; `connectable` on a path and a split graph |
| [test_models_corridor.py](tests/test_models_corridor.py) | the optimum equals a brute force over connected cell sets within the budget, it minimizes, the bought cells are connected and hold every reserve; the data and sizes of the model; the tight validity graphs have their known optima; the seed cut keeps every validity optimum and lifts the LP |
| [test_loaders_gisp.py](tests/test_loaders_gisp.py) | reproducible, sizes, weight 100 and cost 1, distinct edges i < j; the density (0.3) and share of removable edges (0.25) of the generator; G(n, p) and the edge partition |
| [test_models_gisp.py](tests/test_models_gisp.py) | the optimum equals a brute force over node sets, it minimizes, no permanent edge inside, removable edges inside removed; the data on the model; the seed cut keeps the optimum of every validity instance and lifts the LP (x = 1/2 everywhere without it); the tight validity graphs have their known optima |
| [test_loaders_mclp.py](tests/test_loaders_mclp.py) | reproducible, sizes and P = ceil(1% of the sites) as the reference, populations in range; the grid-bucketed coverage sets equal a full distance check (a node out of reach gets one random site) |
| [test_models_mclp.py](tests/test_models_mclp.py) | the optimum equals a brute force over every choice of P sites, it minimizes, covered nodes have an open site in reach; the data is on the model |
| [test_loaders_mcnd.py](tests/test_loaders_mcnd.py) | generated MCND instances: sizes and ranges, distinct arcs and (origin, destination) pairs, the uncapacitated cycle (every instance feasible), capacities and fixed costs within their ranges, names, reproducible, sizes clipped to the graph |
| [test_models_mcnd.py](tests/test_models_mcnd.py) | the optimum equals a brute force over the open arcs (routing by an independent textbook LP), model size as the paper's Table 20 counts, attributes and row names, the strong linking keeps the optimum and raises the LP, the seed cut keeps the optimum and tightens the LP of a validation-size instance |
| [test_problems.py](tests/test_problems.py) | the TSP entry and its set sizes, sizes described for the LLM ("60 to 100 cities"), every problem generates instances with its sizes, a maximizing base model is refused; instance levels: the default files and seeds kept, the hard level's own files, seeds and larger sizes, unknown levels and problems without hard sizes refused; optimality sizes smaller than the evaluation ones; the IMO6, MCND, GISP and corridor entries |
| [test_prepare_instances.py](tests/test_prepare_instances.py) | script flags (`--problem_name` too) default to `parameters.py` and can be overridden (`--data_dir` too), `has_gap` accepts only unsolved instances, test-set defaults (8 instances, 50–100 cities, seeds disjoint from the other sets); `--instance_level`, `--optimality`; an optimality instance is proved optimal with a work of 1–5 |
| [test_review.py](tests/test_review.py) | under several seeds: the base model scores 10 at every seed, one score per seed and per model with its gaps, mean and std, JSON-serializable, table with "beats the seed cut on k/n seeds"; the score formula, instances solved by the base skipped, failed solves counted as gap 1; work ratios to prove optimality (not proved = the work limit, skipped when the base does not prove it), pooled with the gap ratios; the review on optimality instances (work change mean ± std over seeds); the paired best − seed difference |
| [test_recorder.py](tests/test_recorder.py) | one file per attempt in its generation folder, a resumed generation does not overwrite the earlier attempts |
| [test_summary.py](tests/test_summary.py) | the summary has every section (state and duration, settings, best cut with fitness, generation found and code, review with "beats the seed cut on k/n seeds" and the held-out set it used, fitness per generation including the last one, acceptance per operator, rejection reasons), a run without review or proposals, the summarize script; ± said to be over the review seeds, paired difference with its std; the work line of a gap_work review |
| [test_report.py](tests/test_report.py) | a run folder read into a row (run and review fitness, LP gap closed, held out or not, best cut), old runs without fitness mode or review, a killed "running" run shown as stalled, groups per LLM and fitness mode, each computer's best reviewed run, the report's best of all (by the review, not the run fitness) and means over the computers, runs ordered by review score; instance levels never mixed; each ± says its source, no std for one value; between-LLM tables (held-out reviews only); a run whose best cut is invalid (`validity.json`) left out |
| [test_check_run_cuts.py](tests/test_check_run_cuts.py) | known solutions = the best trusted benchmark results, a bound above one is invalid, the public layer recorded in `validity.json` and in the line printed; a cut that keeps every optimum / removes them, the run's cuts ranked fittest first with each code once, an invalid best cut flagged with the run's best valid cut, a valid best cut checked once, `validity.json` written, every run of a folder (and `--problem_name`) |
| [test_report_runs.py](tests/test_report_runs.py) | the date of a run from its name, run folders found with `--since`, the script writes `report.md` |
| [test_proving_lean.py](tests/test_proving_lean.py) | compiler messages read from `lean --json`, axioms of a `#print axioms` line, comments removed, refused words and options; what is missing when Lean is not installed; with Lean: errors with their line, a compile killed at its timeout |
| [test_proving_problems.py](tests/test_proving_problems.py) | every Lean problem has its Lean file and namespace; generated and validity instances satisfy the data assumptions; each broken assumption is named |
| [test_proving_certificate.py](tests/test_proving_certificate.py) | the last lean block of an answer, imports moved to the header and non-Mathlib ones refused, the claim read from the theorem name, the check file; refused before any compile (sorry, no theorem, import); messages described for the LLM; with Lean: a true cut proved, the same proof refused for a false cut, a leaked false hypothesis caught by the clean file, a safe claim, the hand-written seed proof |
| [test_proving_prover.py](tests/test_proving_prover.py) | placeholders, every template fills; statement → judge → proof with the compiler's errors and the judge's reason sent back, the frozen statement in the proof conversation; no statement after the rounds; a statement with a theorem or a refused word; only recent history kept; `sorry` compiled but never accepted; the LLM's INVALID answer; a given statement that does not compile |
| [test_proving_agent.py](tests/test_proving_agent.py) | the agent's file and the proof read back from under the marker only, the command line (allowed tools, no bypass, budget), the last result event; a fake agent: its proof goes through `verify`, no proof or a wrong one, killed at its time limit |
| [test_prove_cut.py](tests/test_prove_cut.py) | the cut named by the arguments (seed, a run's best, a file), data assumptions checked on the sets found, the message without Lean; with Lean and `FakeLLM`: a cut stated, a wrong proof refused with its error sent back, then proved, and the files written |
| [test_llm_claude_code.py](tests/test_llm_claude_code.py) | a conversation as one prompt; the command (no tool, system prompt in a file), the cost added up; a failed call or an error answer is an `LLMError` |
| [test_prompt_bench.py](tests/test_prompt_bench.py) | end to end with `FakeLLM` on tiny sets (two variants, format error, copy of a parent, scores, results.jsonl, report with acceptance and paired comparison), the same requests (LLM seed, parents) in every variant, parent selection (seed + best distinct), outcome classes, code normalization, Wilson interval |
| [test_run_files.py](tests/test_run_files.py) | the run name of a population file (new and old layout) |
| [test_log_archive.py](tests/test_log_archive.py) | a log folder becomes one archive with the same content, a resumed run gets `_2`, nothing to pack |
| [test_status.py](tests/test_status.py) | the status file is created "running", updates keep the other fields, no temporary file left |
| [test_transforms.py](tests/test_transforms.py) | `with_cut` runs the code with `m, gp, GRB, n, c` and keeps the optimum for a valid cut, code errors raise, `relaxed` drops integrality and keeps names, `fixed` to the optimum is feasible and to a bad point infeasible, rounding, partials compose |
| [test_evaluator.py](tests/test_evaluator.py) | valid and useful cut gets a fitness, the fitness follows the gap formula, removing an optimal tour → invalid, implied constraint → useless, lower bound above a known solution → invalid, broken code → rejected with its error, validation set without solutions or solved evaluation instance refused, a cut hitting the time limit rejected as too slow, `check` runs only the validity and usefulness checks; gap_work: work ratios pooled with the gap ratios, optimality instances required and solved by the base, a bound above their optimum refused |
| [test_main.py](tests/test_main.py) | a small run on tiny sets with `FakeLLM` (seed + initializer, 1 generation, saved after each generation), resuming continues without a new first generation, missing sets reported, `--model_name` required, sets from another CPU or thread count refused, the status file follows the run (finished, generation, best, the run's own Gurobi logs packed into `logs/<run>.tar.gz`; a resumed run adds `_2`), an interrupted run is marked "stopped", run name with the host / the resumed file's name, end-of-run review saved (`<run>.review.json`) and the same review by `review_run.py` on the saved population, everything in `runs/<run>/` with a `summary.md`, resuming from the run folder keeps its name, a review with only generation 0, a run end to end on each other problem, CWLP, JSSP, MCND, IMO6, GISP and corridor (its seed cut, its prompts, an accepted cut, reviewed on its own test set), the review runs on the held-out `test.pkl` (no instance in common with the evaluation set), a run without `test.pkl` fails before any LLM call (and runs without a review); a run on the hard level (its own sets, the LLM told the sizes, `review_run.py` refuses another level); a gap_work run (optimality sets required, work change in the review, `review_run.py` refuses another fitness mode) |
| [test_solver.py](tests/test_solver.py) | results stored per formulation, a valid cut keeps the optimum, `SET` and `RANDOM` seed modes, every run kept, log file named with the seed, missing `log_dir` error, infeasible model, unbounded model, isolated and in-process give the same optimum, broken formulation → `ERROR` with its message, `TOO_LARGE`, hanging build → `KILLED`, dead child → `CRASHED`, build time not in `runtime`, lambdas refused when isolated, each solve follows its own `verbose`, `keep_solution` stores the tour (both modes) and is off by default, work recorded and reproducible, `work_limit` stops the solve, machine / CPU / threads recorded (also for failed solves), extra Gurobi `params` applied (both modes), trajectory recorded only when asked, in time order, bound rising, incumbent falling to the final objective |

Tests that need a folder create it in pytest's temporary directory (`tmp_path`), so they
never write to `data/` or `logs/`.

`computer_connection/test_fleet.py` (git-ignored, like `fleet.py`) checks the fleet tool without
any machine: `python3 -m pytest computer_connection/test_fleet.py`. It covers, among others, the
probe's list of problems with ready sets (also its bash line, on a fake `data/` folder), the
problem of a run's arguments, the refusal of a machine without the sets of the run's problem, and
the night plan (window past midnight, alternating problems, free machines, resume first, nothing
outside the window, dry run), `morning` doing nothing on a day of `LAUNCH_ALL_DAY` or a weekend day, the night job working in
the daytime on weekends, a benchmark entry without a machine tried again at the next tick (the old
state format read), and the sets of
the hard level and of the gap_work fitness (`tsp/hard`, `tsp/optimality` in the probe, `prepare
--level hard` / `--optimality`, refusal of a run without them).

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
- [x] Fitness averaged over 3 Gurobi seeds during the run (Step 19)
- [x] Second fitness for the GA: LP relaxation, % of the root gap closed (`--fitness_mode lp`, Step 20)
- [x] End-of-run review on held-out instances (`test.pkl`, 50–100 cities, seeds 0–4, Step 22)
- [x] Both metrics (gap under seeds 0–4, LP gap closed) in the end-of-run review (Step 23)
- [x] Report over all runs: best of all and mean of each computer's best, per LLM (`runs/report.md`, Step 24)
- [x] Problem-independent code and prompts: `problems.py`, `prompts/problems/<name>.yaml`, minimization convention (Step 25)
- [x] Second problem: CWLP (loader, model, seed cut, prompt file, sizes measured; Step 26)
- [x] `fleet.py` per problem (`prepare --problem`, data checked for the run's `--problem_name`)
- [ ] CWLP instance sets on the lab machines (`fleet.py prepare <groups> --problem cwlp`)
- [x] Third problem: JSSP (Step 27)
- [x] LLMs through OpenAI-compatible APIs: Groq, Gemini, OpenRouter (`--llm_backend`, Step 28)
- [x] Automatic nights: plan, resume of cut runs, fetch + report every 30 min (launchd, Step 29)
- [x] Fourth problem: MCND (model as the paper's Table 20, generator shaped like CommaLab R, public test set, sizes, seed cut, validity instances, prompts; Step 37)
- [ ] MCND on the lab: `fleet.py prepare <group> --problem mcnd` (night), the paper's cut `cuts/mcnd/paper_sage.py`, first runs
- [x] The night job builds the sets a planned run needs, and plan entries can be pinned to groups
- [x] A run reviewed on another level's test set (`review_run.py --review_level`, Step 36)
- [x] Fifth problem: IMO6 (model as the paper's Table 20, the paper's cut transcribed, prompts that do not name the olympiad; Step 38)
- [ ] IMO6 on the lab: `fleet.py prepare <group> --problem imo6` (night), benchmark, first runs
- [ ] MCLP (in progress): loader and model done; the reference's sizes are almost all solved at the root (`learnt_results.md` 8.1): sizes and seed cut to choose
- [ ] Next problems: MCLP (maximization: negated objective), 2DBP, PDPTW, SHUC
- [x] Less-known problem GISP (Distributional MIPLIB generator and "hard" test set; Step 39)
- [x] Less-known problem wildlife corridor design (the corridor generator, generated benchmark; Step 40)
- [ ] Corridor on the lab: `fleet.py prepare <group> --problem corridor` (night or weekend), base model on the benchmark, first runs
- [ ] GISP on the lab: `fleet.py prepare <group> --problem gisp` (night), base model on the benchmark, first runs
- [x] Proofs in Lean 4: `prove_cut.py`, the Lean models of TSP, CWLP and JSSP, the loop and agent harnesses (Step 41)
- [ ] Proofs: the best cuts of the runs and the paper's cuts; Lean models of MCND, IMO6, GISP, corridor; proofs joined to the end of a run
- [ ] Less-known problems (few or no published cuts, to tell recall from design): WF_OTS from Distributional MIPLIB (SRPN, MMCN: anonymous files, no generator), disguised MILP-Evolve classes, post-cutoff olympiad problems (TODO P3.11)
- [x] The paper's public test sets (TSPLIB, OR-Library CWLP, Taillard; same sizes as its Table 20)
  and its protocol (2000 s, checkpoints, Δg): `prepare_benchmark.py`, `benchmark_cut.py` (Step 30)
- [x] The paper's own selected cuts as cut files (`cuts/<problem>/paper_sage.py`) and several
  cuts per benchmark run (Step 30)
- [x] Benchmark runs on the fleet: `fleet.py benchmark` (one `--shard` per machine)
- [x] Read the benchmark tables (complete on 2026-11-12: base, seed, paper_sage and our best cut,
  for each problem) and compare with the paper's Table 2 (`runs/analysis/benchmark_tables_20261112.md`)
- [x] The Muse TSP cut on the benchmark (2000 s, 13 Nov: Δg 39.3, `learnt_results.md` 5.19)
- [ ] The Muse CWLP and JSSP cuts on the benchmark (night plan `benchmarks`), and 3 Muse runs per problem
- [x] Harder evaluation and test sets, opt-in (`--instance_level hard`, Step 32)
- [ ] Hard TSP sets on the lab and the Muse and qwen3-coder runs on them (night plan, pinned to rtx3090)
- [x] The source of every ± written in the reports, between-LLM tables (Step 33)
- [ ] The std over the fitness seeds stored with each individual's fitness
- [x] Re-check of every past run's best cut with the validity instances (`check_run_cuts.py`, Step 34)
- [ ] Review the best valid cut of the 4 runs whose best cut is invalid (needs a review of a chosen individual)
- [x] Time-to-optimality term in the fitness (`--fitness_mode gap_work`, Step 35)
- [ ] Optimality sets on the lab (`fleet.py prepare <group> --problem <p> --optimality`) and a first gap_work run
- [x] The bound-above-a-known-solution test as a layer of `check_run_cuts.py` (`--public_seconds`, Step 34)
- [ ] The same test inside the end-of-run review, and on every reviewed run
- [x] Prompts tell the LLM which fitness is used, with its formula (`prompts/fitness.yaml`, Step 21)
- [x] Runs on every lab machine from the Mac: `fleet.py` (start / stop / resume / status) and its
  web dashboard, per-run status file, one instance-set folder per CPU group (Step 13)
- [ ] Optional: stop the runs automatically before class hours (a `--until` time, or a cron job
  on the Mac calling `fleet.py stop all`)
- [ ] First full runs (8 individuals, a few generations), then look at the success rate
  of the LLM and the fitness over the generations

---

### Timeline
What was done when (details in the Progress steps above, results in `learnt_results.md`).
- **21 Oct – 11 Nov** (in this order; the exact days are not recorded): Steps 1–24 (TSP model,
  solver, GA, Ollama client, work-based fitness, several fitness seeds, end-of-run review on
  held-out instances, LP fitness mode, runs on every lab machine with `fleet.py`, run report);
  problem-independent code, CWLP and JSSP (Steps 25–27); API LLMs (Step 28, from 2 Nov); automatic
  nights (Step 29); one named-user Gurobi license per machine, no launch 08:00–18:00; the paper's
  benchmark protocol on one CPU type, with work units, the LP column and the paper's cuts as cut
  files (Step 30); validity check with tight and public instances, LP check at full size, retry
  message naming the violated row (Step 31); logs packed into one archive per run (the lab's
  300k-file quota); Muse and Gemini backends with overload waits and a daily-quota stop.
- **12 Nov:** `morning` respects `LAUNCH_ALL_DAY`; `--instance_level hard` (Step 32); the source of
  every ± in the reports and between-LLM tables (Step 33); `check_run_cuts.py`, invalid runs left
  out of the report (Step 34); `--fitness_mode gap_work` (Step 35); benchmark tables complete.
- **13 Nov:** the night job builds the sets a planned run needs, plan entries pinned to groups
  (Step 29); the public-bound layer in `check_run_cuts.py` (Step 34); `review_run.py
  --review_level` (Step 36); the MCND (Step 37); the Muse TSP cut on the benchmark; IMO6 (Step 38);
  GISP and wildlife corridor design, the first less-known problems (Steps 39, 40); the night job retries a benchmark that found no
  machine, and launching is allowed at any hour on weekends (Step 29).
- **14 Nov:** the proof pipeline in Lean 4 (Step 41).

## Code conventions
- Type hints everywhere: parameters, return types (`-> None` included), class attributes
  and non-obvious local variables.
- Use Python 3.9-compatible `typing` generics: `List`, `Dict`, `Optional`, `Tuple`
  (not `list[...]` or `X | None`).
- Give recurring structures a type alias (`Instance`, `Coords`, `Costs`, ...).
