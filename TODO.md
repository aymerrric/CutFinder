# TODO

What is left to do, in priority order. What is done is described in README.md ("Progress").

## Where we are (2026-10-21)
- Two full TSP runs finished (20 generations, `qwen2.5-coder:14b` on the A4000 machines): best
  fitness **19.14** (finlande) and **14.95** (allemagne), against 11.90 for the seed. Both
  stopped improving after generations 5–10, and generation 0 had 4 copies out of 8.
- allemagne's best cut is mostly a **copy of constraints already in the model** (the MTZ
  constraints) plus a weaker 2-cycle cut, yet it scores +26%. That points to **noise in the
  fitness** (performance variability), so checking the fitness comes first.
- **Review of both runs under Gurobi seeds 0–4 (2026-10-21):**
  - the seed cut's fitness ranges from 2.6 to 14.6 depending on the seed (std 4.7): a single
    seed's fitness is **very noisy**;
  - allemagne's best cut: 10.0 ± 4.5, the same as the seed cut (10.0 ± 4.7). Its +26% was luck;
  - finlande's best cut: 14.4 ± 4.9, and it **beats the seed cut on 5/5 seeds** (mean gap 6.4%
    vs 9.4%, base 11.7%): a real improvement, but smaller than its run fitness (19.1), because
    selecting the best of many noisy scores overestimates it.
  → Score cuts on several seeds during the run (or more instances), and compare mean gaps.
- Every LLM proposal is now recorded (`runs/<run>/llm/gen_<k>/`, with a `summary.md` per run), and broken cuts are rejected
  after at most 70 s instead of 120 s (60 s solve + 10 s build; 20 s was too tight).

## Priorities

**P1. Before any new long run**
0. **Measure whether a cut tightens the relaxation** (`learnt_results.md` 1.7): the 6 best cuts
   of 2026-10-27 all have the same LP relaxation value as the seed cut, yet score 13–15 vs 10.
   Add the root LP bound gain to the evaluation, review and summary; test winners on held-out,
   larger instances.
1. **Is the fitness sound?** Every run now ends with a review of its best cut under 5 seeds
   (README Step 15), and `review_run.py` reviews old runs. Still to do: the known cuts as
   reference points (section below), and deciding from the reviews whether the evaluation must
   change (more instances). Several seeds during the run: done (README Step 19, `--fitness_seeds`).
2. **License error → wait and retry**, never reject the cut (section "Compute and license").
3. **Analysis script** over the recorded proposals (section "Data and analysis").
4. **Relax the validity check**, otherwise symmetry-breaking cuts, often the best ones for the
   TSP, are always rejected (section below: option 1 quickly, option 4 for the long term).

**P2. A first real result on the TSP**
5. Two new runs with what P1 taught us, with the stronger LLM (`qwen3-coder:30b` on a 24 GB
   machine) and a larger `--max_llm_attempts` (5–10: a rejection costs ~7 s of LLM + ~0.4 s of
   checks).
6. **Held-out test** of the best cuts: new and larger instances, with a time limit, as in the
   reference (`evaluate_cut.py`: 300 s, seed 42), but with several seeds. Question: does
   EvoCut rediscover known cuts (e.g. lifted MTZ), or find better ones?
7. Compare the LLMs on the same settings (`qwen3-coder:30b`, `devstral:24b`,
   `qwen2.5-coder:14b`; later reasoning models such as `qwq:32b` or `deepseek-r1:32b`, whose
   `<think>` block must be removed in `OllamaClient.clean`).

**P3. Scale up**
8. **Prompt experiments**: first benchmark done on 2026-10-24 (`runs/prompt_bench/prompts_20261024/
   report.md`; 1,650 answers of `qwen3-coder:30b`, 7 variants × 30 rounds):
   - No variant is significantly better than `default` at producing new accepted cuts. `rules`
     is the best (55% vs 48%, p = 0.09). `catalog` (25%, many too-slow cuts) and `reasoning`
     (39%, many copies of a parent) are significantly worse.
   - (2026-10-27: `--max_init_attempts 15` and duplicate refusal for generation 0, README Step 18.)
   - **The initializer is the weak point**: 1–5 accepted out of 30 for every variant (`concise`
     best, 17%), which explains the copies in generation 0 of the runs. Next: rework the
     initializer prompt (and benchmark only `--tasks init`).
   - **Invalid answers are the main refusal** (17–34%): relaxing the validity check (P1) matters.
   - `focused_ops` gets the most accepted answers, but 17% of them are copies of a parent:
     refuse such copies in the GA (Algorithm ideas: deduplicate).
   - Next: a full run with `rules` vs `default`, reviewed under several seeds. Richer feedback on
     rejection is still to do (section below).
9. **More compute**: the license allows 2 runs at a time. The options and the email to send are
   in `improve_license.md` (campus server, one named-user license per machine, ...).
10. **A second problem**: CWLP first (section "Other problems").

**P4. Later**
11. Algorithm ideas (islands, deduplication, ...).
12. Beyond cuts: reformulations, solver guidance, callbacks.
13. Sandbox the LLM code.
14. Housekeeping.

---

## Is the fitness sound? (P1)
A MIP's search path changes with any change to the model: even a harmless constraint moves the
branching, and at a fixed work limit the final gap can change a lot by luck (*performance
variability*). Our fitness is deterministic, but that does not make it meaningful.
- **Noise test:** partly done: the end-of-run review (seed cut vs best cut vs base, seeds 0–4).
  Still worth adding: the seed plus a redundant constraint (e.g. `x[1,2] <= 1`), to measure
  the noise caused by a change that adds nothing.
- **Known cuts as reference points:** 2-cycle, Desrochers–Laporte lifted MTZ, tight bounds on
  `u`, 3-cycle elimination, `sum u_i = n(n+1)/2`. They tell what a good fitness is and whether
  the score ranks them in the expected order.
- If the noise is large: average over 2–3 seeds, use more evaluation instances (16 instead of
  8), or a longer work limit. All of these cost more solves, which ties back to the license.
- Record the gap of each evaluation instance (not only the mean), to see whether one instance
  dominates the score.
- The validity check stays strict about correctness. Variability only affects the score.

## Relax the validity check (P1)
Today (copied from the reference, `CutVerifier.check_feasibility_with_fixed_vars`): on each
validation instance, the model with the cut, with its variables fixed to **the** stored optimal
solution, must stay feasible. A valid cut only needs to keep **at least one** optimal solution.
In the TSP every tour has a reversed twin of the same cost, so a cut that removes one direction
of the stored tour is rejected, although the reversed tour is still optimal. This rejects every
**symmetry-breaking cut** (e.g. "leave city 1 towards the smaller of its two neighbours").

Options, from the simplest:
1. **Optimal value preserved:** solve each validation instance with the cut to optimality and
   compare the optimal value (15-city instances: ~0.1 s each). Works for every problem.
2. **Tour or reverse:** accept if the stored tour or its reverse stays feasible. TSP-specific.
3. **All optimal solutions:** collect them with the solution pool (`PoolSearchMode=2`), and
   require at least one to stay feasible.
4. **The LLM provides the mapping** (a certificate checked by us): with a cut that removes some
   optima, or a reformulation that changes the variables, the LLM also writes
   `map_solution(solution, instance) -> solution`. It converts the stored optimum into the
   solution the cut keeps (reverse or rotate the tour, compute new variables from the tour). On
   each validation instance we check: (a) the mapped solution, projected on the base variables,
   is feasible for the base model **with the same objective value**, so it is still optimal;
   (b) the model with the cut, fixed to the mapped solution, is feasible. Together they prove
   the cut keeps an optimum. A wrong mapping is simply rejected. It costs no more than today:
   two fixed-variable checks. The feedback can be precise ("your mapping gives a tour of cost
   3.41, the optimum is 3.32"). No mapping = identity = today's check. The mapping runs in the
   same isolated child as the cut. It also covers the reformulations of "Beyond cuts".

Option 1 is the quick fix, option 4 the general one. Keep the reference check behind a flag, to
compare. Also: only 2 validation instances, so an invalid cut can slip through. More 15-city
instances are cheap.

## Data and analysis (P1)
Recorded since 2026-10-21: every proposal with its conversation, reply, code, outcome, reason,
fitness and timings (`runs/<run>/llm/gen_<k>/child_<c>_attempt_<a>.json`), and `summary.md` already gives the
acceptance rate per operator and the main rejection reasons. Still to do:
- `analyze_runs.py`:
  - best / mean fitness per generation;
  - **acceptance rate per operator and per variant**;
  - the most frequent rejection reasons (format, broken code, invalid, useless, too slow);
  - cost per accepted cut (LLM + solver seconds);
  - duplicate ideas;
  - the family tree of the best cut.
- Add to each record: LLM tokens in/out (Ollama returns them), model and temperature, and the
  per-instance gaps of the scored cuts.

## Compute and license (P1 for the retry, P3 for more compute)
- **A solve refused by the license rejects the cut** today (`CutRejected("ERROR ...: Too many
  sessions")`, also shown to the LLM). It should wait and retry, or stop the run.
- **Gurobi WLS license: 2 concurrent sessions** (checked 2026-10-19), shared with the Mac
  (tests included): `fleet.py` allows 2 runs at once (`--max_runs`), and don't run the tests
  while 2 runs are going. To use the 37 machines:
  1. one free **academic named-user license per machine** (`grbgetkey` on each machine from
     the campus network, file in its `/Data/aymeric/gurobi.lic`, `GRB_LICENSE_FILE` set per
     host by `fleet.py`). Check the academic terms first (how many keys per person);
  2. ask Gurobi or the lab for a WLS license with more sessions;
  3. last resort: a **license-free solver** (HiGHS, SCIP). The LLM writes gurobipy code, so this
     needs a modelling layer and a new work measure (SCIP: LP iterations or nodes). It's a big
     change.
- Solve the 8 evaluation instances in parallel (one process each, fewer threads each): ~53 s
  per cut today. But each process is one more license session.

## Prompt experiments (P3)
A full GA per prompt variant takes hours, and the result is noisy. Cheaper:
- **Offline benchmark:** fix a few parents (the seed and 2–3 accepted cuts), ask each prompt
  variant N times (e.g. 20), and run only the validity and usefulness checks (15-city
  instances, ~0.4 s). Measure acceptance rate, diversity (distinct ideas), and the fitness of a
  sample. The LLM is the bottleneck, not Gurobi, so this works even with 2 license sessions.
- Variables:
  - with or without the model code;
  - few-shot examples of accepted / rejected cuts;
  - stating the known strong cuts or not (keep a variant without them, to test rediscovery
    honestly);
  - temperature;
  - IDEA-before-CODE vs a short reasoning step;
  - the length of the feedback.
- **Richer feedback on rejection:** for an invalid cut, show which optimal tour it cuts off
  and which constraint is violated. For a useless cut, show that the LP optimum still satisfies
  it. Probably the cheapest way to raise the acceptance rate.
- Then confirm the best 1–2 variants with full runs.

## Other problems (P3)
The reference has 6: TSP, CWLP (capacitated warehouse location), MCND (multi-commodity network
design), JSSP (job shop), MCLP (maximum covering location), 2DBP (2D bin packing), with data
scripts in `../repos/EvoCut/data/`. The code is ready (`PROBLEMS` in `main.py`: loader +
`create_model` + baseline cut). Order:
1. **CWLP**: classic, strong known cuts (`x_ij <= y_i`), so it tests rediscovery like the TSP,
   with a different structure (capacities, binaries + continuous variables).
2. **JSSP**: big-M scheduling, weak LP bound: where cuts matter most and are least known.
3. MCND for a large network model.

Each new problem needs a loader, its model, a baseline cut, validation / evaluation sets, and a
check that `WORK_LIMIT` leaves a gap on the evaluation instances. If the base model solves some
evaluation instances, a mixed score (gap if unsolved, work if solved) may be needed.

## Algorithm ideas (P4)
- **Islands:** one GA per machine, migrating the best cut every few generations through the
  shared home (`runs/`). Needs more than 2 license sessions.
- **Deduplicate:** refuse a child whose code (normalized) or idea is already in the population.
  Seen in the first prompt benchmark: the `Lifted` mutation sometimes returns its parent's code
  unchanged, and it is accepted (valid and useful, since the parent is). In a run it would be
  a wasted slot. Refusing children identical to a parent (`prompt_bench.normalize`) is cheap.
- **Memory of failures:** show the LLM the ideas already rejected in this run.
- **Combine the final cuts:** the best k cuts together may beat the best one alone.
- **Adaptive operators:** choose variants in proportion to their recent acceptance rate (a
  bandit) instead of uniformly.

## Beyond cuts: other ways to speed up the solve (P4)
The LLM code already receives the full model `m`, so it can technically add variables, remove
constraints or set attributes. What blocks these ideas is the evaluator, which assumes "the same
model plus constraints". On the TSP the LLM would probably rediscover the textbook answers (DFJ
with lazy subtour separation, flow formulations). The real interest is on problems without a
known best formulation (JSSP).

**1. Change the model, same optimum**
- Tighten coefficients / big-M (MTZ: tight bounds on `u`, lifted MTZ).
- Disaggregate constraints (CWLP: `x_ij <= y_i` instead of `sum_j x_ij <= n * y_i`).
- Extended formulations with new variables (single- or multi-commodity flow instead of MTZ).
- Auxiliary integer variables to branch on (CWLP: `k = sum_i y_i` integer).
- Remove redundant constraints or variables.
- Exact variable fixing: heuristic tour (upper bound) + LP reduced costs → fix to 0 the arcs
  that cannot be in an optimal tour. (Not "k nearest neighbours only": that is a heuristic.)
- Symmetry breaking (needs the relaxed validity check).

**2. Guide the solver, same problem**
- Starting solution from an LLM-written heuristic (`Start`), hints (`VarHintVal`,
  `VarHintPri`), branching priorities (`BranchPriority`), `Partition`.
- Gurobi parameters (`MIPFocus`, `Cuts`, `Presolve`, `Symmetry`, ...). Baseline to beat:
  Gurobi's own tuning tool.

**3. Callbacks (code run during the solve)**
- Lazy constraints (`cbLazy`) for exponential families (TSP subtours by min-cut = DFJ).
- User cuts (`cbCut`) on fractional solutions (combs, capacity cuts).
- Heuristics (`cbSetSolution`), e.g. 2-opt repair of LP solutions.

**What the evaluator needs**
1. Validity based on the optimal value, or on a mapping provided by the LLM (option 4 above),
   with the solution projected on the original variables to check it is a real tour.
2. Usefulness per category: cuts and reformulations must cut off the base LP optimum (fixed on
   the original variables only). Starts, hints, priorities, parameters and callbacks: fitness
   only.
3. **Anti-cheating:**
   - Python time is not counted in work: code that solves the instance in Python (or with
     its own Gurobi model) and passes the optimum as a start gets gap 0. Cap the build time,
     and forbid creating models.
   - Reset the protected parameters after the code runs: `Threads`, `WorkLimit`,
     `TimeLimit`, `Seed`, `MIPGap`.
   - An objective offset (`ObjCon`) makes the relative gap look smaller: compute the gap in
     the base objective's units.
   - Cap the Python time inside callbacks.
4. Prompts: one operator family per category, each with its own validity rules.

Order: (1) validity by optimal value or mapping, (2) starts and parameters with the guards,
(3) callbacks.

## Safety (P4)
The LLM code runs in a child process (time limits, size cap) but is not sandboxed: it could read
or write files. Restrict `exec` (no `__import__`, limited builtins) or run it as another user.

## Housekeeping (P4)
- `requirements.txt` with the pinned versions (Python 3.11, gurobipy 13.0.3, PyYAML 6.0.3,
  pytest 9.1.1).
- `check_same_machine` compares the CPU name, but work is identical on the four lab CPU types:
  it could compare only the thread count and the Gurobi version.
- Cache solver results on disk (`cache/`).
- Clean old Gurobi logs in `logs/<run>/` (one file per solve).
