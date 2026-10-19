# TODO

Ideas and known limits to come back to. The build plan itself is in README.md → "Next steps".

## Cut evaluation

### Relax the validity check (symmetric optima)
The current check copies the reference (`CutVerifier.check_feasibility_with_fixed_vars`): on
each validation instance, the model with the cut, with its variables fixed to **the** optimal
solution found by the base model, must stay feasible.

This is stricter than what a valid cut needs, which is to keep **at least one** optimal solution.
In the (symmetric) TSP, every tour has a reversed twin with the same cost, so a cut that removes
only one direction of the stored tour is rejected although the reversed tour is still optimal.
This rejects **symmetry-breaking cuts**, which are valid in practice and often help a lot (e.g.
"leave city 1 towards the smaller of its two neighbours"). The tests ran into it: forbidding one
arc of the optimal tour does not change the optimal value.

Possible relaxations, from the simplest:
1. **Optimal value preserved:** solve each validation instance *with the cut* to optimality and
   compare the optimal value with the base one (they are 15-city instances: about 0.1 s each).
   Accepts any cut that keeps some optimal solution, symmetry-breaking included. Works for every
   problem. (The evaluation set already rejects cuts whose bound goes above a known solution.)
2. **Tour or reverse:** accept if the stored tour *or its reverse* stays feasible. TSP-specific.
3. **All optimal solutions:** collect every optimal solution of the validation instances
   (Gurobi solution pool, `PoolSearchMode=2`) and require at least one to stay feasible.

Option 1 is the natural candidate. Keep the reference check available behind a flag, to compare.

### Other evaluation points
- If an evaluation instance gets solved by the base model, it is replaced (`MIN_BASE_GAP`). With
  another problem or another `WORK_LIMIT`, a mixed score (gap if unsolved, work if solved) may
  be needed.
- Gurobi `Work` depends on the CPU type: base results and cut results must come from the same
  machine type and `THREADS`. Check whether all the 3090 machines have the same CPU before
  spreading the evaluations over several machines.
- One evaluation takes ~53 s (8 instances × ~6.5 s at `WORK_LIMIT = 10`): the evaluation
  instances could be solved in parallel (one process each, fewer threads each).
- A cut whose code never finishes building (e.g. enumerating all subsets on 97 cities) is only
  killed after `TIME_LIMIT + BUILD_TIMEOUT` = 60 + 60 s (on the first evaluation instance: the
  first failure raises `CutRejected`). That is 2 minutes lost per such cut, while a normal solve
  needs ~7 s at `WORK_LIMIT = 10`: give the evaluation solver tighter limits.
- Only 2 validation instances: an invalid cut can slip through. More (cheap, 15 cities)
  validation instances would catch more.

## Safety
- The LLM code runs in a child process (timeout, size cap) but is not sandboxed: it could read or
  write files. Restrict `exec` (no `__import__`, limited builtins) or run it as another user.

## Infrastructure
- Check the Gurobi WLS license limits (how many machines / sessions at once) before running on
  several lab machines.
- `requirements.txt` with the pinned versions (Python 3.11, gurobipy 13.0.3, PyYAML 6.0.3,
  pytest 9.1.1).
- Cache solver results on disk (`cache/`).
- Clean old log files in `logs/` (one file per solve: thousands per run).

## Experiments
- Compare local LLMs (Ollama: change `--model_name`), e.g. `qwen3-coder:30b`, `qwen3:32b`,
  `qwq:32b`, `deepseek-r1:32b` (the reasoning models need the `<think>` block removed in
  `OllamaClient.clean`).
