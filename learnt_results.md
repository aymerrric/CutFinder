# What we learnt from the runs

Every entry is a finding supported by **recorded data**: the files are named, with the numbers
that support the claim, so anyone can check it. Each entry also says what it changed (or should
change). Newest experiments are added at the end of each section, and the open experiments are
listed at the bottom.

Data referenced below:
- `R-all` = `runs/tsp_qwen2.5-coder-14b_allemagne_20261020_180714/` (20 generations,
  `qwen2.5-coder:14b`, A4000, prompts `default`, 3 LLM attempts per child);
- `R-fin` = `runs/tsp_qwen2.5-coder-14b_finlande_20261020_180716/` (same settings);
- `B-1024` = `runs/prompt_bench/prompts_20261024/` (prompt benchmark: 1,650 answers of
  `qwen3-coder:30b`, 7 prompt variants × 30 rounds of the same requests, 14 A5000 machines);
- `P-*` = `runs/launches/prepare_<group>_20261019_*.out` (building the instance sets on each CPU type);
- `R-1027` = the runs `runs/tsp_qwen3-coder-30b_<host>_20261027_*/` (prompt variant `rules` or
  `default` in `status.json` → `arguments.prompt_variant`; `qwen3-coder:30b`, 15 LLM attempts per
  slot of generation 0). All were killed at 11:07, during generation 1 (4.2), so only their
  generation 0 is complete (13 runs). Six of them were resumed or started at 21:09 and finished
  (3 `rules`: albatros, faisan, jabiru; 3 `default`: brochet, epervier, saumon). Comparison:
  `runs/analysis/rules_vs_default_20261028.txt` (script `runs/analysis/rules_vs_default.py`).
- `LP-1028` = `runs/analysis/lp_bounds_20261028.txt` (script `runs/analysis/lp_bounds.py`): LP
  relaxation value with each best cut of `R-1027`, on random instances of 60, 80 and 100 cities.

---

## 1. Measuring a cut

**1.1 Gurobi work is identical on the four lab CPU types; time is not.**
- *Evidence:* `P-a4000`, `P-a5000`, `P-ada4000`, instance `rand_s100_1_74` (same seed, 8 threads):
  gap 0.135247, work 10.0073 on allemagne (i7-13700K), albatros (Xeon W-1270P) and
  ain (i7-14700K); the times are 4.55 s, 7.28 s and 4.87 s. The 8 instances of every set
  match the same way, including saumon's set (`data/rtx3090`, Xeon w3-2435): checked on
  2026-10-27, name, gap, work and node count of the 8 instances of `data/<group>/tsp/evaluation.pkl`. Both runs give the
  seed cut the same fitness, 11.899 (`R-all`, `R-fin`: `population.json`, generation 0).
- *Consequence:* the work limit makes fitness reproducible across machines; a time limit would
  not. Fitness values from different groups can be compared.

**1.2 A fitness from a single Gurobi seed is very noisy.**
- *Evidence:* `R-all/review.json` (same in `R-fin`): the seed cut solved under seeds 0–4 scores
  2.56, 11.99, 12.61, 8.42, 14.60 (mean 10.04, std 4.74), while it scored 11.90 with the run's seed.
- *Consequence:* differences of a few fitness points between two cuts mean nothing with one seed.
  Every run now ends with a review under 5 seeds (README Step 15). Next: score with several seeds
  during the run (TODO P1).

**1.3 A cut that adds nothing can score +26% by luck.**
- *Evidence:* `R-all/population.json`, best cut (fitness 14.95): the MTZ constraints copied
  again, plus a 2-cycle cut weaker than the seed's (pairs with city 1 skipped). Review
  (`R-all/review.json`): 10.01 ± 4.48, versus 10.04 ± 4.74 for the seed cut; it beats the
  seed cut on only 3/5 seeds; mean gap 9.62% vs 9.40%.
- *Consequence:* a copy or a weakening of existing constraints changes Gurobi's search path and
  can win by chance. Duplicates of the model's constraints should be refused, and winners
  confirmed by the review.

**1.4 The best cut's run fitness overestimates it (winner's curse).**
- *Evidence:* `R-fin`: run fitness 19.14, review 14.44 ± 4.86; `R-all`: 14.95 → 10.01
  (`review.json`, `run_fitness` vs `fitness_mean`).
- *Consequence:* the best of many noisy scores is biased upwards. Report the review score, not
  the run fitness.

**1.5 One genuinely better cut was found.**
- *Evidence:* `R-fin/review.json`: the best cut beats the seed cut on **5/5 seeds** (7.71,
  16.87, 19.64, 11.17, 16.82 vs 2.56, 11.99, 12.61, 8.42, 14.60), mean gap 6.39% vs 9.40% for the
  seed cut and 11.70% without a cut. Its code (`R-fin/summary.md`): 3-city subtour elimination
  (on the triples with an even index sum), the 2-cycle cut, and sum of positions
  `sum u_i >= n(n+1)/2` (an equality for every tour, unknown to the LP).
  It also **tightens the LP relaxation**, unlike the 6 best cuts of 2026-10-27 (1.7): LP value
  6.006117 / 6.511976 / 7.275415 on 60 / 80 / 100 cities, against 5.986443 / 6.497463 / 7.193714
  with the seed cut (`LP-1028`, line "best finlande 1020").
- *Consequence:* the pipeline can find real improvements, and so far this is the only cut that is
  stronger in the polyhedral sense. The arbitrary filter on the triples suggests room for a
  cleaner version (all triples, without the auxiliary variables).

**1.6 A 20 s time limit per cut solve rejects valid cuts on a loaded machine.**
- *Evidence:* `R-1027` dindon, `run.log`: `[cut1_eval] rand_s100_1_74: TIME_LIMIT, work=6.4979,
  20.01s`, so the **seed cut** was rejected as too slow and the run failed (`status.json`: "the
  baseline cut is rejected: too slow"). On albatros (same machine type), the same kind of solve
  reaches work 10 in 7.69–10.47 s (`R-1027` albatros, `run.log`, `cut33_eval`).
- *Consequence:* `CUT_TIME_LIMIT` raised from 20 to 60 s (2026-10-27): a 2x margin was too thin
  when Ollama loads a model or another user loads the CPU.

**1.7 The best cuts win without tightening the relaxation: the fitness rewards how Gurobi reacts
to the way constraints are written.**
- *Evidence:* `LP-1028`: the LP relaxation value is **identical, to 6 decimals, with the seed cut
  and with each of the 6 best cuts** of `R-1027` (5.986443 / 6.497463 / 7.193714 on 60 / 80 / 100
  cities; 5.412011 / 5.620464 / 5.695773 without a cut). Yet in their reviews (5 seeds) these cuts
  score 13.06–15.04 against 10.04 for the seed cut, and beat it on 4–5 of 5 seeds
  (`rules_vs_default_20261028.txt`). The clearest case: saumon (generation 0) and jabiru
  (generation 15) found the same cut, `sum_{k != j} (x[i,k] + x[k,i]) >= 1` for every pair, which
  with the degree equalities is exactly the seed cut `x[i,j] + x[j,i] <= 1` rewritten; both review
  at 15.04 ± 4.85 (gap 5.89% vs 9.40%). albatros and faisan win with the MTZ constraints copied
  again plus the 2-cycle cut (13.22 and 13.06, 5/5 seeds).
- *Consequence:* "better fitness" here means "Gurobi's presolve, cuts and branching behave better
  on this formulation of these instances", not "a stronger cut". This may not generalize. Add
  the **LP bound gain** (root relaxation with vs without the cut) to the evaluation and the
  summary, and test the winners on held-out, larger instances with several seeds before calling
  them improvements. 1.3 (allemagne, 10.01) shows the effect is not systematic: there the 2-cycle
  cut was also weakened.

## 2. The genetic algorithm

**2.1 Runs stop improving early.**
- *Evidence:* best fitness per generation (`population.json`): `R-fin` reaches 19.14 at
  generation 5, then nothing better for 15 generations; `R-all` reaches 14.95 at generation 7,
  then 13 generations without progress.
- *Consequence:* 20 generations of 8 individuals mostly re-explore. To test: more diversity
  (deduplication, islands), or a larger population with fewer generations.

**2.2 The first generation was half copies of the seed.**
- *Evidence:* generation 0 of `R-all` and of `R-fin`: seed 1, init 3, **copy 4** (3 LLM attempts
  per slot, `qwen2.5-coder:14b`). Benchmark `B-1024/report.md`, "Accepted answers per task":
  the initializer gets 1/30 to 5/30 accepted answers depending on the prompt variant.
- *Consequence:* since 2026-10-27, the first generation has its own budget (`--max_init_attempts`,
  default 15), and an initial cut repeating a cut of the generation is refused. Effect to
  measure on the runs of 2026-10-27.

**2.3 Many children are copies of their parent because every LLM attempt failed.**
- *Evidence:* children of generations 1–20 (elites excluded): 44 copies out of 140 in `R-all`
  (31%), 13 out of 140 in `R-fin` (9%).
- *Consequence:* a third of the slots can be wasted. Raising the acceptance rate (prompts,
  validity check, feedback) directly gives more real children.

**2.4 With 15 attempts per slot, generation 0 is mostly real cuts.**
- *Evidence:* `R-1027`, generation 0 of the 13 runs that finished it (`population.json`, and
  `llm/gen_000/*.json` for the answers): 75 initial cuts and 16 copies of the seed out of 91 slots
  (82% filled), versus 6 out of 14 slots in `R-all` + `R-fin` with 3 attempts. (The LLM also
  changed, `qwen3-coder:30b` instead of `qwen2.5-coder:14b`, so this comparison is not clean.)
  The duplicate refusal never fired: 0 duplicates among the 548 initializer answers.
- *Consequence:* keep `--max_init_attempts 15`. The cost is visible: up to 72 answers for one
  generation 0 (`autruche`, `loriol`).

**2.5 Every finished run of `R-1027` ends with a cut that beats the seed cut in review.**
- *Evidence:* `rules_vs_default_20261028.txt`: reviewed fitness of the best cuts 13.06–15.04
  (seed cut 10.04 ± 4.74), mean gaps 5.89–8.61% (seed cut 9.40%, no cut 11.70%), beating the
  seed cut on 4 or 5 of 5 seeds in all 6 runs. Run fitness 16.20–18.52, so again 2–5 points
  above the review (winner's curse, 1.4). Best found at generation 0, 1, 9, 10, 15, 16; a run of
  20 generations takes 2.4–3.9 h (`qwen3-coder:30b`, A5000 / RTX 3090).
- *Consequence:* the method improves the fitness reliably, but see 1.7: none of these cuts
  tightens the LP relaxation.

## 3. Prompts (benchmark `B-1024`, 2026-10-24)

**3.1 No prompt variant is significantly better than `default` at producing new accepted cuts.**
- *Evidence:* `B-1024/report.md`, "Ranking (new cuts accepted)": `rules` 55% (49–61%),
  `focused_ops` 51%, `concise` 50%, `default` 48% (42–55%). Paired test against `default`:
  `rules` 48 better / 32 worse, p = 0.093; `concise` p = 0.661.
- *Consequence:* `rules` is promising but not proven: tested in full runs on 2026-10-27.

**3.2 A catalogue of strengthening techniques makes things worse.**
- *Evidence:* `B-1024`: `catalog` 25% new cuts accepted (18 better / 74 worse, p < 0.001);
  27% of its answers are too slow to build or solve (2–3% for `default`, `rules`).
- *Consequence:* suggesting families like "small-subset inequalities" leads to heavy
  enumerations. Do not use; if techniques are listed, state their cost.

**3.3 Asking for a justification first makes the LLM copy more.**
- *Evidence:* `B-1024`: `reasoning` 39% new cuts accepted (34 better / 57 worse, p = 0.021);
  32 of its 125 accepted answers are copies of a parent (10 for `default`).
- *Consequence:* do not use as is.

**3.4 The initializer prompt is the weakest task for every variant.**
- *Evidence:* `B-1024`, "Accepted answers per task", init: `default` 1/30, `catalog` 1/30,
  `focused_ops` 1/30, `reasoning` 2/30, `rules` 2/30, `few_shot` 4/30, `concise` 5/30; mutations
  19–60%, crossovers 37–95%.
- *Consequence:* rework the initializer prompt (benchmark with `--tasks init`); meanwhile more
  attempts per slot (2.2).

**3.5 "Invalid" is the main reason for refusing an answer.**
- *Evidence:* `B-1024`, "Why answers were refused": invalid 17–34% of the answers depending on
  the variant, useless 9–14%, too slow 1–27%, code error 2–8%, format error 0–1%.
- *Consequence:* the strict validity check (the stored optimal tour must stay feasible, so
  symmetry-breaking cuts are refused) costs the most. Relaxing it is TODO P1.

**3.6 Answers that copy a parent pass every check.**
- *Evidence:* `B-1024`, "all accepted (copies)": `focused_ops` 130 accepted, 22 copies;
  `reasoning` 125, 32; `default` 126, 10.
- *Consequence:* the GA must refuse a child identical to a parent (done for the first
  generation on 2026-10-27; still to do for mutations and crossovers).

**3.7 In real runs, `rules` makes the initializer work twice as often as `default`.**
- *Evidence:* `R-1027` generation 0, answers of the initializer (`llm/gen_000/*.json`, outcome
  `accepted`): `rules` 33 accepted out of 153 answers (22%, 5 runs: albatros, bengali, faisan,
  harpie, jabiru), `default` 42 out of 395 (11%, 8 runs: autruche, coucou, epervier, gelinotte,
  hibou, kamiche, loriol, saumon); Fisher exact test p = 0.0013. Slots filled with a real cut:
  `rules` 33/35, `default` 42/56 (p = 0.023).
- *Consequence:* first evidence from full runs in favour of `rules` (the benchmark `B-1024`
  had too few initializer requests to show it: 2/30 vs 1/30). To confirm on the complete runs.

**3.8 In full runs, `rules` helps only the initializer: afterwards it does worse than `default`.**
- *Evidence:* `rules_vs_default_20261028.txt` (3 runs per variant): after generation 0,
  accepted children `rules` 347/687 answers (50.5%) vs `default` 371/636 (58.3%), Fisher p = 0.0048;
  copies of a parent `rules` 79/420 vs `default` 57/420, p = 0.049. Final quality: mean review of
  the best cuts 13.77 (`rules`) vs 14.13 (`default`), too few runs to compare. With 3.7 (`rules`
  doubles the initializer's acceptance, p = 0.0013). Caveat: the 3 `rules` runs all ran on A5000
  machines, the `default` ones on 1 A5000 + 2 RTX 3090 (work is identical, 1.1, but the
  machines differ).
- *Consequence:* use the `rules` system prompt for the initializer only, and `default` for
  mutations and crossovers (a mixed variant), then benchmark it.

## 4. Infrastructure

**4.1 The academic WLS license allows 2 concurrent Gurobi sessions.**
- *Evidence:* test of 2026-10-19 (10 machines opening a session at once: 3 succeeded, the others
  got "Too many sessions, 5 active sessions for a baseline of 2"; quoted in `improve_license.md`).
  *Not stored as a data file.*
- *Consequence:* named-user licenses per machine (21 machines licensed on 2026-10-23/27,
  `improve_license.md`); `fleet.py` limits the WLS runs to 2.

**4.2 Runs on the lab machines get killed during class hours.**
- *Evidence:* `R-1027`: the last line of every `run.log` is between 11:07:01 and 11:07:33 on
  2026-10-27 (a Tuesday), on 15 machines in two rooms (A5000 and RTX 3090), with no error and
  no "stopped" state (`status.json` still "running": shown as dead). The first run of the project
  (2026-10-19) also died at 17:00 without an error (files deleted since).
  Again on 2026-10-29 (a Thursday): 10 runs launched at 12:59 all stopped between 13:04:32 and
  13:05:06 (last line of `run.log` in `runs/tsp_qwen3-coder-30b_{lieu,lotte,labre,piranha,mulet}_20261029_*/`
  and in the resumed `..._{autruche,bengali,coucou,dindon,hibou}_20261027_*/`), with no reboot of the
  machines (`last reboot`: none since 2026-10-27 14:51). Three kills so far: about 11:07, 13:05
  and 17:00, on weekdays.
- *Consequence:* a run of 3–4 h cannot finish during the day on weekdays. Start long runs in the
  evening (after ~18:00) or at the weekend; every run can be resumed from its last generation
  (`fleet.py resume`).

---

## Open experiments (results to add here)

- **`rules` vs `default`, 11 full runs each** (`qwen3-coder:30b`; 4 RTX 3090 + 7 A5000 per
  variant; population 8, 20 generations, elitism 0.2, 15 attempts for the first generation,
  review under 5 seeds). Launched 2026-10-27 at 10:48, killed at 11:07 (4.2). Relaunched at 21:09:
  only 6 machines were free (another user's GPU jobs on the 16 others), 3 per variant, all
  finished (2.5, 3.8, 1.7). The 16 other runs were not run.
- **Next:** measure the LP bound gain of every cut (1.7), a held-out test of the 6 best cuts, and
  a mixed variant (`rules` initializer + `default` operators, 3.8).
