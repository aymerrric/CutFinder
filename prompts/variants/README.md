# Prompt variants

Each folder overrides part of `prompts/` (see `Prompts` in `src/evolution/prompts.py`):
`common.yaml`, `initializer.yaml` and `fitness.yaml` key by key, `mutations.yaml` and `crossovers.yaml` as whole
files. Compare them with `src/prompt_bench.py`, and run with one with
`main.py --prompt_variant <name>`.

| variant | hypothesis tested | what changes |
|---|---|---|
| `default` | reference | `prompts/` as is |
| `rules` | answers fail because the LLM does not know how they are checked | system: explains the validity check (one stored optimal solution, so no symmetry breaking), the usefulness check (must cut off the LP optimum, so no constraint already implied, no copy of the model's constraints), and the cost limits |
| `catalog` | the LLM lacks ideas, so it repeats or copies | system: adds a list of general strengthening techniques (bound/big-M tightening, lifting, small-subset inequalities, valid equalities from aggregation, ...) |
| `reasoning` | asking for a justification makes answers more often valid | response format: the IDEA must state the family, a short validity argument, and which fractional LP point the cut removes |
| `concise` | a shorter prompt suits a small model better | system: about half the length, same rules; `fitness.yaml`: one short bullet per fitness mode |
| `few_shot` | examples of refused and accepted answers teach the checks faster than rules | system: three refused answers with their reason, one accepted answer |
| `focused_ops` | sharper operator texts give more useful children | mutations and crossovers replaced by concrete operators (Tighten, Extend, Prune, Explore; Merge, Graft) |

A variant that replaces `system` must keep the placeholders `{task}`, `{model_code}` and
`{scoring}` (the fitness text of `fitness.yaml` for the run's `--fitness_mode`), and say nothing
about a specific problem: use the problem's placeholders (`{problem}`, `{description}`,
`{solution}`, `{evaluation_size}`, from `prompts/problems/<problem>.yaml` and `src/problems.py`).
Problem-specific text of a variant goes in `variants/<variant>/problems/<problem>.yaml` (e.g.
`few_shot`'s examples, `{examples}`).
