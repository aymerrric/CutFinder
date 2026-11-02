# My_evo_cut

Step-by-step reproduction of EvoCut (reference code: ../repos/EvoCut). The user wants to understand every step: go slowly, one small piece at a time.

## Architecture proposals

- When the user proposes an architecture or design idea that you think would not scale well or could be improved, say so before implementing it: explain the problem and propose your alternative, then let the user choose.

## Code conventions

- Use type hints everywhere: every function/method parameter and return type (`-> None` included), class attributes, and non-obvious local variables.
- Use `typing` generics compatible with Python 3.9 (`List`, `Dict`, `Optional`, `Tuple`), not `list[...]` or `X | None`.
- Define a type alias for recurring structures (e.g. `Instance = Dict[str, Any]`).
- Scripts are run from inside `src/` (imports look like `from utils import parameters`).

## README

- Keep `README.md` up to date so anyone can follow the project: after each new piece of code, update the "Project structure" tree, add/extend the matching entry in "Progress" (what it does + a short usage example), and tick/adjust "Next steps".

## Learnt results

- After every run, benchmark or review, add what it teaches to `learnt_results.md`. Each entry must be supported by recorded data: name the files (`runs/...`) and quote the numbers. No claim without data; say so when a fact is not stored in a file.

## Tests

- `tests/` (pytest, run `pytest` from the project root; `pytest.ini` adds `src/` to the path) holds one `test_<module>.py` per module. With every new or changed piece of code, add/update its tests in the same step and run them.
