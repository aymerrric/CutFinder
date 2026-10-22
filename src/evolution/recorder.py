"""
Every LLM proposal of a run, accepted or not, as one JSON file:
    runs/<run>/gen_<generation>/child_<child>_attempt_<attempt>.json
with the conversation sent, the reply, the parsed idea and code, the outcome and its reason, the
fitness and the timings. The population file keeps only the accepted cuts; this keeps
everything, to see afterwards what the LLM proposed and why it was refused.
"""
import json
import os
from typing import Any, Dict

# outcome of an attempt
ACCEPTED: str = "accepted"
FORMAT_ERROR: str = "format_error"  # the reply does not follow IDEA: / CODE:
REJECTED: str = "rejected"  # CutRejected: broken code, invalid, useless, too slow, ...
LLM_ERROR: str = "llm_error"  # the LLM could not be reached

Attempt = Dict[str, Any]


class AttemptRecorder:
    """Writes each attempt it is called with into `folder`/gen_<k>/ (folders created as needed)."""

    def __init__(self, folder: str) -> None:
        self.folder: str = folder

    def __call__(self, attempt: Attempt) -> str:
        """Save `attempt` (needs "generation", "child", "attempt"); returns the file path."""
        directory: str = os.path.join(self.folder, f"gen_{attempt['generation']:03d}")
        os.makedirs(directory, exist_ok=True)
        stem: str = f"child_{attempt['child']:02d}_attempt_{attempt['attempt']}"
        path: str = os.path.join(directory, f"{stem}.json")
        k: int = 2
        while os.path.exists(path):  # a resumed run redoes the generation it was building
            path = os.path.join(directory, f"{stem}_resumed{k}.json")
            k += 1
        with open(path, "w") as f:
            json.dump(attempt, f, indent=1)
        return path


def load_attempt(path: str) -> Attempt:
    with open(path) as f:
        return json.load(f)
