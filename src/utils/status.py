"""
Progress file of a run, runs/<run>.status.json, rewritten at every change (child started,
generation done, run ended). It lets a remote watcher (computer_connection/fleet.py) see where
each machine is without parsing logs. Written atomically (temporary file + rename), so a reader
never sees half a file.
"""
import json
import os
import time
from typing import Any, Dict

RUNNING: str = "running"
FINISHED: str = "finished"
FAILED: str = "failed"  # an exception stopped the run
STOPPED: str = "stopped"  # SIGTERM / Ctrl-C (e.g. our stop command, or an admin)

Status = Dict[str, Any]


class RunStatus:
    """The fields of the status file; `update` changes some of them and rewrites the file."""

    def __init__(self, path: str, **fields: Any) -> None:
        self.path: str = path
        self.fields: Status = {"state": RUNNING, "started": time.time(), **fields}
        self.write()

    def update(self, **fields: Any) -> None:
        self.fields.update(fields)
        self.write()

    def write(self) -> None:
        self.fields["updated"] = time.time()
        tmp: str = f"{self.path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.fields, f, indent=1)
        os.replace(tmp, self.path)


def read_status(path: str) -> Status:
    with open(path) as f:
        return json.load(f)
