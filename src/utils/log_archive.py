"""
Pack a run's Gurobi logs (one file per solve, thousands per run) into one archive when the run
ends: the lab home allows 300k files (learnt_results.md 5.2), and 65 runs had filled it.
"""
import os
import shutil
import tarfile
import tempfile
from typing import Optional


def archive_path(folder: str) -> str:
    """<folder>.tar.gz, or <folder>_2.tar.gz, _3, ... if a resumed run already packed one."""
    path: str = f"{folder}.tar.gz"
    k: int = 2
    while os.path.exists(path):
        path = f"{folder}_{k}.tar.gz"
        k += 1
    return path


def pack_logs(folder: str) -> Optional[str]:
    """
    Replace `folder` by one .tar.gz next to it (the archive's top folder has the folder's name).
    The archive is written in the local temporary folder first, then the folder is deleted and the
    archive moved in: it works even when the quota of files is already full. Returns the archive's
    path, or None if there is nothing to pack.
    """
    folder = os.path.normpath(folder)
    if not os.path.isdir(folder) or not os.listdir(folder):
        return None
    target: str = archive_path(folder)
    fd, temporary = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    try:
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(folder, arcname=os.path.basename(folder))
        with tarfile.open(temporary, "r:gz") as tar:  # readable before anything is deleted
            tar.getmembers()
        shutil.rmtree(folder)
        shutil.move(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target
