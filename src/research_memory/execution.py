"""Explicit workstation execution without bypassing the university Login Node rule."""
import os
import socket


def require_execution(*, local=False):
    if os.environ.get("JOB_ID"):
        return
    host = socket.gethostname().lower().rstrip(".")
    if local and not (host.startswith("research") or host.endswith(".grid.gsb")):
        return
    raise RuntimeError("Use an approved SGE allocation, or explicit --local on a non-cluster workstation.")
