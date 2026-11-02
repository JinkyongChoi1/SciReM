"""Stage only owner-reviewed, explicitly model-shareable data for one attempt."""

import hashlib
import json
import os
from pathlib import Path
import re

from . import workflow as flow


# shortcut: pilot inputs are capped at 100 MB; review streaming/resource policy for larger studies.
MAX_INPUT_BYTES = 100 * 1024 * 1024


def stage_inputs(manifest_path, track_id, attempt_dir, max_bytes=MAX_INPUT_BYTES):
    """Return a source-linked prompt packet after publishing read-only staged copies."""
    if not os.environ.get("JOB_ID"):
        raise RuntimeError("Input staging must run through the Job Scheduler")
    if track_id not in ("FD", "DC", "PT", "HZ"):
        raise ValueError("Unknown research track")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("Input byte budget must be a positive integer")
    manifest_path = Path(manifest_path).absolute()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("A regular reviewed manifest file is required")
    manifest = flow.read(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {"tracks"} or not isinstance(manifest["tracks"], dict):
        raise ValueError("Manifest must contain only a tracks mapping")
    if any(track not in ("FD", "DC", "PT", "HZ") for track in manifest["tracks"]):
        raise ValueError("Manifest contains an unknown research track")
    records = manifest["tracks"].get(track_id)
    if not isinstance(records, list) or not records:
        raise ValueError("No reviewed inputs are listed for this track")
    planned, names, total = [], set(), 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "name", "allow_external_model"}:
            raise ValueError("Each input must specify only path, name, and allow_external_model")
        if record["allow_external_model"] is not True:
            raise ValueError("Every input needs explicit allow_external_model: true consent")
        name = record["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            raise ValueError("Input name must be a safe basename")
        if name in names or name.lower() == "input-receipt.json":
            raise ValueError("Duplicate or reserved input name")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError("Input source path is required")
        source = Path(record["path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents):
            raise ValueError("Symlink input paths are not supported")
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError("Only regular input files are supported")
        total += source.stat().st_size
        if total > max_bytes:
            raise ValueError("Reviewed input byte budget exceeded")
        names.add(name)
        planned.append((name, source))
    target = Path(attempt_dir).resolve() / "data"
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    files, copied_bytes = [], 0
    for name, source in planned:
        digest, size = hashlib.sha256(), 0
        descriptor = os.open(target / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as original, os.fdopen(descriptor, "wb") as copy:
            for block in iter(lambda: original.read(1024 * 1024), b""):
                copied_bytes += len(block)
                if copied_bytes > max_bytes:
                    raise ValueError("Input grew beyond reviewed byte budget during staging")
                copy.write(block)
                digest.update(block)
                size += len(block)
            copy.flush()
            os.fsync(copy.fileno())
        (target / name).chmod(0o400)
        files.append({"name": name, "source": str(source), "sha256": digest.hexdigest(), "bytes": size})
    scope = "Only data/ files listed here are approved for this attempt and external-model processing."
    receipt_path = target / "input-receipt.json"
    flow.write(receipt_path, {"track_id": track_id, "manifest": str(manifest_path.resolve()),
                              "created_at": flow.now(), "job_id": os.environ["JOB_ID"],
                              "max_bytes": max_bytes, "scope": scope, "files": files})
    receipt_path.chmod(0o400)
    packet = "Approved data-first input scope:\n" + scope + "\n"
    packet += "\n".join("data/" + f["name"] + " · Source: " + json.dumps(f["source"], ensure_ascii=False) +
                        " · SHA256: " + f["sha256"] for f in files)
    packet += "\nDo not access original reference code or paper results. Input receipt: data/input-receipt.json"
    return {"receipt": str(receipt_path), "files": files, "scope": scope, "prompt_packet": packet}
