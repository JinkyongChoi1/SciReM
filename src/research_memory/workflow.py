"""Small artifact lifecycle for manual Codex tasks and reproducible trial measurement."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile
import uuid

from .memory import submit_candidate


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Invalid artifact identifier")
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def task_record(workspace, task_id):
    path = Path(workspace) / "tasks" / (identifier(task_id) + ".json")
    lock = path.with_suffix(".lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("Task writer is active; inspect a stale lock manually") from exc
    try:
        task = read(path)
        yield task
        write(path, task)
    finally:
        lock.rmdir()


def create_task(workspace, track_id, question, agent_id="codex"):
    identifier(track_id)
    identifier(agent_id)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Research question is required")
    Path(workspace).mkdir(parents=True, exist_ok=True, mode=0o700)
    task = {"id": "t-" + uuid.uuid4().hex, "track_id": track_id, "question": question,
            "agent_id": agent_id, "status": "awaiting_retrieval", "created_at": now(),
            "next_action": "Job Scheduler의 검토된 allocation에서 현재 질문의 memory를 검색하세요.",
            "retrievals": [], "attempts": [], "decisions": []}
    write(Path(workspace) / "tasks" / (task["id"] + ".json"), task)
    return task


def save_retrieval(workspace, trace):
    identifier(trace["request_id"])
    identifier(trace["snapshot_id"])
    path = Path(workspace) / "retrievals" / trace["request_id"]
    if path.exists():
        raise FileExistsError("A retrieval request is immutable")
    path.mkdir(parents=True)
    write(path / "trace.json", trace)
    (path / "brief.md").write_text(trace["brief"], encoding="utf-8")
    receipt = {"request_id": trace["request_id"], "snapshot_id": trace["snapshot_id"],
               "query": trace["query"], "track_id": trace["track_id"],
               "selected_ids": trace["selected_ids"], "outcome": trace["outcome"], "created_at": now(),
               "trace_sha256": hashlib.sha256((path / "trace.json").read_bytes()).hexdigest(),
               "brief_sha256": hashlib.sha256((path / "brief.md").read_bytes()).hexdigest()}
    write(path / "receipt.json", receipt)
    return path


def retrieval_receipt(workspace, request_id):
    folder = Path(workspace) / "retrievals" / identifier(request_id)
    receipt = read(folder / "receipt.json")
    for name, key in (("trace.json", "trace_sha256"), ("brief.md", "brief_sha256")):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError("Retrieval artifact changed after its receipt was created")
    trace = read(folder / "trace.json")
    for key in ("request_id", "snapshot_id", "query", "track_id", "selected_ids", "outcome"):
        if trace[key] != receipt[key]:
            raise ValueError("Trace and receipt disagree")
    return receipt


def attach_retrieval(workspace, task_id, request_id):
    receipt = retrieval_receipt(workspace, request_id)
    with task_record(workspace, task_id) as task:
        if task["status"] == "completed":
            raise ValueError("Completed task cannot accept a new retrieval")
        if receipt["query"] != task["question"] or receipt["track_id"] != task["track_id"]:
            raise ValueError("Retrieval must match this task's question and track")
        if receipt["outcome"] not in ("retrieved", "no-match"):
            raise ValueError("Retrieval error cannot make a task ready")
        if request_id not in task["retrievals"]:
            task["retrievals"].append(request_id)
        task["status"] = "ready"
        task["next_action"] = "Evidence packet을 확인하고 검토된 범위의 analysis plan을 준비하세요."
    return read(Path(workspace) / "tasks" / (task_id + ".json"))


def record_decision(workspace, task_id, user_text, actor="user"):
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("Decision text must be the user's verbatim words")
    identifier(actor)
    with task_record(workspace, task_id) as task:
        decision = {"id": "d-" + uuid.uuid4().hex, "text": user_text, "actor": actor,
                    "created_at": now(), "authority": "user_statement", "task_id": task_id}
        write(Path(workspace) / "decisions" / (decision["id"] + ".json"), decision)
        task["decisions"].append(decision["id"])
        if task["status"] == "awaiting_decision":
            task["status"] = "ready"
            task["decision"] = None
            task["next_action"] = "기록된 결정을 확인한 뒤 허용된 범위에서 재개하세요."
    return decision


def record_attempt(workspace, task_id, record, used_memory_ids, adaptation, outcome_source):
    if adaptation not in ("reuse", "avoid", "revise", "test", "no-new-learning"):
        raise ValueError("Unknown adaptation kind")
    if not isinstance(used_memory_ids, list) or any(not isinstance(i, str) for i in used_memory_ids):
        raise ValueError("used_memory_ids must be a list of memory IDs")
    if not isinstance(outcome_source, str) or not outcome_source.strip():
        raise ValueError("An outcome source is required")
    with task_record(workspace, task_id) as task:
        if task["status"] != "ready" or not task["retrievals"]:
            raise ValueError("A matching retrieval receipt is required before an attempt")
        receipt = retrieval_receipt(workspace, task["retrievals"][-1])
        if not set(used_memory_ids).issubset(receipt["selected_ids"]):
            raise ValueError("Memory use must refer to evidence actually delivered")
        if record.get("track_id") != task["track_id"] or record.get("agent_id") != task["agent_id"]:
            raise ValueError("Candidate track and owner must match the task")
        candidate = submit_candidate(workspace, record)
        attempt = {"id": "a-" + uuid.uuid4().hex, "task_id": task_id, "created_at": now(),
                   "request_id": receipt["request_id"], "snapshot_id": receipt["snapshot_id"],
                   "memory_id": record["memory_id"], "candidate": str(candidate.relative_to(workspace)),
                   "used_memory_ids": used_memory_ids, "adaptation": adaptation,
                   "outcome_source": outcome_source, "status": "candidate-written"}
        attempt["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        write(Path(workspace) / "attempts" / (attempt["id"] + ".json"), attempt)
        task["attempts"].append(attempt["id"])
        task["next_action"] = "결과를 검토하고 index job으로 새 memory candidate를 게시하세요."
    return attempt


def complete_task(workspace, task_id):
    with task_record(workspace, task_id) as task:
        if task["status"] == "awaiting_decision":
            raise ValueError("Resolve the pending user decision before completion")
        if not task["retrievals"] or not task["attempts"]:
            raise ValueError("Retrieval and candidate-write receipts are required before completion")
        for attempt_id in task["attempts"]:
            attempt = read(Path(workspace) / "attempts" / (attempt_id + ".json"))
            candidate = Path(workspace) / attempt["candidate"]
            if not candidate.is_file():
                raise ValueError("Missing memory candidate; task cannot be completed")
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != attempt["candidate_sha256"]:
                raise ValueError("Memory candidate changed after its write receipt")
            receipt = retrieval_receipt(workspace, attempt["request_id"])
            if receipt["query"] != task["question"] or receipt["track_id"] != task["track_id"] or receipt["snapshot_id"] != attempt["snapshot_id"]:
                raise ValueError("Attempt and retrieval receipt do not match the task")
        task["status"] = "completed"
        task["completed_at"] = now()
        task["next_action"] = "다음 연구 질문은 새 task로 시작하세요."
    return read(Path(workspace) / "tasks" / (task_id + ".json"))


def create_trial(workspace, task_id, condition, reference_targets):
    identifier(condition)
    read(Path(workspace) / "tasks" / (identifier(task_id) + ".json"))
    if not isinstance(reference_targets, list) or not reference_targets or any(not isinstance(t, str) or not t for t in reference_targets):
        raise ValueError("Explicit reference target names are required")
    if len(set(reference_targets)) != len(reference_targets):
        raise ValueError("Reference targets must be unique")
    trial = {"id": "trial-" + uuid.uuid4().hex, "task_id": task_id, "condition": condition,
             "created_at": now(), "reference_targets": reference_targets, "events": [], "metrics": {}}
    write(Path(workspace) / "trials" / (trial["id"] + ".json"), trial)
    return trial


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Event timestamps require a timezone")
    return stamp


def record_trial_event(workspace, trial_id, event):
    path = Path(workspace) / "trials" / (identifier(trial_id) + ".json")
    lock = path.with_suffix(".lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("Trial writer is active") from exc
    try:
        trial = read(path)
        item = dict(event)
        item.setdefault("at", now())
        if _timestamp(item["at"]) < _timestamp(trial["created_at"]):
            raise ValueError("Event predates this trial")
        if item.get("kind") not in ("submitted", "started", "finished", "error", "reference_check", "mechanism_review"):
            raise ValueError("Unknown trial event")
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ValueError("Measured events require a source artifact")
        if item["kind"] in ("submitted", "started", "finished", "error"):
            identifier(item.get("run_id"))
        if item["kind"] in ("submitted", "started", "finished") and any(e["kind"] == item["kind"] and e.get("run_id") == item["run_id"] for e in trial["events"]):
            raise ValueError("A run stage may be recorded only once; retries require a new run_id")
        if item["kind"] == "error" and any(e["kind"] == "error" and e.get("run_id") == item["run_id"] and e["at"] == item["at"] and e["source"] == item["source"] for e in trial["events"]):
            raise ValueError("Duplicate error observation")
        if item["kind"] == "reference_check":
            checks = item.get("checks")
            if not isinstance(checks, dict) or set(checks) != set(trial["reference_targets"]) or any(type(v) is not bool for v in checks.values()):
                raise ValueError("Reference check must explicitly pass/fail every target")
        if item["kind"] == "mechanism_review" and (not isinstance(item.get("reviewer"), str) or not item["reviewer"].strip()):
            raise ValueError("Mechanism assessment requires a named reviewer")
        if any(e.get("event_id") == item.get("event_id") for e in trial["events"] if item.get("event_id")):
            raise ValueError("Duplicate trial event ID")
        trial["events"].append(item)
        trial["metrics"] = trial_metrics(trial)
        write(path, trial)
        return trial
    finally:
        lock.rmdir()


def trial_metrics(trial):
    events = sorted(trial["events"], key=lambda e: _timestamp(e["at"]))
    metrics = {"errors": sum(e["kind"] == "error" for e in events), "wall_time": None,
               "queue_time": None, "compute_time": None, "fidelity": None,
               "mechanism_validity": None, "time_to_first_faithful_seconds": None}
    runs = {e["run_id"] for e in events if "run_id" in e}
    queues, computes = [], []
    for run in runs:
        stages = {e["kind"]: _timestamp(e["at"]) for e in events if e.get("run_id") == run}
        for first, last, values in (("submitted", "started", queues), ("started", "finished", computes)):
            if first in stages and last in stages:
                delta = (stages[last] - stages[first]).total_seconds()
                if delta < 0:
                    raise ValueError("Run timestamps are out of order")
                values.append(delta)
    if queues:
        metrics["queue_time"] = sum(queues)
    if computes:
        metrics["compute_time"] = sum(computes)
    checks = [e for e in events if e["kind"] == "reference_check"]
    if checks:
        metrics["fidelity"] = sum(checks[-1]["checks"].values()) / len(trial["reference_targets"])
        metrics["wall_time"] = (_timestamp(checks[-1]["at"]) - _timestamp(trial["created_at"])).total_seconds()
        if metrics["wall_time"] < 0:
            raise ValueError("Reference check predates the trial")
        faithful = next((e for e in checks if all(e["checks"].values())), None)
        if faithful:
            metrics["time_to_first_faithful_seconds"] = (_timestamp(faithful["at"]) - _timestamp(trial["created_at"])).total_seconds()
    reviews = [e for e in events if e["kind"] == "mechanism_review"]
    if reviews:
        metrics["mechanism_validity"] = reviews[-1].get("assessment")
    return metrics
