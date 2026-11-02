"""Bounded, loopback-only lab controller; run it inside an approved SGE job."""

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
import uuid

from . import workflow as flow
from .memory import build_snapshot


FOUR_ROLE_WORKSPACE = Path(__file__).resolve().parents[2] / "workspaces/four-role-20261023"


def _memory_inventory(root, track_id=None):
    """Read one published snapshot; query-specific scores remain in its retrieval trace."""
    root = Path(root)
    missing = {"status": "unavailable", "complete": False, "snapshot_id": None,
               "nodes": [], "edges": []}
    current = root / "CURRENT"
    try:
        if root.resolve() != root or current.resolve() != current:
            raise ValueError("Memory path is a symbolic link")
        if not current.is_file():
            return dict(missing, message="No published memory snapshot is available for this research record.")
        snapshot_id = flow.identifier(current.read_text().strip())
        folder = root / "snapshots" / snapshot_id
        manifest_path, graph_path = folder / "manifest.json", folder / "graph.json"
        if any(path.resolve() != path for path in (folder, manifest_path, graph_path)):
            raise ValueError("Snapshot path is a symbolic link")
        manifest = flow.read(manifest_path)
        raw = graph_path.read_bytes()
        if (manifest["snapshot_id"] != snapshot_id or
                hashlib.sha256(raw).hexdigest() != manifest["files"]["graph.json"]):
            raise ValueError("Published graph does not match its manifest")
        graph = json.loads(raw)
        nodes, edges = graph["nodes"], graph["edges"]
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Malformed graph inventory")
        identifiers = {node["memory_id"] for node in nodes}
        if len(identifiers) != len(nodes) or any(
                edge["origin"] not in identifiers or edge["target"] not in identifiers for edge in edges):
            raise ValueError("Graph endpoints are not closed")
        if track_id is not None:
            nodes = [node for node in nodes if node.get("track_id") == track_id]
            identifiers = {node["memory_id"] for node in nodes}
            edges = [edge for edge in edges if edge["origin"] in identifiers and edge["target"] in identifiers]
        return {"status": "available", "complete": True, "snapshot_id": snapshot_id,
                "track_id": track_id,
                "nodes": [{key: value for key, value in node.items() if key != "vector"} for node in nodes],
                "edges": edges}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return dict(missing, status="error", message="The published memory inventory could not be verified.")


def _recorded_activity(experiment):
    """Describe recorded operations, never infer that an agent read or understood them."""
    if not experiment:
        return []
    activity = []
    inventory = experiment.get("memory_inventory") or {}
    if inventory.get("status") == "available":
        activity.append({"id": "snapshot-" + inventory["snapshot_id"], "kind": "memory_published",
            "label": "Published memory snapshot: %d records" % len(inventory["nodes"]),
            "snapshot_id": inventory["snapshot_id"], "source": "shared-memory/CURRENT",
            "status": "recorded"})
    for role, trace in experiment.get("traces", {}).items():
        ids = list(trace.get("selected_ids") or trace.get("included_ids") or [])
        activity.append({"id": "retrieval-" + role, "kind": "retrieval", "role": role,
            "label": "Memory retrieval recorded", "memory_ids": ids,
            "snapshot_id": trace.get("snapshot_id"), "source": role + " retrieval trace",
            "status": trace.get("outcome", "recorded")})
        if (trace.get("actual_delivery") or {}).get("full_packet_delivered") is True:
            activity.append({"id": "delivery-" + role, "kind": "context_delivery", "role": role,
                "label": "Context delivery recorded; reading and understanding are not verified",
                "source": role + " retrieval trace: actual_delivery", "status": "recorded"})
    for index, call in enumerate((experiment.get("report") or {}).get("model_calls", [])):
        activity.append({"id": "model-call-%d" % index, "kind": "model_call",
            "label": "Model call recorded: " + str(call.get("stage") or "stage unavailable"),
            "attempt": call.get("attempt"), "status": call.get("parse_status") or "recorded",
            "source": "report.json:model_calls[%d]" % index})
    return activity


class LiveController:
    def __init__(self, workspace, registry_path=None, runner=None, timeout_seconds=900, input_manifest=None,
                 experiment_report=None, experiment_trajectory=None):
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
            raise ValueError("Attempt timeout must be 1..3600 seconds")
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        registry = Path(registry_path) if registry_path else Path(__file__).resolve().parents[2] / "registry/research-tracks.tsv"
        with registry.open(encoding="utf-8", newline="") as stream:
            self.tracks = [{"id": flow.identifier(row["track_id"]), "topic": row["topic"]}
                           for row in csv.DictReader(stream, delimiter="\t")]
        if not self.tracks:
            raise ValueError("An authoritative topic registry is required")
        self.account_runner = runner is None
        if self.account_runner:
            from .agent import run_codex_attempt
            runner = run_codex_attempt
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.lock = threading.RLock()
        self.active_task_id = None
        self.worker = None
        self.latest_result = None
        self.input_manifest = Path(input_manifest).resolve() if input_manifest else None
        self.experiment_report = None
        self.experiment_trajectory = None
        if experiment_report is not None:
            report_path = Path(experiment_report).resolve()
            try:
                relative = report_path.relative_to(FOUR_ROLE_WORKSPACE.resolve())
            except ValueError as error:
                raise ValueError("Experiment report must be under the reviewed four-role workspace") from error
            if (len(relative.parts) != 3 or not relative.parts[0].startswith("round-") or
                    not relative.parts[0][6:].isdigit() or not relative.parts[1].isdigit() or
                    report_path.name != "report.json"):
                raise ValueError("Experiment report must be the exact four-role report.json path")
            self.experiment_report = report_path
            trajectory_path = Path(experiment_trajectory).resolve() if experiment_trajectory else None
            try:
                trajectory_relative = trajectory_path.relative_to(
                    (FOUR_ROLE_WORKSPACE / "trajectories").resolve()) if trajectory_path else None
            except ValueError as error:
                raise ValueError("Experiment trajectory must be under the reviewed trajectory workspace") from error
            if trajectory_relative is None or len(trajectory_relative.parts) != 1:
                raise ValueError("Provide the exact reviewed four-role trajectory directory")
            self.experiment_trajectory = trajectory_path
        elif experiment_trajectory is not None:
            raise ValueError("An experiment trajectory requires an exact report path")
        self.inputs = []
        if self.input_manifest:
            manifest = flow.read(self.input_manifest)
            self.inputs = [{"track_id": track, "files": [record["name"] for record in records]}
                           for track, records in manifest["tracks"].items()]

    def _can_run(self):
        return ((self.root / "CURRENT").is_file() and
                (not self.account_runner or
                 (os.environ.get("RESEARCH_AGENT_ALLOW_CHILD_CLEANUP") == "1" and
                  shutil.which("codex") is not None)))

    def _task(self, task_id):
        return flow.read(self.root / "tasks" / (flow.identifier(task_id) + ".json"))

    def _event(self, task_id, phase, message):
        # Persist only bounded, human-readable milestones, never raw CLI transcripts.
        item = {"id": "e-" + uuid.uuid4().hex, "task_id": task_id,
                "phase": phase, "message": str(message)[:500], "updated_at": flow.now()}
        item["created_at"] = item["updated_at"]
        path = self.root / "live_events" / (item["id"] + ".json")
        flow.write(path, item)
        return item

    def create_task(self, track_id, question):
        if track_id not in {track["id"] for track in self.tracks}:
            raise ValueError("Unknown research track")
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError("A bounded research question is required")
        with self.lock:
            task = flow.create_task(self.root, track_id, question.strip(), "codex-" + uuid.uuid4().hex)
            dispatch = {"owner": task["agent_id"], "task_id": task["id"], "goal": question.strip(),
                        "scope": "One bounded attempt in the already reviewed SGE allocation; no new jobs or external messages.",
                        "done": "Validated retrieval/write receipts, shared publication, and a source-linked result or explicit user decision.",
                        "inputs": str(self.input_manifest) if self.input_manifest else "No empirical data; evidence review only.",
                        "output": "agent_runs/<request_id>/outputs and task/memory receipts",
                        "scheduler_budget": {"job_id": os.environ.get("JOB_ID"), "attempt_timeout_seconds": self.timeout_seconds},
                        "created_at": flow.now()}
            flow.write(self.root / "dispatch" / (task["id"] + ".json"), dispatch)
            self._event(task["id"], "created", "Research question recorded. Review the scope before starting.")
            return task

    def decision(self, task_id, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("A bounded decision is required")
        with self.lock:
            if self.active_task_id:
                raise RuntimeError("Wait for the current attempt")
            if self._task(task_id)["status"] != "awaiting_decision":
                raise ValueError("This task has no pending decision")
            decision = flow.record_decision(self.root, task_id, text)
            self._event(task_id, "ready", "Your decision was recorded. Use Start to begin the next attempt.")
            return decision

    def run(self, task_id):
        if not os.environ.get("JOB_ID"):
            raise RuntimeError("Agent execution requires an approved SGE job")
        with self.lock:
            if self.active_task_id:
                raise RuntimeError("Only one bounded attempt may run at a time")
            task = self._task(task_id)
            if task["status"] not in ("awaiting_retrieval", "ready", "failed"):
                raise ValueError("Task is not ready for an attempt")
            current = self.root / "CURRENT"
            if not current.is_file():
                raise ValueError("Initialize a reviewed memory snapshot before starting")
            manifest = flow.read(self.root / "snapshots" / flow.identifier(current.read_text().strip()) / "manifest.json")
            if manifest["backend"] not in ("lexical", "fastembed"):
                raise ValueError("This controller requires a local embedding backend")
            if not self._can_run():
                raise ValueError("Account attempts require Codex CLI and explicit child-cleanup approval")
            self.active_task_id = task_id
            self.latest_result = None
            self._event(task_id, "running", "Retrieving shared memory for the current question.")
            self.worker = threading.Thread(target=self._work, args=(task_id, manifest), daemon=False)
            self.worker.start()
            return {"task_id": task_id, "status": "running"}

    def _work(self, task_id, manifest):
        try:
            options = {"timeout_seconds": self.timeout_seconds,
                       "progress": lambda phase, message: self._event(task_id, phase, message)}
            if self.input_manifest:
                options["input_manifest"] = self.input_manifest
            output = self.runner(self.root, task_id, **options)
            with self.lock:
                result = output["result"]
                self.latest_result = {key: result.get(key) for key in
                                      ("summary", "needs_user_decision", "decision_question", "outcome_source", "used_memory_ids")}
                self.latest_result["task_id"] = task_id
                self.latest_result["synthetic"] = bool(output.get("synthetic", False))
                with flow.task_record(self.root, task_id) as task:
                    task["result"] = dict(self.latest_result)
            self._event(task_id, "publishing", "Publishing new memory records to the shared snapshot.")
            build_snapshot(self.root, manifest["backend"], manifest["model_version"])
            if output["needs_user_decision"]:
                self._event(task_id, "awaiting_decision", "Your decision is needed. Review the question and respond.")
            else:
                flow.complete_task(self.root, task_id)
                self._event(task_id, "completed", "Attempt and memory records are complete. Review the results.")
        except BaseException as exc:
            # Native embedding panics can inherit BaseException; every worker failure needs a status.
            # Exception text may contain paths, prompts, environment values, or credentials.
            with flow.task_record(self.root, task_id) as task:
                if task["status"] != "awaiting_decision":
                    task["status"] = "failed"
                    task["next_action"] = "Execution failed. Review the records before retrying within the same scope."
                else:
                    task["next_action"] = "Memory publication failed. Your decision is still needed; execution will not resume automatically."
                task["failure_type"] = type(exc).__name__
            self._event(task_id, "failed", "Execution or result validation failed. Raw credentials and transcripts are not displayed.")
        finally:
            with self.lock:
                self.active_task_id = None

    def wait(self, timeout=None):
        if self.worker:
            self.worker.join(timeout)
            return not self.worker.is_alive()
        return True

    def state(self):
        with self.lock:
            tasks = sorted((flow.read(path) for path in (self.root / "tasks").glob("*.json")),
                           key=lambda task: task["created_at"], reverse=True)
            if self.active_task_id:
                tasks.sort(key=lambda task: task["id"] != self.active_task_id)
            tasks = tasks[:20]
            for task in tasks:
                if task["retrievals"]:
                    request_id = task["retrievals"][-1]
                    flow.retrieval_receipt(self.root, request_id)
                    task["trace"] = flow.read(self.root / "retrievals" / flow.identifier(request_id) / "trace.json")
                if task["id"] == self.active_task_id:
                    task["status"] = "running"
            events = sorted((flow.read(path) for path in (self.root / "live_events").glob("*.json")),
                            key=lambda event: event["updated_at"])[-20:]
            activity = events[-1] if events else {"phase": "idle", "message": "Enter a research question.", "updated_at": None, "task_id": None}
            trace = None
            selected_task = next((t for t in tasks if t["id"] == self.active_task_id), tasks[0] if tasks else None)
            if selected_task and selected_task["retrievals"]:
                request_id = selected_task["retrievals"][-1]
                flow.retrieval_receipt(self.root, request_id)
                trace = flow.read(self.root / "retrievals" / flow.identifier(request_id) / "trace.json")
            trials = [flow.read(path) for path in (self.root / "trials").glob("*.json")]
            latest_result = selected_task.get("result") if selected_task else None
            experiment = self._experiment_state()
            return {"tasks": tasks, "tracks": self.tracks, "active_task_id": self.active_task_id,
                    "busy": self.active_task_id is not None,
                    "activity": activity, "events": events, "trace": trace, "trials": trials,
                    "latest_result": latest_result,
                    "inputs": self.inputs,
                    "experiment": experiment,
                    "memory_inventory": (experiment.get("memory_inventory") if experiment else
                        _memory_inventory(self.root, selected_task.get("track_id") if selected_task else None)),
                    "synthetic": bool((trace or {}).get("synthetic") or (latest_result or {}).get("synthetic")),
                    "capabilities": {"agent_run": self._can_run()},
                    "attempt_timeout_seconds": self.timeout_seconds}

    def _experiment_state(self):
        if self.experiment_report is None:
            return None
        if not self.experiment_report.is_file():
            return {"status": "unknown", "message": "Waiting for the scheduler report."}
        try:
            if getattr(self, "read_only", False) and self.experiment_report.resolve() != self.experiment_report:
                raise ValueError("Selected report path changed")
            stat = self.experiment_report.stat()
            if stat.st_size > 8_000_000:
                return {"status": "error", "message": "The scheduler report exceeds the viewer's supported size."}
            report = flow.read(self.experiment_report)
            trajectory = Path(report.get("trajectory_root", "")).resolve()
            if getattr(self, "read_only", False):
                if trajectory != self.experiment_trajectory or self.experiment_report.resolve() != self.experiment_report:
                    raise ValueError("Report does not match the explicitly selected run")
            else:
                trajectory_part = trajectory.relative_to((FOUR_ROLE_WORKSPACE / "trajectories").resolve())
                round_number = int(self.experiment_report.parent.parent.name.removeprefix("round-"))
                if (len(trajectory_part.parts) != 1 or trajectory != self.experiment_trajectory or
                        report.get("round") != round_number or
                        str(report.get("job_id")) != self.experiment_report.parent.name):
                    return {"status": "error", "message": "Report, Scheduler job, round, and trajectory do not match."}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"status": "error", "message": "The scheduler report or selected run path could not be verified."}
        active = report.get("active_attempt")
        attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
        resumptions = report.get("resumptions") if isinstance(report.get("resumptions"), list) else []
        latest = attempts[-1] if attempts else resumptions[-1] if resumptions else None
        selected = active if isinstance(active, dict) else latest if isinstance(latest, dict) else {}
        attempt_id = selected.get("attempt")
        current = None
        receipt, traces, artifacts = None, {}, {}
        if type(attempt_id) is int and attempt_id > 0:
            directory = selected.get("artifact_directory", "attempt-%04d" % attempt_id)
            if not isinstance(directory, str) or not directory or Path(directory).is_absolute() or ".." in Path(directory).parts or not Path(directory).parts:
                return {"status": "error", "message": "Invalid attempt artifact directory."}
            current = trajectory / directory
            if current.resolve() != current:
                return {"status": "error", "message": "Current attempt path is a symbolic link."}
            receipt_path = current / "orchestrator.json"
            if receipt_path.is_file():
                try:
                    if receipt_path.resolve() != receipt_path:
                        return {"status": "error", "message": "Orchestrator receipt path is a symbolic link."}
                    if receipt_path.stat().st_size <= 1_000_000:
                        receipt = flow.read(receipt_path)
                except (OSError, ValueError):
                    pass
            for role, filename in (("Hypothesis", "hypothesis-retrieval.json"),
                                   ("Empiricist", "empiricist-retrieval.json")):
                path = current / filename
                if path.is_file():
                    try:
                        if path.resolve() != path:
                            return {"status": "error", "message": "Retrieval trace path is a symbolic link."}
                        if path.stat().st_size <= 8_000_000:
                            traces[role] = flow.read(path)
                    except (OSError, ValueError):
                        pass
            for name, fields in (("hypothesis.json", ("hypothesis", "expected_pattern", "rival_explanation", "focal_test")),
                                 ("plan.json", ("design_rationale", "analysis", "focal_test", "model_specification", "sample_recipe")),
                                 ("current-result-facts.json", ("significance", "direction", "support", "alpha",
                                                                  "primary_test", "estimate", "analysis",
                                                                  "analysis_status", "evidence_id", "metrics", "p_value")),
                                 ("decision.json", ("action", "basis_keywords", "next_inquiry", "evidence_ids"))):
                path = current / name
                if path.is_file():
                    try:
                        if path.resolve() != path:
                            return {"status": "error", "message": "Attempt artifact path is a symbolic link."}
                        if path.stat().st_size <= 1_000_000:
                            data = flow.read(path)
                            artifacts[name.removesuffix(".json")] = {key: data.get(key) for key in fields
                                                                     if key in data}
                    except (OSError, ValueError):
                        pass
        safe_report = {key: report.get(key) for key in
                       ("job_id", "round", "status", "active_attempt", "active_stage", "question",
                        "retrieval_strategy", "embedding_backend", "research_role_layout", "orchestrator_assessment_enabled",
                        "synthetic_only", "model_backend", "statistics_backend")}
        report_age = max(0, round(time.time() - stat.st_mtime))
        safe_report["report_age_seconds"] = report_age
        safe_report["freshness"] = "stale" if report.get("status") == "running" and report_age > 300 else "fresh"
        safe_report["attempts"] = [{key: attempt.get(key) for key in
                                    ("attempt", "episode", "artifact_directory", "status", "analysis_status", "decision_status", "seconds")}
                                   for attempt in attempts if isinstance(attempt, dict)]
        if not attempts and isinstance(latest, dict):
            safe_report["attempts"] = [{key: latest.get(key) for key in
                ("attempt", "artifact_directory", "status", "analysis_status", "decision_status", "seconds")}]
        model_calls = report.get("model_calls") if isinstance(report.get("model_calls"), list) else []
        safe_report["model_calls"] = [{key: call.get(key) for key in
                                       ("attempt", "stage", "parse_status", "failure_type")}
                                      for call in model_calls if isinstance(call, dict)]
        return {"status": report.get("status", "unknown"), "report": safe_report,
                "receipt": receipt, "traces": traces, "artifacts": artifacts,
                "memory_inventory": _memory_inventory(trajectory / "shared-memory")}


class ArtifactController(LiveController):
    """Read an explicitly selected run; never launch or modify its research work."""
    read_only = True

    def __init__(self, report_path, trajectory_path):
        self.experiment_report = Path(report_path).expanduser().resolve()
        self.experiment_trajectory = Path(trajectory_path).expanduser().resolve()
        self.root = self.experiment_trajectory

    def memory_review(self, memory_id, action, note):
        if not isinstance(memory_id, str) or action not in ("approve", "request_change"):
            raise ValueError("Invalid memory review")
        if not isinstance(note, str) or len(note) > 8000:
            raise ValueError("A bounded review note is required")
        experiment = self._experiment_state()
        inventory = experiment.get("memory_inventory") or {}
        if inventory.get("status") != "available" or memory_id not in {
                node["memory_id"] for node in inventory.get("nodes", [])}:
            raise ValueError("Review target is absent from the verified selected store")
        receipt = {"id": "review-" + uuid.uuid4().hex, "memory_id": memory_id,
            "action": action, "note": note.strip(), "created_at": flow.now(),
            "snapshot_id": inventory["snapshot_id"], "status": "recorded",
            "application_status": "not_applied"}
        folder = self.root / "human-decisions"
        if folder.resolve() != folder:
            raise ValueError("Review directory is a symbolic link")
        folder.mkdir(mode=0o700, exist_ok=True)
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            descriptor = os.open(receipt["id"] + ".json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(directory)
        return receipt

    def _human_reviews(self):
        folder = self.root / "human-decisions"
        if folder.resolve() != folder:
            raise ValueError("Review directory is a symbolic link")
        records = []
        for path in sorted(folder.glob("review-*.json")):
            if path.resolve() != path:
                raise ValueError("Review receipt is a symbolic link")
            record = flow.read(path)
            if (record.get("id") + ".json" != path.name or
                    record.get("action") not in ("approve", "request_change") or
                    record.get("status") != "recorded" or record.get("application_status") != "not_applied"):
                raise ValueError("Invalid review receipt")
            records.append(record)
        return sorted(records, key=lambda item: item["created_at"])

    def state(self):
        experiment = self._experiment_state()
        report = experiment.get("report", {})
        status = experiment["status"]
        stage = report.get("active_stage")
        message = experiment.get("message") or (
            "Running · " + str(stage or "waiting for a status record") if status == "running" else
            "Recorded research · " + status)
        if report.get("freshness") == "stale":
            message = "Waiting for a status update · last recorded stage: " + str(stage or status)
        inventory = experiment.get("memory_inventory") or {}
        reviews, review_status = [], "available"
        try:
            if inventory.get("status") == "available":
                reviews = self._human_reviews()
        except (OSError, ValueError, KeyError, TypeError):
            review_status = "error"
        return {"tasks": [], "tracks": [], "active_task_id": None,
                "busy": status == "running", "activity": {"phase": stage, "message": message},
                "events": [], "trace": None, "trials": [], "latest_result": None,
                "inputs": [], "experiment": experiment,
                "memory_inventory": experiment.get("memory_inventory", {
                    "status": "unavailable", "complete": False, "snapshot_id": None,
                    "nodes": [], "edges": [], "message": "No verified research record is available."}),
                "synthetic": bool(report.get("synthetic_only")),
                "ai_activity": _recorded_activity(experiment),
                "human_reviews": reviews, "human_review_status": review_status,
                "capabilities": {"agent_run": False, "read_only": True,
                    "memory_review": inventory.get("status") == "available" and review_status == "available"}}

    def wait(self, timeout=None):
        return True


def open_artifact_server(workspace, report_path, trajectory_path, port=8765, *, local=False):
    """Create a private connection receipt for the existing run observer."""
    import socket
    controller = ArtifactController(report_path, trajectory_path)
    server = create_server(controller, port, local=local)
    try:
        sessions = Path(workspace).resolve() / "sessions"
        sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
        session = sessions / ("watch-" + uuid.uuid4().hex + ".json")
        descriptor = os.open(session, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"host": socket.gethostname(), "port": server.server_port,
                       "job_id": os.environ.get("JOB_ID"), "created_at": flow.now(),
                       "browser_url": f"http://127.0.0.1:{server.server_port}/#token={server.token}"}, stream, indent=2)
    except BaseException:
        server.server_close()
        raise
    return server, session


def create_server(controller, port=0, token=None, *, local=False):
    from .execution import require_execution
    require_execution(local=local)
    token = token or secrets.token_urlsafe(32)
    if not isinstance(token, str) or len(token) < 20:
        raise ValueError("A strong session token is required")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass

        def _reply(self, code, value, html=False):
            body = value.encode("utf-8") if html else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self, write=False):
            origins = {"http://127.0.0.1:" + str(self.server.server_port), "http://localhost:" + str(self.server.server_port)}
            host = self.headers.get("Host", "")
            if "http://" + host not in origins:
                self._reply(403, {"error": "Only loopback connections are allowed."})
                return False
            if write and self.headers.get("Origin") not in origins:
                self._reply(403, {"error": "Only same-origin requests are allowed."})
                return False
            if self.path not in ("/", "/assets/NotoSansKR.ttf") and not secrets.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                self._reply(403, {"error": "Session authentication is required."})
                return False
            return True

        def do_GET(self):
            if not self._allowed():
                return
            if self.path == "/":
                from .live_viewer import render_live_dashboard
                # Other users on this compute node can also access localhost.
                self._reply(200, render_live_dashboard(""), html=True)
            elif self.path == "/assets/NotoSansKR.ttf":
                font = Path(__file__).with_name("assets") / "NotoSansKR.ttf"
                if not font.is_file():
                    self._reply(404, {"error": "Font unavailable; system font fallback."})
                    return
                body = font.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/ttf")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                try:
                    self._reply(200, controller.state())
                    self.server.first_read.set()
                except (ValueError, OSError, KeyError):
                    self._reply(409, {"error": "Record validation failed. Inspect the source records."})
            else:
                self._reply(404, {"error": "Unknown route."})

        def do_POST(self):
            if not self._allowed(write=True):
                return
            memory_review = self.path == "/api/memory-review" and isinstance(controller, ArtifactController)
            if getattr(controller, "read_only", False) and not memory_review:
                self._reply(405, {"error": "This dashboard observes a run; it cannot start or modify research."})
                return
            routes = {"/api/memory-review": (controller.memory_review, ("memory_id", "action", "note"))} if memory_review else {
                      "/api/task": (controller.create_task, ("track_id", "question")),
                      "/api/run": (controller.run, ("task_id",)),
                      "/api/decision": (controller.decision, ("task_id", "text"))}
            if self.path not in routes:
                self._reply(404, {"error": "Unknown route."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Bounded JSON required")
                data = json.loads(self.rfile.read(size))
                method, keys = routes[self.path]
                if not isinstance(data, dict) or set(data) != set(keys):
                    raise ValueError("Invalid request fields")
                result = method(*(data[key] for key in keys))
                self._reply(200, result)
            except (ValueError, TypeError, KeyError, FileNotFoundError):
                self._reply(400, {"error": "Check the input or task state."})
            except RuntimeError:
                self._reply(409, {"error": "Wait for the active attempt to finish before retrying."})
            except OSError:
                self._reply(500, {"error": "The record could not be saved. Check the service logs."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.token = token
    server.first_read = threading.Event()
    return server
