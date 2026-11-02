"""Bounded, loopback-only lab controller; run it inside an approved SGE job."""

import csv
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
            self._event(task["id"], "created", "연구 질문을 기록했습니다. 실행 범위를 확인한 뒤 시작하세요.")
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
            self._event(task_id, "ready", "사용자 결정을 기록했습니다. 새 attempt는 시작 버튼으로 실행하세요.")
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
            self._event(task_id, "running", "질문에 맞는 shared memory retrieval을 시작합니다.")
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
            self._event(task_id, "publishing", "새 memory candidate를 shared snapshot에 게시합니다.")
            build_snapshot(self.root, manifest["backend"], manifest["model_version"])
            if output["needs_user_decision"]:
                self._event(task_id, "awaiting_decision", "사용자 결정이 필요합니다. 질문을 확인하고 답변하세요.")
            else:
                flow.complete_task(self.root, task_id)
                self._event(task_id, "completed", "Attempt와 memory 기록이 완료되었습니다. 결과를 검토하세요.")
        except BaseException as exc:
            # Native embedding panics can inherit BaseException; every worker failure needs a status.
            # Exception text may contain paths, prompts, environment values, or credentials.
            with flow.task_record(self.root, task_id) as task:
                if task["status"] != "awaiting_decision":
                    task["status"] = "failed"
                    task["next_action"] = "실행이 실패했습니다. 기록을 검토한 뒤 같은 범위에서 재시도하세요."
                else:
                    task["next_action"] = "Memory 게시가 실패했습니다. 사용자 결정은 여전히 필요하며 자동 resume되지 않습니다."
                task["failure_type"] = type(exc).__name__
            self._event(task_id, "failed", "실행 또는 결과 검증에 실패했습니다. 원본 secret나 transcript는 표시하지 않습니다.")
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
            activity = events[-1] if events else {"phase": "idle", "message": "연구 질문을 입력하세요.", "updated_at": None, "task_id": None}
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
                    "synthetic": bool((trace or {}).get("synthetic") or (latest_result or {}).get("synthetic")),
                    "capabilities": {"agent_run": self._can_run()},
                    "attempt_timeout_seconds": self.timeout_seconds}

    def _experiment_state(self):
        if self.experiment_report is None:
            return None
        if not self.experiment_report.is_file():
            return {"status": "unknown", "message": "아직 Scheduler report가 생성되지 않았습니다."}
        try:
            stat = self.experiment_report.stat()
            if stat.st_size > 8_000_000:
                return {"status": "error", "message": "Scheduler report가 표시 가능한 크기를 넘었습니다."}
            report = flow.read(self.experiment_report)
            trajectory = Path(report.get("trajectory_root", "")).resolve()
            trajectory_part = trajectory.relative_to((FOUR_ROLE_WORKSPACE / "trajectories").resolve())
            round_number = int(self.experiment_report.parent.parent.name.removeprefix("round-"))
            if (len(trajectory_part.parts) != 1 or trajectory != self.experiment_trajectory or
                    report.get("round") != round_number or
                    str(report.get("job_id")) != self.experiment_report.parent.name):
                return {"status": "error", "message": "Report, Scheduler job, round, and trajectory do not match."}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"status": "error", "message": "Scheduler report 또는 trajectory 경로를 검증할 수 없습니다."}
        active = report.get("active_attempt")
        attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
        attempt_id = active.get("attempt") if isinstance(active, dict) else None
        if attempt_id is None and attempts:
            attempt_id = attempts[-1].get("attempt") if isinstance(attempts[-1], dict) else None
        current = None
        receipt, traces, artifacts = None, {}, {}
        if type(attempt_id) is int and attempt_id > 0:
            current = trajectory / ("attempt-%04d" % attempt_id)
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
                                 ("plan.json", ("design_rationale", "analysis", "focal_test")),
                                 ("current-result-facts.json", ("significance", "direction", "support", "alpha",
                                                                  "primary_test", "estimate", "analysis",
                                                                  "analysis_status", "evidence_id", "metrics")),
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
                        "retrieval_strategy", "embedding_backend")}
        report_age = max(0, round(time.time() - stat.st_mtime))
        safe_report["report_age_seconds"] = report_age
        safe_report["freshness"] = "stale" if report.get("status") == "running" and report_age > 300 else "fresh"
        safe_report["attempts"] = [{key: attempt.get(key) for key in
                                    ("attempt", "status", "analysis_status", "decision_status", "seconds")}
                                   for attempt in attempts if isinstance(attempt, dict)]
        model_calls = report.get("model_calls") if isinstance(report.get("model_calls"), list) else []
        safe_report["model_calls"] = [{key: call.get(key) for key in
                                       ("attempt", "stage", "parse_status", "failure_type")}
                                      for call in model_calls if isinstance(call, dict)]
        return {"status": report.get("status", "unknown"), "report": safe_report,
                "receipt": receipt, "traces": traces, "artifacts": artifacts}


def create_server(controller, port=0, token=None):
    if not os.environ.get("JOB_ID"):
        raise RuntimeError("The live service must run inside an approved SGE job")
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
                self._reply(403, {"error": "Loopback 접속만 허용됩니다."})
                return False
            if write and self.headers.get("Origin") not in origins:
                self._reply(403, {"error": "동일 Origin 요청만 허용됩니다."})
                return False
            if self.path not in ("/", "/assets/NotoSansKR.ttf") and not secrets.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                self._reply(403, {"error": "Session 인증이 필요합니다."})
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
                except (ValueError, OSError, KeyError):
                    self._reply(409, {"error": "기록 검증에 실패했습니다. 관리자에게 확인하세요."})
            else:
                self._reply(404, {"error": "없는 경로입니다."})

        def do_POST(self):
            if not self._allowed(write=True):
                return
            routes = {"/api/task": (controller.create_task, ("track_id", "question")),
                      "/api/run": (controller.run, ("task_id",)),
                      "/api/decision": (controller.decision, ("task_id", "text"))}
            if self.path not in routes:
                self._reply(404, {"error": "없는 경로입니다."})
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
                self._reply(400, {"error": "입력 또는 task 상태를 확인하세요."})
            except RuntimeError:
                self._reply(409, {"error": "진행 중인 attempt가 끝난 뒤 다시 시도하세요."})
            except OSError:
                self._reply(500, {"error": "기록을 저장하지 못했습니다. 관리자에게 확인하세요."})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.token = token
    return server
