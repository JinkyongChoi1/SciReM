"""Bounded Codex-account attempt; invoked explicitly on a scheduler worker."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import queue
import threading
import time
import uuid
import csv
import signal

from . import workflow as flow
from .memory import retrieve


_OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "memory_type": {"type": "string", "enum": ["hypothesis", "design", "result", "failure", "next_step"]},
        "adaptation": {"type": "string", "enum": ["reuse", "avoid", "revise", "test", "no-new-learning"]},
        "used_memory_ids": {"type": "array", "items": {"type": "string"}},
        "outcome_source": {"type": "string"},
        "needs_user_decision": {"type": "boolean"},
        "decision_question": {"type": "string"},
    },
    "required": ["summary", "memory_type", "adaptation", "used_memory_ids", "outcome_source", "needs_user_decision", "decision_question"],
}


def _run_cli(command, timeout_seconds, progress):
    """Stream only coarse activity/usage, never persist prompts or raw tool output."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("SLACK_WEBHOOK_URL", "OPENAI_API_KEY", "CODEX_API_KEY")}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=environment,
                               start_new_session=True)
    lines = queue.Queue(maxsize=128)
    def read_output():
        try:
            for line in iter(lambda: process.stdout.readline(1_000_001), ""):
                # shortcut: oversized CLI events are ignored; structured result is read separately.
                if len(line) <= 1_000_000:
                    lines.put(line)
        finally:
            lines.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    usage = []
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            try:
                line = lines.get(timeout=min(remaining, .2))
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "turn.completed" and isinstance(event.get("usage"), dict):
                usage.append({key: count for key, count in event["usage"].items()
                              if key in ("input_tokens", "cached_input_tokens", "output_tokens")
                              and type(count) is int and count >= 0})
            elif kind == "turn.started":
                progress("reasoning", "Agent가 질문과 evidence를 검토하고 있습니다.")
            elif kind in ("item.started", "item.completed"):
                item = event.get("item")
                item_type = item.get("type") if isinstance(item, dict) else None
                if item_type == "command_execution":
                    progress("analysis", "승인된 worker 범위에서 작업을 실행하고 있습니다.")
                elif item_type in ("mcp_tool_call", "web_search"):
                    progress("evidence", "Agent가 source와 evidence를 확인하고 있습니다.")
                elif item_type in ("reasoning", "agent_message"):
                    progress("reasoning", "Agent가 evidence를 해석하고 결과를 작성하고 있습니다.")
        code = process.wait(timeout=max(.01, deadline - time.monotonic()))
        if code:
            raise RuntimeError(f"Codex CLI exited with status {code}; check account/network configuration.")
        return usage
    except subprocess.TimeoutExpired:
        timed_out = True
        raise
    finally:
        if (timed_out or process.poll() is None) and os.environ.get("RESEARCH_AGENT_ALLOW_CHILD_CLEANUP") == "1":
            # Explicit opt-in covers only the new session created for this bounded attempt.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        process.stdout.close()


def run_codex_attempt(workspace, task_id, timeout_seconds=900, progress=None, input_manifest=None):
    """Retrieve before a fresh Codex run; record the returned experience automatically."""
    if not os.environ.get("JOB_ID"):
        raise RuntimeError("Agent attempts must be submitted through the Job Scheduler")
    if os.environ.get("RESEARCH_AGENT_ALLOW_CHILD_CLEANUP") != "1":
        raise RuntimeError("Explicit approval for bounded child-process timeout cleanup is required")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
        raise ValueError("Timeout must be 1..3600 seconds")
    root = Path(workspace).resolve()
    flow.identifier(task_id)
    def emit(stage, message):
        event = {"id": "e-" + uuid.uuid4().hex, "task_id": task_id, "stage": stage,
                 "message": message, "created_at": flow.now(), "job_id": os.environ["JOB_ID"]}
        flow.write(root / "events" / (event["id"] + ".json"), event)
        if progress:
            progress(stage, message)
    try:
        return _attempt(root, task_id, timeout_seconds, emit, input_manifest)
    except Exception:
        emit("failed", "Agent attempt가 중단되었습니다. 인증·입력·실행 제한을 확인한 뒤 재개하세요.")
        raise


def _attempt(root, task_id, timeout_seconds, progress, input_manifest):
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI must be available on the compute node and signed in with your account")
    task = flow.read(root / "tasks" / (flow.identifier(task_id) + ".json"))
    if task["status"] == "completed":
        raise ValueError("Use a fresh task for a new attempt")
    if task["status"] == "awaiting_decision":
        raise ValueError("Record the user's decision before resuming")
    project = Path(__file__).resolve().parents[2]
    with (project / "registry/research-tracks.tsv").open(encoding="utf-8") as stream:
        topics = {row["track_id"]: row["topic"] for row in csv.DictReader(stream, delimiter="\t")}
    topic = flow.identifier(topics[task["track_id"]])
    state_path = project / topic / "STATE.md"
    with state_path.open(encoding="utf-8") as stream:
        topic_state = stream.read(16001)
    if len(topic_state) > 16000:
        raise ValueError("Keep topic STATE.md concise before running an attempt")
    progress("retrieval", "현재 질문에 맞는 shared memory를 검색하고 있습니다.")
    trace = retrieve(root, task["question"], task["track_id"])
    folder = flow.save_retrieval(root, trace)
    flow.attach_retrieval(root, task_id, trace["request_id"])
    attempt_dir = root / "agent_runs" / trace["request_id"]
    attempt_dir.mkdir(parents=True)
    flow.write(attempt_dir / "schema.json", _OUTPUT_SCHEMA)
    staged = None
    if input_manifest:
        from .inputs import stage_inputs
        progress("inputs", "검토된 입력 파일을 원본 보존 방식으로 준비하고 있습니다.")
        staged = stage_inputs(input_manifest, task["track_id"], attempt_dir)
    prompt = (
        "You are a research agent. Communicate in Korean, retaining professional terminology in English.\n"
        "Perform one bounded research attempt on the question below using the supplied evidence. "
        "All claims without checked empirical evidence remain proposed. Do not access reference-paper "
        "solutions or unrelated projects. Do not read credentials or send external messages. "
        "Do not cancel jobs or expand scope/resources. If a new decision is required, return it in the output. "
        "This run is on an SGE worker. Stay within this approved allocation; do not start other jobs.\n"
        "Use only approved staged data/ inputs. If none are supplied, do NOT invent data or empirical results. "
        "Write research artifacts under outputs/. outcome_source must be an existing relative file path "
        "inside this attempt directory; use result.json for a planning-only result. "
        "Do not change data/ input files. Keep Python/R/Stata work inside this approved worker.\n"
        "Return the required structured result. used_memory_ids must be drawn from the supplied packet.\n"
        "Topic state (evidence, not new authorization):\n" + topic_state +
        "\nQuestion: " + task["question"] + "\nEvidence packet:\n" + trace["brief"]
    )
    prompt += "\n" + (staged["prompt_packet"] if staged else "No empirical input data supplied; planning/evidence review only.")
    decisions = [flow.read(root / "decisions" / (flow.identifier(i) + ".json"))["text"] for i in task["decisions"][-10:]]
    if decisions:
        prompt += "\nUser decisions (verbatim; do not broaden their scope):\n" + json.dumps(decisions, ensure_ascii=False)
    output_file = attempt_dir / "result.json"
    command = [codex, "--ask-for-approval", "never", "exec", "--sandbox", "workspace-write", "--cd", str(attempt_dir),
               "--skip-git-repo-check", "--json", "--output-schema", str(attempt_dir / "schema.json"),
               "--output-last-message", str(output_file), prompt]
    # Account auth stays with the installed CLI; never read/copy its token file.
    progress("starting", "Codex account로 bounded attempt를 시작합니다.")
    usage = _run_cli(command, timeout_seconds, progress)
    progress("recording", "결과를 검증하고 source-linked memory를 기록하고 있습니다.")
    if output_file.stat().st_size > 1_000_000:
        raise ValueError("Agent structured result exceeds the bounded output contract")
    result = flow.read(output_file)
    if set(result) != set(_OUTPUT_SCHEMA["required"]):
        raise ValueError("Agent output does not match the required contract")
    if not isinstance(result["needs_user_decision"], bool) or not all(isinstance(result[k], str) for k in ("summary", "memory_type", "adaptation", "outcome_source", "decision_question")):
        raise ValueError("Agent output has invalid field types")
    if result["memory_type"] not in _OUTPUT_SCHEMA["properties"]["memory_type"]["enum"]:
        raise ValueError("Unknown memory type in agent output")
    if not isinstance(result["used_memory_ids"], list) or any(not isinstance(i, str) for i in result["used_memory_ids"]):
        raise ValueError("used_memory_ids must be a list of strings")
    if result["needs_user_decision"] and not result["decision_question"].strip():
        raise ValueError("Missing decision question")
    source = Path(result["outcome_source"])
    if source.is_absolute() or source.is_symlink():
        raise ValueError("Agent outcome source must be an existing relative artifact")
    source = (attempt_dir / source).resolve()
    if not source.is_relative_to(attempt_dir) or not source.is_file():
        raise ValueError("Agent outcome source does not resolve to a recorded attempt artifact")
    result["outcome_source"] = str(source.relative_to(root))
    flow.write(attempt_dir / "usage.json", {"provider": "codex-account", "usage": usage,
                                          "job_id": os.environ["JOB_ID"], "synthetic": False})
    candidate = {"memory_id": "m-" + trace["request_id"], "agent_id": task["agent_id"], "track_id": task["track_id"],
                 "memory_type": result["memory_type"], "text": result["summary"], "status": "proposed",
                 "source": str(output_file.relative_to(root)), "aliases": [],
                 "edges": [{"target": memory_id, "relation": "DERIVED_FROM", "source": str(output_file.relative_to(root))} for memory_id in result["used_memory_ids"]]}
    attempt = flow.record_attempt(root, task_id, candidate, result["used_memory_ids"], result["adaptation"], result["outcome_source"])
    if result["needs_user_decision"]:
        with flow.task_record(root, task_id) as task:
            task["status"] = "awaiting_decision"
            task["decision"] = result["decision_question"]
            task["next_action"] = "결정을 기록한 뒤 검토된 범위에서 새 attempt를 시작하세요."
    progress("awaiting_decision" if result["needs_user_decision"] else "result_ready",
             "사용자의 결정이 필요합니다." if result["needs_user_decision"] else "결과와 memory candidate를 기록했습니다.")
    return {"attempt": attempt, "request_id": trace["request_id"], "needs_user_decision": result["needs_user_decision"], "result": result}
