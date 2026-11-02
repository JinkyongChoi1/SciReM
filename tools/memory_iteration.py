"""One reviewed memory-only development round; scheduling and QA stay external."""
import argparse
import importlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

from two_role_discovery import (ROOT, DATA, SERVER_RUNTIME, QUESTION, PrivateInference,
    TwoRoleMemory, fixed_model_prefix, execute_attempt, run_analysis, write, checkpoint, encode)
from research_memory.evidence import _source
from validate_local_synthetic import WEIGHT
from private_inference import ContextCapacityError, RuntimeIsolationError

ITERATIONS = ROOT / "workspaces/memory-iterations-20261022"
TRAJECTORIES = ITERATIONS / "trajectories"
MAX_ROUNDS = 5
MEMORY_FORMAT = "json"
TOPOLOGY = "hypothesis-analysis-hypothesis"
INFERENCE_DEADLINE_RESERVE_SECONDS = 45
SOURCE_FILES = ("tools/memory_iteration.py", "tools/two_role_discovery.py",
    "tools/discovery_pilot.py", "tools/discovery_analysis.py", "tools/private_inference.py",
    "tools/validate_local_synthetic.py", "src/research_memory/two_role.py",
    "src/research_memory/evidence.py", "src/research_memory/memory.py",
    "src/research_memory/tabular.py", "src/research_memory/model_catalog.py",
    "tools/tsv_memory_iteration.py", "src/research_memory/gpu_device.py", "tools/analysis_reuse.py",
    "tools/agent_r_diagnostic.py")
BENCHMARK_CONDITIONS = ()
RESUME_RESULT = None


def private_worker(directory, started, budget_seconds, *, runtime=SERVER_RUNTIME, device="cpu", cuda_toolkit=None):
    return PrivateInference(directory, runtime, fixed_model_prefix(),
                            started + budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS,
                            device=device, cuda_toolkit=cuda_toolkit)


def reviewed_trajectory(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Trajectory path must be an explicitly reviewed absolute path")
    resolved, base = path.resolve(), TRAJECTORIES.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError("Trajectory must be inside the reviewed iteration trajectories directory")
    return resolved


def source_reference(path, root):
    return _source(path, root)[1]


def stage_receipt_label(counts, label):
    """Keep the first legacy filename and make repeated role calls immutable."""
    counts[label] = counts.get(label, 0) + 1
    return label if counts[label] == 1 else label + "-" + str(counts[label])


def may_start_attempt(condition, attempt_index, elapsed_seconds, budget_seconds, repeat_one_shot=False):
    """A is exactly one full cycle; B/C can iterate until the same finite deadline."""
    if elapsed_seconds >= budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS:
        return False
    if condition == "A" and not repeat_one_shot:
        return attempt_index == 1
    return True


def one_shot_episode(trajectory, index):
    """Separate role-local histories and identical within-episode attempt numbering."""
    root = Path(trajectory) / f"episode-{index:04d}"
    root.mkdir(mode=0o700)
    session = root / "attempt-0001"
    session.mkdir(mode=0o700)
    return session


def fatal_attempt_stop_reason(record):
    """Stop run-level retries only for failures that another bounded attempt cannot repair."""
    if record.get("failure_type") in ("ModuleNotFoundError", "ImportError"):
        return "environment_dependency_no_retry"
    if record.get("failure_type") == "JSONDecodeError":
        return "json_parse_failure_no_retry"
    if record.get("failure_type") in ("OSError", "PermissionError", "FileNotFoundError"):
        return "filesystem_failure_no_retry"
    if record.get("failure_type") == "RuntimeIsolationError":
        return "runtime_failed_closed_no_retry"
    if record.get("failure_type") == "ContextCapacityError":
        return "context_capacity_no_retry"
    if record.get("failure_type") == "ModelCallBudgetExceeded":
        return "model_call_budget_reached"
    # A rejected research design can be revised using the role's rejection receipt.
    # Environment, isolation and transport failures above cannot be fixed that way.
    return None


def verify_codex_dependencies():
    """Check the Codex schema validator before starting any research attempts."""
    return importlib.import_module("jsonschema").Draft202012Validator


class ModelCallBudgetExceeded(RuntimeError):
    """The prospectively fixed request budget is exhausted, not a scientific failure."""


def context_capacity_attempt(attempt, stage, metadata):
    """Keep an oversized prompt visible as one stopped attempt, without storing its text."""
    return {"attempt": attempt, "status": "needs-attention", "analysis_status": "not_started",
            "decision_status": "not_started", "failure_type": "ContextCapacityError",
            "failure_domain": "context_capacity", "failure_stage": stage,
            "context": context_capacity_metadata(metadata)}


def context_capacity_metadata(metadata):
    return {key: metadata.get(key) for key in (
        "rendered_prompt_tokens", "requested_generation_tokens", "context_limit",
        "safety_margin_tokens", "required_tokens", "capacity_status",
        "prompt_chars", "prompt_bytes", "schema_bytes", "transport_input_char_limit",
        "completion_submitted", "failure_boundary")}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_fingerprint(path):
    path = Path(path)
    info = path.stat()
    # Client-local NFS device/inode identities need not match across compute hosts.
    return {"path": str(path.resolve()), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def pinned_model_identity(weights, binary, previous=None):
    fingerprint = weight_fingerprint(weights)
    if previous is not None:
        if previous.get("weights_fingerprint") != fingerprint:
            raise ValueError("Pinned model weights metadata changed; review required before resume")
        digest = previous["weights_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Missing valid initial model digest")
    else:
        digest = file_sha256(weights)
        if weight_fingerprint(weights) != fingerprint:
            raise ValueError("Weights changed while computing their initial digest")
    # Metadata-bound reuse is not a security boundary against malicious same-UID modification.
    libraries = {path.name: file_sha256(path) for path in sorted({p.resolve() for p in Path(binary).parent.glob("*.so*")})
                 if path.is_file()}
    if previous is not None and previous.get("runtime_library_sha256", libraries) != libraries:
        raise ValueError("Pinned inference runtime libraries changed before trajectory resume")
    return {"weights_sha256": digest, "weights_fingerprint": fingerprint,
            "server_binary_sha256": file_sha256(binary), "runtime_library_sha256": libraries}


def read_reference(reference, root):
    value, actual = _source(Path(root) / reference["path"], root)
    if actual != reference:
        raise ValueError("Resume source changed after cursor publication")
    return value


def decision_location(path):
    match = re.fullmatch(r"attempt-(\d+)/(?:resume-[A-Za-z0-9_-]+/)?decision.json", path)
    if match is None:
        raise ValueError("Decision resume source must be an own attempt decision")
    return int(match[1]), str(Path(path).with_name("attempt.json"))


def load_progress(trajectory):
    """Only own immutable scientific artifacts can resume context; no QA input channel."""
    trajectory = Path(trajectory)
    numbers = [int(match[1]) for path in trajectory.glob("attempt-*")
               if path.is_dir() and (match := re.fullmatch(r"attempt-(\d+)", path.name))]
    result = {"next_attempt": max(numbers, default=0) + 1, "previous_plan": None,
              "previous_graph": None, "next_inquiry": QUESTION, "cursor": {}}
    cursor_path = trajectory / "CURRENT.json"
    if not cursor_path.exists():
        return result
    cursor = json.loads(cursor_path.read_text())
    allowed = {"format", "round", "job_id", "last_plan", "last_attempt", "last_decision", "last_decision_attempt"}
    if set(cursor) != allowed or cursor["format"] != "memory-iteration-cursor-v1":
        raise ValueError("Unknown or expanded resume cursor")
    if type(cursor["round"]) is not int or not 1 <= cursor["round"] <= 5:
        raise ValueError("Invalid resume round")
    if cursor["last_plan"] is not None:
        plan = read_reference(cursor["last_plan"], trajectory)
        record = read_reference(cursor["last_attempt"], trajectory)
        attempt = record["attempt"]
        if record.get("hypothesis_id") != f"hypothesis-{attempt:04d}":
            raise ValueError("Cursor hypothesis identifier mismatch")
        if cursor["last_plan"]["path"] != f"attempt-{attempt:04d}/plan.json":
            raise ValueError("Cursor plan/attempt mismatch")
        if cursor["last_attempt"]["path"] != f"attempt-{attempt:04d}/attempt.json":
            raise ValueError("Cursor attempt source is not its own attempt record")
        result.update(previous_plan=plan, previous_graph={"hypothesis_id": record["hypothesis_id"]})
    if cursor["last_decision"] is not None:
        decision = read_reference(cursor["last_decision"], trajectory)
        decision_index, decision_record_path = decision_location(cursor["last_decision"]["path"])
        decision_record = read_reference(cursor["last_decision_attempt"], trajectory)
        if cursor["last_decision_attempt"]["path"] != decision_record_path:
            raise ValueError("Cursor decision/attempt mismatch")
        if decision_record.get("decision_status") != "succeeded":
            raise ValueError("Cannot resume a failed decision")
        if decision_record.get("attempt") != decision_index or decision_record.get("decision_id") != f"decision-{decision_index:04d}":
            raise ValueError("Cursor decision identifier mismatch")
        if result["previous_graph"] is None:
            raise ValueError("Decision resume requires a preceding hypothesis plan")
        result["previous_graph"]["decision_id"] = decision_record["decision_id"]
        result["next_inquiry"] = decision["next_inquiry"]
    result["cursor"] = cursor
    return result


def publish_cursor(trajectory, round_number, job_id, last_plan, last_attempt, last_decision, checkpoint_only=False):
    decision_attempt = None
    if last_decision:
        _, decision_record_path = decision_location(last_decision["path"])
        decision_attempt = source_reference(trajectory / decision_record_path, trajectory)
    cursor = {"format": "memory-iteration-cursor-v1", "round": round_number, "job_id": job_id,
              "last_plan": last_plan, "last_attempt": last_attempt, "last_decision": last_decision,
              "last_decision_attempt": decision_attempt}
    if not checkpoint_only:
        write(trajectory / f"round-{round_number:02d}-{job_id}.cursor.json", cursor)
    pending = trajectory / "CURRENT.pending.json"
    pending.write_text(json.dumps(cursor, indent=2))
    pending.replace(trajectory / "CURRENT.json")


def use_adaptive_shared_context(condition, matched_private_history, requested=False):
    if requested and (condition != "C" or matched_private_history):
        raise ValueError("Adaptive shared context is C-only and cannot use matched full-history policy")
    return condition == "C" and not matched_private_history


def main():
    if not os.environ.get("JOB_ID"):
        raise SystemExit("Reviewed finite SGE allocation required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=range(1, MAX_ROUNDS + 1),
                        required=not bool(BENCHMARK_CONDITIONS), default=1)
    parser.add_argument("--budget-seconds", type=int, default=3600 if BENCHMARK_CONDITIONS else 900)
    if BENCHMARK_CONDITIONS:
        parser.add_argument("--condition", choices=BENCHMARK_CONDITIONS, required=True)
        parser.add_argument("--repeat-one-shot", action="store_true")
    else:
        parser.add_argument("--arm", choices=("memory",), default="memory")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--backend", choices=("qwen", "codex"), default="qwen")
    parser.add_argument("--codex-model")
    parser.add_argument("--matched-private-history", action="store_true")
    parser.add_argument("--adaptive-shared-context", action="store_true")
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--retrieval-strategy", choices=("ppr", "dense", "similarity"), default="ppr")
    parser.add_argument("--embedding-backend", choices=("fastembed", "lexical"), default="fastembed")
    parser.add_argument("--runtime", type=Path, default=SERVER_RUNTIME)
    parser.add_argument("--cuda-toolkit", type=Path)
    parser.add_argument("--trajectory-root", type=reviewed_trajectory, required=True)
    args = parser.parse_args()
    # New C runs use bounded active context; explicit historical full-history runs remain reproducible.
    args.adaptive_shared_context = use_adaptive_shared_context(
        getattr(args, "condition", "C"), args.matched_private_history, args.adaptive_shared_context)
    import four_role_discovery
    four_role_discovery.ADAPTIVE_SHARED_CONTEXT = args.adaptive_shared_context
    if args.matched_private_history:
        import four_role_discovery
        four_role_discovery.COMPACT_SHARED_HISTORY = False
    if getattr(args, "repeat_one_shot", False) and args.condition != "A":
        raise SystemExit("Repeated one-shot is only valid for condition A")
    if args.backend == "codex" and (not args.codex_model or args.device != "cpu"):
        raise SystemExit("Codex requires an explicit model and CPU Scheduler allocation")
    if args.backend == "codex":
        verify_codex_dependencies()
    if args.max_model_calls is not None and args.max_model_calls < 1:
        raise SystemExit("Model call budget must be positive")
    budget_cap = 3600 if BENCHMARK_CONDITIONS else 1800
    if not 120 <= args.budget_seconds <= budget_cap:
        raise SystemExit("Reviewed finite allocation cap is %s seconds" % budget_cap)
    if not args.runtime.is_absolute() or (args.cuda_toolkit is not None and not args.cuda_toolkit.is_absolute()):
        raise SystemExit("Use exact reviewed absolute runtime/toolkit paths")
    if args.device == "gpu" and args.cuda_toolkit is None:
        raise SystemExit("GPU mode requires the reviewed private CUDA toolkit")
    started = time.monotonic()
    os.umask(0o077)
    for fd in (1, 2):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.fchmod(fd, 0o600)
    trajectory = args.trajectory_root
    trajectory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One scheduler-owned writer per trajectory; no Login controller or background service.
    with (trajectory / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_round(args, trajectory, started)


def run_round(args, trajectory, started):
    job_id = os.environ["JOB_ID"]
    benchmark = bool(BENCHMARK_CONDITIONS)
    condition = args.condition if benchmark else None
    arm = ("memory" if condition == "C" else "no-memory") if benchmark else args.arm
    runroot = ITERATIONS / condition / job_id if benchmark else ITERATIONS / f"round-{args.round:02d}" / job_id
    runroot.mkdir(parents=True, mode=0o700)
    os.environ.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    progress = load_progress(trajectory) if arm == "memory" else {
        "next_attempt": 1, "previous_plan": None, "previous_graph": None,
        "next_inquiry": QUESTION, "cursor": {}}
    if progress["cursor"] and progress["cursor"]["round"] >= args.round:
        raise ValueError("Round must follow the trajectory's previously published round")
    config_path = trajectory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    if config is not None and not isinstance(config.get("model_identity"), dict):
        raise ValueError("Existing trajectory lacks reviewed model identity")
    if benchmark and config is not None:
        raise ValueError("Benchmark conditions are fresh episodes; use a new trajectory namespace, never resume")
    data_identity = {"sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()}
    pin_started = time.monotonic()
    backend = getattr(args, "backend", "qwen")
    if backend == "codex":
        model_identity = {"provider": "codex-account", "model": args.codex_model,
                          "server_binary_sha256": file_sha256(shutil.which("codex")),
                          "reasoning_effort": "medium", "seed_supported": False}
        runtime_contract = {"backend": "codex", "device": "hosted", "model": args.codex_model,
                            "session_policy": "fresh_ephemeral_harness_supplied_context"}
    else:
        model_identity = pinned_model_identity(WEIGHT, args.runtime / "llama-server",
                                              config.get("model_identity") if config else None)
        runtime_contract = {"device": args.device, "runtime": str(args.runtime),
                            "cuda_toolkit": str(args.cuda_toolkit) if args.cuda_toolkit else None}
    pin_seconds = round(time.monotonic() - pin_started, 3)
    if config is not None:
        if (config.get("question") != QUESTION or config.get("data_identity") != data_identity
                or config.get("model_identity") != model_identity):
            raise ValueError("Cumulative trajectory task/data/model changed; choose a fresh reviewed trajectory")
        if config.get("memory_format", "json") != MEMORY_FORMAT:
            raise ValueError("Context representation changed; use a fresh reviewed trajectory")
        if (config.get("runtime_contract", runtime_contract) != runtime_contract or
                config.get("retrieval_strategy", "ppr") != args.retrieval_strategy or
                config.get("embedding_backend", "fastembed") != args.embedding_backend):
            raise ValueError("Reviewed runtime contract changed across trajectory rounds")
    else:
        write(config_path, {"question": QUESTION, "data_identity": data_identity, "model_identity": model_identity,
            "memory_format": MEMORY_FORMAT,
            "runtime_contract": runtime_contract,
            "role_history_policy": ("fresh_episode_only" if getattr(args, "repeat_one_shot", False) else
                                    "bounded_c_checkpoint_with_private_recall" if args.adaptive_shared_context else
                                    "latest_output_per_stage_with_full_archive" if condition == "C" and not getattr(args, "matched_private_history", False) else
                                    "full_role_local_history"),
            "retrieval_strategy": args.retrieval_strategy if arm == "memory" else "disabled",
            "embedding_backend": args.embedding_backend if arm == "memory" else "disabled",
            "condition": condition,
            "context_policy": ("independent-fresh-one-shot-episodes" if getattr(args, "repeat_one_shot", False) else
                               "role-local-history-plus-shared-graph-retrieval" if condition == "C" else
                               "role-local-history-and-handoffs-only" if condition == "B" else
                               "one-cycle-role-local-context" if condition == "A" else
                               "own scientific graph only; no evaluator or QA feedback"),
            "created_by_job": job_id, "created_round": args.round})
    report = {"job_id": job_id, "phase": "three-condition-benchmark" if benchmark else "memory-iteration",
        "arm": arm, "condition": condition, "round": args.round,
        "memory_format": MEMORY_FORMAT, "retrieval_strategy": args.retrieval_strategy if arm == "memory" else "disabled",
        "embedding_backend": args.embedding_backend if arm == "memory" else "disabled",
        "role_history_policy": ("fresh_episode_only" if getattr(args, "repeat_one_shot", False) else
                                "bounded_c_checkpoint_with_private_recall" if args.adaptive_shared_context else
                                "latest_output_per_stage_with_full_archive" if condition == "C" and not getattr(args, "matched_private_history", False) else "full_role_local_history"),
        "status": "running", "budget_seconds": args.budget_seconds, "allocated_cpus": int(os.environ.get("NSLOTS", "4")),
        "backend": backend, "max_model_calls": getattr(args, "max_model_calls", None),
        "inference_deadline_reserve_seconds": INFERENCE_DEADLINE_RESERVE_SECONDS,
        "inference_budget_seconds": args.budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS,
        "gpu": args.device == "gpu", "runtime_contract": runtime_contract,
        "question": QUESTION, "attempts": [], "model_calls": [],
        "active_attempt": None, "active_stage": "initialization", "trajectory_root": str(trajectory),
        "resumed_from": progress["cursor"], "topology": TOPOLOGY,
        "cross_attempt_context_policy": ("independent-fresh-one-shot-episodes" if getattr(args, "repeat_one_shot", False) else
            "single-cycle" if condition == "A" else
            "role-local-history-and-handoffs-only" if condition == "B" else
            "role-local-history-plus-shared-graph-retrieval" if condition == "C" else
            "own scientific graph only; no evaluator or QA feedback"),
        "endpoint_scores": "independent QA only; never fed back", "source_sha256": {},
        "data_identity": data_identity, "model_identity": model_identity,
        "weight_digest_mode": ("hosted-model-unpinned-cli-sha256" if backend == "codex" else
                               "metadata-bound-reuse" if config else "initial-streamed-sha256"),
        "model_pin_seconds": pin_seconds,
        "weight_digest_limit": "metadata reuse is not an adversarial same-UID security boundary"}
    source_dir = runroot / "source"
    source_dir.mkdir(mode=0o700)
    sources = (*SOURCE_FILES, "tools/codex_role_backend.py") if backend == "codex" else SOURCE_FILES
    for location in sources:
        source = ROOT / location
        shutil.copyfile(source, source_dir / source.name)
        report["source_sha256"][location] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["server_binary_sha256"] = model_identity["server_binary_sha256"]
    write(runroot / "config.json", report)
    checkpoint(runroot, report, started)
    memory = (TwoRoleMemory(trajectory, arm="memory", embedding_backend=args.embedding_backend,
                            retrieval_strategy=args.retrieval_strategy)
              if arm == "memory" else None)
    previous_plan, previous_graph = progress["previous_plan"], progress["previous_graph"]
    next_inquiry = progress["next_inquiry"]
    cursor = progress["cursor"]
    last_plan, last_attempt, last_decision = (cursor.get(key) for key in ("last_plan", "last_attempt", "last_decision"))
    worker = None
    index = progress["next_attempt"]

    stage_counts = {}

    def model(label, prompt, schema, tokens, seed):
        nonlocal worker
        limit = getattr(args, "max_model_calls", None)
        if limit is not None and len(report["model_calls"]) >= limit:
            raise ModelCallBudgetExceeded("Prospective model request budget exhausted")
        report["active_stage"] = label
        checkpoint(runroot, report, started)
        print("MEMORY_ITERATION_STAGE", encode({"round": args.round, "attempt": index,
              "stage": label, "event": "started"}), flush=True)
        receipt_label = stage_receipt_label(stage_counts, label)
        try:
            if worker is None:
                if backend == "codex":
                    from codex_role_backend import CodexRoleBackend
                    worker = CodexRoleBackend(runroot / "inference",
                        started + args.budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS,
                        model=args.codex_model)
                else:
                    worker = private_worker(runroot / "inference", started, args.budget_seconds,
                        runtime=args.runtime, device=args.device, cuda_toolkit=args.cuda_toolkit)
                worker.__enter__()
                report["runtime_startup_seconds"] = worker.startup_seconds
                report["device_report"] = worker.device_report
            request_prompt = fixed_model_prefix() + "\n" + prompt if backend == "codex" else prompt
            value, metadata, content = worker.complete(request_prompt, schema, tokens, seed)
        except Exception as error:
            (session / (receipt_label + ".stdout")).write_text(getattr(worker, "last_content", ""))
            metadata = dict(getattr(worker, "last_metadata", {}) or {})
            if isinstance(error, RuntimeIsolationError):
                failure = error.__cause__ or error
                metadata.setdefault("failure_boundary", "runtime_startup"
                                    if report.get("runtime_startup_seconds") is None else "preflight")
                metadata.setdefault("failure_cause", type(failure).__name__)
                metadata.setdefault("deadline_exhausted", getattr(error, "deadline_exhausted", False) is True)
                error.deadline_exhausted = metadata["deadline_exhausted"] is True
            write(session / (receipt_label + ".inference.json"), metadata)
            report["model_calls"].append({**metadata, "attempt": index, "stage": label,
                "parse_status": ("rejected" if metadata.get("completion_submitted") else "not_generated")
                    if isinstance(error, ContextCapacityError) else "failed",
                "failure_type": type(error).__name__})
            raise
        (session / (receipt_label + ".stdout")).write_text(content)
        write(session / (receipt_label + ".inference.json"), metadata)
        report["model_calls"].append({**metadata, "attempt": index, "stage": label, "parse_status": "succeeded"})
        return value

    if backend == "codex":
        from codex_role_backend import CODEX_INPUT_CHARACTER_LIMIT, ACTION_TRANSPORT_INSTRUCTION
        model.input_character_limit = (CODEX_INPUT_CHARACTER_LIMIT - len(fixed_model_prefix()) - 1
                                       - len(ACTION_TRANSPORT_INSTRUCTION))

    from analysis_reuse import VerifiedAnalysisCache
    analysis_cache = VerifiedAnalysisCache(data_identity)

    def analysis_runner(session, action):
        report["active_stage"] = "analysis"
        checkpoint(runroot, report, started)
        result = (run_analysis(session, action) if getattr(args, "repeat_one_shot", False)
                  else analysis_cache.run(session, action, run_analysis))
        field = "reused_analysis_receipts" if result.get("execution_reuse") else "executed_analysis_calls"
        report[field] = report.get(field, 0) + 1
        return result

    try:
        resume_finished = True
        if RESUME_RESULT is not None and arm == "memory" and not benchmark:
            from four_role_resume import pending_result
            original = pending_result(trajectory, cursor)
            if original is not None:
                session = original / ("resume-" + job_id)
                session.mkdir(mode=0o700)
                index = int(original.name.split("-")[1])
                report["active_attempt"] = {"attempt": index, "status": "resuming_post_result",
                                            "artifact_directory": str(session.relative_to(trajectory))}
                checkpoint(runroot, report, started)
                begin = time.monotonic()
                record, plan, decision = RESUME_RESULT(session, original, memory, model,
                    data_identity=data_identity, research_question=QUESTION)
                record.update(round=args.round, job_id=job_id, artifact_directory=str(session.relative_to(trajectory)), seconds=round(time.monotonic()-begin, 3),
                              elapsed_end=round(time.monotonic()-started, 3))
                write(session / "attempt.json", record)
                report["resumptions"] = [record]
                resume_finished = record.get("decision_status") == "succeeded"
                if resume_finished:
                    last_decision = source_reference(session / "decision.json", trajectory)
                    next_inquiry = decision["next_inquiry"]
                    previous_graph["decision_id"] = record["decision_id"]
                else:
                    report.update(status="window-ended" if record["status"] == "censored" else
                                  "needs-attention" if record["status"] == "needs-attention" else "failed",
                                  stop_reason="post_result_resume_context_capacity" if
                                  record["status"] == "needs-attention" and
                                  record.get("failure_type") == "ContextCapacityError" else
                                  "post_result_resume_incomplete")
                publish_cursor(trajectory, args.round, job_id, last_plan, last_attempt, last_decision, checkpoint_only=True)
                report["active_attempt"] = None
                checkpoint(runroot, report, started)
                print("MEMORY_ITERATION_RESUME", encode({key: record.get(key) for key in
                    ("attempt", "status", "analysis_status", "decision_status", "seconds", "runner_reexecuted")}), flush=True)
                index = progress["next_attempt"]
        while resume_finished and may_start_attempt(condition, index, time.monotonic() - started, args.budget_seconds,
                                                    getattr(args, "repeat_one_shot", False)):
            repeated_one_shot = getattr(args, "repeat_one_shot", False)
            if repeated_one_shot:
                session = one_shot_episode(trajectory, index)
            else:
                session = trajectory / f"attempt-{index:04d}"
                session.mkdir(mode=0o700)
            report["active_attempt"] = {"attempt": index, "status": "started",
                                        "execution_attempt": 1 if repeated_one_shot else index,
                                        "artifact_directory": str(session.relative_to(trajectory))}
            checkpoint(runroot, report, started)
            begin = time.monotonic()
            call_kwargs = {"query": next_inquiry if benchmark else (next_inquiry if arm == "memory" else QUESTION),
                "previous_plan": previous_plan if arm == "memory" else None,
                "data_identity": data_identity, "previous_graph": previous_graph if arm == "memory" else None,
                "memory_format": MEMORY_FORMAT}
            if TOPOLOGY == "orchestrator-hypothesis-empiricist-runner":
                from agent_r_diagnostic import run_agent_r
                def diagnostic_runner(folder, code):
                    report["active_stage"] = "diagnostic"
                    checkpoint(runroot, report, started)
                    return run_agent_r(folder, code, data=DATA,
                        cpu_seconds=max(1, int(started + args.budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS - time.monotonic())))
                call_kwargs["diagnostic_runner"] = diagnostic_runner
            if benchmark:
                call_kwargs["condition"] = condition
                call_kwargs["research_question"] = QUESTION
            record, plan, decision = execute_attempt(session, 1 if repeated_one_shot else index, arm, memory, model, analysis_runner,
                                                     **call_kwargs)
            if repeated_one_shot:
                record["episode"] = index
            if record.get("failure_type") == "ContextCapacityError":
                call = next((item for item in reversed(report["model_calls"])
                             if item.get("attempt") == index and item.get("failure_type") == "ContextCapacityError"), {})
                record["context"] = context_capacity_metadata(call)
            record.update(round=args.round, job_id=job_id, artifact_directory=str(session.relative_to(trajectory)), seconds=round(time.monotonic()-begin, 3),
                          elapsed_end=round(time.monotonic()-started, 3))
            write(session / "attempt.json", record)
            report["attempts"].append(record)
            if arm == "memory" and plan and record.get("hypothesis_id") and (session / "plan.json").exists():
                previous_plan, previous_graph = plan, dict(record)
                last_plan = source_reference(session / "plan.json", trajectory)
                last_attempt = source_reference(session / "attempt.json", trajectory)
            if record.get("decision_status") == "succeeded" and not repeated_one_shot:
                last_decision = source_reference(session / "decision.json", trajectory)
                next_inquiry = decision["next_inquiry"]
            if previous_graph is not None and last_decision is not None:
                decision_index, _ = decision_location(last_decision["path"])
                previous_graph["decision_id"] = f"decision-{decision_index:04d}"
            # Preserve resumable own evidence before the next attempt or Scheduler hard limit.
            if arm == "memory":
                publish_cursor(trajectory, args.round, job_id, last_plan, last_attempt, last_decision, checkpoint_only=True)
            report["active_attempt"] = None
            report["active_stage"] = "between_attempts"
            checkpoint(runroot, report, started)
            print("MEMORY_ITERATION_ATTEMPT", encode({key: record.get(key) for key in
                ("round", "attempt", "status", "analysis_status", "decision_status", "seconds")}), flush=True)
            stop_reason = fatal_attempt_stop_reason(record)
            if stop_reason:
                report.update(status="window-ended" if stop_reason == "model_call_budget_reached" else
                              "needs-attention" if stop_reason == "context_capacity_no_retry" else "failed",
                              stop_reason=stop_reason)
                resume_finished = False
                break
            index += 1
            if condition == "A" and not repeated_one_shot:
                break
        if resume_finished:
            report["status"] = "one-cycle-ended" if condition == "A" and not getattr(args, "repeat_one_shot", False) else "window-ended"
    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__, diagnostic=str(error)[:300])
        report["failure_cause"] = type(error.__cause__).__name__ if error.__cause__ else None
        if isinstance(error, ContextCapacityError):
            failed = context_capacity_attempt(index, report.get("active_stage"), worker.last_metadata if worker else {})
            write(session / "attempt.json", failed)
            report["attempts"].append(failed)
            report.update(status="needs-attention", stop_reason="context_capacity_no_evidence_removed",
                          active_attempt=None, active_stage="context_capacity")
        elif (isinstance(error, RuntimeIsolationError) and isinstance(error.__cause__, TimeoutError)
              and time.monotonic() >= started + args.budget_seconds - INFERENCE_DEADLINE_RESERVE_SECONDS):
            report.update(status="window-ended", stop_reason="inference_deadline_partial_attempt_preserved")
    finally:
        if worker is not None:
            worker.__exit__(None, None, None)
            report["runtime_cleanup"] = worker.cleanup_metadata
            report["runtime_cleanup_verified"] = (worker.cleanup_metadata.get("verified", False)
                if backend == "codex" else worker.process is not None and worker.process.poll() == 0)
            if not report["runtime_cleanup_verified"]:
                report.update(status="failed", failure_type="RuntimeCleanupUnverified")
        if arm == "memory":
            publish_cursor(trajectory, args.round, job_id, last_plan, last_attempt, last_decision)
        checkpoint(runroot, report, started)
    print("MEMORY_ITERATION_WINDOW", encode({"round": args.round, "condition": condition, "job_id": job_id,
        "status": report["status"], "attempts": len(report["attempts"]), "root": str(runroot),
        "trajectory_root": str(trajectory), "endpoint_scores": "not yet scored"}), flush=True)


if __name__ == "__main__":
    main()
