"""CLI for scheduler-produced memory and dashboard artifacts."""

import argparse
import html
import json
import os
from pathlib import Path
import socket
import time

from .memory import build_snapshot, retrieve, submit_candidate
from .viewer import render_dashboard
from . import workflow as flow
from .execution import require_execution


def dashboard(workspace, trace):
    root = Path(workspace)
    tasks = [flow.read(path) for path in sorted((root / "tasks").glob("*.json"))]
    trials = [flow.read(path) for path in sorted((root / "trials").glob("*.json"))]
    return render_dashboard(trace, tasks, trials)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared research memory; submit computation via the Job Scheduler.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--local", action="store_true", help="Explicit non-cluster workstation execution only")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("file", type=Path)
    index = commands.add_parser("index")
    index.add_argument("--backend", choices=("lexical", "fastembed", "external"), default="fastembed")
    index.add_argument("--model")
    query = commands.add_parser("retrieve")
    query.add_argument("--query", required=True)
    query.add_argument("--track", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--snapshot")
    query.add_argument("--task")
    query.add_argument("--strategy", choices=("ppr", "dense"), default="ppr")
    task = commands.add_parser("task")
    task.add_argument("--query", required=True)
    task.add_argument("--track", required=True)
    task.add_argument("--agent", default="codex")
    attempt = commands.add_parser("attempt")
    attempt.add_argument("--task", required=True)
    attempt.add_argument("--record", type=Path, required=True)
    attempt.add_argument("--used", nargs="*", default=[])
    attempt.add_argument("--adaptation", choices=("reuse", "avoid", "revise", "test", "no-new-learning"), required=True)
    attempt.add_argument("--source", required=True)
    decision = commands.add_parser("decision")
    decision.add_argument("--task", required=True)
    decision.add_argument("--text", required=True)
    complete = commands.add_parser("complete")
    complete.add_argument("--task", required=True)
    trial = commands.add_parser("trial")
    trial.add_argument("--task", required=True)
    trial.add_argument("--condition", required=True)
    trial.add_argument("--targets", nargs="+", required=True)
    event = commands.add_parser("trial-event")
    event.add_argument("--trial", required=True)
    event.add_argument("--file", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--request", required=True)
    agent = commands.add_parser("agent-run")
    agent.add_argument("--task", required=True)
    agent.add_argument("--timeout", type=int, default=900)
    agent.add_argument("--inputs", type=Path, help="Owner-reviewed data-first input manifest")
    serve = commands.add_parser("serve", help="Bounded loopback live UI inside a reviewed SGE allocation")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--duration", type=int, default=3600, help="Session lifetime in seconds")
    serve.add_argument("--timeout", type=int, default=900, help="Maximum duration of one attempt")
    serve.add_argument("--allow-child-cleanup", action="store_true", help="Use only after owner approval for timeout cleanup of this attempt's own child processes")
    serve.add_argument("--inputs", type=Path, help="Owner-reviewed input manifest, fixed for this session")
    serve.add_argument("--experiment-report", type=Path,
                       help="Exact four-role Scheduler report.json to monitor read-only")
    serve.add_argument("--experiment-trajectory", type=Path,
                       help="Exact four-role trajectory directory paired with --experiment-report")
    watch = commands.add_parser("watch", help="Observe an existing research run and record separate human reviews")
    watch.add_argument("--report", type=Path, required=True, help="Exact research report.json")
    watch.add_argument("--trajectory", type=Path, required=True, help="Exact trajectory_root named in that report")
    watch.add_argument("--port", type=int, default=8765)
    watch.add_argument("--duration", type=int, default=3600, help="Finite viewer lifetime in seconds")
    audit = commands.add_parser("audit-reference")
    audit.add_argument("--trial", required=True)
    audit.add_argument("--actual", type=Path, required=True)
    audit.add_argument("--expected", type=Path, required=True)
    audit.add_argument("--rtol", type=float, default=1e-5)
    audit.add_argument("--atol", type=float, default=1e-8)
    demo = commands.add_parser("demo")
    demo.add_argument("--backend", choices=("lexical", "fastembed"), default="fastembed")
    args = parser.parse_args(argv)
    try:
        require_execution(local=args.local)
    except RuntimeError as error:
        parser.error(str(error))
    root = args.workspace.resolve()
    if args.command == "ingest":
        records = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
        output = {"candidates": [str(submit_candidate(root, record)) for record in records]}
    elif args.command == "index":
        output = build_snapshot(root, args.backend, args.model)
    elif args.command == "retrieve":
        trace = retrieve(root, args.query, args.track, args.top_k, args.snapshot, strategy=args.strategy)
        folder = flow.save_retrieval(root, trace)
        if args.task:
            flow.attach_retrieval(root, args.task, trace["request_id"])
        (folder / "dashboard.html").write_text(dashboard(root, trace), encoding="utf-8")
        output = {"request_id": trace["request_id"], "snapshot_id": trace["snapshot_id"], "outcome": trace["outcome"], "dashboard": str(folder / "dashboard.html")}
    elif args.command == "task":
        output = flow.create_task(root, args.track, args.query, args.agent)
    elif args.command == "attempt":
        output = flow.record_attempt(root, args.task, flow.read(args.record), args.used, args.adaptation, args.source)
    elif args.command == "decision":
        output = flow.record_decision(root, args.task, args.text)
    elif args.command == "complete":
        output = flow.complete_task(root, args.task)
    elif args.command == "trial":
        output = flow.create_trial(root, args.task, args.condition, args.targets)
    elif args.command == "trial-event":
        output = flow.record_trial_event(root, args.trial, flow.read(args.file))
    elif args.command == "export":
        trace = flow.read(root / "retrievals" / flow.identifier(args.request) / "trace.json")
        folder = root / "retrievals" / args.request
        (folder / "dashboard.html").write_text(dashboard(root, trace), encoding="utf-8")
        output = {"dashboard": str(folder / "dashboard.html")}
    elif args.command == "agent-run":
        from .agent import run_codex_attempt
        output = run_codex_attempt(root, args.task, args.timeout, input_manifest=args.inputs)
    elif args.command == "watch":
        from .live import ArtifactController, open_artifact_server
        from .live_viewer import render_live_dashboard
        if not 1 <= args.port <= 65535 or args.duration <= 0:
            parser.error("Use a valid TCP port and positive finite viewer duration")
        os.umask(0o077)
        server, session = open_artifact_server(root, args.report, args.trajectory, args.port, local=args.local)
        print("Live observer ready. Private connection file:", session, flush=True)
        print("The viewer follows recorded events; it does not launch agents. Use an approved tunnel on a cluster.", flush=True)
        server.timeout = .5
        try:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                server.handle_request()
        finally:
            server.server_close()
            snapshot = root / "dashboard-snapshot.html"
            snapshot.write_text(render_live_dashboard("", ArtifactController(args.report, args.trajectory).state(), offline=True), encoding="utf-8")
        output = {"status": "observer-ended", "snapshot": str(snapshot), "connection_file": str(session)}
    elif args.command == "serve":
        from .live import LiveController, create_server
        if not os.environ.get("JOB_ID"):
            parser.error("Live cluster sessions require a reviewed SGE allocation")
        if not 1 <= args.port <= 65535 or not 1 <= args.duration <= 14400:
            parser.error("Port must be 1..65535 and session lifetime 1..14400 seconds")
        os.umask(0o077)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = root / ".live-controller-lock"
        try:
            lock.mkdir()
        except FileExistsError:
            parser.error("A controller already owns this workspace; review a stale lock manually")
        server = None
        controller = None
        try:
            if args.allow_child_cleanup:
                os.environ["RESEARCH_AGENT_ALLOW_CHILD_CLEANUP"] = "1"
            controller = LiveController(root, timeout_seconds=args.timeout, input_manifest=args.inputs,
                                        experiment_report=args.experiment_report,
                                        experiment_trajectory=args.experiment_trajectory)
            server = create_server(controller, args.port)
            session = root / "sessions" / ("job-" + flow.identifier(os.environ["JOB_ID"]) + ".json")
            session.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(session, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"host": socket.gethostname(), "port": args.port, "job_id": os.environ["JOB_ID"],
                           "browser_url": f"http://127.0.0.1:{args.port}/#token={server.token}",
                           "created_at": flow.now(), "duration_seconds": args.duration,
                           "attempt_timeout_seconds": args.timeout}, stream, ensure_ascii=False, indent=2)
            launcher = session.parent / "open-dashboard.html"
            document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Research Dashboard</title>'
                        '<body style="font:18px system-ui;padding:40px"><h1>Research Dashboard</h1>'
                        '<p>Open the dashboard after connecting your approved SSH tunnel.</p><p><a href="' +
                        html.escape(f"http://127.0.0.1:{args.port}/#token={server.token}", quote=True) +
                        '">Open dashboard</a></p><p>Private connection file. Do not share it.</p></body></html>')
            launcher_fd = os.open(launcher, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(launcher_fd, "w", encoding="utf-8") as stream:
                stream.write(document)
            print("Live session ready. Private connection file:", session, flush=True)
            print("Use the university-approved SSH/tunnel route; connection token is never printed.", flush=True)
            deadline = time.monotonic() + args.duration
            server.timeout = .5
            while time.monotonic() < deadline:
                server.handle_request()
            # A session expiry stops accepting new requests; its existing bounded attempt may finish.
            finished = controller.wait(timeout=args.timeout + 5)
            output = {"status": "session-expired" if finished else "session-expired-worker-finishing",
                      "connection_file": str(session), "job_id": os.environ["JOB_ID"]}
            if not finished:
                flow.write(root / "inbox" / ("session-" + os.environ["JOB_ID"] + ".json"),
                           {"type": "ESCALATE", "summary": "ESCALATE SYSTEM · shared-memory: session expired with a worker still finishing; preserve ownership lock and review before restart.",
                            "job_id": os.environ["JOB_ID"], "created_at": flow.now()})
        finally:
            if server:
                server.server_close()
            if not controller or controller.wait(timeout=0):
                lock.rmdir()
    elif args.command == "audit-reference":
        from .reference import compare_reference
        checks = compare_reference(flow.read(args.actual), flow.read(args.expected), args.rtol, args.atol)
        output = flow.record_trial_event(root, args.trial, {"kind": "reference_check", "checks": checks,
                  "source": str(args.actual.resolve()), "reference_source": str(args.expected.resolve()),
                  "relative_tolerance": args.rtol, "absolute_tolerance": args.atol})
    else:
        fixture = Path(__file__).with_name("assets") / "synthetic-memories.json"
        # shortcut: synthetic demo starts in a fresh workspace; real data is ingested explicitly.
        if (root / "CURRENT").exists() or (root / "candidates").exists():
            raise ValueError("Use a fresh workspace for the synthetic demo")
        for record in json.loads(fixture.read_text()):
            submit_candidate(root, record)
        manifest = build_snapshot(root, args.backend)
        task = flow.create_task(root, "FD", "firm differentiation", "agent-b")
        trace = retrieve(root, task["question"], "FD", top_k=3)
        trace["synthetic"] = True
        folder = flow.save_retrieval(root, trace)
        flow.attach_retrieval(root, task["id"], trace["request_id"])
        candidate = {"memory_id": "fd-followup", "agent_id": "agent-b", "track_id": "FD",
                     "memory_type": "next_step", "text": "Synthetic example: revise the design using the peer's prior selection failure.",
                     "source": "synthetic demo outcome (not an empirical finding)", "aliases": [],
                     "edges": [{"target": "fd-failure", "relation": "DERIVED_FROM", "source": "synthetic demo outcome"}]}
        flow.record_attempt(root, task["id"], candidate, trace["selected_ids"], "revise", candidate["source"])
        flow.complete_task(root, task["id"])
        (folder / "dashboard.html").write_text(dashboard(root, trace), encoding="utf-8")
        output = {"snapshot_id": manifest["snapshot_id"], "task_id": task["id"], "request_id": trace["request_id"],
                  "algorithm": trace["algorithm"], "dashboard": str(folder / "dashboard.html"), "synthetic": True}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    main()
