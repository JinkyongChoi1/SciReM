"""Two isolated finite CPU model processes and real synthetic shared-memory retrieval."""

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WEIGHT = Path(os.environ.get("SCIREM_MODEL_WEIGHTS", ROOT / ".runtime/model.gguf")).expanduser().resolve()


def parse_object(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text).strip()
    text = re.sub(r"\s*\[end of text\]\s*\Z", "", text)
    if not text.startswith("{"):
        raise ValueError("Model did not return an unambiguous JSON object")
    value, end = json.JSONDecoder().raw_decode(text)
    if not isinstance(value, dict) or text[end:].strip():
        raise ValueError("Unexpected extra model output")
    return value


def validate_tool(value):
    if set(value) != {"tool", "values"} or value["tool"] != "sum_values":
        raise ValueError("Tool not allowlisted")
    values = value["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= 20:
        raise ValueError("Invalid tool arguments")
    if any(type(item) is not int or abs(item) > 1000 for item in values):
        raise ValueError("Only bounded integer arguments are permitted")
    return values


def sandbox(session, runtime=None, weights=False):
    args = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--unshare-pid", "--proc", "/proc", "--dev", "/dev", "--unshare-net",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--tmpfs", "/tmp",
            "--bind", str(session), "/work", "--chdir", "/work"]
    if runtime:
        args += ["--ro-bind", str(runtime), "/runtime"]
    if weights:
        args += ["--ro-bind", str(WEIGHT), "/model.gguf"]
    # setuid bwrap strips inherited LD_LIBRARY_PATH; set it inside the namespace instead.
    return args + ["--setenv", "LD_LIBRARY_PATH", "/runtime", "--"]


def main():
    if not os.environ.get("JOB_ID"):
        raise SystemExit("Use the reviewed CPU SGE allocation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--backend", choices=("lexical", "fastembed"), default="fastembed")
    args = parser.parse_args()
    os.umask(0o077)
    for descriptor in (1, 2):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.fchmod(descriptor, 0o600)
    directory = ROOT / ".runtime/local-qwen36/runs" / ("synthetic-" + os.environ["JOB_ID"])
    directory.mkdir(mode=0o700)
    sessions = [directory / name for name in ("agent-a", "agent-b")]
    for session in sessions:
        session.mkdir(mode=0o700)
    report = {"job_id": os.environ["JOB_ID"], "status": "failed", "device": "cpu",
              "gpu_inference": False, "cloud_inference": False, "synthetic_only": True,
              "checks": {}, "backend": args.backend, "model_calls": []}
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/work",
                   "LD_LIBRARY_PATH": "/runtime", "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1")

    def model_call(session, prompt, schema):
        command = sandbox(session, args.runtime, weights=True) + [
            "/runtime/llama-completion", "-m", "/model.gguf", "-ngl", "0", "-t", "4", "-tb", "4",
            "-c", "2048", "-n", "128", "--temp", "0", "--seed", "42", "--no-warmup",
            "-cnv", "--single-turn", "--reasoning", "off", "--no-display-prompt", "--simple-io",
            "-j", json.dumps(schema), "-p", prompt]
        started = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, env=environment, stdin=subprocess.DEVNULL)
        (session / "model.stdout").write_text(proc.stdout)
        (session / "model.stderr").write_text(proc.stderr)
        report["model_calls"].append({"session": session.name, "returncode": proc.returncode,
                                      "seconds": round(time.monotonic() - started, 2)})
        if proc.returncode:
            raise RuntimeError("Local model process failed")
        return parse_object(proc.stdout)

    try:
        probe = sandbox(sessions[0]) + ["/bin/sh", "-c",
            'test ! -e /user && test ! -e /proc/1/root/user && test ! -e "$1" && test ! -e /dev/nvidia0',
            "probe", str(sessions[1])]
        proc = subprocess.run(probe, env=environment, capture_output=True)
        if proc.returncode:
            raise AssertionError("Session/reference isolation failed")
        report["checks"]["sandbox_denies_sibling_reference_and_gpu"] = True
        schema = {"type": "object", "properties": {"tool": {"const": "sum_values"},
                  "values": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 20}},
                  "required": ["tool", "values"], "additionalProperties": False}
        request = model_call(sessions[0], "Use the sum_values tool to add these integers: 2, 4, 6. "
                             "Return only one JSON object with tool and values; do not compute the answer yourself.", schema)
        values = validate_tool(request)
        if values != [2, 4, 6]:
            raise AssertionError("Model changed the tool inputs")
        (sessions[0] / "tool-request.json").write_text(json.dumps(request))
        # The model selects bounded arguments, never executable code or arbitrary paths.
        tool = sandbox(sessions[0]) + ["/usr/bin/python3", "-c",
            "import json; r=json.load(open('/work/tool-request.json')); print(json.dumps({'total':sum(r['values'])}))"]
        proc = subprocess.run(tool, capture_output=True, text=True, env=environment)
        result = json.loads(proc.stdout)
        if proc.returncode or result != {"total": 12}:
            raise AssertionError("Synthetic tool execution failed")
        (sessions[0] / "tool-result.json").write_text(json.dumps(result))
        report["checks"]["model_requested_and_tool_computed_sum"] = True
        print("SYNTHETIC_STAGE tool-passed", flush=True)
        from research_memory.memory import build_snapshot, retrieve, submit_candidate
        memory = directory / "arm-ppr/replicate-1/memory"
        source = str(sessions[0] / "tool-result.json")
        submit_candidate(memory, {"memory_id": "synthetic-procedure", "agent_id": "agent-a", "track_id": "SYN",
            "text": "Synthetic arithmetic procedure: use the sum_values allowlisted integer tool; this is not empirical evidence.",
            "source": source, "aliases": ["synthetic arithmetic procedure"]})
        submit_candidate(memory, {"memory_id": "synthetic-total", "agent_id": "agent-a", "track_id": "SYN",
            "text": "The synthetic batch total is 12, measured by sum_values on [2,4,6]. This is synthetic tool output, not a research finding.",
            "source": source, "aliases": ["synthetic batch total"],
            "edges": [{"target": "synthetic-procedure", "relation": "computed_using", "source": source}]})
        manifest = build_snapshot(memory, args.backend)
        trace = retrieve(memory, "synthetic batch total", "SYN", top_k=2)
        if "synthetic-total" not in trace["selected_ids"] or not trace["anchors"] or not trace["converged"]:
            raise AssertionError("Published evidence did not reach retrieval")
        (directory / "retrieval-trace.json").write_text(json.dumps(trace, indent=2))
        from research_memory.viewer import render_dashboard
        (directory / "retrieval.html").write_text(render_dashboard(trace))
        report["checks"]["candidate_snapshot_anchors_ppr_retrieval"] = True
        report.update(snapshot_id=manifest["snapshot_id"], retrieval_backend=trace["backend"],
                      selected_ids=trace["selected_ids"], ppr_iterations=trace["iterations"])
        print("SYNTHETIC_STAGE retrieval-passed", flush=True)
        peer_schema = {"type": "object", "properties": {"memory_id": {"type": "string"}, "answer": {"type": "integer"}},
                       "required": ["memory_id", "answer"], "additionalProperties": False}
        peer = model_call(sessions[1], "You are a fresh session. From the following retrieved evidence, double the "
                          "previous synthetic batch total. Return only JSON memory_id (the evidence ID of the total) "
                          "and answer. Do not request file access.\n\n" + trace["brief"], peer_schema)
        if peer != {"memory_id": "synthetic-total", "answer": 24}:
            raise AssertionError("Fresh model session did not use the retrieved total correctly")
        (sessions[1] / "result.json").write_text(json.dumps(peer))
        report["checks"]["fresh_model_session_used_retrieved_evidence"] = True
        report.update(status="passed", limitations=["CPU diagnostic, not GPU validation",
            "Bounded structured tool request, not a general coding agent or native API tool-call test",
            "Two synthetic memories, not empirical research or memory-effect evaluation",
            "Model processes are sandboxed; trusted orchestration/indexing runs outside that sandbox"])
    except Exception as error:
        report["failure_type"] = type(error).__name__
        if isinstance(error, (AssertionError, ValueError, RuntimeError)):
            report["diagnostic"] = str(error)[:300]
        # Synthetic-only bounded diagnostics; never enable prompt logging for research-data runs.
        for session in sessions:
            if (session / "model.stdout").exists():
                report.setdefault("synthetic_stdout", {})[session.name] = (session / "model.stdout").read_text()[-1200:]
                with (session / "model.stderr").open("rb") as stream:
                    stream.seek(max(0, os.fstat(stream.fileno()).st_size - 1800))
                    report.setdefault("synthetic_stderr_tail", {})[session.name] = stream.read().decode(errors="replace")
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print("LOCAL_SYNTHETIC", json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
