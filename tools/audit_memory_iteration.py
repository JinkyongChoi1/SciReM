"""Read-only evaluator QA for a finite memory-only research development round."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "workspaces/memory-iterations-20261022"
OWNERS = {"ResearchQuestion": "user", "Hypothesis": "hypothesis-agent",
          "AnalysisModel": "analysis-agent", "ResultAssessment": "analysis-agent", "Decision": "hypothesis-agent"}
EDGE_TYPES = {("Hypothesis", "addresses"): "ResearchQuestion",
    ("Hypothesis", "revision_of"): "Hypothesis", ("Hypothesis", "follows_decision"): "Decision",
    ("AnalysisModel", "intended_test_for"): "Hypothesis",
    ("ResultAssessment", "produced_by"): "AnalysisModel", ("Decision", "based_on"): "ResultAssessment"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def audit_round(run, trajectory, round_number, run_job):
    """Read artifacts only; return evaluator findings, never agent instructions."""
    run, trajectory = Path(run).resolve(), Path(trajectory).resolve()
    flags, items, nodes = [], [], {}
    summary = {"round": round_number, "run_job": run_job, "trajectory_root": str(trajectory),
        "scope": "engineering integrity and observed behavior; not scientific discovery validity",
        "status": "failed", "attempts": items, "flags": flags, "flag_count": 0, "error_count": 0,
        "qa_feedback_to_agents": False}

    def flag(code, detail="", attempt=None, severity="error"):
        summary["flag_count"] += 1
        summary["error_count"] += severity == "error"
        if len(flags) < 200:
            flags.append({"code": code, "detail": str(detail)[:240], "attempt": attempt, "severity": severity})

    def check(condition, code, attempt=None, detail="", severity="error"):
        if not condition:
            flag(code, detail, attempt, severity)

    try:
        config, report, task = read(run / "config.json"), read(run / "report.json"), read(trajectory / "config.json")
        summary.update(run_config_sha256=digest(run / "config.json"),
                       run_report_sha256=digest(run / "report.json"),
                       trajectory_config_sha256=digest(trajectory / "config.json"),
                       runtime_status=report.get("status"), elapsed_seconds=report.get("elapsed_seconds"))
        for artifact in (config, report):
            check(artifact.get("job_id") == run_job and artifact.get("round") == round_number,
                  "round_identity_mismatch")
            check(artifact.get("arm") == "memory" and artifact.get("trajectory_root") == str(trajectory),
                  "trajectory_identity_mismatch")
            check(artifact.get("question") == task.get("question") and artifact.get("data_identity") == task.get("data_identity"),
                  "task_or_data_changed")
            check(bool(task.get("model_identity")) and artifact.get("model_identity") == task["model_identity"],
                  "cross_round_model_identity_changed")
        hashes = config.get("source_sha256", {})
        check(hashes == report.get("source_sha256"), "config_report_source_mismatch")
        check(config.get("server_binary_sha256") == report.get("server_binary_sha256"), "model_runtime_identity_changed")
        check(bool(hashes), "missing_frozen_sources")
        for relative, expected in hashes.items():
            check(digest(run / "source" / Path(relative).name) == expected, "frozen_source_hash_mismatch", detail=relative)
        # Never execute a source snapshot whose declared checksum failed validation.
        if summary["error_count"]:
            return summary
        evidence = frozen_module(run / "source/evidence.py", "iteration_frozen_evidence")
        analysis = frozen_module(run / "source/discovery_analysis.py", "iteration_frozen_analysis")
        states = list(report.get("attempts", []))
        check(len({s["attempt"] for s in states}) == len(states), "duplicate_attempt_identity")
        active = report.get("active_attempt")
        if active and not any(s["attempt"] == active.get("attempt") for s in states):
            states.append({**active, "status": "censored", "round": round_number, "job_id": run_job})
        check(bool(states), "no_attempt_evidence")
        last_attempt = max((s["attempt"] for s in states), default=0)
        for path in sorted((trajectory / "shared-memory/candidates").glob("*/*.json")):
            node = read(path)
            match = re.fullmatch(r"(?:hypothesis|model|result|decision)-(\d+)", node.get("memory_id", ""))
            if match and int(match[1]) > last_attempt:
                continue
            identifier = node.get("memory_id")
            check(identifier not in nodes, "duplicate_node_id", detail=identifier)
            nodes[identifier] = node
        snapshot_id = (trajectory / "shared-memory/CURRENT").read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_id):
            raise ValueError("Invalid snapshot identifier")
        snapshot = trajectory / "shared-memory/snapshots" / snapshot_id
        manifest = read(snapshot / "manifest.json")
        check(manifest.get("snapshot_id") == snapshot_id, "snapshot_identity_mismatch")
        for name, expected in manifest["files"].items():
            target = (snapshot / name).resolve()
            check(target.is_relative_to(snapshot.resolve()) and digest(target) == expected,
                  "snapshot_file_hash_mismatch", detail=name)
        published = {n["memory_id"]: n for n in read(snapshot / "graph.json")["nodes"]}
        check(all(published.get(identifier) == node for identifier, node in nodes.items()),
              "candidate_missing_or_changed_in_published_snapshot")
        summary["snapshot_id"] = snapshot_id
        for identifier, node in nodes.items():
            try:
                kind = node["memory_type"]
                check(node["actor"] == OWNERS[kind], "wrong_role_owner", detail=identifier)
                original, provenance = evidence._source(node["source"], trajectory)
                check(provenance == node["provenance"], "node_source_changed", detail=identifier)
                if kind in ("Hypothesis", "AnalysisModel", "Decision"):
                    check(original == node["payload"], "node_payload_changed", detail=identifier)
                for edge in node["edges"]:
                    target = nodes.get(edge["target"])
                    check(target is not None and target["memory_type"] == EDGE_TYPES.get((kind, edge["relation"])),
                          "invalid_typed_edge", detail=identifier)
                    check(edge["source"] == node["source"] and edge["provenance"] == provenance,
                          "edge_provenance_mismatch", detail=identifier)
                if kind == "ResultAssessment":
                    evidence.verify_record(node["evidence"], trajectory)
                    check(node.get("data_identity") == task["data_identity"], "result_data_identity_changed", detail=identifier)
                    metrics = node["evidence"]["result"]["metrics"]
                    md5 = next((m.get("sample_membership_md5") for m in metrics if m.get("sample_membership_md5")), None)
                    expected_sample = "md5:" + md5 if isinstance(md5, str) and re.fullmatch(r"[a-fA-F0-9]{32}", md5) else None
                    check(node.get("sample_identity") == expected_sample, "sample_membership_identity_mismatch", detail=identifier)
            except (OSError, ValueError, KeyError, TypeError) as error:
                flag("invalid_node_source", identifier + ": " + str(error))

        for state in states:
            index = state["attempt"]
            directory = trajectory / f"attempt-{index:04d}"
            item = {key: state.get(key) for key in ("attempt", "status", "analysis_status", "decision_status", "seconds", "failure_type", "failure_domain", "diagnostic")}
            item.update(retrieved_results=0, hypothesis_changed=None, model_changed=None)
            items.append(item)
            try:
                check(state.get("round") == round_number and state.get("job_id") == run_job,
                      "attempt_round_identity_mismatch", index)
                if state.get("status") == "censored":
                    flag("censored_attempt", "Partial artifacts preserved; no success imputed", index, "warning")
                if state.get("status") == "failed":
                    flag("failed_attempt", {"domain": state.get("failure_domain"),
                         "type": state.get("failure_type"), "diagnostic": state.get("diagnostic")}, index, "warning")
                plan_path, tool_path = directory / "plan.json", directory / "tool-result.json"
                if plan_path.exists():
                    plan = read(plan_path)
                    item.update(hypothesis=plan.get("hypothesis"), focal_test=plan.get("focal_test"),
                                model=plan.get("analysis"))
                    evidence.validate_focal_test(plan)
                    h = nodes[state.get("hypothesis_id", f"hypothesis-{index:04d}")]
                    for field in ("hypothesis", "expected_pattern", "focal_test"):
                        check(h["payload"].get(field) == plan.get(field), "analysis_changed_hypothesis_contract", index, field)
                    previous_paths = sorted(p for p in trajectory.glob("attempt-*/plan.json")
                        if re.fullmatch(r"attempt-(\d+)", p.parent.name) and int(p.parent.name.split("-")[1]) < index)
                    if previous_paths:
                        previous = read(previous_paths[-1])
                        item["hypothesis_changed"] = plan["hypothesis"] != previous["hypothesis"]
                        item["model_changed"] = evidence.analysis_key(plan["analysis"]) != evidence.analysis_key(previous["analysis"])
                        item["primary_contract_changed"] = plan["focal_test"] != previous["focal_test"]
                        declared = plan.get("adaptation", {}).get("decision")
                        item["declared_adaptation"] = declared
                        if declared == "revise" and not any(item[k] for k in ("hypothesis_changed", "model_changed", "primary_contract_changed")):
                            flag("declared_revision_without_observed_change", "Not scientific improvement", index, "warning")
                    previous_decisions = sorted((n for n in nodes.values() if n["memory_type"] == "Decision"
                        and int(n["memory_id"].split("-")[1]) < index), key=lambda n: n["memory_id"])
                    if previous_decisions:
                        latest = previous_decisions[-1]["memory_id"]
                        check(any(e["relation"] == "follows_decision" and e["target"] == latest for e in h["edges"]),
                              "missing_decision_to_next_hypothesis_link", index, latest, "warning")
                if tool_path.exists():
                    tool = read(tool_path)
                    check(analysis.validate_action(plan["analysis"]) == tool["effective_action"], "effective_model_differs_from_request", index)
                    result_id = state.get("result_id", f"result-{index:04d}")
                    check(result_id in nodes, "successful_analysis_not_published", index)
                    machine = evidence.statistical_assessment(plan, tool)
                    check(machine == read(directory / "statistical-assessment.json"), "computed_primary_verdict_mismatch", index)
                    item["primary"] = evidence.statistical_summary(machine)
                    item["primary_estimate"] = machine.get("estimate")
                    item["machine_result_preserved"] = result_id in nodes
                for role in ("hypothesis", "analysis"):
                    path = directory / (role + "-retrieval.json")
                    if not path.exists():
                        continue
                    trace = read(path)
                    packet = trace["context_packet"]
                    views = packet["evidence"]
                    check(trace["brief"] == evidence.encode(packet) and len(trace["brief"]) <= 3200,
                          "context_trace_or_budget_mismatch", index, role)
                    check(trace["selected_ids"] == [v["evidence_id"] for v in views], "packet_membership_mismatch", index, role)
                    check(packet.get("result_facts_available") == bool(views) and packet.get("agent_prose_included") is False,
                          "availability_manifest_mismatch", index, role)
                    item["retrieved_results"] += len(views)
                    for view in views:
                        record = nodes[view["evidence_id"]]["evidence"]
                        check(evidence.attempt_order(record) < index, "future_or_current_result_in_history", index)
                        check(view["source"] == record["result"]["source"], "packet_source_mismatch", index)
                        check(all(row in record["result"]["estimates"] for row in view["estimates"]), "altered_numeric_packet_row", index)
                        check(Counter(r["term"] for r in record["result"]["estimates"]) ==
                              Counter(r["term"] for r in view["estimates"]) + Counter(view["omitted_estimate_terms"]),
                              "unreported_coefficient_omission", index)
                        check(view["statistical_check"] == evidence.statistical_summary(record["assessment"]["statistical_check"]),
                              "packet_primary_facts_mismatch", index)
                        if role == "analysis":
                            check("hypotheses" not in view and "decisions" not in view, "analysis_view_contains_other_role_payload", index)
                            for identifier in view.get("hypothesis_ids", []):
                                check(nodes[identifier]["memory_type"] == "Hypothesis", "invalid_hypothesis_reference", index)
                            for identifier in view.get("decision_ids", []):
                                check(nodes[identifier]["memory_type"] == "Decision", "invalid_decision_reference", index)
                        else:
                            check(bool(view.get("hypotheses")) and len(view.get("hypotheses", [])) <= 2
                                  and len(view.get("decisions", [])) <= 1, "unbounded_or_missing_hypothesis_view", index)
                            for delivered in view.get("hypotheses", []):
                                original = nodes[delivered["hypothesis_id"]]
                                check(delivered["hypothesis"] == original["payload"]["hypothesis"] and
                                      delivered["focal_test"] == original["payload"]["focal_test"] and
                                      delivered["source"] == original["provenance"], "altered_hypothesis_packet", index)
                            for delivered in view.get("decisions", []):
                                original = nodes[delivered["decision_id"]]
                                check(delivered == {"decision_id": original["memory_id"],
                                      "source": original["provenance"], **original["payload"]}, "altered_decision_packet", index)
                        for equivalent in view.get("observed_in", []):
                            other = nodes[equivalent]
                            representative = nodes[view["evidence_id"]]
                            check(other.get("evidence_group") == representative.get("evidence_group")
                                  and (equivalent == view["evidence_id"] or other.get("sample_identity") is not None),
                                  "invalid_duplicate_group", index)
                            check(other.get("data_identity") == representative.get("data_identity") and
                                  other.get("sample_identity") == representative.get("sample_identity") and
                                  evidence.analysis_key(other["evidence"]["result"]["analysis"]) == evidence.analysis_key(record["result"]["analysis"]) and
                                  other["evidence"]["result"]["estimates"] == record["result"]["estimates"] and
                                  other["evidence"]["result"]["metrics"] == record["result"]["metrics"] and
                                  other["evidence"]["assessment"]["statistical_check"] == record["assessment"]["statistical_check"],
                                  "conflated_duplicate_evidence", index)
                            check(evidence.attempt_order(other["evidence"]) < index, "future_equivalent_result_in_history", index)
                if state.get("decision_status") == "succeeded":
                    decision = read(directory / "decision.json")
                    check(state["decision_id"] in nodes and state["result_id"] in decision["evidence_ids"],
                          "decision_missing_machine_result", index)
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                flag("attempt_audit_incomplete", str(error), index)
        check(report.get("status") == "window-ended", "round_not_cleanly_ended", detail=report.get("status"))
        summary.update(node_count=len(nodes), edge_count=sum(len(n.get("edges", [])) for n in nodes.values()),
            node_type_counts=dict(Counter(n.get("memory_type") for n in nodes.values())),
            completed_attempts=sum(i.get("status") == "completed" for i in items),
            failed_attempts=sum(i.get("status") == "failed" for i in items),
            censored_attempts=sum(i.get("status") == "censored" for i in items),
            successful_analyses=sum(i.get("analysis_status") == "succeeded" for i in items),
            model_changes=sum(i.get("model_changed") is True for i in items),
            unchanged_models=sum(i.get("model_changed") is False for i in items),
            hypothesis_changes=sum(i.get("hypothesis_changed") is True for i in items),
            attempts_with_retrieval=sum(i["retrieved_results"] > 0 for i in items),
            unique_model_count=len({evidence.analysis_key(i["model"]) for i in items if i.get("model")}),
            unique_hypothesis_count=len({i["hypothesis"] for i in items if i.get("hypothesis")}),
            primary_term_coverage=sorted({i["focal_test"]["term"] for i in items if i.get("focal_test")}),
            model_call_count=len(report.get("model_calls", [])),
            parse_failure_count=sum(c.get("parse_status") == "failed" for c in report.get("model_calls", [])),
            runtime_startup_seconds=report.get("runtime_startup_seconds"),
            runtime_cleanup_verified=report.get("runtime_cleanup_verified"),
            observed_attempt_seconds=sum(i.get("seconds") or 0 for i in items),
            status="failed" if summary["error_count"] else "partial" if any(i.get("status") in ("censored", "failed") for i in items) else "passed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        flag("round_audit_incomplete", str(error))
    return summary


def main():
    if not os.environ.get("JOB_ID"):
        raise SystemExit("Reviewed SGE allocation required")
    os.umask(0o077)
    for fd in (1, 2):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.fchmod(fd, 0o600)
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--run-job", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    args = parser.parse_args()
    trajectory = args.trajectory_root.resolve()
    allowed = (BASE / "trajectories").resolve()
    if not args.run_job.isdigit() or not args.trajectory_root.is_absolute() or not trajectory.is_relative_to(allowed) or trajectory == allowed:
        raise SystemExit("Exact run job and reviewed trajectory path required")
    run = BASE / f"round-{args.round:02d}" / args.run_job
    summary = audit_round(run, trajectory, args.round, args.run_job)
    output = BASE / "evaluator-qa" / f"round-{args.round:02d}-{args.run_job}" / os.environ["JOB_ID"]
    output.mkdir(parents=True, mode=0o700)
    target = output / "report.json"
    target.write_text(json.dumps(summary, indent=2))
    print("MEMORY_ITERATION_QA", json.dumps({k: summary.get(k) for k in
        ("round", "run_job", "status", "completed_attempts", "failed_attempts", "censored_attempts",
         "successful_analyses", "model_changes", "unchanged_models", "unique_model_count",
         "hypothesis_changes", "primary_term_coverage", "attempts_with_retrieval",
         "parse_failure_count", "runtime_cleanup_verified", "error_count", "flag_count")}), flush=True)
    for item in summary.get("flags", [])[:20]:
        print("MEMORY_ITERATION_QA_FLAG", json.dumps(item), flush=True)
    print("MEMORY_ITERATION_QA_WRITTEN", str(target), flush=True)


if __name__ == "__main__":
    main()
