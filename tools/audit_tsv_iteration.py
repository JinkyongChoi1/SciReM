"""Read-only, Scheduler-only audit of full-context TSV memory development."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType

from audit_memory_iteration import read, digest, frozen_module, OWNERS, EDGE_TYPES as LEGACY_EDGE_TYPES

EDGE_TYPES = {**LEGACY_EDGE_TYPES,
    ("AnalysisModel", "assumption_constraint"): "Hypothesis",
    ("ResultAssessment", "supports"): "Hypothesis",
    ("ResultAssessment", "challenges"): "Hypothesis"}
JUDGMENT_RELATIONS = {"supports", "challenges", "assumption_constraint"}

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "workspaces/tsv-memory-iterations-20261023"
EXPECTED_ACTORS = dict(OWNERS)
REQUIRED_SOURCE_BASENAMES = {"evidence.py", "tabular.py", "two_role.py", "two_role_discovery.py",
                             "private_inference.py", "discovery_analysis.py", "model_catalog.py"}
RETRIEVAL_ROLES = (("hypothesis", "hypothesis"), ("analysis", "analysis"))
EXPECTED_TOPOLOGY = "hypothesis-analysis-hypothesis"
REQUIRE_WORKFLOW = False
REQUIRE_SAMPLE_RECIPE = False


def audit_judgment_edges(node, nodes, trajectory, evidence):
    """Independently reconstruct semantic edge fields from original plan/results."""
    actual = [edge for edge in node["edges"] if edge["relation"] in JUDGMENT_RELATIONS]
    if node.get("judgment_edge_schema") is None and not actual:
        return  # Historical candidates are not retroactively assigned new semantics.
    if node.get("judgment_edge_schema") != "scirem-judgment-v1":
        raise ValueError("Unknown or missing judgment edge schema")
    kind = node["memory_type"]
    original, provenance = evidence._source(node["source"], trajectory)
    if provenance != node["provenance"]:
        raise ValueError("Judgment edge source changed")
    details = None
    if kind == "AnalysisModel":
        bases = [edge for edge in node["edges"] if edge["relation"] == "intended_test_for"]
        if len(bases) != 1 or original != node["payload"]:
            raise ValueError("Judgment model contract changed")
        target = bases[0]["target"]
        assumptions = (original.get("model_specification") or {}).get("identification_assumptions")
        if isinstance(assumptions, str) and assumptions.strip():
            details = {"relation": "assumption_constraint", "basis": assumptions,
                "assertion_status": "declared_not_validated",
                "scope": "Interpretation of this hypothesis through this model is conditional on the declared assumptions; their validity is unverified."}
    elif kind == "ResultAssessment":
        bases = [edge for edge in node["edges"] if edge["relation"] == "produced_by"]
        if len(bases) != 1 or nodes[bases[0]["target"]]["memory_type"] != "AnalysisModel":
            raise ValueError("Judgment result model endpoint changed")
        model = nodes[bases[0]["target"]]
        plan, source = evidence._source(model["source"], trajectory)
        if source != model["provenance"] or plan != model["payload"]:
            raise ValueError("Judgment result plan changed")
        targets = [edge["target"] for edge in model["edges"] if edge["relation"] == "intended_test_for"]
        if len(targets) != 1:
            raise ValueError("Judgment hypothesis contract changed")
        target = targets[0]
        check = evidence.statistical_assessment(plan, original)
        relation = None
        if check["support"] == "supported":
            relation = "supports"
        elif check["reason"] == "significant_effect_opposes_declared_direction":
            coefficient = Decimal(str(check["estimate"]["coefficient"]))
            direction = check["primary_test"]["expected_direction"]
            if (direction == "positive" and coefficient < 0) or (direction == "negative" and coefficient > 0):
                relation = "challenges"
        if relation:
            details = {"relation": relation, "assertion_status": "derived_primary_statistical_test",
                "scope": check["scope"], "basis": {key: check.get(key) for key in
                    ("primary_test", "estimate", "alpha", "reason")}}
    else:
        raise ValueError("Judgment relation originates from an unsupported node type")
    if target not in nodes or nodes[target]["memory_type"] != "Hypothesis":
        raise ValueError("Judgment edge requires its hypothesis endpoint")
    expected = ([{"target": target, "source": str(Path(node["source"]).resolve()),
        "provenance": provenance, "scientific_validity": "unverified", **details}] if details else [])
    if actual != expected:
        raise ValueError("Judgment edge meaning/provenance differs from source recomputation")


def node_order(identifier):
    match = re.fullmatch(r"(?:hypothesis|model|result|decision)-(\d+)", identifier)
    return int(match[1]) if match else None


def expected_context_nodes(nodes, attempt, role):
    expected = {key for key in nodes if node_order(key) is not None and
                (node_order(key) < attempt or (role == "analysis" and key == f"hypothesis-{attempt:04d}"))}
    if "research-question" in nodes:
        expected.add("research-question")
    return expected


def validate_runtime_contract(config, report, task):
    contract = report.get("runtime_contract")
    if any("runtime_contract" in artifact for artifact in (config, report, task)):
        if not contract or contract != config.get("runtime_contract") or contract != task.get("runtime_contract"):
            raise ValueError("Runtime/device/toolkit contract changed across the trajectory")
    if report.get("gpu") is True:
        if not contract or contract.get("device") != "gpu":
            raise ValueError("GPU execution lacks its pinned runtime contract")
        for key in ("runtime", "cuda_toolkit"):
            if not isinstance(contract.get(key), str) or not Path(contract[key]).is_absolute():
                raise ValueError("GPU runtime/toolkit path is not explicitly pinned")
        device = report.get("device_report", {})
        for key in ("device_count", "requested_gpu_count"):
            if type(device.get(key)) is not int or device[key] != 1:
                raise ValueError("GPU job/worker must use exactly one GPU: " + key)
        if type(device.get("cuda_offloaded_layers")) is not int or device["cuda_offloaded_layers"] <= 0:
            raise ValueError("GPU execution did not verify actual CUDA layer offload")
        if device.get("exclusive") is not False:
            raise ValueError("Shared-grid GPU execution must not claim exclusive ownership")
    elif contract and contract.get("device") != "cpu":
        raise ValueError("CPU/GPU report and runtime contract disagree")


def audit_trace(trace, nodes, trajectory, attempt, role, evidence, render,
                expected_backend=None, expected_strategy=None):
    """Verify what was delivered, not merely what a retrieval algorithm ranked."""
    if expected_backend is not None and trace.get("backend") != expected_backend:
        raise ValueError("Retrieval trace embedding backend differs from the frozen run")
    if expected_strategy is not None and trace.get("strategy") != expected_strategy:
        raise ValueError("Retrieval trace strategy differs from the frozen run")
    packet = trace["context_packet"]
    if packet.get("format") != "facts-tsv-v1" or trace["brief"] != render(packet):
        raise ValueError("TSV rendering differs from the recorded full packet")
    views = packet["evidence"]
    if trace["selected_ids"] != [view["evidence_id"] for view in views]:
        raise ValueError("TSV representative/context membership mismatch")
    expected_nodes = expected_context_nodes(nodes, attempt, role)
    scoped = packet.get("selection_policy") == "ranked_seeds_with_scientific_source_closure"
    if scoped:
        snapshot_ids = {node["memory_id"] for node in trace["nodes"]}
        seeds = trace.get("upstream_selected_ids", [])
        if (not snapshot_ids.issubset(expected_nodes) or
                len(seeds) != len(set(seeds)) or not set(seeds).issubset(snapshot_ids)):
            raise ValueError("Ranked context contains unknown/future/duplicate seeds")
        required, pending = set(), list(seeds)
        if "research-question" in snapshot_ids:
            pending.append("research-question")
        while pending:
            key = pending.pop()
            if key in required:
                continue
            if key not in snapshot_ids:
                raise ValueError("Required selected source is absent from snapshot")
            required.add(key)
            node = nodes[key]
            if node.get("result_id") in snapshot_ids:
                pending.append(node["result_id"])
            pending.extend(edge["target"] for edge in node.get("edges", []) if edge["relation"] in
                           {"produced_by", "intended_test_for", "addresses", "based_on"} | JUDGMENT_RELATIONS)
        expected_nodes = required
        exclusions = [{"memory_id": key, "reason": "outside_ranked_source_closure"}
                      for key in sorted(snapshot_ids - required)]
        if trace.get("context_exclusions") != exclusions:
            raise ValueError("Explicit retrieval exclusions differ from selected scope")
    delivered = {node["memory_id"]: node for node in packet["graph_nodes"]}
    if len(delivered) != len(packet["graph_nodes"]) or set(delivered) != expected_nodes:
        raise ValueError("Published own scientific nodes silently omitted or future nodes delivered")
    for key, node in delivered.items():
        expected = dict(nodes[key])
        audit_judgment_edges(expected, nodes, trajectory, evidence)
        if node["memory_type"] == "AnalysisModel":
            samples = [candidate.get("sample_identity") for candidate in nodes.values()
                if candidate["memory_type"] == "ResultAssessment" and node_order(candidate["memory_id"]) < attempt
                and any(edge["relation"] == "produced_by" and edge["target"] == key for edge in candidate["edges"])]
            samples = sorted(set(samples), key=str)
            expected["sample_identity"] = samples[0] if len(samples) == 1 else samples if samples else None
        if node != expected:
            changed = sorted(key for key in set(node) | set(expected) if node.get(key) != expected.get(key))
            raise ValueError("Delivered scientific node differs from immutable candidate: " + key + ":" + ",".join(changed))
    groups = {}
    for key in expected_nodes:
        node = nodes[key]
        if node["memory_type"] == "ResultAssessment":
            groups.setdefault(node.get("evidence_group") or key, set()).add(key)
    observed_groups = set()
    support = Counter()
    for view in views:
        key = view["evidence_id"]
        node = nodes[key]
        record = node["evidence"]
        evidence.verify_record(record, trajectory)
        group = node.get("evidence_group") or key
        if group in observed_groups or set(view["observed_in"]) != groups.get(group):
            raise ValueError("Equivalent observations omitted, duplicated, or conflated")
        observed_groups.add(group)
        tool, source = evidence._source(Path(trajectory) / record["result"]["source"]["path"], trajectory)
        if (view["source"] != source or view["tool"] != tool or
            view["analysis"] != tool["effective_action"] or
            view["estimates"] != tool.get("estimates", []) or view["metrics"] != tool.get("metrics", []) or
            view.get("omitted_estimate_terms") != []):
            raise ValueError("TSV loses or changes executed evidence fields")
        check = record["assessment"]["statistical_check"]
        if view["full_statistical_check"] != check or view["statistical_check"] != evidence.statistical_summary(check):
            raise ValueError("TSV changes machine-derived significance/direction/support")
        if (view["data_identity"] != node["data_identity"] or view["sample_identity"] != node["sample_identity"]
                or view.get("sample_recipe") != node.get("sample_recipe")):
            raise ValueError("TSV sample/data identity differs from source")
        support[check["support"]] += 1
        support["significance_" + evidence.statistical_summary(check)["significance"]] += 1
        if (check.get("primary_test") or {}).get("effect_type") == "curvature":
            support["curvature"] += 1
    if observed_groups != set(groups):
        raise ValueError("A complete own-result group was silently omitted")
    manifest = {"source_node_ids": [node["memory_id"] for node in packet["graph_nodes"]],
        "executed_result_ids": sorted({key for view in views for key in view["observed_in"]}),
        "representative_ids": trace["selected_ids"],
        "coefficient_row_count": sum(len(view["estimates"]) for view in views),
        "metric_row_count": sum(len(view["metrics"]) for view in views),
        "omitted_evidence_ids": [], "omitted_coefficient_count": 0}
    if "rendered_chars" in trace.get("context_manifest", {}):
        manifest["rendered_chars"] = len(trace["brief"])
    if trace.get("context_manifest") != manifest or packet.get("omitted_evidence_ids") != []:
        raise ValueError("TSV manifest differs from actual complete context")
    for node in trace["nodes"]:
        if (node.get("included") != (node["memory_id"] in delivered) or
            node.get("numeric_representative_included") != (node["memory_id"] in trace["selected_ids"])):
            raise ValueError("Trace node/representative inclusion differs from actual context")
    return {"group_count": len(groups), "node_count": len(delivered), "delivered_support": dict(support)}


def audit_round(run, trajectory, round_number, run_job):
    run, trajectory = Path(run).resolve(), Path(trajectory).resolve()
    summary = {"round": round_number, "run_job": run_job, "status": "failed", "flags": [],
               "attempts": [], "scope": "engineering integrity and observed adaptation; scientific validity unscored",
               "qa_feedback_to_agents": False, "source_integrity": "unverified", "runtime_health": "unverified"}
    def flag(code, detail, severity="error"):
        summary["flags"].append({"code": code, "detail": str(detail)[:300], "severity": severity})
    package_name = "_frozen_tsv_audit"
    try:
        config, report, task = read(run / "config.json"), read(run / "report.json"), read(trajectory / "config.json")
        validate_runtime_contract(config, report, task)
        embedding_backend = config.get("embedding_backend", "fastembed")
        retrieval_strategy = config.get("retrieval_strategy", "ppr")
        if (embedding_backend not in ("fastembed", "lexical") or
                report.get("embedding_backend", "fastembed") != embedding_backend or
                task.get("embedding_backend", "fastembed") != embedding_backend or
                report.get("retrieval_strategy", "ppr") != retrieval_strategy or
                task.get("retrieval_strategy", "ppr") != retrieval_strategy):
            raise ValueError("Embedding backend or retrieval strategy changed across run and trajectory")
        for artifact in (config, report):
            if (artifact.get("job_id") != run_job or artifact.get("round") != round_number or
                artifact.get("arm") != "memory" or artifact.get("memory_format") != "tsv" or
                artifact.get("trajectory_root") != str(trajectory)):
                raise ValueError("Run/round/context identity mismatch")
            for key in ("question", "data_identity", "model_identity"):
                if not task.get(key) or artifact.get(key) != task[key]:
                    raise ValueError("Pinned cross-round identity differs: " + key)
        hashes = config["source_sha256"]
        if hashes != report["source_sha256"] or not hashes:
            raise ValueError("Frozen source declaration mismatch")
        if not REQUIRED_SOURCE_BASENAMES.issubset({Path(key).name for key in hashes}):
            raise ValueError("Missing reproducibility-critical frozen source")
        if report.get("topology") != EXPECTED_TOPOLOGY:
            raise ValueError("Unexpected research-loop topology")
        if any(artifact.get("server_binary_sha256") != task["model_identity"].get("server_binary_sha256") for artifact in (config, report)):
            raise ValueError("Server binary differs from pinned model identity")
        for relative, expected in hashes.items():
            if digest(run / "source" / Path(relative).name) != expected:
                raise ValueError("Frozen source checksum mismatch: " + relative)
        package = ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        evidence = frozen_module(run / "source/evidence.py", package_name + ".evidence")
        sys.modules[package_name + ".evidence"] = evidence
        tabular = frozen_module(run / "source/tabular.py", package_name + ".tabular")
        analysis = frozen_module(run / "source/discovery_analysis.py", package_name + ".analysis")
        states = list(report.get("attempts", []))
        if report.get("active_attempt") and report["active_attempt"]["attempt"] not in {s["attempt"] for s in states}:
            states.append({**report["active_attempt"], "status": "censored"})
        if not states or len({s["attempt"] for s in states}) != len(states):
            raise ValueError("Missing or duplicate attempt receipts")
        last = max(s["attempt"] for s in states)
        nodes = {}
        candidate_paths = list((trajectory / "shared-memory/candidates").glob("*/*.json"))
        for path in candidate_paths:
            node = read(path)
            key = node["memory_id"]
            if node_order(key) is not None and node_order(key) > last:
                raise ValueError("Graph candidate is outside the recorded attempt window")
            if key in nodes:
                raise ValueError("Duplicate graph node identity")
            nodes[key] = node
        for key, node in nodes.items():
            original, source = evidence._source(node["source"], trajectory)
            if node["actor"] != EXPECTED_ACTORS[node["memory_type"]] or node["provenance"] != source:
                raise ValueError("Graph source or role ownership mismatch: " + key)
            if node["memory_type"] in ("Hypothesis", "AnalysisModel", "Decision") and node["payload"] != original:
                raise ValueError("Agent payload changed after publication")
            for edge in node["edges"]:
                if (edge["target"] not in nodes or nodes[edge["target"]]["memory_type"] !=
                    EDGE_TYPES.get((node["memory_type"], edge["relation"])) or edge["provenance"] != source or
                    edge["source"] != node["source"]):
                    raise ValueError("Graph edge type/provenance mismatch")
            audit_judgment_edges(node, nodes, trajectory, evidence)
            if node["memory_type"] == "AnalysisModel":
                hypothesis_id = next(edge["target"] for edge in node["edges"] if edge["relation"] == "intended_test_for")
                for field in ("hypothesis", "expected_pattern", "focal_test"):
                    if node["payload"][field] != nodes[hypothesis_id]["payload"][field]:
                        raise ValueError("Analysis agent changed the hypothesis contract")
            if node["memory_type"] == "ResultAssessment":
                evidence.verify_record(node["evidence"], trajectory)
                record = node["evidence"]
                if node.get("data_identity") != task["data_identity"]:
                    raise ValueError("Result dataset identity differs from pinned trajectory")
                md5 = next((row.get("sample_membership_md5") for row in record["result"]["metrics"]
                            if row.get("sample_membership_md5")), None)
                expected_sample = "md5:" + md5 if isinstance(md5, str) and re.fullmatch(r"[a-fA-F0-9]{32}", md5) else None
                if node.get("sample_identity") != expected_sample:
                    flag("sample_hash_drift", identifier + ": optional sample hash differs from runner output", "warning")
                recipe = node.get("sample_recipe")
                metric = next((row for row in record["result"]["metrics"] if row.get("n") is not None), {})
                action = record["result"]["analysis"]
                if REQUIRE_SAMPLE_RECIPE and (not isinstance(recipe, dict) or recipe.get("data_sha256") != task["data_identity"].get("sha256") or \
                        recipe.get("filter") != action.get("filter") or recipe.get("n_obs") != metric.get("n")):
                    raise ValueError("Human-readable sample recipe differs from the executed analysis")
                group = None
                if node.get("data_identity") and node.get("sample_identity"):
                    payload = {"data": node["data_identity"], "sample": node["sample_identity"],
                        "action": evidence.analysis_key(record["result"]["analysis"]),
                        "estimates": record["result"]["estimates"], "metrics": record["result"]["metrics"],
                        "returncode": record["result"]["returncode"],
                        "primary_check": record["assessment"]["statistical_check"]}
                    group = hashlib.sha256(evidence.encode(payload).encode()).hexdigest()
                if group != node.get("evidence_group"):
                    raise ValueError("Result grouping differs from complete scientific identity")
        snapshot_pointer = trajectory / "shared-memory/CURRENT"
        if snapshot_pointer.is_file():
            snapshot_id = snapshot_pointer.read_text().strip()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_id):
                raise ValueError("Invalid snapshot identity")
            snapshot = trajectory / "shared-memory/snapshots" / snapshot_id
            manifest = read(snapshot / "manifest.json")
            if manifest.get("backend") != embedding_backend:
                raise ValueError("Published snapshot backend differs from the frozen run")
            for filename, expected in manifest["files"].items():
                target = (snapshot / filename).resolve()
                if not target.is_relative_to(snapshot.resolve()) or digest(target) != expected:
                    raise ValueError("Published graph/vector snapshot checksum mismatch")
            published = {node["memory_id"]: node for node in read(snapshot / "graph.json")["nodes"]}
            if any(published.get(key) != node for key, node in nodes.items()):
                raise ValueError("Candidate not present in published consistent graph snapshot")
        elif any(state.get("status") == "completed" for state in states):
            raise ValueError("Graph candidates or completed attempts exist without a published snapshot")
        elif candidate_paths:
            flag("graph_snapshot_missing", "Candidates are source-verified but no consistent snapshot was published; they are not retrievable", "warning")
        else:
            flag("no_memory_published", "No attempt reached graph publication before failure", "warning")
        models, terms = [], set()
        for state in states:
            index = state["attempt"]
            directory = trajectory / f"attempt-{index:04d}"
            item = {key: state.get(key) for key in ("attempt", "status", "analysis_status", "decision_status", "failure_type")}
            if state.get("status") in ("failed", "censored"):
                flag("incomplete_attempt", f"{index}: {state.get('failure_type')}", "warning")
            for artifact, role in RETRIEVAL_ROLES:
                path = directory / (artifact + "-retrieval.json")
                if path.exists():
                    item[role + "_delivery"] = audit_trace(read(path), nodes, trajectory, index, role, evidence,
                        tabular.render_tsv, expected_backend=embedding_backend,
                        expected_strategy=retrieval_strategy)
                elif state.get("status") == "completed":
                    raise ValueError("Completed agent cycle lacks its actual TSV context receipt")
            if REQUIRE_WORKFLOW and state.get("status") == "completed":
                workflow = read(directory / "orchestrator.json")
                stages = [event.get("stage") for event in workflow.get("events", [])]
                expected = ["hypothesis", "empiricist_plan", "runner", "empiricist_assessment", "decision"]
                if workflow.get("status") != "completed" or stages != expected:
                    raise ValueError("Four-role orchestration receipt is incomplete")
                history = directory.parent / "role-history"
                if not all((history / (role + ".jsonl")).is_file() for role in ("orchestrator", "hypothesis", "empiricist")):
                    raise ValueError("Four-role role-local history is missing")
                response = read(directory / "hypothesis-response.json")
                validation = read(directory / "historical-validation.json")
                published_hypothesis = read(directory / "hypothesis.json")
                verified_claims = [item["claim"] for item in validation if item.get("status") == "verified"]
                if (len(validation) != len(response.get("historical_claims", [])) or
                        published_hypothesis.get("historical_claims") != verified_claims):
                    raise ValueError("Unverified historical claims entered the published hypothesis")
            item["decision"] = None
            item["sources"] = {}
            if (directory / "plan.json").exists():
                plan = read(directory / "plan.json")
                item.update(hypothesis=plan["hypothesis"], focal_test=plan["focal_test"], requested_model=plan["analysis"])
                item["sources"]["plan"] = evidence._source(directory / "plan.json", trajectory)[1]
            elif f"hypothesis-{index:04d}" in nodes:
                hypothesis = nodes[f"hypothesis-{index:04d}"]
                item.update(hypothesis=hypothesis["payload"]["hypothesis"], focal_test=hypothesis["payload"]["focal_test"])
                item["sources"]["hypothesis"] = hypothesis["provenance"]
            if (directory / "tool-result.json").exists():
                plan, tool = read(directory / "plan.json"), read(directory / "tool-result.json")
                if analysis.validate_action(plan["analysis"]) != tool["effective_action"]:
                    raise ValueError("Executed specification differs from the requested analysis")
                check = evidence.statistical_assessment(plan, tool)
                if check != read(directory / "statistical-assessment.json") or f"result-{index:04d}" not in nodes:
                    raise ValueError("Executed result or primary coefficient verdict was not preserved")
                item["primary"] = evidence.statistical_summary(check)
                item.update(effective_model=tool["effective_action"], primary_estimate=check.get("estimate"),
                    N=next((row.get("n") for row in tool.get("metrics", []) if row.get("n") is not None), None),
                    analysis_key=evidence.analysis_key(tool["effective_action"]))
                item["sources"]["tool_result"] = evidence._source(directory / "tool-result.json", trajectory)[1]
                models.append(evidence.analysis_key(tool["effective_action"]))
                terms.add(plan["focal_test"]["term"])
            if state.get("decision_status") == "succeeded":
                decision = read(directory / "decision.json")
                if state["decision_id"] not in nodes or state["result_id"] not in decision["evidence_ids"]:
                    raise ValueError("Decision does not cite the current machine result")
                item["decision"] = decision
                item["sources"]["decision"] = evidence._source(directory / "decision.json", trajectory)[1]
            summary["attempts"].append(item)
        cleanup = report.get("runtime_cleanup_verified") is True
        if any(state.get("status") == "completed" for state in states) and not report.get("model_calls"):
            raise ValueError("Completed agent cycle lacks inference/capacity metadata")
        for call in report.get("model_calls", []):
            if call.get("failure_type") in {"RuntimeIsolationError", "AllocationDeadlineExceeded"}:
                if call.get("failure_boundary") not in {"runtime_startup", "preflight", "cache_restore", "render",
                        "prompt_render", "tokenize_and_capacity", "completion_request", "dynamic_cache_erase"} or \
                        not isinstance(call.get("failure_cause"), str) or type(call.get("deadline_exhausted")) is not bool:
                    raise ValueError("Runtime isolation failure lacks a bounded, prompt-free cause receipt")
                if call.get("deadline_exhausted") and (report.get("active_wall_seconds", 0) <
                        report.get("budget_seconds", 0) - report.get("inference_deadline_reserve_seconds", 45)):
                    raise ValueError("A censored attempt is not near the reviewed inference deadline")
            if call.get("parse_status") == "failed" and not call.get("completion_submitted"):
                continue
            required = call["rendered_prompt_tokens"] + call["requested_generation_tokens"] + call["safety_margin_tokens"]
            if (call["capacity_scope"] != "actual_rendered_chat_prompt" or call["required_tokens"] != required
                or not 0 < call["context_limit"] <= 32768 or call.get("context_shift_enabled") is not False):
                raise ValueError("Missing native rendered-prompt capacity verification")
            if call.get("capacity_status") == "fits":
                if required > call["context_limit"] or call.get("truncated") is True or call.get("completion_submitted") is not True:
                    raise ValueError("Inference capacity/anti-truncation metadata mismatch")
            elif call.get("capacity_status") == "exceeded":
                if required <= call["context_limit"] or call.get("completion_submitted") is not False:
                    raise ValueError("False capacity-exceeded explanation")
            elif call.get("capacity_status") == "server_reported_truncation":
                if call.get("truncated") is not True:
                    raise ValueError("Unverified server truncation explanation")
            else:
                raise ValueError("Missing actual native-tokenizer capacity verification")
        # Source/context proof remains valid even when later native cleanup fails.
        summary.update(source_integrity="passed", report_sha256=digest(run / "report.json"),
            run_config_sha256=digest(run / "config.json"), trajectory_config_sha256=digest(trajectory / "config.json"))
        repeats = len(models) - len(set(models))
        if repeats:
            flag("repeated_effective_model", f"{repeats} repeated fits; not evidence of new scientific learning", "warning")
        deliveries = [item["hypothesis_delivery"]["group_count"] for item in summary["attempts"] if "hypothesis_delivery" in item]
        summary.update(distinct_model_count=len(set(models)), repeated_model_count=repeats,
            completed_attempt_count=sum(state.get("status") == "completed" for state in states),
            failed_attempt_count=sum(state.get("status") == "failed" for state in states),
            censored_attempt_count=sum(state.get("status") == "censored" for state in states),
            actual_fit_count=sum(state.get("analysis_status") == "succeeded" for state in states),
            max_rendered_prompt_tokens=max((call.get("rendered_prompt_tokens", 0) for call in report.get("model_calls", [])), default=0),
            delivered_group_min=min(deliveries, default=None), delivered_group_max=max(deliveries, default=None),
            primary_term_coverage=sorted(terms), model_calls=report.get("model_calls", []),
            device_report=report.get("device_report"), runtime_contract=report.get("runtime_contract"),
            runtime_cleanup_verified=report.get("runtime_cleanup_verified"))
        deadline = str(report.get("stop_reason", "")).lower()
        if not cleanup:
            summary["runtime_health"] = "failed"
            raise ValueError("Owned runtime cleanup was not verified")
        if report.get("status") == "failed" and not any(word in deadline for word in ("budget", "deadline", "capacity")):
            summary["runtime_health"] = "failed"
            raise ValueError("Unexplained runtime/transport/cache failure: " + str(report.get("failure_type")))
        incomplete = any(s.get("status") in ("failed", "censored") for s in states)
        no_successful_fits = summary["actual_fit_count"] == 0
        if incomplete and no_successful_fits:
            summary.update(runtime_health="failed", status="failed")
        elif incomplete or report.get("status") != "window-ended":
            summary.update(runtime_health="partial", status="partial")
        else:
            summary.update(runtime_health="passed", status="passed")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        flag("integrity_or_runtime_failure", error)
    finally:
        sys.modules.pop(package_name + ".tabular", None)
        sys.modules.pop(package_name + ".evidence", None)
        sys.modules.pop(package_name, None)
    return summary


def main():
    if not os.environ.get("JOB_ID"):
        raise SystemExit("Reviewed SGE allocation required")
    os.umask(0o077)
    for fd in (1, 2):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.fchmod(fd, 0o600)
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=range(1, 4), required=True)
    parser.add_argument("--run-job", required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    args = parser.parse_args()
    trajectory, allowed = args.trajectory_root.resolve(), (BASE / "trajectories").resolve()
    if not args.run_job.isdigit() or not args.trajectory_root.is_absolute() or trajectory == allowed or not trajectory.is_relative_to(allowed):
        raise SystemExit("Exact run identity and reviewed new-series trajectory required")
    summary = audit_round(BASE / f"round-{args.round:02d}" / args.run_job, trajectory, args.round, args.run_job)
    output = BASE / "evaluator-qa" / f"round-{args.round:02d}-{args.run_job}" / os.environ["JOB_ID"]
    output.mkdir(parents=True, mode=0o700)
    (output / "report.json").write_text(json.dumps(summary, indent=2))
    print("TSV_ITERATION_QA", json.dumps({k: summary.get(k) for k in
        ("round", "run_job", "status", "source_integrity", "runtime_health", "completed_attempt_count", "failed_attempt_count",
         "censored_attempt_count", "actual_fit_count", "distinct_model_count", "repeated_model_count",
         "primary_term_coverage", "max_rendered_prompt_tokens", "delivered_group_min", "delivered_group_max")}), flush=True)
    print("TSV_ITERATION_QA_FLAGS", json.dumps({"flags": summary["flags"][:20],
        "omitted_flag_count": max(0, len(summary["flags"]) - 20)}), flush=True)
    print("TSV_ITERATION_QA_WRITTEN", str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
